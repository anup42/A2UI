"""Compute exact, bounded Stage 3 reports without blocking GPU generation."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def schedule_aggregate(*, genui_path, responses_path, aggregates_path,
                       aggregate_weights, metric_kwargs, work_dir) -> dict:
    """Coalesce requests; workers serialize on one OS lock per output report.

    Requests are private, application-written pickle files so the worker receives
    the exact runtime reward configurations, including dataclass instances.
    """
    key = hashlib.sha256(str(Path(aggregates_path).resolve()).encode()).hexdigest()[:16]
    state_dir = Path(work_dir) / key
    state_dir.mkdir(parents=True, exist_ok=True)
    files = [Path(genui_path), Path(responses_path),
             Path(genui_path).parent / "render.jsonl",
             Path(genui_path).parent / "native_render_checks.jsonl"]
    request = {
        "revision": uuid.uuid4().hex,
        "genui_path": str(genui_path), "responses_path": str(responses_path),
        "aggregates_path": str(aggregates_path),
        "snapshot_bytes": {str(p): p.stat().st_size if p.exists() else 0 for p in files},
        "weights": aggregate_weights, "metric_kwargs": metric_kwargs,
    }
    _atomic_write(state_dir / "request.pkl", pickle.dumps(request))
    with (state_dir / "worker.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(state_dir)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    return {"revision": request["revision"], "pid": process.pid,
            "snapshot_bytes": request["snapshot_bytes"][str(genui_path)],
            "state_dir": str(state_dir)}


def _snapshot_rows(path: Path, size: int):
    if not size:
        return
    with path.open("rb") as handle:
        while handle.tell() < size:
            line = handle.readline(size - handle.tell())
            # Never include a concurrently appended partial record.
            if not line.endswith(b"\n"):
                break
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compute_snapshot(request: dict) -> dict:
    from pipeline.metrics import (
        aggregate_metrics,
        compute_media_score,
        compute_overall_score,
    )

    bounds = request["snapshot_bytes"]
    genui = Path(request["genui_path"])
    responses = Path(request["responses_path"])
    rows = list(_snapshot_rows(genui, bounds[str(genui)]))
    response_map = {
        row["response_id"]: row["response_text"]
        for row in _snapshot_rows(responses, bounds[str(responses)])
        if isinstance(row.get("response_id"), str) and isinstance(row.get("response_text"), str)
    }
    for row in rows:
        if not row.get("response_text") and row.get("response_id") in response_map:
            row["response_text"] = response_map[row["response_id"]]
    render_rows = {}
    for name in ("render.jsonl", "native_render_checks.jsonl"):
        path = genui.parent / name
        for row in _snapshot_rows(path, bounds[str(path)]):
            if isinstance(row.get("ui_id"), str) and row["ui_id"]:
                render_rows[row["ui_id"]] = row
    report = aggregate_metrics(rows, render_rows_by_ui_id=render_rows, **request["metric_kwargs"])
    if request["metric_kwargs"]["metric_version"] in {"legacy", "dual"}:
        legacy = compute_overall_score(report, request["weights"] or {})
        report.update(legacy_structural_richness_score=legacy, overall_score=legacy,
                      legacy_score_deprecation_date="2026-10-01")
    report.update(media_score=compute_media_score(report),
                  aggregated_scope="all_generated_samples",
                  aggregated_sample_count=len(rows), aggregated_input_bytes=bounds[str(genui)],
                  updated_at=datetime.now(timezone.utc).isoformat())
    return report


def wait_for_aggregate(*, genui_path, aggregates_path, work_dir) -> None:
    """At the final corpus target, require the exact final report to finish."""
    key = hashlib.sha256(str(Path(aggregates_path).resolve()).encode()).hexdigest()[:16]
    state_dir = Path(work_dir) / key
    with (state_dir / "worker.log").open("ab", buffering=0) as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(state_dir)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads(Path(aggregates_path).read_text())
    if report.get("aggregated_input_bytes") != Path(genui_path).stat().st_size:
        raise RuntimeError("Final aggregate does not cover the complete generated corpus")


def run_worker(state_dir: Path) -> None:
    # POSIX only; caller leaves synchronous aggregation enabled elsewhere.
    import fcntl

    with (state_dir / "worker.lock").open("a") as lock:
        # Waiting happens in the detached worker, never in Stage 3.
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            request = pickle.loads((state_dir / "request.pkl").read_bytes())
            status_path = state_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("revision") == request["revision"] and status.get("status") == "completed":
                return
            started = time.monotonic()
            print(f"Aggregate start revision={request['revision']} bytes={request['snapshot_bytes'][request['genui_path']]}", flush=True)
            status = {"revision": request["revision"], "pid": os.getpid(), "status": "running"}
            _atomic_write(status_path, json.dumps(status).encode())
            try:
                report = compute_snapshot(request)
                _atomic_write(Path(request["aggregates_path"]), json.dumps(report, indent=2).encode())
                status.update(status="completed", rows=report["aggregated_sample_count"],
                              snapshot_bytes=report["aggregated_input_bytes"],
                              elapsed_seconds=time.monotonic() - started)
                print(f"Aggregate completed {json.dumps(status)}", flush=True)
            except Exception as exc:
                status.update(status="failed", error=str(exc))
                print(f"Aggregate failed {json.dumps(status)}", flush=True)
                _atomic_write(status_path, json.dumps(status).encode())
                raise
            _atomic_write(status_path, json.dumps(status).encode())
            # A later cycle may have queued a newer snapshot while this ran.
            # Re-read under the lock; lock waiters cover the exit/submission race.


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run_worker(Path(sys.argv[1]))
