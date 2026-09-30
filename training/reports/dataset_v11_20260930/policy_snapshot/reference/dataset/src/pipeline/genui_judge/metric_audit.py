"""Fresh v5.4 rescoring for selected rows using native capture evidence."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping

from pipeline.genui_quality.aggregate_v5_4 import score_record_v5_4
from pipeline.genui_quality.config_v5_4 import load_v5_4_reward_config
from pipeline.genui_quality._v5_4 import render_artifact_quality_v5_4


_SELECTION_WORKER_CONFIG: Any = None


def _selection_score_worker(record: Mapping[str, Any]) -> dict[str, Any]:
    global _SELECTION_WORKER_CONFIG
    if _SELECTION_WORKER_CONFIG is None:
        _SELECTION_WORKER_CONFIG = load_v5_4_reward_config()
    intent = str(
        record.get("intent_bucket") or record.get("intent") or ""
    ) or None
    result = render_artifact_quality_v5_4(
        record.get("genui_json"),
        str(record.get("response_text") or ""),
        intent=intent,
        assets=record.get("assets"),
        render_ok=None,
        config=_SELECTION_WORKER_CONFIG,
    )
    return {
        "ui_id": record.get("ui_id"),
        "render_artifact_quality_v5_4": {
            "metric_version": result.metric_version,
            "metric_fingerprint": result.metric_fingerprint,
            "quality_0_100": float(result.quality_0_100),
            "artifact_quality_0_100": float(result.quality_0_100),
            "active_caps": result.active_caps,
            "dimensions": result.dimensions,
        },
        "active_caps": result.active_caps,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "sd_population": pstdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values) if values else 0.0,
        "p05": _quantile(values, 0.05),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "maximum": max(values) if values else 0.0,
    }


def _render_evidence_by_ui(
    benchmark_dir: Path,
) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(benchmark_dir / "native_capture_manifest.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["ui_id"]), []).append(row)
    output: dict[str, dict[str, Any]] = {}
    for ui_id, captures in grouped.items():
        candidate_failures = [
            row
            for row in captures
            if not bool(row.get("ok"))
            and row.get("failure_class") == "candidate"
        ]
        infrastructure_failures = [
            row
            for row in captures
            if not bool(row.get("ok"))
            and row.get("failure_class") == "infrastructure"
        ]
        if infrastructure_failures:
            # Infrastructure failures do not become candidate quality evidence.
            continue
        required = [
            row for row in captures if bool(row.get("required", True))
        ]
        ok = bool(required) and all(bool(row.get("ok")) for row in required)
        if candidate_failures:
            ok = False
        output[ui_id] = {
            "adapter": "android_native_judgeCapture",
            "attempted": True,
            "ok": ok,
            "native_render_ok": ok,
            "capture_count": len(captures),
            "required_capture_count": len(required),
            "candidate_failure_count": len(candidate_failures),
        }
    return output


def rescore_selected_v5_4(
    benchmark_dir: str | Path,
    *,
    require_complete_native_evidence: bool = True,
) -> dict[str, Any]:
    root = Path(benchmark_dir).resolve()
    output_path = root / "metric_v5_4_native_scores.jsonl"
    if output_path.exists():
        raise FileExistsError(f"immutable score artifact exists: {output_path}")
    records = _read_jsonl(root / "selected_genui.jsonl")
    contracts = {
        row["ui_id"]: row
        for row in _read_jsonl(root / "expected_contracts.jsonl")
    }
    selection = {
        row["ui_id"]: row
        for row in _read_jsonl(root / "selection_manifest.jsonl")
    }
    render_by_ui = _render_evidence_by_ui(root)
    if require_complete_native_evidence:
        missing = [
            str(record["ui_id"])
            for record in records
            if str(record["ui_id"]) not in render_by_ui
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} rows lack conclusive native evidence"
            )
    config = load_v5_4_reward_config()
    scored: list[dict[str, Any]] = []
    for record in records:
        ui_id = str(record["ui_id"])
        contract = contracts[ui_id]
        enriched = dict(record)
        enriched["expected_ui_contract_v5_4"] = contract[
            "expected_ui_contract"
        ]
        enriched["expected_ui_contract_v5_4_source"] = contract[
            "expected_ui_contract_source"
        ]
        result = score_record_v5_4(
            enriched,
            render_row=render_by_ui.get(ui_id),
            config=config,
        )
        breakdown = asdict(result)
        value = float(result.quality_0_100)
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise AssertionError(f"invalid v5.4 score for {ui_id}: {value}")
        scored.append(
            {
                "ui_id": ui_id,
                "query_id": record.get("query_id"),
                "response_id": record.get("response_id"),
                "intent_bucket": (
                    record.get("intent_bucket") or record.get("intent")
                ),
                "selection_stratum": selection[ui_id][
                    "selection_stratum"
                ],
                "split": selection[ui_id]["split"],
                "render_artifact_quality_v5_4": breakdown,
                "quality_0_100": value,
                "active_caps": breakdown.get("active_caps", []),
                "native_render_evidence": render_by_ui.get(ui_id),
            }
        )
    _atomic_write_jsonl(output_path, scored)
    values = [float(row["quality_0_100"]) for row in scored]
    cap_counts = Counter(
        str(cap.get("name") or "unknown")
        for row in scored
        for cap in row["active_caps"]
        if isinstance(cap, Mapping)
    )
    aggregate = {
        "schema_version": "genui_metric_v5_4_native_audit.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_version": "5.4.0",
        "score_count": len(scored),
        "distribution": _distribution(values),
        "cap_counts": dict(sorted(cap_counts.items())),
        "scores_path": str(output_path),
        "scores_sha256": _hash_file(output_path),
        "native_evidence_count": len(render_by_ui),
    }
    aggregate_path = root / "metric_v5_4_native_aggregate.json"
    aggregate_path.write_text(
        json.dumps(
            aggregate,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return aggregate


def precompute_selection_v5_4(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 8,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Stream a compact no-native v5.4 index used only for stress sampling."""

    source_dir = Path(source_dir).resolve()
    source_path = source_dir / "genui.jsonl"
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"immutable selection score index exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    source_hash = _hash_file(source_path)
    output_path = output_dir / "scores.jsonl"
    values: list[float] = []
    cap_counts: Counter[str] = Counter()
    metric_fingerprint: str | None = None
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")

    def score_batch(
        executor: ProcessPoolExecutor | None,
        records: list[dict[str, Any]],
    ) -> Iterable[dict[str, Any]]:
        if executor is None:
            return map(_selection_score_worker, records)
        return executor.map(
            _selection_score_worker,
            records,
            chunksize=max(1, min(16, len(records) // workers)),
        )

    executor = (
        ProcessPoolExecutor(max_workers=workers)
        if workers > 1
        else None
    )
    try:
        source_handle = source_path.open(
            "r", encoding="utf-8", errors="replace"
        )
        output_handle = output_path.open(
            "w", encoding="utf-8", newline="\n"
        )
        with source_handle, output_handle:
            pending: list[dict[str, Any]] = []
            source_line = 0
            for source_line, line in enumerate(source_handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{source_path}:{source_line} is not an object"
                    )
                pending.append(record)
                if len(pending) < batch_size:
                    continue
                for row in score_batch(executor, pending):
                    score = float(
                        row["render_artifact_quality_v5_4"][
                            "quality_0_100"
                        ]
                    )
                    if not math.isfinite(score):
                        raise AssertionError(
                            f"non-finite v5.4 score near line {source_line}"
                        )
                    metric_fingerprint = row[
                        "render_artifact_quality_v5_4"
                    ]["metric_fingerprint"]
                    values.append(score)
                    for cap in row["active_caps"]:
                        cap_counts[
                            str(cap.get("name") or "unknown")
                        ] += 1
                    output_handle.write(_canonical_json(row) + "\n")
                pending.clear()
                output_handle.flush()
            if pending:
                for row in score_batch(executor, pending):
                    score = float(
                        row["render_artifact_quality_v5_4"][
                            "quality_0_100"
                        ]
                    )
                    if not math.isfinite(score):
                        raise AssertionError(
                            "non-finite v5.4 score in final batch"
                        )
                    metric_fingerprint = row[
                        "render_artifact_quality_v5_4"
                    ]["metric_fingerprint"]
                    values.append(score)
                    for cap in row["active_caps"]:
                        cap_counts[
                            str(cap.get("name") or "unknown")
                        ] += 1
                    output_handle.write(_canonical_json(row) + "\n")
            output_handle.flush()
            os.fsync(output_handle.fileno())
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if _hash_file(source_path) != source_hash:
        raise RuntimeError("source genui.jsonl changed during selection scoring")
    aggregate = {
        "schema_version": "genui_judge_selection_metric_index.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_version": "5.4.0",
        "metric_fingerprint": metric_fingerprint,
        "native_render_evidence_available": False,
        "selection_use_only": True,
        "workers": workers,
        "batch_size": batch_size,
        "count": len(values),
        "distribution": _distribution(values),
        "cap_counts": dict(sorted(cap_counts.items())),
        "source_genui": str(source_path),
        "source_genui_sha256": source_hash,
        "scores_sha256": _hash_file(output_path),
    }
    (output_dir / "aggregates.json").write_text(
        json.dumps(
            aggregate,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return aggregate


__all__ = ["precompute_selection_v5_4", "rescore_selected_v5_4"]
