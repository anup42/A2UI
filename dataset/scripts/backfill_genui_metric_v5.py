#!/usr/bin/env python3
"""Create immutable GenUI metric v5 sidecars for one or more dataset runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping


DATASET_ROOT = Path(__file__).resolve().parents[1]
DATASET_SRC = DATASET_ROOT / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.genui_quality import (  # noqa: E402
    SourceContractCache,
    aggregate_v5_records,
    breakdown_to_mapping,
    load_default_reward_config,
    metric_fingerprint,
    resolve_expected_ui_contract,
    score_record,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def load_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        str(row[key]): row
        for row in iter_jsonl(path)
        if row.get(key) not in (None, "")
    }


def resolve_run_dir(value: str) -> Path:
    supplied = Path(value)
    if supplied.is_absolute() or supplied.exists():
        return supplied.resolve()
    return (DATASET_ROOT / "data" / "runs" / value).resolve()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate immutable GenUI metric v5 sidecars. Source genui.jsonl "
            "and aggregates.json files are never modified."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run IDs under dataset/data/runs, or run directories",
    )
    parser.add_argument("--limit", type=int, default=0, help="Score at most N total rows")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Required for multiple runs; default for one run is <run>/metric_v5_shadow",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [resolve_run_dir(value) for value in args.runs]
    for run_dir in run_dirs:
        if not (run_dir / "genui.jsonl").exists():
            raise FileNotFoundError(f"Missing run genui.jsonl: {run_dir / 'genui.jsonl'}")
    if len(run_dirs) > 1 and not args.output_dir:
        raise ValueError("--output-dir is required when rescoring multiple runs")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dirs[0] / "metric_v5_shadow"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Immutable v5 output already exists and is non-empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_default_reward_config()
    fingerprint = metric_fingerprint(config)
    contract_cache = SourceContractCache(output_dir / "contract_cache")
    started = time.perf_counter()
    sidecar_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        genui_path = run_dir / "genui.jsonl"
        source_files.append(
            {
                "run_dir": str(run_dir),
                "genui_path": str(genui_path),
                "genui_sha256": file_sha256(genui_path),
            }
        )
        responses = load_by_id(run_dir / "responses.jsonl", "response_id")
        native_checks = load_by_id(run_dir / "native_render_checks.jsonl", "ui_id")
        for genui in iter_jsonl(genui_path):
            if args.limit > 0 and len(sidecar_rows) >= args.limit:
                break
            response_id = str(genui.get("response_id") or "")
            response = responses.get(response_id, {})
            source_text = str(
                genui.get("response_text") or response.get("response_text") or ""
            )
            intent = str(
                genui.get("intent_bucket") or genui.get("intent") or ""
            ) or None
            assets = genui.get("assets") or response.get("assets")
            persisted = (
                genui.get("expected_ui_contract")
                or response.get("expected_ui_contract")
            )
            persisted_source = (
                genui.get("expected_ui_contract_source")
                or response.get("expected_ui_contract_source")
            )
            resolution = resolve_expected_ui_contract(
                source_text,
                intent=intent,
                assets=assets,
                persisted=persisted if isinstance(persisted, Mapping) else None,
                persisted_source=str(persisted_source) if persisted_source else None,
                cache=contract_cache,
            )
            candidate = genui.get("genui_raw_completion")
            if candidate is None:
                candidate = genui.get("genui_json", genui.get("a2ui_json"))
            score_input = {
                **genui,
                "response_text": source_text,
                "assets": assets,
                "genui_raw_completion": candidate,
                "expected_ui_contract": resolution.contract,
                "expected_ui_contract_source": resolution.source,
            }
            ui_id = str(genui.get("ui_id") or "")
            native_row = native_checks.get(ui_id)
            result = score_record(score_input, render_row=native_row, config=config)
            generation = (
                genui.get("gen") if isinstance(genui.get("gen"), Mapping) else {}
            )
            breakdown = breakdown_to_mapping(result)
            sidecar = {
                "source_run": str(run_dir),
                "ui_id": genui.get("ui_id"),
                "query_id": genui.get("query_id"),
                "response_id": genui.get("response_id"),
                "intent_bucket": intent,
                "model_checkpoint": generation.get("model"),
                "metric_name": result.metric_name,
                "metric_version": result.metric_version,
                "metric_fingerprint": result.metric_fingerprint,
                "identity": result.identity,
                "contract_version": resolution.contract.get("contract_version"),
                "contract_source": resolution.source,
                "contract_resolution_errors": list(resolution.errors),
                "renderer_check_result": (
                    native_row.get("renderer_check_result")
                    if isinstance(native_row, Mapping)
                    else {
                        "adapter": "android_native",
                        "attempted": False,
                        "ok": None,
                    }
                ),
                "genui_quality_v5": breakdown,
            }
            sidecar_rows.append(sidecar)
            aggregate_rows.append(
                {
                    **score_input,
                    "model_checkpoint": generation.get("model"),
                    "genui_quality_v5": breakdown,
                }
            )
        if args.limit > 0 and len(sidecar_rows) >= args.limit:
            break

    elapsed = time.perf_counter() - started
    summary = aggregate_v5_records(aggregate_rows, config=config)
    summary.update(
        {
            "source_runs": source_files,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "shadow_only": True,
            "source_files_modified": False,
            "immutable": True,
            "metric_fingerprint": fingerprint,
            "elapsed_seconds": elapsed,
            "samples_per_second": (
                len(sidecar_rows) / elapsed if elapsed > 0.0 else None
            ),
        }
    )
    atomic_write_jsonl(output_dir / "scores.jsonl", sidecar_rows)
    atomic_write_json(output_dir / "aggregates.json", summary)
    quality = summary.get("quality_0_100") or {}
    report = (
        "# GenUI metric v5 immutable sidecar audit\n\n"
        f"- Source runs: {len(run_dirs)}\n"
        f"- Samples: {len(sidecar_rows)}\n"
        f"- Metric fingerprint: `{fingerprint}`\n"
        f"- Mean score: {float(quality.get('mean', 0.0)):.4f}\n"
        f"- Median score: {float(quality.get('median', 0.0)):.4f}\n"
        f"- Runtime: {elapsed:.3f} seconds "
        f"({summary.get('samples_per_second') or 0.0:.2f} samples/s)\n"
        "- Calibration: **uncalibrated engineering score**; no weights were "
        "fit to these rows.\n"
        "- Native-render evidence: included only when an Android-native check "
        "manifest was present.\n"
        "- Historical inputs: unchanged; this directory is an immutable sidecar.\n"
    )
    (output_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "count": len(sidecar_rows),
                "metric_fingerprint": fingerprint,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
