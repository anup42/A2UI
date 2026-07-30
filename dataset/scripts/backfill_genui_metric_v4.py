#!/usr/bin/env python3
"""Shadow-score an existing dataset run without rewriting historical records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    aggregate_v4_records,
    breakdown_to_mapping,
    load_v4_reward_config,
    resolve_expected_ui_contract,
    score_record_v4,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
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
    return {
        str(row[key]): row
        for row in iter_jsonl(path)
        if row.get(key) not in (None, "")
    }


def load_native_checks(run_dir: Path) -> dict[str, dict[str, Any]]:
    return load_by_id(run_dir / "native_render_checks.jsonl", "ui_id")


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate immutable GenUI metric v4 shadow outputs for a historical run. "
            "The source genui.jsonl and aggregates.json are never modified."
        )
    )
    parser.add_argument(
        "run",
        help="Run ID under dataset/data/runs, or an absolute/relative run directory",
    )
    parser.add_argument("--limit", type=int, default=0, help="Score at most N rows (0 = all)")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Default: <run>/metric_v4_shadow",
    )
    return parser.parse_args()


def resolve_run_dir(value: str) -> Path:
    supplied = Path(value)
    if supplied.is_absolute() or supplied.exists():
        return supplied.resolve()
    return (DATASET_ROOT / "data" / "runs" / value).resolve()


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.run)
    genui_path = run_dir / "genui.jsonl"
    if not genui_path.exists():
        raise FileNotFoundError(f"Missing run genui.jsonl: {genui_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "metric_v4_shadow"
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = load_by_id(run_dir / "responses.jsonl", "response_id")
    native_checks = load_native_checks(run_dir)
    contract_cache = SourceContractCache(output_dir / "contract_cache")
    config = load_v4_reward_config()

    started = time.perf_counter()
    shadow_rows: list[dict[str, Any]] = []
    for index, genui in enumerate(iter_jsonl(genui_path)):
        if args.limit > 0 and index >= args.limit:
            break
        response_id = str(genui.get("response_id") or "")
        response = responses.get(response_id, {})
        source_text = str(genui.get("response_text") or response.get("response_text") or "")
        intent = str(genui.get("intent_bucket") or genui.get("intent") or "") or None
        assets = genui.get("assets") or response.get("assets")
        persisted = genui.get("expected_ui_contract") or response.get("expected_ui_contract")
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

        score_input = dict(genui)
        score_input.update(
            response_text=source_text,
            assets=assets,
            expected_ui_contract=resolution.contract,
            expected_ui_contract_source=resolution.source,
        )
        ui_id = str(genui.get("ui_id") or "")
        native_row = native_checks.get(ui_id)
        result = score_record_v4(score_input, render_row=native_row, config=config)
        generation = genui.get("gen") if isinstance(genui.get("gen"), Mapping) else {}
        shadow_rows.append(
            {
                "ui_id": genui.get("ui_id"),
                "query_id": genui.get("query_id"),
                "response_id": genui.get("response_id"),
                "intent_bucket": intent,
                "model_checkpoint": generation.get("model"),
                "metric_version": result.metric_version,
                "contract_version": resolution.contract.get("contract_version"),
                "contract_source": resolution.source,
                "expected_ui_contract": resolution.contract,
                "renderer_check_result": (
                    native_row.get("renderer_check_result")
                    if isinstance(native_row, Mapping)
                    else {"adapter": "android_native", "attempted": False, "ok": None}
                ),
                "genui_quality_v4": breakdown_to_mapping(result),
            }
        )

    elapsed = time.perf_counter() - started
    # Aggregate the immutable, already-computed per-sample breakdowns. Renderer
    # checks are deliberately omitted here to prevent a second scoring pass.
    aggregate_rows = [
        {
            "ui_id": row.get("ui_id"),
            "intent_bucket": row.get("intent_bucket"),
            "model_checkpoint": row.get("model_checkpoint"),
            "genui_quality_v4": row["genui_quality_v4"],
        }
        for row in shadow_rows
    ]
    summary = aggregate_v4_records(aggregate_rows, config=config)
    summary.update(
        {
            "run_dir": str(run_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "shadow_only": True,
            "source_files_modified": False,
            "elapsed_seconds": elapsed,
            "samples_per_second": len(shadow_rows) / elapsed if elapsed > 0 else None,
        }
    )

    atomic_write_jsonl(output_dir / "scores.jsonl", shadow_rows)
    atomic_write_json(output_dir / "aggregates.json", summary)
    quality = summary.get("quality_0_100", {})
    report = (
        "# GenUI metric v4 shadow benchmark\n\n"
        f"- Run: `{run_dir}`\n"
        f"- Samples: {len(shadow_rows)}\n"
        f"- Mean score: {float(quality.get('mean', 0.0)):.4f}\n"
        f"- Median score: {float(quality.get('median', 0.0)):.4f}\n"
        f"- Runtime: {elapsed:.3f} seconds ({summary['samples_per_second'] or 0.0:.2f} samples/s)\n"
        "- Calibration: **uncalibrated engineering score**; do not interpret as an equal-interval percentage.\n"
        "- Mutation robustness: run the repository metamorphic test suite.\n"
        "- Human preference agreement: pending owner-reviewed calibration labels.\n"
        "- Historical inputs: unchanged; all results are immutable sidecars.\n"
    )
    (output_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "count": len(shadow_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
