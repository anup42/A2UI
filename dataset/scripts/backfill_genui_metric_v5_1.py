#!/usr/bin/env python3
"""Create an immutable dual-surface GenUI metric v5.1 audit sidecar."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


DATASET_ROOT = Path(__file__).resolve().parents[1]
DATASET_SRC = DATASET_ROOT / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.genui_quality import (  # noqa: E402
    aggregate_v5_1_records,
    breakdown_to_mapping,
    load_v5_1_reward_config,
    load_v5_reward_config,
    score_genui_completion_v5_1,
    score_genui_completion_v5_0,
)
from pipeline.genui_quality.identity_v5_1 import (  # noqa: E402
    metric_fingerprint_v5_1,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "sd": 0.0}
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "q05": quantile(0.05),
        "q25": quantile(0.25),
        "q75": quantile(0.75),
        "q95": quantile(0.95),
    }


def correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator > 0.0 else None


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Run ID or directory containing genui.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    supplied = Path(args.run)
    run_dir = (
        supplied.resolve()
        if supplied.is_absolute() or supplied.exists()
        else (DATASET_ROOT / "data" / "runs" / supplied).resolve()
    )
    input_path = run_dir / "genui.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Immutable output already exists and is non-empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    v5_1_config = load_v5_1_reward_config()
    v5_0_config = load_v5_reward_config()
    fingerprint = metric_fingerprint_v5_1(v5_1_config)
    rows: list[dict[str, Any]] = []
    aggregate_inputs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for source_row in iter_jsonl(input_path):
        if args.limit > 0 and len(rows) >= args.limit:
            break
        source = str(source_row.get("response_text") or "")
        intent = str(
            source_row.get("intent_bucket") or source_row.get("intent") or ""
        ) or None
        assets = source_row.get("assets")
        contract = (
            source_row.get("expected_ui_contract")
            if isinstance(source_row.get("expected_ui_contract"), Mapping)
            else None
        )
        raw = source_row.get("genui_raw_completion")
        if raw is None:
            raw = source_row.get("genui_json")
        final = source_row.get("genui_json")
        render = source_row.get("renderer_check_result")
        render_ok = (
            bool(render.get("ok"))
            if isinstance(render, Mapping) and render.get("ok") is not None
            else None
        )
        stored_v5 = (
            source_row.get("genui_quality_v5")
            if isinstance(source_row.get("genui_quality_v5"), Mapping)
            else {}
        )
        fresh_v5 = score_genui_completion_v5_0(
            raw,
            source,
            intent=intent,
            assets=assets,
            expected_ui_contract=contract,
            config=v5_0_config,
        )
        raw_v5_1 = score_genui_completion_v5_1(
            raw,
            source,
            intent=intent,
            assets=assets,
            expected_ui_contract=contract,
            config=v5_1_config,
        )
        final_v5_1 = score_genui_completion_v5_1(
            final,
            source,
            intent=intent,
            assets=assets,
            expected_ui_contract=contract,
            render_ok=render_ok,
            config=v5_1_config,
        )
        final_mapping = breakdown_to_mapping(final_v5_1)
        raw_mapping = breakdown_to_mapping(raw_v5_1)
        final_spec = final if isinstance(final, Mapping) else {}
        elements = (
            final_spec.get("elements")
            if isinstance(final_spec.get("elements"), Mapping)
            else {}
        )
        rows.append(
            {
                "ui_id": source_row.get("ui_id"),
                "intent": intent,
                "stored_v5_0_raw_score": stored_v5.get("quality_0_100"),
                "fresh_v5_0_raw_score": fresh_v5.quality_0_100,
                "generation_reward_v5_1": raw_mapping,
                "render_artifact_quality_v5_1": final_mapping,
                "raw_final_delta": (
                    final_v5_1.quality_0_100 - raw_v5_1.quality_0_100
                ),
                "metric_fingerprint_v5_1": fingerprint,
                "source_hash": final_v5_1.identity.get("source_hash"),
                "contract_hash": final_v5_1.identity.get(
                    "expected_contract_hash"
                ),
                "raw_candidate_hash": raw_v5_1.identity.get(
                    "raw_candidate_hash"
                ),
                "raw_canonical_candidate_hash": raw_v5_1.identity.get(
                    "canonical_candidate_hash"
                ),
                "final_candidate_hash": final_v5_1.identity.get(
                    "raw_candidate_hash"
                ),
                "final_canonical_candidate_hash": final_v5_1.identity.get(
                    "canonical_candidate_hash"
                ),
                "effective_atomic_weights": final_v5_1.effective_atomic_weights,
                "dimensions": final_v5_1.dimensions,
                "atomics": final_v5_1.atomics,
                "active_caps": final_v5_1.active_caps,
                "binding_caps": final_v5_1.binding_caps,
                "matching_complete": final_v5_1.evidence.get(
                    "matching_complete"
                ),
                "dynamic_unknowns": final_v5_1.evidence.get(
                    "dynamic_expression_unknown_codes", []
                ),
                "native_render_status": render
                or {
                    "adapter": "android_native",
                    "attempted": False,
                    "ok": None,
                },
                "component_count_diagnostic": len(elements),
                "final_json_length_diagnostic": len(
                    json.dumps(final, ensure_ascii=False, sort_keys=True)
                ),
            }
        )
        aggregate_inputs.append(
            {
                **source_row,
                "generation_reward_v5_1": raw_mapping,
                "render_artifact_quality_v5_1": final_mapping,
            }
        )

    elapsed = time.perf_counter() - started
    final_scores = [
        float(row["render_artifact_quality_v5_1"]["quality_0_100"])
        for row in rows
    ]
    raw_scores = [
        float(row["generation_reward_v5_1"]["quality_0_100"])
        for row in rows
    ]
    components = [float(row["component_count_diagnostic"]) for row in rows]
    lengths = [float(row["final_json_length_diagnostic"]) for row in rows]
    aggregate = aggregate_v5_1_records(
        aggregate_inputs, config=v5_1_config
    )
    summary = {
        **aggregate,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "shadow_only": True,
        "source_run": str(run_dir),
        "source_genui_sha256": file_sha256(input_path),
        "source_files_modified": False,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(rows) / elapsed if elapsed else None,
        "raw_score_distribution": distribution(raw_scores),
        "final_score_distribution": distribution(final_scores),
        "matching_incomplete_rate": (
            sum(not bool(row["matching_complete"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "dynamic_unknown_rate": (
            sum(bool(row["dynamic_unknowns"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "native_render_attempt_rate": (
            sum(
                bool(
                    isinstance(row["native_render_status"], Mapping)
                    and row["native_render_status"].get("attempted")
                )
                for row in rows
            )
            / len(rows)
            if rows
            else 0.0
        ),
        "descriptive_correlations": {
            "component_count_vs_final_quality": correlation(
                components, final_scores
            ),
            "json_length_vs_final_quality": correlation(lengths, final_scores),
            "not_score_inputs": True,
        },
    }
    atomic_write(
        output_dir / "scores.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
    )
    atomic_write(
        output_dir / "aggregates.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    report = (
        "# GenUI metric v5.1 immutable ten-row audit\n\n"
        f"- Samples: {len(rows)}\n"
        f"- Fingerprint: `{fingerprint}`\n"
        f"- Raw mean: {float(summary['raw_score_distribution']['mean']):.4f}\n"
        f"- Final mean: {float(summary['final_score_distribution']['mean']):.4f}\n"
        f"- Active cap rate: {float(summary['active_cap_rate']):.4f}\n"
        f"- Binding cap rate: {float(summary['binding_cap_rate']):.4f}\n"
        f"- Matching-incomplete rate: {float(summary['matching_incomplete_rate']):.4f}\n"
        f"- Dynamic-unknown rate: {float(summary['dynamic_unknown_rate']):.4f}\n"
        f"- Native-render attempt rate: {float(summary['native_render_attempt_rate']):.4f}\n"
        "- Status: non-gating, uncalibrated engineering score; no weights were "
        "fit to these rows.\n"
    )
    atomic_write(output_dir / "audit_report.md", report)
    atomic_write(
        output_dir / "manifest.json",
        json.dumps(
            {
                "metric_version": "5.1.0",
                "metric_fingerprint": fingerprint,
                "source_run": str(run_dir),
                "source_genui_sha256": file_sha256(input_path),
                "scores_sha256": file_sha256(output_dir / "scores.jsonl"),
                "aggregates_sha256": file_sha256(
                    output_dir / "aggregates.json"
                ),
                "created_at": summary["created_at"],
                "immutable": True,
            },
            indent=2,
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "count": len(rows),
                "metric_fingerprint": fingerprint,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
