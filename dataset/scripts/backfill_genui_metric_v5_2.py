#!/usr/bin/env python3
"""Write an immutable v5.1/v5.2 shadow-score sidecar."""

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
    breakdown_to_mapping,
    load_default_reward_config,
    load_v5_1_reward_config,
    score_genui_completion,
    score_genui_completion_v5_1,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "sd": 0.0,
            "q05": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "q95": 0.0,
        }
    ordered = sorted(float(value) for value in values)

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = position - lower
        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "sd": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "q05": quantile(0.05),
        "q25": quantile(0.25),
        "q75": quantile(0.75),
        "q95": quantile(0.95),
    }


def correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def candidate_length(value: Any) -> int:
    return len(
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str
        )
    )


def component_count(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    elements = value.get("elements")
    return len(elements) if isinstance(elements, Mapping) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Run directory or dataset run ID")
    parser.add_argument("--output-dir", type=Path, required=True)
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
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Immutable v5.2 sidecar path already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    source_hash_before = sha256(input_path)
    v5_1_config = load_v5_1_reward_config()
    v5_2_config = load_default_reward_config()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for source_row in iter_jsonl(input_path):
        if args.limit > 0 and len(rows) >= args.limit:
            break
        source = str(source_row.get("response_text") or "")
        intent = (
            str(
                source_row.get("intent_bucket")
                or source_row.get("intent")
                or ""
            )
            or None
        )
        assets = source_row.get("assets")
        expected = (
            source_row.get("expected_ui_contract")
            if isinstance(
                source_row.get("expected_ui_contract"), Mapping
            )
            else None
        )
        raw = source_row.get("genui_raw_completion")
        if raw is None:
            raw = source_row.get("genui_json")
        final = source_row.get("genui_json")
        render_status = source_row.get("renderer_check_result")
        attempted = bool(
            isinstance(render_status, Mapping)
            and render_status.get("attempted")
        )
        render_ok = (
            bool(render_status.get("ok"))
            if attempted
            and isinstance(render_status, Mapping)
            and isinstance(render_status.get("ok"), bool)
            else None
        )
        raw_v5_1 = score_genui_completion_v5_1(
            raw,
            source,
            intent=intent,
            assets=assets,
            expected_ui_contract=expected,
            config=v5_1_config,
        )
        final_v5_1 = score_genui_completion_v5_1(
            final,
            source,
            intent=intent,
            assets=assets,
            expected_ui_contract=expected,
            render_ok=render_ok,
            config=v5_1_config,
        )
        raw_v5_2 = score_genui_completion(
            raw,
            source,
            intent=intent,
            assets=assets,
            expected_ui_contract=expected,
            config=v5_2_config,
        )
        final_v5_2 = score_genui_completion(
            final,
            source,
            intent=intent,
            assets=assets,
            expected_ui_contract=expected,
            render_ok=render_ok,
            config=v5_2_config,
        )
        source_evidence = (
            final_v5_2.evidence.get("source")
            if isinstance(final_v5_2.evidence.get("source"), Mapping)
            else {}
        )
        rows.append(
            {
                "ui_id": source_row.get("ui_id"),
                "intent": intent,
                "v5_1": {
                    "generation": breakdown_to_mapping(raw_v5_1),
                    "final": breakdown_to_mapping(final_v5_1),
                },
                "v5_2": {
                    "generation": breakdown_to_mapping(raw_v5_2),
                    "final": breakdown_to_mapping(final_v5_2),
                },
                "delta": {
                    "v5_2_minus_v5_1_generation_0_100": (
                        raw_v5_2.quality_0_100
                        - raw_v5_1.quality_0_100
                    ),
                    "v5_2_minus_v5_1_final_0_100": (
                        final_v5_2.quality_0_100
                        - final_v5_1.quality_0_100
                    ),
                    "v5_2_final_minus_generation_0_100": (
                        final_v5_2.quality_0_100
                        - raw_v5_2.quality_0_100
                    ),
                },
                "matching_certification": (
                    final_v5_2.matching_certification
                ),
                "dynamic_semantics": final_v5_2.dynamic_semantics,
                "contract_semantic_specificity": source_evidence.get(
                    "contract_semantic_specificity"
                ),
                "count_only_role_count": source_evidence.get(
                    "count_only_role_count", 0
                ),
                "identity": {
                    "metric_fingerprint_v5_1": (
                        final_v5_1.metric_fingerprint
                    ),
                    "metric_fingerprint_v5_2": (
                        final_v5_2.metric_fingerprint
                    ),
                    **{
                        f"v5_2_{key}": value
                        for key, value in final_v5_2.identity.items()
                    },
                    "v5_2_raw_candidate_hash": (
                        raw_v5_2.identity.get("raw_candidate_hash")
                    ),
                    "v5_2_raw_canonical_candidate_hash": (
                        raw_v5_2.identity.get(
                            "canonical_candidate_hash"
                        )
                    ),
                },
                "native_render_status": render_status
                or {
                    "adapter": "android_native",
                    "attempted": False,
                    "ok": None,
                },
                "component_count_diagnostic": component_count(final),
                "final_length_diagnostic": candidate_length(final),
            }
        )

    source_hash_after = sha256(input_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Source genui.jsonl changed during sidecar scoring")
    elapsed = time.perf_counter() - started

    def values(path: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            result.append(float(value))
        return result

    v51_raw = values(("v5_1", "generation", "quality_0_100"))
    v51_final = values(("v5_1", "final", "quality_0_100"))
    v52_raw = values(("v5_2", "generation", "quality_0_100"))
    v52_final = values(("v5_2", "final", "quality_0_100"))
    components = [
        float(row["component_count_diagnostic"]) for row in rows
    ]
    lengths = [float(row["final_length_diagnostic"]) for row in rows]
    count = len(rows)
    summary = {
        "metric_name": "GenUI Representation Quality",
        "metric_version": "5.2.0",
        "metric_fingerprint": (
            rows[0]["identity"]["metric_fingerprint_v5_2"]
            if rows
            else None
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "shadow_only": True,
        "calibration_status": "uncalibrated_engineering_score",
        "weight_learning_performed": False,
        "source_run": str(run_dir),
        "source_genui_sha256": source_hash_before,
        "source_files_modified": False,
        "count": count,
        "elapsed_seconds": elapsed,
        "samples_per_second": count / elapsed if elapsed else None,
        "v5_1_generation": distribution(v51_raw),
        "v5_1_final": distribution(v51_final),
        "v5_2_generation": distribution(v52_raw),
        "v5_2_final": distribution(v52_final),
        "v5_2_minus_v5_1_generation": distribution(
            [new - old for new, old in zip(v52_raw, v51_raw)]
        ),
        "v5_2_minus_v5_1_final": distribution(
            [new - old for new, old in zip(v52_final, v51_final)]
        ),
        "v5_2_final_minus_generation": distribution(
            [final - raw for final, raw in zip(v52_final, v52_raw)]
        ),
        "active_cap_rate": (
            sum(bool(row["v5_2"]["final"]["active_caps"]) for row in rows)
            / count
            if count
            else 0.0
        ),
        "binding_cap_rate": (
            sum(bool(row["v5_2"]["final"]["binding_caps"]) for row in rows)
            / count
            if count
            else 0.0
        ),
        "matching_uncertified_rate": (
            sum(
                not bool(
                    row["matching_certification"].get(
                        "optimality_certified"
                    )
                )
                for row in rows
            )
            / count
            if count
            else 0.0
        ),
        "dynamic_unknown_rate": (
            sum(
                int(row["dynamic_semantics"].get("unknown_count") or 0)
                > 0
                for row in rows
            )
            / count
            if count
            else 0.0
        ),
        "contract_count_only_rate": (
            sum(int(row["count_only_role_count"] or 0) > 0 for row in rows)
            / count
            if count
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
            / count
            if count
            else 0.0
        ),
        "diagnostic_correlations": {
            "component_count_vs_v5_2_final": correlation(
                components, v52_final
            ),
            "completion_length_vs_v5_2_final": correlation(
                lengths, v52_final
            ),
            "not_score_inputs": True,
        },
        "limitations": [
            "Ten rows are insufficient for weight learning or calibration.",
            "No human equal-interval calibration is claimed.",
            "Native-render evidence is present only for attempted renders.",
        ],
    }
    scores_path = output_dir / "scores.jsonl"
    aggregates_path = output_dir / "aggregates.json"
    atomic_write(
        scores_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )
    atomic_write(
        aggregates_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    report = (
        "# GenUI metric v5.2 immutable shadow audit\n\n"
        f"- Samples: {count}\n"
        f"- V5.1 final mean: {summary['v5_1_final']['mean']:.4f}\n"
        f"- V5.2 final mean: {summary['v5_2_final']['mean']:.4f}\n"
        f"- Final delta mean: "
        f"{summary['v5_2_minus_v5_1_final']['mean']:.4f}\n"
        f"- Matching uncertified rate: "
        f"{summary['matching_uncertified_rate']:.4f}\n"
        f"- Dynamic unknown rate: "
        f"{summary['dynamic_unknown_rate']:.4f}\n"
        f"- Native render attempt rate: "
        f"{summary['native_render_attempt_rate']:.4f}\n"
        "- This is an uncalibrated, non-gating engineering audit. Ten rows "
        "were not used to fit weights or thresholds.\n"
    )
    atomic_write(output_dir / "audit_report.md", report)
    manifest = {
        "metric_version": "5.2.0",
        "metric_fingerprint": summary["metric_fingerprint"],
        "source_run": str(run_dir),
        "source_genui_sha256": source_hash_before,
        "scores_sha256": sha256(scores_path),
        "aggregates_sha256": sha256(aggregates_path),
        "created_at": summary["created_at"],
        "immutable": True,
    }
    atomic_write(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "count": count,
                "metric_fingerprint": summary["metric_fingerprint"],
                "v5_1_final_mean": summary["v5_1_final"]["mean"],
                "v5_2_final_mean": summary["v5_2_final"]["mean"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
