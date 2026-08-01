"""Backtest and reliability utilities for GACJ v3.

The legacy replay adapter is deliberately labelled synthetic.  It decomposes
an existing v2 dimension score into five anchored criterion levels to exercise
and verify the v3 host formulas.  It is not a fresh visual judgment and cannot
establish that the refreshed rubric improves judge reliability.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median, pstdev, stdev
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .criterion_judgments_v3 import combine_criterion_passes
from .criterion_protocol_v3 import (
    COMPOSITE_WEIGHTS,
    DIMENSIONS,
    PASS_DIMENSIONS,
    criterion_names,
    protocol_fingerprint,
    validate_criterion_judgment_pass,
)
from .criterion_packets_v3 import build_criterion_review_packets


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            output.append(value)
    return output


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p10": float(np.quantile(ordered, 0.10)),
        "p25": float(np.quantile(ordered, 0.25)),
        "median": median(ordered),
        "mean": mean(ordered),
        "p75": float(np.quantile(ordered, 0.75)),
        "p90": float(np.quantile(ordered, 0.90)),
        "maximum": ordered[-1],
        "population_sd": pstdev(ordered),
        "unique_count": len(set(ordered)),
    }


def _icc_absolute_agreement(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    ratings = np.column_stack(
        [np.asarray(first, dtype=float), np.asarray(second, dtype=float)]
    )
    n, k = ratings.shape
    if n < 2:
        return 0.0
    row_means = ratings.mean(axis=1)
    column_means = ratings.mean(axis=0)
    grand = ratings.mean()
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = ratings - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = (
        ms_rows
        + (k - 1) * ms_error
        + (k * (ms_columns - ms_error) / n)
    )
    if abs(float(denominator)) < 1e-12:
        return 1.0 if np.allclose(ratings[:, 0], ratings[:, 1]) else 0.0
    return float((ms_rows - ms_error) / denominator)


def _repeat_statistics(
    original: Sequence[float],
    repeated: Sequence[float],
) -> dict[str, Any]:
    if len(original) != len(repeated):
        raise ValueError("repeat vectors must be aligned")
    if len(original) < 2:
        return {"count": len(original), "complete": False}
    differences = [
        float(repeated[index]) - float(original[index])
        for index in range(len(original))
    ]
    absolute = [abs(value) for value in differences]
    bias = mean(differences)
    difference_sd = stdev(differences)
    return {
        "count": len(original),
        "complete": len(original) == 96,
        "icc_a_1_absolute_agreement": _icc_absolute_agreement(
            original, repeated
        ),
        "mae": mean(absolute),
        "rmse": math.sqrt(mean(value * value for value in differences)),
        "signed_bias": bias,
        "difference_sd_sample": difference_sd,
        "lower_bland_altman_95": bias - 1.96 * difference_sd,
        "upper_bland_altman_95": bias + 1.96 * difference_sd,
        "absolute_delta_at_least_5_count": sum(value >= 5 for value in absolute),
        "absolute_delta_at_least_10_count": sum(value >= 10 for value in absolute),
        "passes_frozen_gate": (
            _icc_absolute_agreement(original, repeated) >= 0.85
            and mean(absolute) <= 5.0
            and abs(bias) < 2.0
        ),
    }


def _decompose_score_into_anchor_levels(score: float) -> list[int]:
    """Represent any multiple-of-five score as five 0..4 anchor levels."""

    numeric = float(score)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
        raise ValueError("legacy score must be finite and in [0,100]")
    if abs(numeric / 5.0 - round(numeric / 5.0)) > 1e-9:
        raise ValueError("legacy score must be a multiple of five")
    target_sum = int(round(numeric / 5.0))
    base, remainder = divmod(target_sum, 5)
    levels = [base + (1 if index < remainder else 0) for index in range(5)]
    if any(level not in range(5) for level in levels):
        raise AssertionError("invalid anchor decomposition")
    if sum(levels) * 5 != int(round(numeric)):
        raise AssertionError("anchor decomposition does not reproduce score")
    return levels


def legacy_v2_pass_to_criterion_v3(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Synthetic formula replay; this is not a new LLM judgment."""

    pass_type = str(row.get("pass_type") or "")
    if pass_type not in PASS_DIMENSIONS:
        raise ValueError("legacy row has unsupported pass_type")
    raw_dimensions = row.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        raise ValueError("legacy row lacks dimensions")
    dimensions: dict[str, Any] = {}
    for dimension in PASS_DIMENSIONS[pass_type]:
        score = float(raw_dimensions[dimension])
        levels = _decompose_score_into_anchor_levels(score)
        names = criterion_names(dimension)
        dimensions[dimension] = {
            "criteria": {
                name: {
                    "status": "scored",
                    "anchor_level": levels[index],
                    "evidence": [
                        "Synthetic legacy-v2 score replay; no fresh image judgment was performed."
                    ],
                    "defects": [],
                }
                for index, name in enumerate(names)
            },
            "defects": [],
            "worst_defect_severity": "none",
            "confidence_0_1": float(row.get("confidence_0_1", 0.0)),
            "rationale": (
                "Synthetic legacy-v2 decomposition used only to verify the "
                "criterion host formulas and 96-sample data path."
            ),
        }
    output = {
        "schema_version": "genui_anchored_criterion_judgment.v3",
        "protocol_version": "genui_anchored_criterion_rubric.v3.0.0",
        "packet_id": str(row.get("packet_id") or ""),
        "pass_type": pass_type,
        "protocol_fingerprint": protocol_fingerprint(),
        "dimensions": dimensions,
        "fatal_findings": [],
        "cannot_assess": [],
        "overall_rationale": (
            "Legacy replay adapter. This record must not be published as a "
            "fresh criterion-referenced visual label."
        ),
        "judge_task_id": "legacy-v2-replay",
        "judge_model_identifier": "legacy-v2-recorded-judge",
        "judged_at": str(row.get("judged_at") or "2026-08-01T00:00:00+00:00"),
    }
    return validate_criterion_judgment_pass(output)


def _sensitivity_analysis(
    finalized_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_criterion: list[dict[str, Any]] = []
    # The criterion weights are 0.2, so a one-anchor improvement raises its
    # dimension base by five points before a severity ceiling.
    for dimension in DIMENSIONS:
        for criterion in criterion_names(dimension):
            theoretical_composite_delta = COMPOSITE_WEIGHTS[dimension] * 5.0
            per_criterion.append(
                {
                    "dimension": dimension,
                    "criterion": criterion,
                    "one_anchor_level_dimension_delta": 5.0,
                    "one_anchor_level_raw_composite_delta": (
                        theoretical_composite_delta
                    ),
                    "bounded_nonzero": (
                        0.0 < theoretical_composite_delta <= 0.9 + 1e-12
                    ),
                }
            )
    return {
        "sample_count": len(finalized_rows),
        "criterion_count": len(per_criterion),
        "all_one_level_changes_bounded_nonzero": all(
            row["bounded_nonzero"] for row in per_criterion
        ),
        "maximum_one_level_raw_composite_delta": max(
            row["one_anchor_level_raw_composite_delta"] for row in per_criterion
        ),
        "minimum_one_level_raw_composite_delta": min(
            row["one_anchor_level_raw_composite_delta"] for row in per_criterion
        ),
        "rows": per_criterion,
    }


def _evidence_audit(bundle_root: Path) -> dict[str, Any]:
    capture_rows = _read_jsonl(
        bundle_root / "capture" / "native_capture_manifest_96.jsonl"
    )
    by_ui: dict[str, list[dict[str, Any]]] = {}
    hash_failures: list[str] = []
    for row in capture_rows:
        ui_id = str(row.get("ui_id") or "")
        by_ui.setdefault(ui_id, []).append(row)
        if bool(row.get("image_captured", row.get("ok"))):
            relative = Path(str(row.get("local_path") or ""))
            path = relative if relative.is_absolute() else bundle_root / relative
            if not path.exists():
                hash_failures.append(f"missing:{relative.as_posix()}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = str(row.get("screenshot_sha256") or "")
            if expected and actual != expected:
                hash_failures.append(f"hash:{relative.as_posix()}")
    profiles_by_ui = {
        ui_id: {
            str(row.get("viewport_profile") or "")
            for row in rows
            if bool(row.get("image_captured", row.get("ok")))
        }
        for ui_id, rows in by_ui.items()
    }
    required_profiles = {"compact", "medium_700dp", "expanded_900dp"}
    missing_profiles = {
        ui_id: sorted(required_profiles - profiles)
        for ui_id, profiles in profiles_by_ui.items()
        if not required_profiles.issubset(profiles)
    }
    return {
        "ui_count": len(by_ui),
        "capture_row_count": len(capture_rows),
        "captured_image_count": sum(
            bool(row.get("image_captured", row.get("ok")))
            for row in capture_rows
        ),
        "candidate_render_failure_count": sum(
            row.get("failure_class") == "candidate"
            and bool(row.get("required", True))
            for row in capture_rows
        ),
        "all_required_viewport_profiles_present": not missing_profiles,
        "missing_viewport_profiles": missing_profiles,
        "screenshot_hash_failure_count": len(hash_failures),
        "screenshot_hash_failures": hash_failures,
    }


def backtest_v2_repeat_bundle(
    review_bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    build_packets: bool = True,
) -> dict[str, Any]:
    """Exercise GACJ v3 over the 96-repeat review bundle.

    No LLM or external API is called. Existing v2 labels are replayed only to
    verify exact score computation and the data path. Reliability remains the
    v2 reliability until fresh v3 judgments are collected.
    """

    started = time.perf_counter()
    bundle_root = Path(review_bundle_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    packet_result: dict[str, Any] | None = None
    if build_packets:
        packet_result = build_criterion_review_packets(
            bundle_root,
            output_root / "fresh_rejudge_workspace",
        )
    raw_v2 = _read_jsonl(bundle_root / "judgments" / "v2_raw_judgments_96.jsonl")
    v2_by_packet = {
        str(row["packet_id"]): row
        for row in _read_jsonl(
            bundle_root / "judgments" / "v2_judgments_by_packet_96.jsonl"
        )
    }
    replay_passes = [legacy_v2_pass_to_criterion_v3(row) for row in raw_v2]
    replay_by_packet_raw: dict[str, dict[str, dict[str, Any]]] = {}
    for row in replay_passes:
        replay_by_packet_raw.setdefault(row["packet_id"], {})[
            row["pass_type"]
        ] = row
    replay_combined: list[dict[str, Any]] = []
    score_deltas: list[float] = []
    for packet_id in sorted(replay_by_packet_raw):
        passes = replay_by_packet_raw[packet_id]
        if set(passes) != {"screenshot_only", "source_conditioned"}:
            continue
        combined = combine_criterion_passes(
            passes["screenshot_only"], passes["source_conditioned"]
        )
        combined["legacy_replay_only"] = True
        replay_combined.append(combined)
        old = v2_by_packet[packet_id]
        score_deltas.append(
            float(combined["raw_composite_0_100"])
            - float(old["composite_0_100"])
        )
    pair_rows = _read_jsonl(bundle_root / "samples" / "pair_manifest_96.jsonl")
    replay_by_packet = {row["packet_id"]: row for row in replay_combined}
    repeat_rows: list[dict[str, Any]] = []
    original_scores: list[float] = []
    repeated_scores: list[float] = []
    for pair in pair_rows:
        original = replay_by_packet[str(pair["original_packet_id"])]
        repeated = replay_by_packet[str(pair["repeat_packet_id"])]
        original_score = float(original["raw_composite_0_100"])
        repeated_score = float(repeated["raw_composite_0_100"])
        original_scores.append(original_score)
        repeated_scores.append(repeated_score)
        repeat_rows.append(
            {
                "pair_index": pair["pair_index"],
                "ui_id": pair["ui_id"],
                "original_packet_id": pair["original_packet_id"],
                "repeat_packet_id": pair["repeat_packet_id"],
                "original_raw_composite_0_100": original_score,
                "repeat_raw_composite_0_100": repeated_score,
                "signed_delta": repeated_score - original_score,
                "absolute_delta": abs(repeated_score - original_score),
                "legacy_replay_only": True,
            }
        )
    finalized_v2 = _read_jsonl(
        bundle_root / "samples" / "finalized_groundtruth_96.jsonl"
    )
    finalized_replay: list[dict[str, Any]] = []
    final_deltas: list[float] = []
    for row in finalized_v2:
        dimensions = row["dimensions"]
        # Replay finalized dimensions through criterion decompositions.  This
        # verifies the exact criterion formula but does not infer subcriteria.
        criterion_dimension_scores: dict[str, float] = {}
        criterion_breakdown: dict[str, Any] = {}
        for dimension in DIMENSIONS:
            levels = _decompose_score_into_anchor_levels(
                float(dimensions[dimension])
            )
            criteria = {
                name: {
                    "status": "scored",
                    "anchor_level": levels[index],
                    "evidence": ["Synthetic finalized-v2 replay."],
                    "defects": [],
                }
                for index, name in enumerate(criterion_names(dimension))
            }
            from .criterion_protocol_v3 import compute_dimension_score

            result = compute_dimension_score(
                dimension,
                criteria,
                worst_defect_severity="none",
            )
            criterion_dimension_scores[dimension] = result.score_0_100
            criterion_breakdown[dimension] = {
                "criteria": criteria,
                **asdict(result),
            }
        from .criterion_protocol_v3 import compute_criterion_judged_scores

        scores = compute_criterion_judged_scores(criterion_dimension_scores)
        final_deltas.append(
            scores.raw_composite_0_100 - float(row["composite_0_100"])
        )
        finalized_replay.append(
            {
                "ui_id": row["ui_id"],
                "query_id": row["query_id"],
                "intent_bucket": row["intent_bucket"],
                "split": row["split"],
                "criterion_v3_dimension_scores_0_100": criterion_dimension_scores,
                "criterion_v3_raw_composite_0_100": scores.raw_composite_0_100,
                "v2_finalized_composite_0_100": row["composite_0_100"],
                "score_delta": (
                    scores.raw_composite_0_100
                    - float(row["composite_0_100"])
                ),
                "criterion_breakdown": criterion_breakdown,
                "legacy_replay_only": True,
            }
        )
    evidence = _evidence_audit(bundle_root)
    sensitivity = _sensitivity_analysis(finalized_replay)
    reliability = _repeat_statistics(original_scores, repeated_scores)
    summary = {
        "schema_version": "genui_anchored_criterion_backtest.v3",
        "judge_name": "GenUI Anchored Criterion Judge",
        "protocol_fingerprint": protocol_fingerprint(),
        "evaluation_type": "legacy_v2_formula_replay_and_data_path_audit",
        "fresh_llm_rejudgment_performed": False,
        "warning": (
            "The 96 samples were not freshly rejudged by an LLM under the v3 "
            "rubric. Reliability values reproduce v2 labels and must not be "
            "interpreted as v3 reliability."
        ),
        "input_counts": {
            "unique_ui_samples": len(finalized_v2),
            "repeat_pairs": len(pair_rows),
            "raw_v2_pass_rows": len(raw_v2),
            "complete_v2_packets": len(v2_by_packet),
            "criterion_replay_complete_packets": len(replay_combined),
        },
        "packet_build": packet_result,
        "evidence_audit": evidence,
        "formula_replay": {
            "maximum_absolute_packet_score_delta": max(
                (abs(value) for value in score_deltas), default=0.0
            ),
            "maximum_absolute_finalized_score_delta": max(
                (abs(value) for value in final_deltas), default=0.0
            ),
            "all_scores_exactly_reproduced": (
                all(abs(value) < 1e-9 for value in score_deltas)
                and all(abs(value) < 1e-9 for value in final_deltas)
            ),
            "finalized_distribution": _distribution(
                [row["criterion_v3_raw_composite_0_100"] for row in finalized_replay]
            ),
        },
        "repeat_reliability_replayed_from_v2": reliability,
        "sensitivity": {
            key: value for key, value in sensitivity.items() if key != "rows"
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Run the 192-occurrence fresh_rejudge_workspace with the frozen v3 "
            "rubric, then calculate ICC(A,1), MAE, and bias before adjudication."
        ),
    }
    _atomic_write_jsonl(
        output_root / "criterion_v3_legacy_replay_passes.jsonl",
        replay_passes,
    )
    _atomic_write_jsonl(
        output_root / "criterion_v3_legacy_replay_by_packet.jsonl",
        replay_combined,
    )
    _atomic_write_jsonl(
        output_root / "criterion_v3_repeat_replay_96.jsonl",
        repeat_rows,
    )
    _atomic_write_jsonl(
        output_root / "criterion_v3_finalized_replay_96.jsonl",
        finalized_replay,
    )
    _atomic_write_json(
        output_root / "criterion_v3_sensitivity.json",
        sensitivity,
    )
    _atomic_write_json(
        output_root / "criterion_v3_evidence_audit.json",
        evidence,
    )
    _atomic_write_json(
        output_root / "criterion_v3_backtest_summary.json",
        summary,
    )
    return summary


def analyze_fresh_criterion_repeats(
    workspace_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate pre-adjudication reliability for a completed v3 repeat run."""

    root = Path(workspace_dir).resolve()
    identity_rows = _read_jsonl(
        root / "sealed" / "criterion_v3_pair_identity_map.jsonl"
    )
    judgment_rows = _read_jsonl(
        root / "criterion_v3_judgments_by_packet.jsonl"
    )
    judgments = {str(row["packet_id"]): row for row in judgment_rows}
    originals: list[float] = []
    repeats: list[float] = []
    pair_rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    same_task_block: list[str] = []
    identity_by_packet = {str(row["packet_id"]): row for row in identity_rows}
    for identity in identity_rows:
        if int(identity.get("occurrence", 0)) != 1:
            continue
        repeat_id = str(identity["packet_id"])
        original_id = str(identity.get("repeat_of_packet_id") or "")
        original_identity = identity_by_packet.get(original_id)
        if original_identity is None:
            incomplete.append(repeat_id + ":missing_original_identity")
            continue
        if int(original_identity["task_block_index"]) == int(
            identity["task_block_index"]
        ):
            same_task_block.append(repeat_id)
        original = judgments.get(original_id)
        repeated = judgments.get(repeat_id)
        if original is None or repeated is None:
            incomplete.append(repeat_id + ":missing_judgment")
            continue
        if not original.get("evidence_complete") or not repeated.get(
            "evidence_complete"
        ):
            incomplete.append(repeat_id + ":incomplete_evidence")
            continue
        first_score = float(original["raw_composite_0_100"])
        second_score = float(repeated["raw_composite_0_100"])
        originals.append(first_score)
        repeats.append(second_score)
        dimension_deltas = {
            dimension: (
                float(repeated["dimension_scores_0_100"][dimension])
                - float(original["dimension_scores_0_100"][dimension])
            )
            for dimension in DIMENSIONS
        }
        pair_rows.append(
            {
                "ui_id": identity["ui_id"],
                "original_packet_id": original_id,
                "repeat_packet_id": repeat_id,
                "original_raw_composite_0_100": first_score,
                "repeat_raw_composite_0_100": second_score,
                "signed_delta": second_score - first_score,
                "absolute_delta": abs(second_score - first_score),
                "dimension_deltas": dimension_deltas,
                "original_task_block_index": original_identity[
                    "task_block_index"
                ],
                "repeat_task_block_index": identity["task_block_index"],
            }
        )
    overall = _repeat_statistics(originals, repeats)
    dimension_statistics: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        first = [
            float(judgments[row["original_packet_id"]][
                "dimension_scores_0_100"
            ][dimension])
            for row in pair_rows
        ]
        second = [
            float(judgments[row["repeat_packet_id"]][
                "dimension_scores_0_100"
            ][dimension])
            for row in pair_rows
        ]
        dimension_statistics[dimension] = _repeat_statistics(first, second)
    result = {
        "schema_version": "genui_anchored_criterion_repeat_analysis.v3",
        "protocol_fingerprint": protocol_fingerprint(),
        "analysis_stage": "pre_adjudication",
        "scheduled_pair_count": sum(
            int(row.get("occurrence", 0)) == 1 for row in identity_rows
        ),
        "complete_pair_count": len(pair_rows),
        "incomplete_pairs": incomplete,
        "same_fresh_task_block_pairs": same_task_block,
        "overall": overall,
        "dimensions": dimension_statistics,
        "publication_recommendation": (
            "eligible_for_provisional publication review"
            if overall.get("passes_frozen_gate")
            and len(pair_rows) == 96
            and not incomplete
            and not same_task_block
            else "remain provisional"
        ),
        "frozen_gate": {
            "icc_a_1_minimum": 0.85,
            "mae_maximum": 5.0,
            "absolute_bias_strictly_less_than": 2.0,
            "all_thresholds_required": True,
        },
    }
    if output_path is not None:
        target = Path(output_path).resolve()
        _atomic_write_json(target, result)
        _atomic_write_jsonl(
            target.with_name(target.stem + "_pairs.jsonl"), pair_rows
        )
    return result


__all__ = [
    "analyze_fresh_criterion_repeats",
    "backtest_v2_repeat_bundle",
    "legacy_v2_pass_to_criterion_v3",
]
