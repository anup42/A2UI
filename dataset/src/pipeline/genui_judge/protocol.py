"""Frozen scoring protocol for the single-Codex GenUI visual judge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


JUDGE_SCHEMA_VERSION = "genui_single_codex_groundtruth.v2"
JUDGE_PROTOCOL_VERSION = "genui_single_codex_visual_rubric.v2.0.0"
JUDGE_AUTHORITY = "provisional_single_codex_groundtruth"

DIMENSIONS: tuple[str, ...] = (
    "visible_source_representation",
    "semantic_component_appropriateness",
    "visual_hierarchy_task_focus",
    "layout_spacing_alignment",
    "typography_readability_contrast",
    "information_density_progressive_disclosure",
    "interaction_affordance_learnability",
    "visual_consistency_polish_trust",
    "responsive_adaptive_behavior",
    "visible_accessibility",
)

SOURCE_DIMENSIONS: tuple[str, ...] = DIMENSIONS[:2]
VISUAL_DIMENSIONS: tuple[str, ...] = DIMENSIONS[2:]

COMPOSITE_WEIGHTS: dict[str, float] = {
    "visible_source_representation": 0.18,
    "semantic_component_appropriateness": 0.14,
    "visual_hierarchy_task_focus": 0.14,
    "layout_spacing_alignment": 0.12,
    "typography_readability_contrast": 0.10,
    "information_density_progressive_disclosure": 0.10,
    "interaction_affordance_learnability": 0.08,
    "visual_consistency_polish_trust": 0.04,
    "responsive_adaptive_behavior": 0.05,
    "visible_accessibility": 0.05,
}

PASS_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "screenshot_only": VISUAL_DIMENSIONS,
    "source_conditioned": SOURCE_DIMENSIONS,
}

SCALE_ANCHORS: dict[int, str] = {
    0: "absent, broken, or unusable",
    25: "major failure",
    50: "materially partial",
    75: "good with noticeable defects",
    100: "no material defect",
}

RUBRIC_DEFINITIONS: dict[str, str] = {
    "visible_source_representation": (
        "Visible coverage and correctness of source facts, values, sections, "
        "relationships, actions, and required evidence across captured states."
    ),
    "semantic_component_appropriateness": (
        "Whether the visible UI uses renderer-supported components whose roles "
        "fit the source, including tables, charts, formulas, media, forms, and actions."
    ),
    "visual_hierarchy_task_focus": (
        "Whether the primary task and most important information are immediately "
        "clear, ordered, grouped, and visually prioritized."
    ),
    "layout_spacing_alignment": (
        "Alignment, spacing, containment, clipping, overflow, empty regions, and "
        "the stability of the layout across the captured page."
    ),
    "typography_readability_contrast": (
        "Legibility of text and values, type hierarchy, wrapping, truncation, "
        "contrast, and readability at the supplied viewport sizes."
    ),
    "information_density_progressive_disclosure": (
        "Whether the amount of visible information is manageable and whether "
        "detail is exposed in an understandable order without hiding essentials."
    ),
    "interaction_affordance_learnability": (
        "Whether visible controls and captured internal states communicate what "
        "can be done and how, without evaluating external side effects."
    ),
    "visual_consistency_polish_trust": (
        "Consistency, finish, credible presentation, absence of broken placeholders, "
        "and visual details that affect confidence in the UI."
    ),
    "responsive_adaptive_behavior": (
        "Whether compact, medium, and expanded captures preserve content, hierarchy, "
        "readability, and usable layout without clipping or inappropriate stretching."
    ),
    "visible_accessibility": (
        "Visible accessibility only: contrast, text sizing, identifiable controls, "
        "labels, touch-target presentation, and non-color-only communication."
    ),
}


def judge_instructions_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "prompts"
        / "genui_single_codex_judge_v2.md"
    )


def judge_instructions_sha256() -> str:
    return hashlib.sha256(judge_instructions_path().read_bytes()).hexdigest()


@dataclass(frozen=True)
class JudgedScores:
    source_representation_0_100: float
    rendered_ux_0_100: float
    composite_0_100: float


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def protocol_mapping() -> dict[str, Any]:
    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "protocol_version": JUDGE_PROTOCOL_VERSION,
        "authority": JUDGE_AUTHORITY,
        "packet_schema_version": "genui_single_codex_blinded_packet.v2",
        "judgment_schema_version": JUDGE_SCHEMA_VERSION,
        "dimensions": list(DIMENSIONS),
        "pass_dimensions": {
            key: list(value) for key, value in PASS_DIMENSIONS.items()
        },
        "weights": dict(COMPOSITE_WEIGHTS),
        "scale_anchors": {
            str(key): value for key, value in SCALE_ANCHORS.items()
        },
        "definitions": dict(RUBRIC_DEFINITIONS),
        "score_increment": 5,
        "formulas": {
            "source_representation": (
                "(0.18*S1 + 0.14*S2) / 0.32"
            ),
            "rendered_ux": (
                "(0.14*S3 + 0.12*S4 + 0.10*S5 + 0.10*S6 + "
                "0.08*S7 + 0.04*S8 + 0.05*S9 + 0.05*S10) / 0.68"
            ),
            "composite": "sum(weight_i * S_i)",
        },
        "reference_truth_policy": (
            "Treat source response as reference truth; do not judge its factual "
            "or writing quality."
        ),
        "ordered_passes": ["screenshot_only", "source_conditioned"],
        "repeat_adjudication": {
            "composite_absolute_delta_trigger": 10.0,
            "dimension_absolute_delta_trigger": 20.0,
            "confidence_below_trigger": 0.75,
            "anchor_band_width": 25.0,
            "adjudicated_value": "per_dimension_median_of_three",
            "otherwise": "retain_original",
        },
        "model_identity_policy": (
            "Stop and create a new protocol version if the available Codex "
            "model identifier changes."
        ),
        "render_failure_policy": (
            "A candidate native-render failure verified on the checkout-matched "
            "APK receives zero on every dimension. Infrastructure failures are "
            "retried twice and are not quality failures."
        ),
        "blinding": {
            "metric_scores": False,
            "generator_identity": False,
            "flat_spec_json": False,
            "previous_judgments": False,
        },
    }


def protocol_fingerprint() -> str:
    return hashlib.sha256(
        _canonical_json(protocol_mapping()).encode("utf-8")
    ).hexdigest()


def _coerce_score(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError(f"{name} must be finite and in [0, 100]")
    if abs(score / 5.0 - round(score / 5.0)) > 1e-9:
        raise ValueError(f"{name} must be in increments of 5")
    return score


def compute_judged_scores(
    dimensions: Mapping[str, Any],
) -> JudgedScores:
    missing = [name for name in DIMENSIONS if name not in dimensions]
    extra = sorted(set(dimensions) - set(DIMENSIONS))
    if missing:
        raise ValueError(f"missing dimensions: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unknown dimensions: {', '.join(extra)}")
    values = {
        name: _coerce_score(dimensions[name], name=name)
        for name in DIMENSIONS
    }
    representation = (
        COMPOSITE_WEIGHTS[DIMENSIONS[0]] * values[DIMENSIONS[0]]
        + COMPOSITE_WEIGHTS[DIMENSIONS[1]] * values[DIMENSIONS[1]]
    ) / 0.32
    ux = sum(
        COMPOSITE_WEIGHTS[name] * values[name]
        for name in VISUAL_DIMENSIONS
    ) / 0.68
    composite = sum(
        COMPOSITE_WEIGHTS[name] * values[name] for name in DIMENSIONS
    )
    output = JudgedScores(
        source_representation_0_100=representation,
        rendered_ux_0_100=ux,
        composite_0_100=composite,
    )
    for name, score in asdict(output).items():
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            raise AssertionError(f"invalid computed {name}: {score}")
    return output


def validate_judgment_pass(
    value: Mapping[str, Any],
    *,
    expected_packet_id: str | None = None,
    expected_pass_type: str | None = None,
) -> dict[str, Any]:
    packet_id = str(value.get("packet_id") or "").strip()
    if not packet_id:
        raise ValueError("packet_id is required")
    if expected_packet_id is not None and packet_id != expected_packet_id:
        raise ValueError("packet_id mismatch")
    pass_type = str(value.get("pass_type") or "").strip()
    if pass_type not in PASS_DIMENSIONS:
        raise ValueError(f"unsupported pass_type: {pass_type!r}")
    if expected_pass_type is not None and pass_type != expected_pass_type:
        raise ValueError("pass_type mismatch")
    fingerprint = str(value.get("protocol_fingerprint") or "")
    if fingerprint != protocol_fingerprint():
        raise ValueError("protocol_fingerprint mismatch")
    raw_dimensions = value.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        raise ValueError("dimensions must be an object")
    required = PASS_DIMENSIONS[pass_type]
    missing = [name for name in required if name not in raw_dimensions]
    extra = sorted(set(raw_dimensions) - set(required))
    if missing:
        raise ValueError(f"missing pass dimensions: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unexpected pass dimensions: {', '.join(extra)}")
    dimensions = {
        name: _coerce_score(raw_dimensions[name], name=name)
        for name in required
    }
    confidence = value.get("confidence_0_1")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("confidence_0_1 must be finite and in [0, 1]")

    def string_list(name: str) -> list[str]:
        raw = value.get(name, [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(f"{name} must be an array")
        result = [str(item).strip() for item in raw if str(item).strip()]
        return result[:32]

    rationale = str(value.get("rationale") or "").strip()
    if not rationale:
        raise ValueError("rationale is required")
    judged_at = str(value.get("judged_at") or "").strip()
    if not judged_at:
        judged_at = datetime.now(timezone.utc).isoformat()
    try:
        parsed_judged_at = datetime.fromisoformat(
            judged_at.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("judged_at must be an ISO-8601 timestamp") from exc
    if parsed_judged_at.tzinfo is None:
        raise ValueError("judged_at must include a timezone")
    judge_task_id = str(value.get("judge_task_id") or "").strip()
    judge_model_identifier = str(
        value.get("judge_model_identifier") or ""
    ).strip()
    if not judge_task_id:
        raise ValueError("judge_task_id is required")
    if not judge_model_identifier:
        raise ValueError("judge_model_identifier is required")
    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "packet_id": packet_id,
        "pass_type": pass_type,
        "protocol_version": JUDGE_PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "dimensions": dimensions,
        "confidence_0_1": float(confidence),
        "evidence_observations": string_list("evidence_observations"),
        "visible_defects": string_list("visible_defects"),
        "cannot_assess": string_list("cannot_assess"),
        "rationale": rationale,
        "judge_task_id": judge_task_id,
        "judge_model_identifier": judge_model_identifier,
        "judged_at": judged_at,
    }


__all__ = [
    "COMPOSITE_WEIGHTS",
    "DIMENSIONS",
    "JUDGE_AUTHORITY",
    "JUDGE_PROTOCOL_VERSION",
    "JUDGE_SCHEMA_VERSION",
    "JudgedScores",
    "PASS_DIMENSIONS",
    "RUBRIC_DEFINITIONS",
    "SCALE_ANCHORS",
    "SOURCE_DIMENSIONS",
    "VISUAL_DIMENSIONS",
    "compute_judged_scores",
    "judge_instructions_path",
    "judge_instructions_sha256",
    "protocol_fingerprint",
    "protocol_mapping",
    "validate_judgment_pass",
]
