from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from jsonschema import Draft202012Validator


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_judge.criterion_analysis_v3 import (  # noqa: E402
    legacy_v2_pass_to_criterion_v3,
)
from pipeline.genui_judge.criterion_judgments_v3 import (  # noqa: E402
    append_criterion_judgment_pass,
    combine_criterion_passes,
)
from pipeline.genui_judge.criterion_protocol_v3 import (  # noqa: E402
    COMPOSITE_WEIGHTS,
    DIMENSIONS,
    FATAL_POLICY_CAPS,
    PASS_DIMENSIONS,
    compute_criterion_judged_scores,
    compute_dimension_score,
    criterion_names,
    protocol_fingerprint,
    protocol_mapping,
    validate_criterion_judgment_pass,
)


def _dimension(
    name: str,
    *,
    level: int = 3,
    severity: str = "none",
    not_observable: str | None = None,
) -> dict:
    criteria = {}
    for criterion in criterion_names(name):
        criteria[criterion] = {
            "status": (
                "not_observable" if criterion == not_observable else "scored"
            ),
            "anchor_level": None if criterion == not_observable else level,
            "evidence": ["Specific evidence."],
            "defects": [],
        }
    defects = (
        []
        if severity == "none"
        else [{"severity": severity, "description": "Independent defect."}]
    )
    return {
        "criteria": criteria,
        "defects": defects,
        "worst_defect_severity": severity,
        "confidence_0_1": 0.9,
        "rationale": "Anchored criterion synthesis.",
    }


def _pass(packet_id: str, pass_type: str, *, level: int = 3) -> dict:
    return {
        "schema_version": "genui_anchored_criterion_judgment.v3",
        "protocol_version": "genui_anchored_criterion_rubric.v3.0.0",
        "packet_id": packet_id,
        "pass_type": pass_type,
        "protocol_fingerprint": protocol_fingerprint(),
        "dimensions": {
            name: _dimension(name, level=level)
            for name in PASS_DIMENSIONS[pass_type]
        },
        "fatal_findings": [],
        "cannot_assess": [],
        "overall_rationale": "Pass synthesis.",
        "judge_task_id": "fresh-task-a",
        "judge_model_identifier": "frozen-model",
        "judged_at": (
            "2026-08-01T00:01:00+00:00"
            if pass_type == "source_conditioned"
            else "2026-08-01T00:00:00+00:00"
        ),
    }


def test_every_dimension_has_five_equal_budget_criteria() -> None:
    mapping = protocol_mapping()
    for dimension in DIMENSIONS:
        criteria = mapping["dimension_definitions"][dimension]["criteria"]
        assert len(criteria) == 5
        assert sum(item["weight"] for item in criteria) == pytest.approx(1.0)


def test_dimension_is_computed_from_anchors_not_direct_score() -> None:
    dimension = DIMENSIONS[0]
    names = criterion_names(dimension)
    levels = [4, 4, 4, 3, 3]
    criteria = {
        name: {
            "status": "scored",
            "anchor_level": levels[index],
            "evidence": [],
            "defects": [],
        }
        for index, name in enumerate(names)
    }
    result = compute_dimension_score(
        dimension,
        criteria,
        worst_defect_severity="none",
    )
    assert result.base_0_100 == 90.0
    assert result.score_0_100 == 90.0


def test_major_defect_cannot_be_averaged_away() -> None:
    dimension = DIMENSIONS[2]
    criteria = {
        name: {
            "status": "scored",
            "anchor_level": 4,
            "evidence": [],
            "defects": [],
        }
        for name in criterion_names(dimension)
    }
    result = compute_dimension_score(
        dimension,
        criteria,
        worst_defect_severity="major",
    )
    assert result.base_0_100 == 100.0
    assert result.score_0_100 == 55.0


def test_not_observable_prevents_full_score() -> None:
    dimension = DIMENSIONS[8]
    missing = criterion_names(dimension)[0]
    raw = _dimension(dimension, not_observable=missing)
    result = compute_dimension_score(
        dimension,
        raw["criteria"],
        worst_defect_severity="none",
    )
    assert not result.evidence_complete
    assert result.score_0_100 != result.score_0_100  # NaN


def test_headline_scores_are_deterministic_and_bounded() -> None:
    dimensions = {name: 75.0 for name in DIMENSIONS}
    result = compute_criterion_judged_scores(dimensions)
    assert result.source_representation_0_100 == 75.0
    assert result.rendered_ux_0_100 == 75.0
    assert result.raw_composite_0_100 == pytest.approx(75.0)
    assert result.policy_capped_composite_0_100 == pytest.approx(75.0)
    assert result.policy_cap_0_100 == 100.0


def test_policy_caps_are_separate_from_raw_score() -> None:
    dimensions = {name: 95.0 for name in DIMENSIONS}
    result = compute_criterion_judged_scores(
        dimensions,
        fatal_findings=["core_required_semantic_role_absent"],
    )
    assert result.raw_composite_0_100 == pytest.approx(95.0)
    assert result.policy_cap_0_100 == FATAL_POLICY_CAPS[
        "core_required_semantic_role_absent"
    ]
    assert result.policy_capped_composite_0_100 == 55.0


def test_one_anchor_improvement_is_positive_and_bounded() -> None:
    base = {name: 50.0 for name in DIMENSIONS}
    original = compute_criterion_judged_scores(base).raw_composite_0_100
    for dimension in DIMENSIONS:
        changed = dict(base)
        changed[dimension] += 5.0
        improved = compute_criterion_judged_scores(changed).raw_composite_0_100
        delta = improved - original
        assert delta == pytest.approx(COMPOSITE_WEIGHTS[dimension] * 5.0)
        assert 0.0 < delta <= 0.9 + 1e-9


def test_pass_validation_rejects_direct_dimension_score() -> None:
    value = _pass("p-one", "screenshot_only")
    value["dimensions"][DIMENSIONS[2]]["score_0_100"] = 75
    # Runtime validator ignores unknown nested keys only if hand-coded; the
    # strict JSON Schema must reject it.
    schema = json.loads(
        (
            DATASET_ROOT
            / "schema"
            / "genui_anchored_criterion_judgment_v3.schema.json"
        ).read_text(encoding="utf-8")
    )
    errors = list(Draft202012Validator(schema).iter_errors(value))
    assert errors
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_criterion_judgment_pass(value)


def test_worst_severity_must_match_defect_ledger() -> None:
    value = _pass("p-one", "screenshot_only")
    dimension = DIMENSIONS[2]
    value["dimensions"][dimension]["defects"] = [
        {"severity": "major", "description": "Central task is unclear."}
    ]
    value["dimensions"][dimension]["worst_defect_severity"] = "minor"
    with pytest.raises(ValueError, match="does not match"):
        validate_criterion_judgment_pass(value)


def test_two_pass_combination_computes_scores_in_host() -> None:
    result = combine_criterion_passes(
        _pass("p-complete", "screenshot_only"),
        _pass("p-complete", "source_conditioned"),
    )
    assert result["evidence_complete"]
    assert result["raw_composite_0_100"] == pytest.approx(75.0)
    assert result["dimension_scores_0_100"] == {
        name: 75.0 for name in DIMENSIONS
    }


def test_append_enforces_order_and_model_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="screenshot_only"):
        append_criterion_judgment_pass(
            tmp_path,
            _pass("p-order", "source_conditioned"),
        )
    append_criterion_judgment_pass(
        tmp_path,
        _pass("p-order", "screenshot_only"),
    )
    source = _pass("p-order", "source_conditioned")
    source["judge_model_identifier"] = "different-model"
    with pytest.raises(ValueError, match="model identifier changed"):
        append_criterion_judgment_pass(tmp_path, source)


def test_legacy_replay_exactly_reproduces_any_multiple_of_five() -> None:
    for score in range(0, 101, 5):
        row = {
            "packet_id": f"p-{score}",
            "pass_type": "source_conditioned",
            "dimensions": {
                name: score
                for name in PASS_DIMENSIONS["source_conditioned"]
            },
            "confidence_0_1": 0.9,
            "judged_at": "2026-08-01T00:01:00+00:00",
        }
        replay = legacy_v2_pass_to_criterion_v3(row)
        for dimension in PASS_DIMENSIONS["source_conditioned"]:
            result = compute_dimension_score(
                dimension,
                replay["dimensions"][dimension]["criteria"],
                worst_defect_severity="none",
            )
            assert result.score_0_100 == score


def test_source_pass_must_follow_screenshot_timestamp() -> None:
    screenshot = _pass("p-time", "screenshot_only")
    source = _pass("p-time", "source_conditioned")
    source["judged_at"] = screenshot["judged_at"]
    with pytest.raises(ValueError, match="must follow"):
        combine_criterion_passes(screenshot, source)


def test_verified_render_failure_requires_zero_dimensions() -> None:
    screenshot = _pass("p-failed", "screenshot_only", level=3)
    source = _pass("p-failed", "source_conditioned", level=3)
    screenshot["fatal_findings"] = ["verified_candidate_render_failure"]
    with pytest.raises(ValueError, match="requires zero"):
        combine_criterion_passes(screenshot, source)


def test_packet_schema_is_closed() -> None:
    schema = json.loads(
        (
            DATASET_ROOT
            / "schema"
            / "genui_anchored_criterion_packet_v3.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    screenshot = schema["properties"]["screenshots"]["items"]
    assert screenshot["additionalProperties"] is False
