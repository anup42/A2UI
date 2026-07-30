from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_4_support import (  # noqa: E402
    TABLE_SOURCE,
    root_spec,
    table,
    text,
)
from pipeline.genui_quality import (  # noqa: E402
    generation_reward_v5_4,
    render_artifact_quality_v5_4,
)


def test_public_formula_identity_and_weight_invariants() -> None:
    result = render_artifact_quality_v5_4(
        root_spec({"text": text("Hello world")}, ["text"]),
        "Hello world",
    )
    assert result.metric_version == "5.4.0"
    assert result.quality_0_100 == pytest.approx(
        100.0 * result.quality_0_1
    )
    assert result.reward == pytest.approx(2.0 * result.quality_0_1 - 1.0)
    assert result.metric_fingerprint
    assert result.reward_pipeline_fingerprint
    assert result.anti_domination_feasible
    assert sum(result.effective_atomic_weights.values()) == pytest.approx(
        1.0, abs=1e-12
    )
    assert max(result.effective_atomic_weights.values()) <= 0.10 + 1e-12
    assert result.artifact_quality_0_100 == pytest.approx(
        result.quality_0_100
    )


@pytest.mark.parametrize("completion", ["", "{", "null", [], 7])
def test_malformed_is_finite_bounded_and_nonthrowing(completion: object) -> None:
    result = generation_reward_v5_4(completion, "Hello")
    assert all(
        math.isfinite(value)
        for value in (
            result.reward,
            result.quality_0_1,
            result.quality_0_100,
            result.cap_0_1,
        )
    )
    assert -1.0 <= result.reward <= 1.0
    assert 0.0 <= result.quality_0_1 <= 1.0


def test_table_and_code_duplication_cannot_improve_score() -> None:
    table_only = root_spec({"table": table()}, ["table"])
    table_duplicate = root_spec(
        {
            "table": table(),
            "duplicate": text("Paris 21 C Rome 25 C"),
        },
        ["table", "duplicate"],
    )
    before = render_artifact_quality_v5_4(table_only, TABLE_SOURCE)
    after = render_artifact_quality_v5_4(table_duplicate, TABLE_SOURCE)
    assert after.quality_0_1 <= before.quality_0_1
    assert after.evidence_ownership["redundant_generic_unit_ids"]

    source = "```python\nprint(42)\n```"
    code = {
        "type": "CodeBlock",
        "props": {"code": "print(42)", "language": "python"},
        "children": [],
    }
    before = render_artifact_quality_v5_4(
        root_spec({"code": code}, ["code"]), source
    )
    after = render_artifact_quality_v5_4(
        root_spec(
            {"code": code, "duplicate": text("print(42)")},
            ["code", "duplicate"],
        ),
        source,
    )
    assert after.quality_0_1 <= before.quality_0_1
    assert after.evidence_ownership["redundant_generic_unit_ids"]


def test_structured_values_are_not_generic_but_requested_summary_is() -> None:
    table_only = render_artifact_quality_v5_4(
        root_spec({"table": table()}, ["table"]),
        TABLE_SOURCE,
    )
    assert table_only.evidence_ownership["required_generic_count"] == 0

    source = TABLE_SOURCE + "\n\nSummary: Paris is cooler than Rome."
    missing_summary = render_artifact_quality_v5_4(
        root_spec({"table": table()}, ["table"]), source
    )
    represented_summary = render_artifact_quality_v5_4(
        root_spec(
            {
                "table": table(),
                "summary": text("Paris is cooler than Rome."),
            },
            ["table", "summary"],
        ),
        source,
    )
    assert represented_summary.evidence_ownership[
        "required_generic_count"
    ] == 1
    assert not represented_summary.evidence_ownership[
        "redundant_generic_unit_ids"
    ]
    assert represented_summary.quality_0_1 > missing_summary.quality_0_1


def test_accessibility_is_penalty_only_and_candidate_created() -> None:
    baseline = root_spec({"text": text("Hello")}, ["text"])
    labeled_button = {
        "type": "Button",
        "props": {
            "label": "Irrelevant",
            "action": {
                "action": "openUrl",
                "params": {"url": "https://example.test"},
            },
        },
        "children": [],
    }
    unlabeled_button = {
        "type": "Button",
        "props": {
            "action": {
                "action": "openUrl",
                "params": {"url": "https://example.test"},
            }
        },
        "children": [],
    }
    base_score = render_artifact_quality_v5_4(baseline, "Hello")
    labeled_score = render_artifact_quality_v5_4(
        root_spec(
            {"text": text("Hello"), "button": labeled_button},
            ["text", "button"],
        ),
        "Hello",
    )
    unlabeled_score = render_artifact_quality_v5_4(
        root_spec(
            {"text": text("Hello"), "button": unlabeled_button},
            ["text", "button"],
        ),
        "Hello",
    )
    assert labeled_score.quality_0_1 <= base_score.quality_0_1 + 1e-12
    assert unlabeled_score.quality_0_1 < labeled_score.quality_0_1
    assert unlabeled_score.accessibility_conformance["penalty_only"]

    labeled_field = {
        "type": "TextField",
        "props": {"label": "Irrelevant local field"},
        "children": [],
    }
    field_score = render_artifact_quality_v5_4(
        root_spec(
            {"text": text("Hello"), "field": labeled_field},
            ["text", "field"],
        ),
        "Hello",
    )
    assert field_score.quality_0_1 <= base_score.quality_0_1 + 1e-12


def test_raw_format_penalty_is_generation_only() -> None:
    candidate = root_spec({"text": text("Hello")}, ["text"])
    raw = json.dumps(candidate, separators=(",", ":"))
    wrapped = f"Here is the JSON:\n```json\n{raw}\n```"
    strict_generation = generation_reward_v5_4(raw, "Hello")
    wrapped_generation = generation_reward_v5_4(wrapped, "Hello")
    strict_artifact = render_artifact_quality_v5_4(raw, "Hello")
    wrapped_artifact = render_artifact_quality_v5_4(wrapped, "Hello")
    assert wrapped_generation.quality_0_1 < strict_generation.quality_0_1
    assert wrapped_artifact.quality_0_1 == pytest.approx(
        strict_artifact.quality_0_1, abs=1e-12
    )
    assert not wrapped_generation.raw_json_envelope[
        "exact_single_json_value"
    ]
    assert wrapped_generation.raw_json_envelope["utility"] < 1.0
    multiple = generation_reward_v5_4(f"{raw}\n{{}}", "Hello")
    assert multiple.raw_json_envelope["extra_json_value_present"]
    assert multiple.raw_json_envelope["utility"] < 1.0


def test_production_invalidity_is_non_dilutable_under_padding() -> None:
    invalid = {
        "type": "UnsupportedProductionType",
        "props": {},
        "children": [],
    }
    small = root_spec(
        {"text": text("Hello"), "invalid": invalid},
        ["text", "invalid"],
    )
    padded_elements = {
        "text": text("Hello"),
        "invalid": invalid,
        **{
            f"padding_{index}": text(f"padding {index}")
            for index in range(100)
        },
    }
    padded = root_spec(
        padded_elements,
        ["text", "invalid", *[f"padding_{index}" for index in range(100)]],
    )
    small_score = generation_reward_v5_4(small, "Hello")
    padded_score = generation_reward_v5_4(padded, "Hello")
    assert small_score.quality_0_1 <= 0.25
    assert padded_score.quality_0_1 <= 0.25
    assert padded_score.quality_0_1 <= small_score.quality_0_1 + 1e-12
