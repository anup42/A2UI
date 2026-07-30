from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_2_support import (  # noqa: E402
    contract,
    score,
    simple_spec,
)
from pipeline.genui_quality import load_default_reward_config  # noqa: E402


def test_public_breakdown_is_v5_2_and_formula_is_exact() -> None:
    result = score(simple_spec("Hello world"), "Hello world")
    assert result.metric_version == "5.2.0"
    assert result.metric_name == "GenUI Representation Quality"
    assert result.quality_0_100 == pytest.approx(
        100.0 * result.quality_0_1
    )
    assert result.reward == pytest.approx(
        2.0 * result.quality_0_1 - 1.0
    )
    assert result.quality_0_1 == pytest.approx(
        min(result.base_quality_before_caps, result.cap_0_1)
    )
    assert result.matching_certification
    assert result.dynamic_semantics
    assert result.identity["source_hash"]
    assert result.identity["computed_registry_manifest_hash"]


def test_effective_weights_obey_global_cap_and_sum() -> None:
    result = score(simple_spec(), "Hello world")
    assert result.anti_domination_feasible
    assert sum(result.effective_atomic_weights.values()) == pytest.approx(
        1.0, abs=1e-12
    )
    assert max(result.effective_atomic_weights.values()) <= 0.10 + 1e-12
    assert result.base_quality_before_caps == pytest.approx(
        sum(
            weight
            * float(
                result.atomics[qualified.split(".", 1)[0]][
                    qualified.split(".", 1)[1]
                ]
            )
            for qualified, weight in result.effective_atomic_weights.items()
        ),
        abs=1e-12,
    )


def test_production_invalidity_is_non_dilutable_under_padding() -> None:
    elements = {
        "root": {
            "type": "Stack",
            "props": {},
            "children": ["unsupported", *[f"text_{i}" for i in range(100)]],
        },
        "unsupported": {
            "type": "ProductionUnknown",
            "props": {},
            "children": [],
        },
    }
    elements.update(
        {
            f"text_{index}": {
                "type": "Text",
                "props": {"text": f"Useful {index}"},
                "children": [],
            }
            for index in range(100)
        }
    )
    result = score(
        {"root": "root", "state": {}, "elements": elements},
        "Useful content",
    )
    assert not result.normalization["production_valid"]
    assert result.quality_0_1 <= 0.25
    assert any(
        cap["name"] == "production_invalid"
        for cap in result.active_caps
    )


def test_forbidden_top_level_property_activates_strict_policy() -> None:
    candidate = simple_spec()
    candidate["forbidden"] = True
    result = score(candidate, "Hello world")
    assert not result.normalization["strict_schema_valid"]
    assert any(
        cap["name"] == "strict_format" for cap in result.active_caps
    )


def test_solver_budget_exhaustion_is_uncertified_and_fail_closed() -> None:
    source = "Alpha apple\nBeta banana"
    expected = contract(
        source,
        content_units=["Alpha apple", "Beta banana"],
    )
    candidate = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {},
                "children": ["first", "second"],
            },
            "first": {
                "type": "Text",
                "props": {"text": "Alpha banana"},
                "children": [],
            },
            "second": {
                "type": "Text",
                "props": {"text": "Beta apple"},
                "children": [],
            },
        },
    }
    config = load_default_reward_config()
    config.max_matching_hungarian_work = 1
    config.max_matching_sparse_relaxations = 1
    result = score(
        candidate,
        source,
        expected=expected,
        config=config,
    )
    assert not result.matching_certification["optimality_certified"]
    assert result.matching_certification["approximate_matching_used"]
    assert result.atomics["fidelity"][
        "content_unit_fidelity"
    ] is None
    assert any(
        cap["name"] == "matching_uncertified"
        for cap in result.active_caps
    )
    assert result.cap_0_1 <= 0.50


@pytest.mark.parametrize("completion", ["", "{", "null", [], 7])
def test_malformed_results_are_finite_bounded_and_never_raise(
    completion,
) -> None:
    result = score(completion, "Hello")
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
