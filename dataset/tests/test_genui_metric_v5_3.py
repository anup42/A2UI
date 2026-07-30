from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_3_support import score, simple_spec  # noqa: E402
from pipeline.genui_quality import load_v5_3_reward_config  # noqa: E402


def test_public_formula_identity_and_weight_invariants() -> None:
    result = score(simple_spec(), "Hello world")
    assert result.metric_version == "5.3.0"
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


@pytest.mark.parametrize("completion", ["", "{", "null", [], 7])
def test_malformed_is_finite_bounded_and_nonthrowing(completion) -> None:
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


def test_breakdown_has_required_v5_3_diagnostics() -> None:
    result = score(simple_spec(), "Hello world")
    assert result.atomic_applicability
    assert result.evidence_ownership
    assert result.unsupported_external_additions
    assert result.dynamic_evidence_certification


def test_matching_input_budget_fails_closed_with_diagnostics() -> None:
    config = load_v5_3_reward_config()
    config.max_matching_inputs = 1
    result = score(simple_spec(), "Hello world", config=config)
    assert result.atomics["fidelity"]["content_unit_fidelity"] is None
    assert any(
        code.startswith("matching_input_budget:content:")
        for code in result.matching_certification["diagnostic_codes"]
    )
    assert any(
        cap["name"] == "matching_uncertified"
        for cap in result.active_caps
    )
