from __future__ import annotations

import math
from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_1_support import score, simple_spec  # noqa: E402
from pipeline.genui_quality import load_v5_1_reward_config  # noqa: E402


def test_public_schema_formula_identity_and_bounds() -> None:
    result = score(simple_spec(), "Hello world")
    assert result.metric_name == "GenUI Representation Quality"
    assert result.metric_version == "5.1.0"
    assert len(result.metric_fingerprint) == 64
    assert result.quality_0_100 == 100.0 * result.quality_0_1
    assert result.reward == 2.0 * result.quality_0_1 - 1.0
    assert result.quality_0_1 == min(
        result.base_quality_before_caps, result.cap_0_1
    )
    assert set(result.identity) == {
        "source_hash",
        "raw_candidate_hash",
        "canonical_candidate_hash",
        "expected_contract_hash",
    }
    assert math.isfinite(result.reward)
    assert -1.0 <= result.reward <= 1.0


def test_global_effective_weight_runtime_invariants() -> None:
    config = load_v5_1_reward_config()
    result = score(simple_spec(), "Hello world", config=config)
    assert result.anti_domination_feasible
    assert math.isclose(
        sum(result.effective_atomic_weights.values()),
        1.0,
        abs_tol=1e-10,
    )
    assert all(
        math.isfinite(value)
        and 0.0 <= value <= config.max_atomic_global_weight + 1e-10
        for value in result.effective_atomic_weights.values()
    )


def test_active_and_binding_caps_are_distinct() -> None:
    candidate = simple_spec()
    candidate["debug"] = True
    result = score(candidate, "Hello world")
    assert any(cap["name"] == "strict_format" for cap in result.active_caps)
    assert result.cap_margin == (
        result.base_quality_before_caps - result.cap_0_1
    )
    assert all(
        result.base_quality_before_caps > float(cap["cap"]) + 1e-12
        for cap in result.binding_caps
    )


def test_no_count_padding_reward() -> None:
    source = "Hello world"
    original = score(simple_spec(source), source)
    padded = simple_spec(source)
    for index in range(100):
        padded["elements"][f"unused_{index}"] = {
            "type": "Text",
            "props": {"text": "decorative padding"},
            "children": [],
        }
    changed = score(padded, source)
    assert changed.quality_0_1 <= original.quality_0_1 + 1e-12
