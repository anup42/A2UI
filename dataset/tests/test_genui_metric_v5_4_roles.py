from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_ROOT.parent
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_quality import (  # noqa: E402
    extract_expected_ui_contract_v5_4,
    load_v5_4_reward_config,
)
from pipeline.genui_quality.evidence_v5_4 import (  # noqa: E402
    DYNAMIC_PARITY_VECTOR_VERSION,
    DynamicEvidenceResolverV54,
)
from pipeline.genui_quality.metrics_v5_4 import (  # noqa: E402
    semantic_role_coverage_v5_4,
)


def _role_score(expected: dict, actual: dict):
    return semantic_role_coverage_v5_4(
        expected,
        actual,
        expected_exact_indices=None,
        threshold=0.70,
        exact_dense_limit=64,
        max_edges=65536,
        top_k=16,
    )


def test_duplicate_signatures_are_capacity_constrained_inside_assignment() -> None:
    expected = {
        "chart": [
            {
                "required": True,
                "minimum_count": 1,
                "title": "Revenue by month",
            },
            {
                "required": True,
                "minimum_count": 1,
                "title": "Orders by region",
            },
        ]
    }
    valid = {
        "chart": [
            {"title": "Revenue by month"},
            {"title": "Orders by region"},
        ]
    }
    padded = {
        "chart": [
            {"title": "Revenue by month"},
            {"title": "Revenue by month"},
            {"title": "Orders by region"},
        ]
    }
    baseline = _role_score(expected, valid)
    duplicate = _role_score(expected, padded)
    assert duplicate[0] == pytest.approx(baseline[0])
    assert duplicate[1] == pytest.approx(baseline[1])
    diagnostics = duplicate[4]["chart"]
    assert diagnostics["distinct_candidate_signature_count"] == 2
    assert diagnostics["semantic_capacity_slot_count"] == 2
    assert diagnostics["optimality_certifies_constrained_problem"]


def test_count_only_roles_are_explicitly_low_specificity() -> None:
    result = _role_score(
        {"chart": [{"required": True, "minimum_count": 2}]},
        {"chart": [{"title": "A"}, {"title": "B"}]},
    )
    assert result[0] is None
    assert result[1] == pytest.approx(1.0)
    assert result[4]["chart"]["constraint_model"] == (
        "count_only_low_specificity"
    )
    assert result[5] == 0.0


@pytest.mark.parametrize(
    ("source", "count"),
    [
        ("# Revenue (as chart)\nRevenue by month.", 1),
        ("Revenue by month as a chart", 1),
        ("Create two charts: Revenue by month and Orders by region.", 2),
        (
            "Charts to include:\n- Revenue by month\n- Orders by region",
            2,
        ),
    ],
)
def test_common_source_chart_phrases_create_semantic_instances(
    source: str, count: int
) -> None:
    contract = extract_expected_ui_contract_v5_4(source)
    requirements = contract["role_requirements"]["chart"]
    assert len(requirements) == count
    assert all(item.get("title") for item in requirements)
    assert contract["contract_semantic_specificity"] == pytest.approx(1.0)


def test_checked_in_source_role_benchmark() -> None:
    path = (
        DATASET_ROOT
        / "tests"
        / "fixtures"
        / "genui_metric_v5_4_source_role_benchmark.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["renderer_effective_semantics"] == "5.4.1"
    assert payload["reapproved_for_renderer_capabilities"] == "2.0.0"
    for case in payload["cases"]:
        contract = extract_expected_ui_contract_v5_4(case["source"])
        requirements = contract["role_requirements"].get(case["role"], [])
        assert len(requirements) == case["count"], case["source"]
        assert contract["contract_semantic_specificity"] == pytest.approx(
            case["specificity"]
        )


def test_python_and_android_parity_corpus_is_identical_and_passes() -> None:
    python_path = (
        DATASET_ROOT
        / "tests"
        / "fixtures"
        / "flat_expr_parity_vectors_v5_4.json"
    )
    android_path = (
        REPO_ROOT
        / "android"
        / "app"
        / "src"
        / "test"
        / "resources"
        / "flat_expr_parity_vectors_v5_4.json"
    )
    assert python_path.read_bytes() == android_path.read_bytes()
    payload = json.loads(python_path.read_text(encoding="utf-8"))
    assert payload["version"] == DYNAMIC_PARITY_VECTOR_VERSION
    config = load_v5_4_reward_config()
    for vector in payload["vectors"]:
        resolver = DynamicEvidenceResolverV54(vector["state"], config)
        actual = resolver.resolve(
            vector["expression"],
            item=vector.get("item"),
            index=vector.get("index"),
            base_path=vector.get("base_path"),
        )
        assert actual == vector["expected_value"], vector["name"]
        assert resolver.unknown == vector["expected_unknown_codes"]
