from __future__ import annotations

from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_quality import (  # noqa: E402
    extract_expected_ui_contract_v5_2,
    prepare_source_context_v5_2,
)
from pipeline.genui_quality.source_contract import (  # noqa: E402
    resolve_expected_ui_contract_v5_2,
    source_contract_cache_key_v5_2,
)
from pipeline.genui_quality.metrics_v5_2 import (  # noqa: E402
    semantic_role_coverage_v5_2,
)


MATCHING_ARGS = {
    "threshold": 0.70,
    "exact_dense_limit": 64,
    "max_edges": 65536,
    "top_k": 16,
}


def test_two_chart_heading_and_inline_language_extract_instances() -> None:
    heading = extract_expected_ui_contract_v5_2(
        "## Two Charts to Include\n"
        "1. Revenue by quarter\n"
        "2. Profit by region"
    )
    inline = extract_expected_ui_contract_v5_2(
        "Create two charts: Revenue by quarter and Profit by region"
    )
    explicit = extract_expected_ui_contract_v5_2(
        "Chart 1: Revenue by quarter\nChart 2: Profit by region"
    )
    for contract in (heading, inline, explicit):
        requirements = contract["role_requirements"]["chart"]
        assert [item["title"] for item in requirements] == [
            "Revenue by quarter",
            "Profit by region",
        ]
        assert contract["contract_semantic_specificity"] == 1.0


def test_formula_and_code_lists_extract_source_supported_semantics() -> None:
    formulas = extract_expected_ui_contract_v5_2(
        "## Three Formulas\n- Net margin\n- Gross margin\n- ROI"
    )
    assert len(formulas["role_requirements"]["formula"]) == 3
    code = extract_expected_ui_contract_v5_2(
        "## Two CodeBlocks\n"
        "1. Python data loader\n"
        "2. SQL aggregation query"
    )
    assert len(code["role_requirements"]["code"]) == 2


def test_unrelated_charts_do_not_satisfy_semantic_instances() -> None:
    expected = {
        "chart": [
            {"id": "revenue", "title": "Revenue by quarter"},
            {"id": "profit", "title": "Profit by region"},
        ]
    }
    unrelated = {
        "chart": [
            {"component_id": "a", "title": "Weather by city"},
            {"component_id": "b", "title": "Flights by carrier"},
        ]
    }
    exact = {
        "chart": [
            {"component_id": "a", "title": "Revenue by quarter"},
            {"component_id": "b", "title": "Profit by region"},
        ]
    }
    bad, _, _, _, _, _ = semantic_role_coverage_v5_2(
        expected, unrelated, **MATCHING_ARGS
    )
    good, _, _, _, _, _ = semantic_role_coverage_v5_2(
        expected, exact, **MATCHING_ARGS
    )
    assert bad == 0.0
    assert good == 1.0


def test_count_only_requirement_is_never_semantic_fidelity_one() -> None:
    expected = {
        "chart": [
            {
                "id": "count_only_1",
                "required": True,
                "minimum_count": 2,
                "interchangeable": True,
            }
        ]
    }
    actual = {
        "chart": [
            {"component_id": "a", "title": "One"},
            {"component_id": "b", "title": "Two"},
        ]
    }
    semantic, count, _, semantic_values, diagnostics, specificity = (
        semantic_role_coverage_v5_2(
            expected, actual, **MATCHING_ARGS
        )
    )
    assert count == 1.0
    assert semantic is None
    assert semantic_values["chart"] is None
    assert diagnostics["chart"]["count_only"]
    assert specificity == 0.0


def test_duplicate_signatures_cannot_satisfy_distinct_noninterchangeable_roles() -> None:
    expected = {
        "chart": [
            {"id": "one", "title": "Revenue"},
            {"id": "two", "title": "Revenue"},
        ]
    }
    duplicates = {
        "chart": [
            {"component_id": "a", "title": "Revenue"},
            {"component_id": "b", "title": "Revenue"},
        ]
    }
    semantic, _, _, _, diagnostics, _ = semantic_role_coverage_v5_2(
        expected, duplicates, **MATCHING_ARGS
    )
    assert semantic == 0.5
    assert diagnostics["chart"]["matched_required_count"] == 1

    for requirement in expected["chart"]:
        requirement["interchangeable"] = True
    semantic_interchangeable, _, _, _, diagnostics, _ = (
        semantic_role_coverage_v5_2(
            expected, duplicates, **MATCHING_ARGS
        )
    )
    assert semantic_interchangeable == 1.0
    assert diagnostics["chart"]["matched_required_count"] == 2


def test_explicit_empty_human_role_requirements_remain_authoritative() -> None:
    source = "Create two charts: Revenue and Profit"
    persisted = extract_expected_ui_contract_v5_2(source)
    persisted["role_requirements"] = {}
    persisted["contract_semantic_specificity"] = 0.0
    prepared = prepare_source_context_v5_2(
        source,
        expected_ui_contract=persisted,
        expected_ui_contract_source="human benchmark",
    )
    assert prepared.contract_resolution.source == "human benchmark"
    assert prepared.expected_contract["role_requirements"] == {}


def test_v5_2_cache_identity_is_intent_sensitive() -> None:
    source = "Create a compact result."
    assert source_contract_cache_key_v5_2(
        source, intent="weather"
    ) != source_contract_cache_key_v5_2(
        source, intent="finance"
    )


def test_stale_persisted_source_hash_is_rejected_not_called_human() -> None:
    stale = extract_expected_ui_contract_v5_2("Different source")
    resolved = resolve_expected_ui_contract_v5_2(
        "Actual source",
        persisted=stale,
        persisted_source="human benchmark",
    )
    assert resolved.source == "deterministic fallback"
    assert any("source_hash" in error for error in resolved.errors)
