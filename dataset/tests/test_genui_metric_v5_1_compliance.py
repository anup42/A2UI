from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_1_support import action_spec, contract, score  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    MetricV51InitializationError,
    ensure_v5_1_validation_ready,
)
from pipeline.genui_quality import _core  # noqa: E402
from pipeline.genui_quality.metrics_v5_1 import (  # noqa: E402
    semantic_role_coverage_v5_1,
    table_fidelity_v5_1,
    table_pair_metrics_v5_1,
)


def _chart(
    element_id: str,
    title: str,
    x_key: str,
    y_key: str,
) -> dict[str, object]:
    return {
        "component_id": element_id,
        "title": title,
        "chart_type": "bar",
        "x_fields": [x_key],
        "y_fields": [y_key],
        "series": [y_key],
        "data_identity": f"/{element_id}",
        "signature": f"{title}:{x_key}:{y_key}",
    }


def test_semantic_chart_instances_require_correct_one_to_one_matches() -> None:
    expected = {
        "chart": [
            {
                "id": "revenue",
                "title": "Revenue by quarter",
                "chart_type": "bar",
                "x_fields": ["quarter"],
                "y_fields": ["revenue"],
            },
            {
                "id": "cost",
                "title": "Cost by region",
                "chart_type": "bar",
                "x_fields": ["region"],
                "y_fields": ["cost"],
            },
        ]
    }
    correct = {
        "chart": [
            _chart("a", "Revenue by quarter", "quarter", "revenue"),
            _chart("b", "Cost by region", "region", "cost"),
        ]
    }
    wrong = {
        "chart": [
            _chart("a", "Visitors by day", "day", "visitors"),
            _chart("b", "Inventory by store", "store", "inventory"),
        ]
    }
    duplicate = {
        "chart": [
            _chart("a", "Revenue by quarter", "quarter", "revenue"),
            _chart("b", "Revenue by quarter", "quarter", "revenue"),
        ]
    }
    correct_value, _, correct_diag = semantic_role_coverage_v5_1(
        expected, correct, threshold=0.70, exact_dense_limit=64,
        max_edges=65536, top_k=16
    )
    wrong_value, _, _ = semantic_role_coverage_v5_1(
        expected, wrong, threshold=0.70, exact_dense_limit=64,
        max_edges=65536, top_k=16
    )
    duplicate_value, _, duplicate_diag = semantic_role_coverage_v5_1(
        expected, duplicate, threshold=0.70, exact_dense_limit=64,
        max_edges=65536, top_k=16
    )
    assert correct_value == 1.0
    assert correct_diag["chart"]["matched_required_count"] == 2
    assert wrong_value == 0.0
    assert duplicate_value <= 0.5
    assert duplicate_diag["chart"]["matched_required_count"] <= 1


def test_formula_code_and_email_content_matters() -> None:
    expected = {
        "formula": [{"content": "profit equals revenue minus costs"}],
        "code": [{"content": "print total", "language": "python"}],
        "email": [{"subject": "Quarterly results", "to": "team@example.com"}],
    }
    wrong = {
        "formula": [{"content": "distance equals speed times time"}],
        "code": [{"content": "SELECT * FROM users", "language": "sql"}],
        "email": [{"subject": "Holiday notice", "to": "other@example.com"}],
    }
    value, roles, _ = semantic_role_coverage_v5_1(
        expected, wrong, threshold=0.70, exact_dense_limit=64,
        max_edges=65536, top_k=16
    )
    assert value == 0.0
    assert set(roles.values()) == {0.0}


def test_row_key_is_a_hard_association_gate_and_compound_keys_work() -> None:
    source = _core.SourceTable(
        headers=("Region", "Year", "Revenue"),
        rows=(("US", "2025", "10"), ("EU", "2025", "20")),
        row_key=("Region", "Year"),
    )
    correct = _core.OutputTable(
        "table", "Table", ["Region", "Year", "Revenue"],
        ["region", "year", "revenue"],
        [
            {"region": "US", "year": "2025", "revenue": "10"},
            {"region": "EU", "year": "2025", "revenue": "20"},
        ],
    )
    wrong = _core.OutputTable(
        "table", "Table", ["Region", "Year", "Revenue"],
        ["region", "year", "revenue"],
        [
            {"region": "APAC", "year": "2025", "revenue": "10"},
            {"region": "LATAM", "year": "2025", "revenue": "20"},
        ],
    )
    kwargs = dict(
        beta=2.0, exact_dense_limit=64, max_edges=65536, top_k=16
    )
    correct_result = table_pair_metrics_v5_1(source, correct, **kwargs)
    wrong_result = table_pair_metrics_v5_1(source, wrong, **kwargs)
    assert correct_result.key_exact_match_rate == 1.0
    assert correct_result.score == 1.0
    assert wrong_result.key_exact_match_rate == 0.0
    assert wrong_result.score < correct_result.score - 0.40


def test_chart_cannot_satisfy_table_without_explicit_policy() -> None:
    table = _core.SourceTable(
        headers=("Item", "Value"), rows=(("A", "1"),)
    )
    chart = _core.OutputTable(
        "chart", "Chart", ["Item", "Value"], ["item", "value"],
        [{"item": "A", "value": "1"}],
    )
    kwargs = dict(
        beta=2.0, exact_dense_limit=64, max_edges=65536, top_k=16
    )
    default_score, default_diag = table_fidelity_v5_1(
        [table], [], [chart], **kwargs
    )
    equivalent = _core.SourceTable(
        headers=table.headers,
        rows=table.rows,
        representation_policy="structured_equivalent",
    )
    equivalent_score, equivalent_diag = table_fidelity_v5_1(
        [equivalent], [], [chart], **kwargs
    )
    assert default_score == 0.0
    assert default_diag["matched_required_count"] == 0
    assert equivalent_score == 1.0
    assert equivalent_diag["matched_required_count"] == 1


def test_same_url_actions_remain_distinct_and_caps_follow_matches() -> None:
    source = (
        "Action: [Read details] https://example.com/x\n"
        "Action: [Save details] https://example.com/x"
    )
    expected = contract(source)
    one = score(
        action_spec([("Read details", "https://example.com/x", "openUrl")]),
        source,
        expected=expected,
    )
    two = score(
        action_spec(
            [
                ("Read details", "https://example.com/x", "openUrl"),
                ("Save details", "https://example.com/x", "openUrl"),
            ]
        ),
        source,
        expected=expected,
    )
    assert one.evidence["action_matching"]["required_count"] == 2
    assert one.evidence["action_matching"]["matched_required_count"] == 1
    assert one.evidence["action_matching"]["role_coverage"] == 0.5
    assert any(cap["name"] == "partial_action" for cap in one.active_caps)
    assert two.evidence["action_matching"]["role_coverage"] == 1.0
    assert not any(cap["name"] == "partial_action" for cap in two.active_caps)


def test_jsonschema_absence_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_v5_1_validation_ready.cache_clear()
    original = importlib.import_module

    def blocked(name: str, package: str | None = None):
        if name == "jsonschema":
            raise ModuleNotFoundError("blocked for test")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked)
    with pytest.raises(
        MetricV51InitializationError, match="requires the jsonschema"
    ):
        ensure_v5_1_validation_ready()
    ensure_v5_1_validation_ready.cache_clear()
