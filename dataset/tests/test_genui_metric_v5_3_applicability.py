from __future__ import annotations

from pathlib import Path
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_3_support import action_spec, score, simple_spec  # noqa: E402


def test_candidate_only_table_chart_never_changes_applicability() -> None:
    source = "A short greeting."
    plain = score(simple_spec(source), source)
    table = simple_spec(source)
    table["elements"]["table"] = {
        "type": "Table",
        "props": {"columns": [{"key": "x", "label": "X"}], "rows": [{"x": "1"}]},
        "children": [],
    }
    table["elements"]["root"]["children"].append("table")
    candidate = score(table, source)
    for result in (plain, candidate):
        assert result.atomics["fidelity"]["markdown_table_fidelity"] is None
        assert not result.atomic_applicability[
            "fidelity.markdown_table_fidelity"
        ]


def test_local_control_is_not_source_action_fidelity() -> None:
    result = score(
        action_spec([("Toggle", "/enabled", "setState")]),
        "Toggle the local view.",
    )
    assert result.atomics["fidelity"][
        "action_and_source_link_fidelity"
    ] is None
    assert result.unsupported_external_additions[
        "unsupported_external_action_count"
    ] == 0


def test_unrelated_external_url_uses_dedicated_signal() -> None:
    result = score(
        action_spec([("Elsewhere", "https://unrelated.example/x", "openUrl")]),
        "Read this summary.",
    )
    assert result.atomics["fidelity"][
        "action_and_source_link_fidelity"
    ] is None
    assert result.atomics["fidelity"][
        "unsupported_external_addition_precision"
    ] == 0.0


def test_decorative_icon_does_not_enter_source_media_fidelity() -> None:
    spec = simple_spec("Hello")
    spec["elements"]["icon"] = {
        "type": "Icon",
        "props": {"name": "sparkle", "decorative": True},
        "children": [],
    }
    spec["elements"]["root"]["children"].append("icon")
    result = score(spec, "Hello")
    assert result.atomics["fidelity"]["media_fidelity"] is None


def test_structured_equivalent_chart_satisfies_source_table_without_table_cap() -> None:
    source = (
        "Chart: Revenue by quarter\n\n"
        "| Quarter | Revenue |\n|---|---|\n| Q1 | 10 |\n| Q2 | 20 |"
    )
    spec = {
        "root": "chart",
        "state": {
            "rows": [
                {"Quarter": "Q1", "Revenue": "10"},
                {"Quarter": "Q2", "Revenue": "20"},
            ]
        },
        "elements": {
            "chart": {
                "type": "Chart",
                "props": {
                    "title": "Revenue by quarter",
                    "chartType": "bar",
                    "xKey": "Quarter",
                    "yKey": "Revenue",
                    "statePath": "/rows",
                },
                "children": [],
            }
        },
    }
    result = score(spec, source)
    assert result.atomics["fidelity"]["markdown_table_fidelity"] == 1.0
    assert not any(
        cap["name"] == "missing_table" for cap in result.active_caps
    )


def test_product_list_table_is_not_penalized_below_text_dump() -> None:
    source = "Products:\n- Alpha: $10\n- Beta: $20"
    table = {
        "root": "table",
        "state": {},
        "elements": {
            "table": {
                "type": "Table",
                "props": {
                    "columns": [
                        {"key": "name", "label": "Product"},
                        {"key": "price", "label": "Price"},
                    ],
                    "rows": [
                        {"name": "Alpha", "price": "$10"},
                        {"name": "Beta", "price": "$20"},
                    ],
                },
                "children": [],
            }
        },
    }
    text = {
        "root": "text",
        "state": {},
        "elements": {
            "text": {
                "type": "Text",
                "props": {"text": source},
                "children": [],
            }
        },
    }
    table_result = score(table, source)
    text_result = score(text, source)
    assert table_result.atomics["fidelity"][
        "markdown_table_fidelity"
    ] is None
    assert table_result.quality_0_100 >= text_result.quality_0_100
