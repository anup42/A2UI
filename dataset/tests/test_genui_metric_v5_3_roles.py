from __future__ import annotations

from pathlib import Path
import sys

import pytest

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_quality import extract_expected_ui_contract_v5_3  # noqa: E402
from pipeline.genui_quality import score_genui_completion_v5_3  # noqa: E402


PHRASES = [
    "Create a bar chart showing quarterly revenue.",
    "Show a line chart of temperature over time.",
    "Visualize profit by region as a pie chart.",
    "Include a chart for revenue by quarter.",
    "Add a chart showing cost by region.",
    "Please provide a chart of monthly sales.",
    "Use two charts to show revenue and profit.",
    "## Visualizations\n- Revenue by quarter\n- Profit by region",
    "Display the formula for compound interest.",
    "Include a code block showing hello world.",
    "Draft an email preview to Alice.",
    "Create a form for contact details.",
    "Add two formulas for area and volume.",
    "Please provide console output showing the result.",
]


@pytest.mark.parametrize("source", PHRASES)
def test_ordinary_role_phrases_are_semantic_or_explicitly_low_specificity(
    source: str,
) -> None:
    value = extract_expected_ui_contract_v5_3(source)
    requirements = value.get("role_requirements") or {}
    diagnostics = value.get("extraction_diagnostics") or []
    assert requirements or any(
        str(item).startswith("count_only_role:") for item in diagnostics
    )


def test_count_only_role_lowers_specificity() -> None:
    value = extract_expected_ui_contract_v5_3("Please show a chart.")
    assert value["count_only_role_count"] >= 1
    assert value["contract_semantic_specificity"] < 1.0


def test_correct_chart_differs_materially_from_unrelated_chart() -> None:
    source = "Create a bar chart showing quarterly revenue."

    def candidate(title: str, x_key: str, y_key: str):
        return {
            "root": "chart",
            "state": {"rows": [{x_key: "Q1", y_key: 10}]},
            "elements": {
                "chart": {
                    "type": "Chart",
                    "props": {
                        "title": title,
                        "chartType": "bar",
                        "xKey": x_key,
                        "yKey": y_key,
                        "statePath": "/rows",
                    },
                    "children": [],
                }
            },
        }

    correct = score_genui_completion_v5_3(
        candidate("quarterly revenue", "quarter", "revenue"), source
    )
    unrelated = score_genui_completion_v5_3(
        candidate("regional costs", "region", "cost"), source
    )
    assert correct.atomics["semantic_mapping"][
        "semantic_role_instance_fidelity"
    ] == 1.0
    assert correct.atomics["fidelity"]["content_unit_fidelity"] is None
    assert correct.atomics["fidelity"]["output_block_precision"] is None
    assert unrelated.atomics["semantic_mapping"][
        "semantic_role_instance_fidelity"
    ] == 0.0
    assert correct.quality_0_100 > unrelated.quality_0_100 + 5.0
