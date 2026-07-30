from __future__ import annotations

import copy
from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_2_support import (  # noqa: E402
    action_spec,
    contract,
    score,
    simple_spec,
)


def test_id_renaming_and_element_reordering_are_invariant() -> None:
    original = simple_spec("Hello world")
    renamed = {
        "root": "renamed_root",
        "state": {},
        "elements": {
            "renamed_body": copy.deepcopy(original["elements"]["body"]),
            "renamed_root": {
                **copy.deepcopy(original["elements"]["root"]),
                "children": ["renamed_body"],
            },
        },
    }
    first = score(original, "Hello world")
    second = score(renamed, "Hello world")
    assert abs(first.quality_0_1 - second.quality_0_1) < 1e-9


def test_unreachable_and_reachable_empty_padding_cannot_raise_score() -> None:
    base = simple_spec("Hello world")
    unreachable = copy.deepcopy(base)
    unreachable["elements"].update(
        {
            f"unused_{index}": {
                "type": "Text",
                "props": {"text": ""},
                "children": [],
            }
            for index in range(50)
        }
    )
    reachable = copy.deepcopy(base)
    for index in range(50):
        element_id = f"padding_{index}"
        reachable["elements"][element_id] = {
            "type": "Text",
            "props": {"text": ""},
            "children": [],
        }
        reachable["elements"]["root"]["children"].append(element_id)
    baseline = score(base, "Hello world").quality_0_1
    assert score(unreachable, "Hello world").quality_0_1 <= baseline
    assert score(reachable, "Hello world").quality_0_1 <= baseline


def test_wrong_action_target_lowers_fidelity_and_role_gate() -> None:
    source = "Open the report"
    expected = contract(
        source,
        actions=[
            {
                "id": "report",
                "required": True,
                "label": "Open report",
                "target": "https://example.com/report",
                "action_type": "openUrl",
            }
        ],
        source_links=[],
        expected_role_counts={"action": 1},
    )
    correct = score(
        action_spec(
            [
                (
                    "Open report",
                    "https://example.com/report",
                    "openUrl",
                )
            ]
        ),
        source,
        expected=expected,
    )
    wrong = score(
        action_spec(
            [
                (
                    "Open report",
                    "https://evil.example/wrong",
                    "openUrl",
                )
            ]
        ),
        source,
        expected=expected,
    )
    assert wrong.atomics["fidelity"][
        "action_and_source_link_fidelity"
    ] < correct.atomics["fidelity"][
        "action_and_source_link_fidelity"
    ]
    assert any(
        cap["name"] == "missing_action" for cap in wrong.active_caps
    )


def test_table_association_swap_lowers_full_score() -> None:
    source = "| ID | Value |\n|---|---|\n| a | red |\n| b | blue |"
    expected = contract(
        source,
        tables=[
            {
                "id": "values",
                "headers": ["ID", "Value"],
                "rows": [["a", "red"], ["b", "blue"]],
                "row_key": "ID",
                "required": True,
            }
        ],
    )

    def table(rows):
        return {
            "root": "table",
            "state": {},
            "elements": {
                "table": {
                    "type": "Table",
                    "props": {
                        "columns": [
                            {"key": "id", "label": "ID"},
                            {"key": "value", "label": "Value"},
                        ],
                        "rows": rows,
                    },
                    "children": [],
                }
            },
        }

    correct = score(
        table([{"id": "a", "value": "red"}, {"id": "b", "value": "blue"}]),
        source,
        expected=expected,
    )
    swapped = score(
        table([{"id": "a", "value": "blue"}, {"id": "b", "value": "red"}]),
        source,
        expected=expected,
    )
    assert swapped.atomics["fidelity"][
        "markdown_table_fidelity"
    ] < correct.atomics["fidelity"]["markdown_table_fidelity"]
    assert swapped.quality_0_1 < correct.quality_0_1


def test_content_order_reversal_lowers_order_atomic() -> None:
    source = "First connect the cable.\nThen start the device.\nFinally verify status."
    expected = contract(
        source,
        content_units=[
            "First connect the cable.",
            "Then start the device.",
            "Finally verify status.",
        ],
    )

    def blocks(values):
        return {
            "root": "root",
            "state": {},
            "elements": {
                "root": {
                    "type": "Stack",
                    "props": {},
                    "children": [f"text_{i}" for i in range(len(values))],
                },
                **{
                    f"text_{index}": {
                        "type": "Text",
                        "props": {"text": value},
                        "children": [],
                    }
                    for index, value in enumerate(values)
                },
            },
        }

    ordered = score(
        blocks(expected["content_units"]), source, expected=expected
    )
    reversed_result = score(
        blocks(list(reversed(expected["content_units"]))),
        source,
        expected=expected,
    )
    assert reversed_result.atomics["fidelity"][
        "content_order_preservation"
    ] < ordered.atomics["fidelity"]["content_order_preservation"]
