from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_support import contract, score, simple_spec  # noqa: E402


def test_order_reversal_lowers_order_sensitive_content() -> None:
    source = "First alpha statement.\nSecond beta statement."
    expected = contract(source)
    candidate = simple_spec("")
    candidate["elements"].pop("body")
    candidate["elements"]["root"]["children"] = ["first", "second"]
    candidate["elements"]["first"] = {
        "type": "Text",
        "props": {"text": "First alpha statement."},
        "children": [],
    }
    candidate["elements"]["second"] = {
        "type": "Text",
        "props": {"text": "Second beta statement."},
        "children": [],
    }
    original = score(candidate, source, expected=expected)
    reversed_candidate = copy.deepcopy(candidate)
    reversed_candidate["elements"]["root"]["children"].reverse()
    changed = score(reversed_candidate, source, expected=expected)
    assert (
        changed.atomics["fidelity"]["content_order_preservation"]
        < original.atomics["fidelity"]["content_order_preservation"]
    )
    assert changed.quality_0_1 < original.quality_0_1


def test_repeat_and_static_equivalent_are_close() -> None:
    source = "Alpha 1\nBeta 2"
    expected = contract(source)
    repeated = {
        "root": "root",
        "state": {
            "items": [
                {"name": "Alpha", "value": 1},
                {"name": "Beta", "value": 2},
            ]
        },
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["repeat"]},
            "repeat": {
                "type": "Column",
                "props": {},
                "children": ["row"],
                "repeat": {"statePath": "/items"},
            },
            "row": {
                "type": "Text",
                "props": {
                    "text": {"$template": "${$item.name} ${$item.value}"}
                },
                "children": [],
            },
        },
    }
    static = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["a", "b"]},
            "a": {"type": "Text", "props": {"text": "Alpha 1"}, "children": []},
            "b": {"type": "Text", "props": {"text": "Beta 2"}, "children": []},
        },
    }
    repeated_score = score(repeated, source, expected=expected)
    static_score = score(static, source, expected=expected)
    assert not repeated_score.evidence["output"][
        "dynamic_expression_unknown_codes"
    ]
    assert abs(repeated_score.quality_0_1 - static_score.quality_0_1) <= 0.03


def test_unreachable_and_reachable_padding_cannot_increase() -> None:
    source = "Hello world"
    original_spec = simple_spec(source)
    original = score(original_spec, source)

    unreachable = copy.deepcopy(original_spec)
    for index in range(50):
        unreachable["elements"][f"unused_{index}"] = {
            "type": "Card",
            "props": {},
            "children": [],
        }
    unreachable_score = score(unreachable, source)
    assert unreachable_score.quality_0_1 <= original.quality_0_1

    reachable = copy.deepcopy(original_spec)
    for index in range(20):
        element_id = f"divider_{index}"
        reachable["elements"][element_id] = {
            "type": "Divider",
            "props": {},
            "children": [],
        }
        reachable["elements"]["root"]["children"].append(element_id)
    reachable_score = score(reachable, source)
    assert reachable_score.quality_0_1 <= original.quality_0_1


def test_identifier_renaming_and_element_map_order_are_invariant() -> None:
    source = "Hello world"
    candidate = simple_spec(source)
    mapping = {old: f"id_{index}" for index, old in enumerate(candidate["elements"])}
    renamed = copy.deepcopy(candidate)
    renamed["root"] = mapping[candidate["root"]]
    renamed["elements"] = OrderedDict(
        reversed(
            [
                (
                    mapping[old],
                    {
                        **copy.deepcopy(element),
                        "children": [
                            mapping.get(child, child)
                            for child in element.get("children", [])
                        ],
                    },
                )
                for old, element in candidate["elements"].items()
            ]
        )
    )
    assert score(candidate, source).quality_0_1 == score(renamed, source).quality_0_1


def test_text_fragmentation_and_component_count_do_not_raise_score() -> None:
    source = "A clear paragraph with enough words to remain one coherent block."
    original = simple_spec(source)
    fragmented = simple_spec("")
    fragmented["elements"].pop("body")
    fragmented["elements"]["root"]["children"] = []
    for index, word in enumerate(source.split()):
        element_id = f"word_{index}"
        fragmented["elements"][element_id] = {
            "type": "Text",
            "props": {"text": word},
            "children": [],
        }
        fragmented["elements"]["root"]["children"].append(element_id)
    assert score(fragmented, source).quality_0_1 <= score(original, source).quality_0_1


def test_distinct_chart_signatures_cover_distinct_requirements() -> None:
    source = "Two distinct charts"
    expected = contract(
        source,
        required_roles={"chart": 2},
        expected_role_counts={"chart": 2},
    )
    candidate = simple_spec(source)
    candidate["state"] = {
        "first": [{"x": "A", "y": 1}],
        "second": [{"x": "B", "y": 2}],
    }
    for index, path in enumerate(("/first", "/second")):
        candidate["elements"][f"chart_{index}"] = {
            "type": "Chart",
            "props": {
                "title": f"Chart {index + 1}",
                "columns": [
                    {"key": "x", "label": "X"},
                    {"key": "y", "label": "Y"},
                ],
                "statePath": path,
                "xKey": "x",
                "yKey": "y",
            },
            "children": [],
        }
        candidate["elements"]["root"]["children"].append(f"chart_{index}")
    result = score(candidate, source, expected=expected)
    assert result.evidence["role_values"]["chart"] == 1.0
