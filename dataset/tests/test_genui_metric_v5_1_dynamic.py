from __future__ import annotations

from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_1_support import contract, score  # noqa: E402
from pipeline.genui_quality import load_v5_1_reward_config  # noqa: E402
from pipeline.genui_quality.evidence_v5_1 import (  # noqa: E402
    DynamicEvidenceResolver,
)


def _resolver(**kwargs):
    state = {
        "tab": "hotels",
        "enabled": True,
        "count": 3,
        "user": {"name": "Ada"},
        "items": [{"name": "Alpha"}, {"name": "Beta"}],
    }
    return DynamicEvidenceResolver(
        state,
        load_v5_1_reward_config(),
        kwargs.get("computed_functions"),
    )


def test_dynamic_values_bindings_condition_literals_and_templates() -> None:
    resolver = _resolver(computed_functions={"join": lambda args: args["a"] + args["b"]})
    item = {"name": "Beta", "status": "shown"}
    assert resolver.resolve({"$state": "/user/name"}) == "Ada"
    assert resolver.resolve({"$bindState": "/user/name"}) == "Ada"
    assert resolver.resolve({"$item": "name"}, item=item) == "Beta"
    assert resolver.resolve(
        {"$bindItem": "name"}, item=item, base_path="/items/1"
    ) == "Beta"
    assert resolver.resolve({"$index": True}, index=1) == 1
    assert resolver.resolve(
        {
            "$cond": {"$state": "/tab", "eq": "hotels"},
            "$then": "Hotel panel",
            "$else": "Flight panel",
        }
    ) == "Hotel panel"
    assert resolver.resolve({"literalString": "text"}) == "text"
    assert resolver.resolve({"literalNumber": 7}) == 7
    assert resolver.resolve({"literalBoolean": True}) is True
    assert resolver.resolve(
        {"$computed": "join", "args": {"a": "A", "b": "B"}}
    ) == "AB"
    assert resolver.resolve(
        {"$template": "${$item.name} ${index}/${index_0}/${index_1} ${/user/name}"},
        item=item,
        index=1,
    ) == "Beta 1/1/2 Ada"


def test_visibility_comparison_parity() -> None:
    resolver = _resolver()
    item = {"status": "shown", "value": 4}
    assert resolver.evaluate_condition({"$state": "/tab", "eq": "hotels"})
    assert not resolver.evaluate_condition({"$state": "/tab", "eq": "flights"})
    assert resolver.evaluate_condition({"$item": "status", "neq": "hidden"}, item=item)
    assert resolver.evaluate_condition({"$index": True, "gte": 2}, index=2)
    assert resolver.evaluate_condition({"value": 4, "gt": 3})
    assert resolver.evaluate_condition({"value": 4, "gte": 4})
    assert resolver.evaluate_condition({"value": 4, "lt": 5})
    assert resolver.evaluate_condition({"value": 4, "lte": 4})
    assert resolver.evaluate_condition({"$state": "/enabled", "not": True}) is False
    assert resolver.evaluate_condition(
        {"$and": [{"value": 4, "gt": 3}, {"value": 4, "lt": 5}]}
    )
    assert resolver.evaluate_condition(
        {"$or": [{"value": 4, "lt": 0}, {"value": 4, "eq": 4}]}
    )


def test_unknown_computed_is_explicit() -> None:
    resolver = _resolver()
    value = resolver.resolve({"$computed": "network_lookup", "args": {}})
    assert value is not None
    assert resolver.unknown == ["computed_function_unknown:network_lookup"]


def test_repeat_conditional_and_static_equivalents_are_close() -> None:
    source = "Alpha 1\nBeta 2"
    expected = contract(source, content_units=["Alpha 1", "Beta 2"])
    repeated = {
        "root": "root",
        "state": {
            "items": [
                {"name": "Alpha", "visible": True},
                {"name": "Beta", "visible": True},
            ]
        },
        "elements": {
            "root": {
                "type": "Stack",
                "props": {},
                "children": ["repeat"],
            },
            "repeat": {
                "type": "Column",
                "props": {},
                "children": ["row"],
                "repeat": {"statePath": "/items"},
            },
            "row": {
                "type": "Text",
                "visible": {"$item": "visible", "eq": True},
                "props": {
                    "text": {
                        "$cond": {"$item": "visible", "eq": True},
                        "$then": {"$template": "${$item.name} ${index_1}"},
                        "$else": "hidden",
                    }
                },
                "children": [],
            },
        },
    }
    static = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {},
                "children": ["a", "b"],
            },
            "a": {"type": "Text", "props": {"text": "Alpha 1"}, "children": []},
            "b": {"type": "Text", "props": {"text": "Beta 2"}, "children": []},
        },
    }
    repeat_result = score(repeated, source, expected=expected)
    static_result = score(static, source, expected=expected)
    assert repeat_result.evidence["dynamic_evidence_complete"]
    assert (
        repeat_result.atomics["fidelity"]["content_unit_fidelity"]
        == static_result.atomics["fidelity"]["content_unit_fidelity"]
        == 1.0
    )
    assert abs(
        repeat_result.quality_0_100 - static_result.quality_0_100
    ) <= 1.0
