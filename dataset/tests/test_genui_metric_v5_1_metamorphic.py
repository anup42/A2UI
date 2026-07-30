from __future__ import annotations

import copy
from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_1_support import contract, score, simple_spec  # noqa: E402
from pipeline.genui_quality import load_v5_1_reward_config  # noqa: E402
from pipeline.genui_quality.evidence_v5_1 import (  # noqa: E402
    collect_output_evidence_v5_1,
)
from pipeline.genui_quality.graph import audit_renderer_graph  # noqa: E402
from pipeline.genui_quality.structure_v5_1 import (  # noqa: E402
    canonical_reachable_payload_v5_1,
    heading_grouping_utility_v5_1,
    renderer_parent_map,
)


def test_id_renaming_and_element_map_order_are_score_invariant() -> None:
    source = "Hello world"
    original = simple_spec(source)
    renamed = {
        "root": "z_root",
        "state": {},
        "elements": {
            "z_body": copy.deepcopy(original["elements"]["body"]),
            "z_root": {
                **copy.deepcopy(original["elements"]["root"]),
                "children": ["z_body"],
            },
        },
    }
    left = score(original, source)
    right = score(renamed, source)
    assert abs(left.quality_0_1 - right.quality_0_1) < 1e-9


def test_order_reversal_lowers_content_order_fidelity() -> None:
    source = "First relation alpha\nSecond relation beta\nThird relation gamma"
    expected = contract(
        source,
        content_units=[
            "First relation alpha",
            "Second relation beta",
            "Third relation gamma",
        ],
    )
    correct = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {},
                "children": ["a", "b", "c"],
            },
            "a": {"type": "Text", "props": {"text": "First relation alpha"}, "children": []},
            "b": {"type": "Text", "props": {"text": "Second relation beta"}, "children": []},
            "c": {"type": "Text", "props": {"text": "Third relation gamma"}, "children": []},
        },
    }
    reversed_spec = copy.deepcopy(correct)
    reversed_spec["elements"]["root"]["children"] = ["c", "b", "a"]
    ordered = score(correct, source, expected=expected)
    reversed_result = score(reversed_spec, source, expected=expected)
    assert (
        reversed_result.atomics["fidelity"]["content_order_preservation"]
        < ordered.atomics["fidelity"]["content_order_preservation"]
    )
    assert reversed_result.quality_0_1 < ordered.quality_0_1


def test_tabs_modal_references_participate_in_parent_grouping_once() -> None:
    spec = {
        "root": "tabs",
        "state": {},
        "elements": {
            "tabs": {
                "type": "Tabs",
                "props": {"tabs": [{"title": "A", "child": "panel"}]},
                "children": [],
            },
            "panel": {
                "type": "Modal",
                "props": {"trigger": "heading", "content": "body"},
                "children": [],
            },
            "heading": {
                "type": "Text",
                "props": {"text": "Heading", "variant": "h2"},
                "children": [],
            },
            "body": {
                "type": "Text",
                "props": {"text": "Body"},
                "children": [],
            },
        },
    }
    audit = audit_renderer_graph(spec)
    evidence = collect_output_evidence_v5_1(
        spec, audit, load_v5_1_reward_config()
    ).output
    parents = renderer_parent_map(evidence)
    canonical = canonical_reachable_payload_v5_1(spec, audit)
    assert parents["panel"] == ("tabs",)
    assert parents["heading"] == ("panel",)
    assert parents["body"] == ("panel",)
    assert heading_grouping_utility_v5_1(evidence) == 1.0
    assert len(canonical["elements"]) == len(spec["elements"])


def test_raw_component_padding_cannot_increase_score() -> None:
    source = "Concise source"
    original = score(simple_spec(source), source)
    padded = simple_spec(source)
    padded["elements"]["root"]["children"].append("wrapper")
    padded["elements"]["wrapper"] = {
        "type": "Stack",
        "props": {},
        "children": ["padding"],
    }
    padded["elements"]["padding"] = {
        "type": "Text",
        "props": {"text": "unrelated unrelated unrelated"},
        "children": [],
    }
    changed = score(padded, source)
    assert changed.quality_0_1 <= original.quality_0_1
