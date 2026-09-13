"""Tests that distinguish provable recovery from guessed or missing content."""

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.data.archive_refinement import (
    FamilyUnion,
    NearSourceIndex,
    choose_validation,
    numeric_anchors,
    repair_exact_text,
    review_warnings,
    source_signature,
)


def graph(text):
    return {
        "root": "root",
        "state": {},
        "elements": {"root": {"type": "Text", "props": {"text": text}}},
    }


def test_thousands_formatting_is_equivalent_without_changing_decimals_or_dates():
    assert numeric_anchors("$2,100 2024-03-02 1,23 10.5%") == {
        "2100",
        "2024-03-02",
        "1,23",
        "10.5%",
    }
    warnings, resolutions, _ = review_warnings(
        "Costs: 2100 3100 4100", "", graph("Costs: $2,100 $3,100 $4,100")
    )
    assert "half_numeric_anchors_missing" not in warnings
    assert "numeric_grouping_false_positive_resolved" in resolutions
    warnings, _, _ = review_warnings(
        "Costs: 2100 3100 4100", "", graph("Costs: $2,100 $3,100 $5,100")
    )
    assert "half_numeric_anchors_missing" in warnings


def test_literal_quote_unescape_requires_whole_value_in_source_and_excludes_code():
    value = 'This is a \\"quoted\\" message with adequate context.'
    clean = value.replace('\\"', '"')
    original = graph(value)
    fixed, metrics = repair_exact_text(clean, original)
    assert fixed["elements"]["root"]["props"]["text"] == clean
    assert metrics["source_proven_literal_quote_unescape"] == 1
    assert original["elements"]["root"]["props"]["text"] == value
    assert repair_exact_text("Unrelated source", original)[0] == original
    code = deepcopy(original)
    code["elements"]["root"]["type"] = "CodeBlock"
    assert repair_exact_text(clean, code)[0] == code


def test_clipped_word_completion_requires_unique_exact_source_prefix():
    prefix = "This source paragraph contains the same words and context. " * 4
    source = prefix + "responsibilities."
    source = source[:175].rstrip() + " " + "responsibilities."
    clipped = source[:180]
    assert len(clipped) == 180 and clipped[-1].isalnum()
    fixed, metrics = repair_exact_text(source, graph(clipped))
    assert fixed["elements"]["root"]["props"]["text"] == source.rstrip(".")
    assert metrics["source_proven_clipped_word_completion"] == 1
    assert not repair_exact_text(source + " " + source, graph(clipped))[1]


def test_existing_button_label_repair_never_creates_an_action_or_changes_a_grounded_label():
    source = "[Button: Track Delivery] [ACTION_URL_1]"
    value = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Button",
                "props": {"label": "Visit page"},
                "on": {
                    "press": {"action": "openUrl", "params": {"url": "[ACTION_URL_1]"}}
                },
            }
        },
    }
    fixed, changes = repair_exact_text(source, value)
    assert fixed["elements"]["root"]["props"]["label"] == "Track Delivery"
    assert fixed["elements"]["root"]["on"] == value["elements"]["root"]["on"]
    assert changes["source_proven_existing_button_label"] == 1
    assert not repair_exact_text(
        source + "\n[Button: Another choice] [ACTION_URL_1]", value
    )[1]
    assert not repair_exact_text(source + "\nVisit page", value)[1]
    del value["elements"]["root"]["on"]
    assert not repair_exact_text(source, value)[1]


def test_label_in_state_or_dead_button_does_not_prove_a_rendered_action():
    source = "[Button: Track Delivery] [ACTION_URL_1]"
    value = graph("Track Delivery")
    value["state"] = {"label": "Track Delivery", "actionUrl": "[ACTION_URL_1]"}
    assert (
        "requested_action_binding_not_rendered" in review_warnings(source, "", value)[0]
    )
    value["elements"]["root"] = {"type": "Button", "props": {"label": "Track Delivery"}}
    assert (
        "requested_action_binding_not_rendered" in review_warnings(source, "", value)[0]
    )


def test_reference_namespace_and_codec_changes_cannot_disguise_identical_source():
    original = "Café details: https://example.org/article"
    damaged = original.encode("utf-8").decode("cp437")
    assert source_signature(original) == source_signature(damaged)
    assert source_signature(original) == source_signature(
        "CAFÉ details: [ACTION_URL_8]"
    )


def test_family_union_is_transitive_and_split_is_order_independent():
    families = FamilyUnion()
    assert families.union("b", "c") and families.union("a", "b")
    assert families.find("c") == "a"
    assert not families.union("c", "b")
    groups = {str(i): ("Table", i * 100) for i in range(100)}
    assert choose_validation(groups, 0.02, 123) == choose_validation(
        dict(reversed(list(groups.items()))), 0.02, 123
    )
    assert choose_validation(groups, 0.02, 123)


def test_near_source_gate_uses_full_overlap_after_anchor_retrieval():
    source = " ".join(f"token{i}" for i in range(120))
    index = NearSourceIndex({"golden": source})
    assert list(index.matches(source + " additional minor detail"))
    assert not list(index.matches(" ".join(f"unrelated{i}" for i in range(120))))


def test_empty_tab_content_is_rejected_but_data_backed_content_is_not_empty():
    value = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Tabs",
                "props": {"tabs": [{"title": "Instructions", "child": "panel"}]},
            },
            "panel": {"type": "Column", "props": {}, "children": []},
        },
    }
    assert "empty_interactive_panel" in review_warnings("Instructions", "", value)[0]
    value["elements"]["panel"]["props"]["items"] = ["Actual instructions"]
    assert (
        "empty_interactive_panel" not in review_warnings("Instructions", "", value)[0]
    )
