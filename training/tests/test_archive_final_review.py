import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.data.archive_final_review import (
    paragraph_gaps,
    repair_missing_button_labels,
)


def button(label):
    return {
        "type": "Button",
        "props": {"label": label},
        "on": {"press": {"action": "openUrl", "params": {"url": "[ACTION_URL_1]"}}},
    }


def test_secondary_button_is_not_renamed_when_requested_action_exists():
    graph = {
        "root": "root",
        "state": {},
        "elements": {"root": button("Required action"), "citation": button("Acronym")},
    }
    original = deepcopy(graph)
    fixed, changes = repair_missing_button_labels(
        "[Button: Required action] [ACTION_URL_1]", graph
    )
    assert fixed == original and not changes
    assert graph == original


def test_only_unique_existing_button_can_restore_missing_label():
    graph = {"root": "root", "state": {}, "elements": {"root": button("Visit website")}}
    fixed, changes = repair_missing_button_labels(
        "[Button: Required action] [ACTION_URL_1]", graph
    )
    assert fixed["elements"]["root"]["props"]["label"] == "Required action"
    assert changes == {"source_proven_missing_button_label": 1}
    graph["elements"]["other"] = button("Another link")
    assert not repair_missing_button_labels(
        "[Button: Required action] [ACTION_URL_1]", graph
    )[1]


def test_missing_letter_paragraph_is_not_hidden_by_overall_word_recall():
    first = "The persistent roof leak remains unresolved and requires immediate attention because water damage affects our living space."
    missing = "Please dispatch a licensed contractor within twenty four hours, confirm the appointment tomorrow morning, and supply a written maintenance schedule documenting the necessary repairs."
    graph = {
        "root": "root",
        "state": {"unused": missing},
        "elements": {"root": {"type": "Text", "props": {"text": first}}},
    }
    assert paragraph_gaps(first + "\n\n" + missing, graph)
    graph["elements"]["root"]["props"]["text"] = first + " " + missing
    assert not paragraph_gaps(first + "\n\n" + missing, graph)


def test_bound_table_prose_is_counted_without_counting_unused_state():
    text = "This complete explanation covers transportation arrangements, hotel reservations, itinerary changes, dietary preferences, accessibility assistance, travel insurance, essential documentation and emergency support contacts."
    graph = {
        "root": "root",
        "state": {"rows": [{"details": text}]},
        "elements": {
            "root": {
                "type": "Table",
                "props": {
                    "statePath": "/rows",
                    "columns": [{"key": "details", "label": "Details"}],
                },
            }
        },
    }
    assert not paragraph_gaps(text, graph)
