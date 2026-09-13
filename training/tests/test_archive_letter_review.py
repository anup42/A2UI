import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.data.archive_letter_review import letter_gaps


def graph(text):
    return {
        "root": "root",
        "state": {},
        "elements": {"root": {"type": "Text", "props": {"text": text}}},
    }


def test_short_missing_request_is_not_ignored():
    source = "Subject: Pipe repair\n\nThe leak is still present in my bathroom.\n\nPlease confirm an appointment by the end of this week."
    assert letter_gaps(
        source, graph("Pipe repair. The leak is still present in my bathroom.")
    )
    assert not letter_gaps(source, graph(source))


def test_letter_clause_cut_off_after_word_repair_is_rejected():
    source = "Subject: Formal notice\n\nFailure to comply may include further action and additional remedial measures."
    assert letter_gaps(source, graph("Formal notice. Failure to comply may include"))


def test_extra_ui_heading_does_not_break_complete_letter():
    source = "Dear Alex,\n\nPlease confirm the appointment date. Thank you for the quick response."
    assert not letter_gaps(
        source,
        graph(
            "Dear Alex, Overview: Please confirm the appointment date. Closing: Thank you for the quick response."
        ),
    )


def test_other_responses_are_not_subject_to_verbatim_letter_policy():
    assert not letter_gaps(
        "Compare three travel itineraries and suggest the best option.",
        graph("Travel comparison"),
    )
