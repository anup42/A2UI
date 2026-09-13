import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.data.archive_boundary_review import repair_join_boundaries
from ir_training.data.archive_recovery import _join_midword_text_chunks


def original(left, right):
    return {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Column", "props": {}, "children": ["a", "b"]},
            "a": {"type": "Text", "props": {"text": left}},
            "b": {"type": "Text", "props": {"text": right}},
        },
    }


def test_source_punctuation_restored_at_false_midword_boundary():
    left = "Ingredients: " + "fresh vegetables and mushrooms " * 6
    left = left[:176] + " oil"
    right = "Prep: Roast the garlic wrapped in foil for thirty minutes."
    before = original(left, right)
    merged, _ = _join_midword_text_chunks(before)
    fixed, proofs, issues = repair_join_boundaries(
        left + ".\n- **Prep**:" + right[5:], merged, before
    )
    assert not issues and len(proofs) == 1
    assert fixed["elements"]["a"]["props"]["text"] == left + ". " + right
    assert not repair_join_boundaries(left + ". " + right, fixed, before)[1]
    assert merged["elements"]["a"]["props"]["text"] == left + right


def test_real_split_word_is_preserved():
    left = ("A source sentence with enough surrounding context " * 5)[:178] + "re"
    right = "sponsibilities include reviewing the entire source document."
    before = original(left, right)
    merged, _ = _join_midword_text_chunks(before)
    fixed, proofs, issues = repair_join_boundaries(left + right, merged, before)
    assert not issues and not proofs and fixed == merged


def test_ambiguous_or_absent_source_context_is_not_guessed():
    left = ("The original source has reliable surrounding context " * 5)[:180]
    left = left[:-1] + "x"
    right = "Instruction follows here with complete surrounding context."
    before = original(left, right)
    merged, _ = _join_midword_text_chunks(before)
    assert repair_join_boundaries("Unrelated source", merged, before)[2]
    assert repair_join_boundaries((left + ". " + right + "\n") * 2, merged, before)[2]


def test_short_final_chunk_can_use_unique_left_context():
    left = ("A source sentence with enough surrounding context " * 5)[:178] + "re"
    right = "quirements."
    before = original(left, right)
    merged, _ = _join_midword_text_chunks(before)
    fixed, proofs, issues = repair_join_boundaries(left + right, merged, before)
    assert not issues and not proofs and fixed == merged
