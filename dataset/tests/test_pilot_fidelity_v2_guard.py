"""Pilot-only strict acceptance; unchanged thresholds and legacy defaults."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline import v11_regeneration_guard as guard
from pipeline.genui_quality.matching_v5_1 import prepare_text_block
from pipeline.genui_quality.pilot_content_v2 import (
    project_exact_pipe_cards,
    split_exact_multiline,
)


def assess(source, body, **kwargs):
    return guard.assess_candidate({"id": "pilot-regression", "response_text": source},
                                  "<a2ui>" + body + "</a2ui>", **kwargs)


def test_exact_alert_is_eligible_but_default_behavior_is_unchanged():
    source = "The appointment is at 14:00."
    body = "root=Alert(message=" + json.dumps(source) + ")"
    old = assess(source, body)
    assert old == assess(source, body, pilot_fidelity_v2=False)
    assert not old["eligible"]
    fixed = assess(source, body, pilot_fidelity_v2=True)
    assert fixed["eligible"], fixed
    assert fixed["canonical_policy"]["thresholds"] == "unchanged"


@pytest.mark.parametrize("extra", ["Your appointment is free and includes lunch.",
                                  "The appointment is at 15:00."])
def test_alert_cannot_hide_invented_claim_or_changed_number(extra):
    source = "The appointment is at 14:00."
    body = 'root=Column([a,b])\na=Text(' + json.dumps(source) + ')\nb=Alert(message=' + json.dumps(extra) + ')'
    result = assess(source, body, pilot_fidelity_v2=True)
    assert not result["eligible"]
    assert "unsupported_visible_prose" in result["reasons"]


def test_hidden_or_negation_changed_alert_does_not_replace_source():
    source = "Breakfast is not included."
    for body in ['root=Alert(message="Breakfast is included.")',
                 'root=Alert(message="Breakfast is not included.",visible=false)']:
        assert not assess(source, body, pilot_fidelity_v2=True)["eligible"]


def test_source_policy_leakage_remains_rejected():
    source = "The appointment is at 14:00."
    body = 'root=Column([a,b])\na=Text(' + json.dumps(source) + ')\nb=Text("Asset reference policy for this request: never invent URLs.")'
    result = assess(source, body, pilot_fidelity_v2=True)
    assert not result["eligible"]
    assert "unsupported_visible_prose" in result["reasons"]


def test_entity_values_and_qualifiers_must_remain_with_their_owner():
    source = "- Alder; Price: $120; Breakfast: not included\n- Birch; Price: $80; Breakfast: included"
    body = '''root=Column([a,b])
a=Card([ap,ab],title="Alder")
ap=Text("Price: $120")
ab=Text("Breakfast: not included")
b=Card([bp,bb],title="Birch")
bp=Text("Price: $80")
bb=Text("Breakfast: included")'''
    assert assess(source, body, pilot_fidelity_v2=True)["eligible"]
    for old, new in [("$120", "$80"), ("not included", "included")]:
        result = assess(source, body.replace(old, new), pilot_fidelity_v2=True)
        assert not result["eligible"]
        assert "entity_binding_incomplete" in result["reasons"]


def test_multiline_partition_requires_whole_ordered_literal_units():
    source = tuple(map(prepare_text_block, ["Fare: $80.", "Breakfast: not included."]))
    whole = prepare_text_block("Fare: $80.\nBreakfast: not included.")
    blocks, proof = split_exact_multiline(source, [whole])
    assert blocks == source
    assert proof[0]["ordered_literal_tokens_exact"]
    for text in ["Fare: $800.\nBreakfast: not included.",
                 "Fare: $80.\nBreakfast: included.",
                 "Fare: $80.\nBreakfast: included not.",
                 "Fare: $80.\nBreakfast: not included. Free lunch."]:
        original = prepare_text_block(text)
        blocks, proof = split_exact_multiline(source, [original])
        assert blocks == (original,)
        assert proof == []


def test_exact_partition_does_not_swallow_duplicate_occurrences():
    source = tuple(map(prepare_text_block, ["First fact.", "Second fact."]))
    original = prepare_text_block("First fact.\nSecond fact.\nFirst fact.")
    blocks, proof = split_exact_multiline(source, [original])
    assert blocks == (original,)  # Matching must still see the extra occurrence.
    assert proof == []


def test_ambiguous_multiline_partition_gets_no_certificate():
    source = tuple(map(prepare_text_block, ["Alpha", "Beta", "Alpha Beta", "Gamma"]))
    block = prepare_text_block("Alpha\nBeta\nGamma")
    assert split_exact_multiline(source, [block]) == ((block,), [])


@pytest.mark.parametrize("symbol", ["≈", "≠", "×", "÷", "∑", "^", "_"])
def test_multiline_certificate_never_discards_mathematical_symbols(symbol):
    source = tuple(map(prepare_text_block, [f"Gain {symbol} 7", "Loss 2"]))
    block = prepare_text_block("Gain 7\nLoss 2")
    assert split_exact_multiline(source, [block]) == ((block,), [])


@pytest.mark.parametrize("props", ['latex="The moon is made of cheese."',
                                  'latex="x=2",title="The moon is made of cheese."',
                                  'latex="x=2",subtitle="The moon is made of cheese."',
                                  'latex="x=2",result="The moon is made of cheese."'])
def test_formula_cannot_be_an_unchecked_prose_channel(props):
    source = "Keep the door closed.\n\\(x=2\\)"
    body = 'root=Column([a,b])\na=Text(' + json.dumps(source) + ')\nb=Formula(' + props + ')'
    result = assess(source, body, pilot_fidelity_v2=True)
    assert not result["eligible"]
    assert "unsupported_visible_prose" in result["reasons"]


def test_formula_cannot_drop_an_approximation_or_change_a_script():
    for source, formula in [(r"\(x \approx 2\)", "x=2"), (r"\(x^2\)", "x_2")]:
        body = "root=Formula(latex=" + json.dumps(formula) + ")"
        result = assess(source, body, pilot_fidelity_v2=True)
        assert not result["eligible"]
        assert "unsupported_visible_prose" in result["reasons"]


def test_exact_headings_and_multiline_prose_are_not_double_counted():
    source = "## Travel notes\nThe first note.\nThe second note.\n## Reminder\nKeep your ID."
    body = 'root=Column([h,p,r,t])\nh=Text("Travel notes","h2")\np=Text("The first note.\\nThe second note.")\nr=Text("Reminder","h2")\nt=Text("Keep your ID.")'
    assert assess(source, body, pilot_fidelity_v2=True)["eligible"]
    for bad in [body.replace('"Travel notes"', '"Free travel"'),
                body.replace('root=Column([h,p,r,t])', 'root=Column([h,h,p,r,t])'),
                body.replace('root=Column([h,p,r,t])', 'root=Column([p,r,t])')]:
        assert not assess(source, bad, pilot_fidelity_v2=True)["eligible"]


def test_pipe_fields_require_whole_contiguous_fields_in_one_card():
    source = [prepare_text_block("Name: Alpha | Price: $80 | Breakfast: not included")]
    rows = [{"text": text, "owner": "generic_visible_content", "component_type": kind,
             "card_instance": "card:a", "card_ancestors": ["card:a"]}
            for text, kind in [("Name: Alpha", "card"), ("Price: $80", "text"),
                               ("Breakfast: not included", "text")]]
    grouped, proof = project_exact_pipe_cards(source, rows, {"generic_visible_content"})
    assert len(grouped) == 1 and grouped[0]["text"] == source[0].raw and proof
    for bad in [rows[:1] + rows[1:][::-1],
                [rows[0], {**rows[1], "card_instance": "card:b"}, rows[2]],
                [rows[0], rows[1], {**rows[2], "text": "Breakfast: included"}],
                [rows[0], {**rows[1], "text": "Price: $800"}, rows[2]]]:
        grouped, proof = project_exact_pipe_cards(source, bad, {"generic_visible_content"})
        assert grouped == bad and proof == []


@pytest.mark.parametrize("source,fields", [
    ("Regex: foo|bar", ["Regex: foo", "bar"]),
    ("Permission mask: READ | WRITE", ["Permission mask: READ", "WRITE"]),
    ("Formula: x | y: 2", ["Formula: x", "y: 2"]),
])
def test_semantic_pipe_is_not_a_card_field_separator(source, fields):
    rows = [{"text": text, "owner": "generic_visible_content", "component_type": "text",
             "card_instance": "card:a", "card_ancestors": ["card:a"]} for text in fields]
    grouped, proof = project_exact_pipe_cards([prepare_text_block(source)], rows, {"generic_visible_content"})
    assert grouped == rows and proof == []
    body = 'root=Card([a,b])\na=Text(' + json.dumps(fields[0]) + ')\nb=Text(' + json.dumps(fields[1]) + ')'
    assert not assess(source, body, pilot_fidelity_v2=True)["eligible"]


def test_profile_flag_is_strict_boolean():
    with pytest.raises(ValueError, match="boolean"):
        assess("Fact.", 'root=Text("Fact.")', pilot_fidelity_v2="true")
