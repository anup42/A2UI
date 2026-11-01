"""Reviewed 100-sample regressions; exact pilot-only layout/role certificates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline.genui_quality._v5_4 import prepare_source_context_v5_4
from pipeline.genui_quality.config_v5_4 import RewardConfigV54
from pipeline.genui_quality.pilot_layout_v2 import exact_fragments, prose_spelling
from pipeline.training_fidelity import _tokens_v2
from pipeline.v11_regeneration_guard import assess_candidate


def assess(source, body, *, pilot=True):
    return assess_candidate({"id": "pilot-layout-regression", "response_text": source},
                            "<a2ui>" + body + "</a2ui>", pilot_fidelity_v2=pilot)


def test_prose_emphasis_has_no_false_literal_or_added_prose_failure():
    source = "The *Kaufnebenkosten* are **mandatory**.\n*Limitation: availability is not verified.*"
    body = 'root=Column([a,b])\na=Text("The Kaufnebenkosten are mandatory.")\nb=Text("Limitation: availability is not verified.")'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("not verified", "verified"))["eligible"]
    assert not assess(source, body.replace("mandatory", "optional"))["eligible"]


@pytest.mark.parametrize("literal", ["a * b * c", "x_2", "____", "2**3", "~3", "x^2", "`*literal*`", "$x_2 * y$"])
def test_emphasis_normalization_preserves_math_code_and_blanks(literal):
    assert prose_spelling(literal) == literal


def test_emphasis_normalization_keeps_unbalanced_and_escaped_markers():
    assert prose_spelling(r"Keep \*word* and *unclosed") == r"Keep \*word* and *unclosed"
    assert _tokens_v2("~2 * 3") != _tokens_v2("2 3")


@pytest.mark.parametrize("source", [
    "- Hours: 08:00-20:00, open at 16:00",
    "- Pharmacy North, ARS 3400, 0.7 km, open 09:00-19:00",
    "3. In the Scratch Disks dialog, uncheck C:. Check D: and keep the order.",
])
def test_time_and_drive_colons_do_not_create_entity_fields(source):
    old = prepare_source_context_v5_4(source, config=RewardConfigV54(grpo_entity_extraction=True))
    new = prepare_source_context_v5_4(source, config=RewardConfigV54(grpo_entity_extraction=True, pilot_fidelity_v2=True))
    assert old.grpo_entity_records  # Reproduce the earlier false recognition.
    assert not new.grpo_entity_records
    assert new.base.applicability.generic_source_units  # Prose remains required.


def test_clock_prose_still_requires_unchanged_times():
    source = "- Hours: 08:00-20:00, open at 16:00"
    body = 'root=List(items=["Hours: 08:00-20:00, open at 16:00"])'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("16:00", "17:00"))["eligible"]


PROFILE = "**Profile used**\n- Age: 34 y, Sex: female\n- Height: 168 cm, Weight: 72 kg"
PROFILE_BODY = ('root=Card([age,sex,height,weight],title="Profile used")\n'
                'age=Text("Age: 34 y")\nsex=Text("Sex: female")\n'
                'height=Text("Height: 168 cm")\nweight=Text("Weight: 72 kg")')


def test_split_property_bundles_keep_exact_source_section_ownership():
    result = assess(PROFILE, PROFILE_BODY)
    assert result["eligible"], result
    assert result["entity_ownership"]["structured_record_count"] == 2


@pytest.mark.parametrize("old,new", [
    ("34 y", "35 y"), ("72 kg", "168 kg"), ("Profile used", "Other profile"),
    ("[age,sex,height,weight]", "[age,height,sex,weight]"),
    ("[age,sex,height,weight]", "[age,sex,height,weight,age]"),
    ('Sex: female', 'Sex: male'),
])
def test_split_property_bundles_reject_changed_moved_and_duplicate_fields(old, new):
    assert not assess(PROFILE, PROFILE_BODY.replace(old, new))["eligible"]


def test_property_fields_cannot_cross_cards():
    body = PROFILE_BODY.replace('root=Card([age,sex,height,weight],title="Profile used")',
                                'root=Column([one,two])\none=Card([age,height],title="Profile used")\ntwo=Card([sex,weight])')
    assert not assess(PROFILE, body)["eligible"]


def test_dated_pipe_record_preserves_every_ordered_field():
    source = "- Dec 2 (Mon): **Network Security** | Firewalls, VPNs, and Secure Tunneling | 2 Hours"
    body = ('root=Card([topics,duration],title="Dec 2 (Mon): Network Security")\n'
            'topics=Text("Firewalls, VPNs, and Secure Tunneling")\nduration=Text("2 Hours")')
    assert assess(source, body)["eligible"]
    for old, new in [("2 Hours", "3 Hours"), ("topics,duration", "duration,topics"),
                     ("topics,duration", "topics"), ("topics,duration", "topics,duration,duration")]:
        assert not assess(source, body.replace(old, new))["eligible"]

    split = body.replace('[topics,duration],title="Dec 2 (Mon): Network Security"',
                         '[subject,topics,duration],title="Dec 2 (Mon)"') + '\nsubject=Text("Network Security")'
    assert assess(source, split)["eligible"]
    assert not assess(source, split.replace("Network Security", "Cloud Security"))["eligible"]


def test_same_parent_comma_split_preserves_source_qualifiers():
    source = "Family roundtrip cost JPY 145,000, family one-way JPY 72,500"
    body = 'root=Column([a,b])\na=Text("Family roundtrip cost JPY 145,000")\nb=Text("family one-way JPY 72,500")'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("one-way", "roundtrip"))["eligible"]
    assert not assess(source, body.replace("[a,b]", "[b,a]"))["eligible"]


@pytest.mark.parametrize("source,parts", [
    ("x | y", ["x", "y"]), ("x = y", ["x", "y"]),
    ("2 * 3", ["2", "3"]), ("Price: ~50", ["Price", "50"]),
    ("Not allowed", ["Allowed"]), ("USD 12.50", ["USD 12", "50"]),
])
def test_layout_boundaries_cannot_remove_operators_or_digits(source, parts):
    assert not exact_fragments(source, parts)


def test_inline_source_table_comparison_and_currency_spelling():
    source = "| Document | Requirement |\n|---|---|\n| Passport | Valid for $\\ge 6$ months |\n| Funds | Recommended: \\$2,000+ in cash |"
    body = 'root=Table(columns=["Document","Requirement"],rows=[["Passport","Valid for ≥ 6 months"],["Funds","Recommended: $2,000+ in cash"]])'
    assert assess(source, body)["eligible"]
    for old,new in [("≥ 6", "> 6"),("≥ 6", "≥ 5"),("$2,000+", "$2,000"),("Recommended:", "Required:")]:
        assert not assess(source, body.replace(old,new))["eligible"]


def test_bare_labeled_source_links_keep_role_label_and_destination():
    url = "https://docs.python.org/3/?view=full#intro"
    source = f"Sources:\n- Python docs: {url}\n\nAction: [Read] <{url}>"
    body = (f'root=Column([sources,action])\nsources=Card([links],title="Sources")\n'
            f'links=List(items=[{{label:"Python docs",url:"{url}"}}])\n'
            f'action=Button("Read",onPress=openUrl(url="{url}"))')
    assert assess(source, body)["eligible"]
    bad = body.replace(f'List(items=[{{label:"Python docs",url:"{url}"}}])',
                       f'Button("Python docs",onPress=openUrl(url="{url}"))')
    # A source Card's native source-link route owns child buttons, too. Move it
    # outside the source Card to reproduce the pilot's actual role mutation.
    bad = bad.replace('root=Column([sources,action])', 'root=Column([links,action])')
    assert not assess(source, bad)["eligible"]
    assert not assess(source, body.replace('#intro', '#other'))["eligible"]
    assert not assess(source, body.replace('label:"Python docs"', 'label:"New docs"'))["eligible"]


def test_card_title_can_own_a_source_heading_without_becoming_extra_prose():
    source = "## Budget 2026\nKeep at least €400 for emergencies."
    body = 'root=Card([note],title="Budget 2026")\nnote=Text("Keep at least €400 for emergencies.")'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("Budget 2026", "Budget 2027"))["eligible"]
    assert not assess(source, body.replace("[note]", "[note,duplicate]") + '\nduplicate=Text("Budget 2026")')["eligible"]


def test_full_plain_pipe_table_allows_only_empty_omitted_trailing_cells():
    source = "Time | Activity | AV | Catering\n10:00 | Setup | Mic |\n11:00 | Break |\n12:00 | Lunch | Music | Meals"
    body = 'root=Table(columns=["Time","Activity","AV","Catering"],rows=[["10:00","Setup","Mic",""],["11:00","Break","",""],["12:00","Lunch","Music","Meals"]])'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace('"Mic",""', '"Mic","Meals"'))["eligible"]
    assert not assess(source, body.replace(',["12:00","Lunch","Music","Meals"]', ''))["eligible"]


def test_markdown_bordered_table_header_and_cells_remain_required():
    source = "| Shop | Price |\n|---|---|\n| North | $20 |\n| South | $30 |"
    body = 'root=Table(columns=["Shop","Price"],rows=[["North","$20"],["South","$30"]])'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace('"North","$20"', '"North","$30"'))["eligible"]
    duplicated = 'root=Column([a,b])\n' + body.replace('root=', 'a=') + '\n' + body.replace('root=', 'b=')
    assert not assess(source, duplicated)["eligible"]


def test_numbered_equations_in_table_keep_identity_operator_and_multiplicity():
    source = "Year 1: 2,800 ×12 = 33,600\nYear 2: 2,884 ×12 ≈ 34,608"
    body = 'root=Table(columns=["Year","Calculation"],rows=[["Year 1","2,800 ×12 = 33,600"],["Year 2","2,884 ×12 ≈ 34,608"]])'
    assert assess(source, body)["eligible"]
    for old,new in [("≈", "="),("2,884", "2,883"),("Year 2", "Year 1"),
                    ("Calculation", "Guaranteed return"),("Year 1", "Year 3")]:
        assert not assess(source, body.replace(old,new))["eligible"]


def test_all_rows_of_bordered_table_receive_exact_literal_credit():
    source = "| Course | Live sessions |\n| A | Tue/Thu 19:00-21:00 SGT |\n| B | Sat 10:00-12:00 SGT |"
    body = 'root=Table(columns=["Course","Live sessions"],rows=[["A","Tue/Thu 19:00-21:00 SGT"],["B","Sat 10:00-12:00 SGT"]])'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace('Sat 10:00-12:00 SGT', 'Sat 10:00-11:00 SGT'))["eligible"]


def test_inline_code_currency_and_math_remain_literal():
    from pipeline.genui_quality.pilot_layout_v2 import table_cell_spelling

    assert table_cell_spelling(r"`\$HOME` and \$30") == r"`\$HOME` and $30"
    assert table_cell_spelling(r"`$\ge 6$`") == r"`$\ge 6$`"


def test_option_pipe_card_may_split_identifier_and_subject_only_in_order():
    source = "Option 1: Hotel X | Superior accessibility | 2-min station access\nOption 2: Hotel Y | Quiet rooms | 7-min station access"
    body = ('root=Column([a,b])\na=Card([an,ax,ay],title="Option 1")\nan=Text("Hotel X")\n'
            'ax=Text("Superior accessibility")\nay=Text("2-min station access")\n'
            'b=Card([bn,bx,by],title="Option 2")\nbn=Text("Hotel Y")\n'
            'bx=Text("Quiet rooms")\nby=Text("7-min station access")')
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace('2-min station access', '7-min station access'))["eligible"]


@pytest.mark.parametrize("formula", [
    r"T = \sum_{i=1}^{n} (B_i \times R_i)",
    r"M = P \frac{i(1+i)^n}{(1+i)^n - 1}",
    r"\text{Price per Ounce} = \frac{\text{Total Price}}{\text{Total Ounces}}",
])
def test_explicit_formula_title_and_rendered_spelling_are_source_owned(formula):
    source = "Formula: $" + formula + "$"
    body = 'root=Formula(' + json.dumps(formula) + ',title="Formula")'
    assert assess(source, body)["eligible"]
    assert not assess(source, body, pilot=False)["eligible"]
    assert not assess(source, body.replace('title="Formula"', 'title="Guaranteed result"'))["eligible"]
    assert not assess(source, body.replace('title="Formula"', 'title="Formula",result="42"'))["eligible"]
    assert not assess(source, body.replace(' = ', ' > '))["eligible"]


def test_formula_certificate_does_not_change_operand_script_or_unit():
    formula = r"FV = PMT \times \frac{(1 + r)^n - 1}{r}"
    source = "Formula: $" + formula + "$"
    for bad in (formula.replace("^n", "^2"), formula.replace("- 1", "+ 1"),
                formula.replace("{r}", "{PMT}"), formula + r"\text{ USD}"):
        assert not assess(source, 'root=Formula(' + json.dumps(bad) + ',title="Formula")')["eligible"]
