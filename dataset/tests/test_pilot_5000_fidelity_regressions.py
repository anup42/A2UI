"""Exact renderer evidence for reviewed failures in the 5,000-output pilot."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline.genui_quality._v5_4 import prepare_source_context_v5_4
from pipeline.genui_quality.config_v5_4 import RewardConfigV54
from pipeline.genui_quality.formula_evidence_v2 import formula_literal_key_v2
from pipeline.genui_quality.pilot_source_contract_v2 import declared_media_occurrences
from pipeline.reference_boundaries import reference_destination_errors
from pipeline.stage3_genui import _mask_model_references, _restore_model_references
from pipeline.v11_regeneration_guard import assess_candidate


def assess(source, body, *, pilot=True):
    return assess_candidate({"id": "pilot-5000-regression", "response_text": source},
                            f"<a2ui>\n{body}\n</a2ui>", pilot_fidelity_v2=pilot)


BUDGET = "## Expenses\n* Mortgage CAD 2,650\n* Food: CAD 300, optional on weekdays"
BUDGET_TABLE = ('root=Column([heading,budget])\nheading=Text("Expenses","h2")\n'
                'budget=Table(columns=["Expense","Amount"],'
                'rows=[["Mortgage","CAD 2,650"],["Food","CAD 300, optional on weekdays"]])')


def test_cost_table_credits_exact_ordered_records_under_their_heading():
    result = assess(BUDGET, BUDGET_TABLE)
    assert result["eligible"], result["reasons"]
    assert len(result["literal_missing_units_certified_by_pilot_fields"]) == 2
    assert not assess(BUDGET, BUDGET_TABLE, pilot=False)["eligible"]


@pytest.mark.parametrize(("before", "after"), [
    ('"CAD 2,650"', '"CAD 2,651"'),
    ('"CAD 300, optional on weekdays"', '"CAD 300"'),
    ('["Mortgage","CAD 2,650"],["Food","CAD 300, optional on weekdays"]',
     '["Food","CAD 300, optional on weekdays"],["Mortgage","CAD 2,650"]'),
    ('["Mortgage","CAD 2,650"],["Food","CAD 300, optional on weekdays"]',
     '["Mortgage","CAD 300, optional on weekdays"],["Food","CAD 2,650"]'),
    ('[heading,budget]', '[budget,heading]'),
    ('[heading,budget]', '[heading,budget,budget]'),
    ('["Expense","Amount"]', '["Required item","Total"]'),
])
def test_cost_table_rejects_mutated_rows_qualifiers_owners_and_counts(before, after):
    assert not assess(BUDGET, BUDGET_TABLE.replace(before, after))["eligible"]


def test_schedule_table_requires_complete_time_activity_rows_in_the_same_card():
    source = ("## Monday morning\n- 08:00-09:30 Check the equipment\n- 09:30-10:00 Break\n\n"
              "## Tuesday morning\n- 08:00-09:30 Walk through the venue\n- 09:30-10:00 Debrief")
    body = ('root=Column([monday,tuesday])\nmonday=Card([m],title="Monday morning")\n'
            'tuesday=Card([t],title="Tuesday morning")\n'
            'm=Table(columns=["Time","Activity"],rows=[["08:00-09:30","Check the equipment"],'
            '["09:30-10:00","Break"]])\n'
            't=Table(columns=["Time","Activity"],rows=[["08:00-09:30","Walk through the venue"],'
            '["09:30-10:00","Debrief"]])')
    result = assess(source, body)
    assert result["eligible"], result["reasons"]
    swapped = body.replace("monday=Card([m]", "monday=Card([t]").replace("tuesday=Card([t]", "tuesday=Card([m]")
    assert not assess(source, swapped)["eligible"]
    assert not assess(source, body.replace("09:30-10:00", "09:30-10:05", 1))["eligible"]


CHART = ("Chart Type: Stacked Bar Chart\nChart Title: Units by month\n"
         "X-Axis: Month | Y-Axis: Units sold\n\n"
         "| Month | Books | Games |\n|---|---|---|\n| March | 6 | 6 |\n| April | 8 | 8 |")
CHART_BODY = ('root=Column([chart,table])\n'
              'chart=Chart(chartType="stackedbar",title="Units by month",xKey="Month",'
              'xLabel="Month",yLabel="Units sold",columns=["Month","Books","Games"],'
              'rows=[["March",6,6],["April",8,8]],series=[{yKey:"Books",label:"Books"},'
              '{yKey:"Games",label:"Games"}])\n'
              'table=Table(columns=["Month","Books","Games"],rows=[["March",6,6],["April",8,8]])')


def test_combined_axes_bind_complete_adjacent_multiseries_table():
    result = assess(CHART, CHART_BODY)
    assert result["eligible"], result["reasons"]
    assert result["literal_missing_units_certified_by_pilot_fields"]


@pytest.mark.parametrize(("before", "after"), [
    ('yLabel="Units sold"', 'yLabel="Revenue"'),
    ('xLabel="Month"', 'xLabel="Week"'),
    ('["March",6,6],["April",8,8]', '["April",8,8],["March",6,6]'),
    ('["April",8,8]', '["April",7,8]'),
    ('{yKey:"Books",label:"Books"},{yKey:"Games",label:"Games"}',
     '{yKey:"Games",label:"Books"},{yKey:"Books",label:"Games"}'),
    (',{yKey:"Games",label:"Games"}', ''),
])
def test_chart_cannot_borrow_exact_data_from_the_separate_table(before, after):
    assert not assess(CHART, CHART_BODY.replace(before, after, 1))["eligible"]


def test_graphics_setting_does_not_create_a_chart_requirement_in_the_pilot():
    source = 'Disable "Use graphics acceleration when available" to troubleshoot the browser.'
    context = prepare_source_context_v5_4(source, config=RewardConfigV54(pilot_fidelity_v2=True))
    legacy = prepare_source_context_v5_4(source, config=RewardConfigV54())
    assert legacy.expected_contract["expected_role_counts"]["chart"] == 1
    assert context.expected_contract["expected_role_counts"]["chart"] == 0
    with_chart = prepare_source_context_v5_4(source + "\nShow a chart.",
                                            config=RewardConfigV54(pilot_fidelity_v2=True))
    assert with_chart.expected_contract["expected_role_counts"]["chart"] == 1


def test_citation_and_qualified_action_keep_different_roles_for_same_target():
    url = "https://example.com/guidance#local"
    source = ("## Local coordination\n"
              f"- Waste disposal: [Button: Municipal service] <{url}>\n\n"
              f"## Regulatory guidance\n- City waste guidance: <{url}>")
    body = ('root=Column([coordination,card,guidance,links])\n'
            'coordination=Text("Local coordination","h2")\n'
            'card=Card([button],title="Waste disposal")\n'
            f'button=Button("Municipal service",onPress=openUrl("{url}"))\n'
            'guidance=Text("Regulatory guidance","h2")\n'
            f'links=List(items=[{{label:"City waste guidance",url:"{url}"}}])')
    result = assess(source, body)
    assert result["eligible"], result["reasons"]
    for changed in (
        body.replace("#local", "#other", 1),
        body.replace("Waste disposal", "Local pickup"),
        body.replace("[coordination,card,guidance,links]", "[coordination,guidance,card,links]"),
        body.replace("card=Card([button],", "card=Card([],").replace("card,guidance", "card,button,guidance"),
        body.replace('links=List(items=[{label:"City waste guidance",url:',
                     'links=Button("City waste guidance",onPress=openUrl(').replace('"}])', '"))'),
    ):
        assert not assess(source, changed)["eligible"]


def test_original_url_column_is_literal_and_cannot_declare_a_button():
    first = "https://example.com/docs/a?region=eu#one"
    second = "https://example.com/docs/b?region=eu#two"
    source = f"| Document | Original URL |\n|---|---|\n| Manual | {first} |\n| Help | {second} |"
    table = f'Table(columns=["Document","Original URL"],rows=[["Manual","{first}"],["Help","{second}"]])'
    result = assess(source, "root=" + table)
    assert result["eligible"], result["reasons"]
    assert not assess(source, "root=" + table.replace("#one", "#two"))["eligible"]
    with_button = f'root=Column([table,button])\ntable={table}\nbutton=Button("Manual",onPress=openUrl("{first}"))'
    assert not assess(source, with_button)["eligible"]


def test_apostrophe_in_media_destination_masks_the_whole_exact_field():
    image = "https://example.com/photos/Piazza_de'_Ricci.jpg"
    icon = "https://example.com/icon.svg"
    source = f"  Media: Image=<{image}> Icon={icon}\r\n"
    fields = declared_media_occurrences(source)
    assert [source[slice(*item["_url_span"])] for item in fields] == [image, icon]
    masked, tokens, reverse = _mask_model_references(source, [], pilot_fidelity_v2=True)
    assert masked == f"  Media: Image=<{tokens[image]}> Icon={tokens[icon]}\r\n"
    assert reverse == {tokens[image]: image, tokens[icon]: icon}
    graph = {"elements": {"image": {"type": "Image", "props": {"url": tokens[image]}}}}
    assert not reference_destination_errors(graph, reverse)
    assert _restore_model_references(graph, reverse)["elements"]["image"]["props"]["url"] == image
    graph["elements"]["image"]["props"]["url"] += "'_Ricci.jpg"
    assert reference_destination_errors(graph, reverse) == ["invalid_reference_destination:elements.image.props.url"]
    legacy, legacy_tokens, _ = _mask_model_references(source, [])
    assert image not in legacy_tokens and "'_Ricci.jpg" in legacy


def test_formula_label_and_script_braces_preserve_operand_and_ownership():
    source = "## Mortgage\nMonthly amount = 520,000 × (1.003)^-300"
    body = ('root=Card([label,equation],title="Mortgage")\n'
            'label=Text("Monthly amount")\nequation=Formula(latex="520,000 × (1.003)^{-300}")')
    result = assess(source, body)
    assert result["eligible"], result["reasons"]
    for changed in (body.replace("-300", "300"), body.replace("1.003", "1.03"),
                    body.replace("Monthly amount", "Total amount"),
                    body.replace("[label,equation]", "[equation,label]")):
        assert not assess(source, changed)["eligible"]


def test_formula_spelling_only_normalizes_integer_exponents_and_bold_numbers():
    source = r"Calculation: $\$210 \times 7 = \mathbf{\$1,470}$"
    body = 'root=Formula(title="Calculation:",latex="$210 × 7 = $1,470")'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("1,470", "1,471"))["eligible"]
    assert formula_literal_key_v2(r"x^{2+3}") != formula_literal_key_v2(r"x^2+3")
    assert formula_literal_key_v2(r"\mathbf{x}") != formula_literal_key_v2("x")


def test_degree_spelling_in_exact_table_cell_preserves_operator_number_and_unit():
    source = r"| Check | Limit |" "\n|---|---|\n" r"| Deviation | $\le 2^\circ\text{F}$ |"
    body = 'root=Table(columns=["Check","Limit"],rows=[["Deviation","≤ 2°F"]])'
    result = assess(source, body)
    assert result["eligible"], result["reasons"]
    for value in ("< 2°F", "≤ 2°C", "≤ 3°F", "≤ -2°F"):
        assert not assess(source, body.replace("≤ 2°F", value))["eligible"]


def test_exact_email_heading_and_body_stay_visible_and_bound_to_one_draft():
    paragraphs = ("Dear Landlord,\n\nPlease approve work at 14 Oak Street under clause 4.2.\n\n"
                  "I will restore the door to its original condition at move-out at my cost.\n\n"
                  "Kind regards,\n[Your Name]\n[Tenancy Reference]")
    source = "Email Draft\n\nSubject: Request to install cat flap\n\n" + paragraphs
    body = ('root=Column([heading,email])\nheading=Text("Email Draft","h2")\n'
            'email=EmailPreview("Request to install cat flap",' + json.dumps(paragraphs) + ")")
    result = assess(source, body)
    assert result["eligible"], result["reasons"]
    for changed in (body.replace("Email Draft", "Approved Request"),
                    body.replace("under clause 4.2", "under clause 4.3"),
                    body.replace(" at my cost", ""), body.replace("[Tenancy Reference]", "")):
        assert not assess(source, changed)["eligible"]


def test_email_body_line_cannot_move_into_a_heading_before_the_draft():
    paragraphs = "Dear Landlord,\n\nEmail Draft\n\nPlease approve work under clause 4.2.\n\nThank you."
    source = "Subject: Request approval\n\n" + paragraphs
    changed = paragraphs.replace("Email Draft\n\n", "")
    body = ('root=Column([heading,email])\nheading=Text("Email Draft","h2")\n'
            'email=EmailPreview("Request approval",' + json.dumps(changed) + ")")
    assert not assess(source, body)["eligible"]
