"""Reviewed 1,000-output pilot: exact evidence, negative cases and isolation."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline.genui_quality.formula_evidence_v2 import (
    formula_literal_key_v2,
    formula_syntax_errors_v2,
)
from pipeline.genui_quality.pilot_layout_v2 import prose_spelling
from pipeline.genui_quality.pilot_source_contract_v2 import declared_media_occurrences
from pipeline.openai_stage3_fidelity_v2 import SUPPLEMENT_PATH
from pipeline.stage3_genui import _mask_model_references
from pipeline.training_fidelity import audit_training_fidelity
from pipeline.v11_regeneration_guard import assess_candidate


def assess(source, body, *, pilot=True):
    return assess_candidate({"id": "pilot-1000-regression", "response_text": source},
                            "<a2ui>" + body + "</a2ui>", pilot_fidelity_v2=pilot)


def test_bulleted_action_keeps_balanced_url_and_exact_destination():
    url = "https://www.who.int/facts/ambient-(outdoor)-air-quality#guidance"
    source = f"- Action: [Button: WHO Guidelines] <{url}>"
    body = f'root=Button("WHO Guidelines",onPress=openUrl("{url}"))'
    assert assess(source, body)["eligible"]
    for bad in (body.replace("#guidance", "#other"), body.replace("(outdoor)-air-quality", "(outdoor"),
                body.replace("WHO Guidelines", "WHO Homepage")):
        assert not assess(source, bad)["eligible"]
    assert not assess(source + "\n" + source, body)["eligible"]
    # A separate untyped occurrence is not owned by the explicit Action.
    assert not assess(source + f"\nKeep this URL in text: {url}", body)["eligible"]


def test_explicit_publisher_label_is_visible_and_exact():
    url = "https://www.gov.uk/tenancy-deposit-protection"
    label = "GOV.UK - Tenancy Deposit Protection"
    source = f"## Official References\n- [{label}]({url})"
    body = ('root=Column([heading,sources])\nheading=Text("Official References","h2")\n'
            'sources=Card([links])\n' + f'links=List(items=[{{label:"{label}",url:"{url}"}}])')
    assert assess(source, body)["eligible"]
    for bad in (body.replace(label, "Gov"), body.replace("gov.uk/", "gov.uk/elsewhere/"),
                body.replace('GOV.UK -', 'unrelated.org -')):
        assert not assess(source, bad)["eligible"]


def test_table_cta_binds_exact_complete_table_and_next_button():
    url = "https://tickets.imaxparis.fr/qr/ticket-17"
    source = ("| Detail | Information |\n|---|---|\n| Seat | G12 |\n"
              f"| Digital Entry | [View QR Code]({url}) |")
    body = ('root=Column([summary,entry])\nsummary=Table(columns=["Detail","Information"],'
            'rows=[["Seat","G12"],["Digital Entry","View QR Code"]])\n'
            f'entry=Button("View QR Code",onPress=openUrl("{url}"))')
    assert assess(source, body)["eligible"]
    for bad in (body.replace("G12", "G13"), body.replace("ticket-17", "ticket-18"),
                body.replace("summary,entry", "entry,summary"),
                body.replace('"Digital Entry","View QR Code"', '"Digital Entry",""')):
        assert not assess(source, bad)["eligible"]
    literal_source, literal_body = (text.replace("View QR Code", "View ticket_id") for text in (source, body))
    assert assess(literal_source, literal_body)["eligible"]
    assert not assess(literal_source, literal_body.replace("ticket_id", "ticketid"))["eligible"]


def test_equation_title_keeps_exact_body_qualifiers_and_units():
    source = "Nozomi 101 total = 14,500 × 3 = JPY 43,500 per party"
    body = 'root=Formula(latex="14,500 × 3 = JPY 43,500 per party",title="Nozomi 101 total")'
    assert assess(source, body)["eligible"]
    for bad in (body.replace("per party", "per rider"), body.replace(" × 3", " + 3"),
                body.replace("JPY", "USD"), body.replace("101", "102"), body.replace("14,500", "14,501")):
        assert not assess(source, bad)["eligible"]


def test_chart_spec_counts_once_and_requires_exact_ordered_plotted_data():
    source = "Chart Type: Donut Chart\nChart Title: Trip Expenses\nLabels: Hotel, Food\nValues: 1200, 600"
    body = ('root=Chart(chartType="donut",columns=["Category","Value"],rows=[["Hotel",1200],["Food",600]],'
            'title="Trip Expenses",xKey="Category",yKey="Value")')
    assert assess(source, body)["eligible"]
    for bad in (body.replace("donut", "pie"), body.replace('"Food",600', '"Food",601'),
                body.replace('["Hotel",1200],["Food",600]', '["Hotel",600],["Food",1200]')):
        assert not assess(source, bad)["eligible"]
    second = source.replace("Trip Expenses", "Other Expenses").replace("1200", "300")
    assert not assess(source + "\n\n" + second, body)["eligible"]
    both = ('root=Column([first,second])\n' + body.replace("root=", "first=") + "\n"
            + body.replace("root=", "second=").replace("Trip Expenses", "Other Expenses").replace("1200", "300"))
    assert assess(source + "\n\n" + second, both)["eligible"]


def test_chart_series_rows_are_scored_as_chart_data_with_exact_bindings():
    source = ("Chart Title: Spending\nX-Axis: Month\nY-Axis: Spending (USD)\n"
              "- January | Rent: 1200 | Food: 400\n- February | Rent: 1100 | Food: 300")
    body = ('root=Chart(chartType="stackedbar",columns=["Month","Rent","Food"],'
            'rows=[["January",1200,400],["February",1100,300]],title="Spending",xKey="Month",'
            'xLabel="Month",yLabel="Spending (USD)",series=[{yKey:"Rent",label:"Rent"},{yKey:"Food",label:"Food"}])')
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace('["February",1100,300]', '["February",300,1100]'))["eligible"]
    assert not assess(source, body.replace('yKey:"Rent",label:"Rent"', 'yKey:"Food",label:"Rent"'))["eligible"]
    equal_source = source.replace("Food: 400", "Food: 1200").replace("Food: 300", "Food: 1100")
    equal_body = body.replace("1200,400", "1200,1200").replace("1100,300", "1100,1100")
    assert assess(equal_source, equal_body)["eligible"]
    swapped = equal_body.replace('yKey:"Rent",label:"Rent"', 'yKey:"Food",label:"Rent"').replace(
        'yKey:"Food",label:"Food"', 'yKey:"Rent",label:"Food"')
    assert not assess(equal_source, swapped)["eligible"]


def test_invalid_time_axis_bar_orientation_stays_blocked():
    source = "Chart Title: Clients by Day\nX-Axis: Date (UTC)\nY-Axis: Clients"
    body = ('root=Chart(chartType="bar",columns=["Date","Clients"],'
            'rows=[["2026-11-01",5],["2026-11-02",7]],title="Clients by Day",xType="time",'
            'xKey="Date",yKey="Clients",xLabel="Date",yLabel="Clients")')
    result = assess(source, body)
    assert not result["eligible"]
    assert "renderer_component_contract" in result["canonical_acceptance"]["blocking_reasons"]


def test_explicit_horizontal_axis_labels_keep_their_table_bindings():
    source = ("Chart Title: Revenue by Product Category\nX-Axis: Revenue (USD)\nY-Axis: Category\n\n"
              "| Category | Revenue | Share |\n|---|---|---|\n| Books | 500 | 50% |\n| Games | 500 | 50% |")
    body = ('root=Column([chart,table])\nchart=Chart(chartType="bar",orientation="horizontal",'
            'columns=["Category","Revenue"],rows=[["Books",500],["Games",500]],title="Revenue by Product Category",'
            'xKey="Category",yKey="Revenue",xLabel="Revenue (USD)",yLabel="Category")\n'
            'table=Table(columns=["Category","Revenue","Share"],rows=[["Books",500,"50%"],["Games",500,"50%"]])')
    assert assess(source, body)["eligible"]
    for bad in (body.replace('"Revenue (USD)"', '"Revenue (CAD)"'),
                body.replace('"horizontal"', '"vertical"'), body.replace('["Games",500]', '["Games",501]', 1)):
        assert not assess(source, bad)["eligible"]


def test_isolated_constraint_is_prose_but_named_entities_still_require_fields():
    source = "- Vehicle: 2.5 m refrigerated van, clearance requirement: avoid bridges under 3.0 m."
    body = 'root=List(items=["Vehicle: 2.5 m refrigerated van, clearance requirement: avoid bridges under 3.0 m."])'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("3.0 m", "2.9 m"))["eligible"]
    named = "- Hotel: North Inn, Rate: $130"
    assert not assess(named, 'root=List(items=["Hotel: North Inn, Rate: $130"])')["eligible"]


def test_numbered_stop_card_labels_preserve_each_window_and_service():
    source = ("Stops, time windows and service:\n1. North Station 09:00-09:45, service 30 min\n"
              "2. South Station 10:00-10:45, service 45 min")
    body = ('root=Column([heading,north,south])\nheading=Text("Stops, time windows and service:","h3")\n'
            'north=Card([nw,ns],title="1. North Station")\nnw=Text("Time window: 09:00-09:45")\nns=Text("Service: 30 min")\n'
            'south=Card([sw,ss],title="2. South Station")\nsw=Text("Time window: 10:00-10:45")\nss=Text("Service: 45 min")')
    assert assess(source, body)["eligible"]
    for bad in (body.replace("north=Card([nw,ns]", "north=Card([nw,ss]").replace("south=Card([sw,ss]", "south=Card([sw,ns]"),
                body.replace("1. North Station", "North Station"), body.replace("45 min", "45 h")):
        assert not assess(source, bad)["eligible"]


def test_table_math_preserves_approximation_and_grouped_operands():
    source = r"| Item | Value |" "\n|---|---|\n" r"| Cost | $\approx$ $15–$50 |" "\n" r"| Energy | $2 \times 7,700$ |"
    body = 'root=Table(columns=["Item","Value"],rows=[["Cost","≈ $15–$50"],["Energy","2 × 7,700"]])'
    assert assess(source, body)["eligible"]
    for bad in (body.replace("≈", "="), body.replace("7,700", "7,701"), body.replace("×", "+"), body.replace("$50", "50")):
        assert not assess(source, bad)["eligible"]


def test_literal_markdown_table_keeps_exact_cells_and_headers():
    source = ("| option_id | Cost | Actual | Note* |\n|---|---|---|---|\n"
              "| A | ~12.50 | ___ | Keep optional |\n| B | 10.25 | 8.25 | Keep exact |")
    body = ('root=Table(columns=["option_id","Cost","Actual","Note*"],'
            'rows=[["A","~12.50","___","Keep optional"],["B","10.25","8.25","Keep exact"]])')
    assert assess(source, body)["eligible"]
    for old, new in (("~12.50", "12.50"), ("___", "0"), ("option_id", "optionid"),
                     ("Note*", "Note"), ("Keep optional", "Keep required"), ("~12.50", "~12.5"),
                     ('"10.25","8.25"', '"8.25","10.25"'), (',["B","10.25","8.25","Keep exact"]', '')):
        assert not assess(source, body.replace(old, new))["eligible"]


def test_table_only_recognized_inline_spelling_preserves_values_and_code():
    source = r"| Path | Outcome |" + "\n|---|---|\n" + r"| `editor.folding` | Move A $\to$ B |"
    body = 'root=Table(columns=["Path","Outcome"],rows=[["editor.folding","Move A → B"]])'
    assert assess(source, body)["eligible"]
    for old, new in (("editor.folding", "editorFolding"), ("A → B", "A ← B")):
        assert not assess(source, body.replace(old, new))["eligible"]


def test_bold_dollar_amounts_in_one_cell_keep_both_exact_values():
    source = ("| Category | Penalty |\n|---|---|\n"
              "| Max penalties | Up to **$2,500** per unintentional or **$7,500** per intentional violation. |")
    body = ('root=Table(columns=["Category","Penalty"],rows=[["Max penalties",'
            '"Up to $2,500 per unintentional or $7,500 per intentional violation."]])')
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("$2,500", "$2,600"))["eligible"]
    assert not assess(source, body.replace("$7,500", "7,500"))["eligible"]


def test_ambiguous_literal_tables_do_not_use_a_lossy_shape_as_proof():
    source = "| Item | Price |\n|---|---|\n| A | ~20 |\n\n| Item | Price |\n|---|---|\n| A | 20 |"
    body = ('root=Column([first,second])\nfirst=Table(columns=["Item","Price"],rows=[["A","20"]])\n'
            'second=Table(columns=["Item","Price"],rows=[["A","20"]])')
    assert not assess(source, body)["eligible"]


@pytest.mark.parametrize("source, visible", [
    ("Check **CRITICAL_PROCESS_DIED** in **Safe Mode**.", "Check CRITICAL_PROCESS_DIED in Safe Mode."),
    ("Example: *We learn water + sunlight = food. Why does a plant need water?*",
     "Example: We learn water + sunlight = food. Why does a plant need water?"),
])
def test_only_prose_delimiters_change(source, visible):
    body = "root=Text(" + json.dumps(visible) + ")"
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("CRITICAL_PROCESS_DIED", "CRITICALPROCESSDIED")
                      .replace("sunlight = food", "sunlight food"))["eligible"]


def test_email_metadata_and_bullet_spelling_match_visible_fields():
    source = ("**Subject:** Meeting at 18:00\n**From:** Meeting Team\n**To:** Event Guests\n"
              "**Date:** 2026-11-10\n\nHi everyone,\n\n* Arrival: 17:45\n* Start: 18:00\n\nThank you.")
    body = ('root=EmailPreview(subject="Meeting at 18:00",from="Meeting Team",to="Event Guests",'
            'date="2026-11-10",body="Hi everyone,\\n\\n• Arrival: 17:45\\n• Start: 18:00\\n\\nThank you.")')
    assert assess(source, body)["eligible"]
    for bad in (body.replace("17:45", "17:46"), body.replace('date="2026-11-10"', 'date=""'),
                body.replace('from="Meeting Team",to="Event Guests"', 'from="Event Guests",to="Meeting Team"'),
                body.replace("• Arrival: 17:45\\n• Start: 18:00", "• Start: 18:00\\n• Arrival: 17:45"),
                body.replace("Thank you.", "Thank you. Tickets are free.")):
        assert not assess(source, bad)["eligible"]


def test_qualified_reference_heading_keeps_exact_role_and_visible_heading():
    url = "https://example.com/manual?mode=full#page"
    source = (f"Action: [Open manual] <{url}>\n\n## Official References\n"
              f"- Provenance: provided by the author.\n- Technical manual: <{url}>")
    body = (f'root=Column([open,heading,note,sources])\nopen=Button("Open manual",onPress=openUrl("{url}"))\n'
            f'heading=Text("Official References","h2")\nnote=Text("Provenance: provided by the author.")\n'
            f'sources=Card([links])\nlinks=List(items=[{{label:"Technical manual",url:"{url}"}}])')
    assert assess(source, body)["eligible"]
    for bad in (body.replace("Official References", "Additional links"),
                body.replace('"Technical manual"', '"Other manual"'),
                body.replace('url:"' + url, 'url:"' + url.replace("#page", "#wrong")),
                body.replace("open,heading,note,sources", "open,heading,sources")
                    .replace('note=Text("Provenance: provided by the author.")\n', "")
                    .replace("items=[{", 'items=["Provenance: provided by the author.",{')):
        assert not assess(source, bad)["eligible"]


def test_pipe_card_keeps_references_after_visible_prose_with_the_same_owner():
    source = ("Option 1: North | Quiet seating | 2-hour limit\n"
              "Reserve a seat early to secure a place.\nMedia: Icon=https://example.com/north.svg")
    body = ('root=Card([place,limit,note,icon],title="Option 1: North")\n'
            'place=Text("Quiet seating")\nlimit=Text("2-hour limit")\n'
            'note=Text("Reserve a seat early to secure a place.")\n'
            'icon=Icon(url="https://example.com/north.svg")')
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("north.svg", "south.svg"))["eligible"]
    moved = 'root=Column([card,note])\n' + body.replace(
        "root=Card([place,limit,note,icon]", "card=Card([place,limit,icon]")
    assert not assess(source, moved)["eligible"]


def test_chart_axes_must_be_visible_on_the_title_matched_chart():
    source = ("Chart Title: Enrollment\nX-Axis: Campus location\nY-Axis: Students per class\n\n"
              "| Campus | Count |\n|---|---|\n| North | 24 |\n| South | 18 |")
    body = ('root=Chart(chartType="column",columns=["Campus","Count"],rows=[["North",24],["South",18]],'
            'title="Enrollment",xKey="Campus",yKey="Count",xType="category",'
            'xLabel="Campus location",yLabel="Students per class")')
    result = assess(source, body)
    assert "content_unit_fidelity" not in result["canonical_acceptance"]["review_reasons"]
    assert "fidelity:source_literal_not_exactly_represented" not in result["reasons"]
    for bad in (body.replace("Students per class", "Students per school"),
                body.replace('title="Enrollment"', 'title="Other chart"'),
                body.replace('rows=[["North",24],["South",18]]', "rows=[]")):
        assert not assess(source, bad)["eligible"]


def test_checklist_requires_items_the_android_renderer_can_display():
    source = "Verify the schedule.\nConfirm the list."
    plain = 'root=List(items=["Verify the schedule.","Confirm the list."])'
    assert assess(source, plain)["eligible"]
    strings = plain.replace("List", "Checklist")
    result = assess(source, strings)
    assert not result["eligible"]
    assert "dynamic_evidence_incomplete" in result["canonical_acceptance"]["review_reasons"]
    objects = 'root=Checklist(items=[{label:"Verify the schedule."},{label:"Confirm the list."}])'
    assert assess(source, objects)["eligible"]
    assert not assess(source, objects.replace("Confirm the list.", ""))["eligible"]


def test_comma_then_clause_stays_procedural_prose():
    source = "- Clean the area, then wipe with disposable cloths: floors, walls, and window sills."
    body = 'root=List(items=["Clean the area, then wipe with disposable cloths: floors, walls, and window sills."])'
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace(", and window sills", ""))["eligible"]


def test_instructional_paths_and_slashes_are_not_reference_bindings():
    source = r"The asset/liability ratio. Delete C:\Windows\spool\pending.txt, then GET /assets/{id}."
    assert _mask_model_references(source, [], pilot_fidelity_v2=True) == (source, {}, {})
    assert _mask_model_references(source, []) == _mask_model_references(source, [], pilot_fidelity_v2=False)
    assert _mask_model_references(source, [])[1]  # Original default behavior stays unchanged.


@pytest.mark.parametrize("source", [
    "Media: Image=assets/photo.png Icon=https://example.com/a.svg",
    'Image: "C:\\Photos\\local image.png"',
    "Action: [Open] https://example.com/a?x=1#section",
])
def test_declared_references_still_have_exact_donors(source):
    masked, _, mapping = _mask_model_references(source, [], pilot_fidelity_v2=True)
    assert masked != source and mapping
    assert all(value in source for value in mapping.values())


def test_currency_does_not_protect_unrelated_markdown_emphasis():
    source = "Budget **$25**, fare **$10**. Keep $C$ unchanged."
    assert prose_spelling(source) == "Budget $25, fare $10. Keep C unchanged."
    body = 'root=Text("Budget $25, fare $10. Keep C unchanged.")'
    assert assess(source, body)["eligible"]
    for changed in ("Budget $26, fare $10. Keep C unchanged.", "Budget 25, fare 10. Keep C unchanged."):
        assert not assess(source, "root=Text(" + json.dumps(changed) + ")")["eligible"]


@pytest.mark.parametrize("literal", ["$x_2 * y$", "$a+b$", "$2 **x** + 3$", "$2 **x** + 3$kg", "`$C$`", "2**3", "a * b * c", "$25"])
def test_math_code_currency_and_operators_stay_exact(literal):
    assert prose_spelling(literal) == literal


PROFILE = "Inputs used\n\n- Sex: female, Age: 34 y\n- Height: 165 cm, Weight: 78 kg"
PROFILE_BODY = ('root=Card([sex,age,height,weight],title="Inputs used")\n'
                'sex=Text("Sex: female")\nage=Text("Age: 34 y")\n'
                'height=Text("Height: 165 cm")\nweight=Text("Weight: 78 kg")')


def test_plain_profile_section_owns_all_split_fields():
    assert assess(PROFILE, PROFILE_BODY)["eligible"]
    for bad in (PROFILE_BODY.replace("78 kg", "79 kg"), PROFILE_BODY.replace("[sex,age,height,weight]", "[sex,age,height]"),
                PROFILE_BODY.replace("Inputs used", "Other person")):
        assert not assess(PROFILE, bad)["eligible"]


@pytest.mark.parametrize("first,trailer", [("**Route Map**", ""), ("Option 3: Route Map", " |")])
def test_named_pipe_cards_and_trailing_delimiter_keep_all_fields(first, trailer):
    source = f"{first} | Best for NYC specifics | High accuracy{trailer}\nAction: [Button: Open map] https://example.com/map#one"
    title = first.strip("*")
    body = (f'root=Card([a,b,go],title="{title}")\na=Text("Best for NYC specifics")\n'
            'b=Text("High accuracy")\ngo=Button("Open map",onPress=openUrl(url="https://example.com/map#one"))')
    assert assess(source, body)["eligible"]
    for bad in (body.replace("High accuracy", "Low accuracy"), body.replace("[a,b,go]", "[b,a,go]"),
                body.replace("#one", "#two"), body.replace("[a,b,go]", "[a,go]")):
        assert not assess(source, bad)["eligible"]
    moved = body.replace("root=Card([a,b,go]", "root=Column([card,go])\ncard=Card([a,b]")
    assert not assess(source, moved)["eligible"]


EMAIL_SOURCE = "Subject: Meeting notes\n\nDear Alex,\n\nThe price is $25, pending approval.\n\nRegards,\nSam"
EMAIL_BODY = "root=EmailPreview(subject=\"Meeting notes\",body=" + json.dumps("Dear Alex,\n\nThe price is $25, pending approval.\n\nRegards,\nSam") + ")"


def test_email_subject_and_complete_body_are_visible_source_evidence():
    result = assess(EMAIL_SOURCE, EMAIL_BODY)
    assert result["eligible"], result
    for bad in (EMAIL_BODY.replace("$25", "$26"), EMAIL_BODY.replace(", pending approval", ""),
                EMAIL_BODY.replace("Dear Alex", "Dear Bob"), EMAIL_BODY.replace("Regards,\\nSam", "")):
        assert not assess(EMAIL_SOURCE, bad)["eligible"]


def test_email_cannot_erase_subject_or_add_an_unprovided_paragraph():
    for bad in (EMAIL_BODY.replace("Meeting notes", ""), EMAIL_BODY.replace("pending approval.", "pending approval. Guaranteed delivery.")):
        assert not assess(EMAIL_SOURCE, bad)["eligible"]


def test_formula_qualifier_stays_with_its_equation_and_owner():
    source = "**Oats**\n* Protein: 4 x 10 = 40 g *syrup protein not provided*"
    body = ('root=Card([equation,note],title="Oats")\n'
            'equation=Formula(latex="4 x 10 = 40 g",title="Protein")\n'
            'note=Text("syrup protein not provided")')
    assert assess(source, body)["eligible"]
    for bad in (body.replace("40 g", "41 g"), body.replace("not provided", "provided"),
                body.replace("[equation,note]", "[equation]"), body.replace('title="Oats"', 'title="Milk"')):
        assert not assess(source, bad)["eligible"]
    moved = body.replace('root=Card([equation,note]', 'root=Column([card,note])\ncard=Card([equation]')
    assert not assess(source, moved)["eligible"]


def test_named_record_keeps_action_after_its_qualifier_bullets():
    source = ("Option 1: Car A | Low running cost | Best for commuting\n"
              "- **Pros**: Low fuel usage.\n- **Cons**: Higher capital loss.\n"
              "Action: [Button: View specs] https://example.com/car")
    body = ('root=Card([a,b,p,c,go],title="Option 1: Car A")\n'
            'a=Text("Low running cost")\nb=Text("Best for commuting")\n'
            'p=Text("Pros: Low fuel usage.")\nc=Text("Cons: Higher capital loss.")\n'
            'go=Button("View specs",onPress=openUrl(url="https://example.com/car"))')
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("Higher capital loss.", "Low capital loss."))["eligible"]
    moved = body.replace("root=Card([a,b,p,c,go]", "root=Column([card,go])\ncard=Card([a,b,p,c]")
    assert not assess(source, moved)["eligible"]


def test_record_actions_after_plain_bullets_cannot_move_between_cards():
    source = ("Option 1: Route A | Walk | Quiet\n- Use the north entrance.\nAction: [Button: Map A] https://example.com/a\n\n"
              "Option 2: Route B | Bike | Fast\n- Use the south entrance.\nAction: [Button: Map B] https://example.com/b")
    body = ('root=Column([a,b])\na=Card([a1,a2,an,ab],title="Option 1: Route A")\n'
            'a1=Text("Walk")\na2=Text("Quiet")\nan=Text("Use the north entrance.")\n'
            'ab=Button("Map A",onPress=openUrl(url="https://example.com/a"))\n'
            'b=Card([b1,b2,bn,bb],title="Option 2: Route B")\nb1=Text("Bike")\nb2=Text("Fast")\n'
            'bn=Text("Use the south entrance.")\nbb=Button("Map B",onPress=openUrl(url="https://example.com/b"))')
    assert assess(source, body)["eligible"]
    swapped = body.replace("[a1,a2,an,ab]", "[a1,a2,an,bb]").replace("[b1,b2,bn,bb]", "[b1,b2,bn,ab]")
    assert not assess(source, swapped)["eligible"]


def test_plain_heading_equations_keep_their_source_section():
    source = "Maintenance estimate\nTDEE = 1480 × 1.55\n\nDaily target\nTarget = 2294 − 550"
    body = ('root=Column([a,b])\na=Card([eq1],title="Maintenance estimate")\n'
            'b=Card([eq2],title="Daily target")\neq1=Formula("TDEE = 1480 × 1.55")\n'
            'eq2=Formula("Target = 2294 − 550")')
    assert assess(source, body)["eligible"]
    swapped = body.replace("[eq1]", "[temp]").replace("[eq2]", "[eq1]").replace("[temp]", "[eq2]")
    assert not assess(source, swapped)["eligible"]


def test_equation_after_blank_line_is_not_a_new_section_heading():
    source = "**Target**\nDaily deficit = 3850 / 7 ≈ 550 kcal/day\n\nDaily target ≈ TDEE - 550\n≈ 2230 - 550\n≈ 1680 kcal/day"
    body = ('root=Card([a,b,c,d],title="Target")\na=Formula("Daily deficit = 3850 / 7 ≈ 550 kcal/day")\n'
            'b=Formula("Daily target ≈ TDEE - 550")\nc=Formula("≈ 2230 - 550")\nd=Formula("≈ 1680 kcal/day")')
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("1680", "1690"))["eligible"]


TRACK_SOURCE = ("## Playlist\n- **Spring** | Vivaldi | 3:30\n- **Serenade** | Mozart | 4:50\n\n"
                "## Budget\n| Item | Price |\n| --- | --- |\n| Tickets | $25 |")
TRACK_BODY = ('root=Column([heading,tracks,budgetHeading,budget])\nheading=Text("Playlist","h2")\n'
              'tracks=Table(columns=["Track","Artist","Duration"],rows=[["Spring","Vivaldi","3:30"],'
              '["Serenade","Mozart","4:50"]],domain="music",preferredPresentation="cards")\n'
              'budgetHeading=Text("Budget","h2")\nbudget=Table(columns=["Item","Price"],rows=[["Tickets","$25"]])')


def test_record_table_is_not_lost_when_a_separate_markdown_table_exists():
    assert assess(TRACK_SOURCE, TRACK_BODY)["eligible"]
    for bad in (TRACK_BODY.replace('"3:30"', '"3:31"'), TRACK_BODY.replace('"Artist"', '"Conductor"'),
                TRACK_BODY.replace(',["Serenade","Mozart","4:50"]', ''),
                TRACK_BODY.replace('"Spring","Vivaldi"', '"Vivaldi","Spring"'),
                TRACK_BODY.replace('[heading,tracks,budgetHeading,budget]', '[heading,budgetHeading,tracks,budget]')):
        assert not assess(TRACK_SOURCE, bad)["eligible"]


def test_labeled_pipe_fields_preserve_numeric_occurrences():
    source = "## Investment\nTotal Investment: **$10,000** | Target Allocation: **40/40/20** | Total Units Purchased: **65**"
    body = ('root=Card([a,b,c],title="Investment")\na=Text("Total Investment: $10,000")\n'
            'b=Text("Target Allocation: 40/40/20")\nc=Text("Total Units Purchased: 65")')
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("65", "66"))["eligible"]
    assert not assess(source, body.replace("[a,b,c]", "[b,a,c]"))["eligible"]


def test_floor_comparison_does_not_allow_unsupported_output_commands():
    assert formula_literal_key_v2(r"\left\lfloor\frac{10}{3}\right\rfloor") == formula_literal_key_v2(r"⌊\frac{10}{3}⌋")
    assert formula_syntax_errors_v2(r"\lfloor x \rfloor")
    assert formula_literal_key_v2("⌊10/3⌋") != formula_literal_key_v2("⌈10/3⌉")


@pytest.mark.parametrize("expression", [
    "$2 **x** + 3$kg", "$2 **x** + 3$2", "$ a *b* c $kg",
    "$ 2 *b* c $kg", "$2 **rate** c$kg",
])
def test_inline_math_followed_by_unit_cannot_lose_operators(expression):
    source = "The value is " + expression + "."
    assert prose_spelling(source) == source
    assert not assess(source, "root=Text(" + json.dumps(source.replace("*", "")) + ")")["eligible"]


def test_typed_media_on_one_line_keeps_every_kind_and_destination():
    source = "Media: Image=https://example.com/a.png Icon=https://example.com/b.svg"
    assert [(x["kind"], x["url"]) for x in declared_media_occurrences(source)] == [
        ("Image", "https://example.com/a.png"), ("Icon", "https://example.com/b.svg")]
    assert not declared_media_occurrences("```\n" + source + "\n```")
    assert not declared_media_occurrences(source + " extra prose")
    body = 'root=Column([a,b])\na=Image(src="https://example.com/a.png")\nb=Icon(src="https://example.com/b.svg")'
    result = assess(source, body)
    assert result["eligible"], result
    for bad in (body.replace("[a,b]", "[a]"), body.replace("a.png", "wrong.png"), body.replace("b=Icon", "b=Image")):
        assert not assess(source, bad)["eligible"]


@pytest.mark.parametrize("kind", ["Image", "Icon"])
@pytest.mark.parametrize("placement", ["after", "before", "heading"])
def test_named_record_media_cannot_be_swapped_or_detached(kind, placement):
    first, second = "https://example.com/shared.png#alpha", "https://example.com/shared.png#beta"
    source = (f"**Alpha** | Budget choice | Ten units\nMedia: {kind}={first}\n\n"
              f"**Beta** | Premium choice | Twenty units\nMedia: {kind}={second}")
    body = (f'root=Column([a,b])\na=Card([ad,an,ai],title="Alpha")\nad=Text("Budget choice")\n'
            f'an=Text("Ten units")\nai={kind}(src="{first}")\nb=Card([bd,bn,bi],title="Beta")\n'
            f'bd=Text("Premium choice")\nbn=Text("Twenty units")\nbi={kind}(src="{second}")')
    if placement != "after":
        source = (f"Media: {kind}={first}\n**Alpha** | Budget choice | Ten units\n\n"
                  f"Media: {kind}={second}\n**Beta** | Premium choice | Twenty units")
    if placement == "heading":
        source = "## Choices\n" + source
        body = body.replace("Column([a,b])", "Column([heading,a,b])") + '\nheading=Text("Choices","h2")'
    assert assess(source, body)["eligible"]
    swapped = body.replace(first, "temporary").replace(second, first).replace("temporary", second)
    detached = body.replace("a,b])", "a,b,ai,bi])").replace("[ad,an,ai]", "[ad,an]").replace("[bd,bn,bi]", "[bd,bn]")
    duplicated = body.replace("[bd,bn,bi]", "[bd,bn,ai]")
    for bad in (swapped, detached, duplicated):
        assert not assess(source, bad)["eligible"]


def test_labeled_source_link_keeps_exact_label_target_and_role():
    source = "Remote learning link: https://example.com/learn?x=1#one"
    body = 'root=List(items=[{label:"Remote learning link",url:"https://example.com/learn?x=1#one"}])'
    assert assess(source, body)["eligible"]
    for bad in (body.replace("#one", "#two"), body.replace("Remote learning link", "Learn"),
                'root=Button("Remote learning link",onPress=openUrl(url="https://example.com/learn?x=1#one"))'):
        assert not assess(source, bad)["eligible"]


@pytest.mark.parametrize("kind", ["Image", "Icon"])
def test_declared_media_fragments_and_occurrence_counts_stay_exact(kind):
    source = f"Media: {kind}=https://example.com/item.svg#one"
    body = f'root={kind}(src="https://example.com/item.svg#one")'
    assert assess(source, body)["eligible"]
    changed = assess(source, body.replace("#one", "#two"))
    assert not changed["eligible"]
    assert "pilot_declared_media_mismatch" in changed["reasons"]
    assert not assess(source + "\n" + source, body)["eligible"]
    extra = ('root=Column([a,b])\n' + body.replace("root=", "a=") + "\n"
             + body.replace("root=", "b=").replace("#one", "#two"))
    assert not assess(source, extra)["eligible"]


def test_complete_literal_proof_is_not_capped_by_sixteen_examples():
    source = "\n".join(f"Option {i}: Name {i} | Full field {i} | Note {i}" for i in range(1, 20))
    children = ",".join(f"c{i}" for i in range(1, 20))
    lines = [f"root=Column([{children}])"]
    for i in range(1, 20):
        lines.extend([f'c{i}=Card([a{i},b{i}],title="Option {i}: Name {i}")',
                      f'a{i}=Text("Full field {i}")', f'b{i}=Text("Note {i}")'])
    body = "\n".join(lines)
    result = assess(source, body)
    assert result["eligible"], result
    assert result["fidelity"]["missing_unit_count"] > 16
    assert len(result["fidelity"]["missing_units"]) == 16
    assert result["fidelity"]["missing_unit_examples_truncated"]
    assert len(result["literal_missing_units_certified_by_pilot_fields"]) == 19
    for bad in (body.replace("Note 19", "Unprovided fact"), body.replace("[a19,b19]", "[a19]")):
        assert not assess(source, bad)["eligible"]


def test_complete_internal_audit_is_explicit_and_does_not_change_default():
    graph = {"root": "r", "state": {}, "elements": {"r": {"type": "Text", "props": {"text": "Original."}}}}
    untouched = copy.deepcopy(graph)
    assert audit_training_fidelity("Original.", graph) == audit_training_fidelity("Original.", graph, complete_missing_units=False)
    with pytest.raises(ValueError):
        audit_training_fidelity("Original.", graph, complete_missing_units=True)
    assert graph == untouched


def test_rejected_shapes_stay_rejected_and_future_prompt_explains_them():
    assert not assess("A blank form: Name | Age", 'root=Table(columns=["Name","Age"],rows=[])')["eligible"]
    assert not assess("Keep this.", 'root=Column([bad id])\nbad id=Text("Keep this.")')["eligible"]
    supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
    for instruction in ("[A-Za-z_][A-Za-z0-9_]*", "Do not emit Table(rows=[])",
                        "native source-link Card is an exception", "asset/liability"):
        assert instruction in supplement


@pytest.mark.parametrize("source,body", [(PROFILE, PROFILE_BODY), (EMAIL_SOURCE, EMAIL_BODY)])
def test_new_cases_do_not_enable_pilot_in_ordinary_pipelines(source, body):
    row = {"id": "default-isolation", "response_text": source}
    target = "<a2ui>" + body + "</a2ui>"
    assert assess_candidate(row, target) == assess_candidate(row, target, pilot_fidelity_v2=False)


def test_named_list_table_retains_each_total_duration_and_owner():
    source = (
        "## Hotel Rates\n| Hotel | Nightly Price |\n| --- | --- |\n| Alpha | $20 |\n| Beta | $30 |\n\n"
        "## Stay Cost Estimates\n- **Alpha**: $120 total for 6 nights.\n- **Beta**: $180 total for 6 nights."
    )
    body = (
        'root=Column([ratesHeading,rates,costHeading,costs])\n'
        'ratesHeading=Text("Hotel Rates","h2")\n'
        'rates=Table(columns=["Hotel","Nightly Price"],rows=[["Alpha","$20"],["Beta","$30"]])\n'
        'costHeading=Text("Stay Cost Estimates","h2")\n'
        'costs=Table(columns=["Hotel","Stay Cost Estimate"],'
        'rows=[["Alpha","$120 total for 6 nights."],["Beta","$180 total for 6 nights."]])'
    )
    assert assess(source, body)["eligible"]
    for bad in (body.replace('$120 total for 6 nights.', '$120 total.'),
                body.replace('"Alpha","$120', '"Beta","$120').replace('"Beta","$180', '"Alpha","$180'),
                body.replace('Stay Cost Estimate"', 'Fees included"'),
                body.replace('[ratesHeading,rates,costHeading,costs]', '[ratesHeading,rates,costs,costHeading]')):
        assert not assess(source, bad)["eligible"]


def test_pipe_summary_with_punctuation_owns_literal_fields_not_the_section_icon():
    icon = "https://example.org/calc.svg"
    source = (
        f"## Monthly Split\nMedia: Icon={icon}\n"
        "Total: **$260.00** | Tax & Reserve: **$94.67** | Person B/C Avg: **$82.67**"
    )
    body = (
        'root=Column([heading,icon,summary])\nheading=Text("Monthly Split","h2")\n'
        f'icon=Icon(url="{icon}")\n'
        'summary=Card([total,tax,avg])\ntotal=Text("Total: $260.00")\n'
        'tax=Text("Tax & Reserve: $94.67")\navg=Text("Person B/C Avg: $82.67")'
    )
    assert assess(source, body)["eligible"]
    for bad in (body.replace("$94.67", "$94.76"), body.replace("$82.67", "$82.670"),
                body.replace("[total,tax,avg]", "[avg,tax,total]"), body.replace("[heading,icon,summary]", "[heading,summary]")):
        assert not assess(source, bad)["eligible"]


def test_inline_media_stays_on_its_own_record_and_out_of_number_and_action_matching():
    a = "https://example.org/v1.11.3/ai.svg"
    b = "https://example.org/v2.0.5/cloud.svg"
    source = (f"## Conference Tracks\nAI Track | Models and agents | Media: Icon={a}\n"
              f"Cloud Track | Network orchestration | Media: Icon={b}")
    body = (
        'root=Column([heading,ai,cloud])\nheading=Text("Conference Tracks","h2")\n'
        'ai=Card([aiIcon,aiText],title="AI Track")\ncloud=Card([cloudIcon,cloudText],title="Cloud Track")\n'
        f'aiIcon=Icon(url="{a}")\ncloudIcon=Icon(url="{b}")\n'
        'aiText=Text("Models and agents")\ncloudText=Text("Network orchestration")'
    )
    assert assess(source, body)["eligible"]
    for bad in (body.replace(a, "SWAP").replace(b, a).replace("SWAP", b),
                body.replace("aiIcon,aiText", "aiText"), body.replace(b, a),
                body.replace("[heading,ai,cloud]", "[heading,ai,cloud,extra]") + f'\nextra=Icon(url="{a}")',
                body.replace("Models and agents", "Models and agents v1.11.3")):
        assert not assess(source, bad)["eligible"]


def test_qualified_link_section_preserves_both_labels_for_a_shared_destination():
    url = "https://example.org/guide#main"
    source = f"## Nutrition & Health Sources\n- First source: <{url}>\n- Second source: <{url}>"
    body = (
        'root=Column([heading,sourceA,sourceB])\nheading=Text("Nutrition & Health Sources","h2")\n'
        'sourceA=Card([sourceLinksA])\nsourceB=Card([sourceLinksB])\n'
        f'sourceLinksA=List(items=[{{label:"First source",url:"{url}"}}])\n'
        f'sourceLinksB=List(items=[{{label:"Second source",url:"{url}"}}])'
    )
    assert assess(source, body)["eligible"]
    assert not assess(source, body.replace("[heading,sourceA,sourceB]", "[heading,sourceA]"))["eligible"]
    assert not assess(source, body.replace("#main", "#other", 1))["eligible"]
    invalid_source = source.replace(f"<{url}>", f"<{url} />", 1)
    assert "pilot_invalid_source_link_destination:line_2" in assess(invalid_source, body)["reasons"]


def test_comma_constraints_stay_literal_but_repeated_named_records_need_binding():
    source = ("## Trip constraints\n- Departure: 2035-11-02, Return: 2035-11-12\n"
              "- Passengers: 1 adult, cabin: economy")
    body = (
        'root=Column([heading,fields])\nheading=Text("Trip constraints","h2")\n'
        'fields=List(items=["Departure: 2035-11-02, Return: 2035-11-12","Passengers: 1 adult, cabin: economy"])'
    )
    result = assess(source, body)
    assert result["eligible"]
    assert result["entity_ownership"]["eligible_record_count"] == 0
    assert not assess(source, body.replace("11-12", "11-13"))["eligible"]

    second_section = '\nother=Text("Travel notes","h2")\nnote=Text("Carry printed copies.")'
    source += "\n## Travel notes\nCarry printed copies."
    assert assess(source, body.replace("[heading,fields]", "[heading,fields,other,note]") + second_section)["eligible"]
    moved = body.replace("[heading,fields]", "[heading,other,fields,note]") + second_section
    assert not assess(source, moved)["eligible"]

    source = "- Alpha Flight, Time: 07:30, Gate: A2\n- Beta Flight, Time: 09:40, Gate: B3"
    body = 'root=List(items=["Alpha Flight, Time: 07:30, Gate: A2","Beta Flight, Time: 09:40, Gate: B3"])'
    assert "entity_fields_not_structured" in assess(source, body)["reasons"]


def test_inline_math_matches_rendered_spelling_without_changing_operators_or_code():
    from pipeline.genui_quality.pilot_layout_v2 import inline_math_spelling

    source = r"Since $5 \times 5 = 25$, the ladder is 5 feet."
    body = 'root=Text("Since 5 × 5 = 25, the ladder is 5 feet.")'
    assert assess(source, body)["eligible"]
    for bad in (body.replace("×", "+"), body.replace("25", "20"), body.replace("feet", "meters")):
        assert not assess(source, bad)["eligible"]
    assert inline_math_spelling(r"Press $\pm 15$ seconds.") == "Press ± 15 seconds."
    raw = r"Budget $5 or $10. Code: `$5 \times 5 = 25$`." + "\n```\n" + r"$\pm 15$" + "\n```"
    assert inline_math_spelling(raw) == raw


def test_numeric_chart_points_are_exact_and_text_keeps_its_display_precision():
    source = (
        "Chart Title: Fuel Price vs. MPG\nX-Axis: Fuel Price ($)\nY-Axis: MPG\n\n"
        "| Price | MPG |\n| --- | --- |\n| 3.40 | 30 |\n| 3.60 | 26 |"
    )
    body = (
        'root=Column([plot,table])\n'
        'plot=Chart(chartType="scatter",columns=["price","mpg"],rows=[{price:3.4,mpg:30},{price:3.6,mpg:26}],'
        'title="Fuel Price vs. MPG",xKey="price",yKey="mpg",xType="number",xLabel="Fuel Price ($)",yLabel="MPG")\n'
        'table=Table(columns=["Price","MPG"],rows=[["3.40","30"],["3.60","26"]])'
    )
    assert assess(source, body)["eligible"]
    swapped = body.replace("3.4,mpg:30", "3.4,mpg:26").replace("3.6,mpg:26", "3.6,mpg:30")
    assert "fidelity:source_chart_data_mismatch" in assess(source, swapped)["reasons"]
    assert not assess(source, body.replace('"3.40"', '"3.4"'))["eligible"]
    extra = body.replace("[plot,table]", "[plot,table,extra]") + '\nextra=Text("3.4")'
    assert "fidelity:unsupported_visible_numeric_anchor" in assess(source, extra)["reasons"]


def test_procedure_mixed_units_fall_back_to_text_and_unsupported_spacing_still_holds():
    source = "0.6 km ≈ 1.2 min, 0.4 km ≈ 0.8 min."
    assert assess(source, "root=Text(" + json.dumps(source, ensure_ascii=False) + ")")["eligible"]
    unsupported = r"0.6\text{ km} ≈ 1.2\text{ min},\quad 0.4\text{ km} ≈ 0.8\text{ min}."
    result = assess(source, "root=Formula(latex=" + json.dumps(unsupported, ensure_ascii=False) + ")")
    assert "renderer_evidence_incomplete" in result["reasons"]
