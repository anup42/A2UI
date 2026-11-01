"""Opt-in evidence parity for the reviewed OpenAI pilot (CPU only)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.genui_quality import generation_reward_a2ui_express_v1 as score
from pipeline.genui_quality.config_v5_4 import RewardConfigV54
from pipeline.genui_quality.evidence_v5_4 import collect_output_evidence_v5_4
from pipeline.genui_quality.formula_evidence_v2 import normalize_formula_evidence_v2
from pipeline.genui_quality.graph import audit_renderer_graph
from pipeline.genui_quality.identity_v5_4 import metric_fingerprint_v5_4


def node(kind, props=None, children=(), **extra):
    return {"type": kind, "props": props or {}, "children": list(children), **extra}


def collect(elements, state=None, config=None):
    spec = {"root": "root", "state": state or {}, "elements": elements}
    return collect_output_evidence_v5_4(
        spec, audit_renderer_graph(spec), config or RewardConfigV54(pilot_fidelity_v2=True)
    )


def test_pilot_is_explicit_boolean_and_identity_distinct():
    legacy = RewardConfigV54()
    assert legacy.pilot_fidelity_v2 is False
    assert RewardConfigV54.from_mapping({}).pilot_fidelity_v2 is False
    assert RewardConfigV54.from_mapping({"policies": {"pilot_fidelity_v2": True}}).pilot_fidelity_v2
    with pytest.raises(ValueError, match="pilot_fidelity_v2"):
        RewardConfigV54(pilot_fidelity_v2="true")
    assert metric_fingerprint_v5_4(legacy) != metric_fingerprint_v5_4(replace(legacy, pilot_fidelity_v2=True))


def test_legacy_alert_evidence_is_unchanged():
    elements = {"root": node("Alert", {"title": "Notice", "message": "Keep the door closed."})}
    default = collect(elements, config=RewardConfigV54())
    explicit = collect(elements, config=RewardConfigV54(pilot_fidelity_v2=False))
    assert default == explicit
    assert default.output.visible_blocks == []
    assert default.output.headings == []


def test_source_cued_card_without_links_collects_native_fallback_headers():
    elements = {
        "root": node("Card", {"title": "Sources: Safety", "subtitle": "Door policy"}, ["body"]),
        "body": node("Text", {"text": "Keep the door closed."}),
    }
    result = collect(elements)
    assert result.dynamic_evidence_complete
    assert result.output.visible_blocks == ["Sources: Safety", "Door policy", "Keep the door closed."]
    legacy = collect(elements, config=RewardConfigV54())
    assert legacy == collect(elements, config=RewardConfigV54(pilot_fidelity_v2=False))
    assert legacy.output.visible_blocks == ["Keep the door closed."]


@pytest.mark.parametrize("url", [
    "https://example.com/", "https://example.test/", "http://docs.python.org/",
    "https://localhost/", "https://user@docs.python.org/", "https://docs.python.org/%zz",
    "https://docs.python.org/a b", "docs.python.org",
])
def test_uncertified_source_card_url_cannot_hide_headers_with_complete_evidence(url):
    result = collect({
        "root": node("Card", {"title": "Sources: invented guarantee"}, ["links"]),
        "links": node("List", {"items": [{"label": "Python docs", "url": url}]}),
    })
    assert not result.dynamic_evidence_complete


@pytest.mark.parametrize("kind,props,on", [
    ("Text", {"text": "[Python docs](https://docs.python.org/3/#intro)"}, {}),
    ("List", {"items": [{"label": "Python docs", "url": "https://docs.python.org/3/#intro"}]}, {}),
    ("Button", {"label": "Python docs"}, {"press": {"action": "openUrl", "params": {"url": "https://docs.python.org/3/#intro"}}}),
])
def test_valid_native_source_card_interception_still_hides_declared_header(kind, props, on):
    result = collect({
        "root": node("Card", {"title": "Sources: hidden declared header"}, ["links"]),
        "links": node(kind, props, on=on),
    })
    assert result.dynamic_evidence_complete
    assert "Sources: hidden declared header" not in result.output.visible_blocks


def test_alert_matches_actual_android_blocks_and_not_hidden_props():
    result = collect({"root": node("Alert", {
        "title": " Notice ", "message": " Do not enter. ", "text": "Hidden fallback",
        "timestamp": " 10:30 ", "source": " Station staff ", "tone": "warning",
        "description": "Invented description", "metadata": "Secret 99",
    })})
    assert result.output.visible_blocks == ["Notice", "Do not enter.", "10:30 • Station staff"]
    assert result.output.headings == ["Notice"]
    assert all(row["owner"] == "generic_visible_content" for row in result.evidence_ownership)
    assert all(row["component_type"] == "alert" for row in result.evidence_ownership)
    assert result.dynamic_evidence_complete


@pytest.mark.parametrize("props,expected", [
    ({"text": "Fallback"}, ["Fallback"]),
    ({"message": None, "text": "Fallback"}, ["Fallback"]),
    ({"message": "", "text": "Hidden fallback"}, []),
    ({"message": "  ", "text": "Hidden fallback"}, []),
    ({"message": False, "text": "Hidden fallback"}, ["false"]),
    ({"message": 0, "text": "Hidden fallback"}, ["0"]),
    ({"timestamp": "12:00"}, ["12:00"]),
    ({"source": "Station staff"}, ["Station staff"]),
])
def test_alert_null_only_precedence_and_kotlin_strings(props, expected):
    assert collect({"root": node("Alert", props)}).output.visible_blocks == expected


def test_alert_visibility_bindings_reachability_and_children():
    result = collect({
        "root": node("Column", children=["alert", "hidden"]),
        "alert": node("Alert", {"message": {"$state": "/message"}}, ["child"]),
        "child": node("Text", {"text": "Child detail"}),
        "hidden": node("Alert", {"message": "Hidden detail"}, visible=False),
        "orphan": node("Alert", {"message": "Orphan detail"}),
    }, state={"message": "Bound content", "unused": "Unused content"})
    assert result.output.visible_blocks == ["Bound content", "Child detail"]


def test_alert_respects_dynamic_evidence_budget():
    result = collect({"root": node("Alert", {"message": "Long content"})},
                     config=RewardConfigV54(pilot_fidelity_v2=True, max_dynamic_evidence_bytes=5))
    assert result.truncated
    assert not result.dynamic_evidence_complete
    assert "dynamic_evidence_byte_budget" in result.unknown_diagnostics


def test_rendered_heading_marker_is_opt_in_and_never_marks_body_rows():
    elements = {
        "root": node("Column", children=["heading", "alert", "plain"]),
        "heading": node("Text", {"text": "Heading", "variant": "h2"}),
        "alert": node("Alert", {"title": "Notice", "message": "Body"}),
        "plain": node("Text", {"text": "Plain body"}),
    }
    rows = collect(elements).evidence_ownership
    assert [row["text"] for row in rows if row.get("rendered_heading")] == ["Heading", "Notice"]
    assert all("rendered_heading" not in row for row in collect(elements, config=RewardConfigV54()).evidence_ownership)


def test_correct_alert_is_credited_and_invented_alert_is_penalized():
    source = "Keep the door closed."
    correct = '<a2ui>\nroot=Alert(message="Keep the door closed.")\n</a2ui>'
    invented = '<a2ui>\nroot=Column([body,notice])\nbody=Text("Keep the door closed.")\nnotice=Alert(message="The moon is made of cheese.")\n</a2ui>'
    config = RewardConfigV54(pilot_fidelity_v2=True)
    good = score(correct, source, config=config)
    bad = score(invented, source, config=config)
    assert good.evidence["training_acceptance"]["eligible"]
    assert not bad.evidence["training_acceptance"]["eligible"]
    assert bad.atomics["fidelity"]["content_unit_fidelity"] < 1


@pytest.mark.parametrize("raw,expected", [
    (r"\sum_{i=1}^{n} x_i \approx 1{,}234", "∑_{i=1}^{n} x_i ≈ 1,234"),
    (r"\mathrm{Total} = \$1{,}234\,\text{USD}", "Total = $1,234 USD"),
    (r"\begin{aligned}x &= 7 \\ y &= 9\end{aligned}", "x = 7\ny = 9"),
    (r"x \rightarrow y", "x → y"),
    (r"\text{Do not \mathrm{enter}}", "Do not enter"),
    (r"\times\text{weight}", "×weight"),
    (r"\begin{aligned}\text{R&D} &= 7 \\ y &= 9\end{aligned}", "R&D = 7\ny = 9"),
    (r"\begin{aligned}x &= 7", r"\begin{aligned}x &= 7"),
    (r"\begin{aligned}\begin{aligned}x &= 7\end{aligned}\end{aligned}", r"\begin{aligned}\begin{aligned}x &= 7\end{aligned}\end{aligned}"),
    (r"\fraction{7}{9} + \unknown{5}", r"\fraction{7}{9} + \unknown{5}"),
    (r"\text{unfinished", r"\text{unfinished"),
    (r"\begin{unknown}x=7\end{unknown}", r"\begin{unknown}x=7\end{unknown}"),
    (r"\frac{7}{9}", r"\frac{7}{9}"),
])
def test_formula_supported_spellings_preserve_structure(raw, expected):
    assert normalize_formula_evidence_v2(raw) == expected


@pytest.mark.parametrize("left,right", [
    (r"x=1{,}234", r"x=1{,}235"),
    (r"\text{Do not enter}", r"\text{Do enter}"),
    (r"x \approx 7", "x = 7"),
    (r"\frac{7}{9}", r"\frac{9}{7}"),
    (r"x^2", r"x_2"),
    (r"7\text{ USD}", r"7\text{ EUR}"),
])
def test_formula_normalization_never_erases_semantic_changes(left, right):
    assert normalize_formula_evidence_v2(left) != normalize_formula_evidence_v2(right)


def test_formula_evidence_precedence_and_ancillary_prose():
    props = {"latex": r"x=1{,}234", "formula": "wrong 8", "text": "wrong 9", "value": "wrong 10",
             "title": "Total", "subtitle": "Before tax", "result": "1234 USD"}
    result = collect({"root": node("Formula", props)})
    assert result.output.visible_blocks == ["Total", "Before tax", "x=1,234", "1234 USD"]
    assert "1 234" in result.output.exact_values
    assert "234" not in result.output.exact_values
    assert "8" not in result.output.exact_values
    assert all(item["owner"] == "source_formula" for item in result.evidence_ownership)
    legacy = collect({"root": node("Formula", props)}, config=RewardConfigV54())
    assert legacy.output.visible_blocks == ["wrong 9", r"x=1{,}234", "wrong 8", "Total"]


def test_blank_formula_does_not_credit_headers_result_or_fallback():
    result = collect({"root": node("Formula", {"latex": "", "text": "Hidden 7", "title": "Hidden title", "result": "Hidden 9"})})
    assert not result.output.visible_blocks
    assert not result.role_instances.get("formula")


def test_source_list_map_uses_native_label_precedence_only_in_pilot():
    elements = {
        "root": node("Card", {"title": "Sources"}, ["links"]),
        "links": node("List", {"items": [{"text": "Hidden prose", "label": "Python docs", "title": "Hidden title",
                                            "description": "Hidden description", "url": "https://docs.python.org/3/"}]}),
    }
    result = collect(elements)
    assert result.output.visible_blocks == ["Python docs"]
    assert [(a.label, a.url, a.source) for a in result.output.actions] == [("Python docs", "https://docs.python.org/3/", "source_link")]
    assert collect(elements, config=RewardConfigV54()).output.actions[0].label == "Hidden prose"


def test_source_map_ambiguous_destinations_fail_closed_without_hiding_later_url():
    result = collect({
        "root": node("Card", {"title": "Sources"}, ["links"]),
        "links": node("List", {"items": [{"label": "Docs", "url": "https://example.com/x", "href": "https://docs.python.org/3/"}]}),
    })
    assert not result.dynamic_evidence_complete
    assert "pilot_source_link_destination_ambiguity:links[0]" in result.unknown_diagnostics
    assert {action.url for action in result.output.actions} == {"https://example.com/x", "https://docs.python.org/3"}


def test_fragment_only_map_destination_ambiguity_is_not_normalized_away():
    result = collect({
        "root": node("Card", {"title": "Sources"}, ["links"]),
        "links": node("List", {"items": [{"label": "Docs", "url": "https://docs.python.org/3/#intro", "href": "https://docs.python.org/3/#other"}]}),
    })
    assert not result.dynamic_evidence_complete
    assert "pilot_source_link_destination_ambiguity:links[0]" in result.unknown_diagnostics


def test_raw_action_destination_is_restored_from_resolved_event_only():
    result = collect({
        "root": node("Button", {"label": "Read", "url": "https://invented.org/"},
                     on={"press": {"action": "openUrl", "params": {"url": {"$state": "/target"}}}}),
    }, state={"target": "https://docs.python.org/3/?v=3#intro", "unused": "https://invented.org/"})
    assert result.dynamic_evidence_complete
    assert [action.url for action in result.output.actions] == ["https://docs.python.org/3/?v=3#intro"]


@pytest.mark.parametrize("ancestor_id,props", [("sources", {}), ("container", {"title": "Sources"})])
def test_unused_ancestor_source_cues_do_not_classify_plain_list_links(ancestor_id, props):
    elements = {
        "root": node("Column", children=[ancestor_id]),
        ancestor_id: node("Column", props, ["links"]),
        "links": node("List", {"items": [{"label": "Docs", "url": "https://docs.python.org/3/"}]}),
    }
    assert collect(elements).output.actions[0].source == "list.item"
    assert collect(elements, config=RewardConfigV54()).output.actions[0].source == "source_link"


def test_source_card_context_does_not_leak_into_reused_list_occurrence():
    elements = {
        "root": node("Column", children=["sourceCard", "ordinary"]),
        "sourceCard": node("Card", {"title": "Sources"}, ["links"]),
        "ordinary": node("Column", children=["links"]),
        "links": node("List", {"items": [{"label": "Docs", "url": "https://docs.python.org/3/"}]}),
    }
    assert [action.source for action in collect(elements).output.actions] == ["source_link", "list.item"]


def test_source_list_map_skips_url_label_and_deduplicates_destination():
    result = collect({
        "root": node("Card", {"title": "Sources"}, ["links"]),
        "links": node("List", {"items": [
            {"label": "https://docs.python.org/3/", "title": "Python docs", "url": "https://docs.python.org/3/"},
            {"label": "Duplicate", "url": "https://docs.python.org/3/"},
        ]}),
    })
    assert result.output.visible_blocks == ["Python docs"]
    assert len(result.output.actions) == 1


def test_source_map_without_links_retains_ordinary_list_content():
    result = collect({
        "root": node("Card", {"title": "Sources"}, ["links"]),
        "links": node("List", {"items": [{"text": "No links", "description": "Ordinary content"}]}),
    })
    assert result.output.visible_blocks == ["Sources", "No links", "Ordinary content"]
