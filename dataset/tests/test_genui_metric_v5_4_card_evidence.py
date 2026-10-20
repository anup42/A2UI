"""Renderer-visible Card evidence and per-render occurrence boundaries."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.genui_quality.config_v5_4 import RewardConfigV54
from pipeline.genui_quality.evidence_v5_3 import collect_output_evidence_v5_3
from pipeline.genui_quality.evidence_v5_4 import collect_output_evidence_v5_4
from pipeline.genui_quality.graph import audit_renderer_graph
from pipeline.genui_quality.identity_v5_4 import (
    metric_fingerprint_v5_4,
    score_source_manifest_v5_4,
)


def node(kind, props=None, children=(), **extra):
    return {"type": kind, "props": props or {}, "children": list(children), **extra}


def collect(elements, state=None, config=None, *, legacy=False):
    spec = {"root": "root", "state": state or {}, "elements": elements}
    collector = collect_output_evidence_v5_3 if legacy else collect_output_evidence_v5_4
    return collector(spec, audit_renderer_graph(spec), config or RewardConfigV54())


def test_card_title_subtitle_are_visible_in_render_order_not_arbitrary_props():
    result = collect({
        "root": node("Card", {"title": "Hotel A", "subtitle": "City: Paris", "metadata": "Hidden 99"}, ["price"]),
        "price": node("Text", {"text": "Price: 120 EUR"}),
    }, state={"unused": "Invented 42"})
    assert result.output.visible_blocks == ["Hotel A", "City: Paris", "Price: 120 EUR"]
    assert result.generic_content_tokens["99"] == 0
    assert result.generic_content_tokens["42"] == 0
    rows = result.evidence_ownership
    assert len({row["card_instance"] for row in rows}) == 1
    assert [row["component_type"] for row in rows] == ["card", "card", "text"]


def test_repeat_card_instances_resolve_fields_without_merging_entities():
    result = collect({
        "root": node("Column", children=["entry"], repeat={"statePath": "/hotels"}),
        "entry": node("Card", {"title": {"$item": "name"}, "subtitle": {"$item": "city"}}, ["price"]),
        "price": node("Text", {"text": {"$item": "price"}}),
    }, state={"hotels": [
        {"name": "Hotel A", "city": "Paris", "price": "120 EUR"},
        {"name": "Hotel B", "city": "Rome", "price": "90 EUR"},
    ]})
    assert result.output.visible_blocks == ["Hotel A", "Paris", "120 EUR", "Hotel B", "Rome", "90 EUR"]
    rows = result.evidence_ownership
    assert len({row["card_instance"] for row in rows[:3]}) == 1
    assert len({row["card_instance"] for row in rows[3:]}) == 1
    assert rows[0]["card_instance"] != rows[3]["card_instance"]
    assert result.dynamic_evidence_complete


def test_shared_child_keeps_two_visible_occurrences_with_distinct_card_owners():
    result = collect({
        "root": node("Column", children=["a", "b"]),
        "a": node("Card", children=["shared"]),
        "b": node("Card", children=["shared"]),
        "shared": node("Text", {"text": "Duplicated 17"}),
    })
    assert result.output.visible_blocks == ["Duplicated 17", "Duplicated 17"]
    assert result.generic_content_tokens["duplicated"] == 2
    assert len({row["card_instance"] for row in result.evidence_ownership}) == 2


def test_nested_cards_keep_nearest_card_boundary_and_ancestry():
    result = collect({
        "root": node("Card", {"title": "Hotels"}, ["a", "b"]),
        "a": node("Card", {"title": "Paris"}, ["price_a"]),
        "b": node("Card", {"title": "Rome"}, ["price_b"]),
        "price_a": node("Text", {"text": "120 EUR"}),
        "price_b": node("Text", {"text": "90 EUR"}),
    })
    rows = result.evidence_ownership
    assert len({row["card_instance"] for row in rows}) == 3
    assert rows[1]["card_instance"] == rows[2]["card_instance"]
    assert rows[3]["card_instance"] == rows[4]["card_instance"]
    assert rows[1]["card_ancestors"] == (rows[0]["card_instance"], rows[1]["card_instance"])


def test_hidden_card_and_hidden_fields_never_enter_evidence():
    result = collect({
        "root": node("Column", children=["hidden", "shown"]),
        "hidden": node("Card", {"title": "Secret 99"}, ["secret"], visible=False),
        "secret": node("Text", {"text": "Hidden data"}),
        "shown": node("Card", {"title": {"$state": "/name"}}, ["field"]),
        "field": node("Text", {"text": "Secret 42"}, visible=False),
    }, state={"name": "Hotel A"})
    assert result.output.visible_blocks == ["Hotel A"]
    assert len(result.evidence_ownership) == 1


def test_card_headers_consume_the_existing_dynamic_evidence_budget():
    result = collect({"root": node("Card", {"title": "x" * 20})},
                     config=RewardConfigV54(max_dynamic_evidence_bytes=10))
    assert result.dynamic_evidence_certification["evidence_bytes"] == 20
    assert result.truncated
    assert not result.dynamic_evidence_complete
    assert "dynamic_evidence_byte_budget" in result.unknown_diagnostics


def test_source_link_card_does_not_credit_intercepted_declared_headers():
    result = collect({
        "root": node("Card", {"title": "Sources", "subtitle": "Secret 99"}, ["links"]),
        "links": node("List", {"items": ["[Docs](https://example.test/a)"]}),
    })
    assert "Secret 99" not in result.output.visible_blocks
    assert result.output.visible_blocks == ["Docs"]


def test_email_card_does_not_credit_intercepted_declared_headers():
    texts = ["Subject: Hello", "To: Pat", "From: Sam", "Dear Pat,", "Thank you"]
    result = collect({
        "root": node("Card", {"title": "Secret 99", "subtitle": "Hidden 42"}, [f"t{i}" for i in range(5)]),
        **{f"t{i}": node("Text", {"text": text}) for i, text in enumerate(texts)},
    })
    assert result.output.visible_blocks == texts


def test_hidden_email_children_still_intercept_declared_card_headers():
    texts = ["Subject: Hello", "To: Pat", "From: Sam", "Dear Pat,", "Thank you"]
    result = collect({
        "root": node("Card", {"title": {"$state": "/secret"}, "subtitle": "Hidden 42"}, [f"t{i}" for i in range(5)]),
        **{f"t{i}": node("Text", {"text": {"$state": f"/texts/{i}"}}, visible=False)
           for i in range(5)},
    }, state={"texts": texts, "secret": "Secret 99"})
    assert result.output.visible_blocks == []
    assert result.evidence_ownership == ()


def test_bound_source_heading_intercepts_declared_card_headers():
    result = collect({
        "root": node("Card", {"title": {"$state": "/secret"}}, ["heading", "links"]),
        "heading": node("Text", {"text": {"$state": "/heading"}}),
        "links": node("List", {"items": ["[Docs](https://example.test/a)"]}),
    }, state={"heading": "Sources", "secret": "Secret 99"})
    assert "Secret 99" not in result.output.visible_blocks
    assert all(row["component_type"] != "card" for row in result.evidence_ownership)


def test_source_table_route_cannot_credit_declared_card_headers():
    result = collect({
        "root": node("Card", {"title": "Secret 99"}, ["table"]),
        "table": node("Table", {"columns": ["Source", "URL"],
                                  "rows": [["Docs", "https://example.test/a"]]}),
    })
    assert "Secret 99" not in result.output.visible_blocks


def test_uncertain_hidden_child_route_withholds_header_credit():
    result = collect({
        "root": node("Card", {"title": "Secret 99"}, ["child"]),
        "child": node("Text", {"text": {"$computed": "unknownRoute"}}, visible=False),
    })
    assert "Secret 99" not in result.output.visible_blocks
    assert not result.dynamic_evidence_complete


def test_legacy_v53_evidence_does_not_change():
    result = collect({
        "root": node("Card", {"title": "Legacy ignored header"}, ["field"]),
        "field": node("Text", {"text": "Visible"}),
    }, legacy=True)
    assert result.output.visible_blocks == ["Visible"]
    assert all("card_instance" not in row for row in result.evidence_ownership)


def test_card_grouping_implementation_is_bound_to_metric_identity():
    assert "card_content_v5_4.py" in score_source_manifest_v5_4()
    config = RewardConfigV54()
    assert metric_fingerprint_v5_4(config) != metric_fingerprint_v5_4(
        config, source_hash_overrides={"card_content_v5_4.py": "different-implementation"},
    )
