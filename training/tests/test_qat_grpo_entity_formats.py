"""Exact source-bound extraction; legacy/default scoring stays opt-out."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))

from ir_training.train.qat_grpo_reward import make_qat_grpo_reward
from pipeline.genui_quality._v5_4 import (
    generation_reward_v5_4,
    prepare_source_context_v5_4,
)
from pipeline.genui_quality.config_v5_4 import load_v5_4_reward_config
from pipeline.genui_quality.grpo_entity_content import (
    GRPO_ENTITY_POLICY_VERSION,
    audit_grpo_entity_representation,
    extract_grpo_entity_records,
    summarize_grpo_entity_detail,
)

HOTELS = (
    ("Cedar", "Pune", "INR 4200/night", "4.6/5"),
    ("Elm", "Kochi", "INR 5300/night", "4.8/5"),
)
SOURCE = "Hotels:\n" + "\n".join(
    f"- Hotel: {name}; City: {city}; Price: {price}; Rating: {rating}"
    for name, city, price, rating in HOTELS
)


def cards(rows=HOTELS):
    lines = ["root=Column([heading,c0,c1])", 'heading=Text("Hotels","h2")']
    for i, (name, city, price, rating) in enumerate(rows):
        lines += [f'c{i}=Card([f{i}1,f{i}2,f{i}3],title="Hotel: {name}")']
        lines += [f"f{i}{j}=Text({json.dumps(text)})" for j, text in enumerate(
            (f"City: {city}", f"Price: {price}", f"Rating: {rating}"), 1)]
    return "<a2ui>" + "\n".join(lines) + "</a2ui>"


def copied(source=SOURCE):
    items = [line[2:] for line in source.splitlines() if line.startswith("- ")]
    return ('<a2ui>root=Column([h,l])\nh=Text("Hotels","h2")\nl=List(items='
            + json.dumps(items) + ")</a2ui>")


def table(rows=HOTELS, labels=("Hotel", "City", "Price", "Rating"), *, cards_mode=True):
    props = ',presentation="cards",domain="booking"' if cards_mode else ""
    return ('<a2ui>root=Column([h,t])\nh=Text("Hotels","h2")\nt=Table(columns='
            + json.dumps(labels) + ",rows=" + json.dumps(rows) + props + ")</a2ui>")


def scores(candidates, source=SOURCE):
    extra = {}
    values = make_qat_grpo_reward()(candidates, source,
                                   log_extra=lambda key, value: extra.update({key: value}))
    return values, extra["qat_reward"]


def test_exact_production_hotel_example_prefers_extraction_over_copy():
    values, results = scores([cards(), copied()])
    assert values[0] > values[1]
    assert results[0]["components"]["source_conditioned_extraction"]["coverage"] == 1
    assert results[1]["components"]["source_conditioned_extraction"]["coverage"] == 0
    audit = audit_grpo_entity_representation(SOURCE, cards())
    assert audit["policy_version"] == GRPO_ENTITY_POLICY_VERSION == "2.0.0"
    assert audit["eligible_record_count"] == audit["structured_record_count"] == 2
    assert audit["binding_complete"] is True
    assert all(row["original_source_units"] for row in audit["certified_records"])


@pytest.mark.parametrize("separator", ["; ", " | ", ", "])
def test_labeled_delimiters_preserve_exact_values(separator):
    source = SOURCE.replace("; ", separator).replace("4200", "4,200")
    candidate = cards().replace("4200", "4,200")
    values, results = scores([candidate, copied(source)], source)
    assert values[0] > values[1]
    assert results[0]["components"]["source_fidelity"] == 1
    assert results[0]["components"]["source_conditioned_extraction"]["coverage"] == 1


@pytest.mark.parametrize("bad", [
    lambda text: text.replace("INR 4200/night", "INR 5300/night").replace(
        'f12=Text("Price: INR 5300/night")', 'f12=Text("Price: INR 4200/night")'),
    lambda text: text.replace("f01,f02,f03", "f01,f02"),
    lambda text: text.replace("INR 4200/night", "INR 4200"),
    lambda text: text.replace("f01,f02,f03", "f01,f02,f03,x") + "",
    lambda text: text.replace("f01,f02,f03", "f01,f02,f03,f03"),
])
def test_wrong_owner_missing_qualifier_or_duplicate_never_get_full_credit(bad):
    good = cards()
    broken = bad(good)
    if ",x]" in broken:
        broken = broken.replace("</a2ui>", '\nx=Text("Breakfast included")</a2ui>')
    values, results = scores([good, broken])
    assert values[1] < values[0]
    assert results[1]["components"].get("source_conditioned_extraction", {}).get("coverage", 0) < 1
    assert not audit_grpo_entity_representation(SOURCE, broken)["binding_complete"]


@pytest.mark.parametrize("cards_mode", [False, True])
def test_complete_compact_table_fields_are_certified(cards_mode):
    candidate = table(cards_mode=cards_mode)
    values, results = scores([candidate, copied()])
    assert values[0] > values[1]
    assert results[0]["strict_valid"]
    assert results[0]["components"]["source_fidelity"] == 1
    assert results[0]["components"]["precision_guard"] == 1
    assert results[0]["components"]["source_conditioned_extraction"]["coverage"] == 1
    assert all("table_instance" in row for row in
               audit_grpo_entity_representation(SOURCE, candidate)["certified_records"])


@pytest.mark.parametrize("rows,labels", [
    ((HOTELS[0], HOTELS[0]), ("Hotel", "City", "Price", "Rating")),
    ((HOTELS[0],), ("Hotel", "City", "Price", "Rating")),
    ((HOTELS[0][:2] + (HOTELS[1][2], HOTELS[0][3]), HOTELS[1][:2]
      + (HOTELS[0][2], HOTELS[1][3])), ("Hotel", "City", "Price", "Rating")),
    (tuple((*row, "Breakfast included") for row in HOTELS),
     ("Hotel", "City", "Price", "Rating", "Benefit")),
    (HOTELS, ("Hotel", "City", "Price", "Price")),
    (tuple(row[:2] + (row[2].replace("/night", ""), row[3]) for row in HOTELS),
     ("Hotel", "City", "Price", "Rating")),
])
def test_table_missing_swapped_duplicate_extra_fields_fail_closed(rows, labels):
    good, broken = table(), table(rows, labels)
    values, results = scores([good, broken])
    assert values[1] < values[0]
    assert results[1]["components"].get("source_conditioned_extraction", {}).get("coverage", 0) < 1
    assert not audit_grpo_entity_representation(SOURCE, broken)["binding_complete"]


def test_table_column_reordering_keeps_labels_bound_to_values():
    order = (3, 0, 2, 1)
    labels = ("Hotel", "City", "Price", "Rating")
    candidate = table(tuple(tuple(row[i] for i in order) for row in HOTELS),
                      tuple(labels[i] for i in order))
    assert audit_grpo_entity_representation(SOURCE, candidate)["binding_complete"]


def test_heading_plus_subfields_preserve_owner_and_reject_swaps_or_extra_claims():
    source = "## Hotel Aurora\n- Price: INR 5000\n- Rating: 4.5\n\n## Hotel Cedar\n- Price: INR 4000\n- Rating: 4.2"
    good = ('<a2ui>root=Column([a,b])\na=Card([ah,ap,ar])\n'
            'ah=Text("Hotel Aurora","h2")\nap=Text("Price: INR 5000")\nar=Text("Rating: 4.5")\n'
            'b=Card([bh,bp,br])\nbh=Text("Hotel Cedar","h2")\nbp=Text("Price: INR 4000")\n'
            'br=Text("Rating: 4.2")</a2ui>')
    swapped = good.replace("5000", "TEMP").replace("4000", "5000").replace("TEMP", "4000")
    invented = good.replace("[ah,ap,ar]", "[ah,ap,ar,x]").replace(
        "</a2ui>", '\nx=Text("Breakfast included")</a2ui>')
    audits = [audit_grpo_entity_representation(source, item) for item in (good, swapped, invented)]
    assert audits[0]["binding_complete"] is True
    assert audits[0]["structured_record_count"] == 2
    assert all(row["source_format"] == "heading_fields" for row in audits[0]["certified_records"])
    assert all(item["binding_complete"] is False for item in audits[1:])
    values, _ = scores([good, swapped, invented], source)
    assert values[0] > max(values[1:])


@pytest.mark.parametrize("source", [
    "- Hotel: Cedar; City: Pune; Price: 4000\n- Hotel: Cedar; City: Goa; Price: 5000",
    "- Hotel: Cedar; City: Pune; City: Goa",
    "- Step 1: Open settings; Status: Complete",
    "## Hotels\n- Open settings\n- Enable notifications",
    "## Packing\n- Shoes: 2 pairs\n- Chargers: 1",
    "## Cedar\n- Price: 4000\n- Rating: 4.5\n## Elm\n- Price: 4000\n- Rating: 4.5",
    "- Hotel Cedar, Pune, 4000, 4.5",
    '- Hotel: Cedar, Note: "pending, Rating: uncertain", City: Pune',
    "- Hotel: Cedar, Note: pending (today, Rating: uncertain), City: Pune",
])
def test_ambiguous_duplicate_or_plain_sources_stay_inapplicable(source):
    base = prepare_source_context_v5_4(source).base
    assert extract_grpo_entity_records(base.applicability.generic_source_units,
                                       source_text=source) == ()
    audit = audit_grpo_entity_representation(source, copied(source))
    assert audit["applicable"] is False
    assert audit["binding_complete"] is None


def test_opt_out_and_ordinary_checklist_behavior_remain_identical():
    cfg = load_v5_4_reward_config()
    for source, candidate in [(SOURCE, cards()), (SOURCE, table()),
                              ("- Open settings\n- Enable notifications",
                               '<a2ui>root=List(items=["Open settings","Enable notifications"])</a2ui>')]:
        baseline = generation_reward_v5_4(candidate, source)
        explicit = generation_reward_v5_4(candidate, source, config=replace(cfg, grpo_entity_extraction=False))
        assert baseline.atomics == explicit.atomics
        assert baseline.reward == explicit.reward
    source, candidate = "- Open settings\n- Enable notifications", '<a2ui>root=List(items=["Open settings","Enable notifications"])</a2ui>'
    baseline = generation_reward_v5_4(candidate, source)
    enabled = generation_reward_v5_4(candidate, source, config=replace(cfg, grpo_entity_extraction=True))
    assert baseline.reward == enabled.reward
    assert baseline.atomics == enabled.atomics


@pytest.mark.parametrize("before,after", [
    ("-5 C", "5 C"), ("5%", "5"), ("USD 4.20/day", "USD 420/day"),
    ("not included", "included"), ("estimated 5 days", "5 days"),
])
@pytest.mark.parametrize("kind", ["card", "table"])
def test_certification_never_discards_value_signs_units_or_qualifiers(before, after, kind):
    source = f"- Hotel: Cedar; Offer: {before}; City: Pune"
    if kind == "card":
        good = ('<a2ui>root=Card([a,b],title="Hotel: Cedar")\na=Text('
                + json.dumps("Offer: " + before) + ')\nb=Text("City: Pune")</a2ui>')
    else:
        good = ('<a2ui>root=Table(columns=["Hotel","Offer","City"],rows='
                + json.dumps([["Cedar", before, "Pune"]]) + ')</a2ui>')
    bad = good.replace(before, after)
    assert audit_grpo_entity_representation(source, good)["binding_complete"] is True
    assert audit_grpo_entity_representation(source, bad)["binding_complete"] is False


def test_shared_values_stay_bound_to_distinct_entity_and_field_names():
    rows = (("Cedar", "Pune", "INR 4200/night", "4.6/5"),
            ("Elm", "Pune", "INR 4200/night", "4.6/5"))
    source = "Hotels:\n" + "\n".join(
        f"- Hotel: {name}; City: {city}; Price: {price}; Rating: {rating}"
        for name, city, price, rating in rows)
    assert audit_grpo_entity_representation(source, cards(rows))["binding_complete"] is True
    assert audit_grpo_entity_representation(source, table(rows))["binding_complete"] is True
    assert audit_grpo_entity_representation(source, table((rows[0], rows[0])))["binding_complete"] is False


def test_graph_input_is_an_explicit_error_not_an_inapplicable_source():
    with pytest.raises(TypeError, match="canonical scoring text"):
        audit_grpo_entity_representation(SOURCE, {"root": "r", "elements": {}})


def test_trusted_detail_summary_reuses_the_same_bound_certificates():
    result = generation_reward_v5_4(cards(), SOURCE, config=replace(
        load_v5_4_reward_config(), grpo_entity_extraction=True))
    assert summarize_grpo_entity_detail(result.evidence["grpo_entity_extraction"]) == (
        audit_grpo_entity_representation(SOURCE, cards()))


def test_linked_table_cannot_drop_source_owned_actions():
    source = SOURCE.replace("4.6/5", "4.6/5; [Book](https://example.test/cedar)").replace(
        "4.8/5", "4.8/5; [Book](https://example.test/elm)")
    audit = audit_grpo_entity_representation(source, table())
    assert audit["eligible_record_count"] == audit["linked_record_count"] == 2
    assert audit["bound_record_count"] == 0
    assert audit["binding_complete"] is False


def test_table_projection_does_not_hide_additional_global_claims():
    good = table()
    bad = good.replace("[h,t]", "[h,t,x]").replace(
        "</a2ui>", '\nx=Text("Breakfast included for both hotels")</a2ui>')
    values, results = scores([good, bad])
    assert values[1] < values[0]
    assert "source_conditioned_extraction" not in results[1]["components"]
    assert results[1]["components"]["precision_guard"] < 1
