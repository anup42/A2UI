"""Source-grounded card-field reward regressions using only synthetic records."""
from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))

from ir_training.train.qat_grpo_reward import make_qat_grpo_reward
from pipeline.genui_quality import generation_reward_v5_4

SOURCE = (
    "- Hotel Aurora: Goa; Price: INR 5000; Rating: 4.5\n"
    "- Hotel Cedar: Pune; Price: INR 4000; Rating: 4.2"
)
RECORDS = (
    ("Hotel Aurora", "Goa", "Price: INR 5000", "Rating: 4.5"),
    ("Hotel Cedar", "Pune", "Price: INR 4000", "Rating: 4.2"),
)


def express(body: str) -> str:
    return f"<a2ui>{body}</a2ui>"


def cards(records=RECORDS, *, titles: bool = True) -> str:
    lines = ["root=Column([" + ",".join(f"c{i}" for i in range(len(records))) + "])"]
    for index, record in enumerate(records):
        fields = record[1:] if titles else record
        children = ",".join(f"c{index}f{field}" for field in range(len(fields)))
        title = f",title={json.dumps(record[0])}" if titles else ""
        lines.append(f"c{index}=Card([{children}]{title})")
        lines.extend(f"c{index}f{field}=Text({json.dumps(value)})"
                     for field, value in enumerate(fields))
    return express("\n".join(lines))


def capture(candidates, source=SOURCE):
    evidence = {}
    values = make_qat_grpo_reward()(
        candidates, source, log_extra=lambda key, value: evidence.update({key: value}),
    )
    return values, evidence["qat_reward"]


@pytest.mark.parametrize("titles", [True, False])
def test_exact_entity_field_extraction_has_full_fidelity_and_high_reward(titles):
    values, rows = capture([cards(titles=titles)])
    assert values[0] > 0.95
    assert rows[0]["strict_valid"]
    components = rows[0]["components"]
    assert components["source_fidelity"] == 1.0
    assert components["precision_guard"] == 1.0
    assert components["critical_fidelity"]["exact_numbers_dates_units_fbeta"] == 1.0
    assert all(value == 1.0 for value in components["applicable_fidelity"].values())
    assert rows[0]["reasons"] == () or rows[0]["reasons"] == []


@pytest.mark.parametrize("records, penalty", [
    ((RECORDS[0][:-1], RECORDS[1]), "content_unit_fidelity"),
    ((RECORDS[0],), "content_unit_fidelity"),
    ((RECORDS[0] + ("Pool included",), RECORDS[1]), "visible_content_multiset_fbeta"),
    ((RECORDS[0] + (RECORDS[0][-1],), RECORDS[1]), "visible_content_multiset_fbeta"),
    ((RECORDS[0], RECORDS[1], RECORDS[0]), "visible_content_multiset_fbeta"),
])
def test_missing_added_and_duplicate_card_facts_remain_penalized(records, penalty):
    values, rows = capture([cards(), cards(records)])
    assert values[0] > values[1]
    assert rows[1]["strict_valid"]
    assert rows[1]["components"]["applicable_fidelity"][penalty] < 1.0
    assert rows[1]["reasons"]


@pytest.mark.parametrize("records", [
    # Global word/value inventories are unchanged, but associations are false.
    (("Hotel Aurora", "Pune", "Price: INR 5000", "Rating: 4.5"),
     ("Hotel Cedar", "Goa", "Price: INR 4000", "Rating: 4.2")),
    (("Hotel Aurora", "Goa", "Price: INR 4000", "Rating: 4.5"),
     ("Hotel Cedar", "Pune", "Price: INR 5000", "Rating: 4.2")),
    (("Hotel Aurora", "Goa", "Price: 4.5", "Rating: INR 5000"), RECORDS[1]),
])
def test_entity_and_label_value_associations_cannot_be_bag_matched(records):
    values, rows = capture([cards(), cards(records)])
    assert values[0] > values[1]
    bad = rows[1]["components"]
    assert bad["applicable_fidelity"]["visible_content_multiset_fbeta"] == 1.0
    assert bad["critical_fidelity"]["exact_numbers_dates_units_fbeta"] == 1.0
    assert bad["applicable_fidelity"]["content_unit_fidelity"] < 1.0
    assert bad["precision_guard"] < 1.0


def test_field_fragments_cannot_be_pooled_across_unrelated_card_occurrences():
    # Each fragment is individually visible, but no card represents one entity.
    fragments = tuple((field,) for record in RECORDS for field in record)
    values, rows = capture([cards(), cards(fragments, titles=False)])
    assert values[0] > values[1]
    assert rows[1]["components"]["applicable_fidelity"]["visible_content_multiset_fbeta"] == 1.0
    assert rows[1]["components"]["applicable_fidelity"]["content_unit_fidelity"] < 1.0


def test_wrapping_all_list_fragments_in_one_card_does_not_earn_grouping_credit():
    wrapped = cards((tuple(field for record in RECORDS for field in record),), titles=False)
    values, rows = capture([cards(), wrapped])
    assert values[0] > values[1]
    assert rows[1]["components"]["applicable_fidelity"]["visible_content_multiset_fbeta"] == 1.0
    assert rows[1]["components"]["applicable_fidelity"]["content_unit_fidelity"] < 1.0


@pytest.mark.parametrize("bad", [
    # Preserve the omitted fact in state, an invisible subtree, or an orphan.
    cards().replace('c0f2=Text("Rating: 4.5")', 'c0f2=Text("Rating: 4.5",visible=false)'),
    cards().replace("[c0f0,c0f1,c0f2]", "[c0f0,c0f1]"),
    cards((RECORDS[0][:-1], RECORDS[1])).replace(
        "<a2ui>", '<a2ui>$/={hidden:"Rating: 4.5"}\n',
    ),
])
def test_hidden_or_unreachable_facts_do_not_repair_visible_extraction(bad):
    values, rows = capture([cards(), bad])
    assert values[0] > values[1]
    assert rows[1]["components"]["applicable_fidelity"]["visible_content_multiset_fbeta"] < 1.0


def test_card_extraction_never_bypasses_raw_express_or_reachability_gates():
    good = cards()
    candidates = [
        "Here is your UI: " + good,
        good + good,
        good.replace("root=Column([c0,c1])", "root=Column([c0,c1,missing])"),
        good.replace("c0=Card", "c0=UnknownCard"),
    ]
    values, rows = capture(candidates)
    assert values == [-1.0] * len(candidates)
    assert all(not row["strict_valid"] for row in rows)


def test_plain_steps_still_receive_full_fidelity_as_a_list():
    source = "- Open settings\n- Select notifications\n- Enable alerts"
    candidate = express('root=List(items=["Open settings","Select notifications","Enable alerts"])')
    values, rows = capture([candidate], source)
    assert values[0] > 0.95
    assert rows[0]["components"]["source_fidelity"] == 1.0
    assert rows[0]["components"]["precision_guard"] == 1.0
    # The unchanged per-block path must remain authoritative outside Cards.
    canonical = generation_reward_v5_4(candidate, source)
    assert rows[0]["canonical_reward"] == canonical.reward
    assert rows[0]["components"]["applicable_fidelity"]["content_unit_fidelity"] == 1.0


def test_simple_non_card_text_keeps_the_existing_reward_mapping():
    source = "Paris is 21 C."
    candidate = express(f"root=Text({json.dumps(source)})")
    values, rows = capture([candidate], source)
    canonical = generation_reward_v5_4(candidate, source)
    # With perfect content and no semantic/critical penalties, the existing
    # structural factor is the only non-unit multiplier; no Card bonus exists.
    expected = 2.0 * (0.95 + 0.05 * canonical.quality_0_1) - 1.0
    assert values[0] == pytest.approx(expected)
    assert rows[0]["components"]["source_fidelity"] == 1.0
    assert rows[0]["components"]["precision_guard"] == 1.0


def repeated_cards(records=RECORDS):
    state = {"hotels": [
        dict(zip(("name", "city", "price", "rating"), row, strict=True))
        for row in records
    ]}
    return express(
        "$/=" + json.dumps(state) + '\n'
        'root=Column([],repeat={statePath:"/hotels",template:entry})\n'
        'entry=Card([price,rating],title="${$item.name}",subtitle="${$item.city}")\n'
        'price=Text("${$item.price}")\nrating=Text("${$item.rating}")'
    )


def test_repeated_card_template_is_scored_per_visible_entity_occurrence():
    candidate = repeated_cards()
    values, rows = capture([candidate])
    assert values[0] > 0.95
    assert rows[0]["strict_valid"]
    assert rows[0]["components"]["source_fidelity"] == 1.0
    assert rows[0]["components"]["precision_guard"] == 1.0
    result = generation_reward_v5_4(candidate, SOURCE)
    grouping = result.evidence["content_assignment"]["card_field_grouping"]
    assert grouping["raw_block_count"] == 8
    assert grouping["assignment_block_count"] == 2
    groups = grouping["certified_groups"]
    assert len(groups) == 2
    assert groups[0]["card_instance"] != groups[1]["card_instance"]
    assert [group["field_count"] for group in groups] == [4, 4]


def test_visible_plain_card_title_and_subtitle_are_preserved_as_source_fields():
    candidate = cards().replace(
        'c0=Card([c0f0,c0f1,c0f2],title="Hotel Aurora")\nc0f0=Text("Goa")',
        'c0=Card([c0f1,c0f2],title="Hotel Aurora",subtitle="Goa")',
    ).replace(
        'c1=Card([c1f0,c1f1,c1f2],title="Hotel Cedar")\nc1f0=Text("Pune")',
        'c1=Card([c1f1,c1f2],title="Hotel Cedar",subtitle="Pune")',
    )
    values, rows = capture([candidate])
    assert values[0] > 0.95
    assert rows[0]["components"]["source_fidelity"] == 1.0
    assert rows[0]["components"]["precision_guard"] == 1.0
    wrong_subtitle = candidate.replace('subtitle="Goa"', 'subtitle="Pune"')
    # A stale subtitle is a visible false fact, not harmless Card metadata.
    changed, diagnostics = capture([wrong_subtitle])
    assert changed[0] < values[0]
    assert diagnostics[0]["components"]["applicable_fidelity"]["visible_content_multiset_fbeta"] < 1.0


def test_outer_card_wrapper_keeps_each_inner_card_entity_boundary():
    candidate = cards().replace("root=Column([c0,c1])", "root=Card([c0,c1])")
    values, rows = capture([candidate])
    assert values[0] > 0.95
    assert rows[0]["components"]["source_fidelity"] == 1.0
    assert rows[0]["components"]["precision_guard"] == 1.0
    result = generation_reward_v5_4(candidate, SOURCE)
    groups = result.evidence["content_assignment"]["card_field_grouping"]["certified_groups"]
    assert len(groups) == 2
    assert groups[0]["card_instance"] != groups[1]["card_instance"]
    assert all(group["field_count"] == 4 for group in groups)


def test_certified_repeated_card_groups_do_not_hide_raw_duplicates():
    duplicated = repeated_cards(RECORDS + (RECORDS[0],))
    values, rows = capture([repeated_cards(), duplicated])
    assert values[1] < values[0]
    bad = rows[1]["components"]
    assert bad["applicable_fidelity"]["visible_content_multiset_fbeta"] < 1.0
    assert bad["precision"]["semantic_non_duplication"] < 1.0
    result = generation_reward_v5_4(duplicated, SOURCE)
    grouping = result.evidence["content_assignment"]["card_field_grouping"]
    assert grouping["raw_block_count"] == 12
    assert grouping["assignment_block_count"] == 3
    assert len(grouping["certified_groups"]) == 3
    assert result.evidence["content_assignment"]["output_block_precision"] < 1.0


def test_canonical_prompt_worked_example_has_complete_source_fidelity():
    policy = (ROOT / "dataset/prompts/a2ui_express_quality_policy_v1.md").read_text(encoding="utf-8")
    example = policy.split("Field extraction example", 1)[1]
    source_match = re.search(r"\n  Source:\n(.*?)\n\s*<a2ui>", example, flags=re.DOTALL)
    output_match = re.search(r"<a2ui>.*?</a2ui>", example, flags=re.DOTALL)
    assert source_match is not None and output_match is not None
    source = textwrap.dedent(source_match.group(1)).strip()
    values, rows = capture([output_match.group(0)], source)
    assert values[0] > 0.95
    assert rows[0]["strict_valid"]
    assert rows[0]["components"]["source_fidelity"] == 1.0
    assert rows[0]["components"]["precision_guard"] == 1.0
    assert all(value == 1.0 for value in rows[0]["components"]["applicable_fidelity"].values())
