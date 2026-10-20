"""Synthetic GRPO-only exact entity/field/action reward regressions."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))

from ir_training.train.qat_grpo_reward import (
    make_qat_grpo_reward,
    reward_implementation_identity,
)
from pipeline.genui_quality._v5_4 import (
    generation_reward_v5_4,
    prepare_source_context_v5_4,
)
from pipeline.genui_quality.config_v5_4 import load_v5_4_reward_config
from pipeline.genui_quality.grpo_entity_content import prepare_grpo_entity_source

RECORDS = (
    ("Hotel Aurora", "Goa", "Price: INR 5000", "Rating: 4.5", "https://example.test/aurora"),
    ("Hotel Cedar", "Pune", "Price: INR 4000", "Rating: 4.2", "https://example.test/cedar"),
)
SOURCE = "\n".join(f"- {name}: {city}; {price}; {rating}; [Book now]({url})"
                   for name, city, price, rating, url in RECORDS)


def cards(*, records=RECORDS, order=(0, 1, 2), copy_prose=False, copy_markdown=False):
    rows = ["root=Column([" + ",".join(f"c{i}" for i in range(len(records))) + "])"]
    for i, (name, city, price, rating, url) in enumerate(records):
        if copy_prose or copy_markdown:
            text = f"{name}: {city}; {price}; {rating}"
            if copy_markdown:
                text += f"; [Book now]({url})"
            rows += [f"c{i}=Card([t{i},b{i}])", f"t{i}=Text({json.dumps(text)})"]
        else:
            rows.append(f"c{i}=Card([" + ",".join(f"f{i}{j}" for j in order)
                        + f",b{i}],title={json.dumps(name)})")
            rows.extend(f"f{i}{j}=Text({json.dumps(value)})"
                        for j, value in enumerate((city, price, rating)))
        rows.append(f'b{i}=Button("Book now",onPress=openUrl({json.dumps(url)}))')
    return "<a2ui>" + "\n".join(rows) + "</a2ui>"


def score(candidates, source=SOURCE):
    evidence = {}
    values = make_qat_grpo_reward()(
        candidates, source, log_extra=lambda key, value: evidence.update({key: value}),
    )
    return values, evidence["qat_reward"]


def test_correct_linked_fields_have_full_content_and_action_fidelity():
    values, rows = score([cards()])
    assert values[0] > 0.95
    assert rows[0]["strict_valid"]
    assert rows[0]["components"]["source_fidelity"] == 1.0
    assert rows[0]["components"]["precision_guard"] == 1.0
    assert rows[0]["components"]["critical_fidelity"]["action_and_source_link_fidelity"] == 1.0
    assert rows[0]["components"]["source_conditioned_extraction"]["coverage"] == 1.0


def test_exact_extraction_is_preferred_to_whole_prose_and_duplicate_markdown():
    values, rows = score([cards(), cards(copy_prose=True), cards(copy_markdown=True)])
    assert values[0] > values[1] > values[2]
    assert rows[1]["components"]["source_fidelity"] == 1.0
    assert rows[1]["components"]["precision_guard"] == 1.0
    assert rows[1]["components"]["source_conditioned_extraction"]["coverage"] == 0.0
    assert rows[2]["components"]["applicable_fidelity"]["visible_content_multiset_fbeta"] < 1


def test_labeled_fields_can_reorder_but_entity_and_unlabeled_prefix_cannot():
    values, rows = score([cards(), cards(order=(0, 2, 1)), cards(order=(1, 0, 2))])
    assert rows[1]["components"]["source_fidelity"] == 1.0
    assert values[1] == pytest.approx(values[0])
    assert values[2] < values[0]


@pytest.mark.parametrize("mutation", [
    lambda text: text.replace('b0=Button("Book now",onPress=openUrl("https://example.test/aurora"))',
                              'b0=Button("Book now",onPress=openUrl("https://example.test/cedar"))'),
    lambda text: text.replace("example.test/aurora", "example.test/tmp").replace(
        "example.test/cedar", "example.test/aurora").replace("example.test/tmp", "example.test/cedar"),
    lambda text: text.replace('Button("Book now"', 'Button("Cancel"', 1),
    lambda text: text.replace("f02,b0", "f02"),
    lambda text: text.replace("f02,b0", "f02,b0,b0"),
    lambda text: text.replace('"Price: INR 5000"', '"Price: INR 4000"'),
    lambda text: text.replace('f02=Text("Rating: 4.5")', 'f02=Text("Rating: 4.5",visible=false)'),
    lambda text: text.replace('"Goa"', '"Pune"', 1),
    lambda text: text.replace('title="Hotel Aurora"', 'title="Hotel Cedar"'),
])
def test_bad_associations_omissions_duplicates_and_hidden_facts_fail(mutation):
    values, rows = score([cards(), mutation(cards())])
    assert values[1] < values[0]
    assert rows[1]["reasons"]
    assert "source_conditioned_extraction" not in rows[1]["components"]


def test_same_labels_do_not_allow_swapping_destinations_between_entities():
    good = cards()
    swapped = good.replace("example.test/aurora", "example.test/tmp").replace(
        "example.test/cedar", "example.test/aurora").replace("example.test/tmp", "example.test/cedar")
    _, rows = score([swapped])
    assert rows[0]["components"]["critical_fidelity"]["action_and_source_link_fidelity"] == 0.0


def test_global_fact_inventory_cannot_hide_entity_price_swaps():
    altered = list(RECORDS)
    altered[0] = (*RECORDS[0][:2], RECORDS[1][2], *RECORDS[0][3:])
    altered[1] = (*RECORDS[1][:2], RECORDS[0][2], *RECORDS[1][3:])
    values, rows = score([cards(), cards(records=altered)])
    assert values[1] < values[0]
    fidelity = rows[1]["components"]["applicable_fidelity"]
    assert fidelity["visible_content_multiset_fbeta"] == 1.0
    assert fidelity["content_unit_fidelity"] < 1.0
    assert "source_conditioned_extraction" not in rows[1]["components"]


def test_source_links_are_not_projected_without_contract_destinations():
    base = prepare_source_context_v5_4(SOURCE).base
    base = replace(base, expected_actions=())
    projected, records = prepare_grpo_entity_source(base)
    assert projected is base
    assert records == ()


def test_entity_source_contract_is_prepared_once_per_candidate_group(monkeypatch):
    from pipeline.genui_quality import _v5_4
    original = _v5_4.prepare_grpo_entity_source
    calls = []

    def tracked(base):
        calls.append(base)
        return original(base)

    monkeypatch.setattr(_v5_4, "prepare_grpo_entity_source", tracked)
    assert len(score([cards(), cards(copy_prose=True)])[0]) == 2
    assert len(calls) == 1


def test_plain_steps_and_non_entity_prose_are_not_affected():
    source = "- Open settings\n- Enable notifications"
    candidate = '<a2ui>root=List(items=["Open settings","Enable notifications"])</a2ui>'
    normal = generation_reward_v5_4(candidate, source)
    configured = generation_reward_v5_4(candidate, source, config=replace(
        load_v5_4_reward_config(), grpo_entity_extraction=True))
    assert normal.atomics == configured.atomics
    assert normal.reward == configured.reward
    assert score([candidate], source)[1][0]["components"]["source_fidelity"] == 1.0


def test_default_canonical_scoring_does_not_silently_enable_grpo_rules():
    cfg = load_v5_4_reward_config()
    assert cfg.grpo_entity_extraction is False
    normal = generation_reward_v5_4(cards(), SOURCE)
    explicit = generation_reward_v5_4(cards(), SOURCE, config=replace(cfg, grpo_entity_extraction=False))
    opted = generation_reward_v5_4(cards(), SOURCE, config=replace(cfg, grpo_entity_extraction=True))
    assert normal.reward == explicit.reward
    assert normal.atomics == explicit.atomics
    assert normal.atomics["fidelity"]["content_unit_fidelity"] < 1.0
    assert opted.atomics["fidelity"]["content_unit_fidelity"] == 1.0
    assert normal.metric_fingerprint != opted.metric_fingerprint


@pytest.mark.parametrize("source", [
    "- Hotel Aurora: Goa; Price: INR 5000; Price: INR 6000; [Book now](https://example.test/a)",
    "- Hotel Aurora: Goa; Price: INR 5000; book [Book now](https://example.test/a)",
    "- Hotel Aurora: Goa; Price: INR 5000; https://example.test/a",
    "- Hotel Aurora: Goa; Price: INR 5000\n- Hotel Aurora: Pune; Price: INR 4000",
    "- Step 1: Open settings; Step 2: Enable notifications",
])
def test_ambiguous_or_unrecognized_sources_keep_original_units(source):
    base = prepare_source_context_v5_4(source).base
    projected, records = prepare_grpo_entity_source(base)
    assert not records
    assert projected is base


def test_metric_and_reward_identity_are_bound_and_portable():
    identity = reward_implementation_identity()
    assert identity["reward_policy"] == "qat_source_fidelity_v2"
    assert identity["metric_fingerprint"] != identity["evaluation_metric_fingerprint"]
    assert identity == reward_implementation_identity()
    assert all(str(ROOT) not in value for value in identity.values())
    assert len(identity["mapping_sha256"]) == 64


def test_extraction_cannot_bypass_strict_raw_validity():
    assert score(["Here is your UI: " + cards()])[0] == [-1.0]
