"""Opt-in regeneration safeguards cannot waive canonical training admission."""
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset/src"))
sys.path.insert(0, str(ROOT / "training/src"))

from pipeline import v11_regeneration_guard as guard

SOURCE = "- Alder; Price: $120; Breakfast: not included\n- Birch; Price: $80; Breakfast: included"
TARGET = '''<a2ui>
root=Column([a,b])
a=Card([ap,ab],title="Alder")
ap=Text("Price: $120")
ab=Text("Breakfast: not included")
b=Card([bp,bb],title="Birch")
bp=Text("Price: $80")
bb=Text("Breakfast: included")
</a2ui>'''


def row(source=SOURCE):
    return {"id": "test-row", "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def test_default_catalog_is_hash_bound_and_identity_is_reproducible():
    entries = guard._catalog()
    assert len(entries) >= 10
    assert all(len(item["source_sha256"]) == 64 and item["source_id"] for item in entries)
    assert guard.guard_identity() == guard.guard_identity()
    assert guard.guard_identity()["source_hold_catalog_sha256"] == hashlib.sha256(
        guard.SOURCE_HOLDS_PATH.read_bytes()).hexdigest()


def test_declared_source_hash_cannot_be_tampered():
    bad = row()
    bad["source"] += "\nInvented statement."
    report = guard.assess_source(bad)
    assert not report["eligible"]
    assert "source_hash_mismatch:source_sha256" in report["reasons"]


def test_source_aliases_and_final_task_must_agree_exactly():
    good = row()
    good["response_text"] = SOURCE
    good["messages"] = [{"role": "user", "content": guard._TASK_PREFIX + SOURCE}]
    assert guard.source_text(good) == SOURCE
    good["messages"][0]["content"] += " "
    assert guard.assess_source(good)["reasons"] == ["source_alias_mismatch"]


def test_malformed_source_hold_catalog_is_setup_failure(tmp_path, monkeypatch):
    path = tmp_path / "holds.json"
    path.write_text(json.dumps({"policy_version": guard.POLICY_VERSION,
                              "holds": [{"row_id": "x", "source_sha256": "tampered"}]}))
    monkeypatch.setattr(guard, "SOURCE_HOLDS_PATH", path)
    with pytest.raises(ValueError, match="invalid_source_hold_entry"):
        guard.assess_source(row())


def test_valid_exact_entity_fields_pass_all_gates_without_mutation():
    original = row()
    before = deepcopy(original)
    result = guard.assess_candidate(original, TARGET)
    assert result["eligible"], result
    assert result["canonical_acceptance"]["eligible"] is True
    assert result["entity_ownership"]["binding_complete"] is True
    assert result["entity_ownership"]["structured_record_count"] == 2
    assert original == before


def test_swapped_entity_prices_cannot_pass_via_global_number_bag():
    swapped = TARGET.replace("$120", "$TEMP").replace("$80", "$120").replace("$TEMP", "$80")
    result = guard.assess_candidate(row(), swapped)
    assert not result["eligible"]
    assert "entity_binding_incomplete" in result["reasons"]


def test_entity_qualifiers_cannot_be_swapped_even_when_all_words_are_present():
    swapped = TARGET.replace("Breakfast: not included", "Breakfast: TEMP")
    swapped = swapped.replace("Breakfast: included", "Breakfast: not included")
    swapped = swapped.replace("Breakfast: TEMP", "Breakfast: included")
    result = guard.assess_candidate(row(), swapped)
    assert not result["eligible"]
    assert "entity_binding_incomplete" in result["reasons"]


def test_invented_breakfast_claim_is_unsupported_even_when_numbers_match():
    source = "- Alder; Price: $120; Room: standard\n- Birch; Price: $80; Room: compact"
    target = TARGET.replace("Breakfast: not included", "Room: standard. Breakfast included")
    target = target.replace("Breakfast: included", "Room: compact")
    result = guard.assess_candidate(row(source), target)
    assert not result["eligible"]
    assert "unsupported_visible_prose" in result["reasons"]


def test_qualifier_and_negation_loss_are_not_waived():
    source = "Respond within 4 hours of notification. Breakfast is not included."
    target = '<a2ui>\nroot=Text("Respond within 4 hours. Breakfast is included.")\n</a2ui>'
    result = guard.assess_candidate(row(source), target)
    assert not result["eligible"]
    assert "fidelity:source_literal_not_exactly_represented" in result["reasons"]


def test_complete_entity_cards_cannot_hide_missing_nonentity_qualifier():
    source = SOURCE + "\n\nPrices are valid only for stays before 30 November."
    result = guard.assess_candidate(row(source), TARGET)
    assert result["entity_ownership"]["binding_complete"] is True
    assert not result["eligible"]
    assert "fidelity:source_literal_not_exactly_represented" in result["reasons"]
    assert any("Prices are valid only" in item["text"] for item in result["fidelity"]["missing_units"])


def test_correct_plain_checklist_does_not_require_entity_cards():
    source = "- Pack warm clothes.\n- Carry your ID."
    target = '<a2ui>\nroot=Column([a,b])\na=Text("Pack warm clothes.")\nb=Text("Carry your ID.")\n</a2ui>'
    result = guard.assess_candidate(row(source), target)
    assert result["eligible"], result
    assert result["entity_ownership"]["applicable"] is False


def test_exact_compact_table_for_source_table_keeps_existing_acceptance():
    source = "Name | Price | Breakfast\n--- | --- | ---\nAlder | $120 | not included\nBirch | $80 | included"
    target = '<a2ui>\nroot=Table(columns=["Name","Price","Breakfast"],rows=[["Alder","$120","not included"],["Birch","$80","included"]])\n</a2ui>'
    result = guard.assess_candidate(row(source), target)
    assert result["eligible"], result
    assert result["canonical_acceptance"]["eligible"] is True


def test_exact_list_to_table_passes_explicit_source_bound_canonical_policy():
    target = '<a2ui>\nroot=Table(columns=["Name","Price","Breakfast"],rows=[["Alder","$120","not included"],["Birch","$80","included"]])\n</a2ui>'
    result = guard.assess_candidate(row(), target)
    assert result["entity_ownership"]["binding_complete"] is True
    assert result["entity_ownership"]["structured_record_count"] == 2
    assert result["canonical_acceptance"]["eligible"] is True
    assert result["canonical_policy"]["grpo_entity_extraction"] is True
    assert result["eligible"], result


@pytest.mark.parametrize("table_rows", [
    '[["Alder","$80","not included"],["Birch","$120","included"]]',
    '[["Alder","$120","included"],["Birch","$80","not included"]]',
    '[["Alder","$120","included"],["Birch","$80","included"]]',
    '[["Alder","$120","not included; Free parking"],["Birch","$80","included"]]',
])
def test_table_swaps_negation_loss_and_extra_claims_still_fail(table_rows):
    target = '<a2ui>\nroot=Table(columns=["Name","Price","Breakfast"],rows=' + table_rows + ')\n</a2ui>'
    result = guard.assess_candidate(row(), target)
    assert not result["eligible"]
    assert result["entity_ownership"]["binding_complete"] is False
    assert "entity_binding_incomplete" in result["reasons"]


def test_exact_heading_cards_pass_explicit_source_bound_canonical_policy():
    source = "## Hotel Aurora\n- Price: INR 5000\n- Rating: 4.5\n\n## Hotel Cedar\n- Price: INR 4000\n- Rating: 4.2"
    target = ('<a2ui>root=Column([a,b])\na=Card([ah,ap,ar])\n'
              'ah=Text("Hotel Aurora","h2")\nap=Text("Price: INR 5000")\nar=Text("Rating: 4.5")\n'
              'b=Card([bh,bp,br])\nbh=Text("Hotel Cedar","h2")\nbp=Text("Price: INR 4000")\n'
              'br=Text("Rating: 4.2")</a2ui>')
    result = guard.assess_candidate(row(source), target)
    assert result["entity_ownership"]["binding_complete"] is True
    assert result["entity_ownership"]["structured_record_count"] == 2
    assert all(item["source_format"] == "heading_fields"
               for item in result["entity_ownership"]["certified_records"])
    assert result["canonical_acceptance"]["eligible"] is True
    assert result["eligible"], result
    swapped = target.replace("5000", "TEMP").replace("4000", "5000").replace("TEMP", "4000")
    invented = target.replace("[ah,ap,ar]", "[ah,ap,ar,x]").replace(
        "</a2ui>", '\nx=Text("Breakfast included")</a2ui>')
    for broken in (swapped, invented):
        rejected = guard.assess_candidate(row(source), broken)
        assert not rejected["eligible"]
        assert "entity_binding_incomplete" in rejected["reasons"]


def test_whole_inline_entity_records_are_not_counted_as_structured_extraction():
    target = '<a2ui>\nroot=Column([a,b])\na=Card([at])\nat=Text("Alder; Price: $120; Breakfast: not included")\nb=Card([bt])\nbt=Text("Birch; Price: $80; Breakfast: included")\n</a2ui>'
    result = guard.assess_candidate(row(), target)
    assert not result["eligible"]
    assert "entity_fields_not_structured" in result["reasons"]


def test_forged_saved_admission_does_not_override_recomputed_gate():
    original = row()
    original["training_acceptance"] = {"eligible": True, "blocking_reasons": [], "review_reasons": []}
    result = guard.assess_candidate(original, '<a2ui>\nroot=Text("Unrelated answer")\n</a2ui>')
    assert not result["eligible"]
    assert "canonical_training_ineligible" in result["reasons"]


def test_even_perfect_supplemental_checks_cannot_waive_canonical_review(monkeypatch):
    monkeypatch.setattr(guard, "generation_reward_a2ui_express_v1", lambda *a, **kw: SimpleNamespace(
        evidence={"training_acceptance": {"eligible": False, "blocking_reasons": [],
                                          "review_reasons": ["missing_or_mismatched_action"]}}))
    result = guard.assess_candidate(row(), TARGET)
    assert not result["eligible"]
    assert "canonical_training_ineligible" in result["reasons"]


def test_guard_policy_is_explicit_and_does_not_mutate_default_scorer_config():
    original = guard.coerce_reward_config_v5_4(None)
    assert original.grpo_entity_extraction is False
    result = guard.assess_candidate(row(), TARGET)
    assert result["canonical_policy"] == {"scorer": "generation_reward_a2ui_express_v1",
                                          "grpo_entity_extraction": True, "thresholds": "unchanged"}
    assert guard.coerce_reward_config_v5_4(None) == original
    assert original.grpo_entity_extraction is False


def test_literal_diagnostic_resolution_is_whole_unit_owned_and_not_global_bag():
    fidelity = {"reasons": ["source_literal_not_exactly_represented"], "missing_unit_count": 1,
                "missing_units": [{"unit_id": "u", "text": "Alder; Price: $120; Breakfast: not included"}],
                "missing_unit_examples_truncated": False}
    certificate = {"binding_complete": True, "issues": [], "eligible_record_count": 1,
                   "structured_record_count": 1, "certified_records": [{
                       "structured": True, "actions_exact": True,
                       "source_text": "Alder; Price: $120; Breakfast: not included"}]}
    assert guard._fidelity_reasons(fidelity, certificate) == ([], ["u"])
    wrong = deepcopy(certificate)
    wrong["certified_records"][0]["source_text"] = "Birch; Price: $120; Breakfast: not included"
    assert guard._fidelity_reasons(fidelity, wrong)[0] == fidelity["reasons"]
    # Same tokens in a different sequence or an added/dropped qualifier are not proof.
    wrong["certified_records"][0]["source_text"] = "Alder; Price: $120; Breakfast: included not"
    assert guard._fidelity_reasons(fidelity, wrong)[0] == fidelity["reasons"]
    fidelity["missing_unit_examples_truncated"] = True
    assert guard._fidelity_reasons(fidelity, certificate)[0] == fidelity["reasons"]


def test_release_review_hold_is_still_mandatory(monkeypatch):
    from ir_training.data import release_review
    monkeypatch.setattr(release_review, "reviewed_hold", lambda *args: ("reviewed_text_boundary_hold", "held"))
    result = guard.assess_candidate(row(), TARGET)
    assert not result["eligible"]
    assert "reviewed_text_boundary_hold" in result["reasons"]


def test_invented_reference_has_no_old_target_restoration_escape():
    source = "Action: [Button: Book] <[ACTION_URL_1]>"
    target = '<a2ui>\nroot=Button("Book",onPress=openUrl("[ACTION_URL_2]"))\n</a2ui>'
    original = row(source)
    original["canonical_graph"] = {"stale": "old target is not candidate proof"}
    original["canonical_graph_hash"] = "0" * 64
    result = guard.assess_candidate(original, target)
    assert not result["eligible"]
    assert "unbound_target_references" in result["validation_error"]


def test_exact_entity_fields_cannot_waive_a_missing_source_action():
    source = SOURCE + "\n\nAction: [Button: Book] <https://example.com/book>"
    result = guard.assess_candidate(row(source), TARGET)
    assert not result["eligible"]
    assert "canonical_training_ineligible" in result["reasons"]
    assert result["canonical_acceptance"]["eligible"] is False


def test_detached_state_and_elements_cannot_supply_hidden_evidence():
    target = '<a2ui>\nroot=Text("Alder")\nother=Text("Price: $120")\n</a2ui>'
    result = guard.assess_candidate(row(), target)
    assert not result["eligible"]
    assert "unreachable_components" in result["reasons"]


# Exact audited source bytes are intentional small regression evidence, not a training artifact.
AUDITED_BUDGET_SOURCE = "## Monthly Budget Analysis\nMedia: Icon=[ICON_URL_1]\nThe current monthly spending totals **$5,900**, leaving a remaining margin of only **$100** from the $6,000 budget. This provides a very thin safety net for unexpected expenses or savings.\n\n## Expense Breakdown & Evaluation\nMedia: Icon=[ICON_URL_2]\n\nExpense | Monthly Cost | % of Budget | Status\n--- | --- | --- | ---\nMortgage | $2,100 | 35% | Stable\nCar Payments | $600 | 10% | Moderate\nDining Out | $800 | 13% | High\nInsurance | $300 | 5% | Stable\nUtilities | $400 | 7% | Stable\nEntertainment | $500 | 8% | High\nGroceries | $600 | 10% | Stable\nMisc | $700 | 12% | High\n\n## High-Spending Areas & Savings Plan\nMedia: Icon=[ICON_URL_3]\n\n**Dining Out**\nCurrent spend is disproportionately high compared to groceries.\n- Reduction: Limit dining to twice a month and embrace meal prepping.\n- Target Savings: **$300**\n\n**Miscellaneous**\nA $700 \"Misc\" category is too vague and often masks impulsive spending.\n- Reduction: Implement a strict $300 cap on non-essential unplanned purchases.\n- Target Savings: **$250**\n\n**Entertainment**\nSpending $500 monthly on top of dining and misc is excessive for this budget level.\n- Reduction: Switch to one primary streaming service and utilize free local events.\n- Target Savings: **$150**\n\n## Optimized Budget Target\nMedia: Icon=[ICON_URL_4]\n\n- Total Potential Savings: **$700**\n- New Monthly Spend: **$5,200**\n- New Monthly Surplus: **$800** (includes the original $100)\n\nQuick Actions\nAction: [Button: Budget Template] <[ACTION_URL_1]>\nAction: [Button: Savings Calculator] <[ACTION_URL_2]>"


def test_actual_audited_budget_arithmetic_is_held_by_source_content_hash():
    original = row(AUDITED_BUDGET_SOURCE)
    original["id"] = "archive-train-038207-a96e6afa71525792"
    original["source_id"] = "archive-source-family-sha256:4579fe9c4735a2d5075978d8b451f7175a52e3875f7d9b23747bc9efa02f7ef1"
    report = guard.assess_source(original)
    assert report["source_sha256"] == "4579fe9c4735a2d5075978d8b451f7175a52e3875f7d9b23747bc9efa02f7ef1"
    assert "audited_source_quality_hold" in report["reasons"]
    assert not report["eligible"]
    # Renaming a row cannot release the same known-defective source bytes.
    original["id"] = "renamed"
    original["source_id"] = "different-family"
    assert not guard.assess_source(original)["eligible"]


def test_changed_source_under_audited_identity_requires_new_review():
    original = row(AUDITED_BUDGET_SOURCE + "\nEdited source.")
    original["id"] = "archive-train-038207-a96e6afa71525792"
    result = guard.assess_source(original)
    assert "audited_source_identity_hash_mismatch" in result["reasons"]
    assert not result["eligible"]
