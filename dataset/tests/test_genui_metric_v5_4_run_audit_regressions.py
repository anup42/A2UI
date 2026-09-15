"""CPU-only regressions from the September 2026 generation audit."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.genui_quality import generation_reward_a2ui_express_v1 as score
from pipeline.genui_quality.aggregate_v5_4 import score_record_generation_v5_4
from pipeline.genui_quality.candidate_normalization_v5_4 import normalize_and_validate_express_candidate_v5_4
from pipeline.genui_quality.config_v5_4 import RewardConfigV54
from pipeline.genui_quality.evidence_v5_4 import collect_output_evidence_v5_4
from pipeline.genui_quality.graph import audit_renderer_graph
from pipeline.genui_quality.metrics_v5_4 import normalize_formula_wrapper, semantic_role_coverage_v5_4
from pipeline.genui_quality.source_contract_v5_4 import extract_expected_ui_contract_v5_4


SOURCE = "Action: [Button: Open] https://example.test/a/"
REFERENCES = {"[URL_1]": "https://example.test/a"}


def button(target="[URL_1]", label="Open"):
    return f'<a2ui>\nroot=Button("{label}",onPress=openUrl(url="{target}"))\n</a2ui>'


def evidence(program):
    normalization = normalize_and_validate_express_candidate_v5_4(program)
    assert normalization.production_valid, normalization.errors
    return collect_output_evidence_v5_4(normalization.canonical_spec, audit_renderer_graph(normalization.canonical_spec), config=RewardConfigV54())


def test_reference_restoration_is_semantic_only_and_trailing_slash_safe():
    raw = score(button(), SOURCE)
    restored = score(button(), SOURCE, reference_map=REFERENCES)
    literal = score(button("https://example.test/a"), SOURCE)
    assert raw.evidence["action_matching"]["matched_required_count"] == 0
    assert restored.evidence["action_matching"]["matched_required_count"] == 1
    assert restored.identity["raw_candidate_hash"] == raw.identity["raw_candidate_hash"]
    assert restored.identity["canonical_candidate_hash"] == literal.identity["canonical_candidate_hash"]
    assert restored.identity["reference_map_hash"] != raw.identity["reference_map_hash"]
    assert restored.evidence["training_acceptance"]["eligible"]


@pytest.mark.parametrize("target", ["[URL_2]", "[URL_999]", "https://invented.test"])
def test_missing_swapped_and_invented_references_remain_penalized(target):
    mapping = {**REFERENCES, "[URL_2]": "https://example.test/wrong"}
    result = score(button(target), SOURCE, reference_map=mapping)
    assert result.evidence["action_matching"]["matched_required_count"] == 0
    assert "missing_or_mismatched_action" in result.evidence["training_acceptance"]["review_reasons"]
    assert not result.evidence["training_acceptance"]["blocking_reasons"]


@pytest.mark.parametrize("placeholder", ["[URL_1]", "[ACTION_URL_1]", "[SOURCE_URL_1]"])
def test_opaque_reference_identity_on_both_sides_needs_no_network(placeholder):
    result = score(button(placeholder), f"Action: [Button: Open] {placeholder}")
    assert result.evidence["action_matching"]["matched_required_count"] == 1
    assert result.evidence["training_acceptance"]["media_readiness_required"] is False


def test_map_cannot_rescue_invalid_raw_syntax_and_invalid_map_is_explicit():
    broken = button().replace("</a2ui>", "")
    result = score(broken, SOURCE, reference_map=REFERENCES)
    assert not result.normalization["production_valid"]
    assert not result.evidence["training_acceptance"]["eligible"]
    with pytest.raises(ValueError, match="reference_map"):
        score(button(), SOURCE, reference_map={"Open": "Changed"})


def test_map_change_invalidates_stored_score_reuse():
    first = score(button(), SOURCE, reference_map=REFERENCES)
    record = {"response_text": SOURCE, "target_format": "a2ui_express_v1", "a2ui_express": button(),
              "reference_map": REFERENCES, "generation_reward_v5_4": asdict(first)}
    assert score_record_generation_v5_4(record).evidence["score_reuse"]["reused"]
    record["reference_map"] = {"[URL_1]": "https://example.test/wrong"}
    changed = score_record_generation_v5_4(record)
    assert not changed.evidence["score_reuse"]["reused"]
    assert "reference_map_hash_mismatch" in changed.evidence["score_reuse"]["stale_reasons"]
    assert changed.evidence["action_matching"]["matched_required_count"] == 0


@pytest.mark.parametrize("kind", ["Icon", "Image"])
def test_source_icon_with_added_accessible_label_is_supported(kind):
    result = score(f'<a2ui>\nroot={kind}(src="[IMAGE_1]",alt="Location")\n</a2ui>',
                   "Media: Icon = https://example.test/icons/pin.svg",
                   reference_map={"[IMAGE_1]": "https://example.test/icons/pin.svg"})
    assert result.normalization["production_valid"], result.normalization
    assert result.unsupported_external_additions["unsupported_semantic_media_count"] == 0
    assert result.unsupported_external_additions["source_icon_supported_count"] == 1


def test_invented_and_duplicate_labelled_icons_are_still_unsupported():
    result = score('<a2ui>\nroot=Column([a,b,c])\na=Image(src="https://example.test/pin.svg",alt="Pin")\nb=Icon(src="https://example.test/pin.svg",alt="Pin")\nc=Image(src="https://invented.test/pin.svg",alt="Pin")\n</a2ui>',
                   "Media: Icon = https://example.test/pin.svg")
    assert result.normalization["production_valid"]
    assert result.unsupported_external_additions["unsupported_semantic_media_count"] == 2


@pytest.mark.parametrize("wrapper", ["$x = 7$", "$$x = 7$$", r"\(x = 7\)", r"\[x = 7\]"])
def test_formula_wrapper_equivalence_preserves_numeric_differences(wrapper):
    kwargs = dict(expected_exact_indices=None, threshold=.5, exact_dense_limit=64, max_edges=500, top_k=16)
    expected = {"formula": [{"content": wrapper, "required": True}]}
    same = semantic_role_coverage_v5_4(expected, {"formula": [{"content": "x = 7"}]}, **kwargs)
    wrong = semantic_role_coverage_v5_4(expected, {"formula": [{"content": "x = 9"}]}, **kwargs)
    assert same[0] == 1.0
    assert wrong[0] < same[0]


@pytest.mark.parametrize("formula", ["$x$ + $y$", "$x", r"\(x\]", r"$x\$"])
def test_unbalanced_or_multiple_formula_wrappers_are_not_erased(formula):
    assert normalize_formula_wrapper(formula) == formula


def test_paths_do_not_require_console_but_explicit_console_does():
    paths = extract_expected_ui_contract_v5_4("Mount `/mnt/data` and save in `C:\\temp`. The result has 42 files.")
    assert not paths["required_roles"].get("console")
    console = extract_expected_ui_contract_v5_4("Console output: 42 files copied")
    assert console["required_roles"]["console"]
    assert console["role_requirements"]["console"]


def test_reference_metadata_numbers_are_excluded_but_visible_numbers_remain():
    contract = extract_expected_ui_contract_v5_4("Total: 42\nMedia: Icon = https://cdn.test/bootstrap/1.11.3/icon.svg\n| Name | Value |\n|---|---|\n| A | 17 |")
    assert "42" in contract["exact_values"] and "17" in contract["exact_values"]
    assert "1.11" not in contract["exact_values"] and "3" not in contract["exact_values"]


def test_literal_list_items_are_visible_and_explicit_links_are_actions():
    result = evidence('<a2ui>\nroot=List(items=["First",{title:"Second",description:"Details",url:"https://example.test/a"}])\n</a2ui>')
    assert all(text in result.output.visible_blocks for text in ("First", "Second", "Details"))
    assert any(action.url == "https://example.test/a" for action in result.output.actions)


def test_source_list_markdown_links_are_actions_but_ordinary_strings_are_text():
    source = evidence('<a2ui>\nroot=Card([links],title="Sources")\nlinks=List(items=["[Docs](https://example.test/a)"])\n</a2ui>')
    ordinary = evidence('<a2ui>\nroot=List(items=["[Docs](https://example.test/a)"])\n</a2ui>')
    assert len(source.output.actions) == 1
    assert source.output.actions[0].label == "Docs"
    assert not ordinary.output.actions


def test_hidden_list_items_and_opaque_component_maps_do_not_gain_credit():
    hidden = evidence('<a2ui>\nroot=Column([items])\nitems=List(items=["Hidden"],visible=false)\n</a2ui>')
    assert "Hidden" not in hidden.output.visible_blocks
    invalid = normalize_and_validate_express_candidate_v5_4('<a2ui>\nroot=List(items=[{type:"Text",props:{text:"Hidden"}}])\n</a2ui>')
    assert not invalid.production_valid


def test_reachable_loss_is_blocked_even_when_syntax_valid():
    result = score('<a2ui>\nroot=Text("Hello")\norphan=Text("Important fact")\n</a2ui>', "Hello\nImportant fact")
    assert "renderer_unreachable_elements" in result.evidence["training_acceptance"]["blocking_reasons"]


@pytest.mark.parametrize("source,reason", [
    ("Show two charts.", "inferred_role_count_gap:chart"),
    ("| Name | Value |\n|---|---|\n| A | 7 |", "missing_or_mismatched_table"),
    ("Formula: x = 7", "missing_or_mismatched_role:formula"),
])
def test_certified_semantic_assignment_never_proves_a_hard_rejection(source, reason):
    result = score('<a2ui>\nroot=Text("Summary")\n</a2ui>', source)
    gate = result.evidence["training_acceptance"]
    assert not gate["blocking_reasons"]
    assert reason in gate["review_reasons"]
    assert not gate["eligible"]
