"""Exact source-owned source/action roles remain separate at shared URLs."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[2] / "training" / "src")]

from pipeline.genui_quality import generation_reward_a2ui_express_v1 as score
from pipeline.genui_quality._v5_4 import prepare_source_context_v5_4
from pipeline.genui_quality.config_v5_4 import RewardConfigV54
from pipeline.genui_quality.pilot_source_contract_v2 import (
    source_action_occurrences,
    source_contract_v2,
    source_link_occurrences,
)
from pipeline.training_fidelity import audit_training_fidelity
from pipeline.v11_regeneration_guard import assess_candidate

URL = "https://docs.python.org/3/"
SOURCE = f"Action: [Button: Read] {URL}\nSources:\n- [Python docs]({URL})"


def candidate(action_label="Read", source_label="Python docs", action_url=URL, source_url=URL,
              *, include_action=True, include_source=True):
    children = (["action"] if include_action else []) + (["sources"] if include_source else [])
    lines = ["root=Column([" + ",".join(children) + "])"]
    if include_action:
        lines.append(f'action=Button("{action_label}",onPress=openUrl(url="{action_url}"))')
    if include_source:
        lines += ['sources=Card([links],title="Sources")',
                  f'links=List(items=[{{label:"{source_label}",url:"{source_url}"}}])']
    return "<a2ui>\n" + "\n".join(lines) + "\n</a2ui>"


def assess(target, source=SOURCE):
    return assess_candidate({"id": "source-role-test", "response_text": source}, target, pilot_fidelity_v2=True)


def test_no_link_source_card_native_fallback_rejects_invented_title():
    target = '<a2ui>root=Card([body],title="Sources: invented guarantees")\nbody=Text("Keep the door closed.")</a2ui>'
    assert not assess(target, "Keep the door closed.")["eligible"]


def test_no_link_source_card_exact_visible_title_passes():
    target = '<a2ui>root=Card([body],title="Sources: Safety")\nbody=Text("Keep the door closed.")</a2ui>'
    assert assess(target, "Sources: Safety\nKeep the door closed.")["eligible"]


def test_same_url_action_and_source_require_two_owned_occurrences():
    result = score(candidate(), SOURCE, config=RewardConfigV54(pilot_fidelity_v2=True))
    assert result.evidence["training_acceptance"]["eligible"]
    assert result.evidence["action_matching"]["required_count"] == 2
    assert result.evidence["action_matching"]["matched_required_count"] == 2
    assert assess(candidate())["eligible"]


@pytest.mark.parametrize("destination", [
    "https://docs.python.org/3/#intro",
    "https://docs.python.org/3/?version=3&view=full#intro",
    "https://docs.python.org/3//tutorial/#intro",
])
def test_exact_source_action_destination_bytes_pass(destination):
    source = f"Action: [Button: Read] {destination}\nSources:\n- [Python docs]({destination})"
    assert assess(candidate(action_url=destination, source_url=destination), source)["eligible"]


@pytest.mark.parametrize("destination", [
    "https://docs.python.org/3/#unrelated", "https://docs.python.org/3/",
    "https://docs.python.org/3/?view=other#intro", "https://docs.python.org/4/#intro",
    "https://docs.python.org/3#intro", "https://docs.python.org//3/#intro",
])
def test_changed_fragment_query_or_path_cannot_share_canonical_url_credit(destination):
    original = "https://docs.python.org/3/#intro"
    source = f"Action: [Button: Read] {original}\nSources:\n- [Python docs]({original})"
    assert not assess(candidate(action_url=destination, source_url=destination), source)["eligible"]


def test_action_only_fragment_mismatch_is_also_rejected():
    original = "https://docs.python.org/3/#intro"
    source = f"Action: [Button: Read] {original}"
    good = candidate(action_url=original, include_source=False)
    bad = candidate(action_url="https://docs.python.org/3/#unrelated", include_source=False)
    assert assess(good, source)["eligible"]
    assert not assess(bad, source)["eligible"]


def test_real_pilot_angle_action_and_labeled_angle_source_are_distinct_roles():
    source = f"Action: [Button: Read] <{URL}>\nSources:\n- Python docs: <{URL}>"
    result = score(candidate(), source, config=RewardConfigV54(pilot_fidelity_v2=True))
    assert result.evidence["action_matching"]["required_count"] == 2
    assert result.evidence["action_matching"]["matched_required_count"] == 2
    assert assess(candidate(), source)["eligible"]
    assert not assess(candidate(action_label="Python docs", source_label="Read"), source)["eligible"]
    assert not assess(candidate(action_url=URL + "#invented", source_url=URL + "#invented"), source)["eligible"]


def test_real_pilot_angle_actions_preserve_duplicate_occurrences():
    source = f"Action: [Button: Read] <{URL}>\nAction: [Button: Read] <{URL}>"
    prepared = prepare_source_context_v5_4(source, config=RewardConfigV54(pilot_fidelity_v2=True))
    assert len(prepared.base.expected_actions) == 2
    assert not assess(candidate(include_source=False), source)["eligible"]


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_angle_action_and_source_grammar_never_claims_fenced_content(fence):
    source = f"{fence}\nAction: [Button: Read] <{URL}>\nSources:\n- Python docs: <{URL}>\n{fence}"
    assert not source_action_occurrences(source)
    assert not source_link_occurrences(source)


@pytest.mark.parametrize("angle", [False, True])
def test_unproven_same_base_fragment_occurrence_is_never_dropped(angle):
    original = URL + "#intro"
    action_target = f"<{original}>" if angle else original
    source = f"Action: [Button: Read] {action_target}\nUnclassified reference: <{URL}#other>"
    prepared = prepare_source_context_v5_4(source, config=RewardConfigV54(pilot_fidelity_v2=True))
    assert prepared.expected_contract["source_links"]
    assert len(prepared.base.expected_actions) >= 2
    assert not assess(candidate(action_url=original, include_source=False), source)["eligible"]


def test_ordinary_list_source_role_requires_visible_sibling_source_heading():
    source = f"Action: [Button: Read] <{URL}>\nSources:\n- Python docs: <{URL}>"
    template = f'<a2ui>root=Column([action,heading,sources])\naction=Button("Read",onPress=openUrl(url="{URL}"))\nheading=Text("Sources",variant="h2",visible=VISIBLE)\nsources=List(items=[{{text:"Python docs",url:"{URL}"}}])</a2ui>'
    good = score(template.replace("VISIBLE", "true"), source, config=RewardConfigV54(pilot_fidelity_v2=True))
    bad = score(template.replace("VISIBLE", "false"), source, config=RewardConfigV54(pilot_fidelity_v2=True))
    assert good.evidence["action_matching"]["matched_required_count"] == 2
    assert bad.evidence["action_matching"]["matched_required_count"] < 2
    assert not bad.evidence["training_acceptance"]["eligible"]


def test_invisible_ancestor_id_cannot_assign_source_role():
    source = f"Action: [Button: Read] <{URL}>\nSources:\n- Python docs: <{URL}>"
    target = f'<a2ui>root=Column([action,sources])\naction=Button("Read",onPress=openUrl(url="{URL}"))\nsources=Column([links])\nlinks=List(items=[{{label:"Python docs",url:"{URL}"}}])</a2ui>'
    result = score(target, source, config=RewardConfigV54(pilot_fidelity_v2=True))
    assert result.normalization["production_valid"]
    assert result.evidence["action_matching"]["matched_required_count"] == 1
    assert not assess(target, source)["eligible"]


@pytest.mark.parametrize("kwargs", [
    {"include_action": False}, {"include_source": False},
    {"action_label": "Python docs", "source_label": "Read"},
    {"source_label": "Made up source"}, {"action_label": "Made up action"},
    {"source_url": "https://www.python.org/"}, {"action_url": "https://www.python.org/"},
])
def test_missing_swapped_wrong_label_or_target_never_pass(kwargs):
    assert not assess(candidate(**kwargs))["eligible"]


def test_swapped_targets_with_correct_labels_are_rejected():
    source = f"Action: [Button: Read] {URL}\nSources:\n- [Python docs](https://www.python.org/)"
    assert not assess(candidate(action_url="https://www.python.org/", source_url=URL), source)["eligible"]


def test_two_buttons_cannot_satisfy_separate_source_role():
    target = f'<a2ui>root=Column([a,b])\na=Button("Read",onPress=openUrl(url="{URL}"))\nb=Button("Python docs",onPress=openUrl(url="{URL}"))</a2ui>'
    assert not assess(target)["eligible"]


def test_default_source_extraction_unchanged_and_no_source_role_from_destination_alone():
    default = prepare_source_context_v5_4(SOURCE)
    explicit = prepare_source_context_v5_4(SOURCE, config=RewardConfigV54(pilot_fidelity_v2=False))
    assert default.expected_contract == explicit.expected_contract
    assert len(default.base.expected_actions) == 1
    assert source_link_occurrences(f"Ordinary prose [Python docs]({URL})") == []
    assert source_link_occurrences(f"```\nSources:\n- [Python docs]({URL})\n```") == []


def test_source_projection_retains_qualifying_prose_and_explicit_constraints():
    source = f"Sources:\n- [Python docs]({URL})\nOnly use version 3."
    contract = {"actions": [{"label": "Read", "target": URL, "minimum_count": 2, "expected_component_id": "read"}],
                "source_links": [{"url": URL}], "content_units": ["Sources:", f"- [Python docs]({URL})", "Only use version 3."]}
    transformed = source_contract_v2(source, contract)
    assert transformed["content_units"] == ["Only use version 3."]
    assert transformed["actions"][0]["minimum_count"] == 2
    assert transformed["actions"][0]["expected_component_id"] == "read"
    assert contract["content_units"][0] == "Sources:"


def test_formula_grouped_number_normalization_is_source_role_scoped():
    graph = {"root": "root", "state": {}, "elements": {
        "root": {"type": "Formula", "props": {"latex": r"x = 1{,}234"}, "children": []}}}
    correct = audit_training_fidelity(r"Formula: x = 1{,}234", graph, pilot_fidelity_v2=True)
    assert not correct["unsupported_numeric_anchors"]
    wrong = audit_training_fidelity(r"Formula: x = 1{,}235", graph, pilot_fidelity_v2=True)
    assert wrong["unsupported_numeric_anchors"] == ["1234"]
    prose = audit_training_fidelity(r"Literal prose x = 1{,}234", graph, pilot_fidelity_v2=True)
    assert prose["unsupported_numeric_anchors"] == ["1234"]
