"""Repair-driven source fidelity fixtures; no model calls or target mutation."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.genui_quality import generation_reward_a2ui_express_v1
from pipeline.ir_formats import decode_express_completion
from pipeline.renderer_capability import canonical_chart_subtype
from pipeline.renderer_effective_semantics_v5_4 import effective_chart
from pipeline.training_fidelity import audit_training_fidelity, guidance_identity
from test_generation_quality_regressions import run_fake


def graph(program):
    return decode_express_completion("<a2ui>\n" + program + "\n</a2ui>")


def test_missing_intersection_prose_is_diagnostic_and_source_is_unchanged():
    source = "## 50/30/20 Targets and Needs Reduction Plan\nAllocation from $4,800 net\n- Needs: $2,400"
    target = graph('root=Column([heading,needs])\nheading=Text("50/30/20 Targets and Needs Reduction Plan","h2")\nneeds=Text("Needs: $2,400")')
    before = deepcopy(target)
    audit = audit_training_fidelity(source, target)
    assert audit["status"] == "needs_review"
    assert audit["checked_source_unit_count"] == audit["source_unit_count"] == 3
    assert any(item["text"] == "Allocation from $4,800 net" for item in audit["missing_units"])
    assert all(item["source_span"] is not None for item in audit["missing_units"])
    assert audit["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert target == before
    assert audit["diagnostic_only"]


@pytest.mark.parametrize("source,target", [
    ("The Cabinet of Dr Caligari 1920 – 71 min", "The Cabinet of Dr Caligari 1920"),
    ("Role information. Four applicants are in the screening pipeline with the latest activity dates supplied.", "Role information."),
    ("Use reusable tools, but replacement adhesive is not included.", "Use reusable tools; replacement adhesive is included."),
])
def test_short_qualification_clause_and_negation_loss_remain_reviewed(source, target):
    result = audit_training_fidelity(source, graph(f'root=Text("{target}")'))
    assert result["missing_unit_count"] == 1
    assert "source_literal_not_exactly_represented" in result["reasons"]


def test_independent_lines_joined_into_numeric_word_are_reported():
    source = "2026-03-14 San Lorenzo 50\nTrend: possession rises steadily."
    result = audit_training_fidelity(source, graph('root=Text("2026-03-14 San Lorenzo 50Trend: possession rises steadily.")'))
    assert result["suspected_word_join_count"] == 1
    assert result["suspected_word_joins"][0]["target_word"] == "50Trend"
    assert "suspected_source_word_join" in result["reasons"]


def test_markdown_heading_separator_loss_is_reported():
    source = "Mon Fri Rest\n\n## Week 5 review"
    result = audit_training_fidelity(source, graph('root=Text("Mon Fri RestWeek 5 review")'))
    assert result["suspected_word_joins"][0]["target_word"] == "RestWeek"


def test_adjacent_text_separators_preserve_complete_source_units():
    source = "2026-03-14 San Lorenzo 50\nTrend: possession rises steadily."
    complete = graph('root=Column([result,trend])\nresult=Text("2026-03-14 San Lorenzo 50")\ntrend=Text("Trend: possession rises steadily.")')
    result = audit_training_fidelity(source, complete)
    assert result["status"] == "checks_passed"
    assert result["suspected_word_join_count"] == 0


def test_long_literal_keeps_end_of_source_in_renderer_evidence():
    source = ("Long display sentence. " * 300) + "Final reservation caveat: 71 min remaining."
    target = graph('root=Text("' + source + '")')
    result = audit_training_fidelity(source, target)
    assert result["renderer_evidence_complete"]
    assert result["evidence_string_bound"] > 4096
    assert result["missing_unit_count"] == 0
    omitted = graph('root=Text("' + ("Long display sentence. " * 300) + '")')
    assert audit_training_fidelity(source, omitted)["missing_unit_count"] == 1


def test_literal_list_and_ordinary_card_labels_count_but_unused_state_does_not():
    target = graph('$/unused="Secret 42"\nroot=Card([items],title="Summary",subtitle="Planning notes")\nitems=List(items=["First complete item","Second complete item"])')
    result = audit_training_fidelity("## Summary\nPlanning notes\nFirst complete item\nSecond complete item", target)
    assert result["status"] == "checks_passed"
    assert result["normal_card_label_count"] == 2
    assert audit_training_fidelity("Secret 42", target)["missing_unit_count"] == 1


def test_table_field_associations_count_and_ignored_table_title_does_not():
    target = graph('root=Table(columns=["Name","Duration"],rows=[["Caligari","71 min"]],title="Film details")')
    associated = audit_training_fidelity("Name: Caligari Duration: 71 min", target)
    assert associated["status"] == "checks_passed"
    assert associated["table_row_binding_context_count"] == 1
    assert audit_training_fidelity("## Film details", target)["missing_unit_count"] == 1
    wrong = graph('root=Table(columns=["Name","Duration"],rows=[["71 min","Caligari"]])')
    assert audit_training_fidelity("Name: Caligari Duration: 71 min", wrong)["missing_unit_count"] == 1


def test_hidden_list_and_email_special_card_title_do_not_gain_credit():
    hidden = graph('root=Column([items])\nitems=List(items=["Hidden source fact"],visible=false)')
    assert audit_training_fidelity("Hidden source fact", hidden)["missing_unit_count"] == 1
    special = graph('root=Card([subject],title="Lost heading")\nsubject=Text("Subject: Update")')
    result = audit_training_fidelity("## Lost heading", special)
    assert result["normal_card_label_count"] == 0
    assert result["missing_unit_count"] == 1


@pytest.mark.parametrize("value", [None, "", "bar", "column", " BAR ", "bar_chart"])
def test_native_supported_chart_aliases_and_default(value):
    assert canonical_chart_subtype(value) in {"bar", "column"}
    assert effective_chart({"chartType": value, "columns": ["Month", "Value"], "rows": [["Jan", 7]]}, {}).complete


@pytest.mark.parametrize("value", ["line", "radar", "pie", "scatter", "stackedbar", "area"])
def test_unsupported_chart_data_is_not_a_renderable_training_target(value):
    completion = f'<a2ui>\nroot=Chart(columns=["Month","Value"],rows=[["Jan",7]],chartType="{value}")\n</a2ui>'
    result = generation_reward_a2ui_express_v1(completion, "| Month | Value |\n|---|---|\n| Jan | 7 |")
    assert result.normalization["production_valid"]
    gate = result.evidence["training_acceptance"]
    assert "renderer_component_contract" in gate["blocking_reasons"]
    assert not gate["eligible"]
    audit = audit_training_fidelity("Jan 7", decode_express_completion(completion))
    assert audit["unsupported_chart_count"] == 1
    assert audit["unsupported_charts"] == [{"component_id": "root", "chart_type": value}]


def test_supported_dynamic_chart_binding_is_not_misclassified():
    completion = '<a2ui>\n$/kind="bar"\nroot=Chart(columns=["Month","Value"],rows=[["Jan",7]],chartType="${/kind}")\n</a2ui>'
    result = generation_reward_a2ui_express_v1(completion, "| Month | Value |\n|---|---|\n| Jan | 7 |")
    assert result.normalization["production_valid"]
    assert "renderer_component_contract" not in result.evidence["training_acceptance"]["blocking_reasons"]


@pytest.mark.parametrize("subtypes", [("line", "bar"), ("bar", "line"), ("bar", "column")])
def test_repeated_chart_contract_checks_every_instance_in_either_order(subtypes):
    first, last = subtypes
    completion = ('<a2ui>\n' + f'$/items=[{{kind:"{first}"}},{{kind:"{last}"}}]\n' +
                  'root=Column([],repeat={statePath:"/items",template:chart})\n' +
                  'chart=Chart(columns=["Month","Value"],rows=[["Jan",7]],chartType="${$item.kind}")\n</a2ui>')
    result = generation_reward_a2ui_express_v1(completion, "| Month | Value |\n|---|---|\n| Jan | 7 |")
    assert result.normalization["production_valid"]
    contract = result.evidence["type_contract"]
    blockers = result.evidence["training_acceptance"]["blocking_reasons"]
    if "line" in subtypes:
        assert contract["per_element"]["chart"] == 0.0
        assert "unsupported_chart_subtype" in contract["diagnostics"]["chart"]
        assert "renderer_component_contract" in blockers
    else:
        assert contract["per_element"]["chart"] == 1.0
        assert "renderer_component_contract" not in blockers


@pytest.mark.parametrize("subtype,expected_status", [("bar", "accepted"), ("line", "quality_rejected")])
def test_stage3_dynamic_chart_subtype_uses_resolved_final_gate(tmp_path, monkeypatch, subtype, expected_status):
    completion = f'<a2ui>\n$/kind="{subtype}"\nroot=Chart(columns=["Month","Value"],rows=[["Jan",7]],chartType="${{/kind}}")\n</a2ui>'
    _, rows = run_fake(tmp_path, monkeypatch, [completion], source="| Month | Value |\n|---|---|\n| Jan | 7 |", metric="v5_4")
    assert rows[0]["record_status"] == expected_status
    if subtype == "line":
        assert "renderer_component_contract" in rows[0]["training_acceptance"]["blocking_reasons"]


def test_stage3_authoritative_metric_preserves_long_literal_evidence(tmp_path, monkeypatch):
    source = ("Long literal display text. " * 200) + "Final caveat is still present."
    _, rows = run_fake(tmp_path, monkeypatch, ['<a2ui>\nroot=Text("' + source + '")\n</a2ui>'],
                       source=source, metric="v5_4")
    result = rows[0]["render_artifact_quality_v5_4"]
    assert result["dynamic_evidence_certification"]["complete"]
    assert "dynamic_evidence_incomplete" not in rows[0]["training_acceptance"]["review_reasons"]
    assert rows[0]["generation_fidelity_audit"]["missing_unit_count"] == 0


def test_actual_generic_stage3_prompt_and_record_include_guidance_and_audit(tmp_path, monkeypatch):
    source = "The Cabinet of Dr Caligari 1920 – 71 min"
    adapter, rows = run_fake(tmp_path, monkeypatch, ['<a2ui>\nroot=Text("The Cabinet of Dr Caligari 1920")\n</a2ui>'], source=source)
    assert source in adapter.calls[0]["prompt"]
    assert "Dataset training fidelity guidance v1" in adapter.calls[0]["prompt"]
    assert rows[0]["record_status"] == "accepted"  # exact-literal diagnostics do not rewrite or waive gates
    assert rows[0]["generation_fidelity_audit"]["status"] == "needs_review"
    assert rows[0]["gen"]["training_fidelity_guidance"] == guidance_identity()


def test_unsupported_chart_participates_in_existing_teacher_repair(tmp_path, monkeypatch):
    bad = '<a2ui>\nroot=Chart(columns=["Month","Value"],rows=[["Jan",7]],chartType="line")\n</a2ui>'
    good = '<a2ui>\nroot=Table(columns=["Month","Value"],rows=[["Jan",7]])\n</a2ui>'
    adapter, rows = run_fake(tmp_path, monkeypatch, [bad, good], repairs=1, source="| Month | Value |\n|---|---|\n| Jan | 7 |")
    assert len(adapter.calls) == 2
    assert "unsupported_chart_subtype" in adapter.calls[1]["prompt"]
    assert "Dataset training fidelity guidance v1" in adapter.calls[0]["prompt"]
    assert rows[0]["validation"]["schema_repair_attempts"] == 1
    assert rows[0]["canonical_graph"]["elements"]["root"]["type"] == "Table"


def test_all_missing_units_are_counted_when_examples_are_bounded():
    source = "\n".join(f"Complete literal item {i}" for i in range(40))
    result = audit_training_fidelity(source, graph('root=Text("Other content")'))
    assert result["checked_source_unit_count"] == result["missing_unit_count"] == 40
    assert len(result["missing_units"]) == 16
    assert result["missing_unit_examples_truncated"]
