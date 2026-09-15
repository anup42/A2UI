"""Report views are evidence-preserving and work after partial failures."""
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline import result_tables as module


def evaluation(count, *, reward=72.5, strict=.875, unique=71.2):
    return {"row_count": count, "aggregate": {module.REWARD_METRIC: reward,
            module.STRICT_METRIC: strict, module.UNIQUE_REWARD_METRIC: unique}}


def trial(name):
    return {"name": name, "parameters": {"learning_rate": 2e-5, "weight_decay": .01,
                                          "warmup_ratio": .03, "augmentation": "none"}}


def experiment(tmp_path):
    planned = [trial("baseline"), trial("candidate"), trial("not_started")]
    baseline = {**planned[0], "status": "complete", "elapsed_seconds": 61.2,
                "evaluations": {"best_golden32": evaluation(32), "final_golden32": evaluation(32, unique=69.1)}}
    return {"plan": {"output_dir": str(tmp_path), "trials": planned, "evaluate_selected_holdout": False},
            "trials": [baseline], "status": "failed", "active_trial": "candidate", "error": "RuntimeError: native engine failed"}


def test_deployment_includes_all_fourteen_slots_and_correct_weighting():
    state = {"status": "complete", "results": {
        f"{label}_{cohort}": evaluation(count)
        for label in module.DEPLOYMENT_LABELS for cohort, count in (("golden32", 32), ("golden35", 35))}}
    before = deepcopy(state)
    text = module.render_deployment_results(state)
    assert state == before
    assert text.count("| evaluated") == 14
    assert "32/32" in text and "35/35" in text and "87.5" in text
    assert "72.5000" in text and "71.2000" in text
    assert "32 occurrences / 31 unique sources" in text
    assert "equal weight (selection metric)" in text
    assert "quality-threshold pass" in text


def test_deployment_failure_does_not_invent_remaining_scores():
    state = {"status": "failed", "active_stage": "w32_golden32", "error": "missing Vulkan",
             "results": {"checkpoint_best_golden32": evaluation(32)}}
    text = module.render_deployment_results(state)
    rows = [line for line in text.splitlines() if line.startswith("| ")]
    assert len(rows) == 16
    assert any("w32" in row and "Golden32" in row and "failed" in row for row in rows)
    assert any("w4" in row and "Golden35" in row and "not reported" in row and "unknown/35" in row for row in rows)
    assert "0.0000" not in text
    assert "Stopped at: w32_golden32" in text
    assert "missing Vulkan" in text


def test_export_failure_marks_both_cohorts_blocked():
    text = module.render_deployment_results({"status": "failed", "active_stage": "export_w16", "results": {}})
    assert text.count("blocked: export failed") == 2


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, "0.5"])
def test_missing_or_nonfinite_metrics_are_unknown_not_zero(value):
    text = module.render_deployment_results({"status": "failed", "results": {
        "w32_golden32": evaluation(32, reward=value, unique=value, strict=value)}})
    row = next(row for row in text.splitlines() if row.startswith("| w32") and "Golden32" in row)
    assert row.count("unknown") == 3
    assert "0.0000" not in row


def test_genuine_zero_is_preserved_and_partial_count_is_not_complete():
    text = module.render_deployment_results({"status": "failed", "results": {
        "w32_golden32": evaluation(12, reward=0, unique=0, strict=0)}})
    row = next(row for row in text.splitlines() if row.startswith("| w32") and "Golden32" in row)
    assert "incomplete" in row and "12/32" in row and "0.0000" in row


def test_hpo_failure_prints_completed_failed_and_not_started_trials(tmp_path):
    state = experiment(tmp_path)
    before = deepcopy(state)
    text = module.render_experiment_results(state)
    assert state == before
    assert "baseline" in text and "complete" in text
    assert "candidate" in text and "failed" in text
    assert "not_started" in text and "not run" in text
    assert "71.2000 / 69.1000" in text
    assert "87.5 / 87.5" in text and "32/32 / 32/32" in text
    assert "Locked winner: not selected" in text
    assert "Golden35: intentionally deferred" in text
    assert "2e-05/0.01/0.03" in text
    assert "72.5000" not in text  # trial selection uses unique-source, not occurrence score


def test_holdout_is_separate_and_never_reselects_winner(tmp_path):
    state = experiment(tmp_path)
    state.update(status="complete", selected_trial="baseline", active_trial=None, selected_golden35=evaluation(35, reward=1.25))
    state["plan"]["evaluate_selected_holdout"] = True
    state.pop("error")
    text = module.render_experiment_results(state)
    assert "baseline [selected]" in text and "1.2500" in text
    assert "holdout (not used for selection)" in text
    assert "Golden35 never selects a trial" in text


def test_holdout_failure_preserves_locked_selection(tmp_path):
    state = experiment(tmp_path)
    state["plan"]["evaluate_selected_holdout"] = True
    state.update(selected_trial="baseline", active_trial=None)
    text = module.render_experiment_results(state)
    assert "baseline [selected]" in text
    assert "no verified result reported" in text


def test_success_state_missing_a_planned_trial_is_not_called_not_run(tmp_path):
    state = experiment(tmp_path)
    state.update(status="complete", active_trial=None)
    text = module.render_experiment_results(state)
    row = next(row for row in text.splitlines() if row.startswith("| candidate"))
    assert "not reported" in row


def test_publish_full_report_and_preserve_json_artifacts(tmp_path, capsys):
    state = experiment(tmp_path)
    original = tmp_path / "experiments_manifest.json"
    original.write_text(json.dumps(state), encoding="utf-8")
    before = original.read_bytes()
    path = module.publish_experiment_results(state)
    assert path == tmp_path / "comparison.md"
    assert path.read_text(encoding="utf-8") == module.render_experiment_results(state)
    assert "# Hyperparameter tuning results" in capsys.readouterr().out
    assert original.read_bytes() == before
    assert not path.with_suffix(".md.partial").exists()


def test_combined_deployment_report_accepts_attempt_path_and_tuning(tmp_path, capsys):
    state = {"plan": {"output_dir": str(tmp_path)}, "status": "failed", "results": {}}
    destination = tmp_path / "attempts/2/results.md"
    assert module.publish_deployment_results(state, tuning_state=experiment(tmp_path), path=destination) == destination
    content = destination.read_text(encoding="utf-8")
    assert "# Deployment results" in content and "# Hyperparameter tuning results" in content
    assert "not_started" in capsys.readouterr().out


def test_unwritable_report_still_prints_without_masking_training_failure(tmp_path, capsys):
    bad_parent = tmp_path / "not_a_directory"
    bad_parent.write_text("fixture", encoding="utf-8")
    assert module.publish_experiment_results(experiment(tmp_path), path=bad_parent / "results.md") is None
    text = capsys.readouterr().out
    assert "Could not persist final result table" in text
    assert "native engine failed" in text


def test_no_trials_yet_is_safe(tmp_path):
    assert "No trial results" in module.render_experiment_results({"plan": {"output_dir": str(tmp_path)}})


def test_manifest_text_cannot_break_table_rows():
    text = module.render_deployment_results({"status": "failed\n|evil|", "error": "first\r\nsecond|third"})
    assert "Run status: failed /evil/" in text
    assert "Failure: first second/third" in text


def test_partial_manifest_nulls_do_not_break_failure_report(tmp_path):
    state = experiment(tmp_path)
    state["trials"][0]["evaluations"]["best_golden32"] = None
    assert "unknown/32" in module.render_experiment_results(state)
    text = module.render_deployment_results({"status": "failed", "results": {"w32_golden32": None}})
    assert "not reported" in text and "unknown/32" in text


def test_ansi_control_sequences_are_not_executed_by_report():
    text = module.render_deployment_results({"error": "bad\x1b[31merror"})
    assert "\x1b" not in text
