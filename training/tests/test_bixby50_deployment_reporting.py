"""Bixby50 coverage, isolated holdout reporting and recovery without a GPU run."""
import json
from dataclasses import replace

import pytest
import test_golden_deployment as fixtures
from ir_training.pipeline import golden_deployment as deployment
from ir_training.pipeline import result_tables
from test_golden_deployment import Writer, fake_pipeline, inventory, mock_runner

options = fixtures.options
export_validator = fixtures.export_validator


def evaluation(count, reward=72.5):
    return {"row_count": count, "aggregate": {result_tables.REWARD_METRIC: reward,
            result_tables.STRICT_METRIC: .9, result_tables.UNIQUE_REWARD_METRIC: reward - 1}}


def cohort_plan():
    return {"goldens": {"golden32": {"rows": 32}, "golden35": {"rows": 35},
                        "bixby50": {"rows": 50, "reference_available": False,
                                    "benchmark_kind": "source_only_holdout"}}}


def run(options, *, calls=None, failure=None, writer=None, pipeline=fake_pipeline, experiment=None):
    extra = {"experiment_runner": experiment} if experiment is not None else {}
    return deployment.run_deployment(options, execute=True,
        command_runner=mock_runner([] if calls is None else calls, failure=failure),
        pipeline_runner=pipeline, writer_factory=lambda **_: writer or Writer(), gpu_probe=inventory,
        **extra)


def test_new_deployment_plan_requires_all_three_cohorts(options):
    plan = deployment.build_deployment_plan(options)
    assert list(plan["training"]["goldens"]) == ["golden32", "golden35", "bixby50"]
    assert "best_bixby50" in plan["training"]["stages"]
    assert "final_bixby50" in plan["training"]["stages"]
    for label in ("merged", "w32", "w16", "w8", "w4"):
        assert f"{label}_bixby50" in plan["stages"]
    assert "Golden35 and Bixby50 never select" in plan["selection_policy"]
    with pytest.raises(ValueError, match="requires Bixby50"):
        deployment.build_deployment_plan(replace(options, base=replace(options.base, evaluate_bixby50=False)))


def test_bixby50_gpu_commands_scorecard_and_minimal_tensorboard(options, export_validator):
    calls, writer = [], Writer()
    result = run(options, calls=calls, writer=writer)
    bixby_calls = [argv for argv, _, _ in calls if "--evaluation-name" in argv
                  and argv[argv.index("--evaluation-name") + 1].endswith("_bixby50")]
    assert len(bixby_calls) == 5  # merged HF plus four LiteRT weight precisions
    for argv in bixby_calls:
        assert argv[argv.index("--required-rows") + 1] == "50"
        assert argv[argv.index("--split") + 1].endswith("bixby50.jsonl")
        assert "--builtin-gpu" in argv or "--require-gpu" in argv
    for label in result_tables.DEPLOYMENT_LABELS:
        assert result["results"][f"{label}_bixby50"]["row_count"] == 50
        assert any(args[0] == f"evaluation/{label}/bixby50/{result_tables.REWARD_METRIC}"
                   for args in writer.scalars)
    assert f"hparam/w4_bixby50/{result_tables.REWARD_METRIC}" in writer.hparams[0][1]
    scorecard = json.loads((options.base.output_dir / "deployment_scorecard.json").read_text())
    assert scorecard["bixby50_used_for_selection"] is False
    assert scorecard["evaluation_cohorts"]["bixby50"] == {
        "required_rows": 50, "reference_available": False, "benchmark_kind": "source_only_holdout"}
    report = (options.base.output_dir / "deployment_results.md").read_text()
    assert report.count("| evaluated") == 21
    assert "50/50" in report and "Bixby50 uses source-response scoring; no reference IR." in report


@pytest.mark.parametrize("failure", ["merged_bixby50", "w32_bixby50", "w4_bixby50"])
def test_bixby50_failure_blocks_success_but_preserves_completed_evidence(options, export_validator, failure):
    with pytest.raises(RuntimeError, match="simulated"):
        run(options, failure=failure)
    output = options.base.output_dir
    assert not (output / "deployment_scorecard.json").exists()
    state = json.loads((output / "deployment_manifest.json").read_text())
    assert state["status"] == "failed" and state["active_stage"] == failure
    assert state["results"]["checkpoint_best_bixby50"]["row_count"] == 50
    report = (output / "deployment_results.md").read_text()
    assert any("Bixby50" in row and "failed" in row for row in report.splitlines() if row.startswith("|"))


def test_resume_failed_bixby50_does_not_retrain_or_repeat_prior_cohorts(options, export_validator):
    with pytest.raises(RuntimeError, match="simulated"):
        run(options, failure="w4_bixby50")
    def no_training(*args, **kwargs):
        pytest.fail("Recovery must not retrain")
    calls = []
    result = run(replace(options, resume_run=True), calls=calls, pipeline=no_training)
    assert len(result["results"]) == 21 and result["status"] == "complete"
    evaluations = [argv for argv, _, _ in calls if "--evaluation-name" in argv]
    assert len(evaluations) == 1
    assert evaluations[0][evaluations[0].index("--evaluation-name") + 1] == "w4_bixby50"


@pytest.mark.parametrize("location", ["selected", "trial"])
def test_deployment_screening_rejects_any_bixby50_evaluation(options, export_validator, location):
    def experiment(opts, **kwargs):
        assert opts.evaluate_selected_holdout is False
        result = {"status": "complete", "selected_golden35": None, "selected_bixby50": None}
        if location == "selected":
            result["selected_bixby50"] = evaluation(50)
        else:
            result["trials"] = [{"evaluations": {"best_bixby50": evaluation(50)}}]
        return result
    with pytest.raises(ValueError, match="without evaluating Golden35 or Bixby50"):
        run(replace(options, tune=True), experiment=experiment)
    assert not (options.base.output_dir / "deployment_scorecard.json").exists()


def test_final_table_shows_all_21_planned_slots_even_before_inference():
    state = {"status": "failed", "active_stage": "runtime_preflight", "results": {},
             "plan": {"training": cohort_plan()}}
    text = result_tables.render_deployment_results(state)
    assert text.count("| not reported") == 21
    assert text.count("unknown/50") == 7
    assert "Bixby50 uses source-response scoring; no reference IR." in text
    assert "0.0000" not in text


def test_partial_bixby50_count_is_never_complete_or_golden35():
    text = result_tables.render_deployment_results({"status": "failed", "results": {
        "w4_bixby50": evaluation(49)}})
    row = next(line for line in text.splitlines() if line.startswith("| w4") and "Bixby50" in line)
    assert "49/50" in row and "incomplete" in row and "n/a" in row
    assert "Golden35" not in row


def test_tuning_report_lists_both_selected_holdouts_without_using_them_for_selection():
    state = {"status": "complete", "selected_trial": "baseline", "trials": [],
             "plan": {**cohort_plan(), "evaluate_selected_holdout": True},
             "selected_golden35": evaluation(35, 10), "selected_bixby50": evaluation(50, 20)}
    text = result_tables.render_experiment_results(state)
    assert "Golden35 never selects a trial." in text and "Bixby50 never selects a trial." in text
    assert "35/35" in text and "50/50" in text and "20.0000" in text
    assert "Locked winner: baseline." in text


def test_tuning_report_explains_both_deferred_holdouts():
    text = result_tables.render_experiment_results({"status": "complete", "selected_trial": "baseline",
        "plan": {"trials": [{"name": "baseline", "plan": cohort_plan()}], "evaluate_selected_holdout": False}})
    assert "Golden35: intentionally deferred" in text and "Bixby50: intentionally deferred" in text


def test_tuning_report_preserves_first_holdout_when_second_fails():
    text = result_tables.render_experiment_results({"status": "failed", "selected_trial": "baseline",
        "plan": {**cohort_plan(), "evaluate_selected_holdout": True}, "selected_golden35": evaluation(35)})
    assert "35/35" in text
    assert "Bixby50: no verified result reported" in text
    assert "Locked winner: baseline." in text


def test_old_dual_cohort_manifest_does_not_invent_bixby50():
    text = result_tables.render_deployment_results({"status": "complete", "results": {
        f"{label}_{cohort}": evaluation(count) for label in result_tables.DEPLOYMENT_LABELS
        for cohort, count in (("golden32", 32), ("golden35", 35))}})
    assert text.count("| evaluated") == 14
    assert "Bixby50" not in text
