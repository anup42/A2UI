"""Recovery uses local fixture artifacts, never a real GPU or trained model."""
import json
from dataclasses import replace

import pytest
import test_golden_deployment as fixtures
from ir_training.pipeline import golden_deployment as deployment
from ir_training.pipeline.deployment_recovery import deployment_lock
from test_golden_deployment import (
    Writer,
    dump,
    fake_pipeline,
    inventory,
    mock_runner,
)

options = fixtures.options
export_validator = fixtures.export_validator


def run(options, calls, *, failure=None, pipeline=fake_pipeline):
    return deployment.run_deployment(options, execute=True, command_runner=mock_runner(calls, failure=failure),
        pipeline_runner=pipeline, writer_factory=Writer, gpu_probe=inventory)


def fail_after_training(options, calls, failure="w32_golden32"):
    with pytest.raises(RuntimeError, match="simulated"):
        run(options, calls, failure=failure)


def no_training(*args, **kwargs):
    pytest.fail("Completed training must never rerun during recovery")


def test_resume_native_failure_preserves_artifacts_and_uses_fresh_attempt(options, export_validator, capsys):
    fail_after_training(options, [])
    output = options.base.output_dir
    dump(output / "evaluations/w32_golden32/partial.json", {"retained": True})
    before = (output / "deployment/variants/w32/model.litertlm").read_bytes()
    calls = []
    result = run(replace(options, resume_run=True), calls, pipeline=no_training)
    assert result["status"] == "complete" and len(result["results"]) == 21
    assert (output / "deployment/variants/w32/model.litertlm").read_bytes() == before
    assert (output / "evaluations/w32_golden32/partial.json").exists()
    assert not any("prepare" in argv for argv, _, _ in calls)
    assert not any("convert" in argv and "w32" in argv for argv, _, _ in calls)
    assert len([argv for argv, _, _ in calls if "--builtin-gpu" in argv]) == 12
    assert (output / "recovery/attempt_0002/evaluations/w32_golden32/evaluation_result.json").exists()
    assert (output / "recovery/attempt_0002/previous_deployment_manifest.json").exists()
    text = capsys.readouterr().out
    assert "checkpoint_best" in text and "w4" in text and "Golden35" in text
    assert "failed" in text and "complete" in text


@pytest.mark.parametrize("failure", ["merge", "export_w16", "w4_golden35"])
def test_resume_other_post_training_failures(options, export_validator, failure):
    fail_after_training(options, [], failure)
    result = run(replace(options, resume_run=True), [], pipeline=no_training)
    assert result["status"] == "complete"


def test_legacy_manifest_without_new_options_can_recover(options, export_validator):
    fail_after_training(options, [])
    path = options.base.output_dir / "deployment_manifest.json"
    state = json.loads(path.read_text())
    state.pop("attempt")
    state.pop("artifact_directories")
    state["plan"]["training"]["options"].pop("tensorboard_detail", None)
    dump(path, state)
    assert run(replace(options, resume_run=True), [], pipeline=no_training)["status"] == "complete"


def test_completed_run_is_read_only_and_does_not_repeat_evaluations(options, export_validator):
    run(options, [])
    path = options.base.output_dir / "deployment_manifest.json"
    before = path.read_bytes()
    calls = []
    assert run(replace(options, resume_run=True), calls, pipeline=no_training)["status"] == "complete"
    assert not calls and path.read_bytes() == before


@pytest.mark.parametrize("tamper", ["package", "merged", "summary", "export_summary"])
def test_resume_rejects_changed_evidence_before_any_gpu_work(options, export_validator, tamper):
    fail_after_training(options, [])
    output = options.base.output_dir
    if tamper == "package":
        dump(output / "deployment/variants/w32/model.litertlm", {"changed": True})
    elif tamper == "merged":
        dump(output / "deployment/merged_hf/model.safetensors", {"changed": True})
    else:
        path = output / "deployment_manifest.json"
        state = json.loads(path.read_text())
        if tamper == "summary":
            state["results"]["checkpoint_best_golden32"]["aggregate"]["generation_reward_v5_4_avg"] = 100
        else:
            state["exports"]["w32"]["actual_precision"] = {"tampered": True}
        dump(path, state)
    calls = []
    with pytest.raises(ValueError, match="changed|differs"):
        run(replace(options, resume_run=True), calls, pipeline=no_training)
    assert not calls


def test_resume_rejects_changed_semantic_options(options, export_validator):
    fail_after_training(options, [])
    with pytest.raises(ValueError, match="options differ"):
        run(replace(options, resume_run=True, base=replace(options.base, max_new_tokens=1024)), [])


def test_resume_does_not_restart_incomplete_training(options, export_validator):
    fail_after_training(options, [], "runtime_preflight")
    with pytest.raises(ValueError, match="AFTER completed full training"):
        run(replace(options, resume_run=True), [], pipeline=no_training)


def test_resume_runs_new_vulkan_preflight_before_evaluation(options, export_validator):
    fail_after_training(options, [])
    calls = []
    with pytest.raises(RuntimeError, match="simulated"):
        run(replace(options, resume_run=True), calls, failure="runtime_preflight", pipeline=no_training)
    assert not any("--builtin-gpu" in argv or "convert" in argv for argv, _, _ in calls)


def test_deployment_lock_rejects_concurrent_writer_and_releases(tmp_path):
    output = tmp_path / "locked_run"
    with deployment_lock(output), pytest.raises(RuntimeError, match="Another deployment"), deployment_lock(output):
        pytest.fail("Concurrent lock unexpectedly acquired")
    with deployment_lock(output):
        assert not output.exists()


def test_multiple_recoveries_preserve_previous_attempts(options, export_validator):
    fail_after_training(options, [])
    resumed = replace(options, resume_run=True)
    fail_after_training(resumed, [], "w16_golden35")
    calls = []
    result = run(resumed, calls, pipeline=no_training)
    assert result["status"] == "complete" and result["attempt"] == 3
    assert (options.base.output_dir / "recovery/attempt_0002/evaluations/w16_golden32/evaluation_result.json").exists()
    assert (options.base.output_dir / "recovery/attempt_0003/evaluations/w16_golden35/evaluation_result.json").exists()
    assert len([argv for argv, _, _ in calls if "--builtin-gpu" in argv]) == 8


def test_resume_allows_only_dashboard_and_deadline_adjustment(options, export_validator):
    fail_after_training(options, [])
    result = run(replace(options, resume_run=True, generation_timeout_seconds=9000,
                         base=replace(options.base, tensorboard_detail="full", progress_seconds=5)), [], pipeline=no_training)
    assert result["status"] == "complete"
