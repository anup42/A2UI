"""CPU orchestration contracts for exporting packages without native runtime tests.

Fixtures stand in for model/GPU work. These tests do not claim an actual H100,
LiteRT conversion, Vulkan runtime, or benchmark quality result.
"""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import test_golden_deployment as fixtures
import yaml

from ir_training.pipeline import golden_deployment as deployment
from test_golden_deployment import Writer, dump, fake_pipeline, inventory, mock_runner

options = fixtures.options
export_validator = fixtures.export_validator
COHORTS = ("golden32", "golden35", "bixby50")
VARIANTS = ("w32", "w16", "w8", "w4")
HF_KEYS = {f"{label}_{cohort}" for label in ("checkpoint_best", "checkpoint_final", "merged")
           for cohort in COHORTS}
LITERT_KEYS = {f"{variant}_{cohort}" for variant in VARIANTS for cohort in COHORTS}


def export_only(options, **changes):
    return replace(options, skip_litert_evaluation=True, runtime_python=None, **changes)


def run(options, *, calls=None, failure=None, pipeline=fake_pipeline, writer=None,
        gpu_probe=inventory, experiment=None):
    extra = {"experiment_runner": experiment} if experiment is not None else {}
    return deployment.run_deployment(options, execute=True,
        command_runner=mock_runner([] if calls is None else calls, failure=failure),
        pipeline_runner=pipeline, writer_factory=lambda **_: writer or Writer(),
        gpu_probe=gpu_probe, **extra)


def no_training(*args, **kwargs):
    pytest.fail("Verified completed training must not rerun during recovery")


def test_skip_plan_keeps_checkpoint_tests_and_every_export_without_runtime(options):
    selected = export_only(options)
    plan = deployment.build_deployment_plan(selected)
    assert plan["skip_litert_evaluation"] is True
    assert plan["runtime_python"] is None and plan["runtime_probe_command"] is None
    assert set(plan["training"]["goldens"]) == set(COHORTS)
    assert {f"{role}_{cohort}" for role in ("best", "final") for cohort in COHORTS}.issubset(
        plan["training"]["stages"])
    assert "runtime_preflight" not in plan["stages"]
    assert not LITERT_KEYS.intersection(plan["stages"])
    assert {f"merged_{cohort}" for cohort in COHORTS}.issubset(plan["stages"])
    assert {f"export_{variant}" for variant in VARIANTS}.issubset(plan["stages"])
    assert set(plan["export"]["variants"]) == set(VARIANTS)
    assert not selected.base.output_dir.exists()


def test_default_plan_still_requires_runtime_and_all_native_evaluations(options):
    plan = deployment.build_deployment_plan(options)
    assert not plan.get("skip_litert_evaluation", False)
    assert plan["runtime_python"] == str(options.runtime_python.absolute())
    assert "runtime_preflight" in plan["stages"]
    assert LITERT_KEYS.issubset(plan["stages"])
    with pytest.raises(ValueError, match="runtime-python"):
        deployment.build_deployment_plan(replace(options, runtime_python=None))


@pytest.mark.parametrize("profile", ["e2b", "270m"])
@pytest.mark.parametrize("gpu_count", [2, 4, 8])
def test_skip_run_retains_hf_gpu_tests_all_exports_and_honest_scorecard(
        options, export_validator, profile, gpu_count, monkeypatch):
    selected = export_only(options, base=replace(options.base, profile=profile))
    calls, writer, trained = [], Writer(), []

    def cuda_inventory():
        value = inventory(gpu_count)
        # UUID/Vulkan allocation checks belong only to native evaluation. CUDA
        # training and HF evaluation still require the selected GPU inventory.
        for device in value["devices"]:
            device["uuid"] = None
        return value

    def forbid_native_uuid_lookup(*args, **kwargs):
        pytest.fail("Export-only mode must not consult native GPU UUIDs")

    monkeypatch.setattr(deployment, "_litert_allowed_gpu_uuids", forbid_native_uuid_lookup)

    def train(base, **kwargs):
        trained.append(base)
        result = fake_pipeline(base, **kwargs)
        path = base.output_dir / "fit/training_config.yaml"
        config = yaml.safe_load(path.read_text())
        config["runtime"]["gpu_profile"]["world_size"] = gpu_count
        config["runtime"]["cuda_visible_devices"] = ",".join(map(str, range(gpu_count)))
        path.write_text(yaml.safe_dump(config))
        return result

    result = run(selected, calls=calls, pipeline=train, writer=writer, gpu_probe=cuda_inventory)
    assert result["status"] == "complete" and set(result["results"]) == HF_KEYS
    assert len(trained) == 1 and trained[0].evaluate_golden35 and trained[0].evaluate_bixby50
    assert "runtime_preflight" not in result["completed"]
    assert not LITERT_KEYS.intersection(result["completed"])
    assert not any("--preflight" in argv or "--builtin-gpu" in argv for argv, _, _ in calls)
    assert not any(any(Path(part).name == "evaluate_litertlm_on_golden.py" for part in argv)
                   for argv, _, _ in calls)
    merged_calls = [(argv, env) for argv, env, _ in calls if "--evaluation-name" in argv]
    assert len(merged_calls) == 3
    for argv, env in merged_calls:
        assert "--require-gpu" in argv
        assert argv[argv.index("--devices") + 1] == "auto"
        assert env["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, range(gpu_count)))
    assert len([argv for argv, _, _ in calls if "convert" in argv]) == 4
    assert set(result["exports"]) == set(VARIANTS)
    for variant in VARIANTS:
        assert result["completed"][f"export_{variant}"]["files"]

    scorecard = json.loads((selected.base.output_dir / "deployment_scorecard.json").read_text())
    assert scorecard["skip_litert_evaluation"] is True
    assert scorecard["litert_evaluation_status"] == "skipped_by_request"
    assert scorecard["native_runtime_validated"] is False
    assert scorecard["completion_scope"] == "checkpoint_tests_and_exports"
    assert set(scorecard["skipped_evaluations"]) == LITERT_KEYS
    assert set(scorecard["results"]) == HF_KEYS
    assert set(scorecard["variants"]) == set(VARIANTS)
    assert scorecard["golden35_used_for_selection"] is False
    assert scorecard["bixby50_used_for_selection"] is False
    assert scorecard["evaluation_cohorts"]["bixby50"]["reference_available"] is False

    parameters, metrics = writer.hparams[0]
    assert parameters["litert_evaluation_enabled"] is False
    assert parameters["training_gpus"] == parameters["evaluation_gpus"] == gpu_count
    assert all(key.split("/")[1] in HF_KEYS for key in metrics)
    assert not any(args[0].startswith(tuple(f"evaluation/{variant}/" for variant in VARIANTS))
                   for args in writer.scalars)


def test_skip_mode_still_requires_gpu_for_training_and_checkpoint_tests(options, export_validator):
    with pytest.raises(ValueError):
        run(export_only(options), pipeline=no_training, gpu_probe=lambda: inventory(0))
    state = json.loads((options.base.output_dir / "deployment_manifest.json").read_text())
    assert state["status"] == "failed" and state["active_stage"] == "host_preflight"
    assert not (options.base.output_dir / "deployment_scorecard.json").exists()


@pytest.mark.parametrize("failure", ["exporter_preflight", "merged_bixby50", "export_w4"])
def test_skip_does_not_skip_exporter_failures_hf_holdouts_or_failed_exports(options, export_validator, failure):
    with pytest.raises(RuntimeError, match="simulated"):
        run(export_only(options), failure=failure)
    state = json.loads((options.base.output_dir / "deployment_manifest.json").read_text())
    assert state["status"] == "failed" and state["active_stage"] == failure
    assert not (options.base.output_dir / "deployment_scorecard.json").exists()


def test_skip_mode_still_validates_each_exported_package(options, export_validator, monkeypatch):
    validator = deployment.validate_deployment_export_output
    checked = []

    def reject_invalid_package(plan, variant):
        checked.append(variant)
        if variant == "w8":
            raise ValueError("invalid package precision fixture")
        return validator(plan, variant)

    monkeypatch.setattr(deployment, "validate_deployment_export_output", reject_invalid_package)
    with pytest.raises(ValueError, match="invalid package"):
        run(export_only(options))
    assert checked == ["w32", "w16", "w8"]
    assert not (options.base.output_dir / "deployment_scorecard.json").exists()
    state = json.loads((options.base.output_dir / "deployment_manifest.json").read_text())
    assert state["active_stage"] == "export_w8" and "w8" not in state["exports"]


def test_skip_resume_reuses_training_hf_tests_and_successful_exports(options, export_validator):
    selected = export_only(options)
    with pytest.raises(RuntimeError, match="simulated"):
        run(selected, failure="export_w16")
    calls = []
    result = run(replace(selected, resume_run=True), calls=calls, pipeline=no_training)
    assert result["status"] == "complete" and set(result["results"]) == HF_KEYS
    assert not any("prepare" in argv or "--evaluation-name" in argv or "--preflight" in argv
                   for argv, _, _ in calls)
    assert [argv[argv.index("--variant") + 1] for argv, _, _ in calls if "convert" in argv] == [
        "w16", "w8", "w4"]
    assert (selected.base.output_dir / "recovery/attempt_0002/deployment/variants/w16/model.litertlm").is_file()


@pytest.mark.parametrize("first_skip", [False, True])
def test_resume_cannot_flip_native_evaluation_policy(options, export_validator, first_skip):
    first = export_only(options) if first_skip else options
    with pytest.raises(RuntimeError, match="simulated"):
        run(first, failure="export_w16")
    second = replace(options, resume_run=True, skip_litert_evaluation=not first_skip,
                     runtime_python=options.runtime_python if first_skip else None)
    calls = []
    with pytest.raises(ValueError, match="options differ"):
        run(second, calls=calls, pipeline=no_training)
    assert not calls


def test_skip_completed_resume_is_read_only(options, export_validator):
    selected = export_only(options)
    run(selected)
    path = selected.base.output_dir / "deployment_manifest.json"
    before = path.read_bytes()
    calls = []
    result = run(replace(selected, resume_run=True), calls=calls, pipeline=no_training)
    assert result["status"] == "complete" and not calls and path.read_bytes() == before


def test_skip_mode_preserves_tuning_then_full_training_and_all_holdouts(options, export_validator):
    observed = []

    def experiment(opts, **kwargs):
        assert opts.evaluate_selected_holdout is False
        directory = opts.base.output_dir
        for name in ("experiments_manifest.json", "selection_locked.json", "comparison.json"):
            dump(directory / name, {})
        parameters = {"learning_rate": 1e-5, "weight_decay": .01, "warmup_ratio": .03,
                      "augmentation": "none"}
        return {"status": "complete", "selected_golden35": None, "selected_bixby50": None,
                "selected_trial": "lr_half", "plan": {"trials": [
                    {"name": "lr_half", "parameters": parameters}]}}

    def train(base, **kwargs):
        observed.append(base)
        return fake_pipeline(base, **kwargs)

    result = run(export_only(options, tune=True), experiment=experiment, pipeline=train)
    assert result["status"] == "complete" and set(result["results"]) == HF_KEYS
    assert len(observed) == 1 and observed[0].learning_rate == 1e-5
    assert observed[0].steps is None and observed[0].evaluate_golden35 and observed[0].evaluate_bixby50
    assert "lock_hyperparameters" in result["completed"]


@pytest.mark.parametrize("skip", [False, True])
def test_cli_runtime_python_is_optional_only_when_skipping_native_tests(options, skip):
    script = Path(__file__).resolve().parents[1] / "scripts/run_golden_deployment.py"
    command = [sys.executable, str(script), "--profile", options.base.profile,
               "--model-dir", str(options.base.model_dir), "--input-dir", str(options.base.input_dir),
               "--output-dir", str(options.base.output_dir), "--exporter-python", sys.executable,
               "--allow-experimental-formats"]
    if skip:
        command.append("--skip-litert-evaluation")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if skip:
        assert completed.returncode == 0, completed.stderr
        plan = json.loads(completed.stdout)
        assert plan["skip_litert_evaluation"] is True and plan["runtime_python"] is None
        assert "runtime_preflight" not in plan["stages"]
        assert plan["training_executed"] is False
    else:
        assert completed.returncode == 2
        assert "runtime-python" in completed.stderr
    assert not options.base.output_dir.exists()
