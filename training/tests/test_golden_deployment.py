"""CPU-only orchestration/recovery contracts; never a real LiteRT/H100 run."""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline import golden_deployment as deployment
from ir_training.pipeline.golden_training import GoldenTrainingOptions, sha256


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Writer:
    def __init__(self, **kwargs):
        self.scalars, self.hparams = [], []

    def add_scalar(self, *args, **kwargs):
        self.scalars.append(args)

    def add_text(self, *args, **kwargs):
        pass

    def add_hparams(self, *args, **kwargs):
        self.hparams.append(args)

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture
def options(tmp_path):
    model, data = tmp_path / "model", tmp_path / "data"
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        dump(model / name, {})
    for name in ("train.jsonl", "val.jsonl"):
        dump(data / name, {})
    return deployment.GoldenDeploymentOptions(
        base=GoldenTrainingOptions(model_dir=model, input_dir=data, output_dir=tmp_path / "run",
                                   tensorboard_root=str(tmp_path / "tensorboard")),
        exporter_python=Path(sys.executable), runtime_python=Path(sys.executable),
        allow_experimental_formats=True)


def gpu_uuid(index):
    return f"GPU-12345678-abcd-4321-abcd-{index:012x}"


def inventory(count=8):
    return {"visible_gpu_count": count, "devices": [
        {"visible_index": i, "launch_identifier": str(i), "uuid": gpu_uuid(i), "name": "NVIDIA H100 80GB",
         "total_memory_bytes": 80 * 1024**3, "compute_capability": [9, 0]} for i in range(count)]}


def evaluation(path, count, artifact, *, litert=False, gpu_uuids=None):
    aggregate = {"generation_reward_v5_4_avg": .5, "unique_source_generation_reward_v5_4_avg": .49}
    result = {"model" if litert else "checkpoint": str(artifact), "aggregate": aggregate,
              "row_count": count, "tensorboard_record": str(path / "tb.json")}
    dump(path / "aggregate_metrics.json", aggregate)
    dump(path / "tb.json", {})
    for name in ("predictions.jsonl", "scored_predictions.jsonl"):
        (path / name).write_text("{}\n" * count)
    if litert:
        manifest = {"row_count": count, "model_sha256": sha256(artifact),
                    "gpu_execution": {"requested_backend": "gpu", "engine_backend": "gpu", "allocation_observed": True,
                                      "tokenizer_parity_passed": True, "gpu_uuids": [gpu_uuid(0)] if gpu_uuids is None else gpu_uuids}}
        for field in ("runner_outputs_path", "requests_path", "runner_log_path"):
            target = path / (field + ".json")
            dump(target, {})
            manifest[field] = str(target)
        dump(path / "external_runner_manifest.json", manifest)
        result["external_runner_manifest"] = str(path / "external_runner_manifest.json")
    dump(path / "evaluation_result.json", result)
    return result


def fake_pipeline(base, *, execute, command_runner):
    assert execute and base.evaluate_golden35
    output = base.output_dir
    config = {"runtime": {"cuda_visible_devices": "0,1,2,3,4,5,6,7", "gpu_profile": {"world_size": 8, "effective_batch_size": 32}},
              "training": {"learning_rate": base.learning_rate or 2e-5, "weight_decay": .01, "warmup_ratio": .03}}
    (output / "fit").mkdir(parents=True)
    (output / "fit/training_config.yaml").write_text(yaml.safe_dump(config))
    completed = {}
    for role in ("best", "final"):
        checkpoint = output / "fit/training" / ("best_golden_checkpoint" if role == "best" else
                      ("final_adapter" if base.profile == "e2b" else "final_model"))
        dump(checkpoint / "training_metadata.json", {"checkpoint_step": 100, "checkpoint_role": role,
                                                     "best_golden_eval": {"step": 80}})
        for cohort, count in (("golden32", 32), ("golden35", 35)):
            path = output / "evaluations" / f"{role}_{cohort}"
            evaluation(path, count, checkpoint)
            completed[f"{role}_{cohort}"] = {"files": {str(path / "evaluation_result.json"): sha256(path / "evaluation_result.json")}}
    result = {"status": "complete", "completed": completed}
    dump(output / "pipeline_manifest.json", result)
    dump(output / "evaluation_scorecard.json", {})
    return result


def mock_runner(calls, *, failure=None, expected_gpu_uuids=None):
    expected_gpu_uuids = [gpu_uuid(i) for i in range(8)] if expected_gpu_uuids is None else expected_gpu_uuids
    def run(argv, logfile, env, **kwargs):
        calls.append((argv, env, kwargs))
        assert kwargs["timeout_seconds"] > 0
        def value(flag):
            return argv[argv.index(flag) + 1]
        if failure and failure in logfile.stem:
            raise RuntimeError("simulated process failure")
        if "--preflight" in argv:
            assert json.loads(env["A2UI_LITERT_ALLOWED_GPU_UUIDS"]) == expected_gpu_uuids
            dump(Path(value("--report")), {"status": "prerequisites_passed", "vulkan_compute_device_verified": True})
        elif "probe" in argv:
            dump(Path(value("--report")), {"status": "passed"})
        elif "prepare" in argv:
            folder = Path(value("--output-dir"))
            dump(folder / "model.safetensors", {"fixture": True})
            dump(folder / "deployment_source.json", {"merged_files": {"model.safetensors": sha256(folder / "model.safetensors")}})
        elif "convert" in argv:
            folder = Path(value("--output-dir"))
            dump(folder / "model.litertlm", {"variant": value("--variant")})
            dump(folder / "export_manifest.json", {})
            dump(folder / "package_inspection.json", {})
        else:
            litert = "--builtin-gpu" in argv
            assert "--require-prepared-contract" in argv
            if not litert:
                assert "--require-gpu" in argv and value("--devices") == "auto"
            else:
                assert json.loads(env["A2UI_LITERT_ALLOWED_GPU_UUIDS"]) == expected_gpu_uuids
            evaluation(Path(value("--output-dir")), int(value("--required-rows")),
                       Path(value("--model" if litert else "--checkpoint")), litert=litert,
                       gpu_uuids=expected_gpu_uuids[:1])
    return run


@pytest.fixture
def export_validator(monkeypatch):
    def validate(plan, variant):
        folder = Path(plan["variants"][variant]["output_dir"])
        return {"actual_precision": {"verified_fixture": True},
                "files": [str(folder / name) for name in ("model.litertlm", "export_manifest.json", "package_inspection.json")]}
    monkeypatch.setattr(deployment, "validate_deployment_export_output", validate)


def test_plan_does_not_execute_and_defaults_all_four_variants(options):
    result = deployment.run_deployment(options)
    assert result["status"] == "plan_only"
    assert list(result["export"]["variants"]) == ["w32", "w16", "w8", "w4"]
    assert result["training"]["options"]["max_new_tokens"] == 2048
    assert not options.base.output_dir.exists()
    assert not result["gpu_policy"]["cpu_inference_fallback"]


@pytest.mark.parametrize("change,match", [({"allow_experimental_formats": False}, "experimental"),
                                         ({"stage_timeout_seconds": 0}, "Timeout"),
                                         ({"runtime_python": Path("python")}, "absolute"),
                                         ({"include_augmentation": True}, "require --tune")])
def test_bad_options_stop_before_output(options, change, match):
    with pytest.raises(ValueError, match=match):
        deployment.run_deployment(replace(options, **change), execute=True)
    assert not options.base.output_dir.exists()


@pytest.mark.parametrize("profile", ["e2b", "270m"])
def test_complete_orchestration_validates_14_results_and_logs_hparams(options, export_validator, profile):
    options = replace(options, base=replace(options.base, profile=profile))
    calls, writer = [], Writer()
    result = deployment.run_deployment(options, execute=True, command_runner=mock_runner(calls),
               pipeline_runner=fake_pipeline, writer_factory=lambda **_: writer, gpu_probe=inventory)
    assert result["status"] == "complete"
    assert len(result["results"]) == 14
    assert len(writer.hparams) == 1
    assert len([call for call in calls if "--builtin-gpu" in call[0]]) == 8
    assert (options.base.output_dir / "deployment_results.md").is_file()
    assert result["results"]["w4_golden35"]["row_count"] == 35
    assert result["completed"]["host_preflight"]["files"]
    assert list(result["completed"])[-1] == "scorecard"


@pytest.mark.parametrize("failure", ["exporter_preflight", "runtime_preflight", "export_w16", "w4_golden35"])
def test_failure_never_publishes_success_scorecard(options, export_validator, failure):
    with pytest.raises(RuntimeError, match="simulated"):
        deployment.run_deployment(options, execute=True, command_runner=mock_runner([], failure=failure),
              pipeline_runner=fake_pipeline, writer_factory=Writer, gpu_probe=inventory)
    output = options.base.output_dir
    assert not (output / "deployment_scorecard.json").exists()
    state = json.loads((output / "deployment_manifest.json").read_text())
    assert state["status"] == "failed"
    assert failure in state["active_stage"]


def test_tuning_locks_settings_then_trains_fresh_full_epoch(options, export_validator):
    observed = []
    def experiment(opts, **kwargs):
        assert not opts.evaluate_selected_holdout and opts.base.steps is None
        directory = opts.base.output_dir
        for name in ("experiments_manifest.json", "selection_locked.json", "comparison.json"):
            dump(directory / name, {})
        parameters = {"learning_rate": 1e-5, "weight_decay": .01, "warmup_ratio": .03, "augmentation": "none"}
        return {"status": "complete", "selected_golden35": None, "selected_trial": "lr_half",
                "plan": {"trials": [{"name": "lr_half", "parameters": parameters}]}}
    def training(base, **kwargs):
        observed.append(base)
        return fake_pipeline(base, **kwargs)
    result = deployment.run_deployment(replace(options, tune=True), execute=True, command_runner=mock_runner([]),
        pipeline_runner=training, experiment_runner=experiment, writer_factory=Writer, gpu_probe=inventory)
    assert result["status"] == "complete"
    assert len(observed) == 1 and observed[0].steps is None and observed[0].epochs == 1
    assert observed[0].learning_rate == 1e-5 and observed[0].evaluate_golden35
    assert "lock_hyperparameters" in result["completed"]


def test_final_checkpoint_step_is_not_best_step(tmp_path):
    path = tmp_path / "final_adapter"
    dump(path / "training_metadata.json", {"checkpoint_role": "final", "checkpoint_step": 120, "best_golden_eval": {"step": 100}})
    assert deployment._checkpoint_step(path) == 120


def test_real_deployment_tensorboard_hparams_and_golden_scalars(options, export_validator):
    pytest.importorskip("tensorboard")
    pytest.importorskip("torch")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.plugins.hparams import metadata
    result = deployment.run_deployment(options, execute=True, command_runner=mock_runner([]),
        pipeline_runner=fake_pipeline, gpu_probe=inventory)
    directory = Path(result["plan"]["tensorboard_dir"])
    events = EventAccumulator(str(directory)).Reload()
    assert events.Scalars("evaluation/w4/golden35/generation_reward_v5_4_avg")[0].value == pytest.approx(.5)
    child = EventAccumulator(str(directory / "final_comparison")).Reload()
    assert metadata.SESSION_START_INFO_TAG in child.PluginTagToContent(metadata.PLUGIN_NAME)
    assert child.Scalars("hparam/w4_golden32/unique_source_generation_reward_v5_4_avg")[0].step == 80


@pytest.mark.parametrize("field,value", [("allocation_observed", False), ("engine_backend", "cpu"), ("gpu_uuids", []), ("tokenizer_parity_passed", False)])
def test_litert_success_requires_real_gpu_evidence(tmp_path, field, value):
    artifact = tmp_path / "model.litertlm"
    dump(artifact, {})
    directory = tmp_path / "eval"
    evaluation(directory, 32, artifact, litert=True)
    path = directory / "external_runner_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["gpu_execution"][field] = value
    dump(path, manifest)
    with pytest.raises(ValueError, match="GPU allocation"):
        deployment._evaluation(directory, 32, artifact=artifact, litert=True)


@pytest.mark.skipif(os.name == "nt", reason="venv symlink regression is POSIX-specific")
def test_plan_preserves_runtime_venv_invocation(options, tmp_path):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    plan = deployment.build_deployment_plan(replace(options, runtime_python=python))
    assert plan["runtime_probe_command"][0] == str(python)
