"""CPU fixtures for sequential screening; no tokenizer/model/GPU is loaded."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline import experiments as module
from ir_training.pipeline.golden_training import GoldenTrainingOptions, sha256


@pytest.fixture
def options(tmp_path):
    model, source = tmp_path / "model", tmp_path / "source"
    model.mkdir()
    source.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"not real model weights")
    for name in ("train.jsonl", "val.jsonl"):
        (source / name).write_text("{}\n", encoding="utf-8")
    return module.ExperimentOptions(
        base=GoldenTrainingOptions(model_dir=model, output_dir=tmp_path / "suite", input_dir=source,
                                   tensorboard_root=str(tmp_path / "tensorboard"), prepare_workers=1),
        trial_steps=20,
    )


class Writer:
    def __init__(self, **kwargs):
        self.log_dir = kwargs["log_dir"]
        self.hparams = []
        self.scalars = []
        self.closed = False

    def add_text(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        self.scalars.append(args)

    def add_hparams(self, *args, **kwargs):
        self.hparams.append((args, kwargs))

    def flush(self):
        pass

    def close(self):
        self.closed = True


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")


def _fake_pipeline(calls, *, scores=None, corrupt=None, batch=None):
    def run(options, *, execute):
        assert execute is True and options.evaluate_golden35 is False
        assert options.steps == 20
        index = len(calls)
        calls.append(options)
        output = options.output_dir
        output.mkdir(parents=True)
        score = scores[index] if scores else index / 10
        metrics = {module.SELECTION_METRIC: score, "generation_reward_v5_4_avg": 0.01}
        results = {name: {"row_count": 32, "aggregate": metrics} for name in ("best_golden32", "final_golden32")}
        scorecard = output / "evaluation_scorecard.json"
        _json(scorecard, {"evaluations": results})
        config = output / "fit/training_config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(yaml.safe_dump({"runtime": {"cuda_visible_devices": "0,1", "gpu_profile": {"world_size": 2}},
                          "training": {"expected_effective_batch_size": batch[index] if batch else 32,
                                       "learning_rate": options.learning_rate, "weight_decay": options.weight_decay,
                                       "warmup_ratio": options.warmup_ratio, "seed": options.seed, "max_steps": options.steps}}), encoding="utf-8")
        checkpoint = output / "fit/training/best_golden_checkpoint/model.safetensors"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"mock checkpoint")
        names = ("prepare", "configure", "preflight", "training", "best_golden32", "final_golden32", "scorecard")
        state = {"status": "complete", "completed": {name: {"files": {}} for name in names}}
        state["completed"]["configure"]["files"] = {str(config): sha256(config)}
        state["completed"]["training"]["files"] = {str(checkpoint): sha256(checkpoint)}
        state["completed"]["scorecard"]["files"] = {str(scorecard): sha256(scorecard)}
        _json(output / "pipeline_manifest.json", state)
        if corrupt:
            corrupt(output, state, options)
        return state
    return run


def _holdout_runner(options, calls, *, count=35):
    def run(command, log_file, environment):
        assert (options.base.output_dir / "selection_locked.json").is_file()
        locked = json.loads((options.base.output_dir / "selection_locked.json").read_text())
        assert locked["golden35_seen"] is False
        assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert "--require-prepared-contract" in command
        assert command[command.index("--required-rows") + 1] == "35"
        assert command[command.index("--checkpoint") + 1] == locked["checkpoint"]
        calls.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        _json(output / "evaluation_result.json", {"row_count": count, "aggregate": {"generation_reward_v5_4_avg": 0.123}})
        _json(output / "aggregate_metrics.json", {})
        for name in ("predictions.jsonl", "scored_predictions.jsonl"):
            (output / name).write_text("{}\n" * count, encoding="utf-8")
    return run


def test_plan_is_side_effect_free_bounded_and_holdout_is_not_a_trial(options, monkeypatch):
    monkeypatch.setattr(module, "_summary_writer_factory", lambda: pytest.fail("no TensorBoard import in plan"))
    result = module.run_experiments(options)
    assert result["status"] == "plan_only"
    assert len(result["trials"]) == 4
    assert result["total_optimizer_step_budget"] == 80
    assert result["max_concurrent_training_runs"] == 1
    assert [item["name"] for item in result["trials"]] == ["baseline", "lr_half", "lr_double", "regularization"]
    assert all(not item["plan"]["options"]["evaluate_golden35"] for item in result["trials"])
    assert all(not any("golden35" in stage for stage in item["plan"]["stages"]) for item in result["trials"])
    assert not options.base.output_dir.exists()
    assert not Path(options.base.tensorboard_root).exists()
    shared_cache = options.base.output_dir.parent / ".golden-preparation-cache"
    assert {item["plan"]["options"]["preparation_cache_dir"] for item in result["trials"]} == {str(shared_cache)}
    assert {item["plan"]["options"]["token_cache_dir"] for item in result["trials"]} == {str(shared_cache / "tokens")}
    assert not shared_cache.exists()


def test_explicit_cache_options_survive_trials_and_full_training_handoff(options, tmp_path):
    base = replace(options.base, token_cache=False, preparation_cache=False,
                   preparation_cache_dir=tmp_path / "prepared-store", token_cache_dir=tmp_path / "token-store")
    plan = module.build_experiment_plan(replace(options, base=base))
    for trial in plan["trials"]:
        roundtrip = module._options_from_plan(trial["plan"])
        assert roundtrip.token_cache is False and roundtrip.preparation_cache is False
        assert roundtrip.token_cache_dir == base.token_cache_dir
        assert roundtrip.preparation_cache_dir == base.preparation_cache_dir
    handoff = module._full_training_handoff(plan, {"index": 0})
    command = handoff["plan_only_command_argv"]
    assert "--no-token-cache" in command and "--no-preparation-cache" in command
    assert command[command.index("--token-cache-dir") + 1] == str(base.token_cache_dir)


@pytest.mark.parametrize("field", ["preparation_cache_dir", "token_cache_dir"])
def test_experiment_cache_cannot_be_inside_suite_output(options, field):
    with pytest.raises(ValueError, match="Experiment caches"):
        module.build_experiment_plan(replace(options, base=replace(options.base, **{field: options.base.output_dir / "cache"})))


def test_experiment_binds_training_implementation_even_when_preparation_cache_ignores_it(options):
    plan = module.build_experiment_plan(options)
    bindings = module._input_bindings(plan, interval=10)
    training_source = Path(__file__).resolve().parents[1] / "src/ir_training/train/sft.py"
    assert str(training_source) in bindings


@pytest.mark.parametrize("profile,qat,rate", [("e2b", False, 2e-5), ("270m", False, 2e-5), ("270m", True, 5e-6)])
def test_profile_specific_defaults_and_optional_resampling(options, profile, qat, rate):
    result = module.build_experiment_plan(replace(options, base=replace(options.base, profile=profile, qat=qat), include_augmentation=True))
    assert len(result["trials"]) == 5
    assert result["trials"][0]["parameters"]["learning_rate"] == rate
    assert result["trials"][-1]["parameters"]["augmentation"] == "rare_components"
    assert result["trials"][0]["parameters"]["augmentation"] == "none"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_explicit_optimizer_budget_is_required(options, value):
    with pytest.raises(ValueError, match="trial-steps"):
        module.build_experiment_plan(replace(options, trial_steps=value))


@pytest.mark.parametrize("spec", [
    [{"name": "escape", "output_dir": "bad"}], [{"name": "bad", "learning_rate": float("nan")}],
    [{"name": "bad", "learning_rate": 0.1}], [{"name": "bad", "warmup_ratio": 1}],
    [{"name": "bad", "augmentation": "rewrite_targets"}], [{"name": "../bad", "learning_rate": 1e-5}],
    [{"name": "baseline", "learning_rate": 1e-5}], [{"name": "duplicate"}],
])
def test_custom_trials_fail_closed(options, tmp_path, spec):
    path = tmp_path / "trials.json"
    _json(path, spec)
    with pytest.raises(ValueError):
        module.build_experiment_plan(replace(options, trials_file=path))


def test_custom_trials_preserve_global_budget_and_baseline(options, tmp_path):
    path = tmp_path / "trials.json"
    _json(path, [{"name": "candidate", "learning_rate": 3e-5}])
    result = module.build_experiment_plan(replace(options, trials_file=path))
    assert len(result["trials"]) == 2
    assert result["trials"][0]["name"] == "baseline"
    assert result["trials"][1]["plan"]["options"]["steps"] == 20


def test_runs_one_by_one_logs_hparams_then_locks_before_single_holdout(options):
    calls, holdouts, writers = [], [], []
    def factory(**kwargs):
        writer = Writer(**kwargs)
        writers.append(writer)
        return writer
    state = module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline(calls),
                                   command_runner=_holdout_runner(options, holdouts), writer_factory=factory)
    assert state["status"] == "complete"
    assert len(calls) == 4 and len(holdouts) == 1
    assert state["selected_trial"] == "regularization"
    assert len(writers[0].hparams) == 4 and writers[0].closed
    assert all(item.steps == 20 and item.seed == 42 for item in calls)
    assert len({item.output_dir for item in calls}) == 4
    assert state["golden35_used_for_selection"] is False
    comparison = json.loads((options.base.output_dir / "comparison.json").read_text())
    assert comparison["selected_trial"] == "regularization"
    handoff = json.loads((options.base.output_dir / "selected_full_training_options.json").read_text())
    assert handoff["options"]["steps"] is None
    assert handoff["options"]["evaluate_golden35"] is True
    assert "--execute" not in handoff["plan_only_command_argv"]


def test_deployment_screening_defers_holdout_until_fresh_full_training(options):
    calls = []
    state = module.run_experiments(replace(options, evaluate_selected_holdout=False), execute=True,
        pipeline_runner=_fake_pipeline(calls), writer_factory=Writer,
        command_runner=lambda *_: pytest.fail("screening must not call Golden35"))
    assert state["status"] == "complete" and state["selected_golden35"] is None
    assert len(calls) == 4 and all(not item.evaluate_golden35 for item in calls)
    assert not (options.base.output_dir / "selected_golden35").exists()
    assert (options.base.output_dir / "selection_locked.json").is_file()


def test_ties_prefer_baseline_not_holdout_score(options):
    calls, holdouts = [], []
    state = module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline(calls, scores=[.4] * 4),
                                   command_runner=_holdout_runner(options, holdouts), writer_factory=Writer)
    assert state["selected_trial"] == "baseline"
    assert len(holdouts) == 1


def test_failed_trial_stops_durably_and_is_not_auto_resumed(options):
    calls = []
    def fail(trial, **kwargs):
        calls.append(trial)
        raise RuntimeError("fixture training failure")
    with pytest.raises(RuntimeError, match="fixture training failure"):
        module.run_experiments(options, execute=True, pipeline_runner=fail, writer_factory=Writer)
    state = json.loads((options.base.output_dir / "experiments_manifest.json").read_text())
    assert state["status"] == "failed" and state["active_trial"] == "baseline"
    assert len(calls) == 1 and not (options.base.output_dir / "selection_locked.json").exists()
    with pytest.raises(FileExistsError):
        module.run_experiments(options, execute=True, pipeline_runner=fail, writer_factory=Writer)
    assert len(calls) == 1


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), True])
def test_missing_or_nonfinite_development_metric_refuses_selection(options, score):
    calls = []
    with pytest.raises(ValueError, match="required development metric"):
        module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline(calls, scores=[score]), writer_factory=Writer,
                               command_runner=lambda *args: pytest.fail("holdout must not run"))
    assert len(calls) == 1


def test_effective_batch_drift_stops_unfair_comparison(options):
    calls = []
    with pytest.raises(ValueError, match="effective batch changed"):
        module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline(calls, batch=[32, 16]), writer_factory=Writer,
                               command_runner=lambda *args: pytest.fail("holdout must not run"))
    assert len(calls) == 2


def test_trial_evidence_tamper_and_source_tamper_fail_before_ranking(options):
    calls = []
    def corrupt(output, state, trial):
        (output / "fit/training/best_golden_checkpoint/model.safetensors").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="artifact changed"):
        module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline(calls, corrupt=corrupt), writer_factory=Writer)
    assert len(calls) == 1


def test_missing_tensorboard_stops_before_any_training(options):
    def fail(**kwargs):
        raise RuntimeError("fixture TensorBoard unavailable")
    with pytest.raises(RuntimeError, match="TensorBoard unavailable"):
        module.run_experiments(options, execute=True, pipeline_runner=lambda *args: pytest.fail("must not start training"), writer_factory=fail)
    assert json.loads((options.base.output_dir / "experiments_manifest.json").read_text())["status"] == "failed"


def test_partial_selected_holdout_retains_lock_but_fails(options):
    holdouts = []
    with pytest.raises(ValueError, match="all 35"):
        module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline([]), writer_factory=Writer,
                               command_runner=_holdout_runner(options, holdouts, count=34))
    assert len(holdouts) == 1
    assert (options.base.output_dir / "selection_locked.json").is_file()
    assert json.loads((options.base.output_dir / "experiments_manifest.json").read_text())["status"] == "failed"


def test_source_change_stops_before_second_trial(options):
    calls = []
    def corrupt(output, state, trial):
        (trial.input_dir / "train.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline(calls, corrupt=corrupt), writer_factory=Writer)
    assert len(calls) == 1


def test_unbound_model_file_is_rejected(options):
    calls = []
    def corrupt(output, state, trial):
        (trial.model_dir / "new-weights.safetensors").write_bytes(b"new unbound weights")
    with pytest.raises(ValueError, match="inventory changed"):
        module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline(calls, corrupt=corrupt), writer_factory=Writer)
    assert len(calls) == 1


def test_realized_step_budget_must_match_declared_trial(options):
    def corrupt(output, state, trial):
        path = output / "fit/training_config.yaml"
        config = yaml.safe_load(path.read_text())
        config["training"]["max_steps"] = 21
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer-step budget"):
        module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline([], corrupt=corrupt), writer_factory=Writer)


def test_model_or_selection_tamper_during_holdout_fails(options):
    holdouts = []
    original = _holdout_runner(options, holdouts)
    def change(command, log_file, environment):
        original(command, log_file, environment)
        locked = options.base.output_dir / "selection_locked.json"
        locked.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline([]), writer_factory=Writer, command_runner=change)
    assert len(holdouts) == 1


def test_execution_context_detects_changed_package_identity(options, monkeypatch):
    plan = module.build_experiment_plan(options)
    bindings = module._input_bindings(plan, 10)
    expected = module._execution_context(plan, bindings)
    monkeypatch.setattr(module, "version", lambda name: "changed-package-version")
    with pytest.raises(ValueError, match="package versions changed"):
        module._verify_execution_context(plan, bindings, expected)


def test_logical_snapshot_symlinks_preserve_inventory_membership(options, tmp_path):
    target = tmp_path / "blob"
    target.write_bytes(b"snapshot weights fixture")
    link = options.base.model_dir / "linked.safetensors"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this Windows host")
    plan = module.build_experiment_plan(options)
    bindings = module._input_bindings(plan, 10)
    assert str(link.absolute()) in bindings
    module._verify_directory_inventory(options.base.model_dir, bindings)


def test_holdout_requires_finite_headline_metric(options):
    original = _holdout_runner(options, [])
    def empty(command, log_file, environment):
        original(command, log_file, environment)
        output = Path(command[command.index("--output-dir") + 1])
        _json(output / "evaluation_result.json", {"row_count": 35, "aggregate": {}})
    with pytest.raises(ValueError, match="generation_reward_v5_4_avg"):
        module.run_experiments(options, execute=True, pipeline_runner=_fake_pipeline([]), writer_factory=Writer, command_runner=empty)
