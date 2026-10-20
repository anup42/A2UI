from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from ir_training.train import grpo_probe as probe
from ir_training.train.grpo_coverage import dataset_coverage, with_coverage


def report(path, rank=0):
    return {"passed": True, "config_sha256": probe.file_hash(path), "implementation": probe.probe_implementation(),
            "world_size": 2, "rank": rank, "optimizer_steps": 2,
            "max_reserved_fraction": .8, "minimum_free_fraction": .2}


@pytest.mark.parametrize("changes", [{"rank": 1}, {"world_size": 1}, {"optimizer_steps": 1},
    {"passed": False}, {"implementation": {}}, {"config_sha256": "0"*64},
    {"max_reserved_fraction": .91}, {"minimum_free_fraction": .09},
    {"max_reserved_fraction": float("nan")}, {"minimum_free_fraction": None}])
def test_probe_is_fail_closed(tmp_path, changes):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    with pytest.raises(ValueError):
        probe.validate_probe_report({**report(config), **changes}, config, 2, 0)


def test_probe_requires_every_rank_and_config_identity(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    for rank in range(2):
        (tmp_path / f"grpo_runtime_probe.rank{rank}.json").write_text(json.dumps(report(config, rank)))
    assert len(probe.verify_probe_reports(tmp_path, config, 2)) == 2
    config.write_text("changed: true")
    with pytest.raises(ValueError, match="stale"):
        probe.verify_probe_reports(tmp_path, config, 2)


def test_probe_selects_longest_complete_generation_batch_without_truncating():
    rows = [{"prompt_provenance": {"prompt_tokens": size}} for size in (10, 5120, 4096, 5120)]
    before = copy.deepcopy(rows)
    assert probe.longest_prompt_indices(rows, 3) == [1, 3, 2]
    assert before == rows
    with pytest.raises(ValueError):
        probe.longest_prompt_indices(rows, 5)


def test_runtime_arguments_apply_controls_not_profile_suggestions():
    keys = {"optim", "lr_scheduler_type", "tf32", "dataloader_num_workers", "dataloader_pin_memory",
            "dataloader_persistent_workers", "dataloader_prefetch_factor"}
    controls = {"optim": "adamw_torch", "lr_scheduler_type": "cosine", "tf32": False,
                "dataloader_num_workers": 2, "dataloader_pin_memory": True,
                "dataloader_persistent_workers": True, "dataloader_prefetch_factor": 2}
    assert probe.runtime_arguments(controls, keys, 200) == controls
    for bad in ({"tf32": True}, {"warmup_steps": 200}, {"dataloader_num_workers": -1},
                {"dataloader_num_workers": 0}, {"warmup_steps": float("nan")}):
        with pytest.raises(ValueError):
            probe.runtime_arguments({**controls, **bad}, keys, 200)
    with pytest.raises(ValueError, match="cannot apply"):
        probe.runtime_arguments(controls, set(), 200)


def test_coverage_does_not_change_reward_or_mix_train_and_eval(tmp_path):
    rows = [{"source_id": "same", "intent_bucket": "booking", "prompt_provenance": {"prompt_tokens": 4000}}] * 2
    assert dataset_coverage(rows)["unique_sources"] == 1
    fn = with_coverage(lambda **kwargs: [.25, -.2], tmp_path, 0)
    for mode in ("train", "eval", "train"):
        assert fn(completions=["a", "b"], source_id=["same", "same"],
                  grpo_rollout_mode=[mode, mode]) == [.25, -.2]
    result = json.loads((tmp_path / "grpo_source_exposure.rank0.json").read_text())
    assert result["modes"]["train"]["candidates"] == 4
    assert result["modes"]["eval"]["candidates"] == 2
    assert result["modes"]["train"]["unique_sources"] == 1


def test_explicit_grpo_scorer_config_only_changes_opted_in_eval(tmp_path, monkeypatch):
    from ir_training.eval import android_repair
    from ir_training.eval import compare_to_baseline as evaluation
    monkeypatch.setattr(android_repair, "resolve_runtime", lambda path: None)
    seen = []
    def score(*args, **kwargs):
        seen.append(kwargs["reward_config"])
        return {}
    monkeypatch.setattr(evaluation, "score_prediction", score)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"response_text": "source", "generated_text": "same"}) + "\n")
    weights = tmp_path / "weights.yaml"
    weights.write_text("evaluation: {weights: {}}\n")
    evaluation.evaluate_predictions(predictions, tmp_path / "canonical", weights_config_path=weights)
    assert seen and all(item is None for item in seen)
    seen.clear()
    weights.write_text("grpo_reward_config:\n  policies: {grpo_entity_extraction: true}\n")
    evaluation.evaluate_predictions(predictions, tmp_path / "grpo", weights_config_path=weights)
    assert all(item == {"policies": {"grpo_entity_extraction": True}} for item in seen)


def test_legacy_weights_tolerance_and_explicit_bad_optin(tmp_path):
    from ir_training.eval.metrics import load_grpo_evaluation_config
    path = tmp_path / "weights.yaml"
    assert load_grpo_evaluation_config(path) is None
    path.write_text("bad: [")
    assert load_grpo_evaluation_config(path) is None
    path.write_text("grpo_reward_config: false")
    with pytest.raises(ValueError, match="explicit scorer"):
        load_grpo_evaluation_config(path)


def test_real_probe_loop_disposable_and_reports_headroom_failure(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20, 100))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(
        name="synthetic", total_memory=100, major=9, minor=0))
    for name, value in (("memory_allocated", 10), ("memory_reserved", 15),
                        ("max_memory_allocated", 85), ("max_memory_reserved", 91)):
        monkeypatch.setattr(torch.cuda, name, lambda device, v=value: v)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    class Trainer:
        accelerator = SimpleNamespace(device="cpu", process_index=0, num_processes=1)
        state = SimpleNamespace(global_step=0)
        optimizer = object()
        def add_callback(self, cb):
            self.callback = cb
        def train(self):
            self.state.global_step = 2
            self.callback.on_step_end(None, self.state, None, model=torch.nn.Linear(1, 1))
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    with pytest.raises(ValueError):
        probe.run_disposable_probe(Trainer(), output=tmp_path, config_path=config, selected_rows=[])
    result = json.loads((tmp_path / "grpo_runtime_probe.rank0.json").read_text())
    assert result["optimizer_steps"] == 2 and result["passed"] is False
    assert result["checkpoints_written"] is False and result["peak_reserved_bytes"] == 91
    probe.verify_local_probe_device(tmp_path / "grpo_runtime_probe.rank0.json", torch, "cpu")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(
        name="smaller", total_memory=80, major=9, minor=0))
    with pytest.raises(ValueError, match="hardware"):
        probe.verify_local_probe_device(tmp_path / "grpo_runtime_probe.rank0.json", torch, "cpu")


def test_nonfinite_final_optimizer_parameter_cannot_pass_probe(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (80, 100))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(
        name="synthetic", total_memory=100, major=9, minor=0))
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda device: 20)
    class Trainer:
        accelerator = SimpleNamespace(device="cpu", process_index=0, num_processes=1)
        state = SimpleNamespace(global_step=0)
        optimizer = object()
        def add_callback(self, cb):
            self.callback = cb
        def train(self):
            self.state.global_step = 2
            model = torch.nn.Linear(1, 1)
            model.weight.data.fill_(float("nan"))
            self.callback.on_step_end(None, self.state, None, model=model)
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    with pytest.raises(ValueError, match="Nonfinite updated"):
        probe.run_disposable_probe(Trainer(), output=tmp_path, config_path=config, selected_rows=[])
    assert not json.loads((tmp_path / "grpo_runtime_probe.rank0.json").read_text())["passed"]


def test_rerun_cli_preserves_original_settings_and_rejects_ignored_overrides(tmp_path, monkeypatch):
    from ir_training.pipeline.qat_grpo import QATGRPOOptions
    from ir_training.train import grpo_continuation
    from ir_training.train.qat_grpo import _script_module
    module = _script_module("run_qat_grpo_pipeline")
    options = QATGRPOOptions(family="e2b", model_dir=tmp_path / "base", sft_config=tmp_path / "sft.yaml",
        sft_checkpoint=tmp_path / "original-sft", output_dir=tmp_path / "new", official_litertlm=tmp_path / "official",
        exporter_python=tmp_path / "python", max_steps=123, learning_rate=5e-7)
    monkeypatch.setattr(grpo_continuation, "options_from_previous_run", lambda *args: options)
    captured = []
    monkeypatch.setattr(module, "run_pipeline", lambda opt, **kw: captured.append((opt, kw)) or {"status": "fixture"})
    args = ["--rerun-from", str(tmp_path / "old"), "--output-dir", str(tmp_path / "new")]
    assert module.main([*args, "--defer-native-eval", "--execute"]) == 0
    actual, kwargs = captured[0]
    assert actual.sft_checkpoint == options.sft_checkpoint
    assert actual.max_steps == 123 and actual.learning_rate == 5e-7
    assert actual.defer_native_eval and kwargs == {"execute": True}
    with pytest.raises(SystemExit):
        module.main([*args, "--max-steps", "200"])
    with pytest.raises(SystemExit):
        module.main(["--continue-run", str(tmp_path / "old"), "--max-steps", "200"])
