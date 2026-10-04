from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.train.grpo_runtime import GRPOHealthMonitor, HealthThresholds
from ir_training.train.qat_grpo import (
    QATGRPOModel,
    _bind_selected_health,
    _health_snapshot,
    assert_qat_policy_active,
    build_qat_grpo_trainer,
    disable_policy_dropout,
    make_qat_rollout,
    validate_qat_grpo_config,
)


def recipe():
    value = load_yaml(ROOT / "configs/models/gemma3_270m_a2ui_express_qat.yaml")
    value["run"]["purpose"] = "qat_lora_grpo_v1"
    value["training"].update(method="qat_lora_grpo", logging_steps=1)
    value["grpo"] = {"family": "270m", "sft_checkpoint": "source", "sft_training_config": "source.yaml", "beta": 0.0}
    return value


def test_runtime_and_cli_import_without_ml_loading():
    probe = "import sys; sys.path.insert(0,sys.argv[1]); import ir_training.train.qat_grpo; assert not any(n in sys.modules for n in ('torch','transformers','trl','peft'))"
    subprocess.run([sys.executable, "-c", probe, str(ROOT / "src")], check=True)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/train_qat_grpo.py"), "--help"], capture_output=True, text=True, check=True)
    assert "--dependency-preflight-only" in result.stdout


def test_original_q8_configuration_is_accepted_with_explicit_grpo_contract():
    validate_qat_grpo_config(recipe())


@pytest.mark.parametrize(("section", "key", "value", "match"), [
    ("grpo", "beta", 0.1, "beta=0"),
    ("grpo", "num_generations", 1, "num_generations"),
    ("grpo", "use_vllm", True, "HF generation"),
    ("grpo", "top_p", 0.9, "unfiltered sampling"),
    ("grpo", "num_iterations", 2, "fresh generation"),
    ("grpo", "temperature", 0.0, "finite and positive"),
    ("training", "distributed_backend", "sharded", "single-device/DDP"),
    ("training", "logging_steps", 20, "logging_steps=1"),
    ("training", "ddp_broadcast_buffers", True, "ddp_broadcast_buffers"),
    ("training", "gradient_checkpointing_kwargs", {"use_reentrant": True}, "non-reentrant"),
    ("lora", "dropout", 0.1, "zero LoRA dropout"),
])
def test_unsupported_policy_paths_fail_before_model_loading(section, key, value, match):
    config = recipe()
    config[section][key] = value
    with pytest.raises(ValueError, match=match):
        validate_qat_grpo_config(config)


def toy_policy():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(3, 3, bias=False)
            self.base_layer.weight.requires_grad_(False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False)})
            self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(0.0)})
            self.scaling = {"default": 1.0}
            self.active_adapters = ["default"]
            self.disable_adapters = False
            self.merged = False
            self.use_dora = {"default": False}

        def get_delta_weight(self, adapter):
            return self.lora_B[adapter].weight @ self.lora_A[adapter].weight

        def forward(self, x):
            return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = Projection()
            self.dropout = nn.Dropout(0.4)
            self.attention_dropout = 0.2
            self.generation_config = SimpleNamespace(cache_implementation="hybrid")

        def forward(self, x):
            return self.q_proj(self.dropout(x))

    from ir_training.qat.fake_quant import prepare_qat_model
    model = Model()
    config = {"qat": {"enabled": True, "effective_merged_weight": True, "quantizer": "ste_ai_edge",
        "weight_bits": 8, "activation_bits": 32, "exclude_modules": []}}
    disable_policy_dropout(model)
    controller = prepare_qat_model(model, config)
    return torch, model, controller


def test_qat_shared_effective_weight_policy_stays_active_through_gradient_update():
    torch, model, controller = toy_policy()
    base = model.q_proj.base_layer.weight.detach().clone()
    old_adapter = model.q_proj.lora_B["default"].weight.detach().clone()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
    x = torch.tensor([[0.2, -0.5, 0.7]])
    with torch.no_grad():
        model.eval()
        rollout = model(x)
    model.train()
    assert torch.equal(rollout, model(x).detach())
    assert_qat_policy_active(model, controller)
    model(x).square().sum().backward()
    assert all(p.grad is None for p in model.q_proj.base_layer.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters() if p.requires_grad)
    optimizer.step()
    assert torch.equal(base, model.q_proj.base_layer.weight)
    assert not torch.equal(old_adapter, model.q_proj.lora_B["default"].weight)
    assert_qat_policy_active(model, controller)
    assert not any("qat" in key for key in model.state_dict())
    controller.restore()
    with pytest.raises(RuntimeError, match="wrappers disappeared"):
        assert_qat_policy_active(model, controller)


def test_policy_guard_rejects_trainable_base_and_restored_forward():
    _, model, controller = toy_policy()
    model.q_proj.base_layer.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="only LoRA"):
        assert_qat_policy_active(model, controller)
    model.q_proj.base_layer.weight.requires_grad_(False)
    model.q_proj.forward = controller._original_forwards[model.q_proj]
    with pytest.raises(RuntimeError, match="forward was restored"):
        assert_qat_policy_active(model, controller)


def test_serialization_preserves_live_qat_wrappers_and_restores_cache_on_failure():
    _, model, controller = toy_policy()
    bundle = QATGRPOModel({}, model, None, None, {}, [2], controller)

    class Trainer:
        def save_model(self, output_dir=None, _internal_call=False):
            assert self.model.generation_config.cache_implementation is None
            assert_qat_policy_active(self.model, controller)
            raise OSError("simulated save failure")

    instance = build_qat_grpo_trainer(Trainer, bundle)()
    instance.model = model
    with pytest.raises(OSError, match="simulated"):
        instance.save_model("unused")
    assert model.generation_config.cache_implementation == "hybrid"
    assert_qat_policy_active(model, controller)


def test_likelihood_and_rollout_enter_the_same_amp_context(monkeypatch):
    _, model, controller = toy_policy()
    bundle = QATGRPOModel({}, model, None, None, {}, [2], controller)
    entered = []
    accelerator = SimpleNamespace(autocast=lambda: (entered.append("autocast") or nullcontext()))
    monkeypatch.setattr("ir_training.train.grpo_runtime.make_express_rollout",
        lambda *a, **k: lambda prompts, trainer: {"completion_ids": [[3]], "env_mask": [[1]]})
    trainer = SimpleNamespace(model=model, accelerator=accelerator)
    assert make_qat_rollout(bundle, "unused", audit_limit=1)(["test"], trainer)["completion_ids"] == [[3]]

    class Trainer:
        def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
            assert_qat_policy_active(model, controller)
            return "likelihood"

    instance = build_qat_grpo_trainer(Trainer, bundle)()
    instance.accelerator = accelerator
    assert instance._get_per_token_logps_and_entropies(model) == "likelihood"
    assert entered == ["autocast", "autocast"]


def test_health_snapshot_retains_raw_window_and_recomputable_aggregates():
    thresholds = HealthThresholds(window_steps=2)
    monitor = GRPOHealthMonitor(thresholds)
    row = {"loss": 0.1, "frac_reward_zero_std": 0.0, "completions/clipped_ratio": 0.0,
        "grad_norm": 1.0, "sampled_parameter_max_delta": 0.2, "nonfinite_parameters_or_gradients": 0.0}
    monitor.observe(1, row)
    result = monitor.observe(2, row)
    snapshot = _health_snapshot(SimpleNamespace(monitor=monitor, passed_window=True), 2, thresholds)
    assert snapshot["aggregates"] == result["aggregates"]
    assert snapshot["thresholds"] == asdict(thresholds)
    assert snapshot["window_metrics"] == [row, row]
    assert snapshot["last_step"] == snapshot["optimizer_steps"] == 2
    json.dumps(snapshot, allow_nan=False)


def test_best_checkpoint_cannot_inherit_later_optimizer_health(tmp_path):
    checkpoint = tmp_path / "best"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
    (checkpoint / "training_metadata.json").write_text(json.dumps({
        "checkpoint_step": 20, "grpo": {"health": {"optimizer_steps": 40, "passed_window": True}}}), encoding="utf-8")
    earlier = {"optimizer_steps": 20, "passed_window": True}
    _bind_selected_health({"checkpoint_dir": str(checkpoint), "step": 20}, {20: earlier})
    metadata = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
    assert metadata["grpo"]["health"] == earlier
    assert metadata["checkpoint_step"] == 20
    assert metadata["checkpoint_role"] == "best_golden"
    _bind_selected_health({"checkpoint_dir": str(checkpoint), "step": 20}, {})
    metadata = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
    assert metadata["grpo"]["health"] == {"passed_window": False}


def test_mocked_runner_writes_export_verifiable_selected_checkpoint(tmp_path, monkeypatch):
    """Exercise orchestration, actual QAT controller and provenance, not CUDA/TRL.

    The model load, forward/greedy/backward measurements and TRL optimizer loop
    are explicit fixtures. Real source lineage, numeric evidence comparison,
    health callback, checkpoint callback and export verifier remain in play.
    """
    import copy
    from types import ModuleType

    import yaml
    from ir_training import generation_policy
    from ir_training.train import grpo_runtime, prepared_binding, sft
    from ir_training.train import qat_grpo as runtime
    from ir_training.train.qat_grpo_contract import (
        file_identity,
        verify_qat_grpo_provenance,
        verify_sft_adapter_lineage,
    )
    from test_qat_grpo_export import _fixture
    monkeypatch.syspath_prepend(str(ROOT.parent / "dataset/src"))

    config, prior, _, config_path = _fixture(tmp_path)
    torch, model, initial_controller = toy_policy()
    initial_controller.restore()
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    for name in ("train", "val"):
        (dataset / f"{name}.jsonl").write_text('{}\n', encoding="utf-8")
    output, selected = tmp_path / "run", tmp_path / "run/best"
    config["run"].update(output_dir=str(output), dataset_dir=str(dataset), id="fixture")
    config["model"].update(dtype="float32", max_context_tokens=128)
    config["qat"]["final_runtime_validation_required"] = True
    config["training"].update(max_steps=3, epochs=1, max_seq_length=64, learning_rate=0.01,
        per_device_train_batch_size=1, gradient_accumulation_steps=2, per_device_eval_batch_size=2,
        logging_steps=1, eval_steps=2, save_steps=2, mixed_precision="off")
    config["grpo"].update(num_generations=2, max_prompt_length=16, max_completion_length=16)
    config["golden_eval"].update(enabled=True, best_checkpoint_dir=str(selected))
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    initial_lineage = verify_sft_adapter_lineage(config)
    assert initial_lineage["verified"], initial_lineage
    metadata = {"training_metadata_version": 4, "checkpoint_kind": "lora_adapter", "git_commit": "a" * 40,
        "training": config["training"], "model": config["model"], "lora": config["lora"],
        "config_path": str(config_path), "training_config_sha256": file_identity(config_path)["sha256"],
        "initialization_seed": 42, "source_adapter": prior["numeric_preflight"]["source_adapter"],
        "grpo": {"schema_version": 1, "algorithm": "grpo", "family": "270m", "source_lineage": initial_lineage}}

    def save_model(path, **kwargs):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapter_config.json").write_text('{}', encoding="utf-8")
        (destination / "adapter_model.safetensors").write_bytes(b"mock optimizer adapter")

    model.save_pretrained = save_model
    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    bundle = QATGRPOModel(config, model, tokenizer, None, metadata, [2])
    monkeypatch.setattr(runtime, "load_qat_grpo_model", lambda *a, **k: bundle)
    monkeypatch.setattr(runtime, "verify_prepared_training_contract", lambda *a: {"verified": True})
    monkeypatch.setattr(prepared_binding, "verify_tokenizer_binding", lambda *a, **k: {"verified": True})
    rows = [{"prompt": "prompt", "completion": "completion", "prompt_provenance": {"prompt_tokens": 6}}]
    monkeypatch.setattr(runtime, "_script_module", lambda name: SimpleNamespace(load_training_dataset=lambda *a, **k: rows))
    monkeypatch.setattr(sft, "_model_position_limit", lambda *a: 128)
    monkeypatch.setattr(sft, "_require_model_input_vocab_size", lambda *a: 32)
    monkeypatch.setattr(sft, "_require_model_label_vocab_size", lambda *a: 32)
    monkeypatch.setattr(sft, "_tokenize_completion_only_row", lambda **k: {"input_ids": [1, 2], "labels": [-100, 2]})
    numeric = prior["numeric_preflight"]
    monkeypatch.setattr(sft, "_run_forward_numeric_gate", lambda **k: copy.deepcopy(numeric["baseline"]))
    monkeypatch.setattr(sft, "_run_deterministic_greedy_gate", lambda **k: copy.deepcopy(numeric["greedy_generation"]["qat_on"]))
    monkeypatch.setattr(runtime, "_backward_probe", lambda *a, **k: copy.deepcopy(numeric["backward"]))
    monkeypatch.setattr(generation_policy, "preserve_generation_eos", lambda *a, **k: [2])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(grpo_runtime, "dependency_report", lambda: {"packages": {"trl": {"imported_version": "0.29.1"}}})
    monkeypatch.setattr(grpo_runtime, "validate_runtime_features", lambda *a: None)
    monkeypatch.setattr(runtime, "make_qat_rollout", lambda *a, **k: lambda *a: {})

    class Callback:
        def on_pre_optimizer_step(self, *a, **k):
            pass

    transformers = ModuleType("transformers")
    transformers.TrainerCallback = Callback
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class Golden(Callback):
        def on_evaluate(self, args, state, control, **kwargs):
            save_model(selected)

        def summary(self):
            return {"checkpoint_dir": str(selected), "step": 2, "epoch": 0.5,
                    "metric": "generation_reward_v5_4_avg", "metric_value": 75.0}

    monkeypatch.setattr(sft, "_build_optional_golden_callback", lambda **k: Golden())

    class Config:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    class Trainer:
        def __init__(self, *, model, args, **kwargs):
            assert "peft_config" not in kwargs  # Existing adapter must not be wrapped a second time.
            self.model, self.args, self.callbacks = model, args, []
            self.state = SimpleNamespace(global_step=0, epoch=0.0, is_world_process_zero=True, log_history=[])
            self.accelerator = SimpleNamespace(num_processes=1, process_index=0, device="cpu", gather=lambda x: x)
            self.generation_config = SimpleNamespace()
            self.generation_kwargs = {}

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def fire(self, event, **kwargs):
            for callback in self.callbacks:
                method = getattr(callback, event, None)
                if method:
                    method(self.args, self.state, None, model=self.model, **kwargs)

        def log(self, logs):
            self.state.log_history.append(logs)
            self.fire("on_log", logs=logs)

        def train(self):
            self.fire("on_train_begin")
            for step in (1, 2, 3):
                for parameter in self.model.parameters():
                    if parameter.requires_grad:
                        parameter.grad = torch.ones_like(parameter)
                self.fire("on_pre_optimizer_step")
                with torch.no_grad():
                    for parameter in self.model.parameters():
                        if parameter.requires_grad:
                            parameter.add_(0.01)
                self.state.global_step, self.state.epoch = step, step / 3
                self.fire("on_step_end")
                self.log({"reward": 0.5, "frac_reward_zero_std": 0.0, "completions/clipped_ratio": 0.0, "loss": 0.2})
                if step == 2:
                    self.fire("on_evaluate")
                    self.save_model(str(output / "checkpoint-2"))
                    self.fire("on_save")
            self.fire("on_train_end")

        def save_model(self, output_dir=None, _internal_call=False):
            self.model.save_pretrained(output_dir)

        def is_world_process_zero(self):
            return True

    trl = ModuleType("trl")
    trl.__version__, trl.GRPOConfig, trl.GRPOTrainer = "0.29.1", Config, Trainer
    monkeypatch.setitem(sys.modules, "trl", trl)
    result = runtime.train_qat_grpo(config, config_path)
    assert result["checkpoint_step"] == 3
    saved = json.loads((selected / "training_metadata.json").read_text(encoding="utf-8"))
    assert saved["checkpoint_step"] == saved["grpo"]["health"]["last_step"] == 2
    report = verify_qat_grpo_provenance(config, saved)
    assert report["verified"], report
    assert json.loads((output / "checkpoint-2/training_metadata.json").read_text())["grpo"]["health"]["optimizer_steps"] == 2
    assert not hasattr(model, "_a2ui_qat_controller")
