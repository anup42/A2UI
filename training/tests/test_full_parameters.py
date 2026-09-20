from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

import ir_training.train.full_parameters as full_parameters_module
from ir_training.train.full_parameters import (
    FullParameterScopeError,
    enable_full_parameter_training,
    probe_full_optimizer_step,
    validate_full_training_runtime,
)
from ir_training.train.tensor_checks import tensor_all_finite


def test_runtime_guard_rejects_old_trainer_before_loading_a_model():
    with pytest.raises(FullParameterScopeError, match="_build_accelerator_args"):
        validate_full_training_runtime(SimpleNamespace())


def test_runtime_guard_requires_bucket_view_api_without_claiming_gpu_execution():
    trainer = SimpleNamespace(_build_accelerator_args=dict)
    with pytest.raises(FullParameterScopeError, match="gradient_as_bucket_view"):
        validate_full_training_runtime(trainer, lambda: SimpleNamespace())
    result = validate_full_training_runtime(
        trainer, lambda: SimpleNamespace(gradient_as_bucket_view=False)
    )
    assert result["verified"] is True
    assert result["gpu_execution_tested"] is False


class TinyTiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 3, dtype=torch.bfloat16)
        self.projection = torch.nn.Linear(3, 7, bias=False, dtype=torch.bfloat16)
        self.projection.weight = self.embedding.weight
        self.norm = torch.nn.LayerNorm(3, dtype=torch.float16)
        self.norm.weight.requires_grad_(False)

    def forward(self, tokens):
        return self.projection(self.norm(self.embedding(tokens)))


def test_enables_every_unique_parameter_in_fp32_and_preserves_tied_aliases():
    model = TinyTiedModel()
    tied_id = id(model.embedding.weight)

    report = enable_full_parameter_training(model)

    named = dict(model.named_parameters(remove_duplicate=False))
    assert id(named["embedding.weight"]) == tied_id
    assert id(named["projection.weight"]) == tied_id
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert report["verified"] is True
    assert report["unique_parameter_count"] == 3
    assert report["named_parameter_count"] == 4
    assert report["tied_alias_count"] == 1
    assert report["frozen_parameter_count"] == 0
    assert json.loads(json.dumps(report)) == report


def test_small_lr_optimizer_step_updates_true_full_parameters():
    torch.manual_seed(7)
    model = TinyTiedModel()
    enable_full_parameter_training(model)
    before = {id(p): p.detach().clone() for p in model.parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    loss = model(torch.tensor([[1, 2, 3]])).square().mean()
    loss.backward()
    optimizer.step()

    assert all(parameter.grad is not None for parameter in model.parameters())
    assert any(not torch.equal(before[id(parameter)], parameter) for parameter in model.parameters())


class AccidentalLoRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.lora_A = torch.nn.Parameter(torch.ones(1, 2))


@pytest.mark.parametrize("marker", ["name", "peft_config"])
def test_rejects_adapter_backed_models(marker):
    model = AccidentalLoRA() if marker == "name" else torch.nn.Linear(2, 2)
    if marker == "peft_config":
        model.peft_config = {"default": object()}
    with pytest.raises(FullParameterScopeError, match="LoRA/adapter"):
        enable_full_parameter_training(model)


def test_rejects_quantized_wrapper_before_mutating_parameters():
    model = torch.nn.Linear(2, 2)
    model.is_loaded_in_4bit = True
    before = model.weight.detach().clone()
    with pytest.raises(FullParameterScopeError, match="quantized"):
        enable_full_parameter_training(model)
    assert torch.equal(model.weight, before)


def test_rejects_nonfloating_parameter_before_partial_mutation():
    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.good = torch.nn.Parameter(torch.ones(2, dtype=torch.float16), requires_grad=False)
            self.bad = torch.nn.Parameter(torch.ones(2, dtype=torch.int64), requires_grad=False)

    model = Mixed()
    with pytest.raises(FullParameterScopeError, match="requires floating parameters"):
        enable_full_parameter_training(model)
    assert model.good.dtype == torch.float16
    assert model.good.requires_grad is False


def test_rejects_quantization_config_marker():
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(quantization_config={"load_in_4bit": True})
    with pytest.raises(FullParameterScopeError, match="quantized"):
        enable_full_parameter_training(model)


def test_disposable_adafactor_probe_materializes_state_and_updates_model():
    torch.manual_seed(11)
    model = torch.nn.Linear(3, 2)
    enable_full_parameter_training(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    def backward():
        model(torch.ones(2, 3)).square().mean().backward()

    report = probe_full_optimizer_step(
        model, backward=backward, learning_rate=1e-4, force_cpu=True
    )

    assert report["passed"] is True
    assert report["disposable_optimizer_steps"] == 1
    assert report["checkpoint_writes"] == 0
    assert report["model_must_not_be_reused"] is True
    assert report["optimizer"]["state_tensor_count"] > 0
    assert report["optimizer"]["state_numel"] > 0
    assert report["optimizer"]["external_max_grad_norm_required"] == 0.0
    assert report["memory"]["cuda_local_rank_only"] is False
    assert report["memory"]["ddp_collectives_certified"] is False
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.named_parameters()
    )
    assert all(value.grad is None for value in model.parameters())
    assert json.loads(json.dumps(report)) == report


def test_optimizer_probe_rejects_incomplete_full_parameter_scope():
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    with pytest.raises(FullParameterScopeError, match="every unique parameter"):
        probe_full_optimizer_step(
            model, backward=lambda: None, learning_rate=1e-4, force_cpu=True
        )


@pytest.mark.parametrize("failure", ["missing", "nonfinite"])
def test_optimizer_probe_rejects_invalid_gradients(failure):
    model = torch.nn.Linear(2, 1)
    enable_full_parameter_training(model)

    def backward():
        model(torch.ones(1, 2)).sum().backward()
        if failure == "missing":
            model.bias.grad = None
        else:
            model.weight.grad.fill_(float("nan"))

    with pytest.raises(FullParameterScopeError, match="gradients failed"):
        probe_full_optimizer_step(
            model, backward=backward, learning_rate=1e-4, force_cpu=True
        )
    assert all(value.grad is None for value in model.parameters())


def test_cpu_optimizer_probe_checks_gradients_and_post_step_parameters(monkeypatch):
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    enable_full_parameter_training(model)
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    before = {id(parameter): parameter.detach().clone() for parameter in model.parameters()}
    checked_ids = []

    def checked_in_small_chunks(tensor):
        checked_ids.append(id(tensor))
        return tensor_all_finite(tensor, chunk_elements=2)

    monkeypatch.setattr(full_parameters_module, "tensor_all_finite", checked_in_small_chunks)

    def backward():
        model(torch.ones(2, 3)).square().mean().backward()

    report = probe_full_optimizer_step(
        model, backward=backward, learning_rate=1e-4, force_cpu=True
    )

    assert report["passed"] is True
    assert parameter_ids.issubset(checked_ids)
    assert any(checked_id not in parameter_ids for checked_id in checked_ids)
    assert any(
        not torch.equal(before[id(parameter)], parameter)
        for parameter in model.parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cpu_optimizer_probe_rejects_late_post_step_nan_via_chunked_check(monkeypatch):
    from transformers.optimization import Adafactor

    model = torch.nn.Linear(3, 2)
    enable_full_parameter_training(model)
    original_step = Adafactor.step

    def step_with_late_nan(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        optimizer.param_groups[0]["params"][0].data.reshape(-1)[-1] = float("nan")
        return result

    monkeypatch.setattr(Adafactor, "step", step_with_late_nan)
    monkeypatch.setattr(
        full_parameters_module,
        "tensor_all_finite",
        lambda tensor: tensor_all_finite(tensor, chunk_elements=2),
    )

    def backward():
        model(torch.ones(1, 3)).sum().backward()

    with pytest.raises(FullParameterScopeError, match="non-finite parameters"):
        probe_full_optimizer_step(
            model, backward=backward, learning_rate=1e-4, force_cpu=True
        )

    assert torch.isnan(model.weight.reshape(-1)[-1])
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("value", [0.0, 1.0, float("nan")])
def test_optimizer_probe_rejects_invalid_memory_fraction(value):
    model = torch.nn.Linear(2, 1)
    enable_full_parameter_training(model)
    with pytest.raises(FullParameterScopeError, match="max_reserved_fraction"):
        probe_full_optimizer_step(
            model,
            backward=lambda: None,
            learning_rate=1e-4,
            max_reserved_fraction=value,
            force_cpu=True,
        )
