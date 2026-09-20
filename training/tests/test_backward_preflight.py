"""CPU backward probes exercise shapes/state without training or GPU claims."""
from __future__ import annotations

import random
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")
import ir_training.train.backward_preflight as backward_preflight_module
from ir_training.train.backward_preflight import run_backward_preflight
from ir_training.train.tensor_checks import tensor_finite_and_nonzero


class TinyTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def pad(self, features, padding=True, return_tensors="pt"):
        width = max(len(row["input_ids"]) for row in features)
        return {"input_ids": torch.tensor([row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in features]),
                "attention_mask": torch.tensor([row["attention_mask"] + [0] * (width - len(row["input_ids"])) for row in features])}


class BackwardFailure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError("synthetic cuDNN backward workspace failure")


class TinyLoRA(torch.nn.Module):
    def __init__(self, *, failure=False, nan_gradient=False, zero_gradient=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.embedding.weight.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.randn(4, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 16))
        self.register_buffer("constant", torch.ones(1))
        self.dropout = torch.nn.Dropout(0.1)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = object()
        self.calls = []
        self.checkpoint_options = []
        self.failure, self.zero_gradient = failure, zero_gradient
        if nan_gradient:
            self.lora_B.register_hook(lambda grad: torch.full_like(grad, float("nan")))

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = dict(gradient_checkpointing_kwargs)
        self.checkpoint_options.append(dict(gradient_checkpointing_kwargs))
        self._require_grads_hook = self.embedding.register_forward_hook(lambda module, inputs, output: output.requires_grad_(True))

    def forward(self, input_ids, attention_mask=None):
        random.random()
        np.random.random()
        self.calls.append({"shape": tuple(input_ids.shape), "mask": attention_mask is not None,
                           "training": self.training, "checkpointing": self.gradient_checkpointing,
                           "resident_gradients": self.lora_B.grad is not None})
        values = self.dropout(self.embedding(input_ids))

        def layers(hidden):
            logits = hidden @ self.lora_A @ self.lora_B
            return logits * 0 if self.zero_gradient else logits

        if self.gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            logits = checkpoint(layers, values, **self._gradient_checkpointing_func)
        else:
            logits = layers(values)
        if self.failure:
            logits = BackwardFailure.apply(logits)
        return SimpleNamespace(logits=logits)


def rows(lengths=(3, 9, 5)):
    return [{"input_ids": [1] * (size - 1) + [2], "attention_mask": [1] * size,
             "labels": [-100] * (size - 2) + [1, 2]} for size in lengths]


def probe(model, dataset=None, config=None, **kwargs):
    return run_backward_preflight(model, TinyTokenizer(), rows() if dataset is None else dataset,
                                  {"per_device_train_batch_size": 2, **(config or {})},
                                  16, 16, 16, 16, force_cpu=True, **kwargs)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_default_cpu_skip_never_scans_or_runs_model():
    model = TinyLoRA()
    result = run_backward_preflight(model, None, None, {}, 16, 16, 16, 16)
    assert result["status"] == "skipped"
    assert model.calls == []


def test_explicit_disable_never_imports_or_touches_model():
    result = run_backward_preflight(None, None, None, {"backward_preflight": False}, 16, 16, 16, 16)
    assert result["status"] == "disabled"
    assert result["optimizer_steps"] == 0


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None])
def test_flag_requires_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        run_backward_preflight(None, None, None, {"backward_preflight": value}, 16, 16, 16, 16)


def test_worst_and_padded_shapes_use_real_rows_and_restore_all_state(monkeypatch):
    model = TinyLoRA()
    model.eval()
    model.dropout.train()  # Mixed module modes must survive the gate.
    original_hook = model.embedding.register_forward_hook(lambda module, inputs, output: None)
    original_hooks = dict(model.embedding._forward_hooks)
    original_checkpoint = model._gradient_checkpointing_func
    dataset = rows()
    original_rows = deepcopy(dataset)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    monkeypatch.setattr(torch.optim.AdamW, "__init__", lambda *args, **kwargs: pytest.fail("Optimizer construction is forbidden"))
    result = probe(model, dataset, {"gradient_checkpointing": True,
                                    "gradient_checkpointing_kwargs": {"use_reentrant": False, "preserve_rng_state": True}})
    assert result["status"] == "passed" and result["optimizer_steps"] == 0
    assert result["distributed_collectives_tested"] is False
    assert result["selection"]["longest_index"] == 1
    assert [item["row_indices"] for item in result["batches"]] == [[1, 1], [1, 0]]
    assert [call["shape"] for call in model.calls] == [(2, 9), (2, 9)]
    assert [call["mask"] for call in model.calls] == [False, True]
    assert all(call["training"] and call["checkpointing"] for call in model.calls)
    assert all(item["gradient_tensors"] == 2 and item["nonzero_gradient_tensors"] == 1 for item in result["batches"])
    assert all(item["memory"] == {} for item in result["batches"])
    assert model.checkpoint_options == [{"use_reentrant": False, "preserve_rng_state": True}]
    assert model.gradient_checkpointing is False and model._gradient_checkpointing_func is original_checkpoint
    assert model.training is False and model.dropout.training is True
    assert dict(model.embedding._forward_hooks) == original_hooks
    assert not hasattr(model, "_require_grads_hook")
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in weights.items())
    assert dataset == original_rows
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    original_hook.remove()


@pytest.mark.parametrize("lengths,microbatch,expected", [((5, 5), 2, 1), ((3, 9, 5), 1, 1), ((8,), 3, 1)])
def test_equal_lengths_or_single_microbatch_do_not_invent_padding(lengths, microbatch, expected):
    result = probe(TinyLoRA(), rows(lengths), {"per_device_train_batch_size": microbatch})
    assert len(result["batches"]) == expected
    assert result["batches"][0]["input_shape"] == [microbatch, max(lengths)]


@pytest.mark.parametrize("accumulation", [2, 4, 8, 16])
def test_bounded_accumulation_checks_resident_gradients_without_updates(accumulation):
    model = TinyLoRA()
    result = probe(model, config={"gradient_accumulation_steps": accumulation})
    assert result["gradient_accumulation_steps"] == accumulation
    assert result["backward_passes"] == 4
    assert [item["accumulation_microstep"] for item in result["batches"]] == [1, 2, 1, 2]
    assert [call["resident_gradients"] for call in model.calls] == [False, True, False, True]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_forward_success_backward_failure_fails_before_training_and_restores(monkeypatch):
    model = TinyLoRA(failure=True)
    model.eval()
    state = torch.get_rng_state()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setenv("RANK", "3")
    with pytest.raises(RuntimeError, match=r"rank=3.*phase=backward.*workspace failure"):
        probe(model, config={"gradient_checkpointing": True})
    assert len(model.calls) == 1
    assert not model.training and not model.gradient_checkpointing
    assert not model.embedding._forward_hooks
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in weights.items())


def test_refuses_preexisting_gradients_without_clearing_them():
    model = TinyLoRA()
    prior = torch.ones_like(model.lora_A)
    model.lora_A.grad = prior
    with pytest.raises(ValueError, match="pre-existing gradients"):
        probe(model)
    assert model.lora_A.grad is prior
    assert model.calls == []


@pytest.mark.parametrize("option,message", [("nan_gradient", "non-finite gradient"), ("zero_gradient", "no nonzero trainable gradients")])
def test_invalid_gradients_fail_closed(option, message):
    model = TinyLoRA(**{option: True})
    with pytest.raises(RuntimeError, match=message):
        probe(model)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cpu_preflight_uses_chunked_tensor_check_and_preserves_model(monkeypatch):
    model = TinyLoRA()
    weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    calls = []

    def checked_in_small_chunks(tensor):
        calls.append((tensor.device.type, tensor.numel()))
        return tensor_finite_and_nonzero(tensor, chunk_elements=2)

    monkeypatch.setattr(
        backward_preflight_module, "tensor_finite_and_nonzero", checked_in_small_chunks
    )
    result = probe(model)

    assert result["status"] == "passed"
    assert calls and all(device == "cpu" for device, _ in calls)
    assert {numel for _, numel in calls} == {8, 32}
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(model.state_dict()[name], value) for name, value in weights.items()
    )


def test_cpu_preflight_chunked_check_rejects_nan_in_final_chunk(monkeypatch):
    model = TinyLoRA()

    def add_late_nan(gradient):
        gradient = gradient.clone()
        gradient.reshape(-1)[-1] = float("nan")
        return gradient

    model.lora_B.register_hook(add_late_nan)
    calls = []

    def checked_in_small_chunks(tensor):
        calls.append(tensor.numel())
        return tensor_finite_and_nonzero(tensor, chunk_elements=3)

    monkeypatch.setattr(
        backward_preflight_module, "tensor_finite_and_nonzero", checked_in_small_chunks
    )
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        probe(model)

    assert calls
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("dataset", [[], rows((1,)), rows((17,)), [{"labels": [1, 2]}]])
def test_invalid_token_lengths_fail_without_truncation(dataset):
    model = TinyLoRA()
    with pytest.raises(ValueError, match="non-empty|invalid sequence length"):
        probe(model, dataset)
    assert model.calls == []


@pytest.mark.parametrize("microbatch", [0, -1, 1.5, True])
def test_microbatch_requires_positive_integer(microbatch):
    with pytest.raises(ValueError, match="positive integer"):
        probe(TinyLoRA(), config={"per_device_train_batch_size": microbatch})


def test_checkpointing_disabled_temporarily_and_prior_configuration_restored():
    model = TinyLoRA()
    model.gradient_checkpointing_enable({"use_reentrant": False})
    checkpoint = model._gradient_checkpointing_func
    hook = model._require_grads_hook
    hooks = dict(model.embedding._forward_hooks)
    probe(model, config={"gradient_checkpointing": False})
    assert all(not call["checkpointing"] for call in model.calls)
    assert model.gradient_checkpointing and model._gradient_checkpointing_func is checkpoint
    assert model._require_grads_hook is hook and model.embedding._forward_hooks == hooks


def test_rejects_forward_that_mutates_model_state_in_place():
    model = TinyLoRA()
    model.register_forward_pre_hook(lambda module, inputs: module.constant.add_(1) and None)
    with pytest.raises(RuntimeError, match="mutated parameters/buffers"):
        probe(model)


def test_restores_replaced_buffer_references():
    model = TinyLoRA()
    original = model.constant

    def replace(module, inputs):
        module.constant = torch.zeros_like(module.constant)

    handle = model.register_forward_pre_hook(replace)
    probe(model)
    assert model.constant is original
    handle.remove()


def test_real_gemma4_peft_checkpoint_backward_preserves_existing_input_hook():
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    peft = pytest.importorskip("peft", minversion="0.19.0")
    config = transformers.Gemma4TextConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        max_position_embeddings=32, vocab_size_per_layer_input=16, hidden_size_per_layer_input=4,
        layer_types=["full_attention", "full_attention"], num_kv_shared_layers=0,
        pad_token_id=0, bos_token_id=3, eos_token_id=2,
    )
    model = peft.get_peft_model(transformers.Gemma4ForCausalLM(config), peft.LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=2, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]))
    model.config.use_cache = False
    model.enable_input_require_grads()  # Match SFT's pre-existing hook.
    model.eval()
    hooks = dict(model.get_input_embeddings()._forward_hooks)
    weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = torch.get_rng_state()
    result = probe(model, config={"gradient_checkpointing": True, "gradient_accumulation_steps": 4,
                                  "gradient_checkpointing_kwargs": {"use_reentrant": False}})
    assert result["status"] == "passed" and result["backward_passes"] == 4
    assert all(batch["nonzero_gradient_tensors"] > 0 for batch in result["batches"])
    assert not model.training and not model.is_gradient_checkpointing
    assert dict(model.get_input_embeddings()._forward_hooks) == hooks
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(dict(model.named_parameters())[name], weight) for name, weight in weights.items())
    assert torch.equal(torch.get_rng_state(), state)
