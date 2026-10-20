"""GRPO-only rollout bounds and exact QAT inference restoration on CPU."""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.train.grpo_runtime import (
    RolloutOptions,
    _tracked_rollout_stopping_criteria,
    make_express_rollout,
    normalize_rollout_completion,
    rollout_batch_report,
    rollout_inference_scope,
    validate_rollout_options,
)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1", None])
def test_invalid_rollout_microbatch_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        validate_rollout_options({"rollout_microbatch_size": value})


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_cache_option_is_explicit_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        validate_rollout_options({"rollout_qat_weight_cache": value})


def test_independent_rollout_batch_does_not_change_effective_batch():
    trainer = SimpleNamespace(
        args=SimpleNamespace(per_device_train_batch_size=1, gradient_accumulation_steps=8),
        accelerator=SimpleNamespace(num_processes=4), num_generations=4,
    )
    options = validate_rollout_options({})
    assert options == RolloutOptions(microbatch_size=1, qat_weight_cache=False)
    report = rollout_batch_report(trainer, 8, options)
    assert report["peak_simultaneous_candidates"] == 1
    assert report["generation_calls"] == 8
    assert report["effective_completion_batch"] == 32
    assert report["num_generations"] == 4
    assert report["candidate_order_preserved"] is True
    assert rollout_batch_report(trainer, 8, None)["generation_calls"] == 1


class CharacterTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 2, 1, 0
    chat_template = "test"

    def __init__(self):
        self.decode_calls = 0

    @staticmethod
    def encode(text):
        return [ord(char) + 10 for char in text]

    def decode(self, ids, **kwargs):
        del kwargs
        self.decode_calls += 1
        return "".join(chr(int(value) - 10) for value in ids if int(value) > 3)

    def __call__(self, texts, **kwargs):
        import torch

        assert kwargs["add_special_tokens"] is False
        rows = [self.encode(text) for text in texts]
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([[0] * (width - len(ids)) + ids for ids in rows]),
            "attention_mask": torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in rows]),
        }


@pytest.mark.parametrize("text,eos", [
    ("unfinished " * 180, False),
    ('<a2ui>root=Text("literal </a2ui>")</a2ui>overshoot', False),
    ('<a2ui>root=Text("escaped \\\" </a2ui>")</a2ui>', False),
    ("unfinished", True),
])
def test_actual_stop_evidence_eliminates_normalization_decodes_without_changing_tokens(text, eos):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    tokenizer = CharacterTokenizer()
    raw = tokenizer.encode(text) + ([3, 0, 0] if eos else [])
    criteria, checked = _tracked_rollout_stopping_criteria(tokenizer, 1, 1)
    for end in range(1, len(raw) + 1):
        criteria(torch.tensor([[2, *raw[:end]]]), scores=None)
    prior = tokenizer.decode_calls
    actual = normalize_rollout_completion(tokenizer, raw, eos_token_ids=[1, 3],
        pad_token_id=0, max_new_tokens=len(raw), _checked_prefix_stops=checked[0])
    assert tokenizer.decode_calls == prior
    expected = normalize_rollout_completion(CharacterTokenizer(), raw, eos_token_ids=[1, 3],
        pad_token_id=0, max_new_tokens=len(raw))
    assert actual == expected


def test_missing_prefix_evidence_falls_back_to_exact_decoder():
    tokenizer = CharacterTokenizer()
    raw = tokenizer.encode("</a2ui>trailing")
    expected = normalize_rollout_completion(CharacterTokenizer(), raw, eos_token_ids=[1],
        pad_token_id=0, max_new_tokens=50)
    actual = normalize_rollout_completion(tokenizer, raw, eos_token_ids=[1],
        pad_token_id=0, max_new_tokens=50, _checked_prefix_stops={1: False, 3: False})
    assert actual == expected
    assert tokenizer.decode_calls == len("</a2ui>") - 2


def test_non_prefix_stable_decoder_cannot_use_final_text_as_stop_proof():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class RewritingTokenizer:
        def decode(self, ids, **kwargs):
            return {1: "<a2ui>", 2: "</a2ui>", 3: "no closing tag"}[len(ids)]

    tokenizer = RewritingTokenizer()
    criteria, checked = _tracked_rollout_stopping_criteria(tokenizer, 1, 1)
    for size in (1, 2, 3):
        criteria(torch.tensor([[2, *([9] * size)]]), scores=None)
    expected = normalize_rollout_completion(tokenizer, [9, 9, 9], eos_token_ids=[1],
        pad_token_id=0, max_new_tokens=3)
    actual = normalize_rollout_completion(tokenizer, [9, 9, 9], eos_token_ids=[1],
        pad_token_id=0, max_new_tokens=3, _checked_prefix_stops=checked[0])
    assert actual == expected
    assert actual["sampled_token_ids"] == [9, 9]


def _qat_policy(dtype, retained):
    torch = pytest.importorskip("torch")
    from ir_training.qat.fake_quant import QATController, QATSpec

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(8, 4, bias=False)
            self.base_layer.weight.requires_grad_(False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})
            self.scaling = {"default": 0.5}
            self.active_adapters = ["default"]

        def get_delta_weight(self, name):
            return self.lora_B[name].weight @ self.lora_A[name].weight * self.scaling[name]

        def forward(self, inputs):
            return self.base_layer(inputs) + self.lora_B["default"](self.lora_A["default"](inputs)) * 0.5

    class Scales:
        def resolve_weight_key(self, candidates):
            return candidates[0]

        def load_scale(self, weight_key, *, weight_shape, **kwargs):
            return torch.full((weight_shape[0], 1), 0.125)

        def activation_scale(self, weight_key, edge):
            return 0.0625 if edge == "input" else 0.125

    torch.manual_seed(17)
    model = torch.nn.Sequential(Projection()).to(dtype=dtype)
    spec = QATSpec(weight_bits=4, activation_bits=8, exclude_modules=(),
        scale_mode="retained_mobile" if retained else "dynamic",
        activation_quantizer="gemma_mobile_srq" if retained else "legacy",
        ste_gradient="clipped", quantizer="ste_ai_edge")
    controller = QATController(spec, Scales() if retained else None).prepare(model)
    model._a2ui_qat_controller = controller
    return model, controller


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_qat_cache_exact_logits_and_restoration_including_failure(dtype_name, retained, fails):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    model, controller = _qat_policy(dtype, retained)
    model[0].lora_A.eval()  # Heterogeneous modes must not all become train().
    modes = [module.training for module in model.modules()]
    inputs = [torch.randn(2, 8, dtype=dtype), torch.randn(1, 8, dtype=dtype)]
    with rollout_inference_scope(model, RolloutOptions(qat_weight_cache=False)):
        reference = [model(value) for value in inputs]
    report = {}
    try:
        with rollout_inference_scope(model, RolloutOptions(qat_weight_cache=True)) as report:
            assert not torch.is_grad_enabled()
            assert all(not module.training for module in model.modules())
            actual = [model(value) for value in inputs]
            assert controller.inference_cache_stats["hits"] == 1
            assert not controller._compile_srq_enabled
            for expected, value in zip(reference, actual, strict=True):
                torch.testing.assert_close(expected, value, rtol=0, atol=0)
            if fails:
                raise RuntimeError("generation failed")
    except RuntimeError as exc:
        assert fails and str(exc) == "generation failed"
    assert [module.training for module in model.modules()] == modes
    assert torch.is_grad_enabled()
    assert report["cached_bytes_after_scope"] == 0
    assert controller.inference_cache_stats["cached_bytes"] == 0
    assert controller._inference_cache_enabled is False
    assert all(parameter.grad is None for parameter in model.parameters())
    model(inputs[0]).float().sum().backward()
    assert model[0].lora_B["default"].weight.grad is not None
    assert model[0].base_layer.weight.grad is None


def test_active_saturation_monitor_is_not_bypassed_by_cache():
    torch = pytest.importorskip("torch")
    model, controller = _qat_policy(torch.float32, True)
    observations = []
    controller.saturation_monitor = SimpleNamespace(active=True,
        observe=lambda *args, **kwargs: observations.append(kwargs["role"]))
    with rollout_inference_scope(model, RolloutOptions(qat_weight_cache=True)) as report:
        model(torch.ones(1, 8))
        model(torch.ones(1, 8))
    assert report["cache_effective"] is False
    assert report["saturation_monitor_active"] is True
    assert observations.count("weight") == 2
    assert report["weight_cache"]["hits"] == 0
    assert report["cached_bytes_after_scope"] == 0


def test_cache_requires_qat_controller_and_rejects_nested_cache():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="prepared QAT controller"), rollout_inference_scope(
        torch.nn.Linear(1, 1), RolloutOptions(qat_weight_cache=True)
    ):
        pytest.fail("unreachable")
    model, controller = _qat_policy(torch.float32, False)
    with controller.inference_cache(), pytest.raises(RuntimeError, match="pre-existing"), rollout_inference_scope(
        model, RolloutOptions()
    ):
        pytest.fail("unreachable")


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 8])
@pytest.mark.parametrize("fails", [False, True])
def test_bounded_rollout_preserves_groups_and_restores_generation_state(monkeypatch, tmp_path, chunk_size, fails):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizer = CharacterTokenizer()
    calls, checkpoints = [], []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = torch.nn.Dropout(0.0).eval()
            self.generation_config = transformers.GenerationConfig(use_cache=False)
            self.config = SimpleNamespace(use_cache=False)
            self.is_gradient_checkpointing = True

        def gradient_checkpointing_enable(self, **kwargs):
            checkpoints.append(kwargs)

        def generate(self, **kwargs):
            assert self.training is self.child.training is False
            assert not torch.is_grad_enabled()
            assert self.config.use_cache is True
            assert kwargs["synced_gpus"] is False
            inputs, mask = kwargs["input_ids"], kwargs["attention_mask"]
            labels = [tokenizer.decode(ids[keep.bool()]) for ids, keep in zip(inputs, mask, strict=True)]
            calls.append(labels)
            if fails:
                raise RuntimeError("mock generation failure")
            tails = [tokenizer.encode(f'<a2ui>root=Text("{label}")</a2ui>') for label in labels]
            width = max(map(len, tails))
            padded = torch.tensor([ids + [0] * (width - len(ids)) for ids in tails])
            for end in range(1, width + 1):
                kwargs["stopping_criteria"](torch.cat((inputs, padded[:, :end]), dim=1), scores=None)
            return torch.cat((inputs, padded), dim=1)

    model = Model()
    saved = model.generation_config
    @contextmanager
    def unwrap(*args, **kwargs):
        try:
            yield model
        finally:
            model.gradient_checkpointing_enable()
    trl_models = ModuleType("trl.models")
    trl_models.unwrap_model_for_generation = unwrap
    monkeypatch.setitem(sys.modules, "trl.models", trl_models)
    trainer = SimpleNamespace(model=model, model_wrapped=model, processing_class=tokenizer,
        use_vllm=False, is_fsdp_enabled=False, is_deepspeed_enabled=False,
        accelerator=SimpleNamespace(device="cpu", process_index=0, num_processes=4),
        args=SimpleNamespace(gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_train_batch_size=1, gradient_accumulation_steps=8),
        generation_kwargs={"do_sample": True}, max_completion_length=64,
        num_generations=4, state=SimpleNamespace(global_step=7))
    prompts = ["a", "a", "bbb", "bbb", "c"]
    rollout = make_express_rollout(str(tmp_path), [1, 3], audit_limit=0,
        rollout_options=RolloutOptions(microbatch_size=chunk_size))
    if fails:
        with pytest.raises(RuntimeError, match="mock generation failure"):
            rollout(prompts, trainer)
    else:
        result = rollout(prompts, trainer)
        assert [item for chunk in calls for item in chunk] == prompts
        assert len(calls) == (len(prompts) + chunk_size - 1) // chunk_size
        assert result["prompt_ids"] == [tokenizer.encode(value) for value in prompts]
        for prompt, ids, mask in zip(prompts, result["completion_ids"], result["env_mask"], strict=True):
            assert tokenizer.decode(ids) == f'<a2ui>root=Text("{prompt}")</a2ui>'
            assert ids[-1] == 0 and mask[-1] == 0
            assert all(value == 1 for value in mask[:-1])
    assert model.training is True and model.child.training is False
    assert model.generation_config is saved and model.config.use_cache is False
    assert checkpoints[-1] == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    report = json.loads((tmp_path / "grpo_rollout_batches.rank0.jsonl").read_text())
    assert report["status"] == ("failed" if fails else "passed")
    assert report["effective_completion_batch"] == 32
    assert report["peak_simultaneous_candidates"] == min(chunk_size, len(prompts))
    assert report["completed_candidates"] == (0 if fails else len(prompts))
