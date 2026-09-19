"""Small numerical oracles; no model downloads, CUDA, training or conversion."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")
from ir_training.qat.fake_quant import (
    QATSpec,
    fake_quantize_activation,
    mobile_srq_ste,
    prepare_qat_model,
)
from ir_training.qat.mobile_qparams import MobileQParams, MobileQParamsError
from torch import nn


def reference_srq(x, scale):
    # Pinned Google/HF gemma_quant.py apply_srq forward contract (see metadata).
    scale = scale.to(x.dtype)
    calibrated = scale != 0
    safe = torch.where(calibrated, scale, torch.ones_like(scale))
    rounded = torch.clamp(torch.round(x / safe), -128, 127) * safe
    return torch.where(calibrated, rounded, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1.0, 0.037, 0.0, 1e-12])
def test_mobile_srq_exact_published_forward(dtype, scale):
    x = torch.cat((torch.linspace(-5, 5, 2001), torch.tensor([-128., -127., 127., 128., -.5, .5]))).to(dtype)
    s = torch.tensor(scale, dtype=torch.float32)
    assert torch.equal(mobile_srq_ste(x, s), reference_srq(x, s))


def test_mobile_srq_full_a8_range_and_clipped_gradient():
    x = torch.tensor([-129., -128., -127., 0., 127., 128.], requires_grad=True)
    y = mobile_srq_ste(x, 1.)
    assert y.tolist() == [-128., -128., -127., 0., 127., 127.]
    y.sum().backward()
    assert x.grad.tolist() == [0., 1., 1., 1., 1., 0.]


def test_zero_scale_bypasses_and_never_trains_scale():
    x = torch.tensor([-999., .25, 999.], requires_grad=True)
    scale = torch.tensor(0., requires_grad=True)
    y = mobile_srq_ste(x, scale)
    assert torch.equal(x, y)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert scale.grad is None


@pytest.mark.parametrize("scale", [None, -1., float("nan"), float("inf"), [1., 2.]])
def test_mobile_srq_rejects_bad_scales(scale):
    with pytest.raises(ValueError):
        mobile_srq_ste(torch.ones(2), scale)


def test_legacy_ai_edge_activation_behavior_is_unchanged():
    spec = QATSpec(quantizer="ste_ai_edge")
    assert fake_quantize_activation(torch.tensor([-128., 127.]), spec, scale_override=torch.tensor(1.)).tolist() == [-127., 127.]
    new = QATSpec(quantizer="ste_ai_edge", scale_mode="retained_mobile", activation_quantizer="gemma_mobile_srq")
    assert fake_quantize_activation(torch.tensor([-128., 127.]), new, scale_override=torch.tensor(1.)).tolist() == [-128., 127.]


def test_qparams_preserves_present_zero_and_distinguishes_missing():
    provider = object.__new__(MobileQParams)
    provider.inventory = {"head.weight": {"input_activation_scale_f32_le_hex": struct.pack("<f", 0.).hex()}}
    assert provider.activation_scale("head.weight", "input") == 0.
    assert provider.activation_scale("head.weight", "output") is None
    provider.inventory["head.weight"]["input_activation_scale_f32_le_hex"] = struct.pack("<f", -1.).hex()
    with pytest.raises(MobileQParamsError):
        provider.activation_scale("head.weight", "input")


class ToyLora(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(0.)})
        self.active_adapters = ["default"]
        self.disable_adapters = self.merged = False
        self.use_dora = {"default": False}
        nn.init.zeros_(self.lora_B["default"].weight)

    def get_delta_weight(self, adapter):
        return self.lora_B[adapter].weight @ self.lora_A[adapter].weight

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))


QKEY = "model.layers.0.self_attn.q_proj.weight"
FROZEN_KEYS = {f"model.layers.0.{name}.weight" for name in ("per_layer_input_gate", "per_layer_projection")}


class ToyQParams:
    def resolve_weight_key(self, candidates):
        return next((key for key in candidates if key in FROZEN_KEYS | {QKEY}), None)

    def load_scale(self, key, *, weight_shape, bits, group_size):
        assert bits == (4 if key == QKEY else 8)
        return torch.full((weight_shape[0], 1), 0.25)

    def activation_scale(self, key, role):
        return 1.0 if role == "input" else .5

    def trainable_projection_weight_keys(self):
        return (QKEY,)

    def frozen_activation_weight_keys(self):
        return tuple(sorted(FROZEN_KEYS))

    def summary(self):
        return {"verified": True}


def toy_model():
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = ToyLora()
    for name in ("per_layer_input_gate", "per_layer_projection"):
        linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.eye(2))
        linear.requires_grad_(False)
        setattr(layer, name, linear)
    model.model.layers = nn.ModuleList([layer])
    return model


def toy_config():
    return {"qat": {"enabled": True, "scale_mode": "retained_mobile", "mobile_qparams_contract": "fixture",
        "quantizer": "ste_ai_edge", "ste_gradient": "clipped", "activation_quantizer": "gemma_mobile_srq",
        "effective_merged_weight": True, "effective_lora_only": True,
        "fixed_scale_required": True, "fixed_activation_scale_required": True,
        "expected_effective_lora_modules": 1, "simulate_frozen_activations": True,
        "expected_frozen_activation_modules": 2, "require_lora_trainable_scope": True,
        "exclude_modules": [], "module_quant_configs": {r"self_attn\.q_proj$": {"num_bits": 4}}}}


def bind(monkeypatch, model, config=None):
    monkeypatch.setattr("ir_training.qat.mobile_qparams.MobileQParams", lambda _path: ToyQParams())
    return prepare_qat_model(model, config or toy_config())


def test_all_frozen_activation_edges_are_wrapped_without_weight_changes(monkeypatch):
    model = toy_model()
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    controller = bind(monkeypatch, model)
    layer = model.model.layers[0]
    x = torch.tensor([[-129., 128.]], requires_grad=True)
    y = layer.per_layer_input_gate(x)
    assert y.tolist() == [[-64., 63.5]]
    y.sum().backward()
    assert layer.per_layer_input_gate.weight.grad is None
    assert all(torch.equal(before[key], value) for key, value in model.named_parameters())
    summary = controller.summary()
    assert summary["wrapped_effective_lora_count"] == 1
    assert summary["frozen_activation_module_count"] == 2
    assert summary["trainable_scope"]["all_adapters_trainable"]
    assert summary["native_kv_cache_simulated"] is False
    controller.restore()
    assert torch.equal(layer.per_layer_input_gate(x), x)


def test_missing_frozen_module_fails_and_restores_wrappers(monkeypatch):
    model = toy_model()
    del model.model.layers[0].per_layer_projection
    original = model.model.layers[0].per_layer_input_gate.forward
    with pytest.raises(ValueError, match="Frozen A8 scope mismatch"):
        bind(monkeypatch, model)
    assert model.model.layers[0].per_layer_input_gate.forward == original


@pytest.mark.parametrize("unexpected", ["embedding", "partial_adapter", "frozen_layer"])
def test_trainable_scope_rejects_unexpected_or_partial_weights(monkeypatch, unexpected):
    model = toy_model()
    if unexpected == "embedding":
        model.extra_embedding = nn.Embedding(4, 2)
    elif unexpected == "partial_adapter":
        model.model.layers[0].self_attn.q_proj.lora_A["default"].requires_grad_(False)
    else:
        model.model.layers[0].per_layer_input_gate.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        bind(monkeypatch, model)


def test_frozen_checkpoint_evaluation_is_supported(monkeypatch):
    model = toy_model().requires_grad_(False)
    controller = bind(monkeypatch, model)
    assert controller.summary()["trainable_scope"]["all_parameters_frozen"]


def test_saturation_hooks_observe_mutable_and_frozen_edges(monkeypatch):
    class Recorder:
        def __init__(self):
            self.calls = []
        def observe(self, name, values, scale, **kwargs):
            self.calls.append((name, kwargs["role"], kwargs["qmin"], kwargs["qmax"]))
    model = toy_model()
    controller = bind(monkeypatch, model)
    monitor = Recorder()
    controller.saturation_monitor = monitor
    layer = model.model.layers[0]
    layer.self_attn.q_proj(torch.ones(1, 2))
    layer.per_layer_projection(torch.ones(1, 2))
    assert [item[1] for item in monitor.calls] == ["input", "weight", "output", "input", "output"]
    assert all(item[2:] == (-128, 127) for item in monitor.calls if item[1] != "weight")


@pytest.mark.parametrize("rank,attached", [("0", True), ("1", False), ("7", False)])
def test_only_rank_zero_owns_saturation_reports(monkeypatch, rank, attached):
    monkeypatch.setenv("RANK", rank)
    config = toy_config()
    config["qat"]["saturation"] = {"enabled": True}
    controller = bind(monkeypatch, toy_model(), config)
    assert (controller.saturation_monitor is not None) is attached
