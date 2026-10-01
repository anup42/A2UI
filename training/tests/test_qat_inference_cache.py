from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")
from torch import nn
from ir_training.qat.fake_quant import QATController, QATSpec, prepare_qat_model, qat_inference_cache_scope


class LoraLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(8, 4, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(8, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
        self.scaling = {"default": 0.5}
        self.active_adapters = ["default"]
        self.delta_calls = 0

    def get_delta_weight(self, name):
        self.delta_calls += 1
        return self.lora_B[name].weight @ self.lora_A[name].weight * self.scaling[name]

    def forward(self, inputs):
        return self.base_layer(inputs) + self.lora_B["default"](self.lora_A["default"](inputs)) * self.scaling["default"]


class RetainedScales:
    def resolve_weight_key(self, candidates):
        return candidates[0]

    def load_scale(self, weight_key, *, weight_shape, **kwargs):
        return torch.full((weight_shape[0], 1), 0.125)

    def activation_scale(self, weight_key, edge):
        return 0.0625 if edge == "input" else 0.125


def lora_fixture(dtype, retained):
    torch.manual_seed(17)
    model = nn.Sequential(LoraLinear()).to(dtype=dtype).eval()
    spec = QATSpec(weight_bits=4, activation_bits=8, exclude_modules=(),
        scale_mode="retained_mobile" if retained else "dynamic",
        activation_quantizer="gemma_mobile_srq" if retained else "legacy",
        ste_gradient="clipped", quantizer="ste_ai_edge")
    controller = QATController(spec, RetainedScales() if retained else None).prepare(model)
    return model, controller


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lora_cache_is_bit_exact_and_reuses_only_weights(dtype, retained):
    model, controller = lora_fixture(dtype, retained)
    inputs = [torch.randn(3, 8, dtype=dtype), torch.randn(1, 8, dtype=dtype)]
    with torch.inference_mode():
        expected = [model(value) for value in inputs]
        baseline_calls = model[0].delta_calls
        with controller.inference_cache():
            actual = [model(value) for value in inputs]
            assert model[0].delta_calls == baseline_calls + 1
            assert controller.inference_cache_stats["hits"] == 1
            assert controller.inference_cache_stats["misses"] == 1
            assert controller.inference_cache_stats["cached_bytes"] > 0
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert controller.inference_cache_stats["cached_bytes"] == 0


@pytest.mark.parametrize("mutation", ["base", "adapter", "scaling", "replace"])
def test_cache_invalidates_after_parameter_and_lora_mutation(mutation):
    model, controller = lora_fixture(torch.float32, False)
    inputs = torch.randn(2, 8)
    with torch.no_grad(), controller.inference_cache():
        model(inputs)
        if mutation == "base":
            model[0].base_layer.weight.add_(0.5)
        elif mutation == "adapter":
            model[0].lora_B["default"].weight.mul_(2)
        elif mutation == "scaling":
            model[0].scaling["default"] = 2
        else:
            model[0].lora_A["default"].weight = nn.Parameter(torch.randn(2, 8))
        actual = model(inputs)
        assert controller.inference_cache_stats["misses"] == 2
    with torch.no_grad():
        expected = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cache_never_reuses_weights_for_gradients_or_training():
    model, controller = lora_fixture(torch.float32, False)
    inputs = torch.randn(2, 8)
    with controller.inference_cache():
        with torch.no_grad():
            model(inputs)
            model(inputs)
        assert controller.inference_cache_stats["hits"] == 1
        model(inputs).sum().backward()
        assert model[0].lora_A["default"].weight.grad is not None
        assert controller.inference_cache_stats["cached_bytes"] == 0
        model.train()
        with torch.no_grad():
            model(inputs)
            model(inputs)
        assert controller.inference_cache_stats["hits"] == 1


def test_cache_scope_releases_weights_on_failure_and_restores_original_forward():
    model = nn.Sequential(nn.Linear(8, 4)).eval()
    original = model[0].forward
    controller = prepare_qat_model(model, {"qat": {"exclude_modules": []}})
    with pytest.raises(RuntimeError, match="failed"):
        with torch.inference_mode(), qat_inference_cache_scope(model):
            model(torch.ones(1, 8))
            assert controller.inference_cache_stats["cached_bytes"] > 0
            raise RuntimeError("failed")
    assert controller.inference_cache_stats["cached_bytes"] == 0
    assert not controller._inference_cache_enabled
    controller.restore()
    assert model[0].forward == original
    assert not hasattr(model, "_a2ui_qat_controller")


def test_config_optout_and_zero3_bypass_cache():
    for zero3 in (False, True):
        model = nn.Sequential(nn.Linear(8, 4)).eval()
        controller = prepare_qat_model(model, {"qat": {
            "exclude_modules": [], "inference_weight_cache": zero3,
        }})
        if zero3:
            model[0].weight.ds_id = 7
        with torch.inference_mode(), qat_inference_cache_scope(model):
            model(torch.ones(1, 8))
            model(torch.ones(1, 8))
            assert controller.inference_cache_stats["hits"] == 0
            assert controller.inference_cache_stats["cached_bytes"] == 0


def test_active_saturation_monitor_preserves_per_forward_observation():
    model, controller = lora_fixture(torch.float32, True)
    observations = []
    controller.saturation_monitor = SimpleNamespace(active=True, observe=lambda *args, **kwargs: observations.append(kwargs["role"]))
    with torch.inference_mode(), controller.inference_cache():
        model(torch.ones(1, 8))
        model(torch.ones(1, 8))
        assert controller.inference_cache_stats["hits"] == 0
    assert observations.count("weight") == 2


def test_cache_invalidates_when_autocast_context_changes():
    model, controller = lora_fixture(torch.float32, False)
    inputs = torch.randn(2, 8)
    with torch.inference_mode(), controller.inference_cache():
        model(inputs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model(inputs)
        assert controller.inference_cache_stats["misses"] == 2
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_inference_created_tensor_scale_bypasses_cache_without_version_counter():
    model, controller = lora_fixture(torch.float32, False)
    inputs = torch.randn(2, 8)
    with torch.inference_mode(), controller.inference_cache():
        model(inputs)
        model[0].scaling["default"] = torch.tensor(0.5)
        actual = model(inputs)
        assert controller.inference_cache_stats["cached_bytes"] == 0
        model[0].scaling["default"].mul_(2)
        changed = model(inputs)
    with torch.inference_mode():
        expected_changed = model(inputs)
        model[0].scaling["default"].mul_(0.5)
        expected = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(changed, expected_changed, rtol=0, atol=0)


def test_host_latency_metrics_separate_first_token_and_decode():
    from ir_training.eval.generate import TokenLatencyStreamer, aggregate_generation_performance
    times = iter([10.0, 12.0, 12.5, 13.0])
    streamer = TokenLatencyStreamer(clock=lambda: next(times))
    streamer.put(torch.tensor([[1, 2, 3]]))  # Prompt is excluded.
    for token in (4, 5, 6):
        streamer.put(torch.tensor([token]))
    metrics = streamer.metrics()
    assert metrics["time_to_first_token_seconds"] == 2.0
    assert metrics["decode_seconds"] == 1.0
    assert metrics["decode_tokens_per_second"] == 2.0
    report = aggregate_generation_performance([
        {"runtime": {**metrics, "generation_seconds": 3.1, "output_tokens": 3}},
        {"runtime": {**metrics, "generation_seconds": 4.1, "output_tokens": 5}},
    ])
    assert report["generation_runtime_generation_seconds_p50"] == pytest.approx(3.6)
    assert report["generation_runtime_generation_seconds_p95"] == pytest.approx(4.05)
    assert report["generation_runtime_time_to_first_token_seconds_measured_rows"] == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_retained_scales_preserve_model_nonfinite_and_signed_zero(dtype, monkeypatch):
    from ir_training.qat import fake_quant

    class ZeroScales(RetainedScales):
        def activation_scale(self, weight_key, edge):
            return 0.0

    torch.manual_seed(71)
    model = nn.Sequential(LoraLinear()).to(dtype=dtype).eval()
    controller = QATController(QATSpec(weight_bits=4, activation_bits=8, exclude_modules=(),
        scale_mode="retained_mobile", activation_quantizer="gemma_mobile_srq",
        ste_gradient="clipped", quantizer="ste_ai_edge"), ZeroScales()).prepare(model)
    inputs = torch.tensor([
        [-0.0] * 8, [0.0] * 8,
        [float("inf")] * 8, [-float("inf")] * 8,
        [float("nan")] * 8,
        [0.125, -0.25, 0.5, 1.0, -1.0, 2.0, -2.0, 0.0],
    ], dtype=dtype)
    calls = []
    original = fake_quant.fake_quantize_activation

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fake_quant, "fake_quantize_activation", tracked)
    with torch.inference_mode():
        expected = model(inputs)
        assert len(calls) == 2
        with controller.inference_cache():
            actual = model(inputs)
        assert len(calls) == 4
    bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.view(bits), expected.view(bits))
    with controller.inference_cache():
        model(torch.ones(2, 8, dtype=dtype)).sum().backward()
    assert len(calls) == 6
    assert model[0].lora_A["default"].weight.grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gradient", ["identity", "clipped"])
def test_inference_srq_omits_only_backward_mask(dtype, gradient):
    from ir_training.qat.fake_quant import mobile_srq_ste
    values = torch.tensor([-float("inf"), -130, -127.5, -0.0, 0.0, 0.5, 127, 130, float("inf"), float("nan")], dtype=dtype)
    with torch.inference_mode():
        for scale in (0.0, 0.03125, 1.0, 27.84252):
            original = mobile_srq_ste(values, scale, ste_gradient=gradient)
            fast = mobile_srq_ste(values, scale, ste_gradient=gradient, inference_forward=True)
            bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
            assert torch.equal(original.view(bits), fast.view(bits))
    with pytest.raises(RuntimeError, match="gradients"):
        mobile_srq_ste(values, 1.0, inference_forward=True)


def test_srq_compile_scope_restores_and_optout_is_explicit():
    model, controller = lora_fixture(torch.float32, True)
    with controller.inference_cache():
        assert controller._compile_srq_enabled
        with controller.inference_cache(compile_srq=False):
            assert not controller._compile_srq_enabled
        assert controller._compile_srq_enabled
    assert not controller._compile_srq_enabled
    controller.inference_compile_srq = False
    with controller.inference_cache():
        assert not controller._compile_srq_enabled
