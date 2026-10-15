"""Effective QAT weights must match the actual PEFT serialization merge."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from ir_training.qat.fake_quant import (
    QATSpec,
    _effective_lora_weight,
    fake_quantize_weight,
)


class LoraProjection(torch.nn.Module):
    def __init__(self, adapters: int = 1):
        super().__init__()
        self.base_layer = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(4, 32) * 0.1) for _ in range(adapters)
        ])
        self.lora_B = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(32, 4) * 0.1) for _ in range(adapters)
        ])

    def get_delta_weight(self, adapter):
        return self.lora_B[int(adapter)] @ self.lora_A[int(adapter)]


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("adapters", [1, 2])
def test_effective_weight_matches_inplace_export_rounding_and_preserves_gradients(amp, adapters):
    torch.manual_seed(42)
    module = LoraProjection(adapters)
    original = module.base_layer.weight.detach().clone()
    names = tuple(str(index) for index in range(adapters))
    expected = original.clone()
    with torch.no_grad():
        for name in names:
            expected.add_(module.get_delta_weight(name))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        observed = _effective_lora_weight(module, names)
    assert observed.dtype == original.dtype
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(module.base_layer.weight, original, rtol=0, atol=0)
    observed.float().square().sum().backward()
    assert module.base_layer.weight.grad is None
    for parameter in (*module.lora_A, *module.lora_B):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.count_nonzero(parameter.grad))


@pytest.mark.parametrize("amp", [False, True])
def test_real_peft_bfloat16_merge_has_identical_effective_and_w8_weights(amp):
    peft = pytest.importorskip("peft")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)

        def forward(self, inputs):
            return self.q_proj(inputs)

    torch.manual_seed(42)
    model = peft.get_peft_model(Model(), peft.LoraConfig(
        r=4, lora_alpha=4, target_modules=["q_proj"], lora_dropout=0,
        init_lora_weights=False,
    ))
    projection = model.get_base_model().q_proj
    assert projection.base_layer.weight.dtype == torch.bfloat16
    assert projection.lora_A["default"].weight.dtype == torch.float32
    # This is the same default merge used by merge_lora_adapter, outside AMP.
    merged = copy.deepcopy(model).merge_and_unload().q_proj.weight
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        effective = _effective_lora_weight(projection, ("default",))
    torch.testing.assert_close(effective, merged, rtol=0, atol=0)
    spec = QATSpec.from_config({"qat": {
        "weight_bits": 8, "activation_bits": 32, "quantizer": "ste_ai_edge",
        "weight_per_channel": True, "weight_axis": 0,
    }})
    torch.testing.assert_close(
        fake_quantize_weight(effective, spec), fake_quantize_weight(merged, spec),
        rtol=0, atol=0,
    )
