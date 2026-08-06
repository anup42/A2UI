from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
from torch import nn

from ir_training.common.config import load_yaml
from ir_training.qat.fake_quant import QATSpec, fake_quantize_ste, prepare_qat_model
from ir_training.qat.workflow import validate_qat_config


def test_fake_quant_ste_quantizes_forward_and_keeps_gradient():
    values = torch.tensor([-1.0, -0.25, 0.25, 1.0], requires_grad=True)
    quantized = fake_quantize_ste(values, bits=2, symmetric=True)

    assert not torch.equal(quantized.detach(), values.detach())
    quantized.sum().backward()
    assert values.grad is not None
    assert torch.all(values.grad == 1)


def test_qat_controller_wraps_base_layers_and_restores_them():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(4, 4)
            self.lm_head = nn.Linear(4, 4)

        def forward(self, inputs):
            return self.lm_head(self.base_layer(inputs))

    model = TinyModel()
    inputs = torch.randn(2, 4, requires_grad=True)
    original_forward = model.base_layer.forward
    controller = prepare_qat_model(
        model,
        {
            "qat": {
                "enabled": True,
                "weight_bits": 4,
                "activation_bits": 8,
                "only_base_layers": True,
            }
        },
    )

    assert controller.wrapped_count == 1
    assert controller.wrapped_names == ("base_layer",)
    loss = model(inputs).square().mean()
    loss.backward()
    assert model.base_layer.weight.grad is not None
    assert model.lm_head.forward.__name__ == "forward"

    controller.restore()
    assert model.base_layer.forward == original_forward
    assert controller.summary()["true_fake_quant"] is True
    assert controller.summary()["wrapped_linear_count"] == 1


@pytest.mark.parametrize(
    "config_name",
    [
        "gemma4_e2b_ir_qat_sft.yaml",
        "gemma3_270m_ir_qat_sft.yaml",
        "functiongemma_270m_ir_qat_sft.yaml",
    ],
)
def test_supported_qat_profiles_pass_static_validation(config_name: str):
    config = load_yaml(ROOT / "configs" / "models" / config_name)
    assert validate_qat_config(config) == []
    assert QATSpec.from_config(config).weight_bits == 8
    assert QATSpec.from_config(config).activation_bits == 8


def test_qat_validation_rejects_qlora_and_disabled_qat():
    config = load_yaml(ROOT / "configs" / "models" / "gemma3_270m_ir_qat_sft.yaml")
    config["model"]["load_in_4bit"] = True
    config["qat"]["enabled"] = False
    codes = {issue.code for issue in validate_qat_config(config)}
    assert {"qlora_is_not_qat", "qat_not_enabled"}.issubset(codes)
