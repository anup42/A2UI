from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
from ir_training.common.config import load_yaml
from ir_training.qat.fake_quant import (
    QATSpec,
    _scale_and_zero_point,
    fake_quantize_ste,
    prepare_qat_model,
)
from ir_training.qat.mobile_schema import (
    audit_module_names,
    compare_to_public_schema,
    load_mobile_quant_schema,
)
from ir_training.qat.workflow import validate_qat_config
from torch import nn


def test_fake_quant_ste_quantizes_forward_and_keeps_gradient():
    values = torch.tensor([-1.0, -0.25, 0.25, 1.0], requires_grad=True)
    quantized = fake_quantize_ste(values, bits=2, symmetric=True)

    assert not torch.equal(quantized.detach(), values.detach())
    quantized.sum().backward()
    assert values.grad is not None
    assert torch.all(values.grad == 1)


def test_ai_edge_fake_quant_uses_full_low_bit_and_narrow_w8_ranges():
    values = torch.tensor([[-2.0, 2.0]], dtype=torch.float32)
    scale, _, qmin, qmax = _scale_and_zero_point(
        values,
        bits=2,
        symmetric=True,
        reduce_dims=(1,),
        eps=1e-8,
        quantizer="ste_ai_edge",
    )
    assert qmin == -2
    assert qmax == 1
    assert torch.allclose(scale, torch.tensor([[2.0]]))

    scale, _, qmin, qmax = _scale_and_zero_point(
        values,
        bits=8,
        symmetric=True,
        reduce_dims=(1,),
        eps=1e-8,
        quantizer="ste_ai_edge",
    )
    assert qmin == -127
    assert qmax == 127
    assert torch.allclose(scale, torch.tensor([[2.0 / 127.0]]))


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
    ("config_name", "expected_activation_bits"),
    [
        ("gemma4_e2b_ir_qat_sft.yaml", 8),
        ("gemma3_270m_ir_qat_sft.yaml", 32),
        ("functiongemma_270m_ir_qat_sft.yaml", 8),
    ],
)
def test_supported_qat_profiles_pass_static_validation(
    config_name: str, expected_activation_bits: int
):
    config = load_yaml(ROOT / "configs" / "models" / config_name)
    assert validate_qat_config(config) == []
    assert QATSpec.from_config(config).weight_bits == 8
    assert QATSpec.from_config(config).activation_bits == expected_activation_bits


def test_qat_validation_rejects_qlora_and_disabled_qat():
    config = load_yaml(ROOT / "configs" / "models" / "gemma3_270m_ir_qat_sft.yaml")
    config["model"]["load_in_4bit"] = True
    config["qat"]["enabled"] = False
    codes = {issue.code for issue in validate_qat_config(config)}
    assert {"qlora_is_not_qat", "qat_not_enabled"}.issubset(codes)


def test_public_gemma4_mobile_schema_preserves_ordered_bit_assignments():
    schema = load_mobile_quant_schema()

    assert schema.quant_method == "gemma"
    assert schema.num_bits == 4
    assert schema.quantize_embeddings is True
    assert schema.bits_for_module("model.language_model.layers.2.mlp.gate_proj") == 4
    assert schema.bits_for_module("model.language_model.layers.20.mlp.gate_proj") == 2
    assert schema.bits_for_module("model.language_model.layers.2.self_attn.q_proj") == 4
    assert schema.bits_for_module("model.language_model.layers.2.per_layer_input_gate") == 8
    assert schema.bits_for_module("model.vision_tower.patch_embedder") is None
    assert schema.bits_for_module("unmatched.module") == 4

    assert compare_to_public_schema(schema.to_dict(), schema) == []


def test_public_schema_audit_reports_module_distribution():
    schema = load_mobile_quant_schema()
    audit = audit_module_names(
        [
            "model.language_model.layers.2.mlp.gate_proj",
            "model.language_model.layers.20.mlp.gate_proj",
            "model.vision_tower.patch_embedder",
        ],
        schema,
    )

    assert audit["module_count"] == 3
    assert audit["bit_counts"] == {"2": 1, "4": 1, "excluded": 1}


def test_qat_spec_can_load_public_schema_for_module_bits():
    spec = QATSpec.from_config(
        {
            "qat": {
                "schema_path": "configs/quantization/gemma4_e2b_mobile_public_schema.yaml",
                "weight_bits": 8,
                "activation_bits": 8,
            }
        }
    )

    assert spec.weight_bits_for_module("language_model.layers.2.mlp.gate_proj") == 4
    assert spec.weight_bits_for_module("language_model.layers.20.mlp.gate_proj") == 2
    assert spec.weight_bits_for_module("model.vision_tower.patch_embedder") is None


def test_public_schema_wraps_embeddings_and_module_specific_linear_bits():
    class TinyMobileModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.embed_tokens = nn.Embedding(16, 4)
            self.language_model.layers = nn.Module()
            self.language_model.layers.layer_20 = nn.Module()
            self.language_model.layers.layer_20.mlp = nn.Module()
            self.language_model.layers.layer_20.mlp.gate_proj = nn.Linear(4, 4)

        def forward(self, token_ids):
            hidden = self.language_model.embed_tokens(token_ids)
            return self.language_model.layers.layer_20.mlp.gate_proj(hidden)

    model = TinyMobileModel()
    controller = prepare_qat_model(
        model,
        {
            "qat": {
                "schema_path": "configs/quantization/gemma4_e2b_mobile_public_schema.yaml",
                "quantizer": "ste_ai_edge",
                "exclude_modules": [],
                "only_base_layers": False,
                "quantize_embeddings": True,
            }
        },
    )

    assert "language_model.embed_tokens" in controller.wrapped_names
    assert "language_model.layers.layer_20.mlp.gate_proj" in controller.wrapped_names
    assert controller.spec.quantize_embeddings is True
    output = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert output.shape == (1, 3, 4)
    controller.restore()
