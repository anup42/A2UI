from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
from ir_training.common.config import load_yaml
from ir_training.qat import fake_quant as fake_quant_module
from ir_training.qat.fake_quant import (
    QATSpec,
    _scale_and_zero_point,
    fake_quantize_ste,
    fake_quantize_weight,
    prepare_qat_model,
)
from ir_training.qat.mobile_schema import (
    audit_module_names,
    compare_to_public_schema,
    load_mobile_quant_schema,
)
from ir_training.qat.workflow import validate_qat_config
from ir_training.train.sft import _adapter_checkpoint_manifest
from torch import nn


class FakePeftLoraLinear(nn.Module):
    def __init__(self, *, dropout: float = 0.0):
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(2, 1, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(1, 2, bias=False)}
        )
        self.lora_dropout = nn.ModuleDict(
            {"default": nn.Dropout(dropout)}
        )
        self.scaling = {"default": 1.0}
        self.use_dora = {"default": False}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False

    def get_delta_weight(self, adapter: str):
        return (
            self.lora_B[adapter].weight
            @ self.lora_A[adapter].weight
            * self.scaling[adapter]
        )

    def forward(self, inputs):
        result = self.base_layer(inputs)
        for adapter in self.active_adapters:
            adapter_inputs = self.lora_dropout[adapter](inputs)
            result = result + (
                self.lora_B[adapter](self.lora_A[adapter](adapter_inputs))
                * self.scaling[adapter]
            )
        return result


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


def test_qat_controller_wraps_plain_base_linears_but_skips_lora_matrices():
    class TinyPeftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = FakePeftLoraLinear()
            self.frozen_aux_projection = nn.Linear(2, 2)

    controller = prepare_qat_model(
        TinyPeftModel(),
        {
            "qat": {
                "weight_bits": 4,
                "activation_bits": 8,
                "exclude_modules": [],
                "only_base_layers": True,
            }
        },
    )

    assert set(controller.wrapped_names) == {
        "q_proj",
        "frozen_aux_projection",
    }
    assert "q_proj.lora_A.default" not in controller.wrapped_names
    assert "q_proj.lora_B.default" not in controller.wrapped_names
    assert controller.summary()["wrapped_weight_bits_by_module"] == {
        "q_proj": 4,
        "frozen_aux_projection": 4,
    }
    assert controller.summary()["wrapped_weight_bit_histogram"] == {"4": 2}
    assert controller.summary()["wrapped_effective_lora_names"] == ["q_proj"]
    assert controller.summary()["uncovered_lora_adapter_linear_names"] == []


def test_effective_lora_qat_quantizes_merged_weight_and_keeps_adapter_gradients():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = FakePeftLoraLinear()

        def forward(self, inputs):
            return self.q_proj(inputs)

    model = TinyModel()
    with torch.no_grad():
        model.q_proj.base_layer.weight.copy_(
            torch.tensor([[0.9, 0.2], [-0.7, 0.4]])
        )
        model.q_proj.lora_A["default"].weight.copy_(
            torch.tensor([[0.6, -0.3]])
        )
        model.q_proj.lora_B["default"].weight.copy_(
            torch.tensor([[0.5], [-0.25]])
        )
    original_forward = model.q_proj.forward
    controller = prepare_qat_model(
        model,
        {
            "qat": {
                "weight_bits": 2,
                "activation_bits": 32,
                "weight_per_channel": True,
                "exclude_modules": [],
                "only_base_layers": True,
                "effective_merged_weight": True,
            }
        },
    )
    inputs = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    effective_weight = (
        model.q_proj.base_layer.weight
        + model.q_proj.get_delta_weight("default")
    )
    expected = torch.nn.functional.linear(
        inputs,
        fake_quantize_weight(effective_weight, controller.spec),
    )
    legacy_base_only = torch.nn.functional.linear(
        inputs,
        fake_quantize_weight(model.q_proj.base_layer.weight, controller.spec),
    ) + model.q_proj.lora_B["default"](
        model.q_proj.lora_A["default"](inputs)
    )

    actual = model(inputs)
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, legacy_base_only)
    actual.square().mean().backward()
    assert model.q_proj.lora_A["default"].weight.grad is not None
    assert model.q_proj.lora_B["default"].weight.grad is not None
    assert model.q_proj.base_layer.weight.grad is None

    controller.restore()
    assert model.q_proj.forward == original_forward
    assert torch.allclose(model(inputs), torch.nn.functional.linear(inputs, effective_weight))


def test_effective_lora_qat_rejects_nonzero_adapter_dropout():
    model = nn.Module()
    model.q_proj = FakePeftLoraLinear(dropout=0.05)

    with pytest.raises(ValueError, match="requires dropout=0"):
        prepare_qat_model(
            model,
            {
                "qat": {
                    "weight_bits": 4,
                    "activation_bits": 8,
                    "exclude_modules": [],
                    "effective_merged_weight": True,
                }
            },
        )


@pytest.mark.parametrize(
    ("config_name", "expected_activation_bits"),
    [
        ("gemma4_e2b_mobile_seed_ir_qat_sft.yaml", 8),
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


def test_gemma4_true_qat_keeps_peft_language_model_default_scope():
    config = load_yaml(
        ROOT / "configs" / "models" / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
    )

    # PEFT 0.19+ owns the Gemma 4 language_model q_proj/v_proj regex. The
    # GemmaAdapter `.linear` fallback targets clipped modality wrappers instead.
    assert config["lora"]["target_modules"] == "peft-default"


def test_qat_validation_rejects_qlora_and_disabled_qat():
    config = load_yaml(ROOT / "configs" / "models" / "gemma3_270m_ir_qat_sft.yaml")
    config["model"]["load_in_4bit"] = True
    config["qat"]["enabled"] = False
    codes = {issue.code for issue in validate_qat_config(config)}
    assert {"qlora_is_not_qat", "qat_not_enabled"}.issubset(codes)


def test_qat_validation_requires_effective_weight_and_zero_lora_dropout():
    config = load_yaml(ROOT / "configs" / "models" / "gemma3_270m_ir_qat_sft.yaml")
    config["qat"]["effective_merged_weight"] = False
    config["lora"]["dropout"] = 0.05

    codes = {issue.code for issue in validate_qat_config(config)}

    assert {
        "effective_merged_weight_qat_required",
        "nonzero_lora_dropout_breaks_merged_qat",
    }.issubset(codes)


def test_gemma4_qat_validation_requires_grouped_per_layer_embedding():
    config = load_yaml(
        ROOT / "configs" / "models" / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
    )
    config["qat"]["module_quant_configs"] = {
        "language_model\\.embed_tokens$": {"num_bits": 2},
        "language_model\\.embed_tokens_per_layer$": {
            "num_bits": 4,
            "group_size": 128,
        },
    }

    codes = {issue.code for issue in validate_qat_config(config)}

    assert "gemma4_mobile_observable_layout_mismatch" in codes


def test_training_adapter_manifest_binds_checkpoint_bytes(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"adapter-v1")

    first = _adapter_checkpoint_manifest(tmp_path, role="best_golden")
    weights.write_bytes(b"adapter-v2")
    second = _adapter_checkpoint_manifest(tmp_path, role="best_golden")

    assert first["role"] == "best_golden"
    assert len(first["files"]) == 2
    first_weight = next(
        item for item in first["files"] if item["path"] == weights.name
    )
    second_weight = next(
        item for item in second["files"] if item["path"] == weights.name
    )
    assert first_weight["sha256"] != second_weight["sha256"]


def test_public_gemma4_mobile_schema_preserves_ordered_bit_assignments():
    schema = load_mobile_quant_schema()

    assert schema.quant_method == "gemma"
    assert schema.num_bits == 4
    assert schema.quantize_embeddings is True
    assert schema.bits_for_module("model.language_model.layers.2.mlp.gate_proj") == 4
    assert schema.bits_for_module("model.language_model.layers.20.mlp.gate_proj") == 2
    assert schema.bits_for_module("model.language_model.layers.2.self_attn.q_proj") == 4
    assert schema.bits_for_module("model.language_model.layers.2.per_layer_input_gate") == 8
    assert schema.bits_for_module("model.language_model.per_layer_model_projection") is None
    assert schema.bits_for_module("model.vision_tower.patch_embedder") is None
    assert schema.bits_for_module("unmatched.module") == 4

    assert compare_to_public_schema(schema.to_dict(), schema) == []

    litertlm_schema = load_mobile_quant_schema(
        ROOT
        / "configs"
        / "quantization"
        / "gemma4_e2b_mobile_litertlm_schema.yaml"
    )
    assert (
        litertlm_schema.group_size_for_module(
            "language_model.embed_tokens_per_layer"
        )
        == 256
    )


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
    assert spec.weight_bits_for_module("language_model.per_layer_model_projection") is None
    assert spec.weight_bits_for_module("model.vision_tower.patch_embedder") is None


def test_qat_precision_rules_match_peft_and_multimodal_wrapper_prefixes():
    spec = QATSpec.from_config(
        {
            "qat": {
                "schema_path": "configs/quantization/gemma4_e2b_mobile_litertlm_schema.yaml",
                "exclude_modules": [],
            }
        }
    )

    assert spec.weight_bits_for_module("base_model.model.lm_head") == 2
    assert (
        spec.weight_bits_for_module(
            "base_model.model.model.language_model.layers.20.mlp.gate_proj"
        )
        == 2
    )
    assert (
        spec.weight_bits_for_module(
            "base_model.model.model.language_model.layers.2.self_attn.q_proj"
        )
        == 4
    )
    # Text-only Gemma4ForCausalLM drops the multimodal ``language_model``
    # component; these aliases must still hit the official precision rules.
    assert (
        spec.weight_bits_for_module(
            "base_model.model.model.layers.2.mlp.gate_proj"
        )
        == 4
    )
    assert (
        spec.weight_bits_for_module(
            "base_model.model.model.layers.20.mlp.gate_proj"
        )
        == 2
    )
    assert spec.weight_bits_for_module("base_model.model.model.embed_tokens") == 2
    assert (
        spec.weight_bits_for_module(
            "base_model.model.model.embed_tokens_per_layer"
        )
        == 4
    )
    assert (
        spec.group_size_for_module(
            "base_model.model.model.embed_tokens_per_layer"
        )
        == 256
    )
    assert (
        spec.group_size_for_module("base_model.model.model.embed_tokens") is None
    )
    assert (
        spec.weight_bits_for_module(
            "base_model.model.model.per_layer_model_projection"
        )
        == 8
    )
    public_spec = QATSpec.from_config(
        {
            "qat": {
                "schema_path": "configs/quantization/gemma4_e2b_mobile_public_schema.yaml"
            }
        }
    )
    assert (
        public_spec.weight_bits_for_module(
            "base_model.model.model.vision_tower.patch_embedder"
        )
        is None
    )


def test_litertlm_schema_wraps_embeddings_and_module_specific_linear_bits():
    class TinyMobileModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.embed_tokens = nn.Embedding(16, 4)
            layer_20 = nn.Module()
            layer_20.mlp = nn.Module()
            layer_20.mlp.gate_proj = nn.Linear(4, 4)
            self.language_model.layers = nn.ModuleDict({"20": layer_20})
            self.language_model.per_layer_model_projection = nn.Linear(4, 4)

        def forward(self, token_ids):
            hidden = self.language_model.embed_tokens(token_ids)
            hidden = self.language_model.layers["20"].mlp.gate_proj(hidden)
            return self.language_model.per_layer_model_projection(hidden)

    model = TinyMobileModel()
    controller = prepare_qat_model(
        model,
        {
            "qat": {
                "schema_path": "configs/quantization/gemma4_e2b_mobile_litertlm_schema.yaml",
                "quantizer": "ste_ai_edge",
                "exclude_modules": [],
                "only_base_layers": True,
                "quantize_embeddings": True,
            }
        },
    )

    assert "language_model.embed_tokens" in controller.wrapped_names
    assert "language_model.layers.20.mlp.gate_proj" in controller.wrapped_names
    assert "language_model.per_layer_model_projection" in controller.wrapped_names
    assert controller.spec.quantize_embeddings is True
    assert controller.summary()["wrapped_weight_bits_by_module"] == {
        "language_model.embed_tokens": 2,
        "language_model.layers.20.mlp.gate_proj": 2,
        "language_model.per_layer_model_projection": 8,
    }
    assert controller.summary()["wrapped_weight_bit_histogram"] == {"2": 2, "8": 1}
    output = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert output.shape == (1, 3, 4)
    controller.restore()


def test_grouped_embedding_qat_quantizes_only_selected_rows(monkeypatch):
    class TinyGroupedEmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens_per_layer = nn.Embedding(4, 4)

        def forward(self, token_ids):
            return self.embed_tokens_per_layer(token_ids)

    model = TinyGroupedEmbeddingModel()
    with torch.no_grad():
        model.embed_tokens_per_layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, 1.0, 100.0, 100.0],
                    [2.0, 2.0, 20.0, 20.0],
                    [3.0, 3.0, 30.0, 30.0],
                    [4.0, 4.0, 40.0, 40.0],
                ]
            )
        )
    observed_shapes: list[tuple[int, ...]] = []
    original_fake_quantize_weight = fake_quant_module.fake_quantize_weight

    def capture_selected_shape(weight, spec):
        observed_shapes.append(tuple(weight.shape))
        return original_fake_quantize_weight(weight, spec)

    monkeypatch.setattr(
        fake_quant_module, "fake_quantize_weight", capture_selected_shape
    )
    controller = prepare_qat_model(
        model,
        {
            "qat": {
                "weight_bits": 8,
                "activation_bits": 32,
                "quantizer": "ste_ai_edge",
                "exclude_modules": [],
                "quantize_embeddings": True,
                "module_quant_configs": {
                    "^embed_tokens_per_layer$": {
                        "num_bits": 2,
                        "group_size": 2,
                    }
                },
            }
        },
    )

    output = model(torch.tensor([[0, 0, 1]], dtype=torch.long))

    assert observed_shapes == [(2, 4)]
    torch.testing.assert_close(
        output,
        torch.tensor(
            [[[1.0, 1.0, 100.0, 100.0], [1.0, 1.0, 100.0, 100.0], [2.0, 2.0, 20.0, 20.0]]]
        ),
    )
    assert controller.summary()["wrapped_group_sizes_by_module"] == {
        "embed_tokens_per_layer": 2
    }
    controller.restore()
