from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
from ir_training.common.config import load_yaml
from ir_training.qat import fake_quant as fake_quant_module
from ir_training.qat import mobile_qparams as mobile_qparams_module
from ir_training.qat.fake_quant import (
    AI_EDGE_MIN_SCALE,
    QATSpec,
    _round_ai_edge_blockwise_scale,
    _scale_and_zero_point,
    fake_quantize_ste,
    fake_quantize_weight,
    prepare_qat_model,
    qat_numeric_contract,
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


class SyntheticRetainedMobileQParams:
    """Small in-memory qparameter authority for live-scope tests."""

    def __init__(
        self,
        *,
        expected_keys: set[str],
        resolved_keys: dict[str, str],
    ):
        self.expected_keys = set(expected_keys)
        self.resolved_keys = dict(resolved_keys)

    def resolve_weight_key(self, candidates):
        for candidate in candidates:
            if candidate in self.resolved_keys:
                return self.resolved_keys[candidate]
        return None

    def load_scale(self, weight_key, *, weight_shape, bits, group_size):
        assert weight_key
        assert bits == 4
        assert group_size is None
        return torch.full((int(weight_shape[0]), 1), 0.25, dtype=torch.float32)

    def activation_scale(self, weight_key, edge):
        assert weight_key
        assert edge in {"input", "output"}
        return None

    def trainable_projection_weight_keys(self):
        return tuple(sorted(self.expected_keys))

    def summary(self):
        return {
            "verified": True,
            "trainable_projection_count": len(self.expected_keys),
        }


def _synthetic_retained_scope_model(*module_names: str):
    model = nn.Module()
    for module_name in module_names:
        setattr(model, module_name, FakePeftLoraLinear())
    return model


def _synthetic_retained_scope_config(*, expected_count: int):
    return {
        "qat": {
            "enabled": True,
            "weight_bits": 4,
            "activation_bits": 32,
            "weight_symmetric": True,
            "activation_symmetric": True,
            "weight_per_channel": True,
            "weight_axis": 0,
            "group_size": None,
            "only_base_layers": True,
            "effective_merged_weight": True,
            "effective_lora_only": True,
            "exclude_modules": [],
            "quantize_embeddings": False,
            "quantizer": "ste_ai_edge",
            "scale_mode": "retained_mobile",
            "mobile_qparams_contract": "synthetic-mobile-qparams.json",
            "fixed_scale_required": True,
            "fixed_activation_scale_required": False,
            "expected_effective_lora_modules": expected_count,
            "ste_gradient": "clipped",
        }
    }


def _install_synthetic_mobile_qparams(monkeypatch, qparams):
    monkeypatch.setattr(
        mobile_qparams_module,
        "MobileQParams",
        lambda _contract_path: qparams,
    )


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


def test_ai_edge_fake_quant_uses_float32_scale_and_preserves_bfloat16_forward():
    values = torch.tensor(
        [[-0.76953125, -0.310546875, 0.109375, 0.73046875]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    scale, zero_point, qmin, qmax = _scale_and_zero_point(
        values,
        bits=8,
        symmetric=True,
        reduce_dims=(1,),
        eps=AI_EDGE_MIN_SCALE,
        quantizer="ste_ai_edge",
    )

    expected_scale = values.detach().float().abs().amax(dim=1, keepdim=True) / 127
    assert scale.dtype == torch.float32
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    assert (qmin, qmax) == (-127, 127)
    assert zero_point.shape == scale.shape

    simulated = fake_quantize_ste(
        values,
        bits=8,
        symmetric=True,
        per_channel=True,
        axis=0,
        eps=AI_EDGE_MIN_SCALE,
        quantizer="ste_ai_edge",
    )
    assert simulated.dtype == torch.bfloat16
    simulated.float().sum().backward()
    assert torch.all(values.grad == 1)


def test_ai_edge_fake_quant_matches_public_channelwise_codes_when_available():
    qtyping = pytest.importorskip("ai_edge_quantizer.qtyping")
    public = pytest.importorskip(
        "ai_edge_quantizer.algorithms.uniform_quantize.uniform_quantize_tensor"
    )
    values = torch.tensor(
        [
            [-0.76953125, -0.310546875, -0.109375, 0.109375, 0.73046875],
            [-0.421875, -0.203125, 0.015625, 0.28125, 0.578125],
        ],
        dtype=torch.bfloat16,
    )
    source = values.float().numpy()

    for bits in (2, 4, 8):
        scale, zero_point, qmin, qmax = _scale_and_zero_point(
            values,
            bits=bits,
            symmetric=True,
            reduce_dims=(1,),
            eps=AI_EDGE_MIN_SCALE,
            quantizer="ste_ai_edge",
        )
        public_zp, public_scale = public.tensor_zp_scale_from_min_max(
            source.min(axis=1),
            source.max(axis=1),
            bits,
            True,
            qtyping.QuantGranularity.CHANNELWISE,
        )
        params = qtyping.UniformQuantParams(
            num_bits=bits,
            scale=public_scale,
            zero_point=public_zp,
            symmetric=True,
            quantized_dimension=0,
        )
        public_codes = public.uniform_quantize(source, params)
        ours_codes = (
            torch.round(values.float() / scale + zero_point)
            .clamp(qmin, qmax)
            .to(torch.int8)
            .numpy()
        )
        assert scale.squeeze(1).numpy().tobytes() == public_scale.tobytes()
        assert ours_codes.tobytes() == public_codes.tobytes()


@pytest.mark.parametrize(
    ("bits", "codes", "scales"),
    [
        (
            2,
            [[-2, -1, 0, 1], [-1, 0, 1, -2]],
            [[0.5], [0.125]],
        ),
        (
            4,
            [
                [-8, -7, -1, 0, 1, 5, 6, 7],
                [-6, -3, -1, 0, 1, 2, 4, 6],
            ],
            [[0.25], [0.0625]],
        ),
        (
            8,
            [
                [-100, -37, -1, 0, 1, 44, 95],
                [-64, -17, -1, 0, 1, 23, 63],
            ],
            [[0.03125], [0.0078125]],
        ),
    ],
)
def test_retained_mobile_scales_are_idempotent_and_preserve_codes(
    bits: int,
    codes: list[list[int]],
    scales: list[list[float]],
):
    """A zero-delta QAT pass must preserve Google's released cell centers.

    The W8 rows deliberately do not occupy either endpoint. Recomputing an
    abs-max scale can therefore not pass this test by coincidence.
    """

    integer_codes = torch.tensor(codes, dtype=torch.int8)
    retained_scales = torch.tensor(scales, dtype=torch.float32)
    released_centers = integer_codes.float() * retained_scales

    simulated = fake_quantize_ste(
        released_centers,
        bits=bits,
        symmetric=True,
        per_channel=True,
        axis=0,
        quantizer="ste_ai_edge",
        eps=AI_EDGE_MIN_SCALE,
        scale_override=retained_scales,
    )
    recovered_codes = torch.round(simulated / retained_scales).to(torch.int8)

    torch.testing.assert_close(simulated, released_centers, rtol=0, atol=0)
    assert torch.equal(recovered_codes, integer_codes)
    assert torch.equal(recovered_codes == 0, integer_codes == 0)


def test_retained_grouped_w4_scales_preserve_each_group_and_code():
    integer_codes = torch.tensor(
        [
            [-8, -3, 0, 7, -7, -1, 1, 6],
            [-6, -2, 2, 5, -8, 0, 3, 7],
        ],
        dtype=torch.int8,
    )
    # This is the sidecar/storage shape: [rows, groups]. fake_quantize_ste
    # must normalize it for the internal [rows, groups, group_width] view.
    retained_scales = torch.tensor(
        [[0.5, 0.125], [0.25, 0.0625]], dtype=torch.float32
    )
    expanded_scales = retained_scales.repeat_interleave(4, dim=1)
    released_centers = integer_codes.float() * expanded_scales

    simulated = fake_quantize_ste(
        released_centers,
        bits=4,
        symmetric=True,
        per_channel=True,
        axis=0,
        group_size=4,
        quantizer="ste_ai_edge",
        eps=AI_EDGE_MIN_SCALE,
        scale_override=retained_scales,
    )
    recovered_codes = torch.round(simulated / expanded_scales).to(torch.int8)

    torch.testing.assert_close(simulated, released_centers, rtol=0, atol=0)
    assert torch.equal(recovered_codes, integer_codes)


def test_retained_quantization_rejects_zero_point_without_scale():
    with pytest.raises(ValueError, match="scale_override"):
        fake_quantize_ste(
            torch.tensor([[-1.0, 0.0, 0.5]]),
            bits=2,
            symmetric=True,
            quantizer="ste_ai_edge",
            zero_point_override=torch.zeros((1, 1)),
        )


def test_retained_asymmetric_zero_point_preserves_codes():
    integer_codes = torch.tensor([0, 3, 8, 15], dtype=torch.int8)
    retained_scale = torch.tensor(0.25)
    retained_zero_point = torch.tensor(8.0)
    released_centers = (
        integer_codes.float() - retained_zero_point
    ) * retained_scale

    simulated = fake_quantize_ste(
        released_centers,
        bits=4,
        symmetric=False,
        scale_override=retained_scale,
        zero_point_override=retained_zero_point,
    )
    recovered_codes = torch.round(
        simulated / retained_scale + retained_zero_point
    ).to(torch.int8)

    torch.testing.assert_close(simulated, released_centers, rtol=0, atol=0)
    assert torch.equal(recovered_codes, integer_codes)


@pytest.mark.parametrize(
    "bad_scale",
    [
        torch.ones((2, 3)),
        torch.ones((2, 1, 1)),
    ],
)
def test_retained_channelwise_scale_rejects_shape_mismatch(bad_scale):
    with pytest.raises(ValueError, match="scale_override.*shape"):
        fake_quantize_ste(
            torch.ones((2, 4)),
            bits=4,
            symmetric=True,
            per_channel=True,
            axis=0,
            quantizer="ste_ai_edge",
            scale_override=bad_scale,
        )


def test_retained_grouped_scale_rejects_wrong_group_count():
    with pytest.raises(ValueError, match="scale_override.*shape"):
        fake_quantize_ste(
            torch.ones((2, 8)),
            bits=4,
            symmetric=True,
            per_channel=True,
            axis=0,
            group_size=4,
            quantizer="ste_ai_edge",
            scale_override=torch.ones((2, 1)),
        )


def test_clipped_ste_blocks_gradients_outside_retained_integer_range():
    values = torch.tensor(
        [-2.0, -1.0, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
        dtype=torch.float32,
        requires_grad=True,
    )

    simulated = fake_quantize_ste(
        values,
        bits=2,
        symmetric=True,
        quantizer="ste_ai_edge",
        scale_override=torch.tensor(0.5),
        ste_gradient="clipped",
    )
    simulated.sum().backward()

    # W2 with scale 0.5 represents [-1.0, 0.5]. Bounds are inclusive.
    torch.testing.assert_close(
        values.grad,
        torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        rtol=0,
        atol=0,
    )


def test_clipped_ste_rejects_unknown_gradient_policy():
    with pytest.raises(ValueError, match="ste_gradient"):
        fake_quantize_ste(
            torch.ones(2),
            bits=4,
            scale_override=torch.tensor(0.5),
            ste_gradient="straight_through_everywhere",
        )


def test_ai_edge_grouped_scale_uses_public_bfloat16_float16_storage_rounding():
    values = torch.linspace(-0.8, 0.7, 512, dtype=torch.float32).reshape(
        2, 1, 256
    ).to(torch.bfloat16)
    scale, _, _, _ = _scale_and_zero_point(
        values,
        bits=4,
        symmetric=True,
        reduce_dims=(2,),
        eps=AI_EDGE_MIN_SCALE,
        quantizer="ste_ai_edge",
    )

    rounded = _round_ai_edge_blockwise_scale(scale)

    assert rounded.dtype == torch.float32
    torch.testing.assert_close(
        rounded,
        scale.to(torch.bfloat16).to(torch.float16).to(torch.float32),
        rtol=0,
        atol=0,
    )


def test_grouped_qat_rejects_non_divisible_deployment_shape():
    with pytest.raises(ValueError, match="not divisible"):
        fake_quantize_ste(
            torch.ones((2, 7)),
            bits=4,
            symmetric=True,
            per_channel=True,
            axis=0,
            group_size=4,
            quantizer="ste_ai_edge",
            eps=AI_EDGE_MIN_SCALE,
        )


def test_ai_edge_qat_spec_defaults_to_public_minimum_scale():
    spec = QATSpec.from_config({"qat": {"quantizer": "ste_ai_edge"}})

    assert spec.eps == AI_EDGE_MIN_SCALE
    assert qat_numeric_contract(spec)["public_ai_edge_numeric_contract"] is True


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


def test_zero_lora_effective_weight_matches_fixed_scale_mobile_base():
    model = FakePeftLoraLinear()
    retained_scale = torch.tensor([[0.5], [0.125]], dtype=torch.float32)
    integer_codes = torch.tensor([[-2, -1], [0, 1]], dtype=torch.int8)
    released_centers = integer_codes.float() * retained_scale
    with torch.no_grad():
        model.base_layer.weight.copy_(released_centers)
        model.lora_A["default"].weight.copy_(torch.tensor([[0.75, -0.25]]))
        model.lora_B["default"].weight.zero_()

    effective_weight = (
        model.base_layer.weight + model.get_delta_weight("default")
    )
    simulated_weight = fake_quantize_ste(
        effective_weight,
        bits=2,
        symmetric=True,
        per_channel=True,
        axis=0,
        quantizer="ste_ai_edge",
        eps=AI_EDGE_MIN_SCALE,
        scale_override=retained_scale,
        ste_gradient="clipped",
    )
    inputs = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    actual = torch.nn.functional.linear(inputs, simulated_weight)
    expected = torch.nn.functional.linear(inputs, released_centers)

    torch.testing.assert_close(effective_weight, released_centers, rtol=0, atol=0)
    torch.testing.assert_close(simulated_weight, released_centers, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    recovered_codes = torch.round(simulated_weight / retained_scale).to(torch.int8)
    assert torch.equal(recovered_codes, integer_codes)

    actual.square().mean().backward()
    assert model.lora_A["default"].weight.grad is not None
    assert model.lora_B["default"].weight.grad is not None


def test_retained_mobile_live_scope_rejects_partial_bound_set(monkeypatch):
    qparams = SyntheticRetainedMobileQParams(
        expected_keys={"q_proj.weight", "v_proj.weight"},
        resolved_keys={"q_proj.weight": "q_proj.weight"},
    )
    _install_synthetic_mobile_qparams(monkeypatch, qparams)
    model = _synthetic_retained_scope_model("q_proj")

    with pytest.raises(ValueError, match="Live effective-LoRA scope differs") as exc:
        prepare_qat_model(
            model,
            _synthetic_retained_scope_config(expected_count=2),
        )

    assert "v_proj.weight" in str(exc.value)
    assert "wrapped=1" in str(exc.value)
    assert "bindings=1" in str(exc.value)


def test_retained_mobile_live_scope_rejects_wrong_and_extra_bound_names(
    monkeypatch,
):
    qparams = SyntheticRetainedMobileQParams(
        expected_keys={"q_proj.weight", "v_proj.weight"},
        resolved_keys={
            "q_proj.weight": "unexpected_proj.weight",
            "v_proj.weight": "v_proj.weight",
        },
    )
    _install_synthetic_mobile_qparams(monkeypatch, qparams)
    model = _synthetic_retained_scope_model("q_proj", "v_proj")

    with pytest.raises(ValueError, match="Live effective-LoRA scope differs") as exc:
        prepare_qat_model(
            model,
            _synthetic_retained_scope_config(expected_count=2),
        )

    message = str(exc.value)
    assert "q_proj.weight" in message
    assert "unexpected_proj.weight" in message
    assert "wrapped=2" in message
    assert "bindings=2" in message


def test_retained_mobile_live_scope_accepts_exact_names_and_count(monkeypatch):
    expected_keys = {"q_proj.weight", "v_proj.weight"}
    qparams = SyntheticRetainedMobileQParams(
        expected_keys=expected_keys,
        resolved_keys={key: key for key in expected_keys},
    )
    _install_synthetic_mobile_qparams(monkeypatch, qparams)
    model = _synthetic_retained_scope_model("q_proj", "v_proj")

    controller = prepare_qat_model(
        model,
        _synthetic_retained_scope_config(expected_count=2),
    )

    summary = controller.summary()
    assert controller.wrapped_effective_lora_count == 2
    assert summary["retained_qparams_binding_count"] == 2
    assert {
        binding["weight_key"]
        for binding in summary["retained_qparams_bindings"].values()
    } == expected_keys
    controller.restore()


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


@pytest.mark.parametrize(
    "config_name",
    [
        "gemma4_e2b_mobile_seed_ir_qat_sft.yaml",
        "gemma3_270m_ir_qat_sft.yaml",
        "functiongemma_270m_ir_qat_sft.yaml",
    ],
)
def test_deployable_qat_profiles_use_exact_golden100_v5_4_tensorboard_eval(
    config_name: str,
):
    config = load_yaml(ROOT / "configs" / "models" / config_name)
    training = config["training"]
    golden = config["golden_eval"]

    assert training["report_to"] == "tensorboard"
    assert training["logging_dir"].endswith("/tensorboard")
    assert golden["enabled"] is True
    assert golden["dataset_dir"] == "outputs/datasets/golden100_stage3_eval"
    assert golden["max_rows"] == 100
    assert golden["required_rows"] == 100
    assert golden["require_exact_rows"] is True
    assert golden["require_unique_rows"] is True
    assert golden["trigger"] == "evaluate"
    assert golden["interval"] == 1
    assert golden["metric_version"] == "dual"
    assert golden["metric_for_best_model"] == "generation_reward_v5_4_avg"
    assert golden["tensorboard"] is True
    assert golden["metric_log_prefix"] == "golden100"


def test_golden100_preparation_config_fails_closed_on_count_and_identity():
    config = load_yaml(ROOT / "configs" / "datasets" / "golden100_stage3_eval.yaml")

    assert config["run"]["output_dir"] == "outputs/datasets/golden100_stage3_eval"
    assert config["filters"]["required_accepted_rows"] == 100
    assert config["filters"]["require_exact_accepted_rows"] is True
    assert config["filters"]["require_unique_source_ids"] is True


@pytest.mark.parametrize("name", [
    "gemma4_e2b_mobile_seed_ir_qat_sft.yaml",
    "gemma4_e2b_a2ui_express_official_qat.yaml",
    "gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml",
])
def test_gemma4_retained_qat_default_is_resolved_from_verified_mobile_contract(name):
    config = load_yaml(ROOT / "configs" / "models" / name)

    # This spelling is kept for saved-config compatibility. The official mobile
    # SFT path resolves it from seed-bound qparams, not PEFT's narrower q/v map.
    assert config["lora"]["target_modules"] == "peft-default"
    assert config["qat"]["scale_mode"] == "retained_mobile"
    assert config["qat"]["expected_effective_lora_modules"] == 205


def test_qat_validation_rejects_qlora_and_disabled_qat():
    config = load_yaml(ROOT / "configs" / "models" / "gemma3_270m_ir_qat_sft.yaml")
    config["model"]["load_in_4bit"] = True
    config["qat"]["enabled"] = False
    codes = {issue.code for issue in validate_qat_config(config)}
    assert {"qlora_is_not_qat", "qat_not_enabled"}.issubset(codes)


def test_qat_validation_rejects_ai_edge_minimum_scale_drift():
    config = load_yaml(ROOT / "configs" / "models" / "gemma3_270m_ir_qat_sft.yaml")
    config["qat"]["eps"] = 1e-8

    codes = {issue.code for issue in validate_qat_config(config)}

    assert "ai_edge_numeric_contract_mismatch" in codes


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


def test_gemma4_qat_validation_requires_both_architecture_load_gates():
    config = load_yaml(
        ROOT / "configs" / "models" / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
    )
    config["model"].pop("architecture_preflight_required")
    config["model"]["require_exact_checkpoint_keys"] = False

    codes = {issue.code for issue in validate_qat_config(config)}

    assert "gemma4_architecture_preflight_required" in codes
    assert "gemma4_exact_checkpoint_keys_required" in codes


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


def test_embedding_qat_preserves_scaled_embedding_forward_contract():
    class ScaledEmbedding(nn.Embedding):
        def __init__(self):
            super().__init__(8, 4)
            self.register_buffer("embed_scale", torch.tensor(2.5), persistent=False)

        def forward(self, input_ids):
            return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)

    class TinyScaledEmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = ScaledEmbedding()

        def forward(self, token_ids):
            return self.embed_tokens(token_ids)

    torch.manual_seed(7)
    baseline_model = TinyScaledEmbeddingModel()
    qat_model = copy.deepcopy(baseline_model)
    controller = prepare_qat_model(
        qat_model,
        {
            "qat": {
                "weight_bits": 8,
                "activation_bits": 32,
                "quantizer": "ste_ai_edge",
                "exclude_modules": [],
                "quantize_embeddings": True,
            }
        },
    )

    token_ids = torch.tensor([[1, 2, 1]], dtype=torch.long)
    expected = baseline_model(token_ids)
    observed = qat_model(token_ids)
    torch.testing.assert_close(observed, expected, rtol=0.05, atol=0.05)
    controller.restore()
