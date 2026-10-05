"""CPU numerical regressions for the retained-scale round-then-clamp STE."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

from ir_training.qat.fake_quant import QATSpec, fake_quantize_ste, qat_numeric_contract


def test_bf16_retained_endpoints_keep_gradients_but_saturation_does_not():
    values = torch.tensor(
        [-0.3, -0.2, 0.1, 0.2], dtype=torch.bfloat16, requires_grad=True
    )
    scale = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    result = fake_quantize_ste(
        values, bits=2, quantizer="ste_ai_edge", scale_override=scale,
        ste_gradient="clipped",
    )
    upstream = torch.tensor([2., -3., 4., -5.], dtype=values.dtype)
    result.backward(upstream)

    assert result.dtype == values.dtype
    assert torch.equal(result, torch.tensor([-0.2, -0.2, 0.1, 0.1], dtype=values.dtype))
    assert torch.equal(values.grad, torch.tensor([0., -3., 4., 0.], dtype=values.dtype))
    # The retained calibration is immutable, even if a caller supplies a tensor
    # whose requires_grad flag was set accidentally.
    assert scale.grad is None
    assert scale.item() == torch.tensor(0.1, dtype=torch.float32).item()


def test_clipped_weight_rule_is_versioned_without_relabeling_mobile_activation_policy():
    contract = qat_numeric_contract(QATSpec(ste_gradient="clipped", activation_quantizer="gemma_mobile_srq"))
    assert contract["weight_ste_rule"] == "rounded_code_range_v1"
    assert contract["activation_scale_arithmetic"] == "input_dtype"
    assert contract["activation_signed_range"] == [-128, 127]
    assert "weight_ste_rule" not in qat_numeric_contract(QATSpec(ste_gradient="identity"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("quantizer", ["ste_absmax", "ste_ai_edge"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_rounding_cells_match_pytorch_forward_and_backward(dtype, bits, quantizer, symmetric):
    qmin, qmax = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if symmetric else (0, (1 << bits) - 1)
    if symmetric and quantizer == "ste_ai_edge" and bits == 8:
        qmin += 1
    zero_point = 0 if symmetric else 1 << (bits - 1)
    # A binary-exact scale isolates the clamp/round rule from differences
    # between division and the reference kernel's reciprocal multiplication.
    scale = 0.25
    coordinates = [
        qmin - 1., qmin - 0.75, qmin - 0.5, qmin - 0.25, qmin,
        zero_point, qmax, qmax + 0.25, qmax + 0.5, qmax + 0.75, qmax + 1.,
    ]
    values = ((torch.tensor(coordinates) - zero_point) * scale).to(dtype).requires_grad_()
    # Compare the same low-precision input values in FP32; PyTorch's CPU
    # affine fake quantizer does not support every input dtype on every build.
    reference_input = values.detach().float().requires_grad_()
    reference = torch.fake_quantize_per_tensor_affine(reference_input, scale, zero_point, qmin, qmax)
    result = fake_quantize_ste(
        values, bits=bits, symmetric=symmetric, quantizer=quantizer,
        scale_override=torch.tensor(scale), zero_point_override=zero_point,
        ste_gradient="clipped",
    )
    upstream = torch.arange(1, values.numel() + 1, dtype=dtype)
    result.backward(upstream)
    reference.backward(upstream.float())

    assert torch.equal(result, reference.to(dtype))
    assert torch.equal(values.grad, reference_input.grad.to(dtype))


def test_nonzero_odd_zero_point_preserves_existing_round_before_clamp_contract():
    # Existing converter arithmetic rounds after adding the zero point. At an
    # exact tie an odd offset changes ties-to-even relative to rounding first.
    # Keep that forward policy, and use that same decision in backward.
    values = torch.tensor([-1.5, -1.25, -1.125, -1., 0., 0.125, 0.25, 0.5], requires_grad=True)
    result = fake_quantize_ste(
        values, bits=2, scale_override=torch.tensor(0.5), zero_point_override=1,
        ste_gradient="clipped",
    )
    result.sum().backward()
    assert result.tolist() == [-1.5, -1.5, -1., -1., 0., 0., 0., 0.]
    assert values.grad.tolist() == [1., 1., 1., 1., 1., 1., 0., 0.]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["per_channel", "grouped"])
def test_retained_broadcast_scales_preserve_forward_and_gradient(dtype, layout):
    scale = torch.tensor([[0.1], [0.2]], dtype=torch.float32, requires_grad=True)
    if layout == "grouped":
        scale = scale.detach().repeat(1, 2).requires_grad_()
        expected_scale = scale.detach().repeat_interleave(4, dim=-1)
        options = {"group_size": 4}
        codes = torch.tensor([[-3., -2., 1., 2., -3., -2., 1., 2.]]).repeat(2, 1)
    else:
        expected_scale = scale.detach()
        options = {"per_channel": True, "axis": 0}
        codes = torch.tensor([[-3., -2., 1., 2.]]).repeat(2, 1)
    values = (codes * expected_scale).to(dtype).requires_grad_()
    result = fake_quantize_ste(
        values, bits=2, quantizer="ste_ai_edge", scale_override=scale,
        ste_gradient="clipped", **options,
    )
    result.sum().backward()
    expected_forward = (codes.clamp(-2, 1) * expected_scale).to(dtype)
    expected_gradient = torch.tensor([[0., 1., 1., 0.]], dtype=dtype).repeat(2, values.shape[-1] // 4)
    assert torch.equal(result, expected_forward)
    assert torch.equal(values.grad, expected_gradient)
    assert scale.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gradient", ["identity", "clipped"])
def test_existing_forward_is_unchanged_and_identity_gradient_remains_identity(dtype, gradient):
    generator = torch.Generator().manual_seed(512)
    values = (torch.randn((5, 257), generator=generator) * 3).to(dtype).requires_grad_()
    scale = torch.tensor([[0.001], [0.037], [0.1], [0.25], [1.]], dtype=torch.float32)
    zero_point = torch.tensor([[0.], [1.], [-1.], [2.], [-2.]])
    result = fake_quantize_ste(
        values, bits=4, per_channel=True, scale_override=scale,
        zero_point_override=zero_point, ste_gradient=gradient,
    )
    # Preserve the old forward byte-for-byte, including its surrogate formula.
    prior_dequantized = ((torch.round(values.detach() / scale + zero_point).clamp(-8, 7) - zero_point) * scale).to(dtype)
    prior_surrogate = values.detach()
    if gradient == "clipped":
        prior_inside = ((values.detach() >= (-8 - zero_point) * scale) & (values.detach() <= (7 - zero_point) * scale)).to(dtype)
        prior_surrogate = prior_surrogate * prior_inside
    prior_forward = prior_surrogate + (prior_dequantized - prior_surrogate).detach()
    assert result.dtype == dtype
    assert torch.equal(result, prior_forward)
    if gradient == "identity":
        result.sum().backward()
        assert torch.equal(values.grad, torch.ones_like(values))
