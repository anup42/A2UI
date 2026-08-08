#!/usr/bin/env python3
"""Compare the training STE with the tested public AI Edge quantizer.

This is a small, deterministic, no-model/no-training check. It verifies the
integer ranges, FLOAT32 min/max scale calculation, ties-to-even rounding,
minimum scale, blockwise scale storage conversion, and training dtype/gradient
behavior used by the Gemma 4 E2B and Gemma 3 270M QAT profiles.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.qat.fake_quant import (  # noqa: E402
    AI_EDGE_MIN_SCALE,
    _round_ai_edge_blockwise_scale,
    _scale_and_zero_point,
    fake_quantize_ste,
)
from ir_training.qat.toolchain import edge_export_toolchain_report  # noqa: E402


class AIEdgeQATNumericContractError(RuntimeError):
    """Raised when the public reference package cannot be checked."""


def _ours_codes(values: Any, scale: Any, zero_point: Any, qmin: int, qmax: int) -> np.ndarray:
    import torch

    return (
        torch.round(values.float() / scale + zero_point)
        .clamp(qmin, qmax)
        .to(torch.int8)
        .cpu()
        .numpy()
    )


def run(*, allow_version_drift: bool = False) -> dict[str, Any]:
    try:
        import torch
        from ai_edge_quantizer import qtyping
        from ai_edge_quantizer.algorithms.uniform_quantize import (
            uniform_quantize_tensor as public_quantizer,
        )
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise AIEdgeQATNumericContractError(
            "Install training/requirements-edge-export-tested.txt before "
            "running the public QAT numeric-contract check."
        ) from exc

    toolchain = edge_export_toolchain_report()
    source = torch.tensor(
        [
            [-0.76953125, -0.310546875, -0.109375, 0.109375, 0.73046875],
            [-0.421875, -0.203125, 0.015625, 0.28125, 0.578125],
        ],
        dtype=torch.bfloat16,
    )
    channelwise: list[dict[str, Any]] = []
    for bits in (2, 4, 8):
        scale, zero_point, qmin, qmax = _scale_and_zero_point(
            source,
            bits=bits,
            symmetric=True,
            reduce_dims=(1,),
            eps=AI_EDGE_MIN_SCALE,
            quantizer="ste_ai_edge",
        )
        source_np = source.float().cpu().numpy()
        public_zp, public_scale = public_quantizer.tensor_zp_scale_from_min_max(
            source_np.min(axis=1),
            source_np.max(axis=1),
            bits,
            True,
            qtyping.QuantGranularity.CHANNELWISE,
        )
        public_params = qtyping.UniformQuantParams(
            num_bits=bits,
            scale=public_scale,
            zero_point=public_zp,
            symmetric=True,
            quantized_dimension=0,
        )
        public_codes = public_quantizer.uniform_quantize(source_np, public_params)
        ours_codes = _ours_codes(source, scale, zero_point, qmin, qmax)
        scale_match = np.array_equal(
            scale.squeeze(1).cpu().numpy(), public_scale
        )
        codes_match = np.array_equal(ours_codes, public_codes)
        channelwise.append(
            {
                "bits": bits,
                "qmin": qmin,
                "qmax": qmax,
                "scale_dtype": str(scale.dtype).removeprefix("torch."),
                "scale_exact": bool(scale_match),
                "codes_exact": bool(codes_match),
                "pass": bool(scale_match and codes_match),
            }
        )

    block_source = torch.linspace(-0.8, 0.7, 512, dtype=torch.float32).reshape(
        2, 256
    ).to(torch.bfloat16)
    grouped = block_source.reshape(2, 1, 256)
    block_scale, block_zp, block_qmin, block_qmax = _scale_and_zero_point(
        grouped,
        bits=4,
        symmetric=True,
        reduce_dims=(2,),
        eps=AI_EDGE_MIN_SCALE,
        quantizer="ste_ai_edge",
    )
    block_scale = _round_ai_edge_blockwise_scale(block_scale)
    block_np = block_source.float().cpu().numpy()
    block_np_grouped = block_np.reshape(2, 1, 256)
    public_block_zp, public_block_scale = (
        public_quantizer.tensor_zp_scale_from_min_max(
            block_np_grouped.min(axis=2),
            block_np_grouped.max(axis=2),
            4,
            True,
            qtyping.QuantGranularity.BLOCKWISE_256,
        )
    )
    public_block_params = qtyping.UniformQuantParams(
        num_bits=4,
        scale=public_block_scale,
        zero_point=public_block_zp,
        symmetric=True,
        quantized_dimension=1,
        block_size=256,
    )
    public_block_codes = public_quantizer.uniform_quantize(
        block_np, public_block_params, is_blockwise_quant=True
    )
    ours_block_codes = _ours_codes(
        grouped, block_scale, block_zp, block_qmin, block_qmax
    ).reshape(2, 256)
    block_scale_match = np.array_equal(
        block_scale.squeeze(2).cpu().numpy(), public_block_scale
    )
    block_codes_match = np.array_equal(ours_block_codes, public_block_codes)

    tiny = torch.zeros((1, 4), dtype=torch.bfloat16)
    tiny_scale, _, _, _ = _scale_and_zero_point(
        tiny,
        bits=8,
        symmetric=True,
        reduce_dims=(1,),
        eps=AI_EDGE_MIN_SCALE,
        quantizer="ste_ai_edge",
    )
    minimum_scale_match = bool(
        np.float32(tiny_scale.item()).tobytes()
        == np.float32(AI_EDGE_MIN_SCALE).tobytes()
    )

    differentiable = source.detach().clone().requires_grad_(True)
    simulated = fake_quantize_ste(
        differentiable,
        bits=8,
        symmetric=True,
        per_channel=True,
        axis=0,
        eps=AI_EDGE_MIN_SCALE,
        quantizer="ste_ai_edge",
    )
    simulated.float().sum().backward()
    dtype_preserved = simulated.dtype == source.dtype
    ste_identity = bool(torch.all(differentiable.grad == 1).item())

    numerical_checks_pass = bool(
        all(item["pass"] for item in channelwise)
        and block_scale_match
        and block_codes_match
        and minimum_scale_match
        and dtype_preserved
        and ste_identity
    )
    version_gate_pass = bool(
        toolchain["tested_versions_match"] or allow_version_drift
    )
    return {
        "schema_version": 1,
        "training_executed": False,
        "model_loaded": False,
        "public_reference_executed": True,
        "private_google_qat_recipe_recovered": False,
        "toolchain": toolchain,
        "allow_version_drift": bool(allow_version_drift),
        "channelwise": channelwise,
        "blockwise_w4_g256": {
            "scale_exact": bool(block_scale_match),
            "codes_exact": bool(block_codes_match),
            "pass": bool(block_scale_match and block_codes_match),
        },
        "minimum_scale_exact": minimum_scale_match,
        "training_dtype_preserved": dtype_preserved,
        "ste_gradient_identity": ste_identity,
        "numerical_checks_pass": numerical_checks_pass,
        "version_gate_pass": version_gate_pass,
        "overall_pass": bool(numerical_checks_pass and version_gate_pass),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="Report numerical parity without requiring the exact tested package lock.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run(allow_version_drift=args.allow_version_drift)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
