"""Separate generic full-finetuning parameter storage from compute precision.

The retained-mobile LoRA and dedicated all-parameter QAT workflows retain their
own numerical policies. Generic full SFT/QAT keeps FP32 optimizer parameters,
including when reloading a checkpoint, and uses the requested autocast policy
for both Trainer and direct evaluation/preflight calls.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

FULL_FINETUNE_PRECISION_POLICY = "fp32_parameters_v1"
_COMPUTE_DTYPE_ATTRIBUTE = "_a2ui_full_finetune_compute_dtype"


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except (ImportError, OSError, RuntimeError, AttributeError):
        return False


def _cuda_bf16_supported() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except (ImportError, OSError, RuntimeError, AttributeError):
        return False


def resolve_training_dtype(
    requested_dtype: str, *, cuda_available: Callable[[], bool] | None = None,
    bf16_supported: Callable[[], bool] | None = None,
) -> str:
    """Use the same hardware fallback for training and standalone evaluation."""
    cuda_available = cuda_available or _cuda_available
    bf16_supported = bf16_supported or _cuda_bf16_supported
    dtype = (requested_dtype or "bfloat16").strip().lower()
    if dtype in {"bf16", "bfloat16"}:
        if bf16_supported():
            print("Training precision: bfloat16", flush=True)
            return "bfloat16"
        if cuda_available():
            print("Training precision fallback: bfloat16 is unsupported; using float16.", flush=True)
            return "float16"
        print("Training precision fallback: CUDA is unavailable; using float32.", flush=True)
        return "float32"
    if dtype in {"fp16", "float16", "half"}:
        if cuda_available():
            print("Training precision: float16", flush=True)
            return "float16"
        print("Training precision fallback: CUDA is unavailable; using float32.", flush=True)
        return "float32"
    if dtype in {"fp32", "float32", "full"}:
        print("Training precision: float32", flush=True)
        return "float32"
    print(f"Training precision: unknown dtype {requested_dtype!r}; using float32.", flush=True)
    return "float32"


def training_precision_flags(
    dtype_name: str, training_cfg: dict[str, Any] | None = None,
) -> dict[str, bool]:
    cfg = training_cfg if isinstance(training_cfg, dict) else {}
    precision_mode = str(cfg.get("mixed_precision", "auto")).strip().lower()
    if precision_mode in {"none", "off", "false", "disabled", "no"}:
        return {"bf16": False, "fp16": False}
    if precision_mode in {"bf16", "bfloat16"}:
        return {"bf16": True, "fp16": False}
    if precision_mode in {"fp16", "float16", "half"}:
        return {"bf16": False, "fp16": True}
    dtype = dtype_name.strip().lower()
    return {"bf16": dtype in {"bf16", "bfloat16"},
            "fp16": dtype in {"fp16", "float16", "half"}}


def full_finetune_precision_policy(
    config: dict[str, Any], *, resolved_dtype: str | None = None,
) -> dict[str, Any] | None:
    from ir_training.qat.full_model_contract import is_full_qat
    from ir_training.train.recipe import FULL_METHODS

    training = config.get("training") or {}
    method = str(training.get("method", "lora_sft")).strip().lower()
    if method not in FULL_METHODS or is_full_qat(config):
        return None
    dtype = resolved_dtype or resolve_training_dtype(
        str((config.get("model") or {}).get("dtype", "bfloat16"))
    )
    flags = training_precision_flags(dtype, training)
    compute_dtype = "bfloat16" if flags["bf16"] else "float16" if flags["fp16"] else "float32"
    return {
        "policy": FULL_FINETUNE_PRECISION_POLICY,
        "parameter_dtype": "float32",
        "compute_dtype": compute_dtype,
    }


def full_finetune_load_config(
    model_config: dict[str, Any], policy: dict[str, Any] | None,
) -> dict[str, Any]:
    config = dict(model_config)
    if policy is not None:
        # Loading a saved FP32 checkpoint through BF16 before promotion would
        # discard the residual updates this policy exists to preserve.
        config["dtype"] = "float32"
    return config


def bind_full_finetune_precision(model: Any, policy: dict[str, Any] | None) -> None:
    if policy is None:
        return
    import torch

    invalid = [name for name, parameter in model.named_parameters()
               if parameter.dtype != torch.float32]
    if invalid:
        raise ValueError(
            "Full-finetuning requires FP32 parameters loaded directly from the checkpoint; "
            f"invalid parameters: {invalid[:12]}. Reload with dtype=float32, do not "
            "round through a low-precision model first."
        )
    setattr(model, _COMPUTE_DTYPE_ATTRIBUTE, policy["compute_dtype"])


def full_finetune_autocast(model: Any):
    """Return None for unrelated workflows, otherwise the configured context."""
    dtype_name = getattr(model, _COMPUTE_DTYPE_ATTRIBUTE, None)
    if dtype_name is None:
        return None
    if dtype_name == "float32":
        return nullcontext()
    import torch

    device = next(model.parameters()).device
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"Full-finetuning autocast is unsupported on {device.type}.")
    if dtype_name == "float16" and device.type == "cpu":
        raise ValueError(
            "Full-finetuning FP16 compute requires CUDA; use FP32 or explicit CPU BF16."
        )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return torch.autocast(device.type, dtype=dtype)
