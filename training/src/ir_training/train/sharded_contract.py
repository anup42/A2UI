"""Explicit opt-in contract for full-parameter DeepSpeed ZeRO-2 training.

The default remains ordinary replicated DDP.  This module deliberately builds
an inline DeepSpeed configuration instead of accepting an arbitrary file: the
small surface is reviewable, hash-bound, and cannot silently enable parameter
sharding or offload.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Mapping
from enum import Enum
from typing import Any


class DistributedBackend(str, Enum):
    DDP = "ddp"
    SHARDED = "sharded"


_BUCKET_ELEMENTS = 5_000_000
_SHARDED_OPTIMIZER = "adamw_torch"


def deepspeed_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash a configuration using deterministic, whitespace-free JSON."""

    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_backend(training_cfg: Mapping[str, Any]) -> str:
    """Resolve the narrow distributed selector without importing DeepSpeed."""

    raw = training_cfg.get("distributed_backend", DistributedBackend.DDP.value)
    if type(raw) is not str:  # Reject bools and string-like objects explicitly.
        raise ValueError("training.distributed_backend must be exactly 'ddp' or 'sharded'.")
    try:
        return DistributedBackend(raw).value
    except ValueError as exc:
        raise ValueError(
            "training.distributed_backend must be exactly 'ddp' or 'sharded'."
        ) from exc


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_sharded_config(training_cfg: Mapping[str, Any]) -> None:
    """Validate sharded-only recipe choices before constructing Trainer."""

    if validate_backend(training_cfg) != DistributedBackend.SHARDED.value:
        raise ValueError("DeepSpeed configuration is only valid for distributed_backend='sharded'.")
    _positive_int(
        training_cfg.get("per_device_train_batch_size"),
        "training.per_device_train_batch_size",
    )
    _positive_int(
        training_cfg.get("gradient_accumulation_steps"),
        "training.gradient_accumulation_steps",
    )
    if training_cfg.get("optim") != _SHARDED_OPTIMIZER:
        raise ValueError(
            "Sharded full-parameter training requires explicit optim='adamw_torch'; "
            "it is a distinct optimizer contract from the DDP Adafactor lane."
        )
    forbidden = ("deepspeed", "fsdp", "offload_optimizer", "offload_param")
    configured = [name for name in forbidden if training_cfg.get(name) not in (None, False, "")]
    if configured:
        raise ValueError(
            "Sharded training uses the repository-owned inline ZeRO-2 config; "
            f"remove unsupported settings: {', '.join(configured)}."
        )


def build_deepspeed_config(
    training_cfg: Mapping[str, Any], world_size: int
) -> dict[str, Any]:
    """Build and hash the only supported ZeRO-2 configuration.

    Optimizer and scheduler are intentionally absent. Hugging Face Trainer owns
    ``adamw_torch`` and its scheduler; duplicated DeepSpeed definitions can
    silently change the recipe.
    """

    validate_sharded_config(training_cfg)
    world_size = _positive_int(world_size, "world_size")
    microbatch = int(training_cfg["per_device_train_batch_size"])
    accumulation = int(training_cfg["gradient_accumulation_steps"])
    effective_batch = microbatch * accumulation * world_size
    configured_effective = training_cfg.get("expected_effective_batch_size")
    if configured_effective is not None:
        configured_effective = _positive_int(configured_effective, "training.expected_effective_batch_size")
        if configured_effective != effective_batch:
            raise ValueError("training.expected_effective_batch_size does not match microbatch * gradient_accumulation_steps * world_size.")
    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": microbatch,
        "gradient_accumulation_steps": accumulation,
        "train_batch_size": effective_batch,
        "gradient_clipping": float(training_cfg.get("max_grad_norm", 0.0)),
        "communication_data_type": "fp32",
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "torch_autocast": {
            "enabled": True,
            "dtype": "bfloat16",
            "lower_precision_safe_modules": [],
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "overlap_comm": False,
            "reduce_bucket_size": _BUCKET_ELEMENTS,
            "allgather_bucket_size": _BUCKET_ELEMENTS,
            "allgather_partitions": True,
        },
    }
    return config


def validate_sharded_runtime() -> dict[str, Any]:
    """Require the reviewed Linux/CUDA package set before loading weights."""
    from ir_training.train.sharded_environment import (
        NVTX_REPAIR,
        NVTX_VERSION,
        probe_nvtx_compatibility,
    )

    required = {"deepspeed": "0.19.7", "transformers": "5.16.1", "accelerate": "1.15.0",
                "nvtx": NVTX_VERSION}
    installed: dict[str, str] = {}
    for package, version in required.items():
        repair = f" {NVTX_REPAIR}" if package == "nvtx" else ""
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Sharded training requires {package} {version}; it is not installed.{repair}"
            ) from exc
        if installed[package] != version:
            raise RuntimeError(
                f"Sharded training requires reviewed {package} {version}; "
                f"found {installed[package]}.{repair}"
            )
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Sharded training requires the reviewed Linux/CUDA runtime.")
    import torch  # Deliberately sharded-only; DDP validation never reaches this import.

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Sharded training requires CUDA with native BF16 support.")
    nvtx_probe = probe_nvtx_compatibility()
    return {"packages": installed, "platform": sys.platform,
            "cuda_available": True, "native_bf16": True, "nvtx": nvtx_probe}


def configure_sharded_accumulation(
    accelerator_args: dict[str, Any], training_cfg: Mapping[str, Any], *,
    trainer_accumulation_steps: int,
) -> None:
    """Align Accelerate's plugin before construction, only for DeepSpeed.

    Pinned Trainer creates num_steps=1 because it normalizes loss itself.
    Accelerate's DEEPSPEED branch does not divide loss a second time, and
    Trainer passes scale_wrt_gas=False to DeepSpeed. Thus matching the plugin
    to the configured accumulation removes the mismatch without changing
    normalization or optimizer boundaries. Never apply this to DDP.
    """
    validate_sharded_config(training_cfg)
    expected = training_cfg["gradient_accumulation_steps"]
    if type(trainer_accumulation_steps) is not int or trainer_accumulation_steps != expected:
        raise ValueError("Sharded Trainer accumulation differs from the configured recipe")
    plugin = accelerator_args.get("gradient_accumulation_plugin")
    if plugin is None or not hasattr(plugin, "num_steps"):
        raise ValueError("Sharded Trainer requires a configurable GradientAccumulationPlugin")
    if accelerator_args.get("gradient_accumulation_steps", 1) != 1:
        raise ValueError("Configure sharded accumulation through its plugin only")
    plugin.num_steps = expected


def assert_sharded_accumulation(
    accelerator: Any, training_cfg: Mapping[str, Any], *,
    trainer_accumulation_steps: int,
) -> dict[str, int]:
    """Reject live Trainer/Accelerate/DeepSpeed divergence, including fallback."""
    validate_sharded_config(training_cfg)
    expected = training_cfg["gradient_accumulation_steps"]
    state = getattr(accelerator, "state", None)
    distributed_type = getattr(accelerator, "distributed_type", None)
    if getattr(distributed_type, "value", distributed_type) != "DEEPSPEED":
        raise ValueError("Sharded accumulation requires the live DEEPSPEED backend")
    ds_plugin = getattr(state, "deepspeed_plugin", None)
    if ds_plugin is None or not callable(getattr(ds_plugin, "get_value", None)):
        raise ValueError("Sharded accumulation requires the live DeepSpeed plugin")
    values = {
        "trainer": trainer_accumulation_steps,
        "accelerate": getattr(accelerator, "gradient_accumulation_steps", None),
        "deepspeed": ds_plugin.get_value("gradient_accumulation_steps"),
    }
    if any(type(value) is not int or value != expected for value in values.values()):
        raise ValueError(f"Sharded accumulation mismatch: expected {expected}, found {values}")
    return values


def _runtime_value(engine: Any, method: str, attribute: str | None = None) -> Any:
    candidate = getattr(engine, method, None)
    if callable(candidate):
        return candidate()
    if candidate is not None:
        return candidate
    if attribute is not None:
        return getattr(engine, attribute, None)
    return candidate


def assert_sharded_engine(
    engine: Any, training_cfg: Mapping[str, Any], world_size: int
) -> dict[str, Any]:
    """Fail closed unless a live engine implements the requested ZeRO-2 lane.

    This uses public duck-typed engine APIs and therefore remains unit-testable
    without importing DeepSpeed. It must be called only for the sharded branch.
    """

    validate_sharded_config(training_cfg)
    expected_world_size = _positive_int(world_size, "world_size")
    stage = _runtime_value(engine, "zero_optimization_stage")
    partitions_gradients = _runtime_value(
        engine, "zero_optimization_partition_gradients"
    )
    world_size = _runtime_value(engine, "dp_world_size", "world_size")
    autocast_enabled = _runtime_value(engine, "torch_autocast_enabled")
    autocast_dtype = _runtime_value(engine, "torch_autocast_dtype")
    module = getattr(engine, "module", None)
    parameters = list(module.parameters()) if module is not None else []
    expected_config = build_deepspeed_config(training_cfg, expected_world_size)
    runtime_config = getattr(engine, "_config", None)
    if runtime_config is None:
        runtime_config = getattr(engine, "config", None)
    if not isinstance(runtime_config, Mapping):
        runtime_config = getattr(runtime_config, "_param_dict", None)
    failures: list[str] = []
    if stage != 2:
        failures.append(f"ZeRO stage is {stage!r}, expected 2")
    if partitions_gradients is not True:
        failures.append("ZeRO gradient partitioning is not active")
    if world_size != expected_world_size:
        failures.append(
            f"data-parallel world size is {world_size!r}, expected {expected_world_size}"
        )
    if autocast_enabled is not True or str(autocast_dtype) not in {
        "torch.bfloat16",
        "bfloat16",
        "bf16",
    }:
        failures.append("DeepSpeed BF16 torch_autocast is not active")
    if not parameters:
        failures.append("engine module exposes no parameters")
    if any(getattr(parameter, "requires_grad", None) is not True for parameter in parameters):
        failures.append("model parameters are not all trainable")
    non_fp32 = [str(getattr(parameter, "dtype", None)) for parameter in parameters
                if str(getattr(parameter, "dtype", None)) not in {"torch.float32", "float32", "fp32"}]
    if non_fp32:
        failures.append(f"model parameters are not all FP32: {non_fp32[:3]}")
    if not isinstance(runtime_config, Mapping):
        failures.append("DeepSpeed engine does not expose its effective configuration")
    else:
        for key in expected_config:
            if runtime_config.get(key) != expected_config[key]:
                failures.append(f"effective DeepSpeed {key} differs from the reviewed contract")
        if "optimizer" in runtime_config or "scheduler" in runtime_config:
            failures.append("effective DeepSpeed config must not own optimizer or scheduler")
    if failures:
        raise RuntimeError("Sharded runtime contract failed: " + "; ".join(failures))
    return {
        "backend": DistributedBackend.SHARDED.value,
        "zero_stage": stage,
        "partition_gradients": True,
        "world_size": world_size,
        "model_parameters_replicated": True,
        "model_parameter_dtype": "float32",
        "compute_autocast_dtype": "bfloat16",
    }
