"""Bounded DeepSpeed 0.19.7 ZeRO-3 partition inspection.

This module never gathers a logical parameter.  It validates the local shard
which DeepSpeed already owns and describes only the unpadded logical interval.
"""
from __future__ import annotations

import math
from typing import Any

from ir_training.train.full_parameters import FullParameterScopeError
from ir_training.train.tensor_checks import tensor_all_finite, tensor_finite_and_nonzero


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FullParameterScopeError(f"Invalid ZeRO-3 {label}: {value!r}")
    return value


def logical_numel(parameter: Any, name: str) -> int:
    """Return DeepSpeed's pre-partition logical size, never placeholder numel."""
    return _integer(getattr(parameter, "ds_numel", None), f"logical numel for {name}", minimum=1)


def local_parameter_partition(parameter: Any, name: str, *, rank: int,
                              world_size: int) -> tuple[Any, int, int]:
    """Return (local shard, logical start, unpadded count)."""
    total = logical_numel(parameter, name)
    rank = _integer(rank, "rank")
    world_size = _integer(world_size, "world size", minimum=1)
    if rank >= world_size:
        raise FullParameterScopeError(f"Invalid ZeRO-3 rank {rank}/{world_size}")
    shard = getattr(parameter, "ds_tensor", None)
    if shard is None or not hasattr(shard, "numel") or shard.numel() <= 0:
        raise FullParameterScopeError(f"Missing/empty ZeRO-3 parameter partition: {name}")
    if str(getattr(shard, "dtype", None)) != "torch.float32":
        raise FullParameterScopeError(f"ZeRO-3 parameter partition is not FP32: {name}")
    partition_size = math.ceil(total / world_size)
    if shard.numel() != partition_size:
        raise FullParameterScopeError(
            f"Mismatched ZeRO-3 parameter partition: {name}; "
            f"expected {partition_size}, found {shard.numel()}"
        )
    start = rank * partition_size
    count = max(0, min(partition_size, total - start))
    return shard, start, count


def local_gradient_intervals(parameters: list[tuple[str, Any]], *, rank: int,
                             world_size: int) -> tuple[dict, bool]:
    """Inspect ``safe_get_local_grad`` without assembling full gradients."""
    try:
        from deepspeed.utils import safe_get_local_grad
    except Exception as exc:  # pragma: no cover - pinned runtime gate handles this
        raise FullParameterScopeError("DeepSpeed safe_get_local_grad is unavailable") from exc

    intervals: dict[str, list[int]] = {}
    any_nonzero = False
    for name, parameter in parameters:
        shard, start, count = local_parameter_partition(
            parameter, name, rank=rank, world_size=world_size
        )
        gradient = safe_get_local_grad(parameter)
        if gradient is None or not hasattr(gradient, "numel") or gradient.numel() != shard.numel():
            raise FullParameterScopeError(f"Missing/mismatched ZeRO-3 local gradient: {name}")
        # The final rank may contain alignment padding. Padding is not logical
        # coverage and is deliberately excluded from numerical evidence.
        flat_gradient = gradient.reshape(-1)
        if str(gradient.dtype) != "torch.float32":
            raise FullParameterScopeError(f"ZeRO-3 gradient accumulation must remain FP32: {name}")
        if not tensor_all_finite(flat_gradient):
            raise FullParameterScopeError(f"Non-finite ZeRO-3 local gradient: {name}")
        padding = flat_gradient[count:]
        if padding.numel() and bool(padding.count_nonzero().item()):
            raise FullParameterScopeError(f"Nonzero ZeRO-3 gradient padding: {name}")
        logical_gradient = flat_gradient[:count]
        if count:
            finite, nonzero = tensor_finite_and_nonzero(logical_gradient)
            if not finite:
                raise FullParameterScopeError(f"Non-finite ZeRO-3 local gradient: {name}")
            intervals[name] = [start, count]
            any_nonzero |= nonzero
    return intervals, any_nonzero


def audit_adamw_state(zero_optimizer: Any, *, require_state: bool) -> dict:
    """Validate ZeRO-3 FP32 flat masters and their exact AdamW state tensors."""
    import torch

    raw = getattr(zero_optimizer, "optimizer", None)
    if not isinstance(raw, torch.optim.AdamW):
        raise FullParameterScopeError("ZeRO-3 Trainer requires torch AdamW")
    groups = getattr(raw, "param_groups", None)
    masters = getattr(zero_optimizer, "fp32_partitioned_groups_flat", None)
    if not isinstance(groups, list) or not groups or not isinstance(masters, list) or not masters:
        raise FullParameterScopeError("ZeRO-3 FP32 optimizer partitions are unavailable")
    raw_masters: list[Any] = []
    for group in groups:
        if (group.get("betas") != (0.9, 0.999) or group.get("eps") != 1e-8
                or group.get("weight_decay") != 0.0):
            raise FullParameterScopeError("ZeRO-3 AdamW recipe changed")
        raw_masters.extend(group.get("params") or [])
    # DeepSpeed may temporarily empty raw group params around its update. When
    # present, however, they must be the exact flat masters whose state we read.
    if raw_masters and ([id(value) for value in raw_masters] != [id(value) for value in masters]):
        raise FullParameterScopeError("ZeRO-3 optimizer group/master partition mismatch")
    count = state_numel = master_numel = 0
    for master in masters:
        if (not torch.is_tensor(master) or master.numel() <= 0 or master.dtype != torch.float32
                or not tensor_all_finite(master)):
            raise FullParameterScopeError("Invalid ZeRO-3 FP32 master partition")
        master_numel += master.numel()
        state = raw.state.get(master, {})
        if not state and not require_state:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if (not torch.is_tensor(value) or value.shape != master.shape
                    or value.dtype != torch.float32 or not tensor_all_finite(value)):
                raise FullParameterScopeError(f"Missing/invalid ZeRO-3 AdamW state partition: {key}")
            count += 1
            state_numel += value.numel()
        step = state.get("step")
        minimum_step = 1 if require_state else 0
        if (step is None or not math.isfinite(float(step))
                or float(step) < minimum_step):
            message = ("ZeRO-3 AdamW did not perform an update" if require_state
                       else "Invalid initialized ZeRO-3 AdamW step")
            raise FullParameterScopeError(message)
    if require_state and (count == 0 or state_numel == 0):
        raise FullParameterScopeError("ZeRO-3 optimizer state was not materialized")
    return {"name": "AdamW", "betas": [0.9, 0.999], "epsilon": 1e-8,
            "weight_decay": 0.0, "external_max_grad_norm_required": 0.0,
            "state_tensor_count": count, "state_numel": state_numel,
            "master_partition_numel": master_numel,
            "state_scope": "rank_local_zero3_fp32_flat_partitions"}


def parameters_finite(parameters: list[tuple[str, Any]], *, rank: int,
                      world_size: int) -> None:
    for name, parameter in parameters:
        shard, _start, count = local_parameter_partition(
            parameter, name, rank=rank, world_size=world_size
        )
        if count and not tensor_all_finite(shard.reshape(-1)[:count]):
            raise FullParameterScopeError(f"Non-finite updated ZeRO-3 parameter partition: {name}")


def summarize_partitions(parameters: list[tuple[str, Any]], *, rank: int,
                         world_size: int) -> dict:
    """Small truthful diagnostic; it never claims a ZeRO-2 fragment mapping."""
    logical = local = owned = 0
    for name, parameter in parameters:
        shard, _start, count = local_parameter_partition(
            parameter, name, rank=rank, world_size=world_size
        )
        logical += logical_numel(parameter, name)
        local += shard.numel()
        owned += count
    return {"layout": "zero3_ds_tensor_equal_partitions", "parameter_count": len(parameters),
            "global_logical_numel": logical, "local_partition_numel_with_padding": local,
            "local_logical_numel": owned}
