"""Disposable ZeRO-2 Trainer gate; never substitute a DDP memory receipt.

The partition adapter below is deliberately pinned to DeepSpeed 0.19.7. Reading
its existing fragment views avoids gathering a multi-gigabyte embedding gradient
merely to check finiteness. Every rank contributes interval evidence, and the
union must cover every unique parameter exactly once before either update.
"""
from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import replace
from typing import Any

from ir_training.common.progress import log
from ir_training.train.full_parameters import FullParameterScopeError
from ir_training.train.sharded_contract import (
    assert_sharded_engine,
    build_deepspeed_config,
    deepspeed_config_sha256,
)
from ir_training.train.tensor_checks import tensor_all_finite, tensor_finite_and_nonzero

PROBE_KIND = "disposable_full_parameter_zero2_trainer_v1"
STEPS = 2


def validate_partition_coverage(shapes: dict[str, int], rank_intervals: list[dict]) -> None:
    """Prove one and only one owner for every logical gradient element."""
    unknown = set().union(*(set(item) for item in rank_intervals)) - set(shapes)
    if unknown:
        raise FullParameterScopeError(f"Unknown ZeRO gradient parameters: {sorted(unknown)}")
    for name, numel in shapes.items():
        intervals = [item[name] for item in rank_intervals if name in item]
        if any(not isinstance(item, (list, tuple)) or len(item) != 2
               or any(type(value) is not int for value in item) for item in intervals):
            raise FullParameterScopeError(f"Malformed gradient partition: {name}")
        cursor = 0
        for start, length in sorted(intervals):
            if start != cursor or length <= 0 or start + length > numel:
                raise FullParameterScopeError(f"Overlapping/missing gradient partition: {name}")
            cursor += length
        if cursor != numel:
            raise FullParameterScopeError(f"Incomplete gradient partition coverage: {name}")


def local_gradient_intervals(parameters: list[tuple[str, Any]]) -> tuple[dict, bool]:
    """Inspect only local ZeRO-2 fragment views, without full-gradient assembly."""
    intervals: dict = {}
    any_nonzero = False
    for name, parameter in parameters:
        if not hasattr(parameter, "_hp_mapping"):
            raise FullParameterScopeError(f"Missing DeepSpeed 0.19.7 fragment mapping: {name}")
        mapping = parameter._hp_mapping
        if mapping is None:  # This rank owns no elements of this parameter.
            continue
        address = mapping.lp_fragment_address
        start, count = address.start, address.numel
        if (type(start) is not int or type(count) is not int or start < 0
                or count <= 0 or start + count > parameter.numel()):
            raise FullParameterScopeError(f"Invalid ZeRO fragment address: {name}")
        gradient = mapping.get_lp_grad_fragment(parameter._index_in_param_group)
        if gradient is None or gradient.numel() != count:
            raise FullParameterScopeError(f"Missing or mismatched local gradient fragment: {name}")
        finite, nonzero = tensor_finite_and_nonzero(gradient)
        if not finite:
            raise FullParameterScopeError(f"Non-finite local gradient: {name}")
        if str(gradient.dtype) != "torch.float32":
            raise FullParameterScopeError(f"Gradient accumulation must remain FP32: {name}")
        intervals[name] = [start, count]
        any_nonzero |= nonzero
    return intervals, any_nonzero


def _collective_gradient_audit(parameters: list[tuple[str, Any]], world_size: int) -> dict:
    import torch.distributed as dist

    # Exchange error data as well, so an invalid fragment on one rank does not
    # leave peers entering the update while that rank exits validation.
    result: dict = {}
    try:
        intervals, nonzero = local_gradient_intervals(parameters)
        result.update(intervals=intervals, nonzero=nonzero)
    except (ValueError, RuntimeError, AttributeError, KeyError, IndexError) as exc:
        result["error"] = repr(exc)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, result)
    errors = [item["error"] for item in gathered if "error" in item]
    if errors:
        raise FullParameterScopeError("ZeRO gradient audit failed: " + "; ".join(errors))
    shapes = {name: parameter.numel() for name, parameter in parameters}
    validate_partition_coverage(shapes, [item["intervals"] for item in gathered])
    if not any(item["nonzero"] for item in gathered):
        raise FullParameterScopeError("Sharded preflight has no nonzero gradients")
    return {"verified": True, "unique_parameters": len(shapes), "global_numel": sum(shapes.values()),
            "ranks": world_size, "method": "exhaustive_local_fragments_exact_global_interval_union"}


def audit_adamw_state(zero_optimizer: Any, *, require_state: bool) -> dict:
    """Check local FP32 master/moment partitions, not replicated model grads."""
    import torch

    raw = getattr(zero_optimizer, "optimizer", None)
    if not isinstance(raw, torch.optim.AdamW):
        raise FullParameterScopeError("Sharded Trainer requires torch AdamW")
    count = state_numel = 0
    for group in raw.param_groups:
        if (group.get("betas") != (0.9, 0.999) or group.get("eps") != 1e-8
                or group.get("weight_decay") != 0.0):
            raise FullParameterScopeError("Sharded AdamW recipe changed")
        for parameter in group["params"]:
            if parameter.dtype != torch.float32 or not tensor_all_finite(parameter):
                raise FullParameterScopeError("Invalid sharded FP32 master partition")
            state = raw.state.get(parameter, {})
            if not state and not require_state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if (not torch.is_tensor(value) or value.shape != parameter.shape
                        or value.dtype != torch.float32 or not tensor_all_finite(value)):
                    raise FullParameterScopeError(f"Missing/invalid AdamW state partition: {key}")
                count += 1
                state_numel += value.numel()
            step = state.get("step")
            if step is None or not math.isfinite(float(step)) or float(step) < 1:
                raise FullParameterScopeError("Sharded AdamW did not perform an update")
    if require_state and (count == 0 or state_numel == 0):
        raise FullParameterScopeError("Sharded optimizer state was not materialized")
    return {"name": "AdamW", "betas": [0.9, 0.999], "epsilon": 1e-8,
            "weight_decay": 0.0, "external_max_grad_norm_required": 0.0,
            "state_tensor_count": count, "state_numel": state_numel,
            "state_scope": "rank_local_fp32_partitions"}


def validate_sharded_probe(probe: dict, *, accumulation_steps: int, world_size: int,
                           local_rank: int | None = None) -> None:
    """Strict persisted evidence, distinct from the existing DDP v2 schema."""
    def positive(value: Any) -> bool:
        return type(value) is int and value > 0

    def finite(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value)

    if local_rank is not None and (type(local_rank) is not int or local_rank < 0):
        raise FullParameterScopeError("Sharded preflight local rank must be a nonnegative integer")
    if (not positive(accumulation_steps) or not positive(world_size) or world_size < 2
            or not isinstance(probe, dict) or type(probe.get("schema_version")) is not int
            or probe.get("schema_version") != 3
            or probe.get("probe") != PROBE_KIND or probe.get("passed") is not True
            or probe.get("disposable_worker_required") is not True
            or probe.get("model_must_not_be_reused") is not True
            or type(probe.get("checkpoint_writes")) is not int or probe["checkpoint_writes"] != 0
            or not positive(probe.get("disposable_optimizer_steps"))
            or probe.get("disposable_optimizer_steps") != STEPS):
        raise FullParameterScopeError("Missing successful disposable sharded Trainer probe")
    for name in ("sharded_probe", "scope", "optimizer", "memory", "selection"):
        if not isinstance(probe.get(name), dict):
            raise FullParameterScopeError(f"Sharded preflight {name} must be an object")
    shard = probe.get("sharded_probe") or {}
    config = shard.get("config") or {}
    if (not isinstance(config, dict) or not isinstance(shard.get("gradient_coverage"), dict)
            or any(not positive(shard.get(key)) for key in (
                "zero_stage", "world_size", "gradient_accumulation_steps", "microsteps", "optimizer_steps"))
            or shard.get("backend") != "sharded" or shard.get("zero_stage") != 2
            or shard.get("world_size") != world_size
            or shard.get("gradient_accumulation_steps") != accumulation_steps
            or shard.get("microsteps") != STEPS * accumulation_steps
            or shard.get("optimizer_steps") != STEPS
            or shard.get("nccl_collectives_certified") is not True
            or shard.get("model_parameters_replicated") is not True
            or shard.get("config_sha256") != deepspeed_config_sha256(config)):
        raise FullParameterScopeError("Incomplete sharded backend/accumulation evidence")
    scope = probe.get("scope") or {}
    coverage = shard.get("gradient_coverage") or {}
    if (any(scope.get(key) is not True for key in (
            "all_trainable_fp32", "all_gradients_finite", "all_parameters_finite_after_step"))
            or not positive(scope.get("unique_parameter_count")) or not positive(scope.get("trainable_numel"))
            or any(not positive(coverage.get(key)) for key in ("ranks", "global_numel", "unique_parameters"))
            or coverage.get("verified") is not True or coverage.get("ranks") != world_size
            or coverage.get("global_numel") != scope.get("trainable_numel")
            or coverage.get("unique_parameters") != scope.get("unique_parameter_count")
            or coverage.get("method") != "exhaustive_local_fragments_exact_global_interval_union"):
        raise FullParameterScopeError("Sharded probe did not verify all parameter/gradient partitions")
    optimizer = probe.get("optimizer") or {}
    if (any(not finite(optimizer.get(key)) for key in (
            "epsilon", "weight_decay", "external_max_grad_norm_required", "learning_rate"))
            or optimizer.get("name") != "AdamW" or optimizer.get("betas") != [0.9, 0.999]
            or optimizer.get("epsilon") != 1e-8 or optimizer.get("weight_decay") != 0.0
            or optimizer.get("external_max_grad_norm_required") != 0.0
            or optimizer.get("state_scope") != "rank_local_fp32_partitions"
            or not positive(optimizer.get("state_tensor_count")) or not positive(optimizer.get("state_numel"))
            or not finite(optimizer.get("learning_rate")) or optimizer["learning_rate"] <= 0):
        raise FullParameterScopeError("Incomplete sharded AdamW evidence")
    memory = probe.get("memory") or {}
    if (memory.get("cuda_local_rank_only") is not True
            or memory.get("nccl_collectives_certified") is not True
            or not isinstance(memory.get("device"), str)
            or re.fullmatch(r"cuda:[0-9]+", memory["device"]) is None
            or (local_rank is not None and memory["device"] != f"cuda:{local_rank}")):
        raise FullParameterScopeError("Missing rank-local sharded CUDA memory evidence")
    keys = ("baseline_allocated_bytes", "baseline_reserved_bytes", "peak_allocated_bytes",
            "peak_reserved_bytes", "device_total_bytes", "device_free_bytes_min")
    if any(type(memory.get(key)) is not int or memory[key] < 0 for key in keys):
        raise FullParameterScopeError("Invalid sharded memory counters")
    allocated, reserved = memory["peak_allocated_bytes"], memory["peak_reserved_bytes"]
    total, free = memory["device_total_bytes"], memory["device_free_bytes_min"]
    fraction = memory.get("peak_reserved_fraction")
    if (total <= 0 or allocated < memory["baseline_allocated_bytes"]
            or reserved < memory["baseline_reserved_bytes"] or reserved < allocated
            or memory["baseline_reserved_bytes"] < memory["baseline_allocated_bytes"]
            or reserved > total or free > total or not finite(fraction)
            or memory.get("max_reserved_fraction") != 0.90
            or not math.isclose(fraction, reserved / total, rel_tol=1e-12, abs_tol=1e-12)
            or reserved / total >= 0.90 or free / total <= 0.10):
        raise FullParameterScopeError("Sharded CUDA memory/headroom gate failed")
    selection = probe.get("selection") or {}
    if (not positive(selection.get("longest_sequence_length"))
            or selection.get("repeated_real_prepared_row") is not True):
        raise FullParameterScopeError("Sharded preflight did not test the longest prepared row")


def run_sharded_trainer_preflight(checked_trainer_cls: Any, trainer_kwargs: dict, *,
                                  training_cfg: dict, longest_row: dict) -> dict:
    """Run two real Trainer/ZeRO-2 updates in disposable CUDA workers only."""
    import torch
    import torch.distributed as dist

    from ir_training.train.full_trainer_preflight import _RepeatedLongestRow

    args = trainer_kwargs["args"]
    parameters = list(trainer_kwargs["model"].named_parameters())
    world = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation = args.gradient_accumulation_steps
    if world < 2 or not parameters or args.per_device_train_batch_size != 1:
        raise FullParameterScopeError("Sharded preflight requires multiple CUDA ranks and microbatch 1")
    device = parameters[0][1].device
    if device.type != "cuda" or any(p.dtype != torch.float32 or not p.requires_grad
            or p.device != device or p.grad is not None for _, p in parameters):
        raise FullParameterScopeError("Sharded preflight requires clean trainable rank-local FP32 parameters")
    ds_config = build_deepspeed_config(training_cfg, world)
    if args.deepspeed != ds_config or args.bf16 or args.fp16 or args.max_grad_norm != 0.0:
        raise FullParameterScopeError("Sharded preflight precision/config differs from production")
    probe_args = replace(args, deepspeed=copy.deepcopy(ds_config),
                         accelerator_config=copy.deepcopy(args.accelerator_config),
                         max_steps=STEPS, save_strategy="no", eval_strategy="no",
                         logging_strategy="no", report_to=[], push_to_hub=False,
                         load_best_model_at_end=False, disable_tqdm=True, skip_memory_metrics=True)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    minimum_free = total
    microsteps = completed_steps = 0
    optimizer_evidence: dict = {}
    coverage: dict = {}

    def sample_memory():
        nonlocal minimum_free
        torch.cuda.synchronize(device)
        free, _ = torch.cuda.mem_get_info(device)
        minimum_free = min(minimum_free, free)

    class ProbeTrainer(checked_trainer_cls):
        def training_step(self, engine, inputs, *step_args, **step_kwargs):
            nonlocal microsteps, completed_steps, optimizer_evidence, coverage
            assert_sharded_engine(engine, training_cfg, world)
            if not dist.is_initialized() or dist.get_world_size() != world or dist.get_backend() != "nccl":
                raise FullParameterScopeError("Sharded preflight requires the actual NCCL process group")
            zero = engine.optimizer
            if not getattr(self, "_a2ui_zero_probe_installed", False):
                # Accelerate performs engine.step INSIDE backward, before the
                # HF on_pre_optimizer_step callback. Inspect at the actual ZeRO
                # step boundary, not after gradients have already been cleared.
                original_step = zero.step

                def audited_step(*args, **kwargs):
                    nonlocal completed_steps, optimizer_evidence, coverage
                    if microsteps + 1 != (completed_steps + 1) * accumulation:
                        raise FullParameterScopeError("ZeRO optimizer updated before a full accumulation window")
                    coverage = _collective_gradient_audit(parameters, world)
                    audit_adamw_state(zero, require_state=completed_steps > 0)
                    sample_memory()
                    result = original_step(*args, **kwargs)
                    if getattr(zero, "overflow", False):
                        raise FullParameterScopeError("ZeRO optimizer skipped a non-finite update")
                    optimizer_evidence = audit_adamw_state(zero, require_state=True)
                    for name, parameter in parameters:
                        if not tensor_all_finite(parameter):
                            raise FullParameterScopeError(f"Non-finite updated parameter: {name}")
                    completed_steps += 1
                    sample_memory()
                    log(f"Sharded Trainer probe rank={dist.get_rank()}: update {completed_steps}/{STEPS}; disposable")
                    return result

                zero.step = audited_step
                self._a2ui_zero_probe_installed = True
            loss = super().training_step(engine, inputs, *step_args, **step_kwargs)
            if not tensor_all_finite(loss):
                raise FullParameterScopeError("Sharded preflight produced non-finite loss")
            microsteps += 1
            sample_memory()
            return loss

        def save_model(self, *args, **kwargs):
            raise FullParameterScopeError("Disposable sharded preflight cannot save a model")

        def _save_checkpoint(self, *args, **kwargs):
            raise FullParameterScopeError("Disposable sharded preflight cannot save a checkpoint")

    trainer = ProbeTrainer(**{**trainer_kwargs, "args": probe_args, "eval_dataset": None,
                             "callbacks": [], "train_dataset": _RepeatedLongestRow(
                                 longest_row, STEPS * accumulation * world)})
    trainer.train()
    if completed_steps != STEPS or microsteps != STEPS * accumulation or trainer.state.global_step != STEPS:
        raise FullParameterScopeError("Sharded probe did not complete both full accumulation windows")
    sample_memory()
    peak_reserved = torch.cuda.max_memory_reserved(device)
    optimizer_evidence["learning_rate"] = float(args.learning_rate)
    report = {
        "schema_version": 3, "probe": PROBE_KIND, "passed": True,
        "disposable_worker_required": True, "model_must_not_be_reused": True,
        "checkpoint_writes": 0, "disposable_optimizer_steps": completed_steps,
        "optimizer": optimizer_evidence,
        "scope": {"unique_parameter_count": len(parameters), "trainable_numel": sum(p.numel() for _, p in parameters),
                  "all_trainable_fp32": True, "all_gradients_finite": True, "all_parameters_finite_after_step": True},
        "sharded_probe": {"backend": "sharded", "zero_stage": 2, "world_size": world,
                          "gradient_accumulation_steps": accumulation, "microsteps": microsteps,
                          "optimizer_steps": completed_steps, "nccl_collectives_certified": True,
                          "model_parameters_replicated": True, "gradient_coverage": coverage,
                          "config": ds_config, "config_sha256": deepspeed_config_sha256(ds_config)},
        "memory": {"device": str(device), "cuda_local_rank_only": True, "nccl_collectives_certified": True,
                   "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": peak_reserved,
                   "device_total_bytes": total, "device_free_bytes_min": minimum_free,
                   "peak_reserved_fraction": peak_reserved / total, "max_reserved_fraction": 0.90},
        "selection": {"longest_sequence_length": len(longest_row["input_ids"]), "repeated_real_prepared_row": True},
    }
    validate_sharded_probe(report, accumulation_steps=accumulation, world_size=world,
                           local_rank=int(os.environ["LOCAL_RANK"]))
    return report
