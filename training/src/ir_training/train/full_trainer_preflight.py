"""Disposable full-QAT memory gate using the production Trainer execution path.

No hand-built DDP wrapper, separate optimizer, or abbreviated backward is used.
The first update exercises startup; the next exercises rebuilt buckets and
resident optimizer state, with the configured accumulation on both updates.
"""
from __future__ import annotations

import copy
import os
from dataclasses import replace
from typing import Any

from ir_training.common.progress import log
from ir_training.qat.full_model_contract import (
    OPTIMIZER_PREFLIGHT_KIND,
    OPTIMIZER_PREFLIGHT_STEPS,
)
from ir_training.train.full_parameters import FullParameterScopeError
from ir_training.train.tensor_checks import (
    DEFAULT_CHECK_CHUNK_ELEMENTS,
    tensor_all_finite,
)


class _RepeatedLongestRow:
    """Map-style dataset; repeat references, not copies of the real token cache."""

    def __init__(self, row: dict, count: int):
        self.row, self.count = row, count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.count:
            raise IndexError(index)
        return dict(self.row)


def run_full_trainer_preflight(
    checked_trainer_cls: Any,
    trainer_kwargs: dict,
    *,
    longest_row: dict,
    force_cpu: bool = False,
) -> dict:
    """Destructively probe a disposable model; never save or reuse it for training."""
    import torch
    from transformers import TrainerCallback

    model = trainer_kwargs["model"]
    production_args = trainer_kwargs["args"]
    accumulation = production_args.gradient_accumulation_steps
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    parameters = list(model.named_parameters())  # Unique parameters, not tied aliases.
    if not parameters or any(not p.requires_grad or p.dtype != torch.float32 for _, p in parameters):
        raise FullParameterScopeError("Trainer preflight requires every unique parameter trainable in FP32")
    device = parameters[0][1].device
    if any(p.device != device or p.grad is not None for _, p in parameters):
        raise FullParameterScopeError("Trainer preflight requires one device and no pre-existing gradients")
    if (force_cpu and device.type != "cpu") or (not force_cpu and device.type != "cuda"):
        raise FullParameterScopeError("Trainer preflight requires CUDA; force_cpu is only for CPU tests")
    if (type(accumulation) is not int or accumulation < 1
            or production_args.per_device_train_batch_size != 1):
        raise FullParameterScopeError("Trainer preflight requires microbatch 1 and positive accumulation")
    if not force_cpu and not production_args.bf16:
        raise FullParameterScopeError("Full-QAT Trainer preflight requires production BF16 AMP")
    if production_args.max_grad_norm != 0.0:
        raise FullParameterScopeError("Full-QAT Trainer preflight requires no external gradient clipping")

    # Preserve optimizer, precision, checkpointing, dataloader and accumulation
    # settings. Only bound duration and disable external logging/eval/saves.
    # Retain production warmup/scheduler options: LR does not change state size.
    args = replace(
        production_args,
        accelerator_config=copy.deepcopy(production_args.accelerator_config),
        max_steps=OPTIMIZER_PREFLIGHT_STEPS,
        save_strategy="no", eval_strategy="no", logging_strategy="no",
        report_to=[], push_to_hub=False, load_best_model_at_end=False,
        disable_tqdm=True, skip_memory_metrics=True,
    )
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device)) if cuda else 0
    baseline_reserved = int(torch.cuda.memory_reserved(device)) if cuda else 0
    total_memory = int(torch.cuda.get_device_properties(device).total_memory) if cuda else 0
    minimum_free = total_memory
    microsteps = synchronized_microsteps = completed_steps = 0
    optimizer_evidence: dict = {}
    steps: list[dict] = []

    def memory_sample() -> dict:
        nonlocal minimum_free
        if not cuda:
            return {}
        torch.cuda.synchronize(device)
        free, _ = torch.cuda.mem_get_info(device)
        minimum_free = min(minimum_free, int(free))
        return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "device_free_bytes": int(free)}

    def verify_optimizer(optimizer: Any) -> dict:
        raw = getattr(optimizer, "optimizer", optimizer)
        if type(raw).__name__ != "Adafactor":
            raise FullParameterScopeError("Trainer preflight requires the production Adafactor optimizer")
        for group in raw.param_groups:
            if (group.get("beta1") is not None or group.get("scale_parameter") is not False
                    or group.get("relative_step") is not False or group.get("warmup_init") is not False
                    or group.get("weight_decay") != 0.0 or group.get("clip_threshold") != 1.0):
                raise FullParameterScopeError("Trainer preflight Adafactor recipe differs from full-QAT")
        count = numel = 0
        for state in raw.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    if not tensor_all_finite(value):
                        raise FullParameterScopeError("Trainer preflight produced non-finite optimizer state")
                    count += 1
                    numel += value.numel()
        return {"name": "Adafactor", "learning_rate": float(production_args.learning_rate),
                "beta1": None, "scale_parameter": False, "relative_step": False,
                "warmup_init": False, "weight_decay": 0.0, "clip_threshold": 1.0,
                "external_max_grad_norm_required": 0.0,
                "state_tensor_count": count, "state_numel": numel}

    class ProbeCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if microsteps != (completed_steps + 1) * accumulation:
                raise FullParameterScopeError("Trainer probe did not execute a complete accumulation window")
            for name, parameter in parameters:
                if parameter.grad is None or not tensor_all_finite(parameter.grad):
                    raise FullParameterScopeError(f"Trainer preflight missing/non-finite gradient: {name}")
            verify_optimizer(kwargs["optimizer"])
            memory_sample()

        def on_optimizer_step(self, args, state, control, **kwargs):
            nonlocal completed_steps, optimizer_evidence
            for name, parameter in parameters:
                if not tensor_all_finite(parameter):
                    raise FullParameterScopeError(f"Trainer preflight non-finite parameter after step: {name}")
            optimizer_evidence = verify_optimizer(kwargs["optimizer"])
            if not optimizer_evidence["state_tensor_count"] or not optimizer_evidence["state_numel"]:
                raise FullParameterScopeError("Trainer preflight did not materialize optimizer state")
            completed_steps += 1
            snapshot = memory_sample()
            steps.append({"optimizer_step": completed_steps, "microsteps": microsteps, **snapshot})
            log(f"Full-QAT Trainer probe rank={os.environ.get('RANK', '0')}: "
                f"update {completed_steps}/{OPTIMIZER_PREFLIGHT_STEPS}, "
                f"microsteps={microsteps}, memory={snapshot}; disposable, no checkpoint")

    class ProbeTrainer(checked_trainer_cls):
        def training_step(self, live_model, inputs, *step_args, **step_kwargs):
            nonlocal microsteps, synchronized_microsteps
            if world_size > 1:
                if not isinstance(live_model, torch.nn.parallel.DistributedDataParallel):
                    raise FullParameterScopeError("Trainer preflight requires actual Accelerate-prepared DDP")
                if not torch.distributed.is_initialized() or torch.distributed.get_world_size() != world_size:
                    raise FullParameterScopeError("Trainer preflight distributed world does not match launch")
                if (not live_model.require_backward_grad_sync or not live_model.gradient_as_bucket_view
                        or live_model.find_unused_parameters or live_model.broadcast_buffers):
                    raise FullParameterScopeError("Trainer preflight DDP memory policy differs from production")
            if self.accelerator.gradient_state.plugin_kwargs.get("sync_each_batch") is not True:
                raise FullParameterScopeError("Trainer preflight requires sync_each_batch=True")
            if self.accelerator.gradient_accumulation_steps != 1:
                raise FullParameterScopeError("Trainer must own loss normalization; Accelerator must not divide twice")
            if cuda and self.accelerator.mixed_precision != "bf16":
                raise FullParameterScopeError("Trainer preflight did not activate production BF16 AMP")
            loss = super().training_step(live_model, inputs, *step_args, **step_kwargs)
            if not tensor_all_finite(loss):
                raise FullParameterScopeError("Trainer preflight produced non-finite loss")
            microsteps += 1
            synchronized_microsteps += 1
            memory_sample()
            return loss

        def save_model(self, *save_args, **save_kwargs):
            raise FullParameterScopeError("Disposable Trainer preflight must never save a model")

        def _save_checkpoint(self, *save_args, **save_kwargs):
            raise FullParameterScopeError("Disposable Trainer preflight must never save a checkpoint")

    kwargs = {**trainer_kwargs, "args": args, "eval_dataset": None,
              "callbacks": [ProbeCallback()],
              "train_dataset": _RepeatedLongestRow(
                  longest_row, OPTIMIZER_PREFLIGHT_STEPS * accumulation * world_size)}
    memory_sample()
    trainer = ProbeTrainer(**kwargs)
    trainer.train()
    if (completed_steps != OPTIMIZER_PREFLIGHT_STEPS
            or trainer.state.global_step != OPTIMIZER_PREFLIGHT_STEPS
            or microsteps != OPTIMIZER_PREFLIGHT_STEPS * accumulation):
        raise FullParameterScopeError("Trainer preflight ended before startup and resident-state accumulation completed")
    memory_sample()
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if cuda else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if cuda else 0
    fraction = peak_reserved / total_memory if cuda else None
    passed = not cuda or (fraction < 0.90 and minimum_free / total_memory > 0.10)
    report = {
        "schema_version": 2, "probe": OPTIMIZER_PREFLIGHT_KIND, "passed": passed,
        "disposable_worker_required": True, "model_must_not_be_reused": True,
        "checkpoint_writes": 0, "disposable_optimizer_steps": completed_steps,
        "optimizer": optimizer_evidence,
        "scope": {"unique_parameter_count": len(parameters),
                  "trainable_numel": sum(p.numel() for _, p in parameters),
                  "all_trainable_fp32": True, "all_gradients_finite": True,
                  "all_parameters_finite_after_step": True},
        "tensor_validation": {"method": "exhaustive_bounded_chunks",
                              "chunk_elements": DEFAULT_CHECK_CHUNK_ELEMENTS},
        "ddp_probe": {"world_size": world_size, "all_reduce_exercised": world_size > 1,
                      "gradient_as_bucket_view": True, "sync_each_batch": True,
                      "trainer_backend": "transformers", "trainer_path": "checked_causal_lm_trainer",
                      "gradient_accumulation_steps": accumulation, "microsteps": microsteps,
                      "synchronized_microsteps": synchronized_microsteps, "optimizer_steps": completed_steps},
        "memory": {"device": str(device), "cuda_local_rank_only": cuda,
                   "ddp_collectives_certified": cuda and world_size > 1,
                   "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
                   "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                   "device_total_bytes": total_memory, "device_free_bytes_min": minimum_free,
                   "peak_reserved_fraction": fraction, "max_reserved_fraction": 0.90},
        "steps": steps,
        "selection": {"longest_sequence_length": len(longest_row["input_ids"]),
                      "repeated_real_prepared_row": True},
    }
    if not passed:
        raise FullParameterScopeError(
            "Disposable Trainer accumulation probe exceeded CUDA memory/headroom limit: "
            f"reserved_fraction={fraction:.6f} (limit <0.90), "
            f"minimum_free_fraction={minimum_free / total_memory:.6f} (limit >0.10)"
        )
    return report
