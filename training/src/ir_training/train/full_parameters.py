"""Fail-closed preparation of a model for true all-parameter SFT.

This module is deliberately independent of the existing LoRA and retained-mobile
QAT paths.  It makes every unique model parameter trainable in FP32 while
preserving tied-parameter aliases, and returns JSON-serializable scope evidence.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ir_training.train.tensor_checks import (
    DEFAULT_CHECK_CHUNK_ELEMENTS,
    tensor_all_finite,
)


class FullParameterScopeError(ValueError):
    """Raised when a model cannot satisfy the true full-parameter contract."""


def validate_full_training_runtime(trainer_class: Any, ddp_kwargs_class: Any = None) -> dict:
    """Check the new lane's DDP API before loading multi-gigabyte weights."""
    if not callable(getattr(trainer_class, "_build_accelerator_args", None)):
        raise FullParameterScopeError(
            "All-parameter QAT requires Trainer._build_accelerator_args; install "
            "training/requirements-full-parameter-qat.txt in a separate training environment."
        )
    if ddp_kwargs_class is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs_class = DistributedDataParallelKwargs
    handler = ddp_kwargs_class()
    if not hasattr(handler, "gradient_as_bucket_view"):
        raise FullParameterScopeError(
            "All-parameter QAT requires Accelerate DDP gradient_as_bucket_view support."
        )
    return {"verified": True, "trainer_preconstruction_hook": True,
            "ddp_gradient_bucket_views_available": True, "gpu_execution_tested": False}


def _named_parameters_with_aliases(model: Any) -> list[tuple[str, Any]]:
    try:
        return list(model.named_parameters(remove_duplicate=False))
    except TypeError as exc:  # pragma: no cover - old torch is not supported in production
        raise FullParameterScopeError(
            "True full-parameter SFT requires named_parameters(remove_duplicate=False) "
            "so tied aliases can be audited."
        ) from exc


def _looks_adapter_backed(model: Any, names: list[str]) -> bool:
    if any("lora_" in name.lower() or ".adapter" in name.lower() for name in names):
        return True
    if getattr(model, "peft_config", None):
        return True
    return type(model).__module__.startswith("peft.")


def _looks_quantized(model: Any) -> bool:
    if bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
        getattr(model, "is_loaded_in_8bit", False)
    ):
        return True
    if getattr(model, "quantization_method", None) is not None:
        return True
    config = getattr(model, "config", None)
    return getattr(config, "quantization_config", None) is not None


def enable_full_parameter_training(model: Any) -> dict[str, Any]:
    """Enable true all-parameter FP32 training and return scope evidence.

    The function rejects adapter-backed, quantized, empty, non-floating, or
    structurally unstable models.  Tied parameters are converted once and their
    aliases remain tied.  The returned dictionary contains only JSON-compatible
    values and can be embedded directly in checkpoint provenance.
    """

    import torch

    before = _named_parameters_with_aliases(model)
    if not before:
        raise FullParameterScopeError("True full-parameter SFT requires at least one parameter.")
    before_names = [name for name, _ in before]
    if len(before_names) != len(set(before_names)):
        raise FullParameterScopeError("Model exposes duplicate named-parameter paths.")
    if _looks_adapter_backed(model, before_names):
        raise FullParameterScopeError("True full-parameter SFT rejects LoRA/adapter-backed models.")
    if _looks_quantized(model):
        raise FullParameterScopeError("True full-parameter SFT rejects quantized model wrappers.")

    by_id: dict[int, dict[str, Any]] = {}
    for name, parameter in before:
        if not isinstance(parameter, torch.nn.Parameter):
            raise FullParameterScopeError(f"Named value {name!r} is not a torch Parameter.")
        if parameter.is_quantized or not (parameter.is_floating_point() or parameter.is_complex()):
            raise FullParameterScopeError(
                f"True full-parameter SFT requires floating parameters; {name!r} has dtype {parameter.dtype}."
            )
        if parameter.is_complex():
            raise FullParameterScopeError(
                f"True full-parameter SFT does not support complex parameter {name!r}."
            )
        record = by_id.setdefault(
            id(parameter),
            {
                "parameter": parameter,
                "aliases": [],
                "source_dtype": str(parameter.dtype).removeprefix("torch."),
                "numel": int(parameter.numel()),
            },
        )
        record["aliases"].append(name)

    for record in by_id.values():
        parameter = record["parameter"]
        if parameter.dtype != torch.float32:
            parameter.data = parameter.detach().to(dtype=torch.float32)
        parameter.requires_grad_(True)

    after = _named_parameters_with_aliases(model)
    after_ids = {name: id(parameter) for name, parameter in after}
    before_ids = {name: id(parameter) for name, parameter in before}
    if after_ids != before_ids:
        raise FullParameterScopeError("Named-parameter identities or aliases changed during FP32 preparation.")
    if any(parameter.dtype != torch.float32 for _, parameter in after):
        raise FullParameterScopeError("Not every model parameter was converted to FP32.")
    if any(not parameter.requires_grad for _, parameter in after):
        raise FullParameterScopeError("Not every model parameter is trainable after preparation.")

    parameters = []
    for record in by_id.values():
        parameters.append(
            {
                "canonical_name": record["aliases"][0],
                "aliases": list(record["aliases"]),
                "source_dtype": record["source_dtype"],
                "trainable_dtype": "float32",
                "numel": record["numel"],
            }
        )
    parameters.sort(key=lambda item: item["canonical_name"])
    alias_count = sum(len(item["aliases"]) for item in parameters)
    return {
        "schema_version": 1,
        "scope": "all_model_parameters",
        "verified": True,
        "master_trainable_dtype": "float32",
        "unique_parameter_count": len(parameters),
        "named_parameter_count": alias_count,
        "tied_alias_count": alias_count - len(parameters),
        "trainable_numel": sum(item["numel"] for item in parameters),
        "frozen_parameter_count": 0,
        "adapter_parameter_count": 0,
        "parameters": parameters,
    }


def probe_full_optimizer_step(
    model: Any,
    *,
    backward: Callable[[], None],
    learning_rate: float,
    max_reserved_fraction: float = 0.90,
    force_cpu: bool = False,
) -> dict[str, Any]:
    """Materialize Adafactor state with one destructive, disposable step.

    ``backward`` must run the caller's real longest-batch BF16/QAT loss and
    populate gradients for every parameter.  This function deliberately does
    not clone or restore parameters or optimizer state: its worker must exit
    immediately and must never publish a checkpoint or continue into training.
    CUDA evidence is local-rank evidence only; it does not certify DDP peers or
    collective-memory behavior.
    """

    import torch
    from transformers.optimization import Adafactor

    if not callable(backward):
        raise FullParameterScopeError("Optimizer probe requires a callable backward operation.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise FullParameterScopeError("Optimizer probe learning_rate must be positive and finite.")
    if (
        not math.isfinite(max_reserved_fraction)
        or max_reserved_fraction <= 0
        or max_reserved_fraction >= 1
    ):
        raise FullParameterScopeError("max_reserved_fraction must be finite and strictly between 0 and 1.")

    named = _named_parameters_with_aliases(model)
    unique: dict[int, tuple[str, Any]] = {}
    for name, parameter in named:
        unique.setdefault(id(parameter), (name, parameter))
    if not unique:
        raise FullParameterScopeError("Optimizer probe requires at least one parameter.")
    invalid_scope = [
        name
        for name, parameter in unique.values()
        if not parameter.requires_grad or parameter.dtype != torch.float32
    ]
    if invalid_scope:
        raise FullParameterScopeError(
            "Optimizer probe requires every unique parameter trainable in FP32; invalid="
            + ", ".join(invalid_scope[:12])
        )
    devices = {parameter.device for _, parameter in unique.values()}
    if len(devices) != 1:
        raise FullParameterScopeError("Optimizer probe requires all parameters on one local-rank device.")
    device = next(iter(devices))
    cuda = device.type == "cuda" and not force_cpu
    if not force_cpu and device.type != "cuda":
        raise FullParameterScopeError(
            "Optimizer probe requires CUDA unless force_cpu=True is explicitly used for tests."
        )
    if force_cpu and device.type != "cpu":
        raise FullParameterScopeError("force_cpu=True requires a CPU-resident model.")

    for _, parameter in unique.values():
        parameter.grad = None
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        baseline_allocated = baseline_reserved = total_memory = 0

    optimizer = Adafactor(
        [parameter for _, parameter in unique.values()],
        lr=float(learning_rate),
        beta1=None,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.0,
        clip_threshold=1.0,
    )
    backward()
    missing = [name for name, parameter in unique.values() if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in unique.values()
        if parameter.grad is not None and not tensor_all_finite(parameter.grad)
    ]
    if missing or nonfinite:
        optimizer.zero_grad(set_to_none=True)
        raise FullParameterScopeError(
            "Optimizer probe gradients failed: "
            f"missing={missing[:12]}, nonfinite={nonfinite[:12]}."
        )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    nonfinite_parameters = [
        name
        for name, parameter in unique.values()
        if not tensor_all_finite(parameter)
    ]
    if nonfinite_parameters:
        raise FullParameterScopeError(
            "Optimizer probe produced non-finite parameters: "
            + ", ".join(nonfinite_parameters[:12])
        )

    state_tensor_count = 0
    state_numel = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                state_tensor_count += 1
                state_numel += int(value.numel())
    if state_tensor_count == 0:
        raise FullParameterScopeError("Adafactor probe did not materialize optimizer tensor state.")

    if cuda:
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        reserved_fraction = peak_reserved / total_memory
    else:
        peak_allocated = peak_reserved = 0
        reserved_fraction = None

    report = {
        "schema_version": 1,
        "probe": "disposable_full_parameter_adafactor_step",
        "passed": not cuda or reserved_fraction < max_reserved_fraction,
        "disposable_worker_required": True,
        "model_must_not_be_reused": True,
        "checkpoint_writes": 0,
        "disposable_optimizer_steps": 1,
        "optimizer": {
            "name": "Adafactor",
            "learning_rate": float(learning_rate),
            "beta1": None,
            "scale_parameter": False,
            "relative_step": False,
            "warmup_init": False,
            "weight_decay": 0.0,
            "clip_threshold": 1.0,
            "external_max_grad_norm_required": 0.0,
            "state_tensor_count": state_tensor_count,
            "state_numel": state_numel,
        },
        "scope": {
            "unique_parameter_count": len(unique),
            "trainable_numel": sum(int(parameter.numel()) for _, parameter in unique.values()),
            "all_trainable_fp32": True,
            "all_gradients_finite": True,
            "all_parameters_finite_after_step": True,
        },
        "tensor_validation": {
            "method": "exhaustive_bounded_chunks",
            "chunk_elements": DEFAULT_CHECK_CHUNK_ELEMENTS,
        },
        "memory": {
            "device": str(device),
            "cuda_local_rank_only": cuda,
            "ddp_collectives_certified": False,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "device_total_bytes": total_memory,
            "peak_reserved_fraction": reserved_fraction,
            "max_reserved_fraction": float(max_reserved_fraction),
        },
    }
    if not report["passed"]:
        raise FullParameterScopeError(
            "Disposable optimizer probe exceeded CUDA reserved-memory limit: "
            f"peak_fraction={reserved_fraction:.6f}, limit={max_reserved_fraction:.6f}."
        )
    return report
