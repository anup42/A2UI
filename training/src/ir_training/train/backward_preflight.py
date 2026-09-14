"""Live, non-updating worst-shape backward smoke check for each SFT worker.

This intentionally uses the same full-vocabulary checked loss as training. It
does not certify optimizer-state headroom, every kernel shape, or NCCL health.
No result from this module is cacheable. Models must not mutate their weights
or running-stat buffers during the probe; unexpected in-place mutations fail
closed rather than allowing a modified initialization to proceed to training.
"""
from __future__ import annotations

import os
import random
from typing import Any

from ir_training.common.progress import Progress, log

_MISSING = object()
_CHECKPOINT_ATTRIBUTES = ("gradient_checkpointing", "_gradient_checkpointing_func", "_require_grads_hook")
_HOOK_REGISTRIES = ("_forward_hooks", "_forward_pre_hooks", "_forward_hooks_with_kwargs",
                    "_forward_pre_hooks_with_kwargs", "_forward_hooks_always_called")


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _select_probe_batches(dataset: Any, microbatch: int, max_seq_length: int, interval: float) -> tuple[list[dict], dict]:
    if not len(dataset):
        raise ValueError("Backward preflight requires a non-empty tokenized training split.")
    # Arrow column projection avoids decoding labels/masks for the length scan.
    lengths = dataset.select_columns(["input_ids"]) if callable(getattr(dataset, "select_columns", None)) else dataset
    longest_index = shortest_index = 0
    longest_length, shortest_length = 0, max_seq_length + 1
    with Progress("Backward preflight: scan training sequence lengths", total=len(dataset), interval=interval) as progress:
        for index, row in enumerate(lengths):
            ids = row.get("input_ids")
            if ids is None or not 2 <= len(ids) <= max_seq_length:
                raise ValueError(f"Backward preflight row {index} has invalid sequence length; truncation is forbidden.")
            size = len(ids)
            if size > longest_length:
                longest_index, longest_length = index, size
            if size < shortest_length:
                shortest_index, shortest_length = index, size
            progress.advance()
    batches = [{"kind": "longest_unpadded", "indices": [longest_index] * microbatch}]
    if microbatch > 1 and shortest_length < longest_length:
        batches.append({"kind": "longest_padded", "indices": [longest_index] * (microbatch - 1) + [shortest_index]})
    return batches, {"rows_scanned": len(dataset), "longest_index": longest_index,
                     "longest_length": longest_length, "shortest_index": shortest_index,
                     "shortest_length": shortest_length}


def _snapshot_modules(model: Any) -> list[dict]:
    # Preserve references/flags only: cloning model weights or all buffers could
    # itself exhaust H100 memory. Hook snapshots preserve SFT's pre-existing
    # input-gradient hook when Transformers adds another during GC enablement.
    return [{"module": module, "training": module.training,
             "attributes": {name: vars(module).get(name, _MISSING) for name in _CHECKPOINT_ATTRIBUTES},
             "hooks": {name: dict(getattr(module, name)) for name in _HOOK_REGISTRIES if hasattr(module, name)},
             "parameters": dict(module._parameters), "buffers": dict(module._buffers),
             "nonpersistent": set(module._non_persistent_buffers_set)}
            for module in model.modules()]


def _restore_modules(snapshots: list[dict]) -> None:
    for snapshot in snapshots:
        module = snapshot["module"]
        # Calling train() recursively would destroy independently frozen/eval
        # child modes. Restore each module's original flag exactly.
        module.training = snapshot["training"]
        for name, value in snapshot["attributes"].items():
            if value is _MISSING:
                vars(module).pop(name, None)
            else:
                setattr(module, name, value)
        for name, original in snapshot["hooks"].items():
            registry = getattr(module, name)
            registry.clear()
            registry.update(original)
        module._parameters.clear()
        module._parameters.update(snapshot["parameters"])
        module._buffers.clear()
        module._buffers.update(snapshot["buffers"])
        module._non_persistent_buffers_set = snapshot["nonpersistent"]


def _memory(torch: Any, device: Any) -> dict:
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "device_free_bytes": free, "device_total_bytes": total}


def run_backward_preflight(model: Any, tokenizer: Any, dataset: Any, training_cfg: dict,
                           max_seq_length: int, input_vocab_size: int, label_vocab_size: int,
                           max_position_embeddings: int | None, force_cpu: bool = False) -> dict:
    """Run local backward without updating weights, RNG or caller-owned grads.

    ``force_cpu`` permits tiny CPU regression fixtures; production defaults to
    CUDA only. ``dataset`` is the actual tokenized train split, not raw JSONL.
    """
    enabled = training_cfg.get("backward_preflight", True)
    if type(enabled) is not bool:
        raise ValueError("training.backward_preflight must be a boolean.")
    base = {"enabled": enabled, "optimizer_steps": 0, "cached": False,
            "distributed_collectives_tested": False}
    if not enabled:
        log("SFT backward preflight: disabled explicitly.")
        return {**base, "status": "disabled"}
    import torch

    from ir_training.train.sft import (
        _CausalLMDataCollator,
        _checked_shifted_causal_lm_loss,
        _drop_trivial_attention_mask,
        _extract_logits,
        _model_input_device,
    )

    device = torch.device(_model_input_device(model))
    if device.type != "cuda" and not force_cpu:
        log("SFT backward preflight: skipped on non-CUDA model (CPU fixtures may opt in).")
        return {**base, "status": "skipped", "reason": "non_cuda_model", "device": str(device)}
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Backward preflight supports CPU test fixtures or a rank-local CUDA model.")
    microbatch = _positive_integer(training_cfg.get("per_device_train_batch_size", 1), "Training microbatch")
    accumulation = _positive_integer(training_cfg.get("gradient_accumulation_steps", 1), "Gradient accumulation steps")
    max_seq_length = _positive_integer(max_seq_length, "Training sequence limit")
    if max_position_embeddings is not None:
        max_seq_length = min(max_seq_length, _positive_integer(max_position_embeddings, "Model sequence limit"))
    checkpointing = training_cfg.get("gradient_checkpointing", False)
    if type(checkpointing) is not bool:
        raise ValueError("training.gradient_checkpointing must be a boolean.")
    checkpoint_kwargs = training_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})
    if not isinstance(checkpoint_kwargs, dict):
        raise ValueError("training.gradient_checkpointing_kwargs must be an object.")  # noqa: TRY004 - configuration errors
    parameters = list(model.named_parameters())
    if any(parameter.grad is not None for _, parameter in parameters):
        raise ValueError("Backward preflight refuses pre-existing gradients; it must run before training updates.")
    if not any(parameter.requires_grad for _, parameter in parameters):
        raise ValueError("Backward preflight requires trainable parameters.")
    if any(parameter.device != device for _, parameter in parameters):
        raise ValueError("Backward preflight requires one rank-local model device, not a sharded device map.")
    interval = float(training_cfg.get("cache_progress_seconds", 10))
    batches, selection = _select_probe_batches(dataset, microbatch, max_seq_length, interval)
    # At most two backwards per shape: the first allocates gradients, the next
    # exercises forward/recompute with those gradients still resident. Do not
    # consume a whole accumulation cycle, sampler or optimizer step.
    passes = min(2, accumulation)
    batches = [{**batch, "accumulation_microstep": step + 1, "keep_gradients": step + 1 < passes}
               for batch in batches for step in range(passes)]
    collator = _CausalLMDataCollator(tokenizer, input_vocab_size=input_vocab_size,
                                    label_vocab_size=label_vocab_size,
                                    max_position_embeddings=max_position_embeddings)
    snapshots = _snapshot_modules(model)
    # Version counters catch ordinary forbidden in-place parameter/buffer writes
    # without allocating a second model. Reference replacements are restored.
    tensor_versions = [(name, tensor, tensor._version) for name, tensor in
                       list(model.named_parameters()) + list(model.named_buffers())]
    python_state = random.getstate()
    import numpy as np
    numpy_state = np.random.get_state()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    results = []
    phase, kind, batch_shape = "configure", "none", None
    memory_before = _memory(torch, device)
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            if checkpointing:
                enabler = getattr(model, "gradient_checkpointing_enable", None)
                if not callable(enabler):
                    raise ValueError("Configured gradient checkpointing is unsupported by this model.")
                enabler(gradient_checkpointing_kwargs=dict(checkpoint_kwargs))
            else:
                # Match Trainer's requested disabled path even if the loader
                # left a checkpoint-capable child enabled. Restore afterwards.
                for module in model.modules():
                    if hasattr(module, "gradient_checkpointing"):
                        module.gradient_checkpointing = False
            model.train()
            for planned in batches:
                kind = planned["kind"]
                phase = "collate"
                features = [dataset[index] for index in planned["indices"]]
                batch = {key: tensor.to(device) for key, tensor in collator(features).items()}
                labels = batch.pop("labels")
                batch_shape = list(batch["input_ids"].shape)
                _drop_trivial_attention_mask(batch)
                log(f"SFT backward preflight: rank={os.environ.get('RANK', '0')}, {kind}, "
                    f"rows={planned['indices']}, shape={batch_shape}, checkpointing={checkpointing}, "
                    f"microstep={planned['accumulation_microstep']}/{passes}; no optimizer step.")
                with Progress(f"SFT backward preflight {kind}", unit="stage", interval=interval):
                    phase = "forward"
                    outputs = model(**batch)
                    phase = "loss"
                    loss = _checked_shifted_causal_lm_loss(_extract_logits(outputs), labels)
                    if not bool(torch.isfinite(loss).all().item()):
                        raise ValueError("Backward preflight produced non-finite loss.")
                    loss_value = float(loss.detach().item())
                    phase = "backward"
                    (loss / accumulation).backward()
                    phase = "gradient_validation"
                    present = nonzero = 0
                    for name, parameter in parameters:
                        if not parameter.requires_grad or parameter.grad is None:
                            continue
                        present += 1
                        gradient = parameter.grad._values() if parameter.grad.is_sparse else parameter.grad
                        if not bool(torch.isfinite(gradient).all().item()):
                            raise ValueError(f"Backward preflight produced non-finite gradient in {name}.")
                        nonzero += int(bool(torch.count_nonzero(gradient).item()))
                    if not nonzero:
                        raise ValueError("Backward preflight produced no nonzero trainable gradients.")
                    phase = "synchronize"
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    phase = "state_validation"
                    changed = [name for name, tensor, version in tensor_versions if tensor._version != version]
                    if changed:
                        raise ValueError("Backward preflight model mutated parameters/buffers in place: " + ", ".join(changed[:8]))
                    results.append({"kind": kind, "row_indices": planned["indices"], "input_shape": batch_shape,
                                    "accumulation_microstep": planned["accumulation_microstep"],
                                    "completion_loss": loss_value, "gradient_tensors": present,
                                    "nonzero_gradient_tensors": nonzero, "memory": _memory(torch, device)})
                    del gradient
                    if not planned["keep_gradients"]:
                        model.zero_grad(set_to_none=True)
                    del outputs, loss, labels, batch
        report = {**base, "status": "passed", "device": str(device), "microbatch": microbatch,
                  "gradient_accumulation_steps": accumulation, "backward_passes": len(results),
                  "gradient_checkpointing": checkpointing, "gradient_checkpointing_kwargs": checkpoint_kwargs,
                  "selection": selection, "batches": results, "memory_before": memory_before,
                  "scope": "local worst-shape forward/backward only; optimizer memory, all shapes and NCCL are not certified"}
        log(f"SFT backward preflight passed: {len(results)} backward passes; no optimizer step; weights and RNG preserved.")
        return report
    except Exception as exc:
        raise RuntimeError(
            "SFT backward preflight failed before optimizer step 1; no automatic retry. "
            f"rank={os.environ.get('RANK', '0')}, device={device}, phase={phase}, "
            f"batch={kind}, shape={batch_shape}, microbatch={microbatch}, "
            f"checkpointing={checkpointing}, error={exc!r}. "
            "Use a new worker process after a CUDA failure; reduce microbatch or select a healthy GPU allocation."
        ) from exc
    finally:
        model.zero_grad(set_to_none=True)
        _restore_modules(snapshots)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
