"""Collective, dense Hugging Face checkpoint saving for DeepSpeed ZeRO-3."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class Zero3CheckpointError(RuntimeError):
    """Raised when a ZeRO-3 consolidation is not a complete FP32 model."""


def _distributed_context() -> tuple[int, int, Any | None]:
    try:
        import torch.distributed as dist  # type: ignore
    except ImportError:
        return 0, 1, None
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1, None
    try:
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
    except Exception as exc:
        raise Zero3CheckpointError(
            "Could not resolve the initialized distributed ZeRO-3 context."
        ) from exc
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise Zero3CheckpointError(
            f"Invalid distributed ZeRO-3 context: rank={rank}, world_size={world_size}."
        )
    return rank, world_size, dist


def _logical_model(trainer: Any) -> Any:
    engine = getattr(trainer, "deepspeed", None)
    model = getattr(engine, "module", None) or getattr(trainer, "model", None)
    if model is None:
        raise Zero3CheckpointError("ZeRO-3 checkpoint save has no logical model.")
    return model


def _shape(tensor: Any) -> tuple[int, ...]:
    logical = getattr(tensor, "ds_shape", None)
    if logical is None:
        logical = getattr(tensor, "shape", None)
    return tuple(int(value) for value in logical)


def validate_zero3_full_state(
    state: Mapping[str, Any], logical_model: Any
) -> dict[str, Any]:
    """Reject missing, partition-placeholder, non-FP32, or extra state tensors."""

    try:
        expected_state = logical_model.state_dict(keep_vars=True)
    except TypeError:
        expected_state = logical_model.state_dict()
    expected = {name: _shape(tensor) for name, tensor in expected_state.items()}
    try:
        named_parameters = logical_model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = logical_model.named_parameters()
    for name, parameter in named_parameters:
        if name in expected:
            expected[name] = _shape(parameter)
    actual_names = set(state)
    expected_names = set(expected)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise Zero3CheckpointError(
            f"Consolidated ZeRO-3 state keys differ from the logical model; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )

    total_numel = 0
    for name, expected_shape in expected.items():
        tensor = state[name]
        actual_shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
        if actual_shape != expected_shape or int(tensor.numel()) == 0:
            raise Zero3CheckpointError(
                f"Consolidated ZeRO-3 tensor {name!r} is a partition placeholder or "
                f"has shape {actual_shape}; expected {expected_shape}."
            )
        if str(getattr(tensor, "dtype", "")) not in {"torch.float32", "float32", "fp32"}:
            raise Zero3CheckpointError(
                f"Consolidated ZeRO-3 tensor {name!r} is not FP32: {getattr(tensor, 'dtype', None)}."
            )
        total_numel += int(tensor.numel())
    return {
        "verified": True,
        "state_tensor_count": len(expected),
        "state_numel": total_numel,
        "dtype": "float32",
    }


def _exchange_error(error: Exception | None, *, dist: Any | None, world_size: int) -> None:
    if world_size <= 1:
        if error is not None:
            raise error
        return
    failures: list[Any] = [None] * world_size
    dist.all_gather_object(
        failures, f"{type(error).__name__}: {error}" if error is not None else None
    )
    if any(item is not None for item in failures):
        raise Zero3CheckpointError(f"ZeRO-3 checkpoint save failed by rank: {failures}")


def save_zero3_full_model_checkpoint(
    *, trainer: Any, checkpoint_dir: str | Path, tokenizer: Any | None = None
) -> dict[str, Any]:
    """Collectively consolidate state; only rank zero writes the dense HF bundle."""

    rank, world_size, dist = _distributed_context()
    engine = getattr(trainer, "deepspeed", None)
    accelerator = getattr(trainer, "accelerator", None)
    if engine is None or accelerator is None or not callable(
        getattr(accelerator, "get_state_dict", None)
    ):
        raise Zero3CheckpointError(
            "ZeRO-3 checkpoint save requires trainer.deepspeed and accelerator.get_state_dict."
        )
    stage_value = getattr(engine, "zero_optimization_stage", None)
    try:
        stage = stage_value() if callable(stage_value) else stage_value
        stage = int(stage)
    except (TypeError, ValueError) as exc:
        raise Zero3CheckpointError(
            "DeepSpeed engine does not expose a valid zero_optimization_stage."
        ) from exc
    if stage != 3:
        raise Zero3CheckpointError(
            f"Collective ZeRO-3 save requires engine stage 3, got {stage}."
        )

    state = accelerator.get_state_dict(engine)  # collective: every rank must call
    evidence: dict[str, Any] | None = None
    error: Exception | None = None
    destination = Path(checkpoint_dir)
    if rank == 0:
        try:
            model = _logical_model(trainer)
            evidence = validate_zero3_full_state(state, model)
            destination.mkdir(parents=True, exist_ok=True)
            trainer_save = getattr(trainer, "_save", None)
            if callable(trainer_save):
                # _save is intentionally used instead of save_model: the latter
                # re-enters DeepSpeed consolidation and may swallow a failed
                # ZeRO-3 state-dict extraction as a shard-only checkpoint.
                trainer_save(str(destination), state_dict=state)
            else:
                model.save_pretrained(
                    str(destination), state_dict=state, safe_serialization=True
                )
            if tokenizer is not None:
                tokenizer.save_pretrained(str(destination))
            files = sorted(destination.glob("model*.safetensors"))
            if not files:
                raise Zero3CheckpointError(
                    "ZeRO-3 dense save produced no model*.safetensors files."
                )
            evidence.update(
                {
                    "method": "accelerator.get_state_dict(deepspeed)",
                    "world_size": world_size,
                    "checkpoint_dir": str(destination),
                    "files": [
                        {
                            "path": path.name,
                            "size": path.stat().st_size,
                            "sha256": _sha256_file(path),
                        }
                        for path in files
                    ],
                }
            )
        except Exception as exc:  # noqa: BLE001 - exchange every writer error before any barrier
            error = exc
    _exchange_error(error, dist=dist, world_size=world_size)
    if world_size > 1:
        payload: list[Any] = [evidence]
        dist.broadcast_object_list(payload, src=0)
        evidence = payload[0]
        dist.barrier()
    if not isinstance(evidence, dict):
        raise Zero3CheckpointError("ZeRO-3 checkpoint evidence was not published.")
    return evidence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_zero3_checkpoint(
    trainer: Any, checkpoint_dir: str | Path, *, tokenizer: Any | None = None
) -> dict[str, Any]:
    """Public final/best checkpoint entry point; every rank must call it."""

    if tokenizer is None:
        tokenizer = getattr(trainer, "processing_class", None)
    return save_zero3_full_model_checkpoint(
        trainer=trainer, checkpoint_dir=checkpoint_dir, tokenizer=tokenizer
    )
