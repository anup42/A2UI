"""Disposable real-Trainer GRPO rehearsal; never publishes updated adapters."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_implementation() -> dict[str, str]:
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_text(encoding="utf-8").encode()).hexdigest() for name in
            ("grpo_probe.py", "qat_grpo.py", "grpo_runtime.py", "qat_grpo_reward.py")}


def longest_prompt_indices(rows: Any, count: int) -> list[int]:
    if count < 1 or len(rows) < count:
        raise ValueError("GRPO probe requires a complete generation group")
    return sorted(range(len(rows)), key=lambda i: (-int(rows[i]["prompt_provenance"]["prompt_tokens"]), i))[:count]


def runtime_arguments(training: dict, parameters: Any, max_steps: int | None) -> dict:
    """Forward requested supported knobs rather than silently ignoring a profile."""
    defaults = {"optim": "adamw_torch", "lr_scheduler_type": "linear", "tf32": False,
                "dataloader_num_workers": 0, "dataloader_pin_memory": True,
                "dataloader_persistent_workers": False}
    result = {}
    for key, default in defaults.items():
        value = training.get(key, default)
        if key not in parameters:
            raise ValueError(f"Installed GRPOConfig cannot apply training.{key}")
        result[key] = value
    if result["tf32"] is not False:
        raise ValueError("QAT GRPO requires tf32=false for retained merge arithmetic")
    workers = result["dataloader_num_workers"]
    if type(workers) is not int or workers < 0:
        raise ValueError("dataloader_num_workers must be a nonnegative integer")
    if workers == 0 and result["dataloader_persistent_workers"]:
        raise ValueError("persistent workers require nonzero dataloader workers")
    if training.get("dataloader_prefetch_factor") is not None:
        if workers == 0 or "dataloader_prefetch_factor" not in parameters:
            raise ValueError("Requested dataloader prefetch is not supported by this runtime")
        result["dataloader_prefetch_factor"] = training["dataloader_prefetch_factor"]
    warmup = training.get("warmup_steps", 0)
    if (isinstance(warmup, bool) or not isinstance(warmup, (float, int)) or not math.isfinite(warmup)
            or warmup < 0 or (max_steps is not None and warmup >= max_steps)):
        raise ValueError("GRPO warmup_steps must be nonnegative and below max_steps")
    return result


def validate_probe_report(report: dict, config_path: Path, world_size: int, rank: int) -> None:
    expected = {"config_sha256": file_hash(config_path), "implementation": probe_implementation(),
                "world_size": world_size, "rank": rank, "optimizer_steps": 2}
    if any(report.get(k) != value for k, value in expected.items()) or report.get("passed") is not True:
        raise ValueError(f"Missing/stale/failed real GRPO Trainer probe for rank {rank}")
    fractions = (report.get("max_reserved_fraction"), report.get("minimum_free_fraction"))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in fractions):
        raise ValueError(f"Invalid memory evidence for GRPO runtime probe rank {rank}")
    if fractions[0] > .90 or fractions[1] < .10:
        raise ValueError(f"GRPO runtime probe failed 90% memory/headroom gate on rank {rank}")


def verify_probe_reports(directory: Path, config_path: Path, world_size: int) -> list[Path]:
    paths = [directory / f"grpo_runtime_probe.rank{rank}.json" for rank in range(world_size)]
    for rank, path in enumerate(paths):
        validate_probe_report(json.loads(path.read_text(encoding="utf-8")), config_path, world_size, rank)
    return paths


def device_identity(torch: Any, device: Any) -> dict:
    properties = torch.cuda.get_device_properties(device)
    return {"name": properties.name, "total_memory": properties.total_memory,
            "capability": [properties.major, properties.minor]}


def verify_local_probe_device(path: Path, torch: Any, device: Any) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("device") != device_identity(torch, device):
        raise ValueError("GRPO runtime probe was not executed on matching GPU hardware")


def run_disposable_probe(trainer: Any, *, output: Path, config_path: Path, selected_rows: list[dict]) -> dict:
    import torch
    from transformers import TrainerCallback

    from ir_training.train.tensor_checks import tensor_finite_and_nonzero

    device = trainer.accelerator.device
    rank, world = trainer.accelerator.process_index, trainer.accelerator.num_processes
    report = {"schema_version": 1, "rank": rank, "world_size": world,
              "config_sha256": file_hash(config_path), "implementation": probe_implementation(),
              "passed": False, "optimizer_steps": 0, "selected_rows": selected_rows,
              "scope": "two disposable real GRPO updates; not an exhaustive worst-case completion guarantee",
              "checkpoints_written": False}
    report["device"] = device_identity(torch, device)
    minimum_free = 1.0

    class Probe(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **unused):
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not tensor_finite_and_nonzero(parameter.grad)[0]:
                    raise ValueError(f"Nonfinite GRPO probe gradient: {name}")

        def on_step_end(self, args, state, control, model=None, **unused):
            nonlocal minimum_free
            if model is None:
                raise ValueError("GRPO runtime probe did not expose the updated model")
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and not tensor_finite_and_nonzero(parameter.detach())[0]:
                    raise ValueError(f"Nonfinite updated GRPO probe parameter: {name}")
            free, total = torch.cuda.mem_get_info(device)
            minimum_free = min(minimum_free, free / total)

        def on_substep_end(self, args, state, control, **unused):
            nonlocal minimum_free
            free, total = torch.cuda.mem_get_info(device)
            minimum_free = min(minimum_free, free / total)

        def on_log(self, args, state, control, logs=None, **unused):
            for name, value in (logs or {}).items():
                if isinstance(value, (float, int)) and not math.isfinite(value):
                    raise ValueError(f"Nonfinite GRPO probe metric: {name}")

    trainer.add_callback(Probe())
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    minimum_free = free / total
    report.update(baseline_allocated_bytes=torch.cuda.memory_allocated(device),
                  baseline_reserved_bytes=torch.cuda.memory_reserved(device), total_bytes=total)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        trainer.train()
        torch.cuda.synchronize(device)
        report["optimizer_steps"] = int(trainer.state.global_step)
        report["passed"] = report["optimizer_steps"] == 2 and trainer.optimizer is not None
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      max_reserved_fraction=torch.cuda.max_memory_reserved(device) / total,
                      minimum_free_fraction=minimum_free)
        report["passed"] &= report["max_reserved_fraction"] <= .90 and minimum_free >= .10
        output.mkdir(parents=True, exist_ok=True)
        (output / f"grpo_runtime_probe.rank{rank}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    validate_probe_report(report, config_path, world, rank)
    return report
