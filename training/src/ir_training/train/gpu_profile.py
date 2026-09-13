"""GPU inventory and conservative launch profiles; never allocate model weights."""
from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Mapping


def detect_cuda_devices(*, torch_module: Any = None, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Inspect only CUDA-visible devices, preserving scheduler masks and UUIDs.

    Numeric user selections refer to logical indices within this inventory.
    Launch identifiers map those indices back to the inherited CUDA mask so a
    worker never accidentally escapes its scheduler-assigned devices.
    """
    if torch_module is None:
        import torch as torch_module
    environment = os.environ if environment is None else environment
    mask = environment.get("CUDA_VISIBLE_DEVICES")
    identifiers = None if mask is None else [part.strip() for part in mask.split(",") if part.strip()]
    count = int(torch_module.cuda.device_count()) if torch_module.cuda.is_available() else 0
    if count <= 0:
        raise ValueError("No CUDA-visible GPUs. Prepare the launch on the GPU host with its scheduler mask and CUDA PyTorch environment.")
    if identifiers is not None and (len(identifiers) != count or len(set(identifiers)) != count):
        raise ValueError("CUDA-visible device count disagrees with CUDA_VISIBLE_DEVICES; resolve the mask before preparing training.")
    devices = []
    for index in range(count):
        properties = torch_module.cuda.get_device_properties(index)
        memory = int(properties.total_memory)
        if memory <= 0:
            raise ValueError(f"CUDA device {index} reports invalid memory capacity.")
        uuid = getattr(properties, "uuid", None)
        devices.append({
            "visible_index": index,
            "launch_identifier": identifiers[index] if identifiers is not None else str(index),
            "uuid": str(uuid) if uuid else None,
            "name": str(properties.name),
            "total_memory_bytes": memory,
            "compute_capability": [int(properties.major), int(properties.minor)],
        })
    return {"version": 1, "inherited_cuda_visible_devices": mask, "visible_gpu_count": count, "devices": devices}


def select_devices(inventory: dict[str, Any], selection: str | None = "auto") -> list[dict[str, Any]]:
    devices = list(inventory["devices"])
    value = str(selection or "auto").strip()
    if value.lower() in {"auto", "all"}:
        return devices
    tokens = [part.strip() for part in value.split(",") if part.strip()]
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("--devices requires unique visible logical indices or exact GPU/MIG UUIDs.")
    selected = []
    for token in tokens:
        if token.isdecimal():
            index = int(token)
            if index >= len(devices):
                raise ValueError(f"Visible GPU index {index} is out of range for {len(devices)} devices.")
            device = devices[index]
        else:
            matches = [device for device in devices if token in {device.get("uuid"), device["launch_identifier"]}]
            if len(matches) != 1:
                raise ValueError(f"GPU UUID {token!r} is not uniquely visible in the inherited CUDA mask.")
            device = matches[0]
        if any(previous["visible_index"] == device["visible_index"] for previous in selected):
            raise ValueError("--devices selects the same visible GPU more than once.")
        selected.append(device)
    return selected


def available_cpu_count() -> int:
    """CPU budget visible to this process, not the entire physical host."""
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            count = min(count, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError, ZeroDivisionError):
        for directory in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
            try:
                quota = int((Path(directory) / "cpu.cfs_quota_us").read_text())
                period = int((Path(directory) / "cpu.cfs_period_us").read_text())
                if quota > 0 and period > 0:
                    count = min(count, max(1, math.ceil(quota / period)))
            except (OSError, ValueError):
                pass
    return max(1, count)


def build_gpu_profile(
    inventory: dict[str, Any], *, model: str, devices: str | None = "auto",
    microbatch: int | None = None, effective_batch: int | None = None,
    dataloader_workers: int | None = None, cpu_count: int | None = None,
) -> dict[str, Any]:
    if model not in {"e2b", "270m"}:
        raise ValueError("GPU profiles support e2b and 270m.")
    selected = select_devices(inventory, devices)
    world_size = len(selected)
    if world_size <= 0:
        raise ValueError("A training launch requires at least one visible GPU.")
    h100 = all("H100" in device["name"].upper() and device["total_memory_bytes"] >= 70 * 1024**3 for device in selected)
    native_bf16 = all(device["compute_capability"][0] >= 8 for device in selected)
    effective = int(effective_batch if effective_batch is not None else (32 if h100 else 16))
    if effective <= 0 or effective % world_size:
        raise ValueError("Effective batch must be positive and divisible by selected GPU count; supply --effective-batch for this host.")
    if microbatch is None:
        preferred = (2 if model == "e2b" else 4) if h100 else 1
        per_rank = effective // world_size
        micro = next(candidate for candidate in range(min(preferred, per_rank), 0, -1) if per_rank % candidate == 0)
    else:
        micro = int(microbatch)
    if micro <= 0 or effective % (world_size * micro):
        raise ValueError("--effective-batch must be divisible by GPU count * microbatch.")
    available_cpus = int(cpu_count if cpu_count is not None else available_cpu_count())
    if available_cpus <= 0:
        raise ValueError("Available CPU count must be positive.")
    workers = int(dataloader_workers) if dataloader_workers is not None else min(4, max(0, available_cpus // world_size // 2))
    if workers < 0:
        raise ValueError("--dataloader-workers must be nonnegative.")
    return {
        "name": f"{model}_h100_70gb_starting" if h100 else f"{model}_conservative_starting",
        "benchmark_verified": False,
        "note": "Conservative starting profile; verify peak memory, tokens/sec and Golden quality on this host. Learning rate is unchanged.",
        "inventory": inventory,
        "selection": str(devices or "auto"),
        "selected_devices": selected,
        "cuda_visible_devices": ",".join(device["launch_identifier"] for device in selected),
        "world_size": world_size,
        "microbatch": micro,
        "effective_batch_size": effective,
        "gradient_accumulation_steps": effective // (world_size * micro),
        "dtype": "bfloat16" if native_bf16 else "float16",
        "tf32": native_bf16,
        "dataloader_num_workers": workers,
        "available_cpu_count": available_cpus,
        "total_dataloader_workers": workers * world_size,
        "cpu_worker_budget_overridden": dataloader_workers is not None,
        "cpu_oversubscribed": world_size * (1 + workers) > available_cpus,
        "dataloader_pin_memory": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "attn_implementation": "sdpa",
    }


def verify_gpu_profile(profile: dict[str, Any], current: dict[str, Any] | None = None) -> None:
    if not isinstance(profile, dict) or not isinstance(profile.get("inventory"), dict):
        raise ValueError("Launch has no bound GPU inventory; prepare a new plan on the GPU host.")
    current = detect_cuda_devices() if current is None else current
    saved = profile["inventory"]
    if current.get("visible_gpu_count") != saved.get("visible_gpu_count") or current.get("inherited_cuda_visible_devices") != saved.get("inherited_cuda_visible_devices"):
        raise ValueError("CUDA visibility changed since preparation; regenerate the host launch plan.")
    selected = select_devices(current, profile.get("selection", "auto"))
    if selected != profile.get("selected_devices"):
        raise ValueError("Selected GPU identity, model or capacity changed since preparation; regenerate the host launch plan.")
    if len(selected) != profile.get("world_size") or ",".join(device["launch_identifier"] for device in selected) != profile.get("cuda_visible_devices"):
        raise ValueError("GPU profile worker count or launch mask changed after preparation.")


def apply_gpu_profile(config: dict[str, Any], profile: dict[str, Any]) -> None:
    """Apply execution settings; keep loss definition, LR, LoRA scope and QAT config."""
    runtime = config.setdefault("runtime", {})
    runtime.update(distributed="ddp", cuda_visible_devices=profile["cuda_visible_devices"],
                   world_size=profile["world_size"], gpu_profile=profile)
    config.setdefault("model", {}).update(dtype=profile["dtype"], device_map="none", attn_implementation=profile["attn_implementation"])
    training = config.setdefault("training", {})
    training.update(
        per_device_train_batch_size=profile["microbatch"],
        per_device_eval_batch_size=min(int(training.get("per_device_eval_batch_size", 1)), profile["microbatch"]),
        gradient_accumulation_steps=profile["gradient_accumulation_steps"],
        expected_effective_batch_size=profile["effective_batch_size"],
        gradient_checkpointing=profile["gradient_checkpointing"],
        gradient_checkpointing_kwargs=profile["gradient_checkpointing_kwargs"],
        require_ddp_for_multi_gpu=True, ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        tf32=profile["tf32"], dataloader_num_workers=profile["dataloader_num_workers"], dataloader_pin_memory=True,
    )
    if profile["dataloader_num_workers"] > 0:
        training.update(dataloader_persistent_workers=True, dataloader_prefetch_factor=2)
    else:
        training.pop("dataloader_persistent_workers", None)
        training.pop("dataloader_prefetch_factor", None)


def training_environment(profile: dict[str, Any], *, tensorboard_root: str = "/tensorboard") -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=profile["cuda_visible_devices"],
                       A2UI_SKIP_CUDA_DEVICE_NORMALIZE="1", TOKENIZERS_PARALLELISM="false")
    environment.setdefault("A2UI_TENSORBOARD_ROOT", tensorboard_root)
    # Each DDP rank and DataLoader worker is a separate process. Large default
    # BLAS/OpenMP pools multiply across these processes and can starve CUDA
    # feeders. Respect intentional user overrides, and print them at launch.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(name, "1")
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.pop("A2UI_CUDA_VISIBLE_DEVICES", None)
    environment.pop("A2UI_EXCLUDE_CUDA_DEVICES", None)
    return environment


def visible_launch_profile(*, model: str, num_gpus: int | None = None, launch_ids: str | None = None,
                           microbatch: int | None = None, effective_batch: int | None = None) -> dict[str, Any]:
    """Resolve legacy physical/UUID launch identifiers within visible devices."""
    inventory = detect_cuda_devices()
    if launch_ids is not None:
        tokens = [item.strip() for item in launch_ids.split(",") if item.strip()]
        if not tokens or len(set(tokens)) != len(tokens):
            raise ValueError("GPU launch IDs must be a nonempty unique list of visible identifiers.")
        selected = []
        for token in tokens:
            matches = [device for device in inventory["devices"] if token in {device["launch_identifier"], device.get("uuid")}]
            if len(matches) != 1:
                raise ValueError(f"GPU {token!r} is outside the inherited CUDA-visible assignment.")
            selected.append(matches[0]["visible_index"])
    else:
        count = inventory["visible_gpu_count"] if num_gpus is None else int(num_gpus)
        if count < 1 or count > inventory["visible_gpu_count"]:
            raise ValueError("Requested GPU count must be positive and no greater than the CUDA-visible count.")
        selected = list(range(count))
    if num_gpus is not None and len(selected) != num_gpus:
        raise ValueError("--num-gpus disagrees with the number of selected visible GPU IDs.")
    return build_gpu_profile(inventory, model=model, devices=",".join(map(str, selected)),
                             microbatch=microbatch, effective_batch=effective_batch)
