from __future__ import annotations

import os
import subprocess
from collections.abc import MutableMapping


def normalize_cuda_visible_devices(env: MutableMapping[str, str] | None = None) -> str:
    """Set a safe CUDA_VISIBLE_DEVICES value before torch is imported.

    The training host can expose multiple GPUs while one device is unhealthy. Use
    all GPUs reported by nvidia-smi as queryable, and exclude any explicitly
    blocked devices, before torch is imported.
    """

    target_env = env if env is not None else os.environ
    if _truthy(target_env.get("A2UI_SKIP_CUDA_DEVICE_NORMALIZE")):
        return target_env.get("CUDA_VISIBLE_DEVICES", "")

    if _distributed_launch(target_env):
        # The parent launcher is the only authority once ranks are assigned.
        return target_env.get("CUDA_VISIBLE_DEVICES", "")

    explicit = str(target_env.get("A2UI_CUDA_VISIBLE_DEVICES", "")).strip()
    if explicit:
        selected = _apply_exclusions(explicit, target_env)
        target_env["CUDA_VISIBLE_DEVICES"] = selected
        return selected

    current = str(target_env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    detected = _detect_queryable_gpu_indices()
    if current:
        selected = _intersect_devices(current, detected) if detected else current
    else:
        selected = ",".join(detected)
    selected = _apply_exclusions(selected, target_env)
    if selected:
        target_env["CUDA_VISIBLE_DEVICES"] = selected
    return selected


def _detect_queryable_gpu_indices() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    indices: list[str] = []
    for line in result.stdout.splitlines():
        value = line.strip().split(",", 1)[0].strip()
        if value.isdigit():
            indices.append(value)
    return [index for index in _dedupe(indices) if _is_gpu_queryable(index)]


def _is_gpu_queryable(index: str) -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", index, "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0 and index in {line.strip() for line in result.stdout.splitlines()}


def _intersect_devices(current: str, detected: list[str]) -> str:
    allowed = set(detected)
    selected = [item for item in _split_devices(current) if item in allowed]
    return ",".join(selected or detected)


def _apply_exclusions(devices: str, env: MutableMapping[str, str]) -> str:
    exclude = set(_split_devices(str(env.get("A2UI_EXCLUDE_CUDA_DEVICES", "")).strip()))
    selected = [item for item in _split_devices(devices) if item not in exclude]
    return ",".join(selected)


def _split_devices(value: str) -> list[str]:
    return _dedupe(item.strip() for item in value.split(",") if item.strip())


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _distributed_launch(env: MutableMapping[str, str]) -> bool:
    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        value = str(env.get(key, "")).strip()
        if value:
            try:
                if int(value) > 1:
                    return True
            except ValueError:
                pass
    return bool(str(env.get("LOCAL_RANK", "")).strip())
