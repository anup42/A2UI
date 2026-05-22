from __future__ import annotations

import os
from collections.abc import MutableMapping


def normalize_cuda_visible_devices(env: MutableMapping[str, str] | None = None) -> str:
    """Set a safe CUDA_VISIBLE_DEVICES value before torch is imported.

    The training host can expose multiple GPUs while one device is unhealthy.
    If CUDA_VISIBLE_DEVICES includes that bad device, PyTorch can fail CUDA
    initialization and report zero GPUs even though nvidia-smi lists GPUs.
    """

    target_env = env if env is not None else os.environ
    if _truthy(target_env.get("A2UI_SKIP_CUDA_DEVICE_NORMALIZE")):
        return target_env.get("CUDA_VISIBLE_DEVICES", "")

    explicit = str(target_env.get("A2UI_CUDA_VISIBLE_DEVICES", "")).strip()
    if explicit:
        target_env["CUDA_VISIBLE_DEVICES"] = explicit
        return explicit

    current = str(target_env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    allow_multi = _truthy(target_env.get("A2UI_ALLOW_MULTI_GPU_VISIBLE"))
    if not current or ("," in current and not allow_multi):
        target_env["CUDA_VISIBLE_DEVICES"] = "0"
        return "0"
    return current


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
