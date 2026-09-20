"""Scoped SDPA backend policy and best-effort diagnostics, never GPU recovery."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from ir_training.common.progress import log

_ACTIVE_POLICY: ContextVar[dict | None] = ContextVar("sft_sdpa_policy", default=None)


def allocator_environment() -> dict[str, str]:
    """Report requested allocator settings, without claiming runtime support."""
    return {name: os.environ[name] for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
            if name in os.environ}


def _flags(cuda_backend: Any) -> dict:
    return {name: bool(getattr(cuda_backend, f"{name}_sdp_enabled")())
            if callable(getattr(cuda_backend, f"{name}_sdp_enabled", None)) else None
            for name in ("flash", "mem_efficient", "math", "cudnn")}


@contextmanager
def sdpa_policy(config: dict, *, torch_module: Any = None):
    """Disable only cuDNN attention; keep existing Flash/efficient/math choices.

    This covers forward AND backward, including checkpoint recomputation and
    in-process Golden callbacks. A forward-only sdpa_kernel context is not
    sufficient. Backend state is restored when the run returns or fails.
    """
    if torch_module is None:
        import torch as torch_module
    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    disable = training.get("disable_cudnn_sdpa", True)
    if type(disable) is not bool:
        raise ValueError("training.disable_cudnn_sdpa must be a boolean")
    backend = torch_module.backends.cuda
    before = _flags(backend)
    setter = getattr(backend, "enable_cudnn_sdp", None)
    changed = disable and before["cudnn"] is not None and callable(setter)
    if disable and before["cudnn"] is True and not callable(setter):
        raise RuntimeError("Cannot disable cuDNN SDPA in this PyTorch build; use a supported training environment")
    try:
        if changed:
            setter(False)
        effective = _flags(backend)
        if disable and effective["cudnn"] is True:
            raise RuntimeError("cuDNN SDPA remained enabled after applying the safe training policy")
        report = {"disable_cudnn_sdpa": disable, "before": before, "effective": effective,
                  "torch_version": str(torch_module.__version__),
                  "cuda_build": getattr(torch_module.version, "cuda", None),
                  "cudnn_version": torch_module.backends.cudnn.version(),
                  "allocator_environment": allocator_environment(),
                  "scope": "complete_sft_run_including_backward_and_in_process_evaluation"}
        log(f"SDPA runtime [rank {os.environ.get('RANK', '0')}]: {json.dumps(report, sort_keys=True)}")
        token = _ACTIVE_POLICY.set(report)
        try:
            yield report
        finally:
            _ACTIVE_POLICY.reset(token)
    finally:
        if changed:
            setter(before["cudnn"])


def training_attention_policy(function):
    @wraps(function)
    def wrapped(config, *args, **kwargs):
        with sdpa_policy(config):
            return function(config, *args, **kwargs)
    return wrapped


def active_attention_policy() -> dict:
    return dict(_ACTIVE_POLICY.get() or {})


def cuda_memory_snapshot(*, torch_module: Any = None) -> dict:
    """Never synchronize, reset, retry, or raise while handling a CUDA failure."""
    report: dict[str, Any] = {"rank": os.environ.get("RANK", "0"), "local_rank": os.environ.get("LOCAL_RANK", "0"),
                              "allocator_environment": allocator_environment()}
    try:
        if torch_module is None:
            import torch as torch_module
        report.update(torch_version=str(torch_module.__version__), cuda_build=getattr(torch_module.version, "cuda", None),
                      cudnn_version=torch_module.backends.cudnn.version(), cuda_available=torch_module.cuda.is_available())
        if not report["cuda_available"]:
            return report
        device = torch_module.cuda.current_device()
        report.update(device=device, device_name=torch_module.cuda.get_device_name(device))
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            try:
                report[name + "_bytes"] = int(getattr(torch_module.cuda, name)(device))
            except Exception as exc:  # noqa: BLE001 - diagnostics must never mask the original CUDA failure
                report[name + "_error"] = repr(exc)
        try:
            free, total = torch_module.cuda.mem_get_info(device)
            report.update(free_bytes=int(free), total_bytes=int(total))
        except Exception as exc:  # noqa: BLE001 - the CUDA context may already be unusable
            report["free_memory_error"] = repr(exc)
    except Exception as exc:  # noqa: BLE001 - best-effort reporting only, never recovery
        report["diagnostic_error"] = repr(exc)
    return report
