"""Lightweight environment checks for the opt-in sharded training lane only."""
from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

NVTX_VERSION = "0.2.15"
NVTX_REPAIR = (
    "Activate the dedicated sharded venv, then run: "
    "python -m pip install --ignore-installed --no-deps nvtx==0.2.15. "
    "Do not uninstall system packages or disable NVTX/preflight."
)


def probe_nvtx_compatibility() -> dict[str, Any]:
    """Exercise the exact pinned DeepSpeed NVTX domain push/pop path.

    No model, optimizer, distributed process group, or training tensors are
    created here. A no-profiler DummyDomain is valid, but must support the same
    keyword API as a real NVTX Domain. Do not catch errors and fall back to the
    old torch.cuda.nvtx path: that would hide the original compatibility bug.
    """
    module_path: str | None = None
    try:
        nvtx = importlib.import_module("nvtx")
        version = importlib.metadata.version("nvtx")
        module_path = str(Path(nvtx.__file__).resolve())
        prefix = Path(sys.prefix).resolve()
        if prefix == Path(sys.base_prefix).resolve():
            raise RuntimeError("Sharded NVTX must be installed in a dedicated Python venv")
        if version != NVTX_VERSION:
            raise RuntimeError(f"Expected nvtx {NVTX_VERSION}, found {version}")
        if not Path(module_path).is_relative_to(prefix):
            raise RuntimeError("NVTX is inherited from outside the active venv")
        distribution = importlib.metadata.distribution("nvtx")
        recorded_module = Path(distribution.locate_file("nvtx/__init__.py")).resolve()
        if recorded_module != Path(module_path):
            raise RuntimeError("Imported NVTX does not match the selected distribution metadata")
        if nvtx.enabled() is not True:
            raise RuntimeError("NVTX is disabled; remove NVTX_DISABLE instead of bypassing the check")

        domain = nvtx.get_domain("DeepSpeed")
        # .14's DummyDomain rejects these keywords even without a profiler.
        domain.push_range(message="a2ui-sharded-nvtx-direct-probe", category=None)
        domain.pop_range()

        ds_accelerator = importlib.import_module("deepspeed.accelerator").get_accelerator()
        ds_nvtx = importlib.import_module("deepspeed.utils.nvtx")
        if (ds_accelerator.device_name() != "cuda"
                or getattr(ds_accelerator, "supports_nvtx_domain", False) is not True):
            raise RuntimeError("DeepSpeed must use the CUDA NVTX domain path")
        if getattr(ds_nvtx, "enable_nvtx", False) is not True:
            raise RuntimeError("DeepSpeed NVTX instrumentation is disabled")
        # These are the same pinned wrappers called by instrument_w_nvtx in
        # ZeRO-2 backward/step, not a signature-only inspection or substitute.
        ds_nvtx._range_push(ds_accelerator, "a2ui-sharded-deepspeed-nvtx-probe")
        ds_nvtx._range_pop(ds_accelerator)
    except Exception as exc:
        raise RuntimeError(
            f"Sharded NVTX compatibility probe failed (module={module_path!r}): "
            f"{type(exc).__name__}: {exc}. {NVTX_REPAIR}"
        ) from exc
    return {
        "passed": True, "version": version, "module": module_path,
        "venv_prefix": str(prefix), "venv_local": True,
        "domain_type": type(domain).__name__, "direct_domain_push_pop": True,
        "deepspeed_range_push_pop": True, "instrumentation_enabled": True,
        "model_loaded": False, "training_executed": False,
    }


def check_sharded_environment(*, world_size: int, effective_batch: int) -> dict[str, Any]:
    """Check dependencies and accumulation configuration without loading a model.

    This is not the memory/optimizer preflight. No Accelerator, DeepSpeed
    engine, or process group is constructed; the production gate still runs.
    """
    if type(world_size) is not int or world_size not in {2, 4, 8}:
        raise ValueError("Sharded full-QAT world_size must be 2, 4, or 8")
    if (type(effective_batch) is not int or effective_batch < world_size
            or effective_batch % world_size):
        raise ValueError("effective_batch must be a positive multiple of world_size")
    from ir_training.train.sharded_contract import (
        build_deepspeed_config,
        configure_sharded_accumulation,
        validate_sharded_runtime,
    )

    runtime = validate_sharded_runtime()
    from accelerate.utils import GradientAccumulationPlugin

    accumulation = effective_batch // world_size
    config = {
        "distributed_backend": "sharded", "optim": "adamw_torch",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": accumulation,
        "expected_effective_batch_size": effective_batch,
    }
    plugin = GradientAccumulationPlugin(num_steps=1)
    configure_sharded_accumulation(
        {"gradient_accumulation_plugin": plugin}, config,
        trainer_accumulation_steps=accumulation,
    )
    ds_config = build_deepspeed_config(config, world_size)
    return {
        "passed": True, "runtime": runtime,
        "accumulation_configuration": {
            "world_size": world_size, "per_device_train_batch_size": 1,
            "effective_batch": effective_batch, "trainer": accumulation,
            "accelerate_plugin": plugin.num_steps,
            "deepspeed": ds_config["gradient_accumulation_steps"],
        },
        "model_loaded": False, "training_executed": False,
        "distributed_engine_created": False, "memory_preflight_passed": False,
    }
