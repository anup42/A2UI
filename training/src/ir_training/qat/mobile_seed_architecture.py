"""Framework-level architecture preflight for the reconstructed Gemma 4 seed.

The mobile-seed manifest proves byte identity and deterministic reconstruction.
This module adds an independent contract against the actual Transformers model
class: instantiate ``Gemma4ForCausalLM`` on the meta device, inspect all state
keys/shapes, and compare them with Safetensors headers without loading weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from ir_training.common.config import resolve_path, training_root
from ir_training.qat.mobile_training_seed import (
    EXPECTED_TENSOR_COUNT,
    verify_configured_mobile_training_seed,
)

MIN_TRANSFORMERS_VERSION = (5, 10, 1)
EXPECTED_CONFIG_CLASS = "Gemma4TextConfig"
EXPECTED_MODEL_CLASS = "Gemma4ForCausalLM"


class MobileSeedArchitectureError(RuntimeError):
    """Raised when the framework architecture cannot be inspected safely."""


def _version_tuple(value: Any) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(value or "").strip())
    if match is None:
        return None
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def _shape_inventory_sha256(inventory: Mapping[str, tuple[int, ...]]) -> str:
    canonical = [
        {"key": key, "shape": list(inventory[key])}
        for key in sorted(inventory)
    ]
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compare_architecture_inventories(
    checkpoint: Mapping[str, tuple[int, ...]],
    framework: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Compare exact key and shape inventories without loading tensor values."""

    checkpoint_keys = set(checkpoint)
    framework_keys = set(framework)
    missing = sorted(framework_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - framework_keys)
    mismatched = [
        {
            "key": key,
            "checkpoint_shape": list(checkpoint[key]),
            "framework_shape": list(framework[key]),
        }
        for key in sorted(checkpoint_keys & framework_keys)
        if tuple(checkpoint[key]) != tuple(framework[key])
    ]
    return {
        "checkpoint_tensor_count": len(checkpoint),
        "framework_state_count": len(framework),
        "checkpoint_inventory_sha256": _shape_inventory_sha256(checkpoint),
        "framework_inventory_sha256": _shape_inventory_sha256(framework),
        "missing_checkpoint_keys": missing,
        "unexpected_checkpoint_keys": unexpected,
        "mismatched_shapes": mismatched,
        "exact": not missing and not unexpected and not mismatched,
    }


def _checkpoint_inventory(
    directory: Path,
    *,
    safe_open_fn: Callable[..., Any],
) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
    index_path = directory / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MobileSeedArchitectureError(
            f"Could not read Safetensors index {index_path}: {exc}"
        ) from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise MobileSeedArchitectureError(
            f"Safetensors index has no weight_map: {index_path}"
        )
    declared = {str(key): str(value) for key, value in weight_map.items()}
    inventory: dict[str, tuple[int, ...]] = {}
    observed_shard_by_key: dict[str, str] = {}
    dtype_histogram: dict[str, int] = {}
    for shard_name in sorted(set(declared.values())):
        shard_path = (directory / shard_name).resolve()
        try:
            shard_path.relative_to(directory.resolve())
        except ValueError as exc:
            raise MobileSeedArchitectureError(
                f"Unsafe shard path in index: {shard_name}"
            ) from exc
        if not shard_path.is_file():
            raise MobileSeedArchitectureError(f"Missing checkpoint shard: {shard_path}")
        try:
            with safe_open_fn(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    key = str(key)
                    if key in inventory:
                        raise MobileSeedArchitectureError(
                            f"Duplicate key across checkpoint shards: {key}"
                        )
                    tensor_slice = handle.get_slice(key)
                    inventory[key] = tuple(
                        int(item) for item in tensor_slice.get_shape()
                    )
                    observed_shard_by_key[key] = shard_name
                    dtype = str(tensor_slice.get_dtype()).upper()
                    dtype_histogram[dtype] = dtype_histogram.get(dtype, 0) + 1
        except MobileSeedArchitectureError:
            raise
        except Exception as exc:
            raise MobileSeedArchitectureError(
                f"Could not inspect checkpoint shard {shard_path}: {exc}"
            ) from exc
    if set(inventory) != set(declared):
        raise MobileSeedArchitectureError(
            "Safetensors headers differ from the index weight_map."
        )
    storage_mismatches = [
        key
        for key, shard_name in declared.items()
        if observed_shard_by_key.get(key) != shard_name
    ]
    if storage_mismatches:
        raise MobileSeedArchitectureError(
            "Safetensors index maps keys to the wrong shard: "
            + ", ".join(storage_mismatches[:12])
        )
    return inventory, dtype_histogram


def validate_mobile_seed_architecture(
    model_config: Mapping[str, Any],
    *,
    base: str | Path | None = None,
    verified_seed: Mapping[str, Any] | None = None,
    torch_module: Any | None = None,
    transformers_module: Any | None = None,
    safe_open_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Validate exact Gemma 4 framework keys/shapes on the meta device."""

    anchor = Path(base).resolve() if base is not None else training_root()
    seed = (
        dict(verified_seed)
        if verified_seed is not None
        else verify_configured_mobile_training_seed(
            dict(model_config),
            base=anchor,
            require_materialized=True,
        )
    )
    if not seed.get("verified"):
        failed = [
            name
            for name, passed in seed.get("checks", {}).items()
            if not passed
        ]
        raise MobileSeedArchitectureError(
            "Mobile training seed failed before architecture preflight: "
            + ", ".join(failed)
        )
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception as exc:  # pragma: no cover - dependency failure path
            raise MobileSeedArchitectureError(
                "Install training/requirements-gemma4-qat.txt before the architecture preflight."
            ) from exc
    if transformers_module is None:
        try:
            import transformers as transformers_module  # type: ignore[no-redef]
        except Exception as exc:  # pragma: no cover - dependency failure path
            raise MobileSeedArchitectureError(
                "Install training/requirements-gemma4-qat.txt before the architecture preflight."
            ) from exc
    if safe_open_fn is None:
        try:
            from safetensors import safe_open as safe_open_fn  # type: ignore[no-redef]
        except Exception as exc:  # pragma: no cover - dependency failure path
            raise MobileSeedArchitectureError(
                "Install Safetensors before the architecture preflight."
            ) from exc

    source = resolve_path(str(model_config.get("model_source") or ""), anchor)
    if not source.is_dir():
        raise MobileSeedArchitectureError(f"Model source is not a directory: {source}")
    version_text = str(getattr(transformers_module, "__version__", ""))
    parsed_version = _version_tuple(version_text)
    try:
        config = transformers_module.AutoConfig.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
        )
        with torch_module.device("meta"):
            model = transformers_module.AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=False,
            )
    except Exception as exc:
        raise MobileSeedArchitectureError(
            f"Could not instantiate the local Gemma 4 architecture on meta: {exc}"
        ) from exc

    state = model.state_dict()
    framework_inventory = {
        str(key): tuple(int(item) for item in value.shape)
        for key, value in state.items()
    }
    all_framework_state_is_meta = all(
        str(getattr(getattr(value, "device", None), "type", "")) == "meta"
        for value in state.values()
    )
    config_class = type(config).__name__
    model_class = type(model).__name__
    del state
    del model

    checkpoint_inventory, dtype_histogram = _checkpoint_inventory(
        source,
        safe_open_fn=safe_open_fn,
    )
    comparison = compare_architecture_inventories(
        checkpoint_inventory,
        framework_inventory,
    )
    checks = {
        "mobile_seed_verified": seed.get("verified") is True,
        "preflight_required_by_config": model_config.get(
            "architecture_preflight_required"
        )
        is True,
        "exact_checkpoint_keys_required": model_config.get(
            "require_exact_checkpoint_keys"
        )
        is True,
        "transformers_minimum_version": bool(
            parsed_version and parsed_version >= MIN_TRANSFORMERS_VERSION
        ),
        "config_class_exact": config_class == EXPECTED_CONFIG_CLASS,
        "model_class_exact": model_class == EXPECTED_MODEL_CLASS,
        "config_model_type_exact": str(getattr(config, "model_type", ""))
        == "gemma4_text",
        "config_architecture_exact": list(
            getattr(config, "architectures", None) or []
        )
        == [EXPECTED_MODEL_CLASS],
        "framework_state_on_meta": all_framework_state_is_meta,
        "checkpoint_tensor_count": len(checkpoint_inventory)
        == EXPECTED_TENSOR_COUNT,
        "framework_state_count": len(framework_inventory)
        == EXPECTED_TENSOR_COUNT,
        "checkpoint_all_bf16": dtype_histogram == {"BF16": EXPECTED_TENSOR_COUNT},
        "key_and_shape_inventory_exact": comparison["exact"] is True,
        "inventory_hashes_match": comparison["checkpoint_inventory_sha256"]
        == comparison["framework_inventory_sha256"],
        "no_weight_load_forward_or_training": True,
    }
    return {
        "report_version": 1,
        "required": True,
        "model_id": model_config.get("model_id"),
        "model_source": str(source),
        "seed_manifest_sha256": seed.get("manifest_sha256"),
        "transformers_version": version_text,
        "minimum_transformers_version": ".".join(
            str(item) for item in MIN_TRANSFORMERS_VERSION
        ),
        "config_class": config_class,
        "model_class": model_class,
        "checkpoint_dtype_histogram": dtype_histogram,
        "comparison": comparison,
        "checks": checks,
        "verified": all(checks.values()),
        "model_weights_loaded": False,
        "forward_executed": False,
        "training_executed": False,
        "private_google_recipe_recovered": False,
    }


__all__ = [
    "MIN_TRANSFORMERS_VERSION",
    "MobileSeedArchitectureError",
    "compare_architecture_inventories",
    "validate_mobile_seed_architecture",
]
