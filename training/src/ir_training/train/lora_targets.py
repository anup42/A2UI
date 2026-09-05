"""Resolve LoRA selectors without model execution or PEFT wrapping."""
from __future__ import annotations

import re
from typing import Any


def resolve_lora_config_targets(config: Any, model: Any) -> set[str]:
    """Materialize supported base-model Linear names before PEFT construction.

    Matching a Gemma projection wrapper resolves its actual ``.linear`` child.
    Passing the wrapper itself to PEFT would fail. The result is saved back into
    config.target_modules so fresh/resumed adapters share the same exact scope.
    """
    import torch.nn as nn

    requested = getattr(config, "target_modules", None)
    if requested is None:
        from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        requested = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(model_type)
        if requested is None:
            raise ValueError(f"Cannot resolve PEFT default targets for model_type={model_type!r}; choose explicit language targets")
    if getattr(config, "layers_to_transform", None) is not None:
        raise ValueError("Resolve layer selection into an explicit target regex before materializing LoRA modules")
    all_linear = isinstance(requested, str) and requested.strip().lower() == "all-linear"
    if not isinstance(requested, (str, list, tuple, set)) or not requested:
        raise ValueError("LoRA target_modules must be a nonempty regex or collection of module names")

    def matches(selector: Any, name: str) -> bool:
        if selector is None:
            return False
        if isinstance(selector, str):
            return re.fullmatch(selector, name) is not None
        return any(name == suffix or name.endswith("." + suffix) for suffix in selector)

    output_module = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    modules = dict(model.named_modules())
    output_names = {name for name, module in modules.items() if module is output_module}
    excluded = getattr(config, "exclude_modules", None)

    def is_excluded(name: str) -> bool:
        parts = name.split(".")
        return any(matches(excluded, ".".join(parts[:index])) for index in range(1, len(parts) + 1))

    resolved: set[str] = set()
    for name, module in modules.items():
        if not name or is_excluded(name):
            continue
        if not all_linear and not matches(requested, name):
            continue
        if isinstance(module, nn.Linear):
            actual_name, actual_module = name, module
        else:
            actual_name, actual_module = f"{name}.linear", modules.get(f"{name}.linear")
            if not isinstance(actual_module, nn.Linear):
                continue
        if is_excluded(actual_name):
            continue
        if all_linear and (actual_module is output_module or "lm_head" in actual_name.split(".")
                           or any(actual_name == prefix or actual_name.startswith(prefix + ".") for prefix in output_names)):
            continue
        resolved.add(actual_name)
    if not resolved:
        raise ValueError(f"LoRA selector {requested!r} matches no supported nn.Linear modules; inspect the actual model module inventory")
    config.target_modules = set(resolved)
    return set(resolved)
