"""Resolve LoRA selectors without model execution or PEFT wrapping."""
from __future__ import annotations

import re
from typing import Any


def _retained_mobile_linear_targets(model: Any, qparams: Any) -> set[str]:
    """Bind the canonical retained projection inventory to the live text seed.

    Use the same qparams projection keys as ``prepare_qat_model``. Do not
    reconstruct the shared-KV layer boundary or consult PEFT's q/v defaults.
    """
    from torch import nn

    if getattr(getattr(model, "config", None), "model_type", None) != "gemma4_text":
        raise ValueError("Retained mobile LoRA requires the reconstructed gemma4_text seed")
    if getattr(qparams, "report", {}).get("verified") is not True:
        raise ValueError("Retained mobile LoRA requires verified qparams")
    weight_keys = tuple(qparams.trainable_projection_weight_keys())
    projection = re.compile(
        r"model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
    )
    expected = {key.removesuffix(".weight") for key in weight_keys}
    if (len(weight_keys) != 205 or len(expected) != 205
            or any(not key.endswith(".weight") for key in weight_keys)
            or any(projection.fullmatch(name) is None for name in expected)):
        raise ValueError("Retained mobile LoRA requires exactly 205 unique mapped projection weights")

    # Keep aliases visible: silently deduplicating modules can hide a malformed
    # seed where two expected projection paths share the same Linear/weight.
    entries = list(model.named_modules(remove_duplicate=False))
    modules = dict(entries)
    if len(entries) != len(modules):
        raise ValueError("Duplicate module names in retained mobile LoRA inventory")
    present = {name for name in modules if projection.fullmatch(name)}
    if present != expected:
        raise ValueError(
            "Retained mobile projection inventory mismatch: "
            f"missing={sorted(expected - present)}, extra={sorted(present - expected)}"
        )
    targets: set[str] = set()
    weight_owners: dict[int, str] = {}
    for name in sorted(expected):
        candidates = [
            candidate for candidate in (name, f"{name}.linear")
            if isinstance(modules.get(candidate), nn.Linear)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Retained mobile projection {name!r} must resolve to exactly one nn.Linear; "
                f"found={candidates}"
            )
        actual = candidates[0]
        weight = modules[actual].weight
        if id(weight) in weight_owners:
            raise ValueError(
                f"Duplicate retained mobile projection weight: {actual!r} and {weight_owners[id(weight)]!r}"
            )
        weight_owners[id(weight)] = actual
        targets.add(actual)
    return targets


def resolve_lora_config_targets(
    config: Any, model: Any, *, retained_mobile_qparams: Any | None = None,
) -> set[str]:
    """Materialize supported base-model Linear names before PEFT construction.

    Matching a Gemma projection wrapper resolves its actual ``.linear`` child.
    Passing the wrapper itself to PEFT would fail. The result is saved back into
    config.target_modules so fresh/resumed adapters share the same exact scope.
    Only the verified retained-mobile caller supplies ``retained_mobile_qparams``;
    ordinary PEFT defaults and explicit selectors retain their existing behavior.
    """
    from torch import nn

    requested = getattr(config, "target_modules", None)
    retained_targets = (
        _retained_mobile_linear_targets(model, retained_mobile_qparams)
        if retained_mobile_qparams is not None else None
    )
    if requested is None and retained_targets is not None:
        requested = retained_targets
    if requested is None:
        from peft.utils.constants import (
            TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
        )
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
        linear_names = sorted(name for name, module in modules.items() if name and isinstance(module, nn.Linear))
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        raise ValueError(
            f"LoRA selector {requested!r} matches no supported nn.Linear modules; "
            f"model_class={type(model).__name__}, model_type={model_type!r}, "
            f"linear_module_count={len(linear_names)}, "
            f"linear_module_examples={linear_names[:12]!r}. "
            "Compare the selector with the loaded model's module paths "
            "(for example model.layers vs model.language_model.layers). "
            "No fallback or target-scope widening was applied."
        )
    if retained_targets is not None and resolved != retained_targets:
        raise ValueError(
            "LoRA selector differs from the exact retained mobile projection contract: "
            f"missing={sorted(retained_targets - resolved)}, extra={sorted(resolved - retained_targets)}. "
            "No target-scope widening or partial adapter is allowed."
        )
    config.target_modules = set(resolved)
    return set(resolved)


def bind_retained_mobile_peft_targets(model: Any, expected_targets: set[str]) -> None:
    """Keep exact saved/resumed targets despite PEFT's suffix condensation.

    PEFT may compress a long explicit target set during adapter construction.
    Verify the attached scope first, then restore the equivalent exact paths
    in memory. No checkpoint files, hashes or other adapter settings change.
    """
    configs = getattr(model, "peft_config", {})
    if len(configs) != 1 or len(expected_targets) != 205:
        raise ValueError("Retained mobile PEFT requires one exact 205-projection adapter")
    adapter_name, config = next(iter(configs.items()))
    attached = [
        name.removeprefix("base_model.model.")
        for name, module in model.named_modules(remove_duplicate=False)
        if adapter_name in getattr(module, "lora_A", {})
        and adapter_name in getattr(module, "lora_B", {})
    ]
    if len(attached) != 205 or set(attached) != expected_targets:
        raise ValueError(
            "Attached PEFT scope differs from exact retained mobile targets: "
            f"count={len(attached)}, missing={sorted(expected_targets - set(attached))}, "
            f"extra={sorted(set(attached) - expected_targets)}"
        )
    config.target_modules = set(expected_targets)
