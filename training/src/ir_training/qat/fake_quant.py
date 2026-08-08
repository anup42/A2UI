from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

# Public AI Edge range compatibility is opt-in through QATSpec.quantizer.


@dataclass(frozen=True)
class QATSpec:
    """Configuration for the training-time fake quantizer.

    The quantizer uses a straight-through estimator (STE): the forward pass
    sees rounded/clamped values while the backward pass receives the identity
    gradient.  Scales are calculated from detached values on every forward
    pass, so this implementation does not add observer state to checkpoints.
    ``ste_ai_edge`` opts into the public AI Edge signed range convention; it
    is not a disclosure of Google's private QAT observer.

    ``effective_merged_weight`` makes PEFT projections fake-quantize the same
    ``base_weight + LoRA_delta`` matrix later produced by ``merge_and_unload``.
    """

    weight_bits: int = 8
    activation_bits: int = 8
    weight_symmetric: bool = True
    activation_symmetric: bool = True
    weight_per_channel: bool = True
    weight_axis: int = 0
    group_size: int | None = None
    only_base_layers: bool = True
    effective_merged_weight: bool = True
    exclude_modules: tuple[str, ...] = (
        r"(^|\.)(lm_head|embed_tokens|embed_positions|output_projection)(\.|$)",
    )
    module_quant_configs: tuple[tuple[str, int], ...] = ()
    modules_to_not_convert: tuple[str, ...] = ()
    quantize_embeddings: bool = False
    quantizer: str = "ste_absmax"
    eps: float = 1e-8

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QATSpec:
        qat = config.get("qat") if isinstance(config.get("qat"), dict) else config
        qat = _merge_public_schema(qat)
        exclude = qat.get("exclude_modules", cls.exclude_modules)
        if isinstance(exclude, str):
            exclude = (exclude,)
        elif isinstance(exclude, list | tuple):
            exclude = tuple(str(item) for item in exclude)
        else:
            exclude = cls.exclude_modules
        group_size = qat.get("group_size")
        module_quant_configs = qat.get("module_quant_configs", {})
        if isinstance(module_quant_configs, dict):
            module_quant_configs = tuple(
                (str(pattern), int(rule.get("num_bits", rule)) if isinstance(rule, dict) else int(rule))
                for pattern, rule in module_quant_configs.items()
            )
        elif isinstance(module_quant_configs, list | tuple):
            module_quant_configs = tuple(
                (str(item[0]), int(item[1]))
                for item in module_quant_configs
                if isinstance(item, list | tuple) and len(item) == 2
            )
        else:
            module_quant_configs = ()
        modules_to_not_convert = qat.get("modules_to_not_convert", ())
        if isinstance(modules_to_not_convert, str):
            modules_to_not_convert = (modules_to_not_convert,)
        elif isinstance(modules_to_not_convert, list | tuple):
            modules_to_not_convert = tuple(str(item) for item in modules_to_not_convert)
        else:
            modules_to_not_convert = ()
        return cls(
            weight_bits=int(qat.get("weight_bits", cls.weight_bits)),
            activation_bits=int(qat.get("activation_bits", cls.activation_bits)),
            weight_symmetric=bool(qat.get("weight_symmetric", cls.weight_symmetric)),
            activation_symmetric=bool(qat.get("activation_symmetric", cls.activation_symmetric)),
            weight_per_channel=bool(qat.get("weight_per_channel", cls.weight_per_channel)),
            weight_axis=int(qat.get("weight_axis", cls.weight_axis)),
            group_size=int(group_size) if group_size not in (None, "", 0) else None,
            only_base_layers=bool(qat.get("only_base_layers", cls.only_base_layers)),
            effective_merged_weight=bool(
                qat.get("effective_merged_weight", cls.effective_merged_weight)
            ),
            exclude_modules=exclude,
            module_quant_configs=module_quant_configs,
            modules_to_not_convert=modules_to_not_convert,
            quantize_embeddings=bool(qat.get("quantize_embeddings", cls.quantize_embeddings)),
            quantizer=str(qat.get("quantizer", cls.quantizer)).strip().lower(),
            eps=float(qat.get("eps", cls.eps)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def weight_bits_for_module(self, module_name: str) -> int | None:
        if any(
            _module_pattern_matches(pattern, module_name)
            for pattern in self.modules_to_not_convert
        ):
            return None
        for pattern, bits in self.module_quant_configs:
            if _module_pattern_matches(pattern, module_name):
                return bits
        return self.weight_bits

    def excludes_module(self, module_name: str) -> bool:
        return any(
            _module_pattern_matches(pattern, module_name)
            for pattern in self.exclude_modules
        )


def _module_name_candidates(module_name: str) -> tuple[str, ...]:
    """Return deployment-style aliases for a possibly PEFT-prefixed module.

    PEFT normally exposes a Gemma module such as ``lm_head`` as
    ``base_model.model.lm_head`` and can add an additional ``model`` component
    for multimodal wrappers.  Google's public precision map is written against
    the unwrapped model names and includes anchored rules such as
    ``^lm_head$``.  Matching only the raw ``named_modules`` path silently turns
    that W2 head into the default W4 rule during QAT.

    Keep the original name first, then remove only known wrapper prefixes.  We
    do not generate arbitrary suffixes, so an anchored rule for one top-level
    module cannot accidentally match a nested module with the same leaf name.
    """

    value = str(module_name).strip(".")
    if not value:
        return ("",)
    candidates = [value]
    queue = [value]
    prefixes = ("base_model.model.", "base_model.", "model.")
    while queue:
        current = queue.pop(0)
        for prefix in prefixes:
            if not current.startswith(prefix):
                continue
            stripped = current[len(prefix) :]
            if stripped and stripped not in candidates:
                candidates.append(stripped)
                queue.append(stripped)
    return tuple(candidates)


def _module_pattern_matches(pattern: str, module_name: str) -> bool:
    return any(
        re.search(pattern, candidate) is not None
        for candidate in _module_name_candidates(module_name)
    )


def _quant_bounds(bits: int, symmetric: bool) -> tuple[int, int]:
    if symmetric:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def _scale_and_zero_point(
    values: Any,
    *,
    bits: int,
    symmetric: bool,
    reduce_dims: tuple[int, ...],
    eps: float,
    quantizer: str = "ste_absmax",
) -> tuple[Any, Any, int, int]:
    """Return detached scale/zero point tensors and integer bounds."""

    import torch

    qmin, qmax = _quant_bounds(bits, symmetric)
    normalized_quantizer = str(quantizer).strip().lower()
    if normalized_quantizer not in {"ste_absmax", "ste_ai_edge"}:
        raise ValueError(
            f"Unsupported QAT quantizer {quantizer!r}; expected 'ste_absmax' or 'ste_ai_edge'."
        )
    # AI Edge's public min/max uniform quantizer uses the full signed range for
    # W2/W4, but the narrow signed range for symmetric W8 and above. Its scale
    # denominator is qmax (1, 7, 127, ...), rather than the magnitude of qmin.
    # Keep the historical ste_absmax behavior as the default so existing
    # profiles remain reproducible; opt into this mode explicitly when the
    # final converter is the public AI Edge min/max implementation.
    if normalized_quantizer == "ste_ai_edge" and symmetric:
        scale_denominator = max(float(qmax), 1.0)
        if bits >= 8:
            qmin += 1
    else:
        scale_denominator = float(max(abs(qmin), abs(qmax)))
    detached = values.detach()
    if symmetric:
        max_abs = detached.abs()
        if reduce_dims:
            max_abs = max_abs.amax(dim=reduce_dims, keepdim=True)
        scale = (max_abs / scale_denominator).clamp_min(float(eps))
        zero_point = torch.zeros_like(scale)
        return scale, zero_point, qmin, qmax

    min_value = detached
    max_value = detached
    if reduce_dims:
        min_value = min_value.amin(dim=reduce_dims, keepdim=True)
        max_value = max_value.amax(dim=reduce_dims, keepdim=True)
    scale = ((max_value - min_value) / max(qmax - qmin, 1)).clamp_min(float(eps))
    zero_point = (qmin - min_value / scale).round().clamp(qmin, qmax)
    return scale, zero_point, qmin, qmax


def _fake_quantize_with_scale(values: Any, scale: Any, zero_point: Any, qmin: int, qmax: int) -> Any:
    import torch

    quantized = torch.round(values / scale + zero_point).clamp(qmin, qmax)
    dequantized = (quantized - zero_point) * scale
    # STE: use the quantized forward value but identity d(output)/d(values).
    return values + (dequantized - values).detach()


def fake_quantize_ste(
    values: Any,
    *,
    bits: int,
    symmetric: bool = True,
    per_channel: bool = False,
    axis: int = 0,
    group_size: int | None = None,
    eps: float = 1e-8,
    quantizer: str = "ste_absmax",
) -> Any:
    """Fake-quantize a tensor with an STE backward pass.

    Weight tensors may use per-channel or group-wise scales.  Activations use
    the default per-tensor path.  Non-floating tensors and 16-bit-or-higher
    requests are returned unchanged.
    """

    if not getattr(values, "is_floating_point", lambda: False)() or bits >= 16:
        return values
    if bits < 2:
        raise ValueError(f"QAT bit width must be at least 2, got {bits}.")
    if group_size is not None and group_size <= 0:
        raise ValueError(f"QAT group_size must be positive, got {group_size}.")

    if group_size is not None:
        if values.ndim < 2 or values.shape[-1] % group_size != 0:
            # A non-divisible projection falls back to per-channel scales so
            # the model remains runnable instead of silently dropping layers.
            group_size = None
        else:
            groups = values.shape[-1] // group_size
            grouped = values.reshape(*values.shape[:-1], groups, group_size)
            reduce_dims = (grouped.ndim - 1,)
            scale, zero_point, qmin, qmax = _scale_and_zero_point(
                grouped,
                bits=bits,
                symmetric=symmetric,
                reduce_dims=reduce_dims,
                eps=eps,
                quantizer=quantizer,
            )
            return _fake_quantize_with_scale(grouped, scale, zero_point, qmin, qmax).reshape_as(values)

    if per_channel and values.ndim > 1:
        normalized_axis = axis if axis >= 0 else values.ndim + axis
        if normalized_axis < 0 or normalized_axis >= values.ndim:
            raise ValueError(f"QAT per-channel axis {axis} is invalid for shape {tuple(values.shape)}.")
        reduce_dims = tuple(index for index in range(values.ndim) if index != normalized_axis)
    else:
        reduce_dims = tuple(range(values.ndim))
    scale, zero_point, qmin, qmax = _scale_and_zero_point(
        values,
        bits=bits,
        symmetric=symmetric,
        reduce_dims=reduce_dims,
        eps=eps,
        quantizer=quantizer,
    )
    return _fake_quantize_with_scale(values, scale, zero_point, qmin, qmax)


def fake_quantize_weight(weight: Any, spec: QATSpec) -> Any:
    return fake_quantize_ste(
        weight,
        bits=spec.weight_bits,
        symmetric=spec.weight_symmetric,
        per_channel=spec.weight_per_channel,
        axis=spec.weight_axis,
        group_size=spec.group_size,
        eps=spec.eps,
        quantizer=spec.quantizer,
    )


def fake_quantize_activation(activation: Any, spec: QATSpec) -> Any:
    return fake_quantize_ste(
        activation,
        bits=spec.activation_bits,
        symmetric=spec.activation_symmetric,
        per_channel=False,
        eps=spec.eps,
        quantizer=spec.quantizer,
    )


class QATController:
    """Owns reversible fake-quantization wrappers for a model."""

    def __init__(self, spec: QATSpec):
        self.spec = spec
        self._original_forwards: dict[Any, Callable[..., Any]] = {}
        self._wrapped_names: list[str] = []
        self._wrapped_linear_names: list[str] = []
        self._wrapped_embedding_names: list[str] = []
        self._wrapped_effective_lora_names: list[str] = []
        self._wrapped_weight_bits: dict[str, int] = {}
        self._detected_lora_adapter_linear_count = 0
        self._uncovered_lora_adapter_linear_names: list[str] = []

    @property
    def wrapped_count(self) -> int:
        # Keep the count available after ``restore()`` for run metadata.
        return len(self._wrapped_names)

    @property
    def wrapped_names(self) -> tuple[str, ...]:
        return tuple(self._wrapped_names)

    @property
    def wrapped_effective_lora_count(self) -> int:
        return len(self._wrapped_effective_lora_names)

    def prepare(self, model: Any) -> QATController:
        from torch import nn
        from torch.nn import functional

        named_modules = list(model.named_modules())
        self._detected_lora_adapter_linear_count = sum(
            1
            for module_name, module in named_modules
            if isinstance(module, nn.Linear) and _is_adapter_layer_name(module_name)
        )
        effective_lora_prefixes: list[str] = []
        if self.spec.effective_merged_weight:
            for module_name, module in named_modules:
                if not _is_lora_linear_wrapper(module, nn):
                    continue
                if self.spec.excludes_module(module_name):
                    continue
                module_weight_bits = self.spec.weight_bits_for_module(module_name)
                if module_weight_bits is None:
                    continue
                adapter_names = _validate_effective_lora_wrapper(module, module_name)
                original_forward = module.forward
                module_spec = _module_spec_for_bits(self.spec, module_weight_bits)

                def qat_lora_forward(
                    input_tensor: Any,
                    *args: Any,
                    _module: Any = module,
                    _module_name: str = module_name,
                    _module_spec: QATSpec = module_spec,
                    _adapter_names: tuple[str, ...] = adapter_names,
                    **kwargs: Any,
                ) -> Any:
                    if args or kwargs:
                        raise TypeError(
                            "Effective merged-weight QAT does not support extra "
                            f"arguments for {_module_name!r}; mixed-adapter batches "
                            "would bypass the deployment-equivalent forward."
                        )
                    quantized_input = fake_quantize_activation(
                        input_tensor, _module_spec
                    )
                    effective_weight = _effective_lora_weight(
                        _module, _adapter_names
                    )
                    quantized_weight = fake_quantize_weight(
                        effective_weight, _module_spec
                    )
                    return functional.linear(
                        quantized_input,
                        quantized_weight,
                        _module.base_layer.bias,
                    )

                module.forward = qat_lora_forward
                self._original_forwards[module] = original_forward
                self._wrapped_names.append(module_name)
                self._wrapped_linear_names.append(module_name)
                self._wrapped_effective_lora_names.append(module_name)
                self._wrapped_weight_bits[module_name] = module_weight_bits
                effective_lora_prefixes.append(module_name)

        self._uncovered_lora_adapter_linear_names = [
            module_name
            for module_name, module in named_modules
            if isinstance(module, nn.Linear)
            and _is_adapter_layer_name(module_name)
            and not any(
                _is_descendant_module_name(module_name, prefix)
                for prefix in effective_lora_prefixes
            )
        ]

        for module_name, module in named_modules:
            if any(
                _is_descendant_module_name(module_name, prefix)
                for prefix in effective_lora_prefixes
            ):
                continue
            is_linear = isinstance(module, nn.Linear)
            is_embedding = isinstance(module, nn.Embedding)
            if not is_linear and not is_embedding:
                continue
            # ``only_base_layers`` means base-model weights rather than only
            # modules whose PEFT path literally contains ``base_layer``.
            # Ordinary frozen architecture linears must also see the exact
            # deployment quantization noise. Skip only LoRA adapter matrices;
            # embeddings remain controlled by ``quantize_embeddings``.
            if (
                self.spec.only_base_layers
                and not is_embedding
                and _is_adapter_layer_name(module_name)
            ):
                continue
            if is_embedding and not self.spec.quantize_embeddings:
                continue
            if self.spec.excludes_module(module_name):
                continue
            module_weight_bits = self.spec.weight_bits_for_module(module_name)
            if module_weight_bits is None:
                continue
            if module in self._original_forwards:
                continue
            original_forward = module.forward
            module_spec = _module_spec_for_bits(self.spec, module_weight_bits)

            if is_linear:

                def qat_forward(
                    input_tensor: Any,
                    *args: Any,
                    _module: Any = module,
                    _original_forward: Callable[..., Any] = original_forward,
                    _module_spec: QATSpec = module_spec,
                    **kwargs: Any,
                ) -> Any:
                    # Linear normally receives only its input. Preserve unusual
                    # callers by delegating rather than changing their signature.
                    if args or kwargs:
                        return _original_forward(input_tensor, *args, **kwargs)
                    quantized_input = fake_quantize_activation(input_tensor, _module_spec)
                    quantized_weight = fake_quantize_weight(_module.weight, _module_spec)
                    return functional.linear(quantized_input, quantized_weight, _module.bias)

            else:

                def qat_forward(
                    input_tensor: Any,
                    *args: Any,
                    _module: Any = module,
                    _original_forward: Callable[..., Any] = original_forward,
                    _module_spec: QATSpec = module_spec,
                    **kwargs: Any,
                ) -> Any:
                    # Preserve uncommon Embedding options by delegating only
                    # when a caller supplies them. The usual path uses the
                    # same fake-quantized rows as the mobile exporter.
                    if args or kwargs:
                        return _original_forward(input_tensor, *args, **kwargs)
                    quantized_weight = fake_quantize_weight(_module.weight, _module_spec)
                    return functional.embedding(
                        input_tensor,
                        quantized_weight,
                        _module.padding_idx,
                        _module.max_norm,
                        _module.norm_type,
                        _module.scale_grad_by_freq,
                        _module.sparse,
                    )

            module.forward = qat_forward
            self._original_forwards[module] = original_forward
            self._wrapped_names.append(module_name)
            self._wrapped_weight_bits[module_name] = module_weight_bits
            if is_linear:
                self._wrapped_linear_names.append(module_name)
            else:
                self._wrapped_embedding_names.append(module_name)
        return self

    def restore(self) -> None:
        for module, original_forward in self._original_forwards.items():
            module.forward = original_forward
        self._original_forwards.clear()

    def summary(self) -> dict[str, Any]:
        bit_histogram: dict[str, int] = {}
        for bits in self._wrapped_weight_bits.values():
            label = str(bits)
            bit_histogram[label] = bit_histogram.get(label, 0) + 1
        return {
            "enabled": True,
            "true_fake_quant": True,
            "wrapped_module_count": self.wrapped_count,
            "wrapped_linear_count": len(self._wrapped_linear_names),
            "wrapped_embedding_count": len(self._wrapped_embedding_names),
            "effective_merged_weight_qat_enabled": self.spec.effective_merged_weight,
            "wrapped_effective_lora_count": self.wrapped_effective_lora_count,
            "wrapped_effective_lora_names": list(
                self._wrapped_effective_lora_names
            ),
            "detected_lora_adapter_linear_count": (
                self._detected_lora_adapter_linear_count
            ),
            "uncovered_lora_adapter_linear_names": list(
                self._uncovered_lora_adapter_linear_names
            ),
            "wrapped_linear_names": list(self._wrapped_linear_names),
            "wrapped_embedding_names": list(self._wrapped_embedding_names),
            "wrapped_weight_bits_by_module": dict(self._wrapped_weight_bits),
            "wrapped_weight_bit_histogram": dict(
                sorted(bit_histogram.items(), key=lambda item: int(item[0]))
            ),
            "spec": self.spec.to_dict(),
        }


def _module_spec_for_bits(spec: QATSpec, weight_bits: int) -> QATSpec:
    if weight_bits == spec.weight_bits:
        return spec
    return QATSpec(
        **{
            **spec.to_dict(),
            "weight_bits": weight_bits,
            "module_quant_configs": (),
            "modules_to_not_convert": (),
        }
    )


def _is_lora_linear_wrapper(module: Any, nn: Any) -> bool:
    return bool(
        isinstance(getattr(module, "base_layer", None), nn.Linear)
        and callable(getattr(module, "get_delta_weight", None))
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
    )


def _active_lora_adapter_names(module: Any) -> tuple[str, ...]:
    active = getattr(module, "active_adapters", None)
    if active is None:
        active = getattr(module, "active_adapter", None)
    if isinstance(active, str):
        return (active,)
    if isinstance(active, list | tuple | set):
        return tuple(str(item) for item in active)
    return ()


def _container_has(container: Any, key: str) -> bool:
    try:
        return key in container
    except (TypeError, AttributeError):
        return False


def _container_item(container: Any, key: str, default: Any = None) -> Any:
    try:
        return container[key]
    except (KeyError, TypeError, AttributeError):
        return default


def _validate_effective_lora_wrapper(
    module: Any, module_name: str
) -> tuple[str, ...]:
    if bool(getattr(module, "disable_adapters", False)):
        raise ValueError(
            f"LoRA adapters are disabled for {module_name!r}; effective-weight QAT cannot run."
        )
    if bool(getattr(module, "merged", False)):
        raise ValueError(
            f"LoRA wrapper {module_name!r} is already merged; prepare QAT before merging."
        )
    active = _active_lora_adapter_names(module)
    selected = tuple(
        adapter
        for adapter in active
        if _container_has(module.lora_A, adapter)
        and _container_has(module.lora_B, adapter)
    )
    if not selected:
        raise ValueError(
            f"LoRA wrapper {module_name!r} has no active A/B adapter pair."
        )
    for adapter in selected:
        dropout = _container_item(getattr(module, "lora_dropout", {}), adapter)
        dropout_probability = float(getattr(dropout, "p", 0.0) or 0.0)
        if dropout_probability != 0.0:
            raise ValueError(
                f"LoRA adapter {adapter!r} in {module_name!r} has dropout="
                f"{dropout_probability}; exact post-merge QAT requires dropout=0."
            )
        use_dora = getattr(module, "use_dora", False)
        adapter_uses_dora = (
            bool(_container_item(use_dora, adapter, False))
            if not isinstance(use_dora, bool)
            else use_dora
        )
        if adapter_uses_dora:
            raise ValueError(
                f"DoRA adapter {adapter!r} in {module_name!r} is not supported by "
                "the exact effective-weight QAT path."
            )
        lora_variant = getattr(module, "lora_variant", {})
        if _container_item(lora_variant, adapter) is not None:
            raise ValueError(
                f"LoRA variant for adapter {adapter!r} in {module_name!r} is not "
                "supported by the exact effective-weight QAT path."
            )
        lora_bias = getattr(module, "lora_bias", {})
        if bool(_container_item(lora_bias, adapter, False)):
            raise ValueError(
                f"LoRA bias for adapter {adapter!r} in {module_name!r} is not "
                "supported by the exact effective-weight QAT path."
            )
    return selected


def _effective_lora_weight(module: Any, adapter_names: tuple[str, ...]) -> Any:
    base_weight = module.base_layer.weight
    effective_weight = base_weight
    for adapter in adapter_names:
        delta = module.get_delta_weight(adapter)
        if tuple(delta.shape) != tuple(base_weight.shape):
            raise ValueError(
                f"LoRA delta shape {tuple(delta.shape)} does not match base weight "
                f"shape {tuple(base_weight.shape)} for adapter {adapter!r}."
            )
        delta = delta.to(device=base_weight.device, dtype=base_weight.dtype)
        effective_weight = effective_weight + delta
    return effective_weight


def _is_descendant_module_name(module_name: str, parent_name: str) -> bool:
    if not parent_name:
        return bool(module_name)
    return module_name.startswith(parent_name + ".")


def _is_adapter_layer_name(module_name: str) -> bool:
    """Return whether a module is a trainable PEFT LoRA adapter matrix."""

    adapter_components = {
        "lora_a",
        "lora_b",
        "lora_embedding_a",
        "lora_embedding_b",
        "lora_magnitude_vector",
    }
    return any(
        component.strip().lower() in adapter_components
        for component in module_name.split(".")
    )


def _merge_public_schema(qat: dict[str, Any]) -> dict[str, Any]:
    schema_path = qat.get("schema_path")
    if not schema_path:
        return dict(qat)
    try:
        from ir_training.common.config import load_yaml
        from ir_training.qat.mobile_schema import resolve_schema_path

        schema = load_yaml(resolve_schema_path(str(schema_path)))
    except Exception as exc:
        raise ValueError(f"Unable to load qat.schema_path={schema_path!r}: {exc}") from exc
    merged = dict(schema)
    merged.update(qat)
    if "quantization_config" in schema and isinstance(schema["quantization_config"], dict):
        quantization = dict(schema["quantization_config"])
        quantization.update({key: qat[key] for key in ("module_quant_configs", "modules_to_not_convert") if key in qat})
        merged.update(quantization)
    return merged


def prepare_qat_model(model: Any, config: dict[str, Any]) -> QATController:
    """Apply configured fake quantization and fail if no base weights match."""

    controller = QATController(QATSpec.from_config(config)).prepare(model)
    if (
        controller.spec.effective_merged_weight
        and controller._uncovered_lora_adapter_linear_names
    ):
        raise ValueError(
            "Effective merged-weight QAT could not bind these LoRA adapter "
            "matrices to compatible PEFT linear wrappers: "
            + ", ".join(controller._uncovered_lora_adapter_linear_names)
        )
    if controller.wrapped_count == 0:
        raise ValueError(
            "QAT preparation matched no eligible base nn.Linear/nn.Embedding modules. "
            "Check qat.exclude_modules, qat.modules_to_not_convert, and "
            "qat.quantize_embeddings."
        )
    return controller
