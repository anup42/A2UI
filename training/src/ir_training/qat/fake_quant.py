from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

# Public AI Edge range compatibility is opt-in through QATSpec.quantizer.


# These values are taken from the public ai-edge-quantizer 0.8.0 uniform
# quantizer used by the verified LiteRT-LM export environment.  Keep the
# version visible in run metadata: this is a public implementation contract,
# not a claim about Google's private QAT observer or calibration recipe.
AI_EDGE_REFERENCE_QUANTIZER_VERSION = "0.8.0"
AI_EDGE_MIN_SCALE = 1e-9
AI_EDGE_SCALE_COMPUTE_DTYPE = "float32"
AI_EDGE_BLOCKWISE_SCALE_CAST = ("bfloat16", "float16", "float32")
AI_EDGE_ROUNDING = "ties_to_even"
MOBILE_SRQ_REFERENCE = (
    "huggingface/transformers@c587bc884db2c2e31fc2b8102314656b17aa07b1:"
    "src/transformers/integrations/gemma_quant.py:apply_srq"
)


@dataclass(frozen=True)
class QATSpec:
    """Configuration for the training-time fake quantizer.

    The quantizer uses a straight-through estimator (STE): the forward pass
    sees rounded/clamped values. Legacy profiles calculate scales from detached
    values on every forward. ``scale_mode=retained_mobile`` instead consumes
    the exact scale tensors published with Google's packed mobile checkpoint;
    those immutable tensors are part of checkpoint provenance and export.
    ``ste_ai_edge`` opts into the public AI Edge signed range convention; it is
    not a disclosure of Google's private QAT observer.

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
    module_group_sizes: tuple[tuple[str, int], ...] = ()
    modules_to_not_convert: tuple[str, ...] = ()
    quantize_embeddings: bool = False
    quantizer: str = "ste_absmax"
    scale_mode: str = "dynamic"
    mobile_qparams_contract: str | None = None
    expected_effective_lora_modules: int | None = None
    fixed_scale_required: bool = False
    fixed_activation_scale_required: bool = False
    effective_lora_only: bool = False
    ste_gradient: str = "identity"
    # Separate the released mobile activation consumer from the generic
    # AI Edge *weight* quantizer. Old configs retain their historical behavior.
    activation_quantizer: str = "legacy"
    simulate_frozen_activations: bool = False
    expected_frozen_activation_modules: int | None = None
    require_lora_trainable_scope: bool = False
    eps: float = 1e-8

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QATSpec:
        qat = config.get("qat") if isinstance(config.get("qat"), dict) else config
        qat = _merge_public_schema(qat)
        quantizer = str(qat.get("quantizer", cls.quantizer)).strip().lower()
        scale_mode = str(qat.get("scale_mode", cls.scale_mode)).strip().lower()
        ste_gradient = str(qat.get("ste_gradient", cls.ste_gradient)).strip().lower()
        default_eps = (
            AI_EDGE_MIN_SCALE
            if quantizer == "ste_ai_edge"
            else cls.eps
        )
        exclude = qat.get("exclude_modules", cls.exclude_modules)
        if isinstance(exclude, str):
            exclude = (exclude,)
        elif isinstance(exclude, list | tuple):
            exclude = tuple(str(item) for item in exclude)
        else:
            exclude = cls.exclude_modules
        group_size = qat.get("group_size")
        module_quant_configs = qat.get("module_quant_configs", {})
        module_group_sizes: tuple[tuple[str, int], ...] = ()
        if isinstance(module_quant_configs, dict):
            module_group_sizes = tuple(
                (str(pattern), int(rule["group_size"]))
                for pattern, rule in module_quant_configs.items()
                if isinstance(rule, dict)
                and rule.get("group_size") not in (None, "", 0)
            )
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
            module_group_sizes=module_group_sizes,
            modules_to_not_convert=modules_to_not_convert,
            quantize_embeddings=bool(qat.get("quantize_embeddings", cls.quantize_embeddings)),
            quantizer=quantizer,
            scale_mode=scale_mode,
            mobile_qparams_contract=(
                str(qat.get("mobile_qparams_contract")).strip()
                if qat.get("mobile_qparams_contract")
                else None
            ),
            expected_effective_lora_modules=(
                int(qat.get("expected_effective_lora_modules"))
                if qat.get("expected_effective_lora_modules") is not None
                else None
            ),
            fixed_scale_required=bool(
                qat.get("fixed_scale_required", cls.fixed_scale_required)
            ),
            fixed_activation_scale_required=bool(
                qat.get(
                    "fixed_activation_scale_required",
                    cls.fixed_activation_scale_required,
                )
            ),
            effective_lora_only=bool(
                qat.get("effective_lora_only", cls.effective_lora_only)
            ),
            ste_gradient=ste_gradient,
            activation_quantizer=str(qat.get("activation_quantizer", "legacy")).strip().lower(),
            simulate_frozen_activations=bool(qat.get("simulate_frozen_activations", False)),
            expected_frozen_activation_modules=(
                int(qat["expected_frozen_activation_modules"])
                if qat.get("expected_frozen_activation_modules") is not None else None
            ),
            require_lora_trainable_scope=bool(qat.get("require_lora_trainable_scope", False)),
            eps=float(qat.get("eps", default_eps)),
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

    def group_size_for_module(self, module_name: str) -> int | None:
        for pattern, group_size in self.module_group_sizes:
            if _module_pattern_matches(pattern, module_name):
                return group_size
        return self.group_size

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
    # ``Gemma4ForCausalLM`` is text-only, so its modules are exposed as
    # ``model.layers.*`` / ``model.embed_tokens`` rather than the multimodal
    # checkpoint's ``language_model.*`` paths used by Google's public mobile
    # precision map. Add only the known text-model roots; without these aliases
    # W2/W4 matrices silently receive the default W8 fake quantizer.
    text_roots = (
        "layers.",
        "embed_tokens",
        "embed_tokens_per_layer",
        "per_layer_model_projection",
    )
    for candidate in tuple(candidates):
        if candidate.startswith("language_model."):
            continue
        if candidate.startswith(text_roots):
            alias = "language_model." + candidate
            if alias not in candidates:
                candidates.append(alias)
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
    # It computes min/max and scales in FLOAT32 even when the checkpoint source
    # is BF16.  This matters during QAT: deriving the scale in BF16 can move
    # multiple W8 codes relative to the final converter.
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
    if normalized_quantizer == "ste_ai_edge":
        detached = detached.to(dtype=torch.float32)
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


def _quant_bounds_for_quantizer(
    bits: int, symmetric: bool, quantizer: str
) -> tuple[int, int]:
    qmin, qmax = _quant_bounds(bits, symmetric)
    if str(quantizer).strip().lower() == "ste_ai_edge" and symmetric and bits >= 8:
        qmin += 1
    return qmin, qmax


def _validate_override_tensor(
    value: Any,
    *,
    values: Any,
    expected_shape: tuple[int, ...],
    label: str,
    positive: bool,
) -> Any:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().to(device=values.device, dtype=torch.float32)
    # Retained grouped scales are stored without the final singleton dimension.
    if tuple(tensor.shape) == expected_shape[:-1] and expected_shape[-1] == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim == 0:
        if label == "zero_point_override":
            tensor = tensor.reshape(tuple(1 for _ in expected_shape)).expand(
                expected_shape
            )
        elif expected_shape == tuple(1 for _ in range(values.ndim)):
            tensor = tensor.reshape(expected_shape)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{label} shape {tuple(tensor.shape)} does not match required "
            f"broadcast shape {expected_shape} for values {tuple(values.shape)}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} contains NaN or Inf.")
    if positive and not bool((tensor > 0).all()):
        raise ValueError(f"{label} must contain only positive values.")
    return tensor


def _fake_quantize_with_scale(
    values: Any,
    scale: Any,
    zero_point: Any,
    qmin: int,
    qmax: int,
    *,
    ste_gradient: str = "identity",
) -> Any:
    import torch

    quantized = torch.round(values / scale + zero_point).clamp(qmin, qmax)
    dequantized = (quantized - zero_point) * scale
    # FLOAT32 scale computation must not promote a BF16/FP16 training model's
    # forward pass.  The integer codes and scales match the converter; the
    # simulated dequantized value returns to the model's original dtype.
    if dequantized.dtype != values.dtype:
        dequantized = dequantized.to(dtype=values.dtype)
    normalized_gradient = str(ste_gradient).strip().lower()
    if normalized_gradient == "identity":
        # Historical STE: use the quantized forward value but identity
        # d(output)/d(values), including outside the representable interval.
        return values + (dequantized - values).detach()
    if normalized_gradient == "clipped":
        # Saturation-aware STE. The forward value is still exactly quantized,
        # but values outside the retained code range receive no gradient that
        # would push them farther beyond an immutable mobile scale.
        lower = (qmin - zero_point) * scale
        upper = (qmax - zero_point) * scale
        inside = ((values >= lower) & (values <= upper)).to(dtype=values.dtype)
        surrogate = values * inside
        return surrogate + (dequantized - surrogate).detach()
    raise ValueError(
        f"Unsupported ste_gradient {ste_gradient!r}; expected 'identity' or 'clipped'."
    )


def _round_ai_edge_blockwise_scale(scale: Any) -> Any:
    """Apply the public AI Edge blockwise scale storage conversion.

    ai-edge-quantizer 0.8.0 computes the scale in FLOAT32, rounds it through
    BF16, stores it as FP16, and then continues quantization with the resulting
    FLOAT32 value.  The explicit conversion here keeps grouped embedding QAT
    numerically aligned with the final public converter.
    """

    import torch

    return scale.to(torch.bfloat16).to(torch.float16).to(torch.float32)


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
    scale_override: Any | None = None,
    zero_point_override: Any | None = None,
    ste_gradient: str = "identity",
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
    if zero_point_override is not None and scale_override is None:
        raise ValueError("zero_point_override requires scale_override.")

    if group_size is not None:
        if values.ndim < 2:
            raise ValueError(
                "Grouped QAT requires a rank-2-or-higher weight tensor, got "
                f"shape {tuple(values.shape)}."
            )
        if values.shape[-1] % group_size != 0:
            raise ValueError(
                "Grouped QAT cannot reproduce the deployment layout because "
                f"the final dimension {values.shape[-1]} is not divisible by "
                f"group_size={group_size}."
            )
        groups = values.shape[-1] // group_size
        grouped = values.reshape(*values.shape[:-1], groups, group_size)
        reduce_dims = (grouped.ndim - 1,)
        if scale_override is None:
            scale, zero_point, qmin, qmax = _scale_and_zero_point(
                grouped,
                bits=bits,
                symmetric=symmetric,
                reduce_dims=reduce_dims,
                eps=eps,
                quantizer=quantizer,
            )
            if str(quantizer).strip().lower() == "ste_ai_edge":
                scale = _round_ai_edge_blockwise_scale(scale)
        else:
            expected_shape = tuple(grouped.shape[:-1]) + (1,)
            scale = _validate_override_tensor(
                scale_override,
                values=grouped,
                expected_shape=expected_shape,
                label="scale_override",
                positive=True,
            )
            zero_point = _validate_override_tensor(
                0.0 if zero_point_override is None else zero_point_override,
                values=grouped,
                expected_shape=expected_shape,
                label="zero_point_override",
                positive=False,
            )
            qmin, qmax = _quant_bounds_for_quantizer(bits, symmetric, quantizer)
        return _fake_quantize_with_scale(
            grouped,
            scale,
            zero_point,
            qmin,
            qmax,
            ste_gradient=ste_gradient,
        ).reshape_as(values)

    if per_channel and values.ndim > 1:
        normalized_axis = axis if axis >= 0 else values.ndim + axis
        if normalized_axis < 0 or normalized_axis >= values.ndim:
            raise ValueError(f"QAT per-channel axis {axis} is invalid for shape {tuple(values.shape)}.")
        reduce_dims = tuple(index for index in range(values.ndim) if index != normalized_axis)
    else:
        reduce_dims = tuple(range(values.ndim))
    if scale_override is None:
        scale, zero_point, qmin, qmax = _scale_and_zero_point(
            values,
            bits=bits,
            symmetric=symmetric,
            reduce_dims=reduce_dims,
            eps=eps,
            quantizer=quantizer,
        )
    else:
        if per_channel and values.ndim > 1:
            expected_shape = tuple(
                values.shape[index] if index == normalized_axis else 1
                for index in range(values.ndim)
            )
        else:
            expected_shape = tuple(1 for _ in range(values.ndim))
        scale = _validate_override_tensor(
            scale_override,
            values=values,
            expected_shape=expected_shape,
            label="scale_override",
            positive=True,
        )
        zero_point = _validate_override_tensor(
            0.0 if zero_point_override is None else zero_point_override,
            values=values,
            expected_shape=expected_shape,
            label="zero_point_override",
            positive=False,
        )
        qmin, qmax = _quant_bounds_for_quantizer(bits, symmetric, quantizer)
    return _fake_quantize_with_scale(
        values,
        scale,
        zero_point,
        qmin,
        qmax,
        ste_gradient=ste_gradient,
    )


def fake_quantize_weight(
    weight: Any,
    spec: QATSpec,
    *,
    scale_override: Any | None = None,
    zero_point_override: Any | None = None,
) -> Any:
    return fake_quantize_ste(
        weight,
        bits=spec.weight_bits,
        symmetric=spec.weight_symmetric,
        per_channel=spec.weight_per_channel,
        axis=spec.weight_axis,
        group_size=spec.group_size,
        eps=spec.eps,
        quantizer=spec.quantizer,
        scale_override=scale_override,
        zero_point_override=zero_point_override,
        ste_gradient=spec.ste_gradient,
    )


def mobile_srq_ste(
    values: Any, scale: Any, *, ste_gradient: str = "clipped", validate_scale: bool = True,
) -> Any:
    """Published mobile HF A8 forward, with an explicitly independent STE.

    This is a pinned *HF consumer* oracle, not a claim of native kernel parity
    or Google's unpublished training backward rule. Scales stay immutable;
    arithmetic casts them to the activation dtype just as ``apply_srq`` does.
    Zero scales bypass quantization. Validation can be skipped only by wrappers
    that have already validated their immutable retained-scale contract.
    """
    import torch

    if not values.is_floating_point():
        return values
    if scale is None:
        raise ValueError("Mobile SRQ requires an explicit retained activation scale (zero means bypass).")
    tensor = torch.as_tensor(scale, device=values.device).detach()
    if tensor.numel() != 1:
        raise ValueError("Mobile SRQ requires a scalar activation scale.")
    if validate_scale and (not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any())):
        raise ValueError("Mobile SRQ scale must be finite and non-negative.")
    tensor = tensor.reshape(()).to(dtype=values.dtype)
    calibrated = tensor != 0
    safe_scale = torch.where(calibrated, tensor, torch.ones_like(tensor))
    rounded = torch.clamp(torch.round(values / safe_scale), -128.0, 127.0) * safe_scale
    quantized = torch.where(calibrated, rounded, values)
    if ste_gradient == "clipped":
        mask = (~calibrated | ((values >= -128 * safe_scale) & (values <= 127 * safe_scale))).to(values.dtype)
    elif ste_gradient == "identity":
        mask = 1.0
    else:
        raise ValueError(f"Unsupported mobile SRQ STE: {ste_gradient!r}")
    # Adding a zero-valued surrogate preserves the exact low-precision forward
    # value. x + (q - x).detach() can introduce a second BF16 rounding error.
    return quantized.detach() + (values - values.detach()) * mask


def fake_quantize_activation(
    activation: Any,
    spec: QATSpec,
    *,
    scale_override: Any | None = None,
    validated_scale: bool = False,
) -> Any:
    if spec.activation_quantizer == "gemma_mobile_srq":
        if spec.activation_bits != 8 or spec.scale_mode != "retained_mobile":
            raise ValueError("gemma_mobile_srq requires retained_mobile A8 scales.")
        return mobile_srq_ste(
            activation, scale_override, ste_gradient=spec.ste_gradient,
            validate_scale=not validated_scale,
        )
    if spec.activation_quantizer != "legacy":
        raise ValueError(f"Unknown activation quantizer: {spec.activation_quantizer!r}")
    return fake_quantize_ste(
        activation,
        bits=spec.activation_bits,
        symmetric=spec.activation_symmetric,
        per_channel=False,
        eps=spec.eps,
        quantizer=spec.quantizer,
        scale_override=scale_override,
        ste_gradient=spec.ste_gradient,
    )


def _mobile_weight_key_candidates(module_name: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for module_candidate in _module_name_candidates(module_name):
        normalized = module_candidate
        if normalized.startswith("language_model."):
            normalized = "model." + normalized[len("language_model.") :]
        elif normalized.startswith(("layers.", "embed_tokens", "embed_tokens_per_layer", "per_layer_model_projection")):
            normalized = "model." + normalized
        for value in (module_candidate, normalized):
            if not value:
                continue
            weight_key = value if value.endswith(".weight") else value + ".weight"
            if weight_key not in candidates:
                candidates.append(weight_key)
            if weight_key.endswith(".linear.weight"):
                flattened = weight_key[: -len(".linear.weight")] + ".weight"
                if flattened not in candidates:
                    candidates.append(flattened)
    return tuple(candidates)


class QATController:
    """Owns reversible fake-quantization wrappers for a model."""

    def __init__(self, spec: QATSpec, mobile_qparams: Any | None = None):
        self.spec = spec
        self.mobile_qparams = mobile_qparams
        self._original_forwards: dict[Any, Callable[..., Any]] = {}
        self._wrapped_names: list[str] = []
        self._wrapped_linear_names: list[str] = []
        self._wrapped_embedding_names: list[str] = []
        self._wrapped_effective_lora_names: list[str] = []
        self._wrapped_weight_bits: dict[str, int] = {}
        self._detected_lora_adapter_linear_count = 0
        self._uncovered_lora_adapter_linear_names: list[str] = []
        self._retained_qparams_bindings: dict[str, dict[str, Any]] = {}
        self._frozen_activation_bindings: dict[str, dict[str, Any]] = {}
        self.saturation_monitor: Any | None = None
        self.trainable_scope: dict[str, Any] | None = None

    def _retained_qparams_for_module(
        self,
        module_name: str,
        module_spec: QATSpec,
        weight_shape: tuple[int, ...],
        *, frozen: bool = False,
    ) -> dict[str, Any] | None:
        if module_spec.scale_mode != "retained_mobile":
            return None
        if self.mobile_qparams is None:
            raise ValueError(
                "QAT scale_mode=retained_mobile requires a verified mobile "
                "qparams contract."
            )
        candidates = _mobile_weight_key_candidates(module_name)
        weight_key = self.mobile_qparams.resolve_weight_key(candidates)
        if weight_key is None:
            if module_spec.fixed_scale_required:
                raise ValueError(
                    "No retained mobile scale matches QAT module "
                    f"{module_name!r}; candidates={list(candidates)!r}."
                )
            return None
        scale = self.mobile_qparams.load_scale(
            weight_key,
            weight_shape=tuple(int(value) for value in weight_shape),
            bits=module_spec.weight_bits,
            group_size=module_spec.group_size,
        )
        input_activation_scale = self.mobile_qparams.activation_scale(
            weight_key, "input"
        )
        output_activation_scale = self.mobile_qparams.activation_scale(
            weight_key, "output"
        )
        if (
            module_spec.activation_bits < 16
            and module_spec.fixed_activation_scale_required
            and (
                input_activation_scale is None
                or output_activation_scale is None
            )
        ):
            raise ValueError(
                f"Retained A{module_spec.activation_bits} input/output scales "
                f"are incomplete for QAT module {module_name!r} ({weight_key!r})."
            )
        binding = {
            "weight_key": weight_key,
            "bits": module_spec.weight_bits,
            "group_size": module_spec.group_size,
            "scale_shape": list(scale.shape),
            "input_activation_scale": input_activation_scale,
            "output_activation_scale": output_activation_scale,
        }
        bindings = self._frozen_activation_bindings if frozen else self._retained_qparams_bindings
        bindings[module_name] = binding
        return {**binding, "scale": scale}

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

    def _observe(self, name: str, values: Any, scale: Any, *, role: str, spec: QATSpec) -> None:
        if (self.saturation_monitor is None or scale is None
                or not getattr(self.saturation_monitor, "active", True)):
            return
        if role == "weight":
            qmin, qmax = _quant_bounds_for_quantizer(spec.weight_bits, spec.weight_symmetric, spec.quantizer)
        elif spec.activation_quantizer == "gemma_mobile_srq":
            qmin, qmax = -128, 127
            scale = scale.to(dtype=values.dtype)
        else:
            qmin, qmax = _quant_bounds_for_quantizer(spec.activation_bits, spec.activation_symmetric, spec.quantizer)
        self.saturation_monitor.observe(name, values, scale, qmin=qmin, qmax=qmax, role=role)

    def _prepare_frozen_activations(self, model: Any) -> None:
        """Wrap only inventoried frozen mobile FC edges; never re-quantize weights."""
        import torch
        from torch import nn

        if self.spec.scale_mode != "retained_mobile" or self.spec.activation_quantizer != "gemma_mobile_srq":
            raise ValueError("Frozen mobile activation simulation requires retained_mobile gemma_mobile_srq.")
        if self.mobile_qparams is None:
            raise ValueError("Frozen mobile activation simulation requires verified qparams.")
        expected = set(self.mobile_qparams.frozen_activation_weight_keys())
        count = self.spec.expected_frozen_activation_modules
        if count is None or len(expected) != count or count <= 0:
            raise ValueError(f"Frozen activation inventory must contain exactly {count} modules; found {len(expected)}.")
        for name, module in list(model.named_modules()):
            key = self.mobile_qparams.resolve_weight_key(_mobile_weight_key_candidates(name))
            if key not in expected:
                continue
            if not isinstance(module, nn.Linear) or module in self._original_forwards:
                raise ValueError(f"Expected an ordinary frozen Linear for {key}, not an adapter or unsupported layer.")
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError(f"Frozen mobile activation layer {name!r} has trainable weights/bias.")
            bits = self.spec.weight_bits_for_module(name)
            if bits != 8:
                raise ValueError(f"Frozen mobile activation layer {name!r} must retain its W8 inventory.")
            module_spec = _module_spec_for_module(self.spec, name, bits)
            binding = self._retained_qparams_for_module(name, module_spec, tuple(module.weight.shape), frozen=True)
            original_forward = module.forward
            cache: dict[tuple[str, str], Any] = {}

            def frozen_forward(
                inputs: Any, *args: Any, _name: str = name,
                _original: Callable = original_forward, _spec: QATSpec = module_spec,
                _binding: dict = binding, _cache: dict = cache, **kwargs: Any,
            ) -> Any:
                if args or kwargs:
                    raise TypeError(f"Unexpected arguments for frozen mobile Linear {_name!r}.")

                def scale_for(role: str, tensor: Any) -> Any:
                    cache_key = (role, str(tensor.device))
                    if cache_key not in _cache:
                        _cache[cache_key] = torch.tensor(
                            _binding[f"{role}_activation_scale"], dtype=torch.float32, device=tensor.device,
                        )
                    return _cache[cache_key]

                input_scale = scale_for("input", inputs)
                self._observe(_name, inputs, input_scale, role="input", spec=_spec)
                output = _original(fake_quantize_activation(inputs, _spec, scale_override=input_scale, validated_scale=True))
                output_scale = scale_for("output", output)
                self._observe(_name, output, output_scale, role="output", spec=_spec)
                return fake_quantize_activation(output, _spec, scale_override=output_scale, validated_scale=True)

            module.forward = frozen_forward
            self._original_forwards[module] = original_forward
            self._wrapped_names.append(name)
            self._wrapped_linear_names.append(name)
        bound = [binding["weight_key"] for binding in self._frozen_activation_bindings.values()]
        if len(bound) != count or set(bound) != expected:
            raise ValueError(f"Frozen A8 scope mismatch: missing={sorted(expected-set(bound))}, extra={sorted(set(bound)-expected)}.")

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
                module_spec = _module_spec_for_module(
                    self.spec, module_name, module_weight_bits
                )
                retained_qparams = self._retained_qparams_for_module(
                    module_name,
                    module_spec,
                    tuple(int(value) for value in module.base_layer.weight.shape),
                )
                device_scale_cache: dict[str, Any] = {}

                def retained_scale(
                    name: str,
                    reference: Any,
                    *,
                    _binding: dict[str, Any] | None = retained_qparams,
                    _cache: dict[str, Any] = device_scale_cache,
                ) -> Any | None:
                    if _binding is None:
                        return None
                    cache_key = f"{name}:{reference.device}"
                    cached = _cache.get(cache_key)
                    if cached is None:
                        raw = (
                            _binding["scale"]
                            if name == "weight"
                            else _binding.get(f"{name}_activation_scale")
                        )
                        if raw is None:
                            return None
                        import torch

                        cached = (
                            raw.to(device=reference.device, dtype=torch.float32)
                            if isinstance(raw, torch.Tensor)
                            else torch.tensor(
                                float(raw),
                                device=reference.device,
                                dtype=torch.float32,
                            )
                        )
                        _cache[cache_key] = cached
                    return cached

                def qat_lora_forward(
                    input_tensor: Any,
                    *args: Any,
                    _module: Any = module,
                    _module_name: str = module_name,
                    _module_spec: QATSpec = module_spec,
                    _adapter_names: tuple[str, ...] = adapter_names,
                    _retained_scale: Callable[[str, Any], Any | None] = retained_scale,
                    _has_retained_qparams: bool = retained_qparams is not None,
                    **kwargs: Any,
                ) -> Any:
                    if args or kwargs:
                        raise TypeError(
                            "Effective merged-weight QAT does not support extra "
                            f"arguments for {_module_name!r}; mixed-adapter batches "
                            "would bypass the deployment-equivalent forward."
                        )
                    input_scale = _retained_scale("input", input_tensor)
                    self._observe(_module_name, input_tensor, input_scale, role="input", spec=_module_spec)
                    quantized_input = fake_quantize_activation(
                        input_tensor,
                        _module_spec,
                        scale_override=input_scale,
                        validated_scale=_has_retained_qparams,
                    )
                    effective_weight = _effective_lora_weight(
                        _module, _adapter_names
                    )
                    weight_scale = _retained_scale("weight", effective_weight)
                    self._observe(_module_name, effective_weight, weight_scale, role="weight", spec=_module_spec)
                    quantized_weight = fake_quantize_weight(
                        effective_weight,
                        _module_spec,
                        scale_override=weight_scale,
                    )
                    output = functional.linear(
                        quantized_input,
                        quantized_weight,
                        _module.base_layer.bias,
                    )
                    if _has_retained_qparams:
                        output_scale = _retained_scale("output", output)
                        self._observe(_module_name, output, output_scale, role="output", spec=_module_spec)
                        return fake_quantize_activation(
                            output,
                            _module_spec,
                            scale_override=output_scale,
                            validated_scale=True,
                        )
                    return output

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

        if self.spec.simulate_frozen_activations:
            self._prepare_frozen_activations(model)
        if self.spec.require_lora_trainable_scope:
            self.trainable_scope = self._check_trainable_scope(model)
        if self.spec.effective_lora_only:
            return self

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
            module_spec = _module_spec_for_module(
                self.spec, module_name, module_weight_bits
            )

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
                    return _fake_quantized_embedding_lookup(
                        input_tensor,
                        _module,
                        _module_spec,
                        functional,
                        _original_forward,
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

    def _check_trainable_scope(self, model: Any) -> dict[str, Any]:
        modules = dict(model.named_modules())
        expected: set[int] = set()
        for name in self._wrapped_effective_lora_names:
            module = modules[name]
            for adapter in _active_lora_adapter_names(module):
                for container in (module.lora_A, module.lora_B):
                    weight = container[adapter].weight
                    expected.add(id(weight))
        actual = {id(value) for value in model.parameters() if value.requires_grad}
        unexpected = [name for name, value in model.named_parameters() if value.requires_grad and id(value) not in expected]
        # An entirely frozen model is the supported standalone evaluation path.
        # A partially trainable adapter is never a supported training recipe.
        if unexpected or (actual and actual != expected) or not expected:
            raise ValueError(f"Retained mobile trainable scope must be exactly the active LoRA A/B weights; unexpected={unexpected[:12]}.")
        return {"verified": True, "expected_lora_tensors": len(expected),
                "trainable_lora_tensors": len(actual), "all_adapters_trainable": actual == expected,
                "all_parameters_frozen": not actual}

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
            "wrapped_group_sizes_by_module": {
                name: self.spec.group_size_for_module(name)
                for name in self._wrapped_names
                if self.spec.group_size_for_module(name) is not None
            },
            "wrapped_weight_bit_histogram": dict(
                sorted(bit_histogram.items(), key=lambda item: int(item[0]))
            ),
            "numeric_contract": qat_numeric_contract(self.spec),
            "retained_qparams": (
                self.mobile_qparams.summary()
                if self.mobile_qparams is not None
                else None
            ),
            "retained_qparams_binding_count": len(
                self._retained_qparams_bindings
            ),
            "retained_qparams_bindings": dict(self._retained_qparams_bindings),
            "frozen_activation_module_count": len(self._frozen_activation_bindings),
            "frozen_activation_bindings": dict(self._frozen_activation_bindings),
            "trainable_scope": self.trainable_scope,
            "native_kv_cache_simulated": False,
            "spec": self.spec.to_dict(),
        }


def _module_spec_for_module(
    spec: QATSpec, module_name: str, weight_bits: int
) -> QATSpec:
    group_size = spec.group_size_for_module(module_name)
    if weight_bits == spec.weight_bits and group_size == spec.group_size:
        return spec
    return QATSpec(
        **{
            **spec.to_dict(),
            "weight_bits": weight_bits,
            "group_size": group_size,
            "module_quant_configs": (),
            "module_group_sizes": (),
            "modules_to_not_convert": (),
        }
    )


def _fake_quantized_embedding_lookup(
    input_tensor: Any,
    module: Any,
    spec: QATSpec,
    functional: Any,
    original_forward: Callable[..., Any],
) -> Any:
    """Fake-quantize only embedding rows referenced by this input batch.

    The Gemma 4 per-layer table is roughly 4.7 GiB in BF16. Quantizing the
    complete frozen table on every token lookup is unnecessary: embedding
    output depends only on selected rows, and the released scales are local to
    each row (with 256-column groups for ``embed_tokens_per_layer``).
    """

    import torch

    if not hasattr(input_tensor, "numel") or input_tensor.numel() == 0:
        return original_forward(input_tensor)
    if module.max_norm is not None:
        # ``max_norm`` mutates selected source rows in PyTorch. Gemma does not
        # use it, so preserve generic nn.Embedding semantics rather than
        # pretending the optimized path is exact.
        return original_forward(input_tensor)

    flat_indices = input_tensor.reshape(-1)
    unique_indices, inverse_indices = torch.unique(
        flat_indices, sorted=True, return_inverse=True
    )
    selected_weight = torch.index_select(module.weight, 0, unique_indices)
    quantized_weight = fake_quantize_weight(selected_weight, spec)
    local_padding_idx = None
    if module.padding_idx is not None:
        padding_matches = torch.nonzero(
            unique_indices == int(module.padding_idx), as_tuple=False
        )
        if padding_matches.numel():
            local_padding_idx = int(padding_matches[0, 0].item())
    output = functional.embedding(
        inverse_indices.reshape_as(input_tensor),
        quantized_weight,
        local_padding_idx,
        None,
        module.norm_type,
        module.scale_grad_by_freq,
        module.sparse,
    )

    # Gemma 3 exposes ``Gemma3TextScaledWordEmbedding`` as an nn.Embedding
    # subclass. Its forward method applies a non-unit ``embed_scale`` after
    # the lookup. Calling functional.embedding above is intentional (it lets
    # us quantize only the referenced rows), but it would otherwise silently
    # drop that architecture-specific post-processing and shrink every hidden
    # state by roughly sqrt(hidden_size). Preserve the public embedding
    # contract for scaled embedding subclasses while keeping the fast path.
    embed_scale = getattr(module, "embed_scale", None)
    if embed_scale is None:
        embed_scale = getattr(module, "scalar_embed_scale", None)
    if embed_scale is not None:
        if hasattr(embed_scale, "to"):
            embed_scale = embed_scale.to(
                device=output.device,
                dtype=output.dtype,
            )
        else:
            embed_scale = output.new_tensor(embed_scale)
        output = output * embed_scale
    return output


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


def qat_numeric_contract(spec: QATSpec) -> dict[str, Any]:
    """Return the explicit numerical contract represented by a QAT spec."""

    is_ai_edge = spec.quantizer == "ste_ai_edge"
    min_scale_matches = bool(
        is_ai_edge and float(spec.eps) == float(AI_EDGE_MIN_SCALE)
    )
    return {
        "quantizer": spec.quantizer,
        "reference_package": "ai-edge-quantizer",
        "reference_version": AI_EDGE_REFERENCE_QUANTIZER_VERSION,
        "scale_compute_dtype": (
            AI_EDGE_SCALE_COMPUTE_DTYPE if is_ai_edge else "source_dtype"
        ),
        "minimum_scale": float(spec.eps),
        "expected_minimum_scale": (
            AI_EDGE_MIN_SCALE if is_ai_edge else float(spec.eps)
        ),
        "minimum_scale_matches": min_scale_matches if is_ai_edge else None,
        "rounding": AI_EDGE_ROUNDING if is_ai_edge else "torch_round",
        "low_bit_signed_range": "full" if is_ai_edge else "symmetric_absmax",
        "w8_signed_range": "narrow" if is_ai_edge else "symmetric_absmax",
        "blockwise_scale_cast": (
            list(AI_EDGE_BLOCKWISE_SCALE_CAST) if is_ai_edge else []
        ),
        "public_ai_edge_numeric_contract": bool(
            is_ai_edge and min_scale_matches
        ),
        "scale_mode": spec.scale_mode,
        "retained_mobile_scales": spec.scale_mode == "retained_mobile",
        "fixed_scale_required": spec.fixed_scale_required,
        "fixed_activation_scale_required": spec.fixed_activation_scale_required,
        "effective_lora_only": spec.effective_lora_only,
        "ste_gradient": spec.ste_gradient,
        "scales_recomputed_from_weight_absmax": spec.scale_mode == "dynamic",
        "private_google_observer_recovered": False,
        "activation_quantizer": spec.activation_quantizer,
        "activation_reference": MOBILE_SRQ_REFERENCE if spec.activation_quantizer == "gemma_mobile_srq" else None,
        "activation_signed_range": [-128, 127] if spec.activation_quantizer == "gemma_mobile_srq" else None,
        "activation_scale_arithmetic": "input_dtype" if spec.activation_quantizer == "gemma_mobile_srq" else "legacy",
        "zero_activation_scale": "bypass" if spec.activation_quantizer == "gemma_mobile_srq" else "legacy",
        "frozen_activation_simulation": spec.simulate_frozen_activations,
        "native_runtime_numeric_parity_verified": False,
    }


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

    spec = QATSpec.from_config(config)
    mobile_qparams = None
    if spec.scale_mode == "retained_mobile":
        contract_path = spec.mobile_qparams_contract
        if not contract_path:
            model_config = (
                config.get("model")
                if isinstance(config.get("model"), dict)
                else {}
            )
            contract_path = model_config.get("mobile_qparams_contract")
        if not contract_path:
            raise ValueError(
                "QAT scale_mode=retained_mobile requires "
                "qat.mobile_qparams_contract or model.mobile_qparams_contract."
            )
        from ir_training.qat.mobile_qparams import MobileQParams

        mobile_qparams = MobileQParams(contract_path)
    controller = QATController(spec, mobile_qparams=mobile_qparams)
    try:
        controller.prepare(model)
    except Exception:
        controller.restore()
        raise
    qat_config = config.get("qat", config)
    saturation_config = qat_config.get("saturation") if isinstance(qat_config, dict) else None
    if isinstance(saturation_config, dict) and saturation_config.get("enabled", False):
        from ir_training.qat.saturation import SaturationMonitor

        monitor = SaturationMonitor.from_config(qat_config)
        # Nonzero ranks must not publish disabled snapshots over rank zero's
        # preflight/telemetry files in a shared DDP output directory.
        if monitor.enabled:
            controller.saturation_monitor = monitor
    if spec.scale_mode == "retained_mobile":
        expected_keys = set(mobile_qparams.trainable_projection_weight_keys())
        bound_keys = {
            str(binding.get("weight_key"))
            for binding in controller._retained_qparams_bindings.values()
        }
        expected_count = spec.expected_effective_lora_modules
        if expected_count is None:
            raise ValueError(
                "Retained mobile QAT requires qat.expected_effective_lora_modules."
            )
        if (
            len(expected_keys) != int(expected_count)
            or controller.wrapped_effective_lora_count != int(expected_count)
            or len(controller._retained_qparams_bindings) != int(expected_count)
            or bound_keys != expected_keys
        ):
            missing = sorted(expected_keys - bound_keys)
            extra = sorted(bound_keys - expected_keys)
            raise ValueError(
                "Live effective-LoRA scope differs from the exact retained "
                "mobile projection contract: "
                f"expected={expected_count}, contract_keys={len(expected_keys)}, "
                f"wrapped={controller.wrapped_effective_lora_count}, "
                f"bindings={len(controller._retained_qparams_bindings)}, "
                f"missing={missing[:12]}, extra={extra[:12]}. Verify the pinned "
                "PEFT version/target mapping; never train a partial adapter."
            )
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
