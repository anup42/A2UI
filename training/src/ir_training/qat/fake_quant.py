from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class QATSpec:
    """Configuration for the training-time fake quantizer.

    The quantizer uses a straight-through estimator (STE): the forward pass
    sees rounded/clamped values while the backward pass receives the identity
    gradient.  Scales are calculated from detached values on every forward
    pass, so this implementation does not add observer state to checkpoints.
    """

    weight_bits: int = 8
    activation_bits: int = 8
    weight_symmetric: bool = True
    activation_symmetric: bool = True
    weight_per_channel: bool = True
    weight_axis: int = 0
    group_size: int | None = None
    only_base_layers: bool = True
    exclude_modules: tuple[str, ...] = (
        r"(^|\.)(lm_head|embed_tokens|embed_positions|output_projection)(\.|$)",
    )
    quantizer: str = "ste_absmax"
    eps: float = 1e-8

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QATSpec":
        qat = config.get("qat") if isinstance(config.get("qat"), dict) else config
        exclude = qat.get("exclude_modules", cls.exclude_modules)
        if isinstance(exclude, str):
            exclude = (exclude,)
        elif isinstance(exclude, list | tuple):
            exclude = tuple(str(item) for item in exclude)
        else:
            exclude = cls.exclude_modules
        group_size = qat.get("group_size")
        return cls(
            weight_bits=int(qat.get("weight_bits", cls.weight_bits)),
            activation_bits=int(qat.get("activation_bits", cls.activation_bits)),
            weight_symmetric=bool(qat.get("weight_symmetric", cls.weight_symmetric)),
            activation_symmetric=bool(qat.get("activation_symmetric", cls.activation_symmetric)),
            weight_per_channel=bool(qat.get("weight_per_channel", cls.weight_per_channel)),
            weight_axis=int(qat.get("weight_axis", cls.weight_axis)),
            group_size=int(group_size) if group_size not in (None, "", 0) else None,
            only_base_layers=bool(qat.get("only_base_layers", cls.only_base_layers)),
            exclude_modules=exclude,
            quantizer=str(qat.get("quantizer", cls.quantizer)).strip().lower(),
            eps=float(qat.get("eps", cls.eps)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
) -> tuple[Any, Any, int, int]:
    """Return detached scale/zero point tensors and integer bounds."""

    import torch

    qmin, qmax = _quant_bounds(bits, symmetric)
    detached = values.detach()
    if symmetric:
        max_abs = detached.abs()
        if reduce_dims:
            max_abs = max_abs.amax(dim=reduce_dims, keepdim=True)
        scale = (max_abs / max(abs(qmin), abs(qmax))).clamp_min(float(eps))
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
) -> Any:
    """Fake-quantize a tensor with an STE backward pass.

    Weight tensors may use per-channel or group-wise scales.  Activations use
    the default per-tensor path.  Non-floating tensors and 16-bit-or-higher
    requests are returned unchanged.
    """

    import torch

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
    )


def fake_quantize_activation(activation: Any, spec: QATSpec) -> Any:
    return fake_quantize_ste(
        activation,
        bits=spec.activation_bits,
        symmetric=spec.activation_symmetric,
        per_channel=False,
        eps=spec.eps,
    )


class QATController:
    """Owns reversible fake-quantization wrappers for a model."""

    def __init__(self, spec: QATSpec):
        self.spec = spec
        self._original_forwards: dict[Any, Callable[..., Any]] = {}
        self._wrapped_names: list[str] = []

    @property
    def wrapped_count(self) -> int:
        # Keep the count available after ``restore()`` for run metadata.
        return len(self._wrapped_names)

    @property
    def wrapped_names(self) -> tuple[str, ...]:
        return tuple(self._wrapped_names)

    def prepare(self, model: Any) -> "QATController":
        import torch.nn as nn
        import torch.nn.functional as functional

        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.spec.only_base_layers and not _is_base_layer_name(module_name):
                continue
            if any(re.search(pattern, module_name) for pattern in self.spec.exclude_modules):
                continue
            if module in self._original_forwards:
                continue
            original_forward = module.forward

            def qat_forward(
                input_tensor: Any,
                *args: Any,
                _module: Any = module,
                _original_forward: Callable[..., Any] = original_forward,
                **kwargs: Any,
            ) -> Any:
                # Linear normally receives only its input. Preserve unusual
                # callers by delegating rather than changing their signature.
                if args or kwargs:
                    return _original_forward(input_tensor, *args, **kwargs)
                quantized_input = fake_quantize_activation(input_tensor, self.spec)
                quantized_weight = fake_quantize_weight(_module.weight, self.spec)
                return functional.linear(quantized_input, quantized_weight, _module.bias)

            module.forward = qat_forward
            self._original_forwards[module] = original_forward
            self._wrapped_names.append(module_name)
        return self

    def restore(self) -> None:
        for module, original_forward in self._original_forwards.items():
            module.forward = original_forward
        self._original_forwards.clear()

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "true_fake_quant": True,
            "wrapped_linear_count": self.wrapped_count,
            "wrapped_linear_names": list(self._wrapped_names),
            "spec": self.spec.to_dict(),
        }


def _is_base_layer_name(module_name: str) -> bool:
    return module_name == "base_layer" or ".base_layer" in module_name


def prepare_qat_model(model: Any, config: dict[str, Any]) -> QATController:
    """Apply configured fake quantization and fail if no base linears match."""

    controller = QATController(QATSpec.from_config(config)).prepare(model)
    if controller.wrapped_count == 0:
        raise ValueError(
            "QAT preparation matched no base nn.Linear modules. "
            "Run it after LoRA wrapping or set qat.only_base_layers=false for a full model."
        )
    return controller
