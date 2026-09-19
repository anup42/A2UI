from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _Observation:
    counts: Any
    max_abs_scaled_value: Any
    role: str


def _merge_observation(left: _Observation, right: _Observation) -> _Observation:
    import torch

    return _Observation(
        counts=left.counts + right.counts.to(device=left.counts.device),
        max_abs_scaled_value=torch.maximum(
            left.max_abs_scaled_value,
            right.max_abs_scaled_value.to(device=left.max_abs_scaled_value.device),
        ),
        role=left.role,
    )


class SaturationMonitor:
    """Bounded, opt-in retained-scale saturation telemetry.

    Observation is deliberately device-only. Host synchronization happens in
    ``drain()``, outside model forwards, and no quality threshold is inferred
    from the measurements.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        sample_every_optimizer_steps: int = 20,
        max_values_per_module: int = 2048,
        rank_zero_only: bool = True,
        rank: int = 0,
    ) -> None:
        self.enabled = bool(enabled) and (not rank_zero_only or int(rank) == 0)
        self.sample_every_optimizer_steps = max(
            1, int(sample_every_optimizer_steps)
        )
        self.max_values_per_module = max(1, int(max_values_per_module))
        self.rank_zero_only = bool(rank_zero_only)
        self.rank = int(rank)
        self._active = False
        self._step: int | None = None
        self._seen: set[tuple[str, str]] = set()
        self._window: dict[tuple[str, str], _Observation] = {}
        self._pending: dict[tuple[str, str], _Observation] = {}
        self._cumulative: dict[tuple[str, str], _Observation] = {}
        self._pending_windows = 0
        self._cumulative_windows = 0
        self._pending_steps: list[int] = []
        self._cumulative_steps: list[int] = []
        self._errors: dict[str, str] = {}

    @classmethod
    def from_config(cls, qat_cfg: dict[str, Any] | None) -> SaturationMonitor:
        qat = qat_cfg if isinstance(qat_cfg, dict) else {}
        telemetry = qat.get("saturation")
        if not isinstance(telemetry, dict):
            telemetry = (
                qat.get("saturation_telemetry")
                if isinstance(qat.get("saturation_telemetry"), dict)
                else {}
            )
        # Keep every legacy/dynamic-scale training profile behaviorally inert.
        retained = str(qat.get("scale_mode", "dynamic")).strip().lower() == "retained_mobile"
        enabled = retained and bool(telemetry.get("enabled", True))
        rank_text = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        try:
            rank = int(rank_text)
        except (TypeError, ValueError):
            rank = 0
        return cls(
            enabled=enabled,
            sample_every_optimizer_steps=int(
                telemetry.get("sample_every_optimizer_steps", 20)
            ),
            max_values_per_module=int(telemetry.get("max_values_per_module", 2048)),
            rank_zero_only=bool(telemetry.get("rank_zero_only", True)),
            rank=rank,
        )

    @property
    def active(self) -> bool:
        return self.enabled and self._active

    def begin_step(self, step: int, *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        step = int(step)
        if not force and (
            step <= 0 or step % self.sample_every_optimizer_steps != 0
        ):
            return False
        if self._active:
            self.end_step()
        self._active = True
        self._step = step
        self._seen.clear()
        self._window.clear()
        return True

    def observe(
        self,
        module_name: str,
        values: Any,
        scale: Any,
        *,
        qmin: int,
        qmax: int,
        role: str,
    ) -> None:
        if not self.active:
            return
        key = (str(module_name), str(role).strip().lower())
        if key in self._seen:
            return
        self._seen.add(key)
        if isinstance(scale, (int, float)) and float(scale) == 0.0:
            return
        if not getattr(values, "is_floating_point", lambda: False)():
            return
        try:
            sampled_values, sampled_scale = _sample_values_and_scale(
                values, scale, self.max_values_per_module
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self._errors[f"{key[0]}::{key[1]}"] = str(exc)
            return
        if sampled_values.numel() == 0:
            return

        import torch

        with torch.no_grad():
            valid_scale = torch.isfinite(sampled_scale) & (sampled_scale != 0)
            safe_scale = torch.where(
                valid_scale, sampled_scale, torch.ones_like(sampled_scale)
            )
            scaled = torch.where(
                valid_scale,
                sampled_values / safe_scale,
                torch.zeros_like(sampled_values),
            )
            finite = valid_scale & torch.isfinite(scaled)
            rounded = torch.round(scaled)
            low = finite & (rounded < int(qmin))
            high = finite & (rounded > int(qmax))
            counts = torch.stack(
                (
                    torch.count_nonzero(valid_scale),
                    torch.count_nonzero(finite),
                    torch.count_nonzero(low),
                    torch.count_nonzero(high),
                )
            ).to(dtype=torch.int64)
            finite_magnitude = torch.where(
                finite, scaled.abs(), torch.zeros_like(scaled)
            )
            maximum = finite_magnitude.amax().to(dtype=torch.float32)
        self._window[key] = _Observation(counts, maximum, key[1])

    def end_step(self) -> bool:
        if not self._active:
            return False
        for key, observation in self._window.items():
            self._pending[key] = (
                _merge_observation(self._pending[key], observation)
                if key in self._pending
                else observation
            )
            self._cumulative[key] = (
                _merge_observation(self._cumulative[key], observation)
                if key in self._cumulative
                else observation
            )
        step = int(self._step or 0)
        self._pending_windows += 1
        self._cumulative_windows += 1
        self._pending_steps.append(step)
        self._cumulative_steps.append(step)
        self._active = False
        self._step = None
        self._seen.clear()
        self._window.clear()
        return True

    def drain(self) -> dict[str, Any]:
        if self._active:
            self.end_step()
        pending_modules, pending_summary = _materialize(
            self._pending,
            windows=self._pending_windows,
            steps=self._pending_steps,
        )
        cumulative_modules, cumulative_summary = _materialize(
            self._cumulative,
            windows=self._cumulative_windows,
            steps=self._cumulative_steps,
        )
        scalars = _summary_scalars(pending_summary)
        report = {
            "enabled": self.enabled,
            "sampling": {
                "sample_every_optimizer_steps": self.sample_every_optimizer_steps,
                "max_values_per_module": self.max_values_per_module,
                "rank_zero_only": self.rank_zero_only,
                "rank": self.rank,
                "same_module_role_once_per_window": True,
            },
            "summary": pending_summary,
            "scalars": scalars,
            "modules": pending_modules,
            "cumulative": {
                "summary": cumulative_summary,
                "modules": cumulative_modules,
            },
            "errors": dict(self._errors),
        }
        self._pending.clear()
        self._pending_windows = 0
        self._pending_steps.clear()
        self._errors.clear()
        return report


def _sample_values_and_scale(values: Any, scale: Any, limit: int) -> tuple[Any, Any]:
    import torch

    values = values.detach()
    if values.ndim == 0:
        value_shape = (1,)
        values = values.reshape(value_shape)
    else:
        value_shape = tuple(int(item) for item in values.shape)
    total = int(values.numel())
    if total == 0:
        return values.reshape(-1), values.reshape(-1)
    stride = max(1, math.ceil(total / max(1, int(limit))))
    indices = torch.arange(0, total, stride, device=values.device)[:limit]
    value_strides: list[int] = []
    running = 1
    for size in reversed(value_shape):
        value_strides.append(running)
        running *= size
    value_strides.reverse()
    value_coordinates = tuple(
        (indices // value_stride) % value_shape[dimension]
        for dimension, value_stride in enumerate(value_strides)
    )
    # Coordinate indexing copies only the bounded sample. ``reshape(-1)`` can
    # materialize an entire non-contiguous activation before sampling it.
    sampled_values = values[value_coordinates]

    scale_tensor = (
        scale
        if isinstance(scale, torch.Tensor)
        else torch.as_tensor(scale, device=values.device, dtype=values.dtype)
    )
    scale_tensor = scale_tensor.detach().to(device=values.device)
    if scale_tensor.ndim > len(value_shape):
        raise ValueError(
            f"scale rank {scale_tensor.ndim} cannot broadcast to values {value_shape}"
        )

    # Retained W4/W8 projection scales can be blockwise per output row:
    # [rows, columns / group_width] for a [rows, columns] effective weight.
    # Resolve only the bounded sampled coordinates instead of expanding the
    # scale or reshaping a potentially non-contiguous full weight tensor.
    if (
        len(value_shape) == 2
        and scale_tensor.ndim == 2
        and int(scale_tensor.shape[0]) in (1, value_shape[0])
        and 1 < int(scale_tensor.shape[1]) < value_shape[1]
        and value_shape[1] % int(scale_tensor.shape[1]) == 0
    ):
        group_width = value_shape[1] // int(scale_tensor.shape[1])
        scale_rows = (
            torch.zeros_like(indices)
            if int(scale_tensor.shape[0]) == 1
            else value_coordinates[0]
        )
        scale_columns = value_coordinates[1] // group_width
        return sampled_values, scale_tensor[scale_rows, scale_columns]

    aligned_scale_shape = (1,) * (len(value_shape) - scale_tensor.ndim) + tuple(
        int(item) for item in scale_tensor.shape
    )
    if any(
        scale_size not in (1, value_size)
        for scale_size, value_size in zip(aligned_scale_shape, value_shape, strict=True)
    ):
        raise ValueError(
            f"scale shape {tuple(scale_tensor.shape)} cannot broadcast to values {value_shape}"
        )
    if scale_tensor.numel() == 1:
        return sampled_values, scale_tensor.reshape(1).expand_as(sampled_values)

    leading = len(value_shape) - scale_tensor.ndim
    scale_coordinates = []
    for scale_dimension, scale_size in enumerate(scale_tensor.shape):
        value_dimension = leading + scale_dimension
        if int(scale_size) == 1:
            scale_coordinates.append(torch.zeros_like(indices))
        else:
            scale_coordinates.append(value_coordinates[value_dimension])
    sampled_scale = scale_tensor[tuple(scale_coordinates)]
    return sampled_values, sampled_scale


def _materialize(
    observations: dict[tuple[str, str], _Observation],
    *,
    windows: int,
    steps: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    modules: dict[str, Any] = {}
    totals = {"sampled_values": 0, "finite_values": 0, "clipped_low": 0, "clipped_high": 0}
    by_role: dict[str, dict[str, int]] = {}
    ordered = sorted(observations.items())
    if ordered:
        counts_rows = torch.stack([item[1].counts for item in ordered]).detach().cpu().tolist()
        maximums = (
            torch.stack([item[1].max_abs_scaled_value for item in ordered])
            .detach()
            .cpu()
            .tolist()
        )
    else:
        counts_rows = []
        maximums = []
    for ((name, role), _observation), counts, maximum in zip(
        ordered, counts_rows, maximums, strict=True
    ):
        sampled, finite, low, high = (int(item) for item in counts)
        clipped = low + high
        record = {
            "role": role,
            "sampled_values": sampled,
            "finite_values": finite,
            "clipped_low": low,
            "clipped_high": high,
            "saturation_fraction": clipped / finite if finite else 0.0,
            "max_abs_scaled_value": float(maximum),
        }
        modules[f"{name}::{role}"] = record
        totals["sampled_values"] += sampled
        totals["finite_values"] += finite
        totals["clipped_low"] += low
        totals["clipped_high"] += high
        role_totals = by_role.setdefault(
            role,
            {"sampled_values": 0, "finite_values": 0, "clipped_low": 0, "clipped_high": 0},
        )
        for key in role_totals:
            role_totals[key] += record[key]
    for role_totals in by_role.values():
        clipped = role_totals["clipped_low"] + role_totals["clipped_high"]
        finite = role_totals["finite_values"]
        role_totals["saturation_fraction"] = clipped / finite if finite else 0.0
    clipped = totals["clipped_low"] + totals["clipped_high"]
    summary = {
        "windows": int(windows),
        "steps": list(steps),
        **totals,
        "saturation_fraction": clipped / totals["finite_values"] if totals["finite_values"] else 0.0,
        "by_role": by_role,
    }
    return modules, summary


def _summary_scalars(summary: dict[str, Any]) -> dict[str, float]:
    by_role = summary.get("by_role") if isinstance(summary.get("by_role"), dict) else {}
    return {
        "qat_saturation_fraction": float(summary.get("saturation_fraction", 0.0)),
        "qat_saturation_weight_fraction": float(by_role.get("weight", {}).get("saturation_fraction", 0.0)),
        "qat_saturation_input_fraction": float(by_role.get("input", {}).get("saturation_fraction", 0.0)),
        "qat_saturation_output_fraction": float(by_role.get("output", {}).get("saturation_fraction", 0.0)),
        "qat_saturation_sampled_values": float(summary.get("sampled_values", 0)),
        "qat_saturation_windows": float(summary.get("windows", 0)),
    }


def write_saturation_report(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def build_saturation_trainer_callback(
    monitor: SaturationMonitor,
    *,
    report_path: str | Path,
    log_dir: str | Path | None,
    writer_factory: Any | None = None,
) -> Any | None:
    if not monitor.enabled:
        return None
    from transformers import TrainerCallback

    destination = Path(report_path)

    class SaturationTrainerCallback(TrainerCallback):
        def __init__(self) -> None:
            self.writer = None
            self.last_report: dict[str, Any] | None = None

        def on_step_begin(self, args, state, control, **kwargs):
            monitor.begin_step(int(getattr(state, "global_step", 0)) + 1)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            monitor.end_step()
            return control

        def on_prediction_step(self, args, state, control, **kwargs):
            # Safety for evaluation invoked outside the usual step-end cadence.
            monitor.end_step()
            return control

        def _publish(self, args: Any, state: Any, logs: dict[str, Any] | None) -> None:
            report = monitor.drain()
            if int(report["summary"]["windows"]) <= 0:
                return
            self.last_report = report
            write_saturation_report(destination, report)
            scalars = report["scalars"]
            if logs is not None:
                logs.update(scalars)
            history = getattr(state, "log_history", None)
            if isinstance(history, list) and history:
                history[-1].update(scalars)
            if self.writer is None:
                if writer_factory is None:
                    from ir_training.eval.tensorboard_logging import (
                        _summary_writer_factory,
                    )

                    factory = _summary_writer_factory()
                else:
                    factory = writer_factory
                writer_dir = Path(log_dir or getattr(args, "logging_dir", "") or args.output_dir)
                self.writer = factory(log_dir=str(writer_dir))
            for name, value in scalars.items():
                self.writer.add_scalar(
                    "train/qat_saturation/" + name.removeprefix("qat_saturation_"),
                    value,
                    int(getattr(state, "global_step", 0)),
                )
            self.writer.flush()

        def on_log(self, args, state, control, logs=None, **kwargs):
            monitor.end_step()
            self._publish(args, state, logs)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            monitor.end_step()
            self._publish(args, state, None)
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
                self.writer = None
            return control

    return SaturationTrainerCallback()
