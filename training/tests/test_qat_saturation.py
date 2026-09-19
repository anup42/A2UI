from __future__ import annotations

import gc
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.qat.saturation import (
    SaturationMonitor,
    build_saturation_trainer_callback,
)
from ir_training.train.sft import _require_trainable_qat_scope
from ir_training.train.tensorboard_callback import select_training_tensorboard_metrics


def _config(**telemetry):
    return {
        "scale_mode": "retained_mobile",
        "saturation_telemetry": telemetry,
    }


def test_non_retained_monitor_is_inert():
    monitor = SaturationMonitor.from_config(
        {"scale_mode": "dynamic", "saturation_telemetry": {"enabled": True}}
    )

    assert monitor.enabled is False
    assert monitor.begin_step(20) is False
    monitor.observe(
        "ignored",
        torch.tensor([100.0]),
        1.0,
        qmin=-2,
        qmax=1,
        role="weight",
    )
    report = monitor.drain()
    assert report["summary"]["windows"] == 0
    assert report["summary"]["sampled_values"] == 0


def test_monitor_bounds_sampling_broadcasts_scales_and_deduplicates_window():
    monitor = SaturationMonitor.from_config(
        _config(
            sample_every_optimizer_steps=1,
            max_values_per_module=4,
            rank_zero_only=False,
        )
    )
    values = torch.tensor(
        [[-2.0, -1.0, 2.0, 0.0], [-1.0, 0.0, 1.0, 2.0]],
        dtype=torch.float32,
    )
    scales = torch.tensor([[1.0], [0.5]], dtype=torch.float32)

    assert monitor.begin_step(1)
    monitor.observe(
        "model.q_proj",
        values,
        scales,
        qmin=-2,
        qmax=1,
        role="weight",
    )
    # Gradient checkpoint recomputation of the same boundary must not inflate it.
    monitor.observe(
        "model.q_proj",
        values * 100,
        scales,
        qmin=-2,
        qmax=1,
        role="weight",
    )
    assert monitor.end_step()

    report = monitor.drain()
    module = report["modules"]["model.q_proj::weight"]
    assert module["sampled_values"] == 4
    assert module["clipped_low"] == 0
    assert module["clipped_high"] == 2
    assert module["saturation_fraction"] == pytest.approx(0.5)
    assert report["summary"]["windows"] == 1
    assert report["scalars"]["qat_saturation_weight_fraction"] == pytest.approx(0.5)

    reset = monitor.drain()
    assert reset["summary"]["windows"] == 0
    assert reset["modules"] == {}
    assert reset["cumulative"]["summary"]["windows"] == 1


def test_monitor_schedule_rank_and_zero_scale_bypass(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    rank_one = SaturationMonitor.from_config(_config())
    assert rank_one.enabled is False

    monkeypatch.setenv("RANK", "0")
    monitor = SaturationMonitor.from_config(_config(max_values_per_module=8))
    assert monitor.begin_step(19) is False
    assert monitor.begin_step(20) is True
    monitor.observe(
        "model.head",
        torch.tensor([100.0]),
        0.0,
        qmin=-128,
        qmax=127,
        role="output",
    )
    monitor.end_step()
    report = monitor.drain()
    assert report["summary"]["sampled_values"] == 0
    assert report["modules"] == {}


def test_observation_detaches_autograd_and_bounds_noncontiguous_sample():
    monitor = SaturationMonitor.from_config(
        _config(
            sample_every_optimizer_steps=1,
            max_values_per_module=5,
            rank_zero_only=False,
        )
    )
    source = torch.arange(24, dtype=torch.float32, requires_grad=True)
    values = source.reshape(4, 6).transpose(0, 1)
    assert values.is_contiguous() is False
    scale = torch.ones((6, 1), dtype=torch.float32, requires_grad=True)
    values_ref = weakref.ref(values)
    scale_ref = weakref.ref(scale)

    monitor.begin_step(1)
    monitor.observe(
        "model.noncontiguous",
        values,
        scale,
        qmin=-2,
        qmax=1,
        role="weight",
    )
    observation = monitor._window[("model.noncontiguous", "weight")]
    assert observation.counts.requires_grad is False
    assert observation.max_abs_scaled_value.requires_grad is False
    assert observation.max_abs_scaled_value.grad_fn is None
    monitor.end_step()
    del observation, values, scale
    gc.collect()
    assert values_ref() is None
    assert scale_ref() is None

    report = monitor.drain()
    assert report["summary"]["sampled_values"] == 5
    assert source.grad is None


def test_monitor_maps_grouped_per_row_scales_without_contiguous_expansion():
    monitor = SaturationMonitor.from_config(
        _config(
            sample_every_optimizer_steps=1,
            max_values_per_module=16,
            rank_zero_only=False,
        )
    )
    values = torch.tensor(
        [
            [-3.0, -1.0],
            [-2.0, -0.5],
            [0.0, 0.0],
            [1.0, 0.5],
            [4.0, 8.0],
            [2.0, 4.0],
            [0.0, 0.0],
            [-2.0, -8.0],
        ]
    ).transpose(0, 1)
    scales = torch.tensor([[1.0, 0.5], [2.0, 4.0]]).transpose(0, 1)
    assert values.shape == (2, 8)
    assert scales.shape == (2, 2)
    assert values.is_contiguous() is False
    assert scales.is_contiguous() is False

    monitor.begin_step(1)
    monitor.observe(
        "model.grouped_w4",
        values,
        scales,
        qmin=-2,
        qmax=1,
        role="weight",
    )
    monitor.end_step()

    report = monitor.drain()
    module = report["modules"]["model.grouped_w4::weight"]
    assert report["errors"] == {}
    assert module["sampled_values"] == 16
    assert module["clipped_low"] == 1
    assert module["clipped_high"] == 2
    assert module["saturation_fraction"] == pytest.approx(3 / 16)


def test_saturation_config_alias_and_trainable_scope_gate(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monitor = SaturationMonitor.from_config(
        {
            "scale_mode": "retained_mobile",
            "saturation": {
                "enabled": True,
                "sample_every_optimizer_steps": 7,
                "max_values_per_module": 13,
            },
        }
    )
    assert monitor.sample_every_optimizer_steps == 7
    assert monitor.max_values_per_module == 13

    good = SimpleNamespace(
        trainable_scope={
            "verified": True,
            "all_adapters_trainable": True,
            "trainable_lora_tensors": 410,
        }
    )
    _require_trainable_qat_scope({"require_lora_trainable_scope": True}, good)

    frozen = SimpleNamespace(
        trainable_scope={
            "verified": True,
            "all_adapters_trainable": False,
            "trainable_lora_tensors": 0,
            "all_parameters_frozen": True,
        }
    )
    with pytest.raises(RuntimeError, match="all-frozen scope"):
        _require_trainable_qat_scope(
            {"require_lora_trainable_scope": True}, frozen
        )


class _Writer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scalars = []
        self.flushes = 0
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


def test_callback_windows_logs_tensorboard_and_json(tmp_path):
    pytest.importorskip("transformers")
    monitor = SaturationMonitor.from_config(
        _config(
            sample_every_optimizer_steps=20,
            max_values_per_module=8,
            rank_zero_only=False,
        )
    )
    writers = []

    def writer_factory(**kwargs):
        writer = _Writer(**kwargs)
        writers.append(writer)
        return writer

    report_path = tmp_path / "saturation.json"
    callback = build_saturation_trainer_callback(
        monitor,
        report_path=report_path,
        log_dir=tmp_path / "tensorboard",
        writer_factory=writer_factory,
    )
    args = SimpleNamespace(logging_dir=str(tmp_path), output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=19, log_history=[{"loss": 1.0}])
    control = SimpleNamespace()

    callback.on_step_begin(args, state, control)
    monitor.observe(
        "model.q_proj",
        torch.tensor([-3.0, 0.0, 2.0]),
        1.0,
        qmin=-2,
        qmax=1,
        role="input",
    )
    callback.on_step_end(args, state, control)
    state.global_step = 20
    logs = {"loss": 0.9}
    callback.on_log(args, state, control, logs=logs)

    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["windows"] == 1
    assert logs["qat_saturation_fraction"] == pytest.approx(2 / 3)
    assert state.log_history[-1]["qat_saturation_input_fraction"] == pytest.approx(2 / 3)
    assert len(writers) == 1
    assert len(writers[0].scalars) == 6
    assert {item[0] for item in writers[0].scalars} == {
        "train/qat_saturation/fraction",
        "train/qat_saturation/weight_fraction",
        "train/qat_saturation/input_fraction",
        "train/qat_saturation/output_fraction",
        "train/qat_saturation/sampled_values",
        "train/qat_saturation/windows",
    }
    # The existing minimal writer keeps loss but deliberately ignores the six
    # saturation keys, which are written once by the dedicated callback.
    assert select_training_tensorboard_metrics(logs) == {"train/loss": 0.9}

    callback.on_train_end(args, state, control)
    assert writers[0].closed is True
    assert len(writers[0].scalars) == 6
