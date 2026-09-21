"""CPU-only tests for rank-local sharded memory diagnostics."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
import torch
from ir_training.train.sharded_diagnostics import (
    ShardedProbeDiagnostics,
    memory_gate_checks,
)


def _memory(*, reserved=899, free=101, total=1000):
    return {
        "device": "cuda:0",
        "cuda_local_rank_only": True,
        "nccl_collectives_certified": True,
        "baseline_allocated_bytes": 10,
        "baseline_reserved_bytes": 20,
        "peak_allocated_bytes": min(600, reserved),
        "peak_reserved_bytes": reserved,
        "device_total_bytes": total,
        "device_free_bytes_min": free,
        "peak_reserved_fraction": reserved / total,
        "max_reserved_fraction": 0.90,
    }


def test_memory_gate_preserves_strict_original_boundaries():
    assert all(memory_gate_checks(_memory(reserved=899, free=101), 0).values())
    at_reserved_limit = memory_gate_checks(_memory(reserved=900, free=101), 0)
    assert at_reserved_limit["peak_reserved_below_90_percent"] is False
    at_free_limit = memory_gate_checks(_memory(reserved=899, free=100), 0)
    assert at_free_limit["sampled_free_above_10_percent"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"baseline_allocated_bytes": -1},
        {"peak_reserved_bytes": 800.0},
        {"device_total_bytes": True},
        {"peak_reserved_fraction": float("nan")},
        {"peak_reserved_fraction": 0.5},
        {"device": "cuda:-1"},
        {"cuda_local_rank_only": False},
        {"nccl_collectives_certified": False},
    ],
)
def test_memory_gate_names_invalid_counter_and_metadata_conditions(change):
    memory = _memory()
    memory.update(change)
    checks = memory_gate_checks(memory, 0)
    assert not all(checks.values())


class _FakeCuda:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.index = -1

    @property
    def current(self):
        return self.snapshots[self.index]

    def synchronize(self, _device):
        self.index += 1

    def memory_allocated(self, _device):
        return self.current["allocated"]

    def memory_reserved(self, _device):
        return self.current["reserved"]

    def mem_get_info(self, _device):
        return self.current["free"], self.current["driver_total"]

    def get_device_properties(self, _device):
        return SimpleNamespace(
            total_memory=self.current["total"], name="Fake H100", uuid="GPU-test"
        )

    def memory_stats(self, _device):
        return self.current.get("stats", {})

    def max_memory_allocated(self, _device):
        return self.current["peak_allocated"]

    def max_memory_reserved(self, _device):
        return self.current["peak_reserved"]


@pytest.fixture
def fake_cuda(monkeypatch):
    fake = _FakeCuda(
        [
            {
                "allocated": 300,
                "reserved": 400,
                "free": 900,
                "driver_total": 1000,
                "total": 1000,
                "peak_allocated": 700,
                "peak_reserved": 850,
                "stats": {"num_ooms": 0},
            },
            {
                "allocated": 100,
                "reserved": 200,
                "free": 500,
                "driver_total": 1000,
                "total": 1000,
                "peak_allocated": 650,
                "peak_reserved": 800,
                "stats": {"num_ooms": 0},
            },
        ]
    )
    monkeypatch.setattr(torch, "cuda", fake)
    return fake


def _diagnostics(tmp_path):
    return ShardedProbeDiagnostics(
        tmp_path,
        device="cuda:0",
        rank=0,
        local_rank=0,
        world_size=4,
        accumulation_steps=8,
        config={"zero_optimization": {"stage": 2}},
        sequence_length=6144,
    )


def test_sample_uses_same_epoch_for_raw_external_estimate_and_persists_running(
    tmp_path, fake_cuda
):
    diagnostics = _diagnostics(tmp_path)
    first = diagnostics.sample("post_engine", microsteps=1, optimizer_steps=0)
    assert first["non_pytorch_used_estimate_bytes"] == 1000 - 900 - 400 == -300
    assert first["reserved_unallocated_bytes"] == 100
    assert diagnostics.minimum_free == 900
    persisted = json.loads(diagnostics.path.read_text(encoding="utf-8"))
    assert persisted["status"] == "running"
    assert persisted["diagnostic_only"] is True
    assert persisted["certifies_training"] is False
    assert persisted["samples"] == diagnostics.samples


def test_raw_external_estimate_can_be_negative_and_is_never_clamped(tmp_path, fake_cuda):
    diagnostics = _diagnostics(tmp_path)
    diagnostics.sample("first")
    assert diagnostics.samples[0]["non_pytorch_used_estimate_bytes"] == -300


def test_memory_report_separates_allocator_peaks_from_discrete_minimum_free(
    tmp_path, fake_cuda
):
    diagnostics = _diagnostics(tmp_path)
    first = diagnostics.sample("allocator_peak")
    second = diagnostics.sample("free_minimum")
    report = diagnostics.memory_report(10, 20)
    assert first["peak_reserved_bytes"] == 850
    assert second["device_free_bytes"] == diagnostics.minimum_free == 500
    assert report["peak_reserved_bytes"] == 850
    assert report["device_free_bytes_min"] == 500
    assert report["device_free_fraction_min"] == 0.5
    assert report["device_free_bytes_min_is_sampled"] is True
    assert report["final_reserved_bytes"] == 200
    assert diagnostics.events[-1] == {
        "phase": "allocator_peak_counter_decreased", "observed_at": "free_minimum",
        "previous_phase": "allocator_peak",
        "counters": ["peak_allocated_bytes", "peak_reserved_bytes"],
        "note": "An intervening peak-counter reset is indicated; earlier observed peaks are retained.",
    }
    assert report["nccl_collectives_certified"] is False
    diagnostics.nccl_collectives_certified = True
    assert diagnostics.memory_report(10, 20)["nccl_collectives_certified"] is True


def test_publish_persists_failed_diagnostic_never_canonical_receipt(tmp_path, fake_cuda):
    diagnostics = _diagnostics(tmp_path)
    diagnostics.partition_snapshots.append(
        {"phase": "pre_step", "local_fragment_numel": 123}
    )
    diagnostics.sample("pre_step", microsteps=8, optimizer_steps=0)
    memory = diagnostics.memory_report(10, 20)
    diagnostics.publish("failed", memory=memory, error=RuntimeError("boom"))
    assert diagnostics.path.name == "sharded_preflight_diagnostics_rank0.json"
    persisted = json.loads(diagnostics.path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["diagnostic_only"] is True
    assert persisted["error"] == {"type": "RuntimeError", "message": "boom"}
    assert persisted["partition_snapshots"] == diagnostics.partition_snapshots
    assert "full_optimizer_preflight" not in diagnostics.path.name
    assert "full_optimizer_preflight" not in diagnostics.path.read_text(encoding="utf-8")


@pytest.mark.parametrize("fail", [False, True])
def test_setup_capture_records_cpu_cuda_decisions_and_restores_logger(tmp_path, fail):
    diagnostics = _diagnostics(tmp_path)
    logger = logging.getLogger("DeepSpeed")
    original_level, original_handlers = logger.level, list(logger.handlers)
    try:
        logger.setLevel(logging.WARNING)
        try:
            with diagnostics.capture_deepspeed_setup_events():
                logger.info("Flattening param group 0 on cuda (sufficient memory)")
                logger.info("Flattening param group 1 on CPU (insufficient memory)")
                logger.info("Ignore ordinary DeepSpeed informational output")
                if fail:
                    raise RuntimeError("Trainer failed")
        except RuntimeError as error:
            assert fail and str(error) == "Trainer failed"
        assert logger.level == logging.WARNING
        assert logger.handlers == original_handlers
    finally:
        logger.setLevel(original_level)
    assert len(diagnostics.events) == 2
    assert [event["flatten_device"] for event in diagnostics.events] == ["cuda", "CPU"]
    assert [event["group_index"] for event in diagnostics.events] == [0, 1]
    assert all(event["source"] == "DeepSpeed INFO log" for event in diagnostics.events)


def test_failure_record_keeps_last_sample_if_cuda_cannot_be_queried(tmp_path, fake_cuda, monkeypatch):
    diagnostics = _diagnostics(tmp_path)
    diagnostics.sample("before_backward", microsteps=7)

    def broken_cuda(_device):
        raise RuntimeError("CUDA unavailable after failure")

    monkeypatch.setattr(fake_cuda, "synchronize", broken_cuda)
    diagnostics.record_failure(RuntimeError("original OOM"), baseline_allocated=10,
                               baseline_reserved=20, microsteps=7, optimizer_steps=0)
    persisted = json.loads(diagnostics.path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["phase"] == "before_backward"
    assert persisted["error"]["message"] == "original OOM"
    assert persisted["memory"]["peak_reserved_bytes"] == 850
    assert persisted["events"][-1]["phase"] == "exception_sample"
    assert "CUDA unavailable" in persisted["events"][-1]["error"]
