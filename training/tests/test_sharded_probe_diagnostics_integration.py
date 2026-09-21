from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from ir_training.qat.full_model_contract import validate_optimizer_probe
from ir_training.train import sharded_diagnostics, sharded_preflight
from ir_training.train.full_parameters import FullParameterScopeError
from ir_training.train.sharded_contract import build_deepspeed_config


@dataclass
class _Arguments:
    output_dir: str
    deepspeed: dict
    gradient_accumulation_steps: int = 8
    per_device_train_batch_size: int = 1
    learning_rate: float = 1e-5
    bf16: bool = False
    fp16: bool = False
    max_grad_norm: float = 0.0
    accelerator_config: dict = field(default_factory=dict)
    max_steps: int = 100
    save_strategy: str = "steps"
    eval_strategy: str = "steps"
    logging_strategy: str = "steps"
    report_to: list = field(default_factory=list)
    push_to_hub: bool = False
    load_best_model_at_end: bool = False
    disable_tqdm: bool = False
    skip_memory_metrics: bool = False


class _Device:
    type = "cuda"

    def __init__(self, index: int):
        self.index = index

    def __str__(self):
        return f"cuda:{self.index}"

    def __eq__(self, other):
        return isinstance(other, _Device) and self.index == other.index


class _Parameter:
    dtype = torch.float32
    requires_grad = True
    grad = None

    def __init__(self, device: _Device):
        self.device = device

    def numel(self):
        return 10


class _Model:
    def __init__(self, device: _Device):
        self.parameter = _Parameter(device)

    def named_parameters(self):
        return [("weight", self.parameter)]


class _Cuda:
    def __init__(self, *, total=1000, free=200, allocated=400, reserved=800,
                 peak_allocated=500, peak_reserved=850):
        self.total = total
        self.free = free
        self.allocated = allocated
        self.reserved = reserved
        self.peak_allocated = peak_allocated
        self.peak_reserved = peak_reserved
        self.fail_sampling = False

    def synchronize(self, _device):
        if self.fail_sampling:
            raise RuntimeError("cuda sampling failed")

    def empty_cache(self):
        pass

    def reset_peak_memory_stats(self, _device):
        pass

    def memory_allocated(self, _device):
        return self.allocated

    def memory_reserved(self, _device):
        return self.reserved

    def mem_get_info(self, _device):
        return self.free, self.total

    def get_device_properties(self, _device):
        return SimpleNamespace(total_memory=self.total, name="fake", uuid="fake-uuid")

    def memory_stats(self, _device):
        return {}

    def max_memory_allocated(self, _device):
        return self.peak_allocated

    def max_memory_reserved(self, _device):
        return self.peak_reserved


class _Zero:
    overflow = False

    def __init__(self, fail_update=False):
        self.fail_update = fail_update

    def step(self):
        if self.fail_update:
            raise RuntimeError("optimizer update failed")


def _trainer_class(*, fail_at: str | None = None, cuda: _Cuda | None = None):
    class FakeCheckedTrainer:
        def __init__(self, **kwargs):
            if fail_at == "init":
                if cuda is not None:
                    cuda.fail_sampling = True
                raise RuntimeError("trainer init failed")
            self.args = kwargs["args"]
            self.state = SimpleNamespace(global_step=0)
            self.engine = SimpleNamespace(optimizer=_Zero(fail_at == "optimizer"))
            self._base_microstep = 0

        def training_step(self, engine, _inputs, *_args, **_kwargs):
            if (self._base_microstep + 1) % self.args.gradient_accumulation_steps == 0:
                engine.optimizer.step()
            self._base_microstep += 1
            return object()

        def train(self):
            if fail_at == "train":
                raise RuntimeError("trainer train failed")
            for _ in range(2 * self.args.gradient_accumulation_steps):
                self.training_step(self.engine, {})
            self.state.global_step = 2

    return FakeCheckedTrainer


@pytest.fixture
def probe_runtime(monkeypatch, tmp_path):
    rank = 1
    world = 4
    cuda = _Cuda()
    training = {
        "distributed_backend": "sharded",
        "optim": "adamw_torch",
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 0.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
    }
    args = _Arguments(str(tmp_path), build_deepspeed_config(training, world))
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setattr(torch, "cuda", cuda)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(sharded_preflight, "assert_sharded_engine", lambda *_args: None)
    monkeypatch.setattr(sharded_preflight, "summarize_zero_partitions",
                        lambda *_args, **_kwargs: {"logical_partition_bytes": 40})
    monkeypatch.setattr(sharded_preflight, "_collective_gradient_audit", lambda *_args: {
        "verified": True, "unique_parameters": 1, "global_numel": 10, "ranks": world,
        "method": "exhaustive_local_fragments_exact_global_interval_union",
    })
    monkeypatch.setattr(sharded_preflight, "audit_adamw_state", lambda *_args, **_kwargs: {
        "name": "AdamW", "betas": [0.9, 0.999], "epsilon": 1e-8,
        "weight_decay": 0.0, "external_max_grad_norm_required": 0.0,
        "state_tensor_count": 2, "state_numel": 10,
        "state_scope": "rank_local_fp32_partitions",
    })
    monkeypatch.setattr(sharded_preflight, "tensor_all_finite", lambda _value: True)
    return SimpleNamespace(
        rank=rank, world=world, cuda=cuda, training=training, args=args,
        kwargs={"args": args, "model": _Model(_Device(rank))},
        row={"input_ids": list(range(6144))},
        path=tmp_path / f"sharded_preflight_diagnostics_rank{rank}.json",
        canonical=tmp_path / f"full_optimizer_preflight_rank{rank}.json",
    )


def _run(runtime, trainer_cls):
    return sharded_preflight.run_sharded_trainer_preflight(
        trainer_cls, runtime.kwargs, training_cfg=runtime.training, longest_row=runtime.row,
    )


def test_success_keeps_schema3_effective_batch_and_diagnostic_is_not_a_receipt(probe_runtime):
    report = _run(probe_runtime, _trainer_class())

    validate_optimizer_probe(
        report, accumulation_steps=8, world_size=4, local_rank=1,
        distributed_backend="sharded",
    )
    diagnostic = json.loads(probe_runtime.path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 3
    assert report["sharded_probe"]["microsteps"] == 16
    assert diagnostic["status"] == "validated"
    assert diagnostic["effective_batch"] == 32
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["certifies_training"] is False
    with pytest.raises(FullParameterScopeError, match="Missing successful disposable"):
        validate_optimizer_probe(
            diagnostic, accumulation_steps=8, world_size=4, local_rank=1,
            distributed_backend="sharded",
        )


def test_all_ranks_flush_and_emit_complete_pending_counters_before_validation_failure(
        probe_runtime, monkeypatch):
    order = []
    emitted = []
    real_validate = sharded_preflight.validate_sharded_probe

    def capture_log(message):
        if message.startswith("Sharded preflight memory: "):
            emitted.append(json.loads(message.removeprefix("Sharded preflight memory: ")))
            order.append("emit")

    def barrier():
        persisted = json.loads(probe_runtime.path.read_text(encoding="utf-8"))
        assert persisted["status"] == "pending_validation"
        assert persisted["samples"] and persisted["partition_snapshots"]
        order.append("barrier_after_flush")

    def reject(report, **kwargs):
        order.append("validate")
        return real_validate(report, **kwargs)

    probe_runtime.cuda.peak_reserved = 900
    probe_runtime.cuda.free = 100
    monkeypatch.setattr(sharded_diagnostics, "log", capture_log)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(sharded_preflight, "validate_sharded_probe", reject)

    with pytest.raises(FullParameterScopeError, match="peak_reserved_below_90_percent.*sampled_free_above_10_percent"):
        _run(probe_runtime, _trainer_class())

    failed = json.loads(probe_runtime.path.read_text(encoding="utf-8"))
    pending = next(item for item in emitted if item["status"] == "pending_validation")
    assert order.index("emit") < order.index("barrier_after_flush") < order.index("validate")
    assert pending["rank"] == 1 and pending["local_rank"] == 1
    assert pending["memory"]["peak_reserved_bytes"] == 900
    assert pending["memory"]["device_free_bytes_min"] == 100
    assert pending["checks"]["peak_reserved_below_90_percent"] is False
    assert pending["checks"]["sampled_free_above_10_percent"] is False
    assert failed["status"] == "failed"
    assert failed["error"]["type"] == "FullParameterScopeError"
    assert failed["samples"][-1]["microsteps"] == 16
    assert failed["samples"][-1]["optimizer_steps"] == 2
    assert failed["validation_attempt"]["memory"] == pending["memory"]
    assert failed["validation_attempt"]["failed_conditions"] == pending["failed_conditions"]
    assert not probe_runtime.canonical.exists()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("init", "trainer init failed"),
        ("train", "trainer train failed"),
        ("gradient", "gradient audit failed"),
        ("optimizer", "optimizer update failed"),
        ("validation", "final validation failed"),
    ],
)
def test_original_lifecycle_errors_are_preserved(probe_runtime, monkeypatch, failure, message):
    if failure == "gradient":
        monkeypatch.setattr(
            sharded_preflight, "_collective_gradient_audit",
            lambda *_args: (_ for _ in ()).throw(RuntimeError(message)),
        )
    if failure == "validation":
        monkeypatch.setattr(
            sharded_preflight, "validate_sharded_probe",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(message)),
        )
    trainer_failure = failure if failure in {"init", "train", "optimizer"} else None

    with pytest.raises(RuntimeError, match=message):
        _run(probe_runtime, _trainer_class(fail_at=trainer_failure))

    diagnostic = json.loads(probe_runtime.path.read_text(encoding="utf-8"))
    assert diagnostic["status"] == "failed"
    assert diagnostic["error"] == {"type": "RuntimeError", "message": message}
    assert diagnostic["memory"]["nccl_collectives_certified"] is (failure in {"optimizer", "validation"})
    assert diagnostic["baseline_allocated_bytes"] == 400
    assert diagnostic["baseline_reserved_bytes"] == 800


def test_failed_exception_sampling_does_not_replace_original_trainer_error(probe_runtime):
    with pytest.raises(RuntimeError, match="trainer init failed"):
        _run(probe_runtime, _trainer_class(fail_at="init", cuda=probe_runtime.cuda))

    diagnostic = json.loads(probe_runtime.path.read_text(encoding="utf-8"))
    assert diagnostic["status"] == "failed"
    assert diagnostic["error"]["message"] == "trainer init failed"
    exception_event = next(item for item in diagnostic["events"] if item["phase"] == "exception_sample")
    assert "cuda sampling failed" in exception_event["error"]
