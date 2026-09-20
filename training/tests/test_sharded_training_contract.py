from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from ir_training.train.sharded_contract import (
    assert_sharded_engine,
    build_deepspeed_config,
    deepspeed_config_sha256,
    validate_backend,
    validate_sharded_config,
    validate_sharded_runtime,
)


def _sharded_config() -> dict:
    return {
        "distributed_backend": "sharded",
        "optim": "adamw_torch",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 0.0,
    }


def test_ddp_is_default_and_never_requires_deepspeed(monkeypatch):
    def forbidden_import(*_args, **_kwargs):
        raise AssertionError("DDP validation imported a dependency")

    monkeypatch.setattr("builtins.__import__", forbidden_import)
    assert validate_backend({}) == "ddp"
    assert validate_backend({"distributed_backend": "ddp"}) == "ddp"


@pytest.mark.parametrize("value", ["zero2", "fsdp", "DDP", "", 2, True, None])
def test_backend_selector_has_a_narrow_allowlist(value):
    with pytest.raises(ValueError, match="exactly 'ddp' or 'sharded'"):
        validate_backend({"distributed_backend": value})


def test_builds_exact_hash_bound_zero2_contract():
    config = build_deepspeed_config(_sharded_config(), world_size=8)
    assert config["train_micro_batch_size_per_gpu"] == 1
    assert config["gradient_accumulation_steps"] == 4
    assert config["train_batch_size"] == 32
    assert config["communication_data_type"] == "fp32"
    assert config["fp16"] == {"enabled": False}
    assert config["bf16"] == {"enabled": False}
    assert config["torch_autocast"] == {
        "enabled": True,
        "dtype": "bfloat16",
        "lower_precision_safe_modules": [],
    }
    assert config["zero_optimization"] == {
        "stage": 2,
        "contiguous_gradients": True,
        "reduce_scatter": True,
        "overlap_comm": False,
        "reduce_bucket_size": 5_000_000,
        "allgather_bucket_size": 5_000_000,
        "allgather_partitions": True,
    }
    assert "optimizer" not in config
    assert "scheduler" not in config
    assert "offload_optimizer" not in config["zero_optimization"]
    assert "offload_param" not in config["zero_optimization"]
    assert len(deepspeed_config_sha256(config)) == 64


def test_canonical_hash_is_key_order_independent():
    left = {"a": 1, "b": {"x": 2, "y": 3}}
    right = json.loads('{"b":{"y":3,"x":2},"a":1}')
    assert deepspeed_config_sha256(left) == deepspeed_config_sha256(right)


def test_effective_batch_must_match_derived_product():
    config = _sharded_config()
    config["expected_effective_batch_size"] = 7
    with pytest.raises(ValueError, match="does not match"):
        build_deepspeed_config(config, world_size=2)
    config["expected_effective_batch_size"] = 8
    assert build_deepspeed_config(config, world_size=2)["train_batch_size"] == 8


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"distributed_backend": "ddp"}, "only valid"),
        ({"optim": "adafactor"}, "adamw_torch"),
        ({"deepspeed": "arbitrary.json"}, "repository-owned"),
        ({"offload_optimizer": True}, "unsupported settings"),
        ({"per_device_train_batch_size": 0}, "positive integer"),
    ],
)
def test_sharded_validation_fails_closed(change, message):
    config = _sharded_config()
    config.update(change)
    with pytest.raises(ValueError, match=message):
        validate_sharded_config(config)


def test_runtime_version_gate_calls_nvtx_probe_only_after_runtime_checks(monkeypatch):
    versions = {"deepspeed": "0.19.7", "transformers": "5.16.1", "accelerate": "1.15.0",
                "nvtx": "0.2.15"}
    probes = []
    monkeypatch.setattr("ir_training.train.sharded_environment.probe_nvtx_compatibility",
                        lambda: probes.append(True) or {"passed": True})
    monkeypatch.setattr("importlib.metadata.version", versions.__getitem__)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, is_bf16_supported=lambda: True)))
    assert validate_sharded_runtime()["packages"] == versions
    assert probes == [True]


def test_runtime_version_gate_rejects_unreviewed_release(monkeypatch):
    versions = {"deepspeed": "0.19.6", "transformers": "5.16.1", "accelerate": "1.15.0"}
    monkeypatch.setattr("importlib.metadata.version", versions.__getitem__)
    with pytest.raises(RuntimeError, match="requires reviewed deepspeed 0.19.7"):
        validate_sharded_runtime()


class _Parameter:
    def __init__(self, dtype="torch.float32"):
        self.dtype = dtype
        self.requires_grad = True


class _Module:
    def __init__(self, dtype="torch.float32"):
        self._parameters = [_Parameter(dtype)]

    def parameters(self):
        return iter(self._parameters)


class _Engine:
    def __init__(self, *, stage=2, partition=True, world_size=4,
                 autocast=True, dtype="torch.bfloat16", parameter_dtype="torch.float32"):
        self.module = _Module(parameter_dtype)
        self.world_size = world_size
        self._stage = stage
        self._partition = partition
        self._autocast = autocast
        self._dtype = dtype
        self._config = build_deepspeed_config(_sharded_config(), world_size)

    def zero_optimization_stage(self):
        return self._stage

    def zero_optimization_partition_gradients(self):
        return self._partition

    def torch_autocast_enabled(self):
        return self._autocast

    def torch_autocast_dtype(self):
        return self._dtype


def test_runtime_assertion_accepts_exact_fake_engine():
    report = assert_sharded_engine(_Engine(), _sharded_config(), world_size=4)
    assert report["zero_stage"] == 2
    assert report["model_parameters_replicated"] is True
    assert report["model_parameter_dtype"] == "float32"


@pytest.mark.parametrize(
    "engine",
    [
        _Engine(stage=3),
        _Engine(partition=False),
        _Engine(world_size=2),
        _Engine(autocast=False),
        _Engine(dtype="torch.float16"),
        _Engine(parameter_dtype="torch.bfloat16"),
    ],
)
def test_runtime_assertion_rejects_contract_drift(engine):
    with pytest.raises(RuntimeError, match="Sharded runtime contract failed"):
        assert_sharded_engine(engine, _sharded_config(), world_size=4)


@pytest.mark.parametrize(("key", "value"), [
    ("train_micro_batch_size_per_gpu", 2), ("gradient_accumulation_steps", 3),
    ("train_batch_size", 99), ("gradient_clipping", 1.0),
])
def test_runtime_assertion_rejects_canonical_batch_or_clipping_drift(key, value):
    engine = _Engine()
    engine._config[key] = value
    with pytest.raises(RuntimeError, match=f"effective DeepSpeed {key}"):
        assert_sharded_engine(engine, _sharded_config(), world_size=4)
