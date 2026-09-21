"""CPU contracts for bounded ZeRO-3 partition evidence."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from ir_training.train.full_parameters import FullParameterScopeError
from ir_training.train.sharded_contract import (
    build_deepspeed_config,
    deepspeed_config_sha256,
)
from ir_training.train.sharded_preflight import (
    ZERO3_PROBE_KIND,
    validate_partition_coverage,
    validate_sharded_probe,
)
from ir_training.train.zero3_partitions import (
    audit_adamw_state,
    local_gradient_intervals,
    local_parameter_partition,
    parameters_finite,
)


def _parameter(total: int, values: list[float]):
    return SimpleNamespace(ds_numel=total, ds_tensor=torch.tensor(values, dtype=torch.float32))


def _install_safe_grad(monkeypatch, gradients):
    package = ModuleType("deepspeed")
    utils = ModuleType("deepspeed.utils")
    utils.safe_get_local_grad = lambda parameter: gradients[id(parameter)]
    package.utils = utils
    monkeypatch.setitem(sys.modules, "deepspeed", package)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", utils)


def test_zero3_local_gradient_intervals_exclude_validated_padding(monkeypatch):
    first = _parameter(5, [1, 2, 3])
    last = _parameter(5, [4, 5, 0])
    _install_safe_grad(monkeypatch, {
        id(first): torch.tensor([0.0, 2.0, 0.0]),
        id(last): torch.tensor([3.0, 0.0, 0.0]),
    })
    left, left_nonzero = local_gradient_intervals([("p", first)], rank=0, world_size=2)
    right, right_nonzero = local_gradient_intervals([("p", last)], rank=1, world_size=2)
    assert left == {"p": [0, 3]}
    assert right == {"p": [3, 2]}
    assert left_nonzero and right_nonzero
    validate_partition_coverage({"p": 5}, [left, right])


@pytest.mark.parametrize(
    ("gradient", "message"),
    [
        (torch.tensor([1.0, 2.0]), "Missing/mismatched"),
        (torch.tensor([1.0, 2.0, float("nan")]), "Non-finite"),
        (torch.tensor([1.0, 2.0, 1.0]), "padding"),
        (torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64), "remain FP32"),
    ],
)
def test_zero3_local_gradient_rejects_bad_partition_or_padding(monkeypatch, gradient, message):
    parameter = _parameter(5, [1, 2, 0])
    _install_safe_grad(monkeypatch, {id(parameter): gradient})
    with pytest.raises(FullParameterScopeError, match=message):
        local_gradient_intervals([("p", parameter)], rank=1, world_size=2)


def test_zero3_partition_requires_exact_nonempty_equal_shard():
    with pytest.raises(FullParameterScopeError, match="Missing/empty"):
        local_parameter_partition(_parameter(4, []), "p", rank=0, world_size=2)
    with pytest.raises(FullParameterScopeError, match="Mismatched"):
        local_parameter_partition(_parameter(5, [1, 2]), "p", rank=0, world_size=2)


def test_zero3_adamw_audits_exact_flat_master_state():
    master = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    raw = torch.optim.AdamW([master], lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    raw.state[master] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(master),
        "exp_avg_sq": torch.zeros_like(master),
    }
    result = audit_adamw_state(
        SimpleNamespace(optimizer=raw, fp32_partitioned_groups_flat=[master]),
        require_state=True,
    )
    assert result["state_scope"] == "rank_local_zero3_fp32_flat_partitions"
    assert result["state_numel"] == 4
    assert result["master_partition_numel"] == 2


def test_zero3_adamw_allows_temporarily_empty_raw_group_params():
    master = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    raw = torch.optim.AdamW([master], lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    raw.state[master] = {"step": torch.tensor(1.0), "exp_avg": torch.zeros_like(master),
                         "exp_avg_sq": torch.zeros_like(master)}
    raw.param_groups[0]["params"] = []
    result = audit_adamw_state(
        SimpleNamespace(optimizer=raw, fp32_partitioned_groups_flat=[master]),
        require_state=True,
    )
    assert result["state_tensor_count"] == 2


def test_zero3_adamw_allows_initialized_step_zero_only_before_update():
    master = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    raw = torch.optim.AdamW([master], lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    raw.state[master] = {"step": torch.tensor(0.0), "exp_avg": torch.zeros_like(master),
                         "exp_avg_sq": torch.zeros_like(master)}
    zero = SimpleNamespace(optimizer=raw, fp32_partitioned_groups_flat=[master])
    assert audit_adamw_state(zero, require_state=False)["state_tensor_count"] == 2
    with pytest.raises(FullParameterScopeError, match="did not perform an update"):
        audit_adamw_state(zero, require_state=True)


def _probe3() -> dict:
    training = {"distributed_backend": "sharded", "zero_stage": 3,
                "optim": "adamw_torch", "weight_decay": 0.0, "max_grad_norm": 0.0,
                "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4}
    config = build_deepspeed_config(training, 2)
    return {
        "schema_version": 4, "probe": ZERO3_PROBE_KIND, "passed": True,
        "disposable_worker_required": True, "model_must_not_be_reused": True,
        "checkpoint_writes": 0, "disposable_optimizer_steps": 2,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "epsilon": 1e-8,
                      "weight_decay": 0.0, "external_max_grad_norm_required": 0.0,
                      "state_tensor_count": 2, "state_numel": 8,
                      "master_partition_numel": 4,
                      "state_scope": "rank_local_zero3_fp32_flat_partitions",
                      "learning_rate": 1e-5},
        "scope": {"unique_parameter_count": 2, "trainable_numel": 10,
                  "all_trainable_fp32": True, "all_gradients_finite": True,
                  "all_parameters_finite_after_step": True},
        "sharded_probe": {"backend": "sharded", "zero_stage": 3, "world_size": 2,
                          "gradient_accumulation_steps": 4, "microsteps": 8,
                          "optimizer_steps": 2, "nccl_collectives_certified": True,
                          "model_parameters_replicated": False,
                          "gradient_coverage": {"verified": True, "unique_parameters": 2,
                                                "global_numel": 10, "ranks": 2,
                                                "method": "exhaustive_zero3_local_grad_partitions_exact_global_interval_union"},
                          "config": config, "config_sha256": deepspeed_config_sha256(config)},
        "memory": {"device": "cuda:0", "cuda_local_rank_only": True,
                   "nccl_collectives_certified": True, "baseline_allocated_bytes": 10,
                   "baseline_reserved_bytes": 20, "peak_allocated_bytes": 60,
                   "peak_reserved_bytes": 80, "device_total_bytes": 1000,
                   "device_free_bytes_min": 200, "peak_reserved_fraction": 0.08,
                   "max_reserved_fraction": 0.90},
        "selection": {"longest_sequence_length": 64, "repeated_real_prepared_row": True},
    }


def test_schema4_accepts_zero3_and_cross_stage_rejects():
    validate_sharded_probe(_probe3(), accumulation_steps=4, world_size=2,
                           local_rank=0, zero_stage=3)
    with pytest.raises(FullParameterScopeError):
        validate_sharded_probe(_probe3(), accumulation_steps=4, world_size=2,
                               local_rank=0, zero_stage=2)


@pytest.mark.parametrize("value", [3.0, True, "3"])
def test_schema4_rejects_non_integer_persisted_zero_stage(value):
    probe = _probe3()
    probe["sharded_probe"]["zero_stage"] = value
    with pytest.raises(FullParameterScopeError):
        validate_sharded_probe(probe, accumulation_steps=4, world_size=2,
                               local_rank=0, zero_stage=3)


@pytest.mark.parametrize("value", [2, 3.0, True, "3"])
def test_schema4_rejects_wrong_or_non_integer_config_zero_stage(value):
    probe = _probe3()
    probe["sharded_probe"]["config"]["zero_optimization"]["stage"] = value
    probe["sharded_probe"]["config_sha256"] = deepspeed_config_sha256(
        probe["sharded_probe"]["config"]
    )
    with pytest.raises(FullParameterScopeError):
        validate_sharded_probe(probe, accumulation_steps=4, world_size=2,
                               local_rank=0, zero_stage=3)


def test_zero3_parameter_finiteness_reads_local_shard_only():
    good = _parameter(3, [1.0, 2.0])
    parameters_finite([("p", good)], rank=0, world_size=2)
    bad = _parameter(3, [1.0, float("nan")])
    with pytest.raises(FullParameterScopeError, match="Non-finite updated"):
        parameters_finite([("p", bad)], rank=0, world_size=2)
