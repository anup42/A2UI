"""CPU-only contract tests for the disposable ZeRO-2 preflight evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ir_training.common.config import load_yaml
from ir_training.qat import full_model_contract as full_contract
from ir_training.qat.fake_quant import QATSpec
from ir_training.qat.full_model_contract import (
    require_optimizer_preflight,
    validate_optimizer_probe,
    write_optimizer_preflight,
)
from ir_training.train.full_parameters import FullParameterScopeError
from ir_training.train.sft import _build_checked_causal_lm_trainer
from ir_training.train.sharded_contract import (
    build_deepspeed_config,
    deepspeed_config_sha256,
)
from ir_training.train.sharded_preflight import (
    PROBE_KIND,
    local_gradient_intervals,
    validate_partition_coverage,
    validate_sharded_probe,
)
from safetensors.torch import save_file


def _training(*, output_dir: Path | None = None) -> dict:
    result = {
        "distributed_backend": "sharded",
        "optim": "adamw_torch",
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 0.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
    }
    return result


def _probe(*, world_size: int = 2, accumulation: int = 4) -> dict:
    training = _training()
    training["gradient_accumulation_steps"] = accumulation
    ds_config = build_deepspeed_config(training, world_size)
    return {
        "schema_version": 3,
        "probe": PROBE_KIND,
        "passed": True,
        "disposable_worker_required": True,
        "model_must_not_be_reused": True,
        "checkpoint_writes": 0,
        "disposable_optimizer_steps": 2,
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "external_max_grad_norm_required": 0.0,
            "state_tensor_count": 4,
            "state_numel": 20,
            "state_scope": "rank_local_fp32_partitions",
            "learning_rate": 1e-5,
        },
        "scope": {
            "unique_parameter_count": 2,
            "trainable_numel": 10,
            "all_trainable_fp32": True,
            "all_gradients_finite": True,
            "all_parameters_finite_after_step": True,
        },
        "sharded_probe": {
            "backend": "sharded",
            "zero_stage": 2,
            "world_size": world_size,
            "gradient_accumulation_steps": accumulation,
            "microsteps": 2 * accumulation,
            "optimizer_steps": 2,
            "nccl_collectives_certified": True,
            "model_parameters_replicated": True,
            "gradient_coverage": {
                "verified": True,
                "unique_parameters": 2,
                "global_numel": 10,
                "ranks": world_size,
                "method": "exhaustive_local_fragments_exact_global_interval_union",
            },
            "config": ds_config,
            "config_sha256": deepspeed_config_sha256(ds_config),
        },
        "memory": {
            "device": "cuda:0",
            "cuda_local_rank_only": True,
            "nccl_collectives_certified": True,
            "baseline_allocated_bytes": 10,
            "baseline_reserved_bytes": 20,
            "peak_allocated_bytes": 60,
            "peak_reserved_bytes": 80,
            "device_total_bytes": 1000,
            "device_free_bytes_min": 200,
            "peak_reserved_fraction": 0.08,
            "max_reserved_fraction": 0.90,
        },
        "selection": {
            "longest_sequence_length": 6144,
            "repeated_real_prepared_row": True,
        },
    }


def test_partition_interval_union_exactly_covers_each_parameter():
    validate_partition_coverage(
        {"a": 7, "b": 3},
        [{"a": [0, 3], "b": [0, 1]}, {"a": [3, 4], "b": [1, 2]}],
    )


@pytest.mark.parametrize(
    ("shapes", "intervals", "message"),
    [
        ({"a": 5}, [{"a": [0, 2]}, {"a": [3, 2]}], "Overlapping/missing"),
        ({"a": 5}, [{"a": [0, 3]}, {"a": [2, 3]}], "Overlapping/missing"),
        ({"a": 5}, [{"a": [0, 4]}], "Incomplete"),
        ({"a": 5}, [{"a": [0, 5], "extra": [0, 1]}], "Unknown"),
        ({"a": 5}, [{"a": [0, 0]}], "Overlapping/missing"),
    ],
)
def test_partition_interval_union_rejects_gaps_overlap_and_unexpected(
    shapes, intervals, message
):
    with pytest.raises(FullParameterScopeError, match=message):
        validate_partition_coverage(shapes, intervals)


class _Mapping:
    def __init__(self, start: int, count: int, gradient):
        self.lp_fragment_address = SimpleNamespace(start=start, numel=count)
        self.gradient = gradient

    def get_lp_grad_fragment(self, _index):
        return self.gradient


class _Parameter:
    def __init__(self, numel: int, mapping=...):
        self._numel = numel
        self._index_in_param_group = 0
        if mapping is not ...:
            self._hp_mapping = mapping

    def numel(self):
        return self._numel


def test_local_gradient_intervals_reads_only_fragment_and_handles_zero_owner():
    fragment = torch.tensor([0.0, 2.0], dtype=torch.float32)
    parameters = [
        ("large", _Parameter(10_000_000, _Mapping(123, 2, fragment))),
        ("elsewhere", _Parameter(4, None)),
    ]
    intervals, nonzero = local_gradient_intervals(parameters)
    assert intervals == {"large": [123, 2]}
    assert nonzero is True
    assert fragment.numel() == 2


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        (_Parameter(4), "Missing DeepSpeed"),
        (_Parameter(4, _Mapping(0, 2, None)), "Missing or mismatched"),
        (_Parameter(4, _Mapping(0, 2, torch.ones(1))), "Missing or mismatched"),
        (_Parameter(4, _Mapping(0, 2, torch.tensor([1.0, float("nan")]))), "Non-finite"),
        (_Parameter(4, _Mapping(0, 2, torch.ones(2, dtype=torch.float64))), "remain FP32"),
        (_Parameter(4, _Mapping(3, 2, torch.ones(2))), "Invalid ZeRO fragment"),
    ],
)
def test_local_gradient_intervals_rejects_invalid_fragments(parameter, message):
    with pytest.raises(FullParameterScopeError, match=message):
        local_gradient_intervals([("p", parameter)])


def test_local_gradient_intervals_accepts_all_zero_finite_fragment():
    intervals, nonzero = local_gradient_intervals(
        [("p", _Parameter(3, _Mapping(0, 3, torch.zeros(3, dtype=torch.float32))))]
    )
    assert intervals == {"p": [0, 3]}
    assert nonzero is False


def test_validate_sharded_probe_accepts_strict_schema3_fixture():
    validate_sharded_probe(_probe(), accumulation_steps=4, world_size=2, local_rank=0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version=2),
        lambda p: p.update(probe="disposable_full_parameter_trainer_v2"),
        lambda p: p.update(disposable_optimizer_steps=1),
        lambda p: p["sharded_probe"].update(backend="ddp"),
        lambda p: p["sharded_probe"].update(zero_stage=3),
        lambda p: p["sharded_probe"].update(microsteps=7),
        lambda p: p["sharded_probe"].update(config_sha256="0" * 64),
        lambda p: p["scope"].update(trainable_numel=11),
        lambda p: p["sharded_probe"]["gradient_coverage"].update(verified=False),
        lambda p: p["sharded_probe"]["gradient_coverage"].update(global_numel=9),
        lambda p: p["optimizer"].update(name="Adafactor"),
        lambda p: p["optimizer"].update(state_tensor_count=0),
        lambda p: p["memory"].update(peak_reserved_bytes=900, peak_reserved_fraction=0.9),
        lambda p: p["memory"].update(device_free_bytes_min=100),
        lambda p: p["selection"].update(repeated_real_prepared_row=False),
    ],
)
def test_validate_sharded_probe_rejects_tampering(mutate):
    probe = _probe()
    mutate(probe)
    with pytest.raises(FullParameterScopeError):
        validate_sharded_probe(probe, accumulation_steps=4, world_size=2, local_rank=0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version=3.0),
        lambda p: p.update(disposable_optimizer_steps=2.0),
        lambda p: p["sharded_probe"].update(world_size=2.0),
        lambda p: p["sharded_probe"].update(gradient_accumulation_steps=True),
        lambda p: p["sharded_probe"].update(microsteps=8.0),
        lambda p: p["sharded_probe"].update(optimizer_steps=True),
        lambda p: p["sharded_probe"]["gradient_coverage"].update(ranks=2.0),
        lambda p: p["sharded_probe"]["gradient_coverage"].update(global_numel=10.0),
        lambda p: p["sharded_probe"]["gradient_coverage"].update(unique_parameters=True),
        lambda p: p.update(scope=[]),
        lambda p: p.update(sharded_probe=[]),
        lambda p: p.update(optimizer=[]),
        lambda p: p.update(memory=[]),
        lambda p: p.update(selection=[]),
        lambda p: p["sharded_probe"].update(gradient_coverage=[]),
        lambda p: p["memory"].update(device="cuda"),
        lambda p: p["memory"].update(device="cuda:-1"),
        lambda p: p["memory"].update(device="cuda:00"),
    ],
)
def test_validate_sharded_probe_rejects_wrong_types_and_shapes(mutate):
    probe = _probe()
    mutate(probe)
    with pytest.raises(FullParameterScopeError):
        validate_sharded_probe(probe, accumulation_steps=4, world_size=2, local_rank=0)


@pytest.mark.parametrize("local_rank", [-1, 0.0, True])
def test_validate_sharded_probe_rejects_invalid_local_rank(local_rank):
    with pytest.raises(FullParameterScopeError):
        validate_sharded_probe(
            _probe(), accumulation_steps=4, world_size=2, local_rank=local_rank
        )


def test_cross_backend_validator_rejects_each_others_receipt():
    sharded = _probe()
    validate_optimizer_probe(
        sharded, accumulation_steps=4, world_size=2, local_rank=0,
        distributed_backend="sharded",
    )
    with pytest.raises(ValueError):
        validate_optimizer_probe(
            sharded, accumulation_steps=4, world_size=2, local_rank=0,
            distributed_backend="ddp",
        )


def test_write_and_require_bind_sharded_probe_to_config_and_rank(
    tmp_path: Path, monkeypatch
):
    output = tmp_path / "training"
    config = {
        "run": {"output_dir": str(output)},
        "training": _training(),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    written = write_optimizer_preflight(config, config_path, _probe())
    required = require_optimizer_preflight(config, config_path)
    assert required["sha256"] == written["sha256"]
    assert required["probe"]["sharded_probe"]["backend"] == "sharded"

    config["training"]["learning_rate"] = 2e-5
    with pytest.raises(ValueError, match="matching disposable optimizer preflight"):
        require_optimizer_preflight(config, config_path)


class _FakeTrainer:
    def __init__(
        self,
        *,
        is_deepspeed_enabled=False,
        handler=None,
        gradient_accumulation_plugin=None,
    ):
        self.is_deepspeed_enabled = is_deepspeed_enabled
        self.args = SimpleNamespace(gradient_accumulation_steps=4)
        self.handler = handler
        self.gradient_accumulation_plugin = gradient_accumulation_plugin or SimpleNamespace(
            sync_each_batch=False, num_steps=1
        )
        self.accelerator_args = self._build_accelerator_args(
            gradient_accumulation_plugin=self.gradient_accumulation_plugin
        )
        self.accelerator = SimpleNamespace(
            gradient_state=SimpleNamespace(
                plugin_kwargs={"sync_each_batch": False}, num_steps=4
            ),
            gradient_accumulation_steps=4,
            distributed_type="DEEPSPEED" if is_deepspeed_enabled else "NO",
            state=SimpleNamespace(
                distributed_type="DEEPSPEED" if is_deepspeed_enabled else "NO",
                deepspeed_plugin=SimpleNamespace(
                    get_value=lambda key: 4
                    if key == "gradient_accumulation_steps"
                    else None
                )
                if is_deepspeed_enabled
                else None,
            ),
        )
        self.model_accepts_loss_kwargs = True
        self.state = SimpleNamespace(global_step=0)

    def _build_accelerator_args(self, **kwargs):
        result = dict(kwargs)
        if self.handler is not None:
            result["kwargs_handlers"] = [self.handler]
        return result

    def training_step(self, model, inputs, *args, **kwargs):
        return (model, inputs, args, kwargs)


def _checked_sharded_config():
    return {
        "distributed_backend": "sharded",
        "full_parameter_training": True,
        "optim": "adamw_torch",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 0.0,
    }


def test_checked_trainer_rejects_silent_ddp_fallback():
    checked = _build_checked_causal_lm_trainer(_FakeTrainer, _checked_sharded_config())
    with pytest.raises(ValueError, match="DDP fallback is forbidden"):
        checked(is_deepspeed_enabled=False)


def test_checked_sharded_trainer_does_not_mutate_ddp_handlers():
    handler = SimpleNamespace(
        gradient_as_bucket_view=False,
        broadcast_buffers=True,
        find_unused_parameters=True,
    )
    plugin = SimpleNamespace(sync_each_batch=False, num_steps=1)
    checked = _build_checked_causal_lm_trainer(_FakeTrainer, _checked_sharded_config())
    trainer = checked(
        is_deepspeed_enabled=True,
        handler=handler,
        gradient_accumulation_plugin=plugin,
    )
    assert handler.gradient_as_bucket_view is False
    assert handler.broadcast_buffers is True
    assert handler.find_unused_parameters is True
    assert plugin.sync_each_batch is False
    assert trainer.accelerator_args["kwargs_handlers"] == [handler]


def test_checked_sharded_training_step_requires_exact_live_deepspeed_engine(
    monkeypatch,
):
    calls = []

    def exact_assert(engine, training_cfg, world_size):
        calls.append((engine, training_cfg, world_size))
        return {"backend": "sharded", "zero_stage": 2}

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(
        "ir_training.train.sharded_contract.assert_sharded_engine", exact_assert
    )
    config = _checked_sharded_config()
    checked = _build_checked_causal_lm_trainer(_FakeTrainer, config)
    trainer = checked(is_deepspeed_enabled=True)
    engine = object()
    result = trainer.training_step(engine, {"input_ids": [1]})
    assert calls == [(engine, config, 4)]
    assert result[0] is engine


def _checkpoint_fixture(tmp_path: Path, monkeypatch):
    """Materialize the same tiny complete 541-state fixture as the core test."""
    from ir_training.qat import mobile_training_seed

    root = Path(__file__).resolve().parents[1]
    base = load_yaml(root / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml")
    config = full_contract.configure_full_qat(base, distributed_backend="sharded")
    config["runtime"]["gpu_profile"] = {"world_size": 2}
    monkeypatch.setattr(
        mobile_training_seed,
        "verify_configured_mobile_training_seed",
        lambda *args, **kwargs: {"verified": True},
    )
    matrices = {
        "model.embed_tokens.weight": [8, 4],
        "model.embed_tokens_per_layer.weight": [8, 4],
        "model.per_layer_model_projection.weight": [4, 4],
        "lm_head.weight": [8, 4],
    }
    shapes = (
        {f"param_{index}": [2] for index in range(501)}
        | {"norm": [2]}
        | matrices
        | {name: [1] for name in full_contract.EXPECTED_PERSISTENT_BUFFER_NAMES}
    )
    monkeypatch.setattr(full_contract, "seed_shapes", lambda cfg: shapes)
    tensors = {name: torch.ones(shape) for name, shape in shapes.items()}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    parameters = [
        {
            "canonical_name": name,
            "aliases": [name],
            "source_dtype": "bfloat16",
            "trainable_dtype": "float32",
            "numel": tensor.numel(),
        }
        for name, tensor in tensors.items()
        if name not in full_contract.EXPECTED_PERSISTENT_BUFFER_NAMES
    ]
    trainable_numel = sum(item["numel"] for item in parameters)
    scope = {
        "schema_version": 1,
        "scope": "all_model_parameters",
        "verified": True,
        "master_trainable_dtype": "float32",
        "unique_parameter_count": len(parameters),
        "named_parameter_count": len(parameters),
        "tied_alias_count": 0,
        "trainable_numel": trainable_numel,
        "frozen_parameter_count": 0,
        "adapter_parameter_count": 0,
        "parameters": parameters,
    }
    allocation = full_contract._seed_matrix_allocation(shapes)
    probe = _probe()
    probe["scope"].update(
        unique_parameter_count=len(parameters), trainable_numel=trainable_numel
    )
    probe["sharded_probe"]["gradient_coverage"].update(
        unique_parameters=len(parameters), global_numel=trainable_numel
    )

    # Reuse the established numeric-evidence constructor without importing its
    # whole test module as a package.
    fixture_path = root / "tests/test_full_model_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_full_model_contract_fixture", fixture_path
    )
    assert spec is not None and spec.loader is not None
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    metadata = {
        "checkpoint_kind": "full_model",
        "full_parameter_scope": scope,
        "trainable_parameter_names": [item["canonical_name"] for item in parameters],
        "trainable_parameter_counts": {
            "trainable": trainable_numel,
            "total": trainable_numel,
        },
        "qat": {
            "true_fake_quant": True,
            "spec": json.loads(json.dumps(QATSpec.from_config(config).to_dict())),
            "wrapped_effective_lora_count": 0,
            "retained_qparams_binding_count": 0,
            "wrapped_weight_bits_by_module": allocation,
        },
        "full_qat_coverage": {
            "verified": True,
            "matrices": allocation,
            "matrix_count": len(allocation),
        },
        "numeric_preflight": fixture_module.numeric_evidence(config),
        "backward_preflight": {"status": "passed"},
        "full_model_inventory": full_contract._model_inventory_evidence(
            shapes,
            {
                name: full_contract._buffer_value_sha256(tensors[name])
                for name in full_contract.EXPECTED_PERSISTENT_BUFFER_NAMES
            },
        ),
        "training_config_sha256": "a" * 64,
        "full_optimizer_preflight": {
            "training_config_sha256": "a" * 64,
            "rank": 0,
            "world_size": 2,
            "probe": probe,
        },
    }
    return config, json.loads(json.dumps(metadata))


def test_full_checkpoint_accepts_sharded_schema3_provenance(tmp_path, monkeypatch):
    config, metadata = _checkpoint_fixture(tmp_path, monkeypatch)
    report = full_contract.validate_full_qat_checkpoint(config, metadata, tmp_path)
    assert report["verified"] is True
    assert report["state_tensor_count"] == 541


@pytest.mark.parametrize("tamper", ["canonical_config", "ddp_receipt"])
def test_full_checkpoint_rejects_wrong_sharded_optimizer_provenance(
    tmp_path, monkeypatch, tamper
):
    config, metadata = _checkpoint_fixture(tmp_path, monkeypatch)
    probe = metadata["full_optimizer_preflight"]["probe"]
    if tamper == "canonical_config":
        probe["sharded_probe"]["config"]["zero_optimization"]["overlap_comm"] = True
        probe["sharded_probe"]["config_sha256"] = deepspeed_config_sha256(
            probe["sharded_probe"]["config"]
        )
    else:
        probe.clear()
        probe.update(
            schema_version=2,
            probe="disposable_full_parameter_trainer_v2",
            passed=True,
        )
    with pytest.raises(ValueError, match="preflight|sharded"):
        full_contract.validate_full_qat_checkpoint(config, metadata, tmp_path)
