"""Fast synthetic provenance checks for the explicit mobile SRQ contract."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gemma4_retained_scale_litertlm as exporter
from ir_training.export import merge_lora
from ir_training.export.merge_lora import (
    _retained_binding_contract_matches,
    _strict_mobile_srq_contract_required,
    _strict_mobile_srq_metadata_checks,
)
from ir_training.qat.mobile_qparams import canonical_activation_scale_schema_valid


def _hex(value: float) -> str:
    return struct.pack("<f", value).hex()


def _strict_fixture() -> tuple[dict, dict, dict]:
    mutable_keys = [
        f"model.layers.{index}.self_attn.q_proj.weight" for index in range(205)
    ]
    frozen_keys = [
        f"model.layers.{layer}.{suffix}.weight"
        for layer in range(35)
        for suffix in ("per_layer_input_gate", "per_layer_projection")
    ]
    inventory = {
        key: {
            "bits": 4,
            "group_size": None,
            "scale_shape": [1, 1],
            "input_activation_scale_f32_le_hex": _hex(0.125),
            "output_activation_scale_f32_le_hex": _hex(0.25),
        }
        for key in mutable_keys
    }
    inventory.update(
        {
            key: {
                "bits": 8,
                "group_size": None,
                "scale_shape": [1, 1],
                "input_activation_scale_f32_le_hex": _hex(0.5),
                "output_activation_scale_f32_le_hex": _hex(1.0),
            }
            for key in frozen_keys
        }
    )
    mutable_names = [f"module_{index}" for index in range(205)]
    mutable_bindings = {
        name: {
            "weight_key": key,
            "bits": 4,
            "group_size": None,
            "scale_shape": [1, 1],
            "input_activation_scale": 0.125,
            "output_activation_scale": 0.25,
        }
        for name, key in zip(mutable_names, mutable_keys, strict=True)
    }
    frozen_bindings = {
        f"frozen_{index}": {
            "weight_key": key,
            "bits": 8,
            "group_size": None,
            "scale_shape": [1, 1],
            "input_activation_scale": 0.5,
            "output_activation_scale": 1.0,
        }
        for index, key in enumerate(frozen_keys)
    }
    config_qat = {
        "scale_mode": "retained_mobile",
        "activation_quantizer": "gemma_mobile_srq",
        "simulate_frozen_activations": True,
        "expected_frozen_activation_modules": 70,
        "require_lora_trainable_scope": True,
    }
    qat = {
        "wrapped_effective_lora_count": 205,
        "wrapped_effective_lora_names": mutable_names,
        "retained_qparams_binding_count": 205,
        "retained_qparams_bindings": mutable_bindings,
        "frozen_activation_module_count": 70,
        "frozen_activation_bindings": frozen_bindings,
        "trainable_scope": {
            "verified": True,
            "expected_lora_tensors": 410,
            "trainable_lora_tensors": 410,
            "all_adapters_trainable": True,
            "all_parameters_frozen": False,
        },
        "native_kv_cache_simulated": False,
        "spec": dict(config_qat),
    }
    return config_qat, qat, {"verified": True, "inventory": inventory}


def test_canonical_contract_allows_zero_only_for_lm_head():
    zero = {
        "input_activation_scale_f32_le_hex": _hex(0.0),
        "output_activation_scale_f32_le_hex": _hex(0.0),
    }
    assert canonical_activation_scale_schema_valid("lm_head.weight", zero)
    assert not canonical_activation_scale_schema_valid(
        "model.layers.0.self_attn.q_proj.weight", zero
    )
    assert canonical_activation_scale_schema_valid(
        "model.layers.0.self_attn.q_proj.weight",
        {
            "input_activation_scale_f32_le_hex": _hex(0.125),
            "output_activation_scale_f32_le_hex": _hex(0.25),
        },
    )


def test_retained_bindings_require_exact_decoded_source_f32_values():
    _, qat, qparams = _strict_fixture()
    assert _retained_binding_contract_matches(qat, qparams)

    qat["retained_qparams_bindings"]["module_17"][
        "output_activation_scale"
    ] = 0.2501
    assert not _retained_binding_contract_matches(qat, qparams)


@pytest.mark.parametrize(
    ("tamper", "failed_check"),
    [
        ("spec", "mobile_srq_spec_matches_resolved_config"),
        ("frozen_binding", "exact_70_frozen_activation_bindings"),
        ("trainable_scope", "trainable_lora_ab_scope_bound"),
        ("cache_claim", "native_kv_cache_simulation_not_claimed"),
    ],
)
def test_strict_metadata_rejects_tampered_new_contract(tamper, failed_check):
    config_qat, qat, qparams = _strict_fixture()
    if tamper == "spec":
        qat["spec"]["activation_quantizer"] = "legacy"
    elif tamper == "frozen_binding":
        qat["frozen_activation_bindings"]["frozen_3"][
            "input_activation_scale"
        ] = 0.75
    elif tamper == "trainable_scope":
        qat["trainable_scope"]["trainable_lora_tensors"] = 409
    else:
        qat["native_kv_cache_simulated"] = True

    checks = _strict_mobile_srq_metadata_checks(config_qat, qat, qparams)
    assert checks[failed_check] is False


def test_legacy_retained_config_does_not_require_new_metadata_fields():
    legacy = {"scale_mode": "retained_mobile"}
    assert not _strict_mobile_srq_contract_required(legacy)
    assert _strict_mobile_srq_metadata_checks(legacy, {}, {}) == {}


class _FixtureQParams:
    contract_sha256 = "a" * 64
    scale_storage_sha256 = "b" * 64

    def __init__(self, inventory):
        self.inventory = inventory

    def trainable_projection_weight_keys(self):
        return tuple(
            sorted(
                key
                for key in self.inventory
                if key.endswith("self_attn.q_proj.weight")
            )
        )


def _export_fixture(tmp_path: Path, *, strict: bool) -> tuple[Path, Path, object, dict]:
    config_qat, qat, qparams = _strict_fixture()
    qat["spec"].update(
        {
            "fixed_scale_required": True,
            "fixed_activation_scale_required": True,
            "effective_lora_only": True,
            "effective_merged_weight": True,
            "ste_gradient": "clipped",
            "quantize_embeddings": False,
            "expected_effective_lora_modules": 205,
        }
    )
    qat["retained_qparams"] = {
        "verified": True,
        "mode": "retained_mobile",
        "contract_sha256": "a" * 64,
        "scale_storage_sha256": "b" * 64,
        "inventory_sha256": "e" * 64,
        "tensor_count": 278,
    }
    qat.update(
        {
            "enabled": True,
            "effective_merged_weight_qat_enabled": True,
            "uncovered_lora_adapter_linear_names": [],
        }
    )
    if not strict:
        config_qat = {}
        for field in (
            "activation_quantizer",
            "simulate_frozen_activations",
            "expected_frozen_activation_modules",
            "require_lora_trainable_scope",
        ):
            qat["spec"].pop(field, None)
        for field in (
            "frozen_activation_module_count",
            "frozen_activation_bindings",
            "trainable_scope",
            "native_kv_cache_simulated",
        ):
            qat.pop(field, None)

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(
        "qat:\n" + "".join(f"  {key}: {json.dumps(value)}\n" for key, value in config_qat.items()),
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_file.write_bytes(b"adapter")
    adapter_record = {
        "path": adapter_file.name,
        "size": adapter_file.stat().st_size,
        "sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
    }
    metadata = {
        "training_metadata_version": 4,
        "checkpoint_role": "best_golden",
        "checkpoint_step": 10,
        "training": {"method": "qat_lora_sft"},
        "lora": {"dropout": 0.0},
        "git_commit": "f" * 40,
        "training_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "adapter_checkpoints": [{"role": "best_golden", "files": [adapter_record]}],
        "numeric_preflight": {
            "passed": True,
            "qat_enabled": True,
            "top1_probe_match_fraction": 0.95,
            "zero_adapter_initialization": {
                "verified_zero_delta": True,
                "wrapper_count": 205,
                "adapter_pair_count": 205,
                "nonzero_or_invalid_pairs": [],
            },
            "greedy_generation": {
                "passed": True,
                "qat_enabled": True,
                "qat_greedy_deterministic": True,
                "baseline_qat_min_common_prefix_tokens": 8,
            },
        },
        "qat": qat,
        "mobile_training_seed": {
            "verified": True,
            "manifest_sha256": "c" * 64,
            "transformation_plan_sha256": "d" * 64,
        },
    }
    metadata_path = adapter / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return adapter, config_path, _FixtureQParams(qparams["inventory"]), metadata


def test_export_provenance_conditionally_enforces_new_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(
        exporter,
        "_golden_selection_binding",
        lambda *_args, **_kwargs: {"verified": True},
    )
    seed = {
        "manifest_sha256": "c" * 64,
        "transformation_plan_sha256": "d" * 64,
    }

    strict_dir = tmp_path / "strict"
    strict_dir.mkdir()
    adapter, config, qparams, metadata = _export_fixture(strict_dir, strict=True)
    report = exporter._best_adapter_provenance_report(
        adapter, config, qparams=qparams, seed_report=seed
    )
    assert report["verified"] is True
    assert report["checks"]["exact_70_frozen_activation_bindings"] is True

    metadata["qat"]["frozen_activation_bindings"]["frozen_0"][
        "output_activation_scale"
    ] = 2.0
    (adapter / "training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    tampered = exporter._best_adapter_provenance_report(
        adapter, config, qparams=qparams, seed_report=seed
    )
    assert tampered["verified"] is False
    assert tampered["checks"]["exact_70_frozen_activation_bindings"] is False

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    adapter, config, qparams, _ = _export_fixture(legacy_dir, strict=False)
    legacy = exporter._best_adapter_provenance_report(
        adapter, config, qparams=qparams, seed_report=seed
    )
    assert legacy["verified"] is True
    assert "exact_70_frozen_activation_bindings" not in legacy["checks"]


def test_merge_provenance_conditionally_enforces_new_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(
        merge_lora,
        "_golden_selection_binding",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        merge_lora,
        "_portable_launcher_contract_matches",
        lambda *_args, **_kwargs: True,
    )
    adapter, config_path, fixture_qparams, metadata = _export_fixture(
        tmp_path, strict=True
    )
    config_qat, _, _ = _strict_fixture()
    config = {
        "qat": config_qat,
        "preflight": {
            "min_top1_probe_match": 0.90,
            "min_baseline_qat_greedy_prefix_tokens": 8,
        },
    }
    qparams = {
        "verified": True,
        "contract_sha256": "a" * 64,
        "scale_storage_sha256": "b" * 64,
        "inventory_sha256": "e" * 64,
        "tensor_count": 278,
        "inventory": fixture_qparams.inventory,
    }
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

    report = merge_lora._verify_qat_training_metadata(
        adapter,
        training_config_sha256=config_sha,
        training_method="qat_lora_sft",
        mobile_training_seed={"required": False},
        training_config=config,
        mobile_qparams=qparams,
    )
    assert report["verified"] is True
    assert report["checks"]["mobile_srq_spec_matches_resolved_config"] is True

    metadata["qat"]["trainable_scope"]["all_adapters_trainable"] = False
    (adapter / "training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    tampered = merge_lora._verify_qat_training_metadata(
        adapter,
        training_config_sha256=config_sha,
        training_method="qat_lora_sft",
        mobile_training_seed={"required": False},
        training_config=config,
        mobile_qparams=qparams,
    )
    assert tampered["verified"] is False
    assert tampered["checks"]["trainable_lora_ab_scope_bound"] is False
