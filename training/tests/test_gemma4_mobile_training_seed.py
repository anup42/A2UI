from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_checkpoint_official_topology as topology
import pytest
import reconstruct_gemma4_mobile_training_seed as reconstruction
from ir_training.common.config import load_yaml
from ir_training.models import base as model_base
from ir_training.models.registry import create_adapter
from ir_training.qat import mobile_training_seed as seed_verifier
from ir_training.train.sft import train_sft


def _bf16_bytes_to_float32(raw: bytes) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2").astype("<u4") << np.uint32(16)
    return words.view("<f4")


def test_mobile_seed_dequantization_applies_low_bit_offsets_and_grouped_scales():
    # W2 codes 0,1,2,3 represent signed values -2,-1,0,1 in low-bit order.
    w2 = reconstruction._dequantize_chunk_to_bf16(
        bytes([0b11100100]),
        np.asarray([[0.5]], dtype="<f4").tobytes(),
        rows=1,
        source_columns=1,
        logical_columns=4,
        scale_columns=1,
        bits=2,
    )
    np.testing.assert_array_equal(
        _bf16_bytes_to_float32(w2),
        np.asarray([-1.0, -0.5, 0.0, 0.5], dtype=np.float32),
    )

    # Each W4 scale column covers a separate two-value logical group.
    w4 = reconstruction._dequantize_chunk_to_bf16(
        bytes([0x10, 0x32]),
        np.asarray([[1.0, 10.0]], dtype="<f4").tobytes(),
        rows=1,
        source_columns=2,
        logical_columns=4,
        scale_columns=2,
        bits=4,
    )
    np.testing.assert_array_equal(
        _bf16_bytes_to_float32(w4),
        np.asarray([-8.0, -7.0, -60.0, -50.0], dtype=np.float32),
    )


def test_mobile_seed_transform_rules_omit_shared_kv_duplicates():
    assert reconstruction._weight_bits("lm_head.weight", "U8") == 2
    assert (
        reconstruction._weight_bits(
            "model.language_model.layers.14.mlp.up_proj.weight", "U8"
        )
        == 4
    )
    assert (
        reconstruction._weight_bits(
            "model.language_model.layers.15.mlp.up_proj.weight", "U8"
        )
        == 2
    )
    assert reconstruction._output_key(
        "model.language_model.layers.15.self_attn.k_proj.weight"
    ) is None
    assert reconstruction._output_key(
        "model.language_model.layers.14.self_attn.k_proj.weight"
    ) == "model.layers.14.self_attn.k_proj.weight"


def test_mobile_seed_streaming_writer_emits_valid_atomic_safetensors(tmp_path):
    source = tmp_path / "packed.bin"
    direct_bf16 = np.asarray([0x3F80], dtype="<u2").tobytes()
    packed_w2 = bytes([0b11100100])
    scale = np.asarray([[0.5]], dtype="<f4").tobytes()
    source.write_bytes(direct_bf16 + packed_w2 + scale)
    direct = reconstruction.TensorTransform(
        output_key="model.direct.weight",
        output_shape=(1,),
        output_dtype="BF16",
        source_key="source.direct",
        source_dtype="BF16",
        source_shape=(1,),
        source_begin=0,
        source_end=2,
        transform="copy_bf16",
    )
    quantized = reconstruction.TensorTransform(
        output_key="model.quantized.weight",
        output_shape=(1, 4),
        output_dtype="BF16",
        source_key="source.quantized",
        source_dtype="U8",
        source_shape=(1, 1),
        source_begin=2,
        source_end=3,
        transform="dequantize_w2_to_bf16_rne",
        bits=2,
        scale_key="source.quantized_scale",
        scale_shape=(1, 1),
        scale_begin=3,
        scale_end=7,
        scale_group_width=4,
    )
    destination = tmp_path / "model.safetensors"

    tensor_hashes, record = reconstruction._write_shard(
        source,
        destination,
        reconstruction.ShardPlan(destination.name, (direct, quantized)),
        working_set_bytes=1024,
    )

    assert destination.is_file()
    assert not destination.with_name(destination.name + ".partial").exists()
    assert record["sha256"] == _sha256(destination)
    assert set(tensor_hashes) == {
        "model.direct.weight",
        "model.quantized.weight",
    }
    _header_size, data_start, header = reconstruction._read_safetensors_header(
        destination
    )
    with destination.open("rb") as handle:
        _, _, direct_begin, direct_end = reconstruction._entry(
            header,
            "model.direct.weight",
            data_start=data_start,
            file_size=destination.stat().st_size,
        )
        _, _, quantized_begin, quantized_end = reconstruction._entry(
            header,
            "model.quantized.weight",
            data_start=data_start,
            file_size=destination.stat().st_size,
        )
        assert reconstruction._read_exact(
            handle, direct_begin, direct_end - direct_begin, label="direct"
        ) == direct_bf16
        quantized_bytes = reconstruction._read_exact(
            handle,
            quantized_begin,
            quantized_end - quantized_begin,
            label="quantized",
        )
    np.testing.assert_array_equal(
        _bf16_bytes_to_float32(quantized_bytes),
        np.asarray([-1.0, -0.5, 0.0, 0.5], dtype=np.float32),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_small_verified_manifest(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(seed_verifier, "EXPECTED_TENSOR_COUNT", 2)
    monkeypatch.setattr(seed_verifier, "EXPECTED_DIRECT_BF16_COUNT", 1)
    monkeypatch.setattr(seed_verifier, "EXPECTED_DEQUANTIZED_COUNT", 1)
    monkeypatch.setattr(seed_verifier, "EXPECTED_BIT_HISTOGRAM", {"W2": 1})

    mappings = [
        {
            "output_key": "model.embed_tokens.weight",
            "output_shape": [1],
            "output_dtype": "BF16",
            "source_key": "packed.embed",
            "source_dtype": "BF16",
            "source_shape": [1],
            "source_begin": 0,
            "source_end": 2,
            "transform": "copy_bf16",
            "bits": None,
            "scale_key": None,
            "scale_shape": None,
            "scale_begin": None,
            "scale_end": None,
            "scale_group_width": None,
            "output_nbytes": 2,
        },
        {
            "output_key": "lm_head.weight",
            "output_shape": [1],
            "output_dtype": "BF16",
            "source_key": "lm_head.weight",
            "source_dtype": "U8",
            "source_shape": [1],
            "source_begin": 2,
            "source_end": 3,
            "transform": "dequantize_w2_to_bf16_rne",
            "bits": 2,
            "scale_key": "lm_head.weight_scale",
            "scale_shape": [1, 1],
            "scale_begin": 3,
            "scale_end": 7,
            "scale_group_width": 1,
            "output_nbytes": 2,
        },
    ]
    plan_hash = seed_verifier._canonical_plan_sha256(mappings)
    assert plan_hash
    monkeypatch.setattr(
        seed_verifier, "EXPECTED_TRANSFORMATION_PLAN_SHA256", plan_hash
    )

    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"header-and-payload")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "architectures": ["Gemma4ForCausalLM"],
                "tie_word_embeddings": False,
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "manifest_version": 1,
        "seed_format": "gemma4_e2b_mobile_dequantized_bf16_text",
        "seed_id": "test-seed",
        "ready": True,
        "executed": True,
        "training_executed": False,
        "private_google_recipe_recovered": False,
        "source": {
            "model_id": seed_verifier.OFFICIAL_MOBILE_MODEL_ID,
            "revision": seed_verifier.OFFICIAL_MOBILE_REVISION,
            "safetensors_sha256_expected": seed_verifier.OFFICIAL_MOBILE_SAFETENSORS_SHA256,
            "safetensors_sha256_observed": seed_verifier.OFFICIAL_MOBILE_SAFETENSORS_SHA256,
            "config_sha256_expected": seed_verifier.OFFICIAL_MOBILE_CONFIG_SHA256,
            "config_sha256": seed_verifier.OFFICIAL_MOBILE_CONFIG_SHA256,
        },
        "official_graph_authority": {
            "artifact_sha256": seed_verifier.OFFICIAL_LITERTLM_SHA256,
            "retained_compiled_parity": {
                "verified": True,
                "sha256": seed_verifier.OFFICIAL_RETAINED_COMPILED_REPORT_SHA256,
                "checks": {"compiled_mapping": True},
            },
        },
        "output": {
            "directory": str(tmp_path),
            "tensor_count": 2,
            "direct_bf16_tensor_count": 1,
            "dequantized_tensor_count": 1,
            "payload_size_bytes": 4,
            "shard_count": 1,
            "materialized": True,
            "shards": [
                {
                    "path": shard.name,
                    "size_bytes": shard.stat().st_size,
                    "payload_bytes": 4,
                    "tensor_count": 2,
                    "sha256": _sha256(shard),
                }
            ],
            "weight_map": {
                "model.embed_tokens.weight": shard.name,
                "lm_head.weight": shard.name,
            },
            "tensor_sha256": {
                "model.embed_tokens.weight": "a" * 64,
                "lm_head.weight": "b" * 64,
            },
            "auxiliary_files": [
                {
                    "path": config.name,
                    "size_bytes": config.stat().st_size,
                    "sha256": _sha256(config),
                }
            ],
        },
        "transformation": {
            "plan_sha256": plan_hash,
            "tensor_count": 2,
            "bit_histogram": {"W2": 1},
            "tensor_mappings": mappings,
        },
    }
    manifest_path = tmp_path / "mobile_training_seed_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_mobile_seed_manifest_verifier_binds_files_and_model_source(
    tmp_path, monkeypatch
):
    manifest = _write_small_verified_manifest(tmp_path, monkeypatch)

    report = seed_verifier.verify_mobile_training_seed_manifest(
        manifest,
        expected_model_source=tmp_path,
    )

    assert report["verified"] is True
    assert all(report["checks"].values())
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    rejected = seed_verifier.verify_mobile_training_seed_manifest(
        manifest,
        expected_model_source=tmp_path,
    )
    assert rejected["verified"] is False
    assert rejected["checks"]["shard_hashes_match"] is False


def test_model_adapter_keeps_canonical_identity_but_loads_local_source(
    tmp_path, monkeypatch
):
    calls: list[str] = []
    monkeypatch.setattr(
        model_base,
        "load_hf_model",
        lambda source, _config: calls.append(source) or object(),
    )
    adapter = create_adapter(
        {
            "family": "gemma",
            "model_id": seed_verifier.OFFICIAL_MOBILE_MODEL_ID,
            "model_source": str(tmp_path),
        }
    )

    adapter.load_model()

    assert adapter.model_id == seed_verifier.OFFICIAL_MOBILE_MODEL_ID
    assert calls == [str(tmp_path.resolve())]


def test_mobile_training_config_fails_before_model_load_when_seed_is_missing():
    config = load_yaml(
        ROOT / "configs" / "models" / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
    )

    with pytest.raises(RuntimeError, match="mobile training seed"):
        train_sft(config)


def test_compiler_merge_provenance_binds_mobile_seed_hash(tmp_path):
    training_config = (
        ROOT / "configs" / "models" / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
    )
    merged_model = tmp_path / "model.safetensors"
    merged_model.write_bytes(b"merged-model")
    seed_report = {
        "required": True,
        "verified": True,
        "manifest_sha256": "c" * 64,
        "transformation_plan_sha256": "d" * 64,
        "output": {"directory": str((tmp_path / "seed").resolve())},
    }
    metadata = {
        "manifest_version": 4,
        "base_model_id": seed_verifier.OFFICIAL_MOBILE_MODEL_ID,
        "base_model_source": seed_report["output"]["directory"],
        "mobile_training_seed": dict(seed_report),
        "exact_checkpoint_keys_required": True,
        "training_config_sha256": hashlib.sha256(
            training_config.read_bytes()
        ).hexdigest(),
        "training_method": "qat_lora_sft",
        "qat_enabled": True,
        "qat_effective_merged_weight": True,
        "lora_dropout": 0.0,
        "training_run_metadata": {"verified": True},
        "continued_qat_performed": True,
        "merge_performed_qat": False,
        "adapter_files": [
            {
                "path": "adapter_model.safetensors",
                "size": 1,
                "sha256": "a" * 64,
            }
        ],
        "merged_model_files": [
            {
                "path": merged_model.name,
                "size": merged_model.stat().st_size,
                "sha256": _sha256(merged_model),
            }
        ],
        "requires_post_merge_quantization": True,
        "packed_int4_output": False,
        "mtp_assistant_trained_or_modified": False,
    }
    metadata_path = tmp_path / "qat_mtp_merge_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = topology._merge_provenance_report(
        tmp_path,
        training_config,
        official_base_model_id=seed_verifier.OFFICIAL_MOBILE_MODEL_ID,
        mobile_training_seed=seed_report,
    )

    assert report["verified"] is True
    metadata["mobile_training_seed"]["manifest_sha256"] = "e" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    rejected = topology._merge_provenance_report(
        tmp_path,
        training_config,
        official_base_model_id=seed_verifier.OFFICIAL_MOBILE_MODEL_ID,
        mobile_training_seed=seed_report,
    )
    assert rejected["verified"] is False
    assert rejected["checks"]["mobile_training_seed_matches"] is False
