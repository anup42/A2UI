from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.qat import published_qparams as published


def _hex(value: float) -> str:
    return struct.pack("<f", value).hex()


def _fixture(tmp_path, monkeypatch):
    mappings = []
    inventory = {}
    tensors = {}
    definitions = (
        ("model.language_model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj.weight", 0.25, 0.5),
        ("model.language_model.layers.0.per_layer_projection", "model.layers.0.per_layer_projection.weight", 1.0, 2.0),
        ("lm_head", "lm_head.weight", 0.0, 0.0),
    )
    for stem, output_key, input_scale, output_scale in definitions:
        source_key = f"{stem}.weight"
        scale_key = f"{stem}.weight_scale"
        mappings.append(
            {
                "source_key": source_key,
                "output_key": output_key,
                "scale_key": scale_key,
            }
        )
        inventory[output_key] = {
            "input_activation_scale_f32_le_hex": _hex(input_scale),
            "output_activation_scale_f32_le_hex": _hex(output_scale),
        }
        tensors[source_key] = np.zeros((1,), dtype=np.uint8)
        tensors[scale_key] = np.ones((1,), dtype=np.float32)
        tensors[f"{stem}.input_activation_scale"] = np.array(input_scale, dtype=np.float32)
        tensors[f"{stem}.output_activation_scale"] = np.array(output_scale, dtype=np.float32)
    embedding_source = "model.language_model.embed_tokens.embedding_quantized"
    embedding_output = "model.embed_tokens.weight"
    embedding_scale = "model.language_model.embed_tokens.embedding_scale"
    mappings.append(
        {
            "source_key": embedding_source,
            "output_key": embedding_output,
            "scale_key": embedding_scale,
        }
    )
    inventory[embedding_output] = {}
    tensors[embedding_source] = np.zeros((1, 1), dtype=np.uint8)
    tensors[embedding_scale] = np.ones((1, 1), dtype=np.float32)
    source = tmp_path / "model.safetensors"
    save_file(tensors, source)
    manifest = tmp_path / "mobile_training_seed_manifest.json"
    manifest.write_text(
        json.dumps({"transformation": {"tensor_mappings": mappings}}), encoding="utf-8"
    )
    contract = tmp_path / "mobile_qparams.json"
    contract.write_text(json.dumps({"inventory": inventory}), encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    monkeypatch.setattr(published, "OFFICIAL_MOBILE_SAFETENSORS_SHA256", source_sha)
    monkeypatch.setattr(published, "EXPECTED_MUTABLE_A8_WEIGHTS", 1)
    monkeypatch.setattr(published, "EXPECTED_FROZEN_A8_WEIGHTS", 1)
    monkeypatch.setattr(published, "EXPECTED_ZERO_A8_WEIGHTS", 1)
    monkeypatch.setattr(
        published,
        "verify_mobile_training_seed_manifest",
        lambda *_args, **_kwargs: {"verified": True},
    )

    def verify_contract(path, **_kwargs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "verified": True,
            "contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "inventory": payload["inventory"],
        }

    monkeypatch.setattr(published, "verify_mobile_qparams_contract", verify_contract)
    return source, manifest, contract, tensors


def test_verifies_all_mapped_published_activation_scale_bytes(tmp_path, monkeypatch):
    source, manifest, contract, _ = _fixture(tmp_path, monkeypatch)

    report = published.verify_published_activation_scales(source, manifest, contract)

    assert report["verified"] is True
    assert report["source_hash_verification"] == "computed"
    assert report["a8_weight_scope_counts"] == {
        "mutable": 1,
        "frozen": 1,
        "zero_head": 1,
    }
    assert report["a8_scalar_count"] == 6
    assert report["scaled_weight_mapping_count"] == 4
    assert report["roles"]["input"]["count"] == 3
    assert "f32_le_hex" not in report["roles"]["input"]


def test_rejects_wrong_source_hash(tmp_path, monkeypatch):
    source, manifest, contract, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(published, "OFFICIAL_MOBILE_SAFETENSORS_SHA256", "f" * 64)

    with pytest.raises(published.PublishedQParamsError, match="SHA-256 differs"):
        published.verify_published_activation_scales(source, manifest, contract)


def test_rejects_positive_scale_contract_tamper(tmp_path, monkeypatch):
    source, manifest, contract, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["inventory"]["model.layers.0.self_attn.q_proj.weight"][
        "input_activation_scale_f32_le_hex"
    ] = _hex(0.375)
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(published.PublishedQParamsError, match="bytes differ"):
        published.verify_published_activation_scales(source, manifest, contract)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_rejects_missing_or_extra_published_role(tmp_path, monkeypatch, change):
    source, manifest, contract, tensors = _fixture(tmp_path, monkeypatch)
    if change == "missing":
        tensors.pop("model.language_model.layers.0.self_attn.q_proj.output_activation_scale")
    else:
        tensors["model.language_model.unmapped.input_activation_scale"] = np.array(
            1.0, dtype=np.float32
        )
    save_file(tensors, source)
    monkeypatch.setattr(
        published,
        "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    with pytest.raises(published.PublishedQParamsError, match="presence differs|missing or extra"):
        published.verify_published_activation_scales(source, manifest, contract)


@pytest.mark.parametrize("unmapped_modules", [1, 40])
def test_role_mismatch_reports_counts_and_every_sorted_extra_name(
    tmp_path, monkeypatch, unmapped_modules
):
    source, manifest, contract, tensors = _fixture(tmp_path, monkeypatch)
    extra_names = [
        f"model.language_model.unmapped.{index}.{role}_activation_scale"
        for index in reversed(range(unmapped_modules))
        for role in ("output", "input")
    ]
    tensors.update({name: np.array(1.0, dtype=np.float32) for name in extra_names})
    save_file(tensors, source)
    monkeypatch.setattr(
        published,
        "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    with pytest.raises(published.PublishedQParamsError) as failure:
        published.verify_published_activation_scales(source, manifest, contract)

    assert str(failure.value) == (
        "Packed source contains missing or extra activation-scale tensors "
        "relative to the mapped qparams contract. "
        f"actual_source_roles_count={6 + len(extra_names)}, expected_source_roles_count=6; "
        "actual_source_roles - expected_source_roles "
        f"(extra_count={len(extra_names)})={json.dumps(sorted(extra_names))}; "
        "expected_source_roles - actual_source_roles (missing_count=0)=[]"
    )


def test_role_mismatch_reports_expected_names_excluded_by_source_filter(tmp_path, monkeypatch):
    source, manifest, contract, _ = _fixture(tmp_path, monkeypatch)
    missing_name = "model.language_model.layers.0.self_attn.q_proj.output_activation_scale"
    original_filter = published._is_text_activation_scale_key
    # Simulate a source-role filter omitting a mapped tensor: the diagnostic
    # must distinguish this direction from extra source tensors. Production
    # filtering and all provenance/scale checks stay unchanged.
    monkeypatch.setattr(
        published,
        "_is_text_activation_scale_key",
        lambda key: original_filter(key) and key != missing_name,
    )

    with pytest.raises(published.PublishedQParamsError) as failure:
        published.verify_published_activation_scales(source, manifest, contract)

    assert str(failure.value) == (
        "Packed source contains missing or extra activation-scale tensors "
        "relative to the mapped qparams contract. "
        "actual_source_roles_count=5, expected_source_roles_count=6; "
        "actual_source_roles - expected_source_roles (extra_count=0)=[]; "
        "expected_source_roles - actual_source_roles "
        f"(missing_count=1)={json.dumps([missing_name])}"
    )


@pytest.mark.parametrize(
    ("weight_key", "role", "value"),
    [
        ("model.layers.0.self_attn.q_proj.weight", "input", 0.0),
        ("lm_head.weight", "output", 0.25),
    ],
)
def test_rejects_zero_outside_head_or_nonzero_head(
    tmp_path, monkeypatch, weight_key, role, value
):
    source, manifest, contract, tensors = _fixture(tmp_path, monkeypatch)
    mapping = next(
        item
        for item in json.loads(manifest.read_text(encoding="utf-8"))["transformation"][
            "tensor_mappings"
        ]
        if item["output_key"] == weight_key
    )
    source_role = mapping["source_key"].removesuffix(".weight") + f".{role}_activation_scale"
    tensors[source_role] = np.array(value, dtype=np.float32)
    save_file(tensors, source)
    monkeypatch.setattr(
        published,
        "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["inventory"][weight_key][f"{role}_activation_scale_f32_le_hex"] = _hex(value)
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(published.PublishedQParamsError, match="Invalid published"):
        published.verify_published_activation_scales(source, manifest, contract)


def test_accepts_exact_caller_verified_pinned_hash(tmp_path, monkeypatch):
    source, manifest, contract, _ = _fixture(tmp_path, monkeypatch)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    report = published.verify_published_activation_scales(
        source, manifest, contract, verified_source_sha256=digest
    )

    assert report["source_hash_verification"] == "caller_verified"


def test_ignores_pinned_non_text_activation_scales(tmp_path, monkeypatch):
    source, manifest, contract, tensors = _fixture(tmp_path, monkeypatch)
    tensors["model.vision_tower.block.input_activation_scale"] = np.array(
        3.0, dtype=np.float32
    )
    save_file(tensors, source)
    monkeypatch.setattr(
        published,
        "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    report = published.verify_published_activation_scales(source, manifest, contract)

    assert report["verified"] is True
