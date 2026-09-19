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


def _fixture(tmp_path, monkeypatch, *, full_scope=False):
    mappings = []
    inventory = {}
    tensors = {}
    definitions = (
        ("model.language_model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj.weight", 0.25, 0.5),
        ("model.language_model.layers.0.per_layer_projection", "model.layers.0.per_layer_projection.weight", 1.0, 2.0),
        ("lm_head", "lm_head.weight", 0.0, 0.0),
    )
    if full_scope:
        families = (
            "self_attn.q_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "per_layer_input_gate", "per_layer_projection",
        )
        definitions = [
            (
                f"model.language_model.layers.{layer}.{family}",
                f"model.layers.{layer}.{family}.weight",
                0.25, 0.5,
            )
            for layer in range(35)
            for family in (*families, *(("self_attn.k_proj", "self_attn.v_proj") if layer < 15 else ()))
        ]
        definitions.append(("lm_head", "lm_head.weight", 0.0, 0.0))
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
    if full_scope:
        per_layer_embedding_source = "model.language_model.embed_tokens_per_layer.embedding_quantized"
        per_layer_embedding_scale = "model.language_model.embed_tokens_per_layer.embedding_scale"
        mappings.append({
            "source_key": per_layer_embedding_source,
            "output_key": "model.embed_tokens_per_layer.weight",
            "scale_key": per_layer_embedding_scale,
        })
        inventory["model.embed_tokens_per_layer.weight"] = {}
        tensors[per_layer_embedding_source] = np.zeros((1, 1), dtype=np.uint8)
        tensors[per_layer_embedding_scale] = np.ones((1, 1), dtype=np.float32)
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
    if not full_scope:
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


def test_rejects_missing_mapped_published_role(tmp_path, monkeypatch):
    source, manifest, contract, tensors = _fixture(tmp_path, monkeypatch)
    tensors.pop("model.language_model.layers.0.self_attn.q_proj.output_activation_scale")
    save_file(tensors, source)
    monkeypatch.setattr(
        published,
        "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    with pytest.raises(published.PublishedQParamsError, match="presence differs"):
        published.verify_published_activation_scales(source, manifest, contract)


@pytest.mark.parametrize("unmapped_modules", [1, 40])
def test_reports_pinned_unmapped_text_scales_without_rejecting_them(
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

    report = published.verify_published_activation_scales(source, manifest, contract)
    assert report["verified"] is True
    assert report["activation_validation_scope"] == "verified_seed_tensor_mappings"
    assert report["a8_scalar_count"] == 6
    assert report["source_text_a8_scalar_count"] == 6 + len(extra_names)
    assert report["unmapped_source_text_a8_scalar_count"] == len(extra_names)
    assert report["unmapped_source_text_a8_keys"] == sorted(extra_names)


def test_mapped_role_validation_does_not_depend_on_text_inventory_filter(tmp_path, monkeypatch):
    source, manifest, contract, _ = _fixture(tmp_path, monkeypatch)
    missing_name = "model.language_model.layers.0.self_attn.q_proj.output_activation_scale"
    original_filter = published._is_text_activation_scale_key
    # The text-prefix inventory is diagnostic only. All mapped keys still go
    # through the scalar presence/byte/type checks even if that filter omits one.
    monkeypatch.setattr(
        published,
        "_is_text_activation_scale_key",
        lambda key: original_filter(key) and key != missing_name,
    )

    report = published.verify_published_activation_scales(source, manifest, contract)
    assert report["verified"] is True
    assert report["a8_scalar_count"] == 6

    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["inventory"]["model.layers.0.self_attn.q_proj.weight"][
        "output_activation_scale_f32_le_hex"
    ] = _hex(0.75)
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(published.PublishedQParamsError, match="bytes differ"):
        published.verify_published_activation_scales(source, manifest, contract)


def _with_shared_kv_extras(tmp_path, monkeypatch):
    source, manifest, contract, tensors = _fixture(tmp_path, monkeypatch, full_scope=True)
    extra_names = [
        f"model.language_model.layers.{layer}.self_attn.{projection}.{role}_activation_scale"
        for layer in range(15, 35)
        for projection in ("k_proj", "v_proj")
        for role in ("input", "output")
    ]
    tensors.update({name: np.array(0.25, dtype=np.float32) for name in extra_names})
    save_file(tensors, source)
    monkeypatch.setattr(
        published, "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    return source, manifest, contract, tensors, extra_names


def test_accepts_80_shared_kv_extras_with_exact_production_mapped_counts(tmp_path, monkeypatch):
    source, manifest, contract, _, extras = _with_shared_kv_extras(tmp_path, monkeypatch)
    original_files = [path.read_bytes() for path in (source, manifest, contract)]

    report = published.verify_published_activation_scales(source, manifest, contract)

    assert report["verified"] is True
    assert report["scaled_weight_mapping_count"] == 278
    assert report["a8_weight_scope_counts"] == {"mutable": 205, "frozen": 70, "zero_head": 1}
    assert report["a8_scalar_count"] == 552
    assert report["source_text_a8_scalar_count"] == 632
    assert report["unmapped_source_text_a8_scalar_count"] == 80
    assert report["unmapped_source_text_a8_keys"] == sorted(extras)
    assert [path.read_bytes() for path in (source, manifest, contract)] == original_files


@pytest.mark.parametrize("role", ["input", "output"])
@pytest.mark.parametrize("change", ["missing_source", "changed_source", "missing_contract", "wrong_dtype"])
def test_shared_kv_extras_do_not_hide_invalid_mapped_scales(tmp_path, monkeypatch, role, change):
    source, manifest, contract, tensors, _ = _with_shared_kv_extras(tmp_path, monkeypatch)
    mapped_key = f"model.language_model.layers.14.self_attn.k_proj.{role}_activation_scale"
    error = "presence differs"
    if change == "missing_source":
        tensors.pop(mapped_key)
    elif change == "changed_source":
        tensors[mapped_key] = np.array(0.375, dtype=np.float32)
        error = "bytes differ"
    elif change == "wrong_dtype":
        tensors[mapped_key] = np.array(0.25, dtype=np.float16)
        error = "not scalar F32"
    else:
        payload = json.loads(contract.read_text(encoding="utf-8"))
        payload["inventory"]["model.layers.14.self_attn.k_proj.weight"].pop(
            f"{role}_activation_scale_f32_le_hex"
        )
        contract.write_text(json.dumps(payload), encoding="utf-8")
    # Bind the changed tiny fixture so the mapped-role checks are exercised;
    # production always retains the published SHA rather than accepting edits.
    save_file(tensors, source)
    monkeypatch.setattr(
        published, "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    with pytest.raises(published.PublishedQParamsError, match=error):
        published.verify_published_activation_scales(source, manifest, contract)


def test_shared_kv_extras_do_not_allow_removing_a_mapped_weight(tmp_path, monkeypatch):
    source, manifest, contract, _, _ = _with_shared_kv_extras(tmp_path, monkeypatch)
    removed = "model.layers.14.self_attn.k_proj.weight"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["transformation"]["tensor_mappings"] = [
        item for item in payload["transformation"]["tensor_mappings"] if item["output_key"] != removed
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["inventory"].pop(removed)
    contract.write_text(json.dumps(payload), encoding="utf-8")
    # Even with the fixture's provenance stubs, exact production scope counts
    # reject moving a required weight out of the mapping and into the extras.
    with pytest.raises(published.PublishedQParamsError, match="scope counts differ"):
        published.verify_published_activation_scales(source, manifest, contract)


def test_unmapped_extras_still_require_the_pinned_source_hash(tmp_path, monkeypatch):
    source, manifest, contract, tensors, extras = _with_shared_kv_extras(tmp_path, monkeypatch)
    tensors[extras[0]] = np.array(0.5, dtype=np.float32)
    save_file(tensors, source)
    with pytest.raises(published.PublishedQParamsError, match="SHA-256 differs"):
        published.verify_published_activation_scales(source, manifest, contract)


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
