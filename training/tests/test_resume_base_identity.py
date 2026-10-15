"""Small byte fixtures exercise base identity without loading a real model."""
from __future__ import annotations

import json
import shutil

import pytest
from ir_training.train.resume_contract import (
    build_resume_contract,
    file_sha256,
    local_base_identity,
    verify_resume_contract,
)


def base_config(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"gemma3"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"base weights")
    return {"model": {"model_id": "google/gemma-3-270m-it", "model_source": str(root)}}


def test_local_base_contract_binds_weight_and_config_bytes(tmp_path):
    config = base_config(tmp_path)
    before = build_resume_contract(config, tmp_path, effective_batch=16)
    root = tmp_path / "base"
    assert before["version"] == 2
    (root / "model.safetensors").write_bytes(b"different weights")
    assert build_resume_contract(config, tmp_path, effective_batch=16) != before
    (root / "model.safetensors").write_bytes(b"base weights")
    (root / "config.json").write_text('{"model_type":"gemma3","vocab_size":32}')
    assert build_resume_contract(config, tmp_path, effective_batch=16) != before


def test_same_hub_label_does_not_hide_different_local_base(tmp_path):
    config = base_config(tmp_path)
    before = build_resume_contract(config, tmp_path, effective_batch=16)
    shutil.copytree(tmp_path / "base", tmp_path / "other")
    config["model"]["model_source"] = str(tmp_path / "other")
    assert build_resume_contract(config, tmp_path, effective_batch=16) == before
    (tmp_path / "other/model.safetensors").write_bytes(b"another model")
    assert build_resume_contract(config, tmp_path, effective_batch=16) != before


def test_sharded_base_requires_every_indexed_shard(tmp_path):
    config = base_config(tmp_path)
    root = tmp_path / "base"
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-00001-of-00001.safetensors"}
    }))
    with pytest.raises(ValueError, match="weight shard"):
        local_base_identity(config)
    (root / "model-00001-of-00001.safetensors").write_bytes(b"shard")
    identity = local_base_identity(config)
    assert "model-00001-of-00001.safetensors" in identity["files"]


def test_remote_and_separately_verified_mobile_contracts_unchanged(tmp_path):
    for model in (
        {"model_id": "google/gemma-3-270m-it"},
        {"model_source": "/unavailable/seed", "mobile_training_seed_manifest": "seed.json"},
    ):
        contract = build_resume_contract({"model": model}, tmp_path, effective_batch=16)
        assert contract["version"] == 1
        assert "local_base_identity" not in contract


@pytest.mark.parametrize("checkpoint_kind", ["sft", "grpo"])
def test_merge_rejects_mutated_bound_local_base(tmp_path, checkpoint_kind):
    from ir_training.export.merge_lora import _training_provenance

    config = base_config(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    metadata = (
        {"resume_contract": build_resume_contract(config, tmp_path, effective_batch=16)}
        if checkpoint_kind == "sft" else {"local_base_identity": local_base_identity(config)}
    )
    (adapter / "training_metadata.json").write_text(json.dumps(metadata))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "base/model.safetensors").write_bytes(b"mutated base")
    with pytest.raises(ValueError, match="base bytes differ"):
        _training_provenance(adapter, config_path, base=tmp_path)


def test_resume_rejects_changed_base_and_legacy_unbound_contract(tmp_path):
    config = base_config(tmp_path)
    expected = build_resume_contract(config, tmp_path, effective_batch=16)
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"test fixture")
    (checkpoint / "trainer_state.json").write_text('{"global_step":1}')
    metadata = {
        "resume_contract": expected,
        "adapter_checkpoints": [{"files": [{
            "path": "adapter_model.safetensors",
            "sha256": file_sha256(checkpoint / "adapter_model.safetensors"),
        }]}],
    }
    path = checkpoint / "training_metadata.json"
    path.write_text(json.dumps(metadata))
    assert verify_resume_contract(checkpoint, expected)["verified"]
    (tmp_path / "base/model.safetensors").write_bytes(b"changed")
    changed = build_resume_contract(config, tmp_path, effective_batch=16)
    with pytest.raises(ValueError, match="local_base_identity"):
        verify_resume_contract(checkpoint, changed)
    metadata["resume_contract"].pop("local_base_identity")
    metadata["resume_contract"]["version"] = 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="local_base_identity"):
        verify_resume_contract(checkpoint, changed)
