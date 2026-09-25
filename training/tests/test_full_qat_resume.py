"""CPU-only full-QAT continuation and provenance regression tests."""
from __future__ import annotations

import copy
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.pipeline.full_parameter_qat import (
    FullParameterQATOptions,
    build_plan,
    training_config,
)
from ir_training.train.callbacks import _checkpoint_adapter_manifest
from ir_training.train.full_qat_resume import POLICY, verify_continuation
from ir_training.train.gpu_profile import build_gpu_profile
from ir_training.train.resume_contract import (
    build_resume_contract,
    file_sha256,
    resolve_export_training_lineage,
)


def _file(path: Path, data: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _profile() -> dict:
    devices = [{"visible_index": i, "launch_identifier": str(i), "uuid": f"GPU-{i}",
                "name": "NVIDIA H100 80GB HBM3", "total_memory_bytes": 80 * 1024**3,
                "compute_capability": [9, 0]} for i in range(4)]
    return build_gpu_profile({"version": 1, "inherited_cuda_visible_devices": None,
                              "visible_gpu_count": 4, "devices": devices},
                             model="e2b", cpu_count=64)


def _fixture(tmp_path: Path, *, backend: str = "ddp", eval_steps: int = 5) -> tuple[FullParameterQATOptions, Path, dict]:
    seed, inputs = tmp_path / "seed", tmp_path / "inputs"
    for name in ("config.json", "tokenizer_config.json", "mobile_training_seed_manifest.json", "mobile_qparams.json"):
        _file(seed / name, b"{}")
    for name in ("model.safetensors", "mobile_qparams.safetensors"):
        _file(seed / name)
    for name in ("train.jsonl", "val.jsonl", "golden32.jsonl", "golden35.jsonl", "bixby50.jsonl"):
        _file(inputs / name, b"{}\n")
    options = FullParameterQATOptions(
        model_dir=seed, input_dir=inputs, output_dir=tmp_path / "original",
        exporter_python=Path(sys.executable), steps=20, eval_steps=eval_steps,
        golden_every_steps=1000 if eval_steps > 10 else 10, distributed_backend=backend,
        zero_stage=3 if backend == "sharded" else 2,
    )
    plan = build_plan(options)
    prepared = options.output_dir / "prepared"
    for name in ("train.jsonl", "val.jsonl", "golden32.jsonl", "golden35.jsonl", "bixby50.jsonl"):
        shutil.copy2(inputs / name, _file(prepared / name))
    report = {"tokenizer": {}, "final_evaluation_datasets": {}}
    config = training_config(plan, _profile(), report)
    config_path = Path(plan["paths"]["config"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint = Path(plan["paths"]["training"]) / "checkpoint-5"
    _file(checkpoint / "model.safetensors")
    _file(checkpoint / "config.json", b"{}")
    _file(checkpoint / "tokenizer_config.json", b"{}")
    shutil.copy2(config_path, checkpoint / "training_config.yaml")
    _file(checkpoint / "trainer_state.json", json.dumps({"global_step": 5, "max_steps": 20}).encode())
    _file(checkpoint / "golden_callback_state.json", b"{}")
    for rank in range(4):
        _file(checkpoint / f"rng_state_{rank}.pth")
    if backend == "sharded":
        _file(checkpoint / "latest", b"global_step5\n")
        for rank in range(4):
            _file(checkpoint / "global_step5" / f"zero_pp_rank_{rank}_optim_states.pt")
        _file(checkpoint / "global_step5" / "mp_rank_00_model_states.pt")
    else:
        _file(checkpoint / "optimizer.pt")
        _file(checkpoint / "scheduler.pt")
    manifest = _checkpoint_adapter_manifest(checkpoint, role="trainer_intermediate")
    metadata = {
        "checkpoint_role": "trainer_intermediate", "checkpoint_kind": "full_model",
        "checkpoint_step": 5, "config_path": str(config_path),
        "training_config_sha256": file_sha256(config_path),
        "resume_contract": build_resume_contract(config, prepared, effective_batch=32),
        "effective_batch_size": 32, "training": config["training"],
        "lora": {}, "model": config["model"], "dataset_dir": str(prepared),
        "adapter_checkpoints": [manifest],
        "checkpoint_adapter_files": [
            {"path": item["path"], "sha256": item["sha256"], "size_bytes": item["size"]}
            for item in manifest["files"]
        ],
    }
    _file(checkpoint / "training_metadata.json", json.dumps(metadata).encode())
    return options, checkpoint, report


def _continuation(options: FullParameterQATOptions, checkpoint: Path, report: dict) -> tuple[dict, Path]:
    next_options = replace(options, output_dir=options.output_dir.parent / "continued",
                           steps=30, resume_from_checkpoint=checkpoint)
    plan = build_plan(next_options)
    prepared = next_options.output_dir / "prepared"
    for name in ("train.jsonl", "val.jsonl", "golden32.jsonl", "golden35.jsonl", "bixby50.jsonl"):
        shutil.copy2(options.output_dir / "prepared" / name, _file(prepared / name))
    return training_config(plan, _profile(), report), prepared


@pytest.mark.parametrize("backend", ["ddp", "sharded"])
def test_full_qat_resume_restores_bound_trainer_state_without_touching_source(tmp_path, backend):
    options, checkpoint, report = _fixture(tmp_path, backend=backend)
    before = file_sha256(checkpoint / "training_metadata.json")
    config, prepared = _continuation(options, checkpoint, report)
    assert config["training"]["resume_policy"] == POLICY
    assert config["training"]["refuse_resume"] is False
    assert config["training"]["max_steps"] == 30
    state = verify_continuation(checkpoint, config)
    assert state["global_step"] == 5 and state["verified"] is True
    assert state["continuation"]["scheduler_warmup_steps"] == config["training"]["warmup_steps"]
    assert len(state["state_files_sha256"]) >= 8
    assert file_sha256(checkpoint / "training_metadata.json") == before
    assert prepared.is_dir()


def test_full_qat_resume_rejects_recipe_data_and_state_changes(tmp_path):
    options, checkpoint, report = _fixture(tmp_path)
    config, prepared = _continuation(options, checkpoint, report)
    bad = copy.deepcopy(config)
    bad["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="horizon|contract|config changes"):
        verify_continuation(checkpoint, bad)
    _file(prepared / "train.jsonl", b"{\"changed\":true}\n")
    with pytest.raises(ValueError, match="identical prepared"):
        verify_continuation(checkpoint, config)
    shutil.copy2(options.output_dir / "prepared/train.jsonl", prepared / "train.jsonl")
    _file(prepared / "bixby50.jsonl", b"{\"changed\":true}\n")
    with pytest.raises(ValueError, match="identical prepared holdout"):
        verify_continuation(checkpoint, config)
    shutil.copy2(options.output_dir / "prepared/bixby50.jsonl", prepared / "bixby50.jsonl")
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="state files"):
        verify_continuation(checkpoint, config)


def test_full_qat_resume_preserves_capped_eval_and_save_cadence(tmp_path):
    options, checkpoint, report = _fixture(tmp_path, eval_steps=500)
    config, _ = _continuation(options, checkpoint, report)
    source = yaml.safe_load((options.output_dir / "fit/training_config.yaml").read_text())
    assert config["training"]["eval_steps"] == source["training"]["eval_steps"] == 20
    assert config["training"]["save_steps"] == source["training"]["save_steps"] == 20
    assert config["golden_eval"]["interval"] == source["golden_eval"]["interval"]
    assert verify_continuation(checkpoint, config)["verified"]


def test_full_qat_resume_rejects_weight_only_and_backend_switch(tmp_path):
    options, checkpoint, report = _fixture(tmp_path)
    config, _ = _continuation(options, checkpoint, report)
    bad = copy.deepcopy(config)
    bad["training"]["distributed_backend"] = "sharded"
    with pytest.raises(ValueError, match="backend"):
        verify_continuation(checkpoint, bad)
    best = checkpoint.parent / "best_golden_checkpoint"
    shutil.copytree(checkpoint, best)
    with pytest.raises(ValueError, match="numbered Trainer checkpoint"):
        verify_continuation(best, {**config, "training": {**config["training"], "resume_from_checkpoint": str(best)}})


def test_sharded_resume_rejects_incomplete_zero_partition(tmp_path):
    options, checkpoint, report = _fixture(tmp_path, backend="sharded")
    config, _ = _continuation(options, checkpoint, report)
    (checkpoint / "global_step5/zero_pp_rank_3_optim_states.pt").unlink()
    with pytest.raises(ValueError, match="complete optimizer/model state shards"):
        verify_continuation(checkpoint, config)


def test_full_qat_export_rechecks_physical_resume_lineage(tmp_path):
    options, checkpoint, report = _fixture(tmp_path)
    config, prepared = _continuation(options, checkpoint, report)
    config_path = prepared.parent / "fit/training_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _file(config_path.parent / "preparation_report.json", json.dumps({
        "training_config_sha256": file_sha256(config_path), "model_files": {"config.json": "digest"},
    }).encode())
    best = prepared.parent / "fit/training/best_golden_checkpoint"
    _file(best / "model.safetensors")
    _file(best / "config.json", b"{}")
    _file(best / "tokenizer_config.json", b"{}")
    shutil.copy2(config_path, best / "training_config.yaml")
    metadata = {"training_config_sha256": file_sha256(config_path),
                "resume_state": verify_continuation(checkpoint, config)}
    _file(best / "training_metadata.json", json.dumps(metadata).encode())
    result = resolve_export_training_lineage(config_path, best)
    assert result["resume_lineage"]["physical_chain_complete"] is True
    _file(checkpoint / "optimizer.pt", b"tampered")
    with pytest.raises(ValueError, match="different resume source evidence"):
        resolve_export_training_lineage(config_path, best)
