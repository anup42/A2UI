"""Post-training identity must survive a run-only copy to the Android host."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from ir_training.eval import android_native_quality as native
from ir_training.eval.qat_grpo_native import inspect_qat_grpo_run
from test_qat_grpo_native import fixture, write


@pytest.mark.parametrize("family", ["e2b", "270m"])
@pytest.mark.parametrize("objective", ["grpo_sft", "dpo"])
def test_native_posttrain_objectives_relocate_without_external_preference_access(tmp_path, monkeypatch, family, objective):
    run, _ = fixture(tmp_path, monkeypatch, family, objective=objective)
    original_hash = native.file_sha256
    def local_hash(path):
        assert str(path) != "/external/preferences.jsonl", "native host must not open the training input"
        return original_hash(path)
    monkeypatch.setattr(native, "file_sha256", local_hash)
    binding = inspect_qat_grpo_run(run)
    assert binding["objective"] == objective
    moved = tmp_path / "another-host"
    shutil.copytree(run, moved)
    assert inspect_qat_grpo_run(moved)["objective"] == objective


@pytest.mark.parametrize("family", ["e2b", "270m"])
@pytest.mark.parametrize("tamper", ["plan_hash", "config_path", "config_hash", "receipt_hash", "missing_receipt", "extra_external"])
def test_native_dpo_preference_history_requires_exact_identity(tmp_path, monkeypatch, family, tamper):
    run, paths = fixture(tmp_path, monkeypatch, family, objective="dpo")
    state = native.read_json(run / "qat_grpo_manifest.json")
    if tamper == "plan_hash":
        state["plan"]["preference_files"]["/external/preferences.jsonl"] = "f" * 64
        write(run / "qat_grpo_plan.json", state["plan"])
    elif tamper.startswith("config_"):
        config = native.read_json(paths["config"])
        key = "preferences" if tamper == "config_path" else "preferences_sha256"
        config["grpo"]["dpo"][key] = "/other/file.jsonl" if key == "preferences" else "f" * 64
        write(paths["config"], config)
    else:
        receipt = state["completed"]["configure"]
        if tamper == "receipt_hash":
            receipt["files"]["/external/preferences.jsonl"] = "f" * 64
        elif tamper == "missing_receipt":
            receipt["files"].pop("/external/preferences.jsonl")
        else:
            receipt["files"]["/external/undeclared.jsonl"] = "f" * 64
        write(run / "stage_receipts/configure.json", receipt)
    write(run / "qat_grpo_manifest.json", state)
    with pytest.raises(ValueError, match="preference|escapes|outside"):
        inspect_qat_grpo_run(run)


@pytest.mark.parametrize("family", ["e2b", "270m"])
@pytest.mark.parametrize("objective", ["grpo_sft", "dpo"])
@pytest.mark.parametrize("tamper", ["plan", "config", "metadata"])
def test_native_rejects_posttrain_objective_disagreement(tmp_path, monkeypatch, family, objective, tamper):
    run, paths = fixture(tmp_path, monkeypatch, family, objective=objective)
    state = native.read_json(run / "qat_grpo_manifest.json")
    if tamper == "plan":
        state["plan"]["options"]["objective"] = "grpo"
        write(run / "qat_grpo_plan.json", state["plan"])
    elif tamper == "config":
        config = native.read_json(paths["config"])
        config["grpo"]["objective"] = "grpo"
        write(paths["config"], config)
    else:
        path = paths["best_checkpoint"] / "training_metadata.json"
        metadata = native.read_json(path)
        metadata["grpo"]["algorithm"] = "grpo"
        write(path, metadata)
        receipt = state["completed"]["training"]
        receipt["files"]["/training/grpo-run/training/best_golden_checkpoint/training_metadata.json"] = native.file_sha256(path)
        write(run / "stage_receipts/training.json", receipt)
    write(run / "qat_grpo_manifest.json", state)
    with pytest.raises(ValueError, match="training objective"):
        inspect_qat_grpo_run(run)
