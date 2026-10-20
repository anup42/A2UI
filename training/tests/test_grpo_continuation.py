"""Continuation gates do no model loading, generation, training, or file writes."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset/src"))

from ir_training.pipeline.qat_grpo import (
    MANIFEST_NAME,
    PLAN_NAME,
    WORKFLOW,
    QATGRPOOptions,
)
from ir_training.train import grpo_continuation as continuation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def stopped_run(tmp_path, monkeypatch):
    from ir_training.train import qat_grpo_reward

    monkeypatch.setattr(continuation, "continuation_code_identity", lambda: {"sha256": "code-v1"})
    monkeypatch.setattr(qat_grpo_reward, "reward_implementation_identity", lambda path=None: {"sha256": "reward-v1"})
    output = tmp_path / "grpo"
    model = tmp_path / "model"
    seed_file = write(model / "config.json", {"model": "seed"})
    sft_config = write(tmp_path / "sft.yaml", {"original": "sft"})
    official = write(tmp_path / "official.litertlm", {"official": True})
    source = write(tmp_path / "sft/adapter_model.safetensors", {"source": "sft adapter"})
    raw = tmp_path / "input"
    for name in ("train.jsonl", "val.jsonl"):
        write(raw / name, {})
    options = QATGRPOOptions(family="e2b", model_dir=model, sft_config=sft_config,
        sft_checkpoint=source.parent, output_dir=output, official_litertlm=official,
        exporter_python=Path(sys.executable), input_dir=raw, max_steps=800,
        learning_rate=5e-7, num_generations=4, effective_batch=32, seed=17,
        rollout_microbatch=4, rollout_qat_weight_cache=True, defer_native_eval=True)
    values = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(options).items()}
    stages = ["assets", "prepare", "configure", "no_op_export", "runtime_preflight", "training",
              "best_golden32", "best_golden35", "best_bixby50", "merge", "export", "native_quality"]
    plan = {"schema_version": 1, "workflow": WORKFLOW, "options": values, "stages": stages,
            "paths": {"reward_config": str(output / "configs/reward.yaml"),
                      "merged": str(output / "merged"), "export_dir": str(output / "export")},
            "source_config_sha256": sha(sft_config), "source_adapter_files": {str(source): sha(source)},
            "official_artifact_sha256": sha(official)}
    write(output / PLAN_NAME, plan)
    digest = sha(output / PLAN_NAME)
    report = write(output / "configs/preparation_report.json", {"model_files": {seed_file.name: sha(seed_file)}})
    contract = write(output / "configs" / continuation.CONTRACT_NAME, continuation.build_continuation_contract(plan))
    state = {"workflow": WORKFLOW, "plan": plan, "status": "failed", "active_stage": "best_golden32", "completed": {}}
    for stage in stages[:stages.index("training") + 1]:
        artifact = write(output / f"evidence/{stage}.json", {"stage": stage})
        files = {str(artifact): sha(artifact)}
        if stage == "configure":
            files.update({str(contract): sha(contract), str(report): sha(report)})
        receipt = {"stage": stage, "plan_sha256": digest, "files": files}
        write(output / f"stage_receipts/{stage}.json", receipt)
        state["completed"][stage] = receipt
    write(output / MANIFEST_NAME, state)
    return output, plan, state


def test_only_verified_downstream_stages_are_returned_without_writing(stopped_run):
    output, plan, state = stopped_run
    before = {str(path): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    result = continuation.verify_post_training_continuation(output)
    assert result["plan"] == plan and result["state"] == state
    assert result["pending_stages"] == ["best_golden32", "best_golden35", "best_bixby50", "merge", "export", "native_quality"]
    assert not {"prepare", "runtime_preflight", "training"} & set(result["pending_stages"])
    assert before == {str(path): path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_awaiting_native_continues_only_device_stage_and_preserves_partial_device_outputs(stopped_run):
    output, plan, state = stopped_run
    digest = sha(output / PLAN_NAME)
    for stage in plan["stages"][len(state["completed"]):-1]:
        artifact = write(output / f"evidence/{stage}.json", {"stage": stage})
        receipt = {"stage": stage, "plan_sha256": digest, "files": {str(artifact): sha(artifact)}}
        state["completed"][stage] = receipt
        write(output / f"stage_receipts/{stage}.json", receipt)
    state.update(status="awaiting_native_evaluation", active_stage=None)
    write(output / MANIFEST_NAME, state)
    assert continuation.verify_post_training_continuation(output)["pending_stages"] == ["native_quality"]
    partial = write(output.parent / f"{output.name}_native_quality/manifest.json", {"partial": True})
    with pytest.raises(ValueError, match="preserved.*Archive"):
        continuation.verify_post_training_continuation(output)
    assert partial.is_file()


def test_code_identity_detects_dirty_edits_and_new_python_sources(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for directory in ("training/src", "training/scripts", "dataset/src"):
        write(root / directory / "implementation.py", {"code": 1})
    monkeypatch.setattr(continuation, "training_root", lambda: root / "training")
    original = continuation.continuation_code_identity()
    assert len(original["files"]) == 3
    write(root / "training/src/implementation.py", {"code": 2})
    assert continuation.continuation_code_identity()["sha256"] != original["sha256"]
    write(root / "training/src/implementation.py", {"code": 1})
    assert continuation.continuation_code_identity() == original
    write(root / "dataset/src/new_module.py", {"code": 3})
    assert continuation.continuation_code_identity()["sha256"] != original["sha256"]


@pytest.mark.parametrize("status", ["running", "complete", "awaiting_native_evaluation"])
def test_inconsistent_status_cannot_continue(stopped_run, status):
    output, _, state = stopped_run
    state["status"] = status
    write(output / MANIFEST_NAME, state)
    with pytest.raises(ValueError, match="status|stopped"):
        continuation.verify_post_training_continuation(output)


def test_training_must_have_a_completed_receipt(stopped_run):
    output, _, state = stopped_run
    state["completed"].pop("training")
    state["active_stage"] = "training"
    write(output / MANIFEST_NAME, state)
    with pytest.raises(ValueError, match="post-training only"):
        continuation.verify_post_training_continuation(output)


@pytest.mark.parametrize("tampering", ["receipt", "artifact", "plan", "prefix", "code", "reward", "source", "seed", "official"])
def test_changed_identity_or_evidence_fails_closed(stopped_run, monkeypatch, tampering):
    from ir_training.train import qat_grpo_reward

    output, plan, state = stopped_run
    if tampering == "receipt":
        write(output / "stage_receipts/training.json", {})
    elif tampering == "artifact":
        write(output / "evidence/training.json", {"modified": True})
    elif tampering == "plan":
        plan["options"]["max_steps"] = 900
        write(output / PLAN_NAME, plan)
        write(output / MANIFEST_NAME, state)
    elif tampering == "prefix":
        state["completed"].pop("prepare")
        write(output / MANIFEST_NAME, state)
    elif tampering == "code":
        monkeypatch.setattr(continuation, "continuation_code_identity", lambda: {"sha256": "code-v2"})
    elif tampering == "reward":
        monkeypatch.setattr(qat_grpo_reward, "reward_implementation_identity", lambda path=None: {"sha256": "reward-v2"})
    elif tampering == "source":
        write(Path(plan["options"]["sft_checkpoint"]) / "adapter_model.safetensors", {"changed": True})
    elif tampering == "seed":
        write(Path(plan["options"]["model_dir"]) / "config.json", {"changed": True})
    elif tampering == "official":
        write(Path(plan["options"]["official_litertlm"]), {"changed": True})
    with pytest.raises(ValueError):
        continuation.verify_post_training_continuation(output)


@pytest.mark.parametrize("partial", [
    "evaluations/best_golden32/partial.json", "logs/best_golden32.log", "logs/best_golden32_worker.log",
    "stage_receipts/best_golden32.json", "merged/model.safetensors", "export/model.litertlm",
])
def test_partial_outputs_and_logs_are_never_overwritten_or_deleted(stopped_run, partial):
    output, _, _ = stopped_run
    path = write(output / partial, {"partial": True})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="preserved.*Archive"):
        continuation.verify_post_training_continuation(output)
    assert path.read_bytes() == before


def test_old_run_without_sealed_contract_is_not_silently_migrated(stopped_run):
    output, _, state = stopped_run
    contract = output / "configs" / continuation.CONTRACT_NAME
    state["completed"]["configure"]["files"].pop(str(contract))
    write(output / "stage_receipts/configure.json", state["completed"]["configure"])
    write(output / MANIFEST_NAME, state)
    with pytest.raises(ValueError, match="no sealed continuation contract"):
        continuation.verify_post_training_continuation(output)


def test_fresh_rerun_reuses_original_sft_and_budget_but_never_prepared_data(stopped_run):
    output, plan, _ = stopped_run
    new = output.parent / "grpo_fresh"
    options = continuation.options_from_previous_run(output, new)
    assert options.sft_checkpoint == Path(plan["options"]["sft_checkpoint"])
    assert options.sft_config == Path(plan["options"]["sft_config"])
    assert options.input_dir == Path(plan["options"]["input_dir"])
    assert options.prepared_input_dir is None
    assert options.output_dir == new and not new.exists()
    assert options.max_steps == 800 and options.learning_rate == 5e-7
    assert options.seed == 17 and options.effective_batch == 32
    assert options.rollout_microbatch == 1 and options.rollout_qat_weight_cache is False


def test_fresh_rerun_of_prepared_run_requires_explicit_original_raw_input(stopped_run):
    output, plan, _ = stopped_run
    raw = Path(plan["options"]["input_dir"])
    plan["options"].update(input_dir=None, prepared_input_dir=str(output / "prepared"))
    write(output / PLAN_NAME, plan)
    new = output.parent / "fresh"
    with pytest.raises(ValueError, match="supply the original raw"):
        continuation.options_from_previous_run(output, new)
    assert continuation.options_from_previous_run(output, new, raw).input_dir == raw


@pytest.mark.parametrize("bad", ["existing", "nested", "prepared", "missing_val"])
def test_fresh_rerun_rejects_reused_output_or_nonraw_inputs(stopped_run, bad):
    output, plan, _ = stopped_run
    target = output.parent / "fresh"
    raw = None
    if bad == "existing":
        target.mkdir()
    elif bad == "nested":
        target = output / "new"
    elif bad == "prepared":
        raw = output / "prepared"
        write(raw / "train.jsonl", {})
        write(raw / "val.jsonl", {})
    elif bad == "missing_val":
        raw = output.parent / "missing"
        write(raw / "train.jsonl", {})
    with pytest.raises((ValueError, FileNotFoundError)):
        continuation.options_from_previous_run(output, target, raw)
    assert Path(plan["options"]["sft_config"]).is_file()
