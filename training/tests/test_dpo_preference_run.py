"""Offline preference source bindings; no GPU/model loading or generation."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset/src"))

from ir_training.train import dpo_preference_run as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture(params=["e2b", "270m"])
def run(request, tmp_path, monkeypatch):
    root, model = tmp_path / "run", tmp_path / "base"
    prepared, configs = root / "prepared", root / "configs"
    output = root / "training"
    output.mkdir(parents=True)
    for name in ("train", "val", "golden32", "golden35", "bixby50"):
        write(prepared / f"{name}.jsonl", {"source_id": name})
    scaffold = write(prepared / "prompt_scaffolds.json", [])
    manifest = {"splits": {name: {"output_sha256": module._sha(prepared / f"{name}.jsonl")}
                           for name in ("train", "val")},
                "prompt_scaffolds_sha256": module._sha(scaffold)}
    write(prepared / "manifest.json", manifest)
    finals = {name: {"split_path": str(prepared / f"{name}.jsonl"),
                    "split_sha256": module._sha(prepared / f"{name}.jsonl"),
                    "selection_role": "development_checkpoint_selection" if name == "golden32" else "final_only_holdout"}
              for name in ("golden32", "golden35", "bixby50")}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        write(model / name, {"name": name})
    config = {"run": {"purpose": "qat_lora_grpo_v1", "dataset_dir": str(prepared), "output_dir": str(output)},
              "model": {"model_source": str(model), "tokenizer_source": str(model)},
              "training": {"method": "qat_lora_grpo", "max_seq_length": 4096,
                           "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
                           "expected_effective_batch_size": 32},
              "grpo": {"family": request.param, "max_prompt_length": 4096, "max_completion_length": 2048,
                       "reward_config": str(configs / "grpo_reward.yaml")},
              "golden_eval": {"split_path": str(prepared / "golden32.jsonl")},
              "final_evaluation_datasets": finals}
    config_path = write(configs / "grpo_training.yaml", config)
    report = {"training_config_sha256": module._sha(config_path),
              "dataset_manifest_sha256": module._sha(prepared / "manifest.json"),
              "final_evaluation_datasets": finals,
              "model_files": {name: module._sha(model / name) for name in ("config.json", "tokenizer_config.json", "tokenizer.json")}}
    write(configs / "preparation_report.json", report)
    write(configs / "gpu_profile.json", {"world_size": 4, "effective_batch_size": 32})
    write(configs / "grpo_reward.yaml", {})
    plan = {"schema_version": 1, "workflow": "qat_lora_grpo_v1",
            "options": {"output_dir": str(root), "family": request.param, "model_dir": str(model), "effective_batch": 32},
            "paths": {"config": str(config_path), "prepared": str(prepared),
                      "reward_config": str(configs / "grpo_reward.yaml"), "gpu_profile": str(configs / "gpu_profile.json")}}
    write(root / "qat_grpo_plan.json", plan)
    write(output / "grpo_periodic_rewards.rank0.jsonl", {"mode": "train", "source_id": "train"})
    state = {"root": root, "config": config, "plan": plan, "report": report,
             "calls": [], "tokenizer": object(), "model": model}

    def seal():
        write(config_path, config)
        report["training_config_sha256"] = module._sha(config_path)
        write(configs / "preparation_report.json", report)
        write(root / "qat_grpo_plan.json", plan)
        write(root / "stage_receipts/configure.json", {"stage": "configure",
              "plan_sha256": module._sha(root / "qat_grpo_plan.json"),
              "files": {str(path): module._sha(path) for path in
                        (config_path, configs / "preparation_report.json", configs / "gpu_profile.json", configs / "grpo_reward.yaml")}})

    state["seal"] = seal
    seal()
    from ir_training.data import audit_filter
    from ir_training.train import prepared_binding, qat_grpo, qat_grpo_contract
    monkeypatch.setattr(qat_grpo_contract, "validate_qat_grpo_config", lambda value: state["calls"].append("config"))

    def prepared_contract(value):
        state["calls"].append("prepared")
        return {key: report[key] for key in ("dataset_manifest_sha256", "final_evaluation_datasets")}

    monkeypatch.setattr(qat_grpo, "verify_prepared_training_contract", prepared_contract)
    monkeypatch.setattr(audit_filter, "load_reserved_cohorts", lambda paths: {
        "evidence": [{"path": str(path), "sha256": module._sha(path)} for path in paths]})
    monkeypatch.setattr(prepared_binding, "verify_tokenizer_binding", lambda *a, **kw: {"verified": True})

    def tokenizer(path, config):
        state["calls"].append("tokenizer")
        assert path == model
        return state["tokenizer"]

    monkeypatch.setattr(module, "_load_tokenizer", tokenizer)
    monkeypatch.setattr(module, "_load_training_rows", lambda path, *args: [{"source_id": path.stem}])
    return state


def test_reads_only_periodic_audits_and_canonical_train_context(run):
    root = run["root"]
    write(root / "evaluations/golden32/grpo_periodic_rewards.rank0.jsonl", {"heldout": True})
    write(root / "runtime_preflight/grpo_periodic_rewards.rank0.jsonl", {"probe": True})
    write(root / "training/grpo_rewards.rank0.jsonl", {"legacy": True})
    result = module.load_preference_run(root)
    assert result["train_rows"] == [{"source_id": "train"}]
    assert result["forbidden_rows"] == [{"source_id": "val"}]
    assert result["family"] == run["plan"]["options"]["family"]
    assert result["effective_batch"] == 32
    assert result["audit_paths"] == [root / "training/grpo_periodic_rewards.rank0.jsonl"]
    assert result["tokenizer"] is run["tokenizer"]
    assert result["evidence"]["model_weights_loaded"] is False
    assert run["calls"] == ["config", "prepared", "tokenizer"]
    json.dumps(result["evidence"])


@pytest.mark.parametrize("artifact", ["grpo_training.yaml", "preparation_report.json", "grpo_reward.yaml", "gpu_profile.json"])
def test_sealed_input_drift_fails_before_tokenizer(run, artifact):
    write(run["root"] / "configs" / artifact, {"tampered": True})
    with pytest.raises(ValueError, match="changed after preparation"):
        module.load_preference_run(run["root"])
    assert "tokenizer" not in run["calls"]


def test_sealed_report_still_must_bind_config(run):
    run["report"]["training_config_sha256"] = "0" * 64
    path = write(run["root"] / "configs/preparation_report.json", run["report"])
    receipt_path = run["root"] / "stage_receipts/configure.json"
    receipt = module._object(receipt_path)
    receipt["files"][str(path)] = module._sha(path)
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="preparation report"):
        module.load_preference_run(run["root"])
    assert "tokenizer" not in run["calls"]


@pytest.mark.parametrize("cohort", ["golden32", "golden35", "bixby50"])
def test_missing_reserved_cohort_fails_before_tokenizer(run, cohort):
    del run["config"]["final_evaluation_datasets"][cohort]
    run["seal"]()
    with pytest.raises(ValueError, match="Required reserved cohort"):
        module.load_preference_run(run["root"])
    assert "tokenizer" not in run["calls"]


def test_canonical_leakage_failure_propagates_before_tokenizer(run, monkeypatch):
    from ir_training.train import qat_grpo

    def reject(config):
        raise ValueError("Reserved Golden source overlaps train")

    monkeypatch.setattr(qat_grpo, "verify_prepared_training_contract", reject)
    with pytest.raises(ValueError, match="Reserved Golden"):
        module.load_preference_run(run["root"])
    assert "tokenizer" not in run["calls"]


@pytest.mark.parametrize("field,value", [("objective", "dpo"), ("objective", "unknown"), ("family", "other")])
def test_only_original_supported_grpo_modes(run, field, value):
    run["config"]["grpo"][field] = value
    run["seal"]()
    with pytest.raises(ValueError, match="Config must match"):
        module.load_preference_run(run["root"])
    assert "tokenizer" not in run["calls"]


def test_rehearsal_mode_supported(run):
    run["plan"]["options"]["objective"] = "grpo_sft"
    run["config"]["grpo"]["objective"] = "grpo_sft"
    run["seal"]()
    assert module.load_preference_run(run["root"])["evidence"]["objective"] == "grpo_sft"


def test_tokenizer_tampering_fails_before_loading(run):
    write(run["model"] / "tokenizer.json", {"tampered": True})
    with pytest.raises(ValueError, match="tokenizer.json"):
        module.load_preference_run(run["root"])
    assert "tokenizer" not in run["calls"]


def test_wrong_effective_batch_fails(run):
    run["config"]["training"]["gradient_accumulation_steps"] = 4
    run["seal"]()
    with pytest.raises(ValueError, match="effective batch"):
        module.load_preference_run(run["root"])


def test_missing_audits_does_not_adopt_evaluation_files(run):
    path = run["root"] / "training/grpo_periodic_rewards.rank0.jsonl"
    path.rename(path.with_name("scored_predictions.jsonl"))
    write(run["root"] / "evaluations/golden32/grpo_periodic_rewards.rank0.jsonl", {})
    with pytest.raises(ValueError, match="No periodic"):
        module.load_preference_run(run["root"])


def test_invalid_audit_filename_fails(run):
    write(run["root"] / "training/grpo_periodic_rewards.rankbad.jsonl", {})
    with pytest.raises(ValueError, match="Invalid periodic"):
        module.load_preference_run(run["root"])


def test_read_only_context_detects_input_drift_during_load(run, monkeypatch):
    def rows(path, *args):
        if path.stem == "val":
            write(run["root"] / "prepared/train.jsonl", {"drift": True})
        return [{"source_id": path.stem}]

    monkeypatch.setattr(module, "_load_training_rows", rows)
    with pytest.raises(ValueError, match="changed while loading"):
        module.load_preference_run(run["root"])


@pytest.mark.parametrize("config,expected_loader,extra", [
    ({"tokenizer_loader": "pretrained_tokenizer_fast", "processor_fallback": True}, "fast", {}),
    ({}, "auto", {"use_fast": True}),
    ({"use_fast_tokenizer": False}, "auto", {"use_fast": False}),
    ({"tokenizer_loader": "auto_processor"}, "processor", {}),
])
def test_tokenizer_configured_loader_is_offline(monkeypatch, tmp_path, config, expected_loader, extra):
    observed = []
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")

    def load(name):
        def apply(*args, **kwargs):
            observed.append((name, kwargs))
            return SimpleNamespace(tokenizer=tokenizer) if name == "processor" else tokenizer
        return SimpleNamespace(from_pretrained=apply)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=load("auto"), PreTrainedTokenizerFast=load("fast"), AutoProcessor=load("processor")))
    assert module._load_tokenizer(tmp_path, {**config, "padding_side": "left"}) is tokenizer
    assert observed == [(expected_loader, {"local_files_only": True, "trust_remote_code": False, **extra})]
    assert tokenizer.pad_token == "<eos>" and tokenizer.padding_side == "left"


@pytest.mark.parametrize("enabled", [False, True])
def test_configured_e2b_processor_fallback_is_offline(monkeypatch, tmp_path, enabled):
    observed = []
    tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")

    def fast(*args, **kwargs):
        observed.append(("fast", kwargs))
        raise ValueError("fast tokenizer unavailable")

    def processor(*args, **kwargs):
        observed.append(("processor", kwargs))
        return SimpleNamespace(tokenizer=tokenizer)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=None, PreTrainedTokenizerFast=SimpleNamespace(from_pretrained=fast),
        AutoProcessor=SimpleNamespace(from_pretrained=processor)))
    config = {"tokenizer_loader": "pretrained_tokenizer_fast", "processor_fallback": enabled}
    if enabled:
        assert module._load_tokenizer(tmp_path, config) is tokenizer
        assert tokenizer.pad_token == "pad" and tokenizer.padding_side == "right"
    else:
        with pytest.raises(ValueError, match="fast tokenizer unavailable"):
            module._load_tokenizer(tmp_path, config)
    assert [name for name, _ in observed] == (["fast", "processor"] if enabled else ["fast"])
    assert all(kwargs == {"local_files_only": True, "trust_remote_code": False} for _, kwargs in observed)


def test_known_model_context_overflow_fails_without_silent_shortening(run):
    path = write(run["model"] / "config.json", {"text_config": {"max_position_embeddings": 2048}})
    run["report"]["model_files"]["config.json"] = module._sha(path)
    run["seal"]()
    with pytest.raises(ValueError, match="sequence budget exceeds"):
        module.load_preference_run(run["root"])
    assert "tokenizer" not in run["calls"]
