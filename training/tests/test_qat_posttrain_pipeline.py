"""Opt-in post-training orchestration; no generation, model or device loading."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset" / "src"))

from ir_training.pipeline import qat_grpo as workflow
from ir_training.train import grpo_continuation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture(params=["e2b", "270m"])
def options(request, tmp_path):
    model, inputs, adapter = tmp_path / "base", tmp_path / "pairs", tmp_path / "sft"
    for name in ("config.json", "tokenizer_config.json", "mobile_training_seed_manifest.json", "mobile_qparams.json", "mobile_qparams.safetensors"):
        write(model / name, {})
    for name in ("train.jsonl", "val.jsonl"):
        write(inputs / name, {})
    source = write(tmp_path / "source.yaml", {
        "run": {"id": "sft"}, "model": {"model_id": "fixture", "model_source": str(model)},
        "training": {"method": "qat_lora_sft"}, "golden_eval": {},
        "lora": {"r": 16}, "qat": {"enabled": True, "scale_mode": "retained_mobile" if request.param == "e2b" else "dynamic"},
    })
    write(adapter / "adapter_config.json", {})
    write(adapter / "adapter_model.safetensors", {})
    write(adapter / "training_metadata.json", {"training_config_sha256": workflow.sha256(source)})
    return workflow.QATGRPOOptions(
        family=request.param, model_dir=model, input_dir=inputs, sft_config=source,
        sft_checkpoint=adapter, output_dir=tmp_path / "new", official_litertlm=write(tmp_path / "official.litertlm", {}),
        exporter_python=Path(sys.executable), source_safetensors=write(tmp_path / "source.safetensors", {}),
    )


def profile(batch=32):
    return {"dtype": "bfloat16", "effective_batch_size": batch, "world_size": 1, "microbatch": 1,
            "gradient_accumulation_steps": batch, "cuda_visible_devices": "0", "attn_implementation": "sdpa",
            "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "tf32": True, "dataloader_num_workers": 0}


def prepared():
    return {"tokenizer": {}, "final_evaluation_datasets": {"golden35": {}, "bixby50": {}}}


def preference_file(options):
    return write(options.output_dir.parent / "preferences.jsonl", {
        "source_id": "fixture", "chosen": "chosen", "rejected": "rejected",
    })


def test_plain_grpo_config_matches_legacy_options_exactly(options):
    plan = workflow.build_plan(options)
    legacy = {**plan, "options": dict(plan["options"])}
    for key in ("objective", "sft_rehearsal_weight", "dpo_preferences", "dpo_beta"):
        legacy["options"].pop(key)
    assert workflow.training_config(plan, profile(), prepared()) == workflow.training_config(legacy, profile(), prepared())
    assert "algorithm" not in plan and "preference_files" not in plan
    assert "objective" not in workflow.training_config(plan, profile(), prepared())["grpo"]


def test_rehearsal_is_explicit_preserves_qat_export_and_selection(options):
    baseline = workflow.build_plan(options)
    plan = workflow.build_plan(replace(options, objective="grpo_sft"))
    config = workflow.training_config(plan, profile(), prepared())
    original = workflow.training_config(baseline, profile(), prepared())
    assert config["grpo"]["objective"] == "grpo_sft"
    assert config["grpo"]["rehearsal"] == {"weight": 0.1}
    assert plan["effective_batch_unit"] == "generated_completions"
    for key in ("qat", "lora", "golden_eval", "training", "final_evaluation_datasets"):
        assert config[key] == original[key]
    assert plan["stages"] == baseline["stages"] and plan["selection"] == baseline["selection"]
    assert config["grpo"]["native_quality_policy"] == original["grpo"]["native_quality_policy"]


def test_dpo_counts_pairs_and_binds_input_before_stage_work(options, monkeypatch):
    preferences = preference_file(options)
    plan = workflow.build_plan(replace(options, objective="dpo", dpo_preferences=preferences, effective_batch=31))
    config = workflow.training_config(plan, profile(31), prepared())
    assert config["grpo"]["objective"] == "dpo"
    assert config["grpo"]["dpo"] == {"preferences": str(preferences), "beta": 0.1,
                                      "preferences_sha256": workflow.sha256(preferences)}
    assert config["training"]["per_device_eval_batch_size"] == 1
    assert config["training"]["expected_effective_batch_size"] == 31
    assert plan["algorithm"] == "dpo" and plan["effective_batch_unit"] == "preference_pairs"
    assert plan["preference_files"] == {str(preferences): workflow.sha256(preferences)}
    write(preferences, {"tampered": True})
    monkeypatch.setattr(workflow, "prepare_data", lambda *args: pytest.fail("must reject before preparing"))
    with pytest.raises(ValueError):
        workflow.run_stage(plan, "prepare")
    with pytest.raises(ValueError):
        workflow.training_config(plan, profile(31), prepared())


@pytest.mark.parametrize("changes,error", [
    ({"objective": "unknown"}, "objective"),
    ({"sft_rehearsal_weight": 0.1}, "only valid"),
    ({"dpo_beta": 0.1}, "only valid"),
    ({"objective": "grpo_sft", "sft_rehearsal_weight": 0}, "positive and finite"),
    ({"objective": "grpo_sft", "sft_rehearsal_weight": float("nan")}, "positive and finite"),
    ({"objective": "grpo_sft", "sft_rehearsal_weight": True}, "positive and finite"),
    ({"objective": "dpo"}, "requires --dpo-preferences"),
])
def test_objective_options_fail_closed_before_output(options, changes, error):
    with pytest.raises(ValueError, match=error):
        workflow.build_plan(replace(options, **changes))
    assert not options.output_dir.exists()


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf"), True])
def test_invalid_dpo_beta_is_rejected(options, beta):
    with pytest.raises(ValueError, match="positive and finite"):
        workflow.build_plan(replace(options, objective="dpo", dpo_preferences=preference_file(options), dpo_beta=beta))


def test_preference_source_cannot_be_inside_output(options):
    preferences = write(options.output_dir / "preferences.jsonl", {})
    with pytest.raises(ValueError, match="contain an input artifact"):
        workflow.build_plan(replace(options, objective="dpo", dpo_preferences=preferences))


def test_dpo_rejects_rollout_cache_before_expensive_stages(options):
    with pytest.raises(ValueError, match="does not support rollout"):
        workflow.build_plan(replace(options, objective="dpo", dpo_preferences=preference_file(options),
                                    rollout_qat_weight_cache=True))


def test_dpo_configure_receipt_binds_external_preference_file(options, monkeypatch):
    import ir_training.train.gpu_profile as gpu
    import ir_training.train.qat_grpo_contract as contract
    plan = workflow.build_plan(replace(options, objective="dpo", dpo_preferences=preference_file(options)))
    monkeypatch.setattr(workflow.official_mobile, "_scripts", lambda: None)
    monkeypatch.setitem(sys.modules, "prepare_review_training", SimpleNamespace(verify_prepared=lambda *args, **kwargs: prepared()))
    monkeypatch.setattr(gpu, "detect_cuda_devices", dict)
    monkeypatch.setattr(gpu, "build_gpu_profile", lambda *args, **kwargs: profile())
    monkeypatch.setattr(contract, "verify_sft_adapter_lineage", lambda config: {"verified": True})
    monkeypatch.setattr(workflow, "_run", lambda *args, **kwargs: None)
    files = workflow.run_stage(plan, "configure")
    assert Path(plan["options"]["dpo_preferences"]) in files


@pytest.mark.parametrize("objective", ["grpo_sft", "dpo"])
def test_training_receipt_rejects_wrong_objective_metadata(options, monkeypatch, objective):
    plan = workflow.build_plan(replace(options, objective=objective,
        dpo_preferences=preference_file(options) if objective == "dpo" else None))
    config_path = write(Path(plan["paths"]["config"]), {})
    write(Path(plan["paths"]["gpu_profile"]), {"world_size": 1})
    write(Path(plan["paths"]["best_checkpoint"]) / "training_metadata.json", {
        "checkpoint_role": "best_golden", "training_config_sha256": workflow.sha256(config_path),
        "best_golden_eval": {"metric": workflow.official_mobile.SELECTOR},
        "training": {"method": "qat_lora_grpo"}, "grpo": {"algorithm": "grpo"},
    })
    monkeypatch.setattr(workflow.official_mobile, "_require_no_op_export", lambda plan: None)
    monkeypatch.setattr(workflow, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow, "verify_quality_gate", lambda *args: pytest.fail("wrong objective must fail first"))
    with pytest.raises(ValueError, match="algorithm differs"):
        workflow.run_stage(plan, "training")


def cli_module():
    spec = importlib.util.spec_from_file_location("posttrain_cli", ROOT / "scripts/run_qat_grpo_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("objective", ["grpo_sft", "dpo"])
def test_fresh_rerun_objective_overrides_do_not_reset_original_budgets(options, monkeypatch, objective):
    cli = cli_module()
    original = replace(options, max_steps=400, learning_rate=5e-7, seed=17)
    monkeypatch.setattr(grpo_continuation, "options_from_previous_run", lambda *args: original)
    captured = []
    monkeypatch.setattr(cli, "run_pipeline", lambda opt, **kw: captured.append(opt) or {})
    args = ["--rerun-from", "old", "--output-dir", str(options.output_dir), "--objective", objective]
    if objective == "dpo":
        args += ["--dpo-preferences", str(preference_file(options)), "--dpo-beta", "0.2"]
    else:
        args += ["--sft-rehearsal-weight", "0.05"]
    assert cli.main(args) == 0
    actual = captured[0]
    assert actual.objective == objective and actual.sft_checkpoint == original.sft_checkpoint
    assert (actual.max_steps, actual.learning_rate, actual.seed) == (400, 5e-7, 17)
    assert actual.dpo_beta == (0.2 if objective == "dpo" else None)
    assert actual.sft_rehearsal_weight == (0.05 if objective == "grpo_sft" else None)


def test_rerun_switching_back_to_plain_clears_other_mode_settings(options, monkeypatch):
    cli = cli_module()
    original = replace(options, objective="dpo", dpo_beta=0.2, dpo_preferences=preference_file(options))
    monkeypatch.setattr(grpo_continuation, "options_from_previous_run", lambda *args: original)
    captured = []
    monkeypatch.setattr(cli, "run_pipeline", lambda opt, **kw: captured.append(opt) or {})
    cli.main(["--rerun-from", "old", "--output-dir", str(options.output_dir), "--objective", "grpo"])
    assert captured[0].objective == "grpo"
    assert captured[0].dpo_preferences is None and captured[0].dpo_beta is None


def test_fresh_rerun_restores_dpo_preference_path_as_path(options):
    original = replace(options, objective="dpo", dpo_preferences=preference_file(options), dpo_beta=0.2)
    plan = workflow.build_plan(original)
    write(original.output_dir / workflow.PLAN_NAME, plan)
    replay = grpo_continuation.options_from_previous_run(original.output_dir, original.output_dir.parent / "fresh")
    assert replay.objective == "dpo" and replay.dpo_beta == 0.2
    assert replay.dpo_preferences == original.dpo_preferences
