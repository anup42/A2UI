"""CPU-only QAT GRPO orchestration contracts; no models or device are loaded."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.pipeline import qat_grpo as workflow
from ir_training.train.qat_grpo_reward import QAT_GRPO_REWARD_VERSION


def write(path: Path, value=b"synthetic fixture, never loaded") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode("utf-8"))
    return path


@pytest.fixture(params=["e2b", "270m"])
def options(request, tmp_path):
    family = request.param
    model, inputs, adapter = tmp_path / "original-base", tmp_path / "inputs", tmp_path / "sft-best"
    for name in ("config.json", "tokenizer_config.json", "mobile_training_seed_manifest.json", "mobile_qparams.json"):
        write(model / name, {})
    for name in ("model.safetensors", "mobile_qparams.safetensors"):
        write(model / name)
    for name in ("train.jsonl", "val.jsonl"):
        write(inputs / name, b"{}\n")
    config = {
        "run": {"id": "original-sft", "output_dir": "original-output"},
        "model": {"model_id": ("google/gemma-4-E2B-it-qat-mobile-transformers" if family == "e2b"
                                else "google/gemma-3-270m-it"), "model_source": str(model)},
        "training": {"method": "qat_lora_sft", "learning_rate": 2e-5,
                     "resume_from_checkpoint": "old", "deepspeed": "old-zero-config"},
        "lora": {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        "qat": {"enabled": True, "scale_mode": "retained_mobile" if family == "e2b" else "dynamic",
                "weight_bits": 8, "activation_bits": 16},
        "golden_eval": {}, "qat_mtp": {"enabled": True},
    }
    source = write(tmp_path / "original-resolved.yaml", config)
    write(adapter / "adapter_config.json", {"r": 8})
    write(adapter / "adapter_model.safetensors")
    write(adapter / "training_metadata.json", {"training_config_sha256": workflow.sha256(source)})
    return workflow.QATGRPOOptions(
        family=family, model_dir=model, input_dir=inputs, sft_config=source,
        sft_checkpoint=adapter, output_dir=tmp_path / "grpo-new",
        official_litertlm=write(tmp_path / "official.litertlm"), exporter_python=Path(sys.executable),
        source_safetensors=write(tmp_path / "original-mobile.safetensors") if family == "e2b" else None,
        defer_native_eval=True,
    )


def profile():
    return {
        "dtype": "bfloat16", "effective_batch_size": 32, "world_size": 2, "microbatch": 1,
        "gradient_accumulation_steps": 16, "cuda_visible_devices": "0,1", "attn_implementation": "sdpa",
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "tf32": True, "dataloader_num_workers": 0,
    }


def prepared():
    return {"tokenizer": {"chat_template_kwargs": {"enable_thinking": False}},
            "final_evaluation_datasets": {"golden35": {"split": "golden35"}, "bixby50": {"split": "bixby50"}}}


def test_selection_cannot_precede_first_complete_health_window(options):
    with pytest.raises(ValueError, match="20-update GRPO health window"):
        workflow.build_plan(replace(options, golden_every_steps=10))


def test_plan_is_read_only_and_keeps_family_export_contract(options, monkeypatch):
    from ir_training.train import gpu_profile
    monkeypatch.setattr(gpu_profile, "detect_cuda_devices", lambda: pytest.fail("planning must not probe GPUs"))
    before = options.sft_config.read_bytes()
    result = workflow.run_pipeline(options)
    plan = result["plan"]
    assert result["status"] == "plan_only"
    assert not options.output_dir.exists()
    assert options.sft_config.read_bytes() == before
    assert plan["workflow"] == "qat_lora_grpo_v1"
    assert plan["selection"]["cohort"] == "golden32"
    assert not plan["selection"]["golden35_used"] and not plan["selection"]["bixby50_used"]
    assert plan["stages"][-2:] == ["export", "native_quality"]
    assert ("no_op_export" in plan["stages"]) == (options.family == "e2b")
    assert plan["mtp"]["training"] is False and plan["mtp"]["inference"] is False
    assert plan["mtp"]["official_section_preserved"] == (options.family == "e2b")
    assert "evaluate_official_mobile_native.py" in result["native_quality_command"][1]
    assert "--execute" not in result["native_quality_command"]
    assert plan["options"]["max_seq_length"] == 6144
    assert plan["options"]["max_input_tokens"] == 5120
    assert len(plan["source_adapter_files"]) == 3


@pytest.mark.parametrize("change,error", [
    ({"max_steps": 19}, "20-update"),
    ({"num_generations": 1}, "num-generations"),
    ({"effective_batch": 31}, "divisible"),
    ({"learning_rate": float("nan")}, "Learning rate"),
    ({"max_seq_length": 0}, "Token budgets"),
    ({"max_input_tokens": 32768}, "context"),
    ({"max_new_tokens": 4097}, "4096|context"),
    ({"stage_timeout_seconds": 0}, "Timeouts"),
])
def test_invalid_training_budgets_are_rejected_before_output(options, change, error):
    with pytest.raises(ValueError, match=error):
        workflow.build_plan(replace(options, **change))
    assert not options.output_dir.exists()


def test_input_modes_and_output_disjointness_are_enforced(options, monkeypatch):
    with pytest.raises(ValueError, match="exactly one"):
        workflow.build_plan(replace(options, prepared_input_dir=options.input_dir))
    with pytest.raises(ValueError, match="exactly one"):
        workflow.build_plan(replace(options, input_dir=None))
    for output in (options.model_dir / "run", options.sft_checkpoint / "run", options.input_dir / "run", options.model_dir.parent):
        with pytest.raises(ValueError, match="disjoint"):
            workflow.build_plan(replace(options, output_dir=output))
    import ir_training.data.prepared_input as frozen
    monkeypatch.setattr(frozen, "prepared_input_files", lambda path: [path / "train.jsonl", path / "val.jsonl"])
    plan = workflow.build_plan(replace(options, input_dir=None, prepared_input_dir=options.input_dir))
    assert plan["preparation"]["options"]["input_dir"] is None
    assert plan["preparation"]["options"]["prepared_input_dir"] == str(options.input_dir)


def test_adapter_bound_sft_config_and_original_base_are_required(options):
    config = load_yaml(options.sft_config)
    config["training"]["learning_rate"] = 4e-5
    write(options.sft_config, config)
    with pytest.raises(ValueError, match="checkpoint-bound"):
        workflow.build_plan(options)
    config["model"]["model_source"] = str(options.model_dir.parent / "different-base")
    write(options.sft_config, config)
    with pytest.raises(ValueError, match="original local base"):
        workflow.build_plan(options)


def test_original_config_requires_resolved_absolute_base(options):
    config = load_yaml(options.sft_config)
    config["model"]["model_source"] = "models/relative-original-base"
    write(options.sft_config, config)
    with pytest.raises(ValueError, match="absolute local model_source"):
        workflow.build_plan(options)


@pytest.mark.parametrize("valid_lineage", [False, True])
def test_configure_checks_source_lineage_then_model_free_dependencies(options, monkeypatch, valid_lineage):
    import ir_training.train.gpu_profile as gpu
    import ir_training.train.qat_grpo_contract as contract

    plan = workflow.build_plan(options)
    calls = []
    monkeypatch.setattr(workflow.official_mobile, "_scripts", lambda: None)
    monkeypatch.setitem(sys.modules, "prepare_review_training", SimpleNamespace(verify_prepared=lambda *args, **kwargs: prepared()))
    monkeypatch.setattr(gpu, "detect_cuda_devices", lambda: {"fixture": True})
    monkeypatch.setattr(gpu, "build_gpu_profile", lambda *args, **kwargs: profile())
    monkeypatch.setattr(contract, "verify_sft_adapter_lineage", lambda config: {"verified": valid_lineage})
    monkeypatch.setattr(workflow, "_run", lambda plan, command, stage, **kwargs: calls.append((command, stage, kwargs)))
    if not valid_lineage:
        with pytest.raises(ValueError, match="provenance failed"):
            workflow.run_stage(plan, "configure")
        assert calls == []
        assert not Path(plan["paths"]["config"]).exists()
    else:
        workflow.run_stage(plan, "configure")
        assert len(calls) == 1
        command, stage, kwargs = calls[0]
        assert stage == "dependency_preflight"
        assert "--dependency-preflight-only" in command
        assert command[command.index("--config") + 1] == plan["paths"]["config"]
        assert kwargs["gpu"] is True


def test_new_config_keeps_qat_lora_and_separates_train_eval_budgets(options):
    plan = workflow.build_plan(options)
    before = options.sft_config.read_bytes()
    original = load_yaml(options.sft_config)
    config = workflow.training_config(plan, profile(), prepared())
    assert options.sft_config.read_bytes() == before
    assert config["lora"] == original["lora"]
    assert config["qat"] == original["qat"]
    assert config["model"]["tokenizer_source"] == str(options.model_dir)
    assert config["training"]["method"] == "qat_lora_grpo"
    assert config["training"]["learning_rate"] == 1e-6
    assert config["training"]["max_seq_length"] == 6144
    assert config["grpo"]["max_prompt_length"] == 5120
    assert config["grpo"]["quality_control"]["enabled"] is True
    assert config["grpo"]["validation_max_rows"] == 32
    assert config["golden_eval"]["max_input_tokens"] == 5120
    assert config["grpo"]["max_completion_length"] == 2048
    assert config["grpo"]["num_generations"] == 4
    assert config["training"]["expected_effective_batch_size"] == 32
    assert config["runtime"]["distributed"] == "ddp"
    assert config["training"]["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert config["training"]["ddp_broadcast_buffers"] is False
    assert "deepspeed" not in config["training"] and "resume_from_checkpoint" not in config["training"]
    assert config["grpo"]["beta"] == 0
    assert config["grpo"]["reward_policy"] == QAT_GRPO_REWARD_VERSION
    assert "qat_mtp" not in config
    assert config["golden_eval"]["split"] == "golden32"
    assert set(config["final_evaluation_datasets"]) == {"golden35", "bixby50"}
    with pytest.raises(ValueError, match="BF16"):
        workflow.training_config(plan, {**profile(), "dtype": "float16"}, prepared())


def test_shared_family_export_commands_are_distinct_and_bind_actual_artifact(options):
    plan = workflow.build_plan(options)
    if options.family == "270m":
        config = workflow.training_config(plan, profile(), prepared())
        write(Path(plan["paths"]["config"]), config)
        deployment = deepcopy(load_yaml(ROOT / "configs/pipelines/gemma3_270m_qat_litertlm.yaml"))
        pipeline = deployment["pipeline"]
        pipeline.update(training_config=plan["paths"]["config"], output_dir=str(options.output_dir))
        pipeline["source"].update(merged_model_dir=plan["paths"]["merged"], official_litertlm=str(options.official_litertlm))
        pipeline["export"]["artifact"] = plan["paths"]["litertlm"]
        pipeline["exact_topology"]["output_dir"] = plan["paths"]["export_dir"]
        write(Path(plan["paths"]["deployment_config"]), deployment)
    command = workflow.export_command(plan)
    assert command[0] == str(options.exporter_python)
    assert command[command.index("--checkpoint") + 1] == plan["paths"]["merged"]
    assert command[command.index("--training-config") + 1] == plan["paths"]["config"]
    assert "--execute" in command
    if options.family == "e2b":
        assert command[1].endswith("build_gemma4_retained_scale_litertlm.py")
        assert "--qat-compatible-weights" in command
        assert command[command.index("--adapter-checkpoint") + 1] == plan["paths"]["best_checkpoint"]
        assert command[command.index("--output-litertlm") + 1] == plan["paths"]["litertlm"]
    else:
        assert command[1].endswith("build_checkpoint_official_topology.py")
        assert command[command.index("--family") + 1] == "gemma3_270m"
        assert command[command.index("--package-output") + 1] == plan["paths"]["litertlm"]
        assert "--qat-compatible-weights" not in command


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_validation_completion_batch_contains_full_grpo_groups(options, world_size):
    plan = workflow.build_plan(options)
    gpu = {**profile(), "world_size": world_size, "gradient_accumulation_steps": 32 // world_size}
    config = workflow.training_config(plan, gpu, prepared())
    assert config["training"]["per_device_eval_batch_size"] * world_size % config["grpo"]["num_generations"] == 0


def test_export_report_must_certify_actual_package_bytes(options):
    plan = workflow.build_plan(options)
    artifact = write(Path(plan["paths"]["litertlm"]))
    digest = workflow.sha256(artifact)
    report = ({"passed": True, "executed": True, "output_sha256": digest,
               "mode": "retained_scale_qat_compatible_weights_v1"} if options.family == "e2b" else
              {"final_artifact_gate_pass": True, "gates": {"fixture_gate": True},
               "package_boundary": {"output_sha256": digest}})
    write(Path(plan["paths"]["export_report"]), report)
    assert workflow.verify_export_report(plan) == report
    artifact.write_bytes(b"changed after export")
    with pytest.raises(ValueError, match="actual package bytes"):
        workflow.verify_export_report(plan)


@pytest.mark.parametrize("world_size", [1, 2])
def test_training_stage_uses_qat_runner_and_validates_selected_adapter(options, monkeypatch, world_size):
    plan = workflow.build_plan(options)
    config = write(Path(plan["paths"]["config"]), {"synthetic": True})
    write(Path(plan["paths"]["gpu_profile"]), {"world_size": world_size})
    calls, no_op = [], []
    monkeypatch.setattr(workflow.official_mobile, "_require_no_op_export", lambda p: no_op.append(p))

    def run(p, command, stage, **kwargs):
        calls.append((command, stage, kwargs))
        write(Path(p["paths"]["best_checkpoint"]) / "training_metadata.json", {
            "checkpoint_role": "best_golden", "training_config_sha256": workflow.sha256(config),
            "best_golden_eval": {"metric": workflow.official_mobile.SELECTOR},
            "training": {"method": "qat_lora_grpo"},
        })

    quality_calls = []
    def quality_gate(output, checkpoint, config_path):
        quality_calls.append((output, checkpoint, config_path))
        return [write(output / "grpo_quality_gate.json", {"fixture": True})]
    monkeypatch.setattr(workflow, "verify_quality_gate", quality_gate)
    monkeypatch.setattr(workflow, "_run", run)
    workflow.run_stage(plan, "training")
    assert quality_calls == [(options.output_dir / "training", Path(plan["paths"]["best_checkpoint"]), config)]
    command, stage, kwargs = calls[0]
    assert stage == "training" and kwargs["gpu"] is True
    assert ("torch.distributed.run" in command) == (world_size > 1)
    assert any(value.endswith("train_qat_grpo.py") for value in command)
    assert not any(value.endswith("train_sft.py") for value in command)
    assert bool(no_op) == (options.family == "e2b")


def test_cpu_export_process_keeps_offline_contract_and_clears_gpu_overrides(options, monkeypatch):
    plan = workflow.build_plan(options)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("A2UI_CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("A2UI_EXCLUDE_CUDA_DEVICES", "1")
    calls = []
    monkeypatch.setattr(workflow, "run_bounded_command", lambda *args, **kwargs: calls.append((args, kwargs)))
    workflow._run(plan, ["fixture-exporter"], "export", cpu=True)
    args, kwargs = calls[0]
    env = args[2]
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "A2UI_CUDA_VISIBLE_DEVICES" not in env and "A2UI_EXCLUDE_CUDA_DEVICES" not in env
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert kwargs["timeout_seconds"] == options.stage_timeout_seconds


def test_merge_uses_original_local_base_and_seed_manifest_not_hub_id(options, monkeypatch):
    import ir_training.export.merge_lora as merge

    plan = workflow.build_plan(options)
    config = workflow.training_config(plan, profile(), prepared())
    manifest = str(options.model_dir / "mobile_training_seed_manifest.json") if options.family == "e2b" else None
    if manifest:
        config["model"]["mobile_training_seed_manifest"] = manifest
    write(Path(plan["paths"]["config"]), config)
    calls = []

    def merge_fixture(**kwargs):
        calls.append(kwargs)
        write(Path(kwargs["output_dir"]) / "fixture.json", {"merged": False})

    monkeypatch.setattr(merge, "merge_lora_adapter", merge_fixture)
    workflow.run_stage(plan, "merge")
    assert calls[0]["base_model_source"] == str(options.model_dir)
    assert calls[0]["base_model_id"] == config["model"]["model_id"]
    assert calls[0]["base_model_source"] != calls[0]["base_model_id"]
    assert calls[0]["mobile_training_seed_manifest"] == manifest
    assert calls[0]["training_config_path"] == plan["paths"]["config"]
    assert calls[0]["adapter_dir"] == plan["paths"]["best_checkpoint"]


def fixture_runner(*, empty_receipt=False, stale_receipt=False):
    def runner(command, *args, **kwargs):
        plan_path = Path(command[command.index("--plan-file") + 1])
        stage = command[command.index("--worker-stage") + 1]
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        output = Path(plan["options"]["output_dir"])
        evidence = write(output / "fixture-stage-evidence" / f"{stage}.json", {"mock_stage": stage})
        write(output / "stage_receipts" / f"{stage}.json", {
            "stage": stage, "plan_sha256": "bad" if stale_receipt else workflow.sha256(plan_path),
            "files": {} if empty_receipt else {str(evidence): workflow.sha256(evidence)},
        })
    return runner


def test_deferred_native_status_never_claims_device_success(options):
    result = workflow.run_pipeline(options, execute=True, command_runner=fixture_runner())
    assert result["status"] == "awaiting_native_evaluation"
    assert "export" in result["completed"]
    assert "native_quality" not in result["completed"]
    results = json.loads((options.output_dir / "results.json").read_text(encoding="utf-8"))
    assert results["export_passed"] and not results["native_test_passed"]
    handoff = json.loads((options.output_dir / "native_quality_command.json").read_text(encoding="utf-8"))
    assert handoff["command"][-1] == "--execute"
    assert "<ANDROID_SERIAL>" in handoff["command"]
    with pytest.raises(FileExistsError):
        workflow.run_pipeline(options, execute=True, command_runner=fixture_runner())


def test_same_host_native_requires_explicit_device_and_finishes_last(options):
    with pytest.raises(ValueError, match="serial"):
        workflow.run_pipeline(replace(options, defer_native_eval=False), execute=True, command_runner=fixture_runner())
    assert not options.output_dir.exists()
    result = workflow.run_pipeline(replace(options, defer_native_eval=False, serial="fixture-device"),
                                   execute=True, command_runner=fixture_runner())
    assert result["status"] == "complete"
    assert list(result["completed"])[-1] == "native_quality"


@pytest.mark.parametrize("kind", ["empty", "stale"])
def test_receipts_require_immutable_plan_and_nonempty_evidence(options, kind):
    with pytest.raises(ValueError, match="receipt|evidence|binding"):
        workflow.run_pipeline(options, execute=True,
                              command_runner=fixture_runner(empty_receipt=kind == "empty", stale_receipt=kind == "stale"))
    results = json.loads((options.output_dir / "results.json").read_text(encoding="utf-8"))
    assert results["status"] == "failed"
    assert not results["export_passed"] and not results["native_test_passed"]


def test_worker_rejects_skipped_stage_before_any_work(options, monkeypatch):
    plan = workflow.build_plan(options)
    plan_path = write(options.output_dir / workflow.PLAN_NAME, plan)
    write(options.output_dir / workflow.MANIFEST_NAME, {
        "plan": plan, "status": "running", "active_stage": "training", "completed": {},
    })
    monkeypatch.setattr(workflow, "run_stage", lambda *args: pytest.fail("must not run a skipped stage"))
    with pytest.raises(ValueError, match="prerequisites"):
        workflow.worker(plan_path, "training")
