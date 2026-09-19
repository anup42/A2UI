"""CPU-only contract tests for the official retained-mobile pipeline.

The fixtures are intentionally tiny and no test imports a model, starts CUDA,
trains, merges weights, exports LiteRT-LM, or invokes adb.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.pipeline import official_mobile as workflow
from ir_training.pipeline.official_mobile import OfficialMobileOptions
from ir_training.common.jsonl import write_jsonl
from ir_training.data.chat_templates import build_messages, build_prompt
from ir_training.train.gpu_profile import build_gpu_profile


class FixtureTokenizer:
    name_or_path = "official-mobile-fixture-tokenizer"
    chat_template = "fixture-role-prefix"
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def get_vocab(self):
        return {"fixture": 0}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        rendered = "".join(f"{item['role']}:\n{item['content']}\n" for item in messages)
        return rendered + ("assistant:\n" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": list(range(len(text.split())))}


def _row(identity: str) -> dict:
    source = f"Independent official mobile fixture response for {identity}"
    completion = f'<a2ui>\nroot=Text("{identity}")\n</a2ui>'
    return {
        "id": identity,
        "source_id": identity,
        "response_id": f"response-{identity}",
        "response_text": source,
        "completion": completion,
        "metadata": {"query_id": identity},
        "messages": build_messages(
            "Historical short fixture prompt",
            source,
            completion,
            target_format="a2ui_express_v1",
        ),
        "prompt": build_prompt(
            "Historical short fixture prompt",
            source,
            target_format="a2ui_express_v1",
        ),
        "intent_bucket": "fixture",
    }


def _write(path: Path, value: bytes = b"fixture; never loaded") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


@pytest.fixture
def options(tmp_path: Path) -> OfficialMobileOptions:
    model = tmp_path / "mobile-seed"
    inputs = tmp_path / "inputs"
    model.mkdir()
    inputs.mkdir()
    for name in ("config.json", "tokenizer_config.json", "mobile_training_seed_manifest.json",
                 "mobile_qparams.json"):
        (model / name).write_text("{}", encoding="utf-8")
    for name in ("model.safetensors", "mobile_qparams.safetensors"):
        _write(model / name)
    for name in ("train.jsonl", "val.jsonl"):
        (inputs / name).write_text("{}\n", encoding="utf-8")
    return OfficialMobileOptions(
        model_dir=model,
        input_dir=inputs,
        source_safetensors=_write(tmp_path / "official-mobile.safetensors"),
        official_litertlm=_write(tmp_path / "official-mobile.litertlm"),
        output_dir=tmp_path / "official-mobile-run",
        exporter_python=Path(sys.executable),
        steps=20,
        eval_steps=5,
        golden_every_steps=10,
        progress_seconds=0.01,
        stage_timeout_seconds=30,
        generation_timeout_seconds=30,
    )


def _plan(options: OfficialMobileOptions) -> dict:
    plan = workflow.build_plan(options)
    assert not options.output_dir.exists()
    return plan


def _h100_inventory(count: int) -> dict:
    devices = [
        {
            "visible_index": index,
            "launch_identifier": str(index),
            "uuid": f"GPU-{index}",
            "name": "NVIDIA H100 80GB HBM3",
            "total_memory_bytes": 80 * 1024**3,
            "compute_capability": [9, 0],
        }
        for index in range(count)
    ]
    return {
        "version": 1,
        "inherited_cuda_visible_devices": None,
        "visible_gpu_count": count,
        "devices": devices,
    }


def test_plan_only_requires_official_mobile_artifacts_and_has_no_writes(options):
    plan = _plan(options)

    assert plan["workflow"] == workflow.WORKFLOW
    assert plan["format"] == {
        "name": "official-mobile wNa8o8",
        "target_weight_bits": [2, 4, 8],
        "activation_bits": 8,
        "official_sha256": workflow.OFFICIAL_LITERTLM_SHA256,
        "layout_policy": "change only 205 projection code buffers; retain graph, scales and 72 frozen constants",
    }
    assert plan["selection"] == {
        "cohort": "golden32",
        "metric": workflow.SELECTOR,
        "golden35_used": False,
        "bixby50_used": False,
    }
    assert plan["mtp"] == {
        "training": False,
        "inference": False,
        "official_section_preserved_but_unused": True,
    }
    assert plan["stages"][-2:] == ["merge", "export"]
    assert plan["native_litert_golden_tests"] is False


@pytest.mark.parametrize(
    "missing",
    [
        "mobile_training_seed_manifest.json",
        "mobile_qparams.json",
        "mobile_qparams.safetensors",
        "source_safetensors",
        "official_litertlm",
    ],
)
def test_plan_rejects_each_missing_mobile_artifact(options, missing):
    path = (options.model_dir / missing) if "." in missing else getattr(options, missing)
    path.unlink()

    with pytest.raises(FileNotFoundError, match="Required mobile artifact missing"):
        workflow.build_plan(options)
    assert not options.output_dir.exists()


def test_dense_hf_directory_without_mobile_manifest_is_rejected(options):
    dense = options.model_dir.parent / "dense-hf-only"
    dense.mkdir()
    (dense / "config.json").write_text("{}", encoding="utf-8")
    (dense / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    _write(dense / "model.safetensors")

    with pytest.raises(FileNotFoundError, match="mobile_training_seed_manifest.json"):
        workflow.build_plan(replace(options, model_dir=dense))
    assert not options.output_dir.exists()


@pytest.mark.parametrize("location", ["inside_model", "inside_inputs", "contains_inputs"])
def test_output_cannot_overlap_input_trees(options, location):
    output = {
        "inside_model": options.model_dir / "run",
        "inside_inputs": options.input_dir / "run",
        "contains_inputs": options.input_dir.parent,
    }[location]
    with pytest.raises(ValueError, match="separate from model and input"):
        workflow.build_plan(replace(options, output_dir=output))
    assert not (output / "official_mobile_plan.json").exists()


def test_exporter_python_preserves_virtualenv_symlink(options, tmp_path):
    executable = tmp_path / "export-venv/bin/python"
    executable.parent.mkdir(parents=True)
    try:
        executable.symlink_to(sys.executable)
    except OSError:
        pytest.skip("Host does not allow executable symlinks")
    plan = workflow.build_plan(replace(options, exporter_python=executable))
    assert plan["options"]["exporter_python"] == os.path.abspath(executable)
    assert workflow._mobile_command(plan, "export")[0] == os.path.abspath(executable)


def test_exporter_executable_is_never_realpath_resolved(options, tmp_path, monkeypatch):
    """Platform-independent guard, including hosts where symlinks need admin rights."""
    executable = _write(tmp_path / "isolated-env/bin/python")
    resolve = Path.resolve

    def guarded_resolve(path, *args, **kwargs):
        if path == executable:
            pytest.fail("Resolving a venv executable would drop its environment")
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    plan = workflow.build_plan(replace(options, exporter_python=executable))
    assert workflow._mobile_command(plan, "export")[0] == os.path.abspath(executable)


def test_short_run_records_rounded_cadence_and_still_evaluates_at_end(options):
    plan = workflow.build_plan(replace(options, steps=300, eval_steps=500, golden_every_steps=1000))
    profile = build_gpu_profile(_h100_inventory(2), model="e2b", cpu_count=32)
    config = workflow.training_config(plan, profile, {"tokenizer": {}, "final_evaluation_datasets": {}})
    assert config["training"]["eval_steps"] == 300
    assert config["golden_eval"]["interval"] == 4
    assert config["golden_eval"]["requested_every_optimizer_steps"] == 1000
    assert config["golden_eval"]["resolved_every_optimizer_steps"] == 1200
    assert config["golden_eval"]["evaluate_at_end"] is True


def test_export_environment_roundtrip_without_loading_model():
    pytest.importorskip("flatbuffers")
    if importlib.util.find_spec("tflite") is None:
        pytest.importorskip("ai_edge_litert")
    pytest.importorskip("safetensors")
    result = workflow.probe_export_environment()
    assert result["passed"] is True
    assert result["schema_roundtrip"] is True
    assert result["python"] == sys.executable
    assert result["model_loaded"] is False
    assert result["full_official_model_compatibility_tested"] is False


@pytest.mark.parametrize("complete", [True, False])
def test_export_environment_checks_actual_target_schema_before_training(options, monkeypatch, complete):
    pytest.importorskip("flatbuffers")
    if importlib.util.find_spec("tflite") is None:
        pytest.importorskip("ai_edge_litert")
    pytest.importorskip("safetensors")
    workflow._scripts()
    import build_gemma4_retained_scale_litertlm as exporter

    def section(model_type, graph):
        return {"data_type_name": "TFLiteModel", "items": [{"key": "model_type", "value": model_type}],
                "graph": graph}

    package = {"sections": [section(exporter.TARGET_MODEL_TYPE, {
        "available": True, "execution_contract_complete": complete,
        "structural_sha256": "a" * 64, "quantization_layout_sha256": "b" * 64,
        "execution_contract_sha256": "c" * 64,
    }), section(exporter.MTP_MODEL_TYPE, {"available": False})]}
    monkeypatch.setattr(exporter, "inspect_litertlm", lambda path, **kwargs: package)
    if not complete:
        with pytest.raises(ValueError, match="cannot completely decode official"):
            workflow.probe_export_environment(options.official_litertlm)
    else:
        result = workflow.probe_export_environment(options.official_litertlm)
        assert result["official_graphs"][0]["structural_sha256"] == "a" * 64
        assert result["model_loaded"] is False


def test_default_context_generation_and_golden_cadence_are_explicit(options):
    defaults = replace(
        options,
        steps=None,
        eval_steps=500,
        golden_every_steps=1000,
        max_seq_length=4096,
        max_new_tokens=2048,
    )
    plan = _plan(defaults)
    profile = build_gpu_profile(_h100_inventory(2), model="e2b", cpu_count=32)
    config = workflow.training_config(
        plan,
        profile,
        {"tokenizer": {}, "final_evaluation_datasets": {}},
    )

    assert plan["options"]["max_seq_length"] == 4096
    assert plan["options"]["max_new_tokens"] == 2048
    assert plan["options"]["golden_every_steps"] == 1000
    assert config["training"]["eval_steps"] == 500
    assert config["golden_eval"]["interval"] == 2
    assert config["golden_eval"]["requested_every_optimizer_steps"] == 1000


@pytest.mark.parametrize("gpu_count,accumulation", [(2, 16), (4, 8), (8, 4)])
def test_generated_h100_config_keeps_batch32_microbatch1_and_retained_qat(
    options, gpu_count, accumulation
):
    plan = _plan(options)
    profile = build_gpu_profile(
        _h100_inventory(gpu_count), model="e2b", cpu_count=64
    )
    report = {
        "tokenizer": {"chat_template_kwargs": {"enable_thinking": True}},
        "final_evaluation_datasets": {
            "golden35": {"selection_role": "final_only_holdout"},
            "bixby50": {"selection_role": "final_only_holdout"},
        },
    }

    config = workflow.training_config(plan, profile, report)

    assert profile["world_size"] == gpu_count
    assert profile["effective_batch_size"] == 32
    assert profile["microbatch"] == 1
    assert profile["gradient_accumulation_steps"] == accumulation
    assert config["runtime"]["world_size"] == gpu_count
    assert config["training"]["per_device_train_batch_size"] == 1
    assert config["training"]["expected_effective_batch_size"] == 32
    assert config["training"]["gradient_accumulation_steps"] == accumulation
    assert config["run"]["prepared_manifest_required"] is True
    assert config["qat"]["enabled"] is True
    assert config["qat"]["scale_mode"] == "retained_mobile"
    assert config["qat"]["fixed_scale_required"] is True
    assert config["qat"]["fixed_activation_scale_required"] is True
    assert config["qat"]["effective_lora_only"] is True
    assert config["qat"]["expected_effective_lora_modules"] == 205
    assert "qat_mtp" not in config


def test_real_cpu_prepare_and_portable_configure_bind_all_evaluation_contracts(
    options, monkeypatch
):
    options = replace(options, prepare_workers=1, progress_seconds=10)
    write_jsonl(options.input_dir / "train.jsonl", [_row("train-a"), _row("train-b")])
    write_jsonl(options.input_dir / "val.jsonl", [_row("val-a"), _row("val-b")])
    plan = workflow.build_plan(options)
    options.output_dir.mkdir()

    preparation = workflow.prepare_data(
        plan["preparation"],
        tokenizer_loader=lambda *_: FixtureTokenizer(),
    )
    assert preparation["prepared_counts"]["train"]["accepted_rows"] == 2
    assert preparation["prepared_counts"]["val"]["accepted_rows"] == 2

    import ir_training.train.gpu_profile as gpu_profile_module

    workflow._scripts()
    import run_gemma4_mobile_qat as mobile_launcher

    monkeypatch.setattr(
        gpu_profile_module,
        "detect_cuda_devices",
        lambda: _h100_inventory(2),
    )
    monkeypatch.setattr(
        mobile_launcher,
        "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        workflow.sha256(options.source_safetensors),
    )

    configured_files = workflow._configure(plan)
    assert Path(plan["paths"]["config"]) in configured_files
    assert Path(plan["paths"]["launch_plan"]) in configured_files
    launcher, launch, launch_paths = workflow._launch(plan)
    assert launcher is mobile_launcher
    assert launch["checks"]["contract_ok"] is True
    assert launch["checks"]["issues"] == []

    from launch_review_training import verify_launch_binding
    from ir_training.eval.prepared_contract import verify_evaluation_prepared_contract

    verify_launch_binding(launch_paths["resolved_config"])
    for cohort, (_, count, _) in workflow.GOLDENS.items():
        result = verify_evaluation_prepared_contract(
            launch_paths["resolved_config"],
            options.output_dir / f"prepared/{cohort}.jsonl",
            required_rows=count,
            max_input_tokens=options.max_seq_length,
        )
        assert result["required_rows"] == count
        assert Path(result["split_path"]) == (options.output_dir / f"prepared/{cohort}.jsonl").resolve()


def test_golden32_alone_selects_and_all_holdout_commands_require_gpu_fake_qat(options):
    plan = _plan(options)
    profile = build_gpu_profile(_h100_inventory(2), model="e2b", cpu_count=32)
    final = {
        "golden35": {"selection_role": "final_only_holdout"},
        "bixby50": {"selection_role": "final_only_holdout"},
    }
    config = workflow.training_config(
        plan,
        profile,
        {"tokenizer": {}, "final_evaluation_datasets": final},
    )

    golden = config["golden_eval"]
    assert golden["split"] == "golden32"
    assert golden["required_rows"] == golden["max_rows"] == 32
    assert golden["require_unique_rows"] is True
    assert golden["metric_for_best_model"] == workflow.SELECTOR
    assert golden["best_checkpoint_dir"] == plan["paths"]["best_checkpoint"]
    assert config["final_evaluation_datasets"] == final
    for cohort, (_, count, _) in workflow.GOLDENS.items():
        command = workflow.evaluation_command(plan, cohort)
        assert command[command.index("--qat-mode") + 1] == "on"
        assert "--require-gpu" in command
        assert command[command.index("--devices") + 1] == "auto"
        assert command[command.index("--required-rows") + 1] == str(count)
        assert command[command.index("--checkpoint") + 1] == plan["paths"]["best_checkpoint"]
    assert plan["selection"]["golden35_used"] is False
    assert plan["selection"]["bixby50_used"] is False


def test_export_is_strict_retained_scale_and_preserves_unused_official_mtp(options):
    plan = _plan(options)
    config = workflow.mobile_export_config(plan)["pipeline"]
    command = workflow._mobile_command(plan, "export")

    assert config["public_export"]["enabled"] is False
    assert config["retained_scale_export"]["enabled"] is True
    assert config["retained_scale_export"]["mode"] == "retained_scale_code_only_v1"
    assert config["retained_scale_export"]["official_artifact_sha256"] == workflow.OFFICIAL_LITERTLM_SHA256
    assert config["source"]["base_litertlm"] == str(options.official_litertlm.resolve())
    assert config["source"]["output_litertlm"] == plan["paths"]["litertlm"]
    assert config["mtp"]["enabled"] is False
    assert config["mtp"]["train_assistant"] is False
    assert config["mtp"]["weight_source"] == "official"
    assert config["mtp"]["preserve_official_section"] is True
    assert command[0] == os.path.abspath(options.exporter_python)
    assert command[-1] == "--execute-retained-scale-export"
    assert "--execute-public-export" not in command
    assert "--compose" not in command


def test_optional_android_benchmark_is_target_only_and_does_not_enable_mtp(options):
    options = replace(options, benchmark_android=True, serial="SERIAL-123")
    plan = _plan(options)
    deployment = workflow.mobile_export_config(plan)["pipeline"]
    command = workflow.benchmark_command(plan)

    assert plan["stages"][-1] == "android_benchmark"
    assert deployment["mtp"]["enabled"] is False
    assert command[command.index("--official") + 1] == str(options.official_litertlm.resolve())
    assert command[command.index("--candidate") + 1] == plan["paths"]["litertlm"]
    assert command[command.index("--serial") + 1] == "SERIAL-123"
    assert "mtp" not in " ".join(command).lower()


def test_assets_bind_seed_qparams_and_both_official_inputs(options, monkeypatch):
    plan = _plan(options)
    seen = {}

    def verify_seed(model, *, base, require_materialized):
        seen["seed"] = (model, base, require_materialized)
        return {"verified": True, "contract": "fixture-seed"}

    def verify_qparams(path, *, base):
        seen["qparams"] = (Path(path), base)
        return {"verified": True, "contract": "fixture-qparams"}

    import ir_training.qat.mobile_qparams as qparams_module
    import ir_training.qat.mobile_training_seed as seed_module

    monkeypatch.setattr(seed_module, "verify_configured_mobile_training_seed", verify_seed)
    monkeypatch.setattr(qparams_module, "verify_mobile_qparams_contract", verify_qparams)
    monkeypatch.setattr(workflow, "run_bounded_command", lambda *args, **kwargs: None)

    def verify_bindings(records):
        seen["official_inputs"] = records

    monkeypatch.setattr(workflow, "_verify_bindings", verify_bindings)
    files = workflow._assets(plan)

    assert seen["seed"][0]["mobile_training_seed_manifest"].endswith("mobile_training_seed_manifest.json")
    assert seen["seed"][2] is True
    assert seen["qparams"][0] == options.model_dir / "mobile_qparams.json"
    assert seen["official_inputs"] == {
        str(options.source_safetensors.resolve()): workflow.OFFICIAL_MOBILE_SAFETENSORS_SHA256,
        str(options.official_litertlm.resolve()): workflow.OFFICIAL_LITERTLM_SHA256,
    }
    assert options.model_dir / "mobile_training_seed_manifest.json" in files
    assert options.model_dir / "mobile_qparams.json" in files
    report = json.loads((options.output_dir / "mobile_assets_verified.json").read_text(encoding="utf-8"))
    assert report["seed"]["verified"] is True
    assert report["qparams"]["verified"] is True
    assert report["mtp_training"] is report["mtp_inference"] is False


def test_execute_refuses_existing_output_without_running_a_stage(options):
    options.output_dir.mkdir()
    marker = options.output_dir / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")
    calls = []

    with pytest.raises(FileExistsError):
        workflow.run_pipeline(options, execute=True, command_runner=lambda *a, **k: calls.append(a))

    assert calls == []
    assert marker.read_text(encoding="utf-8") == "untouched"
    assert not (options.output_dir / "official_mobile_manifest.json").exists()


def test_bounded_stage_failure_stops_before_export_and_always_writes_summary(options):
    calls = []

    def runner(command, log_path, environment, **kwargs):
        stage = command[command.index("--worker-stage") + 1]
        calls.append((stage, kwargs, environment))
        if stage == "training":
            raise RuntimeError("synthetic bounded training failure")
        evidence = options.output_dir / "evidence" / f"{stage}.json"
        workflow._write(evidence, {"stage": stage})
        plan_path = options.output_dir / "official_mobile_plan.json"
        workflow._write(
            options.output_dir / f"stage_receipts/{stage}.json",
            {
                "stage": stage,
                "plan_sha256": workflow.sha256(plan_path),
                "files": {str(evidence.resolve()): workflow.sha256(evidence)},
            },
        )

    with pytest.raises(RuntimeError, match="synthetic bounded training failure"):
        workflow.run_pipeline(options, execute=True, command_runner=runner)

    assert [stage for stage, _, _ in calls] == ["assets", "prepare", "configure", "preflight", "training"]
    assert all(call[1]["timeout_seconds"] == options.stage_timeout_seconds for call in calls)
    assert all(call[1]["progress_seconds"] == options.progress_seconds for call in calls)
    assert "export" not in {stage for stage, _, _ in calls}
    manifest = json.loads((options.output_dir / "official_mobile_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["active_stage"] == "training"
    assert "synthetic bounded training failure" in manifest["error"]
    assert list(manifest["completed"]) == ["assets", "prepare", "configure", "preflight"]
    summary = (options.output_dir / "results.md").read_text(encoding="utf-8")
    assert "Run status: failed" in summary
    assert "LiteRT-LM export: not completed" in summary
    assert "Failure: synthetic bounded training failure" in summary


def test_parent_and_worker_complete_in_order_and_report_all_three_cohorts(options, monkeypatch):
    """Exercise real orchestration/receipts with explicitly synthetic stage outputs."""
    seen = []

    def run_stage(plan, stage):
        seen.append(stage)
        evidence = options.output_dir / "synthetic" / f"{stage}.json"
        workflow._write(evidence, {"stage": stage})
        files = [evidence]
        if stage.startswith("best_"):
            cohort = stage.removeprefix("best_")
            result = options.output_dir / f"evaluations/{stage}/evaluation_result.json"
            workflow._write(result, {
                "row_count": workflow.GOLDENS[cohort][1],
                "aggregate": {"generation_reward_v5_4_avg": 73.5,
                              "schema_valid_strict_rate": 1.0, workflow.SELECTOR: 72.5},
            })
            files.append(result)
        return files

    monkeypatch.setattr(workflow, "run_stage", run_stage)

    def runner(command, log_path, environment, **kwargs):
        workflow.worker(Path(command[command.index("--plan-file") + 1]),
                        command[command.index("--worker-stage") + 1])

    state = workflow.run_pipeline(options, execute=True, command_runner=runner)
    assert state["status"] == "complete"
    assert state["active_stage"] is None
    assert seen == state["plan"]["stages"] == list(state["completed"])
    assert set(state["results"]) == {"golden32", "golden35", "bixby50"}
    assert seen.index("best_bixby50") < seen.index("merge") < seen.index("export")
    summary = (options.output_dir / "results.md").read_text(encoding="utf-8")
    for cohort in workflow.GOLDENS:
        assert cohort in summary
    assert "32/32" in summary and "35/35" in summary and "50/50" in summary
    assert "Device speed parity: not measured" in summary
    assert "no reference IR" in summary


def test_worker_rechecks_evaluated_checkpoint_bytes_before_export(options, monkeypatch):
    plan = workflow.build_plan(options)
    plan_path = options.output_dir / "official_mobile_plan.json"
    workflow._write(plan_path, plan)
    evidence = options.output_dir / "adapter.safetensors"
    _write(evidence, b"original adapter")
    bindings = {str(evidence): workflow.sha256(evidence)}
    prior = plan["stages"][:plan["stages"].index("export")]
    completed = {name: {"plan_sha256": workflow.sha256(plan_path), "files": bindings} for name in prior}
    workflow._write(options.output_dir / "official_mobile_manifest.json", {
        "plan": plan, "status": "running", "active_stage": "export", "completed": completed,
    })
    _write(evidence, b"changed adapter")
    monkeypatch.setattr(workflow, "run_stage", lambda *_: pytest.fail("must not export changed checkpoint"))
    with pytest.raises(ValueError, match="Bound artifact changed"):
        workflow.worker(plan_path, "export")


def test_worker_cannot_skip_incomplete_prerequisite_stages(options, monkeypatch):
    plan = workflow.build_plan(options)
    options.output_dir.mkdir()
    plan_path = options.output_dir / "official_mobile_plan.json"
    workflow._write(plan_path, plan)
    workflow._write(
        options.output_dir / "official_mobile_manifest.json",
        {
            "workflow": workflow.WORKFLOW,
            "plan": plan,
            "status": "running",
            "active_stage": "configure",
            "completed": {},
            "results": {},
        },
    )
    monkeypatch.setattr(workflow, "run_stage", lambda *_: pytest.fail("skipped worker must not run"))

    with pytest.raises(ValueError, match="prerequisites are incomplete or out of order"):
        workflow.worker(plan_path, "configure")

    assert not (options.output_dir / "stage_receipts/configure.json").exists()


@pytest.mark.parametrize(
    "metadata,error",
    [
        ({"checkpoint_role": "last", "best_golden_eval": {"metric": workflow.SELECTOR}},
         "Golden32-selected best checkpoint"),
        ({"checkpoint_role": "best_golden", "best_golden_eval": {"metric": workflow.SELECTOR},
          "training_config_sha256": "wrong-config"},
         "different training config"),
    ],
)
def test_direct_training_stage_rejects_bad_best_checkpoint_metadata(
    options, monkeypatch, metadata, error
):
    plan = workflow.build_plan(options)
    output = options.output_dir
    config = Path(plan["paths"]["config"])
    config.parent.mkdir(parents=True)
    config.write_text("fixture: true\n", encoding="utf-8")
    preflight = config.parent / "preflight_report.json"
    workflow._write(preflight, {"all_passed": True})
    best = Path(plan["paths"]["best_checkpoint"])
    best.mkdir(parents=True)
    workflow._write(best / "training_metadata.json", metadata)
    _write(best / "adapter_model.safetensors")
    launch_paths = {
        "resolved_config": config,
        "preflight_report": preflight,
        "training_log": output / "logs/training-worker.log",
    }
    launch = {"training_command": ["never-run-training"], "host_gpu_profile": {}}
    mobile = SimpleNamespace(_verify_bound_artifacts=lambda _: [])

    monkeypatch.setattr(workflow, "_launch", lambda _: (mobile, launch, launch_paths))
    monkeypatch.setattr(workflow, "_environment", lambda *args, **kwargs: {})
    monkeypatch.setattr(workflow, "run_bounded_command", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "launch_review_training",
        SimpleNamespace(verify_launch_binding=lambda _: None),
    )

    with pytest.raises(ValueError, match=error):
        workflow.run_stage(plan, "training")


def test_cli_plan_mode_forwards_options_without_execution(options, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "fixture_run_official_mobile_pipeline",
        ROOT / "scripts/run_official_mobile_pipeline.py",
    )
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    captured = {}

    def fake_run_pipeline(actual_options, *, execute):
        captured.update(options=actual_options, execute=execute)
        return {"workflow": workflow.WORKFLOW, "status": "plan-only-fixture"}

    monkeypatch.setattr(script, "run_pipeline", fake_run_pipeline)
    args = [
        "--model-dir", str(options.model_dir),
        "--input-dir", str(options.input_dir),
        "--source-safetensors", str(options.source_safetensors),
        "--official-litertlm", str(options.official_litertlm),
        "--output-dir", str(options.output_dir),
        "--exporter-python", str(options.exporter_python),
    ]

    assert script.main(args) == 0
    assert captured["execute"] is False
    assert captured["options"].output_dir == options.output_dir
    output = capsys.readouterr().out
    assert '"status": "plan-only-fixture"' in output
    assert "Plan only. Nothing trained/exported" in output
    assert not options.output_dir.exists()
