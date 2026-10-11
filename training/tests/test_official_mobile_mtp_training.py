"""Opt-in assistant adaptation must leave the retained-mobile target lane intact.

All artifacts are tiny placeholders. These tests never load models, download,
train, convert weights, inspect real packages, or invoke an Android device.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.pipeline import official_mobile as workflow
from ir_training.pipeline import official_mobile_mtp as mtp
from ir_training.pipeline.official_mobile import OfficialMobileOptions
from ir_training.train.gpu_profile import build_gpu_profile

MTP_OPTIONS = {"train_mtp", "mtp_epochs", "mtp_steps", "mtp_learning_rate"}
MTP_PATHS = {
    "mtp_config", "mtp_training_dir", "mtp_checkpoint", "mtp_training_report",
    "mtp_export_dir", "mtp_export_report", "target_litertlm",
}
TARGET_STAGES = [
    "assets", "prepare", "configure", "no_op_export", "preflight", "training",
    "best_golden32", "best_golden35", "best_bixby50", "merge", "export",
]


@pytest.fixture
def options(tmp_path: Path) -> OfficialMobileOptions:
    seed, inputs = tmp_path / "seed", tmp_path / "inputs"
    seed.mkdir()
    inputs.mkdir()
    for name in (
        "config.json", "tokenizer_config.json", "mobile_training_seed_manifest.json",
        "mobile_qparams.json", "mobile_qparams.safetensors",
    ):
        (seed / name).write_text("{}", encoding="utf-8")
    for name in ("train.jsonl", "val.jsonl"):
        (inputs / name).write_text("{}\n", encoding="utf-8")
    source, official = tmp_path / "source.safetensors", tmp_path / "official.litertlm"
    source.write_bytes(b"not a real checkpoint")
    official.write_bytes(b"not a real package")
    return OfficialMobileOptions(
        model_dir=seed, input_dir=inputs, source_safetensors=source,
        official_litertlm=official, output_dir=tmp_path / "mobile-run",
        exporter_python=Path(sys.executable), steps=20, eval_steps=5,
        golden_every_steps=10,
    )


def _profile(count: int = 4) -> dict:
    inventory = {
        "version": 1, "inherited_cuda_visible_devices": None,
        "visible_gpu_count": count,
        "devices": [
            {"visible_index": index, "launch_identifier": str(index),
             "uuid": f"GPU-{index}", "name": "NVIDIA H100 80GB HBM3",
             "total_memory_bytes": 80 * 1024**3, "compute_capability": [9, 0]}
            for index in range(count)
        ],
    }
    return build_gpu_profile(inventory, model="e2b", cpu_count=32)


def test_no_flag_keeps_legacy_plan_and_excludes_all_new_option_keys(options):
    plan = workflow.build_plan(options)
    explicit = workflow.build_plan(replace(options, train_mtp=False))
    assert explicit == plan
    assert plan["stages"] == TARGET_STAGES
    assert set(plan["options"]) == set(asdict(options)) - MTP_OPTIONS
    assert not MTP_OPTIONS.intersection(plan["options"])
    assert not MTP_PATHS.intersection(plan["paths"])
    assert plan["mtp"] == {
        "training": False, "inference": False,
        "official_section_preserved_but_unused": True,
    }
    assert not options.output_dir.exists()


def test_opt_in_inserts_only_sequential_assistant_stage_after_merge(options):
    before = workflow.build_plan(options)
    plan = workflow.build_plan(replace(options, train_mtp=True))
    expected = TARGET_STAGES.copy()
    expected.insert(expected.index("export"), "mtp_training")
    assert plan["stages"] == expected
    assert plan["mtp"]["training"] is True
    assert plan["mtp"]["inference"] is True
    assert plan["mtp"]["official_section_preserved_but_unused"] is False
    assert plan["selection"] == before["selection"]
    assert plan["preparation"] == before["preparation"]
    assert MTP_PATHS.issubset(plan["paths"])
    for key in MTP_PATHS:
        assert Path(plan["paths"][key]).is_relative_to(options.output_dir)
    assert plan["paths"]["target_litertlm"] != plan["paths"]["litertlm"]
    assert plan["paths"]["mtp_checkpoint"] != plan["paths"]["best_checkpoint"]
    assert not options.output_dir.exists()


@pytest.mark.parametrize("gpus", [1, 4, 8])
def test_assistant_flag_does_not_change_target_training_config(options, gpus):
    plain = workflow.build_plan(options)
    trained = workflow.build_plan(replace(options, train_mtp=True))
    report = {"tokenizer": {}, "final_evaluation_datasets": {}}
    profile = _profile(gpus)
    target = workflow.training_config(plain, copy.deepcopy(profile), report)
    assert workflow.training_config(trained, copy.deepcopy(profile), report) == target
    assert "qat_mtp" not in target
    assert target["qat"]["expected_effective_lora_modules"] == 205
    assert target["qat"]["expected_frozen_activation_modules"] == 70
    assert target["training"]["expected_effective_batch_size"] == 32
    assert workflow.no_op_export_command(trained) == workflow.no_op_export_command(plain)
    for cohort in workflow.GOLDENS:
        assert workflow.evaluation_command(trained, cohort) == workflow.evaluation_command(plain, cohort)


def test_opt_in_export_uses_existing_trained_official_topology_route(options):
    plain = workflow.build_plan(options)
    trained = workflow.build_plan(replace(options, train_mtp=True))
    old = workflow.mobile_export_config(plain)["pipeline"]
    new = workflow.mobile_export_config(trained)["pipeline"]
    assert old["mtp"]["enabled"] is False
    assert old["mtp"]["train_assistant"] is False
    assert old["mtp"]["weight_source"] == "official"
    assert new["mtp"]["enabled"] is True
    assert new["mtp"]["train_assistant"] is True
    assert new["mtp"]["weight_source"] == "trained"
    assert new["mtp"]["training_config"] == trained["paths"]["mtp_config"]
    assert new["mtp"]["trained_checkpoint"] == trained["paths"]["mtp_checkpoint"]
    assert new["mtp"]["target_intermediate_litertlm"] == trained["paths"]["target_litertlm"]
    assert new["mtp"]["exact_topology_output_dir"] == trained["paths"]["mtp_export_dir"]
    assert new["public_export"]["enabled"] is False
    assert new["retained_scale_export"] == old["retained_scale_export"]
    assert new["source"] == old["source"]


@pytest.mark.parametrize("knob,value", [
    ("mtp_epochs", 2), ("mtp_steps", 5), ("mtp_learning_rate", 5e-6),
])
def test_nondefault_assistant_knobs_require_explicit_flag(options, knob, value):
    with pytest.raises(ValueError, match="(?i)mtp"):
        workflow.build_plan(replace(options, **{knob: value}))


@pytest.mark.parametrize("knob,value", [
    ("mtp_epochs", 0), ("mtp_epochs", -1), ("mtp_epochs", True),
    ("mtp_epochs", 1.5), ("mtp_steps", 0), ("mtp_steps", -1),
    ("mtp_steps", True), ("mtp_learning_rate", 0),
    ("mtp_learning_rate", -1), ("mtp_learning_rate", float("nan")),
    ("mtp_learning_rate", float("inf")), ("mtp_learning_rate", True),
])
def test_invalid_assistant_recipe_fails_before_preparation(options, knob, value):
    with pytest.raises(ValueError, match="(?i)mtp"):
        workflow.build_plan(replace(options, train_mtp=True, **{knob: value}))
    assert not options.output_dir.exists()


def test_assistant_recipe_is_recorded_without_changing_target_horizon(options):
    plan = workflow.build_plan(replace(
        options, train_mtp=True, mtp_epochs=2, mtp_steps=15, mtp_learning_rate=5e-6,
    ))
    assert plan["options"]["train_mtp"] is True
    assert plan["options"]["mtp_epochs"] == 2
    assert plan["options"]["mtp_steps"] == 15
    assert plan["options"]["mtp_learning_rate"] == 5e-6
    assert plan["options"]["steps"] == 20
    assert plan["options"]["learning_rate"] == 1e-5


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_passes_explicit_assistant_options_without_executing(options, monkeypatch, enabled):
    spec = importlib.util.spec_from_file_location(
        "official_mobile_mtp_cli_test", ROOT / "scripts/run_official_mobile_pipeline.py",
    )
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    seen = []

    def record(options, *, execute):
        seen.append((options, execute))
        return {}

    monkeypatch.setattr(entry, "run_pipeline", record)
    argv = []
    for name in ("model_dir", "input_dir", "source_safetensors", "official_litertlm", "output_dir"):
        argv.extend(("--" + name.replace("_", "-"), str(getattr(options, name))))
    if enabled:
        argv += ["--train-mtp", "--mtp-epochs", "2", "--mtp-steps", "3", "--mtp-learning-rate", "5e-6"]
    assert entry.main(argv) == 0
    captured, execute = seen.pop()
    assert captured.train_mtp is enabled
    assert execute is False
    assert captured.mtp_epochs == (2 if enabled else 1)
    assert captured.mtp_steps == (3 if enabled else None)
    assert captured.mtp_learning_rate == (5e-6 if enabled else 1e-5)


def test_drafter_config_binds_only_prepared_training_data_and_selected_target(options):
    import yaml

    plan = workflow.build_plan(replace(options, train_mtp=True, mtp_steps=7))
    config_path = Path(plan["paths"]["config"])
    config_path.parent.mkdir(parents=True)
    target_config = {"model": {"chat_template_kwargs": {"enable_thinking": True}}}
    config_path.write_text(yaml.safe_dump(target_config), encoding="utf-8")
    config = mtp.drafter_config(plan)
    assert config["run"]["dataset_dir"] == plan["paths"]["prepared"]
    assert config["run"]["output_dir"] == plan["paths"]["mtp_training_dir"]
    assert config["target"]["model_id_or_path"] == plan["paths"]["merged"]
    assert config["target"]["adapter_checkpoint"] == plan["paths"]["best_checkpoint"]
    assert config["target"]["training_config"] == plan["paths"]["config"]
    assert config["target"]["tokenizer_id_or_path"] == plan["options"]["model_dir"]
    assert config["training"]["max_steps"] == 7
    assert config["training"]["require_cuda"] is True
    assert config["training"]["overflow_policy"] == "error"
    assert config["training"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "golden_eval" not in config
    assert "final_evaluation_datasets" not in config
    assert all(name not in json.dumps(config) for name in ("golden32.jsonl", "golden35.jsonl", "bixby50.jsonl"))
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == target_config


def test_assistant_stage_uses_one_selected_gpu_without_target_ddp_environment(options, monkeypatch):
    plan = workflow.build_plan(replace(options, train_mtp=True))
    incoming = {
        "CUDA_VISIBLE_DEVICES": "GPU-2,GPU-4", "RANK": "0", "LOCAL_RANK": "0",
        "WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12345", "HF_HUB_OFFLINE": "1",
    }
    captured = []
    monkeypatch.setattr(workflow, "_environment", lambda *args, **kwargs: incoming.copy())
    monkeypatch.setattr(workflow, "run_bounded_command", lambda command, log, env, **kwargs: captured.append((command, env)))
    evidence = [Path(plan["paths"]["mtp_training_report"])]
    monkeypatch.setattr(mtp, "verify_mtp_training", lambda _: evidence)
    assert workflow.run_stage(plan, "mtp_training") == evidence
    command, env = captured.pop()
    assert "--execute-drafter-training" in command
    assert "torchrun" not in " ".join(command)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-2"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert not {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}.intersection(env)
    assert incoming["CUDA_VISIBLE_DEVICES"] == "GPU-2,GPU-4"


def test_assistant_stage_refuses_absent_gpu_before_launch(options, monkeypatch):
    plan = workflow.build_plan(replace(options, train_mtp=True))
    monkeypatch.setattr(workflow, "_environment", lambda *args, **kwargs: {})
    monkeypatch.setattr(workflow, "run_bounded_command", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="selected CUDA GPU"):
        workflow.run_stage(plan, "mtp_training")


def test_default_lane_cannot_enter_assistant_stage(options):
    with pytest.raises(ValueError, match="requires --train-mtp"):
        workflow.run_stage(workflow.build_plan(options), "mtp_training")


def test_native_quality_handoff_is_not_falsely_advertised_for_trained_mtp(options):
    plain = workflow.build_plan(options)
    trained = workflow.build_plan(replace(options, train_mtp=True))
    assert "evaluate_official_mobile_native.py" in " ".join(workflow.native_quality_command(plain))
    with pytest.raises(ValueError, match="(?i)(mtp|quality)"):
        workflow.native_quality_command(trained)
    target_command = workflow.benchmark_command(trained)
    mtp_command = workflow.mtp_benchmark_command(trained)
    assert "--mtp" not in target_command
    assert "--mtp" in mtp_command
    assert Path(mtp_command[mtp_command.index("--output-dir") + 1]).name == "mtp_on"


@pytest.mark.parametrize("enabled", [False, True])
def test_export_remains_cpu_isolated_in_both_modes(options, monkeypatch, enabled):
    plan = workflow.build_plan(replace(options, train_mtp=enabled))
    captured = []
    validated = []

    class StopBeforeExecution(Exception):
        pass

    def stop(command, log, env, **kwargs):
        captured.append((command, env))
        raise StopBeforeExecution

    monkeypatch.setattr(workflow, "_environment", lambda *args, **kwargs: {
        "CUDA_VISIBLE_DEVICES": "0,1", "A2UI_CUDA_VISIBLE_DEVICES": "0,1",
        "A2UI_EXCLUDE_CUDA_DEVICES": "2", "HF_HUB_OFFLINE": "1",
    })
    monkeypatch.setattr(mtp, "verify_mtp_training", lambda _: validated.append(True))
    monkeypatch.setattr(workflow, "run_bounded_command", stop)
    with pytest.raises(StopBeforeExecution):
        workflow.run_stage(plan, "export")
    command, env = captured.pop()
    assert command[0] == plan["options"]["exporter_python"]
    assert "--execute-retained-scale-export" in command
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "A2UI_CUDA_VISIBLE_DEVICES" not in env
    assert "A2UI_EXCLUDE_CUDA_DEVICES" not in env
    assert validated == ([True] if enabled else [])
