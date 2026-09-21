"""Focused tests for the opt-in full-QAT distributed backend selector."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.pipeline import full_parameter_qat as workflow
from ir_training.pipeline.full_parameter_qat import FullParameterQATOptions
from ir_training.qat.full_model_contract import configure_full_qat
from ir_training.train.recipe import validate_sft_recipe


def _script_module():
    path = ROOT / "scripts" / "run_full_parameter_qat_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_full_parameter_qat_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _options(tmp_path: Path, *, backend: str = "ddp") -> FullParameterQATOptions:
    model = tmp_path / "model"
    inputs = tmp_path / "inputs"
    model.mkdir()
    inputs.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "mobile_training_seed_manifest.json",
        "mobile_qparams.json",
        "model.safetensors",
        "mobile_qparams.safetensors",
    ):
        (model / name).write_bytes(b"fixture")
    for name in ("train.jsonl", "val.jsonl"):
        (inputs / name).write_text("{}\n", encoding="utf-8")
    return FullParameterQATOptions(
        model_dir=model,
        input_dir=inputs,
        output_dir=tmp_path / "output",
        exporter_python=Path(sys.executable),
        distributed_backend=backend,
    )


def test_cli_defaults_to_ddp_and_accepts_sharded():
    parser = _script_module().build_parser()
    assert parser.parse_args([]).distributed_backend == "ddp"
    assert parser.parse_args([]).zero_stage == 2
    assert parser.parse_args(["--distributed-backend", "sharded"]).distributed_backend == "sharded"
    selected = parser.parse_args(["--distributed-backend", "sharded", "--zero-stage", "3"])
    assert selected.zero_stage == 3
    with pytest.raises(SystemExit):
        parser.parse_args(["--distributed-backend", "invalid"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--zero-stage", "4"])


def test_offline_plan_selects_backend_without_writing_output(tmp_path: Path):
    ddp_options = _options(tmp_path)
    ddp = workflow.build_plan(ddp_options)
    assert ddp["distributed_training"] == {"backend": "ddp", "optimizer": "adafactor"}
    assert ddp["training_contract"]["optimizer"] == "adafactor"
    assert ddp["stages"][0] == "assets"
    assert "environment" not in ddp["stages"]
    assert not ddp_options.output_dir.exists()

    sharded_options = replace(
        ddp_options,
        output_dir=tmp_path / "sharded-output",
        distributed_backend="sharded",
    )
    sharded = workflow.build_plan(sharded_options)
    assert sharded["distributed_training"] == {
        "backend": "sharded",
        "optimizer": "adamw_torch",
    }
    assert sharded["training_contract"]["optimizer"] == "adamw_torch"
    assert sharded["stages"] == ["environment", *ddp["stages"]]
    assert not sharded_options.output_dir.exists()

    zero3_options = replace(
        ddp_options,
        output_dir=tmp_path / "zero3-output",
        distributed_backend="sharded",
        zero_stage=3,
    )
    zero3 = workflow.build_plan(zero3_options)
    assert zero3["distributed_training"] == {
        "backend": "sharded",
        "optimizer": "adamw_torch",
        "zero_stage": 3,
    }


def test_offline_plan_rejects_zero3_with_ddp(tmp_path: Path):
    options = replace(_options(tmp_path), zero_stage=3)
    with pytest.raises(ValueError, match="requires distributed_backend='sharded'"):
        workflow.build_plan(options)
    assert not options.output_dir.exists()


def test_environment_stage_records_probe_evidence_without_touching_seed(tmp_path, monkeypatch):
    options = _options(tmp_path, backend="sharded")
    plan = workflow.build_plan(options)
    report = {"nvtx": {"passed": True, "module": "/venv/lib/nvtx/__init__.py"}}
    monkeypatch.setattr("ir_training.train.sharded_contract.validate_sharded_runtime", lambda: report)
    files = workflow.run_stage(plan, "environment")
    assert files == [options.output_dir / "sharded_environment.json"]
    assert json.loads(files[0].read_text()) == report
    assert not (options.output_dir / "fit").exists()


def test_ddp_cannot_invoke_sharded_environment_stage(tmp_path, monkeypatch):
    plan = workflow.build_plan(_options(tmp_path))

    def forbidden_probe():
        raise AssertionError("DDP must not probe sharded packages")

    monkeypatch.setattr("ir_training.train.sharded_contract.validate_sharded_runtime", forbidden_probe)
    with pytest.raises(ValueError, match="must not run in the DDP lane"):
        workflow.run_stage(plan, "environment")


def test_environment_failure_stops_pipeline_before_expensive_stages(tmp_path, monkeypatch):
    options = replace(_options(tmp_path, backend="sharded"), allow_experimental_export=True)
    calls = []

    def failing_probe():
        raise RuntimeError("incompatible NVTX")

    monkeypatch.setattr("ir_training.train.sharded_contract.validate_sharded_runtime", failing_probe)

    def command_runner(command, log_path, environment, **kwargs):
        stage = command[command.index("--worker-stage") + 1]
        calls.append((stage, kwargs["timeout_seconds"]))
        workflow.worker(Path(command[-1]), stage)

    with pytest.raises(RuntimeError, match="incompatible NVTX"):
        workflow.run_pipeline(options, execute=True, command_runner=command_runner)
    assert calls == [("environment", 120.0)]
    manifest = json.loads((options.output_dir / "full_parameter_qat_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed"] == {}
    assert not (options.output_dir / "sharded_environment.json").exists()
    assert not (options.output_dir / "prepared").exists()


def test_offline_plan_rejects_unknown_backend_before_output_write(tmp_path: Path):
    options = _options(tmp_path, backend="unknown")
    with pytest.raises(ValueError, match="distributed_backend"):
        workflow.build_plan(options)
    assert not options.output_dir.exists()


def test_training_config_forwards_selected_backend(tmp_path: Path, monkeypatch):
    options = _options(tmp_path, backend="sharded")
    plan = workflow.build_plan(options)
    captured = {}

    def fake_mobile_config(plan_value, profile, report):
        return {
            "run": {},
            "training": {},
            "golden_eval": {},
        }

    def fake_configure(config, *, distributed_backend="ddp", zero_stage=2):
        captured["backend"] = distributed_backend
        captured["zero_stage"] = zero_stage
        return config

    def fake_validate(config):
        return None

    import ir_training.qat.full_model_contract as full_contract
    from ir_training.pipeline import official_mobile

    monkeypatch.setattr(official_mobile, "training_config", fake_mobile_config)
    monkeypatch.setattr(full_contract, "configure_full_qat", fake_configure)
    monkeypatch.setattr(full_contract, "validate_full_qat_config", fake_validate)
    monkeypatch.setattr(workflow, "validate_h100_profile", lambda profile: None)

    workflow.training_config(plan, {}, {})
    assert captured["backend"] == "sharded"
    assert captured["zero_stage"] == 2


def test_actual_full_qat_configure_produces_valid_sharded_recipe_without_lora():
    base = load_yaml(ROOT / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml")
    config = configure_full_qat(base, distributed_backend="sharded")
    assert validate_sft_recipe(config) == "full_finetune_qat"
    assert config["training"]["distributed_backend"] == "sharded"
    assert config["training"]["optim"] == "adamw_torch"
    assert config["runtime"]["distributed"] == "sharded"
    assert "ddp_sync_each_batch" not in config["training"]
    assert "lora" not in config
    assert "qat_mtp" not in config


def test_default_configure_is_identical_to_explicit_ddp_and_has_no_ds_arguments():
    base = load_yaml(ROOT / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml")
    implicit = configure_full_qat(base)
    explicit = configure_full_qat(base, distributed_backend="ddp")
    assert implicit == explicit
    assert explicit["training"]["optim"] == "adafactor"
    assert explicit["training"]["ddp_sync_each_batch"] is True
    assert "distributed_backend" not in explicit["training"]
    assert "deepspeed" not in explicit["training"]
    assert "fsdp" not in explicit["training"]


def test_generic_training_cannot_opt_into_sharded_backend():
    generic = {
        "run": {},
        "model": {"load_in_4bit": False},
        "training": {"method": "full_finetune_sft", "distributed_backend": "sharded"},
    }
    with pytest.raises(ValueError, match="only by the separate all-parameter QAT workflow"):
        validate_sft_recipe(generic)
