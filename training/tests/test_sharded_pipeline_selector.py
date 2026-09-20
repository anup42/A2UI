"""Focused tests for the opt-in full-QAT distributed backend selector."""
from __future__ import annotations

import importlib.util
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
    assert parser.parse_args(["--distributed-backend", "sharded"]).distributed_backend == "sharded"
    with pytest.raises(SystemExit):
        parser.parse_args(["--distributed-backend", "invalid"])


def test_offline_plan_selects_backend_without_writing_output(tmp_path: Path):
    ddp_options = _options(tmp_path)
    ddp = workflow.build_plan(ddp_options)
    assert ddp["distributed_training"] == {"backend": "ddp", "optimizer": "adafactor"}
    assert ddp["training_contract"]["optimizer"] == "adafactor"
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
    assert not sharded_options.output_dir.exists()


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

    def fake_configure(config, *, distributed_backend="ddp"):
        captured["backend"] = distributed_backend
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
