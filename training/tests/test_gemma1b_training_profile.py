"""Gemma 3 1B planning/config tests: no model loading, CUDA, or downloads."""
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

from ir_training.common.config import load_yaml
from ir_training.pipeline.golden_training import (
    GoldenTrainingOptions,
    build_plan,
    configure_command,
    evaluation_command,
)
from ir_training.train.gpu_profile import build_gpu_profile, detect_cuda_devices
from ir_training.train.precision import full_finetune_precision_policy
from ir_training.train.recipe import validate_sft_recipe


def script(name):
    spec = importlib.util.spec_from_file_location(f"gemma1b_test_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(count=4, name="NVIDIA H100 80GB HBM3"):
    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: count,
        get_device_properties=lambda index: SimpleNamespace(
            name=name, total_memory=80 * 1024**3, major=9, minor=0, uuid=f"GPU-{index}"),
    )
    return detect_cuda_devices(torch_module=SimpleNamespace(cuda=cuda), environment={})


@pytest.fixture
def inputs(tmp_path):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    config = {"model_type": "gemma3_text", "hidden_size": 1152,
              "num_hidden_layers": 26, "vocab_size": 262144,
              "intermediate_size": 6912, "num_attention_heads": 4,
              "num_key_value_heads": 1, "head_dim": 256,
              "max_position_embeddings": 32768, "tie_word_embeddings": True}
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fake weights: must never load")
    for name in ("train", "val", "golden32", "golden35", "bixby50"):
        (data / f"{name}.jsonl").write_text("", encoding="utf-8")
    return model, data


def test_profile_defaults_are_full_qat_with_separate_evaluation_limit(inputs, tmp_path):
    model, data = inputs
    options = GoldenTrainingOptions(model, tmp_path / "run", profile="1b", input_dir=data)
    assert options.qat is True
    assert options.max_seq_length == 4096
    assert options.max_input_tokens == 5120
    plan = build_plan(options)
    command = configure_command(plan)
    assert command[command.index("--profile") + 1] == "1b"
    assert command[command.index("--max-seq-length") + 1] == "4096"
    assert command[command.index("--max-input-tokens") + 1] == "5120"
    assert "--qat" in command
    assert "--token-cache" in command
    assert "--gradient-checkpointing" in command
    assert plan["goldens"]["golden32"]["selection_role"] == "development_checkpoint_selection"
    assert plan["goldens"]["golden35"]["selection_role"] == "final_only_holdout"
    assert plan["goldens"]["bixby50"]["selection_role"] == "final_only_holdout"
    for cohort in ("golden32", "golden35", "bixby50"):
        evaluation = evaluation_command(plan, "best", cohort, tmp_path / cohort)
        assert evaluation[evaluation.index("--checkpoint-kind") + 1] == "merged"
        assert evaluation[evaluation.index("--max-input-tokens") + 1] == "5120"
        assert evaluation[evaluation.index("--qat-mode") + 1] == "on"
    assert not options.output_dir.exists()


@pytest.mark.parametrize("profile", ["e2b", "270m"])
def test_legacy_profile_defaults_and_shared_limit_are_unchanged(inputs, tmp_path, profile):
    model, data = inputs
    options = GoldenTrainingOptions(model, tmp_path / "run", profile=profile, input_dir=data)
    assert options.qat is False
    assert options.max_input_tokens == 4096
    assert "--max-input-tokens" not in configure_command(build_plan(options))
    with pytest.raises(ValueError, match="requires --max-input-tokens"):
        build_plan(replace(options, max_input_tokens=5120))


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_1b_h100_profile_keeps_microbatch_one_and_effective_batch32(count):
    profile = build_gpu_profile(inventory(count), model="1b", cpu_count=64)
    assert profile["microbatch"] == 1
    assert profile["world_size"] == count
    assert profile["gradient_accumulation_steps"] == 32 // count
    assert profile["effective_batch_size"] == 32
    assert profile["dtype"] == "bfloat16"
    assert profile["attn_implementation"] == "sdpa"
    assert profile["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert not profile["benchmark_verified"]
    assert not profile["memory_safety"]["maximum_sequence_verified"]


def test_non_h100_preserves_explicit_effective_batch_and_keeps_micro_one():
    assert build_gpu_profile(inventory(name="A100"), model="1b")["effective_batch_size"] == 32
    profile = build_gpu_profile(inventory(), model="1b", effective_batch=16, microbatch=2)
    assert profile["microbatch"] == 2
    assert profile["gradient_accumulation_steps"] == 2
    assert profile["effective_batch_size"] == 16


def test_yaml_has_no_lora_and_preserves_full_precision_master_policy():
    config = load_yaml(ROOT / "configs/models/gemma3_1b_ir_full_qat_sft.yaml")
    assert validate_sft_recipe(config) == "full_finetune_qat"
    assert not config.get("lora")
    assert not config["training"].get("full_parameter_training")
    assert config["training"]["backward_preflight"] is True
    assert config["model"]["model_id"] == "google/gemma-3-1b-it"
    assert config["qat"]["weight_bits"] == 8
    assert config["qat"]["activation_bits"] == 32
    assert config["qat"]["quantize_embeddings"] is True
    assert config["qat"]["exclude_modules"] == []
    assert full_finetune_precision_policy(config, resolved_dtype="bfloat16") == {
        "policy": "fp32_parameters_v1", "parameter_dtype": "float32", "compute_dtype": "bfloat16"}


def test_configure_uses_one_config_with_bound_separate_prompt_limits(inputs, tmp_path, monkeypatch):
    model, data = inputs
    prepare = script("prepare_review_training")
    monkeypatch.setattr("ir_training.train.gpu_profile.detect_cuda_devices", lambda: inventory())
    bindings = {name: {"split_path": str(data / f"{name}.jsonl")} for name in ("golden35", "bixby50")}
    calls = []

    def verify(*args, **kwargs):
        calls.append(kwargs)
        return {"tokenizer": {"chat_template_kwargs": {}}, "split_rows": {"train": 100},
                "benchmark": {"benchmark_kind": "explicit_repeated_case"},
                "final_evaluation_datasets": bindings}

    monkeypatch.setattr(prepare, "verify_prepared", verify)
    args = SimpleNamespace(profile="1b", model_dir=model, dataset_dir=data,
        golden_file=data / "golden32.jsonl", golden35_file=data / "golden35.jsonl",
        bixby50_file=data / "bixby50.jsonl", output_dir=tmp_path / "fit",
        epochs=1, steps=20, max_seq_length=4096, resume=None, qv_baseline=False, qat=False)
    config, report = prepare.build_config(args)
    assert config["training"]["method"] == "full_finetune_qat"
    assert config["training"]["max_seq_length"] == 4096
    assert config["golden_eval"]["max_input_tokens"] == 5120
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["training"]["learning_rate"] == 5e-6
    assert config["model"]["model_source"] == str(model.resolve())
    assert config["run"]["purpose"] == "gemma3_1b_full_qat_v1"
    assert config["final_evaluation_datasets"] == bindings
    assert calls[0]["max_sequence"] == 4096 and calls[0]["max_prompt"] == 5120
    assert report["model_loaded"] is False
    assert report["training_executed"] is False
    assert not args.output_dir.exists()


@pytest.mark.parametrize("name", ["hidden_size", "num_hidden_layers", "vocab_size"])
def test_1b_plan_rejects_other_architecture_before_gpu_or_weight_load(inputs, tmp_path, name):
    model, data = inputs
    path = model / "config.json"
    config = json.loads(path.read_text())
    config[name] += 1
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        build_plan(GoldenTrainingOptions(model, tmp_path / "run", profile="1b", input_dir=data))


def test_cli_1b_default_is_resolved_before_planning(inputs, tmp_path):
    model, data = inputs
    launcher = script("run_golden_training")
    args = vars(launcher.build_parser().parse_args([
        "--profile", "1b", "--model-dir", str(model), "--input-dir", str(data),
        "--output-dir", str(tmp_path / "run")]))
    for name in ("execute", "prepare_only", "continue_run"):
        args.pop(name)
    options = GoldenTrainingOptions(**args)
    assert options.max_input_tokens == 5120 and options.qat
    assert build_plan(options)["options"]["max_seq_length"] == 4096
