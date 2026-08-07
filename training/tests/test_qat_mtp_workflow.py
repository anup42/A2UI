from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.eval.mtp_benchmark import build_mtp_benchmark_plan
from ir_training.export.merge_lora import merge_lora_adapter
from ir_training.export.q4_0 import build_q4_0_conversion_plan
from ir_training.models.hf_loading import load_hf_model
from ir_training.models.registry import create_adapter
from ir_training.qat_mtp.workflow import (
    OFFICIAL_QAT_ASSISTANT,
    OFFICIAL_QAT_TARGET,
    checkpoint_precision_family,
    validate_benchmark_config,
    validate_training_config,
)
from ir_training.train.lora_config import build_lora_config


def _recommended_training_config() -> dict:
    return load_yaml(ROOT / "configs" / "models" / "gemma4_e2b_ir_qat_lora.yaml")


def _recommended_benchmark_config() -> dict:
    return load_yaml(ROOT / "configs" / "eval" / "gemma4_e2b_qat_mtp.yaml")


def test_recommended_qat_mtp_configs_pass_static_validation():
    training_config = _recommended_training_config()
    benchmark_config = _recommended_benchmark_config()

    assert validate_training_config(training_config) == []
    assert validate_benchmark_config(benchmark_config) == []
    assert training_config["model"]["model_id"] == OFFICIAL_QAT_TARGET
    assert training_config["model"]["load_in_4bit"] is False
    assert training_config["qat_mtp"]["assistant_model_id"] == OFFICIAL_QAT_ASSISTANT
    assert training_config["qat_mtp"]["train_assistant"] is False
    assert checkpoint_precision_family(OFFICIAL_QAT_TARGET) == "q4_0"
    assert checkpoint_precision_family(OFFICIAL_QAT_ASSISTANT) == "q4_0"


def test_static_validation_rejects_qlora_joint_assistant_training_and_false_int4_claim():
    training_config = _recommended_training_config()
    training_config["model"]["load_in_4bit"] = True
    training_config["qat_mtp"]["train_assistant"] = True
    training_codes = {issue.code for issue in validate_training_config(training_config)}

    benchmark_config = _recommended_benchmark_config()
    benchmark_config["benchmark"]["packed_int4"] = True
    benchmark_codes = {issue.code for issue in validate_benchmark_config(benchmark_config)}

    assert {"qlora_is_not_qat", "assistant_training_unsupported"}.issubset(training_codes)
    assert "false_int4_claim" in benchmark_codes


def test_peft_default_lora_scope_omits_target_modules(monkeypatch):
    captured: dict = {}

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(LoraConfig=FakeLoraConfig))
    adapter = create_adapter({"family": "gemma", "model_id": OFFICIAL_QAT_TARGET})

    build_lora_config(
        adapter,
        {
            "r": 16,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": "peft-default",
            "modules_to_save": [],
        },
    )

    assert captured["r"] == 16
    assert captured["lora_alpha"] == 16
    assert "target_modules" not in captured
    assert "modules_to_save" not in captured


def test_hf_loader_selects_transformers_5_causal_loader(monkeypatch):
    calls: list[tuple[str, dict]] = []
    fake_bfloat16 = object()

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return object()

    fake_torch = types.SimpleNamespace(bfloat16=fake_bfloat16, float16=object(), float32=object())
    fake_transformers = types.SimpleNamespace(__version__="5.10.1", AutoModelForCausalLM=FakeLoader)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    load_hf_model(
        OFFICIAL_QAT_TARGET,
        {
            "model_loader": "auto_causal_lm",
            "dtype": "bfloat16",
            "device_map": "none",
            "load_in_4bit": False,
        },
    )

    assert calls == [
        (
            OFFICIAL_QAT_TARGET,
            {
                "trust_remote_code": False,
                "dtype": fake_bfloat16,
            },
        )
    ]


def test_merge_records_qat_provenance_without_claiming_int4(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "merged"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-adapter")

    class FakeMerged:
        def save_pretrained(self, output, safe_serialization):
            assert safe_serialization is True
            Path(output, "model.safetensors").write_text("fake", encoding="utf-8")

    class FakeWrapped:
        def merge_and_unload(self):
            return FakeMerged()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter):
            assert adapter == str(adapter_dir)
            return FakeWrapped()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return FakeProcessor()

        def save_pretrained(self, output):
            Path(output, "processor.json").write_text("{}", encoding="utf-8")

    class UnexpectedTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("processor path should succeed")

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoProcessor=FakeProcessor, AutoTokenizer=UnexpectedTokenizer),
    )
    monkeypatch.setattr("ir_training.export.merge_lora.load_hf_model", lambda *args, **kwargs: object())

    merged_dir = merge_lora_adapter(
        OFFICIAL_QAT_TARGET,
        adapter_dir,
        output_dir,
        model_loader="auto_causal_lm",
        dtype="bfloat16",
        training_config_path=(
            ROOT / "configs" / "models" / "gemma4_e2b_ir_qat_sft.yaml"
        ),
    )

    metadata = json.loads((merged_dir / "qat_mtp_merge_metadata.json").read_text(encoding="utf-8"))
    assert metadata["base_is_qat_derived"] is True
    assert metadata["continued_qat_performed"] is False
    assert metadata["packed_int4_output"] is False
    assert metadata["requires_post_merge_quantization"] is True
    assert metadata["mtp_assistant_trained_or_modified"] is False
    assert metadata["manifest_version"] == 2
    assert metadata["training_method"] == "qat_lora_sft"
    assert metadata["qat_enabled"] is True
    assert metadata["training_config_sha256"]
    assert len(metadata["adapter_files"]) == 2
    assert len(metadata["merged_model_files"]) == 1
    assert metadata["merged_model_files"][0]["path"] == "model.safetensors"
    assert len(metadata["merged_model_files"][0]["sha256"]) == 64


def test_reference_benchmark_plan_never_claims_packed_int4():
    plan = build_mtp_benchmark_plan(_recommended_benchmark_config())

    assert plan["validation"]["ok"] is True
    assert plan["mode"] == "transformers_reference"
    assert plan["packed_int4"] is False
    assert plan["acceptance_rate_available"] is False
    assert plan["final_runtime_validation_required"] is True


def test_q4_0_plan_converts_target_and_assistant_with_same_precision(tmp_path):
    config = load_yaml(ROOT / "configs" / "export" / "gemma4_e2b_qat_q4_0.yaml")
    plan = build_q4_0_conversion_plan(config, llama_cpp_dir_override=tmp_path / "llama.cpp")

    assert plan["quantization"] == "Q4_0"
    assert plan["include_assistant"] is True
    assert [step["name"] for step in plan["steps"]] == [
        "convert_target_to_f16_gguf",
        "quantize_target_q4_0",
        "convert_assistant_to_f16_gguf",
        "quantize_assistant_q4_0",
    ]
    assert plan["steps"][1]["command"][-1] == "Q4_0"
    assert plan["steps"][3]["command"][-1] == "Q4_0"
    assert plan["mobile_wna8o8_output"] is False
    assert plan["final_runtime_validation_required"] is True


def test_q4_0_plan_rejects_nonmatching_assistant(tmp_path):
    config = load_yaml(ROOT / "configs" / "export" / "gemma4_e2b_qat_q4_0.yaml")
    config["source"]["assistant_model_id"] = "google/gemma-4-E2B-it-assistant"

    with pytest.raises(ValueError, match="exact matching assistant"):
        build_q4_0_conversion_plan(config, llama_cpp_dir_override=tmp_path / "llama.cpp")
