from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gemma4_retained_scale_litertlm as retained_exporter

from ir_training.common.config import load_yaml
from ir_training.eval.mtp_benchmark import build_mtp_benchmark_plan
from ir_training.export.merge_lora import (
    _verify_qat_training_metadata,
    merge_lora_adapter,
)
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


def _write_effective_qat_training_metadata(
    adapter_dir: Path, training_config: Path
) -> None:
    config = load_yaml(training_config)
    adapter_files = [
        {
            "path": candidate.name,
            "size": candidate.stat().st_size,
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        }
        for candidate in sorted(adapter_dir.glob("adapter*"))
        if candidate.is_file()
    ]
    metadata = {
        "training_metadata_version": 2,
        "training_config_sha256": hashlib.sha256(
            training_config.read_bytes()
        ).hexdigest(),
        "training": config["training"],
        "lora": config["lora"],
        "qat": {
            "enabled": True,
            "effective_merged_weight_qat_enabled": True,
            "wrapped_effective_lora_count": 1,
            "uncovered_lora_adapter_linear_names": [],
            "spec": {"effective_merged_weight": True},
        },
        "git_commit": "a" * 40,
        "adapter_checkpoints": [
            {"role": "best_golden", "path": str(adapter_dir), "files": adapter_files}
        ],
    }
    (adapter_dir / "training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


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


def test_hf_loader_requires_exact_checkpoint_key_inventory(monkeypatch):
    calls: list[tuple[str, dict]] = []
    expected_model = object()
    fake_bfloat16 = object()

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return expected_model, {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    fake_torch = types.SimpleNamespace(
        bfloat16=fake_bfloat16, float16=object(), float32=object()
    )
    fake_transformers = types.SimpleNamespace(
        __version__="5.10.1", AutoModelForCausalLM=FakeLoader
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = load_hf_model(
        "local-mobile-seed",
        {
            "model_loader": "auto_causal_lm",
            "dtype": "bfloat16",
            "device_map": "none",
            "require_exact_checkpoint_keys": True,
        },
    )

    assert model is expected_model
    assert calls[0][1]["output_loading_info"] is True


def test_hf_loader_rejects_non_exact_checkpoint_key_inventory(monkeypatch):
    class FakeLoader:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            return object(), {
                "missing_keys": ["model.layers.0.input_layernorm.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    fake_torch = types.SimpleNamespace(
        bfloat16=object(), float16=object(), float32=object()
    )
    fake_transformers = types.SimpleNamespace(
        __version__="5.10.1", AutoModelForCausalLM=FakeLoader
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="exact model-state inventory"):
        load_hf_model(
            "local-mobile-seed",
            {
                "model_loader": "auto_causal_lm",
                "dtype": "bfloat16",
                "device_map": "none",
                "require_exact_checkpoint_keys": True,
            },
        )


def test_merge_records_qat_provenance_without_claiming_int4(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "merged"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-adapter")
    training_config = ROOT / "configs" / "models" / "gemma4_e2b_ir_qat_sft.yaml"
    _write_effective_qat_training_metadata(adapter_dir, training_config)

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
        training_config_path=training_config,
    )

    metadata = json.loads((merged_dir / "qat_mtp_merge_metadata.json").read_text(encoding="utf-8"))
    assert metadata["base_is_qat_derived"] is True
    assert metadata["continued_qat_performed"] is True
    assert metadata["merge_performed_qat"] is False
    assert metadata["packed_int4_output"] is False
    assert metadata["requires_post_merge_quantization"] is True
    assert metadata["mtp_assistant_trained_or_modified"] is False
    assert metadata["manifest_version"] == 4
    assert metadata["training_method"] == "qat_lora_sft"
    assert metadata["qat_enabled"] is True
    assert metadata["qat_effective_merged_weight"] is True
    assert metadata["lora_dropout"] == 0.0
    assert metadata["training_run_metadata"]["verified"] is True
    assert metadata["training_config_sha256"]
    assert len(metadata["adapter_files"]) == 2
    assert len(metadata["merged_model_files"]) == 1
    assert metadata["merged_model_files"][0]["path"] == "model.safetensors"
    assert len(metadata["merged_model_files"][0]["sha256"]) == 64


def test_merge_rejects_unbound_or_tampered_qat_adapter_before_model_load(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    adapter_weights.write_bytes(b"fake-adapter")
    training_config = ROOT / "configs" / "models" / "gemma4_e2b_ir_qat_sft.yaml"

    with pytest.raises(ValueError, match="metadata_present"):
        merge_lora_adapter(
            "google/gemma-4-E2B-it",
            adapter_dir,
            tmp_path / "unbound-output",
            training_config_path=training_config,
        )

    _write_effective_qat_training_metadata(adapter_dir, training_config)
    adapter_weights.write_bytes(b"tampered-adapter")
    with pytest.raises(ValueError, match="adapter_checkpoint_hashes_match"):
        merge_lora_adapter(
            "google/gemma-4-E2B-it",
            adapter_dir,
            tmp_path / "tampered-output",
            training_config_path=training_config,
        )


def _identity(path: Path) -> dict:
    return {
        "path": str(path),
        "present": True,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_retained_mobile_metadata_fixture(tmp_path: Path) -> tuple[Path, dict, dict, str]:
    adapter_dir = tmp_path / "best_golden_checkpoint"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    config_bytes = b"resolved-config"
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    names = [f"base_model.model.model.layers.{index}.self_attn.q_proj" for index in range(205)]
    keys = [f"model.layers.{index}.self_attn.q_proj.weight" for index in range(205)]
    inventory = {
        key: {
            "bits": 4,
            "group_size": None,
            "scale_shape": [1, 1],
            "input_activation_scale_f32_le_hex": "0000803f",
            "output_activation_scale_f32_le_hex": "0000803f",
        }
        for key in keys
    }
    qparams = {
        "verified": True,
        "contract_sha256": "1" * 64,
        "scale_storage_sha256": "2" * 64,
        "inventory_sha256": "3" * 64,
        "tensor_count": 278,
        "inventory": inventory,
    }
    preflight_path = tmp_path / "preflight_report.json"
    reports = [
        {
            "id": gate,
            "passed": True,
            "log": {
                "path": str(tmp_path / f"{gate}.log"),
                "present": True,
                "size_bytes": 1,
                "sha256": "4" * 64,
            },
            "declared_outputs": [],
        }
        for gate in (
            "cuda_bf16_environment",
            "static_qat_profile",
            "mobile_seed_architecture",
            "scale_preserving_qat",
            "model_numeric_preflight",
        )
    ]
    preflight_path.write_text(
        json.dumps({"run_id": "retained-r1", "all_passed": True, "reports": reports}),
        encoding="utf-8",
    )
    generic_bound = {"sha256": "5" * 64, "size_bytes": 1}
    launch_path = tmp_path / "launch_plan.json"
    launch_path.write_text(
        json.dumps(
            {
                "mode": "fresh_run_only_no_resume",
                "run_id": "retained-r1",
                "checks": {"contract_ok": True},
                "bound_artifacts": {
                    **{
                        role: dict(generic_bound)
                        for role in (
                            "mobile_seed_manifest",
                            "mobile_qparams_contract",
                            "official_packed_source",
                            "training_train",
                            "training_val",
                            "golden100",
                        )
                    },
                    "resolved_training_config": {
                        "sha256": config_sha256,
                        "size_bytes": len(config_bytes),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    adapter_files = [
        {
            "path": candidate.name,
            "size": candidate.stat().st_size,
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        }
        for candidate in sorted(adapter_dir.glob("adapter*"))
    ]
    bindings = {
        name: {
            "weight_key": key,
            "bits": 4,
            "group_size": None,
            "scale_shape": [1, 1],
            "input_activation_scale": 1.0,
            "output_activation_scale": 1.0,
        }
        for name, key in zip(names, keys, strict=True)
    }
    metadata = {
        "training_metadata_version": 4,
        "training_config_sha256": config_sha256,
        "run_id": "retained-r1",
        "training": {"method": "qat_lora_sft"},
        "lora": {"dropout": 0.0},
        "git_commit": "a" * 40,
        "checkpoint_role": "best_golden",
        "checkpoint_step": 10,
        "adapter_checkpoints": [
            {"role": "best_golden", "path": str(adapter_dir), "files": adapter_files}
        ],
        "best_golden_eval": {
            "metric": "generation_reward_v5_4_avg",
            "metric_value": 0.75,
            "step": 10,
        },
        "qat": {
            "enabled": True,
            "effective_merged_weight_qat_enabled": True,
            "wrapped_effective_lora_count": 205,
            "wrapped_effective_lora_names": names,
            "uncovered_lora_adapter_linear_names": [],
            "retained_qparams_binding_count": 205,
            "retained_qparams_bindings": bindings,
            "retained_qparams": {
                "verified": True,
                "mode": "retained_mobile",
                "contract_sha256": qparams["contract_sha256"],
                "scale_storage_sha256": qparams["scale_storage_sha256"],
                "inventory_sha256": qparams["inventory_sha256"],
                "tensor_count": 278,
            },
            "spec": {
                "scale_mode": "retained_mobile",
                "fixed_scale_required": True,
                "fixed_activation_scale_required": True,
                "effective_lora_only": True,
                "effective_merged_weight": True,
                "ste_gradient": "clipped",
                "expected_effective_lora_modules": 205,
            },
        },
        "numeric_preflight": {
            "passed": True,
            "qat_enabled": True,
            "top1_probe_match_fraction": 0.95,
            "zero_adapter_initialization": {
                "verified_zero_delta": True,
                "wrapper_count": 205,
                "adapter_pair_count": 205,
                "nonzero_or_invalid_pairs": [],
            },
            "greedy_generation": {
                "passed": True,
                "qat_enabled": True,
                "qat_greedy_deterministic": True,
                "baseline_qat_min_common_prefix_tokens": 8,
            },
        },
        "launcher_provenance": {
            "launch_plan": _identity(launch_path),
            "preflight_report": _identity(preflight_path),
        },
    }
    (adapter_dir / "training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    config = {
        "qat": {
            "scale_mode": "retained_mobile",
            "expected_effective_lora_modules": 205,
        },
        "preflight": {
            "min_top1_probe_match": 0.90,
            "min_baseline_qat_greedy_prefix_tokens": 8,
        },
        "golden_eval": {"metric_for_best_model": "generation_reward_v5_4_avg"},
    }
    return adapter_dir, config, qparams, config_sha256


def test_retained_mobile_merge_requires_full_best_golden_provenance(tmp_path):
    adapter_dir, config, qparams, config_sha256 = _write_retained_mobile_metadata_fixture(
        tmp_path
    )

    report = _verify_qat_training_metadata(
        adapter_dir,
        training_config_sha256=config_sha256,
        training_method="qat_lora_sft",
        mobile_training_seed={"required": False},
        training_config=config,
        mobile_qparams=qparams,
    )

    assert report["verified"] is True
    assert all(report["checks"].values())


def test_merge_and_export_reports_share_exact_golden_selection_shape(tmp_path):
    adapter_dir, config, qparams, _ = _write_retained_mobile_metadata_fixture(
        tmp_path
    )
    config_path = tmp_path / "resolved_training_config.yaml"
    config_path.write_text(
        "qat:\n"
        "  scale_mode: retained_mobile\n"
        "  expected_effective_lora_modules: 205\n"
        "preflight:\n"
        "  min_top1_probe_match: 0.90\n"
        "  min_baseline_qat_greedy_prefix_tokens: 8\n"
        "golden_eval:\n"
        "  metric_for_best_model: generation_reward_v5_4_avg\n",
        encoding="utf-8",
    )
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    metadata_path = adapter_dir / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    launch_path = Path(metadata["launcher_provenance"]["launch_plan"]["path"])
    launch_plan = json.loads(launch_path.read_text(encoding="utf-8"))
    launch_plan["bound_artifacts"]["resolved_training_config"] = {
        "sha256": config_sha256,
        "size_bytes": config_path.stat().st_size,
    }
    launch_path.write_text(json.dumps(launch_plan), encoding="utf-8")
    metadata["training_config_sha256"] = config_sha256
    metadata["launcher_provenance"]["launch_plan"] = _identity(launch_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    merge_report = _verify_qat_training_metadata(
        adapter_dir,
        training_config_sha256=config_sha256,
        training_method="qat_lora_sft",
        mobile_training_seed={"required": False},
        training_config=config,
        mobile_qparams=qparams,
    )

    class FixtureQParams:
        contract_sha256 = qparams["contract_sha256"]
        scale_storage_sha256 = qparams["scale_storage_sha256"]

        def trainable_projection_weight_keys(self):
            return tuple(sorted(qparams["inventory"]))

    export_report = retained_exporter._best_adapter_provenance_report(
        adapter_dir,
        config_path,
        qparams=FixtureQParams(),
        seed_report={},
    )
    merge_provenance = {
        "metadata": {"training_run_metadata": merge_report}
    }

    assert merge_report["verified"] is True
    assert merge_report["golden_selection"] == export_report["golden_selection"]
    assert "launcher" in merge_report["golden_selection"]
    assert retained_exporter._merged_golden_selection_matches(
        merge_provenance, export_report["golden_selection"]
    )


def test_retained_mobile_merge_rejects_partial_204_projection_metadata(tmp_path):
    adapter_dir, config, qparams, config_sha256 = _write_retained_mobile_metadata_fixture(
        tmp_path
    )
    metadata_path = adapter_dir / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["qat"]["wrapped_effective_lora_count"] = 204
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = _verify_qat_training_metadata(
        adapter_dir,
        training_config_sha256=config_sha256,
        training_method="qat_lora_sft",
        mobile_training_seed={"required": False},
        training_config=config,
        mobile_qparams=qparams,
    )

    assert report["verified"] is False
    assert report["checks"]["exact_205_effective_lora_modules"] is False

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
