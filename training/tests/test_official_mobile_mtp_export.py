"""Small, offline fixtures for the optional trained-drafter publication gates.

No model, LiteRT converter, CUDA runtime, or Android device is used here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.pipeline import official_mobile as mobile
from ir_training.pipeline import official_mobile_mtp as mtp

EXPORT_GATES = {
    "training_scope_supported", "training_provenance_verified",
    "complete_23_weight_mapping", "converter_inventory_count",
    "converter_quantization_layout", "converter_constants_transferred_exactly",
    "non_mapped_state_byte_exact", "official_graph_structure",
    "official_quantization_layout", "official_execution_topology_ignoring_buffer_indices",
    "official_layout_ignoring_buffer_indices", "execution_contract_complete",
    "official_execution_contract", "official_execution_contract_ignoring_buffer_indices",
    "section_size_unchanged", "package_outside_mtp_byte_exact",
    "fine_tuned_target_section_byte_exact",
}


def _write(path: str | Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _json(path: str | Path, payload: dict) -> Path:
    return _write(path, json.dumps(payload).encode("utf-8"))


def _yaml(path: str | Path, payload: dict) -> Path:
    return _write(path, yaml.safe_dump(payload).encode("utf-8"))


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def plan(tmp_path):
    seed, inputs = tmp_path / "seed", tmp_path / "inputs"
    for name in ("config.json", "tokenizer_config.json", "mobile_training_seed_manifest.json",
                 "mobile_qparams.json", "mobile_qparams.safetensors", "model.safetensors"):
        _write(seed / name, b"{}")
    for name in ("train.jsonl", "val.jsonl"):
        _write(inputs / name, b"{}\n")
    options = mobile.OfficialMobileOptions(
        model_dir=seed, input_dir=inputs,
        source_safetensors=_write(tmp_path / "source.safetensors", b"source"),
        official_litertlm=_write(tmp_path / "official.litertlm", b"official"),
        output_dir=tmp_path / "run", exporter_python=Path(sys.executable),
        steps=20, eval_steps=5, golden_every_steps=10, train_mtp=True,
    )
    return mobile.build_plan(options)


@pytest.fixture
def export_evidence(plan, monkeypatch):
    """Isolate packaging checks from the separately tested training verifier."""
    paths = plan["paths"]
    _write(paths["target_litertlm"], b"selected-target-with-default-drafter")
    _write(paths["litertlm"], b"selected-target-with-trained-drafter")
    target_report = {
        "passed": True, "executed": True,
        "output_litertlm": paths["target_litertlm"],
        "output_sha256": _sha(paths["target_litertlm"]),
    }
    _json(paths["export_report"], target_report)
    report = {
        "ready": True, "issues": [], "assistant_trained": True,
        "assistant_weight_source": "trained_checkpoint",
        "final_artifact_gate_pass": True,
        "official_artifact": plan["options"]["official_litertlm"],
        "official_artifact_sha256_expected": plan["format"]["official_sha256"],
        "official_artifact_sha256_observed": plan["format"]["official_sha256"],
        "assistant_checkpoint": paths["mtp_checkpoint"],
        "assistant_training_config": paths["mtp_config"],
        "package_input": paths["target_litertlm"],
        "package_output": paths["litertlm"],
        "gates": dict.fromkeys(EXPORT_GATES, True),
        "training_scope": {"supported_projection_only_transplant": True},
        "training_provenance": {"verified": True, "metadata": {"fixture": "trained-assistant"}},
        "package_boundary": {
            "output": paths["litertlm"], "output_sha256": _sha(paths["litertlm"]),
            "bytes_outside_mtp_verified_unchanged": True,
            "target_section_sha256_before": "a" * 64,
            "target_section_sha256_after": "a" * 64,
        },
    }
    _json(Path(paths["mtp_checkpoint"]) / "mtp_drafter_training_metadata.json",
          report["training_provenance"]["metadata"])
    _json(paths["mtp_export_report"], report)
    calls = []

    def checked_training(request):
        calls.append(request)
        return []

    monkeypatch.setattr(mtp, "verify_mtp_training", checked_training)
    return report, target_report, calls


def test_export_binds_distinct_target_and_final_packages(plan, export_evidence):
    _, _, calls = export_evidence
    evidence = mtp.verify_mtp_export(plan)
    assert calls == [plan]
    assert Path(plan["paths"]["mtp_export_report"]) in evidence
    assert Path(plan["paths"]["litertlm"]) in evidence


@pytest.mark.parametrize("gate", sorted(EXPORT_GATES))
@pytest.mark.parametrize("mutation", ["missing", "false", "truthy_not_bool"])
def test_export_rejects_missing_false_or_nonboolean_gate(plan, export_evidence, gate, mutation):
    report, _, _ = export_evidence
    if mutation == "missing":
        report["gates"].pop(gate)
    else:
        report["gates"][gate] = False if mutation == "false" else "true"
    _json(plan["paths"]["mtp_export_report"], report)
    with pytest.raises(ValueError):
        mtp.verify_mtp_export(plan)


@pytest.mark.parametrize("field", [
    "official_artifact", "assistant_checkpoint", "assistant_training_config",
    "package_input", "package_output",
])
def test_export_rejects_report_from_different_inputs(plan, export_evidence, field, tmp_path):
    report, _, _ = export_evidence
    report[field] = str(tmp_path / "other-run")
    _json(plan["paths"]["mtp_export_report"], report)
    with pytest.raises(ValueError):
        mtp.verify_mtp_export(plan)


@pytest.mark.parametrize("field,value", [
    ("assistant_trained", False), ("assistant_trained", "true"),
    ("assistant_weight_source", "official"), ("final_artifact_gate_pass", False),
    ("final_artifact_gate_pass", "true"),
])
def test_export_requires_actual_trained_assistant_report(plan, export_evidence, field, value):
    report, _, _ = export_evidence
    report[field] = value
    _json(plan["paths"]["mtp_export_report"], report)
    with pytest.raises(ValueError):
        mtp.verify_mtp_export(plan)


@pytest.mark.parametrize("artifact", ["target_litertlm", "litertlm"])
def test_export_rejects_tampered_or_swapped_package_bytes(plan, export_evidence, artifact):
    _write(plan["paths"][artifact], b"tampered")
    with pytest.raises(ValueError):
        mtp.verify_mtp_export(plan)


@pytest.mark.parametrize("field,value", [
    ("passed", False), ("executed", False),
    ("output_litertlm", "another-package.litertlm"), ("output_sha256", "0" * 64),
])
def test_export_still_requires_original_target_export_evidence(plan, export_evidence, field, value):
    _, target_report, _ = export_evidence
    target_report[field] = value
    _json(plan["paths"]["export_report"], target_report)
    with pytest.raises(ValueError):
        mtp.verify_mtp_export(plan)


def test_export_cannot_publish_if_training_source_verifier_fails(plan, export_evidence, monkeypatch):
    def reject(_):
        raise ValueError("Changed selected target conditioning source")

    monkeypatch.setattr(mtp, "verify_mtp_training", reject)
    with pytest.raises(ValueError, match="conditioning source"):
        mtp.verify_mtp_export(plan)


def test_export_rejects_checkpoint_provenance_swapped_after_conversion(plan, export_evidence):
    _json(Path(plan["paths"]["mtp_checkpoint"]) / "mtp_drafter_training_metadata.json",
          {"fixture": "different-assistant"})
    with pytest.raises(ValueError, match="provenance"):
        mtp.verify_mtp_export(plan)


@pytest.fixture
def training_evidence(plan, monkeypatch):
    from ir_training.train import mtp_drafter

    paths = plan["paths"]
    _json(paths["config"], {"model": {"chat_template_kwargs": {"enable_thinking": False}}})
    config = mtp.drafter_config(plan)
    _yaml(paths["mtp_config"], config)
    checkpoint = Path(paths["mtp_checkpoint"])
    _json(checkpoint / "config.json", {"model_type": "gemma4_assistant"})
    _json(checkpoint / "generation_config.json", {"use_cache": True})
    _write(checkpoint / "model.safetensors", b"trained fixture weights")
    # Real save_checkpoint also calls tokenizer.save_pretrained. These files
    # are valid, but the existing trainer's checkpoint manifest excludes them.
    _json(checkpoint / "tokenizer_config.json", {"fixture": "tokenizer"})
    _json(checkpoint / "tokenizer.json", {"fixture": "vocabulary"})
    bindings = {str(Path(paths["config"]).resolve()): _sha(paths["config"])}
    monkeypatch.setattr(mtp_drafter, "official_mobile_source_bindings", lambda _: bindings.copy())
    target_qat = {
        "enabled": True, "true_fake_quant": True,
        "wrapped_effective_lora_count": 205, "frozen_activation_module_count": 70,
    }
    report = {
        "ready": True, "training_executed": True, "global_steps": 1,
        "best_selection_loss": 0.5, "source_bindings": bindings.copy(),
        "target_qat": target_qat.copy(),
    }
    metadata = {
        "role": "best", "target_frozen": True,
        "target_model_id_or_path": paths["merged"],
        "training_config_sha256": _sha(paths["mtp_config"]),
        "source_bindings": bindings.copy(), "target_qat": target_qat.copy(),
        "checkpoint_files": mtp_drafter._checkpoint_manifest(checkpoint),
    }
    _json(paths["mtp_training_report"], report)
    _json(checkpoint / "mtp_drafter_training_metadata.json", metadata)
    return report, metadata, bindings


def test_training_verifier_accepts_real_manifest_scope_with_saved_tokenizer(plan, training_evidence):
    evidence = mtp.verify_mtp_training(plan)
    checkpoint = Path(plan["paths"]["mtp_checkpoint"])
    assert checkpoint / "model.safetensors" in evidence
    assert checkpoint / "tokenizer.json" in evidence
    assert checkpoint / "tokenizer_config.json" in evidence


def test_drafter_config_preserves_target_recipe_and_uses_selected_target_data(plan, training_evidence):
    paths = plan["paths"]
    target_bytes = Path(paths["config"]).read_bytes()
    base = load_yaml(ROOT / "configs/models/gemma4_e2b_mtp_drafter_qat.yaml")
    config = mtp.drafter_config(plan)
    assert config["qat"] == base["qat"]
    assert config["run"]["purpose"] == "official_mobile_mtp_training_v1"
    assert config["run"]["dataset_dir"] == paths["prepared"]
    assert config["run"]["output_dir"] == paths["mtp_training_dir"]
    assert config["target"]["model_id_or_path"] == paths["merged"]
    assert config["target"]["adapter_checkpoint"] == paths["best_checkpoint"]
    assert config["target"]["training_config"] == paths["config"]
    assert config["training"]["max_seq_length"] == plan["options"]["max_seq_length"]
    assert config["training"]["require_cuda"] is True
    assert config["training"]["overflow_policy"] == "error"
    assert config["training"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert Path(paths["config"]).read_bytes() == target_bytes


def test_drafter_config_requires_explicit_training_opt_in(plan):
    plan["mtp"]["training"] = False
    with pytest.raises(ValueError, match="train-mtp"):
        mtp.drafter_config(plan)


@pytest.mark.parametrize("field,value", [
    ("training_executed", False), ("training_executed", "true"), ("ready", False),
    ("global_steps", 0), ("global_steps", -1), ("global_steps", True), ("global_steps", 1.5),
    ("best_selection_loss", float("nan")), ("best_selection_loss", float("inf")),
    ("best_selection_loss", "0.5"),
])
def test_training_verifier_requires_completed_finite_training(plan, training_evidence, field, value):
    report, _, _ = training_evidence
    report[field] = value
    _json(plan["paths"]["mtp_training_report"], report)
    with pytest.raises(ValueError):
        mtp.verify_mtp_training(plan)


@pytest.mark.parametrize("field,value", [
    ("role", "final"), ("target_frozen", False), ("target_frozen", "true"),
    ("target_model_id_or_path", "wrong-selected-target"),
    ("training_config_sha256", "0" * 64),
])
def test_training_verifier_requires_selected_frozen_target(plan, training_evidence, field, value):
    _, metadata, _ = training_evidence
    metadata[field] = value
    _json(Path(plan["paths"]["mtp_checkpoint"]) / "mtp_drafter_training_metadata.json", metadata)
    with pytest.raises(ValueError):
        mtp.verify_mtp_training(plan)


@pytest.mark.parametrize("evidence", ["report", "metadata"])
@pytest.mark.parametrize("field,value", [
    ("enabled", False), ("true_fake_quant", False),
    ("wrapped_effective_lora_count", 204), ("frozen_activation_module_count", 69),
])
def test_training_verifier_requires_exact_retained_target_qat(plan, training_evidence, evidence, field, value):
    report, metadata, _ = training_evidence
    data = report if evidence == "report" else metadata
    data["target_qat"][field] = value
    destination = (plan["paths"]["mtp_training_report"] if evidence == "report" else
                   Path(plan["paths"]["mtp_checkpoint"]) / "mtp_drafter_training_metadata.json")
    _json(destination, data)
    with pytest.raises(ValueError):
        mtp.verify_mtp_training(plan)


@pytest.mark.parametrize("evidence", ["report", "metadata"])
def test_training_verifier_rejects_changed_conditioning_inputs(plan, training_evidence, evidence):
    report, metadata, _ = training_evidence
    data = report if evidence == "report" else metadata
    data["source_bindings"] = {"wrong-target": "0" * 64}
    destination = (plan["paths"]["mtp_training_report"] if evidence == "report" else
                   Path(plan["paths"]["mtp_checkpoint"]) / "mtp_drafter_training_metadata.json")
    _json(destination, data)
    with pytest.raises(ValueError, match="binding"):
        mtp.verify_mtp_training(plan)


@pytest.mark.parametrize("change", ["changed_weight", "extra_weight", "changed_config"])
def test_training_verifier_rejects_checkpoint_file_drift(plan, training_evidence, change):
    checkpoint = Path(plan["paths"]["mtp_checkpoint"])
    filename = {"changed_weight": "model.safetensors", "extra_weight": "model-02.safetensors",
                "changed_config": "config.json"}[change]
    _write(checkpoint / filename, b"changed")
    with pytest.raises(ValueError, match="manifest"):
        mtp.verify_mtp_training(plan)


def test_training_verifier_rejects_configuration_drift(plan, training_evidence):
    config = load_yaml(plan["paths"]["mtp_config"])
    config["training"]["learning_rate"] *= 2
    _yaml(plan["paths"]["mtp_config"], config)
    with pytest.raises(ValueError, match="opted-in plan"):
        mtp.verify_mtp_training(plan)


def test_generated_mtp_export_config_passes_reused_orchestrator_validation(plan, monkeypatch):
    from ir_training.pipeline import gemma4_mobile_mtp as exporter

    paths = plan["paths"]
    target = load_yaml(ROOT / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml")
    seed = Path(plan["options"]["model_dir"])
    target["model"].update(
        model_source=str(seed), tokenizer_source=str(seed),
        mobile_training_seed_manifest=str(seed / "mobile_training_seed_manifest.json"),
        mobile_qparams_contract=str(seed / "mobile_qparams.json"),
    )
    target["golden_eval"]["best_checkpoint_dir"] = paths["best_checkpoint"]
    _yaml(paths["config"], target)
    _yaml(paths["mtp_config"], mtp.drafter_config(plan))
    # Only large, hash-pinned real artifacts are replaced. The existing
    # orchestrator's complete config validation and command construction run.
    monkeypatch.setattr(exporter, "verify_configured_mobile_training_seed",
                        lambda *args, **kwargs: {"verified": True})
    monkeypatch.setattr(exporter, "verify_mobile_qparams_contract",
                        lambda *args, **kwargs: {"verified": True})
    config = mobile.mobile_export_config(plan)
    accepted = exporter.build_pipeline_plan(config, config_path=paths["mobile_config"])
    assert accepted["validation"] == {"ok": True, "issues": []}
    assert accepted["mtp"]["training"]["enabled"] is True
    assert accepted["mtp"]["exact_topology"]["enabled"] is True
    assert accepted["retained_scale_export"]["output_litertlm"] == paths["target_litertlm"]
    assert accepted["mtp"]["exact_topology"]["package_input"] == paths["target_litertlm"]
    assert accepted["package"]["output_litertlm"] == paths["litertlm"]
    assert accepted["retained_scale_export"]["preserves_default_mtp_byte_exact"] is True
    assert accepted["retained_scale_export"]["final_package_preserves_default_mtp_byte_exact"] is False
    assert accepted["android_gpu"]["required_modes"] == ["target_only", "mtp_on"]

    # Reproduce the integration bug: this switch refers to preserving the
    # intermediate template, not preserving the final assistant weights.
    config["pipeline"]["mtp"]["preserve_official_section"] = False
    rejected = exporter.build_pipeline_plan(config, config_path=paths["mobile_config"])
    assert rejected["validation"]["ok"] is False
    assert {issue["code"] for issue in rejected["validation"]["issues"]} == {
        "official_mtp_template_required",
    }
