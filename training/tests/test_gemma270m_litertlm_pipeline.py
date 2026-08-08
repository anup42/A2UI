from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ir_training.common.config import load_yaml
from ir_training.pipeline.gemma270m_litertlm import build_pipeline_plan


def test_gemma270m_plan_selects_best_checkpoint_and_disables_mtp():
    config_path = ROOT / "configs" / "pipelines" / "gemma3_270m_qat_litertlm.yaml"
    config = load_yaml(config_path)
    plan = build_pipeline_plan(config, config_path=config_path)

    assert plan["training"]["model_id"] == "google/gemma-3-270m-it"
    assert plan["training"]["qat_profile"] == "gemma3_270m_wi8_afp32"
    assert plan["training"]["best_checkpoint_required"] is True
    assert plan["export"]["recipe"] == "dynamic_wi8_afp32"
    assert plan["package"]["mtp"]["enabled"] is False
    assert plan["package"]["mtp"]["status"] == "not_applicable_for_gemma3_270m"
    assert plan["android_gpu"]["mtp_flag"] is False
    assert plan["exact_topology"]["enabled"] is True
    assert plan["exact_topology"]["family"] == "gemma3_270m"
    assert plan["exact_topology"]["model_type"] == "TF_LITE_PREFILL_DECODE"
    assert "--execute" in plan["exact_topology"]["command"]
    assert plan["validation"]["ok"] is True


def test_gemma270m_plan_rejects_mtp_configuration():
    config_path = ROOT / "configs" / "pipelines" / "gemma3_270m_qat_litertlm.yaml"
    config = load_yaml(config_path)
    config["pipeline"]["mtp"]["enabled"] = True
    plan = build_pipeline_plan(config, config_path=config_path)

    assert plan["validation"]["ok"] is False
    assert any(
        issue["code"] == "mtp_not_applicable"
        for issue in plan["validation"]["issues"]
    )


def test_gemma270m_validator_accepts_prefill_without_mtp(monkeypatch, tmp_path):
    import validate_gemma270m_litertlm as validator

    fake_report = {
        "sections": [
            {
                "data_type_name": "TFLiteModel",
                "items": [
                    {"key": "model_type", "value": "tf_lite_prefill_decode"}
                ],
                "alignment_ok": True,
                "ordered_after_previous": True,
                "index": 0,
                "sha256": "target",
            }
        ],
        "graphs": [],
    }
    monkeypatch.setattr(validator, "inspect_litertlm", lambda *args, **kwargs: fake_report)
    result = validator.validate_package(tmp_path / "candidate.litertlm")

    assert result["ok"] is True
    assert result["checks"]["prefill_decode_present"] is True
    assert result["checks"]["mtp_not_required"] is True


def test_gemma270m_validator_accepts_schema_v4_gpu_parity_report(monkeypatch, tmp_path):
    import validate_gemma270m_litertlm as validator

    fake_report = {
        "sections": [
            {
                "data_type_name": "TFLiteModel",
                "items": [
                    {"key": "model_type", "value": "tf_lite_prefill_decode"}
                ],
                "alignment_ok": True,
                "ordered_after_previous": True,
                "index": 0,
            }
        ],
        "graphs": [],
    }
    monkeypatch.setattr(validator, "inspect_litertlm", lambda *args, **kwargs: fake_report)
    candidate_path = tmp_path / "candidate.litertlm"
    candidate_path.write_bytes(b"candidate")
    candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    official_path = tmp_path / "official.litertlm"
    official_path.write_bytes(b"official")
    official_sha256 = hashlib.sha256(official_path.read_bytes()).hexdigest()
    device_report = tmp_path / "device.json"
    device_report.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "mtp_enabled": False,
                "comparison": {
                    "official_artifact_identity_verified": True,
                    "candidate_artifact_identity_verified": True,
                    "official_artifact_identity": {
                        "host_sha256": official_sha256
                    },
                    "candidate_artifact_identity": {
                        "host_sha256": candidate_sha256
                    },
                    "structural_gpu_parity_pass": True,
                    "throughput_sample_comparable": True,
                    "throughput_gate_pass": True,
                    "overall_pass": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = validator.validate_package(
        candidate_path,
        device_report=device_report,
        official_artifact=official_path,
    )

    assert result["ok"] is True
    assert result["checks"]["gpu_device_validation"] is True
    assert result["device"]["pipeline_gate_checks"]["throughput_gate"] is True

    candidate_path.write_bytes(b"candidate-tampered-after-device-run")
    tampered = validator.validate_package(
        candidate_path,
        device_report=device_report,
        official_artifact=official_path,
    )
    assert tampered["ok"] is False
    assert any(
        "candidate_artifact_sha256_matches_pipeline_output" in error
        for error in tampered["errors"]
    )

    candidate_path.write_bytes(b"candidate")
    official_path.write_bytes(b"official-tampered-after-device-run")
    wrong_reference = validator.validate_package(
        candidate_path,
        device_report=device_report,
        official_artifact=official_path,
    )
    assert wrong_reference["ok"] is False
    assert any(
        "official_artifact_sha256_matches_pipeline_reference" in error
        for error in wrong_reference["errors"]
    )
