from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ir_training.common.config import load_yaml
from ir_training.export.litertlm_mtp import (
    LiteRTLMMTPPackagingError,
    _replace_same_size_section,
    compare_target_layout,
)
from ir_training.pipeline.gemma4_mobile_mtp import (
    Gemma4MobileMTPPipelineError,
    _load_android_gpu_report,
    build_pipeline_plan,
)


def test_mobile_mtp_pipeline_plan_is_qat_and_plan_only():
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = load_yaml(config_path)
    plan = build_pipeline_plan(config, config_path=config_path)

    assert plan["training"]["qat_profile"] == "gemma4_e2b_mobile_observable_wna8o8_approx"
    assert plan["training"]["best_checkpoint_required"] is True
    assert plan["package"]["mtp_enabled"] is True
    assert plan["package"]["mtp_model_type"] == "tf_lite_mtp_drafter"
    assert plan["android_gpu"]["required_modes"] == ["target_only", "mtp_on"]
    assert plan["android_gpu"]["target_only"]["mtp_flag"] is False
    assert plan["android_gpu"]["mtp_on"]["mtp_flag"] is True
    assert "--mtp" not in plan["android_gpu"]["target_only"]["command"]
    assert "--mtp" in plan["android_gpu"]["mtp_on"]["command"]
    assert "--mtp-max-decode-overshoot" in plan["android_gpu"]["mtp_on"]["command"]
    assert (
        plan["android_gpu"]["target_only"]["output_dir"]
        != plan["android_gpu"]["mtp_on"]["output_dir"]
    )
    assert plan["exact_topology"]["enabled"] is True
    assert plan["exact_topology"]["family"] == "gemma4_e2b"
    assert plan["exact_topology"]["preserves_default_mtp_byte_exact"] is True
    assert "--execute" in plan["exact_topology"]["command"]
    assert any(item["code"] == "missing_base_package" for item in plan["validation"]["issues"])


def test_mobile_mtp_pipeline_can_plan_trained_drafter_in_official_graph():
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = copy.deepcopy(load_yaml(config_path))
    config["pipeline"]["mtp"]["weight_source"] = "trained"
    config["pipeline"]["mtp"]["train_assistant"] = True

    plan = build_pipeline_plan(config, config_path=config_path)

    assert plan["validation"]["ok"] is True
    assert plan["mtp"]["weight_source"] == "trained"
    assert plan["mtp"]["official_weights_preserved"] is False
    assert plan["mtp"]["training"]["enabled"] is True
    assert plan["mtp"]["training"]["private_google_recipe_recovered"] is False
    assert "--execute" in plan["mtp"]["training"]["command"]
    assert plan["mtp"]["exact_topology"]["enabled"] is True
    assert plan["exact_topology"]["preserves_default_mtp_byte_exact"] is True
    assert (
        plan["exact_topology"]["final_package_preserves_default_mtp_byte_exact"]
        is False
    )
    assert "build_gemma4_mtp_drafter_official_topology.py" in " ".join(
        plan["mtp"]["exact_topology"]["command"]
    )
    assert (
        plan["exact_topology"]["output_litertlm"]
        == plan["mtp"]["exact_topology"]["package_input"]
    )
    assert plan["package"]["output_litertlm"] == plan["exact_topology"][
        "final_output_litertlm"
    ]
    assert "private" in plan["limitations"][0].lower()


def _write_device_report(path: Path, *, mtp: bool, mtp_acceptance: bool | None) -> None:
    comparison = {
        "structural_gpu_parity_pass": True,
        "throughput_sample_comparable": True,
        "throughput_gate_pass": True,
        "overall_pass": True,
    }
    if mtp_acceptance is not None:
        comparison["mtp_acceptance_gate_pass"] = mtp_acceptance
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mtp_enabled": mtp,
                "comparison": comparison,
            }
        ),
        encoding="utf-8",
    )


def test_android_gpu_pipeline_report_gate_requires_target_only_and_mtp_evidence(tmp_path):
    target_report = tmp_path / "target.json"
    mtp_report = tmp_path / "mtp.json"
    _write_device_report(target_report, mtp=False, mtp_acceptance=None)
    _write_device_report(mtp_report, mtp=True, mtp_acceptance=True)

    target = _load_android_gpu_report(
        target_report, mode="target_only", expected_mtp=False
    )
    mtp = _load_android_gpu_report(mtp_report, mode="mtp_on", expected_mtp=True)

    assert all(target["pipeline_gate_checks"].values())
    assert all(mtp["pipeline_gate_checks"].values())


def test_android_gpu_pipeline_report_gate_rejects_old_or_wrong_mode_reports(tmp_path):
    report_path = tmp_path / "report.json"
    _write_device_report(report_path, mtp=True, mtp_acceptance=False)

    with pytest.raises(Gemma4MobileMTPPipelineError, match="required gates"):
        _load_android_gpu_report(
            report_path, mode="target_only", expected_mtp=False
        )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload["comparison"]["mtp_acceptance_gate_pass"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Gemma4MobileMTPPipelineError, match="schema_v2_or_newer"):
        _load_android_gpu_report(report_path, mode="mtp_on", expected_mtp=True)


def test_mtp_package_validator_requires_paired_schema_v2_device_reports(
    monkeypatch, tmp_path
):
    import validate_litertlm_mtp_gpu as validator

    fake_package_report = {
        "sections": [
            {
                "data_type_name": "TFLiteModel",
                "items": [
                    {"key": "model_type", "value": "tf_lite_mtp_drafter"}
                ],
                "alignment_ok": True,
                "ordered_after_previous": True,
                "index": 0,
            }
        ],
        "graphs": [],
    }
    monkeypatch.setattr(
        validator, "inspect_litertlm", lambda *args, **kwargs: fake_package_report
    )
    target_report = tmp_path / "target.json"
    mtp_report = tmp_path / "mtp.json"
    _write_device_report(target_report, mtp=False, mtp_acceptance=None)
    _write_device_report(mtp_report, mtp=True, mtp_acceptance=True)

    paired = validator.validate_package(
        tmp_path / "candidate.litertlm",
        target_only_device_report=target_report,
        device_report=mtp_report,
    )
    unpaired = validator.validate_package(
        tmp_path / "candidate.litertlm",
        device_report=mtp_report,
    )

    assert paired["ok"] is True
    assert paired["checks"]["dual_mode_gpu_device_validation"] is True
    assert unpaired["ok"] is False
    assert any("Provide both" in error for error in unpaired["errors"])


def test_replace_same_size_section_preserves_prefix_and_suffix(tmp_path):
    source = tmp_path / "base.litertlm"
    source.write_bytes(b"prefix" + b"target" + b"suffix")
    output = tmp_path / "out.litertlm"
    _replace_same_size_section(
        source,
        {"begin_offset": 6, "size": 6},
        b"packed",
        output,
    )
    assert output.read_bytes() == b"prefix" + b"packed" + b"suffix"


def test_replace_same_size_section_rejects_size_change(tmp_path):
    source = tmp_path / "base.litertlm"
    source.write_bytes(b"prefix" + b"target" + b"suffix")
    with pytest.raises(LiteRTLMMTPPackagingError, match="header rewrite"):
        _replace_same_size_section(
            source,
            {"begin_offset": 6, "size": 6},
            b"larger-target",
            tmp_path / "out.litertlm",
        )


def test_compare_target_layout_ignores_buffer_indices_but_requires_tflite_graphs():
    pytest.importorskip("flatbuffers")
    pytest.importorskip("ai_edge_litert.schema_py_generated")
    from build_fresh_random_quantized_graph import _build_fresh_tflite

    records = [
        {
            "ordinal": 0,
            "operator": "FULLY_CONNECTED",
            "bits": 8,
            "shape": [4, 3],
            "type_value": 9,
        }
    ]
    graph = _build_fresh_tflite(records, seed=7)
    result = compare_target_layout(graph, graph)
    assert result["available"] is True
    assert result["ok"] is True
    assert result["structural_match"] is True
    assert result["quantization_layout_match"] is True
    assert result["buffer_storage_match"] is True
