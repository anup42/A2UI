from __future__ import annotations

import copy
import hashlib
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
    _validate_retained_scale_export_report,
    build_pipeline_plan,
    run_pipeline,
)
from ir_training.qat.mobile_training_seed import OFFICIAL_MOBILE_MODEL_ID
from ir_training.qat.retained_constants import verify_retained_constant_contract
from ir_training.qat_mtp.workflow import OFFICIAL_QAT_ASSISTANT, OFFICIAL_QAT_TARGET


def _use_unmaterialized_mobile_seed(
    config: dict, tmp_path: Path
) -> dict:
    """Keep plan tests independent of ignored workstation seed artifacts."""

    import yaml

    training_path = (
        ROOT
        / "configs"
        / "models"
        / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
    )
    training = copy.deepcopy(load_yaml(training_path))
    missing_seed = tmp_path / "unmaterialized_mobile_seed"
    training["model"]["model_source"] = str(missing_seed)
    training["model"]["mobile_training_seed_manifest"] = str(
        missing_seed / "mobile_training_seed_manifest.json"
    )
    temporary = tmp_path / "unmaterialized_mobile_training.yaml"
    temporary.write_text(yaml.safe_dump(training), encoding="utf-8")
    config["pipeline"]["training_config"] = str(temporary)
    return config


def test_mobile_mtp_pipeline_plan_is_qat_and_plan_only(tmp_path):
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = _use_unmaterialized_mobile_seed(
        copy.deepcopy(load_yaml(config_path)), tmp_path
    )
    plan = build_pipeline_plan(config, config_path=config_path)

    assert (
        plan["training"]["qat_profile"]
        == "gemma4_e2b_mobile_retained_scale_wna8o8"
    )
    assert plan["training"]["model_id"] == OFFICIAL_MOBILE_MODEL_ID
    assert plan["training"]["mobile_training_seed"]["verified"] is False
    assert plan["mtp"]["assistant_model_id"] == OFFICIAL_QAT_ASSISTANT
    assert plan["training"]["best_checkpoint_required"] is True
    assert plan["training"]["architecture_preflight"]["required"] is True
    assert plan["training"]["architecture_preflight"]["loads_weights"] is False
    assert "validate_gemma4_mobile_seed_architecture.py" in " ".join(
        plan["training"]["architecture_preflight"]["command"]
    )
    assert plan["package"]["mtp_enabled"] is True
    assert plan["package"]["mtp_model_type"] == "tf_lite_mtp_drafter"
    assert plan["android_gpu"]["required_modes"] == ["target_only", "mtp_on"]
    assert plan["android_gpu"]["target_only"]["mtp_flag"] is False
    assert plan["android_gpu"]["mtp_on"]["mtp_flag"] is True
    assert plan["android_gpu"]["warm_runs"] == 3
    assert plan["android_gpu"]["minimum_performance_warm_runs"] == 3
    assert plan["android_gpu"]["performance_selection_policy"] == (
        "median_of_all_warm_runs_all_must_be_valid"
    )
    assert "--mtp" not in plan["android_gpu"]["target_only"]["command"]
    assert "--mtp" in plan["android_gpu"]["mtp_on"]["command"]
    assert "--mtp-max-decode-overshoot" in plan["android_gpu"]["mtp_on"]["command"]
    assert (
        plan["android_gpu"]["target_only"]["output_dir"]
        != plan["android_gpu"]["mtp_on"]["output_dir"]
    )
    export = plan["retained_scale_export"]
    command_text = " ".join(export["command"])
    assert export["enabled"] is True
    assert export["mode"] == "retained_scale_code_only_v1"
    assert export["preserves_default_mtp_byte_exact"] is True
    assert export["requires_exact_205_projection_mapping"] is True
    assert export["legacy_absmax_export_blocked"] is True
    assert "build_gemma4_retained_scale_litertlm.py" in command_text
    assert "build_checkpoint_official_topology.py" not in command_text
    assert "--adapter-checkpoint" in export["command"]
    assert "--mobile-training-seed-manifest" in export["command"]
    assert "--mobile-qparams-contract" in export["command"]
    assert "--zero-adapter-checkpoint" in export["command"]
    assert "--report" in export["command"]
    assert "--execute" in export["command"]
    assert any(item["code"] == "missing_base_package" for item in plan["validation"]["issues"])
    assert any(
        item["code"] == "mobile_training_seed_unverified"
        for item in plan["validation"]["issues"]
    )


def test_mobile_mtp_pipeline_rejects_too_few_gpu_warm_runs(tmp_path):
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = _use_unmaterialized_mobile_seed(
        copy.deepcopy(load_yaml(config_path)), tmp_path
    )
    config["pipeline"]["android"]["warm_runs"] = 2

    plan = build_pipeline_plan(config, config_path=config_path)

    assert any(
        issue["code"] == "insufficient_android_gpu_warm_runs"
        for issue in plan["validation"]["issues"]
    )


def test_mobile_mtp_pipeline_rejects_non_qat_or_mismatched_seeds(tmp_path):
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = copy.deepcopy(load_yaml(config_path))
    training_path = (
        ROOT
        / "configs"
        / "models"
        / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
    )
    training = load_yaml(training_path)
    training["model"]["model_id"] = "google/gemma-4-E2B-it"

    temporary = tmp_path / "non_qat_e2b_training_seed.yaml"
    try:
        import yaml

        temporary.write_text(yaml.safe_dump(training), encoding="utf-8")
        config["pipeline"]["training_config"] = str(temporary)
        config["pipeline"]["mtp"]["assistant_model_id"] = (
            "google/gemma-4-E2B-it-assistant"
        )

        plan = build_pipeline_plan(config, config_path=config_path)
    finally:
        temporary.unlink(missing_ok=True)

    codes = {item["code"] for item in plan["validation"]["issues"]}
    assert "non_mobile_e2b_training_seed" in codes
    assert "retained_mobile_qat_contract_mismatch" not in codes
    assert "non_matching_qat_assistant_seed" not in codes
    assert plan["validation"]["ok"] is False


def test_mobile_mtp_pipeline_can_plan_trained_drafter_in_official_graph(tmp_path):
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = _use_unmaterialized_mobile_seed(
        copy.deepcopy(load_yaml(config_path)), tmp_path
    )
    config["pipeline"]["mtp"]["weight_source"] = "trained"
    config["pipeline"]["mtp"]["train_assistant"] = True

    plan = build_pipeline_plan(config, config_path=config_path)

    assert plan["validation"]["ok"] is False
    assert any(
        item["code"] == "mobile_training_seed_unverified"
        for item in plan["validation"]["issues"]
    )
    assert plan["mtp"]["weight_source"] == "trained"
    assert plan["mtp"]["official_weights_preserved"] is False
    assert plan["mtp"]["training"]["enabled"] is True
    assert plan["mtp"]["training"]["private_google_recipe_recovered"] is False
    assert "--execute" in plan["mtp"]["training"]["command"]
    assert plan["mtp"]["exact_topology"]["enabled"] is True
    assert plan["retained_scale_export"]["preserves_default_mtp_byte_exact"] is True
    assert (
        plan["retained_scale_export"]["final_package_preserves_default_mtp_byte_exact"]
        is False
    )
    assert "build_gemma4_mtp_drafter_official_topology.py" in " ".join(
        plan["mtp"]["exact_topology"]["command"]
    )
    assert (
        plan["retained_scale_export"]["output_litertlm"]
        == plan["mtp"]["exact_topology"]["package_input"]
    )
    assert plan["package"]["output_litertlm"] == plan["retained_scale_export"][
        "final_output_litertlm"
    ]
    assert "private" in plan["limitations"][0].lower()


def test_mobile_mtp_pipeline_can_disable_mtp_for_target_only_runtime(tmp_path):
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = _use_unmaterialized_mobile_seed(
        copy.deepcopy(load_yaml(config_path)), tmp_path
    )
    config["pipeline"]["mtp"]["enabled"] = False

    plan = build_pipeline_plan(config, config_path=config_path)
    codes = {item["code"] for item in plan["validation"]["issues"]}

    assert "mtp_disabled" not in codes
    assert "mtp_disabled_with_trained_drafter" not in codes
    assert plan["package"]["mtp_enabled"] is False
    assert plan["package"]["mtp_section_present"] is True
    assert plan["package"]["official_mtp_bytes_preserved"] is True
    assert plan["mtp"]["enabled"] is False
    assert plan["mtp"]["official_weights_preserved"] is True
    assert plan["mtp"]["training"]["enabled"] is False
    assert plan["mtp"]["exact_topology"]["enabled"] is False
    assert plan["retained_scale_export"]["preserves_default_mtp_byte_exact"] is True
    assert (
        plan["retained_scale_export"]["final_package_preserves_default_mtp_byte_exact"]
        is True
    )
    assert plan["android_gpu"]["required_modes"] == ["target_only"]
    assert plan["android_gpu"]["target_only"]["required"] is True
    assert plan["android_gpu"]["mtp_on"]["required"] is False
    assert plan["android_gpu"]["device_validation"] == (
        "target_only_required_after_packaging"
    )


def test_mobile_mtp_pipeline_rejects_trained_drafter_when_mtp_disabled(tmp_path):
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = _use_unmaterialized_mobile_seed(
        copy.deepcopy(load_yaml(config_path)), tmp_path
    )
    config["pipeline"]["mtp"].update(
        {"enabled": False, "weight_source": "trained", "train_assistant": True}
    )

    plan = build_pipeline_plan(config, config_path=config_path)
    codes = {item["code"] for item in plan["validation"]["issues"]}

    assert "mtp_disabled_with_trained_drafter" in codes
    assert plan["mtp"]["training"]["enabled"] is False
    assert plan["mtp"]["exact_topology"]["enabled"] is False


def test_official_mtp_bytes_do_not_require_transformers_assistant_identity(tmp_path):
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = _use_unmaterialized_mobile_seed(
        copy.deepcopy(load_yaml(config_path)), tmp_path
    )
    config["pipeline"]["mtp"]["assistant_model_id"] = "not-downloaded-in-official-mode"

    official = build_pipeline_plan(config, config_path=config_path)
    official_codes = {item["code"] for item in official["validation"]["issues"]}
    assert "non_matching_qat_assistant_seed" not in official_codes

    config["pipeline"]["mtp"]["weight_source"] = "trained"
    trained = build_pipeline_plan(config, config_path=config_path)
    trained_codes = {item["code"] for item in trained["validation"]["issues"]}
    assert "non_matching_qat_assistant_seed" in trained_codes


def test_retained_constant_contract_rejects_known_q4_cross_checkpoint_seed():
    contract = (
        ROOT
        / "configs"
        / "quantization"
        / "gemma4_e2b_mobile_observable_contract.yaml"
    )

    report = verify_retained_constant_contract(
        contract,
        family="gemma4_e2b",
        training_model_id=OFFICIAL_QAT_TARGET,
    )

    assert report["verified"] is False
    assert report["production_status"] == "compatible_exact"
    assert report["selection"]["selected_tensor_count"] == 262
    assert report["comparison"]["exact_tensor_count"] == 262
    assert report["comparison"]["value_mismatch_count"] == 0
    assert report["checks"]["training_seed_matches"] is False
    assert report["rejected_q4_seed_audit"]["value_mismatch_count"] == 212
    assert report["checks"]["compiled_graph_mapping_verified"] is True


def test_retained_constant_contract_requires_exact_values_and_compiled_mapping(
    tmp_path,
):
    contract = tmp_path / "retained.json"
    payload = {
        "retained_constant_compatibility": {
            "production_status": "compatible_exact",
            "dense_training_seed": {"repo_id": OFFICIAL_QAT_TARGET},
            "packed_mobile_checkpoint": {"repo_id": "google/mobile"},
            "selection": {"selected_tensor_count": 262},
            "comparison": {
                "schema_mismatch_count": 0,
                "exact_tensor_count": 262,
                "value_mismatch_count": 0,
                "all_selected_tensors_exact": True,
            },
            "compiled_graph_mapping_verified": True,
        }
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_retained_constant_contract(
        contract,
        family="gemma4_e2b",
        training_model_id=OFFICIAL_QAT_TARGET,
    )

    assert report["verified"] is True
    payload["retained_constant_compatibility"][
        "compiled_graph_mapping_verified"
    ] = False
    contract.write_text(json.dumps(payload), encoding="utf-8")
    rejected = verify_retained_constant_contract(
        contract,
        family="gemma4_e2b",
        training_model_id=OFFICIAL_QAT_TARGET,
    )
    assert rejected["verified"] is False


def test_retained_scale_stage_stops_before_conversion_without_mobile_seed(tmp_path):
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = _use_unmaterialized_mobile_seed(
        copy.deepcopy(load_yaml(config_path)), tmp_path
    )

    with pytest.raises(
        Gemma4MobileMTPPipelineError,
        match="mobile_training_seed_unverified",
    ):
        run_pipeline(
            config,
            config_path=config_path,
            execute_retained_scale_export=True,
        )


@pytest.mark.parametrize(
    "legacy_kwargs, expected",
    [
        ({"execute_exact_topology_export": True}, "retired Gemma 4 abs-max"),
        ({"execute_public_export": True}, "blocked for Gemma 4 retained_mobile"),
        ({"compose_package": True}, "--compose is blocked"),
    ],
)
def test_retained_mobile_pipeline_hard_blocks_legacy_export_routes(
    legacy_kwargs, expected
):
    with pytest.raises(Gemma4MobileMTPPipelineError, match=expected):
        run_pipeline({}, **legacy_kwargs)


def test_pipeline_overrides_bind_resolved_run_artifacts(tmp_path):
    import yaml

    pipeline_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = copy.deepcopy(load_yaml(pipeline_path))
    training = load_yaml(
        ROOT / "configs" / "models" / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
    )
    missing_seed = tmp_path / "seed"
    training["model"]["model_source"] = str(missing_seed)
    training["model"]["mobile_training_seed_manifest"] = str(
        missing_seed / "mobile_training_seed_manifest.json"
    )
    training["model"]["mobile_qparams_contract"] = str(
        missing_seed / "mobile_qparams.json"
    )
    resolved_config = tmp_path / "run" / "launch" / "resolved_training_config.yaml"
    resolved_config.parent.mkdir(parents=True)
    resolved_config.write_text(yaml.safe_dump(training), encoding="utf-8")
    best = tmp_path / "run" / "best_golden_checkpoint"
    official = tmp_path / "official.litertlm"
    merged = tmp_path / "export-r1" / "merged"
    export_dir = tmp_path / "export-r1" / "retained"
    report = export_dir / "report.json"
    output = export_dir / "candidate.litertlm"

    plan = build_pipeline_plan(
        config,
        config_path=pipeline_path,
        training_config_override=resolved_config,
        best_checkpoint_override=best,
        base_litertlm_override=official,
        merged_model_dir_override=merged,
        exact_output_dir_override=export_dir,
        export_report_override=report,
        output_litertlm_override=output,
    )

    export = plan["retained_scale_export"]
    command = export["command"]
    assert plan["training"]["config"] == str(resolved_config.resolve())
    assert plan["training"]["best_checkpoint"] == str(best.resolve())
    assert plan["merge"]["merged_model_dir"] == str(merged.resolve())
    assert export["official_litertlm"] == str(official.resolve())
    assert export["output_dir"] == str(export_dir.resolve())
    assert export["report"] == str(report.resolve())
    assert export["output_litertlm"] == str(output.resolve())
    assert command[command.index("--training-config") + 1] == str(
        resolved_config.resolve()
    )
    assert command[command.index("--adapter-checkpoint") + 1] == str(best.resolve())


_RETAINED_EXPORT_GATES = {
    "zero_adapter_target_byte_exact",
    "processed_205",
    "base_lora_candidate_dtype_match_205",
    "base_lora_candidate_bfloat16_205",
    "base_lora_numerical_parity_205",
    "base_lora_code_parity_205",
    "base_delta_norms_finite_205",
    "at_least_one_lora_delta_nonzero",
    "changed_buffers_within_expected_205",
    "at_least_one_trained_code_changed",
    "restored_official_payload_target_byte_exact",
    "frozen_72_byte_exact",
    "weight_qparams_byte_exact",
    "activation_a8_qparams_byte_exact",
    "all_tensor_qparams_byte_exact",
    "official_retained_weight_scales_exact",
    "official_retained_a8_scales_exact",
    "graph_layout_execution_identity",
    "section_size_unchanged",
    "exact_205_key_buffer_bijection",
    "expected_bit_histogram",
    "retained_qparams_verified",
    "legacy_metadata_rejected",
    "package_outside_target_byte_exact",
    "mtp_byte_exact",
    "package_parseable",
}


def _test_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_retained_export_plan(
    tmp_path: Path,
    *,
    merged: Path | None = None,
    output: Path | None = None,
    report: Path | None = None,
) -> dict:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    official = inputs / "official.litertlm"
    official.write_bytes(b"official-package")
    config = inputs / "resolved.yaml"
    config.write_text("run:\n  id: smoke\n", encoding="utf-8")
    seed_manifest = inputs / "mobile_training_seed_manifest.json"
    seed_manifest.write_text('{"verified": true}\n', encoding="utf-8")
    qparams = inputs / "mobile_qparams.json"
    qparams.write_text('{"verified": true}\n', encoding="utf-8")
    scale_storage = inputs / "mobile_qparams.safetensors"
    scale_storage.write_bytes(b"retained-scales")
    zero = inputs / "zero_seed"
    zero.mkdir(exist_ok=True)
    adapter = inputs / "best_golden_checkpoint"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "training_metadata.json").write_text(
        '{"checkpoint_role": "best_golden"}\n', encoding="utf-8"
    )
    export_dir = tmp_path / "fresh" / "retained"
    merged = merged or (tmp_path / "fresh" / "merged")
    output = output or (tmp_path / "fresh" / "candidate.litertlm")
    report = report or (export_dir / "report.json")
    return {
        "enabled": True,
        "official_litertlm": str(official.resolve()),
        "official_artifact_sha256": _test_sha256(official),
        "merged_checkpoint": str(merged.resolve()),
        "adapter_checkpoint": str(adapter.resolve()),
        "training_config": str(config.resolve()),
        "mobile_training_seed_manifest": str(seed_manifest.resolve()),
        "mobile_qparams_contract": str(qparams.resolve()),
        "mobile_qparams": {
            "scale_storage_path": str(scale_storage.resolve()),
            "scale_storage_sha256": _test_sha256(scale_storage),
        },
        "zero_adapter_checkpoint": str(zero.resolve()),
        "output_dir": str(export_dir.resolve()),
        "report": str(report.resolve()),
        "output_litertlm": str(output.resolve()),
        "command": ["retained-export"],
    }


def _write_passing_retained_export_report(
    report: Path, output: Path, expected_plan: dict
) -> None:
    official = Path(expected_plan["official_litertlm"])
    merged = Path(expected_plan["merged_checkpoint"])
    adapter = Path(expected_plan["adapter_checkpoint"])
    config = Path(expected_plan["training_config"])
    seed_manifest = Path(expected_plan["mobile_training_seed_manifest"])
    qparams = Path(expected_plan["mobile_qparams_contract"])
    scale_storage = Path(expected_plan["mobile_qparams"]["scale_storage_path"])
    zero = Path(expected_plan["zero_adapter_checkpoint"])
    merged.mkdir(parents=True, exist_ok=True)
    merged_model = merged / "model.safetensors"
    if not merged_model.exists():
        merged_model.write_bytes(b"merged")
    merged_metadata = merged / "qat_mtp_merge_metadata.json"
    merged_metadata.write_text('{"verified": true}\n', encoding="utf-8")
    adapter_files = [
        {
            "path": item.name,
            "size": item.stat().st_size,
            "sha256": _test_sha256(item),
        }
        for item in sorted(adapter.glob("adapter*"))
        if item.is_file()
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"retained-scale-package")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "mode": "retained_scale_code_only_v1",
                "executed": True,
                "plan_passed": True,
                "passed": True,
                "official_litertlm": str(official.resolve()),
                "official_artifact_sha256": _test_sha256(official),
                "official_artifact_identity": {
                    "verified": True,
                    "declared_sha256": _test_sha256(official),
                    "observed_sha256": _test_sha256(official),
                },
                "checkpoint": str(merged.resolve()),
                "merged_checkpoint_identity": {
                    "path": str(merged.resolve()),
                    "metadata_path": str(merged_metadata.resolve()),
                    "metadata_sha256": _test_sha256(merged_metadata),
                    "verified": True,
                    "file_verification": [
                        {
                            "path": merged_model.name,
                            "expected_size": merged_model.stat().st_size,
                            "expected_sha256": _test_sha256(merged_model),
                            "valid": True,
                        }
                    ],
                },
                "adapter_checkpoint": str(adapter.resolve()),
                "adapter_identity": {
                    "path": str(adapter.resolve()),
                    "training_metadata_sha256": _test_sha256(
                        adapter / "training_metadata.json"
                    ),
                    "files": adapter_files,
                    "verified": True,
                },
                "resolved_training_config_identity": {
                    "path": str(config.resolve()),
                    "sha256": _test_sha256(config),
                    "verified": True,
                },
                "mobile_training_seed_identity": {
                    "manifest_path": str(seed_manifest.resolve()),
                    "manifest_sha256": _test_sha256(seed_manifest),
                    "verified": True,
                },
                "mobile_qparams_identity": {
                    "contract_path": str(qparams.resolve()),
                    "contract_sha256": _test_sha256(qparams),
                    "scale_storage_path": str(scale_storage.resolve()),
                    "scale_storage_sha256": _test_sha256(scale_storage),
                    "verified": True,
                },
                "zero_adapter_checkpoint": str(zero.resolve()),
                "gates": {name: True for name in _RETAINED_EXPORT_GATES},
                "package_checks": {
                    "package_size_unchanged": True,
                    "package_prefix_byte_exact": True,
                    "package_suffix_byte_exact": True,
                    "candidate_target_section_written_exactly": True,
                    "mtp_byte_exact": True,
                    "output_package_parseable": True,
                },
                "output_litertlm": str(output.resolve()),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_retained_export_report_gate_rejects_any_false_gate(tmp_path):
    expected_plan = _synthetic_retained_export_plan(tmp_path)
    output = Path(expected_plan["output_litertlm"])
    report = Path(expected_plan["report"])
    _write_passing_retained_export_report(report, output, expected_plan)
    accepted = _validate_retained_scale_export_report(
        report, expected_plan=expected_plan
    )
    assert accepted["passed"] is True

    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["gates"]["mtp_byte_exact"] = False
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Gemma4MobileMTPPipelineError, match="mtp_byte_exact"):
        _validate_retained_scale_export_report(report, expected_plan=expected_plan)


def test_retained_export_report_rejects_cross_run_or_mutated_input(tmp_path):
    expected_plan = _synthetic_retained_export_plan(tmp_path)
    output = Path(expected_plan["output_litertlm"])
    report = Path(expected_plan["report"])
    _write_passing_retained_export_report(report, output, expected_plan)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["adapter_identity"]["path"] = str(tmp_path / "another_run" / "best")
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Gemma4MobileMTPPipelineError, match="input-or-output-identity"):
        _validate_retained_scale_export_report(report, expected_plan=expected_plan)

    _write_passing_retained_export_report(report, output, expected_plan)
    adapter_model = Path(expected_plan["adapter_checkpoint"]) / "adapter_model.safetensors"
    adapter_model.write_bytes(b"mutated-after-export")
    with pytest.raises(Gemma4MobileMTPPipelineError, match="input-or-output-identity"):
        _validate_retained_scale_export_report(report, expected_plan=expected_plan)


def _synthetic_combined_plan(tmp_path: Path) -> dict:
    merged = tmp_path / "fresh" / "merged"
    export_dir = tmp_path / "fresh" / "retained"
    report = export_dir / "report.json"
    output = tmp_path / "fresh" / "candidate.litertlm"
    retained = _synthetic_retained_export_plan(
        tmp_path, merged=merged, output=output, report=report
    )
    best = Path(retained["adapter_checkpoint"])
    return {
        "validation": {"issues": []},
        "training": {
            "command": [],
            "best_checkpoint": str(best),
            "best_checkpoint_ready": True,
            "config": retained["training_config"],
        },
        "merge": {
            "base_model_id": OFFICIAL_MOBILE_MODEL_ID,
            "base_model_source": retained["zero_adapter_checkpoint"],
            "mobile_training_seed_manifest": retained[
                "mobile_training_seed_manifest"
            ],
            "merged_model_dir": str(merged),
        },
        "retained_scale_export": retained,
        "mtp": {
            "enabled": True,
            "weight_source": "official",
            "training": {"enabled": False},
            "exact_topology": {"enabled": False},
        },
        "package": {"output_litertlm": str(output)},
        "android_gpu": {"required_modes": []},
    }


def test_combined_fresh_merge_then_retained_export_allows_initially_absent_merge(
    monkeypatch, tmp_path
):
    plan = _synthetic_combined_plan(tmp_path)
    merged = Path(plan["merge"]["merged_model_dir"])
    report = Path(plan["retained_scale_export"]["report"])
    output = Path(plan["retained_scale_export"]["output_litertlm"])
    assert not merged.exists()

    monkeypatch.setattr(
        "ir_training.pipeline.gemma4_mobile_mtp.build_pipeline_plan",
        lambda *args, **kwargs: plan,
    )

    def fake_merge(**kwargs):
        destination = Path(kwargs["output_dir"])
        destination.mkdir(parents=True)
        (destination / "model.safetensors").write_bytes(b"merged")
        return destination

    def fake_run(command, log_path, *, cwd):
        assert command == ["retained-export"]
        _write_passing_retained_export_report(
            report, output, plan["retained_scale_export"]
        )

    monkeypatch.setattr(
        "ir_training.pipeline.gemma4_mobile_mtp.merge_lora_adapter", fake_merge
    )
    monkeypatch.setattr(
        "ir_training.pipeline.gemma4_mobile_mtp._run_command", fake_run
    )

    result = run_pipeline(
        {"pipeline": {"source": {}}},
        execute_merge=True,
        execute_retained_scale_export=True,
    )

    assert merged.is_dir()
    assert result["merge"]["executed"] is True
    assert result["retained_scale_export"]["executed"] is True
    assert result["package"]["manifest"]["passed"] is True


def test_retained_export_refuses_existing_output_directory(monkeypatch, tmp_path):
    plan = _synthetic_combined_plan(tmp_path)
    merged = Path(plan["merge"]["merged_model_dir"])
    merged.mkdir(parents=True)
    output_dir = Path(plan["retained_scale_export"]["output_dir"])
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "ir_training.pipeline.gemma4_mobile_mtp.build_pipeline_plan",
        lambda *args, **kwargs: plan,
    )

    with pytest.raises(Gemma4MobileMTPPipelineError, match="choose fresh paths"):
        run_pipeline({}, execute_retained_scale_export=True)


def _write_device_report(
    path: Path,
    *,
    mtp: bool,
    mtp_acceptance: bool | None,
    candidate_sha256: str = "b" * 64,
) -> None:
    comparison = {
        "official_artifact_identity_verified": True,
        "candidate_artifact_identity_verified": True,
        "official_artifact_identity": {"host_sha256": "a" * 64},
        "candidate_artifact_identity": {"host_sha256": candidate_sha256},
        "structural_gpu_parity_pass": True,
        "throughput_sample_comparable": True,
        "throughput_gate_pass": True,
        "warm_run_gate_pass": True,
        "warm_structural_gate_pass": True,
        "overall_pass": True,
    }
    if mtp_acceptance is not None:
        comparison["mtp_acceptance_gate_pass"] = mtp_acceptance
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "mtp_enabled": mtp,
                "warm_run_count": 3,
                "performance_selection_policy": (
                    "median_of_all_warm_runs_all_must_be_valid"
                ),
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
    payload["schema_version"] = 4
    payload["comparison"]["mtp_acceptance_gate_pass"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Gemma4MobileMTPPipelineError, match="schema_v5_or_newer"):
        _load_android_gpu_report(report_path, mode="mtp_on", expected_mtp=True)


def test_mtp_package_validator_requires_paired_schema_v5_device_reports(
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
    candidate_path = tmp_path / "candidate.litertlm"
    candidate_path.write_bytes(b"candidate")
    candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    _write_device_report(
        target_report,
        mtp=False,
        mtp_acceptance=None,
        candidate_sha256=candidate_sha256,
    )
    _write_device_report(
        mtp_report,
        mtp=True,
        mtp_acceptance=True,
        candidate_sha256=candidate_sha256,
    )

    paired = validator.validate_package(
        candidate_path,
        target_only_device_report=target_report,
        device_report=mtp_report,
    )
    unpaired = validator.validate_package(
        candidate_path,
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
    assert result["execution_contract_complete"] is True
    assert result["execution_contract_match"] is True
