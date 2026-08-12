"""Gemma 4 E2B mobile QAT -> best checkpoint -> MTP package orchestration.

The pipeline is deliberately split into explicit stages:

1. QAT SFT (the existing ``train_sft.py`` path) saves a golden-set best
   adapter;
2. the best adapter is merged back into the floating-point base model;
3. the dedicated retained-scale exporter changes only the 205 trained target
   projection code buffers in a copy of the released package;
4. MTP weights use either official bytes or a trained 23-matrix transplant;
5. legacy abs-max/public target export and section composition are blocked for
   this retained-scale profile; and
6. package/device validation records what was actually proven.

The default command is plan-only.  Training and public conversion are
expensive and the public converter is not Google's private Gemma mobile
exporter, so neither is run unless explicitly requested.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.eval.android_gpu_report import (
    AndroidGpuParityReportError,
    MIN_PERFORMANCE_WARM_RUNS,
    PERFORMANCE_SELECTION_POLICY,
    load_android_gpu_parity_report,
)
from ir_training.export.merge_lora import merge_lora_adapter
from ir_training.qat.mobile_training_seed import (
    OFFICIAL_MOBILE_MODEL_ID,
    verify_configured_mobile_training_seed,
)
from ir_training.qat.mobile_qparams import verify_mobile_qparams_contract
from ir_training.qat_mtp.workflow import OFFICIAL_QAT_ASSISTANT


class Gemma4MobileMTPPipelineError(RuntimeError):
    """Raised when a pipeline stage cannot be completed safely."""


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _path_or_empty(value: Any, base: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return resolve_path(str(value), base)


def _best_checkpoint_dir(training_config: dict[str, Any], base: Path) -> Path:
    run = _section(training_config, "run")
    golden = _section(training_config, "golden_eval")
    configured = golden.get("best_checkpoint_dir")
    if configured:
        return resolve_path(str(configured), base)
    return resolve_path(
        str(run.get("output_dir", "runs/gemma4_e2b_mobile_seed_ir_qat_sft"))
        + "/best_golden_checkpoint",
        base,
    )


def _checkpoint_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    marker_names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
        "config.json",
    }
    return any((path / name).is_file() for name in marker_names)


def _find_single_litertlm(directory: Path) -> Path | None:
    files = sorted(directory.rglob("*.litertlm")) if directory.is_dir() else []
    return files[0] if len(files) == 1 else None


def build_pipeline_plan(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    training_config_override: str | Path | None = None,
    best_checkpoint_override: str | Path | None = None,
    base_litertlm_override: str | Path | None = None,
    merged_model_dir_override: str | Path | None = None,
    exact_output_dir_override: str | Path | None = None,
    export_report_override: str | Path | None = None,
    target_litertlm_override: str | Path | None = None,
    target_section_override: str | Path | None = None,
    output_litertlm_override: str | Path | None = None,
) -> dict[str, Any]:
    """Build a no-download/no-training plan for the complete workflow."""

    base = training_root()
    pipeline_cfg = _section(config, "pipeline")
    training_config_value = pipeline_cfg.get(
        "training_config",
        "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml",
    )
    training_config_path = resolve_path(
        str(training_config_override or training_config_value), base
    )
    if not training_config_path.is_file():
        raise Gemma4MobileMTPPipelineError(
            f"Training config does not exist: {training_config_path}"
        )
    training_config = load_yaml(training_config_path)
    model_cfg = _section(training_config, "model")
    qat_cfg = _section(training_config, "qat")
    train_run_cfg = _section(training_config, "run")
    training_output = resolve_path(
        str(
            train_run_cfg.get(
                "output_dir", "runs/gemma4_e2b_mobile_seed_ir_qat_sft"
            )
        ),
        base,
    )
    best_checkpoint = (
        resolve_path(str(best_checkpoint_override), base)
        if best_checkpoint_override
        else _best_checkpoint_dir(training_config, base)
    )

    source_cfg = _section(pipeline_cfg, "source")
    merged_model_dir = (
        resolve_path(str(merged_model_dir_override), base)
        if merged_model_dir_override
        else _path_or_empty(source_cfg.get("merged_model_dir"), base)
    ) or resolve_path(
        str(pipeline_cfg.get("output_dir", "outputs/pipelines/gemma4_e2b_mobile_mtp"))
        + "/merged_best_hf",
        base,
    )
    base_litertlm = (
        resolve_path(str(base_litertlm_override), base)
        if base_litertlm_override
        else _path_or_empty(source_cfg.get("base_litertlm"), base)
    )
    target_litertlm = (
        resolve_path(str(target_litertlm_override), base)
        if target_litertlm_override
        else _path_or_empty(source_cfg.get("target_litertlm"), base)
    )
    target_section = (
        resolve_path(str(target_section_override), base)
        if target_section_override
        else _path_or_empty(source_cfg.get("target_section"), base)
    )
    output_litertlm = (
        resolve_path(str(output_litertlm_override), base)
        if output_litertlm_override
        else _path_or_empty(source_cfg.get("output_litertlm"), base)
    )
    public_export_cfg = _section(pipeline_cfg, "public_export")
    public_export_dir = _path_or_empty(public_export_cfg.get("output_dir"), base)
    if public_export_dir is None:
        public_export_dir = resolve_path(
            str(pipeline_cfg.get("output_dir", "outputs/pipelines/gemma4_e2b_mobile_mtp"))
            + "/public_export",
            base,
        )
    retained_export_cfg = _section(pipeline_cfg, "retained_scale_export")
    retained_export_enabled = bool(retained_export_cfg.get("enabled", True))
    retained_output_dir = (
        resolve_path(str(exact_output_dir_override), base)
        if exact_output_dir_override
        else _path_or_empty(retained_export_cfg.get("output_dir"), base)
    )
    if retained_output_dir is None:
        retained_output_dir = resolve_path(
            str(pipeline_cfg.get("output_dir", "outputs/pipelines/gemma4_e2b_mobile_mtp"))
            + "/retained_scale_export",
            base,
        )
    if output_litertlm is None:
        output_litertlm = retained_output_dir / "gemma4_e2b_retained_scale.litertlm"
    export_report = (
        resolve_path(str(export_report_override), base)
        if export_report_override
        else _path_or_empty(retained_export_cfg.get("report"), base)
    ) or (retained_output_dir / "gemma4_retained_scale_code_only_report.json")

    training_script = base / "scripts" / "train_sft.py"
    train_command = [sys.executable, str(training_script), "--config", str(training_config_path)]
    architecture_preflight_script = (
        base / "scripts" / "validate_gemma4_mobile_seed_architecture.py"
    )
    architecture_preflight_output = resolve_path(
        str(
            pipeline_cfg.get(
                "output_dir", "outputs/pipelines/gemma4_e2b_mobile_mtp"
            )
        )
        + "/mobile_seed_architecture_report.json",
        base,
    )
    architecture_preflight_command = [
        sys.executable,
        str(architecture_preflight_script),
        "--training-config",
        str(training_config_path),
        "--output",
        str(architecture_preflight_output),
    ]
    export_config_value = public_export_cfg.get(
        "config", "configs/export/edge_gallery_gemma4_e2b.yaml"
    )
    export_config_path = resolve_path(str(export_config_value), base)
    model_id = str(model_cfg.get("model_id") or "")
    model_source = _path_or_empty(model_cfg.get("model_source"), base)
    mobile_training_seed_manifest = _path_or_empty(
        model_cfg.get("mobile_training_seed_manifest"), base
    )
    mobile_qparams_contract = _path_or_empty(
        model_cfg.get("mobile_qparams_contract"), base
    )
    mobile_training_seed = verify_configured_mobile_training_seed(
        model_cfg,
        base=base,
        require_materialized=True,
    )
    mtp_cfg = _section(pipeline_cfg, "mtp")
    mtp_enabled = bool(mtp_cfg.get("enabled", True))
    train_assistant = bool(mtp_cfg.get("train_assistant", False))
    target_model_type = str(
        mtp_cfg.get("target_model_type", "tf_lite_prefill_decode")
    )
    mtp_model_type = str(mtp_cfg.get("model_type", "tf_lite_mtp_drafter"))
    mtp_weight_source = str(mtp_cfg.get("weight_source", "official")).strip().lower()
    trained_mtp_enabled = mtp_enabled and mtp_weight_source == "trained"
    drafter_training_config_path = resolve_path(
        str(
            mtp_cfg.get("training_config")
            or "configs/models/gemma4_e2b_mtp_drafter_qat.yaml"
        ),
        base,
    )
    drafter_checkpoint = _path_or_empty(mtp_cfg.get("trained_checkpoint"), base)
    if drafter_checkpoint is None:
        drafter_checkpoint = resolve_path(
            "runs/gemma4_e2b_mtp_drafter_qat/best_checkpoint", base
        )
    target_intermediate_litertlm = _path_or_empty(
        mtp_cfg.get("target_intermediate_litertlm"), base
    ) or (retained_output_dir / "gemma4_e2b_target_with_official_mtp.litertlm")
    drafter_exact_output_dir = _path_or_empty(
        mtp_cfg.get("exact_topology_output_dir"), base
    ) or (retained_output_dir / "drafter")
    target_retained_output_dir = retained_output_dir
    target_retained_package_output = (
        target_intermediate_litertlm
        if trained_mtp_enabled
        else output_litertlm
    )
    public_export_enabled = bool(public_export_cfg.get("enabled", False))
    official_artifact_sha256 = str(
        retained_export_cfg.get("official_artifact_sha256") or ""
    ).strip().lower()
    mobile_qparams = verify_mobile_qparams_contract(
        mobile_qparams_contract, base=base
    )
    retained_export_script = (
        base / "scripts" / "build_gemma4_retained_scale_litertlm.py"
    )
    retained_export_command = [
        sys.executable,
        str(retained_export_script),
        "--official-litertlm",
        str(base_litertlm) if base_litertlm else "<official-litertlm-required>",
        "--official-artifact-sha256",
        official_artifact_sha256 or "<official-artifact-sha256-required>",
        "--checkpoint",
        str(merged_model_dir),
        "--adapter-checkpoint",
        str(best_checkpoint),
        "--training-config",
        str(training_config_path),
        "--mobile-training-seed-manifest",
        str(mobile_training_seed_manifest)
        if mobile_training_seed_manifest
        else "<mobile-training-seed-manifest-required>",
        "--mobile-qparams-contract",
        str(mobile_qparams_contract)
        if mobile_qparams_contract
        else "<mobile-qparams-contract-required>",
        "--zero-adapter-checkpoint",
        str(model_source) if model_source else "<zero-adapter-checkpoint-required>",
        "--output-dir",
        str(target_retained_output_dir),
        "--output-litertlm",
        str(target_retained_package_output),
        "--report",
        str(export_report),
        "--execute",
    ]
    drafter_training_script = base / "scripts" / "train_gemma4_mtp_drafter.py"
    drafter_training_command = [
        sys.executable,
        str(drafter_training_script),
        "--config",
        str(drafter_training_config_path),
        "--target-model",
        str(merged_model_dir),
        "--execute",
    ]
    drafter_export_script = (
        base / "scripts" / "build_gemma4_mtp_drafter_official_topology.py"
    )
    drafter_export_command = [
        sys.executable,
        str(drafter_export_script),
        str(base_litertlm) if base_litertlm else "<official-base-litertlm-required>",
        "--package-input",
        str(target_intermediate_litertlm),
        "--assistant-checkpoint",
        str(drafter_checkpoint),
        "--assistant-training-config",
        str(drafter_training_config_path),
        "--official-assistant-model-id",
        str(mtp_cfg.get("assistant_model_id") or OFFICIAL_QAT_ASSISTANT),
        "--official-artifact-sha256",
        official_artifact_sha256 or "<official-artifact-sha256-required>",
        "--output-dir",
        str(drafter_exact_output_dir),
        "--package-output",
        str(output_litertlm),
        "--calibration-samples",
        str(int(mtp_cfg.get("calibration_samples", 2))),
        "--threads",
        str(int(mtp_cfg.get("threads", 1))),
        "--execute",
    ]
    converter_batch_size = mtp_cfg.get("converter_batch_size")
    if converter_batch_size not in (None, "", 0):
        drafter_export_command.extend(
            ["--converter-batch-size", str(int(converter_batch_size))]
        )
    if bool(mtp_cfg.get("retain_intermediates", False)):
        drafter_export_command.append("--retain-intermediates")
    if bool(mtp_cfg.get("runtime_allocate", False)):
        drafter_export_command.extend(
            ["--runtime-allocate", "--runtime-threads", str(int(mtp_cfg.get("runtime_threads", 2)))]
        )
        if bool(mtp_cfg.get("runtime_without_default_delegates", True)):
            drafter_export_command.append("--runtime-without-default-delegates")
    android_cfg = _section(pipeline_cfg, "android")
    android_warm_runs = int(
        android_cfg.get("warm_runs", MIN_PERFORMANCE_WARM_RUNS)
    )
    android_output_dir = _path_or_empty(android_cfg.get("output_dir"), base)
    if android_output_dir is None:
        android_output_dir = resolve_path(
            str(pipeline_cfg.get("output_dir", "outputs/pipelines/gemma4_e2b_mobile_mtp"))
            + "/android_gpu_parity",
            base,
        )
    android_runner = str(base / "scripts" / "benchmark_android_litertlm_gpu_parity.py")

    def android_command(*, mtp: bool, output_dir: Path) -> list[str]:
        command = [
            sys.executable,
            android_runner,
            "--official",
            str(base_litertlm)
            if base_litertlm
            else "<official-base-litertlm-required>",
            "--candidate",
            str(output_litertlm),
            "--output-dir",
            str(output_dir),
            "--max-num-tokens",
            str(int(android_cfg.get("max_num_tokens", 2048))),
            "--output-tokens",
            str(int(android_cfg.get("output_tokens", 64))),
            "--warm-runs",
            str(android_warm_runs),
            "--max-throughput-regression-percent",
            str(float(android_cfg.get("max_throughput_regression_percent", 10.0))),
            "--top-k",
            str(int(android_cfg.get("top_k", 1))),
            "--top-p",
            str(float(android_cfg.get("top_p", 1.0))),
            "--temperature",
            str(float(android_cfg.get("temperature", 0.0))),
            "--seed",
            str(int(android_cfg.get("seed", 42))),
        ]
        if mtp:
            command.extend(
                [
                    "--mtp",
                    "--max-mtp-success-rate-drop",
                    str(float(android_cfg.get("max_mtp_success_rate_drop", 0.10))),
                    "--mtp-max-decode-overshoot",
                    str(int(android_cfg.get("mtp_max_decode_overshoot", 4))),
                ]
            )
        if str(android_cfg.get("prompt") or "").strip():
            command.extend(["--prompt", str(android_cfg["prompt"])])
        return command

    target_only_output_dir = android_output_dir / "target_only"
    mtp_on_output_dir = android_output_dir / "mtp_on"
    target_only_command = android_command(
        mtp=False, output_dir=target_only_output_dir
    )
    mtp_on_command = android_command(mtp=True, output_dir=mtp_on_output_dir)

    validation: list[dict[str, str]] = []
    assistant_model_id = str(
        mtp_cfg.get("assistant_model_id") or OFFICIAL_QAT_ASSISTANT
    )
    if model_id != OFFICIAL_MOBILE_MODEL_ID:
        validation.append(
            {
                "severity": "error",
                "code": "non_mobile_e2b_training_seed",
                "message": (
                    "The mobile QAT pipeline must retain Google's packed mobile "
                    f"identity {OFFICIAL_MOBILE_MODEL_ID}; observed "
                    f"{model_id or '<missing>'}."
                ),
            }
        )
    if not mobile_training_seed["verified"]:
        validation.append(
            {
                "severity": "error",
                "code": "mobile_training_seed_unverified",
                "message": (
                    "Materialize and verify the hash-bound BF16 text seed before "
                    "training, merge, or exact-topology export."
                ),
            }
        )
    if not mobile_qparams.get("verified"):
        validation.append(
            {
                "severity": "error",
                "code": "mobile_qparams_unverified",
                "scope": "retained_scale_export",
                "message": (
                    "The hash-bound retained weight/A8 qparams sidecar must verify "
                    "before merge or retained-scale export."
                ),
            }
        )
    if trained_mtp_enabled and assistant_model_id != OFFICIAL_QAT_ASSISTANT:
        validation.append(
            {
                "severity": "error",
                "code": "non_matching_qat_assistant_seed",
                "message": (
                    "Use Google's matching QAT assistant seed "
                    f"{OFFICIAL_QAT_ASSISTANT}; observed {assistant_model_id or '<missing>'}."
                ),
            }
        )
    if not bool(qat_cfg.get("enabled", False)):
        validation.append(
            {
                "severity": "error",
                "code": "qat_disabled",
                "message": "The referenced training config must enable qat.enabled.",
            }
        )
    retained_qat_matches = bool(
        qat_cfg.get("scale_mode") == "retained_mobile"
        and qat_cfg.get("fixed_scale_required") is True
        and qat_cfg.get("fixed_activation_scale_required") is True
        and qat_cfg.get("effective_lora_only") is True
        and qat_cfg.get("effective_merged_weight") is True
        and qat_cfg.get("ste_gradient") == "clipped"
        and int(qat_cfg.get("expected_effective_lora_modules", 0) or 0) == 205
    )
    if not retained_qat_matches:
        validation.append(
            {
                "severity": "error",
                "code": "retained_mobile_qat_contract_mismatch",
                "message": (
                    "This pipeline accepts only exact-205 retained_mobile QAT "
                    "with fixed weight/A8 scales, effective-LoRA-only and clipped STE."
                ),
            }
        )
    if model_cfg.get("architecture_preflight_required") is not True:
        validation.append(
            {
                "severity": "error",
                "code": "architecture_preflight_not_required",
                "message": (
                    "The E2B training config must require a 541-key/shape "
                    "Gemma4ForCausalLM meta-device architecture preflight."
                ),
            }
        )
    if model_cfg.get("require_exact_checkpoint_keys") is not True:
        validation.append(
            {
                "severity": "error",
                "code": "exact_checkpoint_keys_not_required",
                "message": (
                    "The E2B training config must reject missing, unexpected, "
                    "mismatched, or errored keys during the real model load."
                ),
            }
        )
    if android_warm_runs < MIN_PERFORMANCE_WARM_RUNS:
        validation.append(
            {
                "severity": "error",
                "code": "insufficient_android_gpu_warm_runs",
                "message": (
                    "Schema-v5 Android GPU promotion requires at least "
                    f"{MIN_PERFORMANCE_WARM_RUNS} warm runs per artifact."
                ),
            }
        )
    if not mtp_enabled and (
        mtp_weight_source == "trained" or train_assistant
    ):
        validation.append(
            {
                "severity": "error",
                "code": "mtp_disabled_with_trained_drafter",
                "message": (
                    "MTP-disabled mode preserves the official drafter section but "
                    "does not train or inject drafter weights. Set weight_source=official "
                    "and train_assistant=false, or enable MTP."
                ),
            }
        )
    if not bool(mtp_cfg.get("preserve_official_section", True)):
        validation.append(
            {
                "severity": "error",
                "code": "official_mtp_template_required",
                "message": (
                    "The target stage must preserve the released MTP section; "
                    "trained matrices are injected only in the second gated stage."
                ),
            }
        )
    if mtp_weight_source not in {"official", "trained"}:
        validation.append(
            {
                "severity": "error",
                "code": "invalid_mtp_weight_source",
                "message": "mtp.weight_source must be official or trained.",
            }
        )
    if mtp_enabled and train_assistant and mtp_weight_source != "trained":
        validation.append(
            {
                "severity": "error",
                "code": "drafter_training_requires_trained_weight_source",
                "message": "Set mtp.weight_source=trained when train_assistant=true.",
            }
        )
    if trained_mtp_enabled and not retained_export_enabled:
        validation.append(
            {
                "severity": "error",
                "code": "trained_mtp_requires_retained_scale_export",
                "message": (
                    "Trained assistant weights can only be injected through the "
                    "dedicated retained-scale target export path."
                ),
            }
        )
    if trained_mtp_enabled and not drafter_training_config_path.is_file():
        validation.append(
            {
                "severity": "error",
                "code": "missing_drafter_training_config",
                "message": (
                    "The trained MTP path requires its QAT training config: "
                    f"{drafter_training_config_path}"
                ),
            }
        )
    if base_litertlm is None:
        validation.append(
            {
                "severity": "warning",
                "code": "missing_base_package",
                "message": (
                    "Provide the official base_litertlm package before retained-scale export."
                ),
            }
        )
    if target_retained_package_output.parent.resolve() != retained_output_dir.resolve():
        validation.append(
            {
                "severity": "error",
                "code": "retained_export_output_outside_fresh_directory",
                "message": (
                    "The retained-scale target package must be a direct child of "
                    "pipeline.retained_scale_export.output_dir. Choose "
                    "--output-litertlm inside --exact-output-dir."
                ),
            }
        )
    if export_report.parent.resolve() != retained_output_dir.resolve():
        validation.append(
            {
                "severity": "error",
                "code": "retained_export_report_outside_fresh_directory",
                "message": (
                    "The retained-scale report must be a direct child of "
                    "pipeline.retained_scale_export.output_dir."
                ),
            }
        )
    if target_litertlm is not None or target_section is not None:
        validation.append(
            {
                "severity": "error",
                "code": "legacy_target_source_blocked",
                "message": (
                    "Retained-mobile deployment cannot consume a pre-exported target "
                    "or raw target section; use the dedicated code-only exporter."
                ),
            }
        )
    if public_export_enabled:
        validation.append(
            {
                "severity": "error",
                "code": "legacy_public_export_blocked",
                "message": (
                    "Generic LiteRT Torch/abs-max target export is forbidden for "
                    "qat.scale_mode=retained_mobile."
                ),
            }
        )
    if _section(pipeline_cfg, "exact_topology"):
        validation.append(
            {
                "severity": "error",
                "code": "legacy_exact_topology_config_blocked",
                "message": (
                    "pipeline.exact_topology is the retired abs-max target exporter. "
                    "Use pipeline.retained_scale_export only."
                ),
            }
        )
    if not retained_export_enabled:
        validation.append(
            {
                "severity": "error",
                "code": "retained_scale_export_disabled",
                "message": "pipeline.retained_scale_export.enabled must remain true.",
            }
        )
    if not retained_export_script.is_file():
        validation.append(
            {
                "severity": "error",
                "code": "retained_scale_exporter_missing",
                "message": f"Dedicated exporter does not exist: {retained_export_script}",
            }
        )
    if target_retained_package_output.parent != retained_output_dir:
        validation.append(
            {
                "severity": "error",
                "code": "retained_output_outside_fresh_export_dir",
                "message": (
                    "The retained target package must be a direct child of the "
                    "fresh retained-scale output directory."
                ),
            }
        )
    if export_report.parent != retained_output_dir:
        validation.append(
            {
                "severity": "error",
                "code": "retained_report_outside_fresh_export_dir",
                "message": (
                    "The retained exporter report must be a direct child of the "
                    "fresh retained-scale output directory."
                ),
            }
        )
    if target_retained_package_output == export_report:
        validation.append(
            {
                "severity": "error",
                "code": "retained_output_report_collision",
                "message": "The retained package and JSON report paths must differ.",
            }
        )

    return {
        "pipeline_id": str(pipeline_cfg.get("id", "gemma4_e2b_mobile_mtp")),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "training": {
            "config": str(training_config_path),
            "command": train_command,
            "model_id": model_id,
            "model_source": str(model_source) if model_source else None,
            "mobile_training_seed_manifest": str(mobile_training_seed_manifest)
            if mobile_training_seed_manifest
            else None,
            "mobile_training_seed": mobile_training_seed,
            "qat_profile": qat_cfg.get("profile"),
            "output_dir": str(training_output),
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_ready": _checkpoint_ready(best_checkpoint),
            "best_checkpoint_required": True,
            "architecture_preflight": {
                "required": True,
                "script": str(architecture_preflight_script),
                "output": str(architecture_preflight_output),
                "command": architecture_preflight_command,
                "loads_weights": False,
                "runs_forward": False,
                "runs_training": False,
            },
        },
        "merge": {
            "base_model_id": model_id,
            "base_model_source": str(model_source) if model_source else None,
            "mobile_training_seed_manifest": str(mobile_training_seed_manifest)
            if mobile_training_seed_manifest
            else None,
            "adapter_dir": str(best_checkpoint),
            "merged_model_dir": str(merged_model_dir),
            "training_qat_mode": "effective_merged_weight",
            "merge_performs_qat": False,
            "packed_int4_output": False,
            "assistant_modified": False,
        },
        "public_export": {
            "enabled": public_export_enabled,
            "config": str(export_config_path),
            "output_dir": str(public_export_dir),
            "warning": (
                "Public LiteRT Torch export is a standalone candidate only; it is not "
                "Google's private Gemma 4 mobile wNa8o8 exporter."
            ),
        },
        "retained_scale_export": {
            "enabled": retained_export_enabled,
            "mode": "retained_scale_code_only_v1",
            "official_artifact_sha256": official_artifact_sha256 or None,
            "official_litertlm": str(base_litertlm) if base_litertlm else None,
            "merged_checkpoint": str(merged_model_dir),
            "adapter_checkpoint": str(best_checkpoint),
            "training_config": str(training_config_path),
            "mobile_training_seed_manifest": str(mobile_training_seed_manifest)
            if mobile_training_seed_manifest
            else None,
            "mobile_qparams_contract": str(mobile_qparams_contract)
            if mobile_qparams_contract
            else None,
            "mobile_qparams": mobile_qparams,
            "mobile_training_seed_manifest": str(mobile_training_seed_manifest)
            if mobile_training_seed_manifest
            else None,
            "mobile_training_seed": mobile_training_seed,
            "zero_adapter_checkpoint": str(model_source) if model_source else None,
            "output_dir": str(target_retained_output_dir),
            "output_litertlm": str(target_retained_package_output),
            "report": str(export_report),
            "final_output_litertlm": str(output_litertlm),
            "command": retained_export_command,
            "training_executed": False,
            "preserves_default_mtp_byte_exact": True,
            "final_package_preserves_default_mtp_byte_exact": (
                not trained_mtp_enabled
            ),
            "requires_exact_205_projection_mapping": True,
            "legacy_absmax_export_blocked": True,
        },
        "mtp": {
            "enabled": mtp_enabled,
            "weight_source": mtp_weight_source,
            "official_weights_preserved": not trained_mtp_enabled,
            "section_present": True,
            "assistant_model_id": assistant_model_id,
            "train_assistant": train_assistant,
            "training": {
                "enabled": mtp_enabled and train_assistant,
                "config": str(drafter_training_config_path),
                "target_model": str(merged_model_dir),
                "checkpoint": str(drafter_checkpoint),
                "command": drafter_training_command,
                "executed": False,
                "private_google_recipe_recovered": False,
            },
            "exact_topology": {
                "enabled": trained_mtp_enabled,
                "package_input": str(target_intermediate_litertlm),
                "output_dir": str(drafter_exact_output_dir),
                "output_litertlm": str(output_litertlm),
                "command": drafter_export_command,
                "requires_complete_23_weight_mapping": True,
                "preserves_target_section_byte_exact": True,
            },
        },
        "package": {
            "base_litertlm": str(base_litertlm) if base_litertlm else None,
            "target_litertlm": str(target_litertlm) if target_litertlm else None,
            "target_section": str(target_section) if target_section else None,
            "output_litertlm": str(output_litertlm),
            "target_model_type": target_model_type,
            "mtp_model_type": mtp_model_type,
            "mtp_enabled": mtp_enabled,
            "mtp_section_present": True,
            "mtp_assistant_model_id": assistant_model_id,
            "mtp_assistant_weight_source": (
                mtp_weight_source if mtp_enabled else "official_preserved_disabled"
            ),
            "official_mtp_bytes_preserved": not trained_mtp_enabled,
            "target_export_authority": (
                "official_package_plus_retained_scale_205_code_only_patch"
            ),
        },
        "android_gpu": {
            "delegate": "gpu",
            "warm_runs": android_warm_runs,
            "minimum_performance_warm_runs": MIN_PERFORMANCE_WARM_RUNS,
            "performance_selection_policy": PERFORMANCE_SELECTION_POLICY,
            "device_validation": (
                "target_only_and_mtp_on_required_after_packaging"
                if mtp_enabled
                else "target_only_required_after_packaging"
            ),
            "parity_runner": android_runner,
            "output_dir": str(android_output_dir),
            "required_modes": (
                ["target_only", "mtp_on"] if mtp_enabled else ["target_only"]
            ),
            "target_only": {
                "mtp_flag": False,
                "required": True,
                "purpose": "isolate target graph GPU throughput from draft acceptance",
                "output_dir": str(target_only_output_dir),
                "command": target_only_command,
            },
            "mtp_on": {
                "mtp_flag": True,
                "required": mtp_enabled,
                "purpose": "validate preserved drafter acceptance and speculative throughput",
                "output_dir": str(mtp_on_output_dir),
                "command": mtp_on_command,
            },
        },
        "validation": {
            "ok": not any(item["severity"] == "error" for item in validation),
            "issues": validation,
        },
        "limitations": [
            (
                "MTP inference is disabled; the released drafter section remains byte-for-byte in the package so graph topology stays official and MTP can be enabled later."
                if not mtp_enabled
                else (
                    "The released MTP drafter is preserved byte-for-byte."
                    if mtp_weight_source == "official"
                    else "The trained drafter path is a public reconstruction; Google's private data mixture, loss weighting, optimizer, and observer schedule are not recovered."
                )
            ),
            "Fine-tuning the target can lower MTP acceptance; measure it on device.",
            "A real Android LiteRT-LM GPU run is required for a runtime claim.",
        ],
    }


def _run_command(command: list[str], log_path: Path, *, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command:\n" + " ".join(command) + "\n\n")
        process = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(process.stdout or "")
    if process.returncode != 0:
        raise Gemma4MobileMTPPipelineError(
            f"Command failed with exit code {process.returncode}; see {log_path}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_file_records(directory: Path, candidates: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": candidate.relative_to(directory).as_posix(),
            "size": int(candidate.stat().st_size),
            "sha256": _sha256_file(candidate),
        }
        for candidate in sorted(candidates, key=lambda item: item.as_posix())
        if candidate.is_file()
    ]


def _adapter_file_records(directory: Path) -> list[dict[str, Any]]:
    return _current_file_records(directory, list(directory.glob("adapter*")))


def _merged_file_records(directory: Path) -> list[dict[str, Any]]:
    candidates = list(directory.glob("model*.safetensors"))
    candidates.append(directory / "model.safetensors.index.json")
    return _current_file_records(directory, candidates)


def _normalized_artifact_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            size = int(item.get("size", -1))
        except (TypeError, ValueError):
            size = -1
        records.append(
            {
                "path": str(item.get("path") or "").replace("\\", "/"),
                "size": size,
                "sha256": str(item.get("sha256") or "").lower(),
            }
        )
    return sorted(records, key=lambda item: item["path"])


def _normalized_merged_verification(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or item.get("valid") is not True:
            return []
        try:
            expected_size = int(item.get("expected_size", -1))
            observed_size = int(item.get("observed_size", -1))
        except (TypeError, ValueError):
            return []
        expected_sha = str(item.get("expected_sha256") or "").lower()
        observed_sha = str(item.get("observed_sha256") or "").lower()
        if expected_size != observed_size or expected_sha != observed_sha:
            return []
        records.append(
            {
                "path": str(item.get("path") or "").replace("\\", "/"),
                "size": observed_size,
                "sha256": observed_sha,
            }
        )
    return sorted(records, key=lambda item: item["path"])


def _validate_retained_scale_export_report(
    report_path: str | Path,
    *,
    expected_plan: dict[str, Any],
) -> dict[str, Any]:
    """Recheck the dedicated exporter's final fail-closed report."""

    path = Path(report_path)
    if not path.is_file():
        raise Gemma4MobileMTPPipelineError(
            f"Retained-scale exporter did not write its report: {path}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Gemma4MobileMTPPipelineError(
            f"Retained-scale exporter report is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise Gemma4MobileMTPPipelineError(
            "Retained-scale exporter report root is not an object."
        )
    required_gates = {
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
    gates = report.get("gates")
    gates = gates if isinstance(gates, dict) else {}
    package_checks = report.get("package_checks")
    package_checks = package_checks if isinstance(package_checks, dict) else {}
    failed = sorted(name for name in required_gates if gates.get(name) is not True)
    failed.extend(
        f"package_checks.{name}"
        for name, passed in sorted(package_checks.items())
        if passed is not True
    )
    output = Path(str(expected_plan["output_litertlm"])).resolve()
    declared_output = report.get("output_litertlm")
    output_identity_ok = bool(
        output.is_file()
        and declared_output
        and Path(str(declared_output)).resolve() == output
        and str(report.get("output_sha256") or "").lower() == _sha256_file(output)
    )
    expected_official = Path(str(expected_plan["official_litertlm"])).resolve()
    expected_merged = Path(str(expected_plan["merged_checkpoint"])).resolve()
    expected_adapter = Path(str(expected_plan["adapter_checkpoint"])).resolve()
    expected_config = Path(str(expected_plan["training_config"])).resolve()
    expected_seed_manifest = Path(
        str(expected_plan["mobile_training_seed_manifest"])
    ).resolve()
    expected_qparams = Path(str(expected_plan["mobile_qparams_contract"])).resolve()
    expected_zero = Path(str(expected_plan["zero_adapter_checkpoint"])).resolve()
    official_identity = report.get("official_artifact_identity")
    official_identity = official_identity if isinstance(official_identity, dict) else {}
    merged_identity = report.get("merged_checkpoint_identity")
    merged_identity = merged_identity if isinstance(merged_identity, dict) else {}
    adapter_identity = report.get("adapter_identity")
    adapter_identity = adapter_identity if isinstance(adapter_identity, dict) else {}
    config_identity = report.get("resolved_training_config_identity")
    config_identity = config_identity if isinstance(config_identity, dict) else {}
    seed_identity = report.get("mobile_training_seed_identity")
    seed_identity = seed_identity if isinstance(seed_identity, dict) else {}
    qparams_identity = report.get("mobile_qparams_identity")
    qparams_identity = qparams_identity if isinstance(qparams_identity, dict) else {}
    expected_qparams_report = expected_plan.get("mobile_qparams")
    expected_qparams_report = (
        expected_qparams_report
        if isinstance(expected_qparams_report, dict)
        else {}
    )
    expected_scale_storage = Path(
        str(expected_qparams_report.get("scale_storage_path") or "")
    ).resolve()

    def file_matches(path_value: Any, size_value: Any, sha_value: Any) -> bool:
        try:
            candidate = Path(str(path_value)).resolve()
            expected_size = int(size_value)
            expected_sha = str(sha_value or "").lower()
        except (OSError, TypeError, ValueError):
            return False
        return bool(
            candidate.is_file()
            and expected_size >= 0
            and len(expected_sha) == 64
            and candidate.stat().st_size == expected_size
            and _sha256_file(candidate) == expected_sha
        )

    adapter_records = adapter_identity.get("files")
    adapter_records = adapter_records if isinstance(adapter_records, list) else []
    adapter_record_names = {
        str(item.get("path") or "")
        for item in adapter_records
        if isinstance(item, dict)
    }
    actual_adapter_names = {
        item.name for item in expected_adapter.glob("adapter*") if item.is_file()
    }
    adapter_files_live = bool(
        adapter_records
        and adapter_record_names == actual_adapter_names
        and all(
            isinstance(item, dict)
            and file_matches(
                expected_adapter / str(item.get("path") or ""),
                item.get("size"),
                item.get("sha256"),
            )
            for item in adapter_records
        )
    )
    adapter_metadata_path = expected_adapter / "training_metadata.json"
    adapter_metadata_live = bool(
        adapter_metadata_path.is_file()
        and str(adapter_identity.get("training_metadata_sha256") or "").lower()
        == _sha256_file(adapter_metadata_path)
    )

    merged_records = merged_identity.get("file_verification")
    merged_records = merged_records if isinstance(merged_records, list) else []
    merged_record_names = {
        str(item.get("path") or "").replace("\\", "/")
        for item in merged_records
        if isinstance(item, dict)
    }
    actual_merged_names = {
        item.relative_to(expected_merged).as_posix()
        for item in expected_merged.glob("model*.safetensors")
        if item.is_file()
    }
    merged_index = expected_merged / "model.safetensors.index.json"
    if merged_index.is_file():
        actual_merged_names.add(merged_index.relative_to(expected_merged).as_posix())
    merged_files_live = bool(
        merged_records
        and merged_record_names == actual_merged_names
        and all(
            isinstance(item, dict)
            and item.get("valid") is True
            and Path(str(item.get("path") or "")).is_absolute() is False
            and ".." not in Path(str(item.get("path") or "")).parts
            and file_matches(
                expected_merged / str(item.get("path") or ""),
                item.get("expected_size"),
                item.get("expected_sha256"),
            )
            for item in merged_records
        )
    )
    merged_metadata_path = expected_merged / "qat_mtp_merge_metadata.json"
    merged_metadata_live = bool(
        merged_metadata_path.is_file()
        and Path(str(merged_identity.get("metadata_path") or "")).resolve()
        == merged_metadata_path.resolve()
        and str(merged_identity.get("metadata_sha256") or "").lower()
        == _sha256_file(merged_metadata_path)
    )
    expected_official_sha = str(
        expected_plan.get("official_artifact_sha256") or ""
    ).lower()
    input_identity_ok = bool(
        Path(str(report.get("official_litertlm") or "")).resolve()
        == expected_official
        and str(report.get("official_artifact_sha256") or "").lower()
        == expected_official_sha
        and official_identity.get("verified") is True
        and str(official_identity.get("declared_sha256") or "").lower()
        == expected_official_sha
        and str(official_identity.get("observed_sha256") or "").lower()
        == expected_official_sha
        and Path(str(merged_identity.get("path") or "")).resolve()
        == expected_merged
        and merged_identity.get("verified") is True
        and merged_files_live
        and merged_metadata_live
        and Path(str(adapter_identity.get("path") or "")).resolve()
        == expected_adapter
        and adapter_identity.get("verified") is True
        and adapter_files_live
        and adapter_metadata_live
        and Path(str(config_identity.get("path") or "")).resolve()
        == expected_config
        and config_identity.get("verified") is True
        and str(config_identity.get("sha256") or "").lower()
        == _sha256_file(expected_config)
        and Path(str(seed_identity.get("manifest_path") or "")).resolve()
        == expected_seed_manifest
        and seed_identity.get("verified") is True
        and str(seed_identity.get("manifest_sha256") or "").lower()
        == _sha256_file(expected_seed_manifest)
        and Path(str(qparams_identity.get("contract_path") or "")).resolve()
        == expected_qparams
        and qparams_identity.get("verified") is True
        and str(qparams_identity.get("contract_sha256") or "").lower()
        == _sha256_file(expected_qparams)
        and Path(str(qparams_identity.get("scale_storage_path") or "")).resolve()
        == expected_scale_storage
        and expected_scale_storage.is_file()
        and str(qparams_identity.get("scale_storage_sha256") or "").lower()
        == _sha256_file(expected_scale_storage)
        and Path(str(report.get("zero_adapter_checkpoint") or "")).resolve()
        == expected_zero
    )
    if (
        report.get("mode") != "retained_scale_code_only_v1"
        or report.get("executed") is not True
        or report.get("plan_passed") is not True
        or report.get("passed") is not True
        or set(gates) != required_gates
        or not package_checks
        or failed
        or not output_identity_ok
        or not input_identity_ok
    ):
        detail = failed or [
            "mode/executed/plan/passed/gate-set/input-or-output-identity"
        ]
        raise Gemma4MobileMTPPipelineError(
            "Retained-scale exporter report failed pipeline verification: "
            + ", ".join(detail)
        )
    return report


def _load_android_gpu_report(
    report_path: str | Path,
    *,
    mode: str,
    expected_mtp: bool,
    expected_official_artifact: str | Path | None = None,
    expected_candidate_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Load one fail-closed parity report and enforce its mode-specific gates."""

    try:
        return load_android_gpu_parity_report(
            report_path,
            expected_mtp=expected_mtp,
            require_mtp_acceptance=expected_mtp,
            expected_official_artifact=expected_official_artifact,
            expected_candidate_artifact=expected_candidate_artifact,
        )
    except AndroidGpuParityReportError as exc:
        raise Gemma4MobileMTPPipelineError(
            f"Android GPU {mode} report {exc}: {Path(report_path)}"
        ) from exc


def run_pipeline(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    execute_training: bool = False,
    execute_merge: bool = False,
    execute_drafter_training: bool = False,
    execute_public_export: bool = False,
    execute_exact_topology_export: bool = False,
    execute_retained_scale_export: bool = False,
    compose_package: bool = False,
    validate_android_gpu: bool = False,
    adb_override: str | Path | None = None,
    serial_override: str | None = None,
    training_config_override: str | Path | None = None,
    best_checkpoint_override: str | Path | None = None,
    base_litertlm_override: str | Path | None = None,
    merged_model_dir_override: str | Path | None = None,
    exact_output_dir_override: str | Path | None = None,
    export_report_override: str | Path | None = None,
    target_litertlm_override: str | Path | None = None,
    target_section_override: str | Path | None = None,
    output_litertlm_override: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Plan and optionally execute explicitly selected stages."""

    if execute_exact_topology_export:
        raise Gemma4MobileMTPPipelineError(
            "--execute-exact-topology-export is the retired Gemma 4 abs-max path. "
            "Use --execute-retained-scale-export."
        )
    if execute_public_export:
        raise Gemma4MobileMTPPipelineError(
            "--execute-public-export is blocked for Gemma 4 retained_mobile QAT; "
            "it does not preserve the released scales."
        )
    if compose_package:
        raise Gemma4MobileMTPPipelineError(
            "--compose is blocked for Gemma 4 retained_mobile QAT. The dedicated "
            "retained-scale exporter writes the final official-topology package."
        )

    plan = build_pipeline_plan(
        config,
        config_path=config_path,
        training_config_override=training_config_override,
        best_checkpoint_override=best_checkpoint_override,
        base_litertlm_override=base_litertlm_override,
        merged_model_dir_override=merged_model_dir_override,
        exact_output_dir_override=exact_output_dir_override,
        export_report_override=export_report_override,
        target_litertlm_override=target_litertlm_override,
        target_section_override=target_section_override,
        output_litertlm_override=output_litertlm_override,
    )
    stage_errors = [
        item
        for item in plan["validation"]["issues"]
        if item["severity"] == "error"
    ]
    if stage_errors and (
        execute_training
        or execute_merge
        or execute_drafter_training
        or execute_retained_scale_export
        or validate_android_gpu
    ):
        raise Gemma4MobileMTPPipelineError(
            json.dumps(stage_errors, ensure_ascii=False)
        )

    base = training_root()
    pipeline_root = Path(plan["retained_scale_export"]["output_dir"]).parent
    logs_dir = pipeline_root / "logs"
    if execute_training:
        _run_command(plan["training"]["command"], logs_dir / "train_sft.log", cwd=base.parent)
        plan["training"]["best_checkpoint_ready"] = _checkpoint_ready(
            Path(plan["training"]["best_checkpoint"])
        )
        if not plan["training"]["best_checkpoint_ready"]:
            raise Gemma4MobileMTPPipelineError(
                "Training completed but no best golden checkpoint was found at "
                f"{plan['training']['best_checkpoint']}"
            )

    if execute_merge:
        checkpoint = Path(plan["training"]["best_checkpoint"])
        if not _checkpoint_ready(checkpoint):
            raise Gemma4MobileMTPPipelineError(
                f"Best checkpoint is missing or not a PEFT model: {checkpoint}"
            )
        merged_destination = Path(plan["merge"]["merged_model_dir"])
        if merged_destination.exists():
            raise Gemma4MobileMTPPipelineError(
                "Refusing to merge into an existing path; choose a unique fresh "
                f"--merged-model-dir: {merged_destination}"
            )
        merged = merge_lora_adapter(
            base_model_id=str(plan["merge"]["base_model_id"]),
            adapter_dir=checkpoint,
            output_dir=plan["merge"]["merged_model_dir"],
            model_loader=str(_section(_section(config, "pipeline"), "source").get("model_loader", "auto_causal_lm")),
            dtype=str(_section(_section(config, "pipeline"), "source").get("dtype", "bfloat16")),
            trust_remote_code=bool(_section(_section(config, "pipeline"), "source").get("trust_remote_code", False)),
            processor_model_id=str(plan["merge"]["base_model_id"]),
            training_config_path=plan["training"]["config"],
            base_model_source=plan["merge"]["base_model_source"],
            mobile_training_seed_manifest=plan["merge"][
                "mobile_training_seed_manifest"
            ],
        )
        plan["merge"]["merged_model_dir"] = str(merged)
        plan["merge"]["executed"] = True

    if execute_drafter_training:
        if not plan["mtp"]["enabled"]:
            raise Gemma4MobileMTPPipelineError(
                "Drafter training cannot run while pipeline.mtp.enabled=false."
            )
        if plan["mtp"]["weight_source"] != "trained":
            raise Gemma4MobileMTPPipelineError(
                "Drafter training requires pipeline.mtp.weight_source=trained."
            )
        if not plan["mtp"]["training"]["enabled"]:
            raise Gemma4MobileMTPPipelineError(
                "Drafter training is disabled. Set pipeline.mtp.train_assistant=true "
                "or provide an existing trained checkpoint."
            )
        merged_dir = Path(plan["merge"]["merged_model_dir"])
        if not merged_dir.is_dir():
            raise Gemma4MobileMTPPipelineError(
                f"Merged target model is missing: {merged_dir}. Run --execute-merge first."
            )
        _run_command(
            list(plan["mtp"]["training"]["command"]),
            logs_dir / "train_mtp_drafter.log",
            cwd=base.parent,
        )
        checkpoint = Path(plan["mtp"]["training"]["checkpoint"])
        if not _checkpoint_ready(checkpoint):
            raise Gemma4MobileMTPPipelineError(
                f"Drafter training did not produce the configured checkpoint: {checkpoint}"
            )
        plan["mtp"]["training"]["executed"] = True

    if execute_retained_scale_export:
        retained_plan = plan["retained_scale_export"]
        if not retained_plan["enabled"]:
            raise Gemma4MobileMTPPipelineError(
                "pipeline.retained_scale_export.enabled=false."
            )
        if not retained_plan["official_litertlm"]:
            raise Gemma4MobileMTPPipelineError(
                "Retained-scale export requires source.base_litertlm or --base-litertlm."
            )
        merged_dir = Path(plan["merge"]["merged_model_dir"])
        if not merged_dir.is_dir():
            raise Gemma4MobileMTPPipelineError(
                f"Merged best-checkpoint model is missing: {merged_dir}. Run --execute-merge first."
            )
        output_dir = Path(retained_plan["output_dir"])
        report_path = Path(retained_plan["report"])
        output_path = Path(retained_plan["output_litertlm"])
        collisions = [
            str(path)
            for path in (output_dir, report_path, output_path)
            if path.exists()
        ]
        if collisions:
            raise Gemma4MobileMTPPipelineError(
                "Refusing to reuse retained-scale export outputs; choose fresh paths: "
                + ", ".join(collisions)
            )
        _run_command(
            list(retained_plan["command"]),
            logs_dir / "retained_scale_export.log",
            cwd=base.parent,
        )
        report = _validate_retained_scale_export_report(
            report_path, expected_plan=retained_plan
        )
        final_manifest: dict[str, Any] | None = None
        if plan["mtp"]["exact_topology"]["enabled"]:
            checkpoint = Path(plan["mtp"]["training"]["checkpoint"])
            if not _checkpoint_ready(checkpoint):
                raise Gemma4MobileMTPPipelineError(
                    f"Trained MTP checkpoint is missing: {checkpoint}. Run "
                    "--execute-drafter-training or provide mtp.trained_checkpoint."
                )
            _run_command(
                list(plan["mtp"]["exact_topology"]["command"]),
                logs_dir / "exact_topology_mtp_drafter.log",
                cwd=base.parent,
            )
            drafter_report_path = (
                Path(plan["mtp"]["exact_topology"]["output_dir"])
                / "mtp_checkpoint_official_topology_report.json"
            )
            if not drafter_report_path.is_file():
                raise Gemma4MobileMTPPipelineError(
                    "Trained MTP exporter did not write its report: "
                    f"{drafter_report_path}"
                )
            drafter_report = json.loads(
                drafter_report_path.read_text(encoding="utf-8")
            )
            if not bool(drafter_report.get("final_artifact_gate_pass")):
                raise Gemma4MobileMTPPipelineError(
                    "Trained MTP exporter did not pass all graph/layout/package gates."
                )
            drafter_gates = drafter_report.get("gates") or {}
            if not bool(
                drafter_gates.get("fine_tuned_target_section_byte_exact")
                and drafter_gates.get("package_outside_mtp_byte_exact")
                and drafter_gates.get("official_graph_structure")
                and drafter_gates.get("official_quantization_layout")
            ):
                raise Gemma4MobileMTPPipelineError(
                    "Trained MTP export did not prove target preservation and "
                    "official MTP graph/layout parity."
                )
            plan["mtp"]["exact_topology"]["executed"] = True
            plan["mtp"]["exact_topology"]["report"] = drafter_report
            final_manifest = {"target": report, "mtp_drafter": drafter_report}
        retained_plan["executed"] = True
        retained_plan["report_payload"] = report
        plan["package"]["executed"] = True
        plan["package"]["manifest"] = final_manifest or report

    if validate_android_gpu:
        official_package = plan["package"].get("base_litertlm")
        candidate_package = plan["package"].get("output_litertlm")
        if not official_package or not Path(official_package).is_file():
            raise Gemma4MobileMTPPipelineError(
                "Android GPU parity requires the official base LiteRT-LM package."
            )
        if not candidate_package or not Path(candidate_package).is_file():
            raise Gemma4MobileMTPPipelineError(
                "Android GPU parity requires an exported candidate package first."
            )
        reports: dict[str, Any] = {}
        for mode in plan["android_gpu"]["required_modes"]:
            mode_plan = plan["android_gpu"][mode]
            command = list(mode_plan["command"])
            if adb_override:
                command.extend(
                    ["--adb", str(Path(adb_override).expanduser().resolve())]
                )
            if serial_override:
                command.extend(["--serial", str(serial_override)])
            _run_command(
                command,
                logs_dir / f"android_gpu_{mode}.log",
                cwd=base.parent,
            )
            report_path = (
                Path(mode_plan["output_dir"])
                / "android_litertlm_gpu_parity_report.json"
            )
            reports[mode] = _load_android_gpu_report(
                report_path,
                mode=mode,
                expected_mtp=bool(mode_plan["mtp_flag"]),
                expected_official_artifact=official_package,
                expected_candidate_artifact=candidate_package,
            )
        plan["android_gpu"]["executed"] = True
        plan["android_gpu"]["reports"] = reports

    return plan
