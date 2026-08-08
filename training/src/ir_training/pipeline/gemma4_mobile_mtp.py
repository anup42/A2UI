"""Gemma 4 E2B mobile QAT -> best checkpoint -> MTP package orchestration.

The pipeline is deliberately split into explicit stages:

1. QAT SFT (the existing ``train_sft.py`` path) saves a golden-set best
   adapter;
2. the best adapter is merged back into the floating-point base model;
3. the preferred exact-topology stage quantizes merged projection weights into
   a copy of the released target graph;
4. MTP weights use either official bytes or a trained 23-matrix transplant;
5. an optional generic LiteRT Torch export produces a *standalone* diagnostic;
6. a legacy compatible target TFLite section can be composed with the official package,
   preserving the default ``tf_lite_mtp_drafter`` section byte-for-byte; and
7. package/device validation records what was actually proven.

The default command is plan-only.  Training and public conversion are
expensive and the public converter is not Google's private Gemma mobile
exporter, so neither is run unless explicitly requested.
"""

from __future__ import annotations

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
from ir_training.export.edge_gallery import export_edge_gallery_model
from ir_training.export.litertlm_mtp import compose_with_default_mtp
from ir_training.export.merge_lora import merge_lora_adapter
from ir_training.qat.mobile_training_seed import (
    OFFICIAL_MOBILE_MODEL_ID,
    verify_configured_mobile_training_seed,
)
from ir_training.qat.retained_constants import verify_retained_constant_contract
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
    best_checkpoint_override: str | Path | None = None,
    base_litertlm_override: str | Path | None = None,
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
    training_config_path = resolve_path(str(training_config_value), base)
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
    merged_model_dir = _path_or_empty(
        source_cfg.get("merged_model_dir"), base
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
    ) or resolve_path(
        str(pipeline_cfg.get("output_dir", "outputs/pipelines/gemma4_e2b_mobile_mtp"))
        + "/gemma4_e2b_mobile_mtp.litertlm",
        base,
    )
    public_export_cfg = _section(pipeline_cfg, "public_export")
    public_export_dir = _path_or_empty(public_export_cfg.get("output_dir"), base)
    if public_export_dir is None:
        public_export_dir = resolve_path(
            str(pipeline_cfg.get("output_dir", "outputs/pipelines/gemma4_e2b_mobile_mtp"))
            + "/public_export",
            base,
        )
    exact_cfg = _section(pipeline_cfg, "exact_topology")
    exact_enabled = bool(exact_cfg.get("enabled", True))
    exact_output_dir = _path_or_empty(exact_cfg.get("output_dir"), base)
    if exact_output_dir is None:
        exact_output_dir = resolve_path(
            str(pipeline_cfg.get("output_dir", "outputs/pipelines/gemma4_e2b_mobile_mtp"))
            + "/exact_topology_export",
            base,
        )

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
    ) or resolve_path(
        str(pipeline_cfg.get("output_dir", "outputs/pipelines/gemma4_e2b_mobile_mtp"))
        + "/gemma4_e2b_target_with_official_mtp.litertlm",
        base,
    )
    drafter_exact_output_dir = _path_or_empty(
        mtp_cfg.get("exact_topology_output_dir"), base
    ) or (exact_output_dir / "drafter")
    target_exact_output_dir = (
        exact_output_dir / "target"
        if trained_mtp_enabled
        else exact_output_dir
    )
    target_exact_package_output = (
        target_intermediate_litertlm
        if trained_mtp_enabled
        else output_litertlm
    )
    public_export_enabled = bool(public_export_cfg.get("enabled", False))
    official_base_model_id = str(
        exact_cfg.get("official_base_model_id") or model_id
    )
    official_artifact_sha256 = str(
        exact_cfg.get("official_artifact_sha256") or ""
    ).strip().lower()
    retained_constant_contract = _path_or_empty(
        exact_cfg.get("retained_constant_contract"), base
    )
    retained_constant_compatibility = verify_retained_constant_contract(
        retained_constant_contract,
        family="gemma4_e2b",
        training_model_id=model_id,
    )
    exact_script = base / "scripts" / "build_checkpoint_official_topology.py"
    exact_command = [
        sys.executable,
        str(exact_script),
        str(base_litertlm) if base_litertlm else "<official-base-litertlm-required>",
        "--checkpoint",
        str(merged_model_dir),
        "--family",
        "gemma4_e2b",
        "--model-type",
        target_model_type,
        "--training-config",
        str(training_config_path),
        "--retained-constant-contract",
        str(retained_constant_contract)
        if retained_constant_contract
        else "<retained-constant-contract-required>",
        "--mobile-training-seed-manifest",
        str(mobile_training_seed_manifest)
        if mobile_training_seed_manifest
        else "<mobile-training-seed-manifest-required>",
        "--official-base-model-id",
        official_base_model_id,
        "--official-artifact-sha256",
        official_artifact_sha256 or "<official-artifact-sha256-required>",
        "--output-dir",
        str(target_exact_output_dir),
        "--package-output",
        str(target_exact_package_output),
        "--calibration-samples",
        str(int(exact_cfg.get("calibration_samples", 2))),
        "--threads",
        str(int(exact_cfg.get("threads", 1))),
        "--execute",
    ]
    converter_batch_size = exact_cfg.get("converter_batch_size")
    if converter_batch_size not in (None, "", 0):
        exact_command.extend(["--converter-batch-size", str(int(converter_batch_size))])
    if bool(exact_cfg.get("retain_intermediates", False)):
        exact_command.append("--retain-intermediates")
    if bool(exact_cfg.get("runtime_allocate", False)):
        exact_command.extend(
            ["--runtime-allocate", "--runtime-threads", str(int(exact_cfg.get("runtime_threads", 2)))]
        )
        if bool(exact_cfg.get("runtime_without_default_delegates", True)):
            exact_command.append("--runtime-without-default-delegates")
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
        str(int(exact_cfg.get("calibration_samples", 2))),
        "--threads",
        str(int(exact_cfg.get("threads", 1))),
        "--execute",
    ]
    if converter_batch_size not in (None, "", 0):
        drafter_export_command.extend(
            ["--converter-batch-size", str(int(converter_batch_size))]
        )
    if bool(exact_cfg.get("retain_intermediates", False)):
        drafter_export_command.append("--retain-intermediates")
    if bool(exact_cfg.get("runtime_allocate", False)):
        drafter_export_command.extend(
            ["--runtime-allocate", "--runtime-threads", str(int(exact_cfg.get("runtime_threads", 2)))]
        )
        if bool(exact_cfg.get("runtime_without_default_delegates", True)):
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
    if official_base_model_id != model_id:
        validation.append(
            {
                "severity": "error",
                "code": "training_seed_provenance_mismatch",
                "message": (
                    "exact_topology.official_base_model_id must equal the training "
                    "model_id so adapter merge provenance is bound to the same seed."
                ),
            }
        )
    if not retained_constant_compatibility["verified"]:
        validation.append(
            {
                "severity": "error",
                "code": "retained_constant_compatibility_unverified",
                "scope": "exact_topology_export",
                "message": (
                    "The dense training seed is not proven compatible with learned "
                    "constants retained from the packed mobile package. Current "
                    "evidence must be compatible_exact and mapped into the compiled graph."
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
    if trained_mtp_enabled and not exact_enabled:
        validation.append(
            {
                "severity": "error",
                "code": "trained_mtp_requires_exact_topology",
                "message": (
                    "Trained assistant weights can only be injected through the "
                    "official exact-topology path."
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
                    "Provide the official base_litertlm package before exact export "
                    "or legacy composition."
                ),
            }
        )
    if target_litertlm is not None and target_section is not None:
        validation.append(
            {
                "severity": "warning",
                "code": "two_target_sources",
                "message": "Provide only target_litertlm or target_section.",
            }
        )
    if public_export_enabled and not export_config_path.is_file():
        validation.append(
            {
                "severity": "warning",
                "code": "missing_public_export_config",
                "message": f"Public export config does not exist: {export_config_path}",
            }
        )
    if (
        not exact_enabled
        and not public_export_enabled
        and target_litertlm is None
        and target_section is None
    ):
        validation.append(
            {
                "severity": "warning",
                "code": "private_target_export_pending",
                "message": (
                    "No compatible target TFLite section/package is configured. "
                    "The public generic exporter cannot be assumed to reproduce Google's mobile recipe."
                ),
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
        "exact_topology": {
            "enabled": exact_enabled,
            "family": "gemma4_e2b",
            "official_base_model_id": official_base_model_id,
            "official_artifact_sha256": official_artifact_sha256 or None,
            "official_litertlm": str(base_litertlm) if base_litertlm else None,
            "merged_checkpoint": str(merged_model_dir),
            "training_config": str(training_config_path),
            "retained_constant_contract": str(retained_constant_contract)
            if retained_constant_contract
            else None,
            "retained_constant_compatibility": retained_constant_compatibility,
            "mobile_training_seed": mobile_training_seed,
            "model_type": target_model_type,
            "output_dir": str(target_exact_output_dir),
            "output_litertlm": str(target_exact_package_output),
            "final_output_litertlm": str(output_litertlm),
            "command": exact_command,
            "training_executed": False,
            "preserves_default_mtp_byte_exact": True,
            "final_package_preserves_default_mtp_byte_exact": (
                not trained_mtp_enabled
            ),
            "requires_complete_277_weight_mapping": True,
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
                "official_graph_template_plus_public_quantized_checkpoint_constants"
                if exact_enabled
                else "compatible_mobile_section_required"
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
    compose_package: bool = False,
    validate_android_gpu: bool = False,
    adb_override: str | Path | None = None,
    serial_override: str | None = None,
    best_checkpoint_override: str | Path | None = None,
    base_litertlm_override: str | Path | None = None,
    target_litertlm_override: str | Path | None = None,
    target_section_override: str | Path | None = None,
    output_litertlm_override: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Plan and optionally execute explicitly selected stages."""

    plan = build_pipeline_plan(
        config,
        config_path=config_path,
        best_checkpoint_override=best_checkpoint_override,
        base_litertlm_override=base_litertlm_override,
        target_litertlm_override=target_litertlm_override,
        target_section_override=target_section_override,
        output_litertlm_override=output_litertlm_override,
    )
    stage_errors = [
        item
        for item in plan["validation"]["issues"]
        if item["severity"] == "error"
    ]
    blocking_stage_errors = [
        item
        for item in stage_errors
        if item.get("code") != "retained_constant_compatibility_unverified"
        or execute_exact_topology_export
    ]
    if blocking_stage_errors and (
        execute_training
        or execute_merge
        or execute_drafter_training
        or execute_public_export
        or execute_exact_topology_export
        or compose_package
        or validate_android_gpu
    ):
        raise Gemma4MobileMTPPipelineError(
            json.dumps(blocking_stage_errors, ensure_ascii=False)
        )

    if execute_exact_topology_export and compose_package:
        raise Gemma4MobileMTPPipelineError(
            "Exact-topology export already writes the final package and preserves the "
            "official MTP section; do not combine it with the legacy --compose stage."
        )

    base = training_root()
    pipeline_root = Path(plan["public_export"]["output_dir"]).parent
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
        if merged_destination.exists() and any(merged_destination.iterdir()):
            raise Gemma4MobileMTPPipelineError(
                "Refusing to merge into a non-empty directory; choose a new "
                f"merged_model_dir or clear it explicitly: {merged_destination}"
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

    if execute_public_export:
        if not plan["public_export"]["enabled"]:
            raise Gemma4MobileMTPPipelineError(
                "Public export is disabled in the pipeline config. Set "
                "pipeline.public_export.enabled=true only when you explicitly want "
                "a standalone public-converter candidate."
            )
        export_config_path = Path(plan["public_export"]["config"])
        if not export_config_path.is_file():
            raise Gemma4MobileMTPPipelineError(
                f"Public export config does not exist: {export_config_path}"
            )
        merged_dir = Path(plan["merge"]["merged_model_dir"])
        if not merged_dir.is_dir():
            raise Gemma4MobileMTPPipelineError(
                f"Merged best-checkpoint model is missing: {merged_dir}. Run --execute-merge first."
            )
        export_config = load_yaml(plan["public_export"]["config"])
        export_config.setdefault("run", {})["output_dir"] = plan["public_export"]["output_dir"]
        export_config.setdefault("source", {})["base_model_id"] = plan["merge"]["base_model_id"]
        export_config.setdefault("source", {})["merged_model_dir"] = str(merged_dir)
        export_config["source"].pop("adapter_dir", None)
        export_manifest = export_edge_gallery_model(export_config, dry_run=False)
        plan["public_export"]["executed"] = True
        plan["public_export"]["manifest"] = export_manifest
        candidate = _find_single_litertlm(Path(plan["public_export"]["output_dir"]))
        if candidate is None:
            raise Gemma4MobileMTPPipelineError(
                "Public export did not produce exactly one .litertlm candidate."
            )
        plan["package"]["target_litertlm"] = str(candidate)

    if execute_exact_topology_export:
        if not plan["exact_topology"]["enabled"]:
            raise Gemma4MobileMTPPipelineError(
                "pipeline.exact_topology.enabled=false; exact-topology export was requested."
            )
        if not plan["exact_topology"]["official_litertlm"]:
            raise Gemma4MobileMTPPipelineError(
                "Exact-topology export requires source.base_litertlm or --base-litertlm."
            )
        merged_dir = Path(plan["merge"]["merged_model_dir"])
        if not merged_dir.is_dir():
            raise Gemma4MobileMTPPipelineError(
                f"Merged best-checkpoint model is missing: {merged_dir}. Run --execute-merge first."
            )
        _run_command(
            list(plan["exact_topology"]["command"]),
            logs_dir / "exact_topology_export.log",
            cwd=base.parent,
        )
        report_path = (
            Path(plan["exact_topology"]["output_dir"])
            / "checkpoint_official_topology_report.json"
        )
        if not report_path.is_file():
            raise Gemma4MobileMTPPipelineError(
                f"Exact-topology exporter did not write its report: {report_path}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(report.get("final_artifact_gate_pass")):
            raise Gemma4MobileMTPPipelineError(
                "Exact-topology exporter report did not pass all artifact gates."
            )
        mtp = report.get("mtp_preservation") or {}
        if not bool(mtp.get("byte_exact")):
            raise Gemma4MobileMTPPipelineError(
                "Exact-topology exporter did not prove byte-exact MTP preservation."
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
        plan["exact_topology"]["executed"] = True
        plan["exact_topology"]["report"] = report
        plan["package"]["executed"] = True
        plan["package"]["manifest"] = final_manifest or report

    if compose_package:
        if plan["mtp"]["weight_source"] != "official":
            raise Gemma4MobileMTPPipelineError(
                "Legacy composition only preserves official MTP weights; use the "
                "exact-topology export for mtp.weight_source=trained."
            )
        base_package = plan["package"].get("base_litertlm")
        target_package = plan["package"].get("target_litertlm")
        target_section = plan["package"].get("target_section")
        if not base_package:
            raise Gemma4MobileMTPPipelineError(
                "Package composition requires package.base_litertlm (official package)."
            )
        if bool(target_package) == bool(target_section):
            raise Gemma4MobileMTPPipelineError(
                "Package composition requires exactly one target_litertlm or target_section."
            )
        manifest = compose_with_default_mtp(
            base_litertlm=base_package,
            target_litertlm=target_package,
            target_section=target_section,
            output_litertlm=plan["package"]["output_litertlm"],
            target_model_type=plan["package"]["target_model_type"],
            mtp_model_type=plan["package"]["mtp_model_type"],
            force=force,
        )
        plan["package"]["executed"] = True
        plan["package"]["manifest"] = manifest

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
