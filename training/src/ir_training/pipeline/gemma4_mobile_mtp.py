"""Gemma 4 E2B mobile QAT -> best checkpoint -> MTP package orchestration.

The pipeline is deliberately split into explicit stages:

1. QAT SFT (the existing ``train_sft.py`` path) saves a golden-set best
   adapter;
2. the best adapter is merged back into the floating-point base model;
3. an optional public LiteRT Torch export produces a *standalone* candidate;
4. a compatible target TFLite section is composed with the official package,
   preserving the default ``tf_lite_mtp_drafter`` section byte-for-byte; and
5. package/device validation records what was actually proven.

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
from ir_training.export.edge_gallery import export_edge_gallery_model
from ir_training.export.litertlm_mtp import compose_with_default_mtp
from ir_training.export.merge_lora import merge_lora_adapter


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
        str(run.get("output_dir", "runs/gemma4_e2b_ir_qat_sft"))
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
        "training_config", "configs/models/gemma4_e2b_ir_qat_sft.yaml"
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
        str(train_run_cfg.get("output_dir", "runs/gemma4_e2b_ir_qat_sft")), base
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

    training_script = base / "scripts" / "train_sft.py"
    train_command = [sys.executable, str(training_script), "--config", str(training_config_path)]
    export_config_value = public_export_cfg.get(
        "config", "configs/export/edge_gallery_gemma4_e2b.yaml"
    )
    export_config_path = resolve_path(str(export_config_value), base)
    model_id = str(model_cfg.get("model_id") or "")
    mtp_cfg = _section(pipeline_cfg, "mtp")
    target_model_type = str(
        mtp_cfg.get("target_model_type", "tf_lite_prefill_decode")
    )
    mtp_model_type = str(mtp_cfg.get("model_type", "tf_lite_mtp_drafter"))
    public_export_enabled = bool(public_export_cfg.get("enabled", False))

    validation: list[dict[str, str]] = []
    if not bool(qat_cfg.get("enabled", False)):
        validation.append(
            {
                "severity": "error",
                "code": "qat_disabled",
                "message": "The referenced training config must enable qat.enabled.",
            }
        )
    if not bool(mtp_cfg.get("enabled", True)):
        validation.append(
            {
                "severity": "error",
                "code": "mtp_disabled",
                "message": "mtp.enabled must remain true for the requested Android path.",
            }
        )
    if base_litertlm is None:
        validation.append(
            {
                "severity": "warning",
                "code": "missing_base_package",
                "message": "Provide the official base_litertlm package before --compose.",
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
    if not public_export_enabled and target_litertlm is None and target_section is None:
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
            "qat_profile": qat_cfg.get("profile"),
            "output_dir": str(training_output),
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_ready": _checkpoint_ready(best_checkpoint),
            "best_checkpoint_required": True,
        },
        "merge": {
            "base_model_id": model_id,
            "adapter_dir": str(best_checkpoint),
            "merged_model_dir": str(merged_model_dir),
            "continued_qat_performed": False,
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
        "package": {
            "base_litertlm": str(base_litertlm) if base_litertlm else None,
            "target_litertlm": str(target_litertlm) if target_litertlm else None,
            "target_section": str(target_section) if target_section else None,
            "output_litertlm": str(output_litertlm),
            "target_model_type": target_model_type,
            "mtp_model_type": mtp_model_type,
            "mtp_enabled": bool(mtp_cfg.get("enabled", True)),
            "mtp_assistant_model_id": mtp_cfg.get("assistant_model_id"),
            "target_export_authority": "compatible_mobile_section_required",
        },
        "android_gpu": {
            "mtp_flag": True,
            "delegate": "gpu",
            "device_validation": "required_after_packaging",
        },
        "validation": {
            "ok": not any(item["severity"] == "error" for item in validation),
            "issues": validation,
        },
        "limitations": [
            "The released MTP drafter is preserved, not trained or adapted.",
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
        )
        log.write(process.stdout or "")
    if process.returncode != 0:
        raise Gemma4MobileMTPPipelineError(
            f"Command failed with exit code {process.returncode}; see {log_path}"
        )


def run_pipeline(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    execute_training: bool = False,
    execute_merge: bool = False,
    execute_public_export: bool = False,
    compose_package: bool = False,
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
    if stage_errors:
        if execute_training or execute_merge or execute_public_export or compose_package:
            raise Gemma4MobileMTPPipelineError(json.dumps(stage_errors, ensure_ascii=False))

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
        )
        plan["merge"]["merged_model_dir"] = str(merged)
        plan["merge"]["executed"] = True

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

    if compose_package:
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

    return plan
