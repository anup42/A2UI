"""Gemma 3 270M QAT -> best checkpoint -> public LiteRT-LM pipeline.

Gemma 3 270M is handled separately from Gemma 4 E2B.  Its checked-in QAT
profile is weight-only INT8 with floating-point activations and its LiteRT
export profile is the public
``dynamic_wi8_afp32`` recipe.  There is no Gemma 4 MTP drafter section in this
model family, so this pipeline explicitly records MTP as not applicable rather
than attaching an unrelated assistant model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.export.edge_gallery import export_edge_gallery_model
from ir_training.export.merge_lora import merge_lora_adapter


class Gemma270MPipelineError(RuntimeError):
    """Raised when a Gemma 270M pipeline stage cannot be completed safely."""


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _resolve_optional(value: Any, base: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return resolve_path(str(value), base)


def _best_checkpoint_dir(training_config: dict[str, Any], base: Path) -> Path:
    run_cfg = _section(training_config, "run")
    golden_cfg = _section(training_config, "golden_eval")
    if golden_cfg.get("best_checkpoint_dir"):
        return resolve_path(str(golden_cfg["best_checkpoint_dir"]), base)
    return resolve_path(
        str(run_cfg.get("output_dir", "runs/gemma3_270m_ir_qat_sft"))
        + "/best_golden_checkpoint",
        base,
    )


def _checkpoint_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        (path / marker).is_file()
        for marker in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
            "pytorch_model.bin",
            "config.json",
        )
    )


def _find_single_litertlm(directory: Path) -> Path | None:
    files = sorted(directory.rglob("*.litertlm")) if directory.is_dir() else []
    return files[0] if len(files) == 1 else None


def build_pipeline_plan(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    best_checkpoint_override: str | Path | None = None,
    output_litertlm_override: str | Path | None = None,
    artifact_override: str | Path | None = None,
    official_litertlm_override: str | Path | None = None,
) -> dict[str, Any]:
    """Build a no-download/no-training plan."""

    base = training_root()
    pipeline_cfg = _section(config, "pipeline")
    training_config_path = resolve_path(
        str(
            pipeline_cfg.get(
                "training_config", "configs/models/gemma3_270m_ir_qat_sft.yaml"
            )
        ),
        base,
    )
    if not training_config_path.is_file():
        raise Gemma270MPipelineError(
            f"Training config does not exist: {training_config_path}"
        )
    training_config = load_yaml(training_config_path)
    model_cfg = _section(training_config, "model")
    qat_cfg = _section(training_config, "qat")
    run_cfg = _section(training_config, "run")
    model_id = str(model_cfg.get("model_id") or "")
    training_output = resolve_path(
        str(run_cfg.get("output_dir", "runs/gemma3_270m_ir_qat_sft")), base
    )
    best_checkpoint = (
        resolve_path(str(best_checkpoint_override), base)
        if best_checkpoint_override
        else _best_checkpoint_dir(training_config, base)
    )

    pipeline_output = resolve_path(
        str(pipeline_cfg.get("output_dir", "outputs/pipelines/gemma3_270m_litertlm")),
        base,
    )
    source_cfg = _section(pipeline_cfg, "source")
    merged_model_dir = _resolve_optional(source_cfg.get("merged_model_dir"), base)
    if merged_model_dir is None:
        merged_model_dir = pipeline_output / "merged_best_hf"
    official_litertlm = (
        resolve_path(str(official_litertlm_override), base)
        if official_litertlm_override
        else _resolve_optional(source_cfg.get("official_litertlm"), base)
    )
    export_cfg = _section(pipeline_cfg, "export")
    export_config_path = resolve_path(
        str(
            export_cfg.get(
                "config", "configs/export/litertlm_gemma270m_int8.yaml"
            )
        ),
        base,
    )
    export_output = _resolve_optional(export_cfg.get("output_dir"), base)
    if export_output is None:
        export_output = pipeline_output / "public_export"
    artifact_path = (
        resolve_path(str(artifact_override), base)
        if artifact_override
        else _resolve_optional(export_cfg.get("artifact"), base)
    )
    if artifact_path is None:
        artifact_path = pipeline_output / "gemma3_270m_qat_int8.litertlm"
    output_litertlm = (
        resolve_path(str(output_litertlm_override), base)
        if output_litertlm_override
        else artifact_path
    )
    exact_cfg = _section(pipeline_cfg, "exact_topology")
    exact_enabled = bool(exact_cfg.get("enabled", True))
    exact_output_dir = _resolve_optional(exact_cfg.get("output_dir"), base)
    if exact_output_dir is None:
        exact_output_dir = pipeline_output / "exact_topology_export"
    official_base_model_id = str(
        exact_cfg.get("official_base_model_id") or model_id
    )
    official_artifact_sha256 = str(
        exact_cfg.get("official_artifact_sha256") or ""
    ).strip().lower()
    exact_command = [
        sys.executable,
        str(base / "scripts" / "build_checkpoint_official_topology.py"),
        str(official_litertlm)
        if official_litertlm
        else "<official-litertlm-required>",
        "--checkpoint",
        str(merged_model_dir),
        "--family",
        "gemma3_270m",
        "--model-type",
        str(exact_cfg.get("model_type", "TF_LITE_PREFILL_DECODE")),
        "--training-config",
        str(training_config_path),
        "--official-base-model-id",
        official_base_model_id,
        "--official-artifact-sha256",
        official_artifact_sha256 or "<official-artifact-sha256-required>",
        "--output-dir",
        str(exact_output_dir),
        "--package-output",
        str(output_litertlm),
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
    android_cfg = _section(pipeline_cfg, "android")
    android_output_dir = _resolve_optional(android_cfg.get("output_dir"), base)
    if android_output_dir is None:
        android_output_dir = pipeline_output / "android_gpu_parity"
    android_command = [
        sys.executable,
        str(base / "scripts" / "benchmark_android_litertlm_gpu_parity.py"),
        "--official",
        str(official_litertlm)
        if official_litertlm
        else "<official-litertlm-required>",
        "--candidate",
        str(output_litertlm),
        "--output-dir",
        str(android_output_dir),
        "--max-num-tokens",
        str(int(android_cfg.get("max_num_tokens", 2048))),
        "--output-tokens",
        str(int(android_cfg.get("output_tokens", 64))),
        "--warm-runs",
        str(int(android_cfg.get("warm_runs", 1))),
        "--max-throughput-regression-percent",
        str(float(android_cfg.get("max_throughput_regression_percent", 10.0))),
    ]
    if str(android_cfg.get("prompt") or "").strip():
        android_command.extend(["--prompt", str(android_cfg["prompt"])])

    validation: list[dict[str, str]] = []
    if "gemma-3-270m" not in model_id.lower():
        validation.append(
            {
                "severity": "error",
                "code": "unexpected_model",
                "message": f"Expected google/gemma-3-270m, got {model_id or '<missing>'}.",
            }
        )
    if not bool(qat_cfg.get("enabled", False)):
        validation.append(
            {
                "severity": "error",
                "code": "qat_disabled",
                "message": "The referenced Gemma 270M training config must enable qat.enabled.",
            }
        )
    if (
        int(qat_cfg.get("weight_bits", 8)) != 8
        or int(qat_cfg.get("activation_bits", 0)) < 16
        or str(qat_cfg.get("quantizer") or "").strip().lower() != "ste_ai_edge"
        or not bool(qat_cfg.get("quantize_embeddings", False))
    ):
        validation.append(
            {
                "severity": "error",
                "code": "unexpected_precision",
                "message": (
                    "The official Gemma 3 270M Q8 graph is weight-only INT8 "
                    "including its embedding table, with floating-point "
                    "activation edges (WI8/AFP32)."
                ),
            }
        )
    mtp_cfg = _section(pipeline_cfg, "mtp")
    if bool(mtp_cfg.get("enabled", False)):
        validation.append(
            {
                "severity": "error",
                "code": "mtp_not_applicable",
                "message": "Gemma 3 270M has no Gemma 4 MTP drafter section; keep mtp.enabled=false.",
            }
        )
    if not export_config_path.is_file():
        validation.append(
            {
                "severity": "error",
                "code": "missing_export_config",
                "message": f"LiteRT export config does not exist: {export_config_path}",
            }
        )
    if exact_enabled and official_litertlm is None:
        validation.append(
            {
                "severity": "warning",
                "code": "missing_official_package",
                "message": (
                    "Provide source.official_litertlm or --official-litertlm before "
                    "running the exact-topology export."
                ),
            }
        )

    return {
        "pipeline_id": str(pipeline_cfg.get("id", "gemma3_270m_qat_litertlm")),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "training": {
            "config": str(training_config_path),
            "command": [
                sys.executable,
                str(base / "scripts" / "train_sft.py"),
                "--config",
                str(training_config_path),
            ],
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
            "packed_int8_output": False,
            "assistant_modified": False,
        },
        "export": {
            "enabled": bool(export_cfg.get("enabled", True)),
            "config": str(export_config_path),
            "output_dir": str(export_output),
            "artifact": str(output_litertlm),
            "recipe": "dynamic_wi8_afp32",
            "runtime": "litertlm",
            "note": (
                "This is the public Gemma 270M INT8 LiteRT export path; it is "
                "not an INT4/Q4_0 artifact."
            ),
        },
        "exact_topology": {
            "enabled": exact_enabled,
            "family": "gemma3_270m",
            "official_base_model_id": official_base_model_id,
            "official_artifact_sha256": official_artifact_sha256 or None,
            "official_litertlm": str(official_litertlm)
            if official_litertlm
            else None,
            "merged_checkpoint": str(merged_model_dir),
            "training_config": str(training_config_path),
            "model_type": str(exact_cfg.get("model_type", "TF_LITE_PREFILL_DECODE")),
            "output_dir": str(exact_output_dir),
            "output_litertlm": str(output_litertlm),
            "command": exact_command,
            "training_executed": False,
            "requires_complete_127_weight_mapping": True,
        },
        "package": {
            "format": "litertlm",
            "expected_model_type": "TF_LITE_PREFILL_DECODE",
            "artifact": str(output_litertlm),
            "mtp": {
                "enabled": False,
                "status": "not_applicable_for_gemma3_270m",
            },
        },
        "android_gpu": {
            "delegate": "gpu",
            "mtp_flag": False,
            "device_validation": "required_after_export",
            "parity_runner": str(base / "scripts" / "benchmark_android_litertlm_gpu_parity.py"),
            "output_dir": str(android_output_dir),
            "command": android_command,
        },
        "validation": {
            "ok": not any(item["severity"] == "error" for item in validation),
            "issues": validation,
        },
        "limitations": [
            "MTP is not available for this Gemma 3 270M pipeline.",
            "The configured LiteRT export is WI8/AFP32; it is not Q4_0/INT4.",
            "A real Android GPU run is still required for runtime claims.",
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
        raise Gemma270MPipelineError(
            f"Command failed with exit code {process.returncode}; see {log_path}"
        )


def run_pipeline(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    execute_training: bool = False,
    execute_merge: bool = False,
    execute_export: bool = False,
    execute_exact_topology_export: bool = False,
    validate_android_gpu: bool = False,
    adb_override: str | Path | None = None,
    serial_override: str | None = None,
    best_checkpoint_override: str | Path | None = None,
    output_litertlm_override: str | Path | None = None,
    artifact_override: str | Path | None = None,
    official_litertlm_override: str | Path | None = None,
) -> dict[str, Any]:
    plan = build_pipeline_plan(
        config,
        config_path=config_path,
        best_checkpoint_override=best_checkpoint_override,
        output_litertlm_override=output_litertlm_override,
        artifact_override=artifact_override,
        official_litertlm_override=official_litertlm_override,
    )
    stage_errors = [
        item for item in plan["validation"]["issues"] if item["severity"] == "error"
    ]
    if stage_errors and (
        execute_training
        or execute_merge
        or execute_export
        or execute_exact_topology_export
        or validate_android_gpu
    ):
        raise Gemma270MPipelineError(json.dumps(stage_errors, ensure_ascii=False))

    if execute_export and execute_exact_topology_export:
        raise Gemma270MPipelineError(
            "Choose either the standalone generic export or the official exact-topology "
            "export for one output path, not both."
        )

    base = training_root()
    pipeline_output = Path(plan["export"]["output_dir"]).parent
    logs_dir = pipeline_output / "logs"
    if execute_training:
        _run_command(plan["training"]["command"], logs_dir / "train_sft.log", cwd=base.parent)
        plan["training"]["best_checkpoint_ready"] = _checkpoint_ready(
            Path(plan["training"]["best_checkpoint"])
        )
        if not plan["training"]["best_checkpoint_ready"]:
            raise Gemma270MPipelineError(
                "Training completed but the golden best checkpoint was not found at "
                f"{plan['training']['best_checkpoint']}"
            )

    if execute_merge:
        checkpoint = Path(plan["training"]["best_checkpoint"])
        if not _checkpoint_ready(checkpoint):
            raise Gemma270MPipelineError(
                f"Best checkpoint is missing or not a PEFT model: {checkpoint}"
            )
        merged_destination = Path(plan["merge"]["merged_model_dir"])
        if merged_destination.exists() and any(merged_destination.iterdir()):
            raise Gemma270MPipelineError(
                "Refusing to merge into a non-empty directory; choose a new "
                f"merged_model_dir or clear it explicitly: {merged_destination}"
            )
        merged = merge_lora_adapter(
            base_model_id=str(plan["merge"]["base_model_id"]),
            adapter_dir=checkpoint,
            output_dir=merged_destination,
            model_loader="auto_causal_lm",
            dtype="bfloat16",
            trust_remote_code=False,
            processor_model_id=str(plan["merge"]["base_model_id"]),
            training_config_path=plan["training"]["config"],
        )
        plan["merge"]["merged_model_dir"] = str(merged)
        plan["merge"]["executed"] = True

    if execute_export:
        if not plan["export"]["enabled"]:
            raise Gemma270MPipelineError("pipeline.export.enabled=false; export was explicitly requested.")
        merged_dir = Path(plan["merge"]["merged_model_dir"])
        if not merged_dir.is_dir():
            raise Gemma270MPipelineError(
                f"Merged best-checkpoint model is missing: {merged_dir}. Run --execute-merge first."
            )
        export_root = Path(plan["export"]["output_dir"])
        litert_output_dir = export_root / "litertlm"
        export_config = load_yaml(plan["export"]["config"])
        export_config.setdefault("run", {})["output_dir"] = str(export_root)
        export_config.setdefault("source", {})["base_model_id"] = plan["merge"]["base_model_id"]
        export_config["source"]["merged_model_dir"] = str(merged_dir)
        export_config["source"].pop("adapter_dir", None)
        export_config.setdefault("export", {})["litert_output_dir"] = str(litert_output_dir)
        export_manifest = export_edge_gallery_model(export_config, dry_run=False)
        candidate = _find_single_litertlm(litert_output_dir)
        if candidate is None:
            raise Gemma270MPipelineError(
                f"Expected exactly one .litertlm under {litert_output_dir}."
            )
        artifact = Path(plan["export"]["artifact"])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.exists() and artifact.resolve() != candidate.resolve():
            raise Gemma270MPipelineError(
                f"Refusing to overwrite existing artifact: {artifact}"
            )
        if artifact.resolve() != candidate.resolve():
            shutil.copy2(candidate, artifact)
        plan["export"]["executed"] = True
        plan["export"]["candidate"] = str(candidate)
        plan["export"]["manifest"] = export_manifest
        plan["package"]["artifact"] = str(artifact)

    if execute_exact_topology_export:
        if not plan["exact_topology"]["enabled"]:
            raise Gemma270MPipelineError(
                "pipeline.exact_topology.enabled=false; exact-topology export was requested."
            )
        if not plan["exact_topology"]["official_litertlm"]:
            raise Gemma270MPipelineError(
                "Exact-topology export requires source.official_litertlm or --official-litertlm."
            )
        merged_dir = Path(plan["merge"]["merged_model_dir"])
        if not merged_dir.is_dir():
            raise Gemma270MPipelineError(
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
            raise Gemma270MPipelineError(
                f"Exact-topology exporter did not write its report: {report_path}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(report.get("final_artifact_gate_pass")):
            raise Gemma270MPipelineError(
                "Exact-topology exporter report did not pass all artifact gates."
            )
        plan["exact_topology"]["executed"] = True
        plan["exact_topology"]["report"] = report
        plan["package"]["artifact"] = plan["exact_topology"]["output_litertlm"]

    if validate_android_gpu:
        official_package = plan["exact_topology"].get("official_litertlm")
        candidate_package = plan["package"].get("artifact")
        if not official_package or not Path(official_package).is_file():
            raise Gemma270MPipelineError(
                "Android GPU parity requires the official Q8 LiteRT-LM package."
            )
        if not candidate_package or not Path(candidate_package).is_file():
            raise Gemma270MPipelineError(
                "Android GPU parity requires an exported candidate package first."
            )
        command = list(plan["android_gpu"]["command"])
        if adb_override:
            command.extend(["--adb", str(Path(adb_override).expanduser().resolve())])
        if serial_override:
            command.extend(["--serial", str(serial_override)])
        _run_command(
            command,
            logs_dir / "android_gpu_parity.log",
            cwd=base.parent,
        )
        report_path = (
            Path(plan["android_gpu"]["output_dir"])
            / "android_litertlm_gpu_parity_report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool((report.get("comparison") or {}).get("overall_pass")):
            raise Gemma270MPipelineError(
                "Android GPU parity report did not pass structure and throughput gates."
            )
        plan["android_gpu"]["executed"] = True
        plan["android_gpu"]["report"] = report

    return plan
