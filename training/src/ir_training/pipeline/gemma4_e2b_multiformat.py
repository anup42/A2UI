"""Dry-run-first Gemma 4 E2B A2UI Express multi-format pipeline.

The official lane delegates to the retained-scale mobile QAT pipeline. Public
W32/W16/W8/W4 comparison lanes use the installed LiteRT Torch exporter and are
deliberately labelled non-official.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from ir_training.common.config import load_yaml, repo_root, resolve_path, training_root
from ir_training.eval.evaluation_evidence import (
    EvaluationEvidenceError,
    validate_litertlm_precision_contract,
    verify_evaluation_evidence,
)
from ir_training.eval.external_runner import validate_external_runner_config
from ir_training.eval.golden_set import (
    validate_prepared_golden_contract,
)
from ir_training.eval.tensorboard_logging import resolve_tensorboard_root
from ir_training.export.edge_gallery import export_edge_gallery_model
from ir_training.pipeline.gemma4_mobile_mtp import (
    _validate_retained_scale_export_report,
    build_pipeline_plan as build_official_pipeline_plan,
    run_pipeline as run_official_pipeline,
)


EXPECTED_VARIANTS: dict[str, dict[str, Any]] = {
    "w32": {"requested_bits": 32, "artifact_kind": "fp32", "recipe": "none"},
    "w16": {"requested_bits": 16, "artifact_kind": "fp16", "recipe": "none"},
    "w8": {
        "requested_bits": 8,
        "artifact_kind": "int8",
        "recipe": "dynamic_wi8_afp32",
    },
    "w4": {
        "requested_bits": 4,
        "artifact_kind": "mixed_w4_w8",
        "recipe": "gemma4_mixed48_b32",
    },
}
STAGE_ORDER = (
    "prepare_golden",
    "training",
    "checkpoint_evaluation",
    "merge",
    "mtp_training",
    "official_export",
    "public_exports",
    "litertlm_evaluations",
    "scorecard",
)


class Gemma4E2BMultiformatPipelineError(RuntimeError):
    """Raised when a requested stage cannot be proven safe or complete."""


def _section(value: dict[str, Any], name: str) -> dict[str, Any]:
    section = value.get(name)
    return section if isinstance(section, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _training_path(value: Any) -> Path:
    return resolve_path(str(value), training_root())


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()


def _file_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _checkpoint_ready(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).is_file()
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
            "config.json",
        )
    )


def _one_litertlm(path: Path) -> Path | None:
    candidates = sorted(path.rglob("*.litertlm")) if path.is_dir() else []
    return candidates[0] if len(candidates) == 1 else None


def _variant_export_config(
    *,
    pipeline: dict[str, Any],
    variant_id: str,
    variant: dict[str, Any],
    merged_model_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    export = _section(pipeline, "export")
    extra_flags = [str(value) for value in (variant.get("extra_flags") or [])]
    extra_flags.extend(
        [
            f"--cache_length={int(export.get('cache_length', 4096))}",
            f"--prefill_lengths={str(export.get('prefill_lengths', '128,512,1024,2048'))}",
            "--enable_gpu_dynamic_prefill=True",
            "--enable_gpu_dynamic_cache=True",
        ]
    )
    return {
        "run": {
            "id": f"{pipeline.get('id', 'gemma4_e2b_multiformat')}_{variant_id}",
            "output_dir": str(output_dir),
        },
        "source": {
            "base_model_id": "google/gemma-4-E2B-it-qat-mobile-transformers",
            "merged_model_dir": str(merged_model_dir),
        },
        "export": {
            "runtime": "litertlm",
            "format": "litertlm",
            "litert_output_dir": str(output_dir / "litertlm"),
            "command": str(export.get("command", "litert-torch")),
            "min_app_version": "1.1.0",
            "max_input_tokens": int(export.get("max_input_tokens", 4096)),
            "max_output_tokens": int(export.get("max_output_tokens", 4096)),
            "externalize_embedder": bool(export.get("externalize_embedder", True)),
            "jinja_chat_template_override": str(
                export.get(
                    "jinja_chat_template_override",
                    "training/configs/export/gemma4_e2b_training_minijinja.jinja",
                )
            ),
            "quantization_recipe": str(variant.get("quantization_recipe") or ""),
            "extra_flags": extra_flags,
        },
        "android": {
            "package_name": f"gemma4_e2b_a2ui_express_{variant_id}",
            "display_name": f"Gemma 4 E2B A2UI Express {variant_id.upper()}",
            "stage": "stage3_ir",
        },
    }


def _export_command(export_config: dict[str, Any]) -> list[str]:
    export = _section(export_config, "export")
    source = _section(export_config, "source")
    command = [
        str(export.get("command", "litert-torch")),
        "export_hf",
        f"--model={source['merged_model_dir']}",
        f"--output_dir={export['litert_output_dir']}",
        f"--quantization_recipe={export['quantization_recipe']}",
    ]
    if export.get("jinja_chat_template_override"):
        command.append(
            "--jinja_chat_template_override="
            + str(export["jinja_chat_template_override"])
        )
    if export.get("externalize_embedder") is True:
        command.append("--externalize_embedder")
    command.extend(str(value) for value in export.get("extra_flags") or [])
    return command


def build_pipeline_plan(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    run_id_override: str | None = None,
    source_safetensors_override: str | Path | None = None,
    base_litertlm_override: str | Path | None = None,
    num_gpus_override: int | None = None,
    gpu_ids_override: str | None = None,
    mtp_enabled_override: bool | None = None,
    runner_config_override: str | Path | None = None,
) -> dict[str, Any]:
    pipeline = _section(config, "pipeline")
    golden = _section(pipeline, "golden")
    training = _section(pipeline, "training")
    official = _section(pipeline, "official_qat")
    mtp = copy.deepcopy(_section(pipeline, "mtp"))
    evaluation = _section(pipeline, "evaluation")
    public_variants = _section(pipeline, "public_variants")

    run_id = str(run_id_override or pipeline.get("run_id") or "").strip()
    if not re.fullmatch(
        r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9-])?", run_id
    ) or re.fullmatch(
        r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", run_id, re.IGNORECASE
    ):
        raise Gemma4E2BMultiformatPipelineError(
            "pipeline.run_id/--run-id must be one safe path component using "
            "letters, digits, dots, underscores, and hyphens."
        )
    if mtp_enabled_override is not None:
        mtp["enabled"] = bool(mtp_enabled_override)
    num_gpus = int(num_gpus_override or training.get("num_gpus", 1))
    gpu_ids = gpu_ids_override if gpu_ids_override is not None else training.get("gpu_ids")

    output_root = _training_path(
        pipeline.get("output_root", "outputs/pipelines/gemma4_e2b_a2ui_express_multiformat")
    ) / run_id
    runs_root = _repo_path(
        training.get("runs_root", "training/runs/gemma4_e2b_a2ui_express_multiformat")
    )
    trainer_run_root = runs_root / run_id
    resolved_training_config = trainer_run_root / "launch" / "resolved_training_config.yaml"
    best_checkpoint = trainer_run_root / "best_golden_checkpoint"
    training_config = _training_path(
        training.get("config", "configs/models/gemma4_e2b_a2ui_express_official_qat.yaml")
    )
    portable_launcher = _training_path(
        training.get("portable_launcher", "scripts/run_gemma4_mobile_qat.py")
    )
    golden_config = _training_path(golden.get("dataset_config", ""))
    golden_split = _training_path(golden.get("split", ""))
    golden_source = _training_path(golden.get("source_genui", ""))
    golden_responses = _training_path(golden.get("source_responses", ""))
    tensorboard_root = resolve_tensorboard_root(
        _section(pipeline, "tensorboard").get("root", "tensorboard")
    )
    source_safetensors = (
        _repo_path(source_safetensors_override)
        if source_safetensors_override
        else (
            _repo_path(training.get("source_safetensors"))
            if str(training.get("source_safetensors") or "").strip()
            else None
        )
    )
    base_litertlm = (
        _repo_path(base_litertlm_override)
        if base_litertlm_override
        else (
            _repo_path(official.get("base_litertlm"))
            if str(official.get("base_litertlm") or "").strip()
            else None
        )
    )
    merged_model_dir = output_root / "merged_best_hf"
    mtp_source_training_config = _training_path(
        mtp.get("training_config", "configs/models/gemma4_e2b_mtp_drafter_qat.yaml")
    )
    mtp_run_root = trainer_run_root / "mtp_drafter"
    mtp_resolved_training_config = (
        trainer_run_root / "launch" / "resolved_mtp_training_config.yaml"
    )
    mtp_trained_checkpoint = mtp_run_root / "best_checkpoint"
    mtp_resolved_config_sha256: str | None = None
    if str(mtp.get("weight_source", "official")) == "trained":
        mtp["source_training_config"] = str(mtp_source_training_config)
        mtp["training_config"] = str(mtp_resolved_training_config)
        mtp["run_root"] = str(mtp_run_root)
        mtp["trained_checkpoint"] = str(mtp_trained_checkpoint)
        if mtp_source_training_config.is_file():
            mtp_training_payload = copy.deepcopy(load_yaml(mtp_source_training_config))
            mtp_training_payload.setdefault("run", {}).update(
                {"id": run_id, "output_dir": str(mtp_run_root)}
            )
            mtp_training_payload.setdefault("target", {})["model_id_or_path"] = str(
                merged_model_dir
            )
            mtp_training_payload.setdefault("assistant", {})[
                "best_checkpoint_dir"
            ] = str(mtp_trained_checkpoint)
            mtp_training_payload.setdefault("training", {}).update(
                {
                    "tensorboard_root": str(tensorboard_root),
                    "tensorboard_subdir": "mtp_training",
                }
            )
            mtp_resolved_text = yaml.safe_dump(
                mtp_training_payload, sort_keys=False, allow_unicode=True
            )
            mtp_resolved_config_sha256 = hashlib.sha256(
                mtp_resolved_text.encode("utf-8")
            ).hexdigest()
            mtp["resolved_training_config_sha256"] = mtp_resolved_config_sha256
    official_output_dir = output_root / "exports" / "official_qat"
    official_artifact = official_output_dir / str(
        official.get("output_name", "gemma4_e2b_official_retained_w2w4w8.litertlm")
    )
    official_report = official_output_dir / "retained_scale_export_report.json"
    evaluation_root = output_root / str(evaluation.get("output_dir", "evaluations"))

    prepare_command = [
        sys.executable,
        str(training_root() / "scripts" / "prepare_dataset.py"),
        "--config",
        str(golden_config),
    ]
    training_command = [
        sys.executable,
        str(portable_launcher),
        "--config",
        str(training_config),
        "--run-id",
        run_id,
        "--runs-root",
        str(runs_root),
        "--num-gpus",
        str(num_gpus),
        "--execute",
    ]
    if source_safetensors is not None:
        training_command.extend(["--source-safetensors", str(source_safetensors)])
    if gpu_ids is not None and str(gpu_ids).strip():
        training_command.extend(["--gpu-ids", str(gpu_ids)])

    checkpoint_script = _training_path(
        evaluation.get("checkpoint_script", "scripts/evaluate_checkpoint_on_golden.py")
    )
    litertlm_script = _training_path(
        evaluation.get("litertlm_script", "scripts/evaluate_litertlm_on_golden.py")
    )
    package_audit_script = _training_path(
        evaluation.get("package_audit_script", "scripts/audit_litertlm_package.py")
    )
    runner_config = (
        _repo_path(runner_config_override)
        if runner_config_override
        else _training_path(
            evaluation.get(
                "litertlm_runner_config",
                "configs/eval/litertlm_external_runner.example.yaml",
            )
        )
    )
    weights_config = _training_path(golden.get("weights_config", "../dataset/configs/run.yaml"))
    required_rows = int(golden.get("required_rows", 32))
    common_eval = [
        "--split",
        str(golden_split),
        "--tensorboard-root",
        str(tensorboard_root),
        "--run-id",
        run_id,
        "--max-rows",
        str(required_rows),
        "--required-rows",
        str(required_rows),
        "--max-input-tokens",
        str(int(golden.get("max_input_tokens", 4096))),
        "--max-new-tokens",
        str(int(golden.get("max_new_tokens", 4096))),
        "--weights-config",
        str(weights_config),
        "--metric-version",
        str(golden.get("metric_version", "dual")),
    ]
    checkpoint_eval_command = [
        sys.executable,
        str(checkpoint_script),
        "--config",
        str(resolved_training_config),
        "--checkpoint",
        str(best_checkpoint),
        "--checkpoint-kind",
        "adapter",
        "--output-dir",
        str(evaluation_root / "checkpoint"),
        "--evaluation-name",
        "checkpoint",
        "--step",
        "<checkpoint-step>",
        *common_eval,
    ]

    variants: dict[str, Any] = {}
    for variant_id, expected in EXPECTED_VARIANTS.items():
        variant = _section(public_variants, variant_id)
        export_dir = output_root / "exports" / variant_id
        export_config = _variant_export_config(
            pipeline=pipeline,
            variant_id=variant_id,
            variant=variant,
            merged_model_dir=merged_model_dir,
            output_dir=export_dir,
        )
        eval_name = f"litertlm_{variant_id}"
        eval_dir = evaluation_root / eval_name
        package_report = eval_dir / "package_inspection.json"
        variants[variant_id] = {
            "enabled": bool(variant.get("enabled", True)),
            "requested_bits": variant.get("requested_bits"),
            "artifact_kind": variant.get("artifact_kind"),
            "quantization_recipe": variant.get("quantization_recipe"),
            "official_format": False,
            "support": (
                "experimental_public_fp16"
                if variant_id == "w16"
                else (
                    "public_supported_mixed_w4_w8_not_pure_int4"
                    if variant_id == "w4"
                    else "public_supported"
                )
            ),
            "output_dir": str(export_dir),
            "litertlm_dir": str(export_dir / "litertlm"),
            "artifact": str(_one_litertlm(export_dir / "litertlm") or ""),
            "resolved_export_config": export_config,
            "converter_command": _export_command(export_config),
            "evaluation": {
                "name": eval_name,
                "output_dir": str(eval_dir),
                "command_template": [
                    sys.executable,
                    str(litertlm_script),
                    "--model",
                    f"<single-litertlm-under:{export_dir / 'litertlm'}>",
                    "--runner-config",
                    str(runner_config),
                    "--output-dir",
                    str(eval_dir),
                    "--evaluation-name",
                    eval_name,
                    "--step",
                    "<checkpoint-step>",
                    *common_eval,
                ],
            },
            "package_validation": {
                "report": str(package_report),
                "command_template": [
                    sys.executable,
                    str(package_audit_script),
                    f"<single-litertlm-under:{export_dir / 'litertlm'}>",
                    "--include-hashes",
                    "--output",
                    str(package_report),
                ],
            },
            "contract_matches": bool(
                variant.get("requested_bits") == expected["requested_bits"]
                and variant.get("artifact_kind") == expected["artifact_kind"]
                and variant.get("quantization_recipe") == expected["recipe"]
            ),
        }

    official_eval_name = "litertlm_official_qat"
    official_eval_dir = evaluation_root / official_eval_name
    official_package_report = official_eval_dir / "package_inspection.json"
    official_eval_command = [
        sys.executable,
        str(litertlm_script),
        "--model",
        str(official_artifact),
        "--runner-config",
        str(runner_config),
        "--output-dir",
        str(official_eval_dir),
        "--evaluation-name",
        official_eval_name,
        "--step",
        "<checkpoint-step>",
        *common_eval,
    ]
    if bool(mtp.get("enabled", True)):
        official_eval_command.append("--mtp-enabled")

    errors: list[dict[str, str]] = []
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    prepared_golden_contract: dict[str, Any] | None = None

    def issue(target: list[dict[str, str]], code: str, message: str) -> None:
        target.append({"code": code, "message": message})

    if not training_config.is_file():
        issue(errors, "missing_training_config", str(training_config))
    else:
        training_payload = load_yaml(training_config)
        training_model = _section(training_payload, "model")
        training_qat = _section(training_payload, "qat")
        training_golden = _section(training_payload, "golden_eval")
        training_runtime = _section(training_payload, "training")
        if not bool(
            training_model.get("model_id")
            == "google/gemma-4-E2B-it-qat-mobile-transformers"
            and training_qat.get("scale_mode") == "retained_mobile"
            and training_qat.get("fixed_scale_required") is True
            and training_qat.get("fixed_activation_scale_required") is True
            and training_qat.get("expected_effective_lora_modules") == 205
        ):
            issue(errors, "non_official_qat_training_contract", str(training_config))
        if not bool(
            training_golden.get("required_rows") == required_rows == 32
            and training_golden.get("max_rows") == required_rows
            and training_golden.get("require_exact_rows") is True
            and training_golden.get("require_unique_rows") is True
            and training_golden.get("metric_for_best_model")
            == "generation_reward_v5_4_avg"
            and training_golden.get("source_genui_sha256")
            == golden.get("source_genui_sha256")
            and training_golden.get("source_responses_sha256")
            == golden.get("source_responses_sha256")
        ):
            issue(errors, "training_golden32_contract_mismatch", str(training_config))
        if str(training_runtime.get("tensorboard_root") or "") != "tensorboard":
            issue(errors, "training_tensorboard_root_mismatch", str(training_config))
    if not golden_config.is_file():
        issue(errors, "missing_golden_config", str(golden_config))
    else:
        golden_config_run = _section(load_yaml(golden_config), "run")
        if not bool(
            golden_config_run.get("source_genui_sha256")
            == golden.get("source_genui_sha256")
            and golden_config_run.get("source_responses_sha256")
            == golden.get("source_responses_sha256")
        ):
            issue(
                errors,
                "golden_dataset_source_contract_mismatch",
                str(golden_config),
            )
    expected_source_sha = str(golden.get("source_genui_sha256") or "").lower()
    expected_responses_sha = str(
        golden.get("source_responses_sha256") or ""
    ).lower()
    expected_config_sha = str(
        golden.get("dataset_config_sha256") or ""
    ).lower()
    expected_split_sha = str(golden.get("split_sha256") or "").lower()
    if not _is_sha256(expected_config_sha):
        issue(
            errors,
            "golden_dataset_config_sha256_required",
            "Golden dataset config SHA-256 must be independently pinned.",
        )
    if not _is_sha256(expected_split_sha):
        issue(
            errors,
            "golden_prepared_split_sha256_required",
            "Prepared Golden split SHA-256 must be independently pinned.",
        )
    if not _file_ready(golden_source):
        issue(errors, "missing_golden_source", str(golden_source))
    elif _sha256(golden_source) != expected_source_sha:
        issue(errors, "golden_source_sha256_mismatch", str(golden_source))
    if not _file_ready(golden_responses):
        issue(errors, "missing_golden_responses", str(golden_responses))
    elif _sha256(golden_responses) != expected_responses_sha:
        issue(
            errors,
            "golden_responses_sha256_mismatch",
            str(golden_responses),
        )
    if not _file_ready(golden_split):
        issue(blockers, "golden_not_prepared", str(golden_split))
    else:
        try:
            prepared_golden_contract = validate_prepared_golden_contract(
                golden_split,
                dataset_config_path=golden_config,
                source_genui_path=golden_source,
                source_responses_path=golden_responses,
                expected_genui_sha256=expected_source_sha,
                expected_responses_sha256=expected_responses_sha,
                expected_dataset_config_sha256=expected_config_sha,
                expected_split_sha256=expected_split_sha,
                required_rows=required_rows,
            )
        except (OSError, TypeError, ValueError) as exc:
            issue(errors, "invalid_prepared_golden", str(exc))
    if num_gpus < 1:
        issue(errors, "invalid_num_gpus", "training.num_gpus must be positive")
    training_prerequisites: dict[str, Any] = {}
    if training_config.is_file():
        training_payload = load_yaml(training_config)
        training_model = _section(training_payload, "model")
        training_run = _section(training_payload, "run")
        model_source_value = str(training_model.get("model_source") or "").strip()
        seed_manifest_value = str(
            training_model.get("mobile_training_seed_manifest") or ""
        ).strip()
        qparams_contract_value = str(
            training_model.get("mobile_qparams_contract") or ""
        ).strip()
        dataset_dir_value = str(training_run.get("dataset_dir") or "").strip()
        model_source = _training_path(model_source_value)
        seed_manifest = _training_path(seed_manifest_value)
        qparams_contract = _training_path(qparams_contract_value)
        dataset_dir = _training_path(dataset_dir_value)
        prerequisite_paths = {
            "model_source": (model_source_value, model_source),
            "seed_manifest": (seed_manifest_value, seed_manifest),
            "qparams_contract": (qparams_contract_value, qparams_contract),
            "train_split": (dataset_dir_value, dataset_dir / "train.jsonl"),
            "validation_split": (dataset_dir_value, dataset_dir / "val.jsonl"),
        }
        for name, (configured_value, path) in prerequisite_paths.items():
            ready = bool(configured_value) and (
                path.is_dir() if name == "model_source" else _file_ready(path)
            )
            training_prerequisites[name] = {"path": str(path), "ready": ready}
            if not ready:
                issue(blockers, f"training_{name}_missing", str(path))
    if source_safetensors is not None and not _file_ready(source_safetensors):
        issue(blockers, "source_safetensors_missing", str(source_safetensors))
    if str(mtp.get("weight_source", "official")) == "trained":
        if not mtp_source_training_config.is_file():
            issue(errors, "mtp_training_config_missing", str(mtp_source_training_config))
        elif mtp_resolved_config_sha256 is None:
            issue(errors, "mtp_resolved_config_invalid", str(mtp_source_training_config))
    if base_litertlm is None:
        issue(blockers, "official_litertlm_required", "Pass --base-litertlm for official export.")
    elif not _file_ready(base_litertlm):
        issue(blockers, "official_litertlm_missing", str(base_litertlm))
    elif _sha256(base_litertlm) != str(official.get("official_artifact_sha256") or "").lower():
        issue(errors, "official_litertlm_sha256_mismatch", str(base_litertlm))
    if not _checkpoint_ready(best_checkpoint):
        issue(blockers, "best_checkpoint_not_ready", str(best_checkpoint))
    if not merged_model_dir.is_dir():
        issue(blockers, "merged_checkpoint_not_ready", str(merged_model_dir))
    if not runner_config.is_file():
        issue(blockers, "litertlm_runner_config_missing", str(runner_config))
    else:
        try:
            runner_validation = validate_external_runner_config(load_yaml(runner_config))
        except (OSError, TypeError, ValueError) as exc:
            issue(errors, "invalid_litertlm_runner_config", str(exc))
        else:
            if runner_validation["placeholder"]:
                issue(
                    blockers,
                    "litertlm_runner_is_placeholder",
                    "Pass --runner-config pointing to a real external LiteRT-LM runner.",
                )
    for script_name, script_path in (
        ("portable_launcher", portable_launcher),
        ("checkpoint_evaluator", checkpoint_script),
        ("litertlm_evaluator", litertlm_script),
        ("litertlm_package_auditor", package_audit_script),
    ):
        if not script_path.is_file():
            issue(errors, f"missing_{script_name}", str(script_path))
    for variant_id, variant in variants.items():
        if variant["enabled"] and not variant["contract_matches"]:
            issue(errors, f"invalid_{variant_id}_export_contract", json.dumps(variant))
    if bool(mtp.get("enabled", True)) and str(mtp.get("weight_source", "official")) not in {
        "official",
        "trained",
    }:
        issue(errors, "invalid_mtp_weight_source", str(mtp.get("weight_source")))
    if str(mtp.get("weight_source", "official")) == "trained":
        issue(
            warnings,
            "trained_mtp_is_not_private_google_recipe",
            "The optional teacher-forced drafter trainer is not Google's private recipe.",
        )

    return {
        "schema_version": 1,
        "pipeline_id": str(pipeline.get("id", "gemma4_e2b_a2ui_express_multiformat")),
        "run_id": run_id,
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "config_sha256": _sha256(Path(config_path).resolve())
        if config_path and Path(config_path).is_file()
        else None,
        "mode": "plan_only_unless_execute_stage_or_execute_all",
        "stage_order": list(STAGE_ORDER),
        "execute_all_stage_order": [
            stage
            for stage in STAGE_ORDER
            if stage != "mtp_training"
            or (
                mtp.get("enabled") is True
                and mtp.get("weight_source") == "trained"
                and mtp.get("train_assistant") is True
            )
        ],
        "paths": {
            "output_root": str(output_root),
            "trainer_run_root": str(trainer_run_root),
            "resolved_training_config": str(resolved_training_config),
            "best_checkpoint": str(best_checkpoint),
            "merged_model_dir": str(merged_model_dir),
            "evaluation_root": str(evaluation_root),
            "scorecard": str(output_root / str(evaluation.get("scorecard", "evaluation_scorecard.json"))),
            "tensorboard_root": str(tensorboard_root),
            "tensorboard_run_dir": str(tensorboard_root / run_id),
        },
        "golden": {
            "dataset_config": str(golden_config),
            "dataset_config_sha256": expected_config_sha,
            "source_genui": str(golden_source),
            "source_genui_sha256": expected_source_sha,
            "source_responses": str(golden_responses),
            "source_responses_sha256": expected_responses_sha,
            "split": str(golden_split),
            "split_sha256": expected_split_sha,
            "required_rows": required_rows,
            "metric_version": str(golden.get("metric_version", "dual")),
            "prepare_command": prepare_command,
            "prepared_contract": prepared_golden_contract,
        },
        "training": {
            "config": str(training_config),
            "command": training_command,
            "periodic_golden_evaluation": f"every {int(_section(load_yaml(training_config), 'golden_eval').get('interval', 1))} Trainer eval events, plus final weights" if training_config.is_file() else "configured Trainer eval cadence, plus final weights",
            "periodic_metrics_logged_to_tensorboard": True,
            "checkpoint_evaluation_command_template": checkpoint_eval_command,
            "best_checkpoint": str(best_checkpoint),
            "resolved_config": str(resolved_training_config),
            "prerequisites": training_prerequisites,
        },
        "official_qat": {
            "enabled": bool(official.get("enabled", True)),
            "official_format": True,
            "format": "retained_mobile_w2_w4_w8_a8_code_only",
            "public_recipe": False,
            "base_pipeline_config": str(_training_path(official.get("base_pipeline_config", ""))),
            "base_litertlm": str(base_litertlm) if base_litertlm else None,
            "official_artifact_sha256": str(
                official.get("official_artifact_sha256") or ""
            ).lower(),
            "artifact": str(official_artifact),
            "report": str(official_report),
            "mtp": mtp,
            "evaluation": {
                "name": official_eval_name,
                "output_dir": str(official_eval_dir),
                "command_template": official_eval_command,
            },
            "package_validation": {
                "report": str(official_package_report),
                "command_template": [
                    sys.executable,
                    str(package_audit_script),
                    str(official_artifact),
                    "--include-hashes",
                    "--output",
                    str(official_package_report),
                ],
            },
        },
        "public_variants": variants,
        "evaluation": {
            "runner_config": str(runner_config),
            "checkpoint_script": str(checkpoint_script),
            "litertlm_script": str(litertlm_script),
            "package_audit_script": str(package_audit_script),
            "score_names": [
                "checkpoint",
                *[
                    f"litertlm_{name}"
                    for name, item in variants.items()
                    if item["enabled"]
                ],
                *([official_eval_name] if bool(official.get("enabled", True)) else []),
            ],
        },
        "tensorboard": {
            "root": str(tensorboard_root),
            "run_dir": str(tensorboard_root / run_id),
            "command": ["tensorboard", "--logdir", str(tensorboard_root)],
            "mlp_override": "A2UI_TENSORBOARD_ROOT=/tensorboard",
        },
        "format_support": {
            "w32": "public FP32 LiteRT-LM; not official mobile QAT",
            "w16": "public FP16 LiteRT-LM; not official mobile QAT",
            "w8": "public dynamic W8/AFP32 LiteRT-LM; not official mobile QAT",
            "w4": "public Gemma 4 mixed W4/W8 block-32 LiteRT-LM; not pure INT4 and not official",
            "official_qat": "released retained W2/W4/W8 + A8 topology/scales; only official-format lane",
        },
        "validation": {
            "config_ok": not errors,
            "ready_for_full_execution": not errors and not blockers,
            "errors": errors,
            "blockers": blockers,
            "warnings": warnings,
        },
    }


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    if log_path.exists():
        raise Gemma4E2BMultiformatPipelineError(
            f"Refusing to overwrite an existing stage log: {log_path}"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("Command:\n" + subprocess.list2cmdline(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = int(process.wait())
    if return_code != 0:
        raise Gemma4E2BMultiformatPipelineError(
            f"Stage command failed with exit code {return_code}; see {log_path}."
        )


def _checkpoint_step(checkpoint: Path) -> int:
    metadata = checkpoint / "training_metadata.json"
    if not metadata.is_file():
        raise Gemma4E2BMultiformatPipelineError(
            f"Checkpoint provenance is missing: {metadata}"
        )
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    golden = payload.get("best_golden_eval") if isinstance(payload, dict) else None
    value = golden.get("step") if isinstance(golden, dict) else payload.get("checkpoint_step")
    if type(value) is not int or value < 0:
        raise Gemma4E2BMultiformatPipelineError(
            f"Checkpoint step is missing from {metadata}."
        )
    return value


def _validate_plan_golden(plan: dict[str, Any]) -> dict[str, Any]:
    return validate_prepared_golden_contract(
        plan["golden"]["split"],
        dataset_config_path=plan["golden"]["dataset_config"],
        source_genui_path=plan["golden"]["source_genui"],
        source_responses_path=plan["golden"]["source_responses"],
        expected_genui_sha256=plan["golden"]["source_genui_sha256"],
        expected_responses_sha256=plan["golden"]["source_responses_sha256"],
        expected_dataset_config_sha256=plan["golden"]["dataset_config_sha256"],
        expected_split_sha256=plan["golden"]["split_sha256"],
        required_rows=plan["golden"]["required_rows"],
    )


def _replace_template(command: Iterable[str], *, step: int, model: Path | None = None) -> list[str]:
    result: list[str] = []
    for value in command:
        text = str(value).replace("<checkpoint-step>", str(step))
        if text.startswith("<single-litertlm-under:"):
            if model is None:
                raise Gemma4E2BMultiformatPipelineError("LiteRT-LM artifact was not resolved.")
            text = str(model)
        result.append(text)
    return result


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        }
    if not resolved.is_dir():
        return {"path": str(resolved), "kind": "missing"}
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    entries = [
        {
            "path": item.relative_to(resolved).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in files
    ]
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(resolved),
        "kind": "directory",
        "file_count": len(entries),
        "size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "sha256": digest,
        "files": entries,
    }


def _materialize_mtp_training_config(plan: dict[str, Any]) -> None:
    mtp = plan["official_qat"]["mtp"]
    if not (
        mtp.get("enabled") is True
        and mtp.get("weight_source") == "trained"
        and mtp.get("train_assistant") is True
    ):
        return
    source = Path(str(mtp.get("source_training_config") or ""))
    destination = Path(str(mtp.get("training_config") or ""))
    if not source.is_file():
        raise Gemma4E2BMultiformatPipelineError(
            f"MTP source training config is missing: {source}"
        )
    payload = copy.deepcopy(load_yaml(source))
    payload.setdefault("run", {}).update(
        {"id": plan["run_id"], "output_dir": str(mtp["run_root"])}
    )
    payload.setdefault("target", {})["model_id_or_path"] = plan["paths"][
        "merged_model_dir"
    ]
    payload.setdefault("assistant", {})["best_checkpoint_dir"] = mtp[
        "trained_checkpoint"
    ]
    payload.setdefault("training", {}).update(
        {
            "tensorboard_root": plan["tensorboard"]["root"],
            "tensorboard_subdir": "mtp_training",
        }
    )
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != mtp.get("resolved_training_config_sha256"):
        raise Gemma4E2BMultiformatPipelineError(
            "Resolved MTP config changed after planning; rebuild the plan."
        )
    if destination.is_file():
        if _sha256(destination) != digest:
            raise Gemma4E2BMultiformatPipelineError(
                f"Refusing to replace a different resolved MTP config: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(text.encode("utf-8"))
    temporary.replace(destination)


def _resolved_official_config(plan: dict[str, Any]) -> dict[str, Any]:
    official_plan = plan["official_qat"]
    base_config = load_yaml(official_plan["base_pipeline_config"])
    underlying = base_config.setdefault("pipeline", {})
    underlying["output_dir"] = str(
        Path(plan["paths"]["output_root"]) / "official_pipeline"
    )
    underlying.setdefault("source", {})["base_litertlm"] = official_plan["base_litertlm"] or ""
    underlying["source"]["merged_model_dir"] = plan["paths"]["merged_model_dir"]
    official_output_dir = Path(official_plan["artifact"]).parent
    underlying.setdefault("mtp", {}).update(
        {
            "enabled": bool(official_plan["mtp"].get("enabled", True)),
            "weight_source": str(official_plan["mtp"].get("weight_source", "official")),
            "train_assistant": bool(official_plan["mtp"].get("train_assistant", False)),
            "training_config": str(
                official_plan["mtp"].get("training_config")
                or underlying["mtp"].get("training_config")
            ),
            "trained_checkpoint": str(
                official_plan["mtp"].get("trained_checkpoint")
                or underlying["mtp"].get("trained_checkpoint")
            ),
            "target_intermediate_litertlm": str(
                official_output_dir / "target_with_official_mtp.litertlm"
            ),
            "exact_topology_output_dir": str(
                official_output_dir / "mtp_drafter"
            ),
        }
    )
    underlying.setdefault("retained_scale_export", {}).update(
        {
            "official_artifact_sha256": official_plan[
                "official_artifact_sha256"
            ],
            "output_dir": str(Path(official_plan["artifact"]).parent),
            "report": official_plan["report"],
        }
    )
    return base_config


def _official_export_evidence(plan: dict[str, Any]) -> dict[str, Any]:
    official_plan = plan["official_qat"]
    resolved = _resolved_official_config(plan)
    expected = build_official_pipeline_plan(
        resolved,
        config_path=official_plan["base_pipeline_config"],
        training_config_override=plan["training"]["resolved_config"],
        best_checkpoint_override=plan["training"]["best_checkpoint"],
        base_litertlm_override=official_plan["base_litertlm"],
        merged_model_dir_override=plan["paths"]["merged_model_dir"],
        exact_output_dir_override=str(Path(official_plan["artifact"]).parent),
        export_report_override=official_plan["report"],
        output_litertlm_override=official_plan["artifact"],
    )
    retained_report_path = Path(official_plan["report"])
    retained = _validate_retained_scale_export_report(
        retained_report_path,
        expected_plan=expected["retained_scale_export"],
    )
    evidence: dict[str, Any] = {
        "retained_scale_report": _artifact_record(retained_report_path),
        "mode": retained.get("mode"),
        "processed_projection_count": 205,
        "passed": retained.get("passed") is True,
    }
    mtp = official_plan["mtp"]
    if (
        mtp.get("enabled") is True
        and mtp.get("weight_source") == "trained"
        and mtp.get("train_assistant") is True
    ):
        drafter_report_path = (
            Path(official_plan["artifact"]).parent
            / "mtp_drafter"
            / "mtp_checkpoint_official_topology_report.json"
        )
        if not drafter_report_path.is_file():
            raise Gemma4E2BMultiformatPipelineError(
                f"Official trained-MTP report is missing: {drafter_report_path}"
            )
        drafter = json.loads(drafter_report_path.read_text(encoding="utf-8"))
        gates = drafter.get("gates") if isinstance(drafter, dict) else {}
        boundary = (
            drafter.get("package_boundary") if isinstance(drafter, dict) else {}
        )
        artifact = Path(official_plan["artifact"]).resolve()
        if not bool(
            isinstance(gates, dict)
            and isinstance(boundary, dict)
            and drafter.get("final_artifact_gate_pass") is True
            and drafter.get("assistant_trained") is True
            and gates.get("fine_tuned_target_section_byte_exact") is True
            and gates.get("package_outside_mtp_byte_exact") is True
            and gates.get("official_graph_structure") is True
            and gates.get("official_quantization_layout") is True
            and Path(str(boundary.get("output") or "")).resolve() == artifact
            and boundary.get("output_sha256") == _sha256(artifact)
        ):
            raise Gemma4E2BMultiformatPipelineError(
                "Official trained-MTP report does not bind the final package and required gates."
            )
        evidence["trained_mtp_report"] = _artifact_record(drafter_report_path)
    return evidence


def _write_scorecard(plan: dict[str, Any]) -> dict[str, Any]:
    plan["golden"]["prepared_contract"] = _validate_plan_golden(plan)
    expected_step = _checkpoint_step(Path(plan["training"]["best_checkpoint"]))
    entries: dict[str, Any] = {}
    checkpoint_score: float | None = None
    for name in plan["evaluation"]["score_names"]:
        aggregate = Path(plan["paths"]["evaluation_root"]) / name / "aggregate_metrics.json"
        if not aggregate.is_file():
            raise Gemma4E2BMultiformatPipelineError(
                f"Final scorecard is incomplete; missing {aggregate}."
            )
        payload = json.loads(aggregate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise Gemma4E2BMultiformatPipelineError(
                f"Evaluation aggregate is not an object: {aggregate}"
            )
        primary_value = payload.get("generation_reward_v5_4_avg")
        if isinstance(primary_value, bool) or not isinstance(
            primary_value, (int, float)
        ):
            raise Gemma4E2BMultiformatPipelineError(
                "Evaluation is missing finite generation_reward_v5_4_avg: "
                f"{aggregate}"
            )
        primary_score = float(primary_value)
        if not math.isfinite(primary_score):
            raise Gemma4E2BMultiformatPipelineError(
                f"Evaluation primary score is not finite: {aggregate}"
            )
        package_precision_contract: str | None = None
        if name == "checkpoint":
            checkpoint_score = primary_score
            artifact = Path(plan["training"]["best_checkpoint"])
            package_report = None
        elif name == "litertlm_official_qat":
            artifact = Path(plan["official_qat"]["artifact"])
            package_report = Path(
                plan["official_qat"]["package_validation"]["report"]
            )
        else:
            variant_id = name.removeprefix("litertlm_")
            variant = plan["public_variants"][variant_id]
            package_precision_contract = (
                "mixed_w4_w8" if variant_id == "w4" else variant_id
            )
            artifact = _one_litertlm(Path(variant["litertlm_dir"]))
            if artifact is None:
                raise Gemma4E2BMultiformatPipelineError(
                    f"Scorecard requires exactly one {variant_id} LiteRT-LM artifact."
                )
            package_report = Path(variant["package_validation"]["report"])
        if package_report is not None and not package_report.is_file():
            raise Gemma4E2BMultiformatPipelineError(
                f"Scorecard requires the package inspection report: {package_report}"
            )
        try:
            evidence = verify_evaluation_evidence(
                aggregate_path=aggregate,
                artifact_path=artifact,
                artifact_field=(
                    "checkpoint" if name == "checkpoint" else "litertlm"
                ),
                evaluation_name=name,
                run_id=plan["run_id"],
                required_rows=int(plan["golden"]["required_rows"]),
                tensorboard_run_dir=plan["tensorboard"]["run_dir"],
                golden_split_path=plan["golden"]["split"],
                golden_split_sha256=str(
                    plan["golden"]["prepared_contract"]["split_sha256"]
                ),
                metric_version=str(plan["golden"]["metric_version"]),
                expected_step=expected_step,
                package_inspection_path=package_report,
                package_precision_contract=package_precision_contract,
            )
        except EvaluationEvidenceError as exc:
            raise Gemma4E2BMultiformatPipelineError(
                f"Evaluation evidence is invalid for {name}: {exc}"
            ) from exc
        entries[name] = {
            "aggregate": str(aggregate),
            "aggregate_sha256": _sha256(aggregate),
            "artifact": _artifact_record(artifact),
            "primary_metric": "generation_reward_v5_4_avg",
            "primary_score": primary_score,
            "delta_vs_checkpoint": None,
            "metrics": payload,
            "evidence": evidence,
        }
        if package_report is not None:
            entries[name]["package_inspection"] = _artifact_record(package_report)
        if name == "litertlm_official_qat":
            entries[name]["official_format_export"] = _official_export_evidence(plan)
    if checkpoint_score is None:
        raise Gemma4E2BMultiformatPipelineError(
            "Final scorecard requires the actual checkpoint evaluation."
        )
    for name, entry in entries.items():
        entry["delta_vs_checkpoint"] = float(entry["primary_score"]) - checkpoint_score
    scorecard = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_id": plan["pipeline_id"],
        "run_id": plan["run_id"],
        "primary_metric": "generation_reward_v5_4_avg",
        "checkpoint_primary_score": checkpoint_score,
        "golden": plan["golden"],
        "tensorboard": plan["tensorboard"],
        "evaluations": entries,
    }
    destination = Path(plan["paths"]["scorecard"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    partial.replace(destination)
    return scorecard


def run_pipeline(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    stages: Iterable[str] = (),
    execute_all: bool = False,
    run_id_override: str | None = None,
    source_safetensors_override: str | Path | None = None,
    base_litertlm_override: str | Path | None = None,
    num_gpus_override: int | None = None,
    gpu_ids_override: str | None = None,
    mtp_enabled_override: bool | None = None,
    runner_config_override: str | Path | None = None,
) -> dict[str, Any]:
    plan = build_pipeline_plan(
        config,
        config_path=config_path,
        run_id_override=run_id_override,
        source_safetensors_override=source_safetensors_override,
        base_litertlm_override=base_litertlm_override,
        num_gpus_override=num_gpus_override,
        gpu_ids_override=gpu_ids_override,
        mtp_enabled_override=mtp_enabled_override,
        runner_config_override=runner_config_override,
    )
    if execute_all:
        mtp = plan["official_qat"]["mtp"]
        selected = [
            stage
            for stage in STAGE_ORDER
            if stage != "mtp_training"
            or (
                mtp.get("enabled") is True
                and mtp.get("weight_source") == "trained"
                and mtp.get("train_assistant") is True
            )
        ]
    else:
        selected = list(stages)
    invalid = sorted(set(selected) - set(STAGE_ORDER))
    if invalid:
        raise Gemma4E2BMultiformatPipelineError(f"Unknown stages: {invalid}")
    if not selected:
        return plan
    if not plan["validation"]["config_ok"]:
        raise Gemma4E2BMultiformatPipelineError(
            "Pipeline configuration failed: "
            + ", ".join(item["code"] for item in plan["validation"]["errors"])
        )
    external_blockers = {
        item["code"] for item in plan["validation"]["blockers"]
    } & {
        "official_litertlm_required",
        "official_litertlm_missing",
        "litertlm_runner_config_missing",
        "litertlm_runner_is_placeholder",
        "source_safetensors_missing",
        "training_model_source_missing",
        "training_seed_manifest_missing",
        "training_qparams_contract_missing",
        "training_train_split_missing",
        "training_validation_split_missing",
    }
    if execute_all and external_blockers:
        raise Gemma4E2BMultiformatPipelineError(
            "Full execution requires external artifacts/configuration first: "
            + ", ".join(sorted(external_blockers))
        )
    training_blockers = {
        item["code"] for item in plan["validation"]["blockers"]
    } & {
        "source_safetensors_missing",
        "training_model_source_missing",
        "training_seed_manifest_missing",
        "training_qparams_contract_missing",
        "training_train_split_missing",
        "training_validation_split_missing",
    }
    if "training" in selected and training_blockers:
        raise Gemma4E2BMultiformatPipelineError(
            "Training prerequisites are missing: "
            + ", ".join(sorted(training_blockers))
        )

    output_root = Path(plan["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    state_path = output_root / "pipeline_execution.json"
    execution: dict[str, Any] = {
        "schema_version": 1,
        "run_id": plan["run_id"],
        "selected_stages": selected,
        "completed_stages": [],
    }

    def completed(stage: str) -> None:
        execution["completed_stages"].append(stage)
        state_path.write_text(json.dumps(execution, indent=2), encoding="utf-8")

    for stage in selected:
        if stage == "prepare_golden":
            if not _file_ready(Path(plan["golden"]["split"])):
                _run_command(plan["golden"]["prepare_command"], cwd=repo_root(), log_path=logs / "prepare_golden.log")
            plan["golden"]["prepared_contract"] = _validate_plan_golden(plan)
        elif stage == "training":
            _run_command(plan["training"]["command"], cwd=repo_root(), log_path=logs / "training.log")
            if not _checkpoint_ready(Path(plan["training"]["best_checkpoint"])):
                raise Gemma4E2BMultiformatPipelineError("Training produced no best Golden checkpoint.")
        elif stage == "checkpoint_evaluation":
            plan["golden"]["prepared_contract"] = _validate_plan_golden(plan)
            checkpoint = Path(plan["training"]["best_checkpoint"])
            command = _replace_template(
                plan["training"]["checkpoint_evaluation_command_template"],
                step=_checkpoint_step(checkpoint),
            )
            _run_command(command, cwd=repo_root(), log_path=logs / "checkpoint_evaluation.log")
        elif stage in {"merge", "mtp_training", "official_export"}:
            if plan["official_qat"]["mtp"].get("weight_source") == "trained":
                _materialize_mtp_training_config(plan)
            if stage == "mtp_training" and not (
                plan["official_qat"]["mtp"].get("enabled") is True
                and plan["official_qat"]["mtp"].get("weight_source") == "trained"
                and plan["official_qat"]["mtp"].get("train_assistant") is True
            ):
                raise Gemma4E2BMultiformatPipelineError(
                    "MTP training requires enabled=true, weight_source=trained, "
                    "and train_assistant=true."
                )
            official_config = _resolved_official_config(plan)
            kwargs = {
                "config_path": plan["official_qat"]["base_pipeline_config"],
                "training_config_override": plan["training"]["resolved_config"],
                "best_checkpoint_override": plan["training"]["best_checkpoint"],
                "base_litertlm_override": plan["official_qat"]["base_litertlm"],
                "merged_model_dir_override": plan["paths"]["merged_model_dir"],
                "exact_output_dir_override": str(Path(plan["official_qat"]["artifact"]).parent),
                "export_report_override": plan["official_qat"]["report"],
                "output_litertlm_override": plan["official_qat"]["artifact"],
            }
            if stage == "merge":
                run_official_pipeline(official_config, execute_merge=True, **kwargs)
            elif stage == "mtp_training":
                run_official_pipeline(official_config, execute_drafter_training=True, **kwargs)
            else:
                run_official_pipeline(official_config, execute_retained_scale_export=True, **kwargs)
        elif stage == "public_exports":
            for variant_id, variant in plan["public_variants"].items():
                if not variant["enabled"]:
                    continue
                output_dir = Path(variant["output_dir"])
                if output_dir.exists() and any(output_dir.iterdir()):
                    raise Gemma4E2BMultiformatPipelineError(
                        f"Refusing to overwrite public export: {output_dir}"
                    )
                export_edge_gallery_model(variant["resolved_export_config"], dry_run=False)
                if _one_litertlm(Path(variant["litertlm_dir"])) is None:
                    raise Gemma4E2BMultiformatPipelineError(
                        f"{variant_id} export did not produce exactly one LiteRT-LM artifact."
                    )
        elif stage == "litertlm_evaluations":
            plan["golden"]["prepared_contract"] = _validate_plan_golden(plan)
            runner_blockers = {
                item["code"] for item in plan["validation"]["blockers"]
            } & {
                "litertlm_runner_config_missing",
                "litertlm_runner_is_placeholder",
            }
            if runner_blockers:
                raise Gemma4E2BMultiformatPipelineError(
                    "LiteRT-LM evaluation runner is not ready: "
                    + ", ".join(sorted(runner_blockers))
                )
            step = _checkpoint_step(Path(plan["training"]["best_checkpoint"]))
            evaluation_items: list[
                tuple[str, dict[str, Any], dict[str, Any], Path]
            ] = []
            for variant_id, variant in plan["public_variants"].items():
                if variant["enabled"]:
                    artifact = _one_litertlm(Path(variant["litertlm_dir"]))
                    if artifact is None:
                        raise Gemma4E2BMultiformatPipelineError(
                            f"Cannot evaluate {variant_id}: expected exactly one artifact."
                        )
                    evaluation_items.append(
                        (
                            variant_id,
                            variant["evaluation"],
                            variant["package_validation"],
                            artifact,
                        )
                    )
            if plan["official_qat"]["enabled"]:
                evaluation_items.append(
                    (
                        "official_qat",
                        plan["official_qat"]["evaluation"],
                        plan["official_qat"]["package_validation"],
                        Path(plan["official_qat"]["artifact"]),
                    )
                )
            for name, evaluation_plan, package_plan, artifact in evaluation_items:
                if not _file_ready(artifact):
                    raise Gemma4E2BMultiformatPipelineError(f"Missing evaluation artifact: {artifact}")
                audit_command = _replace_template(
                    package_plan["command_template"], step=step, model=artifact
                )
                _run_command(
                    audit_command,
                    cwd=repo_root(),
                    log_path=logs / f"inspect_{name}.log",
                )
                if not _file_ready(Path(package_plan["report"])):
                    raise Gemma4E2BMultiformatPipelineError(
                        f"Package audit wrote no inspection report for {name}."
                    )
                if name != "official_qat":
                    try:
                        package_payload = json.loads(
                            Path(package_plan["report"]).read_text(encoding="utf-8")
                        )
                        if not isinstance(package_payload, dict):
                            raise TypeError("inspection report is not a JSON object")
                        package_plan["precision_contract"] = (
                            validate_litertlm_precision_contract(
                                package_payload,
                                expected_format=(
                                    "mixed_w4_w8" if name == "w4" else name
                                ),
                            )
                        )
                    except (
                        OSError,
                        TypeError,
                        json.JSONDecodeError,
                        EvaluationEvidenceError,
                    ) as exc:
                        raise Gemma4E2BMultiformatPipelineError(
                            f"{name} package precision validation failed: {exc}"
                        ) from exc
                command = _replace_template(
                    evaluation_plan["command_template"], step=step, model=artifact
                )
                _run_command(command, cwd=repo_root(), log_path=logs / f"evaluate_{name}.log")
        elif stage == "scorecard":
            plan["scorecard"] = _write_scorecard(plan)
        completed(stage)
    plan["execution"] = execution
    return plan
