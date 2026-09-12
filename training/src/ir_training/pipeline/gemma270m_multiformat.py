"""Dry-run-first Gemma 3 270M A2UI Express multi-format pipeline.

The pipeline trains one W8-QAT LoRA adapter, selects the best Golden-32
checkpoint, merges it once, and exports four comparable LiteRT-LM variants.
It deliberately distinguishes QAT alignment from post-training conversion:
only W8 matches the training fake-quantization objective. W32 and W16 are
floating baselines, while W4 is an explicitly experimental sensitivity run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from ir_training.common.config import load_yaml, repo_root, resolve_path, training_root
from ir_training.eval.external_runner import validate_external_runner_config
from ir_training.eval.evaluation_evidence import (
    EvaluationEvidenceError,
    validate_litertlm_precision_contract,
    verify_evaluation_evidence,
)
from ir_training.eval.golden_set import validate_prepared_golden_contract
from ir_training.eval.tensorboard_logging import resolve_tensorboard_root
from ir_training.export.edge_gallery import (
    build_litert_export_command,
    export_edge_gallery_model,
)
from ir_training.export.merge_lora import merge_lora_adapter
from ir_training.train.gpu_profile import apply_gpu_profile, build_gpu_profile, detect_cuda_devices, training_environment, verify_gpu_profile

FORMAT_ORDER = ("w32", "w16", "w8", "w4")
FORMAT_CONTRACT = {
    "w32": {"bits": 32, "recipe": "none", "experimental": False},
    "w16": {"bits": 16, "recipe": "none", "experimental": True},
    "w8": {"bits": 8, "recipe": "dynamic_wi8_afp32", "experimental": False},
    "w4": {"bits": 4, "recipe": "dynamic_wi4b32_afp32", "experimental": True},
}

_ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
_ADAPTER_IDENTITY_NAMES = {
    *_ADAPTER_WEIGHT_NAMES,
    "adapter_config.json",
    "training_metadata.json",
    "training_config.yaml",
    "best_metric_info.json",
}


class Gemma270MMultiformatError(RuntimeError):
    """Raised when a stage cannot proceed without weakening the contract."""


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _pipeline_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    return _section(_section(config, "pipeline"), name)


def _resolve_training_path(value: Any) -> Path:
    return resolve_path(str(value), training_root())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _checkpoint_ready(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and any((path / marker).is_file() for marker in _ADAPTER_WEIGHT_NAMES)
    )


def _artifact_identity(path: Path, *, require_adapter: bool = False) -> dict[str, Any]:
    """Return a deterministic file or directory identity for a scorecard.

    Adapter checkpoints are directories, so a simple ``Path.is_file`` test
    both mislabels them as absent and fails to bind a score to the bytes that
    were evaluated. Directory identities hash a sorted manifest of every
    regular file while surfacing the adapter/provenance files separately.
    """

    if path.is_file():
        return {
            "artifact_exists": True,
            "artifact_type": "file",
            "artifact_size_bytes": path.stat().st_size,
            "artifact_sha256": _sha256(path),
            "artifact_hash_semantics": "file_sha256",
            "artifact_identity_complete": not require_adapter,
        }
    if not path.is_dir():
        return {
            "artifact_exists": False,
            "artifact_type": "missing",
            "artifact_identity_complete": False,
        }

    records: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        records.append(
            {
                "path": candidate.relative_to(path).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256(candidate),
            }
        )
    manifest_bytes = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    identity_records = [
        item
        for item in records
        if Path(str(item["path"])).name in _ADAPTER_IDENTITY_NAMES
    ]
    identity_names = {Path(str(item["path"])).name for item in identity_records}
    required_adapter_files = {
        "adapter_weights": any(
            name in identity_names for name in _ADAPTER_WEIGHT_NAMES
        ),
        "adapter_config.json": "adapter_config.json" in identity_names,
        "training_metadata.json": "training_metadata.json" in identity_names,
    }
    adapter_identity_complete = all(required_adapter_files.values())
    return {
        "artifact_exists": True,
        "artifact_type": "directory",
        "artifact_file_count": len(records),
        "artifact_size_bytes": sum(int(item["size_bytes"]) for item in records),
        "artifact_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "artifact_hash_semantics": "sha256_of_sorted_path_size_sha256_manifest",
        "artifact_key_files": identity_records,
        "artifact_required_files_present": required_adapter_files,
        "artifact_identity_complete": (
            adapter_identity_complete if require_adapter else bool(records)
        ),
    }


def _jsonl_identity_report(path: Path) -> dict[str, Any]:
    identities: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row {index} is not an object: {path}")
            identity = next(
                (
                    str(row.get(key)).strip()
                    for key in ("id", "response_id", "scenario_id", "source_id")
                    if row.get(key) is not None and str(row.get(key)).strip()
                ),
                f"row:{index}",
            )
            identities.append(identity)
    return {
        "path": str(path),
        "rows": len(identities),
        "unique_rows": len(set(identities)),
        "sha256": _sha256(path),
    }


def _row_leakage_signatures(row: dict[str, Any]) -> set[str]:
    """Return content identities, deliberately excluding reused run-local IDs.

    The source datasets reuse identifiers such as ``q_000001`` across runs, so
    matching raw query/response/UI IDs would reject unrelated examples. The
    actual held-out boundary is the normalized Stage-2 response content seen by
    the model.
    """

    response_text = str(row.get("response_text") or "")
    if not response_text:
        for message in reversed(list(row.get("messages") or [])):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            response_text = str(message.get("content") or "")
            marker = "Create A2UI Express v1 GenUI IR for this response:\n\n"
            if marker in response_text:
                response_text = response_text.split(marker, 1)[1]
            break
    normalized = " ".join(unicodedata.normalize("NFKC", response_text).split())
    if not normalized:
        return set()
    return {
        "response_text_sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    }


def _jsonl_leakage_signatures(path: Path) -> tuple[set[str], int]:
    signatures: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row {index} is not an object: {path}")
            rows += 1
            signatures.update(_row_leakage_signatures(row))
    return signatures, rows


def _leakage_split_report(path: Path, golden_signatures: set[str]) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "checked": False,
            "row_count": None,
            "overlap_count": None,
            "overlap_sample": [],
        }
    signatures, rows = _jsonl_leakage_signatures(path)
    overlap = sorted(signatures & golden_signatures)
    return {
        "path": str(path),
        "exists": True,
        "checked": True,
        "row_count": rows,
        "response_signature_count": len(signatures),
        "overlap_count": len(overlap),
        "overlap_sample": overlap[:20],
    }


def _issue(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def _best_checkpoint(training_config: dict[str, Any]) -> Path:
    golden = _section(training_config, "golden_eval")
    if golden.get("best_checkpoint_dir"):
        return _resolve_training_path(golden["best_checkpoint_dir"])
    output_dir = _resolve_training_path(
        _section(training_config, "run").get(
            "output_dir", "runs/gemma3_270m_a2ui_express_multiformat"
        )
    )
    return output_dir / "best_golden_checkpoint"


def _format_selection(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return FORMAT_ORDER
    selected: list[str] = []
    for value in values:
        for item in str(value).split(","):
            name = item.strip().lower()
            if not name:
                continue
            if name == "all":
                return FORMAT_ORDER
            if name not in FORMAT_ORDER:
                raise Gemma270MMultiformatError(
                    f"Unknown format {name!r}; choose {', '.join(FORMAT_ORDER)} or all."
                )
            if name not in selected:
                selected.append(name)
    return tuple(selected)


def _evaluation_command(
    *,
    script: Path,
    model_args: list[str],
    split: Path,
    output_dir: Path,
    golden: dict[str, Any],
    tensorboard_root: str,
    run_id: str,
    evaluation_name: str,
    step: int | str,
) -> list[str]:
    return [
        sys.executable,
        str(script),
        *model_args,
        "--split",
        str(split),
        "--output-dir",
        str(output_dir),
        "--max-rows",
        str(int(golden.get("required_rows", 32))),
        "--required-rows",
        str(int(golden.get("required_rows", 32))),
        "--max-input-tokens",
        str(int(golden.get("max_input_tokens", 3072))),
        "--max-new-tokens",
        str(int(golden.get("max_new_tokens", 1024))),
        "--metric-version",
        str(golden.get("metric_version", "dual")),
        "--weights-config",
        str(
            _resolve_training_path(
                golden.get("weights_config", "../dataset/configs/run.yaml")
            )
        ),
        "--tensorboard-root",
        tensorboard_root,
        "--run-id",
        run_id,
        "--evaluation-name",
        evaluation_name,
        "--step",
        str(step),
    ]


def build_pipeline_plan(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    run_id_override: str | None = None,
    formats: Iterable[str] | None = None,
    best_checkpoint_override: str | Path | None = None,
    merged_model_override: str | Path | None = None,
    runner_config_override: str | Path | None = None,
    official_q8_source_override: str | Path | None = None,
) -> dict[str, Any]:
    """Build a complete plan without loading a model or writing artifacts."""

    pipeline = _section(config, "pipeline")
    pipeline_id = str(
        pipeline.get("id") or "gemma3_270m_a2ui_express_multiformat"
    )
    run_id = str(
        run_id_override
        or pipeline.get("run_id")
        or f"{pipeline_id}_001"
    ).strip()
    if not re.fullmatch(
        r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9-])?", run_id
    ) or re.fullmatch(
        r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", run_id, re.IGNORECASE
    ):
        raise Gemma270MMultiformatError(
            "pipeline.run_id/--run-id must be one safe path component using "
            "letters, digits, dots, underscores, and hyphens."
        )
    output_base = _resolve_training_path(
        pipeline.get(
            "output_dir", "outputs/pipelines/gemma3_270m_a2ui_express_multiformat"
        )
    )
    output_dir = output_base / run_id
    training = _pipeline_section(config, "training")
    training_config_path = _resolve_training_path(
        training.get(
            "config",
            "configs/models/gemma3_270m_a2ui_express_qat.yaml",
        )
    )
    if not training_config_path.is_file():
        raise Gemma270MMultiformatError(
            f"Training config does not exist: {training_config_path}"
        )
    training_config = load_yaml(training_config_path)
    model = _section(training_config, "model")
    training_job = _section(training_config, "training")
    qat = _section(training_config, "qat")
    training_run = _section(training_config, "run")
    training_output_base = _resolve_training_path(
        training_run.get("output_dir", "runs/gemma3_270m_a2ui_express_multiformat")
    )
    training_output = training_output_base / run_id
    training_dataset_dir = _resolve_training_path(training_run.get("dataset_dir", ""))
    best_checkpoint = (
        _resolve_training_path(best_checkpoint_override)
        if best_checkpoint_override
        else training_output / "best_golden_checkpoint"
    )
    resolved_training_config_path = (
        training_output / "launch" / "resolved_training_config.yaml"
    )
    host_gpu_profile = training.get("host_gpu_profile")
    if host_gpu_profile is None and resolved_training_config_path.is_file():
        # Later export/evaluation commands reuse the original bound execution
        # settings without detecting GPUs or changing the training identity.
        host_gpu_profile = _section(load_yaml(resolved_training_config_path), "runtime").get("gpu_profile")
    merged = _pipeline_section(config, "merge")
    merged_model = (
        _resolve_training_path(merged_model_override)
        if merged_model_override
        else output_dir / Path(str(merged.get("output_dir") or "merged_best_hf")).name
    )

    golden = _pipeline_section(config, "golden")
    golden_dataset_config = _resolve_training_path(
        golden.get("dataset_config", "configs/datasets/golden32_20260903_eval.yaml")
    )
    golden_source = _resolve_training_path(golden.get("source_jsonl", ""))
    golden_responses = _resolve_training_path(golden.get("responses_jsonl", ""))
    golden_split = _resolve_training_path(
        golden.get("split_path", "outputs/datasets/golden32_20260903_eval/all.jsonl")
    )
    tensorboard_value = str(resolve_tensorboard_root("/tensorboard")) if host_gpu_profile else str(training.get("tensorboard_root") or "tensorboard")
    tensorboard_root = resolve_tensorboard_root(tensorboard_value)
    tensorboard_subdir = str(training_job.get("tensorboard_subdir") or "training")
    expected_training_tb = tensorboard_root / run_id / tensorboard_subdir
    resolved_training_config = copy.deepcopy(training_config)
    resolved_run = resolved_training_config.setdefault("run", {})
    resolved_training = resolved_training_config.setdefault("training", {})
    resolved_golden = resolved_training_config.setdefault("golden_eval", {})
    if not all(
        isinstance(value, dict)
        for value in (resolved_run, resolved_training, resolved_golden)
    ):
        raise Gemma270MMultiformatError(
            "Training config run/training/golden_eval sections must be YAML objects."
        )
    resolved_run["id"] = run_id
    resolved_run["output_dir"] = str(training_output)
    resolved_training["logging_dir"] = str(expected_training_tb)
    if host_gpu_profile is not None:
        apply_gpu_profile(resolved_training_config, host_gpu_profile)
        resolved_training["tensorboard_root"] = str(tensorboard_root)
    resolved_golden["best_checkpoint_dir"] = str(best_checkpoint)
    resolved_golden["output_dir"] = str(training_output / "golden32")
    resolved_training_config_text = yaml.safe_dump(
        resolved_training_config,
        sort_keys=False,
        allow_unicode=True,
    )
    resolved_training_config_sha256 = hashlib.sha256(
        resolved_training_config_text.encode("utf-8")
    ).hexdigest()
    training_config = resolved_training_config
    training_job = _section(training_config, "training")
    selected = _format_selection(formats)
    formats_config = _pipeline_section(config, "formats")
    export_common = _pipeline_section(config, "export")
    evaluation = _pipeline_section(config, "evaluation")
    checkpoint_eval_dir = output_dir / "evaluation" / "checkpoint"
    merged_eval_dir = output_dir / "evaluation" / "merged"
    litert_eval_root = output_dir / "evaluation" / "litertlm"
    checkpoint_eval_script = (
        training_root() / "scripts" / "evaluate_checkpoint_on_golden.py"
    )
    litert_eval_script = training_root() / "scripts" / "evaluate_litertlm_on_golden.py"
    litert_runner_config = (
        _resolve_training_path(runner_config_override)
        if runner_config_override
        else _resolve_training_path(
            evaluation.get(
                "litertlm_runner_config",
                "configs/eval/litertlm_external_runner.example.yaml",
            )
        )
    )

    issues: list[dict[str, str]] = []
    model_id = str(model.get("model_id") or "")
    if model_id != "google/gemma-3-270m-it":
        issues.append(
            _issue(
                "error",
                "unexpected_model",
                f"Expected google/gemma-3-270m-it, got {model_id or '<missing>'}.",
            )
        )
    if not (
        bool(qat.get("enabled", False))
        and int(qat.get("weight_bits", 0)) == 8
        and int(qat.get("activation_bits", 0)) == 32
        and str(qat.get("quantizer") or "") == "ste_ai_edge"
        and bool(qat.get("effective_merged_weight", False))
        and bool(qat.get("quantize_embeddings", False))
    ):
        issues.append(
            _issue(
                "error",
                "unexpected_qat_contract",
                "Gemma 270M multi-format training must retain the canonical W8/AFP32 effective-weight QAT profile.",
            )
        )
    if bool(_pipeline_section(config, "mtp").get("enabled", False)):
        issues.append(
            _issue(
                "error",
                "mtp_not_applicable",
                "Gemma 3 270M has no Gemma 4 MTP drafter; mtp.enabled must remain false.",
            )
        )

    required_rows = int(golden.get("required_rows", 32))
    expected_sha = str(golden.get("source_sha256") or "").strip().lower()
    expected_responses_sha = (
        str(golden.get("responses_sha256") or "").strip().lower()
    )
    expected_config_sha = (
        str(golden.get("dataset_config_sha256") or "").strip().lower()
    )
    expected_split_sha = str(golden.get("split_sha256") or "").strip().lower()
    if not _is_sha256(expected_config_sha):
        issues.append(
            _issue(
                "error",
                "golden_dataset_config_sha256_required",
                "Golden dataset config SHA-256 must be independently pinned.",
            )
        )
    if not _is_sha256(expected_split_sha):
        issues.append(
            _issue(
                "error",
                "golden_prepared_split_sha256_required",
                "Prepared Golden split SHA-256 must be independently pinned.",
            )
        )
    if not golden_dataset_config.is_file():
        issues.append(
            _issue(
                "error",
                "missing_golden_config",
                f"Missing Golden config: {golden_dataset_config}",
            )
        )
    else:
        try:
            golden_dataset_payload = load_yaml(golden_dataset_config)
            golden_dataset_run = _section(golden_dataset_payload, "run")
            configured_source_dir = _resolve_training_path(
                golden_dataset_run.get("source_run_dir", "")
            )
            configured_output_dir = _resolve_training_path(
                golden_dataset_run.get("output_dir", "")
            )
            if not bool(
                golden_dataset_run.get("source_genui_sha256") == expected_sha
                and golden_dataset_run.get("source_responses_sha256")
                == expected_responses_sha
                and (configured_source_dir / "genui.jsonl").resolve()
                == golden_source.resolve()
                and (configured_source_dir / "responses.jsonl").resolve()
                == golden_responses.resolve()
                and (configured_output_dir / golden_split.name).resolve()
                == golden_split.resolve()
            ):
                issues.append(
                    _issue(
                        "error",
                        "golden_dataset_config_mismatch",
                        "Golden dataset config paths and source hashes must match the pipeline pins.",
                    )
                )
        except (OSError, TypeError, ValueError) as exc:
            issues.append(
                _issue("error", "golden_dataset_config_unreadable", str(exc))
            )
    source_report = None
    if not golden_source.is_file():
        issues.append(
            _issue(
                "error",
                "missing_golden_source",
                f"Missing supplied Golden JSONL: {golden_source}",
            )
        )
    else:
        try:
            source_report = _jsonl_identity_report(golden_source)
            if not _is_sha256(expected_sha):
                issues.append(
                    _issue(
                        "error",
                        "golden_source_sha256_required",
                        "Golden source SHA-256 must be explicitly pinned.",
                    )
                )
            if source_report["rows"] != required_rows:
                issues.append(
                    _issue(
                        "error",
                        "golden_source_row_count",
                        f"Golden source has {source_report['rows']} rows; expected exactly {required_rows}.",
                    )
                )
            if source_report["unique_rows"] != required_rows:
                issues.append(
                    _issue(
                        "error",
                        "golden_source_not_unique",
                        "Golden source identities are not exactly 32 unique rows.",
                    )
                )
            if source_report["sha256"] != expected_sha:
                issues.append(
                    _issue(
                        "error",
                        "golden_source_sha256_mismatch",
                        "Golden source SHA-256 differs from the pipeline pin.",
                    )
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(_issue("error", "golden_source_unreadable", str(exc)))

    responses_report = None
    if not golden_responses.is_file():
        issues.append(
            _issue(
                "error",
                "missing_golden_responses",
                f"Missing supplied Golden responses JSONL: {golden_responses}",
            )
        )
    else:
        responses_report = {
            "path": str(golden_responses),
            "sha256": _sha256(golden_responses),
        }
        if (
            not _is_sha256(expected_responses_sha)
            or responses_report["sha256"] != expected_responses_sha
        ):
            issues.append(
                _issue(
                    "error",
                    "golden_responses_sha256_mismatch",
                    "Golden responses SHA-256 differs from the pipeline pin.",
                )
            )

    signature_source = golden_split if golden_split.is_file() else golden_source
    golden_signatures: set[str] = set()
    if signature_source.is_file():
        try:
            golden_signatures, _ = _jsonl_leakage_signatures(signature_source)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                _issue("error", "golden_signature_source_unreadable", str(exc))
            )

    leakage_splits = {
        split_name: _leakage_split_report(
            training_dataset_dir / f"{split_name}.jsonl", golden_signatures
        )
        for split_name in ("train", "val")
    }
    leakage_checked = bool(golden_signatures) and all(
        bool(item["checked"]) for item in leakage_splits.values()
    )
    leakage_overlap = sum(
        int(item["overlap_count"] or 0) for item in leakage_splits.values()
    )
    leakage_report = {
        "checked": leakage_checked,
        "ok": leakage_checked and leakage_overlap == 0,
        "signature_source": str(signature_source),
        "golden_response_signature_count": len(golden_signatures),
        "overlap_count": leakage_overlap,
        "splits": leakage_splits,
    }
    if not leakage_checked:
        issues.append(
            _issue(
                "warning",
                "golden_training_leakage_not_yet_checkable",
                "Prepared train.jsonl and val.jsonl are both required before the Golden-32 leakage gate can run.",
            )
        )
    elif leakage_overlap:
        issues.append(
            _issue(
                "error",
                "golden_training_content_overlap",
                f"Training/validation data overlaps Golden-32 in {leakage_overlap} normalized response signature(s).",
            )
        )

    split_report = None
    prepared_contract_report = None
    if golden_split.is_file():
        try:
            split_report = _jsonl_identity_report(golden_split)
            if (
                split_report["rows"] != required_rows
                or split_report["unique_rows"] != required_rows
            ):
                issues.append(
                    _issue(
                        "error",
                        "prepared_golden_contract_mismatch",
                        f"Prepared Golden split must contain exactly {required_rows} unique rows.",
                    )
                )
            prepared_contract_report = validate_prepared_golden_contract(
                golden_split,
                dataset_config_path=golden_dataset_config,
                source_genui_path=golden_source,
                source_responses_path=golden_responses,
                expected_genui_sha256=expected_sha,
                expected_responses_sha256=expected_responses_sha,
                expected_dataset_config_sha256=expected_config_sha,
                expected_split_sha256=expected_split_sha,
                required_rows=required_rows,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(_issue("error", "prepared_golden_unreadable", str(exc)))
    else:
        issues.append(
            _issue(
                "warning",
                "prepared_golden_missing",
                "Prepare the immutable Golden-32 split before training or final evaluation.",
            )
        )

    train_golden = _section(training_config, "golden_eval")
    if not (
        bool(train_golden.get("enabled", False))
        and int(train_golden.get("required_rows", 0)) == required_rows
        and bool(train_golden.get("require_exact_rows", False))
        and bool(train_golden.get("require_unique_rows", False))
        and _resolve_training_path(train_golden.get("dataset_dir", ""))
        == golden_split.parent
    ):
        issues.append(
            _issue(
                "error",
                "training_golden_contract_mismatch",
                "Training config must use the exact immutable Golden-32 only for evaluation.",
            )
        )
    configured_training_tb = _resolve_training_path(training_job.get("logging_dir", ""))
    tensorboard_subdir = str(training_job.get("tensorboard_subdir") or "training")
    expected_training_tb = tensorboard_root / run_id / tensorboard_subdir
    if (
        training_job.get("tensorboard_root") is None
        and configured_training_tb != expected_training_tb
    ):
        issues.append(
            _issue(
                "error",
                "training_tensorboard_outside_root",
                f"Training logging_dir must resolve to {expected_training_tb}, got {configured_training_tb}.",
            )
        )
    if str(training_job.get("tensorboard_root") or "") != tensorboard_value:
        issues.append(
            _issue(
                "error",
                "tensorboard_root_mismatch",
                "Model training.tensorboard_root must match pipeline.training.tensorboard_root.",
            )
        )
    if tensorboard_subdir != "training":
        issues.append(
            _issue(
                "error",
                "tensorboard_subdir_mismatch",
                "Model training.tensorboard_subdir must remain 'training'.",
            )
        )
    if str(training_job.get("report_to") or "").lower() != "tensorboard":
        issues.append(
            _issue(
                "error",
                "tensorboard_disabled",
                "training.report_to must be tensorboard.",
            )
        )
    for path, code in (
        (checkpoint_eval_script, "missing_checkpoint_evaluator"),
        (litert_eval_script, "missing_litertlm_evaluator"),
    ):
        if not path.is_file():
            issues.append(
                _issue("error", code, f"Required evaluation file is missing: {path}")
            )
    runner_report: dict[str, Any] = {
        "path": str(litert_runner_config),
        "protocol": None,
        "placeholder": False,
        "ready": False,
    }
    if not litert_runner_config.is_file():
        issues.append(
            _issue(
                "blocker",
                "missing_litertlm_runner_config",
                f"LiteRT-LM runner config is missing: {litert_runner_config}",
            )
        )
    else:
        try:
            runner_payload = load_yaml(litert_runner_config)
            runner_validation = validate_external_runner_config(runner_payload)
            runner_report.update(runner_validation)
            if runner_validation["placeholder"]:
                issues.append(
                    _issue(
                        "blocker",
                        "litertlm_runner_is_placeholder",
                        "Pass --runner-config pointing to a real LiteRT-LM runner before evaluation.",
                    )
                )
            else:
                runner_report["ready"] = True
        except (OSError, TypeError, ValueError) as exc:
            issues.append(
                _issue("error", "invalid_litertlm_runner_config", str(exc))
            )

    format_plans: dict[str, Any] = {}
    for name in FORMAT_ORDER:
        contract = FORMAT_CONTRACT[name]
        item = formats_config.get(name)
        item = item if isinstance(item, dict) else {}
        if not item:
            issues.append(
                _issue(
                    "error", f"missing_format_{name}", f"formats.{name} is required."
                )
            )
        enabled = bool(item.get("enabled", False))
        recipe = str(item.get("quantization_recipe") or "")
        experimental = bool(item.get("experimental", False))
        extra_flags = [
            str(value) for value in (export_common.get("common_extra_flags") or [])
        ]
        extra_flags.extend(str(value) for value in (item.get("extra_flags") or []))
        export_dir = output_dir / "export" / name
        configured_artifact = Path(
            str(item.get("artifact") or f"gemma3_270m_{name}.litertlm")
        )
        artifact = output_dir / "litertlm" / configured_artifact.name
        litert_output_dir = export_dir / "litertlm"
        if int(item.get("weight_bits", -1)) != int(contract["bits"]):
            issues.append(
                _issue(
                    "error",
                    f"{name}_bits_mismatch",
                    f"formats.{name}.weight_bits is invalid.",
                )
            )
        if recipe != contract["recipe"]:
            issues.append(
                _issue(
                    "error",
                    f"{name}_recipe_mismatch",
                    f"formats.{name}.quantization_recipe must be {contract['recipe']!r}.",
                )
            )
        if experimental != bool(contract["experimental"]):
            issues.append(
                _issue(
                    "error",
                    f"{name}_experimental_mismatch",
                    f"formats.{name}.experimental is invalid.",
                )
            )
        if name == "w16" and "--experimental_use_fp16=True" not in extra_flags:
            issues.append(
                _issue(
                    "error",
                    "w16_missing_exporter_flag",
                    "The public FP16 path requires --experimental_use_fp16=True.",
                )
            )
        format_eval_dir = litert_eval_root / name
        format_plans[name] = {
            "enabled": enabled,
            "selected": name in selected,
            "label": str(item.get("label") or name),
            "weight_bits": int(contract["bits"]),
            "quantization_recipe": recipe,
            "experimental": experimental,
            "requires_explicit_opt_in": bool(
                item.get("require_explicit_opt_in", False)
            ),
            "qat_aligned": bool(item.get("qat_aligned", name == "w8")),
            "export_output_dir": str(export_dir),
            "litert_output_dir": str(litert_output_dir),
            "artifact": str(artifact),
            "artifact_source": "public_export",
            "artifact_ready": artifact.is_file(),
            "export_config": {
                "run": {"id": f"{run_id}_{name}", "output_dir": str(export_dir)},
                "source": {
                    "base_model_id": model_id,
                    "merged_model_dir": str(merged_model),
                },
                "export": {
                    "runtime": "litertlm",
                    "format": "litertlm",
                    "litert_output_dir": str(litert_output_dir),
                    "command": str(export_common.get("command") or "litert-torch"),
                    "task": str(export_common.get("task") or "text_generation"),
                    "quantization_recipe": recipe,
                    "externalize_embedder": bool(
                        export_common.get("externalize_embedder", False)
                    ),
                    "max_input_tokens": int(
                        export_common.get("max_input_tokens", 3072)
                    ),
                    "max_output_tokens": int(
                        export_common.get("max_output_tokens", 1024)
                    ),
                    "extra_flags": extra_flags,
                },
                "android": {
                    "package_name": f"gemma3_270m_a2ui_express_{name}",
                    "display_name": f"Gemma 3 270M A2UI Express {name.upper()}",
                    "stage": "stage3_a2ui_express",
                },
            },
            "evaluation": {
                "output_dir": str(format_eval_dir),
                "aggregate": str(format_eval_dir / "aggregate_metrics.json"),
                "evaluation_name": f"litertlm_{name}",
                "command": _evaluation_command(
                    script=litert_eval_script,
                    model_args=[
                        "--model",
                        str(artifact),
                        "--runner-config",
                        str(litert_runner_config),
                    ],
                    split=golden_split,
                    output_dir=format_eval_dir,
                    golden=golden,
                    tensorboard_root=tensorboard_value,
                    run_id=run_id,
                    evaluation_name=f"litertlm_{name}",
                    step="<checkpoint-step>",
                ),
            },
            "package_validation": {
                "artifact": str(artifact),
                "report": str(format_eval_dir / "package_inspection.json"),
                "command": [
                    sys.executable,
                    str(training_root() / "scripts" / "audit_litertlm_package.py"),
                    str(artifact),
                    "--include-hashes",
                    "--output",
                    str(format_eval_dir / "package_inspection.json"),
                ],
            },
        }
        format_plans[name]["export_command"] = build_litert_export_command(
            model_source=merged_model,
            output_dir=litert_output_dir,
            export_cfg=format_plans[name]["export_config"]["export"],
        )

    official = _pipeline_section(config, "official_q8")
    official_source_value = str(official.get("source_litertlm") or "").strip()
    official_source = (
        _resolve_training_path(official_q8_source_override)
        if official_q8_source_override
        else (
            _resolve_training_path(official_source_value)
            if official_source_value
            else None
        )
    )
    configured_official_artifact = Path(
        str(
            official.get("artifact")
            or "gemma3_270m_w8_official_topology.litertlm"
        )
    )
    official_artifact = output_dir / "litertlm" / configured_official_artifact.name
    official_output = output_dir / "official_q8"
    official_command = [
        sys.executable,
        str(training_root() / "scripts" / "build_checkpoint_official_topology.py"),
        str(official_source) if official_source else "<official-q8-litertlm-required>",
        "--checkpoint",
        str(merged_model),
        "--family",
        "gemma3_270m",
        "--model-type",
        str(official.get("model_type") or "TF_LITE_PREFILL_DECODE"),
        "--training-config",
        str(resolved_training_config_path),
        "--official-base-model-id",
        str(official.get("base_model_id") or model_id),
        "--official-artifact-sha256",
        str(official.get("source_sha256") or "<sha256-required>"),
        "--output-dir",
        str(official_output),
        "--package-output",
        str(official_artifact),
        "--calibration-samples",
        str(int(official.get("calibration_samples", 2))),
        "--converter-batch-size",
        str(int(official.get("converter_batch_size", 8))),
        "--threads",
        str(int(official.get("threads", 1))),
        "--execute",
    ]
    if bool(official.get("enabled", False)):
        if official_source is None or not official_source.is_file():
            issues.append(
                _issue(
                    "error",
                    "official_q8_source_missing",
                    "official_q8.enabled=true requires the released Q8 .litertlm package.",
                )
            )
        else:
            expected_official_sha = (
                str(official.get("source_sha256") or "").strip().lower()
            )
            if (
                not expected_official_sha
                or _sha256(official_source) != expected_official_sha
            ):
                issues.append(
                    _issue(
                        "error",
                        "official_q8_source_sha256_mismatch",
                        "Released Q8 package does not match official_q8.source_sha256.",
                    )
                )

    checkpoint_eval_command = _evaluation_command(
        script=checkpoint_eval_script,
        model_args=[
            "--config",
            str(resolved_training_config_path),
            "--checkpoint",
            str(best_checkpoint),
            "--checkpoint-kind",
            "adapter",
        ],
        split=golden_split,
        output_dir=checkpoint_eval_dir,
        golden=golden,
        tensorboard_root=tensorboard_value,
        run_id=run_id,
        evaluation_name="checkpoint",
        step="<checkpoint-step>",
    )
    merged_eval_command = _evaluation_command(
        script=checkpoint_eval_script,
        model_args=[
            "--config",
            str(resolved_training_config_path),
            "--checkpoint",
            str(merged_model),
            "--checkpoint-kind",
            "merged",
        ],
        split=golden_split,
        output_dir=merged_eval_dir,
        golden=golden,
        tensorboard_root=tensorboard_value,
        run_id=run_id,
        evaluation_name="merged_checkpoint",
        step="<checkpoint-step>",
    )

    return {
        "schema_version": 1,
        "pipeline_id": pipeline_id,
        "run_id": run_id,
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "output_dir": str(output_dir),
        "training": {
            "config": str(training_config_path),
            "resolved_config": str(resolved_training_config_path),
            "resolved_config_sha256": resolved_training_config_sha256,
            "host_gpu_profile": host_gpu_profile,
            "world_size": int(host_gpu_profile["world_size"]) if host_gpu_profile else 1,
            "model_id": model_id,
            "qat_profile": qat.get("profile"),
            "qat_aligned_format": "w8",
            "output_dir": str(training_output),
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_ready": _checkpoint_ready(best_checkpoint),
            "tensorboard_root": str(tensorboard_root),
            "tensorboard_run_dir": str(expected_training_tb),
            "tensorboard_evaluation_run_dir": str(tensorboard_root / run_id),
            "prepare_golden_command": [
                sys.executable,
                str(training_root() / "scripts" / "prepare_dataset.py"),
                "--config",
                str(golden_dataset_config),
            ],
            "command": [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={int(host_gpu_profile['world_size']) if host_gpu_profile else 1}",
                str(training_root() / "scripts" / "train_sft.py"),
                "--config",
                str(resolved_training_config_path),
            ],
        },
        "golden": {
            "dataset_config": str(golden_dataset_config),
            "dataset_config_sha256": expected_config_sha,
            "required_rows": required_rows,
            "source": str(golden_source),
            "source_report": source_report,
            "responses": str(golden_responses),
            "responses_report": responses_report,
            "prepared_split": str(golden_split),
            "prepared_split_sha256": expected_split_sha,
            "prepared_report": split_report,
            "prepared_contract": prepared_contract_report,
            "metric_version": str(golden.get("metric_version") or "dual"),
            "metric_for_report": str(
                golden.get("metric_for_report") or "generation_reward_v5_4_avg"
            ),
            "training_leakage": leakage_report,
        },
        "checkpoint_evaluation": {
            "output_dir": str(checkpoint_eval_dir),
            "aggregate": str(checkpoint_eval_dir / "aggregate_metrics.json"),
            "command": checkpoint_eval_command,
        },
        "merge": {
            "base_model_id": model_id,
            "adapter_dir": str(best_checkpoint),
            "output_dir": str(merged_model),
            "output_ready": (merged_model / "config.json").is_file(),
            "merge_performs_qat": False,
            "requires_post_merge_quantization": True,
        },
        "merged_evaluation": {
            "output_dir": str(merged_eval_dir),
            "aggregate": str(merged_eval_dir / "aggregate_metrics.json"),
            "command": merged_eval_command,
        },
        "formats": format_plans,
        "official_q8": {
            "enabled": bool(official.get("enabled", False)),
            "scope": "w8_only",
            "source_litertlm": str(official_source) if official_source else None,
            "source_sha256": str(official.get("source_sha256") or ""),
            "artifact": str(official_artifact),
            "output_dir": str(official_output),
            "command": official_command,
        },
        "scorecard": {
            "path": str(output_dir / "final_scorecard.json"),
            "required_evaluations": [
                "checkpoint",
                *[f"litertlm_{name}" for name in FORMAT_ORDER],
            ],
        },
        "litertlm_runner": runner_report,
        "mtp": {"enabled": False, "status": "not_applicable_for_gemma3_270m"},
        "validation": {
            "config_ok": not any(item["severity"] == "error" for item in issues),
            "ok": not any(
                item["severity"] in {"error", "blocker"} for item in issues
            ),
            "ready_for_litertlm_evaluation": runner_report["ready"]
            and not any(item["severity"] == "error" for item in issues),
            "issues": issues,
        },
        "limitations": [
            "Only W8 is aligned with the W8 fake-QAT training objective.",
            "W16 uses the public exporter's experimental FP16 switch and requires explicit opt-in.",
            "W4 uses public block-32 PTQ and requires explicit opt-in; it is not an official 270M graph.",
            "Official-topology transplantation is valid only for W8 and requires the pinned released Q8 package.",
            "Every LiteRT-LM artifact must pass the full Golden-32 run through the configured external runtime; device/delegation checks remain separate promotion evidence.",
        ],
    }


def _run_command(command: list[str], log_path: Path, *, cwd: Path, environment: dict[str, str] | None = None) -> None:
    if log_path.exists():
        raise Gemma270MMultiformatError(
            f"Refusing to overwrite an existing stage log: {log_path}"
        )
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
            env=environment,
        )
        log.write(process.stdout or "")
    if process.returncode != 0:
        raise Gemma270MMultiformatError(
            f"Command failed with exit code {process.returncode}; see {log_path}"
        )


def _materialize_resolved_training_config(plan: dict[str, Any]) -> None:
    config = copy.deepcopy(load_yaml(plan["training"]["config"]))
    run = config.setdefault("run", {})
    training = config.setdefault("training", {})
    golden = config.setdefault("golden_eval", {})
    if not all(isinstance(value, dict) for value in (run, training, golden)):
        raise Gemma270MMultiformatError(
            "Training config run/training/golden_eval sections must be YAML objects."
        )
    run["id"] = plan["run_id"]
    run["output_dir"] = plan["training"]["output_dir"]
    training["logging_dir"] = plan["training"]["tensorboard_run_dir"]
    host_gpu_profile = plan["training"].get("host_gpu_profile")
    if host_gpu_profile is not None:
        apply_gpu_profile(config, host_gpu_profile)
        training["tensorboard_root"] = plan["training"]["tensorboard_root"]
    golden["best_checkpoint_dir"] = plan["training"]["best_checkpoint"]
    golden["output_dir"] = str(Path(plan["training"]["output_dir"]) / "golden32")
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != plan["training"]["resolved_config_sha256"]:
        raise Gemma270MMultiformatError(
            "Resolved training config changed after planning; rebuild the plan."
        )
    destination = Path(plan["training"]["resolved_config"])
    if destination.is_file():
        if _sha256(destination) != digest:
            raise Gemma270MMultiformatError(
                f"Refusing to replace a different resolved training config: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(text.encode("utf-8"))
    temporary.replace(destination)


def _require_fresh_training_output(plan: dict[str, Any]) -> None:
    output = Path(plan["training"]["output_dir"])
    if not output.exists():
        return
    allowed = {Path(plan["training"]["resolved_config"]).resolve()}
    unexpected = [
        candidate
        for candidate in output.rglob("*")
        if candidate.is_file() and candidate.resolve() not in allowed
    ]
    if unexpected:
        raise Gemma270MMultiformatError(
            "Refusing to reuse a non-empty Gemma 270M training run: "
            f"{output}. Choose a new --run-id."
        )


def _require_execution_inputs(plan: dict[str, Any]) -> None:
    errors = [
        item for item in plan["validation"]["issues"] if item["severity"] == "error"
    ]
    if errors:
        raise Gemma270MMultiformatError(json.dumps(errors, ensure_ascii=False))
    split = Path(plan["golden"]["prepared_split"])
    if not split.is_file():
        raise Gemma270MMultiformatError(
            f"Prepared Golden-32 is missing: {split}. Run --prepare-golden first."
        )
    plan["golden"]["prepared_contract"] = _validate_plan_golden(plan)


def _validate_plan_golden(plan: dict[str, Any]) -> dict[str, Any]:
    return validate_prepared_golden_contract(
        plan["golden"]["prepared_split"],
        dataset_config_path=plan["golden"]["dataset_config"],
        source_genui_path=plan["golden"]["source"],
        source_responses_path=plan["golden"]["responses"],
        expected_genui_sha256=plan["golden"]["source_report"]["sha256"],
        expected_responses_sha256=plan["golden"]["responses_report"]["sha256"],
        expected_dataset_config_sha256=plan["golden"]["dataset_config_sha256"],
        expected_split_sha256=plan["golden"]["prepared_split_sha256"],
        required_rows=plan["golden"]["required_rows"],
    )


def _require_training_leakage_gate(plan: dict[str, Any]) -> None:
    report = plan["golden"].get("training_leakage") or {}
    if not bool(report.get("checked")):
        raise Gemma270MMultiformatError(
            "Golden-32 leakage cannot be checked until both prepared training train.jsonl and val.jsonl exist."
        )
    if not bool(report.get("ok")):
        raise Gemma270MMultiformatError(
            "Golden-32 normalized response content overlaps the prepared "
            "training/validation data; refusing to train."
        )


def _require_litertlm_runner_ready(plan: dict[str, Any]) -> None:
    runner = plan.get("litertlm_runner")
    if not isinstance(runner, dict) or runner.get("ready") is not True:
        codes = [
            item["code"]
            for item in plan["validation"]["issues"]
            if item["code"]
            in {
                "missing_litertlm_runner_config",
                "invalid_litertlm_runner_config",
                "litertlm_runner_is_placeholder",
            }
        ]
        raise Gemma270MMultiformatError(
            "LiteRT-LM evaluation runner is not ready: "
            + ", ".join(codes or ["unknown_runner_error"])
        )


def _checkpoint_step(checkpoint: Path) -> int:
    metadata_path = checkpoint / "training_metadata.json"
    if not metadata_path.is_file():
        raise Gemma270MMultiformatError(
            f"Checkpoint provenance is missing: {metadata_path}"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Gemma270MMultiformatError(
            f"Checkpoint metadata must be an object: {metadata_path}"
        )
    golden = (
        payload.get("best_golden_eval")
        if isinstance(payload.get("best_golden_eval"), dict)
        else {}
    )
    value = golden.get("step", payload.get("checkpoint_step"))
    if type(value) is not int or value < 0:
        raise Gemma270MMultiformatError(
            f"Checkpoint step is missing from {metadata_path}."
        )
    return value


def _with_checkpoint_step(command: list[str], checkpoint: Path) -> list[str]:
    step = str(_checkpoint_step(checkpoint))
    return [str(value).replace("<checkpoint-step>", step) for value in command]


def _find_single_litertlm(path: Path) -> Path:
    candidates = sorted(path.rglob("*.litertlm")) if path.is_dir() else []
    if len(candidates) != 1:
        raise Gemma270MMultiformatError(
            f"Expected exactly one .litertlm under {path}, found {len(candidates)}."
        )
    return candidates[0]


def _export_format(
    plan: dict[str, Any], name: str, *, allow_experimental: bool
) -> None:
    format_plan = plan["formats"][name]
    if not format_plan["enabled"]:
        raise Gemma270MMultiformatError(f"formats.{name}.enabled=false")
    if format_plan["requires_explicit_opt_in"] and not allow_experimental:
        raise Gemma270MMultiformatError(
            f"{name} is experimental; pass --allow-experimental-formats to execute it."
        )
    merged = Path(plan["merge"]["output_dir"])
    if not (merged / "config.json").is_file():
        raise Gemma270MMultiformatError(
            f"Merged checkpoint is missing: {merged}. Run --execute-merge first."
        )
    export_dir = Path(format_plan["export_output_dir"])
    if export_dir.exists() and any(export_dir.iterdir()):
        raise Gemma270MMultiformatError(
            f"Refusing to mix an export with existing files: {export_dir}"
        )
    artifact = Path(format_plan["artifact"])
    if artifact.exists():
        raise Gemma270MMultiformatError(f"Refusing to overwrite artifact: {artifact}")
    export_edge_gallery_model(format_plan["export_config"], dry_run=False)
    candidate = _find_single_litertlm(Path(format_plan["litert_output_dir"]))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, artifact)
    format_plan["artifact_ready"] = True
    format_plan["artifact_sha256"] = _sha256(artifact)
    format_plan["artifact_size_bytes"] = artifact.stat().st_size
    format_plan["export_executed"] = True


def _merge(plan: dict[str, Any]) -> None:
    checkpoint = Path(plan["training"]["best_checkpoint"])
    if not _checkpoint_ready(checkpoint):
        raise Gemma270MMultiformatError(
            f"Golden-best LoRA checkpoint is missing: {checkpoint}"
        )
    output = Path(plan["merge"]["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise Gemma270MMultiformatError(
            f"Refusing to merge into non-empty output directory: {output}"
        )
    training_config = load_yaml(plan["training"]["resolved_config"])
    model = _section(training_config, "model")
    merge_lora_adapter(
        base_model_id=str(model.get("model_id") or ""),
        adapter_dir=checkpoint,
        output_dir=output,
        model_loader=str(model.get("model_loader") or "auto_causal_lm"),
        dtype=str(model.get("dtype") or "bfloat16"),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        processor_model_id=str(model.get("model_id") or ""),
        training_config_path=plan["training"]["resolved_config"],
    )
    plan["merge"]["output_ready"] = (output / "config.json").is_file()
    plan["merge"]["executed"] = True


def _use_official_q8_for_w8(plan: dict[str, Any]) -> None:
    """Route every W8 consumer in this invocation to the official artifact."""

    artifact = Path(plan["official_q8"]["artifact"])
    if not artifact.is_file():
        raise Gemma270MMultiformatError(
            f"Official Q8 exporter did not write {artifact}"
        )
    w8 = plan["formats"]["w8"]
    previous_artifact = str(w8["artifact"])
    w8["artifact"] = str(artifact)
    w8["artifact_source"] = "official_q8_topology"
    w8["artifact_ready"] = True
    w8["artifact_sha256"] = _sha256(artifact)
    w8["artifact_size_bytes"] = artifact.stat().st_size

    evaluation_command = w8["evaluation"]["command"]
    evaluation_command[evaluation_command.index("--model") + 1] = str(artifact)

    package_validation = w8["package_validation"]
    package_validation["artifact"] = str(artifact)
    package_command = package_validation["command"]
    replacements = 0
    for index, value in enumerate(package_command):
        if value == previous_artifact:
            package_command[index] = str(artifact)
            replacements += 1
    if replacements != 1:
        raise Gemma270MMultiformatError(
            "Could not bind the W8 package-inspection command to the official Q8 artifact."
        )


def _scorecard_entry(
    *,
    name: str,
    aggregate_path: Path,
    artifact: Path | None,
    metric_name: str,
    evaluation_name: str,
    run_id: str,
    required_rows: int,
    tensorboard_run_dir: Path,
    golden_split_path: Path,
    golden_split_sha256: str,
    metric_version: str,
    expected_step: int,
    artifact_field: str,
    package_inspection_path: Path | None = None,
    package_precision_contract: str | None = None,
    require_adapter_identity: bool = False,
) -> dict[str, Any]:
    if not aggregate_path.is_file():
        return {
            "name": name,
            "status": "missing",
            "aggregate": str(aggregate_path),
        }
    metrics = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise Gemma270MMultiformatError(
            f"Evaluation aggregate is not an object: {aggregate_path}"
        )
    entry: dict[str, Any] = {
        "name": name,
        "status": "complete",
        "aggregate": str(aggregate_path),
        "score_metric": metric_name,
        "score": metrics.get(metric_name),
        "metrics": metrics,
    }
    if (
        isinstance(entry["score"], bool)
        or not isinstance(entry["score"], (int, float))
        or not math.isfinite(float(entry["score"]))
    ):
        entry["status"] = "score_missing"
    if artifact is not None:
        entry["artifact"] = str(artifact)
        identity = _artifact_identity(
            artifact, require_adapter=require_adapter_identity
        )
        entry.update(identity)
        if not identity["artifact_exists"]:
            entry["status"] = "artifact_missing"
        elif not identity["artifact_identity_complete"]:
            entry["status"] = "artifact_identity_incomplete"
        if entry["status"] == "complete":
            try:
                entry["evidence"] = verify_evaluation_evidence(
                    aggregate_path=aggregate_path,
                    artifact_path=artifact,
                    artifact_field=artifact_field,
                    evaluation_name=evaluation_name,
                    run_id=run_id,
                    required_rows=required_rows,
                    tensorboard_run_dir=tensorboard_run_dir,
                    golden_split_path=golden_split_path,
                    golden_split_sha256=golden_split_sha256,
                    metric_version=metric_version,
                    expected_step=expected_step,
                    package_inspection_path=package_inspection_path,
                    package_precision_contract=package_precision_contract,
                )
            except EvaluationEvidenceError as exc:
                entry["status"] = "evidence_invalid"
                entry["evidence_error"] = str(exc)
    return entry


def write_final_scorecard(
    plan: dict[str, Any], *, require_complete: bool = True
) -> dict[str, Any]:
    metric_name = str(plan["golden"]["metric_for_report"])
    required_rows = int(plan["golden"]["required_rows"])
    tensorboard_run_dir = Path(
        plan["training"]["tensorboard_evaluation_run_dir"]
    )
    golden_contract = _validate_plan_golden(plan)
    plan["golden"]["prepared_contract"] = golden_contract
    golden_split_path = Path(plan["golden"]["prepared_split"])
    golden_split_sha256 = str(golden_contract.get("split_sha256") or "")
    metric_version = str(plan["golden"]["metric_version"])
    expected_step = _checkpoint_step(Path(plan["training"]["best_checkpoint"]))
    entries = [
        _scorecard_entry(
            name="checkpoint",
            aggregate_path=Path(plan["checkpoint_evaluation"]["aggregate"]),
            artifact=Path(plan["training"]["best_checkpoint"]),
            metric_name=metric_name,
            evaluation_name="checkpoint",
            run_id=plan["run_id"],
            required_rows=required_rows,
            tensorboard_run_dir=tensorboard_run_dir,
            golden_split_path=golden_split_path,
            golden_split_sha256=golden_split_sha256,
            metric_version=metric_version,
            expected_step=expected_step,
            artifact_field="checkpoint",
            require_adapter_identity=True,
        )
    ]
    for name in FORMAT_ORDER:
        item = plan["formats"][name]
        entries.append(
            _scorecard_entry(
                name=f"litertlm_{name}",
                aggregate_path=Path(item["evaluation"]["aggregate"]),
                artifact=Path(item["artifact"]),
                metric_name=metric_name,
                evaluation_name=f"litertlm_{name}",
                run_id=plan["run_id"],
                required_rows=required_rows,
                tensorboard_run_dir=tensorboard_run_dir,
                golden_split_path=golden_split_path,
                golden_split_sha256=golden_split_sha256,
                metric_version=metric_version,
                expected_step=expected_step,
                artifact_field="litertlm",
                package_inspection_path=Path(
                    item["package_validation"]["report"]
                ),
                package_precision_contract=name,
            )
        )
    missing = [item["name"] for item in entries if item["status"] != "complete"]
    if require_complete and missing:
        raise Gemma270MMultiformatError(
            "Cannot write a complete scorecard; missing or incomplete results: "
            + ", ".join(missing)
        )
    checkpoint_score = entries[0].get("score")
    if isinstance(checkpoint_score, (int, float)):
        for entry in entries[1:]:
            score = entry.get("score")
            if isinstance(score, (int, float)):
                entry["score_delta_vs_checkpoint"] = float(score) - float(
                    checkpoint_score
                )
    payload = {
        "schema_version": 1,
        "pipeline_id": plan["pipeline_id"],
        "run_id": plan["run_id"],
        "golden": plan["golden"],
        "score_metric": metric_name,
        "complete": not missing,
        "missing": missing,
        "results": entries,
        "notes": [
            "W8 is QAT-aligned; W32/W16/W4 are deployment sensitivity variants from the same merged checkpoint.",
            "All scores use the same immutable Golden-32 and deterministic generation settings.",
        ],
    }
    path = Path(plan["scorecard"]["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
    payload["path"] = str(path)
    return payload


def run_pipeline(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    run_id_override: str | None = None,
    formats: Iterable[str] | None = None,
    prepare_golden: bool = False,
    preflight_training: bool = False,
    execute_training: bool = False,
    evaluate_checkpoint: bool = False,
    execute_merge: bool = False,
    evaluate_merged: bool = False,
    execute_exports: bool = False,
    evaluate_litertlm: bool = False,
    execute_official_q8: bool = False,
    write_scorecard: bool = False,
    allow_experimental_formats: bool = False,
    best_checkpoint_override: str | Path | None = None,
    merged_model_override: str | Path | None = None,
    runner_config_override: str | Path | None = None,
    official_q8_source_override: str | Path | None = None,
) -> dict[str, Any]:
    if preflight_training or execute_training:
        config = copy.deepcopy(config)
        launch_training = config.setdefault("pipeline", {}).setdefault("training", {})
        if launch_training.get("host_gpu_profile") is None:
            launch_training["host_gpu_profile"] = build_gpu_profile(detect_cuda_devices(), model="270m")
        verify_gpu_profile(launch_training["host_gpu_profile"])
    if execute_official_q8:
        config = copy.deepcopy(config)
        pipeline = config.setdefault("pipeline", {})
        if not isinstance(pipeline, dict):
            raise Gemma270MMultiformatError("pipeline must be a YAML object.")
        official = pipeline.setdefault("official_q8", {})
        if not isinstance(official, dict):
            raise Gemma270MMultiformatError(
                "pipeline.official_q8 must be a YAML object."
            )
        official["enabled"] = True
    plan = build_pipeline_plan(
        config,
        config_path=config_path,
        run_id_override=run_id_override,
        formats=formats,
        best_checkpoint_override=best_checkpoint_override,
        merged_model_override=merged_model_override,
        runner_config_override=runner_config_override,
        official_q8_source_override=official_q8_source_override,
    )
    selected = [
        name
        for name in FORMAT_ORDER
        if plan["formats"][name]["selected"] and plan["formats"][name]["enabled"]
    ]
    logs_dir = Path(plan["output_dir"]) / "logs"
    if prepare_golden:
        _run_command(
            plan["training"]["prepare_golden_command"],
            logs_dir / "prepare_golden32.log",
            cwd=repo_root(),
        )
        plan = build_pipeline_plan(
            config,
            config_path=config_path,
            run_id_override=run_id_override,
            formats=selected,
            best_checkpoint_override=best_checkpoint_override,
            merged_model_override=merged_model_override,
            runner_config_override=runner_config_override,
            official_q8_source_override=official_q8_source_override,
        )

    any_execution = any(
        (
            execute_training,
            evaluate_checkpoint,
            execute_merge,
            evaluate_merged,
            execute_exports,
            evaluate_litertlm,
            execute_official_q8,
            write_scorecard,
        )
    )
    any_execution = any_execution or preflight_training
    if any_execution:
        _require_execution_inputs(plan)
        _materialize_resolved_training_config(plan)

    if preflight_training:
        verify_gpu_profile(plan["training"]["host_gpu_profile"])
        _require_training_leakage_gate(plan)
        _run_command(
            [*plan["training"]["command"], "--preflight-only"],
            logs_dir / "training_preflight.log",
            cwd=repo_root(),
            environment=training_environment(plan["training"]["host_gpu_profile"], tensorboard_root=plan["training"]["tensorboard_root"]),
        )
        plan["training"]["preflight_executed"] = True

    if execute_training:
        verify_gpu_profile(plan["training"]["host_gpu_profile"])
        _require_training_leakage_gate(plan)
        _require_fresh_training_output(plan)
        _run_command(
            plan["training"]["command"],
            logs_dir / "train_sft.log",
            cwd=repo_root(),
            environment=training_environment(plan["training"]["host_gpu_profile"], tensorboard_root=plan["training"]["tensorboard_root"]),
        )
        if not _checkpoint_ready(Path(plan["training"]["best_checkpoint"])):
            raise Gemma270MMultiformatError(
                "Training ended without the exact Golden-32 best LoRA checkpoint."
            )
        plan["training"]["best_checkpoint_ready"] = True
        plan["training"]["executed"] = True

    if evaluate_checkpoint:
        plan["golden"]["prepared_contract"] = _validate_plan_golden(plan)
        if not _checkpoint_ready(Path(plan["training"]["best_checkpoint"])):
            raise Gemma270MMultiformatError(
                "Best checkpoint is missing; cannot evaluate it."
            )
        _run_command(
            _with_checkpoint_step(
                plan["checkpoint_evaluation"]["command"],
                Path(plan["training"]["best_checkpoint"]),
            ),
            logs_dir / "evaluate_checkpoint.log",
            cwd=repo_root(),
        )
        plan["checkpoint_evaluation"]["executed"] = True

    if execute_merge:
        _merge(plan)

    if evaluate_merged:
        plan["golden"]["prepared_contract"] = _validate_plan_golden(plan)
        if not (Path(plan["merge"]["output_dir"]) / "config.json").is_file():
            raise Gemma270MMultiformatError(
                "Merged checkpoint is missing; cannot evaluate it."
            )
        _run_command(
            _with_checkpoint_step(
                plan["merged_evaluation"]["command"],
                Path(plan["training"]["best_checkpoint"]),
            ),
            logs_dir / "evaluate_merged.log",
            cwd=repo_root(),
        )
        plan["merged_evaluation"]["executed"] = True

    if execute_exports:
        for name in selected:
            _export_format(plan, name, allow_experimental=allow_experimental_formats)

    if execute_official_q8:
        if not plan["official_q8"]["enabled"]:
            raise Gemma270MMultiformatError(
                "official_q8.enabled=false; enable it and provide the pinned released Q8 package."
            )
        _run_command(
            plan["official_q8"]["command"],
            logs_dir / "official_q8_export.log",
            cwd=repo_root(),
        )
        official_artifact = Path(plan["official_q8"]["artifact"])
        _use_official_q8_for_w8(plan)
        plan["official_q8"]["executed"] = True
        plan["official_q8"]["artifact_sha256"] = _sha256(official_artifact)
        plan["official_q8"]["artifact_size_bytes"] = official_artifact.stat().st_size

    if evaluate_litertlm:
        plan["golden"]["prepared_contract"] = _validate_plan_golden(plan)
        _require_litertlm_runner_ready(plan)
        for name in selected:
            artifact = Path(plan["formats"][name]["artifact"])
            if not artifact.is_file():
                raise Gemma270MMultiformatError(
                    f"{name} artifact is missing; export it before Golden evaluation: {artifact}"
                )
            if (
                plan["formats"][name]["requires_explicit_opt_in"]
                and not allow_experimental_formats
            ):
                raise Gemma270MMultiformatError(
                    f"{name} is experimental; pass --allow-experimental-formats to evaluate it."
                )
            _run_command(
                plan["formats"][name]["package_validation"]["command"],
                logs_dir / f"inspect_litertlm_{name}.log",
                cwd=repo_root(),
            )
            package_report = Path(
                plan["formats"][name]["package_validation"]["report"]
            )
            if not package_report.is_file():
                raise Gemma270MMultiformatError(
                    f"Package audit wrote no inspection report for {name}: {package_report}"
                )
            try:
                package_payload = json.loads(
                    package_report.read_text(encoding="utf-8")
                )
                if not isinstance(package_payload, dict):
                    raise TypeError("inspection report is not a JSON object")
                plan["formats"][name]["package_validation"][
                    "precision_contract"
                ] = validate_litertlm_precision_contract(
                    package_payload,
                    expected_format=name,
                )
            except (
                OSError,
                TypeError,
                json.JSONDecodeError,
                EvaluationEvidenceError,
            ) as exc:
                raise Gemma270MMultiformatError(
                    f"{name} package precision validation failed: {exc}"
                ) from exc
            plan["formats"][name]["package_validation"]["executed"] = True
            _run_command(
                _with_checkpoint_step(
                    plan["formats"][name]["evaluation"]["command"],
                    Path(plan["training"]["best_checkpoint"]),
                ),
                logs_dir / f"evaluate_litertlm_{name}.log",
                cwd=repo_root(),
            )
            plan["formats"][name]["evaluation"]["executed"] = True

    if write_scorecard:
        plan["scorecard"]["result"] = write_final_scorecard(plan, require_complete=True)
    return plan
