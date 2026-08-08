#!/usr/bin/env python3
"""Inject a trained Gemma 4 E2B assistant into the official MTP graph.

The official package is the immutable graph/operator/layout authority.  The
input package may already contain a fine-tuned target, but its MTP section must
still be byte-identical to the official one.  Only the 23 mapped assistant
matrices and their per-output-channel scales are quantized and replaced; every
byte outside ``tf_lite_mtp_drafter`` is preserved.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_checkpoint_official_topology import (
    CheckpointTopologyError,
    SafetensorCheckpoint,
)
from build_converter_random_inventory_parity import (
    _build_random_float_tflite,
    _counter,
    _recipe,
)
from build_converter_random_topology_injection_parity import (
    _compact_converter_layout,
    _constants_digest,
    _converter_batch_ranges,
    _converter_weight_records,
    _file_range_sha256,
    _graph_report,
    _official_constants_digest,
    _package_boundary_report,
    _patch_official_constants,
    _runtime_allocate_report,
    _sha256,
    _write_injected_package,
)
from build_converter_topology_parity import (
    ConverterTopologyParityError,
    _read_section,
    _section_by_model_type,
)
from build_converter_topology_parity import (
    _quantize as _topology_quantize,
)
from build_fresh_random_quantized_graph import _extract_inventory
from ir_training.common.config import load_yaml
from ir_training.export.litertlm_inspector import inspect_litertlm
from ir_training.mtp.drafter_contract import (
    GEMMA4_E2B_MTP_MODEL_TYPE,
    OFFICIAL_GEMMA4_E2B_ASSISTANT,
    contract_summary,
    deployment_weight_specs,
    source_keys,
)
from ir_training.qat.fake_quant import QATSpec


class MTPDrafterTopologyError(RuntimeError):
    """Raised when trained drafter constants cannot be transplanted safely."""


_FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}


def _drafter_training_scope_report(
    training_config: str | Path | None,
    *,
    official_base_model_id: str | None,
) -> dict[str, Any]:
    """Validate the narrow 23-matrix MTP training/export contract."""

    checks: dict[str, bool] = {
        "training_config_present": False,
        "official_base_model_id_declared": bool(official_base_model_id),
        "assistant_base_matches_declared_official_base": False,
        "target_conditioned_autoregressive_qat": False,
        "four_draft_steps": False,
        "teacher_forcing": False,
        "completion_only_loss": False,
        "qat_enabled": False,
        "public_ai_edge_weight_ranges": False,
        "activation_int8": False,
        "symmetric_per_row_weights": False,
        "symmetric_per_tensor_activations": False,
        "no_group_quantization": False,
        "embedding_qat_enabled": False,
        "no_excluded_modules": False,
        "only_base_layers": False,
        "effective_merged_weight_disabled": False,
        "exact_23_weight_precision_assignment": False,
    }
    result: dict[str, Any] = {
        "path": str(Path(training_config).expanduser().resolve())
        if training_config
        else None,
        "official_base_model_id": official_base_model_id,
        "checks": checks,
        "supported_projection_only_transplant": False,
        "mutation_scope": "exact_23_mapped_matrices_only",
    }
    if training_config is None:
        return result
    path = Path(training_config).expanduser().resolve()
    if not path.is_file():
        result["error"] = f"Training config does not exist: {path}"
        return result
    try:
        config = load_yaml(path)
        spec = QATSpec.from_config(config)
    except (OSError, TypeError, ValueError) as exc:
        result["error"] = f"Could not load drafter training config: {exc}"
        return result

    checks["training_config_present"] = True
    assistant = (
        config.get("assistant")
        if isinstance(config.get("assistant"), dict)
        else {}
    )
    training = (
        config.get("training")
        if isinstance(config.get("training"), dict)
        else {}
    )
    qat = config.get("qat") if isinstance(config.get("qat"), dict) else {}
    assistant_id = str(assistant.get("model_id_or_path") or "")
    result["training_model_id"] = assistant_id
    checks["assistant_base_matches_declared_official_base"] = bool(
        official_base_model_id and assistant_id == str(official_base_model_id)
    )
    checks["target_conditioned_autoregressive_qat"] = (
        str(training.get("method") or "")
        == "target_conditioned_autoregressive_qat"
    )
    checks["four_draft_steps"] = int(training.get("draft_steps", 0) or 0) == 4
    checks["teacher_forcing"] = training.get("teacher_forcing") is True
    checks["completion_only_loss"] = training.get("completion_only_loss") is True
    checks["qat_enabled"] = qat.get("enabled") is True
    checks["public_ai_edge_weight_ranges"] = (
        str(qat.get("quantizer") or "").strip().lower() == "ste_ai_edge"
    )
    checks["activation_int8"] = int(qat.get("activation_bits", 0) or 0) == 8
    checks["symmetric_per_row_weights"] = bool(
        spec.weight_symmetric
        and spec.weight_per_channel
        and spec.weight_axis == 0
    )
    checks["symmetric_per_tensor_activations"] = spec.activation_symmetric
    checks["no_group_quantization"] = spec.group_size is None
    checks["embedding_qat_enabled"] = spec.quantize_embeddings
    checks["no_excluded_modules"] = not spec.exclude_modules
    checks["only_base_layers"] = spec.only_base_layers
    checks["effective_merged_weight_disabled"] = not spec.effective_merged_weight
    precision = [
        {
            "source_key": item.source_key,
            "module_name": item.module_name,
            "official_bits": item.bits,
            "qat_bits": spec.weight_bits_for_module(item.module_name),
        }
        for item in deployment_weight_specs()
    ]
    checks["exact_23_weight_precision_assignment"] = bool(
        len(precision) == 23
        and all(item["qat_bits"] == item["official_bits"] for item in precision)
    )
    result["precision_assignments"] = precision
    result["supported_projection_only_transplant"] = bool(all(checks.values()))
    result["preserved_constant_contract"] = (
        "Only the 23 mapped MTP matrices may differ. RMSNorm, centroid/token "
        "ordering, graph metadata, and all other constants remain official."
    )
    return result


def _checkpoint_metadata_path(checkpoint: Path) -> Path:
    return (
        checkpoint / "mtp_drafter_training_metadata.json"
        if checkpoint.is_dir()
        else checkpoint.parent / "mtp_drafter_training_metadata.json"
    )


def _training_provenance_report(
    checkpoint: Path,
    training_config: str | Path,
    *,
    official_assistant_model_id: str,
) -> dict[str, Any]:
    metadata_path = _checkpoint_metadata_path(checkpoint)
    checks = {
        "metadata_present": metadata_path.is_file(),
        "manifest_v1_or_newer": False,
        "assistant_base_matches": False,
        "training_config_hash_matches": False,
        "target_conditioned_qat_method": False,
        "teacher_forced_ce_loss": False,
        "target_frozen": False,
        "four_draft_steps": False,
        "teacher_forcing": False,
        "constant_position_ids": False,
        "fixed_target_kv": False,
        "unmapped_parameters_frozen": False,
        "public_training_constants_declared": False,
        "official_deployment_constants_declared": False,
        "private_recipe_not_claimed": False,
        "exact_trainable_source_keys": False,
        "exact_mobile_contract": False,
        "checkpoint_hashes_match": False,
    }
    result: dict[str, Any] = {
        "path": str(metadata_path),
        "checks": checks,
        "verified": False,
    }
    if not metadata_path.is_file():
        return result
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"Could not read drafter training metadata: {exc}"
        return result
    if not isinstance(metadata, dict):
        result["error"] = "Drafter training metadata must be a JSON object."
        return result
    result["metadata"] = metadata
    checks["manifest_v1_or_newer"] = int(
        metadata.get("manifest_version", 0) or 0
    ) >= 1
    checks["assistant_base_matches"] = (
        str(metadata.get("assistant_base_model_id_or_path") or "")
        == official_assistant_model_id
    )
    config_path = Path(training_config).expanduser().resolve()
    if config_path.is_file():
        expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        result["expected_training_config_sha256"] = expected_hash
        checks["training_config_hash_matches"] = (
            str(metadata.get("training_config_sha256") or "").lower()
            == expected_hash
        )
    checks["target_conditioned_qat_method"] = (
        str(metadata.get("training_method") or "")
        == "target_conditioned_autoregressive_qat"
    )
    checks["teacher_forced_ce_loss"] = (
        metadata.get("loss") == "teacher_forced_completion_cross_entropy"
    )
    checks["target_frozen"] = metadata.get("target_frozen") is True
    checks["four_draft_steps"] = int(metadata.get("draft_steps", 0) or 0) == 4
    checks["teacher_forcing"] = metadata.get("teacher_forcing") is True
    checks["constant_position_ids"] = metadata.get("constant_position_ids") is True
    checks["fixed_target_kv"] = (
        metadata.get("fixed_target_kv_during_each_draft_rollout") is True
    )
    checks["unmapped_parameters_frozen"] = (
        metadata.get("unmapped_parameters_frozen") is True
    )
    checks["public_training_constants_declared"] = (
        metadata.get("training_unmapped_constants_source")
        == "public_assistant_checkpoint"
    )
    checks["official_deployment_constants_declared"] = (
        metadata.get("deployment_unmapped_constants_source")
        == "official_litertlm_artifact"
    )
    checks["private_recipe_not_claimed"] = (
        metadata.get("private_google_training_recipe_recovered") is False
    )
    checks["exact_trainable_source_keys"] = (
        metadata.get("trainable_source_keys") == source_keys()
    )
    checks["exact_mobile_contract"] = metadata.get("mobile_contract") == contract_summary()

    file_checks: list[dict[str, Any]] = []
    checkpoint_root = checkpoint if checkpoint.is_dir() else checkpoint.parent
    declared = metadata.get("checkpoint_files")
    if isinstance(declared, list) and declared:
        for item in declared:
            if not isinstance(item, dict):
                file_checks.append({"valid": False, "reason": "entry_not_object"})
                continue
            relative = Path(str(item.get("path") or ""))
            safe = bool(
                str(relative)
                and not relative.is_absolute()
                and ".." not in relative.parts
            )
            candidate = (checkpoint_root / relative).resolve()
            within = bool(safe and candidate.is_relative_to(checkpoint_root.resolve()))
            expected_size = int(item.get("size", 0) or 0)
            expected_sha = str(item.get("sha256") or "").lower()
            observed_size = (
                int(candidate.stat().st_size)
                if within and candidate.is_file()
                else None
            )
            observed_sha = (
                _file_range_sha256(candidate, 0, observed_size)
                if observed_size is not None
                else None
            )
            file_checks.append(
                {
                    "path": relative.as_posix(),
                    "expected_size": expected_size,
                    "observed_size": observed_size,
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                    "valid": bool(
                        within
                        and expected_size > 0
                        and len(expected_sha) == 64
                        and expected_size == observed_size
                        and expected_sha == observed_sha
                    ),
                }
            )
    checks["checkpoint_hashes_match"] = bool(
        file_checks and all(item.get("valid", False) for item in file_checks)
    )
    result["checkpoint_file_verification"] = file_checks
    result["verified"] = bool(all(checks.values()))
    return result


def _section_digest(path: Path, section: dict[str, Any]) -> str:
    return _file_range_sha256(
        path,
        int(section["begin_offset"]),
        int(section["size"]),
    )


def build_plan(
    official_artifact: str | Path,
    package_input: str | Path,
    assistant_checkpoint: str | Path,
    *,
    assistant_training_config: str | Path,
    official_artifact_sha256: str,
    official_assistant_model_id: str = OFFICIAL_GEMMA4_E2B_ASSISTANT,
    output_dir: str | Path,
    package_output: str | Path,
) -> dict[str, Any]:
    official_path = Path(official_artifact).expanduser().resolve()
    input_path = Path(package_input).expanduser().resolve()
    checkpoint_path = Path(assistant_checkpoint).expanduser().resolve()
    config_path = Path(assistant_training_config).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_path = Path(package_output).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    declared_sha = str(official_artifact_sha256 or "").strip().lower()
    observed_sha = None
    official_mtp: dict[str, Any] | None = None
    input_mtp: dict[str, Any] | None = None
    official_target: dict[str, Any] | None = None
    input_target: dict[str, Any] | None = None
    target_topology: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []

    if len(declared_sha) != 64 or any(
        character not in "0123456789abcdef" for character in declared_sha
    ):
        issues.append({"code": "official_artifact_sha256_required"})
    if not official_path.is_file():
        issues.append({"code": "official_artifact_missing", "path": str(official_path)})
    else:
        observed_sha = _file_range_sha256(
            official_path, 0, official_path.stat().st_size
        )
        if declared_sha and observed_sha != declared_sha:
            issues.append(
                {
                    "code": "official_artifact_sha256_mismatch",
                    "expected": declared_sha,
                    "observed": observed_sha,
                }
            )
    if not input_path.is_file():
        issues.append({"code": "package_input_missing", "path": str(input_path)})
    if output_path in {official_path, input_path}:
        issues.append(
            {"code": "output_would_overwrite_input", "path": str(output_path)}
        )

    if official_path.is_file() and input_path.is_file():
        try:
            official_package = inspect_litertlm(official_path, inspect_tflite=False)
            input_package = inspect_litertlm(input_path, inspect_tflite=False)
            official_mtp = _section_by_model_type(
                official_package, GEMMA4_E2B_MTP_MODEL_TYPE
            )
            input_mtp = _section_by_model_type(
                input_package, GEMMA4_E2B_MTP_MODEL_TYPE
            )
            official_target = _section_by_model_type(
                official_package, "tf_lite_prefill_decode"
            )
            input_target = _section_by_model_type(
                input_package, "tf_lite_prefill_decode"
            )
            _, records = _extract_inventory(
                official_path,
                GEMMA4_E2B_MTP_MODEL_TYPE,
                include_embeddings=True,
                max_weights=None,
            )
            if official_path.stat().st_size != input_path.stat().st_size:
                issues.append({"code": "package_size_mismatch"})
            if (
                int(official_mtp["begin_offset"]) != int(input_mtp["begin_offset"])
                or int(official_mtp["size"]) != int(input_mtp["size"])
            ):
                issues.append({"code": "mtp_section_boundary_mismatch"})
            if _section_digest(official_path, official_mtp) != _section_digest(
                input_path, input_mtp
            ):
                issues.append(
                    {
                        "code": "input_mtp_not_official_byte_exact",
                        "message": (
                            "The target package must still contain the official drafter "
                            "before trained-weight injection."
                        ),
                    }
                )
            if (
                int(official_target["begin_offset"])
                != int(input_target["begin_offset"])
                or int(official_target["size"]) != int(input_target["size"])
            ):
                issues.append({"code": "target_section_boundary_mismatch"})
            else:
                target_begin = int(official_target["begin_offset"])
                target_size = int(official_target["size"])
                target_end = target_begin + target_size
                prefix_exact = _file_range_sha256(
                    official_path, 0, target_begin
                ) == _file_range_sha256(input_path, 0, target_begin)
                suffix_size = official_path.stat().st_size - target_end
                suffix_exact = _file_range_sha256(
                    official_path, target_end, suffix_size
                ) == _file_range_sha256(input_path, target_end, suffix_size)
                official_target_graph = _graph_report(
                    _read_section(official_path, official_target)
                )
                input_target_graph = _graph_report(
                    _read_section(input_path, input_target)
                )
                target_topology = {
                    "official_graph": official_target_graph,
                    "input_graph": input_target_graph,
                    "official_structure": (
                        official_target_graph["graph"].get("structural_sha256")
                        == input_target_graph["graph"].get("structural_sha256")
                    ),
                    "official_quantization_layout": (
                        official_target_graph["graph"].get(
                            "quantization_layout_sha256"
                        )
                        == input_target_graph["graph"].get(
                            "quantization_layout_sha256"
                        )
                    ),
                    "bytes_outside_target_official_exact": bool(
                        prefix_exact and suffix_exact
                    ),
                }
                if not all(
                    target_topology[key]
                    for key in (
                        "official_structure",
                        "official_quantization_layout",
                        "bytes_outside_target_official_exact",
                    )
                ):
                    issues.append(
                        {
                            "code": "input_target_not_official_topology_package",
                            "checks": {
                                key: target_topology[key]
                                for key in (
                                    "official_structure",
                                    "official_quantization_layout",
                                    "bytes_outside_target_official_exact",
                                )
                            },
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - all inspection failures are gates
            issues.append({"code": "package_inspection_failed", "error": str(exc)})

    expected_specs = deployment_weight_specs()
    if records:
        if len(records) != len(expected_specs):
            issues.append(
                {
                    "code": "unexpected_mtp_inventory_count",
                    "expected": len(expected_specs),
                    "observed": len(records),
                }
            )
        else:
            mismatches = [
                {
                    "ordinal": spec.ordinal,
                    "expected_bits": spec.bits,
                    "observed_bits": int(record["bits"]),
                    "expected_shape": list(spec.shape),
                    "observed_shape": list(record["shape"]),
                }
                for spec, record in zip(expected_specs, records)
                if spec.bits != int(record["bits"])
                or spec.shape != tuple(int(value) for value in record["shape"])
            ]
            if mismatches:
                issues.append(
                    {"code": "official_mtp_contract_mismatch", "items": mismatches}
                )

    checkpoint_index = None
    mappings: list[dict[str, Any]] = []
    try:
        checkpoint_index = SafetensorCheckpoint(checkpoint_path)
        for spec in expected_specs:
            if spec.source_key not in checkpoint_index.entries:
                issues.append(
                    {
                        "code": "assistant_tensor_missing",
                        "source_key": spec.source_key,
                    }
                )
                continue
            entry = checkpoint_index.entries[spec.source_key]
            source_shape = tuple(int(value) for value in entry.get("shape", []))
            if source_shape != spec.shape:
                issues.append(
                    {
                        "code": "assistant_tensor_shape_mismatch",
                        "source_key": spec.source_key,
                        "expected": list(spec.shape),
                        "observed": list(source_shape),
                    }
                )
                continue
            source_dtype = str(entry.get("dtype") or "").upper()
            if source_dtype not in _FLOAT_DTYPES:
                issues.append(
                    {
                        "code": "assistant_tensor_not_float",
                        "source_key": spec.source_key,
                        "observed": source_dtype,
                    }
                )
                continue
            mappings.append(
                {
                    "ordinal": spec.ordinal,
                    "source_key": spec.source_key,
                    "bits": spec.bits,
                    "shape": list(spec.shape),
                    "source_dtype": source_dtype,
                    "source_shard": str(
                        checkpoint_index.key_to_shard[spec.source_key]
                    ),
                }
            )
    except CheckpointTopologyError as exc:
        issues.append({"code": "assistant_checkpoint_invalid", "error": str(exc)})

    training_scope = _drafter_training_scope_report(
        config_path,
        official_base_model_id=official_assistant_model_id,
    )
    if not training_scope["supported_projection_only_transplant"]:
        issues.append(
            {
                "code": "unsupported_drafter_training_scope",
                "checks": training_scope["checks"],
            }
        )
    provenance = _training_provenance_report(
        checkpoint_path,
        config_path,
        official_assistant_model_id=official_assistant_model_id,
    )
    if not provenance["verified"]:
        issues.append(
            {
                "code": "unverified_drafter_training_provenance",
                "checks": provenance["checks"],
            }
        )

    return {
        "schema_version": 1,
        "official_artifact": str(official_path),
        "official_artifact_sha256_expected": declared_sha or None,
        "official_artifact_sha256_observed": observed_sha,
        "package_input": str(input_path),
        "assistant_checkpoint": str(checkpoint_path),
        "assistant_training_config": str(config_path),
        "official_assistant_model_id": official_assistant_model_id,
        "model_type": GEMMA4_E2B_MTP_MODEL_TYPE,
        "official_mtp_section": official_mtp,
        "input_mtp_section": input_mtp,
        "official_target_section": official_target,
        "input_target_section": input_target,
        "input_target_topology": target_topology,
        "official_inventory_count": len(records),
        "mapping_count": len(mappings),
        "mappings": mappings,
        "training_scope": training_scope,
        "training_provenance": provenance,
        "mobile_contract": contract_summary(),
        "output_dir": str(output_root),
        "package_output": str(output_path),
        "training_executed": False,
        "private_google_recipe_recovered": False,
        "official_graph_is_template_authority": True,
        "ready": not issues,
        "issues": issues,
        "remaining_runtime_gates": [
            "Android GPU full-delegation parity with MTP enabled",
            "MTP draft acceptance parity",
            "bounded warm speculative throughput parity",
        ],
    }


def run(
    official_artifact: str | Path,
    package_input: str | Path,
    assistant_checkpoint: str | Path,
    *,
    assistant_training_config: str | Path,
    official_artifact_sha256: str,
    official_assistant_model_id: str,
    output_dir: str | Path,
    package_output: str | Path,
    calibration_samples: int = 2,
    threads: int = 1,
    converter_batch_size: int | None = None,
    retain_intermediates: bool = False,
    runtime_allocate: bool = False,
    runtime_threads: int = 2,
    runtime_without_default_delegates: bool = False,
) -> dict[str, Any]:
    plan = build_plan(
        official_artifact,
        package_input,
        assistant_checkpoint,
        assistant_training_config=assistant_training_config,
        official_artifact_sha256=official_artifact_sha256,
        official_assistant_model_id=official_assistant_model_id,
        output_dir=output_dir,
        package_output=package_output,
    )
    if not plan["ready"]:
        raise MTPDrafterTopologyError(
            "Trained drafter topology plan is not executable: "
            + json.dumps(plan["issues"], ensure_ascii=False)
        )
    output_root = Path(plan["output_dir"])
    output_path = Path(plan["package_output"])
    if output_root.exists() and any(output_root.iterdir()):
        raise MTPDrafterTopologyError(
            f"Output directory is non-empty: {output_root}"
        )
    if output_path.exists():
        raise MTPDrafterTopologyError(f"Refusing to overwrite output: {output_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    official_path = Path(plan["official_artifact"])
    input_path = Path(plan["package_input"])
    official_package = inspect_litertlm(official_path, inspect_tflite=False)
    input_package = inspect_litertlm(input_path, inspect_tflite=False)
    official_section = _section_by_model_type(
        official_package, GEMMA4_E2B_MTP_MODEL_TYPE
    )
    input_section = _section_by_model_type(
        input_package, GEMMA4_E2B_MTP_MODEL_TYPE
    )
    input_target = _section_by_model_type(input_package, "tf_lite_prefill_decode")
    official_bytes = _read_section(official_path, official_section)
    _, official_records = _extract_inventory(
        official_path,
        GEMMA4_E2B_MTP_MODEL_TYPE,
        include_embeddings=True,
        max_weights=None,
    )
    checkpoint = SafetensorCheckpoint(plan["assistant_checkpoint"])
    chosen_batch_size, ranges = _converter_batch_ranges(
        official_records, converter_batch_size
    )
    converter_records: list[dict[str, Any]] = []
    converter_batches: list[dict[str, Any]] = []
    converter_digest = hashlib.sha256()
    quantized_paths: list[Path] = []
    recipe: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    keys = source_keys()

    try:
        with checkpoint:
            for batch_index, (start, end) in enumerate(ranges):
                batch_records = official_records[start:end]

                def provide_weight(
                    ordinal: int, _record: dict[str, Any]
                ) -> np.ndarray:
                    key = keys[int(ordinal)]
                    values = checkpoint.load_float32(key)
                    expected = tuple(
                        int(value) for value in official_records[int(ordinal)]["shape"]
                    )
                    if tuple(values.shape) != expected:
                        raise MTPDrafterTopologyError(
                            f"Loaded {key!r} shape {tuple(values.shape)} != {expected}."
                        )
                    if not bool(np.all(np.isfinite(values))):
                        raise MTPDrafterTopologyError(
                            f"Loaded {key!r} contains NaN or Inf."
                        )
                    loaded.append(
                        {
                            "ordinal": int(ordinal),
                            "source_key": key,
                            "shape": list(expected),
                            "float32_sha256": hashlib.sha256(
                                memoryview(values).cast("B")
                            ).hexdigest(),
                        }
                    )
                    return values

                float_bytes, _ = _build_random_float_tflite(
                    batch_records,
                    seed=0,
                    ordinal_offset=start,
                    weight_provider=provide_weight,
                    graph_description="Trained Gemma 4 MTP deployment inventory",
                )
                batch_recipe = _recipe(batch_records, ordinal_offset=start)
                recipe.extend(batch_recipe)
                float_path = output_root / f"mtp_fp32_batch_{batch_index:04d}.tflite"
                quantized_path = (
                    output_root / f"mtp_quantized_batch_{batch_index:04d}.tflite"
                )
                if retain_intermediates:
                    float_path.write_bytes(float_bytes)
                try:
                    _topology_quantize(
                        float_path if retain_intermediates else float_bytes,
                        quantized_path,
                        batch_recipe,
                        calibration_samples=calibration_samples,
                        threads=threads,
                    )
                except ConverterTopologyParityError as exc:
                    raise MTPDrafterTopologyError(
                        f"Could not quantize trained MTP batch {batch_index}: {exc}"
                    ) from exc
                quantized_bytes = quantized_path.read_bytes()
                observed = _converter_weight_records(quantized_bytes)
                if [int(item["ordinal"]) for item in observed] != list(
                    range(start, end)
                ):
                    raise MTPDrafterTopologyError(
                        f"Converter batch {batch_index} returned unexpected ordinals."
                    )
                converter_records.extend(observed)
                converter_digest.update(quantized_bytes)
                quantized_paths.append(quantized_path)
                converter_batches.append(
                    {
                        "index": batch_index,
                        "start_ordinal": start,
                        "end_ordinal_exclusive": end,
                        "weight_count": end - start,
                        "float_size": len(float_bytes),
                        "quantized_size": len(quantized_bytes),
                    }
                )
    finally:
        gc.collect()

    (output_root / "mtp_checkpoint_quantization_recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    injected_bytes, injection = _patch_official_constants(
        official_bytes, official_records, converter_records
    )
    official_graph = _graph_report(official_bytes)
    candidate_graph = _graph_report(injected_bytes)
    converter_constants = _constants_digest(converter_records)
    injected_constants = _official_constants_digest(
        injected_bytes, official_records
    )
    gates = {
        "training_scope_supported": bool(
            plan["training_scope"]["supported_projection_only_transplant"]
        ),
        "training_provenance_verified": bool(
            plan["training_provenance"]["verified"]
        ),
        "complete_23_weight_mapping": len(loaded) == len(official_records) == 23,
        "converter_inventory_count": len(converter_records)
        == len(official_records),
        "converter_quantization_layout": _counter(official_records)
        == _counter(converter_records),
        "converter_constants_transferred_exactly": converter_constants
        == injected_constants,
        "official_graph_structure": (
            official_graph["graph"].get("structural_sha256")
            == candidate_graph["graph"].get("structural_sha256")
        ),
        "official_quantization_layout": (
            official_graph["graph"].get("quantization_layout_sha256")
            == candidate_graph["graph"].get("quantization_layout_sha256")
        ),
        "official_execution_topology_ignoring_buffer_indices": (
            official_graph["graph_without_buffer_indices"].get("structural_sha256")
            == candidate_graph["graph_without_buffer_indices"].get(
                "structural_sha256"
            )
        ),
        "official_layout_ignoring_buffer_indices": (
            official_graph["graph_without_buffer_indices"].get(
                "quantization_layout_sha256"
            )
            == candidate_graph["graph_without_buffer_indices"].get(
                "quantization_layout_sha256"
            )
        ),
        "section_size_unchanged": len(injected_bytes) == len(official_bytes),
    }
    result: dict[str, Any] = {
        **{key: value for key, value in plan.items() if key != "mappings"},
        "checkpoint_weights_loaded": len(loaded),
        "checkpoint_mappings": sorted(loaded, key=lambda item: item["ordinal"]),
        "converter": {
            "public_ai_edge_quantizer_executed": True,
            "batch_size": int(chosen_batch_size),
            "batch_count": len(converter_batches),
            "batches": converter_batches,
            "combined_model_sha256": converter_digest.hexdigest(),
            "weight_count": len(converter_records),
            "layout": _compact_converter_layout(converter_records),
            "constants_sha256": converter_constants,
        },
        "injection": {
            **injection,
            "section_size": len(injected_bytes),
            "section_sha256": _sha256(injected_bytes),
            "constants_sha256": injected_constants,
        },
        "official_section": official_graph,
        "candidate_section": candidate_graph,
        "gates": gates,
        "assistant_weight_source": "trained_checkpoint",
        "assistant_trained": True,
        "learned_weights_expected_to_differ": True,
        "private_google_recipe_recovered": False,
        "android_gpu_validation_required": True,
    }
    if runtime_allocate:
        result["host_runtime"] = _runtime_allocate_report(
            official_bytes,
            injected_bytes,
            threads=runtime_threads,
            without_default_delegates=runtime_without_default_delegates,
        )
        gates["host_runtime_allocation_match"] = bool(
            result["host_runtime"].get("allocation_match")
        )
    result["final_artifact_gate_pass"] = bool(all(gates.values()))
    report_path = output_root / "mtp_checkpoint_official_topology_report.json"
    if not result["final_artifact_gate_pass"]:
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise MTPDrafterTopologyError(
            "Trained MTP graph/layout gates failed; no final package was promoted."
        )

    package_boundary = _package_boundary_report(
        input_path, input_section, injected_bytes
    )
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        raise MTPDrafterTopologyError(
            f"Refusing to overwrite quarantined partial package: {partial}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_injected_package(input_path, input_section, injected_bytes, partial)
    prefix_size = int(input_section["begin_offset"])
    suffix_offset = prefix_size + int(input_section["size"])
    suffix_size = input_path.stat().st_size - suffix_offset
    outside_exact = bool(
        partial.stat().st_size == input_path.stat().st_size
        and _file_range_sha256(partial, 0, prefix_size)
        == package_boundary["prefix_sha256"]
        and _file_range_sha256(partial, suffix_offset, suffix_size)
        == package_boundary["suffix_sha256"]
    )
    target_begin = int(input_target["begin_offset"])
    target_size = int(input_target["size"])
    target_before = _file_range_sha256(input_path, target_begin, target_size)
    target_after = _file_range_sha256(partial, target_begin, target_size)
    gates["package_outside_mtp_byte_exact"] = outside_exact
    gates["fine_tuned_target_section_byte_exact"] = target_before == target_after
    result["package_boundary"] = {
        **package_boundary,
        "target_section_sha256_before": target_before,
        "target_section_sha256_after": target_after,
        "bytes_outside_mtp_verified_unchanged": outside_exact,
        "quarantined_partial": str(partial),
    }
    result["final_artifact_gate_pass"] = bool(all(gates.values()))
    if not result["final_artifact_gate_pass"]:
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise MTPDrafterTopologyError(
            f"Package boundary verification failed; partial remains at {partial}."
        )
    partial.replace(output_path)
    result["package_boundary"]["output"] = str(output_path)
    result["package_boundary"]["output_sha256"] = _file_range_sha256(
        output_path, 0, output_path.stat().st_size
    )
    result["package_boundary"]["quarantined_partial"] = None

    cleanup: dict[str, Any] = {"requested": not retain_intermediates, "removed": []}
    if not retain_intermediates:
        gc.collect()
        for item in quantized_paths:
            try:
                item.unlink(missing_ok=True)
                cleanup["removed"].append(str(item))
            except PermissionError:
                cleanup.setdefault("locked", []).append(str(item))
    result["temporary_fixture_cleanup"] = cleanup
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_artifact")
    parser.add_argument("--package-input", required=True)
    parser.add_argument("--assistant-checkpoint", required=True)
    parser.add_argument("--assistant-training-config", required=True)
    parser.add_argument(
        "--official-assistant-model-id",
        default=OFFICIAL_GEMMA4_E2B_ASSISTANT,
    )
    parser.add_argument("--official-artifact-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-output", required=True)
    parser.add_argument("--calibration-samples", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--converter-batch-size", type=int)
    parser.add_argument("--retain-intermediates", action="store_true")
    parser.add_argument("--runtime-allocate", action="store_true")
    parser.add_argument("--runtime-threads", type=int, default=2)
    parser.add_argument("--runtime-without-default-delegates", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if min(args.calibration_samples, args.threads, args.runtime_threads) < 1:
        parser.error("sample/thread counts must be positive")
    if args.converter_batch_size is not None and args.converter_batch_size < 1:
        parser.error("--converter-batch-size must be positive")
    kwargs = {
        "assistant_training_config": args.assistant_training_config,
        "official_artifact_sha256": args.official_artifact_sha256,
        "official_assistant_model_id": args.official_assistant_model_id,
        "output_dir": args.output_dir,
        "package_output": args.package_output,
    }
    try:
        if args.execute:
            result = run(
                args.official_artifact,
                args.package_input,
                args.assistant_checkpoint,
                **kwargs,
                calibration_samples=args.calibration_samples,
                threads=args.threads,
                converter_batch_size=args.converter_batch_size,
                retain_intermediates=args.retain_intermediates,
                runtime_allocate=args.runtime_allocate,
                runtime_threads=args.runtime_threads,
                runtime_without_default_delegates=args.runtime_without_default_delegates,
            )
        else:
            result = build_plan(
                args.official_artifact,
                args.package_input,
                args.assistant_checkpoint,
                **kwargs,
            )
    except (OSError, ValueError, CheckpointTopologyError, MTPDrafterTopologyError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.execute:
        print("Plan only: no training, quantization, or package write was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
