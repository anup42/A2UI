"""Read-only relocation-safe binding for the shared Android native evaluator.

Training/export already verified the external base and SFT source. This gate
checks their immutable reports and all local adapter/package bytes; it never
downloads a model or requires the training host's original source directories.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, training_root
from ir_training.eval import android_native_quality as native
from ir_training.export.merge_lora import (
    _adapter_file_records,
    _checkpoint_manifest_matches_adapter,
    _golden_selection_binding,
)
from ir_training.pipeline.official_mobile import NO_OP_CHECKS, SELECTOR
from ir_training.qat.mobile_training_seed import OFFICIAL_LITERTLM_SHA256
from ir_training.train.qat_grpo_contract import validate_qat_grpo_config

_E2B_GATES = {
    "zero_adapter_target_byte_exact", "processed_205", "base_lora_candidate_dtype_match_205",
    "base_lora_candidate_bfloat16_205", "base_lora_numerical_parity_205", "base_lora_code_parity_205",
    "base_delta_norms_finite_205", "at_least_one_lora_delta_nonzero", "changed_buffers_within_expected_205",
    "at_least_one_trained_code_changed", "restored_official_payload_target_byte_exact", "frozen_72_byte_exact",
    "weight_qparams_byte_exact", "activation_a8_qparams_byte_exact", "all_tensor_qparams_byte_exact",
    "official_retained_weight_scales_exact", "official_retained_a8_scales_exact", "graph_layout_execution_identity",
    "section_size_unchanged", "exact_205_key_buffer_bijection", "expected_bit_histogram", "retained_qparams_verified",
    "legacy_metadata_rejected", "package_outside_target_byte_exact", "mtp_byte_exact", "package_parseable",
    "qat_weight_code_parity_205", "qat_dequantized_weight_forward_parity_205",
}
_270M_GATES = {
    "training_scope_supported", "retained_constant_compatibility_verified", "merge_provenance_verified",
    "complete_checkpoint_mapping", "expected_official_inventory", "converter_inventory_count",
    "converter_quantization_layout", "converter_constants_transferred_exactly", "non_mapped_state_byte_exact",
    "official_graph_structure", "official_quantization_layout", "official_execution_topology_ignoring_buffer_indices",
    "official_layout_ignoring_buffer_indices", "execution_contract_complete", "official_execution_contract",
    "official_execution_contract_ignoring_buffer_indices", "section_size_unchanged", "package_outside_target_byte_exact",
}


def _all_gates(value: Any, required: set[str]) -> bool:
    return bool(isinstance(value, dict) and required.issubset(value) and value
                and all(item is True for item in value.values()))


def inspect_qat_grpo_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve()
    plan_path, manifest_path = run / "qat_grpo_plan.json", run / "qat_grpo_manifest.json"
    plan, manifest = native.read_json(plan_path), native.read_json(manifest_path)
    if (plan.get("workflow") != "qat_lora_grpo_v1" or manifest.get("workflow") != plan["workflow"]
            or manifest.get("plan") != plan):
        raise ValueError("GRPO native evaluation requires its own manifest bound to the saved plan")
    status = manifest.get("status")
    if status not in {"awaiting_native_evaluation", "complete"} and not (
        status == "running" and manifest.get("active_stage") == "native_quality"
    ):
        raise ValueError("GRPO native evaluation requires completed export prerequisites")
    values = plan.get("options") or {}
    family = values.get("family")
    if family not in {"e2b", "270m"}:
        raise ValueError("Unknown GRPO model family")
    original_root = str(values["output_dir"])

    def relocated(value: Any) -> Path:
        if not value:
            raise ValueError("Missing run-local artifact path")
        path = native.relocate_run_path(value, original_root=original_root, current_root=run)
        if not path.is_relative_to(run):
            raise ValueError("Artifact path escapes the copied GRPO run")
        return path

    paths = {name: relocated(value) for name, value in (plan.get("paths") or {}).items()}
    required_paths = {"prepared", "config", "best_checkpoint", "merged", "export_report", "litertlm"}
    if family == "e2b":
        required_paths.add("no_op_export_report")
    if not required_paths.issubset(paths):
        raise ValueError("GRPO plan lacks required artifact paths")
    expected_stages = ["assets", "prepare", "configure", *(["no_op_export"] if family == "e2b" else []),
                       "training", *[f"best_{name}" for name in native.COHORTS], "merge", "export"]
    if plan.get("stages") != [*expected_stages, "native_quality"]:
        raise ValueError("GRPO plan must retain every training/export/native prerequisite")
    completed = manifest.get("completed")
    if not isinstance(completed, dict) or not set(expected_stages).issubset(completed):
        raise ValueError("GRPO native evaluation is missing prerequisite stage receipts")
    if status == "complete" and "native_quality" not in completed:
        raise ValueError("Complete GRPO run has no native quality receipt")
    plan_sha = native.file_sha256(plan_path)
    receipts: dict[str, dict[str, str]] = {}
    for stage in expected_stages:
        receipt = native.read_json(run / "stage_receipts" / f"{stage}.json")
        if receipt != completed[stage] or receipt.get("stage") != stage or receipt.get("plan_sha256") != plan_sha:
            raise ValueError(f"GRPO stage receipt does not match the immutable plan: {stage}")
        files = receipt.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError(f"GRPO stage receipt lacks file identities: {stage}")
        # These are historical input/log identities, not copied run outputs.
        # Permit only exactly declared files; never follow an arbitrary external path.
        allowed_external = set()
        if stage == "assets" and family == "e2b":
            seed = str(values["model_dir"]).rstrip("/\\")
            allowed_external = {str(values["source_safetensors"]), str(values["official_litertlm"]),
                                seed + "/mobile_training_seed_manifest.json", seed + "/mobile_qparams.json"}
        elif stage.startswith("best_"):
            result = native.read_json(run / "evaluations" / stage / "evaluation_result.json")
            allowed_external = {str(result.get("tensorboard_record") or "")}
        normalized_allowed = {value.replace("\\", "/").casefold() for value in allowed_external}
        local = {}
        for value, digest in files.items():
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid file digest in {stage} receipt")
            try:
                relocated(value)
            except ValueError:
                if str(value).replace("\\", "/").casefold() not in normalized_allowed:
                    raise
            else:
                local[value] = digest
        receipts[stage] = native._verified_receipt_files(
            {"files": local}, original_root=original_root, current_root=run, required_prefix=run)

    def require_bound(stage: str, path: Path) -> None:
        if receipts[stage].get(str(path)) != native.file_sha256(path):
            raise ValueError(f"{stage} receipt does not bind {path.name}")

    config = load_yaml(paths["config"])
    validate_qat_grpo_config(config)
    if config["grpo"]["family"] != family:
        raise ValueError("GRPO config and plan disagree on model family")
    require_bound("configure", paths["config"])
    config_sha = native.file_sha256(paths["config"])
    preparation_report = paths["config"].parent / "preparation_report.json"
    require_bound("configure", preparation_report)
    if native.read_json(preparation_report).get("training_config_sha256") != config_sha:
        raise ValueError("Preparation report is bound to a different GRPO config")
    prepared_root, _ = native._validated_prepared_root(plan=plan, config=config, run=run, original_root=original_root)
    if paths["prepared"] != prepared_root:
        raise ValueError("Prepared data differs from the run-local GRPO contract")
    require_bound("prepare", prepared_root / "manifest.json")
    for prepared_file in prepared_root.rglob("*"):
        if prepared_file.is_file():
            require_bound("prepare", prepared_file)
    metadata_path = paths["best_checkpoint"] / "training_metadata.json"
    require_bound("training", metadata_path)
    metadata = native.read_json(metadata_path)
    if (metadata.get("training_config_sha256") != config_sha or metadata.get("checkpoint_role") != "best_golden"
            or (metadata.get("training") or {}).get("method") != "qat_lora_grpo"
            or not _golden_selection_binding(metadata, config)["verified"]
            or (metadata.get("best_golden_eval") or {}).get("metric") != SELECTOR):
        raise ValueError("Selected adapter is not bound to GRPO config and Golden32 selection")
    if (plan.get("selection") or {}) != {"cohort": "golden32", "metric": SELECTOR, "golden35_used": False, "bixby50_used": False}:
        raise ValueError("GRPO checkpoint selection must exclude final-only holdouts")
    actual_adapters = _adapter_file_records(paths["best_checkpoint"])
    if not any(item.get("role") == "best_golden" and _checkpoint_manifest_matches_adapter(
        item.get("files"), adapter_path=paths["best_checkpoint"], actual_adapter_files=actual_adapters)
        for item in metadata.get("adapter_checkpoints", []) if isinstance(item, dict)):
        raise ValueError("Selected adapter files differ from training metadata")
    for item in actual_adapters:
        require_bound("training", paths["best_checkpoint"] / item["path"])
    tokenizer_files = [path for path in paths["best_checkpoint"].iterdir() if path.name.startswith("tokenizer") or path.suffix == ".jinja"]
    if not any(path.name in {"tokenizer.json", "tokenizer.model", "tokenizer_config.json"} for path in tokenizer_files):
        raise ValueError("Selected adapter must retain its saved tokenizer for native prompts")
    for path in tokenizer_files:
        require_bound("training", path)
    merge_path = paths["merged"] / "qat_mtp_merge_metadata.json"
    require_bound("merge", merge_path)
    merge = native.read_json(merge_path)
    run_metadata = merge.get("training_run_metadata") or {}
    if (merge.get("training_method") != "qat_lora_grpo" or merge.get("training_config_sha256") != config_sha
            or relocated(merge.get("adapter_dir")) != paths["best_checkpoint"]
            or relocated(merge.get("merged_model_dir")) != paths["merged"]
            or merge.get("adapter_files") != actual_adapters
            or run_metadata.get("sha256") != native.file_sha256(metadata_path)
            or relocated(run_metadata.get("path")) != metadata_path or run_metadata.get("verified") is not True
            or (run_metadata.get("qat_grpo_provenance") or {}).get("verified") is not True
            or not _all_gates(run_metadata.get("checks"), {"qat_grpo_provenance_verified"})):
        raise ValueError("Merge provenance does not bind the selected GRPO adapter and config")
    require_bound("export", paths["export_report"])
    require_bound("export", paths["litertlm"])
    export = native.read_json(paths["export_report"])
    model_sha = native.file_sha256(paths["litertlm"])
    exported_merge = export.get("merge_provenance") or {}
    if (exported_merge.get("verified") is not True or exported_merge.get("metadata") != merge
            or relocated(export.get("checkpoint")) != paths["merged"]):
        raise ValueError("Official export report does not bind the merged GRPO checkpoint")
    pinned = (OFFICIAL_LITERTLM_SHA256 if family == "e2b" else
              load_yaml(training_root() / "configs/pipelines/gemma3_270m_qat_litertlm.yaml")["pipeline"]["exact_topology"]["official_artifact_sha256"])
    if plan.get("official_artifact_sha256") != pinned:
        raise ValueError("GRPO plan uses a different official family package")
    if family == "e2b":
        identity, config_identity = export.get("adapter_identity") or {}, export.get("resolved_training_config_identity") or {}
        if (export.get("mode") != "retained_scale_qat_compatible_weights_v1" or export.get("passed") is not True
                or export.get("executed") is not True or export.get("plan_passed") is not True
                or export.get("official_artifact_sha256") != pinned or export.get("output_sha256") != model_sha
                or relocated(export.get("output_litertlm")) != paths["litertlm"]
                or relocated(export.get("adapter_checkpoint")) != paths["best_checkpoint"]
                or identity.get("verified") is not True or identity.get("files") != actual_adapters
                or identity.get("training_metadata_sha256") != native.file_sha256(metadata_path)
                or config_identity.get("verified") is not True or config_identity.get("sha256") != config_sha
                or relocated(config_identity.get("path")) != paths["config"]
                or not _all_gates(export.get("gates"), _E2B_GATES)):
            raise ValueError("E2B retained-scale export identities or official-layout gates failed")
        require_bound("no_op_export", paths["no_op_export_report"])
        noop = native.read_json(paths["no_op_export_report"])
        bound = noop.get("resolved_training_config_identity") or {}
        if (noop.get("passed") is not True or noop.get("gate_status") != "PASSED"
                or noop.get("mode") != "retained_scale_pretraining_noop_v1"
                or not _all_gates(noop.get("checks"), NO_OP_CHECKS)
                or bound.get("verified") is not True or bound.get("sha256") != config_sha
                or relocated(bound.get("path")) != paths["config"]):
            raise ValueError("E2B GRPO lacks its config-bound pretraining no-op export")
    else:
        package = export.get("package_boundary") or {}
        if (export.get("family") != "gemma3_270m" or export.get("model_type") != "TF_LITE_PREFILL_DECODE"
                or export.get("official_artifact_sha256_expected") != pinned or export.get("official_artifact_sha256_observed") != pinned
                or export.get("final_artifact_gate_pass") is not True or export.get("exact_official_graph_ops_layout_match") is not True
                or export.get("official_inventory_count") != 127 or export.get("mapping_count") != 127
                or package.get("output_sha256") != model_sha or relocated(package.get("output")) != paths["litertlm"]
                or package.get("bytes_outside_selected_section_verified_unchanged") is not True
                or not _all_gates(export.get("gates"), _270M_GATES)):
            raise ValueError("270M official-topology export identities or inventory gates failed")
    if (plan.get("mtp") or {}).get("training") is not False or (plan.get("mtp") or {}).get("inference") is not False:
        raise ValueError("Native GRPO evaluation requires MTP disabled")
    sequence = int(config["training"]["max_seq_length"])
    max_input, max_new = int(config["golden_eval"]["max_input_tokens"]), int(config["model"]["max_output_tokens"])
    if (sequence != values["max_seq_length"] or max_input != values["max_input_tokens"]
            or max_new != values["max_new_tokens"] or not 0 < max_new <= 4096 or max_input < 1):
        raise ValueError("GRPO native token limits differ from the evaluated config")
    prepared_manifest, cohorts = native.inspect_hf_cohorts(
        prepared_root=prepared_root, run=run, original_root=original_root,
        max_sequence=sequence, max_input_tokens=max_input)
    for name, cohort in cohorts.items():
        stage = f"best_{name}"
        for filename in ("evaluation_result.json", "aggregate_metrics.json", "predictions.jsonl", "scored_predictions.jsonl"):
            require_bound(stage, cohort["hf_dir"] / filename)
        result = cohort["hf_result"]
        aggregate = native.read_json(cohort["hf_dir"] / "aggregate_metrics.json")
        score = aggregate.get(SELECTOR if name == "golden32" else "generation_reward_v5_4_avg")
        if (relocated(result.get("checkpoint")) != paths["best_checkpoint"] or result.get("checkpoint_kind") != "adapter"
                or result.get("aggregate") != aggregate or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError(f"HF {name} does not bind the selected adapter and finite metrics")
        if len(native.read_jsonl_strict(cohort["hf_dir"] / "scored_predictions.jsonl")) != native.COHORTS[name]:
            raise ValueError(f"HF {name} scored coverage is incomplete")
    return {"run_dir": run, "original_root": original_root, "plan_path": plan_path, "plan_sha256": plan_sha,
            "plan": plan, "manifest_path": manifest_path, "paths": paths, "config": config, "model_sha256": model_sha,
            "max_input_tokens": max_input, "max_new_tokens": max_new, "prepared_manifest": prepared_manifest,
            "prepared_manifest_sha256": native.file_sha256(prepared_root / "manifest.json"),
            "cohorts": cohorts, "receipt_bindings": receipts}
