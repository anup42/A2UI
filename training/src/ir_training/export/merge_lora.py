from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root
from ir_training.models.hf_loading import load_hf_model
from ir_training.qat.mobile_training_seed import verify_configured_mobile_training_seed
from ir_training.qat.mobile_qparams import verify_mobile_qparams_contract


_RETAINED_MOBILE_PROJECTION_COUNT = 205
_REQUIRED_PORTABLE_PREFLIGHTS = {
    "cuda_bf16_environment",
    "static_qat_profile",
    "mobile_seed_architecture",
    "scale_preserving_qat",
    "model_numeric_preflight",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_file_records(adapter_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": candidate.name,
            "size": int(candidate.stat().st_size),
            "sha256": _sha256_file(candidate),
        }
        for candidate in sorted(adapter_path.glob("adapter*"))
        if candidate.is_file()
    ]


def _normalized_file_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    normalized = []
    for item in records:
        if not isinstance(item, dict):
            continue
        try:
            size = int(item.get("size", -1) or -1)
        except (TypeError, ValueError):
            size = -1
        normalized.append(
            {
                "path": str(item.get("path") or ""),
                "size": size,
                "sha256": str(item.get("sha256") or "").lower(),
            }
        )
    return sorted(normalized, key=lambda item: item["path"])


def _recorded_identity_is_complete(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("present") is not True:
        return False
    digest = str(value.get("sha256") or "").lower()
    try:
        size = int(value.get("size_bytes", -1))
    except (TypeError, ValueError):
        return False
    return bool(str(value.get("path") or "").strip() and size >= 0 and len(digest) == 64)


def _load_bound_json(identity: Any) -> dict[str, Any] | None:
    """Load a launcher artifact only when its recorded size/hash still match."""

    if not _recorded_identity_is_complete(identity):
        return None
    path = Path(str(identity["path"]))
    if not path.is_file():
        return None
    try:
        if path.stat().st_size != int(identity["size_bytes"]):
            return None
        if _sha256_file(path) != str(identity["sha256"]).lower():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _retained_projection_keys(qparams: dict[str, Any]) -> set[str]:
    suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    inventory = qparams.get("inventory")
    if not isinstance(inventory, dict):
        return set()
    return {
        str(key)
        for key, entry in inventory.items()
        if isinstance(entry, dict)
        and str(key).startswith("model.layers.")
        and str(key).endswith(suffixes)
        and entry.get("input_activation_scale_f32_le_hex") is not None
        and entry.get("output_activation_scale_f32_le_hex") is not None
    }


def _retained_binding_contract_matches(
    qat: dict[str, Any], qparams: dict[str, Any]
) -> bool:
    names = qat.get("wrapped_effective_lora_names")
    bindings = qat.get("retained_qparams_bindings")
    inventory = qparams.get("inventory")
    if not isinstance(names, list) or not isinstance(bindings, dict) or not isinstance(inventory, dict):
        return False
    normalized_names = [str(name) for name in names]
    if (
        len(normalized_names) != _RETAINED_MOBILE_PROJECTION_COUNT
        or len(set(normalized_names)) != _RETAINED_MOBILE_PROJECTION_COUNT
        or set(normalized_names) != {str(name) for name in bindings}
    ):
        return False
    bound_weight_keys: set[str] = set()
    for binding in bindings.values():
        if not isinstance(binding, dict):
            return False
        weight_key = str(binding.get("weight_key") or "")
        entry = inventory.get(weight_key)
        if not isinstance(entry, dict):
            return False
        try:
            input_scale = float(binding.get("input_activation_scale"))
            output_scale = float(binding.get("output_activation_scale"))
            bits_match = int(binding.get("bits", -1)) == int(entry.get("bits", -2))
            binding_group = binding.get("group_size")
            entry_group = entry.get("group_size")
            group_match = (
                (binding_group is None and entry_group is None)
                or int(binding_group) == int(entry_group)
            )
            shape_match = [int(value) for value in binding.get("scale_shape", [])] == [
                int(value) for value in entry.get("scale_shape", [])
            ]
        except (TypeError, ValueError):
            return False
        if not (
            bits_match
            and group_match
            and shape_match
            and math.isfinite(input_scale)
            and input_scale > 0
            and math.isfinite(output_scale)
            and output_scale > 0
        ):
            return False
        bound_weight_keys.add(weight_key)
    return bound_weight_keys == _retained_projection_keys(qparams)


def _portable_launcher_contract_matches(
    metadata: dict[str, Any], *, training_config_sha256: str
) -> bool:
    launcher = metadata.get("launcher_provenance")
    if not isinstance(launcher, dict):
        return False
    launch_identity = launcher.get("launch_plan")
    preflight_identity = launcher.get("preflight_report")
    launch_plan = _load_bound_json(launch_identity)
    preflight = _load_bound_json(preflight_identity)
    if launch_plan is None or preflight is None:
        return False
    bound = launch_plan.get("bound_artifacts")
    bound = bound if isinstance(bound, dict) else {}
    resolved_config = bound.get("resolved_training_config")
    if not isinstance(resolved_config, dict):
        return False
    try:
        config_hash_matches = (
            str(resolved_config.get("sha256") or "").lower()
            == training_config_sha256.lower()
            and int(resolved_config.get("size_bytes", -1)) >= 1
        )
    except (TypeError, ValueError):
        config_hash_matches = False
    # New launch plans bind an explicitly sized, exact Golden contract under
    # ``golden_eval``.  Keep accepting ``golden100`` so already trained,
    # provenance-bound checkpoints remain mergeable.
    golden_role = (
        "golden_eval"
        if isinstance(bound.get("golden_eval"), dict)
        else "golden100"
    )
    required_bound = {
        "mobile_seed_manifest",
        "mobile_qparams_contract",
        "official_packed_source",
        "training_train",
        "training_val",
        golden_role,
        "resolved_training_config",
    }
    launch_golden_contract = launch_plan.get("golden_eval_contract")
    if (
        golden_role == "golden_eval"
        and isinstance(launch_golden_contract, dict)
        and str(launch_golden_contract.get("source_genui_sha256") or "").strip()
    ):
        required_bound.add("golden_source_genui")
    if (
        golden_role == "golden_eval"
        and isinstance(launch_golden_contract, dict)
        and str(launch_golden_contract.get("source_responses_sha256") or "").strip()
    ):
        required_bound.add("golden_source_responses")
    if (
        golden_role == "golden_eval"
        and isinstance(launch_golden_contract, dict)
        and launch_golden_contract.get("dataset_config_bound") is True
    ):
        required_bound.add("golden_dataset_config")

    def bound_identity_complete(role: str) -> bool:
        identity = bound.get(role)
        if not isinstance(identity, dict):
            return False
        try:
            size = int(identity.get("size_bytes", -1))
        except (TypeError, ValueError):
            return False
        return len(str(identity.get("sha256") or "")) == 64 and size >= 0

    bound_complete = required_bound.issubset(bound) and all(
        bound_identity_complete(role) for role in required_bound
    )
    reports = preflight.get("reports")
    reports = reports if isinstance(reports, list) else []
    report_ids = {
        str(item.get("id"))
        for item in reports
        if isinstance(item, dict) and item.get("passed") is True
    }
    gate_records_bound = all(
        isinstance(item, dict)
        and item.get("passed") is True
        and _recorded_identity_is_complete(item.get("log"))
        and all(
            _recorded_identity_is_complete(output)
            for output in (item.get("declared_outputs") or [])
        )
        for item in reports
    )
    golden_contract_ok = golden_role == "golden100"
    if golden_role == "golden_eval":
        golden_contract = launch_golden_contract
        configured_golden = metadata.get("golden_eval")
        if not isinstance(golden_contract, dict):
            golden_contract = {}
        if not isinstance(configured_golden, dict):
            configured_golden = {}
        required_rows = golden_contract.get("required_rows")
        bound_golden = bound.get(golden_role)
        expected_source_sha256 = str(
            golden_contract.get("source_genui_sha256") or ""
        ).strip().lower()
        expected_responses_sha256 = str(
            golden_contract.get("source_responses_sha256") or ""
        ).strip().lower()
        bound_source = bound.get("golden_source_genui")
        bound_responses = bound.get("golden_source_responses")
        source_contract_ok = bool(
            (
                not expected_source_sha256
                or (
                    isinstance(bound_source, dict)
                    and str(bound_source.get("sha256") or "").lower()
                    == expected_source_sha256
                )
            )
            and (
                not expected_responses_sha256
                or (
                    isinstance(bound_responses, dict)
                    and str(bound_responses.get("sha256") or "").lower()
                    == expected_responses_sha256
                )
            )
        )
        golden_contract_ok = bool(
            type(required_rows) is int
            and required_rows > 0
            and golden_contract.get("artifact_role") == golden_role
            and golden_contract.get("max_rows") == required_rows
            and golden_contract.get("require_exact_rows") is True
            and golden_contract.get("require_unique_rows") is True
            and golden_contract.get("metric_for_best_model")
            == "generation_reward_v5_4_avg"
            and configured_golden.get("required_rows") == required_rows
            and configured_golden.get("max_rows") == required_rows
            and configured_golden.get("require_exact_rows") is True
            and configured_golden.get("require_unique_rows") is True
            and configured_golden.get("metric_for_best_model")
            == "generation_reward_v5_4_avg"
            and isinstance(bound_golden, dict)
            and len(str(bound_golden.get("sha256") or "")) == 64
            and source_contract_ok
        )
    return bool(
        launch_plan.get("mode") == "fresh_run_only_no_resume"
        and launch_plan.get("run_id") == metadata.get("run_id")
        and isinstance(launch_plan.get("checks"), dict)
        and launch_plan["checks"].get("contract_ok") is True
        and config_hash_matches
        and bound_complete
        and golden_contract_ok
        and preflight.get("run_id") == metadata.get("run_id")
        and preflight.get("all_passed") is True
        and report_ids == _REQUIRED_PORTABLE_PREFLIGHTS
        and gate_records_bound
    )


def _verify_qat_training_metadata(
    adapter_path: Path,
    *,
    training_config_sha256: str,
    training_method: str,
    mobile_training_seed: dict[str, Any],
    training_config: dict[str, Any] | None = None,
    mobile_qparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_path = next(
        (
            candidate
            for candidate in (
                adapter_path / "training_metadata.json",
                adapter_path.parent / "training_metadata.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    checks = {
        "metadata_present": metadata_path is not None,
        "metadata_v2_or_newer": False,
        "training_config_hash_matches": False,
        "training_method_matches": False,
        "qat_enabled": False,
        "effective_merged_weight_qat": False,
        "effective_lora_wrappers_recorded": False,
        "all_lora_adapter_linears_covered": False,
        "zero_lora_dropout": False,
        "git_commit_recorded": False,
        "adapter_checkpoint_hashes_match": False,
        "mobile_training_seed_matches": not mobile_training_seed.get(
            "required", False
        ),
        "mobile_seed_metadata_v4": not mobile_training_seed.get(
            "required", False
        ),
        "mobile_seed_architecture_verified": not mobile_training_seed.get(
            "required", False
        ),
    }
    retained_required = bool(
        isinstance(training_config, dict)
        and isinstance(training_config.get("qat"), dict)
        and training_config["qat"].get("scale_mode") == "retained_mobile"
    )
    if retained_required:
        checks.update(
            {
                "retained_metadata_v4": False,
                "best_golden_checkpoint_role": False,
                "best_golden_v5_4_selected": False,
                "retained_mobile_qat_spec": False,
                "exact_205_effective_lora_modules": False,
                "exact_205_retained_qparams_bindings": False,
                "retained_qparams_identity_matches": False,
                "zero_adapter_initialization_verified": False,
                "initial_numeric_parity_passed": False,
                "deterministic_greedy_prefix_passed": False,
                "portable_launcher_artifacts_bound": False,
            }
        )
    report: dict[str, Any] = {
        "required": True,
        "path": str(metadata_path) if metadata_path else None,
        "sha256": _sha256_file(metadata_path) if metadata_path else None,
        "checks": checks,
        "verified": False,
    }
    if metadata_path is None:
        return report
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["error"] = f"Could not read training metadata: {exc}"
        return report
    if not isinstance(metadata, dict):
        report["error"] = "Training metadata root is not an object."
        return report
    qat = metadata.get("qat") if isinstance(metadata.get("qat"), dict) else {}
    qat_spec = qat.get("spec") if isinstance(qat.get("spec"), dict) else {}
    lora = metadata.get("lora") if isinstance(metadata.get("lora"), dict) else {}
    training = (
        metadata.get("training")
        if isinstance(metadata.get("training"), dict)
        else {}
    )
    checks["metadata_v2_or_newer"] = (
        int(metadata.get("training_metadata_version", 0) or 0) >= 2
    )
    checks["training_config_hash_matches"] = (
        str(metadata.get("training_config_sha256") or "").lower()
        == training_config_sha256.lower()
    )
    checks["training_method_matches"] = (
        str(training.get("method") or "") == training_method
    )
    checks["qat_enabled"] = qat.get("enabled") is True
    checks["effective_merged_weight_qat"] = bool(
        qat.get("effective_merged_weight_qat_enabled") is True
        and qat_spec.get("effective_merged_weight") is True
    )
    checks["effective_lora_wrappers_recorded"] = (
        int(qat.get("wrapped_effective_lora_count", 0) or 0) > 0
    )
    checks["all_lora_adapter_linears_covered"] = not bool(
        qat.get("uncovered_lora_adapter_linear_names")
    )
    try:
        checks["zero_lora_dropout"] = float(lora.get("dropout", -1.0)) == 0.0
    except (TypeError, ValueError):
        checks["zero_lora_dropout"] = False
    git_commit = str(metadata.get("git_commit") or "").strip().lower()
    checks["git_commit_recorded"] = bool(git_commit and git_commit != "unknown")
    actual_files = _normalized_file_records(_adapter_file_records(adapter_path))
    checkpoints = metadata.get("adapter_checkpoints")
    if isinstance(checkpoints, list):
        checks["adapter_checkpoint_hashes_match"] = any(
            isinstance(checkpoint, dict)
            and _normalized_file_records(checkpoint.get("files")) == actual_files
            for checkpoint in checkpoints
        )
    recorded_seed = (
        metadata.get("mobile_training_seed")
        if isinstance(metadata.get("mobile_training_seed"), dict)
        else {}
    )
    if mobile_training_seed.get("required", False):
        checks["mobile_seed_metadata_v4"] = (
            int(metadata.get("training_metadata_version", 0) or 0) >= 4
        )
        checks["mobile_training_seed_matches"] = bool(
            recorded_seed.get("verified") is True
            and str(recorded_seed.get("manifest_sha256") or "").lower()
            == str(mobile_training_seed.get("manifest_sha256") or "").lower()
            and str(recorded_seed.get("transformation_plan_sha256") or "").lower()
            == str(
                mobile_training_seed.get("transformation_plan_sha256") or ""
            ).lower()
        )
        architecture = (
            metadata.get("mobile_seed_architecture")
            if isinstance(metadata.get("mobile_seed_architecture"), dict)
            else {}
        )
        comparison = (
            architecture.get("comparison")
            if isinstance(architecture.get("comparison"), dict)
            else {}
        )
        architecture_checks = (
            architecture.get("checks")
            if isinstance(architecture.get("checks"), dict)
            else {}
        )
        checks["mobile_seed_architecture_verified"] = bool(
            architecture.get("verified") is True
            and architecture.get("model_class") == "Gemma4ForCausalLM"
            and str(architecture.get("seed_manifest_sha256") or "").lower()
            == str(mobile_training_seed.get("manifest_sha256") or "").lower()
            and comparison.get("exact") is True
            and comparison.get("checkpoint_inventory_sha256")
            == comparison.get("framework_inventory_sha256")
            and architecture_checks.get("key_and_shape_inventory_exact") is True
            and architecture_checks.get("framework_state_on_meta") is True
            and architecture.get("model_weights_loaded") is False
            and architecture.get("forward_executed") is False
            and architecture.get("training_executed") is False
        )
    if retained_required:
        config_qat = training_config.get("qat", {})  # type: ignore[union-attr]
        config_preflight = training_config.get("preflight", {})  # type: ignore[union-attr]
        config_golden = training_config.get("golden_eval", {})  # type: ignore[union-attr]
        config_qat = config_qat if isinstance(config_qat, dict) else {}
        config_preflight = (
            config_preflight if isinstance(config_preflight, dict) else {}
        )
        config_golden = config_golden if isinstance(config_golden, dict) else {}
        retained = (
            qat.get("retained_qparams")
            if isinstance(qat.get("retained_qparams"), dict)
            else {}
        )
        numeric = (
            metadata.get("numeric_preflight")
            if isinstance(metadata.get("numeric_preflight"), dict)
            else {}
        )
        zero_adapter = (
            numeric.get("zero_adapter_initialization")
            if isinstance(numeric.get("zero_adapter_initialization"), dict)
            else {}
        )
        greedy = (
            numeric.get("greedy_generation")
            if isinstance(numeric.get("greedy_generation"), dict)
            else {}
        )
        golden = (
            metadata.get("best_golden_eval")
            if isinstance(metadata.get("best_golden_eval"), dict)
            else {}
        )
        expected_count = int(
            config_qat.get(
                "expected_effective_lora_modules",
                _RETAINED_MOBILE_PROJECTION_COUNT,
            )
            or 0
        )
        minimum_prefix = int(
            config_preflight.get("min_baseline_qat_greedy_prefix_tokens", 8)
            or 0
        )
        minimum_top1 = float(config_preflight.get("min_top1_probe_match", 0.90))
        checks["retained_metadata_v4"] = (
            int(metadata.get("training_metadata_version", 0) or 0) >= 4
        )
        checks["best_golden_checkpoint_role"] = bool(
            metadata.get("checkpoint_role") == "best_golden"
            and isinstance(metadata.get("adapter_checkpoints"), list)
            and any(
                isinstance(checkpoint, dict)
                and checkpoint.get("role") == "best_golden"
                and _normalized_file_records(checkpoint.get("files"))
                == actual_files
                for checkpoint in metadata["adapter_checkpoints"]
            )
        )
        try:
            golden_metric_value = float(golden.get("metric_value"))
        except (TypeError, ValueError):
            golden_metric_value = math.nan
        try:
            golden_step_matches = int(golden.get("step", -1)) == int(
                metadata.get("checkpoint_step", -2)
            )
        except (TypeError, ValueError):
            golden_step_matches = False
        checks["best_golden_v5_4_selected"] = bool(
            golden.get("metric") == "generation_reward_v5_4_avg"
            and config_golden.get("metric_for_best_model")
            == "generation_reward_v5_4_avg"
            and math.isfinite(golden_metric_value)
            and golden_step_matches
        )
        checks["retained_mobile_qat_spec"] = bool(
            qat_spec.get("scale_mode") == "retained_mobile"
            and qat_spec.get("fixed_scale_required") is True
            and qat_spec.get("fixed_activation_scale_required") is True
            and qat_spec.get("effective_lora_only") is True
            and qat_spec.get("effective_merged_weight") is True
            and qat_spec.get("ste_gradient") == "clipped"
            and int(qat_spec.get("expected_effective_lora_modules", 0) or 0)
            == expected_count
            and expected_count == _RETAINED_MOBILE_PROJECTION_COUNT
        )
        checks["exact_205_effective_lora_modules"] = bool(
            int(qat.get("wrapped_effective_lora_count", 0) or 0)
            == _RETAINED_MOBILE_PROJECTION_COUNT
            and not qat.get("uncovered_lora_adapter_linear_names")
        )
        qparams_report = mobile_qparams if isinstance(mobile_qparams, dict) else {}
        checks["exact_205_retained_qparams_bindings"] = bool(
            qparams_report.get("verified") is True
            and int(qat.get("retained_qparams_binding_count", 0) or 0)
            == _RETAINED_MOBILE_PROJECTION_COUNT
            and _retained_binding_contract_matches(qat, qparams_report)
        )
        checks["retained_qparams_identity_matches"] = bool(
            retained.get("verified") is True
            and retained.get("mode") == "retained_mobile"
            and str(retained.get("contract_sha256") or "").lower()
            == str(qparams_report.get("contract_sha256") or "").lower()
            and str(retained.get("scale_storage_sha256") or "").lower()
            == str(qparams_report.get("scale_storage_sha256") or "").lower()
            and retained.get("inventory_sha256")
            == qparams_report.get("inventory_sha256")
            and int(retained.get("tensor_count", 0) or 0)
            == int(qparams_report.get("tensor_count", -1) or -1)
        )
        checks["zero_adapter_initialization_verified"] = bool(
            zero_adapter.get("verified_zero_delta") is True
            and int(zero_adapter.get("wrapper_count", 0) or 0)
            == _RETAINED_MOBILE_PROJECTION_COUNT
            and int(zero_adapter.get("adapter_pair_count", 0) or 0)
            >= _RETAINED_MOBILE_PROJECTION_COUNT
            and not zero_adapter.get("nonzero_or_invalid_pairs")
        )
        try:
            top1_fraction = float(numeric.get("top1_probe_match_fraction"))
        except (TypeError, ValueError):
            top1_fraction = math.nan
        checks["initial_numeric_parity_passed"] = bool(
            numeric.get("passed") is True
            and numeric.get("qat_enabled") is True
            and math.isfinite(top1_fraction)
            and top1_fraction >= minimum_top1
        )
        try:
            observed_prefix = int(
                greedy.get("baseline_qat_min_common_prefix_tokens", -1)
            )
        except (TypeError, ValueError):
            observed_prefix = -1
        checks["deterministic_greedy_prefix_passed"] = bool(
            greedy.get("passed") is True
            and greedy.get("qat_enabled") is True
            and greedy.get("qat_greedy_deterministic") is True
            and observed_prefix >= minimum_prefix
            and minimum_prefix >= 8
        )
        checks["portable_launcher_artifacts_bound"] = (
            _portable_launcher_contract_matches(
                metadata, training_config_sha256=training_config_sha256
            )
        )
    report["training_git_commit"] = git_commit or None
    report["adapter_files"] = actual_files
    report["verified"] = bool(all(checks.values()))
    return report


def _recorded_qat_evidence(adapter_path: Path) -> list[str]:
    """Inspect saved evidence before accepting a caller's replacement recipe."""
    from ir_training.common.config import load_yaml

    evidence: list[str] = []
    candidates = {
        adapter_path / "training_metadata.json", adapter_path.parent / "training_metadata.json",
        adapter_path / "training_config.yaml", adapter_path.parent / "config.yaml",
    }
    for path in sorted(candidates):
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else load_yaml(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unreadable saved training provenance: {path}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid saved training provenance: {path}")
        qat = record.get("qat") or {}
        method = str((record.get("training") or {}).get("method", "")).lower()
        if (isinstance(qat, dict) and qat.get("enabled") is True) or method in {"qat_lora_sft", "full_finetune_qat"}:
            evidence.append(str(path))
    return evidence


def _training_provenance(
    adapter_path: Path,
    training_config_path: str | Path | None,
    *,
    base: Path,
    base_model_source: str | Path | None = None,
    mobile_training_seed_manifest: str | Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "training_config": None,
        "training_config_sha256": None,
        "training_method": None,
        "qat_enabled": None,
        "qat_profile": None,
        "qat_effective_merged_weight": None,
        "lora_dropout": None,
        "mobile_training_seed": {
            "required": False,
            "verified": True,
        },
        "mobile_qparams": {
            "required": False,
            "verified": True,
        },
        "training_run_metadata": {
            "required": False,
            "verified": False,
        },
    }
    recorded_qat = _recorded_qat_evidence(adapter_path)
    if training_config_path is None:
        if recorded_qat:
            raise ValueError("Saved checkpoint records QAT; supply its original training config for provenance verification.")
        return result
    from ir_training.common.config import load_yaml

    resolved_config = resolve_path(training_config_path, base)
    if not resolved_config.is_file():
        raise FileNotFoundError(
            f"Missing training config for merge provenance: {resolved_config}"
        )
    config_bytes = resolved_config.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = load_yaml(resolved_config)
    if recorded_qat and (config.get("qat") or {}).get("enabled") is not True:
        raise ValueError("Saved checkpoint records QAT; refusing to relabel it as ordinary LoRA and bypass QAT provenance checks.")
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    model = dict(model)
    configured_source = model.get("model_source")
    if base_model_source is not None:
        supplied_source = resolve_path(base_model_source, base)
        expected_source = (
            resolve_path(configured_source, base) if configured_source else None
        )
        if expected_source is None or supplied_source != expected_source:
            raise ValueError(
                "Merge base_model_source does not match model.model_source in the "
                "training config."
            )
        model["model_source"] = str(supplied_source)
    configured_manifest = model.get("mobile_training_seed_manifest")
    if mobile_training_seed_manifest is not None:
        supplied_manifest = resolve_path(mobile_training_seed_manifest, base)
        expected_manifest = (
            resolve_path(configured_manifest, base) if configured_manifest else None
        )
        if expected_manifest is None or supplied_manifest != expected_manifest:
            raise ValueError(
                "Merge mobile_training_seed_manifest does not match the training config."
            )
        model["mobile_training_seed_manifest"] = str(supplied_manifest)
    mobile_training_seed = verify_configured_mobile_training_seed(
        model,
        base=base,
        require_materialized=True,
    )
    if not mobile_training_seed["verified"]:
        failed = [
            name
            for name, passed in mobile_training_seed.get("checks", {}).items()
            if not passed
        ]
        raise ValueError(
            "Merge mobile training-seed identity is incomplete or mismatched: "
            + ", ".join(failed)
        )
    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    qat = config.get("qat") if isinstance(config.get("qat"), dict) else {}
    lora = config.get("lora") if isinstance(config.get("lora"), dict) else {}
    mobile_qparams: dict[str, Any] = {"required": False, "verified": True}
    if qat.get("scale_mode") == "retained_mobile":
        mobile_qparams = verify_mobile_qparams_contract(
            model.get("mobile_qparams_contract"), base=base
        )
        if not mobile_qparams.get("verified"):
            failed = [
                name
                for name, passed in mobile_qparams.get("checks", {}).items()
                if not passed
            ]
            raise ValueError(
                "Merge retained mobile qparams identity is incomplete or mismatched: "
                + ", ".join(failed)
            )
    training_method = str(training.get("method") or "")
    qat_enabled = bool(qat.get("enabled", False))
    result.update(
        {
            "training_config": str(resolved_config),
            "training_config_sha256": config_sha256,
            "training_method": training_method,
            "qat_enabled": qat_enabled,
            "qat_profile": qat.get("profile"),
            "qat_effective_merged_weight": bool(
                qat.get("effective_merged_weight", False)
            ),
            "lora_dropout": lora.get("dropout"),
            "mobile_training_seed": mobile_training_seed,
            "mobile_qparams": mobile_qparams,
        }
    )
    if qat_enabled:
        run_report = _verify_qat_training_metadata(
            adapter_path,
            training_config_sha256=config_sha256,
            training_method=training_method,
            mobile_training_seed=mobile_training_seed,
            training_config=config,
            mobile_qparams=mobile_qparams,
        )
        result["training_run_metadata"] = run_report
        if not run_report["verified"]:
            failed = [
                name
                for name, passed in run_report["checks"].items()
                if not passed
            ]
            raise ValueError(
                "QAT adapter training provenance is incomplete or mismatched: "
                + ", ".join(failed)
            )
    return result


def merge_lora_adapter(
    base_model_id: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    model_loader: str = "auto_causal_lm",
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    processor_model_id: str | None = None,
    training_config_path: str | Path | None = None,
    base_model_source: str | Path | None = None,
    mobile_training_seed_manifest: str | Path | None = None,
) -> Path:
    """Merge an adapter while recording what this operation does not prove.

    A merge from a QAT-trained adapter remains a floating-point Hugging Face
    model. It is not packed INT4; this function verifies the training-time QAT
    provenance but does not itself run QAT or modify an MTP assistant.
    """

    if not str(base_model_id).strip():
        raise ValueError("base_model_id is required")
    if not str(adapter_dir).strip():
        raise ValueError("adapter_dir is required")
    if not str(output_dir).strip():
        raise ValueError("output_dir is required")

    base = training_root()
    adapter_path = resolve_path(adapter_dir, base)
    out_dir = resolve_path(output_dir, base)
    model_source = (
        resolve_path(base_model_source, base)
        if base_model_source is not None
        else str(base_model_id)
    )
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter directory: {adapter_path}")
    training_provenance = _training_provenance(
        adapter_path,
        training_config_path,
        base=base,
        base_model_source=base_model_source,
        mobile_training_seed_manifest=mobile_training_seed_manifest,
    )
    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoProcessor, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError(
            "Install the training requirements before merging LoRA adapters."
        ) from exc
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config: dict[str, Any] = {
        "model_loader": model_loader,
        "dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
        "load_in_4bit": False,
        "require_exact_checkpoint_keys": bool(
            training_provenance["mobile_training_seed"].get("required", False)
        ),
    }
    model = load_hf_model(str(model_source), model_config)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)

    processor_saved = False
    try:
        processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=trust_remote_code)
        processor.save_pretrained(str(out_dir))
        processor_saved = True
    except Exception:  # noqa: BLE001 - adapter processor metadata is optional
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=trust_remote_code)
            tokenizer.save_pretrained(str(out_dir))
            processor_saved = True
        except Exception:  # noqa: BLE001 - fall back to the declared base below
            processor_saved = False
    if not processor_saved:
        processor_source = processor_model_id or base_model_id
        try:
            processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=trust_remote_code)
            processor.save_pretrained(str(out_dir))
        except Exception:  # noqa: BLE001 - some model families expose only a tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
            tokenizer.save_pretrained(str(out_dir))

    adapter_files = _adapter_file_records(adapter_path)

    merged_model_files = []
    merged_candidates = list(out_dir.glob("model*.safetensors"))
    index_path = out_dir / "model.safetensors.index.json"
    if index_path.is_file():
        merged_candidates.append(index_path)
    for candidate in sorted(merged_candidates):
        if not candidate.is_file():
            continue
        merged_model_files.append(
            {
                "path": candidate.name,
                "size": int(candidate.stat().st_size),
                "sha256": _sha256_file(candidate),
            }
        )
    if not merged_model_files:
        raise RuntimeError(
            "Merged model did not produce model*.safetensors despite safe serialization."
        )

    metadata = {
        "manifest_version": 4,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": base_model_id,
        "base_model_source": str(model_source),
        "adapter_dir": str(adapter_path),
        "merged_model_dir": str(out_dir),
        "model_loader": model_loader,
        "merge_dtype": dtype,
        "exact_checkpoint_keys_required": bool(
            training_provenance["mobile_training_seed"].get("required", False)
        ),
        "base_is_qat_derived": "-qat-" in base_model_id.lower(),
        "continued_qat_performed": bool(training_provenance["qat_enabled"]),
        "merge_performed_qat": False,
        "packed_int4_output": False,
        "requires_post_merge_quantization": True,
        "mtp_assistant_trained_or_modified": False,
        "adapter_files": adapter_files,
        "merged_model_files": merged_model_files,
        **training_provenance,
    }
    (out_dir / "qat_mtp_merge_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_dir
