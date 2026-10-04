"""Shared, model-free provenance gates for continued QAT LoRA GRPO.

GRPO starts from a trained SFT adapter. It must never inherit zero-adapter or
SFT-launcher claims. Export recomputes source identities and optimizer health
from the saved evidence while retaining the ordinary QAT/selection checks.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.qat.fake_quant import QATSpec

WORKFLOW = "qat_lora_grpo_v1"
METHOD = "qat_lora_grpo"


def is_qat_grpo(config: dict[str, Any]) -> bool:
    return (config.get("training") or {}).get("method") == METHOD


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def validate_qat_grpo_config(config: dict[str, Any]) -> None:
    model, training = config.get("model") or {}, config.get("training") or {}
    qat, lora, grpo = config.get("qat") or {}, config.get("lora") or {}, config.get("grpo") or {}
    if not is_qat_grpo(config) or (config.get("run") or {}).get("purpose") != WORKFLOW:
        raise ValueError("QAT GRPO requires its explicit method and workflow")
    family = grpo.get("family")
    if family not in {"e2b", "270m"}:
        raise ValueError("grpo.family must be e2b or 270m")
    if qat.get("enabled") is not True or qat.get("effective_merged_weight") is not True:
        raise ValueError("QAT GRPO requires enabled effective merged-weight QAT")
    if qat.get("only_base_layers") is not True or lora.get("modules_to_save"):
        raise ValueError("Official QAT GRPO permits only projection LoRA updates")
    if float(lora.get("dropout", -1)) != 0:
        raise ValueError("Effective merged-weight QAT requires zero LoRA dropout")
    if training.get("resume_from_checkpoint") or training.get("resume_policy"):
        raise ValueError("QAT GRPO starts a new run from SFT adapter weights, not an SFT optimizer resume")
    if not grpo.get("sft_checkpoint") or not grpo.get("sft_training_config"):
        raise ValueError("QAT GRPO requires a bound SFT checkpoint and training config")
    if family == "e2b":
        from ir_training.qat.mobile_training_seed import OFFICIAL_MOBILE_MODEL_ID
        required = {"scale_mode": "retained_mobile", "fixed_scale_required": True,
                    "fixed_activation_scale_required": True, "effective_lora_only": True,
                    "expected_effective_lora_modules": 205, "activation_quantizer": "gemma_mobile_srq",
                    "simulate_frozen_activations": True, "expected_frozen_activation_modules": 70,
                    "require_lora_trainable_scope": True, "quantize_embeddings": False,
                    "ste_gradient": "clipped"}
        if model.get("model_id") != OFFICIAL_MOBILE_MODEL_ID or any(qat.get(k) != v for k, v in required.items()):
            raise ValueError("E2B GRPO requires the official retained-scale 205/70 QAT contract")
    else:
        spec = QATSpec.from_config(config)
        if ("gemma-3-270m" not in str(model.get("model_id", "")).lower()
                or spec.weight_bits != 8 or spec.activation_bits < 16
                or spec.quantizer != "ste_ai_edge" or not spec.quantize_embeddings
                or qat.get("scale_mode") == "retained_mobile"):
            raise ValueError("270M GRPO requires official Q8 weight-only QAT, not E2B mobile topology")


def verify_sft_adapter_lineage(config: dict[str, Any], numeric: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-read the actual SFT bytes; optional numeric evidence proves they loaded."""
    report: dict[str, Any] = {"verified": False, "checks": {}}
    try:
        validate_qat_grpo_config(config)
        grpo = config["grpo"]
        checkpoint = resolve_path(grpo["sft_checkpoint"], training_root()).resolve(strict=True)
        source_path = resolve_path(grpo["sft_training_config"], training_root()).resolve(strict=True)
        source = load_yaml(source_path)
        metadata_path = checkpoint / "training_metadata.json"
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        from ir_training.export.merge_lora import (
            _adapter_file_records,
            _training_provenance,
        )
        if (source.get("training") or {}).get("method") != "qat_lora_sft":
            raise ValueError("GRPO source must be a genuine QAT LoRA SFT checkpoint")
        # This verifies adapter bytes, source config, QAT coverage, and for E2B
        # retained scales/seed/launcher/numeric evidence using existing code.
        provenance = _training_provenance(checkpoint, source_path, base=training_root())
        model, source_model = config.get("model") or {}, source.get("model") or {}
        model_fields = ("model_id", "model_source", "mobile_training_seed_manifest", "mobile_qparams_contract")
        lora_fields = ("r", "alpha", "dropout", "target_modules", "modules_to_save")
        checks = {
            "source_sft_provenance_verified": provenance.get("training_run_metadata", {}).get("verified") is True,
            "source_config_hash_matches": source_metadata.get("training_config_sha256") == file_identity(source_path)["sha256"],
            "same_model_and_seed": all(model.get(k) == source_model.get(k) for k in model_fields),
            "same_qat_spec": QATSpec.from_config(config).to_dict() == QATSpec.from_config(source).to_dict(),
            "same_lora_contract": all((config.get("lora") or {}).get(k) == (source.get("lora") or {}).get(k) for k in lora_fields),
        }
        files = _adapter_file_records(checkpoint)
        weights = [item for item in files if item["path"] in {"adapter_model.safetensors", "adapter_model.bin"}]
        checks["one_adapter_weight_file"] = len(weights) == 1
        expected_pairs = int((source_metadata.get("qat") or {}).get("wrapped_effective_lora_count", 0))
        checks["source_qat_wrappers_present"] = expected_pairs > 0
        if numeric is not None:
            adapter = numeric.get("source_adapter") or {}
            nonzero = adapter.get("nonzero_adapter_pairs")
            checks["trained_adapter_initialization_verified"] = bool(
                numeric.get("adapter_initialization_mode") == "sft_adapter"
                and numeric.get("zero_adapter_initialization") == {
                    "required": False, "reason": "continued_from_trained_sft_adapter", "verified_zero_delta": False}
                and adapter.get("checkpoint") == str(checkpoint)
                and len(weights) == 1 and adapter.get("adapter_sha256") == weights[0]["sha256"]
                and adapter.get("adapter_pair_count") == expected_pairs
                and adapter.get("finite_adapter_pairs") == expected_pairs
                and type(nonzero) is int and 0 < nonzero <= expected_pairs)
        report.update(checkpoint=str(checkpoint), training_config=file_identity(source_path),
                      metadata=file_identity(metadata_path), files=files, checks=checks, verified=all(checks.values()))
    except (OSError, TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
        report["error"] = str(exc)
    return report


def _load_identity(identity: Any) -> dict[str, Any] | None:
    if not isinstance(identity, dict):
        return None
    try:
        actual = file_identity(identity["path"])
        if any(identity.get(k) != v for k, v in actual.items()):
            return None
        result = json.loads(Path(actual["path"]).read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else None
    except (OSError, KeyError, TypeError, ValueError):
        return None


def verify_qat_grpo_provenance(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Recompute lineage, bound preflight and selected-step optimizer health."""
    report: dict[str, Any] = {"verified": False, "checks": {}}
    try:
        validate_qat_grpo_config(config)
        grpo = metadata.get("grpo") or {}
        numeric = metadata.get("numeric_preflight") or {}
        lineage = verify_sft_adapter_lineage(config, numeric)
        initial = verify_sft_adapter_lineage(config)
        preflight = _load_identity(grpo.get("preflight")) or {}
        from ir_training.qat.numeric_preflight import (
            greedy_mandatory_checks,
            numeric_mandatory_checks,
        )

        settings = config.get("preflight") or {}
        greedy = numeric.get("greedy_generation") or {}
        forward_checks = numeric_mandatory_checks(numeric.get("baseline") or {}, numeric.get("qat_on") or {}, settings)
        greedy_checks = greedy_mandatory_checks(greedy.get("baseline") or {}, greedy.get("qat_on") or {}, settings)
        backward = numeric.get("backward") or {}

        def finite(value: Any, *, positive: bool = False) -> bool:
            return bool(not isinstance(value, bool) and isinstance(value, (int, float))
                        and math.isfinite(value) and (value > 0 if positive else value >= 0))

        trainable = backward.get("trainable_tensors")
        nonzero = backward.get("nonzero_gradient_tensors")
        from ir_training.train.qat_grpo_reward import QAT_GRPO_REWARD_VERSION

        reward = grpo.get("reward") or {}
        reward_path = resolve_path(config["grpo"].get("reward_config", "../dataset/configs/genui_metric_v5_4.yaml"), training_root())
        reward_identity = file_identity(reward_path)
        checks = {
            "grpo_schema": grpo.get("schema_version") == 1 and grpo.get("algorithm") == "grpo"
                and grpo.get("family") == config["grpo"]["family"]
                and (metadata.get("training") or {}).get("method") == METHOD,
            "canonical_reward_config_bound": bool(
                config["grpo"].get("reward_policy", QAT_GRPO_REWARD_VERSION) == QAT_GRPO_REWARD_VERSION
                and reward.get("version") == QAT_GRPO_REWARD_VERSION
                and isinstance(reward.get("config"), dict)
                and all(reward["config"].get(key) == value for key, value in reward_identity.items())),
            "source_sft_lineage_verified": lineage["verified"] is True,
            "source_sft_lineage_bound": grpo.get("source_lineage") == lineage,
            "preflight_bytes_bound": bool(preflight),
            "preflight_config_hash_matches": preflight.get("training_config_sha256") == metadata.get("training_config_sha256")
                and bool(metadata.get("training_config_sha256")),
            "preflight_numeric_evidence_matches": preflight.get("numeric_preflight") == numeric,
            "preflight_source_lineage_matches": preflight.get("source_lineage") == initial,
            "qat_forward_probes_passed": numeric.get("passed") is True and numeric.get("qat_enabled") is True
                and all(forward_checks.values()),
            "qat_greedy_probes_passed": greedy.get("passed") is True and greedy.get("qat_enabled") is True
                and all(greedy_checks.values()),
            "qat_backward_probe_passed": bool(
                backward.get("status") == "passed" and backward.get("kind") == "completion_log_likelihood_backward"
                and backward.get("finite_parameters_and_gradients") is True
                and backward.get("optimizer_steps") == 0 and finite(backward.get("loss"))
                and type(trainable) is int and type(nonzero) is int and 0 < nonzero <= trainable
                and finite(backward.get("gradient_l2_norm"), positive=True)
                and finite(backward.get("gradient_max_abs"), positive=True)),
        }
        from ir_training.train.grpo_runtime import GRPOHealthMonitor, HealthThresholds
        thresholds = HealthThresholds(**(config["grpo"].get("health") or {}))
        health = grpo.get("health") or {}
        rows = health.get("window_metrics") or []
        last_step = health.get("last_step")
        valid = bool(type(last_step) is int and last_step > 0
                     and health.get("optimizer_steps") == last_step
                     and metadata.get("checkpoint_step") == last_step
                     and isinstance(rows, list) and len(rows) == thresholds.window_steps
                     and last_step >= len(rows))
        computed: dict[str, Any] = {}
        if valid:
            monitor = GRPOHealthMonitor(thresholds)
            for offset, row in enumerate(rows, last_step - len(rows) + 1):
                if not isinstance(row, dict) or not all(
                    not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                    for value in row.values()
                ):
                    valid = False
                    break
                if any(not 0 <= row.get(key, -1) <= 1 for key in ("frac_reward_zero_std", "completions/clipped_ratio")):
                    valid = False
                    break
                computed = monitor.observe(offset, row)
                if computed["status"] == "failed":
                    valid = False
                    break
        checks["selected_step_health_window_passed"] = bool(
            valid and computed.get("status") == "passed_window" and health.get("passed_window") is True
            and health.get("thresholds") == asdict(thresholds)
            and health.get("window_updates") == len(rows)
            and health.get("aggregates") == computed.get("aggregates")
            and health.get("last_metrics") == computed.get("metrics"))
        report.update(checks=checks, source_lineage=lineage, verified=all(checks.values()))
    except (OSError, TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
        report["error"] = str(exc)
    return report
