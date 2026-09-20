"""Explicit numeric-preflight policy shared by training and export provenance.

Cross-mode similarity is not a correctness oracle for retained W2/W4 + A8 SRQ.
Only the official v2 workflow opts into diagnostic comparisons; old configs
keep their parity gates. Neither policy can substitute for seed/qparam/scope
verification, which remains independently mandatory.
"""
from __future__ import annotations

import math
from typing import Any

from ir_training.qat.full_model_contract import NUMERIC_POLICY as FULL_QAT_POLICY
from ir_training.qat.full_model_contract import is_full_qat

OFFICIAL_MOBILE_WORKFLOW = "e2b_retained_mobile_golden_bixby_no_mtp_v2"
LEGACY_POLICY = "legacy_bf16_parity_v1"
RETAINED_MOBILE_POLICY = "retained_mobile_safety_v1"


def numeric_policy(preflight: dict[str, Any]) -> str:
    policy = preflight.get("numeric_policy", LEGACY_POLICY)
    if policy not in (LEGACY_POLICY, RETAINED_MOBILE_POLICY, FULL_QAT_POLICY):
        raise ValueError(f"Unknown preflight.numeric_policy: {policy!r}")
    return policy


def resolve_numeric_policy(config: dict[str, Any]) -> str:
    """Fail closed on missing v2 policy, unrelated opt-ins or disabled gates."""
    preflight = config.get("preflight") or {}
    if not isinstance(preflight, dict):
        raise TypeError("preflight must be an object")
    policy = numeric_policy(preflight)
    if is_full_qat(config) or policy == FULL_QAT_POLICY:
        from ir_training.qat.full_model_contract import validate_full_qat_config

        validate_full_qat_config(config)
        return policy
    official = (config.get("run") or {}).get("purpose") == OFFICIAL_MOBILE_WORKFLOW
    if official != (policy == RETAINED_MOBILE_POLICY):
        raise ValueError(
            f"{OFFICIAL_MOBILE_WORKFLOW} requires explicit "
            f"preflight.numeric_policy: {RETAINED_MOBILE_POLICY}; "
            "this diagnostic policy is not available to other workflows."
        )
    if not official:
        return policy
    qat = config.get("qat") or {}
    required = {
        "enabled": True, "scale_mode": "retained_mobile",
        "fixed_scale_required": True, "fixed_activation_scale_required": True,
        "effective_merged_weight": True, "effective_lora_only": True,
        "expected_effective_lora_modules": 205,
        "activation_quantizer": "gemma_mobile_srq",
        "simulate_frozen_activations": True, "expected_frozen_activation_modules": 70,
        "require_lora_trainable_scope": True,
    }
    for key, value in required.items():
        if type(qat.get(key)) is not type(value) or qat.get(key) != value:
            raise ValueError(f"{policy} requires qat.{key}: {value}")
    for key in ("require_zero_adapter_parity", "require_initial_loss_gate",
                "require_greedy_determinism"):
        if preflight.get(key) is not True:
            raise ValueError(f"{policy} requires preflight.{key}: true")
    absolute = _finite_number(preflight.get("max_initial_completion_loss"))
    if absolute is None or not 0 < absolute <= 15.0:
        raise ValueError("Official numeric policy requires a finite absolute QAT loss limit in (0, 15]")
    for key, minimum in (("rows", 1), ("logit_probe_tokens", 1),
                         ("greedy_probe_rows", 1), ("greedy_probe_new_tokens", 8),
                         ("min_greedy_tokens", 8)):
        if not _integer_at_least(preflight.get(key), minimum):
            raise ValueError(f"Official numeric policy requires preflight.{key} >= {minimum}")
    if preflight["min_greedy_tokens"] > preflight["greedy_probe_new_tokens"]:
        raise ValueError("min_greedy_tokens exceeds greedy_probe_new_tokens")
    return policy


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _integer_at_least(value: Any, minimum: int) -> bool:
    return type(value) is int and value >= minimum


def numeric_mandatory_checks(
    baseline: dict, qat_on: dict, preflight: dict,
) -> dict[str, bool]:
    """Recompute gates from probe evidence, not an untrusted 'passed' flag."""
    reports = (baseline, qat_on)
    losses_finite = all(
        _finite_number(item.get("completion_loss")) is not None
        and item["completion_loss"] >= 0
        and isinstance(item.get("row_losses"), list)
        and bool(item["row_losses"])
        and all(_finite_number(loss) is not None and loss >= 0 for loss in item["row_losses"])
        for item in reports
    )
    probes_valid = all(
        _integer_at_least(item.get("rows_checked"), 1)
        and _integer_at_least(item.get("completion_tokens"), 1)
        and isinstance(item.get("row_losses"), list)
        and len(item["row_losses"]) == item["rows_checked"]
        and isinstance(item.get("top1_probe_ids"), list)
        and bool(item["top1_probe_ids"])
        and all(_integer_at_least(token, 0) for token in item["top1_probe_ids"])
        for item in reports
    )
    comparable = bool(
        probes_valid
        and baseline["rows_checked"] == qat_on["rows_checked"]
        and baseline["completion_tokens"] == qat_on["completion_tokens"]
        and len(baseline["top1_probe_ids"]) == len(qat_on["top1_probe_ids"])
    )
    absolute = _finite_number(preflight.get("max_initial_completion_loss"))
    qat_loss = _finite_number(qat_on.get("completion_loss"))
    return {
        "finite_completion_losses": losses_finite,
        "finite_logits": all(item.get("logits_finite") is True for item in reports),
        "comparable_numeric_probes": comparable,
        "absolute_qat_loss_bounded": bool(
            absolute is not None and 0 < absolute <= 15.0
            and qat_loss is not None and 0 <= qat_loss <= absolute
        ),
    }


def _valid_greedy_probe(report: dict, preflight: dict, *, min_repeats: int) -> bool:
    runs = report.get("generated_token_ids")
    rows = report.get("rows_checked")
    minimum = preflight["min_greedy_tokens"]
    maximum = preflight["greedy_probe_new_tokens"]
    if not (
        isinstance(runs, list) and len(runs) >= min_repeats
        and report.get("repeats") == len(runs)
        and _integer_at_least(rows, 1) and rows <= preflight["greedy_probe_rows"]
        and report.get("min_new_tokens") == minimum
        and report.get("max_new_tokens") == maximum
    ):
        return False
    diagnostics = report.get("generation_diagnostics") or []
    for repeat, run in enumerate(runs):
        if not isinstance(run, list) or len(run) != rows:
            return False
        for row, sequence in enumerate(run):
            if not (
                isinstance(sequence, list) and 0 < len(sequence) <= maximum
                and all(_integer_at_least(token, 0) for token in sequence)
            ):
                return False
            if len(sequence) < minimum:
                try:
                    stop = diagnostics[repeat][row]["stop_reason"]
                except (IndexError, KeyError, TypeError):
                    return False
                if stop not in {"closing_sentinel", "eos_token"}:
                    return False
    return report.get("generated_token_counts") == [
        [len(sequence) for sequence in run] for run in runs
    ]


def greedy_mandatory_checks(baseline: dict, qat_on: dict, preflight: dict) -> dict[str, bool]:
    valid = (
        _valid_greedy_probe(baseline, preflight, min_repeats=1)
        and _valid_greedy_probe(qat_on, preflight, min_repeats=2)
        and baseline["rows_checked"] == qat_on["rows_checked"]
    )
    runs = qat_on.get("generated_token_ids") or []
    return {
        "valid_repeated_greedy_probes": bool(valid),
        "qat_greedy_deterministic": bool(
            valid and qat_on.get("deterministic") is True
            and all(run == runs[0] for run in runs[1:])
        ),
    }


def _successful_gates_match(recorded: Any, computed: dict[str, bool]) -> bool:
    return bool(
        isinstance(recorded, dict) and recorded.keys() == computed.keys()
        and all(recorded[key] is True and computed[key] is True for key in computed)
    )


def numeric_preflight_provenance(config: dict, numeric: dict) -> dict[str, Any]:
    """Validate new v2 evidence while leaving historical metadata semantics alone.

    The merge/export callers still verify exact 205/70 scope, qparams, hashes,
    seed, launcher and package identity separately. A new policy cannot bless
    old metadata that merely recorded 'passed: true'.
    """
    report: dict[str, Any] = {"verified": False, "checks": {}}
    try:
        policy = resolve_numeric_policy(config)
        report["policy"] = policy
        # A report cannot opt a legacy config into the diagnostic policy either.
        if numeric.get("policy", LEGACY_POLICY) != policy:
            raise ValueError("Numeric preflight policy does not match the bound training config")
        if policy == LEGACY_POLICY:
            report["verified"] = True
            return report
        preflight = config["preflight"]
        numeric_checks = numeric_mandatory_checks(
            numeric.get("baseline") or {}, numeric.get("qat_on") or {}, preflight,
        )
        greedy = numeric.get("greedy_generation") or {}
        greedy_checks = greedy_mandatory_checks(
            greedy.get("baseline") or {}, greedy.get("qat_on") or {}, preflight,
        )
        zero = numeric.get("zero_adapter_initialization") or {}
        checks = {
            "numeric_mandatory_gates_passed": bool(
                numeric.get("passed") is True and numeric.get("qat_enabled") is True
                and numeric.get("cross_mode_comparison") == "diagnostic"
                and numeric.get("max_initial_completion_loss") == preflight["max_initial_completion_loss"]
                and _successful_gates_match(numeric.get("mandatory_checks"), numeric_checks)
            ),
            "greedy_mandatory_gates_passed": bool(
                greedy.get("policy") == policy
                and greedy.get("passed") is True and greedy.get("qat_enabled") is True
                and greedy.get("cross_mode_comparison") == "diagnostic"
                and greedy.get("require_greedy_determinism") is True
                and greedy.get("qat_greedy_deterministic") is True
                and _successful_gates_match(greedy.get("mandatory_checks"), greedy_checks)
            ),
            "zero_adapter_initialization_verified": bool(
                zero.get("verified_zero_delta") is True and zero.get("wrapper_count") == 205
                and _integer_at_least(zero.get("adapter_pair_count"), 205)
                and zero.get("nonzero_or_invalid_pairs") == []
            ),
        }
        if policy == FULL_QAT_POLICY:
            checks.pop("zero_adapter_initialization_verified")
            checks["full_model_initialization_verified"] = bool(
                numeric.get("adapter_initialization_mode") == "full_model"
                and zero == {"required": False, "reason": "full_finetune", "verified_zero_delta": False}
            )
        report.update(checks=checks, verified=all(checks.values()))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        report["error"] = str(exc)
    return report
