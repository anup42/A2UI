"""Runtime-only checks for optimizer continuation across numerical fixes.

Historical checkpoints remain valid export inputs. Continuing optimizer state,
however, must not silently change the trainable-weight precision or STE rule.
Keep this check at the training entry point, outside export lineage validation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verify_numeric_training_resume(
    checkpoint: Path,
    config: dict[str, Any],
    *,
    expected_full_precision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check numerical provenance before loading an existing optimizer run.

    ``expected_full_precision`` is supplied only by the generic full-finetuning
    lane. Identity-STE LoRA, disabled QAT, and the separate all-parameter workflow
    remain unchanged unless their caller explicitly requests a precision check.
    This helper imports no torch and never loads pickle-based optimizer files.
    """
    if expected_full_precision is not None and not isinstance(expected_full_precision, dict):
        raise TypeError("expected_full_precision must be a dictionary or None")

    expected_weight_rule = None
    qat = config.get("qat")
    if isinstance(qat, dict) and qat.get("enabled", False):
        from ir_training.qat.fake_quant import QATSpec, qat_numeric_contract

        spec = QATSpec.from_config(config)
        quantized_weights = spec.weight_bits < 16 or any(
            bits < 16 for _, bits in spec.module_quant_configs
        )
        # Legacy activations share the same fixed-scale helper. The separately
        # implemented mobile SRQ activation STE has not changed.
        quantized_legacy_activations = (
            spec.activation_quantizer == "legacy" and spec.activation_bits < 16
        )
        if spec.ste_gradient == "clipped" and (
            quantized_weights or quantized_legacy_activations
        ):
            expected_weight_rule = qat_numeric_contract(spec).get("weight_ste_rule")
            if not isinstance(expected_weight_rule, str) or not expected_weight_rule:
                raise RuntimeError("Current clipped-STE implementation lacks its numerical rule version")

    if expected_full_precision is None and expected_weight_rule is None:
        return {"checked": False, "reason": "numerical_policy_unchanged"}

    path = Path(checkpoint) / "training_metadata.json"
    guidance = (
        "Start a new run for an intentional weight-only initialization; "
        "do not resume the old optimizer state under a changed numerical policy."
    )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot verify numerical training resume metadata at {path}. {guidance}") from exc
    if not isinstance(metadata, dict):
        # This is invalid checkpoint content, not an invalid Python argument.
        raise ValueError(f"Numerical training resume metadata must be a JSON object. {guidance}")  # noqa: TRY004

    checked: dict[str, Any] = {}
    if expected_full_precision is not None:
        actual = metadata.get("full_finetune_precision")
        try:
            # JSON comparison distinguishes True from 1, unlike dict equality.
            same = json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
                expected_full_precision, sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError):
            same = False
        if not isinstance(actual, dict) or not same:
            raise ValueError(
                "Checkpoint full_finetune_precision is absent or changed. " + guidance
            )
        checked["full_finetune_precision"] = expected_full_precision

    if expected_weight_rule is not None:
        recorded_qat = metadata.get("qat")
        numeric = recorded_qat.get("numeric_contract") if isinstance(recorded_qat, dict) else None
        actual_rule = numeric.get("weight_ste_rule") if isinstance(numeric, dict) else None
        if actual_rule != expected_weight_rule:
            raise ValueError(
                "Checkpoint clipped-STE weight_ste_rule is absent or changed "
                f"(expected {expected_weight_rule!r}). " + guidance
            )
        checked["weight_ste_rule"] = expected_weight_rule
    return {"checked": True, "verified": True, **checked}
