"""Fail-closed provenance checks for constants retained by topology transplant.

The exact-topology exporter replaces quantized matrix/embedding buffers but
keeps other learned constants from the released LiteRT-LM template.  A
training checkpoint is therefore numerically compatible only when those
retained tensors are proven compatible with the template source.  This module
keeps that evidence separate from graph/operator parity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml


def _integer(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def verify_retained_constant_contract(
    contract_path: str | Path | None,
    *,
    family: str,
    training_model_id: str,
) -> dict[str, Any]:
    """Verify a compact retained-constant evidence contract.

    Gemma 4 E2B requires explicit evidence because its public dense Q4-QAT
    training seed and packed mobile checkpoint are separate releases.  Gemma
    3 270M currently uses the exact base model declared for its released Q8
    template, so this extra cross-checkpoint contract is not required there.
    """

    normalized_family = str(family).strip().lower().replace("-", "_")
    required = normalized_family == "gemma4_e2b"
    checks: dict[str, bool] = {
        "contract_required": required,
        "contract_present": not required,
        "training_seed_matches": not required,
        "status_compatible_exact": not required,
        "selection_nonempty": not required,
        "all_selected_tensors_exact": not required,
        "no_schema_mismatches": not required,
        "no_value_mismatches": not required,
        "exact_count_matches_selection": not required,
        "compiled_graph_mapping_verified": not required,
    }
    result: dict[str, Any] = {
        "required": required,
        "path": str(Path(contract_path).expanduser().resolve())
        if contract_path
        else None,
        "training_model_id": str(training_model_id),
        "checks": checks,
        "verified": not required,
        "evidence_scope": (
            "same_declared_base_model"
            if not required
            else "cross_checkpoint_retained_constants"
        ),
    }
    if not required:
        return result
    if contract_path is None:
        result["error"] = "Gemma 4 exact-topology export requires a retained-constant contract."
        return result

    path = Path(contract_path).expanduser().resolve()
    if not path.is_file():
        result["error"] = f"Retained-constant contract does not exist: {path}"
        return result
    checks["contract_present"] = True
    try:
        document = load_yaml(path)
    except Exception as exc:  # noqa: BLE001 - malformed evidence is a hard gate
        result["error"] = f"Could not load retained-constant contract: {exc}"
        return result
    contract = document.get("retained_constant_compatibility")
    if not isinstance(contract, dict):
        result["error"] = "Contract has no retained_constant_compatibility object."
        return result

    dense_seed = contract.get("dense_training_seed")
    dense_seed = dense_seed if isinstance(dense_seed, dict) else {}
    comparison = contract.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    selection = contract.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    selected = _integer(selection.get("selected_tensor_count"), default=0)
    exact = _integer(comparison.get("exact_tensor_count"), default=-1)
    schema_mismatches = _integer(
        comparison.get("schema_mismatch_count"), default=-1
    )
    value_mismatches = _integer(
        comparison.get("value_mismatch_count"), default=-1
    )

    checks["training_seed_matches"] = (
        str(dense_seed.get("repo_id") or "") == str(training_model_id)
    )
    checks["status_compatible_exact"] = (
        str(contract.get("production_status") or "") == "compatible_exact"
    )
    checks["selection_nonempty"] = selected > 0
    checks["all_selected_tensors_exact"] = bool(
        comparison.get("all_selected_tensors_exact", False)
    )
    checks["no_schema_mismatches"] = schema_mismatches == 0
    checks["no_value_mismatches"] = value_mismatches == 0
    checks["exact_count_matches_selection"] = selected > 0 and exact == selected
    checks["compiled_graph_mapping_verified"] = bool(
        contract.get("compiled_graph_mapping_verified", False)
    )
    result.update(
        {
            "production_status": contract.get("production_status"),
            "dense_training_seed": dense_seed,
            "packed_mobile_checkpoint": contract.get("packed_mobile_checkpoint"),
            "selection": selection,
            "comparison": comparison,
            "compiled_graph_mapping_verified": contract.get(
                "compiled_graph_mapping_verified", False
            ),
            "rejected_q4_seed_audit": contract.get("rejected_q4_seed_audit"),
            "verified": all(checks.values()),
            "limitation": (
                "Graph/operator parity does not make a cross-base checkpoint transplant "
                "numerically valid. A failed contract blocks production export but does "
                "not block the separate random-topology research harnesses."
            ),
        }
    )
    return result
