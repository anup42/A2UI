"""Explainable training eligibility, independent of the scalar reward.

This is a conservative diagnostic gate, not a calibrated quality predictor.
It never checks whether placeholder media files exist or can be downloaded.
"""
from __future__ import annotations

from typing import Any, Mapping

ACCEPTANCE_POLICY_VERSION = "1.0.0"


def training_acceptance_v5_4(result: Any) -> dict[str, Any]:
    evidence = result.evidence
    blocking: list[str] = []
    review: list[str] = []
    if not result.normalization.get("production_valid"):
        blocking.append("production_invalid")
    if not result.normalization.get("strict_schema_valid"):
        blocking.append("strict_schema_invalid")
    output = evidence.get("output") or {}
    for key in ("unreachable_elements", "missing_references", "cycles"):
        if output.get(key):
            blocking.append(f"renderer_{key}")
    if not result.dynamic_evidence_certification.get("complete", False):
        review.append("dynamic_evidence_incomplete")
    contracts = (evidence.get("type_contract") or {}).get("per_element") or {}
    if any(isinstance(value, (int, float)) and value < 1 for value in contracts.values()):
        blocking.append("renderer_component_contract")
    for domain in ("action", "media", "table"):
        detail = evidence.get(f"{domain}_matching") or {}
        required = detail.get("required_count", 0)
        matched = detail.get("matched_required_count", 0)
        if required > matched:
            # A certified assignment proves the optimizer ran correctly, not
            # that an inferred contract or semantic similarity is correct.
            review.append(f"missing_or_mismatched_{domain}")
        if domain == "table":
            for match in detail.get("matches") or []:
                if any(match.get(key) is not None and match[key] < 1 for key in ("column_fbeta", "row_fbeta", "associated_cell_fbeta")):
                    # Fuzzy matching does not prove an omission. Retain it for
                    # targeted review rather than silently accepting a row.
                    review.append("table_content_difference")
                    break
    additions = result.unsupported_external_additions or {}
    if additions.get("unsupported_external_action_count", 0):
        review.append("unsupported_external_action")
    if additions.get("unsupported_semantic_media_count", 0):
        review.append("unsupported_semantic_media")
    for role, detail in (evidence.get("role_matching") or {}).items():
        if not isinstance(detail, Mapping):
            continue
        if detail.get("semantic_required_count", 0) > detail.get("matched_required_count", 0):
            review.append(f"missing_or_mismatched_role:{role}")
        elif detail.get("semantic_role_instance_fidelity") is not None and detail["semantic_role_instance_fidelity"] < 1:
            review.append(f"semantic_role_difference:{role}")
        if detail.get("count_only_role_count_coverage") is not None and detail["count_only_role_count_coverage"] < 1:
            review.append(f"inferred_role_count_gap:{role}")
    fidelity = result.atomics.get("fidelity") or {}
    for key in ("content_unit_fidelity", "exact_numbers_dates_units_fbeta"):
        if fidelity.get(key) is not None and fidelity[key] < 1:
            review.append(key)
    for key in ("action_and_source_link_fidelity", "media_fidelity"):
        if fidelity.get(key) is not None and fidelity[key] < 1:
            review.append(key)
    blocking = list(dict.fromkeys(blocking))
    review = list(dict.fromkeys(review))
    return {
        "policy_version": ACCEPTANCE_POLICY_VERSION,
        "eligible": not blocking and not review,
        "blocking_reasons": blocking,
        "review_reasons": review,
        "media_readiness_required": False,
        "calibration_status": "conservative_diagnostic_gate",
    }
