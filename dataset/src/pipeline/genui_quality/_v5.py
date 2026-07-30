"""Corrected deterministic GenUI Representation Quality metric v5.0.0."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping

from . import _core
from .candidate_normalization import normalize_and_validate_candidate
from .config import (
    V5_REWARD_VERSION as REWARD_VERSION,
    RewardConfig,
    load_v5_reward_config as load_default_reward_config,
)
from .evidence_v5 import collect_output_evidence_v5
from .graph import audit_renderer_graph
from .identity import METRIC_NAME, expected_contract_hash, metric_fingerprint
from .metrics_v5 import (
    action_fidelity_v5,
    clamp01,
    content_fidelity,
    distinct_role_coverage,
    media_fidelity_v5,
    table_fidelity_v5,
)
from .source_contract import (
    CONTRACT_VERSION,
    resolve_expected_ui_contract,
    source_contract_from_mapping,
    source_text_hash,
)


_WEIGHT_TOLERANCE = 1e-12


def _nominal_atomic_budget(
    config: RewardConfig,
    dimension: str,
    atomic: str,
) -> float:
    return float(config.dimension_weights.get(dimension, 0.0)) * float(
        config.atomic_weights.get(dimension, {}).get(atomic, 0.0)
    )


def allocate_capped_atomic_weights(
    atomics: Mapping[str, Mapping[str, float | None]],
    config: RewardConfig,
) -> tuple[dict[str, float], dict[str, float], bool]:
    """Allocate global budgets with capped proportional water-filling."""

    nominal: dict[str, float] = {}
    dimensions: dict[str, str] = {}
    for dimension, values in atomics.items():
        for atomic, value in values.items():
            budget = _nominal_atomic_budget(config, dimension, atomic)
            if value is None or budget <= 0.0:
                continue
            key = f"{dimension}.{atomic}"
            nominal[key] = budget
            dimensions[key] = dimension

    count = len(nominal)
    cap = float(config.max_atomic_global_weight)
    feasible = count > 0 and count * cap >= 1.0 - _WEIGHT_TOLERANCE
    if not feasible:
        return {}, {}, False

    def total(scale: float) -> float:
        return sum(min(cap, scale * budget) for budget in nominal.values())

    low = 0.0
    high = 1.0
    while total(high) < 1.0:
        high *= 2.0
        if high > 1e18:
            return {}, {}, False
    for _ in range(160):
        midpoint = (low + high) / 2.0
        if total(midpoint) < 1.0:
            low = midpoint
        else:
            high = midpoint

    weights = {
        key: min(cap, high * nominal[key])
        for key in sorted(nominal)
    }
    residual = 1.0 - sum(weights.values())
    if abs(residual) > _WEIGHT_TOLERANCE:
        if residual > 0.0:
            for key in sorted(weights):
                available = cap - weights[key]
                addition = min(available, residual)
                weights[key] += addition
                residual -= addition
                if residual <= _WEIGHT_TOLERANCE:
                    break
        else:
            for key in reversed(sorted(weights)):
                removal = min(weights[key], -residual)
                weights[key] -= removal
                residual += removal
                if residual >= -_WEIGHT_TOLERANCE:
                    break

    dimension_weights: dict[str, float] = defaultdict(float)
    for key, weight in weights.items():
        dimension_weights[dimensions[key]] += weight

    assert all(math.isfinite(value) and value >= 0.0 for value in weights.values())
    assert all(value <= cap + 1e-10 for value in weights.values())
    assert abs(sum(weights.values()) - 1.0) <= 1e-10
    return weights, dict(sorted(dimension_weights.items())), True


def _weighted_dimension_scores(
    atomics: Mapping[str, Mapping[str, float | None]],
    config: RewardConfig,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for dimension, configured in config.atomic_weights.items():
        values = atomics.get(dimension, {})
        applicable = [
            (float(configured.get(name, 0.0)), float(value))
            for name, value in values.items()
            if value is not None and configured.get(name, 0.0) > 0.0
        ]
        denominator = sum(weight for weight, _ in applicable)
        result[dimension] = (
            sum(weight * clamp01(value) for weight, value in applicable) / denominator
            if denominator > 0.0
            else None
        )
    return result


def _normalization_evidence(
    normalization: Any,
    spec: Mapping[str, Any] | None,
    audit: _core.GraphAudit | None,
) -> dict[str, Any]:
    reachable_types: list[str] = []
    unsupported: list[str] = []
    if isinstance(spec, Mapping) and audit is not None:
        elements = spec.get("elements")
        if isinstance(elements, Mapping):
            for element_id in sorted(audit.reachable_ids):
                element = elements.get(element_id)
                if not isinstance(element, Mapping):
                    continue
                element_type = str(element.get("type") or "")
                reachable_types.append(element_type)
                if element_type not in _core.ALLOWED_TYPES:
                    unsupported.append(element_type)
    unknown_top = sorted(
        error.rsplit(":", 1)[-1]
        for error in normalization.errors
        if error.startswith("strict_schema.unknown_top_level_property:")
    )
    error_codes = sorted({str(error).split(":", 1)[0] for error in normalization.errors})
    return {
        "raw_parse_ok": normalization.raw_parse_ok,
        "production_valid": normalization.production_valid,
        "strict_schema_valid": normalization.strict_schema_valid,
        "converted_from_legacy": normalization.converted_from_legacy,
        "raw_format_utility": normalization.raw_format_utility,
        "unsupported_reachable_types": sorted(set(unsupported)),
        "unknown_top_level_properties": unknown_top,
        "validation_error_codes": error_codes,
        "reachable_types": reachable_types,
        "errors": list(normalization.errors),
    }


def _empty_breakdown(
    *,
    config: RewardConfig,
    normalization: Any,
    contract: Mapping[str, Any],
    contract_source: str,
    contract_errors: list[str],
    fingerprint: str,
    reason: str,
) -> _core.RewardBreakdown:
    identity = {
        "source_hash": str(contract.get("source_hash") or ""),
        "raw_candidate_hash": normalization.raw_hash,
        "canonical_candidate_hash": normalization.canonical_hash,
        "expected_contract_hash": expected_contract_hash(contract),
    }
    return _core.RewardBreakdown(
        reward=-1.0,
        quality_0_100=0.0,
        quality_0_1=0.0,
        cap_0_1=0.0,
        parse_stage=reason,
        dimensions={dimension: None for dimension in config.dimension_weights},
        atomics={dimension: {} for dimension in config.dimension_weights},
        evidence={
            "source": {
                "contract_source": contract_source,
                "contract_version": contract.get("contract_version"),
                "source_hash": contract.get("source_hash"),
            },
            "output": {},
        },
        metric_version=REWARD_VERSION,
        active_caps=[{"kind": "integrity", "name": reason, "cap": 0.0}],
        errors=list(dict.fromkeys([*contract_errors, *normalization.errors])),
        metric_name=METRIC_NAME,
        metric_fingerprint=fingerprint,
        base_quality_before_caps=0.0,
        effective_atomic_weights={},
        effective_dimension_weights={},
        applicable_atomic_count=0,
        anti_domination_feasible=False,
        normalization=_normalization_evidence(normalization, None, None),
        identity=identity,
    )


def _media_matched_by_role(
    source: _core.SourceContract,
    output: _core.OutputEvidence,
    config: RewardConfig,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for role, kind in (
        ("image", "Image"),
        ("video", "Video"),
        ("audio", "AudioPlayer"),
    ):
        _, diagnostics = media_fidelity_v5(
            [item for item in source.media if item.kind == kind],
            [item for item in output.media if item.kind == kind],
            aliases=source.asset_aliases,
            max_size=config.max_assignment_size,
        )
        result[role] = int(diagnostics.get("matched_required_count") or 0)
    return result


def score_genui_completion_v5(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    expected_ui_contract_source: str | None = None,
    render_ok: bool | None = None,
    config: RewardConfig | None = None,
) -> _core.RewardBreakdown:
    """Score one raw candidate through the authoritative v5 boundary."""

    cfg = config or load_default_reward_config()
    fingerprint = metric_fingerprint(cfg)
    normalization = normalize_and_validate_candidate(completion)
    contract_resolution = resolve_expected_ui_contract(
        response_text,
        intent=intent,
        assets=assets,
        persisted=expected_ui_contract,
        persisted_source=expected_ui_contract_source,
    )
    contract_mapping = contract_resolution.contract
    contract_errors = list(contract_resolution.errors)
    if normalization.canonical_spec is None:
        return _empty_breakdown(
            config=cfg,
            normalization=normalization,
            contract=contract_mapping,
            contract_source=contract_resolution.source,
            contract_errors=contract_errors,
            fingerprint=fingerprint,
            reason="parse_failure" if not normalization.raw_parse_ok else "production_invalid",
        )

    spec = normalization.canonical_spec
    audit = audit_renderer_graph(spec)
    fallback_source = _core.parse_source_contract(
        response_text,
        intent=intent,
        assets=assets,
    )
    source = source_contract_from_mapping(contract_mapping, fallback=fallback_source)
    evidence_result = collect_output_evidence_v5(spec, audit, cfg)
    output = evidence_result.output

    if render_ok is None and cfg.render_check is not None:
        try:
            render_ok = bool(cfg.render_check(spec))
        except Exception:
            render_ok = False

    schema_score, schema_evidence = _core.basic_schema_contract(spec)
    type_score, type_evidence = _core.type_contract_score(spec, output)
    content_values, content_diagnostics = content_fidelity(
        source.content_units,
        output.visible_blocks,
        max_size=cfg.max_assignment_size,
    )
    content_precision, content_recall = _core.counter_pr(
        output.content_tokens,
        source.content_tokens,
    )
    global_content = _core.f_beta(content_precision, content_recall, cfg.content_beta)
    value_precision, value_recall = _core.counter_pr(
        output.exact_values,
        source.exact_values,
    )
    exact_value_score = (
        _core.f_beta(value_precision, value_recall, cfg.value_beta)
        if source.exact_values
        else None
    )
    table_score, table_diagnostics = table_fidelity_v5(
        source.tables,
        output.tables,
        beta=cfg.table_beta,
        max_size=cfg.max_assignment_size,
    )
    action_score, action_diagnostics = action_fidelity_v5(
        source.explicit_actions,
        output.actions,
        aliases=source.asset_aliases,
        max_size=cfg.max_assignment_size,
    )
    media_score, media_diagnostics = media_fidelity_v5(
        source.media,
        output.media,
        aliases=source.asset_aliases,
        max_size=cfg.max_assignment_size,
    )
    media_by_role = _media_matched_by_role(source, output, cfg)
    role_score, role_values = distinct_role_coverage(
        source.expected_role_counts,
        evidence_result.role_signatures,
        action_matched=int(action_diagnostics.get("matched_required_count") or 0),
        table_matched=int(table_diagnostics.get("matched_required_count") or 0),
        media_matched_by_role=media_by_role,
    )
    distinct_values = [
        float(value)
        for role, value in role_values.items()
        if value is not None and role not in {"action", "table", "image", "video", "audio"}
    ]
    distinct_role_score = (
        sum(distinct_values) / len(distinct_values)
        if distinct_values
        else role_score
    )
    accessibility_score, accessibility_evidence = _core._accessibility_score(output)  # type: ignore[attr-defined]

    reachability = audit.reachable_fraction if audit.root_exists else 0.0
    reference_integrity = 1.0 if audit.root_exists and not audit.missing_references else 0.0
    cycle_free = 1.0 if audit.root_exists and not audit.cycle_edges else 0.0
    root_exists = 1.0 if audit.root_exists else 0.0
    atomics: dict[str, dict[str, float | None]] = {
        "integrity": {
            "production_validity": float(normalization.production_valid),
            "strict_schema_validity": float(normalization.strict_schema_valid),
            "raw_format_utility": normalization.raw_format_utility,
            "schema_contract": schema_score,
            "declared_root_reachability": root_exists,
            "reference_integrity": reference_integrity,
            "cycle_free": cycle_free,
            "reachable_fraction": reachability,
            "type_semantic_contract": type_score,
        },
        "fidelity": {
            "content_unit_fidelity": content_values["content_unit_fidelity"],
            "content_order_preservation": content_values["content_order_preservation"],
            "visible_content_multiset_fbeta": global_content,
            "exact_numbers_dates_units_fbeta": exact_value_score,
            "markdown_table_fidelity": table_score,
            "heading_fidelity_and_order": _core.heading_fidelity(source, output),
            "action_and_source_link_fidelity": action_score,
            "media_fidelity": media_score,
            "output_block_precision": content_values["output_block_precision"],
        },
        "semantic_mapping": {
            "required_component_roles": role_score,
            "distinct_role_instances": distinct_role_score,
            "component_appropriateness": _core.component_appropriateness(source, output),
        },
        "hierarchy": {
            "root_layout": _core._root_layout_utility(spec, output),  # type: ignore[attr-defined]
            "root_distance_depth": _core._layout_depth_utility(audit),  # type: ignore[attr-defined]
            "container_fanout": _core._fanout_utility(output),  # type: ignore[attr-defined]
            "heading_grouping": _core._heading_grouping_utility(output),  # type: ignore[attr-defined]
            "text_chunking": _core._text_chunking_utility(output, source),  # type: ignore[attr-defined]
        },
        "economy": {
            "semantic_non_duplication": _core._semantic_duplicate_utility(output),  # type: ignore[attr-defined]
            "renderer_aware_wrapper_economy": _core._redundant_wrapper_utility(output),  # type: ignore[attr-defined]
            "json_to_source_size": _core._json_efficiency_utility(spec, audit, source, cfg),  # type: ignore[attr-defined]
        },
        "accessibility": {
            "applicable_accessibility_contracts": accessibility_score,
        },
    }

    applicable_atomic_count = sum(
        value is not None
        and _nominal_atomic_budget(cfg, dimension, atomic) > 0.0
        for dimension, values in atomics.items()
        for atomic, value in values.items()
    )
    effective_weights, effective_dimensions, feasible = (
        allocate_capped_atomic_weights(atomics, cfg)
    )
    active_caps: list[dict[str, Any]] = []
    if not feasible:
        if cfg.anti_domination_infeasible_policy == "error":
            raise ValueError("anti-domination allocation is infeasible")
        active_caps.append(
            {
                "kind": "integrity",
                "name": "anti_domination_infeasible",
                "cap": float(cfg.caps.get("anti_domination_infeasible", 0.0)),
                "applicable_atomic_count": applicable_atomic_count,
            }
        )
        base_quality = 0.0
    else:
        base_quality = sum(
            effective_weights[f"{dimension}.{atomic}"] * clamp01(float(value))
            for dimension, values in atomics.items()
            for atomic, value in values.items()
            if value is not None and f"{dimension}.{atomic}" in effective_weights
        )
    assert math.isfinite(base_quality) and -1e-12 <= base_quality <= 1.0 + 1e-12
    base_quality = clamp01(base_quality)

    def activate(kind: str, name: str, evidence: Any = None) -> None:
        cap = float(cfg.caps.get(name, 1.0))
        if cap >= 1.0:
            return
        payload: dict[str, Any] = {"kind": kind, "name": name, "cap": cap}
        if evidence is not None:
            payload["evidence"] = evidence
        active_caps.append(payload)

    if not normalization.production_valid:
        activate("integrity", "production_invalid", list(normalization.errors))
    if not normalization.strict_schema_valid:
        activate("format", "strict_format", list(normalization.errors))
    if not audit.root_exists:
        activate("integrity", "missing_root")
    if audit.missing_references or audit.cycle_edges:
        activate(
            "integrity",
            "reference_or_cycle",
            {
                "missing_references": sorted(audit.missing_references),
                "cycles": sorted([list(edge) for edge in audit.cycle_edges]),
            },
        )
    if audit.root_exists and reachability < 0.90:
        activate("integrity", "reachability_below_90", reachability)
    elif audit.root_exists and reachability < 1.0:
        activate("integrity", "partial_reachability", reachability)
    if render_ok is False:
        activate("renderer", "render_failure")

    for role, value in role_values.items():
        if value is None:
            continue
        if role == "table":
            if value <= 0.0:
                activate("semantic", "missing_table", value)
            elif value < 1.0:
                activate("semantic", "partial_table", value)
        elif role == "chart":
            if value <= 0.0:
                activate("semantic", "missing_chart", value)
            elif value < 1.0:
                activate("semantic", "partial_chart", value)
        elif role == "action":
            if value <= 0.0:
                activate("semantic", "missing_action", value)
            elif value < 1.0:
                activate("semantic", "partial_action", value)
        elif role in {"image", "video", "audio"}:
            if value <= 0.0:
                activate("semantic", "missing_media", {"role": role, "coverage": value})
            elif value < 1.0:
                activate("semantic", "partial_media", {"role": role, "coverage": value})
        elif value <= 0.0:
            activate("semantic", "missing_special_role", {"role": role, "coverage": value})
        elif value < 1.0:
            activate("semantic", "partial_special_role", {"role": role, "coverage": value})

    cap = min((float(item["cap"]) for item in active_caps), default=1.0)
    quality = clamp01(min(base_quality, cap))
    reward = 2.0 * quality - 1.0
    assert math.isfinite(quality) and math.isfinite(reward)

    normalization_payload = _normalization_evidence(normalization, spec, audit)
    identity = {
        "source_hash": str(contract_mapping.get("source_hash") or source_text_hash(response_text)),
        "raw_candidate_hash": normalization.raw_hash,
        "canonical_candidate_hash": normalization.canonical_hash,
        "expected_contract_hash": expected_contract_hash(contract_mapping),
    }
    dimensions = _weighted_dimension_scores(atomics, cfg)
    parse_stage = (
        "mapping"
        if isinstance(completion, Mapping)
        else "json"
        if normalization.raw_parse_ok
        else "parse_failure"
    )
    errors = list(
        dict.fromkeys(
            [
                *contract_errors,
                *normalization.errors,
                *evidence_result.unknown_diagnostics,
            ]
        )
    )
    return _core.RewardBreakdown(
        reward=reward,
        quality_0_100=100.0 * quality,
        quality_0_1=quality,
        cap_0_1=cap,
        parse_stage=parse_stage,
        dimensions=dimensions,
        atomics=atomics,
        evidence={
            "source": {
                "contract_source": contract_resolution.source,
                "contract_version": contract_mapping.get("contract_version") or CONTRACT_VERSION,
                "source_hash": identity["source_hash"],
                "cache_hit": contract_resolution.cache_hit,
                "content_unit_count": len(source.content_units),
            },
            "output": {
                "root_exists": audit.root_exists,
                "reachable_elements": len(audit.reachable_ids),
                "unreachable_elements": len(audit.unreachable_ids),
                "reachable_ids": sorted(audit.reachable_ids),
                "unreachable_ids": sorted(audit.unreachable_ids),
                "missing_references": sorted(audit.missing_references),
                "cycles": sorted([list(edge) for edge in audit.cycle_edges]),
                "expanded_evidence_nodes": evidence_result.expanded_evidence_nodes,
                "evidence_truncated": evidence_result.truncated,
                "dynamic_unknowns": list(evidence_result.unknown_diagnostics),
            },
            "schema": schema_evidence,
            "type_contract": type_evidence,
            "content_assignment": content_diagnostics,
            "table_matching": table_diagnostics,
            "action_matching": action_diagnostics,
            "media_matching": media_diagnostics,
            "role_values": role_values,
            "role_signatures": {
                key: list(value) for key, value in evidence_result.role_signatures.items()
            },
            "accessibility": accessibility_evidence,
            "render_ok": render_ok,
        },
        metric_version=REWARD_VERSION,
        active_caps=sorted(active_caps, key=lambda item: (float(item["cap"]), str(item["name"]))),
        errors=errors,
        metric_name=METRIC_NAME,
        metric_fingerprint=fingerprint,
        base_quality_before_caps=base_quality,
        effective_atomic_weights=effective_weights,
        effective_dimension_weights=effective_dimensions,
        applicable_atomic_count=applicable_atomic_count,
        anti_domination_feasible=feasible,
        normalization=normalization_payload,
        identity=identity,
    )


__all__ = ["allocate_capped_atomic_weights", "score_genui_completion_v5"]
