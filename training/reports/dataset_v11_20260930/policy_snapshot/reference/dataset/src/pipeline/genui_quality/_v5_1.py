"""Production-aligned deterministic GenUI Representation Quality v5.1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from . import _core
from ._v5 import (
    _nominal_atomic_budget,
    _normalization_evidence,
    _weighted_dimension_scores,
    allocate_capped_atomic_weights,
)
from .candidate_normalization import normalize_and_validate_candidate
from .config import (
    V5_1_REWARD_VERSION as REWARD_VERSION,
    RewardConfig,
    load_v5_1_reward_config,
)
from .evidence_v5_1 import (
    ComputedFunctionRegistry,
    collect_output_evidence_v5_1,
)
from .graph import audit_renderer_graph
from .identity_v5_1 import (
    METRIC_NAME,
    expected_contract_hash_v5_1,
    metric_fingerprint_v5_1,
)
from .matching_v5_1 import PreparedTextBlock, prepare_text_block
from .metrics_v5_1 import (
    action_fidelity_v5_1,
    clamp01,
    content_fidelity_v5_1,
    media_fidelity_v5_1,
    semantic_role_coverage_v5_1,
    table_fidelity_v5_1,
)
from .source_contract import (
    ContractResolution,
    V5_1_CONTRACT_VERSION,
    resolve_expected_ui_contract_v5_1,
    source_contract_from_mapping,
    source_text_hash,
)
from .structure_v5_1 import (
    fanout_utility_v5_1,
    heading_grouping_utility_v5_1,
    json_efficiency_utility_v5_1,
    root_layout_utility_v5_1,
    wrapper_economy_v5_1,
)
from .validation_v5_1 import ensure_v5_1_validation_ready


_CAP_BINDING_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PreparedSourceContext:
    source_text: str
    intent: str | None
    assets: Any
    expected_contract: Mapping[str, Any]
    contract_resolution: ContractResolution
    source_contract: _core.SourceContract
    content_blocks: tuple[PreparedTextBlock, ...]
    source_hash: str
    contract_hash: str
    config: RewardConfig
    metric_fingerprint: str


def prepare_source_context(
    source_text: str,
    *,
    intent: str | None = None,
    assets: Sequence[Mapping[str, Any]] | None = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    expected_ui_contract_source: str | None = None,
    config: RewardConfig | None = None,
) -> PreparedSourceContext:
    """Resolve and prepare source evidence once for scalar or group scoring."""

    ensure_v5_1_validation_ready()
    cfg = config or load_v5_1_reward_config()
    resolution = resolve_expected_ui_contract_v5_1(
        source_text,
        intent=intent,
        assets=assets,
        persisted=expected_ui_contract,
        persisted_source=expected_ui_contract_source,
    )
    fallback = _core.parse_source_contract(
        source_text, intent=intent, assets=assets
    )
    source = source_contract_from_mapping(resolution.contract, fallback=fallback)
    return PreparedSourceContext(
        source_text=str(source_text or ""),
        intent=intent,
        assets=assets,
        expected_contract=resolution.contract,
        contract_resolution=resolution,
        source_contract=source,
        content_blocks=tuple(
            prepare_text_block(value) for value in source.content_units
        ),
        source_hash=str(
            resolution.contract.get("source_hash")
            or source_text_hash(source_text)
        ),
        contract_hash=expected_contract_hash_v5_1(resolution.contract),
        config=cfg,
        metric_fingerprint=metric_fingerprint_v5_1(cfg),
    )


def _empty_breakdown(
    prepared: PreparedSourceContext,
    normalization: Any,
    reason: str,
) -> _core.RewardBreakdown:
    active = [{"kind": "integrity", "name": reason, "cap": 0.0}]
    return _core.RewardBreakdown(
        reward=-1.0,
        quality_0_100=0.0,
        quality_0_1=0.0,
        cap_0_1=0.0,
        parse_stage=reason,
        dimensions={
            dimension: None
            for dimension in prepared.config.dimension_weights
        },
        atomics={
            dimension: {}
            for dimension in prepared.config.dimension_weights
        },
        evidence={
            "source": {
                "contract_source": prepared.contract_resolution.source,
                "contract_version": prepared.expected_contract.get(
                    "contract_version"
                ),
                "source_hash": prepared.source_hash,
            },
            "output": {},
            "matching_complete": False,
            "dynamic_evidence_complete": False,
        },
        metric_version=REWARD_VERSION,
        active_caps=active,
        binding_caps=list(active),
        cap_margin=0.0,
        errors=list(
            dict.fromkeys(
                [
                    *prepared.contract_resolution.errors,
                    *normalization.errors,
                ]
            )
        ),
        metric_name=METRIC_NAME,
        metric_fingerprint=prepared.metric_fingerprint,
        base_quality_before_caps=0.0,
        effective_atomic_weights={},
        effective_dimension_weights={},
        applicable_atomic_count=0,
        anti_domination_feasible=False,
        normalization=_normalization_evidence(normalization, None, None),
        identity={
            "source_hash": prepared.source_hash,
            "raw_candidate_hash": normalization.raw_hash,
            "canonical_candidate_hash": normalization.canonical_hash,
            "expected_contract_hash": prepared.contract_hash,
        },
    )


def _role_requirements(
    source: _core.SourceContract,
) -> dict[str, list[dict[str, Any]]]:
    if source.role_requirements:
        return {
            role: [dict(item) for item in values]
            for role, values in source.role_requirements.items()
        }
    requirements: dict[str, list[dict[str, Any]]] = {}
    for role in ("chart", "formula", "code", "console", "email"):
        count = max(0, int(source.expected_role_counts.get(role, 0)))
        if count:
            plural = {
                "chart": "chart",
                "formula": "formula",
                "code": "code",
                "console": "console",
                "email": "email",
            }[role]
            requirements[plural] = [
                {
                    "id": f"compat_{role}_{index + 1}",
                    "required": True,
                    "minimum_count": 1,
                    "interchangeable": False,
                }
                for index in range(count)
            ]
    return requirements


def _role_value(matched: int, required: int) -> float | None:
    return min(1.0, matched / required) if required > 0 else None


def _media_by_role(
    source: _core.SourceContract,
    output: _core.OutputEvidence,
    config: RewardConfig,
) -> dict[str, tuple[int, int, bool]]:
    result: dict[str, tuple[int, int, bool]] = {}
    for role, kind in (
        ("image", "Image"),
        ("video", "Video"),
        ("audio", "AudioPlayer"),
    ):
        _, diagnostics = media_fidelity_v5_1(
            [item for item in source.media if item.kind == kind],
            [item for item in output.media if item.kind == kind],
            aliases=source.asset_aliases,
            exact_dense_limit=config.max_assignment_size,
            max_edges=config.max_matching_edges,
            top_k=config.large_matching_top_k,
        )
        result[role] = (
            int(diagnostics.get("matched_required_count") or 0),
            int(diagnostics.get("required_count") or 0),
            bool(diagnostics.get("matching_complete", True)),
        )
    return result


def _score_one(
    completion: Any,
    prepared: PreparedSourceContext,
    *,
    render_ok: bool | None,
    computed_functions: ComputedFunctionRegistry | None,
) -> _core.RewardBreakdown:
    cfg = prepared.config
    normalization = normalize_and_validate_candidate(completion)
    if normalization.canonical_spec is None:
        return _empty_breakdown(
            prepared,
            normalization,
            "parse_failure"
            if not normalization.raw_parse_ok
            else "production_invalid",
        )

    spec = normalization.canonical_spec
    audit = audit_renderer_graph(spec)
    evidence_result = collect_output_evidence_v5_1(
        spec,
        audit,
        cfg,
        computed_functions=computed_functions,
    )
    output = evidence_result.output
    source = prepared.source_contract
    if render_ok is None and cfg.render_check is not None:
        try:
            render_ok = bool(cfg.render_check(spec))
        except Exception:
            render_ok = False

    schema_score, schema_evidence = _core.basic_schema_contract(spec)
    type_score, type_evidence = _core.type_contract_score(spec, output)
    content_values, content_diagnostics = content_fidelity_v5_1(
        prepared.content_blocks,
        evidence_result.prepared_visible_blocks,
        exact_dense_limit=cfg.max_assignment_size,
        max_edges=cfg.max_matching_edges,
        top_k=cfg.large_matching_top_k,
    )
    content_precision, content_recall = _core.counter_pr(
        output.content_tokens, source.content_tokens
    )
    global_content = _core.f_beta(
        content_precision, content_recall, cfg.content_beta
    )
    value_precision, value_recall = _core.counter_pr(
        output.exact_values, source.exact_values
    )
    exact_value_score = (
        _core.f_beta(value_precision, value_recall, cfg.value_beta)
        if source.exact_values
        else None
    )
    table_score, table_diagnostics = table_fidelity_v5_1(
        source.tables,
        evidence_result.output_tables,
        evidence_result.output_charts,
        beta=cfg.table_beta,
        exact_dense_limit=cfg.max_assignment_size,
        max_edges=cfg.max_matching_edges,
        top_k=cfg.large_matching_top_k,
    )
    action_score, action_diagnostics = action_fidelity_v5_1(
        source.explicit_actions,
        output.actions,
        aliases=source.asset_aliases,
        exact_dense_limit=cfg.max_assignment_size,
        max_edges=cfg.max_matching_edges,
        top_k=cfg.large_matching_top_k,
    )
    media_score, media_diagnostics = media_fidelity_v5_1(
        source.media,
        output.media,
        aliases=source.asset_aliases,
        exact_dense_limit=cfg.max_assignment_size,
        max_edges=cfg.max_matching_edges,
        top_k=cfg.large_matching_top_k,
    )
    special_score, special_values, special_diagnostics = (
        semantic_role_coverage_v5_1(
            _role_requirements(source),
            evidence_result.role_instances,
            threshold=cfg.role_match_threshold,
            exact_dense_limit=cfg.max_assignment_size,
            max_edges=cfg.max_matching_edges,
            top_k=cfg.large_matching_top_k,
        )
    )
    media_roles = _media_by_role(source, output, cfg)
    role_values: dict[str, float | None] = dict(special_values)
    role_values["table"] = _role_value(
        int(table_diagnostics.get("matched_required_count") or 0),
        int(table_diagnostics.get("required_count") or 0),
    )
    role_values["action"] = _role_value(
        int(action_diagnostics.get("matched_required_count") or 0),
        int(action_diagnostics.get("required_count") or 0),
    )
    for role, (matched, required, _) in media_roles.items():
        role_values[role] = _role_value(matched, required)
    for role in ("chart", "formula", "code", "console", "email"):
        role_values.setdefault(role, None)
    applicable_roles = [
        float(value) for value in role_values.values() if value is not None
    ]
    role_score = (
        sum(applicable_roles) / len(applicable_roles)
        if applicable_roles
        else None
    )
    accessibility_score, accessibility_evidence = _core._accessibility_score(  # type: ignore[attr-defined]
        output
    )

    reachability = audit.reachable_fraction if audit.root_exists else 0.0
    atomics: dict[str, dict[str, float | None]] = {
        "integrity": {
            "production_validity": float(normalization.production_valid),
            "strict_schema_validity": float(normalization.strict_schema_valid),
            "raw_format_utility": normalization.raw_format_utility,
            "schema_contract": schema_score,
            "declared_root_reachability": float(audit.root_exists),
            "reference_integrity": float(
                audit.root_exists and not audit.missing_references
            ),
            "cycle_free": float(audit.root_exists and not audit.cycle_edges),
            "reachable_fraction": reachability,
            "type_semantic_contract": type_score,
        },
        "fidelity": {
            "content_unit_fidelity": content_values["content_unit_fidelity"],
            "content_order_preservation": content_values[
                "content_order_preservation"
            ],
            "visible_content_multiset_fbeta": global_content,
            "exact_numbers_dates_units_fbeta": exact_value_score,
            "markdown_table_fidelity": table_score,
            "heading_fidelity_and_order": _core.heading_fidelity(source, output),
            "action_and_source_link_fidelity": action_score,
            "media_fidelity": media_score,
            "output_block_precision": content_values[
                "output_block_precision"
            ],
        },
        "semantic_mapping": {
            "required_component_roles": role_score,
            "distinct_role_instances": special_score,
            "component_appropriateness": _core.component_appropriateness(
                source, output
            ),
        },
        "hierarchy": {
            "root_layout": root_layout_utility_v5_1(spec, output),
            "root_distance_depth": _core._layout_depth_utility(audit),  # type: ignore[attr-defined]
            "container_fanout": fanout_utility_v5_1(output),
            "heading_grouping": heading_grouping_utility_v5_1(output),
            "text_chunking": _core._text_chunking_utility(output, source),  # type: ignore[attr-defined]
        },
        "economy": {
            "semantic_non_duplication": _core._semantic_duplicate_utility(output),  # type: ignore[attr-defined]
            "renderer_aware_wrapper_economy": wrapper_economy_v5_1(output),
            "json_to_source_size": json_efficiency_utility_v5_1(
                spec, audit, source, cfg
            ),
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
        base_quality = 0.0
        active_caps.append(
            {
                "kind": "integrity",
                "name": "anti_domination_infeasible",
                "cap": float(
                    cfg.caps.get("anti_domination_infeasible", 0.0)
                ),
                "applicable_atomic_count": applicable_atomic_count,
            }
        )
    else:
        base_quality = sum(
            effective_weights[f"{dimension}.{atomic}"]
            * clamp01(float(value))
            for dimension, values in atomics.items()
            for atomic, value in values.items()
            if value is not None
            and f"{dimension}.{atomic}" in effective_weights
        )
    assert math.isfinite(base_quality) and -1e-12 <= base_quality <= 1.0 + 1e-12
    base_quality = clamp01(base_quality)

    def activate(kind: str, name: str, evidence: Any = None) -> None:
        cap_value = float(cfg.caps.get(name, 1.0))
        if cap_value >= 1.0:
            return
        payload: dict[str, Any] = {
            "kind": kind,
            "name": name,
            "cap": cap_value,
        }
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

    matching_complete = all(
        (
            bool(content_diagnostics.get("matching", {}).get("complete", True)),
            bool(table_diagnostics.get("matching_complete", True)),
            bool(action_diagnostics.get("matching_complete", True)),
            bool(media_diagnostics.get("matching_complete", True)),
            all(
                bool(value.get("matching_complete", True))
                for value in special_diagnostics.values()
                if isinstance(value, Mapping)
            ),
            all(value[2] for value in media_roles.values()),
        )
    )
    if not matching_complete and cfg.matching_incomplete_policy == "fail_closed":
        activate("matching", "matching_incomplete")
    if not evidence_result.dynamic_evidence_complete:
        if cfg.dynamic_unknown_policy == "error":
            raise ValueError("dynamic renderer evidence is incomplete")
        if cfg.dynamic_unknown_policy == "cap":
            activate(
                "evidence",
                "dynamic_evidence_unknown",
                list(evidence_result.unknown_diagnostics),
            )

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
                activate("semantic", "missing_media", {"role": role})
            elif value < 1.0:
                activate("semantic", "partial_media", {"role": role})
        elif value <= 0.0:
            activate("semantic", "missing_special_role", {"role": role})
        elif value < 1.0:
            activate("semantic", "partial_special_role", {"role": role})

    cap = min(
        (float(item["cap"]) for item in active_caps),
        default=1.0,
    )
    quality = clamp01(min(base_quality, cap))
    binding_caps = [
        dict(item)
        for item in active_caps
        if base_quality
        > float(item.get("cap", 1.0)) + _CAP_BINDING_TOLERANCE
    ]
    cap_margin = base_quality - cap
    reward = 2.0 * quality - 1.0
    assert math.isfinite(quality) and math.isfinite(reward)
    assert -1.0 <= reward <= 1.0

    normalization_payload = _normalization_evidence(
        normalization, spec, audit
    )
    identity = {
        "source_hash": prepared.source_hash,
        "raw_candidate_hash": normalization.raw_hash,
        "canonical_candidate_hash": normalization.canonical_hash,
        "expected_contract_hash": prepared.contract_hash,
    }
    dimensions = _weighted_dimension_scores(atomics, cfg)
    errors = list(
        dict.fromkeys(
            [
                *prepared.contract_resolution.errors,
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
        parse_stage=(
            "mapping" if isinstance(completion, Mapping) else "json"
        ),
        dimensions=dimensions,
        atomics=atomics,
        evidence={
            "source": {
                "contract_source": prepared.contract_resolution.source,
                "contract_version": prepared.expected_contract.get(
                    "contract_version", V5_1_CONTRACT_VERSION
                ),
                "source_hash": prepared.source_hash,
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
                "output_tables": len(evidence_result.output_tables),
                "output_charts": len(evidence_result.output_charts),
                "structured_data_payloads": len(
                    evidence_result.structured_data_payloads
                ),
                "expanded_evidence_nodes": evidence_result.expanded_evidence_nodes,
                "evidence_truncated": evidence_result.truncated,
                "dynamic_expression_unknown_count": evidence_result.dynamic_expression_unknown_count,
                "dynamic_expression_unknown_codes": list(
                    evidence_result.unknown_diagnostics
                ),
                "dynamic_evidence_complete": evidence_result.dynamic_evidence_complete,
            },
            "schema": schema_evidence,
            "type_contract": type_evidence,
            "content_assignment": content_diagnostics,
            "table_matching": table_diagnostics,
            "action_matching": action_diagnostics,
            "media_matching": media_diagnostics,
            "role_matching": special_diagnostics,
            "role_values": role_values,
            "role_instances": evidence_result.role_instances,
            "accessibility": accessibility_evidence,
            "matching_complete": matching_complete,
            "dynamic_expression_unknown_count": evidence_result.dynamic_expression_unknown_count,
            "dynamic_expression_unknown_codes": list(
                evidence_result.unknown_diagnostics
            ),
            "dynamic_evidence_complete": evidence_result.dynamic_evidence_complete,
            "render_ok": render_ok,
        },
        metric_version=REWARD_VERSION,
        active_caps=active_caps,
        binding_caps=binding_caps,
        cap_margin=cap_margin,
        errors=errors,
        metric_name=METRIC_NAME,
        metric_fingerprint=prepared.metric_fingerprint,
        base_quality_before_caps=base_quality,
        effective_atomic_weights=effective_weights,
        effective_dimension_weights=effective_dimensions,
        applicable_atomic_count=applicable_atomic_count,
        anti_domination_feasible=feasible,
        normalization=normalization_payload,
        identity=identity,
    )


def score_completion_group(
    completions: Sequence[Any],
    prepared_source: PreparedSourceContext,
    *,
    render_results: Sequence[bool | None] | None = None,
    computed_functions: ComputedFunctionRegistry | None = None,
) -> tuple[_core.RewardBreakdown, ...]:
    renders = (
        list(render_results)
        if render_results is not None
        else [None] * len(completions)
    )
    if len(renders) != len(completions):
        raise ValueError("render_results length must match completions")
    return tuple(
        _score_one(
            completion,
            prepared_source,
            render_ok=renders[index],
            computed_functions=computed_functions,
        )
        for index, completion in enumerate(completions)
    )


def score_genui_completion_v5_1(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    expected_ui_contract_source: str | None = None,
    render_ok: bool | None = None,
    config: RewardConfig | None = None,
    computed_functions: ComputedFunctionRegistry | None = None,
) -> _core.RewardBreakdown:
    prepared = prepare_source_context(
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=expected_ui_contract,
        expected_ui_contract_source=expected_ui_contract_source,
        config=config,
    )
    return score_completion_group(
        [completion],
        prepared,
        render_results=[render_ok],
        computed_functions=computed_functions,
    )[0]


__all__ = [
    "PreparedSourceContext",
    "prepare_source_context",
    "score_completion_group",
    "score_genui_completion_v5_1",
]
