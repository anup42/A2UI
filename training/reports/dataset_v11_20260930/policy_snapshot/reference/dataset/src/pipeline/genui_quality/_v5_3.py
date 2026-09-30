"""Source-applicable Android-aligned GenUI Representation Quality v5.3."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from time import perf_counter
from typing import Any, Callable, Hashable, Mapping, Sequence

from . import _core
from ._v5 import (
    _nominal_atomic_budget,
    _normalization_evidence,
    _weighted_dimension_scores,
    allocate_capped_atomic_weights,
)
from .candidate_normalization import normalize_and_validate_candidate
from .applicability_v5_3 import (
    AtomicApplicabilityPlan,
    build_atomic_applicability_plan,
    filter_source_actions,
    filter_source_media,
)
from .config_v5_3 import (
    REWARD_VERSION_V53,
    RewardBreakdownV53,
    RewardConfigV53,
    coerce_reward_config_v5_3,
    load_v5_3_reward_config,
)
from .evidence_v5_3 import (
    ComputedFunctionRegistryV53,
    collect_output_evidence_v5_3,
    effective_registry_identity_v5_3,
)
from .graph import audit_renderer_graph
from .identity_v5_3 import (
    METRIC_NAME,
    expected_contract_hash_v5_3,
    metric_fingerprint_v5_3,
    reward_pipeline_fingerprint_v5_3,
)
from .matching_v5_1 import PreparedTextBlock, prepare_text_block
from .matching_v5_3 import group_exact_indices
from .metrics_v5_3 import (
    action_exact_key_expected_v5_3,
    action_fidelity_v5_3,
    clamp01,
    combine_certifications,
    content_exact_key_v5_3,
    content_fidelity_v5_3,
    media_exact_key_expected_v5_3,
    media_fidelity_v5_3,
    role_exact_key_expected_v5_3,
    semantic_role_coverage_v5_3,
    table_fidelity_v5_3,
    unsupported_external_addition_precision_v5_3,
)
from .source_contract import (
    ContractResolution,
    source_contract_from_mapping,
    source_text_hash,
)
from .source_contract_v5_3 import (
    V5_3_CONTRACT_VERSION,
    resolve_expected_ui_contract_v5_3,
)
from .structure_v5_1 import (
    fanout_utility_v5_1,
    heading_grouping_utility_v5_1,
    json_efficiency_utility_v5_1,
    root_layout_utility_v5_1,
    wrapper_economy_v5_1,
)
from .validation_v5_3 import ensure_v5_3_validation_ready


_CAP_BINDING_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PreparedSourceContextV53:
    """Immutable source-side work shared by all completions in one group."""

    source_text: str
    intent: str | None
    assets: Any
    expected_contract: Mapping[str, Any]
    contract_resolution: ContractResolution
    source_contract: _core.SourceContract
    applicability: AtomicApplicabilityPlan
    expected_actions: tuple[_core.ActionRef, ...]
    expected_media: tuple[_core.MediaRef, ...]
    content_blocks: tuple[PreparedTextBlock, ...]
    content_exact_index: Mapping[str, tuple[int, ...]]
    action_exact_index: Mapping[Hashable, tuple[int, ...]]
    media_exact_index: Mapping[Hashable, tuple[int, ...]]
    role_exact_indices: Mapping[
        str, Mapping[str, tuple[int, ...]]
    ]
    source_hash: str
    contract_hash: str
    config: RewardConfigV53
    metric_fingerprint: str
    reward_pipeline_fingerprint: str
    computed_registry_identity: Mapping[str, Any]


def _group_indices(values: Sequence[Hashable]) -> dict[Hashable, tuple[int, ...]]:
    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[value].append(index)
    return {
        key: tuple(indices)
        for key, indices in sorted(grouped.items(), key=lambda item: repr(item[0]))
    }


def _content_key(value: PreparedTextBlock) -> str:
    return hashlib.sha256(value.normalized.encode("utf-8")).hexdigest()


def _canonical_alias(
    value: str, aliases: Mapping[str, set[str]]
) -> str:
    normalized = _core.normalize_url(value)
    equivalents = {normalized}
    for key, members in aliases.items():
        group = {
            _core.normalize_url(key),
            *(_core.normalize_url(member) for member in members),
        }
        if normalized in group:
            equivalents.update(group)
    return min(equivalents) if equivalents else normalized


def _role_requirements(
    source: _core.SourceContract,
) -> dict[str, list[dict[str, Any]]]:
    supported_roles = {"chart", "formula", "code", "console", "email", "form"}
    result: dict[str, list[dict[str, Any]]] = {
        str(role): [dict(item) for item in values]
        for role, values in source.role_requirements.items()
        if str(role) in supported_roles
    }
    for role in ("chart", "formula", "code", "console", "email", "form"):
        count = max(0, int(source.expected_role_counts.get(role, 0)))
        represented = sum(
            max(1, int(item.get("minimum_count", 1) or 1))
            for item in result.get(role, ())
            if bool(item.get("required", True))
            and int(item.get("minimum_count", 1) or 0) > 0
        )
        missing = max(0, count - represented)
        if missing:
            result.setdefault(role, []).extend(
                {
                    "id": f"count_only_{role}_{represented + index + 1}",
                    "required": True,
                    "minimum_count": 1,
                    "interchangeable": True,
                }
                for index in range(missing)
            )
    return result


def prepare_source_context_v5_3(
    source_text: str,
    *,
    intent: str | None = None,
    assets: Sequence[Mapping[str, Any]] | None = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    expected_ui_contract_source: str | None = None,
    config: RewardConfigV53 | None = None,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> PreparedSourceContextV53:
    """Resolve source truth, semantic requirements, indices, and identity once."""

    ensure_v5_3_validation_ready()
    cfg = coerce_reward_config_v5_3(config)
    resolution = resolve_expected_ui_contract_v5_3(
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
    applicability = build_atomic_applicability_plan(
        source,
        resolution.contract,
        unsupported_external_additions_policy=(
            cfg.unsupported_external_additions_policy
        ),
    )
    expected_actions = tuple(filter_source_actions(source, applicability))
    expected_media = tuple(filter_source_media(source, applicability))
    content_blocks = tuple(
        prepare_text_block(value)
        for value in applicability.generic_source_units
    )
    action_keys = [
        action_exact_key_expected_v5_3(item, source.asset_aliases)
        for item in expected_actions
        if item.required and item.minimum_count > 0
        for _ in range(max(1, item.minimum_count))
    ]
    media_keys = [
        media_exact_key_expected_v5_3(item, source.asset_aliases)
        for item in expected_media
        if item.required and item.minimum_count > 0
        for _ in range(max(1, item.minimum_count))
    ]
    roles = _role_requirements(source)
    role_indices: dict[str, Mapping[str, tuple[int, ...]]] = {}
    for role, requirements in sorted(roles.items()):
        keys = [
            role_exact_key_expected_v5_3(role, item)
            for item in requirements
            if bool(item.get("required", True))
            and int(item.get("minimum_count", 1) or 0) > 0
            for _ in range(
                max(1, int(item.get("minimum_count", 1) or 1))
            )
        ]
        role_indices[role] = group_exact_indices(keys)
    fingerprint = metric_fingerprint_v5_3(
        cfg, computed_registry=computed_registry
    )
    return PreparedSourceContextV53(
        source_text=str(source_text or ""),
        intent=intent,
        assets=assets,
        expected_contract=resolution.contract,
        contract_resolution=resolution,
        source_contract=source,
        applicability=applicability,
        expected_actions=expected_actions,
        expected_media=expected_media,
        content_blocks=content_blocks,
        content_exact_index=group_exact_indices(
            [content_exact_key_v5_3(value) for value in content_blocks]
        ),
        action_exact_index=group_exact_indices(action_keys),
        media_exact_index=group_exact_indices(media_keys),
        role_exact_indices=role_indices,
        source_hash=str(
            resolution.contract.get("source_hash")
            or source_text_hash(source_text)
        ),
        contract_hash=expected_contract_hash_v5_3(resolution.contract),
        config=cfg,
        metric_fingerprint=fingerprint,
        reward_pipeline_fingerprint=reward_pipeline_fingerprint_v5_3(
            fingerprint
        ),
        computed_registry_identity=effective_registry_identity_v5_3(
            computed_registry
        ),
    )


def _empty_breakdown(
    prepared: PreparedSourceContextV53,
    normalization: Any,
    reason: str,
) -> RewardBreakdownV53:
    active = [{"kind": "integrity", "name": reason, "cap": 0.0}]
    matching = {
        "vertex_edge_coverage_complete": False,
        "full_cardinality_matching_exists": False,
        "optimality_certified": False,
        "approximate_matching_used": False,
        "candidate_generation_complete": False,
        "lower_bound_weight": 0.0,
        "upper_bound_weight": None,
        "evaluated_edge_count": 0,
        "candidate_edge_count": 0,
        "exact_preallocated_count": 0,
        "fuzzy_residual_count": 0,
        "diagnostic_codes": [reason],
        "domains": {},
    }
    dynamic = {
        "complete": False,
        "unknown_count": 0,
        "unknown_codes": [reason],
        "computed_registry": dict(prepared.computed_registry_identity),
    }
    specificity = float(
        prepared.expected_contract.get("contract_semantic_specificity", 0.0)
        or 0.0
    )
    count_only = int(
        prepared.expected_contract.get("count_only_role_count", 0) or 0
    )
    return RewardBreakdownV53(
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
                "contract_semantic_specificity": prepared.expected_contract.get(
                    "contract_semantic_specificity"
                ),
            },
            "output": {},
            "matching_certification": matching,
            "dynamic_semantics": dynamic,
        },
        metric_version=REWARD_VERSION_V53,
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
            "computed_registry_hash": str(
                prepared.computed_registry_identity.get(
                    "custom_identity_hash"
                )
                or prepared.computed_registry_identity.get(
                    "builtin_identity_hash"
                )
                or ""
            ),
        },
        matching_certification=matching,
        dynamic_semantics=dynamic,
        atomic_applicability=prepared.applicability.atomic_mapping(),
        evidence_ownership=dict(prepared.applicability.evidence_ownership),
        unsupported_external_additions={
            "policy": prepared.config.unsupported_external_additions_policy,
            "count": 0,
        },
        contract_semantic_specificity=specificity,
        count_only_role_count=count_only,
        dynamic_evidence_certification={
            "complete": False,
            "diagnostic_codes": [reason],
        },
        reward_pipeline_fingerprint=prepared.reward_pipeline_fingerprint,
    )


def _role_value(matched: int, required: int) -> float | None:
    return min(1.0, matched / required) if required > 0 else None


def _media_role_values(
    expected_media: Sequence[_core.MediaRef],
    output_media: Sequence[_core.MediaRef],
    aliases: Mapping[str, set[str]],
    expected_exact_index: Mapping[Hashable, Sequence[int]],
    config: RewardConfigV53,
) -> tuple[
    dict[str, float | None],
    dict[str, Any],
]:
    values: dict[str, float | None] = {}
    diagnostics: dict[str, Any] = {}
    for role, expected_kind in (
        ("image", "image"),
        ("video", "video"),
        ("audio", "audio"),
    ):
        expected = [
            item
            for item in expected_media
            if item.kind.casefold().replace("audioplayer", "audio")
            == expected_kind
        ]
        actual = [
            item
            for item in output_media
            if item.kind.casefold().replace("audioplayer", "audio")
            == expected_kind
        ]
        exact_index = group_exact_indices(
            [
                media_exact_key_expected_v5_3(item, aliases)
                for item in expected
                if item.required and item.minimum_count > 0
                for _ in range(max(1, item.minimum_count))
            ]
        )
        _, detail = media_fidelity_v5_3(
            expected,
            actual,
            aliases=aliases,
            expected_exact_index=(
                exact_index if expected else expected_exact_index
            ),
            exact_dense_limit=config.max_assignment_size,
            max_edges=config.max_matching_edges,
            top_k=config.large_matching_top_k,
            max_hungarian_work=(
                config.max_matching_hungarian_work
            ),
            max_sparse_relaxations=(
                config.max_matching_sparse_relaxations
            ),
        )
        values[role] = _role_value(
            int(detail.get("matched_required_count") or 0),
            int(detail.get("required_count") or 0),
        )
        diagnostics[role] = detail
    return values, diagnostics


def _mean_applicable(
    values: Mapping[str, float | None],
) -> float | None:
    applicable = [
        clamp01(value) for value in values.values() if value is not None
    ]
    return (
        sum(applicable) / len(applicable)
        if applicable
        else None
    )


def _structured_aware_chunking(
    prepared: PreparedSourceContextV53,
    output_blocks: Sequence[PreparedTextBlock],
) -> float | None:
    """Score generic semantic segments only; structured-only sources are N/A."""

    if not prepared.applicability.text_chunking:
        return None
    lengths = [
        len(block.tokens)
        for block in output_blocks
        if block.tokens
    ]
    if not lengths:
        return 0.0
    total = sum(lengths)
    source_total = max(
        1,
        sum(
            len(_core.tokenize(value))
            for value in prepared.applicability.generic_source_units
        ),
    )
    giant = sum(max(0, length - 90) for length in lengths)
    giant_utility = 1.0 - _core.safe_div(giant, max(1, total))
    tiny_share = _core.safe_div(
        sum(1 for length in lengths if 0 < length <= 2), len(lengths)
    )
    tiny_utility = _core.low_is_good(tiny_share, 0.05, 0.45)
    fragment_density = _core.safe_div(len(lengths), max(1, total))
    density_utility = _core.low_is_good(
        fragment_density, 0.12, 0.50
    )
    longest_share = max(lengths) / source_total
    dump_utility = _core.low_is_good(longest_share, 0.18, 0.55)
    return clamp01(
        0.35 * giant_utility
        + 0.25 * tiny_utility
        + 0.15 * density_utility
        + 0.25 * dump_utility
    )


def _owned_exact_values(values: Sequence[str]) -> Counter[str]:
    result: Counter[str] = Counter()
    for block in values:
        result.update(
            _core.normalize_match_text(value)
            for value in _core._EXACT_VALUE_RE.findall(block)  # type: ignore[attr-defined]
        )
        result.update(
            _core.normalize_match_text(value)
            for value in _core._DATE_RE.findall(block)  # type: ignore[attr-defined]
        )
    result.pop("", None)
    return result


def _matching_budget_detail(domain: str, count: int) -> dict[str, Any]:
    return {
        "domain": domain,
        "input_count": count,
        "matching": {
            "expected_count": 0,
            "actual_count": 0,
            "evaluated_edge_count": 0,
            "candidate_edge_count": 0,
            "exact_preallocated_count": 0,
            "fuzzy_residual_count": count,
            "certification": {
                "vertex_edge_coverage_complete": False,
                "full_cardinality_matching_exists": False,
                "optimality_certified": False,
                "approximate_matching_used": False,
                "candidate_generation_complete": False,
                "lower_bound_weight": 0.0,
                "upper_bound_weight": None,
                "diagnostic_codes": [
                    f"matching_input_budget:{domain}:{count}"
                ],
            },
        },
    }


def _score_one(
    completion: Any,
    prepared: PreparedSourceContextV53,
    *,
    render_ok: bool | None,
    computed_registry: ComputedFunctionRegistryV53 | None,
    evidence_collector: Callable[..., Any] = collect_output_evidence_v5_3,
    type_contract_scorer: Callable[..., tuple[float, dict[str, Any]]] | None = None,
    semantic_role_scorer: Callable[..., Any] = semantic_role_coverage_v5_3,
    semantic_duplication_scorer: Callable[..., float] | None = None,
    json_efficiency_scorer: Callable[..., float] | None = None,
    include_accessibility_atomic: bool = True,
    normalization_result: Any = None,
    matching_timing_callback: Callable[[str, float], None] | None = None,
    unsupported_additions_scorer: Callable[..., Any] = unsupported_external_addition_precision_v5_3,
) -> RewardBreakdownV53:
    cfg = prepared.config
    normalization = (
        normalization_result
        if normalization_result is not None
        else normalize_and_validate_candidate(completion)
    )
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
    evidence_result = evidence_collector(
        spec,
        audit,
        cfg,
        computed_registry=computed_registry,
    )
    output = evidence_result.output
    source = prepared.source_contract
    if render_ok is None and cfg.render_check is not None:
        try:
            render_ok = bool(cfg.render_check(spec))
        except Exception:
            render_ok = False

    schema_score, schema_evidence = _core.basic_schema_contract(spec)
    if type_contract_scorer is None:
        type_score, type_evidence = _core.type_contract_score(spec, output)
    else:
        type_score, type_evidence = type_contract_scorer(
            spec, output, evidence_result, audit, cfg
        )
    generic_owner_names = {"generic_visible_content"}
    if not prepared.applicability.table_fidelity:
        generic_owner_names.add("source_table")
    if not (
        prepared.applicability.table_fidelity
        or prepared.applicability.chart_roles
    ):
        generic_owner_names.add("source_chart")
    if not prepared.applicability.semantic_roles:
        generic_owner_names.add("semantic_role")
    generic_output_blocks = tuple(
        prepare_text_block(str(item.get("text") or ""))
        for item in evidence_result.evidence_ownership
        if item.get("owner") in generic_owner_names
        and str(item.get("text") or "").strip()
    )
    content_input_count = (
        len(prepared.content_blocks) + len(generic_output_blocks)
    )
    if not prepared.applicability.generic_content:
        content_values = {
            "content_unit_fidelity": None,
            "content_order_preservation": None,
            "output_block_precision": None,
        }
        content_diagnostics = {
            "applicable": False,
            "ignored_candidate_count": len(generic_output_blocks),
        }
    elif content_input_count > cfg.max_matching_inputs:
        content_values = {
            "content_unit_fidelity": None,
            "content_order_preservation": None,
            "output_block_precision": None,
        }
        content_diagnostics = _matching_budget_detail(
            "content", content_input_count
        )
    else:
        matching_started = perf_counter()
        content_values, content_diagnostics = content_fidelity_v5_3(
            prepared.content_blocks,
            generic_output_blocks,
            expected_exact_index=prepared.content_exact_index,
            exact_dense_limit=cfg.max_assignment_size,
            max_edges=cfg.max_matching_edges,
            top_k=cfg.large_matching_top_k,
            max_hungarian_work=cfg.max_matching_hungarian_work,
            max_sparse_relaxations=(
                cfg.max_matching_sparse_relaxations
            ),
        )
        if matching_timing_callback is not None:
            matching_timing_callback(
                "content", (perf_counter() - matching_started) * 1000.0
            )
    content_precision, content_recall = _core.counter_pr(
        Counter(
            token
            for block in generic_output_blocks
            for token in block.tokens
        ),
        Counter(
            _core.tokenize(
                " ".join(prepared.applicability.generic_source_units)
            )
        ),
    )
    global_content = _core.f_beta(
        content_precision, content_recall, cfg.content_beta
    )
    generic_source_exact = _owned_exact_values(
        prepared.applicability.generic_source_units
    )
    generic_output_exact = _owned_exact_values(
        [
            block.raw
            for block in generic_output_blocks
        ]
    )
    value_precision, value_recall = _core.counter_pr(
        generic_output_exact, generic_source_exact
    )
    exact_value_score = (
        _core.f_beta(value_precision, value_recall, cfg.value_beta)
        if generic_source_exact
        else None
    )
    table_input_count = (
        len(source.tables)
        + len(evidence_result.output_tables)
        + len(evidence_result.output_charts)
    )
    if not prepared.applicability.table_fidelity:
        table_score, table_diagnostics = None, {
            "required_count": 0,
            "matched_required_count": 0,
            "ignored_candidate_count": (
                len(evidence_result.output_tables)
                + len(evidence_result.output_charts)
            ),
            "applicable": False,
        }
    elif table_input_count > cfg.max_matching_inputs:
        table_score, table_diagnostics = None, _matching_budget_detail(
            "tables", table_input_count
        )
        table_diagnostics.update(
            required_count=len(source.tables),
            matched_required_count=0,
        )
    else:
        matching_started = perf_counter()
        table_score, table_diagnostics = table_fidelity_v5_3(
            source.tables,
            evidence_result.output_tables,
            evidence_result.output_charts,
            beta=cfg.table_beta,
            exact_dense_limit=cfg.max_assignment_size,
            max_edges=cfg.max_matching_edges,
            top_k=cfg.large_matching_top_k,
            max_hungarian_work=cfg.max_matching_hungarian_work,
            max_sparse_relaxations=(
                cfg.max_matching_sparse_relaxations
            ),
        )
        if matching_timing_callback is not None:
            matching_timing_callback(
                "tables", (perf_counter() - matching_started) * 1000.0
            )
    explicitly_required_action_types = {
        item.action_type.casefold()
        for item in prepared.expected_actions
        if item.action_type.casefold() != "openurl"
    }
    applicable_output_actions = tuple(
        item
        for item in output.actions
        if item in evidence_result.source_semantic_actions
        or item.action_type.casefold() in explicitly_required_action_types
    )
    action_input_count = (
        len(prepared.expected_actions) + len(applicable_output_actions)
    )
    if not prepared.applicability.action_fidelity:
        action_score, action_diagnostics = None, {
            "required_count": 0,
            "matched_required_count": 0,
            "ignored_candidate_count": len(applicable_output_actions),
            "applicable": False,
        }
    elif action_input_count > cfg.max_matching_inputs:
        action_score, action_diagnostics = None, _matching_budget_detail(
            "actions", action_input_count
        )
        action_diagnostics.update(
            required_count=len(prepared.expected_actions),
            matched_required_count=0,
        )
    else:
        matching_started = perf_counter()
        action_score, action_diagnostics = action_fidelity_v5_3(
            prepared.expected_actions,
            applicable_output_actions,
            aliases=source.asset_aliases,
            expected_exact_index=prepared.action_exact_index,
            exact_dense_limit=cfg.max_assignment_size,
            max_edges=cfg.max_matching_edges,
            top_k=cfg.large_matching_top_k,
            max_hungarian_work=cfg.max_matching_hungarian_work,
            max_sparse_relaxations=(
                cfg.max_matching_sparse_relaxations
            ),
        )
        if matching_timing_callback is not None:
            matching_timing_callback(
                "actions", (perf_counter() - matching_started) * 1000.0
            )
    media_input_count = (
        len(prepared.expected_media)
        + len(evidence_result.source_semantic_media)
    )
    if not prepared.applicability.media_fidelity:
        media_score, media_diagnostics = None, {
            "required_count": 0,
            "matched_required_count": 0,
            "ignored_candidate_count": len(
                evidence_result.source_semantic_media
            ),
            "applicable": False,
        }
    elif media_input_count > cfg.max_matching_inputs:
        media_score, media_diagnostics = None, _matching_budget_detail(
            "media", media_input_count
        )
        media_diagnostics.update(
            required_count=len(prepared.expected_media),
            matched_required_count=0,
        )
    else:
        matching_started = perf_counter()
        media_score, media_diagnostics = media_fidelity_v5_3(
            prepared.expected_media,
            evidence_result.source_semantic_media,
            aliases=source.asset_aliases,
            expected_exact_index=prepared.media_exact_index,
            exact_dense_limit=cfg.max_assignment_size,
            max_edges=cfg.max_matching_edges,
            top_k=cfg.large_matching_top_k,
            max_hungarian_work=cfg.max_matching_hungarian_work,
            max_sparse_relaxations=(
                cfg.max_matching_sparse_relaxations
            ),
        )
        if matching_timing_callback is not None:
            matching_timing_callback(
                "media", (perf_counter() - matching_started) * 1000.0
            )
    role_requirements = _role_requirements(source)
    role_input_count = sum(
        len(values) for values in role_requirements.values()
    ) + sum(
        len(values) for values in evidence_result.role_instances.values()
    )
    if not prepared.applicability.semantic_roles:
        semantic_role_score = None
        special_count_score = None
        special_count_values = {}
        special_semantic_values = {}
        special_diagnostics = {}
        measured_specificity = 1.0
    elif role_input_count > cfg.max_matching_inputs:
        semantic_role_score = None
        special_count_score = None
        special_count_values = {
            role: None for role in role_requirements
        }
        special_semantic_values = dict(special_count_values)
        special_diagnostics = {
            "_budget": _matching_budget_detail(
                "semantic_roles", role_input_count
            )
        }
        measured_specificity = float(
            prepared.expected_contract.get(
                "contract_semantic_specificity", 0.0
            )
        )
    else:
        (
            semantic_role_score,
            special_count_score,
            special_count_values,
            special_semantic_values,
            special_diagnostics,
            measured_specificity,
        ) = semantic_role_scorer(
            role_requirements,
            evidence_result.role_instances,
            expected_exact_indices=prepared.role_exact_indices,
            threshold=cfg.role_match_threshold,
            exact_dense_limit=cfg.max_assignment_size,
            max_edges=cfg.max_matching_edges,
            top_k=cfg.large_matching_top_k,
            max_hungarian_work=cfg.max_matching_hungarian_work,
            max_sparse_relaxations=(
                cfg.max_matching_sparse_relaxations
            ),
        )
    if not prepared.applicability.media_fidelity:
        media_role_values = {
            "image": None,
            "video": None,
            "audio": None,
        }
        media_role_diagnostics = {"applicable": False}
    elif media_input_count > cfg.max_matching_inputs:
        media_role_values = {
            "image": None,
            "video": None,
            "audio": None,
        }
        media_role_diagnostics = {
            "_budget": _matching_budget_detail(
                "media_roles", media_input_count
            )
        }
    else:
        media_role_values, media_role_diagnostics = _media_role_values(
            prepared.expected_media,
            evidence_result.source_semantic_media,
            source.asset_aliases,
            prepared.media_exact_index,
            cfg,
        )
    (
        unsupported_addition_score,
        unsupported_addition_diagnostics,
    ) = unsupported_additions_scorer(
        prepared.expected_actions,
        output.actions,
        prepared.expected_media,
        output.media,
        aliases=source.asset_aliases,
    )
    if not prepared.applicability.unsupported_external_additions:
        unsupported_addition_score = None
    role_count_values: dict[str, float | None] = dict(
        special_count_values
    )
    role_count_values["table"] = _role_value(
        int(table_diagnostics.get("matched_required_count") or 0),
        int(table_diagnostics.get("required_count") or 0),
    )
    role_count_values["action"] = _role_value(
        int(action_diagnostics.get("matched_required_count") or 0),
        int(action_diagnostics.get("required_count") or 0),
    )
    role_count_values.update(media_role_values)
    role_count_score = _mean_applicable(role_count_values)
    if role_count_score is None:
        role_count_score = special_count_score
    contract_specificity = clamp01(
        float(
            prepared.expected_contract.get(
                "contract_semantic_specificity", measured_specificity
            )
        )
    )
    if (
        contract_specificity < 1.0
        and cfg.low_specificity_policy == "not_applicable"
    ):
        role_count_score = None
        semantic_role_score = None

    # Semantic requirements gate their own role. Count-only requirements use
    # count coverage, but are never presented as semantic-instance fidelity.
    role_gate_values = dict(role_count_values)
    for role, semantic_value in special_semantic_values.items():
        if semantic_value is not None:
            role_gate_values[role] = semantic_value

    accessibility_score, accessibility_evidence = (
        _core._accessibility_score(output)  # type: ignore[attr-defined]
    )
    semantic_duplication_score = (
        semantic_duplication_scorer(prepared, evidence_result, output)
        if semantic_duplication_scorer is not None
        else _core._semantic_duplicate_utility(  # type: ignore[attr-defined]
            output
        )
    )
    json_efficiency_score = (
        json_efficiency_scorer(
            spec, audit, source, cfg, evidence_result
        )
        if json_efficiency_scorer is not None
        else json_efficiency_utility_v5_1(spec, audit, source, cfg)
    )
    reachability = audit.reachable_fraction if audit.root_exists else 0.0
    atomics: dict[str, dict[str, float | None]] = {
        "integrity": {
            "production_validity": float(normalization.production_valid),
            "strict_schema_validity": float(
                normalization.strict_schema_valid
            ),
            "raw_format_utility": normalization.raw_format_utility,
            "schema_contract": schema_score,
            "declared_root_reachability": float(audit.root_exists),
            "reference_integrity": float(
                audit.root_exists and not audit.missing_references
            ),
            "cycle_free": float(
                audit.root_exists and not audit.cycle_edges
            ),
            "reachable_fraction": reachability,
            "type_semantic_contract": type_score,
        },
        "fidelity": {
            "content_unit_fidelity": (
                content_values["content_unit_fidelity"]
                if prepared.applicability.generic_content
                else None
            ),
            "content_order_preservation": (
                content_values["content_order_preservation"]
                if prepared.applicability.generic_content
                else None
            ),
            "visible_content_multiset_fbeta": (
                global_content
                if prepared.applicability.generic_content
                else None
            ),
            "exact_numbers_dates_units_fbeta": exact_value_score,
            "markdown_table_fidelity": table_score,
            "heading_fidelity_and_order": (
                _core.heading_fidelity(source, output)
                if prepared.applicability.headings
                else None
            ),
            "action_and_source_link_fidelity": (
                action_score
                if prepared.applicability.action_fidelity
                else None
            ),
            "media_fidelity": (
                media_score
                if prepared.applicability.media_fidelity
                else None
            ),
            "output_block_precision": (
                content_values["output_block_precision"]
                if prepared.applicability.generic_content
                else None
            ),
            "unsupported_external_addition_precision": (
                unsupported_addition_score
            ),
        },
        "semantic_mapping": {
            "role_count_coverage": role_count_score,
            "semantic_role_instance_fidelity": semantic_role_score,
            "component_appropriateness": (
                _core.component_appropriateness(source, output)
            ),
        },
        "hierarchy": {
            "root_layout": root_layout_utility_v5_1(spec, output),
            "root_distance_depth": _core._layout_depth_utility(  # type: ignore[attr-defined]
                audit
            ),
            "container_fanout": fanout_utility_v5_1(output),
            "heading_grouping": heading_grouping_utility_v5_1(output),
            "text_chunking": _structured_aware_chunking(
                prepared, generic_output_blocks
            ),
        },
        "economy": {
            "semantic_non_duplication": semantic_duplication_score,
            "renderer_aware_wrapper_economy": wrapper_economy_v5_1(
                output
            ),
            "json_to_source_size": json_efficiency_score,
        },
        "accessibility": {
            "applicable_accessibility_contracts": (
                accessibility_score if include_accessibility_atomic else None
            ),
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
    assert (
        math.isfinite(base_quality)
        and -1e-12 <= base_quality <= 1.0 + 1e-12
    )
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
        activate(
            "integrity", "production_invalid", list(normalization.errors)
        )
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
                "cycles": sorted(
                    [list(edge) for edge in audit.cycle_edges]
                ),
            },
        )
    if audit.root_exists and reachability < 0.90:
        activate("integrity", "reachability_below_90", reachability)
    elif audit.root_exists and reachability < 1.0:
        activate("integrity", "partial_reachability", reachability)
    if render_ok is False:
        activate("renderer", "render_failure")

    matching_certification = combine_certifications(
        {
            "content": content_diagnostics,
            "tables": table_diagnostics,
            "actions": action_diagnostics,
            "media": media_diagnostics,
            "media_roles": media_role_diagnostics,
            "semantic_roles": special_diagnostics,
        }
    )
    if (
        not bool(matching_certification.get("optimality_certified"))
        and cfg.matching_incomplete_policy == "fail_closed"
    ):
        activate(
            "matching",
            "matching_uncertified",
            {
                "diagnostic_codes": matching_certification.get(
                    "diagnostic_codes", []
                ),
                "lower_bound_weight": matching_certification.get(
                    "lower_bound_weight"
                ),
                "upper_bound_weight": matching_certification.get(
                    "upper_bound_weight"
                ),
            },
        )
    if not evidence_result.dynamic_evidence_complete:
        if cfg.dynamic_unknown_policy == "error":
            raise ValueError("dynamic renderer evidence is incomplete")
        if cfg.dynamic_unknown_policy == "cap":
            activate(
                "evidence",
                "dynamic_evidence_unknown",
                list(evidence_result.unknown_diagnostics),
            )
    if (
        contract_specificity < 1.0
        and cfg.low_specificity_policy == "fail_closed"
    ):
        activate(
            "contract",
            "low_contract_specificity",
            contract_specificity,
        )

    for role, value in role_gate_values.items():
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
            activate(
                "semantic", "missing_special_role", {"role": role}
            )
        elif value < 1.0:
            activate(
                "semantic", "partial_special_role", {"role": role}
            )

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
    registry = prepared.computed_registry_identity.get("custom")
    if not isinstance(registry, Mapping):
        registry = prepared.computed_registry_identity.get("builtin")
    registry = registry if isinstance(registry, Mapping) else {}
    identity = {
        "source_hash": prepared.source_hash,
        "raw_candidate_hash": normalization.raw_hash,
        "canonical_candidate_hash": normalization.canonical_hash,
        "expected_contract_hash": prepared.contract_hash,
        "computed_registry_id": str(registry.get("registry_id") or ""),
        "computed_registry_version": str(
            registry.get("registry_version") or ""
        ),
        "computed_registry_manifest_hash": str(
            registry.get("manifest_hash") or ""
        ),
        "computed_registry_hash": str(
            prepared.computed_registry_identity.get(
                "custom_identity_hash"
            )
            or prepared.computed_registry_identity.get(
                "builtin_identity_hash"
            )
            or ""
        ),
    }
    dimensions = _weighted_dimension_scores(atomics, cfg)
    errors = list(
        dict.fromkeys(
            [
                *prepared.contract_resolution.errors,
                *normalization.errors,
                *evidence_result.unknown_diagnostics,
                *[
                    str(code)
                    for code in matching_certification.get(
                        "diagnostic_codes", []
                    )
                    if not bool(
                        matching_certification.get(
                            "optimality_certified"
                        )
                    )
                ],
            ]
        )
    )
    count_only_role_count = int(
        prepared.expected_contract.get("count_only_role_count")
        or sum(
            int(detail.get("required_count") or 0)
            if isinstance(detail, Mapping) and detail.get("count_only")
            else 0
            for detail in special_diagnostics.values()
        )
    )
    dynamic_semantics = dict(evidence_result.dynamic_semantics)
    atomic_applicability = {
        f"{dimension}.{atomic}": value is not None
        for dimension, values in atomics.items()
        for atomic, value in values.items()
    }
    atomic_applicability.update(prepared.applicability.atomic_mapping())
    evidence_ownership = {
        "policy": dict(prepared.applicability.evidence_ownership),
        "candidate_blocks": [
            dict(item) for item in evidence_result.evidence_ownership
        ],
        "action_taxonomy": [
            dict(item) for item in evidence_result.action_taxonomy
        ],
        "media_taxonomy": [
            dict(item) for item in evidence_result.media_taxonomy
        ],
    }
    return RewardBreakdownV53(
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
                    "contract_version", V5_3_CONTRACT_VERSION
                ),
                "source_hash": prepared.source_hash,
                "content_unit_count": len(source.content_units),
                "contract_semantic_specificity": contract_specificity,
                "count_only_role_count": count_only_role_count,
                "extraction_diagnostics": prepared.expected_contract.get(
                    "extraction_diagnostics", []
                ),
                "contract_migration": prepared.expected_contract.get(
                    "contract_migration"
                ),
                "prepared_exact_indices": {
                    "content_keys": len(prepared.content_exact_index),
                    "action_keys": len(prepared.action_exact_index),
                    "media_keys": len(prepared.media_exact_index),
                    "role_keys": {
                        role: len(values)
                        for role, values in prepared.role_exact_indices.items()
                    },
                },
            },
            "output": {
                "root_exists": audit.root_exists,
                "reachable_elements": len(audit.reachable_ids),
                "unreachable_elements": len(audit.unreachable_ids),
                "reachable_ids": sorted(audit.reachable_ids),
                "unreachable_ids": sorted(audit.unreachable_ids),
                "missing_references": sorted(audit.missing_references),
                "cycles": sorted(
                    [list(edge) for edge in audit.cycle_edges]
                ),
                "output_tables": len(evidence_result.output_tables),
                "output_charts": len(evidence_result.output_charts),
                "structured_data_payloads": len(
                    evidence_result.structured_data_payloads
                ),
                "expanded_evidence_nodes": (
                    evidence_result.expanded_evidence_nodes
                ),
                "evidence_truncated": evidence_result.truncated,
                "dynamic_expression_unknown_count": (
                    evidence_result.dynamic_expression_unknown_count
                ),
                "dynamic_expression_unknown_codes": list(
                    evidence_result.unknown_diagnostics
                ),
                "dynamic_evidence_complete": (
                    evidence_result.dynamic_evidence_complete
                ),
            },
            "schema": schema_evidence,
            "type_contract": type_evidence,
            "content_assignment": content_diagnostics,
            "table_matching": table_diagnostics,
            "action_matching": action_diagnostics,
            "media_matching": media_diagnostics,
            "role_matching": special_diagnostics,
            "role_count_values": role_count_values,
            "role_semantic_values": special_semantic_values,
            "role_gate_values": role_gate_values,
            "role_instances": evidence_result.role_instances,
            "accessibility": accessibility_evidence,
            "matching_certification": matching_certification,
            "dynamic_semantics": dynamic_semantics,
            "dynamic_evidence_certification": (
                evidence_result.dynamic_evidence_certification
            ),
            "atomic_applicability": atomic_applicability,
            "evidence_ownership": evidence_ownership,
            "unsupported_external_additions": (
                unsupported_addition_diagnostics
            ),
            "render_ok": render_ok,
        },
        metric_version=REWARD_VERSION_V53,
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
        matching_certification=matching_certification,
        dynamic_semantics=dynamic_semantics,
        atomic_applicability=atomic_applicability,
        evidence_ownership=evidence_ownership,
        unsupported_external_additions=unsupported_addition_diagnostics,
        contract_semantic_specificity=contract_specificity,
        count_only_role_count=count_only_role_count,
        dynamic_evidence_certification=(
            evidence_result.dynamic_evidence_certification
        ),
        reward_pipeline_fingerprint=prepared.reward_pipeline_fingerprint,
    )


def score_completion_group_v5_3(
    completions: Sequence[Any],
    prepared_source: PreparedSourceContextV53,
    *,
    render_results: Sequence[bool | None] | None = None,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> tuple[RewardBreakdownV53, ...]:
    """Score a group without repeating source extraction or source indices."""

    registry_identity = effective_registry_identity_v5_3(computed_registry)
    if registry_identity != prepared_source.computed_registry_identity:
        raise ValueError(
            "computed_registry identity differs from prepared source context"
        )
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
            computed_registry=computed_registry,
        )
        for index, completion in enumerate(completions)
    )


def score_genui_completion_v5_3(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    expected_ui_contract_source: str | None = None,
    render_ok: bool | None = None,
    config: RewardConfigV53 | None = None,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> RewardBreakdownV53:
    prepared = prepare_source_context_v5_3(
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=expected_ui_contract,
        expected_ui_contract_source=expected_ui_contract_source,
        config=config,
        computed_registry=computed_registry,
    )
    return score_completion_group_v5_3(
        [completion],
        prepared,
        render_results=[render_ok],
        computed_registry=computed_registry,
    )[0]


__all__ = [
    "PreparedSourceContextV53",
    "prepare_source_context_v5_3",
    "score_completion_group_v5_3",
    "score_genui_completion_v5_3",
]
