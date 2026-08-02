"""Production-aligned GenUI Representation Quality v5.4 scoring path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from time import perf_counter
import tracemalloc
from typing import Any, Mapping, Sequence

from ..renderer_effective_semantics_v5_4 import (
    COMPONENT_CONTRACT_POLICY_VERSION,
    RENDERER_EFFECTIVE_SEMANTICS_VERSION,
    renderer_effective_size_utility,
    renderer_effective_type_contract,
)
from ._v5_3 import (
    PreparedSourceContextV53,
    _score_one,
    prepare_source_context_v5_3,
)
from .candidate_normalization_v5_4 import (
    NORMALIZATION_POLICY_VERSION_V54,
    RAW_ENVELOPE_POLICY_VERSION,
    normalize_and_validate_candidate_v5_4,
    normalize_and_validate_express_candidate_v5_4,
)
from .config_v5_4 import (
    REWARD_VERSION_V54,
    RewardBreakdownV54,
    RewardConfigV54,
    coerce_reward_config_v5_4,
)
from .evidence_v5_4 import (
    COMPUTED_REGISTRY_IDENTITY_VERSION,
    DYNAMIC_PARITY_VECTOR_VERSION,
    DYNAMIC_SEMANTICS_VERSION,
    EVIDENCE_POLICY_VERSION,
    ComputedFunctionRegistryV54,
    collect_output_evidence_v5_4,
    effective_registry_identity_v5_4,
)
from .identity_v5_4 import (
    METRIC_NAME,
    expected_contract_hash_v5_4,
    metric_fingerprint_v5_4,
    reward_pipeline_fingerprint_v5_4,
)
from .metrics_v5_4 import (
    ROLE_MATCHING_POLICY_VERSION,
    SCORING_POLICY_VERSION,
    semantic_duplication_score_v5_4,
    semantic_role_coverage_v5_4,
)
from .ownership_v5_4 import (
    OWNERSHIP_POLICY_VERSION,
    SemanticEvidenceUnit,
    build_output_ownership,
    build_source_ownership,
    cross_channel_duplication_utility,
)
from .source_contract import (
    ContractResolution,
    source_text_hash,
)
from .source_contract_v5_3 import (
    V5_3_CONTRACT_VERSION,
    V5_3_EXTRACTOR_VERSION,
    V5_3_EXTRACTION_POLICY_VERSION,
    source_contract_cache_key_v5_3,
)
from .source_contract_v5_4 import (
    V5_4_CONTRACT_VERSION,
    V5_4_EXTRACTOR_VERSION,
    V5_4_EXTRACTION_POLICY_VERSION,
    resolve_expected_ui_contract_v5_4,
)
from .validation_v5_4 import ensure_v5_4_validation_ready


ACCESSIBILITY_POLICY_VERSION = "penalty-only-1.0.0"
FORMAT_APPLICATION_POLICY_VERSION = "generation-only-1.0.0"
_BINDING_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PreparedSourceContextV54:
    source_text: str
    intent: str | None
    assets: Any
    expected_contract: Mapping[str, Any]
    contract_resolution: ContractResolution
    source_ownership: tuple[SemanticEvidenceUnit, ...]
    base: PreparedSourceContextV53
    config: RewardConfigV54
    source_hash: str
    contract_hash: str
    metric_fingerprint: str
    reward_pipeline_fingerprint: str
    computed_registry_identity: Mapping[str, Any]
    preparation_time_ms: float


def _v5_3_compat_contract(
    contract: Mapping[str, Any],
    *,
    source_text: str,
    intent: str | None,
    assets: Any,
) -> dict[str, Any]:
    result = {
        str(key): value
        for key, value in contract.items()
        if key not in {"contract_migration"}
    }
    result["contract_version"] = V5_3_CONTRACT_VERSION
    result["extractor_version"] = V5_3_EXTRACTOR_VERSION
    result["extraction_policy_version"] = V5_3_EXTRACTION_POLICY_VERSION
    result["source_hash"] = source_text_hash(source_text)
    result["cache_key"] = source_contract_cache_key_v5_3(
        source_text, intent=intent, assets=assets
    )
    return result


def prepare_source_context_v5_4(
    source_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    expected_ui_contract_source: str | None = None,
    config: RewardConfigV54 | None = None,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
) -> PreparedSourceContextV54:
    started = perf_counter()
    ensure_v5_4_validation_ready()
    cfg = coerce_reward_config_v5_4(config)
    resolution = resolve_expected_ui_contract_v5_4(
        source_text,
        intent=intent,
        assets=assets,
        persisted=expected_ui_contract,
        persisted_source=expected_ui_contract_source,
    )
    ownership = build_source_ownership(source_text, resolution.contract)
    compat = _v5_3_compat_contract(
        resolution.contract,
        source_text=source_text,
        intent=intent,
        assets=assets,
    )
    base = prepare_source_context_v5_3(
        source_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=compat,
        expected_ui_contract_source="persisted",
        config=cfg,
        computed_registry=computed_registry,
    )
    fingerprint = metric_fingerprint_v5_4(
        cfg, computed_registry=computed_registry
    )
    return PreparedSourceContextV54(
        source_text=str(source_text or ""),
        intent=intent,
        assets=assets,
        expected_contract=resolution.contract,
        contract_resolution=resolution,
        source_ownership=ownership,
        base=base,
        config=cfg,
        source_hash=str(
            resolution.contract.get("source_hash")
            or source_text_hash(source_text)
        ),
        contract_hash=expected_contract_hash_v5_4(resolution.contract),
        metric_fingerprint=fingerprint,
        reward_pipeline_fingerprint=reward_pipeline_fingerprint_v5_4(
            fingerprint
        ),
        computed_registry_identity=effective_registry_identity_v5_4(
            computed_registry
        ),
        preparation_time_ms=(perf_counter() - started) * 1000.0,
    )


def _raw_utility(
    normalization: Any,
    config: RewardConfigV54,
) -> float:
    envelope = normalization.raw_envelope
    if envelope.exact_single_json_value:
        return 1.0
    values = [1.0]
    if envelope.markdown_fence_present:
        values.append(config.raw_fence_utility)
    if envelope.leading_non_whitespace or envelope.trailing_non_whitespace:
        values.append(config.raw_wrapper_utility)
    if envelope.extra_json_value_present:
        values.append(config.raw_extra_json_utility)
    return min(values)


def _accessibility_adjustment(
    base: RewardBreakdownV54 | Any,
    config: RewardConfigV54,
) -> tuple[float, dict[str, Any]]:
    raw = base.evidence.get("accessibility")
    values = raw if isinstance(raw, Mapping) else {}
    applicable = [
        float(value)
        for value in values.values()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    conformance = (
        sum(applicable) / len(applicable) if applicable else 1.0
    )
    multiplier = 1.0 - config.accessibility_penalty_alpha * (
        1.0 - conformance
    )
    return multiplier, {
        "policy_version": ACCESSIBILITY_POLICY_VERSION,
        "penalty_only": True,
        "applicable_check_count": len(applicable),
        "conformance": conformance,
        "alpha": config.accessibility_penalty_alpha,
        "multiplier": multiplier,
        "checks": dict(values),
    }


def _base_payload(base: Any) -> dict[str, Any]:
    return {
        item.name: getattr(base, item.name)
        for item in fields(type(base))
        if item.name
        in {field.name for field in fields(RewardBreakdownV54)}
    }


def _score_one_v5_4(
    completion: Any,
    prepared: PreparedSourceContextV54,
    *,
    render_ok: bool | None,
    computed_registry: ComputedFunctionRegistryV54 | None,
    generation_mode: bool,
    active_express: bool = False,
) -> RewardBreakdownV54:
    started = perf_counter()
    timings: dict[str, float | int | bool | None] = {
        "source_preparation_ms": prepared.preparation_time_ms,
        "source_contract_cache_hit": prepared.contract_resolution.cache_hit,
        "source_contract_cache_hit_rate": (
            1.0 if prepared.contract_resolution.cache_hit else 0.0
        ),
    }
    ownership_detail: dict[str, Any] = {}
    normalization = (
        normalize_and_validate_express_candidate_v5_4(completion)
        if active_express
        else normalize_and_validate_candidate_v5_4(completion)
    )
    matching_times: dict[str, float] = {}

    def collect(*args: Any, **kwargs: Any) -> Any:
        evidence_started = perf_counter()
        result = collect_output_evidence_v5_4(*args, **kwargs)
        timings["dynamic_evidence_ms"] = (
            perf_counter() - evidence_started
        ) * 1000.0
        return result

    def roles(*args: Any, **kwargs: Any) -> Any:
        matching_started = perf_counter()
        result = semantic_role_coverage_v5_4(*args, **kwargs)
        timings["semantic_role_matching_ms"] = (
            perf_counter() - matching_started
        ) * 1000.0
        matching_times["semantic_roles"] = float(
            timings["semantic_role_matching_ms"]
        )
        return result

    def matching_timing(domain: str, elapsed_ms: float) -> None:
        matching_times[domain] = float(elapsed_ms)

    def duplication(
        base_prepared: Any, evidence_result: Any, output: Any
    ) -> float:
        ownership_started = perf_counter()
        output_units = build_output_ownership(evidence_result)
        utility, detail = cross_channel_duplication_utility(
            prepared.source_ownership, output_units
        )
        historical = semantic_duplication_score_v5_4(
            base_prepared,
            evidence_result,
            output,
            cross_channel_utility=utility,
        )
        ownership_detail.update(detail)
        diagnostic_limit = 256
        ownership_detail["source_units"] = [
            asdict(item)
            for item in prepared.source_ownership[:diagnostic_limit]
        ]
        ownership_detail["output_units"] = [
            asdict(item) for item in output_units[:diagnostic_limit]
        ]
        ownership_detail["source_unit_count"] = len(
            prepared.source_ownership
        )
        ownership_detail["output_unit_count"] = len(output_units)
        ownership_detail["diagnostic_units_truncated"] = (
            len(prepared.source_ownership) > diagnostic_limit
            or len(output_units) > diagnostic_limit
        )
        timings["ownership_ms"] = (
            perf_counter() - ownership_started
        ) * 1000.0
        return min(utility, historical)

    if render_ok is None and prepared.config.render_check is not None:
        if normalization.canonical_spec is not None:
            try:
                render_ok = bool(
                    prepared.config.render_check(
                        normalization.canonical_spec
                    )
                )
            except Exception:
                render_ok = False

    base = _score_one(
        completion,
        prepared.base,
        render_ok=render_ok,
        computed_registry=computed_registry,
        evidence_collector=collect,
        type_contract_scorer=renderer_effective_type_contract,
        semantic_role_scorer=roles,
        semantic_duplication_scorer=duplication,
        json_efficiency_scorer=renderer_effective_size_utility,
        include_accessibility_atomic=False,
        normalization_result=normalization.boundary,
        matching_timing_callback=matching_timing,
    )
    accessibility_multiplier, accessibility = _accessibility_adjustment(
        base, prepared.config
    )
    artifact_base = max(
        0.0,
        min(1.0, base.base_quality_before_caps * accessibility_multiplier),
    )
    artifact_quality = max(0.0, min(artifact_base, base.cap_0_1))
    raw_utility = (
        _raw_utility(normalization, prepared.config)
        if generation_mode
        else 1.0
    )
    final_base = artifact_base * raw_utility
    quality = max(0.0, min(final_base, base.cap_0_1))
    reward = 2.0 * quality - 1.0
    assert all(
        math.isfinite(value)
        for value in (artifact_base, artifact_quality, final_base, quality, reward)
    )

    envelope = {
        **asdict(normalization.raw_envelope),
        "policy_version": RAW_ENVELOPE_POLICY_VERSION,
        "application_policy": FORMAT_APPLICATION_POLICY_VERSION,
        "generation_mode": generation_mode,
        "utility": raw_utility,
    }
    evidence = dict(base.evidence)
    evidence["source"] = {
        **(
            dict(evidence.get("source") or {})
            if isinstance(evidence.get("source"), Mapping)
            else {}
        ),
        "contract_source": prepared.contract_resolution.source,
        "contract_version": V5_4_CONTRACT_VERSION,
        "extractor_version": V5_4_EXTRACTOR_VERSION,
        "extraction_policy_version": V5_4_EXTRACTION_POLICY_VERSION,
        "source_hash": prepared.source_hash,
        "cache_hit": prepared.contract_resolution.cache_hit,
    }
    evidence["evidence_ownership"] = ownership_detail
    evidence["accessibility_conformance"] = accessibility
    evidence["raw_express_envelope"] = envelope
    evidence["raw_json_envelope"] = envelope
    evidence["base_quality_before_caps"] = base.base_quality_before_caps
    evidence["quality_after_accessibility_before_caps"] = artifact_base
    evidence["generation_quality_before_caps"] = final_base
    evidence["artifact_quality_0_1"] = artifact_quality
    evidence["generation_mode"] = generation_mode
    performance = {
        **timings,
        "scalar_reward_ms": (perf_counter() - started) * 1000.0,
        "matching_time_by_domain_ms": {
            **matching_times,
            "semantic_roles": matching_times.get("semantic_roles", 0.0),
        },
        "peak_memory_bytes": (
            tracemalloc.get_traced_memory()[1]
            if tracemalloc.is_tracing()
            else None
        ),
        "budget_exhaustion": bool(
            base.dynamic_evidence_certification.get(
                "diagnostic_codes"
            )
        ),
    }
    evidence["performance"] = performance

    atomics = {
        dimension: dict(values)
        for dimension, values in base.atomics.items()
    }
    atomics.setdefault("accessibility", {})[
        "applicable_accessibility_contracts"
    ] = None
    dimensions = dict(base.dimensions)
    dimensions["accessibility"] = float(
        accessibility["conformance"]
    )
    normalization_payload = dict(base.normalization)
    normalization_payload.update(
        {
            "native_syntax_valid": bool(normalization.raw_valid) if active_express else bool(normalization.raw_parse_ok),
            "native_catalog_valid": bool(normalization.production_valid) if active_express else bool(normalization.production_valid),
            "repaired_syntax_valid": False,
            "repair_applied": False,
            "standard_a2ui_valid": bool(normalization.production_valid) if active_express else None,
        }
    )
    normalization_payload["raw_express_envelope"] = envelope
    normalization_payload["raw_json_envelope"] = envelope
    identity = {
        **dict(base.identity),
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
    }
    binding_caps = [
        dict(item)
        for item in base.active_caps
        if final_base
        > float(item.get("cap", 1.0)) + _BINDING_TOLERANCE
    ]
    payload = _base_payload(base)
    payload.update(
        reward=reward,
        quality_0_100=100.0 * quality,
        quality_0_1=quality,
        cap_0_1=base.cap_0_1,
        dimensions=dimensions,
        atomics=atomics,
        evidence=evidence,
        metric_version=REWARD_VERSION_V54,
        metric_name=METRIC_NAME,
        metric_fingerprint=prepared.metric_fingerprint,
        base_quality_before_caps=base.base_quality_before_caps,
        normalization=normalization_payload,
        identity=identity,
        binding_caps=binding_caps,
        cap_margin=final_base - base.cap_0_1,
        evidence_ownership=ownership_detail,
        reward_pipeline_fingerprint=(
            prepared.reward_pipeline_fingerprint
        ),
        artifact_quality_0_1=artifact_quality,
        artifact_quality_0_100=100.0 * artifact_quality,
        raw_json_envelope=envelope,
        accessibility_conformance=accessibility,
        policy_versions={
            "ownership": OWNERSHIP_POLICY_VERSION,
            "renderer_effective_semantics": (
                RENDERER_EFFECTIVE_SEMANTICS_VERSION
            ),
            "component_contracts": COMPONENT_CONTRACT_POLICY_VERSION,
            "role_matching": ROLE_MATCHING_POLICY_VERSION,
            "scoring": SCORING_POLICY_VERSION,
            "dynamic_semantics": DYNAMIC_SEMANTICS_VERSION,
            "dynamic_parity": DYNAMIC_PARITY_VECTOR_VERSION,
            "evidence": EVIDENCE_POLICY_VERSION,
            "raw_envelope": RAW_ENVELOPE_POLICY_VERSION,
            "raw_format_application": FORMAT_APPLICATION_POLICY_VERSION,
            "normalization": NORMALIZATION_POLICY_VERSION_V54,
            "accessibility": ACCESSIBILITY_POLICY_VERSION,
            "source_contract": V5_4_CONTRACT_VERSION,
            "source_extractor": V5_4_EXTRACTOR_VERSION,
            "source_extraction": V5_4_EXTRACTION_POLICY_VERSION,
            "computed_registry_identity": (
                COMPUTED_REGISTRY_IDENTITY_VERSION
            ),
        },
        performance=performance,
    )
    return RewardBreakdownV54(**payload)


def score_completion_group_v5_4(
    completions: Sequence[Any],
    prepared_source: PreparedSourceContextV54,
    *,
    render_results: Sequence[bool | None] | None = None,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
    generation_mode: bool = False,
    active_express: bool = False,
) -> tuple[RewardBreakdownV54, ...]:
    if (
        effective_registry_identity_v5_4(computed_registry)
        != prepared_source.computed_registry_identity
    ):
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
    group_started = perf_counter()
    results = tuple(
        _score_one_v5_4(
            completion,
            prepared_source,
            render_ok=renders[index],
            computed_registry=computed_registry,
            generation_mode=generation_mode,
            active_express=active_express,
        )
        for index, completion in enumerate(completions)
    )
    group_ms = (perf_counter() - group_started) * 1000.0
    if len(completions) == 8:
        for result in results:
            result.performance["group_of_eight_reward_ms"] = group_ms
            result.evidence["performance"][
                "group_of_eight_reward_ms"
            ] = group_ms
    return results


def _score(
    completion: Any,
    response_text: str,
    *,
    intent: str | None,
    assets: Any,
    expected_ui_contract: Mapping[str, Any] | None,
    expected_ui_contract_source: str | None,
    render_ok: bool | None,
    config: RewardConfigV54 | None,
    computed_registry: ComputedFunctionRegistryV54 | None,
    generation_mode: bool,
    active_express: bool = False,
) -> RewardBreakdownV54:
    prepared = prepare_source_context_v5_4(
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=expected_ui_contract,
        expected_ui_contract_source=expected_ui_contract_source,
        config=config,
        computed_registry=computed_registry,
    )
    return score_completion_group_v5_4(
        [completion],
        prepared,
        render_results=[render_ok],
        computed_registry=computed_registry,
        generation_mode=generation_mode,
        active_express=active_express,
    )[0]


def render_artifact_quality_v5_4(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    expected_ui_contract_source: str | None = None,
    render_ok: bool | None = None,
    config: RewardConfigV54 | None = None,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
) -> RewardBreakdownV54:
    return _score(
        completion,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=expected_ui_contract,
        expected_ui_contract_source=expected_ui_contract_source,
        render_ok=render_ok,
        config=config,
        computed_registry=computed_registry,
        generation_mode=False,
    )


def generation_reward_v5_4(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    expected_ui_contract_source: str | None = None,
    render_ok: bool | None = None,
    config: RewardConfigV54 | None = None,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
) -> RewardBreakdownV54:
    return _score(
        completion,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=expected_ui_contract,
        expected_ui_contract_source=expected_ui_contract_source,
        render_ok=render_ok,
        config=config,
        computed_registry=computed_registry,
        generation_mode=True,
    )


def generation_reward_a2ui_express_v1(
    completion: Any,
    response_text: str,
    **kwargs: Any,
) -> RewardBreakdownV54:
    """Strict generation reward used by active GRPO/inference evaluation."""
    return _score(
        completion,
        response_text,
        intent=kwargs.pop("intent", None),
        assets=kwargs.pop("assets", None),
        expected_ui_contract=kwargs.pop("expected_ui_contract", None),
        expected_ui_contract_source=kwargs.pop("expected_ui_contract_source", None),
        render_ok=kwargs.pop("render_ok", None),
        config=kwargs.pop("config", None),
        computed_registry=kwargs.pop("computed_registry", None),
        generation_mode=True,
        active_express=True,
    )


score_genui_completion_v5_4 = render_artifact_quality_v5_4
genui_quality_v5_4 = render_artifact_quality_v5_4


__all__ = [
    "ACCESSIBILITY_POLICY_VERSION",
    "FORMAT_APPLICATION_POLICY_VERSION",
    "PreparedSourceContextV54",
    "generation_reward_v5_4",
    "generation_reward_a2ui_express_v1",
    "genui_quality_v5_4",
    "prepare_source_context_v5_4",
    "render_artifact_quality_v5_4",
    "score_completion_group_v5_4",
    "score_genui_completion_v5_4",
]
