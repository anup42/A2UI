"""Single-sample scoring, safe reuse, and aggregation for metric v5."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import math
import random
import statistics
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from . import _core
from ._v5 import score_genui_completion_v5
from ._v5_1 import score_genui_completion_v5_1 as _score_genui_completion_v5_1
from ._v5_2 import score_genui_completion_v5_2 as _score_genui_completion_v5_2
from ._v5_3 import score_genui_completion_v5_3 as _score_genui_completion_v5_3
from .candidate_normalization import normalize_and_validate_candidate
from .evidence_v5_2 import ComputedFunctionRegistryV52
from .evidence_v5_3 import ComputedFunctionRegistryV53
from .config_v5_3 import (
    REWARD_VERSION_V53,
    RewardBreakdownV53,
    RewardConfigV53,
    load_v5_3_reward_config,
)
from .config import (
    REWARD_VERSION,
    V5_1_REWARD_VERSION,
    V5_REWARD_VERSION,
    V4_REWARD_VERSION,
    RewardConfig,
    load_default_reward_config,
    load_v5_1_reward_config,
    load_v5_reward_config,
)
from .identity import (
    METRIC_NAME,
    expected_contract_hash,
    metric_fingerprint as metric_fingerprint_v5_0,
)
from .identity_v5_1 import (
    expected_contract_hash_v5_1,
    metric_fingerprint_v5_1,
)
from .identity_v5_2 import (
    expected_contract_hash_v5_2,
    metric_fingerprint_v5_2,
)
from .identity_v5_3 import (
    expected_contract_hash_v5_3,
    metric_fingerprint_v5_3,
    reward_pipeline_fingerprint_v5_3,
)
from .source_contract import (
    CONTRACT_VERSION,
    resolve_expected_ui_contract,
    resolve_expected_ui_contract_v5_1,
    resolve_expected_ui_contract_v5_2,
    source_contract_cache_key,
)
from .source_contract_v5_3 import resolve_expected_ui_contract_v5_3
from .validation_v5_1 import MetricV51InitializationError
from .validation_v5_2 import MetricV52InitializationError
from .validation_v5_3 import MetricV53InitializationError


RewardBreakdown = _core.RewardBreakdown


@runtime_checkable
class RendererSmokeAdapter(Protocol):
    """Adapter boundary for an attempted Android-native render smoke check."""

    def render_smoke(self, spec: Mapping[str, Any]) -> bool:
        """Return True only when the native renderer consumed the candidate."""


def with_renderer_smoke_adapter(
    config: RewardConfig,
    adapter: RendererSmokeAdapter,
) -> RewardConfig:
    """Return a validated config bound to a renderer adapter."""
    return replace(config, render_check=adapter.render_smoke)


def score_genui_completion(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    render_ok: bool | None = None,
    config: RewardConfigV53 | None = None,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> RewardBreakdownV53:
    """Score one raw completion through the shared current v5.3 boundary."""
    try:
        return _score_genui_completion_v5_3(
            completion,
            response_text,
            intent=intent,
            assets=assets,
            expected_ui_contract=expected_ui_contract,
            render_ok=render_ok,
            config=config,
            computed_registry=computed_registry,
        )
    except MetricV53InitializationError:
        raise
    except Exception as exc:
        q = 0.0
        cfg = config or load_v5_3_reward_config()
        from .identity_v5_3 import (
            metric_fingerprint_v5_3,
            reward_pipeline_fingerprint_v5_3,
        )

        fingerprint = metric_fingerprint_v5_3(
            cfg, computed_registry=computed_registry
        )
        return RewardBreakdownV53(
            reward=2.0 * q - 1.0,
            quality_0_100=100.0 * q,
            quality_0_1=q,
            cap_0_1=0.0,
            parse_stage="metric_exception",
            dimensions={},
            atomics={},
            evidence={"source": {"contract_source": "error"}},
            metric_version=REWARD_VERSION_V53,
            active_caps=[{"kind": "integrity", "name": "metric_exception", "cap": 0.0}],
            errors=[f"{type(exc).__name__}: {exc}"],
            metric_name=METRIC_NAME,
            metric_fingerprint=fingerprint,
            base_quality_before_caps=0.0,
            anti_domination_feasible=False,
            binding_caps=[
                {"kind": "integrity", "name": "metric_exception", "cap": 0.0}
            ],
            atomic_applicability={},
            evidence_ownership={},
            unsupported_external_additions={},
            dynamic_evidence_certification={
                "complete": False,
                "diagnostic_codes": ["metric_exception"],
            },
            reward_pipeline_fingerprint=reward_pipeline_fingerprint_v5_3(
                fingerprint
            ),
        )


def score_genui_completion_v5_2(
    completion: Any,
    response_text: str,
    **kwargs: Any,
) -> RewardBreakdown:
    """Explicit immutable v5.2 scorer retained for comparison."""

    return _score_genui_completion_v5_2(
        completion,
        response_text,
        **kwargs,
    )


score_genui_completion_v5_3 = score_genui_completion


def score_genui_completion_v5_1(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    render_ok: bool | None = None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Explicit immutable v5.1 scorer retained for historical comparison."""
    return _score_genui_completion_v5_1(
        completion,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=expected_ui_contract,
        render_ok=render_ok,
        config=config or load_v5_1_reward_config(),
    )


def score_genui_completion_v5_0(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    render_ok: bool | None = None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Explicit retained v5.0 scorer for immutable historical comparison."""

    return score_genui_completion_v5(
        completion,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=expected_ui_contract,
        render_ok=render_ok,
        config=config or load_v5_reward_config(),
    )


def score_genui_completion_v4(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    render_ok: bool | None = None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Explicit retained v4 scorer; never interpret its output as v5."""
    result = _core.score_genui_completion(
        completion,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=expected_ui_contract,
        render_ok=render_ok,
        config=config,
    )
    source_evidence = result.evidence.setdefault("source", {})
    if isinstance(source_evidence, dict):
        source_evidence["contract_source"] = (
            "persisted" if isinstance(expected_ui_contract, Mapping) else "deterministic fallback"
        )
        source_evidence["source_hash"] = str(
            (expected_ui_contract or {}).get("source_hash")
            or source_contract_cache_key(response_text, intent=intent, assets=assets)
        )
    return result


def breakdown_to_mapping(result: RewardBreakdown) -> dict[str, Any]:
    return asdict(result)


def breakdown_from_mapping(value: Any) -> RewardBreakdown | None:
    if not isinstance(value, Mapping):
        return None
    try:
        result_type: Any = (
            RewardBreakdownV53
            if str(value.get("metric_version") or "") == REWARD_VERSION_V53
            else RewardBreakdown
        )
        v53_fields = (
            {
                "atomic_applicability": dict(
                    value.get("atomic_applicability") or {}
                ),
                "evidence_ownership": dict(
                    value.get("evidence_ownership") or {}
                ),
                "unsupported_external_additions": dict(
                    value.get("unsupported_external_additions") or {}
                ),
                "contract_semantic_specificity": float(
                    value.get("contract_semantic_specificity") or 0.0
                ),
                "count_only_role_count": int(
                    value.get("count_only_role_count") or 0
                ),
                "dynamic_evidence_certification": dict(
                    value.get("dynamic_evidence_certification") or {}
                ),
                "reward_pipeline_fingerprint": str(
                    value.get("reward_pipeline_fingerprint") or ""
                ),
            }
            if result_type is RewardBreakdownV53
            else {}
        )
        result = result_type(
            reward=float(value["reward"]),
            quality_0_100=float(value["quality_0_100"]),
            quality_0_1=float(value["quality_0_1"]),
            cap_0_1=float(value["cap_0_1"]),
            parse_stage=str(value["parse_stage"]),
            dimensions={
                str(key): None if score is None else float(score)
                for key, score in dict(value.get("dimensions") or {}).items()
            },
            atomics={
                str(group): {
                    str(key): None if score is None else float(score)
                    for key, score in dict(scores or {}).items()
                }
                for group, scores in dict(value.get("atomics") or {}).items()
            },
            evidence=dict(value.get("evidence") or {}),
            metric_version=str(value.get("metric_version") or REWARD_VERSION),
            active_caps=[dict(item) for item in value.get("active_caps") or [] if isinstance(item, Mapping)],
            errors=[str(item) for item in value.get("errors") or []],
            metric_name=str(value.get("metric_name") or METRIC_NAME),
            metric_fingerprint=str(value.get("metric_fingerprint") or ""),
            base_quality_before_caps=float(value.get("base_quality_before_caps") or 0.0),
            effective_atomic_weights={
                str(key): float(weight)
                for key, weight in dict(value.get("effective_atomic_weights") or {}).items()
            },
            effective_dimension_weights={
                str(key): float(weight)
                for key, weight in dict(value.get("effective_dimension_weights") or {}).items()
            },
            applicable_atomic_count=int(value.get("applicable_atomic_count") or 0),
            anti_domination_feasible=bool(value.get("anti_domination_feasible", True)),
            normalization=dict(value.get("normalization") or {}),
            identity={
                str(key): None if item is None else str(item)
                for key, item in dict(value.get("identity") or {}).items()
            },
            binding_caps=[
                dict(item)
                for item in value.get("binding_caps") or []
                if isinstance(item, Mapping)
            ],
            cap_margin=float(value.get("cap_margin") or 0.0),
            matching_certification=dict(
                value.get("matching_certification") or {}
            ),
            dynamic_semantics=dict(
                value.get("dynamic_semantics") or {}
            ),
            **v53_fields,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    finite_values = (result.reward, result.quality_0_100, result.quality_0_1, result.cap_0_1)
    return result if all(math.isfinite(item) for item in finite_values) else None


def _native_render_status(
    record: Mapping[str, Any],
    render_row: Mapping[str, Any] | None,
) -> tuple[bool, bool | None]:
    """Return (attempted, ok); generic web screenshot success is deliberately ignored."""
    candidates: list[Mapping[str, Any]] = []
    explicit = record.get("renderer_check_result")
    if isinstance(explicit, Mapping):
        candidates.append(explicit)
    if isinstance(render_row, Mapping):
        nested = render_row.get("renderer_check_result")
        if isinstance(nested, Mapping):
            candidates.insert(0, nested)
        candidates.append(render_row)
        render = render_row.get("render")
        if isinstance(render, Mapping):
            candidates.append(render)

    for candidate in candidates:
        for key in ("native_render_ok", "android_render_ok"):
            if isinstance(candidate.get(key), bool):
                return True, bool(candidate[key])
        adapter_name = " ".join(
            str(candidate.get(key) or "")
            for key in ("adapter", "renderer", "source", "kind")
        ).casefold()
        is_native = "android" in adapter_name or "native" in adapter_name or "device" in adapter_name
        attempted = candidate.get("attempted")
        if attempted is False:
            continue
        if is_native:
            for key in ("ok", "success", "image_ok", "render_ok"):
                if isinstance(candidate.get(key), bool):
                    return True, bool(candidate[key])
            status = str(candidate.get("status") or "").casefold()
            if status:
                return True, status in {"ok", "success", "passed", "rendered"}
            return True, False
    return False, None


def score_record_v4(
    record: Mapping[str, Any],
    *,
    render_row: Mapping[str, Any] | None = None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    attempted, render_ok = _native_render_status(record, render_row)
    if not attempted:
        stored = breakdown_from_mapping(record.get("genui_quality_v4"))
        if stored is not None:
            return stored
    result = score_genui_completion_v4(
        record.get("genui_json", record.get("a2ui_json")),
        str(record.get("response_text") or ""),
        intent=str(record.get("intent_bucket") or record.get("intent") or "") or None,
        assets=record.get("assets"),
        expected_ui_contract=(
            record.get("expected_ui_contract")
            if isinstance(record.get("expected_ui_contract"), Mapping)
            else None
        ),
        render_ok=render_ok if attempted else None,
        config=config,
    )
    contract_source = record.get("expected_ui_contract_source")
    source_evidence = result.evidence.get("source")
    if contract_source and isinstance(source_evidence, dict):
        source_evidence["contract_source"] = str(contract_source)
    return result


def _record_candidate(record: Mapping[str, Any]) -> Any:
    for key in ("genui_raw_completion", "raw_completion", "genui_json", "a2ui_json"):
        if key in record and record.get(key) is not None:
            return record.get(key)
    return None


def score_record_v5_0(
    record: Mapping[str, Any],
    *,
    render_row: Mapping[str, Any] | None = None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Reuse v5 only on exact fingerprint and payload identity equality."""

    cfg = config or load_v5_reward_config()
    candidate = _record_candidate(record)
    response_text = str(record.get("response_text") or "")
    intent = str(record.get("intent_bucket") or record.get("intent") or "") or None
    assets = record.get("assets")
    persisted = (
        record.get("expected_ui_contract")
        if isinstance(record.get("expected_ui_contract"), Mapping)
        else None
    )
    contract_resolution = resolve_expected_ui_contract(
        response_text,
        intent=intent,
        assets=assets,
        persisted=persisted,
        persisted_source=(
            str(record.get("expected_ui_contract_source"))
            if record.get("expected_ui_contract_source")
            else None
        ),
    )
    normalization = normalize_and_validate_candidate(candidate)
    expected_identity = {
        "source_hash": str(contract_resolution.contract.get("source_hash") or ""),
        "raw_candidate_hash": normalization.raw_hash,
        "canonical_candidate_hash": normalization.canonical_hash,
        "expected_contract_hash": expected_contract_hash(contract_resolution.contract),
    }
    expected_fingerprint = metric_fingerprint_v5_0(cfg)
    attempted, render_ok = _native_render_status(record, render_row)

    stored = breakdown_from_mapping(record.get("genui_quality_v5"))
    stale_reasons: list[str] = []
    if stored is None:
        stale_reasons.append("missing_or_invalid_stored_v5")
    else:
        if stored.metric_version != V5_REWARD_VERSION:
            stale_reasons.append("metric_version_mismatch")
        if stored.metric_fingerprint != expected_fingerprint:
            stale_reasons.append("metric_fingerprint_mismatch")
        for key, expected in expected_identity.items():
            if stored.identity.get(key) != expected:
                stale_reasons.append(f"{key}_mismatch")
        stored_render = stored.evidence.get("render_ok")
        expected_render = render_ok if attempted else None
        if stored_render is not expected_render:
            stale_reasons.append("render_evidence_mismatch")

    if stored is not None and not stale_reasons:
        stored.evidence["score_reuse"] = {
            "reused": True,
            "stale_reasons": [],
        }
        return stored

    result = score_genui_completion_v5_0(
        candidate,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=persisted,
        render_ok=render_ok if attempted else None,
        config=cfg,
    )
    result.evidence["score_reuse"] = {
        "reused": False,
        "stale_reasons": stale_reasons,
    }
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    share = position - low
    return ordered[low] * (1.0 - share) + ordered[high] * share


def _bootstrap_mean_ci(values: Sequence[float], iterations: int = 400) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0, "confidence": 0.95, "iterations": 0}
    if len(values) == 1:
        value = float(values[0])
        return {"low": value, "high": value, "confidence": 0.95, "iterations": iterations}
    rng = random.Random(0xA2A1)
    n = len(values)
    means = [
        sum(float(values[rng.randrange(n)]) for _ in range(n)) / n
        for _ in range(iterations)
    ]
    return {
        "low": _quantile(means, 0.025),
        "high": _quantile(means, 0.975),
        "confidence": 0.95,
        "iterations": iterations,
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric) if numeric else 0.0,
        "median": statistics.median(numeric) if numeric else 0.0,
        "sd_population": statistics.pstdev(numeric) if len(numeric) > 1 else 0.0,
        "quantiles": {
            str(probability): _quantile(numeric, probability)
            for probability in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
        },
        "bootstrap_mean_ci_95": _bootstrap_mean_ci(numeric),
    }


def aggregate_v4_records(
    records: Sequence[Mapping[str, Any]],
    *,
    render_rows_by_ui_id: Mapping[str, Mapping[str, Any]] | None = None,
    config: RewardConfig | None = None,
) -> dict[str, Any]:
    """Aggregate already-scored samples; never feed aggregate means back through caps."""
    results: list[tuple[Mapping[str, Any], RewardBreakdown]] = []
    for record in records:
        ui_id = record.get("ui_id")
        render_row = (
            render_rows_by_ui_id.get(str(ui_id))
            if render_rows_by_ui_id is not None and ui_id is not None
            else None
        )
        results.append((record, score_record_v4(record, render_row=render_row, config=config)))

    qualities = [result.quality_0_100 for _, result in results]
    rewards = [result.reward for _, result in results]
    dimensions: dict[str, list[float]] = {}
    cap_counts: dict[str, int] = {}
    intent_values: dict[str, list[float]] = {}
    parse_failures = 0
    root_failures = 0
    reference_failures = 0
    cycle_failures = 0
    render_failures = 0
    rendered = 0
    component_counts: list[float] = []
    action_fidelity: list[float] = []
    table_fidelity: list[float] = []
    json_source_economy: list[float] = []
    role_omissions: dict[str, int] = {}
    role_applicable: dict[str, int] = {}
    contract_sources: dict[str, int] = {}
    model_checkpoints: dict[str, int] = {}

    for record, result in results:
        for name, value in result.dimensions.items():
            if value is not None:
                dimensions.setdefault(name, []).append(float(value))
        for cap in result.active_caps:
            name = str(cap.get("name") or "unknown")
            cap_counts[name] = cap_counts.get(name, 0) + 1
        intent = str(record.get("intent_bucket") or record.get("intent") or "unknown")
        intent_values.setdefault(intent, []).append(result.quality_0_100)
        parse_failures += int(result.parse_stage not in {"json", "mapping"})
        root_failures += int(any(cap.get("name") == "missing_root" for cap in result.active_caps))
        output_evidence = result.evidence.get("output")
        if isinstance(output_evidence, Mapping):
            reference_failures += int(bool(output_evidence.get("missing_references")))
            cycle_failures += int(bool(output_evidence.get("cycles")))
            component_counts.append(float(output_evidence.get("reachable_elements") or 0.0))
        source_evidence = result.evidence.get("source")
        if isinstance(source_evidence, Mapping):
            contract_source = str(source_evidence.get("contract_source") or "unknown")
            contract_sources[contract_source] = contract_sources.get(contract_source, 0) + 1
        role_values = result.evidence.get("role_values")
        if isinstance(role_values, Mapping):
            for role, value in role_values.items():
                if value is None:
                    continue
                role_name = str(role)
                role_applicable[role_name] = role_applicable.get(role_name, 0) + 1
                role_omissions[role_name] = role_omissions.get(role_name, 0) + int(float(value) < 1.0)
        fidelity_atomics = result.atomics.get("fidelity", {})
        if fidelity_atomics.get("action_and_source_link_fidelity") is not None:
            action_fidelity.append(float(fidelity_atomics["action_and_source_link_fidelity"]))
        if fidelity_atomics.get("markdown_table_fidelity") is not None:
            table_fidelity.append(float(fidelity_atomics["markdown_table_fidelity"]))
        economy_atomics = result.atomics.get("economy", {})
        if economy_atomics.get("json_to_source_size") is not None:
            json_source_economy.append(float(economy_atomics["json_to_source_size"]))
        generation = record.get("gen")
        model_checkpoint = record.get("model_checkpoint")
        if not model_checkpoint and isinstance(generation, Mapping):
            model_checkpoint = generation.get("model")
        if model_checkpoint:
            model_name = str(model_checkpoint)
            model_checkpoints[model_name] = model_checkpoints.get(model_name, 0) + 1
        render_value = result.evidence.get("render_ok")
        if render_value is not None:
            rendered += 1
            render_failures += int(render_value is False)

    count = len(results)
    intent_means = {
        intent: {"count": len(values), "mean": statistics.fmean(values)}
        for intent, values in sorted(intent_values.items())
        if values
    }
    macro_intent = (
        statistics.fmean(item["mean"] for item in intent_means.values())
        if intent_means
        else 0.0
    )
    return {
        "metric_version": V4_REWARD_VERSION,
        "contract_version": CONTRACT_VERSION,
        "calibration_status": "uncalibrated_engineering_score",
        "count": count,
        "quality_0_100": _distribution(qualities),
        "reward_minus1_1": _distribution(rewards),
        "dimensions": {
            name: _distribution(values) for name, values in sorted(dimensions.items())
        },
        "intent": {
            "micro_mean": statistics.fmean(qualities) if qualities else 0.0,
            "macro_mean": macro_intent,
            "by_intent": intent_means,
        },
        "capped_rate": (
            sum(result.cap_0_1 < 1.0 for _, result in results) / count if count else 0.0
        ),
        "cap_rates": {
            name: occurrences / count for name, occurrences in sorted(cap_counts.items())
        } if count else {},
        "parse_failure_rate": parse_failures / count if count else 0.0,
        "root_failure_rate": root_failures / count if count else 0.0,
        "reference_failure_rate": reference_failures / count if count else 0.0,
        "cycle_failure_rate": cycle_failures / count if count else 0.0,
        "native_render_attempt_rate": rendered / count if count else 0.0,
        "native_render_failure_rate": render_failures / rendered if rendered else None,
        "diagnostics": {
            "component_count": _distribution(component_counts),
            "action_fidelity": _distribution(action_fidelity),
            "table_fidelity": _distribution(table_fidelity),
            "json_to_source_economy_utility": _distribution(json_source_economy),
            "role_omission_rates": {
                role: role_omissions.get(role, 0) / applicable
                for role, applicable in sorted(role_applicable.items())
                if applicable
            },
            "contract_source_counts": dict(sorted(contract_sources.items())),
            "model_checkpoint_counts": dict(sorted(model_checkpoints.items())),
        },
        "floor_rate_0_05": sum(value <= 5.0 for value in qualities) / count if count else 0.0,
        "ceiling_rate_0_95": sum(value >= 95.0 for value in qualities) / count if count else 0.0,
    }


def aggregate_v5_0_records(
    records: Sequence[Mapping[str, Any]],
    *,
    render_rows_by_ui_id: Mapping[str, Mapping[str, Any]] | None = None,
    config: RewardConfig | None = None,
) -> dict[str, Any]:
    """Aggregate independent v5 sample scores without nonlinear re-aggregation."""

    cfg = config or load_v5_reward_config()
    results: list[tuple[Mapping[str, Any], RewardBreakdown]] = []
    for record in records:
        ui_id = record.get("ui_id")
        render_row = (
            render_rows_by_ui_id.get(str(ui_id))
            if render_rows_by_ui_id is not None and ui_id is not None
            else None
        )
        results.append(
            (record, score_record_v5_0(record, render_row=render_row, config=cfg))
        )

    qualities = [result.quality_0_100 for _, result in results]
    rewards = [result.reward for _, result in results]
    dimension_values: dict[str, list[float]] = defaultdict(list)
    cap_counts: dict[str, int] = defaultdict(int)
    stale_counts: dict[str, int] = defaultdict(int)
    intent_values: dict[str, list[float]] = defaultdict(list)
    reused = 0
    for record, result in results:
        for dimension, value in result.dimensions.items():
            if value is not None:
                dimension_values[dimension].append(float(value))
        for cap in result.active_caps:
            cap_counts[str(cap.get("name") or "unknown")] += 1
        reuse = result.evidence.get("score_reuse")
        if isinstance(reuse, Mapping):
            reused += int(bool(reuse.get("reused")))
            for reason in reuse.get("stale_reasons") or []:
                stale_counts[str(reason)] += 1
        intent = str(record.get("intent_bucket") or record.get("intent") or "unknown")
        intent_values[intent].append(result.quality_0_100)

    count = len(results)
    intent_means = {
        intent: {"count": len(values), "mean": statistics.fmean(values)}
        for intent, values in sorted(intent_values.items())
    }
    return {
        "metric_name": METRIC_NAME,
        "metric_version": V5_REWARD_VERSION,
        "metric_fingerprint": metric_fingerprint_v5_0(cfg),
        "contract_version": CONTRACT_VERSION,
        "calibration_status": "uncalibrated_engineering_score",
        "count": count,
        "quality_0_100": _distribution(qualities),
        "reward_minus1_1": _distribution(rewards),
        "dimensions": {
            name: _distribution(values)
            for name, values in sorted(dimension_values.items())
        },
        "intent": {
            "micro_mean": statistics.fmean(qualities) if qualities else 0.0,
            "macro_mean": (
                statistics.fmean(value["mean"] for value in intent_means.values())
                if intent_means
                else 0.0
            ),
            "by_intent": intent_means,
        },
        "cap_rates": {
            name: occurrences / count
            for name, occurrences in sorted(cap_counts.items())
        }
        if count
        else {},
        "capped_rate": (
            sum(result.cap_0_1 < 1.0 for _, result in results) / count
            if count
            else 0.0
        ),
        "score_reuse_rate": reused / count if count else 0.0,
        "stale_score_reasons": dict(sorted(stale_counts.items())),
        "parse_failure_rate": (
            sum(result.parse_stage == "parse_failure" for _, result in results) / count
            if count
            else 0.0
        ),
        "production_invalid_rate": (
            sum(
                not bool(result.normalization.get("production_valid"))
                for _, result in results
            )
            / count
            if count
            else 0.0
        ),
        "strict_schema_invalid_rate": (
            sum(
                not bool(result.normalization.get("strict_schema_valid"))
                for _, result in results
            )
            / count
            if count
            else 0.0
        ),
        "floor_rate_0_05": (
            sum(value <= 5.0 for value in qualities) / count if count else 0.0
        ),
        "ceiling_rate_0_95": (
            sum(value >= 95.0 for value in qualities) / count if count else 0.0
        ),
    }


def _record_final_candidate(record: Mapping[str, Any]) -> Any:
    for key in ("genui_json", "a2ui_json", "genui_raw_completion", "raw_completion"):
        if key in record and record.get(key) is not None:
            return record.get(key)
    return None


def _score_record_v5_1_variant(
    record: Mapping[str, Any],
    *,
    candidate: Any,
    stored_keys: Sequence[str],
    render_row: Mapping[str, Any] | None,
    config: RewardConfig,
    use_render_evidence: bool,
) -> RewardBreakdown:
    response_text = str(record.get("response_text") or "")
    intent = str(record.get("intent_bucket") or record.get("intent") or "") or None
    assets = record.get("assets")
    persisted = (
        record.get("expected_ui_contract")
        if isinstance(record.get("expected_ui_contract"), Mapping)
        else None
    )
    resolution = resolve_expected_ui_contract_v5_1(
        response_text,
        intent=intent,
        assets=assets,
        persisted=persisted,
        persisted_source=(
            str(record.get("expected_ui_contract_source"))
            if record.get("expected_ui_contract_source")
            else None
        ),
    )
    normalization = normalize_and_validate_candidate(candidate)
    expected_identity = {
        "source_hash": str(resolution.contract.get("source_hash") or ""),
        "raw_candidate_hash": normalization.raw_hash,
        "canonical_candidate_hash": normalization.canonical_hash,
        "expected_contract_hash": expected_contract_hash_v5_1(
            resolution.contract
        ),
    }
    expected_fingerprint = metric_fingerprint_v5_1(config)
    attempted, render_ok = (
        _native_render_status(record, render_row)
        if use_render_evidence
        else (False, None)
    )
    stored: RewardBreakdown | None = None
    for key in stored_keys:
        stored = breakdown_from_mapping(record.get(key))
        if stored is not None:
            break
    stale_reasons: list[str] = []
    if stored is None:
        stale_reasons.append("missing_or_invalid_stored_v5_1")
    else:
        if stored.metric_version != V5_1_REWARD_VERSION:
            stale_reasons.append("metric_version_mismatch")
        if stored.metric_fingerprint != expected_fingerprint:
            stale_reasons.append("metric_fingerprint_mismatch")
        for key, expected in expected_identity.items():
            if stored.identity.get(key) != expected:
                stale_reasons.append(f"{key}_mismatch")
        expected_render = render_ok if attempted else None
        if stored.evidence.get("render_ok") is not expected_render:
            stale_reasons.append("render_evidence_mismatch")
    if stored is not None and not stale_reasons:
        stored.evidence["score_reuse"] = {
            "reused": True,
            "stale_reasons": [],
        }
        return stored
    result = score_genui_completion_v5_1(
        candidate,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=resolution.contract,
        render_ok=render_ok if attempted else None,
        config=config,
    )
    result.evidence["score_reuse"] = {
        "reused": False,
        "stale_reasons": stale_reasons,
    }
    return result


def score_record_v5_1(
    record: Mapping[str, Any],
    *,
    render_row: Mapping[str, Any] | None = None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score/reuse the final renderer artifact as the v5.1 headline."""

    cfg = config or load_v5_1_reward_config()
    return _score_record_v5_1_variant(
        record,
        candidate=_record_final_candidate(record),
        stored_keys=(
            "render_artifact_quality_v5_1",
            "genui_quality_v5_1",
        ),
        render_row=render_row,
        config=cfg,
        use_render_evidence=True,
    )


def score_record_generation_v5_1(
    record: Mapping[str, Any],
    *,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score/reuse the raw generation path used by deterministic GRPO."""

    cfg = config or load_v5_1_reward_config()
    return _score_record_v5_1_variant(
        record,
        candidate=_record_candidate(record),
        stored_keys=("generation_reward_v5_1",),
        render_row=None,
        config=cfg,
        use_render_evidence=False,
    )


def _score_record_v5_2_variant(
    record: Mapping[str, Any],
    *,
    candidate: Any,
    stored_keys: Sequence[str],
    render_row: Mapping[str, Any] | None,
    config: RewardConfig,
    use_render_evidence: bool,
    computed_registry: ComputedFunctionRegistryV52 | None = None,
) -> RewardBreakdown:
    """Reuse a v5.2 score only when every score identity is exact."""

    response_text = str(record.get("response_text") or "")
    intent = (
        str(record.get("intent_bucket") or record.get("intent") or "")
        or None
    )
    assets = record.get("assets")
    persisted = (
        record.get("expected_ui_contract")
        if isinstance(record.get("expected_ui_contract"), Mapping)
        else None
    )
    resolution = resolve_expected_ui_contract_v5_2(
        response_text,
        intent=intent,
        assets=assets,
        persisted=persisted,
        persisted_source=(
            str(record.get("expected_ui_contract_source"))
            if record.get("expected_ui_contract_source")
            else None
        ),
    )
    normalization = normalize_and_validate_candidate(candidate)
    expected_identity = {
        "source_hash": str(resolution.contract.get("source_hash") or ""),
        "raw_candidate_hash": normalization.raw_hash,
        "canonical_candidate_hash": normalization.canonical_hash,
        "expected_contract_hash": expected_contract_hash_v5_2(
            resolution.contract
        ),
    }
    expected_fingerprint = metric_fingerprint_v5_2(
        config, computed_registry=computed_registry
    )
    attempted, render_ok = (
        _native_render_status(record, render_row)
        if use_render_evidence
        else (False, None)
    )
    stored: RewardBreakdown | None = None
    for key in stored_keys:
        stored = breakdown_from_mapping(record.get(key))
        if stored is not None:
            break
    stale_reasons: list[str] = []
    if stored is None:
        stale_reasons.append("missing_or_invalid_stored_v5_2")
    else:
        if stored.metric_version != REWARD_VERSION:
            stale_reasons.append("metric_version_mismatch")
        if stored.metric_fingerprint != expected_fingerprint:
            stale_reasons.append("metric_fingerprint_mismatch")
        for key, expected in expected_identity.items():
            if stored.identity.get(key) != expected:
                stale_reasons.append(f"{key}_mismatch")
        expected_render = render_ok if attempted else None
        if stored.evidence.get("render_ok") is not expected_render:
            stale_reasons.append("render_evidence_mismatch")
    if stored is not None and not stale_reasons:
        stored.evidence["score_reuse"] = {
            "reused": True,
            "stale_reasons": [],
        }
        return stored
    result = _score_genui_completion_v5_2(
        candidate,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=resolution.contract,
        expected_ui_contract_source=resolution.source,
        render_ok=render_ok if attempted else None,
        config=config,
        computed_registry=computed_registry,
    )
    result.evidence["score_reuse"] = {
        "reused": False,
        "stale_reasons": stale_reasons,
    }
    return result


def score_record(
    record: Mapping[str, Any],
    *,
    render_row: Mapping[str, Any] | None = None,
    config: RewardConfig | None = None,
    computed_registry: ComputedFunctionRegistryV52 | None = None,
) -> RewardBreakdown:
    """Score/reuse the final renderer artifact as the v5.2 headline."""

    cfg = config or load_default_reward_config()
    return _score_record_v5_2_variant(
        record,
        candidate=_record_final_candidate(record),
        stored_keys=(
            "render_artifact_quality_v5_2",
            "genui_quality_v5_2",
        ),
        render_row=render_row,
        config=cfg,
        use_render_evidence=True,
        computed_registry=computed_registry,
    )


def score_record_generation_v5_2(
    record: Mapping[str, Any],
    *,
    config: RewardConfig | None = None,
    computed_registry: ComputedFunctionRegistryV52 | None = None,
) -> RewardBreakdown:
    """Score/reuse the raw v5.2 generation path used by GRPO."""

    cfg = config or load_default_reward_config()
    return _score_record_v5_2_variant(
        record,
        candidate=_record_candidate(record),
        stored_keys=("generation_reward_v5_2",),
        render_row=None,
        config=cfg,
        use_render_evidence=False,
        computed_registry=computed_registry,
    )


def _score_record_v5_3_variant(
    record: Mapping[str, Any],
    *,
    candidate: Any,
    stored_keys: Sequence[str],
    render_row: Mapping[str, Any] | None,
    config: RewardConfigV53,
    use_render_evidence: bool,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> RewardBreakdownV53:
    """Reuse v5.3 only when score, pipeline, contract, and payload agree."""

    response_text = str(record.get("response_text") or "")
    intent = (
        str(record.get("intent_bucket") or record.get("intent") or "")
        or None
    )
    assets = record.get("assets")
    persisted = (
        record.get("expected_ui_contract_v5_3")
        if isinstance(record.get("expected_ui_contract_v5_3"), Mapping)
        else record.get("expected_ui_contract")
        if isinstance(record.get("expected_ui_contract"), Mapping)
        else None
    )
    persisted_source = (
        record.get("expected_ui_contract_v5_3_source")
        or record.get("expected_ui_contract_source")
    )
    resolution = resolve_expected_ui_contract_v5_3(
        response_text,
        intent=intent,
        assets=assets,
        persisted=persisted,
        persisted_source=(
            str(persisted_source) if persisted_source else None
        ),
    )
    normalization = normalize_and_validate_candidate(candidate)
    expected_identity = {
        "source_hash": str(resolution.contract.get("source_hash") or ""),
        "raw_candidate_hash": normalization.raw_hash,
        "canonical_candidate_hash": normalization.canonical_hash,
        "expected_contract_hash": expected_contract_hash_v5_3(
            resolution.contract
        ),
    }
    expected_fingerprint = metric_fingerprint_v5_3(
        config, computed_registry=computed_registry
    )
    expected_pipeline = reward_pipeline_fingerprint_v5_3(
        expected_fingerprint
    )
    attempted, render_ok = (
        _native_render_status(record, render_row)
        if use_render_evidence
        else (False, None)
    )
    stored: RewardBreakdown | None = None
    for key in stored_keys:
        stored = breakdown_from_mapping(record.get(key))
        if stored is not None:
            break
    stale_reasons: list[str] = []
    if stored is None:
        stale_reasons.append("missing_or_invalid_stored_v5_3")
    else:
        if stored.metric_version != REWARD_VERSION_V53:
            stale_reasons.append("metric_version_mismatch")
        if stored.metric_fingerprint != expected_fingerprint:
            stale_reasons.append("metric_fingerprint_mismatch")
        if not isinstance(stored, RewardBreakdownV53):
            stale_reasons.append("v5_3_diagnostics_missing")
        elif stored.reward_pipeline_fingerprint != expected_pipeline:
            stale_reasons.append("reward_pipeline_fingerprint_mismatch")
        for key, expected in expected_identity.items():
            if stored.identity.get(key) != expected:
                stale_reasons.append(f"{key}_mismatch")
        expected_render = render_ok if attempted else None
        if stored.evidence.get("render_ok") is not expected_render:
            stale_reasons.append("render_evidence_mismatch")
    if (
        isinstance(stored, RewardBreakdownV53)
        and not stale_reasons
    ):
        stored.evidence["score_reuse"] = {
            "reused": True,
            "stale_reasons": [],
        }
        return stored
    result = _score_genui_completion_v5_3(
        candidate,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=resolution.contract,
        expected_ui_contract_source=resolution.source,
        render_ok=render_ok if attempted else None,
        config=config,
        computed_registry=computed_registry,
    )
    result.evidence["score_reuse"] = {
        "reused": False,
        "stale_reasons": list(dict.fromkeys(stale_reasons)),
    }
    return result


def score_record_v5_3(
    record: Mapping[str, Any],
    *,
    render_row: Mapping[str, Any] | None = None,
    config: RewardConfigV53 | None = None,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> RewardBreakdownV53:
    cfg = config or load_v5_3_reward_config()
    return _score_record_v5_3_variant(
        record,
        candidate=_record_final_candidate(record),
        stored_keys=(
            "render_artifact_quality_v5_3",
            "genui_quality_v5_3",
        ),
        render_row=render_row,
        config=cfg,
        use_render_evidence=True,
        computed_registry=computed_registry,
    )


def score_record_generation_v5_3(
    record: Mapping[str, Any],
    *,
    config: RewardConfigV53 | None = None,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> RewardBreakdownV53:
    cfg = config or load_v5_3_reward_config()
    return _score_record_v5_3_variant(
        record,
        candidate=_record_candidate(record),
        stored_keys=("generation_reward_v5_3",),
        render_row=None,
        config=cfg,
        use_render_evidence=False,
        computed_registry=computed_registry,
    )


def aggregate_v5_3_records(
    records: Sequence[Mapping[str, Any]],
    *,
    render_rows_by_ui_id: Mapping[str, Mapping[str, Any]] | None = None,
    config: RewardConfigV53 | None = None,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> dict[str, Any]:
    """Aggregate immutable per-sample v5.3 results without rescoring means."""

    cfg = config or load_v5_3_reward_config()
    results: list[
        tuple[Mapping[str, Any], RewardBreakdownV53, RewardBreakdownV53]
    ] = []
    for record in records:
        ui_id = record.get("ui_id")
        render_row = (
            render_rows_by_ui_id.get(str(ui_id))
            if render_rows_by_ui_id is not None and ui_id is not None
            else None
        )
        results.append(
            (
                record,
                score_record_v5_3(
                    record,
                    render_row=render_row,
                    config=cfg,
                    computed_registry=computed_registry,
                ),
                score_record_generation_v5_3(
                    record,
                    config=cfg,
                    computed_registry=computed_registry,
                ),
            )
        )
    final_scores = [item.quality_0_100 for _, item, _ in results]
    generation_scores = [item.quality_0_100 for _, _, item in results]
    rewards = [item.reward for _, _, item in results]
    dimensions: dict[str, list[float]] = defaultdict(list)
    caps: dict[str, int] = defaultdict(int)
    stale: dict[str, int] = defaultdict(int)
    reused = 0
    for _, result, _ in results:
        for name, value in result.dimensions.items():
            if value is not None:
                dimensions[name].append(float(value))
        for cap in result.active_caps:
            caps[str(cap.get("name") or "unknown")] += 1
        reuse = result.evidence.get("score_reuse")
        if isinstance(reuse, Mapping):
            reused += int(bool(reuse.get("reused")))
            for reason in reuse.get("stale_reasons") or ():
                stale[str(reason)] += 1
    count = len(results)
    fingerprint = metric_fingerprint_v5_3(
        cfg, computed_registry=computed_registry
    )
    return {
        "metric_name": METRIC_NAME,
        "metric_version": REWARD_VERSION_V53,
        "metric_fingerprint": fingerprint,
        "reward_pipeline_fingerprint": (
            reward_pipeline_fingerprint_v5_3(fingerprint)
        ),
        "calibration_status": "uncalibrated_engineering_score",
        "headline": "render_artifact_quality_v5_3",
        "count": count,
        "quality_0_100": _distribution(final_scores),
        "generation_reward_quality_0_100": _distribution(
            generation_scores
        ),
        "generation_reward_minus1_1": _distribution(rewards),
        "dimensions": {
            name: _distribution(values)
            for name, values in sorted(dimensions.items())
        },
        "per_cap_active_rate": (
            {
                name: occurrences / count
                for name, occurrences in sorted(caps.items())
            }
            if count
            else {}
        ),
        "score_reuse_rate": reused / count if count else 0.0,
        "stale_score_reasons": dict(sorted(stale.items())),
        "matching_uncertified_rate": (
            sum(
                not bool(item.matching_certification.get(
                    "optimality_certified"
                ))
                for _, item, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "dynamic_unknown_rate": (
            sum(
                not bool(item.dynamic_evidence_certification.get("complete"))
                for _, item, _ in results
            )
            / count
            if count
            else 0.0
        ),
    }


def aggregate_v5_1_records(
    records: Sequence[Mapping[str, Any]],
    *,
    render_rows_by_ui_id: Mapping[str, Mapping[str, Any]] | None = None,
    config: RewardConfig | None = None,
) -> dict[str, Any]:
    """Aggregate v5.1 final-artifact headlines and raw-generation rewards."""

    cfg = config or load_v5_1_reward_config()
    results: list[
        tuple[Mapping[str, Any], RewardBreakdown, RewardBreakdown]
    ] = []
    for record in records:
        ui_id = record.get("ui_id")
        render_row = (
            render_rows_by_ui_id.get(str(ui_id))
            if render_rows_by_ui_id is not None and ui_id is not None
            else None
        )
        final_result = score_record_v5_1(
            record, render_row=render_row, config=cfg
        )
        generation_result = score_record_generation_v5_1(
            record, config=cfg
        )
        results.append((record, final_result, generation_result))

    final_qualities = [
        result.quality_0_100 for _, result, _ in results
    ]
    generation_qualities = [
        result.quality_0_100 for _, _, result in results
    ]
    generation_rewards = [result.reward for _, _, result in results]
    deltas = [
        final.quality_0_100 - generation.quality_0_100
        for _, final, generation in results
    ]
    dimension_values: dict[str, list[float]] = defaultdict(list)
    active_cap_counts: dict[str, int] = defaultdict(int)
    binding_cap_counts: dict[str, int] = defaultdict(int)
    stale_counts: dict[str, int] = defaultdict(int)
    intent_values: dict[str, list[float]] = defaultdict(list)
    reused = 0
    for record, final, _ in results:
        for dimension, value in final.dimensions.items():
            if value is not None:
                dimension_values[dimension].append(float(value))
        for cap in final.active_caps:
            active_cap_counts[str(cap.get("name") or "unknown")] += 1
        for cap in final.binding_caps:
            binding_cap_counts[str(cap.get("name") or "unknown")] += 1
        reuse = final.evidence.get("score_reuse")
        if isinstance(reuse, Mapping):
            reused += int(bool(reuse.get("reused")))
            for reason in reuse.get("stale_reasons") or []:
                stale_counts[str(reason)] += 1
        intent = str(
            record.get("intent_bucket") or record.get("intent") or "unknown"
        )
        intent_values[intent].append(final.quality_0_100)

    count = len(results)
    intent_means = {
        intent: {
            "count": len(values),
            "mean": statistics.fmean(values),
        }
        for intent, values in sorted(intent_values.items())
    }
    active_cap_rate = (
        sum(bool(result.active_caps) for _, result, _ in results) / count
        if count
        else 0.0
    )
    binding_cap_rate = (
        sum(bool(result.binding_caps) for _, result, _ in results) / count
        if count
        else 0.0
    )
    return {
        "metric_name": METRIC_NAME,
        "metric_version": V5_1_REWARD_VERSION,
        "metric_fingerprint": metric_fingerprint_v5_1(cfg),
        "contract_version": (
            next(
                (
                    final.evidence.get("source", {}).get("contract_version")
                    for _, final, _ in results
                    if isinstance(final.evidence.get("source"), Mapping)
                ),
                None,
            )
        ),
        "calibration_status": "uncalibrated_engineering_score",
        "headline": "render_artifact_quality_v5_1",
        "count": count,
        "quality_0_100": _distribution(final_qualities),
        "render_artifact_quality_0_100": _distribution(final_qualities),
        "generation_reward_quality_0_100": _distribution(
            generation_qualities
        ),
        "generation_reward_minus1_1": _distribution(generation_rewards),
        "raw_final_delta_0_100": _distribution(deltas),
        "dimensions": {
            name: _distribution(values)
            for name, values in sorted(dimension_values.items())
        },
        "intent": {
            "micro_mean": (
                statistics.fmean(final_qualities)
                if final_qualities
                else 0.0
            ),
            "macro_mean": (
                statistics.fmean(
                    value["mean"] for value in intent_means.values()
                )
                if intent_means
                else 0.0
            ),
            "by_intent": intent_means,
        },
        "active_cap_rate": active_cap_rate,
        "binding_cap_rate": binding_cap_rate,
        "per_cap_active_rate": (
            {
                name: occurrences / count
                for name, occurrences in sorted(
                    active_cap_counts.items()
                )
            }
            if count
            else {}
        ),
        "per_cap_binding_rate": (
            {
                name: occurrences / count
                for name, occurrences in sorted(
                    binding_cap_counts.items()
                )
            }
            if count
            else {}
        ),
        "cap_margin_distribution": _distribution(
            [final.cap_margin for _, final, _ in results]
        ),
        "capped_rate": active_cap_rate,
        "capped_rate_deprecated_semantics": "active_cap_rate",
        "score_reuse_rate": reused / count if count else 0.0,
        "stale_score_reasons": dict(sorted(stale_counts.items())),
        "matching_incomplete_rate": (
            sum(
                not bool(final.evidence.get("matching_complete", True))
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "dynamic_unknown_rate": (
            sum(
                int(
                    final.evidence.get(
                        "dynamic_expression_unknown_count", 0
                    )
                    or 0
                )
                > 0
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "parse_failure_rate": (
            sum(
                final.parse_stage == "parse_failure"
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "production_invalid_rate": (
            sum(
                not bool(final.normalization.get("production_valid"))
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "strict_schema_invalid_rate": (
            sum(
                not bool(final.normalization.get("strict_schema_valid"))
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "floor_rate_0_05": (
            sum(value <= 5.0 for value in final_qualities) / count
            if count
            else 0.0
        ),
        "ceiling_rate_0_95": (
            sum(value >= 95.0 for value in final_qualities) / count
            if count
            else 0.0
        ),
    }


def aggregate_v5_records(
    records: Sequence[Mapping[str, Any]],
    *,
    render_rows_by_ui_id: Mapping[str, Mapping[str, Any]] | None = None,
    config: RewardConfig | None = None,
    computed_registry: ComputedFunctionRegistryV52 | None = None,
) -> dict[str, Any]:
    """Aggregate per-sample v5.2 final and generation scores."""

    cfg = config or load_default_reward_config()
    results: list[
        tuple[Mapping[str, Any], RewardBreakdown, RewardBreakdown]
    ] = []
    for record in records:
        ui_id = record.get("ui_id")
        render_row = (
            render_rows_by_ui_id.get(str(ui_id))
            if render_rows_by_ui_id is not None and ui_id is not None
            else None
        )
        final = score_record(
            record,
            render_row=render_row,
            config=cfg,
            computed_registry=computed_registry,
        )
        generation = score_record_generation_v5_2(
            record,
            config=cfg,
            computed_registry=computed_registry,
        )
        results.append((record, final, generation))

    final_scores = [
        final.quality_0_100 for _, final, _ in results
    ]
    generation_scores = [
        generation.quality_0_100 for _, _, generation in results
    ]
    generation_rewards = [
        generation.reward for _, _, generation in results
    ]
    deltas = [
        final.quality_0_100 - generation.quality_0_100
        for _, final, generation in results
    ]
    dimension_values: dict[str, list[float]] = defaultdict(list)
    active_caps: dict[str, int] = defaultdict(int)
    binding_caps: dict[str, int] = defaultdict(int)
    stale: dict[str, int] = defaultdict(int)
    intent_values: dict[str, list[float]] = defaultdict(list)
    reused = 0
    for record, final, _ in results:
        for dimension, value in final.dimensions.items():
            if value is not None:
                dimension_values[dimension].append(float(value))
        for cap in final.active_caps:
            active_caps[str(cap.get("name") or "unknown")] += 1
        for cap in final.binding_caps:
            binding_caps[str(cap.get("name") or "unknown")] += 1
        reuse = final.evidence.get("score_reuse")
        if isinstance(reuse, Mapping):
            reused += int(bool(reuse.get("reused")))
            for reason in reuse.get("stale_reasons") or []:
                stale[str(reason)] += 1
        intent = str(
            record.get("intent_bucket")
            or record.get("intent")
            or "unknown"
        )
        intent_values[intent].append(final.quality_0_100)

    count = len(results)
    intent_means = {
        intent: {
            "count": len(values),
            "mean": statistics.fmean(values),
        }
        for intent, values in sorted(intent_values.items())
    }
    active_rate = (
        sum(bool(final.active_caps) for _, final, _ in results) / count
        if count
        else 0.0
    )
    binding_rate = (
        sum(bool(final.binding_caps) for _, final, _ in results) / count
        if count
        else 0.0
    )
    return {
        "metric_name": METRIC_NAME,
        "metric_version": REWARD_VERSION,
        "metric_fingerprint": metric_fingerprint_v5_2(
            cfg, computed_registry=computed_registry
        ),
        "contract_version": next(
            (
                final.evidence.get("source", {}).get(
                    "contract_version"
                )
                for _, final, _ in results
                if isinstance(final.evidence.get("source"), Mapping)
            ),
            None,
        ),
        "calibration_status": "uncalibrated_engineering_score",
        "headline": "render_artifact_quality_v5_2",
        "count": count,
        "quality_0_100": _distribution(final_scores),
        "render_artifact_quality_0_100": _distribution(final_scores),
        "generation_reward_quality_0_100": _distribution(
            generation_scores
        ),
        "generation_reward_minus1_1": _distribution(
            generation_rewards
        ),
        "raw_final_delta_0_100": _distribution(deltas),
        "dimensions": {
            name: _distribution(values)
            for name, values in sorted(dimension_values.items())
        },
        "intent": {
            "micro_mean": (
                statistics.fmean(final_scores) if final_scores else 0.0
            ),
            "macro_mean": (
                statistics.fmean(
                    value["mean"] for value in intent_means.values()
                )
                if intent_means
                else 0.0
            ),
            "by_intent": intent_means,
        },
        "active_cap_rate": active_rate,
        "binding_cap_rate": binding_rate,
        "per_cap_active_rate": (
            {
                name: occurrences / count
                for name, occurrences in sorted(active_caps.items())
            }
            if count
            else {}
        ),
        "per_cap_binding_rate": (
            {
                name: occurrences / count
                for name, occurrences in sorted(binding_caps.items())
            }
            if count
            else {}
        ),
        "cap_margin_distribution": _distribution(
            [final.cap_margin for _, final, _ in results]
        ),
        "score_reuse_rate": reused / count if count else 0.0,
        "stale_score_reasons": dict(sorted(stale.items())),
        "matching_uncertified_rate": (
            sum(
                not bool(
                    final.matching_certification.get(
                        "optimality_certified"
                    )
                )
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "dynamic_unknown_rate": (
            sum(
                int(final.dynamic_semantics.get("unknown_count") or 0)
                > 0
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "contract_count_only_rate": (
            sum(
                int(
                    (
                        final.evidence.get("source")
                        if isinstance(
                            final.evidence.get("source"), Mapping
                        )
                        else {}
                    ).get("count_only_role_count")
                    or 0
                )
                > 0
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "parse_failure_rate": (
            sum(
                final.parse_stage == "parse_failure"
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "production_invalid_rate": (
            sum(
                not bool(final.normalization.get("production_valid"))
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "strict_schema_invalid_rate": (
            sum(
                not bool(final.normalization.get("strict_schema_valid"))
                for _, final, _ in results
            )
            / count
            if count
            else 0.0
        ),
        "floor_rate_0_05": (
            sum(value <= 5.0 for value in final_scores) / count
            if count
            else 0.0
        ),
        "ceiling_rate_0_95": (
            sum(value >= 95.0 for value in final_scores) / count
            if count
            else 0.0
        ),
    }


__all__ = [
    "RendererSmokeAdapter",
    "RewardBreakdown",
    "aggregate_v5_records",
    "aggregate_v5_3_records",
    "aggregate_v5_1_records",
    "aggregate_v5_0_records",
    "aggregate_v4_records",
    "breakdown_from_mapping",
    "breakdown_to_mapping",
    "score_genui_completion",
    "score_genui_completion_v4",
    "score_genui_completion_v5_0",
    "score_genui_completion_v5_1",
    "score_genui_completion_v5_2",
    "score_genui_completion_v5_3",
    "score_record",
    "score_record_v5_1",
    "score_record_generation_v5_1",
    "score_record_generation_v5_2",
    "score_record_generation_v5_3",
    "score_record_v5_3",
    "score_record_v5_0",
    "score_record_v4",
    "with_renderer_smoke_adapter",
]
