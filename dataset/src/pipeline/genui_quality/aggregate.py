"""Single-sample scoring and distributional aggregation for metric v4."""

from __future__ import annotations

from dataclasses import asdict
import math
import random
import statistics
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from . import _core
from .config import REWARD_VERSION, RewardConfig
from .source_contract import CONTRACT_VERSION, source_contract_cache_key


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
    return RewardConfig(
        dimension_weights=config.dimension_weights,
        atomic_weights=config.atomic_weights,
        caps=config.caps,
        arithmetic_share=config.arithmetic_share,
        geometric_floor=config.geometric_floor,
        content_beta=config.content_beta,
        value_beta=config.value_beta,
        table_beta=config.table_beta,
        good_json_to_source_ratio=config.good_json_to_source_ratio,
        bad_json_to_source_ratio=config.bad_json_to_source_ratio,
        max_atomic_global_weight=config.max_atomic_global_weight,
        render_check=adapter.render_smoke,
    )


def score_genui_completion(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    render_ok: bool | None = None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score one completion without propagating malformed-candidate exceptions."""
    try:
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
            source_evidence["contract_version"] = str(
                (expected_ui_contract or {}).get("contract_version") or CONTRACT_VERSION
            )
            source_evidence["source_hash"] = str(
                (expected_ui_contract or {}).get("source_hash")
                or source_contract_cache_key(response_text)
            )
        return result
    except Exception as exc:
        q = 0.01
        return RewardBreakdown(
            reward=2.0 * q - 1.0,
            quality_0_100=100.0 * q,
            quality_0_1=q,
            cap_0_1=0.0,
            parse_stage="metric_exception",
            dimensions={},
            atomics={},
            evidence={"source": {"contract_source": "error"}},
            metric_version=REWARD_VERSION,
            active_caps=[{"kind": "integrity", "name": "metric_exception", "cap": 0.0}],
            errors=[f"{type(exc).__name__}: {exc}"],
        )


def breakdown_to_mapping(result: RewardBreakdown) -> dict[str, Any]:
    return asdict(result)


def breakdown_from_mapping(value: Any) -> RewardBreakdown | None:
    if not isinstance(value, Mapping):
        return None
    try:
        result = RewardBreakdown(
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


def score_record(
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
    result = score_genui_completion(
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
        results.append((record, score_record(record, render_row=render_row, config=config)))

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
        "metric_version": REWARD_VERSION,
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


__all__ = [
    "RendererSmokeAdapter",
    "RewardBreakdown",
    "aggregate_v4_records",
    "breakdown_from_mapping",
    "breakdown_to_mapping",
    "score_genui_completion",
    "score_record",
    "with_renderer_smoke_adapter",
]
