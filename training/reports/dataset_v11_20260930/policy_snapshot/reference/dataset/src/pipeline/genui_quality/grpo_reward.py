"""TRL-compatible GRPO reward entry points and structured diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from .aggregate import RewardBreakdown
from ._v5_3 import (
    prepare_source_context_v5_3,
    score_completion_group_v5_3,
)
from ._v5_2 import (
    prepare_source_context_v5_2,
    score_completion_group_v5_2,
)
from .config import RewardConfig
from .config import DEFAULT_DIMENSION_WEIGHTS
from .config_v5_3 import (
    REWARD_VERSION_V53,
    RewardConfigV53,
    coerce_reward_config_v5_3,
    load_v5_3_reward_config,
)
from ._core import completion_to_text
from .evidence_v5_3 import ComputedFunctionRegistryV53
from .evidence_v5_2 import ComputedFunctionRegistryV52


@dataclass
class RewardInflationMonitor:
    """Detect representation growth that is not accompanied by fidelity gain."""

    component_growth_threshold: float = 0.20
    length_growth_threshold: float = 0.20
    min_fidelity_gain: float = 0.01
    ema_alpha: float = 0.25
    _baselines: dict[str, tuple[float, float, float]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("component_growth_threshold", self.component_growth_threshold),
            ("length_growth_threshold", self.length_growth_threshold),
            ("min_fidelity_gain", self.min_fidelity_gain),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")

    @staticmethod
    def _growth(current: float, baseline: float) -> float:
        if baseline <= 1e-12:
            return 0.0 if current <= 1e-12 else 1.0
        return current / baseline - 1.0

    def observe(
        self,
        *,
        component_count: float,
        completion_length: float,
        fidelity: float,
        cohort: str = "default",
    ) -> dict[str, float]:
        current = (float(component_count), float(completion_length), float(fidelity))
        baseline = self._baselines.get(cohort)
        if baseline is None:
            component_growth = 0.0
            length_growth = 0.0
            fidelity_gain = 0.0
            comparable = 0.0
        else:
            component_growth = self._growth(current[0], baseline[0])
            length_growth = self._growth(current[1], baseline[1])
            fidelity_gain = current[2] - baseline[2]
            comparable = 1.0

        insufficient_gain = baseline is not None and fidelity_gain < self.min_fidelity_gain
        result = {
            "component_growth_ratio": component_growth,
            "completion_length_growth_ratio": length_growth,
            "fidelity_gain": fidelity_gain,
            "comparable_window": comparable,
            "alert_component_inflation_without_fidelity_gain": float(
                insufficient_gain and component_growth >= self.component_growth_threshold
            ),
            "alert_length_inflation_without_fidelity_gain": float(
                insufficient_gain and length_growth >= self.length_growth_threshold
            ),
        }
        if baseline is None:
            self._baselines[cohort] = current
        else:
            alpha = self.ema_alpha
            self._baselines[cohort] = tuple(
                alpha * value + (1.0 - alpha) * old
                for value, old in zip(current, baseline)
            )
        return result


def _broadcast_metadata(
    value: Any, count: int, default: Any = None
) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        values = list(value)
        if len(values) == count:
            return values
    return [default if value is None else value for _ in range(count)]


@dataclass(frozen=True)
class AssetCollection:
    """One source's assets; this collection broadcasts as one value."""

    items: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class PerCompletionAssetCollections:
    """Explicit vector wrapper for one asset collection per completion."""

    rows: tuple[AssetCollection | None, ...]


def normalize_scalar_or_vector_text(
    value: str | Sequence[str],
    count: int,
    *,
    field: str,
) -> list[str]:
    if isinstance(value, str):
        return [value] * count
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, Mapping)
    ):
        values = list(value)
        if len(values) != count or any(
            not isinstance(item, str) for item in values
        ):
            raise ValueError(
                f"{field} must be a string or exactly {count} strings"
            )
        return values
    raise ValueError(
        f"{field} must be a string or exactly {count} strings"
    )


def normalize_optional_scalar_or_vector_text(
    value: str | Sequence[str | None] | None,
    count: int,
    *,
    field: str,
) -> list[str | None]:
    if value is None:
        return [None] * count
    if isinstance(value, str):
        return [value] * count
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, Mapping)
    ):
        values = list(value)
        if len(values) != count or any(
            item is not None and not isinstance(item, str)
            for item in values
        ):
            raise ValueError(
                f"{field} must be null, a string, or exactly "
                f"{count} string-or-null values"
            )
        return values
    raise ValueError(
        f"{field} must be null, a string, or exactly "
        f"{count} string-or-null values"
    )


def _asset_collection(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    if isinstance(value, AssetCollection):
        return [dict(item) for item in value.items]
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    ):
        values = list(value)
        if all(isinstance(item, Mapping) for item in values):
            return [dict(item) for item in values]
    raise ValueError(
        f"{field} must contain only asset mappings"
    )


def normalize_asset_rows(
    value: (
        AssetCollection
        | PerCompletionAssetCollections
        | Sequence[Any]
        | Mapping[str, Any]
        | None
    ),
    count: int,
) -> list[list[Mapping[str, Any]] | None]:
    if value is None:
        return [None] * count
    if isinstance(value, PerCompletionAssetCollections):
        if len(value.rows) != count:
            raise ValueError(
                "assets per-completion wrapper length must match completions"
            )
        return [
            None
            if row is None
            else _asset_collection(row, field=f"assets[{index}]")
            for index, row in enumerate(value.rows)
        ]
    if isinstance(value, (AssetCollection, Mapping)):
        collection = _asset_collection(value, field="assets")
        return [collection] * count
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    ):
        values = list(value)
        # A list of asset mappings is one collection even when its length
        # happens to equal the completion count.
        if all(isinstance(item, Mapping) for item in values):
            collection = _asset_collection(values, field="assets")
            return [collection] * count
        if len(values) != count:
            raise ValueError(
                "per-completion assets must contain exactly one "
                "collection per completion"
            )
        rows: list[list[Mapping[str, Any]] | None] = []
        for index, row in enumerate(values):
            if isinstance(row, Mapping):
                raise ValueError(
                    "per-completion assets must use collections; "
                    f"assets[{index}] is a bare mapping"
                )
            rows.append(
                None
                if row is None
                else _asset_collection(row, field=f"assets[{index}]")
            )
        return rows
    raise ValueError(
        "assets must be one asset collection or explicit per-completion "
        "collections"
    )


def normalize_contract_rows(
    value: (
        Mapping[str, Any]
        | Sequence[Mapping[str, Any] | None]
        | None
    ),
    count: int,
) -> list[Mapping[str, Any] | None]:
    if value is None:
        return [None] * count
    if isinstance(value, Mapping):
        return [value] * count
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    ):
        values = list(value)
        if len(values) != count or any(
            item is not None and not isinstance(item, Mapping)
            for item in values
        ):
            raise ValueError(
                "expected_ui_contract must be one mapping or exactly "
                f"{count} mapping-or-null values"
            )
        return values
    raise ValueError(
        "expected_ui_contract must be one mapping or a per-completion vector"
    )


def _first_extra(kwargs: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
    return None


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def genui_grpo_reward(
    completions: Sequence[Any],
    response_text: Sequence[str] | str,
    intent_bucket: Sequence[str] | str | None = None,
    assets: Sequence[Any] | Any = None,
    expected_ui_contract: Sequence[Mapping[str, Any] | None] | Mapping[str, Any] | None = None,
    *,
    config: RewardConfigV53 | None = None,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
    **kwargs: Any,
) -> list[float]:
    """Score samples independently and return one bounded scalar per completion."""
    count = len(completions)
    try:
        responses = normalize_scalar_or_vector_text(
            response_text, count, field="response_text"
        )
        intents = normalize_optional_scalar_or_vector_text(
            intent_bucket, count, field="intent_bucket"
        )
        asset_rows = normalize_asset_rows(assets, count)
        contract_rows = normalize_contract_rows(
            expected_ui_contract, count
        )
    except ValueError:
        log_metric = kwargs.get("log_metric")
        if callable(log_metric):
            log_metric("grpo/input_shape_errors", 1.0)
        raise
    grouped: dict[str, list[int]] = defaultdict(list)
    rows = list(zip(responses, intents, asset_rows, contract_rows))
    for index, (source, intent, asset_row, contract) in enumerate(rows):
        key = json.dumps(
            {
                "source": str(source or ""),
                "intent": None if intent is None else str(intent),
                "assets": asset_row,
                "contract": contract if isinstance(contract, Mapping) else None,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        grouped[key].append(index)
    ordered_results: list[RewardBreakdown | None] = [None] * count
    reward_latencies_ms: list[float] = []
    for indices in grouped.values():
        first = indices[0]
        source, intent, asset_row, contract = rows[first]
        prepared = prepare_source_context_v5_3(
            str(source or ""),
            intent=None if intent is None else str(intent),
            assets=asset_row,
            expected_ui_contract=(
                contract if isinstance(contract, Mapping) else None
            ),
            config=coerce_reward_config_v5_3(config),
            computed_registry=computed_registry,
        )
        started = time.perf_counter()
        group_results = score_completion_group_v5_3(
            [completions[index] for index in indices],
            prepared,
            computed_registry=computed_registry,
        )
        elapsed_each = (
            (time.perf_counter() - started) * 1000.0
            / max(1, len(indices))
        )
        reward_latencies_ms.extend([elapsed_each] * len(indices))
        for index, result in zip(indices, group_results):
            ordered_results[index] = result
    results = [
        result
        for result in ordered_results
        if result is not None
    ]
    if len(results) != count:
        raise RuntimeError("GRPO group scorer did not return one result per completion")

    log_extra = kwargs.get("log_extra")
    if callable(log_extra):
        log_extra("genui_quality_0_100", [round(item.quality_0_100, 3) for item in results])
        log_extra("genui_reward", [round(item.reward, 6) for item in results])
        log_extra("genui_cap_0_1", [round(item.cap_0_1, 6) for item in results])
        log_extra("genui_metric_version", [item.metric_version for item in results])
        log_extra(
            "genui_metric_fingerprint",
            [item.metric_fingerprint for item in results],
        )
        log_extra(
            "genui_reward_pipeline_fingerprint",
            [
                item.reward_pipeline_fingerprint
                for item in results
            ],
        )
        log_extra(
            "genui_contract_version",
            [str((item.evidence.get("source") or {}).get("contract_version") or "") for item in results],
        )
        log_extra("genui_intent_bucket", [str(value or "unknown") for value in intents])
        model_checkpoint = _first_extra(kwargs, "model_checkpoint", "policy_checkpoint")
        if model_checkpoint is not None:
            log_extra(
                "genui_model_checkpoint",
                [str(value or "unknown") for value in _broadcast_metadata(model_checkpoint, count, "unknown")],
            )
        source_model_family = kwargs.get("source_model_family")
        if source_model_family is not None:
            log_extra(
                "genui_source_model_family",
                [str(value or "unknown") for value in _broadcast_metadata(source_model_family, count, "unknown")],
            )
        for dimension in DEFAULT_DIMENSION_WEIGHTS:
            log_extra(
                f"genui_{dimension}",
                [
                    None if item.dimensions.get(dimension) is None else round(float(item.dimensions[dimension]), 6)
                    for item in results
                ],
            )

    log_metric = kwargs.get("log_metric")
    if callable(log_metric) and results:
        rewards = [item.reward for item in results]
        completion_texts = [completion_to_text(value) for value in completions]
        completion_char_lengths = [len(value) for value in completion_texts]
        raw_completion_ids = _first_extra(kwargs, "completion_ids", "generated_token_ids")
        completion_ids = _broadcast_metadata(raw_completion_ids, count, None)
        completion_token_lengths = [
            len(value)
            for value in completion_ids
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping))
        ]
        raw_truncated = _first_extra(
            kwargs,
            "completion_truncated",
            "completion_is_truncated",
            "is_truncated",
            "truncated",
        )
        truncation_flags = [bool(value) for value in _broadcast_metadata(raw_truncated, count, False)]
        grouped: dict[str, list[float]] = defaultdict(list)
        raw_source_ids = kwargs.get("source_id")
        group_keys = (
            _broadcast_metadata(raw_source_ids, count, None)
            if raw_source_ids is not None
            else responses
        )
        for source_id, reward in zip(group_keys, rewards):
            grouped[str(source_id)].append(reward)
        comparable_groups = [values for values in grouped.values() if len(values) > 1]
        zero_variance = (
            sum(max(values) - min(values) <= 1e-12 for values in comparable_groups) / len(comparable_groups)
            if comparable_groups
            else 0.0
        )
        log_metric("genui/reward_mean", statistics.fmean(rewards))
        log_metric("genui/reward_sd", statistics.pstdev(rewards) if len(rewards) > 1 else 0.0)
        log_metric("genui/zero_variance_group_fraction", zero_variance)
        log_metric("genui/quality_mean", statistics.fmean(item.quality_0_100 for item in results))
        log_metric("reward/p50_ms", _percentile(reward_latencies_ms, 0.50))
        log_metric("reward/p95_ms", _percentile(reward_latencies_ms, 0.95))
        for field, key in (
            (
                "matching/vertex_edge_coverage_complete",
                "vertex_edge_coverage_complete",
            ),
            (
                "matching/full_cardinality_exists",
                "full_cardinality_matching_exists",
            ),
            (
                "matching/optimality_certified",
                "optimality_certified",
            ),
            (
                "matching/approximate_used",
                "approximate_matching_used",
            ),
        ):
            log_metric(
                field,
                statistics.fmean(
                    float(bool(item.matching_certification.get(key)))
                    for item in results
                ),
            )
        for field, key in (
            ("matching/evaluated_edges", "evaluated_edge_count"),
            (
                "matching/exact_preallocated_count",
                "exact_preallocated_count",
            ),
            (
                "matching/fuzzy_residual_count",
                "fuzzy_residual_count",
            ),
            ("dynamic/unknown_count", "unknown_count"),
        ):
            source = (
                "dynamic_semantics"
                if field.startswith("dynamic/")
                else "matching_certification"
            )
            log_metric(
                field,
                statistics.fmean(
                    float(getattr(item, source).get(key) or 0.0)
                    for item in results
                ),
            )
        log_metric(
            "contract/semantic_specificity",
            statistics.fmean(
                float(
                    (
                        item.evidence.get("source")
                        if isinstance(
                            item.evidence.get("source"), Mapping
                        )
                        else {}
                    ).get("contract_semantic_specificity")
                    or 0.0
                )
                for item in results
            ),
        )
        log_metric(
            "contract/count_only_role_count",
            statistics.fmean(
                float(
                    (
                        item.evidence.get("source")
                        if isinstance(
                            item.evidence.get("source"), Mapping
                        )
                        else {}
                    ).get("count_only_role_count")
                    or 0.0
                )
                for item in results
            ),
        )
        log_metric("grpo/input_shape_errors", 0.0)
        log_metric("genui/completion_char_length_mean", statistics.fmean(completion_char_lengths))
        if completion_token_lengths:
            log_metric("genui/completion_token_length_mean", statistics.fmean(completion_token_lengths))
        if raw_truncated is not None:
            log_metric("genui/completion_truncation_rate", statistics.fmean(truncation_flags))
        log_metric("genui/parse_failure_rate", sum(item.parse_stage not in {"json", "mapping"} for item in results) / len(results))
        log_metric(
            "genui/production_invalid_rate",
            sum(not bool(item.normalization.get("production_valid")) for item in results)
            / len(results),
        )
        log_metric(
            "genui/strict_schema_invalid_rate",
            sum(not bool(item.normalization.get("strict_schema_valid")) for item in results)
            / len(results),
        )
        log_metric("genui/root_failure_rate", sum(any(cap.get("name") == "missing_root" for cap in item.active_caps) for item in results) / len(results))
        log_metric(
            "genui/active_cap_rate",
            sum(bool(item.active_caps) for item in results) / len(results),
        )
        log_metric(
            "genui/binding_cap_rate",
            sum(bool(item.binding_caps) for item in results) / len(results),
        )
        # Deprecated compatibility alias: historically this measured active caps.
        log_metric(
            "genui/capped_rate",
            sum(bool(item.active_caps) for item in results) / len(results),
        )
        log_metric(
            "genui/render_failure_rate",
            sum(
                any(cap.get("name") == "render_failure" for cap in item.active_caps)
                for item in results
            )
            / len(results),
        )
        output_evidence = [item.evidence.get("output") or {} for item in results]
        component_counts = [float(value.get("reachable_elements") or 0.0) for value in output_evidence]
        log_metric("genui/component_count_mean", statistics.fmean(component_counts))
        log_metric(
            "genui/reference_failure_rate",
            sum(bool(value.get("missing_references")) for value in output_evidence) / len(results),
        )
        log_metric(
            "genui/cycle_failure_rate",
            sum(bool(value.get("cycles")) for value in output_evidence) / len(results),
        )
        cap_names = sorted(
            {
                str(cap.get("name"))
                for item in results
                for cap in item.active_caps
                if cap.get("name")
            }
        )
        for cap_name in cap_names:
            log_metric(
                f"genui/cap/{cap_name}",
                sum(
                    any(str(cap.get("name")) == cap_name for cap in item.active_caps)
                    for item in results
                )
                / len(results),
            )
        for atomic_name in (
            "markdown_table_fidelity",
            "action_and_source_link_fidelity",
        ):
            values = [
                float(item.atomics["fidelity"][atomic_name])
                for item in results
                if item.atomics.get("fidelity", {}).get(atomic_name) is not None
            ]
            if values:
                log_metric(f"genui/{atomic_name}_mean", statistics.fmean(values))
        for dimension in DEFAULT_DIMENSION_WEIGHTS:
            values = [float(item.dimensions[dimension]) for item in results if item.dimensions.get(dimension) is not None]
            if values:
                log_metric(f"genui/{dimension}_mean", statistics.fmean(values))

        inflation_monitor = kwargs.get("inflation_monitor")
        if isinstance(inflation_monitor, RewardInflationMonitor):
            fidelity_values = [
                float(item.dimensions["fidelity"])
                for item in results
                if item.dimensions.get("fidelity") is not None
            ]
            if fidelity_values:
                checkpoint = _first_extra(kwargs, "model_checkpoint", "policy_checkpoint")
                cohort = str(checkpoint or "default")
                length_values = completion_token_lengths or completion_char_lengths
                alert_metrics = inflation_monitor.observe(
                    component_count=statistics.fmean(component_counts),
                    completion_length=statistics.fmean(length_values),
                    fidelity=statistics.fmean(fidelity_values),
                    cohort=cohort,
                )
                for name, value in alert_metrics.items():
                    log_metric(f"genui/{name}", value)

    return [float(item.reward) for item in results]


def genui_grpo_reward_v5_2(
    completions: Sequence[Any],
    response_text: Sequence[str] | str,
    intent_bucket: Sequence[str] | str | None = None,
    assets: Sequence[Any] | Any = None,
    expected_ui_contract: (
        Sequence[Mapping[str, Any] | None]
        | Mapping[str, Any]
        | None
    ) = None,
    *,
    config: RewardConfig | None = None,
    computed_registry: ComputedFunctionRegistryV52 | None = None,
    **_: Any,
) -> list[float]:
    """Frozen v5.2 reward adapter for historical replay only."""

    count = len(completions)
    responses = normalize_scalar_or_vector_text(
        response_text, count, field="response_text"
    )
    intents = normalize_optional_scalar_or_vector_text(
        intent_bucket, count, field="intent_bucket"
    )
    asset_rows = normalize_asset_rows(assets, count)
    contracts = normalize_contract_rows(expected_ui_contract, count)
    grouped: dict[str, list[int]] = defaultdict(list)
    rows = list(zip(responses, intents, asset_rows, contracts))
    for index, row in enumerate(rows):
        grouped[
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        ].append(index)
    results: list[RewardBreakdown | None] = [None] * count
    for indices in grouped.values():
        source, intent, asset_row, contract = rows[indices[0]]
        prepared = prepare_source_context_v5_2(
            source,
            intent=intent,
            assets=asset_row,
            expected_ui_contract=contract,
            config=config,
            computed_registry=computed_registry,
        )
        scored = score_completion_group_v5_2(
            [completions[index] for index in indices],
            prepared,
            computed_registry=computed_registry,
        )
        for index, result in zip(indices, scored):
            results[index] = result
    if any(result is None for result in results):
        raise RuntimeError("v5.2 reward replay returned an incomplete group")
    return [float(result.reward) for result in results if result is not None]


def make_genui_grpo_reward(
    config: RewardConfigV53,
    *,
    model_checkpoint: str | None = None,
    inflation_monitor: RewardInflationMonitor | None = None,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
) -> Callable[..., list[float]]:
    def reward_function(
        completions: Sequence[Any],
        response_text: Sequence[str] | str,
        intent_bucket: Sequence[str] | str | None = None,
        assets: Sequence[Any] | Any = None,
        expected_ui_contract: Sequence[Mapping[str, Any] | None] | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        if model_checkpoint is not None:
            kwargs.setdefault("model_checkpoint", model_checkpoint)
        if inflation_monitor is not None:
            kwargs.setdefault("inflation_monitor", inflation_monitor)
        return genui_grpo_reward(
            completions,
            response_text,
            intent_bucket=intent_bucket,
            assets=assets,
            expected_ui_contract=expected_ui_contract,
            config=config,
            computed_registry=computed_registry,
            **kwargs,
        )

    reward_function.__name__ = (
        f"genui_grpo_reward_v{REWARD_VERSION_V53.replace('.', '_')}"
    )
    return reward_function


__all__ = [
    "AssetCollection",
    "PerCompletionAssetCollections",
    "RewardBreakdown",
    "RewardInflationMonitor",
    "genui_grpo_reward",
    "genui_grpo_reward_v5_2",
    "make_genui_grpo_reward",
    "normalize_asset_rows",
    "normalize_contract_rows",
    "normalize_optional_scalar_or_vector_text",
    "normalize_scalar_or_vector_text",
]
