"""Deterministic grouped GRPO reward dispatch for metric v5.4."""

from __future__ import annotations

from collections import defaultdict
import json
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from ._v5_4 import (
    prepare_source_context_v5_4,
    score_completion_group_v5_4,
)
from .config_v5_4 import (
    REWARD_VERSION_V54,
    RewardConfigV54,
    coerce_reward_config_v5_4,
)
from ._core import completion_to_text
from .evidence_v5_4 import ComputedFunctionRegistryV54
from .grpo_reward import (
    RewardInflationMonitor,
    normalize_asset_rows,
    normalize_contract_rows,
    normalize_optional_scalar_or_vector_text,
    normalize_scalar_or_vector_text,
)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def genui_grpo_reward_v5_4(
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
    config: RewardConfigV54 | None = None,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
    **kwargs: Any,
) -> list[float]:
    count = len(completions)
    responses = normalize_scalar_or_vector_text(
        response_text, count, field="response_text"
    )
    intents = normalize_optional_scalar_or_vector_text(
        intent_bucket, count, field="intent_bucket"
    )
    asset_rows = normalize_asset_rows(assets, count)
    contracts = normalize_contract_rows(expected_ui_contract, count)
    rows = list(zip(responses, intents, asset_rows, contracts))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (source, intent, asset_row, contract) in enumerate(rows):
        key = json.dumps(
            {
                "source": source,
                "intent": intent,
                "assets": asset_row,
                "contract": contract,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        grouped[key].append(index)

    cfg = coerce_reward_config_v5_4(config)
    ordered: list[Any | None] = [None] * count
    group_latencies: list[float] = []
    for indices in grouped.values():
        source, intent, asset_row, contract = rows[indices[0]]
        prepared = prepare_source_context_v5_4(
            str(source or ""),
            intent=None if intent is None else str(intent),
            assets=asset_row,
            expected_ui_contract=(
                contract if isinstance(contract, Mapping) else None
            ),
            config=cfg,
            computed_registry=computed_registry,
        )
        started = time.perf_counter()
        results = score_completion_group_v5_4(
            [completions[index] for index in indices],
            prepared,
            computed_registry=computed_registry,
            generation_mode=True,
            active_express=True,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        group_latencies.append(elapsed)
        for index, result in zip(indices, results):
            ordered[index] = result
    results = [item for item in ordered if item is not None]
    if len(results) != count:
        raise RuntimeError("v5.4 GRPO scorer did not return one result per completion")

    log_extra = kwargs.get("log_extra")
    if callable(log_extra):
        log_extra("genui_metric_version", [item.metric_version for item in results])
        log_extra("genui_quality_0_100", [item.quality_0_100 for item in results])
        log_extra("genui_reward", [item.reward for item in results])
        log_extra(
            "genui_artifact_quality_0_100",
            [item.artifact_quality_0_100 for item in results],
        )
        log_extra("genui_group_reward_ms", group_latencies)
        log_extra(
            "genui_raw_envelope_exact",
            [
                bool(item.raw_json_envelope.get("exact_single_json_value"))
                for item in results
            ],
        )
        log_extra(
            "genui_raw_express_envelope_exact",
            [
                bool(
                    item.evidence.get("raw_express_envelope", {}).get(
                        "exact_single_express_block"
                    )
                )
                for item in results
            ],
        )
        for dimension in sorted(
            {
                key
                for item in results
                for key in item.dimensions
            }
        ):
            log_extra(
                f"genui_{dimension}",
                [item.dimensions.get(dimension) for item in results],
            )
    log_metric = kwargs.get("log_metric")
    if callable(log_metric) and results:
        rewards = [float(item.reward) for item in results]
        log_metric("genui/reward_mean", statistics.fmean(rewards))
        log_metric(
            "genui/reward_sd",
            statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        )
        log_metric(
            "genui/quality_mean",
            statistics.fmean(item.quality_0_100 for item in results),
        )
        log_metric(
            "reward/group_p50_ms", _percentile(group_latencies, 0.50)
        )
        log_metric(
            "reward/group_p95_ms", _percentile(group_latencies, 0.95)
        )
        log_metric(
            "genui/raw_envelope_violation_rate",
            statistics.fmean(
                not bool(
                    item.raw_json_envelope.get(
                        "exact_single_json_value"
                    )
                )
                for item in results
            ),
        )
        log_metric(
            "genui/raw_express_envelope_violation_rate",
            statistics.fmean(
                not bool(
                    item.evidence.get("raw_express_envelope", {}).get(
                        "exact_single_express_block"
                    )
                )
                for item in results
            ),
        )
        log_metric(
            "genui/cap_activation_rate",
            statistics.fmean(bool(item.active_caps) for item in results),
        )
        for dimension in sorted(
            {
                key
                for item in results
                for key in item.dimensions
            }
        ):
            values = [
                float(item.dimensions[dimension])
                for item in results
                if isinstance(item.dimensions.get(dimension), (int, float))
            ]
            if values:
                log_metric(
                    f"genui/{dimension}_mean",
                    statistics.fmean(values),
                )
        monitor = kwargs.get("inflation_monitor")
        if isinstance(monitor, RewardInflationMonitor):
            fidelity = [
                float(item.dimensions["fidelity"])
                for item in results
                if isinstance(
                    item.dimensions.get("fidelity"), (int, float)
                )
            ]
            if fidelity:
                components = [
                    len(item.normalization.get("reachable_types") or ())
                    for item in results
                ]
                lengths = [
                    len(completion_to_text(completion))
                    for completion in completions
                ]
                alerts = monitor.observe(
                    component_count=statistics.fmean(components),
                    completion_length=statistics.fmean(lengths),
                    fidelity=statistics.fmean(fidelity),
                    cohort=str(
                        kwargs.get("model_checkpoint")
                        or kwargs.get("policy_checkpoint")
                        or "default"
                    ),
                )
                for name, value in alerts.items():
                    log_metric(f"genui/{name}", value)
    return [float(item.reward) for item in results]


def make_genui_grpo_reward_v5_4(
    config: RewardConfigV54,
    *,
    model_checkpoint: str | None = None,
    inflation_monitor: RewardInflationMonitor | None = None,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
) -> Callable[..., list[float]]:
    def reward_function(
        completions: Sequence[Any],
        response_text: Sequence[str] | str,
        intent_bucket: Sequence[str] | str | None = None,
        assets: Sequence[Any] | Any = None,
        expected_ui_contract: (
            Sequence[Mapping[str, Any] | None]
            | Mapping[str, Any]
            | None
        ) = None,
        **kwargs: Any,
    ) -> list[float]:
        if model_checkpoint is not None:
            kwargs.setdefault("model_checkpoint", model_checkpoint)
        if inflation_monitor is not None:
            kwargs.setdefault("inflation_monitor", inflation_monitor)
        return genui_grpo_reward_v5_4(
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
        f"genui_grpo_reward_v{REWARD_VERSION_V54.replace('.', '_')}"
    )
    return reward_function


__all__ = [
    "genui_grpo_reward_v5_4",
    "make_genui_grpo_reward_v5_4",
]
