"""TRL-compatible GRPO reward entry points and structured diagnostics."""

from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Any, Callable, Mapping, Sequence

from .aggregate import RewardBreakdown, score_genui_completion
from .config import DEFAULT_DIMENSION_WEIGHTS, REWARD_VERSION, RewardConfig
from ._core import completion_to_text


def _broadcast(value: Any, count: int, default: Any = None) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        values = list(value)
        if len(values) == count:
            return values
    return [default if value is None else value for _ in range(count)]


def _first_extra(kwargs: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
    return None


def genui_grpo_reward(
    completions: Sequence[Any],
    response_text: Sequence[str] | str,
    intent_bucket: Sequence[str] | str | None = None,
    assets: Sequence[Any] | Any = None,
    expected_ui_contract: Sequence[Mapping[str, Any] | None] | Mapping[str, Any] | None = None,
    *,
    config: RewardConfig | None = None,
    **kwargs: Any,
) -> list[float]:
    """Score samples independently and return one bounded scalar per completion."""
    count = len(completions)
    responses = _broadcast(response_text, count, "")
    intents = _broadcast(intent_bucket, count, None)
    asset_rows = _broadcast(assets, count, None)
    contract_rows = _broadcast(expected_ui_contract, count, None)
    results = [
        score_genui_completion(
            completion,
            str(source or ""),
            intent=None if intent is None else str(intent),
            assets=asset_row,
            expected_ui_contract=contract if isinstance(contract, Mapping) else None,
            config=config,
        )
        for completion, source, intent, asset_row, contract in zip(
            completions, responses, intents, asset_rows, contract_rows
        )
    ]

    log_extra = kwargs.get("log_extra")
    if callable(log_extra):
        log_extra("genui_quality_0_100", [round(item.quality_0_100, 3) for item in results])
        log_extra("genui_reward", [round(item.reward, 6) for item in results])
        log_extra("genui_cap_0_1", [round(item.cap_0_1, 6) for item in results])
        log_extra("genui_metric_version", [item.metric_version for item in results])
        log_extra(
            "genui_contract_version",
            [str((item.evidence.get("source") or {}).get("contract_version") or "") for item in results],
        )
        log_extra("genui_intent_bucket", [str(value or "unknown") for value in intents])
        model_checkpoint = _first_extra(kwargs, "model_checkpoint", "policy_checkpoint")
        if model_checkpoint is not None:
            log_extra(
                "genui_model_checkpoint",
                [str(value or "unknown") for value in _broadcast(model_checkpoint, count, "unknown")],
            )
        source_model_family = kwargs.get("source_model_family")
        if source_model_family is not None:
            log_extra(
                "genui_source_model_family",
                [str(value or "unknown") for value in _broadcast(source_model_family, count, "unknown")],
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
        completion_ids = _broadcast(raw_completion_ids, count, None)
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
        truncation_flags = [bool(value) for value in _broadcast(raw_truncated, count, False)]
        grouped: dict[str, list[float]] = defaultdict(list)
        raw_source_ids = kwargs.get("source_id")
        group_keys = (
            _broadcast(raw_source_ids, count, None)
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
        log_metric("genui/completion_char_length_mean", statistics.fmean(completion_char_lengths))
        if completion_token_lengths:
            log_metric("genui/completion_token_length_mean", statistics.fmean(completion_token_lengths))
        if raw_truncated is not None:
            log_metric("genui/completion_truncation_rate", statistics.fmean(truncation_flags))
        log_metric("genui/parse_failure_rate", sum(item.parse_stage not in {"json", "mapping"} for item in results) / len(results))
        log_metric("genui/root_failure_rate", sum(any(cap.get("name") == "missing_root" for cap in item.active_caps) for item in results) / len(results))
        log_metric("genui/capped_rate", sum(item.cap_0_1 < 1.0 for item in results) / len(results))
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

    return [float(item.reward) for item in results]


def make_genui_grpo_reward(
    config: RewardConfig,
    *,
    model_checkpoint: str | None = None,
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
        return genui_grpo_reward(
            completions,
            response_text,
            intent_bucket=intent_bucket,
            assets=assets,
            expected_ui_contract=expected_ui_contract,
            config=config,
            **kwargs,
        )

    reward_function.__name__ = f"genui_grpo_reward_v{REWARD_VERSION.replace('.', '_')}"
    return reward_function


__all__ = ["RewardBreakdown", "genui_grpo_reward", "make_genui_grpo_reward"]
