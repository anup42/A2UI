"""Source-grounded GRPO reward for the official-format QAT LoRA pipelines.

This is a versioned training mapping of the existing v5.4 evidence, not a new
canonical evaluation metric. All parsing, renderer-visible evidence, matching,
reference semantics and source contract extraction remain owned by v5.4. The
mapping is an uncalibrated engineering objective; held-out evaluation is still
needed to establish learning quality.

Use ``audited_reward(make_qat_grpo_reward(...), ...)`` from ``grpo_runtime``.
That shared boundary applies serving stops and restores URL placeholders. The
runner must supply training-split sources; this module does not open datasets,
read reference completions, call a judge, repair candidates, or use Golden data.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

QAT_GRPO_REWARD_VERSION = "qat_source_fidelity_v1"

# Renormalize only over source-applicable atomics. These are all source/output
# matches, never completion length, component count, novelty, or group rank.
FIDELITY_WEIGHTS = {
    "content_unit_fidelity": 0.25,
    "visible_content_multiset_fbeta": 0.15,
    "content_order_preservation": 0.05,
    "exact_numbers_dates_units_fbeta": 0.15,
    "markdown_table_fidelity": 0.15,
    "heading_fidelity_and_order": 0.05,
    "action_and_source_link_fidelity": 0.15,
    "media_fidelity": 0.05,
}
CRITICAL_FIDELITY = (
    "exact_numbers_dates_units_fbeta", "markdown_table_fidelity",
    "action_and_source_link_fidelity", "media_fidelity",
)


@dataclass(frozen=True)
class QATRewardBreakdown:
    reward: float
    quality: float
    strict_valid: bool
    components: dict[str, Any]
    reasons: tuple[str, ...]
    canonical_reward: float
    canonical_quality: float
    canonical_caps: tuple[dict[str, Any], ...]
    metric_fingerprint: str
    policy_version: str = QAT_GRPO_REWARD_VERSION


def _unit(value: Any, name: str) -> float:
    """A broken scorer is an execution error, never silent positive reward."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Non-finite or missing v5.4 reward component: {name}")
    if not -1e-9 <= float(value) <= 1.0 + 1e-9:
        raise ValueError(f"Out-of-range v5.4 reward component: {name}={value}")
    return min(1.0, max(0.0, float(value)))


def map_qat_reward(result: Any) -> QATRewardBreakdown:
    """Map one v5.4 breakdown without modifying its evidence or score.

    Invalid raw Express gets -1. For valid candidates, Q is the product of
    source fidelity, inherited cap, precision/non-duplication, and critical
    value preservation. Required semantic roles contribute at most 10%; the
    canonical score contributes at most 5%. Thus valid-looking layout cannot
    compensate for missing visible facts. Multiplicative caps retain genuine
    differences between otherwise capped candidates rather than manufacturing
    variance with noise or novelty bonuses. Reward is exactly ``2 * Q - 1``.
    """
    normalization = result.normalization
    atomics = result.atomics
    fidelity = atomics.get("fidelity", {})
    integrity = atomics.get("integrity", {})
    envelope = result.evidence.get("raw_express_envelope", {})
    reasons = []
    for name, valid in (
        ("raw_express_envelope", envelope.get("exact_single_express_block")),
        ("native_syntax", normalization.get("native_syntax_valid")),
        ("native_catalog", normalization.get("native_catalog_valid")),
        ("production", normalization.get("production_valid")),
        ("strict_schema", normalization.get("strict_schema_valid")),
    ):
        if valid is not True:
            reasons.append("invalid:" + name)
    for name in ("declared_root_reachability", "reference_integrity", "cycle_free"):
        if integrity.get(name) != 1.0:
            reasons.append("invalid:" + name)
    strict_valid = not reasons
    canonical_quality = _unit(result.quality_0_1, "quality_0_1")
    cap = _unit(result.cap_0_1, "cap_0_1")
    applicable = {
        name: _unit(fidelity[name], name)
        for name in FIDELITY_WEIGHTS if fidelity.get(name) is not None
    }
    weight_sum = sum(FIDELITY_WEIGHTS[name] for name in applicable)
    source_fidelity = (
        sum(FIDELITY_WEIGHTS[name] * value for name, value in applicable.items()) / weight_sum
        if weight_sum else 0.0
    )
    if not applicable:
        reasons.append("no_applicable_source_fidelity")
    critical = {name: applicable[name] for name in CRITICAL_FIDELITY if name in applicable}
    critical_min = min(critical.values(), default=1.0)
    critical_guard = 0.25 + 0.75 * critical_min
    precision = {
        name: _unit(fidelity[name], name)
        for name in ("output_block_precision", "unsupported_external_addition_precision")
        if fidelity.get(name) is not None
    }
    non_duplication = atomics.get("economy", {}).get("semantic_non_duplication")
    if non_duplication is not None:
        precision["semantic_non_duplication"] = _unit(non_duplication, "semantic_non_duplication")
    precision_guard = min(precision.values(), default=1.0)
    semantic = {
        name: _unit(value, name)
        for name, value in atomics.get("semantic_mapping", {}).items()
        if name in {"role_count_coverage", "semantic_role_instance_fidelity"} and value is not None
    }
    semantic_guard = 0.9 + 0.1 * min(semantic.values(), default=1.0)
    structure_guard = 0.95 + 0.05 * canonical_quality
    components = {
        "applicable_fidelity": applicable,
        "effective_fidelity_weights": {
            name: FIDELITY_WEIGHTS[name] / weight_sum for name in applicable
        },
        "source_fidelity": source_fidelity,
        "critical_fidelity": critical,
        "critical_guard": critical_guard,
        "precision": precision,
        "precision_guard": precision_guard,
        "semantic_fidelity": semantic,
        "semantic_guard": semantic_guard,
        "structure_guard": structure_guard,
        "canonical_cap": cap,
    }
    reasons.extend("fidelity:" + name for name, value in applicable.items() if value < 1.0 - 1e-9)
    reasons.extend("precision:" + name for name, value in precision.items() if value < 1.0 - 1e-9)
    reasons.extend("semantic:" + name for name, value in semantic.items() if value < 1.0 - 1e-9)
    reasons.extend("cap:" + str(item.get("name")) for item in result.active_caps)
    quality = _unit(
        cap * source_fidelity * critical_guard * precision_guard * semantic_guard * structure_guard
        if strict_valid else 0.0,
        "qat_quality",
    )
    return QATRewardBreakdown(
        reward=2.0 * quality - 1.0, quality=quality, strict_valid=strict_valid,
        components=components, reasons=tuple(reasons),
        canonical_reward=float(result.reward), canonical_quality=canonical_quality,
        canonical_caps=tuple(dict(item) for item in result.active_caps),
        metric_fingerprint=str(result.metric_fingerprint),
    )


def make_qat_grpo_reward(
    config: Any = None, *, model_checkpoint: str | None = None,
) -> Callable[..., list[float]]:
    """Build a deterministic TRL callable using grouped, shared v5.4 scoring.

    Source preparation is shared within each identical source/context group,
    using the same normalization helpers and scorer as canonical GRPO. Dataset
    ``reference_completion`` and other TRL metadata are deliberately ignored.
    This preserves alternative faithful UIs instead of matching SFT text.
    """
    from pipeline.genui_quality._v5_4 import (
        prepare_source_context_v5_4,
        score_completion_group_v5_4,
    )
    from pipeline.genui_quality.config_v5_4 import coerce_reward_config_v5_4
    from pipeline.genui_quality.grpo_reward import (
        normalize_asset_rows,
        normalize_contract_rows,
        normalize_optional_scalar_or_vector_text,
        normalize_scalar_or_vector_text,
    )

    cfg = coerce_reward_config_v5_4(config)
    if cfg.unsupported_external_additions_policy != "penalize":
        raise ValueError("QAT GRPO requires unsupported_external_additions_policy=penalize")
    policy_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def reward_function(
        completions: Sequence[Any], response_text: Sequence[str] | str,
        intent_bucket: Sequence[str] | str | None = None, assets: Any = None,
        expected_ui_contract: Any = None, **kwargs: Any,
    ) -> list[float]:
        count = len(completions)
        sources = normalize_scalar_or_vector_text(response_text, count, field="response_text")
        if any(not source.strip() for source in sources):
            raise ValueError("QAT GRPO requires a nonempty training response_text for every candidate")
        intents = normalize_optional_scalar_or_vector_text(intent_bucket, count, field="intent_bucket")
        rows = list(zip(sources, intents, normalize_asset_rows(assets, count),
                        normalize_contract_rows(expected_ui_contract, count), strict=True))
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)].append(index)
        ordered: list[QATRewardBreakdown | None] = [None] * count
        for indices in groups.values():
            source, intent, asset_row, contract = rows[indices[0]]
            prepared = prepare_source_context_v5_4(
                source, intent=intent, assets=asset_row,
                expected_ui_contract=contract if isinstance(contract, Mapping) else None,
                config=cfg,
            )
            results = score_completion_group_v5_4(
                [completions[index] for index in indices], prepared,
                generation_mode=True, active_express=True,
            )
            for index, result in zip(indices, results, strict=True):
                ordered[index] = map_qat_reward(result)
        if any(item is None for item in ordered):
            raise RuntimeError("QAT reward scorer did not return every candidate")
        scored = [item for item in ordered if item is not None]
        values = [item.reward for item in scored]
        log_extra = kwargs.get("log_extra")
        if callable(log_extra):
            log_extra("qat_reward", [asdict(item) for item in scored])
            log_extra("qat_reward_policy_sha256", [policy_hash] * count)
            log_extra("qat_reward_model_checkpoint", [model_checkpoint] * count)
            # Preserve canonical score separately for analysis; it is not the
            # policy's optimized reward and must never be reported as such.
            log_extra("genui_reward", [item.canonical_reward for item in scored])
            log_extra("genui_quality_0_100", [100.0 * item.canonical_quality for item in scored])
        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and scored:
            log_metric("qat_reward/mean", statistics.fmean(values))
            log_metric("qat_reward/sd", statistics.pstdev(values))
            log_metric("qat_reward/strict_valid_fraction", statistics.fmean(item.strict_valid for item in scored))
        return values

    reward_function.__name__ = QAT_GRPO_REWARD_VERSION
    return reward_function


__all__ = ["QAT_GRPO_REWARD_VERSION", "QATRewardBreakdown", "make_qat_grpo_reward", "map_qat_reward"]
