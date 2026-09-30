"""Safe v5.4 score reuse and immutable per-record aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import MISSING, fields
from typing import Any, Mapping, Sequence

from ._v5_4 import (
    generation_reward_a2ui_express_v1,
    generation_reward_v5_4,
    render_artifact_quality_v5_4,
)
from .aggregate import (
    _distribution,
    _native_render_status,
    _record_candidate,
    _record_final_candidate,
)
from .candidate_normalization_v5_4 import (
    normalize_and_validate_express_candidate_v5_4,
    normalize_and_validate_legacy_candidate_v5_4,
)
from .config_v5_4 import (
    REWARD_VERSION_V54,
    RewardBreakdownV54,
    RewardConfigV54,
    load_v5_4_reward_config,
)
from .evidence_v5_4 import ComputedFunctionRegistryV54
from .identity_v5_4 import (
    METRIC_NAME,
    expected_contract_hash_v5_4,
    metric_fingerprint_v5_4,
    reward_pipeline_fingerprint_v5_4,
)
from .source_contract_v5_4 import resolve_expected_ui_contract_v5_4


def breakdown_from_mapping_v5_4(
    value: Any,
) -> RewardBreakdownV54 | None:
    if not isinstance(value, Mapping):
        return None
    if str(value.get("metric_version") or "") != REWARD_VERSION_V54:
        return None
    payload: dict[str, Any] = {}
    try:
        for item in fields(RewardBreakdownV54):
            if item.name in value:
                payload[item.name] = value[item.name]
            elif item.default is not MISSING:
                payload[item.name] = item.default
            elif item.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                payload[item.name] = item.default_factory()  # type: ignore[misc]
            else:
                return None
        return RewardBreakdownV54(**payload)
    except (TypeError, ValueError):
        return None


def _score_record_variant(
    record: Mapping[str, Any],
    *,
    candidate: Any,
    stored_keys: Sequence[str],
    generation_mode: bool,
    render_row: Mapping[str, Any] | None,
    config: RewardConfigV54,
    computed_registry: ComputedFunctionRegistryV54 | None,
) -> RewardBreakdownV54:
    response_text = str(record.get("response_text") or "")
    intent = str(
        record.get("intent_bucket") or record.get("intent") or ""
    ) or None
    assets = record.get("assets")
    persisted = (
        record.get("expected_ui_contract_v5_4")
        if isinstance(record.get("expected_ui_contract_v5_4"), Mapping)
        else None
    )
    persisted_source = record.get("expected_ui_contract_v5_4_source")
    resolution = resolve_expected_ui_contract_v5_4(
        response_text,
        intent=intent,
        assets=assets,
        persisted=persisted,
        persisted_source=(
            str(persisted_source) if persisted_source else None
        ),
    )
    active_express = _is_active_express_record(record, candidate)
    if active_express:
        candidate = _active_express_candidate(record)
    normalization = (
        normalize_and_validate_express_candidate_v5_4(candidate, reference_map=record.get("reference_map"))
        if active_express
        else normalize_and_validate_legacy_candidate_v5_4(candidate, reference_map=record.get("reference_map"))
    )
    fingerprint = metric_fingerprint_v5_4(
        config, computed_registry=computed_registry
    )
    pipeline = reward_pipeline_fingerprint_v5_4(fingerprint)
    expected_identity = {
        "source_hash": str(resolution.contract.get("source_hash") or ""),
        "raw_candidate_hash": normalization.raw_hash,
        "canonical_candidate_hash": normalization.canonical_hash,
        "reference_map_hash": normalization.reference_map_hash,
        "expected_contract_hash": expected_contract_hash_v5_4(
            resolution.contract
        ),
    }
    attempted, render_ok = (
        _native_render_status(record, render_row)
        if not generation_mode
        else (False, None)
    )
    stored = next(
        (
            parsed
            for key in stored_keys
            for parsed in [breakdown_from_mapping_v5_4(record.get(key))]
            if parsed is not None
        ),
        None,
    )
    stale: list[str] = []
    if stored is None:
        stale.append("missing_or_invalid_stored_v5_4")
    else:
        if stored.metric_fingerprint != fingerprint:
            stale.append("metric_fingerprint_mismatch")
        if stored.reward_pipeline_fingerprint != pipeline:
            stale.append("reward_pipeline_fingerprint_mismatch")
        for key, expected in expected_identity.items():
            if stored.identity.get(key) != expected:
                stale.append(f"{key}_mismatch")
        expected_render = render_ok if attempted else None
        if stored.evidence.get("render_ok") is not expected_render:
            stale.append("render_evidence_mismatch")
        if bool(stored.raw_json_envelope.get("generation_mode")) != generation_mode:
            stale.append("score_variant_mismatch")
    if stored is not None and not stale:
        stored.evidence["score_reuse"] = {
            "reused": True,
            "stale_reasons": [],
        }
        return stored
    if generation_mode and active_express:
        scorer = generation_reward_a2ui_express_v1
    else:
        scorer = generation_reward_v5_4 if generation_mode else render_artifact_quality_v5_4
    score_kwargs = {
        "intent": intent,
        "assets": assets,
        "expected_ui_contract": resolution.contract,
        "expected_ui_contract_source": resolution.source,
        "render_ok": render_ok if attempted else None,
        "config": config,
        "computed_registry": computed_registry,
        "reference_map": record.get("reference_map"),
    }
    if not active_express:
        score_kwargs["legacy_comparison"] = True
    result = scorer(candidate, response_text, **score_kwargs)
    result.evidence["score_reuse"] = {
        "reused": False,
        "stale_reasons": list(dict.fromkeys(stale)),
    }
    return result


def _is_active_express_record(record: Mapping[str, Any], candidate: Any) -> bool:
    target = str(record.get("target_format") or "").strip().lower()
    if target == "a2ui_express_v1":
        return True
    source = str(record.get("source_format") or "").strip().lower()
    if source == "a2ui_express_v1":
        return True
    if target in {"flat_spec_v1", "compact_ir_v2", "compact_ir", "gci2"}:
        return False
    if source in {"flat_spec_v1", "compact_ir_v2", "compact_ir", "gci2"}:
        return False
    # Active records are Express by contract.  A missing format tag must not
    # silently re-open the legacy JSON normalizer.
    return True


def _active_express_candidate(record: Mapping[str, Any]) -> Any:
    for key in (
        "genui_raw_completion",
        "a2ui_express",
        "raw_completion",
        "completion",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # A canonical graph without its source completion is not an active
    # candidate; returning None makes the Express boundary fail closed.
    return None


def score_record_v5_4(
    record: Mapping[str, Any],
    *,
    render_row: Mapping[str, Any] | None = None,
    config: RewardConfigV54 | None = None,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
) -> RewardBreakdownV54:
    return _score_record_variant(
        record,
        candidate=_record_final_candidate(record),
        stored_keys=(
            "render_artifact_quality_v5_4",
            "genui_quality_v5_4",
        ),
        generation_mode=False,
        render_row=render_row,
        config=config or load_v5_4_reward_config(),
        computed_registry=computed_registry,
    )


def score_record_generation_v5_4(
    record: Mapping[str, Any],
    *,
    config: RewardConfigV54 | None = None,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
) -> RewardBreakdownV54:
    return _score_record_variant(
        record,
        candidate=_record_candidate(record),
        stored_keys=("generation_reward_v5_4",),
        generation_mode=True,
        render_row=None,
        config=config or load_v5_4_reward_config(),
        computed_registry=computed_registry,
    )


def aggregate_v5_4_records(
    records: Sequence[Mapping[str, Any]],
    *,
    render_rows_by_ui_id: Mapping[str, Mapping[str, Any]] | None = None,
    config: RewardConfigV54 | None = None,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
) -> dict[str, Any]:
    cfg = config or load_v5_4_reward_config()
    rows: list[tuple[RewardBreakdownV54, RewardBreakdownV54]] = []
    for record in records:
        ui_id = record.get("ui_id")
        render = (
            render_rows_by_ui_id.get(str(ui_id))
            if render_rows_by_ui_id is not None and ui_id is not None
            else None
        )
        rows.append(
            (
                score_record_v5_4(
                    record,
                    render_row=render,
                    config=cfg,
                    computed_registry=computed_registry,
                ),
                score_record_generation_v5_4(
                    record,
                    config=cfg,
                    computed_registry=computed_registry,
                ),
            )
        )
    final = [item.quality_0_100 for item, _ in rows]
    generation = [item.quality_0_100 for _, item in rows]
    rewards = [item.reward for _, item in rows]
    dimensions: dict[str, list[float]] = defaultdict(list)
    caps: dict[str, int] = defaultdict(int)
    stale: dict[str, int] = defaultdict(int)
    reused = 0
    for result, _ in rows:
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
    count = len(rows)
    fingerprint = metric_fingerprint_v5_4(
        cfg, computed_registry=computed_registry
    )
    return {
        "metric_name": METRIC_NAME,
        "metric_version": REWARD_VERSION_V54,
        "metric_fingerprint": fingerprint,
        "reward_pipeline_fingerprint": reward_pipeline_fingerprint_v5_4(
            fingerprint
        ),
        "calibration_status": "uncalibrated_engineering_score",
        "headline": "render_artifact_quality_v5_4",
        "count": count,
        "quality_0_100": _distribution(final),
        "generation_reward_quality_0_100": _distribution(generation),
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
    }


__all__ = [
    "aggregate_v5_4_records",
    "breakdown_from_mapping_v5_4",
    "score_record_generation_v5_4",
    "score_record_v5_4",
]
