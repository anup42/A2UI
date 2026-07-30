from __future__ import annotations

from dataclasses import replace
import copy
import math
from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_support import compact, contract, score, simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    breakdown_to_mapping,
    load_v5_1_reward_config as load_default_reward_config,
    score_genui_completion_v5_1 as score_genui_completion,
    score_record_v5_1 as score_record,
)
from pipeline.genui_quality._v5 import allocate_capped_atomic_weights  # noqa: E402
from pipeline.genui_quality.source_contract import (  # noqa: E402
    resolve_expected_ui_contract_v5_1 as resolve_expected_ui_contract,
    source_contract_cache_key_v5_1 as source_contract_cache_key,
)
import pipeline.genui_quality.identity_v5_1 as metric_identity  # noqa: E402
from pipeline.genui_quality.identity_v5_1 import (  # noqa: E402
    metric_fingerprint_v5_1 as metric_fingerprint,
)


def test_effective_global_atomic_weight_invariants() -> None:
    result = score(simple_spec(), "Hello world")
    config = load_default_reward_config()
    assert result.anti_domination_feasible
    assert result.applicable_atomic_count >= 10
    assert math.isclose(sum(result.effective_atomic_weights.values()), 1.0, abs_tol=1e-10)
    assert max(result.effective_atomic_weights.values()) <= config.max_atomic_global_weight + 1e-10
    assert math.isclose(
        sum(result.effective_dimension_weights.values()),
        1.0,
        abs_tol=1e-10,
    )


def test_one_atomic_cannot_move_base_more_than_cap() -> None:
    config = load_default_reward_config()
    atomics = {
        dimension: {name: 0.5 for name in values}
        for dimension, values in config.atomic_weights.items()
    }
    weights, _, feasible = allocate_capped_atomic_weights(atomics, config)
    assert feasible
    key = sorted(weights)[0]
    dimension, atomic = key.split(".", 1)
    before = sum(
        weights[f"{group}.{name}"] * value
        for group, values in atomics.items()
        for name, value in values.items()
        if f"{group}.{name}" in weights
    )
    atomics[dimension][atomic] = 1.0
    after = sum(
        weights[f"{group}.{name}"] * value
        for group, values in atomics.items()
        for name, value in values.items()
        if f"{group}.{name}" in weights
    )
    assert after - before <= config.max_atomic_global_weight + 1e-12


def test_metric_fingerprint_changes_with_resolved_config() -> None:
    config = load_default_reward_config()
    changed = replace(config, content_beta=config.content_beta + 0.25)
    assert metric_fingerprint(config) != metric_fingerprint(changed)


def test_schema_and_renderer_semantics_change_metric_fingerprint(
    monkeypatch,
    tmp_path,
) -> None:
    config = load_default_reward_config()
    schema_copy = tmp_path / "genui_flatspec.schema.json"
    schema_copy.write_text(
        metric_identity.DEFAULT_STRICT_SCHEMA_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        metric_identity,
        "DEFAULT_STRICT_SCHEMA_PATH",
        schema_copy,
    )
    before = metric_fingerprint(config)
    schema_copy.write_text(
        schema_copy.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    after_schema = metric_fingerprint(config)
    assert after_schema != before

    monkeypatch.setattr(
        metric_identity,
        "RENDERER_SEMANTICS_VERSION",
        "test-changed-semantics",
    )
    after_semantics = metric_fingerprint(config)
    assert after_semantics != after_schema


def test_anti_domination_infeasible_policy_fails_closed_with_diagnostics() -> None:
    config = replace(
        load_default_reward_config(),
        max_atomic_global_weight=0.01,
    )
    result = score_genui_completion(
        compact(simple_spec("Hello")),
        "Hello",
        config=config,
    )
    assert not result.anti_domination_feasible
    assert result.applicable_atomic_count > 0
    assert result.quality_0_1 == 0.0
    assert any(
        cap["name"] == "anti_domination_infeasible"
        for cap in result.active_caps
    )


def test_intent_sensitive_contract_cache_keys_differ() -> None:
    assert source_contract_cache_key("Same source", intent="booking") != source_contract_cache_key(
        "Same source",
        intent="technical_support",
    )


def test_stale_persisted_source_hash_is_rejected() -> None:
    source = "Hello world"
    stale = contract(source)
    stale["source_hash"] = "0" * 64
    resolved = resolve_expected_ui_contract(
        source,
        intent=None,
        persisted=stale,
        persisted_source="human benchmark",
    )
    assert resolved.source == "deterministic fallback"
    assert any("source_hash" in error for error in resolved.errors)


def test_exact_score_reuse_and_config_staleness() -> None:
    config = load_default_reward_config()
    source = "Hello world"
    candidate = simple_spec(source)
    expected = contract(source)
    result = score_genui_completion(
        candidate,
        source,
        expected_ui_contract=expected,
        config=config,
    )
    record = {
        "ui_id": "reuse",
        "response_text": source,
        "genui_raw_completion": compact(candidate),
        "genui_json": candidate,
        "expected_ui_contract": expected,
        "expected_ui_contract_source": "persisted",
        "render_artifact_quality_v5_1": breakdown_to_mapping(result),
    }
    reused = score_record(record, config=config)
    assert reused.evidence["score_reuse"]["reused"]

    changed = replace(config, value_beta=config.value_beta + 0.25)
    recomputed = score_record(record, config=changed)
    assert not recomputed.evidence["score_reuse"]["reused"]
    assert "metric_fingerprint_mismatch" in recomputed.evidence["score_reuse"]["stale_reasons"]

    changed_record = copy.deepcopy(record)
    changed_record["genui_json"] = simple_spec("Different")
    recomputed_payload = score_record(changed_record, config=config)
    assert not recomputed_payload.evidence["score_reuse"]["reused"]
    assert "raw_candidate_hash_mismatch" in recomputed_payload.evidence["score_reuse"]["stale_reasons"]


def test_alias_and_canonical_paths_share_semantic_fidelity_with_format_penalty() -> None:
    source = "Hello world"
    canonical = simple_spec(source)
    alias = copy.deepcopy(canonical)
    alias["elements"]["root"]["type"] = "column"
    alias["elements"]["body"]["type"] = "text"
    strict = score(canonical, source)
    canonicalizable = score(alias, source)
    assert canonicalizable.normalization["production_valid"]
    assert not canonicalizable.normalization["strict_schema_valid"]
    assert (
        canonicalizable.atomics["fidelity"]["content_unit_fidelity"]
        == strict.atomics["fidelity"]["content_unit_fidelity"]
    )
    assert canonicalizable.quality_0_1 < strict.quality_0_1


def test_scores_are_process_order_independent_and_deterministic() -> None:
    source = "Hello world"
    candidate = simple_spec(source)
    reordered = copy.deepcopy(candidate)
    reordered["elements"] = dict(reversed(list(reordered["elements"].items())))
    first = score(candidate, source)
    second = score(reordered, source)
    third = score(candidate, source)
    assert first.quality_0_1 == second.quality_0_1 == third.quality_0_1
    assert first.reward == third.reward
