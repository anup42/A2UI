from __future__ import annotations

import copy
from pathlib import Path
import statistics
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_support import compact, contract, score, simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    aggregate_v5_1_records as aggregate_v5_records,
    breakdown_to_mapping,
    load_v5_1_reward_config as load_default_reward_config,
    prepare_source_context_v5_1,
    score_completion_group_v5_1,
    score_genui_completion_v5_1 as score_genui_completion,
    score_record_v5_1 as score_record,
)


def genui_grpo_reward(completions, response_text):
    source = (
        response_text[0]
        if isinstance(response_text, list)
        else response_text
    )
    prepared = prepare_source_context_v5_1(source)
    return [
        result.reward
        for result in score_completion_group_v5_1(
            completions, prepared
        )
    ]


def test_grpo_and_offline_paths_are_identical_for_same_raw_candidate() -> None:
    source = "Hello world"
    raw = compact(simple_spec(source))
    offline = score_genui_completion(raw, source)
    reward = genui_grpo_reward([raw], [source])[0]
    assert reward == offline.reward


def test_canonicalizable_raw_alias_has_same_semantic_score_across_paths() -> None:
    source = "Hello world"
    candidate = simple_spec(source)
    candidate["elements"]["root"]["type"] = "column"
    raw = compact(candidate)
    offline = score_genui_completion(raw, source)
    reward = genui_grpo_reward([raw], source)[0]
    stage3_like = score_record(
        {
            "ui_id": "stage3-like",
            "response_text": source,
            "genui_raw_completion": raw,
            "genui_json": candidate,
        }
    )
    assert reward == offline.reward
    assert stage3_like.reward == offline.reward
    assert offline.normalization["production_valid"]
    assert not offline.normalization["strict_schema_valid"]


def test_aggregate_is_distribution_of_per_sample_scores() -> None:
    records = [
        {
            "ui_id": "a",
            "response_text": "Alpha",
            "genui_raw_completion": compact(simple_spec("Alpha")),
            "genui_json": simple_spec("Alpha"),
        },
        {
            "ui_id": "b",
            "response_text": "Beta",
            "genui_raw_completion": compact(simple_spec("Wrong")),
            "genui_json": simple_spec("Wrong"),
        },
    ]
    individual = [score_record(record).quality_0_100 for record in records]
    aggregate = aggregate_v5_records(records)
    assert aggregate["quality_0_100"]["mean"] == statistics.fmean(individual)


def test_v4_breakdown_is_never_reused_as_v5() -> None:
    source = "Hello world"
    candidate = simple_spec(source)
    record = {
        "ui_id": "legacy",
        "response_text": source,
        "genui_json": candidate,
        "genui_quality_v5": {
            "metric_version": "4.0.0",
            "reward": 1.0,
            "quality_0_100": 100.0,
            "quality_0_1": 1.0,
            "cap_0_1": 1.0,
            "parse_stage": "mapping",
            "dimensions": {},
            "atomics": {},
            "evidence": {},
        },
    }
    result = score_record(record)
    assert result.metric_version == "5.1.0"
    assert not result.evidence["score_reuse"]["reused"]
    assert "missing_or_invalid_stored_v5_1" in result.evidence[
        "score_reuse"
    ]["stale_reasons"]


def test_changed_expected_contract_invalidates_stored_score() -> None:
    source = "Hello world"
    candidate = simple_spec(source)
    expected = contract(source)
    result = score_genui_completion(
        candidate,
        source,
        expected_ui_contract=expected,
    )
    record = {
        "ui_id": "contract-change",
        "response_text": source,
        "genui_raw_completion": compact(candidate),
        "genui_json": candidate,
        "expected_ui_contract": expected,
        "render_artifact_quality_v5_1": breakdown_to_mapping(result),
    }
    assert score_record(record).evidence["score_reuse"]["reused"]
    changed = copy.deepcopy(record)
    changed_contract = contract(source)
    changed_contract["content_units"] = ["Different expected representation"]
    changed["expected_ui_contract"] = changed_contract
    rescored = score_record(changed)
    assert not rescored.evidence["score_reuse"]["reused"]
    assert "expected_contract_hash_mismatch" in rescored.evidence["score_reuse"]["stale_reasons"]


def test_failed_native_render_activates_v5_cap() -> None:
    source = "Hello world"
    candidate = simple_spec(source)
    record = {
        "ui_id": "render",
        "response_text": source,
        "genui_json": candidate,
    }
    native_failure = {
        "ui_id": "render",
        "renderer_check_result": {
            "adapter": "android_native_flat_renderer",
            "attempted": True,
            "ok": False,
        },
    }
    result = score_record(record, render_row=native_failure)
    assert result.evidence["render_ok"] is False
    assert result.cap_0_1 <= 0.30
    assert any(cap["name"] == "render_failure" for cap in result.active_caps)


def test_configured_v5_fingerprint_is_present_in_aggregate() -> None:
    config = load_default_reward_config()
    aggregate = aggregate_v5_records(
        [
            {
                "ui_id": "one",
                "response_text": "Hello",
                "genui_json": simple_spec("Hello"),
            }
        ],
        config=config,
    )
    assert aggregate["metric_version"] == "5.1.0"
    assert len(aggregate["metric_fingerprint"]) == 64
