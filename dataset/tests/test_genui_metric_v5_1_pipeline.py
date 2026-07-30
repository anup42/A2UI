from __future__ import annotations

from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_1_support import compact, contract, simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    aggregate_v5_1_records,
    breakdown_to_mapping,
    load_v5_1_reward_config,
    prepare_source_context_v5_1,
    score_completion_group_v5_1,
    score_genui_completion_v5_1,
    score_genui_completion_v5_0,
    score_record_v5_1,
    score_record_generation_v5_1,
)
from pipeline.genui_quality import identity_v5_1  # noqa: E402
from pipeline.genui_quality.identity_v5_1 import (  # noqa: E402
    metric_fingerprint_v5_1,
)


def test_prepared_group_and_scalar_scores_are_equivalent() -> None:
    source = "Hello world"
    expected = contract(source)
    candidates = [simple_spec(source), simple_spec("Wrong text")]
    config = load_v5_1_reward_config()
    prepared = prepare_source_context_v5_1(
        source, expected_ui_contract=expected, config=config
    )
    group = score_completion_group_v5_1(candidates, prepared)
    scalar = [
        score_genui_completion_v5_1(
            candidate,
            source,
            expected_ui_contract=expected,
            config=config,
        )
        for candidate in candidates
    ]
    assert [item.quality_0_1 for item in group] == [
        item.quality_0_1 for item in scalar
    ]
    assert [item.reward for item in group] == [item.reward for item in scalar]


def test_grpo_uses_raw_generation_reward() -> None:
    source = "Hello world"
    candidate = simple_spec(source)
    scalar = score_genui_completion_v5_1(candidate, source)
    prepared = prepare_source_context_v5_1(source)
    rewards = [
        score_completion_group_v5_1([candidate], prepared)[0].reward
    ]
    assert rewards == [scalar.reward]


def test_raw_and_final_scores_hashes_and_aggregate_headline_are_separate() -> None:
    source = "Hello world"
    expected = contract(source)
    raw = simple_spec("Wrong text")
    final = simple_spec(source)
    record = {
        "ui_id": "dual",
        "response_text": source,
        "genui_raw_completion": compact(raw),
        "genui_json": final,
        "expected_ui_contract": expected,
    }
    generation = score_record_generation_v5_1(record)
    artifact = score_record_v5_1(record)
    record["generation_reward_v5_1"] = breakdown_to_mapping(generation)
    record["render_artifact_quality_v5_1"] = breakdown_to_mapping(artifact)
    aggregate = aggregate_v5_1_records([record])
    assert generation.identity["raw_candidate_hash"] != artifact.identity[
        "raw_candidate_hash"
    ]
    assert aggregate["headline"] == "render_artifact_quality_v5_1"
    assert aggregate["quality_0_100"]["mean"] == artifact.quality_0_100
    assert (
        aggregate["generation_reward_quality_0_100"]["mean"]
        == generation.quality_0_100
    )


def test_v5_0_is_never_reused_as_v5_1() -> None:
    record = {
        "ui_id": "historical",
        "response_text": "Hello world",
        "genui_json": simple_spec(),
        "genui_quality_v5": {
            "metric_version": "5.0.0",
            "metric_fingerprint": "historical",
            "quality_0_1": 1.0,
        },
    }
    result = score_record_v5_1(record)
    assert result.metric_version == "5.1.0"
    assert not result.evidence["score_reuse"]["reused"]


def test_historical_v5_0_reader_remains_explicitly_available() -> None:
    result = score_genui_completion_v5_0(
        simple_spec("Hello world"), "Hello world"
    )
    assert result.metric_version == "5.0.0"


def test_policy_versions_are_part_of_fingerprint(monkeypatch) -> None:
    config = load_v5_1_reward_config()
    before = metric_fingerprint_v5_1(config)
    monkeypatch.setattr(
        identity_v5_1, "MATCHING_POLICY_VERSION", "5.1.0-test"
    )
    after = identity_v5_1.metric_fingerprint_v5_1(config)
    assert after != before


def test_stage3_persists_dual_scores_and_has_no_count_quality_warnings() -> None:
    source = (DATASET_ROOT / "src" / "pipeline" / "stage3_genui.py").read_text(
        encoding="utf-8"
    )
    assert "generation_reward_v5_1" in source
    assert "render_artifact_quality_v5_1" in source
    assert "low_component_count" not in source
    assert "sparse_ir" not in source
