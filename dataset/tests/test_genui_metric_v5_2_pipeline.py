from __future__ import annotations

from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_2_support import (  # noqa: E402
    compact,
    contract,
    simple_spec,
)
from pipeline.genui_quality import (  # noqa: E402
    aggregate_v5_records,
    breakdown_to_mapping,
    genui_grpo_reward_v5_2 as genui_grpo_reward,
    prepare_source_context_v5_2,
    score_completion_group_v5_2,
    score_genui_completion_v5_2 as score_genui_completion,
    score_genui_completion_v5_1,
    score_record,
    score_record_generation_v5_2,
)


def test_prepared_group_scalar_stage3_and_grpo_paths_agree() -> None:
    source = "Hello world"
    expected = contract(source)
    candidates = [simple_spec(source), simple_spec("Wrong content")]
    prepared = prepare_source_context_v5_2(
        source, expected_ui_contract=expected
    )
    group = score_completion_group_v5_2(candidates, prepared)
    scalar = [
        score_genui_completion(
            candidate,
            source,
            expected_ui_contract=expected,
        )
        for candidate in candidates
    ]
    grpo = genui_grpo_reward(
        candidates,
        source,
        expected_ui_contract=expected,
    )
    assert [result.quality_0_1 for result in group] == [
        result.quality_0_1 for result in scalar
    ]
    assert [result.reward for result in group] == grpo


def test_raw_generation_and_final_artifact_are_separate_v5_2_surfaces() -> None:
    source = "Hello world"
    expected = contract(source)
    raw = simple_spec("Wrong content")
    final = simple_spec(source)
    record = {
        "ui_id": "dual",
        "response_text": source,
        "genui_raw_completion": compact(raw),
        "genui_json": final,
        "expected_ui_contract": expected,
    }
    generation = score_record_generation_v5_2(record)
    artifact = score_record(record)
    assert generation.identity["raw_candidate_hash"] != artifact.identity[
        "raw_candidate_hash"
    ]
    record["generation_reward_v5_2"] = breakdown_to_mapping(generation)
    record["render_artifact_quality_v5_2"] = breakdown_to_mapping(artifact)
    aggregate = aggregate_v5_records([record])
    assert aggregate["headline"] == "render_artifact_quality_v5_2"
    assert aggregate["quality_0_100"]["mean"] == artifact.quality_0_100
    assert (
        aggregate["generation_reward_quality_0_100"]["mean"]
        == generation.quality_0_100
    )


def test_v5_1_remains_explicit_and_is_never_reused_as_v5_2() -> None:
    historical = score_genui_completion_v5_1(
        simple_spec(), "Hello world"
    )
    assert historical.metric_version == "5.1.0"
    record = {
        "response_text": "Hello world",
        "genui_json": simple_spec(),
        "genui_quality_v5_1": breakdown_to_mapping(historical),
    }
    current = score_record(record)
    assert current.metric_version == "5.2.0"
    assert not current.evidence["score_reuse"]["reused"]


def test_stage3_and_training_dispatch_are_v5_2_aware() -> None:
    stage3 = (
        DATASET_ROOT / "src" / "pipeline" / "stage3_genui.py"
    ).read_text(encoding="utf-8")
    training = (
        DATASET_ROOT.parent / "training" / "scripts" / "train_grpo.py"
    ).read_text(encoding="utf-8")
    assert "generation_reward_v5_2" in stage3
    assert "render_artifact_quality_v5_2" in stage3
    assert "metric_identity_v5_2" in stage3
    assert "genui_metric_v5_4.yaml" in training
    assert "make_genui_grpo_reward_v5_4" in training
    assert "ensure_v5_4_validation_ready" in training
