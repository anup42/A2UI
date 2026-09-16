"""CPU protocol/scoring checks; synthetic outputs are not model-quality results."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.bixby50 import APPROVED_SOURCE_IDS, validate_source_only_predictions
from ir_training.eval import compare_to_baseline as scoring
from ir_training.eval import parallel_generate as parallel
from ir_training.eval.generate import build_prediction_record
from ir_training.eval.metrics import aggregate_scores, score_prediction


@pytest.fixture
def source_rows():
    return list(read_jsonl(ROOT / "data/eval/bixby50_v1/bixby50.jsonl"))


def _prediction(row):
    # This deliberately simple valid output tests the pipeline, not semantic
    # fidelity or a trained checkpoint. Never save it as a reference answer.
    text = '<a2ui>\nroot=Text("Synthetic evaluation fixture")\n</a2ui>'
    return build_prediction_record(row, text, runtime={"output_tokens": 12, "inference_device": "cuda:0"})


@pytest.mark.parametrize("metric_version", ["legacy", "v5_4"])
def test_full_bixby50_scoring_never_reports_reference_match_as_zero(tmp_path, source_rows, metric_version):
    predictions = [_prediction(row) for row in source_rows]
    validate_source_only_predictions(predictions)
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, predictions)
    output = tmp_path / "scores"
    result = scoring.evaluate_predictions(path, output_dir=output, metric_version=metric_version)
    assert result["count"] == result["unique_source_count"] == 50
    assert result["benchmark_id"] == "bixby50_v1"
    assert result["reference_available"] is False
    assert result["reference_match_metrics_available"] is False
    assert result["evaluation_only"] is True
    assert not any(key.startswith(("exact_match", "semantic_match")) for key in result)
    scored = list(read_jsonl(output / "scored_predictions.jsonl"))
    assert [row["source_id"] for row in scored] == list(APPROVED_SOURCE_IDS)
    assert all(row["expected"] is None and row["reference_available"] is False for row in scored)
    for row in scored:
        for key in ("metrics", "raw_metrics", "serving_stopped_metrics"):
            assert "exact_match" not in row[key] and "semantic_match" not in row[key]


def test_refusal_is_evaluated_as_captured_source_not_as_original_question(source_rows):
    row = next(item for item in source_rows if item["id"] == "BXP-038")
    assert "cannot provide a source-cited comparison" in row["response_text"]
    prediction = '<a2ui>\nroot=Text(' + json.dumps(row["response_text"], ensure_ascii=False) + ')\n</a2ui>'
    record = build_prediction_record(row, prediction)
    metrics = score_prediction(record["response_text"], record["expected"], prediction, metric_version="v5_4")
    assert record["expected"] is None
    assert record["response_text"] == row["response_text"]
    assert record["response_text"] != row["metadata"]["original_query"]
    assert metrics["schema_valid_strict"] is True
    assert metrics["scoring_errors_v5_4"] == []
    assert isinstance(metrics["generation_reward_v5_4"], float)
    assert "exact_match" not in metrics and "semantic_match" not in metrics
    aggregate = aggregate_scores([{"metrics": metrics}])
    assert not any(key.startswith(("exact_match", "semantic_match")) for key in aggregate)


@pytest.mark.parametrize("corruption", [
    "partial", "duplicate", "wrong_source_id", "altered_source", "invented_reference",
    "completion_alias", "expected_json_alias", "inferred_contract", "metadata_contract",
    "missing_reference_flag", "missing_evaluation_flag", "missing_selection_role",
    "missing_benchmark", "changed_benchmark",
])
def test_invalid_cohort_fails_before_any_scoring_or_publication(tmp_path, monkeypatch, source_rows, corruption):
    predictions = [_prediction(row) for row in source_rows]
    if corruption == "partial":
        predictions.pop()
    elif corruption == "duplicate":
        predictions[-1] = deepcopy(predictions[0])
    elif corruption == "wrong_source_id":
        predictions[0]["source_id"] = "BXP-999"
    elif corruption == "altered_source":
        predictions[0]["response_text"] += " changed"
    elif corruption == "invented_reference":
        predictions[0]["expected"] = '<a2ui>root=Text("Reference")</a2ui>'
    elif corruption == "completion_alias":
        predictions[0]["completion"] = '<a2ui>root=Text("Reference")</a2ui>'
    elif corruption == "expected_json_alias":
        predictions[0]["expected_json"] = {"root": "invented"}
    elif corruption == "inferred_contract":
        predictions[0]["expected_ui_contract_v5_4"] = {"invented": True}
    elif corruption == "metadata_contract":
        predictions[0]["metadata"] = {"expected_ui_contract": {"invented": True}}
    elif corruption == "missing_reference_flag":
        del predictions[0]["reference_available"]
    elif corruption == "missing_evaluation_flag":
        del predictions[0]["evaluation_only"]
    elif corruption == "missing_selection_role":
        del predictions[0]["selection_role"]
    elif corruption == "missing_benchmark":
        del predictions[0]["benchmark"]
    elif corruption == "changed_benchmark":
        predictions[0]["benchmark"]["unique_source_count"] = 49
    with pytest.raises(ValueError):
        validate_source_only_predictions(predictions)
    monkeypatch.setattr(scoring, "score_prediction", lambda *a, **k: pytest.fail("Invalid source-only cohort reached scoring"))
    path = tmp_path / "predictions.jsonl"
    output = tmp_path / "scores"
    write_jsonl(path, predictions)
    with pytest.raises(ValueError):
        scoring.evaluate_predictions(path, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_fifty_source_only_rows_merge_across_gpu_shards_without_none_hash_failure(tmp_path, source_rows, workers):
    shards = parallel.shard_indices(len(source_rows), workers)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    paths = []
    for rank, indices in enumerate(shards):
        path = tmp_path / f"rank_{rank}.jsonl"
        write_jsonl(path, (_prediction(source_rows[index]) for index in indices))
        paths.append(path)
    output = tmp_path / "predictions.jsonl"
    assert parallel.merge_shards(source_rows, shards, paths, output) == 50
    result = list(read_jsonl(output))
    validate_source_only_predictions(result)
    assert [row["source_id"] for row in result] == list(APPROVED_SOURCE_IDS)
    assert all(row["expected"] is None for row in result)


@pytest.mark.parametrize("field", ["reference_available", "evaluation_only", "selection_role", "expected"])
def test_shard_merge_binds_reference_and_holdout_flags(tmp_path, source_rows, field):
    shards = parallel.shard_indices(50, 8)
    paths = []
    for rank, indices in enumerate(shards):
        predictions = [_prediction(source_rows[index]) for index in indices]
        if rank == 7:
            if field == "expected":
                predictions[0][field] = "forged reference"
            else:
                del predictions[0][field]
        path = tmp_path / f"rank_{rank}.jsonl"
        write_jsonl(path, predictions)
        paths.append(path)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(ValueError, match="source identity mismatch"):
        parallel.merge_shards(source_rows, shards, paths, output)
    assert not output.exists()


class _FinishedProcess:
    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_bixby50_parallel_launch_isolates_all_assigned_gpu_workers(tmp_path, monkeypatch, source_rows, workers):
    # Simulated child processes exercise allocation, job payloads, merging and
    # reporting on CPU; this does not claim CUDA inference was executed.
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setattr(parallel, "resolve_generation_devices", lambda *a, **k: [f"GPU-fixture-{rank}" for rank in range(workers)])
    launched = []

    def launch(command, env):
        job = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        rows = list(read_jsonl(job["split_path"]))
        assert job["gpu"] is True
        assert job["max_new_tokens"] == 2048
        assert job["max_rows"] == len(rows)
        launched.append((dict(env), len(rows)))
        write_jsonl(job["output_path"], (_prediction(row) for row in rows))
        return _FinishedProcess()

    monkeypatch.setattr(parallel.subprocess, "Popen", launch)
    split = tmp_path / "source.jsonl"
    write_jsonl(split, source_rows)
    output = tmp_path / "predictions.jsonl"
    performance = {}
    assert parallel.generate_predictions_parallel(
        {}, split, output, max_rows=50, max_new_tokens=2048, require_gpu=True,
        performance_metrics=performance,
    ) == 50
    assert [env["CUDA_VISIBLE_DEVICES"] for env, _ in launched] == [f"GPU-fixture-{rank}" for rank in range(workers)]
    assert all("RANK" not in env and "WORLD_SIZE" not in env for env, _ in launched)
    assert max(count for _, count in launched) - min(count for _, count in launched) <= 1
    assert performance["generation_runtime_gpu_count"] == workers
    assert performance["generation_runtime_case_count"] == 50
    validate_source_only_predictions(list(read_jsonl(output)))
