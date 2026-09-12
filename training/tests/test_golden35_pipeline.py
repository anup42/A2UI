"""CPU-only integration of the frozen Golden35 preparation and score contract."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.common.config import load_yaml
from ir_training.common.jsonl import write_jsonl
from ir_training.data.audit_filter import load_reserved_cohorts
from ir_training.data.build_pairs import prepare_dataset
from ir_training.data.express_preparation import prepare_splits
from ir_training.data.golden35_subset import APPROVED_SOURCE_IDS, FROZEN_EXCLUSION_REASONS
from ir_training.eval.compare_to_baseline import evaluate_predictions, repeated_benchmark_scores
from ir_training.eval.generate import build_prediction_record
from ir_training.eval.golden_set import benchmark_contract_for_split, load_fixed_golden_rows

ARTIFACT = ROOT / "data/eval/golden35_v1/golden35.jsonl"
CONFIG = ROOT / "configs/datasets/golden35_stage3_eval.yaml"


@pytest.fixture
def prepared(tmp_path):
    config = load_yaml(CONFIG)
    directory = tmp_path / "golden35"
    config["run"]["output_dir"] = str(directory)
    manifest = prepare_dataset(config, config_path=CONFIG)
    return directory, manifest


def test_config_prepares_exact_frozen_subset_without_reselection(prepared):
    directory, manifest = prepared
    rows = load_fixed_golden_rows(directory / "all.jsonl", required_rows=35)
    assert {row["source_id"] for row in rows} == set(APPROVED_SOURCE_IDS)
    contract = benchmark_contract_for_split(directory / "all.jsonl")
    assert contract["benchmark_id"] == "golden35_v1"
    assert contract["output_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert manifest["splits"]["all"]["quarantined_rows"] == 0
    assert not (ROOT / "configs/datasets/golden50_stage3_eval.yaml").exists()


def test_config_pin_and_no_overwrite_are_enforced(prepared, tmp_path):
    directory, _ = prepared
    config = load_yaml(CONFIG)
    config["run"]["output_dir"] = str(directory)
    with pytest.raises(FileExistsError):
        prepare_dataset(config)
    config["run"]["output_dir"] = str(tmp_path / "new")
    config["run"]["frozen_eval_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="independent config pin"):
        prepare_dataset(config)


def test_subset_survives_another_preparation_but_not_manifest_removal(prepared, tmp_path):
    directory, _ = prepared
    again = tmp_path / "again"
    prepare_splits({"golden35": directory / "all.jsonl"}, again, ordering="bottom-up")
    rows = load_fixed_golden_rows(again / "golden35.jsonl", required_rows=35)
    with pytest.raises(ValueError, match="35"):
        load_fixed_golden_rows(again / "golden35.jsonl", max_rows=34, required_rows=35)
    unbound = tmp_path / "unbound.jsonl"
    write_jsonl(unbound, rows)
    with pytest.raises(ValueError, match="manifest"):
        load_fixed_golden_rows(unbound, required_rows=35)


def test_raw_and_prepared_reservations_include_all_15_excluded_sources(prepared):
    directory, _ = prepared
    raw = load_reserved_cohorts([ARTIFACT])
    cooked = load_reserved_cohorts([directory / "all.jsonl"])
    manifest = json.loads(ARTIFACT.with_name("benchmark_manifest.json").read_text(encoding="utf-8"))
    for reserved in (raw, cooked):
        assert set(APPROVED_SOURCE_IDS) | set(FROZEN_EXCLUSION_REASONS) <= reserved["identities"]
        for excluded in manifest["excluded_sources"]:
            assert set(excluded["response_sha256s"]) <= reserved["responses"]


def test_score_preserves_golden35_identity_and_rejects_partial_mixed_or_duplicate_runs(prepared, tmp_path, monkeypatch):
    directory, _ = prepared
    source_rows = load_fixed_golden_rows(directory / "all.jsonl", required_rows=35)
    predictions = [build_prediction_record(row, row["completion"]) for row in source_rows]
    # These are plumbing fixtures, not model inference or a quality benchmark.
    monkeypatch.setattr("ir_training.eval.compare_to_baseline.score_prediction", lambda *a, **k: {"generation_reward_v5_4": 1.0})
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, predictions)
    aggregate = evaluate_predictions(path, metric_version="v5_4")
    assert aggregate["count"] == 35
    assert aggregate["benchmark"]["unique_source_count"] == 35
    assert aggregate["benchmark"]["id"] == "golden35_v1"
    assert "unique_source_metrics" not in aggregate  # No repeated-case reweighting.
    with pytest.raises(ValueError, match="35-case"):
        repeated_benchmark_scores(predictions[:-1], {})
    mixed = deepcopy(predictions)
    mixed[0].pop("benchmark")
    with pytest.raises(ValueError, match="mix"):
        repeated_benchmark_scores(mixed, {})
    duplicate = deepcopy(predictions)
    duplicate[1] = deepcopy(duplicate[0])
    with pytest.raises(ValueError, match="membership"):
        repeated_benchmark_scores(duplicate, {})
