from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.common.jsonl import write_jsonl
from ir_training.data.express_preparation import prepare_splits
from ir_training.eval.compare_to_baseline import repeated_benchmark_scores
from ir_training.eval.generate import build_prediction_record
from ir_training.eval.golden_set import benchmark_contract_for_split, load_fixed_golden_rows

ARTIFACT = ROOT / "data/eval/golden32_archive_repeat_v1/golden32.jsonl"


def test_checked_in_repeat_is_hash_bound_and_strict_valid():
    rows = load_fixed_golden_rows(ARTIFACT, required_rows=32)
    contract = benchmark_contract_for_split(ARTIFACT, rows)
    assert len(rows) == 32 and contract["unique_source_count"] == 31
    assert contract["output_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert len({row["source_id"] for row in rows}) == 31


def test_repeat_survives_second_preparation_with_embedded_manifest(tmp_path):
    first = tmp_path / "prepared"
    prepare_splits({"golden32": ARTIFACT}, first)
    result = prepare_splits({"golden32": first / "golden32.jsonl"}, tmp_path / "again", ordering="bottom-up")
    assert result["splits"]["golden32"]["benchmark"]["unique_source_count"] == 31
    assert result["splits"]["golden32"]["accepted_rows"] == 32
    assert len(load_fixed_golden_rows(tmp_path / "again/golden32.jsonl", required_rows=32)) == 32


def test_repeated_rows_cannot_bypass_loader_by_dropping_manifest(tmp_path):
    rows = load_fixed_golden_rows(ARTIFACT, required_rows=32)
    write_jsonl(tmp_path / "golden32.jsonl", rows)
    with pytest.raises(ValueError, match="manifest"):
        load_fixed_golden_rows(tmp_path / "golden32.jsonl", required_rows=32)


def test_unique_source_scoring_does_not_double_weight_donor():
    rows = [build_prediction_record(row, "observed fixture") for row in load_fixed_golden_rows(ARTIFACT, required_rows=32)]
    donor_id = next(row["source_id"] for row in rows if row.get("repeated_from"))
    for row in rows:
        row["metrics"] = {"generation_reward_v5_4": 100.0 if row["source_id"] == donor_id else 0.0}
    result = repeated_benchmark_scores(rows, {})
    assert result["unique_source_generation_reward_v5_4_avg"] == pytest.approx(100 / 31)
    assert result["benchmark"]["row_count"] == 32
    assert not result["benchmark"]["independent_test_set"]
    next(row for row in rows if row.get("repeated_from"))["metrics"]["generation_reward_v5_4"] = 0
    assert repeated_benchmark_scores(rows, {})["unique_source_generation_reward_v5_4_avg"] == pytest.approx(50 / 31)


def test_unique_source_scoring_rejects_partial_or_mixed_predictions():
    rows = [build_prediction_record(row, "fixture") for row in load_fixed_golden_rows(ARTIFACT, required_rows=32)]
    for row in rows:
        row["metrics"] = {"generation_reward_v5_4": 1.0}
    with pytest.raises(ValueError, match="32 occurrences"):
        repeated_benchmark_scores(rows[:-1], {})
    changed = deepcopy(rows)
    changed[0].pop("benchmark")
    with pytest.raises(ValueError, match="mix"):
        repeated_benchmark_scores(changed, {})


def test_jsonl_writer_has_portable_lf_bytes(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [{"a": 1}, {"a": 2}])
    assert path.read_bytes() == b'{"a":1}\n{"a":2}\n'
