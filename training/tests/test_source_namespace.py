from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import prepare_review_training
from ir_training.data.audit_filter import (
    _identity_keys,
    _identity_keys_overlap,
    _row_identity_keys,
    audit_and_filter_rows,
    load_reserved_cohorts,
)
from ir_training.data.build_pairs import _assign_split_groups, prepare_dataset
from ir_training.data.chat_templates import build_messages
from ir_training.data.golden_replacement import source_identity_record
from ir_training.data.source_namespace import scoped_source_id
from ir_training.data.splits import stratified_split
from ir_training.eval.prepared_contract import verify_reserved_train_validation


def _prepared(source_path: str, text: str) -> dict:
    target = '<a2ui>\nroot=Text("Response")\n</a2ui>'
    return {"id": source_path + ":u1", "source_id": scoped_source_id(source_path, "q_000001"),
            "response_id": "r_000001_01", "response_text": text, "completion": target,
            "metadata": {"query_id": "q_000001", "source_path": source_path},
            "messages": build_messages("System", text, target, target_format="a2ui_express_v1")}


def test_preparation_namespaces_only_inferred_ids_and_keeps_raw_query(tmp_path):
    source_paths = []
    spec = {"root": "root", "state": {},
            "elements": {"root": {"type": "Text", "props": {"text": "Answer"}, "children": []}}}
    for name in ("run_a", "run_b"):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "genui.jsonl"
        rows = [{"query_id": "q_000001", "response_id": "r1", "response_text": f"Source from {name}",
                 "genui_json": spec},
                {"query_id": "q_000002", "response_id": "r2", "source_id": "global-explicit-source",
                 "scenario_family_id": "global-explicit-family", "response_text": f"Sibling from {name}",
                 "genui_json": spec}]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        source_paths.append(str(path))
    output = tmp_path / "prepared"
    prepare_dataset({"run": {"source_genui_files": source_paths, "output_dir": str(output)},
                     "split": {"train": 1, "val": 0, "test": 0}})
    rows = [json.loads(line) for line in (output / "all.jsonl").read_text(encoding="utf-8").splitlines()]
    inferred = [row for row in rows if row["metadata"]["query_id"] == "q_000001"]
    assert len({row["source_id"] for row in inferred}) == 2
    assert all(row["source_id"].startswith("run-local-v1:") for row in inferred)
    explicit = [row for row in rows if row["metadata"]["query_id"] == "q_000002"]
    assert {row["source_id"] for row in explicit} == {"global-explicit-source"}
    assert {row["scenario_family_id"] for row in explicit} == {"global-explicit-family"}


def test_review_identity_and_reserved_checks_distinguish_run_local_ids(tmp_path):
    first = _prepared("run_a/genui.jsonl", "First source")
    second = _prepared("run_b/genui.jsonl", "Unrelated second source")
    assert prepare_review_training.source_identity(first)[0] != prepare_review_training.source_identity(second)[0]
    assert not _identity_keys(source_identity_record(first)) & _identity_keys(source_identity_record(second))
    reserved_file = tmp_path / "golden.jsonl"
    reserved_file.write_text(json.dumps(first) + "\n", encoding="utf-8")
    accepted, _, report = audit_and_filter_rows([second], reserved=load_reserved_cohorts([reserved_file]))
    assert accepted == [second]
    assert report["quarantined_rows"] == 0
    duplicate = _prepared("run_b/genui.jsonl", " First   source ")
    accepted, _, report = audit_and_filter_rows([duplicate], reserved=load_reserved_cohorts([reserved_file]))
    assert not accepted
    assert report["quarantine_reasons"] == {"reserved_evaluation_source": 1}


def test_same_run_query_siblings_and_global_ids_remain_reserved(tmp_path):
    first = _prepared("run_a/genui.jsonl", "First source")
    sibling = _prepared("run_a/genui.jsonl", "Another response for the same query")
    assert _identity_keys(source_identity_record(first)) & _identity_keys(source_identity_record(sibling))
    global_record = {"source_id": "global-source", "query_id": "q_017270", "response_id": "r1", "row_id": "g1"}
    assert set(global_record.values()) <= _identity_keys(global_record)
    old_row = {"response_text": "actual response", "source_id": "source-group", "metadata": {"query_id": "q_017270"}}
    assert prepare_review_training.source_identity(old_row)[0] == "q_017270"


def test_legacy_reserved_origin_rejects_changed_sibling_not_unrelated_query(tmp_path):
    reserved = _prepared("run_a/genui.jsonl", "Original held-out response")
    reserved["source_id"] = "q_000001"
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps(reserved) + "\n", encoding="utf-8")
    same_origin = _prepared("run_a/genui.jsonl", "Changed held-out response")
    unrelated = _prepared("run_b/genui.jsonl", "Independent generated response")
    accepted, _, report = audit_and_filter_rows(
        [same_origin, unrelated], reserved=load_reserved_cohorts([golden]))
    assert accepted == [unrelated]
    assert report["quarantine_reasons"] == {"reserved_evaluation_source": 1}
    assert _row_identity_keys(reserved) & _row_identity_keys(same_origin)
    assert not _row_identity_keys(reserved) & _row_identity_keys(unrelated)


def test_legacy_excluded_row_id_recovers_origin_without_changing_manifest():
    origin = "C:/data/run_a/genui.jsonl"
    record = {"source_id": "q_000001", "query_id": "q_000001", "response_id": "r_000001_01",
              "row_id": origin + ":u_000001_01:a2ui_express_v1"}
    original = dict(record)
    assert _identity_keys(record) & _row_identity_keys(_prepared(origin, "Changed response"))
    assert not _identity_keys(record) & _row_identity_keys(_prepared("run_b/genui.jsonl", "Different"))
    assert record == original


def test_originless_legacy_reserved_ids_remain_conservatively_blocked(tmp_path):
    old = _prepared("run_a/genui.jsonl", "Legacy held-out response")
    old["source_id"] = "q_000001"
    old["id"] = "old-row"
    old["metadata"].pop("source_path")
    new = _prepared("run_b/genui.jsonl", "Possibly changed response")
    assert _identity_keys_overlap(_row_identity_keys(old), _row_identity_keys(new))
    assert _identity_keys_overlap(_row_identity_keys(new), _row_identity_keys(old))
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps(old) + "\n", encoding="utf-8")
    accepted, _, report = audit_and_filter_rows([new], reserved=load_reserved_cohorts([golden]))
    assert not accepted
    assert report["quarantine_reasons"] == {"reserved_evaluation_source": 1}
    (tmp_path / "train.jsonl").write_text(json.dumps(new) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Reserved Golden source overlaps train"):
        verify_reserved_train_validation(tmp_path, [golden])


def test_normalized_duplicates_link_explicit_families_transitively_before_split():
    rows = [
        {"id": "a", "scenario_family_id": "family-a", "response_text": "Same response", "intent_bucket": "a"},
        {"id": "b", "scenario_family_id": "family-b", "response_text": " Same   response ", "intent_bucket": "b"},
        {"id": "c", "scenario_family_id": "family-b", "response_text": "Augmented sibling", "intent_bucket": "b"},
        *[{"id": f"extra-{i}", "scenario_family_id": f"family-{i}", "response_text": f"Unrelated {i}"} for i in range(20)],
    ]
    original_families = [row["scenario_family_id"] for row in rows]
    _assign_split_groups(rows)
    assert rows[0]["split_group_id"] == rows[1]["split_group_id"] == rows[2]["split_group_id"]
    assert [row["scenario_family_id"] for row in rows] == original_families
    for seed in range(10):
        splits = stratified_split(rows, .5, .25, .25, "intent_bucket", seed, group_key="split_group_id")
        places = {row["id"]: split for split, values in splits.items() for row in values}
        assert places["a"] == places["b"] == places["c"]
