from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.data.audit_filter import audit_and_filter_rows, load_reserved_cohorts
from ir_training.data.chat_templates import build_messages
from ir_training.data.golden_replacement import source_identity_record
from ir_training.data.url_preprocess import preprocess_training_urls


def _row(index, text=None, completion=None):
    text = text or f"Source {index}"
    completion = completion or f'<a2ui>\nroot=Text("Value {index}")\n</a2ui>'
    return {"id": f"row{index}", "source_id": f"q{index}", "response_text": text,
            "intent_bucket": "test", "messages": build_messages("System", text, completion, target_format="a2ui_express_v1"),
            "completion": completion}


def test_filter_reserves_both_cohorts_and_replaced_original(tmp_path):
    golden32, golden35 = tmp_path / "golden32.jsonl", tmp_path / "golden35.jsonl"
    golden32.write_text(json.dumps(_row(1)) + "\n", encoding="utf-8")
    golden35.write_text(json.dumps(_row(2)) + "\n", encoding="utf-8")
    manifest = tmp_path / "excluded.json"
    manifest.write_text(json.dumps({"excluded_sources": [source_identity_record(_row(3))]}), encoding="utf-8")
    reserved = load_reserved_cohorts([golden32, golden35], [manifest])
    rows = [_row(1), _row(9, "Source 2"), _row(3), _row(4)]
    original = deepcopy(rows)
    accepted, quarantine, report = audit_and_filter_rows(rows, reserved=reserved)
    assert accepted == [rows[-1]] and rows == original
    assert report["quarantine_reasons"] == {"reserved_evaluation_source": 3}
    assert len(quarantine) == 3 and report["synthetic_targets_created"] == 0


def test_wire_invalid_and_semantic_duplicate_are_quarantined():
    valid = _row(1)
    duplicate = deepcopy(valid)
    duplicate["id"] = "different-occurrence"
    invalid = _row(2, completion='<a2ui>\nroot=Table(columns=["A"],rows=[[1]],highlightColumns="A")\n</a2ui>')
    accepted, _, report = audit_and_filter_rows([valid, duplicate, invalid])
    assert accepted == [valid]
    assert report["quarantine_reasons"] == {"duplicate_source_target": 1, "wire_schema_invalid": 1}
    assert report["accepted_component_counts"] == {"Text": 1}
    assert report["augmentation_candidates"][0]["original_rows"] == 3


def test_alternative_valid_targets_are_reported_without_synthesis():
    first, second = _row(1), _row(2, "Source 1")
    accepted, _, report = audit_and_filter_rows([first, second])
    assert accepted == [first, second]
    assert report["same_source_multiple_target_count"] == 1
    assert report["retained_source_rows_unchanged"] is True


def test_missing_source_identity_and_token_limit_are_explicit():
    row = _row(1)
    row.pop("source_id")
    accepted, _, report = audit_and_filter_rows([row], require_source_identities=True)
    assert not accepted and report["quarantine_reasons"] == {"missing_source_identity": 1}
    with pytest.raises(ValueError, match="explicit tokenizer"):
        audit_and_filter_rows([_row(1)], max_seq_length=4096)


def test_raw_golden_urls_match_masked_training_sources(tmp_path):
    raw = "View the report at https://example.org/report"
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(_row(1, raw)) + "\n", encoding="utf-8")
    reserved = load_reserved_cohorts([path])
    masked = preprocess_training_urls(raw, {}, enabled=True).response_text
    assert masked != raw
    accepted, _, report = audit_and_filter_rows([_row(900, masked)], reserved=reserved)
    assert not accepted and report["quarantine_reasons"] == {"reserved_evaluation_source": 1}


def test_parallel_filter_and_streaming_sinks_match_serial_decisions():
    rows = [_row(index) for index in range(40)]
    rows += [deepcopy(rows[0]), _row(99, completion='<a2ui>\nroot=Table(columns=["A"],rows=[[1]],highlightColumns="A")\n</a2ui>')]
    reserved = {"identities": {"q3"}, "responses": set(), "evidence": []}
    expected = audit_and_filter_rows(rows, reserved=reserved, require_source_identities=True)
    accepted, quarantined = [], []
    actual = audit_and_filter_rows(iter(rows), reserved=reserved, require_source_identities=True, workers=2,
                                  accepted_sink=accepted.append, quarantine_sink=quarantined.append)
    assert actual[:2] == ([], [])
    assert (accepted, quarantined, actual[2]) == expected
    assert actual[2]["quarantine_reasons"] == {"reserved_evaluation_source": 1, "duplicate_source_target": 1, "wire_schema_invalid": 1}
