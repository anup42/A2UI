"""Regression checks for report aggregation, not model training tests."""
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace


def load_audit(name):
    path = Path(__file__).resolve().parents[1] / "scripts/audits" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_ir_summary_counters_and_occurrences(tmp_path):
    audit = load_audit("full_data_ir_20260913")
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
      CREATE TABLE targets(sha TEXT,valid INTEGER,reason TEXT,detail TEXT,features TEXT);
      CREATE TABLE rows(split TEXT,line INTEGER,target_sha TEXT,binding_error TEXT);
    """)
    features = {"components": {"Text": 2}, "properties": {"Text.text": 2},
                "references": {}, "text_length_histogram": {"10": 2},
                "canonical_equals_raw": True, "empty_layout_leaves": 1}
    connection.execute("INSERT INTO targets VALUES(?,?,?,?,?)", ("valid", 1, "valid", "", json.dumps(features)))
    connection.execute("INSERT INTO targets VALUES(?,?,?,?,?)", ("bad", 0, "syntax_invalid", "example", "{}"))
    connection.executemany("INSERT INTO rows VALUES(?,?,?,?)", [("train", 1, "valid", "[]"), ("train", 2, "valid", "[]"), ("val", 1, "bad", "[]")])
    audit.summarize(connection, SimpleNamespace(report=tmp_path, limit=None), [], {"engine": "test"}, tmp_path / "test.sqlite", 0)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["splits"]["train"]["target_validation_counts"] == {"valid": 2}
    assert result["splits"]["train"]["valid_occurrence_components"] == {"Text": 4}
    assert result["splits"]["train"]["valid_row_flags"]["has_empty_layout_leaves"] == 2
    assert result["splits"]["train"]["text_length_histogram"] == {"10": 4}
    assert result["splits"]["val"]["target_validation_counts"] == {"syntax_invalid": 1}
    assert "syntax_invalid" in (tmp_path / "val_invalid_rows.csv").read_text()
    connection.close()


def test_synthesis_distribution_uses_occurrence_weights():
    audit = load_audit("full_data_synthesis_20260913")
    result = audit.distribution({1: 9, 10: 1})
    assert result["count"] == 10
    assert result["mean"] == 1.9
    assert result["p50"] == 1
    assert result["p99"] == 10
    assert audit.distribution({}) == {"count": 0}
