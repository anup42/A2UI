from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.data.chat_templates import build_messages
from ir_training.data.express_preparation import prepare_row
from ir_training.data.golden_replacement import build_replacement, read_rows_strict, serialize_rows, validate_replacement_rows


def _source():
    rows = []
    for index in range(32):
        completion = f'<a2ui>\nroot=Text("Case {index}")\n</a2ui>'
        if index == 25:
            completion = '<a2ui>\nroot=Table(columns=["A"],rows=[[1]],highlightColumns="A")\n</a2ui>'
        rows.append({"id": f"row-{index}", "source_id": f"q{index}", "response_id": f"r{index}",
                     "response_text": f"Source case {index}", "intent_bucket": "shared" if index in {4, 25} else "other",
                     "messages": build_messages("System", f"Source case {index}", completion, target_format="a2ui_express_v1"),
                     "completion": completion, "metadata": {"query_id": f"q{index}"}})
    return rows


def _build(rows=None, **kwargs):
    return build_replacement(rows or _source(), original_source_sha256="a" * 64,
                             failed_identity="q25", benchmark_id="test_repeat", **kwargs)


def test_replacement_preserves_source_and_donor_and_reports_true_counts():
    original = _source()
    untouched = deepcopy(original)
    rows, manifest = _build(original)
    assert original == untouched
    assert manifest["row_count"] == 32 and manifest["unique_source_count"] == 31
    assert manifest["donor_selection"] == "same_intent_first_valid"
    assert manifest["output_sha256"] == hashlib.sha256(serialize_rows(rows)).hexdigest()
    assert rows[25]["source_id"] == "q4"
    assert rows[25]["completion"] == original[4]["completion"]
    assert rows[25]["messages"] == original[4]["messages"]
    assert rows[25]["response_text"] == original[4]["response_text"]
    assert rows[25]["metadata"]["replaces"]["source_id"] == "q25"
    assert len({row["id"] for row in rows}) == 32
    assert all(row["metadata"]["benchmark"]["unique_source_count"] == 31 for row in rows)


def test_first_valid_fallback_and_explicit_donor():
    original = _source()
    original[25]["intent_bucket"] = "missing-intent"
    rows, manifest = _build(original)
    assert rows[25]["source_id"] == "q0"
    assert manifest["donor_selection"] == "first_strictly_valid"
    rows, _ = _build(donor_identity="r9")
    assert rows[25]["source_id"] == "q9"


def test_canonical_preparation_retains_valid_repeat_contract():
    rows, manifest = _build()
    prepared = [prepare_row(row, "bottom-up")[0] for row in rows]
    validate_replacement_rows(prepared, manifest)
    prepared[25]["completion"] = prepared[0]["completion"]
    with pytest.raises(ValueError):
        validate_replacement_rows(prepared, manifest)


def test_refuse_other_invalid_source_and_missing_identity():
    rows = _source()
    rows[2]["completion"] = "invalid"
    with pytest.raises(ValueError, match="named failed"):
        _build(rows)
    with pytest.raises(ValueError, match="donor_identity"):
        _build(donor_identity="q25")
    rows = _source()
    rows[0].pop("source_id")
    rows[0]["metadata"] = {}
    with pytest.raises(ValueError, match="unique source IDs"):
        _build(rows)


def test_refuse_missing_exclusion_and_fabricated_occurrence():
    rows, manifest = _build()
    manifest["excluded_sources"] = []
    with pytest.raises(ValueError, match="reserved"):
        validate_replacement_rows(rows, manifest)
    rows, manifest = _build()
    rows[25]["metadata"].pop("repeated_from")
    with pytest.raises(ValueError, match="payload or provenance"):
        validate_replacement_rows(rows, manifest)


def test_strict_reader_never_silently_drops_malformed_rows(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(_source()[0]) + '\n{"broken":\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        read_rows_strict(path)


def test_checked_in_benchmark_retains_exact_byte_hash_and_true_identity_count():
    folder = Path(__file__).resolve().parents[1] / "data/eval/golden32_archive_repeat_v1"
    path = folder / "golden32.jsonl"
    manifest = json.loads((folder / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["output_sha256"]
    rows = read_rows_strict(path)
    validate_replacement_rows(rows, manifest)
    assert rows[25]["source_id"] == rows[0]["source_id"] == "q_017270"
    assert manifest["excluded_sources"][0]["source_id"] == "q_012053"
