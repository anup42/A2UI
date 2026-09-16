from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.data.bixby50 import (
    APPROVED_SOURCE_IDS, SOURCE_RESPONSES_SHA256, build_bixby50,
    serialize_rows, validate_source_only_rows,
)


def _source():
    return [{"scenario_id": source_id, "provider": "Perplexity", "typed_query_exact": True,
             "status": "complete", "markdown_status": "markdown", "domain": "Fixture",
             "user_query": f"Question {source_id}", "markdown_response": f"Response {source_id}\n",
             "artifact_dir": "PRIVATE", "request_id": "PRIVATE", "device_serial": "PRIVATE"}
            for source_id in APPROVED_SOURCE_IDS]


def _build():
    return build_bixby50(_source(), source_responses_sha256=SOURCE_RESPONSES_SHA256)


def test_builder_preserves_sources_without_private_metadata_or_targets():
    source = _source()
    untouched = deepcopy(source)
    rows, manifest = build_bixby50(source, source_responses_sha256=SOURCE_RESPONSES_SHA256)
    assert source == untouched
    assert [row["response_text"] for row in rows] == [row["markdown_response"] for row in source]
    assert b"PRIVATE" not in serialize_rows(rows)
    assert not any("completion" in row or "messages" in row or "canonical_graph" in row for row in rows)
    assert manifest["benchmark_kind"] == "source_only_holdout"
    validate_source_only_rows(rows, manifest)


def test_prepared_rows_and_reordering_preserve_source_identity():
    rows, manifest = _build()
    for row in rows:
        row.update(raw_response_text=row["response_text"], completion="",
                   target_validation="not_applicable_source_only", messages=[
                       {"role": "system", "content": "Contract"},
                       {"role": "user", "content": "Example"},
                       {"role": "assistant", "content": "Example only"},
                       {"role": "user", "content": row["response_text"]},
                   ])
    validate_source_only_rows(list(reversed(rows)), manifest)
    rows[0]["response_text"] += " Invented citation https://example.org/"
    with pytest.raises(ValueError, match="prepared source response changed"):
        validate_source_only_rows(rows, manifest)


@pytest.mark.parametrize("field,value", [
    ("completion", '<a2ui>root=Text("Invented")</a2ui>'),
    ("canonical_graph", {"root": "fake"}), ("completion_targets", {"fake": "target"}),
    ("reference_available", True), ("evaluation_only", False),
    ("selection_role", "selection"), ("id", "changed"), ("response_id", "changed"),
    ("response_text", "changed"), ("target_validation", "strict_valid"),
    ("messages", [{"role": "assistant", "content": "invented"}]),
])
def test_reject_target_fabrication_or_identity_changes(field, value):
    rows, manifest = _build()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_source_only_rows(rows, manifest)


def test_membership_flags_and_manifest_cannot_silently_change():
    rows, manifest = _build()
    with pytest.raises(ValueError, match="50 unique"):
        validate_source_only_rows(rows[:-1], manifest)
    with pytest.raises(ValueError, match="50 unique"):
        validate_source_only_rows(rows[:-1] + [rows[0]], manifest)
    broken = deepcopy(rows)
    del broken[0]["metadata"]["reference_available"]
    with pytest.raises(ValueError, match="metadata"):
        validate_source_only_rows(broken, manifest)
    manifest["accepted_sources"][0]["response_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest checksum"):
        validate_source_only_rows(rows, manifest)


def test_builder_rejects_different_delivery_incomplete_or_duplicate_collection():
    with pytest.raises(ValueError, match="source hash"):
        build_bixby50(_source(), source_responses_sha256="0" * 64)
    source = _source()
    source[0]["typed_query_exact"] = False
    with pytest.raises(ValueError, match="collection evidence"):
        build_bixby50(source, source_responses_sha256=SOURCE_RESPONSES_SHA256)
    source = _source()
    source[-1] = source[0]
    with pytest.raises(ValueError, match="50 approved"):
        build_bixby50(source, source_responses_sha256=SOURCE_RESPONSES_SHA256)


def test_checked_in_artifact_has_exact_hash_membership_and_captured_refusal():
    folder = ROOT / "data/eval/bixby50_v1"
    raw = (folder / "bixby50.jsonl").read_bytes()
    manifest = json.loads((folder / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(raw).hexdigest() == manifest["output_sha256"]
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    validate_source_only_rows(rows, manifest)
    assert [row["source_id"] for row in rows] == list(APPROVED_SOURCE_IDS)
    refusal = next(row for row in rows if row["source_id"] == "BXP-038")
    assert "cannot provide a source-cited comparison" in refusal["response_text"]
    assert refusal["metadata"]["collection_status"] == "complete_non_markdown"


def test_generator_refuses_existing_directory_before_reading_source(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/create_bixby50.py"),
                             "--output-dir", str(tmp_path), "--source", str(tmp_path / "absent")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
