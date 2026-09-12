from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.data.chat_templates import build_messages
from ir_training.data.express_preparation import prepare_row
from ir_training.data.golden_replacement import read_rows_strict
from ir_training.data.golden35_subset import (
    APPROVED_SOURCE_IDS, FROZEN_EXCLUSION_REASONS, SOURCE_GENUI_SHA256,
    SOURCE_RESPONSES_SHA256, build_golden35_subset, validate_subset_rows,
)


def _fixtures():
    originals = []
    for index in range(1, 51):
        source_id = f"q_{index:06d}"
        response = f"Independent source response {index} at https://example.org/{index}"
        completion = f'<a2ui>\nroot=Text("Case {index}")\n</a2ui>'
        originals.append({"id": f"row{index}", "source_id": source_id, "response_id": f"r_{index:06d}_01",
                          "response_text": response, "intent_bucket": "fixture", "metadata": {"query_id": source_id},
                          "messages": build_messages("System", response, completion, target_format="a2ui_express_v1"),
                          "completion": completion})
    return originals


def _build(prepared=None, originals=None, **kwargs):
    originals = originals or _fixtures()
    prepared = prepared if prepared is not None else [row for row in originals if row["source_id"] in APPROVED_SOURCE_IDS]
    return build_golden35_subset(prepared, originals,
        source_genui_sha256=kwargs.get("source_genui_sha256", SOURCE_GENUI_SHA256),
        source_responses_sha256=SOURCE_RESPONSES_SHA256)


def test_builder_freezes_approved_membership_even_if_other_sources_become_valid():
    originals = _fixtures()
    untouched = deepcopy(originals)
    rows, manifest = _build(prepared=originals, originals=originals)
    assert originals == untouched
    assert tuple(row["source_id"] for row in rows) == APPROVED_SOURCE_IDS
    assert len(manifest["accepted_sources"]) == 35 and len(manifest["excluded_sources"]) == 15
    assert {item["source_id"] for item in manifest["excluded_sources"]} == set(FROZEN_EXCLUSION_REASONS)
    assert all(item["raw_response_sha256"] != item["masked_response_sha256"] for item in manifest["excluded_sources"])


def test_missing_approved_case_is_not_filled_from_an_excluded_case():
    originals = _fixtures()
    prepared = [row for row in originals if row["source_id"] != APPROVED_SOURCE_IDS[0]]
    with pytest.raises(ValueError, match="no longer materialize"):
        _build(prepared=prepared, originals=originals)
    with pytest.raises(ValueError, match="source hashes"):
        _build(source_genui_sha256="0" * 64)


def test_prepared_reordering_is_allowed_but_target_or_source_tampering_is_not():
    rows, manifest = _build()
    prepared = [prepare_row(row, "bottom-up")[0] for row in reversed(rows)]
    validate_subset_rows(prepared, manifest)
    prepared[0]["response_text"] = "Changed source"
    with pytest.raises(ValueError):
        validate_subset_rows(prepared, manifest)
    rows, manifest = _build()
    manifest["accepted_sources"][0]["semantic_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        validate_subset_rows(rows, manifest)


def test_exclusion_and_membership_evidence_cannot_shrink():
    rows, manifest = _build()
    manifest["excluded_sources"].pop()
    with pytest.raises(ValueError, match="exclusion count"):
        validate_subset_rows(rows, manifest)
    rows, manifest = _build()
    manifest["accepted_source_ids"][0] = "q_000001"
    with pytest.raises(ValueError, match="approved membership"):
        validate_subset_rows(rows, manifest)


def test_checked_in_golden35_artifact_has_exact_hash_and_strict_membership():
    folder = ROOT / "data/eval/golden35_v1"
    source = folder / "golden35.jsonl"
    manifest = json.loads((folder / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["output_sha256"]
    rows = read_rows_strict(source)
    validate_subset_rows(rows, manifest)
    assert len(rows) == len({row["source_id"] for row in rows}) == 35
    assert not set(FROZEN_EXCLUSION_REASONS) & {row["source_id"] for row in rows}
