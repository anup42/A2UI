"""Bounded regressions for the dry-run URL repair admission policy."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/audits"))
from offline_url_probe_20260913 import express, probe


def target(text):
    return express.encode({"root": "body", "state": {}, "elements": {
        "body": {"type": "Text", "props": {"text": text}, "children": []}
    }}, shorten_ids=False)


def check(source, output):
    return probe(("synthetic", 1, source, target(output), "source", "target"))


def test_raw_url_roundtrip_and_closure():
    result = check("Read https://example.com/article", "Read https://example.com/article")
    assert result["status"] == "verified_reversible_url_normalization"
    assert result["exact_source_restored"]
    assert result["exact_graph_restored"]
    assert result["normalized_strict_valid"]


def test_target_only_raw_url_is_ambiguous():
    result = check("Read the article", "Read https://example.com/article")
    assert result["status"] == "ambiguous_not_counted"
    assert result["detail"] == "normalized_target_placeholders_absent_source"


def test_closed_existing_symbolic_tokens_are_not_rewritten():
    result = check("Read [SOURCE_URL_1]", "Read [SOURCE_URL_1]")
    assert result["status"] == "already_symbolic_closed"
    assert result["normalized_source_sha256"] == ""


def test_mixed_existing_symbolic_and_literal_reference_is_ambiguous():
    result = check("Read [SOURCE_URL_1] or https://example.com", "Read [SOURCE_URL_1]")
    assert result["status"] == "ambiguous_not_counted"
    assert result["detail"] == "mixed_existing_placeholders_and_explicit_references"


def test_no_reference_requires_no_url_transform():
    assert check("Read the article", "Read the article")["status"] == "no_url_transform_needed"
