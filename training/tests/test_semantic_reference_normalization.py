"""Stage3 token targets and raw sources become consistent student examples."""
from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))

from ir_training.data.express_preparation import _api, serialize_checked
from ir_training.data.reference_binding import reference_tokens
from ir_training.data.semantic_reference_normalization import (
    normalize_semantic_candidate,
    verify_normalized_semantic_candidate,
)
from ir_training.data.url_preprocess import restore_url_placeholders


def candidate():
    return {
        "response_text": "Synthetic status source. Media: Icon=https://synthetic.invalid/weather.svg",
        "a2ui_express": '<a2ui>\nroot=Icon("[ICON_URL_7]")\n</a2ui>',
        "reference_map": {"[ICON_URL_7]": "https://synthetic.invalid/weather.svg"},
        "assets": [],
        "augmentation": {"source_sha256": "producer hash is kept by caller"},
    }


def test_masked_target_and_raw_source_share_registry_without_mutation():
    source = candidate()
    original = deepcopy(source)
    result = normalize_semantic_candidate(source)
    assert source == original
    assert "https://" not in result["response_text"]
    assert "[ICON_URL_7]" not in result["completion"]
    graph = serialize_checked(result["completion"], "root-first").graph
    assert reference_tokens(graph) <= reference_tokens(result["response_text"])
    metadata = result["metadata"]
    assert metadata["semantic_reference_normalization"]["producer_source_sha256"] == hashlib.sha256(source["response_text"].encode()).hexdigest()
    assert restore_url_placeholders(result["response_text"], metadata["url_preprocessing"]["url_map"]) == source["response_text"]
    assert restore_url_placeholders(graph, metadata["url_preprocessing"]["url_map"])["elements"]["root"]["props"]["name"] == source["reference_map"]["[ICON_URL_7]"]
    verify_normalized_semantic_candidate(result, source)


def test_same_destination_shared_across_source_roles():
    source = {"response_text": "Source: https://example.test/page\nAction: [Button: View] https://example.test/page",
              "a2ui_express": '<a2ui>\nroot=Text("[URL_9]")\n</a2ui>',
              "reference_map": {"[URL_9]": "https://example.test/page"}}
    result = normalize_semantic_candidate(source)
    assert len(result["metadata"]["url_preprocessing"]["url_map"]) == 1
    verify_normalized_semantic_candidate(result, source)


def test_plain_candidate_roundtrips():
    source = {"response_text": "There are 12 boxes.", "a2ui_express": '<a2ui>\nroot=Text("There are 12 boxes.")\n</a2ui>'}
    result = normalize_semantic_candidate(source)
    assert result["response_text"] == source["response_text"]
    assert result["metadata"]["url_preprocessing"]["url_map"] == {}
    verify_normalized_semantic_candidate(result, source)


def test_invented_target_reference_rejected():
    source = candidate()
    source["reference_map"]["[ICON_URL_7]"] = "https://invented.example/icon.svg"
    with pytest.raises(ValueError, match="unbound_target_references"):
        normalize_semantic_candidate(source)


@pytest.mark.parametrize("masked_source", [False, True])
def test_missing_target_binding_not_inferred(masked_source):
    source = candidate()
    source.pop("reference_map")
    if masked_source:
        source["response_text"] = "Media: Icon=[ICON_URL_7]"
    with pytest.raises(ValueError, match="unbound_target_references|verified reference bindings"):
        normalize_semantic_candidate(source)


@pytest.mark.parametrize("field", ["source", "target", "mapping", "evidence"])
def test_portable_verification_rejects_changed_normalized_row(field):
    source = candidate()
    row = normalize_semantic_candidate(source)
    if field == "source":
        row["response_text"] += " Modified fact."
    elif field == "target":
        row["completion"] = '<a2ui>\nroot=Icon("[ICON_URL_99]")\n</a2ui>'
    elif field == "mapping":
        next(iter(row["metadata"]["url_preprocessing"]["url_map"].values()))["url"] = "https://tampered.example/icon.svg"
    else:
        row["metadata"]["semantic_reference_normalization"]["producer_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Semantic normalized|Semantic reference normalization"):
        verify_normalized_semantic_candidate(row, source)


def test_hash_verified_historical_graph_can_restore_without_explicit_map():
    source = candidate()
    mapping = source.pop("reference_map")
    graph = serialize_checked(source["a2ui_express"], "root-first").graph
    source["canonical_graph"] = restore_url_placeholders(graph, mapping)
    _, _, semantic_hash, _ = _api()
    source["canonical_graph_hash"] = semantic_hash(source["canonical_graph"])
    result = normalize_semantic_candidate(source)
    assert result["metadata"]["reference_binding"]["method"] == "hashed_graph_reference_alignment"
    verify_normalized_semantic_candidate(result, source)
