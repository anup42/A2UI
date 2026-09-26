"""Lossless Stage3-to-student reference normalization, with portable evidence.

This is reference transport through existing codecs, not target repair. The
teacher's source/completion and hashes remain untouched in the sealed record.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ir_training.data.express_preparation import _api, serialize_checked
from ir_training.data.reference_binding import (
    bind_source_target_references,
    ground_preprocessed_references,
)
from ir_training.data.url_preprocess import (
    SOURCE_IDENTITY_BINDING,
    preprocess_training_urls,
    restore_url_placeholders,
)

VERSION = "semantic-reference-normalization-v1"
METADATA_KEYS = ("semantic_reference_normalization", "url_preprocessing", "reference_binding")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_semantic_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a completed Stage3 record using one source-first URL registry."""
    source, completion = candidate.get("response_text"), candidate.get("a2ui_express")
    if not isinstance(source, str) or not source.strip() or not isinstance(completion, str) or not completion.strip():
        raise ValueError("Semantic reference normalization requires source and Express completion")
    target = serialize_checked(completion, "root-first")
    bound = bind_source_target_references(source, target.graph, candidate)
    if bound.evidence["method"] == "source_visible_placeholders":
        # Unlike legacy general imports, these newly generated examples must
        # prove bindings. Seeing an unknown token twice is not proof of a URL.
        raise ValueError("Semantic reference normalization requires verified reference bindings")
    assets = candidate.get("assets") or []
    processed = preprocess_training_urls(
        bound.source, {"graph": bound.graph, "assets": assets},
        enabled=True, binding_policy=SOURCE_IDENTITY_BINDING,
    )
    graph, masked_assets = processed.canonical_graph["graph"], processed.canonical_graph["assets"]
    grounded_source = ground_preprocessed_references(processed.response_text, graph, masked_assets, processed.url_map)
    # No reference pass may change facts, component structure or action data.
    if restore_url_placeholders(graph, processed.url_map) != bound.graph:
        raise ValueError("Semantic reference normalization changed target semantics")
    if restore_url_placeholders(processed.response_text, processed.url_map) != bound.source:
        raise ValueError("Semantic reference normalization changed source content")
    _, express, semantic_hash, _ = _api()
    normalized = serialize_checked(express.encode(graph, shorten_ids=False), "root-first")
    if normalized.graph != graph:
        raise ValueError("Semantic reference normalization changed canonical graph")
    evidence = {
        "version": VERSION,
        "producer_source_sha256": _sha(source),
        "producer_completion_sha256": _sha(completion),
        "normalized_source_sha256": _sha(grounded_source),
        "normalized_semantic_sha256": normalized.semantic_sha256,
        "restored_semantic_sha256": semantic_hash(bound.graph),
    }
    return {"response_text": grounded_source, "completion": normalized.text,
            "metadata": {
                "semantic_reference_normalization": evidence,
                "reference_binding": bound.evidence,
                "url_preprocessing": {"enabled": True, "binding_policy": SOURCE_IDENTITY_BINDING,
                                      "url_map": processed.url_map, "metrics": processed.metrics},
            }}


def verify_normalized_semantic_candidate(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """Recompute portable normalization from the sealed original Stage3 record."""
    expected = normalize_semantic_candidate(candidate)
    if row.get("response_text") != expected["response_text"]:
        raise ValueError("Semantic normalized source differs from Stage3 evidence")
    actual = serialize_checked(row.get("completion"), "root-first")
    target = serialize_checked(expected["completion"], "root-first")
    if actual.graph != target.graph:
        raise ValueError("Semantic normalized target differs from Stage3 evidence")
    metadata = row.get("metadata") or {}
    if any(metadata.get(key) != expected["metadata"][key] for key in METADATA_KEYS):
        raise ValueError("Semantic reference normalization evidence changed")
