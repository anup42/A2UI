"""Lossless reference binding at the historical dataset import boundary.

No media is loaded, downloaded, or checked for existence. A reference is an
identity, not a file dependency. Old masked targets can be recovered only from
an explicit map or an independently hashed graph with identical non-reference
content. Placeholder suffixes are never interpreted as destination indices.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ir_training.data.ir_targets import canonical_graph_from_source, semantic_hash
from ir_training.data.url_preprocess import _PLACEHOLDER_RE, restore_url_placeholders

REFERENCE_BINDING_VERSION = "source-target-reference-v1"


@dataclass(frozen=True)
class BoundReferences:
    source: str
    graph: dict[str, Any]
    evidence: dict[str, Any]


def reference_tokens(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(_PLACEHOLDER_RE.findall(value))
    if isinstance(value, Mapping):
        return set().union(*(reference_tokens(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(reference_tokens(v) for v in value)) if value else set()
    return set()


def _reference_map(record: Mapping[str, Any]) -> dict[str, str]:
    metadata = record.get("metadata")
    url_meta = metadata.get("url_preprocessing") if isinstance(metadata, Mapping) else None
    candidates = [record.get("reference_map"), record.get("asset_placeholder_map"), record.get("url_map")]
    if isinstance(url_meta, Mapping):
        candidates.append(url_meta.get("url_map"))
    merged: dict[str, str] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for key, value in candidate.items():
            if not _PLACEHOLDER_RE.fullmatch(str(key)):
                raise ValueError(f"invalid_reference_map_key:{key}")
            raw = value.get("url") if isinstance(value, Mapping) else value
            if not isinstance(raw, str) or not raw or reference_tokens(raw):
                raise ValueError(f"invalid_reference_map_value:{key}")
            if key in merged and merged[key] != raw:
                raise ValueError(f"conflicting_reference_map:{key}")
            merged[str(key)] = raw
    return merged


def _align_references(raw: Any, restored: Any, mapping: dict[str, str], path: str = "$") -> None:
    if isinstance(raw, Mapping) and isinstance(restored, Mapping):
        if set(raw) != set(restored):
            raise ValueError(f"reference_graph_shape_changed:{path}")
        for key in raw:
            _align_references(raw[key], restored[key], mapping, f"{path}.{key}")
        return
    if isinstance(raw, list) and isinstance(restored, list):
        if len(raw) != len(restored):
            raise ValueError(f"reference_graph_shape_changed:{path}")
        for index, (left, right) in enumerate(zip(raw, restored)):
            _align_references(left, right, mapping, f"{path}[{index}]")
        return
    if isinstance(raw, str) and isinstance(restored, str):
        matches = list(_PLACEHOLDER_RE.finditer(raw))
        if not matches:
            if raw != restored:
                raise ValueError(f"reference_nonreference_content_changed:{path}")
            return
        pattern = ""
        offset = 0
        for match in matches:
            # Adjacent tokens have no provable split boundary.
            if pattern and match.start() == offset:
                raise ValueError(f"ambiguous_adjacent_references:{path}")
            pattern += re.escape(raw[offset:match.start()]) + "(.+?)"
            offset = match.end()
        pattern += re.escape(raw[offset:])
        aligned = re.fullmatch(pattern, restored, flags=re.DOTALL)
        if aligned is None:
            raise ValueError(f"reference_nonreference_content_changed:{path}")
        for token, value in zip((m.group() for m in matches), aligned.groups()):
            if reference_tokens(value):
                raise ValueError(f"reference_not_restored:{path}")
            if token in mapping and mapping[token] != value:
                raise ValueError(f"conflicting_reference_binding:{token}")
            mapping[token] = value
        return
    if type(raw) is not type(restored) or raw != restored:
        raise ValueError(f"reference_nonreference_content_changed:{path}")


def bind_source_target_references(
    source: str, graph: Mapping[str, Any], record: Mapping[str, Any]
) -> BoundReferences:
    graph = copy.deepcopy(dict(graph))
    tokens = reference_tokens(graph)
    evidence: dict[str, Any] = {"version": REFERENCE_BINDING_VERSION, "method": "unmasked", "restored_tokens": 0}
    if not tokens and not reference_tokens(source):
        return BoundReferences(source, graph, evidence)
    mapping = _reference_map(record)
    if mapping:
        missing = (tokens | reference_tokens(source)) - set(mapping)
        if missing:
            raise ValueError("missing_reference_mapping:" + ",".join(sorted(missing)))
        expected_source_hash = record.get("reference_source_sha256")
        if expected_source_hash and hashlib.sha256(source.encode("utf8")).hexdigest() != expected_source_hash:
            raise ValueError("reference_source_hash_mismatch")
        evidence["method"] = "explicit_reference_map"
    elif tokens <= reference_tokens(source):
        # Both sides already use the same source-visible opaque identities.
        # The caller still checks for token collisions after preprocessing.
        evidence["method"] = "source_visible_placeholders"
        return BoundReferences(source, graph, evidence)
    else:
        restored = record.get("canonical_graph")
        recorded_hash = record.get("canonical_graph_hash")
        if not isinstance(restored, Mapping) or not isinstance(recorded_hash, str):
            raise ValueError("unbound_target_references:no_verified_restoration")
        if semantic_hash(restored) != recorded_hash:
            raise ValueError("restored_graph_hash_mismatch")
        normalized = record.get("model_native_output_normalized")
        if isinstance(normalized, str) and normalized.strip():
            check = canonical_graph_from_source(normalized, source_format="a2ui_express_v1")
            if semantic_hash(check) != recorded_hash:
                raise ValueError("normalized_restoration_hash_mismatch")
        _align_references(graph, restored, mapping)
        evidence["method"] = "hashed_graph_reference_alignment"
        evidence["restored_graph_sha256"] = recorded_hash
    restored_graph = restore_url_placeholders(graph, mapping)
    restored_source = restore_url_placeholders(source, mapping)
    if reference_tokens(restored_graph) or reference_tokens(restored_source):
        raise ValueError("unresolved_reference_after_restoration")
    if evidence["method"] == "hashed_graph_reference_alignment" and semantic_hash(restored_graph) != recorded_hash:
        raise ValueError("restoration_changes_graph_semantics")
    if evidence["method"] == "explicit_reference_map" and record.get("canonical_graph_hash"):
        if semantic_hash(restored_graph) != record["canonical_graph_hash"]:
            raise ValueError("explicit_reference_restoration_hash_mismatch")
    evidence["restored_tokens"] = len(tokens)
    evidence["mapping_sha256"] = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode("utf8")).hexdigest()
    return BoundReferences(restored_source, restored_graph, evidence)


def ground_preprocessed_references(source: str, graph: Any, assets: Any, url_map: Mapping[str, Any]) -> str:
    """Expose only explicitly supplied assets/aliases; reject invented targets.

    Asset declarations supply identity to the model; their physical bytes are
    not necessary. Role-scoped aliases may reuse a source-visible destination.
    """
    source_tokens = reference_tokens(source)
    target_tokens = reference_tokens(graph)
    declared_assets = reference_tokens(assets)
    extra_assets = sorted((target_tokens - source_tokens) & declared_assets)
    if extra_assets:
        source += "\n\nAvailable asset references: " + ", ".join(extra_assets)
        source_tokens.update(extra_assets)
    aliases = []
    for token in sorted(target_tokens - source_tokens):
        entry = url_map.get(token)
        destination = entry.get("url") if isinstance(entry, Mapping) else None
        equivalent = next((other for other in sorted(source_tokens)
                           if isinstance(url_map.get(other), Mapping)
                           and url_map[other].get("url") == destination), None) if destination else None
        if equivalent:
            aliases.append(f"{token} = {equivalent}")
            source_tokens.add(token)
    if aliases:
        source += "\n\nReference aliases (same destination):\n" + "\n".join(aliases)
    unbound = target_tokens - source_tokens
    if unbound:
        raise ValueError("unbound_target_references:" + ",".join(sorted(unbound)))
    return source
