"""Explicit, lossless reference namespaces for semantic scoring only.

Mappings are provenance supplied by the caller, never inferred from a candidate.
They cannot repair syntax, invent missing references, or alter the raw hash.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .candidate_normalization import _canonical_json, _hash_text


REFERENCE_POLICY_VERSION = "1.0.0"
REFERENCE_TOKEN = re.compile(r"\[(?:[A-Z][A-Z0-9]*_)*\d+\]")


def validated_reference_map(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("reference_map must be a placeholder-to-reference mapping")
    result: dict[str, str] = {}
    for token, reference in value.items():
        if (
            not isinstance(token, str)
            or REFERENCE_TOKEN.fullmatch(token) is None
            or not isinstance(reference, str)
            or not reference.strip()
        ):
            raise ValueError("reference_map entries require a placeholder token and nonempty string")
        result[token] = reference
    return result


def reference_map_hash(value: Mapping[str, str] | None) -> str:
    return _hash_text(_canonical_json(validated_reference_map(value)))


def restore_semantic_references(value: Any, references: Mapping[str, str]) -> Any:
    """Replace exact tokens in values once; never reinterpret replacement text."""
    if isinstance(value, str):
        return REFERENCE_TOKEN.sub(lambda match: references.get(match[0], match[0]), value)
    if isinstance(value, Mapping):
        return {key: restore_semantic_references(item, references) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_semantic_references(item, references) for item in value]
    return value
