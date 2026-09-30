"""Shared helpers for lossless GenUICraft IR codecs."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..renderer_semantics import iter_renderer_references, renderer_reference_inventory_hash
from .canonical import rewrite_element_ids as _rewrite_canonical_element_ids
from .canonical import semantic_hash as _canonical_semantic_hash

ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = ROOT / "schema" / "genuicraft_a2ui_catalog_v1.json"
MANIFEST_PATH = ROOT / "schema" / "genuicraft_ir_formats.manifest.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _set_path(root: Any, path: str, value: Any) -> None:
    """Set one simple dot/bracket path emitted by flat_spec_semantics."""
    parts: list[str | int] = []
    token = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if token:
                parts.append(token); token = ""
            i += 1; continue
        if ch == "[":
            if token:
                parts.append(token); token = ""
            end = path.index("]", i)
            parts.append(int(path[i + 1:end]))
            i = end + 1; continue
        token += ch; i += 1
    if token:
        parts.append(token)
    if not parts:
        raise ValueError("Renderer reference path is empty")
    current = root
    for part in parts[:-1]:
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Renderer reference path {path!r} does not exist") from exc
    current[parts[-1]] = value


def rewrite_element_ids(
    spec: Mapping[str, Any],
    *,
    shorten: bool = True,
    reserve_root: bool = False,
) -> dict[str, Any]:
    """Deterministically rename IDs without invoking legacy normalization."""
    return _rewrite_canonical_element_ids(spec, shorten=shorten, reserve_root=reserve_root)


def semantic_canonical(spec: Mapping[str, Any]) -> dict[str, Any]:
    return rewrite_element_ids(spec, shorten=True, reserve_root=True)


def semantic_hash(spec: Mapping[str, Any]) -> str:
    return _canonical_semantic_hash(spec)


def codec_identity() -> dict[str, Any]:
    manifest = load_manifest()
    return {
        "manifestVersion": manifest.get("manifestVersion"),
        "upstreamCommit": manifest.get("upstreamCommit"),
        "protocolVersion": manifest.get("protocolVersion"),
        "expressVersion": manifest.get("expressVersion"),
        "catalogIdentityHash": manifest.get("catalogIdentityHash"),
        "rendererReferenceInventoryHash": renderer_reference_inventory_hash(),
    }
