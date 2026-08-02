"""Shared helpers for lossless GenUICraft IR codecs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..flat_spec_contract import canonicalize_flat_spec, coerce_and_validate
from ..flat_spec_semantics import iter_renderer_references, renderer_reference_inventory_hash

ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = ROOT / "schema" / "genuicraft_a2ui_catalog_v1.json"
MANIFEST_PATH = ROOT / "schema" / "genuicraft_ir_formats.manifest.json"


def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


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
    """Deterministically rename IDs while rewriting every renderer reference."""
    canonical = canonicalize_flat_spec(dict(spec))
    elements = canonical.get("elements", {})
    if not isinstance(elements, dict) or not elements:
        return canonical
    root = str(canonical.get("root", ""))
    order: list[str] = []
    seen: set[str] = set()

    def walk(element_id: str) -> None:
        if element_id in seen or element_id not in elements:
            return
        seen.add(element_id); order.append(element_id)
        element = elements.get(element_id)
        if isinstance(element, Mapping):
            for ref in iter_renderer_references(element):
                walk(ref.target_id)

    walk(root)
    for element_id in elements:
        walk(str(element_id))
    if not shorten:
        mapping = {old: old for old in order}
        if reserve_root and root != "root":
            if "root" in mapping:
                replacement_index = 1
                replacement = "root_1"
                while replacement in mapping.values():
                    replacement_index += 1
                    replacement = f"root_{replacement_index}"
                mapping["root"] = replacement
            mapping[root] = "root"
    else:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        def name(index: int) -> str:
            out = ""
            n = index
            while True:
                out = alphabet[n % 26] + out
                n = n // 26 - 1
                if n < 0:
                    return out
        mapping = {root: "root"}
        mapping.update({old: name(i) for i, old in enumerate([x for x in order if x != root])})
    rewritten: dict[str, Any] = {}
    for old_id in order:
        raw = deepcopy(elements[old_id])
        if isinstance(raw, dict):
            for ref in iter_renderer_references(raw):
                if ref.target_id in mapping:
                    _set_path(raw, ref.source_path, mapping[ref.target_id])
        rewritten[mapping[old_id]] = raw
    return {
        "root": mapping.get(root, root),
        "state": deepcopy(canonical.get("state") if isinstance(canonical.get("state"), dict) else {}),
        "elements": rewritten,
    }


def semantic_canonical(spec: Mapping[str, Any]) -> dict[str, Any]:
    result = coerce_and_validate(dict(spec))
    if not result.is_valid or result.spec is None:
        raise ValueError(result.error or "Invalid FlatSpec")
    return rewrite_element_ids(result.spec, shorten=True)


def semantic_hash(spec: Mapping[str, Any]) -> str:
    payload = semantic_canonical(spec)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def codec_identity() -> dict[str, Any]:
    manifest = load_manifest()
    return {
        "manifestVersion": manifest.get("manifestVersion"),
        "upstreamCommit": manifest.get("upstreamCommit"),
        "protocolVersion": manifest.get("protocolVersion"),
        "compactIrVersion": manifest.get("compactIrVersion"),
        "expressVersion": manifest.get("expressVersion"),
        "catalogIdentityHash": manifest.get("catalogIdentityHash"),
        "rendererReferenceInventoryHash": renderer_reference_inventory_hash(),
    }
