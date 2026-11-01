"""Bounded proof of a nonempty native source Card link collection.

Only explicit HTTPS URLs in native-owned Text/List/Button fields certify the
early return. Other URL-looking forms are uncertain, never proof of absence.
This helper supplies no content or action credit and never rewrites a target.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from .list_evidence_v5_4 import _NATIVE_URL_LIKE, parse_source_links

_BLOCKED_HOSTS = frozenset({
    "example.com", "example.org", "example.net", "localhost", "loremflickr.com",
    "picsum.photos", "placehold.co", "placeholder.com", "dummyimage.com",
    "via.placeholder.com", "placekitten.com", "placebear.com", "fakeimg.pl",
    "icon.url", "static.icons",
})
_BLOCKED_TLDS = frozenset({
    "local", "localhost", "test", "invalid", "example", "png", "jpg", "jpeg",
    "svg", "webp", "gif", "bmp", "ico", "json", "xml", "txt", "csv", "md",
    "pdf", "zip", "apk",
})
_TEXT_FIELDS = ("text", "title", "label", "content", "value")
_LINK_FIELDS = ("url", "href", "link", "source")


def native_https_link_proven_v2(value: Any) -> bool:
    """A strict subset of SafeContentPolicy.sanitizeActionUrl, not a replacement."""
    if not isinstance(value, str):
        return False
    value = value.strip()
    # Java URI rejects these characters; avoid Python's more permissive parser.
    if (not value.isascii() or re.search(r'[\s<>"{}|\\^`\[\]]', value)
            or re.search(r"%(?![0-9a-fA-F]{2})", value)):
        return False
    try:
        uri = urlsplit(value)
        host = (uri.hostname or "").lower()
        if uri.scheme.lower() != "https" or uri.username is not None or uri.port is not None:
            return False
    except ValueError:
        return False
    labels = host.split(".")
    return (
        len(labels) >= 2
        and all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part) for part in labels)
        and re.fullmatch(r"[a-z]{2,24}", labels[-1]) is not None
        and labels[-1] not in _BLOCKED_TLDS
        and not any(host == blocked or host.endswith("." + blocked) for blocked in _BLOCKED_HOSTS)
    )


def _possible(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_NATIVE_URL_LIKE.search(value) or re.search(r"\{\{u\d+\}\}|\btel:", value, re.IGNORECASE))
    if isinstance(value, Mapping):
        return any(_possible(item) for item in value.values())
    if isinstance(value, list):
        return any(_possible(item) for item in value)
    return False


def source_card_link_presence_v2(kind: str, props: Mapping[str, Any], on: Any) -> tuple[bool, bool]:
    """Return (proven nonempty, possibly nonempty) for one native-owned node."""
    candidates: list[Any] = []
    possible = False
    if kind == "text":
        raw = next((props[key] for key in _TEXT_FIELDS if props.get(key) is not None), "")
        possible = _possible(raw)
        if isinstance(raw, str):
            candidates.extend(url for _, url in parse_source_links(raw, pilot_fidelity_v2=True))
    elif kind == "list":
        items = props.get("items")
        possible = _possible(items)
        for item in items if isinstance(items, list) else ():
            if isinstance(item, str):
                candidates.extend(url for _, url in parse_source_links(item, pilot_fidelity_v2=True))
            elif isinstance(item, Mapping):
                candidates.extend(item.get(key) for key in _LINK_FIELDS)
                for key in (*_TEXT_FIELDS, "name"):
                    raw = item.get(key)
                    if isinstance(raw, str):
                        candidates.extend(url for _, url in parse_source_links(raw, pilot_fidelity_v2=True))
    elif kind == "button":
        candidates.extend(props.get(key) for key in _LINK_FIELDS)
        if isinstance(on, Mapping):
            for event in ("press", "click", "tap", "select", "open"):
                action = on.get(event)
                if isinstance(action, Mapping):
                    params = action.get("params")
                    for values in (params, action):
                        if isinstance(values, Mapping):
                            candidates.extend(values.get(key) for key in ("url", "href", "link"))
        possible = _possible(candidates)
    elif kind == "table":
        # Native table-specific extraction is deliberately not certified here.
        possible = _possible(props) or bool(props.get("statePath"))
    proven = any(native_https_link_proven_v2(url) for url in candidates)
    return proven, possible or proven
