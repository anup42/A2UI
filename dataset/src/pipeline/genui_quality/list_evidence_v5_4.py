"""CPU evidence for the literal List.items renderer contract.

Source-section strings use the existing source-link renderer route. Ordinary
strings remain text; explicit link fields render buttons in either context.

Parity limit: text parsing covers explicit HTTPS links and opaque reference
tokens. Native bare/www/protocol-relative host normalization also applies the
Android public-host policy; it is not inferred here without shared parity
fixtures. This helper does not certify all source-section interception paths.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlsplit

from ..list_items import LIST_LINK_FIELDS, LIST_TEXT_FIELDS, validate_list_items
from . import _core


def source_cue(value: Any) -> bool:
    return isinstance(value, str) and any(word in value.casefold() for word in ("source", "reference", "citation"))


def source_heading(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip(" #*_-:–—\t\n")
    return len(text) <= 44 and re.fullmatch(
        r"sources?|references?|citations?|links?|(?:source|reference|useful|external|related)\s+links?", text, re.I
    ) is not None


def self_source_cue(element_id: str, props: Mapping[str, Any]) -> bool:
    return source_cue(element_id) or any(source_cue(props.get(key)) for key in (
        "role", "semanticRole", "domain", "section", "title", "label", "heading", "accessibilityLabel", "ariaLabel"
    ))


def parse_source_links(text: str, *, pilot_fidelity_v2: bool = False) -> list[tuple[str, str]]:
    """Supported explicit HTTPS/opaque subset of native source-link parsing."""
    markdown = list(_core._MARKDOWN_LINK_RE.finditer(text))
    target_value = str.strip if pilot_fidelity_v2 else _core.normalize_url
    if markdown:
        return list(dict.fromkeys((_core.normalize_markdown(m[1]), target_value(m[2])) for m in markdown))
    result: list[tuple[str, str]] = []
    cursor = 0
    for match in _core._URL_RE.finditer(text):
        label = text[cursor:match.start()].strip(" -*:|[]()\t")
        label = re.sub(r"(?i)^(?:sources?|references?|citations?|links?)\s*:?\s*", "", label)
        label = re.sub(r"^\d+[.):\-]?\s*", "", label)
        result.append((_core.normalize_markdown(label), target_value(match[0])))
        cursor = match.end()
    return list(dict.fromkeys(result))


def literal_list_evidence(value: Any, *, source_context: bool = False, pilot_fidelity_v2: bool = False) -> tuple[list[str], list[tuple[str, str]]]:
    texts: list[str] = []
    links: list[tuple[str, str]] = []
    target_value = str.strip if pilot_fidelity_v2 else _core.normalize_url
    if not isinstance(value, list):
        return texts, links
    for item in value:
        if validate_list_items([item]) is not None:
            continue
        if isinstance(item, str):
            texts.append(item)
            if source_context:
                links.extend(parse_source_links(item, pilot_fidelity_v2=pilot_fidelity_v2))
            continue
        if source_context and pilot_fidelity_v2:
            texts.extend(dict.fromkeys(item[key] for key in LIST_TEXT_FIELDS if item.get(key, "").strip()))
            # FlatSpecRenderer.parseFlatSourceLinksFromListItems selects one
            # explicit target and its label, not all ordinary List fields.
            # Keep the existing supported HTTPS/opaque URL domain; this does
            # not claim parity for native bare-host normalization/safety.
            target = next((item[key] for key in LIST_LINK_FIELDS if item.get(key, "").strip() and classify_external(item[key])), "")
            if target:
                label = next((item[key].strip() for key in ("label", "title", "text", "name", "content", "value")
                              if item.get(key, "").strip() and (
                                  _explicit_source_label(item[key], target) if key == "label"
                                  else not _NATIVE_URL_LIKE.search(item[key]))), "")
                if not label:
                    host = (urlsplit(target).hostname or "").removeprefix("www.").split(".")[0]
                    label = " ".join(token[:1].upper() + token[1:] for token in re.split(r"[-_\s]+", host) if token) or "Source"
                links.append((label, target_value(target)))
            else:
                for key in ("text", "label", "title", "content", "value", "name"):
                    if item.get(key, "").strip():
                        links.extend(parse_source_links(item[key], pilot_fidelity_v2=pilot_fidelity_v2))
            continue
        item_texts = list(dict.fromkeys(item[key] for key in LIST_TEXT_FIELDS if item.get(key, "").strip()))
        texts.extend(item_texts)
        label = next(iter(item_texts), "")
        explicit = list(dict.fromkeys(item[key] for key in LIST_LINK_FIELDS if item.get(key, "").strip()))
        for target in explicit:
            if classify_external(target):
                links.append((label, target_value(target)))
        if source_context and not explicit:
            for text in item_texts:
                links.extend(parse_source_links(text, pilot_fidelity_v2=pilot_fidelity_v2))
    if source_context:
        # The source-section renderer collapses repeated URL buttons.
        by_url: dict[str, tuple[str, str]] = {}
        for item in links:
            by_url.setdefault(_source_url_key_v2(item[1]) if pilot_fidelity_v2 else item[1], item)
        links = list(by_url.values())
        if links:
            # Native source sections display link labels, not the serialized
            # markdown/URL line. Non-link paragraphs require ordinary Text.
            texts = [label for label, _ in links if label]
    return texts, links


def _source_url_key_v2(raw: str) -> str:
    """Native source-section dedup key; never used as destination identity."""
    # FlatSpecRenderer.flatCanonicalSourceUrl ignores fragments only when
    # deciding which source buttons are rendered. Keep the winning raw URL.
    try:
        uri = urlsplit(raw.strip())
        host = (uri.hostname or "").lower().removeprefix("www.")
        if not host:
            return raw.strip().lower()
        path = unquote(uri.path).rstrip("/")
        query = unquote(uri.query).strip()
        return (uri.scheme or "https").lower() + "://" + host + path + ("?" + query if query else "")
    except ValueError:
        return raw.strip().lower()


# NativeTextFormatter.URL_REGEX
_NATIVE_URL_LIKE = re.compile(
    r"(?:https?://|//)[^\s<>\]]+|(?<![@\w])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:[/?#][^\s<>\]]*)?", re.IGNORECASE
)


def _explicit_source_label(label: str, target: str) -> bool:
    """Mirror the renderer's display-only exception for its own bare host."""
    urls = list(_NATIVE_URL_LIKE.finditer(label))
    if not urls:
        return True
    from .pilot_source_card_v2 import native_https_link_proven_v2

    if not native_https_link_proven_v2(target):
        return False
    host = (urlsplit(target).hostname or "").lower().removeprefix("www.")
    return all(match[0].lower().removeprefix("www.") == host for match in urls)


def classify_external(target: str) -> bool:
    from .applicability_v5_3 import classify_action
    return classify_action("openUrl", target) == "external_semantic"
