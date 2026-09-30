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
from typing import Any, Mapping

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


def parse_source_links(text: str) -> list[tuple[str, str]]:
    """Supported explicit HTTPS/opaque subset of native source-link parsing."""
    markdown = list(_core._MARKDOWN_LINK_RE.finditer(text))
    if markdown:
        return list(dict.fromkeys((_core.normalize_markdown(m[1]), _core.normalize_url(m[2])) for m in markdown))
    result: list[tuple[str, str]] = []
    cursor = 0
    for match in _core._URL_RE.finditer(text):
        label = text[cursor:match.start()].strip(" -*:|[]()\t")
        label = re.sub(r"(?i)^(?:sources?|references?|citations?|links?)\s*:?\s*", "", label)
        label = re.sub(r"^\d+[.):\-]?\s*", "", label)
        result.append((_core.normalize_markdown(label), _core.normalize_url(match[0])))
        cursor = match.end()
    return list(dict.fromkeys(result))


def literal_list_evidence(value: Any, *, source_context: bool = False) -> tuple[list[str], list[tuple[str, str]]]:
    texts: list[str] = []
    links: list[tuple[str, str]] = []
    if not isinstance(value, list):
        return texts, links
    for item in value:
        if validate_list_items([item]) is not None:
            continue
        if isinstance(item, str):
            texts.append(item)
            if source_context:
                links.extend(parse_source_links(item))
            continue
        item_texts = list(dict.fromkeys(item[key] for key in LIST_TEXT_FIELDS if item.get(key, "").strip()))
        texts.extend(item_texts)
        label = next(iter(item_texts), "")
        explicit = list(dict.fromkeys(item[key] for key in LIST_LINK_FIELDS if item.get(key, "").strip()))
        for target in explicit:
            if classify_external(target):
                links.append((label, _core.normalize_url(target)))
        if source_context and not explicit:
            for text in item_texts:
                links.extend(parse_source_links(text))
    if source_context:
        # The source-section renderer collapses repeated URL buttons.
        by_url: dict[str, tuple[str, str]] = {}
        for item in links:
            by_url.setdefault(item[1], item)
        links = list(by_url.values())
        if links:
            # Native source sections display link labels, not the serialized
            # markdown/URL line. Non-link paragraphs require ordinary Text.
            texts = [label for label, _ in links if label]
    return texts, links


def classify_external(target: str) -> bool:
    from .applicability_v5_3 import classify_action
    return classify_action("openUrl", target) == "external_semantic"
