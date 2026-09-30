"""Lossless recovery helpers for source-bound messages-only Express archives.

This module repairs transport/representation defects only. It never invents
missing prose, state, components, actions, references, or target structure.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ir_training.data.express_preparation import serialize_checked
from ir_training.data.ir_targets import A2UI_EXPRESS_V1, materialize_completion_targets
from ir_training.data.url_preprocess import (
    _PLACEHOLDER_RE,
    _REFERENCE_RE,
    SOURCE_IDENTITY_BINDING,
    preprocess_training_urls,
    restore_url_placeholders,
)

WORDS_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
NUMBERS_RE = re.compile(r"(?<!\w)\d+(?:[.,:/-]\d+)*%?")
ACTION_LABEL_RE = re.compile(r"\[Button:\s*([^\]]+)\]", re.IGNORECASE)
BUTTON_BINDING_RE = re.compile(
    rf"\[Button:\s*(?P<label>[^\]]+)\]\s*<?(?P<token>{_PLACEHOLDER_RE.pattern})>?",
    re.IGNORECASE,
)
MARKDOWN_LINK_BINDING_RE = re.compile(
    rf"\[(?P<label>[^\]]+)\]\s*\(\s*<?(?P<token>{_PLACEHOLDER_RE.pattern})>?\s*\)",
    re.IGNORECASE,
)
STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
    ]
)
LAYOUT_TYPES = frozenset({"Column", "Row", "Stack", "Card", "List", "Divider"})
LINK_PLACEHOLDER_KINDS = frozenset({"ACTION_URL", "SOURCE_URL", "URL"})
IMAGE_PLACEHOLDER_KINDS = frozenset({"IMAGE_URL", "IMAGE_ASSET"})
ICON_PLACEHOLDER_KINDS = frozenset({"ICON_URL", "ICON_ASSET"})
MEDIA_PLACEHOLDER_KINDS = frozenset({"MEDIA_URL", "MEDIA_ASSET"})
MEDIA_TOKEN_KINDS = (
    IMAGE_PLACEHOLDER_KINDS | ICON_PLACEHOLDER_KINDS | MEDIA_PLACEHOLDER_KINDS
)
GENERIC_SOURCE_LABELS = frozenset(
    {
        "action",
        "actions",
        "icon",
        "icons",
        "image",
        "images",
        "link",
        "links",
        "media",
        "source",
        "sources",
        "url",
    }
)
REFERENCE_KEYS = frozenset(
    {
        "actionurl",
        "avatar",
        "bookingurl",
        "href",
        "icon",
        "image",
        "imagesrc",
        "link",
        "logo",
        "photo",
        "poster",
        "source",
        "sourceurl",
        "src",
        "thumbnail",
        "url",
    }
)
EMPTY_LAYOUT_TYPES = frozenset({"Stack", "Column", "Row", "Card", "List"})
MEANINGFUL_LAYOUT_PROPS = frozenset(
    {"data", "items", "source", "statePath", "subtitle", "text", "title", "value"}
)
LABEL_KEYS = (
    "label",
    "actionLabel",
    "title",
    "name",
    "movie",
    "model",
    "product",
    "chair",
    "airline",
    "provider",
    "service",
    "option",
    "item",
    "dayDate",
    "area",
    "text",
)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mojibake_score(value: str) -> int:
    """Count glyphs strongly associated with UTF-8 bytes decoded as CP437."""
    return sum(
        1
        for char in value
        if char in {"Ã", "â"}
        or "\u0370" <= char <= "\u03ff"
        or "\u2500" <= char <= "\u259f"
    )


@dataclass(frozen=True)
class TextRecovery:
    text: str
    applied: bool
    source_sha256: str
    recovered_sha256: str
    codec: str | None = None


def recover_cp437_utf8(value: str) -> TextRecovery:
    """Undo a whole-string CP437 rendering only when it is exact and clearer.

    The conversion is accepted only when every original code point maps to a
    CP437 byte sequence, those bytes are valid UTF-8, the inverse reproduces
    the original string exactly, and suspicious mojibake glyphs decrease.
    ASCII and genuine Unicode that cannot meet those conditions stay intact.
    """
    before = text_sha256(value)
    try:
        recovered = value.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return TextRecovery(value, False, before, before)
    if recovered == value or _mojibake_score(recovered) >= _mojibake_score(value):
        return TextRecovery(value, False, before, before)
    if recovered.encode("utf-8").decode("cp437") != value:
        raise ValueError("CP437/UTF-8 recovery failed its exact inverse")
    return TextRecovery(
        recovered, True, before, text_sha256(recovered), "cp437_bytes_as_utf8"
    )


def placeholder_tokens(value: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(value))


def _placeholder_kind(token: str) -> str:
    return token[1 : token.rfind("_")]


def _placeholder_index(token: str) -> int:
    return int(token[token.rfind("_") + 1 : -1])


def _compatible_placeholder_kinds(token: str) -> frozenset[str]:
    kind = _placeholder_kind(token)
    for group in (
        LINK_PLACEHOLDER_KINDS,
        IMAGE_PLACEHOLDER_KINDS,
        ICON_PLACEHOLDER_KINDS,
        MEDIA_PLACEHOLDER_KINDS,
    ):
        if kind in group:
            return group
    return frozenset({kind})


def contains_explicit_reference(value: str) -> bool:
    return bool(_REFERENCE_RE.search(value))


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)


def _words(value: str) -> set[str]:
    without_references = _REFERENCE_RE.sub(" ", _PLACEHOLDER_RE.sub(" ", value))
    return {
        item.casefold()
        for item in WORDS_RE.findall(without_references)
        if len(item) > 2
    } - STOP_WORDS


def _normalized_phrase(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    return " ".join(re.findall(r"[^\W_]+", value, re.UNICODE))


def _clean_label(value: str) -> str:
    value = re.sub(r"^[\s>*#\-+|]+|[\s<>()|]+$", "", value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    return value.strip(" []{}")


def _usable_label(value: str) -> str | None:
    cleaned = _clean_label(value)
    normalized = _normalized_phrase(cleaned)
    if not normalized or normalized in GENERIC_SOURCE_LABELS or len(normalized) > 120:
        return None
    return normalized


def _source_label_bindings(source: str) -> dict[str, set[str]]:
    """Extract only explicit, same-line label-to-placeholder declarations."""
    bindings: dict[str, set[str]] = defaultdict(set)
    for line in source.splitlines():
        explicitly_bound: set[str] = set()
        for pattern in (BUTTON_BINDING_RE, MARKDOWN_LINK_BINDING_RE):
            for match in pattern.finditer(line):
                label = _usable_label(match.group("label"))
                if label:
                    bindings[label].add(match.group("token"))
                    explicitly_bound.add(match.group("token"))
        for match in _PLACEHOLDER_RE.finditer(line):
            token = match.group(0)
            if token in explicitly_bound:
                continue
            prefix = line[: match.start()].rstrip(" <([={")
            candidates: list[str] = []
            if "|" in line:
                raw_cells = line.strip().strip("|").split("|")
                cells = [_clean_label(cell) for cell in raw_cells]
                token_cell = next(
                    (index for index, cell in enumerate(raw_cells) if token in cell),
                    None,
                )
                if token_cell is not None:
                    candidates.extend(cells[:token_cell][:1])
            if ":" in prefix:
                candidates.append(prefix.rsplit(":", 1)[0])
            for candidate in candidates:
                label = _usable_label(candidate)
                if label:
                    bindings[label].add(token)
    return dict(bindings)


def _context_labels(value: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    props = value.get("props") if isinstance(value.get("props"), dict) else {}
    sources = (value, props)
    for source in sources:
        for key in LABEL_KEYS:
            candidate = source.get(key)
            if not isinstance(candidate, str) or _PLACEHOLDER_RE.search(candidate):
                continue
            label = _usable_label(candidate)
            if label:
                labels.add(label)
    return labels


def _is_reference_key(key: str | None, element_type: str | None) -> bool:
    normalized = re.sub(r"[^a-z]", "", (key or "").casefold())
    if normalized in REFERENCE_KEYS or normalized.endswith(
        ("url", "href", "link", "image", "icon")
    ):
        return True
    return normalized == "name" and element_type == "Icon"


def _iter_string_tokens(value: Any):
    for item in _strings(value):
        yield from _PLACEHOLDER_RE.findall(item)


def _choose_source_token(
    token: str,
    labels: set[str],
    source_tokens: set[str],
    label_bindings: dict[str, set[str]],
) -> tuple[str, str] | None:
    compatible = _compatible_placeholder_kinds(token)
    labelled = {
        candidate
        for label in labels
        for candidate in label_bindings.get(label, set())
        if _placeholder_kind(candidate) in compatible
    }
    if len(labelled) == 1:
        chosen = next(iter(labelled))
        if chosen != token:
            return chosen, "placeholder_label_rebindings"
        return None
    if token in source_tokens:
        return None
    # URL-vs-ASSET namespace drift is representational when media kind and
    # numeric identity are both unchanged. Link roles are deliberately not
    # repaired by index because SOURCE_URL_1 and ACTION_URL_1 can differ.
    if compatible in {
        IMAGE_PLACEHOLDER_KINDS,
        ICON_PLACEHOLDER_KINDS,
        MEDIA_PLACEHOLDER_KINDS,
    }:
        same_index = {
            candidate
            for candidate in source_tokens
            if _placeholder_kind(candidate) in compatible
            and _placeholder_index(candidate) == _placeholder_index(token)
        }
        if len(same_index) == 1:
            return next(iter(same_index)), "placeholder_media_namespace_rebindings"
    return None


def _rebind_placeholders(
    source: str, graph: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    source_tokens = placeholder_tokens(source)
    label_bindings = _source_label_bindings(source)
    metrics: dict[str, int] = defaultdict(int)

    def visit(
        value: Any,
        labels: set[str],
        *,
        key: str | None = None,
        element_type: str | None = None,
    ) -> Any:
        if isinstance(value, dict):
            current_type = str(value.get("type") or element_type or "") or None
            current_labels = labels | _context_labels(value)
            return {
                item_key: visit(
                    item_value,
                    current_labels,
                    key=str(item_key),
                    element_type=current_type,
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [
                visit(item, labels, key=key, element_type=element_type)
                for item in value
            ]
        if isinstance(value, str) and _is_reference_key(key, element_type):
            match = _PLACEHOLDER_RE.fullmatch(value)
            if match:
                choice = _choose_source_token(
                    match.group(0), labels, source_tokens, label_bindings
                )
                if choice:
                    replacement, metric = choice
                    metrics[metric] += 1
                    return replacement
        return value

    rebound = visit(deepcopy(graph), set())

    # A legacy mask sometimes shifted an otherwise identical media token's
    # numeric suffix. Repair that identity only when the source and target
    # leave one unique unmatched token of the exact same concrete media kind.
    # Link roles are excluded because their destination cannot be inferred.
    target_tokens = set(_iter_string_tokens(rebound))
    unique_media_mapping: dict[str, str] = {}
    for kind in MEDIA_TOKEN_KINDS:
        target_only = sorted(
            token
            for token in target_tokens - source_tokens
            if _placeholder_kind(token) == kind
        )
        source_only = sorted(
            token
            for token in source_tokens - target_tokens
            if _placeholder_kind(token) == kind
        )
        if len(target_only) == 1 and len(source_only) == 1:
            unique_media_mapping[target_only[0]] = source_only[0]

    def apply_unique_media_mapping(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: apply_unique_media_mapping(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [apply_unique_media_mapping(item) for item in value]
        if isinstance(value, str):
            for target_token, source_token in unique_media_mapping.items():
                occurrences = value.count(target_token)
                if occurrences:
                    value = value.replace(target_token, source_token)
                    metrics["placeholder_unique_media_rebindings"] += occurrences
        return value

    return apply_unique_media_mapping(rebound), dict(metrics)


def _contains_unresolved_nonreference(
    value: Any,
    unresolved: set[str],
    *,
    key: str | None = None,
    element_type: str | None = None,
) -> bool:
    if isinstance(value, dict):
        current_type = str(value.get("type") or element_type or "") or None
        return any(
            _contains_unresolved_nonreference(
                item, unresolved, key=str(item_key), element_type=current_type
            )
            for item_key, item in value.items()
        )
    if isinstance(value, list):
        return any(
            _contains_unresolved_nonreference(
                item, unresolved, key=key, element_type=element_type
            )
            for item in value
        )
    if not isinstance(value, str):
        return False
    matches = set(_PLACEHOLDER_RE.findall(value)) & unresolved
    return bool(matches) and not (
        _PLACEHOLDER_RE.fullmatch(value) and _is_reference_key(key, element_type)
    )


def _resolve_state_path(state: Any, pointer: str) -> Any:
    current = state
    if not pointer.startswith("/"):
        return None
    for raw in pointer.split("/")[1:]:
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
            current = current[int(key)]
        else:
            return None
    return current


def _remove_element_ids(graph: dict[str, Any], element_ids: set[str]) -> set[str]:
    from pipeline.renderer_semantics import iter_renderer_references

    incoming_kinds: dict[str, set[str]] = defaultdict(set)
    for element in graph.get("elements", {}).values():
        for reference in iter_renderer_references(element):
            incoming_kinds[reference.target_id].add(reference.reference_kind)
    removable = {
        element_id
        for element_id in element_ids
        if incoming_kinds.get(element_id) == {"child"}
    }
    for element in graph.get("elements", {}).values():
        children = element.get("children")
        if isinstance(children, list):
            element["children"] = [
                child for child in children if child not in removable
            ]
    for element_id in removable:
        graph.get("elements", {}).pop(element_id, None)
    return removable


def _remove_empty_layout_branches(graph: dict[str, Any]) -> int:
    """Delete non-root containers with no child, event, or data-bearing prop."""
    removed = 0
    while True:
        empty = set()
        for element_id, element in graph.get("elements", {}).items():
            if (
                element_id == graph.get("root")
                or element.get("type") not in EMPTY_LAYOUT_TYPES
            ):
                continue
            children = element.get("children")
            props = (
                element.get("props") if isinstance(element.get("props"), dict) else {}
            )
            meaningful = any(
                key in MEANINGFUL_LAYOUT_PROPS and value not in (None, "", [], {})
                for key, value in props.items()
            )
            if not children and not meaningful and not element.get("on"):
                empty.add(element_id)
        if not empty:
            return removed
        actually_removed = _remove_element_ids(graph, empty)
        if not actually_removed:
            return removed
        removed += len(actually_removed)


def _prune_ungrounded_references(
    source: str, graph: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int], set[str]]:
    """Remove only exact reference-valued fields/elements absent from source."""
    source_tokens = placeholder_tokens(source)
    unresolved = set(_iter_string_tokens(graph)) - source_tokens
    requested_labels = {
        _normalized_phrase(label) for label in ACTION_LABEL_RE.findall(source)
    }
    generic_source_buttons = {
        element_id
        for element_id, element in graph.get("elements", {}).items()
        if element.get("type") == "Button"
        and _normalized_phrase(str((element.get("props") or {}).get("label") or ""))
        == "open source"
        and "open source" not in requested_labels
    }
    has_empty_layout = any(
        element_id != graph.get("root")
        and element.get("type") in EMPTY_LAYOUT_TYPES
        and not element.get("children")
        and not element.get("on")
        and not any(
            key in MEANINGFUL_LAYOUT_PROPS and value not in (None, "", [], {})
            for key, value in (element.get("props") or {}).items()
        )
        for element_id, element in graph.get("elements", {}).items()
    )
    if not unresolved and not generic_source_buttons and not has_empty_layout:
        return graph, {}, set()
    fixed = deepcopy(graph)
    metrics: dict[str, int] = defaultdict(int)

    generic_candidates: set[str] = set()
    ungrounded_candidates: set[str] = set()
    for element_id, element in fixed.get("elements", {}).items():
        element_tokens = set(_iter_string_tokens(element)) & unresolved
        if element_id in generic_source_buttons:
            generic_candidates.add(element_id)
        elif (
            element_tokens
            and element.get("type") in {"Button", "Image", "Icon"}
            and not _contains_unresolved_nonreference(
                element, unresolved, element_type=str(element.get("type") or "")
            )
        ):
            ungrounded_candidates.add(element_id)
    removable = generic_candidates | ungrounded_candidates
    if fixed.get("root") in removable:
        return fixed, dict(metrics), unresolved
    removed = _remove_element_ids(fixed, removable)
    metrics["generic_open_source_buttons_removed"] += len(removed & generic_candidates)
    metrics["ungrounded_reference_elements_removed"] += len(
        removed & ungrounded_candidates
    )
    empty_removed = _remove_empty_layout_branches(fixed)
    if empty_removed:
        metrics["empty_layout_elements_removed"] += empty_removed

    def prune(
        value: Any, *, key: str | None = None, element_type: str | None = None
    ) -> Any:
        if isinstance(value, dict):
            current_type = str(value.get("type") or element_type or "") or None
            output = {}
            for item_key, item in value.items():
                if (
                    isinstance(item, str)
                    and item in unresolved
                    and _is_reference_key(str(item_key), current_type)
                ):
                    metrics["ungrounded_reference_fields_removed"] += 1
                    continue
                if isinstance(item, list) and _is_reference_key(
                    str(item_key), current_type
                ):
                    filtered = [
                        entry
                        for entry in item
                        if not (isinstance(entry, str) and entry in unresolved)
                    ]
                    metrics["ungrounded_reference_fields_removed"] += len(item) - len(
                        filtered
                    )
                    output[item_key] = [
                        prune(entry, key=str(item_key), element_type=current_type)
                        for entry in filtered
                    ]
                else:
                    output[item_key] = prune(
                        item, key=str(item_key), element_type=current_type
                    )
            return output
        if isinstance(value, list):
            return [prune(item, key=key, element_type=element_type) for item in value]
        return value

    fixed = prune(fixed)
    for element in fixed.get("elements", {}).values():
        events = element.get("on")
        if not isinstance(events, dict):
            continue
        for event_name, event in list(events.items()):
            if (
                not isinstance(event, dict)
                or str(event.get("action") or "").casefold() != "openurl"
            ):
                continue
            params = event.get("params")
            if not isinstance(params, dict) or not any(
                key in params for key in ("url", "href", "link")
            ):
                del events[event_name]
                metrics["ungrounded_open_url_events_removed"] += 1
        if not events:
            element.pop("on", None)

    empty_tables: set[str] = set()
    for element_id, element in fixed.get("elements", {}).items():
        if element.get("type") != "Table":
            continue
        props = element.get("props") if isinstance(element.get("props"), dict) else {}
        rows = _resolve_state_path(
            fixed.get("state", {}), str(props.get("statePath") or "")
        )
        if (
            not isinstance(rows, list)
            or not rows
            or not all(isinstance(row, dict) for row in rows)
        ):
            continue
        present = {key for row in rows for key in row}
        columns = props.get("columns")
        if not isinstance(columns, list):
            continue
        retained = [
            column
            for column in columns
            if not isinstance(column, dict) or column.get("key") in present
        ]
        metrics["empty_table_columns_removed"] += len(columns) - len(retained)
        props["columns"] = retained
        column_keys = {
            str(column.get("key")) for column in retained if isinstance(column, dict)
        }
        for list_key in ("highlightColumns", "numericColumns", "secondaryColumns"):
            if isinstance(props.get(list_key), list):
                props[list_key] = [
                    item for item in props[list_key] if str(item) in column_keys
                ]
                if not props[list_key]:
                    props.pop(list_key, None)
        for scalar_key in ("primaryColumn",):
            if (
                props.get(scalar_key) is not None
                and str(props[scalar_key]) not in column_keys
            ):
                props.pop(scalar_key, None)
        if not retained:
            empty_tables.add(element_id)
    if fixed.get("root") not in empty_tables:
        removed_tables = _remove_element_ids(fixed, empty_tables)
        metrics["empty_tables_removed"] += len(removed_tables)
    empty_removed = _remove_empty_layout_branches(fixed)
    if empty_removed:
        metrics["empty_layout_elements_removed"] += empty_removed
    remaining = set(_iter_string_tokens(fixed)) - source_tokens
    return fixed, {key: value for key, value in metrics.items() if value}, remaining


def _join_midword_text_chunks(graph: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Join legacy 180-character Text chunks only at a proven mid-word edge."""
    if not any(
        element.get("type") == "Text"
        and isinstance((element.get("props") or {}).get("text"), str)
        and len(element["props"]["text"]) == 180
        for element in graph.get("elements", {}).values()
    ):
        return graph, 0
    fixed = deepcopy(graph)
    elements = fixed.get("elements", {})
    inbound: dict[str, int] = defaultdict(int)
    for element in elements.values():
        children = (
            element.get("children", [])
            if isinstance(element.get("children"), list)
            else []
        )
        for child in children:
            inbound[str(child)] += 1
    removed: set[str] = set()
    joins = 0
    for parent in list(elements.values()):
        children = parent.get("children")
        if not isinstance(children, list):
            continue
        output: list[str] = []
        index = 0
        while index < len(children):
            left_id = children[index]
            left = elements.get(left_id)
            if (
                index + 1 < len(children)
                and isinstance(left, dict)
                and left.get("type") == "Text"
            ):
                right_id = children[index + 1]
                right = elements.get(right_id)
                left_props = (
                    left.get("props") if isinstance(left.get("props"), dict) else {}
                )
                right_props = (
                    right.get("props")
                    if isinstance(right, dict) and isinstance(right.get("props"), dict)
                    else {}
                )
                left_text, right_text = left_props.get("text"), right_props.get("text")
                same_style = {k: v for k, v in left_props.items() if k != "text"} == {
                    k: v for k, v in right_props.items() if k != "text"
                }
                if (
                    isinstance(right, dict)
                    and right.get("type") == "Text"
                    and isinstance(left_text, str)
                    and isinstance(right_text, str)
                    and len(left_text) == 180
                    and left_text
                    and right_text
                    and left_text[-1].isalnum()
                    and right_text[0].isalnum()
                    and same_style
                    and inbound.get(str(left_id), 0) == 1
                    and inbound.get(str(right_id), 0) == 1
                ):
                    left_props["text"] = left_text + right_text
                    removed.add(right_id)
                    output.append(left_id)
                    joins += 1
                    index += 2
                    continue
            output.append(left_id)
            index += 1
        parent["children"] = output
    for element_id in removed:
        elements.pop(element_id, None)
    return fixed, joins


def _content_numbers(value: str) -> set[str]:
    value = _REFERENCE_RE.sub(" ", _PLACEHOLDER_RE.sub(" ", value))
    return set(NUMBERS_RE.findall(value))


def content_warning_reasons(
    source: str, target: str, graph: dict[str, Any]
) -> list[str]:
    """Return conservative unresolved-content warnings after representation repair."""
    reasons: list[str] = []
    types = {str(item.get("type") or "") for item in graph.get("elements", {}).values()}
    if not (types - LAYOUT_TYPES):
        reasons.append("layout_only")

    target_text = "\n".join(_strings(graph))
    requested_actions = [
        _normalized_phrase(label) for label in ACTION_LABEL_RE.findall(source)
    ]
    button_labels = []
    for element in graph.get("elements", {}).values():
        if element.get("type") != "Button":
            continue
        props = element.get("props") if isinstance(element.get("props"), dict) else {}
        button_labels.extend(
            _normalized_phrase(str(props[key]))
            for key in ("label", "text", "title")
            if props.get(key) is not None
        )
    if requested_actions and any(
        label and label not in button_labels for label in requested_actions
    ):
        reasons.append("missing_named_action")

    media_kinds = (
        IMAGE_PLACEHOLDER_KINDS | ICON_PLACEHOLDER_KINDS | MEDIA_PLACEHOLDER_KINDS
    )
    declared_media = {
        token
        for token in placeholder_tokens(source)
        if _placeholder_kind(token) in media_kinds
    }
    target_media = set(_iter_string_tokens(graph))
    if declared_media - target_media:
        reasons.append("missing_declared_media")

    source_words, target_words = _words(source), _words(target_text)
    recall = (
        len(source_words & target_words) / len(source_words) if source_words else 1.0
    )
    if recall < 0.5:
        reasons.append("lexical_recall_under_half")

    source_numbers, target_numbers = (
        _content_numbers(source),
        _content_numbers(target_text),
    )
    if (
        len(source_numbers) >= 3
        and len(source_numbers - target_numbers) / len(source_numbers) >= 0.5
    ):
        reasons.append("half_numeric_anchors_missing")
    return reasons


@dataclass(frozen=True)
class PairRecovery:
    accepted: bool
    reason: str
    source_text: str
    target_text: str
    graph: dict[str, Any] | None
    url_map: dict[str, dict[str, str]]
    transformations: tuple[str, ...]
    source_transcode: TextRecovery
    target_transcode: TextRecovery
    warnings: tuple[str, ...] = ()
    repair_metrics: tuple[tuple[str, int], ...] = ()


def recover_pair(source: str, target: str) -> PairRecovery:
    """Apply exact transport fixes and removal/rebinding grounded by the source."""
    source_fix, target_fix = recover_cp437_utf8(source), recover_cp437_utf8(target)
    transformations = []
    if source_fix.applied:
        transformations.append("source_cp437_utf8")
    if target_fix.applied:
        transformations.append("target_cp437_utf8")
    try:
        checked = serialize_checked(target_fix.text, "root-first")
    except (TypeError, ValueError, RecursionError) as exc:
        return PairRecovery(
            False,
            f"strict_target_error:{type(exc).__name__}",
            source_fix.text,
            target_fix.text,
            None,
            {},
            tuple(transformations),
            source_fix,
            target_fix,
        )
    graph = checked.graph
    source_tokens, target_tokens = (
        placeholder_tokens(source_fix.text),
        placeholder_tokens(target_fix.text),
    )
    explicit = contains_explicit_reference(source_fix.text) or any(
        contains_explicit_reference(item) for item in _strings(graph)
    )
    url_map: dict[str, dict[str, str]] = {}
    final_source, final_target, final_graph = source_fix.text, checked.text, graph

    if source_tokens or target_tokens:
        if explicit:
            return PairRecovery(
                False,
                "mixed_symbolic_and_literal_references",
                final_source,
                final_target,
                final_graph,
                {},
                tuple(transformations),
                source_fix,
                target_fix,
            )
    elif explicit:
        try:
            masked = preprocess_training_urls(
                final_source, graph, binding_policy=SOURCE_IDENTITY_BINDING
            )
            if (
                restore_url_placeholders(masked.response_text, masked.url_map)
                != final_source
            ):
                raise ValueError("source restoration mismatch")
            if (
                restore_url_placeholders(masked.canonical_graph, masked.url_map)
                != graph
            ):
                raise ValueError("graph restoration mismatch")
            encoded = materialize_completion_targets(masked.canonical_graph)[
                A2UI_EXPRESS_V1
            ]
            normalized = serialize_checked(encoded, "root-first")
        except (KeyError, TypeError, ValueError, RecursionError) as exc:
            return PairRecovery(
                False,
                f"reference_normalization_error:{type(exc).__name__}",
                final_source,
                final_target,
                final_graph,
                {},
                tuple(transformations),
                source_fix,
                target_fix,
            )
        final_source, final_target, final_graph = (
            masked.response_text,
            normalized.text,
            normalized.graph,
        )
        url_map = masked.url_map
        if url_map:
            transformations.append("source_identity_url_normalization")

    repair_metrics: dict[str, int] = defaultdict(int)
    final_graph, binding_metrics = _rebind_placeholders(final_source, final_graph)
    for key, value in binding_metrics.items():
        repair_metrics[key] += value
    final_graph, prune_metrics, unresolved = _prune_ungrounded_references(
        final_source, final_graph
    )
    for key, value in prune_metrics.items():
        repair_metrics[key] += value
    if unresolved:
        return PairRecovery(
            False,
            "unresolved_target_reference",
            final_source,
            final_target,
            final_graph,
            url_map,
            tuple(transformations),
            source_fix,
            target_fix,
            repair_metrics=tuple(sorted(repair_metrics.items())),
        )
    final_graph, joins = _join_midword_text_chunks(final_graph)
    if joins:
        repair_metrics["midword_text_boundary_joins"] += joins
    for metric in (
        "placeholder_label_rebindings",
        "placeholder_media_namespace_rebindings",
        "placeholder_unique_media_rebindings",
        "generic_open_source_buttons_removed",
        "ungrounded_reference_elements_removed",
        "ungrounded_reference_fields_removed",
        "ungrounded_open_url_events_removed",
        "empty_table_columns_removed",
        "empty_tables_removed",
        "empty_layout_elements_removed",
        "midword_text_boundary_joins",
    ):
        if repair_metrics.get(metric):
            transformations.append(metric)
    if repair_metrics:
        try:
            encoded = materialize_completion_targets(final_graph)[A2UI_EXPRESS_V1]
            normalized = serialize_checked(encoded, "root-first")
        except (KeyError, TypeError, ValueError, RecursionError) as exc:
            return PairRecovery(
                False,
                f"post_repair_target_error:{type(exc).__name__}",
                final_source,
                final_target,
                final_graph,
                url_map,
                tuple(transformations),
                source_fix,
                target_fix,
                repair_metrics=tuple(sorted(repair_metrics.items())),
            )
        final_target, final_graph = normalized.text, normalized.graph
    if not placeholder_tokens(final_target) <= placeholder_tokens(final_source):
        return PairRecovery(
            False,
            "target_reference_absent_source",
            final_source,
            final_target,
            final_graph,
            url_map,
            tuple(transformations),
            source_fix,
            target_fix,
            repair_metrics=tuple(sorted(repair_metrics.items())),
        )
    if url_map:
        used_tokens = placeholder_tokens(final_source) | placeholder_tokens(
            final_target
        )
        dropped = len(url_map) - sum(token in used_tokens for token in url_map)
        url_map = {
            token: entry for token, entry in url_map.items() if token in used_tokens
        }
        if dropped:
            repair_metrics["unused_target_url_map_entries_removed"] += dropped
            if "unused_target_url_map_entries_removed" not in transformations:
                transformations.append("unused_target_url_map_entries_removed")

    warnings = tuple(content_warning_reasons(final_source, final_target, final_graph))
    if warnings:
        return PairRecovery(
            False,
            "unresolved_content_warning",
            final_source,
            final_target,
            final_graph,
            url_map,
            tuple(transformations),
            source_fix,
            target_fix,
            warnings,
            tuple(sorted(repair_metrics.items())),
        )
    status = "repaired" if transformations else "kept_unchanged"
    return PairRecovery(
        True,
        status,
        final_source,
        final_target,
        final_graph,
        url_map,
        tuple(transformations),
        source_fix,
        target_fix,
        repair_metrics=tuple(sorted(repair_metrics.items())),
    )


__all__ = [
    "PairRecovery",
    "TextRecovery",
    "content_warning_reasons",
    "placeholder_tokens",
    "recover_cp437_utf8",
    "recover_pair",
    "text_sha256",
]
