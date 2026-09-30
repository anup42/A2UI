"""Conservative second review and deterministic split tools for archive copies.

This module does not call a generator. Corrections must be supported by the
source or an exact representation equivalence; missing panels and actions fail.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from ir_training.data.archive_recovery import (
    ACTION_LABEL_RE,
    BUTTON_BINDING_RE,
    EMPTY_LAYOUT_TYPES,
    MEANINGFUL_LAYOUT_PROPS,
    NUMBERS_RE,
    _content_numbers,
    _normalized_phrase,
    _strings,
    content_warning_reasons,
    recover_cp437_utf8,
    text_sha256,
)
from ir_training.data.url_preprocess import _PLACEHOLDER_RE, _REFERENCE_RE

POLICY = "messages-archive-refinement-v6-20260913"
WORD_RE = re.compile(r"\w+", re.UNICODE)
GROUPED_NUMBER_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])")


def source_signature(source: str) -> str:
    """Reference-insensitive, codec/Unicode/case/whitespace normalized source.

    References are stripped for family matching only; training text is intact.
    Their literal destinations therefore cannot disguise benchmark membership.
    """
    source = recover_cp437_utf8(source).text
    source = _PLACEHOLDER_RE.sub(" reference ", source)
    source = _REFERENCE_RE.sub(" reference ", source)
    return " ".join(WORD_RE.findall(unicodedata.normalize("NFKC", source).casefold()))


def plain_source(source: str) -> str:
    return " ".join(source.replace("**", "").replace("__", "").split())


def numeric_anchors(source: str) -> set[str]:
    cleaned = _REFERENCE_RE.sub(" ", _PLACEHOLDER_RE.sub(" ", source))
    # Only conventional groups of exactly three digits are equivalent. A comma
    # decimal, a date, a fraction, units and a percent marker are not discarded.
    cleaned = GROUPED_NUMBER_RE.sub(lambda match: match[0].replace(",", ""), cleaned)
    return set(NUMBERS_RE.findall(cleaned))


def repair_exact_text(
    source: str, graph: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    fixed = deepcopy(graph)
    plain = plain_source(source)
    metrics: Counter[str] = Counter()

    def visit(value: Any, key: str = "", kind: str = "") -> Any:
        if isinstance(value, dict):
            next_kind = str(value.get("type") or kind)
            return {k: visit(v, str(k), next_kind) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(item, key, kind) for item in value]
        if not isinstance(value, str) or kind in {"CodeBlock", "ConsoleLog", "Formula"}:
            return value
        if key not in {
            "text",
            "label",
            "title",
            "subtitle",
            "description",
            "body",
            "alt",
        }:
            return value
        unescaped = value.replace('\\"', '"').replace("\\'", "'")
        if (
            unescaped != value
            and len(unescaped) >= 24
            and plain_source(unescaped) in plain
        ):
            value = unescaped
            metrics["source_proven_literal_quote_unescape"] += 1
        if (
            kind == "Text"
            and key == "text"
            and len(value) == 180
            and value[-1].isalnum()
        ):
            needle = plain_source(value)
            hits = [match.start() for match in re.finditer(re.escape(needle), plain)]
            if len(hits) == 1:
                end = hits[0] + len(needle)
                suffix = re.match(r"[^\W_]+", plain[end:])
                if suffix and not (hits[0] and plain[hits[0] - 1].isalnum()):
                    value += suffix[0]
                    metrics["source_proven_clipped_word_completion"] += 1
        return value

    fixed = visit(fixed)
    # Existing buttons can recover their requested label from one exact URL
    # declaration. The URL and action themselves are never created or changed.
    declarations: dict[str, set[str]] = defaultdict(set)
    for match in BUTTON_BINDING_RE.finditer(source):
        declarations[match["token"]].add(match["label"].strip())
    requested = {_normalized_phrase(label) for label in ACTION_LABEL_RE.findall(source)}
    source_phrase = _normalized_phrase(source)
    for element in fixed.get("elements", {}).values():
        if element.get("type") != "Button":
            continue
        props = element.get("props") or {}
        old_label = _normalized_phrase(str(props.get("label") or ""))
        if old_label in requested or (old_label and old_label in source_phrase):
            continue
        actions = element.get("on") or {}
        events = list(actions.values()) if isinstance(actions, dict) else []
        if len(events) != 1 or not isinstance(events[0], dict):
            continue
        action = events[0]
        if str(action.get("action") or "").casefold() != "openurl":
            continue
        url = (action.get("params") or {}).get("url")
        if isinstance(url, str) and len(declarations.get(url, set())) == 1:
            props["label"] = next(iter(declarations[url]))
            metrics["source_proven_existing_button_label"] += 1
    return fixed, dict(metrics)


def review_warnings(
    source: str, target: str, graph: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, int]]:
    warnings = content_warning_reasons(source, target, graph)
    resolutions: list[str] = []
    target_text = "\n".join(_strings(graph))
    if "half_numeric_anchors_missing" in warnings:
        # Require every formerly missing numeric anchor to be represented after
        # normalization, rather than lowering the old 50 percent threshold.
        missing = _content_numbers(source) - _content_numbers(target_text)
        expected = set().union(*(numeric_anchors(value) for value in missing))
        if expected <= numeric_anchors(target_text):
            warnings.remove("half_numeric_anchors_missing")
            resolutions.append("numeric_grouping_false_positive_resolved")

    expected_urls: dict[str, set[str]] = defaultdict(set)
    for match in BUTTON_BINDING_RE.finditer(source):
        expected_urls[_normalized_phrase(match["label"])].add(match["token"])
    button_actions: dict[str, set[str]] = defaultdict(set)
    elements = graph.get("elements", {})
    for element in elements.values():
        if element.get("type") != "Button":
            continue
        label = _normalized_phrase(str((element.get("props") or {}).get("label") or ""))
        for event in (element.get("on") or {}).values():
            if (
                isinstance(event, dict)
                and str(event.get("action") or "").casefold() == "openurl"
            ):
                url = (event.get("params") or {}).get("url")
                if isinstance(url, str):
                    button_actions[label].add(url)
    if any(
        len(urls) == 1 and not urls <= button_actions[label]
        for label, urls in expected_urls.items()
    ):
        warnings.append("requested_action_binding_not_rendered")

    from ir_training.data.express_preparation import _api

    _, _, _, iter_renderer_references = _api()

    def empty_panel(element_id: str, seen: set[str]) -> bool:
        if element_id in seen:
            return False
        seen.add(element_id)
        element = elements[element_id]
        if element.get("type") not in EMPTY_LAYOUT_TYPES:
            return False
        if element.get("on") or element.get("repeat") or element.get("watch"):
            return False
        if any(
            key in MEANINGFUL_LAYOUT_PROPS and value not in (None, "", [], {})
            for key, value in (element.get("props") or {}).items()
        ):
            return False
        return all(
            empty_panel(edge.target_id, seen)
            for edge in iter_renderer_references(element)
        )

    if any(
        empty_panel(edge.target_id, set())
        for element in elements.values()
        if element.get("type") in {"Tabs", "Modal"}
        for edge in iter_renderer_references(element)
    ):
        warnings.append("empty_interactive_panel")

    features = Counter(str(element.get("type")) for element in elements.values())
    return sorted(set(warnings)), resolutions, dict(features)


class FamilyUnion:
    def __init__(self):
        self.parents: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parents.setdefault(value, value)
        root = value
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[value] != value:
            previous = self.parents[value]
            self.parents[value] = root
            value = previous
        return root

    def union(self, left: str, right: str) -> bool:
        roots = sorted({self.find(left), self.find(right)})
        if len(roots) == 1:
            return False
        self.parents[roots[1]] = roots[0]
        return True


class NearSourceIndex:
    """Bounded rare-trigram retrieval followed by full lexical comparison."""

    def __init__(self, sources: dict[str, str]):
        self.shingles: dict[str, set[tuple[str, ...]]] = {}
        self.lengths: dict[str, int] = {}
        self.anchors: dict[tuple[str, ...], set[str]] = defaultdict(set)
        tokenized = {key: value.split() for key, value in sources.items()}
        frequencies: Counter[str] = Counter()
        for words in tokenized.values():
            frequencies.update(set(words))
        for key, words in tokenized.items():
            all_shingles = list(zip(words, words[1:], words[2:]))
            self.shingles[key] = set(all_shingles)
            self.lengths[key] = len(words)
            selected: set[tuple[str, ...]] = set()
            for quarter in range(4):
                start, end = (
                    len(all_shingles) * quarter // 4,
                    len(all_shingles) * (quarter + 1) // 4,
                )
                ranked = sorted(
                    enumerate(all_shingles[start:end], start),
                    key=lambda item: (
                        -sum(
                            math.log((len(sources) + 1) / (1 + frequencies[word]))
                            for word in item[1]
                        ),
                        item[0],
                    ),
                )
                positions: list[int] = []
                for position, shingle in ranked:
                    if shingle in selected or any(
                        abs(position - prior) < 3 for prior in positions
                    ):
                        continue
                    selected.add(shingle)
                    positions.append(position)
                    if len(positions) == 6:
                        break
            for shingle in selected:
                self.anchors[shingle].add(key)

    def matches(self, source: str, *, containment: bool = False):
        words = source.split()
        shingles = set(zip(words, words[1:], words[2:]))
        hits: Counter[str] = Counter()
        for shingle in shingles:
            hits.update(self.anchors.get(shingle, ()))
        for key, count in sorted(hits.items()):
            if count < 2:
                continue
            ratio = min(len(words), self.lengths[key]) / max(
                1, len(words), self.lengths[key]
            )
            if ratio < (0.4 if containment else 0.65):
                continue
            other = self.shingles[key]
            shared = len(shingles & other)
            jaccard = shared / max(1, len(shingles | other))
            contained = shared / max(1, min(len(shingles), len(other)))
            if jaccard >= 0.60 or (
                containment
                and min(len(words), self.lengths[key]) >= 50
                and contained >= 0.90
            ):
                yield key, round(jaccard, 6), round(contained, 6)


def choose_validation(
    groups: dict[str, tuple[str, int]], fraction: float, seed: int
) -> set[str]:
    """Deterministic source-group selection stratified by shape and length."""
    strata: dict[str, list[str]] = defaultdict(list)
    for family, (shape, length) in groups.items():
        length_bin = (
            0 if length < 4000 else 1 if length < 8000 else 2 if length < 16000 else 3
        )
        strata[f"{shape}:{length_bin}"].append(family)
    selected: set[str] = set()
    for stratum, families in sorted(strata.items()):
        ranked = sorted(
            families, key=lambda value: text_sha256(f"{seed}:{stratum}:{value}")
        )
        n = min(len(ranked) - 1, round(len(ranked) * fraction))
        if len(ranked) >= 10:
            n = max(1, n)
        selected.update(ranked[:n])
    return selected


def component_shape(features: dict[str, int]) -> str:
    return (
        "+".join(
            kind
            for kind in (
                "Table",
                "Image",
                "Chart",
                "Tabs",
                "Modal",
                "EmailPreview",
                "CodeBlock",
                "Formula",
                "Video",
                "AudioPlayer",
            )
            if features.get(kind)
        )
        or "text_other"
    )
