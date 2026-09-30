"""Final content review motivated by observed omissions in repaired letters."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from ir_training.data.archive_recovery import (
    ACTION_LABEL_RE,
    BUTTON_BINDING_RE,
    _normalized_phrase,
    _resolve_state_path,
    _strings,
    _words,
)
from ir_training.data.archive_refinement import plain_source, repair_exact_text

POLICY = "messages-archive-final-review-v7-20260913"


def repair_missing_button_labels(source: str, graph: dict[str, Any]):
    """Only restore a missing requested label on one existing unambiguous link."""
    fixed = deepcopy(graph)
    declarations = defaultdict(set)
    for match in BUTTON_BINDING_RE.finditer(source):
        declarations[match["token"]].add(match["label"].strip())
    present = {
        _normalized_phrase(str((element.get("props") or {}).get("label") or ""))
        for element in fixed["elements"].values()
        if element.get("type") == "Button"
    }
    requested = {_normalized_phrase(label) for label in ACTION_LABEL_RE.findall(source)}
    source_phrase = _normalized_phrase(source)
    eligible = defaultdict(list)
    for key, element in fixed["elements"].items():
        if element.get("type") != "Button":
            continue
        label = _normalized_phrase(str((element.get("props") or {}).get("label") or ""))
        if label in requested or (label and label in source_phrase):
            continue
        events = list((element.get("on") or {}).values())
        if len(events) != 1 or not isinstance(events[0], dict):
            continue
        event = events[0]
        if str(event.get("action") or "").casefold() != "openurl":
            continue
        url = (event.get("params") or {}).get("url")
        if isinstance(url, str):
            eligible[url].append(key)
    count = 0
    for url, keys in eligible.items():
        if len(keys) != 1 or len(declarations.get(url, ())) != 1:
            continue
        desired = next(iter(declarations[url]))
        if _normalized_phrase(desired) in present:
            continue
        fixed["elements"][keys[0]]["props"]["label"] = desired
        present.add(_normalized_phrase(desired))
        count += 1
    return fixed, {"source_proven_missing_button_label": count} if count else {}


def repair_final_text(source: str, graph: dict[str, Any]):
    # Retain v6's source-proven prose corrections, but restore every Button
    # before applying the stricter missing-label rule. This prevents renaming
    # a source/citation button when its requested action already exists.
    fixed, metrics = repair_exact_text(source, graph)
    metrics.pop("source_proven_existing_button_label", None)
    for key, element in graph["elements"].items():
        if element.get("type") == "Button":
            original_label = (element.get("props") or {}).get("label")
            label = original_label
            if isinstance(label, str):
                clean = label.replace('\\"', '"').replace("\\'", "'")
                if len(clean) >= 24 and plain_source(clean) in plain_source(source):
                    label = clean
            fixed["elements"][key]["props"]["label"] = label
    fixed, labels = repair_missing_button_labels(source, fixed)
    return fixed, dict(Counter(metrics) + Counter(labels))


def bound_content_strings(graph: dict[str, Any]) -> list[str]:
    """Count props and referenced state, not unrelated unused state roots.

    Bound table rows are retained whole because native table routes can consume
    implicit columns. This remains a content screen, not a renderer simulator.
    """
    values = []
    for element in graph["elements"].values():
        props = element.get("props") or {}
        values.extend(_strings(props))
        for text in _strings(props):
            if text.startswith("/"):
                bound = _resolve_state_path(graph.get("state", {}), text)
                if bound is not None:
                    values.extend(_strings(bound))
    return values


def paragraph_gaps(source: str, graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Flag long prose blocks whose vocabulary is largely absent from the UI.

    The threshold is conservative and can flag paraphrases. It is deliberately
    separate from factual correctness and does not authorize synthesizing text.
    """
    visible = "\n".join(bound_content_strings(graph))
    target_words = _words(visible)
    source = re.sub(r"```[^\n]*\n[\s\S]*?```", "", source)
    gaps = []
    for index, block in enumerate(re.split(r"\n\s*\n", source), 1):
        lines = []
        for line in block.splitlines():
            line = line.strip()
            if re.match(
                r"^(?:#{1,6}\s|Media\s*:|Action\s*:|Sources?\s*:|Quick Actions|Chart Title\s*:|[XY]-Axis\s*:)",
                line,
                re.IGNORECASE,
            ):
                continue
            if "|" in line or re.match(
                r"^(?:[-*]\s*)?\[?Button\s*:", line, re.IGNORECASE
            ):
                continue
            lines.append(line)
        prose = " ".join(lines)
        words = _words(prose)
        missing = words - target_words
        if (
            len(prose) >= 120
            and len(words) >= 16
            and len(missing) >= 8
            and len(words & target_words) / len(words) < 0.5
        ):
            gaps.append(
                {
                    "block": index,
                    "source_chars": len(prose),
                    "distinct_words": len(words),
                    "missing_word_count": len(missing),
                    "word_recall": round(len(words & target_words) / len(words), 4),
                    "source_excerpt": prose[:200],
                }
            )
    return gaps
