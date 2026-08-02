"""Renderer-reference-aware hierarchy and economy helpers for v5.1."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import re
from typing import Any, Mapping

from ..renderer_semantics import iter_renderer_references
from . import _core


STRUCTURE_POLICY_VERSION = "5.1.0"


def renderer_parent_map(
    output: _core.OutputEvidence,
) -> dict[str, tuple[str, ...]]:
    parents: dict[str, list[str]] = defaultdict(list)
    for parent_id in sorted(output.reachable_ids):
        raw = output.elements.get(parent_id)
        if not isinstance(raw, Mapping):
            continue
        for reference in iter_renderer_references(raw):
            if (
                reference.target_id in output.reachable_ids
                and parent_id not in parents[reference.target_id]
            ):
                parents[reference.target_id].append(parent_id)
    return {
        child: tuple(values)
        for child, values in sorted(parents.items())
    }


def root_layout_utility_v5_1(
    spec: Mapping[str, Any],
    output: _core.OutputEvidence,
) -> float:
    root = spec.get("root")
    element = output.elements.get(str(root), {})
    return float(
        element.get("type")
        in {
            "Stack",
            "Column",
            "Row",
            "List",
            "Card",
            "ScrollView",
            "Grid",
            "Tabs",
            "Modal",
        }
    )


def fanout_utility_v5_1(output: _core.OutputEvidence) -> float:
    counts: list[int] = []
    for element_id in sorted(output.reachable_ids):
        raw = output.elements.get(element_id)
        if not isinstance(raw, Mapping):
            continue
        references = {
            reference.target_id
            for reference in iter_renderer_references(raw)
            if reference.target_id in output.reachable_ids
        }
        if raw.get("type") in _core.CONTAINER_TYPES or references:
            counts.append(len(references))
    if not counts:
        return 0.0
    return 1.0 - _core.safe_div(
        sum(count > 12 for count in counts),
        len(counts),
    )


def heading_grouping_utility_v5_1(
    output: _core.OutputEvidence,
) -> float | None:
    heading_ids: list[str] = []
    for element_id in sorted(output.reachable_ids):
        element = output.elements.get(element_id, {})
        props = (
            element.get("props", {})
            if isinstance(element.get("props"), Mapping)
            else {}
        )
        if (
            element.get("type") == "Text"
            and props.get("variant") in {"h1", "h2", "h3", "h4", "title", "heading"}
        ):
            heading_ids.append(element_id)
    if not heading_ids:
        return None
    parents = renderer_parent_map(output)
    grouped = 0
    for heading_id in heading_ids:
        good = False
        for parent_id in parents.get(heading_id, ()):
            parent = output.elements.get(parent_id, {})
            siblings = {
                reference.target_id
                for reference in iter_renderer_references(parent)
                if reference.target_id in output.reachable_ids
            }
            if any(sibling != heading_id for sibling in siblings):
                good = True
                break
        grouped += int(good)
    return _core.safe_div(grouped, len(heading_ids))


def wrapper_economy_v5_1(output: _core.OutputEvidence) -> float:
    candidates = 0
    redundant = 0
    for element_id in sorted(output.reachable_ids):
        element = output.elements.get(element_id, {})
        element_type = element.get("type")
        if element_type not in _core.LAYOUT_TYPES:
            continue
        children = [
            reference.target_id
            for reference in iter_renderer_references(element)
            if reference.target_id in output.reachable_ids
        ]
        if len(children) != 1:
            continue
        child = output.elements.get(children[0], {})
        child_type = child.get("type")
        candidates += 1
        props = (
            element.get("props", {})
            if isinstance(element.get("props"), Mapping)
            else {}
        )
        if (
            element_type in {"Stack", "Column", "Row", "List"}
            and child_type == element_type
        ):
            redundant += 1
        elif element_type == "Stack" and not props:
            redundant += 1
        # Card -> Stack is purposeful and intentionally exempt.
    return 1.0 - _core.safe_div(redundant, max(1, candidates))


_PATH_TOKEN_RE = re.compile(r"([^.[]+)|\[(\d+)\]")


def _set_reference_path(value: Any, path: str, replacement: str) -> None:
    tokens: list[str | int] = []
    for match in _PATH_TOKEN_RE.finditer(path):
        tokens.append(
            int(match.group(2))
            if match.group(2) is not None
            else str(match.group(1))
        )
    current = value
    for token in tokens[:-1]:
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                return
            current = current[token]
    if not tokens:
        return
    final = tokens[-1]
    if isinstance(final, int):
        if isinstance(current, list) and final < len(current):
            current[final] = replacement
    elif isinstance(current, dict) and final in current:
        current[final] = replacement


def canonical_reachable_payload_v5_1(
    spec: Mapping[str, Any],
    audit: _core.GraphAudit,
) -> dict[str, Any]:
    elements = (
        spec.get("elements")
        if isinstance(spec.get("elements"), Mapping)
        else {}
    )
    root = str(spec.get("root") or "")
    order: list[str] = []
    seen: set[str] = set()

    def visit(element_id: str) -> None:
        if element_id in seen or element_id not in audit.reachable_ids:
            return
        seen.add(element_id)
        order.append(element_id)
        raw = elements.get(element_id)
        if not isinstance(raw, Mapping):
            return
        for reference in iter_renderer_references(raw):
            visit(reference.target_id)

    if audit.root_exists:
        visit(root)
    for element_id in sorted(audit.reachable_ids - seen):
        visit(element_id)
    canonical_ids = {
        element_id: f"e{index}"
        for index, element_id in enumerate(order)
    }
    canonical_elements: list[dict[str, Any]] = []
    for element_id in order:
        raw = elements.get(element_id)
        copied = deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
        if isinstance(raw, Mapping):
            for reference in iter_renderer_references(raw):
                replacement = canonical_ids.get(reference.target_id, "@missing")
                _set_reference_path(copied, reference.source_path, replacement)
        canonical_elements.append(copied)
    return {
        "root": canonical_ids.get(root) if audit.root_exists else None,
        "state": deepcopy(spec.get("state", {})),
        "elements": canonical_elements,
    }


def canonical_reachable_payload_length_v5_1(
    spec: Mapping[str, Any],
    audit: _core.GraphAudit,
) -> int:
    return len(
        json.dumps(
            canonical_reachable_payload_v5_1(spec, audit),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def json_efficiency_utility_v5_1(
    spec: Mapping[str, Any],
    audit: _core.GraphAudit,
    source: _core.SourceContract,
    config: _core.RewardConfig,
) -> float:
    ratio = canonical_reachable_payload_length_v5_1(spec, audit) / max(
        1, len(source.raw_text)
    )
    return _core.low_is_good(
        ratio,
        config.good_json_to_source_ratio,
        config.bad_json_to_source_ratio,
    )


__all__ = [
    "STRUCTURE_POLICY_VERSION",
    "canonical_reachable_payload_length_v5_1",
    "canonical_reachable_payload_v5_1",
    "fanout_utility_v5_1",
    "heading_grouping_utility_v5_1",
    "json_efficiency_utility_v5_1",
    "renderer_parent_map",
    "root_layout_utility_v5_1",
    "wrapper_economy_v5_1",
]
