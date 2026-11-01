"""Source-owned action/source-link occurrences for the opt-in pilot only."""
from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from . import _core
from .list_evidence_v5_4 import source_heading
from .matching_v5_3 import assignment_to_mapping, certified_assignment_v5_3
from .metrics_v5_2 import f_score
from .metrics_v5_3 import _action_match_score, _canonical_alias

SOURCE_PREFIX = "pilot_v2_source_link:"
ACTION_PREFIX = "pilot_v2_action:"
_EXPLICIT_ACTION = re.compile(
    r"(?i)^Action:\s*\[\s*(?:Button\s*:\s*)?([^\]]+?)\s*\]\s*"
    r"(?:<(https://[^\s<>]+)>|(https://[^\s<>]+|\{\{u\d+\}\}|\[(?:[A-Z_]+_)?URL_\d+\]))\s*$"
)
_LABELED_SOURCE = re.compile(r"^(.+?):\s*<(https://[^\s<>]+)>\s*$")


def source_action_occurrences(source: str) -> list[dict[str, Any]]:
    """Complete explicit Action lines, including angle-wrapped destinations."""
    found = []
    fence = None
    offset = 0
    for line in source.splitlines(keepends=True):
        raw = line.strip()
        start, end = offset, offset + len(line)
        offset = end
        if raw.startswith(("```", "~~~")):
            marker = raw[:3]
            fence = None if fence == marker else marker if fence is None else fence
            continue
        if fence is not None:
            continue
        match = _EXPLICIT_ACTION.fullmatch(raw)
        if match is not None:
            label, target = _core.normalize_markdown(match[1]), (match[2] or match[3]).strip()
            if label and target:
                found.append({"id": f"{ACTION_PREFIX}{start}:{end}", "label": label,
                              "target": target, "action_type": "openUrl", "required": True,
                              "minimum_count": 1, "_span": (start, end)})
    return found


def source_link_occurrences(source: str) -> list[dict[str, Any]]:
    """Explicit source-heading Markdown or label-colon-angle link lines."""
    found = []
    in_sources = False
    fence = None
    offset = 0
    heading_span = None
    for line in source.splitlines(keepends=True):
        raw = line.strip()
        start, end = offset, offset + len(line)
        offset = end
        if raw.startswith(("```", "~~~")):
            marker = raw[:3]
            fence = None if fence == marker else marker if fence is None else fence
            in_sources = False
            heading_span = None
            continue
        if fence is not None:
            continue
        if source_heading(raw):
            in_sources = True
            heading_span = (start, end)
            continue
        inline = re.match(r"(?i)^(?:source|reference|citation)\s*:\s*(.+)$", raw)
        body = inline[1] if inline else re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", raw)
        if not in_sources and inline is None:
            continue
        match = _core._MARKDOWN_LINK_RE.fullmatch(body) or _LABELED_SOURCE.fullmatch(body)
        if match is None:
            if raw:
                in_sources = False
                heading_span = None
            continue
        label, target = _core.normalize_markdown(match[1]), match[2].strip()
        if label and target:
            found.append({"id": f"{SOURCE_PREFIX}{start}:{end}", "label": label,
                          "target": target, "action_type": "openUrl", "required": True,
                          "minimum_count": 1, "_span": (start, end),
                          "_heading_span": heading_span if inline is None else None})
    return found


def source_link_owned_spans(source: str) -> set[tuple[int, int]]:
    return {span for item in source_link_occurrences(source)
            for span in (item["_span"], item["_heading_span"]) if span is not None}


def source_contract_v2(source: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(contract))
    links = source_link_occurrences(source)
    source_actions = source_action_occurrences(source)
    claimed_actions: set[int] = set()
    # Preserve every existing explicit action requirement, adding role identity.
    actions = []
    for index, action in enumerate(result.get("actions") or []):
        item = dict(action)
        matching = [position for position, occurrence in enumerate(source_actions)
                    if occurrence["label"] == item.get("label")
                    and _core.normalize_url(occurrence["target"]) == _core.normalize_url(item.get("target"))]
        raw_targets = {source_actions[position]["target"] for position in matching}
        identity = str(item.get("id") or index)
        if len(raw_targets) == 1:
            item["id"] = ACTION_PREFIX + identity
            item["target"] = next(iter(raw_targets))
            count = max(0, int(item.get("minimum_count", 1) or 0))
            claimed_actions.update([position for position in matching if position not in claimed_actions][:count])
        else:
            # The legacy extractor may collapse fragments or unsupported
            # source syntax. Never reconstruct a destination by guessing.
            item["id"] = ACTION_PREFIX + "unproven:" + identity
        actions.append(item)
    actions.extend({key: value for key, value in item.items() if not key.startswith("_")}
                   for position, item in enumerate(source_actions) if position not in claimed_actions)
    targets = {_core.normalize_url(item["target"]) for item in [*links, *source_actions]}
    proven_spans = {item["_span"] for item in [*links, *source_actions]}
    url_occurrences = list(_core._URL_RE.finditer(source))
    def owned_url(match):
        return any(start <= match.start() and match.end() <= end for start, end in proven_spans)
    def fully_owned_url(raw):
        occurrences = [match for match in url_occurrences if _core.normalize_url(match[0]) == _core.normalize_url(raw)]
        return bool(occurrences) and all(owned_url(match) for match in occurrences)
    # Replace only the extractor's loose URL-only entries. Rich persisted
    # requirements retain their original count/label/component constraints.
    result["source_links"] = [item for item in result.get("source_links") or []
                              if not (isinstance(item, Mapping) and set(item) == {"url"}
                                      and _core.normalize_url(item["url"]) in targets
                                      and fully_owned_url(item["url"]))]
    # The legacy extractor can already have dropped a loose same-base URL due
    # to an explicit Action at another fragment. Restore an unresolved required
    # occurrence rather than silently crediting it to the proven Action.
    remaining_targets = {_core.normalize_url(item.get("url") or item.get("target") or "")
                         for item in result["source_links"] if isinstance(item, Mapping)}
    result["source_links"].extend(
        {"url": match[0]} for match in url_occurrences
        if _core.normalize_url(match[0]) in targets - remaining_targets and not owned_url(match)
    )
    result["actions"] = [*actions, *({key: value for key, value in item.items() if not key.startswith("_")} for item in links)]
    # Source-heading/link lines are metadata owned by these exact requirements,
    # not duplicate prose. Remove only the concrete proven occurrences.
    units = list(result.get("content_units") or [])
    headings = list(result.get("headings") or [])
    owned_spans = source_link_owned_spans(source) | {item["_span"] for item in source_actions}
    for start, end in sorted(owned_spans):
        line = source[start:end].strip()
        if line in units:
            units.remove(line)
        if source_heading(line):
            title = re.sub(r"^#{1,6}\s*", "", line).rstrip(":").strip()
            if title in headings:
                headings.remove(title)
    result["content_units"] = units
    result["headings"] = headings
    return result


def restore_bound_action_targets(base, contract):
    """Undo legacy URL normalization only from source-owned contract entries.

    SourceContract conversion strips URL fragments. The immutable source hash,
    role ID and original target retained in this profile's contract bind the
    restored bytes; matching indices are intentionally not used by this profile.
    """
    targets = {item["id"]: item["target"] for item in contract.get("actions", [])
               if isinstance(item, Mapping) and item.get("id") and isinstance(item.get("target"), str)}
    def restore(item):
        return replace(item, url=targets[item.id]) if item.id in targets else item
    return replace(base,
                   expected_actions=tuple(restore(item) for item in base.expected_actions),
                   source_contract=replace(base.source_contract,
                       explicit_actions=[restore(item) for item in base.source_contract.explicit_actions]))


def _role(item: Any, *, expected: bool) -> str:
    if expected:
        if item.id.startswith(SOURCE_PREFIX):
            return "source_link"
        if item.id.startswith(ACTION_PREFIX):
            return "action"
        return ""
    return "source_link" if item.source == "source_link" else "action"


def _label(value: str) -> str:
    return " ".join(value.split())


def action_match_v2(expected: Any, actual: Any, aliases: Any) -> float:
    role = _role(expected, expected=True)
    if role:
        if expected.id.startswith(ACTION_PREFIX + "unproven:"):
            return 0.0
        if role != _role(actual, expected=False) or _label(expected.label) != _label(actual.label):
            return 0.0
        if expected.action_type != actual.action_type:
            return 0.0
        if expected.expected_component_id and expected.expected_component_id != actual.element_id:
            return 0.0
        # These are restored source/renderer target bytes. Path, query and
        # fragment are destination semantics, not URL-normalization noise.
        return float(expected.url.strip() == actual.url.strip())
    if expected.action_type == "openUrl":
        # No certified raw source occurrence/role: retain a conservative hold.
        return 0.0
    return _action_match_score(expected, actual, aliases)


def action_fidelity_v2(expected, actual, *, aliases, expected_exact_index=None,
                       exact_dense_limit, max_edges, top_k,
                       max_hungarian_work=50_000_000, max_sparse_relaxations=50_000_000):
    expanded = [item for item in expected if item.required and item.minimum_count > 0
                for _ in range(item.minimum_count)]
    if not expanded:
        return None, {"required_count": 0, "matched_required_count": 0,
                      "ignored_candidate_count": len(actual), "matching": {}}
    # Disable legacy exact-index shortcuts: they do not contain role identity.
    assignment = certified_assignment_v5_3(
        len(expanded), len(actual),
        score=lambda left, right: action_match_v2(expanded[left], actual[right], aliases),
        cheap_score=lambda left, right: action_match_v2(expanded[left], actual[right], aliases),
        expected_exact_key=lambda index: None, actual_exact_key=lambda index: None,
        expected_block_keys=lambda index: (("target", _canonical_alias(expanded[index].url, aliases)),),
        actual_block_keys=lambda index: (("target", _canonical_alias(actual[index].url, aliases)),),
        expected_exact_index=None, max_edges=max_edges, top_k=top_k,
        exact_dense_limit=exact_dense_limit, max_hungarian_work=max_hungarian_work,
        max_sparse_relaxations=max_sparse_relaxations,
    )
    mass = sum(edge.score for edge in assignment.matches)
    precision = mass / len(actual) if actual else 0.0
    recall = mass / len(expanded)
    matched = sum(edge.score >= .75 for edge in assignment.matches)
    return (f_score(precision, recall, 2.0) if assignment.certification.optimality_certified else None), {
        "required_count": len(expanded), "matched_required_count": matched,
        "precision": precision, "recall": recall, "role_coverage": matched / len(expanded),
        "matching": assignment_to_mapping(assignment), "prepared_exact_index_used": False,
        "pilot_source_role_matching": True,
        "matches": [{"expected_index": edge.expected_index, "output_index": edge.actual_index,
                     "score": edge.score} for edge in assignment.matches],
    }
