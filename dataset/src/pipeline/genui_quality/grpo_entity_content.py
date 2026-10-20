"""Opt-in, exact entity/field/action certification for QAT GRPO only.

This deliberately recognizes a narrow, unambiguous source grammar: bulleted
entity records separated by semicolons, unique labeled fields, and standalone
Markdown action links. Unsupported prose keeps the canonical scoring path.
Links are assigned to their own action channel only when their exact source
destination is also present in the resolved action contract. No URLs are
globally removed and no semantic paraphrase/bag-of-words certification occurs.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from . import _core
from .card_content_v5_4 import card_content_assignment_v5_4
from .matching_v5_1 import PreparedTextBlock, prepare_text_block
from .matching_v5_3 import group_exact_indices
from .metrics_v5_3 import content_exact_key_v5_3

GRPO_ENTITY_POLICY_VERSION = "1.0.0"
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$")
_LINK = re.compile(r"^\[([^\[\]\n]+)\]\((https?://[^\s()]+)\)$")
_INSTRUCTION = re.compile(r"(?i)^(?:step|phase|instruction|task|tip|note|warning|requirement)\b")
_MAX_RECORDS = 128
_MAX_FIELDS = 24


@dataclass(frozen=True)
class EntityRecord:
    source_index: int
    entity: str
    prose: str
    fixed_fields: tuple[tuple[str, ...], ...]
    labeled_fields: tuple[tuple[str, ...], ...]
    actions: tuple[tuple[str, str], ...]


def _record(raw: str, index: int, expected_urls: set[str]) -> EntityRecord | None:
    bullet = _BULLET.fullmatch(raw.strip())
    if bullet is None:
        return None
    segments = [part.strip() for part in bullet.group(1).split(";")]
    if not 2 <= len(segments) <= _MAX_FIELDS or not all(segments):
        return None
    prose, actions = [], []
    for segment in segments:
        link = _LINK.fullmatch(segment)
        if link:
            label, url = link.groups()
            url = _core.normalize_url(url)
            if url not in expected_urls:
                return None
            actions.append((_core.normalize_match_text(label), url))
        elif "[" in segment or "](" in segment or "http://" in segment or "https://" in segment:
            return None
        else:
            prose.append(segment)
    if len(prose) < 2 or segments[0] != prose[0]:
        return None
    head = prose[0].split(":", 1)
    entity = head[0].strip()
    if not entity or _INSTRUCTION.match(entity):
        return None
    fixed = [entity]
    if len(head) == 2:
        if not head[1].strip():
            return None
        fixed.append(head[1].strip())
    labels, fields = [], []
    for segment in prose[1:]:
        pair = segment.split(":", 1)
        if len(pair) != 2 or not all(part.strip() for part in pair):
            return None
        labels.append(_core.normalize_match_text(pair[0]))
        fields.append(segment)
    if len(labels) != len(set(labels)) or len(actions) != len(set(actions)):
        return None
    return EntityRecord(index, entity, "; ".join(prose),
                        tuple(prepare_text_block(value).tokens for value in fixed),
                        tuple(prepare_text_block(value).tokens for value in fields),
                        tuple(actions))


def prepare_grpo_entity_source(base: Any) -> tuple[Any, tuple[EntityRecord, ...]]:
    """Project only certified source link spans to the dedicated action channel."""
    units = base.applicability.generic_source_units
    if len(units) > _MAX_RECORDS:
        return base, ()
    urls = {_core.normalize_url(action.url) for action in base.expected_actions
            if action.required and action.minimum_count > 0}
    records = [record for index, raw in enumerate(units)
               if (record := _record(raw, index, urls)) is not None]
    entity_counts = Counter(_core.normalize_match_text(record.entity) for record in records)
    records = [record for record in records
               if entity_counts[_core.normalize_match_text(record.entity)] == 1]
    # Reused destinations across entities cannot prove entity ownership.
    action_counts = Counter(url for record in records for _, url in record.actions)
    records = [record for record in records
               if all(action_counts[url] == 1 for _, url in record.actions)]
    if not records:
        return base, ()
    mapped = {record.source_index: record.prose for record in records}
    units = tuple(mapped.get(index, raw) for index, raw in enumerate(units))
    blocks = tuple(prepare_text_block(value) for value in units)
    return replace(base, applicability=replace(base.applicability, generic_source_units=units),
                   content_blocks=blocks,
                   content_exact_index=group_exact_indices(
                       [content_exact_key_v5_3(block) for block in blocks])), tuple(records)


def certify_grpo_entity_output(
    records: Sequence[EntityRecord], evidence: Any,
) -> dict[str, Any]:
    """Bind each source occurrence to one visible Card and its exact actions.

Only labeled fields may change order. The entity and any unlabeled prefix stay
in position. Duplicate occurrences are never collapsed into one source record.
"""
    rows = [row for row in evidence.evidence_ownership
            if row.get("owner") == "generic_visible_content" and str(row.get("text") or "").strip()]
    cards: dict[str, list[int]] = defaultdict(list)
    nested = set()
    for index, row in enumerate(rows):
        instance = row.get("card_instance")
        if instance:
            cards[str(instance)].append(index)
            nested.update(value for value in row.get("card_ancestors", ()) if value != instance)
    actions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in evidence.action_taxonomy:
        if row.get("category") == "external_semantic":
            actions[str(row.get("card_instance") or "")].append((
                _core.normalize_match_text(row.get("label") or ""),
                _core.normalize_url(row.get("target") or ""),
            ))
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance, indices in cards.items():
        if instance in nested or indices != list(range(indices[0], indices[-1] + 1)):
            continue
        if any(rows[index].get("component_type") not in {"card", "text"} for index in indices):
            continue
        tokens = [prepare_text_block(str(rows[index]["text"])).tokens for index in indices]
        flattened = tuple(token for item in tokens for token in item)
        for record in records:
            fixed = len(record.fixed_fields)
            structured = (tuple(tokens[:fixed]) == record.fixed_fields
                          and Counter(tokens[fixed:]) == Counter(record.labeled_fields))
            whole = flattened == prepare_text_block(record.prose).tokens
            if not (whole or structured):
                continue
            candidates[record.source_index].append({
                "card_instance": instance, "output_indices": indices,
                "source_index": record.source_index, "structured": structured,
                "actions_exact": Counter(actions[instance]) == Counter(record.actions),
                "source_text": record.prose,
            })
    certified = [matches[0] for matches in candidates.values() if len(matches) == 1]
    linked = [record for record in records if record.actions]
    bound = {row["source_index"] for row in certified if row["actions_exact"]}
    return {
        "policy_version": GRPO_ENTITY_POLICY_VERSION,
        "eligible_record_count": len(records),
        "structured_record_count": sum(row["structured"] and row["actions_exact"] for row in certified),
        "linked_record_count": len(linked),
        "bound_linked_record_count": sum(record.source_index in bound for record in linked),
        "action_binding_fidelity": (
            sum(record.source_index in bound for record in linked) / len(linked) if linked else None
        ),
        "certified_records": certified,
    }


def grpo_entity_assignment(
    source: Sequence[PreparedTextBlock], ownership: Sequence[Mapping[str, Any]],
    generic_owners: set[str], detail: Mapping[str, Any],
) -> tuple[tuple[PreparedTextBlock, ...], dict[str, Any]]:
    # The certificate indexes generic ownership only; never reinterpret tables,
    # charts or semantic roles as record fields.
    if generic_owners != {"generic_visible_content"} and any(
        row.get("owner") in generic_owners - {"generic_visible_content"} for row in ownership
    ):
        return card_content_assignment_v5_4(source, ownership, generic_owners)
    rows = [dict(row) for row in ownership if row.get("owner") in generic_owners
            and str(row.get("text") or "").strip()]
    consumed = set()
    for record in detail.get("certified_records", ()):
        indices = record["output_indices"]
        rows[indices[0]]["text"] = record["source_text"]
        consumed.update(indices[1:])
    reduced = [row for index, row in enumerate(rows) if index not in consumed]
    blocks, grouping = card_content_assignment_v5_4(source, reduced, generic_owners)
    grouping["grpo_entity_certification"] = dict(detail)
    return blocks, grouping
