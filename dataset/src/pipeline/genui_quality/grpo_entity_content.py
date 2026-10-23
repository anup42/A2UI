"""Opt-in, exact entity/field/action certification for QAT GRPO only.

This deliberately recognizes bounded, unambiguous labeled source grammars.
Unsupported prose keeps the canonical scoring path, never an inferred pass.
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

GRPO_ENTITY_POLICY_VERSION = "2.0.0"
_LINK = re.compile(r"^\[([^\[\]\n]+)\]\((https?://[^\s()]+)\)$")
_INSTRUCTION = re.compile(r"(?i)^(?:step|phase|instruction|task|tip|note|warning|requirement)\b")
_MAX_FIELDS = 24


@dataclass(frozen=True)
class EntityRecord:
    source_index: int
    entity: str
    prose: str
    fixed_fields: tuple[tuple[str, ...], ...]
    labeled_fields: tuple[tuple[str, ...], ...]
    actions: tuple[tuple[str, str], ...]
    field_pairs: tuple[tuple[str, str], ...] = ()
    identity_label: str | None = None
    identity_value: str = ""
    source_indices: tuple[int, ...] = ()
    source_format: str = "semicolon"
    original_source_units: tuple[str, ...] = ()
    fixed_texts: tuple[str, ...] = ()


def exact_field_text(value: str) -> str:
    """Normalize layout whitespace only, retaining signs, units and qualifiers."""
    return " ".join(str(value).split())


def _record_segments(
    segments: Sequence[str], index: int, expected_urls: set[str], *,
    identity_labels: set[str] | frozenset[str],
    source_format: str, source_indices: tuple[int, ...],
    original_source_units: tuple[str, ...],
) -> EntityRecord | None:
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
    prefix = head[0].strip()
    if not prefix or _INSTRUCTION.match(prefix):
        return None
    identity_label = None
    entity = prefix
    fixed = [prefix]
    if len(head) == 2:
        if not head[1].strip():
            return None
        if _core.normalize_match_text(prefix) in identity_labels:
            identity_label, entity = prefix, head[1].strip()
            fixed = [prose[0]]
        else:
            fixed.append(head[1].strip())
    labels, fields, pairs = [], [], []
    for segment in prose[1:]:
        pair = segment.split(":", 1)
        if len(pair) != 2 or not all(part.strip() for part in pair):
            return None
        labels.append(_core.normalize_match_text(pair[0]))
        fields.append(segment)
        pairs.append(tuple(part.strip() for part in pair))
    if (len(labels) != len(set(labels)) or len(actions) != len(set(actions))
            or _core.normalize_match_text(identity_label or "") in labels):
        return None
    return EntityRecord(index, entity, "; ".join(prose),
                        tuple(prepare_text_block(value).tokens for value in fixed),
                        tuple(prepare_text_block(value).tokens for value in fields),
                        tuple(actions), tuple(pairs), identity_label, entity,
                        source_indices, source_format, original_source_units, tuple(fixed))


def extract_grpo_entity_records(
    source_units: Sequence[str], expected_urls: Sequence[str] = (), *, source_text: str = "",
) -> tuple[EntityRecord, ...]:
    """Pure bounded candidate API; recognition is not a fidelity/admission gate."""
    from .grpo_entity_formats import extract_records
    return extract_records(source_units, expected_urls, source_text=source_text)


def prepare_grpo_entity_source(base: Any) -> tuple[Any, tuple[EntityRecord, ...]]:
    """Prepare bounded source records, preserving all facts and action bindings."""
    original = base.applicability.generic_source_units
    records = extract_grpo_entity_records(
        original, tuple(action.url for action in base.expected_actions
                        if action.required and action.minimum_count > 0),
        source_text=base.source_text,
    )
    if not records:
        return base, ()
    mapped = {record.source_index: record for record in records}
    consumed = {i for record in records for i in record.source_indices[1:]}
    units, remapped = [], []
    for index, raw in enumerate(original):
        if index in consumed:
            continue
        if index in mapped:
            record = mapped[index]
            remapped.append(replace(record, source_index=len(units)))
            units.append(record.prose)
        else:
            units.append(raw)
    units = tuple(units)
    blocks = tuple(prepare_text_block(value) for value in units)
    return replace(base, applicability=replace(base.applicability, generic_source_units=units),
                   content_blocks=blocks,
                   content_exact_index=group_exact_indices(
                       [content_exact_key_v5_3(block) for block in blocks])), tuple(remapped)


def certify_grpo_entity_output(
    records: Sequence[EntityRecord], evidence: Any,
) -> dict[str, Any]:
    """Bind each source occurrence to one visible Card/Table and exact actions.

    Only labeled fields may change order. The entity and any unlabeled prefix
    stay in position. Duplicate occurrences never collapse to one record.
    """
    rows = [row for row in evidence.evidence_ownership
            if row.get("owner") == "generic_visible_content" and str(row.get("text") or "").strip()]
    cards: dict[str, list[int]] = defaultdict(list)
    nested = set()
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    records_by_index = {record.source_index: record for record in records}
    for index, row in enumerate(rows):
        if "grpo_table_source_index" in row:
            record = records_by_index[int(row["grpo_table_source_index"])]
            candidates[record.source_index].append({
                "table_instance": row["grpo_table_instance"], "output_indices": [index],
                "source_index": record.source_index, "structured": True,
                "actions_exact": True, "source_text": record.prose,
                "original_source_units": record.original_source_units,
                "source_format": record.source_format,
            })
        instance = row.get("card_instance")
        if instance:
            cards[str(instance)].append(index)
            nested.update(value for value in row.get("card_ancestors", ()) if value != instance)
    actions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in evidence.action_taxonomy:
        # Source recognition currently supports only explicit resolved links.
        # Any additional internal/unknown action in that Card also breaks the
        # certificate; it must not disappear from an exact-action comparison.
        if row.get("card_instance"):
            actions[str(row.get("card_instance") or "")].append((
                _core.normalize_match_text(row.get("label") or ""),
                _core.normalize_url(row.get("target") or ""),
            ))
    for instance, indices in cards.items():
        if instance in nested or indices != list(range(indices[0], indices[-1] + 1)):
            continue
        if any(rows[index].get("component_type") not in {"card", "text"} for index in indices):
            continue
        texts = [exact_field_text(rows[index]["text"]) for index in indices]
        for record in records:
            fixed = len(record.fixed_fields)
            structured = (tuple(texts[:fixed]) == tuple(map(exact_field_text, record.fixed_texts))
                          and Counter(texts[fixed:]) == Counter(
                              exact_field_text(f"{label}: {value}")
                              for label, value in record.field_pairs))
            whole = exact_field_text(" ".join(texts)) == exact_field_text(record.prose)
            if not (whole or structured):
                continue
            candidates[record.source_index].append({
                "card_instance": instance, "output_indices": indices,
                "source_index": record.source_index, "structured": structured,
                "actions_exact": Counter(actions[instance]) == Counter(record.actions),
                "source_text": record.prose,
                "original_source_units": record.original_source_units,
                "source_format": record.source_format,
            })
    certified = [matches[0] for matches in candidates.values() if len(matches) == 1]
    linked = [record for record in records if record.actions]
    bound = {row["source_index"] for row in certified if row["actions_exact"]}
    return {
        "policy_version": GRPO_ENTITY_POLICY_VERSION,
        "eligible_record_count": len(records),
        "bound_record_count": len(bound),
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
        if not record["actions_exact"]:
            continue
        indices = record["output_indices"]
        rows[indices[0]]["text"] = record["source_text"]
        consumed.update(indices[1:])
    reduced = [row for index, row in enumerate(rows) if index not in consumed]
    blocks, grouping = card_content_assignment_v5_4(source, reduced, generic_owners)
    grouping["grpo_entity_certification"] = dict(detail)
    return blocks, grouping


def summarize_grpo_entity_detail(detail: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize fresh trusted scorer evidence without repeating full scoring.

    Never use candidate-provided or stale stored certificates as input. This
    helper is not an independent fidelity/admission gate.
    """
    detail = dict(detail or {})
    eligible = int(detail.get("eligible_record_count", 0))
    bound = int(detail.get("bound_record_count", 0))
    structured = int(detail.get("structured_record_count", 0))
    issues = []
    if eligible and bound != eligible:
        issues.append("unbound_or_ambiguous_source_record")
    if eligible and structured != eligible:
        issues.append("records_not_structured")
    return {**detail, "policy_version": GRPO_ENTITY_POLICY_VERSION,
            "applicable": bool(eligible), "eligible_record_count": eligible,
            "bound_record_count": bound, "structured_record_count": structured,
            "binding_complete": bound == eligible if eligible else None,
            "issues": issues}


def audit_grpo_entity_representation(source_text: str, candidate: str) -> dict[str, Any]:
    """Read-only selection diagnostic; never waive canonical/release gates.

    Unsupported grammars return binding_complete=None, not success. A bound
    whole-record Card is faithful but unstructured, so extraction selection
    must also require structured_record_count == eligible_record_count.
    """
    from ._v5_4 import generation_reward_v5_4
    from .config_v5_4 import load_v5_4_reward_config

    if not isinstance(candidate, str):
        raise TypeError("candidate must be the canonical scoring text, not an IR graph")
    result = generation_reward_v5_4(candidate, source_text, config=replace(
        load_v5_4_reward_config(), grpo_entity_extraction=True))
    return summarize_grpo_entity_detail(result.evidence.get("grpo_entity_extraction"))
