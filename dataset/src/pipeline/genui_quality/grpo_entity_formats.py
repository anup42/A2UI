"""Bounded labeled-source parsing and exact compact-Table projection.

These helpers are used only by the opt-in GRPO entity policy. A Table is
projected only when every visible row/cell has a unique exact source owner.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from . import _core
from .grpo_entity_content import EntityRecord, _record_segments, exact_field_text

_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$")
_COMMA_FIELD = re.compile(r",\s+(?=[^\W\d][\w /()-]{0,39}\s*:)", re.UNICODE)
_LABEL = re.compile(r"^[^\W\d][\w /()-]{0,39}$", re.UNICODE)
_IDENTITY_LABELS = frozenset({
    "name", "title", "entity", "hotel", "restaurant", "product", "item", "option",
    "flight", "route", "event", "day", "date", "camera", "service", "plan",
})
_MAX_RECORDS = 128
_MAX_FIELDS = 24


def _segments(text: str) -> tuple[list[str], str]:
    # Quoted separators may be part of one prose value, not field boundaries.
    # Leave these uncommon forms to the canonical path rather than guessing.
    if any(char in text for char in "\"'\u201c\u201d\u2018\u2019"):
        return [text], "unsupported_quoted_fields"
    if ";" in text:
        return [part.strip() for part in text.split(";")], "semicolon"
    if "|" in text:
        return [part.strip() for part in text.split("|")], "pipe"
    # Preserve commas in amounts and prose; only a short explicit label can
    # start another field. Standalone links require semicolon/pipe boundaries.
    if any(char in text for char in "()[]"):
        return [text], "unsupported_nested_comma"
    return [part.strip() for part in _COMMA_FIELD.split(text)], "labeled_comma"


def _parsed(segments: Sequence[str], index: int, urls: set[str], **kwargs: Any) -> EntityRecord | None:
    record = _record_segments(segments, index, urls, identity_labels=_IDENTITY_LABELS, **kwargs)
    if record and all(_LABEL.fullmatch(label) for label, _ in record.field_pairs):
        return record
    return None


def _heading_records(source_text: str, units: Sequence[str], urls: set[str]) -> list[EntityRecord]:
    """Only complete heading + labeled-bullet sections, with exact unit spans.

    Repeated identical spans, free prose, nested content, and role-owned units
    cannot be assigned by a global bag of field strings and are left alone.
    """
    lines = source_text.splitlines()
    records = []
    for line_index, raw in enumerate(lines):
        heading = _HEADING.fullmatch(raw)
        if not heading:
            continue
        # A heading alone is not entity evidence. Require an explicit known
        # identity label/type plus a name; ordinary Packing/Checklist sections
        # retain the unchanged canonical list path.
        identity = re.split(r"\s*:\s*|\s+", heading.group(1), maxsplit=1)
        if (len(identity) != 2 or not identity[1].strip()
                or identity[0].casefold() not in _IDENTITY_LABELS):
            continue
        body = []
        for offset in range(line_index + 1, len(lines)):
            line = lines[offset]
            if _HEADING.fullmatch(line):
                break
            if line.strip():
                body.append(line.strip())
                if len(body) >= _MAX_FIELDS:
                    break
        if not 2 <= len(body) < _MAX_FIELDS:
            continue
        bullets = [_BULLET.fullmatch(line) for line in body]
        if not all(bullets):
            continue
        starts = [i for i in range(len(units) - len(body) + 1)
                  if tuple(str(x).strip() for x in units[i:i + len(body)]) == tuple(body)]
        if len(starts) != 1:
            continue
        start = starts[0]
        record = _parsed(
            [heading.group(1), *(item.group(1) for item in bullets if item)], start, urls,
            source_format="heading_fields", source_indices=tuple(range(start, start + len(body))),
            original_source_units=(raw.strip(), *body),
        )
        if record:
            records.append(record)
    return records


def extract_records(
    source_units: Sequence[str], expected_urls: Sequence[str] = (), *, source_text: str = "",
) -> tuple[EntityRecord, ...]:
    if len(source_units) > _MAX_RECORDS:
        return ()
    urls = {_core.normalize_url(value) for value in expected_urls}
    records = []
    for index, raw in enumerate(source_units):
        bullet = _BULLET.fullmatch(str(raw).strip())
        if bullet:
            segments, kind = _segments(bullet.group(1))
            record = _parsed(segments, index, urls, source_format=kind,
                             source_indices=(index,), original_source_units=(str(raw),))
            if record:
                records.append(record)
    if source_text:
        records.extend(_heading_records(source_text, source_units, urls))
    entity_counts = Counter(_core.normalize_match_text(record.entity) for record in records)
    index_counts = Counter(index for record in records for index in record.source_indices)
    action_counts = Counter(url for record in records for _, url in record.actions)
    return tuple(sorted((record for record in records
                         if entity_counts[_core.normalize_match_text(record.entity)] == 1
                         and all(index_counts[i] == 1 for i in record.source_indices)
                         and all(action_counts[url] == 1 for _, url in record.actions)),
                        key=lambda record: record.source_index))


def _table_record_matches(record: EntityRecord, labels: Sequence[str], values: Sequence[str]) -> bool:
    if record.actions or len(record.fixed_fields) != 1:
        # Table evidence does not yet certify row-owned action label/target
        # bindings. Explicit Card actions keep the existing exact action path.
        return False
    identities = ((record.identity_label,) if record.identity_label else ("Name", "Entity", "Title"))
    for identity in identities:
        expected = [(identity, record.identity_value), *record.field_pairs]
        if len(labels) != len(expected):
            continue
        actual = {exact_field_text(label).casefold(): exact_field_text(value)
                  for label, value in zip(labels, values)}
        wanted = {exact_field_text(label).casefold(): exact_field_text(value)
                  for label, value in expected}
        if len(actual) == len(labels) and actual == wanted:
            return True
    return False


def project_grpo_entity_tables(records: Sequence[EntityRecord], evidence: Any) -> Any:
    """Project an entire, exactly bound Table into the generic matching view.

    Raw graph/output/Table/action/role evidence is retained. Only ownership's
    generic text view changes, expanding a shared header per certified row.
    Missing/extra/duplicate cells, ambiguous occurrences, actions, or unknown
    dynamic evidence prevent projection. Call only for non-table source prose.
    """
    if not records or not evidence.dynamic_evidence_complete or evidence.truncated:
        return evidence
    components = [c for c in evidence.effective_components if c.get("kind") == "table"]
    counts = Counter(c.get("component_id") for c in components)
    replacements: dict[int, list[dict[str, Any]]] = {}
    consumed: set[int] = set()
    for component in components:
        cid = component.get("component_id")
        if counts[cid] != 1 or not component.get("complete") or component.get("diagnostics"):
            continue
        columns = component.get("columns", ())
        labels = [str(column.get("label") or "") for column in columns]
        keys = [str(column.get("key") or "") for column in columns]
        if (not labels or len(labels) > _MAX_FIELDS or len(set(keys)) != len(keys)
                or len({exact_field_text(label).casefold() for label in labels}) != len(labels)):
            continue
        table_rows = component.get("rows", ())
        if not table_rows or len(table_rows) > _MAX_RECORDS:
            continue
        if any(str(action.get("component_id") or "") == cid
               or str(action.get("component_id") or "").startswith(str(cid) + "[")
               for action in evidence.action_taxonomy):
            continue
        matched = []
        for row in table_rows:
            if len(row) != len(labels) or any(not str(value).strip() for value in row):
                break
            matches = [record for record in records
                       if _table_record_matches(record, labels, [str(value) for value in row])]
            if len(matches) != 1:
                break
            matched.append(matches[0])
        if (len(matched) != len(table_rows)
                or len({r.source_index for r in matched}) != len(matched)):
            continue
        indices = [i for i, row in enumerate(evidence.evidence_ownership)
                   if row.get("component_id") == cid and row.get("owner") == "source_table"]
        if (len(indices) != len(labels) + len(table_rows)
                or indices != list(range(indices[0], indices[-1] + 1))):
            continue
        replacements[indices[0]] = [
            {"component_id": cid, "component_type": "table", "owner": "generic_visible_content",
             "text": record.prose, "grpo_table_source_index": record.source_index,
             "grpo_table_instance": f"table:{cid}:{i}"}
            for i, record in enumerate(matched)
        ]
        consumed.update(indices)
    if not replacements:
        return evidence
    ownership = []
    for index, row in enumerate(evidence.evidence_ownership):
        if index in replacements:
            ownership.extend(replacements[index])
        elif index not in consumed:
            ownership.append(row)
    return replace(evidence, evidence_ownership=tuple(ownership))
