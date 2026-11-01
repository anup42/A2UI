"""Opt-in lossless segmentation for literal multiline renderer text.

Only the assignment view changes. Raw evidence still controls additions,
numbers, duplication, actions and roles; no threshold or acceptance override.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .matching_v5_1 import PreparedTextBlock


def project_exact_headings(source_ownership, output_ownership):
    """Assign exact rendered heading occurrences to their existing source role.

    Keep raw headings, counts, order and text unchanged in every other channel.
    Unmatched or duplicated headings stay generic and receive no certificate.
    """
    remaining = Counter(tuple(unit.text.split()) for unit in source_ownership
                        if unit.owner == "heading")
    rows, certificates = [], []
    for index, row in enumerate(output_ownership):
        key = tuple(str(row.get("text") or "").split())
        if (row.get("owner") == "generic_visible_content"
                and row.get("rendered_heading") is True and remaining[key]):
            remaining[key] -= 1
            rows.append({**row, "owner": "pilot_certified_source_heading"})
            certificates.append({"output_index": index, "text": row["text"]})
        else:
            rows.append(row)
    return rows, certificates


def project_exact_pipe_cards(source, ownership, generic_owners):
    """Join only a source's explicit pipe-separated fields in one Card.

    Each field retains its complete ordered literal spelling. No label inference,
    cross-card concatenation, partial-field or reordered-field credit is allowed.
    Source tables never enter this generic-content view.
    """
    from .grpo_entity_formats import _parsed

    rows = [dict(row) for row in ownership]
    indices = [i for i, row in enumerate(rows) if row.get("owner") in generic_owners
               and str(row.get("text") or "").strip()]
    candidates = []
    for unit in source:
        parts = unit.raw.split("|")
        if not 2 <= len(parts) <= 16 or "\\|" in unit.raw:
            continue
        # Pipes also mean regex alternation and bitwise OR. Recognize only
        # the repository's explicit named-entity + labeled-fields grammar;
        # never reinterpret an arbitrary prose/code/operator pipe as layout.
        record = _parsed([part.strip() for part in parts], 0, set(),
                         source_format="pipe", source_indices=(0,),
                         original_source_units=(unit.raw,))
        if record is not None and record.identity_label is not None:
            candidates.append((unit, tuple(tuple(part.split()) for part in parts)))
    consumed, certificates = set(), []
    for position, index in enumerate(indices):
        if index in consumed:
            continue
        matches = []
        for unit, parts in candidates:
            window = indices[position:position + len(parts)]
            if len(window) != len(parts) or any(i in consumed for i in window):
                continue
            if window != list(range(window[0], window[-1] + 1)):
                continue
            group = [rows[i] for i in window]
            instance = group[0].get("card_instance")
            if (not instance or any(row.get("card_instance") != instance for row in group)
                    or any(row.get("component_type") not in {"card", "text"} for row in group)
                    or any(len(row.get("card_ancestors", ())) != 1 for row in group)):
                continue
            if tuple(tuple(str(row["text"]).split()) for row in group) == parts:
                matches.append((unit, window))
        # Equivalent duplicated source occurrences still have one exact shape;
        # downstream assignment enforces their actual multiplicity.
        unique = {(unit.raw, tuple(window)): (unit, window) for unit, window in matches}
        if len(unique) != 1:
            continue
        unit, window = next(iter(unique.values()))
        rows[index]["text"] = unit.raw
        consumed.update(window[1:])
        certificates.append({"output_indices": window, "source_text": unit.raw,
                             "card_instance": rows[index]["card_instance"],
                             "ordered_literal_fields_exact": True})
    return [row for i, row in enumerate(rows) if i not in consumed], certificates


def split_exact_multiline(source: Sequence[PreparedTextBlock],
                          output: Sequence[PreparedTextBlock]):
    # Whitespace alone is a component/layout boundary. Retain EVERY other
    # character, including approximation, signs, scripts and punctuation.
    # Generic lexical tokenizers discard meaningful math symbols and cannot
    # justify replacing output blocks with original source units.
    units = {}
    for unit in source:
        key = tuple(unit.raw.split())
        if key:
            units.setdefault(key, unit)
    trie = {}
    for key, unit in units.items():
        node = trie
        for token in key:
            node = node.setdefault(token, {})
        node[None] = unit
    result, certificates = [], []
    remaining_work = 100_000
    for index, block in enumerate(output):
        tokens = tuple(block.raw.split())
        if "\n" not in block.raw or tokens in units:
            result.append(block)
            continue
        # At most two distinct partitions are retained. Ambiguity fails closed.
        ways = {0: [()]}
        exhausted = False
        for start in range(len(tokens)):
            if start not in ways:
                continue
            paths_here = [path for path in ways[start] if len(path) < len(source)]
            if not paths_here:
                continue
            node = trie
            for end in range(start, len(tokens)):
                remaining_work -= 1
                if remaining_work < 0:
                    exhausted = True
                    break
                node = node.get(tokens[end])
                if node is None:
                    break
                if None in node:
                    paths = ways.setdefault(end + 1, [])
                    for path in paths_here:
                        if len(paths) < 2:
                            paths.append((*path, node[None]))
            if exhausted:
                break
        partitions = [] if exhausted else ways.get(len(tokens), [])
        if len(partitions) == 1 and len(partitions[0]) > 1:
            result.extend(partitions[0])
            certificates.append({"output_index": index,
                                 "source_unit_count": len(partitions[0]),
                                 "ordered_literal_tokens_exact": True})
        else:
            result.append(block)
    return tuple(result), certificates
