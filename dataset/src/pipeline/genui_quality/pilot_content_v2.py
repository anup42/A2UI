"""Opt-in lossless segmentation for literal multiline renderer text.

Only a freshly certified ownership/assignment view changes. Every character
and numeric occurrence survives; raw evidence still controls additions,
duplication, actions and roles. No threshold or acceptance override.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from .matching_v5_1 import PreparedTextBlock
from .pilot_layout_v2 import exact_fragments, layout_text


def project_exact_headings(source_ownership, output_ownership):
    """Assign exact rendered heading occurrences to their existing source role.

    Keep raw headings, counts, order and text unchanged in every other channel.
    Unmatched or duplicated headings stay generic and receive no certificate.
    """
    remaining = Counter(tuple(layout_text(unit.text).split()) for unit in source_ownership
                        if unit.owner == "heading")
    rows, certificates = [], []
    for index, row in enumerate(output_ownership):
        key = tuple(layout_text(str(row.get("text") or "")).split())
        if (row.get("owner") == "generic_visible_content"
                and (row.get("rendered_heading") is True or row.get("component_type") == "card")
                and remaining[key]):
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
        parts = layout_text(unit.raw).split("|")
        if not 2 <= len(parts) <= 16 or "\\|" in unit.raw:
            continue
        # Pipes also mean regex alternation and bitwise OR. Recognize only
        # the repository's explicit named-entity + labeled-fields grammar;
        # never reinterpret an arbitrary prose/code/operator pipe as layout.
        record = _parsed([part.strip() for part in parts], 0, set(),
                         source_format="pipe", source_indices=(0,),
                         original_source_units=(unit.raw,))
        option_record = (
            re.fullmatch(r"Option (?:[A-Z]|\d+): [^|:\n]+", parts[0].strip())
            and len(parts) >= 3 and all(p.strip() for p in parts[1:])
        )
        dated_record = (re.fullmatch(r"(?:[A-Za-z]+ \d{1,2}|\d{4}-\d{2}-\d{2}) \([A-Za-z]+\): .+", parts[0].strip())
                        and len(parts) >= 3 and all(p.strip() for p in parts))
        if (record is not None and record.identity_label is not None) or option_record or dated_record:
            candidates.append((unit, tuple(tuple(part.split()) for part in parts)))
            if dated_record or option_record:
                date, subject = parts[0].split(":", 1)
                candidates.append((unit, tuple(tuple(part.split()) for part in (date, subject, *parts[1:]))))
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
                    or any(row.get("pilot_entity_projected") for row in group)):
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


def project_exact_siblings(source, ownership):
    """Join adjacent Text siblings only when their WHOLE ordered text is exact."""
    units = {unit.raw for unit in source}
    rows = [dict(row) for row in ownership]
    consumed, certificates = set(), []
    for start, row in enumerate(rows):
        parent = row.get("parent_instance")
        if start in consumed or not parent or row.get("component_type") != "text":
            continue
        matches = []
        for end in range(start + 1, min(len(rows), start + 16)):
            group = rows[start:end + 1]
            if any(i in consumed for i in range(start, end + 1)) or any(
                    item.get("parent_instance") != parent
                    or item.get("component_type") != "text"
                    or item.get("owner") != "generic_visible_content"
                    or item.get("pilot_entity_projected")
                    or item.get("card_instance") != row.get("card_instance") for item in group):
                break
            parts = [str(item.get("text") or "") for item in group]
            matches.extend((end, unit) for unit in units if exact_fragments(unit, parts))
        if len(matches) == 1:
            end, original = matches[0]
            row["text"] = original
            consumed.update(range(start + 1, end + 1))
            certificates.append({"source_text": original, "parent_instance": parent,
                                 "output_indices": list(range(start, end + 1)),
                                 "ordered_literal_fragments_exact": True})
    return [row for i, row in enumerate(rows) if i not in consumed], certificates


def project_exact_plain_table_headers(source, ownership, tables):
    """Certify a pipe header only with ALL of its ordered visible rows.

    Arbitrary pipe expressions are not tables. Require a complete contiguous
    source run with a header and >=2 data rows matching one renderer Table.
    Optional Markdown border pipes/separator lines and omitted empty trailing
    cells are syntax; nonempty cells, column ownership and row order stay exact.
    """
    from .pilot_layout_v2 import table_cell_spelling

    def cells(raw, bordered):
        if "\\|" in raw or "|" not in raw:
            return None
        value = raw.strip()
        if bordered:
            if not value.startswith("|") or not value.endswith("|"):
                return None
            value = value[1:-1]
        return tuple(table_cell_spelling(part.strip()) for part in value.split("|"))

    runs = []
    cursor, fence = 0, None
    while cursor < len(source):
        raw = source[cursor].raw.strip()
        if raw.startswith(("```", "~~~")):
            marker = raw[:3]
            fence = None if fence == marker else marker if fence is None else fence
        if fence is not None or "|" not in raw:
            cursor += 1
            continue
        bordered = raw.startswith("|") and raw.endswith("|")
        header = cells(raw, bordered)
        if not header or len(header) < 2 or any(not cell for cell in header):
            cursor += 1
            continue
        end, values, originals = cursor + 1, [header], [source[cursor].raw]
        if end < len(source):
            separator = cells(source[end].raw, bordered)
            if separator and len(separator) == len(header) and all(re.fullmatch(r":?-{3,}:?", c) for c in separator):
                end += 1
        while end < len(source):
            row = cells(source[end].raw, bordered)
            if row is None:
                break
            if len(row) > len(header):
                values = []  # Never certify a partial prefix of a longer run.
                while end < len(source) and "|" in source[end].raw:
                    end += 1
                break
            values.append(row + ("",) * (len(header) - len(row)))
            originals.append(source[end].raw)
            end += 1
        if len(values) >= 3:
            runs.append((originals, values))
        cursor = max(cursor + 1, end)
    rows, certificates = [dict(row) for row in ownership], []
    consumed = set()
    for table in tables:
        expected = [tuple(str(x).strip() for x in table.headers)]
        expected.extend(tuple(str(row.get(key, "")).strip() for key in table.keys)
                        for row in table.rows)
        if len(expected) < 3 or len(expected[0]) < 2 or any(not x for x in expected[0]):
            continue
        matches = {tuple(originals) for originals, values in runs if values == expected}
        if len(matches) != 1:
            continue
        originals = next(iter(matches))
        # Do not associate two occurrences just because their element ids match.
        groups = {}
        for i, row in enumerate(rows):
            if row.get("component_id") == table.element_id and row.get("table_header") is True:
                groups.setdefault(row.get("render_instance"), []).append(i)
        if len(groups) != 1 or None in groups:
            continue
        indices = next(iter(groups.values()))
        if tuple(str(rows[i]["text"]).strip() for i in indices) != expected[0]:
            continue
        rows[indices[0]]["text"] = originals[0]
        consumed.update(indices[1:])
        certificates.extend({"source_text": original, "output_indices": indices,
                             "complete_ordered_table_exact": True} for original in originals)
    return [row for i, row in enumerate(rows) if i not in consumed], certificates


def project_exact_formula_prose(source, ownership):
    """Credit a generic source equation only if every visible Formula field matches."""
    from .formula_evidence_v2 import formula_literal_key_v2

    units = {}
    for unit in source:
        if any(symbol in unit.raw for symbol in ("=", "≈", "≥", "≤")):
            units.setdefault(formula_literal_key_v2(unit.raw), set()).add(unit.raw)
    rows, groups = [dict(row) for row in ownership], {}
    for i, row in enumerate(rows):
        if row.get("owner") == "source_formula" and row.get("render_instance"):
            groups.setdefault(row["render_instance"], []).append(i)
    consumed, certificates = set(), []
    for instance, indices in groups.items():
        texts = [str(rows[i].get("text") or "") for i in indices]
        variants = [" ".join(texts)]
        if len(texts) == 2:
            variants.append(texts[0] + ": " + texts[1])
        matches = set().union(*(units.get(formula_literal_key_v2(text), set()) for text in variants))
        if len(matches) != 1:
            continue
        original = next(iter(matches))
        rows[indices[0]].update(text=original, owner="generic_visible_content")
        consumed.update(indices[1:])
        certificates.append({"source_text": original, "render_instance": instance,
                             "output_indices": indices, "complete_formula_fields_exact": True})
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
