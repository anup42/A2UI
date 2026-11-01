"""Exact, source-bound semantic field projections for the opt-in pilot only."""

import re
from collections import Counter

from .formula_evidence_v2 import formula_literal_key_v2
from .pilot_layout_v2 import layout_text


def _section(source, unit):
    """An unambiguous preceding explicit heading, never an inferred entity."""
    heading, found = None, []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if layout_text(line) == layout_text(unit):
            found.append(heading)
        match = re.fullmatch(r"\s*(?:#{1,6}\s+(.+)|\*\*([^*]+)\*\*)\s*", line)
        if match:
            heading = layout_text(match[1] or match[2])
        elif (line.strip() and (index == 0 or not lines[index - 1].strip())
              and index + 1 < len(lines) and lines[index + 1].strip()
              and len(line.strip()) <= 120 and not re.search(r"[=|≈≥≤×÷→]|://", line)
              and not re.match(r"\s*(?:[-*+]\s+|\d+[.)]\s+)", line)):
            heading = layout_text(line)
    return found[0] if len(found) == 1 else None


def formula_owner_matches(source, unit, group, ownership):
    card = group[0].get("card_instance")
    if not card:
        return True
    titles = [row["text"] for row in ownership if row.get("card_instance") == card
              and row.get("component_type") == "card"]
    heading = _section(source, unit)
    return heading is not None and len(titles) == 1 and layout_text(titles[0]) == heading


def project_exact_formula_fragments(source, ownership, *, source_text):
    """A Formula and its adjacent qualifier must reconstruct one whole unit."""
    units = [unit for unit in source if any(char in unit.raw for char in "=≈≥≤")]
    rows, consumed, certificates = list(ownership), set(), []
    for first, row in enumerate(rows):
        if first in consumed or row.get("owner") != "source_formula":
            continue
        instance = row.get("render_instance")
        indices = [i for i, item in enumerate(rows) if item.get("owner") == "source_formula"
                   and item.get("render_instance") == instance]
        if not instance or first != indices[0] or indices != list(range(first, indices[-1] + 1)):
            continue
        group = [rows[i] for i in indices]
        after = indices[-1] + 1
        if after >= len(rows):
            continue
        qualifier = rows[after]
        if (qualifier.get("component_type") != "text" or qualifier.get("owner") != "generic_visible_content"
                or not row.get("parent_instance") or qualifier.get("parent_instance") != row.get("parent_instance")
                or qualifier.get("card_instance") != row.get("card_instance")):
            continue
        fields = [str(item["text"]) for item in group]
        variants = [" ".join(fields)]
        if len(fields) >= 2:
            variants.append(fields[0] + ": " + " ".join(fields[1:]))
        keys = {formula_literal_key_v2(value + " " + str(qualifier["text"])) for value in variants}
        matches = [unit for unit in units if formula_literal_key_v2(layout_text(unit.raw)) in keys
                   and formula_owner_matches(source_text, unit.raw, group, rows)]
        if len(matches) != 1:
            continue
        unit = matches[0]
        rows[first] = {**row, "owner": "generic_visible_content", "text": unit.raw}
        consumed.update([*indices[1:], after])
        certificates.append({"source_text": unit.raw, "component_id": row.get("component_id"),
                             "proof": "exact_formula_and_adjacent_qualifier"})
        units.remove(unit)
    return [row for i, row in enumerate(rows) if i not in consumed], certificates


def project_exact_email_fields(source, ownership):
    """Reconcile visible EmailPreview fields against complete literal units.

    Only the Subject presentation label can be removed, only from a subject
    field. Body text is partitioned using complete exact source units; there
    must be one unique partition with every character-token represented.
    """
    from .matching_v5_1 import prepare_text_block
    from .pilot_content_v2 import split_exact_multiline

    available = Counter(unit.raw for unit in source)
    rows, certificates = [], []
    for row in ownership:
        field = row.get("semantic_field")
        if row.get("owner") != "source_email" or field not in {"subject", "body"}:
            rows.append(row)
            continue
        text = str(row.get("text") or "")
        if field == "subject":
            candidates = [raw for raw, count in available.items() if count > 0
                          and re.fullmatch(r"Subject:\s*" + re.escape(text), raw, re.DOTALL)]
            matched = candidates if len(candidates) == 1 else []
        else:
            pieces, _ = split_exact_multiline(source, [prepare_text_block(text)])
            matched = [item.raw for item in pieces]
            if any(available[raw] < count for raw, count in Counter(matched).items()):
                matched = []
        if not matched:
            rows.append(row)
            continue
        for raw in matched:
            available[raw] -= 1
            rows.append({**row, "text": raw, "owner": "generic_visible_content"})
            certificates.append({"source_text": raw, "component_id": row.get("component_id"),
                                 "proof": "exact_email_" + field})
    return rows, certificates


def project_exact_track_tables(source, ownership, tables, *, source_text):
    """Certify complete title/artist/duration lists rendered as native cards.

    The declared record grammar supplies header meaning. Arbitrary new table
    headers, changed fields, omitted rows, or different record order fail.
    """
    if not re.search(r"(?i)\b(?:music|playlist|tracks?)\b", source_text):
        return list(ownership), []
    records = []
    pattern = re.compile(r"\s*[-*+]\s+\*\*([^*|]+)\*\*\s*\|\s*([^|]+?)\s*\|\s*(\d+:\d{2})\s*")
    for line in source_text.splitlines():
        match = pattern.fullmatch(line)
        if match:
            records.append((line.strip(), tuple(part.strip() for part in match.groups())))
        elif records and records[-1] is not None:
            records.append(None)
    # Each consecutive list is an indivisible record set.
    runs, current = [], []
    for record in [*records, None]:
        if record is None:
            if current:
                runs.append(current)
                current = []
        else:
            current.append(record)
    rows, consumed, additions, certificates = list(ownership), set(), {}, []
    for table in tables:
        if tuple(table.headers) != ("Track", "Artist", "Duration"):
            continue
        actual = tuple(tuple(str(row.get(key, "")) for key in table.keys) for row in table.rows)
        matching = [run for run in runs if tuple(record[1] for record in run) == actual]
        if len(matching) != 1:
            continue
        run = matching[0]
        group = [i for i, row in enumerate(rows) if row.get("component_id") == table.element_id
                 and row.get("component_type") == "table"]
        expected = Counter([*table.headers, *(" | ".join(record) for record in actual)])
        if not group or Counter(str(rows[i]["text"]) for i in group) != expected:
            continue
        heading = _section(source_text, run[0][0])
        parent = rows[group[0]].get("parent_instance")
        headings = [item["text"] for item in rows[:group[0]] if item.get("rendered_heading")
                    and parent and item.get("parent_instance") == parent]
        if heading is None or not headings or layout_text(headings[-1]) != heading:
            continue
        raw_units = [record[0] for record in run]
        if not all(any(layout_text(unit.raw) == layout_text(raw) for unit in source) for raw in raw_units):
            continue
        additions[group[0]] = [{**rows[group[0]], "text": raw, "owner": "generic_visible_content"} for raw in raw_units]
        consumed.update(group)
        certificates.extend({"source_text": raw, "component_id": table.element_id,
                             "proof": "exact_track_artist_duration_records"} for raw in raw_units)
        runs.remove(run)
    return [item for i, row in enumerate(rows) for item in additions.get(i, [] if i in consumed else [row])], certificates
