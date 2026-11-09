"""Exact, source-bound semantic field projections for the opt-in pilot only."""

import re
from collections import Counter

from .formula_evidence_v2 import formula_literal_key_v2
from .pilot_layout_v2 import layout_text, prose_spelling

EMAIL_FIELDS = {"subject": "Subject", "to": "To", "from": "From", "date": "Date"}


def _email_body_spelling(value):
    return re.sub(r"(?m)^[ \t]*[-*+•][ \t]+(?=[^\W\d_])", "• ", prose_spelling(value))


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

    New presentation equivalences apply only to one source-bound draft and
    one rendered EmailPreview. Do not lend fields from another email.
    """
    from .matching_v5_1 import prepare_text_block
    from .pilot_content_v2 import split_exact_multiline

    available = Counter(unit.raw for unit in source)
    email = [row for row in ownership if row.get("owner") == "source_email"]
    subjects = [raw for raw in available if re.fullmatch(r"Subject:\s*\S.*", prose_spelling(raw))]
    subject_rows = [row for row in email if row.get("semantic_field") == "subject"]
    one_draft = (len(subjects) == len(subject_rows) == 1 and available[subjects[0]] == 1
                 and len({row.get("render_instance") for row in email}) == 1
                 and bool(re.fullmatch(r"Subject:\s*" + re.escape(str(subject_rows[0]["text"])),
                                       prose_spelling(subjects[0]))))
    spellings = {}
    if one_draft:
        for unit in source:
            spellings.setdefault(tuple(_email_body_spelling(unit.raw).split()), set()).add(unit.raw)
    unique_units = tuple(unit for unit in source if one_draft
                         and len(spellings[tuple(_email_body_spelling(unit.raw).split())]) == 1)
    rows, certificates = [], []
    for row in ownership:
        field = row.get("semantic_field")
        fields = {*EMAIL_FIELDS, "body"} if one_draft else {"subject", "body"}
        if row.get("owner") != "source_email" or field not in fields:
            rows.append(row)
            continue
        text = str(row.get("text") or "")
        if field in EMAIL_FIELDS:
            label = EMAIL_FIELDS[field]
            candidates = [raw for raw, count in available.items() if count > 0
                          and re.fullmatch(re.escape(label) + r":\s*" + re.escape(text),
                                           prose_spelling(raw) if one_draft else raw)]
            matched = candidates if len(candidates) == 1 else []
        else:
            pieces, _ = split_exact_multiline(source, [prepare_text_block(text)])
            matched = [item.raw for item in pieces]
            if one_draft and not all(raw in available for raw in matched):
                key = tuple(_email_body_spelling(text).split())
                if len(spellings.get(key, ())) == 1:
                    matched = list(spellings[key])
                else:
                    pieces, _ = split_exact_multiline(unique_units, [prepare_text_block(text)],
                                                       spelling=_email_body_spelling)
                    matched = [item.raw for item in pieces]
                ordered = iter(unit.raw for unit in source)
                if not all(any(raw == unit for unit in ordered) for raw in matched):
                    matched = []
            if any(available[raw] < count for raw, count in Counter(matched).items()):
                matched = []
        if not matched:
            rows.append({**row, "text": f"{EMAIL_FIELDS[field]}: {text}", "owner": "generic_visible_content"}
                        if one_draft and field in EMAIL_FIELDS else row)
            continue
        for raw in matched:
            available[raw] -= 1
            rows.append({**row, "text": raw, "owner": "generic_visible_content"})
            certificates.append({"source_text": raw, "component_id": row.get("component_id"),
                                 "proof": "exact_email_" + field})
    return rows, certificates


_CHART_TITLE = re.compile(r"(?i)^Chart Title:\s*(.+)$")
_CHART_AXIS = re.compile(r"(?i)^(X|Y|Right Y)[ -]Axis:\s*(.+)$")


def project_exact_chart_axes(source_text, ownership):
    """Bind explicit axis directives to one complete, title-matched Chart."""
    scopes, scope = {}, None
    for line in source_text.splitlines():
        raw = layout_text(line)
        if match := _CHART_TITLE.fullmatch(raw):
            scope = {}
            scopes.setdefault(match[1], []).append(scope)
        elif scope is not None and (match := _CHART_AXIS.fullmatch(raw)):
            scope.setdefault(match[1].casefold(), []).append((line.strip(), match[2]))
        elif not raw.startswith(("Chart Type:", "Bubble Size:")):
            scope = None

    instances = {}
    for row in ownership:
        if row.get("component_type") == "chart" and row.get("chart_title"):
            instances.setdefault(row["chart_title"], set()).add(row.get("render_instance"))
    rows, certificates = [], []
    for row in ownership:
        title, field = row.get("chart_title"), row.get("semantic_field")
        matches = scopes.get(title, ())
        if len(matches) == len(instances.get(title, ())) == 1 and len(matches[0].get(field, ())) == 1:
            original, label = matches[0][field][0]
            if layout_text(str(row["text"])) == label:
                rows.append({**row, "owner": "generic_visible_content", "text": original})
                certificates.append({"source_text": original, "component_id": row["component_id"],
                                     "proof": "exact_chart_axis"})
                continue
        rows.append(row)
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
