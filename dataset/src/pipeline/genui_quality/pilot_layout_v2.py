"""Exact presentation certificates used only by the opt-in Stage 3 pilot.

These are spelling/layout projections, not semantic similarity. Source bytes,
raw renderer evidence, action roles and admission thresholds remain untouched.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

_CODE_SPAN = re.compile(r"(?<![\\`])(`+)([^`\n]+)\1(?!`)")
_PREFIX = re.compile(r"^\s*(?:#{1,6}\s+|(?:[-*+•]|\d+[.)])\s+)")
# Recognizable math stays literal even beside a unit/digit or with inner spaces.
# Otherwise dollar delimiters must flank non-space text; a following digit is
# another currency amount, not a closing delimiter across unrelated prose.
_PROTECTED = re.compile(
    r"(?<![\w\\*_])\*\*\$(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\*\*(?![\w*_])|"
    r"(`+)[\s\S]*?\1|"
    r"(?<!\\)\$(?:(?=[^$\n]*[+=^\\_×÷≈≤≥])|(?=\s*\d[\d.,]*\s*\*+\s*\w)|(?!\s*\d))[^$\n]+\$|"
    r"(?<!\\)\$(?=\S)[^$\n]+(?<!\s)\$(?!\d)"
)


def prose_spelling(value: str) -> str:
    """Remove paired prose emphasis; keep code, math and literal operators.

    Do not globally delete stars/underscores: they can be multiplication,
    powers, script markers, path characters or form blanks.
    """
    def plain(text):
        text = re.sub(r"(?<![\w\\*])\*(\d{4}-\d{2}-\d{2})\*(?![\w*])", r"\1", text)
        # Paired strong delimiters can surround a literal identifier. The
        # underscores inside that identifier remain part of its spelling.
        text = re.sub(r"(?<![\w\\*])\*\*([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\*\*(?![\w*])", r"\1", text)
        for marker in ("**", "__", "*", "_"):
            escaped = re.escape(marker)
            pattern = (rf"(?<![\w\\*_]){escaped}([^\s*_](?:[^\n*_]*?[^\s*_])?)"
                       rf"{escaped}(?![\w*_])")
            def unwrap(match, marker=marker):
                body = match[1]
                prose = re.search(r"[^\W\d_]", body) and not re.search(r"[=+×÷<>^~]", body)
                sentence = (marker == "*" and body[0].isupper() and body[-1] in ".!?"
                            and len(re.findall(r"[^\W\d_]+", body)) >= 3)
                return body if prose or sentence or len(marker) == 2 else match[0]
            text = re.sub(pattern, unwrap, text)
        return text
    parts, cursor = [], 0
    for match in _PROTECTED.finditer(str(value)):
        # A single inline variable has the same visible letter without its
        # math delimiters. Currency, compound math and code remain literal.
        protected = match[0]
        if protected.startswith("**$"):
            # Consume each emphasized amount before a later dollar can be
            # mistaken for the closing delimiter of a math span.
            protected = protected[2:-2]
        elif re.fullmatch(r"\$[A-Za-z]\$", protected):
            protected = protected[1:-1]
        parts.extend((plain(value[cursor:match.start()]), protected))
        cursor = match.end()
    parts.append(plain(value[cursor:]))
    return "".join(parts)


def layout_text(value: str) -> str:
    return " ".join(_PREFIX.sub("", prose_spelling(str(value))).split())


def exact_fragments(source: str, parts: list[str]) -> bool:
    """Only component boundaries may replace explicit punctuation/whitespace.

    All text within each field remains byte-exact after prose emphasis and
    whitespace. Operators, pipes, decimal points and units are not separators.
    """
    remaining = layout_text(source)
    for index, part in enumerate(parts):
        part = " ".join(prose_spelling(str(part)).split())
        if not part or not remaining.startswith(part):
            return False
        remaining = remaining[len(part):]
        if index + 1 < len(parts):
            separator = re.match(r"^(?:\s*[,;:]\s+|\s+[–—]\s+|\s+)", remaining)
            if separator is None:
                return False
            remaining = remaining[separator.end():]
    return not remaining


def inline_math_spelling(value: str) -> str:
    """Render only explicit, bounded inline operator spellings outside code."""
    operators = {"ge": "≥", "geq": "≥", "le": "≤", "leq": "≤", "neq": "≠", "approx": "≈",
                 "to": "→", "rightarrow": "→", "times": "×", "cdot": "⋅", "div": "÷",
                 "pm": "±", "mp": "∓"}
    number = r"[+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

    def math(text):
        # Recognize a bounded LaTeX temperature without normalizing the
        # inequality, sign, number, or unit into something different.
        text = re.sub(
            rf"(?<![\\$])\$\s*(?:\\(geq?|leq?|neq|approx)\s+)?({number})\s*\^\\circ"
            r"\s*\\(?:text|mathrm)\{([CF])\}\s*\$(?![$\d])",
            lambda m: (operators[m[1]] + " " if m[1] else "") + m[2] + "°" + m[3],
            text,
        )
        text = re.sub(rf"(?<![\\$])\$\s*\\(geq?|leq?|neq|approx|pm|mp)\s+({number})\s*\$(?![$\d])",
                      lambda m: operators[m[1]] + " " + m[2], text)
        text = re.sub(r"(?<![\\$])\$\s*\\(to|rightarrow|times|cdot|div|approx)\s*\$(?![$\d])",
                      lambda m: operators[m[1]], text)
        return re.sub(
            rf"(?<![\\$])\$\s*({number})\s*\\(times|cdot|div)\s*({number})"
            rf"(?:\s*=\s*({number}))?\s*\$(?![$\d])",
            lambda m: f"{m[1]} {operators[m[2]]} {m[3]}" + (f" = {m[4]}" if m[4] else ""),
            text,
        )

    parts, fence = [], None
    for line in value.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            parts.append(line)
            continue
        if fence is not None:
            parts.append(line)
            continue
        cursor = 0
        for span in _CODE_SPAN.finditer(line):
            parts.extend((math(line[cursor:span.start()]), span[0]))
            cursor = span.end()
        parts.append(math(line[cursor:]))
    return "".join(parts)


def table_cell_spelling(value: str) -> str:
    """Bounded source Markdown spellings, not algebraic equivalence."""
    value = inline_math_spelling(prose_spelling(str(value)))
    simple_sum = re.fullmatch(r"\$([+-]?\d+(?:\.\d+)?)\s*([+-])\s*([+-]?\d+(?:\.\d+)?)\$", value)
    if simple_sum:
        return " ".join(simple_sum.groups())
    parts, cursor = [], 0
    for span in _CODE_SPAN.finditer(value):
        # A simple identifier can render without its code delimiters. Keep
        # commands, escaped currency, paths and expressions byte-literal.
        code = span[2] if re.fullmatch(
            r"[A-Za-z_]\w*(?:[.-]\w+)*|\[\[|\]\]|(?:[A-Z][a-z]* ){1,2}[A-Z][a-z]*", span[2]
        ) else span[0]
        parts.extend((value[cursor:span.start()].replace(r"\$", "$"), code))
        cursor = span.end()
    return "".join((*parts, value[cursor:].replace(r"\$", "$")))


def _literal_source_tables(source_text):
    """Read full Markdown tables before the shared parser strips *, _ and ~."""
    from ._core import _TABLE_SEPARATOR_RE

    def cells(line):
        return tuple(cell.strip() for cell in line.strip().removeprefix("|").removesuffix("|").split("|"))

    lines = source_text.splitlines(keepends=True)
    offsets = [0]
    for raw in lines:
        offsets.append(offsets[-1] + len(raw))
    index, fence = 0, None
    while index + 1 < len(lines):
        line = lines[index].lstrip()
        marker = re.match(r"^(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        if fence or marker or "|" not in line or not _TABLE_SEPARATOR_RE.match(lines[index + 1]):
            index += 1
            continue
        header = cells(line)
        start = index
        index += 2
        rows = []
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            rows.append(cells(lines[index]))
            index += 1
        if (all(header) and len(cells(lines[start + 1])) == len(header) and rows
                and all(len(row) == len(header) for row in rows)
                and not any(r"\|" in raw for raw in lines[start:index])):
            yield header, tuple(rows), (offsets[start], offsets[index])


def _restore_literal_tables(source_text, expected):
    from ._core import normalize_markdown

    by_shape = {}
    for headers, rows, _ in _literal_source_tables(source_text):
        shape = (tuple(map(normalize_markdown, headers)),
                 tuple(tuple(map(normalize_markdown, row)) for row in rows))
        by_shape.setdefault(shape, set()).add((headers, rows))
    restored, ambiguous = [], set()
    for index, table in enumerate(expected):
        originals = by_shape.get((table.headers, table.rows), ())
        if len(originals) == 1:
            headers, rows = next(iter(originals))
            table = replace(table, headers=headers, rows=rows)
        elif len(originals) > 1:
            ambiguous.add(index)
        restored.append(table)
    return restored, ambiguous


def table_fidelity_v2(expected, actual, charts, *, source_text="", cta_cells=None, **kwargs):
    """Supplement canonical matching with exact table cell/column ownership.

    Canonical matching, representation policies and work budgets still run.
    This extra check can only reduce its score; it cannot turn a failure into
    a pass. Comparison operators must not disappear in lexical tokenization.
    """
    from .metrics_v5_3 import table_fidelity_v5_3

    literal, ambiguous = _restore_literal_tables(source_text, expected)
    if cta_cells:
        literal = [replace(table, rows=cta_cells.get((table.headers, table.rows), table.rows))
                   if i not in ambiguous else table for i, table in enumerate(literal)]

    def key(value):
        return " ".join(table_cell_spelling(str(value)).split())

    # Compare the same narrowly normalized spelling in the canonical matcher
    # and the exact gate. Keep the original chart representation policies.
    comparable = [replace(table, headers=tuple(map(key, table.headers)),
                          rows=tuple(tuple(map(key, row)) for row in table.rows))
                  for table in literal]
    actual = [replace(table, headers=tuple(map(key, table.headers)),
                      rows=[{column: key(value) for column, value in row.items()} for row in table.rows])
              for table in actual]
    value, detail = table_fidelity_v5_3(comparable, actual, charts, **kwargs)
    if value is None:
        return value, detail

    remaining = list(actual)
    required, matched = 0, 0
    for index, source in enumerate(comparable):
        if not source.required or source.minimum_count <= 0:
            continue
        # Chart-permitted representations retain their existing dedicated gate.
        if source.representation_policy != "table_only":
            continue
        headers = source.headers
        for _ in range(source.minimum_count):
            required += 1
            if index in ambiguous:
                continue
            for position, candidate in enumerate(remaining):
                actual_headers = candidate.headers
                if len(set(candidate.keys)) != len(candidate.keys) or Counter(headers) != Counter(actual_headers):
                    continue
                if len(set(headers)) != len(headers):
                    if headers != actual_headers:
                        continue
                    order = candidate.keys
                else:
                    order = [candidate.keys[actual_headers.index(header)] for header in headers]
                rows = tuple(tuple(row.get(column, "") for column in order) for row in candidate.rows)
                wanted = source.rows
                exact = rows == wanted if source.order_sensitive else Counter(rows) == Counter(wanted)
                if exact:
                    remaining.pop(position)
                    matched += 1
                    break
    # A second copy of a source-owned table is extra evidence, not another
    # representation of its single required occurrence.
    source_headers = {tuple(sorted(table.headers)) for table in comparable
                      if table.required and table.minimum_count > 0 and table.representation_policy == "table_only"}
    duplicates = sum(tuple(sorted(table.headers)) in source_headers for table in remaining)
    detail = {**detail, "pilot_exact_cell_ownership": {
        "required_count": required, "matched_count": matched, "unassigned_source_table_count": duplicates}}
    return min(value, matched / (required + duplicates)) if required else value, detail


def project_exact_formula_roles(source, expected, actual):
    """Align source LaTeX with its exact rendered spelling, never algebra.

    Formula evidence concatenates title/body/result. The source's explicit
    `Formula:` marker may own that same title, but an extra subtitle/result or
    changed expression cannot acquire a source-expression certificate.
    Keep required counts, component constraints and all other role fields.
    """
    from .formula_evidence_v2 import formula_literal_key_v2

    labeled = {formula_literal_key_v2(match[1]) for match in
               re.finditer(r"(?m)^\s*Formula:\s*([^\n]+)$", source)}
    bodies = {}
    for requirement in expected.get("formula", ()):
        content = requirement.get("content")
        if isinstance(content, str) and content.strip():
            bodies.setdefault(formula_literal_key_v2(content), set()).add(content)
    output = {role: [dict(item) for item in items] for role, items in actual.items()}
    certificates = []
    for candidate in output.get("formula", ()):
        content = str(candidate.get("content") or "")
        key = formula_literal_key_v2(content)
        title_owned = False
        title = candidate.get("title")
        if title in {"Formula", "Formula:"} and content.startswith(title + " "):
            body_key = formula_literal_key_v2(content[len(title) + 1:])
            if body_key in labeled:
                key, title_owned = body_key, True
        matches = bodies.get(key, ())
        if len(matches) == 1:
            candidate["content"] = next(iter(matches))
            certificates.append({"component_id": candidate.get("component_id"),
                                 "source_content": candidate["content"],
                                 "source_formula_title_owned": title_owned,
                                 "complete_formula_fields_exact": True})
    return output, certificates


def restore_literal_source_view(base, source_ownership):
    """Recover raw prose for certificates from uniquely source-owned units."""
    from . import _core
    from .matching_v5_1 import prepare_text_block
    from .matching_v5_3 import group_exact_indices
    from .metrics_v5_3 import content_exact_key_v5_3

    originals = {}
    for unit in source_ownership:
        if unit.owner == "generic_prose":
            originals.setdefault(_core.normalize_markdown(unit.text), set()).add(prose_spelling(unit.text))
    units = []
    for raw in base.applicability.generic_source_units:
        matches = originals.get(_core.normalize_markdown(raw), set())
        original = next(iter(matches)) if len(matches) == 1 else raw
        # Preserve the existing parser's record-recognition scope. Restoring a
        # Markdown '*' bullet must not suddenly turn procedural prose into an
        # entity record that the original source plan did not recognize.
        units.append(inline_math_spelling(original if _PREFIX.match(raw) else _PREFIX.sub("", original)))
    blocks = tuple(prepare_text_block(value) for value in units)
    return replace(base, applicability=replace(base.applicability, generic_source_units=tuple(units)),
                   content_blocks=blocks,
                   content_exact_index=group_exact_indices([content_exact_key_v5_3(x) for x in blocks]))


def explicit_field_delimiters(record) -> bool:
    """Clocks, drive colons and procedural clauses do not label fields."""
    if (record.source_format == "labeled_comma" and record.identity_label is None
            and any(re.match(r"(?i)^(?:and )?then\s+", label) for label, _ in record.field_pairs)):
        return False
    originals = "\n".join(record.original_source_units)
    return all(re.search(re.escape(label) + r":\s+" + re.escape(value), originals)
               for label, value in record.field_pairs)


def _property_bundle(record) -> bool:
    return (record.source_format == "labeled_comma" and record.identity_label is None
            and len(record.fixed_texts) == 2 and not record.actions)


def pilot_entity_records(records, *, source_text):
    """Only repeated named records make comma-separated fields mandatory."""
    from .pilot_charts_v2 import declared_chart_row_texts

    chart_rows = declared_chart_row_texts(source_text) if records else set()
    candidates = {record.source_index: record for record in records if explicit_field_delimiters(record)
                  and not (record.source_format == "pipe" and len(record.original_source_units) == 1
                           and layout_text(record.original_source_units[0]) in chart_rows)}
    retained = []
    for index, record in candidates.items():
        neighbors = [candidates[i] for i in (index - 1, index + 1) if i in candidates]
        if record.source_format != "labeled_comma" or record.identity_label or record.actions:
            retained.append(record)
        elif _property_bundle(record):
            # Consecutive property bundles can be split in their own section.
            # An exact literal bundle does not need its own Card.
            if any(_property_bundle(other) for other in neighbors):
                retained.append(record)
        else:
            labels = {label.casefold() for label, _ in record.field_pairs}
            if any(other.source_format == "labeled_comma" and len(other.fixed_texts) == 1
                   and record.entity != other.entity
                   and labels == {label.casefold() for label, _ in other.field_pairs}
                   for other in neighbors):
                retained.append(record)
    return tuple(retained)


def certify_section_fields(source_text, records, evidence, detail):
    """Bind a property's split fields inside its exact source-section Card.

    A labeled-comma bundle such as Age/Sex is not a named hotel record. Its
    owner is the unique preceding source heading, which must be the Card title.
    Complete ordered fields, unique occurrences and unchanged labels are required.
    """
    rows = [row for row in evidence.evidence_ownership
            if row.get("owner") == "generic_visible_content" and str(row.get("text") or "").strip()]
    def owns_literal(index, heading):
        row = rows[index]
        instance = row.get("card_instance")
        if instance:
            titles = [r["text"] for r in rows if r.get("card_instance") == instance
                      and r.get("component_type") == "card"]
            if len(titles) == 1 and layout_text(titles[0]) == heading:
                return True
        parent = row.get("parent_instance")
        headers = [r["text"] for r in rows[:index] if parent and r.get("parent_instance") == parent
                   and r.get("rendered_heading")]
        return bool(headers and layout_text(headers[-1]) == heading)

    certified = list(detail["certified_records"])
    existing = {item["source_index"] for item in certified}
    literal_bundles, consumed_literals = set(), set()
    lines = source_text.splitlines()
    property_units = {layout_text(unit) for item in records if _property_bundle(item)
                      for unit in item.original_source_units}
    for record in records:
        if not _property_bundle(record):
            continue
        source_unit = layout_text(record.original_source_units[0])
        locations = [i for i, line in enumerate(lines) if layout_text(line) == source_unit]
        if len(locations) != 1:
            continue
        heading = None
        for index in range(locations[0] - 1, -1, -1):
            line = lines[index].strip()
            if not line or layout_text(line) in property_units:
                continue
            if re.fullmatch(r"\s*(?:#{1,6}\s+.+|\*\*[^*]+\*\*|__[^_]+__)\s*", line) or ((index == 0 or not lines[index - 1].strip()
                   or (index + 1 < len(lines) and not lines[index + 1].strip()))
                    and len(line) <= 120 and not _PREFIX.match(line)
                    and not re.search(r"[.:!?=|`]|://", line)):
                heading = layout_text(line)
                break
        if not heading:
            continue
        literals = [i for i, row in enumerate(rows) if i not in consumed_literals
                    and layout_text(row["text"]) == source_unit and owns_literal(i, heading)]
        if len(literals) == 1:
            consumed_literals.add(literals[0])
            literal_bundles.add(record.source_index)
            continue
        if record.source_index in existing:
            continue
        expected = [": ".join(record.fixed_texts), *(f"{label}: {value}" for label, value in record.field_pairs)]
        candidates = []
        for start in range(len(rows) - len(expected) + 1):
            group = rows[start:start + len(expected)]
            instance = group[0].get("card_instance")
            if (not instance or any(row.get("card_instance") != instance
                                    or row.get("component_type") != "text" for row in group)
                    or len({row.get("parent_instance") for row in group}) != 1
                    or any(row.get("parent_kind") != "card" for row in group)):
                continue
            titles = [row for row in rows if row.get("card_instance") == instance
                      and row.get("component_type") == "card"]
            if len(titles) != 1 or layout_text(titles[0]["text"]) != heading:
                continue
            if [" ".join(str(row["text"]).split()) for row in group] == expected:
                candidates.append({"card_instance": instance, "source_section": heading,
                                   "output_indices": list(range(start, start + len(expected))),
                                   "source_index": record.source_index, "structured": True,
                                   "actions_exact": True, "source_text": record.prose,
                                   "original_source_units": record.original_source_units,
                                   "source_format": record.source_format})
        if len(candidates) == 1:
            certified.append(candidates[0])
    # No two source records may consume the same renderer occurrence.
    certified = [item for item in certified if item["source_index"] not in literal_bundles]
    uses = Counter(index for item in certified for index in item["output_indices"])
    certified = [item for item in certified if all(uses[i] == 1 for i in item["output_indices"])]
    bound = {item["source_index"] for item in certified if item["actions_exact"]}
    return {**detail, "certified_records": certified, "bound_record_count": len(bound),
            "eligible_record_count": len(records) - len(literal_bundles),
            "structured_record_count": sum(item["structured"] and item["actions_exact"] for item in certified)}


def project_entity_certificates(ownership, detail):
    """Consume certificates before other projections can change their indices."""
    rows = [dict(row) for row in ownership]
    generic = [i for i, row in enumerate(rows) if row.get("owner") == "generic_visible_content"
               and str(row.get("text") or "").strip()]
    certificates = [item for item in detail.get("certified_records", ()) if item["actions_exact"]]
    usage = Counter(i for item in certificates for i in item["output_indices"])
    consumed = set()
    for item in certificates:
        indices = item["output_indices"]
        if not indices or any(i >= len(generic) or i < 0 or usage[i] != 1 for i in indices):
            continue
        rows[generic[indices[0]]]["text"] = item["source_text"]
        rows[generic[indices[0]]]["pilot_entity_projected"] = True
        consumed.update(generic[i] for i in indices[1:])
    return [row for i, row in enumerate(rows) if i not in consumed]


def project_exact_card_fragments(source, ownership):
    """Certify a complete leaf Card, retaining every field and its order."""
    rows, groups = [dict(row) for row in ownership], {}
    for i, row in enumerate(rows):
        if row.get("card_instance"):
            groups.setdefault(row["card_instance"], []).append(i)
    consumed, certificates = set(), []
    for instance, indices in groups.items():
        if len(indices) < 2 or indices != list(range(indices[0], indices[-1] + 1)):
            continue
        group = [rows[i] for i in indices]
        if any(row.get("owner") != "generic_visible_content"
               or row.get("component_type") not in {"card", "text"}
               or row.get("pilot_entity_projected") for row in group):
            continue
        matches = {unit.raw for unit in source if exact_fragments(unit.raw, [row["text"] for row in group])}
        if len(matches) != 1:
            continue
        original = next(iter(matches))
        rows[indices[0]]["text"] = original
        consumed.update(indices[1:])
        certificates.append({"source_text": original, "card_instance": instance,
                             "output_indices": indices, "ordered_literal_fragments_exact": True})
    return [row for i, row in enumerate(rows) if i not in consumed], certificates


def project_exact_equation_rows(source, ownership, tables):
    """Certify complete numbered equation records in a two-column Table.

    The identity column repeats its explicit source prefix (e.g. Year 1).
    Calculation/Equation is a layout label only when every complete second
    cell is the literal source equation. Never infer a value, row or formula.
    """
    rows, certificates = [dict(row) for row in ownership], []
    for table in tables:
        if (len(table.headers) != 2 or table.headers[1] not in {"Calculation", "Equation"}
                or not re.fullmatch(r"[A-Za-z]+", table.headers[0]) or len(table.rows) < 2):
            continue
        pattern = re.compile(re.escape(table.headers[0]) + r" \d+:\s*(.+)")
        originals = [unit.raw for unit in source if pattern.fullmatch(layout_text(unit.raw))]
        values = [(str(row.get(table.keys[0], "")), str(row.get(table.keys[1], ""))) for row in table.rows]
        if len(originals) != len(values) or any(
                not any(symbol in second for symbol in ("=", "≈", "≥", "≤"))
                or layout_text(original) != first + ": " + second
                for original, (first, second) in zip(originals, values)):
            continue
        indices = [i for i, row in enumerate(rows) if row.get("component_id") == table.element_id
                   and row.get("owner") == "source_table"]
        if (len(indices) != len(values) + 2 or len({rows[i].get("render_instance") for i in indices}) != 1
                or indices != list(range(indices[0], indices[-1] + 1))
                or not all(rows[i].get("table_header") is True for i in indices[:2])):
            continue
        for i in indices[:2]:
            rows[i]["owner"] = "pilot_certified_table_layout"
        for i, original in zip(indices[2:], originals):
            rows[i].update(text=original, owner="generic_visible_content")
            certificates.append({"source_text": original, "output_indices": [i],
                                 "complete_ordered_equation_table_exact": True})
    return rows, certificates
