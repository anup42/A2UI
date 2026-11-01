"""Exact presentation certificates used only by the opt-in Stage 3 pilot.

These are spelling/layout projections, not semantic similarity. Source bytes,
raw renderer evidence, action roles and admission thresholds remain untouched.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

_PREFIX = re.compile(r"^\s*(?:#{1,6}\s+|(?:[-*+•]|\d+[.)])\s+)")
_PROTECTED = re.compile(r"(`+)[\s\S]*?\1|(?<!\\)\$[^$\n]+\$")


def prose_spelling(value: str) -> str:
    """Remove paired prose emphasis; keep code, math and literal operators.

    Do not globally delete stars/underscores: they can be multiplication,
    powers, script markers, path characters or form blanks.
    """
    def plain(text):
        for marker in ("**", "__", "*", "_"):
            escaped = re.escape(marker)
            pattern = (rf"(?<![\w\\*_]){escaped}([^\s*_](?:[^\n*_]*?[^\s*_])?)"
                       rf"{escaped}(?![\w*_])")
            def unwrap(match, marker=marker):
                body = match[1]
                prose = re.search(r"[^\W\d_]", body) and not re.search(r"[=+×÷<>^~]", body)
                return body if prose or len(marker) == 2 else match[0]
            text = re.sub(pattern, unwrap, text)
        return text
    parts, cursor = [], 0
    for match in _PROTECTED.finditer(str(value)):
        parts.extend((plain(value[cursor:match.start()]), match[0]))
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


def table_cell_spelling(value: str) -> str:
    """Bounded source Markdown spellings, not algebraic equivalence.

    Only an inline comparison symbol and its literal numeric operand can shed
    dollar delimiters. General formulas, scripts and unknown commands stay raw.
    """
    value = prose_spelling(str(value))
    parts = re.split(r"(`+[^`]*`+)", value)
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(r"(?<!\\)\$\s*\\(geq?|leq?|neq)\s+([+\-]?\d+(?:\.\d+)?)\s*\$",
                             lambda m: {"ge": "≥", "geq": "≥", "le": "≤", "leq": "≤", "neq": "≠"}[m[1]] + " " + m[2], parts[index])
        parts[index] = parts[index].replace(r"\$", "$")
    return "".join(parts)


def table_fidelity_v2(expected, actual, charts, **kwargs):
    """Supplement canonical matching with exact table cell/column ownership.

    Canonical matching, representation policies and work budgets still run.
    This extra check can only reduce its score; it cannot turn a failure into
    a pass. Comparison operators must not disappear in lexical tokenization.
    """
    from .metrics_v5_3 import table_fidelity_v5_3

    value, detail = table_fidelity_v5_3(expected, actual, charts, **kwargs)
    if value is None:
        return value, detail

    remaining = list(actual)
    required, matched = 0, 0
    key = lambda x: " ".join(table_cell_spelling(str(x)).split())
    for source in expected:
        if not source.required or source.minimum_count <= 0:
            continue
        # Chart-permitted representations retain their existing dedicated gate.
        if source.representation_policy != "table_only":
            continue
        headers = tuple(key(x) for x in source.headers)
        for _ in range(source.minimum_count):
            required += 1
            for position, candidate in enumerate(remaining):
                actual_headers = tuple(key(x) for x in candidate.headers)
                if len(set(headers)) != len(headers) or Counter(headers) != Counter(actual_headers):
                    continue
                order = [candidate.keys[actual_headers.index(header)] for header in headers]
                rows = tuple(tuple(key(row.get(column, "")) for column in order) for row in candidate.rows)
                wanted = tuple(tuple(key(cell) for cell in row) for row in source.rows)
                exact = rows == wanted if source.order_sensitive else Counter(rows) == Counter(wanted)
                if exact:
                    remaining.pop(position)
                    matched += 1
                    break
    # A second copy of a source-owned table is extra evidence, not another
    # representation of its single required occurrence.
    source_headers = {tuple(sorted(key(x) for x in table.headers)) for table in expected
                      if table.required and table.minimum_count > 0 and table.representation_policy == "table_only"}
    duplicates = sum(tuple(sorted(key(x) for x in table.headers)) in source_headers for table in remaining)
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
        if candidate.get("title") == "Formula" and content.startswith("Formula "):
            body_key = formula_literal_key_v2(content[len("Formula "):])
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
        units.append(original if _PREFIX.match(raw) else _PREFIX.sub("", original))
    blocks = tuple(prepare_text_block(value) for value in units)
    # The source-side table projection changes no cell association or counts.
    tables = [replace(table,
                      headers=tuple(table_cell_spelling(x) for x in table.headers),
                      rows=tuple(tuple(table_cell_spelling(x) for x in row) for row in table.rows))
              for table in base.source_contract.tables]
    return replace(base, source_contract=replace(base.source_contract, tables=tables),
                   applicability=replace(base.applicability, generic_source_units=tuple(units)),
                   content_blocks=blocks,
                   content_exact_index=group_exact_indices([content_exact_key_v5_3(x) for x in blocks]))


def explicit_field_delimiters(record) -> bool:
    """A clock or drive colon cannot introduce a labeled entity field."""
    originals = "\n".join(record.original_source_units)
    return all(re.search(re.escape(label) + r":\s+" + re.escape(value), originals)
               for label, value in record.field_pairs)


def certify_section_fields(source_text, records, evidence, detail):
    """Bind a property's split fields inside its exact source-section Card.

    A labeled-comma bundle such as Age/Sex is not a named hotel record. Its
    owner is the unique preceding source heading, which must be the Card title.
    Complete ordered fields, unique occurrences and unchanged labels are required.
    """
    rows = [row for row in evidence.evidence_ownership
            if row.get("owner") == "generic_visible_content" and str(row.get("text") or "").strip()]
    certified = list(detail["certified_records"])
    existing = {item["source_index"] for item in certified}
    lines = source_text.splitlines()
    for record in records:
        if (record.source_index in existing or record.source_format != "labeled_comma"
                or len(record.fixed_texts) != 2 or record.actions or record.identity_label):
            continue
        locations = [i for i, line in enumerate(lines)
                     if layout_text(line) == layout_text(record.original_source_units[0])]
        if len(locations) != 1:
            continue
        heading = None
        for line in reversed(lines[:locations[0]]):
            if re.fullmatch(r"\s*(?:#{1,6}\s+.+|\*\*[^*]+\*\*|__[^_]+__)\s*", line):
                heading = layout_text(line)
                break
        if not heading:
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
    uses = Counter(index for item in certified for index in item["output_indices"])
    certified = [item for item in certified if all(uses[i] == 1 for i in item["output_indices"])]
    bound = {item["source_index"] for item in certified if item["actions_exact"]}
    return {**detail, "certified_records": certified, "bound_record_count": len(bound),
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
