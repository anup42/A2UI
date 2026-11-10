"""Source-owned action/source-link occurrences for the opt-in pilot only."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from . import _core
from .list_evidence_v5_4 import source_heading
from .matching_v5_3 import assignment_to_mapping, certified_assignment_v5_3
from .metrics_v5_2 import f_score
from .metrics_v5_3 import _action_match_score, _canonical_alias
from .pilot_layout_v2 import prose_spelling

SOURCE_PREFIX = "pilot_v2_source_link:"
ACTION_PREFIX = "pilot_v2_action:"
TABLE_ACTION_PREFIX = ACTION_PREFIX + "table:"
_TABLE_ACTION_SOURCE = "pilot_v2_table_action"
_CARD_ACTION_PREFIX = ACTION_PREFIX + "card:"
_CARD_ACTION_SOURCE = "pilot_v2_card_action"
ACTION_LINE_PREFIX = r"(?:[-*+•]\s+|\d+[.)]\s+)?Action:\s*"
_EXPLICIT_ACTION = re.compile(
    rf"(?i)^{ACTION_LINE_PREFIX}\[\s*(?:Button\s*:\s*)?([^\]]+?)\s*\]\s*"
    r"(?:<(https://[^\s<>]+)>|(https://[^\s<>]+|\{\{u\d+\}\}|\[(?:[A-Z_]+_)?URL_\d+\]))\s*$"
)
_LABELED_CARD_ACTION = re.compile(
    r"(?i)^[-*+•]\s+(?!(?:Action|Media):)([^:<>\[\]\x60|]{1,160}):\s*"
    r"\[Button\s*:\s*([^\]]+?)\]\s*<(https://[^\s<>]+)>\s*$"
)
_MARKDOWN_SOURCE = re.compile(r"(?i)^\[([^\]\n]+)\]\((https://[^\s<>]+)\)$")
_LABELED_SOURCE = re.compile(r"^(.+?):\s*(?:<(https://[^\s<>]+)>|(https://[^\s<>]+))\s*$")
_HEADING = re.compile(r"^(?:#{1,6}\s+(.+)|\*\*([^*]+)\*\*)$")
_SOURCE_TITLE = re.compile(
    r"(?i)^(?:(?:[\w][\w &'’/+,().-]*\s+)?(?:sources?|references?|citations?|resources)|"
    r"(?:sources?|references?|citations?|resources) for .+):?$"
)
_MEDIA_FIELD = re.compile(r"(Image|Icon|Video|Audio)\s*=\s*(?:<([^<>\s|]+)>|([^\s<>|]+))")
_INLINE_MEDIA = re.compile(r"[ \t]+\|[ \t]+Media:(.*)$")


def declared_media_occurrences(source: str) -> list[dict]:
    """Parse complete media lines and a record's explicit trailing Media field."""
    found, offset, fence = [], 0, None
    for line in source.splitlines(keepends=True):
        start, offset = offset, offset + len(line)
        raw = line.strip()
        if raw.startswith(("```", "~~~")):
            marker = raw[:3]
            fence = None if fence == marker else marker if fence is None else fence
            continue
        if fence is not None:
            continue
        span, record = (start, offset), None
        line_start = start + len(line) - len(line.lstrip())
        if raw.startswith("Media:"):
            body_start = len("Media:")
        else:
            inline = _INLINE_MEDIA.search(raw)
            if inline is None or r"\|" in raw or raw.startswith("|"):
                continue
            record = raw[:inline.start()].rstrip()
            parts = record.split("|")
            if not 2 <= len(parts) <= 16 or not all(part.strip() for part in parts):
                continue
            begin = line_start + inline.start()
            span = (begin, start + len(line.rstrip("\r\n")))
            body_start = inline.start(1)
        body = raw[body_start:]
        destination_offset = line_start + body_start
        cursor, items = 0, []
        for match in _MEDIA_FIELD.finditer(body):
            if body[cursor:match.start()].strip():
                items = []
                break
            target = 2 if match[2] is not None else 3
            item = {"kind": match[1], "url": match[target], "_span": span,
                    "_url_span": (destination_offset + match.start(target), destination_offset + match.end(target))}
            if record is not None:
                item["_record"] = record
            items.append(item)
            cursor = match.end()
        if items and not body[cursor:].strip():
            found.extend(items)
    return found


def inline_media_view(source: str, *, media=None) -> str:
    """Keep original offsets while excluding proven metadata from visible prose."""
    if media is None:
        media = declared_media_occurrences(source)
    spans = {item["_span"] for item in media if "_record" in item}
    for start, end in sorted(spans, reverse=True):
        source = source[:start] + " " * (end - start) + source[end:]
    return source



def declared_media_fidelity(source: str, actual: Any) -> dict[str, Any]:
    """Require exact visible destinations, including fragments, for declarations."""
    from pipeline.renderer_effective_semantics_v5_4 import canonical_media_kind

    declarations = declared_media_occurrences(source)
    expected = Counter((canonical_media_kind(item["kind"]), item["url"]) for item in declarations)
    inline_keys = {(canonical_media_kind(item["kind"]), item["url"])
                   for item in declarations if "_record" in item}
    visible = Counter((canonical_media_kind(item.get("kind")), item["raw_url"])
                      for item in actual if isinstance(item.get("raw_url"), str))
    missing = expected - visible
    groups = {(kind, _core.normalize_url(url)) for kind, url in expected}
    unexpected = {key: count for key, count in (visible - expected).items()
                  if key in inline_keys or (key not in expected and (key[0], _core.normalize_url(key[1])) in groups)}
    return {"required_count": sum(expected.values()),
            "matched_count": sum((expected & visible).values()),
            "missing": [{"kind": kind, "url": url, "count": count}
                        for (kind, url), count in sorted(missing.items())],
            "unexpected_destinations": [{"kind": kind, "url": url, "count": count}
                                        for (kind, url), count in sorted(unexpected.items())],
            "complete": not missing and not unexpected}


def source_action_occurrences(source: str) -> list[dict[str, Any]]:
    """Complete Action lines and labeled records with an explicit Button."""
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
        explicit = _EXPLICIT_ACTION.fullmatch(raw)
        card = _LABELED_CARD_ACTION.fullmatch(raw)
        if explicit:
            owner, prefix, label, target = "", ACTION_PREFIX, explicit[1], explicit[2] or explicit[3]
        elif card:
            owner, prefix, label, target = prose_spelling(card[1]).strip(), _CARD_ACTION_PREFIX, card[2], card[3]
        else:
            continue
        label = prose_spelling(label).strip()
        if label and (not card or owner):
            found.append({"id": f"{prefix}{start}:{end}", "label": label,
                          "target": target.strip(), "action_type": "openUrl", "required": True,
                          "minimum_count": 1, "_span": (start, end), "_owner": owner})
    return found


def project_card_actions(source, evidence):
    """Bind a qualified button to one rendered Card with its exact source title."""
    from .pilot_layout_v2 import layout_text
    from .pilot_semantic_fields_v2 import _section

    expected_actions = [item for item in source_action_occurrences(source) if item["_owner"]]
    if not expected_actions:
        return evidence, []
    rows = list(evidence.evidence_ownership)
    replacements, certificates = {}, []
    for expected in expected_actions:
        title = expected["_owner"]
        if sum(item["_owner"] == title for item in expected_actions) != 1:
            continue
        titles = [i for i, row in enumerate(rows) if row.get("component_type") == "card"
                  and row.get("owner") == "generic_visible_content" and row.get("text") == title]
        if len(titles) != 1:
            continue
        index = titles[0]
        card = rows[index]
        start, end = expected["_span"]
        literal = source[start:end].strip()
        section = _section(source, literal)
        if section:
            headings = [row["text"] for row in rows[:index] if row.get("rendered_heading")
                        and card.get("parent_instance") and row.get("parent_instance") == card["parent_instance"]]
            if not headings or layout_text(headings[-1]) != section:
                continue
        buttons = [row for row in rows if row.get("component_type") == "button"
                   and card.get("render_instance") and row.get("parent_instance") == card["render_instance"]
                   and row.get("text") == expected["label"]]
        if len(buttons) != 1:
            continue
        button = buttons[0]
        actions = [action for action in evidence.output.actions if action.element_id == button["component_id"]]
        if len(actions) != 1:
            continue
        action = actions[0]
        if (action.source not in {"on.press", "on.click", "on.tap"} or action.action_type != "openUrl"
                or action.label != expected["label"] or action.url != expected["target"]):
            continue
        replacements[id(action)] = replace(action, source=_CARD_ACTION_SOURCE)
        rows[index] = {**card, "text": literal}
        certificates.append({"source_text": literal, "component_id": card["component_id"],
                             "proof": "exact_card_action"})
    if not replacements:
        return evidence, certificates
    actions = [replacements.get(id(action), action) for action in evidence.output.actions]
    semantic = tuple(replacements.get(id(action), action) for action in evidence.source_semantic_actions)
    return replace(evidence, output=replace(evidence.output, actions=actions), source_semantic_actions=semantic,
                   evidence_ownership=tuple(rows)), certificates


def source_link_occurrences(source: str, *, malformed_angles: list[int] | None = None) -> list[dict[str, Any]]:
    """Explicit labeled links, including complete standalone angle citations."""
    found = []
    in_sources = False
    explicit_heading = False
    fence = None
    offset = 0
    heading_span = None
    for number, line in enumerate(source.splitlines(keepends=True), 1):
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
        heading = _HEADING.fullmatch(raw)
        bare = source_heading(raw)
        if heading or bare:
            title = (heading[1] or heading[2]).strip() if heading else ""
            explicit_heading = heading is not None
            in_sources = bare or bool(_SOURCE_TITLE.fullmatch(title))
            # A qualified source title is meaningful visible prose. Only the
            # legacy bare section labels retain their existing optional role.
            heading_span = (start, end) if bare else None
            continue
        inline = re.match(r"(?i)^(?:source|reference|citation)\s*:\s*(.+)$", raw)
        list_prefix = re.match(r"^(?:[-*+•]\s+|\d+[.)]\s+)", raw)
        body = inline[1] if inline else raw[list_prefix.end():] if list_prefix else raw
        labeled_link = _LABELED_SOURCE.fullmatch(body)
        explicit_link_field = labeled_link is not None and re.search(r"(?i)\b(?:link|url)$", labeled_link[1])
        # A standalone labeled angle link has an explicit display label and
        # destination. Bare URLs in instructions, tables and commands do not
        # acquire a clickable role from destination text alone.
        angle_citation = bool(list_prefix and re.fullmatch(
            r"(?i)(?!(?:Action|Media):)[^:<>\[\]`|]{1,160}:\s*<https?://.*", body))
        if not in_sources and inline is None and not explicit_link_field and not angle_citation:
            continue
        match = _MARKDOWN_SOURCE.fullmatch(body) or _core._MARKDOWN_LINK_RE.fullmatch(body)
        if match is not None and match[2].count("(") != match[2].count(")"):
            match = None
        labeled = _LABELED_SOURCE.fullmatch(body) if match is None else None
        match = match or labeled
        if match is None:
            if (malformed_angles is not None
                    and re.match(r"(?i)^.+?:\s*<https?://", body)):
                malformed_angles.append(number)
            if raw and (not explicit_heading or re.match(r"(?i)^(?:Action|Media):", raw)):
                in_sources = False
                heading_span = None
            continue
        label = prose_spelling(match[1]).strip()
        target = (match[2] or match[3]).strip() if labeled else match[2].strip()
        if label and target:
            subtype = "field:" if (explicit_link_field or angle_citation) and not in_sources and inline is None else ""
            found.append({"id": f"{SOURCE_PREFIX}{subtype}{start}:{end}", "label": label,
                          "target": target, "action_type": "openUrl", "required": True,
                          "minimum_count": 1, "_span": (start, end),
                          "_heading_span": heading_span if in_sources and inline is None else None})
    return found


def source_link_owned_spans(source: str) -> set[tuple[int, int]]:
    return {span for item in source_link_occurrences(source)
            for span in (item["_span"], item["_heading_span"]) if span is not None}


def _table_ctas(source):
    """A single complete linked value in a literal two-column key/value table."""
    from .pilot_layout_v2 import _literal_source_tables

    for headers, rows, (start, end) in _literal_source_tables(source):
        if len(headers) != 2 or not all(row[0] for row in rows) or len({row[0] for row in rows}) != len(rows):
            continue
        matches = [(i, match) for i, (_, value) in enumerate(rows) if (match := _MARKDOWN_SOURCE.fullmatch(value))]
        if len(matches) != 1:
            continue
        index, match = matches[0]
        section = source[start:end]
        if section.count(match[0]) != 1 or section.count("](") != 1 or match[2].count("(") != match[2].count(")"):
            continue
        position = start + section.index(match[0])
        span = (position, position + len(match[0]))
        label = prose_spelling(match[1]).strip()
        action = {"id": f"{TABLE_ACTION_PREFIX}{position}", "label": label, "target": match[2],
                  "action_type": "openUrl", "required": True, "minimum_count": 1, "_span": span}
        rendered = (*rows[:index], (rows[index][0], label), *rows[index + 1:])
        yield (headers, rows), rendered, action


def project_table_ctas(source, evidence):
    """Certify a detached table CTA only beside its complete exact table."""
    from .pilot_layout_v2 import table_cell_spelling

    ownership = evidence.evidence_ownership
    actions = list(evidence.output.actions)
    cells, replacements = {}, {}
    for shape, rendered, expected in _table_ctas(source):
        headers, _ = shape
        wanted = tuple(tuple(map(table_cell_spelling, row)) for row in rendered)
        candidates = [table for table in evidence.output_tables
                      if tuple(table.headers) == tuple(map(table_cell_spelling, headers))
                      and tuple(tuple(table_cell_spelling(row.get(key, "")) for key in table.keys)
                                for row in table.rows) == wanted]
        if len(candidates) != 1:
            continue
        table = candidates[0]
        indices = [i for i, row in enumerate(ownership) if row.get("component_id") == table.element_id
                   and row.get("owner") == "source_table"]
        if not indices or len({ownership[i].get("render_instance") for i in indices}) != 1:
            continue
        last = indices[-1]
        if last + 1 >= len(ownership):
            continue
        table_row, button = ownership[last:last + 2]
        table_instance = str(table_row.get("render_instance", "")).split(":", 2)
        button_instance = str(button.get("render_instance", "")).split(":", 2)
        if (len(table_instance) != 3 or len(button_instance) != 3
                or button.get("component_type") != "button" or button.get("text") != expected["label"]
                or not table_row.get("parent_instance") or table_row["parent_instance"] != button.get("parent_instance")
                or int(button_instance[1]) != int(table_instance[1]) + 1):
            continue
        matches = [i for i, action in enumerate(actions) if action.element_id == button["component_id"]]
        if len(matches) != 1:
            continue
        index = matches[0]
        action = actions[index]
        if (action.source not in {"on.press", "on.click", "on.tap"} or action.action_type != "openUrl"
                or action.label != expected["label"] or action.url != expected["target"]):
            continue
        actions[index] = replacements[id(action)] = replace(action, source=_TABLE_ACTION_SOURCE)
        cells[shape] = rendered
    if not replacements:
        return evidence, cells
    semantic = tuple(replacements.get(id(action), action) for action in evidence.source_semantic_actions)
    return replace(evidence, output=replace(evidence.output, actions=actions), source_semantic_actions=semantic), cells


def _display_url_spans(source, expected_tables):
    """Retire URL-only guesses only for literal, table-owned original URLs."""
    if "Original URL" not in source:
        return set()
    from .pilot_layout_v2 import _literal_source_tables

    spans = set()
    for headers, rows, (start, end) in _literal_source_tables(source):
        if headers.count("Original URL") != 1:
            continue
        shape = (tuple(map(_core.normalize_markdown, headers)),
                 tuple(tuple(map(_core.normalize_markdown, row)) for row in rows))
        owners = [table for table in expected_tables
                  if table.get("required", True) and table.get("minimum_count", 1) == 1
                  and table.get("representation_policy") == "table_only"
                  and (tuple(table.get("headers") or ()), tuple(map(tuple, table.get("rows") or ()))) == shape]
        if len(owners) != 1:
            continue
        column = headers.index("Original URL")
        heading, separator, *lines = source[start:end].splitlines(keepends=True)
        offset = start + len(heading) + len(separator)
        for line, row in zip(lines, rows, strict=True):
            url = row[column]
            if re.fullmatch(r"https://[^\s<>|\[\]]+", url) and line.count(url) == 1:
                at = offset + line.index(url)
                spans.add((at, at + len(url)))
            offset += len(line)
    return spans


def source_contract_v2(source: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(contract))
    links = source_link_occurrences(source)
    declared_actions = source_action_occurrences(source)
    source_actions = [item for item in declared_actions if not item["_owner"]]
    card_actions = [item for item in declared_actions if item["_owner"]]
    table_actions = [action for _, _, action in _table_ctas(source)]
    media = declared_media_occurrences(source)
    from pipeline.renderer_effective_semantics_v5_4 import canonical_media_kind

    def media_key(item):
        return canonical_media_kind(item.get("kind")), _core.normalize_url(item.get("url"))

    existing_media = Counter()
    for item in result.get("media") or []:
        if item.get("required", True):
            existing_media[media_key(item)] += int(item.get("minimum_count", 1) or 0)
    missing_media = Counter()
    for item in media:
        key = media_key(item)
        if existing_media[key]:
            existing_media[key] -= 1
        else:
            missing_media[(item["kind"], item["url"])] += 1
    result.setdefault("media", [])
    for (kind, url), count in missing_media.items():
        result["media"].append({"kind": kind, "url": url, "required": True,
                                "minimum_count": count, "media_policy": "exact"})
    claimed_actions: set[int] = set()
    # Preserve every existing explicit action requirement, adding role identity.
    actions = []
    for index, action in enumerate(result.get("actions") or []):
        item = dict(action)
        matching = [position for position, occurrence in enumerate(source_actions)
                    if _core.normalize_markdown(occurrence["label"]) == _core.normalize_markdown(item.get("label"))
                    and _core.normalize_url(occurrence["target"]) == _core.normalize_url(item.get("target"))]
        raw_pairs = {(source_actions[position]["label"], source_actions[position]["target"])
                     for position in matching}
        identity = str(item.get("id") or index)
        if len(raw_pairs) == 1:
            item["id"] = ACTION_PREFIX + identity
            item["label"], item["target"] = next(iter(raw_pairs))
            count = max(0, int(item.get("minimum_count", 1) or 0))
            claimed_actions.update([position for position in matching if position not in claimed_actions][:count])
        else:
            # The legacy extractor may collapse fragments or unsupported
            # source syntax. Never reconstruct a destination by guessing.
            item["id"] = ACTION_PREFIX + "unproven:" + identity
        actions.append(item)
    actions.extend({key: value for key, value in item.items() if not key.startswith("_")}
                   for position, item in enumerate(source_actions) if position not in claimed_actions)
    targets = {_core.normalize_url(item["target"]) for item in [*links, *declared_actions, *table_actions]}
    targets.update(_core.normalize_url(item["url"]) for item in media)
    proven_spans = {item["_span"] for item in [*links, *declared_actions, *table_actions, *media]}
    proven_spans.update(_display_url_spans(source, result.get("tables") or []))
    url_occurrences = list(_core._URL_RE.finditer(source))
    # Inspect the legacy scanner's own spans when retiring its URL-only
    # guesses. It can stop *inside* a proven destination at a close-paren.
    # Every occurrence of that guess must be owned before it can disappear.
    def owned_url(match):
        return any(start <= match.start() and match.end() <= end for start, end in proven_spans)
    def fully_owned_url(raw):
        occurrences = [match for match in url_occurrences if _core.normalize_url(match[0]) == _core.normalize_url(raw)]
        return bool(occurrences) and all(owned_url(match) for match in occurrences)
    # Replace only the extractor's loose URL-only entries. Rich persisted
    # requirements retain their original count/label/component constraints.
    result["source_links"] = [item for item in result.get("source_links") or []
                              if not (isinstance(item, Mapping) and set(item) == {"url"}
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
    result["actions"] = [*actions, *({key: value for key, value in item.items() if not key.startswith("_")}
                                   for item in [*links, *table_actions, *card_actions])]
    # Source-heading/link lines are metadata owned by these exact requirements,
    # not duplicate prose. Remove only the concrete proven occurrences.
    units = list(result.get("content_units") or [])
    if any("_record" in item for item in media):
        visible = inline_media_view(source, media=media)
        inline = {line.strip(): display.strip() for line, display in
                  zip(source.splitlines(), visible.splitlines(), strict=True) if line.strip() != display.strip()}
        units = [inline.get(unit, unit) for unit in units]
    headings = list(result.get("headings") or [])
    owned_spans = source_link_owned_spans(source) | {item["_span"] for item in [*source_actions, *media]}
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
    """Undo legacy label and URL normalization on source-owned requirements.

    SourceContract conversion strips URL fragments. The immutable source hash,
    role ID and original target retained in this profile's contract bind the
    restored bytes; matching indices are intentionally not used by this profile.
    """
    occurrences = {item["id"]: item for item in contract.get("actions", [])
                   if isinstance(item, Mapping) and item.get("id") and isinstance(item.get("target"), str)}
    def restore(item):
        occurrence = occurrences.get(item.id)
        return replace(item, url=occurrence["target"], label=occurrence.get("label", item.label)) if occurrence else item
    return replace(base,
                   expected_actions=tuple(restore(item) for item in base.expected_actions),
                   source_contract=replace(base.source_contract,
                       explicit_actions=[restore(item) for item in base.source_contract.explicit_actions]))


def _role(item: Any, *, expected: bool) -> str:
    if expected:
        if item.id.startswith(SOURCE_PREFIX):
            return "source_link"
        if item.id.startswith(TABLE_ACTION_PREFIX):
            return "table_action"
        if item.id.startswith(_CARD_ACTION_PREFIX):
            return "card_action"
        if item.id.startswith(ACTION_PREFIX):
            return "action"
        return ""
    if item.source == _TABLE_ACTION_SOURCE:
        return "table_action"
    if item.source == _CARD_ACTION_SOURCE:
        return "card_action"
    return "source_link" if item.source == "source_link" else "action"


def _label(value: str) -> str:
    return " ".join(value.split())


def action_match_v2(expected: Any, actual: Any, aliases: Any) -> float:
    role = _role(expected, expected=True)
    if role:
        if expected.id.startswith(ACTION_PREFIX + "unproven:"):
            return 0.0
        actual_role = _role(actual, expected=False)
        # A source-declared labeled link can be a normal List link. The exact
        # label, raw destination, type and occurrence count still bind it;
        # arbitrary Buttons/actions do not acquire this reference role.
        if expected.id.startswith(SOURCE_PREFIX + "field:") and actual.source == "list.item":
            actual_role = "source_link"
        if role != actual_role or _label(expected.label) != _label(actual.label):
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


def project_source_link_labels(source, ownership, actions):
    """Assign only an exact declared link's visible List label to that role."""
    expected = Counter((item["label"], item["target"]) for item in source_link_occurrences(source)
                       if item["id"].startswith(SOURCE_PREFIX + "field:"))
    matched = Counter()
    for action in actions:
        key = (action.label, action.url)
        if action.source == "list.item" and expected[key]:
            expected[key] -= 1
            matched[(action.element_id.rsplit("[", 1)[0], action.label)] += 1
    visible = Counter((row.get("component_id"), row.get("text")) for row in ownership
                      if row.get("component_type") == "list" and row.get("owner") == "generic_visible_content")
    return [{**row, "owner": "source_action"} if (
                row.get("component_type") == "list" and row.get("owner") == "generic_visible_content"
                and matched[(row.get("component_id"), row.get("text"))]
                == visible[(row.get("component_id"), row.get("text"))]) else row
            for row in ownership]


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
