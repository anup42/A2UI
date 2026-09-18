"""Conservative, source-proven v10 repairs and renderer-aware archive gates.

No generator, network, inferred facts, new controls, or new layout nodes. A
failed gate means quarantine for review, not proof that every paraphrase is bad.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from ir_training.data.archive_recovery import _strings, text_sha256

POLICY = "messages-archive-semantic-review-v10-20260916"
REFERENCE = re.compile(r"\[(?:[A-Z_]+URL|URL|[A-Z_]+ASSET)_\d+\]")
BUTTON = re.compile(r"\[Button:\s*([^\]]+)\]\s*<?(\[(?:ACTION_URL|SOURCE_URL|URL)_\d+\])", re.I)
CITATION = re.compile(r"^\s*[-*]?\s*([^\n<>]+?):\s*<?(\[(?:SOURCE_URL|ACTION_URL|URL)_\d+\])>?\s*$", re.M)
STOP = set("a an the and or to of for in on at by is are be it its this that with as from your you into than then can will have has these those they their which how what when".split())
FALSE_ASSET_PROSE = {"asset/liability", "assets/liabilities"}


def plain(text: str) -> str:
    text = text.replace("\\sum", "Σ").replace("\\rightarrow", "→")
    text = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)
    return " ".join(text.replace("**", "").replace("__", "").replace("$", "").split())


def words(text: str) -> list[str]:
    return re.findall(r"\w+", plain(text).casefold())


def norm(text: str) -> str:
    return " ".join(words(text))


def subsequence(needle: list[str], haystack: list[str]) -> bool:
    iterator = iter(haystack)
    return all(any(item == wanted for item in iterator) for wanted in needle)


def pointer(state: Any, path: str) -> Any:
    """Mirror Android JSON-pointer traversal; do not fall back to literal keys."""
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    node = state
    for token in path[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            node = node.get(token)
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return None
    return node


def table_rows(props: dict, state: dict) -> list | None:
    if isinstance(props.get("rows"), list):
        return props["rows"]
    value = pointer(state, props.get("statePath"))
    if isinstance(value, list):
        return value
    payload = props.get("table")
    return payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else None


def columns(props: dict) -> list[tuple[str, str]]:
    result = []
    for col in props.get("columns", []):
        if isinstance(col, str):
            result.append((col, col))
        elif isinstance(col, dict):
            key = str(col.get("key", ""))
            result.append((key, str(col.get("label", key))))
    return result


def visible_strings(graph: dict) -> list[str]:
    """Explicit Table projection, not every arbitrary string in unused state.

    Native inferred-column/restaurant tables are treated conservatively; their
    row fields can be consumed implicitly. This is not a pixel renderer.
    """
    result = []
    state = graph.get("state", {})
    for element in graph["elements"].values():
        props = element.get("props", {})
        if element.get("type") != "Table":
            result.extend(_strings(props))
            for value in _strings(props):
                if value.startswith("/"):
                    result.extend(_strings(pointer(state, value)))
            result.extend(_strings(element.get("on", {})))
            continue
        cols = columns(props)
        result.extend(label for _, label in cols)
        result.extend(_strings(props.get("entityMedia", {})))
        rows = table_rows(props, state) or []
        if not cols or props.get("domain") == "restaurants":
            result.extend(_strings(rows))
        else:
            for row in rows:
                if isinstance(row, dict):
                    for key, _ in cols:
                        result.extend(_strings(row.get(key)))
                elif isinstance(row, list):
                    result.extend(_strings(row[:len(cols)]))
    return result


def source_units(source: str) -> list[str]:
    source = re.sub(r"```[\s\S]*?```", "", source)
    units = []
    for raw in source.splitlines():
        line = raw.strip()
        if not line or "|" in line or REFERENCE.search(line):
            continue
        if re.match(r"^(?:#{1,6}\s|Media\s*:|Action\s*:|Formula\s*:|Sources?\s*:|Quick Actions|Chart Title\s*:|[XY]-Axis\s*:)", line, re.I):
            continue
        line = plain(line)
        if len(line) >= 45 and len(words(line)) >= 8:
            units.append(line)
    return list(dict.fromkeys(units))


def source_headers(source: str) -> list[list[str]]:
    lines = source.splitlines()
    result = []
    for index, raw in enumerate(lines[:-1]):
        if "|" not in raw or re.fullmatch(r"[\s|:–-]+", raw):
            continue
        if re.fullmatch(r"[\s|:–-]+", lines[index + 1]) and "|" in lines[index + 1]:
            result.append([plain(x) for x in raw.strip().strip("|").split("|")])
    return result


def event_urls(element: dict) -> list[dict]:
    """Return existing event dictionaries only; never manufacture an action."""
    found = []
    for events in (element.get("on", {}), element.get("props", {}).get("on", {})):
        if isinstance(events, dict):
            for action in events.values():
                if isinstance(action, dict) and str(action.get("action", "")).lower() == "openurl" and isinstance(action.get("params"), dict):
                    found.append(action)
    return found


def citation_label(value: str) -> str:
    return norm(value).replace(" dept ", " department ")


def repair_graph(source: str, graph: dict, url_map: dict | None = None):
    fixed, mapping = deepcopy(graph), deepcopy(url_map or {})
    proofs = []
    def proof(kind, location, before, after):
        proofs.append({"kind": kind, "location": location, "before": before, "after": after})

    # Undo only the audited lexical false-positive, with retained exact mapping.
    for token, entry in list(mapping.items()):
        raw = entry.get("url", "") if isinstance(entry, dict) else ""
        if raw.casefold() not in FALSE_ASSET_PROSE or token not in source:
            continue
        unsafe = any(token in str(e.get("props", {})) for e in fixed["elements"].values() if e.get("type") in {"Image", "Icon", "Video", "AudioPlayer"})
        unsafe |= any(token in str(action) for e in fixed["elements"].values() for action in event_urls(e))
        if unsafe:
            continue
        def unmask(value):
            if isinstance(value, str):
                return value.replace(token, raw)
            if isinstance(value, dict):
                return {k: unmask(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unmask(v) for v in value]
            return value
        source = source.replace(token, raw)
        fixed = unmask(fixed)
        mapping.pop(token)
        proof("proven_false_asset_mask", token, token, raw)

    headers = source_headers(source)
    units = source_units(source)
    unit_words = [(u, words(u)) for u in units]
    visible_wordset = set(words("\n".join(visible_strings(fixed))))
    citations = defaultdict(set)
    action_labels = {citation_label(m[1]) for m in BUTTON.finditer(source)}
    for match in CITATION.finditer(source):
        if "button:" not in match[1].lower():
            citations[citation_label(match[1])].add(match[2])
    for element_id, element in fixed["elements"].items():
        props, kind = element.get("props", {}), element.get("type")
        if kind == "Table":
            path = props.get("statePath")
            if isinstance(path, str) and table_rows(props, fixed.get("state", {})) is None and isinstance(fixed.get("state", {}).get(path), list):
                escaped = "/" + path.replace("~", "~0").replace("/", "~1")
                if pointer(fixed["state"], escaped) == fixed["state"][path]:
                    props["statePath"] = escaped
                    proof("literal_state_key_pointer_escape", element_id, path, escaped)
            rows = table_rows(props, fixed.get("state", {}))
            cols = columns(props)
            if rows and all(isinstance(row, dict) for row in rows) and cols:
                used = {props.get("primaryColumn")} | set(props.get("highlightColumns", [])) | set(props.get("numericColumns", []))
                removable = {key for key, _ in cols if key in {"image", "imageAlt"} and key not in used and all(row.get(key) in (None, "") for row in rows)}
                if not re.search(r"\[(?:IMAGE|MEDIA)_(?:URL|ASSET)_\d+\]|Media:\s*Image", source, re.I) and removable and len(cols) > len(removable):
                    before = deepcopy(props["columns"])
                    props["columns"] = [col for col in props["columns"] if (col if isinstance(col, str) else col.get("key")) not in removable]
                    proof("remove_unsupported_empty_media_columns", element_id, before, props["columns"])
                    cols = columns(props)
            # Require an unambiguous matching table, not a global label guess.
            labels = {norm(label) for _, label in cols}
            matches = [header for header in headers if len(header) == len(cols) and len(labels & {norm(x) for x in header}) >= max(2, len(cols) - 2)]
            if len(matches) == 1:
                for col in props.get("columns", []):
                    if not isinstance(col, dict):
                        continue
                    label = str(col.get("label", col.get("key", "")))
                    exact = [x for x in matches[0] if norm(x) == norm(label)]
                    expansions = [x for x in matches[0] if len(norm(label)) >= 3 and norm(label) != norm(x) and re.search(r"(?:^| )" + re.escape(norm(label)) + r"(?: |$)", norm(x))]
                    if not exact and len(expansions) == 1:
                        col["label"] = expansions[0]
                        proof("source_table_label_restoration", element_id, label, expansions[0])
        if kind == "Text" and isinstance(props.get("text"), str):
            original = props["text"]
            parts = original.split("\n")
            for index, part in enumerate(parts):
                tokens = words(part)
                if len(tokens) < 8 or len(part) < 32:
                    continue
                candidates = [(u, uw) for u, uw in unit_words if len(uw) > len(tokens) and len(uw) <= len(tokens) * 2.5 and subsequence(tokens, uw)]
                if len(candidates) == 1:
                    full, fullwords = candidates[0]
                    if len(set(fullwords) - visible_wordset - STOP) >= 2:
                        parts[index] = full
                        proof("unique_source_clause_restoration", f"{element_id}.text[{index}]", part, full)
            props["text"] = "\n".join(parts)
        if kind == "Button":
            label = citation_label(str(props.get("label", "")))
            actions = event_urls(element)
            if label not in action_labels and len(citations.get(label, set())) == 1 and len(actions) == 1:
                expected = next(iter(citations[label]))
                old = actions[0]["params"].get("url")
                if isinstance(old, str) and REFERENCE.fullmatch(old) and old != expected:
                    actions[0]["params"]["url"] = expected
                    proof("source_proven_citation_binding", element_id, old, expected)

    source_normalized = norm(source)
    def codec(value, location="", protected=False):
        if isinstance(value, dict):
            protected |= value.get("type") in {"CodeBlock", "ConsoleLog", "Formula"}
            return {k: codec(v, location + "/" + str(k), protected or k in {"on", "watch"}) for k, v in value.items()}
        if isinstance(value, list):
            return [codec(v, f"{location}/{i}", protected) for i, v in enumerate(value)]
        if isinstance(value, str) and not protected and re.search(r"[\u2500-\u259f]", value):
            # The old mojibake score also counts legitimate Greek letters;
            # require an exact inverse AND exact source support instead.
            try:
                recovered = value.encode("cp437").decode("utf-8")
                inverse = recovered.encode("utf-8").decode("cp437")
            except (UnicodeEncodeError, UnicodeDecodeError):
                return value
            if recovered != value and inverse == value and len(norm(recovered)) >= 3 and norm(recovered) in source_normalized:
                proof("source_proven_partial_codec_repair", location, value, recovered)
                return recovered
        return value
    fixed = codec(fixed)
    return source, fixed, mapping, proofs


@lru_cache(maxsize=1)
def reviewed_findings():
    path = Path(__file__).resolve().parents[3] / "data/quality/v9_manual100_findings.json"
    return json.loads(path.read_text(encoding="utf-8"))["findings"]


def review_graph(source: str, graph: dict, *, original_source_sha256="", reviewed=None):
    issues = []
    def issue(code, detail):
        issues.append({"code": code, "detail": detail})
    visible = visible_strings(graph)
    visible_text = "\n".join(visible)
    visible_tokens = set(REFERENCE.findall(visible_text))
    visible_words = set(words(visible_text))
    visible_normalized = norm(visible_text)
    action_tokens = {token for e in graph["elements"].values() for action in event_urls(e) for token in REFERENCE.findall(str(action["params"].get("url", "")))}
    for match in BUTTON.finditer(source):
        if match[2] not in action_tokens and match[2] not in visible_tokens:
            issue("inline_action_not_bound", {"label": match[1], "reference": match[2]})
    for match in CITATION.finditer(source):
        if "button:" not in match[1].lower() and match[2] not in visible_tokens:
            issue("source_citation_not_visible", {"label": match[1], "reference": match[2]})
    for token in sorted(set(re.findall(r"\[(?:ICON|IMAGE|MEDIA)_(?:URL|ASSET)_\d+\]", source)) - visible_tokens):
        issue("source_media_not_bound", token)
    for element_id, e in graph["elements"].items():
        props = e.get("props", {})
        if e.get("type") == "Table":
            rows = table_rows(props, graph.get("state", {}))
            if props.get("statePath") and rows is None:
                issue("unresolved_table_state_path", {"element": element_id, "path": props["statePath"]})
            cols = columns(props)
            if rows and cols:
                keys = {key for key, _ in cols}
                for index, row in enumerate(rows):
                    if isinstance(row, dict):
                        hidden = {k: v for k, v in row.items() if k not in keys and v not in (None, "", [], {})}
                        hidden = {k: v for k, v in hidden.items() if re.match(r"(?:alert_?\d|date_?\d)", k, re.I)}
                        if hidden:
                            issue("hidden_table_date_column", {"element": element_id, "row": index, "fields": hidden})
        if e.get("type") == "TextField" and re.fullmatch(r"(?:Auto-filled from QR|Optional short-text field)", str(props.get("value", "")), re.I):
            issue("description_used_as_input_value", element_id)
        if e.get("type") not in {"CodeBlock", "ConsoleLog"} and any(re.search(r"[╬╠╩╦]", s) for s in _strings(props)):
            issue("residual_suspicious_encoding", element_id)
    if any(re.search(r"[╬╠╩╦]", s) for s in visible):
        issue("residual_suspicious_encoding", "Renderer-visible props or table values")
    for unit in source_units(source):
        substantive = set(words(unit)) - STOP
        missing = substantive - visible_words
        # Pure table-introduction prose is not an additional required fact if
        # every numeric anchor survives. Never exempt instructions/caveats.
        table_intro = bool(re.match(r"The following (?:table|data) (?:details|represents|shows|lists)\b", unit, re.I)) and {w for w in words(unit) if any(c.isdigit() for c in w)} <= visible_words
        if not table_intro and len(substantive) >= 6 and len(missing) >= 3 and len(missing) / len(substantive) >= 0.20:
            issue("source_clause_gap", {"source": unit, "missing_words": sorted(missing)})
        for qualifier in ("at least", "at most", "only after", "before any work", "unless upgraded", "remaining 50", "simultaneous"):
            if qualifier in norm(unit) and qualifier not in visible_normalized:
                issue("lost_source_qualifier", {"source": unit, "qualifier": qualifier})
    if re.search(r"\bline chart\b", source, re.I) and re.search(r"Chart Title:|X-Axis:|Y-Axis:|Q4.{0,40}(?:highlight|peak)", source, re.I) and not any(e.get("type") == "Chart" for e in graph["elements"].values()):
        issue("requested_chart_missing", "Explicit chart specification without a Chart component")
    # Narrow source-risk patterns, never automatic factual rewrites.
    if re.search(r"\|\s*\*?\*?Analytics\*?\*?\s*\|\s*No tracking capabilities", source, re.I) and "Static QR" in source:
        issue("source_static_qr_tracking_claim", "Unsupported absolute no-tracking premise")
    if "GDPR" in source and re.search(r"Confirm erasure within 30 days", source, re.I):
        issue("source_gdpr_fixed_deadline", "Source legal checklist needs qualified review")
    if re.search(r"allergic to shellfish[^\n]*\n[^\n]*甲殻類", source, re.I) and "貝類" not in source:
        issue("source_allergy_scope_mismatch", "Shellfish translated only as crustaceans")
    # Hash-bound audit decisions prevent known defects slipping through a
    # heuristic. Source-risk holds extend to all targets for that same source.
    for finding in reviewed if reviewed is not None else reviewed_findings():
        if original_source_sha256 != finding["original_source_sha256"] and text_sha256(source) != finding["effective_source_sha256"]:
            continue
        if finding["disposition"] == "source_review":
            issue("reviewed_source_requires_verification", {"case": finding["case"], "reason": finding["reason"]})
        else:
            for fragment in finding.get("required_visible_fragments", []):
                if not subsequence(words(fragment), words(visible_text)):
                    issue("reviewed_source_fragment_missing", {"case": finding["case"], "fragment": fragment})
            # Regeneration cases are also covered by actual action/media/field
            # gates above. Do not blacklist a later, genuinely fixed target.
    return issues


def process_graph(source: str, graph: dict, *, url_map=None, original_source_sha256="", reviewed=None):
    fixed_source, fixed, mapping, proofs = repair_graph(source, graph, url_map)
    issues = review_graph(fixed_source, fixed, original_source_sha256=original_source_sha256, reviewed=reviewed)
    return {"source": fixed_source, "graph": fixed, "url_map": mapping, "proofs": proofs, "issues": issues,
            "changes": dict(Counter(p["kind"] for p in proofs))}
