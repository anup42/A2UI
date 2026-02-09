from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

_stopwords = {
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are","was","were","be","as","it","this","that"
}

_URL_RE = re.compile(r"https?://[^\s\]\)\}<>\"']+")
_MARKDOWN_LINE_RE = re.compile(r"^\s*(?:[-*]\s+|#+\s+|\|).*")

_CONTAINER_COMPONENTS = {
    "Column",
    "Row",
    "List",
    "Card",
    "Tabs",
    "Tab",
    "Modal",
    "Grid",
    "Stack",
    "Section",
    "Container",
}

_HEADING_VARIANTS = {"h1", "h2", "h3", "headline", "title", "subtitle"}

_INTENT_TABLE_REQUIRED = {
    "comparison",
    "data_visualisation",
    "data_visualization",
    "calculation",
}

_INTENT_ACTION_REQUIRED = {
    "booking",
    "travel",
    "navigation",
    "event_schedule",
    "product_lookup",
    "status_check",
}

_INTENT_SECTION_REQUIRED = {
    "planning",
    "travel",
    "event_schedule",
    "documentation",
    "technical_support",
    "education",
    "recipe",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_intent(intent: str | None) -> str:
    if not intent:
        return "unknown"
    value = intent.strip().lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    aliases = {
        "data_visualisation": "data_visualization",
        "creating_writing": "creative_writing",
        "information_retrieval": "information_retrieval",
    }
    return aliases.get(value, value) or "unknown"


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    return [_normalize_intent(tag) for tag in tags if isinstance(tag, str)]


def _extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    cleaned: list[str] = []
    for url in urls:
        cleaned_url = url.rstrip(".,);:")
        cleaned.append(cleaned_url)
    return cleaned


def _iter_components(genui_json: Any) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    if isinstance(genui_json, list):
        for msg in genui_json:
            if not isinstance(msg, dict):
                continue
            update = msg.get("updateComponents") or msg.get("surfaceUpdate")
            if isinstance(update, dict):
                comps = update.get("components")
                if isinstance(comps, list):
                    components.extend([c for c in comps if isinstance(c, dict)])
    elif isinstance(genui_json, dict):
        update = genui_json.get("updateComponents") or genui_json.get("surfaceUpdate")
        if isinstance(update, dict):
            comps = update.get("components")
            if isinstance(comps, list):
                components.extend([c for c in comps if isinstance(c, dict)])
    return components


def _build_component_index(components: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for comp in components:
        comp_id = comp.get("id")
        if isinstance(comp_id, str):
            index[comp_id] = comp
    return index


def _component_children(comp: dict[str, Any]) -> list[str]:
    children: list[str] = []
    child = comp.get("child")
    if isinstance(child, str):
        children.append(child)
    comp_children = comp.get("children")
    if isinstance(comp_children, list):
        for item in comp_children:
            if isinstance(item, str):
                children.append(item)
    return children


def _compute_tree_depths(components: list[dict[str, Any]]) -> tuple[int, float]:
    if not components:
        return 0, 0.0
    index = _build_component_index(components)
    children_map: dict[str, list[str]] = {}
    parents: set[str] = set()
    for comp_id, comp in index.items():
        children = _component_children(comp)
        children_map[comp_id] = children
        parents.update(children)
    roots = [comp_id for comp_id in index.keys() if comp_id not in parents]
    if not roots:
        roots = list(index.keys())

    max_depth = 0
    depths: dict[str, int] = {}

    def dfs(node_id: str, stack: set[str]) -> int:
        nonlocal max_depth
        if node_id in stack:
            return 1
        if node_id in depths:
            return depths[node_id]
        stack.add(node_id)
        child_depths = []
        for child_id in children_map.get(node_id, []):
            if child_id in index:
                child_depths.append(dfs(child_id, stack))
        stack.remove(node_id)
        depth = 1 + (max(child_depths) if child_depths else 0)
        depths[node_id] = depth
        max_depth = max(max_depth, depth)
        return depth

    for root_id in roots:
        dfs(root_id, set())

    avg_depth = sum(depths.values()) / len(depths) if depths else 0.0
    return max_depth, avg_depth


def _compute_reference_integrity(components: list[dict[str, Any]]) -> dict[str, float]:
    index = _build_component_index(components)
    total_components = len(index)
    if total_components == 0:
        return {
            "missing_ids": 0.0,
            "missing_ids_rate": 0.0,
            "dangling_components": 0.0,
            "dangling_components_rate": 0.0,
        }

    referenced_ids: set[str] = set()
    missing_refs = 0
    total_refs = 0
    for comp in components:
        for child_id in _component_children(comp):
            total_refs += 1
            referenced_ids.add(child_id)
            if child_id not in index:
                missing_refs += 1

    roots = [comp_id for comp_id in index.keys() if comp_id not in referenced_ids]
    if not roots:
        roots = list(index.keys())

    reachable: set[str] = set()

    def dfs(node_id: str) -> None:
        if node_id in reachable:
            return
        reachable.add(node_id)
        for child_id in _component_children(index.get(node_id, {})):
            if child_id in index:
                dfs(child_id)

    for root_id in roots:
        dfs(root_id)

    dangling_components = max(0, total_components - len(reachable))
    missing_rate = missing_refs / total_refs if total_refs > 0 else 0.0
    dangling_rate = dangling_components / total_components if total_components > 0 else 0.0

    return {
        "missing_ids": float(missing_refs),
        "missing_ids_rate": float(missing_rate),
        "dangling_components": float(dangling_components),
        "dangling_components_rate": float(dangling_rate),
    }


def _extract_expected_headings(response_text: str) -> list[str]:
    headings: list[str] = []
    if not response_text:
        return headings
    lines = response_text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        md_match = re.match(r"^#{1,3}\s+(.*)", stripped)
        if md_match:
            headings.append(md_match.group(1).strip())
            continue
        if stripped.endswith(":") and len(stripped.split()) <= 6:
            headings.append(stripped[:-1].strip())
            continue
        if stripped.isupper() and len(stripped.split()) <= 6:
            headings.append(stripped.title())
            continue
        if stripped.lower() in {
            "summary",
            "assumptions",
            "comparison table",
            "quick actions",
            "sources",
            "recommendation",
            "itinerary",
            "notes",
            "icons",
            "images",
        }:
            headings.append(stripped)
    return headings


def _extract_text_nodes(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in components if c.get("component") == "Text"]


def _extract_button_actions(components: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for comp in components:
        if comp.get("component") != "Button":
            continue
        action = comp.get("action")
        if not isinstance(action, dict):
            continue
        func = action.get("functionCall")
        if not isinstance(func, dict):
            continue
        if func.get("call") != "openUrl":
            continue
        args = func.get("args")
        if isinstance(args, dict):
            url = args.get("url")
            if isinstance(url, str):
                urls.append(url)
    return urls


def _row_text_cell_ids(row_comp: dict[str, Any], index: dict[str, dict[str, Any]]) -> list[str]:
    if row_comp.get("component") != "Row":
        return []
    cells: list[str] = []
    for child_id in _component_children(row_comp):
        child = index.get(child_id)
        if child and child.get("component") == "Text":
            cells.append(child_id)
    return cells


def _append_if_table_like(rows_out: list[list[str]], row_nodes: list[list[str]], min_rows: int) -> None:
    if len(row_nodes) < min_rows:
        return
    counts = [len(row) for row in row_nodes]
    if not counts:
        return
    # Real tables should have stable columns and at least two columns.
    if min(counts) < 2:
        return
    if len(set(counts)) != 1:
        return
    rows_out.extend(row_nodes)


def _find_table_rows(components: list[dict[str, Any]]) -> list[list[str]]:
    index = _build_component_index(components)
    rows: list[list[str]] = []

    # Pattern A: List(direction=vertical) -> Row children
    for comp in components:
        if comp.get("component") != "List":
            continue
        if comp.get("direction") not in (None, "vertical"):
            continue
        list_children = _component_children(comp)
        row_nodes: list[list[str]] = []
        for child_id in list_children:
            child = index.get(child_id)
            if not child:
                continue
            row_cells = _row_text_cell_ids(child, index)
            if row_cells:
                row_nodes.append(row_cells)
        _append_if_table_like(rows, row_nodes, min_rows=2)

    # Pattern B: Column with divider-separated Row children
    for comp in components:
        if comp.get("component") != "Column":
            continue
        column_children = _component_children(comp)
        seq: list[list[str]] = []
        for child_id in column_children:
            child = index.get(child_id)
            if not child:
                # Unknown refs break sequence.
                _append_if_table_like(rows, seq, min_rows=3)
                seq = []
                continue
            child_type = child.get("component")
            if child_type == "Row":
                row_cells = _row_text_cell_ids(child, index)
                if row_cells:
                    seq.append(row_cells)
                else:
                    _append_if_table_like(rows, seq, min_rows=3)
                    seq = []
            elif child_type == "Divider":
                # Allow divider separators within a table block.
                continue
            else:
                _append_if_table_like(rows, seq, min_rows=3)
                seq = []
        _append_if_table_like(rows, seq, min_rows=3)

    return rows


def _extract_table_cells_from_response(response_text: str) -> list[str]:
    cells: list[str] = []
    if not response_text:
        return cells

    all_lines = [line.rstrip() for line in response_text.splitlines() if line.strip()]

    # Pattern A: markdown/pipe-style tables.
    pipe_lines = [line.strip() for line in all_lines if "|" in line]
    if len(pipe_lines) >= 2:
        for line in pipe_lines:
            if re.match(r"^\s*\|?\s*-+\s*(\|\s*-+\s*)+\|?\s*$", line):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            parts = [p for p in parts if p]
            cells.extend(parts)

    # Pattern B: key/value table-like blocks (e.g., "Metric: Value") with >=3 rows.
    kv_rows: list[tuple[str, str]] = []
    for raw_line in all_lines:
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw_line).strip()
        match = re.match(r"^([^:|]{1,80}):\s+(.+)$", line)
        if not match:
            if len(kv_rows) >= 3:
                for key, value in kv_rows:
                    cells.extend([key, value])
            kv_rows = []
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key and value and len(key.split()) <= 10:
            kv_rows.append((key, value))
        else:
            if len(kv_rows) >= 3:
                for key2, value2 in kv_rows:
                    cells.extend([key2, value2])
            kv_rows = []
    if len(kv_rows) >= 3:
        for key, value in kv_rows:
            cells.extend([key, value])

    return cells


def _extract_table_cells_from_ir(table_rows: list[list[str]], index: dict[str, dict[str, Any]]) -> list[str]:
    cells: list[str] = []
    for row in table_rows:
        for cell_id in row:
            comp = index.get(cell_id)
            if not comp:
                continue
            if comp.get("component") == "Text":
                text = comp.get("text")
                if isinstance(text, str):
                    cells.append(text)
    return cells


def _markdown_leakage_rate(text_nodes: list[dict[str, Any]]) -> float:
    if not text_nodes:
        return 0.0
    total_lines = 0
    markdown_lines = 0
    for node in text_nodes:
        text = node.get("text")
        if not isinstance(text, str):
            continue
        lines = text.splitlines()
        total_lines += len(lines)
        for line in lines:
            if _MARKDOWN_LINE_RE.match(line):
                markdown_lines += 1
    if total_lines == 0:
        return 0.0
    return markdown_lines / total_lines


def compute_intent_metrics(
    intent: str | None,
    tags: list[str] | None,
    response_text: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    intent_bucket = _normalize_intent(intent)
    tag_buckets = set(_normalize_tags(tags))
    response_lower = response_text.lower() if response_text else ""

    has_urls = bool(_extract_urls(response_text))
    has_button_marker = "quick actions" in response_lower or "[button" in response_lower
    table_like = bool(_extract_table_cells_from_response(response_text))
    expected_headings = _extract_expected_headings(response_text)

    table_expected = (
        intent_bucket in _INTENT_TABLE_REQUIRED
        or bool(tag_buckets & _INTENT_TABLE_REQUIRED)
        or table_like
    )
    actions_expected = (
        has_urls
        and (
            intent_bucket in _INTENT_ACTION_REQUIRED
            or bool(tag_buckets & _INTENT_ACTION_REQUIRED)
            or has_button_marker
        )
    )
    sections_expected = (
        intent_bucket in _INTENT_SECTION_REQUIRED
        or bool(tag_buckets & _INTENT_SECTION_REQUIRED)
        or len(expected_headings) >= 2
    )

    table_ok = metrics.get("table_pattern_detected", 0.0) >= 1.0 or metrics.get(
        "table_cell_coverage", 0.0
    ) >= 0.4
    actions_ok = metrics.get("action_coverage", 0.0) >= 0.7
    sections_ok = metrics.get("section_heading_coverage", 0.0) >= 0.6

    required_checks = []
    if table_expected:
        required_checks.append(table_ok)
    if actions_expected:
        required_checks.append(actions_ok)
    if sections_expected:
        required_checks.append(sections_ok)

    if required_checks:
        intent_score = sum(1.0 if ok else 0.0 for ok in required_checks) / len(required_checks)
        intent_expectation_pass = all(required_checks)
    else:
        intent_score = 1.0
        intent_expectation_pass = True

    return {
        "intent_bucket": intent_bucket,
        "intent_require_table": float(table_expected),
        "intent_require_actions": float(actions_expected),
        "intent_require_sections": float(sections_expected),
        "intent_table_ok": float(table_ok) if table_expected else 0.0,
        "intent_actions_ok": float(actions_ok) if actions_expected else 0.0,
        "intent_sections_ok": float(sections_ok) if sections_expected else 0.0,
        "intent_expectation_pass": float(intent_expectation_pass),
        "intent_score": float(intent_score),
    }


def compute_ui_metrics(response_text: str, genui_json: Any) -> dict[str, float]:
    components = _iter_components(genui_json)
    component_count = len(components)
    comp_types = [c.get("component") for c in components if isinstance(c.get("component"), str)]
    unique_component_types = len(set(comp_types))

    text_nodes = _extract_text_nodes(components)
    text_lengths = [len(str(node.get("text", ""))) for node in text_nodes]
    total_text_length = sum(text_lengths)
    max_text_length = max(text_lengths) if text_lengths else 0
    information_chunking_score = (
        1.0 - (max_text_length / total_text_length) if total_text_length > 0 else 0.0
    )

    container_count = sum(1 for c in components if c.get("component") in _CONTAINER_COMPONENTS)
    text_count = len(text_nodes)
    container_to_text_ratio = container_count / max(1, text_count)

    max_depth, avg_depth = _compute_tree_depths(components)
    ui_modularity_score = 0.0
    if component_count > 0:
        modular = sum(
            1 for c in components if c.get("component") in {"Card", "List", "Row"}
        )
        ui_modularity_score = modular / component_count

    button_urls = _extract_button_actions(components)
    expected_action_urls = []
    for line in response_text.splitlines():
        if "button" in line.lower():
            expected_action_urls.extend(_extract_urls(line))
    expected_action_urls = list(dict.fromkeys(expected_action_urls))
    action_coverage = 1.0
    if expected_action_urls:
        action_coverage = min(1.0, len(set(button_urls)) / len(expected_action_urls))

    all_urls = _extract_urls(response_text)
    text_urls = []
    for node in text_nodes:
        text = node.get("text")
        if isinstance(text, str):
            text_urls.extend(_extract_urls(text))
    url_as_text_rate = 0.0
    if all_urls:
        url_as_text_rate = min(1.0, len(text_urls) / len(all_urls))

    index = _build_component_index(components)
    table_rows = _find_table_rows(components)
    table_pattern_detected = 1.0 if table_rows else 0.0
    expected_cells = _extract_table_cells_from_response(response_text)
    ir_cells = _extract_table_cells_from_ir(table_rows, index)
    expected_norm = {_normalize_text(c) for c in expected_cells if c}
    ir_norm = {_normalize_text(c) for c in ir_cells if c}
    table_cell_coverage = 0.0
    if expected_norm:
        table_cell_coverage = len(expected_norm & ir_norm) / len(expected_norm)
    elif table_pattern_detected >= 1.0:
        # Some historical runs do not persist response_text in genui.jsonl, so
        # response-derived table cells are unavailable. Use structural evidence.
        table_cell_coverage = 1.0 if not (response_text or "").strip() else 0.5

    expected_headings = _extract_expected_headings(response_text)
    heading_texts = []
    for node in text_nodes:
        variant = node.get("variant")
        if isinstance(variant, str) and variant.lower() in _HEADING_VARIANTS:
            text = node.get("text")
            if isinstance(text, str):
                heading_texts.append(text)
    expected_heading_norm = {_normalize_text(h) for h in expected_headings if h}
    heading_norm = {_normalize_text(h) for h in heading_texts if h}
    section_heading_coverage = 1.0
    if expected_heading_norm:
        matches = 0
        for expected in expected_heading_norm:
            for actual in heading_norm:
                if expected in actual or actual in expected:
                    matches += 1
                    break
        section_heading_coverage = matches / len(expected_heading_norm)

    markdown_leakage_rate = _markdown_leakage_rate(text_nodes)

    # Reference integrity (dangling nodes / missing ids)
    ref_integrity = _compute_reference_integrity(components)

    # UI decomposition: weighted blend of structuring signals
    container_norm = min(1.0, container_to_text_ratio / 1.5) if container_to_text_ratio >= 0 else 0.0
    variety_norm = min(1.0, unique_component_types / 8.0) if unique_component_types >= 0 else 0.0
    ui_decomposition_score = (
        0.35 * container_norm
        + 0.25 * ui_modularity_score
        + 0.2 * information_chunking_score
        + 0.2 * variety_norm
    )

    return {
        "component_count": float(component_count),
        "unique_component_types": float(unique_component_types),
        "max_tree_depth": float(max_depth),
        "avg_tree_depth": float(avg_depth),
        "container_to_text_ratio": float(container_to_text_ratio),
        "information_chunking_score": float(information_chunking_score),
        "ui_modularity_score": float(ui_modularity_score),
        "ui_decomposition_score": float(ui_decomposition_score),
        "actionable_elements": float(len(button_urls)),
        "action_coverage": float(action_coverage),
        "url_as_text_rate": float(url_as_text_rate),
        "table_pattern_detected": float(table_pattern_detected),
        "table_cell_coverage": float(table_cell_coverage),
        "section_heading_coverage": float(section_heading_coverage),
        "markdown_leakage_rate": float(markdown_leakage_rate),
        "missing_ids": ref_integrity["missing_ids"],
        "missing_ids_rate": ref_integrity["missing_ids_rate"],
        "dangling_components": ref_integrity["dangling_components"],
        "dangling_components_rate": ref_integrity["dangling_components_rate"],
    }


def count_tokens(text: str) -> int:
    return len(text.split())


def content_coverage(response_text: str, genui_json: Any) -> float:
    text = response_text.lower()
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in _stopwords]
    if not words:
        return 0.0
    genui_text = json.dumps(genui_json, ensure_ascii=False).lower()
    hits = sum(1 for w in set(words) if w in genui_text)
    return hits / max(1, len(set(words)))


def dup_rate(genui_json: Any) -> float:
    serialized = json.dumps(genui_json, sort_keys=True)
    lines = serialized.split(",")
    if not lines:
        return 0.0
    counts = Counter(lines)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return dup / max(1, len(lines))


def lint_score(genui_json: Any) -> float:
    score = 1.0
    penalties = 0.0

    def walk(obj: Any):
        nonlocal penalties
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in {"children", "child", "components"} and not v:
                    penalties += 0.1
                if k in {"text", "label"} and v in (None, ""):
                    penalties += 0.05
                walk(v)
        elif isinstance(obj, list):
            if len(obj) == 0:
                penalties += 0.1
            for item in obj:
                walk(item)

    walk(genui_json)
    score = max(0.0, score - penalties)
    return score


def compression_ratio(tokens_json: int, tokens_toon: int) -> float:
    if tokens_json <= 0:
        return 0.0
    return tokens_toon / tokens_json


def compute_overall_score(aggregate: dict[str, Any], weights: dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        value = aggregate.get(f"{key}_rate")
        if value is None:
            value = aggregate.get(f"{key}_avg")
        if value is None:
            value = aggregate.get(key)
        if value is None:
            continue
        score += float(value) * float(weight)
    return score


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = int(math.ceil((p / 100.0) * len(values))) - 1
        k = max(0, min(k, len(values) - 1))
        return values[k]

    computed_rows: list[dict[str, Any]] = []
    for row in rows:
        row_metrics = dict(row.get("metrics", {}))
        genui_json = row.get("genui_json")
        if genui_json is None:
            genui_json = row.get("a2ui_json")
        response_text = row.get("response_text", "") or ""
        if genui_json is not None:
            row_metrics.update(compute_ui_metrics(response_text, genui_json))
            row_metrics["content_coverage"] = content_coverage(response_text, genui_json)
            row_metrics["dup_rate"] = dup_rate(genui_json)
            row_metrics["lint_score"] = lint_score(genui_json)
            row_metrics.update(
                compute_intent_metrics(
                    row.get("intent"),
                    row.get("tags"),
                    response_text,
                    row_metrics,
                )
            )
        computed_rows.append({"row": row, "metrics": row_metrics})

    metrics = {
        "schema_valid_strict": [1.0 if item["row"].get("validation", {}).get("schema_valid_strict") else 0.0 for item in computed_rows],
        "content_coverage": [item["metrics"].get("content_coverage", 0.0) for item in computed_rows],
        "lint_score": [item["metrics"].get("lint_score", 0.0) for item in computed_rows],
        "dup_rate": [item["metrics"].get("dup_rate", 0.0) for item in computed_rows],
        "component_count": [item["metrics"].get("component_count", 0.0) for item in computed_rows],
        "unique_component_types": [item["metrics"].get("unique_component_types", 0.0) for item in computed_rows],
        "max_tree_depth": [item["metrics"].get("max_tree_depth", 0.0) for item in computed_rows],
        "avg_tree_depth": [item["metrics"].get("avg_tree_depth", 0.0) for item in computed_rows],
        "container_to_text_ratio": [item["metrics"].get("container_to_text_ratio", 0.0) for item in computed_rows],
        "information_chunking_score": [item["metrics"].get("information_chunking_score", 0.0) for item in computed_rows],
        "ui_modularity_score": [item["metrics"].get("ui_modularity_score", 0.0) for item in computed_rows],
        "ui_decomposition_score": [item["metrics"].get("ui_decomposition_score", 0.0) for item in computed_rows],
        "actionable_elements": [item["metrics"].get("actionable_elements", 0.0) for item in computed_rows],
        "action_coverage": [item["metrics"].get("action_coverage", 0.0) for item in computed_rows],
        "url_as_text_rate": [item["metrics"].get("url_as_text_rate", 0.0) for item in computed_rows],
        "table_pattern_detected": [item["metrics"].get("table_pattern_detected", 0.0) for item in computed_rows],
        "table_cell_coverage": [item["metrics"].get("table_cell_coverage", 0.0) for item in computed_rows],
        "section_heading_coverage": [item["metrics"].get("section_heading_coverage", 0.0) for item in computed_rows],
        "markdown_leakage_rate": [item["metrics"].get("markdown_leakage_rate", 0.0) for item in computed_rows],
        "missing_ids": [item["metrics"].get("missing_ids", 0.0) for item in computed_rows],
        "missing_ids_rate": [item["metrics"].get("missing_ids_rate", 0.0) for item in computed_rows],
        "dangling_components": [item["metrics"].get("dangling_components", 0.0) for item in computed_rows],
        "dangling_components_rate": [item["metrics"].get("dangling_components_rate", 0.0) for item in computed_rows],
        "intent_expectation_pass": [item["metrics"].get("intent_expectation_pass", 0.0) for item in computed_rows],
        "intent_score": [item["metrics"].get("intent_score", 0.0) for item in computed_rows],
        "intent_require_table": [item["metrics"].get("intent_require_table", 0.0) for item in computed_rows],
        "intent_table_ok": [item["metrics"].get("intent_table_ok", 0.0) for item in computed_rows],
        "render_ok": [
            1.0 if item["row"].get("render", {}).get("image_ok") else 0.0
            for item in computed_rows
            if isinstance(item["row"].get("render"), dict)
        ],
        "latency_ms": [item["row"].get("gen", {}).get("latency_ms", 0.0) for item in computed_rows if item["row"].get("gen")],
    }

    intent_stats: dict[str, dict[str, float]] = {}
    for item in computed_rows:
        row = item["row"]
        row_metrics = item["metrics"]
        intent_bucket = row.get("intent_bucket") or _normalize_intent(row.get("intent"))
        if not intent_bucket:
            intent_bucket = "unknown"
        stats = intent_stats.setdefault(
            intent_bucket,
            {
                "count": 0.0,
                "expectation_pass": 0.0,
                "intent_score": 0.0,
                "require_table": 0.0,
                "require_actions": 0.0,
                "require_sections": 0.0,
                "table_ok": 0.0,
                "actions_ok": 0.0,
                "sections_ok": 0.0,
            },
        )
        stats["count"] += 1.0
        stats["expectation_pass"] += float(row_metrics.get("intent_expectation_pass", 0.0))
        stats["intent_score"] += float(row_metrics.get("intent_score", 0.0))
        stats["require_table"] += float(row_metrics.get("intent_require_table", 0.0))
        stats["require_actions"] += float(row_metrics.get("intent_require_actions", 0.0))
        stats["require_sections"] += float(row_metrics.get("intent_require_sections", 0.0))
        stats["table_ok"] += float(row_metrics.get("intent_table_ok", 0.0))
        stats["actions_ok"] += float(row_metrics.get("intent_actions_ok", 0.0))
        stats["sections_ok"] += float(row_metrics.get("intent_sections_ok", 0.0))

    intent_stats_out = {}
    for intent_bucket, stats in intent_stats.items():
        count = stats.get("count", 0.0) or 0.0
        if count <= 0:
            continue
        intent_stats_out[intent_bucket] = {
            "count": int(count),
            "expectation_pass_rate": stats["expectation_pass"] / count,
            "intent_score_avg": stats["intent_score"] / count,
            "table_expected_rate": stats["require_table"] / count,
            "table_ok_rate": stats["table_ok"] / count,
            "action_expected_rate": stats["require_actions"] / count,
            "action_ok_rate": stats["actions_ok"] / count,
            "section_expected_rate": stats["require_sections"] / count,
            "section_ok_rate": stats["sections_ok"] / count,
        }

    table_ok_values = [
        metrics["intent_table_ok"][idx]
        for idx in range(len(computed_rows))
        if metrics["intent_require_table"][idx] >= 1.0
    ]

    return {
        "counts": len(rows),
        "schema_valid_strict_rate": mean(metrics["schema_valid_strict"]),
        "content_coverage_avg": mean(metrics["content_coverage"]),
        "lint_score_avg": mean(metrics["lint_score"]),
        "dup_rate_avg": mean(metrics["dup_rate"]),
        "component_count_avg": mean(metrics["component_count"]),
        "unique_component_types_avg": mean(metrics["unique_component_types"]),
        "max_tree_depth_avg": mean(metrics["max_tree_depth"]),
        "avg_tree_depth_avg": mean(metrics["avg_tree_depth"]),
        "container_to_text_ratio_avg": mean(metrics["container_to_text_ratio"]),
        "information_chunking_score_avg": mean(metrics["information_chunking_score"]),
        "ui_modularity_score_avg": mean(metrics["ui_modularity_score"]),
        "ui_decomposition_score_avg": mean(metrics["ui_decomposition_score"]),
        "actionable_elements_avg": mean(metrics["actionable_elements"]),
        "action_coverage_avg": mean(metrics["action_coverage"]),
        "url_as_text_rate_avg": mean(metrics["url_as_text_rate"]),
        "table_pattern_detected_rate": mean(metrics["table_pattern_detected"]),
        "table_cell_coverage_avg": mean(metrics["table_cell_coverage"]),
        "section_heading_coverage_avg": mean(metrics["section_heading_coverage"]),
        "markdown_leakage_rate_avg": mean(metrics["markdown_leakage_rate"]),
        "missing_ids_avg": mean(metrics["missing_ids"]),
        "missing_ids_rate_avg": mean(metrics["missing_ids_rate"]),
        "dangling_components_avg": mean(metrics["dangling_components"]),
        "dangling_components_rate_avg": mean(metrics["dangling_components_rate"]),
        "intent_expectation_pass_rate": mean(metrics["intent_expectation_pass"]),
        "intent_score_avg": mean(metrics["intent_score"]),
        "table_required_rate": mean(metrics["intent_require_table"]),
        "table_ok_rate": mean(table_ok_values) if table_ok_values else 0.0,
        "intent_stats": intent_stats_out or None,
        "render_ok_rate": mean(metrics["render_ok"]) if metrics["render_ok"] else None,
        "latency_ms_avg": mean(metrics["latency_ms"]),
        "latency_ms_p95": pct(metrics["latency_ms"], 95),
    }
