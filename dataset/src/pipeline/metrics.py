from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Mapping

from pipeline.genui_quality.aggregate import (
    aggregate_v4_records,
    aggregate_v5_0_records,
    aggregate_v5_1_records,
    aggregate_v5_records,
    aggregate_v5_3_records,
)
from pipeline.genui_quality.aggregate_v5_4 import aggregate_v5_4_records
from pipeline.renderer_capability import renderer_coverage
from pipeline.genui_quality.config import RewardConfig, normalize_metric_mode
from pipeline.genui_quality.config_v5_3 import RewardConfigV53
from pipeline.genui_quality.config_v5_4 import RewardConfigV54

_stopwords = {
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are","was","were","be","as","it","this","that"
}

_URL_RE = re.compile(r"https?://[^\s\]\)\}<>\"']+")
_MARKDOWN_LINE_RE = re.compile(r"^\s*(?:[-*]\s+|#+\s+|\|).*")
_ASSET_SECTION_HEADERS = {"images", "icons", "assets", "files"}

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
_HEADING_STYLE_TOKENS = {
    "headline",
    "title",
    "subtitle",
    "display",
    "header",
    "large",
    "xlarge",
    "xl",
}
_URL_FIELD_CANDIDATES = {"url", "href", "link", "bookingUrl", "buttonUrl", "sourceUrl"}

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
    value = text.strip().lower()
    # Strip bold/italic markdown markers
    value = value.replace("**", "").replace("__", "").replace("*", "").replace("_", " ")
    # Strip pipe characters from table remnants
    value = value.replace("|", " ")
    # Collapse whitespace
    return re.sub(r"\s+", " ", value).strip()


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


def _normalize_url_for_compare(url: str) -> str:
    if not isinstance(url, str):
        return ""
    cleaned = url.strip().rstrip(".,);:!?")
    return cleaned.lower()


def _extract_declared_asset_urls(response_text: str) -> list[str]:
    if not response_text:
        return []
    urls: list[str] = []
    active_section: str | None = None
    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            active_section = None
            continue
        header = line.rstrip(":").strip().lower()
        if header in _ASSET_SECTION_HEADERS:
            active_section = header
            continue
        if active_section is None and "http" not in line:
            continue
        line_urls = _extract_urls(line)
        if not line_urls:
            continue
        if active_section in _ASSET_SECTION_HEADERS:
            urls.extend(line_urls)
            continue
        if re.search(r"\b(image|icon)\b", line, flags=re.IGNORECASE):
            urls.extend(line_urls)
    normalized = [_normalize_url_for_compare(u) for u in urls if _normalize_url_for_compare(u)]
    return list(dict.fromkeys(normalized))


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
        # Flat-spec path: {"root","state","elements"}.
        elements = genui_json.get("elements")
        if isinstance(elements, dict):
            for element_id, element in elements.items():
                if not isinstance(element_id, str) or not isinstance(element, dict):
                    continue
                element_type = element.get("type")
                if not isinstance(element_type, str):
                    continue
                props = element.get("props") if isinstance(element.get("props"), dict) else {}
                children = element.get("children") if isinstance(element.get("children"), list) else []
                converted: dict[str, Any] = {
                    "id": element_id,
                    "component": element_type,
                    "children": [c for c in children if isinstance(c, str)],
                }
                for key, value in props.items():
                    converted[key] = value
                if isinstance(element.get("on"), dict):
                    converted["on"] = element.get("on")
                if isinstance(element.get("watch"), dict):
                    converted["watch"] = element.get("watch")
                if isinstance(element.get("repeat"), dict):
                    converted["repeat"] = element.get("repeat")
                elif isinstance(props.get("repeat"), dict):
                    # Some generators place repeat metadata inside props.
                    converted["repeat"] = props.get("repeat")
                if "visible" in element:
                    converted["visible"] = element.get("visible")
                components.append(converted)
            return components

        update = genui_json.get("updateComponents") or genui_json.get("surfaceUpdate")
        if isinstance(update, dict):
            comps = update.get("components")
            if isinstance(comps, list):
                components.extend([c for c in comps if isinstance(c, dict)])
    return components


def _has_table_entity_media(genui_json: Any) -> bool:
    if not isinstance(genui_json, dict):
        return False
    elements = genui_json.get("elements")
    if not isinstance(elements, dict):
        return False
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        if str(element.get("type") or "").strip().lower() != "table":
            continue
        props = element.get("props") if isinstance(element.get("props"), dict) else {}
        entity_media = props.get("entityMedia")
        if not isinstance(entity_media, dict):
            continue
        for value in entity_media.values():
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, dict):
                for key in ("image", "url", "src", "source", "path"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return True
    return False


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
    template = comp.get("template")
    if isinstance(template, str) and template not in children:
        children.append(template)
    item_template = comp.get("itemTemplate")
    if isinstance(item_template, str) and item_template not in children:
        children.append(item_template)
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
    first_non_empty_added = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not first_non_empty_added:
            first_non_empty_added = True
            if len(stripped.split()) <= 10 and not re.search(r"[.!?]\s*$", stripped):
                headings.append(stripped)
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
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", stripped)]
        if 1 <= len(words) <= 8:
            titled = sum(1 for w in words if w[:1].isupper())
            if titled >= max(1, math.ceil(len(words) * 0.6)) and not re.search(r"[.!?]\s*$", stripped):
                headings.append(stripped)
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


def _extract_button_actions(components: list[dict[str, Any]], genui_json: Any = None) -> list[str]:
    def _collect_urls(binding: Any) -> list[str]:
        urls: list[str] = []
        if isinstance(binding, list):
            for item in binding:
                urls.extend(_collect_urls(item))
            return urls

        if not isinstance(binding, dict):
            return urls

        # Flat-spec action binding.
        action_name = binding.get("action")
        if action_name == "openUrl":
            params = binding.get("params")
            if isinstance(params, dict):
                url = params.get("url")
                if isinstance(url, str):
                    urls.append(url)
                elif isinstance(url, dict):
                    dynamic_key = _extract_dynamic_url_key(url)
                    if dynamic_key:
                        urls.extend(_resolve_state_urls(genui_json, dynamic_key))
            # Some model outputs still place url at top-level.
            top_url = binding.get("url")
            if isinstance(top_url, str):
                urls.append(top_url)
            elif isinstance(top_url, dict):
                dynamic_key = _extract_dynamic_url_key(top_url)
                if dynamic_key:
                    urls.extend(_resolve_state_urls(genui_json, dynamic_key))

        # Legacy functionCall action binding.
        func = binding.get("functionCall")
        if isinstance(func, dict) and func.get("call") == "openUrl":
            args = func.get("args")
            if isinstance(args, dict):
                url = args.get("url")
                if isinstance(url, str):
                    urls.append(url)

        return urls

    urls: list[str] = []
    for comp in components:
        # Flat-spec may attach navigational actions on cards/rows too.
        # Flat-spec event map.
        on_map = comp.get("on")
        if isinstance(on_map, dict):
            for event_binding in on_map.values():
                urls.extend(_collect_urls(event_binding))

        # Legacy action object.
        action = comp.get("action")
        urls.extend(_collect_urls(action))
    if not urls and genui_json is not None:
        # Fallback: dynamic URL bindings can hide concrete values inside state.
        urls.extend(_extract_urls(json.dumps(genui_json, ensure_ascii=False)))
    return list(dict.fromkeys(urls))


def _extract_dynamic_url_key(value: dict[str, Any]) -> str | None:
    bind_item = value.get("$bindItem")
    if isinstance(bind_item, str) and bind_item.strip():
        return bind_item.strip()
    item = value.get("$item")
    if isinstance(item, str) and item.strip():
        return item.strip()
    state_path = value.get("$state")
    if isinstance(state_path, str) and state_path.strip():
        return state_path.strip().split("/")[-1] or None
    return None


def _iter_state_urls(value: Any, preferred_key: str | None = None) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        urls.extend(_extract_urls(value))
        return urls
    if isinstance(value, list):
        for item in value:
            urls.extend(_iter_state_urls(item, preferred_key))
        return urls
    if isinstance(value, dict):
        for key, item in value.items():
            if preferred_key and key != preferred_key:
                urls.extend(_iter_state_urls(item, preferred_key))
                continue
            if key in _URL_FIELD_CANDIDATES and isinstance(item, str):
                urls.extend(_extract_urls(item))
            else:
                urls.extend(_iter_state_urls(item, preferred_key))
    return urls


def _resolve_state_urls(genui_json: Any, dynamic_key: str | None = None) -> list[str]:
    if not isinstance(genui_json, dict):
        return []
    state = genui_json.get("state")
    if not isinstance(state, dict):
        return []
    urls = _iter_state_urls(state, dynamic_key)
    if urls:
        return list(dict.fromkeys(urls))
    return list(dict.fromkeys(_iter_state_urls(state, None)))


def _find_first_text_descendant(
    component_id: str,
    index: dict[str, dict[str, Any]],
    max_depth: int = 3,
) -> str | None:
    if component_id not in index:
        return None
    queue: list[tuple[str, int]] = [(component_id, 0)]
    visited: set[str] = set()
    while queue:
        node_id, depth = queue.pop(0)
        if node_id in visited or depth > max_depth:
            continue
        visited.add(node_id)
        node = index.get(node_id)
        if not isinstance(node, dict):
            continue
        if node.get("component") == "Text":
            return node_id
        for child_id in _component_children(node):
            if child_id in index:
                queue.append((child_id, depth + 1))
    return None


def _row_text_cell_ids(row_comp: dict[str, Any], index: dict[str, dict[str, Any]]) -> list[str]:
    if row_comp.get("component") != "Row":
        return []
    cells: list[str] = []
    for child_id in _component_children(row_comp):
        child = index.get(child_id)
        if not child:
            continue
        if child.get("component") == "Text":
            cells.append(child_id)
            continue
        descendant = _find_first_text_descendant(child_id, index)
        if descendant:
            cells.append(descendant)
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
        if isinstance(comp.get("repeat"), dict) and len(row_nodes) == 1 and len(row_nodes[0]) >= 2:
            # Flat-spec repeat lists often contain one row template.
            _append_if_table_like(rows, row_nodes * 3, min_rows=2)
        else:
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
                _append_if_table_like(rows, seq, min_rows=2)
                seq = []
                continue
            child_type = child.get("component")
            if child_type == "Row":
                row_cells = _row_text_cell_ids(child, index)
                if row_cells:
                    seq.append(row_cells)
                else:
                    _append_if_table_like(rows, seq, min_rows=2)
                    seq = []
            elif child_type == "Divider":
                # Allow divider separators within a table block.
                continue
            else:
                _append_if_table_like(rows, seq, min_rows=2)
                seq = []
        _append_if_table_like(rows, seq, min_rows=2)

    return rows


def _extract_direct_table_cells_from_components(
    components: list[dict[str, Any]],
    genui_json: Any,
) -> list[str]:
    if not isinstance(genui_json, dict):
        return []
    state = genui_json.get("state")
    if not isinstance(state, dict):
        state = {}

    cells: list[str] = []
    for comp in components:
        if comp.get("component") != "Table":
            continue
        rows = comp.get("rows")
        if not isinstance(rows, list):
            state_path = comp.get("statePath")
            if isinstance(state_path, str) and state_path.strip():
                resolved = _resolve_state_pointer(state, state_path.strip())
                if isinstance(resolved, list):
                    rows = resolved
        if not isinstance(rows, list) or not rows:
            continue

        columns = comp.get("columns")
        column_keys: list[str] = []
        if isinstance(columns, list):
            for item in columns:
                if isinstance(item, dict):
                    key = item.get("key")
                    label = item.get("label")
                    if isinstance(label, str) and label.strip():
                        cells.append(label.strip())
                    if isinstance(key, str) and key.strip():
                        column_keys.append(key.strip())
                elif isinstance(item, str) and item.strip():
                    column_keys.append(item.strip())
                    cells.append(item.strip())
        source_text = comp.get("sourceText")
        if isinstance(source_text, str) and source_text.strip():
            cells.extend(_extract_table_cells_from_response(source_text))

        for row in rows:
            if isinstance(row, dict):
                if column_keys:
                    for key in column_keys:
                        value = row.get(key)
                        cells.extend(_flatten_scalar_values(value))
                else:
                    for value in row.values():
                        cells.extend(_flatten_scalar_values(value))
            elif isinstance(row, list):
                for value in row:
                    cells.extend(_flatten_scalar_values(value))
            else:
                cells.extend(_flatten_scalar_values(row))
    return cells


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
                elif isinstance(text, dict):
                    bind_item = text.get("$bindItem")
                    if isinstance(bind_item, str) and bind_item.strip():
                        cells.append(bind_item.strip())
                    item_key = text.get("$item")
                    if isinstance(item_key, str) and item_key.strip():
                        cells.append(item_key.strip())
                    template_text = text.get("$template")
                    if isinstance(template_text, str) and template_text.strip():
                        cells.append(template_text.strip())
    return cells


def _has_strong_table_signal(response_text: str) -> bool:
    if not response_text:
        return False
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    pipe_rows = 0
    kv_rows = 0
    for line in lines:
        if "|" in line:
            pipe_rows += 1
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if re.match(r"^([^:|]{1,80}):\s+(.+)$", cleaned):
            kv_rows += 1
    if pipe_rows >= 2:
        return True
    lower = response_text.lower()
    if re.search(r"\b(table|comparison|matrix)\b", lower) and kv_rows >= 3:
        return True
    if re.search(r"\b(breakdown|metrics)\b", lower) and kv_rows >= 6:
        return True
    return False


def _resolve_state_pointer(state: dict[str, Any], pointer: str) -> Any:
    if not isinstance(pointer, str):
        return None
    path = pointer.strip()
    if not path:
        return None
    if path.startswith("/"):
        current: Any = state
        for segment in path.split("/")[1:]:
            if segment == "":
                continue
            if isinstance(current, dict) and segment in current:
                current = current[segment]
                continue
            if isinstance(current, list):
                try:
                    idx = int(segment)
                except Exception:
                    return None
                if idx < 0 or idx >= len(current):
                    return None
                current = current[idx]
                continue
            return None
        return current
    if path.startswith("$."):
        path = path[2:]
    if path.startswith("state."):
        path = path[len("state.") :]
    current = state
    for segment in [p for p in path.split(".") if p]:
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return None
    return current


def _flatten_scalar_values(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for v in value.values():
            out.extend(_flatten_scalar_values(v))
        return out
    if isinstance(value, list):
        for v in value:
            out.extend(_flatten_scalar_values(v))
        return out
    if isinstance(value, (str, int, float, bool)):
        out.append(str(value))
    return out


def _extract_repeat_state_cells(genui_json: Any) -> list[str]:
    if not isinstance(genui_json, dict):
        return []
    state = genui_json.get("state")
    elements = genui_json.get("elements")
    if not isinstance(state, dict) or not isinstance(elements, dict):
        return []
    cells: list[str] = []
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        repeat = element.get("repeat")
        if not isinstance(repeat, dict):
            continue
        state_path = repeat.get("statePath")
        if not isinstance(state_path, str) or not state_path.strip():
            continue
        repeat_value = _resolve_state_pointer(state, state_path)
        if repeat_value is None:
            continue
        cells.extend(_flatten_scalar_values(repeat_value))
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
    table_like = _has_strong_table_signal(response_text)
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


def _canonical_metric_payload(value: Any) -> Any:
    """Decode only active Express or internal standard-wire values for metrics.

    A canonical graph is already the renderer-facing internal value.  Legacy
    FlatSpec auto-detection is intentionally not used here because metrics must
    never turn a malformed model completion into a valid legacy candidate.
    """
    try:
        from pipeline.ir_formats import decode_express_completion
        from pipeline.ir_formats.a2ui_wire import decode as decode_wire

        if isinstance(value, str):
            return decode_express_completion(value)
        if isinstance(value, Mapping) and value.get("version") == "v1.0":
            return decode_wire(value)
    except Exception:
        return value
    return value


def _metric_payload_from_record(record: Mapping[str, Any]) -> Any:
    """Select the active Express artifact before any offline legacy field.

    A present Express completion is authoritative: malformed active output is
    not replaced with a FlatSpec/placeholder candidate.  ``genui_json`` and
    ``a2ui_json`` are retained only as explicit historical comparison fields.
    """

    for key in ("a2ui_express", "completion", "model_completion_raw"):
        candidate = record.get(key)
        if isinstance(candidate, str) and candidate.strip():
            try:
                return _canonical_metric_payload(candidate)
            except Exception:
                return None
    canonical = record.get("canonical_graph")
    if isinstance(canonical, Mapping):
        return canonical
    if record.get("source_format") in {"flat_spec_v1", "compact_ir_v2", "compact_ir", "gci2"}:
        return record.get("genui_json", record.get("a2ui_json"))
    # Historical metric fixtures use the legacy key itself as an explicit
    # migration marker; do not infer this from arbitrary mappings.
    if "genui_json" in record or "a2ui_json" in record:
        return record.get("genui_json", record.get("a2ui_json"))
    return None


def compute_ui_metrics(response_text: str, genui_json: Any) -> dict[str, float]:
    genui_json = _canonical_metric_payload(genui_json)
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
    component_count_norm = min(1.0, component_count / 60.0) if component_count >= 0 else 0.0
    component_count_capped = min(60.0, float(component_count)) if component_count >= 0 else 0.0
    ui_modularity_score = 0.0
    if component_count > 0:
        modular = sum(
            1 for c in components if c.get("component") in {"Card", "List", "Row"}
        )
        ui_modularity_score = modular / component_count

    button_urls = _extract_button_actions(components, genui_json=genui_json)
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
    direct_table_cells = _extract_direct_table_cells_from_components(components, genui_json)
    table_pattern_detected = 1.0 if table_rows or direct_table_cells else 0.0
    expected_cells = _extract_table_cells_from_response(response_text)
    ir_cells = _extract_table_cells_from_ir(table_rows, index)
    ir_cells.extend(direct_table_cells)
    ir_cells.extend(_extract_repeat_state_cells(genui_json))
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
        typography = node.get("typography")
        font_size = node.get("fontSize")
        font_weight = str(node.get("fontWeight") or "").strip().lower()
        comp_id = str(node.get("id") or "").strip().lower()
        is_heading = False
        if isinstance(variant, str) and variant.lower() in _HEADING_VARIANTS:
            is_heading = True
        if not is_heading and isinstance(typography, str):
            t = typography.lower()
            if any(token in t for token in _HEADING_STYLE_TOKENS):
                is_heading = True
        if not is_heading and isinstance(font_size, str):
            f = font_size.lower().replace("-", "")
            if any(token in f for token in _HEADING_STYLE_TOKENS):
                is_heading = True
        if not is_heading and any(token in comp_id for token in ("title", "header", "heading")):
            is_heading = True
        text = node.get("text")
        if not is_heading and isinstance(text, str):
            words = len(text.split())
            if words <= 8 and font_weight in {"bold", "semibold", "medium"}:
                is_heading = True
        if is_heading and isinstance(text, str):
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

    image_presence = 0.0
    icon_presence = 0.0
    for comp in components:
        comp_type = comp.get("component")
        if comp_type == "Image":
            variant = str(comp.get("variant") or "").strip().lower()
            if variant == "icon":
                icon_presence = 1.0
            else:
                image_presence = 1.0
        elif comp_type == "Icon":
            icon_presence = 1.0
        elif comp_type == "Table" and _has_table_entity_media(genui_json):
            image_presence = 1.0

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
        "component_count_norm": float(component_count_norm),
        "component_count_capped": float(component_count_capped),
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
        "image_presence": float(image_presence),
        "icon_presence": float(icon_presence),
        "missing_ids": ref_integrity["missing_ids"],
        "missing_ids_rate": ref_integrity["missing_ids_rate"],
        "dangling_components": ref_integrity["dangling_components"],
        "dangling_components_rate": ref_integrity["dangling_components_rate"],
    }


def lexical_token_estimate(text: str) -> int:
    """Diagnostic-only fallback; never use for format/model selection."""
    return len(text.split())


def count_characters(text: str) -> int:
    # Whitespace-insensitive payload size proxy.
    return len(re.sub(r"\s+", "", text or ""))


def content_coverage(response_text: str, genui_json: Any) -> float:
    genui_json = _canonical_metric_payload(genui_json)
    text = response_text.lower()
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in _stopwords]
    if not words:
        return 0.0
    genui_text = json.dumps(genui_json, ensure_ascii=False).lower()
    hits = sum(1 for w in set(words) if w in genui_text)
    return hits / max(1, len(set(words)))


def dup_rate(genui_json: Any) -> float:
    genui_json = _canonical_metric_payload(genui_json)
    serialized = json.dumps(genui_json, sort_keys=True)
    lines = serialized.split(",")
    if not lines:
        return 0.0
    counts = Counter(lines)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return dup / max(1, len(lines))


def lint_score(genui_json: Any) -> float:
    genui_json = _canonical_metric_payload(genui_json)
    score = 1.0
    penalties = 0.0

    def walk(obj: Any):
        nonlocal penalties
        if isinstance(obj, dict):
            for k, v in obj.items():
                # Empty children arrays are valid for leaf nodes in flat-spec IR.
                if k in {"child", "components"} and not v:
                    penalties += 0.1
                if k in {"text", "label"} and v in (None, ""):
                    penalties += 0.05
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(genui_json)
    score = max(0.0, score - penalties)
    return score


def compression_ratio(tokens_json: int, tokens_toon: int) -> float:
    if tokens_json <= 0:
        return 0.0
    return tokens_toon / tokens_json


OVERALL_SCORE_RAW_MIN = -2.0
OVERALL_SCORE_RAW_MAX = 41.1


def compute_overall_score(aggregate: dict[str, Any], weights: dict[str, float]) -> float:
    raw_score = 0.0
    for key, weight in weights.items():
        value = aggregate.get(f"{key}_rate")
        if value is None:
            value = aggregate.get(f"{key}_avg")
        if value is None:
            value = aggregate.get(key)
        if value is None:
            continue
        raw_score += float(value) * float(weight)
    if OVERALL_SCORE_RAW_MAX <= OVERALL_SCORE_RAW_MIN:
        return 0.0
    normalized = ((raw_score - OVERALL_SCORE_RAW_MIN) / (OVERALL_SCORE_RAW_MAX - OVERALL_SCORE_RAW_MIN)) * 100.0
    return max(0.0, min(100.0, normalized))


def compute_media_score(aggregate: dict[str, Any]) -> float | None:
    values: list[float] = []
    for key in (
        "image_presence_rate",
        "icon_presence_rate",
        "asset_url_valid_rate",
        "rendered_image_ok_rate",
    ):
        value = aggregate.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except Exception:
            continue
    if not values:
        return None
    return (sum(values) / len(values)) * 100.0


def aggregate_metrics(
    rows: list[dict[str, Any]],
    render_rows_by_ui_id: dict[str, dict[str, Any]] | None = None,
    metric_version: str = "dual",
    v4_config: RewardConfig | None = None,
    v5_config: RewardConfig | None = None,
    v5_1_config: RewardConfig | None = None,
    v5_0_config: RewardConfig | None = None,
    v5_3_config: RewardConfigV53 | None = None,
    v5_4_config: RewardConfigV54 | None = None,
) -> dict[str, Any]:
    if not rows:
        return {}
    metric_mode = normalize_metric_mode(metric_version)

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
        genui_json = _metric_payload_from_record(row)
        response_text = row.get("response_text", "") or ""
        row_render = row.get("render") if isinstance(row.get("render"), dict) else None
        if row_render is None and render_rows_by_ui_id:
            ui_id = row.get("ui_id")
            if isinstance(ui_id, str):
                render_row = render_rows_by_ui_id.get(ui_id)
                if isinstance(render_row, dict):
                    candidate = render_row.get("render")
                    if isinstance(candidate, dict):
                        row_render = candidate
        if genui_json is not None:
            row_metrics.update(compute_ui_metrics(response_text, genui_json))
            row_metrics["content_coverage"] = content_coverage(response_text, genui_json)
            row_metrics["dup_rate"] = dup_rate(genui_json)
            row_metrics["lint_score"] = lint_score(genui_json)
            # How much of the generated IR the Android renderer actually
            # consumes. Computed here rather than in flat_spec_contract or
            # renderer_effective_semantics_v5_4 because those are hashed into
            # metric_fingerprint_v5_4; metrics.py is not, so this reports
            # coverage without rotating score identity.
            row_metrics.update(renderer_coverage(genui_json))
            row_metrics.update(
                compute_intent_metrics(
                    row.get("intent"),
                    row.get("tags"),
                    response_text,
                    row_metrics,
                )
            )
        declared_asset_urls = set(_extract_declared_asset_urls(response_text))
        downloaded_asset_urls: set[str] = set()
        assets = row.get("assets")
        if isinstance(assets, list):
            for item in assets:
                if not isinstance(item, dict):
                    continue
                url = item.get("url")
                if isinstance(url, str):
                    norm = _normalize_url_for_compare(url)
                    if norm:
                        downloaded_asset_urls.add(norm)
        declared_count = len(declared_asset_urls)
        valid_assets = len(downloaded_asset_urls & declared_asset_urls)
        row_metrics["asset_url_declared"] = float(declared_count)
        row_metrics["asset_url_valid"] = (
            float(valid_assets / declared_count) if declared_count > 0 else 1.0
        )
        row_metrics["render_image_ok"] = (
            1.0 if isinstance(row_render, dict) and row_render.get("image_ok") else 0.0
        ) if row_render is not None else None
        computed_rows.append({"row": row, "metrics": row_metrics})

    metrics = {
        "schema_valid_strict": [1.0 if item["row"].get("validation", {}).get("schema_valid_strict") else 0.0 for item in computed_rows],
        "content_coverage": [item["metrics"].get("content_coverage", 0.0) for item in computed_rows],
        "lint_score": [item["metrics"].get("lint_score", 0.0) for item in computed_rows],
        "dup_rate": [item["metrics"].get("dup_rate", 0.0) for item in computed_rows],
        "output_tokens_toon": [item["metrics"].get("output_tokens_toon", 0.0) for item in computed_rows],
        "output_tokens_json": [item["metrics"].get("output_tokens_json", 0.0) for item in computed_rows],
        "output_chars_toon": [item["metrics"].get("output_chars_toon", 0.0) for item in computed_rows],
        "output_chars_json": [item["metrics"].get("output_chars_json", 0.0) for item in computed_rows],
        "component_count": [item["metrics"].get("component_count", 0.0) for item in computed_rows],
        "component_count_norm": [item["metrics"].get("component_count_norm", 0.0) for item in computed_rows],
        "component_count_capped": [item["metrics"].get("component_count_capped", 0.0) for item in computed_rows],
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
        "image_presence": [item["metrics"].get("image_presence", 0.0) for item in computed_rows],
        "icon_presence": [item["metrics"].get("icon_presence", 0.0) for item in computed_rows],
        "asset_url_valid": [item["metrics"].get("asset_url_valid", 1.0) for item in computed_rows],
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
        "render_image_ok": [
            float(item["metrics"].get("render_image_ok"))
            for item in computed_rows
            if item["metrics"].get("render_image_ok") is not None
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
    rendered_image_ok_values = [
        float(item["metrics"]["render_image_ok"])
        for item in computed_rows
        if item["metrics"].get("render_image_ok") is not None
        and (
            item["metrics"].get("image_presence", 0.0) >= 1.0
            or item["metrics"].get("icon_presence", 0.0) >= 1.0
            or item["metrics"].get("asset_url_declared", 0.0) > 0.0
        )
    ]

    aggregate = {
        "counts": len(rows),
        "schema_valid_strict_rate": mean(metrics["schema_valid_strict"]),
        "content_coverage_avg": mean(metrics["content_coverage"]),
        "lint_score_avg": mean(metrics["lint_score"]),
        "dup_rate_avg": mean(metrics["dup_rate"]),
        "output_tokens_toon_avg": mean(metrics["output_tokens_toon"]),
        "output_tokens_json_avg": mean(metrics["output_tokens_json"]),
        "output_chars_toon_avg": mean(metrics["output_chars_toon"]),
        "output_chars_json_avg": mean(metrics["output_chars_json"]),
        "component_count_avg": mean(metrics["component_count"]),
        "component_count_capped_avg": mean(metrics["component_count_capped"]),
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
        "image_presence_rate": mean(metrics["image_presence"]),
        "icon_presence_rate": mean(metrics["icon_presence"]),
        "asset_url_valid_rate": mean(metrics["asset_url_valid"]),
        "rendered_image_ok_rate": (
            mean(rendered_image_ok_values) if rendered_image_ok_values else None
        ),
        "missing_ids_avg": mean(metrics["missing_ids"]),
        "missing_ids_rate_avg": mean(metrics["missing_ids_rate"]),
        "dangling_components_avg": mean(metrics["dangling_components"]),
        "dangling_components_rate_avg": mean(metrics["dangling_components_rate"]),
        "intent_expectation_pass_rate": mean(metrics["intent_expectation_pass"]),
        "intent_score_avg": mean(metrics["intent_score"]),
        "table_required_rate": mean(metrics["intent_require_table"]),
        "table_ok_rate": mean(table_ok_values) if table_ok_values else 0.0,
        "intent_stats": intent_stats_out or None,
        "render_ok_rate": mean(metrics["render_ok"]) if metrics["render_ok"] else (
            mean(metrics["render_image_ok"]) if metrics["render_image_ok"] else None
        ),
        "latency_ms_avg": mean(metrics["latency_ms"]),
        "latency_ms_p95": pct(metrics["latency_ms"], 95),
    }
    aggregate["media_score"] = compute_media_score(aggregate)
    aggregate["evaluation_metric_mode"] = metric_mode
    if metric_mode in {"v4", "dual"}:
        aggregate["genui_quality_v4"] = aggregate_v4_records(
            rows,
            render_rows_by_ui_id=render_rows_by_ui_id,
            config=v4_config,
        )
    if metric_mode in {"v5", "v5_2", "dual"}:
        aggregate["genui_quality_v5_2"] = aggregate_v5_records(
            rows,
            render_rows_by_ui_id=render_rows_by_ui_id,
            config=v5_config,
        )
    if metric_mode in {"v5_1", "dual"}:
        aggregate["genui_quality_v5_1"] = aggregate_v5_1_records(
            rows,
            render_rows_by_ui_id=render_rows_by_ui_id,
            config=v5_1_config,
        )
    if metric_mode in {"v5_0", "dual"}:
        aggregate["genui_quality_v5"] = aggregate_v5_0_records(
            rows,
            render_rows_by_ui_id=render_rows_by_ui_id,
            config=v5_0_config,
        )
    if metric_mode in {"v5_3", "dual"}:
        aggregate["genui_quality_v5_3"] = aggregate_v5_3_records(
            rows,
            render_rows_by_ui_id=render_rows_by_ui_id,
            config=v5_3_config,
        )
    if metric_mode in {"v5_4", "dual"}:
        aggregate["genui_quality_v5_4"] = aggregate_v5_4_records(
            rows,
            render_rows_by_ui_id=render_rows_by_ui_id,
            config=v5_4_config,
        )
    return aggregate
