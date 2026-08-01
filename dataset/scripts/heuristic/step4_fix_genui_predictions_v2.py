#!/usr/bin/env python3
"""
Fix genui.jsonl files to match the expected schema.

This script:
1. Loads queries.jsonl to get intent information by query_id (from input dir)
2. Loads responses.jsonl to get response_text and response_id (from input dir)
3. Transforms genui.jsonl to include all required fields
4. COMPUTES actual validation and metrics scores (not placeholders)

Usage:
    python fix_genui_predictions_v2.py <input_folder> [--url_unmask]

Arguments:
    input_folder    Path to input folder containing genui.jsonl, queries.jsonl, responses.jsonl

Options:
    --url_unmask    Unmask URL placeholders (<url1>, <url2>, etc.) using the URL mapping file
                    before computing validation and metrics scores. Looks for golden50_url_mapping.jsonl
                    in the input folder.

Note: The script modifies genui.jsonl in-place (same folder is used for input and output).
      Data loss is avoided by reading all content first before writing.

Examples:
    python fix_genui_predictions_v2.py dataset/data/runs/golden50_trained/
    python fix_genui_predictions_v2.py dataset/data/runs/golden50_trained/ --url_unmask
"""

import json
import argparse
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src directory to path to import pipeline modules
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from pipeline.metrics import (
    _iter_components,
    _compute_tree_depths,
    _compute_reference_integrity,
    _extract_text_nodes,
    _extract_button_actions,
    _find_table_rows,
    _extract_direct_table_cells_from_components,
    _extract_table_cells_from_response,
    _extract_repeat_state_cells,
    _markdown_leakage_rate,
    _has_strong_table_signal,
    _extract_expected_headings,
    _extract_declared_asset_urls,
    _normalize_url_for_compare,
    _normalize_intent,
    _normalize_tags,
    _INTENT_TABLE_REQUIRED,
    _INTENT_ACTION_REQUIRED,
    _INTENT_SECTION_REQUIRED,
    content_coverage,
    dup_rate,
    lint_score,
    count_tokens,
    count_characters,
)


def _has_table_entity_media(genui_json):
    """Check if table has entity media."""
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


def _build_component_index(components):
    """Build component index by id."""
    index = {}
    for comp in components:
        comp_id = comp.get("id")
        if isinstance(comp_id, str):
            index[comp_id] = comp
    return index


def _component_children(comp):
    """Extract children from component."""
    children = []
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


def _extract_urls(text):
    """Extract URLs from text."""
    import re
    if not text:
        return []
    url_re = re.compile(r"https?://[^\s\]\)\}<>\"']+")
    urls = url_re.findall(text)
    cleaned = []
    for url in urls:
        cleaned_url = url.rstrip(".,);:")
        cleaned.append(cleaned_url)
    return cleaned


def _resolve_state_pointer(state, pointer):
    """Resolve state pointer."""
    if not isinstance(pointer, str):
        return None
    path = pointer.strip()
    if not path:
        return None
    if path.startswith("/"):
        current = state
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
        path = path[len("state."):]
    current = state
    for segment in [p for p in path.split(".") if p]:
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return None
    return current


def _iter_state_urls(value, preferred_key=None):
    """Iterate state URLs."""
    _URL_FIELD_CANDIDATES = {"url", "href", "link", "bookingUrl", "buttonUrl", "sourceUrl"}
    urls = []
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


def _resolve_state_urls(genui_json, dynamic_key=None):
    """Resolve state URLs."""
    if not isinstance(genui_json, dict):
        return []
    state = genui_json.get("state")
    if not isinstance(state, dict):
        return []
    urls = _iter_state_urls(state, dynamic_key)
    if urls:
        return list(dict.fromkeys(urls))
    return list(dict.fromkeys(_iter_state_urls(state, None)))


def _extract_dynamic_url_key(value):
    """Extract dynamic URL key."""
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


def _find_first_text_descendant(component_id, index, max_depth=3):
    """Find first text descendant."""
    if component_id not in index:
        return None
    queue = [(component_id, 0)]
    visited = set()
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


def _row_text_cell_ids(row_comp, index):
    """Get text cell ids from row."""
    if row_comp.get("component") != "Row":
        return []
    cells = []
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


def _append_if_table_like(rows_out, row_nodes, min_rows):
    """Append table-like rows."""
    if len(row_nodes) < min_rows:
        return
    counts = [len(row) for row in row_nodes]
    if not counts:
        return
    if min(counts) < 2:
        return
    if len(set(counts)) != 1:
        return
    rows_out.extend(row_nodes)


def _flatten_scalar_values(value):
    """Flatten scalar values."""
    out = []
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


def _unmask_urls_in_object(obj, url_mapping):
    """Recursively unmask URL placeholders in an object using the URL mapping."""
    if not url_mapping:
        return obj

    if isinstance(obj, str):
        result = obj
        # Replace all URL placeholders with their actual URLs
        for placeholder, actual_url in url_mapping.items():
            result = result.replace(placeholder, actual_url)
        return result
    elif isinstance(obj, dict):
        return {k: _unmask_urls_in_object(v, url_mapping) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_unmask_urls_in_object(item, url_mapping) for item in obj]
    else:
        return obj


def _load_url_mapping(input_dir):
    """Load the URL mapping file and return a dict keyed by response_id."""
    mapping_path = input_dir / "golden50_url_mapping.jsonl"
    url_mapping_by_response = {}

    if mapping_path.exists():
        print(f"Loading URL mapping from {mapping_path}...")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line.strip())
                response_id = row.get("response_id")
                url_map = row.get("url_mapping", {})
                if response_id and url_map:
                    url_mapping_by_response[response_id] = url_map
        print(f"  Loaded URL mappings for {len(url_mapping_by_response)} responses")
    else:
        print(f"  Warning: URL mapping file not found at {mapping_path}")

    return url_mapping_by_response


def compute_validation(genui_json):
    """Compute validation fields."""
    # Check if genui_json is valid JSON (it should be since we parsed it)
    json_parse_ok = genui_json is not None

    # Check schema validity - for flat spec format
    schema_valid_strict = False
    schema_valid_lenient = False

    if isinstance(genui_json, dict):
        # Flat spec format check
        has_root = "root" in genui_json
        has_elements = "elements" in genui_json and isinstance(genui_json.get("elements"), dict)
        has_state = "state" in genui_json

        if has_root and has_elements:
            # Check if elements have proper structure
            elements = genui_json.get("elements", {})
            valid_elements = 0
            for elem_id, elem in elements.items():
                if isinstance(elem, dict) and "type" in elem:
                    valid_elements += 1
            if valid_elements > 0:
                schema_valid_strict = True
                schema_valid_lenient = True

    toon_roundtrip_ok = False  # Would need actual toon rendering to verify
    converted_from_legacy = False

    errors = []
    if not json_parse_ok:
        errors.append("JSON parse failed")
    if not schema_valid_strict:
        errors.append("Schema validation failed")

    return {
        "json_parse_ok": json_parse_ok,
        "schema_valid_strict": schema_valid_strict,
        "schema_valid_lenient": schema_valid_lenient,
        "toon_roundtrip_ok": toon_roundtrip_ok,
        "converted_from_legacy": converted_from_legacy,
        "errors": errors,
        "repair_attempts": 0,
        "repair_needed": len(errors) > 0,
    }


def compute_all_metrics(response_text, genui_json, intent, tags):
    """Compute all metrics for a single row."""
    metrics = {}

    # Handle None genui_json
    if genui_json is None:
        # Return all zeros for invalid data
        return {
            "content_coverage": 0.0,
            "dup_rate": 0.0,
            "lint_score": 0.0,
            "output_tokens_toon": 0,
            "output_tokens_json": 0,
            "output_chars_toon": 0,
            "output_chars_json": 0,
            "component_count": 0.0,
            "component_count_norm": 0.0,
            "component_count_capped": 0.0,
            "unique_component_types": 0.0,
            "max_tree_depth": 0.0,
            "avg_tree_depth": 0.0,
            "container_to_text_ratio": 0.0,
            "information_chunking_score": 0.0,
            "ui_modularity_score": 0.0,
            "ui_decomposition_score": 0.0,
            "actionable_elements": 0.0,
            "action_coverage": 0.0,
            "url_as_text_rate": 0.0,
            "table_pattern_detected": 0.0,
            "table_cell_coverage": 0.0,
            "section_heading_coverage": 0.0,
            "markdown_leakage_rate": 0.0,
            "image_presence": 0.0,
            "icon_presence": 0.0,
            "missing_ids": 0.0,
            "missing_ids_rate": 0.0,
            "dangling_components": 0.0,
            "dangling_components_rate": 0.0,
            "intent_require_table": 0.0,
            "intent_require_actions": 0.0,
            "intent_require_sections": 0.0,
            "intent_table_ok": 0.0,
            "intent_actions_ok": 0.0,
            "intent_sections_ok": 0.0,
            "intent_expectation_pass": 0.0,
            "intent_score": 0.0,
        }

    # Basic metrics
    metrics["content_coverage"] = content_coverage(response_text, genui_json)
    metrics["dup_rate"] = dup_rate(genui_json)
    metrics["lint_score"] = lint_score(genui_json)

    # Token/char counts
    genui_str = json.dumps(genui_json, ensure_ascii=False)
    response_str = response_text or ""
    metrics["output_tokens_toon"] = count_tokens(genui_str)
    metrics["output_tokens_json"] = count_tokens(response_str)
    metrics["output_chars_toon"] = count_characters(genui_str)
    metrics["output_chars_json"] = count_characters(response_str)

    # Component analysis
    components = _iter_components(genui_json)
    component_count = len(components)
    comp_types = [c.get("component") for c in components if isinstance(c.get("component"), str)]
    unique_component_types = len(set(comp_types))

    _CONTAINER_COMPONENTS = {
        "Column", "Row", "List", "Card", "Tabs", "Tab", "Modal", "Grid", "Stack", "Section", "Container"
    }

    # Text nodes
    text_nodes = _extract_text_nodes(components)
    text_lengths = [len(str(node.get("text", ""))) for node in text_nodes]
    total_text_length = sum(text_lengths)
    max_text_length = max(text_lengths) if text_lengths else 0
    information_chunking_score = (1.0 - (max_text_length / total_text_length)) if total_text_length > 0 else 0.0

    container_count = sum(1 for c in components if c.get("component") in _CONTAINER_COMPONENTS)
    text_count = len(text_nodes)
    container_to_text_ratio = container_count / max(1, text_count)

    max_depth, avg_depth = _compute_tree_depths(components)
    component_count_norm = min(1.0, component_count / 60.0) if component_count >= 0 else 0.0
    component_count_capped = min(60.0, float(component_count)) if component_count >= 0 else 0.0

    ui_modularity_score = 0.0
    if component_count > 0:
        modular = sum(1 for c in components if c.get("component") in {"Card", "List", "Row"})
        ui_modularity_score = modular / component_count

    # Button actions
    button_urls = _extract_button_actions(components, genui_json=genui_json)
    expected_action_urls = []
    for line in response_text.splitlines():
        if "button" in line.lower():
            expected_action_urls.extend(_extract_urls(line))
    expected_action_urls = list(dict.fromkeys(expected_action_urls))
    action_coverage = 1.0
    if expected_action_urls:
        action_coverage = min(1.0, len(set(button_urls)) / len(expected_action_urls))

    # URL as text rate
    all_urls = _extract_urls(response_text)
    text_urls = []
    for node in text_nodes:
        text = node.get("text")
        if isinstance(text, str):
            text_urls.extend(_extract_urls(text))
    url_as_text_rate = 0.0
    if all_urls:
        url_as_text_rate = min(1.0, len(text_urls) / len(all_urls))

    # Table analysis
    index = _build_component_index(components)
    table_rows = _find_table_rows(components)
    direct_table_cells = _extract_direct_table_cells_from_components(components, genui_json)
    table_pattern_detected = 1.0 if table_rows or direct_table_cells else 0.0
    expected_cells = _extract_table_cells_from_response(response_text)
    ir_cells = []
    for row in table_rows:
        for cell_id in row:
            comp = index.get(cell_id)
            if not comp:
                continue
            if comp.get("component") == "Text":
                text = comp.get("text")
                if isinstance(text, str):
                    ir_cells.append(text)
    ir_cells.extend(direct_table_cells)
    ir_cells.extend(_extract_repeat_state_cells(genui_json))

    import re
    _stopwords = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "is", "are", "was", "were", "be", "as", "it", "this", "that"}
    def _normalize_text(text):
        value = text.strip().lower()
        value = value.replace("**", "").replace("__", "").replace("*", "").replace("_", " ")
        value = value.replace("|", " ")
        return re.sub(r"\s+", " ", value).strip()

    expected_norm = {_normalize_text(c) for c in expected_cells if c}
    ir_norm = {_normalize_text(c) for c in ir_cells if c}
    table_cell_coverage = 0.0
    if expected_norm:
        table_cell_coverage = len(expected_norm & ir_norm) / len(expected_norm)
    elif table_pattern_detected >= 1.0:
        table_cell_coverage = 1.0 if not response_text.strip() else 0.5

    # Section heading coverage
    expected_headings = _extract_expected_headings(response_text)
    heading_texts = []
    _HEADING_VARIANTS = {"h1", "h2", "h3", "headline", "title", "subtitle"}
    _HEADING_STYLE_TOKENS = {"headline", "title", "subtitle", "display", "header", "large", "xlarge", "xl"}

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
        if not is_heading and isinstance(font_weight, str) and font_weight in {"bold", "semibold", "medium"}:
            text_val = node.get("text")
            if isinstance(text_val, str):
                words = len(text_val.split())
                if words <= 8:
                    is_heading = True
        if is_heading and isinstance(node.get("text"), str):
            heading_texts.append(node.get("text"))

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

    # Image/icon presence
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

    # Reference integrity
    ref_integrity = _compute_reference_integrity(components)

    # UI decomposition
    container_norm = min(1.0, container_to_text_ratio / 1.5) if container_to_text_ratio >= 0 else 0.0
    variety_norm = min(1.0, unique_component_types / 8.0) if unique_component_types >= 0 else 0.0
    ui_decomposition_score = (
        0.35 * container_norm
        + 0.25 * ui_modularity_score
        + 0.2 * information_chunking_score
        + 0.2 * variety_norm
    )

    # Add all UI metrics
    metrics.update({
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
    })

    # Intent metrics
    intent_bucket = _normalize_intent(intent)
    tag_buckets = set(_normalize_tags(tags))
    response_lower = response_text.lower() if response_text else ""

    has_urls = bool(_extract_urls(response_text))
    has_button_marker = "quick actions" in response_lower or "[button" in response_lower
    table_like = _has_strong_table_signal(response_text)

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

    table_ok = metrics.get("table_pattern_detected", 0.0) >= 1.0 or metrics.get("table_cell_coverage", 0.0) >= 0.4
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

    metrics.update({
        "intent_require_table": float(table_expected),
        "intent_require_actions": float(actions_expected),
        "intent_require_sections": float(sections_expected),
        "intent_table_ok": float(table_ok) if table_expected else 0.0,
        "intent_actions_ok": float(actions_ok) if actions_expected else 0.0,
        "intent_sections_ok": float(sections_ok) if sections_expected else 0.0,
        "intent_expectation_pass": float(intent_expectation_pass),
        "intent_score": float(intent_score),
    })

    return metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fix genui.jsonl to match expected schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fix_genui_predictions_v2.py dataset/data/runs/golden50_trained/
    python fix_genui_predictions_v2.py dataset/data/runs/golden50_trained/ --url_unmask
        """
    )
    parser.add_argument("input_folder", help="Path to input folder containing genui.jsonl, queries.jsonl, responses.jsonl")
    parser.add_argument("--url_unmask", action="store_true", help="Unmask URL placeholders using URL mapping file")
    args = parser.parse_args()

    # Resolve paths
    input_dir = Path(args.input_folder)
    input_path = input_dir / "genui.jsonl"
    output_path = input_path  # Same file - in-place editing

    if not input_dir.exists():
        print(f"Error: Input folder not found: {input_dir}")
        sys.exit(1)

    if not input_path.exists():
        print(f"Error: genui.jsonl not found in {input_dir}")
        sys.exit(1)

    # Load URL mapping if requested
    url_mapping_by_response = {}
    if args.url_unmask:
        url_mapping_by_response = _load_url_mapping(input_dir)

    # Load queries.jsonl to get intent info (from same directory as input)
    queries_path = input_dir / "queries.jsonl"
    intent_by_query_id = {}
    if queries_path.exists():
        print(f"Loading {queries_path}...")
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line.strip())
                query_id = row.get("query_id")
                if query_id:
                    intent_by_query_id[query_id] = {
                        "intent": row.get("intent", "unknown"),
                        "tags": row.get("tags", []),
                    }
        print(f"  Loaded {len(intent_by_query_id)} queries with intents")
    else:
        print(f"  Note: queries.jsonl not found in {input_dir}, intent info will be 'unknown'")

    # Load responses.jsonl to get response_text (from same directory as input)
    responses_path = input_dir / "responses.jsonl"
    response_by_index = {}
    if responses_path.exists():
        print(f"Loading {responses_path}...")
        with open(responses_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                row = json.loads(line.strip())
                response_by_index[idx] = {
                    "response_id": row.get("response_id"),
                    "query_id": row.get("query_id"),
                    "response_text": row.get("response_text", ""),
                    "gen": row.get("gen", {}),
                    "created_at": row.get("created_at"),
                }
        print(f"  Loaded {len(response_by_index)} responses")
    else:
        print(f"  Note: responses.jsonl not found in {input_dir}, response_text will use input field")

    # Process genui.jsonl
    # Read all lines first to avoid data loss when input == output
    print(f"Processing {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_count = 0
    fixed_rows = []

    for idx, line in enumerate(lines):
        row = json.loads(line.strip())

        # Get response info for this index
        resp_info = response_by_index.get(idx, {})
        response_id = resp_info.get("response_id", f"r_{idx:06d}")
        query_id = resp_info.get("query_id", f"q_{idx:06d}")
        response_text = resp_info.get("response_text", row.get("input", ""))
        gen_info = resp_info.get("gen", {})
        created_at = resp_info.get("created_at")

        # Get intent info from query
        query_intent = intent_by_query_id.get(query_id, {})
        intent = query_intent.get("intent", "unknown")
        tags = query_intent.get("tags", [])

        # Normalize intent bucket
        intent_bucket = intent.lower().replace(" ", "_").replace("&", "and") if intent else "unknown"
        intent_bucket = intent_bucket.strip("_")

        # Get genui_json - try multiple sources
        genui_json = row.get("genui_json")

        # If genui_json is a string (JSON-encoded), parse it
        if isinstance(genui_json, str):
            try:
                genui_json = json.loads(genui_json)
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse genui_json string at index {idx}: {e}")
                genui_json = None

        # If genui_json is not present, try to construct from root/state/elements
        if genui_json is None:
            root = row.get("root")
            state = row.get("state")
            elements = row.get("elements")
            if root is not None and elements is not None:
                genui_json = {
                    "root": root,
                    "state": state if state is not None else {},
                    "elements": elements,
                }

        # If still None, try reference field (used in some input files)
        if genui_json is None:
            reference = row.get("reference")
            if isinstance(reference, dict):
                root = reference.get("root")
                state = reference.get("state")
                elements = reference.get("elements")
                if root is not None and elements is not None:
                    genui_json = {
                        "root": root if isinstance(root, str) else "root",
                        "state": state if state is not None else {},
                        "elements": elements,
                    }

        # Unmask URLs in genui_json if requested and mapping exists
        if args.url_unmask and genui_json is not None and response_id in url_mapping_by_response:
            url_mapping = url_mapping_by_response[response_id]
            genui_json = _unmask_urls_in_object(genui_json, url_mapping)

        # Compute validation
        validation = compute_validation(genui_json)

        # Compute all metrics
        metrics = compute_all_metrics(response_text, genui_json, intent, tags)

        # Create gen object
        gen = {
            "provider": gen_info.get("provider", gen_info.get("llm_provider", "gemini")),
            "model": gen_info.get("model", "gemini-2.5-pro"),
            "prompt_version": gen_info.get("prompt_version", "genui_gen_v11_flatspec_hardcut"),
            "latency_ms": gen_info.get("latency_ms", 0.0),
            "input_tokens": gen_info.get("input_tokens", 0),
            "output_tokens": gen_info.get("output_tokens", 0),
            "cost_usd": gen_info.get("cost_usd"),
            "error": gen_info.get("error"),
        }

        # Create fixed row with expected schema
        fixed_row = {
            "ui_id": f"ui_{idx:06d}",
            "response_id": response_id,
            "query_id": query_id,
            "intent": intent,
            "tags": tags,
            "intent_bucket": intent_bucket,
            "response_text": response_text,
            "genui_json": genui_json,
            "assets": [],
            "toon": None,
            "validation": validation,
            "metrics": metrics,
            "gen": gen,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        }

        fixed_rows.append(json.dumps(fixed_row, ensure_ascii=False) + "\n")
        fixed_count += 1

        if fixed_count % 10 == 0:
            print(f"  Processed {fixed_count} rows...")

    # Write all fixed rows to output file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.writelines(fixed_rows)

    print(f"Fixed {fixed_count} rows written to {output_path}")
    print("\n=== Summary ===")
    print(f"The genui.jsonl file has been fixed with:")
    print(f"  - query_id and response_id fields for linking")
    print(f"  - intent and intent_bucket fields from queries")
    print(f"  - response_text field from responses.jsonl")
    print(f"  - validation object with COMPUTED schema_valid_strict")
    print(f"  - metrics object with ALL COMPUTED scores")
    print(f"  - gen object with model/provider info")
    print(f"  - created_at timestamp")


if __name__ == "__main__":
    main()
