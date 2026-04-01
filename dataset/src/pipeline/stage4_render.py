from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Optional
import re

from pipeline.storage import JsonlWriter, iter_jsonl, load_existing_ids, load_jsonl_by_key


def _looks_like_components_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if "id" not in item or "component" not in item:
            return False
    return True


def _dynamic_string(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if "literalString" in value:
            return {"literalString": str(value.get("literalString") or "")}
        if "literalNumber" in value:
            return {"literalNumber": value.get("literalNumber")}
        if "literalBoolean" in value:
            return {"literalBoolean": bool(value.get("literalBoolean"))}
        if "path" in value:
            return {"path": value.get("path")}
    if value is None:
        return {"literalString": ""}
    return {"literalString": str(value)}


def _split_text_blocks(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    blocks: list[dict[str, str]] = []
    buffer: list[str] = []
    mode = "text"

    def flush_buffer():
        nonlocal buffer
        if buffer:
            blocks.append({"type": "text", "text": "\n".join(buffer).strip()})
            buffer = []

    button_re = re.compile(r"[-*]?\s*\[Button:\s*(.+?)\]\s*(https?://\S+)")
    icon_re = re.compile(r"[-*]?\s*([^:]+):\s*(https?://\S+)")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_buffer()
            continue

        lower = line.lower()
        if lower.startswith("quick actions") or lower.startswith("quick action"):
            flush_buffer()
            mode = "actions"
            continue
        if lower.startswith("icons") or lower.startswith("icon") or lower.startswith("images") or lower.startswith("image"):
            flush_buffer()
            mode = "icons"
            continue
        if lower.startswith("sources"):
            flush_buffer()
            mode = "sources"
            continue

        match = button_re.match(line)
        if match:
            flush_buffer()
            blocks.append(
                {
                    "type": "button",
                    "label": match.group(1).strip(),
                    "url": match.group(2).strip(),
                }
            )
            continue

        if mode == "actions":
            match = button_re.match(line)
            if match:
                blocks.append(
                    {
                        "type": "button",
                        "label": match.group(1).strip(),
                        "url": match.group(2).strip(),
                    }
                )
                continue

        match = icon_re.match(line)
        if match and match.group(2).startswith("http"):
            if mode == "icons":
                flush_buffer()
                blocks.append(
                    {
                        "type": "icon",
                        "label": match.group(1).strip(),
                        "url": match.group(2).strip(),
                    }
                )
                continue

        buffer.append(raw_line)

    flush_buffer()
    return blocks



def _try_build_flight_layout_v09(text: str) -> list[dict[str, Any]] | None:
    lines = [line.rstrip() for line in text.splitlines()]
    known_headers = {
        "snapshot context",
        "travel requirements",
        "flight comparison",
        "flight comparison (live sources)",
        "booking cards",
        "booking options",
        "airline logos",
        "quick actions",
        "sources",
        "icons",
    }

    def first_non_empty() -> str:
        for line in lines:
            if line.strip():
                return line.strip()
        return ""

    def normalize_header(value: str) -> str:
        return re.sub(r"[:\s]+$", "", value.strip().lower())

    def section_lines(header: str) -> list[str]:
        header_l = normalize_header(header)
        start = -1
        for i, line in enumerate(lines):
            if normalize_header(line) == header_l:
                start = i + 1
                break
        if start < 0:
            return []
        out: list[str] = []
        for i in range(start, len(lines)):
            cur = lines[i].strip()
            if not cur:
                continue
            cur_header = normalize_header(cur)
            if cur_header in known_headers:
                break
            out.append(lines[i])
        return out

    if "flight booking recommendation" not in text.lower():
        return None

    title = first_non_empty()
    if not title:
        return None

    title = re.sub(r"\s*\([^)]*live[^)]*\)", "", title, flags=re.IGNORECASE).strip()

    snapshot = " ".join(s.strip() for s in section_lines("Snapshot Context") if s.strip())

    req_raw = [s.strip() for s in section_lines("Travel Requirements") if s.strip()]
    requirements: list[str] = []
    for item in req_raw:
        if item.startswith("-"):
            requirements.append(item.lstrip("- ").strip())
        else:
            requirements.append(item)

    flight_rows: list[list[str]] = []
    comparison_lines = section_lines("Flight Comparison (Live Sources)")
    if not comparison_lines:
        comparison_lines = section_lines("Flight Comparison")
    for line in comparison_lines:
        s = line.strip()
        if not s or s.lower().startswith("airline |"):
            continue
        if "|" not in s:
            continue
        parts = [p.strip() for p in s.split("|")]
        if len(parts) >= 7:
            flight_rows.append(parts[:7])

    fx_to_gbp = {
        "GBP": 1.0,
        "EUR": 0.86,
        "USD": 0.79,
    }

    def normalize_fare_to_gbp(value: str) -> str:
        s = str(value or "").strip()
        if not s:
            return s
        currency = ""
        amount_part = s
        upper = s.upper()
        if upper.startswith("GBP"):
            currency = "GBP"
            amount_part = s[3:].strip()
        elif upper.startswith("EUR"):
            currency = "EUR"
            amount_part = s[3:].strip()
        elif upper.startswith("USD"):
            currency = "USD"
            amount_part = s[3:].strip()
        elif s.startswith("\u00a3"):
            currency = "GBP"
            amount_part = s[1:].strip()
        elif s.startswith("\u20ac"):
            currency = "EUR"
            amount_part = s[1:].strip()
        elif s.startswith("$"):
            currency = "USD"
            amount_part = s[1:].strip()

        if not currency:
            return s

        amount_match = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", amount_part)
        if not amount_match:
            return s

        try:
            amount = float(amount_match.group(1).replace(",", ""))
        except ValueError:
            return s

        gbp_value = amount * fx_to_gbp.get(currency, 1.0)
        shown = f"{int(round(gbp_value)):,}"
        return f"GBP {shown}"

    for row in flight_rows:
        if len(row) >= 2:
            row[1] = normalize_fare_to_gbp(row[1])

    def normalize_airline_name(value: str) -> str:
        base = re.sub(r"\(.*?\)", "", value).strip().lower()
        if "british airways" in base or base == "ba":
            return "british airways"
        if "emirates" in base:
            return "emirates"
        if "ana" in base or "all nippon" in base:
            return "ana"
        return re.sub(r"[^a-z0-9]+", " ", base).strip()

    flight_row_by_airline: dict[str, dict[str, str]] = {}
    for row in flight_rows:
        flight_row_by_airline[normalize_airline_name(row[0])] = {
            "airline": row[0],
            "fare": row[1],
            "departure": row[2],
            "arrival": row[3],
            "travel_time": row[4],
            "stops": row[5],
            "decision": row[6],
        }

    option_re = re.compile(r"^Option\s*(\d+):\s*(.*?)\s*\|\s*(.*)$", re.IGNORECASE)
    action_re = re.compile(r"^(?:Action:\s*)?\[Button:\s*(.+?)\]\s*(https?://\S+)", re.IGNORECASE)
    options: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in section_lines("Booking Cards"):
        line = raw.strip()
        if not line:
            continue
        m_opt = option_re.match(line)
        if m_opt:
            current = {
                "title": m_opt.group(2).strip(),
                "desc": m_opt.group(3).strip(),
            }
            options.append(current)
            continue
        m_act = action_re.match(line)
        if m_act and current is not None:
            current["button_label"] = m_act.group(1).strip()
            current["button_url"] = m_act.group(2).strip()

    logo_map: dict[str, str] = {}
    logo_line_re = re.compile(r"^-\s*([^:]+):\s*(\S+)")
    for raw in section_lines("Airline Logos"):
        m = logo_line_re.match(raw.strip())
        if not m:
            continue
        k = m.group(1).strip().lower()
        v = m.group(2).strip()
        logo_map[k] = v

    def pick_logo(name: str) -> str | None:
        n = name.lower()
        if "ana" in n:
            return logo_map.get("ana")
        if "british" in n or n == "ba":
            return logo_map.get("british airways") or logo_map.get("ba")
        if "emirates" in n:
            return logo_map.get("emirates")
        return None

    quick_actions: list[dict[str, str]] = []
    qa_re = re.compile(r"^(?:Action:\s*)?\[Button:\s*(.+?)\]\s*(https?://\S+)", re.IGNORECASE)
    for raw in section_lines("Quick Actions"):
        m = qa_re.match(raw.strip())
        if m:
            quick_actions.append({"label": m.group(1).strip(), "url": m.group(2).strip()})

    source_links: list[dict[str, str]] = []
    src_re = re.compile(r"^([^:]+):\s*(https?://\S+)")
    for raw in section_lines("Sources"):
        stripped = raw.strip()
        m = src_re.match(stripped)
        if m:
            source_links.append({"label": m.group(1).strip(), "url": m.group(2).strip()})
            continue
        if stripped.startswith("http://") or stripped.startswith("https://"):
            source_links.append({"label": stripped, "url": stripped})

    if not requirements and not flight_rows and not options:
        return None

    components: list[dict[str, Any]] = []
    root_children: list[str] = []

    def add_text(cid: str, value: str, variant: str = "body", weight: float | None = None) -> None:
        comp: dict[str, Any] = {"id": cid, "component": "Text", "text": value, "variant": variant}
        if weight is not None:
            comp["weight"] = weight
        components.append(comp)

    def add_button(cid: str, label: str, url: str, variant: str = "primary") -> None:
        label_id = f"{cid}_label"
        add_text(label_id, label)
        components.append(
            {
                "id": cid,
                "component": "Button",
                "child": label_id,
                "variant": variant,
                "action": {
                    "functionCall": {
                        "call": "openUrl",
                        "args": {"url": {"literalString": url}},
                        "returnType": "void",
                    }
                },
            }
        )

    add_text("flight_title", title, "h2")
    root_children.append("flight_title")

    if snapshot:
        add_text("flight_snapshot", snapshot)
        root_children.append("flight_snapshot")

    if requirements:
        add_text("req_header", "Travel Requirements", "h3")
        root_children.append("req_header")
        req_col_children: list[str] = []
        for i, item in enumerate(requirements, start=1):
            tid = f"req_{i}"
            add_text(tid, item)
            req_col_children.append(tid)
        components.append({"id": "req_col", "component": "Column", "children": req_col_children})
        components.append({"id": "req_card", "component": "Card", "child": "req_col"})
        root_children.append("req_card")

    if flight_rows:
        add_text("cmp_header", "Flight Comparison", "h3")
        root_children.append("cmp_header")
        table_children: list[str] = []

        headers = ["Airline", "Fare", "Departure", "Arrival", "Travel Time", "Stops", "Decision"]
        header_ids: list[str] = []
        for idx, h in enumerate(headers, start=1):
            hid = f"cmp_h_{idx}"
            add_text(hid, h, "h4", 1.0)
            header_ids.append(hid)
        components.append({"id": "cmp_header_row", "component": "Row", "children": header_ids})
        table_children.append("cmp_header_row")

        for r_idx, row in enumerate(flight_rows, start=1):
            divider_id = f"cmp_div_{r_idx}"
            components.append({"id": divider_id, "component": "Divider"})
            table_children.append(divider_id)

            row_ids: list[str] = []
            for c_idx, val in enumerate(row[:7], start=1):
                tid = f"cmp_r{r_idx}_c{c_idx}"
                add_text(tid, val, "body", 1.0)
                row_ids.append(tid)
            rid = f"cmp_row_{r_idx}"
            components.append({"id": rid, "component": "Row", "children": row_ids})
            table_children.append(rid)

        components.append({"id": "cmp_table_col", "component": "Column", "children": table_children})
        components.append({"id": "cmp_table_card", "component": "Card", "child": "cmp_table_col"})
        root_children.append("cmp_table_card")

    if options:
        add_text("opt_header", "Booking Options", "h3")
        root_children.append("opt_header")

        row_children: list[str] = []
        for i, opt in enumerate(options, start=1):
            card_id = f"opt_card_{i}"
            col_id = f"opt_col_{i}"
            row_children.append(card_id)

            content_children: list[str] = []
            logo_url = pick_logo(opt.get("title", ""))
            title_text = opt.get("title", f"Option {i}")

            if logo_url:
                logo_id = f"opt_logo_{i}"
                title_id = f"opt_title_{i}"
                add_text(title_id, title_text, "h4")
                components.append({"id": logo_id, "component": "Image", "url": logo_url, "variant": "icon"})
                row_id = f"opt_head_{i}"
                components.append({"id": row_id, "component": "Row", "children": [logo_id, title_id], "align": "center"})
                content_children.append(row_id)
            else:
                title_id = f"opt_title_{i}"
                add_text(title_id, title_text, "h4")
                content_children.append(title_id)

            desc_id = f"opt_desc_{i}"
            add_text(desc_id, opt.get("desc", ""))
            content_children.append(desc_id)

            detail = flight_row_by_airline.get(normalize_airline_name(title_text))
            if detail is not None:
                kv_rows = [
                    ("Snapshot fare", detail["fare"]),
                    ("Departure", detail["departure"]),
                    ("Arrival", detail["arrival"]),
                    ("Duration", detail["travel_time"]),
                    ("Stops", detail["stops"]),
                ]
                for j, (label, value) in enumerate(kv_rows, start=1):
                    label_id = f"opt_{i}_kv_{j}_label"
                    value_id = f"opt_{i}_kv_{j}_value"
                    row_id = f"opt_{i}_kv_{j}_row"
                    add_text(label_id, label, "body", 1.0)
                    add_text(value_id, value, "body", 1.0)
                    components.append(
                        {
                            "id": row_id,
                            "component": "Row",
                            "children": [label_id, value_id],
                            "justify": "spaceBetween",
                        }
                    )
                    content_children.append(row_id)

            btn_url = opt.get("button_url")
            btn_label = opt.get("button_label")
            if btn_url and btn_label:
                btn_id = f"opt_btn_{i}"
                add_button(btn_id, btn_label, btn_url)
                content_children.append(btn_id)

            components.append({"id": col_id, "component": "Column", "children": content_children})
            components.append({"id": card_id, "component": "Card", "child": col_id, "weight": 1})

        components.append(
            {
                "id": "opt_row",
                "component": "Row",
                "children": row_children,
                "align": "start",
                "justify": "spaceBetween",
            }
        )
        root_children.append("opt_row")

    if quick_actions:
        add_text("qa_header", "Quick Actions", "h3")
        root_children.append("qa_header")
        qa_row_children: list[str] = []
        for i, qa in enumerate(quick_actions, start=1):
            bid = f"qa_btn_{i}"
            add_button(bid, qa["label"], qa["url"])
            qa_row_children.append(bid)
        components.append({"id": "qa_row", "component": "Row", "children": qa_row_children, "justify": "start"})
        root_children.append("qa_row")

    if source_links:
        add_text("src_header", "Sources", "h4")
        root_children.append("src_header")
        src_children: list[str] = []
        for i, src in enumerate(source_links, start=1):
            sid = f"src_btn_{i}"
            add_button(sid, src["label"], src["url"], "borderless")
            src_children.append(sid)
        components.append({"id": "src_col", "component": "Column", "children": src_children})
        root_children.append("src_col")

    components.insert(0, {"id": "root", "component": "Column", "children": root_children})
    return components

def _expand_text_components_v09(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not components:
        return components
    comp_by_id = {comp.get("id"): comp for comp in components if isinstance(comp, dict)}
    root = comp_by_id.get("root") or components[0]
    if not isinstance(root, dict):
        return components
    if root.get("component") != "Column":
        return components
    children = root.get("children")
    if not isinstance(children, list) or len(children) != 1:
        return components
    text_id = children[0]
    text_comp = comp_by_id.get(text_id)
    if not isinstance(text_comp, dict) or text_comp.get("component") != "Text":
        return components
    text_value = text_comp.get("text")
    if not isinstance(text_value, str):
        return components

    flight_layout = _try_build_flight_layout_v09(text_value)
    if flight_layout is not None:
        return flight_layout

    blocks = _split_text_blocks(text_value)
    if not any(block["type"] in ("button", "icon") for block in blocks):
        return components

    new_components: list[dict[str, Any]] = []
    new_children: list[str] = []
    idx = 1
    for block in blocks:
        if block["type"] == "text":
            comp_id = f"text_{idx}"
            new_components.append(
                {
                    "id": comp_id,
                    "component": "Text",
                    "text": block["text"],
                    "variant": "body",
                }
            )
            new_children.append(comp_id)
            idx += 1
            continue
        if block["type"] == "button":
            label_id = f"btn_label_{idx}"
            button_id = f"btn_{idx}"
            new_components.append(
                {
                    "id": label_id,
                    "component": "Text",
                    "text": block["label"],
                    "variant": "body",
                }
            )
            new_components.append(
                {
                    "id": button_id,
                    "component": "Button",
                    "child": label_id,
                    "variant": "primary",
                    "action": {
                        "functionCall": {
                            "call": "openUrl",
                            "args": {"url": {"literalString": block["url"]}},
                            "returnType": "void",
                        }
                    },
                }
            )
            new_children.append(button_id)
            idx += 1
            continue
        if block["type"] == "icon":
            image_id = f"icon_img_{idx}"
            label_id = f"icon_label_{idx}"
            row_id = f"icon_row_{idx}"
            new_components.append(
                {
                    "id": image_id,
                    "component": "Image",
                    "url": block["url"],
                    "variant": "icon",
                }
            )
            new_components.append(
                {
                    "id": label_id,
                    "component": "Text",
                    "text": block["label"],
                    "variant": "body",
                }
            )
            new_components.append(
                {
                    "id": row_id,
                    "component": "Row",
                    "children": [image_id, label_id],
                    "align": "center",
                }
            )
            new_children.append(row_id)
            idx += 1
            continue

    root_copy = dict(root)
    root_copy["children"] = new_children
    new_components.insert(0, root_copy)
    return new_components



_MISSING = object()


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _json_pointer_get(root: Any, path: str) -> Any:
    if path in ("", "/"):
        return root
    if not isinstance(path, str) or not path.startswith("/"):
        return _MISSING
    tokens = [_decode_pointer_token(part) for part in path.lstrip("/").split("/")]
    cur = root
    for token in tokens:
        if isinstance(cur, dict):
            if token not in cur:
                return _MISSING
            cur = cur[token]
            continue
        if isinstance(cur, list):
            if not token.isdigit():
                return _MISSING
            idx = int(token)
            if idx < 0 or idx >= len(cur):
                return _MISSING
            cur = cur[idx]
            continue
        return _MISSING
    return cur


def _json_pointer_set(root: Any, path: str, value: Any) -> Any:
    if path in ("", "/"):
        return deepcopy(value)

    if not isinstance(path, str):
        return deepcopy(root)

    if not isinstance(root, (dict, list)):
        root = {}
    target = deepcopy(root)

    normalized = path if path.startswith("/") else "/" + path.lstrip("/")
    tokens = [_decode_pointer_token(part) for part in normalized.lstrip("/").split("/")]
    if not tokens:
        return deepcopy(value)

    cur = target
    for token in tokens[:-1]:
        if isinstance(cur, dict):
            nxt = cur.get(token)
            if not isinstance(nxt, (dict, list)):
                nxt = {}
                cur[token] = nxt
            cur = nxt
            continue
        if isinstance(cur, list):
            if not token.isdigit():
                return target
            idx = int(token)
            while len(cur) <= idx:
                cur.append({})
            nxt = cur[idx]
            if not isinstance(nxt, (dict, list)):
                nxt = {}
                cur[idx] = nxt
            cur = nxt
            continue
        return target

    last = tokens[-1]
    if isinstance(cur, dict):
        cur[last] = deepcopy(value)
    elif isinstance(cur, list):
        if not last.isdigit():
            return target
        idx = int(last)
        while len(cur) <= idx:
            cur.append(None)
        cur[idx] = deepcopy(value)
    return target


def _resolve_relative_path(root: Any, path: str) -> Any:
    if path == "":
        return root
    parts = path.split(".") if "/" not in path else [p for p in path.split("/") if p]
    cur = root
    for token in parts:
        if isinstance(cur, dict):
            if token not in cur:
                return _MISSING
            cur = cur[token]
            continue
        if isinstance(cur, list):
            if not token.isdigit():
                return _MISSING
            idx = int(token)
            if idx < 0 or idx >= len(cur):
                return _MISSING
            cur = cur[idx]
            continue
        return _MISSING
    return cur


def _resolve_binding_path(path: Any, row_item: Any, model: Any) -> Any:
    if not isinstance(path, str):
        return _MISSING
    candidate = path.strip()
    if not candidate:
        return _MISSING
    if candidate.startswith("/"):
        return _json_pointer_get(model, candidate)
    if candidate.startswith("$."):
        candidate = candidate[2:]
    val = _resolve_relative_path(row_item, candidate)
    if val is not _MISSING:
        return val
    return _resolve_relative_path(model, candidate)


def _apply_component_bindings_v09(component: dict[str, Any], row_item: Any, model: Any) -> None:
    comp_type = component.get("component")

    def resolve_dynamic(value: Any) -> Any:
        if isinstance(value, dict) and set(value.keys()) == {"path"}:
            resolved = _resolve_binding_path(value.get("path"), row_item, model)
            return "" if resolved is _MISSING else resolved
        return value

    if comp_type == "Text":
        component["text"] = resolve_dynamic(component.get("text"))
        return

    if comp_type == "Image":
        component["url"] = resolve_dynamic(component.get("url"))
        return

    if comp_type == "Icon":
        component["name"] = resolve_dynamic(component.get("name"))
        return

    if comp_type == "Button":
        action = component.get("action")
        if not isinstance(action, dict):
            return
        fn = action.get("functionCall")
        if not isinstance(fn, dict):
            return
        args = fn.get("args")
        if not isinstance(args, dict):
            return
        for key, value in list(args.items()):
            args[key] = resolve_dynamic(value)
        return

    if comp_type == "Tabs":
        tabs = component.get("tabs")
        if not isinstance(tabs, list):
            return
        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            tab["title"] = resolve_dynamic(tab.get("title"))


def _clone_repeated_subtree_v09(
    root_component_id: str,
    source_components: dict[str, dict[str, Any]],
    row_item: Any,
    model: Any,
    suffix: str,
    id_counter: Any,
) -> tuple[str | None, list[dict[str, Any]]]:
    created: list[dict[str, Any]] = []
    local_cache: dict[str, str] = {}

    def clone_component(component_id: str) -> str | None:
        if component_id in local_cache:
            return local_cache[component_id]
        source = source_components.get(component_id)
        if not isinstance(source, dict):
            return None

        clone = deepcopy(source)
        new_id = f"{component_id}__{suffix}_{next(id_counter)}"
        local_cache[component_id] = new_id
        clone["id"] = new_id

        child = clone.get("child")
        if isinstance(child, str):
            child_clone = clone_component(child)
            if child_clone:
                clone["child"] = child_clone

        children = clone.get("children")
        if isinstance(children, list):
            rewritten_children: list[Any] = []
            for child_id in children:
                if isinstance(child_id, str):
                    child_clone = clone_component(child_id)
                    if child_clone:
                        rewritten_children.append(child_clone)
                else:
                    rewritten_children.append(child_id)
            clone["children"] = rewritten_children
        elif isinstance(children, dict):
            explicit = children.get("explicitList")
            if isinstance(explicit, list):
                rewritten_explicit: list[Any] = []
                for child_id in explicit:
                    if isinstance(child_id, str):
                        child_clone = clone_component(child_id)
                        if child_clone:
                            rewritten_explicit.append(child_clone)
                    else:
                        rewritten_explicit.append(child_id)
                rewritten_children_dict = dict(children)
                rewritten_children_dict["explicitList"] = rewritten_explicit
                clone["children"] = rewritten_children_dict

        tabs = clone.get("tabs")
        if isinstance(tabs, list):
            rewritten_tabs: list[dict[str, Any]] = []
            for tab in tabs:
                if not isinstance(tab, dict):
                    continue
                tab_copy = deepcopy(tab)
                tab_child = tab_copy.get("child")
                if isinstance(tab_child, str):
                    tab_child_clone = clone_component(tab_child)
                    if tab_child_clone:
                        tab_copy["child"] = tab_child_clone
                rewritten_tabs.append(tab_copy)
            clone["tabs"] = rewritten_tabs

        _apply_component_bindings_v09(clone, row_item, model)
        created.append(clone)
        return new_id

    root_id = clone_component(root_component_id)
    return root_id, created


def _expand_repeated_children_v09(components: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    if not components:
        return components

    expanded: list[dict[str, Any]] = []
    for comp in components:
        if isinstance(comp, dict):
            expanded.append(deepcopy(comp))

    source_components: dict[str, dict[str, Any]] = {
        comp["id"]: comp for comp in expanded if isinstance(comp.get("id"), str)
    }
    id_counter = count(1)
    appended: list[dict[str, Any]] = []

    for comp in expanded:
        children = comp.get("children")
        if not isinstance(children, dict):
            continue

        template_id = children.get("componentId")
        path = children.get("path")
        if not isinstance(template_id, str) or not isinstance(path, str):
            continue

        data_items = _resolve_binding_path(path, None, model)
        if not isinstance(data_items, list):
            comp["children"] = []
            continue

        repeated_child_ids: list[str] = []
        for index, row_item in enumerate(data_items):
            suffix = f"rep{index}"
            row_root_id, row_components = _clone_repeated_subtree_v09(
                root_component_id=template_id,
                source_components=source_components,
                row_item=row_item,
                model=model,
                suffix=suffix,
                id_counter=id_counter,
            )
            if isinstance(row_root_id, str):
                repeated_child_ids.append(row_root_id)
            if row_components:
                appended.extend(row_components)

        comp["children"] = repeated_child_ids

    if appended:
        expanded.extend(appended)
    return expanded

def _convert_component_v09_to_v08(component: dict[str, Any]) -> dict[str, Any]:
    comp_id = component.get("id") or "component"
    comp_type = component.get("component")
    if comp_type == "Text":
        payload: dict[str, Any] = {
            "text": _dynamic_string(component.get("text")),
            "usageHint": component.get("variant", "body"),
        }
        weight = component.get("weight")
        if isinstance(weight, (int, float)) and weight > 0:
            payload["weight"] = float(weight)
        return {
            "id": comp_id,
            "component": {
                "Text": payload
            },
        }
    if comp_type == "Image":
        payload: dict[str, Any] = {"url": _dynamic_string(component.get("url"))}
        usage = component.get("variant")
        if usage == "icon":
            payload["usageHint"] = usage
        fit = component.get("fit")
        if fit:
            payload["fit"] = fit
        return {"id": comp_id, "component": {"Image": payload}}
    if comp_type == "Icon":
        return {
            "id": comp_id,
            "component": {"Icon": {"name": _dynamic_string(component.get("name"))}},
        }
    if comp_type in ("Column", "Row", "List"):
        children = component.get("children")
        if isinstance(children, dict):
            explicit_list = children.get("explicitList") if "explicitList" in children else None
            if isinstance(explicit_list, list):
                children = explicit_list
            else:
                children = []
        if children is None:
            children = []
        if not isinstance(children, list):
            children = [children]
        payload: dict[str, Any] = {"children": {"explicitList": children}}
        if comp_type in ("Column", "Row"):
            justify = component.get("justify")
            align = component.get("align")
            if justify:
                payload["distribution"] = justify
            if align:
                payload["alignment"] = align
        if comp_type == "List":
            direction = component.get("direction")
            if direction:
                payload["direction"] = direction
            align = component.get("align")
            if align:
                payload["alignment"] = align
        return {"id": comp_id, "component": {comp_type: payload}}
    if comp_type == "Divider":
        payload: dict[str, Any] = {}
        axis = component.get("axis")
        if axis:
            payload["axis"] = axis
        return {"id": comp_id, "component": {"Divider": payload}}
    if comp_type == "Card":
        child = component.get("child")
        if isinstance(child, str):
            return {"id": comp_id, "component": {"Card": {"child": child}}}
    if comp_type == "Tabs":
        tabs = component.get("tabs") or component.get("items") or []
        tab_items: list[dict[str, Any]] = []
        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            title = tab.get("title") or tab.get("label")
            child = tab.get("child")
            if not child:
                continue
            tab_items.append({"title": _dynamic_string(title), "child": child})
        if tab_items:
            return {"id": comp_id, "component": {"Tabs": {"tabItems": tab_items}}}
    if comp_type == "Button":
        child = component.get("child")
        if isinstance(child, str):
            action_payload = None
            action = component.get("action")
            if isinstance(action, dict):
                if "functionCall" in action:
                    fn = action.get("functionCall") or {}
                    if isinstance(fn, dict) and fn.get("call"):
                        args = fn.get("args") or {}
                        context = []
                        for key, value in (args.items() if isinstance(args, dict) else []):
                            context.append({"key": str(key), "value": _dynamic_string(value)})
                        action_payload = {"name": fn.get("call"), "context": context} if context else {"name": fn.get("call")}
                if "event" in action:
                    ev = action.get("event") or {}
                    if isinstance(ev, dict) and ev.get("name"):
                        ctx = ev.get("context") or {}
                        context = []
                        if isinstance(ctx, dict):
                            for key, value in ctx.items():
                                context.append({"key": str(key), "value": _dynamic_string(value)})
                        action_payload = {"name": ev.get("name"), "context": context} if context else {"name": ev.get("name")}
            if not action_payload:
                action_payload = {"name": "noop"}
            payload: dict[str, Any] = {"child": child, "action": action_payload}
            if component.get("variant") == "primary":
                payload["primary"] = True
            return {"id": comp_id, "component": {"Button": payload}}
    # Fallback for unsupported components: render a text stub.
    return {
        "id": comp_id,
        "component": {
            "Text": {
                "text": _dynamic_string(f"[{comp_type or 'Unknown'}]"),
                "usageHint": "body",
            }
        },
    }


def _extract_open_url_action(component: dict[str, Any]) -> str | None:
    action = component.get("action")
    if not isinstance(action, dict):
        return None
    fn = action.get("functionCall")
    if not isinstance(fn, dict):
        return None
    if fn.get("call") != "openUrl":
        return None
    args = fn.get("args")
    if isinstance(args, dict):
        url = args.get("url")
        if isinstance(url, str):
            return url.strip() or None
        if isinstance(url, dict):
            lit = url.get("literalString")
            if isinstance(lit, str):
                return lit.strip() or None
    return None


def _to_clickable_link_text_component(
    button_component: dict[str, Any], comp_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comp_id = button_component.get("id") or "component"
    child_id = button_component.get("child")
    label = None
    if isinstance(child_id, str):
        child_comp = comp_by_id.get(child_id)
        if isinstance(child_comp, dict) and child_comp.get("component") == "Text":
            label_value = child_comp.get("text")
            if isinstance(label_value, str):
                label = label_value.strip()
    if not label:
        label = "Open link"
    url = _extract_open_url_action(button_component)
    text = f"[{label}]({url})" if url else label
    return {
        "id": comp_id,
        "component": {
            "Text": {
                "text": _dynamic_string(text),
                "usageHint": "body",
            }
        },
    }


def _wrap_components_as_messages(components: list[dict], surface_id: str = "@default") -> list[dict]:
    comp_by_id: dict[str, dict[str, Any]] = {}
    for comp in components:
        if isinstance(comp, dict) and isinstance(comp.get("id"), str):
            comp_by_id[comp["id"]] = comp

    borderless_button_child_ids: set[str] = set()
    for comp in components:
        if not isinstance(comp, dict):
            continue
        if comp.get("component") != "Button":
            continue
        if comp.get("variant") != "borderless":
            continue
        child = comp.get("child")
        if isinstance(child, str):
            borderless_button_child_ids.add(child)

    converted: list[dict[str, Any]] = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        comp_id = comp.get("id")
        if isinstance(comp_id, str) and comp_id in borderless_button_child_ids:
            # Skip text nodes that were only used as borderless button labels.
            continue
        if comp.get("component") == "Button" and comp.get("variant") == "borderless":
            converted.append(_to_clickable_link_text_component(comp, comp_by_id))
        else:
            converted.append(_convert_component_v09_to_v08(comp))

    root_id = converted[0].get("id") if converted else "root"
    return [
        {"beginRendering": {"root": root_id, "surfaceId": surface_id}},
        {"surfaceUpdate": {"surfaceId": surface_id, "components": converted}},
    ]


def _has_message_content(messages: list[Any]) -> bool:
    keys = {
        "beginRendering",
        "surfaceUpdate",
        "deleteSurface",
        "dataModelUpdate",
        "dataModelDelete",
        "dataModelTransaction",
        "createSurface",
        "updateComponents",
        "updateDataModel",
    }
    for item in messages:
        if isinstance(item, dict) and keys.intersection(item.keys()):
            return True
    return False


def _fallback_messages(text: str) -> list[dict]:
    return _wrap_components_as_messages(
        [
            {
                "id": "root",
                "component": "Column",
                "children": ["fallback-text"],
            },
            {
                "id": "fallback-text",
                "component": "Text",
                "text": text,
                "variant": "body",
            },
        ]
    )


def _convert_genui_messages_to_genui(messages: list[Any]) -> list[Any]:
    # Detect GenUICraft v0.9 style messages and convert to v0.8 genui messages.
    if not isinstance(messages, list):
        return messages
    surfaces: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for item in messages:
        if not isinstance(item, dict):
            continue

        if "createSurface" in item:
            surface_id = item.get("createSurface", {}).get("surfaceId")
            if surface_id and surface_id not in surfaces:
                surfaces[surface_id] = {}
                order.append(surface_id)

        if "updateDataModel" in item:
            payload = item.get("updateDataModel") or {}
            surface_id = payload.get("surfaceId")
            if surface_id:
                if surface_id not in surfaces:
                    surfaces[surface_id] = {}
                    order.append(surface_id)
                model_path = payload.get("path") or "/"
                model_value = payload.get("value")
                existing_model = surfaces[surface_id].get("model", {})
                surfaces[surface_id]["model"] = _json_pointer_set(existing_model, model_path, model_value)

        if "updateComponents" in item:
            surface_payload = item.get("updateComponents") or {}
            surface_id = surface_payload.get("surfaceId")
            comps = surface_payload.get("components") or []
            if surface_id:
                if surface_id not in surfaces:
                    surfaces[surface_id] = {}
                    order.append(surface_id)
                surfaces[surface_id]["components"] = comps

    if not surfaces:
        return messages

    output: list[dict[str, Any]] = []
    for surface_id in order:
        surface_state = surfaces.get(surface_id, {})
        components = surface_state.get("components") or []
        model = surface_state.get("model", {})
        components = _expand_text_components_v09(components)
        components = _expand_repeated_children_v09(components, model)
        output.extend(_wrap_components_as_messages(components, surface_id=surface_id))
    return output


def _normalize_messages(value: Any) -> list[Any]:
    if isinstance(value, list):
        if _looks_like_components_list(value):
            return _wrap_components_as_messages(value)
        return _convert_genui_messages_to_genui(value)
    if isinstance(value, dict):
        for key in ("messages", "payload"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                if _looks_like_components_list(candidate):
                    return _wrap_components_as_messages(candidate)
                return _convert_genui_messages_to_genui(candidate)
    return [value]


def _rewrite_local_asset_urls(value: Any) -> Any:
    def _rewrite(s: str) -> str:
        if s.startswith("../assets/"):
            return s
        if s.startswith("/assets/"):
            return "../" + s.lstrip("/")
        if s.startswith("./assets/"):
            return "../" + s.lstrip("./")
        return s

    if isinstance(value, str):
        return _rewrite(value)
    if isinstance(value, list):
        return [_rewrite_local_asset_urls(item) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_local_asset_urls(item) for key, item in value.items()}
    return value


def _safe_json_dumps(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text.replace("</", "<\\/")


def _compute_renderer_assets_hash(assets_dir: Path, template_text: str) -> str:
    """
    Hash template + OneUI overlay assets so Stage4 can re-render automatically
    when styling changes, even if render.jsonl already has successful entries.
    """
    hasher = hashlib.sha256()
    hasher.update(template_text.encode("utf-8"))

    themes_dir = assets_dir / "themes"
    if themes_dir.exists():
        for path in sorted(p for p in themes_dir.rglob("*") if p.is_file()):
            hasher.update(path.relative_to(assets_dir).as_posix().encode("utf-8"))
            hasher.update(path.read_bytes())

    return hasher.hexdigest()


def _ensure_trailing_slash(uri: str) -> str:
    return uri if uri.endswith("/") else uri + "/"


class HtmlRenderer:
    def __init__(
        self,
        viewport: dict[str, int],
        timeout_ms: int,
        wait_ms: int,
        emulate_mobile: bool = False,
    ) -> None:
        self.viewport = viewport
        self.timeout_ms = timeout_ms
        self.wait_ms = wait_ms
        self.emulate_mobile = emulate_mobile
        self._playwright = None
        self._browser = None

    def start(self) -> Optional[str]:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception as exc:
            return f"playwright_not_installed: {exc}"
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch()
        return None

    def stop(self) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._browser = None
        self._playwright = None

    def render(self, url: str, image_path: Path) -> Optional[str]:
        if not self._browser:
            return "renderer_not_initialized"
        new_page_args: dict[str, Any] = {"viewport": self.viewport}
        if self.emulate_mobile:
            new_page_args["is_mobile"] = True
            new_page_args["has_touch"] = True
            new_page_args["device_scale_factor"] = 2
        page = self._browser.new_page(**new_page_args)
        console_logs: list[str] = []
        page.on("console", lambda msg: console_logs.append(f"{msg.type}: {msg.text}"))
        page.on("pageerror", lambda err: console_logs.append(f"pageerror: {err}"))
        page.on(
            "requestfailed",
            lambda req: console_logs.append(f"requestfailed: {req.url} {req.failure}"),
        )
        try:
            page.goto(url)
            page.wait_for_function("window.GenUICraft_READY === true", timeout=self.timeout_ms)
            page.wait_for_function("window.__GenUICraft_RENDER_DONE === true", timeout=self.timeout_ms)
            if self.wait_ms > 0:
                page.wait_for_timeout(self.wait_ms)
            page.screenshot(path=str(image_path), full_page=True)
            return None
        except Exception as exc:
            detail = "; ".join(console_logs[:5])
            if detail:
                return f"render_error: {exc}; console: {detail}"
            return f"render_error: {exc}"
        finally:
            page.close()


class _QuietHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".json": "application/json",
        ".css": "text/css",
        ".svg": "image/svg+xml",
        ".wasm": "application/wasm",
    }

    def log_message(self, format, *args):
        return


class HttpServer:
    def __init__(self, root: Path, port: int = 0) -> None:
        handler = lambda *args, **kwargs: _QuietHandler(*args, directory=str(root), **kwargs)
        self._server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _render_chunk(
    chunk: list[dict[str, Any]],
    viewport: dict[str, int],
    timeout_ms: int,
    wait_ms: int,
    emulate_mobile: bool,
) -> list[tuple[str, Optional[str]]]:
    renderer = HtmlRenderer(
        viewport=viewport,
        timeout_ms=timeout_ms,
        wait_ms=wait_ms,
        emulate_mobile=emulate_mobile,
    )
    start_error = renderer.start()
    if start_error:
        return [(item["ui_id"], start_error) for item in chunk]

    results: list[tuple[str, Optional[str]]] = []
    try:
        for item in chunk:
            error_text = renderer.render(item["url"], item["image_path"])
            results.append((item["ui_id"], error_text))
    except Exception as exc:
        # Fallback so one worker failure does not abort the whole stage.
        failure = f"render_worker_error: {exc}"
        for item in chunk:
            results.append((item["ui_id"], failure))
    finally:
        renderer.stop()
    return results


def _render_parallel(
    tasks: list[dict[str, Any]],
    workers: int,
    viewport: dict[str, int],
    timeout_ms: int,
    wait_ms: int,
    emulate_mobile: bool,
) -> dict[str, Optional[str]]:
    workers = max(1, min(int(workers), len(tasks)))
    chunks = [tasks[i::workers] for i in range(workers)]

    errors_by_ui: dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _render_chunk,
                chunk,
                viewport,
                timeout_ms,
                wait_ms,
                emulate_mobile,
            )
            for chunk in chunks
            if chunk
        ]
        for future in as_completed(futures):
            for ui_id, render_error in future.result():
                errors_by_ui[ui_id] = render_error
    return errors_by_ui


def run_stage4(
    genui_path: Path,
    output_dir: Path,
    assets_dir: Path,
    server_root: Optional[Path],
    logger,
    max_total: int | None = None,
    render_images: bool = True,
    image_format: str = "png",
    viewport: Optional[dict[str, int]] = None,
    timeout_ms: int = 15000,
    wait_ms: int = 200,
    use_http_server: bool = True,
    parallel_workers: int = 1,
    emulate_mobile: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = assets_dir / "template.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    renderer_assets_hash = _compute_renderer_assets_hash(assets_dir, template)

    render_log_path = output_dir.parent / "render.jsonl"
    existing_rows = load_jsonl_by_key(render_log_path, "ui_id")
    existing = set(existing_rows.keys())
    writer = JsonlWriter(render_log_path)

    server = None
    server_root = server_root or output_dir.parents[3]
    if render_images and use_http_server:
        server = HttpServer(server_root)
        server.start()
        logger.info("Stage4 HTTP server started on port %s", server.port)

    relative_base = Path(os.path.relpath(assets_dir, output_dir)).as_posix()
    asset_base = _ensure_trailing_slash(relative_base)
    viewport = viewport or {"width": 1280, "height": 720}
    parallel_workers = max(1, int(parallel_workers))

    renderer = HtmlRenderer(
        viewport=viewport,
        timeout_ms=timeout_ms,
        wait_ms=wait_ms,
        emulate_mobile=emulate_mobile,
    )
    renderer_error = None
    if render_images and parallel_workers == 1:
        renderer_error = renderer.start()
        if renderer_error:
            logger.error("Stage4 renderer unavailable: %s", renderer_error)

    created = 0
    pending_parallel: list[dict[str, Any]] = []
    pending_records: list[dict[str, Any]] = []
    for row in iter_jsonl(genui_path):
        ui_id = row.get("ui_id")
        if not ui_id:
            continue
        if ui_id in existing:
            prev = existing_rows.get(ui_id) or {}
            prev_render = prev.get("render") if isinstance(prev, dict) else None
            prev_error = prev_render.get("error") if isinstance(prev_render, dict) else None
            prev_image = prev.get("image_path") if isinstance(prev, dict) else None
            prev_assets_hash = (
                prev_render.get("renderer_assets_hash")
                if isinstance(prev_render, dict)
                else None
            )
            needs_rerender_for_assets = prev_assets_hash != renderer_assets_hash
            needs_rerender_for_image = render_images and (prev_error or not prev_image)
            if not (needs_rerender_for_assets or needs_rerender_for_image):
                continue
        if max_total is not None and created >= max_total:
            logger.info("Stage4 reached max_total=%s", max_total)
            break

        genui_json = row.get("genui_json")
        if genui_json is None:
            genui_json = row.get("a2ui_json")
        messages = _normalize_messages(genui_json)
        messages = _rewrite_local_asset_urls(messages)
        if not messages or not _has_message_content(messages):
            err_text = "No renderable GenUICraft messages."
            validation = row.get("validation") if isinstance(row, dict) else None
            errors = validation.get("errors") if isinstance(validation, dict) else None
            if isinstance(errors, list) and errors:
                err_text = f"No renderable GenUICraft messages. First error: {errors[0][:160]}"
            messages = _fallback_messages(err_text)
        messages_json = _safe_json_dumps(messages)
        html_text = (
            template.replace("__GenUICraft_MESSAGES_JSON__", messages_json)
            .replace("__GenUICraft_RESET_VALUE__", "true")
            .replace("__ASSET_BASE__", asset_base)
        )

        html_path = output_dir / f"{ui_id}.html"
        html_path.write_text(html_text, encoding="utf-8")

        image_path = None
        render_error = None
        if render_images and not renderer_error:
            image_path = output_dir / f"{ui_id}.{image_format}"
            if server:
                html_rel = html_path.resolve().relative_to(server_root.resolve()).as_posix()
                url = f"http://127.0.0.1:{server.port}/{html_rel}"
            else:
                url = html_path.as_uri()
            if parallel_workers > 1:
                pending_parallel.append(
                    {"ui_id": ui_id, "url": url, "image_path": image_path}
                )
                pending_records.append(
                    {
                        "ui_id": ui_id,
                        "response_id": row.get("response_id"),
                        "query_id": row.get("query_id"),
                        "html_path": str(html_path.relative_to(output_dir.parent)),
                        "image_path": str(image_path.relative_to(output_dir.parent)),
                    }
                )
            else:
                render_error = renderer.render(url, image_path)
                if render_error:
                    logger.error("Stage4 render error ui_id=%s: %s", ui_id, render_error)

        if parallel_workers == 1:
            record = {
                "ui_id": ui_id,
                "response_id": row.get("response_id"),
                "query_id": row.get("query_id"),
                "html_path": str(html_path.relative_to(output_dir.parent)),
                "image_path": str(image_path.relative_to(output_dir.parent)) if image_path else None,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "render": {
                    "image_ok": render_error is None and image_path is not None,
                    "error": render_error or renderer_error,
                    "renderer_assets_hash": renderer_assets_hash,
                },
            }
            writer.append(record)
        existing.add(ui_id)
        created += 1

    if render_images and parallel_workers > 1 and pending_parallel:
        logger.info(
            "Stage4 rendering %s images with %s workers",
            len(pending_parallel),
            parallel_workers,
        )
        render_errors = _render_parallel(
            tasks=pending_parallel,
            workers=parallel_workers,
            viewport=viewport,
            timeout_ms=timeout_ms,
            wait_ms=wait_ms,
            emulate_mobile=emulate_mobile,
        )
        for record in pending_records:
            ui_id = record["ui_id"]
            render_error = render_errors.get(ui_id)
            if render_error:
                logger.error("Stage4 render error ui_id=%s: %s", ui_id, render_error)
            writer.append(
                {
                    **record,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "render": {
                        "image_ok": render_error is None and record.get("image_path") is not None,
                        "error": render_error or renderer_error,
                        "renderer_assets_hash": renderer_assets_hash,
                    },
                }
            )

    if render_images and parallel_workers == 1:
        renderer.stop()
    if server:
        server.stop()




