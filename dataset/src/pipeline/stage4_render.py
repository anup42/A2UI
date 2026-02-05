from __future__ import annotations

import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
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


def _convert_component_v09_to_v08(component: dict[str, Any]) -> dict[str, Any]:
    comp_id = component.get("id") or "component"
    comp_type = component.get("component")
    if comp_type == "Text":
        return {
            "id": comp_id,
            "component": {
                "Text": {
                    "text": _dynamic_string(component.get("text")),
                    "usageHint": component.get("variant", "body"),
                }
            },
        }
    if comp_type == "Image":
        payload: dict[str, Any] = {"url": _dynamic_string(component.get("url"))}
        usage = component.get("variant")
        if usage:
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


def _wrap_components_as_messages(components: list[dict], surface_id: str = "@default") -> list[dict]:
    converted = [_convert_component_v09_to_v08(comp) for comp in components]
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
        if "updateComponents" in item:
            surface_id = item.get("updateComponents", {}).get("surfaceId")
            comps = item.get("updateComponents", {}).get("components") or []
            if surface_id:
                if surface_id not in surfaces:
                    surfaces[surface_id] = {}
                    order.append(surface_id)
                surfaces[surface_id]["components"] = comps

    if not surfaces:
        return messages

    output: list[dict[str, Any]] = []
    for surface_id in order:
        components = surfaces.get(surface_id, {}).get("components") or []
        components = _expand_text_components_v09(components)
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


def _safe_json_dumps(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text.replace("</", "<\\/")


def _ensure_trailing_slash(uri: str) -> str:
    return uri if uri.endswith("/") else uri + "/"


class HtmlRenderer:
    def __init__(self, viewport: dict[str, int], timeout_ms: int, wait_ms: int) -> None:
        self.viewport = viewport
        self.timeout_ms = timeout_ms
        self.wait_ms = wait_ms
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
        page = self._browser.new_page(viewport=self.viewport)
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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = assets_dir / "template.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")

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

    renderer = HtmlRenderer(viewport=viewport, timeout_ms=timeout_ms, wait_ms=wait_ms)
    renderer_error = None
    if render_images:
        renderer_error = renderer.start()
        if renderer_error:
            logger.error("Stage4 renderer unavailable: %s", renderer_error)

    created = 0
    for row in iter_jsonl(genui_path):
        ui_id = row.get("ui_id")
        if not ui_id:
            continue
        if ui_id in existing:
            prev = existing_rows.get(ui_id) or {}
            prev_render = prev.get("render") if isinstance(prev, dict) else None
            prev_error = prev_render.get("error") if isinstance(prev_render, dict) else None
            prev_image = prev.get("image_path") if isinstance(prev, dict) else None
            if not (render_images and (prev_error or not prev_image)):
                continue
        if max_total is not None and created >= max_total:
            logger.info("Stage4 reached max_total=%s", max_total)
            break

        genui_json = row.get("genui_json")
        if genui_json is None:
            genui_json = row.get("a2ui_json")
        messages = _normalize_messages(genui_json)
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
            render_error = renderer.render(url, image_path)
            if render_error:
                logger.error("Stage4 render error ui_id=%s: %s", ui_id, render_error)

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
            },
        }
        writer.append(record)
        existing.add(ui_id)
        created += 1

    if render_images:
        renderer.stop()
    if server:
        server.stop()
