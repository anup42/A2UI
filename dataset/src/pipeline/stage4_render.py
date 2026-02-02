from __future__ import annotations

import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pipeline.storage import JsonlWriter, iter_jsonl, load_existing_ids


def _looks_like_components_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if "id" not in item or "component" not in item:
            return False
    return True


def _wrap_components_as_messages(components: list[dict], surface_id: str = "@default") -> list[dict]:
    root_id = components[0].get("id") if components else "root"
    return [
        {"beginRendering": {"root": root_id, "surfaceId": surface_id}},
        {"surfaceUpdate": {"surfaceId": surface_id, "components": components}},
    ]


def _has_message_content(messages: list[Any]) -> bool:
    keys = {
        "beginRendering",
        "surfaceUpdate",
        "deleteSurface",
        "dataModelUpdate",
        "dataModelDelete",
        "dataModelTransaction",
    }
    for item in messages:
        if isinstance(item, dict) and keys.intersection(item.keys()):
            return True
    return False


def _fallback_messages(text: str) -> list[dict]:
    return _wrap_components_as_messages(
        [
            {
                "id": "fallback-text",
                "component": {"Text": {"text": {"literalString": text}, "usageHint": "body"}},
            }
        ]
    )


def _normalize_messages(value: Any) -> list[Any]:
    if isinstance(value, list):
        if _looks_like_components_list(value):
            return _wrap_components_as_messages(value)
        return value
    if isinstance(value, dict):
        for key in ("messages", "payload"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                if _looks_like_components_list(candidate):
                    return _wrap_components_as_messages(candidate)
                return candidate
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
    a2ui_path: Path,
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
    existing = load_existing_ids(render_log_path, "ui_id")
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
    for row in iter_jsonl(a2ui_path):
        ui_id = row.get("ui_id")
        if not ui_id or ui_id in existing:
            continue
        if max_total is not None and created >= max_total:
            logger.info("Stage4 reached max_total=%s", max_total)
            break

        a2ui_json = row.get("a2ui_json")
        messages = _normalize_messages(a2ui_json)
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

