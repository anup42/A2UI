#!/usr/bin/env python3
"""Render stage-2 responses as mobile-resolution HTML + PNG screenshots.

This script reads ``responses.jsonl`` (stage 2 output), uses an LLM
(via the existing adapter system synced with models.yaml) to **generate**
self-contained HTML/CSS/JS from each ``response_text``, then uses
Playwright to capture a screenshot at mobile resolution (default
412x915, matching run.yaml render.viewport).

Unlike stage 3/4 (GenUICraft IR -> genui_json -> template + app.bundle.js),
this directly prompts the LLM to produce mobile-first HTML. Unlike stage 5
(which uses a hardcoded Python HTML builder at desktop viewport), this uses
the LLM to create richer, layout-aware mobile UI code.

Usage examples
--------------
    # Basic -- render all responses at mobile resolution using gemma
    py scripts/render_responses_mobile.py \
        --responses_path data/runs/my_run/responses.jsonl \
        --model vllm_gemma_4_26b

    # Skip screenshots, only generate HTML
    py scripts/render_responses_mobile.py \
        --responses_path data/runs/my_run/responses.jsonl \
        --model gemini_2_5_flash \
        --no_screenshots

    # Custom viewport and output dir
    py scripts/render_responses_mobile.py \
        --responses_path data/runs/my_run/responses.jsonl \
        --model gemini_2_5_flash \
        --output_dir data/runs/my_run/rendered_mobile \
        --viewport_width 390 --viewport_height 844

    # Limit to first 50 items
    py scripts/render_responses_mobile.py \
        --responses_path data/runs/my_run/responses.jsonl \
        --model gemini_2_5_flash \
        --max_total 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Make pipeline imports work when running from the dataset root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATASET_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _DATASET_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from llm.base import BaseLLMAdapter, LLMRateLimitError
from llm.factory import build_adapter, load_model_specs
from pipeline.cache import PromptCache
from pipeline.storage import JsonlWriter, iter_jsonl, load_jsonl_by_key
from utils.config import load_yaml
from utils.hashing import hash_text, normalize_text
from utils.rate_limit import RateLimiter
from utils.retry import with_retry


# ═══════════════════════════════════════════════════════════════════════════
# Prompt for generating mobile HTML/CSS/JS from response text
# ═══════════════════════════════════════════════════════════════════════════

_MOBILE_HTML_PROMPT = r"""You are a mobile UI code generator. Given a structured text response, produce a single self-contained HTML file with inline CSS and JavaScript that renders a polished mobile app screen.

## Input
- Response text (may contain sections, tables, lists, buttons, icons, images)
- Intent: {intent}
- Tags: {tags}

## Hard Requirements
1. Output ONLY the raw HTML code. No markdown fences, no explanation, no commentary.
2. The HTML MUST be a complete, self-contained document with <!doctype html>, <html>, <head>, <body>.
3. ALL CSS must be inline in a single <style> block in the <head>.
4. ALL JavaScript must be inline in a single <script> block before </body>.
5. Mobile-first design: target viewport width 390px, max 412px.
6. Use <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">.
7. NO external CSS/JS frameworks (no Tailwind CDN, no Bootstrap, no React, etc.).
8. NO external images except those whose URLs appear in the input text.
9. NO placeholder or lorem ipsum content -- use only the actual data from the input.
10. The page must be scrollable vertically; do NOT use overflow:hidden on body.

## Design Guidelines
- Clean, modern mobile app aesthetic with card-based layout.
- Use CSS custom properties for theming (colors, spacing, radii).
- Cards should have subtle shadows, rounded corners (12-16px), and padding (12-16px).
- Section headings: 16-20px, semibold. Body text: 13-15px. Caption/muted: 11-12px.
- Primary buttons: filled, rounded pill shape (border-radius: 999px), brand blue (#1f78f0).
- Borderless/link buttons: text-only, underlined, brand blue.
- Tables: horizontal scroll if needed, compact cells (6-8px padding), sticky header.
- Lists: bullet points with proper spacing.
- Icons: if icon URLs are present, render as small inline images (16-20px).
- Images: if image URLs are present, render in a responsive grid with aspect-ratio: 4/3.
- Quick Actions: render as a horizontal row of pill buttons.
- Sources: render as a compact list of links.
- Use system font stack: "Segoe UI", system-ui, -apple-system, sans-serif.
- Light mode only. Background: #eef1f6. Card background: #ffffff. Text: #0f172a. Muted: #475569.

## JavaScript Guidelines
- Any interactive elements (tabs, expandable sections, toggles) should work.
- Buttons with URLs should navigate on click: window.open(url, '_blank').
- Keep JS minimal and vanilla -- no frameworks.

## Structure
- Hero section at top with the title.
- Content sections as stacked cards.
- Quick Actions at the bottom as sticky or prominent buttons.
- Sources as a compact footer section.

Now generate the HTML for this response:

{response_text}
"""


def _render_prompt(response_text: str, intent: str, tags: list[str]) -> str:
    """Fill the mobile HTML generation prompt with response data."""
    tags_str = ", ".join(str(t) for t in tags) if tags else ""
    return _MOBILE_HTML_PROMPT.format(
        response_text=response_text,
        intent=intent or "",
        tags=tags_str,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Extract generated HTML from LLM output
# ═══════════════════════════════════════════════════════════════════════════

def _extract_html(raw_output: str) -> str:
    """Extract the HTML document from LLM output.

    The LLM may wrap the HTML in markdown code fences. Strip those.
    """
    text = raw_output.strip()

    # If wrapped in ```html ... ``` or ``` ... ```, strip the fences
    fence_pattern = re.compile(
        r"^```(?:html|HTML)?\s*\n(.*?)\n```\s*$", re.DOTALL
    )
    m = fence_pattern.match(text)
    if m:
        text = m.group(1).strip()

    # Must start with <!doctype or <html
    lower = text.lower()
    if "<!doctype" in lower or "<html" in lower:
        return text

    # If the LLM added commentary before/after, try to find the HTML block
    start_markers = ["<!doctype html", "<html"]
    for marker in start_markers:
        idx = lower.find(marker)
        if idx >= 0:
            # Find closing </html>
            end_idx = lower.rfind("</html>")
            if end_idx >= 0:
                return text[idx:end_idx + len("</html>")]
            return text[idx:]

    # Fallback: return as-is (might still render)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Playwright renderer
# ═══════════════════════════════════════════════════════════════════════════

class MobileRenderer:
    """Playwright-based renderer at mobile viewport."""

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
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=max(2000, min(self.timeout_ms, 10000)),
                )
            except Exception:
                pass
            if self.wait_ms > 0:
                page.wait_for_timeout(self.wait_ms)
            page.screenshot(path=str(image_path), full_page=True)
            return None
        except Exception as exc:
            return f"render_error: {exc}"
        finally:
            page.close()


# ═══════════════════════════════════════════════════════════════════════════
# HTTP server (for serving assets alongside HTML)
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Slug helper
# ═══════════════════════════════════════════════════════════════════════════

def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._")
    return value or "item"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render stage-2 responses as mobile-resolution HTML + PNG screenshots. "
            "Uses an LLM to generate HTML/CSS/JS from each response_text."
        ),
    )
    parser.add_argument(
        "--responses_path",
        type=str,
        required=True,
        help="Path to responses.jsonl (stage 2 output).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name from models.yaml (e.g. gemini_2_5_flash, vllm_gemma_4_26b).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for HTML + PNG files. "
             "Default: <responses_dir>/rendered_mobile",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory (parent of responses.jsonl). Used for resolving "
             "relative asset paths and writing render_mobile.jsonl. "
             "Default: parent directory of responses_path.",
    )
    parser.add_argument(
        "--viewport_width",
        type=int,
        default=None,
        help="Viewport width in pixels. Default: from run.yaml render.viewport.width or 412.",
    )
    parser.add_argument(
        "--viewport_height",
        type=int,
        default=None,
        help="Viewport height in pixels. Default: from run.yaml render.viewport.height or 915.",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "jpeg", "webp"],
        help="Screenshot image format (default: png).",
    )
    parser.add_argument(
        "--timeout_ms",
        type=int,
        default=None,
        help="Playwright navigation timeout in ms. Default: from run.yaml or 15000.",
    )
    parser.add_argument(
        "--wait_ms",
        type=int,
        default=None,
        help="Extra wait after load before screenshot (ms). Default: from run.yaml or 200.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Max output tokens for LLM HTML generation (default: 8192).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Temperature for LLM generation (default: 0.4).",
    )
    parser.add_argument(
        "--no_screenshots",
        action="store_true",
        help="Only generate HTML files, skip Playwright screenshots.",
    )
    parser.add_argument(
        "--max_total",
        type=int,
        default=None,
        help="Maximum number of responses to render.",
    )
    parser.add_argument(
        "--rate_limit_qps",
        type=float,
        default=1.0,
        help="Rate limit in queries per second (default: 1.0).",
    )
    parser.add_argument(
        "--no_http_server",
        action="store_true",
        default=False,
        help="Use file:// URIs instead of HTTP server for screenshots.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local model folder path override for provider=local transformers mode.",
    )
    parser.add_argument(
        "--vllm_model_path",
        type=str,
        default=None,
        help="Local vLLM model folder path override.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    responses_path = Path(args.responses_path).resolve()
    if not responses_path.exists():
        raise SystemExit(f"responses_path not found: {responses_path}")

    run_dir = Path(args.run_dir).resolve() if args.run_dir else responses_path.parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "rendered_mobile"

    # ------------------------------------------------------------------
    # Load run.yaml config for viewport / timeout defaults
    # ------------------------------------------------------------------
    dataset_root = _DATASET_ROOT
    run_cfg_path = dataset_root / "configs" / "run.yaml"
    run_cfg: dict[str, Any] = {}
    if run_cfg_path.exists():
        raw = load_yaml(run_cfg_path)
        run_cfg = raw.get("run", raw)

    render_cfg: dict[str, Any] = run_cfg.get("render", {})
    default_viewport = render_cfg.get("viewport", {"width": 412, "height": 915})
    if not isinstance(default_viewport, dict):
        default_viewport = {"width": 412, "height": 915}

    viewport_width = args.viewport_width or default_viewport.get("width", 412)
    viewport_height = args.viewport_height or default_viewport.get("height", 915)
    viewport = {"width": int(viewport_width), "height": int(viewport_height)}

    timeout_ms = args.timeout_ms if args.timeout_ms is not None else int(render_cfg.get("timeout_ms", 15000))
    wait_ms = args.wait_ms if args.wait_ms is not None else int(render_cfg.get("wait_ms", 200))
    image_format = args.image_format
    render_images = not args.no_screenshots
    use_http_server = not args.no_http_server

    # ------------------------------------------------------------------
    # Load models.yaml and build adapter
    # ------------------------------------------------------------------
    models_cfg_path = dataset_root / "configs" / "models.yaml"
    if not models_cfg_path.exists():
        raise SystemExit(f"models.yaml not found: {models_cfg_path}")

    models_cfg = load_yaml(models_cfg_path)
    specs = load_model_specs(models_cfg)
    if not specs:
        raise SystemExit("No models configured in models.yaml")

    model_map = {spec.name: spec for spec in specs}
    if args.model not in model_map:
        available = ", ".join(sorted(model_map.keys()))
        raise SystemExit(f"Unknown model '{args.model}'. Available: {available}")

    spec = model_map[args.model]

    # Configure local model path overrides
    if spec.provider.lower() == "vllm_gemma" and args.vllm_model_path:
        os.environ["VLLM_GEMMA_MODEL_PATH"] = args.vllm_model_path
    if spec.provider.lower() == "vllm_qwen" and args.vllm_model_path:
        os.environ["VLLM_QWEN_MODEL_PATH"] = args.vllm_model_path
    if spec.provider.lower() == "local" and args.local_model_path:
        model_lower = (spec.model or "").lower()
        if "qwen" in model_lower:
            os.environ["QWEN_MODEL_PATH"] = args.local_model_path
        elif "deepseek" in model_lower:
            os.environ["DEEPSEEK_MODEL_PATH"] = args.local_model_path
        else:
            os.environ["LOCAL_MODEL_PATH"] = args.local_model_path

    adapter = build_adapter(spec)
    rate_limiter = RateLimiter(args.rate_limit_qps, float(run_cfg.get("call_sleep_seconds", 0)))
    cache = PromptCache(dataset_root / run_cfg.get("cache_dir", "data/cache"))

    print(f"[render_mobile] Model: {spec.name} ({spec.provider}/{spec.model})")
    print(f"[render_mobile] Viewport: {viewport}")
    print(f"[render_mobile] Output: {output_dir}")

    # ------------------------------------------------------------------
    # Setup output + render log
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    render_log_path = run_dir / "render_mobile.jsonl"
    existing_rows = load_jsonl_by_key(render_log_path, "response_id")
    writer = JsonlWriter(render_log_path)

    # ------------------------------------------------------------------
    # Start HTTP server + Playwright
    # ------------------------------------------------------------------
    server = None
    server_root = dataset_root
    if render_images and use_http_server:
        server = HttpServer(server_root)
        server.start()
        print(f"[render_mobile] HTTP server started on port {server.port}")

    renderer = MobileRenderer(viewport=viewport, timeout_ms=timeout_ms, wait_ms=wait_ms)
    renderer_error = None
    if render_images:
        renderer_error = renderer.start()
        if renderer_error:
            print(f"[render_mobile] WARNING: Playwright unavailable: {renderer_error}")

    # ------------------------------------------------------------------
    # Process responses
    # ------------------------------------------------------------------
    created = 0
    skipped = 0
    errors = 0
    llm_errors = 0

    for row in iter_jsonl(responses_path):
        response_id = str(row.get("response_id") or "").strip()
        query_id = str(row.get("query_id") or "").strip()
        if not response_id or not query_id:
            continue

        # Incremental: skip already-rendered items
        if response_id in existing_rows:
            prev = existing_rows[response_id]
            prev_render = prev.get("render") if isinstance(prev, dict) else None
            prev_error = prev_render.get("error") if isinstance(prev_render, dict) else None
            prev_html = prev.get("html_path")
            prev_png = prev.get("image_path")
            if prev_html and (not render_images or (prev_png and not prev_error)):
                skipped += 1
                continue

        if args.max_total is not None and created >= args.max_total:
            print(f"[render_mobile] Reached max_total={args.max_total}")
            break

        # ----------------------------------------------------------
        # Build prompt and call LLM
        # ----------------------------------------------------------
        response_text = str(row.get("response_text") or "").strip()
        if not response_text:
            continue

        intent = str(row.get("intent") or "").strip()
        tags = row.get("tags") if isinstance(row.get("tags"), list) else []

        prompt = _render_prompt(response_text, intent, tags)
        prompt_hash = hash_text(f"render_mobile:{spec.name}:{prompt}")

        # Check cache
        cached = cache.get(prompt_hash)
        if cached:
            raw_output = cached.text.strip()
        else:
            raw_output = None
            rate_limiter.acquire()
            try:
                result = with_retry(
                    lambda: adapter.generate(
                        prompt=prompt,
                        system=None,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        seed=42,
                        json_mode=False,
                    ),
                    max_attempts=2,
                )
                if result.error:
                    print(f"[render_mobile] LLM error response_id={response_id}: {result.error}")
                    llm_errors += 1
                    continue
                raw_output = result.text.strip()
                cache.set(prompt_hash, result.text, result.raw)
            except Exception as exc:
                print(f"[render_mobile] LLM exception response_id={response_id}: {exc}")
                llm_errors += 1
                continue

        # Extract HTML from LLM output
        html_content = _extract_html(raw_output)

        # Write HTML file
        file_stem = _slug(response_id)
        html_path = output_dir / f"{file_stem}.html"
        html_path.write_text(html_content, encoding="utf-8")

        # Screenshot
        image_path: Optional[Path] = None
        render_error: Optional[str] = None
        if render_images and not renderer_error:
            image_path = output_dir / f"{file_stem}.{image_format}"
            if server:
                try:
                    html_rel = html_path.resolve().relative_to(server_root.resolve()).as_posix()
                    url = f"http://127.0.0.1:{server.port}/{html_rel}"
                except ValueError:
                    url = html_path.as_uri()
            else:
                url = html_path.as_uri()
            render_error = renderer.render(url, image_path)
            if render_error:
                print(f"[render_mobile] Screenshot error response_id={response_id}: {render_error}")
                errors += 1

        # Write render log entry
        record = {
            "response_id": response_id,
            "query_id": query_id,
            "html_path": str(html_path.relative_to(run_dir)),
            "image_path": str(image_path.relative_to(run_dir)) if image_path else None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model": {
                "model_name": spec.name,
                "provider": spec.provider,
                "model_id": spec.model,
            },
            "render": {
                "image_ok": render_error is None and image_path is not None,
                "error": render_error or renderer_error,
                "viewport": viewport,
                "image_format": image_format,
            },
        }
        writer.append(record)
        existing_rows[response_id] = record
        created += 1

        if created % 10 == 0:
            print(f"[render_mobile] Progress: {created} rendered, {skipped} skipped, {llm_errors} LLM errors, {errors} screenshot errors")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if render_images:
        renderer.stop()
    if server:
        server.stop()

    print(
        f"[render_mobile] Done. created={created} skipped={skipped} "
        f"llm_errors={llm_errors} screenshot_errors={errors} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
