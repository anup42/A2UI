from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pipeline.stage4_render import HttpServer
from pipeline.storage import JsonlWriter, iter_jsonl, load_jsonl_by_key

_BUTTON_RE = re.compile(r"(?:Action:\s*)?\[Button:\s*(?P<label>.+?)\]\s*(?P<url>https?://\S+)", re.IGNORECASE)
_ASSET_OR_HTTP = r"(?:https?://\S+|(?:\.\./)?assets/\S+|/assets/\S+)"
_LABEL_URL_RE = re.compile(rf"^(?:[-*•]\s*)?(?P<label>[^:]{{1,140}}):\s*(?P<url>{_ASSET_OR_HTTP})\s*$")
_URL_ONLY_RE = re.compile(rf"^(?:[-*•]\s*)?(?P<url>{_ASSET_OR_HTTP})\s*$")
_BULLET_RE = re.compile(r"^[-*•]\s+")
_SEPARATOR_ROW_RE = re.compile(r"^\s*:?-{3,}:?\s*$")
_OPTION_LINE_RE = re.compile(r"^Option\s+(?P<num>\d+)\s*:\s*(?P<body>.+)$", re.IGNORECASE)


class _Stage5HtmlRenderer:
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
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=max(2000, min(self.timeout_ms, 10000)))
            except Exception:
                # networkidle can timeout on some pages with ongoing requests; continue.
                pass
            if self.wait_ms > 0:
                page.wait_for_timeout(self.wait_ms)
            page.screenshot(path=str(image_path), full_page=True)
            return None
        except Exception as exc:
            return f"render_error: {exc}"
        finally:
            page.close()


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._")
    return value or "item"


def _escape(text: str) -> str:
    return html.escape(text or "", quote=True)


def _to_render_asset_path(path: str) -> str:
    normalized = str(path or "").replace("\\", "/").strip()
    if not normalized:
        return normalized
    if normalized.startswith("../assets/"):
        return normalized
    if normalized.startswith("/assets/"):
        return "../" + normalized.lstrip("/")
    normalized = normalized.lstrip("/").lstrip("./")
    if normalized.startswith("assets/"):
        return "../" + normalized
    return normalized


def _extract_sections(response_text: str) -> tuple[str, list[dict[str, Any]]]:
    lines = response_text.splitlines()
    first_non_empty = ""
    for line in lines:
        if line.strip():
            first_non_empty = line.strip()
            break

    title = first_non_empty or "Response"
    consumed_title = False
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def _start_section(section_title: str | None) -> None:
        nonlocal current
        current = {"title": section_title, "lines": []}
        sections.append(current)

    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            if current is not None and current["lines"] and current["lines"][-1] != "":
                current["lines"].append("")
            continue

        if not consumed_title and stripped == title:
            consumed_title = True
            continue

        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        next_supports_heading = bool(
            next_line
            and (
                _BULLET_RE.match(next_line)
                or _BUTTON_RE.match(next_line)
                or _LABEL_URL_RE.match(next_line)
                or _URL_ONLY_RE.match(next_line)
                or "|" in next_line
            )
        )
        short_heading = (
            len(stripped) <= 64
            and stripped == raw
            and stripped[0].isalpha()
            and "|" not in stripped
            and "http" not in stripped.lower()
            and not stripped.endswith((".", "!", "?", ";"))
            and next_supports_heading
        )

        heading_like = (
            (stripped.endswith(":") and not _BULLET_RE.match(stripped) and "http" not in stripped and "|" not in stripped)
            or stripped.lower() in {"quick actions", "quick action", "sources", "images", "icons", "logos", "airline logos"}
            or short_heading
        )
        if heading_like:
            section_title = stripped[:-1].strip() if stripped.endswith(":") else stripped
            _start_section(section_title)
            continue

        if current is None:
            _start_section(None)
        current["lines"].append(raw)

    if not sections:
        sections = [{"title": None, "lines": [response_text]}]

    return title, sections


def _section_title_key(section: dict[str, Any]) -> str:
    return str(section.get("title") or "").strip().lower()


def _is_sources_section(section: dict[str, Any]) -> bool:
    return _section_title_key(section) in {"source", "sources", "references", "reference"}


def _is_icons_section(section: dict[str, Any]) -> bool:
    return _section_title_key(section) in {"icons", "icon", "logos", "logo", "airline logos"}


def _coerce_line_to_icon_hint(raw: str) -> str:
    line = (raw or "").strip()
    if not line:
        return raw
    match_labeled_url = _LABEL_URL_RE.match(line)
    if match_labeled_url:
        label = match_labeled_url.group("label").strip()
        url = match_labeled_url.group("url").strip()
        label_l = label.lower()
        if "icon" in label_l or "logo" in label_l:
            return f"{label}: {url}"
        return f"Icon {label}: {url}"
    match_url = _URL_ONLY_RE.match(line)
    if match_url:
        return f"Icon: {match_url.group('url').strip()}"
    return raw


def _merge_sources_and_icons(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not sections:
        return sections
    source_idx = next((idx for idx, sec in enumerate(sections) if _is_sources_section(sec)), None)
    icon_indices = [idx for idx, sec in enumerate(sections) if _is_icons_section(sec)]
    if source_idx is None or not icon_indices:
        return sections

    merged: list[dict[str, Any]] = []
    icon_lines: list[str] = []
    for idx in icon_indices:
        lines = sections[idx].get("lines") or []
        for raw in lines:
            if str(raw).strip():
                icon_lines.append(_coerce_line_to_icon_hint(str(raw)))

    for idx, sec in enumerate(sections):
        if idx in icon_indices:
            continue
        if idx == source_idx:
            source_lines = [str(line) for line in (sec.get("lines") or [])]
            if source_lines and source_lines[-1].strip():
                source_lines.append("")
            source_lines.extend(icon_lines)
            merged.append({"title": sec.get("title"), "lines": source_lines})
            continue
        merged.append(sec)
    return merged


def _split_table_row(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return [cell for cell in cells]


def _is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        if not _SEPARATOR_ROW_RE.match(cell):
            return False
    return True


def _infer_media_kind(label: str, url: str, section_key: str) -> Optional[str]:
    label_l = label.lower().strip()
    url_l = url.lower().strip()
    image_ext = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif")
    icon_ext = (".svg", ".ico")

    if any(k in section_key for k in ("icon", "logo")):
        return "icon"
    if any(k in section_key for k in ("image", "photo", "gallery", "media")):
        return "image"
    if any(k in label_l for k in ("icon", "logo")):
        return "icon"
    if any(k in label_l for k in ("image", "photo", "cover", "thumbnail", "box art", "style")):
        return "image"
    if "/icons/" in url_l:
        return "icon"
    if url_l.endswith(icon_ext):
        return "icon"
    if url_l.endswith(image_ext):
        return "image"
    return None


def _parse_lines_to_blocks(lines: list[str], section_key: str = "") -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        match_button = _BUTTON_RE.match(line)
        if match_button:
            blocks.append(
                {
                    "type": "button",
                    "label": match_button.group("label").strip(),
                    "url": match_button.group("url").strip(),
                }
            )
            i += 1
            continue

        match_option = _OPTION_LINE_RE.match(line)
        if match_option:
            body = match_option.group("body").strip()
            parts = [p.strip() for p in body.split("|") if p.strip()]
            title = parts[0] if parts else body
            sub_lines = parts[1:] if len(parts) > 1 else []
            extra_lines: list[str] = []
            action: dict[str, str] | None = None

            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if not s:
                    j += 1
                    if action:
                        break
                    continue
                if _OPTION_LINE_RE.match(s):
                    break
                mb = _BUTTON_RE.match(s)
                if mb:
                    if action is None:
                        action = {"label": mb.group("label").strip(), "url": mb.group("url").strip()}
                        j += 1
                        continue
                    break
                if "|" in s and not _LABEL_URL_RE.match(s):
                    break
                if _LABEL_URL_RE.match(s) or _URL_ONLY_RE.match(s):
                    break
                if _BULLET_RE.match(s):
                    extra_lines.append(_BULLET_RE.sub("", s).strip())
                    j += 1
                    continue
                extra_lines.append(s)
                j += 1

            blocks.append(
                {
                    "type": "option_card",
                    "index": match_option.group("num"),
                    "title": title,
                    "sub_lines": sub_lines,
                    "extra_lines": extra_lines,
                    "action": action,
                }
            )
            i = j
            continue

        if "|" in line:
            table_lines: list[str] = []
            j = i
            while j < len(lines):
                s = lines[j].strip()
                if not s or "|" not in s:
                    break
                table_lines.append(s)
                j += 1
            if len(table_lines) >= 2:
                rows = [_split_table_row(row) for row in table_lines]
                rows = [row for row in rows if row]
                if rows:
                    header = rows[0]
                    data_start = 1
                    if len(rows) > 1 and _is_separator_row(rows[1]):
                        data_start = 2
                    data_rows = rows[data_start:]
                    blocks.append({"type": "table", "header": header, "rows": data_rows})
                    i = j
                    continue

        match_labeled_url = _LABEL_URL_RE.match(line)
        if match_labeled_url:
            label = match_labeled_url.group("label").strip()
            url = match_labeled_url.group("url").strip()
            media_kind = _infer_media_kind(label, url, section_key)
            if media_kind:
                blocks.append({"type": media_kind, "label": label, "url": url})
            else:
                blocks.append({"type": "link", "label": label, "url": url})
            i += 1
            continue

        match_url = _URL_ONLY_RE.match(line)
        if match_url:
            url = match_url.group("url").strip()
            media_kind = _infer_media_kind(url, url, section_key)
            if media_kind:
                blocks.append({"type": media_kind, "label": url, "url": url})
            else:
                blocks.append({"type": "link", "label": url, "url": url})
            i += 1
            continue

        if _BULLET_RE.match(line):
            items: list[str] = []
            j = i
            while j < len(lines):
                s = lines[j].strip()
                if not s:
                    break
                if not _BULLET_RE.match(s):
                    break
                item = _BULLET_RE.sub("", s).strip()
                lm = _LABEL_URL_RE.match(item)
                um = _URL_ONLY_RE.match(item)
                if lm or um:
                    break
                items.append(item)
                j += 1
            if items:
                blocks.append({"type": "list", "items": items})
                i = j
                continue

        paragraph_lines = [line]
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                break
            if (
                _BUTTON_RE.match(s)
                or _BULLET_RE.match(s)
                or _LABEL_URL_RE.match(s)
                or _URL_ONLY_RE.match(s)
                or "|" in s
            ):
                break
            paragraph_lines.append(s)
            j += 1
        blocks.append({"type": "paragraph", "text": "\n".join(paragraph_lines)})
        i = j

    return blocks

def _normalize_assets(row: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    assets = row.get("assets")
    if not isinstance(assets, list):
        return mapping
    for item in assets:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        path = str(item.get("path") or "").strip()
        if not url or not path:
            continue
        mapping[url] = _to_render_asset_path(path)
    return mapping


def _rewrite_url(url: str, mapping: dict[str, str]) -> str:
    if url in mapping:
        return mapping[url]
    return _to_render_asset_path(url)


def _render_block(block: dict[str, Any], url_map: dict[str, str]) -> str:
    btype = block.get("type")
    if btype == "paragraph":
        text = _escape(str(block.get("text") or "")).replace("\n", "<br />")
        return f'<p class="s5-paragraph">{text}</p>'
    if btype == "list":
        items = "".join(f"<li>{_escape(str(item))}</li>" for item in (block.get("items") or []))
        return f'<ul class="s5-list">{items}</ul>'
    if btype == "table":
        header = block.get("header") or []
        rows = block.get("rows") or []
        th = "".join(f"<th>{_escape(str(cell))}</th>" for cell in header)
        body_rows = []
        for row in rows:
            td = "".join(f"<td>{_escape(str(cell))}</td>" for cell in row)
            body_rows.append(f"<tr>{td}</tr>")
        body = "".join(body_rows)
        return f'<div class="s5-table-wrap"><table class="s5-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'
    if btype == "button":
        label = _escape(str(block.get("label") or "Open"))
        url = _escape(_rewrite_url(str(block.get("url") or ""), url_map))
        return f'<a class="s5-btn" href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
    if btype == "link":
        label = _escape(str(block.get("label") or "Link"))
        url = _escape(_rewrite_url(str(block.get("url") or ""), url_map))
        return f'<a class="s5-link" href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
    if btype == "image":
        label = _escape(str(block.get("label") or "Image"))
        url = _escape(_rewrite_url(str(block.get("url") or ""), url_map))
        return (
            '<figure class="s5-media-card">'
            f'<img class="s5-image" src="{url}" alt="{label}" loading="lazy" />'
            f"<figcaption>{label}</figcaption>"
            "</figure>"
        )
    if btype == "icon":
        label = _escape(str(block.get("label") or "Icon"))
        url = _escape(_rewrite_url(str(block.get("url") or ""), url_map))
        return (
            '<span class="s5-icon-chip">'
            f'<img class="s5-icon" src="{url}" alt="{label}" loading="lazy" />'
            f"<span>{label}</span>"
            "</span>"
        )
    if btype == "option_card":
        index = _escape(str(block.get("index") or "").strip())
        title = _escape(str(block.get("title") or "Option"))
        sub_lines = [str(x).strip() for x in (block.get("sub_lines") or []) if str(x).strip()]
        extra_lines = [str(x).strip() for x in (block.get("extra_lines") or []) if str(x).strip()]
        action = block.get("action") if isinstance(block.get("action"), dict) else None

        description_lines: list[str] = []
        kv_items: list[tuple[str, str]] = []
        for entry in sub_lines + extra_lines:
            if ":" in entry:
                key, value = entry.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key and value and len(key) <= 32:
                    kv_items.append((key, value))
                    continue
            description_lines.append(entry)

        description_html = ""
        if description_lines:
            paragraph = "<br />".join(_escape(line) for line in description_lines)
            description_html = f'<p class="s5-option-desc">{paragraph}</p>'

        kv_html = ""
        if kv_items:
            kv_rows = "".join(
                f'<div class="s5-option-kv-row"><dt>{_escape(k)}</dt><dd>{_escape(v)}</dd></div>'
                for k, v in kv_items
            )
            kv_html = f'<dl class="s5-option-kv">{kv_rows}</dl>'

        action_html = ""
        if action:
            action_html = _render_block({"type": "button", "label": action.get("label"), "url": action.get("url")}, url_map)
            action_html = f'<div class="s5-option-actions">{action_html}</div>'

        prefix = f'<span class="s5-option-index">Option {index}</span>' if index else ""
        heading = (
            '<h3 class="s5-option-title">'
            f"{prefix}<span>{title}</span>"
            "</h3>"
        )
        return f'<article class="s5-option-card">{heading}{description_html}{kv_html}{action_html}</article>'
    return ""


def _extract_media_blocks(lines: list[str], kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        match = _LABEL_URL_RE.match(line)
        if match:
            out.append(
                {
                    "type": kind,
                    "label": match.group("label").strip(),
                    "url": match.group("url").strip(),
                }
            )
            continue
        match_url = _URL_ONLY_RE.match(line)
        if match_url:
            url = match_url.group("url").strip()
            out.append({"type": kind, "label": url, "url": url})
    return out


def _render_section(section: dict[str, Any], url_map: dict[str, str]) -> str:
    title = section.get("title")
    lines = section.get("lines") or []
    title_key = str(title or "").strip().lower()

    if title_key in {"images", "image", "photos", "gallery"}:
        blocks = _extract_media_blocks(lines, "image")
        if not blocks:
            return ""
        media = "".join(_render_block(block, url_map) for block in blocks)
        return (
            '<section class="s5-card s5-section">'
            "<h2>Images</h2>"
            f'<div class="s5-media-grid">{media}</div>'
            "</section>"
        )

    if title_key in {"icons", "icon", "airline logos", "logos", "logo"}:
        blocks = _extract_media_blocks(lines, "icon")
        if not blocks:
            return ""
        media = "".join(_render_block(block, url_map) for block in blocks)
        return (
            '<section class="s5-card s5-section">'
            f"<h2>{_escape(str(title) if title else 'Icons')}</h2>"
            f'<div class="s5-icon-row">{media}</div>'
            "</section>"
        )

    blocks = _parse_lines_to_blocks(lines, section_key=title_key)
    if not blocks:
        return ""

    buttons = [b for b in blocks if b.get("type") == "button"]
    media_images = [b for b in blocks if b.get("type") == "image"]
    media_icons = [b for b in blocks if b.get("type") == "icon"]
    non_buttons = [b for b in blocks if b.get("type") not in {"button", "image", "icon"}]
    raw_links = [b for b in non_buttons if b.get("type") == "link"]
    option_cards = [b for b in non_buttons if b.get("type") == "option_card"]
    non_links = [b for b in non_buttons if b.get("type") not in {"link", "option_card"}]

    # If links resolve to local downloaded assets, render as media instead of plain links.
    links: list[dict[str, Any]] = []
    for link in raw_links:
        url_value = str(link.get("url") or "")
        label_value = str(link.get("label") or url_value)
        rewritten = _rewrite_url(url_value, url_map).lower()
        if rewritten.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif")):
            media_images.append({"type": "image", "label": label_value, "url": url_value})
            continue
        if rewritten.endswith((".svg", ".ico")):
            media_icons.append({"type": "icon", "label": label_value, "url": url_value})
            continue
        links.append(link)

    body = "".join(_render_block(block, url_map) for block in non_links)
    option_html = ""
    if option_cards:
        cards = "".join(_render_block(item, url_map) for item in option_cards)
        option_html = f'<div class="s5-option-grid">{cards}</div>'
    link_html = ""
    if links:
        items = "".join(f"<li>{_render_block(item, url_map)}</li>" for item in links)
        link_html = f'<ul class="s5-link-list">{items}</ul>'

    media_html = ""
    if media_images:
        media_items = "".join(_render_block(item, url_map) for item in media_images)
        media_html += f'<div class="s5-media-grid">{media_items}</div>'
    if media_icons:
        icon_items = "".join(_render_block(item, url_map) for item in media_icons)
        media_html += f'<div class="s5-icon-row">{icon_items}</div>'

    btn_html = ""
    if buttons:
        btn_html = '<div class="s5-button-row">' + "".join(_render_block(btn, url_map) for btn in buttons) + "</div>"

    display_title = str(title) if title else ""
    if title_key in {"source", "sources"} and media_icons:
        display_title = "Sources & Icons"

    heading_html = f"<h2>{_escape(display_title)}</h2>" if display_title else ""
    if title_key in {"source", "sources", "reference", "references"}:
        section_body = f"{body}{option_html}{link_html}{media_html}{btn_html}"
    else:
        section_body = f"{body}{option_html}{media_html}{link_html}{btn_html}"
    return f'<section class="s5-card s5-section">{heading_html}{section_body}</section>'

def _build_html_from_response(row: dict[str, Any]) -> str:
    response_text = str(row.get("response_text") or "").strip()
    title, sections = _extract_sections(response_text)
    sections = _merge_sources_and_icons(sections)
    url_map = _normalize_assets(row)

    rendered_sections = "".join(_render_section(section, url_map) for section in sections)
    if not rendered_sections:
        rendered_sections = '<section class="s5-card s5-section"><p class="s5-paragraph"></p></section>'

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{_escape(title)}</title>
    <style>
      :root {{
        --bg: #eef4fb;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #475569;
        --border: #dbe7f5;
        --accent: #1f78f0;
        --accent-2: #1463cf;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
        color: var(--text);
        background:
          radial-gradient(1200px 600px at -10% -20%, #d9e8fb 0%, transparent 55%),
          radial-gradient(1000px 500px at 120% 10%, #e4f0ff 0%, transparent 50%),
          var(--bg);
      }}
      .s5-page {{
        max-width: 920px;
        margin: 20px auto 28px;
        padding: 0 14px;
      }}
      .s5-hero {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px 18px 14px;
        box-shadow: 0 8px 26px rgba(15, 23, 42, 0.08);
      }}
      .s5-hero h1 {{
        margin: 0;
        font-size: clamp(22px, 2.6vw, 32px);
        line-height: 1.2;
      }}
      .s5-stack {{
        display: grid;
        gap: 12px;
        margin-top: 12px;
      }}
      .s5-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
      }}
      .s5-section {{
        padding: 14px 15px;
      }}
      .s5-section h2 {{
        margin: 0 0 10px;
        font-size: clamp(16px, 2vw, 21px);
      }}
      .s5-paragraph {{
        margin: 0 0 10px;
        color: var(--muted);
        line-height: 1.55;
      }}
      .s5-list {{
        margin: 0;
        padding-left: 18px;
        color: var(--muted);
        line-height: 1.5;
      }}
      .s5-list li {{ margin: 6px 0; }}
      .s5-table-wrap {{
        width: 100%;
        overflow-x: auto;
        border: 1px solid #e5edf8;
        border-radius: 12px;
      }}
      .s5-table {{
        border-collapse: collapse;
        width: 100%;
        min-width: 520px;
        font-size: 13px;
      }}
      .s5-table th, .s5-table td {{
        border-bottom: 1px solid #edf3fb;
        padding: 9px 10px;
        text-align: left;
        vertical-align: top;
      }}
      .s5-table th {{
        background: #f3f8ff;
        font-weight: 700;
      }}
      .s5-table tr:nth-child(even) td {{
        background: #fcfeff;
      }}
      .s5-button-row {{
        margin-top: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .s5-btn {{
        text-decoration: none;
        color: #fff;
        background: linear-gradient(180deg, var(--accent) 0%, var(--accent-2) 100%);
        border: 1px solid rgba(12, 80, 170, 0.3);
        border-radius: 999px;
        padding: 7px 13px;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.1;
        box-shadow: 0 4px 10px rgba(20, 99, 207, 0.25);
      }}
      .s5-btn:hover {{
        filter: brightness(1.05);
      }}
      .s5-link-list {{
        margin: 0;
        padding-left: 18px;
      }}
      .s5-link-list li {{ margin: 6px 0; }}
      .s5-link {{
        color: #1d4ed8;
        text-decoration: underline;
        text-underline-offset: 2px;
      }}
      .s5-media-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 10px;
        margin-top: 8px;
      }}
      .s5-media-card {{
        margin: 0;
        border: 1px solid #dce8f8;
        border-radius: 12px;
        overflow: hidden;
        background: #f8fbff;
      }}
      .s5-image {{
        display: block;
        width: 100%;
        aspect-ratio: 4 / 3;
        object-fit: cover;
        background: #e7eef8;
      }}
      .s5-media-card figcaption {{
        font-size: 12px;
        color: var(--muted);
        padding: 8px 10px;
      }}
      .s5-icon-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
      }}
      .s5-icon-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #d6e4f6;
        border-radius: 999px;
        padding: 5px 9px;
        background: #f7fbff;
        font-size: 12px;
        color: var(--muted);
      }}
      .s5-icon {{
        width: 16px;
        height: 16px;
        object-fit: contain;
      }}
      .s5-option-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 10px;
        margin-top: 8px;
      }}
      .s5-option-card {{
        border: 1px solid #d9e5f5;
        border-radius: 14px;
        padding: 10px;
        background: linear-gradient(180deg, #ffffff 0%, #f9fcff 100%);
      }}
      .s5-option-title {{
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
      }}
      .s5-option-index {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 18px;
        height: 18px;
        border-radius: 999px;
        background: #e8f1ff;
        color: #1a4da7;
        font-size: 10px;
        font-weight: 700;
        padding: 0 6px;
      }}
      .s5-option-desc {{
        margin: 8px 0 0;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
      }}
      .s5-option-kv {{
        margin: 8px 0 0;
      }}
      .s5-option-kv-row {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
        font-size: 12px;
        border-top: 1px dashed #e5edf9;
        padding-top: 6px;
        margin-top: 6px;
      }}
      .s5-option-kv-row dt {{
        margin: 0;
        color: #51627c;
      }}
      .s5-option-kv-row dd {{
        margin: 0;
        font-weight: 700;
        color: #0f2648;
        text-align: right;
      }}
      .s5-option-actions {{
        margin-top: 9px;
      }}
      @media (max-width: 640px) {{
        .s5-page {{
          margin: 12px auto 20px;
          padding: 0 10px;
        }}
        .s5-hero {{
          border-radius: 14px;
          padding: 14px 14px 11px;
        }}
        .s5-section {{
          padding: 12px;
        }}
        .s5-btn {{
          padding: 7px 12px;
        }}
        .s5-option-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="s5-page">
      <header class="s5-hero">
        <h1>{_escape(title)}</h1>
      </header>
      <div class="s5-stack">
        {rendered_sections}
      </div>
    </main>
  </body>
</html>
"""


def run_stage5(
    responses_path: Path,
    output_dir: Path,
    run_dir: Path,
    server_root: Optional[Path],
    logger,
    render_images: bool = True,
    image_format: str = "png",
    viewport: Optional[dict[str, int]] = None,
    timeout_ms: int = 15000,
    wait_ms: int = 200,
    use_http_server: bool = True,
    emulate_mobile: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_log_path = run_dir / "stage5_render.jsonl"
    existing_rows = load_jsonl_by_key(render_log_path, "response_id")
    writer = JsonlWriter(render_log_path)

    server = None
    server_root = server_root or run_dir.parents[1]
    if render_images and use_http_server:
        server = HttpServer(server_root)
        server.start()
        logger.info("Stage5 HTTP server started on port %s", server.port)

    viewport = viewport or {"width": 1280, "height": 720}
    renderer = _Stage5HtmlRenderer(
        viewport=viewport,
        timeout_ms=timeout_ms,
        wait_ms=wait_ms,
        emulate_mobile=emulate_mobile,
    )
    renderer_error = None
    if render_images:
        renderer_error = renderer.start()
        if renderer_error:
            logger.error("Stage5 renderer unavailable: %s", renderer_error)

    created = 0
    for row in iter_jsonl(responses_path):
        response_id = str(row.get("response_id") or "").strip()
        query_id = str(row.get("query_id") or "").strip()
        if not response_id or not query_id:
            continue
        if response_id in existing_rows:
            prev = existing_rows[response_id]
            prev_render = prev.get("render") if isinstance(prev, dict) else None
            prev_error = prev_render.get("error") if isinstance(prev_render, dict) else None
            prev_html = prev.get("html_path")
            prev_png = prev.get("image_path")
            if prev_html and (not render_images or (prev_png and not prev_error)):
                continue

        html_body = _build_html_from_response(row)
        file_stem = _slug(response_id)
        html_path = output_dir / f"{file_stem}.html"
        html_path.write_text(html_body, encoding="utf-8")

        image_path: Path | None = None
        render_error: Optional[str] = None
        if render_images and not renderer_error:
            image_path = output_dir / f"{file_stem}.{image_format}"
            if server:
                html_rel = html_path.resolve().relative_to(server_root.resolve()).as_posix()
                url = f"http://127.0.0.1:{server.port}/{html_rel}"
            else:
                url = html_path.as_uri()
            render_error = renderer.render(url, image_path)
            if render_error:
                logger.error("Stage5 render error response_id=%s: %s", response_id, render_error)

        writer.append(
            {
                "response_id": response_id,
                "query_id": query_id,
                "html_path": str(html_path.relative_to(run_dir)),
                "image_path": str(image_path.relative_to(run_dir)) if image_path else None,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "render": {
                    "image_ok": render_error is None and image_path is not None,
                    "error": render_error or renderer_error,
                },
            }
        )
        created += 1

    if render_images:
        renderer.stop()
    if server:
        server.stop()
    logger.info("Stage5 completed created=%s output_dir=%s", created, output_dir)

