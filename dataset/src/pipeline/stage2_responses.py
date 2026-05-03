from __future__ import annotations

from datetime import datetime
import json
import time
import hashlib
import mimetypes
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from pipeline.common import extract_json, load_prompt, render_prompt
from pipeline.storage import JsonlWriter, iter_jsonl
from pipeline.cache import PromptCache
from llm.base import BaseLLMAdapter, LLMRateLimitError
from utils.hashing import normalize_text, hash_text
from utils.rate_limit import RateLimiter
from utils.retry import with_retry


def _make_response_id(query_id: str, n_idx: int) -> str:
    suffix = query_id.replace("q_", "")
    return f"r_{suffix}_{n_idx:02d}"



_TRANSIENT_ERROR_MARKERS = (
    "getaddrinfo failed",
    "temporary failure in name resolution",
    "name or service not known",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "remote end closed connection",
    "http 5",
)


def _is_transient_error(message: str) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _TRANSIENT_ERROR_MARKERS)


def _sleep_backoff(attempt: int) -> None:
    delay = min(30.0, 2.0 ** (attempt - 1))
    time.sleep(delay)


_URL_RE = re.compile(r"https?://[^\s<>\"')]+")
_ASSET_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    ".tif",
    ".tiff",
    ".pdf",
    ".zip",
}
_ASSET_HOST_HINTS = (
    "icons8.com",
    "unsplash.com",
    "images.unsplash.com",
    "imgur.com",
    "cloudfront.net",
    "googleusercontent.com",
    "loremflickr.com",
)
_RANDOM_PLACEHOLDER_ASSET_HOSTS = (
    "loremflickr.com",
    "picsum.photos",
    "placekitten.com",
    "placehold.co",
    "placeholder.com",
    "dummyimage.com",
)

_VISUAL_INTENT_HINTS = {
    "travel",
    "booking",
    "recipe",
    "product_lookup",
    "entertainment",
    "event_schedule",
    "weather",
    "localization",
    "qr_scanner",
    "status_check",
}


_MIME_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "application/pdf": ".pdf",
    "application/zip": ".zip",
}


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _offline_mode_enabled() -> bool:
    return _is_truthy(os.environ.get("DATASET_OFFLINE_MODE"))


def _real_asset_retry_enabled() -> bool:
    return _is_truthy(os.environ.get("STAGE2_REAL_ASSET_RETRY_ENABLED"))


def _real_asset_retry_max_attempts() -> int:
    raw = os.environ.get("STAGE2_REAL_ASSET_RETRY_MAX_ATTEMPTS", "3").strip()
    try:
        value = int(raw)
    except Exception:
        value = 3
    return max(1, value)


def _real_asset_retry_min_valid_rate() -> float:
    raw = os.environ.get("STAGE2_REAL_ASSET_RETRY_MIN_VALID_RATE", "1.0").strip()
    try:
        value = float(raw)
    except Exception:
        value = 1.0
    return max(0.0, min(1.0, value))


def _use_local_icon_catalog_enabled() -> bool:
    return _is_truthy(os.environ.get("STAGE2_USE_LOCAL_ICON_CATALOG"))


def _icons_only_mode_enabled() -> bool:
    return _is_truthy(os.environ.get("STAGE2_ICONS_ONLY_MODE"))


def _standalone_icon_section_enabled() -> bool:
    return _is_truthy(os.environ.get("STAGE2_APPEND_STANDALONE_ICON_SECTION"))


def _detect_dataset_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "run.yaml").exists() and (candidate / "src").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def _normalize_section_header(value: str) -> str:
    return value.strip().lower().rstrip(":")


def _looks_like_known_heading(value: str) -> bool:
    header = _normalize_section_header(value)
    if header in {
        "summary",
        "assumptions",
        "structured details",
        "quick actions",
        "quick action",
        "sources",
        "images",
        "image",
        "icons",
        "icon",
    }:
        return True
    return bool(re.match(r"^option\s+\d+\b", header))


def _strip_images_section(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        lower = _normalize_section_header(stripped)
        if lower in {"images", "image"}:
            i += 1
            while i < len(lines):
                look = lines[i].strip()
                if not look:
                    i += 1
                    break
                if _looks_like_known_heading(look):
                    break
                i += 1
            continue
        cleaned.append(line)
        i += 1
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)


def _extract_icon_labels(text: str) -> list[str]:
    labels: list[str] = []
    mode = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if mode:
                mode = False
            continue
        header = _normalize_section_header(stripped)
        if header in {"icons", "icon"}:
            mode = True
            continue
        if mode and _looks_like_known_heading(stripped):
            mode = False
        if not mode:
            continue
        match = re.match(r"^[-*]?\s*([^:]+):\s*\S+", stripped)
        if match:
            label = match.group(1).strip()
            if label:
                labels.append(label)
    return labels


def _default_icon_label(icon_id: str) -> str:
    return icon_id.replace("-", " ").replace("_", " ").strip().title() or "Icon"


def _score_icon_for_label(label: str, keyword_map: dict[str, str]) -> str | None:
    lowered = label.lower()
    for keyword, icon_id in keyword_map.items():
        if keyword in lowered:
            return icon_id
    return None


def _load_local_icon_context(dataset_root: Path, logger) -> dict[str, object] | None:
    if not _use_local_icon_catalog_enabled():
        return None

    catalog_rel = os.environ.get(
        "STAGE2_ICON_CATALOG_PATH",
        "assets/icon_catalog/bootstrap-icons/catalog.json",
    )
    mapping_rel = os.environ.get("STAGE2_ICON_MAP_PATH", "configs/icon_map.json")
    catalog_path = (dataset_root / catalog_rel).resolve()
    mapping_path = (dataset_root / mapping_rel).resolve()
    if not catalog_path.exists():
        logger.warning("Stage2 icon catalog enabled but missing catalog: %s", catalog_path)
        return None
    if not mapping_path.exists():
        logger.warning("Stage2 icon catalog enabled but missing mapping: %s", mapping_path)
        return None

    try:
        catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Stage2 icon catalog load failed: %s", exc)
        return None

    icons = catalog_payload.get("icons")
    if not isinstance(icons, list):
        logger.warning("Stage2 icon catalog malformed: missing icons list")
        return None

    icons_by_id: dict[str, dict[str, str]] = {}
    url_to_local_file: dict[str, Path] = {}
    for item in icons:
        if not isinstance(item, dict):
            continue
        icon_id = str(item.get("id") or "").strip()
        source_url = str(item.get("source_url") or "").strip()
        local_path = str(item.get("local_path") or "").strip()
        if not icon_id or not source_url or not local_path:
            continue
        full_local_path = (dataset_root / local_path).resolve()
        if not full_local_path.exists():
            continue
        icons_by_id[icon_id] = {
            "id": icon_id,
            "source_url": source_url,
            "local_path": local_path,
        }
        url_to_local_file[source_url] = full_local_path

    keyword_map_raw = mapping_payload.get("keyword_map")
    intent_defaults_raw = mapping_payload.get("intent_defaults")
    fallback_icons_raw = mapping_payload.get("fallback_icons")
    min_icons = int(mapping_payload.get("min_icons_per_response", 2) or 2)
    max_icons = int(mapping_payload.get("max_icons_per_response", 3) or 3)

    keyword_map: dict[str, str] = {}
    if isinstance(keyword_map_raw, dict):
        for key, value in keyword_map_raw.items():
            key_s = str(key).strip().lower()
            value_s = str(value).strip()
            if key_s and value_s:
                keyword_map[key_s] = value_s

    intent_defaults: dict[str, list[str]] = {}
    if isinstance(intent_defaults_raw, dict):
        for key, value in intent_defaults_raw.items():
            key_s = _normalize_intent_key(str(key))
            if not key_s:
                continue
            if isinstance(value, list):
                intent_defaults[key_s] = [str(v).strip() for v in value if str(v).strip()]

    fallback_icons: list[str] = []
    if isinstance(fallback_icons_raw, list):
        fallback_icons = [str(v).strip() for v in fallback_icons_raw if str(v).strip()]

    if not icons_by_id:
        logger.warning("Stage2 icon catalog enabled but no usable icons were loaded")
        return None

    logger.info(
        "Stage2 icon catalog enabled icons=%s icons_only_mode=%s",
        len(icons_by_id),
        _icons_only_mode_enabled(),
    )

    return {
        "icons_only_mode": _icons_only_mode_enabled(),
        "icons_by_id": icons_by_id,
        "url_to_local_file": url_to_local_file,
        "keyword_map": keyword_map,
        "intent_defaults": intent_defaults,
        "fallback_icons": fallback_icons,
        "min_icons": max(1, min_icons),
        "max_icons": max(1, max_icons),
    }


def _build_icon_rows(
    response_text: str,
    query_text: str,
    intent: str | None,
    tags: list[str] | None,
    icon_context: dict[str, object],
) -> list[tuple[str, str]]:
    icons_by_id = icon_context.get("icons_by_id") if isinstance(icon_context, dict) else None
    keyword_map = icon_context.get("keyword_map") if isinstance(icon_context, dict) else None
    intent_defaults = icon_context.get("intent_defaults") if isinstance(icon_context, dict) else None
    fallback_icons = icon_context.get("fallback_icons") if isinstance(icon_context, dict) else None
    min_icons = int(icon_context.get("min_icons", 2) or 2)
    max_icons = int(icon_context.get("max_icons", 3) or 3)

    if not isinstance(icons_by_id, dict) or not icons_by_id:
        return []
    if not isinstance(keyword_map, dict):
        keyword_map = {}
    if not isinstance(intent_defaults, dict):
        intent_defaults = {}
    if not isinstance(fallback_icons, list):
        fallback_icons = []

    existing_labels = _extract_icon_labels(response_text)
    selected_ids: list[str] = []
    seen: set[str] = set()

    def _add(icon_id: str) -> None:
        if not icon_id or icon_id in seen or icon_id not in icons_by_id:
            return
        seen.add(icon_id)
        selected_ids.append(icon_id)

    for label in existing_labels:
        mapped = _score_icon_for_label(label, keyword_map)
        if mapped:
            _add(mapped)

    bag = " ".join(
        [
            str(query_text or "").lower(),
            str(response_text or "").lower(),
            str(intent or "").lower(),
            " ".join(str(tag).lower() for tag in (tags or [])),
        ]
    )
    for keyword, icon_id in keyword_map.items():
        if keyword and keyword in bag:
            _add(icon_id)
            if len(selected_ids) >= max_icons:
                break

    intent_key = _normalize_intent_key(intent)
    if intent_key and intent_key in intent_defaults:
        for icon_id in intent_defaults[intent_key]:
            _add(str(icon_id))
            if len(selected_ids) >= max_icons:
                break

    for icon_id in fallback_icons:
        _add(str(icon_id))
        if len(selected_ids) >= max(min_icons, max_icons):
            break

    target_count = max(min_icons, min(max_icons, len(existing_labels) or min_icons))
    selected_ids = selected_ids[:target_count]

    rows: list[tuple[str, str]] = []
    for idx, icon_id in enumerate(selected_ids):
        icon_meta = icons_by_id.get(icon_id)
        if not isinstance(icon_meta, dict):
            continue
        url = str(icon_meta.get("source_url") or "").strip()
        if not url:
            continue
        if idx < len(existing_labels):
            label = existing_labels[idx]
        else:
            label = _default_icon_label(icon_id)
        rows.append((label, url))
    return rows


def _remove_icons_section(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        lower = _normalize_section_header(stripped)
        if lower in {"icons", "icon"}:
            i += 1
            while i < len(lines):
                look = lines[i].strip()
                if not look:
                    i += 1
                    break
                if _looks_like_known_heading(look):
                    break
                i += 1
            continue
        cleaned.append(line)
        i += 1
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)


def _strip_unresolved_media_images(text: str, assets: list[dict]) -> str:
    if not text:
        return text
    valid_urls = {
        str(item.get("url") or "").strip()
        for item in assets
        if isinstance(item, dict) and str(item.get("url") or "").strip()
    }
    if not valid_urls:
        valid_urls = set()

    media_image_re = re.compile(r"\s*Image=(https?://[^\s]+)")
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip().lower().startswith("media:"):
            cleaned.append(line)
            continue

        def _replace(match: re.Match[str]) -> str:
            url = _clean_url(match.group(1))
            return match.group(0) if url in valid_urls else ""

        next_line = media_image_re.sub(_replace, line)
        payload = next_line.split(":", 1)[1].strip() if ":" in next_line else ""
        if payload:
            cleaned.append(next_line)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)


def _upsert_icons_section(text: str, icon_rows: list[tuple[str, str]]) -> str:
    if not icon_rows:
        return text
    base = _remove_icons_section(text)
    if not _standalone_icon_section_enabled():
        if "Media:" in base:
            return base
        first_url = icon_rows[0][1]
        lines = base.splitlines()
        insert_idx = 1 if lines else 0
        # Attach the generated icon to the first rendered content block instead
        # of creating a detached trailing media section.
        merged = lines[:insert_idx] + [f"Media: Icon={first_url}"] + lines[insert_idx:]
        while merged and not merged[-1].strip():
            merged.pop()
        return "\n".join(merged)

    lines = base.splitlines()
    insertion_idx = len(lines)
    for idx, line in enumerate(lines):
        header = _normalize_section_header(line)
        if header in {"quick actions", "quick action", "sources"}:
            insertion_idx = idx
            break

    block = ["Icons:"]
    for label, url in icon_rows:
        block.append(f"- {label}: {url}")
    block.append("")

    if insertion_idx > 0 and lines[insertion_idx - 1].strip():
        block.insert(0, "")

    merged = lines[:insertion_idx] + block + lines[insertion_idx:]
    while merged and not merged[-1].strip():
        merged.pop()
    return "\n".join(merged)


def _apply_icon_catalog_postprocess(
    response_text: str,
    query_text: str,
    intent: str | None,
    tags: list[str] | None,
    icon_context: dict[str, object] | None,
) -> str:
    if not icon_context:
        return response_text
    updated = response_text
    if bool(icon_context.get("icons_only_mode")):
        updated = _strip_images_section(updated)
    icon_rows = _build_icon_rows(updated, query_text, intent, tags, icon_context)
    if icon_rows:
        updated = _upsert_icons_section(updated, icon_rows)
    return updated


def _clean_url(value: str) -> str:
    cleaned = value.strip().strip("()[]{}<>\"'").rstrip(".,;:)]}!?")
    if not cleaned:
        return ""
    cleaned = cleaned.split()[0]
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    try:
        parsed = urllib.parse.urlparse(cleaned)
    except ValueError:
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return cleaned


def _is_asset_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in _ASSET_EXTENSIONS):
        return True
    host = parsed.netloc.lower()
    return any(hint in host for hint in _ASSET_HOST_HINTS)


def _extract_asset_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if not text:
        return entries
    section: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            section = None
            continue
        header = stripped.rstrip(":").lower()
        if header in {"images", "icons", "assets", "files"}:
            section = header
            continue
        if stripped.lower().startswith("media:"):
            for media_match in re.finditer(r"\b(Image|Icon)=(https?://[^\s]+)", stripped, flags=re.IGNORECASE):
                cleaned = _clean_url(media_match.group(2))
                if not cleaned:
                    continue
                entries.append({
                    "url": cleaned,
                    "kind": "image" if media_match.group(1).lower() == "image" else "icon",
                })
            continue
        candidates = _URL_RE.findall(line)
        if not candidates:
            continue
        if section == "icons":
            kind = "icon"
        elif section == "images":
            kind = "image"
        else:
            kind = "asset"
        for match in candidates:
            cleaned = _clean_url(match)
            if not cleaned:
                continue
            if _is_asset_url(cleaned):
                entries.append({"url": cleaned, "kind": kind})
    dedup: dict[str, dict[str, str]] = {}
    for entry in entries:
        url = entry["url"]
        if url not in dedup:
            dedup[url] = entry
    return list(dedup.values())


def _extract_asset_urls(text: str) -> list[str]:
    return [item["url"] for item in _extract_asset_entries(text)]


def _normalize_intent_key(value: str | None) -> str:
    if not value:
        return ""
    norm = value.strip().lower()
    norm = re.sub(r"[^a-z0-9]+", "_", norm).strip("_")
    return norm


def _is_visual_intent(intent: str | None, tags: list[str] | None) -> bool:
    intent_key = _normalize_intent_key(intent)
    if intent_key in _VISUAL_INTENT_HINTS:
        return True
    for tag in tags or []:
        if _normalize_intent_key(str(tag)) in _VISUAL_INTENT_HINTS:
            return True
    return False


def _asset_quality_check(
    response_text: str,
    intent: str | None,
    tags: list[str] | None,
    declared_assets_count: int,
    downloaded_assets_count: int,
    min_valid_rate: float,
    icons_only_mode: bool = False,
) -> tuple[bool, str]:
    entries = _extract_asset_entries(response_text)
    has_image = any(item.get("kind") == "image" for item in entries)
    has_icon = any(item.get("kind") == "icon" for item in entries)
    visual = _is_visual_intent(intent, tags)
    random_hosts = []
    for item in entries:
        url = str(item.get("url") or "")
        host = urllib.parse.urlparse(url).netloc.lower()
        if any(host == blocked or host.endswith(f".{blocked}") for blocked in _RANDOM_PLACEHOLDER_ASSET_HOSTS):
            random_hosts.append(host)
    if random_hosts:
        return False, f"random/placeholder media hosts are not allowed: {', '.join(sorted(set(random_hosts)))}"

    if visual and declared_assets_count == 0:
        return False, "visual intent requires inline Media or Icons entries but none were declared"
    if visual and not icons_only_mode and not has_image and not has_icon:
        return False, "visual intent response has no usable image or icon media"
    if visual and not has_icon:
        return False, "visual intent response is missing icon URLs in Icons section"

    if declared_assets_count > 0:
        valid_rate = downloaded_assets_count / declared_assets_count
        if valid_rate < min_valid_rate:
            return (
                False,
                f"asset URL download rate too low ({downloaded_assets_count}/{declared_assets_count}, {valid_rate:.2f})",
            )

    return True, "ok"


def _build_real_asset_retry_prompt(
    base_prompt: str,
    reason: str,
    intent: str | None,
    tags: list[str] | None,
) -> str:
    visual = _is_visual_intent(intent, tags)
    visual_req = (
        "Include both Images and Icons sections with relevant URLs."
        if visual
        else "Images/Icons are optional unless clearly useful."
    )
    return (
        f"{base_prompt}\n\n"
        "Asset validation failed for the previous draft. Regenerate the COMPLETE response from scratch.\n"
        f"Failure reason: {reason}\n\n"
        "Hard requirements for this retry:\n"
        f"- {visual_req}\n"
        "- Use only real, publicly reachable media URLs that are directly downloadable.\n"
        "- Do NOT use upload.wikimedia.org, images.unsplash.com, cdn.pixabay.com, or deep images.pexels.com links (commonly blocked/dead in this pipeline).\n"
        "- Do NOT use random or placeholder media hosts such as loremflickr.com, picsum.photos, placekitten.com, placehold.co, placeholder.com, or dummyimage.com.\n"
        "- Prefer direct verified image URLs (jpg/png/webp) with display-friendly size for cards (around 1200x800, landscape).\n"
        "- When uncertain about a verified image, use icon-only media and omit the image.\n"
        "- For icons, prefer direct lightweight SVGs suitable for UI (roughly 64-256 px square), e.g.\n"
        "  https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/<icon-name>.svg\n"
        "- Keep each URL adjacent to the specific option/row it belongs to.\n"
        "- Keep the original answer quality and structure.\n"
        "Return plain text only."
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._") or "asset"


def _download_asset_url(
    url: str,
    response_id: str,
    asset_index: int,
    assets_dir: Path,
    logger,
    url_cache: dict[str, Path],
    max_bytes: int,
    local_icon_url_map: dict[str, Path] | None = None,
) -> dict | None:
    if url in url_cache:
        path = url_cache[url]
        return {"url": url, "path": str(path.relative_to(assets_dir.parent))}

    parsed = urllib.parse.urlparse(url)
    basename = Path(parsed.path).name
    safe_name = _safe_name(basename) or f"asset_{asset_index}"
    content_type = None

    data: bytes
    local_icon_path = None
    if local_icon_url_map:
        local_icon_path = local_icon_url_map.get(url)
    if local_icon_path and local_icon_path.exists():
        data = local_icon_path.read_bytes()
        guessed_type, _ = mimetypes.guess_type(local_icon_path.name)
        content_type = guessed_type or "image/svg+xml"
    else:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type")
            data = resp.read(max_bytes + 1)

    if len(data) > max_bytes:
        logger.warning("Stage2 asset too large, skipped url=%s", url)
        return None

    ext = ""
    if "." in safe_name:
        ext = Path(safe_name).suffix
    if not ext:
        if content_type:
            content_type = content_type.split(";")[0].strip().lower()
        ext = _MIME_EXTENSION_MAP.get(content_type or "")
        if not ext and content_type:
            guessed = mimetypes.guess_extension(content_type, strict=False)
            if guessed:
                ext = guessed
    if not ext:
        ext = ".bin"

    filename = f"{response_id}_{asset_index}_{_safe_name(Path(safe_name).stem)}{ext}"
    dest = assets_dir / filename
    dest.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    url_cache[url] = dest
    return {
        "url": url,
        "path": str(dest.relative_to(assets_dir.parent)),
        "sha256": sha,
        "bytes": len(data),
    }


def _download_assets(
    response_text: str,
    response_id: str,
    assets_dir: Path,
    logger,
    url_cache: dict[str, Path],
    max_bytes: int = 25 * 1024 * 1024,
    local_icon_url_map: dict[str, Path] | None = None,
) -> tuple[list[dict], dict[str, str], int]:
    assets: list[dict] = []
    rewrites: dict[str, str] = {}
    if _offline_mode_enabled():
        return assets, rewrites, 0

    entries = _extract_asset_entries(response_text)
    declared_count = len(entries)
    if not entries:
        return assets, rewrites, 0

    assets_dir.mkdir(parents=True, exist_ok=True)
    for idx, entry in enumerate(entries, start=1):
        original_url = entry.get("url") or ""
        kind = entry.get("kind") or "asset"
        if not original_url:
            continue

        asset_item = None
        try:
            asset_item = _download_asset_url(
                original_url,
                response_id,
                idx,
                assets_dir,
                logger,
                url_cache,
                max_bytes,
                local_icon_url_map=local_icon_url_map,
            )
        except Exception as exc:
            logger.warning("Stage2 asset download failed url=%s err=%s", original_url, exc)
        if asset_item is None:
            logger.warning(
                "Stage2 asset unresolved response_id=%s kind=%s url=%s",
                response_id,
                kind,
                original_url,
            )
            continue

        assets.append(asset_item)

    return assets, rewrites, declared_count


def run_stage2(
    queries_path: Path,
    prompt_path: Path,
    batch_prompt_path: Path | None,
    adapter: BaseLLMAdapter,
    responses_path: Path,
    n_per_query: int,
    batch_size: int,
    query_batch_size: int,
    group_by_intent: bool,
    batch_fallback_per_query: bool,
    temperatures: list[float],
    max_tokens: int,
    seed: int,
    rate_limiter: RateLimiter,
    cache: PromptCache,
    logger,
    max_total: int | None = None,
    max_attempts: int = 3,
) -> None:
    prompt_template = load_prompt(prompt_path)

    existing_ids = {row.get("response_id") for row in iter_jsonl(responses_path)}
    existing_hashes = set()
    for row in iter_jsonl(responses_path):
        text = row.get("response_text", "")
        if text:
            existing_hashes.add(hash_text(normalize_text(text)))

    writer = JsonlWriter(responses_path)
    assets_dir = responses_path.parent / "assets"
    url_cache: dict[str, Path] = {}

    dataset_root = _detect_dataset_root(responses_path.parent)
    icon_context = _load_local_icon_context(dataset_root, logger)
    local_icon_url_map: dict[str, Path] = {}
    icons_only_mode = False
    if isinstance(icon_context, dict):
        raw_map = icon_context.get("url_to_local_file")
        if isinstance(raw_map, dict):
            local_icon_url_map = {
                str(k): v
                for k, v in raw_map.items()
                if isinstance(k, str) and isinstance(v, Path)
            }
        icons_only_mode = bool(icon_context.get("icons_only_mode"))

    real_asset_retry_enabled = _real_asset_retry_enabled()
    real_asset_retry_max_attempts = _real_asset_retry_max_attempts()
    real_asset_retry_min_valid_rate = _real_asset_retry_min_valid_rate()
    if real_asset_retry_enabled:
        logger.info(
            "Stage2 real-asset retry enabled attempts=%s min_valid_rate=%.2f",
            real_asset_retry_max_attempts,
            real_asset_retry_min_valid_rate,
        )

    query_states: list[dict] = []
    for query in iter_jsonl(queries_path):
        query_id = query.get("query_id")
        query_text = query.get("query_text")
        if not query_id or not query_text:
            continue
        query_states.append(
            {
                "query_id": query_id,
                "query_text": str(query_text),
                "intent_value": query.get("intent") if isinstance(query.get("intent"), str) else "",
                "tags_list": query.get("tags") if isinstance(query.get("tags"), list) else [],
                "remaining": int(n_per_query),
                "n_idx": 1,
            }
        )

    if batch_size <= 0:
        batch_size = 1
    if query_batch_size <= 0:
        query_batch_size = 1

    use_gemini_query_batch = (
        adapter.spec.provider == "gemini"
        and hasattr(adapter, "generate_batch")
        and query_batch_size > 1
    )
    if use_gemini_query_batch:
        logger.info(
            "Stage2 Gemini query batching enabled query_batch_size=%s group_by_intent=%s fallback_per_query=%s",
            query_batch_size,
            group_by_intent,
            batch_fallback_per_query,
        )

    total_created = 0

    def _temperature_for_n(n_idx_value: int) -> float:
        return temperatures[(n_idx_value - 1) % len(temperatures)] if temperatures else 0.7

    def _prepare_entry(state: dict) -> dict | None:
        while state["remaining"] > 0:
            probe_id = _make_response_id(state["query_id"], state["n_idx"])
            if probe_id not in existing_ids:
                break
            state["n_idx"] += 1
            state["remaining"] -= 1
        if state["remaining"] <= 0:
            return None

        batch = max(1, min(batch_size, int(state["remaining"])))
        n_idx = int(state["n_idx"])
        temperature = _temperature_for_n(n_idx)
        tags_list = state["tags_list"]
        tags_value = ", ".join(str(tag).strip() for tag in tags_list if str(tag).strip())
        prompt = render_prompt(
            prompt_template,
            query_text=state["query_text"],
            intent=state["intent_value"],
            tags=tags_value,
        )
        if batch > 1:
            prompt = (
                f"{prompt}\n\nReturn exactly {batch} distinct responses as a JSON array of strings."
            )
        prompt_hash = hash_text(f"{adapter.spec.name}:{prompt}")
        return {
            "state": state,
            "query_id": state["query_id"],
            "query_text": state["query_text"],
            "intent_value": state["intent_value"],
            "tags_list": tags_list,
            "batch": batch,
            "temperature": temperature,
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "seed_value": seed + n_idx,
        }

    def _generate_single_entry(entry: dict):
        cached = cache.get(entry["prompt_hash"])
        if cached:
            return (
                cached.text.strip(),
                0.0,
                0,
                0,
                adapter.spec.provider,
                adapter.spec.model,
            )

        def _call():
            rate_limiter.acquire()
            return adapter.generate(
                prompt=entry["prompt"],
                system=None,
                temperature=entry["temperature"],
                max_tokens=max_tokens,
                seed=entry["seed_value"],
                json_mode=entry["batch"] > 1 and adapter.spec.supports_json_mode,
            )

        result = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = with_retry(_call, max_attempts=1)
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    logger.error(
                        "Stage2 rate limit info: limits=%s headers=%s",
                        exc.limits or "unset",
                        exc.headers or "none",
                    )
                    raise
                if attempt < max_attempts:
                    logger.warning(
                        "Stage2 transient exception query_id=%s attempt=%s/%s err=%s",
                        entry["query_id"],
                        attempt,
                        max_attempts,
                        exc,
                    )
                    _sleep_backoff(attempt)
                    continue
                raise

            if not result.error:
                break
            if _is_transient_error(result.error) and attempt < max_attempts:
                logger.warning(
                    "Stage2 transient error query_id=%s attempt=%s/%s err=%s",
                    entry["query_id"],
                    attempt,
                    max_attempts,
                    result.error,
                )
                _sleep_backoff(attempt)
                continue
            logger.error("Stage2 error query_id=%s: %s", entry["query_id"], result.error)
            result = None
            break

        if result is None or result.error:
            return None

        cache.set(entry["prompt_hash"], result.text, result.raw)
        return (
            result.text.strip(),
            result.latency_ms,
            result.input_tokens,
            result.output_tokens,
            result.provider,
            result.model,
        )

    def _consume_failure(entry: dict) -> None:
        state = entry["state"]
        state["n_idx"] += 1
        state["remaining"] -= 1

    def _consume_generated(
        entry: dict,
        raw_text: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
    ) -> bool:
        nonlocal total_created

        state = entry["state"]
        query_id = entry["query_id"]
        query_text = entry["query_text"]
        intent_value = entry["intent_value"]
        tags_list = entry["tags_list"]
        batch = int(entry["batch"])
        prompt = entry["prompt"]
        temperature = float(entry["temperature"])
        responses: list[str]
        if batch > 1:
            try:
                payload = extract_json(raw_text)
                if isinstance(payload, list):
                    responses = [str(item).strip() for item in payload if str(item).strip()]
                else:
                    responses = [raw_text]
            except Exception:
                responses = [raw_text]
        else:
            responses = [raw_text]

        response_id = _make_response_id(query_id, state["n_idx"])
        for response_text in responses[:batch]:
            if not response_text:
                continue

            selected_text = _apply_icon_catalog_postprocess(
                response_text,
                query_text,
                intent_value,
                tags_list,
                icon_context,
            )
            selected_prompt = prompt
            selected_latency_ms = latency_ms
            selected_input_tokens = input_tokens
            selected_output_tokens = output_tokens
            selected_provider = provider
            selected_model = model

            max_asset_attempts = (
                real_asset_retry_max_attempts if real_asset_retry_enabled else 1
            )
            asset_retry_attempts = 1
            assets: list[dict] = []
            declared_assets_count = 0
            valid_asset_rate = 1.0
            asset_quality_ok = True
            asset_quality_reason = "ok"

            for asset_attempt in range(1, max_asset_attempts + 1):
                selected_text = _apply_icon_catalog_postprocess(
                    selected_text,
                    query_text,
                    intent_value,
                    tags_list,
                    icon_context,
                )
                assets, _, declared_assets_count = _download_assets(
                    selected_text,
                    response_id,
                    assets_dir,
                    logger,
                    url_cache,
                    local_icon_url_map=local_icon_url_map,
                )
                valid_asset_rate = (
                    float(len(assets) / declared_assets_count)
                    if declared_assets_count > 0
                    else 1.0
                )
                asset_quality_ok, asset_quality_reason = _asset_quality_check(
                    selected_text,
                    intent_value,
                    tags_list,
                    declared_assets_count,
                    len(assets),
                    real_asset_retry_min_valid_rate,
                    icons_only_mode=icons_only_mode,
                )

                if not real_asset_retry_enabled or asset_quality_ok:
                    break

                if asset_attempt >= max_asset_attempts:
                    logger.warning(
                        "Stage2 asset retry exhausted response_id=%s reason=%s",
                        response_id,
                        asset_quality_reason,
                    )
                    break

                retry_prompt = _build_real_asset_retry_prompt(
                    prompt,
                    asset_quality_reason,
                    intent_value,
                    tags_list,
                )
                retry_prompt = (
                    f"{retry_prompt}\n\n"
                    f"Retry attempt: {asset_attempt + 1}\n"
                    "Previous draft (for correction):\n"
                    f"{selected_text[:6000]}"
                )
                retry_hash = hash_text(f"{adapter.spec.name}:{retry_prompt}")
                cached_retry = cache.get(retry_hash)
                if cached_retry:
                    retry_text = cached_retry.text.strip()
                    retry_latency_ms = 0.0
                    retry_input_tokens = 0
                    retry_output_tokens = 0
                    retry_provider = adapter.spec.provider
                    retry_model = adapter.spec.model
                else:

                    def _retry_call():
                        rate_limiter.acquire()
                        return adapter.generate(
                            prompt=retry_prompt,
                            system=None,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            seed=seed + int(state["n_idx"]) + asset_attempt,
                            json_mode=False,
                        )

                    retry_result = None
                    for retry_attempt in range(1, max_attempts + 1):
                        try:
                            retry_result = with_retry(_retry_call, max_attempts=1)
                        except Exception as exc:
                            if isinstance(exc, LLMRateLimitError):
                                logger.error(
                                    "Stage2 rate limit info: limits=%s headers=%s",
                                    exc.limits or "unset",
                                    exc.headers or "none",
                                )
                                raise
                            if retry_attempt < max_attempts:
                                logger.warning(
                                    "Stage2 retry transient exception query_id=%s attempt=%s/%s err=%s",
                                    query_id,
                                    retry_attempt,
                                    max_attempts,
                                    exc,
                                )
                                _sleep_backoff(retry_attempt)
                                continue
                            retry_result = None
                            break
                        if retry_result and not retry_result.error:
                            break
                        if (
                            retry_result
                            and retry_result.error
                            and _is_transient_error(retry_result.error)
                            and retry_attempt < max_attempts
                        ):
                            logger.warning(
                                "Stage2 retry transient error query_id=%s attempt=%s/%s err=%s",
                                query_id,
                                retry_attempt,
                                max_attempts,
                                retry_result.error,
                            )
                            _sleep_backoff(retry_attempt)
                            continue
                        retry_result = None
                        break

                    if retry_result is None or retry_result.error:
                        logger.warning(
                            "Stage2 asset retry generation failed response_id=%s reason=%s",
                            response_id,
                            asset_quality_reason,
                        )
                        break

                    retry_text = retry_result.text.strip()
                    retry_latency_ms = retry_result.latency_ms
                    retry_input_tokens = retry_result.input_tokens
                    retry_output_tokens = retry_result.output_tokens
                    retry_provider = retry_result.provider
                    retry_model = retry_result.model
                    cache.set(retry_hash, retry_result.text, retry_result.raw)

                if not retry_text:
                    break

                selected_text = retry_text
                selected_prompt = retry_prompt
                selected_latency_ms = retry_latency_ms
                selected_input_tokens = retry_input_tokens
                selected_output_tokens = retry_output_tokens
                selected_provider = retry_provider
                selected_model = retry_model
                asset_retry_attempts = asset_attempt + 1

            selected_text = _strip_unresolved_media_images(selected_text, assets)
            final_asset_entries = _extract_asset_entries(selected_text)
            final_asset_urls = {entry["url"] for entry in final_asset_entries}
            if final_asset_urls:
                assets = [
                    asset for asset in assets
                    if str(asset.get("url") or "").strip() in final_asset_urls
                ]
            else:
                assets = []
            declared_assets_count = len(final_asset_entries)
            valid_asset_rate = (
                float(len(assets) / declared_assets_count)
                if declared_assets_count > 0
                else 1.0
            )
            asset_quality_ok, asset_quality_reason = _asset_quality_check(
                selected_text,
                intent_value,
                tags_list,
                declared_assets_count,
                len(assets),
                real_asset_retry_min_valid_rate,
                icons_only_mode=icons_only_mode,
            )
            norm_hash = hash_text(normalize_text(selected_text))
            if norm_hash in existing_hashes:
                logger.info("Stage2 duplicate response query_id=%s", query_id)
                state["n_idx"] += 1
                state["remaining"] -= 1
                response_id = _make_response_id(query_id, state["n_idx"])
                continue

            record = {
                "response_id": response_id,
                "query_id": query_id,
                "n_idx": state["n_idx"],
                "response_text": selected_text,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "assets": assets,
                "asset_stats": {
                    "declared_asset_urls": declared_assets_count,
                    "downloaded_assets": len(assets),
                    "asset_url_valid_rate": valid_asset_rate,
                    "real_asset_retry_enabled": real_asset_retry_enabled,
                    "real_asset_retry_attempts": asset_retry_attempts,
                    "asset_quality_ok": asset_quality_ok,
                    "asset_quality_reason": asset_quality_reason,
                    "icon_catalog_enabled": bool(icon_context),
                    "icons_only_mode": icons_only_mode,
                },
                "gen": {
                    "provider": selected_provider,
                    "model": selected_model,
                    "latency_ms": selected_latency_ms,
                    "input_tokens": selected_input_tokens,
                    "output_tokens": selected_output_tokens,
                    "cost_usd": None,
                    "retry_prompt_used": selected_prompt != prompt,
                },
            }
            writer.append(record)
            existing_ids.add(response_id)
            existing_hashes.add(norm_hash)
            logger.info("Stage2 created response_id=%s", response_id)
            state["n_idx"] += 1
            state["remaining"] -= 1
            total_created += 1
            if max_total is not None and total_created >= max_total:
                logger.info("Stage2 reached max_total=%s", max_total)
                return True
            response_id = _make_response_id(query_id, state["n_idx"])

        if batch == 1 and not responses:
            state["n_idx"] += 1
            state["remaining"] -= 1
        return False

    while True:
        if max_total is not None and total_created >= max_total:
            logger.info("Stage2 reached max_total=%s", max_total)
            return

        pending_states = [state for state in query_states if state["remaining"] > 0]
        if not pending_states:
            return

        if not use_gemini_query_batch:
            entry = _prepare_entry(pending_states[0])
            if entry is None:
                continue
            generated = _generate_single_entry(entry)
            if generated is None:
                _consume_failure(entry)
                continue
            if _consume_generated(entry, *generated):
                return
            continue

        ordered_states = pending_states
        if group_by_intent and pending_states:
            base_intent = str(pending_states[0].get("intent_value") or "")
            same_intent = [s for s in pending_states if str(s.get("intent_value") or "") == base_intent]
            other_intent = [s for s in pending_states if str(s.get("intent_value") or "") != base_intent]
            ordered_states = same_intent + other_intent

        entries: list[dict] = []
        base_temperature: float | None = None
        for state in ordered_states:
            entry = _prepare_entry(state)
            if entry is None:
                continue
            if base_temperature is None:
                base_temperature = entry["temperature"]
            if entries and entry["temperature"] != base_temperature:
                continue
            entries.append(entry)
            if len(entries) >= query_batch_size:
                break

        if not entries:
            continue

        cached_payloads: dict[int, tuple[str, float, int, int, str, str]] = {}
        uncached_entries: list[dict] = []
        for entry in entries:
            cached = cache.get(entry["prompt_hash"])
            if cached:
                cached_payloads[id(entry)] = (
                    cached.text.strip(),
                    0.0,
                    0,
                    0,
                    adapter.spec.provider,
                    adapter.spec.model,
                )
            else:
                uncached_entries.append(entry)

        generated_payloads: dict[int, tuple[str, float, int, int, str, str] | None] = {}
        generated_payloads.update(cached_payloads)

        if uncached_entries:
            prompts = [entry["prompt"] for entry in uncached_entries]
            seeds = [entry["seed_value"] for entry in uncached_entries]
            json_mode = adapter.spec.supports_json_mode and any(
                int(entry["batch"]) > 1 for entry in uncached_entries
            )
            batch_results = None

            def _call_batch():
                rate_limiter.acquire()
                return adapter.generate_batch(
                    prompts=prompts,
                    system=None,
                    temperature=float(uncached_entries[0]["temperature"]),
                    max_tokens=max_tokens,
                    seeds=seeds,
                    json_mode=json_mode,
                    batch_name=f"stage2_{int(time.time())}",
                )

            try:
                batch_results = with_retry(_call_batch, max_attempts=max_attempts)
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    logger.error(
                        "Stage2 rate limit info: limits=%s headers=%s",
                        exc.limits or "unset",
                        exc.headers or "none",
                    )
                    raise
                logger.warning("Stage2 batch failed; falling back to per-query calls: %s", exc)
                batch_results = None

            if batch_results is not None and len(batch_results) == len(uncached_entries):
                for entry, result in zip(uncached_entries, batch_results):
                    if result.error:
                        logger.warning(
                            "Stage2 batch item error query_id=%s err=%s",
                            entry["query_id"],
                            result.error,
                        )
                        generated_payloads[id(entry)] = None
                        continue
                    cache.set(entry["prompt_hash"], result.text, result.raw)
                    generated_payloads[id(entry)] = (
                        result.text.strip(),
                        result.latency_ms,
                        result.input_tokens,
                        result.output_tokens,
                        result.provider,
                        result.model,
                    )
            else:
                for entry in uncached_entries:
                    generated_payloads[id(entry)] = None

            for entry in uncached_entries:
                if generated_payloads.get(id(entry)) is not None:
                    continue
                if not batch_fallback_per_query:
                    continue
                generated_payloads[id(entry)] = _generate_single_entry(entry)

        for entry in entries:
            payload = generated_payloads.get(id(entry))
            if payload is None:
                _consume_failure(entry)
                continue
            if _consume_generated(entry, *payload):
                return
