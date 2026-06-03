from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from functools import lru_cache
from typing import Any

from llm.http_transport import urlopen


MAX_IMAGES_TO_VALIDATE = 6
MAX_TABLE_ROW_IMAGES = 4
USER_AGENT = "A2UI GenUICraft Dataset/1.0 image-resolver"


def enrich_response_with_commons_media(
    response_text: str,
    query_text: str,
    intent: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Add verified Commons image Media lines to travel/place blocks before Stage 2 downloads assets."""
    if not response_text or not looks_like_travel(query_text, response_text, intent, tags):
        return response_text
    if len(_image_urls(response_text)) >= 2:
        return response_text

    destination = extract_travel_location_keyword(query_text)
    if not destination:
        return response_text

    seen_urls = set(_image_urls(response_text))
    added = 0
    out: list[str] = []
    lines = response_text.splitlines()
    for index, line in enumerate(lines):
        out.append(line)
        if added >= MAX_TABLE_ROW_IMAGES:
            continue
        if _next_nonempty_is_media(lines, index + 1):
            continue
        title = _travel_block_title(line)
        if not title:
            continue
        queries = travel_search_queries_from_values([title], query_text)
        url = next((item for q in queries if (item := search_commons_image_url(q)) and item not in seen_urls), None)
        if not url:
            continue
        seen_urls.add(url)
        added += 1
        alt = _clean_media_alt(title, destination)
        out.append(f"Media: Image={url} Alt={alt}")
    return "\n".join(out)


def repair_flat_spec_images(genui_json: Any, query_text: str, response_text: str) -> tuple[Any, int]:
    """Mirror Android table-row image repair for dataset-generated flat-spec IR."""
    if not isinstance(genui_json, dict):
        return genui_json, 0
    elements = genui_json.get("elements")
    state = genui_json.get("state")
    if not isinstance(elements, dict):
        return genui_json, 0
    if not isinstance(state, dict):
        state = {}

    resolved_count = 0
    seen_urls: set[str] = set()

    # Port of Android PipelineImageResolver: repair remote Image components first.
    text_index = _build_text_index(elements)
    checked = 0
    for element_id, element in elements.items():
        if checked >= MAX_IMAGES_TO_VALIDATE:
            break
        if not isinstance(element, dict):
            continue
        if str(element.get("type") or "").lower() != "image":
            continue
        props = element.get("props")
        if not isinstance(props, dict):
            props = {}
            element["props"] = props
        current_url = _image_url_from_props(props) or ""
        if not _should_validate_remote_image(current_url):
            continue
        checked += 1
        reachable = is_reachable_image_url(current_url)
        if reachable and props.get("fallbackUrl"):
            continue
        search_query = build_commons_search_query(
            element_id,
            current_url,
            props,
            query_text,
            response_text,
            text_index,
        )
        fallback_url = search_commons_image_url(search_query)
        if not fallback_url:
            continue
        props["fallbackUrl"] = fallback_url
        resolved_count += 1
        if not reachable:
            _put_image_url(props, fallback_url)

    table_images_added = 0
    for element in elements.values():
        if table_images_added >= MAX_TABLE_ROW_IMAGES:
            break
        if not isinstance(element, dict):
            continue
        if str(element.get("type") or "").lower() != "table":
            continue
        props = element.get("props")
        if not isinstance(props, dict):
            continue
        if not _looks_like_travel_table(props, query_text, response_text):
            continue
        rows = _table_rows(genui_json, state, props)
        if not rows:
            continue
        _ensure_table_image_columns(props)
        for row in rows:
            if table_images_added >= MAX_TABLE_ROW_IMAGES:
                break
            if not isinstance(row, dict):
                continue
            existing = _first_string(row, ("image", "photo", "imageUrl", "mediaImage"))
            if existing and should_use_remote_image(existing):
                continue
            queries = travel_search_queries_from_row(row, query_text)
            url = next((item for q in queries if (item := search_commons_image_url(q)) and item not in seen_urls), None)
            if not url:
                continue
            seen_urls.add(url)
            row["image"] = url
            row["imageAlt"] = travel_row_image_alt(row, query_text)
            table_images_added += 1
            resolved_count += 1
    return genui_json, resolved_count


def looks_like_travel(
    query_text: str,
    response_text: str = "",
    intent: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    hay = " ".join(
        [
            query_text or "",
            response_text[:2000] if response_text else "",
            intent or "",
            " ".join(str(tag) for tag in tags or []),
        ]
    ).lower()
    return any(
        token in hay
        for token in (
            "travel",
            "trip",
            "itinerary",
            "vacation",
            "tour",
            "day plan",
            "sightseeing",
            "places to visit",
        )
    )


def extract_travel_location_keyword(query_text: str) -> str:
    text = re.sub(r"\s+", " ", query_text or "").strip()
    patterns = [
        r"\b(?:trip|vacation|itinerary|travel|visit|tour)\s+(?:to|in|for|around)\s+([A-Z][A-Za-z .'-]{2,50}(?:,\s*[A-Z][A-Za-z .'-]{2,40})?)(?=\s+(?:from|for|with|on|between|during)\b|[.;]|$)",
        r"\b(?:trip|vacation|itinerary|travel|visit|tour)\s+(?:to|in|for|around)\s+([a-z][A-Za-z .'-]{2,50}(?:,\s*[A-Za-z .'-]{2,40})?)(?=\s+(?:from|for|with|on|between|during)\b|[.;]|$)",
        r"\b(?:in|to|for|near|around)\s+([A-Z][A-Za-z .'-]{2,50}(?:,\s*[A-Z][A-Za-z .'-]{2,40})?)(?=\s+(?:from|for|with|on|between|during)\b|[.;]|$)",
        r"\b(?:in|to|for|near|around)\s+([a-z][A-Za-z .'-]{2,50}(?:,\s*[A-Za-z .'-]{2,40})?)(?=\s+(?:from|for|with|on|between|during)\b|[.;]|$)",
        r"\b(?:in|to|for|near|around)\s+([A-Z][A-Za-z .'-]{2,50})$",
        r"\b(?:in|to|for|near|around)\s+([a-z][A-Za-z .'-]{2,50})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            value = re.sub(r"\b(?:for|with|from|on|at|by)\b.*$", "", value, flags=re.I)
            return _titleish(value)
    words = [w for w in re.split(r"[^A-Za-z]+", text) if len(w) > 2]
    blocked = {
        "show",
        "plan",
        "trip",
        "travel",
        "vacation",
        "itinerary",
        "days",
        "day",
        "for",
        "with",
    }
    candidates = [w for w in words if w.lower() not in blocked]
    return _titleish(" ".join(candidates[-2:])) if candidates else ""


def travel_search_queries_from_row(row: dict[str, Any], query_text: str) -> list[str]:
    values = [
        str(row.get(key) or "").strip()
        for key in (
            "area",
            "place",
            "location",
            "title",
            "focus",
            "destination",
            "neighborhood",
            "morning",
            "morningActivity",
            "afternoon",
            "afternoonActivity",
            "evening",
            "dinnerSuggestion",
        )
        if str(row.get(key) or "").strip()
    ]
    return travel_search_queries_from_values(values, query_text)


def travel_search_queries_from_values(values: list[str], query_text: str) -> list[str]:
    destination = extract_travel_location_keyword(query_text)
    phrases: list[str] = []
    for value in values:
        phrases.extend(extract_travel_place_phrases(value))
    if not phrases:
        phrases.extend(values[:2])
    phrases = sorted(_distinct(phrases), key=lambda p: (1 if "coast" in p.lower() else 0, len(p)))
    queries = []
    for phrase in phrases:
        queries.append(f"{destination} {phrase}".strip())
    if destination:
        for value in values[:2]:
            queries.append(f"{destination} {value}".strip())
        if any("beach" in value.lower() or "coast" in value.lower() for value in values):
            queries.append(f"{destination} beach")
        if any("old town" in value.lower() or "town" in value.lower() for value in values):
            queries.append(f"{destination} old town")
        if any("island" in value.lower() for value in values):
            queries.append(f"{destination} island")
    return _distinct([normalize_search_query(q) for q in queries if normalize_search_query(q)])


def extract_travel_place_phrases(value: str) -> list[str]:
    suffixes = (
        "Beach|Beaches|Town|Market|Temple|Museum|Palace|Cape|Viewpoint|Island|Islands|"
        "Bay|Garden|Park|Hill|Hills|Buddha|Falls|Lake|Fort|Church|Cathedral|Mosque|"
        "Coast|Harbor|Harbour|Pier|Road|Street|Village"
    )
    pattern = re.compile(
        rf"\b([A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*(?:\s+(?:or|and)\s+[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*)?\s+(?:{suffixes}))\b"
    )
    out: list[str] = []
    for match in pattern.finditer(value or ""):
        out.extend(_expand_joined_place_phrase(match.group(1)))
    return [p.strip(" .,;:") for p in out if len(p.strip(" .,;:")) >= 4]


def normalize_search_query(raw: str) -> str:
    blocked = {
        "image",
        "photo",
        "picture",
        "media",
        "hero",
        "jpg",
        "jpeg",
        "png",
        "webp",
        "show",
        "with",
        "from",
        "into",
        "your",
        "this",
        "that",
        "the",
        "and",
        "for",
        "day",
        "days",
        "vacation",
        "itinerary",
    }
    tokens = re.sub(r"https?://\S+", " ", raw)
    tokens = re.sub(r"[^A-Za-z0-9\s'-]", " ", tokens).split()
    cleaned = []
    for token in tokens:
        item = token.strip("'-")
        if len(item) >= 3 and item.lower() not in blocked:
            cleaned.append(item)
    return " ".join(_distinct(cleaned)[:8]) or "travel landmark"


def build_commons_search_query(
    image_element_id: str,
    current_url: str,
    props: dict[str, Any],
    query_text: str,
    response_text: str,
    text_index: dict[str, str],
) -> str:
    prop_hints = [
        str(props.get(key) or "").strip()
        for key in ("alt", "title", "label", "caption", "description")
        if str(props.get(key) or "").strip()
    ]
    values = [
        *prop_hints,
        _nearby_text_hint(image_element_id, text_index) or "",
        _filename_search_hint(current_url) or "",
        _first_response_title(response_text) or "",
        query_text or "",
    ]
    return normalize_search_query(" ".join(value for value in values if value))


@lru_cache(maxsize=4096)
def search_commons_image_url(search_query: str) -> str | None:
    encoded = urllib.parse.quote(search_query)
    api_url = (
        "https://commons.wikimedia.org/w/api.php"
        f"?action=query&generator=search&gsrsearch={encoded}&gsrnamespace=6&gsrlimit=8"
        "&prop=imageinfo&iiprop=url|extmetadata|mime&format=json"
    )
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=8) as resp:
            raw = resp.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
    except Exception:
        return None
    try:
        root = json.loads(raw)
        pages = ((root.get("query") or {}).get("pages") or {})
        for page in pages.values():
            image_info = (page.get("imageinfo") or [None])[0]
            if not isinstance(image_info, dict):
                continue
            mime = str(image_info.get("mime") or "").lower()
            if mime and not mime.startswith("image/"):
                continue
            url = str(image_info.get("url") or image_info.get("thumburl") or "").strip()
            if should_use_remote_image(url):
                return url
    except Exception:
        return None
    return None


@lru_cache(maxsize=4096)
def is_reachable_image_url(raw_url: str) -> bool:
    url = (raw_url or "").strip()
    if not _should_validate_remote_image(url):
        return False
    try:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            },
        )
        with urlopen(req, timeout=4) as resp:
            content_type = str(resp.headers.get("Content-Type") or "").lower()
            return 200 <= int(resp.status) <= 399 and (not content_type or content_type.startswith("image/"))
    except Exception:
        return False


def should_use_remote_image(raw_url: str) -> bool:
    url = (raw_url or "").strip()
    if not url.lower().startswith("https://"):
        return False
    host = urllib.parse.urlparse(url).netloc.lower()
    if not host:
        return False
    blocked_hosts = {
        "loremflickr.com",
        "picsum.photos",
        "placekitten.com",
        "placehold.co",
        "placeholder.com",
        "dummyimage.com",
    }
    if any(host == blocked or host.endswith(f".{blocked}") for blocked in blocked_hosts):
        return False
    if host.endswith("wikimedia.org") or host.endswith("googleusercontent.com"):
        return True
    path = urllib.parse.urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"))


def _should_validate_remote_image(raw_url: str) -> bool:
    url = (raw_url or "").strip()
    lower = url.lower()
    if not url or lower.startswith(("assets/", "/assets/", "../assets/", "genuicraft:")):
        return False
    return should_use_remote_image(url)


def _build_text_index(elements: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for element_id, element in elements.items():
        if not isinstance(element, dict):
            continue
        if str(element.get("type") or "").lower() != "text":
            continue
        props = element.get("props")
        if not isinstance(props, dict):
            continue
        text = _string_from_value(props.get("text"))
        if text:
            out[str(element_id)] = text.strip()
    return out


def _image_url_from_props(props: dict[str, Any]) -> str | None:
    for key in ("url", "src", "image", "source"):
        value = _extract_url_from_value(props.get(key))
        if value:
            return value
    return None


def _put_image_url(props: dict[str, Any], url: str) -> None:
    preferred_key = next((key for key in ("url", "src", "image", "source") if key in props), "url")
    current = props.get(preferred_key)
    if isinstance(current, dict):
        current["url"] = url
    else:
        props[preferred_key] = url
    if preferred_key != "url" and "url" not in props:
        props["url"] = url


def _extract_url_from_value(value: Any) -> str | None:
    direct = _string_from_value(value)
    if direct:
        return direct
    if isinstance(value, dict):
        for key in ("uri", "url", "src", "path", "value", "source", "image"):
            nested = _string_from_value(value.get(key))
            if nested:
                return nested
    return None


def _string_from_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        literal = value.get("literalString")
        if isinstance(literal, str) and literal.strip():
            return literal.strip()
    return None


def _nearby_text_hint(image_element_id: str, text_index: dict[str, str]) -> str | None:
    suffix = re.search(r"(\d+)$", str(image_element_id))
    if suffix:
        suffix_value = suffix.group(1)
        for element_id, text in text_index.items():
            if str(element_id).endswith(suffix_value):
                return text
    normalized = re.sub(r"image|media|hero|photo|pic", "", str(image_element_id), flags=re.I)
    normalized = re.sub(r"[_-]+", " ", normalized).strip()
    return normalized if len(normalized) >= 3 else None


def _filename_search_hint(raw_url: str) -> str | None:
    try:
        parsed = urllib.parse.urlparse((raw_url or "").strip())
    except Exception:
        return None
    segment = (parsed.path or "").rsplit("/", 1)[-1]
    if not segment:
        return None
    decoded = urllib.parse.unquote(segment)
    decoded = re.sub(r"\.(png|jpe?g|webp|gif)$", "", decoded, flags=re.I)
    decoded = decoded.replace("_", " ").replace("-", " ").strip()
    return decoded if len(decoded) >= 3 else None


def _first_response_title(response_text: str) -> str | None:
    for line in (response_text or "").splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if 8 <= len(cleaned) <= 90 and "http" not in cleaned.lower():
            return cleaned
    return None


def travel_row_image_alt(row: dict[str, Any], query_text: str) -> str:
    destination = extract_travel_location_keyword(query_text)
    values = [str(v).strip() for v in row.values() if isinstance(v, str) and v.strip()]
    phrase = next((p for value in values for p in extract_travel_place_phrases(value)), None)
    phrase = phrase or _first_string(row, ("area", "place", "location", "title")) or destination
    return _clean_media_alt(phrase, destination)


def _looks_like_travel_table(props: dict[str, Any], query_text: str, response_text: str) -> bool:
    domain = str(props.get("domain") or "").lower()
    if domain in {"travel", "itinerary", "tourism", "trip", "schedule"} and looks_like_travel(query_text, response_text):
        return True
    columns = props.get("columns")
    if not isinstance(columns, list):
        return False
    joined = " ".join(
        " ".join(str(col.get(k) or "") for k in ("key", "label"))
        for col in columns
        if isinstance(col, dict)
    ).lower()
    has_shape = "day" in joined and any(
        token in joined for token in ("morning", "afternoon", "evening", "food", "area", "place", "location")
    )
    return has_shape and looks_like_travel(query_text, response_text)


def _table_rows(payload: dict[str, Any], state: dict[str, Any], props: dict[str, Any]) -> list[Any] | None:
    rows = props.get("rows")
    if isinstance(rows, list):
        return rows
    state_path = str(props.get("statePath") or "").strip()
    if not state_path:
        return None
    value = _json_pointer(state, state_path)
    if isinstance(value, list):
        return value
    value = _json_pointer(payload, state_path)
    return value if isinstance(value, list) else None


def _json_pointer(root: Any, pointer: str) -> Any:
    current = root
    for raw_part in pointer.strip().lstrip("#").split("/"):
        if not raw_part:
            continue
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            idx = int(part)
            current = current[idx] if 0 <= idx < len(current) else None
        else:
            return None
    return current


def _ensure_table_image_columns(props: dict[str, Any]) -> None:
    columns = props.get("columns")
    if not isinstance(columns, list):
        return
    keys = {
        str(col.get("key") or "").lower()
        for col in columns
        if isinstance(col, dict)
    }
    if "image" not in keys:
        columns.append({"key": "image", "label": "Image"})
    if "imagealt" not in keys and "image_alt" not in keys:
        columns.append({"key": "imageAlt", "label": "Image Alt"})


def _first_string(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _travel_block_title(line: str) -> str | None:
    cleaned = line.strip().strip("#*-• ").strip()
    if not cleaned or len(cleaned) > 120 or "http" in cleaned.lower():
        return None
    if re.match(r"^(day\s*\d+|day\s*[A-Za-z]+|\d+\.)\s*[:.-]\s*", cleaned, re.I):
        return cleaned
    if any(suffix.lower() in cleaned.lower() for suffix in (" beach", " town", " island", " bay", " park", " temple", " market")):
        return cleaned
    return None


def _next_nonempty_is_media(lines: list[str], start: int) -> bool:
    for line in lines[start : start + 3]:
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.lower().startswith("media:")
    return False


def _image_urls(text: str) -> list[str]:
    out = []
    pattern = re.compile(
        r"Image\s*=\s*(https?://\S+)|(?<![=])\b(https?://\S+\.(?:jpg|jpeg|png|webp|gif)(?:\?\S*)?)",
        flags=re.I,
    )
    for match in pattern.finditer(text or ""):
        value = match.group(1) or match.group(2) or ""
        value = value.strip().strip(").,;")
        if value:
            out.append(value)
    return out


def _clean_media_alt(title: str, destination: str) -> str:
    parts = [title.strip(" .,:;-"), destination.strip()]
    return ", ".join(_distinct([p for p in parts if p]))


def _titleish(value: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"\s+", value.strip()) if part)


def _distinct(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value.strip())
    return out


def _expand_joined_place_phrase(phrase: str) -> list[str]:
    parts = re.split(r"\s+(?:or|and)\s+", phrase)
    if len(parts) < 2:
        return [phrase]
    suffix = parts[-1].split()[-1].strip()
    if not suffix:
        return [phrase]
    expanded = [
        part.strip() if i == len(parts) - 1 or part.lower().endswith(suffix.lower()) else f"{part.strip()} {suffix}"
        for i, part in enumerate(parts)
    ]
    return expanded + [phrase]
