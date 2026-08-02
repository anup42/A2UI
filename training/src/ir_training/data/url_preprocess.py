from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


_LOCAL_ASSET_PREFIX = (
    r"(?:[a-z]:[/\\]|/(?:data|sdcard|storage|mnt|android_asset)/|"
    r"\.{1,2}[/\\]|(?:assets?|media|images?|res|drawable|mipmap|raw)[/\\])"
)
_LOCAL_ASSET_TAIL = r"[^\s<>\"'|]*[-\w._~/#%+&=\\]"
_REFERENCE_RE = re.compile(
    rf"(?:(?P<quote>[\"'])(?P<quoted_local>{_LOCAL_ASSET_PREFIX}[^\"']*[-\w._~/#%+&=\\])(?P=quote)|"
    rf"(?P<reference>\b[a-z][a-z0-9+.-]*://[^\s<>\"']*[\w/#=&%+~_-]|"
    rf"\b(?:mailto|tel|geo|intent|genuicraft|data|javascript|blob|urn|sms|market):[^\s<>\"']*[\w/#=&%+~_-]|"
    rf"(?<![\w]){_LOCAL_ASSET_PREFIX}{_LOCAL_ASSET_TAIL}|@[a-z][a-z0-9_.-]*/[a-z0-9_.-]+))",
    re.IGNORECASE,
)
_LOCAL_ASSET_START_RE = re.compile(
    rf"^(?:{_LOCAL_ASSET_PREFIX}|@[a-z][a-z0-9_.-]*/)",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"\[(?:IMAGE_URL|ICON_URL|ACTION_URL|SOURCE_URL|MEDIA_URL|URL|IMAGE_ASSET|ICON_ASSET|MEDIA_ASSET)_\d+\]"
)
_TRAILING_PUNCT = ".,;:!?"
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif")
_ICON_EXTENSIONS = (".svg",)
_ACTION_KEYS = {"url", "href", "link", "actionurl", "bookingurl", "sourceurl"}
_IMAGE_KEYS = {"image", "imagesrc", "src", "source", "thumbnail", "photo", "url"}
_ICON_KEYS = {"icon", "name"}
_TEXT_KEYS = {"text", "label", "description", "subtitle", "title", "caption", "sourcetext"}


@dataclass(frozen=True)
class UrlPreprocessResult:
    response_text: str
    genui_json: Any
    url_map: dict[str, dict[str, str]]
    metrics: dict[str, int]


class _UrlRegistry:
    def __init__(self) -> None:
        self._by_role_url: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}
        self.url_map: dict[str, dict[str, str]] = {}
        self.response_url_count = 0
        self.target_url_count = 0
        self.raw_visible_text_url_count = 0

    def placeholder(self, raw_url: str, role: str) -> str:
        role = _normalize_role(role, raw_url)
        key = (role, raw_url)
        existing = self._by_role_url.get(key)
        if existing:
            return existing
        placeholder_prefix = _placeholder_prefix(role)
        self._counters[placeholder_prefix] = self._counters.get(placeholder_prefix, 0) + 1
        token = f"[{placeholder_prefix}_{self._counters[placeholder_prefix]}]"
        parsed = urlparse(raw_url if not _is_local_asset_reference(raw_url) else "")
        self._by_role_url[key] = token
        self.url_map[token] = {
            "url": raw_url,
            "role": role,
            "host": parsed.netloc.lower(),
            "kind": "local_asset" if _is_local_asset_reference(raw_url) else "url",
        }
        return token


def preprocess_training_urls(response_text: str, genui_json: Any, *, enabled: bool = True) -> UrlPreprocessResult:
    if not enabled:
        return UrlPreprocessResult(
            response_text=response_text,
            genui_json=copy.deepcopy(genui_json),
            url_map={},
            metrics={
                "url_placeholder_count": 0,
                "response_url_count": 0,
                "target_url_count": 0,
                "raw_visible_text_url_count": 0,
            },
        )
    registry = _UrlRegistry()
    processed_response = _replace_urls_in_text(response_text, registry, key=None, component_type=None, in_response=True)
    processed_genui = _replace_urls_in_value(genui_json, registry, key=None, component_type=None, action_context=False)
    return UrlPreprocessResult(
        response_text=processed_response,
        genui_json=processed_genui,
        url_map=registry.url_map,
        metrics={
            "url_placeholder_count": len(registry.url_map),
            "response_url_count": registry.response_url_count,
            "target_url_count": registry.target_url_count,
            "raw_visible_text_url_count": registry.raw_visible_text_url_count,
        },
    )


def restore_url_placeholders(value: Any, url_map: dict[str, Any] | None) -> Any:
    if not url_map:
        return value
    replacement = {
        str(token): str(entry.get("url") if isinstance(entry, dict) else entry)
        for token, entry in url_map.items()
        if token
    }

    def replace_text(text: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda match: replacement.get(match.group(0), match.group(0)), text)

    if isinstance(value, str):
        return replace_text(value)
    if isinstance(value, list):
        return [restore_url_placeholders(item, url_map) for item in value]
    if isinstance(value, dict):
        return {key: restore_url_placeholders(item, url_map) for key, item in value.items()}
    return value


def _replace_urls_in_value(
    value: Any,
    registry: _UrlRegistry,
    *,
    key: str | None,
    component_type: str | None,
    action_context: bool,
) -> Any:
    if isinstance(value, dict):
        current_type = str(value.get("type") or component_type or "")
        current_action = action_context or str(value.get("action") or "").lower() == "openurl"
        return {
            item_key: _replace_urls_in_value(
                item_value,
                registry,
                key=str(item_key),
                component_type=current_type,
                action_context=current_action,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_urls_in_value(item, registry, key=key, component_type=component_type, action_context=action_context)
            for item in value
        ]
    if isinstance(value, str):
        if key and key.lower() in _TEXT_KEYS:
            registry.raw_visible_text_url_count += len(_find_url_like_values(value))
        return _replace_urls_in_text(
            value,
            registry,
            key=key,
            component_type=component_type,
            action_context=action_context,
            in_response=False,
        )
    return value


def _replace_urls_in_text(
    text: str,
    registry: _UrlRegistry,
    *,
    key: str | None,
    component_type: str | None,
    action_context: bool = False,
    in_response: bool,
) -> str:
    stripped_text = text.strip()
    if "\n" not in text and "\r" not in text and _is_local_asset_reference(stripped_text):
        role = _classify_url(
            stripped_text,
            key=key,
            component_type=component_type,
            action_context=action_context,
            context=text,
        )
        if in_response:
            registry.response_url_count += 1
        else:
            registry.target_url_count += 1
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        return text[:start] + registry.placeholder(stripped_text, role) + text[end:]

    def replace_match(match: re.Match[str]) -> str:
        raw = match.group("quoted_local") or match.group("reference") or match.group(0)
        stripped, suffix = _strip_trailing_punct(raw)
        role = _classify_url(
            stripped,
            key=key,
            component_type=component_type,
            action_context=action_context,
            context=_line_context(text, match.start()),
        )
        if in_response:
            registry.response_url_count += 1
        else:
            registry.target_url_count += 1
        placeholder = registry.placeholder(stripped, role) + suffix
        quote = match.group("quote")
        return f"{quote}{placeholder}{quote}" if quote else placeholder

    return _REFERENCE_RE.sub(replace_match, text)


def _find_url_like_values(text: str) -> list[str]:
    return [
        match.group("quoted_local") or match.group("reference") or match.group(0)
        for match in _REFERENCE_RE.finditer(text)
    ]


def _classify_url(
    url: str,
    *,
    key: str | None,
    component_type: str | None,
    action_context: bool,
    context: str,
) -> str:
    key_l = (key or "").lower()
    type_l = (component_type or "").lower()
    context_l = context.lower()
    url_l = url.lower()
    url_norm = url_l.replace("\\", "/")
    if _is_local_asset_reference(url):
        if (
            url_norm.endswith(_ICON_EXTENSIONS)
            or key_l in _ICON_KEYS
            or type_l == "icon"
            or "media: icon" in context_l
            or re.search(r"\bicons?\s*:", context_l)
        ):
            return "ICON_ASSET"
        if (
            url_norm.endswith(_IMAGE_EXTENSIONS)
            or type_l == "image"
            or "media: image" in context_l
            or re.search(r"\bimages?\s*:", context_l)
        ):
            return "IMAGE_ASSET"
        return "MEDIA_ASSET"
    if action_context or key_l in {"bookingurl", "actionurl", "href", "link"}:
        return "ACTION"
    if type_l == "image" or (key_l in _IMAGE_KEYS and _looks_like_image(url_l)):
        return "IMAGE"
    if type_l == "icon" or key_l in _ICON_KEYS or _looks_like_icon(url_l):
        return "ICON"
    if "media: image" in context_l or re.search(r"\bimages?\s*:", context_l):
        return "IMAGE"
    if "media: icon" in context_l or re.search(r"\bicons?\s*:", context_l):
        return "ICON"
    if "button:" in context_l or "action:" in context_l or "quick action" in context_l:
        return "ACTION"
    if "source" in context_l or "reference" in context_l or key_l in {"source", "sourceurl"}:
        return "SOURCE"
    if key_l in _ACTION_KEYS:
        return "ACTION"
    if _looks_like_image(url_l):
        return "IMAGE"
    if _looks_like_icon(url_l):
        return "ICON"
    return "SOURCE"


def _normalize_role(role: str, url: str) -> str:
    role = role.upper()
    if role in {"IMAGE", "ICON", "ACTION", "SOURCE", "MEDIA", "URL", "IMAGE_ASSET", "ICON_ASSET", "MEDIA_ASSET"}:
        return role
    if not _is_local_asset_reference(url):
        return "URL"
    return "MEDIA_ASSET"


def _placeholder_prefix(role: str) -> str:
    if role in {"IMAGE", "ICON", "ACTION", "SOURCE", "MEDIA"}:
        return f"{role}_URL"
    return role


def _looks_like_image(url_l: str) -> bool:
    return url_l.endswith(_IMAGE_EXTENSIONS) or "upload.wikimedia.org/" in url_l or "googleusercontent.com/" in url_l


def _looks_like_icon(url_l: str) -> bool:
    return url_l.endswith(_ICON_EXTENSIONS) or "/icons/" in url_l or "cdn.jsdelivr.net/npm/bootstrap-icons" in url_l


def _is_local_asset_reference(value: str) -> bool:
    return bool(_LOCAL_ASSET_START_RE.match(value.strip()))


def _strip_trailing_punct(raw: str) -> tuple[str, str]:
    suffix = ""
    while raw and raw[-1] in _TRAILING_PUNCT:
        suffix = raw[-1] + suffix
        raw = raw[:-1]
    return raw, suffix


def _line_context(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end < 0:
        end = len(text)
    return text[start:end]
