from __future__ import annotations

import json
import math
import os
import time
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request

from pipeline.cache import PromptCache
from pipeline.common import load_prompt, render_prompt
from pipeline.image_resolver import repair_canonical_graph_images
from pipeline.ir_formats import (
    A2UI_EXPRESS_V1,
    codec_identity,
    decode_express_completion,
    encode_express_completion,
    compile_express_to_wire,
    semantic_hash,
    serialized_text,
)
from pipeline.genui_quality import (
    SourceContractCacheV5_1,
    SourceContractCacheV5_2,
    SourceContractCacheV5_3,
    SourceContractCacheV5_4,
    breakdown_to_mapping,
    load_default_reward_config,
    load_v5_1_reward_config,
    load_v5_reward_config,
    load_v4_reward_config,
    load_v5_3_reward_config,
    load_v5_4_reward_config,
    normalize_metric_mode,
    resolve_expected_ui_contract_v5_1,
    resolve_expected_ui_contract_v5_2,
    resolve_expected_ui_contract_v5_3,
    resolve_expected_ui_contract_v5_4,
    generation_reward_a2ui_express_v1,
    render_artifact_quality_v5_4,
    score_genui_completion,
    score_genui_completion_v5_1,
    score_genui_completion_v5_2,
    score_genui_completion_v5_0,
    score_genui_completion_v4,
)
from pipeline.metrics import (
    content_coverage,
    dup_rate,
    lint_score,
    lexical_token_estimate,
    count_characters,
    aggregate_metrics,
    compute_overall_score,
    compute_media_score,
    compute_ui_metrics,
    compute_intent_metrics,
)
from pipeline.storage import JsonlWriter, iter_jsonl
from pipeline.toon_convert import encode_toon, roundtrip_ok
from llm.base import BaseLLMAdapter, LLMRateLimitError
from llm.http_transport import urlopen
from utils.hashing import hash_text
from utils.rate_limit import RateLimiter
from utils.retry import with_retry


# URLs and machine-local asset paths are data, not model instructions.  Stage 3
# replaces them with deterministic role placeholders before constructing a
# provider prompt and restores them only after strict Express parsing.
_MODEL_REFERENCE_RE = re.compile(
    r"(?:"
    r"(?P<quote>[\"'])(?P<quoted_local>(?:[A-Za-z]:[/\\]|/(?:data|sdcard|storage|mnt|android_asset)/|\.\.?[/\\]|(?:assets?|media|images?|res|drawable|mipmap|raw)[/\\]|@[a-z][a-z0-9_.-]*/)[^\"']+)(?P=quote)|"
    r"\b(?:https?|ftp)://[^\s<>\"']+|\b(?:mailto|tel|geo|intent|genuicraft|data|javascript|blob|urn|sms|market):[^\s<>\"']+|"
    r"(?<![\w])(?:[A-Za-z]:[/\\]|/(?:data|sdcard|storage|mnt|android_asset)/|\.\.?[/\\]|(?:assets?|media|images?|res|drawable|mipmap|raw)[/\\]|@[a-z][a-z0-9_.-]*/)[^\s<>\"']+"
    r")",
    re.IGNORECASE,
)
_MODEL_REFERENCE_TRAILING = ".,;:!?)]}"
_MODEL_PLACEHOLDER_RE = re.compile(
    r"\[(?:IMAGE_URL|ICON_URL|MEDIA_URL|ACTION_URL|SOURCE_URL|URL|IMAGE_ASSET|ICON_ASSET|MEDIA_ASSET)_\d+\]"
)


def _validate_schema(schema: dict[str, Any], data: Any, schema_dir: Path) -> tuple[bool, list[str], bool]:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return False, ["jsonschema not installed"], False
    store: dict[str, Any] = {}
    for schema_path in schema_dir.glob("*.json"):
        try:
            content = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        store[schema_path.name] = content
        schema_id = content.get("$id")
        if schema_id:
            store[schema_id] = content
        # Map known spec URLs to local copies when filenames are present.
        if schema_path.name == "catalog.json":
            store["https://genui.local/specification/v0_9/catalog.json"] = content
        if schema_path.name == "common_types.json":
            store["https://genui.local/specification/v0_9/common_types.json"] = content
        if schema_path.name == "server_to_client.json":
            store["https://genui.local/specification/v0_9/server_to_client.json"] = content
        if schema_path.name == "server_to_client_list.json":
            store["https://genui.local/specification/v0_9/server_to_client_list.json"] = content

    # Prefer the modern referencing registry to avoid network fetches.
    try:
        from referencing import Registry, Resource  # type: ignore

        registry = Registry()
        for key, value in store.items():
            registry = registry.with_resource(key, Resource.from_contents(value))
        validator = jsonschema.Draft202012Validator(schema, registry=registry)
        validator.validate(instance=data)
        return True, [], True
    except Exception:
        # Fall back to the legacy resolver API.
        try:
            resolver = jsonschema.RefResolver.from_schema(schema, store=store)
            validator = jsonschema.Draft202012Validator(schema, resolver=resolver)
            validator.validate(instance=data)
            return True, [], True
        except Exception as exc:
            return False, [str(exc)], True


def _is_flat_spec_schema(schema: dict[str, Any]) -> bool:
    if not isinstance(schema, dict):
        return False
    if schema.get("type") != "object":
        return False
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return False
    return "root" in properties and "elements" in properties


_MEDIA_ONLY_HEADINGS = {
    "images",
    "icons",
    "assets",
    "files",
    "gallery",
    "visual guide",
    "key feature icons",
    "trip imagery",
    "weather icons",
    "related icons",
}


def _meaningful_response_heading_count(response_text: str) -> int:
    count = 0
    for raw_line in (response_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            continue
        if re.match(r"^[-*]\s+", line):
            continue
        line = re.sub(r"^#{1,6}\s+", "", line).strip()
        line = line[:-1].strip() if line.endswith(":") else line
        if not line:
            continue
        normalized = re.sub(r"\s+", " ", line).lower()
        if normalized in _MEDIA_ONLY_HEADINGS:
            continue
        if raw_line.lstrip().startswith("#"):
            count += 1
            continue
        if 3 <= len(line) <= 90 and not re.search(r"[.!?]$", line):
            words = re.findall(r"[A-Za-z0-9]+", line)
            if 1 <= len(words) <= 10:
                titleish = sum(1 for word in words if word[:1].isupper() or word.isdigit())
                if titleish >= max(1, len(words) // 2):
                    count += 1
    return count


def _source_table_cell_count(response_text: str) -> int:
    cells = 0
    for raw_line in (response_text or "").splitlines():
        line = raw_line.strip()
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        meaningful = [part for part in parts if part and not re.fullmatch(r"[-:\s]+", part)]
        if len(meaningful) >= 2:
            cells += len(meaningful)
    return cells


def _json_pointer_get(root: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return None
    current = root
    for raw_part in pointer.strip("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if 0 <= index < len(current) else None
        else:
            return None
    return current


def _generated_table_cell_count(genui_json: dict[str, Any]) -> int:
    elements = genui_json.get("elements")
    if not isinstance(elements, dict):
        return 0
    state = genui_json.get("state") if isinstance(genui_json.get("state"), dict) else {}
    total = 0
    for element in elements.values():
        if not isinstance(element, dict) or str(element.get("type", "")).lower() != "table":
            continue
        props = element.get("props") if isinstance(element.get("props"), dict) else {}
        columns = props.get("columns") if isinstance(props.get("columns"), list) else []
        rows = props.get("rows")
        if not isinstance(rows, list):
            rows = _json_pointer_get(state, props.get("statePath")) if isinstance(props.get("statePath"), str) else []
        if not isinstance(rows, list):
            continue
        if columns:
            total += len(rows) * len(columns)
        else:
            total += sum(len(row) for row in rows if isinstance(row, dict))
    return total


def _has_substantial_compact_table(genui_json: dict[str, Any]) -> bool:
    """A compact Table plus title can be valid high-quality IR with few elements."""
    elements = genui_json.get("elements")
    if not isinstance(elements, dict):
        return False
    if not any(
        isinstance(element, dict) and str(element.get("type", "")).lower() == "table"
        for element in elements.values()
    ):
        return False
    return _generated_table_cell_count(genui_json) >= 6


def _stage3_quality_warnings(
    response_text: str,
    genui_json: Any,
    metrics: dict[str, Any],
) -> list[str]:
    if not isinstance(genui_json, dict):
        return []
    elements = genui_json.get("elements")
    if not isinstance(elements, dict):
        return []

    generated_heading_count = 0
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        element_type = str(element.get("type", "")).lower()
        if element_type == "text":
            props = element.get("props") if isinstance(element.get("props"), dict) else {}
            variant = str(props.get("variant", "")).lower()
            if variant in {"h2", "h3"}:
                generated_heading_count += 1

    warnings: list[str] = []
    source_heading_count = _meaningful_response_heading_count(response_text)
    if source_heading_count >= 3 and generated_heading_count < max(2, source_heading_count // 2):
        warnings.append(
            "low_heading_preservation: "
            f"source_headings={source_heading_count} generated_h2_h3={generated_heading_count}"
        )

    source_cells = _source_table_cell_count(response_text)
    generated_cells = _generated_table_cell_count(genui_json)
    if source_cells >= 8 and generated_cells < int(source_cells * 0.8):
        warnings.append(
            f"table_cell_loss_risk: source_cells={source_cells} generated_table_cells={generated_cells}"
        )

    if float(metrics.get("section_heading_coverage", 1.0) or 0.0) < 0.25 and source_heading_count >= 2:
        warnings.append(
            "low_section_heading_coverage_metric: "
            f"value={float(metrics.get('section_heading_coverage', 0.0) or 0.0):.3f}"
        )

    return warnings


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _make_ui_id(
    query_id: str,
    n_idx: int,
    candidate_idx: int,
    source_format: str | None = None,
) -> str:
    suffix = query_id.replace("q_", "")
    if candidate_idx == 1:
        base = f"u_{suffix}_{n_idx:02d}"
    else:
        base = f"u_{suffix}_{n_idx:02d}_{candidate_idx:02d}"
    return base


def _normalize_stage3_ir_formats(values: Any) -> tuple[str, ...]:
    aliases = {"express": A2UI_EXPRESS_V1, "a2ui_express": A2UI_EXPRESS_V1}
    if values is None:
        raw = os.getenv("A2UI_STAGE3_IR_FORMATS", A2UI_EXPRESS_V1)
        values = [part for part in raw.split(",") if part.strip()]
    elif isinstance(values, str):
        values = [part for part in values.split(",") if part.strip()]
    elif not isinstance(values, (list, tuple)):
        values = [values]
    normalized: list[str] = []
    for value in values:
        token = str(value).strip().lower()
        resolved = aliases.get(token, token)
        if resolved != A2UI_EXPRESS_V1:
            raise ValueError(
                f"Unsupported Stage 3 IR format {value!r}; Stage 3 is Express-only"
            )
        if resolved not in normalized:
            normalized.append(resolved)
    if not normalized:
        raise ValueError("At least one Stage 3 IR format is required")
    return tuple(normalized)


def _native_stage3_paths(
    source_format: str,
    fallback_prompt: Path,
    fallback_schema: Path,
) -> tuple[Path, Path]:
    dataset_root = Path(__file__).resolve().parents[2]
    if source_format == A2UI_EXPRESS_V1:
        return (
            dataset_root / "prompts" / "genui_gen_mobile_a2ui_express_v1.md",
            dataset_root / "schema" / "canonical_ui_graph_v1.schema.json",
        )
    return fallback_prompt, fallback_schema


def _to_render_asset_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return normalized
    if normalized.startswith("../assets/"):
        return normalized
    normalized = normalized.lstrip("/")
    normalized = normalized.lstrip("./")
    if normalized.startswith("assets/"):
        return "../" + normalized
    return normalized


def _extract_declared_asset_entries(response_text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if not response_text:
        return entries

    section = ""
    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":"):
            header = line[:-1].strip().lower()
            if header in {"images", "icons", "assets", "files"}:
                section = header
            else:
                section = ""
            continue

        if section not in {"images", "icons", "assets", "files"}:
            continue
        # Asset sections are expected as bullet lists. If structure changes,
        # stop consuming so we do not accidentally capture unrelated URLs.
        if not re.match(r"^[-*•]\s+", line):
            section = ""
            continue

        kind = "asset"
        if section == "images":
            kind = "image"
        elif section == "icons":
            kind = "icon"

        url_match = re.search(r"https?://[^\s)]+", line)
        if not url_match:
            continue

        url = url_match.group(0).strip()
        label = line[: url_match.start()].strip(" -:\t")
        entries.append({"kind": kind, "label": label, "url": url})

    return entries


def _guess_ext(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.strip().lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tiff", ".avif"}:
        return suffix
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/avif": ".avif",
    }
    return mapping.get(ctype, ".bin")


def _safe_asset_stem(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return safe or "asset"


def _download_response_asset(
    response_id: str,
    asset_index: int,
    url: str,
    kind: str,
    assets_dir: Path,
) -> dict[str, Any] | None:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; GenUICraft-Stage3/1.0)",
            "Accept": "*/*",
        },
    )
    with urlopen(req, timeout=20) as resp:  # nosec B310 - controlled pipeline input
        body = resp.read(8 * 1024 * 1024 + 1)
        if len(body) == 0 or len(body) > 8 * 1024 * 1024:
            return None
        content_type = resp.headers.get("Content-Type", "")

    ext = _guess_ext(url, content_type)
    stem = _safe_asset_stem(Path(urlparse(url).path).stem or f"{kind}_{asset_index}")
    filename = f"{response_id}_{asset_index}_{stem}{ext}"
    dest = assets_dir / filename
    dest.write_bytes(body)
    return {
        "url": url,
        "path": str(dest.relative_to(assets_dir.parent)),
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
    }


def _auto_download_response_assets(
    response_id: str,
    response_text: str,
    assets_dir: Path,
    logger,
) -> list[dict[str, Any]]:
    entries = _extract_declared_asset_entries(response_text)
    if not entries:
        return []

    assets_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}

    for idx, item in enumerate(entries, start=1):
        url = str(item.get("url") or "").strip()
        kind = str(item.get("kind") or "asset")
        if not url:
            continue
        if url in seen:
            downloaded.append(seen[url])
            continue
        try:
            asset = _download_response_asset(response_id, idx, url, kind, assets_dir)
        except Exception as exc:
            logger.warning("Stage3 asset auto-download failed response_id=%s url=%s err=%s", response_id, url, exc)
            continue
        if not asset:
            continue
        seen[url] = asset
        downloaded.append(asset)

    return downloaded


def _asset_kind(path: str, url: str) -> str:
    ext = Path((path or url).split("?", 1)[0]).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp", ".tiff"}:
        return "image"
    if ext == ".svg" or "bootstrap-icons" in url.lower() or "/icons/" in url.lower():
        return "icon"
    if ext in {".pdf", ".zip"}:
        return "document"
    return "asset"


def _placeholder_prefix(kind: str, raw: str) -> str:
    lowered = raw.lower()
    if kind == "image" or Path(lowered.split("?", 1)[0]).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp", ".tiff"}:
        return "IMAGE_URL"
    if kind == "icon" or lowered.endswith(".svg"):
        return "ICON_URL"
    if kind == "document":
        return "MEDIA_URL"
    return "URL"


def _mask_model_references(
    response_text: str,
    assets: list[dict],
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Mask URLs/local paths and return (text, raw->placeholder, placeholder->raw)."""

    raw_to_placeholder: dict[str, str] = {}
    placeholder_to_raw: dict[str, str] = {}
    counters: dict[str, int] = {}

    def add(raw: str, kind: str) -> str:
        raw = str(raw or "").strip()
        if not raw:
            return raw
        existing = raw_to_placeholder.get(raw)
        if existing:
            return existing
        prefix = _placeholder_prefix(kind, raw)
        counters[prefix] = counters.get(prefix, 0) + 1
        token = f"[{prefix}_{counters[prefix]}]"
        raw_to_placeholder[raw] = token
        placeholder_to_raw[token] = raw
        return token

    # Seed the registry from downloaded/provided assets so URL and local path
    # variants resolve to the same placeholder and later restore deterministically.
    for item in assets:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        path = str(item.get("path") or "").strip()
        kind = _asset_kind(path, url)
        if url:
            add(url, kind)
        if path:
            path_token = add(path, kind)
            raw_to_placeholder[_to_render_asset_path(path)] = path_token

    def replace_match(match: re.Match[str]) -> str:
        quote = match.group("quote") or ""
        raw = match.group("quoted_local") or match.group(0)
        trailing = ""
        while raw and raw[-1] in _MODEL_REFERENCE_TRAILING:
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        if not raw:
            return match.group(0)
        token = raw_to_placeholder.get(raw)
        if token is None:
            token = add(raw, "asset" if _asset_kind("", raw) == "asset" else _asset_kind("", raw))
        return f"{quote}{token}{trailing}{quote}" if quote else token + trailing

    masked = _MODEL_REFERENCE_RE.sub(replace_match, str(response_text or ""))
    return masked, raw_to_placeholder, placeholder_to_raw


def _build_asset_context(assets: list[dict], raw_to_placeholder: dict[str, str]) -> str:
    if not assets:
        return ""

    lines: list[str] = []
    seen: set[str] = set()
    for item in assets:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        path = str(item.get("path") or "").strip()
        token = raw_to_placeholder.get(url) or raw_to_placeholder.get(path)
        if not token or token in seen:
            continue
        seen.add(token)
        kind = _asset_kind(path, url)
        lines.append(f"- [{kind}] {token} (preserve this placeholder exactly)")
    if not lines:
        return ""
    return (
        "Assets are represented only by opaque placeholders. Preserve the supplied "
        "placeholder in Image/Icon/media/action values; never emit a URL or local path:\n"
        + "\n".join(lines)
    )


def _restore_model_references(value: Any, placeholder_to_raw: dict[str, str]) -> Any:
    if not placeholder_to_raw:
        return value

    def restore(text: str) -> str:
        return _MODEL_PLACEHOLDER_RE.sub(
            lambda match: placeholder_to_raw.get(match.group(0), match.group(0)),
            text,
        )

    if isinstance(value, str):
        return restore(value)
    if isinstance(value, list):
        return [_restore_model_references(item, placeholder_to_raw) for item in value]
    if isinstance(value, dict):
        return {key: _restore_model_references(item, placeholder_to_raw) for key, item in value.items()}
    return value


def _apply_asset_replacements(text: str, assets: list[dict]) -> str:
    if not assets:
        return text
    valid_exts = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".bmp",
        ".tiff",
        ".pdf",
        ".zip",
    )
    updated = text
    for item in assets:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        path = str(item.get("path") or "").strip()
        if not url or not path:
            continue
        local_path = _to_render_asset_path(path)
        if not local_path.lower().endswith(valid_exts):
            continue
        updated = updated.replace(url, local_path)
    return updated


def _build_asset_rewrite_map(assets: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not assets:
        return mapping
    valid_exts = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".bmp",
        ".tiff",
        ".pdf",
        ".zip",
    )
    for item in assets:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        path = str(item.get("path") or "").strip()
        if not url or not path:
            continue
        local_path = _to_render_asset_path(path)
        if not local_path.lower().endswith(valid_exts):
            continue
        mapping[url] = local_path
    return mapping


def _rewrite_asset_urls_in_value(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_rewrite_asset_urls_in_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_asset_urls_in_value(item, mapping) for key, item in value.items()}
    return value


def _rewrite_genui_asset_urls(genui_json: Any, assets: list[dict]) -> Any:
    mapping = _build_asset_rewrite_map(assets)
    if not mapping:
        return genui_json
    return _rewrite_asset_urls_in_value(genui_json, mapping)


def _truncate_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    if max_tokens <= 0:
        return "", True
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text, False
    trimmed = " ".join(tokens[:max_tokens]).strip()
    if trimmed:
        trimmed = f"{trimmed}\n\n[TRUNCATED]"
    return trimmed, True


_LEADING_MARKDOWN_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*)$")
_BOLD_MARKDOWN_RE = re.compile(r"\*\*(.+?)\*\*")


def _normalize_text_markdown_for_ir(raw: str) -> tuple[str, str | None]:
    if not isinstance(raw, str):
        return "", None
    if not raw.strip():
        return raw, None

    heading_variant: str | None = None
    out_lines: list[str] = []
    for line in raw.splitlines():
        current = line.rstrip()

        heading_match = _LEADING_MARKDOWN_HEADING_RE.match(current)
        if heading_match:
            level = len(heading_match.group(1))
            if heading_variant is None:
                heading_variant = "h1" if level == 1 else ("h2" if level == 2 else "h3")
            current = heading_match.group(2).strip()
        elif current.lstrip().startswith("#"):
            # Code-snippet comments like "# Output:" should render as plain text,
            # not markdown headers.
            current = re.sub(r"^\s*#+\s*", "", current).strip()

        trimmed = current.lstrip()
        if trimmed.startswith("- "):
            current = f"{current[: len(current) - len(trimmed)]}• {trimmed[2:].strip()}"
        elif trimmed.startswith("* "):
            current = f"{current[: len(current) - len(trimmed)]}• {trimmed[2:].strip()}"

        if "|" in current and current.count("|") >= 2:
            pieces = [part.strip() for part in current.strip().strip("|").split("|")]
            pieces = [part for part in pieces if part]
            if len(pieces) >= 2:
                current = " • ".join(pieces)

        current = _BOLD_MARKDOWN_RE.sub(lambda m: m.group(1), current)
        current = current.replace("```", "").replace("'''", "").replace("`", "")
        out_lines.append(current)

    normalized = "\n".join(out_lines).strip()
    return normalized, heading_variant


def _normalize_canonical_graph_text_content(genui_json: Any) -> Any:
    if not isinstance(genui_json, dict):
        return genui_json
    elements = genui_json.get("elements")
    if not isinstance(elements, dict):
        return genui_json

    for element in elements.values():
        if not isinstance(element, dict):
            continue
        element_type = str(element.get("type") or "").strip().lower()
        if element_type != "text":
            continue
        props = element.get("props")
        if not isinstance(props, dict):
            continue
        variant = props.get("variant")
        normalized_variant = variant.strip().lower() if isinstance(variant, str) else ""

        for text_key in ("text", "title", "label", "content", "value"):
            value = props.get(text_key)
            if not isinstance(value, str):
                continue
            normalized_text, inferred_heading = _normalize_text_markdown_for_ir(value)
            props[text_key] = normalized_text
            if (
                inferred_heading
                and not normalized_variant
            ):
                props["variant"] = inferred_heading
                normalized_variant = inferred_heading

    return genui_json


# Explicit compatibility alias for historical offline scripts.  Stage 3
# calls the canonical-graph name after Express parsing.
_normalize_flat_spec_text_content = _normalize_canonical_graph_text_content

def _extract_prompt_version(template: str, prompt_path: Path) -> str:
    """Extract prompt version from first Markdown heading; fallback to filename stem."""
    try:
        first_line = template.splitlines()[0].lstrip("\ufeff").strip() if template else ""
    except Exception:
        first_line = ""
    match = re.match(r"^#\s*([A-Za-z0-9_.-]+)", first_line)
    if match:
        return match.group(1)
    return prompt_path.stem


def _prepare_prompt_context(
    template: str,
    adapter: BaseLLMAdapter,
    logger,
) -> tuple[str | None, str]:
    """Split Stage3 prompt into static system + small per-item user prompt."""
    provider = (adapter.spec.provider or "").lower()
    mode = (
        os.getenv("STAGE3_PROMPT_MODE")
        or os.getenv("GEMINI_STAGE3_PROMPT_MODE")
        or "system_prefix"
    ).strip().lower()
    if provider not in {"gemini", "azure_openai", "openai"} or mode in {"inline", "legacy", "off", "0", "false"}:
        return None, template

    placeholder = "{response_text}"
    if placeholder not in template:
        logger.warning(
            "Stage3 system-prefix mode requested but prompt has no %s; using inline mode.",
            placeholder,
        )
        return None, template

    before, after = template.split(placeholder, 1)
    system_prompt = (
        f"{before}[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]{after}".strip()
    )
    user_template = (
        "Convert the response text into a rich, lossless GenUICraft A2UI Express v1 program.\n"
        "Return ONLY one complete <a2ui>...</a2ui> block; never return JSON, FlatSpec, Compact IR, or prose.\n\n"
        "Response:\n{response_text}"
    )
    logger.info(
        "Stage3 prompt mode=system_prefix provider=%s system_tokens=%s user_template_tokens=%s",
        provider,
        lexical_token_estimate(system_prompt),
        lexical_token_estimate(user_template),
    )
    return system_prompt, user_template


def run_stage3(
    queries_path: Path | None,
    responses_path: Path,
    prompt_path: Path,
    adapter: BaseLLMAdapter,
    genui_path: Path,
    schema_path: Path,
    artifacts_dir: Path,
    candidates_per_response: int,
    max_repair_attempts: int,
    max_tokens: int,
    prompt_max_tokens: int | None,
    seed: int,
    rate_limiter: RateLimiter,
    cache: PromptCache,
    logger,
    batch_size: int = 100,
    max_total: int | None = None,
    max_attempts: int = 3,
    aggregates_path: Path | None = None,
    aggregate_weights: dict[str, float] | None = None,
    metric_version: str = "v5_4",
    ir_formats: list[str] | tuple[str, ...] | str | None = None,
    _active_ir_format: str | None = None,
) -> None:
    if _active_ir_format is None:
        resolved_formats = _normalize_stage3_ir_formats(ir_formats)
        if resolved_formats != (A2UI_EXPRESS_V1,):
            raise ValueError("Stage 3 accepts exactly one active format: a2ui_express_v1")
        active_ir_format = A2UI_EXPRESS_V1
    else:
        active_ir_format = _normalize_stage3_ir_formats((_active_ir_format,))[0]
    prompt_path, schema_path = _native_stage3_paths(
        active_ir_format,
        prompt_path,
        schema_path,
    )
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing Stage 3 prompt for {active_ir_format}: {prompt_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing Stage 3 schema for {active_ir_format}: {schema_path}")
    prompt_template = load_prompt(prompt_path)
    prompt_version = _extract_prompt_version(prompt_template, prompt_path)
    system_prompt, user_prompt_template = _prepare_prompt_context(prompt_template, adapter, logger)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    native_ir_mode = True
    express_mode = True
    # The schema is the canonical graph contract used after strict Express
    # parsing; it is never supplied to the model as a FlatSpec response schema.
    canonical_graph_mode = True
    logger.info(
        "Stage3 format=%s schema_mode=%s",
        active_ir_format,
        "native_ir" if native_ir_mode else ("canonical_graph" if canonical_graph_mode else "legacy_messages"),
    )
    generation_temperature = _env_float("A2UI_GENUI_TEMPERATURE", 0.2)
    repair_temperature = _env_float("A2UI_GENUI_REPAIR_TEMPERATURE", 0.2)
    final_regen_temperature = _env_float("A2UI_GENUI_FINAL_REGEN_TEMPERATURE", 0.1)
    logger.info(
        "Stage3 temperatures generation=%.3f repair=%.3f final_regen=%.3f",
        generation_temperature,
        repair_temperature,
        final_regen_temperature,
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metric_mode = normalize_metric_mode(metric_version)
    v4_config = load_v4_reward_config() if metric_mode in {"v4", "dual"} else None
    v5_2_config = (
        load_default_reward_config()
        if metric_mode in {"v5", "v5_2", "dual"}
        else None
    )
    v5_3_config = (
        load_v5_3_reward_config()
        if metric_mode in {"v5_3", "dual"}
        else None
    )
    v5_4_config = (
        load_v5_4_reward_config()
        if metric_mode in {"v5_4", "dual"}
        else None
    )
    v5_1_config = (
        load_v5_1_reward_config()
        if metric_mode in {"v5_1", "dual"}
        else None
    )
    v5_0_config = (
        load_v5_reward_config()
        if metric_mode in {"v5_0", "dual"}
        else None
    )
    contract_cache = SourceContractCacheV5_2(
        artifacts_dir / "genui_contract_cache_v5_2"
    )
    contract_cache_v5_3 = SourceContractCacheV5_3(
        artifacts_dir / "genui_contract_cache_v5_3"
    )
    contract_cache_v5_4 = SourceContractCacheV5_4(
        artifacts_dir / "genui_contract_cache_v5_4"
    )
    aggregate_every = max(1, _env_int("GENUI_STAGE3_AGGREGATE_EVERY", 250))
    logger.info(
        "Stage3 evaluation metric mode=%s aggregate_every=%s",
        metric_mode,
        aggregate_every,
    )

    prompt_token_multiplier = max(
        1.0,
        _env_float(
            "STAGE3_PROMPT_TOKEN_MULTIPLIER",
            _env_float("A2UI_STAGE3_PROMPT_TOKEN_MULTIPLIER", 1.65),
        ),
    )

    def _estimated_prompt_tokens(text: str) -> int:
        return int(math.ceil(lexical_token_estimate(text) * prompt_token_multiplier))

    def _resolve_context_limited_prompt_max() -> int | None:
        configured_prompt_max = int(prompt_max_tokens) if prompt_max_tokens else None
        if adapter.spec.provider != "local":
            return configured_prompt_max

        context_tokens = _env_int(
            "VLLM_MAX_MODEL_LEN",
            _env_int("LOCAL_VLLM_MAX_MODEL_LEN", _env_int("A2UI_VLLM_MAX_MODEL_LEN", 0)),
        )
        if context_tokens <= 0:
            return configured_prompt_max

        requested_output_tokens = int(max_tokens)
        local_output_cap = _env_int("LOCAL_VLLM_MAX_OUTPUT_TOKENS", 0)
        if local_output_cap > 0:
            requested_output_tokens = min(requested_output_tokens, local_output_cap)
        context_safety_tokens = max(
            0,
            _env_int(
                "STAGE3_CONTEXT_SAFETY_TOKENS",
                _env_int("A2UI_STAGE3_CONTEXT_SAFETY_TOKENS", 512),
            ),
        )
        context_prompt_budget = context_tokens - requested_output_tokens - context_safety_tokens
        if context_prompt_budget <= 0:
            logger.warning(
                "Stage3 local context budget is non-positive context=%s output=%s safety=%s; "
                "using configured prompt cap=%s",
                context_tokens,
                requested_output_tokens,
                context_safety_tokens,
                configured_prompt_max,
            )
            return configured_prompt_max

        respect_config_cap = os.environ.get(
            "STAGE3_RESPECT_CONFIG_PROMPT_MAX",
            os.environ.get("A2UI_STAGE3_RESPECT_CONFIG_PROMPT_MAX", "0"),
        ).strip().lower() in {"1", "true", "yes", "on"}
        effective = (
            min(configured_prompt_max, context_prompt_budget)
            if configured_prompt_max is not None and respect_config_cap
            else context_prompt_budget
        )
        logger.info(
            "Stage3 local prompt budget context=%s output=%s safety=%s multiplier=%.2f "
            "configured=%s respect_config=%s effective=%s",
            context_tokens,
            requested_output_tokens,
            context_safety_tokens,
            prompt_token_multiplier,
            configured_prompt_max,
            respect_config_cap,
            effective,
        )
        return effective

    effective_prompt_max_tokens = _resolve_context_limited_prompt_max()

    def _clip_prompt_to_effective_budget(prompt_text: str, response_id: str, label: str) -> str:
        if not effective_prompt_max_tokens:
            return prompt_text
        system_tokens_est = _estimated_prompt_tokens(system_prompt) if system_prompt else 0
        estimated_tokens = _estimated_prompt_tokens(prompt_text) + system_tokens_est
        if estimated_tokens <= effective_prompt_max_tokens:
            return prompt_text
        word_budget = max(
            200,
            int((effective_prompt_max_tokens - system_tokens_est) / prompt_token_multiplier),
        )
        clipped, truncated = _truncate_tokens(prompt_text, word_budget)
        if truncated:
            logger.warning(
                "Stage3 %s prompt clipped response_id=%s estimated_tokens=%s effective_budget=%s word_budget=%s",
                label,
                response_id,
                estimated_tokens,
                effective_prompt_max_tokens,
                word_budget,
            )
        return clipped

    intent_lookup: dict[str, dict[str, Any]] = {}
    if queries_path and queries_path.exists():
        for row in iter_jsonl(queries_path):
            query_id = row.get("query_id")
            if not query_id:
                continue
            intent_lookup[query_id] = {
                "intent": row.get("intent"),
                "tags": row.get("tags") if isinstance(row.get("tags"), list) else [],
                "query_text": row.get("query_text") if isinstance(row.get("query_text"), str) else "",
            }

    existing_ids = {row.get("ui_id") for row in iter_jsonl(genui_path)}
    existing_genui_count = len(existing_ids)
    writer = JsonlWriter(genui_path)
    response_text_by_id: dict[str, str] = {}
    auto_assets_dir = responses_path.parent / "assets"
    if responses_path.exists():
        for row in iter_jsonl(responses_path):
            response_id = row.get("response_id")
            response_text = row.get("response_text")
            if isinstance(response_id, str) and isinstance(response_text, str):
                response_text_by_id[response_id] = response_text
    gemini_parallel_workers = (
        max(1, int(os.getenv("GEMINI_STAGE3_PARALLEL_THREADS", "1")))
        if adapter.spec.provider == "gemini"
        else 1
    )

    def _write_aggregates(reason: str = "") -> None:
        if not aggregates_path:
            return
        try:
            rows = list(iter_jsonl(genui_path))
            if response_text_by_id:
                for row in rows:
                    if row.get("response_text"):
                        continue
                    response_id = row.get("response_id")
                    if isinstance(response_id, str):
                        backfill = response_text_by_id.get(response_id)
                        if isinstance(backfill, str):
                            row["response_text"] = backfill
            render_rows_by_ui_id: dict[str, dict[str, Any]] = {}
            # Device-native results load last and therefore override generic
            # render rows when both exist for the same sample.
            for render_log_path in (
                genui_path.parent / "render.jsonl",
                genui_path.parent / "native_render_checks.jsonl",
            ):
                if not render_log_path.exists():
                    continue
                for render_row in iter_jsonl(render_log_path):
                    ui_id = render_row.get("ui_id")
                    if isinstance(ui_id, str) and ui_id:
                        render_rows_by_ui_id[ui_id] = render_row
            aggregates = aggregate_metrics(
                rows,
                render_rows_by_ui_id=render_rows_by_ui_id,
                metric_version=metric_mode,
                v4_config=v4_config,
                v5_config=v5_2_config,
                v5_1_config=v5_1_config,
                v5_0_config=v5_0_config,
                v5_3_config=v5_3_config,
                v5_4_config=v5_4_config,
            )
            if metric_mode in {"legacy", "dual"}:
                legacy_score = compute_overall_score(
                    aggregates,
                    aggregate_weights or {},
                )
                aggregates["legacy_structural_richness_score"] = legacy_score
                # Compatibility field for one migration window.
                aggregates["overall_score"] = legacy_score
                aggregates["legacy_score_deprecation_date"] = "2026-10-01"
            aggregates["media_score"] = compute_media_score(aggregates)
            aggregates["aggregated_scope"] = "all_generated_samples"
            aggregates["aggregated_sample_count"] = len(rows)
            aggregates["updated_at"] = datetime.utcnow().isoformat() + "Z"
            tmp_path = aggregates_path.with_name(
                f".{aggregates_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            tmp_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
            tmp_path.replace(aggregates_path)
            if reason:
                logger.debug(
                    "Stage3 aggregates stored at %s (%s) aggregated_sample_count=%s",
                    aggregates_path,
                    reason,
                    len(rows),
                )
            else:
                logger.info(
                    "Stage3 aggregates stored at %s aggregated_sample_count=%s",
                    aggregates_path,
                    len(rows),
                )
        except Exception as exc:  # best-effort
            logger.warning("Stage3 aggregates failed: %s", exc)

    def _compute_sample_legacy_score(record: dict[str, Any]) -> float | None:
        try:
            sample_aggregate = aggregate_metrics([record], metric_version="legacy")
            return compute_overall_score(sample_aggregate, aggregate_weights or {})
        except Exception as exc:  # best-effort logging metric
            logger.debug("Stage3 sample score failed ui_id=%s: %s", record.get("ui_id"), exc)
            return None

    total_created = 0
    total_failed = 0
    pending: list[dict[str, Any]] = []
    use_parallel = adapter.spec.provider == "gemini" and gemini_parallel_workers > 1
    use_batch = hasattr(adapter, "generate_batch") and not use_parallel
    if batch_size <= 0:
        batch_size = 100
    if use_batch:
        logger.info(
            "Stage3 batch generation enabled provider=%s batch_size=%s",
            adapter.spec.provider,
            batch_size,
        )
    stop = False

    def _build_prompt_for(
        response_id: str,
        response_text: str,
        assets_list: list[dict],
        *,
        masked_response_text: str | None = None,
        raw_to_placeholder: dict[str, str] | None = None,
    ) -> str:
        masked_response_text = response_text if masked_response_text is None else masked_response_text
        raw_to_placeholder = raw_to_placeholder or {}
        asset_context = _build_asset_context(assets_list, raw_to_placeholder)
        if assets_list:
            asset_policy = (
                "Asset reference policy for this request:\n"
                "- Use only the opaque placeholders supplied in the Assets mapping.\n"
                "- Do not emit remote URLs or machine-local paths.\n"
                "- Preserve every placeholder exactly; the pipeline restores it after parsing."
            )
        else:
            asset_policy = (
                "Asset reference policy for this request:\n"
                "- URL-like and local asset references in the response are opaque placeholders.\n"
                "- Preserve supplied placeholders exactly and never emit a raw URL or local path."
            )

        response_with_policy = f"{masked_response_text}\n\n{asset_policy}"
        prompt_response_text = response_with_policy
        if asset_context:
            prompt_response_text = f"{response_with_policy}\n\n{asset_context}"

        prompt = render_prompt(user_prompt_template, response_text=prompt_response_text)
        if effective_prompt_max_tokens:
            system_tokens = lexical_token_estimate(system_prompt) if system_prompt else 0
            system_tokens_est = int(math.ceil(system_tokens * prompt_token_multiplier))
            prompt_tokens = _estimated_prompt_tokens(prompt) + system_tokens_est
            if prompt_tokens > effective_prompt_max_tokens:
                # First attempt: drop asset context to save tokens.
                prompt = render_prompt(user_prompt_template, response_text=response_with_policy)
                prompt_tokens = _estimated_prompt_tokens(prompt) + system_tokens_est
            if prompt_tokens > effective_prompt_max_tokens:
                base_prompt = render_prompt(user_prompt_template, response_text="")
                base_tokens = _estimated_prompt_tokens(base_prompt) + system_tokens_est
                budget = max(
                    200,
                    int((effective_prompt_max_tokens - base_tokens) / prompt_token_multiplier),
                )
                trimmed_text, truncated = _truncate_tokens(response_text, budget)
                if truncated:
                    logger.warning(
                        "Stage3 prompt truncated response_id=%s estimated_tokens=%s effective_budget=%s response_word_budget=%s",
                        response_id,
                        prompt_tokens,
                        effective_prompt_max_tokens,
                        budget,
                    )
                prompt = render_prompt(
                    user_prompt_template,
                    response_text=f"{trimmed_text}\n\n{asset_policy}",
                )
            prompt = _clip_prompt_to_effective_budget(prompt, response_id, "initial")
        return prompt

    def _parse_completion(text: str) -> tuple[Any, Any, bool]:
        """Return native payload, canonical graph, and legacy-conversion flag."""
        native_payload = text.strip()
        if not native_payload.startswith("<a2ui>") or not native_payload.endswith("</a2ui>"):
            raise ValueError("A2UI Express completion must contain exactly one complete sentinel block")
        return native_payload, decode_express_completion(native_payload), False

    def _validate_completion(native_payload: Any, canonical: Any) -> tuple[bool, list[str], bool]:
        valid, validation_errors, validator_ok = _validate_schema(schema, canonical, schema_path.parent)
        if not valid:
            return valid, validation_errors, validator_ok
        # Apply the production wire gate on every attempt, including repaired
        # and regenerated completions. Canonical props alone do not enforce
        # dynamicArray types such as Table.highlightColumns; accepting a repair
        # here previously exited the loop only to fail final compilation.
        try:
            compile_express_to_wire(native_payload)
        except Exception as exc:
            return False, [f"standard_a2ui_compile_error: {exc}"], True
        return True, [], validator_ok

    def _repair_instructions(raw_text: str, errors: list[str]) -> str:
        failure_reason = "; ".join(errors[-5:]) if errors else "format validation failed"
        shared_contract = prompt_template.replace(
            "{response_text}", "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]"
        ).strip()
        return (
            "The previous completion failed strict validation. Re-emit it using this generated "
            "pinned A2UI Express contract; preserve all source facts and interactions.\n\n"
            f"{shared_contract}\n\n"
            f"Errors: {failure_reason}\n\nOriginal:\n{raw_text}"
        )

    def _normalized_native_output(canonical: dict[str, Any]) -> Any:
        return encode_express_completion(canonical)

    def _append_format_rejection(
        task: dict[str, Any],
        *,
        raw_text: str,
        raw_payload: Any,
        errors: list[str],
        repair_attempts: int,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
        error: str | None,
    ) -> None:
        nonlocal total_created
        record = {
            "ui_id": task["ui_id"],
            "response_id": task["response_id"],
            "query_id": task["query_id"],
            "response_text": task["response_text"],
            "intent": task.get("intent"),
            "tags": task.get("tags") if isinstance(task.get("tags"), list) else [],
            "assets": task["assets_list"],
            "record_status": "format_rejected",
            "source_format": active_ir_format,
            "codec_identity": codec_identity(),
            "model_completion_raw": raw_text,
            "model_payload_raw": raw_payload,
            "validation": {
                "json_parse_ok": False,
                "schema_valid_strict": False,
                "schema_valid_lenient": False,
                "errors": [item[:500] for item in errors],
                "repair_attempts": repair_attempts,
                "repair_needed": repair_attempts > 0,
            },
            "format_metrics": {
                "characters": len(raw_text),
                "utf8_bytes": len(raw_text.encode("utf-8")),
                "estimated_tokens": lexical_token_estimate(raw_text),
                "completion_tokens": output_tokens if output_tokens > 0 else None,
                "token_measurement_source": "provider_reported" if output_tokens > 0 else "lexical_diagnostic",
                "reported_output_tokens": output_tokens,
                "latency_ms": latency_ms,
            },
            "gen": {
                "provider": provider,
                "model": model,
                "prompt_version": prompt_version,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": None,
                "error": error,
            },
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        writer.append(record)
        existing_ids.add(task["ui_id"])
        total_created += 1
        error_path = artifacts_dir / f"error_{task['ui_id']}.json"
        error_path.write_text(
            json.dumps(
                {"prompt": task["prompt"], "raw_text": raw_text, "errors": errors},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _process_generated(
        task: dict[str, Any],
        raw_text: str,
        raw_payload: Any,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
        error: str | None,
        reasoning_text: str | None = None,
        reasoning_source: str | None = None,
        reasoning_tokens: int | None = None,
    ) -> None:
        nonlocal total_created
        ui_id = task["ui_id"]
        response_id = task["response_id"]
        query_id = task["query_id"]
        response_text = task["response_text"]
        assets_list = task["assets_list"]
        intent_value = task.get("intent")
        tags_value = task.get("tags") if isinstance(task.get("tags"), list) else []
        query_text = task.get("query_text") if isinstance(task.get("query_text"), str) else ""
        prompt = task["prompt"]
        accepted_reasoning_text = reasoning_text.strip() if isinstance(reasoning_text, str) else None
        accepted_reasoning_source = reasoning_source
        accepted_reasoning_tokens = reasoning_tokens
        accepted_reasoning_attempt = "initial" if accepted_reasoning_text else None

        parsed_ok = True
        errors: list[str] = []
        contract_resolution = resolve_expected_ui_contract_v5_2(
            response_text,
            intent=intent_value,
            assets=assets_list,
            persisted=(
                task.get("expected_ui_contract")
                if isinstance(task.get("expected_ui_contract"), dict)
                else None
            ),
            persisted_source=(
                str(task.get("expected_ui_contract_source"))
                if task.get("expected_ui_contract_source")
                else None
            ),
            cache=contract_cache,
        )
        errors.extend(contract_resolution.errors)
        contract_resolution_v5_3 = (
            resolve_expected_ui_contract_v5_3(
                response_text,
                intent=intent_value,
                assets=assets_list,
                persisted=(
                    task.get("expected_ui_contract_v5_3")
                    if isinstance(
                        task.get("expected_ui_contract_v5_3"), dict
                    )
                    else None
                ),
                persisted_source=(
                    str(task.get("expected_ui_contract_v5_3_source"))
                    if task.get("expected_ui_contract_v5_3_source")
                    else None
                ),
                cache=contract_cache_v5_3,
            )
            if metric_mode in {"v5_3", "dual"}
            else None
        )
        if contract_resolution_v5_3 is not None:
            errors.extend(contract_resolution_v5_3.errors)
        contract_resolution_v5_4 = (
            resolve_expected_ui_contract_v5_4(
                response_text,
                intent=intent_value,
                assets=assets_list,
                persisted=(
                    task.get("expected_ui_contract_v5_4")
                    if isinstance(
                        task.get("expected_ui_contract_v5_4"), dict
                    )
                    else None
                ),
                persisted_source=(
                    str(task.get("expected_ui_contract_v5_4_source"))
                    if task.get("expected_ui_contract_v5_4_source")
                    else None
                ),
                cache=contract_cache_v5_4,
            )
            if metric_mode in {"v5_4", "dual"}
            else None
        )
        if contract_resolution_v5_4 is not None:
            errors.extend(contract_resolution_v5_4.errors)
        converted_from_legacy = False
        parsed_native_payload: Any = None
        try:
            parsed_native_payload, genui_json, converted_from_legacy = _parse_completion(raw_text)
        except Exception as exc:
            parsed_ok = False
            genui_json = None
            errors.append(f"{active_ir_format}_parse_error: {exc}")

        schema_valid_strict = False
        schema_valid_lenient = False
        initial_native_syntax_valid = False
        initial_native_catalog_valid = False
        initial_standard_a2ui_valid = False
        repair_needed = False

        if parsed_ok and genui_json is not None:
            initial_native_syntax_valid = True
            schema_valid_strict, schema_errors, validator_ok = _validate_completion(
                parsed_native_payload, genui_json
            )
            # Keep catalog and production-wire diagnostics separate even
            # though the acceptance/repair gate now requires both.
            initial_native_catalog_valid, _, _ = _validate_schema(schema, genui_json, schema_path.parent)
            initial_standard_a2ui_valid = schema_valid_strict
            if schema_valid_strict:
                schema_valid_lenient = True
            else:
                errors.extend(schema_errors)
                if not validator_ok:
                    schema_valid_lenient = True

        repair_attempts = 0
        while (not parsed_ok or not schema_valid_strict) and repair_attempts < max_repair_attempts:
            repair_attempts += 1
            repair_needed = True
            repaired_text = _repair_instructions(raw_text, errors)
            repaired_text = _clip_prompt_to_effective_budget(
                repaired_text,
                str(response_id),
                "repair",
            )

            def _repair_call():
                rate_limiter.acquire()
                return adapter.generate(
                    prompt=repaired_text,
                    system=system_prompt,
                    temperature=repair_temperature,
                    max_tokens=max_tokens,
                    seed=seed + 100 + repair_attempts,
                    json_mode=(
                        True
                        if adapter.spec.supports_json_mode and not express_mode
                        else False
                    ),
                )

            try:
                result = with_retry(_repair_call, max_attempts=max_attempts)
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    logger.error(
                        "Stage3 rate limit info: limits=%s headers=%s",
                        exc.limits or "unset",
                        exc.headers or "none",
                    )
                errors.append(f"repair_exception: {exc}")
                logger.warning(
                    "Stage3 repair failed ui_id=%s response_id=%s: %s",
                    ui_id,
                    response_id,
                    exc,
                )
                break
            if result.error:
                errors.append(f"repair_error: {result.error}")
                logger.warning(
                    "Stage3 repair returned error ui_id=%s response_id=%s: %s",
                    ui_id,
                    response_id,
                    result.error,
                )
                break
            raw_text = result.text
            accepted_reasoning_text = (
                result.reasoning_text.strip()
                if isinstance(result.reasoning_text, str) and result.reasoning_text.strip()
                else None
            )
            accepted_reasoning_source = result.reasoning_source
            accepted_reasoning_tokens = result.reasoning_tokens
            accepted_reasoning_attempt = (
                f"repair_{repair_attempts}" if accepted_reasoning_text else None
            )
            try:
                parsed_native_payload, genui_json, converted_from_legacy = _parse_completion(raw_text)
                parsed_ok = True
            except Exception as exc:
                parsed_ok = False
                genui_json = None
                errors.append(f"repair_{active_ir_format}_parse_error: {exc}")
                continue

            schema_valid_strict, schema_errors, validator_ok = _validate_completion(
                parsed_native_payload, genui_json
            )
            if schema_valid_strict:
                schema_valid_lenient = True
            else:
                errors.extend(schema_errors)
                if not validator_ok:
                    schema_valid_lenient = True

        final_regen_attempts = max(0, int(os.getenv("STAGE3_FINAL_REGEN_ATTEMPTS", "3")))
        regen_attempt = 0
        while (
            (not parsed_ok or genui_json is None or not schema_valid_strict)
            and canonical_graph_mode
            and regen_attempt < final_regen_attempts
        ):
            regen_attempt += 1
            repair_needed = True
            output_instruction = "Return ONLY one valid complete <a2ui>...</a2ui> Express block."
            regen_instructions = (
                "Previous output was invalid or incomplete.\n"
                f"Regenerate the full {active_ir_format} UI from the source response.\n"
                f"{output_instruction}\n"
                "Preserve the source's useful content and use a meaningful multi-component "
                "composition; do not replace it with a minimal fallback.\n"
                "Keep the representation compact and avoid literal markdown markers in text fields."
            )
            regeneration_prompt = f"{regen_instructions}\n\nSource prompt:\n{prompt}"
            regeneration_prompt = _clip_prompt_to_effective_budget(
                regeneration_prompt,
                str(response_id),
                "final_regen",
            )

            def _regen_call():
                rate_limiter.acquire()
                return adapter.generate(
                    prompt=regeneration_prompt,
                    system=system_prompt,
                    temperature=final_regen_temperature,
                    max_tokens=max_tokens,
                    seed=seed + 900 + regen_attempt,
                    json_mode=(
                        True
                        if adapter.spec.supports_json_mode and not express_mode
                        else False
                    ),
                )

            try:
                regen_result = with_retry(_regen_call, max_attempts=max_attempts)
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    logger.error(
                        "Stage3 final regen rate limit info: limits=%s headers=%s",
                        exc.limits or "unset",
                        exc.headers or "none",
                    )
                errors.append(f"final_regen_exception: {exc}")
                break

            if regen_result.error:
                errors.append(f"final_regen_error: {regen_result.error}")
                break

            raw_text = regen_result.text
            accepted_reasoning_text = (
                regen_result.reasoning_text.strip()
                if isinstance(regen_result.reasoning_text, str) and regen_result.reasoning_text.strip()
                else None
            )
            accepted_reasoning_source = regen_result.reasoning_source
            accepted_reasoning_tokens = regen_result.reasoning_tokens
            accepted_reasoning_attempt = (
                f"final_regen_{regen_attempt}" if accepted_reasoning_text else None
            )
            try:
                parsed_native_payload, genui_json, converted_from_legacy = _parse_completion(raw_text)
                parsed_ok = True
            except Exception as exc:
                parsed_ok = False
                genui_json = None
                errors.append(f"final_regen_{active_ir_format}_parse_error: {exc}")
                continue

            schema_valid_strict, schema_errors, validator_ok = _validate_completion(
                parsed_native_payload, genui_json
            )
            if schema_valid_strict:
                schema_valid_lenient = True
            else:
                errors.extend(schema_errors)
                if not validator_ok:
                    schema_valid_lenient = True

        # Reject specs that are too simple (fewer than 5 elements) -- treat as needing fallback.
        if (
            parsed_ok
            and genui_json is not None
            and canonical_graph_mode
            and isinstance(genui_json.get("elements"), dict)
            and len(genui_json["elements"]) < 5
            and not _has_substantial_compact_table(genui_json)
        ):
            errors.append("spec_too_simple: fewer than 5 elements")
            parsed_ok = False

        if not parsed_ok or genui_json is None or not schema_valid_strict:
            _append_format_rejection(
                task,
                raw_text=raw_text,
                raw_payload=parsed_native_payload,
                errors=errors,
                repair_attempts=repair_attempts + regen_attempt,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider=provider,
                model=model,
                error=error,
            )
            return

        generation_completion = raw_text
        # Every path reaching acceptance has passed the same wire gate,
        # including final regeneration when no repair attempts were enabled.
        final_standard_a2ui_valid = True
        # Restore only after the raw Express completion has parsed and passed
        # catalog/schema checks.  Model-facing prompts never contain these
        # raw URL or local-path values.
        genui_json = _restore_model_references(
            genui_json,
            task.get("asset_placeholder_map")
            if isinstance(task.get("asset_placeholder_map"), dict)
            else {},
        )
        genui_json = _rewrite_genui_asset_urls(genui_json, assets_list)
        if canonical_graph_mode:
            genui_json = _normalize_canonical_graph_text_content(genui_json)
            genui_json, resolved_images = repair_canonical_graph_images(
                genui_json,
                query_text,
                response_text,
            )
            if resolved_images:
                errors.append(f"dataset_image_resolver_added={resolved_images}")

        # The model-facing completion remains the raw/normalized Express text;
        # the graph and compiled wire payload are explicit post-parse artifacts.
        normalized_native_output = _normalized_native_output(genui_json)
        normalized_native_text = serialized_text(normalized_native_output)
        try:
            compiled_a2ui = compile_express_to_wire(genui_json)
        except Exception as exc:
            # A single malformed model completion must not terminate the whole
            # Stage 3 run.  Preserve the raw payload in the error artifact and
            # let the caller continue with the remaining responses; a later
            # resume can retry this ui_id with a fresh generation.
            compile_error = f"standard_a2ui_compile_error: {exc}"
            logger.warning(
                "Stage3 skipping invalid ui_id=%s after final compile failure: %s",
                ui_id,
                compile_error,
            )
            _record_generation_error(task, compile_error, genui_json)
            return

        toon = encode_toon(genui_json)
        toon_ok = roundtrip_ok(genui_json, toon)

        json_text = json.dumps(genui_json, ensure_ascii=False)
        metrics = {
            "content_coverage": content_coverage(response_text, genui_json),
            "dup_rate": dup_rate(genui_json),
            "lint_score": lint_score(genui_json),
            # Size proxy: character-count based for stable JSON vs TOON comparison.
            "output_tokens_toon": count_characters(toon),
            "output_tokens_json": count_characters(json_text),
            "output_chars_toon": count_characters(toon),
            "output_chars_json": count_characters(json_text),
        }
        metrics.update(compute_ui_metrics(response_text, genui_json))
        intent_metrics = compute_intent_metrics(intent_value, tags_value, response_text, metrics)
        intent_bucket = intent_metrics.pop("intent_bucket", "unknown")
        metrics.update(intent_metrics)
        quality_warnings = (
            _stage3_quality_warnings(response_text, genui_json, metrics)
            if canonical_graph_mode
            else []
        )
        if quality_warnings:
            logger.info("Stage3 quality warnings ui_id=%s: %s", ui_id, quality_warnings)

        short_errors = [e[:300] + ("..." if len(e) > 300 else "") for e in errors]
        record = {
            "ui_id": ui_id,
            "response_id": response_id,
            "query_id": query_id,
            "intent": intent_value,
            "tags": tags_value,
            "intent_bucket": intent_bucket,
            "response_text": response_text,
            "expected_ui_contract": contract_resolution.contract,
            "expected_ui_contract_source": contract_resolution.source,
            "expected_ui_contract_version": contract_resolution.contract.get("contract_version"),
            "expected_ui_contract_cache_hit": contract_resolution.cache_hit,
            "record_status": "accepted",
            "source_format": active_ir_format,
            "target_format": A2UI_EXPRESS_V1,
            "codec_identity": codec_identity(),
            "semantic_hash": semantic_hash(genui_json),
            "canonical_graph_hash": semantic_hash(genui_json),
            "model_completion_raw": generation_completion,
            "model_payload_raw": parsed_native_payload,
            "model_native_output_normalized": normalized_native_output,
            "a2ui_express": generation_completion,
            "completion": generation_completion,
            "canonical_graph": genui_json,
            "compiled_a2ui": compiled_a2ui,
            "assets": assets_list,
            "toon": toon,
            "validation": {
                "json_parse_ok": parsed_ok,
                "native_syntax_valid": initial_native_syntax_valid,
                "native_catalog_valid": initial_native_catalog_valid,
                "raw_schema_valid_strict": bool(initial_native_catalog_valid),
                "raw_standard_a2ui_valid": bool(initial_standard_a2ui_valid),
                "repaired_syntax_valid": bool(repair_attempts > 0 and parsed_ok),
                "repaired_catalog_valid": bool(repair_attempts > 0 and schema_valid_strict),
                "repaired_standard_a2ui_valid": bool(repair_attempts > 0 and final_standard_a2ui_valid),
                "canonical_semantic_valid": True,
                "standard_a2ui_valid": final_standard_a2ui_valid,
                "repair_applied": bool(repair_attempts > 0),
                "schema_valid_strict": schema_valid_strict,
                "schema_valid_lenient": schema_valid_lenient,
                "toon_roundtrip_ok": toon_ok,
                "converted_from_legacy": converted_from_legacy,
                "errors": short_errors,
                "warnings": quality_warnings,
                "repair_attempts": repair_attempts,
                "repair_needed": repair_needed,
            },
            "metrics": metrics,
            "format_metrics": {
                "characters": len(normalized_native_text),
                "utf8_bytes": len(normalized_native_text.encode("utf-8")),
            "estimated_tokens": lexical_token_estimate(normalized_native_text),
                "completion_tokens": output_tokens if output_tokens > 0 else None,
                "token_measurement_source": "provider_reported" if output_tokens > 0 else "lexical_diagnostic",
                "reported_output_tokens": output_tokens,
                "latency_ms": latency_ms,
            },
            "gen": {
                "provider": provider,
                "model": model,
                "prompt_version": prompt_version,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": None,
                "error": error,
            },
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        record["evaluation_metric_mode"] = metric_mode
        record["renderer_check_result"] = {
            "adapter": "android_native",
            "attempted": False,
            "ok": None,
            "source": "stage3_not_attempted",
        }
        if accepted_reasoning_text:
            record["reasoning_text"] = accepted_reasoning_text
            record["gen"]["reasoning_available"] = True
            record["gen"]["reasoning_source"] = accepted_reasoning_source
            record["gen"]["reasoning_tokens"] = accepted_reasoning_tokens
            record["gen"]["reasoning_attempt"] = accepted_reasoning_attempt
        if metric_mode in {"v4", "dual"}:
            v4_result = score_genui_completion_v4(
                genui_json,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=contract_resolution.contract,
                render_ok=None,
                config=v4_config,
            )
            source_evidence = v4_result.evidence.get("source")
            if isinstance(source_evidence, dict):
                source_evidence["contract_source"] = contract_resolution.source
                source_evidence["cache_hit"] = contract_resolution.cache_hit
            record["genui_quality_v4"] = breakdown_to_mapping(v4_result)
            record["metrics"]["genui_quality_v4"] = v4_result.quality_0_100
            record["metrics"]["genui_quality_v4_dimensions"] = v4_result.dimensions
            record["metrics"]["genui_quality_v4_active_caps"] = v4_result.active_caps
            record["metrics"]["genui_metric_version"] = v4_result.metric_version

        if metric_mode in {"v5_0", "dual"}:
            v5_0_candidate: Any = (
                genui_json
                if "fallback_generated" in errors
                else generation_completion
            )
            v5_0_result = score_genui_completion_v5_0(
                v5_0_candidate,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=contract_resolution.contract,
                render_ok=None,
                config=v5_0_config,
            )
            source_evidence = v5_0_result.evidence.get("source")
            if isinstance(source_evidence, dict):
                source_evidence["contract_source"] = contract_resolution.source
                source_evidence["cache_hit"] = contract_resolution.cache_hit
            record["genui_quality_v5"] = breakdown_to_mapping(v5_0_result)
            record["metrics"]["genui_quality_v5"] = v5_0_result.quality_0_100
            record["metrics"]["genui_quality_v5_dimensions"] = v5_0_result.dimensions
            record["metrics"]["genui_quality_v5_active_caps"] = v5_0_result.active_caps

        if metric_mode in {"v5_1", "dual"}:
            legacy_contract = resolve_expected_ui_contract_v5_1(
                response_text,
                intent=intent_bucket,
                assets=assets_list,
            )
            record["genui_raw_completion"] = generation_completion
            generation_v5_1 = score_genui_completion_v5_1(
                generation_completion,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=legacy_contract.contract,
                render_ok=None,
                config=v5_1_config,
            )
            artifact_v5_1 = score_genui_completion_v5_1(
                genui_json,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=legacy_contract.contract,
                render_ok=None,
                config=v5_1_config,
            )
            record["generation_reward_v5_1"] = breakdown_to_mapping(
                generation_v5_1
            )
            record["render_artifact_quality_v5_1"] = breakdown_to_mapping(
                artifact_v5_1
            )
            record["genui_quality_v5_1"] = breakdown_to_mapping(
                artifact_v5_1
            )
            record["metric_identity_v5_1"] = {
                "metric_fingerprint": artifact_v5_1.metric_fingerprint,
                **artifact_v5_1.identity,
            }
            record["metrics"]["generation_reward_v5_1"] = (
                generation_v5_1.quality_0_100
            )
            record["metrics"]["render_artifact_quality_v5_1"] = (
                artifact_v5_1.quality_0_100
            )
            record["metrics"]["genui_quality_v5_1"] = (
                artifact_v5_1.quality_0_100
            )

        if metric_mode in {"v5", "v5_2", "dual"}:
            record["genui_raw_completion"] = generation_completion
            generation_result = score_genui_completion_v5_2(
                generation_completion,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=contract_resolution.contract,
                render_ok=None,
                config=v5_2_config,
            )
            artifact_result = score_genui_completion_v5_2(
                genui_json,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=contract_resolution.contract,
                render_ok=None,
                config=v5_2_config,
            )
            for result in (generation_result, artifact_result):
                source_evidence = result.evidence.get("source")
                if isinstance(source_evidence, dict):
                    source_evidence["contract_source"] = contract_resolution.source
                    source_evidence["cache_hit"] = contract_resolution.cache_hit
            record["generation_reward_v5_2"] = breakdown_to_mapping(
                generation_result
            )
            record["render_artifact_quality_v5_2"] = breakdown_to_mapping(
                artifact_result
            )
            record["genui_quality_v5_2"] = breakdown_to_mapping(
                artifact_result
            )
            record["metric_identity_v5_2"] = {
                "metric_fingerprint": artifact_result.metric_fingerprint,
                **artifact_result.identity,
                "raw_candidate_hash": generation_result.identity.get(
                    "raw_candidate_hash"
                ),
                "raw_canonical_candidate_hash": generation_result.identity.get(
                    "canonical_candidate_hash"
                ),
                "final_candidate_hash": artifact_result.identity.get(
                    "raw_candidate_hash"
                ),
                "final_canonical_candidate_hash": artifact_result.identity.get(
                    "canonical_candidate_hash"
                ),
            }
            record["metrics"]["generation_reward_v5_2"] = (
                generation_result.quality_0_100
            )
            record["metrics"]["render_artifact_quality_v5_2"] = (
                artifact_result.quality_0_100
            )
            record["metrics"]["genui_quality_v5_2"] = (
                artifact_result.quality_0_100
            )
            record["metrics"]["genui_quality_v5_2_dimensions"] = (
                artifact_result.dimensions
            )
            record["metrics"]["genui_quality_v5_2_active_caps"] = (
                artifact_result.active_caps
            )
            record["metrics"]["genui_quality_v5_2_binding_caps"] = (
                artifact_result.binding_caps
            )
            record["metrics"]["genui_quality_v5_2_matching_certification"] = (
                artifact_result.matching_certification
            )
            record["metrics"]["genui_quality_v5_2_dynamic_semantics"] = (
                artifact_result.dynamic_semantics
            )
            record["metrics"]["genui_metric_version"] = (
                artifact_result.metric_version
            )

            content_assignment = artifact_result.evidence.get(
                "content_assignment"
            )
            if isinstance(content_assignment, dict):
                recall = content_assignment.get("source_unit_recall")
                if isinstance(recall, (int, float)) and float(recall) < 0.80:
                    quality_warnings.append(
                        f"low_source_unit_recall: value={float(recall):.3f}"
                    )

        if metric_mode in {"v5_3", "dual"}:
            assert contract_resolution_v5_3 is not None
            record["genui_raw_completion"] = generation_completion
            record["expected_ui_contract_v5_3"] = (
                contract_resolution_v5_3.contract
            )
            record["expected_ui_contract_v5_3_source"] = (
                contract_resolution_v5_3.source
            )
            generation_v5_3 = score_genui_completion(
                generation_completion,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=contract_resolution_v5_3.contract,
                render_ok=None,
                config=v5_3_config,
            )
            artifact_v5_3 = score_genui_completion(
                genui_json,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=contract_resolution_v5_3.contract,
                render_ok=None,
                config=v5_3_config,
            )
            for result in (generation_v5_3, artifact_v5_3):
                source_evidence = result.evidence.get("source")
                if isinstance(source_evidence, dict):
                    source_evidence["contract_source"] = (
                        contract_resolution_v5_3.source
                    )
                    source_evidence["cache_hit"] = (
                        contract_resolution_v5_3.cache_hit
                    )
            record["generation_reward_v5_3"] = breakdown_to_mapping(
                generation_v5_3
            )
            record["render_artifact_quality_v5_3"] = breakdown_to_mapping(
                artifact_v5_3
            )
            record["genui_quality_v5_3"] = breakdown_to_mapping(
                artifact_v5_3
            )
            record["metric_identity_v5_3"] = {
                "metric_fingerprint": artifact_v5_3.metric_fingerprint,
                "reward_pipeline_fingerprint": (
                    artifact_v5_3.reward_pipeline_fingerprint
                ),
                **artifact_v5_3.identity,
            }
            record["metrics"]["generation_reward_v5_3"] = (
                generation_v5_3.quality_0_100
            )
            record["metrics"]["render_artifact_quality_v5_3"] = (
                artifact_v5_3.quality_0_100
            )
            record["metrics"]["genui_quality_v5_3"] = (
                artifact_v5_3.quality_0_100
            )
            record["metrics"]["genui_quality_v5_3_dimensions"] = (
                artifact_v5_3.dimensions
            )
            record["metrics"]["genui_quality_v5_3_active_caps"] = (
                artifact_v5_3.active_caps
            )
            record["metrics"]["genui_quality_v5_3_atomic_applicability"] = (
                artifact_v5_3.atomic_applicability
            )
            record["metrics"]["genui_metric_version"] = (
                artifact_v5_3.metric_version
            )
            table_matching = artifact_v5_3.evidence.get("table_matching")
            if isinstance(table_matching, dict):
                for match in table_matching.get("matches") or []:
                    if (
                        isinstance(match, dict)
                        and isinstance(match.get("row_fbeta"), (int, float))
                        and float(match["row_fbeta"]) < 0.80
                    ):
                        quality_warnings.append(
                            "missing_required_table_rows: "
                            f"row_fbeta={float(match['row_fbeta']):.3f}"
                        )
                        break
            action_matching = artifact_v5_3.evidence.get("action_matching")
            if isinstance(action_matching, dict) and int(
                action_matching.get("matched_required_count") or 0
            ) < int(action_matching.get("required_count") or 0):
                quality_warnings.append("missing_required_action")
            if not bool(
                artifact_v5_3.matching_certification.get(
                    "optimality_certified"
                )
            ):
                quality_warnings.append("matching_uncertified")
            if int(
                artifact_v5_3.dynamic_semantics.get("unknown_count", 0)
                or 0
            ):
                quality_warnings.append("renderer_expression_unknown")
            role_values = artifact_v5_3.evidence.get("role_gate_values")
            if isinstance(role_values, dict) and any(
                isinstance(value, (int, float)) and float(value) < 1.0
                for value in role_values.values()
            ):
                quality_warnings.append("missing_required_role")
            quality_warnings = list(dict.fromkeys(quality_warnings))
            record["validation"]["warnings"] = quality_warnings

        if metric_mode in {"v5_4", "dual"}:
            assert contract_resolution_v5_4 is not None
            record["genui_raw_completion"] = generation_completion
            record["expected_ui_contract_v5_4"] = (
                contract_resolution_v5_4.contract
            )
            record["expected_ui_contract_v5_4_source"] = (
                contract_resolution_v5_4.source
            )
            generation_v5_4 = generation_reward_a2ui_express_v1(
                generation_completion,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=contract_resolution_v5_4.contract,
                expected_ui_contract_source=contract_resolution_v5_4.source,
                render_ok=None,
                config=v5_4_config,
            )
            artifact_v5_4 = render_artifact_quality_v5_4(
                generation_completion,
                response_text,
                intent=intent_bucket,
                assets=assets_list,
                expected_ui_contract=contract_resolution_v5_4.contract,
                expected_ui_contract_source=contract_resolution_v5_4.source,
                render_ok=None,
                config=v5_4_config,
            )
            record["generation_reward_v5_4"] = breakdown_to_mapping(
                generation_v5_4
            )
            record["render_artifact_quality_v5_4"] = breakdown_to_mapping(
                artifact_v5_4
            )
            record["genui_quality_v5_4"] = breakdown_to_mapping(
                artifact_v5_4
            )
            record["metric_identity_v5_4"] = {
                "metric_fingerprint": artifact_v5_4.metric_fingerprint,
                "reward_pipeline_fingerprint": (
                    artifact_v5_4.reward_pipeline_fingerprint
                ),
                **artifact_v5_4.identity,
            }
            record["metrics"]["generation_reward_v5_4"] = (
                generation_v5_4.quality_0_100
            )
            record["metrics"]["render_artifact_quality_v5_4"] = (
                artifact_v5_4.quality_0_100
            )
            record["metrics"]["genui_quality_v5_4"] = (
                artifact_v5_4.quality_0_100
            )
            record["metrics"]["genui_quality_v5_4_dimensions"] = (
                artifact_v5_4.dimensions
            )
            record["metrics"]["genui_quality_v5_4_active_caps"] = (
                artifact_v5_4.active_caps
            )
            record["metrics"]["genui_quality_v5_4_atomic_applicability"] = (
                artifact_v5_4.atomic_applicability
            )
            record["metrics"]["genui_quality_v5_4_artifact_quality"] = (
                artifact_v5_4.artifact_quality_0_1 * 100.0
            )
            record["metrics"]["genui_metric_version"] = (
                artifact_v5_4.metric_version
            )
            if not bool(
                artifact_v5_4.matching_certification.get(
                    "optimality_certified"
                )
            ):
                quality_warnings.append("matching_uncertified")
            if int(
                artifact_v5_4.dynamic_semantics.get("unknown_count", 0)
                or 0
            ):
                quality_warnings.append("renderer_expression_unknown")
            if not bool(
                artifact_v5_4.dynamic_semantics.get(
                    "android_parity_certified", False
                )
            ):
                quality_warnings.append("android_parity_uncertified")
            quality_warnings = list(dict.fromkeys(quality_warnings))
            record["validation"]["warnings"] = quality_warnings

        sample_legacy_score = (
            _compute_sample_legacy_score(record)
            if metric_mode in {"legacy", "dual"}
            else None
        )
        if metric_mode in {"legacy", "dual"}:
            record["metrics"]["legacy_structural_richness_score"] = sample_legacy_score
            # Compatibility field for one migration window.
            record["metrics"]["overall_score"] = sample_legacy_score
        writer.append(record)
        existing_ids.add(ui_id)
        if sample_legacy_score is None and metric_mode == "legacy":
            logger.info(
                "Stage3 created ui_id=%s schema_ok=%s contract_source=%s contract_cache_hit=%s",
                ui_id,
                schema_valid_strict,
                contract_resolution.source,
                contract_resolution.cache_hit,
            )
        else:
            quality_v4 = record["metrics"].get("genui_quality_v4")
            quality_v5 = record["metrics"].get("genui_quality_v5")
            logger.info(
                "Stage3 created ui_id=%s schema_ok=%s legacy_score=%s "
                "genui_quality_v4=%s genui_quality_v5=%s "
                "contract_source=%s contract_cache_hit=%s",
                ui_id,
                schema_valid_strict,
                None if sample_legacy_score is None else round(sample_legacy_score, 2),
                None if quality_v4 is None else round(float(quality_v4), 2),
                None if quality_v5 is None else round(float(quality_v5), 2),
                contract_resolution.source,
                contract_resolution.cache_hit,
            )
        total_created += 1
        # A resumed run may already contain thousands of records.  Rebuilding
        # the full aggregate after the first newly-created record makes the
        # single Stage 3 worker spend minutes CPU-bound before it can submit
        # the next batch.  Keep the first-record aggregate for fresh runs, but
        # defer it on resumes until the normal aggregate interval.
        if total_created % aggregate_every == 0 or (
            total_created == 1 and existing_genui_count == 0
        ):
            _write_aggregates(reason=f"after_ui={ui_id}")

        if errors:
            error_path = artifacts_dir / f"error_{ui_id}.json"
            error_payload = {
                "prompt": prompt,
                "raw_text": raw_text,
                "errors": errors,
            }
            error_path.write_text(
                json.dumps(error_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _record_generation_error(task: dict[str, Any], err: str, raw_payload: Any = None) -> None:
        nonlocal total_failed
        total_failed += 1
        logger.error("Stage3 generation error response_id=%s: %s", task.get("response_id"), err)
        error_path = artifacts_dir / f"error_{task.get('ui_id', 'unknown')}.json"
        error_payload = {
            "prompt": task.get("prompt"),
            "error": err,
            "raw_payload": raw_payload,
            "response_id": task.get("response_id"),
            "query_id": task.get("query_id"),
            "ui_id": task.get("ui_id"),
        }
        error_path.write_text(
            json.dumps(error_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _is_transient_local_generation_error(err: str | None) -> bool:
        if adapter.spec.provider != "local" or not err:
            return False
        lowered = str(err).lower()
        transient_markers = (
            "connection refused",
            "connection reset",
            "connection aborted",
            "remote end closed connection",
            "remote disconnected",
            "temporarily unavailable",
            "service unavailable",
            "http error 500",
            "500 internal server error",
            "enginedeaderror",
            "enginecore encountered",
            "asyncllm output_handler failed",
        )
        return any(marker in lowered for marker in transient_markers)

    def _generate_single_result(task: dict[str, Any]):
        def _call():
            rate_limiter.acquire()
            return adapter.generate(
                prompt=task["prompt"],
                system=system_prompt,
                temperature=generation_temperature,
                max_tokens=max_tokens,
                seed=task["seed"],
                json_mode=(
                    True if adapter.spec.supports_json_mode and not express_mode else False
                ),
            )

        retry_result_errors = os.environ.get("LOCAL_VLLM_RETRY_RESULT_ERRORS", "1").strip().lower()
        retry_result_errors_enabled = retry_result_errors not in {"0", "false", "no", "off"}
        retry_interval_s = float(os.environ.get("LOCAL_VLLM_RETRY_INTERVAL_SECONDS", "10") or "10")
        retry_max_s = float(os.environ.get("LOCAL_VLLM_RETRY_MAX_SECONDS", "0") or "0")
        retry_interval_s = max(1.0, retry_interval_s)
        first_failure_at: float | None = None
        transient_attempt = 0

        while True:
            try:
                result = with_retry(_call, max_attempts=max_attempts)
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    logger.error(
                        "Stage3 rate limit info: limits=%s headers=%s",
                        exc.limits or "unset",
                        exc.headers or "none",
                    )
                raise

            if not (
                retry_result_errors_enabled
                and result.error
                and _is_transient_local_generation_error(result.error)
            ):
                return result

            now = time.time()
            if first_failure_at is None:
                first_failure_at = now
            elapsed_s = now - first_failure_at
            if retry_max_s > 0 and elapsed_s >= retry_max_s:
                return result

            transient_attempt += 1
            if transient_attempt == 1 or transient_attempt % 6 == 0:
                logger.warning(
                    "Stage3 local vLLM transient error response_id=%s; retrying same sample in %.0fs "
                    "(attempt=%s elapsed=%.0fs): %s",
                    task.get("response_id"),
                    retry_interval_s,
                    transient_attempt,
                    elapsed_s,
                    str(result.error)[:500],
                )
            time.sleep(retry_interval_s)

    def _generate_single(task: dict[str, Any]) -> None:
        try:
            result = _generate_single_result(task)
        except Exception as exc:
            _record_generation_error(task, str(exc))
            return
        if result.error:
            _record_generation_error(task, result.error, result.raw)
            return
        cache.set(
            task["prompt_hash"],
            result.text,
            result.raw,
            result.reasoning_text,
            result.reasoning_source,
            result.reasoning_tokens,
        )
        _process_generated(
            task,
            result.text,
            result.raw,
            result.latency_ms,
            result.input_tokens,
            result.output_tokens,
            result.provider,
            result.model,
            result.error,
            result.reasoning_text,
            result.reasoning_source,
            result.reasoning_tokens,
        )

    def _flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        tasks = pending
        pending = []
        results = None

        if use_batch and len(tasks) > 1:
            prompts = [task["prompt"] for task in tasks]
            seeds = [task["seed"] for task in tasks]

            def _call_batch():
                rate_limiter.acquire()
                return adapter.generate_batch(
                    prompts=prompts,
                    system=system_prompt,
                    temperature=generation_temperature,
                    max_tokens=max_tokens,
                    seeds=seeds,
                    json_mode=(
                        True if adapter.spec.supports_json_mode and not express_mode else False
                    ),
                    batch_name=f"stage3_{int(time.time())}",
                )

            try:
                results = with_retry(_call_batch, max_attempts=max_attempts)
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    logger.error(
                        "Stage3 rate limit info: limits=%s headers=%s",
                        exc.limits or "unset",
                        exc.headers or "none",
                    )
                if getattr(adapter, "batch_only", False):
                    raise
                logger.warning("Stage3 batch failed; falling back to single calls: %s", exc)
                results = None

        if use_parallel and results is None and len(tasks) > 1:
            parallel_results: list[tuple[dict[str, Any], Any]] = []
            parallel_failed = False
            parallel_error: Exception | None = None
            with ThreadPoolExecutor(max_workers=gemini_parallel_workers) as executor:
                future_to_task = {executor.submit(_generate_single_result, task): task for task in tasks}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        parallel_failed = True
                        parallel_error = exc
                        logger.warning(
                            "Stage3 parallel worker failed response_id=%s: %s",
                            task["response_id"],
                            exc,
                        )
                        continue
                    parallel_results.append((task, result))

            if parallel_failed:
                logger.warning(
                    "Stage3 parallel batch degraded to sequential processing due to worker errors: %s",
                    parallel_error,
                )
                for task in tasks:
                    _generate_single(task)
                return

            for task, result in parallel_results:
                if result.error:
                    _record_generation_error(task, result.error, result.raw)
                    continue
                cache.set(
                    task["prompt_hash"],
                    result.text,
                    result.raw,
                    result.reasoning_text,
                    result.reasoning_source,
                    result.reasoning_tokens,
                )
                _process_generated(
                    task,
                    result.text,
                    result.raw,
                    result.latency_ms,
                    result.input_tokens,
                    result.output_tokens,
                    result.provider,
                    result.model,
                    result.error,
                    result.reasoning_text,
                    result.reasoning_source,
                    result.reasoning_tokens,
                )
            return

        if results is None:
            for task in tasks:
                _generate_single(task)
            return

        if len(results) != len(tasks):
            logger.warning(
                "Stage3 batch size mismatch: expected=%s got=%s",
                len(tasks),
                len(results),
            )

        for task, result in zip_longest(tasks, results):
            if result is None:
                raise RuntimeError("Stage3 batch missing result")
            if result.error:
                _record_generation_error(task, result.error, result.raw)
                continue
            cache.set(
                task["prompt_hash"],
                result.text,
                result.raw,
                result.reasoning_text,
                result.reasoning_source,
                result.reasoning_tokens,
            )
            _process_generated(
                task,
                result.text,
                result.raw,
                result.latency_ms,
                result.input_tokens,
                result.output_tokens,
                result.provider,
                result.model,
                result.error,
                result.reasoning_text,
                result.reasoning_source,
                result.reasoning_tokens,
            )

    try:
        for response in iter_jsonl(responses_path):
            if stop:
                break
            response_id = response.get("response_id")
            query_id = response.get("query_id")
            response_text = response.get("response_text")
            assets = response.get("assets") if isinstance(response, dict) else None
            assets_list = assets if isinstance(assets, list) else []
            persisted_contract = response.get("expected_ui_contract") if isinstance(response, dict) else None
            persisted_contract_source = response.get("expected_ui_contract_source") if isinstance(response, dict) else None
            n_idx = int(response.get("n_idx", 1))
            if not response_id or not query_id or not response_text:
                continue

            if not assets_list:
                auto_assets = _auto_download_response_assets(
                    response_id=response_id,
                    response_text=response_text,
                    assets_dir=auto_assets_dir,
                    logger=logger,
                )
                if auto_assets:
                    assets_list = auto_assets
                    logger.info(
                        "Stage3 auto-downloaded assets response_id=%s count=%s",
                        response_id,
                        len(auto_assets),
                    )

            for c_idx in range(1, candidates_per_response + 1):
                if stop:
                    break
                if max_total is not None:
                    remaining = max_total - total_created
                    if remaining <= 0:
                        stop = True
                        break
                    if len(pending) >= remaining:
                        stop = True
                        break

                ui_id = _make_ui_id(query_id, n_idx, c_idx, active_ir_format)
                if ui_id in existing_ids:
                    continue

                masked_response_text, raw_to_placeholder, placeholder_to_raw = _mask_model_references(
                    response_text,
                    assets_list,
                )
                prompt = _build_prompt_for(
                    response_id,
                    response_text,
                    assets_list,
                    masked_response_text=masked_response_text,
                    raw_to_placeholder=raw_to_placeholder,
                )
                prompt_hash = hash_text(
                    f"{adapter.spec.name}:{system_prompt or ''}\n---\n{prompt}"
                )
                intent_info = intent_lookup.get(query_id, {})
                task = {
                    "ui_id": ui_id,
                    "response_id": response_id,
                    "query_id": query_id,
                    "response_text": response_text,
                    "masked_response_text": masked_response_text,
                    "assets_list": assets_list,
                    "asset_placeholder_map": placeholder_to_raw,
                    "expected_ui_contract": persisted_contract,
                    "expected_ui_contract_source": persisted_contract_source,
                    "intent": intent_info.get("intent"),
                    "tags": intent_info.get("tags"),
                    "query_text": intent_info.get("query_text") or "",
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "seed": seed + c_idx,
                }

                cached = cache.get(prompt_hash)
                if cached:
                    _process_generated(
                        task,
                        cached.text,
                        cached.raw,
                        0.0,
                        0,
                        0,
                        adapter.spec.provider,
                        adapter.spec.model,
                        None,
                        cached.reasoning_text,
                        cached.reasoning_source,
                        cached.reasoning_tokens,
                    )
                    continue

                if not use_batch:
                    if use_parallel:
                        pending.append(task)
                        if len(pending) >= gemini_parallel_workers:
                            _flush_pending()
                    else:
                        _generate_single(task)
                    continue

                pending.append(task)
                if len(pending) >= batch_size:
                    _flush_pending()

        _flush_pending()
        if total_failed > 0:
            logger.warning("Stage3 completed with generation failures=%s created=%s", total_failed, total_created)
        else:
            logger.info("Stage3 completed created=%s", total_created)
    finally:
        _write_aggregates()



