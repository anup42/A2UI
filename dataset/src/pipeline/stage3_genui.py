from __future__ import annotations

import json
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
from urllib.request import Request, urlopen

from pipeline.cache import PromptCache
from pipeline.common import extract_json, load_prompt, render_prompt
from pipeline.flat_spec_contract import (
    build_fallback_flat_spec,
    build_flat_spec_repair_prompt,
    coerce_and_validate,
    extract_json_element,
)
from pipeline.image_resolver import repair_flat_spec_images
from pipeline.metrics import (
    content_coverage,
    dup_rate,
    lint_score,
    count_tokens,
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
from utils.hashing import hash_text
from utils.rate_limit import RateLimiter
from utils.retry import with_retry


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

    response_words = len(re.findall(r"[A-Za-z0-9]+", response_text or ""))
    element_count = len(elements)
    table_count = 0
    text_count = 0
    generated_heading_count = 0
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        element_type = str(element.get("type", "")).lower()
        if element_type == "table":
            table_count += 1
        elif element_type == "text":
            text_count += 1
            props = element.get("props") if isinstance(element.get("props"), dict) else {}
            variant = str(props.get("variant", "")).lower()
            if variant in {"h2", "h3"}:
                generated_heading_count += 1

    warnings: list[str] = []
    if response_words >= 180 and element_count < 16:
        warnings.append(f"low_component_count: words={response_words} elements={element_count}")
    if response_words >= 120 and table_count >= 1 and element_count <= 12 and text_count <= 5:
        warnings.append(
            f"sparse_ir: words={response_words} elements={element_count} tables={table_count} text={text_count}"
        )

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


def _make_ui_id(query_id: str, n_idx: int, candidate_idx: int) -> str:
    suffix = query_id.replace("q_", "")
    if candidate_idx == 1:
        return f"u_{suffix}_{n_idx:02d}"
    return f"u_{suffix}_{n_idx:02d}_{candidate_idx:02d}"


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


def _build_asset_context(assets: list[dict]) -> str:
    if not assets:
        return ""
    lines: list[str] = []
    for item in assets:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        local_path = _to_render_asset_path(path)
        if not local_path:
            continue
        if url:
            lines.append(f"- {url} -> {local_path}")
        else:
            lines.append(f"- {local_path}")
    if not lines:
        return ""
    return "Assets (local copies of any URLs in the response; use ONLY these local paths):\n" + "\n".join(lines)


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


def _normalize_flat_spec_text_content(genui_json: Any) -> Any:
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




def _maybe_compact_prompt_template(template: str, adapter: BaseLLMAdapter, logger) -> str:
    provider = (adapter.spec.provider or "").lower()
    model = (adapter.spec.model or "").lower()
    compact_enabled = os.getenv("GENUI_COMPACT_PROMPT_FOR_GEMMA", "0").strip().lower()
    if compact_enabled in {"0", "false", "no", "off"}:
        return template
    if provider != "gemini" or not model.startswith("gemma-"):
        return template

    compact = template
    marker = "\nSchema ("
    if marker in template:
        compact = template.split(marker, 1)[0].rstrip()
    if "flat-spec" in template.lower() or "\"root\"" in template:
        compact = (
            f"{compact}\n\n"
            "Additional strict requirements:\n"
            "- Output MUST be one JSON object with top-level root/state/elements.\n"
            "- Do not output legacy message arrays.\n"
            "- root must reference an existing id in elements.\n"
            "- Every element must contain type, props, and children.\n"
            "- Output ONLY JSON.\n"
        )
    else:
        compact = (
            f"{compact}\n\n"
            "Additional strict requirements:\n"
            "- Use message types: createSurface, updateComponents, updateDataModel, deleteSurface.\n"
            "- Set version to v0.9.\n"
            "- Include createSurface before updates.\n"
            "- In updateComponents, include exactly one root component with id 'root'.\n"
            "- Output ONLY a JSON array of messages.\n"
        )
    before = count_tokens(template)
    after = count_tokens(compact)
    logger.info(
        "Stage3 compact prompt enabled for model=%s tokens=%s->%s",
        adapter.spec.model,
        before,
        after,
    )
    return compact
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
        "Convert the response text into valid GenUICraft JSON.\n"
        "Return ONLY JSON.\n\n"
        "Response:\n{response_text}"
    )
    logger.info(
        "Stage3 prompt mode=system_prefix provider=%s system_tokens=%s user_template_tokens=%s",
        provider,
        count_tokens(system_prompt),
        count_tokens(user_template),
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
) -> None:
    prompt_template = _maybe_compact_prompt_template(load_prompt(prompt_path), adapter, logger)
    prompt_version = _extract_prompt_version(prompt_template, prompt_path)
    system_prompt, user_prompt_template = _prepare_prompt_context(prompt_template, adapter, logger)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    flat_spec_mode = _is_flat_spec_schema(schema)
    logger.info("Stage3 schema mode: %s", "flat_spec" if flat_spec_mode else "legacy_messages")
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
            render_log_path = genui_path.parent / "render.jsonl"
            if render_log_path.exists():
                for render_row in iter_jsonl(render_log_path):
                    ui_id = render_row.get("ui_id")
                    if isinstance(ui_id, str) and ui_id:
                        render_rows_by_ui_id[ui_id] = render_row
            aggregates = aggregate_metrics(rows, render_rows_by_ui_id=render_rows_by_ui_id)
            aggregates["overall_score"] = compute_overall_score(
                aggregates,
                aggregate_weights or {},
            )
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

    def _compute_sample_overall_score(record: dict[str, Any]) -> float | None:
        try:
            sample_aggregate = aggregate_metrics([record])
            return compute_overall_score(sample_aggregate, aggregate_weights or {})
        except Exception as exc:  # best-effort logging metric
            logger.debug("Stage3 sample score failed ui_id=%s: %s", record.get("ui_id"), exc)
            return None

    total_created = 0
    total_failed = 0
    pending: list[dict[str, Any]] = []
    use_parallel = adapter.spec.provider == "gemini" and gemini_parallel_workers > 1
    use_batch = (
        adapter.spec.provider == "gemini"
        and hasattr(adapter, "generate_batch")
        and not use_parallel
    )
    if batch_size <= 0:
        batch_size = 100
    stop = False

    def _build_prompt_for(response_id: str, response_text: str, assets_list: list[dict]) -> str:
        asset_context = _build_asset_context(assets_list)
        if assets_list:
            asset_policy = (
                "Asset URL policy for this request:\n"
                "- Use only local media paths from the provided Assets mapping.\n"
                "- Do not emit remote media URLs for images/icons.\n"
                "- Do not invent local placeholder paths not present in the mapping."
            )
        else:
            asset_policy = (
                "Asset URL policy for this request:\n"
                "- No local asset mapping is provided.\n"
                "- Preserve media URLs from the response exactly as written.\n"
                "- Do not invent local placeholder paths such as /image.jpg or /asset/foo.png."
            )

        response_with_policy = f"{response_text}\n\n{asset_policy}"
        prompt_response_text = response_with_policy
        if asset_context:
            prompt_response_text = f"{response_with_policy}\n\n{asset_context}"

        prompt = render_prompt(user_prompt_template, response_text=prompt_response_text)
        if prompt_max_tokens:
            system_tokens = count_tokens(system_prompt) if system_prompt else 0
            prompt_tokens = count_tokens(prompt) + system_tokens
            if prompt_tokens > prompt_max_tokens:
                # First attempt: drop asset context to save tokens.
                prompt = render_prompt(user_prompt_template, response_text=response_with_policy)
                prompt_tokens = count_tokens(prompt) + system_tokens
            if prompt_tokens > prompt_max_tokens:
                base_prompt = render_prompt(user_prompt_template, response_text="")
                base_tokens = count_tokens(base_prompt) + system_tokens
                budget = max(200, prompt_max_tokens - base_tokens)
                trimmed_text, truncated = _truncate_tokens(response_text, budget)
                if truncated:
                    logger.warning(
                        "Stage3 prompt truncated response_id=%s tokens=%s budget=%s",
                        response_id,
                        count_tokens(response_text),
                        budget,
                    )
                prompt = render_prompt(
                    user_prompt_template,
                    response_text=f"{trimmed_text}\n\n{asset_policy}",
                )
        return prompt

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

        parsed_ok = True
        errors: list[str] = []
        converted_from_legacy = False
        try:
            parsed_json = extract_json_element(raw_text) if flat_spec_mode else extract_json(raw_text)
            if flat_spec_mode:
                coerce_result = coerce_and_validate(parsed_json)
                if not coerce_result.is_valid:
                    genui_json = None
                    parsed_ok = False
                    errors.append(f"flat_spec_error: {coerce_result.error}")
                else:
                    genui_json = coerce_result.spec
                    converted_from_legacy = bool(coerce_result.converted_from_legacy)
            else:
                genui_json = parsed_json
        except Exception as exc:
            parsed_ok = False
            genui_json = None
            errors.append(f"json_parse_error: {exc}")

        schema_valid_strict = False
        schema_valid_lenient = False
        repair_needed = False

        if parsed_ok and genui_json is not None:
            schema_valid_strict, schema_errors, validator_ok = _validate_schema(
                schema, genui_json, schema_path.parent
            )
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
            if flat_spec_mode:
                failure_reason = "; ".join(errors[-5:]) if errors else None
                repaired_text = build_flat_spec_repair_prompt(raw_text, failure_reason=failure_reason)
            else:
                repair_prompt = (
                    "The previous output was not valid JSON or failed schema validation. "
                    "Fix the output to be valid JSON that satisfies the schema. "
                    f"Errors: {errors}.\n"
                    "Return ONLY the corrected JSON."
                )
                repaired_text = f"{repair_prompt}\n\nOriginal:\n{raw_text}"

            def _repair_call():
                rate_limiter.acquire()
                return adapter.generate(
                    prompt=repaired_text,
                    system=system_prompt,
                    temperature=repair_temperature,
                    max_tokens=max_tokens,
                    seed=seed + 100 + repair_attempts,
                    json_mode=True if adapter.spec.supports_json_mode else False,
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
            try:
                parsed_json = extract_json_element(raw_text) if flat_spec_mode else extract_json(raw_text)
                if flat_spec_mode:
                    coerce_result = coerce_and_validate(parsed_json)
                    if not coerce_result.is_valid:
                        parsed_ok = False
                        genui_json = None
                        errors.append(f"repair_flat_spec_error: {coerce_result.error}")
                        continue
                    genui_json = coerce_result.spec
                    converted_from_legacy = bool(coerce_result.converted_from_legacy)
                else:
                    genui_json = parsed_json
                parsed_ok = True
            except Exception as exc:
                parsed_ok = False
                errors.append(f"repair_json_parse_error: {exc}")
                continue

            schema_valid_strict, schema_errors, validator_ok = _validate_schema(
                schema, genui_json, schema_path.parent
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
            and flat_spec_mode
            and regen_attempt < final_regen_attempts
        ):
            regen_attempt += 1
            repair_needed = True
            regeneration_prompt = (
                f"{prompt}\n\n"
                "Previous output was invalid or incomplete.\n"
                "Regenerate the full flat-spec JSON from the source response.\n"
                "Return ONLY one valid JSON object with root/state/elements.\n"
                "The output MUST contain at least 8 elements with a root Stack, "
                "heading Text elements (h2/h3), content elements, and at least one Button or Table. "
                "A two-element fallback (Column + Text) is not acceptable.\n"
                "Keep JSON compact and avoid literal markdown markers in text fields."
            )

            def _regen_call():
                rate_limiter.acquire()
                return adapter.generate(
                    prompt=regeneration_prompt,
                    system=system_prompt,
                    temperature=final_regen_temperature,
                    max_tokens=max(max_tokens, 8192),
                    seed=seed + 900 + regen_attempt,
                    json_mode=True if adapter.spec.supports_json_mode else False,
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
            try:
                parsed_json = extract_json_element(raw_text) if flat_spec_mode else extract_json(raw_text)
                if flat_spec_mode:
                    coerce_result = coerce_and_validate(parsed_json)
                    if not coerce_result.is_valid:
                        parsed_ok = False
                        genui_json = None
                        errors.append(f"final_regen_flat_spec_error: {coerce_result.error}")
                        continue
                    genui_json = coerce_result.spec
                    converted_from_legacy = bool(coerce_result.converted_from_legacy)
                else:
                    genui_json = parsed_json
                parsed_ok = True
            except Exception as exc:
                parsed_ok = False
                errors.append(f"final_regen_json_parse_error: {exc}")
                continue

            schema_valid_strict, schema_errors, validator_ok = _validate_schema(
                schema, genui_json, schema_path.parent
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
            and flat_spec_mode
            and isinstance(genui_json.get("elements"), dict)
            and len(genui_json["elements"]) < 5
        ):
            errors.append("spec_too_simple: fewer than 5 elements")
            parsed_ok = False

        if not parsed_ok or genui_json is None or not schema_valid_strict:
            # Final fallback: build a minimal valid output matching the configured schema mode.
            fallback_text = _apply_asset_replacements(response_text, assets_list)
            if flat_spec_mode:
                genui_json = build_fallback_flat_spec(fallback_text)
            else:
                surface_id = f"surface_{query_id}"
                catalog_id = "https://genui.local/specification/v0_9/standard_catalog.json"
                genui_json = [
                    {
                        "version": "v0.9",
                        "createSurface": {
                            "surfaceId": surface_id,
                            "catalogId": catalog_id,
                        },
                    },
                    {
                        "version": "v0.9",
                        "updateComponents": {
                            "surfaceId": surface_id,
                            "components": [
                                {
                                    "id": "root",
                                    "component": "Column",
                                    "children": ["text_1"],
                                },
                                {
                                    "id": "text_1",
                                    "component": "Text",
                                    "text": fallback_text,
                                    "variant": "body",
                                },
                            ],
                        },
                    },
                ]
            # Re-validate schema for fallback.
            parsed_ok = True
            errors = ["fallback_generated"]
            schema_valid_strict, schema_errors, validator_ok = _validate_schema(
                schema, genui_json, schema_path.parent
            )
            if schema_valid_strict:
                schema_valid_lenient = True
            else:
                errors.extend(schema_errors)
                if not validator_ok:
                    schema_valid_lenient = True

        genui_json = _rewrite_genui_asset_urls(genui_json, assets_list)
        if flat_spec_mode:
            genui_json = _normalize_flat_spec_text_content(genui_json)
            genui_json, resolved_images = repair_flat_spec_images(
                genui_json,
                query_text,
                response_text,
            )
            if resolved_images:
                errors.append(f"dataset_image_resolver_added={resolved_images}")

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
            if flat_spec_mode
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
            "genui_json": genui_json,
            "assets": assets_list,
            "toon": toon,
            "validation": {
                "json_parse_ok": parsed_ok,
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
        sample_overall_score = _compute_sample_overall_score(record)
        record["metrics"]["overall_score"] = sample_overall_score
        writer.append(record)
        existing_ids.add(ui_id)
        if sample_overall_score is None:
            logger.info("Stage3 created ui_id=%s schema_ok=%s", ui_id, schema_valid_strict)
        else:
            logger.info(
                "Stage3 created ui_id=%s schema_ok=%s overall_score=%.2f",
                ui_id,
                schema_valid_strict,
                sample_overall_score,
            )
        total_created += 1
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

    def _generate_single_result(task: dict[str, Any]):
        def _call():
            rate_limiter.acquire()
            return adapter.generate(
                prompt=task["prompt"],
                system=system_prompt,
                temperature=generation_temperature,
                max_tokens=max_tokens,
                seed=task["seed"],
                json_mode=True if adapter.spec.supports_json_mode else False,
            )

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
        return result

    def _generate_single(task: dict[str, Any]) -> None:
        try:
            result = _generate_single_result(task)
        except Exception as exc:
            _record_generation_error(task, str(exc))
            return
        if result.error:
            _record_generation_error(task, result.error, result.raw)
            return
        cache.set(task["prompt_hash"], result.text, result.raw)
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
                    json_mode=True if adapter.spec.supports_json_mode else False,
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
                cache.set(task["prompt_hash"], result.text, result.raw)
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
            cache.set(task["prompt_hash"], result.text, result.raw)
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

                ui_id = _make_ui_id(query_id, n_idx, c_idx)
                if ui_id in existing_ids:
                    continue

                prompt = _build_prompt_for(response_id, response_text, assets_list)
                prompt_hash = hash_text(
                    f"{adapter.spec.name}:{system_prompt or ''}\n---\n{prompt}"
                )
                intent_info = intent_lookup.get(query_id, {})
                task = {
                    "ui_id": ui_id,
                    "response_id": response_id,
                    "query_id": query_id,
                    "response_text": response_text,
                    "assets_list": assets_list,
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







