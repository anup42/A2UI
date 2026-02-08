from __future__ import annotations

from datetime import datetime
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
)

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


def _clean_url(value: str) -> str:
    cleaned = value.strip().strip("()[]{}<>\"'").rstrip(".,;:)]}!?")
    if not cleaned:
        return ""
    cleaned = cleaned.split()[0]
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return cleaned


def _is_asset_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in _ASSET_EXTENSIONS):
        return True
    host = parsed.netloc.lower()
    return any(hint in host for hint in _ASSET_HOST_HINTS)


def _extract_asset_urls(text: str) -> list[str]:
    urls: list[str] = []
    if not text:
        return urls
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
        candidates = _URL_RE.findall(line)
        if not candidates:
            continue
        for match in candidates:
            cleaned = _clean_url(match)
            if not cleaned:
                continue
            if _is_asset_url(cleaned):
                urls.append(cleaned)
    return list(dict.fromkeys(urls))


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._") or "asset"


def _download_assets(
    response_text: str,
    response_id: str,
    assets_dir: Path,
    logger,
    url_cache: dict[str, Path],
    max_bytes: int = 25 * 1024 * 1024,
) -> list[dict]:
    assets = []
    if _offline_mode_enabled():
        return assets
    urls = _extract_asset_urls(response_text)
    if not urls:
        return assets
    assets_dir.mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(urls, start=1):
        if url in url_cache:
            path = url_cache[url]
            assets.append(
                {"url": url, "path": str(path.relative_to(assets_dir.parent))}
            )
            continue
        try:
            parsed = urllib.parse.urlparse(url)
            basename = Path(parsed.path).name
            safe_name = _safe_name(basename) or f"asset_{idx}"
            dest: Path
            content_type = None
            with urllib.request.urlopen(url, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type")
                data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                logger.warning("Stage2 asset too large, skipped url=%s", url)
                continue
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
            filename = f"{response_id}_{idx}_{_safe_name(Path(safe_name).stem)}{ext}"
            dest = assets_dir / filename
            dest.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            url_cache[url] = dest
            assets.append(
                {
                    "url": url,
                    "path": str(dest.relative_to(assets_dir.parent)),
                    "sha256": sha,
                    "bytes": len(data),
                }
            )
        except Exception as exc:
            logger.warning("Stage2 asset download failed url=%s err=%s", url, exc)
    return assets


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

    total_created = 0

    for query in iter_jsonl(queries_path):
        query_id = query.get("query_id")
        query_text = query.get("query_text")
        if not query_id or not query_text:
            continue
        remaining = n_per_query
        n_idx = 1
        while remaining > 0:
            if max_total is not None and total_created >= max_total:
                logger.info("Stage2 reached max_total=%s", max_total)
                return
            batch = max(1, min(batch_size, remaining))
            response_id = _make_response_id(query_id, n_idx)
            if response_id in existing_ids:
                n_idx += 1
                remaining -= 1
                continue

            temperature = temperatures[(n_idx - 1) % len(temperatures)] if temperatures else 0.7
            prompt = render_prompt(prompt_template, query_text=query_text)
            if batch > 1:
                prompt = (
                    f"{prompt}\n\nReturn exactly {batch} distinct responses as a JSON array of strings."
                )
            prompt_hash = hash_text(f"{adapter.spec.name}:{prompt}")

            cached = cache.get(prompt_hash)
            if cached:
                raw_text = cached.text.strip()
                latency_ms = 0.0
                input_tokens = 0
                output_tokens = 0
                provider = adapter.spec.provider
                model = adapter.spec.model
            else:
                def _call():
                    rate_limiter.acquire()
                    return adapter.generate(
                        prompt=prompt,
                        system=None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed + n_idx,
                        json_mode=batch > 1 and adapter.spec.supports_json_mode,
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
                                query_id,
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
                            query_id,
                            attempt,
                            max_attempts,
                            result.error,
                        )
                        _sleep_backoff(attempt)
                        continue
                    logger.error("Stage2 error query_id=%s: %s", query_id, result.error)
                    result = None
                    break
            
                if result is None or result.error:
                    n_idx += 1
                    remaining -= 1
                    continue
                raw_text = result.text.strip()
                latency_ms = result.latency_ms
                input_tokens = result.input_tokens
                output_tokens = result.output_tokens
                provider = result.provider
                model = result.model
                cache.set(prompt_hash, result.text, result.raw)
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

            for response_text in responses[:batch]:
                if not response_text:
                    continue
                norm_hash = hash_text(normalize_text(response_text))
                if norm_hash in existing_hashes:
                    logger.info("Stage2 duplicate response query_id=%s", query_id)
                    continue

                assets = _download_assets(
                    response_text,
                    response_id,
                    assets_dir,
                    logger,
                    url_cache,
                )
                record = {
                    "response_id": response_id,
                    "query_id": query_id,
                    "n_idx": n_idx,
                    "response_text": response_text,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "assets": assets,
                    "gen": {
                        "provider": provider,
                        "model": model,
                        "latency_ms": latency_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": None,
                    },
                }
                writer.append(record)
                existing_ids.add(response_id)
                existing_hashes.add(norm_hash)
                logger.info("Stage2 created response_id=%s", response_id)
                n_idx += 1
                remaining -= 1
                total_created += 1
                if max_total is not None and total_created >= max_total:
                    logger.info("Stage2 reached max_total=%s", max_total)
                    return
                response_id = _make_response_id(query_id, n_idx)

            if batch == 1 and not responses:
                n_idx += 1
                remaining -= 1
