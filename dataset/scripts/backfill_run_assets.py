from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import mimetypes
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.stage2_responses import (  # noqa: E402
    _MIME_EXTENSION_MAP,
    _downloaded_asset_content_valid,
    _extract_asset_entries,
    _safe_name,
    _sniff_asset_mime,
)
from pipeline.toon_convert import encode_toon, roundtrip_ok  # noqa: E402
from pipeline.metrics import count_characters  # noqa: E402

ASSET_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".bmp", ".tif", ".tiff", ".pdf"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".bmp", ".tif", ".tiff"}
STOP_TOKENS = {
    "the",
    "and",
    "with",
    "image",
    "photo",
    "picture",
    "thumb",
    "thumbnail",
    "1200px",
    "1024px",
    "800px",
    "640px",
    "500px",
    "01",
    "1",
    "2",
    "3",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _asset_ext(url: str, safe_name: str, content_type: str | None) -> str:
    # Prefer the actual response type over sometimes-hallucinated file suffixes.
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if ctype == "image/avif":
        return ".avif"
    if ctype in _MIME_EXTENSION_MAP:
        return _MIME_EXTENSION_MAP[ctype]
    suffix = Path(safe_name).suffix.lower() or Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix:
        return suffix
    ext = _MIME_EXTENSION_MAP.get(ctype)
    if ext:
        return ext
    guessed = mimetypes.guess_extension(ctype, strict=False) if ctype else None
    return guessed or ".bin"


def _local_asset_ref(path: str) -> str:
    normalized = path.replace("\\", "/").strip().lstrip("./")
    if normalized.startswith("../assets/"):
        return normalized
    if normalized.startswith("assets/"):
        return "../" + normalized
    return normalized


def _rewrite_urls(value: Any, url_to_local: dict[str, str]) -> Any:
    if isinstance(value, str):
        return url_to_local.get(value.strip(), value)
    if isinstance(value, list):
        return [_rewrite_urls(item, url_to_local) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_urls(item, url_to_local) for key, item in value.items()}
    return value


def _tokens(value: str) -> set[str]:
    raw = urllib.parse.unquote(value)
    parts = re.split(r"[^a-zA-Z0-9]+", raw.lower())
    return {
        part
        for part in parts
        if len(part) >= 3 and part not in STOP_TOKENS and not part.isdigit()
    }


def _tokens_from_url(url: str) -> set[str]:
    filename = _extract_wikimedia_filename(url)
    if not filename:
        filename = Path(urllib.parse.urlparse(url).path).name
    return _tokens(Path(filename).stem)


def _build_local_asset_index(current_run_dir: Path) -> list[tuple[Path, set[str]]]:
    index: list[tuple[Path, set[str]]] = []
    runs_dir = ROOT / "data" / "runs"
    if not runs_dir.exists():
        return index
    current_assets = (current_run_dir / "assets").resolve()
    for path in runs_dir.glob("*/assets/*"):
        if not path.is_file():
            continue
        try:
            if path.parent.resolve() == current_assets:
                continue
        except Exception:
            pass
        if path.suffix.lower() not in ASSET_SUFFIXES:
            continue
        token_set = _tokens(path.stem)
        if token_set:
            index.append((path, token_set))
    return index


def _copy_local_fallback(
    url: str,
    assets_dir: Path,
    local_index: list[tuple[Path, set[str]]],
) -> dict[str, Any] | None:
    query_tokens = _tokens_from_url(url)
    if not query_tokens:
        return None

    best_path: Path | None = None
    best_score = 0.0
    best_overlap = 0
    for path, token_set in local_index:
        overlap = len(query_tokens & token_set)
        if overlap <= 0:
            continue
        score = overlap / max(1, len(query_tokens))
        if overlap > best_overlap or (overlap == best_overlap and score > best_score):
            best_path = path
            best_score = score
            best_overlap = overlap

    # Require enough overlap to avoid mapping product/place images to unrelated assets.
    if best_path is None or best_overlap < 2 or best_score < 0.4:
        return None

    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    ext = best_path.suffix.lower() if best_path.suffix.lower() in ASSET_SUFFIXES else ".bin"
    dest = assets_dir / f"{url_hash}_{_safe_name(best_path.stem)}{ext}"
    if not dest.exists():
        dest.write_bytes(best_path.read_bytes())
    data = dest.read_bytes()
    return {
        "url": url,
        "path": str(dest.relative_to(assets_dir.parent)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "source_path": str(best_path),
    }


def _extract_wikimedia_filename(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if not (host == "upload.wikimedia.org" or host.endswith(".upload.wikimedia.org") or host == "commons.wikimedia.org"):
        return None
    parts = [urllib.parse.unquote(item) for item in parsed.path.split("/") if item]
    if not parts:
        return None
    filename = parts[-2] if "thumb" in parts and len(parts) >= 2 else parts[-1]
    filename = re.sub(r"^\d+(?:px|p)x?-", "", filename, flags=re.IGNORECASE)
    if "." not in filename:
        return None
    return filename


def _wikimedia_search_url(original_url: str, timeout: int) -> str | None:
    filename = _extract_wikimedia_filename(original_url)
    if not filename:
        return None

    stem = Path(filename).stem
    query = re.sub(r"[_-]+", " ", stem).strip()
    if not query:
        return None

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": "1",
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": "1024",
    }
    api_url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "A2UI-DatasetBackfill/1.0 (local research dataset asset backfill)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - Wikimedia API
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None

    pages = payload.get("query", {}).get("pages", {})
    if not isinstance(pages, dict) or not pages:
        return None
    for page in pages.values():
        infos = page.get("imageinfo")
        if not isinstance(infos, list) or not infos:
            continue
        info = infos[0]
        thumb = info.get("thumburl")
        direct = info.get("url")
        if isinstance(thumb, str) and thumb.startswith("https://"):
            return thumb
        if isinstance(direct, str) and direct.startswith("https://"):
            return direct
    return None


def _content_valid(url: str, safe_name: str, content_type: str | None, data: bytes) -> tuple[bool, str]:
    declared = (content_type or "").split(";", 1)[0].strip().lower()
    if declared in {
        "application/javascript",
        "application/x-javascript",
        "text/javascript",
        "text/css",
        "text/plain",
    }:
        return False, f"downloaded non-media content ({declared})"
    prefix = data[:128].lstrip().lower()
    if prefix.startswith(b"/*!") or prefix.startswith(b"//") or prefix.startswith(b"function "):
        return False, "downloaded script/text instead of an asset"

    valid, reason = _downloaded_asset_content_valid(url, safe_name, content_type, data)
    if valid:
        return True, reason
    # Some CDNs negotiate WebP/AVIF while keeping a .png/.jpg URL. For a local
    # renderer asset, the actual image bytes matter more than the original suffix.
    sniffed = _sniff_asset_mime(data)
    if sniffed and sniffed.startswith("image/"):
        return True, "ok"
    if declared in {"image/avif"}:
        return True, "ok"
    return False, reason


def _download_one(
    url: str,
    assets_dir: Path,
    max_bytes: int,
    timeout: int,
    local_index: list[tuple[Path, set[str]]],
) -> tuple[str, dict[str, Any] | None, str | None]:
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    existing = sorted(assets_dir.glob(f"{url_hash}_*"))
    if existing:
        path = existing[0]
        data = path.read_bytes()
        return (
            url,
            {
                "url": url,
                "path": str(path.relative_to(assets_dir.parent)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            },
            None,
        )

    candidates = [url]
    wikimedia_alt = _wikimedia_search_url(url, timeout)
    if wikimedia_alt and wikimedia_alt not in candidates:
        candidates.append(wikimedia_alt)

    last_error: str | None = None
    selected_url = url
    safe_name = "asset"
    content_type: str | None = None
    data: bytes | None = None
    for candidate in candidates:
        parsed = urllib.parse.urlparse(candidate)
        basename = Path(parsed.path).name or Path(urllib.parse.urlparse(url).path).name or "asset"
        safe_name = _safe_name(basename)
        req = urllib.request.Request(
            candidate,
            headers={
                "User-Agent": "A2UI-DatasetBackfill/1.0 (local research dataset asset backfill)",
                "Accept": "image/avif,image/webp,image/apng,image/*,application/pdf,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - dataset media URLs
                content_type = resp.headers.get("Content-Type")
                data = resp.read(max_bytes + 1)
        except Exception as exc:
            last_error = str(exc)
            continue

        if len(data) > max_bytes:
            last_error = f"asset too large > {max_bytes} bytes"
            data = None
            continue

        valid, reason = _content_valid(candidate, safe_name, content_type, data)
        if not valid:
            last_error = reason
            data = None
            continue
        selected_url = candidate
        break

    if data is None:
        fallback = _copy_local_fallback(url, assets_dir, local_index)
        if fallback:
            return url, fallback, None
        return url, None, last_error or "download failed"

    ext = _asset_ext(selected_url, safe_name, content_type)
    stem = _safe_name(Path(safe_name).stem)
    dest = assets_dir / f"{url_hash}_{stem}{ext}"
    dest.write_bytes(data)
    asset: dict[str, Any] = {
        "url": url,
        "path": str(dest.relative_to(assets_dir.parent)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    if selected_url != url:
        asset["source_url"] = selected_url
    return (
        url,
        asset,
        None,
    )


def _canonicalize_asset_files(
    downloaded: dict[str, dict[str, Any]],
    assets_dir: Path,
) -> dict[str, int]:
    """Map duplicate asset bytes to one file and remove unreferenced copies."""
    by_sha: dict[str, list[Path]] = {}
    for asset in downloaded.values():
        rel_path = str(asset.get("path") or "")
        sha = str(asset.get("sha256") or "")
        if not rel_path or not sha:
            continue
        path = assets_dir.parent / rel_path
        if path.is_file():
            by_sha.setdefault(sha, []).append(path)

    canonical_by_sha: dict[str, Path] = {}
    removed = 0
    for sha, paths in by_sha.items():
        unique_paths = sorted({path.resolve() for path in paths}, key=lambda p: (len(p.name), p.name.lower()))
        if not unique_paths:
            continue
        canonical_by_sha[sha] = unique_paths[0]
        for duplicate in unique_paths[1:]:
            if duplicate == unique_paths[0]:
                continue
            try:
                duplicate.unlink()
                removed += 1
            except FileNotFoundError:
                pass

    referenced: set[Path] = set()
    for asset in downloaded.values():
        sha = str(asset.get("sha256") or "")
        canonical = canonical_by_sha.get(sha)
        if not canonical:
            continue
        rel = canonical.relative_to(assets_dir.parent)
        asset["path"] = str(rel)
        try:
            asset["bytes"] = canonical.stat().st_size
        except FileNotFoundError:
            pass
        referenced.add(canonical.resolve())

    for path in assets_dir.glob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in referenced:
            continue
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass

    physical_files = [path for path in assets_dir.glob("*") if path.is_file()]
    return {
        "physical_asset_files": len(physical_files),
        "physical_asset_bytes": sum(path.stat().st_size for path in physical_files),
        "deduped_asset_files_removed": removed,
    }


def _selected_response_ids(run_dir: Path, scope: str) -> set[str] | None:
    if scope == "all":
        return None
    genui_path = run_dir / "genui.jsonl"
    ids: set[str] = set()
    for row in _read_jsonl(genui_path):
        response_id = row.get("response_id")
        if isinstance(response_id, str) and response_id:
            ids.add(response_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill downloaded media assets for an existing dataset run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scope", choices=["genui", "all"], default="genui")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()

    run_dir = ROOT / "data" / "runs" / args.run_id
    responses_path = run_dir / "responses.jsonl"
    genui_path = run_dir / "genui.jsonl"
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    local_index = _build_local_asset_index(run_dir)

    responses = _read_jsonl(responses_path)
    genui_rows = _read_jsonl(genui_path) if genui_path.exists() else []
    selected_ids = _selected_response_ids(run_dir, args.scope)

    entries_by_response: dict[str, list[dict[str, str]]] = {}
    all_urls: set[str] = set()
    for row in responses:
        response_id = str(row.get("response_id") or "")
        if not response_id:
            continue
        if selected_ids is not None and response_id not in selected_ids:
            continue
        entries = _extract_asset_entries(str(row.get("response_text") or ""))
        if entries:
            entries_by_response[response_id] = entries
            all_urls.update(str(item["url"]) for item in entries if item.get("url"))

    urls = sorted(all_urls)
    print(
        f"Backfilling assets run={args.run_id} scope={args.scope} "
        f"responses={len(entries_by_response)} unique_urls={len(urls)} "
        f"local_fallback_assets={len(local_index)}"
    )

    downloaded: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_download_one, url, assets_dir, args.max_bytes, args.timeout, local_index): url
            for url in urls
        }
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            url, asset, error = future.result()
            if asset:
                downloaded[url] = asset
            elif error:
                failures[url] = error
            if idx % 100 == 0 or idx == len(futures):
                elapsed = time.time() - start
                print(
                    f"progress {idx}/{len(futures)} downloaded={len(downloaded)} "
                    f"failed={len(failures)} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    asset_file_stats = _canonicalize_asset_files(downloaded, assets_dir)

    response_assets_by_id: dict[str, list[dict[str, Any]]] = {}
    updated_responses = 0
    for row in responses:
        response_id = str(row.get("response_id") or "")
        entries = entries_by_response.get(response_id) or []
        if not entries:
            continue
        assets: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in entries:
            url = str(entry.get("url") or "")
            asset = downloaded.get(url)
            if not asset or url in seen:
                continue
            seen.add(url)
            assets.append(asset)
        response_assets_by_id[response_id] = assets

        row["assets"] = assets
        stats = dict(row.get("asset_stats") or {})
        stats["declared_asset_urls"] = len(entries)
        stats["downloaded_assets"] = len(assets)
        stats["asset_url_valid_rate"] = (len(assets) / len(entries)) if entries else 1.0
        stats["asset_quality_ok"] = len(entries) == len(assets)
        stats["asset_quality_reason"] = "ok" if len(entries) == len(assets) else "some asset URLs failed to download"
        row["asset_stats"] = stats
        updated_responses += 1

    _write_jsonl(responses_path, responses)

    updated_genui = 0
    for row in genui_rows:
        response_id = str(row.get("response_id") or "")
        old_asset_path_by_url: dict[str, str] = {}
        for old_asset in row.get("assets") or []:
            if not isinstance(old_asset, dict):
                continue
            old_url = str(old_asset.get("url") or "")
            old_path = str(old_asset.get("path") or "")
            if old_url and old_path:
                old_asset_path_by_url[old_url] = _local_asset_ref(old_path)

        assets = response_assets_by_id.get(response_id)
        if assets is None:
            continue
        row["assets"] = assets
        url_to_local: dict[str, str] = {}
        for item in assets:
            item_url = str(item.get("url") or "")
            item_path = str(item.get("path") or "")
            if not item_url or not item_path:
                continue
            local_ref = _local_asset_ref(item_path)
            url_to_local[item_url] = local_ref
            source_url = str(item.get("source_url") or "")
            if source_url:
                url_to_local[source_url] = local_ref
            old_ref = old_asset_path_by_url.get(item_url)
            if old_ref:
                url_to_local[old_ref] = local_ref
        if url_to_local and isinstance(row.get("genui_json"), dict):
            row["genui_json"] = _rewrite_urls(row["genui_json"], url_to_local)
            toon = encode_toon(row["genui_json"])
            row["toon"] = toon
            validation = dict(row.get("validation") or {})
            validation["toon_roundtrip_ok"] = roundtrip_ok(row["genui_json"], toon)
            row["validation"] = validation
            metrics = dict(row.get("metrics") or {})
            metrics["output_tokens_toon"] = count_characters(toon)
            metrics["output_chars_toon"] = count_characters(toon)
            metrics["output_tokens_json"] = count_characters(json.dumps(row["genui_json"], ensure_ascii=False))
            metrics["output_chars_json"] = metrics["output_tokens_json"]
            row["metrics"] = metrics
        updated_genui += 1

    if genui_rows:
        _write_jsonl(genui_path, genui_rows)

    report = {
        "run_id": args.run_id,
        "scope": args.scope,
        "responses_with_asset_entries": len(entries_by_response),
        "unique_urls": len(urls),
        "downloaded_unique_urls": len(downloaded),
        "failed_unique_urls": len(failures),
        "updated_responses": updated_responses,
        "updated_genui_rows": updated_genui,
        **asset_file_stats,
        "failures_sample": [
            {"url": url, "error": failures[url]}
            for url in sorted(failures)[:50]
        ],
    }
    (run_dir / "asset_backfill_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
