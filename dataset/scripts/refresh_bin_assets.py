from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import urllib.request


MIME_EXTENSION_MAP = {
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


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "asset"


def _resolve_extension(content_type: str | None) -> str | None:
    if not content_type:
        return None
    content_type = content_type.split(";")[0].strip().lower()
    ext = MIME_EXTENSION_MAP.get(content_type)
    if ext:
        return ext
    guessed = mimetypes.guess_extension(content_type, strict=False)
    return guessed


def refresh_bin_assets(run_dir: Path, timeout: int, max_bytes: int) -> int:
    responses_path = run_dir / "responses.jsonl"
    if not responses_path.exists():
        print(f"No responses.jsonl at {responses_path}")
        return 0

    updated = 0
    rows = []
    for line in responses_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assets = row.get("assets") or []
        for asset in assets:
            url = asset.get("url")
            path = asset.get("path")
            if not url or not path:
                continue
            if not str(path).lower().endswith(".bin"):
                continue
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "A2UI-Dataset/1.0"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    content_type = resp.headers.get("Content-Type")
                    data = resp.read(max_bytes + 1)
                if len(data) > max_bytes:
                    continue
                ext = _resolve_extension(content_type)
                if not ext or ext == ".bin":
                    continue
                old_path = run_dir / path
                stem = _safe_name(Path(path).stem)
                new_path = old_path.with_name(f"{stem}{ext}")
                new_path.parent.mkdir(parents=True, exist_ok=True)
                new_path.write_bytes(data)
                sha = hashlib.sha256(data).hexdigest()
                asset["path"] = str(new_path.relative_to(run_dir))
                asset["sha256"] = sha
                asset["bytes"] = len(data)
                updated += 1
                if old_path.exists():
                    try:
                        old_path.unlink()
                    except Exception:
                        pass
            except Exception:
                continue
        rows.append(row)

    if updated:
        responses_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="gemini_3")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max_bytes", type=int, default=25 * 1024 * 1024)
    args = parser.parse_args()

    run_dir = Path("dataset/data/runs") / args.run_id
    updated = refresh_bin_assets(run_dir, args.timeout, args.max_bytes)
    print(f"Updated {updated} .bin assets")


if __name__ == "__main__":
    main()
