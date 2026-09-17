#!/usr/bin/env python3
"""Restore the published training-only v10 splits without third-party packages.

Original bytes and row metadata are preserved. This verifies transport integrity,
not model readiness or factual quality; normal training preparation still runs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
BLOCK = 1024 * 1024
MAX_PART = 40 * BLOCK


def _integer(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)


def load_manifest(bundle):
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "a2ui-training-splits-gzip-v1":
        raise ValueError("Unsupported dataset bundle format")
    files = manifest.get("files")
    if not isinstance(files, list) or [item.get("name") for item in files] != ["train.jsonl", "val.jsonl"]:
        raise ValueError("Bundle must contain only train.jsonl and val.jsonl, in that order")
    for item in files:
        if not _integer(item.get("bytes"), positive=True) or not _integer(item.get("rows"), positive=True) or not _digest(item.get("sha256")):
            raise ValueError("Invalid split size, row count or SHA-256")
        parts = item.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("Split has no compressed parts")
        for number, part in enumerate(parts):
            expected = f"{Path(item['name']).stem}.part-{number:05d}.gz"
            if part.get("name") != expected:
                raise ValueError("Unsafe or out-of-order part name")
            if not _integer(part.get("bytes"), positive=True) or part["bytes"] > MAX_PART or not _digest(part.get("sha256")):
                raise ValueError("Invalid compressed part metadata")
            if not _integer(part.get("raw_bytes"), positive=True) or part["raw_bytes"] > MAX_PART:
                raise ValueError("Invalid decompressed part size")
            path = bundle / expected
            if path.is_symlink() or not path.is_file() or path.stat().st_size != part["bytes"]:
                raise ValueError(f"Missing, linked or wrong-sized part: {expected}")
        if sum(part["raw_bytes"] for part in parts) != item["bytes"]:
            raise ValueError("Part sizes do not match split size")
    return manifest


def _copy_split(bundle, item, target=None):
    digest, size, rows, last = hashlib.sha256(), 0, 0, b""
    output = target.open("xb") if target is not None else nullcontext(None)
    with output as stream:
        for index, part in enumerate(item["parts"], 1):
            path = bundle / part["name"]
            print(f"{item['name']}: verify/restore part {index}/{len(item['parts'])}", flush=True)
            with path.open("rb") as raw:
                compressed = hashlib.file_digest(raw, "sha256").hexdigest()
                if compressed != part["sha256"]:
                    raise ValueError(f"Compressed checksum mismatch: {path.name}")
                raw.seek(0)
                part_size = 0
                with gzip.GzipFile(fileobj=raw, mode="rb") as decoded:
                    while block := decoded.read(min(BLOCK, part["raw_bytes"] - part_size + 1)):
                        part_size += len(block)
                        if part_size > part["raw_bytes"]:
                            raise ValueError(f"Expanded size exceeds manifest: {path.name}")
                        digest.update(block)
                        size += len(block)
                        rows += block.count(b"\n")
                        last = block[-1:]
                        if stream is not None:
                            stream.write(block)
                if part_size != part["raw_bytes"]:
                    raise ValueError(f"Expanded size mismatch: {path.name}")
        if last != b"\n":
            rows += 1
        if size != item["bytes"] or rows != item["rows"] or digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Restored bytes, rows or checksum differ: {item['name']}")
    print(f"Verified {item['name']}: {rows:,} rows; SHA-256 {digest.hexdigest()}", flush=True)


def restore(bundle=BUNDLE, output=None, *, verify_only=False):
    bundle = Path(bundle).resolve()
    manifest = load_manifest(bundle)
    if verify_only:
        for item in manifest["files"]:
            _copy_split(bundle, item)
        return manifest
    if output is None:
        raise ValueError("Use --output-dir for restoration, or --verify-only")
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Fresh output directory required; nothing overwritten: {output}")
    if output.resolve().is_relative_to(bundle) or bundle.is_relative_to(output.resolve()):
        raise ValueError("Output must be separate from the checked-in bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(item["bytes"] for item in manifest["files"]) + 256 * BLOCK
    if shutil.disk_usage(output.parent).free < required:
        raise OSError(f"At least {required:,} free bytes required for restoration")
    partial = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        for item in manifest["files"]:
            _copy_split(bundle, item, partial / item["name"])
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output appeared during restoration: {output}")
        partial.rename(output)
    except Exception:
        print(f"Incomplete files retained at {partial}; do not use for training.", file=sys.stderr)
        raise
    print(f"Ready for --input-dir {output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        restore(output=args.output_dir, verify_only=args.verify_only)
    except (OSError, ValueError, EOFError) as exc:
        print(f"Dataset restore failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
