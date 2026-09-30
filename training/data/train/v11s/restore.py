#!/usr/bin/env python3
"""Restore byte-preserving A2UI training splits using Python 3.11 or newer.

Checksums prove transport integrity, not factual quality or model readiness.
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
MAX_RAW_PART = 64 * BLOCK
MAX_COMPRESSED_PART = 80 * BLOCK
SIDECARS = ("manifest.json", "prompt_scaffold.json")


def _integer(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)


def _verify_file(bundle, item, *, maximum=None):
    if not _integer(item.get("bytes"), positive=True) or not _digest(item.get("sha256")):
        raise ValueError("Invalid file size or SHA-256")
    if maximum is not None and item["bytes"] > maximum:
        raise ValueError("File exceeds permitted size")
    path = bundle / item["name"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError(f"Missing, linked or wrong-sized file: {item['name']}")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
            raise ValueError(f"File checksum mismatch: {item['name']}")


def load_manifest(bundle):
    bundle_manifest = bundle / "bundle.json"
    if bundle_manifest.is_symlink() or bundle_manifest.stat().st_size > 4 * BLOCK:
        raise ValueError("Unsafe or oversized bundle manifest")
    manifest = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "a2ui-training-splits-gzip-v2":
        raise ValueError("Unsupported dataset bundle format")
    files = manifest.get("files")
    if not isinstance(files, list) or any(not isinstance(item, dict) for item in files) or [item.get("name") for item in files] != ["train.jsonl", "val.jsonl"]:
        raise ValueError("Bundle must contain train.jsonl and val.jsonl, in that order")
    for item in files:
        if not _integer(item.get("bytes"), positive=True) or not _integer(item.get("rows"), positive=True) or not _digest(item.get("sha256")):
            raise ValueError("Invalid split size, row count or SHA-256")
        parts = item.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("Split has no compressed parts")
        for number, part in enumerate(parts):
            if not isinstance(part, dict):
                raise ValueError("Invalid compressed part")
            expected = f"{Path(item['name']).stem}.part-{number:05d}.gz"
            if part.get("name") != expected:
                raise ValueError("Unsafe or out-of-order part name")
            if not _integer(part.get("bytes"), positive=True) or part["bytes"] > MAX_COMPRESSED_PART or not _digest(part.get("sha256")):
                raise ValueError("Invalid compressed part metadata")
            if not _integer(part.get("raw_bytes"), positive=True) or part["raw_bytes"] > MAX_RAW_PART:
                raise ValueError("Invalid decompressed part size")
            path = bundle / expected
            if path.is_symlink() or not path.is_file() or path.stat().st_size != part["bytes"]:
                raise ValueError(f"Missing, linked or wrong-sized part: {expected}")
        if sum(part["raw_bytes"] for part in parts) != item["bytes"]:
            raise ValueError("Part sizes do not match split size")
    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, list) or any(not isinstance(item, dict) for item in sidecars) or [item.get("name") for item in sidecars] != list(SIDECARS):
        raise ValueError("Invalid source sidecars")
    for item in sidecars:
        _verify_file(bundle, item, maximum=4 * BLOCK)
    source = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if source.get("dataset") != manifest.get("dataset"):
        raise ValueError("Dataset differs from source manifest")
    for item in files:
        source_output = source.get("outputs", {}).get(item["name"], {})
        if source_output.get("bytes") != item["bytes"] or source_output.get("sha256") != item["sha256"] or source.get("rows", {}).get(Path(item["name"]).stem) != item["rows"]:
            raise ValueError(f"Split differs from source manifest: {item['name']}")
    return manifest


def _copy_split(bundle, item, target=None):
    digest, size, rows, last = hashlib.sha256(), 0, 0, b""
    output = target.open("xb") if target is not None else nullcontext(None)
    with output as stream:
        for index, part in enumerate(item["parts"], 1):
            path = bundle / part["name"]
            print(f"{item['name']}: verify/restore part {index}/{len(item['parts'])}", flush=True)
            with path.open("rb") as raw:
                if hashlib.file_digest(raw, "sha256").hexdigest() != part["sha256"]:
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
    required = sum(item["bytes"] for item in manifest["files"] + manifest["sidecars"]) + 256 * BLOCK
    if shutil.disk_usage(output.parent).free < required:
        raise OSError(f"At least {required:,} free bytes required for restoration")
    partial = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        for item in manifest["files"]:
            _copy_split(bundle, item, partial / item["name"])
        for item in manifest["sidecars"]:
            shutil.copyfile(bundle / item["name"], partial / item["name"])
            _verify_file(partial, item)
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
