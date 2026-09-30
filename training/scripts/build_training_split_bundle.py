#!/usr/bin/env python3
"""Publish source-manifest-bound training splits as byte-preserving gzip parts.

No parsing, repair, filtering or tokenizer preparation is performed. Memory use
is bounded by a 1 MiB streaming block; the restored JSONL bytes remain identical.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

BLOCK = 1024 * 1024
MAX_RAW_PART = 64 * BLOCK
MAX_COMPRESSED_PART = 80 * BLOCK


# Each published bundle is self-contained: it can be restored after cloning
# without importing the training package or installing third-party libraries.
RESTORE_SOURCE = r'''#!/usr/bin/env python3
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
'''


def _metadata(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest}


def _build_split(source: Path, destination: Path, expected: dict, expected_rows: int, part_bytes: int) -> dict:
    digest, size, rows, last = hashlib.sha256(), 0, 0, b""
    parts = []
    with source.open("rb") as stream:
        remaining = source.stat().st_size
        while remaining:
            path = destination / f"{source.stem}.part-{len(parts):05d}.gz"
            raw_bytes = 0
            with path.open("xb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=6) as encoded:
                    while raw_bytes < part_bytes and (block := stream.read(min(BLOCK, part_bytes - raw_bytes))):
                        encoded.write(block)
                        raw_bytes += len(block)
                        size += len(block)
                        digest.update(block)
                        rows += block.count(b"\n")
                        last = block[-1:]
            if raw_bytes == 0:
                raise ValueError(f"Source shrank during bundling: {source.name}")
            part = _metadata(path)
            if part["bytes"] > MAX_COMPRESSED_PART:
                raise ValueError(f"Compressed part exceeds size limit: {path.name}")
            part["raw_bytes"] = raw_bytes
            parts.append(part)
            remaining -= raw_bytes
            print(f"{source.name}: wrote part {len(parts)}; {size:,} source bytes", flush=True)
        if stream.read(1):
            raise ValueError(f"Source grew during bundling: {source.name}")
    if last and last != b"\n":
        rows += 1
    result = {"name": source.name, "bytes": size, "rows": rows, "sha256": digest.hexdigest(), "parts": parts}
    if size != expected.get("bytes") or digest.hexdigest() != expected.get("sha256") or rows != expected_rows:
        raise ValueError(f"Source bytes, rows or checksum differ from manifest: {source.name}")
    return result


def _readme(dataset: str, rows: dict, byte_size: int, source_manifest: dict) -> str:
    relationship = """v11 combines the retained/repaired v10 base and
newly retained sources; v11s is the new-source subset. **Do not concatenate v11
and v11s:** every v11s split is already included in the corresponding v11 split.
v11s validation is the new-only holdout within v11 validation. v11s training is
not an unseen evaluation set for a v11-trained model.""" if dataset in {"v11", "v11s"} else "Review the source manifest's source-family and holdout policy before combining datasets or treating a cohort as unseen."
    review = """A subsequent [generation review](../../../docs/dataset_v11_generation_review_20261001.md)
identified confirmed unsafe text joins and unsupported Chart rows. Normal
training preparation can exclude these reviewed defects while preserving the
original release bytes. The row counts in this README and source manifest are
release counts, not the number admitted by training preparation.

""" if dataset in {"v11", "v11s"} else ""
    return f'''# {dataset} training dataset bundle

This publishes **{rows["train"]:,} training** and **{rows["val"]:,} validation**
records as ordered gzip parts. The original JSONL bytes, row metadata and split
order are preserved; packaging performs no repair, filtering or regeneration.
The supplied `manifest.json` and `prompt_scaffold.json` are preserved byte for
byte and are verified during restoration.

## Restore and train

Use Python 3.11 or newer; no extra packages or Git LFS are required. Restore to a
fresh directory with at least {(byte_size + 256 * BLOCK) / 1_000_000_000:.2f} GB free:

```bash
python training/data/train/{dataset}/restore.py --output-dir /data/a2ui_{dataset}
```

On Windows, supply a Windows output path. Restoration verifies each compressed
part plus the complete original split SHA-256, byte size and row count. Existing
directories are refused. An unsuccessful restore retains a separate partial
directory for diagnosis; only a fully verified copy receives the requested name.
Verify without writing an extracted copy:

```bash
python training/data/train/{dataset}/restore.py --verify-only
```

Use the existing [training and deployment command](../../../docs/GOLDEN_GPU_DEPLOYMENT.md)
with `--input-dir /data/a2ui_{dataset}` and a fresh `--output-dir`. Model-specific
tokenizer preparation must run; do not pass this copy as `--prepared-input-dir`.
Preparation still applies holdout exclusion, validation and tokenizer checks.
Training, checkpoint testing and exports can use `--skip-litert-evaluation`
when native inference is unavailable; the model and exporter environment is
still required. Golden evaluation data is already tracked elsewhere.

## Relationship and quality limits

Source: the supplied `{dataset}` splits, policy `{source_manifest.get("policy", "unspecified")}`,
status `{source_manifest.get("status", "unspecified")}`. {relationship}

{review}Source manifests report offline repair and verification, with tokenizer
preparation deferred and no records truncated. These checks do not certify
source factual correctness, Android rendering or improved trained-model quality.
This publication is an offline candidate and requires normal training preparation
and evaluation. The source manifest references audit files outside this bundle;
those paths are provenance, not portable local prerequisites. The full audit
ledgers and source-generation archive are not duplicated here.
'''


def build_bundle(source_dir: Path, output_dir: Path, *, part_bytes: int = MAX_RAW_PART) -> dict:
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).absolute()
    if not isinstance(part_bytes, int) or isinstance(part_bytes, bool) or not 1 <= part_bytes <= MAX_RAW_PART:
        raise ValueError("Part size must be between 1 byte and 64 MiB")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Fresh output directory required: {output_dir}")
    if output_dir.resolve().is_relative_to(source_dir) or source_dir.is_relative_to(output_dir.resolve()):
        raise ValueError("Output must be separate from source directory")
    sidecar_names = ("manifest.json", "prompt_scaffold.json")
    for name in ("train.jsonl", "val.jsonl", *sidecar_names):
        path = source_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Source must be a regular file: {path}")
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = source_manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset.replace("_", "").isalnum():
        raise ValueError("Source manifest requires a simple dataset name")
    rows = source_manifest.get("rows", {})
    outputs = source_manifest.get("outputs", {})
    for name in ("train.jsonl", "val.jsonl"):
        count = rows.get(Path(name).stem)
        expected = outputs.get(name, {})
        if type(count) is not int or count <= 0 or type(expected.get("bytes")) is not int or expected["bytes"] <= 0:
            raise ValueError(f"Source manifest lacks positive split bytes/rows: {name}")
        if (source_dir / name).stat().st_size != expected["bytes"]:
            raise ValueError(f"Source byte size differs from manifest: {name}")
        digest = expected.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Source manifest lacks valid split SHA-256: {name}")
    sidecars = [_metadata(source_dir / name) for name in sidecar_names]
    if any(not 0 < item["bytes"] <= 4 * BLOCK for item in sidecars):
        raise ValueError("Source sidecars must be nonempty and at most 4 MiB")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    try:
        files = [_build_split(source_dir / name, partial, outputs[name], rows[Path(name).stem], part_bytes) for name in ("train.jsonl", "val.jsonl")]
        for item in sidecars:
            shutil.copyfile(source_dir / item["name"], partial / item["name"])
            if _metadata(partial / item["name"]) != item:
                raise ValueError(f"Source sidecar changed during bundling: {item['name']}")
        bundle = {"format": "a2ui-training-splits-gzip-v2", "dataset": dataset,
                  "scope": "training_inputs_and_source_sidecars", "quality_status": source_manifest.get("status"),
                  "source_manifest_sha256": sidecars[0]["sha256"], "source_policy": source_manifest.get("policy"),
                  "raw_part_bytes": part_bytes, "files": files, "sidecars": sidecars}
        (partial / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        (partial / "restore.py").write_text(RESTORE_SOURCE, encoding="utf-8", newline="\n")
        (partial / "README.md").write_text(_readme(dataset, rows, sum(item["bytes"] for item in files + sidecars), source_manifest), encoding="utf-8", newline="\n")
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Output appeared during bundling: {output_dir}")
        partial.rename(output_dir)
    except Exception:
        # Only the fresh, known staging directory created above is removed.
        if partial.resolve().parent != output_dir.parent.resolve() or not partial.name.startswith(f".{output_dir.name}.building-"):
            raise RuntimeError(f"Unexpected staging path; refusing recursive cleanup: {partial}")
        shutil.rmtree(partial)
        raise
    print(f"Published {dataset}: {rows['train']:,} train; {rows['val']:,} validation; {output_dir}", flush=True)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part-mib", type=int, default=64)
    args = parser.parse_args()
    build_bundle(args.source_dir, args.output_dir, part_bytes=args.part_mib * BLOCK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
