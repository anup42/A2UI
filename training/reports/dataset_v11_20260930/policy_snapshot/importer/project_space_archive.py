#!/usr/bin/env python3
"""Verify a Space tar.gz and project training fields without altering targets.

The complete archive is the immutable raw evidence. Historical subdirectories,
assets, logs, and repeated metric evidence stay in that archive. Only current
top-level stage files enter the training import. No archive path is extracted
with tar's filesystem extraction routines.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

FIELDS = "ui_id response_id query_id intent tags intent_bucket response_text record_status source_format target_format semantic_hash canonical_graph_hash a2ui_express completion canonical_graph validation gen created_at phase_invocation_id reference_map reference_source_sha256 source_quality query_quality scenario_family_id training_acceptance asset_downloads_required_for_training".split()
ROOT_FILES = {"queries.jsonl", "responses.jsonl", "genui.jsonl", "run_manifest.json", "aggregates.json", "generation_timing.jsonl"}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project(archive: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    before = archive.stat()
    inventory, selected, run_stats = [], {}, {}
    started = time.monotonic()
    with archive.open("rb") as raw, gzip.GzipFile(fileobj=raw) as gz:
        with tarfile.open(fileobj=gz, mode="r|") as tar:
            for member in tar:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or not re.fullmatch(r"dataset_muse_glimmer_100k_r\d+(?:_\d+)?", path.parts[0]):
                    raise ValueError(f"Unexpected archive path: {member.name}")
                current = len(path.parts) == 2 and path.name in ROOT_FILES
                inventory.append({"path": member.name, "bytes": member.size, "type": member.type.decode("ascii"), "projected": current})
                if not current:
                    continue
                if not member.isfile() or member.name in selected:
                    raise ValueError(f"Nonregular or duplicate current source: {member.name}")
                dest = partial / path.parts[0]
                dest.mkdir(exist_ok=True)
                digest, rows, invalid = hashlib.sha256(), 0, 0
                stats = run_stats.setdefault(path.parts[0], {})
                counts = Counter()
                source = tar.extractfile(member)
                assert source is not None
                target = dest / ("genui.training.jsonl" if path.name == "genui.jsonl" else path.name)
                with source, target.open("xb") as sink:
                    if path.name == "genui.jsonl":
                        for number, line in enumerate(source, 1):
                            digest.update(line)
                            if not line.strip():
                                continue
                            rows += 1
                            try:
                                row = json.loads(line)
                                if not isinstance(row, dict):
                                    raise ValueError("record is not an object")
                                item = {key: row[key] for key in FIELDS if key in row}
                                metrics = row.get("metrics") or {}
                                item["quality_metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}
                                counts[f"record_status:{row.get('record_status', 'missing')}"] += 1
                                counts[f"eligible:{(row.get('training_acceptance') or {}).get('eligible', 'missing')}"] += 1
                            except (ValueError, UnicodeError) as exc:
                                invalid += 1
                                item = {"import_error": str(exc)[:400]}
                            item["archive_record"] = {"member": member.name, "line": number, "sha256": hashlib.sha256(line).hexdigest()}
                            sink.write((json.dumps(item, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
                            if number % 2000 == 0:
                                print(f"Project {member.name}: {number:,} records", flush=True)
                    else:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                            rows += chunk.count(b"\n")
                            sink.write(chunk)
                entry = {"raw_bytes": member.size, "raw_sha256": digest.hexdigest(), "rows": rows if path.suffix == ".jsonl" else None, "invalid_json_rows": invalid, "projected_path": target.relative_to(partial).as_posix(), "projected_bytes": target.stat().st_size, "projected_sha256": sha_file(target)}
                selected[member.name] = entry
                stats[path.name] = {**entry, "counts": dict(counts)}
                print(f"Verified/projected {member.name}: {member.size:,} raw bytes; {rows:,} lines", flush=True)
        # Tar stops at its end marker. Drain gzip to verify the CRC and trailer.
        while gz.read(1024 * 1024):
            pass
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Archive changed during projection")
    result = {"status": "verified", "archive": str(archive.resolve()), "archive_bytes": after.st_size, "archive_sha256": sha_file(archive), "gzip_crc_verified": True, "archive_members": len(inventory), "archive_uncompressed_member_bytes": sum(x["bytes"] for x in inventory), "selected": selected, "runs": run_stats, "elapsed_seconds": round(time.monotonic() - started, 2), "policy": "Only current root Stage 1/2/3 files; historical backups never become additional samples; target and source values unmodified."}
    dump(partial / "archive_members.json", inventory)
    dump(partial / "archive_manifest.json", result)
    partial.rename(output)
    print(json.dumps({k: v for k, v in result.items() if k not in {"selected", "runs"}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project(args.archive, args.output_dir)
