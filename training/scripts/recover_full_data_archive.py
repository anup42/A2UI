#!/usr/bin/env python3
"""Create a source-bound training candidate from the audited messages archive.

Repairs are limited to exact reversible text recovery, source-identity URL
masking, explicit label/media-token rebinding, mid-word chunk joining, and
removal of references that are absent from the source. Missing content is
never synthesized. Originals remain read-only and every decision is traced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING / "src"))

from ir_training.data.archive_recovery import recover_pair, text_sha256
from ir_training.data.express_preparation import TASK_PREFIX
from ir_training.data.ir_targets import A2UI_EXPRESS_V1, semantic_hash
from ir_training.data.shared_prompt import create_shared_prompt_contract

POLICY_VERSION = "messages-archive-recovery-v5-20260913"
REMOVAL_TRANSFORMATIONS = frozenset(
    {
        "generic_open_source_buttons_removed",
        "ungrounded_reference_elements_removed",
        "ungrounded_reference_fields_removed",
        "ungrounded_open_url_events_removed",
        "empty_table_columns_removed",
        "empty_tables_removed",
        "empty_layout_elements_removed",
    }
)
SOURCE_GROUNDED_TRANSFORMATIONS = REMOVAL_TRANSFORMATIONS | {
    "placeholder_label_rebindings",
    "placeholder_media_namespace_rebindings",
    "placeholder_unique_media_rebindings",
}
LOSSLESS_TRANSFORMATIONS = frozenset(
    {
        "source_cp437_utf8",
        "target_cp437_utf8",
        "source_identity_url_normalization",
        "midword_text_boundary_joins",
        "unused_target_url_map_entries_removed",
    }
)
CONFIRMED_DEFECTS = {
    ("train", 1),
    ("train", 3855),
    ("train", 24112),
    ("train", 40160),
    ("train", 93102),
    ("train", 111526),
    ("train", 121983),
    ("val", 18),
    ("val", 209),
}


def _init_worker() -> None:
    """Use the parity-tested compiled schema engine for the full archive pass."""
    audit_scripts = TRAINING / "scripts/audits"
    if str(audit_scripts) not in sys.path:
        sys.path.insert(0, str(audit_scripts))
    from full_data_ir_20260913 import wire_setup

    wire_setup(check_parity=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_line(stream, value: Any) -> None:
    stream.write(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    )


class SourceFamilies:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            previous = self.parent[value]
            self.parent[value] = root
            value = previous
        return root

    def union(self, values: Iterable[str | None]) -> None:
        roots = sorted({self.find(value) for value in values if value})
        if not roots:
            raise ValueError("Source family requires at least one hash")
        for root in roots[1:]:
            self.parent[root] = roots[0]


def _worker(batch):
    output = []
    for item in batch:
        result = recover_pair(item["source_text"], item["target_text"])
        record = {
            "split": item["split"],
            "line": item["line"],
            "accepted": result.accepted,
            "reason": result.reason,
            "warnings": list(result.warnings),
            "transformations": list(result.transformations),
            "repair_metrics": dict(result.repair_metrics),
            "effective_source_sha256": text_sha256(result.source_text),
            "effective_target_sha256": text_sha256(result.target_text),
            "source_transcode": asdict(result.source_transcode),
            "target_transcode": asdict(result.target_transcode),
            "url_map_entries": len(result.url_map),
        }
        if result.accepted:
            record.update(
                source_text=result.source_text,
                target_text=result.target_text,
                url_map=result.url_map,
                effective_semantic_sha256=semantic_hash(result.graph or {}),
            )
        output.append(record)
    return output


def _source_rows(db, source_dir: Path):
    handles = {
        split: (source_dir / f"{split}.jsonl").open("rb") for split in ("train", "val")
    }
    try:
        query = """SELECT split,line,byte_offset,byte_length,row_sha256,source_sha256,
          source_normalized_sha256,source_masked_sha256,target_sha256
          FROM rows ORDER BY CASE split WHEN 'train' THEN 0 ELSE 1 END,line"""
        for (
            split,
            line,
            offset,
            length,
            row_sha,
            source_sha,
            normalized_sha,
            masked_sha,
            target_sha,
        ) in db.execute(query):
            handle = handles[split]
            handle.seek(offset)
            raw = handle.read(length)
            row = json.loads(raw.decode("utf-16-le").removeprefix("\ufeff"))
            messages = row.get("messages") or []
            if (
                len(messages) < 2
                or messages[-2].get("role") != "user"
                or messages[-1].get("role") != "assistant"
            ):
                raise ValueError(f"Unexpected messages envelope at {split}:{line}")
            task, target = messages[-2].get("content"), messages[-1].get("content")
            if (
                not isinstance(task, str)
                or not task.startswith(TASK_PREFIX)
                or not isinstance(target, str)
            ):
                raise ValueError(f"Cannot bind final source/target at {split}:{line}")
            source = task[len(TASK_PREFIX) :]
            if text_sha256(source) != source_sha or text_sha256(target) != target_sha:
                raise ValueError(
                    f"Original archive differs from inventory at {split}:{line}"
                )
            yield {
                "split": split,
                "line": line,
                "row_sha256": row_sha,
                "source_sha256": source_sha,
                "source_normalized_sha256": normalized_sha,
                "source_masked_sha256": masked_sha,
                "target_sha256": target_sha,
                "source_text": source,
                "target_text": target,
            }
    finally:
        for handle in handles.values():
            handle.close()


def _batch(values, size):
    current = []
    for value in values:
        current.append(value)
        if len(current) == size:
            yield current
            current = []
    if current:
        yield current


def _prepared_row(
    original: dict[str, Any],
    recovered: dict[str, Any],
    family: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    coordinate = f"{original['split']}:{original['line']}"
    occurrence = f"archive-{original['split']}-{original['line']:06d}-{original['row_sha256'][:16]}"
    source_id = f"archive-source-family-sha256:{family}"
    messages = [
        *deepcopy(contract["scaffold"]["messages"]),
        {"role": "user", "content": TASK_PREFIX + recovered["source_text"]},
        {"role": "assistant", "content": recovered["target_text"]},
    ]
    changes = [
        {
            "kind": kind,
            "lossless": kind in LOSSLESS_TRANSFORMATIONS,
            "source_grounded": kind in SOURCE_GROUNDED_TRANSFORMATIONS,
        }
        for kind in recovered["transformations"]
    ]
    metadata = {
        "source_id": source_id,
        "query_id": None,
        "response_id": occurrence,
        "ui_id": occurrence,
        "intent": None,
        "intent_bucket": "unknown_archive",
        "assigned_split": original["split"],
        "source_format": A2UI_EXPRESS_V1,
        "target_format": A2UI_EXPRESS_V1,
        "archive_recovery": {
            "policy_version": POLICY_VERSION,
            "coordinate": coordinate,
            "identity_kind": "archive_derived_source_family_hash_not_original_generator_id",
            "original_row_sha256": original["row_sha256"],
            "original_source_sha256": original["source_sha256"],
            "original_target_sha256": original["target_sha256"],
            "effective_source_sha256": recovered["effective_source_sha256"],
            "effective_target_sha256": recovered["effective_target_sha256"],
            "effective_semantic_sha256": recovered["effective_semantic_sha256"],
            "transformations": recovered["transformations"],
            "repair_metrics": recovered.get("repair_metrics", {}),
            "source_transcode": recovered["source_transcode"],
            "target_transcode": recovered["target_transcode"],
            "missing_historical_metadata": [
                "original_query_id",
                "original_response_id",
                "generator_lineage",
                "original_url_map",
                "asset_manifest",
                "intent",
            ],
            "content_synthesized": False,
            "unsupported_target_content_removed": any(
                kind in REMOVAL_TRANSFORMATIONS for kind in recovered["transformations"]
            ),
            "stage3_run": False,
        },
        "shared_prompt": {
            "version": contract["version"],
            "contract_sha256": contract["contract_sha256"],
            "scaffold_sha256": contract["scaffold_sha256"],
        },
        "url_preprocessing": {
            "enabled": bool(recovered["url_map"]),
            "binding_policy": "source_identity",
            "source_closed": True,
            "url_map": recovered["url_map"],
        },
    }
    return {
        "id": occurrence,
        "row_id": occurrence,
        "response_id": occurrence,
        "source_id": source_id,
        "source_format": A2UI_EXPRESS_V1,
        "target_format": A2UI_EXPRESS_V1,
        "response_text": recovered["source_text"],
        "completion": recovered["target_text"],
        "messages": messages,
        "intent_bucket": "unknown_archive",
        "repair": {"applied": bool(changes), "changes": changes},
        "metadata": metadata,
    }


def _verify_inputs(args, manifest):
    checked = {}
    for split, expected in manifest["sources"].items():
        path = args.source_dir / f"{split}.jsonl"
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if digest != expected["sha256"] or before.st_size != expected["bytes"]:
            raise ValueError(f"{split} differs from the completed full-data audit")
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"{split} changed while it was being hashed")
        checked[split] = {
            "path": str(path.resolve()),
            "bytes": before.st_size,
            "rows": expected["rows"],
            "sha256": digest,
            "mtime_ns": before.st_mtime_ns,
        }
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=TRAINING / "outputs/audits/full_data_20260913",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=TRAINING / "reports/full_data_audit_20260913/audit_manifest.json",
    )
    parser.add_argument(
        "--near-duplicates",
        type=Path,
        default=TRAINING
        / "reports/full_data_audit_20260913/near_duplicates/pairs.json",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=TRAINING / "reports/offline_recovery_20260913",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write a new candidate dataset; originals remain read-only",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError("--workers must be between 1 and 32")
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    sources = _verify_inputs(args, manifest)
    implementation_paths = (
        Path(__file__).resolve(),
        TRAINING / "src/ir_training/data/archive_recovery.py",
        TRAINING / "src/ir_training/data/url_preprocess.py",
        ROOT / "dataset/schema/genuicraft_a2ui_v1_wire.schema.json",
        ROOT / "dataset/src/pipeline/ir_formats/express.py",
        ROOT / "dataset/src/pipeline/renderer_semantics.py",
    )
    plan = {
        "policy_version": POLICY_VERSION,
        "source_files": sources,
        "output_dir": str(args.output_dir.resolve()),
        "report_dir": str(args.report_dir.resolve()),
        "workers": args.workers,
        "actions": [
            "exact reversible whole-string CP437-bytes-as-UTF8 recovery",
            "source-identity URL normalization with exact source/graph restoration",
    "evidence-based exact-label and same-index media-namespace placeholder rebinding",
    "unique one-to-one same-kind media placeholder rebinding",
            "remove only source-absent reference fields/components and generic fallback source buttons",
            "remove non-root layout branches with no children, events, or data-bearing properties",
            "join adjacent legacy 180-character Text chunks only at an alphanumeric mid-word boundary",
            "strict Express validation and conservative content checks",
            "Golden/source-family isolation and effective-pair deduplication",
            "UTF-8/LF source-bound training candidate export",
        ],
        "not_performed": [
            "Stage 3",
            "model inference",
            "content synthesis",
            "guessed link-role/index aliasing",
            "target truncation",
            "GPU training",
        ],
        "model_specific_token_preflight_pending": True,
        "bulk_validator": "compiled unchanged production schema; independently parity-tested against production",
        "implementation_sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in implementation_paths
        },
        "runtime": {"python": sys.version},
    }
    if not args.execute:
        print(json.dumps({"status": "plan_only", **plan}, indent=2))
        return 0
    if args.output_dir.exists() or args.report_dir.exists():
        raise FileExistsError("Choose fresh --output-dir and --report-dir paths")
    for required in (
        args.audit_root / "inventory.sqlite",
        args.audit_root / "ir.sqlite",
        args.near_duplicates,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Missing completed audit input: {required}")

    inventory = sqlite3.connect(
        (args.audit_root / "inventory.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    inventory.row_factory = sqlite3.Row
    expected_rows = inventory.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
    if expected_rows != sum(item["rows"] for item in sources.values()):
        raise ValueError("Inventory and frozen source manifest row counts differ")
    families = SourceFamilies()
    coordinates = {}
    for row in inventory.execute(
        "SELECT split,line,source_sha256,source_normalized_sha256,source_masked_sha256 FROM rows"
    ):
        families.union(
            [
                row["source_sha256"],
                row["source_normalized_sha256"],
                row["source_masked_sha256"],
            ]
        )
        coordinates[(row["split"], row["line"])] = row["source_sha256"]
    near = json.loads(args.near_duplicates.read_text(encoding="utf-8"))
    for pair in near.get("nonexact", []):
        train_key, val_key = (
            ("train", int(pair["train_line"])),
            ("val", int(pair["val_line"])),
        )
        if (
            coordinates[train_key] != pair["train_source_sha256"]
            or coordinates[val_key] != pair["val_source_sha256"]
        ):
            raise ValueError("Near-duplicate evidence is stale")
        families.union([pair["train_source_sha256"], pair["val_source_sha256"]])
    golden_hashes = {
        row[0] for row in inventory.execute("SELECT DISTINCT digest FROM golden_hashes")
    }
    reserved_families = {families.find(value) for value in golden_hashes}
    train_families = {
        families.find(row[0])
        for row in inventory.execute(
            "SELECT source_sha256 FROM rows WHERE split='train'"
        )
    }

    partial = args.output_dir.with_name(
        args.output_dir.name + f".partial-{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    decisions_path = partial / "decisions.csv"
    quarantine_path = partial / "quarantine.csv"
    outputs = {
        split: (partial / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
        for split in ("train", "val")
    }
    decision_fields = [
        "split",
        "line",
        "category",
        "reason",
        "warnings",
        "transformations",
        "repair_metrics",
        "source_family",
        "original_source_sha256",
        "original_target_sha256",
        "effective_source_sha256",
        "effective_target_sha256",
        "effective_semantic_sha256",
        "duplicate_of",
        "url_map_entries",
    ]
    counts = defaultdict(Counter)
    warning_counts = defaultdict(Counter)
    transformation_counts = defaultdict(Counter)
    accepted_transformation_counts = defaultdict(Counter)
    repair_metric_counts = defaultdict(Counter)
    accepted_repair_metric_counts = defaultdict(Counter)
    unique_sources = defaultdict(set)
    seen_semantic: dict[tuple[str, str], str] = {}
    seen_effective: dict[tuple[str, str], str] = {}
    begin = time.monotonic()
    completed = 0
    contract = create_shared_prompt_contract(ordering="root-first")

    def record(original, recovered, decision_writer, quarantine_writer):
        nonlocal completed
        split, line = original["split"], original["line"]
        coordinate = f"{split}:{line}"
        family = families.find(original["source_sha256"])
        category, reason, duplicate_of = "QUARANTINE", recovered["reason"], ""
        if family in reserved_families:
            reason = "reserved_golden_family"
        elif split == "val" and family in train_families:
            reason = "validation_family_seen_in_original_training"
        elif (split, line) in CONFIRMED_DEFECTS:
            reason = "confirmed_content_defect"
        elif recovered["accepted"]:
            semantic_key = family, recovered["effective_semantic_sha256"]
            effective_key = (
                recovered["effective_source_sha256"],
                recovered["effective_target_sha256"],
            )
            if semantic_key in seen_semantic:
                reason, duplicate_of = (
                    "duplicate_source_semantic_target",
                    seen_semantic[semantic_key],
                )
            elif effective_key in seen_effective:
                reason, duplicate_of = (
                    "duplicate_effective_source_target",
                    seen_effective[effective_key],
                )
            else:
                category = "REPAIR" if recovered["transformations"] else "KEEP"
                if category == "KEEP":
                    reason = "unchanged_content_candidate"
                elif (
                    set(recovered["transformations"]) & SOURCE_GROUNDED_TRANSFORMATIONS
                ):
                    reason = "source_grounded_offline_repair"
                else:
                    reason = "lossless_representation_repair"
                seen_semantic[semantic_key] = coordinate
                seen_effective[effective_key] = coordinate
                write_json_line(
                    outputs[split], _prepared_row(original, recovered, family, contract)
                )
                unique_sources[split].add(family)
                unique_sources["combined"].add(family)
        result = {
            "split": split,
            "line": line,
            "category": category,
            "reason": reason,
            "warnings": "|".join(recovered.get("warnings") or []),
            "transformations": "|".join(recovered.get("transformations") or []),
            "repair_metrics": json.dumps(
                recovered.get("repair_metrics") or {},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "source_family": family,
            "original_source_sha256": original["source_sha256"],
            "original_target_sha256": original["target_sha256"],
            "effective_source_sha256": recovered.get("effective_source_sha256", ""),
            "effective_target_sha256": recovered.get("effective_target_sha256", ""),
            "effective_semantic_sha256": recovered.get("effective_semantic_sha256", ""),
            "duplicate_of": duplicate_of,
            "url_map_entries": recovered.get("url_map_entries", 0),
        }
        decision_writer.writerow(result)
        if category == "QUARANTINE":
            quarantine_writer.writerow(result)
        for scope in (split, "combined"):
            counts[scope][category] += 1
            counts[scope][f"reason:{reason}"] += 1
            warning_counts[scope].update(recovered.get("warnings") or [])
            transformation_counts[scope].update(recovered.get("transformations") or [])
            repair_metric_counts[scope].update(recovered.get("repair_metrics") or {})
            if category != "QUARANTINE":
                accepted_transformation_counts[scope].update(
                    recovered.get("transformations") or []
                )
                accepted_repair_metric_counts[scope].update(
                    recovered.get("repair_metrics") or {}
                )
        completed += 1
        if completed % 2048 == 0:
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "elapsed_seconds": round(time.monotonic() - begin, 1),
                        "categories": dict(counts["combined"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    try:
        with (
            decisions_path.open("w", encoding="utf-8", newline="") as decisions,
            quarantine_path.open("w", encoding="utf-8", newline="") as quarantine,
        ):
            decision_writer, quarantine_writer = (
                csv.DictWriter(decisions, decision_fields),
                csv.DictWriter(quarantine, decision_fields),
            )
            decision_writer.writeheader()
            quarantine_writer.writeheader()
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker
            ) as executor:
                pending, original_batches, finished, write_id = {}, {}, {}, 0
                source_iterator = _source_rows(inventory, args.source_dir)
                for submit_id, batch in enumerate(_batch(source_iterator, 16)):
                    future = executor.submit(_worker, batch)
                    pending[future] = submit_id
                    original_batches[submit_id] = batch
                    while len(pending) >= args.workers * 3:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for item in done:
                            finished[pending.pop(item)] = item.result()
                        while write_id in finished:
                            recovered_batch = finished.pop(write_id)
                            originals = original_batches.pop(write_id)
                            for original, recovered in zip(
                                originals, recovered_batch, strict=True
                            ):
                                record(
                                    original,
                                    recovered,
                                    decision_writer,
                                    quarantine_writer,
                                )
                            write_id += 1
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for item in done:
                        finished[pending.pop(item)] = item.result()
                    while write_id in finished:
                        recovered_batch = finished.pop(write_id)
                        originals = original_batches.pop(write_id)
                        for original, recovered in zip(
                            originals, recovered_batch, strict=True
                        ):
                            record(
                                original, recovered, decision_writer, quarantine_writer
                            )
                        write_id += 1
        for stream in outputs.values():
            stream.close()
        if completed != expected_rows:
            raise ValueError(f"Incomplete recovery: {completed} != {expected_rows}")
        if unique_sources["train"] & unique_sources["val"]:
            raise ValueError("Recovered train/validation families overlap")
        if (unique_sources["train"] | unique_sources["val"]) & reserved_families:
            raise ValueError("Recovered data intersects a reserved Golden family")
        for split, expected in sources.items():
            path = Path(expected["path"])
            if (path.stat().st_size, path.stat().st_mtime_ns, file_sha256(path)) != (
                expected["bytes"],
                expected["mtime_ns"],
                expected["sha256"],
            ):
                raise ValueError(f"Original {split} changed during recovery")
        artifact_hashes = {
            path.name: file_sha256(path) for path in partial.iterdir() if path.is_file()
        }
        summary = {
            **plan,
            "schema_version": 1,
            "status": "candidate_export_complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - begin, 2),
            "all_rows_reconciled": completed,
            "categories": {
                scope: {
                    key: counts[scope][key] for key in ("KEEP", "REPAIR", "QUARANTINE")
                }
                for scope in ("train", "val", "combined")
            },
            "first_reason_counts": {
                scope: {
                    key.removeprefix("reason:"): value
                    for key, value in counts[scope].items()
                    if key.startswith("reason:")
                }
                for scope in ("train", "val", "combined")
            },
            "content_warnings_overlap_not_additive": {
                scope: dict(warning_counts[scope])
                for scope in ("train", "val", "combined")
            },
            "attempted_transformations_overlap_not_additive": {
                scope: dict(transformation_counts[scope])
                for scope in ("train", "val", "combined")
            },
            "accepted_transformations_overlap_not_additive": {
                scope: dict(accepted_transformation_counts[scope])
                for scope in ("train", "val", "combined")
            },
            "attempted_repair_operation_counts": {
                scope: dict(repair_metric_counts[scope])
                for scope in ("train", "val", "combined")
            },
            "accepted_repair_operation_counts": {
                scope: dict(accepted_repair_metric_counts[scope])
                for scope in ("train", "val", "combined")
            },
            "accepted_unique_source_families": {
                scope: len(unique_sources[scope])
                for scope in ("train", "val", "combined")
            },
            "accepted_known_golden_or_cross_split_family_overlaps": 0,
            "outputs": artifact_hashes,
            "source_manifest_sha256": file_sha256(args.source_manifest),
            "near_duplicate_report_sha256": file_sha256(args.near_duplicates),
            "shared_prompt_contract": contract,
            "limitations": [
                "KEEP/REPAIR are candidates, not proof of complete semantic fidelity.",
                "REPAIR never adds missing content; it may remove source-absent references or rebind an exact source label.",
                "Existing symbolic references have no recovered destination map.",
                "Near-duplicate family evidence is bounded, not an exhaustive paraphrase search.",
                "Intent, original IDs, asset inventory, and generator provenance are unavailable.",
                "Exact E2B/270M tokenizer and chat-template preparation is intentionally pending.",
                "The existing validation remainder is isolated but not a newly sampled representative split.",
            ],
        }
        with (partial / "manifest.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            write_json_line(stream, summary)
        # Recompute after manifest publication; manifest intentionally does not hash itself.
        os.replace(partial, args.output_dir)
        args.report_dir.mkdir(parents=True, exist_ok=False)
        report_summary = deepcopy(summary)
        report_summary["output_dir"] = str(args.output_dir.resolve())
        (args.report_dir / "summary.json").write_text(
            json.dumps(report_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "categories": summary["categories"],
                    "accepted_unique_source_families": summary[
                        "accepted_unique_source_families"
                    ],
                    "output_dir": str(args.output_dir),
                    "report_dir": str(args.report_dir),
                },
                indent=2,
            )
        )
        return 0
    except BaseException:
        for stream in outputs.values():
            if not stream.closed:
                stream.close()
        raise
    finally:
        inventory.close()


if __name__ == "__main__":
    raise SystemExit(main())
