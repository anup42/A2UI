#!/usr/bin/env python3
"""Read-only integrity verification for a recovered messages archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))

from ir_training.data.archive_recovery import (
    placeholder_tokens,
    text_sha256,
)
from ir_training.data.express_preparation import TASK_PREFIX, prepare_row


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def load_decisions(path: Path) -> tuple[dict[str, tuple[str, str]], Counter[str]]:
    decisions: dict[str, tuple[str, str]] = {}
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            coordinate = f"{row['split']}:{row['line']}"
            if coordinate in decisions:
                fail(f"Duplicate decision coordinate: {coordinate}")
            decisions[coordinate] = (row["category"], row["reason"])
            counts[row["category"]] += 1
    return decisions, counts


def verify_row(
    row: dict[str, Any], split: str, decisions: dict[str, tuple[str, str]]
) -> tuple[str, str, str, str]:
    metadata = row.get("metadata")
    recovery = metadata.get("archive_recovery") if isinstance(metadata, dict) else None
    if not isinstance(recovery, dict):
        fail(f"{split} row lacks archive_recovery metadata")
    coordinate = str(recovery.get("coordinate") or "")
    if coordinate not in decisions:
        fail(f"Output coordinate has no decision: {coordinate}")
    category, _ = decisions[coordinate]
    if category not in {"KEEP", "REPAIR"}:
        fail(f"Quarantined coordinate leaked into {split}: {coordinate}")
    refinement = metadata.get("archive_refinement")
    if isinstance(refinement, dict):
        if (
            metadata.get("assigned_split") != split
            or refinement.get("assigned_split") != split
        ):
            fail(f"Rebuilt split/metadata mismatch: {split} versus {coordinate}")
    elif not coordinate.startswith(split + ":"):
        fail(f"Split/coordinate mismatch: {split} versus {coordinate}")
    source, target = row.get("response_text"), row.get("completion")
    messages = row.get("messages")
    if not isinstance(source, str) or not isinstance(target, str):
        fail(f"Missing source/target text: {coordinate}")
    if (
        not isinstance(messages, list)
        or len(messages) < 2
        or messages[-2] != {"role": "user", "content": TASK_PREFIX + source}
        or messages[-1] != {"role": "assistant", "content": target}
    ):
        fail(f"Final message binding mismatch: {coordinate}")
    if text_sha256(source) != recovery.get("effective_source_sha256"):
        fail(f"Effective source hash mismatch: {coordinate}")
    if text_sha256(target) != recovery.get("effective_target_sha256"):
        fail(f"Effective target hash mismatch: {coordinate}")
    if not placeholder_tokens(target) <= placeholder_tokens(source):
        fail(f"Target placeholder absent from source: {coordinate}")
    repair = row.get("repair")
    repaired = bool(isinstance(repair, dict) and repair.get("applied"))
    if repaired != (category == "REPAIR"):
        fail(f"Decision/repair mismatch: {coordinate}")
    family = str(row.get("source_id") or "")
    semantic = str(recovery.get("effective_semantic_sha256") or "")
    if family != str(metadata.get("source_id") or "") or not family or not semantic:
        fail(f"Source family or semantic hash missing: {coordinate}")
    return coordinate, family, semantic, text_sha256(source + "\0" + target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--strict-sample-size", type=int, default=256)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional new JSON file written only after every verification passes",
    )
    args = parser.parse_args()
    if args.strict_sample_size < 0:
        fail("--strict-sample-size cannot be negative")
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate_export_complete":
        fail("Dataset manifest is not complete")
    for name, expected in manifest.get("outputs", {}).items():
        path = args.dataset_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            fail(f"Artifact hash mismatch: {name}")

    decisions, decision_counts = load_decisions(args.dataset_dir / "decisions.csv")
    assignments = {}
    if manifest.get("split_rebuild"):
        with (args.dataset_dir / "decisions.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            assignments = {
                f"{row['split']}:{row['line']}": row["assigned_split"]
                for row in csv.DictReader(stream)
            }
    expected_categories = manifest["categories"]["combined"]
    if dict(decision_counts) != {
        key: int(value) for key, value in expected_categories.items()
    }:
        fail("Decision counts differ from manifest categories")
    with (args.dataset_dir / "quarantine.csv").open(encoding="utf-8") as stream:
        quarantine_rows = sum(1 for _ in stream) - 1
    if quarantine_rows != decision_counts["QUARANTINE"]:
        fail("Quarantine CSV count differs from decisions")

    split_counts: Counter[str] = Counter()
    split_families: dict[str, set[str]] = {"train": set(), "val": set()}
    seen_family_semantic: set[tuple[str, str]] = set()
    seen_effective_pairs: set[str] = set()
    seen_coordinates: set[str] = set()
    sample: list[tuple[int, str, dict[str, Any]]] = []
    for split in ("train", "val"):
        path = args.dataset_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(f"Invalid JSON at {path}:{line_number}: {exc}")
                coordinate, family, semantic, effective_pair = verify_row(
                    row, split, decisions
                )
                if assignments and assignments.get(coordinate) != split:
                    fail(f"Output split differs from frozen decision: {coordinate}")
                if coordinate in seen_coordinates:
                    fail(f"Duplicate output coordinate: {coordinate}")
                seen_coordinates.add(coordinate)
                family_semantic = (family, semantic)
                if family_semantic in seen_family_semantic:
                    fail(f"Duplicate source-family/semantic target: {coordinate}")
                if effective_pair in seen_effective_pairs:
                    fail(f"Duplicate effective source/target: {coordinate}")
                seen_family_semantic.add(family_semantic)
                seen_effective_pairs.add(effective_pair)
                split_families[split].add(family)
                split_counts[split] += 1
                if args.strict_sample_size:
                    priority = int.from_bytes(
                        hashlib.sha256(coordinate.encode()).digest()[:8]
                    )
                    item = (-priority, coordinate, row)
                    if len(sample) < args.strict_sample_size:
                        heapq.heappush(sample, item)
                    elif item > sample[0]:
                        heapq.heapreplace(sample, item)
    if split_families["train"] & split_families["val"]:
        fail("Recovered train/validation source families overlap")
    for split in ("train", "val"):
        expected = (
            manifest["output_rows"][split]
            if manifest.get("split_rebuild")
            else sum(manifest["categories"][split][key] for key in ("KEEP", "REPAIR"))
        )
        if split_counts[split] != expected:
            fail(
                f"{split} JSONL count differs from manifest: {split_counts[split]} != {expected}"
            )
    if sum(decision_counts.values()) != manifest.get("all_rows_reconciled"):
        fail("All-row reconciliation differs from decision count")
    if seen_coordinates != {
        key
        for key, (category, _) in decisions.items()
        if category in {"KEEP", "REPAIR"}
    }:
        fail("Output coordinates differ from accepted decisions")

    strict_failures = []
    for _, coordinate, row in sorted(sample, reverse=True):
        try:
            _, checked, _ = prepare_row(row, "root-first")
            if manifest.get("split_rebuild"):
                from ir_training.data.archive_refinement import (
                    repair_exact_text,
                    review_warnings,
                )

                repair_function = repair_exact_text
                if manifest.get("schema_version", 1) >= 3:
                    from ir_training.data.archive_final_review import (
                        paragraph_gaps,
                        repair_final_text,
                    )

                    repair_function = repair_final_text
                _, pending_repairs = repair_function(
                    row["response_text"], checked.graph
                )
                warnings, _, _ = review_warnings(
                    row["response_text"], checked.text, checked.graph
                )
                if manifest.get("schema_version", 1) >= 3 and paragraph_gaps(
                    row["response_text"], checked.graph
                ):
                    warnings.append("severely_missing_prose_block")
                if manifest.get("schema_version", 1) >= 4:
                    from ir_training.data.archive_letter_review import letter_gaps

                    if letter_gaps(row["response_text"], checked.graph):
                        warnings.append("incomplete_letter")
                if pending_repairs or warnings:
                    fail(
                        f"Refined sample has residual repairs/warnings: {coordinate}: {pending_repairs}, {warnings}"
                    )
        except (KeyError, RecursionError, TypeError, ValueError) as exc:
            strict_failures.append(
                {"coordinate": coordinate, "error": f"{type(exc).__name__}: {exc}"}
            )
    if strict_failures:
        fail(
            f"Production preparation failed for deterministic sample: {strict_failures[:3]}"
        )
    result = {
        "status": "verified",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "policy_version": manifest.get("policy_version"),
        "all_original_rows": len(decisions),
        "categories": dict(decision_counts),
        "output_rows": dict(split_counts),
        "strict_sample_rows": len(sample),
        "train_val_family_overlap": 0,
        "duplicate_family_semantic_targets": 0,
        "duplicate_effective_pairs": 0,
        "target_placeholders_absent_from_source": 0,
    }
    if manifest.get("split_rebuild"):
        result["independent_content_review_sample_rows"] = len(sample)
        result["residual_same_policy_repairs_in_sample"] = 0
    if args.report:
        if args.report.exists():
            fail(f"Refusing to overwrite verification report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
