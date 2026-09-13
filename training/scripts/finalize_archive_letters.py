#!/usr/bin/env python3
"""Filter incomplete letters into a fresh final archive copy; preserve inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
from ir_training.data.archive_letter_review import LETTER_START, POLICY, letter_gaps
from ir_training.data.express_preparation import _api
from recover_full_data_archive import file_sha256
from refine_recovered_archive import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "plan_only",
                    "policy_version": POLICY,
                    "output_dir": str(args.output_dir),
                },
                indent=2,
            )
        )
        return 0
    for path in (args.output_dir, args.report_dir):
        if path.exists():
            raise FileExistsError(f"Fresh destination required: {path}")
    manifest_path = args.base_dir / "manifest.json"
    base = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        base.get("schema_version") != 3
        or base.get("status") != "candidate_export_complete"
    ):
        raise ValueError("Requires the completed v7 intermediate")
    snapshots = {str(manifest_path.resolve()): file_sha256(manifest_path)}
    for details in base["source_files"].values():
        original = Path(details["path"])
        if file_sha256(original) != details["sha256"]:
            raise ValueError(f"Original input hash differs: {original}")
        snapshots[str(original.resolve())] = details["sha256"]
    for name, expected in base["outputs"].items():
        path = args.base_dir / name
        if file_sha256(path) != expected:
            raise ValueError(f"Input hash differs: {path}")
        snapshots[str(path.resolve())] = expected
    partial = args.output_dir.with_name(
        args.output_dir.name + f".partial-{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    active, *_ = _api()
    removed, counts, families = {}, Counter(), defaultdict(set)
    letter_counts = Counter()
    for split in ("train", "val"):
        with (
            (args.base_dir / f"{split}.jsonl").open(encoding="utf-8") as source_stream,
            (partial / f"{split}.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as output,
        ):
            for line in source_stream:
                row = json.loads(line)
                source = row["response_text"]
                if LETTER_START.search(source):
                    letter_counts["reviewed"] += 1
                    gaps = letter_gaps(
                        source, active.decode_express_completion(row["completion"])
                    )
                    if gaps:
                        removed[row["metadata"]["archive_recovery"]["coordinate"]] = (
                            gaps
                        )
                        letter_counts[f"excluded_{split}"] += 1
                        continue
                    letter_counts[f"accepted_{split}"] += 1
                output.write(line)
                counts[split] += 1
                families[split].add(row["source_id"])
                if sum(counts.values()) % 25000 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "final_letter_review",
                                "retained": sum(counts.values()),
                                "letters": dict(letter_counts),
                            }
                        ),
                        flush=True,
                    )
    categories, reasons, original_categories = (
        Counter(),
        Counter(),
        defaultdict(Counter),
    )
    warnings = Counter()
    with (
        (args.base_dir / "decisions.csv").open(
            encoding="utf-8", newline=""
        ) as source_stream,
        (partial / "decisions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as target_stream,
        (partial / "quarantine.csv").open(
            "w", encoding="utf-8", newline=""
        ) as quarantine_stream,
    ):
        reader = csv.DictReader(source_stream)
        writer, quarantine = (
            csv.DictWriter(target_stream, reader.fieldnames),
            csv.DictWriter(quarantine_stream, reader.fieldnames),
        )
        writer.writeheader()
        quarantine.writeheader()
        for row in reader:
            coordinate = row["split"] + ":" + row["line"]
            if coordinate in removed:
                row.update(
                    category="QUARANTINE",
                    reason="incomplete_letter",
                    assigned_split="",
                    warnings="incomplete_letter",
                )
            writer.writerow(row)
            categories[row["category"]] += 1
            original_categories[row["split"]][row["category"]] += 1
            reasons[row["reason"]] += 1
            if row["category"] == "QUARANTINE":
                quarantine.writerow(row)
                warnings.update(filter(None, row["warnings"].split("|")))
    for name in ("benchmark_exclusions.json", "near_source_groups.json"):
        copyfile(args.base_dir / name, partial / name)
    dump(partial / "letter_exclusions.json", removed)
    for path, expected in snapshots.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Input changed: {path}")
    summary = {
        **base,
        "schema_version": 4,
        "policy_version": POLICY,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_rows": dict(counts),
        "source_families": {split: len(values) for split, values in families.items()},
        "categories": {
            **{split: dict(value) for split, value in original_categories.items()},
            "combined": dict(categories),
        },
        "first_reason_counts": dict(reasons),
        "remaining_warning_counts_overlap": dict(warnings),
        "letter_review": {
            **dict(letter_counts),
            "newly_excluded_rows": len(removed),
            "policy": "For letter-style responses, every paragraph of at least four normalized words must be present as an ordered word subsequence in target props plus bound state. Extra headings/punctuation tolerated; shortened or rewritten letters quarantined.",
            "originals_unchanged": True,
        },
        "immutable_input_sha256": {**base["immutable_input_sha256"], **snapshots},
        "implementation_sha256": {
            **base["implementation_sha256"],
            Path(__file__).relative_to(ROOT).as_posix(): file_sha256(Path(__file__)),
            "training/src/ir_training/data/archive_letter_review.py": file_sha256(
                ROOT / "training/src/ir_training/data/archive_letter_review.py"
            ),
        },
        "outputs": {
            path.name: file_sha256(path) for path in partial.iterdir() if path.is_file()
        },
    }
    # Per-split coverage and v7 transition statistics are recomputed separately;
    # do not publish stale v7 counts under the final v8 label.
    for key in (
        "component_presence_rows",
        "original_to_new_split",
        "transitions_from_v6",
        "final_source_proven_repaired_rows_by_kind_overlap",
        "final_review_resolutions_overlap",
        "elapsed_seconds",
        "metadata_corrections",
    ):
        summary.pop(key, None)
    dump(partial / "manifest.json", summary)
    partial.rename(args.output_dir)
    dump(args.report_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "output_rows": dict(counts),
                "categories": dict(categories),
                "letter_review": summary["letter_review"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
