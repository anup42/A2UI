#!/usr/bin/env python3
"""Independently rescan a refined copy for Golden and split contamination."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))

from ir_training.data.archive_boundary_review import repair_join_boundaries
from ir_training.data.archive_letter_review import LETTER_START, letter_gaps
from ir_training.data.archive_recovery import text_sha256
from ir_training.data.archive_refinement import NearSourceIndex, source_signature
from ir_training.data.express_preparation import _api
from refine_recovered_archive import load_goldens
from verify_recovered_archive import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--strict-sample-size", type=int, default=512)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Fresh verification report required: {args.report}")
    manifest = json.loads(
        (args.dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") not in {2, 3, 4, 5, 6} or not manifest.get(
        "split_rebuild"
    ):
        raise ValueError("Requires a completed refinement with rebuilt splits")
    if manifest["schema_version"] >= 6:
        catalog = ROOT / "training/data/quality/v9_manual100_findings.json"
        if file_sha256(catalog) != manifest["reviewed_findings_sha256"]:
            raise ValueError("Reviewed-source catalog differs; use the matching v10 code revision")
    # Invoke the original, unaccelerated production parser/schema verifier on
    # a deterministic sample, plus the all-row/hash/placeholder/identity gates.
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("verify_recovered_archive.py")),
            "--dataset-dir",
            str(args.dataset_dir),
            "--strict-sample-size",
            str(args.strict_sample_size),
        ],
        check=True,
    )
    golden_sources, excluded_original_hashes, golden_files = load_goldens()
    if golden_files != manifest["benchmark_file_sha256"]:
        raise ValueError("Frozen Golden files differ from this candidate's manifest")
    golden_index = NearSourceIndex(golden_sources)
    golden_signatures = {text_sha256(source) for source in golden_sources.values()}
    validation, validation_families = {}, set()
    for line in (args.dataset_dir / "val.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        signature = source_signature(row["response_text"])
        validation[text_sha256(signature)] = signature
        validation_families.add(row["source_id"])
    validation_index = NearSourceIndex(validation)
    counts = Counter()
    checked_signatures = set()
    letter_rows_checked = 0
    historical_join_rows_checked = 0
    active, *_ = _api()
    for split in ("train", "val"):
        with (args.dataset_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                coordinate = row["metadata"]["archive_recovery"]["coordinate"]
                if (
                    manifest["schema_version"] >= 5
                    and "midword_text_boundary_joins"
                    in row["metadata"]["archive_recovery"]["transformations"]
                ):
                    graph = active.decode_express_completion(row["completion"])
                    original_graph = active.decode_express_completion(
                        row["metadata"]["archive_recovery"]["target_transcode"]["text"]
                    )
                    _, pending, issues = repair_join_boundaries(
                        row["response_text"], graph, original_graph
                    )
                    if pending or issues:
                        raise ValueError(
                            f"Historical boundary still needs review: {coordinate}: {pending}, {issues}"
                        )
                    historical_join_rows_checked += 1
                if manifest["schema_version"] >= 4 and LETTER_START.search(
                    row["response_text"]
                ):
                    if letter_gaps(
                        row["response_text"],
                        active.decode_express_completion(row["completion"]),
                    ):
                        raise ValueError(f"Incomplete letter leaked: {coordinate}")
                    letter_rows_checked += 1
                signature = source_signature(row["response_text"])
                digest = text_sha256(signature)
                original_digest = row["metadata"]["archive_recovery"][
                    "original_source_sha256"
                ]
                if (
                    digest in golden_signatures
                    or original_digest in excluded_original_hashes
                ):
                    raise ValueError(
                        f"Golden/excluded original source leaked: {coordinate}"
                    )
                if digest not in checked_signatures:
                    if list(golden_index.matches(signature, containment=True)):
                        raise ValueError(
                            f"Lexical Golden neighbor leaked: {coordinate}"
                        )
                    checked_signatures.add(digest)
                if split == "train":
                    if digest in validation or row["source_id"] in validation_families:
                        raise ValueError(
                            f"Normalized source or family spans train/val: {coordinate}"
                        )
                    if list(validation_index.matches(signature)):
                        raise ValueError(
                            f"Lexical neighbor spans train/val: {coordinate}"
                        )
                counts[split] += 1
                if sum(counts.values()) % 25000 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "independent_source_separation",
                                "rows": sum(counts.values()),
                            }
                        ),
                        flush=True,
                    )
    if dict(counts) != manifest["output_rows"]:
        raise ValueError("Independent output counts differ")
    report = {
        "status": "verified",
        "policy_version": manifest["policy_version"],
        "manifest_sha256": file_sha256(args.dataset_dir / "manifest.json"),
        "output_rows": dict(counts),
        "strict_production_sample_rows": args.strict_sample_size,
        "independent_content_review_sample_rows": args.strict_sample_size,
        "residual_same_policy_repairs_in_sample": 0,
        "unique_normalized_sources_checked": len(checked_signatures),
        "exact_or_normalized_golden_matches": 0,
        "lexical_near_golden_matches": 0,
        "train_val_source_family_overlap": 0,
        "train_val_normalized_source_overlap": 0,
        "train_val_lexical_near_matches": 0,
        "all_row_hash_identity_placeholder_duplicate_checks": "passed",
        "remaining_limit": "No model-tokenizer preflight, semantic paraphrase guarantee, device render, or training score.",
        "verification_script_sha256": file_sha256(Path(__file__)),
    }
    if manifest["schema_version"] >= 4:
        report["all_retained_letter_rows_checked"] = letter_rows_checked
        report["remaining_letter_policy_gaps"] = 0
    if manifest["schema_version"] >= 5:
        report["all_retained_historical_join_rows_checked"] = (
            historical_join_rows_checked
        )
        report["residual_join_boundary_issues"] = 0
    if manifest["schema_version"] >= 6:
        report["semantic_review_sample_rows"] = args.strict_sample_size
        report["residual_semantic_issues_in_sample"] = 0
        report["all_row_semantic_review_export_evidence"] = manifest["semantic_review"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
