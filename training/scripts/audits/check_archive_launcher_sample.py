"""Run the launcher's CPU admission filter on a fixed final-archive sample."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT / "training/scripts"))
from ir_training.data.archive_final_review import bound_content_strings
from ir_training.data.audit_filter import audit_and_filter_rows, load_reserved_cohorts
from ir_training.data.express_preparation import _api
from refine_recovered_archive import dump
from verify_recovered_archive import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Fresh report required: {args.report}")
    sample, targeted = [], {}
    for split in ("train", "val"):
        with (args.dataset_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                recovery = row["metadata"]["archive_recovery"]
                coordinate = recovery["coordinate"]
                priority = int.from_bytes(
                    hashlib.sha256(coordinate.encode()).digest()[:8]
                )
                item = (-priority, coordinate, row)
                if len(sample) < 128:
                    heapq.heappush(sample, item)
                elif item > sample[0]:
                    heapq.heapreplace(sample, item)
                if set(recovery["transformations"]) & {
                    "midword_text_boundary_joins",
                    "source_proven_missing_button_label",
                }:
                    targeted[coordinate] = row
    selected = {coordinate: row for _, coordinate, row in sample}
    selected.update(targeted)
    reserved = load_reserved_cohorts(
        [
            ROOT / "training/data/eval/golden32_archive_repeat_v1/golden32.jsonl",
            ROOT / "training/data/eval/golden35_v1/golden35.jsonl",
        ]
    )
    accepted, quarantine, admission = audit_and_filter_rows(
        list(selected.values()), reserved=reserved, require_source_identities=True
    )
    if quarantine or len(accepted) != len(selected):
        raise ValueError(f"Launcher rejected a final sample: {admission}")
    active, *_ = _api()
    boundary_examples = []
    for coordinate, row in sorted(targeted.items()):
        if (
            "midword_text_boundary_joins"
            in row["metadata"]["archive_recovery"]["transformations"]
        ):
            boundary_examples.append(
                {
                    "coordinate": coordinate,
                    "source": row["response_text"],
                    "target_content": bound_content_strings(
                        active.decode_express_completion(row["completion"])
                    ),
                }
            )
    result = {
        "status": "verified",
        "manifest_sha256": file_sha256(args.dataset_dir / "manifest.json"),
        "deterministic_sample_size": len(sample),
        "targeted_repair_rows": len(targeted),
        "unique_rows_checked": len(selected),
        "admission_report": admission,
        "remaining_midword_join_examples": boundary_examples,
        "not_performed": [
            "model tokenizer/template preparation",
            "training",
            "model inference",
            "device rendering",
        ],
        "script_sha256": file_sha256(Path(__file__)),
    }
    dump(args.report, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "remaining_midword_join_examples"
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
