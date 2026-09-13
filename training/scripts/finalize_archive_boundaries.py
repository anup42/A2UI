"""Create the final v9 copy with source-proven historical Text boundaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile, disk_usage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
from ir_training.data.archive_boundary_review import POLICY, repair_join_boundaries
from ir_training.data.archive_final_review import paragraph_gaps
from ir_training.data.archive_letter_review import letter_gaps
from ir_training.data.archive_recovery import text_sha256
from ir_training.data.archive_refinement import review_warnings
from ir_training.data.express_preparation import _api, serialize_checked
from ir_training.data.ir_targets import (
    A2UI_EXPRESS_V1,
    materialize_completion_targets,
    semantic_hash,
)
from recover_full_data_archive import file_sha256
from refine_recovered_archive import dump


def check_export_space(output_dir, estimated_bytes):
    parent = output_dir.resolve().parent
    while not parent.exists():
        parent = parent.parent
    available = disk_usage(parent).free
    required = estimated_bytes + 1024**3
    if available < required:
        raise OSError(
            f"Insufficient space for a fresh export: need {required:,} bytes "
            f"including 1 GiB headroom, available {available:,} at {parent}. "
            "No output directory was created."
        )
    return {
        "estimated_output_bytes": estimated_bytes,
        "required_bytes": required,
        "available_bytes_before_export": available,
        "headroom_bytes": 1024**3,
    }


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
                }
            )
        )
        return 0
    for path in (args.output_dir, args.report_dir):
        if path.exists():
            raise FileExistsError(f"Fresh destination required: {path}")
    manifest_path = args.base_dir / "manifest.json"
    base = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        base.get("schema_version") != 4
        or base.get("status") != "candidate_export_complete"
    ):
        raise ValueError("Requires the completed v8 intermediate")
    space = check_export_space(
        args.output_dir,
        sum((args.base_dir / name).stat().st_size for name in base["outputs"]),
    )
    snapshots = {str(manifest_path.resolve()): file_sha256(manifest_path)}
    for name, expected in base["outputs"].items():
        path = args.base_dir / name
        if file_sha256(path) != expected:
            raise ValueError(f"Input hash differs: {path}")
        snapshots[str(path.resolve())] = expected
    for details in base["source_files"].values():
        path = Path(details["path"])
        if file_sha256(path) != details["sha256"]:
            raise ValueError(f"Original input hash differs: {path}")
        snapshots[str(path.resolve())] = details["sha256"]
    partial = args.output_dir.with_name(
        args.output_dir.name + f".partial-{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    active, *_ = _api()
    counts, checked_rows, changed = Counter(), 0, {}
    for split in ("train", "val"):
        with (
            (args.base_dir / f"{split}.jsonl").open(encoding="utf-8") as stream,
            (partial / f"{split}.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as output,
        ):
            for line in stream:
                row = json.loads(line)
                recovery = row["metadata"]["archive_recovery"]
                if "midword_text_boundary_joins" in recovery["transformations"]:
                    checked_rows += 1
                    coordinate = recovery["coordinate"]
                    graph = active.decode_express_completion(row["completion"])
                    original_graph = active.decode_express_completion(
                        recovery["target_transcode"]["text"]
                    )
                    fixed, proofs, issues = repair_join_boundaries(
                        row["response_text"], graph, original_graph
                    )
                    if issues:
                        raise ValueError(
                            f"Unproven historical boundary at {coordinate}: {issues}"
                        )
                    if proofs:
                        if not row["repair"]["applied"]:
                            raise ValueError(
                                "Historical join unexpectedly classified KEEP"
                            )
                        checked = serialize_checked(
                            materialize_completion_targets(fixed)[A2UI_EXPRESS_V1],
                            "root-first",
                        )
                        warnings, _, _ = review_warnings(
                            row["response_text"], checked.text, checked.graph
                        )
                        if (
                            warnings
                            or paragraph_gaps(row["response_text"], checked.graph)
                            or letter_gaps(row["response_text"], checked.graph)
                        ):
                            raise ValueError(
                                f"Changed target failed content review: {coordinate}"
                            )
                        row["metadata"]["archive_boundary_review"] = {
                            "policy_version": POLICY,
                            "prior_target_sha256": text_sha256(row["completion"]),
                            "source_proofs": proofs,
                            "stage3_run": False,
                        }
                        row["completion"] = checked.text
                        row["messages"][-1]["content"] = checked.text
                        recovery["effective_target_sha256"] = text_sha256(checked.text)
                        recovery["effective_semantic_sha256"] = semantic_hash(
                            checked.graph
                        )
                        kind = "source_proven_join_separator"
                        recovery["transformations"].append(kind)
                        recovery["repair_metrics"][kind] = len(proofs)
                        row["repair"]["changes"].append(
                            {"kind": kind, "lossless": False, "source_grounded": True}
                        )
                        line = (
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        changed[coordinate] = proofs
                output.write(line)
                counts[split] += 1
                if sum(counts.values()) % 25000 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "source_boundary_review",
                                "rows": sum(counts.values()),
                                "reviewed": checked_rows,
                                "changed": len(changed),
                            }
                        ),
                        flush=True,
                    )
    if dict(counts) != base["output_rows"]:
        raise ValueError("Final row counts differ")
    for name in base["outputs"]:
        if name not in {"train.jsonl", "val.jsonl"}:
            copyfile(args.base_dir / name, partial / name)
    dump(partial / "boundary_repairs.json", changed)
    for path, expected in snapshots.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Input changed: {path}")
    summary = {
        **base,
        "schema_version": 5,
        "policy_version": POLICY,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_review": {
            "historical_join_rows_checked": checked_rows,
            "source_proven_separator_rows_repaired": len(changed),
            "membership_and_splits_unchanged": True,
            "originals_unchanged": True,
        },
        "disk_space_preflight": space,
        "immutable_input_sha256": {**base["immutable_input_sha256"], **snapshots},
        "implementation_sha256": {
            **base["implementation_sha256"],
            Path(__file__).relative_to(ROOT).as_posix(): file_sha256(Path(__file__)),
            "training/src/ir_training/data/archive_boundary_review.py": file_sha256(
                ROOT / "training/src/ir_training/data/archive_boundary_review.py"
            ),
        },
        "outputs": {
            path.name: file_sha256(path) for path in partial.iterdir() if path.is_file()
        },
    }
    dump(partial / "manifest.json", summary)
    partial.rename(args.output_dir)
    dump(args.report_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "output_rows": dict(counts),
                "boundary_review": summary["boundary_review"],
                "changed": changed,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
