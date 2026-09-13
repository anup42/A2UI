#!/usr/bin/env python3
"""Build a fresh v7 copy after the follow-up content review of v6.

Reuses the completed v6 all-row audit; originals, v5, and v6 stay read-only.
The few v6 button-label changes are reconstructed from original source rows
before applying the corrected rule. All other accepted targets are rescanned.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
from ir_training.data.archive_final_review import (
    POLICY,
    paragraph_gaps,
    repair_final_text,
)
from ir_training.data.archive_recovery import (
    placeholder_tokens,
    recover_pair,
    text_sha256,
)
from ir_training.data.archive_refinement import (
    FamilyUnion,
    NearSourceIndex,
    choose_validation,
    component_shape,
    review_warnings,
)
from ir_training.data.express_preparation import TASK_PREFIX, serialize_checked
from ir_training.data.ir_targets import (
    A2UI_EXPRESS_V1,
    materialize_completion_targets,
    semantic_hash,
)
from recover_full_data_archive import _batch, _init_worker, file_sha256
from refine_recovered_archive import dump, pack, unpack


def worker(batch):
    from ir_training.data import express_preparation as prep
    from pipeline.ir_formats import active

    output = []
    for coordinate, blob, original in batch:
        row = unpack(blob)
        source, prior_target = row["response_text"], row["completion"]
        graph = active.decode_express_completion(prior_target)
        prep._validate_wire(graph)
        metadata = row["metadata"]
        recovery = metadata["archive_recovery"]
        reverted = 0
        changes = {}
        if original:
            recovered = recover_pair(*original)
            if recovered.source_text != source or recovered.graph is None:
                raise ValueError(f"Unable to reconstruct v5 source/graph: {coordinate}")
            graph, changes = repair_final_text(source, recovered.graph)
            old_kind = "source_proven_existing_button_label"
            reverted = metadata["archive_refinement"]["additional_repairs"].get(
                old_kind, 0
            )
            recovery["transformations"] = list(recovered.transformations)
            recovery["repair_metrics"] = dict(recovered.repair_metrics)
            row["repair"]["changes"] = [
                item
                for item in row["repair"]["changes"]
                if not item["kind"].startswith("source_proven_")
            ]
            for kind, count in changes.items():
                recovery["transformations"].append(kind)
                recovery["repair_metrics"][kind] = count
                row["repair"]["changes"].append(
                    {"kind": kind, "lossless": False, "source_grounded": True}
                )
            checked = serialize_checked(
                materialize_completion_targets(graph)[A2UI_EXPRESS_V1], "root-first"
            )
            graph, target = checked.graph, checked.text
        else:
            target = prior_target
        warnings, resolutions, features = review_warnings(source, target, graph)
        gaps = paragraph_gaps(source, graph)
        if gaps:
            warnings.append("severely_missing_prose_block")
        if not placeholder_tokens(target) <= placeholder_tokens(source):
            warnings.append("unresolved_target_reference")
        row["completion"] = target
        row["messages"][-1]["content"] = target
        row["repair"]["applied"] = bool(row["repair"]["changes"])
        recovery["effective_target_sha256"] = text_sha256(target)
        recovery["effective_semantic_sha256"] = semantic_hash(graph)
        metadata["archive_final_review"] = {
            "policy_version": POLICY,
            "prior_v6_target_sha256": text_sha256(prior_target),
            "v6_button_relabels_reconstructed": reverted,
            "corrected_rule_repairs": changes,
            "prose_blocks_failed": gaps,
            "stage3_run": False,
        }
        output.append(
            {
                "coordinate": coordinate,
                "accepted": not warnings,
                "warnings": sorted(set(warnings)),
                "gaps": gaps,
                "row_blob": pack(row),
                "semantic": recovery["effective_semantic_sha256"],
                "pair": text_sha256(source + "\0" + target),
                "changes": changes,
                "features": features,
                "resolutions": resolutions,
                "reconstructed_button_labels": reverted,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--review-index", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 0 < args.val_fraction < 0.5:
        raise ValueError("Invalid workers or split fraction")
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
    for path in (args.output_dir, args.audit_dir, args.report_dir):
        if path.exists():
            raise FileExistsError(f"Fresh destination required: {path}")
    base = json.loads((args.base_dir / "manifest.json").read_text(encoding="utf-8"))
    if base.get("policy_version") != "messages-archive-refinement-v6-20260913":
        raise ValueError("Requires a completed v6 intermediate")
    snapshots = {
        str(args.review_index.resolve()): file_sha256(args.review_index),
        str((args.base_dir / "manifest.json").resolve()): file_sha256(
            args.base_dir / "manifest.json"
        ),
    }
    for name, digest in base["outputs"].items():
        path = args.base_dir / name
        if file_sha256(path) != digest:
            raise ValueError(f"Changed v6 artifact: {path}")
        snapshots[str(path.resolve())] = digest
    for split, entry in base["source_files"].items():
        path = args.source_dir / f"{split}.jsonl"
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Changed original: {path}")
        snapshots[str(path.resolve())] = entry["sha256"]
    with (args.base_dir / "decisions.csv").open(encoding="utf-8", newline="") as stream:
        decisions = {
            f"{row['split']}:{row['line']}": row for row in csv.DictReader(stream)
        }
    source_db = sqlite3.connect(
        args.review_index.resolve().as_uri() + "?mode=ro", uri=True
    )
    if source_db.execute("SELECT COUNT(*) FROM rows WHERE accepted=1").fetchone()[
        0
    ] != sum(base["output_rows"].values()):
        raise ValueError("v6 review index and completed output disagree")
    args.audit_dir.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(args.audit_dir / "review.sqlite")
    source_db.backup(db)
    inventory = sqlite3.connect(
        args.inventory.resolve().as_uri() + "?mode=ro", uri=True
    )
    handles = {
        split: (args.source_dir / f"{split}.jsonl").open("rb")
        for split in ("train", "val")
    }

    def records():
        for coordinate, blob, old_changes in source_db.execute(
            "SELECT coordinate,row_blob,changes FROM rows WHERE accepted=1 ORDER BY coordinate"
        ):
            original = None
            if json.loads(old_changes).get("source_proven_existing_button_label"):
                split, number = coordinate.split(":")
                offset, length = inventory.execute(
                    "SELECT byte_offset,byte_length FROM rows WHERE split=? AND line=?",
                    (split, int(number)),
                ).fetchone()
                handles[split].seek(offset)
                raw = json.loads(
                    handles[split]
                    .read(length)
                    .decode("utf-16-le")
                    .removeprefix("\ufeff")
                )
                source, target = (
                    raw["messages"][-2]["content"][len(TASK_PREFIX) :],
                    raw["messages"][-1]["content"],
                )
                bound = unpack(blob)["metadata"]["archive_recovery"]
                if (
                    text_sha256(source) != bound["original_source_sha256"]
                    or text_sha256(target) != bound["original_target_sha256"]
                ):
                    raise ValueError(f"Original row binding differs: {coordinate}")
                original = source, target
            yield coordinate, blob, original

    reviewed, begin = 0, time.monotonic()
    prose_examples, changed_counts, corrected_repairs = [], Counter(), Counter()
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker
    ) as pool:
        pending = set()

        def consume(done):
            nonlocal reviewed
            for future in done:
                for result in future.result():
                    reason = (
                        "passed_final_content_review"
                        if result["accepted"]
                        else "unresolved_content_or_reference"
                    )
                    db.execute(
                        "UPDATE rows SET accepted=?,reason=?,warnings=?,row_blob=?,semantic=?,pair=?,features=? WHERE coordinate=?",
                        (
                            int(result["accepted"]),
                            reason,
                            json.dumps(result["warnings"]),
                            result["row_blob"],
                            result["semantic"],
                            result["pair"],
                            json.dumps(result["features"]),
                            result["coordinate"],
                        ),
                    )
                    if result["gaps"]:
                        prose_examples.append(
                            {"coordinate": result["coordinate"], "gaps": result["gaps"]}
                        )
                    if result["reconstructed_button_labels"]:
                        changed_counts["v6_label_rows_reconstructed"] += 1
                        changed_counts["v6_label_operations_reconstructed"] += result[
                            "reconstructed_button_labels"
                        ]
                    if result["accepted"]:
                        corrected_repairs.update(result["changes"])
                    reviewed += 1
                    if reviewed % 4096 == 0:
                        db.commit()
                        print(
                            json.dumps(
                                {
                                    "phase": "final_content_review",
                                    "rows": reviewed,
                                    "prose_gap_rows": len(prose_examples),
                                    "seconds": round(time.monotonic() - begin),
                                }
                            ),
                            flush=True,
                        )

        for batch in _batch(records(), 16):
            pending.add(pool.submit(worker, batch))
            if len(pending) >= args.workers * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                consume(done)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            consume(done)
    db.commit()
    for handle in handles.values():
        handle.close()
    inventory.close()
    source_db.close()
    families, groups = FamilyUnion(), {}
    for old_family, family in db.execute("SELECT DISTINCT old_family,family FROM rows"):
        families.union(old_family, family)
    for family, features, length in db.execute(
        "SELECT family,features,LENGTH(signature) FROM rows WHERE accepted=1"
    ):
        current = groups.get(family)
        if current is None or length > current[1]:
            groups[family] = component_shape(json.loads(features)), length
    selected = choose_validation(groups, args.val_fraction, args.seed)
    passes, new_edges = [], []
    source_rows = db.execute(
        "SELECT DISTINCT family,signature FROM rows WHERE accepted=1 ORDER BY family,signature"
    ).fetchall()
    for iteration in range(1, 9):
        validation_roots = {families.find(family) for family in selected}
        refs = {
            f"{family}:{text_sha256(signature)}": signature
            for family, signature in source_rows
            if families.find(family) in validation_roots
        }
        index, added = NearSourceIndex(refs), 0
        for family, signature in source_rows:
            if families.find(family) in validation_roots:
                continue
            for key, score, contained in index.matches(signature):
                other = key.split(":", 1)[0]
                if families.union(family, other):
                    added += 1
                    new_edges.append(
                        {
                            "family_a": family,
                            "family_b": other,
                            "word_trigram_jaccard": score,
                            "containment": contained,
                            "pass": iteration,
                        }
                    )
        passes.append(
            {
                "pass": iteration,
                "validation_source_variants": len(refs),
                "new_family_edges": added,
            }
        )
        print(json.dumps({"phase": "final_split_review", **passes[-1]}), flush=True)
        if not added:
            break
    else:
        raise ValueError("Final split grouping did not converge")
    validation_roots = {families.find(family) for family in selected}
    for coordinate, family in db.execute(
        "SELECT coordinate,family FROM rows"
    ).fetchall():
        family = families.find(family)
        db.execute(
            "UPDATE rows SET family=?,assigned_split=? WHERE coordinate=?",
            (family, "val" if family in validation_roots else "train", coordinate),
        )
    seen, pairs = set(), set()
    for coordinate, family, semantic, pair in db.execute(
        "SELECT coordinate,family,semantic,pair FROM rows WHERE accepted=1 ORDER BY sort_key"
    ).fetchall():
        if (family, semantic) in seen or pair in pairs:
            db.execute(
                "UPDATE rows SET accepted=0,reason='duplicate_source_family_target' WHERE coordinate=?",
                (coordinate,),
            )
        else:
            seen.add((family, semantic))
            pairs.add(pair)
    db.commit()
    partial = args.output_dir.with_name(
        args.output_dir.name + f".partial-{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    output_rows, output_families, components, emitted = (
        Counter(),
        defaultdict(set),
        defaultdict(Counter),
        {},
    )
    final_repairs, review_resolutions = Counter(), Counter()
    for split in ("train", "val"):
        with (partial / f"{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for coordinate, family, blob, features in db.execute(
                "SELECT coordinate,family,row_blob,features FROM rows WHERE accepted=1 AND assigned_split=? ORDER BY sort_key",
                (split,),
            ):
                row = unpack(blob)
                row["source_id"] = f"archive-source-family-sha256:{family}"
                metadata = row["metadata"]
                metadata["source_id"] = row["source_id"]
                metadata["assigned_split"] = split
                metadata["archive_refinement"]["assigned_split"] = split
                metadata["archive_final_review"]["assigned_split"] = split
                stream.write(
                    json.dumps(
                        row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                )
                emitted[coordinate] = "REPAIR" if row["repair"]["applied"] else "KEEP"
                output_rows[split] += 1
                output_families[split].add(family)
                components[split].update(json.loads(features).keys())
                final_repairs.update(
                    kind
                    for kind in metadata["archive_recovery"]["transformations"]
                    if kind.startswith("source_proven_")
                )
                review_resolutions.update(
                    metadata["archive_refinement"]["review_resolutions"]
                )
    fields = [
        "split",
        "line",
        "category",
        "reason",
        "assigned_split",
        "prior_category",
        "prior_reason",
        "source_family",
        "warnings",
    ]
    categories, reasons, transitions, warnings = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    original_categories = defaultdict(Counter)
    moved = Counter()
    with (
        (partial / "decisions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as all_stream,
        (partial / "quarantine.csv").open(
            "w", encoding="utf-8", newline=""
        ) as quarantine_stream,
    ):
        writer, quarantine = (
            csv.DictWriter(all_stream, fields),
            csv.DictWriter(quarantine_stream, fields),
        )
        writer.writeheader()
        quarantine.writeheader()
        for coordinate, family, reason, warning_json, split in db.execute(
            "SELECT coordinate,family,reason,warnings,assigned_split FROM rows ORDER BY coordinate"
        ):
            prior = decisions[coordinate]
            category = emitted.get(coordinate, "QUARANTINE")
            record = {
                "split": prior["split"],
                "line": prior["line"],
                "category": category,
                "reason": reason,
                "assigned_split": split if coordinate in emitted else "",
                "prior_category": prior["category"],
                "prior_reason": prior["reason"],
                "source_family": family,
                "warnings": "|".join(json.loads(warning_json)),
            }
            writer.writerow(record)
            categories[category] += 1
            original_categories[prior["split"]][category] += 1
            reasons[reason] += 1
            transitions[f"{prior['category']}->{category}"] += 1
            if category == "QUARANTINE":
                quarantine.writerow(record)
                warnings.update(json.loads(warning_json))
            else:
                moved[f"{prior['split']}->{split}"] += 1
    for name in ("benchmark_exclusions.json", "near_source_groups.json"):
        value = json.loads((args.base_dir / name).read_text(encoding="utf-8"))
        if name == "near_source_groups.json":
            value.extend(new_edges)
        dump(partial / name, value)
    dump(
        args.audit_dir / "prose_gap_cases.json",
        sorted(prose_examples, key=lambda row: row["coordinate"]),
    )
    if output_families["train"] & output_families["val"]:
        raise ValueError("Final source families span both splits")
    for path, expected in snapshots.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Original or intermediate changed: {path}")
    for path, expected in base["benchmark_file_sha256"].items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError(f"Frozen benchmark changed: {path}")
    summary = {
        **base,
        "schema_version": 3,
        "policy_version": POLICY,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - begin, 2),
        "immutable_input_sha256": snapshots,
        "output_rows": dict(output_rows),
        "categories": {
            **{key: dict(value) for key, value in original_categories.items()},
            "combined": dict(categories),
        },
        "source_families": {key: len(value) for key, value in output_families.items()},
        "first_reason_counts": dict(reasons),
        "transitions_from_v6": dict(transitions),
        "remaining_warning_counts_overlap": dict(warnings),
        "component_presence_rows": {
            key: dict(value) for key, value in components.items()
        },
        "original_to_new_split": dict(moved),
        "final_source_proven_repaired_rows_by_kind_overlap": dict(final_repairs),
        "final_review_resolutions_overlap": dict(review_resolutions),
        "final_review": {
            "all_v6_candidates_reviewed": reviewed,
            "additional_prose_gap_rows_excluded": len(prose_examples),
            **dict(changed_counts),
            "corrected_rule_operations_retained": dict(corrected_repairs),
            "paragraph_policy": "prose block >=120 chars, >=16 distinct content words, >=8 absent words, and <50% content-word recall against props plus bound state; heuristic quarantine, not a factual verdict",
        },
        "split_rebuild": {
            **base["split_rebuild"],
            "seed": args.seed,
            "initial_validation_family_fraction": args.val_fraction,
            "near_source_passes": passes,
            "new_near_family_edges": len(new_edges),
            "remaining_detected_near_cross_split_matches": 0,
        },
        "implementation_sha256": {
            p.relative_to(ROOT).as_posix(): file_sha256(p)
            for p in (
                Path(__file__),
                ROOT / "training/src/ir_training/data/archive_final_review.py",
                ROOT / "training/src/ir_training/data/archive_refinement.py",
                ROOT / "training/src/ir_training/data/archive_recovery.py",
            )
        },
        "outputs": {p.name: file_sha256(p) for p in partial.iterdir() if p.is_file()},
    }
    for key in (
        "transitions_from_v5",
        "additional_repaired_rows_by_kind_overlap",
        "additional_repair_operations",
        "accepted_review_resolutions_overlap",
    ):
        summary.pop(key, None)
    dump(partial / "manifest.json", summary)
    partial.rename(args.output_dir)
    dump(args.report_dir / "summary.json", summary)
    db.close()
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "output_rows": dict(output_rows),
                "categories": dict(categories),
                "final_review": summary["final_review"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
