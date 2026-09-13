"""Profile the final offline copy and compare its decisions with frozen v5."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT / "training/scripts"))
from ir_training.data.archive_final_review import bound_content_strings
from ir_training.data.archive_recovery import _normalized_phrase
from ir_training.data.express_preparation import _api
from refine_recovered_archive import dump, unpack
from verify_recovered_archive import file_sha256


def decisions(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return {f"{r['split']}:{r['line']}": r for r in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--review-index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Fresh report required: {args.report}")
    current = decisions(args.dataset_dir / "decisions.csv")
    baseline = decisions(args.baseline_dir / "decisions.csv")
    if current.keys() != baseline.keys():
        raise ValueError("Original coordinate sets differ")
    accepted = {k: r for k, r in current.items() if r["category"] != "QUARANTINE"}
    families = {r["source_family"] for r in accepted.values()}
    transitions, admitted, removed, sibling_reasons = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    for coordinate, row in current.items():
        old = baseline[coordinate]["category"]
        transitions[f"{old}->{row['category']}"] += 1
        if old == "QUARANTINE" and coordinate in accepted:
            admitted[row["category"]] += 1
        if old != "QUARANTINE" and coordinate not in accepted:
            removed[row["reason"]] += 1
        if coordinate not in accepted and row["source_family"] in families:
            sibling_reasons[row["reason"]] += 1

    components = defaultdict(Counter)
    db = sqlite3.connect(args.review_index.resolve().as_uri() + "?mode=ro", uri=True)
    for coordinate, features in db.execute("SELECT coordinate,features FROM rows"):
        if coordinate in accepted:
            components[accepted[coordinate]["assigned_split"]].update(
                json.loads(features)
            )
    # Features are element counts; also report presence separately.
    presence = defaultdict(Counter)
    for coordinate, features in db.execute("SELECT coordinate,features FROM rows"):
        if coordinate in accepted:
            presence[accepted[coordinate]["assigned_split"]].update(
                json.loads(features).keys()
            )

    tiers, transformations, operations, resolutions = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    moved, row_counts = Counter(), Counter()
    family_sizes = defaultdict(Counter)
    chars = defaultdict(list)
    for split in ("train", "val"):
        with (args.dataset_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                recovery = row["metadata"]["archive_recovery"]
                coordinate = recovery["coordinate"]
                if (
                    coordinate not in accepted
                    or accepted[coordinate]["assigned_split"] != split
                ):
                    raise ValueError(f"Unexpected row: {coordinate}")
                row_counts[split] += 1
                moved[coordinate.split(":")[0] + "->" + split] += 1
                family_sizes[split][row["source_id"]] += 1
                changes = row["repair"]["changes"]
                tier = (
                    "unchanged"
                    if not row["repair"]["applied"]
                    else (
                        "lossless_representation"
                        if all(c.get("lossless") for c in changes)
                        else "source_grounded_structural"
                    )
                )
                tiers[tier] += 1
                transformations.update(set(recovery["transformations"]))
                operations.update(recovery["repair_metrics"])
                resolutions.update(
                    row["metadata"]["archive_refinement"]["review_resolutions"]
                )
                chars[f"{split}_source"].append(len(row["response_text"]))
                chars[f"{split}_target"].append(len(row["completion"]))

    excluded_letters = json.loads(
        (args.dataset_dir / "letter_exclusions.json").read_text(encoding="utf-8")
    )
    active, *_ = _api()
    letter_evidence = []
    for coordinate, gaps in sorted(excluded_letters.items()):
        blob = db.execute(
            "SELECT row_blob FROM rows WHERE coordinate=?", (coordinate,)
        ).fetchone()[0]
        row = unpack(blob)
        graph = active.decode_express_completion(row["completion"])
        source_words = Counter(_normalized_phrase(row["response_text"]).split())
        target_words = Counter(
            _normalized_phrase("\n".join(bound_content_strings(graph))).split()
        )
        missing = source_words - target_words
        letter_evidence.append(
            {
                "coordinate": coordinate,
                "gaps": gaps,
                "source_word_occurrences_absent_from_target": sum(missing.values()),
                "missing_distinct_words": sorted(missing),
                "needs_order_or_wording_review_only": not bool(missing),
            }
        )
    db.close()
    stats = {}
    for name, values in chars.items():
        values.sort()
        stats[name] = {
            "count": len(values),
            "min": values[0],
            "max": values[-1],
            **{
                f"p{p}": values[round((len(values) - 1) * p / 100)]
                for p in (50, 90, 95, 99)
            },
        }
    report = {
        "status": "profiled",
        "manifest_sha256": file_sha256(args.dataset_dir / "manifest.json"),
        "baseline_manifest_sha256": file_sha256(args.baseline_dir / "manifest.json"),
        "output_rows": dict(row_counts),
        "transitions_from_v5": dict(transitions),
        "newly_admitted_vs_v5": dict(admitted),
        "newly_excluded_vs_v5": dict(removed),
        "confidence_tiers": dict(tiers),
        "retained_transformations_rows_overlap": dict(transformations),
        "retained_repair_operation_counts_overlap": dict(operations),
        "retained_review_resolutions_rows_overlap": dict(resolutions),
        "component_presence_rows": {k: dict(v) for k, v in presence.items()},
        "component_instances": {k: dict(v) for k, v in components.items()},
        "source_family_size_distribution": {
            k: dict(Counter(v.values())) for k, v in family_sizes.items()
        },
        "original_to_new_split": dict(moved),
        "character_lengths_not_token_lengths": stats,
        "quarantined_rows_with_retained_source_family": dict(sibling_reasons),
        "sibling_note": "A known-good target already covers these source families; duplicating it over a bad sibling is not new data recovery.",
        "letter_exclusion_evidence": letter_evidence,
        "profile_script_sha256": file_sha256(Path(__file__)),
    }
    dump(args.report, report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in {
                    "letter_exclusion_evidence",
                    "component_instances",
                    "retained_repair_operation_counts_overlap",
                }
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
