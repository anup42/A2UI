"""Count exact-source sibling alternatives without changing any training label.

A no-warning sibling is only a repair/review candidate, not proof of semantic
correctness. When source-level deduplication is used, retaining the good sibling
and discarding the bad row adds no new source coverage and needs no IR rewrite.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3


TRAINING = Path(__file__).resolve().parents[2]
WARNING_FIELDS = (
    "empty_layout", "layout_only", "mojibake", "placeholder", "missing_action",
    "low_lexical", "missing_numeric",
)


def disqualifications(row, *, include_reserved=True):
    reasons = []
    if include_reserved and row["reserved"]:
        reasons.append("reserved")
    if not row["valid"]:
        reasons.append("invalid")
    if row["url_error"]:
        reasons.append("url_error")
    if row["sequence_tokens"] is None or row["sequence_tokens"] > 4096:
        reasons.append("sequence_over_4096_or_unknown")
    if row["target_tokens"] is None or row["target_tokens"] > 2048:
        reasons.append("target_over_2048_or_unknown")
    reasons.extend(field for field in WARNING_FIELDS if row[field])
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=TRAINING / "outputs/audits/full_data_20260913/synthesis.sqlite")
    parser.add_argument("--report", type=Path, default=TRAINING / "reports/offline_triage_20260913/siblings")
    args = parser.parse_args()
    connection = sqlite3.connect(args.index.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    groups = defaultdict(list)
    base_counts = defaultdict(Counter)
    base_clean_sources = defaultdict(set)
    base_clean_pairs = defaultdict(set)
    for raw in connection.execute("SELECT * FROM triage ORDER BY split,line"):
        row = dict(raw)
        row["reasons"] = disqualifications(row)
        groups[row["source_sha"]].append(row)
        base_counts[row["split"]]["rows"] += 1
        if row["reserved"]:
            base_counts[row["split"]]["reserved_rows"] += 1
        if not row["reasons"]:
            base_counts[row["split"]]["no_warning_budget_screened_rows"] += 1
            base_clean_sources[row["split"]].add(row["source_sha"])
            base_clean_sources["combined"].add(row["source_sha"])
            base_clean_pairs[row["split"]].add((row["source_sha"], row["semantic_sha"]))
            base_clean_pairs["combined"].add((row["source_sha"], row["semantic_sha"]))
        else:
            base_counts[row["split"]]["disqualified_rows_including_reserved"] += 1
    connection.close()

    candidates = []
    candidate_counts = defaultdict(Counter)
    source_counts = defaultdict(set)
    for split in ("train", "val"):
        for key in (
            "flagged_rows_with_any_split_screened_sibling",
            "flagged_rows_with_same_split_screened_sibling",
            "flagged_rows_with_other_split_sibling_only",
            "one_unique_screened_semantic_target_any_split",
            "multiple_screened_semantic_targets_any_split",
            "one_unique_screened_semantic_target_same_split",
        ):
            candidate_counts[split][key] = 0
        source_counts[f"{split}_all_candidate_sources"] = set()
        source_counts[f"{split}_same_split_candidate_sources"] = set()
    reserved_alternative_examples = []
    for source_sha, rows in groups.items():
        clean = [row for row in rows if not row["reasons"]]
        if not clean:
            # Reserved sources are never eligible, even if an alternative would
            # otherwise pass the tested checks.
            if any(row["reserved"] for row in rows):
                otherwise_clean = [row for row in rows if not disqualifications(row, include_reserved=False)]
                if otherwise_clean:
                    reserved_alternative_examples.append({
                        "source_sha256": source_sha,
                        "otherwise_no_warning_rows": [f"{r['split']}:{r['line']}" for r in otherwise_clean],
                        "all_rows": [f"{r['split']}:{r['line']}" for r in rows],
                    })
            continue
        semantic_shas = {r["semantic_sha"] for r in clean}
        clean_splits = {r["split"] for r in clean}
        for row in rows:
            if row["reserved"] or not row["reasons"]:
                continue
            split = row["split"]
            own_clean = [r for r in clean if r["split"] == split]
            own_semantics = {r["semantic_sha"] for r in own_clean}
            candidate_counts[split]["flagged_rows_with_any_split_screened_sibling"] += 1
            source_counts[f"{split}_all_candidate_sources"].add(source_sha)
            source_counts["all_candidate_sources"].add(source_sha)
            if len(semantic_shas) == 1:
                candidate_counts[split]["one_unique_screened_semantic_target_any_split"] += 1
            else:
                candidate_counts[split]["multiple_screened_semantic_targets_any_split"] += 1
            if own_clean:
                candidate_counts[split]["flagged_rows_with_same_split_screened_sibling"] += 1
                source_counts[f"{split}_same_split_candidate_sources"].add(source_sha)
                if len(own_semantics) == 1:
                    candidate_counts[split]["one_unique_screened_semantic_target_same_split"] += 1
            else:
                candidate_counts[split]["flagged_rows_with_other_split_sibling_only"] += 1
            candidate_counts[split].update(f"candidate_reason:{reason}" for reason in row["reasons"])
            candidates.append({
                "split": split, "line": row["line"], "source_sha256": source_sha,
                "flagged_semantic_sha256": row["semantic_sha"],
                "reasons": "|".join(row["reasons"]),
                "screened_sibling_coordinates": "|".join(f"{r['split']}:{r['line']}" for r in clean),
                "screened_sibling_splits": "|".join(sorted(clean_splits)),
                "screened_sibling_semantic_targets": len(semantic_shas),
                "same_split_screened_siblings": len(own_clean),
                "same_split_semantic_targets": len(own_semantics),
            })
    args.report.mkdir(parents=True, exist_ok=True)
    csv_path = args.report / "candidate_rows.csv"
    if candidates:
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(sorted(candidates, key=lambda r: (r["split"], r["line"])))
    result = {
        "source_index": str(args.index.resolve()),
        "original_data_modified": False,
        "training_labels_modified": False,
        "policy": {
            "match": "Exact final-user source SHA256 only; no normalized or paraphrase substitution",
            "screened_sibling": "Not reserved; strict valid; no URL preprocessing error; reconstructed sequence <=4096; canonical target <=2048; all listed warning flags zero",
            "warning_fields": list(WARNING_FIELDS),
            "important_limit": "No-warning does not certify semantic completeness. These are alternatives for review, not approved repairs.",
            "new_unique_sources_recovered_if_existing_siblings_are_retained": 0,
            "preferred_action": "Verify and retain one correct existing target per exact source; discard defective alternatives instead of copying a sibling into a second duplicate training row.",
            "cross_split": "Cross-split candidates require regrouping and rebuilding splits; never transfer a held-out label into training while continuing to claim it as independent validation.",
            "budget_limit": "Token screens use reconstructed Gemma3 framing, not verified E2B chat-template counts.",
        },
        "base_counts": {key: dict(value) for key, value in base_counts.items()},
        "base_screened_unique_sources": {key: len(value) for key, value in base_clean_sources.items()},
        "base_screened_unique_semantic_pairs": {key: len(value) for key, value in base_clean_pairs.items()},
        "candidate_counts": {key: dict(value) for key, value in candidate_counts.items()},
        "candidate_unique_source_counts": {key: len(value) for key, value in source_counts.items()},
        "reserved_sources_with_otherwise_screened_rows_never_use": reserved_alternative_examples,
        "candidate_rows_csv": csv_path.name,
    }
    (args.report / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
