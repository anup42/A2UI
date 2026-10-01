"""Bounded lexical overlap search between every v10 row and local held-out sources.

Run from the repository root: python training/reports/v10_full_audit_20260926/benchmark_near.py
No model, network, or training invocation is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from leakage_audit import benchmark_rows, jaccard, near_index, sha, trigrams, words  # noqa: E402

sys.path.insert(0, str(REPO / "training/src"))
from ir_training.data.archive_refinement import source_signature  # noqa: E402


def main():
    started = time.monotonic()
    all_benchmarks, _ = benchmark_rows()
    seen = set()
    benchmarks = []
    for row in all_benchmarks:
        if row["source"] is None or (row["cohort"], row["case"]) in seen:
            continue
        seen.add((row["cohort"], row["case"]))
        benchmarks.append({"cohort": row["cohort"], "case": row["case"],
                           "source_text": source_signature(row["source"])})
    anchors, anchor_info, benchmarks = near_index(benchmarks)
    counts: Counter[str] = Counter()
    matches = []
    files = {}
    for split in ("train", "val"):
        path = REPO / f"training/outputs/datasets/full_data_archive_recovered_v10/{split}.jsonl"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line, raw in enumerate(stream, 1):
                digest.update(raw)
                row = json.loads(raw)
                counts[f"{split}_rows"] += 1
                normalized_source = source_signature(row["response_text"])
                tokens = words(normalized_source)
                shingles = trigrams(tokens)
                hits: Counter[int] = Counter()
                for shingle in shingles:
                    for idx in anchors.get(shingle, ()):
                        hits[idx] += 1
                counts["one_anchor_candidate_pairs"] += len(hits)
                for idx, anchor_hits in hits.items():
                    if anchor_hits < 2:
                        continue
                    counts["two_anchor_candidate_pairs"] += 1
                    benchmark = benchmarks[idx]
                    length_ratio = min(len(tokens), len(benchmark["tokens"])) / max(1, len(tokens), len(benchmark["tokens"]))
                    if length_ratio < .4:
                        continue
                    counts["pairs_after_length_filter"] += 1
                    overlap = len(shingles & benchmark["shingles"])
                    union = len(shingles | benchmark["shingles"])
                    jac = overlap / max(1, union)
                    containment = overlap / max(1, min(len(shingles), len(benchmark["shingles"])))
                    if jac < .60 and not (min(len(tokens), len(benchmark["tokens"])) >= 50 and containment >= .90):
                        continue
                    counts["retained_pairs"] += 1
                    matches.append({"split": split, "line": line, "cohort": benchmark["cohort"],
                                    "case": benchmark["case"], "jaccard": round(jac, 6),
                                    "containment": round(containment, 6), "length_ratio": round(length_ratio, 6),
                                    "ordered_word_ratio": round(SequenceMatcher(None, tokens, benchmark["tokens"], autojunk=False).ratio(), 6),
                                    "shared_anchors": anchor_hits,
                                    "response_sha256": sha(row["response_text"]),
                                    "source_preview": " ".join(row["response_text"].split())[:150]})
                if line % 20000 == 0:
                    print(f"{split}: {line:,}; retained {len(matches):,}; {time.monotonic()-started:.1f}s", flush=True)
        files[split] = {"sha256": digest.hexdigest(), "rows": counts[f"{split}_rows"]}
    matches.sort(key=lambda r: (r["cohort"], r["case"], -r["jaccard"], -r["containment"], r["split"], r["line"]))
    report = {"schema_version": 1, "input_files": files,
              "benchmark_source_cases": dict(Counter(r["cohort"] for r in benchmarks)),
              "anchor_info": anchor_info, "counts": dict(counts),
              "matched_benchmark_cases": {cohort: len({r["case"] for r in matches if r["cohort"] == cohort})
                                          for cohort in sorted({r["cohort"] for r in benchmarks})},
              "thresholds": [{"jaccard_at_least": threshold,
                              "pairs": sum(r["jaccard"] >= threshold for r in matches),
                              "benchmark_cases": len({(r["cohort"],r["case"]) for r in matches if r["jaccard"] >= threshold})}
                             for threshold in (.95, .90, .80, .70, .60)],
              "method": "Reference-insensitive source_signature (repo v10 function), Unicode casefolded words, 3-word shingles; <=24 rare anchors per benchmark, >=2 common anchors, token-length ratio >=.40; retain Jaccard >=.60 or >=.90 containment of shorter >=50-word source; ordered ratio only for retained pairs.",
              "limits": "Complete dataset row scan but bounded lexical retrieval, not an exhaustive semantic paraphrase search. Different reference destinations are intentionally removed before comparison; retained matches require human task/fact review.",
              "elapsed_seconds": round(time.monotonic()-started, 2)}
    (HERE / "benchmark_near_counts.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (HERE / "benchmark_near_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["split", "line", "cohort", "case", "jaccard", "containment", "length_ratio",
                  "ordered_word_ratio", "shared_anchors", "response_sha256", "source_preview"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matches)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
