"""Reproduce the v10 source-identity, split and benchmark overlap census.

Run from the repository root: python training/reports/v10_full_audit_20260926/leakage_audit.py
Reads the complete restored v10 train/val JSONL files. Writes only beside this script.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time


REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = REPO / "training/outputs/datasets/full_data_archive_recovered_v10"
EVAL = REPO / "training/data/eval"
sys.path.insert(0, str(REPO / "training/src"))
from ir_training.data.archive_refinement import source_signature  # noqa: E402


WORD = re.compile(r"\w+", re.UNICODE)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def whitespace(text: str) -> str:
    return " ".join(text.split())


def words(text: str) -> list[str]:
    return WORD.findall(text.casefold())


def trigrams(tokens: list[str]) -> set[tuple[str, str, str]]:
    return set(zip(tokens, tokens[1:], tokens[2:]))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def load_jsonl(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            yield line, json.loads(raw), digest


def benchmark_rows() -> tuple[list[dict], dict]:
    result = []
    files = {}
    for cohort, relative in (
        ("golden35_accepted", "golden35_v1/golden35.jsonl"),
        ("golden32_accepted", "golden32_archive_repeat_v1/golden32.jsonl"),
        ("bixby50", "bixby50_v1/bixby50.jsonl"),
    ):
        path = EVAL / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line, raw in enumerate(path.open(encoding="utf-8"), 1):
            row = json.loads(raw)
            result.append({"cohort": cohort, "case": row["source_id"], "line": line,
                           "source": row["response_text"], "known_hashes": set()})

    golden_manifest_path = EVAL / "golden35_v1/benchmark_manifest.json"
    golden_manifest = json.loads(golden_manifest_path.read_text(encoding="utf-8"))
    files[str(golden_manifest_path.relative_to(EVAL))] = hashlib.sha256(golden_manifest_path.read_bytes()).hexdigest()
    run_path = REPO / golden_manifest["source_run"] / "responses.jsonl"
    files[str(run_path.relative_to(REPO))] = hashlib.sha256(run_path.read_bytes()).hexdigest()
    by_query = {row["query_id"]: row for row in map(json.loads, run_path.open(encoding="utf-8"))}
    for entry in golden_manifest["excluded_sources"]:
        case = entry["source_id"]
        result.append({"cohort": "golden35_excluded", "case": case, "line": None,
                       "source": by_query[case]["response_text"],
                       "known_hashes": set(entry.get("response_sha256s", []))})

    golden32_manifest_path = EVAL / "golden32_archive_repeat_v1/benchmark_manifest.json"
    golden32_manifest = json.loads(golden32_manifest_path.read_text(encoding="utf-8"))
    files[str(golden32_manifest_path.relative_to(EVAL))] = hashlib.sha256(golden32_manifest_path.read_bytes()).hexdigest()
    for entry in golden32_manifest["excluded_sources"]:
        result.append({"cohort": "golden32_excluded_hash_only", "case": entry["source_id"],
                       "line": None, "source": None,
                       "known_hashes": {entry["response_sha256"]}})
    for row in result:
        source = row["source"]
        row["raw"] = sha(source) if source is not None else None
        row["whitespace"] = sha(whitespace(source)) if source is not None else None
        row["reference_insensitive"] = sha(source_signature(source)) if source is not None else None
    return result, files


def near_index(validation: list[dict]) -> tuple[dict, dict, list[dict]]:
    word_df: Counter[str] = Counter()
    for row in validation:
        row["tokens"] = words(row["source_text"])
        row["shingles"] = trigrams(row["tokens"])
        word_df.update(set(row["tokens"]))
    anchors: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    selected_counts = []
    for idx, row in enumerate(validation):
        candidates = list(enumerate(zip(row["tokens"], row["tokens"][1:], row["tokens"][2:])))
        selected: set[tuple[str, str, str]] = set()
        for quarter in range(4):
            low, high = len(candidates) * quarter // 4, len(candidates) * (quarter + 1) // 4
            ranked = sorted(candidates[low:high], key=lambda item: (
                -sum(math.log((len(validation) + 1) / (1 + word_df[word])) for word in item[1]), item[0]))
            used = []
            for position, shingle in ranked:
                if shingle in selected or any(abs(position - prior) < 3 for prior in used):
                    continue
                selected.add(shingle)
                used.append(position)
                if len(used) == 6:
                    break
        selected_counts.append(len(selected))
        for shingle in selected:
            anchors[shingle].add(idx)
    return anchors, {"distinct_anchors": len(anchors), "min_per_validation": min(selected_counts),
                     "max_per_validation": max(selected_counts)}, validation


def near_pairs(train: dict, source: str, anchors: dict, validation: list[dict], counts: Counter,
               retained: list[dict]):
    tokens = words(source)
    shingles = trigrams(tokens)
    hits: Counter[int] = Counter()
    for shingle in shingles:
        for idx in anchors.get(shingle, ()):
            hits[idx] += 1
    counts["one_anchor_candidate_pairs"] += len(hits)
    for idx, common in hits.items():
        if common < 2:
            continue
        counts["two_anchor_candidate_pairs"] += 1
        val = validation[idx]
        length_ratio = min(len(tokens), len(val["tokens"])) / max(1, len(tokens), len(val["tokens"]))
        if length_ratio < 0.65:
            continue
        counts["pairs_after_length_filter"] += 1
        score = jaccard(shingles, val["shingles"])
        if score < 0.60:
            continue
        ratio = SequenceMatcher(None, tokens, val["tokens"], autojunk=False).ratio()
        retained.append({"train_line": train["line"], "val_line": val["line"],
                         "train_source_sha256": train["raw"], "val_source_sha256": val["raw"],
                         "exact_whitespace_source": train["whitespace"] == val["whitespace"],
                         "same_reference_insensitive_source": train["reference_insensitive"] == val["reference_insensitive"],
                         "same_target": train["target"] == val["target"],
                         "trigram_jaccard": round(score, 6), "ordered_word_ratio": round(ratio, 6),
                         "word_length_ratio": round(length_ratio, 6), "shared_anchors": common,
                         "source_preview": train["preview"]})


def summarize_index(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    repeats = [group for group in groups.values() if len(group) > 1]
    cross = [group for group in repeats if len({row["split"] for row in group}) > 1]
    divergent = [group for group in repeats if len({row["target"] for row in group}) > 1]
    semantic_divergent = [group for group in repeats if len({row["semantic"] for row in group}) > 1]
    return {"unique_groups": len(groups), "repeated_groups": len(repeats),
            "rows_in_repeated_groups": sum(map(len, repeats)),
            "extra_rows_beyond_one_per_group": len(rows) - len(groups),
            "groups_with_multiple_raw_targets": len(divergent),
            "groups_with_multiple_semantic_hashes": len(semantic_divergent),
            "cross_split_groups": len(cross),
            "cross_split_val_rows": sum(row["split"] == "val" for group in cross for row in group),
            "cross_split_train_rows": sum(row["split"] == "train" for group in cross for row in group)}


def main():
    started = time.monotonic()
    benchmarks, benchmark_files = benchmark_rows()
    rows: list[dict] = []
    validation: list[dict] = []
    provenance = {"train": Counter(), "val": Counter()}
    scan = {}
    near_counts: Counter[str] = Counter()
    retained: list[dict] = []
    anchors = {}
    anchor_info = {}
    for split in ("val", "train"):
        path = DATA / f"{split}.jsonl"
        count = 0
        digest = None
        for line, raw, rolling_digest in load_jsonl(path):
            count = line
            digest = rolling_digest
            source = raw["response_text"]
            target = raw["completion"]
            meta = raw.get("metadata") or {}
            recovery = meta.get("archive_recovery") or {}
            signature = sha(source_signature(source))
            record = {"split": split, "line": line, "id": raw.get("id"), "source_id": raw.get("source_id"),
                      "raw": sha(source), "whitespace": sha(whitespace(source)),
                      "reference_insensitive": signature, "target": sha(target),
                      "semantic": recovery.get("effective_semantic_sha256"),
                      "original_source": recovery.get("original_source_sha256"),
                      "original_target": recovery.get("original_target_sha256"),
                      "coordinate": recovery.get("coordinate"),
                      "preview": whitespace(source)[:180]}
            rows.append(record)
            p = provenance[split]
            p["rows"] += 1
            p["query_id_missing"] += meta.get("query_id") is None
            p["intent_missing"] += meta.get("intent") is None
            p["original_generator_identity_declared"] += recovery.get("identity_kind") != "archive_derived_source_family_hash_not_original_generator_id"
            p["original_file_coordinate_present"] += isinstance(record["coordinate"], str) and ":" in record["coordinate"]
            p["original_source_sha_present"] += bool(record["original_source"])
            p["original_target_sha_present"] += bool(record["original_target"])
            p["stage3_run_claimed"] += recovery.get("stage3_run") is True
            p["synthetic_source_id_prefix"] += str(record["source_id"]).startswith("archive-source-family-sha256:")
            p["assigned_split_mismatch"] += meta.get("assigned_split") != split
            p["source_hash_metadata_mismatch"] += recovery.get("effective_source_sha256") != record["raw"]
            p["target_hash_metadata_mismatch"] += recovery.get("effective_target_sha256") != record["target"]
            for transformation in recovery.get("transformations") or []:
                p[f"transformation:{transformation}"] += 1
            if split == "val":
                record["source_text"] = source
                validation.append(record)
            else:
                near_pairs(record, source, anchors, validation, near_counts, retained)
            if line % 10000 == 0:
                print(f"{split}: {line:,} rows; {time.monotonic() - started:.1f}s", flush=True)
        scan[split] = {"rows": count, "bytes": path.stat().st_size, "sha256": digest.hexdigest() if digest else None}
        if split == "val":
            anchors, anchor_info, validation = near_index(validation)
            print(f"Validation near-source index: {len(anchors):,} anchors", flush=True)

    bundle = json.loads((REPO / "training/data/train/v10/bundle.json").read_text(encoding="utf-8"))
    bundle_match = {item["name"].removesuffix(".jsonl"): (scan[item["name"].removesuffix(".jsonl")]["sha256"] == item["sha256"]
                                                          and scan[item["name"].removesuffix(".jsonl")]["rows"] == item["rows"]
                                                          and scan[item["name"].removesuffix(".jsonl")]["bytes"] == item["bytes"])
                    for item in bundle["files"]}
    indexes = {field: defaultdict(list) for field in ("raw", "whitespace", "reference_insensitive", "original_source", "source_id", "target")}
    for row in rows:
        for field, index in indexes.items():
            key = row[field]
            if key is not None:
                index[key].append(row)
    census = {field: summarize_index(rows, field) for field in ("raw", "whitespace", "reference_insensitive", "source_id", "target")}
    census["original_source"] = summarize_index([row for row in rows if row["original_source"]], "original_source")
    by_split = {split: {field: summarize_index([row for row in rows if row["split"] == split], field)
                        for field in ("raw", "whitespace", "reference_insensitive", "source_id", "target")}
                for split in ("train", "val")}

    matches: dict[tuple[str, str, str, int], dict] = {}
    methods = (("raw", "raw"), ("whitespace", "whitespace"),
               ("reference_insensitive", "reference_insensitive"))
    for benchmark in benchmarks:
        for field, method in methods:
            key = benchmark[field]
            if key is None:
                continue
            for row in indexes[field].get(key, ()):
                coordinate = (benchmark["cohort"], benchmark["case"], row["split"], row["line"])
                match = matches.setdefault(coordinate, {"cohort": benchmark["cohort"], "case": benchmark["case"],
                                                       "split": row["split"], "line": row["line"],
                                                       "source_id": row["source_id"], "methods": []})
                match["methods"].append(method)
        for known_hash in benchmark["known_hashes"]:
            for field, method in (("raw", "manifest_hash_raw"), ("original_source", "manifest_hash_original_source")):
                for row in indexes[field].get(known_hash, ()):
                    coordinate = (benchmark["cohort"], benchmark["case"], row["split"], row["line"])
                    match = matches.setdefault(coordinate, {"cohort": benchmark["cohort"], "case": benchmark["case"],
                                                           "split": row["split"], "line": row["line"],
                                                           "source_id": row["source_id"], "methods": []})
                    match["methods"].append(method)
    benchmark_matches = sorted(matches.values(), key=lambda r: (r["cohort"], r["case"], r["split"], r["line"]))
    benchmark_summary = {}
    for cohort in sorted({r["cohort"] for r in benchmarks}):
        cases = {r["case"] for r in benchmarks if r["cohort"] == cohort}
        hits = [r for r in benchmark_matches if r["cohort"] == cohort]
        benchmark_summary[cohort] = {"benchmark_cases": len(cases), "matched_cases": len({r["case"] for r in hits}),
                                     "matched_train_cases": len({r["case"] for r in hits if r["split"] == "train"}),
                                     "matched_val_cases": len({r["case"] for r in hits if r["split"] == "val"}),
                                     "matching_dataset_rows": len(hits),
                                     "methods": dict(Counter(method for r in hits for method in r["methods"]))}
    near_nonexact = [pair for pair in retained if not pair["exact_whitespace_source"]]
    near_nonexact.sort(key=lambda r: (-r["trigram_jaccard"], -r["ordered_word_ratio"], r["val_line"], r["train_line"]))
    near_summary = {"method": "Casefolded Unicode word trigrams; <=24 rare validation anchors, >=2 common anchors, word-length ratio >=.65, trigram Jaccard >=.60; SequenceMatcher applied to retained pairs. Bounded lexical retrieval, not exhaustive semantic paraphrase detection.",
                    "anchors": anchor_info, "retrieval": dict(near_counts), "retained_pairs": len(retained),
                    "nonexact_pairs": len(near_nonexact),
                    "nonexact_validation_rows": len({r["val_line"] for r in near_nonexact}),
                    "thresholds": [{"jaccard": j, "sequence_ratio": s,
                                    "pairs": len([r for r in near_nonexact if r["trigram_jaccard"] >= j and r["ordered_word_ratio"] >= s]),
                                    "validation_rows": len({r["val_line"] for r in near_nonexact if r["trigram_jaccard"] >= j and r["ordered_word_ratio"] >= s})}
                                   for j, s in ((.95, .98), (.90, .95), (.80, .90), (.70, .85), (.60, 0))]}
    duplicate_groups = []
    for signature, group in indexes["raw"].items():
        if len(group) < 2:
            continue
        duplicate_groups.append({"source_sha256": signature, "rows": len(group),
                                 "raw_target_variants": len({r["target"] for r in group}),
                                 "semantic_variants": len({r["semantic"] for r in group}),
                                 "coordinates": ";".join(f"{r['split']}:{r['line']}" for r in group),
                                 "preview": group[0]["preview"]})
    duplicate_groups.sort(key=lambda r: (-r["rows"], -r["semantic_variants"], r["coordinates"]))
    report = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input_files": scan, "restored_files_match_tracked_bundle": bundle_match,
              "benchmark_files_sha256": benchmark_files, "provenance": {k: dict(v) for k, v in provenance.items()},
              "all_rows_census": census, "per_split_census": by_split, "benchmark_overlap": benchmark_summary,
              "bounded_near_source": near_summary,
              "duplicate_example_limit": 25,
              "limitations": ["Different target/semantic hashes for the same source identify divergent labels, not automatically a factual contradiction.",
                              "The reference-insensitive signature is the v10 archive policy function; it strips references and folds case/Unicode, so a collision is a review candidate.",
                              "The near-source search indexes limited lexical anchors and can miss paraphrases or short shared passages.",
                              "Original generator/run/query IDs and premasked URL maps cannot be recovered from the current v10 row fields.",
                              "This audits input membership, not the examples consumed by any already-trained checkpoint."]}
    (OUT / "leakage_counts.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, records, columns in (
        ("benchmark_overlap.csv", benchmark_matches, ["cohort", "case", "split", "line", "source_id", "methods"]),
        ("duplicate_source_examples.csv", duplicate_groups[:25], ["source_sha256", "rows", "raw_target_variants", "semantic_variants", "coordinates", "preview"]),
        ("near_cross_split.csv", near_nonexact, ["train_line", "val_line", "train_source_sha256", "val_source_sha256", "same_reference_insensitive_source", "same_target", "trigram_jaccard", "ordered_word_ratio", "word_length_ratio", "shared_anchors", "source_preview"]),
    ):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                record = dict(record)
                if filename == "benchmark_overlap.csv":
                    record["methods"] = ";".join(record["methods"])
                writer.writerow(record)
    print(json.dumps({"rows": scan, "bundle_match": bundle_match,
                      "census": {k: {key: v[key] for key in ("repeated_groups", "cross_split_groups", "cross_split_val_rows")}
                                 for k, v in census.items()}, "benchmark": benchmark_summary,
                      "near": near_summary, "elapsed_seconds": round(time.monotonic() - started, 2)}, indent=2))


if __name__ == "__main__":
    main()
