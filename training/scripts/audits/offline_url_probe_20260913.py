"""Dry-run exact, reversible URL normalization; never emit repaired datasets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

TRAINING = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAINING / "src"))
sys.path.insert(0, str(TRAINING.parent / "dataset/src"))
from ir_training.data import express_preparation as prep
from ir_training.data import url_preprocess as urls
from pipeline.ir_formats import active, express
from full_data_ir_20260913 import wire_setup


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def init_worker():
    wire_setup(check_parity=False)


def probe(item):
    split, line, source, target, source_hash, target_hash = item
    result = {"split": split, "line": line, "source_sha256": source_hash,
              "target_sha256": target_hash, "status": "", "detail": "",
              "normalized_source_sha256": "", "normalized_target_sha256": "",
              "url_map_entries": 0, "target_placeholders_absent_source": 0,
              "exact_source_restored": False, "exact_graph_restored": False,
              "normalized_strict_valid": False}
    source_tokens = set(urls._PLACEHOLDER_RE.findall(source))
    target_tokens = set(urls._PLACEHOLDER_RE.findall(target))
    if source_tokens or target_tokens:
        # Already symbolic examples are not rewritten without original maps.
        if not target_tokens <= source_tokens:
            result["status"] = "ambiguous_not_counted"
            result["detail"] = "preexisting_target_placeholders_absent_source"
        elif urls._REFERENCE_RE.search(source) or urls._REFERENCE_RE.search(target):
            result["status"] = "ambiguous_not_counted"
            result["detail"] = "mixed_existing_placeholders_and_explicit_references"
        else:
            result["status"] = "already_symbolic_closed"
        return result
    if not urls._REFERENCE_RE.search(source) and not urls._REFERENCE_RE.search(target):
        result["status"] = "no_url_transform_needed"
        return result
    try:
        original_graph = active.decode_express_completion(target)
        transformed = urls.preprocess_training_urls(source, original_graph)
        result["url_map_entries"] = len(transformed.url_map)
        result["exact_source_restored"] = urls.restore_url_placeholders(transformed.response_text, transformed.url_map) == source
        result["exact_graph_restored"] = urls.restore_url_placeholders(transformed.canonical_graph, transformed.url_map) == original_graph
        if not result["exact_source_restored"] or not result["exact_graph_restored"]:
            result["status"] = "ambiguous_not_counted"
            result["detail"] = "restoration_mismatch"
            return result
        normalized = prep.serialize_checked(express.encode(transformed.canonical_graph, shorten_ids=False), "root-first")
        result["normalized_strict_valid"] = True
        result["normalized_source_sha256"] = sha(transformed.response_text)
        result["normalized_target_sha256"] = sha(normalized.text)
        new_source_tokens = set(urls._PLACEHOLDER_RE.findall(transformed.response_text))
        new_target_tokens = set(urls._PLACEHOLDER_RE.findall(normalized.text))
        result["target_placeholders_absent_source"] = len(new_target_tokens - new_source_tokens)
        if not new_target_tokens <= new_source_tokens:
            result["status"] = "ambiguous_not_counted"
            result["detail"] = "normalized_target_placeholders_absent_source"
        elif not transformed.url_map:
            result["status"] = "no_url_transform_needed"
        else:
            result["status"] = "verified_reversible_url_normalization"
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        result["status"] = "ambiguous_not_counted"
        result["detail"] = type(exc).__name__ + ": " + str(exc)[:300]
    return result


def batch_probe(batch):
    return [probe(item) for item in batch]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--input", type=Path, default=Path("C:/Users/anupk/Downloads/training_data"))
    parser.add_argument("--audits", type=Path, default=TRAINING / "outputs/audits/full_data_20260913")
    parser.add_argument("--output", type=Path, default=TRAINING / "outputs/audits/offline_triage_20260913/url_probe")
    parser.add_argument("--report", type=Path, default=TRAINING / "reports/offline_triage_20260913/url_probe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    target_csv = args.output / "row_status.csv"
    if target_csv.exists():
        raise FileExistsError(target_csv)
    validator = wire_setup()
    with (args.report / "validator.json").open("w", encoding="utf-8") as stream:
        json.dump(validator, stream, indent=2)
    db = sqlite3.connect((args.audits / "synthesis.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("ATTACH DATABASE ? AS inventory", ((args.audits / "inventory.sqlite").resolve().as_uri() + "?mode=ro",))
    query = """SELECT t.split,t.line,i.byte_offset,i.byte_length,i.source_sha256,i.target_sha256
      FROM triage t JOIN inventory.rows i ON t.split=i.split AND t.line=i.line
      WHERE valid=1 AND reserved=0 AND url_error=0 AND sequence_tokens<=4096 AND target_tokens<=2048
      AND empty_layout=0 AND layout_only=0 AND mojibake=0 AND placeholder=0 AND missing_action=0 AND low_lexical=0 AND missing_numeric=0
      ORDER BY t.split,t.line"""
    counts, details = defaultdict(Counter), defaultdict(Counter)
    examples = defaultdict(list)
    before = {split: ((args.input / f"{split}.jsonl").stat().st_size,
                      (args.input / f"{split}.jsonl").stat().st_mtime_ns) for split in ("train", "val")}
    handles = {split: (args.input / f"{split}.jsonl").open("rb") for split in ("train", "val")}
    columns = list(probe(("test", 0, "plain", "plain", "", "")))
    started, finished = time.monotonic(), 0
    def consume(results, writer):
        nonlocal finished
        for result in results:
            writer.writerow(result)
            counts[result["split"]][result["status"]] += 1
            if result["detail"]:
                details[result["split"]][result["detail"]] += 1
            if len(examples[result["status"]]) < 8:
                examples[result["status"]].append(result)
            finished += 1
        if finished % 1024 == 0:
            print(json.dumps({"completed": finished, "elapsed_seconds": round(time.monotonic() - started, 1), "counts": dict(counts)}), flush=True)
    with target_csv.open("w", encoding="utf-8", newline="") as stream, ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        pending, batch = set(), []
        for split, line, offset, length, source_hash, target_hash in db.execute(query):
            handle = handles[split]
            handle.seek(offset)
            row = json.loads(handle.read(length).decode("utf-16-le").removeprefix("\ufeff"))
            source = row["messages"][-2]["content"].removeprefix(prep.TASK_PREFIX)
            target = row["messages"][-1]["content"]
            if sha(source) != source_hash or sha(target) != target_hash:
                raise ValueError(f"Stale original-file index: {split}:{line}")
            batch.append((split, line, source, target, source_hash, target_hash))
            if len(batch) == 32:
                pending.add(executor.submit(batch_probe, batch))
                batch = []
            if len(pending) >= args.workers * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result(), writer)
        if batch:
            pending.add(executor.submit(batch_probe, batch))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                consume(future.result(), writer)
    for handle in handles.values():
        handle.close()
    for split, original_stat in before.items():
        path = args.input / f"{split}.jsonl"
        if original_stat != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError("Original input stat changed")
    summary = {"completed_rows": finished, "candidate_screen": query, "counts": dict(counts), "detail_counts": dict(details),
               "examples": dict(examples), "elapsed_seconds": round(time.monotonic() - started, 2), "workers": args.workers,
               "row_status_csv": str(target_csv), "row_status_sha256": hashlib.sha256(target_csv.read_bytes()).hexdigest(),
               "scope": "Dry run only; no source, target, Golden, pipeline, or dataset edits. Original source and graph restoration is exact, then normalized target must be strict-valid and use only placeholders present in normalized source.",
               "limitations": ["Passing heuristic screens is not proof of complete semantic fidelity.", "Already-symbolic closed examples retain no recoverable original URL map.", "Normalized token lengths require a final model-specific import preflight; the initial screen uses original reconstructed Gemma3 lengths.", "This does not repair encoding corruption, missing facts, clipped text, or ambiguous existing placeholders.", "Fast wire engine is audit-only; see validator parity report."]}
    (args.report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed_rows": finished, "counts": dict(counts), "details": dict(details), "seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
