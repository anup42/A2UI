"""Count every verified URL repair with the prior reconstructed Gemma3 frame."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from offline_url_probe_20260913 import TRAINING, active, express, prep, sha, urls
from full_data_tokens_20260913 import create_shared_prompt_contract, distribution, reconstructed_prefix


TOKENIZER = None
PREFIX = ""
PREFIX_N = 0
REFERENCE_CHECKED = 0


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_worker(path, prefix):
    global TOKENIZER, PREFIX, PREFIX_N
    from tokenizers import Tokenizer
    TOKENIZER = Tokenizer.from_file(path)
    TOKENIZER.no_padding()
    TOKENIZER.no_truncation()
    PREFIX = prefix
    PREFIX_N = len(TOKENIZER.encode(prefix, add_special_tokens=False).ids)


def measure_batch(batch):
    global REFERENCE_CHECKED
    results = []
    for row, source, target in batch:
        transformed = urls.preprocess_training_urls(source, active.decode_express_completion(target))
        # Re-emission must match the exact normalized text already strict-validated.
        normalized = express.encode(transformed.canonical_graph, shorten_ids=False)
        if sha(transformed.response_text) != row["normalized_source_sha256"] or sha(normalized) != row["normalized_target_sha256"]:
            raise ValueError(f"Normalized pair differs from validated URL probe: {row['split']}:{row['line']}")
        dynamic = "user\n" + prep.TASK_PREFIX + transformed.response_text.strip() + "<end_of_turn>\n<start_of_turn>model\n"
        suffix = normalized.strip() + "<end_of_turn>\n"
        dynamic_n = len(TOKENIZER.encode(dynamic, add_special_tokens=False).ids)
        target_n = len(TOKENIZER.encode(normalized, add_special_tokens=False).ids)
        completion_n = len(TOKENIZER.encode(suffix, add_special_tokens=False).ids)
        prompt_n = PREFIX_N + dynamic_n
        result = {key: row[key] for key in ("split", "line", "source_sha256", "target_sha256", "normalized_source_sha256", "normalized_target_sha256")}
        result.update(normalized_target_tokens=target_n, normalized_completion_tokens=completion_n,
                      new_reconstructed_prompt_tokens=prompt_n,
                      new_reconstructed_sequence_tokens=prompt_n + completion_n,
                      length_status="pass" if target_n <= 2048 and prompt_n + completion_n <= 4096 else "reject_length",
                      production_reference_checked=False)
        if REFERENCE_CHECKED < 8:
            reference = prep.serialize_checked(normalized, "root-first")
            if reference.text != normalized:
                raise ValueError("Unmodified production serializer disagrees on normalized target")
            result["production_reference_checked"] = True
            REFERENCE_CHECKED += 1
        # Independently verify the cached special-token prefix boundary per batch.
        if not results and len(TOKENIZER.encode(PREFIX + dynamic, add_special_tokens=False).ids) != prompt_n:
            raise ValueError("Reconstructed token boundary mismatch")
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--input", type=Path, default=Path("C:/Users/anupk/Downloads/training_data"))
    parser.add_argument("--audits", type=Path, default=TRAINING / "outputs/audits/full_data_20260913")
    parser.add_argument("--output", type=Path, default=TRAINING / "outputs/audits/offline_triage_20260913/url_probe")
    parser.add_argument("--report", type=Path, default=TRAINING / "reports/offline_triage_20260913/url_probe")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    previous = json.loads((TRAINING / "reports/full_data_audit_20260913/tokens/summary.json").read_text(encoding="utf-8"))
    probe_summary = json.loads((args.report / "summary.json").read_text(encoding="utf-8"))
    original_csv = args.output / "row_status.csv"
    if file_sha(original_csv) != probe_summary["row_status_sha256"]:
        raise ValueError("URL candidate CSV changed")
    tokenizer_path = Path(previous["tokenizer"]["path"])
    if file_sha(tokenizer_path) != previous["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer differs from prior corpus audit")
    contract = create_shared_prompt_contract()
    if contract["contract_sha256"] != previous["shared_prompt"]["contract_sha256"]:
        raise ValueError("Shared prompt contract differs from prior corpus audit")
    prefix = reconstructed_prefix(contract["scaffold"]["messages"])
    output_csv = args.output / "length_status.csv"
    if output_csv.exists():
        raise FileExistsError(output_csv)
    db = sqlite3.connect((args.audits / "inventory.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    handles = {split: (args.input / f"{split}.jsonl").open("rb") for split in ("train", "val")}
    before = {split: ((args.input / f"{split}.jsonl").stat().st_size, (args.input / f"{split}.jsonl").stat().st_mtime_ns) for split in handles}
    counts = defaultdict(Counter)
    hist = {split: defaultdict(Counter) for split in handles}
    failures = []
    checked, begin = 0, time.monotonic()
    columns = ["split", "line", "source_sha256", "target_sha256", "normalized_source_sha256", "normalized_target_sha256",
               "normalized_target_tokens", "normalized_completion_tokens", "new_reconstructed_prompt_tokens", "new_reconstructed_sequence_tokens", "length_status", "production_reference_checked"]
    def consume(results, writer):
        nonlocal checked
        for result in results:
            writer.writerow(result)
            split = result["split"]
            counts[split][result["length_status"]] += 1
            counts[split]["production_reference_checked"] += int(result["production_reference_checked"])
            for name in columns[6:10]:
                hist[split][name][result[name]] += 1
            if result["length_status"] != "pass":
                failures.append(result)
            checked += 1
        if checked % 1024 == 0:
            print(json.dumps({"completed": checked, "seconds": round(time.monotonic() - begin, 1), "counts": dict(counts)}), flush=True)
    with original_csv.open(encoding="utf-8") as source_stream, output_csv.open("w", encoding="utf-8", newline="") as out, ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(str(tokenizer_path), prefix)) as executor:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        pending, batch = set(), []
        for row in csv.DictReader(source_stream):
            if row["status"] != "verified_reversible_url_normalization":
                continue
            split, line = row["split"], int(row["line"])
            offset, length = db.execute("SELECT byte_offset,byte_length FROM rows WHERE split=? AND line=?", (split, line)).fetchone()
            handle = handles[split]
            handle.seek(offset)
            original = json.loads(handle.read(length).decode("utf-16-le").removeprefix("\ufeff"))
            source = original["messages"][-2]["content"].removeprefix(prep.TASK_PREFIX)
            target = original["messages"][-1]["content"]
            if sha(source) != row["source_sha256"] or sha(target) != row["target_sha256"]:
                raise ValueError(f"Original changed: {split}:{line}")
            batch.append((row, source, target))
            if len(batch) == 32:
                pending.add(executor.submit(measure_batch, batch))
                batch = []
            if len(pending) >= 2 * args.workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result(), writer)
        if batch:
            pending.add(executor.submit(measure_batch, batch))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                consume(future.result(), writer)
    for split, handle in handles.items():
        handle.close()
        path = args.input / f"{split}.jsonl"
        if before[split] != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError("Original file changed")
    expected = sum(c.get("verified_reversible_url_normalization", 0) for c in probe_summary["counts"].values())
    if checked != expected:
        raise ValueError(f"Missing repairs: {checked} != {expected}")
    summary = {"completed_rows": checked, "counts": dict(counts), "rejected_rows": failures,
               "lengths": {s: {k: distribution(v) for k, v in h.items()} for s, h in hist.items()},
               "tokenizer_sha256": previous["tokenizer"]["sha256"], "shared_prompt_contract_sha256": contract["contract_sha256"],
               "frame_source": "training/scripts/audits/full_data_tokens_20260913.py::reconstructed_prefix", "boundaries": "one whole-prefix boundary check per batch",
               "row_status_sha256": probe_summary["row_status_sha256"], "length_status_csv": str(output_csv), "length_status_sha256": file_sha(output_csv),
               "seconds": round(time.monotonic() - begin, 2),
               "limitations": ["Exact local Gemma3 vocabulary plus stated reconstructed frame, not exact deployed E2B or missing-template 270M serialization.", "No transformed training targets or datasets saved; hashes bind lengths to the separately strict-validated normalized text."]}
    (args.report / "length_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed_rows": checked, "counts": dict(counts), "seconds": summary["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
