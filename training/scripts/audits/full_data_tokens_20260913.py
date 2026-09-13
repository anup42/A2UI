"""Read-only streaming token-length audit; never loads weights or changes data.

Local Gemma3 token IDs are exact. Chat counts are explicitly reconstructed:
the available exported tokenizer has no chat_template. These are not E2B IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "training/src"))
from ir_training.data.express_preparation import TASK_PREFIX  # noqa: E402
from ir_training.data.shared_prompt import create_shared_prompt_contract  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def distribution(hist: Counter) -> dict:
    count = sum(hist.values())
    if not count:
        return {"count": 0}
    order = sorted(hist.items())
    def percentile(q):
        rank = max(1, int(__import__("math").ceil(count * q)))
        done = 0
        for length, n in order:
            done += n
            if done >= rank:
                return length
    return {
        "count": count, "min": order[0][0], "max": order[-1][0],
        "mean": round(sum(k*v for k, v in hist.items()) / count, 3),
        **{f"p{int(q*100)}": percentile(q) for q in (.5, .75, .9, .95, .99)},
        "over": {str(k): sum(n for length, n in order if length > k)
                 for k in (512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 16384, 32768)},
    }


def reconstructed_prefix(scaffold: list[dict]) -> str:
    """Gemma3-style text framing, system merged into first example user turn.

    This is a documented audit assumption, not a recovered training template.
    """
    if [m["role"] for m in scaffold] != ["system", "user", "assistant"]:
        raise ValueError("Expected the three-turn system/few-shot scaffold")
    first = scaffold[0]["content"].strip() + "\n\n" + scaffold[1]["content"].strip()
    return ("<bos><start_of_turn>user\n" + first + "<end_of_turn>\n"
            + "<start_of_turn>model\n" + scaffold[2]["content"].strip()
            + "<end_of_turn>\n<start_of_turn>")


def canonical_audit(args, tokenizer) -> None:
    """Join a completed strict-validation audit without retaining target text."""
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    if file_sha(args.rows_output) != summary["row_lengths_sha256"]:
        raise ValueError("Raw token-length evidence changed")
    if file_sha(args.tokenizer) != summary["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer changed between passes")
    connection = sqlite3.connect(args.canonical_db.resolve().as_uri() + "?mode=ro", uri=True)
    expected = sum(summary["lengths"][s]["target_tokens"]["count"] for s in ("train", "val"))
    actual = connection.execute("SELECT count(*) FROM rows").fetchone()[0]
    missing = connection.execute("SELECT count(*) FROM rows r LEFT JOIN targets t ON r.target_sha=t.sha WHERE t.sha IS NULL").fetchone()[0]
    if actual != expected or missing:
        raise ValueError(f"Strict audit incomplete: rows={actual}, expected={expected}, pending_targets={missing}")
    destination = args.rows_output.with_name("canonical_token_lengths.csv")
    metrics = ["canonical_target_tokens", "canonical_completion_tokens", "new_reconstructed_prompt_tokens", "new_reconstructed_sequence_tokens"]
    fields = ["split", "line", "target_sha256", "canonical_sha256", "canonical_equals_raw", "binding_error", *metrics]
    histograms = {split: {key: Counter() for key in metrics} for split in ("train", "val")}
    counts = {split: Counter() for split in ("train", "val")}
    with args.rows_output.open(encoding="utf-8") as source, destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in csv.DictReader(source):
            split, line = row["split"], int(row["line"])
            if split not in counts:
                continue
            record = connection.execute(
                "SELECT r.target_sha,r.binding_error,t.valid,t.features,t.canonical_zlib FROM rows r JOIN targets t ON r.target_sha=t.sha WHERE r.split=? AND r.line=?",
                (split, line)).fetchone()
            if record is None or record[0] != row["target_sha256"]:
                raise ValueError(f"Strict audit coordinate mismatch at {split}:{line}")
            counts[split]["rows"] += 1
            if not record[2]:
                counts[split]["strict_invalid_target"] += 1
                continue
            features = json.loads(record[3])
            same = features["canonical_equals_raw"]
            counts[split]["strict_valid_target"] += 1
            counts[split]["canonical_equals_raw" if same else "canonical_differs_raw"] += 1
            binding_errors = json.loads(record[1] or "[]")
            if not isinstance(binding_errors, list):
                raise ValueError("Strict audit binding_error must contain a JSON list")
            if binding_errors:
                counts[split]["binding_error"] += 1
            if same:
                target_n, suffix_n = int(row["target_tokens"]), int(row["completion_tokens"])
            else:
                text = zlib.decompress(record[4]).decode("utf-8")
                if sha(text) != features["canonical_sha256"]:
                    raise ValueError("Canonical target hash mismatch")
                target_n = len(tokenizer.encode(text, add_special_tokens=False).ids)
                suffix_n = len(tokenizer.encode(text.strip() + "<end_of_turn>\n", add_special_tokens=False).ids)
            result = {"split": split, "line": line, "target_sha256": row["target_sha256"],
                      "canonical_sha256": features["canonical_sha256"], "canonical_equals_raw": int(same),
                      "binding_error": json.dumps(binding_errors) if binding_errors else "", "canonical_target_tokens": target_n,
                      "canonical_completion_tokens": suffix_n,
                      "new_reconstructed_prompt_tokens": int(row["new_reconstructed_prompt_tokens"]),
                      "new_reconstructed_sequence_tokens": int(row["new_reconstructed_prompt_tokens"]) + suffix_n}
            writer.writerow(result)
            for key in metrics:
                histograms[split][key][result[key]] += 1
            for threshold in (4096, 6144, 8192):
                counts[split][f"canonical_sequence_lte_{threshold}"] += int(result["new_reconstructed_sequence_tokens"] <= threshold)
                counts[split][f"canonical_sequence_lte_{threshold}_target_lte_2048"] += int(result["new_reconstructed_sequence_tokens"] <= threshold and target_n <= 2048)
            if line % 25000 == 0:
                output.flush()
                print(f"Canonical {split}: {line:,} rows", flush=True)
    connection.close()
    result = {"schema_version": 1, "raw_lengths_sha256": summary["row_lengths_sha256"], "strict_database": str(args.canonical_db),
              "tokenizer_sha256": summary["tokenizer"]["sha256"], "counts": counts,
              "notes": ["These lengths use production serializer's strict-valid canonical targets from the separate exhaustive IR audit.",
                        "They remain reconstructed Gemma3 chat-frame counts, not exact deployed E2B or missing-template 270M measurements.",
                        "Strict-valid target counts do not by themselves remove Golden leakage, semantic omissions, split leakage, or missing provenance."],
              "lengths": {s: {k: distribution(v) for k, v in h.items()} for s, h in histograms.items()},
              "row_lengths_path": str(destination), "row_lengths_sha256": file_sha(destination)}
    (args.output_dir / "canonical_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"canonical_summary": str(args.output_dir / "canonical_summary.json"), "counts": counts}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("C:/Users/anupk/Downloads/training_data"))
    parser.add_argument("--tokenizer", type=Path, default=Path("C:/Users/anupk/Downloads/training/runs/gemma270m_ir_lora/merged_hf/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training/reports/full_data_audit_20260913/tokens")
    parser.add_argument("--rows-output", type=Path, default=ROOT / "training/outputs/audits/full_data_20260913/token_lengths.csv")
    parser.add_argument("--limit", type=int, default=0, help="Per-split smoke limit; zero audits all rows")
    parser.add_argument("--canonical-only", action="store_true", help="Join existing token CSV to a completed strict IR audit; no raw data re-tokenization")
    parser.add_argument("--canonical-db", type=Path, default=ROOT / "training/outputs/audits/full_data_20260913/ir.sqlite")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if args.canonical_only:
        canonical_audit(args, tokenizer)
        return
    config_path = args.tokenizer.with_name("tokenizer_config.json")
    tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = create_shared_prompt_contract()
    new_scaffold = contract["scaffold"]["messages"]
    text_counts: dict[str, int] = {}
    def count(value: str) -> int:
        key = sha(value)
        if key not in text_counts:
            text_counts[key] = len(tokenizer.encode(value, add_special_tokens=False).ids)
        return text_counts[key]
    @lru_cache(maxsize=64)
    def scaffold_counts(serialized: str) -> tuple[int, int, int]:
        messages = json.loads(serialized)
        return (count(reconstructed_prefix(messages)),
                sum(count(m["content"]) for m in messages), count(messages[0]["content"]))
    new_serialized = json.dumps(new_scaffold, ensure_ascii=False, separators=(",", ":"))
    new_prefix, new_content, new_system = scaffold_counts(new_serialized)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["source_tokens", "target_tokens", "completion_tokens", "old_content_sum_tokens",
               "new_content_sum_tokens", "old_reconstructed_prompt_tokens", "new_reconstructed_prompt_tokens",
               "old_reconstructed_sequence_tokens", "new_reconstructed_sequence_tokens"]
    columns = ["split", "line", "target_sha256", "source_sha256", "scaffold_sha256", *metrics]
    histograms, scaffolds, rejected, examples, source_files = {}, {}, {}, {}, {}
    splits = {"train": args.input_dir / "train.jsonl", "val": args.input_dir / "val.jsonl",
              "golden32": ROOT / "training/data/eval/golden32_archive_repeat_v1/golden32.jsonl",
              "golden35": ROOT / "training/data/eval/golden35_v1/golden35.jsonl"}
    started = time.monotonic()
    checked_boundary = 0
    with args.rows_output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for split, path in splits.items():
            histograms[split] = {key: Counter() for key in metrics}
            rejected[split] = []
            examples[split] = []
            with path.open("rb") as probe:
                first_bytes = probe.read(4)
            encoding = "utf-16" if first_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            source_files[split] = {"path": str(path), "size_bytes": path.stat().st_size,
                                   "sha256": file_sha(path), "encoding": encoding}
            with path.open(encoding=encoding) as stream:
                for line, raw in enumerate(stream, 1):
                    if args.limit and line > args.limit and split in {"train", "val"}:
                        break
                    try:
                        row = json.loads(raw)
                        messages = row["messages"]
                        if [m["role"] for m in messages] != ["system", "user", "assistant", "user", "assistant"]:
                            raise ValueError("Expected system,user,assistant,user,assistant")
                        if not all(isinstance(m["content"], str) for m in messages):
                            raise ValueError("Non-string message content")
                        task, target = messages[-2]["content"], messages[-1]["content"]
                        if not task.startswith(TASK_PREFIX):
                            raise ValueError("Non-canonical source task prefix")
                        source = task[len(TASK_PREFIX):]
                        serialized = json.dumps(messages[:3], ensure_ascii=False, separators=(",", ":"))
                        prefix, content, system = scaffold_counts(serialized)
                        scaffold_sha = sha(serialized)
                        if scaffold_sha not in scaffolds:
                            scaffolds[scaffold_sha] = {"messages": messages[:3], "reconstructed_prefix_tokens": prefix,
                                                       "content_tokens": content, "system_tokens": system, "rows_by_split": Counter()}
                        scaffolds[scaffold_sha]["rows_by_split"][split] += 1
                        # Each cached prefix ends at a special token boundary, so
                        # adding dynamic segment lengths is exact for THIS frame.
                        dynamic = "user\n" + task.strip() + "<end_of_turn>\n<start_of_turn>model\n"
                        canonical_dynamic = "user\n" + TASK_PREFIX + source.strip() + "<end_of_turn>\n<start_of_turn>model\n"
                        suffix = target.strip() + "<end_of_turn>\n"
                        prompt_old = prefix + count(dynamic)
                        prompt_new = new_prefix + count(canonical_dynamic)
                        if checked_boundary < 100 or split.startswith("golden"):
                            assert count(reconstructed_prefix(messages[:3]) + dynamic) == prompt_old
                            assert count(reconstructed_prefix(new_scaffold) + canonical_dynamic) == prompt_new
                            checked_boundary += 1
                        target_n, completion_n = count(target), count(suffix)
                        result = dict(zip(columns[:5], [split, line, sha(target), sha(source), scaffold_sha]))
                        result.update(source_tokens=count(source), target_tokens=target_n, completion_tokens=completion_n,
                                      old_content_sum_tokens=content + count(task) + target_n,
                                      new_content_sum_tokens=new_content + count(TASK_PREFIX + source.strip()) + target_n,
                                      old_reconstructed_prompt_tokens=prompt_old, new_reconstructed_prompt_tokens=prompt_new,
                                      old_reconstructed_sequence_tokens=prompt_old + completion_n,
                                      new_reconstructed_sequence_tokens=prompt_new + completion_n)
                        writer.writerow(result)
                        for name in metrics:
                            histograms[split][name][result[name]] += 1
                        if len(examples[split]) < 10 and (result["new_reconstructed_sequence_tokens"] > 8192 or target_n > 2048):
                            examples[split].append({k: result[k] for k in ("line", "target_sha256", "target_tokens", "new_reconstructed_sequence_tokens")})
                    except (ValueError, KeyError, TypeError) as error:
                        rejected[split].append({"line": line, "error": str(error)[:250]})
                    if line % 10000 == 0:
                        output.flush()
                        print(f"{split}: {line:,} physical rows, {time.monotonic()-started:.1f}s, cache {len(text_counts):,}", flush=True)
            print(f"{split} complete: {sum(histograms[split]['target_tokens'].values()):,} measured rows", flush=True)
    result = {
        "schema_version": 1, "scope": "all physical rows" if not args.limit else f"SMOKE first {args.limit} train/val rows",
        "runtime_seconds": round(time.monotonic()-started, 2), "tokenizer": {
            "path": str(args.tokenizer), "sha256": file_sha(args.tokenizer), "vocabulary_size": tokenizer.get_vocab_size(),
            "config_sha256": file_sha(config_path), "tokenizer_class": tokenizer_config.get("tokenizer_class"),
            "has_config_chat_template": bool(tokenizer_config.get("chat_template")),
            "has_local_chat_template_file": any(args.tokenizer.parent.glob("chat_template*")),
            "special_token_ids": {x: tokenizer.token_to_id(x) for x in ("<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>")}},
        "measurement_notes": [
            "source_tokens and target_tokens use the exact local tokenizer.json, without added special tokens or truncation.",
            "content_sum_tokens sums separately encoded message bodies, not a real model chat serialization.",
            "reconstructed counts use Gemma3-style user/model turns with system prepended to the first example user, explicit BOS and end-of-turn tokens.",
            "No local chat_template exists: reconstructed counts are exact for the audit's stated frame, NOT proof of the original or deployed model's template.",
            "sequence_tokens matches the repo's separately encoded generation prefix + assistant suffix convention under the reconstructed frame.",
            "No E2B tokenizer was found: these are NOT E2B token counts. Retest with its exact deployed tokenizer/template.",
            "Every target is measured as supplied, including targets which may fail strict semantic/schema checks. Join per-row coordinates to the validation audit.",
            "Production scaffold normalization changes the prompt only; no target serialization or repair occurs in this audit."],
        "shared_prompt": {"contract_sha256": contract["contract_sha256"], "scaffold_sha256": contract["scaffold_sha256"],
                          "system_tokens": new_system, "content_tokens": new_content, "reconstructed_prefix_tokens": new_prefix},
        "source_files": source_files, "lengths": {s: {k: distribution(v) for k, v in h.items()} for s, h in histograms.items()},
        "scaffolds": scaffolds, "rejected": rejected, "long_examples": examples,
        "prefix_boundary_checks": checked_boundary, "row_lengths_path": str(args.rows_output),
        "row_lengths_sha256": file_sha(args.rows_output),
        "references": ["https://ai.google.dev/gemma/docs/core/prompt-structure"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "seconds": result["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
