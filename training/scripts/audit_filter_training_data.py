"""Audit/filter prepared supervision while reserving both Golden32 and Golden35."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.data.audit_filter import audit_and_filter_rows, load_reserved_cohorts
from ir_training.data.golden_replacement import read_rows_strict, serialize_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reserve-golden32", type=Path, required=True)
    parser.add_argument("--reserve-golden35", type=Path, required=True)
    parser.add_argument("--reserve-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, help="Optional new directory; omission is a read-only audit")
    parser.add_argument("--require-source-identities", action="store_true")
    parser.add_argument("--tokenizer", help="Explicit local tokenizer bundle; no download is performed")
    parser.add_argument("--tokenizer-loader", choices=("auto_tokenizer", "pretrained_tokenizer_fast"), default="auto_tokenizer")
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--chat-template-kwargs", default="{}")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        loader = PreTrainedTokenizerFast if args.tokenizer_loader == "pretrained_tokenizer_fast" else AutoTokenizer
        tokenizer = loader.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    reserved = load_reserved_cohorts([args.reserve_golden32, args.reserve_golden35], args.reserve_manifest)
    accepted, quarantine, report = audit_and_filter_rows(read_rows_strict(args.input), reserved=reserved,
        tokenizer=tokenizer, max_seq_length=args.max_seq_length, max_input_tokens=args.max_input_tokens,
        chat_template_kwargs=json.loads(args.chat_template_kwargs), require_source_identities=args.require_source_identities)
    report.update(source_path=str(args.input), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "accepted.jsonl").write_bytes(serialize_rows(accepted))
        (args.output_dir / "quarantine.jsonl").write_bytes(serialize_rows(quarantine))
        report["accepted_sha256"] = hashlib.sha256(serialize_rows(accepted)).hexdigest()
        with (args.output_dir / "audit.json").open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
