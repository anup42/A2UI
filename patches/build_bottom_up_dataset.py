"""Portable CPU-only validation/serialization of prepared Express JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
from ir_training.data.express_preparation import prepare_splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="SPLIT=JSONL", help="Repeat for train, val, golden32, etc.; membership is preserved")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing data is never overwritten")
    parser.add_argument("--ordering", choices=("bottom-up", "root-first"), default="root-first", help="Same validation/emitter for both controlled representation experiments")
    parser.add_argument("--tokenizer", help="Optional tokenizer path or HF ID, loaded only when explicitly supplied")
    parser.add_argument("--tokenizer-loader", choices=("auto_tokenizer", "pretrained_tokenizer_fast", "auto_processor"), default="auto_tokenizer", help="Match model.tokenizer_loader in the training YAML")
    parser.add_argument("--max-seq-length", type=int, help="Reject full chats exceeding this selected tokenizer limit")
    parser.add_argument("--max-input-tokens", type=int, help="Reject generation prompts exceeding this selected tokenizer limit")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download tokenizer files")
    parser.add_argument("--chat-template-kwargs", default="{}", help="JSON object matching model.chat_template_kwargs in the training YAML")
    args = parser.parse_args(argv)
    if (args.max_seq_length or args.max_input_tokens) and not args.tokenizer:
        parser.error("Token limits require --tokenizer")
    inputs: dict[str, Path] = {}
    for value in args.input:
        if "=" not in value:
            parser.error("--input must have the form SPLIT=JSONL")
        split, path = value.split("=", 1)
        if split in inputs:
            parser.error(f"Duplicate input split: {split}")
        inputs[split] = Path(path)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer, AutoProcessor, PreTrainedTokenizerFast
        loader = {"auto_tokenizer": AutoTokenizer, "pretrained_tokenizer_fast": PreTrainedTokenizerFast, "auto_processor": AutoProcessor}[args.tokenizer_loader]
        loaded = loader.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
        tokenizer = getattr(loaded, "tokenizer", loaded)
    template_kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(template_kwargs, dict):
        parser.error("--chat-template-kwargs must be a JSON object")
    manifest = prepare_splits(inputs, args.output_dir, ordering=args.ordering, tokenizer=tokenizer, max_seq_length=args.max_seq_length, max_input_tokens=args.max_input_tokens, chat_template_kwargs=template_kwargs)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "ordering": args.ordering, "scaffold_count": manifest["scaffold_count"], "splits": {name: {key: values.get(key, 0) for key in ("input_rows", "accepted_rows", "quarantined_rows", "quarantine_reasons")} for name, values in manifest["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
