from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.data.chat_templates import build_messages, build_prompt


TARGET_FORMAT = "a2ui_express_v1"


def _load_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


def _target(row: dict) -> str:
    completion = row.get("completion") or row.get("a2ui_express")
    if not completion:
        targets = row.get("completion_targets")
        if isinstance(targets, dict):
            completion = targets.get(TARGET_FORMAT)
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError(f"Row has no Express completion: {row.get('id')}")
    completion = completion.strip()
    if completion.count("<a2ui>") != 1 or completion.count("</a2ui>") != 1:
        raise ValueError(f"Row has invalid Express completion: {row.get('id')}")
    return completion


def rebuild_split(source: Path, destination: Path, prompt: str, prompt_version: str) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with source.open("r", encoding="utf-8") as src, temporary.open("w", encoding="utf-8", newline="\n") as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            response_text = row.get("response_text")
            if not isinstance(response_text, str) or not response_text.strip():
                raise ValueError(f"{source}:{line_number}: missing response_text")
            completion = _target(row)
            row["messages"] = build_messages(
                prompt,
                response_text,
                completion,
                target_format=TARGET_FORMAT,
            )
            row["prompt"] = build_prompt(
                prompt,
                response_text,
                target_format=TARGET_FORMAT,
            )
            row["prompt_version"] = prompt_version
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                metadata["prompt_version"] = prompt_version
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(destination)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild prepared split prompts without changing IR targets.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-version", default="genui_gen_mobile_a2ui_express_v1_compact")
    args = parser.parse_args()

    prompt = _load_prompt(args.prompt_file)
    total = 0
    for split in ("train", "val"):
        source = args.source_dir / f"{split}.jsonl"
        destination = args.output_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        count = rebuild_split(source, destination, prompt, args.prompt_version)
        print(f"{split}: {count} rows -> {destination}", flush=True)
        total += count
    print(f"total: {total}", flush=True)


if __name__ == "__main__":
    main()
