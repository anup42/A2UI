#!/usr/bin/env python3
"""Convert GenUICraft FlatSpec/Compact IR/Express/A2UI wire losslessly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DATASET_SRC = ROOT / "dataset" / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.flat_spec_contract import coerce_and_validate  # noqa: E402
from pipeline.ir_formats import (  # noqa: E402
    A2UI_EXPRESS_V1,
    A2UI_V1_WIRE,
    COMPACT_IR_V2,
    FLAT_SPEC_V1,
    decode_to_flat_spec,
    detect_format,
    encode_from_flat_spec,
    semantic_equivalent,
    semantic_hash,
    serialized_text,
)

TARGETS = (COMPACT_IR_V2, A2UI_EXPRESS_V1, A2UI_V1_WIRE, FLAT_SPEC_V1)
ALIASES = {
    "compact": COMPACT_IR_V2,
    "compact_ir": COMPACT_IR_V2,
    "gci2": COMPACT_IR_V2,
    "express": A2UI_EXPRESS_V1,
    "a2ui_express": A2UI_EXPRESS_V1,
    "a2ui": A2UI_V1_WIRE,
    "wire": A2UI_V1_WIRE,
    "flat": FLAT_SPEC_V1,
    "flatspec": FLAT_SPEC_V1,
}


def resolve_format(value: str | None) -> str | None:
    if value is None or value == "auto":
        return None
    normalized = value.strip().lower()
    result = ALIASES.get(normalized, normalized)
    if result not in TARGETS:
        raise SystemExit(f"Unsupported format {value!r}; choose from {', '.join(TARGETS)}")
    return result


def parse_payload(text: str, source_hint: str | None) -> Any:
    if source_hint == A2UI_EXPRESS_V1 or (source_hint is None and "<a2ui>" in text):
        return text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if source_hint is None:
            return text.strip()
        raise


def canonicalize(value: Any, source_hint: str | None) -> dict[str, Any]:
    decoded = decode_to_flat_spec(value, format_hint=source_hint)
    result = coerce_and_validate(decoded.flat_spec)
    if not result.is_valid or result.spec is None:
        raise ValueError(result.error or "IR failed renderer contract validation")
    return result.spec


def convert_one(value: Any, source_hint: str | None, target: str, *, pretty: bool, shorten_ids: bool) -> tuple[Any, dict[str, Any]]:
    source = source_hint or detect_format(value)
    flat = canonicalize(value, source)
    converted = encode_from_flat_spec(flat, target, shorten_ids=shorten_ids, pretty=pretty)
    roundtrip = semantic_equivalent(flat, converted)
    if not roundtrip:
        raise RuntimeError(f"Semantic round-trip failed: {source} -> {target}")
    meta = {
        "source_format": source,
        "target_format": target,
        "semantic_hash": semantic_hash(flat),
        "semantic_roundtrip_ok": True,
        "source_chars": len(serialized_text(value)),
        "target_chars": len(serialized_text(converted)),
    }
    meta["character_reduction"] = (
        0.0 if meta["source_chars"] == 0 else 1.0 - meta["target_chars"] / meta["source_chars"]
    )
    return converted, meta


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            yield value


def write_value(path: Path | None, value: Any, pretty: bool) -> None:
    text = serialized_text(value, pretty=pretty)
    if path is None:
        print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")


def convert_jsonl(args: argparse.Namespace, source_hint: str | None, targets: list[str]) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + ".converted.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(input_path):
            raw = record.get(args.input_field)
            if raw is None:
                raw = record.get("genui_raw_completion")
            if raw is None:
                raw = record.get("genui_json")
            if raw is None:
                raise ValueError(f"Record {record.get('ui_id')!r} has no {args.input_field}/genui_raw_completion/genui_json")
            if isinstance(raw, str):
                value = parse_payload(raw, source_hint)
            else:
                value = raw
            outputs: dict[str, Any] = {}
            metadata: dict[str, Any] = {}
            for target in targets:
                converted, meta = convert_one(value, source_hint, target, pretty=False, shorten_ids=not args.keep_ids)
                outputs[target] = converted
                metadata[target] = meta
            updated = dict(record)
            updated[args.output_field] = outputs if len(targets) > 1 else outputs[targets[0]]
            updated[args.output_field + "_metadata"] = metadata if len(targets) > 1 else metadata[targets[0]]
            out.write(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n")
            converted_count += 1
    print(json.dumps({"input": str(input_path), "output": str(output_path), "records": converted_count, "targets": targets}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Input file, JSON/Express text, or '-' for stdin")
    parser.add_argument("--source", default="auto", help="Source format or auto")
    parser.add_argument("--target", action="append", help="Target format; repeat for multiple")
    parser.add_argument("--both-compact", action="store_true", help="Emit Compact IR v2 and A2UI Express v1")
    parser.add_argument("--all-formats", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--keep-ids", action="store_true")
    parser.add_argument("--jsonl", action="store_true", help="Convert records in a JSONL file")
    parser.add_argument("--input-field", default="genui_json")
    parser.add_argument("--output-field", default="ir_conversions")
    args = parser.parse_args()

    source_hint = resolve_format(args.source)
    if args.all_formats:
        targets = list(TARGETS)
    elif args.both_compact:
        targets = [COMPACT_IR_V2, A2UI_EXPRESS_V1]
    else:
        targets = [resolve_format(value) for value in (args.target or [COMPACT_IR_V2])]
    targets = [target for target in targets if target is not None]

    if args.jsonl:
        if not args.input or args.input == "-":
            raise SystemExit("--jsonl requires an input path")
        return convert_jsonl(args, source_hint, targets)

    if args.input in {None, "-"}:
        text = sys.stdin.read()
    else:
        path = Path(args.input)
        text = path.read_text(encoding="utf-8") if path.exists() else args.input
    value = parse_payload(text, source_hint)
    if len(targets) == 1:
        converted, meta = convert_one(value, source_hint, targets[0], pretty=args.pretty, shorten_ids=not args.keep_ids)
        write_value(Path(args.output) if args.output else None, converted, args.pretty)
        print(json.dumps(meta, indent=2), file=sys.stderr)
        return 0

    output_dir = Path(args.output) if args.output else Path.cwd() / "ir_converted"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    extensions = {COMPACT_IR_V2: ".compact.json", A2UI_EXPRESS_V1: ".express.a2ui", A2UI_V1_WIRE: ".a2ui.json", FLAT_SPEC_V1: ".flat.json"}
    for target in targets:
        converted, meta = convert_one(value, source_hint, target, pretty=args.pretty, shorten_ids=not args.keep_ids)
        write_value(output_dir / ("output" + extensions[target]), converted, args.pretty)
        metadata[target] = meta
    (output_dir / "conversion_report.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "targets": targets}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
