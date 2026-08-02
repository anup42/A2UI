#!/usr/bin/env python3
"""Migrate legacy FlatSpec/Compact IR records to the single Express target.

This is an explicit, offline migration boundary. Active Stage 3 generation and
training never import the legacy decoder; every accepted record is compiled to
Express, decoded again, compiled to standard A2UI v0.9 wire, and checked against
one semantic hash before it is written.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
DATASET_SRC = ROOT / "dataset" / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.flat_spec_contract import coerce_and_validate  # noqa: E402
from pipeline.ir_formats import (  # noqa: E402
    A2UI_EXPRESS_V1,
    A2UI_V1_WIRE,
    FLAT_SPEC_V1,
    decode_to_flat_spec,
    encode_from_flat_spec,
    semantic_hash,
    serialized_text,
    compile_express_to_wire,
    decode_express_completion,
    encode_express_completion,
)
from migration import compact_ir_v2  # noqa: E402


@dataclass(frozen=True)
class InputRecord:
    record: dict[str, Any]
    source_path: str
    row_number: int


def _sha256(value: Any) -> str:
    return hashlib.sha256(serialized_text(value).encode("utf-8")).hexdigest()


def _iter_file(path: Path) -> Iterable[InputRecord]:
    if path.suffix.lower() == ".jsonl":
        for row_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{row_number} is not a JSON object")
            yield InputRecord(dict(value), str(path), row_number)
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        for row_number, item in enumerate(value, 1):
            if not isinstance(item, Mapping):
                raise ValueError(f"{path}:{row_number} is not a JSON object")
            yield InputRecord(dict(item), str(path), row_number)
    elif isinstance(value, Mapping):
        yield InputRecord(dict(value), str(path), 1)
    else:
        raise ValueError(f"{path} does not contain a JSON object or array")


def _iter_input(input_value: str) -> Iterable[InputRecord]:
    if input_value == "-":
        for row_number, line in enumerate(sys.stdin.read().splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"stdin:{row_number} is not a JSON object")
            yield InputRecord(dict(value), "stdin", row_number)
        return
    path = Path(input_value).resolve()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".json", ".jsonl"}:
                yield from _iter_file(child)
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    yield from _iter_file(path)


def _raw_value(record: Mapping[str, Any]) -> Any:
    for key in ("genui_json", "a2ui_json", "genui_raw_completion", "completion", "ir"):
        if key in record and record[key] not in (None, ""):
            return record[key]
    raise ValueError("record has no genui_json/a2ui_json/genui_raw_completion/completion/ir")


def _parse_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if "<a2ui>" in stripped:
        return stripped
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _decode_legacy(value: Any, *, input_format: str = "auto") -> tuple[str, dict[str, Any]]:
    parsed = _parse_text(value)
    hint = str(input_format or "auto").strip().lower()
    if hint in {"express", A2UI_EXPRESS_V1}:
        if not isinstance(parsed, str):
            raise ValueError("a2ui_express_v1 input must be an Express text completion")
        return A2UI_EXPRESS_V1, decode_express_completion(parsed)
    if hint in {"wire", A2UI_V1_WIRE}:
        decoded = decode_to_flat_spec(parsed, format_hint=A2UI_V1_WIRE)
        return A2UI_V1_WIRE, decoded.flat_spec
    if hint in {"flat_spec", FLAT_SPEC_V1}:
        decoded = decode_to_flat_spec(parsed, format_hint=FLAT_SPEC_V1)
        return FLAT_SPEC_V1, decoded.flat_spec
    if hint in {"compact", "compact_ir", "compact_ir_v2", "gci2"}:
        if not isinstance(parsed, Mapping):
            raise ValueError("compact_ir_v2 input must be an object")
        flat = compact_ir_v2.decode(parsed)
        result = coerce_and_validate(flat)
        if not result.is_valid or result.spec is None:
            raise ValueError(result.error or "Compact IR did not satisfy the canonical graph contract")
        return "compact_ir_v2", result.spec
    if hint != "auto":
        raise ValueError(
            "Unsupported --input-format; choose auto, flat_spec_v1, compact_ir_v2, "
            "a2ui_express_v1, or a2ui_v1_wire"
        )
    if isinstance(parsed, Mapping) and parsed.get("v") == compact_ir_v2.VERSION:
        legacy_format = "compact_ir_v2"
        flat = compact_ir_v2.decode(parsed)
        result = coerce_and_validate(flat)
        if not result.is_valid or result.spec is None:
            raise ValueError(result.error or "Compact IR did not satisfy the canonical graph contract")
        return legacy_format, result.spec
    if isinstance(parsed, Mapping) and isinstance(parsed.get("root"), str) and isinstance(parsed.get("elements"), Mapping):
        decoded = decode_to_flat_spec(parsed, format_hint=FLAT_SPEC_V1)
        return FLAT_SPEC_V1, decoded.flat_spec
    if isinstance(parsed, str) and "<a2ui>" in parsed:
        decoded = decode_to_flat_spec(parsed, format_hint=A2UI_EXPRESS_V1)
        return A2UI_EXPRESS_V1, decoded.flat_spec
    try:
        decoded = decode_to_flat_spec(parsed)
    except ValueError as exc:
        raise ValueError(f"unsupported legacy IR: {exc}") from exc
    return decoded.source_format, decoded.flat_spec


def _migrate(record: Mapping[str, Any], *, input_format: str = "auto") -> dict[str, Any]:
    legacy = _raw_value(record)
    legacy_format, flat = _decode_legacy(legacy, input_format=input_format)
    expected_hash = semantic_hash(flat)
    express = encode_express_completion(flat)
    express_flat = decode_express_completion(express)
    express_hash = semantic_hash(express_flat)
    if express_hash != expected_hash:
        raise ValueError("Express migration changed semantic hash")
    wire = compile_express_to_wire(express_flat)
    wire_flat = decode_to_flat_spec(wire, format_hint=A2UI_V1_WIRE).flat_spec
    wire_hash = semantic_hash(wire_flat)
    if wire_hash != expected_hash:
        raise ValueError("A2UI wire compilation changed semantic hash")
    updated = dict(record)
    updated["legacy_completion"] = legacy
    updated["legacy_source_format"] = legacy_format
    updated["legacy_source_hash"] = _sha256(legacy)
    # ``completion`` is the only active training target.  Keep the source
    # graph exclusively in the audit field above; do not overwrite it into a
    # legacy-looking active column.
    updated.pop("genui_json", None)
    updated.pop("a2ui_json", None)
    updated["completion"] = express
    updated["completion_targets"] = {A2UI_EXPRESS_V1: express}
    updated["a2ui_wire"] = wire
    updated["source_format"] = A2UI_EXPRESS_V1
    updated["target_format"] = A2UI_EXPRESS_V1
    updated["semantic_hash"] = expected_hash
    updated["migration_status"] = "accepted"
    updated["migration_codec_identity"] = {
        "source_format": legacy_format,
        "target_format": A2UI_EXPRESS_V1,
        "wire_format": A2UI_V1_WIRE,
        "semantic_hash": expected_hash,
    }
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON/JSONL file, directory, or '-' for JSONL stdin")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=("auto", "flat_spec_v1", "compact_ir_v2", "a2ui_express_v1", "a2ui_v1_wire"),
        help="Override automatic source detection at the migration boundary.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any record is rejected")
    parser.add_argument("--allow-rejects", action="store_true", help="Write rejected.jsonl and continue")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    manifest_path = output_dir / "migration_manifest.json"
    completed: set[str] = set()
    if args.resume and accepted_path.exists():
        for line in accepted_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("legacy_source_hash"):
                completed.add(str(row["legacy_source_hash"]))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen = set(completed)
    for item in _iter_input(args.input):
        try:
            legacy = _raw_value(item.record)
            source_hash = _sha256(legacy)
            if source_hash in seen:
                continue
            migrated = _migrate(item.record, input_format=args.input_format)
            migrated["migration_source_path"] = item.source_path
            migrated["migration_source_row"] = item.row_number
            accepted.append(migrated)
            seen.add(source_hash)
        except Exception as exc:  # noqa: BLE001 - each row is audited independently
            try:
                legacy = _raw_value(item.record)
            except Exception:
                legacy = None
            rejected.append(
                {
                    "migration_status": "rejected",
                    "migration_source_path": item.source_path,
                    "migration_source_row": item.row_number,
                    "legacy_completion": legacy,
                    "legacy_source_hash": _sha256(legacy) if legacy is not None else None,
                    "reason": str(exc),
                }
            )

    component_coverage: dict[str, int] = {}
    action_coverage: dict[str, int] = {}
    before_chars = 0
    after_chars = 0
    for row in accepted:
        legacy = row.get("legacy_completion")
        completion = row.get("completion")
        before_chars += len(serialized_text(legacy))
        after_chars += len(str(completion or ""))
        try:
            graph = decode_express_completion(str(completion))
            for element in graph.get("elements", {}).values():
                if isinstance(element, Mapping):
                    name = str(element.get("type") or "unknown")
                    component_coverage[name] = component_coverage.get(name, 0) + 1
                    events = element.get("on")
                    if isinstance(events, Mapping):
                        for action in events.values():
                            if isinstance(action, Mapping):
                                action_name = str(action.get("action") or "unknown")
                                action_coverage[action_name] = action_coverage.get(action_name, 0) + 1
        except Exception:
            pass
    manifest = {
        "manifest_version": "a2ui_express_legacy_migration_v1",
        "input": args.input,
        "target_format": A2UI_EXPRESS_V1,
        "wire_format": A2UI_V1_WIRE,
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "eligible": len(accepted) + len(rejected),
        "rejection_categories": {
            str(reason): sum(1 for row in rejected if str(row.get("reason")) == reason)
            for reason in sorted({str(row.get("reason")) for row in rejected})
        },
        "component_coverage": dict(sorted(component_coverage.items())),
        "action_coverage": dict(sorted(action_coverage.items())),
        "reference_coverage": "validated by canonical graph and standard A2UI wire round-trip",
        "before_after_character_counts": {"legacy": before_chars, "a2ui_express": after_chars},
        "token_counts": {
            "status": "BLOCKED",
            "reason": "Exact deployed tokenizer is not available in this environment; character counts are diagnostics only.",
        },
        "rejection_policy": "strict" if args.strict else ("allow" if args.allow_rejects else "report"),
        "semantic_hash_policy": "Express decode and standard A2UI wire decode must equal legacy canonical hash",
    }
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume else "w"
        with accepted_path.open(mode, encoding="utf-8") as handle:
            for row in accepted:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        if rejected or not args.resume:
            with rejected_path.open(mode if args.resume else "w", encoding="utf-8") as handle:
                for row in rejected:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.strict and rejected:
        return 2
    if rejected and not args.allow_rejects and not args.dry_run:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
