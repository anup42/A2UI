"""Bounded independent parity of indexed audit results with production validation.

Never imports the accelerated audit or replaces production validator functions.
Source files and SQLite indices are opened read-only; only the report is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.data import express_preparation as prep


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=45)


def load_snapshot(path: Path) -> tuple[list[dict], dict]:
    # Finish reads before invoking the slow reference validator, so a live
    # writer is not blocked for the duration of the parity exercise.
    with readonly(path) as connection:
        connection.row_factory = sqlite3.Row
        query = """SELECT r.split,r.line,t.sha,t.valid,t.reason,t.semantic_sha,
            json_extract(t.features,'$.nodes') AS nodes,
            json_extract(t.features,'$.depth') AS depth,
            json_extract(t.features,'$.canonical_chars') AS canonical_chars,
            json_extract(t.features,'$.canonical_sha256') AS canonical_sha256,
            json_extract(t.features,'$.empty_layout_leaves') AS empty_layout_leaves,
            json_extract(t.features,'$.only_layout_or_divider') AS only_layout_or_divider,
            json_extract(t.features,'$.components') AS components
            FROM rows r JOIN targets t ON t.sha=r.target_sha
            ORDER BY r.split,r.line"""
        rows = [dict(row) for row in connection.execute(query)]
    for row in rows:
        row["components"] = json.loads(row["components"] or "{}")
    return rows, {"indexed_rows_with_finished_results": len(rows),
                  "split_rows": dict(Counter(row["split"] for row in rows)),
                  "unique_targets": len({row["sha"] for row in rows}),
                  "invalid_reasons": dict(Counter(row["reason"] for row in rows if not row["valid"]))}


def select_cases(rows: list[dict], count: int, excluded: set[str], min_train_line: int) -> list[dict]:
    eligible = [row for row in rows if row["sha"] not in excluded and (row["split"] != "train" or row["line"] >= min_train_line)]
    unique = {row["sha"]: row for row in reversed(eligible)}
    population = sorted(unique.values(), key=lambda row: (row["split"], row["line"]))
    selected: dict[str, dict] = {}
    def add(row: dict, reason: str) -> None:
        value = selected.setdefault(row["sha"], {**row, "selection_reasons": []})
        if reason not in value["selection_reasons"]:
            value["selection_reasons"].append(reason)
    invalid = {}
    for row in population:
        if not row["valid"]:
            invalid.setdefault(row["reason"], row)
    for reason, row in sorted(invalid.items()):
        add(row, f"invalid_reason:{reason}")
    for feature in ("nodes", "depth", "canonical_chars"):
        candidates = sorted((r for r in population if r.get(feature) is not None), key=lambda r: (r[feature], r["split"], r["line"]))
        for row in candidates[:3]:
            add(row, f"smallest:{feature}")
        for row in candidates[-3:]:
            add(row, f"largest:{feature}")
    for flag in ("empty_layout_leaves", "only_layout_or_divider"):
        for row in [r for r in population if r.get(flag)][:6]:
            add(row, flag)
    components = {}
    for row in population:
        for component in row["components"]:
            components.setdefault(component, row)
    for component, row in sorted(components.items()):
        add(row, f"component:{component}")
    # Ensure validation coverage whenever any finished validation results exist.
    val = [row for row in population if row["split"] == "val"]
    for position in range(min(12, len(val))):
        add(val[round(position * (len(val)-1) / max(1, min(12, len(val))-1))], "validation_spread")
    if population:
        spread_count = max(0, count - len(selected))
        for position in range(spread_count):
            if len(selected) >= count:
                break
            add(population[round(position * (len(population)-1) / max(1, spread_count-1))], "line_spread")
        for row in population:
            if len(selected) >= count:
                break
            add(row, "deterministic_fill")
    return list(selected.values())


def source_target(inventory: sqlite3.Connection, source: Path, case: dict) -> str:
    identity = inventory.execute("SELECT byte_offset,byte_length,target_sha256 FROM rows WHERE split=? AND line=?", (case["split"],case["line"])).fetchone()
    if identity is None or identity[2] != case["sha"]:
        raise ValueError("IR index and independent inventory disagree on selected target identity")
    path = source / f"{case['split']}.jsonl"
    with path.open("rb") as stream:
        prefix = stream.read(4)
        encoding = "utf-16-le" if prefix.startswith(b"\xff\xfe") else "utf-16-be" if prefix.startswith(b"\xfe\xff") else "utf-8-sig"
        stream.seek(identity[0])
        row = json.loads(stream.read(identity[1]).decode(encoding).lstrip("\ufeff"))
    target = row["messages"][-1]["content"]
    if hashlib.sha256(target.encode("utf-8")).hexdigest() != case["sha"]:
        raise ValueError("Current source target differs from both saved indices")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-index", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--min-train-line", type=int, default=1)
    parser.add_argument("--prior-report", type=Path)
    args = parser.parse_args()
    excluded = set()
    if args.prior_report:
        previous = json.loads(args.prior_report.read_text(encoding="utf-8"))
        excluded = {case["sha"] for case in previous["cases"]}
    started = time.monotonic()
    rows, snapshot = load_snapshot(args.ir_index)
    cases = select_cases(rows, args.sample_count, excluded, args.min_train_line)
    del rows
    if not cases:
        raise ValueError("No completed eligible target results available for parity")
    validator = prep._wire_validator()
    if type(validator).__name__ != "Draft202012Validator" or prep._validate_wire.__module__ != prep.__name__:
        raise RuntimeError("Production validator has been replaced; parity audit must run independently")
    outcomes = []
    with readonly(args.inventory) as inventory:
        for number, case in enumerate(cases,1):
            target = source_target(inventory,args.source,case)
            actual = {"valid": False, "reason": None, "semantic_sha": None, "canonical_sha256": None}
            try:
                checked = prep.serialize_checked(target,"root-first")
                actual.update(valid=True,reason="valid",semantic_sha=checked.semantic_sha256,
                              canonical_sha256=hashlib.sha256(checked.text.encode("utf-8")).hexdigest())
            except (ValueError,TypeError,RecursionError) as exc:
                actual.update(reason=getattr(exc,"reason",type(exc).__name__),detail=str(exc)[:600])
            mismatches = []
            for key in ("valid","reason","semantic_sha","canonical_sha256"):
                if actual[key] != case.get(key):
                    mismatches.append(key)
            outcomes.append({**case,"reference":actual,"mismatches":mismatches})
            if number % 20 == 0:
                print(json.dumps({"validated":number,"total":len(cases),"mismatches":sum(bool(r["mismatches"]) for r in outcomes)}),flush=True)
    result = {"scope":"Bounded varied actual-target production parity, not exhaustive reference revalidation",
              "created_at":datetime.now(timezone.utc).isoformat(),"source":str(args.source.resolve()),
              "ir_index":str(args.ir_index.resolve()),"inventory":str(args.inventory.resolve()),
              "snapshot":snapshot,"prior_report":str(args.prior_report.resolve()) if args.prior_report else None,
              "min_train_line":args.min_train_line,"sample_count":len(outcomes),
              "matched_count":sum(not row["mismatches"] for row in outcomes),
              "mismatch_count":sum(bool(row["mismatches"]) for row in outcomes),
              "production_validator":f"{type(validator).__module__}.{type(validator).__name__}",
              "production_module_sha256":file_hash(Path(prep.__file__)),
              "wire_schema_sha256":file_hash(ROOT.parent / "dataset/schema/genuicraft_a2ui_v1_wire.schema.json"),
              "audit_script_sha256":file_hash(Path(__file__)),
              "elapsed_seconds":round(time.monotonic()-started,3),"cases":outcomes}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"sample_count":len(outcomes),"mismatch_count":result["mismatch_count"]}))
    if result["mismatch_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
