"""Read-only full-corpus IR audit; write indices/reports, never training targets.

Uses production parser, renderer references, emitter and semantic roundtrip.
For scale, compile the unchanged wire schema's Draft-7-compatible keywords
with fastjsonschema; disable format/default behavior to match jsonschema.
Compare both validators on real Golden payloads and deliberate invalid copies.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset/src"))
from ir_training.data import express_preparation as prep
from ir_training.data.repairs import repair_graph
from pipeline.ir_formats import active, a2ui_wire
from pipeline.renderer_semantics import iter_renderer_references


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def wire_setup(check_parity=True):
    import fastjsonschema
    schema_path = ROOT.parent / "dataset/schema/genuicraft_a2ui_v1_wire.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    unsupported = {"unevaluatedProperties", "unevaluatedItems", "prefixItems", "dependentSchemas", "dependentRequired", "$dynamicRef", "$recursiveRef", "minContains", "maxContains"}
    def inspect(value):
        if isinstance(value, dict):
            if unsupported & value.keys():
                raise ValueError("Schema has unsupported fast-audit keywords")
            if "$ref" in value and set(value) - {"$ref", "$comment", "title", "description"}:
                raise ValueError("Schema has reference siblings; use the production validator")
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(schema)
    compiled_schema = deepcopy(schema)
    compiled_schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    compiled = fastjsonschema.compile(compiled_schema, use_default=False, use_formats=False)
    original = prep._wire_validator()
    comparisons = []
    cohorts = (("golden32", ROOT / "data/eval/golden32_archive_repeat_v1/golden32.jsonl"),
               ("golden35", ROOT / "data/eval/golden35_v1/golden35.jsonl")) if check_parity else ()
    for name, path in cohorts:
        for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(text)
            graph = active.decode_express_completion(row["completion"])
            payload = a2ui_wire.encode(graph, shorten_ids=False)
            mutated = deepcopy(payload)
            for message in mutated:
                components = message.get("updateComponents", {}).get("components", [])
                if components:
                    components[0]["__audit_unexpected_property__"] = True
                    break
            for variant, data in (("original", payload), ("invalid_copy", mutated)):
                expected = original.is_valid(data)
                try:
                    compiled(data)
                    observed = True
                except fastjsonschema.JsonSchemaException:
                    observed = False
                if expected != observed:
                    raise ValueError(f"Fast/production mismatch: {name} line {line} {variant}")
                comparisons.append({"cohort": name, "line": line, "variant": variant, "valid": expected})
    def check(graph):
        try:
            payload = a2ui_wire.encode(graph, shorten_ids=False)
        except ValueError as exc:
            raise prep.PreparationError("wire_lowering_invalid", str(exc)) from exc
        try:
            compiled(payload)
        except fastjsonschema.JsonSchemaException as exc:
            # Get concise production component diagnostics without traversing
            # every unrelated component alternative in the top-level oneOf.
            for message in payload:
                for component in message.get("updateComponents", {}).get("components", []):
                    kind = component.get("component")
                    if kind not in schema["$defs"]["components"]:
                        continue
                    validator = original.evolve(schema={"$ref": f"#/$defs/components/{kind}"})
                    failures = list(validator.iter_errors(component))
                    if failures:
                        error = failures[0]
                        location = "/".join(map(str, error.absolute_path))
                        raise prep.PreparationError("wire_schema_invalid", f"{kind}.{location}: {error.message[:400]}") from exc
            raise prep.PreparationError("wire_schema_invalid", str(exc)[:500]) from exc
    prep._validate_wire = check
    return {"schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
            "engine": "fastjsonschema, unchanged schema assertions, compatible keyword subset verified",
            "format_checks": False, "apply_defaults": False, "production_parity_cases": comparisons}


def graph_features(graph):
    elements = graph["elements"]
    counts = Counter(str(item["type"]) for item in elements.values())
    properties = Counter(f"{item['type']}.{key}" for item in elements.values() for key in (item.get("props") or {}))
    references = Counter(edge.reference_kind for item in elements.values() for edge in iter_renderer_references(item))
    stack = [(graph["root"], 1)]
    seen = set()
    depth = 0
    while stack:
        key, level = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        depth = max(depth, level)
        stack.extend((edge.target_id, level + 1) for edge in iter_renderer_references(elements[key]))
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
    literals = list(strings(graph.get("state") or {}))
    for element in elements.values():
        literals.extend(strings(element.get("props") or {}))
    changes = Counter()
    try:
        candidate = repair_graph(graph)
        for change in candidate.changes:
            changes[str(change.kind)] += 1
    except (ValueError, TypeError) as exc:
        changes["repair_probe_error:" + str(exc)[:120]] += 1
    layout_types = {"Column", "Row", "Stack", "Card", "List"}
    text_lengths = Counter(len(item.get("props", {}).get("text", "")) for item in elements.values()
                           if item.get("type") == "Text" and isinstance(item.get("props", {}).get("text"), str))
    empty_layouts = sum(1 for item in elements.values() if item["type"] in layout_types
                        and not list(iter_renderer_references(item))
                        and not any(item.get("props", {}).get(key) for key in ("title", "subtitle", "text", "statePath")))
    return {"components": dict(counts), "properties": dict(properties), "references": dict(references),
            "nodes": len(elements), "depth": depth, "reachable": len(seen),
            "state_keys": len(graph.get("state") or {}), "literal_chars": sum(map(len, literals)),
            "text_length_histogram": dict(text_lengths), "empty_layout_leaves": empty_layouts,
            "only_layout_or_divider": not (set(counts) - layout_types - {"Divider"}),
            "repair_candidate_changes": dict(changes)}


def audit_target(target):
    graph = None
    canonical = None
    try:
        checked = prep.serialize_checked(target, "root-first")
        graph = checked.graph
        canonical = checked.text
        valid, reason, detail, semantic = True, "valid", "", checked.semantic_sha256
    except (ValueError, TypeError, RecursionError) as exc:
        valid, reason, detail, semantic = False, getattr(exc, "reason", type(exc).__name__), str(exc)[:600], None
        try:
            graph = active.decode_express_completion(target)
        except (ValueError, TypeError, RecursionError):
            pass
    features = graph_features(graph) if graph is not None else {}
    if canonical is not None:
        features.update(canonical_chars=len(canonical), canonical_sha256=sha(canonical), canonical_equals_raw=canonical == target)
    return valid, reason, detail, semantic, features, zlib.compress(canonical.encode("utf-8")) if canonical is not None else None


def worker_initialize():
    wire_setup(check_parity=False)


def audit_batch(batch):
    result = []
    for target_hash, target in batch:
        valid, reason, detail, semantic, features, compressed = audit_target(target)
        result.append((target_hash, valid, reason, detail, semantic, json.dumps(features, separators=(",", ":")), compressed))
    return result


def summarize(connection, args, files, evidence, database, elapsed):
    summary = {"files": files, "database": str(database.resolve()), "source_modified": False,
               "actual_tokenizer_loaded": False, "targets_rewritten": False, "limit": args.limit,
               "validator": {k:v for k,v in evidence.items() if k != "production_parity_cases"}, "splits": {}}
    for split in ("train", "val"):
        counts = dict(connection.execute("SELECT t.reason,COUNT(*) FROM rows r LEFT JOIN targets t ON t.sha=r.target_sha WHERE r.split=? GROUP BY t.reason", (split,)))
        components, properties, references, repairs, text_lengths, flags = (Counter() for _ in range(6))
        with (args.report / f"{split}_invalid_rows.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["split", "line", "target_sha256", "reason", "detail"])
            writer.writerows(connection.execute("SELECT r.split,r.line,t.sha,t.reason,t.detail FROM rows r JOIN targets t ON t.sha=r.target_sha WHERE r.split=? AND t.valid=0 ORDER BY r.line", (split,)))
        for feature_json, n in connection.execute("SELECT t.features,COUNT(*) FROM rows r JOIN targets t ON t.sha=r.target_sha WHERE r.split=? AND t.valid=1 GROUP BY t.sha", (split,)):
            features = json.loads(feature_json)
            for counter, name in ((components,"components"),(properties,"properties"),(references,"references"),(repairs,"repair_candidate_changes"),(text_lengths,"text_length_histogram")):
                counter.update({k:v*n for k,v in features.get(name, {}).items()})
            for flag in ("only_layout_or_divider", "canonical_equals_raw"):
                if features.get(flag):
                    flags[flag] += n
            if features.get("empty_layout_leaves", 0):
                flags["has_empty_layout_leaves"] += n
        summary["splits"][split] = {"target_validation_counts": counts,
            "binding_errors": dict(connection.execute("SELECT binding_error,COUNT(*) FROM rows WHERE split=? GROUP BY binding_error", (split,))),
            "valid_occurrence_components": dict(components), "valid_occurrence_properties": dict(properties),
            "valid_occurrence_reference_kinds": dict(references), "existing_repair_layer_candidate_changes": dict(repairs),
            "text_length_histogram": dict(text_lengths), "valid_row_flags": dict(flags)}
    summary["unique_targets"] = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    summary["seconds"] = elapsed
    dump(args.report / "summary.json", summary)
    print(json.dumps({"completed": True, "summary": str(args.report / "summary.json"), "unique_targets": summary["unique_targets"], "seconds": elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("C:/Users/anupk/Downloads/training_data"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/audits/full_data_20260913")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/full_data_audit_20260913/ir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reference", action="store_true", help="Use unmodified production validator for a bounded parity audit")
    parser.add_argument("--summarize-existing", action="store_true", help="Regenerate aggregate reports from a completed full index after checking source hashes and coverage")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    database = args.output / ("ir_reference.sqlite" if args.reference else "ir.sqlite")
    if args.reference and args.workers != 1:
        raise ValueError("Reference validation requires --workers 1")
    if args.summarize_existing:
        if args.limit or args.reference:
            raise ValueError("Summary-only resume supports the complete accelerated index only")
        inventory = json.loads((args.report.parent / "inventory/files.json").read_text(encoding="utf-8"))
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        if connection.execute("SELECT COUNT(*) FROM rows r LEFT JOIN targets t ON r.target_sha=t.sha WHERE t.sha IS NULL").fetchone()[0]:
            raise ValueError("Pending or missing target validation results")
        files = []
        for split in ("train", "val"):
            expected = inventory[split]
            path = args.source / f"{split}.jsonl"
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError("Source hash changed since inventory")
            if connection.execute("SELECT COUNT(*) FROM rows WHERE split=?", (split,)).fetchone()[0] != expected["physical_lines"]:
                raise ValueError("Incomplete source row coverage")
            files.append({"path": str(path.resolve()), "bytes": expected["bytes"], "encoding": "utf-16", "sha256": digest.hexdigest(), "limited": False})
        evidence = json.loads((args.report / "validator.json").read_text(encoding="utf-8"))
        summarize(connection, args, files, evidence, database, None)
        connection.close()
        return
    if database.exists():
        raise FileExistsError(database)
    evidence = {"engine": "production jsonschema Draft202012Validator"} if args.reference else wire_setup()
    dump(args.report / ("reference_validator.json" if args.reference else "validator.json"), evidence)
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE targets(sha TEXT PRIMARY KEY,valid INTEGER,reason TEXT,detail TEXT,semantic_sha TEXT,features TEXT,canonical_zlib BLOB);
        CREATE TABLE rows(split TEXT,line INTEGER,target_sha TEXT,binding_error TEXT,PRIMARY KEY(split,line));
        CREATE INDEX row_target ON rows(target_sha);
    """)
    seen = set()
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=worker_initialize) if args.workers > 1 else None
    pending = set()
    batch = []
    def receive(block=False):
        if not pending:
            return
        done, _ = wait(pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for future in done:
            connection.executemany("INSERT INTO targets VALUES(?,?,?,?,?,?,?)", future.result())
            pending.remove(future)
    def submit():
        if not batch:
            return
        if pool is None:
            connection.executemany("INSERT INTO targets VALUES(?,?,?,?,?,?,?)", audit_batch(batch))
        else:
            pending.add(pool.submit(audit_batch, list(batch)))
            if len(pending) >= args.workers * 2:
                receive(block=True)
        batch.clear()
    started = time.time()
    files = []
    for split in ("train", "val"):
        path = args.source / f"{split}.jsonl"
        source_stat = path.stat()
        with path.open("rb") as raw_stream:
            bom = raw_stream.read(4)
        encoding = "utf-16" if bom.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        with path.open("r", encoding=encoding) as stream:
            for line, raw in enumerate(stream, 1):
                if args.limit and line > args.limit:
                    break
                errors = []
                target_hash = None
                try:
                    row = json.loads(raw)
                    messages = row.get("messages")
                    if not isinstance(messages, list) or len(messages) < 2:
                        raise ValueError("missing_chat_messages")
                    if any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
                        raise ValueError("invalid_message_shape")
                    if messages[-1].get("role") != "assistant" or messages[-2].get("role") != "user":
                        errors.append("invalid_final_roles")
                    if not messages[-2]["content"].startswith(prep.TASK_PREFIX):
                        errors.append("invalid_task_prefix")
                    target = messages[-1]["content"]
                    target_hash = sha(target)
                    if target_hash not in seen:
                        batch.append((target_hash, target))
                        seen.add(target_hash)
                        if len(batch) >= 64:
                            submit()
                    # Historical fewshots are input-only; inspect separately
                    # and do not fabricate missing source IDs/completion aliases.
                    for message in messages[:-1]:
                        if message.get("role") == "assistant":
                            try:
                                prep.serialize_checked(message["content"], "root-first")
                            except (ValueError, TypeError, RecursionError) as exc:
                                errors.append("invalid_fewshot:" + getattr(exc, "reason", type(exc).__name__))
                except (ValueError, TypeError, AttributeError, KeyError) as exc:
                    errors.append(str(exc)[:200])
                connection.execute("INSERT INTO rows VALUES(?,?,?,?)", (split, line, target_hash, json.dumps(errors)))
                if line % 1000 == 0:
                    connection.commit()
                    print(json.dumps({"split": split, "rows": line, "unique_targets": len(seen), "seconds": round(time.time()-started, 1)}), flush=True)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise ValueError("Source changed during audit")
        digest = hashlib.sha256()
        with path.open("rb") as raw_stream:
            for chunk in iter(lambda: raw_stream.read(8*1024*1024), b""):
                digest.update(chunk)
        files.append({"path": str(path.resolve()), "bytes": source_stat.st_size, "encoding": encoding, "sha256": digest.hexdigest(), "limited": bool(args.limit)})
        connection.commit()
    submit()
    while pending:
        receive(block=True)
        connection.commit()
    if pool:
        pool.shutdown()
    summarize(connection, args, files, evidence, database, round(time.time()-started, 2))
    connection.close()


if __name__ == "__main__":
    main()
