#!/usr/bin/env python3
"""Review every v9 row into an immutable v10 copy; never generate Stage 3 data."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
from ir_training.common.parallel import ordered_bounded_map, resolve_prepare_workers
from ir_training.data.archive_final_review import paragraph_gaps
from ir_training.data.archive_letter_review import letter_gaps
from ir_training.data.archive_recovery import placeholder_tokens, text_sha256
from ir_training.data.archive_refinement import review_warnings
from ir_training.data.archive_semantic_review import POLICY, process_graph, review_graph
from ir_training.data.express_preparation import TASK_PREFIX, _api, serialize_checked
from finalize_archive_boundaries import check_export_space
from recover_full_data_archive import _init_worker, file_sha256
from refine_recovered_archive import dump, load_goldens
from verify_recovered_archive import load_decisions, verify_row


def process_row(item):
    split, number, raw = item
    row = json.loads(raw)
    active, express, semantic_hash, _ = _api()
    metadata, source, target = row["metadata"], row["response_text"], row["completion"]
    recovery = metadata["archive_recovery"]
    graph = active.decode_express_completion(target)
    result = process_graph(source, graph,
        url_map=metadata.get("url_preprocessing", {}).get("url_map", {}),
        original_source_sha256=recovery["original_source_sha256"])
    issues = result["issues"]
    event = {"coordinate": recovery["coordinate"], "input_split": split, "input_line": number,
             "input_source_sha256": text_sha256(source), "input_target_sha256": text_sha256(target),
             "changes": result["changes"], "proofs": result["proofs"], "issues": issues}
    if issues:
        return None, event
    changed = bool(result["proofs"])
    checked = serialize_checked(express.encode(result["graph"], shorten_ids=False) if changed else target, "root-first")
    if checked.graph != result["graph"]:
        raise ValueError(f"Graph changed during canonical serialization: {event['coordinate']}")
    warnings, _, _ = review_warnings(result["source"], checked.text, checked.graph)
    if paragraph_gaps(result["source"], checked.graph):
        warnings.append("legacy_paragraph_gap")
    if letter_gaps(result["source"], checked.graph):
        warnings.append("legacy_letter_gap")
    if not placeholder_tokens(checked.text) <= placeholder_tokens(result["source"]):
        warnings.append("target_reference_absent_from_source")
    if warnings:
        event["issues"] = [{"code": code, "detail": "Existing content/closure gate"} for code in sorted(set(warnings))]
        return None, event
    # A second pass proves no opportunistic repair remains after emission.
    replay = process_graph(result["source"], checked.graph, url_map=result["url_map"],
                           original_source_sha256=recovery["original_source_sha256"])
    if replay["proofs"] or replay["issues"]:
        raise ValueError(f"Non-idempotent review at {event['coordinate']}: {replay['changes']}, {replay['issues']}")
    if not changed and checked.text != target:
        # Preserve an already validated target byte-for-byte when no repair ran.
        output_target = target
    else:
        output_target = checked.text
    row["response_text"], row["completion"] = result["source"], output_target
    row["messages"][-2] = {"role": "user", "content": TASK_PREFIX + result["source"]}
    row["messages"][-1] = {"role": "assistant", "content": output_target}
    for key in ("a2ui_express",):
        if key in row:
            row[key] = output_target
    if "completion_targets" in row:
        row["completion_targets"] = {"a2ui_express_v1": output_target}
    if "canonical_graph" in row:
        row["canonical_graph"] = checked.graph
    if "semantic_hash" in row:
        row["semantic_hash"] = checked.semantic_sha256
    recovery["effective_source_sha256"] = text_sha256(result["source"])
    recovery["effective_target_sha256"] = text_sha256(output_target)
    recovery["effective_semantic_sha256"] = semantic_hash(checked.graph)
    for kind, count in result["changes"].items():
        recovery["transformations"].append(kind)
        recovery["repair_metrics"][kind] = count
        row["repair"]["changes"].append({"kind": kind, "lossless": False, "source_grounded": True})
    row["repair"]["applied"] = bool(row["repair"]["changes"])
    if "url_preprocessing" in metadata:
        metadata["url_preprocessing"]["url_map"] = result["url_map"]
    metadata["archive_semantic_review"] = {"policy_version": POLICY, "prior_source_sha256": text_sha256(source),
        "prior_target_sha256": text_sha256(target), "repairs": result["changes"], "proofs": result["proofs"],
        "stage3_run": False, "original_source_family_preserved": True}
    event["output_source_sha256"] = recovery["effective_source_sha256"]
    event["output_target_sha256"] = recovery["effective_target_sha256"]
    return row, event


def iter_rows(base, decisions):
    for split in ("train", "val"):
        with (base / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for number, raw in enumerate(stream, 1):
                row = json.loads(raw)
                verify_row(row, split, decisions)
                yield split, number, raw


def export(base_dir: Path, output_dir: Path, report_dir: Path, workers: int):
    for path in (output_dir, report_dir):
        if path.exists():
            raise FileExistsError(f"Fresh destination required: {path}")
        if path.resolve() == base_dir.resolve() or base_dir.resolve() in path.resolve().parents:
            raise ValueError("Output/report must not be inside the immutable base dataset")
    manifest_path = base_dir / "manifest.json"
    base = json.loads(manifest_path.read_text(encoding="utf-8"))
    if base.get("schema_version") != 5 or base.get("status") != "candidate_export_complete":
        raise ValueError("Requires a completed v9 dataset")
    space = check_export_space(output_dir, 2 * sum((base_dir / name).stat().st_size for name in base["outputs"]))
    snapshots = {str(manifest_path.resolve()): file_sha256(manifest_path)}
    for name, expected in base["outputs"].items():
        path = base_dir / name
        if Path(name).name != name or file_sha256(path) != expected:
            raise ValueError(f"Unsafe or changed base artifact: {name}")
        snapshots[str(path.resolve())] = expected
    _, _, golden_files = load_goldens()
    if golden_files != base["benchmark_file_sha256"]:
        raise ValueError("Golden sources differ from the frozen v9 benchmark inventory")
    decisions, _ = load_decisions(base_dir / "decisions.csv")
    partial = output_dir.with_name(output_dir.name + f".partial-{os.getpid()}")
    partial.mkdir(parents=True, exist_ok=False)
    counts, changes, reasons, input_counts = Counter(), Counter(), Counter(), Counter()
    split_outcomes = defaultdict(Counter)
    transitions, seen_pairs, seen_family_semantics, families = {}, set(), set(), defaultdict(set)
    started, last_log = time.monotonic(), 0.0
    print(f"Review {sum(base['output_rows'].values()):,} v9 rows using {workers} CPU workers; output staging: {partial}", flush=True)
    with (partial / "train.jsonl").open("w", encoding="utf-8", newline="\n") as train, (partial / "val.jsonl").open("w", encoding="utf-8", newline="\n") as val, (partial / "semantic_review.jsonl").open("w", encoding="utf-8", newline="\n") as audit:
        outputs = {"train": train, "val": val}
        for row, event in ordered_bounded_map(process_row, iter_rows(base_dir, decisions), workers=workers, initializer=_init_worker, batch_size=16):
            split = event["input_split"]
            input_counts[split] += 1
            if row is not None:
                semantic = row["metadata"]["archive_recovery"]["effective_semantic_sha256"]
                pair = text_sha256(row["response_text"] + "\0" + row["completion"])
                family_semantic = (row["source_id"], semantic)
                if pair in seen_pairs or family_semantic in seen_family_semantics:
                    event["issues"].append({"code": "post_repair_duplicate", "detail": "Duplicate effective pair or family/target"})
                    row = None
                else:
                    seen_pairs.add(pair)
                    seen_family_semantics.add(family_semantic)
            if row is None:
                event["outcome"] = "QUARANTINE"
                codes = sorted({issue["code"] for issue in event["issues"]})
                reasons.update(codes)
                transitions[event["coordinate"]] = ("QUARANTINE", codes)
            else:
                event["outcome"] = "REPAIR" if event["changes"] else "KEEP"
                event["output_line"] = counts[split] + 1
                changes.update(event["changes"].keys())
                category = "REPAIR" if row["repair"]["applied"] else "KEEP"
                transitions[event["coordinate"]] = (category, [])
                counts[split] += 1
                families[split].add(row["source_id"])
                outputs[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            split_outcomes[split][event["outcome"]] += 1
            audit.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            now = time.monotonic()
            if now - last_log >= 10:
                processed = sum(input_counts.values())
                rate = processed / max(now - started, .001)
                print(f"Reviewed {processed:,}/{sum(base['output_rows'].values()):,}; retained={sum(counts.values()):,}; quarantined={sum(x['QUARANTINE'] for x in split_outcomes.values()):,}; {rate:.1f} rows/s; ETA {(sum(base['output_rows'].values())-processed)/max(rate,.001):.0f}s", flush=True)
                last_log = now
    if dict(input_counts) != base["output_rows"] or families["train"] & families["val"]:
        raise ValueError("Input counts or source-family separation failed")
    if not counts["train"] or not counts["val"]:
        raise ValueError("Refusing to publish an empty training or validation split")
    categories, original_categories, first_reasons = Counter(), defaultdict(Counter), Counter()
    with (base_dir / "decisions.csv").open(encoding="utf-8", newline="") as old, (partial / "decisions.csv").open("w", encoding="utf-8", newline="") as new, (partial / "quarantine.csv").open("w", encoding="utf-8", newline="") as rejected:
        reader = csv.DictReader(old)
        writer, quarantine = csv.DictWriter(new, reader.fieldnames), csv.DictWriter(rejected, reader.fieldnames)
        writer.writeheader()
        quarantine.writeheader()
        for decision in reader:
            coordinate = decision["split"] + ":" + decision["line"]
            if coordinate in transitions:
                category, codes = transitions[coordinate]
                decision["category"] = category
                if category == "QUARANTINE":
                    decision.update(reason="semantic_review_v10", assigned_split="", warnings="|".join(codes))
                elif category != decisions[coordinate][0]:
                    decision["reason"] = "source_proven_semantic_repair_v10"
            writer.writerow(decision)
            categories[decision["category"]] += 1
            original_categories[decision["split"]][decision["category"]] += 1
            first_reasons[decision["reason"]] += 1
            if decision["category"] == "QUARANTINE":
                quarantine.writerow(decision)
    for name in base["outputs"]:
        if name not in {"train.jsonl", "val.jsonl", "decisions.csv", "quarantine.csv"}:
            copyfile(base_dir / name, partial / name)
    catalog = ROOT / "training/data/quality/v9_manual100_findings.json"
    copyfile(catalog, partial / "reviewed_findings.json")
    for path, digest in snapshots.items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Input changed during export: {path}")
    if any(file_sha256(ROOT / name) != value for name, value in golden_files.items()):
        raise ValueError("Golden files changed during export")
    summary = {**base, "schema_version": 6, "policy_version": POLICY,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "output_rows": dict(counts),
        "categories": {**{key: dict(value) for key, value in original_categories.items()}, "combined": dict(categories)},
        "first_reason_counts": dict(first_reasons), "source_families": {k: len(v) for k,v in families.items()},
        "semantic_review": {"input_rows": dict(input_counts), "outcomes_from_v9": {k: dict(v) for k,v in split_outcomes.items()},
            "retained_repair_rows_by_kind_overlap": dict(changes), "new_quarantine_reasons_overlap": dict(reasons),
            "all_retained_rows_strict_checked": sum(counts.values()), "all_retained_rows_idempotence_checked": sum(counts.values()),
            "split_membership_preserved_no_reshuffle": True, "originals_unchanged": True,
            "stage3_run": False, "elapsed_seconds": round(time.monotonic()-started, 2),
            "remaining_limit": "Heuristic content gates and a finite reviewed-source holdlist are not a full factual/semantic/device/tokenizer certification."},
        "disk_space_preflight": space, "immutable_input_sha256": {**base["immutable_input_sha256"], **snapshots},
        "reviewed_findings_sha256": file_sha256(catalog),
        "implementation_sha256": {**base["implementation_sha256"], **{name: file_sha256(ROOT / name) for name in [
            "training/scripts/finalize_archive_semantics.py", "training/src/ir_training/data/archive_semantic_review.py",
            "training/src/ir_training/data/url_preprocess.py"]}},
        "outputs": {p.name: file_sha256(p) for p in partial.iterdir() if p.is_file()}}
    for stale in ("remaining_warning_counts_overlap", "component_presence_rows", "original_to_new_split", "elapsed_seconds"):
        summary.pop(stale, None)
    dump(partial / "manifest.json", summary)
    partial.rename(output_dir)
    dump(report_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "rows": dict(counts), "semantic_review": summary["semantic_review"]}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    workers = resolve_prepare_workers(args.workers)
    if workers > 32:
        parser.error("--workers must be between 0 (auto) and 32")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "policy": POLICY, "workers": workers, "base_dir": str(args.base_dir), "output_dir": str(args.output_dir)}))
        return 0
    export(args.base_dir, args.output_dir, args.report_dir, workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
