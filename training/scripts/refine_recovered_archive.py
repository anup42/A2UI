#!/usr/bin/env python3
"""Review/repair an archive copy, reserve Goldens, and rebuild validation.

Consumes immutable v5 output and original quarantined rows. All new data and
evidence go to fresh directories. No generator, network, or model is invoked.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
from ir_training.data.archive_recovery import (
    placeholder_tokens,
    recover_pair,
    text_sha256,
)
from ir_training.data.archive_refinement import (
    POLICY,
    FamilyUnion,
    NearSourceIndex,
    choose_validation,
    component_shape,
    repair_exact_text,
    review_warnings,
    source_signature,
)
from ir_training.data.express_preparation import TASK_PREFIX, serialize_checked
from ir_training.data.golden_replacement import response_text
from ir_training.data.ir_targets import (
    A2UI_EXPRESS_V1,
    materialize_completion_targets,
    semantic_hash,
)
from ir_training.data.shared_prompt import create_shared_prompt_contract
from recover_full_data_archive import (
    CONFIRMED_DEFECTS,
    _batch,
    _init_worker,
    _prepared_row,
    file_sha256,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def pack(value):
    return zlib.compress(
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def unpack(value):
    return json.loads(zlib.decompress(value))


def _review_batch(items):
    from ir_training.data import express_preparation as prep
    from pipeline.ir_formats import active
    from pipeline.renderer_semantics import iter_renderer_references

    results = []
    for item in items:
        coordinate, prior = item["coordinate"], item["prior"]
        row = item.get("row")
        if row is None:
            original = item["original"]
            result = recover_pair(original["source_text"], original["target_text"])
            source, target, graph = result.source_text, result.target_text, result.graph
            if graph is None or (
                not result.accepted
                and result.reason
                not in {"unresolved_content_warning", "unresolved_target_reference"}
            ):
                results.append(
                    {
                        "coordinate": coordinate,
                        "accepted": False,
                        "source": source,
                        "reason": result.reason,
                        "warnings": list(result.warnings),
                        "features": {},
                        "changes": {},
                        "resolutions": [],
                    }
                )
                continue
            recovered = {
                "source_text": source,
                "target_text": target,
                "url_map": result.url_map,
                "transformations": list(result.transformations),
                "repair_metrics": dict(result.repair_metrics),
                "effective_source_sha256": text_sha256(source),
                "effective_target_sha256": text_sha256(target),
                "effective_semantic_sha256": semantic_hash(graph),
                "source_transcode": asdict(result.source_transcode),
                "target_transcode": asdict(result.target_transcode),
            }
            row = _prepared_row(
                original, recovered, prior["source_family"], item["contract"]
            )
        else:
            source, target = row["response_text"], row["completion"]
            if row["messages"][-2:] != [
                {"role": "user", "content": TASK_PREFIX + source},
                {"role": "assistant", "content": target},
            ]:
                raise ValueError(f"Broken original message binding: {coordinate}")
            recovery = row["metadata"]["archive_recovery"]
            if (text_sha256(source), text_sha256(target)) != (
                recovery["effective_source_sha256"],
                recovery["effective_target_sha256"],
            ):
                raise ValueError(f"Broken original effective hashes: {coordinate}")
            graph = active.decode_express_completion(target)
            prep._validate_wire(graph)
            seen, queue = set(), [graph["root"]]
            while queue:
                key = queue.pop()
                if key in seen:
                    continue
                seen.add(key)
                queue.extend(
                    edge.target_id
                    for edge in iter_renderer_references(graph["elements"][key])
                )
            if seen != set(graph["elements"]):
                raise ValueError(f"Unreachable original component: {coordinate}")
            if semantic_hash(graph) != recovery["effective_semantic_sha256"]:
                raise ValueError(f"Broken original semantic hash: {coordinate}")
        graph, changes = repair_exact_text(source, graph)
        if changes or item.get("row") is None:
            checked = serialize_checked(
                materialize_completion_targets(graph)[A2UI_EXPRESS_V1], "root-first"
            )
            graph, target = checked.graph, checked.text
        warnings, resolutions, features = review_warnings(source, target, graph)
        if not placeholder_tokens(target) <= placeholder_tokens(source):
            warnings.append("unresolved_target_reference")
        # The El Nino example's only confirmed defect was reversible CP437
        # corruption, now verified by the v5 transport repair. Other reviewed
        # missing-content cases remain excluded regardless of lexical scores.
        old_split, old_line = coordinate.split(":")
        if (old_split, int(old_line)) in CONFIRMED_DEFECTS - {("train", 3855)}:
            warnings.append("confirmed_content_defect")
        if coordinate == "train:3855" and not warnings:
            resolutions.append("confirmed_encoding_defect_resolved")
        accepted = not warnings
        info = {
            "coordinate": coordinate,
            "accepted": accepted,
            "source": source,
            "reason": "passed_refined_review"
            if accepted
            else "unresolved_content_or_reference",
            "warnings": sorted(set(warnings)),
            "features": features,
            "changes": changes,
            "resolutions": resolutions,
        }
        if accepted:
            metadata = row["metadata"]
            recovery = metadata["archive_recovery"]
            metadata["archive_refinement"] = {
                "policy_version": POLICY,
                "coordinate": coordinate,
                "prior_category": prior["category"],
                "prior_reason": prior["reason"],
                "prior_effective_target_sha256": recovery["effective_target_sha256"],
                "additional_repairs": changes,
                "review_resolutions": resolutions,
                "original_assigned_split": old_split,
                "stage3_run": False,
                "content_synthesized": False,
            }
            recovery["effective_target_sha256"] = text_sha256(target)
            recovery["effective_semantic_sha256"] = semantic_hash(graph)
            for key, count in changes.items():
                recovery["transformations"].append(key)
                recovery.setdefault("repair_metrics", {})[key] = count
                row["repair"]["changes"].append(
                    {"kind": key, "lossless": False, "source_grounded": True}
                )
            row["repair"]["applied"] = bool(row["repair"]["changes"])
            row["completion"] = target
            row["messages"][-1]["content"] = target
            info.update(
                row_blob=pack(row),
                semantic=recovery["effective_semantic_sha256"],
                pair=text_sha256(source + "\0" + target),
            )
        results.append(info)
    return results


def ordered_results(items, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        pending, ready, index = {}, {}, 0
        for batch_index, batch in enumerate(_batch(items, 16)):
            pending[pool.submit(_review_batch, batch)] = batch_index
            while len(pending) + len(ready) >= workers * 3:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    ready[pending.pop(future)] = future.result()
                while index in ready:
                    yield from ready.pop(index)
                    index += 1
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                ready[pending.pop(future)] = future.result()
            while index in ready:
                yield from ready.pop(index)
                index += 1


def inputs(args, decisions, contract):
    for split in ("train", "val"):
        with (args.base_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                coordinate = row["metadata"]["archive_recovery"]["coordinate"]
                yield {
                    "coordinate": coordinate,
                    "prior": decisions[coordinate],
                    "row": row,
                }
    inventory = sqlite3.connect(
        args.inventory.resolve().as_uri() + "?mode=ro", uri=True
    )
    handles = {
        split: (args.source_dir / f"{split}.jsonl").open("rb")
        for split in ("train", "val")
    }
    try:
        for coordinate, prior in decisions.items():
            if prior["category"] != "QUARANTINE":
                continue
            split, line = prior["split"], int(prior["line"])
            offset, length, row_sha, src_sha, target_sha = inventory.execute(
                "SELECT byte_offset,byte_length,row_sha256,source_sha256,target_sha256 FROM rows WHERE split=? AND line=?",
                (split, line),
            ).fetchone()
            handles[split].seek(offset)
            raw = json.loads(
                handles[split].read(length).decode("utf-16-le").removeprefix("\ufeff")
            )
            messages = raw["messages"]
            source, target = (
                messages[-2]["content"][len(TASK_PREFIX) :],
                messages[-1]["content"],
            )
            if (
                text_sha256(source) != src_sha
                or text_sha256(target) != target_sha
                or src_sha != prior["original_source_sha256"]
            ):
                raise ValueError(f"Raw/index/decision mismatch: {coordinate}")
            original = {
                "split": split,
                "line": line,
                "row_sha256": row_sha,
                "source_sha256": src_sha,
                "target_sha256": target_sha,
                "source_text": source,
                "target_text": target,
            }
            yield {
                "coordinate": coordinate,
                "prior": prior,
                "original": original,
                "contract": contract,
            }
    finally:
        inventory.close()
        for handle in handles.values():
            handle.close()


def load_goldens():
    sources, hashes, files = {}, set(), {}
    for name, folder in (
        ("golden32", "golden32_archive_repeat_v1"),
        ("golden35", "golden35_v1"),
    ):
        path = ROOT / "training/data/eval" / folder / f"{name}.jsonl"
        manifest_path = path.with_name("benchmark_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if file_sha256(path) != manifest["output_sha256"]:
            raise ValueError(f"Frozen {name} file changed")
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != manifest["row_count"]:
            raise ValueError(f"Frozen {name} row count changed")
        for i, row in enumerate(rows, 1):
            sources[f"{name}:{i}"] = source_signature(response_text(row))
        for entry in manifest.get("excluded_sources", []):
            if entry.get("response_sha256"):
                hashes.add(entry["response_sha256"])
        for p in (path, manifest_path):
            files[p.relative_to(ROOT).as_posix()] = file_sha256(p)
    return sources, hashes, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 0 < args.val_fraction < 0.5:
        raise ValueError("Invalid workers or validation fraction")
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "plan_only",
                    "policy_version": POLICY,
                    "new_copy": str(args.output_dir),
                    "validation_fraction": args.val_fraction,
                    "seed": args.seed,
                },
                indent=2,
            )
        )
        return 0
    for path in (args.output_dir, args.audit_dir, args.report_dir):
        if path.exists():
            raise FileExistsError(f"Fresh destination required: {path}")
        for source in (args.source_dir, args.base_dir):
            if (
                path.resolve() == source.resolve()
                or path.resolve() in source.resolve().parents
            ):
                raise ValueError("An output cannot overwrite an input or its ancestor")
    base = json.loads((args.base_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        base.get("status") != "candidate_export_complete"
        or base.get("policy_version") != "messages-archive-recovery-v5-20260913"
    ):
        raise ValueError("This refinement requires the verified frozen v5 candidate")
    snapshots = {}
    for split, expected in base["source_files"].items():
        path = args.source_dir / f"{split}.jsonl"
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Original source differs: {path}")
        snapshots[str(path.resolve())] = expected["sha256"]
    for name, digest in base["outputs"].items():
        path = args.base_dir / name
        if file_sha256(path) != digest:
            raise ValueError(f"v5 input differs: {path}")
        snapshots[str(path.resolve())] = digest
    snapshots[str((args.base_dir / "manifest.json").resolve())] = file_sha256(
        args.base_dir / "manifest.json"
    )
    golden_sources, excluded_hashes, golden_files = load_goldens()
    golden_index = NearSourceIndex(golden_sources)
    golden_exact: dict[str, list[str]] = defaultdict(list)
    for key, signature in golden_sources.items():
        golden_exact[text_sha256(signature)].append(key)
    with (args.base_dir / "decisions.csv").open(encoding="utf-8", newline="") as stream:
        decisions = {
            f"{row['split']}:{row['line']}": row for row in csv.DictReader(stream)
        }
    if len(decisions) != base["all_rows_reconciled"]:
        raise ValueError("v5 decisions do not reconcile")
    partial = args.output_dir.with_name(
        args.output_dir.name + f".partial-{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    args.audit_dir.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(args.audit_dir / "review.sqlite")
    db.execute(
        "CREATE TABLE rows(coordinate TEXT PRIMARY KEY, old_family TEXT, signature TEXT, accepted INTEGER, reason TEXT, warnings TEXT, changes TEXT, resolutions TEXT, features TEXT, semantic TEXT, pair TEXT, row_blob BLOB, family TEXT, assigned_split TEXT, sort_key TEXT)"
    )
    families = FamilyUnion()
    normalized_families, benchmark_matches, reserved_families = {}, {}, set()
    benchmark_cache = {}
    begin = time.monotonic()
    contract = create_shared_prompt_contract(ordering="root-first")
    for n, result in enumerate(
        ordered_results(inputs(args, decisions, contract), args.workers), 1
    ):
        coordinate = result["coordinate"]
        prior = decisions[coordinate]
        family = prior["source_family"]
        signature = source_signature(result["source"])
        signature_hash = text_sha256(signature)
        families.find(family)
        if signature_hash in normalized_families:
            families.union(family, normalized_families[signature_hash])
        else:
            normalized_families[signature_hash] = family
        if signature_hash not in benchmark_cache:
            matches = [
                {
                    "benchmark_case": key,
                    "kind": "normalized_reference_insensitive_exact",
                }
                for key in golden_exact.get(signature_hash, [])
            ]
            if not matches:
                matches = [
                    {
                        "benchmark_case": key,
                        "kind": "lexical_near_source",
                        "jaccard": score,
                        "containment": contained,
                    }
                    for key, score, contained in golden_index.matches(
                        signature, containment=True
                    )
                ]
            benchmark_cache[signature_hash] = matches
        matches = list(benchmark_cache[signature_hash])
        if (
            prior["reason"] == "reserved_golden_family"
            or prior["original_source_sha256"] in excluded_hashes
        ):
            matches.append({"kind": "prior_reserved_or_excluded_golden_source"})
        if matches:
            reserved_families.add(family)
            benchmark_matches[coordinate] = matches
        db.execute(
            "INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                coordinate,
                family,
                signature,
                int(result["accepted"]),
                result["reason"],
                json.dumps(result["warnings"]),
                json.dumps(result["changes"]),
                json.dumps(result["resolutions"]),
                json.dumps(result["features"]),
                result.get("semantic", ""),
                result.get("pair", ""),
                result.get("row_blob"),
                None,
                None,
                text_sha256(f"{args.seed}:{coordinate}"),
            ),
        )
        if n % 2048 == 0:
            db.commit()
            print(
                json.dumps(
                    {
                        "phase": "full_review",
                        "rows": n,
                        "elapsed_seconds": round(time.monotonic() - begin),
                        "benchmark_matches": len(benchmark_matches),
                    }
                ),
                flush=True,
            )
    db.commit()
    if db.execute("SELECT COUNT(*) FROM rows").fetchone()[0] != len(decisions):
        raise ValueError("Full review omitted original rows")
    reserved_roots = {families.find(family) for family in reserved_families}
    for coordinate, old_family in db.execute(
        "SELECT coordinate,old_family FROM rows"
    ).fetchall():
        family = families.find(old_family)
        db.execute("UPDATE rows SET family=? WHERE coordinate=?", (family, coordinate))
        if family in reserved_roots:
            db.execute(
                "UPDATE rows SET accepted=0,reason='reserved_golden_family' WHERE coordinate=?",
                (coordinate,),
            )
            benchmark_matches.setdefault(
                coordinate, [{"kind": "transitive_reserved_source_family"}]
            )
    db.commit()
    groups = {}
    for family, features, length in db.execute(
        "SELECT family,features,LENGTH(signature) FROM rows WHERE accepted=1"
    ):
        shape = component_shape(json.loads(features))
        prior_group = groups.get(family)
        if prior_group is None or length > prior_group[1]:
            groups[family] = shape, length
    selected_validation = choose_validation(groups, args.val_fraction, args.seed)
    near_evidence, passes = [], []
    for iteration in range(1, 9):
        selected_roots = {families.find(family) for family in selected_validation}
        source_rows = db.execute(
            "SELECT DISTINCT family,signature FROM rows WHERE accepted=1 ORDER BY family,signature"
        ).fetchall()
        validation_sources = {
            f"{family}:{text_sha256(signature)}": signature
            for family, signature in source_rows
            if families.find(family) in selected_roots
        }
        validation_owners = {key: key.split(":", 1)[0] for key in validation_sources}
        index = NearSourceIndex(validation_sources)
        added = 0
        for count, (family, signature) in enumerate(source_rows, 1):
            # The selected roots are frozen for this pass. Newly pulled-in
            # validation families become retrieval anchors in the next pass.
            if families.find(family) in selected_roots:
                continue
            for other_key, score, contained in index.matches(signature):
                other = validation_owners[other_key]
                if families.union(family, other):
                    added += 1
                    near_evidence.append(
                        {
                            "family_a": family,
                            "family_b": other,
                            "word_trigram_jaccard": score,
                            "containment": contained,
                            "pass": iteration,
                        }
                    )
            if count % 25000 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "validation_family_review",
                            "pass": iteration,
                            "sources": count,
                            "new_family_edges": added,
                        }
                    ),
                    flush=True,
                )
        passes.append(
            {
                "pass": iteration,
                "validation_source_variants": len(validation_sources),
                "new_family_edges": added,
            }
        )
        print(
            json.dumps({"phase": "validation_family_review_complete", **passes[-1]}),
            flush=True,
        )
        if added == 0:
            break
    else:
        raise ValueError(
            "Validation near-family grouping did not converge after eight full scans"
        )
    validation_roots = {families.find(family) for family in selected_validation}
    for coordinate, old_family in db.execute(
        "SELECT coordinate,old_family FROM rows"
    ).fetchall():
        family = families.find(old_family)
        db.execute(
            "UPDATE rows SET family=?,assigned_split=? WHERE coordinate=?",
            (family, "val" if family in validation_roots else "train", coordinate),
        )
    seen_semantics, seen_pairs = {}, {}
    for coordinate, family, semantic, pair in db.execute(
        "SELECT coordinate,family,semantic,pair FROM rows WHERE accepted=1 ORDER BY sort_key"
    ).fetchall():
        if (family, semantic) in seen_semantics or pair in seen_pairs:
            db.execute(
                "UPDATE rows SET accepted=0,reason='duplicate_source_family_target' WHERE coordinate=?",
                (coordinate,),
            )
        else:
            seen_semantics[family, semantic] = coordinate
            seen_pairs[pair] = coordinate
    db.commit()
    output_counts, categories, reasons, transitions = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    category_by_original = defaultdict(Counter)
    remaining_warnings, accepted_repairs, repair_operations, resolutions = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    component_rows = defaultdict(Counter)
    output_families = defaultdict(set)
    moved = Counter()
    emitted_decisions = {}
    for split in ("train", "val"):
        with (partial / f"{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for coordinate, family, blob, feature_json in db.execute(
                "SELECT coordinate,family,row_blob,features FROM rows WHERE accepted=1 AND assigned_split=? ORDER BY sort_key",
                (split,),
            ):
                row = unpack(blob)
                row["source_id"] = f"archive-source-family-sha256:{family}"
                row["metadata"]["source_id"] = row["source_id"]
                row["metadata"]["assigned_split"] = split
                row["metadata"]["archive_refinement"]["assigned_split"] = split
                stream.write(
                    json.dumps(
                        row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                )
                category = "REPAIR" if row["repair"]["applied"] else "KEEP"
                emitted_decisions[coordinate] = category
                output_counts[split] += 1
                output_families[split].add(family)
                component_rows[split].update(json.loads(feature_json).keys())
                moved[f"{decisions[coordinate]['split']}->{split}"] += 1
    fields = [
        "split",
        "line",
        "category",
        "reason",
        "assigned_split",
        "prior_category",
        "prior_reason",
        "source_family",
        "warnings",
        "additional_repairs",
        "review_resolutions",
    ]
    with (
        (partial / "decisions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as all_stream,
        (partial / "quarantine.csv").open(
            "w", encoding="utf-8", newline=""
        ) as quarantine_stream,
    ):
        writer, quarantine = (
            csv.DictWriter(all_stream, fields),
            csv.DictWriter(quarantine_stream, fields),
        )
        writer.writeheader()
        quarantine.writeheader()
        for (
            coordinate,
            family,
            accepted,
            reason,
            warnings,
            changes,
            resolved,
            assigned_split,
        ) in db.execute(
            "SELECT coordinate,family,accepted,reason,warnings,changes,resolutions,assigned_split FROM rows ORDER BY coordinate"
        ):
            prior = decisions[coordinate]
            category = emitted_decisions.get(coordinate, "QUARANTINE")
            record = {
                "split": prior["split"],
                "line": prior["line"],
                "category": category,
                "reason": reason,
                "assigned_split": assigned_split if accepted else "",
                "prior_category": prior["category"],
                "prior_reason": prior["reason"],
                "source_family": family,
                "warnings": "|".join(json.loads(warnings)),
                "additional_repairs": changes,
                "review_resolutions": "|".join(json.loads(resolved)),
            }
            writer.writerow(record)
            categories[category] += 1
            category_by_original[prior["split"]][category] += 1
            reasons[reason] += 1
            transitions[f"{prior['category']}->{category}"] += 1
            if accepted:
                accepted_repairs.update(json.loads(changes).keys())
                repair_operations.update(json.loads(changes))
                resolutions.update(json.loads(resolved))
            else:
                quarantine.writerow(record)
                remaining_warnings.update(json.loads(warnings))
    dump(partial / "benchmark_exclusions.json", benchmark_matches)
    dump(partial / "near_source_groups.json", near_evidence)
    if output_families["train"] & output_families["val"]:
        raise ValueError("Final split families overlap")
    for path, digest in snapshots.items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Input changed during refinement: {path}")
    for name, digest in golden_files.items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen benchmark changed: {name}")
    summary = {
        "status": "candidate_export_complete",
        "schema_version": 2,
        "policy_version": POLICY,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - begin, 2),
        "all_rows_reconciled": len(decisions),
        "source_files": base["source_files"],
        "immutable_input_sha256": snapshots,
        "benchmark_file_sha256": golden_files,
        "benchmark_source_signature_sha256": sorted(golden_exact),
        "categories": {
            **{key: dict(value) for key, value in category_by_original.items()},
            "combined": dict(categories),
        },
        "output_rows": dict(output_counts),
        "source_families": {key: len(value) for key, value in output_families.items()},
        "first_reason_counts": dict(reasons),
        "transitions_from_v5": dict(transitions),
        "additional_repaired_rows_by_kind_overlap": dict(accepted_repairs),
        "additional_repair_operations": dict(repair_operations),
        "accepted_review_resolutions_overlap": dict(resolutions),
        "remaining_warning_counts_overlap": dict(remaining_warnings),
        "component_presence_rows": {
            key: dict(value) for key, value in component_rows.items()
        },
        "original_to_new_split": dict(moved),
        "benchmark_excluded_rows": len(benchmark_matches),
        "split_rebuild": {
            "seed": args.seed,
            "initial_validation_family_fraction": args.val_fraction,
            "stratification": "observed component shape plus source character-length bins; source families indivisible",
            "assignment": "seeded SHA256 selection and row order; near-source connected families move together to validation",
            "near_source_passes": passes,
            "new_near_family_edges": len(near_evidence),
            "near_method": "up to 24 rare trigram anchors, >=2 shared anchors, length ratio >=0.65, full trigram Jaccard >=0.60",
            "benchmark_near_method": "same plus length ratio >=0.40 and >=0.90 trigram containment for >=50-word sources",
            "remaining_detected_near_cross_split_matches": 0,
        },
        "checks": {
            "all_original_and_v5_hashes_unchanged": True,
            "frozen_goldens_unchanged": True,
            "all_retained_targets_reparsed_wire_validated": True,
            "all_changed_targets_strict_roundtrip": True,
            "train_val_family_overlap": 0,
            "accepted_known_golden_family_matches": 0,
            "target_placeholder_absent_source": 0,
        },
        "limitations": [
            "Lexical near-source retrieval is bounded, not exhaustive semantic paraphrase detection.",
            "Valid means passed this offline schema/content policy, not proof every fact is visible or correct.",
            "Exact model tokenizer/template and sequence-budget preparation is still required.",
            "The archive has no original generator IDs, URL maps for premasked references, or intent labels.",
            "Rebuilt validation is held out only for a fresh training run; an old checkpoint may already have seen it.",
            "No Stage3, model training, device rendering, or Golden score was produced.",
        ],
        "implementation_sha256": {
            str(p.relative_to(ROOT).as_posix()): file_sha256(p)
            for p in (
                Path(__file__),
                ROOT / "training/src/ir_training/data/archive_refinement.py",
                ROOT / "training/src/ir_training/data/archive_recovery.py",
            )
        },
        "outputs": {
            path.name: file_sha256(path) for path in partial.iterdir() if path.is_file()
        },
    }
    dump(partial / "manifest.json", summary)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(args.output_dir)
    dump(args.report_dir / "summary.json", summary)
    db.close()
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(args.output_dir),
                "output_rows": dict(output_counts),
                "categories": dict(categories),
                "transitions": dict(transitions),
                "benchmark_excluded_rows": len(benchmark_matches),
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
