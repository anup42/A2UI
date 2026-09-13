"""Join completed read-only audits; produce analysis queues, never training files."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

TRAINING = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAINING / "src"))
sys.path.insert(0, str(TRAINING.parent / "dataset/src"))
from ir_training.data import express_preparation as prep


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def distribution(hist):
    total = sum(hist.values())
    result = {"count": total}
    if not total:
        return result
    result.update(min=min(hist), max=max(hist), mean=sum(k*v for k, v in hist.items()) / total)
    for name, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
        seen = 0
        for value, n in sorted(hist.items()):
            seen += n
            if seen >= total*q:
                result[name] = value
                break
    return result


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TRAINING / "outputs/audits/full_data_20260913")
    parser.add_argument("--report", type=Path, default=TRAINING / "reports/full_data_audit_20260913/synthesis")
    args = parser.parse_args()
    database = args.output / "synthesis.sqlite"
    if database.exists():
        raise FileExistsError(database)
    # A final summary is the completion marker, not the current SQLite row count.
    ir_summary = json.loads((args.report.parent / "ir/summary.json").read_text(encoding="utf-8"))
    canonical_summary = json.loads((args.report.parent / "tokens/canonical_summary.json").read_text(encoding="utf-8"))
    if ir_summary["limit"]:
        raise ValueError("Refuse a limited structural audit")
    connection = sqlite3.connect(database.resolve().as_uri(), uri=True)
    connection.execute("ATTACH DATABASE ? AS inv", ((args.output / "inventory.sqlite").resolve().as_uri() + "?mode=ro",))
    connection.execute("ATTACH DATABASE ? AS ir", ((args.output / "ir.sqlite").resolve().as_uri() + "?mode=ro",))
    mismatch = connection.execute("SELECT COUNT(*) FROM inv.rows i LEFT JOIN ir.rows r ON i.split=r.split AND i.line=r.line WHERE r.target_sha IS NULL OR r.target_sha!=i.target_sha256").fetchone()[0]
    if mismatch:
        raise ValueError(f"Audit coordinate mismatch: {mismatch}")
    connection.executescript("""
      CREATE TABLE tokens(split TEXT,line INTEGER,target_sha TEXT,target_tokens INTEGER,sequence_tokens INTEGER,PRIMARY KEY(split,line));
      CREATE TABLE quality(split TEXT,line INTEGER,missing_action INTEGER,low_lexical INTEGER,missing_numeric INTEGER,PRIMARY KEY(split,line));
      CREATE TABLE triage(split TEXT,line INTEGER,source_sha TEXT,semantic_sha TEXT,reserved INTEGER,valid INTEGER,url_error INTEGER,empty_layout INTEGER,layout_only INTEGER,mojibake INTEGER,placeholder INTEGER,missing_action INTEGER,low_lexical INTEGER,missing_numeric INTEGER,target_tokens INTEGER,sequence_tokens INTEGER,PRIMARY KEY(split,line));
    """)
    token_csv = args.output / "canonical_token_lengths.csv"
    if file_sha(token_csv) != canonical_summary["row_lengths_sha256"]:
        raise ValueError("Canonical token CSV hash differs from completed token report")
    with token_csv.open(encoding="utf-8") as stream:
        connection.executemany("INSERT INTO tokens VALUES(?,?,?,?,?)", ((r["split"], int(r["line"]), r["target_sha256"], int(r["canonical_target_tokens"]), int(r["new_reconstructed_sequence_tokens"])) for r in csv.DictReader(stream)))
    with (args.output / "quality/row_text_signals.csv").open(encoding="utf-8") as stream:
        connection.executemany("INSERT INTO quality VALUES(?,?,?,?,?)", ((r["split"], int(r["line_1based"]), int(r["source_action_label_missing"] == "True"), int(r["lexical_recall_under_half"] == "True"), int(r["numeric_half_missing_with_three_anchors"] == "True")) for r in csv.DictReader(stream)))
    expected = connection.execute("SELECT COUNT(*) FROM inv.rows").fetchone()[0]
    if expected != connection.execute("SELECT COUNT(*) FROM quality").fetchone()[0]:
        raise ValueError("Missing quality rows")
    if connection.execute("SELECT COUNT(*) FROM inv.rows i LEFT JOIN quality q ON i.split=q.split AND i.line=q.line WHERE q.line IS NULL").fetchone()[0]:
        raise ValueError("Quality row coordinates differ from inventory")
    valid_expected = sum(v["strict_valid_target"] for v in canonical_summary["counts"].values())
    if valid_expected != connection.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]:
        raise ValueError("Missing canonical token rows")
    if connection.execute("SELECT COUNT(*) FROM ir.rows r JOIN ir.targets t ON r.target_sha=t.sha LEFT JOIN tokens k ON r.split=k.split AND r.line=k.line WHERE t.valid=1 AND (k.line IS NULL OR k.target_sha!=r.target_sha)").fetchone()[0]:
        raise ValueError("Canonical token target/coordinates differ from strict audit")
    reserved_hashes = {row[0] for row in connection.execute("SELECT digest FROM inv.golden_hashes")}
    component_rows, component_nodes, property_rows = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    measurements = defaultdict(lambda: defaultdict(Counter))
    examples = defaultdict(list)
    query = """SELECT i.split,i.line,i.source_sha256,i.source_normalized_sha256,i.source_masked_sha256,
      i.source_mojibake,i.target_mojibake,i.target_extra_placeholders,i.target_missing_placeholders,i.error,
      t.valid,t.semantic_sha,t.features,r.binding_error,k.target_tokens,k.sequence_tokens,
      q.missing_action,q.low_lexical,q.missing_numeric
      FROM inv.rows i JOIN ir.rows r ON i.split=r.split AND i.line=r.line
      JOIN ir.targets t ON t.sha=r.target_sha
      JOIN quality q ON i.split=q.split AND i.line=q.line
      LEFT JOIN tokens k ON i.split=k.split AND i.line=k.line ORDER BY i.split,i.line"""
    for row in connection.execute(query):
        split, line, source_sha, norm_sha, masked_sha, source_moji, target_moji, extra, missing, error, valid, semantic, features_json, binding, target_n, sequence_n, action, lexical, numeric = row
        features = json.loads(features_json)
        reserved = bool({source_sha, norm_sha, masked_sha} & reserved_hashes)
        if json.loads(binding or "[]"):
            raise ValueError("Unexpected message binding error; do not conflate with target validity")
        empty, layout = int(features.get("empty_layout_leaves", 0) > 0), int(features.get("only_layout_or_divider", False))
        connection.execute("INSERT INTO triage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (split, line, source_sha, semantic, int(reserved), valid, int(error == "url_preprocessing_error"), empty, layout, int(bool(source_moji or target_moji)), int(bool(extra or missing)), action, lexical, numeric, target_n, sequence_n))
        if valid:
            component_rows[split].update(features["components"].keys())
            component_nodes[split].update(features["components"])
            property_rows[split].update(features["properties"].keys())
            if not reserved and error != "url_preprocessing_error" and sequence_n <= 4096 and target_n <= 2048:
                # These are screened occurrences, NOT approved clean rows.
                cohort = split + "_hash_url_length_screened"
                component_rows[cohort].update(features["components"].keys())
                component_nodes[cohort].update(features["components"])
                property_rows[cohort].update(features["properties"].keys())
            for key in ("nodes", "depth", "state_keys", "literal_chars", "canonical_chars", "empty_layout_leaves"):
                measurements[split][key][features.get(key, 0)] += 1
        for flag, present in (("layout_only", layout), ("empty_layout", empty), ("reserved", reserved)):
            if present and len(examples[f"{split}_{flag}"]) < 20:
                examples[f"{split}_{flag}"].append({"line": line, "source_sha256": source_sha, "strict_valid": bool(valid)})
    connection.commit()
    if connection.execute("SELECT COUNT(*) FROM triage").fetchone()[0] != expected:
        raise ValueError("Joined triage does not cover every input coordinate")
    for name, path in (("golden32", TRAINING / "data/eval/golden32_archive_repeat_v1/golden32.jsonl"), ("golden35", TRAINING / "data/eval/golden35_v1/golden35.jsonl")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            graph = prep.serialize_checked(json.loads(raw)["completion"], "root-first").graph
            components = Counter(e["type"] for e in graph["elements"].values())
            component_rows[name].update(components.keys())
            component_nodes[name].update(components)
            property_rows[name].update(set(f"{e['type']}.{p}" for e in graph["elements"].values() for p in e.get("props", {})))
    stats = {"row_count": expected, "source_modified": False, "training_files_created": False,
             "interpretation": ["No count in this report certifies training readiness.",
               "Reserved removal detects known accepted/excluded exact, normalized and URL-masked source hashes, not all paraphrase families.",
               "Hash/URL/length-screened cohorts retain semantic warnings and train/validation family overlaps; splits must still be rebuilt.",
               "Length screens use reconstructed Gemma3 frames, not actual deployed model-template verification.",
               "2048 target-token screen is an analytical deployment-budget scenario, not proof that every longer training label must be dropped.",
               "Component coverage counts record occurrences; not unique sources or manually approved examples."],
             "component_row_presence": component_rows, "component_node_instances": component_nodes,
             "property_row_presence": property_rows,
             "structural_distributions": {s: {k: distribution(v) for k, v in d.items()} for s, d in measurements.items()},
             "examples": examples, "splits": {}}
    for split in ("train", "val"):
        def count(where="1=1"):
            return connection.execute(f"SELECT COUNT(*) FROM triage WHERE split=? AND ({where})", (split,)).fetchone()[0]
        stages = [("all_rows", "1=1"), ("after_known_reserved_golden_source_hash_removal", "reserved=0"),
                  ("after_strict_target_check", "reserved=0 AND valid=1"),
                  ("after_url_preprocessor_quarantine", "reserved=0 AND valid=1 AND url_error=0")]
        funnel = {name: count(where) for name, where in stages}
        base = stages[-1][1]
        distinct_pairs = connection.execute(f"SELECT COUNT(*) FROM (SELECT source_sha,semantic_sha FROM triage WHERE split=? AND {base} GROUP BY source_sha,semantic_sha)", (split,)).fetchone()[0]
        flag_counts = {flag: count(f"{flag}>0") for flag in ("reserved", "url_error", "empty_layout", "layout_only", "mojibake", "placeholder", "missing_action", "low_lexical", "missing_numeric")}
        objective = base + " AND sequence_tokens<=4096 AND target_tokens<=2048"
        review = "layout_only>0 OR empty_layout>0 OR mojibake>0 OR placeholder>0 OR missing_action>0 OR low_lexical>0 OR missing_numeric>0"
        stats["splits"][split] = {"funnel": funnel, "flags_overlap_not_additive": flag_counts,
          "distinct_semantic_pairs_after_hash_strict_url_checks_NOT_row_count": distinct_pairs,
          "after_objective_screen_reconstructed_sequence_lte4096_target_lte2048": count(objective),
          "after_objective_screen_with_any_review_signal": count(f"{objective} AND ({review})"),
          "after_objective_screen_no_tested_review_signal_NOT_a_clean_quality_certificate": count(f"{objective} AND NOT ({review})"),
          "duplicate_semantic_pair_groups": connection.execute("SELECT COUNT(*) FROM (SELECT source_sha,semantic_sha FROM triage WHERE split=? AND valid=1 GROUP BY source_sha,semantic_sha HAVING COUNT(*)>1)", (split,)).fetchone()[0],
          "source_groups_with_multiple_semantic_targets": connection.execute("SELECT COUNT(*) FROM (SELECT source_sha FROM triage WHERE split=? AND valid=1 GROUP BY source_sha HAVING COUNT(DISTINCT semantic_sha)>1)", (split,)).fetchone()[0]}
    args.report.mkdir(parents=True, exist_ok=True)
    with (args.report / "component_coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        cohorts = ("train", "val", "train_hash_url_length_screened", "val_hash_url_length_screened", "golden32", "golden35")
        writer.writerow(["component", *[s + "_rows" for s in cohorts]])
        for component in sorted(set().union(*(set(v) for v in component_rows.values()))):
            writer.writerow([component, *[component_rows[s][component] for s in cohorts]])
    with (args.output / "triage_rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        cursor = connection.execute("SELECT * FROM triage ORDER BY split,line")
        writer.writerow([c[0] for c in cursor.description])
        writer.writerows(cursor)
    dump(args.report / "summary.json", stats)
    connection.close()
    print(json.dumps({"completed": True, "rows": expected, "summary": str(args.report / "summary.json")}))


if __name__ == "__main__":
    main()
