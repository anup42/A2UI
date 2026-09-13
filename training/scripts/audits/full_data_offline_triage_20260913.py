"""Count a conservative no-generation policy; never create training examples.

This joins completed audits. KEEP is provisional, REPAIR means a demonstrated
lossless representation conversion, and REJECT means exclude from this run.
It does not certify semantic fidelity or implement an archive importer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

TRAINING = Path(__file__).resolve().parents[2]
POLICY_VERSION = "offline-conservative-v1-20260913"
WARNINGS = ("layout_only", "empty_layout", "mojibake", "placeholder",
            "missing_action", "low_lexical", "missing_numeric")
CONFIRMED_DEFECTS = {
    ("train", 1), ("train", 3855), ("train", 24112), ("train", 40160),
    ("train", 93102), ("train", 111526), ("train", 121983),
    ("val", 18), ("val", 209),
}
KEEP_STATUSES = {"already_symbolic_closed", "no_url_transform_needed"}
REPAIR_STATUS = "verified_reversible_url_normalization"


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_rows(path):
    result = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["split"], int(row["line"])
            if key in result:
                raise ValueError(f"Repeated coordinate in {path}: {key}")
            result[key] = row
    return result


def base_screen(row):
    return (not row["reserved"] and row["valid"] and not row["url_error"]
            and row["sequence_tokens"] is not None and row["sequence_tokens"] <= 4096
            and row["target_tokens"] is not None and row["target_tokens"] <= 2048
            and not any(row[name] for name in WARNINGS))


def preliminary_classification(row, url_status, *, reserved_family=False,
                               validation_overlap=False, confirmed_defect=False,
                               repaired_budget_pass=None):
    """First-match precedence makes counts mutually exclusive, not additive flags."""
    if row["reserved"] or reserved_family:
        return "REJECT", "reserved_golden_family"
    if validation_overlap:
        return "REJECT", "validation_family_seen_in_original_training"
    if confirmed_defect:
        return "REJECT", "confirmed_content_defect"
    if not row["valid"]:
        return "REJECT", "invalid_target"
    if row["url_error"]:
        return "REJECT", "url_preprocessing_error"
    if (row["sequence_tokens"] is None or row["sequence_tokens"] > 4096
            or row["target_tokens"] is None or row["target_tokens"] > 2048):
        return "REJECT", "outside_selected_original_token_budget"
    if any(row[name] for name in WARNINGS):
        return "REJECT", "unresolved_quality_warning"
    if url_status in KEEP_STATUSES:
        return "KEEP", url_status
    if url_status == REPAIR_STATUS:
        if repaired_budget_pass is None:
            raise ValueError("A URL repair needs a completed post-transform token check")
        if not repaired_budget_pass:
            return "REJECT", "outside_selected_post_repair_token_budget"
        return "REPAIR", REPAIR_STATUS
    if url_status == "ambiguous_not_counted":
        return "REJECT", "unresolved_url_grounding"
    raise ValueError(f"Missing/unknown URL probe status for screened row: {url_status}")


class Families:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        if key not in self.parent:
            self.parent[key] = key
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            previous = self.parent[key]
            self.parent[key] = root
            key = previous
        return root

    def union(self, keys):
        keys = [key for key in keys if key]
        if not keys:
            raise ValueError("Source family has no hashes")
        roots = sorted({self.find(key) for key in keys})
        for root in roots[1:]:
            self.parent[root] = roots[0]


def duplicate_reason(original_pair, effective_pair, coordinate, seen_original, seen_effective):
    """Deduplicate admitted records only; keep legitimate alternative layouts."""
    if original_pair in seen_original:
        return "duplicate_exact_source_semantic_target", seen_original[original_pair]
    if effective_pair in seen_effective:
        return "duplicate_effective_source_target_text", seen_effective[effective_pair]
    seen_original[original_pair] = coordinate
    seen_effective[effective_pair] = coordinate
    return "", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, default=TRAINING / "outputs/audits/full_data_20260913")
    parser.add_argument("--report", type=Path, default=TRAINING / "reports/offline_triage_20260913")
    parser.add_argument("--output", type=Path, default=TRAINING / "outputs/audits/offline_triage_20260913")
    parser.add_argument("--source-dir", type=Path, default=Path("C:/Users/anupk/Downloads/training_data"))
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.report / "summary.json"
    row_path = args.output / "classified_rows.csv"
    if result_path.exists() or row_path.exists():
        raise FileExistsError("Use new --report/--output paths for a new policy run")

    full_report = TRAINING / "reports/full_data_audit_20260913"
    manifest_path = full_report / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_stats = {}
    for split, spec in manifest["sources"].items():
        path = args.source_dir / f"{split}.jsonl"
        before = path.stat()
        actual_hash = file_sha(path)
        after = path.stat()
        if (actual_hash != spec["sha256"] or before.st_size != spec["bytes"]
                or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise ValueError(f"Original {split} no longer matches full audit")
        source_stats[split] = {**spec, "path": str(path.resolve()), "sha256_verified_this_run": True}

    url_summary_path = args.report / "url_probe/summary.json"
    url_summary = json.loads(url_summary_path.read_text(encoding="utf-8"))
    url_path = args.output / "url_probe/row_status.csv"
    if file_sha(url_path) != url_summary["row_status_sha256"]:
        raise ValueError("URL probe output differs from its completion marker")
    url_rows = read_rows(url_path)
    if len(url_rows) != url_summary["completed_rows"]:
        raise ValueError("URL completion count mismatch")

    # The follow-on dry run retokenizes every verified URL conversion.
    length_summary_path = args.report / "url_probe/length_summary.json"
    length_summary = json.loads(length_summary_path.read_text(encoding="utf-8"))
    length_path = args.output / "url_probe/length_status.csv"
    if (file_sha(length_path) != length_summary["length_status_sha256"]
            or length_summary["row_status_sha256"] != url_summary["row_status_sha256"]):
        raise ValueError("Post-repair length output differs from completion marker")
    length_rows = read_rows(length_path)
    repair_keys = {key for key, row in url_rows.items() if row["status"] == REPAIR_STATUS}
    if set(length_rows) != repair_keys or len(length_rows) != length_summary["completed_rows"]:
        raise ValueError("Post-repair token audit must cover every URL repair exactly once")

    db = sqlite3.connect((args.audits / "synthesis.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("ATTACH DATABASE ? AS inv", ((args.audits / "inventory.sqlite").resolve().as_uri() + "?mode=ro",))
    db.execute("ATTACH DATABASE ? AS ir", ((args.audits / "ir.sqlite").resolve().as_uri() + "?mode=ro",))
    rows = [dict(row) for row in db.execute("""SELECT t.*,i.source_normalized_sha256,i.source_masked_sha256,
      i.source_sha256,i.target_sha256,r.target_sha,targets.semantic_sha AS checked_semantic_sha
      FROM triage t JOIN inv.rows i ON i.split=t.split AND i.line=t.line
      JOIN ir.rows r ON r.split=t.split AND r.line=t.line
      JOIN ir.targets targets ON targets.sha=r.target_sha ORDER BY t.split,t.line""")]
    if Counter(row["split"] for row in rows) != Counter({s: v["rows"] for s, v in source_stats.items()}):
        raise ValueError("Joined rows do not match original archive counts")
    if len({(row["split"], row["line"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate joined coordinate")
    for row in rows:
        if (row["source_sha"] != row["source_sha256"] or row["target_sha"] != row["target_sha256"]
                or row["semantic_sha"] != row["checked_semantic_sha"]):
            raise ValueError("Stale source/target/semantic binding across indices")
    expected_url_keys = {(row["split"], row["line"]) for row in rows if base_screen(row)}
    if set(url_rows) != expected_url_keys:
        raise ValueError("URL probe and full-audit screen cover different coordinates")

    families = Families()
    for row in rows:
        families.union([row["source_sha"], row["source_normalized_sha256"], row["source_masked_sha256"]])
    near_path = full_report / "near_duplicates/pairs.json"
    near = json.loads(near_path.read_text(encoding="utf-8"))
    coordinates = {(row["split"], row["line"]): row for row in rows}
    for pair in near["nonexact"]:
        if (coordinates[("train", pair["train_line"])]["source_sha"] != pair["train_source_sha256"]
                or coordinates[("val", pair["val_line"])]["source_sha"] != pair["val_source_sha256"]):
            raise ValueError("Near-duplicate evidence no longer matches source coordinates")
        families.union([pair["train_source_sha256"], pair["val_source_sha256"]])
    golden_hashes = {row[0] for row in db.execute("SELECT digest FROM inv.golden_hashes")}
    db.close()
    reserved_families = {families.find(digest) for digest in golden_hashes}
    train_families = {families.find(row["source_sha"]) for row in rows if row["split"] == "train"}

    counts, first_reasons, overlapping_flags = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    accepted_sources, accepted_families, accepted_pairs = defaultdict(set), defaultdict(set), defaultdict(set)
    seen_pairs, seen_effective_pairs = {}, {}
    decisions, detail_examples = {}, defaultdict(list)
    columns = ("split", "line", "source_sha256", "target_sha256", "original_semantic_sha256", "source_family",
               "category", "first_reason", "all_quality_flags", "url_status", "duplicate_of",
               "effective_source_sha256", "effective_target_sha256", "target_tokens", "sequence_tokens")
    with row_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            key = row["split"], row["line"]
            split = row["split"]
            family = families.find(row["source_sha"])
            url = url_rows.get(key, {})
            status = url.get("status", "not_screened")
            effective_source, effective_target = row["source_sha"], row["target_sha256"]
            target_n, sequence_n, budget_pass = row["target_tokens"], row["sequence_tokens"], None
            if url and (url["source_sha256"] != row["source_sha"] or url["target_sha256"] != row["target_sha256"]):
                raise ValueError(f"URL probe binding mismatch: {key}")
            if status == REPAIR_STATUS:
                length = length_rows[key]
                for name in ("normalized_source_sha256", "normalized_target_sha256"):
                    if length[name] != url[name]:
                        raise ValueError(f"Post-repair token hash mismatch: {key}")
                effective_source, effective_target = url["normalized_source_sha256"], url["normalized_target_sha256"]
                target_n = int(length["normalized_target_tokens"])
                sequence_n = int(length["new_reconstructed_sequence_tokens"])
                budget_pass = target_n <= 2048 and sequence_n <= 4096
                if (length["length_status"] == "pass") != budget_pass:
                    raise ValueError(f"Post-repair budget flag mismatch: {key}")
            category, reason = preliminary_classification(
                row, status, reserved_family=family in reserved_families,
                validation_overlap=split == "val" and family in train_families,
                confirmed_defect=key in CONFIRMED_DEFECTS, repaired_budget_pass=budget_pass)
            pair = row["source_sha"], row["semantic_sha"]
            duplicate_of = ""
            if category != "REJECT":
                duplicate, duplicate_of = duplicate_reason(pair, (effective_source, effective_target),
                    f"{split}:{row['line']}", seen_pairs, seen_effective_pairs)
                if duplicate:
                    category, reason = "REJECT", duplicate
            flags = [flag for flag in WARNINGS if row[flag]]
            output = dict(zip(columns, (split, row["line"], row["source_sha"], row["target_sha256"],
                row["semantic_sha"], family, category, reason, "|".join(flags), status, duplicate_of,
                effective_source, effective_target, target_n, sequence_n)))
            writer.writerow(output)
            decisions[key] = output
            for scope in (split, "combined"):
                counts[scope][category] += 1
                first_reasons[scope][reason] += 1
                overlapping_flags[scope].update(flags)
                if category != "REJECT":
                    accepted_sources[scope].add(row["source_sha"])
                    accepted_families[scope].add(family)
                    accepted_pairs[scope].add(pair)
            if len(detail_examples[reason]) < 5:
                detail_examples[reason].append(output)
    if accepted_families["train"] & accepted_families["val"]:
        raise ValueError("Accepted train/validation source-family leakage")
    if (accepted_families["train"] | accepted_families["val"]) & reserved_families:
        raise ValueError("Accepted Golden family leakage")
    if sum(counts["combined"].values()) != len(rows):
        raise ValueError("Category totals do not reconcile")

    sibling_path = args.report / "siblings/candidate_rows.csv"
    sibling_summary = Counter()
    for key, sibling in read_rows(sibling_path).items():
        donors = [tuple(coordinate.split(":")) for coordinate in sibling["screened_sibling_coordinates"].split("|")]
        usable = [decisions[(split, int(line))] for split, line in donors
                  if decisions[(split, int(line))]["category"] != "REJECT"]
        own = [donor for donor in usable if donor["split"] == key[0]]
        sibling_summary["baseline_flagged_rows_with_screened_sibling"] += 1
        if own:
            sibling_summary["with_final_eligible_same_split_existing_sibling"] += 1
        elif usable:
            sibling_summary["with_final_eligible_other_split_sibling_only"] += 1
        else:
            sibling_summary["without_final_eligible_sibling"] += 1
    policy = {
        "version": POLICY_VERSION,
        "categories": {"KEEP": "Provisional content candidate; target unchanged, no tested warning; not a semantic certificate.",
                       "REPAIR": "Only exact-reversible URL representation normalization proved in memory; not a repaired dataset or recovery of missing content.",
                       "REJECT": "Exclude/quarantine from this selected run, including valid reserved, overlap, duplicate, long, or unresolved rows. Never delete originals."},
        "universal_packaging_not_counted_as_content_repair": ["UTF-16LE/BOM to UTF-8/LF", "Bind final user response and final assistant target, excluding few-shot pairs", "Add aliases and archive-derived hash IDs/group lineage without pretending original generator IDs were recovered", "Normalize shared production prompt and retain old scaffold fingerprint"],
        "warning_fields": list(WARNINGS),
        "limits": {"reconstructed_sequence_tokens": 4096, "target_tokens": 2048},
        "tokenizer_scope": "Local Gemma3 vocabulary with reconstructed current shared prompt and chat framing; not actual E2B or authoritative chat-template preflight.",
        "dedup": "Keep first admitted occurrence of an exact raw-source/original-semantic-target pair, and also remove identical effective source/target text pairs after URL conversion. Distinct layouts remain together within one family/split; do not duplicate cleaner siblings as repairs.",
        "families": "Transitive exact, whitespace-normalized, production-URL-masked hashes plus every nonexact pair in the bounded near-duplicate audit. Not exhaustive semantic-family discovery.",
        "split_policy": "Keep original training families on training side. Exclude validation families found anywhere in original training, even if corresponding training label is later rejected. No validation-to-training target transfer.",
        "symbolic_contract": "Already-symbolic rows may be SFT candidates only when all target tokens occur in source and no mixed literal references exist. Missing destination maps remain unknown; no URL-restoration or device-render certification.",
        "conservatism": ["Initial placeholder warning gate requires identical source/target token sets, stricter than target-subset closure; unused source references may be harmless.", "Original token budget is screened before URL conversion, so long rows that could shorten sufficiently are not probed.", "Unsupported role aliases are quarantined even if the underlying raw destinations are available. A separately tested, source-consistent reference normalization policy could recover more without Stage3; no such change is assumed here."],
        "no_semantic_guesses": ["No clipped-word joins claimed to restore dropped content", "No global mojibake replacement", "No guessed placeholder aliases/destinations", "No invented facts or new labels", "No target truncation to meet budget"],
    }
    policy_path = args.report / "policy.json"
    dump(policy_path, policy)
    evidence_paths = [manifest_path, near_path, url_summary_path, url_path, length_summary_path, length_path,
                      sibling_path, policy_path, Path(__file__), *(args.audits / name for name in ("inventory.sqlite", "ir.sqlite", "synthesis.sqlite"))]
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "policy_version": POLICY_VERSION,
        "original_data_modified": False, "training_dataset_created": False, "stage3_run": False,
        "sources": source_stats, "total_rows": len(rows),
        "categories": {scope: {name: counts[scope][name] for name in ("KEEP", "REPAIR", "REJECT")} for scope in ("train", "val", "combined")},
        "first_reason_counts_disjoint": {key: dict(value) for key, value in first_reasons.items()},
        "raw_quality_flags_overlap_not_additive": {key: dict(value) for key, value in overlapping_flags.items()},
        "eligible_unique_exact_sources": {key: len(value) for key, value in accepted_sources.items()},
        "eligible_source_families": {key: len(value) for key, value in accepted_families.items()},
        "eligible_unique_semantic_pairs": {key: len(value) for key, value in accepted_pairs.items()},
        "eligible_known_cross_split_or_golden_family_overlaps": 0,
        "sibling_followthrough": {**dict(sibling_summary), "new_unique_sources_added_by_copying_existing_siblings": 0},
        "schema_repairs_required": 0, "confirmed_clipped_or_missing_content_repairs_proved": 0,
        "examples": dict(detail_examples),
        "classified_rows_csv": str(row_path.resolve()), "classified_rows_sha256": file_sha(row_path),
        "evidence_sha256": {str(path.resolve()): file_sha(path) for path in evidence_paths},
        "interpretation": ["Exact policy-based row counts, not exact numbers of factually perfect or irreparable examples.",
                           "Final model-specific import and strict validation must run before these are usable training files.",
                           "Reject includes reviewable warnings and profile/split exclusions; no originals deleted.",
                           "Golden32 and Golden35 stay frozen and separate. Prior models trained on overlapping sources are not clean unseen baselines."],
    }
    dump(result_path, result)
    print(json.dumps({"categories": result["categories"], "first_reasons": result["first_reason_counts_disjoint"],
                      "unique_sources": result["eligible_unique_exact_sources"], "siblings": result["sibling_followthrough"],
                      "summary": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
