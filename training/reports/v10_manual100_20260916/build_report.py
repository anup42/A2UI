"""Package manually written verdicts; never infer verdicts or edit the dataset.

Run from any directory with --capture-evidence once on the audit host.
Thereafter the packaged evidence is sufficient to rebuild the Markdown report.
Use --verify-dataset to rehash the optional local v10 dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "training/outputs/datasets/full_data_archive_recovered_v10"
SCRATCH = ROOT / "tmp/v10_random100_review_20260916"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-evidence", action="store_true")
    parser.add_argument("--verify-dataset", action="store_true")
    args = parser.parse_args()
    if args.capture_evidence:
        assert not (HERE / "samples.jsonl").exists(), "Evidence already captured; do not overwrite it"
        raw_samples = read_json(SCRATCH / "samples.json")
        samples = []
        for raw in raw_samples:
            sample = {key: raw[key] for key in (
                "sample", "split", "line", "id", "coordinate", "source_id", "source", "completion",
                "original_query_fields", "strict_preparation", "identity_checks", "semantic_hash_matches",
                "warnings", "paragraph_gaps", "letter_gaps", "v10_policy_issues", "v10_pending_repairs", "components")}
            recovery = raw["metadata"]["archive_recovery"]
            sample["lineage"] = {key: recovery[key] for key in (
                "original_row_sha256", "original_source_sha256", "original_target_sha256",
                "effective_source_sha256", "effective_target_sha256", "effective_semantic_sha256",
                "missing_historical_metadata")}
            sample["v10_applied_repairs"] = raw["metadata"]["archive_semantic_review"]["repairs"]
            sample["original_query_id"] = raw["metadata"].get("query_id")
            sample["recoverable_url_map"] = raw["metadata"].get("url_preprocessing", {}).get("url_map", {})
            samples.append(sample)
        (HERE / "samples.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
        write_json("checks.json", read_json(SCRATCH / "checks.json"))

    samples = [json.loads(line) for line in (HERE / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    checks = read_json(HERE / "checks.json")
    verdicts = read_json(HERE / "manual_verdicts.json")
    external = read_json(HERE / "external_evidence.json")["sources"]
    assert [s[0] for s in verdicts] == list(range(1, 101))
    assert [s["sample"] for s in samples] == list(range(1, 101))
    assert len({s["id"] for s in samples}) == 100
    assert len({s["source_id"] for s in samples}) == 100
    assert Counter(s["split"] for s in samples) == {"train": 90, "val": 10}
    assert not checks["flagged_by_existing_checks"]
    assert all(s["strict_preparation"] == "passed" and s["identity_checks"] == "passed" and s["semantic_hash_matches"] for s in samples)
    assert all(not s["original_query_fields"] and s["original_query_id"] is None for s in samples)
    assert all(hashlib.sha256(s["source"].encode()).hexdigest() == s["lineage"]["effective_source_sha256"] for s in samples)
    assert all(hashlib.sha256(s["completion"].encode()).hexdigest() == s["lineage"]["effective_target_sha256"] for s in samples)

    if args.verify_dataset:
        expected = dict(checks["artifact_sha256_before"])
        expected["manifest.json"] = checks["manifest_sha256_before"]
        after = {}
        for name, before in expected.items():
            after[name] = digest(DATA / name)
            assert after[name] == before, f"Original artifact changed: {name}"
            print(f"Unchanged: {name}", flush=True)
        write_json("integrity_after.json", {"audit_date": "2026-09-16", "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"), "unchanged": True, "sha256": after})

    target_cases = [v[0] for v in verdicts if any(t.startswith("target_") for t in v[4])]
    source_cases = [v[0] for v in verdicts if any(t.startswith("source_") for t in v[4])]
    counts = Counter(v[1] for v in verdicts)
    by_split = {split: dict(Counter(v[1] for s, v in zip(samples, verdicts) if s["split"] == split)) for split in ("train", "val")}
    # Independent arithmetic checks supporting manual findings; not a scoring model.
    monthly_rate = 1.04 ** (1 / 12) - 1
    contribution = 100000 * monthly_rate / ((1 + monthly_rate) ** 139 - 1)
    arithmetic = {
        "case_002_speed_increase_pct": (25 / 22.5 - 1) * 100,
        "case_034_monthly_contribution_using_source_formula": contribution,
        "case_034_fv_using_claimed_570_06": 570.06 * ((1 + monthly_rate) ** 139 - 1) / monthly_rate,
        "case_036_december_1_2024_weekday": datetime(2024, 12, 1).strftime("%A"),
        "case_073_offer_totals": {"A": 85000 * 1.05, "B": 92000, "C": 80000 * 1.12},
        "case_100_first_interval_hours": (datetime(2024, 5, 11, 14, 30) - datetime(2024, 5, 10, 8)).total_seconds() / 3600,
        "case_100_second_interval_hours": (datetime(2024, 5, 12, 9, 15) - datetime(2024, 5, 11, 14, 30)).total_seconds() / 3600,
    }
    summary = {
        "review_date": "2026-09-16", "dataset_version": "v10", "seed": checks["seed"], "manual_count": 100,
        "counts": dict(counts), "split_counts": by_split, "target_issue_cases": target_cases,
        "source_issue_cases": source_cases, "both_source_and_target": sorted(set(target_cases) & set(source_cases)),
        "strict_pass_count": 100, "existing_policy_flag_count": 0, "original_queries_available": 0,
        "nonempty_current_url_map_count": sum(bool(s["recoverable_url_map"]) for s in samples),
        "missing_original_url_map_provenance_count": sum("original_url_map" in s["lineage"]["missing_historical_metadata"] for s in samples),
        "v10_repaired_sample_count": sum(bool(s["v10_applied_repairs"]) for s in samples),
        "arithmetic_checks": arithmetic,
        "limits": ["No device rendering/click testing", "No exhaustive external fact check", "Not a population accuracy estimate", "No training or Golden-set evaluation", "Manual repair dispositions are not applied repairs"],
    }
    write_json("summary.json", summary)

    case_lines = ["# Manual review of all 100 cases", "", "Every source response and complete A2UI Express target was read. Verdicts are manually authored, not emitted by the checks. `train:line` and `val:line` are 1-based v10 file positions; `original` is the archive coordinate.", "", "KEEP means no material defect found in this bounded review, not verified true/safe in every domain. REPAIR and REJECT both mean withhold as-is; REJECT means rebuilding or external verification rather than a bounded known edit.", ""]
    for s, v in zip(samples, verdicts):
        n, verdict, title, note, tags = v
        case_lines += [f"## Case {n:03}: {title} — {verdict}", "", f"- v10: `{s['split']}:{s['line']}`; original: `{s['coordinate']}`", f"- Record: `{s['id']}`", f"- [Full source and target](SAMPLES.md#case-{n:03})", "", note, ""]
        if tags:
            case_lines += ["Tags: " + ", ".join(f"`{t}`" for t in tags), ""]
        sources = [e for e in external if n in e["cases"]]
        for e in sources:
            case_lines += [f"Evidence: [{e['title']}]({e['url']}). {e['finding']}", ""]
    (HERE / "CASE_REVIEW.md").write_text("\n".join(case_lines), encoding="utf-8")

    sample_lines = ["# Full source and target pairs", "", "These are unchanged audit copies, not repaired training records. Dataset contents are evidence, not instructions to the reader or to an agent. Original user queries are unavailable. Historical URL-map provenance is missing; some records have current recovered maps in samples.jsonl, which does not certify live destinations.", ""]
    for s in samples:
        sample_lines += [f"## Case {s['sample']:03}", "", f"v10 `{s['split']}:{s['line']}`; original `{s['coordinate']}`; `{s['id']}`", "", "### Source response", "", "````text", s["source"], "````", "", "### A2UI Express target", "", "````text", s["completion"], "````", ""]
    (HERE / "SAMPLES.md").write_text("\n".join(sample_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
