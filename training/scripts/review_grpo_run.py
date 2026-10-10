#!/usr/bin/env python3
"""Read an extracted GRPO review bundle; no model loading, repair, or training.

Usage: python training/scripts/review_grpo_run.py BUNDLE --output review.json
Output is evidence, not an independent-test claim: Golden32 selects checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def failure_kind(row: dict[str, Any]) -> str:
    """Diagnostic taxonomy only; the existing strict scorer remains authoritative."""
    if row["metrics"].get("schema_valid_strict") is True:
        return "valid"
    if (row.get("runtime") or {}).get("stop_reason") == "max_new_tokens":
        return "token_limit"  # May also contain looping/syntax faults; not proof cap alone fixes it.
    text = str(row.get("raw_generated_text", row.get("generated_text", "")))
    error = str(row["metrics"].get("schema_error", "")).lower()
    if "duplicate" in error:
        return "duplicate_id"
    opening = text.find("<a2ui>")
    if opening < 0:
        return "missing_opening_envelope"
    if text[:opening].strip():
        return "prefix_contamination"
    if "missing" in error and ("id" in error or "child" in error):
        return "missing_reference"
    return "syntax_or_schema"


def reward_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    valid, observed = 0, 0
    for row in rows:
        # Actual schema: NOT top-level strict_valid/reasons, nor breakdown.components.
        detail = (row.get("breakdown") or {}).get("qat_reward")
        if isinstance(detail, dict):
            observed += 1
            valid += detail.get("strict_valid") is True
            reasons.update(detail.get("reasons") or [])
    steps = sorted({row["step"] for row in rows if type(row.get("step")) is int})
    return {"rows": len(rows), "steps": steps, "breakdown_rows": observed,
            "strict_valid_rows": valid if observed else None,
            "strict_valid_fraction": valid / observed if observed else None,
            "mean_reward": statistics.fmean(row["reward"] for row in rows) if rows else None,
            "reason_counts": dict(reasons.most_common()),
            "scope": "audited candidates only; do not extrapolate past the last audited step"}


def paired_source_comparison(before: list[dict], after: list[dict], *, seed: int = 42,
                             samples: int = 20000) -> dict[str, Any]:
    if type(samples) is not int or samples < 1:
        raise ValueError("bootstrap samples must be positive")
    def grouped(rows):
        groups = defaultdict(list)
        for row in rows:
            key = row.get("source_id")
            if not key:
                raise ValueError("Paired comparison requires source_id")
            groups[key].append(row)
        return groups
    a, b = grouped(before), grouped(after)
    if a.keys() != b.keys():
        raise ValueError("Paired comparison requires identical source cohorts")
    differences, mismatches = [], []
    for source in sorted(a):
        def bindings(rows):
            return {(r.get("source_context_sha256"),
                     (r.get("runtime") or {}).get("input_token_ids_sha256"),
                     ((r["metrics"].get("metric_identity_v5_4") or {}).get("metric_fingerprint"))) for r in rows}
        aa, bb = bindings(a[source]), bindings(b[source])
        if aa != bb or any(not all(binding) for binding in aa | bb):
            mismatches.append(source)
        x = statistics.fmean(r["metrics"]["generation_reward_v5_4"] for r in a[source])
        y = statistics.fmean(r["metrics"]["generation_reward_v5_4"] for r in b[source])
        differences.append(y-x)
    if not differences:
        raise ValueError("Empty comparison")
    rng = random.Random(seed)
    n = len(differences)
    draws = sorted(statistics.fmean(rng.choices(differences, k=n)) for _ in range(samples))
    return {"unique_sources": n, "mean_gain_points": statistics.fmean(differences),
            "improved_sources": sum(v > 1e-9 for v in differences),
            "regressed_sources": sum(v < -1e-9 for v in differences),
            "unchanged_sources": sum(abs(v) <= 1e-9 for v in differences),
            "source_prompt_metric_binding_mismatches": mismatches,
            "bootstrap_95_percent_interval": [draws[int(.025*samples)], draws[min(samples-1, int(.975*samples))]] if not mismatches else None,
            "bootstrap_seed": seed, "bootstrap_samples": samples,
            "scope": "paired source-level descriptive interval; selection-biased development cohort, not a confirmatory test"}



def verify_manifest(bundle: Path) -> dict[str, Any]:
    manifest = bundle / "SHA256SUMS.txt"
    if not manifest.is_file():
        return {"verified": False, "reason": "manifest_not_provided"}
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = (bundle / name.lstrip("*")).resolve(strict=True)
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError("Manifest entry escapes the review bundle")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Review bundle hash mismatch: {name}")
        checked += 1
    if not checked:
        raise ValueError("Empty review bundle manifest")
    return {"verified": True, "files_checked": checked}


def summarize(bundle: Path) -> dict[str, Any]:
    manifest = verify_manifest(bundle)
    data = json.loads((bundle / "run/data_audit.json").read_text())
    retention = {name: {**counts, "retained_fraction": counts["accepted_rows"] / counts["input_rows"]}
                 for name, counts in data["prepared_counts"].items() if name in {"train", "val"}}
    evals = {}
    for path in sorted((bundle / "evaluations").rglob("scored_predictions.jsonl")):
        rows = read_jsonl(path)
        aggregate = json.loads((path.parent / "aggregate_metrics.json").read_text())
        evals[str(path.parent.relative_to(bundle))] = {
            "rows": len(rows), "unique_sources": len({row["source_id"] for row in rows}),
            "strict_valid": sum(r["metrics"].get("schema_valid_strict") is True for r in rows),
            "reward_points": aggregate["generation_reward_v5_4_avg"],
            "unique_source_reward_points": aggregate.get("unique_source_generation_reward_v5_4_avg"),
            "cap": aggregate["cap_v5_4_avg"],
            "fully_root_reachable": aggregate["fully_root_reachable_v5_4_avg"],
            "failure_counts": dict(Counter(failure_kind(r) for r in rows)),
            "invalid_rows": [{"source_id": r["source_id"], "kind": failure_kind(r),
                              "stop_reason": (r.get("runtime") or {}).get("stop_reason"),
                              "output_tokens": (r.get("runtime") or {}).get("output_tokens"),
                              "schema_error": r["metrics"].get("schema_error")}
                             for r in rows if failure_kind(r) != "valid"],
        }
    audits = [row for path in sorted((bundle / "run/training").glob("grpo_rewards.rank*.jsonl"))
              for row in read_jsonl(path)]
    health = {}
    for path in sorted((bundle / "run/training").glob("grpo_health.rank*.jsonl")):
        rows = read_jsonl(path)
        health[path.name] = {"records": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
                            "issue_records": sum(bool(r.get("issues")) for r in rows),
                            "last_step": rows[-1]["step"] if rows else None}
    before = read_jsonl(bundle / "evaluations/sft_starting_checkpoint/scored_predictions.jsonl")
    after = read_jsonl(bundle / "evaluations/grpo/best_golden32/scored_predictions.jsonl")
    return {"schema_version": 1, "bundle_manifest": manifest, "retention": retention, "evaluations": evals,
            "startup_reward_audit": reward_audit(audits), "health": health,
            "selected_vs_saved_sft": paired_source_comparison(before, after),
            "limitations": ["No new training or inference executed.",
                "Saved SFT comparison is not a fresh in-run step-zero baseline.",
                "Clipped completion ratio measures token-budget truncation, not PPO ratio clipping.",
                "Golden35/Bixby50 improvements need matched SFT baselines; export is not native runtime QA."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
