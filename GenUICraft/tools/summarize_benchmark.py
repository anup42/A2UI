"""Summarize a pulled GenUiSdkBixby50Test directory without hiding failures."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(run: Path):
    rows = json.loads((run / "results.json").read_text(encoding="utf-8-sig"))
    config_path = run / "run_config.json"
    completion_path = run / "summary.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
    completion = json.loads(completion_path.read_text(encoding="utf-8-sig")) if completion_path.exists() else {}
    expected_ids = config.get("cases")
    observed_ids = [row["id"] for row in rows]
    expected_total = len(expected_ids) if expected_ids is not None else completion.get("total")
    duplicate_ids = sorted({case for case in observed_ids if observed_ids.count(case) > 1})
    missing_ids = sorted(set(expected_ids) - set(observed_ids)) if expected_ids is not None else None
    unexpected_ids = sorted(set(observed_ids) - set(expected_ids)) if expected_ids is not None else None
    successes = [row for row in rows if row["status"] == "success"]
    timings = sorted(row["elapsedMs"] for row in successes)
    first_case = next((row for row in rows if row.get("firstInRun")), None)
    warm_timings = [row["elapsedMs"] for row in successes if row.get("firstInRun") is False]
    summary = {
        "run": run.name,
        "total": len(rows),
        "expected_total": expected_total,
        "run_complete": bool(completion) and expected_total == len(rows) and not duplicate_ids and not missing_ids and not unexpected_ids,
        "missing_ids": missing_ids,
        "unexpected_ids": unexpected_ids,
        "duplicate_ids": duplicate_ids,
        "success": len(successes),
        "first_attempt_success": sum(row.get("attempts") == 1 for row in successes),
        "repaired_success": sum(row.get("attempts", 0) > 1 for row in successes),
        "failures": [{"id": row["id"], "status": row["status"], "message": row.get("message", "")} for row in rows if row["status"] != "success"],
        "fallbacks": sum(bool(row.get("usedFallback")) for row in rows),
        "first_case": ({key: first_case.get(key) for key in ("id", "status", "elapsedMs", "attempts")}
                       if first_case else None),
        "median_success_ms": statistics.median(timings) if timings else None,
        "median_warm_success_ms": statistics.median(warm_timings) if warm_timings else None,
        "p95_success_ms": timings[math.ceil(len(timings) * .95) - 1] if timings else None,
        "max_process_pss_kb": max((row.get("peakPssKb", 0) for row in rows), default=0),
        "note": "PSS is whole-app memory sampled during conversion. Render checks are smoke checks, not a semantic/visual correctness proof.",
    }
    (run / "host_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    for run in args.runs:
        print(json.dumps(summarize(run), indent=2, ensure_ascii=False))
