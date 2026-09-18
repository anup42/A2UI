"""Summarize completed real-device prompt runs, retaining failed-case latency."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from prompt_study_evidence import validate_attempt_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries, cases = [], []
    for run in sorted(args.root.iterdir()):
        if not (run / "experiment.json").is_file():
            continue
        meta = json.loads((run / "experiment.json").read_text(encoding="utf-8-sig"))
        if not (run / "summary.json").is_file() and not meta.get("earlyRejected"):
            continue
        config = json.loads((run / "run_config.json").read_text(encoding="utf-8-sig"))
        result = json.loads((run / "results.json").read_text(encoding="utf-8-sig"))
        expected = meta["cases"]
        observed = [r["id"] for r in result]
        assert len(observed) == len(set(observed)) and set(expected) == set(config["cases"]), run
        assert set(observed) <= set(expected) and (set(observed) == set(expected) or meta.get("earlyRejected")), run
        assert hashlib.sha256((run / "experiment_prompt.txt").read_bytes()).hexdigest() == meta["promptSha256"], run
        assert config["accelerator"] == "GPU" and config["mtp"] and config["thinkingEnabled"], run
        assert config["maxRepairAttempts"] == meta["repairs"] and config["temperature"] == 0, run
        if "promptSha256" in config:
            assert config["promptSha256"] == meta["promptSha256"] and config["sourceBindings"], run
        if "corpusSha256" in config:
            assert config["corpusSha256"] == meta["corpusSha256"], run
        successes = [r for r in result if r["status"] == "success"]
        all_times = [r["elapsedMs"] for r in result]
        warm_times = [r["elapsedMs"] for r in result if not r["firstInRun"]]
        evidence = validate_attempt_evidence(run, meta, config, result)
        thermal = []
        for key in ("thermalBefore", "thermalAfter"):
            text = meta[key]["stdout"]
            status = re.search(r"Thermal Status: (\d+)", text)
            skin = re.findall(r"mValue=([0-9.]+).*?mName=SKIN", text)
            thermal.append({"status": int(status[1]) if status else None, "skinReadingsC": [float(v) for v in skin]})
        summaries.append({
            "run": run.name, "prompt": Path(meta["promptHostPath"]).stem,
            "prompt_sha256": meta["promptSha256"], "total": len(result), "planned": len(expected),
            "input_scaffold": bool(meta.get("inputScaffold")),
            "sdk_layout_scaffold": bool(meta.get("sdkLayoutScaffold")),
            "packaged_prompt": bool(meta.get("packagedPrompt")),
            "recorded_inputs": bool(meta.get("recordInputs")),
            "early_rejected": bool(meta.get("earlyRejected")), "unfinished_ids": sorted(set(expected) - set(observed)),
            "first_attempt_success": sum(r["attempts"] == 1 for r in successes),
            "repaired_success": sum(r["attempts"] > 1 for r in successes),
            "success": len(successes), "render_failure": sum(r["status"] == "render_failure" for r in result),
            "failure": sum(r["status"] == "failure" for r in result),
            "fallbacks": sum(bool(r.get("usedFallback")) for r in result),
            "repairs_allowed": meta["repairs"], **evidence,
            "median_all_cases_ms": statistics.median(all_times),
            "median_warm_all_cases_ms": statistics.median(warm_times) if warm_times else None,
            "p95_all_cases_ms": sorted(all_times)[math.ceil(len(all_times) * .95) - 1],
            "sum_case_ms": sum(all_times), "wall_seconds": meta["wallSeconds"],
            "max_process_pss_kb": max(r["peakPssKb"] for r in result),
            "thermal_before": thermal[0], "thermal_after": thermal[1],
            "failed_cases": [{"id": r["id"], "message": r.get("message", "render smoke failed")} for r in result if r["status"] != "success"],
        })
        for r in result:
            cases.append({"run": run.name, "prompt": Path(meta["promptHostPath"]).stem,
                "id": r["id"], "status": r["status"], "attempts": r["attempts"],
                "elapsed_ms": r["elapsedMs"], "first_in_run": r["firstInRun"],
                "peak_pss_kb": r["peakPssKb"], "message": r.get("message", "")})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if cases:
        with (args.output / "case_results.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cases[0].keys())
            writer.writeheader()
            writer.writerows(cases)
    for s in summaries:
        print(f"{s['run']}: {s['first_attempt_success']}/{s['total']} first; {s['success']}/{s['total']} final; all-case median {s['median_all_cases_ms']/1000:.3f}s; calls={s['captured_model_calls']}")


if __name__ == "__main__":
    main()
