#!/usr/bin/env python3
"""Summarize a controlled GenUICraft FP32 versus corrected FP16 device run.

The report uses only measured native token counts/rates. A render smoke check is
reported separately from source fidelity; neither is called a 100% quality score.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def median(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else None


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def weighted_native_decode(results: list[dict[str, Any]]) -> dict[str, Any]:
    samples = []
    for result in results:
        tokens = number(result.get("outputTokens"))
        rate = number(result.get("decodeTokensPerSecond"))
        if tokens is not None and tokens > 0 and rate is not None and rate > 0:
            samples.append((tokens, rate))
    tokens = sum(row[0] for row in samples)
    seconds = sum(row[0] / row[1] for row in samples)
    return {
        "tokensPerSecond": tokens / seconds if seconds > 0 else None,
        "measuredSamples": len(samples),
        "eligibleSamples": len(results),
        "nativeDecodeTokens": int(tokens),
        "derivedDecodeSeconds": seconds if seconds > 0 else None,
        "method": "sum(native output tokens) / sum(native output tokens / native decode tokens per second)",
    }


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "measuredCases": len(rows),
        "rawStrictValid": sum(row.get("rawStrictValid") is True for row in rows),
        "repairedStrictValid": sum(row.get("repairedStrictValid") is True for row in rows),
        "sdkRenderSmokeValid": sum(row.get("sdkRenderSmokeValid") is True for row in rows),
        "sourceFidelityWarningCases": sum((row.get("sourceFidelityWarnings") or 0) > 0 for row in rows),
        "fallbackCases": sum(row.get("usedFallback") is True for row in rows),
        "finishReasons": dict(sorted(_counts(row.get("finishReason") for row in rows).items())),
        "medianProviderWallMs": median([value for row in rows if (value := number(row.get("providerElapsedMs"))) is not None]),
        "medianCaseWallMs": median([value for row in rows if (value := number(row.get("caseElapsedMs"))) is not None]),
        "medianTtftSeconds": median([value for row in rows if (value := number(row.get("timeToFirstTokenSeconds"))) is not None]),
        "medianPrefillTokensPerSecond": median([value for row in rows if (value := number(row.get("prefillTokensPerSecond"))) is not None]),
        "weightedNativeDecode": weighted_native_decode(rows),
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        label = str(value) if value is not None else "unavailable"
        counts[label] = counts.get(label, 0) + 1
    return counts


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(root: Path) -> tuple[dict[str, Any], str]:
    protocol = read_json(root / "protocol.json")
    observations: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for planned in protocol["plan"]:
        label = planned["label"]
        directory = root / "runs" / label
        if not directory.is_dir():
            runs.append({**planned, "state": "not_run"})
            continue
        config_file = directory / "run_config.json"
        summary_file = directory / "summary.json"
        config = read_json(config_file) if config_file.is_file() else {}
        run_summary = read_json(summary_file) if summary_file.is_file() else {}
        result_file = directory / "results.json"
        rows = read_json(result_file) if result_file.is_file() else []
        if not isinstance(rows, list):
            rows = []
        mismatches = []
        for field in ("label", "precision"):
            if config.get(field) != planned.get(field):
                mismatches.append(field)
        if config.get("mtpRequested") != planned.get("mtp"):
            mismatches.append("mtp")
        if config.get("cases") != protocol["cases"]:
            mismatches.append("cases")
        if config.get("corpusSha256") != protocol["corpusSha256"]:
            mismatches.append("corpusSha256")
        if config.get("promptContractSha256") != protocol["promptContractSha256"]:
            mismatches.append("promptContractSha256")
        if config.get("modelPathArgument") != planned["modelPath"]:
            mismatches.append("modelPathArgument")
        if protocol.get("warmupCase") and config.get("warmupCase") != protocol["warmupCase"]:
            mismatches.append("warmupCase")
        if protocol.get("cooldownTimeoutMs", 0):
            gate_file = directory / "thermal_ready.json"
            gate = read_json(gate_file) if gate_file.is_file() else {}
            if (config.get("cooldownTimeoutMs") != protocol["cooldownTimeoutMs"]
                    or gate.get("ready") is not True):
                mismatches.append("thermalReadyGate")
        warmup_file = directory / "warmup_result.json"
        warmup = read_json(warmup_file) if warmup_file.is_file() else {}
        if warmup.get("rawComplete") is not True or not warmup.get("runtime"):
            mismatches.append("nativeWarmup")
        expected_runtime = "+FP32" if planned["precision"] == "FP32" else "+FP16_CORRECTED"
        for row in rows:
            runtime = row.get("runtime") if isinstance(row, dict) else None
            if runtime is not None and ("/GPU" not in runtime or expected_runtime not in runtime
                                        or ("+MTP" in runtime) != planned["mtp"]):
                mismatches.append(f"runtimeIdentity:{row.get('id', 'unknown')}")
            if isinstance(row, dict) and row.get("engineInitializedForRequest") is True:
                mismatches.append(f"coldMeasuredCase:{row.get('id', 'unknown')}")
        for row in rows:
            if isinstance(row, dict):
                observations.append({**row, "label": label,
                                     "precision": planned["precision"],
                                     "mtp": planned["mtp"],
                                     "repetition": planned["repetition"]})
        session_file = directory / "session_metrics.json"
        session_metrics = read_json(session_file) if session_file.is_file() else None
        if not isinstance(session_metrics, dict) or session_metrics.get("speculativeDecodingEnabled") != planned["mtp"]:
            mismatches.append("actualMtpState")
        run_complete = (run_summary.get("runComplete") is True
                        and run_summary.get("completed") == len(protocol["cases"])
                        and len(rows) == len(protocol["cases"])
                        and run_summary.get("closeError") is None)
        runs.append({**planned, "state": "complete" if run_complete else "partial",
                     "measuredCases": len(rows), "configMismatches": mismatches,
                     "sessionMetrics": session_metrics})

    by_arm: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_arm[row["precision"], row["mtp"]].append(row)
    arms = {
        f"{precision}_mtp_{'on' if mtp else 'off'}": arm_summary(rows)
        for (precision, mtp), rows in sorted(by_arm.items())
    }
    matched_cases = []
    for mtp in protocol["mtpModes"]:
        for case in protocol["cases"]:
            fp32 = [row for row in by_arm["FP32", mtp] if row.get("id") == case]
            fp16 = [row for row in by_arm["FP16_CORRECTED", mtp] if row.get("id") == case]
            def med(rows: list[dict[str, Any]], field: str) -> float | None:
                return median([value for row in rows if (value := number(row.get(field))) is not None])
            fp32_wall = med(fp32, "providerElapsedMs")
            fp16_wall = med(fp16, "providerElapsedMs")
            fp32_decode = med(fp32, "decodeTokensPerSecond")
            fp16_decode = med(fp16, "decodeTokensPerSecond")
            matched_cases.append({
                "case": case, "mtp": mtp, "fp32Samples": len(fp32), "fp16Samples": len(fp16),
                "fp32MedianProviderMs": fp32_wall, "fp16MedianProviderMs": fp16_wall,
                "fp16WallSpeedup": fp32_wall / fp16_wall if fp32_wall and fp16_wall else None,
                "fp32MedianNativeDecodeTokensPerSecond": fp32_decode,
                "fp16MedianNativeDecodeTokensPerSecond": fp16_decode,
                "fp16DecodeSpeedup": fp16_decode / fp32_decode if fp32_decode and fp16_decode else None,
                "fp32RawStrictValid": sum(row.get("rawStrictValid") is True for row in fp32),
                "fp16RawStrictValid": sum(row.get("rawStrictValid") is True for row in fp16),
                "fp32RepairedStrictValid": sum(row.get("repairedStrictValid") is True for row in fp32),
                "fp16RepairedStrictValid": sum(row.get("repairedStrictValid") is True for row in fp16),
                "fp32SdkRenderSmokeValid": sum(row.get("sdkRenderSmokeValid") is True for row in fp32),
                "fp16SdkRenderSmokeValid": sum(row.get("sdkRenderSmokeValid") is True for row in fp16),
            })
    comparable = len(runs) == len(protocol["plan"]) and all(
        run["state"] == "complete" and not run.get("configMismatches") for run in runs
    )
    summary = {
        "schemaVersion": 1,
        "completeAndMatched": comparable,
        "protocol": protocol,
        "runs": runs,
        "arms": arms,
        "sameCaseMedians": matched_cases,
        "observations": observations,
        "interpretation": {
            "quality": "Raw strict, repaired strict, and SDK render smoke counts are distinct. Source fidelity warnings remain separate; no aggregate score proves faithful conversion.",
            "timing": "Provider wall time includes native generation and prompt prefill. Native decode tokens/s excludes engine initialization and prefill. Warmup cases are excluded.",
            "scope": f"{len(protocol['cases'])} selected Bixby50 cases on one device are a diagnostic comparison, not a 50-case or holdout quality estimate.",
        },
    }
    lines = [
        "# Gemma 4 corrected FP16 versus FP32 app comparison", "",
        f"Complete matched protocol: **{'yes' if comparable else 'no'}**. "
        f"Measured case results: **{len(observations)}**. Warmups excluded.", "",
        "The arms use the same frozen Bixby50 source cases and trained SDK prompt. "
        "FP16_CORRECTED labels a separately prepared model package; its model hash is recorded in `protocol.json`. "
        "This experiment measures GPU execution with the app's SDK conversion and render path.", "",
        "| Arm | Cases | Raw strict | Repaired strict | SDK render smoke | Native decode tok/s (weighted) | Median provider ms | Median TTFT s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in sorted(arms.items()):
        lines.append(f"| {arm} | {values['measuredCases']} | {values['rawStrictValid']} | "
                     f"{values['repairedStrictValid']} | {values['sdkRenderSmokeValid']} | "
                     f"{fmt(values['weightedNativeDecode']['tokensPerSecond'])} "
                     f"({values['weightedNativeDecode']['measuredSamples']}/{values['weightedNativeDecode']['eligibleSamples']}) | "
                     f"{fmt(values['medianProviderWallMs'])} | {fmt(values['medianTtftSeconds'])} |")
    lines += ["", "Weighted native decode throughput is `sum(native output tokens) / "
              "sum(native output tokens / native decode tokens per second)` over cases with both native measurements. "
              "Missing measurements are excluded and counted in the table. A median of per-case native rates is "
              "shown below for each matched case.", "",
              "| Case | MTP | FP32 / FP16 samples | FP16 wall speedup | FP16 decode speedup | Raw strict FP32 / FP16 | Repaired strict FP32 / FP16 | SDK render FP32 / FP16 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in matched_cases:
        lines.append(f"| {row['case']} | {'on' if row['mtp'] else 'off'} | "
                     f"{row['fp32Samples']} / {row['fp16Samples']} | {fmt(row['fp16WallSpeedup'])}× | "
                     f"{fmt(row['fp16DecodeSpeedup'])}× | "
                     f"{row['fp32RawStrictValid']} / {row['fp16RawStrictValid']} | "
                     f"{row['fp32RepairedStrictValid']} / {row['fp16RepairedStrictValid']} | "
                     f"{row['fp32SdkRenderSmokeValid']} / {row['fp16SdkRenderSmokeValid']} |")
    lines += ["", "## Evidence and limits", "",
              "Each run directory preserves exact raw output, frozen source, generation prompt, repaired Express/JSON, "
              "per-case native token/timing metrics, finish reason, runtime identity, SDK render screenshot, and status. "
              "`telemetry.jsonl` records host-sampled thermal, battery, and GPU state; check it before attributing "
              "small timing differences to precision. First-case warmups are recorded separately and excluded.", "",
              "Raw strict validity means the exact native output compiled unchanged. Repaired strict validity means "
              "the SDK returned a document that compiles. SDK render smoke means the app displayed that document "
              "without a visible error and captured a screenshot. These mechanical checks do not prove content fidelity. "
              "The per-case source fidelity warning count in `summary.json` provides another diagnostic, not a score.", "",
              "Selected cases: " + ", ".join(protocol["cases"]) + ". These development-informed samples "
              "do not establish a 50-case success rate or production readiness.", ""]
    incomplete = [run for run in runs if run["state"] != "complete" or run.get("configMismatches")]
    if incomplete:
        lines += ["Incomplete or mismatched runs: " + ", ".join(
            f"{run['label']} ({run['state']}{'; config=' + ','.join(run['configMismatches']) if run.get('configMismatches') else ''})"
            for run in incomplete), ""]
    return summary, "\n".join(lines)


def write_report(root: Path) -> dict[str, Any]:
    summary, markdown = build_report(root)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "REPORT.md").write_text(markdown, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = write_report(args.run_dir.resolve())
    print(f"Reported {len(summary['observations'])} measured cases; complete={summary['completeAndMatched']}")


if __name__ == "__main__":
    main()
