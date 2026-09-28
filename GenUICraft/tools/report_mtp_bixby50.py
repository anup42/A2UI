#!/usr/bin/env python3
"""Audit and compare the paired, five-case Bixby50 GPU FP32 MTP runs.

The default is deliberately strict: all 20 complete batches, with each frozen
case once per mode, are required before any report is emitted. --partial is
only for a clearly marked progress report and never declares a winner.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
import statistics
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote

import report_trained_bixby50 as trained


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def obj(path: Path) -> dict:
    value = trained.read_json(path)
    need(isinstance(value, dict), f"missing or invalid object: {path}")
    return value


def rows(path: Path) -> list[dict]:
    value = trained.read_json(path)
    need(isinstance(value, list) and all(isinstance(x, dict) for x in value),
         f"missing or invalid array: {path}")
    return value


def number(value: object) -> float | None:
    return trained.as_number(value)


def med(values: list[float | None]) -> float | None:
    present = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.median(present) if present else None


def avg(values: list[float | None]) -> float | None:
    present = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.mean(present) if present else None


def fmt(value: object, digits: int = 2) -> str:
    x = number(value)
    return "n/a" if x is None else f"{x:,.{digits}f}"


def expected_plan(ids: list[str]) -> list[dict]:
    plan = []
    for group in range(10):
        cases = ids[group * 5:(group + 1) * 5]
        modes = (False, True) if group % 2 == 0 else (True, False)
        for mode in modes:
            plan.append({"batch": f"group_{group + 1:02}_mtp_{'on' if mode else 'off'}",
                         "mtp": mode, "cases": cases})
    return plan


def audit_batch(root: Path, entry: dict, protocol: dict, frozen: dict[str, dict],
                baseline: dict | None, partial: bool) -> tuple[dict, dict, dict, dict | None]:
    name = entry["batch"]
    path = root / "batches" / name
    config = obj(path / "run_config.json")
    result_rows = rows(path / "results.json")
    interruption = obj(path / "host_interruption.json") if (path / "host_interruption.json").is_file() else None
    need(partial or interruption is None, f"{name}: interrupted batch cannot enter a full report")
    need(config.get("cases") == entry["cases"], f"{name}: selected cases differ from protocol")
    if interruption is None:
        summary = obj(path / "summary.json")
        need(summary.get("runComplete") is True and summary.get("completed") == 5
             and summary.get("total") == 5 and len(result_rows) == 5,
             f"{name}: batch is incomplete")
        need(summary.get("runId") == config.get("runId"), f"{name}: run IDs differ")
    else:
        need(interruption.get("reason") == "user_requested_stop"
             and interruption.get("inferenceResumed") is False
             and interruption.get("selectedCases") == entry["cases"]
             and interruption.get("runId") == config.get("runId"),
             f"{name}: interruption marker differs from protocol")
        status = obj(path / "status.json")
        need(status.get("runId") == config.get("runId")
             and status.get("completed") == len(result_rows)
             and interruption.get("completedCases") == [r.get("id") for r in result_rows]
             and 0 < len(result_rows) < 5,
             f"{name}: interrupted completion count differs")
    ids = [row.get("id") for row in result_rows]
    need(ids == entry["cases"][:len(ids)], f"{name}: result IDs/order differ from protocol")
    runtime = config.get("runtime")
    need(isinstance(runtime, dict), f"{name}: runtime configuration missing")
    expected_settings = {
        "accelerator": "GPU", "gpuPrecision": protocol["precision"],
        "maxContextTokens": protocol["contextTokens"],
        "maxOutputTokens": protocol["maxOutputTokens"],
        "temperature": protocol["temperature"],
        "thinkingEnabled": protocol["thinking"],
        "sourceFallbackEnabled": protocol["sourceFallback"],
        "generatedDslRepairEnabled": protocol["generatedDslRecovery"],
        "mtpEnabled": entry["mtp"], "metricsEnabled": True,
    }
    for key, expected in expected_settings.items():
        need(runtime.get(key) == expected, f"{name}: runtime.{key}={runtime.get(key)!r}, expected {expected!r}")
    need(config.get("model", {}).get("basename") == Path(protocol["modelPath"]).name,
         f"{name}: model basename differs from protocol")
    need(config.get("device", {}).get("model") == obj(root / "provenance.json")["deviceModel"],
         f"{name}: device differs from provenance")
    need(config.get("corpus", {}).get("selectedRows") == 5,
         f"{name}: corpus selection size is not five")
    need(config.get("corpus", {}).get("sha256") == trained.sha256_file(trained.ANDROID_CORPUS),
         f"{name}: Android corpus hash differs from repository snapshot")
    if interruption is None:
        host = obj(path / "host_run.json")
        need(host.get("entry") == entry, f"{name}: host invocation plan differs")
        command = host.get("command")
        need(isinstance(command, list) and all(isinstance(x, str) for x in command),
             f"{name}: host invocation missing")
        need("-s" in command and command[command.index("-s") + 1] == protocol["serial"],
             f"{name}: host serial differs")
        extras = {}
        for index, part in enumerate(command[:-2]):
            if part == "-e":
                extras[command[index + 1]] = command[index + 2]
        for key, expected in {
            "runId": config.get("runId"), "cases": ",".join(entry["cases"]),
            "backend": "GPU", "gpuPrecision": protocol["precision"],
            "mtp": str(entry["mtp"]).lower(), "modelPath": protocol["modelPath"],
            "allowSourceTextFallback": str(protocol["sourceFallback"]).lower(),
            "allowGeneratedDslRepair": str(protocol["generatedDslRecovery"]).lower(),
        }.items():
            need(extras.get(key) == expected, f"{name}: host invocation {key} differs")
    shared = path / "shared_prompt.json"
    need(config.get("prompt", {}).get("assetSha256") == trained.sha256_file(shared),
         f"{name}: shared prompt hash differs from run configuration")
    prompt_doc = obj(shared)
    need(config.get("prompt", {}).get("contractSha256") == prompt_doc.get("contract_sha256"),
         f"{name}: shared prompt contract hash differs")
    if baseline is not None:
        for key in ("profile", "corpus", "prompt", "model", "device"):
            need(config.get(key) == baseline.get(key), f"{name}: {key} differs across batches")
        stable_runtime = lambda c: {k: v for k, v in c["runtime"].items() if k != "mtpEnabled"}
        need(stable_runtime(config) == stable_runtime(baseline),
             f"{name}: runtime settings differ beyond MTP")
    by_id = {}
    for index, result in enumerate(result_rows):
        case_id = result["id"]
        case_dir = path / case_id
        need(obj(case_dir / "result.json") == result, f"{name}/{case_id}: result.json differs")
        if interruption is not None:
            need(number(result.get("finishedAtEpochMs")) is not None,
                 f"{name}/{case_id}: interrupted batch result did not finish")
        source_ok, reason = trained.verify_source(case_dir / "source.json", frozen[case_id])
        need(source_ok, f"{name}/{case_id}: {reason}")
        parity = trained.audit_rendered_prompt(case_dir, result, True, prompt_doc,
                                               str(frozen[case_id].get("response_text") or ""))
        need(parity["promptJsonMatchesContract"] is True
             and parity["promptRecordedHashMatchesExpected"] is True
             and parity["status"] != "mismatch",
             f"{name}/{case_id}: prompt parity {parity['status']}: {parity['reason']}")
        native = trained.read_json(case_dir / "metrics.json", {})
        need(isinstance(native, dict), f"{name}/{case_id}: metrics.json is not an object")
        for key, value in result.get("metrics", {}).items():
            if key in native:
                need(native[key] == value, f"{name}/{case_id}: metrics.json {key} differs")
        need(result.get("firstInRun") is (index == 0),
             f"{name}/{case_id}: cold firstInRun flag inconsistent")
        runtime_label = str(result.get("runtime") or "")
        if runtime_label:
            need("GPU" in runtime_label and ("MTP" in runtime_label) == entry["mtp"],
                 f"{name}/{case_id}: reported runtime differs from mode")
        by_id[case_id] = result
    session = (obj(path / "session_metrics.json") if interruption is None else
               {"speculativeDecodingEnabled": entry["mtp"], "drafterAcceptanceRate": None,
                "unavailableReason": interruption.get("sessionAcceptanceUnavailableReason")})
    need(session.get("speculativeDecodingEnabled") is entry["mtp"],
         f"{name}: session MTP flag differs")
    return config, by_id, session, interruption


def audit(root: Path, partial: bool) -> tuple[dict, list[dict], dict, dict]:
    protocol = obj(root / "protocol.json")
    provenance = obj(root / "provenance.json")
    for key in ("mainApkDeviceSha256", "testApkHostSha256", "modelDeviceSha256"):
        need(isinstance(provenance.get(key), str) and len(provenance[key]) == 64,
             f"provenance.{key} missing")
    need(protocol.get("precision") == "FP32" and protocol.get("batchSize") == 5,
         "protocol is not the five-case FP32 benchmark")
    need(protocol.get("order") == "ABBA across adjacent five-case groups",
         "protocol order differs")
    frozen_rows = trained.frozen_index()
    frozen = {row["id"]: row for row in frozen_rows}
    plan = expected_plan(list(frozen))
    need(protocol.get("plan") == plan, "protocol plan does not cover each frozen case in ABBA order")
    batches = []
    mode_rows = {"off": {}, "on": {}}
    baseline = None
    missing = []
    interrupted = []
    for entry in plan:
        batch = root / "batches" / entry["batch"]
        if not batch.is_dir():
            missing.append(entry["batch"])
            continue
        config, results, session, interruption = audit_batch(root, entry, protocol, frozen, baseline, partial)
        baseline = baseline or config
        mode = "on" if entry["mtp"] else "off"
        for case_id, result in results.items():
            need(case_id not in mode_rows[mode], f"duplicate {mode}/{case_id}")
            mode_rows[mode][case_id] = {"result": result, "batch": entry["batch"]}
        if interruption is not None:
            interrupted.append(entry["batch"])
        batches.append({"name": entry["batch"], "mode": mode, "cases": list(results),
                        "plannedCases": entry["cases"], "interruption": interruption,
                        "runId": config["runId"], "session": session,
                        "startedAtEpochMs": config.get("startedAtEpochMs"),
                        "finishedAtEpochMs": (obj(batch / "summary.json").get("finishedAtEpochMs")
                                              if interruption is None else None)})
    if missing and not partial:
        raise ValueError(f"incomplete full-50 run: {len(missing)} batches missing: {', '.join(missing)}; use --partial for a progress report")
    for mode in ("off", "on"):
        if not partial:
            need(set(mode_rows[mode]) == set(frozen), f"{mode}: expected exactly 50 unique cases")
    need(batches, "no complete batches to report")
    return protocol, batches, mode_rows, {"provenance": provenance, "missingBatches": missing,
                                          "interruptedBatches": interrupted,
                                          "frozenIds": list(frozen), "baseline": baseline}


def merge(root: Path, out: Path, mode: str, cases: dict, baseline: dict,
          official_ids: list[str], prompt_source_batch: str) -> Path:
    target = out / f"mtp_{mode}"
    expected = out.resolve() / f"mtp_{mode}"
    need(target.resolve() == expected, f"merge target resolves outside report output: {target}")
    if target.exists():
        need(not target.is_symlink()
             and not (hasattr(target, "is_junction") and target.is_junction()),
             f"refusing to replace linked directory: {target}")
        marker_path = target / ".mtp_report_owned.json"
        need(marker_path.is_file() and not marker_path.is_symlink(),
             f"refusing to replace unowned directory: {target}")
        marker = obj(marker_path)
        need(marker == {"tool": "report_mtp_bixby50.py", "rawRoot": str(root), "mode": mode},
             f"refusing to replace directory owned by another report: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    ordered = [case_id for case_id in official_ids if case_id in cases]
    config = json.loads(json.dumps(baseline))
    config["runId"] = f"merged_mtp_{mode}"
    config["cases"] = ordered
    config["runtime"]["mtpEnabled"] = mode == "on"
    config["corpus"]["selectedRows"] = len(ordered)
    config["mergedFromBatches"] = [cases[case_id]["batch"] for case_id in ordered]
    trained.write_json(target / "run_config.json", config)
    first = root / "batches" / (cases[ordered[0]]["batch"] if ordered else prompt_source_batch) / "shared_prompt.json"
    shutil.copy2(first, target / "shared_prompt.json")
    results = []
    for case_id in ordered:
        source = root / "batches" / cases[case_id]["batch"] / case_id
        shutil.copytree(source, target / case_id)
        results.append(cases[case_id]["result"])
    trained.write_json(target / "results.json", results)
    trained.write_json(target / "summary.json", {"runId": config["runId"], "total": len(ordered),
                       "completed": len(results), "runComplete": len(results) == 50,
                       "mergedBatches": sorted(set(config["mergedFromBatches"]))})
    trained.write_json(target / ".mtp_report_owned.json", {"tool": "report_mtp_bixby50.py",
                       "rawRoot": str(root), "mode": mode})
    return target


def metric(result: dict | None, key: str) -> float | None:
    metrics = result.get("metrics") if isinstance(result, dict) else None
    return number(metrics.get(key)) if isinstance(metrics, dict) else None


def scored_index(path: Path) -> dict[str, dict]:
    return {row["id"]: row["metrics"] for row in trained.read_jsonl(path / "scored_predictions.jsonl")}


def case_record(case_id: str, entry: dict | None, score: dict | None,
                root: Path) -> dict:
    if entry is None:
        return {"present": False}
    result, batch = entry["result"], entry["batch"]
    case_dir = root / "batches" / batch / case_id
    output = case_dir / "output.express"
    before = result.get("deviceBefore") if isinstance(result.get("deviceBefore"), dict) else {}
    after = result.get("deviceAfter") if isinstance(result.get("deviceAfter"), dict) else {}
    metrics = trained.read_json(case_dir / "metrics.json", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "present": True, "batch": batch, "status": result.get("status"),
        "firstInRun": result.get("firstInRun") is True,
        "rawAndroidStrict": result.get("rawStrictValid") is True,
        "androidStrictAfterRepair": result.get("strictValid") is True,
        "pythonStrictRaw": score.get("schema_valid_strict") is True if score else None,
        "rendererContractValid": result.get("rendererContractValid") is True,
        "screenConfirmed": result.get("renderValid") is True and result.get("screenCaptured") is True
                           and result.get("currentCaseVisible") is True,
        "screenCaptured": result.get("screenCaptured") is True,
        "currentCaseVisible": result.get("currentCaseVisible") is True,
        "usedFallback": result.get("usedFallback") is True,
        "repairKind": result.get("repairKind"),
        "finishReason": metrics.get("finishReason"), "finishDetail": metrics.get("finishDetail"),
        "outputTokens": metric(result, "outputTokens"),
        "inputTokens": metric(result, "inputTokens"),
        "decodeTokensPerSecond": metric(result, "decodeTokensPerSecond"),
        "prefillTokensPerSecond": metric(result, "prefillTokensPerSecond"),
        "timeToFirstTokenSeconds": metric(result, "timeToFirstTokenSeconds"),
        "engineInitializationSeconds": metric(result, "engineInitializationSeconds"),
        "nativeInitializationPhaseSeconds": metric(result, "nativeInitializationPhaseSeconds"),
        "engineInitializedForRequest": result.get("metrics", {}).get("engineInitializedForRequest"),
        "providerCallMs": metric(result, "providerCallMs"),
        "elapsedMs": number(result.get("elapsedMs")),
        "startedAtEpochMs": number(result.get("startedAtEpochMs")),
        "finishedAtEpochMs": number(result.get("finishedAtEpochMs")),
        "thermalBefore": number(before.get("thermalStatus")),
        "thermalAfter": number(after.get("thermalStatus")),
        "batteryTemperatureBeforeC": number(before.get("batteryTemperatureC")),
        "batteryTemperatureAfterC": number(after.get("batteryTemperatureC")),
        "generationRewardV54": score.get("generation_reward_v5_4") if score else None,
        "renderArtifactQualityV54": score.get("render_artifact_quality_v5_4") if score else None,
        "contentCoverage": score.get("content_coverage") if score else None,
        "fidelityAtomicsV54": score.get("fidelity_atomics_v5_4") if score else None,
        "outputPath": str(output) if output.is_file() else None,
        "screenPath": str(case_dir / "screen.png") if (case_dir / "screen.png").is_file() else None,
        "screenScrolledPath": str(case_dir / "screen_scrolled.png") if (case_dir / "screen_scrolled.png").is_file() else None,
        "sourcePath": str(case_dir / "source.json"),
        "sourceFidelityWarnings": result.get("sourceFidelityWarnings"),
    }


def add_case_telemetry(cases: list[dict], root: Path) -> None:
    """Attach observed host clock caps during each case, without interpolation."""
    path = root / "telemetry.jsonl"
    samples = trained.read_jsonl(path) if path.is_file() else []
    for row in cases:
        for mode in ("off", "on"):
            case = row[mode]
            if not case.get("present"):
                continue
            start, end = case.get("startedAtEpochMs"), case.get("finishedAtEpochMs")
            observed = [x for x in samples if x.get("batch") == case["batch"]
                        and start is not None and end is not None
                        and start <= (number(x.get("epochMs")) or -1) <= end]
            for name, source in (("sampledGpuMaxClockHz", "gpuMaxClockHz"),
                                 ("sampledGpuClockHz", "gpuClockHz"),
                                 ("sampledThermalStatus", "thermalStatus")):
                values = [number(x.get(source)) for x in observed]
                values = [x for x in values if x is not None]
                case[name + "Min"] = min(values, default=None)
                case[name + "Median"] = med(values)
                case[name + "Max"] = max(values, default=None)
            case["hostTelemetrySamplesDuringCase"] = len(observed)


def matched_cap_subset(cases: list[dict], tolerance: float = 0.10) -> dict:
    """Exploratory warm-case subset with host-observed median GPU cap within 10%."""
    ids = []
    for row in cases:
        off, on = row["off"], row["on"]
        if not off.get("present") or not on.get("present") or off["firstInRun"] or on["firstInRun"]:
            continue
        a, b = number(off.get("sampledGpuMaxClockHzMedian")), number(on.get("sampledGpuMaxClockHzMedian"))
        if a is not None and b is not None and a > 0 and abs(a - b) / max(a, b) <= tolerance:
            ids.append(row["id"])
    selected = [row for row in cases if row["id"] in ids]
    return {"caseIds": ids, "toleranceFraction": tolerance,
            "hostSamplingNote": "15-second host samples can miss cap changes; matched sampled caps do not control all heat or workload differences.",
            "warmDecode": paired_wins(selected, "decodeTokensPerSecond", warm=True)}


def paired_wins(cases: list[dict], key: str, higher: bool = True, warm: bool = False) -> dict:
    wins = Counter()
    differences = []
    ratios = []
    for row in cases:
        off, on = row["off"], row["on"]
        if not off.get("present") or not on.get("present"):
            continue
        if warm and (off["firstInRun"] or on["firstInRun"]):
            continue
        a, b = number(off.get(key)), number(on.get(key))
        if a is None or b is None:
            continue
        differences.append(b - a)
        if a > 0 and b > 0:
            ratios.append(b / a)
        if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
            wins["tie"] += 1
        elif (b > a) == higher:
            wins["on"] += 1
        else:
            wins["off"] += 1
    return {"pairedCount": sum(wins.values()), "onWins": wins["on"],
            "offWins": wins["off"], "ties": wins["tie"],
            "medianOnMinusOff": med(differences), "medianOnDivOff": med(ratios)}


def summarize_mode(cases: list[dict], mode: str, batches: list[dict], max_tokens: int) -> dict:
    records = [r[mode] for r in cases if r[mode].get("present")]
    warm = [r for r in records if not r["firstInRun"]]
    cold = [r for r in records if r["firstInRun"]]
    selection = lambda selected, key: med([number(r.get(key)) for r in selected])
    acceptance = [{"batch": b["name"], "rate": number(b["session"].get("drafterAcceptanceRate"))}
                  for b in batches if b["mode"] == mode]
    observed_acceptance = [x["rate"] for x in acceptance if x["rate"] is not None]
    caps = [r["sampledGpuMaxClockHzMin"] for r in records
            if r.get("sampledGpuMaxClockHzMin") is not None]
    cap_max = [r["sampledGpuMaxClockHzMax"] for r in records
               if r.get("sampledGpuMaxClockHzMax") is not None]
    return {
        "cases": len(records), "coldHeads": len(cold), "warmCases": len(warm),
        "rawAndroidStrict": sum(r["rawAndroidStrict"] for r in records),
        "pythonStrictRaw": sum(r["pythonStrictRaw"] is True for r in records),
        "androidStrictAfterRepair": sum(r["androidStrictAfterRepair"] for r in records),
        "rendererContractValid": sum(r["rendererContractValid"] for r in records),
        "screenConfirmed": sum(r["screenConfirmed"] for r in records),
        "fallbacks": sum(r["usedFallback"] for r in records),
        "generatedDslRepairs": sum(r["repairKind"] == "GENERATED_DSL_REPAIR" for r in records),
        "finishReasons": dict(Counter(str(r["finishReason"] or "unavailable") for r in records)),
        "repetitionFinishes": sum("REPET" in str(r["finishReason"] or "").upper()
                                  or "REPET" in str(r["finishDetail"] or "").upper() for r in records),
        "outputAtOrAboveCap": sum((r["outputTokens"] or 0) >= max_tokens for r in records),
        "medianDecodeTokensPerSecondWarm": selection(warm, "decodeTokensPerSecond"),
        "medianDecodeTokensPerSecondCold": selection(cold, "decodeTokensPerSecond"),
        "medianPrefillTokensPerSecondWarm": selection(warm, "prefillTokensPerSecond"),
        "medianTimeToFirstTokenSecondsWarm": selection(warm, "timeToFirstTokenSeconds"),
        "medianProviderCallMsWarm": selection(warm, "providerCallMs"),
        "medianElapsedMsWarm": selection(warm, "elapsedMs"),
        "medianEngineInitializationSecondsCold": selection(cold, "engineInitializationSeconds"),
        "medianNativeInitializationPhaseSecondsCold": selection(cold, "nativeInitializationPhaseSeconds"),
        "medianGenerationRewardV54": selection(records, "generationRewardV54"),
        "medianRenderArtifactQualityV54": selection(records, "renderArtifactQualityV54"),
        "medianContentCoverage": selection(records, "contentCoverage"),
        "fidelityAtomicsObserved": fidelity_observed(records),
        "thermalStatusBeforeRange": [min([r["thermalBefore"] for r in records if r["thermalBefore"] is not None], default=None),
                                     max([r["thermalBefore"] for r in records if r["thermalBefore"] is not None], default=None)],
        "batteryTemperatureBeforeCRange": [min([r["batteryTemperatureBeforeC"] for r in records if r["batteryTemperatureBeforeC"] is not None], default=None),
                                           max([r["batteryTemperatureBeforeC"] for r in records if r["batteryTemperatureBeforeC"] is not None], default=None)],
        "observedGpuMaxClockMHzRange": [min(caps, default=None) / 1e6 if caps else None,
                                        max(cap_max, default=None) / 1e6 if cap_max else None],
        "casesWithHostClockSamples": sum(r["hostTelemetrySamplesDuringCase"] > 0 for r in records),
        "sessionAcceptanceRates": acceptance,
        "sessionAcceptanceUnweightedMedian": med(observed_acceptance),
        "sessionAcceptanceNote": "Per-batch SDK rates; no accepted/drafted counts were recorded, so no global weighted acceptance rate is computed.",
    }


def fidelity_observed(records: list[dict]) -> dict:
    values: dict[str, list[float]] = {}
    scored_count = sum(record.get("generationRewardV54") is not None for record in records)
    for record in records:
        atomics = record.get("fidelityAtomicsV54")
        if isinstance(atomics, Mapping):
            for key, value in atomics.items():
                n = number(value)
                if n is not None:
                    values.setdefault(str(key), []).append(n)
    return {key: {"mean": avg(nums), "observed": len(nums), "scoredOutputs": scored_count}
            for key, nums in sorted(values.items())}


def telemetry_summary(root: Path, batches: list[dict]) -> dict:
    path = root / "telemetry.jsonl"
    if not path.is_file():
        return {"available": False}
    records = trained.read_jsonl(path)
    by_batch = {}
    for batch in batches:
        observed = [r for r in records if r.get("batch") == batch["name"]]
        clocks = [number(r.get("gpuMaxClockHz")) for r in observed]
        clocks = [x for x in clocks if x is not None]
        thermal = [number(r.get("thermalStatus")) for r in observed]
        thermal = [x for x in thermal if x is not None]
        by_batch[batch["name"]] = {
            "samples": len(observed), "gpuMaxClockHzMin": min(clocks, default=None),
            "gpuMaxClockHzMax": max(clocks, default=None),
            "thermalStatusMin": min(thermal, default=None),
            "thermalStatusMax": max(thermal, default=None),
        }
    return {"available": True, "source": str(path), "sampleCount": len(records),
            "batches": by_batch}


def write_csv(path: Path, cases: list[dict]) -> None:
    keys = ["id", "domain", "paired", "off_batch", "on_batch", "off_firstInRun", "on_firstInRun"]
    fields = ["status", "rawAndroidStrict", "pythonStrictRaw", "androidStrictAfterRepair",
              "rendererContractValid", "screenConfirmed", "usedFallback", "repairKind",
              "finishReason", "finishDetail", "outputTokens", "inputTokens",
              "decodeTokensPerSecond", "prefillTokensPerSecond", "timeToFirstTokenSeconds",
              "engineInitializationSeconds", "nativeInitializationPhaseSeconds",
              "providerCallMs", "elapsedMs", "thermalBefore", "thermalAfter",
              "batteryTemperatureBeforeC", "batteryTemperatureAfterC", "generationRewardV54",
              "renderArtifactQualityV54", "contentCoverage", "outputPath", "screenPath"]
    keys += [f"{mode}_{field}" for mode in ("off", "on") for field in fields]
    fields += ["hostTelemetrySamplesDuringCase", "sampledGpuMaxClockHzMin",
               "sampledGpuMaxClockHzMedian", "sampledGpuMaxClockHzMax",
               "sampledGpuClockHzMin", "sampledGpuClockHzMedian", "sampledGpuClockHzMax"]
    keys += [f"{mode}_{field}" for mode in ("off", "on") for field in fields[-7:]]
    keys += ["on_minus_off_decodeTokensPerSecond", "on_minus_off_providerCallMs",
             "on_minus_off_generationRewardV54"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in cases:
            off, on = row["off"], row["on"]
            out = {"id": row["id"], "domain": row["domain"],
                   "paired": off.get("present") and on.get("present"),
                   "off_batch": off.get("batch"), "on_batch": on.get("batch"),
                   "off_firstInRun": off.get("firstInRun"), "on_firstInRun": on.get("firstInRun")}
            for mode in ("off", "on"):
                out.update({f"{mode}_{key}": row[mode].get(key) for key in fields})
            for field in ("decodeTokensPerSecond", "providerCallMs", "generationRewardV54"):
                a, b = number(off.get(field)), number(on.get(field))
                out[f"on_minus_off_{field}"] = b - a if a is not None and b is not None else None
            writer.writerow(out)


def link(path: str | None, out: Path, label: str) -> str:
    if not path:
        return ""
    href = quote(Path(os.path.relpath(path, out)).as_posix(), safe="/._-")
    return f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'


def make_html(out: Path, cases: list[dict], complete: bool) -> str:
    cards = []
    rates = []
    for row in cases:
        case_id = row["id"]
        columns = []
        for mode in ("off", "on"):
            r = row[mode]
            if not r.get("present"):
                columns.append(f"<section><h3>MTP {mode}</h3><p>Missing batch</p></section>")
                continue
            raw = Path(r["outputPath"]).read_text(encoding="utf-8-sig") if r.get("outputPath") else ""
            screen = r.get("screenPath")
            image = (f'<img loading="lazy" src="{html.escape(quote(Path(os.path.relpath(screen, out)).as_posix(), safe="/._-"), quote=True)}" alt="{case_id} MTP {mode} screen">'
                     if screen else "<p>No screen artifact</p>")
            artifact_links = " · ".join(x for x in (
                link(r.get("outputPath"), out, "raw output"), link(screen, out, "screen"),
                link(r.get("screenScrolledPath"), out, "scrolled screen"),
                link(r.get("sourcePath"), out, "source")) if x)
            columns.append(f'<section><h3>MTP {mode}</h3><p>Decode {fmt(r.get("decodeTokensPerSecond"))} tok/s · '
                           f'Wall {fmt(r.get("providerCallMs"), 0)} ms · Raw strict {r["rawAndroidStrict"]} · '
                           f'Screen confirmed {r["screenConfirmed"]}</p><p>{artifact_links}</p>'
                           f'{image}<details><summary>Full raw output</summary><pre>{html.escape(raw)}</pre></details></section>')
        cards.append(f'<article id="{case_id}"><h2>{case_id} · {html.escape(row["domain"])}</h2>'
                     f'<div class="pair">{"".join(columns)}</div></article>')
        a, b = number(row["off"].get("decodeTokensPerSecond")), number(row["on"].get("decodeTokensPerSecond"))
        if a is not None and b is not None:
            maximum = max(a, b, 1)
            rates.append(f'<tr><th><a href="#{case_id}">{case_id}</a></th>'
                         f'<td><span class="bar off" style="width:{a/maximum*100:.1f}%"></span>{a:.1f}</td>'
                         f'<td><span class="bar on" style="width:{b/maximum*100:.1f}%"></span>{b:.1f}</td></tr>')
    status = "COMPLETE: 50 paired cases" if complete else "INCOMPLETE PROGRESS REPORT — NO WINNER"
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bixby50 MTP on/off comparison</title><style>
body{{font:15px system-ui,sans-serif;color:#192337;background:#f3f5f9;margin:0}}main{{max-width:1550px;margin:auto;padding:20px}}
article,header,table{{background:white;border:1px solid #d6dce8;border-radius:10px;margin:16px 0;padding:14px}}
.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}section{{min-width:0}}h2{{margin:0 0 10px}}
img{{max-width:100%;max-height:650px;object-fit:contain;background:#222}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:600px;overflow:auto;background:#f0f2f6;padding:12px}}
table{{width:100%;border-spacing:8px;text-align:left}}td{{width:42%}}.bar{{display:inline-block;height:12px;min-width:2px;margin-right:8px;border-radius:3px}}.off{{background:#4066aa}}.on{{background:#e28932}}
@media(max-width:850px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>Bixby50 GPU FP32 · MTP comparison</h1><strong>{status}</strong>
<p>Bars show native decode tokens/s for each paired case. Cold batch heads and thermal conditions appear in the CSV. The full raw output is expandable under each screenshot.</p></header>
<table><thead><tr><th>Case</th><th>MTP off tok/s</th><th>MTP on tok/s</th></tr></thead><tbody>{''.join(rates)}</tbody></table>
{''.join(cards)}</main></body></html>'''


def make_markdown(report: dict) -> str:
    complete = report["complete"]
    modes, paired = report["modes"], report["paired"]
    off, on = modes["off"], modes["on"]
    matched_off, matched_on = report["matchedModes"]["off"], report["matchedModes"]["on"]
    status = "COMPLETE — 50 per mode" if complete else "INCOMPLETE — no winner or full-50 claim"
    def metric_row(label: str, key: str) -> str:
        return f"| Paired {label} | {fmt(matched_off[key])} | {fmt(matched_on[key])} |"
    atomics = sorted(set(matched_off["fidelityAtomicsObserved"]) | set(matched_on["fidelityAtomicsObserved"]))
    atomic_rows = []
    for key in atomics:
        a, b = matched_off["fidelityAtomicsObserved"].get(key, {}), matched_on["fidelityAtomicsObserved"].get(key, {})
        atomic_rows.append(f"| `{key}` | {fmt(a.get('mean'), 3)} ({a.get('observed', 0)}/{a.get('scoredOutputs', 0)}) | "
                           f"{fmt(b.get('mean'), 3)} ({b.get('observed', 0)}/{b.get('scoredOutputs', 0)}) |")
    review_links = ", ".join(f"[{html.escape(name)}]({href})" for name, href in report["reviewDocuments"].items())
    acceptance_rows = "\n".join(f"| `{item['batch']}` | {fmt(item['rate'], 4)} |"
                                for item in on["sessionAcceptanceRates"])
    return f"""# Bixby50 MTP on/off comparison

**{status}.** {off['cases']} completed off cases and {on['cases']} completed on cases; **{paired['caseCount']} matched pairs**, {paired['offOnlyCount']} off-only, {paired['onOnlyCount']} on-only. The user stopped the run; the interrupted batch contributes only finished records. Same frozen source, prompt, model path, GPU FP32 settings, and device provenance were checked in each included batch. The paired five-case batches alternate in ABBA order across adjacent groups. The speed and score rows below use only the matched cases, since the all-completed mode sets have different case counts.

| Measure | MTP off | MTP on |
|---|---:|---:|
| Raw Android strict valid | {off['rawAndroidStrict']}/{off['cases']} | {on['rawAndroidStrict']}/{on['cases']} |
| Python strict valid, raw output | {off['pythonStrictRaw']}/{off['cases']} | {on['pythonStrictRaw']}/{on['cases']} |
| Raw Android strict valid, matched cases | {matched_off['rawAndroidStrict']}/{matched_off['cases']} | {matched_on['rawAndroidStrict']}/{matched_on['cases']} |
| Android strict after generated-DSL repair | {off['androidStrictAfterRepair']}/{off['cases']} | {on['androidStrictAfterRepair']}/{on['cases']} |
| Generated-DSL repairs | {off['generatedDslRepairs']} | {on['generatedDslRepairs']} |
| Renderer contract valid | {off['rendererContractValid']}/{off['cases']} | {on['rendererContractValid']}/{on['cases']} |
| Screen confirmed | {off['screenConfirmed']}/{off['cases']} | {on['screenConfirmed']}/{on['cases']} |
| Source fallback used | {off['fallbacks']} | {on['fallbacks']} |
| Output at/above {report['protocol']['maxOutputTokens']} token cap | {off['outputAtOrAboveCap']} | {on['outputAtOrAboveCap']} |
| Repetition finish reason/detail | {off['repetitionFinishes']} | {on['repetitionFinishes']} |
{metric_row('Warm median native decode tokens/s', 'medianDecodeTokensPerSecondWarm')}
{metric_row('Warm median native prefill tokens/s', 'medianPrefillTokensPerSecondWarm')}
{metric_row('Warm median time to first token, s', 'medianTimeToFirstTokenSecondsWarm')}
{metric_row('Warm median provider wall, ms', 'medianProviderCallMsWarm')}
{metric_row('Cold median engine initialization, s', 'medianEngineInitializationSecondsCold')}
{metric_row('Cold median native initialization phase, s', 'medianNativeInitializationPhaseSecondsCold')}
{metric_row('Median v5.4 generation reward, scored outputs', 'medianGenerationRewardV54')}
{metric_row('Median v5.4 render artifact quality, scored outputs', 'medianRenderArtifactQualityV54')}
{metric_row('Median legacy lexical content coverage, scored outputs', 'medianContentCoverage')}

The legacy lexical `content_coverage` scans the canonical graph, including unreachable content. It does not measure visible-content accuracy. The paired-set v5.4 fidelity atomics below are means of **observed numeric values only**; null, absent, and inapplicable values are excluded. Denominators show observed/scored outputs, and raw composite v5.4 reward keeps its scorer-defined zero for parse-failed raw outputs. These are source-comparison diagnostics, not certification of cited-claim truth.

| v5.4 fidelity atomic | MTP off observed mean (n/scored) | MTP on observed mean (n/scored) |
|---|---:|---:|
{chr(10).join(atomic_rows) if atomic_rows else '| unavailable | n/a | n/a |'}

Warm paired native decode: **{paired['warmDecode']['onWins']} on wins, {paired['warmDecode']['offWins']} off wins, {paired['warmDecode']['ties']} ties** across {paired['warmDecode']['pairedCount']} comparable cases; median on/off ratio {fmt(paired['warmDecode']['medianOnDivOff'])}. Warm provider wall: {paired['providerWallWarm']['onWins']} on shorter, {paired['providerWallWarm']['offWins']} off shorter, {paired['providerWallWarm']['ties']} ties across {paired['providerWallWarm']['pairedCount']} pairs. Paired generation reward: {paired['reward']['onWins']} on wins, {paired['reward']['offWins']} off wins, {paired['reward']['ties']} ties across {paired['reward']['pairedCount']} scored pairs. These are descriptive paired observations, not a significance test.

Native `decodeTokensPerSecond` measures decode throughput; `providerCallMs` includes request work and depends on output length. The first case of **each** five-case batch is cold, giving {matched_off['coldHeads']} off and {matched_on['coldHeads']} on heads in the matched set. Cold initialization is reported separately. The run's time-to-first-token and prefill fields are native measurements where available. Session MTP drafter acceptance is recorded **per on batch** in `comparison.json`; its unweighted median across available sessions is {fmt(on['sessionAcceptanceUnweightedMedian'], 3)}. The interrupted session has no acceptance value. No accepted/drafted token counts were recorded, so a global acceptance ratio cannot be inferred.

| MTP-on batch | Session drafter acceptance |
|---|---:|
{acceptance_rows if acceptance_rows else '| unavailable | n/a |'}

Finish reasons: off `{json.dumps(off['finishReasons'], sort_keys=True)}`; on `{json.dumps(on['finishReasons'], sort_keys=True)}`. Token-cap counts use observed output tokens; a finish detail is preserved per case in the CSV/JSON. A cap, repetition finish, or damaged output can affect quality and wall time independently of decode throughput.

Thermal status before cases ranges {off['thermalStatusBeforeRange']} off and {on['thermalStatusBeforeRange']} on; battery temperature ranges {off['batteryTemperatureBeforeCRange']} °C off and {on['batteryTemperatureBeforeCRange']} °C on. The **observed GPU clock cap during cases** ranges {off['observedGpuMaxClockMHzRange']} MHz off and {on['observedGpuMaxClockMHzRange']} MHz on, with host samples in {off['casesWithHostClockSamples']}/{off['cases']} off and {on['casesWithHostClockSamples']}/{on['cases']} on cases. Falling clock caps under sustained load are a plausible contributor to lower decode speed; the per-case samples in the CSV make this checkable. ABBA balances order across adjacent groups but cannot make paired device heat, clock limits, or timing identical. These observations do not isolate FP32 or MTP as the sole cause of a speed difference.

An exploratory subset has {paired['similarSampledGpuCap']['warmDecode']['pairedCount']} warm pairs whose **observed median GPU clock caps** differ by at most 10%: {paired['similarSampledGpuCap']['warmDecode']['onWins']} on wins, {paired['similarSampledGpuCap']['warmDecode']['offWins']} off wins, {paired['similarSampledGpuCap']['warmDecode']['ties']} ties. Fifteen-second host samples can miss clock changes; this subset is not a normalized performance estimate.

Raw Android and Python strict validity use separate implementations. Renderer contract validity and screen confirmation are separate: a capture failure does not by itself establish invalid model output. Likewise, a rendering success or v5.4 score does not prove full source/content fidelity. SDK literal-only source warnings are diagnostic and may overcount omissions in reachable table `statePath` data; they are not a factual-accuracy numerator. An exact-number atomic can also include citation digits, and a source-link atomic does not certify which claim a citation supports. Scores exclude unavailable outputs rather than assigning zeros; review the full raw outputs and source in [side_by_side.html](side_by_side.html). MTP is a draft-and-verify decoding mechanism; different observed outputs alone do not show that it changed target weights or is inherently lower quality.

Evidence: [comparison.json](comparison.json), [per_case_comparison.csv](per_case_comparison.csv), [side_by_side.html](side_by_side.html), [off mode](mtp_off/REPORT.md), [on mode](mtp_on/REPORT.md){', ' + review_links if review_links else ''}. Missing batches: {', '.join(report['missingBatches']) if report['missingBatches'] else 'none'}. Interrupted batches: {', '.join(report['interruptedBatches']) if report['interruptedBatches'] else 'none'}.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="Report directory (default: run_dir/report)")
    parser.add_argument("--partial", action="store_true", help="Emit an explicitly incomplete progress report")
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    out = (args.out or root / "report").resolve()
    try:
        need(root.is_dir(), f"run directory does not exist: {root}")
        need(out != root and out not in root.parents,
             "report output cannot equal or contain the raw run directory")
        need(out == root / "report" or root not in out.parents,
             "report output must be run_dir/report or outside the raw run directory")
        protocol, batches, mode_rows, audit_info = audit(root, args.partial)
        complete = not audit_info["missingBatches"] and not audit_info["interruptedBatches"]
        if not args.partial:
            need(complete, "full run incomplete")
        out.mkdir(parents=True, exist_ok=True)
        scored = {}
        for mode in ("off", "on"):
            target = merge(root, out, mode, mode_rows[mode], audit_info["baseline"],
                           audit_info["frozenIds"], batches[0]["name"])
            summary, errors = trained.build_report(target)
            need(not errors, f"{mode}: scorer provenance errors: {errors}")
            if complete:
                need(not summary["require50Errors"], f"{mode}: {summary['require50Errors']}")
            scored[mode] = scored_index(target)
        frozen = {row["id"]: row for row in trained.frozen_index()}
        cases = []
        for case_id in audit_info["frozenIds"]:
            if case_id not in mode_rows["off"] and case_id not in mode_rows["on"]:
                continue
            source = trained.expected_source(frozen[case_id])
            cases.append({"id": case_id, "domain": source["domain"],
                          "off": case_record(case_id, mode_rows["off"].get(case_id),
                                             scored["off"].get(case_id), root),
                          "on": case_record(case_id, mode_rows["on"].get(case_id),
                                            scored["on"].get(case_id), root)})
        add_case_telemetry(cases, root)
        paired = {
            "caseCount": sum(r["off"].get("present") and r["on"].get("present") for r in cases),
            "offOnlyCount": sum(r["off"].get("present") and not r["on"].get("present") for r in cases),
            "onOnlyCount": sum(r["on"].get("present") and not r["off"].get("present") for r in cases),
            "warmDecode": paired_wins(cases, "decodeTokensPerSecond", warm=True),
            "coldDecode": paired_wins(cases, "decodeTokensPerSecond", warm=False),
            "providerWallWarm": paired_wins(cases, "providerCallMs", higher=False, warm=True),
            "reward": paired_wins(cases, "generationRewardV54"),
            "contentCoverage": paired_wins(cases, "contentCoverage"),
            "similarSampledGpuCap": matched_cap_subset(cases),
        }
        report = {"complete": complete, "winnerClaimAllowed": complete,
                  "protocol": protocol, "provenance": audit_info["provenance"],
                  "includedBatches": batches, "missingBatches": audit_info["missingBatches"],
                  "interruptedBatches": audit_info["interruptedBatches"],
                  "modes": {mode: summarize_mode(cases, mode, batches, protocol["maxOutputTokens"])
                            for mode in ("off", "on")},
                  "matchedModes": {mode: summarize_mode(
                      [r for r in cases if r["off"].get("present") and r["on"].get("present")],
                      mode, batches, protocol["maxOutputTokens"])
                      for mode in ("off", "on")},
                  "paired": paired, "telemetry": telemetry_summary(root, batches),
                  "cases": cases,
                  "interpretationLimit": "Descriptive device result; ABBA does not equalize thermal and GPU clocks. Rendering and scores do not prove full source fidelity."}
        report["reviewDocuments"] = {
            name: quote(Path(os.path.relpath(root / name, out)).as_posix(), safe="/._-")
            for name in ("SPEED_CONTEXT.md", "QUALITY_REVIEW.md") if (root / name).is_file()
        }
        trained.write_json(out / "comparison.json", report)
        write_csv(out / "per_case_comparison.csv", cases)
        trained.atomic_text(out / "side_by_side.html", make_html(out, cases, complete))
        trained.atomic_text(out / "REPORT.md", make_markdown(report))
        print(json.dumps({"complete": complete, "off": report["modes"]["off"]["cases"],
                          "on": report["modes"]["on"]["cases"], "paired": paired["caseCount"],
                          "offOnly": paired["offOnlyCount"], "onOnly": paired["onOnlyCount"],
                          "missingBatches": audit_info["missingBatches"],
                          "interruptedBatches": audit_info["interruptedBatches"],
                          "report": str(out)}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI gives a concise validation failure
        print(f"report_mtp_bixby50.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
