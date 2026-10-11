#!/usr/bin/env python3
"""Summarize bounded, on-device GenUICraft inference probes.

Reads copied instrumentation evidence only; it never starts adb or changes a device.
Reports are snapshots, so rerun as arms arrive. Final mode requires the expected
device/arm identities and refuses missing or incomplete evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "serial", "device", "arm", "arm_state", "arm_order", "case_ordinal", "case_id",
    "case_occurrence", "reuse", "mtp", "streaming", "runtime", "model_path",
    "model_bytes", "corpus_sha256", "prompt_asset_sha256", "status", "repair_kind",
    "engine_initialized", "expected_engine_initialized", "engine_reuse_verified",
    "thermal_ready", "thermal_timed_out", "thermal_start", "thermal_end",
    "battery_c_start", "battery_c_end", "cooldown_ms", "gpu_mhz_min",
    "gpu_mhz_median", "gpu_mhz_max", "telemetry_thermal_max",
    "conversion_ms", "provider_ms", "init_ms", "prefill_ms_derived",
    "decode_ms_derived", "provider_residual_ms_derived", "sdk_outside_provider_ms",
    "native_init_phase_ms_nonadditive", "native_ttft_ms", "first_partial_callback_ms",
    "first_preview_callback_ms", "first_ready_preview_callback_ms", "preview_count",
    "callback_observer_body_ms", "input_tokens", "output_tokens", "prefill_tok_s",
    "decode_tok_s", "attempts", "raw_accepted_without_repair", "final_a2ui_saved",
    "source_text_sha256", "rendered_prompt_sha256", "raw_sha256", "final_express_sha256",
    "final_a2ui_sha256", "profile_fingerprint",
    "qualified", "flags",
]


def read_json(path: Path, warnings: list[str], default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Unreadable {path}: {exc}")
        return default


def num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and abs(value) != float("inf") else None


def numeric_telemetry_line(value: str) -> float | None:
    """Parse one plain numeric sysfs line without accepting arbitrary text or NaN."""
    value = value.strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return None
    return num(float(value))


def rounded(value: Any, digits: int = 2) -> float | None:
    value = num(value)
    return round(value, digits) if value is not None else None


def median(values: list[Any]) -> float | None:
    good = [x for v in values if (x := num(v)) is not None]
    return rounded(statistics.median(good)) if good else None


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parsed_telemetry(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not path.is_file():
        return samples
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"Telemetry line {line_number} incomplete: {path}")
                continue
            lines = str(raw.get("raw", "")).splitlines()
            # Sampler raw payload: device epoch seconds, GPU MHz, GPU busy, max MHz, ...
            gpu = numeric_telemetry_line(lines[1]) if len(lines) > 1 else None
            if gpu is not None and gpu <= 0:
                gpu = None
            status_match = re.search(r"Thermal Status:\s*(\d+)", str(raw.get("raw", "")))
            samples.append({
                "epoch_ms": num(raw.get("hostEpochMs")),
                "gpu_mhz": gpu,
                "thermal_status": int(status_match.group(1)) if status_match else None,
            })
    except (OSError, UnicodeError) as exc:
        warnings.append(f"Could not read telemetry {path}: {exc}")
    return samples


def telemetry_for_case(samples: list[dict[str, Any]], before: Any, after: Any) -> dict[str, Any]:
    start = num((before or {}).get("epochMs"))
    end = num((after or {}).get("epochMs"))
    matching = [s for s in samples if start is not None and end is not None and
                s["epoch_ms"] is not None and start <= s["epoch_ms"] <= end]
    gpu = [x for s in matching if (x := num(s["gpu_mhz"])) is not None]
    thermal = [x for s in matching if (x := num(s["thermal_status"])) is not None]
    return {
        "gpu_mhz_min": min(gpu) if gpu else None,
        "gpu_mhz_median": median(gpu),
        "gpu_mhz_max": max(gpu) if gpu else None,
        "telemetry_thermal_max": max(thermal) if thermal else None,
    }


def phase_values(result: dict[str, Any]) -> dict[str, Any]:
    native = result.get("nativeMetrics") or {}
    provider = num(result.get("providerTotalMs"))
    conversion = num(result.get("conversionWallMs"))
    initialized = native.get("engineInitializedForRequest")
    init_sec = num(native.get("engineInitializationSeconds"))
    init = init_sec * 1000 if init_sec is not None else (0.0 if initialized is False else None)
    input_tokens = num(native.get("inputTokens"))
    output_tokens = num(native.get("outputTokens"))
    prefill_rate = num(native.get("prefillTokensPerSecond"))
    decode_rate = num(native.get("decodeTokensPerSecond"))
    prefill = input_tokens / prefill_rate * 1000 if input_tokens is not None and prefill_rate and prefill_rate > 0 else None
    decode = output_tokens / decode_rate * 1000 if output_tokens is not None and decode_rate and decode_rate > 0 else None
    residual = provider - init - prefill - decode if None not in (provider, init, prefill, decode) else None
    outside = conversion - provider if None not in (conversion, provider) else None
    init_phase = num(native.get("nativeInitializationPhaseSeconds"))
    ttft = num(native.get("timeToFirstTokenSeconds"))
    return {
        "conversion_ms": rounded(conversion), "provider_ms": rounded(provider),
        "init_ms": rounded(init), "prefill_ms_derived": rounded(prefill),
        "decode_ms_derived": rounded(decode),
        "provider_residual_ms_derived": rounded(residual),
        "sdk_outside_provider_ms": rounded(outside),
        "native_init_phase_ms_nonadditive": rounded(init_phase * 1000 if init_phase is not None else None),
        "native_ttft_ms": rounded(ttft * 1000 if ttft is not None else None),
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "prefill_tok_s": rounded(prefill_rate), "decode_tok_s": rounded(decode_rate),
    }


PROFILE_KEYS = (
    "gpuPrecision", "profile", "maxContextTokens", "maxOutputTokens", "temperature",
    "thinkingEnabled", "sourceFallbackEnabled", "generatedDslRepairEnabled",
    "promptContractSha256",
)


def profile_fingerprint(config: dict[str, Any]) -> str:
    stable = {key: config.get(key) for key in PROFILE_KEYS}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()


def load_arm(arm_dir: Path, arm_order: int, warnings: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device_dir = arm_dir / "device"
    config = read_json(device_dir / "run_config.json", warnings, {}) or {}
    results = read_json(device_dir / "results.json", warnings, []) or []
    summary = read_json(device_dir / "summary.json", warnings, {}) or {}
    teardown = read_json(device_dir / "teardown.json", warnings, []) or []
    warmup = read_json(device_dir / "warmup_result.json", warnings)
    host = read_json(arm_dir / "host_result.json", warnings, {}) or {}
    serial = arm_dir.parent.parent.name
    device = (config.get("device") or {}).get("model")
    measured_order = config.get("measuredOrder") or []
    expected = len(measured_order) or None
    ordered = bool(expected and all(isinstance(result, dict) and result.get("id") == case_id
                                    for result, case_id in zip(results, measured_order)))
    artifacts_present = bool(expected and all(
        (device_dir / f"{index:03d}_{case_id}" / "final.a2ui.json").is_file()
        for index, case_id in enumerate(measured_order, 1)))
    complete = bool(expected and summary.get("runComplete") is True and host.get("passed") is True and
                    host.get("pullExit") == 0 and len(results) == expected and ordered and artifacts_present and
                    summary.get("completed") == expected and summary.get("total") == expected and
                    summary.get("successful") == expected)
    completeness_gaps = []
    if config and not expected:
        completeness_gaps.append("measured-order-missing")
    if expected and len(results) != expected:
        completeness_gaps.append("result-count-mismatch")
    if expected and len(results) == expected and not ordered:
        completeness_gaps.append("case-order-mismatch")
    if expected and not artifacts_present:
        completeness_gaps.append("final-a2ui-artifact-missing")
    if config and summary.get("runComplete") is not True:
        completeness_gaps.append("device-summary-incomplete")
    if config and host.get("passed") is not True:
        completeness_gaps.append("host-pass-receipt-missing")
    if config and host.get("pullExit") != 0:
        completeness_gaps.append("host-pull-receipt-missing")
    if expected and (summary.get("completed"), summary.get("total"), summary.get("successful")) != (
            expected, expected, expected):
        completeness_gaps.append("device-summary-count-mismatch")
    reported_modes = [(entry.get("sessionMetrics") or {}).get("speculativeDecodingEnabled")
                      for entry in teardown if isinstance(entry, dict)]
    mtp_state_mismatch = bool(complete and (not reported_modes or
        any(mode is not config.get("mtpRequested") for mode in reported_modes)))
    state = "complete" if complete else ("partial" if config or results else "pending")
    sample_set = parsed_telemetry(arm_dir / "telemetry.jsonl", warnings)
    rows: list[dict[str, Any]] = []
    occurrences: Counter[str] = Counter()
    for index, result in enumerate(results, 1):
        if not isinstance(result, dict):
            warnings.append(f"Non-object result {index} in {arm_dir}")
            continue
        case_id = str(result.get("id", "unknown"))
        occurrences[case_id] += 1
        case_dir = device_dir / f"{index:03d}_{case_id}"
        source = read_json(case_dir / "source.json", warnings, {}) or {}
        native = result.get("nativeMetrics") or {}
        before = result.get("deviceBefore") or {}
        after = result.get("deviceAfter") or {}
        cooling = result.get("cooldown") or {}
        reuse = config.get("reuseEngine")
        actual_init = native.get("engineInitializedForRequest")
        expected_init = False if reuse is True else True if reuse is False else None
        flags = []
        if actual_init is not expected_init:
            flags.append("engine-reuse-mismatch" if actual_init is not None else "engine-state-missing")
        if cooling.get("timedOut"):
            flags.append("cooldown-timeout")
        if cooling.get("thermalReady") is not True:
            flags.append("thermal-gate-unverified")
        if before.get("thermalStatus") != 0:
            flags.append("thermal-start-nonzero")
        if after.get("thermalStatus") != 0:
            flags.append("thermal-end-nonzero")
        if result.get("status") != "complete":
            flags.append("conversion-incomplete")
        if result.get("attemptCount") != 1:
            flags.append("multiple-attempts")
        if not complete:
            flags.append("arm-partial")
        if mtp_state_mismatch:
            flags.append("mtp-state-mismatch")
        if not (case_dir / "final.a2ui.json").is_file() and result.get("status") == "complete":
            flags.append("final-artifact-missing")
        row = {
            "serial": serial, "device": device, "arm": arm_dir.name,
            "arm_state": state, "arm_order": arm_order, "case_ordinal": index,
            "case_id": case_id, "case_occurrence": occurrences[case_id],
            "reuse": reuse, "mtp": config.get("mtpRequested"),
            "streaming": config.get("streamingRenderingEnabled"),
            "runtime": result.get("runtime"),
            "model_path": config.get("modelPath"), "model_bytes": config.get("modelBytes"),
            "profile_fingerprint": profile_fingerprint(config),
            "corpus_sha256": config.get("corpusSha256"),
            "prompt_asset_sha256": config.get("promptSha256"),
            "status": result.get("status"), "repair_kind": result.get("repairKind"),
            "engine_initialized": actual_init, "expected_engine_initialized": expected_init,
            "engine_reuse_verified": actual_init is expected_init,
            "thermal_ready": cooling.get("thermalReady"),
            "thermal_timed_out": cooling.get("timedOut"),
            "thermal_start": before.get("thermalStatus"),
            "thermal_end": after.get("thermalStatus"),
            "battery_c_start": before.get("batteryTemperatureC"),
            "battery_c_end": after.get("batteryTemperatureC"),
            "cooldown_ms": rounded(cooling.get("elapsedMs")),
            "first_partial_callback_ms": rounded(result.get("firstPartialCallbackMs")),
            "first_preview_callback_ms": rounded(result.get("firstPreviewCallbackMs")),
            "first_ready_preview_callback_ms": rounded(result.get("firstReadyPreviewCallbackMs")),
            "preview_count": result.get("previewSnapshotCount"),
            "callback_observer_body_ms": rounded(result.get("callbackBodyWorkMs"), 3),
            "attempts": result.get("attemptCount"),
            "raw_accepted_without_repair": result.get("status") == "complete" and result.get("repairKind") == "NONE",
            "final_a2ui_saved": (case_dir / "final.a2ui.json").is_file(),
            "source_text_sha256": sha256_text(source.get("text")),
            "rendered_prompt_sha256": (result.get("attempts") or [{}])[0].get("renderedPromptSha256")
                if result.get("attempts") else None,
            "raw_sha256": sha256_file(case_dir / "attempt_1_raw.express"),
            "final_express_sha256": sha256_file(case_dir / "final.express"),
            "final_a2ui_sha256": sha256_file(case_dir / "final.a2ui.json"),
            **phase_values(result),
            **telemetry_for_case(sample_set, before, after),
        }
        if (residual := num(row["provider_residual_ms_derived"])) is not None and residual < -100:
            flags.append("native-phase-accounting-negative")
        if (outside := num(row["sdk_outside_provider_ms"])) is not None and outside < -100:
            flags.append("sdk-boundary-negative")
        row["qualified"] = not any(flag in flags for flag in (
            "engine-reuse-mismatch", "engine-state-missing", "cooldown-timeout",
            "thermal-gate-unverified", "thermal-start-nonzero", "thermal-end-nonzero", "conversion-incomplete",
            "multiple-attempts", "arm-partial", "final-artifact-missing",
            "native-phase-accounting-negative", "sdk-boundary-negative"))
        if mtp_state_mismatch:
            row["qualified"] = False
        row["flags"] = flags
        rows.append(row)
    if reuse := config.get("reuseEngine"):
        warm_init = (warmup or {}).get("engineInitializedForRequest")
        if warmup and warm_init is not True:
            warnings.append(f"{serial}/{arm_dir.name}: excluded warmup did not initialize engine")
    warmup_diagnostic = None
    if isinstance(warmup, dict):
        warmup_diagnostic = {
            "status": warmup.get("status"), "runtime": warmup.get("runtime"),
            "engineInitializedForRequest": (warmup.get("nativeMetrics") or {}).get("engineInitializedForRequest"),
            "thermalStart": (warmup.get("deviceBefore") or {}).get("thermalStatus"),
            "thermalEnd": (warmup.get("deviceAfter") or {}).get("thermalStatus"),
            "cooldownApplied": warmup.get("cooldown") is not None,
            **phase_values(warmup),
        }
    acceptance_values = [num((entry.get("sessionMetrics") or {}).get("drafterAcceptanceRate"))
                         for entry in teardown if isinstance(entry, dict)]
    acceptance_values = [x for x in acceptance_values if x is not None]
    arm = {
        "serial": serial, "device": device, "name": arm_dir.name, "order": arm_order,
        "state": state, "hostPassed": host.get("passed"),
        "completenessGaps": completeness_gaps,
        "startedAtEpochMs": host.get("startedAtEpochMs"),
        "runComplete": summary.get("runComplete"), "expectedRequests": expected,
        "observedRequests": len(rows), "qualifiedRequests": sum(r["qualified"] for r in rows),
        "config": config, "warmup": warmup,
        "warmupDiagnostic": warmup_diagnostic,
        "teardown": teardown,
        "mtpAcceptanceArm": acceptance_values[0] if config.get("reuseEngine") and len(acceptance_values) == 1 else None,
        "mtpAcceptanceEngineSessions": acceptance_values,
        "finishMetricsMedianMs": median([x.get("finishMetricsMs") for x in teardown if isinstance(x, dict)]),
        "closeMedianMs": median([x.get("closeAndAwaitMs") for x in teardown if isinstance(x, dict)]),
        "providerConstructionMedianMs": median([x.get("providerConstructionMs") for x in teardown if isinstance(x, dict)]),
        "telemetrySamples": len(sample_set),
        "mtpStateMismatch": mtp_state_mismatch,
    }
    valid = [r for r in rows if r["qualified"]]
    for key in ("conversion_ms", "provider_ms", "init_ms", "prefill_ms_derived",
                "decode_ms_derived", "provider_residual_ms_derived", "sdk_outside_provider_ms",
                "native_ttft_ms", "first_partial_callback_ms", "first_ready_preview_callback_ms"):
        arm[f"median_{key}"] = median([r[key] for r in rows])
        arm[f"qualifiedMedian_{key}"] = median([r[key] for r in valid])
    def weighted_decode(requests: list[dict[str, Any]]) -> float | None:
        rate_rows = [r for r in requests if num(r["output_tokens"]) is not None and
                     num(r["decode_ms_derived"]) is not None]
        total_tokens = sum(r["output_tokens"] for r in rate_rows)
        total_decode_ms = sum(r["decode_ms_derived"] for r in rate_rows)
        return rounded(total_tokens / (total_decode_ms / 1000)) if total_decode_ms > 0 else None
    arm["weightedDecodeTokS"] = weighted_decode(rows)
    arm["qualifiedWeightedDecodeTokS"] = weighted_decode(valid)
    arm["rawAcceptedWithoutRepair"] = sum(r["raw_accepted_without_repair"] for r in rows)
    arm["finalA2uiSaved"] = sum(r["final_a2ui_saved"] for r in rows)
    arm["thermalTimeouts"] = sum(r["thermal_timed_out"] is True for r in rows)
    arm["engineStateMismatches"] = sum(r["engine_reuse_verified"] is not True for r in rows)
    return arm, rows


def pairing(rows: list[dict[str, Any]], varied: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[bool, list[dict[str, Any]]]] = defaultdict(lambda: {False: [], True: []})
    other = "streaming" if varied == "mtp" else "mtp"
    for row in rows:
        if row[varied] not in (True, False):
            continue
        if not all(row.get(key) for key in ("source_text_sha256", "rendered_prompt_sha256",
                                             "model_path", "corpus_sha256", "prompt_asset_sha256",
                                             "profile_fingerprint")):
            continue
        key = (row["serial"], row["case_id"], row["case_occurrence"], row["reuse"],
               row[other], row["model_path"], row["model_bytes"],
               row["corpus_sha256"], row["prompt_asset_sha256"], row["profile_fingerprint"],
               row["source_text_sha256"], row["rendered_prompt_sha256"])
        groups[key][row[varied]].append(row)
    pairs = []
    for group in groups.values():
        off = sorted(group[False], key=lambda x: x["arm_order"])
        on = sorted(group[True], key=lambda x: x["arm_order"])
        for left, right in zip(off, on):
            same_raw = bool(left["raw_sha256"] and left["raw_sha256"] == right["raw_sha256"])
            same_final = bool(left["final_express_sha256"] and
                              left["final_express_sha256"] == right["final_express_sha256"] and
                              left["final_a2ui_sha256"] and
                              left["final_a2ui_sha256"] == right["final_a2ui_sha256"])
            left_ms, right_ms = num(left["conversion_ms"]), num(right["conversion_ms"])
            delta = right_ms - left_ms if left_ms is not None and right_ms is not None else None
            token_delta = (right["output_tokens"] - left["output_tokens"]
                           if isinstance(right["output_tokens"], int) and isinstance(left["output_tokens"], int)
                           else None)
            pairs.append({
                "comparison": varied, "serial": left["serial"], "device": left["device"],
                "caseId": left["case_id"], "occurrence": left["case_occurrence"],
                "reuse": left["reuse"], other: left[other],
                "offArm": left["arm"], "onArm": right["arm"],
                "offArmOrder": left["arm_order"], "onArmOrder": right["arm_order"],
                "offMs": left_ms, "onMs": right_ms, "onMinusOffMs": rounded(delta),
                "onMinusOffPct": rounded(delta / left_ms * 100) if delta is not None and left_ms else None,
                "outputTokenDelta": token_delta,
                "sourceTextEqual": left["source_text_sha256"] == right["source_text_sha256"],
                "renderedPromptEqual": left["rendered_prompt_sha256"] == right["rendered_prompt_sha256"],
                "rawOutputEqual": same_raw, "finalOutputEqual": same_final,
                "rawAcceptedWithoutRepairBoth": left["raw_accepted_without_repair"] and right["raw_accepted_without_repair"],
                "bothQualified": left["qualified"] and right["qualified"],
                "lengthConfounded": token_delta != 0,
                "interpretation": "matched-output timing" if same_raw and same_final and token_delta == 0 and
                    left["qualified"] and right["qualified"]
                    else "descriptive only; output or conditions differ",
            })
    return sorted(pairs, key=lambda x: (x["serial"], x["comparison"], x["caseId"], x["occurrence"]))


def render_probe_runs(root: Path, devices: dict[str, str | None], warnings: list[str]) -> list[dict[str, Any]]:
    """Read the separate saved-A2UI Activity probe; never mix it into inference phases."""
    runs = []
    for results_path in sorted(root.glob("*/render_timing/device/results.json")):
        device_dir = results_path.parent
        serial = device_dir.parent.parent.name
        records = read_json(results_path, warnings, [])
        if not isinstance(records, list):
            warnings.append(f"Render probe results are not an array: {results_path}")
            continue
        config = read_json(device_dir / "run_config.json", warnings, {}) or {}
        launch = read_json(device_dir / "activity_launch.json", warnings, {}) or {}
        completion = read_json(device_dir / "summary.json", warnings, {}) or {}
        good = [r for r in records if isinstance(r, dict)]
        observed = [r for r in good if r.get("labelObserved") is True and not r.get("error")]
        requested = completion.get("requestedRenders")
        if requested is not None and requested != len(good):
            warnings.append(f"{serial} render probe has {len(good)}/{requested} requested records")
        if completion.get("failedChecks"):
            warnings.append(f"{serial} render probe failed checks: {completion['failedChecks']}")
        cases = []
        for case_id in sorted({str(r.get("caseId")) for r in good if r.get("caseId")}):
            case_rows = [r for r in good if r.get("caseId") == case_id]
            visible_rows = [r for r in case_rows if r.get("labelObserved") is True and not r.get("error")]
            hashes = {r.get("inputSha256") for r in case_rows if r.get("inputSha256")}
            if len(hashes) > 1:
                warnings.append(f"{serial} render probe {case_id}: saved A2UI hashes differ across repeats")
            cases.append({
                "caseId": case_id, "renders": len(case_rows), "observed": len(visible_rows),
                "inputSha256": next(iter(hashes)) if len(hashes) == 1 else None,
                "medianCompilerMs": median([r.get("compilerTimeMs") for r in case_rows]),
                "medianShowDocumentCallbackMs": median([r.get("showDocumentCallbackTimeMs") for r in case_rows]),
                "medianLabelObservationWallMs": median([r.get("labelObservationWallMs") for r in visible_rows]),
            })
        runs.append({
            "serial": serial, "device": devices.get(serial),
            "sourceArm": Path(str(config.get("inputDir") or "")).name or None,
            "resultsPath": str(results_path.relative_to(root)),
            "firstActivityLaunchMs": rounded(launch.get("firstActivityLaunchMs")),
            "processColdGuaranteed": launch.get("processColdGuaranteed") is True,
            "requestedRenders": requested, "failedChecks": completion.get("failedChecks") or [],
            "recordedRenders": len(good), "observedLabels": len(observed),
            "medianCompilerMs": median([r.get("compilerTimeMs") for r in good]),
            "medianShowDocumentCallbackMs": median([r.get("showDocumentCallbackTimeMs") for r in good]),
            "medianActivityDispatchWallMs": median([r.get("activityDispatchWallMs") for r in good]),
            "medianLabelObservationWallMs": median([r.get("labelObservationWallMs") for r in observed]),
            "cases": cases,
        })
    return runs


def fmt(value: Any, digits: int = 1) -> str:
    value = num(value)
    return f"{value:,.{digits}f}" if value is not None else "—"


def ms_s(value: Any) -> str:
    value = num(value)
    return f"{value / 1000:.2f}" if value is not None else "—"


def markdown(summary: dict[str, Any]) -> str:
    arms = summary["arms"]
    rows = summary["requests"]
    pairs = summary["pairs"]
    title = "Interim snapshot" if not summary["finalized"] else "Completed run plan"
    out = ["# Inference bottleneck profile", "", f"**{title}** · {summary['generatedAtUtc']} · "
           f"{len(arms)} discovered arms, {sum(a['state'] == 'complete' for a in arms)} complete, "
           f"{len(rows)} saved measured requests and {summary['excludedWarmupCount']} excluded startup warmups.", "",
           "The comparison uses the trained E2B converter, corrected FP16 model, frozen Bixby50 inputs and native LiteRT metrics. "
           "Arm medians and weighted rates below include **all observed requests with available values**; strict-cooled counts are shown separately. "
           "All time values below are milliseconds unless marked seconds or tokens/s. Device results remain separate.", ""]
    if (Path(summary["root"]) / "latency_breakdown.png").is_file():
        out.extend(["![Latency breakdown by device and arm](latency_breakdown.png)", ""])
    if (Path(summary["root"]) / "mtp_comparison.png").is_file():
        out.extend(["![MTP paired comparison with thermal and output caveats](mtp_comparison.png)", ""])
    out.extend(["## Arm summary", "",
           "| Device | Order | Arm | State | Strict-cooled/observed | All-run convert median s | All-run provider median s | All-run init median s | All-run prefill median s* | All-run decode median s* | All-run native decode tok/s† | Strict-cooled convert median s | Raw accepted/final saved | Thermal / engine flags |",
           "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for arm in arms:
        out.append(f"| {arm['device'] or arm['serial']} | {arm['order']} | `{arm['name']}` | {arm['state']} | "
                   f"{arm['qualifiedRequests']}/{arm['observedRequests']} | {ms_s(arm['median_conversion_ms'])} | "
                   f"{ms_s(arm['median_provider_ms'])} | {ms_s(arm['median_init_ms'])} | "
                   f"{ms_s(arm['median_prefill_ms_derived'])} | {ms_s(arm['median_decode_ms_derived'])} | "
                   f"{fmt(arm['weightedDecodeTokS'])} | {ms_s(arm['qualifiedMedian_conversion_ms'])} | "
                   f"{arm['rawAcceptedWithoutRepair']}/{arm['finalA2uiSaved']} | "
                   f"timeouts {arm['thermalTimeouts']}; engine mismatches {arm['engineStateMismatches']} |")
    out.extend(["", "* Prefill and decode durations are derived from native token counts and native rates; "
                "they are estimates. † Native decode throughput is token weighted, not an average of per-request rates.", ""])
    warmups = [a for a in arms if a["warmupDiagnostic"] is not None]
    if warmups:
        out.extend(["## Excluded startup warmups", "",
                    "These are real cold engine starts before warm reused requests. They are excluded from arm medians and all off/on pairs. "
                    "No per-case cooldown was required before a warmup, so prior arm heat can affect it; thermal status is shown where available.", "",
                    "| Device | Arm | Thermal start→end | Runtime | Engine initialized | Conversion s | Init s | Prefill s* | Decode s* | Output tokens | Native decode tok/s |",
                    "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"])
        for arm in warmups:
            w = arm["warmupDiagnostic"]
            out.append(f"| {arm['device'] or arm['serial']} | `{arm['name']}` | "
                       f"{w['thermalStart']}→{w['thermalEnd']} | {w['runtime'] or '—'} | "
                       f"{'yes' if w['engineInitializedForRequest'] is True else 'no' if w['engineInitializedForRequest'] is False else '?'} | "
                       f"{ms_s(w['conversion_ms'])} | {ms_s(w['init_ms'])} | "
                       f"{ms_s(w['prefill_ms_derived'])} | {ms_s(w['decode_ms_derived'])} | "
                       f"{w['output_tokens'] if w['output_tokens'] is not None else '—'} | "
                       f"{fmt(w['decode_tok_s'])} |")
        out.append("")
    out.extend(["## Request phases", "",
                "| Device | Arm | Case | Engine | Thermal start→end | Init s | Prefill s* | Decode s* | Provider residual s* | SDK outside provider s | Convert s | First partial s | First ready preview s | Output tokens | Flags |",
                "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in rows:
        out.append(f"| {row['device'] or row['serial']} | `{row['arm']}` | {row['case_id']} #{row['case_occurrence']} | "
                   f"{'cold' if row['engine_initialized'] else 'warm' if row['engine_initialized'] is False else '?'} | "
                   f"{row['thermal_start']}→{row['thermal_end']} | {ms_s(row['init_ms'])} | "
                   f"{ms_s(row['prefill_ms_derived'])} | {ms_s(row['decode_ms_derived'])} | "
                   f"{ms_s(row['provider_residual_ms_derived'])} | {ms_s(row['sdk_outside_provider_ms'])} | "
                   f"{ms_s(row['conversion_ms'])} | {ms_s(row['first_partial_callback_ms'])} | "
                   f"{ms_s(row['first_ready_preview_callback_ms'])} | {row['output_tokens'] or '—'} | "
                   f"{', '.join(row['flags']) or 'clear'} |")
    out.extend(["", "## Paired controls", "",
                "Pairs require the same device, case occurrence, reuse mode, model and SDK profile settings, source text and rendered prompt. "
                "Strict matched-output interpretation also requires equal raw and final outputs, equal native output-token count, and strict-cooled conditions on both sides. "
                "Run order is retained.", "",
                "| Device | Comparison | Case | Off → on arm order | Off s | On s | Δ s | Output tokens Δ | Raw text equal | Final output equal | Quality |",
                "|---|---|---|---|---:|---:|---:|---:|---|---|---|"])
    for pair in pairs:
        out.append(f"| {pair['device'] or pair['serial']} | {pair['comparison']} | {pair['caseId']} #{pair['occurrence']} | "
                   f"{pair['offArmOrder']}→{pair['onArmOrder']} | {ms_s(pair['offMs'])} | {ms_s(pair['onMs'])} | "
                   f"{ms_s(pair['onMinusOffMs'])} | {pair['outputTokenDelta']} | "
                   f"{'yes' if pair['rawOutputEqual'] else 'no'} | "
                   f"{'yes' if pair['finalOutputEqual'] else 'no'} | {pair['interpretation']} |")
    if not pairs:
        out.append("| — | — | — | — | — | — | — | — | — | — | Matching arms have not arrived. |")
    out.extend(["", "## Teardown and MTP acceptance", "",
                "| Device | Arm | Metrics finish median ms | Close median ms | MTP acceptance | Interpretation |",
                "|---|---|---:|---:|---|---|"])
    for arm in arms:
        acceptance = arm["mtpAcceptanceArm"]
        if acceptance is not None:
            acc_text = f"{acceptance * 100:.2f}%"
            note = "one engine close covers the warm arm"
        elif arm["mtpAcceptanceEngineSessions"]:
            acc_text = ", ".join(f"{v * 100:.2f}%" for v in arm["mtpAcceptanceEngineSessions"])
            note = "separate cold engine sessions; no token-weighted aggregate available"
        else:
            acc_text = "—"
            note = "no acceptance value"
        out.append(f"| {arm['device'] or arm['serial']} | `{arm['name']}` | "
                   f"{fmt(arm['finishMetricsMedianMs'])} | {fmt(arm['closeMedianMs'])} | "
                   f"{acc_text} | {note} |")
    if summary["renderTiming"]:
        out.extend(["", "## Saved-document display probe", "",
                    "This separate replay compiles saved model-generated `final.a2ui.json` and calls `showDocument`; no model generation runs. "
                    "The status-label observation is an **upper bound on accessibility-visible state under UiAutomator polling**, "
                    "not a measured pixel-present or paint time. First Activity launch occurs within instrumentation and is not guaranteed process-cold.", "",
                    "| Device | Source arm | Label observed/renders | First Activity launch ms | Compiler median ms | `showDocument` callback median ms | Dispatch median ms | Status observation median ms |",
                    "|---|---|---:|---:|---:|---:|---:|---:|"])
        for run in summary["renderTiming"]:
            out.append(f"| {run['device'] or run['serial']} | `{run['sourceArm'] or 'unknown'}` | "
                       f"{run['observedLabels']}/{run['recordedRenders']} | "
                       f"{fmt(run['firstActivityLaunchMs'])} | {fmt(run['medianCompilerMs'])} | "
                       f"{fmt(run['medianShowDocumentCallbackMs'])} | "
                       f"{fmt(run['medianActivityDispatchWallMs'])} | "
                       f"{fmt(run['medianLabelObservationWallMs'])} |")
        out.extend(["", "Compiler, callback and dispatch medians use recorded attempts; status-observation medians use only passes where the unique label was observed. "
                    "Failed checks are listed in the evidence warnings. Case-level timings and input hashes are in `summary.json` under `renderTiming`. "
                    "Screenshots are captured after each case's measured interval.", ""])
    out.extend(["", "Native `nativeInitializationPhaseSeconds` can overlap other phases and is never added to the phase breakdown. "
                "Prefill/decode and provider residual are estimates; separate stage medians need not add to the conversion median. "
                "The first partial callback is a host observation; native time to first token is recorded separately in `per_request.csv`. "
                "First preview is an SDK callback, not a painted frame. `callbackBodyWorkMs` measures only the probe observer body. "
                "No screenshot or UI readback runs inside the measured interval. Provider residual and SDK outside-provider time are boundary differences, not isolated CPU profiling.", "",
                "Strict-cooled comparisons require thermal status 0 at both request start and end, plus a successful cooldown and valid engine/phase evidence. "
                "All-observed arm statistics are descriptive even when the strict-cooled subset is empty. End status, cooldown timeout, sampled GPU clocks, and all warnings remain in `per_request.csv` and `summary.json`. "
                "A changed output hash or token length can change full-answer latency even when native decode rate is unchanged. "
                "The run configuration records model path and byte size, but not a per-arm model-content hash; paired comparisons assume the file at that path was unchanged.", ""])
    docs = [("provenance.json", "Run provenance"), ("NATIVE_ANALYSIS.md", "Native analysis"),
            ("CODE_AUDIT.md", "Code audit"), ("RESOURCE_ANALYSIS.md", "Resource analysis"),
            ("FINDINGS.md", "Findings")]
    available = [(name, label) for name, label in docs if (Path(summary["root"]) / name).is_file()]
    if available:
        out.extend(["## Related evidence", ""])
        out.extend(f"- [{label}]({name})" for name, label in available)
        out.append("")
    if summary["warnings"]:
        out.extend(["## Evidence warnings", ""])
        out.extend(f"- {warning}" for warning in summary["warnings"])
        out.append("")
    return "\n".join(out)


def html_report(summary: dict[str, Any]) -> str:
    arms = summary["arms"]
    rows = summary["requests"]
    pairs = summary["pairs"]
    esc = lambda value: html.escape(str(value))
    palette = [("init_ms", "#5b6de8", "Engine init"),
               ("prefill_ms_derived", "#45a8d4", "Prefill"),
               ("decode_ms_derived", "#36b985", "Decode"),
               ("provider_residual_ms_derived", "#e7a345", "Provider residual"),
               ("sdk_outside_provider_ms", "#a88edc", "SDK outside provider")]
    max_ms = max((sum(max(0, num(a.get(f"median_{key}")) or 0) for key, _, _ in palette)
                  for a in arms), default=1) or 1
    chart = []
    for arm in arms:
        total = num(arm["median_conversion_ms"]) or 0
        segments = []
        for key, color, label in palette:
            value = max(0, num(arm.get(f"median_{key}")) or 0)
            if value:
                segments.append(f'<span title="{esc(label)} {value / 1000:.2f}s" style="width:{value / max_ms * 100:.2f}%;background:{color}"></span>')
        chart.append(f'<div class="barrow"><div class="barlabel">{esc(arm["device"] or arm["serial"])} · {esc(arm["name"])}</div>'
                     f'<div class="bartrack">{"".join(segments)}</div><strong>{total / 1000:.2f}s</strong></div>')
    legend = "".join(f'<span><i style="background:{color}"></i>{esc(label)}</span>' for _, color, label in palette)
    def table(headers: list[str], data: list[list[Any]]) -> str:
        head = "".join(f"<th>{esc(v)}</th>" for v in headers)
        body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in values) + "</tr>" for values in data)
        return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    arm_table = table(["Device", "Order", "Arm", "State", "Strict-cooled / observed", "All-run median conversion", "All-run native decode", "Strict-cooled median conversion", "Raw/final", "Flags"], [
        [a["device"] or a["serial"], a["order"], a["name"], a["state"],
         f'{a["qualifiedRequests"]}/{a["observedRequests"]}', f'{ms_s(a["median_conversion_ms"])} s',
         f'{fmt(a["weightedDecodeTokS"])} tok/s', f'{ms_s(a["qualifiedMedian_conversion_ms"])} s',
         f'{a["rawAcceptedWithoutRepair"]}/{a["finalA2uiSaved"]}',
         f'thermal {a["thermalTimeouts"]}; reuse {a["engineStateMismatches"]}'] for a in arms])
    warmups = [(a, a["warmupDiagnostic"]) for a in arms if a["warmupDiagnostic"] is not None]
    warmup_table = table(["Device", "Arm", "Thermal start→end", "Runtime", "Initialized", "Conversion", "Init", "Prefill*", "Decode*", "Output tokens", "Decode tok/s"], [
        [a["device"] or a["serial"], a["name"], f'{w["thermalStart"]}→{w["thermalEnd"]}',
         w["runtime"] or "—", "yes" if w["engineInitializedForRequest"] is True else
         "no" if w["engineInitializedForRequest"] is False else "?",
         *(f'{ms_s(w[key])}s' for key in ("conversion_ms", "init_ms", "prefill_ms_derived", "decode_ms_derived")),
         w["output_tokens"] if w["output_tokens"] is not None else "—", fmt(w["decode_tok_s"])]
        for a, w in warmups])
    warmup_section = ("<section><h2>Excluded startup warmups</h2><p>Each reused engine starts with a real cold request, "
                      "reported here separately from measured requests and off/on pairs. No per-case cooldown was required before warmup; prior arm heat can affect it.</p>"
                      + warmup_table + "</section>") if warmups else ""
    request_table = table(["Device", "Arm", "Case", "Engine", "Thermal", "Init", "Prefill*", "Decode*", "Residual*", "SDK", "Total", "Preview", "Tokens", "Flags"], [
        [r["device"] or r["serial"], r["arm"], f'{r["case_id"]} #{r["case_occurrence"]}',
         "cold" if r["engine_initialized"] else "warm" if r["engine_initialized"] is False else "?",
         f'{r["thermal_start"]}→{r["thermal_end"]}', *(f'{ms_s(r[key])}s' for key in (
             "init_ms", "prefill_ms_derived", "decode_ms_derived", "provider_residual_ms_derived",
             "sdk_outside_provider_ms", "conversion_ms", "first_ready_preview_callback_ms")),
         r["output_tokens"] or "—", ", ".join(r["flags"]) or "clear"] for r in rows])
    pair_table = table(["Device", "Control", "Case", "Order", "Off", "On", "On − off", "Token Δ", "Same raw text", "Same final output", "Interpretation"], [
        [p["device"] or p["serial"], p["comparison"], f'{p["caseId"]} #{p["occurrence"]}',
         f'{p["offArmOrder"]}→{p["onArmOrder"]}', f'{ms_s(p["offMs"])}s', f'{ms_s(p["onMs"])}s',
         f'{ms_s(p["onMinusOffMs"])}s', p["outputTokenDelta"],
         "yes" if p["rawOutputEqual"] else "no", "yes" if p["finalOutputEqual"] else "no",
         p["interpretation"]] for p in pairs])
    teardown_table = table(["Device", "Arm", "Finish metrics", "Close", "MTP acceptance", "Scope"], [
        [a["device"] or a["serial"], a["name"], f'{fmt(a["finishMetricsMedianMs"])} ms',
         f'{fmt(a["closeMedianMs"])} ms',
         f'{a["mtpAcceptanceArm"] * 100:.2f}%' if a["mtpAcceptanceArm"] is not None else
         ", ".join(f'{v * 100:.2f}%' for v in a["mtpAcceptanceEngineSessions"]) or "—",
         "one shared session" if a["mtpAcceptanceArm"] is not None else "separate engine sessions"] for a in arms])
    render_table = table(["Device", "Source arm", "Label observed / renders", "First Activity launch", "Compiler median", "showDocument callback median", "Dispatch median", "Status observation median"], [
        [r["device"] or r["serial"], r["sourceArm"] or "unknown",
         f'{r["observedLabels"]}/{r["recordedRenders"]}', f'{fmt(r["firstActivityLaunchMs"])} ms',
         f'{fmt(r["medianCompilerMs"])} ms', f'{fmt(r["medianShowDocumentCallbackMs"])} ms',
         f'{fmt(r["medianActivityDispatchWallMs"])} ms',
         f'{fmt(r["medianLabelObservationWallMs"])} ms'] for r in summary["renderTiming"]])
    render_section = ("<section><h2>Saved-document display probe</h2><p>Saved model-generated A2UI is compiled and shown without live model inference. "
                      "The unique status-label observation is an upper bound on accessibility-visible state under UiAutomator polling, not a pixel-present or paint timestamp. "
                      "First Activity launch is within instrumentation and is not guaranteed process-cold. Compiler/callback/dispatch medians include recorded attempts; "
                      "status-observation medians include only observed passes. Failed checks appear below. Screenshots are taken after the measured interval.</p>"
                      + render_table + "</section>") if summary["renderTiming"] else ""
    warning_html = "".join(f"<li>{esc(w)}</li>" for w in summary["warnings"])
    phase = "Interim evidence snapshot" if not summary["finalized"] else "Completed run plan"
    chart_files = [("latency_breakdown.png", "Latency breakdown by device and arm"),
                   ("mtp_comparison.png", "MTP paired comparison")]
    available_charts = [(name, label) for name, label in chart_files
                        if (Path(summary["root"]) / name).is_file()]
    figure_section = ("<section><h2>Evidence charts</h2><p>Derived phase estimates, separate teardown, "
                      "and thermal/output confounds are labeled in the figures.</p>" +
                      "".join(f'<figure><img src="{name}" alt="{esc(label)}"><figcaption>{esc(label)}</figcaption></figure>'
                              for name, label in available_charts) + "</section>") if available_charts else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GenUICraft inference bottleneck profile</title><style>
:root{{--ink:#18253b;--muted:#65748b;--line:#dfe5ee;--paper:#fff;--wash:#f5f7fb;--accent:#4057c8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1440px;margin:auto;padding:32px 28px 70px}}header,section{{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:24px;margin:0 0 18px;box-shadow:0 3px 14px #22314d08}}
h1{{font-size:30px;line-height:1.15;margin:0 0 12px}}h2{{font-size:19px;margin:0 0 15px}}p{{margin:10px 0;color:var(--muted)}}
.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:750;color:var(--accent)}}.badge{{display:inline-block;background:#fff3df;color:#92591a;border:1px solid #f1d4a0;border-radius:999px;padding:4px 12px;font-weight:700}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0 0}}.stat{{border:1px solid var(--line);background:#fafbfe;border-radius:12px;padding:14px}}.stat strong{{display:block;font-size:25px}}.stat span{{color:var(--muted);font-size:12px}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12.5px}}th,td{{padding:10px 9px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}}th{{position:sticky;top:0;background:#f7f9fc;color:#506078;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}tbody tr:hover{{background:#f8faff}}
.barrow{{display:grid;grid-template-columns:240px 1fr 65px;align-items:center;gap:12px;margin:12px 0}}.barlabel{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px}}.bartrack{{height:19px;background:#eef1f6;border-radius:5px;display:flex;overflow:hidden}}.bartrack span{{height:100%}}.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0 0;color:var(--muted);font-size:12px}}.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}.note{{background:#f5f7fc;border-left:3px solid #8da0e9;padding:10px 14px;border-radius:4px;color:#40506a}}ul{{color:var(--muted)}}
figure{{margin:16px 0 24px}}figure img{{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:10px}}figcaption{{color:var(--muted);font-size:12px;margin-top:7px}}
@media(max-width:700px){{main{{padding:15px}}header,section{{padding:17px}}.barrow{{grid-template-columns:100px 1fr 48px}}}}
</style></head><body><main><header><div class="eyebrow">GenUICraft · Native inference</div><h1>Where conversion time goes</h1>
<span class="badge">{esc(phase)}</span><p>Generated {esc(summary['generatedAtUtc'])}. Measurements stay within each device and exact frozen case. Arms are marked complete only when device and host receipts agree.</p>
<div class="stats"><div class="stat"><strong>{len(arms)}</strong><span>arms discovered</span></div><div class="stat"><strong>{sum(a['state']=='complete' for a in arms)}</strong><span>arms complete</span></div><div class="stat"><strong>{len(rows)}</strong><span>measured requests saved</span></div><div class="stat"><strong>{summary['excludedWarmupCount']}</strong><span>excluded startup warmups</span></div><div class="stat"><strong>{sum(r['qualified'] for r in rows)}</strong><span>strict-cooled requests</span></div></div></header>
{figure_section}
<section><h2>Stage profile</h2><p>All-observed request medians in each arm, including thermally flagged cases. Strict-cooled counts appear in the table. Component durations are approximate; use them to locate large costs.</p>{''.join(chart) or '<p>No complete measurements yet.</p>'}<div class="legend">{legend}</div></section>
<section><h2>Arms and order</h2>{arm_table}</section><section><h2>Every measured request</h2>{request_table}</section>
{warmup_section}
<section><h2>Matched controls</h2><p>Off/on comparisons require the same device, case occurrence, model and SDK profile, source and rendered prompt. Strict matched-output interpretation also requires equal raw/final output, equal token count, and strict-cooled conditions.</p>{pair_table}</section>
<section><h2>Metrics finalization and teardown</h2>{teardown_table}</section>
{render_section}
<section><h2>How to read these timings</h2><p class="note">All-observed arm medians and weighted decode rates are descriptive. Strict-cooled comparisons require thermal status 0 at both ends, successful cooldown, and valid engine/phase evidence. Native initialization phase is non-additive. Prefill and decode durations are derived from token counts and native rates; separate stage medians need not add to the conversion median. Provider residual and SDK outside-provider are boundary differences. First preview means SDK callback, not screen paint; callback work is only the probe observer body. MTP acceptance is reported at engine close, once per warm arm or separately for cold sessions. Paired runs match model path and byte size; no per-arm model-content hash is recorded.</p><p>Thermal status at both ends, cooldown timeouts, sampled GPU clocks, output hashes, and exact native metrics are in <a href="per_request.csv">per_request.csv</a> and <a href="summary.json">summary.json</a>.</p><p>{' · '.join(f'<a href="{name}">{esc(label)}</a>' for name, label in [("provenance.json", "Run provenance"), ("NATIVE_ANALYSIS.md", "Native analysis"), ("CODE_AUDIT.md", "Code audit"), ("RESOURCE_ANALYSIS.md", "Resource analysis"), ("FINDINGS.md", "Findings")] if (Path(summary["root"]) / name).is_file())}</p></section>
<section><h2>Evidence warnings</h2>{('<ul>'+warning_html+'</ul>') if warning_html else '<p>No parser or completeness warnings in the arms currently discovered.</p>'}</section>
</main></body></html>"""


def write_atomic(path: Path, content: str) -> None:
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(content, encoding="utf-8")
    pending.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Host evidence directory")
    parser.add_argument("--final", action="store_true", help="Label final only after all explicitly expected device arms are complete")
    parser.add_argument("--expected-arm", action="append", default=[], metavar="SERIAL/ARM",
                        help="Required with --final; repeat for every planned device arm")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Missing evidence root: {root}")
    arm_dirs = sorted(root.glob("*/runs/*"), key=lambda p: (p.parent.parent.name, p.name))
    arm_dirs = [p for p in arm_dirs if p.is_dir()]
    if not arm_dirs:
        parser.error("No device arm directories under */runs/*")
    warnings: list[str] = []
    arms, rows = [], []
    for serial in sorted({p.parent.parent.name for p in arm_dirs}):
        own = [p for p in arm_dirs if p.parent.parent.name == serial]
        own.sort(key=lambda p: (num((read_json(p / "host_result.json", warnings, {}) or {}).get("startedAtEpochMs")) or p.stat().st_mtime * 1000, p.name))
        for order, arm_dir in enumerate(own, 1):
            arm, requests = load_arm(arm_dir, order, warnings)
            arms.append(arm)
            rows.extend(requests)
    pairs = pairing(rows, "mtp") + pairing(rows, "streaming")
    observed_names = {f"{a['serial']}/{a['name']}" for a in arms}
    expected_names = set(args.expected_arm)
    if args.final:
        if not expected_names:
            parser.error("--final requires --expected-arm SERIAL/ARM for every planned arm")
        missing = sorted(expected_names - observed_names)
        if missing:
            parser.error(f"--final refused: missing planned arms: {', '.join(missing)}")
        unlisted = sorted(observed_names - expected_names)
        if unlisted:
            parser.error(f"--final refused: discovered arms not listed as expected: {', '.join(unlisted)}")
        if any(a["state"] != "complete" for a in arms):
            details = "; ".join(f"{a['serial']}/{a['name']}: {','.join(a['completenessGaps']) or a['state']}"
                                for a in arms if a["state"] != "complete")
            parser.error(f"--final refused: incomplete device/host evidence: {details}")
    if len({a["serial"] for a in arms}) < 2:
        warnings.append("Only one device has discoverable arms; cross-device coverage is incomplete.")
    for arm in arms:
        if arm["state"] != "complete":
            gaps = ", ".join(arm["completenessGaps"])
            warnings.append(f"{arm['serial']}/{arm['name']} is {arm['state']}; its results are provisional"
                            + (f" ({gaps})." if gaps else "."))
        if arm["engineStateMismatches"]:
            warnings.append(f"{arm['serial']}/{arm['name']}: {arm['engineStateMismatches']} requests did not show the expected native cold/warm state.")
        if arm["mtpStateMismatch"]:
            warnings.append(f"{arm['serial']}/{arm['name']}: native MTP state at engine close differs from the requested arm.")
        if arm["thermalTimeouts"]:
            warnings.append(f"{arm['serial']}/{arm['name']}: {arm['thermalTimeouts']} thermal cooldowns timed out.")
    device_names = {a["serial"]: a["device"] for a in arms if a["device"]}
    render_runs = render_probe_runs(root, device_names, warnings)
    summary = {
        "schemaVersion": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root), "finalized": bool(args.final),
        "expectedArms": sorted(expected_names),
        "discoveredArmCount": len(arms), "completeArmCount": sum(a["state"] == "complete" for a in arms),
        "requestCount": len(rows), "qualifiedRequestCount": sum(r["qualified"] for r in rows),
        "excludedWarmupCount": sum(a["warmupDiagnostic"] is not None for a in arms),
        "devices": sorted({a["serial"] for a in arms}),
        "arms": arms, "requests": rows, "pairs": pairs, "renderTiming": render_runs,
        "warnings": list(dict.fromkeys(warnings)),
        "method": {
            "phaseBreakdown": "init wall + input tokens/native prefill rate + output tokens/native decode rate + provider residual + SDK outside provider",
            "nativeInitializationPhase": "separate and non-additive",
            "callbackWork": "probe observer body only",
            "preview": "SDK callback publication; not screen paint",
            "warmup": "one startup request per reuse arm, reported separately and excluded from measured-request medians and pairs; no case cooldown is required before warmup",
            "mtpAcceptance": "engine-close session measurement; one warm arm value or separate cold sessions",
            "pairing": "same device, case occurrence, reuse, other control, model path/size, corpus, prompt, profile settings, source text and rendered prompt; strict interpretation also requires equal raw/final output and native output token count",
            "renderTiming": "saved A2UI compiler and Activity handoff measured separately; UiAutomator status-label observation is an accessibility polling upper bound, not pixel paint",
        },
    }
    write_atomic(root / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    csv_path = root / "per_request.csv"
    pending = csv_path.with_name(csv_path.name + ".tmp")
    with pending.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "; ".join(value) if isinstance(value, list) else value
                             for key, value in row.items()})
    pending.replace(csv_path)
    write_atomic(root / "REPORT.md", markdown(summary).rstrip() + "\n")
    write_atomic(root / "index.html", html_report(summary))
    print(f"{len(arms)} arms ({summary['completeArmCount']} complete), {len(rows)} requests, "
          f"{len(pairs)} pairs; wrote {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
