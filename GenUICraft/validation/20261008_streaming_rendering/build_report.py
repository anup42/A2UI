"""Build the local validation report from archived evidence; Python standard library only.

Run after pulling the native run. Missing inputs remain explicitly pending/incomplete.
This script performs no network, inference, builds, or device operations.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
from pathlib import Path
from urllib.parse import quote

BASE = Path(__file__).resolve().parent
DEFAULT_NATIVE = BASE / "native" / "fold7_fp16_mtp_20261008"
DEFAULT_CASES = ["BXP-001", "BXP-003", "BXP-004", "BXP-008", "BXP-011"]


def read_json(path: Path, default, issues: list[str]):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        issues.append(f"Could not read {path.name}: {error}")
        return default


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def seconds(value):
    value = number(value)
    return "—" if value is None else f"{value / 1000:.3f}"


def rate(value):
    value = number(value)
    return "—" if value is None else f"{value:.2f}"


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def relative(path: Path):
    return path.resolve().relative_to(BASE).as_posix()


def link(path: Path, label: str):
    return f'<a href="{quote(relative(path), safe="/")}">{html.escape(label)}</a>'


def cold_label(metrics: dict):
    flags = [item.get("engineInitializedForRequest") for item in metrics.get("attempts", []) if isinstance(item, dict)]
    init = number(metrics.get("nativeEngineInitializationNanos"))
    if any(flag is True for flag in flags) or (init is not None and init > 0):
        return "cold init"
    if (flags and all(flag is False for flag in flags)) or init == 0:
        return "warm reuse"
    return "unreported"


def arm_data(item: dict, native: Path, issues: list[str]):
    case = item.get("fixtureId", "")
    enabled = item.get("streamingEnabled")
    if not re.fullmatch(r"BXP-\d{3}", case) or not isinstance(enabled, bool):
        issues.append("Ignored native record with invalid fixtureId or streamingEnabled.")
        return None
    directory = native / f'{case}_{"on" if enabled else "off"}'
    source = read_json(directory / "source.json", {}, issues)
    metrics = item.get("metrics") or {}
    raw = {path.name: sha(path) for path in sorted(directory.glob("attempt_*_raw.express"))}
    final_hash = sha(directory / "final.express")
    reported_hash = item.get("finalExpressSha256")
    if final_hash and reported_hash and final_hash != reported_hash:
        issues.append(f"{case} {'on' if enabled else 'off'} final file hash differs from results.json.")
    return {
        **item, "scenario": str(source.get("domain", "")).replace("|", "/").replace("\n", " "),
        "artifactDirectory": relative(directory), "rawHashes": raw,
        "verifiedFinalHash": final_hash, "coldState": cold_label(metrics),
        "nativeInitMs": number(metrics.get("nativeEngineInitializationNanos")) / 1e6
            if number(metrics.get("nativeEngineInitializationNanos")) is not None else None,
        "decodeTokensPerSecond": number(metrics.get("nativeDecodeTokensPerSecond")),
        "outputTokens": number(metrics.get("outputTokens")),
    }


def median(values):
    values = [value for value in values if number(value) is not None]
    return statistics.median(values) if values else None


def build(native: Path):
    native = native.resolve()
    native.relative_to(BASE)  # Keep all linked/read artifact paths inside this report directory.
    issues: list[str] = []
    verification = read_json(BASE / "verification.json", {}, issues)
    replay = read_json(BASE / "bixby50_replay.json", {}, issues)
    diagnosis = read_json(BASE / "bixby50_no_preview_diagnosis.json", {}, issues)
    config = read_json(native / "run_config.json", {}, issues)
    records = read_json(native / "results.json", [], issues)
    warmup = read_json(native / "warmup_result.json", {}, issues)
    completion = read_json(native / "summary.json", {}, issues)
    if not isinstance(records, list):
        issues.append("results.json is not a list.")
        records = []
    expected = config.get("cases", DEFAULT_CASES)
    if not isinstance(expected, list) or not expected or any(not isinstance(case, str) or not re.fullmatch(r"BXP-\d{3}", case) for case in expected):
        issues.append("run_config.json has invalid cases; using the requested five-case set.")
        expected = DEFAULT_CASES
    arms: dict[tuple[str, bool], dict] = {}
    for item in records:
        if not isinstance(item, dict):
            issues.append("Ignored non-object native record.")
            continue
        arm = arm_data(item, native, issues)
        if arm is not None:
            key = (arm["fixtureId"], arm["streamingEnabled"])
            if key in arms:
                issues.append(f"Duplicate native arm: {key}.")
            arms[key] = arm
    pairs = []
    for case in expected:
        on, off = arms.get((case, True)), arms.get((case, False))
        pair = {"case": case, "on": on, "off": off, "rawEqual": None, "finalEqual": None,
                "wallDeltaMs": None, "wallDeltaPercent": None, "earlyUiLeadMs": None}
        if on and off:
            if on["rawHashes"] and off["rawHashes"]:
                pair["rawEqual"] = on["rawHashes"] == off["rawHashes"]
            if on["verifiedFinalHash"] and off["verifiedFinalHash"]:
                pair["finalEqual"] = on["verifiedFinalHash"] == off["verifiedFinalHash"]
            wall_on, wall_off = number(on.get("sessionWallElapsedMs")), number(off.get("sessionWallElapsedMs"))
            if wall_on is not None and wall_off is not None:
                pair["wallDeltaMs"] = wall_on - wall_off
                if wall_off > 0:
                    pair["wallDeltaPercent"] = 100 * (wall_on - wall_off) / wall_off
            frame = number(on.get("firstPreviewFrameElapsedMs"))
            if frame is not None and wall_on is not None:
                pair["earlyUiLeadMs"] = wall_on - frame
        pairs.append(pair)
    paired = [pair for pair in pairs if pair["on"] and pair["off"]]
    expected_keys = {(case, enabled) for case in expected for enabled in (True, False)}
    terminal = {"COMPLETE", "FAILED", "CANCELLED"}
    run_complete = bool(config and completion and expected_keys.issubset(arms) and
        all(arms[key].get("phase") in terminal for key in expected_keys))
    status = "complete" if run_complete else "partial" if arms else "pending"
    enabled = [item for item in arms.values() if item["streamingEnabled"]]
    off = [item for item in arms.values() if not item["streamingEnabled"]]
    identical = [pair for pair in paired if pair["rawEqual"] is True and pair["finalEqual"] is True]
    log_paths = sorted(set(native.glob("*.log")) | set(BASE.glob("*native*.log")))
    log_markers = []
    runtime_lines = []
    for path in log_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            runtime_lines.append(line)
            if re.search(r"MTP|drafter|acceptance|finishGenerationMetrics|cleanup|destroy|release", line, re.I):
                log_markers.append({"source": relative(path), "line": line[:1000]})
    summary = {
        "schemaVersion": 1, "verification": verification, "replay": {
            "caseCount": replay.get("case_count"), "earlyPreviewCases": replay.get("cases_with_preview_before_document_completion"),
            "finalRepairOnlyCases": diagnosis.get("no_preview_terminal_generated_dsl_repair_accepted"),
            "terminalRejectedCases": diagnosis.get("no_preview_terminal_rejected"),
        }, "native": {
            "status": status, "runComplete": run_complete, "runDirectory": relative(native),
            "configuration": config, "warmup": warmup, "pairs": pairs,
            "pairedCases": len(paired), "expectedPairs": len(expected), "observedArms": len(arms),
            "successfulConversions": sum(item.get("phase") == "COMPLETE" for item in arms.values()),
            "enabledCasesWithFrameBoundary": sum(number(item.get("firstPreviewFrameElapsedMs")) is not None for item in enabled),
            "offCasesWithNoPreview": sum(item.get("previewSnapshotCount") == 0 and not item.get("observedNativeRevisions") for item in off),
            "enabledFinalSnapshotsMatchDocument": sum(item.get("finalSnapshotMatchesDocument") is True for item in enabled),
            "rawEqualPairs": sum(pair["rawEqual"] is True for pair in paired),
            "finalEqualPairs": sum(pair["finalEqual"] is True for pair in paired),
            "identicalRawAndFinalPairs": len(identical),
            "allArmsColdInitialized": bool(arms) and all(item["coldState"] == "cold init" for item in arms.values()),
            "medianFrameBoundaryMs": median([item.get("firstPreviewFrameElapsedMs") for item in enabled]),
            "medianEarlyUiLeadMs": median([pair["earlyUiLeadMs"] for pair in paired]),
            "medianWallOnMs": median([item.get("sessionWallElapsedMs") for item in enabled]),
            "medianWallOffMs": median([item.get("sessionWallElapsedMs") for item in off]),
            "medianPairedWallDeltaPercent": median([pair["wallDeltaPercent"] for pair in paired]),
            "medianIdenticalOutputWallDeltaPercent": median([pair["wallDeltaPercent"] for pair in identical]),
            "medianDecodeRateOn": median([item.get("decodeTokensPerSecond") for item in enabled]),
            "medianDecodeRateOff": median([item.get("decodeTokensPerSecond") for item in off]),
            "nativeRuntimeLogFiles": [relative(path) for path in log_paths],
            "nativeRuntimeLogMarkerCount": len(log_markers), "nativeRuntimeLogMarkers": log_markers[-30:],
            "mtpRuntimeLogLines": sum("MTP=true" in line for line in runtime_lines),
            "cleanupLogLines": sum(bool(re.search(r"cleanup|destroy|finishGenerationMetrics|\bclose\b|\breleased\b", line, re.I)) for line in runtime_lines),
            "maximumQdqRejected": max((int(match.group(1)) for line in runtime_lines
                if (match := re.search(r"GenUICraftFp16.*rejected=(\d+)", line))), default=None),
        }, "issues": issues,
        "timingDefinitions": {
            "firstPreviewElapsedMs": "Session conversion start to compiled snapshot publication.",
            "firstPreviewFrameElapsedMs": "First Compose frame boundary after preview composition; not verified draw or physical presentation.",
            "conversionElapsedMs": "Converter-reported inference/prompt/validation/repair interval, separate from total session wall.",
            "sessionWallElapsedMs": "Observed completed state before post-run screenshot capture; includes runtime initialization and metrics cleanup, sampled every 40 ms.",
        },
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_views(summary, native)
    print(json.dumps({key: summary["native"][key] for key in ("status", "pairedCases", "successfulConversions", "rawEqualPairs", "finalEqualPairs", "medianFrameBoundaryMs", "medianEarlyUiLeadMs", "medianPairedWallDeltaPercent")}, indent=2))


def write_views(summary: dict, native: Path):
    native_summary = summary["native"]
    replay = summary["replay"]
    verification = summary["verification"]
    bixby = verification.get("bixbySourceChecks", {})
    count = native_summary["pairedCases"]
    header = (f'{native_summary["successfulConversions"]}/{native_summary["observedArms"]} measured conversions completed; '
              f'{native_summary["enabledCasesWithFrameBoundary"]}/{count} streaming cases reached an early Compose frame boundary.') if native_summary["runComplete"] else "Native validation pending or incomplete; no completed-run performance claim."
    lines = ["# Progressive native rendering validation — 2026-10-08", "", header, "",
        f'SDK 0.6.0: **{verification.get("sdkUnitTests", {}).get("tests", "pending")} unit tests passed**; '
        f'**{verification.get("deviceStateTests", {}).get("passed", "pending")} deterministic Fold7 device tests passed**. '
        'The device cases cover early native preview, recreation/final handoff, failure/cancellation clearing, and streaming off. '
        '[Original full SDK build/test log](sdk_build_tests.txt) preserves the full-suite run; the test-count aggregate was captured before targeted replays rewrote generated XML.', "",
        f'Pipeline bridge: **{verification.get("pipelineBridgeTests", {}).get("passed", "pending")} unit tests passed** '
        '([log](pipeline_bridge_tests.txt)). Connected progressive-preview checks exercise the SDK demo; this report does not claim a separate Pipeline/IR early-preview device test.', "",
        f'Bixby source checks: **{bixby.get("javascriptPassed", "pending")} JavaScript / '
        f'{bixby.get("hostPassed", "pending")} host / {bixby.get("managerPassed", "pending")} manager tests passed**. '
        f'The [{bixby.get("archiveEntries", "pending")}-entry changed-file handoff]({bixby.get("archive", "")}) '
        'is source/package evidence; a Bixby application build/device run is not claimed. '
        '[Host tests](bixby_host_tests.txt) · [Manager tests](bixby_manager_tests.txt) · [Delivery manifest](delivery.json).', "",
        f'Saved Bixby50 replay: **{replay["earlyPreviewCases"]}/{replay["caseCount"]} early previews**, '
        f'**{replay["finalRepairOnlyCases"]} accepted only by terminal generated-DSL repair**, '
        f'**{replay["terminalRejectedCases"]} terminal rejects**. Replay measures character readiness rather than native latency.', "",
        "## Native comparison", "",
        f'Artifact status: **{native_summary["status"]}**, {count}/{native_summary["expectedPairs"]} paired cases.', "",
        "| Case | Raw equal | Final equal | First frame boundary on (s) | Preview updates | Session wall on / off (s) | Wall change | Converter on / off (s) | Decode on / off (tok/s) | Init on / off (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    table_rows = []
    for pair in native_summary["pairs"]:
        on, off = pair["on"] or {}, pair["off"] or {}
        parity = lambda value: "yes" if value is True else "NO" if value is False else "pending"
        delta = "—" if pair["wallDeltaPercent"] is None else f'{pair["wallDeltaPercent"]:+.1f}%'
        cells = [pair["case"] + (" · " + on["scenario"] if on.get("scenario") else ""), parity(pair["rawEqual"]), parity(pair["finalEqual"]), seconds(on.get("firstPreviewFrameElapsedMs")),
            str(on.get("previewSnapshotCount", "—")), f'{seconds(on.get("sessionWallElapsedMs"))} / {seconds(off.get("sessionWallElapsedMs"))}',
            delta, f'{seconds(on.get("conversionElapsedMs"))} / {seconds(off.get("conversionElapsedMs"))}',
            f'{rate(on.get("decodeTokensPerSecond"))} / {rate(off.get("decodeTokensPerSecond"))}',
            f'{seconds(on.get("nativeInitMs"))} / {seconds(off.get("nativeInitMs"))}']
        lines.append("| " + " | ".join(cells) + " |")
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
    lines += ["", f'Raw bytes match in **{native_summary["rawEqualPairs"]}/{count} pairs**; final Express bytes match in '
        f'**{native_summary["finalEqualPairs"]}/{count} pairs**. Matching hashes establish byte parity, not source fidelity.', ""]
    if native_summary["runComplete"]:
        lines += [f'Median first frame boundary: **{seconds(native_summary["medianFrameBoundaryMs"])} s**. Median lead before the same '
            f'on-run completed state: **{seconds(native_summary["medianEarlyUiLeadMs"])} s**. Median observed session wall: '
            f'**{seconds(native_summary["medianWallOnMs"])} s on / {seconds(native_summary["medianWallOffMs"])} s off**. '
            f'Median paired wall change: **{rate(native_summary["medianPairedWallDeltaPercent"])}%**.', "",
            f'Identical raw-and-final pairs: **{native_summary["identicalRawAndFinalPairs"]}/{count}**; their median paired wall change is '
            f'**{rate(native_summary["medianIdenticalOutputWallDeltaPercent"])}%**. One observation per arm and changing initialization '
            'times prevent this small run from establishing a general rendering-overhead rate.', "",
            f'Enabled final snapshots matched the returned document in **{native_summary["enabledFinalSnapshotsMatchDocument"]}/{count} cases**. '
            f'Disabled cases emitted no snapshots in **{native_summary["offCasesWithNoPreview"]}/{count} cases**.', ""]
    lines += ["## Timing and evidence boundaries", "",
        "- `firstPreviewFrameElapsedMs` is a Compose frame-boundary observation, not verified drawing or physical screen presentation. First UI may be heading/text rather than a completed card; later revisions add content. The warmup live PNG shows a heading while generation remains active.",
        "- `sessionWallElapsedMs` ends at the observed completed state before final screenshot capture. It includes native initialization and metrics cleanup; observation polling is 40 ms. Converter elapsed is reported separately.",
        "- MTP metrics finalization closes its engine. Initialization flags and durations distinguish cold setup from reuse; a warmup does not make subsequent metrics-on calls warm.",
        "- Measured pairs do not take screenshots during inference. The warmup live capture is excluded from measured pairs.",
        "- A nonempty preview does not establish source fidelity; final compilation/repair and failure status remain authoritative.", "",
        f'All measured arms cold-initialized: **{native_summary["allArmsColdInitialized"]}**. '
        f'Runtime log: **{native_summary["mtpRuntimeLogLines"]} GPU FP16+MTP startup lines**, '
        f'**{native_summary["cleanupLogLines"]} explicit cleanup lines**; maximum Q/DQ rejected counter: '
        f'**{native_summary["maximumQdqRejected"] if native_summary["maximumQdqRejected"] is not None else "unreported"}**. '
        'The filtered log records initialization/policy evidence; it does not isolate cleanup duration.', ""]
    if not native_summary["nativeRuntimeLogFiles"]:
        lines += ["A native runtime log has not been archived in this report directory. Per-arm MTP flags/acceptance and initialization counters are available; cleanup time cannot be isolated from the observed session wall.", ""]
    else:
        lines += ["Runtime evidence: " + ", ".join(f"[{path}]({path})" for path in native_summary["nativeRuntimeLogFiles"]), ""]
    config = native_summary["configuration"]
    lines += [f'Device: **{config.get("device", {}).get("model", "pending")}**; precision: **{config.get("precision", "pending")}**; '
        f'MTP requested: **{config.get("mtpRequested", "pending")}**. Root-verified corrected-model SHA-256: '
        f'`{verification.get("modelSha256", "pending")}`.', "",
        "[Gallery](index.html) · [Summary and full hashes](summary.json) · [SDK verification manifest](verification.json) · "
        "[Deterministic device log](device_state_tests.txt) · [Native device log](device_native_tests.txt) · "
        "[Replay](bixby50_replay.json) · [No-preview diagnosis](bixby50_no_preview_diagnosis.md)", "",
        "Regenerate after archiving evidence with `python build_report.py`. Missing native inputs remain pending. This script does not run models, builds, or device commands.", ""]
    if summary["issues"]:
        lines += ["Input issues: " + "; ".join(summary["issues"]), ""]
    (BASE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    gallery = []
    images = [(native / "warmup" / "live_native.png", "Warmup · live native preview", native_summary["warmup"]),
              (native / "warmup" / "final_native.png", "Warmup · final native UI", native_summary["warmup"])]
    for pair in native_summary["pairs"]:
        for enabled, arm in ((True, pair["on"]), (False, pair["off"])):
            if arm:
                images.append((BASE / arm["artifactDirectory"] / "final_native.png",
                    f'{pair["case"]} · streaming {"on" if enabled else "off"} · final', arm))
    for path, label, arm in images:
        if path.is_file():
            src = quote(relative(path), safe="/")
            gallery.append(f'<figure><a href="{src}"><img src="{src}" alt="{html.escape(label, quote=True)}" loading="lazy"></a>'
                f'<figcaption>{html.escape(label)}<br><small>{html.escape(str(arm.get("phase", "pending")))} · '
                f'wall {seconds(arm.get("sessionWallElapsedMs"))} s</small></figcaption></figure>')
    heads = ["Case", "Raw equal", "Final equal", "First boundary on (s)", "Updates", "Wall on / off (s)", "Wall change", "Converter on / off (s)", "Decode on / off (tok/s)", "Init on / off (s)"]
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GenUICraft 0.6.0 streaming validation</title><style>
    :root{{color-scheme:light dark}}body{{font:16px/1.55 system-ui,sans-serif;margin:0;background:#101724;color:#e7edf5}}main{{max-width:1320px;margin:auto;padding:28px}}h1{{font-size:30px;line-height:1.2}}h2{{font-size:22px;margin-top:30px}}a{{color:#9cc8ff}}.cards{{display:flex;flex-wrap:wrap;gap:14px}}.card{{flex:1;min-width:170px;background:#1c293d;padding:18px;border-radius:12px}}.card b{{font-size:27px;display:block}}.note{{background:#1a2739;padding:16px;border-left:4px solid #73aeed}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;min-width:1150px;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #344155;text-align:left;padding:10px}}th{{background:#1c293d}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px}}figure{{margin:0;background:#1a2739;border-radius:12px;padding:12px}}img{{width:100%;height:520px;object-fit:contain;background:#090e17}}small{{color:#afbdd0}}code{{overflow-wrap:anywhere}}details{{margin-top:16px}}pre{{white-space:pre-wrap;font-size:12px}}@media(max-width:600px){{main{{padding:16px}}img{{height:440px}}}}
    </style></head><body><main><p>GenUICraft SDK 0.6.0 · 8 October 2026</p><h1>Progressive native rendering</h1><p>{html.escape(header)}</p>
    <div class="cards"><div class="card"><b>{verification.get("sdkUnitTests", {}).get("tests", "pending")}</b>SDK unit tests passed</div><div class="card"><b>{verification.get("deviceStateTests", {}).get("passed", "pending")}</b>deterministic device tests passed</div><div class="card"><b>{replay["earlyPreviewCases"]}/{replay["caseCount"]}</b>saved-output early previews</div><div class="card"><b>{html.escape(native_summary["status"])}</b>native artifact coverage · {count}/{native_summary["expectedPairs"]} pairs</div></div>
    <p>Replay remainder: {replay["finalRepairOnlyCases"]} accepted only after terminal repair; {replay["terminalRejectedCases"]} rejected. Replay measures character readiness.</p>
    <p>{link(BASE / "REPORT.md", "Report")} · {link(BASE / "summary.json", "Summary / full hashes")} · {link(BASE / "sdk_build_tests.txt", "Original full SDK log")} · {link(BASE / "pipeline_bridge_tests.txt", "Bridge 9-test log")} · {link(BASE / "device_state_tests.txt", "State-test log")} · {link(BASE / "device_native_tests.txt", "Native-test log")} · {link(BASE / "bixby50_no_preview_diagnosis.md", "Replay diagnosis")}</p>
    <p>Bixby source checks: {bixby.get("javascriptPassed", "pending")} JavaScript / {bixby.get("hostPassed", "pending")} host / {bixby.get("managerPassed", "pending")} manager passed. <a href="../../artifacts/Bixby_GenUICraft_0.6.0_Streaming_ChangedFiles.zip">Changed-file handoff ({bixby.get("archiveEntries", "pending")} entries)</a> · {link(BASE / "bixby_host_tests.txt", "Host tests")} · {link(BASE / "bixby_manager_tests.txt", "Manager tests")} · {link(BASE / "delivery.json", "Delivery manifest")}. Source/package evidence; no Bixby app build/device claim.</p>
    <h2>Five paired native cases</h2><div class="scroll"><table><thead><tr>{"".join(f"<th>{html.escape(head)}</th>" for head in heads)}</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
    <p>Raw matches: {native_summary["rawEqualPairs"]}/{count}; final Express matches: {native_summary["finalEqualPairs"]}/{count}. Matching hashes establish byte parity, not source fidelity.</p>
    <div class="note"><strong>Timing interpretation</strong><br>First boundary is the next Compose frame boundary after preview composition, not verified draw or physical presentation. First UI can be heading/text rather than a completed card; the live warmup image shows a heading while generation remains active. Later revisions add content. Session wall includes initialization and MTP metrics cleanup and ends at observed completion before post-run screenshots (40 ms sampling). Converter elapsed is separate. One run per arm and different initialization/output can affect speed comparisons. All measured arms cold-initialized: {native_summary["allArmsColdInitialized"]}.</div>
    <h2>Native screenshots</h2><p>Warmup live capture is separate from measured pairs. Final captures occur after inference. Open an image for full resolution.</p><div class="gallery">{"".join(gallery) or "<p>Native screenshots pending.</p>"}</div>
    <details><summary>Runtime, cleanup, and provenance</summary><p>Device {html.escape(str(config.get("device", {}).get("model", "pending")))} · {html.escape(str(config.get("precision", "pending")))} · MTP {html.escape(str(config.get("mtpRequested", "pending")))}</p><p>Model SHA-256 <code>{html.escape(verification.get("modelSha256", "pending"))}</code></p><p>GPU FP16+MTP startup lines: {native_summary["mtpRuntimeLogLines"]}; explicit cleanup lines: {native_summary["cleanupLogLines"]}; maximum Q/DQ rejected: {native_summary["maximumQdqRejected"]}. The filtered log does not isolate cleanup duration.</p><pre>{html.escape(chr(10).join(item["line"] for item in native_summary["nativeRuntimeLogMarkers"]))}</pre></details>
    </main></body></html>'''
    (BASE / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, default=DEFAULT_NATIVE)
    build(parser.parse_args().native_dir)
