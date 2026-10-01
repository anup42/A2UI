#!/usr/bin/env python3
"""Validate one fresh paired Bixby50 device run, replay SDK repair, and score v5.4.

This is a host-only postprocessing command. It never invokes adb or inference.
All 100 attempts must be present in devices/fold7/batches before any report is
published. Raw model text is not edited; the published SDK sees generated text
only. A rejected or output-free attempt contributes zero to its 50-case mean.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import report_bixby50_v54_models as quality
import report_trained_bixby50 as trained


ROOT = Path(__file__).resolve().parents[2]
OLD_RUN = ROOT / "GenUICraft/validation/20260929_bixby50_full_gpu_fp32"
OLD_REPORT = OLD_RUN / "full_native_repaired_report"
MODES = ("litert_mtp_off", "litert_mtp_on")
SDK_VERSION = "0.5.6"
SDK_AAR_SHA = "c59eeae4debf4e8428a806ac7b7d7cb1b13f2d3559a12a655a56ef2f0be723b1"
SDK_JAR_SHA = "69954347411d7d3b4959cc914ecfd290f9c399202ea2d83279c023062a1bcf11"
GSON_SHA = "57928d6e5a6edeb2abd3770a8f95ba44dce45f3b23b7a9dc2b309c581552a78b"
KOTLIN_SHA = "6558a3d233da56a20934b32159f9db5f86ed5816ef098f78a2c223dc6abb79dd"
DRIVER_SHA = "456d6fcb8c791731727152aef516bde0266e48db5de3654acd8d8dfd31c20f03"
DRIVER_CLASS_SHA = "f81630931ea1905b70a87e1d5aee2e153dd5d251db91794b95e4eac81e1c9ad1"
EXPECTED_IDS = {f"BXP-{i:03d}" for i in range(1, 51)}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    need(path.is_file(), f"missing {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    need(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def score(source: str, express: str) -> tuple[float, dict, dict]:
    metrics = quality.score_prediction(source, None, express, metric_version="v5_4")
    breakdown = dataclasses.asdict(quality.generation_reward_v5_4(express, source))
    value = metrics["generation_reward_v5_4"]
    need(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 100
         and value == breakdown["quality_0_100"], "official v5.4 scorers disagree")
    return float(value), metrics, breakdown


def select_renderer_snapshot(snapshot: Path | None) -> dict | None:
    """Use verified prior renderer sources without touching the live checkout."""
    if snapshot is None:
        return None
    from pipeline.genui_quality import identity_v5_4 as identity
    reference = ROOT / "GenUICraft/validation/20260929_bixby50_v54_model_scores/evidence"
    expected_scorer = read_json(reference / "scorer_source_manifest.json")
    need(identity.score_source_manifest_v5_4() == expected_scorer,
         "scoring Python/schema sources differ from the frozen reference")
    expected_renderer = read_json(reference / "android_renderer_manifest.json")
    snapshot = snapshot.resolve()
    need(snapshot.is_dir(), "renderer snapshot directory missing")
    live_manifest = identity.android_renderer_manifest()
    identity.ANDROID_RENDERER_DIR = snapshot
    need(identity.android_renderer_manifest() == expected_renderer,
         "renderer snapshot differs from the prior score manifest")
    return {"path": str(snapshot), "reference_manifest": str(reference / "android_renderer_manifest.json"),
            "reference_manifest_sha256": sha_file(reference / "android_renderer_manifest.json"),
            "manifest": expected_renderer, "scorer_sources_match_reference": True,
            "live_renderer_files_different": sorted(k for k in set(live_manifest) | set(expected_renderer)
                                                    if live_manifest.get(k) != expected_renderer.get(k)),
            "live_checkout_modified_by_report": False}


def validate_result_flags(result: dict, case: Path) -> None:
    """The Android harness writes usedFallback only for conversion successes."""
    status = result.get("status")
    successes = {"valid", "render_invalid"}
    failures = {"strict_invalid", "timeout", "runtime_error", "case_error", "cancelled"}
    need(status in successes | failures, f"unknown conversion status: {case}: {status!r}")
    if status in successes:
        need(result.get("usedFallback") is False,
             f"successful conversion must explicitly disallow source fallback: {case}")
    else:
        need("usedFallback" not in result or result["usedFallback"] is False,
             f"failed conversion reports source fallback or invalid flag: {case}")
    calls = result.get("providerCalls")
    need(type(calls) is int and calls in (0, 1),
         f"provider call count missing or invalid: {case}")


def validate_collection(run: Path) -> tuple[list[dict], dict, dict]:
    protocol = read_json(run / "protocol.json")
    plan = read_json(run / "fold7_plan.json")
    device_dir = run / "devices/fold7"
    provenance = read_json(device_dir / "provenance.json")
    complete = read_json(device_dir / "COMPLETE.json")
    need(plan == read_json(device_dir / "plan.json"), "collected plan differs from declared plan")
    need(protocol.get("plans") == [plan], "protocol plan differs from Fold7 plan")
    need(plan.get("serial") == "R3CY30QFWLP" and plan.get("deviceAlias") == "fold7",
         "expected one Fold7 collector plan")
    need(provenance.get("serial") == plan["serial"] and complete.get("serial") == plan["serial"],
         "Fold7 serial mismatch")
    model_sha = protocol.get("modelSha256")
    need(isinstance(model_sha, str) and len(model_sha) == 64
         and all(ch in "0123456789abcdef" for ch in model_sha), "protocol model SHA-256 missing")
    need(provenance.get("modelSha256") == model_sha, "collected model SHA-256 differs from protocol")
    need(provenance.get("mainApkSha256") == protocol.get("mainApkSha256")
         and provenance.get("installedTestApkSha256") == protocol.get("installedTestApkSha256"),
         "installed APK identity differs from protocol")
    need(protocol.get("repairSdkVersion") == SDK_VERSION, "wrong repair SDK version")
    expected_basename = Path(protocol["modelPath"]).name
    need(expected_basename and expected_basename.endswith(".litertlm"), "protocol model path missing")
    need(complete.get("generations") == 100 and complete.get("completedBatches") ==
         [entry["name"] for entry in plan["batches"]], "collector completion is incomplete")
    expected_keys = {(mode, case_id) for mode in MODES for case_id in EXPECTED_IDS}
    batch_names = {entry["name"] for entry in plan["batches"]}
    need(len(batch_names) == len(plan["batches"]), "duplicate batch names in collector plan")
    actual_batches = {path.name for path in (device_dir / "batches").iterdir() if path.is_dir()}
    need(actual_batches == batch_names,
         f"unexpected or missing collected batches: {sorted(actual_batches ^ batch_names)}")
    planned_keys = [("litert_mtp_on" if entry["mtp"] else "litert_mtp_off", case_id)
                    for entry in plan["batches"] for case_id in entry["cases"]]
    need(len(planned_keys) == 100 and set(planned_keys) == expected_keys,
         "plan must contain each Bixby50 case exactly once in each mode")

    corpus_path = ROOT / "android/app/src/main/assets/genuicraft_bixby50.jsonl"
    corpus_rows = quality.read_jsonl(corpus_path)
    corpus = {row["id"]: row for row in corpus_rows}
    need(len(corpus_rows) == len(corpus) == 50 and set(corpus) == EXPECTED_IDS,
         "Android corpus is not the frozen 50 cases")
    frozen = {row["id"]: trained.expected_source(row) for row in trained.frozen_index()}
    need(set(frozen) == EXPECTED_IDS, "training scorer corpus is not 50 cases")
    for case_id in EXPECTED_IDS:
        need(corpus[case_id]["query"] == frozen[case_id]["query"]
             and corpus[case_id]["text"] == frozen[case_id]["text"],
             f"Android/scorer source mismatch: {case_id}")

    old_config = read_json(OLD_RUN / "devices/fold7/batches/cases_032_036_mtp_off/run_config.json")
    old_prompts = quality.read_jsonl(
        ROOT / "GenUICraft/validation/20260929_bixby50_v54_model_scores/scored/checkpoint_r64.jsonl")
    prompt_by_id = {row["id"]: row["runtime"]["prompt_sha256"] for row in old_prompts}
    need(set(prompt_by_id) == EXPECTED_IDS, "prior frozen prompt reference incomplete")
    prompt_ref = old_config["prompt"]
    prompt_asset = ROOT / "GenUICraft/genuicraft/src/main/assets" / prompt_ref["asset"]
    need(sha_file(prompt_asset) == prompt_ref["assetSha256"], "installed prompt asset reference changed")
    hashes: dict[str, str] = {}
    def hash_input(path: Path) -> None:
        if path.is_file():
            hashes[str(path.relative_to(ROOT))] = sha_file(path)
    for path in (run / "protocol.json", run / "fold7_plan.json", device_dir / "plan.json",
                 device_dir / "provenance.json", device_dir / "COMPLETE.json", corpus_path,
                 trained.FROZEN_CORPUS, prompt_asset):
        hash_input(path)

    rows = []
    batch_evidence = []
    for entry in plan["batches"]:
        batch = device_dir / "batches" / entry["name"]
        summary = read_json(batch / "summary.json")
        config = read_json(batch / "run_config.json")
        host_run = read_json(batch / "host_run.json")
        need(summary.get("runComplete") is True and summary.get("completed") == len(entry["cases"])
             and summary.get("total") == len(entry["cases"]),
             f"batch incomplete: {batch}")
        need(host_run.get("entry") == entry and host_run.get("serial") == plan["serial"]
             and host_run.get("runId") == config.get("runId") == summary.get("runId")
             and host_run.get("adbExitCode") == 0,
             f"collector invocation and device run disagree: {batch}")
        need(config.get("cases") == entry["cases"], f"batch case list differs: {batch}")
        actual_cases = {path.name for path in batch.iterdir()
                        if path.is_dir() and path.name.startswith("BXP-")}
        need(actual_cases == set(entry["cases"]),
             f"batch has missing or extra case directories: {batch}")
        need(config.get("prompt") == prompt_ref, f"prompt identity differs: {batch}")
        need(sha_file(batch / "shared_prompt.json") == prompt_ref["assetSha256"],
             f"captured APK prompt bytes differ: {batch}")
        need(config.get("corpus", {}).get("sha256") == sha_file(corpus_path),
             f"source corpus hash differs: {batch}")
        need(config.get("model", {}).get("basename") == expected_basename
             and config["model"].get("sizeBytes") == protocol["modelSizeBytes"],
             f"model file identity differs: {batch}")
        need(config.get("device", {}).get("model") == provenance.get("deviceModel"),
             f"device model differs: {batch}")
        rt = config.get("runtime", {})
        fixed = protocol["runtime"]
        need(rt.get("accelerator") == fixed["accelerator"] == "GPU"
             and rt.get("gpuPrecision") == fixed["gpuPrecision"] == "FP32"
             and rt.get("maxContextTokens") == fixed["contextTokens"] == 8192
             and rt.get("maxOutputTokens") == fixed["maxOutputTokens"] == 2048
             and rt.get("temperature") == fixed["temperature"] == 0.0
             and rt.get("thinkingEnabled") is fixed["thinkingEnabled"] is False
             and rt.get("mtpEnabled") is entry["mtp"]
             and rt.get("sourceFallbackEnabled") is fixed["sourceFallback"] is False
             and rt.get("generatedDslRepairEnabled") is True
             and rt.get("requireSourceIntegrity") is False
             and rt.get("repairAttempts") == 0,
             f"native settings differ from protocol: {batch}")
        session_path = batch / "session_metrics.json"
        session = read_json(session_path)
        need(session.get("speculativeDecodingEnabled") is entry["mtp"],
             f"MTP native session evidence differs: {batch}")
        rate = session.get("drafterAcceptanceRate")
        if entry["mtp"]:
            need(isinstance(rate, (int, float)) and not isinstance(rate, bool)
                 and math.isfinite(rate) and 0 <= rate <= 1,
                 f"MTP-on drafter acceptance rate missing or invalid: {batch}")
        else:
            need(rate is None, f"MTP-off reports a drafter acceptance rate: {batch}")
        batch_evidence.append({"batch": entry["name"], "mode": "on" if entry["mtp"] else "off",
                               "cases": entry["cases"], "speculative_decoding_enabled":
                               session["speculativeDecodingEnabled"],
                               "drafter_acceptance_rate": rate,
                               "session_metrics_file": str(session_path)})
        for path in (batch / "summary.json", batch / "run_config.json", batch / "shared_prompt.json", session_path,
                     batch / "host_run.json"):
            hash_input(path)
        for case_id in entry["cases"]:
            case = batch / case_id
            source = read_json(case / "source.json")
            result = read_json(case / "result.json")
            need(source.get("id") == result.get("id") == case_id
                 and source.get("query") == corpus[case_id]["query"]
                 and source.get("text") == corpus[case_id]["text"],
                 f"raw source differs from frozen corpus: {case}")
            validate_result_flags(result, case)
            raw_path = case / "output.express"
            raw = raw_path.read_text(encoding="utf-8") if raw_path.is_file() else None
            metrics_path = case / "metrics.json"
            metrics = read_json(metrics_path) if metrics_path.is_file() else result.get("metrics") or {}
            need(isinstance(metrics, dict), f"invalid native metrics: {case}")
            if raw is not None:
                need(result["providerCalls"] == 1, f"raw output has no provider call: {case}")
                expected_runtime = "LiteRT-LM/Gemma4/GPU+FP32" + ("+MTP" if entry["mtp"] else "")
                need(metrics.get("runtime") == expected_runtime,
                     f"reported native runtime differs: {case}")
                need(result.get("renderedPromptSha256") == prompt_by_id[case_id],
                     f"rendered prompt differs from frozen case: {case}")
            else:
                need(result.get("status") in ("timeout", "runtime_error", "case_error", "cancelled"),
                     f"unexplained missing raw output: {case}")
            for path in (case / "source.json", case / "result.json", metrics_path,
                         raw_path, case / "a2ui.json"):
                hash_input(path)
            rows.append({
                "model": "litert_mtp_on" if entry["mtp"] else "litert_mtp_off",
                "id": case_id, "serial": plan["serial"], "device_model": provenance["deviceModel"],
                "batch": entry["name"], "source_run": str(batch), "generation_origin": "fresh_fold7",
                "query": source["query"], "response_text": source["text"],
                "raw_input_file": str(raw_path) if raw is not None else None,
                "raw_output": raw, "raw_output_sha256": sha_text(raw) if raw is not None else None,
                "runtime": metrics, "provider_result": result,
                "finish_reason": metrics.get("finishReason") or result.get("status"),
                "device_a2ui_file": str(case / "a2ui.json") if (case / "a2ui.json").is_file() else None,
            })
    need(len(rows) == 100 and {(row["model"], row["id"]) for row in rows} == expected_keys,
         "collection has missing or duplicate attempts")
    rows.sort(key=lambda row: (row["model"], row["id"]))
    return rows, {"protocol": protocol, "plan": plan, "device_provenance": provenance,
                  "collection_complete": complete, "prompt_reference": prompt_ref,
                  "batch_native_sessions": batch_evidence}, hashes


def replay_repair(rows: list[dict], out: Path, java: str | None) -> tuple[dict, dict]:
    aar = ROOT / "GenUICraft/build/repo/com/samsung/genuicraft/genuicraft/0.5.6/genuicraft-0.5.6.aar"
    driver_source = ROOT / "GenUICraft/tools/GenUiRepairReplay.java"
    runtime = OLD_RUN / "runtime"
    classes = runtime / "driver_classes_verified"
    driver_class = classes / "GenUiRepairReplay.class"
    jars = [runtime / "genuicraft-0.5.6-classes.jar", runtime / "gson-2.11.0.jar",
            runtime / "kotlin-stdlib-2.2.21.jar"]
    for path, expected in [(aar, SDK_AAR_SHA), (driver_source, DRIVER_SHA),
                           (driver_class, DRIVER_CLASS_SHA),
                           *zip(jars, (SDK_JAR_SHA, GSON_SHA, KOTLIN_SHA))]:
        need(path.is_file() and sha_file(path) == expected, f"published repair dependency changed: {path}")
    with zipfile.ZipFile(aar) as archive:
        need(hashlib.sha256(archive.read("classes.jar")).hexdigest() == SDK_JAR_SHA,
             "reused SDK jar differs from published AAR")
    old_verification = read_json(runtime / "clean_compile_verification.json")
    need(old_verification.get("returncode") == 0
         and old_verification.get("class_sha256") == DRIVER_CLASS_SHA,
         "verified Java driver provenance missing")
    if java is None:
        old_repair = read_json(OLD_RUN / "repair_provenance.json")
        candidate = Path(old_repair["command"][0])
        java = str(candidate) if candidate.is_file() else shutil.which("java")
    need(bool(java), "Java runtime unavailable; pass --java")
    work = out / "runtime"
    work.mkdir()
    available = [{key: row[key] for key in ("model", "id", "raw_output")}
                 for row in rows if row["raw_output"] is not None]
    input_path = work / "repair_input.jsonl"
    result_path = work / "repair_captured_outcomes.jsonl"
    write_jsonl(input_path, available)
    command = [str(java), "-Xmx1g", "-cp", os.pathsep.join([str(classes), *map(str, jars)]),
               "GenUiRepairReplay", str(input_path), str(result_path)]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (work / "repair.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    need(process.returncode == 0, f"published SDK replay failed; inspect {work / 'repair.log'}")
    captured = quality.read_jsonl(result_path)
    need(len(captured) == len(available), "SDK replay returned wrong number of outcomes")
    repairs = {}
    for repair in captured:
        key = (repair.get("model"), repair.get("id"))
        need(key not in repairs, f"duplicate SDK replay result: {key}")
        repairs[key] = repair
    for row in rows:
        key = (row["model"], row["id"])
        if row["raw_output"] is None:
            repairs[key] = {"model": row["model"], "id": row["id"], "repair_success": False,
                            "repair_kind": "RUNTIME_NO_OUTPUT", "model_calls": 0,
                            "source_text_supplied_to_repair": False, "source_fallback_enabled": False,
                            "repair_error": row["provider_result"].get("error")}
    need(len(repairs) == len(rows), "SDK repair outcomes do not cover all attempts")
    provenance = {"repair_api": "GenUiCompiler.compileWithRepair(raw, null, false, true)",
                  "repair_library": "published GenUICraft 0.5.6", "aar": str(aar),
                  "aar_sha256": sha_file(aar), "driver_source": str(driver_source),
                  "driver_source_sha256": sha_file(driver_source), "verified_driver_class": str(driver_class),
                  "verified_driver_class_sha256": sha_file(driver_class),
                  "classpath": [{"path": str(path), "sha256": sha_file(path)} for path in jars],
                  "source_text_supplied_to_repair": False, "source_fallback_enabled": False,
                  "new_model_inference": False, "raw_outputs_sent": len(available),
                  "runtime_no_output": len(rows) - len(available), "command": command,
                  "repair_input_sha256": sha_file(input_path),
                  "repair_output_sha256": sha_file(result_path)}
    return repairs, provenance


def score_rows(rows: list[dict], repairs: dict) -> tuple[dict, list[dict]]:
    scored = {mode: [] for mode in MODES}
    parity = []
    identity = None
    for item in rows:
        mode, case_id = item["model"], item["id"]
        prefix = f"{mode}/{case_id}"
        repair = repairs[(mode, case_id)]
        need(repair.get("source_text_supplied_to_repair") is False
             and repair.get("source_fallback_enabled") is False
             and repair.get("model_calls") == 0,
             f"repair policy violated: {prefix}")
        raw = item["raw_output"]
        kind = repair.get("repair_kind")
        success = repair.get("repair_success")
        need(type(success) is bool, f"repair result missing status: {prefix}")
        if raw is None:
            need(not success and kind == "RUNTIME_NO_OUTPUT", f"missing raw output incorrectly repaired: {prefix}")
        elif success:
            need(kind in ("NONE", "STRUCTURAL", "GENERATED_DSL_REPAIR")
                 and isinstance(repair.get("repaired_express"), str)
                 and isinstance(repair.get("repaired_a2ui_json"), str),
                 f"accepted SDK repair incomplete: {prefix}")
        else:
            need(kind == "REJECTED" and not repair.get("repaired_express"),
                 f"rejected SDK repair contains output: {prefix}")
        raw_score = raw_metrics = raw_breakdown = None
        if raw is not None:
            raw_score, raw_metrics, raw_breakdown = score(item["response_text"], raw)
        repaired = repair.get("repaired_express") if success else None
        repaired_score = repaired_metrics = repaired_breakdown = None
        contract_equal = None
        if success:
            repaired_score, repaired_metrics, repaired_breakdown = score(item["response_text"], repaired)
            need(repaired_metrics["schema_valid_strict"] is True,
                 f"SDK accepted text that strict v5.4 rejects: {prefix}")
            raw_contract = raw_metrics["metric_identity_v5_4"]
            repaired_contract = repaired_metrics["metric_identity_v5_4"]
            keys = ("source_hash", "expected_contract_hash")
            contract_equal = all(raw_contract.get(key) is not None
                                 and raw_contract[key] == repaired_contract.get(key) for key in keys)
            need(contract_equal, f"raw/repaired source contract differs: {prefix}")
            device_file = item["device_a2ui_file"]
            need(device_file is not None, f"accepted repair lacks device A2UI: {prefix}")
            device_json = json.loads(Path(device_file).read_text(encoding="utf-8-sig"))
            host_json = json.loads(repair["repaired_a2ui_json"])
            equal = device_json == host_json
            parity.append({"model": mode, "id": case_id, "device_a2ui_file": device_file,
                           "device_a2ui_sha256": sha_file(Path(device_file)),
                           "host_a2ui_sha256": sha_text(repair["repaired_a2ui_json"]),
                           "json_structure_and_values_equal": equal})
            need(equal, f"published host SDK A2UI differs from device output: {prefix}")
        else:
            need(item["device_a2ui_file"] is None,
                 f"device accepted A2UI where host SDK rejected: {prefix}")
        for metrics in (raw_metrics, repaired_metrics):
            if metrics:
                m = metrics["metric_identity_v5_4"]
                marker = (m["metric_version"], m["metric_fingerprint"],
                          m["reward_pipeline_fingerprint"])
                if identity is None:
                    identity = marker
                need(marker == identity, f"v5.4 metric identity changed: {prefix}")
        result = item["provider_result"]
        capture_failure = (result.get("status") == "render_invalid"
                           and result.get("strictValid") is True
                           and result.get("rendererContractValid") is True
                           and any(result.get(flag) is False for flag in
                                   ("screenCaptured", "scrolledScreenCaptured", "currentCaseVisible")))
        failures = []
        if raw is None:
            failures.append("RUNTIME_NO_OUTPUT")
        elif not success:
            failures.append("SDK_REPAIR_REJECTED")
        if capture_failure:
            failures.append("SCREEN_CAPTURE_FAILED")
        elif result.get("status") == "render_invalid":
            failures.append("ANDROID_RENDER_INVALID_OTHER")
        native = item["runtime"]
        native_output_tokens = native.get("actualOutputTokens", native.get("outputTokens"))
        row = {**item, "raw_score": raw_score,
               "raw_score_for_aggregate": raw_score if raw_score is not None else 0.0,
               "raw_strict_valid": raw_metrics["schema_valid_strict"] if raw_metrics else None,
               "raw_metrics": raw_metrics, "raw_breakdown": raw_breakdown,
               "repair_success": success, "repair_kind": kind,
               "repair_error": repair.get("repair_error"), "repair_diagnostics": repair.get("diagnostics"),
               "repaired_express": repaired, "repaired_a2ui_json": repair.get("repaired_a2ui_json") if success else None,
               "repaired_express_sha256": sha_text(repaired) if repaired is not None else None,
               "repaired_score": repaired_score,
               "repaired_score_for_aggregate": repaired_score if repaired_score is not None else 0.0,
               "repaired_strict_valid": repaired_metrics["schema_valid_strict"] if repaired_metrics else None,
               "repaired_metrics": repaired_metrics, "repaired_breakdown": repaired_breakdown,
               "source_contract_equal": contract_equal, "device_a2ui_matches_host": True if success else None,
               "native_output_tokens": native_output_tokens,
               "native_decode_tokens_per_second": native.get("decodeTokensPerSecond"),
               "native_provider_call_ms": native.get("providerCallMs"),
               "capture_failure": capture_failure, "failure_categories": failures}
        scored[mode].append(row)
    need(len(parity) == sum(row["repair_success"] for mode in MODES for row in scored[mode]),
         "not every accepted SDK repair was compared against the device")
    return scored, parity


def summarize(rows: list[dict]) -> dict:
    need(len(rows) == 50, "each mode must have exactly 50 attempts")
    measured = [r for r in rows if r["finish_reason"] == "COMPLETED"
                and isinstance(r["native_output_tokens"], (int, float))
                and isinstance(r["native_decode_tokens_per_second"], (int, float))
                and r["native_output_tokens"] > 0 and r["native_decode_tokens_per_second"] > 0]
    native_tokens = sum(r["native_output_tokens"] for r in measured)
    native_seconds = sum(r["native_output_tokens"] / r["native_decode_tokens_per_second"]
                         for r in measured)
    return {"n": 50,
            "raw_mean_failures_zero": sum(r["raw_score_for_aggregate"] for r in rows) / 50,
            "repaired_mean_failures_zero": sum(r["repaired_score_for_aggregate"] for r in rows) / 50,
            "raw_output_missing": sum(r["raw_output"] is None for r in rows),
            "raw_strict_valid": sum(r["raw_strict_valid"] is True for r in rows),
            "repair_accepted": sum(r["repair_success"] is True for r in rows),
            "repair_failed": sum(r["repair_success"] is False for r in rows),
            "repaired_strict_valid": sum(r["repaired_strict_valid"] is True for r in rows),
            "repair_kinds": dict(Counter(r["repair_kind"] for r in rows)),
            "finish_reason_counts": dict(Counter(str(r["finish_reason"] or "unavailable") for r in rows)),
            "failure_category_counts": dict(Counter(category for r in rows for category in r["failure_categories"])),
            "repetition_guard_stopped": sum(r["finish_reason"] == "REPETITION_LIMIT" for r in rows),
            "at_or_above_output_cap": sum((r["native_output_tokens"] or 0) >= 2048 for r in rows),
            "native_decode": {"completed_measured_cases": len(measured), "output_tokens": native_tokens,
                              "estimated_decode_seconds_from_native_rates": native_seconds,
                              "tokens_per_second_weighted": native_tokens / native_seconds if native_seconds else None}}


def render_markdown(summary: dict, scored: dict) -> str:
    modes = summary["modes"]
    off, on = (modes[m] for m in MODES)
    prior = summary["prior_r64"]
    prior_fold7 = summary["prior_r64_fold7_subset"]
    rows = []
    failures = []
    for left, right in zip(scored[MODES[0]], scored[MODES[1]]):
        rows.append([left["id"], left["query"], f'{left["raw_score_for_aggregate"]:.2f}',
                     f'{left["repaired_score_for_aggregate"]:.2f}', left["repair_kind"],
                     f'{right["raw_score_for_aggregate"]:.2f}',
                     f'{right["repaired_score_for_aggregate"]:.2f}', right["repair_kind"]])
    for mode in MODES:
        for r in scored[mode]:
            if r["failure_categories"]:
                failures.append([r["id"], "on" if mode == MODES[1] else "off",
                                 ", ".join(r["failure_categories"]), r["finish_reason"],
                                 r["repair_error"] or r["provider_result"].get("error") or "—"])
    if not failures:
        failures = [["None", "—", "—", "—", "—"]]
    return "\n".join([
        "# R32 LiteRT Bixby50, native Fold7 GPU FP32", "",
        f"Generated: {summary['created_at_utc']}", "",
        "The collection contains **100 fresh native Fold7 attempts**, one for every Bixby50 case in each MTP mode. "
        "Each mode has a fixed denominator of 50. Raw outputs were scored unchanged. Captured outputs then passed "
        "through the published GenUICraft 0.5.6 `compileWithRepair(raw, null, false, true)` route. "
        "No source text, model call, or source fallback was supplied to host repair.", "",
        quality.table(["Population", "n", "Raw mean /100", "Repaired mean /100", "Raw strict", "SDK accepted"], [
            ["R32 MTP off", 50, f'{off["raw_mean_failures_zero"]:.2f}', f'{off["repaired_mean_failures_zero"]:.2f}', off["raw_strict_valid"], off["repair_accepted"]],
            ["R32 MTP on", 50, f'{on["raw_mean_failures_zero"]:.2f}', f'{on["repaired_mean_failures_zero"]:.2f}', on["raw_strict_valid"], on["repair_accepted"]],
            ["Prior R64 MTP off", 50, "—", f'{prior[MODES[0]]["repaired_mean_failures_zero"]:.2f}', "—", "—"],
            ["Prior R64 MTP on", 50, "—", f'{prior[MODES[1]]["repaired_mean_failures_zero"]:.2f}', "—", "—"]]), "",
        f"R32 versus prior R64 repaired mean: off **{off['repaired_mean_failures_zero'] - prior[MODES[0]]['repaired_mean_failures_zero']:+.2f}**; "
        f"on **{on['repaired_mean_failures_zero'] - prior[MODES[1]]['repaired_mean_failures_zero']:+.2f}** points. "
        "The prior R64 run used Flip8 for cases 001–031 and Fold7 for 032–050; this run uses Fold7 for all cases. "
        "The difference mixes model, device, and run conditions and is descriptive only. "
        "The R32 label comes from the supplied model filename; the checkpoint/export lineage has not been independently certified.", "",
        f"For the shared Fold7 BXP-032–050 subset (19 cases per mode), prior R64 repaired means were "
        f"{prior_fold7[MODES[0]]['repaired_mean_failures_zero']:.2f} off and "
        f"{prior_fold7[MODES[1]]['repaired_mean_failures_zero']:.2f} on. "
        f"Fresh R32 subset means are {summary['r32_fold7_matching_subset'][MODES[0]]['repaired_mean_failures_zero']:.2f} "
        f"off and {summary['r32_fold7_matching_subset'][MODES[1]]['repaired_mean_failures_zero']:.2f} on. "
        "This narrows the device difference but remains a separate run under different conditions.", "",
        "The v5.4 score measures source-to-UI representation quality on a 0–100 scale. It is not source factual accuracy "
        "or a certificate of visual quality. Rejected and output-free attempts have null individual repaired scores "
        "and contribute zero to their mode mean. Screen capture failures are recorded separately and do not zero "
        "an SDK-accepted Express score.", "",
        "## Finish, failures, and acceptance", "",
        quality.table(["Mode", "Missing raw", "SDK accepted", "SDK rejected/no output", "Repetition stops", "At output cap", "Measured native decode tok/s"], [
            ["off", off["raw_output_missing"], off["repair_accepted"], off["repair_failed"], off["repetition_guard_stopped"], off["at_or_above_output_cap"],
             f'{off["native_decode"]["tokens_per_second_weighted"]:.2f}' if off["native_decode"]["tokens_per_second_weighted"] else "—"],
            ["on", on["raw_output_missing"], on["repair_accepted"], on["repair_failed"], on["repetition_guard_stopped"], on["at_or_above_output_cap"],
             f'{on["native_decode"]["tokens_per_second_weighted"]:.2f}' if on["native_decode"]["tokens_per_second_weighted"] else "—"]]), "",
        "Native decode rate is total output tokens divided by the sum of each completed case's tokens/rate seconds; "
        "it excludes initialization, prefill, capture, and host repair. The alternating batch order reduces a simple "
        "order bias but does not control thermal or output-length effects.", "",
        f"All {summary['native_sessions']['batches']} batch sessions reported the requested MTP setting. "
        f"The MTP-on drafter acceptance rate averaged {summary['native_sessions']['mtp_on_drafter_acceptance_simple_mean']:.3f} "
        "across its ten batch-reported rates; this is an unweighted mean of batch values, not a token-weighted rate.", "",
        quality.table(["Case", "Mode", "Failure evidence", "Finish", "Repair/provider error"], failures), "",
        "## Per-case scores", "",
        quality.table(["Case", "Source query", "Off raw", "Off repaired", "Off SDK route", "On raw", "On repaired", "On SDK route"], rows), "",
        "## Provenance and checks", "",
        f"Fold7 serial: `{summary['device']['serial']}`; reported model: `{summary['device']['deviceModel']}`. "
        f"Supplied R32 model SHA-256: `{summary['model_sha256']}`. Published AAR SHA-256: `{summary['repair']['aar_sha256']}`. "
        f"Metric fingerprint: `{summary['metric_identity']['metric_fingerprint']}`. "
        f"Host SDK A2UI equals device output as parsed JSON for **{summary['parity']['passed']}/{summary['parity']['checked']}** accepted attempts. "
        "Every source/query matched the frozen corpus. Every rendered prompt hash matched the prior frozen prompt. "
        "Native session metrics verified speculative decoding off/on for every batch; the per-batch drafter "
        "acceptance rates are in `provenance.json`. "
        "Raw/repaired source-contract hashes matched for accepted outputs. See `provenance.json`, "
        "`repair_device_parity.json`, and `scoring_input_hashes.json` for paths and hashes.", "",
        "Scoring uses the earlier report's verified renderer-source snapshot when configured. "
        "This keeps the v5.4 identity comparable while preserving concurrent edits in the live checkout; "
        "the Python scorer and all snapshot renderer hashes must match the earlier reference. "
        "Device inference is not repeated during reporting.", "",
        "[Interactive HTML](index.html) · [Per-case CSV](per_case_scores.csv) · [Summary JSON](summary.json) · "
        "[MTP off JSONL](scored/litert_mtp_off.jsonl) · [MTP on JSONL](scored/litert_mtp_on.jsonl)", "",
    ]) + "\n"


def render_html(summary: dict, scored: dict) -> str:
    e = html.escape
    cards = []
    for off, on in zip(scored[MODES[0]], scored[MODES[1]]):
        parts = []
        for row, label in ((off, "MTP off"), (on, "MTP on")):
            error = row["repair_error"] or row["provider_result"].get("error") or ""
            parts.append(f'<section class="mode"><h3>{label}</h3><strong>{row["raw_score_for_aggregate"]:.2f} → {row["repaired_score_for_aggregate"]:.2f} /100</strong>'
                         f'<p>{e(row["repair_kind"])} · {e(str(row["finish_reason"]))} · {e(", ".join(row["failure_categories"]))}</p>'
                         f'<details><summary>Raw model output</summary><pre>{e(row["raw_output"] or "No native output")}</pre></details>'
                         f'<details><summary>SDK repaired Express</summary><pre>{e(row["repaired_express"] or "No repaired artifact")}</pre></details>'
                         f'<details><summary>Native and repair evidence</summary><pre>{e(json.dumps({"error": error, "runtime": row["runtime"], "provider_result": row["provider_result"], "batch": row["batch"], "source_run": row["source_run"]}, ensure_ascii=False, indent=2))}</pre></details></section>')
        cards.append(f'<article data-find="{e((off["id"] + " " + off["query"]).lower(), quote=True)}"><h2>{off["id"]}</h2><p>{e(off["query"])}</p>'
                     f'<details><summary>Frozen source answer (not supplied to repair)</summary><pre>{e(off["response_text"])}</pre></details>'
                     f'<div class="modes">{"".join(parts)}</div></article>')
    off, on = (summary["modes"][mode] for mode in MODES)
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R32 Bixby50 native SDK repair</title><style>
:root{{font:15px/1.5 system-ui,sans-serif;color:#14243c;background:#f3f6fb}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}header,article{{background:#fff;border:1px solid #dce3ee;border-radius:14px;padding:22px;margin-bottom:16px}}h1{{margin:0}}h2{{font-size:20px}}h3{{margin:0}}.totals,.modes{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.mode{{border:1px solid #dce3ee;border-radius:10px;padding:16px;min-width:0}}strong{{font-size:23px}}input{{padding:12px;width:min(650px,100%);font:inherit;border:1px solid #aab7ca;border-radius:8px}}details{{margin:12px 0}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow:auto;padding:12px;background:#f5f7fa;border-radius:8px;font:12px/1.45 Consolas,monospace}}@media(max-width:850px){{.modes,.totals{{grid-template-columns:1fr}}main{{padding:10px}}}}
</style><main><header><h1>R32 LiteRT Bixby50 on Fold7</h1><p>100 fresh attempts, 50 per mode. Published SDK 0.5.6 repair, generated output only; failures count as zero in each 50-case mean. V5.4 is source-to-UI representation quality, not factual accuracy.</p><div class="totals"><p>MTP off<br><strong>{off['raw_mean_failures_zero']:.2f} → {off['repaired_mean_failures_zero']:.2f}</strong> /100</p><p>MTP on<br><strong>{on['raw_mean_failures_zero']:.2f} → {on['repaired_mean_failures_zero']:.2f}</strong> /100</p></div><p>Prior R64 repaired means: off {summary['prior_r64'][MODES[0]]['repaired_mean_failures_zero']:.2f}, on {summary['prior_r64'][MODES[1]]['repaired_mean_failures_zero']:.2f}. Prior cases 001–031 ran on Flip8, so this is a descriptive comparison. <a href="REPORT.md">Full report</a> · <a href="per_case_scores.csv">CSV</a> · <a href="summary.json">JSON</a></p><label>Find case or query<br><input id="find" placeholder="BXP-001"></label><p id="count">Showing 50 cases</p></header>{''.join(cards)}</main><script>
const input=document.getElementById('find'),cards=[...document.querySelectorAll('article[data-find]')],count=document.getElementById('count');input.addEventListener('input',()=>{{const q=input.value.trim().toLowerCase();let n=0;for(const card of cards){{card.hidden=!card.dataset.find.includes(q);if(!card.hidden)n++}}count.textContent=`Showing ${{n}} of 50 cases`}});
</script></html>'''


def publish(run: Path, out: Path, scored: dict, parity: list[dict], evidence: dict,
            hashes: dict, repair_provenance: dict, prior_report: Path) -> dict:
    old_summary_path = prior_report / "summary.json"
    old_summary = read_json(old_summary_path)
    prior_modes = old_summary.get("modes", {})
    expected_prior = {MODES[0]: 83.25794194833765, MODES[1]: 80.89039717826645}
    for mode in MODES:
        need(prior_modes.get(mode, {}).get("n") == 50
             and math.isclose(prior_modes[mode]["repaired_mean_failures_zero"], expected_prior[mode], abs_tol=1e-9),
             f"prior R64 full 50-case mean differs: {mode}")
    same_device_prior = {}
    for mode in MODES:
        path = prior_report / "scored" / f"{mode}.jsonl"
        old_rows = quality.read_jsonl(path)
        subset = [row for row in old_rows if row.get("serial") == "R3CY30QFWLP"]
        need(len(old_rows) == 50 and len(subset) == 19
             and {row["id"] for row in subset} == {f"BXP-{i:03d}" for i in range(32, 51)},
             f"prior R64 same-device subset is incomplete: {mode}")
        same_device_prior[mode] = {"n": 19, "repaired_mean_failures_zero":
                                   sum(row["repaired_score_for_aggregate"] for row in subset) / 19}
        hashes[str(path.relative_to(ROOT))] = sha_file(path)
    modes = {mode: summarize(scored[mode]) for mode in MODES}
    metric = old_summary["metric_identity"]
    for mode in MODES:
        for row in scored[mode]:
            for key in ("raw_metrics", "repaired_metrics"):
                if row[key] is not None:
                    observed = row[key]["metric_identity_v5_4"]
                    need(all(observed.get(k) == metric[k] for k in
                             ("metric_version", "metric_fingerprint", "reward_pipeline_fingerprint")),
                         f"current scorer differs from prior R64 scorer: {mode}/{row['id']}")
    hashes[str(old_summary_path.relative_to(ROOT))] = sha_file(old_summary_path)
    hashes[str(Path(__file__).relative_to(ROOT))] = sha_file(Path(__file__))
    native_sessions = evidence["batch_native_sessions"]
    on_rates = [row["drafter_acceptance_rate"] for row in native_sessions if row["mode"] == "on"]
    need(len(native_sessions) == 20 and len(on_rates) == 10,
         "expected ten native sessions per MTP mode")
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "population": "100 fresh Fold7 native attempts; 50 MTP off and 50 MTP on",
               "model_label": evidence["protocol"]["modelLabel"],
               "model_sha256": evidence["protocol"]["modelSha256"],
               "device": evidence["device_provenance"], "metric_identity": metric,
               "repair": repair_provenance, "modes": modes,
               "prior_r64": {mode: {"n": 50, "repaired_mean_failures_zero":
                                      prior_modes[mode]["repaired_mean_failures_zero"]} for mode in MODES},
               "prior_r64_fold7_subset": same_device_prior,
               "r32_fold7_matching_subset": {mode: {"n": 19, "repaired_mean_failures_zero":
                                            sum(row["repaired_score_for_aggregate"] for row in scored[mode]
                                                if 32 <= int(row["id"][-3:]) <= 50) / 19}
                                            for mode in MODES},
               "prior_r64_report": str(prior_report),
               "comparison_caveat": evidence["protocol"].get("comparisonCaveat"),
               "native_sessions": {"batches": 20, "mtp_on_drafter_acceptance_simple_mean":
                                   sum(on_rates) / len(on_rates),
                                   "mean_method": "unweighted arithmetic mean of ten batch-reported rates"},
               "paired": {"n": 50, "raw_on_minus_off_mean":
                          modes[MODES[1]]["raw_mean_failures_zero"] - modes[MODES[0]]["raw_mean_failures_zero"],
                          "repaired_on_minus_off_mean":
                          modes[MODES[1]]["repaired_mean_failures_zero"] - modes[MODES[0]]["repaired_mean_failures_zero"]},
               "parity": {"checked": len(parity), "passed": sum(x["json_structure_and_values_equal"] for x in parity)},
               "scoring_input_hashes": hashes}
    write_json(out / "summary.json", summary)
    write_json(out / "provenance.json", evidence)
    write_json(out / "repair_provenance.json", repair_provenance)
    write_json(out / "repair_device_parity.json", parity)
    write_json(out / "scoring_input_hashes.json", hashes)
    (out / "scored").mkdir()
    for mode in MODES:
        write_jsonl(out / "scored" / f"{mode}.jsonl", scored[mode])
    fields = ["model", "id", "serial", "batch", "raw_score", "raw_score_for_aggregate",
              "repaired_score", "repaired_score_for_aggregate", "raw_strict_valid",
              "repair_success", "repair_kind", "repaired_strict_valid", "source_contract_equal",
              "device_a2ui_matches_host", "finish_reason", "native_output_tokens",
              "native_decode_tokens_per_second", "native_provider_call_ms", "raw_output_sha256",
              "repaired_express_sha256", "failure_categories", "repair_error", "raw_input_file",
              "device_a2ui_file"]
    with (out / "per_case_scores.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode in MODES:
            for row in scored[mode]:
                record = {name: row.get(name) for name in fields}
                record["failure_categories"] = ";".join(row["failure_categories"])
                writer.writerow(record)
    (out / "REPORT.md").write_text(render_markdown(summary, scored), encoding="utf-8", newline="\n")
    (out / "index.html").write_text(render_html(summary, scored), encoding="utf-8", newline="\n")
    return summary


def preflight_batch(run: Path, batch_name: str, out: Path, java: str | None) -> dict:
    """Exercise the same SDK replay and scorer on a real completed collector batch."""
    protocol = read_json(run / "protocol.json")
    batch = run / "devices/fold7/batches" / batch_name
    config = read_json(batch / "run_config.json")
    summary = read_json(batch / "summary.json")
    session = read_json(batch / "session_metrics.json")
    expected_entries = [entry for entry in read_json(run / "fold7_plan.json")["batches"]
                        if entry["name"] == batch_name]
    need(len(expected_entries) == 1, "preflight batch is not in the run plan")
    entry = expected_entries[0]
    need(summary.get("runComplete") is True and summary.get("completed") == len(entry["cases"])
         and config.get("cases") == entry["cases"], "preflight batch incomplete")
    need(config["model"]["basename"] == Path(protocol["modelPath"]).name
         and config["model"]["sizeBytes"] == protocol["modelSizeBytes"], "wrong preflight model")
    need(config["runtime"]["mtpEnabled"] is entry["mtp"]
         and session.get("speculativeDecodingEnabled") is entry["mtp"]
         and config["runtime"]["sourceFallbackEnabled"] is False,
         "preflight MTP or fallback policy differs")
    frozen = {row["id"]: trained.expected_source(row) for row in trained.frozen_index()}
    rows = []
    for case_id in entry["cases"]:
        case = batch / case_id
        source, result = read_json(case / "source.json"), read_json(case / "result.json")
        validate_result_flags(result, case)
        need(source.get("id") == result.get("id") == case_id
             and source.get("query") == frozen[case_id]["query"]
             and source.get("text") == frozen[case_id]["text"],
             f"preflight source differs: {case_id}")
        raw_path = case / "output.express"
        raw = raw_path.read_text(encoding="utf-8") if raw_path.is_file() else None
        metrics_path = case / "metrics.json"
        metrics = read_json(metrics_path) if metrics_path.is_file() else result.get("metrics") or {}
        rows.append({"model": "litert_mtp_on" if entry["mtp"] else "litert_mtp_off",
                     "id": case_id, "serial": "R3CY30QFWLP", "device_model": config["device"]["model"],
                     "batch": batch_name, "source_run": str(batch), "generation_origin": "fresh_fold7",
                     "query": source["query"], "response_text": source["text"],
                     "raw_input_file": str(raw_path) if raw is not None else None,
                     "raw_output": raw, "raw_output_sha256": sha_text(raw) if raw is not None else None,
                     "runtime": metrics, "provider_result": result,
                     "finish_reason": metrics.get("finishReason") or result.get("status"),
                     "device_a2ui_file": str(case / "a2ui.json") if (case / "a2ui.json").is_file() else None})
    out.mkdir(parents=True)
    repairs, repair_provenance = replay_repair(rows, out, java)
    scored, parity = score_rows(rows, repairs)
    mode = rows[0]["model"]
    report = {"status": "five-case preflight only; full-run result not yet available",
              "batch": batch_name, "mode": mode, "cases": len(rows),
              "raw_mean": sum(r["raw_score_for_aggregate"] for r in scored[mode]) / len(rows),
              "repaired_mean": sum(r["repaired_score_for_aggregate"] for r in scored[mode]) / len(rows),
              "sdk_accepted": sum(r["repair_success"] for r in scored[mode]),
              "device_a2ui_parity": len(parity),
              "speculative_decoding_enabled": session["speculativeDecodingEnabled"],
              "drafter_acceptance_rate": session.get("drafterAcceptanceRate"),
              "repair_provenance": repair_provenance}
    write_json(out / "summary.json", report)
    write_jsonl(out / "scored.jsonl", scored[mode])
    write_json(out / "repair_device_parity.json", parity)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="new report directory inside the run root")
    parser.add_argument("--prior-report", type=Path, default=OLD_REPORT)
    parser.add_argument("--java", help="Java executable; defaults to the prior verified replay runtime")
    parser.add_argument("--renderer-snapshot", type=Path,
                        help="renderer directory whose hashes exactly match the prior v5.4 report")
    parser.add_argument("--validate-only", action="store_true", help="validate collection without replay or writing")
    parser.add_argument("--preflight-batch", help="replay and score one real completed batch before full collection")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    out = (args.out or run / ("preflight_" + args.preflight_batch if args.preflight_batch
                              else "fresh_native_repaired_report")).resolve()
    try:
        need(run.is_dir(), f"run root missing: {run}")
        need(out != run and run in out.parents, "report directory must be inside run root")
        need(not out.exists(), f"report target exists; preserve it and choose another --out: {out}")
        renderer_snapshot = select_renderer_snapshot(args.renderer_snapshot)
        if args.preflight_batch:
            need(not args.validate_only, "--preflight-batch cannot be combined with --validate-only")
            report = preflight_batch(run, args.preflight_batch, out, args.java)
            print(json.dumps({k: v for k, v in report.items() if k != "repair_provenance"}, indent=2))
            return 0
        rows, evidence, hashes = validate_collection(run)
        evidence["scoring_renderer_snapshot"] = renderer_snapshot
        if args.validate_only:
            print("Validated 100 fresh Fold7 attempts, source identity, model identity, and protocol")
            return 0
        out.mkdir(parents=True)
        repairs, repair_provenance = replay_repair(rows, out, args.java)
        scored, parity = score_rows(rows, repairs)
        summary = publish(run, out, scored, parity, evidence, hashes, repair_provenance,
                          args.prior_report.resolve())
        print(json.dumps({"report": str(out), "modes": summary["modes"], "parity": summary["parity"]}, indent=2))
        return 0
    except Exception as exc:
        print(f"report_bixby50_fresh_native.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
