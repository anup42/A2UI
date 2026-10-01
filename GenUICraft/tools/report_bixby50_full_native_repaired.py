#!/usr/bin/env python3
"""Score the complete paired native Bixby50 run after published SDK repair.

This program performs no inference or repair. It scores immutable raw completions
and the separate GenUiRepairReplay outcomes, retaining runtime and repair failures
in each 50-case aggregate. The prior checkpoint files are comparison baselines.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import html
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import report_bixby50_v54_models as quality
import report_trained_bixby50 as trained


MODES = ("litert_mtp_off", "litert_mtp_on")
CHECKPOINTS = ("checkpoint_r32", "checkpoint_r64")
PRIOR = quality.ROOT / "GenUICraft/validation/20260929_bixby50_v54_repaired_scores"
EXPECTED_MODEL_SHA256 = "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
EXPECTED_AAR_SHA256 = "c59eeae4debf4e8428a806ac7b7d7cb1b13f2d3559a12a655a56ef2f0be723b1"
EXPECTED_PRIOR_COUNTS = {"litert_mtp_off": 15, "litert_mtp_on": 12}
MAX_OUTPUT_TOKENS = 2_048
REPETITION_GUARD_LIMIT = 20
CSV_FIELDS = (
    "model", "id", "serial", "device_model", "generation_origin", "source_run", "batch",
    "raw_score", "raw_score_for_aggregate", "repaired_score", "repaired_score_for_aggregate",
    "raw_strict_valid", "repaired_strict_valid", "repair_success", "repair_kind",
    "source_contract_equal",
    "finish_reason", "native_output_tokens", "native_decode_tokens_per_second",
    "native_provider_call_ms", "raw_output_sha256", "repaired_express_sha256", "repair_error",
    "provider_status", "renderer_contract_valid", "screen_captured", "scrolled_screen_captured",
    "current_case_visible", "capture_failure", "failure_categories",
    "raw_input_file",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class SdkPythonContractMismatch(ValueError):
    """An SDK-accepted candidate violates the independent scoring contract."""

    def __init__(self, evidence: dict):
        self.evidence = evidence
        super().__init__(f"SDK_PYTHON_CONTRACT_MISMATCH {evidence['model']}/{evidence['id']}: {evidence['reason']}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    need(path.is_file(), f"missing input: {path}")
    return trained.read_jsonl(path)


def index_rows(rows: list[dict], label: str) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        need(row.get("model") in MODES, f"{label}: unknown model {row.get('model')!r}")
        key = (row["model"], row.get("id"))
        need(key not in indexed, f"{label}: duplicate {key}")
        indexed[key] = row
    return indexed


def number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def fmt(value: object) -> str:
    value = number(value)
    return "—" if value is None else f"{value:.2f}"


def score(source: str, express: str) -> tuple[float, dict, dict]:
    metrics = quality.score_prediction(source, None, express, metric_version="v5_4")
    breakdown = dataclasses.asdict(quality.generation_reward_v5_4(express, source))
    reward = metrics["generation_reward_v5_4"]
    need(isinstance(reward, (int, float)) and math.isfinite(reward)
         and 0 <= reward <= 100 and reward == breakdown["quality_0_100"],
         "v5.4 score and generation_reward_v5_4 disagree")
    return float(reward), metrics, breakdown


def prior_rows() -> tuple[dict, dict]:
    summary = quality.read_json(PRIOR / "summary.json")
    need(summary.get("repair_library") == "GenUICraft 0.5.6"
         and summary.get("repair_api") == "GenUiCompiler.compileWithRepair(raw, null, false, true)",
         "historical checkpoint repair policy differs")
    old_native = {}
    checkpoints = {}
    for mode in MODES:
        rows = read_rows(PRIOR / "scored" / f"{mode}.jsonl")
        need(len(rows) == EXPECTED_PRIOR_COUNTS[mode], f"historical {mode} count differs")
        old_native[mode] = {row["id"]: row for row in rows}
        need(len(old_native[mode]) == len(rows), f"historical {mode} duplicates")
    for model in CHECKPOINTS:
        rows = read_rows(PRIOR / "scored" / f"{model}.jsonl")
        need(len(rows) == 50 and len({row["id"] for row in rows}) == 50,
             f"historical {model} is not a complete 50-case baseline")
        checkpoints[model] = {row["id"]: row for row in rows}
    return old_native, checkpoints


def provenance(run_dir: Path) -> tuple[dict, list[Path]]:
    files = [run_dir / name for name in ("protocol.json", "provenance.json", "repair_provenance.json")]
    files += sorted((run_dir / "preflight").glob("*.json")) if (run_dir / "preflight").is_dir() else []
    captured = {}
    for path in files:
        if path.is_file():
            value = quality.read_json(path)
            need(isinstance(value, dict), f"{path}: expected JSON object")
            captured[path.relative_to(run_dir).as_posix()] = value
    repair = captured.get("repair_provenance.json")
    need(isinstance(repair, dict) and bool(repair),
         "repair_provenance.json is required to verify the published GenUICraft 0.5.6 AAR")
    aar_sha = repair.get("aar_sha256") or repair.get("aarSha256")
    need(aar_sha == EXPECTED_AAR_SHA256,
         "repair provenance AAR SHA-256 differs from published GenUICraft 0.5.6")
    for key in ("source_text_supplied_to_repair", "fallback_enabled", "source_fallback_enabled", "new_model_inference"):
        need(repair.get(key) is not True, f"repair provenance reports {key}=true")
    for name, value in captured.items():
        if name.startswith("preflight/"):
            if "model_device_sha256" in value:
                words = str(value["model_device_sha256"] or "").split()
                need(bool(words) and words[0] == EXPECTED_MODEL_SHA256,
                     f"{name}: model device checksum missing or different")
            if name.endswith("model_copy.json"):
                need(value.get("sha256") == EXPECTED_MODEL_SHA256,
                     f"{name}: copied model checksum missing or different")
    return captured, [path for path in files if path.is_file()]


def native_fields(row: dict) -> dict:
    runtime = row.get("runtime")
    need(isinstance(runtime, dict), f"{row.get('model')}/{row.get('id')}: runtime metrics missing")
    output_tokens = number(runtime.get("actualOutputTokens"))
    if output_tokens is None:
        output_tokens = number(runtime.get("outputTokens"))
    return {
        "native_output_tokens": output_tokens,
        "native_decode_tokens_per_second": number(runtime.get("decodeTokensPerSecond")),
        "native_provider_call_ms": number(runtime.get("providerCallMs")),
    }


def failure_categories(row: dict) -> list[str]:
    """Keep SDK repair, native failure, and screenshot evidence independent."""
    result = row["provider_result"]
    categories = []
    if row["raw_output"] is None:
        categories.append("RUNTIME_NO_OUTPUT")
    elif row["repair_success"] is False:
        categories.append("SDK_REPAIR_REJECTED")
    if row["capture_failure"]:
        categories.append("SCREEN_CAPTURE_FAILED")
    elif result.get("status") == "render_invalid":
        categories.append("ANDROID_RENDER_INVALID_OTHER")
    elif result.get("status") not in (None, "valid") and not categories:
        categories.append("ANDROID_" + str(result["status"]).upper())
    return categories


def summarize(rows: list[dict]) -> dict:
    need(bool(rows), "cannot summarize empty population")
    raw = [row["raw_score_for_aggregate"] for row in rows]
    repaired = [row["repaired_score_for_aggregate"] for row in rows]
    native = [row for row in rows if row["raw_output"] is not None
              and row["finish_reason"] == "COMPLETED"
              and number(row["native_output_tokens"]) is not None
              and number(row["native_output_tokens"]) > 0
              and number(row["native_decode_tokens_per_second"]) is not None
              and number(row["native_decode_tokens_per_second"]) > 0]
    native_tokens = sum(row["native_output_tokens"] for row in native)
    native_seconds = sum(row["native_output_tokens"] / row["native_decode_tokens_per_second"]
                         for row in native)
    return {
        "n": len(rows),
        "raw_mean_failures_zero": statistics.mean(raw),
        "repaired_mean_failures_zero": statistics.mean(repaired),
        "raw_output_missing": sum(row["raw_output"] is None for row in rows),
        "raw_strict_valid": sum(row["raw_strict_valid"] is True for row in rows),
        "repair_accepted": sum(row["repair_success"] is True for row in rows),
        "repair_failed": sum(row["repair_success"] is False for row in rows),
        "repaired_strict_valid": sum(row["repaired_strict_valid"] is True for row in rows),
        "raw_repaired_source_contract_verified": sum(row["source_contract_equal"] is True for row in rows),
        "repair_kinds": dict(Counter(row["repair_kind"] for row in rows)),
        "finish_reason_counts": dict(Counter(str(row["finish_reason"] or "unavailable") for row in rows)),
        "repetition_guard_stopped": sum(row["finish_reason"] == "REPETITION_LIMIT" for row in rows),
        "repetition_guard_repair_accepted": sum(row["finish_reason"] == "REPETITION_LIMIT"
                                                 and row["repair_success"] is True for row in rows),
        "repetition_guard_repair_rejected": sum(row["finish_reason"] == "REPETITION_LIMIT"
                                                 and row["repair_success"] is False for row in rows),
        "at_or_above_output_cap": sum((number(row["native_output_tokens"]) or 0) >= MAX_OUTPUT_TOKENS
                                      for row in rows),
        "failure_category_counts": dict(Counter(category for row in rows
                                                for category in row["failure_categories"])),
        "screen_capture_failed": sum(row["capture_failure"] for row in rows),
        "origins": dict(Counter(row["generation_origin"] for row in rows)),
        "native_decode": {
            "method": "sum(native output tokens) / sum(native output tokens / native decode tokens per second)",
            "tokens_per_second_weighted": native_tokens / native_seconds if native_seconds else None,
            "completed_measured_cases": len(native),
            "output_tokens": native_tokens,
            "estimated_decode_seconds_from_native_rates": native_seconds,
        },
    }


def load_and_score(run_dir: Path) -> tuple[dict, dict, dict, dict, list[Path]]:
    input_path = run_dir / "model_inputs.jsonl"
    repair_path = run_dir / "repair_outcomes.jsonl"
    inputs = index_rows(read_rows(input_path), "model inputs")
    repairs = index_rows(read_rows(repair_path), "repair outcomes")
    frozen = {row["id"]: trained.expected_source(row) for row in trained.frozen_index()}
    expected = {(mode, case_id) for mode in MODES for case_id in frozen}
    need(set(inputs) == expected, f"model inputs: expected exactly 50 unique IDs/mode; missing={sorted(expected-set(inputs))}, extra={sorted(set(inputs)-expected)}")
    need(set(repairs) == expected, f"repair outcomes: expected same 100 keys; missing={sorted(expected-set(repairs))}, extra={sorted(set(repairs)-expected)}")
    old_native, checkpoints = prior_rows()
    captured_provenance, provenance_files = provenance(run_dir)
    scored = {mode: [] for mode in MODES}
    identity = None
    origin_counts = Counter()
    serial_cases: dict[str, set[str]] = {}
    serial_models: dict[str, str] = {}
    for case_id in frozen:
        source = frozen[case_id]
        for mode in MODES:
            key = (mode, case_id)
            item, repair = inputs[key], repairs[key]
            prefix = f"{mode}/{case_id}"
            need(item.get("query") == source["query"]
                 and item.get("response_text") == source["text"],
                 f"{prefix}: source or query differs from frozen corpus")
            need(isinstance(item.get("serial"), str) and item["serial"]
                 and isinstance(item.get("device_model"), str) and item["device_model"],
                 f"{prefix}: device identity missing")
            peer = inputs[(MODES[1] if mode == MODES[0] else MODES[0], case_id)]
            need(item["serial"] == peer.get("serial")
                 and item["device_model"] == peer.get("device_model"),
                 f"{case_id}: MTP modes ran on different devices")
            need(item.get("generation_origin") in ("prior_verified", "new_run"),
                 f"{prefix}: generation_origin must be prior_verified or new_run")
            need(isinstance(item.get("source_run"), str) and item["source_run"],
                 f"{prefix}: source_run missing")
            serial = item["serial"]
            serial_cases.setdefault(serial, set()).add(case_id)
            if serial in serial_models:
                need(serial_models[serial] == item["device_model"], f"{serial}: device model changed")
            serial_models[serial] = item["device_model"]
            raw = item.get("raw_output")
            raw_sha = item.get("raw_output_sha256")
            need(raw is None or isinstance(raw, str), f"{prefix}: raw_output must be text or null")
            need((raw is None and raw_sha is None) or
                 (raw is not None and raw_sha == quality.sha_text(raw)),
                 f"{prefix}: raw output hash mismatch")
            if item["generation_origin"] == "prior_verified":
                earlier = old_native[mode].get(case_id)
                need(earlier is not None and raw is not None
                     and raw_sha == earlier.get("raw_output_sha256")
                     and raw == earlier.get("raw_output"),
                     f"{prefix}: reused output differs from archived verified generation")
            origin_counts[(mode, item["generation_origin"])] += 1
            for flag in ("source_text_supplied_to_repair", "source_fallback_enabled"):
                need(repair.get(flag) is not True, f"{prefix}: repair used {flag}")
            need(repair.get("model_calls") in (None, 0), f"{prefix}: repair invoked a model")
            kind = repair.get("repair_kind")
            success = repair.get("repair_success")
            need(type(success) is bool, f"{prefix}: repair_success must be boolean")
            need(kind != "SOURCE_TEXT_FALLBACK", f"{prefix}: source fallback is forbidden")
            if raw is None:
                need(not success and kind == "RUNTIME_NO_OUTPUT"
                     and repair.get("repaired_express") in (None, "")
                     and repair.get("repaired_a2ui_json") in (None, ""),
                     f"{prefix}: runtime failure must have explicit RUNTIME_NO_OUTPUT without repair text")
            elif success:
                need(kind in ("NONE", "STRUCTURAL", "GENERATED_DSL_REPAIR")
                     and isinstance(repair.get("repaired_express"), str)
                     and bool(repair["repaired_express"])
                     and isinstance(repair.get("repaired_a2ui_json"), str)
                     and bool(repair["repaired_a2ui_json"]),
                     f"{prefix}: accepted repair is incomplete")
            else:
                need(kind == "REJECTED"
                     and repair.get("repaired_express") in (None, ""),
                     f"{prefix}: failed repair contains a generated candidate")
            raw_score = raw_metrics = raw_breakdown = None
            if raw is not None:
                raw_score, raw_metrics, raw_breakdown = score(source["text"], raw)
            repaired_text = repair.get("repaired_express") if success else None
            repaired_score = repaired_metrics = repaired_breakdown = None
            source_contract_equal = None
            if success:
                repaired_score, repaired_metrics, repaired_breakdown = score(source["text"], repaired_text)
                if repaired_metrics["schema_valid_strict"] is not True:
                    raise SdkPythonContractMismatch({
                        "model": mode, "id": case_id,
                        "reason": "SDK accepted repaired Express, but the independent Python strict validator rejected it",
                        "repair_kind": kind,
                        "python_schema_error": repaired_metrics.get("schema_error"),
                        "raw_output_sha256": raw_sha,
                        "repaired_express_sha256": quality.sha_text(repaired_text),
                    })
                raw_contract = raw_metrics["metric_identity_v5_4"]
                repaired_contract = repaired_metrics["metric_identity_v5_4"]
                contract_keys = ("source_hash", "expected_contract_hash")
                source_contract_equal = all(
                    raw_contract.get(key) is not None
                    and raw_contract.get(key) == repaired_contract.get(key)
                    for key in contract_keys
                )
                if not source_contract_equal:
                    raise SdkPythonContractMismatch({
                        "model": mode, "id": case_id,
                        "reason": "raw and repaired v5.4 source contracts differ or are unavailable",
                        "repair_kind": kind,
                        "raw_contract": {key: raw_contract.get(key) for key in contract_keys},
                        "repaired_contract": {key: repaired_contract.get(key) for key in contract_keys},
                        "raw_output_sha256": raw_sha,
                        "repaired_express_sha256": quality.sha_text(repaired_text),
                    })
            for metrics in (raw_metrics, repaired_metrics):
                if metrics:
                    current = metrics["metric_identity_v5_4"]
                    marker = (current["metric_version"], current["metric_fingerprint"],
                              current["reward_pipeline_fingerprint"])
                    if identity is None:
                        identity = marker
                    need(marker == identity, f"{prefix}: v5.4 scorer identity changed")
            native = native_fields(item)
            provider_result = item.get("provider_result")
            need(isinstance(provider_result, dict), f"{prefix}: provider_result must be an object")
            capture_failure = (
                provider_result.get("status") == "render_invalid"
                and provider_result.get("strictValid") is True
                and provider_result.get("rendererContractValid") is True
                and any(provider_result.get(field) is False for field in
                        ("screenCaptured", "scrolledScreenCaptured", "currentCaseVisible"))
            )
            row = {
                "model": mode, "id": case_id, "query": source["query"],
                "response_text": source["text"], "source_sha256": quality.sha_text(source["text"]),
                "serial": serial, "device_model": item["device_model"],
                "generation_origin": item["generation_origin"], "source_run": item["source_run"],
                "batch": item.get("batch"), "finish_reason": item.get("finish_reason"),
                "provider_result": provider_result, "runtime": item["runtime"],
                "provider_status": provider_result.get("status"),
                "renderer_contract_valid": provider_result.get("rendererContractValid"),
                "screen_captured": provider_result.get("screenCaptured"),
                "scrolled_screen_captured": provider_result.get("scrolledScreenCaptured"),
                "current_case_visible": provider_result.get("currentCaseVisible"),
                "capture_failure": capture_failure,
                "raw_input_file": item.get("raw_input_file"), "raw_output": raw,
                "raw_output_sha256": raw_sha, "raw_score": raw_score,
                "raw_score_for_aggregate": raw_score if raw_score is not None else 0.0,
                "raw_strict_valid": raw_metrics["schema_valid_strict"] if raw_metrics else None,
                "raw_metrics": raw_metrics, "raw_breakdown": raw_breakdown,
                "repair_success": success, "repair_kind": kind,
                "repair_error": repair.get("repair_error"), "repair_diagnostics": repair.get("diagnostics"),
                "repaired_express": repaired_text,
                "repaired_a2ui_json": repair.get("repaired_a2ui_json") if success else None,
                "repaired_express_sha256": quality.sha_text(repaired_text) if repaired_text else None,
                "repaired_score": repaired_score,
                "repaired_score_for_aggregate": repaired_score if repaired_score is not None else 0.0,
                "repaired_strict_valid": repaired_metrics["schema_valid_strict"] if repaired_metrics else None,
                "source_contract_equal": source_contract_equal,
                "repaired_metrics": repaired_metrics, "repaired_breakdown": repaired_breakdown,
                **native,
            }
            row["failure_categories"] = failure_categories(row)
            scored[mode].append(row)
    need(len(serial_cases) == 2, f"expected two devices, found {len(serial_cases)}")
    sizes = sorted(len(ids) for ids in serial_cases.values())
    need(sizes == [19, 31], f"expected 31/19 paired device split, found {sizes}")
    flip = serial_cases.get("R3GL203AKSF")
    need(flip == {f"BXP-{n:03d}" for n in range(1, 32)},
         "Flip8 must own BXP-001 through BXP-031 in both modes")
    for mode in MODES:
        need(origin_counts[(mode, "prior_verified")] == EXPECTED_PRIOR_COUNTS[mode],
             f"{mode}: prior verified count differs")
        reused_ids = {r["id"] for r in scored[mode] if r["generation_origin"] == "prior_verified"}
        need(reused_ids == set(old_native[mode]), f"{mode}: reused ID set differs from archive")
    need(sum(v for (mode, origin), v in origin_counts.items() if origin == "prior_verified") == 27,
         "expected 27 reused generations")
    prior_metric = quality.read_json(PRIOR / "summary.json")["metric_identity"]
    need(identity == (prior_metric["metric_version"], prior_metric["metric_fingerprint"],
                      prior_metric["reward_pipeline_fingerprint"]),
         "fresh v5.4 scorer differs from checkpoint baseline")
    return scored, checkpoints, captured_provenance, serial_cases, [input_path, repair_path, *provenance_files]


def make_summary(scored: dict, checkpoints: dict, captured: dict, serial_cases: dict,
                 input_paths: list[Path]) -> dict:
    baselines = {}
    for model in CHECKPOINTS:
        rows = list(checkpoints[model].values())
        baselines[model] = {
            "n": 50,
            "raw_mean_failures_zero": statistics.mean(r["raw_score"] for r in rows),
            "repaired_mean_failures_zero": statistics.mean(r["repaired_score_for_aggregate"] for r in rows),
            "raw_strict_valid": sum(r["raw_strict_valid"] is True for r in rows),
            "repair_accepted": sum(r["repair_success"] is True for r in rows),
            "repaired_strict_valid": sum(r["repaired_strict_valid"] is True for r in rows),
        }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "population": "100 evaluated native attempts: 27 archived verified generations and 73 new generations",
        "collection_dates": {"prior_verified": "2026-09-28", "new_run": "2026-09-29"},
        "repair_library": "published GenUICraft 0.5.6",
        "repair_api": "GenUiCompiler.compileWithRepair(raw, null, false, true)",
        "repair_policy": "generated output only; no source fallback; repair failures and runtime no-output contribute zero to each 50-case mean",
        "metric_identity": quality.read_json(PRIOR / "summary.json")["metric_identity"],
        "model_sha256_expected": EXPECTED_MODEL_SHA256,
        "aar_sha256_expected": EXPECTED_AAR_SHA256,
        "aar_sha256_verified": captured["repair_provenance.json"].get("aar_sha256")
        or captured["repair_provenance.json"].get("aarSha256"),
        "modes": {mode: summarize(scored[mode]) for mode in MODES},
        "devices": {serial: {
            "device_model": scored[MODES[0]][next(i for i, r in enumerate(scored[MODES[0]]) if r["serial"] == serial)]["device_model"],
            "ids": sorted(ids), "paired_case_count": len(ids),
            "modes": {mode: summarize([r for r in scored[mode] if r["serial"] == serial])
                      for mode in MODES},
        } for serial, ids in serial_cases.items()},
        "checkpoint_baselines": baselines,
        "provenance": captured,
        "input_hashes": {str(path.resolve()): sha256_file(path) for path in [
            *input_paths, PRIOR / "summary.json", *(PRIOR / "scored" / f"{model}.jsonl"
                                               for model in (*MODES, *CHECKPOINTS)),
            trained.FROZEN_CORPUS, Path(__file__),
        ]},
    }
    summary["paired"] = {
        "n": 50,
        "raw_on_minus_off_mean": summary["modes"][MODES[1]]["raw_mean_failures_zero"] - summary["modes"][MODES[0]]["raw_mean_failures_zero"],
        "repaired_on_minus_off_mean": summary["modes"][MODES[1]]["repaired_mean_failures_zero"] - summary["modes"][MODES[0]]["repaired_mean_failures_zero"],
    }
    for device in summary["devices"].values():
        device["repaired_on_minus_off_mean"] = (
            device["modes"][MODES[1]]["repaired_mean_failures_zero"]
            - device["modes"][MODES[0]]["repaired_mean_failures_zero"]
        )
    return summary


def failure_records(scored: dict) -> list[dict]:
    return [row for mode in MODES for row in scored[mode] if row["failure_categories"]]


def markdown(summary: dict, scored: dict) -> str:
    modes = summary["modes"]
    baselines = summary["checkpoint_baselines"]
    rows = []
    for model in (*MODES, *CHECKPOINTS):
        data = modes.get(model) or baselines[model]
        rows.append([quality.LABELS[model], data["n"], fmt(data["raw_mean_failures_zero"]),
                     fmt(data["repaired_mean_failures_zero"]), data["raw_strict_valid"],
                     data["repair_accepted"], data["repaired_strict_valid"]])
    device_rows = []
    for serial, device in summary["devices"].items():
        off, on = (device["modes"][mode] for mode in MODES)
        device_rows.append([f"{device['device_model']} ({serial})", device["paired_case_count"],
                            fmt(off["repaired_mean_failures_zero"]),
                            fmt(on["repaired_mean_failures_zero"]),
                            fmt(device["repaired_on_minus_off_mean"]),
                            f"{off['origins'].get('prior_verified', 0)} / {on['origins'].get('prior_verified', 0)}"])
    per_case = []
    for off, on in zip(scored[MODES[0]], scored[MODES[1]]):
        per_case.append([off["id"], off["device_model"],
                         f"{fmt(off['raw_score'])} → {fmt(off['repaired_score_for_aggregate'])}",
                         f"{fmt(on['raw_score'])} → {fmt(on['repaired_score_for_aggregate'])}",
                         f"{off['generation_origin']} / {on['generation_origin']}"])
    speeds = quality.table(["Mode", "Completed measured cases", "Native output tokens", "Token-weighted native decode tokens/s"],
                           [[quality.LABELS[m], modes[m]["native_decode"]["completed_measured_cases"],
                             modes[m]["native_decode"]["output_tokens"],
                             fmt(modes[m]["native_decode"]["tokens_per_second_weighted"])] for m in MODES])
    device_speeds = quality.table(
        ["Device", "Mode", "Completed measured cases", "Token-weighted native decode tokens/s"],
        [[f"{device['device_model']} ({serial})", "off" if mode == MODES[0] else "on",
          device["modes"][mode]["native_decode"]["completed_measured_cases"],
          fmt(device["modes"][mode]["native_decode"]["tokens_per_second_weighted"])]
         for serial, device in summary["devices"].items() for mode in MODES],
    )
    origin_off, origin_on = (modes[m]["origins"] for m in MODES)
    reasons = sorted(set(modes[MODES[0]]["finish_reason_counts"])
                     | set(modes[MODES[1]]["finish_reason_counts"]))
    finish_rows = [[reason, modes[MODES[0]]["finish_reason_counts"].get(reason, 0),
                    modes[MODES[1]]["finish_reason_counts"].get(reason, 0)]
                   for reason in reasons]
    failure_rows = []
    for item in failure_records(scored):
        effect = "zero: no repaired artifact" if not item["repair_success"] else "v5.4 Express score retained"
        failure_rows.append([
            item["id"], "off" if item["model"] == MODES[0] else "on", item["device_model"],
            ", ".join(item["failure_categories"]), item["finish_reason"] or "unavailable",
            len(item["raw_output"]) if item["raw_output"] is not None else "no output",
            item["native_output_tokens"] if item["native_output_tokens"] is not None else "—",
            item["provider_status"] or "—", item["repair_kind"], effect,
        ])
    if not failure_rows:
        failure_rows = [["none", "—", "—", "—", "—", "—", "—", "—", "—", "—"]]
    return "\n".join([
        "# Bixby50 full native scores after GenUICraft 0.5.6 repair", "",
        f"Generated: {summary['created_at_utc']}", "",
        "This report scores **100 native attempts in 50 pairs**: 27 verified outputs collected on September 28 and 73 newly generated outputs from September 29. Both modes cover the same 50 frozen Bixby50 cases, once per mode, with each case's MTP-off and MTP-on runs on the same device. Every captured raw output was sent through the same published GenUICraft 0.5.6 generated-output repair; any runtime no-output attempt has an explicit failure record. No source text or fallback was supplied to repair.", "",
        quality.table(["Population", "Cases", "Raw mean /100", "After repair mean /100", "Raw strict", "Repair accepted", "Repaired strict"], rows), "",
        f"The full paired repaired mean difference (on minus off) is **{summary['paired']['repaired_on_minus_off_mean']:+.2f} points**. These are v5.4 source-to-UI representation-quality scores, not factual-answer accuracy or visual certification. A failed repair contributes zero to the all-attempt mean; missing runtime output has null individual scores and also contributes zero to the aggregate.", "",
        "## Hardware and generation origin", "",
        quality.table(["Device", "Paired cases", "Off repaired mean", "On repaired mean", "On − off", "Reused off / on"], device_rows), "",
        f"Generation origins: MTP off {origin_off.get('prior_verified', 0)} prior verified and {origin_off.get('new_run', 0)} new; MTP on {origin_on.get('prior_verified', 0)} prior verified and {origin_on.get('new_run', 0)} new. The archived 27 raw outputs were checked byte for byte against their earlier saved records. Each case's source response and query were checked against the frozen Bixby50 corpus.", "",
        "## Native speed", "", speeds, "", device_speeds, "",
        "Decode throughput uses only completed cases with native output-token counts and native decode-token rates. It is computed as total native output tokens divided by the sum of per-case token-count/rate times. It excludes engine initialization, prefill, screen capture, and host repair time. These rows mix archived September 28 and new September 29 generations and were not collected under controlled, equal thermal conditions. They are descriptive diagnostics, not a causal MTP speed comparison; different output lengths and device heat also affect the rate.", "",
        "## Generation finish and failures", "",
        quality.table(["Finish reason", "MTP off", "MTP on"], finish_rows), "",
        f"The native repetition guard stopped {modes[MODES[0]]['repetition_guard_stopped']} off and {modes[MODES[1]]['repetition_guard_stopped']} on generations after its fixed {REPETITION_GUARD_LIMIT}-reference limit. Of those, SDK repair accepted {modes[MODES[0]]['repetition_guard_repair_accepted']} off and {modes[MODES[1]]['repetition_guard_repair_accepted']} on; the remaining guard-stopped outputs were rejected. A repetition stop alone does not assign a zero score. The unchanged native output cap was {MAX_OUTPUT_TOKENS:,} tokens; observed outputs at or above that cap were {modes[MODES[0]]['at_or_above_output_cap']} off and {modes[MODES[1]]['at_or_above_output_cap']} on.", "",
        "The table below lists each failed SDK repair, runtime no-output attempt, or Android render/capture failure separately. `SCREEN_CAPTURE_FAILED` means Android reported a valid strict/renderer contract but did not confirm a screenshot or visible case. That affects screenshot evidence, not the host v5.4 Express score. `SDK_REPAIR_REJECTED` means the published SDK returned no generated-only candidate; its repaired aggregate contribution is zero. Full errors and capture flags remain in scored JSONL and the interactive case browser.", "",
        quality.table(["Case", "Mode", "Device", "Failure evidence", "Finish", "Raw chars", "Native tokens", "Android status", "SDK repair", "Score effect"], failure_rows), "",
        "## Repair and measurement policy", "",
        "`GenUiCompiler.compileWithRepair(raw, null, false, true)` receives only the generated Express text. `NONE`, `STRUCTURAL`, and `GENERATED_DSL_REPAIR` are accepted routes. `REJECTED` and `RUNTIME_NO_OUTPUT` provide no repaired candidate. The raw output is scored unchanged; accepted repaired Express is scored unchanged with `score_prediction(source, None, text, metric_version='v5_4')` and cross-checked against `generation_reward_v5_4`. Output-free runtime failures are explicitly labeled and retained in the denominator.", "",
        "The model, frozen prompt, 20-reference repetition guard, 2,048-token output cap, temperature zero, and no-source-fallback policy stayed fixed. This scoring pass performs no holdout tuning or response regeneration.", "",
        "The checkpoint R32/R64 rows come from the previous complete 50-case published-AAR repair report. They are comparison populations, not new device generations. The 27 archived LiteRT raw outputs are included here: every selected output was repaired and rescored exactly once into its new 50-case mode mean. The earlier partial LiteRT aggregate means were never averaged into this report.", "",
        f"Expected model SHA-256: `{summary['model_sha256_expected']}`. Verified published AAR SHA-256: `{summary['aar_sha256_verified']}`. Metric fingerprint: `{summary['metric_identity']['metric_fingerprint']}`. Raw/repaired source-contract equality was verified for every accepted repair; a mismatch aborts scoring and creates `contract_mismatch.json` for review.", "",
        "## Per-case scores", "",
        "Each cell is raw → after repair. A repaired zero with no artifact is the stated aggregate policy; inspect the JSONL record for its null individual score and error.", "",
        quality.table(["Case", "Device", "MTP off", "MTP on", "Off / on origin"], per_case), "",
        "## Artifacts", "",
        "[Interactive case browser](index.html) · [Summary JSON](summary.json) · [Per-case CSV](per_case_scores.csv) · [Scored MTP-off JSONL](scored/litert_mtp_off.jsonl) · [Scored MTP-on JSONL](scored/litert_mtp_on.jsonl) · [Input hashes](scoring_input_hashes.json).", "",
        "Repair success and a high v5.4 score do not establish complete source fidelity. MTP may change both output text and generation time; this single paired pass does not establish a statistically stable quality difference.", "",
    ])


def webpage(summary: dict, scored: dict) -> str:
    e = html.escape
    cards = []
    failure_table_rows = []
    for item in failure_records(scored):
        effect = "zero: SDK returned no artifact" if not item["repair_success"] else "v5.4 Express score retained"
        failure_table_rows.append(
            f'<tr><td>{e(item["id"])}</td><td>{"off" if item["model"] == MODES[0] else "on"}</td>'
            f'<td>{e(", ".join(item["failure_categories"]))}</td>'
            f'<td>{e(str(item["finish_reason"] or "unavailable"))}</td>'
            f'<td>{e(str(item["provider_status"] or "—"))}</td><td>{e(effect)}</td></tr>'
        )
    failure_table = ''.join(failure_table_rows) if failure_table_rows else '<tr><td colspan="6">None recorded</td></tr>'
    finish_note = '; '.join(
        f'{label}: ' + ', '.join(f'{e(reason)} {count}' for reason, count in
                               summary["modes"][mode]["finish_reason_counts"].items())
        for mode, label in ((MODES[0], "MTP off"), (MODES[1], "MTP on"))
    )
    for off, on in zip(scored[MODES[0]], scored[MODES[1]]):
        case_id = off["id"]
        halves = []
        for item, label in ((off, "MTP off"), (on, "MTP on")):
            raw = item["raw_output"] if item["raw_output"] is not None else "No native output captured."
            repaired = item["repaired_express"] if item["repaired_express"] is not None else "No repaired artifact returned."
            error = item["repair_error"] or (item["provider_result"] or {}).get("error") or ""
            halves.append(
                f'<section class="mode"><h3>{label}</h3>'
                f'<p class="score">{fmt(item["raw_score"])} → {fmt(item["repaired_score_for_aggregate"])} <small>/100</small></p>'
                f'<p class="meta">{e(item["generation_origin"])} · {e(item["repair_kind"] or "unavailable")} · '
                f'{e(item["device_model"])} · {e(item["serial"])}</p>'
                f'<details><summary>Raw output</summary><pre>{e(raw)}</pre></details>'
                f'<details><summary>Repaired output</summary><pre>{e(repaired)}</pre></details>'
                f'<details><summary>Errors and provenance</summary><pre>{e(json.dumps({"repair_error": error, "diagnostics": item["repair_diagnostics"], "source_run": item["source_run"], "raw_input_file": item["raw_input_file"], "finish_reason": item["finish_reason"], "runtime": item["runtime"]}, ensure_ascii=False, indent=2))}</pre></details>'
                '</section>')
        searchable = e(f"{case_id} {off['query']} {off['device_model']} {off['serial']}".lower(), quote=True)
        cards.append(f'<article class="case" data-search="{searchable}"><h2>{case_id}</h2><p>{e(off["query"])}</p>'
                     f'<details><summary>Original source answer (not passed to repair)</summary><pre>{e(off["response_text"])}</pre></details>'
                     f'<div class="modes">{"".join(halves)}</div></article>')
    off, on = (summary["modes"][mode] for mode in MODES)
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bixby50 full native repaired scores</title><style>
:root{{font:15px/1.5 system-ui,sans-serif;color:#17243a;background:#f3f6fb}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1600px;margin:auto;padding:24px}}header,.case{{background:#fff;border:1px solid #dce3ee;border-radius:14px;padding:22px;margin:0 0 16px}}h1{{margin:0 0 8px;font-size:29px}}h2{{font-size:20px;margin:0 0 8px}}h3{{font-size:17px;margin:0}}.lead{{color:#465972}}.totals,.modes{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.total,.mode{{border:1px solid #dce3ee;border-radius:10px;padding:16px;min-width:0}}.total strong,.score{{font-size:24px;font-weight:750;margin:4px 0}}.meta{{color:#53627a;font-size:13px;overflow-wrap:anywhere}}input{{width:min(640px,100%);padding:12px;border:1px solid #9daec5;border-radius:8px;font:inherit;margin:10px 0 22px}}details{{margin:12px 0}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:520px;overflow:auto;padding:12px;background:#f5f7fa;border-radius:8px;font:12px/1.45 Consolas,monospace}}small{{font-size:14px;color:#526179}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:7px;border-bottom:1px solid #dce3ee;vertical-align:top}}.tablewrap{{overflow-x:auto}}@media(max-width:850px){{.totals,.modes{{grid-template-columns:1fr}}main{{padding:10px}}}}
</style><main><header><h1>Bixby50 native output after SDK 0.5.6 repair</h1><p class="lead">100 native attempts in 50 pairs: 27 archived verified generations from September 28 + 73 new generations from September 29. Same 50 frozen source cases in both modes; 31 cases on Flip8 and 19 on Fold7. Failed repair and runtime no-output attempts remain in each 50-case aggregate.</p><div class="totals"><div class="total">MTP off<br><strong>{off['raw_mean_failures_zero']:.2f} → {off['repaired_mean_failures_zero']:.2f}</strong><br>raw → repaired /100</div><div class="total">MTP on<br><strong>{on['raw_mean_failures_zero']:.2f} → {on['repaired_mean_failures_zero']:.2f}</strong><br>raw → repaired /100</div></div><p class="lead">V5.4 measures source-to-UI representation quality; it is not a factual-answer or visual-quality certificate. Speed data mixes September 28 and 29 runs under different thermal conditions and supports no causal claim. <a href="REPORT.md">Full report</a> · <a href="per_case_scores.csv">Per-case CSV</a> · <a href="summary.json">Summary JSON</a></p></header><section class="case"><h2>Generation finish and capture failures</h2><p>Finish reasons by mode: {finish_note}. A repetition guard stop is recorded independently of SDK repair success. Screenshot failure does not zero an accepted Express score.</p><div class="tablewrap"><table><thead><tr><th>Case</th><th>Mode</th><th>Failure evidence</th><th>Finish</th><th>Android status</th><th>Score effect</th></tr></thead><tbody>{failure_table}</tbody></table></div></section><label for="find">Filter cases by ID, query, device, or serial</label><br><input id="find" placeholder="BXP-001, weather, Flip8…" aria-label="Filter cases"><p id="count">Showing 50 cases</p>{''.join(cards)}</main><script>
const input=document.getElementById('find'),cards=[...document.querySelectorAll('article.case')],count=document.getElementById('count');input.addEventListener('input',()=>{{const q=input.value.trim().toLowerCase();let visible=0;for(const card of cards){{card.hidden=!card.dataset.search.includes(q);if(!card.hidden)visible++}}count.textContent=`Showing ${{visible}} of 50 cases`}});
</script></html>'''


def write_report(run_dir: Path, out: Path, scored: dict, summary: dict) -> None:
    need(out != run_dir and run_dir in out.parents,
         "output must be a child directory of --run-dir")
    targets = [out / name for name in ("summary.json", "scoring_input_hashes.json", "REPORT.md", "index.html", "per_case_scores.csv")]
    targets += [out / "scored" / f"{mode}.jsonl" for mode in MODES]
    need(not any(path.exists() for path in targets),
         "report target already exists; preserve earlier reports and choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    (out / "scored").mkdir(exist_ok=True)
    with (out / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (out / "scoring_input_hashes.json").open("x", encoding="utf-8") as stream:
        json.dump(summary["input_hashes"], stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for mode in MODES:
        with (out / "scored" / f"{mode}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in scored[mode]:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (out / "per_case_scores.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for mode in MODES:
            for row in scored[mode]:
                cells = {field: row.get(field) for field in CSV_FIELDS}
                cells["failure_categories"] = ";".join(row["failure_categories"])
                writer.writerow(cells)
    (out / "REPORT.md").write_text(markdown(summary, scored), encoding="utf-8", newline="\n")
    (out / "index.html").write_text(webpage(summary, scored), encoding="utf-8", newline="\n")


def write_contract_mismatch(run_dir: Path, out: Path, mismatch: SdkPythonContractMismatch) -> Path:
    need(out != run_dir and run_dir in out.parents,
         "contract mismatch evidence output must be a child directory of --run-dir")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "contract_mismatch.json"
    evidence = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "aborted_pending_manual_review",
        "code": "SDK_PYTHON_CONTRACT_MISMATCH",
        **mismatch.evidence,
        "inputs": {
            name: sha256_file(run_dir / name) for name in
            ("model_inputs.jsonl", "repair_outcomes.jsonl") if (run_dir / name).is_file()
        },
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Directory containing model_inputs.jsonl and repair_outcomes.jsonl")
    parser.add_argument("--out", type=Path,
                        help="New report directory (default: RUN_DIR/full_native_repaired_report)")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    out = (args.out or run_dir / "full_native_repaired_report").resolve()
    try:
        need(run_dir.is_dir(), f"run directory does not exist: {run_dir}")
        need(not (out / "summary.json").exists(),
             f"existing report summary must be preserved: {out / 'summary.json'}")
        scored, checkpoints, captured, serial_cases, inputs = load_and_score(run_dir)
        summary = make_summary(scored, checkpoints, captured, serial_cases, inputs)
        write_report(run_dir, out, scored, summary)
        print(json.dumps({"report": str(out), "modes": {
            mode: {"raw_mean": summary["modes"][mode]["raw_mean_failures_zero"],
                   "repaired_mean": summary["modes"][mode]["repaired_mean_failures_zero"]}
            for mode in MODES}}, indent=2))
        return 0
    except SdkPythonContractMismatch as exc:
        try:
            evidence_path = write_contract_mismatch(run_dir, out, exc)
            print(f"{exc}; evidence: {evidence_path}", file=sys.stderr)
        except Exception as write_error:
            print(f"{exc}; could not save mismatch evidence: {write_error}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"report_bixby50_full_native_repaired.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
