#!/usr/bin/env python3
"""Run exact prompts through the Android production pipeline and materialize a golden run."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DATASET_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
TEST_CLASS = "com.samsung.genuicraft.GoldenScenarioUiTest"
TEST_RUNNER = "com.samsung.genuicraft.test/androidx.test.runner.AndroidJUnitRunner"
APP_PACKAGE = "com.samsung.genuicraft"

sys.path.insert(0, str(DATASET_ROOT / "src"))
from pipeline.ir_formats.active import validate_express_completion  # noqa: E402


def utc_iso(epoch_ms: int | None = None) -> str:
    stamp = time.time() if epoch_ms is None else epoch_ms / 1000.0
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_command(
    command: list[str],
    *,
    timeout_sec: int = 60,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}"
        )
    return completed


def adb_prefix(adb_bin: str, serial: str) -> list[str]:
    return [adb_bin, "-s", serial]


def safe_id(value: Any, field: str) -> str:
    text = str(value or "")
    if not SAFE_ID.fullmatch(text):
        raise ValueError(f"{field} contains unsupported characters: {text!r}")
    return text


def ids_for(ordinal: int) -> tuple[str, str, str]:
    return (
        f"q_{ordinal:06d}",
        f"r_{ordinal:06d}_01",
        f"u_{ordinal:06d}_01",
    )


def case_dir(run_dir: Path, scenario: dict[str, Any]) -> Path:
    pair_id = safe_id(scenario["pair_id"], "pair_id")
    slug = safe_id(scenario["scenario_slug"], "scenario_slug")
    variant = safe_id(scenario["variant"], "variant")
    return run_dir / "cases" / f"{pair_id}_{slug}" / variant


def load_device_result(run_dir: Path, case_id: str) -> dict[str, Any] | None:
    path = run_dir / "device_results" / f"{case_id}.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid device result object: {path}")
    return value


def git_value(*args: str) -> str:
    result = run_command(["git", *args], timeout_sec=20)
    return result.stdout.strip() if result.returncode == 0 else ""


def capture_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "android_device_rendered" / "capture_manifest.jsonl"
    if not path.exists():
        return {}
    return {str(row.get("ui_id") or ""): row for row in read_jsonl(path)}


def visual_review_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "visual_review.jsonl"
    if not path.exists():
        return {}
    return {str(row.get("case_id") or ""): row for row in read_jsonl(path)}


def issue(
    scenario: dict[str, Any],
    issue_type: str,
    message: str,
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "case_id": scenario["case_id"],
        "pair_id": scenario["pair_id"],
        "variant": scenario["variant"],
        "severity": severity,
        "type": issue_type,
        "message": message,
    }


def materialize(run_dir: Path, scenarios: list[dict[str, Any]]) -> None:
    captures = capture_map(run_dir)
    visual_reviews = visual_review_map(run_dir)
    queries: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    genui_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    completed_count = 0
    pipeline_success_count = 0
    accepted_count = 0

    for scenario in scenarios:
        ordinal = int(scenario["ordinal"])
        query_id, response_id, ui_id = ids_for(ordinal)
        prompt = str(scenario["prompt"])
        output_dir = case_dir(run_dir, scenario)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        created_at = str(scenario.get("created_at") or utc_iso())
        queries.append(
            {
                "query_id": query_id,
                "intent": scenario["scenario_slug"],
                "query_text": prompt,
                "difficulty": "standard",
                "tags": ["golden40", scenario["pair_id"], scenario["variant"]],
                "created_at": created_at,
                "source": "android_genui_demo_golden",
                "gen": {
                    "case_id": scenario["case_id"],
                    "pair_id": scenario["pair_id"],
                    "variant": scenario["variant"],
                },
            }
        )

        result = load_device_result(run_dir, str(scenario["case_id"]))
        if result is None:
            continue
        completed_count += 1
        execution = dict(scenario)
        execution.update(result)
        execution["query_id"] = query_id
        execution["response_id"] = response_id
        execution["ui_id"] = ui_id
        capture = captures.get(ui_id)
        visual_review = visual_reviews.get(str(scenario["case_id"]))
        if capture is not None:
            execution["capture"] = capture
        execution_rows.append(execution)

        response_text = str(result.get("response_text") or "")
        ir_text = str(result.get("genui_json") or "")
        if not response_text:
            (output_dir / "response.md").unlink(missing_ok=True)
        if not ir_text:
            (output_dir / "ir.a2ui").unlink(missing_ok=True)
        if response_text:
            (output_dir / "response.md").write_text(response_text.rstrip() + "\n", encoding="utf-8")
        if ir_text:
            (output_dir / "ir.a2ui").write_text(ir_text.rstrip() + "\n", encoding="utf-8")

        validation = validate_express_completion(ir_text) if ir_text else None
        express_valid = bool(validation and validation.production_valid)
        semantic_hash = validation.semantic_hash if validation else None
        success = result.get("success") is True
        if success:
            pipeline_success_count += 1

        finished_at = utc_iso(int(result.get("finished_at_ms") or 0)) if result.get("finished_at_ms") else None
        if response_text:
            responses.append(
                {
                    "response_id": response_id,
                    "query_id": query_id,
                    "n_idx": 1,
                    "response_text": response_text,
                    "created_at": finished_at or created_at,
                    "assets": [],
                    "asset_stats": {},
                    "gen": {
                        "provider": result.get("response_provider"),
                        "model": result.get("response_model"),
                        "mcp_enabled": result.get("mcp_enabled"),
                        "case_id": scenario["case_id"],
                    },
                }
            )

        if success and express_valid:
            genui_rows.append(
                {
                    "ui_id": ui_id,
                    "response_id": response_id,
                    "query_id": query_id,
                    "intent": scenario["scenario_slug"],
                    "tags": ["golden40", scenario["pair_id"], scenario["variant"]],
                    "intent_bucket": scenario["scenario_slug"],
                    "response_text": response_text,
                    "genui_json": ir_text,
                    "ir_format": "a2ui_express_v1",
                    "validation": {
                        "valid": True,
                        "raw_valid": validation.raw_valid,
                        "repaired_valid": validation.repaired_valid,
                        "repair_applied": validation.repair_applied,
                        "errors": list(validation.errors),
                        "semantic_hash": semantic_hash,
                    },
                    "gen": {
                        "provider": result.get("ir_provider"),
                        "model": result.get("ir_model"),
                        "used_fallback": result.get("used_fallback"),
                        "warnings": result.get("warnings") or [],
                        "case_id": scenario["case_id"],
                    },
                    "created_at": finished_at or created_at,
                }
            )

        capture_ok = capture is not None and capture.get("ok") is True
        native_render_ok = capture is not None and capture.get("native_render_ok") is True
        selected_screenshot = ""
        viewport_screenshot = ""
        if capture is not None:
            source_dir = run_dir / "android_device_rendered"
            screenshot_name = str(capture.get("screenshot") or "")
            viewport_name = str(capture.get("viewport_screenshot") or "")
            if screenshot_name and (source_dir / screenshot_name).exists():
                shutil.copy2(source_dir / screenshot_name, output_dir / "app_screenshot.png")
                selected_screenshot = "app_screenshot.png"
            if viewport_name and (source_dir / viewport_name).exists():
                shutil.copy2(source_dir / viewport_name, output_dir / "viewport_screenshot.png")
                viewport_screenshot = "viewport_screenshot.png"

        render_error = str(result.get("render_error") or "")
        if not success:
            issues.append(issue(scenario, "pipeline_failure", str(result.get("error") or "Unknown pipeline failure")))
        if ir_text and not express_valid:
            errors = "; ".join(validation.errors) if validation else "Unknown validation failure"
            issues.append(issue(scenario, "express_validation", errors))
        if render_error:
            issues.append(issue(scenario, "pipeline_render_error", render_error))
        if result.get("used_fallback") is True:
            issues.append(issue(scenario, "pipeline_fallback", "Pipeline reported used_fallback=true", "warning"))

        noteworthy_updates: list[str] = []
        for update in result.get("stage_updates") or []:
            message = str(update.get("message") or "")
            lowered = message.lower()
            if any(marker in lowered for marker in ("fallback", "key missing", "returned an error", "data empty")):
                noteworthy_updates.append(message)
        for message in dict.fromkeys(noteworthy_updates):
            issues.append(issue(scenario, "mcp_or_pipeline_notice", message, "warning"))

        if capture is not None and not capture_ok:
            issues.append(
                issue(
                    scenario,
                    "device_capture_failure",
                    str(capture.get("native_render_error") or "Native renderer or screenshot capture failed"),
                )
            )

        visual_review_status = "pending"
        status = "pipeline_failed"
        if success and express_valid:
            status = "pipeline_pass"
            if capture is not None:
                status = (
                    "machine_render_pass"
                    if capture_ok and native_render_ok and selected_screenshot
                    else "render_failed"
                )
        if status == "machine_render_pass" and visual_review is not None:
            visual_review_status = str(visual_review.get("status") or "pending")
            if visual_review_status == "pass":
                status = "accepted"
            elif visual_review_status == "issues":
                status = "visual_issues"
                for review_issue in visual_review.get("issues") or []:
                    issues.append(
                        issue(
                            scenario,
                            str(review_issue.get("type") or "visual_issue"),
                            str(review_issue.get("message") or "Visual review found an issue"),
                            str(review_issue.get("severity") or "warning"),
                        )
                    )
        if status == "accepted":
            accepted_count += 1

        metadata = {
            "case_id": scenario["case_id"],
            "pair_id": scenario["pair_id"],
            "scenario": scenario["scenario_slug"],
            "variant": scenario["variant"],
            "attempt": 1,
            "query_id": query_id,
            "response_id": response_id,
            "ui_id": ui_id,
            "status": status,
            "timestamps": {
                "started_at": utc_iso(int(result.get("started_at_ms") or 0)) if result.get("started_at_ms") else None,
                "completed_at": finished_at,
            },
            "pipeline": {
                "response_provider": result.get("response_provider"),
                "response_model": result.get("response_model"),
                "ir_provider": result.get("ir_provider"),
                "ir_model": result.get("ir_model"),
                "gemini_api_mode": result.get("gemini_api_mode"),
                "mcp_enabled": result.get("mcp_enabled"),
                "used_fallback": result.get("used_fallback"),
                "warnings": result.get("warnings") or [],
                "stage3_input_tokens": result.get("stage3_input_tokens"),
                "stage3_output_tokens": result.get("stage3_output_tokens"),
                "stage_durations_ms": result.get("stage_durations_ms") or {},
            },
            "validation": {
                "express_valid": express_valid,
                "semantic_hash": semantic_hash,
                "native_render_ok": native_render_ok if capture is not None else None,
                "capture_ok": capture_ok if capture is not None else None,
                "visual_review": visual_review_status,
            },
            "artifacts": {
                "prompt": "prompt.txt",
                "response": "response.md" if response_text else None,
                "ir": "ir.a2ui" if ir_text else None,
                "manual_ui_screenshot": (
                    "manual_ui_screenshot.png"
                    if (output_dir / "manual_ui_screenshot.png").exists()
                    else None
                ),
                "ui_hierarchy": (
                    "ui_hierarchy.xml" if (output_dir / "ui_hierarchy.xml").exists() else None
                ),
                "device_logcat": (
                    "device_logcat.txt" if (output_dir / "device_logcat.txt").exists() else None
                ),
                "app_screenshot": selected_screenshot or None,
                "viewport_screenshot": viewport_screenshot or None,
                "canonical_screenshot": (
                    f"../../../android_device_rendered/{capture.get('screenshot')}"
                    if capture and capture.get("screenshot")
                    else None
                ),
            },
        }
        write_json(output_dir / "metadata.json", metadata)

    write_jsonl(run_dir / "queries.jsonl", queries)
    write_jsonl(run_dir / "responses.jsonl", responses)
    write_jsonl(run_dir / "genui.jsonl", genui_rows)
    write_jsonl(run_dir / "execution_results.jsonl", execution_rows)
    write_jsonl(run_dir / "issues.jsonl", issues)

    models = sorted(
        {
            (str(row.get("response_model") or ""), str(row.get("ir_model") or ""))
            for row in execution_rows
        }
    )
    manifest = {
        "manifest_version": 1,
        "generated_at": utc_iso(),
        "run_id": run_dir.name,
        "purpose": "Twenty regular/short GenUI Demo scenario pairs captured on Android",
        "planned_cases": len(scenarios),
        "completed_cases": completed_count,
        "pipeline_successes": pipeline_success_count,
        "accepted_cases": accepted_count,
        "models": [{"response": response, "ir": ir} for response, ir in models],
        "mcp_enabled_values": sorted({row.get("mcp_enabled") for row in execution_rows}, key=str),
        "repo": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "git_dirty": bool(git_value("status", "--porcelain")),
        },
        "devices": [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((run_dir / "device_snapshots").glob("*.json"))
        ] if (run_dir / "device_snapshots").exists() else [],
        "paths": {
            "scenarios": "scenarios.jsonl",
            "queries": "queries.jsonl",
            "responses": "responses.jsonl",
            "genui": "genui.jsonl",
            "execution_results": "execution_results.jsonl",
            "issues": "issues.jsonl",
            "screenshots": "android_device_rendered",
            "cases": "cases",
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(
        run_dir / "progress.json",
        {
            "updated_at": utc_iso(),
            "planned": len(scenarios),
            "completed": completed_count,
            "pipeline_successes": pipeline_success_count,
            "valid_ir": len(genui_rows),
            "captures": len(captures),
            "accepted": accepted_count,
            "issues": len(issues),
        },
    )


def run_case(
    args: argparse.Namespace,
    run_dir: Path,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    case_id = safe_id(scenario["case_id"], "case_id")
    run_id = safe_id(run_dir.name, "run_id")
    prompt = str(scenario["prompt"])
    encoded_prompt = base64.urlsafe_b64encode(prompt.encode("utf-8")).decode("ascii").rstrip("=")
    remote_path = (
        f"/sdcard/Android/data/{APP_PACKAGE}/files/golden_scenarios/{run_id}/{case_id}.json"
    )
    remote_screenshot = remote_path.removesuffix(".json") + "_ui.png"
    remote_hierarchy = remote_path.removesuffix(".json") + "_window.xml"
    prefix = adb_prefix(args.adb_bin, args.serial)
    run_command(
        [*prefix, "shell", "rm", "-f", remote_path, remote_screenshot, remote_hierarchy],
        timeout_sec=30,
        check=True,
    )
    run_command([*prefix, "shell", "am", "force-stop", APP_PACKAGE], timeout_sec=30, check=True)
    command = [
        *prefix,
        "shell",
        "am",
        "instrument",
        "-w",
        "-r",
        "-e",
        "class",
        TEST_CLASS,
        "-e",
        "run_id",
        run_id,
        "-e",
        "case_id",
        case_id,
        "-e",
        "prompt_b64",
        encoded_prompt,
        TEST_RUNNER,
    ]
    log_path = run_dir / "instrumentation_logs" / f"{case_id}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = case_dir(run_dir, scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    logcat_path = output_dir / "device_logcat.txt"
    logcat_handle = logcat_path.open("w", encoding="utf-8", newline="")
    logcat_process = subprocess.Popen(
        [
            *prefix,
            "logcat",
            "-v",
            "threadtime",
            "-T",
            "1",
            "GenUiStagePipeline:V",
            "McpLlmRouter:V",
            "McpClient:V",
            "ActivityTaskManager:I",
            "*:S",
        ],
        stdout=logcat_handle,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    host_started_ms = int(time.time() * 1000)
    try:
        completed = run_command(command, timeout_sec=args.timeout_sec)
        output = completed.stdout
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        output += f"\nHOST TIMEOUT after {args.timeout_sec} seconds\n"
        return_code = 124
    finally:
        logcat_process.terminate()
        try:
            logcat_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logcat_process.kill()
            logcat_process.wait(timeout=10)
        logcat_handle.close()
        redact_log_file(logcat_path)
    log_path.write_text(output, encoding="utf-8")

    local_path = run_dir / "device_results" / f"{case_id}.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    pull = run_command([*prefix, "pull", remote_path, str(local_path)], timeout_sec=60)
    run_command(
        [*prefix, "pull", remote_screenshot, str(output_dir / "manual_ui_screenshot.png")],
        timeout_sec=60,
    )
    run_command(
        [*prefix, "pull", remote_hierarchy, str(output_dir / "ui_hierarchy.xml")],
        timeout_sec=60,
    )
    if pull.returncode == 0 and local_path.exists():
        result = json.loads(local_path.read_text(encoding="utf-8"))
        if result.get("case_id") != case_id:
            raise RuntimeError(f"Pulled result case mismatch for {case_id}")
        if result.get("prompt") != prompt:
            raise RuntimeError(f"Pulled prompt mismatch for {case_id}")
        result["instrumentation_return_code"] = return_code
        result["instrumentation_log"] = f"instrumentation_logs/{case_id}.txt"
        result["device_serial"] = args.serial
        result["device_model"] = args.device_model
        write_json(local_path, result)
        return result

    failure = {
        "schema_version": 1,
        "run_id": run_id,
        "case_id": case_id,
        "prompt": prompt,
        "started_at_ms": host_started_ms,
        "finished_at_ms": int(time.time() * 1000),
        "elapsed_ms": int(time.time() * 1000) - host_started_ms,
        "success": False,
        "outcome_stage": "instrumentation",
        "error": f"Instrumentation return code {return_code}; no device result was pullable",
        "instrumentation_return_code": return_code,
        "instrumentation_log": f"instrumentation_logs/{case_id}.txt",
        "device_serial": args.serial,
        "device_model": args.device_model,
    }
    write_json(local_path, failure)
    return failure


def redact_log_file(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        (re.compile(r"(?i)([?&](?:key|api_key|apikey)=)[^&\s]+"), r"\1[REDACTED]"),
        (re.compile(r'(?i)("(?:key|api_key|apikey)"\s*:\s*")[^"]+'), r"\1[REDACTED]"),
        (re.compile(r"(?i)(authorization:\s*(?:bearer|api-key)\s+)[^\s]+"), r"\1[REDACTED]"),
    )
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    path.write_text(text, encoding="utf-8")


def sync_captures(run_dir: Path, scenarios: list[dict[str, Any]]) -> None:
    manifest = run_dir / "android_device_rendered" / "capture_manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Capture manifest not found: {manifest}")
    materialize(run_dir, scenarios)
    print(f"[ok] Synchronized {len(read_jsonl(manifest))} device capture record(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--serial", default="")
    parser.add_argument("--adb-bin", default="adb")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--max-cases", type=int, default=-1)
    parser.add_argument("--ordinal-from", type=int, default=1)
    parser.add_argument("--ordinal-to", type=int, default=2_147_483_647)
    parser.add_argument(
        "--defer-materialize",
        action="store_true",
        help="Write disjoint raw case artifacts only; aggregate in a later single-host pass.",
    )
    parser.add_argument(
        "--inter-case-delay-sec",
        type=int,
        default=0,
        help="Cooling interval after each executed case (maximum 60 seconds).",
    )
    parser.add_argument("--sync-captures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = safe_id(args.run_id, "run_id")
    run_dir = DATASET_ROOT / "data" / "runs" / run_id
    scenario_path = run_dir / "scenarios.jsonl"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario manifest: {scenario_path}")
    scenarios = read_jsonl(scenario_path)
    case_ids = [safe_id(row.get("case_id"), "case_id") for row in scenarios]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Scenario case_id values must be unique")

    if args.sync_captures:
        sync_captures(run_dir, scenarios)
        return 0
    if not args.serial:
        raise ValueError("--serial is required for device execution")
    if args.inter_case_delay_sec < 0 or args.inter_case_delay_sec > 60:
        raise ValueError("--inter-case-delay-sec must be between 0 and 60")

    prefix = adb_prefix(args.adb_bin, args.serial)
    state = run_command([*prefix, "get-state"], timeout_sec=30, check=True).stdout.strip()
    if state != "device":
        raise RuntimeError(f"Device {args.serial} is not ready: {state}")
    package_check = run_command(
        [*prefix, "shell", "pm", "path", APP_PACKAGE], timeout_sec=30, check=True
    )
    if "package:" not in package_check.stdout:
        raise RuntimeError(f"{APP_PACKAGE} is not installed on {args.serial}")

    getprop = lambda name: run_command(  # noqa: E731
        [*prefix, "shell", "getprop", name], timeout_sec=30, check=True
    ).stdout.strip()
    args.device_model = getprop("ro.product.model")
    wm_size_output = run_command(
        [*prefix, "shell", "wm", "size"], timeout_sec=30, check=True
    ).stdout.strip()
    display_sizes = re.findall(r"(?:Physical|Override) size:\s*(\d+x\d+)", wm_size_output)
    display_size = display_sizes[-1] if display_sizes else wm_size_output
    write_json(
        run_dir / "device_snapshots" / f"{safe_id(args.serial, 'serial')}.json",
        {
            "serial": args.serial,
            "manufacturer": getprop("ro.product.manufacturer"),
            "model": args.device_model,
            "android_release": getprop("ro.build.version.release"),
            "sdk": getprop("ro.build.version.sdk"),
            "build_fingerprint": getprop("ro.build.fingerprint"),
            "display_size": display_size,
            "captured_at": utc_iso(),
        },
    )

    selected = scenarios
    selected = [
        row for row in selected
        if args.ordinal_from <= int(row["ordinal"]) <= args.ordinal_to
    ]
    if args.case_id:
        selected = [row for row in selected if row["case_id"] == args.case_id]
        if not selected:
            raise ValueError(f"Unknown case id: {args.case_id}")
    if args.max_cases >= 0:
        selected = selected[: args.max_cases]

    executed = 0
    for scenario in selected:
        case_id = str(scenario["case_id"])
        if args.resume and load_device_result(run_dir, case_id) is not None:
            print(f"[skip] {case_id}: local device result already exists", flush=True)
            continue
        result = run_case(args, run_dir, scenario)
        executed += 1
        if not args.defer_materialize:
            materialize(run_dir, scenarios)
        print(
            f"[{int(scenario['ordinal']):02d}/{len(scenarios):02d}] {case_id} "
            f"success={result.get('success')} elapsed_ms={result.get('elapsed_ms')} "
            f"stage={result.get('outcome_stage')}",
            flush=True,
        )
        if args.inter_case_delay_sec > 0:
            print(
                f"[cooldown] waiting {args.inter_case_delay_sec}s before the next prompt",
                flush=True,
            )
            time.sleep(args.inter_case_delay_sec)

    if not args.defer_materialize:
        materialize(run_dir, scenarios)
    print(f"[ok] Executed {executed} case(s); artifacts: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
