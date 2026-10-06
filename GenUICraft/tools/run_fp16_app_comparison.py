#!/usr/bin/env python3
"""Run alternating warm Gemma 4 FP32/corrected-FP16 app instrumentation arms.

No model is modified by this script. Supply two distinct, already installed
LiteRT packages and a fresh host output directory. MTP is off by default; only
request the on arm after the corrected package's drafter is verified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from report_fp16_app_comparison import write_report


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "com.samsung.genuicraft"
TEST_RUNNER = f"{PACKAGE}.test/androidx.test.runner.AndroidJUnitRunner"
CORPUS = ROOT / "android/app/src/main/assets/genuicraft_bixby50.jsonl"
PROMPT_CONTRACT_SHA256 = "005ef700eee29a4429c895ac6c86580003c43d0ad35e67cbb92a9d4cf0e8c1d2"
DEFAULT_CASES = ("BXP-001", "BXP-003", "BXP-004")
NAME = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


class Adb:
    def __init__(self, executable: str, serial: str):
        self.prefix = [executable, "-s", serial]

    def command(self, *parts: str, timeout: int = 30, check: bool = True) -> str:
        result = subprocess.run(self.prefix + list(parts), capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
        if check and result.returncode:
            raise RuntimeError(f"adb {parts}: exit {result.returncode}: {result.stderr[-600:]}")
        return result.stdout

    def file_hash(self, remote: str) -> str:
        line = self.command("shell", "sha256sum", remote, timeout=240).strip()
        digest = line.split()[0] if line else ""
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise RuntimeError(f"Could not hash remote file {remote}: {line[:300]}")
        return digest.lower()


def telemetry(adb: Adb, output: Path, label: str, phase: str) -> dict[str, Any]:
    row: dict[str, Any] = {"epochMs": int(time.time() * 1000), "label": label, "phase": phase}
    try:
        thermal = adb.command("shell", "dumpsys", "thermalservice")
        match = re.search(r"Thermal Status:\s*(\d+)", thermal)
        row["thermalStatus"] = int(match.group(1)) if match else None
        current = thermal.split("Current temperatures from HAL:")[-1].split("Current cooling devices")[0]
        row["halTemperaturesC"] = {
            match.group(2).strip(): float(match.group(1))
            for match in re.finditer(r"mValue=([\d.-]+).*?mName=([^,}]+)", current)
        }
    except Exception as exc:
        row["thermalError"] = str(exc)
    try:
        battery = adb.command("shell", "dumpsys", "battery")
        match = re.search(r"temperature:\s*(\d+)", battery)
        row["batteryTemperatureC"] = int(match.group(1)) / 10 if match else None
        match = re.search(r"level:\s*(\d+)", battery)
        row["batteryLevel"] = int(match.group(1)) if match else None
    except Exception as exc:
        row["batteryError"] = str(exc)
    try:
        gpu = adb.command("shell", "cat", "/sys/class/kgsl/kgsl-3d0/gpuclk",
                          "/sys/class/kgsl/kgsl-3d0/max_gpuclk",
                          "/sys/class/kgsl/kgsl-3d0/gpubusy")
        lines = gpu.splitlines()
        row["gpuClockHz"] = int(lines[0]) if len(lines) > 0 else None
        row["gpuMaxClockHz"] = int(lines[1]) if len(lines) > 1 else None
        row["gpuBusyCounters"] = lines[2].split() if len(lines) > 2 else None
    except Exception as exc:
        row["gpuError"] = str(exc)
    with (output / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def build_plan(cases: list[str], repetitions: int, mtp_modes: list[bool],
               original_model: str, corrected_model: str) -> list[dict[str, Any]]:
    del cases  # each arm replays the same ordered case list from protocol.json
    plan = []
    for mtp in mtp_modes:
        for repetition in range(1, repetitions + 1):
            order = ("FP32", "FP16_CORRECTED") if repetition % 2 else ("FP16_CORRECTED", "FP32")
            for precision in order:
                plan.append({
                    "label": f"r{repetition:02}_{precision.lower()}_mtp_{'on' if mtp else 'off'}",
                    "repetition": repetition,
                    "precision": precision,
                    "mtp": mtp,
                    "modelPath": original_model if precision == "FP32" else corrected_model,
                })
    return plan


def preflight(args: argparse.Namespace, adb: Adb, output: Path,
              original_model: str, corrected_model: str) -> dict[str, Any]:
    test_apk = ROOT / "android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
    if not test_apk.is_file():
        raise FileNotFoundError(f"Build instrumentation APK first: {test_apk}")
    main_apk = adb.command("shell", "pm", "path", PACKAGE).strip().splitlines()
    test_installed = adb.command("shell", "pm", "path", PACKAGE + ".test").strip().splitlines()
    if not main_apk or not test_installed:
        raise RuntimeError("Main and instrumentation APKs must both be installed")
    main_path = main_apk[0].removeprefix("package:")
    test_path = test_installed[0].removeprefix("package:")
    hashes = {"original": adb.file_hash(original_model),
              "corrected": adb.file_hash(corrected_model)}
    if hashes["original"] == hashes["corrected"]:
        raise RuntimeError("Original and corrected model packages have identical SHA-256")
    if args.original_sha256 and hashes["original"] != args.original_sha256.lower():
        raise RuntimeError("Original model SHA-256 differs from --original-sha256")
    if args.corrected_sha256 and hashes["corrected"] != args.corrected_sha256.lower():
        raise RuntimeError("Corrected model SHA-256 differs from --corrected-sha256")
    provenance = {
        "serial": args.serial,
        "deviceModel": adb.command("shell", "getprop", "ro.product.model").strip(),
        "mainApkPath": main_path, "mainApkSha256": adb.file_hash(main_path),
        "testApkPath": test_path, "testApkDeviceSha256": adb.file_hash(test_path),
        "testApkHostSha256": sha256(test_apk),
        "models": {"FP32": {"path": original_model, "sha256": hashes["original"]},
                   "FP16_CORRECTED": {"path": corrected_model, "sha256": hashes["corrected"]}},
    }
    if provenance["testApkDeviceSha256"] != provenance["testApkHostSha256"]:
        raise RuntimeError("Installed instrumentation APK does not match host build")
    if args.corrected_manifest:
        manifest = args.corrected_manifest.resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        (output / "corrected_model_manifest.json").write_bytes(manifest.read_bytes())
        provenance["correctedManifestHostSha256"] = sha256(manifest)
    write_json(output / "provenance.json", provenance)
    return provenance


def run_arm(adb: Adb, output: Path, protocol: dict[str, Any], entry: dict[str, Any],
            args: argparse.Namespace) -> None:
    label = entry["label"]
    remote_leaf = f"fp16cmp_{args.tag}_{label}_{int(time.time())}"
    remote_dir = f"/sdcard/Android/data/{PACKAGE}/files/sdk_fp16_comparison/{remote_leaf}"
    local_dir = output / "runs" / label
    if local_dir.exists():
        raise RuntimeError(f"Refusing to overwrite arm: {local_dir}")
    if adb.file_hash(entry["modelPath"]) != protocol["modelSha256"][entry["precision"]]:
        raise RuntimeError(f"{label}: model package changed after preflight")
    invocation = [
        "shell", "am", "instrument", "-w", "-r", "-e", "class",
        f"{PACKAGE}.Gemma4Fp16ComparisonTest", "-e", "outputDir", remote_leaf,
        "-e", "label", label, "-e", "precision", entry["precision"],
        "-e", "modelPath", entry["modelPath"], "-e", "mtp", str(entry["mtp"]).lower(),
        "-e", "cases", ",".join(protocol["cases"]),
        "-e", "caseTimeoutMs", str(args.case_timeout_ms), TEST_RUNNER,
    ]
    if getattr(args, "warmup_case", None):
        invocation[-1:-1] = ["-e", "warmupCase", args.warmup_case]
    if getattr(args, "cooldown_timeout_ms", 0):
        invocation[-1:-1] = ["-e", "cooldownTimeoutMs", str(args.cooldown_timeout_ms)]
    (output / "commands").mkdir(exist_ok=True)
    write_json(output / "commands" / f"{label}.json", {"argv": adb.prefix + invocation,
                                                       "remoteDir": remote_dir})
    print(f"START {label}: {','.join(protocol['cases'])}", flush=True)
    telemetry(adb, output, label, "before")
    started = time.monotonic()
    deadline = ((len(protocol["cases"]) + 1) * args.case_timeout_ms / 1000
                + getattr(args, "cooldown_timeout_ms", 0) / 1000 + 180)
    process: subprocess.Popen[bytes] | None = None
    exit_code: int | None = None
    failure: str | None = None
    try:
        with (output / "commands" / f"{label}.instrumentation.txt").open("wb") as log:
            process = subprocess.Popen(adb.prefix + invocation, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(args.telemetry_interval_seconds)
                state = telemetry(adb, output, label, "running")
                raw_status = adb.command("shell", "cat", remote_dir + "/status.json", check=False)
                try:
                    status = json.loads(raw_status)
                except json.JSONDecodeError:
                    status = None
                write_json(output / "progress.json", {"activeArm": label,
                                                       "elapsedSeconds": time.monotonic() - started,
                                                       "deviceStatus": status,
                                                       "thermalStatus": state.get("thermalStatus")})
                if time.monotonic() - started > deadline:
                    raise TimeoutError(f"Host watchdog exceeded {deadline:.0f}s for {label}")
            exit_code = process.returncode
    except Exception as exc:
        failure = str(exc)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            exit_code = process.returncode
    finally:
        telemetry(adb, output, label, "after")
        pull_error = None
        try:
            adb.command("pull", remote_dir, str(local_dir), timeout=180)
        except Exception as exc:
            pull_error = str(exc)
        write_json(output / "commands" / f"{label}.host_result.json", {
            "label": label, "remoteDir": remote_dir, "localDir": str(local_dir),
            "instrumentationExitCode": exit_code, "hostElapsedSeconds": time.monotonic() - started,
            "failure": failure, "pullError": pull_error,
        })
        write_report(output)
    if failure or pull_error:
        raise RuntimeError(f"{label}: {failure or pull_error}; partial evidence preserved")
    config = json.loads((local_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((local_dir / "summary.json").read_text(encoding="utf-8"))
    if config["precision"] != entry["precision"] or config["mtpRequested"] != entry["mtp"]:
        raise RuntimeError(f"{label}: pulled arm identity mismatch")
    if getattr(args, "warmup_case", None) and config.get("warmupCase") != args.warmup_case:
        raise RuntimeError(f"{label}: instrumentation did not apply the requested warmup case")
    if getattr(args, "cooldown_timeout_ms", 0):
        gate_file = local_dir / "thermal_ready.json"
        if (config.get("cooldownTimeoutMs") != args.cooldown_timeout_ms or not gate_file.is_file()
                or json.loads(gate_file.read_text(encoding="utf-8")).get("ready") is not True):
            raise RuntimeError(f"{label}: requested thermal gate did not pass")
    if not summary.get("runComplete") or summary.get("completed") != len(protocol["cases"]):
        raise RuntimeError(f"{label}: incomplete arm; evidence preserved")
    warmup = json.loads((local_dir / "warmup_result.json").read_text(encoding="utf-8"))
    if warmup.get("rawComplete") is not True or not warmup.get("runtime"):
        raise RuntimeError(f"{label}: native warmup did not complete; evidence preserved")
    session_metrics = json.loads((local_dir / "session_metrics.json").read_text(encoding="utf-8"))
    if not isinstance(session_metrics, dict) or session_metrics.get("speculativeDecodingEnabled") != entry["mtp"]:
        raise RuntimeError(f"{label}: actual MTP state differs from requested arm; evidence preserved")
    if exit_code:
        raise RuntimeError(f"{label}: instrumentation exit {exit_code}; evidence preserved")
    print(f"DONE {label}: raw strict {summary['rawStrictValid']}/{summary['total']}; "
          f"repaired strict {summary['repairedStrictValid']}/{summary['total']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", required=True, help="Path to adb executable")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--original-model", required=True, help="Absolute device path to original rank-64 model")
    parser.add_argument("--corrected-model", required=True, help="Absolute device path to prepared corrected model")
    parser.add_argument("--corrected-manifest", type=Path)
    parser.add_argument("--original-sha256")
    parser.add_argument("--corrected-sha256")
    parser.add_argument("--out", required=True, type=Path, help="Fresh host evidence directory")
    parser.add_argument("--tag", default="comparison", help="Short unique device-directory tag")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--warmup-case", help="Optional frozen corpus ID for a shorter excluded warmup")
    parser.add_argument("--cooldown-timeout-ms", type=int, default=0,
                        help="After warmup, retain the engine and wait for thermal status 0; 0 disables the gate")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--mtp-mode", choices=("off", "on", "both"), default="off")
    parser.add_argument("--mtp-drafter-verified", action="store_true",
                        help="Required with --mtp-mode on/both after verifying the corrected package's drafter")
    parser.add_argument("--case-timeout-ms", type=int, default=360_000)
    parser.add_argument("--telemetry-interval-seconds", type=int, default=10)
    parser.add_argument("--cooldown-seconds", type=int, default=30)
    args = parser.parse_args()
    if not NAME.fullmatch(args.tag):
        parser.error("--tag must contain only letters, digits, underscores, or hyphens")
    if args.repetitions < 1 or args.repetitions > 10:
        parser.error("--repetitions must be 1..10; use at least 2 for counterbalanced order")
    if not 1_000 <= args.case_timeout_ms <= 3_600_000:
        parser.error("--case-timeout-ms out of range")
    if args.telemetry_interval_seconds < 5 or args.cooldown_seconds < 0:
        parser.error("telemetry interval must be >=5 and cooldown >=0")
    if args.mtp_mode != "off" and not args.mtp_drafter_verified:
        parser.error("MTP on requires --mtp-drafter-verified")
    cases = [part.strip() for part in args.cases.split(",")]
    if len(cases) != len(set(cases)) or not cases or not all(re.fullmatch(r"BXP-\d{3}", c) for c in cases):
        parser.error("--cases must be a nonempty comma-separated list of unique BXP IDs")
    frozen = {json.loads(line)["id"] for line in CORPUS.read_text(encoding="utf-8").splitlines() if line}
    if not set(cases) <= frozen:
        parser.error("--cases includes an ID outside frozen Bixby50")
    if args.warmup_case and args.warmup_case not in frozen:
        parser.error("--warmup-case must be a frozen Bixby50 ID")
    if not 0 <= args.cooldown_timeout_ms <= 600_000:
        parser.error("--cooldown-timeout-ms must be 0..600000")
    original = args.original_model.strip()
    corrected = args.corrected_model.strip()
    if not original.startswith("/") or not corrected.startswith("/") or original == corrected:
        parser.error("Supply distinct absolute device model paths")
    output = args.out.resolve()
    if output.exists():
        parser.error(f"--out must be new; existing path: {output}")
    output.mkdir(parents=True)
    (output / "runs").mkdir()
    adb = Adb(args.adb, args.serial)
    mtp_modes = [False, True] if args.mtp_mode == "both" else [args.mtp_mode == "on"]
    plan = build_plan(cases, args.repetitions, mtp_modes, original, corrected)
    protocol = {
        "schemaVersion": 1, "startedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "serial": args.serial, "cases": cases, "corpusSha256": sha256(CORPUS),
        "promptContractSha256": PROMPT_CONTRACT_SHA256,
        "repetitions": args.repetitions, "mtpModes": mtp_modes,
        "caseTimeoutMs": args.case_timeout_ms, "cooldownSeconds": args.cooldown_seconds,
        "order": ("AB only; single-pair pilot, not counterbalanced" if args.repetitions == 1
                  else "Alternating AB/BA within each MTP mode (ABBA for two repetitions)"),
        "warmup": (f"Each arm uses {args.warmup_case} as an excluded warmup" if args.warmup_case
                   else "Each arm reruns its first case once before measured cases; warmup excluded"),
        "warmupCase": args.warmup_case,
        "cooldownTimeoutMs": args.cooldown_timeout_ms,
        "plan": plan,
    }
    write_json(output / "protocol.json", protocol)
    provenance = preflight(args, adb, output, original, corrected)
    protocol["modelSha256"] = {
        "FP32": provenance["models"]["FP32"]["sha256"],
        "FP16_CORRECTED": provenance["models"]["FP16_CORRECTED"]["sha256"],
    }
    write_json(output / "protocol.json", protocol)
    write_report(output)
    for index, entry in enumerate(plan):
        run_arm(adb, output, protocol, entry, args)
        if index + 1 < len(plan) and args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
    summary = write_report(output)
    print(f"COMPLETE: {len(summary['observations'])} measured cases; "
          f"matched={summary['completeAndMatched']}; report={output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
