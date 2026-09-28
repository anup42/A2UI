#!/usr/bin/env python3
"""Run every frozen Bixby50 case in paired five-case GPU FP32 MTP batches.

Only the instrumentation harness runs inference. The host samples read-only
thermal/GPU telemetry, pulls evidence, and preserves failed/partial batches.
Re-running skips complete batches; it never overwrites an incomplete attempt.
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

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "com.samsung.genuicraft"
MODEL = f"/sdcard/Android/data/{PACKAGE}/files/sdk_models/gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm"


def save(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "batches").mkdir(exist_ok=True)
    adb = [args.adb, "-s", args.serial]

    def command(*parts: str, timeout: int = 20) -> str:
        result = subprocess.run(adb + list(parts), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"adb {parts}: {result.stderr[:300]}")
        return result.stdout

    def snapshot(batch: str | None, stage: str) -> dict:
        record = {"epochMs": int(time.time() * 1000), "batch": batch, "stage": stage}
        try:
            thermal = command("shell", "dumpsys", "thermalservice")
            status = re.search(r"Thermal Status:\s*(\d+)", thermal)
            record["thermalStatus"] = int(status[1]) if status else None
            current = thermal.split("Current temperatures from HAL:")[-1].split("Current cooling devices")[0]
            record["temperaturesC"] = {m[1]: float(m[0]) for m in re.findall(r"mValue=([\d.-]+).*?mName=([^,}]+)", current)}
            clocks = command("shell", "cat", "/sys/class/kgsl/kgsl-3d0/gpuclk", "/sys/class/kgsl/kgsl-3d0/max_gpuclk", "/sys/class/kgsl/kgsl-3d0/gpubusy").splitlines()
            record["gpuClockHz"] = int(clocks[0])
            record["gpuMaxClockHz"] = int(clocks[1])
            record["gpuBusyCounters"] = clocks[2].split()
        except Exception as exc:
            record["telemetryError"] = str(exc)
        with (destination / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return record

    ids = [f"BXP-{i:03}" for i in range(1, 51)]
    plan = []
    for group in range(10):
        modes = [False, True] if group % 2 == 0 else [True, False]
        for mtp in modes:
            name = f"group_{group + 1:02}_{'mtp_on' if mtp else 'mtp_off'}"
            plan.append({"batch": name, "mtp": mtp, "cases": ids[group * 5:(group + 1) * 5]})
    protocol_file = destination / "protocol.json"
    if protocol_file.exists():
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        assert protocol["plan"] == plan and protocol["serial"] == args.serial
    else:
        protocol = {
            "startedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "serial": args.serial, "modelPath": MODEL, "precision": "FP32",
            "contextTokens": 8192, "maxOutputTokens": 2048,
            "temperature": 0.0, "thinking": False, "sourceFallback": False,
            "generatedDslRecovery": True, "batchSize": 5, "restBetweenBatchesSeconds": 15,
            "telemetryIntervalSeconds": 15, "order": "ABBA across adjacent five-case groups",
            "plan": plan,
        }
        save(protocol_file, protocol)
    apk = command("shell", "pm", "path", PACKAGE).strip().removeprefix("package:")
    provenance = {
        "mainApkDeviceSha256": command("shell", "sha256sum", apk).split()[0],
        "testApkHostSha256": hashlib.file_digest((ROOT / "android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk").open("rb"), "sha256").hexdigest(),
        "modelDeviceSha256": command("shell", "sha256sum", MODEL, timeout=90).split()[0],
        "deviceModel": command("shell", "getprop", "ro.product.model").strip(),
    }
    expected = destination / "provenance.json"
    if expected.exists():
        assert json.loads(expected.read_text(encoding="utf-8")) == provenance, "Installed APK or model changed since the run started."
    else:
        save(expected, provenance)
    completed = []
    for entry in plan:
        name = entry["batch"]
        batch = destination / "batches" / name
        if batch.exists():
            summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
            config = json.loads((batch / "run_config.json").read_text(encoding="utf-8"))
            assert summary["runComplete"] and summary["completed"] == 5 and config["cases"] == entry["cases"], f"Incomplete existing batch: {name}"
            completed.append(name)
            continue
        run_id = f"fp32_full_{name}_{int(time.time())}"
        remote = f"/sdcard/Android/data/{PACKAGE}/files/sdk_benchmark/{run_id}"
        invocation = ["shell", "am", "instrument", "-w", "-r", "-e", "class", f"{PACKAGE}.GenUiTrainedBixby50Test",
                      "-e", "runId", run_id, "-e", "cases", ",".join(entry["cases"]),
                      "-e", "backend", "GPU", "-e", "gpuPrecision", "FP32",
                      "-e", "mtp", str(entry["mtp"]).lower(), "-e", "modelPath", MODEL,
                      "-e", "allowSourceTextFallback", "false", "-e", "allowGeneratedDslRepair", "true",
                      "-e", "requireSourceIntegrity", "false", "-e", "caseTimeoutMs", "360000",
                      f"{PACKAGE}.test/androidx.test.runner.AndroidJUnitRunner"]
        save(destination / "progress.json", {"activeBatch": name, "runId": run_id, "completedBatches": completed, "completedGenerations": len(completed) * 5, "plan": entry})
        print(f"START {name}: {','.join(entry['cases'])}", flush=True)
        snapshot(name, "before")
        started = time.monotonic()
        with (destination / f"{name}.instrumentation.txt").open("wb") as output:
            process = subprocess.Popen(adb + invocation, stdout=output, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(15)
                state = snapshot(name, "running")
                try:
                    progress = json.loads(command("shell", "cat", remote + "/status.json"))
                except Exception:
                    progress = {}
                state["runStatus"] = progress
                state["completedBatches"] = completed
                save(destination / "progress.json", state)
                print(f"PROGRESS {name}: {progress.get('completed', '?')}/5; thermal={state.get('thermalStatus')}; GPU={state.get('gpuClockHz')}; elapsed={time.monotonic()-started:.0f}s", flush=True)
                if time.monotonic() - started > 5 * 360 + 180:
                    raise TimeoutError(f"Batch exceeded host watchdog: {name}; inspect device before resuming.")
        snapshot(name, "after")
        command("pull", remote, str(batch), timeout=90)
        summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
        assert summary["runComplete"] and summary["completed"] == 5, f"Incomplete {name}: {summary}"
        # A benchmark can fail its allValid assertion while still measuring every case.
        # Keep those failures and continue; never silently replace them with a passing retry.
        save(batch / "host_run.json", {"entry": entry, "adbExitCode": process.returncode, "command": adb + invocation, "hostElapsedSeconds": time.monotonic() - started})
        completed.append(name)
        print(f"DONE {name}: rendered={summary['renderValid']}/5; total={len(completed)*5}/100", flush=True)
        save(destination / "progress.json", {"completedBatches": completed, "completedGenerations": len(completed) * 5, "activeBatch": None})
        if len(completed) < len(plan):
            time.sleep(15)
    save(destination / "COMPLETE.json", {"completedGenerations": 100, "completedBatches": completed, "finishedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print("COMPLETE: 50 cases per mode; 100 generations.", flush=True)


if __name__ == "__main__":
    main()
