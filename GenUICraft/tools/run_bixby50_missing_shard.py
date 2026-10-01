"""Collect one declared shard of missing Bixby50 generations on a single device.

Two independent invocations operate different explicit serials. Outputs and
failed candidates are retained; no generation is retried for a better score.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path

PKG = "com.samsung.genuicraft"
MODEL = f"/sdcard/Android/data/{PKG}/files/sdk_models/gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm"
EXPECTED_MODEL = "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
EXPECTED_APK = "0426477f7c874a9bf42c1ac32a2c6a24605d99cda5b29059fdac8e5cb16cdc3f"
EXPECTED_TEST = "176b2bd532efecf72dd27519bad3530b118d574b5f6d9914a719ab1593a2a0cf"


def save(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", default=MODEL)
    ap.add_argument("--expected-model-sha256", default=EXPECTED_MODEL)
    ap.add_argument("--run-prefix", default="v54_full")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", args.run_prefix):
        ap.error("--run-prefix must contain 1..24 letters, digits, underscores or hyphens")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_model_sha256):
        ap.error("--expected-model-sha256 must be a lowercase SHA-256")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "batches").mkdir(exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    assert plan["serial"] == args.serial
    adb = ["adb", "-s", args.serial]

    def cmd(*parts, timeout=60):
        r = subprocess.run(adb + list(parts), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"adb {parts}: {r.stderr[-1500:]}")
        return r.stdout.strip()

    def apk_hash(package):
        path = cmd("shell", "pm", "path", package).removeprefix("package:")
        return cmd("shell", "sha256sum", path).split()[0]

    provenance = {
        "serial": args.serial, "deviceModel": cmd("shell", "getprop", "ro.product.model"),
        "androidVersion": cmd("shell", "getprop", "ro.build.version.release"),
        "mainApkSha256": apk_hash(PKG), "installedTestApkSha256": apk_hash(PKG + ".test"),
        "modelPath": args.model_path,
        "modelSha256": cmd("shell", "sha256sum", args.model_path, timeout=90).split()[0],
        "originalStayOn": cmd("shell", "settings", "get", "global", "stay_on_while_plugged_in"),
        "startedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    assert provenance["mainApkSha256"] == EXPECTED_APK
    assert provenance["installedTestApkSha256"] == EXPECTED_TEST
    assert provenance["modelSha256"] == args.expected_model_sha256
    save(out / "provenance.json", provenance)
    save(out / "plan.json", plan)
    completed = []
    cmd("shell", "svc", "power", "stayon", "true")
    cmd("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    try:
        for entry in plan["batches"]:
            name = entry["name"]
            target = out / "batches" / name
            if target.exists():
                summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                assert summary["runComplete"] and summary["completed"] == len(entry["cases"]), f"Preserve incomplete batch {name}; inspect before resuming"
                completed.append(name)
                continue
            run_id = f"{args.run_prefix}_{plan['deviceAlias']}_{name}_{int(time.time())}"
            remote = f"/sdcard/Android/data/{PKG}/files/sdk_benchmark/{run_id}"
            invocation = ["shell", "am", "instrument", "-w", "-r", "-e", "class", f"{PKG}.GenUiTrainedBixby50Test",
                "-e", "runId", run_id, "-e", "cases", ",".join(entry["cases"]),
                "-e", "backend", "GPU", "-e", "gpuPrecision", "FP32", "-e", "mtp", str(entry["mtp"]).lower(),
                "-e", "modelPath", args.model_path, "-e", "allowSourceTextFallback", "false",
                "-e", "allowGeneratedDslRepair", "true", "-e", "requireSourceIntegrity", "false",
                "-e", "caseTimeoutMs", "360000", f"{PKG}.test/androidx.test.runner.AndroidJUnitRunner"]
            receipt = {"entry": entry, "runId": run_id, "remote": remote, "command": adb + invocation,
                       "serial": args.serial, "deviceModel": provenance["deviceModel"], "startedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat()}
            save(out / f"{name}.invocation.json", receipt)
            started = time.monotonic()
            progress = {**receipt, "completedBatches": completed, "phase": "running"}
            save(out / "progress.json", progress)
            print(f"START {args.serial} {name}: {','.join(entry['cases'])}", flush=True)
            with (out / f"{name}.instrumentation.txt").open("wb") as log:
                process = subprocess.Popen(adb + invocation, stdout=log, stderr=subprocess.STDOUT)
                last_completed = 0
                last_advance = time.monotonic()
                while process.poll() is None:
                    time.sleep(10)
                    try:
                        status = json.loads(cmd("shell", "cat", remote + "/status.json", timeout=15))
                    except Exception as error:
                        status = {"statusReadError": str(error)}
                    count = status.get("completed", last_completed)
                    if count != last_completed:
                        last_advance = time.monotonic()
                        last_completed = count
                    progress.update({"runStatus": status, "hostElapsedSeconds": round(time.monotonic()-started, 1), "updatedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat()})
                    try:
                        thermal = cmd("shell", "dumpsys", "thermalservice", timeout=15)
                        match = re.search(r"Thermal Status:\s*(\d+)", thermal)
                        progress["thermalStatus"] = int(match[1]) if match else None
                    except Exception as error:
                        progress["telemetryError"] = str(error)
                    save(out / "progress.json", progress)
                    print(f"PROGRESS {args.serial} {name}: {count}/{len(entry['cases'])}, {progress['hostElapsedSeconds']}s, thermal={progress.get('thermalStatus')}", flush=True)
                    if time.monotonic()-last_advance > 540:
                        # Preserve evidence before ending a stuck app. The parent
                        # must explicitly record the failure and schedule only
                        # unattempted cases; this collector never silently retries.
                        cmd("pull", remote, str(target), timeout=90)
                        save(target / "host_watchdog.json", progress)
                        cmd("shell", "am", "force-stop", PKG)
                        process.wait(timeout=30)
                        raise TimeoutError(f"No completed-case progress for 540s; evidence saved: {target}")
            cmd("pull", remote, str(target), timeout=90)
            receipt.update({"adbExitCode": process.returncode, "hostElapsedSeconds": round(time.monotonic()-started, 3)})
            save(target / "host_run.json", receipt)
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            assert summary["runComplete"] and summary["completed"] == len(entry["cases"]), f"Incomplete {name}; evidence retained"
            config = json.loads((target / "run_config.json").read_text(encoding="utf-8"))
            assert config["cases"] == entry["cases"] and config["runtime"]["mtpEnabled"] == entry["mtp"]
            completed.append(name)
            print(f"DONE {args.serial} {name}: completed {summary['completed']}, renderValid {summary['renderValid']}", flush=True)
            save(out / "progress.json", {"serial":args.serial,"phase":"between_batches","completedBatches":completed,"lastRunId":run_id})
            time.sleep(5)
        save(out / "COMPLETE.json", {"serial":args.serial,"completedBatches":completed,
             "generations":sum(len(x["cases"]) for x in plan["batches"]), "finishedAtUtc":dt.datetime.now(dt.timezone.utc).isoformat()})
    finally:
        original = provenance["originalStayOn"]
        if original.isdigit():
            cmd("shell", "settings", "put", "global", "stay_on_while_plugged_in", original)


if __name__ == "__main__":
    main()
