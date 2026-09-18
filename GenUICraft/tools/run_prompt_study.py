"""Run sequential, real-device prompt comparisons through the installed AAR consumer.

Input is a JSON array of {name, prompt, cases, repairs?}. Prompts are host paths.
Every run is new; failures are preserved and do not prevent later candidates.
No model/runtime/thermal setting is changed by this runner.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from summarize_benchmark import summarize
from prompt_study_evidence import validate_attempt_evidence


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial", default="R3GL203AKSF")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    args.output.mkdir(parents=True, exist_ok=True)
    adb = ["adb", "-s", args.serial]
    device_root = "/sdcard/Android/data/com.samsung.genuicraft/files"

    def capture(*command):
        result = subprocess.run(adb + list(command), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
        return {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    for item in plan:
        name = item["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Unsafe run name")
        run = args.output / name
        if run.exists():
            raise FileExistsError(run)
        prompt = Path(item["prompt"]).resolve()
        cases = item["cases"]
        corpus = Path(item["corpus"]).resolve() if item.get("corpus") else Path(__file__).resolve().parents[2] / "android/app/src/main/assets/genuicraft_bixby50.jsonl"
        corpus_ids = [json.loads(line)["id"] for line in corpus.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(cases) != len(set(cases)) or not all(re.fullmatch(r"[A-Za-z0-9_-]{1,60}", c) and c in corpus_ids for c in cases):
            raise ValueError("Invalid or duplicated case IDs")
        exists = capture("shell", "test", "-e", f"{device_root}/sdk_benchmark/{name}")
        if exists["exit"] == 0:
            raise FileExistsError(f"Device run already exists: {name}")
        run.mkdir()
        (run / "experiment_prompt.txt").write_bytes(prompt.read_bytes())
        remote_prompt = f"{device_root}/sdk_prompt_study/{name}.txt"
        subprocess.run(adb + ["shell", "mkdir", "-p", f"{device_root}/sdk_prompt_study"], check=True, capture_output=True)
        subprocess.run(adb + ["push", str(prompt), remote_prompt], check=True, capture_output=True)
        started = datetime.now(timezone.utc).isoformat()
        metadata = {
            "name": name, "startedUtc": started, "serial": args.serial,
            "promptHostPath": str(prompt), "promptSha256": sha(prompt),
            "promptCharacters": len(prompt.read_text(encoding="utf-8-sig")),
            "cases": cases, "repairs": item.get("repairs", 0),
            "accelerator": "GPU", "mtp": True, "thinkingEnabled": True,
            "thinkingBudget": 1024, "temperature": 0.0, "sourceBindings": True,
            "packagedPrompt": item.get("packagedPrompt", False),
            "inputScaffold": item.get("inputScaffold", False),
            "sdkLayoutScaffold": item.get("sdkLayoutScaffold", False),
            "recordInputs": item.get("recordInputs", False),
            "corpusHostPath": str(corpus), "corpusSha256": sha(corpus),
            "stopAfterFailures": item.get("stopAfterFailures"),
            "caseTimeoutMs": item.get("caseTimeoutMs", 600000),
            "thermalBefore": capture("shell", "dumpsys", "thermalservice"),
            "batteryBefore": capture("shell", "dumpsys", "battery"),
            "appPath": capture("shell", "pm", "path", "com.samsung.genuicraft"),
            "testAppPath": capture("shell", "pm", "path", "com.samsung.genuicraft.test"),
            "devicePromptHash": capture("shell", "sha256sum", remote_prompt),
        }
        for field, hash_field in (("appPath", "appSha256"), ("testAppPath", "testAppSha256")):
            package_paths = [line.removeprefix("package:") for line in metadata[field]["stdout"].splitlines() if line.startswith("package:")]
            metadata[hash_field] = [capture("shell", "sha256sum", path) for path in package_paths]
        if metadata["promptSha256"] not in metadata["devicePromptHash"]["stdout"]:
            raise RuntimeError(f"Device prompt checksum mismatch: {name}")
        command = adb + ["shell", "am", "instrument", "-w", "-r", "-e", "class",
            "com.samsung.genuicraft.GenUiSdkBixby50Test#convertAndRenderBixbyCorpus",
            "-e", "provider", "gemma", "-e", "accelerator", "GPU", "-e", "mtp", "true",
            "-e", "thinkingBudget", "1024", "-e", "temperature", "0.0",
            "-e", "modelPath", f"{device_root}/sdk_models/gemma-4-E2B-it.litertlm",
            "-e", "runId", name,
            "-e", "sourceBindings", "true", "-e", "repairs", str(item.get("repairs", 0)),
            "-e", "cases", ",".join(cases), "-e", "caseTimeoutMs", str(item.get("caseTimeoutMs", 600000))]
        if not item.get("packagedPrompt", False):
            command += ["-e", "promptPath", remote_prompt]
        if item.get("inputScaffold", False):
            command += ["-e", "inputScaffold", "true"]
        if item.get("recordInputs", False):
            command += ["-e", "recordInputs", "true"]
        if item.get("corpus"):
            remote_corpus = f"{device_root}/sdk_prompt_study/{name}.jsonl"
            subprocess.run(adb + ["push", str(corpus), remote_corpus], check=True, capture_output=True)
            command += ["-e", "corpusPath", remote_corpus]
        command += ["com.samsung.genuicraft.test/androidx.test.runner.AndroidJUnitRunner"]
        metadata["command"] = command
        (run / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"START {name}: {len(cases)} cases, prompt={metadata['promptSha256'][:12]}", flush=True)
        elapsed = time.monotonic()
        with (run / "instrumentation.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + item.get("caseTimeoutMs", 600000) / 1000 * len(cases) + 180
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise subprocess.TimeoutExpired(command, deadline)
                    time.sleep(10)
                    if not item.get("stopAfterFailures") or process.poll() is not None:
                        continue
                    snapshot = capture("shell", "cat", f"{device_root}/sdk_benchmark/{name}/results.json")
                    try:
                        current_results = json.loads(snapshot["stdout"])
                    except json.JSONDecodeError:
                        continue
                    failed = [r["id"] for r in current_results if r["status"] != "success"]
                    if len(failed) >= item["stopAfterFailures"] and len(current_results) < len(cases):
                        metadata["earlyRejected"] = True
                        metadata["rejectionReason"] = f"Stopped after {len(failed)} completed validation/render failures: {failed}. Any in-flight case is excluded from completed-case statistics."
                        metadata["completedIdsAtStop"] = [r["id"] for r in current_results]
                        capture("shell", "am", "force-stop", "com.samsung.genuicraft")
                        process.wait(timeout=45)
                        print(f"REJECT {name}: {metadata['rejectionReason']}", flush=True)
                        break
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError(f"Instrumentation timed out: {name}; inspect device before continuing")
        metadata["processExit"] = process.returncode
        metadata["wallSeconds"] = round(time.monotonic() - elapsed, 3)
        metadata["endedUtc"] = datetime.now(timezone.utc).isoformat()
        metadata["thermalAfter"] = capture("shell", "dumpsys", "thermalservice")
        runtime_log = capture("logcat", "-d", "-v", "threadtime", "-s", "GenUICraftRuntime:I", "OpenCL:I", "*:S")
        (run / "runtime_log.txt").write_text(runtime_log["stdout"], encoding="utf-8")
        pulled = capture("pull", f"{device_root}/sdk_benchmark/{name}/.", str(run))
        metadata["pull"] = pulled
        (run / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if pulled["exit"] != 0:
            raise RuntimeError(f"Failed to pull {name}: {pulled}")
        result = summarize(run)
        config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        if "promptSha256" in config and config["promptSha256"] != metadata["promptSha256"]:
            raise RuntimeError(f"Effective prompt checksum mismatch: {name}")
        if "corpusSha256" in config and config["corpusSha256"] != metadata["corpusSha256"]:
            raise RuntimeError(f"Effective corpus checksum mismatch: {name}")
        evidence = validate_attempt_evidence(run, metadata, config, json.loads((run / "results.json").read_text(encoding="utf-8")))
        (run / "attempt_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        if not result["run_complete"] and not metadata.get("earlyRejected"):
            raise RuntimeError(f"Incomplete run or unexpected runtime: {name}")
        print(f"DONE {name}: {result['first_attempt_success']}/{result['total']} completed first; planned={len(cases)}; complete={result['run_complete']}; earlyRejected={metadata.get('earlyRejected', False)}; median={result['median_success_ms']} ms; failures={result['failures']}", flush=True)


if __name__ == "__main__":
    main()
