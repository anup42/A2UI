"""Run isolated native controls on frozen Bixby prompts; never repair their output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
LIB_SHA = "e9cbdddb0f1c693c549e1cde40bf90ad8aaa124d15944d0dd18faaf016dd6938"
MODEL_SHA = "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
DEVICE_ROOT = "/data/local/tmp/r64_gpu_controls"
MODEL = "/sdcard/Android/data/com.samsung.genuicraft/files/sdk_models/gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm"
VARIANTS = {
    "gpu_default": ["--backend", "gpu"],
    "gpu_cpu_sampler": ["--backend", "gpu", "--sampler-backend", "cpu"],
    "gpu_fp32": ["--backend", "gpu", "--force-f32"],
    "gpu_fp32_cpu_sampler": ["--backend", "gpu", "--force-f32", "--sampler-backend", "cpu"],
    "cpu": ["--backend", "cpu"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="R3GL203AKSF")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--cases", default="BXP-001,BXP-003")
    parser.add_argument("--label", default=None)
    parser.add_argument("--evidence", type=Path, default=ROOT / "GenUICraft/validation/20260928_r64_gpu_controls")
    args = parser.parse_args()
    cases = args.cases.split(",")
    if any(case not in {"BXP-001", "BXP-003"} for case in cases):
        parser.error("This diagnostic is limited to the two frozen cases.")
    label = args.label or args.variant
    if not label.replace("_", "").replace("-", "").isalnum():
        parser.error("label must contain only letters, numbers, underscores or hyphens")
    destination = args.evidence / label
    destination.mkdir(parents=True, exist_ok=False)
    adb = ["adb", "-s", args.serial]

    def checked(*parts):
        return subprocess.check_output(adb + list(parts), text=True, encoding="utf-8", errors="replace").strip()

    for remote, expected in [(MODEL, MODEL_SHA), (DEVICE_ROOT + "/liblitert-lm.so", LIB_SHA)]:
        actual = checked("shell", "sha256sum", remote).split()[0]
        if actual != expected:
            raise RuntimeError(f"Refusing unpinned artifact: {remote}: {actual}")
    probe_sha = checked("shell", "sha256sum", DEVICE_ROOT + "/probe").split()[0]
    remote_out = DEVICE_ROOT + "/results/" + label
    checked("shell", "mkdir", "-p", remote_out)
    manifest = {"serial": args.serial, "variant": args.variant, "probe_sha256": probe_sha,
                "library_sha256": LIB_SHA, "model_sha256": MODEL_SHA, "cases": cases, "runs": []}
    for case in cases:
        checked("shell", "mkdir", "-p", remote_out + "/" + case + ".cache")
        command = [DEVICE_ROOT + "/probe", "--model", MODEL, "--prompt", DEVICE_ROOT + "/prompts/" + case + ".txt",
                   "--output-prefix", remote_out + "/" + case, "--max-context", "8192", "--max-output", "2048",
                   *VARIANTS[args.variant]]
        shell = "cd " + shlex.quote(DEVICE_ROOT) + " && LD_LIBRARY_PATH=" + shlex.quote(DEVICE_ROOT) + " " + shlex.join(command)
        print(f"Running {label} {case}", flush=True)
        start = time.monotonic()
        with (destination / f"{case}.native.log").open("wb") as log:
            completed = subprocess.run(adb + ["shell", shell], stdout=log, stderr=subprocess.STDOUT, timeout=600)
        manifest["runs"].append({"id": case, "command": shell, "exit_code": completed.returncode,
                                 "host_wall_seconds": time.monotonic() - start})
        (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for suffix in ("metrics.json", "raw.txt", "input_token_ids.txt", "output_retokenized_ids.txt", "selected_token_scores.txt"):
            remote_file = remote_out + "/" + case + "." + suffix
            exists = subprocess.run(adb + ["shell", "test", "-f", remote_file], capture_output=True).returncode == 0
            if exists:
                subprocess.run(adb + ["pull", remote_file, str(destination)], check=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f"{label}/{case} failed; inspect its native log and metrics")
        raw = (destination / f"{case}.raw.txt").read_text(encoding="utf-8")
        prior = ROOT / "GenUICraft/validation/20260928_r64_accuracy_gap"
        source = json.loads((prior / "cpu" / case / "source.json").read_text(encoding="utf-8"))["text"]
        sys.path.insert(0, str(ROOT / "training/src"))
        from ir_training.eval.metrics import score_prediction
        metrics = score_prediction(response_text=source, expected=None, generated_text=raw, metric_version="v5_4")
        reference = (prior / "gpu" / case / "output.express").read_text(encoding="utf-8")
        scored = {"id": case, "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                  "matches_previous_jni_gpu_after_outer_whitespace_trim": raw.strip() == reference.strip(), "metrics": metrics}
        (destination / f"{case}.scored.json").write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"case": case, "raw_valid": metrics.get("schema_valid_strict"),
                          "reward": metrics.get("generation_reward_v5_4"), "matches_jni_gpu": scored["matches_previous_jni_gpu_after_outer_whitespace_trim"]}), flush=True)


if __name__ == "__main__":
    main()
