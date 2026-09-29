"""Run one immutable native precision experiment and retain raw evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
NATIVE_ROOT = "/data/local/tmp/r64_gpu_controls"
LIB_SHA = "e9cbdddb0f1c693c549e1cde40bf90ad8aaa124d15944d0dd18faaf016dd6938"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--model", required=True, help="Existing device model; never replaced by this runner")
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case", choices=["BXP-001", "BXP-003"], required=True)
    parser.add_argument("--force-f32", action="store_true")
    parser.add_argument("--cpu-sampler", action="store_true", help="Pinned diagnostic sampler bridge with FP32 logits")
    parser.add_argument("--score-target", type=Path)
    parser.add_argument("--max-output", type=int, default=2048)
    parser.add_argument("--trace-opencl", action="store_true")
    parser.add_argument("--mixed-accum", action="store_true", help="Diagnostic public GPU precision-option interposition; verify application in native log")
    parser.add_argument("--patch-dir", help="Existing device directory of exact-source-guarded diagnostic shader patches")
    parser.add_argument("--evidence", type=Path, default=ROOT / "GenUICraft/validation/20260929_fp16_rootcause")
    args = parser.parse_args()
    if not args.label.replace("_", "").replace("-", "").isalnum():
        parser.error("label must be alphanumeric with hyphens/underscores")
    destination = args.evidence / args.label
    destination.mkdir(parents=True, exist_ok=False)
    adb = ["adb", "-s", args.serial]

    def checked(*parts):
        return subprocess.check_output(adb + list(parts), encoding="utf-8", errors="replace").strip()

    for path, expected in [(args.model, args.model_sha), (NATIVE_ROOT + "/liblitert-lm.so", LIB_SHA)]:
        actual = checked("shell", "sha256sum", path).split()[0]
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {path}: {actual}")
    probe_sha = checked("shell", "sha256sum", NATIVE_ROOT + "/probe_score").split()[0]
    hook_sha = (checked("shell", "sha256sum", NATIVE_ROOT + "/libopencl_trace.so").split()[0]
                if args.trace_opencl or args.patch_dir or args.mixed_accum else None)
    patch_manifest_sha = (checked("shell", "sha256sum", args.patch_dir + "/manifest.json").split()[0]
                          if args.patch_dir else None)
    remote = NATIVE_ROOT + "/rootcause/" + args.label
    checked("shell", "mkdir", "-p", remote)
    prefix = remote + "/" + args.case
    command = [NATIVE_ROOT + "/probe_score", "--model", args.model,
               "--prompt", NATIVE_ROOT + "/prompts/" + args.case + ".txt",
               "--output-prefix", prefix, "--backend", "gpu", "--max-context", "8192",
               "--max-output", str(args.max_output)]
    if args.force_f32:
        command += ["--force-f32"]
    if args.cpu_sampler:
        command += ["--sampler-backend", "cpu"]
    target_sha = None
    if args.score_target:
        target_sha = hashlib.sha256(args.score_target.read_bytes()).hexdigest()
        checked("push", str(args.score_target), remote + "/target.txt")
        command += ["--score-target", remote + "/target.txt"]
    prompt_sha = checked("shell", "sha256sum", NATIVE_ROOT + "/prompts/" + args.case + ".txt").split()[0]
    environment = "LD_LIBRARY_PATH=" + shlex.quote(NATIVE_ROOT)
    if args.trace_opencl or args.patch_dir or args.mixed_accum:
        environment += " LD_PRELOAD=" + shlex.quote(NATIVE_ROOT + "/libopencl_trace.so")
        environment += " CL_TRACE_DIR=" + shlex.quote(remote)
    if args.patch_dir:
        environment += " CL_PATCH_DIR=" + shlex.quote(args.patch_dir)
    if args.mixed_accum:
        environment += " CL_MIXED_ACCUM=1"
    shell = "cd " + shlex.quote(NATIVE_ROOT) + " && " + environment + " " + shlex.join(command)
    manifest = {"serial": args.serial, "case": args.case, "model": args.model,
                "model_sha256": args.model_sha, "native_library_sha256": LIB_SHA,
                "probe_sha256": probe_sha, "prompt_sha256": prompt_sha,
                "score_target_sha256": target_sha, "force_f32": args.force_f32,
                "sampler_backend_requested": "cpu" if args.cpu_sampler else "default",
                "command": shell, "mtp_enabled": False, "opencl_capture": args.trace_opencl,
                "mixed_accum_requested": args.mixed_accum,
                "diagnostic_hook_sha256": hook_sha,
                "diagnostic_patch_manifest_sha256": patch_manifest_sha,
                "diagnostic_shader_patch_directory": args.patch_dir}
    def save_manifest():
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    save_manifest()
    (destination / "thermal_before.txt").write_text(checked("shell", "dumpsys", "thermalservice"), encoding="utf-8")
    start = time.monotonic()
    print(f"Running {args.label} {args.case}", flush=True)
    with (destination / "native.log").open("wb") as log:
        completed = subprocess.run(adb + ["shell", shell], stdout=log, stderr=subprocess.STDOUT, timeout=600)
    manifest.update(exit_code=completed.returncode, host_wall_seconds=time.monotonic() - start)
    native_log = (destination / "native.log").read_text(encoding="utf-8", errors="replace")
    manifest["diagnostic_shader_patches_applied"] = len(re.findall(r"^CL_PATCH applied ", native_log, re.MULTILINE))
    manifest["mixed_accum_setter_applied"] = bool(re.search(
        r"^CL_PRECISION requested=1 resolved=3 status=0$", native_log, re.MULTILINE))
    save_manifest()
    (destination / "thermal_after.txt").write_text(checked("shell", "dumpsys", "thermalservice"), encoding="utf-8")
    for suffix in ("metrics.json", "raw.txt", "input_token_ids.txt", "output_retokenized_ids.txt",
                   "scored_target.txt", "target_token_ids.txt", "target_token_scores.txt"):
        path = prefix + "." + suffix
        if subprocess.run(adb + ["shell", "test", "-f", path], capture_output=True).returncode == 0:
            checked("pull", path, str(destination))
    if completed.returncode:
        raise RuntimeError(f"Native probe failed: see {destination / 'native.log'}")
    if args.mixed_accum and not manifest["mixed_accum_setter_applied"]:
        raise RuntimeError("Mixed precision was requested but NOT applied; this run must not be reported as mixed precision")
    if args.patch_dir and not manifest["diagnostic_shader_patches_applied"]:
        raise RuntimeError("No guarded shader patch was applied; this run is not a patched precision result")
    metrics = json.loads((destination / (args.case + ".metrics.json")).read_text(encoding="utf-8"))
    raw_path = destination / (args.case + ".raw.txt")
    if raw_path.exists():
        raw = raw_path.read_text(encoding="utf-8")
        source = json.loads((ROOT / "GenUICraft/validation/20260928_r64_accuracy_gap/cpu" / args.case / "source.json").read_text(encoding="utf-8"))["text"]
        sys.path.insert(0, str(ROOT / "training/src"))
        from ir_training.eval.metrics import score_prediction
        scored = score_prediction(response_text=source, expected=None, generated_text=raw, metric_version="v5_4")
        (destination / "raw_score.json").write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"label": args.label, "raw_valid": scored.get("schema_valid_strict"),
                          "v54": scored.get("generation_reward_v5_4"),
                          "decode_tps": metrics.get("decode_tokens_per_s")}), flush=True)
    else:
        print(json.dumps({"label": args.label, "score": metrics.get("target_score"),
                          "token_scores": metrics.get("target_token_scores_status")}), flush=True)


if __name__ == "__main__":
    main()
