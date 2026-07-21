#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def adb(adb_bin: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command([adb_bin, *args], check=check)


def adb_file_exists(adb_bin: str, remote_path: str) -> bool:
    result = subprocess.run([adb_bin, "shell", "test", "-f", remote_path])
    return result.returncode == 0


def ensure_device_connected(adb_bin: str) -> None:
    result = adb(adb_bin, "get-state", check=False)
    state = (result.stdout or "").strip().lower()
    if result.returncode != 0 or state != "device":
        raise RuntimeError("No adb device in 'device' state. Connect a device and enable USB debugging.")


def count_png_files(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.png"))


def write_native_render_checks(run_dir: Path, output_dir: Path) -> int:
    """Publish device capture results in the metric's renderer-adapter format."""
    capture_manifest = output_dir / "capture_manifest.jsonl"
    if not capture_manifest.exists():
        raise RuntimeError(f"Android capture did not produce a manifest: {capture_manifest}")

    try:
        output_relative = output_dir.relative_to(run_dir)
    except ValueError:
        output_relative = Path(output_dir.name)

    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        capture_manifest.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            capture = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid Android capture manifest JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(capture, dict):
            raise RuntimeError(
                f"Invalid Android capture manifest row at line {line_number}: expected object"
            )

        screenshot_name = str(capture.get("screenshot") or "")
        screenshot_path = (
            (output_relative / screenshot_name).as_posix() if screenshot_name else ""
        )
        ok = capture.get("ok") is True
        rows.append(
            {
                "ui_id": str(capture.get("ui_id") or ""),
                "query_id": str(capture.get("query_id") or ""),
                "response_id": str(capture.get("response_id") or ""),
                "renderer_check_result": {
                    "adapter": "android_native_flat_renderer",
                    "source": "DatasetRenderCaptureActivity.capture_manifest",
                    "attempted": True,
                    "ok": ok,
                    "status": "rendered" if ok else "failed",
                    "screenshot": screenshot_path,
                    "full_height_px": capture.get("full_height_px"),
                    "tile_count": capture.get("tile_count"),
                },
            }
        )

    checks_path = run_dir / "native_render_checks.jsonl"
    temp_path = checks_path.with_name(f".{checks_path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temp_path.replace(checks_path)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a dataset run on Android device and capture per-IR screenshots."
    )
    parser.add_argument("--run_id", required=True, help="Run id under dataset/data/runs/<run_id>")
    parser.add_argument(
        "--output_dir_name",
        default="android_device_rendered",
        help="Output folder name inside the run directory.",
    )
    parser.add_argument("--adb_bin", default="adb", help="adb executable path (default: adb)")
    parser.add_argument("--package", default="com.samsung.genuicraft", help="Android app package name")
    parser.add_argument(
        "--device_base_dir",
        default="",
        help="Override remote base dir on device. Default: /sdcard/Android/data/<package>/files/dataset_capture/<run_id>",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Start record index for capture")
    parser.add_argument("--max_count", type=int, default=-1, help="Max records to capture (-1 = all)")
    parser.add_argument("--settle_ms", type=int, default=1200, help="Wait time per record before capture")
    parser.add_argument("--timeout_sec", type=int, default=900, help="Total timeout waiting for capture completion")
    parser.add_argument(
        "--keep_remote",
        action="store_true",
        help="Do not delete remote capture folder after pulling screenshots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "data" / "runs" / args.run_id
    genui_path = run_dir / "genui.jsonl"
    assets_dir = run_dir / "assets"
    output_dir = run_dir / args.output_dir_name

    if not run_dir.exists():
        raise RuntimeError(f"Run dir not found: {run_dir}")
    if not genui_path.exists():
        raise RuntimeError(f"Missing genui.jsonl: {genui_path}")

    ensure_device_connected(args.adb_bin)
    # This capture replaces the screenshot directory, so its canonical metric
    # sidecar must not remain stale if the new device attempt fails midway.
    (run_dir / "native_render_checks.jsonl").unlink(missing_ok=True)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_base = args.device_base_dir.strip() or (
        f"/sdcard/Android/data/{args.package}/files/dataset_capture/{args.run_id}"
    )
    device_input_dir = f"{device_base}/data"
    device_assets_dir = f"{device_base}/assets"
    device_output_dir = f"{device_base}/screenshots"
    device_input_file = f"{device_input_dir}/genui.jsonl"
    device_complete_file = f"{device_output_dir}/_COMPLETE.json"
    device_error_file = f"{device_output_dir}/_ERROR.txt"

    adb(args.adb_bin, "shell", "rm", "-rf", device_base)
    adb(args.adb_bin, "shell", "mkdir", "-p", device_input_dir, device_assets_dir, device_output_dir)
    adb(args.adb_bin, "push", str(genui_path), device_input_file)

    if assets_dir.exists():
        # This copies local assets dir to <device_base>/assets
        adb(args.adb_bin, "push", str(assets_dir), device_base)
    else:
        print(f"[warn] No assets directory found for run: {assets_dir}")

    component = f"{args.package}/.DatasetRenderCaptureActivity"
    adb(
        args.adb_bin,
        "shell",
        "am",
        "start",
        "-n",
        component,
        "--es",
        "input_jsonl_path",
        device_input_file,
        "--es",
        "output_dir",
        device_output_dir,
        "--ei",
        "start_index",
        str(args.start_index),
        "--ei",
        "max_count",
        str(args.max_count),
        "--ei",
        "settle_ms",
        str(args.settle_ms),
    )

    deadline = time.time() + max(30, args.timeout_sec)
    while time.time() < deadline:
        if adb_file_exists(args.adb_bin, device_complete_file):
            break
        if adb_file_exists(args.adb_bin, device_error_file):
            error_text = adb(args.adb_bin, "shell", "cat", device_error_file, check=False).stdout.strip()
            raise RuntimeError(f"Android capture failed: {error_text or 'unknown error'}")
        time.sleep(2.0)
    else:
        raise RuntimeError(
            f"Timed out waiting for capture completion after {args.timeout_sec}s. "
            f"Expected marker: {device_complete_file}"
        )

    adb(args.adb_bin, "pull", f"{device_output_dir}/.", str(output_dir))

    complete_local = output_dir / "_COMPLETE.json"
    if not complete_local.exists():
        # Pull complete marker explicitly for older adb variations.
        adb(args.adb_bin, "pull", device_complete_file, str(output_dir / "_COMPLETE.json"))

    if complete_local.exists():
        complete_payload = complete_local.read_text(encoding="utf-8", errors="replace").strip()
        if complete_payload:
            try:
                parsed = json.loads(complete_payload)
                print("[info] completion:", json.dumps(parsed, ensure_ascii=False))
            except Exception:
                print("[info] completion:", complete_payload)

    captured_png = count_png_files(output_dir)
    print(f"[ok] Captured {captured_png} screenshot(s) to: {output_dir}")
    published_checks = write_native_render_checks(run_dir, output_dir)
    print(
        f"[ok] Published {published_checks} native renderer check(s) to: "
        f"{run_dir / 'native_render_checks.jsonl'}"
    )

    if not args.keep_remote:
        adb(args.adb_bin, "shell", "rm", "-rf", device_base)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
