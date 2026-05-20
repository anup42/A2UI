from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_ROOT.parent


def load_env() -> None:
    for env_path in (DATASET_ROOT / ".env", REPO_ROOT / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def split_keys(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.replace(";", ",").replace("\n", ",").split(",") if item.strip()]


def log(message: str, log_path: Path) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    line = f"{stamp} {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def common_env(keys: list[str], batch_prompts: int, batch_output_tokens: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GEMINI_API_KEYS": ",".join(keys),
            "GEMINI_EXPRESS_SINGLE_CALL_BATCH_MAX_PROMPTS": str(batch_prompts),
            "GEMINI_EXPRESS_SINGLE_CALL_BATCH_MAX_OUTPUT_TOKENS": str(batch_output_tokens),
            "GEMINI_EXPRESS_SINGLE_CALL_BATCH_MAX_CHARS": os.getenv("GEMINI_EXPRESS_SINGLE_CALL_BATCH_MAX_CHARS", "500000"),
            "GEMINI_EXPRESS_SINGLE_CALL_BATCH_SEQUENTIAL_FALLBACK": "0",
            "GEMINI_DISABLE_KEY_ON_RATE_LIMIT": "1",
            "GEMINI_RATE_LIMIT_DISABLE_AFTER": os.getenv("GEMINI_RATE_LIMIT_DISABLE_AFTER", "2"),
            "GEMINI_RATE_LIMIT_MAX_RETRIES": os.getenv("GEMINI_RATE_LIMIT_MAX_RETRIES", "2"),
            "GEMINI_REQUEST_CYCLES": os.getenv("GEMINI_REQUEST_CYCLES", "1"),
            "GEMINI_TIMEOUT_SECONDS": os.getenv("GEMINI_TIMEOUT_SECONDS", "600"),
            "GEMINI_TIMEOUT_MAX_RETRIES": os.getenv("GEMINI_TIMEOUT_MAX_RETRIES", "1"),
        }
    )
    return env


def start_worker(
    run_id: str,
    stage: str,
    target: int,
    keys: list[str],
    log_path: Path,
) -> subprocess.Popen:
    run_dir = DATASET_ROOT / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "stage2":
        args = [
            sys.executable,
            "dataset/scripts/watch_stage2_for_queries.py",
            "--run_id",
            run_id,
            "--target",
            str(target),
            "--model",
            "gemini_3_1_flash_lite",
            "--worker_index",
            "0",
            "--worker_count",
            "1",
            "--pass_size",
            "64",
            "--poll_seconds",
            "60",
            "--rate_limit_qps",
            "0.2",
            "--call_sleep_seconds",
            "1",
        ]
        env = common_env(keys, batch_prompts=16, batch_output_tokens=32768)
    elif stage == "stage3":
        args = [
            sys.executable,
            "dataset/scripts/watch_stage3_for_responses.py",
            "--run_id",
            run_id,
            "--target",
            str(target),
            "--model",
            "gemini_3_1_flash_lite",
            "--worker_index",
            "0",
            "--worker_count",
            "1",
            "--pass_size",
            "64",
            "--poll_seconds",
            "60",
            "--idle_checks",
            "5",
            "--rate_limit_qps",
            "0.2",
            "--call_sleep_seconds",
            "1",
        ]
        env = common_env(keys, batch_prompts=8, batch_output_tokens=65536)
    else:
        raise ValueError(stage)

    stdout = (run_dir / f"{stage}_gemini31_flash_lite_managed.out.log").open("a", encoding="utf-8")
    stderr = (run_dir / f"{stage}_gemini31_flash_lite_managed.err.log").open("a", encoding="utf-8")
    proc = subprocess.Popen(
        args,
        cwd=str(REPO_ROOT),
        stdout=stdout,
        stderr=stderr,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    log(f"started {stage} run={run_id} pid={proc.pid} keys={len(keys)}", log_path)
    return proc


def stop_worker(proc: subprocess.Popen | None, stage: str, log_path: Path) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    log(f"stopped {stage} pid={proc.pid}", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Gemini Stage 2/3 workers for dataset_v3 then dataset_v1.")
    parser.add_argument("--runs", default="dataset_v3,dataset_v1")
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--poll_seconds", type=float, default=300.0)
    args = parser.parse_args()

    load_env()
    stage2_keys = split_keys(os.getenv("GEMINI_STAGE2_API_KEYS"))
    stage3_keys = split_keys(os.getenv("GEMINI_STAGE3_API_KEYS"))
    all_keys = split_keys(os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_AI_API_KEYS"))
    if not stage2_keys and all_keys:
        stage2_keys = all_keys[: max(1, (len(all_keys) + 1) // 2)]
    if not stage3_keys and all_keys:
        stage3_keys = all_keys[max(1, (len(all_keys) + 1) // 2) :]
    if not stage2_keys or not stage3_keys:
        raise SystemExit("Set GEMINI_STAGE2_API_KEYS and GEMINI_STAGE3_API_KEYS in dataset/.env")

    runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    log_path = DATASET_ROOT / "data" / "runs" / "gemini_worker_manager.log"
    stage2_proc: subprocess.Popen | None = None
    stage3_proc: subprocess.Popen | None = None
    current_index = 0

    while current_index < len(runs):
        run_id = runs[current_index]
        run_dir = DATASET_ROOT / "data" / "runs" / run_id
        queries = count_jsonl(run_dir / "queries.jsonl")
        responses = count_jsonl(run_dir / "responses.jsonl")
        genui = count_jsonl(run_dir / "genui.jsonl")
        log(f"status run={run_id} queries={queries} responses={responses} genui={genui}", log_path)

        if responses < args.target and queries > 0:
            if stage2_proc is None or stage2_proc.poll() is not None:
                stage2_proc = start_worker(run_id, "stage2", args.target, stage2_keys, log_path)
        else:
            stop_worker(stage2_proc, "stage2", log_path)
            stage2_proc = None

        if genui < min(args.target, max(responses, 0)) and responses > 0:
            if stage3_proc is None or stage3_proc.poll() is not None:
                stage3_proc = start_worker(run_id, "stage3", args.target, stage3_keys, log_path)
        else:
            stop_worker(stage3_proc, "stage3", log_path)
            stage3_proc = None

        if responses >= args.target and genui >= args.target:
            stop_worker(stage2_proc, "stage2", log_path)
            stop_worker(stage3_proc, "stage3", log_path)
            stage2_proc = None
            stage3_proc = None
            log(f"completed run={run_id}; moving next", log_path)
            current_index += 1
            continue

        time.sleep(max(10.0, args.poll_seconds))

    log("all requested runs complete", log_path)


if __name__ == "__main__":
    main()
