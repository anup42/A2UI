from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def tail_lines(path: Path, max_lines: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return text.splitlines()[-max_lines:]


def last_matching(path: Path, token: str) -> str:
    for line in reversed(tail_lines(path)):
        if token in line:
            return line
    return ""


def pid_alive(pid: str) -> bool:
    pid = str(pid or "").strip()
    if not pid.isdigit():
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return pid in out


def read_pid(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="ascii", errors="ignore").strip().splitlines()[0]
    except Exception:
        return ""


def shard_status(runs_root: Path, run_group: str, index: int, per_shard_target: int) -> dict:
    part = f"{index:02d}"
    run_id = f"{run_group}_part{part}"
    run_dir = runs_root / run_id
    cycle_log = run_dir / "cycle_generation.log"
    pid = read_pid(run_dir / "cycle_generation.pid")
    return {
        "part": part,
        "run_id": run_id,
        "controller_pid": pid,
        "controller_alive": pid_alive(pid),
        "queries": count_jsonl(run_dir / "queries.jsonl"),
        "responses": count_jsonl(run_dir / "responses.jsonl"),
        "genui": count_jsonl(run_dir / "genui.jsonl"),
        "target": per_shard_target,
        "latest_stage_start": last_matching(cycle_log, "START stage"),
        "latest_stage_end": last_matching(cycle_log, "END stage"),
        "latest_cycle": last_matching(cycle_log, "CYCLE "),
    }


def write_status(args: argparse.Namespace, dataset_root: Path) -> bool:
    runs_root = dataset_root / "data" / "runs"
    group_dir = runs_root / args.run_group
    group_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        shard_status(runs_root, args.run_group, index, args.per_shard_target)
        for index in range(1, args.shards + 1)
    ]
    total_target = args.shards * args.per_shard_target
    totals = {
        "queries": sum(int(row["queries"]) for row in rows),
        "responses": sum(int(row["responses"]) for row in rows),
        "genui": sum(int(row["genui"]) for row in rows),
    }
    payload = {
        "run_group": args.run_group,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "monitor_pid": os.getpid(),
        "active_controllers": sum(1 for row in rows if row["controller_alive"]),
        "target_total": total_target,
        "totals": totals,
        "complete": all(value >= total_target for value in totals.values()),
        "shards": rows,
    }
    (group_dir / "hourly_status_latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    line = (
        f"[{datetime.now().isoformat()}] total "
        f"queries={totals['queries']}/{total_target} "
        f"responses={totals['responses']}/{total_target} "
        f"ir={totals['genui']}/{total_target} "
        f"active_controllers={payload['active_controllers']}"
    )
    with (group_dir / "hourly_status.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        for row in rows:
            handle.write(
                f"  part{row['part']}: "
                f"q={row['queries']}/{row['target']} "
                f"r={row['responses']}/{row['target']} "
                f"ir={row['genui']}/{row['target']} "
                f"alive={row['controller_alive']}\n"
            )
    return bool(payload["complete"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hourly status monitor for GPT-5.4 parallel dataset generation.")
    parser.add_argument("--run-group", default="dataset_gpt54_no_reasoning_p5_20260619")
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--per-shard-target", type=int, default=4000)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-when-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(__file__).resolve().parents[1]
    group_dir = dataset_root / "data" / "runs" / args.run_group
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "hourly_monitor.pid").write_text(str(os.getpid()), encoding="ascii")
    while True:
        complete = write_status(args, dataset_root)
        if args.once or (args.stop_when_complete and complete):
            return
        time.sleep(max(60, int(args.interval_seconds)))


if __name__ == "__main__":
    main()
