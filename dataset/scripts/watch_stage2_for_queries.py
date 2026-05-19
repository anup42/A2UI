from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DATASET_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm.factory import build_adapter, load_model_specs  # noqa: E402
from pipeline.cache import PromptCache  # noqa: E402
from pipeline.stage2_responses import run_stage2  # noqa: E402
from pipeline.storage import get_run_paths, iter_jsonl  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.logging import setup_logger  # noqa: E402
from utils.rate_limit import RateLimiter  # noqa: E402


def load_env(root: Path) -> None:
    for env_path in (root / ".env", root.parent / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if key.strip() and key.strip() not in os.environ:
                os.environ[key.strip()] = value


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def query_number(query_id: str) -> int:
    match = re.search(r"(\d+)$", query_id)
    return int(match.group(1)) if match else 0


def make_response_id(query_id: str) -> str:
    return f"r_{query_number(query_id):06d}_01"


def load_assigned_queries(queries_path: Path, worker_index: int, worker_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(queries_path):
        query_id = row.get("query_id")
        if not isinstance(query_id, str):
            continue
        if query_number(query_id) % worker_count == worker_index:
            rows.append(row)
    return rows


def existing_response_ids(responses_path: Path) -> set[str]:
    return {str(row.get("response_id")) for row in iter_jsonl(responses_path) if row.get("response_id")}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch Stage 1 queries and generate Stage 2 responses for a shard.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--model", default="azure_gpt54_mini")
    parser.add_argument("--worker_index", type=int, required=True)
    parser.add_argument("--worker_count", type=int, default=2)
    parser.add_argument("--pass_size", type=int, default=8)
    parser.add_argument("--poll_seconds", type=float, default=30.0)
    parser.add_argument("--rate_limit_qps", type=float, default=None)
    parser.add_argument("--call_sleep_seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(DATASET_ROOT)
    run_cfg = load_yaml(DATASET_ROOT / "configs" / "run.yaml").get("run", {})
    model_specs = load_model_specs(load_yaml(DATASET_ROOT / "configs" / "models.yaml"))
    spec = {model.name: model for model in model_specs}.get(args.model)
    if spec is None:
        raise SystemExit(f"Unknown model {args.model}")

    output_dir = Path(run_cfg.get("output_dir", "data/runs"))
    if not output_dir.is_absolute():
        output_dir = DATASET_ROOT / output_dir
    run_paths = get_run_paths(output_dir, args.run_id, run_cfg.get("artifact_dir", "artifacts"))
    logger = setup_logger(run_paths.run_dir)
    adapter = build_adapter(spec)
    rate_limiter = RateLimiter(
        float(args.rate_limit_qps if args.rate_limit_qps is not None else run_cfg.get("rate_limit_qps", 1.0)),
        float(
            args.call_sleep_seconds
            if args.call_sleep_seconds is not None
            else run_cfg.get("call_sleep_seconds", 1.0)
        ),
    )
    cache = PromptCache(run_paths.run_dir / f".prompt_cache_stage2_worker_{args.worker_index}.jsonl")
    shard_queries_path = run_paths.run_dir / f".stage2_worker_{args.worker_index}_queries.jsonl"
    progress_path = run_paths.run_dir / f"progress_stage2_worker_{args.worker_index}.json"
    logger.info(
        "Stage2 worker started run_id=%s worker=%s/%s target=%s",
        args.run_id,
        args.worker_index,
        args.worker_count,
        args.target,
    )

    while True:
        assigned = load_assigned_queries(run_paths.queries_path, args.worker_index, args.worker_count)
        existing = existing_response_ids(run_paths.responses_path)
        pending = [row for row in assigned if make_response_id(str(row.get("query_id"))) not in existing]
        counts = {
            "queries": count_jsonl(run_paths.queries_path),
            "assigned_queries": len(assigned),
            "responses": count_jsonl(run_paths.responses_path),
            "pending_assigned": len(pending),
        }
        write_progress(
            progress_path,
            {
                "run_id": args.run_id,
                "target": args.target,
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "counts": counts,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            },
        )
        if counts["responses"] >= args.target:
            logger.info("Stage2 worker complete counts=%s", counts)
            return
        if not pending:
            logger.info("Stage2 worker waiting for assigned queries counts=%s", counts)
            time.sleep(max(1.0, args.poll_seconds))
            continue

        write_jsonl(shard_queries_path, assigned)
        before = count_jsonl(run_paths.responses_path)
        max_total = min(max(1, args.pass_size), args.target - before, len(pending))
        run_stage2(
            queries_path=shard_queries_path,
            prompt_path=DATASET_ROOT / "prompts" / "response_gen.md",
            batch_prompt_path=DATASET_ROOT / "prompts" / "response_gen_batch.md",
            adapter=adapter,
            responses_path=run_paths.responses_path,
            n_per_query=1,
            batch_size=int(run_cfg.get("response_batch_size", 1)),
            query_batch_size=int(run_cfg.get("query_batch_size", 1)),
            group_by_intent=bool(run_cfg.get("response_group_by_intent", False)),
            batch_fallback_per_query=bool(run_cfg.get("response_batch_fallback_per_query", True)),
            temperatures=[0.7],
            max_tokens=int(run_cfg.get("response_max_tokens", 4096)),
            seed=int(run_cfg.get("seed", 42)) + args.worker_index,
            rate_limiter=rate_limiter,
            cache=cache,
            logger=logger,
            max_total=max_total,
            max_attempts=int(run_cfg.get("max_attempts", 6)),
        )
        after = count_jsonl(run_paths.responses_path)
        logger.info("Stage2 worker progress responses=%s/%s created=%s", after, args.target, after - before)
        if after <= before:
            logger.warning("Stage2 worker made no progress; sleeping counts=%s", counts)
            time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
