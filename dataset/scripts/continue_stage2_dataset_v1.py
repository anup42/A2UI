from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DATASET_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm.factory import build_adapter, load_model_specs  # noqa: E402
from pipeline.cache import PromptCache  # noqa: E402
from pipeline.stage1_queries import run_stage1  # noqa: E402
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
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def count_intents(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue dataset_v1 source generation. It fills Stage 2 for existing "
            "queries, then extends Stage 1 by query chunks and fills Stage 2 again."
        )
    )
    parser.add_argument("--run_id", default="dataset_v1")
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--model", default="azure_gpt54_mini")
    parser.add_argument("--pass_size", type=int, default=32)
    parser.add_argument("--query_chunk_size", type=int, default=10000)
    parser.add_argument(
        "--stage1_per_intent_batch_size",
        type=int,
        default=10,
        help="Maximum number of queries requested per intent in one Stage 1 API call.",
    )
    parser.add_argument("--sleep_seconds", type=float, default=5.0)
    parser.add_argument("--rate_limit_qps", type=float, default=None)
    parser.add_argument("--call_sleep_seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(DATASET_ROOT)

    run_cfg = load_yaml(DATASET_ROOT / "configs" / "run.yaml").get("run", {})
    model_specs = load_model_specs(load_yaml(DATASET_ROOT / "configs" / "models.yaml"))
    model_map = {spec.name: spec for spec in model_specs}
    spec = model_map.get(args.model)
    if spec is None:
        raise SystemExit(f"Unknown model {args.model}. Available: {', '.join(sorted(model_map))}")

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
    cache = PromptCache(run_paths.run_dir / ".prompt_cache.jsonl")
    progress_path = run_paths.run_dir / "progress_stage2_watch.json"
    intents_file = DATASET_ROOT / run_cfg.get("intents_file", "intents.info")
    intent_count = count_intents(intents_file)
    if intent_count <= 0:
        raise SystemExit(f"No intents found in {intents_file}")
    stage1_intent_batch_size = min(intent_count, int(run_cfg.get("stage1_intent_batch_size", intent_count)))

    logger.info(
        "Stage2 continuation started run_id=%s target=%s pass_size=%s query_chunk_size=%s model=%s",
        args.run_id,
        args.target,
        args.pass_size,
        args.query_chunk_size,
        args.model,
    )
    while True:
        queries = count_jsonl(run_paths.queries_path)
        responses = count_jsonl(run_paths.responses_path)
        counts = {"queries": queries, "responses": responses, "genui": count_jsonl(run_paths.genui_path)}
        write_progress(
            progress_path,
            {
                "run_id": args.run_id,
                "target": args.target,
                "counts": counts,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            },
        )
        if responses >= args.target:
            logger.info("Stage2 continuation complete counts=%s", counts)
            return
        if queries <= responses:
            next_query_target = min(args.target, queries + max(1, args.query_chunk_size))
            if next_query_target <= queries:
                logger.info("Stage2 waiting for queries counts=%s", counts)
                time.sleep(max(1.0, args.sleep_seconds))
                continue
            logger.info(
                "Stage2 source exhausted at responses=%s; extending Stage1 queries to %s",
                responses,
                next_query_target,
            )
            while count_jsonl(run_paths.queries_path) < next_query_target:
                before_queries = count_jsonl(run_paths.queries_path)
                stage1_batch_size = max(
                    1,
                    min(
                        int(args.stage1_per_intent_batch_size),
                        math.ceil((next_query_target - before_queries) / max(1, stage1_intent_batch_size)),
                    ),
                )
                k_per_intent = max(
                    int(run_cfg.get("k_queries_per_intent", 1)),
                    math.ceil(next_query_target / intent_count),
                )
                run_stage1(
                    intents_file=intents_file,
                    prompt_path=DATASET_ROOT / "prompts" / "query_gen.md",
                    adapter=adapter,
                    run_dir=run_paths.run_dir,
                    queries_path=run_paths.queries_path,
                    k_per_intent=k_per_intent,
                    batch_size=stage1_batch_size,
                    intent_batch_size=stage1_intent_batch_size,
                    seed=int(run_cfg.get("seed", 42)),
                    temperature=0.7,
                    max_tokens=int(run_cfg.get("query_max_tokens", 2048)),
                    rate_limiter=rate_limiter,
                    cache=cache,
                    logger=logger,
                    max_total=next_query_target - before_queries,
                    max_failures_per_intent=int(run_cfg.get("stage1_max_failures_per_intent", 50)),
                    fill_missing_with_fallback=False,
                    max_attempts=int(run_cfg.get("max_attempts", 6)),
                )
                after_queries = count_jsonl(run_paths.queries_path)
                logger.info(
                    "Stage1 extension progress queries=%s/%s created=%s",
                    after_queries,
                    next_query_target,
                    after_queries - before_queries,
                )
                if after_queries <= before_queries:
                    raise RuntimeError(
                        f"Stage1 extension made no progress toward {next_query_target}; current={after_queries}"
                    )
            continue

        max_total = min(max(1, args.pass_size), args.target - responses, queries - responses)
        before = responses
        run_stage2(
            queries_path=run_paths.queries_path,
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
            seed=int(run_cfg.get("seed", 42)),
            rate_limiter=rate_limiter,
            cache=cache,
            logger=logger,
            max_total=max_total,
            max_attempts=int(run_cfg.get("max_attempts", 6)),
        )
        after = count_jsonl(run_paths.responses_path)
        logger.info("Stage2 continuation progress responses=%s/%s created=%s", after, args.target, after - before)
        if after <= before:
            logger.warning("Stage2 continuation made no progress; sleeping counts=%s", counts)
            time.sleep(max(1.0, args.sleep_seconds))


if __name__ == "__main__":
    main()
