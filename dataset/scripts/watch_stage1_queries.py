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
from llm.base import LLMRateLimitError  # noqa: E402
from pipeline.cache import PromptCache  # noqa: E402
from pipeline.stage1_queries import run_stage1  # noqa: E402
from pipeline.storage import get_run_paths, iter_jsonl  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.logging import setup_logger  # noqa: E402
from utils.rate_limit import RateLimiter  # noqa: E402
from utils.versioning import build_run_manifest, write_run_manifest  # noqa: E402


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


def count_intents(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rate_limit_sleep_seconds(exc: LLMRateLimitError, default: float = 90.0) -> float:
    headers = exc.headers or {}
    values: list[float] = []
    for key in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        try:
            values.append(float(headers.get(key, 0)))
        except (TypeError, ValueError):
            pass
    return max([default, *values]) + 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 1 queries for a dataset run.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--model", default="azure_gpt54_mini")
    parser.add_argument("--pass_size", type=int, default=320)
    parser.add_argument("--stage1_per_intent_batch_size", type=int, default=10)
    parser.add_argument("--rate_limit_qps", type=float, default=None)
    parser.add_argument("--call_sleep_seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(DATASET_ROOT)
    run_cfg_path = DATASET_ROOT / "configs" / "run.yaml"
    models_cfg_path = DATASET_ROOT / "configs" / "models.yaml"
    run_cfg = load_yaml(run_cfg_path).get("run", {})
    model_specs = load_model_specs(load_yaml(models_cfg_path))
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
    cache = PromptCache(run_paths.run_dir / ".prompt_cache_stage1_worker.jsonl")
    intents_file = DATASET_ROOT / run_cfg.get("intents_file", "intents.info")
    intent_count = count_intents(intents_file)
    if intent_count <= 0:
        raise SystemExit(f"No intents found in {intents_file}")
    stage1_intent_batch_size = min(intent_count, int(run_cfg.get("stage1_intent_batch_size", intent_count)))
    manifest = build_run_manifest(
        root=DATASET_ROOT,
        run_id=args.run_id,
        stage="stage1_watch",
        model_spec=spec,
        run_paths=run_paths,
        run_cfg_path=run_cfg_path,
        models_cfg_path=models_cfg_path,
        argv=sys.argv,
    )
    manifest["dataset_worker_settings"] = {
        "run_id": args.run_id,
        "stage": "stage1",
        "target": args.target,
        "pass_size": args.pass_size,
        "model": args.model,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }
    write_run_manifest(run_paths.manifest_path, manifest)

    progress_path = run_paths.run_dir / "progress_stage1_watch.json"
    logger.info("Stage1 watcher started run_id=%s target=%s pass_size=%s", args.run_id, args.target, args.pass_size)
    while count_jsonl(run_paths.queries_path) < args.target:
        before = count_jsonl(run_paths.queries_path)
        remaining_total = args.target - before
        source_target = min(args.target, before + max(1, args.pass_size))
        stage1_batch_size = max(
            1,
            min(
                int(args.stage1_per_intent_batch_size),
                math.ceil((source_target - before) / max(1, stage1_intent_batch_size)),
            ),
        )
        k_per_intent = max(int(run_cfg.get("k_queries_per_intent", 1)), math.ceil(source_target / intent_count))
        write_progress(
            progress_path,
            {
                "run_id": args.run_id,
                "target": args.target,
                "queries": before,
                "remaining": remaining_total,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            },
        )
        try:
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
                max_total=source_target - before,
                max_failures_per_intent=int(run_cfg.get("stage1_max_failures_per_intent", 50)),
                fill_missing_with_fallback=False,
                max_attempts=int(run_cfg.get("max_attempts", 6)),
            )
        except LLMRateLimitError as exc:
            sleep_seconds = rate_limit_sleep_seconds(exc)
            logger.warning("Stage1 rate limited; sleeping %.1fs before retry", sleep_seconds)
            time.sleep(sleep_seconds)
            continue
        after = count_jsonl(run_paths.queries_path)
        logger.info("Stage1 progress queries=%s/%s created=%s", after, args.target, after - before)
        if after <= before:
            logger.warning("Stage1 made no progress; sleeping")
            time.sleep(30)
    write_progress(
        progress_path,
        {
            "run_id": args.run_id,
            "target": args.target,
            "queries": count_jsonl(run_paths.queries_path),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        },
    )
    logger.info("Stage1 watcher complete run_id=%s target=%s", args.run_id, args.target)


if __name__ == "__main__":
    main()
