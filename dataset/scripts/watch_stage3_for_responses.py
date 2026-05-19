from __future__ import annotations

import argparse
import json
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
from pipeline.stage3_genui import _make_ui_id, run_stage3  # noqa: E402
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


def expected_ui_ids(responses_path: Path) -> set[str]:
    expected: set[str] = set()
    for row in iter_jsonl(responses_path):
        query_id = row.get("query_id")
        response_id = row.get("response_id")
        response_text = row.get("response_text")
        if not query_id or not response_id or not response_text:
            continue
        n_idx = int(row.get("n_idx", 1))
        expected.add(_make_ui_id(str(query_id), n_idx, 1))
    return expected


def existing_ui_ids(genui_path: Path) -> set[str]:
    return {str(row.get("ui_id")) for row in iter_jsonl(genui_path) if row.get("ui_id")}


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch responses.jsonl and generate missing Stage 3 IR.")
    parser.add_argument("--run_id", default="dataset_v1")
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--model", default="azure_gpt54_mini")
    parser.add_argument("--pass_size", type=int, default=8)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--idle_checks", type=int, default=5)
    parser.add_argument("--rate_limit_qps", type=float, default=None)
    parser.add_argument("--call_sleep_seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(DATASET_ROOT)

    run_cfg = load_yaml(DATASET_ROOT / "configs" / "run.yaml").get("run", {})
    eval_cfg = load_yaml(DATASET_ROOT / "configs" / "run.yaml").get("evaluation", {})
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
    cache = PromptCache(run_paths.run_dir / ".prompt_cache_stage3_watch.jsonl")
    progress_path = run_paths.run_dir / "progress_stage3_watch.json"

    schema_path = DATASET_ROOT / run_cfg.get("stage3_schema_file", "schema/genui_flatspec.schema.json")
    if not schema_path.exists():
        schema_path = DATASET_ROOT / "schema" / "genui_flatspec.schema.json"
    prompt_path = DATASET_ROOT / run_cfg.get("stage3_prompt_file", "prompts/genui_gen_mobile_flatspec_v11.md")
    if not prompt_path.exists():
        prompt_path = DATASET_ROOT / "prompts" / "genui_gen_mobile_flatspec_v11.md"

    logger.info(
        "Stage3 response watcher started run_id=%s target=%s pass_size=%s model=%s",
        args.run_id,
        args.target,
        args.pass_size,
        args.model,
    )
    idle_count = 0
    last_counts: dict[str, int] | None = None
    while True:
        expected = expected_ui_ids(run_paths.responses_path)
        existing = existing_ui_ids(run_paths.genui_path)
        missing = expected - existing
        counts = {
            "queries": count_jsonl(run_paths.queries_path),
            "responses": len(expected),
            "genui": len(existing),
            "missing_ir": len(missing),
        }
        write_progress(
            progress_path,
            {
                "run_id": args.run_id,
                "target": args.target,
                "counts": counts,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            },
        )

        if missing:
            before = len(existing)
            max_total = min(max(1, args.pass_size), len(missing))
            run_stage3(
                queries_path=run_paths.queries_path,
                responses_path=run_paths.responses_path,
                prompt_path=prompt_path,
                adapter=adapter,
                genui_path=run_paths.genui_path,
                schema_path=schema_path,
                artifacts_dir=run_paths.artifacts_dir,
                candidates_per_response=1,
                max_repair_attempts=int(run_cfg.get("max_repair_attempts", 1)),
                max_tokens=int(run_cfg.get("genui_max_tokens", 8192)),
                prompt_max_tokens=int(run_cfg.get("genui_prompt_max_tokens", 60000)),
                batch_size=int(run_cfg.get("genui_batch_size", 1)),
                seed=int(run_cfg.get("seed", 42)),
                rate_limiter=rate_limiter,
                cache=cache,
                logger=logger,
                max_total=max_total,
                max_attempts=int(run_cfg.get("max_attempts", 6)),
                aggregates_path=run_paths.aggregates_path,
                aggregate_weights=eval_cfg.get("weights", {}),
            )
            after = count_jsonl(run_paths.genui_path)
            logger.info(
                "Stage3 watcher progress genui=%s responses=%s created=%s",
                after,
                counts["responses"],
                after - before,
            )
            idle_count = 0
            if after <= before:
                logger.warning("Stage3 watcher made no progress with missing_ir=%s; sleeping", len(missing))
                time.sleep(max(1.0, args.poll_seconds))
            continue

        if counts["responses"] >= args.target:
            if last_counts == counts:
                idle_count += 1
            else:
                idle_count = 1
            logger.info(
                "Stage3 watcher caught up idle_count=%s/%s counts=%s",
                idle_count,
                args.idle_checks,
                counts,
            )
            if idle_count >= max(1, args.idle_checks):
                logger.info("Stage3 watcher complete counts=%s", counts)
                return
        else:
            logger.info("Stage3 watcher caught up; waiting for new responses counts=%s", counts)
            idle_count = 0

        last_counts = counts
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
