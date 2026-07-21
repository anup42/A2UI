from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_ROOT.parent
SRC_ROOT = DATASET_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm.factory import build_adapter, load_model_specs  # noqa: E402
from pipeline.cache import PromptCache  # noqa: E402
from pipeline.stage1_queries import run_stage1  # noqa: E402
from pipeline.stage2_responses import run_stage2  # noqa: E402
from pipeline.stage3_genui import run_stage3  # noqa: E402
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


def env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    return int(raw)


def env_float_list(name: str, fallback: list[float]) -> list[float]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return fallback
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    return values or fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset_v1 in aligned Stage1->Stage2->Stage3 chunks."
    )
    parser.add_argument("--run_id", default="dataset_v1")
    parser.add_argument("--target", type=int, default=32000)
    parser.add_argument("--chunk_size", type=int, default=320)
    parser.add_argument("--model", default="azure_gpt54_mini")
    parser.add_argument("--rate_limit_qps", type=float, default=None)
    parser.add_argument("--call_sleep_seconds", type=float, default=None)
    parser.add_argument(
        "--stage1_per_intent_batch_size",
        type=int,
        default=10,
        help="Maximum number of queries requested per intent in one Stage 1 API call.",
    )
    parser.add_argument(
        "--max_extra_source",
        type=int,
        default=1000,
        help=(
            "Extra query/response budget used only when Stage 3 has permanently "
            "failed records but still needs more valid IR rows."
        ),
    )
    parser.add_argument(
        "--replacement_batch_size",
        type=int,
        default=8,
        help="How many extra query/response records to add after a Stage 3 no-progress pass.",
    )
    parser.add_argument("--max_cycles", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(DATASET_ROOT)

    run_cfg_path = DATASET_ROOT / "configs" / "run.yaml"
    models_cfg_path = DATASET_ROOT / "configs" / "models.yaml"
    run_yaml = load_yaml(run_cfg_path)
    run_cfg = run_yaml.get("run", {})
    eval_cfg = run_yaml.get("evaluation", {})
    model_specs = load_model_specs(load_yaml(models_cfg_path))
    model_map = {spec.name: spec for spec in model_specs}
    spec = model_map.get(args.model)
    if spec is None:
        raise SystemExit(f"Unknown model {args.model}. Available: {', '.join(sorted(model_map))}")

    output_dir = Path(run_cfg.get("output_dir", "data/runs"))
    if not output_dir.is_absolute():
        output_dir = DATASET_ROOT / output_dir
    run_paths = get_run_paths(output_dir, args.run_id, run_cfg.get("artifact_dir", "artifacts"))
    logger = setup_logger(run_paths.run_dir)

    intents_file = DATASET_ROOT / run_cfg.get("intents_file", "intents.info")
    intent_count = count_intents(intents_file)
    if intent_count <= 0:
        raise SystemExit(f"No intents found in {intents_file}")

    stage1_intent_batch_size = min(intent_count, int(run_cfg.get("stage1_intent_batch_size", intent_count)))
    # Keep per-intent calls small enough for reliable JSON parsing even when the
    # requested chunk is large (for example, 10k sequential dataset generation).
    stage1_batch_size = max(
        1,
        min(
            int(args.stage1_per_intent_batch_size),
            math.ceil(args.chunk_size / max(1, stage1_intent_batch_size)),
        ),
    )
    source_ceiling = args.target + max(0, int(args.max_extra_source))
    target_per_intent = math.ceil(source_ceiling / intent_count)
    k_per_intent = max(int(run_cfg.get("k_queries_per_intent", 1)), target_per_intent)

    rate_limit_qps = float(args.rate_limit_qps if args.rate_limit_qps is not None else run_cfg.get("rate_limit_qps", 1.0))
    call_sleep_seconds = float(
        args.call_sleep_seconds
        if args.call_sleep_seconds is not None
        else run_cfg.get("call_sleep_seconds", 1.0)
    )
    query_temperature = env_float("A2UI_QUERY_TEMPERATURE", 0.7)
    response_temperature_env = (
        "A2UI_RESPONSE_TEMPERATURES"
        if (os.environ.get("A2UI_RESPONSE_TEMPERATURES") or "").strip()
        else "A2UI_RESPONSE_TEMPERATURE"
    )
    response_temperatures = env_float_list(response_temperature_env, [0.7])
    query_max_tokens = env_int("A2UI_QUERY_MAX_TOKENS", int(run_cfg.get("query_max_tokens", 2048)))
    response_max_tokens = env_int("A2UI_RESPONSE_MAX_TOKENS", int(run_cfg.get("response_max_tokens", 4096)))

    settings = {
        "run_id": args.run_id,
        "target": args.target,
        "chunk_size": args.chunk_size,
        "max_extra_source": max(0, int(args.max_extra_source)),
        "replacement_batch_size": max(1, int(args.replacement_batch_size)),
        "model": args.model,
        "intent_count": intent_count,
        "k_per_intent": k_per_intent,
        "target_per_intent": target_per_intent,
        "stage1_batch_size": stage1_batch_size,
        "stage1_intent_batch_size": stage1_intent_batch_size,
        "stage2_response_batch_size": 1,
        "stage2_query_batch_size": 1,
        "stage3_batch_size": 1,
        "rate_limit_qps": rate_limit_qps,
        "call_sleep_seconds": call_sleep_seconds,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }

    manifest = build_run_manifest(
        root=DATASET_ROOT,
        run_id=args.run_id,
        stage="dataset_v1_chunked",
        model_spec=spec,
        run_paths=run_paths,
        run_cfg_path=run_cfg_path,
        models_cfg_path=models_cfg_path,
        argv=sys.argv,
    )
    manifest["dataset_v1_settings"] = settings
    write_run_manifest(run_paths.manifest_path, manifest)

    progress_path = run_paths.run_dir / "progress.json"
    initial_counts = {
        "queries": count_jsonl(run_paths.queries_path),
        "responses": count_jsonl(run_paths.responses_path),
        "genui": count_jsonl(run_paths.genui_path),
    }
    write_progress(progress_path, {**settings, "counts": initial_counts, "dry_run": args.dry_run})
    logger.info("dataset_v1 settings: %s", json.dumps(settings, sort_keys=True))
    logger.info("dataset_v1 initial counts: %s", initial_counts)

    if args.dry_run:
        print(json.dumps({**settings, "counts": initial_counts, "run_dir": str(run_paths.run_dir)}, indent=2))
        return

    adapter = build_adapter(spec)
    rate_limiter = RateLimiter(rate_limit_qps, call_sleep_seconds)
    # Use a run-local cache so model/config experiments cannot poison the shared cache
    # with partial or incompatible generations.
    cache = PromptCache(run_paths.run_dir / ".prompt_cache.jsonl")

    prompts_dir = DATASET_ROOT / "prompts"
    schema_path = DATASET_ROOT / run_cfg.get("stage3_schema_file", "schema/genui_flatspec.schema.json")
    if not schema_path.exists():
        schema_path = DATASET_ROOT / "schema" / "genui_flatspec.schema.json"
    stage3_prompt_path = DATASET_ROOT / run_cfg.get("stage3_prompt_file", "prompts/genui_gen_mobile_flatspec_v11.md")
    if not stage3_prompt_path.exists():
        stage3_prompt_path = DATASET_ROOT / "prompts" / "genui_gen_mobile_flatspec_v11.md"

    max_cycles = args.max_cycles
    cycle = 0
    def _ensure_stage1_source(source_target: int) -> None:
        while count_jsonl(run_paths.queries_path) < source_target:
            before = count_jsonl(run_paths.queries_path)
            cycle_k_per_intent = min(k_per_intent, math.ceil(source_target / intent_count))
            logger.info("dataset_v1 stage1 cycle_k_per_intent=%s source_target=%s", cycle_k_per_intent, source_target)
            run_stage1(
                intents_file=intents_file,
                prompt_path=prompts_dir / "query_gen.md",
                adapter=adapter,
                run_dir=run_paths.run_dir,
                queries_path=run_paths.queries_path,
                k_per_intent=cycle_k_per_intent,
                batch_size=stage1_batch_size,
                intent_batch_size=stage1_intent_batch_size,
                seed=int(run_cfg.get("seed", 42)),
                temperature=query_temperature,
                max_tokens=query_max_tokens,
                rate_limiter=rate_limiter,
                cache=cache,
                logger=logger,
                max_total=source_target - before,
                max_failures_per_intent=int(run_cfg.get("stage1_max_failures_per_intent", 50)),
                fill_missing_with_fallback=False,
                max_attempts=int(run_cfg.get("max_attempts", 6)),
            )
            after = count_jsonl(run_paths.queries_path)
            logger.info("dataset_v1 progress stage1 queries=%s/%s created=%s", after, source_target, after - before)
            if after <= before:
                raise RuntimeError(f"Stage1 made no progress toward {source_target}; current={after}")

    def _ensure_stage2_source(source_target: int) -> None:
        while count_jsonl(run_paths.responses_path) < source_target:
            before = count_jsonl(run_paths.responses_path)
            run_stage2(
                queries_path=run_paths.queries_path,
                prompt_path=prompts_dir / "response_gen.md",
                batch_prompt_path=prompts_dir / "response_gen_batch.md",
                adapter=adapter,
                responses_path=run_paths.responses_path,
                n_per_query=1,
                batch_size=int(run_cfg.get("response_batch_size", 1)),
                query_batch_size=int(run_cfg.get("query_batch_size", 1)),
                group_by_intent=bool(run_cfg.get("response_group_by_intent", False)),
                batch_fallback_per_query=bool(run_cfg.get("response_batch_fallback_per_query", True)),
                temperatures=response_temperatures,
                max_tokens=response_max_tokens,
                seed=int(run_cfg.get("seed", 42)),
                rate_limiter=rate_limiter,
                cache=cache,
                logger=logger,
                max_total=source_target - before,
                max_attempts=int(run_cfg.get("max_attempts", 6)),
            )
            after = count_jsonl(run_paths.responses_path)
            logger.info("dataset_v1 progress stage2 responses=%s/%s created=%s", after, source_target, after - before)
            if after <= before:
                raise RuntimeError(f"Stage2 made no progress toward {source_target}; current={after}")

    while True:
        counts = {
            "queries": count_jsonl(run_paths.queries_path),
            "responses": count_jsonl(run_paths.responses_path),
            "genui": count_jsonl(run_paths.genui_path),
        }
        if min(counts.values()) >= args.target:
            logger.info("dataset_v1 complete counts=%s", counts)
            write_progress(progress_path, {**settings, "counts": counts, "completed_at": datetime.utcnow().isoformat() + "Z"})
            return
        if max_cycles is not None and cycle >= max_cycles:
            logger.info("dataset_v1 stopped by max_cycles=%s counts=%s", max_cycles, counts)
            write_progress(progress_path, {**settings, "counts": counts, "stopped_at": datetime.utcnow().isoformat() + "Z"})
            return

        cycle += 1
        floor_count = min(counts.values())
        cycle_target = min(args.target, floor_count + args.chunk_size)
        source_target = min(source_ceiling, max(cycle_target, counts["responses"]))
        logger.info("dataset_v1 cycle=%s target=%s counts=%s", cycle, cycle_target, counts)
        write_progress(
            progress_path,
            {
                **settings,
                "cycle": cycle,
                "cycle_target": cycle_target,
                "source_target": source_target,
                "counts": counts,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            },
        )

        _ensure_stage1_source(source_target)
        _ensure_stage2_source(source_target)

        while count_jsonl(run_paths.genui_path) < cycle_target:
            before = count_jsonl(run_paths.genui_path)
            run_stage3(
                queries_path=run_paths.queries_path,
                responses_path=run_paths.responses_path,
                prompt_path=stage3_prompt_path,
                adapter=adapter,
                genui_path=run_paths.genui_path,
                schema_path=schema_path,
                artifacts_dir=run_paths.artifacts_dir,
                candidates_per_response=1,
                max_repair_attempts=int(run_cfg.get("max_repair_attempts", 1)),
                max_tokens=int(run_cfg.get("genui_max_tokens", 8192)),
                prompt_max_tokens=int(run_cfg.get("genui_prompt_max_tokens", 60000)),
                batch_size=1,
                seed=int(run_cfg.get("seed", 42)),
                rate_limiter=rate_limiter,
                cache=cache,
                logger=logger,
                max_total=cycle_target - before,
                max_attempts=int(run_cfg.get("max_attempts", 6)),
                aggregates_path=run_paths.aggregates_path,
                aggregate_weights=eval_cfg.get("weights", {}),
                metric_version=eval_cfg.get("metric_version", "dual"),
            )
            after = count_jsonl(run_paths.genui_path)
            logger.info("dataset_v1 progress stage3 genui=%s/%s created=%s", after, cycle_target, after - before)
            if after <= before:
                source_counts = {
                    "queries": count_jsonl(run_paths.queries_path),
                    "responses": count_jsonl(run_paths.responses_path),
                    "genui": after,
                }
                if source_counts["responses"] >= source_ceiling:
                    raise RuntimeError(
                        "Stage3 made no progress and replacement source ceiling was reached; "
                        f"target={cycle_target} counts={source_counts}"
                    )
                source_target = min(
                    source_ceiling,
                    source_counts["responses"] + max(1, int(args.replacement_batch_size)),
                )
                logger.warning(
                    "dataset_v1 Stage3 made no progress, likely due permanently filtered responses. "
                    "Adding replacement source records source_target=%s counts=%s",
                    source_target,
                    source_counts,
                )
                write_progress(
                    progress_path,
                    {
                        **settings,
                        "cycle": cycle,
                        "cycle_target": cycle_target,
                        "source_target": source_target,
                        "counts": source_counts,
                        "stage3_no_progress_at": datetime.utcnow().isoformat() + "Z",
                    },
                )
                _ensure_stage1_source(source_target)
                _ensure_stage2_source(source_target)


if __name__ == "__main__":
    main()
