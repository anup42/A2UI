from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.cache import PromptCache
from pipeline.metrics import aggregate_metrics, compute_overall_score
from pipeline.stage1_queries import run_stage1
from pipeline.stage2_responses import run_stage2
from pipeline.stage3_genui import run_stage3
from pipeline.stage4_render import run_stage4
from pipeline.storage import JsonlWriter, get_run_paths, iter_jsonl
from llm.base import ModelSpec
from llm.factory import build_adapter, load_model_specs
from utils.config import load_yaml
from utils.logging import setup_logger
from utils.rate_limit import RateLimiter


def _resolve_run_id(run_cfg: dict) -> str:
    run_id = run_cfg.get("run_id", "auto_timestamp")
    if run_id == "auto_timestamp":
        return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    return run_id


def _write_subset_queries(queries_path: Path, subset: list[dict]) -> None:
    if queries_path.exists():
        return
    writer = JsonlWriter(queries_path)
    for row in subset:
        writer.append(row)


def _load_subset(queries_path: Path, size: int) -> list[dict]:
    subset = []
    for row in iter_jsonl(queries_path):
        subset.append(row)
        if len(subset) >= size:
            break
    return subset


def _compute_aggregates(genui_path: Path, weights: dict) -> dict:
    rows = list(iter_jsonl(genui_path))
    aggregate = aggregate_metrics(rows)
    aggregate["overall_score"] = compute_overall_score(aggregate, weights)
    return aggregate


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _print_limits(specs) -> None:
    for spec in specs:
        limits = spec.limits or {}
        payload = {
            "name": spec.name,
            "provider": spec.provider,
            "model": spec.model,
            "limits": limits if limits else None,
        }
        print(json.dumps(payload, ensure_ascii=False))


def _load_env(root: Path) -> None:
    multiline_keys = {"GEMINI_API_KEYS", "GAUSS_OPENAPI_TOKEN", "GAUSS_CLIENT_KEY"}
    for env_path in [
        root / ".env",
        root / ".env.example",
        root.parent / ".env",
        root.parent / ".env.example",
    ]:
        if not env_path.exists():
            continue
        is_example = env_path.name.endswith(".example")
        pending_key = None
        pending_parts: list[str] = []
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "=" not in line:
                if pending_key in multiline_keys:
                    pending_parts.append(line.strip())
                continue
            if pending_key in multiline_keys and pending_parts:
                if not (is_example and pending_key in os.environ):
                    os.environ[pending_key] = ",".join(pending_parts)
            pending_key = None
            pending_parts = []
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key in multiline_keys:
                if not (is_example and key in os.environ):
                    pending_key = key
                    pending_parts = [value]
                continue
            if key and key not in os.environ:
                os.environ[key] = value
        if pending_key in multiline_keys and pending_parts:
            if not (is_example and pending_key in os.environ):
                os.environ[pending_key] = ",".join(pending_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="Run a single stage")
    parser.add_argument("--model", type=str, default=None, help="Model name from models.yaml")
    parser.add_argument("--benchmark_models", nargs="*", default=None, help="Benchmark models by name")
    parser.add_argument("--run_id", type=str, default=None, help="Override run id")
    parser.add_argument("--print_limits", action="store_true", help="Print configured model limits")
    parser.add_argument("--list_models", action="store_true", help="List models for a provider")
    parser.add_argument(
        "--genui_batch_size",
        type=int,
        default=None,
        help="Override GenUI batch size for stage 3 (default from run.yaml)",
    )
    args = parser.parse_args()

    root = ROOT
    _load_env(root)
    configs_dir = root / "configs"
    prompts_dir = root / "prompts"
    schema_dir = root / "schema"

    run_cfg = load_yaml(configs_dir / "run.yaml").get("run", {})
    eval_cfg = load_yaml(configs_dir / "run.yaml").get("evaluation", {})
    benchmark_cfg = load_yaml(configs_dir / "run.yaml").get("benchmark", {})
    genui_batch_size = run_cfg.get("genui_batch_size", 100)
    if args.genui_batch_size is not None:
        genui_batch_size = args.genui_batch_size
    genui_batch_size = int(genui_batch_size)

    output_dir = Path(run_cfg.get("output_dir", "data/runs"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    models_cfg = load_yaml(configs_dir / "models.yaml")

    run_id = args.run_id or _resolve_run_id(run_cfg)
    run_paths = get_run_paths(output_dir, run_id, run_cfg.get("artifact_dir", "artifacts"))
    logger = setup_logger(run_paths.run_dir)

    specs = load_model_specs(models_cfg)
    if not specs:
        raise SystemExit("No models configured in configs/models.yaml")

    model_map = {spec.name: spec for spec in specs}

    if args.print_limits:
        _print_limits(specs)
        return

    if args.list_models:
        spec = model_map.get(args.model) if args.model else None
        if spec is None:
            spec = next((item for item in specs if item.provider.lower() == "gauss"), None)
        if spec is None:
            spec = ModelSpec(
                name="gauss_list",
                provider="gauss",
                model="GAUSS_MODEL_ID",
                supports_json_mode=False,
            )
        adapter = build_adapter(spec)
        if not hasattr(adapter, "list_models"):
            raise SystemExit(f"Provider {spec.provider} does not support list_models")
        payload = adapter.list_models()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.benchmark_models is not None and len(args.benchmark_models) > 0:
        base_queries_path = run_paths.queries_path
        if not base_queries_path.exists():
            spec = model_map.get(args.model) if args.model else specs[0]
            adapter = build_adapter(spec)
            logger.info("Benchmark: generating base queries using %s", spec.name)
        rate_limiter = RateLimiter(
            float(run_cfg.get("rate_limit_qps", 2)),
            float(run_cfg.get("call_sleep_seconds", 0)),
        )
        run_stage1(
            intents_file=root / run_cfg.get("intents_file", "intents.info"),
            prompt_path=prompts_dir / "query_gen.md",
            adapter=adapter,
            run_dir=run_paths.run_dir,
            queries_path=base_queries_path,
            k_per_intent=int(run_cfg.get("k_queries_per_intent", 20)),
            batch_size=int(run_cfg.get("batch_size_queries", 10)),
            seed=int(run_cfg.get("seed", 42)),
            temperature=0.7,
            max_tokens=int(run_cfg.get("query_max_tokens", 512)),
            rate_limiter=rate_limiter,
            cache=PromptCache(root / run_cfg.get("cache_dir", "data/cache")),
            logger=logger,
            max_total=run_cfg.get("max_queries_total"),
            max_failures_per_intent=int(run_cfg.get("stage1_max_failures_per_intent", 3)),
            fill_missing_with_fallback=bool(run_cfg.get("stage1_fill_missing_with_fallback", True)),
            max_attempts=int(run_cfg.get("max_attempts", 3)),
        )

        subset_size = int(benchmark_cfg.get("fixed_subset_size", 50))
        subset = _load_subset(base_queries_path, subset_size)
        if not subset:
            raise SystemExit("No queries available for benchmarking")

        aggregates = {}
        for model_name in args.benchmark_models:
            spec = model_map.get(model_name)
            if not spec:
                logger.error("Unknown model for benchmark: %s", model_name)
                continue
            model_run_id = f"{run_id}_{model_name}"
            model_paths = get_run_paths(data_dir / "runs", model_run_id, run_cfg.get("artifact_dir", "artifacts"))
            model_logger = setup_logger(model_paths.run_dir)
            _write_subset_queries(model_paths.queries_path, subset)

            adapter = build_adapter(spec)
            rate_limiter = RateLimiter(
                float(run_cfg.get("rate_limit_qps", 2)),
                float(run_cfg.get("call_sleep_seconds", 0)),
            )
            run_stage2(
                queries_path=model_paths.queries_path,
                prompt_path=prompts_dir / "response_gen.md",
                batch_prompt_path=prompts_dir / "response_gen_batch.md",
                adapter=adapter,
                responses_path=model_paths.responses_path,
                n_per_query=int(run_cfg.get("n_responses_per_query", 2)),
                batch_size=int(run_cfg.get("response_batch_size", 1)),
                query_batch_size=int(run_cfg.get("query_batch_size", 1)),
                group_by_intent=bool(run_cfg.get("response_group_by_intent", False)),
                batch_fallback_per_query=bool(run_cfg.get("response_batch_fallback_per_query", True)),
                temperatures=_ensure_list(run_cfg.get("response_temperatures", [0.7])),
                max_tokens=int(run_cfg.get("response_max_tokens", 512)),
                seed=int(benchmark_cfg.get("fixed_seed", 123)),
                rate_limiter=rate_limiter,
                cache=PromptCache(root / run_cfg.get("cache_dir", "data/cache")),
                logger=model_logger,
                max_total=run_cfg.get("max_responses_total"),
                max_attempts=int(run_cfg.get("max_attempts", 3)),
            )
            run_stage3(
                responses_path=model_paths.responses_path,
                prompt_path=prompts_dir / "genui_gen.md",
                adapter=adapter,
                genui_path=model_paths.genui_path,
                schema_path=schema_dir / "genui.schema.json",
                artifacts_dir=model_paths.artifacts_dir,
                candidates_per_response=int(run_cfg.get("genui_candidates_per_response", 1)),
                max_repair_attempts=int(run_cfg.get("max_repair_attempts", 1)),
                max_tokens=int(run_cfg.get("genui_max_tokens", 1024)),
                batch_size=genui_batch_size,
                seed=int(benchmark_cfg.get("fixed_seed", 123)),
                rate_limiter=rate_limiter,
                cache=PromptCache(root / run_cfg.get("cache_dir", "data/cache")),
                logger=model_logger,
                max_total=run_cfg.get("max_genui_total"),
                max_attempts=int(run_cfg.get("max_attempts", 3)),
                aggregates_path=model_paths.aggregates_path,
                aggregate_weights=eval_cfg.get("weights", {}),
            )
            aggregates[model_name] = _compute_aggregates(model_paths.genui_path, eval_cfg.get("weights", {}))

        run_paths.aggregates_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
        logger.info("Benchmark complete. Aggregates stored at %s", run_paths.aggregates_path)
        return

    if args.model and args.model not in model_map:
        available = ", ".join(sorted(model_map.keys()))
        raise SystemExit(f"Unknown model '{args.model}'. Available: {available}")
    spec = model_map.get(args.model) if args.model else specs[0]
    adapter = build_adapter(spec)
    rate_limiter = RateLimiter(
        float(run_cfg.get("rate_limit_qps", 2)),
        float(run_cfg.get("call_sleep_seconds", 0)),
    )

    prompt_max_tokens = run_cfg.get("genui_prompt_max_tokens")
    if prompt_max_tokens is None and adapter.spec.provider == "gauss":
        prompt_max_tokens = 6000

    if args.stage == 1:
        run_stage1(
            intents_file=root / run_cfg.get("intents_file", "intents.info"),
            prompt_path=prompts_dir / "query_gen.md",
            adapter=adapter,
            run_dir=run_paths.run_dir,
            queries_path=run_paths.queries_path,
            k_per_intent=int(run_cfg.get("k_queries_per_intent", 20)),
            batch_size=int(run_cfg.get("batch_size_queries", 10)),
            seed=int(run_cfg.get("seed", 42)),
            temperature=0.7,
            max_tokens=int(run_cfg.get("query_max_tokens", 512)),
            rate_limiter=rate_limiter,
            cache=PromptCache(root / run_cfg.get("cache_dir", "data/cache")),
            logger=logger,
            max_total=run_cfg.get("max_queries_total"),
            max_failures_per_intent=int(run_cfg.get("stage1_max_failures_per_intent", 3)),
            fill_missing_with_fallback=bool(run_cfg.get("stage1_fill_missing_with_fallback", True)),
            max_attempts=int(run_cfg.get("max_attempts", 3)),
        )
        return

    if args.stage == 2:
        run_stage2(
            queries_path=run_paths.queries_path,
            prompt_path=prompts_dir / "response_gen.md",
            batch_prompt_path=prompts_dir / "response_gen_batch.md",
            adapter=adapter,
            responses_path=run_paths.responses_path,
            n_per_query=int(run_cfg.get("n_responses_per_query", 2)),
            batch_size=int(run_cfg.get("response_batch_size", 1)),
            query_batch_size=int(run_cfg.get("query_batch_size", 1)),
            group_by_intent=bool(run_cfg.get("response_group_by_intent", False)),
            batch_fallback_per_query=bool(run_cfg.get("response_batch_fallback_per_query", True)),
            temperatures=_ensure_list(run_cfg.get("response_temperatures", [0.7])),
            max_tokens=int(run_cfg.get("response_max_tokens", 512)),
            seed=int(run_cfg.get("seed", 42)),
            rate_limiter=rate_limiter,
            cache=PromptCache(root / run_cfg.get("cache_dir", "data/cache")),
            logger=logger,
            max_total=run_cfg.get("max_responses_total"),
            max_attempts=int(run_cfg.get("max_attempts", 3)),
        )
        return

    if args.stage == 3:
        run_stage3(
            responses_path=run_paths.responses_path,
            prompt_path=prompts_dir / "genui_gen.md",
            adapter=adapter,
            genui_path=run_paths.genui_path,
            schema_path=schema_dir / "genui.schema.json",
            artifacts_dir=run_paths.artifacts_dir,
            candidates_per_response=int(run_cfg.get("genui_candidates_per_response", 1)),
            max_repair_attempts=int(run_cfg.get("max_repair_attempts", 1)),
            max_tokens=int(run_cfg.get("genui_max_tokens", 1024)),
            prompt_max_tokens=int(prompt_max_tokens) if prompt_max_tokens else None,
            batch_size=genui_batch_size,
            seed=int(run_cfg.get("seed", 42)),
            rate_limiter=rate_limiter,
            cache=PromptCache(root / run_cfg.get("cache_dir", "data/cache")),
            logger=logger,
            max_total=run_cfg.get("max_genui_total"),
            max_attempts=int(run_cfg.get("max_attempts", 3)),
            aggregates_path=run_paths.aggregates_path,
            aggregate_weights=eval_cfg.get("weights", {}),
        )
        logger.info("Stage3 complete.")
        return

    if args.stage == 4:
        render_cfg = run_cfg.get("render", {})
        output_dir = run_paths.run_dir / render_cfg.get("output_dir", "rendered")
        assets_dir = root / render_cfg.get("assets_dir", "renderer/lit")
        viewport = render_cfg.get("viewport", {"width": 1280, "height": 720})
        run_stage4(
            genui_path=run_paths.genui_path,
            output_dir=output_dir,
            assets_dir=assets_dir,
            server_root=root,
            logger=logger,
            max_total=render_cfg.get("max_total"),
            render_images=bool(render_cfg.get("render_images", True)),
            image_format=str(render_cfg.get("image_format", "png")),
            viewport=viewport if isinstance(viewport, dict) else {"width": 1280, "height": 720},
            timeout_ms=int(render_cfg.get("timeout_ms", 15000)),
            wait_ms=int(render_cfg.get("wait_ms", 200)),
            use_http_server=bool(render_cfg.get("use_http_server", True)),
        )
        logger.info("Stage4 complete. Rendered outputs stored at %s", output_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
