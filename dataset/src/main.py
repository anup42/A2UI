from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.cache import PromptCache
from pipeline.metrics import aggregate_metrics, compute_overall_score, compute_media_score
from pipeline.stage1_queries import run_stage1
from pipeline.stage2_responses import run_stage2
from pipeline.stage3_genui import run_stage3
from pipeline.stage4_render import run_stage4
from pipeline.stage5_direct_html import run_stage5
from pipeline.storage import JsonlWriter, get_run_paths, iter_jsonl
from llm.base import ModelSpec
from llm.factory import build_adapter, load_model_specs
from utils.config import load_yaml
from utils.logging import setup_logger
from utils.rate_limit import RateLimiter
from utils.versioning import (
    build_run_manifest,
    format_version_report,
    write_run_manifest,
)


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
    render_rows_by_ui_id = _load_render_rows_by_ui_id(genui_path.parent)
    aggregate = aggregate_metrics(rows, render_rows_by_ui_id=render_rows_by_ui_id)
    aggregate["overall_score"] = compute_overall_score(aggregate, weights)
    aggregate["media_score"] = compute_media_score(aggregate)
    return aggregate


def _load_response_text_map(responses_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not responses_path.exists():
        return mapping
    for row in iter_jsonl(responses_path):
        response_id = row.get("response_id")
        response_text = row.get("response_text")
        if isinstance(response_id, str) and isinstance(response_text, str):
            mapping[response_id] = response_text
    return mapping


def _load_render_rows_by_ui_id(run_dir: Path) -> dict[str, dict]:
    render_path = run_dir / "render.jsonl"
    rows: dict[str, dict] = {}
    if not render_path.exists():
        return rows
    for row in iter_jsonl(render_path):
        ui_id = row.get("ui_id")
        if isinstance(ui_id, str) and ui_id:
            rows[ui_id] = row
    return rows


def _compute_aggregates_with_backfill(
    genui_path: Path,
    responses_path: Path,
    weights: dict,
) -> dict:
    rows = list(iter_jsonl(genui_path))
    if rows:
        response_map = _load_response_text_map(responses_path)
        if response_map:
            for row in rows:
                if row.get("response_text"):
                    continue
                response_id = row.get("response_id")
                if isinstance(response_id, str):
                    backfill = response_map.get(response_id)
                    if isinstance(backfill, str):
                        row["response_text"] = backfill
    render_rows_by_ui_id = _load_render_rows_by_ui_id(genui_path.parent)
    aggregate = aggregate_metrics(rows, render_rows_by_ui_id=render_rows_by_ui_id)
    aggregate["overall_score"] = compute_overall_score(aggregate, weights)
    aggregate["media_score"] = compute_media_score(aggregate)
    return aggregate


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def _expected_stage3_ui_ids(responses_path: Path, candidates_per_response: int) -> set[str]:
    expected: set[str] = set()
    if not responses_path.exists():
        return expected
    candidate_count = max(1, int(candidates_per_response))
    for row in iter_jsonl(responses_path):
        query_id = str(row.get("query_id") or "").strip()
        if not query_id:
            continue
        n_idx_raw = row.get("n_idx", 1)
        try:
            n_idx = int(n_idx_raw)
        except Exception:
            n_idx = 1
        suffix = query_id.replace("q_", "")
        for c_idx in range(1, candidate_count + 1):
            if c_idx == 1:
                expected.add(f"u_{suffix}_{n_idx:02d}")
            else:
                expected.add(f"u_{suffix}_{n_idx:02d}_{c_idx:02d}")
    return expected


def _is_stage3_complete(responses_path: Path, genui_path: Path, candidates_per_response: int) -> bool:
    expected = _expected_stage3_ui_ids(responses_path, candidates_per_response)
    if not expected:
        return False
    if not genui_path.exists():
        return False
    existing = {str(row.get("ui_id") or "").strip() for row in iter_jsonl(genui_path)}
    existing.discard("")
    return expected.issubset(existing)


def _extract_stage3_prompt_version(genui_path: Path) -> str | None:
    if not genui_path.exists():
        return None
    for row in iter_jsonl(genui_path):
        gen_info = row.get("gen")
        if not isinstance(gen_info, dict):
            continue
        prompt_version = gen_info.get("prompt_version")
        if isinstance(prompt_version, str) and prompt_version.strip():
            return prompt_version.strip()
    return None


def _extract_prompt_heading_version(prompt_path: Path) -> str | None:
    if not prompt_path.exists():
        return None
    try:
        first_line = prompt_path.read_text(encoding="utf-8").splitlines()[0].lstrip("\ufeff").strip()
    except Exception:
        return None
    match = re.match(r"^#\s*([A-Za-z0-9_.-]+)", first_line)
    if match:
        return match.group(1)
    return None


def _resolve_path(root: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    root_joined = root / path
    if root_joined.exists():
        return root_joined
    return path


def _load_ir_prompt_versions(root: Path, versions_file: Path) -> tuple[str | None, dict[str, dict]]:
    if not versions_file.is_absolute():
        versions_file = root / versions_file
    if not versions_file.exists():
        return None, {}
    data = load_yaml(versions_file)
    section = data.get("ir_prompt_versions") if isinstance(data, dict) else None
    if not isinstance(section, dict):
        return None, {}
    default_version = section.get("default")
    default_value = str(default_version).strip() if default_version is not None else None
    steps_raw = section.get("steps")
    if not isinstance(steps_raw, list):
        return default_value, {}
    by_id: dict[str, dict] = {}
    for item in steps_raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or "").strip()
        if not key:
            continue
        by_id[key] = item
    return default_value, by_id


def _resolve_stage3_prompt_path(
    *,
    root: Path,
    prompts_dir: Path,
    run_cfg: dict,
    args,
    logger,
) -> tuple[Path, str | None]:
    if args.stage3_prompt_path:
        prompt_path = _resolve_path(root, args.stage3_prompt_path)
        if not prompt_path.exists():
            raise SystemExit(f"Stage3 prompt path not found: {prompt_path}")
        return prompt_path, None

    versions_file_raw = (
        args.stage3_prompt_versions_file
        or run_cfg.get("ir_prompt_versions_file")
        or "versions/ir_prompt_versions.yaml"
    )
    versions_file = _resolve_path(root, str(versions_file_raw))
    default_version, versions_by_id = _load_ir_prompt_versions(root, versions_file)
    selected_version = (
        args.stage3_prompt_version
        or run_cfg.get("default_ir_prompt_version")
        or default_version
    )
    if selected_version:
        version_item = versions_by_id.get(str(selected_version))
        if version_item is None:
            available = ", ".join(sorted(versions_by_id.keys()))
            raise SystemExit(
                f"Unknown --stage3_prompt_version '{selected_version}'. Available: {available}"
            )
        prompt_raw = str(version_item.get("prompt_path") or "").strip()
        if not prompt_raw:
            raise SystemExit(f"Prompt version '{selected_version}' is missing prompt_path")
        prompt_path = _resolve_path(root, prompt_raw)
        if not prompt_path.exists():
            raise SystemExit(f"Prompt file for version '{selected_version}' not found: {prompt_path}")
        return prompt_path, str(selected_version)

    prompt_path = prompts_dir / "genui_gen.md"
    if not prompt_path.exists():
        raise SystemExit(f"Stage3 prompt file not found: {prompt_path}")
    return prompt_path, None


def _coerce_viewport(raw: dict | None, fallback: dict[str, int]) -> dict[str, int]:
    if not isinstance(raw, dict):
        return dict(fallback)
    width = raw.get("width")
    height = raw.get("height")
    try:
        width_int = int(width)
        height_int = int(height)
    except Exception:
        return dict(fallback)
    if width_int <= 0 or height_int <= 0:
        return dict(fallback)
    return {"width": width_int, "height": height_int}


def _resolve_render_viewport_and_mode(
    *,
    render_cfg: dict,
    preset_override: str | None,
    logger,
) -> tuple[dict[str, int], bool, str]:
    default_viewport = {"width": 1280, "height": 720}
    default_preset = render_cfg.get("default_viewport_preset")
    preset_name = preset_override or (str(default_preset).strip() if default_preset else None)
    presets = render_cfg.get("viewport_presets")
    if preset_name and isinstance(presets, dict):
        preset_cfg = presets.get(preset_name)
        if isinstance(preset_cfg, dict):
            viewport = _coerce_viewport(preset_cfg, default_viewport)
            emulate_mobile = bool(preset_cfg.get("emulate_mobile", False))
            return viewport, emulate_mobile, str(preset_name)
        logger.warning("Unknown render viewport preset '%s'; falling back to render.viewport", preset_name)
    viewport = _coerce_viewport(render_cfg.get("viewport"), default_viewport)
    emulate_mobile = bool(render_cfg.get("emulate_mobile", False))
    selected_name = str(preset_name) if preset_name else "custom"
    return viewport, emulate_mobile, selected_name


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


def _endpoint_ready(endpoint: str, timeout_s: float = 2.0) -> bool:
    check_url = endpoint
    if check_url.endswith("/v1/chat/completions"):
        check_url = check_url.replace("/v1/chat/completions", "/v1/models")
    try:
        with urllib.request.urlopen(check_url, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_for_endpoint(endpoint: str, timeout_s: float = 120.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if _endpoint_ready(endpoint):
            return True
        time.sleep(1.0)
    return False


def _configure_local_model_path(spec: ModelSpec, args, logger) -> None:
    if spec.provider.lower() != "local":
        return

    model_lower = (spec.model or "").lower()
    if "qwen" in model_lower or "deepseek" in model_lower:
        # Default to strict offline for local Qwen/DeepSeek runs.
        os.environ.setdefault("LOCAL_STRICT_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("DATASET_OFFLINE_MODE", "1")

    override = args.local_model_path or args.vllm_model_path
    if not override:
        return

    if "qwen" in model_lower:
        os.environ["QWEN_MODEL_PATH"] = override
        logger.info("Using local model path override for Qwen: %s", override)
        return
    if "deepseek" in model_lower:
        os.environ["DEEPSEEK_MODEL_PATH"] = override
        logger.info("Using local model path override for DeepSeek: %s", override)
        return
    os.environ["LOCAL_MODEL_PATH"] = override
    logger.info("Using local model path override: %s", override)


def _maybe_start_vllm(spec: ModelSpec, args, logger):
    if not args.start_vllm:
        return None
    if args.stage not in (1, 2, 3):
        return None
    if spec.provider.lower() != "local":
        return None
    endpoint = (spec.endpoint or "").strip()
    if not endpoint.lower().startswith(("http://", "https://")):
        logger.info("Skipping vLLM auto-start for %s (local transformers mode)", spec.name)
        return None
    model_lower = spec.model.lower()
    if "qwen3-coder" not in model_lower and "deepseek-coder" not in model_lower:
        return None

    endpoint = spec.endpoint or "http://localhost:8000/v1/chat/completions"
    if _endpoint_ready(endpoint):
        logger.info("vLLM already running at %s", endpoint)
        return None

    if "deepseek-coder" in model_lower:
        model_path = (
            args.vllm_model_path
            or os.environ.get("DEEPSEEK_MODEL_PATH")
            or os.environ.get("QWEN_MODEL_PATH")
        )
    else:
        model_path = args.vllm_model_path or os.environ.get("QWEN_MODEL_PATH")
    if not model_path:
        raise SystemExit(
            "Missing local model path. Use --vllm_model_path or set QWEN_MODEL_PATH / DEEPSEEK_MODEL_PATH."
        )

    script = ROOT / "scripts" / "serve_qwen_vllm.py"
    if not script.exists():
        script = ROOT / "scripts" / "serve_qwen_vllm.py"
    cmd = [
        sys.executable,
        str(script),
        "--model-path",
        model_path,
        "--served-model-name",
        spec.model,
        "--gpus",
        str(args.vllm_gpus),
        "--host",
        args.vllm_host,
        "--port",
        str(args.vllm_port),
        "--dtype",
        args.vllm_dtype,
        "--gpu-memory-utilization",
        str(args.vllm_gpu_mem_util),
    ]
    if args.vllm_max_model_len:
        cmd += ["--max-model-len", str(args.vllm_max_model_len)]
    if args.vllm_swap_space:
        cmd += ["--swap-space", str(args.vllm_swap_space)]
    if args.vllm_trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.vllm_cuda_visible_devices:
        cmd += ["--cuda-visible-devices", args.vllm_cuda_visible_devices]

    logger.info("Starting vLLM server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    if not _wait_for_endpoint(endpoint, timeout_s=args.vllm_start_timeout):
        proc.terminate()
        raise RuntimeError(f"vLLM failed to start within {args.vllm_start_timeout}s")
    logger.info("vLLM ready at %s", endpoint)
    return proc


def _stop_vllm(proc, logger) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    logger.info("vLLM server stopped")



def _write_run_manifest_safe(
    *,
    root: Path,
    run_id: str,
    stage: int | str,
    spec: ModelSpec,
    run_paths,
    run_cfg_path: Path,
    models_cfg_path: Path,
    logger,
) -> None:
    try:
        manifest = build_run_manifest(
            root=root,
            run_id=run_id,
            stage=stage,
            model_spec=spec,
            run_paths=run_paths,
            run_cfg_path=run_cfg_path,
            models_cfg_path=models_cfg_path,
            argv=sys.argv,
        )
        write_run_manifest(run_paths.manifest_path, manifest)
    except Exception as exc:
        logger.warning("Failed to write run manifest: %s", exc)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], help="Run a single stage")
    parser.add_argument("--model", type=str, default=None, help="Model name from models.yaml")
    parser.add_argument("--benchmark_models", nargs="*", default=None, help="Benchmark models by name")
    parser.add_argument("--run_id", type=str, default=None, help="Override run id")
    parser.add_argument("--version", action="store_true", help="Print release/component versions and exit")
    parser.add_argument("--print_limits", action="store_true", help="Print configured model limits")
    parser.add_argument("--list_models", action="store_true", help="List models for a provider")
    parser.add_argument(
        "--genui_batch_size",
        type=int,
        default=None,
        help="Override GenUI batch size for stage 3 (default from run.yaml)",
    )
    parser.add_argument(
        "--render_workers",
        type=int,
        default=None,
        help="Override Stage4 render worker count (default from run.yaml render.parallel_workers or 1).",
    )
    parser.add_argument(
        "--render_viewport_preset",
        type=str,
        default=None,
        help="Viewport preset from run.yaml render.viewport_presets (for example: mobile or desktop).",
    )
    parser.add_argument(
        "--stage3_prompt_path",
        type=str,
        default=None,
        help="Override Stage3 prompt file path (absolute or relative to dataset root).",
    )
    parser.add_argument(
        "--stage3_prompt_version",
        type=str,
        default=None,
        help="Stage3 prompt version id from versions/ir_prompt_versions.yaml.",
    )
    parser.add_argument(
        "--stage3_prompt_versions_file",
        type=str,
        default=None,
        help="YAML file containing ir_prompt_versions definitions.",
    )
    parser.add_argument(
        "--rate_limit_qps",
        type=float,
        default=None,
        help="Override global request rate limit (queries per second) without editing run.yaml",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local model folder path override for provider=local transformers mode.",
    )
    parser.add_argument("--start_vllm", action="store_true", help="Auto-start local vLLM server")
    parser.add_argument("--vllm_model_path", type=str, default=None, help="Local Qwen model folder path")
    parser.add_argument("--vllm_gpus", type=int, default=4, help="Tensor-parallel GPU count for vLLM")
    parser.add_argument("--vllm_host", type=str, default="0.0.0.0", help="vLLM host")
    parser.add_argument("--vllm_port", type=int, default=8000, help="vLLM port")
    parser.add_argument("--vllm_dtype", type=str, default="float16", help="vLLM dtype")
    parser.add_argument(
        "--vllm_gpu_mem_util",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization (0-1)",
    )
    parser.add_argument("--vllm_max_model_len", type=int, default=None, help="vLLM max model length")
    parser.add_argument("--vllm_swap_space", type=int, default=None, help="vLLM swap space (GB)")
    parser.add_argument("--vllm_trust_remote_code", action="store_true", help="vLLM trust remote code")
    parser.add_argument(
        "--vllm_cuda_visible_devices",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids for vLLM",
    )
    parser.add_argument(
        "--vllm_start_timeout",
        type=int,
        default=120,
        help="Timeout (seconds) to wait for vLLM to be ready",
    )
    args = parser.parse_args()

    root = ROOT
    _load_env(root)
    if args.version:
        print(format_version_report(root))
        return
    configs_dir = root / "configs"
    prompts_dir = root / "prompts"
    schema_dir = root / "schema"
    run_cfg_path = configs_dir / "run.yaml"
    models_cfg_path = configs_dir / "models.yaml"

    all_run_cfg = load_yaml(run_cfg_path)
    run_cfg = all_run_cfg.get("run", {})
    render_cfg_global = all_run_cfg.get("render", {})
    stage5_cfg_global = all_run_cfg.get("stage5", {})
    eval_cfg = all_run_cfg.get("evaluation", {})
    benchmark_cfg = all_run_cfg.get("benchmark", {})
    effective_rate_limit_qps = float(run_cfg.get("rate_limit_qps", 2))
    if args.rate_limit_qps is not None:
        effective_rate_limit_qps = float(args.rate_limit_qps)

    genui_batch_size = run_cfg.get("genui_batch_size", 100)
    if args.genui_batch_size is not None:
        genui_batch_size = args.genui_batch_size
    genui_batch_size = int(genui_batch_size)

    output_dir = Path(run_cfg.get("output_dir", "data/runs"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    models_cfg = load_yaml(models_cfg_path)

    run_id = args.run_id or _resolve_run_id(run_cfg)
    run_paths = get_run_paths(output_dir, run_id, run_cfg.get("artifact_dir", "artifacts"))
    logger = setup_logger(run_paths.run_dir)
    stage3_prompt_path, stage3_prompt_version_id = _resolve_stage3_prompt_path(
        root=root,
        prompts_dir=prompts_dir,
        run_cfg=run_cfg,
        args=args,
        logger=logger,
    )
    logger.info(
        "Stage3 prompt selected: path=%s version_id=%s",
        stage3_prompt_path,
        stage3_prompt_version_id or "default",
    )

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
            _configure_local_model_path(spec, args, logger)
            adapter = build_adapter(spec)
            logger.info("Benchmark: generating base queries using %s", spec.name)
            rate_limiter = RateLimiter(
                effective_rate_limit_qps,
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
                intent_batch_size=int(run_cfg.get("stage1_intent_batch_size", 1)),
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
            _configure_local_model_path(spec, args, logger)
            model_run_id = f"{run_id}_{model_name}"
            model_paths = get_run_paths(output_dir, model_run_id, run_cfg.get("artifact_dir", "artifacts"))
            model_logger = setup_logger(model_paths.run_dir)
            _write_subset_queries(model_paths.queries_path, subset)
            _write_run_manifest_safe(
                root=root,
                run_id=model_run_id,
                stage="benchmark",
                spec=spec,
                run_paths=model_paths,
                run_cfg_path=run_cfg_path,
                models_cfg_path=models_cfg_path,
                logger=model_logger,
            )

            adapter = build_adapter(spec)
            rate_limiter = RateLimiter(
                effective_rate_limit_qps,
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
                queries_path=model_paths.queries_path,
                responses_path=model_paths.responses_path,
                prompt_path=stage3_prompt_path,
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
            aggregates[model_name] = _compute_aggregates_with_backfill(
                model_paths.genui_path,
                model_paths.responses_path,
                eval_cfg.get("weights", {}),
            )

        run_paths.aggregates_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
        logger.info("Benchmark complete. Aggregates stored at %s", run_paths.aggregates_path)
        return

    if args.model and args.model not in model_map:
        available = ", ".join(sorted(model_map.keys()))
        raise SystemExit(f"Unknown model '{args.model}'. Available: {available}")
    spec = model_map.get(args.model) if args.model else specs[0]
    _write_run_manifest_safe(
        root=root,
        run_id=run_id,
        stage=args.stage or "unknown",
        spec=spec,
        run_paths=run_paths,
        run_cfg_path=run_cfg_path,
        models_cfg_path=models_cfg_path,
        logger=logger,
    )

    # Recompute-only fast path:
    # If stage 3 outputs already exist for all expected response IDs/candidates,
    # skip adapter initialization/API calls and refresh aggregates.json directly.
    if args.stage == 3:
        stage3_candidates = int(run_cfg.get("genui_candidates_per_response", 1))
        selected_prompt_version = _extract_prompt_heading_version(stage3_prompt_path)
        existing_prompt_version = _extract_stage3_prompt_version(run_paths.genui_path)
        prompt_version_matches = (
            selected_prompt_version is None
            or existing_prompt_version is None
            or existing_prompt_version == selected_prompt_version
        )
        if not prompt_version_matches:
            logger.info(
                "Stage3 output prompt_version mismatch (existing=%s, selected=%s). Regenerating Stage3.",
                existing_prompt_version,
                selected_prompt_version,
            )
        if _is_stage3_complete(
            run_paths.responses_path,
            run_paths.genui_path,
            stage3_candidates,
        ) and prompt_version_matches:
            if not run_paths.genui_path.exists():
                raise SystemExit(f"Missing genui file: {run_paths.genui_path}")
            aggregates = _compute_aggregates_with_backfill(
                run_paths.genui_path,
                run_paths.responses_path,
                eval_cfg.get("weights", {}),
            )
            run_paths.aggregates_path.write_text(
                json.dumps(aggregates, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "Stage3 already complete for run_id=%s. Recomputed aggregates only (%s rows).",
                run_id,
                _count_jsonl_rows(run_paths.genui_path),
            )
            return

    adapter = None
    rate_limiter = None
    if args.stage in (1, 2, 3):
        _configure_local_model_path(spec, args, logger)
        adapter = build_adapter(spec)
        rate_limiter = RateLimiter(
            effective_rate_limit_qps,
            float(run_cfg.get("call_sleep_seconds", 0)),
        )

    prompt_max_tokens = run_cfg.get("genui_prompt_max_tokens")
    if prompt_max_tokens is None and adapter is not None and adapter.spec.provider == "gauss":
        prompt_max_tokens = 6000
    if args.stage == 3 and adapter is not None and adapter.spec.provider == "local":
        local_prompt_cap_raw = (os.environ.get("LOCAL_STAGE3_PROMPT_MAX_TOKENS") or "8192").strip()
        try:
            local_prompt_cap = max(500, int(local_prompt_cap_raw))
        except Exception:
            local_prompt_cap = 8192
        if prompt_max_tokens is None or int(prompt_max_tokens) > local_prompt_cap:
            prompt_max_tokens = local_prompt_cap
            logger.info(
                "Stage3 local prompt token cap applied: %s (LOCAL_STAGE3_PROMPT_MAX_TOKENS)",
                prompt_max_tokens,
            )

    server_proc = _maybe_start_vllm(spec, args, logger) if args.stage in (1, 2, 3) else None
    try:
        if args.stage == 1:
            run_stage1(
                intents_file=root / run_cfg.get("intents_file", "intents.info"),
                prompt_path=prompts_dir / "query_gen.md",
                adapter=adapter,
                run_dir=run_paths.run_dir,
                queries_path=run_paths.queries_path,
                k_per_intent=int(run_cfg.get("k_queries_per_intent", 20)),
                batch_size=int(run_cfg.get("batch_size_queries", 10)),
                intent_batch_size=int(run_cfg.get("stage1_intent_batch_size", 1)),
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
                queries_path=run_paths.queries_path,
                responses_path=run_paths.responses_path,
                prompt_path=stage3_prompt_path,
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
            render_cfg = render_cfg_global
            if not isinstance(render_cfg, dict):
                render_cfg = {}
            output_dir = run_paths.run_dir / render_cfg.get("output_dir", "rendered")
            assets_dir = root / render_cfg.get("assets_dir", "renderer/lit")
            viewport, emulate_mobile, viewport_name = _resolve_render_viewport_and_mode(
                render_cfg=render_cfg if isinstance(render_cfg, dict) else {},
                preset_override=args.render_viewport_preset,
                logger=logger,
            )
            logger.info(
                "Stage4 viewport preset=%s viewport=%s emulate_mobile=%s",
                viewport_name,
                viewport,
                emulate_mobile,
            )
            run_stage4(
                genui_path=run_paths.genui_path,
                output_dir=output_dir,
                assets_dir=assets_dir,
                server_root=root,
                logger=logger,
                max_total=render_cfg.get("max_total"),
                render_images=bool(render_cfg.get("render_images", True)),
                image_format=str(render_cfg.get("image_format", "png")),
                viewport=viewport,
                timeout_ms=int(render_cfg.get("timeout_ms", 15000)),
                wait_ms=int(render_cfg.get("wait_ms", 200)),
                use_http_server=bool(render_cfg.get("use_http_server", True)),
                parallel_workers=int(args.render_workers) if args.render_workers is not None else int(render_cfg.get("parallel_workers", 1)),
                emulate_mobile=emulate_mobile,
            )
            logger.info("Stage4 complete. Rendered outputs stored at %s", output_dir)
            return

        if args.stage == 5:
            render_cfg = render_cfg_global
            if not isinstance(render_cfg, dict):
                render_cfg = {}
            stage5_cfg = stage5_cfg_global if isinstance(stage5_cfg_global, dict) else {}
            output_dir = run_paths.run_dir / stage5_cfg.get("output_dir", "stage5_rendered")
            stage5_render_cfg: dict = dict(render_cfg) if isinstance(render_cfg, dict) else {}
            if isinstance(stage5_cfg, dict):
                stage5_render_cfg.update(stage5_cfg)
            viewport, emulate_mobile, viewport_name = _resolve_render_viewport_and_mode(
                render_cfg=stage5_render_cfg,
                preset_override=args.render_viewport_preset,
                logger=logger,
            )
            logger.info(
                "Stage5 viewport preset=%s viewport=%s emulate_mobile=%s",
                viewport_name,
                viewport,
                emulate_mobile,
            )
            run_stage5(
                responses_path=run_paths.responses_path,
                output_dir=output_dir,
                run_dir=run_paths.run_dir,
                server_root=root,
                logger=logger,
                render_images=bool(stage5_cfg.get("render_images", render_cfg.get("render_images", True))),
                image_format=str(stage5_cfg.get("image_format", render_cfg.get("image_format", "png"))),
                viewport=viewport,
                timeout_ms=int(stage5_cfg.get("timeout_ms", render_cfg.get("timeout_ms", 15000))),
                wait_ms=int(stage5_cfg.get("wait_ms", render_cfg.get("wait_ms", 200))),
                use_http_server=bool(stage5_cfg.get("use_http_server", render_cfg.get("use_http_server", True))),
                emulate_mobile=bool(stage5_cfg.get("emulate_mobile", emulate_mobile)),
            )
            logger.info("Stage5 complete. Direct HTML outputs stored at %s", output_dir)
            return
    finally:
        _stop_vllm(server_proc, logger)

    parser.print_help()


if __name__ == "__main__":
    main()
