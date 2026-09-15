from __future__ import annotations

import argparse
import json
import os
import shutil
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
from pipeline.genui_quality import normalize_metric_mode
from pipeline.metrics import aggregate_metrics, compute_overall_score, compute_media_score
from pipeline.stage1_queries import run_stage1
from pipeline.stage2_responses import run_stage2
from pipeline.stage3_genui import run_stage3
from pipeline.stage4_render import run_stage4
from pipeline.stage5_direct_html import run_stage5
from pipeline.storage import JsonlWriter, get_run_paths, iter_jsonl
from llm.base import ModelSpec
from llm.factory import build_adapter, load_model_specs
from llm.http_transport import urlopen
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


def _build_prompt_cache(root: Path, run_cfg: dict) -> PromptCache:
    enabled = bool(run_cfg.get("enable_prompt_cache", False))
    return PromptCache(root / run_cfg.get("cache_dir", "data/cache"), enabled=enabled)


def _compute_aggregates(genui_path: Path, weights: dict, metric_version: str = "v5_4") -> dict:
    metric_mode = normalize_metric_mode(metric_version)
    rows = list(iter_jsonl(genui_path))
    render_rows_by_ui_id = _load_render_rows_by_ui_id(genui_path.parent)
    aggregate = aggregate_metrics(
        rows,
        render_rows_by_ui_id=render_rows_by_ui_id,
        metric_version=metric_mode,
    )
    if metric_mode in {"legacy", "dual"}:
        legacy_score = compute_overall_score(aggregate, weights)
        aggregate["legacy_structural_richness_score"] = legacy_score
        aggregate["overall_score"] = legacy_score
        aggregate["legacy_score_deprecation_date"] = "2026-10-01"
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
    rows: dict[str, dict] = {}
    # Native checks are loaded last so a real device result supersedes generic
    # browser/screenshot diagnostics for the renderer-smoke atomic metric.
    for render_path in (
        run_dir / "render.jsonl",
        run_dir / "native_render_checks.jsonl",
    ):
        if not render_path.exists():
            continue
        for row in iter_jsonl(render_path):
            ui_id = row.get("ui_id")
            if isinstance(ui_id, str) and ui_id:
                rows[ui_id] = row
    return rows


def _compute_aggregates_with_backfill(
    genui_path: Path,
    responses_path: Path,
    weights: dict,
    metric_version: str = "v5_4",
) -> dict:
    metric_mode = normalize_metric_mode(metric_version)
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
    aggregate = aggregate_metrics(
        rows,
        render_rows_by_ui_id=render_rows_by_ui_id,
        metric_version=metric_mode,
    )
    if metric_mode in {"legacy", "dual"}:
        legacy_score = compute_overall_score(aggregate, weights)
        aggregate["legacy_structural_richness_score"] = legacy_score
        aggregate["overall_score"] = legacy_score
        aggregate["legacy_score_deprecation_date"] = "2026-10-01"
    aggregate["media_score"] = compute_media_score(aggregate)
    return aggregate


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def _stage3_id_suffixes(ir_formats: object) -> tuple[str, ...]:
    """Return the active Stage 3 id suffixes.

    A2UI Express is the single production format, so generated rows use the
    canonical id without a format suffix. Historical suffixed ids remain
    readable in old run folders but are never scheduled by the active path.
    """
    if ir_formats is None:
        return ("",)
    if isinstance(ir_formats, str):
        values = [part.strip().lower() for part in ir_formats.split(",") if part.strip()]
    elif isinstance(ir_formats, (list, tuple)):
        values = [str(part).strip().lower() for part in ir_formats if str(part).strip()]
    else:
        values = [str(ir_formats).strip().lower()]
    allowed = {"express", "a2ui_express", "a2ui_express_v1"}
    unsupported = [value for value in values if value not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported active Stage 3 format(s): {unsupported}")
    return ("",)


def _expected_stage3_ui_ids(
    responses_path: Path,
    candidates_per_response: int,
    ir_formats: object = None,
) -> set[str]:
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
            base_id = (
                f"u_{suffix}_{n_idx:02d}"
                if c_idx == 1
                else f"u_{suffix}_{n_idx:02d}_{c_idx:02d}"
            )
            for format_suffix in _stage3_id_suffixes(ir_formats):
                expected.add(base_id + format_suffix)
    return expected


def _is_stage3_complete(
    responses_path: Path,
    genui_path: Path,
    candidates_per_response: int,
    ir_formats: object = None,
) -> bool:
    expected = _expected_stage3_ui_ids(responses_path, candidates_per_response, ir_formats)
    if not expected:
        return False
    if not genui_path.exists():
        return False
    existing = {str(row.get("ui_id") or "").strip() for row in iter_jsonl(genui_path)}
    existing.discard("")
    return expected.issubset(existing)


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        raise SystemExit(f"{name} must be a float, got: {raw!r}")


def _env_float_list(name: str, fallback) -> list[float]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return [float(item) for item in _ensure_list(fallback)]
    values: list[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except Exception:
            raise SystemExit(f"{name} must be a comma-separated float list, got: {raw!r}")
    return values or [float(item) for item in _ensure_list(fallback)]


def _resolve_cfg_path(root: Path, configured: str | None, fallback: str) -> Path:
    raw = (configured or fallback).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path


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
    multiline_keys = {
        "GEMINI_API_KEYS",
        "GEMINI_VERTEX_EXPRESS_API_KEYS",
        "VERTEX_EXPRESS_API_KEYS",
        "GAUSS_OPENAPI_TOKEN",
        "GAUSS_CLIENT_KEY",
    }
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


def _apply_env_int_override(run_cfg: dict, key: str, env_key: str) -> None:
    raw = (os.environ.get(env_key) or "").strip()
    if not raw:
        return
    try:
        run_cfg[key] = int(raw)
    except Exception:
        raise SystemExit(f"{env_key} must be an integer, got: {raw!r}")


def _apply_generation_env_overrides(run_cfg: dict) -> None:
    for key, env_key, minimum in (
        ("stage1_intent_batch_size", "A2UI_STAGE1_INTENT_BATCH_SIZE", 1),
        ("max_repair_attempts", "A2UI_MAX_REPAIR_ATTEMPTS", 0),
        ("max_attempts", "A2UI_MAX_ATTEMPTS", 1),
    ):
        _apply_env_int_override(run_cfg, key, env_key)
        if key in run_cfg:
            run_cfg[key] = max(minimum, int(run_cfg[key]))
    raw = (os.environ.get("A2UI_CALL_SLEEP_SECONDS") or "").strip()
    if raw:
        import math
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit(f"A2UI_CALL_SLEEP_SECONDS must be a finite float, got: {raw!r}")
        if not math.isfinite(value):
            raise SystemExit("A2UI_CALL_SLEEP_SECONDS must be finite")
        run_cfg["call_sleep_seconds"] = max(0.0, value)


def _apply_env_bool_override(run_cfg: dict, key: str, env_key: str) -> None:
    raw = (os.environ.get(env_key) or "").strip().lower()
    if not raw:
        return
    if raw in {"1", "true", "yes", "y", "on"}:
        run_cfg[key] = True
        return
    if raw in {"0", "false", "no", "n", "off"}:
        run_cfg[key] = False
        return
    raise SystemExit(f"{env_key} must be boolean, got: {raw!r}")


def _endpoint_ready(endpoint: str, timeout_s: float = 2.0) -> bool:
    check_url = endpoint
    if check_url.endswith("/v1/chat/completions"):
        check_url = check_url.replace("/v1/chat/completions", "/v1/models")
    try:
        with urlopen(check_url, timeout=timeout_s) as resp:
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


def _vllm_serve_supports_flag(flag: str) -> bool:
    try:
        result = subprocess.run(
            ["vllm", "serve", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except Exception:
        return False
    return flag in ((result.stdout or "") + (result.stderr or ""))


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
    if "qwen" not in model_lower and "deepseek" not in model_lower:
        return None

    endpoint = spec.endpoint or "http://localhost:8000/v1/chat/completions"
    os.environ.setdefault("LOCAL_ALLOW_HTTP_ENDPOINT", "1")
    os.environ["LOCAL_STRICT_OFFLINE"] = "0"
    if "qwen" in model_lower:
        os.environ.setdefault("LOCAL_VLLM_ENABLE_THINKING", "1")
    if _endpoint_ready(endpoint):
        logger.info("vLLM already running at %s", endpoint)
        return None

    if "deepseek" in model_lower:
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

    if not shutil.which("vllm"):
        raise SystemExit(
            "vLLM executable not found. Run dataset/scripts/setup_qwen36_vllm_python_env.sh "
            "or activate an environment containing the correct vLLM build."
        )
    env = os.environ.copy()
    if args.vllm_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.vllm_cuda_visible_devices
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        spec.model,
        "--tensor-parallel-size",
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
    if args.vllm_swap_space and _vllm_serve_supports_flag("--swap-space"):
        cmd += ["--swap-space", str(args.vllm_swap_space)]
    if args.vllm_trust_remote_code:
        cmd.append("--trust-remote-code")
    if "qwen" in model_lower and os.environ.get("LOCAL_VLLM_ENABLE_THINKING", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        if _vllm_serve_supports_flag("--enable-reasoning"):
            cmd.append("--enable-reasoning")
        if _vllm_serve_supports_flag("--reasoning-parser"):
            cmd += ["--reasoning-parser", os.environ.get("VLLM_REASONING_PARSER", "qwen3")]
        else:
            logger.warning(
                "vLLM does not expose --reasoning-parser; install a Qwen3.6-capable vLLM build."
            )
    if _vllm_serve_supports_flag("--generation-config"):
        cmd += ["--generation-config", os.environ.get("VLLM_GENERATION_CONFIG", "auto")]

    logger.info("Starting vLLM server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
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
    effective_run_config: dict | None = None,
) -> str | None:
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
            effective_run_config=effective_run_config,
        )
        write_run_manifest(run_paths.manifest_path, manifest)
        return manifest["phase_invocation_id"]
    except Exception as exc:
        logger.warning("Failed to write run manifest: %s", exc)
        return None

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
        "--stage1_batch_size",
        type=int,
        default=None,
        help="Override Stage 1 query batch size per intent (batch_size_queries in run.yaml)",
    )
    parser.add_argument(
        "--stage2_batch_size",
        type=int,
        default=None,
        help="Override Stage 2 query batch size, i.e. query records grouped per batch call",
    )
    parser.add_argument(
        "--stage2_response_batch_size",
        type=int,
        default=None,
        help="Override Stage 2 response_batch_size, i.e. responses requested per query prompt",
    )
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
        "--rate_limit_qps",
        type=float,
        default=None,
        help="Override global request rate limit (queries per second) without editing run.yaml",
    )
    parser.add_argument(
        "--max_queries_total",
        type=int,
        default=None,
        help="Override Stage 1 max_queries_total without editing run.yaml",
    )
    parser.add_argument(
        "--max_responses_total",
        type=int,
        default=None,
        help="Override Stage 2 max_responses_total without editing run.yaml",
    )
    parser.add_argument(
        "--max_genui_total",
        type=int,
        default=None,
        help="Override Stage 3 max_genui_total without editing run.yaml",
    )
    parser.add_argument(
        "--k_queries_per_intent",
        type=int,
        default=None,
        help="Override Stage 1 k_queries_per_intent without editing run.yaml",
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
    eval_cfg = all_run_cfg.get("evaluation", {})
    benchmark_cfg = all_run_cfg.get("benchmark", {})
    render_cfg_root = all_run_cfg.get("render", {})
    stage5_cfg_root = all_run_cfg.get("stage5", {})
    stage3_prompt_path = _resolve_cfg_path(
        root,
        os.environ.get("A2UI_STAGE3_PROMPT_FILE") or run_cfg.get("stage3_prompt_file"),
        "prompts/genui_gen_mobile_a2ui_express_v1.md",
    )
    stage1_prompt_path = _resolve_cfg_path(
        root,
        os.environ.get("A2UI_STAGE1_PROMPT_FILE") or run_cfg.get("stage1_prompt_file"),
        "prompts/query_gen.md",
    )
    stage3_schema_path = _resolve_cfg_path(
        root,
        run_cfg.get("stage3_schema_file"),
        "schema/genui.schema.json",
    )
    if not stage1_prompt_path.exists():
        raise SystemExit(f"Missing Stage1 prompt file: {stage1_prompt_path}")
    if not stage3_prompt_path.exists():
        raise SystemExit(f"Missing Stage3 prompt file: {stage3_prompt_path}")
    if not stage3_schema_path.exists():
        raise SystemExit(f"Missing Stage3 schema file: {stage3_schema_path}")
    effective_rate_limit_qps = float(run_cfg.get("rate_limit_qps", 2))
    if args.rate_limit_qps is not None:
        effective_rate_limit_qps = float(args.rate_limit_qps)
    if args.max_queries_total is not None:
        run_cfg["max_queries_total"] = int(args.max_queries_total)
    if args.max_responses_total is not None:
        run_cfg["max_responses_total"] = int(args.max_responses_total)
    if args.max_genui_total is not None:
        run_cfg["max_genui_total"] = int(args.max_genui_total)
    if args.k_queries_per_intent is not None:
        run_cfg["k_queries_per_intent"] = int(args.k_queries_per_intent)
    _apply_generation_env_overrides(run_cfg)
    _apply_env_int_override(run_cfg, "query_max_tokens", "A2UI_QUERY_MAX_TOKENS")
    _apply_env_int_override(run_cfg, "stage1_intent_cycle_size", "A2UI_STAGE1_INTENT_CYCLE_SIZE")
    _apply_env_int_override(run_cfg, "batch_size_queries", "A2UI_STAGE1_BATCH_SIZE")
    _apply_env_int_override(run_cfg, "batch_size_queries", "STAGE1_BATCH_SIZE")
    _apply_env_int_override(run_cfg, "query_batch_size", "A2UI_STAGE2_BATCH_SIZE")
    _apply_env_int_override(run_cfg, "query_batch_size", "STAGE2_BATCH_SIZE")
    _apply_env_int_override(run_cfg, "response_batch_size", "A2UI_STAGE2_RESPONSE_BATCH_SIZE")
    _apply_env_int_override(run_cfg, "response_batch_size", "STAGE2_RESPONSE_BATCH_SIZE")
    _apply_env_int_override(run_cfg, "genui_batch_size", "A2UI_STAGE3_BATCH_SIZE")
    _apply_env_int_override(run_cfg, "genui_batch_size", "STAGE3_BATCH_SIZE")
    _apply_env_int_override(run_cfg, "response_max_tokens", "A2UI_RESPONSE_MAX_TOKENS")
    _apply_env_int_override(run_cfg, "genui_max_tokens", "A2UI_GENUI_MAX_TOKENS")
    _apply_env_int_override(run_cfg, "genui_prompt_max_tokens", "A2UI_GENUI_PROMPT_MAX_TOKENS")
    _apply_env_bool_override(run_cfg, "enable_prompt_cache", "A2UI_ENABLE_PROMPT_CACHE")
    _apply_env_bool_override(run_cfg, "response_group_by_intent", "A2UI_STAGE2_GROUP_BY_INTENT")
    _apply_env_bool_override(run_cfg, "response_batch_fallback_per_query", "A2UI_STAGE2_BATCH_FALLBACK_PER_QUERY")
    if args.stage1_batch_size is not None:
        run_cfg["batch_size_queries"] = int(args.stage1_batch_size)
    if args.stage2_batch_size is not None:
        run_cfg["query_batch_size"] = int(args.stage2_batch_size)
    if args.stage2_response_batch_size is not None:
        run_cfg["response_batch_size"] = int(args.stage2_response_batch_size)
    query_temperature = _env_float("A2UI_QUERY_TEMPERATURE", 0.7)
    response_temperature_env = (
        "A2UI_RESPONSE_TEMPERATURES"
        if (os.environ.get("A2UI_RESPONSE_TEMPERATURES") or "").strip()
        else "A2UI_RESPONSE_TEMPERATURE"
    )
    response_temperatures = _env_float_list(
        response_temperature_env,
        _ensure_list(run_cfg.get("response_temperatures", [0.7])),
    )

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
                prompt_path=stage1_prompt_path,
                adapter=adapter,
                run_dir=run_paths.run_dir,
                queries_path=base_queries_path,
                k_per_intent=int(run_cfg.get("k_queries_per_intent", 20)),
                batch_size=int(run_cfg.get("batch_size_queries", 10)),
                intent_batch_size=int(run_cfg.get("stage1_intent_batch_size", 1)),
                seed=int(run_cfg.get("seed", 42)),
                temperature=query_temperature,
                max_tokens=int(run_cfg.get("query_max_tokens", 8192)),
                rate_limiter=rate_limiter,
                cache=_build_prompt_cache(root, run_cfg),
                logger=logger,
                max_total=run_cfg.get("max_queries_total"),
                max_failures_per_intent=int(run_cfg.get("stage1_max_failures_per_intent", 3)),
                fill_missing_with_fallback=bool(run_cfg.get("stage1_fill_missing_with_fallback", True)),
                max_attempts=int(run_cfg.get("max_attempts", 3)),
                intent_cycle_size=int(run_cfg.get("stage1_intent_cycle_size", 0)),
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
            phase_invocation_id = _write_run_manifest_safe(
                root=root,
                run_id=model_run_id,
                stage="benchmark",
                spec=spec,
                run_paths=model_paths,
                run_cfg_path=run_cfg_path,
                models_cfg_path=models_cfg_path,
                logger=model_logger,
                effective_run_config=run_cfg,
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
                temperatures=response_temperatures,
                max_tokens=int(run_cfg.get("response_max_tokens", 8192)),
                seed=int(benchmark_cfg.get("fixed_seed", 123)),
                rate_limiter=rate_limiter,
                cache=_build_prompt_cache(root, run_cfg),
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
                schema_path=stage3_schema_path,
                artifacts_dir=model_paths.artifacts_dir,
                candidates_per_response=int(run_cfg.get("genui_candidates_per_response", 1)),
                max_repair_attempts=int(run_cfg.get("max_repair_attempts", 1)),
                max_tokens=int(run_cfg.get("genui_max_tokens", 8192)),
                prompt_max_tokens=int(run_cfg.get("genui_prompt_max_tokens", 60000)),
                batch_size=genui_batch_size,
                seed=int(benchmark_cfg.get("fixed_seed", 123)),
                rate_limiter=rate_limiter,
                cache=_build_prompt_cache(root, run_cfg),
                logger=model_logger,
                max_total=run_cfg.get("max_genui_total"),
                max_attempts=int(run_cfg.get("max_attempts", 3)),
                aggregates_path=model_paths.aggregates_path,
                aggregate_weights=eval_cfg.get("weights", {}),
                metric_version=eval_cfg.get("metric_version", "v5_4"),
                ir_formats=run_cfg.get("stage3_ir_formats"),
                phase_invocation_id=phase_invocation_id,
            )
            aggregates[model_name] = _compute_aggregates_with_backfill(
                model_paths.genui_path,
                model_paths.responses_path,
                eval_cfg.get("weights", {}),
                eval_cfg.get("metric_version", "v5_4"),
            )

        run_paths.aggregates_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
        logger.info("Benchmark complete. Aggregates stored at %s", run_paths.aggregates_path)
        return

    if args.model and args.model not in model_map:
        available = ", ".join(sorted(model_map.keys()))
        raise SystemExit(f"Unknown model '{args.model}'. Available: {available}")
    spec = model_map.get(args.model) if args.model else specs[0]
    phase_invocation_id = _write_run_manifest_safe(
        root=root,
        run_id=run_id,
        stage=args.stage or "unknown",
        spec=spec,
        run_paths=run_paths,
        run_cfg_path=run_cfg_path,
        models_cfg_path=models_cfg_path,
        logger=logger,
        effective_run_config=run_cfg,
    )

    # Recompute-only fast path:
    # If stage 3 outputs already exist for all expected response IDs/candidates,
    # skip adapter initialization/API calls and refresh aggregates.json directly.
    if args.stage == 3:
        stage3_candidates = int(run_cfg.get("genui_candidates_per_response", 1))
        if _is_stage3_complete(
            run_paths.responses_path,
            run_paths.genui_path,
            stage3_candidates,
            run_cfg.get("stage3_ir_formats"),
        ):
            if not run_paths.genui_path.exists():
                raise SystemExit(f"Missing genui file: {run_paths.genui_path}")
            aggregates = _compute_aggregates_with_backfill(
                run_paths.genui_path,
                run_paths.responses_path,
                eval_cfg.get("weights", {}),
                eval_cfg.get("metric_version", "v5_4"),
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
                prompt_path=stage1_prompt_path,
                adapter=adapter,
                run_dir=run_paths.run_dir,
                queries_path=run_paths.queries_path,
                k_per_intent=int(run_cfg.get("k_queries_per_intent", 20)),
                batch_size=int(run_cfg.get("batch_size_queries", 10)),
                intent_batch_size=int(run_cfg.get("stage1_intent_batch_size", 1)),
                seed=int(run_cfg.get("seed", 42)),
                temperature=query_temperature,
                max_tokens=int(run_cfg.get("query_max_tokens", 8192)),
                rate_limiter=rate_limiter,
                cache=_build_prompt_cache(root, run_cfg),
                logger=logger,
                max_total=run_cfg.get("max_queries_total"),
                max_failures_per_intent=int(run_cfg.get("stage1_max_failures_per_intent", 3)),
                fill_missing_with_fallback=bool(run_cfg.get("stage1_fill_missing_with_fallback", True)),
                max_attempts=int(run_cfg.get("max_attempts", 3)),
                intent_cycle_size=int(run_cfg.get("stage1_intent_cycle_size", 0)),
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
                temperatures=response_temperatures,
                max_tokens=int(run_cfg.get("response_max_tokens", 8192)),
                seed=int(run_cfg.get("seed", 42)),
                rate_limiter=rate_limiter,
                cache=_build_prompt_cache(root, run_cfg),
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
                schema_path=stage3_schema_path,
                artifacts_dir=run_paths.artifacts_dir,
                candidates_per_response=int(run_cfg.get("genui_candidates_per_response", 1)),
                max_repair_attempts=int(run_cfg.get("max_repair_attempts", 1)),
                max_tokens=int(run_cfg.get("genui_max_tokens", 8192)),
                prompt_max_tokens=int(prompt_max_tokens) if prompt_max_tokens else None,
                batch_size=genui_batch_size,
                seed=int(run_cfg.get("seed", 42)),
                rate_limiter=rate_limiter,
                cache=_build_prompt_cache(root, run_cfg),
                logger=logger,
                max_total=run_cfg.get("max_genui_total"),
                max_attempts=int(run_cfg.get("max_attempts", 3)),
                aggregates_path=run_paths.aggregates_path,
                aggregate_weights=eval_cfg.get("weights", {}),
                metric_version=eval_cfg.get("metric_version", "v5_4"),
                ir_formats=run_cfg.get("stage3_ir_formats"),
                phase_invocation_id=phase_invocation_id,
            )
            logger.info("Stage3 complete.")
            return

        if args.stage == 4:
            render_cfg = render_cfg_root if isinstance(render_cfg_root, dict) else {}
            viewport = render_cfg.get("viewport", {"width": 1280, "height": 720})
            base_viewport = viewport if isinstance(viewport, dict) else {"width": 1280, "height": 720}
            renderers_cfg = render_cfg.get("renderers")
            max_total = render_cfg.get("max_total")
            render_images = bool(render_cfg.get("render_images", True))
            image_format = str(render_cfg.get("image_format", "png"))
            timeout_ms = int(render_cfg.get("timeout_ms", 15000))
            wait_ms = int(render_cfg.get("wait_ms", 200))
            use_http_server = bool(render_cfg.get("use_http_server", True))
            parallel_workers = (
                int(args.render_workers)
                if args.render_workers is not None
                else int(render_cfg.get("parallel_workers", 1))
            )

            if isinstance(renderers_cfg, dict) and renderers_cfg:
                primary_renderer = str(
                    render_cfg.get("primary_renderer") or next(iter(renderers_cfg.keys()))
                )
                comparison_renderers = [
                    str(item)
                    for item in _ensure_list(render_cfg.get("comparison_renderers", []))
                    if str(item).strip()
                ]
                ordered_renderers: list[str] = []
                if primary_renderer in renderers_cfg:
                    ordered_renderers.append(primary_renderer)
                for name in comparison_renderers:
                    if name in renderers_cfg and name not in ordered_renderers:
                        ordered_renderers.append(name)
                for name in renderers_cfg.keys():
                    if name not in ordered_renderers:
                        ordered_renderers.append(name)

                for index, renderer_name in enumerate(ordered_renderers):
                    renderer_conf = renderers_cfg.get(renderer_name)
                    if not isinstance(renderer_conf, dict):
                        continue
                    default_output = "rendered" if index == 0 else f"rendered_{renderer_name}"
                    default_log = "render.jsonl" if index == 0 else f"render_{renderer_name}.jsonl"
                    output_dir = run_paths.run_dir / str(renderer_conf.get("output_dir", default_output))
                    assets_dir = _resolve_cfg_path(root, renderer_conf.get("assets_dir"), "renderer/lit")
                    renderer_viewport = renderer_conf.get("viewport", base_viewport)
                    run_stage4(
                        genui_path=run_paths.genui_path,
                        output_dir=output_dir,
                        assets_dir=assets_dir,
                        server_root=root,
                        logger=logger,
                        max_total=max_total,
                        render_images=render_images,
                        image_format=image_format,
                        viewport=renderer_viewport if isinstance(renderer_viewport, dict) else base_viewport,
                        timeout_ms=timeout_ms,
                        wait_ms=wait_ms,
                        use_http_server=use_http_server,
                        parallel_workers=parallel_workers,
                        renderer_name=renderer_name,
                        payload_format=str(renderer_conf.get("payload_format", "messages")),
                        render_log_filename=str(renderer_conf.get("render_log", default_log)),
                    )
                    logger.info(
                        "Stage4 renderer=%s complete. Outputs stored at %s",
                        renderer_name,
                        output_dir,
                    )
            else:
                output_dir = run_paths.run_dir / render_cfg.get("output_dir", "rendered")
                assets_dir = _resolve_cfg_path(root, render_cfg.get("assets_dir"), "renderer/lit")
                run_stage4(
                    genui_path=run_paths.genui_path,
                    output_dir=output_dir,
                    assets_dir=assets_dir,
                    server_root=root,
                    logger=logger,
                    max_total=max_total,
                    render_images=render_images,
                    image_format=image_format,
                    viewport=base_viewport,
                    timeout_ms=timeout_ms,
                    wait_ms=wait_ms,
                    use_http_server=use_http_server,
                    parallel_workers=parallel_workers,
                    renderer_name="lit",
                    payload_format="messages",
                    render_log_filename="render.jsonl",
                )
                logger.info("Stage4 complete. Rendered outputs stored at %s", output_dir)
            return

        if args.stage == 5:
            render_cfg = render_cfg_root if isinstance(render_cfg_root, dict) else {}
            stage5_cfg = stage5_cfg_root if isinstance(stage5_cfg_root, dict) else {}
            output_dir = run_paths.run_dir / stage5_cfg.get("output_dir", "stage5_rendered")
            viewport = stage5_cfg.get("viewport", render_cfg.get("viewport", {"width": 1280, "height": 720}))
            run_stage5(
                responses_path=run_paths.responses_path,
                output_dir=output_dir,
                run_dir=run_paths.run_dir,
                server_root=root,
                logger=logger,
                render_images=bool(stage5_cfg.get("render_images", render_cfg.get("render_images", True))),
                image_format=str(stage5_cfg.get("image_format", render_cfg.get("image_format", "png"))),
                viewport=viewport if isinstance(viewport, dict) else {"width": 1280, "height": 720},
                timeout_ms=int(stage5_cfg.get("timeout_ms", render_cfg.get("timeout_ms", 15000))),
                wait_ms=int(stage5_cfg.get("wait_ms", render_cfg.get("wait_ms", 200))),
                use_http_server=bool(stage5_cfg.get("use_http_server", render_cfg.get("use_http_server", True))),
            )
            logger.info("Stage5 complete. Direct HTML outputs stored at %s", output_dir)
            return
    finally:
        _stop_vllm(server_proc, logger)

    parser.print_help()


if __name__ == "__main__":
    main()
