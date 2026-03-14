#!/usr/bin/env python3
"""
Benchmark Gemini caching strategies for GenUICraft Stage 2 -> Stage 3 pipeline.

This script compares:
1) Stage 3 direct (no explicit cache; implicit caching may still apply)
2) Stage 3 explicit cache (cold create + generate each run)
3) Stage 3 explicit cache (warm hit; reuse cache)
4) Pipeline sequential cold (stage2 -> create cache -> stage3)
5) Pipeline overlap cold (stage2 || create cache, then stage3)

Outputs:
- JSON report under tmp/selftest/cache_bench
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import string
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, parse, request


ANDROID_DIR = Path(__file__).resolve().parents[2]
ROOT = ANDROID_DIR.parent
PROMPT_STAGE2 = ANDROID_DIR / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "response_gen.md"
PROMPT_STAGE3 = ANDROID_DIR / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen.md"
DEFAULT_ENV_PATHS = [
    ROOT / "dataset" / ".env",
    ROOT / ".env",
    ANDROID_DIR / ".env",
]
OUT_DIR = ANDROID_DIR / "tmp" / "selftest" / "cache_bench"


@dataclass
class ApiResult:
    ok: bool
    status: int
    elapsed_ms: int
    raw: str
    err: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env_key(paths: List[Path]) -> str:
    for p in paths:
        if not p.exists():
            continue
        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            if k == "GEMINI_API_KEY" and v:
                return v
    env = os.getenv("GEMINI_API_KEY", "").strip()
    if env:
        return env
    return ""


def render_prompt(template: str, args: Dict[str, str]) -> str:
    out = template
    for k, v in args.items():
        out = out.replace("{" + k + "}", v)
    return out


def prepare_stage3_prompt_context(template: str) -> Tuple[str, str]:
    placeholder = "{response_text}"
    if placeholder not in template:
        return (
            template.strip(),
            (
                "Convert the response text into valid GenUICraft JSON.\n"
                "Return ONLY the JSON message array.\n\n"
                "Response:\n{response_text}"
            ),
        )
    split = template.split(placeholder, 1)
    system_prompt = (split[0] + "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]" + split[1]).strip()
    user_template = (
        "Convert the response text into valid GenUICraft JSON.\n"
        "Return ONLY the JSON message array.\n\n"
        "Response:\n{response_text}"
    )
    return system_prompt, user_template


def build_stage3_user_prompt(user_template: str, stage2_response: str) -> str:
    asset_policy = (
        "Asset URL policy for this request:\n"
        "- No local asset mapping is provided.\n"
        "- Preserve media URLs from the response exactly as written.\n"
        "- Do not invent local placeholder paths such as /image.jpg or /asset/foo.png."
    )
    response_text = f"{stage2_response.strip()}\n\n{asset_policy}"
    return render_prompt(user_template, {"response_text": response_text})


def post_json(url: str, payload: Dict[str, Any], timeout_sec: int = 210) -> ApiResult:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            elapsed_ms = int((time.time() - started) * 1000)
            return ApiResult(ok=True, status=resp.getcode(), elapsed_ms=elapsed_ms, raw=raw, err=None)
    except error.HTTPError as e:
        elapsed_ms = int((time.time() - started) * 1000)
        body = e.read().decode("utf-8", errors="replace")
        return ApiResult(ok=False, status=e.code, elapsed_ms=elapsed_ms, raw=body, err=f"HTTP {e.code}")
    except Exception as e:  # noqa: BLE001
        elapsed_ms = int((time.time() - started) * 1000)
        return ApiResult(ok=False, status=0, elapsed_ms=elapsed_ms, raw="", err=str(e))


def parse_json(raw: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_text_from_generate(raw: str) -> Optional[str]:
    root = parse_json(raw)
    if not root:
        return None
    candidates = root.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0] if isinstance(candidates[0], dict) else None
    if not first:
        return None
    content = first.get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    chunks: List[str] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        t = p.get("text")
        if isinstance(t, str) and t.strip():
            chunks.append(t)
    text = "\n".join(chunks).strip()
    return text or None


def extract_usage(raw: str) -> Dict[str, Any]:
    root = parse_json(raw) or {}
    usage = root.get("usageMetadata")
    if not isinstance(usage, dict):
        usage = {}
    return {
        "promptTokenCount": usage.get("promptTokenCount"),
        "cachedContentTokenCount": usage.get("cachedContentTokenCount"),
        "candidatesTokenCount": usage.get("candidatesTokenCount"),
        "totalTokenCount": usage.get("totalTokenCount"),
    }


def build_stage3_schema() -> Dict[str, Any]:
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "version": {"type": "STRING"},
                "createSurface": {"type": "OBJECT"},
                "updateComponents": {"type": "OBJECT"},
                "clearSurface": {"type": "OBJECT"},
            },
        },
    }


def build_generate_payload(
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    max_output_tokens: int,
    json_mode: bool,
    enable_google_search: bool,
    cached_content_name: Optional[str],
    structured_output: bool,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": min(max_output_tokens, 8192),
        },
    }
    if cached_content_name:
        body["cachedContent"] = cached_content_name
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if enable_google_search:
        body["tools"] = [{"google_search": {}}]
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
        if structured_output:
            body["generationConfig"]["responseSchema"] = build_stage3_schema()
    return body


def call_generate_with_retry(
    api_key: str,
    model: str,
    payload: Dict[str, Any],
    retries: int = 3,
) -> ApiResult:
    encoded_key = parse.quote(api_key, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={encoded_key}"
    last: Optional[ApiResult] = None
    for attempt in range(1, retries + 1):
        res = post_json(url, payload, timeout_sec=240)
        last = res
        if res.ok:
            return res
        raw_low = (res.raw or "").lower()
        retryable = (
            res.status in (429, 500, 502, 503, 504)
            or "timed out" in raw_low
            or "timeout" in raw_low
            or "candidate text" in raw_low
        )
        if not retryable or attempt == retries:
            return res
        time.sleep(attempt)
    return last if last else ApiResult(False, 0, 0, "", "Unknown")


def create_cache(
    api_key: str,
    model: str,
    system_prompt: str,
    ttl_seconds: int = 21600,
) -> Tuple[ApiResult, Optional[str]]:
    encoded_key = parse.quote(api_key, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/cachedContents?key={encoded_key}"
    display_suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    body = {
        "model": f"models/{model}",
        "displayName": f"genuicraft_bench_{display_suffix}",
        "ttl": f"{ttl_seconds}s",
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Use the cached GenUICraft conversion instructions."}],
            }
        ],
    }
    res = post_json(url, body, timeout_sec=90)
    if not res.ok:
        return res, None
    root = parse_json(res.raw) or {}
    name = root.get("name")
    if not isinstance(name, str) or not name.strip():
        return ApiResult(False, res.status, res.elapsed_ms, res.raw, "Cache name missing"), None
    return res, name


def count_tokens_for_stage3(
    api_key: str,
    model: str,
    stage3_user_prompt: str,
    stage3_system_prompt: str,
) -> Dict[str, Any]:
    encoded_key = parse.quote(api_key, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens?key={encoded_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": stage3_user_prompt}]}],
        "systemInstruction": {"parts": [{"text": stage3_system_prompt}]},
    }
    res = post_json(url, payload, timeout_sec=60)
    if not res.ok:
        return {"ok": False, "error": res.err, "status": res.status}
    root = parse_json(res.raw) or {}
    return {"ok": True, "totalTokens": root.get("totalTokens"), "raw": root}


def median_or_none(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def p95_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
    return vals[idx]


def summarize_runs(runs: List[Dict[str, Any]], key: str = "latency_ms") -> Dict[str, Any]:
    vals = [float(r[key]) for r in runs if isinstance(r.get(key), (int, float))]
    return {
        "count": len(vals),
        "median_ms": median_or_none(vals),
        "p95_ms": p95_or_none(vals),
        "min_ms": min(vals) if vals else None,
        "max_ms": max(vals) if vals else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--query", default="weather in bengaluru next 5 days with current conditions")
    parser.add_argument("--runs-stage3", type=int, default=5)
    parser.add_argument("--runs-pipeline", type=int, default=3)
    args = parser.parse_args()

    api_key = load_env_key(DEFAULT_ENV_PATHS)
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in expected .env locations or env vars.", file=sys.stderr)
        return 2

    stage2_template = PROMPT_STAGE2.read_text(encoding="utf-8")
    stage3_template = PROMPT_STAGE3.read_text(encoding="utf-8")
    stage3_system_prompt, stage3_user_template = prepare_stage3_prompt_context(stage3_template)

    stage2_prompt = render_prompt(
        stage2_template,
        {
            "query_text": args.query.strip(),
        },
    )

    # Build one realistic Stage 2 response for Stage 3 benchmarking.
    stage2_payload = build_generate_payload(
        prompt=stage2_prompt,
        system_prompt=None,
        temperature=0.3,
        max_output_tokens=4096,
        json_mode=False,
        enable_google_search=True,
        cached_content_name=None,
        structured_output=False,
    )
    stage2_res = call_generate_with_retry(api_key, args.model, stage2_payload)
    if not stage2_res.ok:
        print(f"ERROR: Stage 2 seed call failed: {stage2_res.err} status={stage2_res.status}", file=sys.stderr)
        return 3
    stage2_text = extract_text_from_generate(stage2_res.raw)
    if not stage2_text:
        print("ERROR: Stage 2 seed call returned no candidate text.", file=sys.stderr)
        return 4

    stage3_user_prompt = build_stage3_user_prompt(stage3_user_template, stage2_text)
    token_info = count_tokens_for_stage3(api_key, args.model, stage3_user_prompt, stage3_system_prompt)

    results: Dict[str, Any] = {
        "createdAt": now_iso(),
        "model": args.model,
        "query": args.query,
        "stage2SeedLatencyMs": stage2_res.elapsed_ms,
        "stage2SeedUsage": extract_usage(stage2_res.raw),
        "stage3TokenCount": token_info,
        "experiments": {},
    }

    # 1) Stage 3 direct (no explicit cache)
    direct_runs: List[Dict[str, Any]] = []
    for i in range(args.runs_stage3):
        payload = build_generate_payload(
            prompt=stage3_user_prompt,
            system_prompt=stage3_system_prompt,
            temperature=0.2,
            max_output_tokens=8192,
            json_mode=True,
            enable_google_search=False,
            cached_content_name=None,
            structured_output=True,
        )
        res = call_generate_with_retry(api_key, args.model, payload)
        usage = extract_usage(res.raw) if res.raw else {}
        direct_runs.append(
            {
                "run": i + 1,
                "ok": res.ok,
                "latency_ms": res.elapsed_ms,
                "status": res.status,
                "error": res.err,
                "usage": usage,
                "has_text": bool(extract_text_from_generate(res.raw or "")),
            }
        )
    results["experiments"]["stage3_direct_no_explicit"] = {
        "runs": direct_runs,
        "summary": summarize_runs(direct_runs),
    }

    # 2) Stage 3 explicit cache cold (create + generate every run)
    cold_runs: List[Dict[str, Any]] = []
    for i in range(args.runs_stage3):
        create_res, cache_name = create_cache(api_key, args.model, stage3_system_prompt)
        if not create_res.ok or not cache_name:
            cold_runs.append(
                {
                    "run": i + 1,
                    "ok": False,
                    "create_ms": create_res.elapsed_ms,
                    "generate_ms": None,
                    "latency_ms": create_res.elapsed_ms,
                    "status": create_res.status,
                    "error": create_res.err or "cache_create_failed",
                    "usage": {},
                }
            )
            continue
        payload = build_generate_payload(
            prompt=stage3_user_prompt,
            system_prompt=None,
            temperature=0.2,
            max_output_tokens=8192,
            json_mode=True,
            enable_google_search=False,
            cached_content_name=cache_name,
            structured_output=True,
        )
        gen_res = call_generate_with_retry(api_key, args.model, payload)
        usage = extract_usage(gen_res.raw) if gen_res.raw else {}
        cold_runs.append(
            {
                "run": i + 1,
                "ok": gen_res.ok,
                "create_ms": create_res.elapsed_ms,
                "generate_ms": gen_res.elapsed_ms,
                "latency_ms": create_res.elapsed_ms + gen_res.elapsed_ms,
                "status": gen_res.status,
                "error": gen_res.err,
                "usage": usage,
                "has_text": bool(extract_text_from_generate(gen_res.raw or "")),
            }
        )
    results["experiments"]["stage3_explicit_cold_create_each_run"] = {
        "runs": cold_runs,
        "summary_total": summarize_runs(cold_runs, key="latency_ms"),
        "summary_create": summarize_runs(
            [{"latency_ms": r["create_ms"]} for r in cold_runs if isinstance(r.get("create_ms"), (int, float))]
        ),
        "summary_generate": summarize_runs(
            [{"latency_ms": r["generate_ms"]} for r in cold_runs if isinstance(r.get("generate_ms"), (int, float))]
        ),
    }

    # 3) Stage 3 explicit cache warm (single create, repeated generates)
    warm_create, warm_cache_name = create_cache(api_key, args.model, stage3_system_prompt)
    warm_runs: List[Dict[str, Any]] = []
    if warm_create.ok and warm_cache_name:
        for i in range(args.runs_stage3):
            payload = build_generate_payload(
                prompt=stage3_user_prompt,
                system_prompt=None,
                temperature=0.2,
                max_output_tokens=8192,
                json_mode=True,
                enable_google_search=False,
                cached_content_name=warm_cache_name,
                structured_output=True,
            )
            gen_res = call_generate_with_retry(api_key, args.model, payload)
            usage = extract_usage(gen_res.raw) if gen_res.raw else {}
            warm_runs.append(
                {
                    "run": i + 1,
                    "ok": gen_res.ok,
                    "latency_ms": gen_res.elapsed_ms,
                    "status": gen_res.status,
                    "error": gen_res.err,
                    "usage": usage,
                    "has_text": bool(extract_text_from_generate(gen_res.raw or "")),
                }
            )
    results["experiments"]["stage3_explicit_warm_reuse_cache"] = {
        "cache_create": {
            "ok": warm_create.ok,
            "latency_ms": warm_create.elapsed_ms,
            "status": warm_create.status,
            "error": warm_create.err,
        },
        "runs": warm_runs,
        "summary": summarize_runs(warm_runs),
    }

    # 4) Pipeline sequential cold and 5) overlap cold
    pipeline_seq_runs: List[Dict[str, Any]] = []
    pipeline_ov_runs: List[Dict[str, Any]] = []

    for i in range(args.runs_pipeline):
        # Sequential cold: stage2 -> create cache -> stage3
        s2_res = call_generate_with_retry(api_key, args.model, stage2_payload)
        if not s2_res.ok:
            pipeline_seq_runs.append(
                {
                    "run": i + 1,
                    "ok": False,
                    "latency_ms": s2_res.elapsed_ms,
                    "stage2_ms": s2_res.elapsed_ms,
                    "cache_create_ms": None,
                    "stage3_ms": None,
                    "error": s2_res.err,
                }
            )
        else:
            s2_text = extract_text_from_generate(s2_res.raw or "") or stage2_text
            s3_prompt_local = build_stage3_user_prompt(stage3_user_template, s2_text)
            c_res, c_name = create_cache(api_key, args.model, stage3_system_prompt)
            if not c_res.ok or not c_name:
                pipeline_seq_runs.append(
                    {
                        "run": i + 1,
                        "ok": False,
                        "latency_ms": s2_res.elapsed_ms + c_res.elapsed_ms,
                        "stage2_ms": s2_res.elapsed_ms,
                        "cache_create_ms": c_res.elapsed_ms,
                        "stage3_ms": None,
                        "error": c_res.err or "cache_create_failed",
                    }
                )
            else:
                payload = build_generate_payload(
                    prompt=s3_prompt_local,
                    system_prompt=None,
                    temperature=0.2,
                    max_output_tokens=8192,
                    json_mode=True,
                    enable_google_search=False,
                    cached_content_name=c_name,
                    structured_output=True,
                )
                s3_res = call_generate_with_retry(api_key, args.model, payload)
                pipeline_seq_runs.append(
                    {
                        "run": i + 1,
                        "ok": s3_res.ok,
                        "latency_ms": s2_res.elapsed_ms + c_res.elapsed_ms + s3_res.elapsed_ms,
                        "stage2_ms": s2_res.elapsed_ms,
                        "cache_create_ms": c_res.elapsed_ms,
                        "stage3_ms": s3_res.elapsed_ms,
                        "error": s3_res.err,
                    }
                )

        # Overlap cold: (stage2 || create cache) -> stage3
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_stage2 = ex.submit(call_generate_with_retry, api_key, args.model, stage2_payload)
            fut_cache = ex.submit(create_cache, api_key, args.model, stage3_system_prompt)
            s2_res_ov = fut_stage2.result()
            c_pair_ov = fut_cache.result()
        c_res_ov, c_name_ov = c_pair_ov
        if not s2_res_ov.ok:
            pipeline_ov_runs.append(
                {
                    "run": i + 1,
                    "ok": False,
                    "latency_ms": max(s2_res_ov.elapsed_ms, c_res_ov.elapsed_ms),
                    "stage2_ms": s2_res_ov.elapsed_ms,
                    "cache_create_ms": c_res_ov.elapsed_ms,
                    "stage3_ms": None,
                    "error": s2_res_ov.err,
                }
            )
        elif not c_res_ov.ok or not c_name_ov:
            pipeline_ov_runs.append(
                {
                    "run": i + 1,
                    "ok": False,
                    "latency_ms": max(s2_res_ov.elapsed_ms, c_res_ov.elapsed_ms),
                    "stage2_ms": s2_res_ov.elapsed_ms,
                    "cache_create_ms": c_res_ov.elapsed_ms,
                    "stage3_ms": None,
                    "error": c_res_ov.err or "cache_create_failed",
                }
            )
        else:
            s2_text_ov = extract_text_from_generate(s2_res_ov.raw or "") or stage2_text
            s3_prompt_ov = build_stage3_user_prompt(stage3_user_template, s2_text_ov)
            payload = build_generate_payload(
                prompt=s3_prompt_ov,
                system_prompt=None,
                temperature=0.2,
                max_output_tokens=8192,
                json_mode=True,
                enable_google_search=False,
                cached_content_name=c_name_ov,
                structured_output=True,
            )
            s3_res_ov = call_generate_with_retry(api_key, args.model, payload)
            pipeline_ov_runs.append(
                {
                    "run": i + 1,
                    "ok": s3_res_ov.ok,
                    "latency_ms": max(s2_res_ov.elapsed_ms, c_res_ov.elapsed_ms) + s3_res_ov.elapsed_ms,
                    "stage2_ms": s2_res_ov.elapsed_ms,
                    "cache_create_ms": c_res_ov.elapsed_ms,
                    "stage3_ms": s3_res_ov.elapsed_ms,
                    "error": s3_res_ov.err,
                }
            )

    results["experiments"]["pipeline_sequential_cold"] = {
        "runs": pipeline_seq_runs,
        "summary_total": summarize_runs(pipeline_seq_runs),
        "summary_stage2": summarize_runs(
            [{"latency_ms": r["stage2_ms"]} for r in pipeline_seq_runs if isinstance(r.get("stage2_ms"), (int, float))]
        ),
        "summary_cache_create": summarize_runs(
            [{"latency_ms": r["cache_create_ms"]} for r in pipeline_seq_runs if isinstance(r.get("cache_create_ms"), (int, float))]
        ),
        "summary_stage3": summarize_runs(
            [{"latency_ms": r["stage3_ms"]} for r in pipeline_seq_runs if isinstance(r.get("stage3_ms"), (int, float))]
        ),
    }
    results["experiments"]["pipeline_overlap_cold"] = {
        "runs": pipeline_ov_runs,
        "summary_total": summarize_runs(pipeline_ov_runs),
        "summary_stage2": summarize_runs(
            [{"latency_ms": r["stage2_ms"]} for r in pipeline_ov_runs if isinstance(r.get("stage2_ms"), (int, float))]
        ),
        "summary_cache_create": summarize_runs(
            [{"latency_ms": r["cache_create_ms"]} for r in pipeline_ov_runs if isinstance(r.get("cache_create_ms"), (int, float))]
        ),
        "summary_stage3": summarize_runs(
            [{"latency_ms": r["stage3_ms"]} for r in pipeline_ov_runs if isinstance(r.get("stage3_ms"), (int, float))]
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"gemini_cache_benchmark_{args.model.replace('/', '_')}_{ts}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved benchmark report: {out_file}")
    print(json.dumps(results["experiments"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
