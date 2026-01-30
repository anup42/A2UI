from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.cache import PromptCache
from pipeline.common import extract_json, load_prompt, render_prompt
from pipeline.metrics import content_coverage, dup_rate, lint_score, count_tokens
from pipeline.storage import JsonlWriter, iter_jsonl
from pipeline.toon_convert import encode_toon, roundtrip_ok
from llm.base import BaseLLMAdapter, LLMRateLimitError
from utils.hashing import hash_text
from utils.rate_limit import RateLimiter
from utils.retry import with_retry


def _validate_schema(schema: dict[str, Any], data: Any, schema_dir: Path) -> tuple[bool, list[str], bool]:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return False, ["jsonschema not installed"], False
    store: dict[str, Any] = {}
    for schema_path in schema_dir.glob("*.json"):
        try:
            content = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        schema_id = content.get("$id")
        if schema_id:
            store[schema_id] = content
        # Map known spec URLs to local copies when filenames are present.
        if schema_path.name == "catalog.json":
            store["https://a2ui.org/specification/v0_9/catalog.json"] = content
        if schema_path.name == "common_types.json":
            store["https://a2ui.org/specification/v0_9/common_types.json"] = content
        if schema_path.name == "server_to_client.json":
            store["https://a2ui.org/specification/v0_9/server_to_client.json"] = content
        if schema_path.name == "server_to_client_list.json":
            store["https://a2ui.org/specification/v0_9/server_to_client_list.json"] = content
    try:
        resolver = jsonschema.RefResolver.from_schema(schema, store=store)
        validator = jsonschema.Draft202012Validator(schema, resolver=resolver)
        validator.validate(instance=data)
        return True, [], True
    except Exception as exc:
        return False, [str(exc)], True


def _make_ui_id(query_id: str, n_idx: int, candidate_idx: int) -> str:
    suffix = query_id.replace("q_", "")
    if candidate_idx == 1:
        return f"u_{suffix}_{n_idx:02d}"
    return f"u_{suffix}_{n_idx:02d}_{candidate_idx:02d}"


def _build_asset_context(assets: list[dict]) -> str:
    if not assets:
        return ""
    lines: list[str] = []
    for item in assets:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        local_path = path.replace("\\", "/")
        if not local_path.startswith("/"):
            local_path = "/" + local_path.lstrip("/")
        if url:
            lines.append(f"- {url} -> {local_path}")
        else:
            lines.append(f"- {local_path}")
    if not lines:
        return ""
    return "Assets (local copies of any URLs in the response; use ONLY these local paths):\n" + "\n".join(lines)


def run_stage3(
    responses_path: Path,
    prompt_path: Path,
    adapter: BaseLLMAdapter,
    a2ui_path: Path,
    schema_path: Path,
    artifacts_dir: Path,
    candidates_per_response: int,
    max_repair_attempts: int,
    max_tokens: int,
    seed: int,
    rate_limiter: RateLimiter,
    cache: PromptCache,
    logger,
    max_total: int | None = None,
    max_attempts: int = 3,
) -> None:
    prompt_template = load_prompt(prompt_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = {row.get("ui_id") for row in iter_jsonl(a2ui_path)}
    writer = JsonlWriter(a2ui_path)

    total_created = 0
    for response in iter_jsonl(responses_path):
        response_id = response.get("response_id")
        query_id = response.get("query_id")
        response_text = response.get("response_text")
        assets = response.get("assets") if isinstance(response, dict) else None
        assets_list = assets if isinstance(assets, list) else []
        n_idx = int(response.get("n_idx", 1))
        if not response_id or not query_id or not response_text:
            continue

        for c_idx in range(1, candidates_per_response + 1):
            if max_total is not None and total_created >= max_total:
                logger.info("Stage3 reached max_total=%s", max_total)
                return
            ui_id = _make_ui_id(query_id, n_idx, c_idx)
            if ui_id in existing_ids:
                continue

            asset_context = _build_asset_context(assets_list)
            prompt_response_text = response_text
            if asset_context:
                prompt_response_text = f"{response_text}\n\n{asset_context}"
            prompt = render_prompt(prompt_template, response_text=prompt_response_text)
            prompt_hash = hash_text(f"{adapter.spec.name}:{prompt}")

            cached = cache.get(prompt_hash)
            if cached:
                raw_text = cached.text
                raw_payload = cached.raw
                latency_ms = 0.0
                input_tokens = 0
                output_tokens = 0
                provider = adapter.spec.provider
                model = adapter.spec.model
                error = None
            else:
                def _call():
                    rate_limiter.acquire()
                    return adapter.generate(
                        prompt=prompt,
                        system=None,
                        temperature=0.2,
                        max_tokens=max_tokens,
                        seed=seed + c_idx,
                        json_mode=True if adapter.spec.supports_json_mode else False,
                    )

                try:
                    result = with_retry(_call, max_attempts=max_attempts)
                except Exception as exc:
                    if isinstance(exc, LLMRateLimitError):
                        logger.error(
                            "Stage3 rate limit info: limits=%s headers=%s",
                            exc.limits or "unset",
                            exc.headers or "none",
                        )
                    raise
                if result.error:
                    logger.error("Stage3 error response_id=%s: %s", response_id, result.error)
                    raise RuntimeError(result.error)
                raw_text = result.text
                raw_payload = result.raw
                latency_ms = result.latency_ms
                input_tokens = result.input_tokens
                output_tokens = result.output_tokens
                provider = result.provider
                model = result.model
                error = result.error
                cache.set(prompt_hash, raw_text, raw_payload)

            parsed_ok = True
            errors: list[str] = []
            try:
                a2ui_json = extract_json(raw_text)
            except Exception as exc:
                parsed_ok = False
                a2ui_json = None
                errors.append(f"json_parse_error: {exc}")

            schema_valid_strict = False
            schema_valid_lenient = False
            repair_needed = False

            if parsed_ok and a2ui_json is not None:
                schema_valid_strict, schema_errors, validator_ok = _validate_schema(schema, a2ui_json, schema_path.parent)
                if schema_valid_strict:
                    schema_valid_lenient = True
                else:
                    errors.extend(schema_errors)
                    if not validator_ok:
                        schema_valid_lenient = True

            repair_attempts = 0
            while parsed_ok and not schema_valid_strict and repair_attempts < max_repair_attempts:
                repair_attempts += 1
                repair_needed = True
                repair_prompt = (
                    "The previous output failed schema validation."
                    "Fix the JSON to satisfy the schema."
                    f"Errors: {errors}.\n"
                    "Return only the corrected JSON."
                )
                repaired_text = f"{repair_prompt}\n\nOriginal:\n{raw_text}"

                def _repair_call():
                    rate_limiter.acquire()
                    return adapter.generate(
                        prompt=repaired_text,
                        system=None,
                        temperature=0.2,
                        max_tokens=max_tokens,
                        seed=seed + 100 + repair_attempts,
                        json_mode=True if adapter.spec.supports_json_mode else False,
                    )

                try:
                    result = with_retry(_repair_call, max_attempts=max_attempts)
                except Exception as exc:
                    if isinstance(exc, LLMRateLimitError):
                        logger.error(
                            "Stage3 rate limit info: limits=%s headers=%s",
                            exc.limits or "unset",
                            exc.headers or "none",
                        )
                    raise
                if result.error:
                    errors.append(f"repair_error: {result.error}")
                    raise RuntimeError(result.error)
                raw_text = result.text
                try:
                    a2ui_json = extract_json(raw_text)
                    parsed_ok = True
                except Exception as exc:
                    parsed_ok = False
                    errors.append(f"repair_json_parse_error: {exc}")
                    break

                schema_valid_strict, schema_errors, validator_ok = _validate_schema(schema, a2ui_json, schema_path.parent)
                if schema_valid_strict:
                    schema_valid_lenient = True
                else:
                    errors.extend(schema_errors)
                    if not validator_ok:
                        schema_valid_lenient = True

            if not parsed_ok or a2ui_json is None:
                a2ui_json = {}

            toon = encode_toon(a2ui_json)
            toon_ok = roundtrip_ok(a2ui_json, toon)

            metrics = {
                "content_coverage": content_coverage(response_text, a2ui_json),
                "dup_rate": dup_rate(a2ui_json),
                "lint_score": lint_score(a2ui_json),
                "output_tokens_toon": count_tokens(toon),
                "output_tokens_json": count_tokens(json.dumps(a2ui_json, ensure_ascii=False)),
            }

            short_errors = [e[:300] + ("..." if len(e) > 300 else "") for e in errors]
            record = {
                "ui_id": ui_id,
                "response_id": response_id,
                "query_id": query_id,
                "a2ui_json": a2ui_json,
                "assets": assets_list,
                "toon": toon,
                "validation": {
                    "json_parse_ok": parsed_ok,
                    "schema_valid_strict": schema_valid_strict,
                    "schema_valid_lenient": schema_valid_lenient,
                    "toon_roundtrip_ok": toon_ok,
                    "errors": short_errors,
                    "repair_attempts": repair_attempts,
                    "repair_needed": repair_needed,
                },
                "metrics": metrics,
                "gen": {
                    "provider": provider,
                    "model": model,
                    "prompt_version": "a2ui_gen_v1",
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": None,
                    "error": error,
                },
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            writer.append(record)
            existing_ids.add(ui_id)
            logger.info("Stage3 created ui_id=%s schema_ok=%s", ui_id, schema_valid_strict)
            total_created += 1
            if max_total is not None and total_created >= max_total:
                logger.info("Stage3 reached max_total=%s", max_total)
                return

            if errors:
                error_path = artifacts_dir / f"error_{ui_id}.json"
                error_payload = {
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "errors": errors,
                }
                error_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
