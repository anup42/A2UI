from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.cache import PromptCache
from pipeline.common import extract_json, load_prompt, render_prompt
from pipeline.metrics import (
    content_coverage,
    dup_rate,
    lint_score,
    count_tokens,
    count_characters,
    aggregate_metrics,
    compute_overall_score,
    compute_ui_metrics,
    compute_intent_metrics,
)
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
        store[schema_path.name] = content
        schema_id = content.get("$id")
        if schema_id:
            store[schema_id] = content
        # Map known spec URLs to local copies when filenames are present.
        if schema_path.name == "catalog.json":
            store["https://genui.local/specification/v0_9/catalog.json"] = content
        if schema_path.name == "common_types.json":
            store["https://genui.local/specification/v0_9/common_types.json"] = content
        if schema_path.name == "server_to_client.json":
            store["https://genui.local/specification/v0_9/server_to_client.json"] = content
        if schema_path.name == "server_to_client_list.json":
            store["https://genui.local/specification/v0_9/server_to_client_list.json"] = content

    # Prefer the modern referencing registry to avoid network fetches.
    try:
        from referencing import Registry, Resource  # type: ignore

        registry = Registry()
        for key, value in store.items():
            registry = registry.with_resource(key, Resource.from_contents(value))
        validator = jsonschema.Draft202012Validator(schema, registry=registry)
        validator.validate(instance=data)
        return True, [], True
    except Exception:
        # Fall back to the legacy resolver API.
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


def _apply_asset_replacements(text: str, assets: list[dict]) -> str:
    if not assets:
        return text
    valid_exts = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".bmp",
        ".tiff",
        ".pdf",
        ".zip",
    )
    updated = text
    for item in assets:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        path = str(item.get("path") or "").strip()
        if not url or not path:
            continue
        local_path = path.replace("\\", "/")
        if not local_path.lower().endswith(valid_exts):
            continue
        if not local_path.startswith("/"):
            local_path = "/" + local_path.lstrip("/")
        updated = updated.replace(url, local_path)
    return updated


def _truncate_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    if max_tokens <= 0:
        return "", True
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text, False
    trimmed = " ".join(tokens[:max_tokens]).strip()
    if trimmed:
        trimmed = f"{trimmed}\n\n[TRUNCATED]"
    return trimmed, True


def _maybe_compact_prompt_template(template: str, adapter: BaseLLMAdapter, logger) -> str:
    provider = (adapter.spec.provider or "").lower()
    model = (adapter.spec.model or "").lower()
    compact_enabled = os.getenv("GENUI_COMPACT_PROMPT_FOR_GEMMA", "1").strip().lower()
    if compact_enabled in {"0", "false", "no", "off"}:
        return template
    if provider != "gemini" or not model.startswith("gemma-"):
        return template

    compact = template
    marker = "\nSchema ("
    if marker in template:
        compact = template.split(marker, 1)[0].rstrip()
    compact = (
        f"{compact}\n\n"
        "Additional strict requirements:\n"
        "- Use message types: createSurface, updateComponents, updateDataModel, deleteSurface.\n"
        "- Set version to v0.9.\n"
        "- Include createSurface before updates.\n"
        "- In updateComponents, include exactly one root component with id 'root'.\n"
        "- Output ONLY a JSON array of messages.\n"
    )
    before = count_tokens(template)
    after = count_tokens(compact)
    logger.info(
        "Stage3 compact prompt enabled for model=%s tokens=%s->%s",
        adapter.spec.model,
        before,
        after,
    )
    return compact


def _prepare_prompt_context(
    template: str,
    adapter: BaseLLMAdapter,
    logger,
) -> tuple[str | None, str]:
    """Split Gemini Stage3 prompt into static system + small per-item user prompt."""
    provider = (adapter.spec.provider or "").lower()
    mode = os.getenv("GEMINI_STAGE3_PROMPT_MODE", "system_prefix").strip().lower()
    if provider != "gemini" or mode in {"inline", "legacy", "off", "0", "false"}:
        return None, template

    placeholder = "{response_text}"
    if placeholder not in template:
        logger.warning(
            "Stage3 Gemini system-prefix mode requested but prompt has no %s; using inline mode.",
            placeholder,
        )
        return None, template

    before, after = template.split(placeholder, 1)
    system_prompt = (
        f"{before}[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]{after}".strip()
    )
    user_template = (
        "Convert the response text into valid GenUICraft JSON.\n"
        "Return ONLY the JSON message array.\n\n"
        "Response:\n{response_text}"
    )
    logger.info(
        "Stage3 Gemini prompt mode=system_prefix system_tokens=%s user_template_tokens=%s",
        count_tokens(system_prompt),
        count_tokens(user_template),
    )
    return system_prompt, user_template


def run_stage3(
    queries_path: Path | None,
    responses_path: Path,
    prompt_path: Path,
    adapter: BaseLLMAdapter,
    genui_path: Path,
    schema_path: Path,
    artifacts_dir: Path,
    candidates_per_response: int,
    max_repair_attempts: int,
    max_tokens: int,
    prompt_max_tokens: int | None,
    seed: int,
    rate_limiter: RateLimiter,
    cache: PromptCache,
    logger,
    batch_size: int = 100,
    max_total: int | None = None,
    max_attempts: int = 3,
    aggregates_path: Path | None = None,
    aggregate_weights: dict[str, float] | None = None,
) -> None:
    prompt_template = _maybe_compact_prompt_template(load_prompt(prompt_path), adapter, logger)
    system_prompt, user_prompt_template = _prepare_prompt_context(prompt_template, adapter, logger)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    intent_lookup: dict[str, dict[str, Any]] = {}
    if queries_path and queries_path.exists():
        for row in iter_jsonl(queries_path):
            query_id = row.get("query_id")
            if not query_id:
                continue
            intent_lookup[query_id] = {
                "intent": row.get("intent"),
                "tags": row.get("tags") if isinstance(row.get("tags"), list) else [],
            }

    existing_ids = {row.get("ui_id") for row in iter_jsonl(genui_path)}
    writer = JsonlWriter(genui_path)
    response_text_by_id: dict[str, str] = {}
    if responses_path.exists():
        for row in iter_jsonl(responses_path):
            response_id = row.get("response_id")
            response_text = row.get("response_text")
            if isinstance(response_id, str) and isinstance(response_text, str):
                response_text_by_id[response_id] = response_text
    gemini_parallel_workers = (
        max(1, int(os.getenv("GEMINI_STAGE3_PARALLEL_THREADS", "1")))
        if adapter.spec.provider == "gemini"
        else 1
    )

    def _write_aggregates() -> None:
        if not aggregates_path:
            return
        try:
            rows = list(iter_jsonl(genui_path))
            if response_text_by_id:
                for row in rows:
                    if row.get("response_text"):
                        continue
                    response_id = row.get("response_id")
                    if isinstance(response_id, str):
                        backfill = response_text_by_id.get(response_id)
                        if isinstance(backfill, str):
                            row["response_text"] = backfill
            aggregates = aggregate_metrics(rows)
            aggregates["overall_score"] = compute_overall_score(
                aggregates,
                aggregate_weights or {},
            )
            aggregates_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
            logger.info("Stage3 aggregates stored at %s", aggregates_path)
        except Exception as exc:  # best-effort
            logger.warning("Stage3 aggregates failed: %s", exc)

    total_created = 0
    total_failed = 0
    pending: list[dict[str, Any]] = []
    use_parallel = adapter.spec.provider == "gemini" and gemini_parallel_workers > 1
    use_batch = (
        adapter.spec.provider == "gemini"
        and hasattr(adapter, "generate_batch")
        and not use_parallel
    )
    if batch_size <= 0:
        batch_size = 100
    stop = False

    def _build_prompt_for(response_id: str, response_text: str, assets_list: list[dict]) -> str:
        asset_context = _build_asset_context(assets_list)
        prompt_response_text = response_text
        if asset_context:
            prompt_response_text = f"{response_text}\n\n{asset_context}"

        prompt = render_prompt(user_prompt_template, response_text=prompt_response_text)
        if prompt_max_tokens:
            system_tokens = count_tokens(system_prompt) if system_prompt else 0
            prompt_tokens = count_tokens(prompt) + system_tokens
            if prompt_tokens > prompt_max_tokens:
                # First attempt: drop asset context to save tokens.
                prompt = render_prompt(user_prompt_template, response_text=response_text)
                prompt_tokens = count_tokens(prompt) + system_tokens
            if prompt_tokens > prompt_max_tokens:
                base_prompt = render_prompt(user_prompt_template, response_text="")
                base_tokens = count_tokens(base_prompt) + system_tokens
                budget = max(200, prompt_max_tokens - base_tokens)
                trimmed_text, truncated = _truncate_tokens(response_text, budget)
                if truncated:
                    logger.warning(
                        "Stage3 prompt truncated response_id=%s tokens=%s budget=%s",
                        response_id,
                        count_tokens(response_text),
                        budget,
                    )
                prompt = render_prompt(user_prompt_template, response_text=trimmed_text)
        return prompt

    def _process_generated(
        task: dict[str, Any],
        raw_text: str,
        raw_payload: Any,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
        error: str | None,
    ) -> None:
        nonlocal total_created
        ui_id = task["ui_id"]
        response_id = task["response_id"]
        query_id = task["query_id"]
        response_text = task["response_text"]
        assets_list = task["assets_list"]
        intent_value = task.get("intent")
        tags_value = task.get("tags") if isinstance(task.get("tags"), list) else []
        prompt = task["prompt"]

        parsed_ok = True
        errors: list[str] = []
        try:
            genui_json = extract_json(raw_text)
        except Exception as exc:
            parsed_ok = False
            genui_json = None
            errors.append(f"json_parse_error: {exc}")

        schema_valid_strict = False
        schema_valid_lenient = False
        repair_needed = False

        if parsed_ok and genui_json is not None:
            schema_valid_strict, schema_errors, validator_ok = _validate_schema(
                schema, genui_json, schema_path.parent
            )
            if schema_valid_strict:
                schema_valid_lenient = True
            else:
                errors.extend(schema_errors)
                if not validator_ok:
                    schema_valid_lenient = True

        repair_attempts = 0
        while (not parsed_ok or not schema_valid_strict) and repair_attempts < max_repair_attempts:
            repair_attempts += 1
            repair_needed = True
            repair_prompt = (
                "The previous output was not valid JSON or failed schema validation. "
                "Fix the output to be valid JSON that satisfies the schema. "
                f"Errors: {errors}.\n"
                "Return ONLY the corrected JSON."
            )
            repaired_text = f"{repair_prompt}\n\nOriginal:\n{raw_text}"

            def _repair_call():
                rate_limiter.acquire()
                return adapter.generate(
                    prompt=repaired_text,
                    system=system_prompt,
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
                errors.append(f"repair_exception: {exc}")
                logger.warning(
                    "Stage3 repair failed ui_id=%s response_id=%s: %s",
                    ui_id,
                    response_id,
                    exc,
                )
                break
            if result.error:
                errors.append(f"repair_error: {result.error}")
                logger.warning(
                    "Stage3 repair returned error ui_id=%s response_id=%s: %s",
                    ui_id,
                    response_id,
                    result.error,
                )
                break
            raw_text = result.text
            try:
                genui_json = extract_json(raw_text)
                parsed_ok = True
            except Exception as exc:
                parsed_ok = False
                errors.append(f"repair_json_parse_error: {exc}")
                continue

            schema_valid_strict, schema_errors, validator_ok = _validate_schema(
                schema, genui_json, schema_path.parent
            )
            if schema_valid_strict:
                schema_valid_lenient = True
            else:
                errors.extend(schema_errors)
                if not validator_ok:
                    schema_valid_lenient = True

        if not parsed_ok or genui_json is None or not schema_valid_strict:
            # Final fallback: build a minimal valid GenUICraft message list.
            fallback_text = _apply_asset_replacements(response_text, assets_list)
            surface_id = f"surface_{query_id}"
            catalog_id = "https://genui.local/specification/v0_9/standard_catalog.json"
            genui_json = [
                {
                    "version": "v0.9",
                    "createSurface": {
                        "surfaceId": surface_id,
                        "catalogId": catalog_id,
                    },
                },
                {
                    "version": "v0.9",
                    "updateComponents": {
                        "surfaceId": surface_id,
                        "components": [
                            {
                                "id": "root",
                                "component": "Column",
                                "children": ["text_1"],
                            },
                            {
                                "id": "text_1",
                                "component": "Text",
                                "text": fallback_text,
                                "variant": "body",
                            },
                        ],
                    },
                },
            ]
            # Re-validate schema for fallback.
            parsed_ok = True
            errors = ["fallback_generated"]
            schema_valid_strict, schema_errors, validator_ok = _validate_schema(
                schema, genui_json, schema_path.parent
            )
            if schema_valid_strict:
                schema_valid_lenient = True
            else:
                errors.extend(schema_errors)
                if not validator_ok:
                    schema_valid_lenient = True

        toon = encode_toon(genui_json)
        toon_ok = roundtrip_ok(genui_json, toon)

        json_text = json.dumps(genui_json, ensure_ascii=False)
        metrics = {
            "content_coverage": content_coverage(response_text, genui_json),
            "dup_rate": dup_rate(genui_json),
            "lint_score": lint_score(genui_json),
            # Size proxy: character-count based for stable JSON vs TOON comparison.
            "output_tokens_toon": count_characters(toon),
            "output_tokens_json": count_characters(json_text),
            "output_chars_toon": count_characters(toon),
            "output_chars_json": count_characters(json_text),
        }
        metrics.update(compute_ui_metrics(response_text, genui_json))
        intent_metrics = compute_intent_metrics(intent_value, tags_value, response_text, metrics)
        intent_bucket = intent_metrics.pop("intent_bucket", "unknown")
        metrics.update(intent_metrics)

        short_errors = [e[:300] + ("..." if len(e) > 300 else "") for e in errors]
        record = {
            "ui_id": ui_id,
            "response_id": response_id,
            "query_id": query_id,
            "intent": intent_value,
            "tags": tags_value,
            "intent_bucket": intent_bucket,
            "genui_json": genui_json,
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
                "prompt_version": "genui_gen_v1",
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

        if errors:
            error_path = artifacts_dir / f"error_{ui_id}.json"
            error_payload = {
                "prompt": prompt,
                "raw_text": raw_text,
                "errors": errors,
            }
            error_path.write_text(
                json.dumps(error_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _record_generation_error(task: dict[str, Any], err: str, raw_payload: Any = None) -> None:
        nonlocal total_failed
        total_failed += 1
        logger.error("Stage3 generation error response_id=%s: %s", task.get("response_id"), err)
        error_path = artifacts_dir / f"error_{task.get('ui_id', 'unknown')}.json"
        error_payload = {
            "prompt": task.get("prompt"),
            "error": err,
            "raw_payload": raw_payload,
            "response_id": task.get("response_id"),
            "query_id": task.get("query_id"),
            "ui_id": task.get("ui_id"),
        }
        error_path.write_text(
            json.dumps(error_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _generate_single_result(task: dict[str, Any]):
        def _call():
            rate_limiter.acquire()
            return adapter.generate(
                prompt=task["prompt"],
                system=system_prompt,
                temperature=0.2,
                max_tokens=max_tokens,
                seed=task["seed"],
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
        return result

    def _generate_single(task: dict[str, Any]) -> None:
        try:
            result = _generate_single_result(task)
        except Exception as exc:
            _record_generation_error(task, str(exc))
            return
        if result.error:
            _record_generation_error(task, result.error, result.raw)
            return
        cache.set(task["prompt_hash"], result.text, result.raw)
        _process_generated(
            task,
            result.text,
            result.raw,
            result.latency_ms,
            result.input_tokens,
            result.output_tokens,
            result.provider,
            result.model,
            result.error,
        )

    def _flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        tasks = pending
        pending = []
        results = None

        if use_batch and len(tasks) > 1:
            prompts = [task["prompt"] for task in tasks]
            seeds = [task["seed"] for task in tasks]

            def _call_batch():
                rate_limiter.acquire()
                return adapter.generate_batch(
                    prompts=prompts,
                    system=system_prompt,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    seeds=seeds,
                    json_mode=True if adapter.spec.supports_json_mode else False,
                    batch_name=f"stage3_{int(time.time())}",
                )

            try:
                results = with_retry(_call_batch, max_attempts=max_attempts)
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    logger.error(
                        "Stage3 rate limit info: limits=%s headers=%s",
                        exc.limits or "unset",
                        exc.headers or "none",
                    )
                logger.warning("Stage3 batch failed; falling back to single calls: %s", exc)
                results = None

        if use_parallel and results is None and len(tasks) > 1:
            parallel_results: list[tuple[dict[str, Any], Any]] = []
            parallel_failed = False
            parallel_error: Exception | None = None
            with ThreadPoolExecutor(max_workers=gemini_parallel_workers) as executor:
                future_to_task = {executor.submit(_generate_single_result, task): task for task in tasks}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        parallel_failed = True
                        parallel_error = exc
                        logger.warning(
                            "Stage3 parallel worker failed response_id=%s: %s",
                            task["response_id"],
                            exc,
                        )
                        continue
                    parallel_results.append((task, result))

            if parallel_failed:
                logger.warning(
                    "Stage3 parallel batch degraded to sequential processing due to worker errors: %s",
                    parallel_error,
                )
                for task in tasks:
                    _generate_single(task)
                return

            for task, result in parallel_results:
                if result.error:
                    _record_generation_error(task, result.error, result.raw)
                    continue
                cache.set(task["prompt_hash"], result.text, result.raw)
                _process_generated(
                    task,
                    result.text,
                    result.raw,
                    result.latency_ms,
                    result.input_tokens,
                    result.output_tokens,
                    result.provider,
                    result.model,
                    result.error,
                )
            return

        if results is None:
            for task in tasks:
                _generate_single(task)
            return

        if len(results) != len(tasks):
            logger.warning(
                "Stage3 batch size mismatch: expected=%s got=%s",
                len(tasks),
                len(results),
            )

        for task, result in zip_longest(tasks, results):
            if result is None:
                raise RuntimeError("Stage3 batch missing result")
            if result.error:
                _record_generation_error(task, result.error, result.raw)
                continue
            cache.set(task["prompt_hash"], result.text, result.raw)
            _process_generated(
                task,
                result.text,
                result.raw,
                result.latency_ms,
                result.input_tokens,
                result.output_tokens,
                result.provider,
                result.model,
                result.error,
            )

    try:
        for response in iter_jsonl(responses_path):
            if stop:
                break
            response_id = response.get("response_id")
            query_id = response.get("query_id")
            response_text = response.get("response_text")
            assets = response.get("assets") if isinstance(response, dict) else None
            assets_list = assets if isinstance(assets, list) else []
            n_idx = int(response.get("n_idx", 1))
            if not response_id or not query_id or not response_text:
                continue

            for c_idx in range(1, candidates_per_response + 1):
                if stop:
                    break
                if max_total is not None:
                    remaining = max_total - total_created
                    if remaining <= 0:
                        stop = True
                        break
                    if len(pending) >= remaining:
                        stop = True
                        break

                ui_id = _make_ui_id(query_id, n_idx, c_idx)
                if ui_id in existing_ids:
                    continue

                prompt = _build_prompt_for(response_id, response_text, assets_list)
                prompt_hash = hash_text(
                    f"{adapter.spec.name}:{system_prompt or ''}\n---\n{prompt}"
                )
                intent_info = intent_lookup.get(query_id, {})
                task = {
                    "ui_id": ui_id,
                    "response_id": response_id,
                    "query_id": query_id,
                    "response_text": response_text,
                    "assets_list": assets_list,
                    "intent": intent_info.get("intent"),
                    "tags": intent_info.get("tags"),
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "seed": seed + c_idx,
                }

                cached = cache.get(prompt_hash)
                if cached:
                    _process_generated(
                        task,
                        cached.text,
                        cached.raw,
                        0.0,
                        0,
                        0,
                        adapter.spec.provider,
                        adapter.spec.model,
                        None,
                    )
                    continue

                if not use_batch:
                    if use_parallel:
                        pending.append(task)
                        if len(pending) >= gemini_parallel_workers:
                            _flush_pending()
                    else:
                        _generate_single(task)
                    continue

                pending.append(task)
                if len(pending) >= batch_size:
                    _flush_pending()

        _flush_pending()
        if total_failed > 0:
            logger.warning("Stage3 completed with generation failures=%s created=%s", total_failed, total_created)
        else:
            logger.info("Stage3 completed created=%s", total_created)
    finally:
        _write_aggregates()



