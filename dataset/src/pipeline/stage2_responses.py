from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pipeline.common import extract_json, load_prompt, render_prompt
from pipeline.storage import JsonlWriter, iter_jsonl
from pipeline.cache import PromptCache
from llm.base import BaseLLMAdapter, LLMRateLimitError
from utils.hashing import normalize_text, hash_text
from utils.rate_limit import RateLimiter
from utils.retry import with_retry


def _make_response_id(query_id: str, n_idx: int) -> str:
    suffix = query_id.replace("q_", "")
    return f"r_{suffix}_{n_idx:02d}"


def run_stage2(
    queries_path: Path,
    prompt_path: Path,
    batch_prompt_path: Path | None,
    adapter: BaseLLMAdapter,
    responses_path: Path,
    n_per_query: int,
    batch_size: int,
    query_batch_size: int,
    group_by_intent: bool,
    batch_fallback_per_query: bool,
    temperatures: list[float],
    max_tokens: int,
    seed: int,
    rate_limiter: RateLimiter,
    cache: PromptCache,
    logger,
    max_total: int | None = None,
    max_attempts: int = 3,
) -> None:
    prompt_template = load_prompt(prompt_path)
    batch_prompt_template = load_prompt(batch_prompt_path) if batch_prompt_path else ""

    existing_ids = {row.get("response_id") for row in iter_jsonl(responses_path)}
    existing_hashes = set()
    for row in iter_jsonl(responses_path):
        text = row.get("response_text", "")
        if text:
            existing_hashes.add(hash_text(normalize_text(text)))

    writer = JsonlWriter(responses_path)

    total_created = 0

    if query_batch_size > 1:
        queries = []
        for q in iter_jsonl(queries_path):
            if not isinstance(q, dict):
                continue
            if not q.get("query_id") or not q.get("query_text"):
                logger.warning("Stage2 skipping query without id/text: %s", q)
                continue
            queries.append(q)
        if group_by_intent:
            grouped: dict[str, list[dict]] = {}
            order: list[str] = []
            for q in queries:
                intent = q.get("intent", "unknown")
                if intent not in grouped:
                    grouped[intent] = []
                    order.append(intent)
                grouped[intent].append(q)
            intent_order = order
        else:
            grouped = {"all": queries}
            intent_order = ["all"]

        for intent in intent_order:
            rows = grouped[intent]
            idx = 0
            while idx < len(rows):
                if max_total is not None and total_created >= max_total:
                    logger.info("Stage2 reached max_total=%s", max_total)
                    return
                batch = rows[idx: idx + query_batch_size]
                idx += query_batch_size
                # filter already-completed
                batch = [
                    q for q in batch
                    if _make_response_id(q.get("query_id", ""), 1) not in existing_ids
                ]
                if not batch:
                    continue
                created = _process_query_batch(
                    batch,
                    batch_prompt_template,
                    prompt_template,
                    adapter,
                    writer,
                    existing_ids,
                    existing_hashes,
                    max_tokens,
                    seed,
                    rate_limiter,
                    cache,
                    logger,
                    max_attempts,
                    batch_fallback_per_query,
                    temperatures[0] if temperatures else 0.7,
                )
                total_created += created
                if max_total is not None and total_created >= max_total:
                    logger.info("Stage2 reached max_total=%s", max_total)
                    return
        return

    for query in iter_jsonl(queries_path):
        query_id = query.get("query_id")
        query_text = query.get("query_text")
        if not query_id or not query_text:
            continue
        remaining = n_per_query
        n_idx = 1
        while remaining > 0:
            if max_total is not None and total_created >= max_total:
                logger.info("Stage2 reached max_total=%s", max_total)
                return
            batch = max(1, min(batch_size, remaining))
            response_id = _make_response_id(query_id, n_idx)
            if response_id in existing_ids:
                n_idx += 1
                remaining -= 1
                continue

            temperature = temperatures[(n_idx - 1) % len(temperatures)] if temperatures else 0.7
            prompt = render_prompt(prompt_template, query_text=query_text)
            if batch > 1:
                prompt = (
                    f"{prompt}\n\nReturn exactly {batch} distinct responses as a JSON array of strings."
                )
            prompt_hash = hash_text(f"{adapter.spec.name}:{prompt}")

            cached = cache.get(prompt_hash)
            if cached:
                raw_text = cached.text.strip()
                latency_ms = 0.0
                input_tokens = 0
                output_tokens = 0
                provider = adapter.spec.provider
                model = adapter.spec.model
            else:
                def _call():
                    rate_limiter.acquire()
                    return adapter.generate(
                        prompt=prompt,
                        system=None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed + n_idx,
                        json_mode=batch > 1 and adapter.spec.supports_json_mode,
                    )

                try:
                    result = with_retry(_call, max_attempts=max_attempts)
                except Exception as exc:
                    if isinstance(exc, LLMRateLimitError):
                        logger.error(
                            "Stage2 rate limit info: limits=%s headers=%s",
                            exc.limits or "unset",
                            exc.headers or "none",
                        )
                    raise
                if result.error:
                    logger.error("Stage2 error query_id=%s: %s", query_id, result.error)
                    raise RuntimeError(result.error)
                raw_text = result.text.strip()
                latency_ms = result.latency_ms
                input_tokens = result.input_tokens
                output_tokens = result.output_tokens
                provider = result.provider
                model = result.model
                cache.set(prompt_hash, result.text, result.raw)

            responses: list[str]
            if batch > 1:
                try:
                    payload = extract_json(raw_text)
                    if isinstance(payload, list):
                        responses = [str(item).strip() for item in payload if str(item).strip()]
                    else:
                        responses = [raw_text]
                except Exception:
                    responses = [raw_text]
            else:
                responses = [raw_text]

            for response_text in responses[:batch]:
                if not response_text:
                    continue
                norm_hash = hash_text(normalize_text(response_text))
                if norm_hash in existing_hashes:
                    logger.info("Stage2 duplicate response query_id=%s", query_id)
                    continue

                record = {
                    "response_id": response_id,
                    "query_id": query_id,
                    "n_idx": n_idx,
                    "response_text": response_text,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "gen": {
                        "provider": provider,
                        "model": model,
                        "latency_ms": latency_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": None,
                    },
                }
                writer.append(record)
                existing_ids.add(response_id)
                existing_hashes.add(norm_hash)
                logger.info("Stage2 created response_id=%s", response_id)
                n_idx += 1
                remaining -= 1
                total_created += 1
                if max_total is not None and total_created >= max_total:
                    logger.info("Stage2 reached max_total=%s", max_total)
                    return
                response_id = _make_response_id(query_id, n_idx)

            if batch == 1 and not responses:
                n_idx += 1
                remaining -= 1


def _process_query_batch(
    queries: list[dict],
    batch_prompt_template: str,
    single_prompt_template: str,
    adapter: BaseLLMAdapter,
    writer: JsonlWriter,
    existing_ids: set[str],
    existing_hashes: set[str],
    max_tokens: int,
    seed: int,
    rate_limiter: RateLimiter,
    cache: PromptCache,
    logger,
    max_attempts: int,
    fallback_per_query: bool,
    temperature: float,
) -> int:
    if not queries:
        return 0
    payload = []
    for q in queries:
        qid = q.get("query_id")
        qtext = q.get("query_text")
        if not qid or not qtext:
            logger.warning("Stage2 batch skip missing query_id/query_text: %s", q)
            continue
        payload.append({"query_id": qid, "query_text": qtext})
    if not payload:
        return 0
    queries_json = json.dumps(payload, ensure_ascii=False)
    prompt = batch_prompt_template or (
        "Return JSON array of {query_id, response_text} for these queries:\n" + queries_json
    )
    prompt = render_prompt(prompt, queries_json=queries_json)
    prompt_hash = hash_text(f"{adapter.spec.name}:{prompt}")

    cached = cache.get(prompt_hash)
    if cached:
        raw_text = cached.text.strip()
        latency_ms = 0.0
        input_tokens = 0
        output_tokens = 0
        provider = adapter.spec.provider
        model = adapter.spec.model
    else:
        def _call():
            rate_limiter.acquire()
            return adapter.generate(
                prompt=prompt,
                system=None,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                json_mode=adapter.spec.supports_json_mode,
            )

        try:
            result = with_retry(_call, max_attempts=max_attempts)
        except Exception as exc:
            # If rate limited, split the batch to reduce request size.
            if isinstance(exc, LLMRateLimitError):
                logger.error(
                    "Stage2 rate limit info: limits=%s headers=%s",
                    exc.limits or "unset",
                    exc.headers or "none",
                )
                if len(queries) > 1:
                    mid = len(queries) // 2
                    return _process_query_batch(
                        queries[:mid],
                        batch_prompt_template,
                        single_prompt_template,
                        adapter,
                        writer,
                        existing_ids,
                        existing_hashes,
                        max_tokens,
                        seed,
                        rate_limiter,
                        cache,
                        logger,
                        max_attempts,
                        fallback_per_query,
                        temperature,
                    ) + _process_query_batch(
                        queries[mid:],
                        batch_prompt_template,
                        single_prompt_template,
                        adapter,
                        writer,
                        existing_ids,
                        existing_hashes,
                        max_tokens,
                        seed,
                        rate_limiter,
                        cache,
                        logger,
                        max_attempts,
                        fallback_per_query,
                        temperature,
                    )
            if "429" in str(exc) and len(queries) > 1:
                mid = len(queries) // 2
                return _process_query_batch(
                    queries[:mid],
                    batch_prompt_template,
                    single_prompt_template,
                    adapter,
                    writer,
                    existing_ids,
                    existing_hashes,
                    max_tokens,
                    seed,
                    rate_limiter,
                    cache,
                    logger,
                    max_attempts,
                    fallback_per_query,
                    temperature,
                ) + _process_query_batch(
                    queries[mid:],
                    batch_prompt_template,
                    single_prompt_template,
                    adapter,
                    writer,
                    existing_ids,
                    existing_hashes,
                    max_tokens,
                    seed,
                    rate_limiter,
                    cache,
                    logger,
                    max_attempts,
                    fallback_per_query,
                    temperature,
                )
            logger.error("Stage2 batch error: %s", exc)
            return 0
        if result.error:
            logger.error("Stage2 batch error: %s", result.error)
            return 0
        raw_text = result.text.strip()
        latency_ms = result.latency_ms
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        provider = result.provider
        model = result.model
        cache.set(prompt_hash, result.text, result.raw)

    try:
        parsed = extract_json(raw_text)
    except Exception as exc:
        logger.error("Stage2 batch parse error: %s", exc)
        parsed = []
        # Fallback: attempt to recover list of objects from raw text
        if raw_text:
            recovered = []
            depth = 0
            start = None
            in_string = False
            escape = False
            for i, ch in enumerate(raw_text):
                if ch == "\\" and in_string:
                    escape = not escape
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                escape = False if ch != "\\" else escape
                if in_string:
                    continue
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        chunk = raw_text[start : i + 1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                recovered.append(obj)
                        except Exception:
                            pass
                        start = None
            if recovered:
                parsed = recovered

    responses_by_id = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            qid = item.get("query_id")
            text = item.get("response_text")
            if qid and text:
                responses_by_id[str(qid)] = str(text).strip()

    created = 0
    for q in queries:
        qid = q.get("query_id")
        qtext = q.get("query_text")
        if not qid or not qtext:
            logger.warning("Stage2 batch skip missing query_id/query_text: %s", q)
            continue
        response_text = responses_by_id.get(qid, "").strip()
        if not response_text and fallback_per_query:
            single_prompt = render_prompt(single_prompt_template, query_text=qtext)
            single_hash = hash_text(f"{adapter.spec.name}:{single_prompt}")
            cached_single = cache.get(single_hash)
            if cached_single:
                response_text = cached_single.text.strip()
            else:
                def _single_call():
                    rate_limiter.acquire()
                    return adapter.generate(
                        prompt=single_prompt,
                        system=None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed,
                        json_mode=False,
                    )

                try:
                    single_result = with_retry(_single_call, max_attempts=max_attempts)
                except Exception as exc:
                    if isinstance(exc, LLMRateLimitError):
                        logger.error(
                            "Stage2 rate limit info: limits=%s headers=%s",
                            exc.limits or "unset",
                            exc.headers or "none",
                        )
                    logger.error("Stage2 fallback error query_id=%s: %s", qid, exc)
                    continue
                if single_result.error:
                    logger.error("Stage2 fallback error query_id=%s: %s", qid, single_result.error)
                    continue
                response_text = single_result.text.strip()
                cache.set(single_hash, single_result.text, single_result.raw)

        if not response_text:
            continue
        response_id = _make_response_id(qid, 1)
        if response_id in existing_ids:
            continue
        norm_hash = hash_text(normalize_text(response_text))
        if norm_hash in existing_hashes:
            continue
        record = {
            "response_id": response_id,
            "query_id": qid,
            "n_idx": 1,
            "response_text": response_text,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "gen": {
                "provider": provider,
                "model": model,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": None,
            },
        }
        writer.append(record)
        existing_ids.add(response_id)
        existing_hashes.add(norm_hash)
        logger.info("Stage2 created response_id=%s", response_id)
        created += 1
    return created
