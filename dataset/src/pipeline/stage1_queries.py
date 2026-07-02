from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from pipeline.common import extract_json, load_prompt, render_prompt
from pipeline.storage import JsonlWriter, iter_jsonl
from pipeline.cache import PromptCache
from llm.base import BaseLLMAdapter, LLMRateLimitError
from utils.hashing import normalize_text, hash_text, stable_id
from utils.rate_limit import RateLimiter
from utils.retry import with_retry


def _load_intents(path: Path) -> list[str]:
    intents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        intents.append(cleaned)
    return intents


def _extract_objects_fallback(text: str) -> list[dict]:
    objects: list[dict] = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
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
                chunk = text[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except Exception:
                    pass
                start = None
    return objects


def run_stage1(
    intents_file: Path,
    prompt_path: Path,
    adapter: BaseLLMAdapter,
    run_dir: Path,
    queries_path: Path,
    k_per_intent: int,
    batch_size: int,
    seed: int,
    temperature: float,
    max_tokens: int,
    rate_limiter: RateLimiter,
    cache: PromptCache,
    logger,
    intent_batch_size: int = 1,
    max_total: int | None = None,
    max_failures_per_intent: int = 3,
    fill_missing_with_fallback: bool = True,
    max_attempts: int = 3,
    intent_cycle_size: int = 0,
) -> None:
    intents = _load_intents(intents_file)
    prompt_template = load_prompt(prompt_path)

    existing_hashes = set()
    existing_counts = {intent: 0 for intent in intents}
    existing_ids = set()
    for row in iter_jsonl(queries_path):
        existing_ids.add(row.get("query_id"))
        q = row.get("query_text", "")
        if q:
            existing_hashes.add(hash_text(normalize_text(q)))
        intent = row.get("intent")
        if intent in existing_counts:
            existing_counts[intent] += 1

    writer = JsonlWriter(queries_path)
    next_idx = len(existing_ids) + 1

    total_created = 0
    if intent_batch_size <= 0:
        intent_batch_size = 1
    if intent_cycle_size < 0:
        intent_cycle_size = 0
    use_adapter_batch = adapter.spec.provider in {"gemini", "local"} and hasattr(adapter, "generate_batch")
    if use_adapter_batch and intent_batch_size > 1 and intent_cycle_size <= 0:
        failures_by_intent = {intent: 0 for intent in intents}
        stopped_intents: set[str] = set()
        stop_all = False

        def _append_record(
            intent: str,
            query_text: str,
            difficulty: str,
            tags: list,
            source: str,
            provider: str,
            model: str,
            seed_value: int,
        ) -> bool:
            nonlocal next_idx, total_created, stop_all
            if max_total is not None and total_created >= max_total:
                stop_all = True
                return False
            norm_hash = hash_text(normalize_text(query_text))
            if norm_hash in existing_hashes:
                return False
            query_id = stable_id("q", next_idx)
            next_idx += 1
            record = {
                "query_id": query_id,
                "intent": intent,
                "query_text": query_text,
                "difficulty": difficulty,
                "tags": tags,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "source": source,
                "gen": {
                    "llm_provider": provider,
                    "model": model,
                    "prompt_version": "query_gen_v1",
                    "temperature": temperature,
                    "seed": seed_value,
                },
            }
            writer.append(record)
            existing_hashes.add(norm_hash)
            existing_counts[intent] += 1
            total_created += 1
            if max_total is not None and total_created >= max_total:
                stop_all = True
            return True

        def _fill_fallback(intent: str, provider: str, model: str) -> None:
            if intent in stopped_intents:
                return
            logger.error(
                "Stage1 fallback intent=%s after %s failures",
                intent,
                failures_by_intent[intent],
            )
            while existing_counts[intent] < k_per_intent:
                if stop_all:
                    break
                suffix = existing_counts[intent] + 1
                fallback_text = f"{intent} request {suffix}"
                created = _append_record(
                    intent,
                    fallback_text,
                    "easy",
                    [intent.lower().replace(" ", "_")],
                    "fallback",
                    provider,
                    model,
                    seed,
                )
                if not created:
                    existing_counts[intent] += 1
            stopped_intents.add(intent)

        def _handle_failure(intent: str, provider: str, model: str) -> None:
            failures_by_intent[intent] += 1
            if failures_by_intent[intent] >= max_failures_per_intent:
                if fill_missing_with_fallback:
                    _fill_fallback(intent, provider, model)
                else:
                    logger.error(
                        "Stage1 stopping intent=%s after %s failures",
                        intent,
                        failures_by_intent[intent],
                    )
                    stopped_intents.add(intent)

        def _process_raw(
            intent: str,
            raw_text: str,
            provider: str,
            model: str,
            seed_value: int,
        ) -> tuple[int, bool]:
            try:
                payload = extract_json(raw_text)
            except Exception as exc:
                logger.error("Stage1 parse error intent=%s: %s", intent, exc)
                payload = _extract_objects_fallback(raw_text)
                if payload:
                    logger.info(
                        "Stage1 recovered %s objects from fallback parse intent=%s",
                        len(payload),
                        intent,
                    )
                else:
                    return 0, False

            if not isinstance(payload, list) or not payload:
                logger.error("Stage1 unexpected or empty payload intent=%s", intent)
                return 0, False

            created = 0
            for item in payload:
                if not isinstance(item, dict):
                    continue
                query_text = str(item.get("query_text", "")).strip()
                if not query_text:
                    continue
                if _append_record(
                    intent,
                    query_text,
                    item.get("difficulty", "medium"),
                    item.get("tags", []),
                    "generated",
                    provider,
                    model,
                    seed_value,
                ):
                    created += 1
                if stop_all or existing_counts[intent] >= k_per_intent:
                    break

            if created == 0:
                logger.warning(
                    "Stage1 no new queries intent=%s failures=%s/%s",
                    intent,
                    failures_by_intent[intent] + 1,
                    max_failures_per_intent,
                )
                return 0, False

            logger.info("Stage1 intent=%s created=%d total=%d", intent, created, existing_counts[intent])
            return created, True

        while True:
            if max_total is not None and total_created >= max_total:
                logger.info("Stage1 reached max_total=%s", max_total)
                return
            pending_intents = [
                intent
                for intent in intents
                if intent not in stopped_intents and existing_counts[intent] < k_per_intent
            ]
            if not pending_intents:
                return

            batch_intents = pending_intents[:intent_batch_size]
            entries: list[dict[str, object]] = []
            for intent in batch_intents:
                remaining = k_per_intent - existing_counts[intent]
                k = min(batch_size, remaining)
                prompt = render_prompt(prompt_template, intent=intent, k=k)
                seed_value = seed + existing_counts[intent] + failures_by_intent[intent]
                prompt_hash = hash_text(f"{adapter.spec.name}:{prompt}:seed={seed_value}")
                cached = cache.get(prompt_hash)
                if cached:
                    created, ok = _process_raw(
                        intent,
                        cached.text,
                        adapter.spec.provider,
                        adapter.spec.model,
                        seed_value,
                    )
                    if not ok:
                        _handle_failure(intent, adapter.spec.provider, adapter.spec.model)
                    elif fill_missing_with_fallback and existing_counts[intent] < k_per_intent:
                        _fill_fallback(intent, adapter.spec.provider, adapter.spec.model)
                    if stop_all:
                        return
                    continue
                entries.append(
                    {
                        "intent": intent,
                        "prompt": prompt,
                        "seed_value": seed_value,
                        "prompt_hash": prompt_hash,
                    }
                )

            if not entries:
                continue

            prompts = [entry["prompt"] for entry in entries]
            seeds = [entry["seed_value"] for entry in entries]
            results = None

            def _call_batch():
                rate_limiter.acquire()
                return adapter.generate_batch(
                    prompts=prompts,
                    system=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seeds=seeds,
                    json_mode=True if adapter.spec.supports_json_mode else False,
                    batch_name=f"stage1_{int(time.time())}",
                )

            try:
                results = with_retry(_call_batch, max_attempts=max_attempts)
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    logger.error(
                        "Stage1 rate limit info: limits=%s headers=%s",
                        exc.limits or "unset",
                        exc.headers or "none",
                    )
                logger.warning("Stage1 batch failed; falling back to single calls: %s", exc)
                results = None

            if results is None or len(results) != len(entries):
                for entry in entries:
                    prompt = entry["prompt"]
                    intent = entry["intent"]
                    seed_value = entry["seed_value"]

                    def _call():
                        rate_limiter.acquire()
                        return adapter.generate(
                            prompt=prompt,
                            system=None,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            seed=seed_value,
                            json_mode=True if adapter.spec.supports_json_mode else False,
                        )

                    try:
                        result = with_retry(_call, max_attempts=max_attempts)
                    except Exception as exc:
                        if isinstance(exc, LLMRateLimitError):
                            logger.error(
                                "Stage1 rate limit info: limits=%s headers=%s",
                                exc.limits or "unset",
                                exc.headers or "none",
                            )
                        logger.error("Stage1 error intent=%s: %s", intent, exc)
                        _handle_failure(intent, adapter.spec.provider, adapter.spec.model)
                        if stop_all:
                            return
                        continue
                    if result.error:
                        logger.error("Stage1 error intent=%s: %s", intent, result.error)
                        _handle_failure(intent, result.provider, result.model)
                        if stop_all:
                            return
                        continue
                    cache.set(entry["prompt_hash"], result.text, result.raw)
                    created, ok = _process_raw(intent, result.text, result.provider, result.model, seed_value)
                    if not ok:
                        _handle_failure(intent, result.provider, result.model)
                    elif fill_missing_with_fallback and existing_counts[intent] < k_per_intent:
                        _fill_fallback(intent, result.provider, result.model)
                    if stop_all:
                        return
                continue

            for entry, result in zip(entries, results):
                intent = entry["intent"]
                seed_value = entry["seed_value"]
                if result.error:
                    logger.error("Stage1 error intent=%s: %s", intent, result.error)
                    _handle_failure(intent, result.provider, result.model)
                    if stop_all:
                        return
                    continue
                cache.set(entry["prompt_hash"], result.text, result.raw)
                created, ok = _process_raw(intent, result.text, result.provider, result.model, seed_value)
                if not ok:
                    _handle_failure(intent, result.provider, result.model)
                elif fill_missing_with_fallback and existing_counts[intent] < k_per_intent:
                    _fill_fallback(intent, result.provider, result.model)
                if stop_all:
                    return
        return

    if intent_cycle_size > 0:
        failures_by_intent = {intent: 0 for intent in intents}
        stopped_intents: set[str] = set()

        def _append_cyclic_record(
            intent: str,
            query_text: str,
            difficulty: str,
            tags: list,
            source: str,
            provider: str,
            model: str,
            seed_value: int,
        ) -> bool:
            nonlocal next_idx, total_created
            if max_total is not None and total_created >= max_total:
                return False
            norm_hash = hash_text(normalize_text(query_text))
            if norm_hash in existing_hashes:
                return False
            query_id = stable_id("q", next_idx)
            next_idx += 1
            writer.append(
                {
                    "query_id": query_id,
                    "intent": intent,
                    "query_text": query_text,
                    "difficulty": difficulty,
                    "tags": tags,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "source": source,
                    "gen": {
                        "llm_provider": provider,
                        "model": model,
                        "prompt_version": "query_gen_v1",
                        "temperature": temperature,
                        "seed": seed_value,
                    },
                }
            )
            existing_hashes.add(norm_hash)
            existing_counts[intent] += 1
            total_created += 1
            return True

        def _fill_cyclic_fallback(
            intent: str,
            target: int,
            provider: str,
            model: str,
            seed_value: int,
        ) -> None:
            logger.error(
                "Stage1 fallback intent=%s after %s failures target=%s",
                intent,
                failures_by_intent[intent],
                target,
            )
            while existing_counts[intent] < target:
                if max_total is not None and total_created >= max_total:
                    return
                suffix = existing_counts[intent] + 1
                fallback_text = f"{intent} request {suffix}"
                created = _append_cyclic_record(
                    intent,
                    fallback_text,
                    "easy",
                    [intent.lower().replace(" ", "_")],
                    "fallback",
                    provider,
                    model,
                    seed_value,
                )
                if not created:
                    existing_counts[intent] += 1

        def _handle_cyclic_failure(
            intent: str,
            target: int,
            provider: str,
            model: str,
            seed_value: int,
        ) -> None:
            failures_by_intent[intent] += 1
            if failures_by_intent[intent] < max_failures_per_intent:
                return
            if fill_missing_with_fallback:
                _fill_cyclic_fallback(intent, target, provider, model, seed_value)
            else:
                logger.error(
                    "Stage1 stopping intent=%s after %s failures",
                    intent,
                    failures_by_intent[intent],
                )
                stopped_intents.add(intent)

        def _process_cyclic_intent(intent: str, target: int) -> bool:
            before_count = existing_counts[intent]
            while existing_counts[intent] < target:
                if max_total is not None and total_created >= max_total:
                    return existing_counts[intent] > before_count

                remaining = target - existing_counts[intent]
                k = min(batch_size, remaining, intent_cycle_size)
                prompt = render_prompt(prompt_template, intent=intent, k=k)
                seed_value = seed + existing_counts[intent] + failures_by_intent[intent]
                prompt_hash = hash_text(f"{adapter.spec.name}:{prompt}:seed={seed_value}")

                cached = cache.get(prompt_hash)
                if cached:
                    raw_text = cached.text
                    result_provider = adapter.spec.provider
                    result_model = adapter.spec.model
                else:
                    def _call():
                        rate_limiter.acquire()
                        return adapter.generate(
                            prompt=prompt,
                            system=None,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            seed=seed_value,
                            json_mode=True if adapter.spec.supports_json_mode else False,
                        )

                    try:
                        result = with_retry(_call, max_attempts=max_attempts)
                    except Exception as exc:
                        if isinstance(exc, LLMRateLimitError):
                            logger.error(
                                "Stage1 rate limit info: limits=%s headers=%s",
                                exc.limits or "unset",
                                exc.headers or "none",
                            )
                        logger.error("Stage1 error intent=%s: %s", intent, exc)
                        _handle_cyclic_failure(
                            intent,
                            target,
                            adapter.spec.provider,
                            adapter.spec.model,
                            seed_value,
                        )
                        break
                    if result.error:
                        logger.error("Stage1 error intent=%s: %s", intent, result.error)
                        _handle_cyclic_failure(intent, target, result.provider, result.model, seed_value)
                        break
                    raw_text = result.text
                    result_provider = result.provider
                    result_model = result.model
                    cache.set(prompt_hash, raw_text, result.raw)

                try:
                    payload = extract_json(raw_text)
                except Exception as exc:
                    logger.error("Stage1 parse error intent=%s: %s", intent, exc)
                    payload = _extract_objects_fallback(raw_text)
                    if payload:
                        logger.info(
                            "Stage1 recovered %s objects from fallback parse intent=%s",
                            len(payload),
                            intent,
                        )
                    else:
                        _handle_cyclic_failure(
                            intent,
                            target,
                            result_provider,
                            result_model,
                            seed_value,
                        )
                        break

                if not isinstance(payload, list) or not payload:
                    logger.error("Stage1 unexpected or empty payload intent=%s", intent)
                    _handle_cyclic_failure(intent, target, result_provider, result_model, seed_value)
                    break

                created = 0
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    query_text = str(item.get("query_text", "")).strip()
                    if not query_text:
                        continue
                    if _append_cyclic_record(
                        intent,
                        query_text,
                        item.get("difficulty", "medium"),
                        item.get("tags", []),
                        "generated",
                        result_provider,
                        result_model,
                        seed_value,
                    ):
                        created += 1
                    if existing_counts[intent] >= target:
                        break
                    if max_total is not None and total_created >= max_total:
                        break

                if created == 0:
                    logger.warning(
                        "Stage1 no new queries intent=%s failures=%s/%s",
                        intent,
                        failures_by_intent[intent] + 1,
                        max_failures_per_intent,
                    )
                    _handle_cyclic_failure(intent, target, result_provider, result_model, seed_value)
                    break

                logger.info(
                    "Stage1 cyclic intent=%s created=%d total=%d target=%d",
                    intent,
                    created,
                    existing_counts[intent],
                    target,
                )

                if fill_missing_with_fallback and existing_counts[intent] < target:
                    _fill_cyclic_fallback(intent, target, result_provider, result_model, seed_value)

            return existing_counts[intent] > before_count

        logger.info("Stage1 intent cycling enabled max_per_intent=%s", intent_cycle_size)
        while True:
            if max_total is not None and total_created >= max_total:
                logger.info("Stage1 reached max_total=%s", max_total)
                return
            active_intents = [
                intent
                for intent in intents
                if intent not in stopped_intents and existing_counts[intent] < k_per_intent
            ]
            if not active_intents:
                return
            active_intents.sort(key=lambda intent: (existing_counts[intent], intents.index(intent)))
            progressed = False
            for intent in active_intents:
                if max_total is not None and total_created >= max_total:
                    logger.info("Stage1 reached max_total=%s", max_total)
                    return
                if intent in stopped_intents or existing_counts[intent] >= k_per_intent:
                    continue
                target = min(k_per_intent, existing_counts[intent] + intent_cycle_size)
                if _process_cyclic_intent(intent, target):
                    progressed = True
            if not progressed:
                logger.error("Stage1 intent cycling made no progress; stopping.")
                return

    for intent in intents:
        target = k_per_intent
        failures = 0
        while existing_counts[intent] < target:
            if max_total is not None and total_created >= max_total:
                logger.info("Stage1 reached max_total=%s", max_total)
                return
            remaining = target - existing_counts[intent]
            k = min(batch_size, remaining)
            prompt = render_prompt(prompt_template, intent=intent, k=k)
            seed_value = seed + existing_counts[intent] + failures
            prompt_hash = hash_text(f"{adapter.spec.name}:{prompt}:seed={seed_value}")

            cached = cache.get(prompt_hash)
            if cached:
                raw_text = cached.text
                result_provider = adapter.spec.provider
                result_model = adapter.spec.model
            else:
                def _call():
                    rate_limiter.acquire()
                    return adapter.generate(
                        prompt=prompt,
                        system=None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed_value,
                        json_mode=True if adapter.spec.supports_json_mode else False,
                    )

                try:
                    result = with_retry(_call, max_attempts=max_attempts)
                except Exception as exc:
                    if isinstance(exc, LLMRateLimitError):
                        logger.error(
                            "Stage1 rate limit info: limits=%s headers=%s",
                            exc.limits or "unset",
                            exc.headers or "none",
                        )
                    logger.error("Stage1 error intent=%s: %s", intent, exc)
                    failures += 1
                    if failures >= max_failures_per_intent:
                        if fill_missing_with_fallback:
                            logger.error("Stage1 fallback intent=%s after %s failures", intent, failures)
                            while existing_counts[intent] < target:
                                if max_total is not None and total_created >= max_total:
                                    logger.info("Stage1 reached max_total=%s", max_total)
                                    return
                                suffix = existing_counts[intent] + 1
                                fallback_text = f"{intent} request {suffix}"
                                norm_hash = hash_text(normalize_text(fallback_text))
                                if norm_hash not in existing_hashes:
                                    query_id = stable_id("q", next_idx)
                                    next_idx += 1
                                    record = {
                                        "query_id": query_id,
                                        "intent": intent,
                                        "query_text": fallback_text,
                                        "difficulty": "easy",
                                        "tags": [intent.lower().replace(" ", "_")],
                                        "created_at": datetime.utcnow().isoformat() + "Z",
                                        "source": "fallback",
                                        "gen": {
                                            "llm_provider": adapter.spec.provider,
                                            "model": adapter.spec.model,
                                            "prompt_version": "query_gen_v1",
                                            "temperature": temperature,
                                            "seed": seed,
                                        },
                                    }
                                    writer.append(record)
                                    existing_hashes.add(norm_hash)
                                    existing_counts[intent] += 1
                                    total_created += 1
                                else:
                                    existing_counts[intent] += 1
                            break
                        logger.error("Stage1 stopping intent=%s after %s failures", intent, failures)
                        break
                    continue
                if result.error:
                    logger.error("Stage1 error intent=%s: %s", intent, result.error)
                    failures += 1
                    if failures >= max_failures_per_intent:
                        if fill_missing_with_fallback:
                            logger.error("Stage1 fallback intent=%s after %s failures", intent, failures)
                            while existing_counts[intent] < target:
                                if max_total is not None and total_created >= max_total:
                                    logger.info("Stage1 reached max_total=%s", max_total)
                                    return
                                suffix = existing_counts[intent] + 1
                                fallback_text = f"{intent} request {suffix}"
                                norm_hash = hash_text(normalize_text(fallback_text))
                                if norm_hash not in existing_hashes:
                                    query_id = stable_id("q", next_idx)
                                    next_idx += 1
                                    record = {
                                        "query_id": query_id,
                                        "intent": intent,
                                        "query_text": fallback_text,
                                        "difficulty": "easy",
                                        "tags": [intent.lower().replace(" ", "_")],
                                        "created_at": datetime.utcnow().isoformat() + "Z",
                                        "source": "fallback",
                                        "gen": {
                                            "llm_provider": adapter.spec.provider,
                                            "model": adapter.spec.model,
                                            "prompt_version": "query_gen_v1",
                                            "temperature": temperature,
                                            "seed": seed,
                                        },
                                    }
                                    writer.append(record)
                                    existing_hashes.add(norm_hash)
                                    existing_counts[intent] += 1
                                    total_created += 1
                                else:
                                    existing_counts[intent] += 1
                            break
                        logger.error("Stage1 stopping intent=%s after %s failures", intent, failures)
                        break
                    continue
                raw_text = result.text
                result_provider = result.provider
                result_model = result.model
                cache.set(prompt_hash, raw_text, result.raw)

            try:
                payload = extract_json(raw_text)
            except Exception as exc:
                logger.error("Stage1 parse error intent=%s: %s", intent, exc)
                payload = _extract_objects_fallback(raw_text)
                if payload:
                    logger.info("Stage1 recovered %s objects from fallback parse intent=%s", len(payload), intent)
                else:
                    failures += 1
                    if failures >= max_failures_per_intent:
                        if fill_missing_with_fallback:
                            logger.error("Stage1 fallback intent=%s after %s failures", intent, failures)
                            while existing_counts[intent] < target:
                                if max_total is not None and total_created >= max_total:
                                    logger.info("Stage1 reached max_total=%s", max_total)
                                    return
                                suffix = existing_counts[intent] + 1
                                fallback_text = f"{intent} request {suffix}"
                                norm_hash = hash_text(normalize_text(fallback_text))
                                if norm_hash not in existing_hashes:
                                    query_id = stable_id("q", next_idx)
                                    next_idx += 1
                                    record = {
                                        "query_id": query_id,
                                        "intent": intent,
                                        "query_text": fallback_text,
                                        "difficulty": "easy",
                                        "tags": [intent.lower().replace(" ", "_")],
                                        "created_at": datetime.utcnow().isoformat() + "Z",
                                        "source": "fallback",
                                        "gen": {
                                            "llm_provider": result_provider,
                                            "model": result_model,
                                            "prompt_version": "query_gen_v1",
                                            "temperature": temperature,
                                            "seed": seed,
                                        },
                                    }
                                    writer.append(record)
                                    existing_hashes.add(norm_hash)
                                    existing_counts[intent] += 1
                                    total_created += 1
                                else:
                                    existing_counts[intent] += 1
                            break
                        logger.error("Stage1 stopping intent=%s after %s failures", intent, failures)
                        break
                    continue

            if not isinstance(payload, list) or not payload:
                logger.error("Stage1 unexpected or empty payload intent=%s", intent)
                failures += 1
                if failures >= max_failures_per_intent:
                    if fill_missing_with_fallback:
                        logger.error("Stage1 fallback intent=%s after %s failures", intent, failures)
                        while existing_counts[intent] < target:
                            if max_total is not None and total_created >= max_total:
                                logger.info("Stage1 reached max_total=%s", max_total)
                                return
                            suffix = existing_counts[intent] + 1
                            fallback_text = f"{intent} request {suffix}"
                            norm_hash = hash_text(normalize_text(fallback_text))
                            if norm_hash not in existing_hashes:
                                query_id = stable_id("q", next_idx)
                                next_idx += 1
                                record = {
                                    "query_id": query_id,
                                    "intent": intent,
                                    "query_text": fallback_text,
                                    "difficulty": "easy",
                                    "tags": [intent.lower().replace(" ", "_")],
                                    "created_at": datetime.utcnow().isoformat() + "Z",
                                    "source": "fallback",
                                    "gen": {
                                        "llm_provider": result_provider,
                                        "model": result_model,
                                        "prompt_version": "query_gen_v1",
                                        "temperature": temperature,
                                        "seed": seed,
                                    },
                                }
                                writer.append(record)
                                existing_hashes.add(norm_hash)
                                existing_counts[intent] += 1
                                total_created += 1
                            else:
                                existing_counts[intent] += 1
                        break
                    logger.error("Stage1 stopping intent=%s after %s failures", intent, failures)
                    break
                continue

            created = 0
            for item in payload:
                if not isinstance(item, dict):
                    continue
                query_text = str(item.get("query_text", "")).strip()
                if not query_text:
                    continue
                norm_hash = hash_text(normalize_text(query_text))
                if norm_hash in existing_hashes:
                    continue
                query_id = stable_id("q", next_idx)
                next_idx += 1
                record = {
                    "query_id": query_id,
                    "intent": intent,
                    "query_text": query_text,
                    "difficulty": item.get("difficulty", "medium"),
                    "tags": item.get("tags", []),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "source": "generated",
                    "gen": {
                        "llm_provider": result_provider,
                        "model": result_model,
                        "prompt_version": "query_gen_v1",
                        "temperature": temperature,
                        "seed": seed_value,
                    },
                }
                writer.append(record)
                existing_hashes.add(norm_hash)
                existing_counts[intent] += 1
                created += 1
                total_created += 1
                if max_total is not None and total_created >= max_total:
                    logger.info("Stage1 reached max_total=%s", max_total)
                    return
                if existing_counts[intent] >= target:
                    break
            if created == 0:
                failures += 1
                logger.warning(
                    "Stage1 no new queries intent=%s failures=%s/%s",
                    intent,
                    failures,
                    max_failures_per_intent,
                )
                if failures >= max_failures_per_intent:
                    if fill_missing_with_fallback:
                        logger.error("Stage1 fallback intent=%s after %s failures", intent, failures)
                        while existing_counts[intent] < target:
                            if max_total is not None and total_created >= max_total:
                                logger.info("Stage1 reached max_total=%s", max_total)
                                return
                            suffix = existing_counts[intent] + 1
                            fallback_text = f"{intent} request {suffix}"
                            norm_hash = hash_text(normalize_text(fallback_text))
                            if norm_hash not in existing_hashes:
                                query_id = stable_id("q", next_idx)
                                next_idx += 1
                                record = {
                                    "query_id": query_id,
                                    "intent": intent,
                                    "query_text": fallback_text,
                                    "difficulty": "easy",
                                    "tags": [intent.lower().replace(" ", "_")],
                                    "created_at": datetime.utcnow().isoformat() + "Z",
                                    "source": "fallback",
                                    "gen": {
                                        "llm_provider": result_provider,
                                        "model": result_model,
                                        "prompt_version": "query_gen_v1",
                                        "temperature": temperature,
                                        "seed": seed,
                                    },
                                }
                                writer.append(record)
                                existing_hashes.add(norm_hash)
                                existing_counts[intent] += 1
                                total_created += 1
                            else:
                                existing_counts[intent] += 1
                    else:
                        logger.error("Stage1 stopping intent=%s after %s failures", intent, failures)
                        break
            if fill_missing_with_fallback and existing_counts[intent] < target:
                while existing_counts[intent] < target:
                    if max_total is not None and total_created >= max_total:
                        logger.info("Stage1 reached max_total=%s", max_total)
                        return
                    suffix = existing_counts[intent] + 1
                    fallback_text = f"{intent} request {suffix}"
                    norm_hash = hash_text(normalize_text(fallback_text))
                    if norm_hash not in existing_hashes:
                        query_id = stable_id("q", next_idx)
                        next_idx += 1
                        record = {
                            "query_id": query_id,
                            "intent": intent,
                            "query_text": fallback_text,
                            "difficulty": "easy",
                            "tags": [intent.lower().replace(" ", "_")],
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "source": "fallback",
                            "gen": {
                                "llm_provider": result_provider,
                                "model": result_model,
                                "prompt_version": "query_gen_v1",
                                "temperature": temperature,
                                "seed": seed,
                            },
                        }
                        writer.append(record)
                        existing_hashes.add(norm_hash)
                        existing_counts[intent] += 1
                        total_created += 1
                    else:
                        existing_counts[intent] += 1
            logger.info("Stage1 intent=%s created=%d total=%d", intent, created, existing_counts[intent])
