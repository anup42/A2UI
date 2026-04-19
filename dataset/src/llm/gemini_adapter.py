from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Optional

from .base import BaseLLMAdapter, LLMResult, LLMRateLimitError
from .http_transport import urlopen


class GeminiAdapter(BaseLLMAdapter):
    _rate_lock = threading.Lock()
    _last_request_at: dict[str, float] = {}

    def __init__(self, spec):
        super().__init__(spec)
        self.single_call_batch_attempts = 0
        self.single_call_batch_successes = 0
        self.single_call_batch_fallbacks = 0

    def _split_keys(self, raw_value: str | None) -> list[str]:
        if not raw_value:
            return []
        normalized = raw_value.replace("\n", ",").replace(";", ",")
        return [item.strip() for item in normalized.split(",") if item.strip()]

    def _is_truthy(self, raw_value: str | None) -> bool:
        if raw_value is None:
            return False
        return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _load_keys(self) -> list[str]:
        keys: list[str] = []
        for env_name in (
            "VERTEX_EXPRESS_API_KEYS",
            "GEMINI_VERTEX_EXPRESS_API_KEYS",
        ):
            for key in self._split_keys(os.getenv(env_name)):
                if key not in keys:
                    keys.append(key)

        for env_name in (
            "VERTEX_EXPRESS_API_KEY",
            "GEMINI_VERTEX_EXPRESS_API_KEY",
        ):
            key = (os.getenv(env_name) or "").strip()
            if key and key not in keys:
                keys.append(key)
        return keys

    def _normalize_model_name(self, raw_model: str) -> str:
        model = (raw_model or "").strip()
        if model.startswith("publishers/google/models/"):
            return model[len("publishers/google/models/") :].strip()
        if model.startswith("models/"):
            return model[len("models/") :].strip()
        return model

    def _single_call_batch_enabled(self) -> bool:
        # Vertex AI Express mode exposes generateContent/streamGenerateContent,
        # but not a native batch generate endpoint. This switch enables
        # one-call multiplexing as the server-side batching path for Express keys.
        raw = os.getenv("GEMINI_EXPRESS_SINGLE_CALL_BATCH_ENABLED", "1")
        return self._is_truthy(raw)

    def _single_call_batch_max_prompts(self) -> int:
        raw = os.getenv("GEMINI_EXPRESS_SINGLE_CALL_BATCH_MAX_PROMPTS", "20").strip()
        try:
            value = int(raw)
        except Exception:
            value = 20
        return max(2, value)

    def _single_call_batch_max_chars(self) -> int:
        raw = os.getenv("GEMINI_EXPRESS_SINGLE_CALL_BATCH_MAX_CHARS", "120000").strip()
        try:
            value = int(raw)
        except Exception:
            value = 120000
        return max(5000, value)

    def _single_call_batch_max_output_tokens(self) -> int:
        raw = os.getenv("GEMINI_EXPRESS_SINGLE_CALL_BATCH_MAX_OUTPUT_TOKENS")
        if raw is not None and str(raw).strip():
            try:
                value = int(str(raw).strip())
            except Exception:
                value = 8192
            return max(512, value)

        value = 8192
        limits = self.spec.limits if isinstance(self.spec.limits, dict) else {}
        spec_cap = limits.get("max_output_tokens")
        try:
            parsed = int(spec_cap) if spec_cap is not None else None
        except Exception:
            parsed = None
        if parsed and parsed > 0:
            value = parsed
        return max(512, value)

    def _request_timeout(self) -> float:
        raw = os.getenv("GEMINI_TIMEOUT_SECONDS", "180")
        try:
            timeout = float(raw)
        except Exception:
            timeout = 180.0
        return max(10.0, timeout)

    def _min_interval(self) -> float:
        raw = os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "0")
        try:
            value = float(raw)
        except Exception:
            value = 0.0
        return max(0.0, value)

    def _thinking_level(self) -> Optional[str]:
        explicit = (os.getenv("GEMINI_THINKING_LEVEL") or "").strip()
        if explicit:
            return explicit
        model = (self.spec.model or "").lower()
        if "gemini-3.1-pro" in model:
            return "low"
        return None

    def _apply_thinking_config(self, generation_config: dict[str, object]) -> None:
        level = self._thinking_level()
        if not level:
            return
        thinking_cfg: dict[str, object] = {"thinkingLevel": level}
        budget_raw = (os.getenv("GEMINI_THINKING_BUDGET") or "").strip()
        if budget_raw:
            try:
                thinking_cfg["thinkingBudget"] = int(budget_raw)
            except Exception:
                pass
        generation_config["thinkingConfig"] = thinking_cfg

    def _wait_for_slot(self, api_key: str, min_interval: float) -> None:
        if min_interval <= 0:
            return
        while True:
            with self._rate_lock:
                now = time.time()
                last = self._last_request_at.get(api_key, 0.0)
                wait = min_interval - (now - last)
                if wait <= 0:
                    self._last_request_at[api_key] = now
                    return
            time.sleep(min(wait, 1.0))

    def _read_http_error_body(self, exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read()
            if isinstance(body, bytes):
                return body.decode("utf-8", errors="replace")
            return str(body)
        except Exception:
            return ""

    def _infer_max_tokens_from_error(self, message: str) -> Optional[int]:
        lowered = message.lower()
        if "maxoutputtokens" not in lowered and "max_output_tokens" not in lowered:
            return None
        match = re.search(r"(?:maxoutputtokens|max_output_tokens)[^0-9]*([0-9]{2,})", lowered)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
        return None

    def _infer_retry_delay_seconds(self, message: str) -> Optional[float]:
        if not message:
            return None
        try:
            payload = json.loads(message)
            details = payload.get("error", {}).get("details", [])
            if isinstance(details, list):
                for item in details:
                    if not isinstance(item, dict):
                        continue
                    retry = item.get("retryDelay")
                    if isinstance(retry, str):
                        value = retry[:-1] if retry.endswith("s") else retry
                        return float(value)
        except Exception:
            pass

        lowered = message.lower()
        match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", lowered)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return None

    def _extract_json_blob(self, text: str) -> Optional[Any]:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            pass

        start = raw.find("{")
        while start >= 0:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(raw)):
                ch = raw[idx]
                if in_string:
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = raw[start : idx + 1]
                        try:
                            return json.loads(chunk)
                        except Exception:
                            break
            start = raw.find("{", start + 1)
        return None

    def _build_single_call_batch_prompt(
        self,
        prompts: list[str],
        seeds: Optional[list[int]],
    ) -> str:
        tasks: list[dict[str, Any]] = []
        for idx, prompt in enumerate(prompts):
            task: dict[str, Any] = {"id": idx, "prompt": prompt}
            if seeds and idx < len(seeds):
                task["seed"] = seeds[idx]
            tasks.append(task)

        envelope = {"tasks": tasks}
        envelope_json = json.dumps(envelope, ensure_ascii=False)
        return (
            "You are running independent generation tasks in one request.\n"
            "For each task, execute only that task prompt and produce exactly the JSON value requested by that prompt.\n"
            "Return only one compact JSON object with this shape:\n"
            '{"results":[{"id":<integer>,"output":<json value>}, ...]}\n'
            "Rules:\n"
            "- Include every task id exactly once.\n"
            "- output must be a JSON value (array/object/string/number/bool/null), not markdown.\n"
            "- No prose, no code fences, no comments.\n"
            "- Use compact JSON (no pretty formatting).\n\n"
            f"Batch input:\n{envelope_json}"
        )

    def _try_single_call_batch(
        self,
        prompts: list[str],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seeds: Optional[list[int]],
        json_mode: bool,
        batch_name: Optional[str],
    ) -> Optional[list[LLMResult]]:
        if not self._single_call_batch_enabled():
            return None
        if len(prompts) < 2:
            return None
        if len(prompts) > self._single_call_batch_max_prompts():
            return None
        total_chars = sum(len(p or "") for p in prompts)
        if total_chars > self._single_call_batch_max_chars():
            return None

        merged_prompt = self._build_single_call_batch_prompt(prompts, seeds)
        base_seed = seeds[0] if seeds else None
        requested = max(1, len(prompts)) * max(1, max_tokens)
        batch_cap = self._single_call_batch_max_output_tokens()
        batch_max_tokens = min(max(max_tokens, requested), batch_cap)

        batch_result = self.generate(
            prompt=merged_prompt,
            system=system,
            temperature=temperature,
            max_tokens=batch_max_tokens,
            seed=base_seed,
            json_mode=True if json_mode else False,
        )
        if batch_result.error:
            return None

        parsed = self._extract_json_blob(batch_result.text)
        if not isinstance(parsed, dict):
            return None
        records = parsed.get("results")
        if not isinstance(records, list):
            return None

        by_id: dict[int, Any] = {}
        for item in records:
            if not isinstance(item, dict):
                continue
            idx = item.get("id")
            if not isinstance(idx, int):
                continue
            by_id[idx] = item.get("output")

        if len(by_id) < len(prompts):
            return None

        out: list[LLMResult] = []
        input_per = int((batch_result.input_tokens or 0) / max(1, len(prompts)))
        output_per = int((batch_result.output_tokens or 0) / max(1, len(prompts)))

        for idx in range(len(prompts)):
            if idx not in by_id:
                return None
            value = by_id[idx]
            if isinstance(value, str):
                text_value = value.strip()
            else:
                text_value = json.dumps(value, ensure_ascii=False)
            out.append(
                LLMResult(
                    text=text_value,
                    raw={
                        "batch_mode": "express_single_call",
                        "batch_name": batch_name or "",
                        "task_index": idx,
                    },
                    latency_ms=batch_result.latency_ms,
                    input_tokens=input_per,
                    output_tokens=output_per,
                    cost_usd=None,
                    model=self.spec.model,
                    provider=self.spec.provider,
                    error=None,
                )
            )
        return out

    def _generate_batch_sequential(
        self,
        prompts: list[str],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seeds: Optional[list[int]],
        json_mode: bool = False,
    ) -> list[LLMResult]:
        results: list[LLMResult] = []
        for idx, prompt in enumerate(prompts):
            seed = seeds[idx] if seeds and idx < len(seeds) else None
            result = self.generate(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                json_mode=json_mode,
            )
            results.append(result)
        return results

    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        keys = self._load_keys()
        if not keys:
            return LLMResult(
                text="",
                raw=None,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error=(
                    "VERTEX_EXPRESS_API_KEY/GEMINI_VERTEX_EXPRESS_API_KEY not set. "
                    "Dataset Gemini calls require Vertex AI Express API keys."
                ),
            )

        model_name = self._normalize_model_name(self.spec.model)
        endpoint = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent"

        limits = self.spec.limits if isinstance(self.spec.limits, dict) else {}
        spec_cap = limits.get("max_output_tokens")
        try:
            model_cap = int(spec_cap) if spec_cap is not None else None
        except Exception:
            model_cap = None
        if not model_cap or model_cap <= 0:
            model_cap = 8192

        max_cap_env = os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
        if max_cap_env:
            try:
                model_cap = min(model_cap, int(max_cap_env))
            except Exception:
                pass
        max_tokens = min(max_tokens, model_cap)

        def _build_body(use_seed: bool, token_limit: int) -> dict[str, object]:
            generation_config: dict[str, object] = {
                "temperature": temperature,
                "maxOutputTokens": token_limit,
            }
            self._apply_thinking_config(generation_config)
            if json_mode:
                generation_config["responseMimeType"] = "application/json"

            body: dict[str, object] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": generation_config,
            }
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            if use_seed and seed is not None:
                body["generationConfig"]["seed"] = seed
            return body

        use_seed = seed is not None
        token_limit = max_tokens
        start = time.time()
        timeout_seconds = self._request_timeout()
        min_interval = self._min_interval()
        retry_backoff = max(1.0, float(os.getenv("GEMINI_RETRY_BACKOFF_SECONDS", "4")))
        max_rate_limit_retries = max(3, int(os.getenv("GEMINI_RATE_LIMIT_MAX_RETRIES", "20")))
        max_timeout_retries = max(1, int(os.getenv("GEMINI_TIMEOUT_MAX_RETRIES", "2")))
        cycles = max(1, int(os.getenv("GEMINI_REQUEST_CYCLES", "3")))

        raw_text: str | None = None
        last_error: Exception | None = None
        last_rate_detail = ""
        last_rate_headers: dict[str, str] = {}
        last_rate_retry_after: float | None = None
        attempts = 0

        for _ in range(cycles):
            for api_key in keys:
                attempts += 1
                params = urllib.parse.urlencode({"key": api_key})
                url = f"{endpoint}?{params}"

                rate_limit_retries = 0
                timeout_retries = 0
                retry_without_seed = False
                retry_with_clamp = False

                while True:
                    body = _build_body(use_seed, token_limit)
                    data = json.dumps(body).encode("utf-8")
                    req = urllib.request.Request(
                        url,
                        data=data,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": api_key,
                        },
                    )
                    try:
                        self._wait_for_slot(api_key, min_interval)
                        with urlopen(req, timeout=timeout_seconds) as resp:
                            raw_text = resp.read().decode("utf-8")
                        break
                    except urllib.error.HTTPError as exc:
                        last_error = exc
                        detail = self._read_http_error_body(exc)
                        headers = dict(exc.headers.items()) if exc.headers else {}
                        lowered = detail.lower()

                        if exc.code in (401, 403) or "api key" in lowered and "invalid" in lowered:
                            break

                        if exc.code in (429, 503):
                            last_rate_detail = detail
                            last_rate_headers = headers
                            retry_after = self._infer_retry_delay_seconds(detail)
                            last_rate_retry_after = retry_after
                            rate_limit_retries += 1
                            if rate_limit_retries > max_rate_limit_retries:
                                break
                            sleep_seconds = max(
                                min_interval,
                                retry_backoff,
                                retry_after if retry_after is not None else 0.0,
                            )
                            time.sleep(sleep_seconds)
                            continue

                        if exc.code == 400:
                            if use_seed and "seed" in lowered and "unsupported" in lowered:
                                use_seed = False
                                if not retry_without_seed:
                                    retry_without_seed = True
                                    continue

                            limit = self._infer_max_tokens_from_error(detail)
                            if limit and limit < token_limit:
                                token_limit = limit
                                if not retry_with_clamp:
                                    retry_with_clamp = True
                                    continue

                            if not retry_with_clamp and token_limit > 2048:
                                token_limit = 2048
                                retry_with_clamp = True
                                continue

                        return LLMResult(
                            text="",
                            raw=None,
                            latency_ms=(time.time() - start) * 1000,
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=None,
                            model=self.spec.model,
                            provider=self.spec.provider,
                            error=f"HTTP {exc.code}: {exc.reason} {detail}".strip(),
                        )
                    except Exception as exc:
                        last_error = exc
                        if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
                            timeout_retries += 1
                            if timeout_retries > max_timeout_retries:
                                break
                            time.sleep(max(min_interval, retry_backoff))
                            continue

                        return LLMResult(
                            text="",
                            raw=None,
                            latency_ms=(time.time() - start) * 1000,
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=None,
                            model=self.spec.model,
                            provider=self.spec.provider,
                            error=str(exc),
                        )

                if raw_text:
                    break
            if raw_text:
                break

        if raw_text is None:
            if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 429:
                limit_detail = dict(self.spec.limits or {})
                if last_rate_retry_after is not None:
                    limit_detail["retry_after_s"] = last_rate_retry_after
                if last_rate_detail:
                    limit_detail["detail"] = last_rate_detail[:500]
                raise LLMRateLimitError(
                    provider=self.spec.provider,
                    model=self.spec.model,
                    limits=limit_detail,
                    headers=last_rate_headers,
                )

            return LLMResult(
                text="",
                raw=None,
                latency_ms=(time.time() - start) * 1000,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error=f"Vertex Express keys exhausted after {attempts} attempts: {last_error}",
            )

        elapsed = (time.time() - start) * 1000
        payload = json.loads(raw_text)
        text = ""
        candidates = payload.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        usage = payload.get("usageMetadata", {})

        return LLMResult(
            text=text or "",
            raw=payload,
            latency_ms=elapsed,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
            error=None,
        )

    def generate_batch(
        self,
        prompts: list[str],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seeds: Optional[list[int]],
        json_mode: bool = False,
        batch_name: Optional[str] = None,
    ) -> list[LLMResult]:
        if not prompts:
            return []
        self.single_call_batch_attempts += 1
        batched = self._try_single_call_batch(
            prompts=prompts,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            seeds=seeds,
            json_mode=json_mode,
            batch_name=batch_name,
        )
        if batched is not None:
            self.single_call_batch_successes += 1
            return batched
        self.single_call_batch_fallbacks += 1
        return self._generate_batch_sequential(
            prompts=prompts,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            seeds=seeds,
            json_mode=json_mode,
        )
