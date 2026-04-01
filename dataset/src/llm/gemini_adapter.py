from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Optional

from .base import BaseLLMAdapter, LLMResult, LLMRateLimitError
from .http_transport import urlopen


class GeminiAdapter(BaseLLMAdapter):
    _rate_lock = threading.Lock()
    _last_request_at: dict[str, float] = {}

    def _is_vertex_mode(self) -> bool:
        mode = (os.getenv("GEMINI_API_MODE") or "vertex").strip().lower()
        return mode not in {"legacy", "generativelanguage", "studio"}

    def _normalize_model_name(self) -> str:
        model = (self.spec.model or "").strip()
        model = model.replace("publishers/google/models/", "")
        model = model.replace("models/", "")
        return model

    def _generate_endpoint(self) -> str:
        normalized_model = self._normalize_model_name()
        if self._is_vertex_mode():
            return (
                "https://aiplatform.googleapis.com/v1/"
                f"publishers/google/models/{normalized_model}:generateContent"
            )
        return f"https://generativelanguage.googleapis.com/v1beta/models/{normalized_model}:generateContent"

    def _load_keys(self) -> list[str]:
        keys = []
        vertex_single = os.getenv("GEMINI_VERTEX_EXPRESS_API_KEY") or os.getenv("VERTEX_EXPRESS_API_KEY")
        if vertex_single:
            keys.append(vertex_single.strip())
        multi = os.getenv("GEMINI_API_KEYS")
        if multi:
            # Support comma-separated and newline-separated keys (e.g., "key1,\nkey2,\nkey3,")
            normalized = multi.replace("\n", ",")
            keys.extend([item.strip() for item in normalized.split(",") if item.strip()])
        single = os.getenv("GEMINI_API_KEY")
        if single and single not in keys:
            keys.append(single.strip())
        deduped: list[str] = []
        for key in keys:
            if key and key not in deduped:
                deduped.append(key)
        return deduped

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
        # Prefer structured retryDelay when present.
        try:
            payload = json.loads(message)
            details = payload.get("error", {}).get("details", [])
            if isinstance(details, list):
                for item in details:
                    if not isinstance(item, dict):
                        continue
                    retry = item.get("retryDelay")
                    if isinstance(retry, str):
                        if retry.endswith("s"):
                            retry = retry[:-1]
                        return float(retry)
        except Exception:
            pass
        # Fallback to plain-text message format:
        # "Please retry in 58.694381314s."
        lowered = message.lower()
        match = re.search(r"retry in\\s+([0-9]+(?:\\.[0-9]+)?)s", lowered)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return None

    def _request_timeout(self) -> float:
        raw = os.getenv("GEMINI_TIMEOUT_SECONDS", "180")
        try:
            timeout = float(raw)
        except Exception:
            timeout = 180.0
        return max(10.0, timeout)

    def _min_interval(self) -> float:
        # Set GEMINI_MIN_INTERVAL_SECONDS=12.5 for free tier style pacing (5 RPM).
        raw = os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "0")
        try:
            value = float(raw)
        except Exception:
            value = 0.0
        return max(0.0, value)

    def _thinking_level(self) -> Optional[str]:
        # Override via env for any model.
        explicit = (os.getenv("GEMINI_THINKING_LEVEL") or "").strip()
        if explicit:
            return explicit
        # Gemini 3.1 Pro can be very slow with default dynamic/high thinking.
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
                    "Gemini API key is missing. Set GEMINI_VERTEX_EXPRESS_API_KEY "
                    "(or VERTEX_EXPRESS_API_KEY), GEMINI_API_KEY, or GEMINI_API_KEYS."
                ),
            )

        endpoint = self._generate_endpoint()

        max_cap_env = os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
        if max_cap_env:
            try:
                max_tokens = min(max_tokens, int(max_cap_env))
            except Exception:
                pass
        else:
            # Conservative cap to avoid invalid maxOutputTokens errors.
            max_tokens = min(max_tokens, 8192)

        def _build_body(use_seed: bool, token_limit: int) -> dict[str, object]:
            generation_config: dict[str, object] = {
                "temperature": temperature,
                "maxOutputTokens": token_limit,
            }
            self._apply_thinking_config(generation_config)
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
        raw = None
        last_error: Exception | None = None
        last_rate_detail = ""
        last_rate_headers: dict[str, str] = {}
        last_rate_retry_after: float | None = None
        attempts = 0
        cycles = max(1, int(os.getenv("GEMINI_REQUEST_CYCLES", "3")))
        for _ in range(cycles):
            for api_key in keys:
                attempts += 1
                params = urllib.parse.urlencode({"key": api_key})
                url = f"{endpoint}?{params}"
                retry_without_seed = False
                retry_with_clamp = False
                rate_limit_retries = 0
                timeout_retries = 0
                while True:
                    body = _build_body(use_seed, token_limit)
                    if json_mode:
                        body.setdefault("generationConfig", {})["responseMimeType"] = "application/json"
                    data = json.dumps(body).encode("utf-8")
                    req = urllib.request.Request(
                        url,
                        data=data,
                        headers={"Content-Type": "application/json"},
                    )
                    try:
                        self._wait_for_slot(api_key, min_interval)
                        with urlopen(req, timeout=timeout_seconds) as resp:
                            raw = resp.read().decode("utf-8")
                        break
                    except urllib.error.HTTPError as exc:
                        last_error = exc
                        detail = self._read_http_error_body(exc)
                        headers = dict(exc.headers.items()) if exc.headers else {}
                        lowered = detail.lower()
                        if exc.code in (401, 403) or "api_key_invalid" in lowered or "api key not valid" in lowered:
                            # Invalid key: try the next key in the list.
                            break
                        if exc.code in (429, 503):
                            last_rate_detail = detail
                            last_rate_headers = headers
                            retry_after = self._infer_retry_delay_seconds(detail)
                            last_rate_retry_after = retry_after
                            rate_limit_retries += 1
                            if rate_limit_retries > max_rate_limit_retries:
                                # Try next key after too many throttled retries on this key.
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
                            # If maxOutputTokens is too large but not explicit, try a smaller cap once.
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
                        # Timeout or transient error -> try next key
                        if isinstance(exc, TimeoutError):
                            timeout_retries += 1
                            if timeout_retries > max_timeout_retries:
                                break
                            time.sleep(max(min_interval, retry_backoff))
                            continue
                        if "timed out" in str(exc).lower():
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
            if raw:
                break

        if raw is None:
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
                error=f"Gemini keys exhausted after {attempts} attempts: {last_error}",
            )

        elapsed = (time.time() - start) * 1000
        payload = json.loads(raw)
        text = ""
        candidates = payload.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
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

        if self._is_vertex_mode():
            seed_list = seeds if seeds and len(seeds) == len(prompts) else [None] * len(prompts)
            results: list[LLMResult] = []
            for prompt, seed_value in zip(prompts, seed_list):
                results.append(
                    self.generate(
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed_value,
                        json_mode=json_mode,
                    )
                )
            return results

        keys = self._load_keys()
        if not keys:
            return [
                LLMResult(
                    text="",
                    raw=None,
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=None,
                    model=self.spec.model,
                    provider=self.spec.provider,
                    error=(
                        "Gemini API key is missing. Set GEMINI_VERTEX_EXPRESS_API_KEY "
                        "(or VERTEX_EXPRESS_API_KEY), GEMINI_API_KEY, or GEMINI_API_KEYS."
                    ),
                )
                for _ in prompts
            ]

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self._normalize_model_name()}:batchGenerateContent"

        max_cap_env = os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
        if max_cap_env:
            try:
                max_tokens = min(max_tokens, int(max_cap_env))
            except Exception:
                pass
        else:
            max_tokens = min(max_tokens, 8192)

        def _build_request(prompt: str, seed_value: Optional[int]) -> dict[str, object]:
            generation_config: dict[str, object] = {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
            self._apply_thinking_config(generation_config)
            if json_mode:
                generation_config["responseMimeType"] = "application/json"
            if seed_value is not None:
                generation_config["seed"] = seed_value
            req: dict[str, object] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": generation_config,
            }
            if system:
                req["systemInstruction"] = {"parts": [{"text": system}]}
            return req

        def _error_results(message: str, elapsed_ms: float) -> list[LLMResult]:
            return [
                LLMResult(
                    text="",
                    raw=None,
                    latency_ms=elapsed_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=None,
                    model=self.spec.model,
                    provider=self.spec.provider,
                    error=message,
                )
                for _ in prompts
            ]

        poll_timeout = float(os.getenv("GEMINI_BATCH_POLL_TIMEOUT", "300"))
        poll_interval = float(os.getenv("GEMINI_BATCH_POLL_INTERVAL", "2"))
        timeout_seconds = self._request_timeout()
        min_interval = self._min_interval()
        retry_backoff = max(1.0, float(os.getenv("GEMINI_RETRY_BACKOFF_SECONDS", "4")))
        max_timeout_retries = max(1, int(os.getenv("GEMINI_TIMEOUT_MAX_RETRIES", "2")))
        start = time.time()
        last_error: Exception | None = None

        cycles = max(1, int(os.getenv("GEMINI_REQUEST_CYCLES", "3")))
        for _ in range(cycles):
            for api_key in keys:
                timeout_retries = 0
                params = urllib.parse.urlencode({"key": api_key})
                url = f"{endpoint}?{params}"
                seed_list = seeds if seeds and len(seeds) == len(prompts) else [None] * len(prompts)
                requests_payload = [
                    {"request": _build_request(prompt, seed_value), "metadata": {"index": idx}}
                    for idx, (prompt, seed_value) in enumerate(zip(prompts, seed_list))
                ]
                payload = {
                    "batch": {
                        "displayName": batch_name or f"stage3_batch_{int(time.time())}",
                        "inputConfig": {"requests": {"requests": requests_payload}},
                    }
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    self._wait_for_slot(api_key, min_interval)
                    with urlopen(req, timeout=timeout_seconds) as resp:
                        raw = resp.read().decode("utf-8")
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    detail = self._read_http_error_body(exc)
                    lowered = detail.lower()
                    if exc.code in (401, 403) or "api_key_invalid" in lowered or "api key not valid" in lowered:
                        continue
                    if exc.code in (429, 503):
                        time.sleep(max(min_interval, retry_backoff))
                        break
                    return _error_results(f"HTTP {exc.code}: {exc.reason} {detail}".strip(), (time.time() - start) * 1000)
                except Exception as exc:
                    last_error = exc
                    if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
                        timeout_retries += 1
                        if timeout_retries > max_timeout_retries:
                            break
                        time.sleep(max(min_interval, retry_backoff))
                        continue
                    return _error_results(str(exc), (time.time() - start) * 1000)

                try:
                    op = json.loads(raw)
                except Exception as exc:
                    return _error_results(f"batch_response_parse_error: {exc}", (time.time() - start) * 1000)

                op_name = op.get("name") or op.get("batch", {}).get("name")
                if op_name:
                    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}?{params}"
                    while True:
                        if isinstance(op, dict) and (op.get("done") or op.get("state") == "SUCCEEDED"):
                            break
                        if isinstance(op, dict) and op.get("state") in ("FAILED", "CANCELLED"):
                            return _error_results(
                                f"batch_failed_state: {op.get('state')}",
                                (time.time() - start) * 1000,
                            )
                        if (time.time() - start) >= poll_timeout:
                            return _error_results("batch_timeout", (time.time() - start) * 1000)
                        time.sleep(poll_interval)
                        try:
                            with urlopen(poll_url, timeout=timeout_seconds) as resp:
                                op_raw = resp.read().decode("utf-8")
                            op = json.loads(op_raw)
                        except urllib.error.HTTPError as exc:
                            last_error = exc
                            if exc.code in (429, 503):
                                time.sleep(poll_interval)
                                continue
                            detail = self._read_http_error_body(exc)
                            return _error_results(
                                f"batch_poll_error: HTTP {exc.code}: {exc.reason} {detail}".strip(),
                                (time.time() - start) * 1000,
                            )
                        except Exception as exc:
                            last_error = exc
                            return _error_results(f"batch_poll_error: {exc}", (time.time() - start) * 1000)

                if isinstance(op, dict) and op.get("error"):
                    err = op.get("error", {})
                    message = err.get("message") if isinstance(err, dict) else str(err)
                    return _error_results(f"batch_error: {message}", (time.time() - start) * 1000)

                batch = {}
                if isinstance(op, dict):
                    batch = op.get("response") or op.get("batch") or op
                output = batch.get("output") if isinstance(batch, dict) else {}
                if not isinstance(output, dict):
                    output = {}

                if output.get("responsesFile") or output.get("responses_file"):
                    return _error_results("batch_responses_file_not_supported", (time.time() - start) * 1000)

                inlined = output.get("inlinedResponses") or output.get("inlined_responses")
                if isinstance(inlined, dict):
                    responses = inlined.get("inlinedResponses") or inlined.get("responses") or []
                elif isinstance(inlined, list):
                    responses = inlined
                else:
                    responses = []
                if not responses:
                    return _error_results("batch_no_inlined_responses", (time.time() - start) * 1000)

                elapsed = (time.time() - start) * 1000
                results: list[LLMResult] = []
                for item in responses:
                    if not isinstance(item, dict):
                        results.append(
                            LLMResult(
                                text="",
                                raw=None,
                                latency_ms=elapsed,
                                input_tokens=0,
                                output_tokens=0,
                                cost_usd=None,
                                model=self.spec.model,
                                provider=self.spec.provider,
                                error="batch_invalid_response_item",
                            )
                        )
                        continue
                    if item.get("error"):
                        err = item.get("error")
                        message = err.get("message") if isinstance(err, dict) else str(err)
                        results.append(
                            LLMResult(
                                text="",
                                raw=item,
                                latency_ms=elapsed,
                                input_tokens=0,
                                output_tokens=0,
                                cost_usd=None,
                                model=self.spec.model,
                                provider=self.spec.provider,
                                error=message,
                            )
                        )
                        continue
                    response = item.get("response") or item.get("inlineResponse") or item
                    candidates = response.get("candidates", []) if isinstance(response, dict) else []
                    text = ""
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(part.get("text", "") for part in parts)
                    usage = response.get("usageMetadata", {}) if isinstance(response, dict) else {}
                    results.append(
                        LLMResult(
                            text=text or "",
                            raw=response,
                            latency_ms=elapsed,
                            input_tokens=usage.get("promptTokenCount", 0),
                            output_tokens=usage.get("candidatesTokenCount", 0),
                            cost_usd=None,
                            model=self.spec.model,
                            provider=self.spec.provider,
                            error=None,
                        )
                    )
                return results

        if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 429:
            raise LLMRateLimitError(
                provider=self.spec.provider,
                model=self.spec.model,
                limits=self.spec.limits,
                headers={},
            )
        elapsed = (time.time() - start) * 1000
        return _error_results(f"Gemini keys exhausted after batch attempts: {last_error}", elapsed)
