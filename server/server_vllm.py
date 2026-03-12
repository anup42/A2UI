from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import os
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    VLLM_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    VLLM_IMPORT_ERROR = exc


LOGGER = logging.getLogger("local_genui_server_vllm")
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"
JSON_MODE_INSTRUCTION = "Return only valid JSON. Do not include markdown code fences."
QWEN_DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    enable_thinking: bool | None = None
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)
    json_mode: bool = False
    model_path: str | None = None


class GenerateResponse(BaseModel):
    text: str
    model_path: str
    usage: dict[str, int]
    timings: dict[str, float]


class VllmGenerationEngine:
    def __init__(
        self,
        default_model_path: str,
        trust_remote_code: bool,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        enforce_eager: bool
    ) -> None:
        self._default_model_path = default_model_path.strip() or DEFAULT_MODEL_PATH
        self._trust_remote_code = trust_remote_code
        self._dtype = dtype
        self._tensor_parallel_size = max(1, int(tensor_parallel_size))
        self._gpu_memory_utilization = max(0.1, min(0.99, float(gpu_memory_utilization)))
        self._max_model_len = max(0, int(max_model_len))
        self._enforce_eager = enforce_eager

        self._request_lock = asyncio.Lock()
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._loaded_model_path: str | None = None
        self._last_load_ms: float | None = None
        self._last_failed_load_key: tuple[str, str, bool] | None = None
        self._last_failed_load_error: str | None = None

    @property
    def loaded_model_path(self) -> str | None:
        return self._loaded_model_path

    @property
    def last_load_ms(self) -> float | None:
        return self._last_load_ms

    @property
    def last_failed_load_error(self) -> str | None:
        return self._last_failed_load_error

    async def warmup(self) -> None:
        async with self._request_lock:
            await self._ensure_model_locked(self._normalize_model_path(self._default_model_path))

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        target_model_path = self._normalize_model_path(
            (request.model_path or self._default_model_path).strip() or self._default_model_path
        )
        async with self._request_lock:
            await self._ensure_model_locked(target_model_path)
            return await asyncio.to_thread(self._generate_sync, request, target_model_path)

    async def _ensure_model_locked(self, model_path: str) -> None:
        if self._llm is not None and self._tokenizer is not None and self._loaded_model_path == model_path:
            return

        if self._llm is not None and self._loaded_model_path != model_path:
            LOGGER.info(
                "Switching loaded model from '%s' to '%s' due request model path.",
                self._loaded_model_path,
                model_path
            )

        failed_key = (model_path, self._dtype, self._enforce_eager)
        if self._last_failed_load_key == failed_key and self._last_failed_load_error is not None:
            raise RuntimeError(
                "Previous load attempt for this model/backend already failed. "
                f"Last error: {self._last_failed_load_error}"
            )

        try:
            await asyncio.to_thread(self._load_model_sync, model_path)
        except Exception as exc:
            self._last_failed_load_key = failed_key
            self._last_failed_load_error = str(exc)
            raise

    def _load_model_sync(self, model_path: str) -> None:
        if VLLM_IMPORT_ERROR is not None or LLM is None:
            raise RuntimeError(
                "vLLM is not available in this environment. "
                f"Original import error: {VLLM_IMPORT_ERROR}"
            )

        started = time.perf_counter()
        LOGGER.info("Loading vLLM model from '%s'...", model_path)
        self._cleanup_engine()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self._trust_remote_code,
            use_fast=True
        )

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tokenizer": model_path,
            "dtype": self._dtype,
            "trust_remote_code": self._trust_remote_code,
            "tensor_parallel_size": self._tensor_parallel_size,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "enforce_eager": self._enforce_eager
        }
        if self._max_model_len > 0:
            llm_kwargs["max_model_len"] = self._max_model_len

        self._llm = LLM(**llm_kwargs)
        self._loaded_model_path = model_path
        self._last_load_ms = (time.perf_counter() - started) * 1000.0
        self._last_failed_load_key = None
        self._last_failed_load_error = None
        LOGGER.info(
            "vLLM model ready: path='%s', dtype='%s', tp=%d, load_ms=%.0f",
            self._loaded_model_path,
            self._dtype,
            self._tensor_parallel_size,
            self._last_load_ms
        )

    def _cleanup_engine(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()

    def _normalize_model_path(self, model_path: str) -> str:
        value = model_path.strip()
        if not value:
            return self._default_model_path
        expanded = os.path.expanduser(value)
        if os.path.isabs(expanded):
            return os.path.realpath(os.path.normpath(expanded))
        return value

    def _is_qwen_instruct_model(self, model_path: str) -> bool:
        normalized = model_path.replace("\\", "/").lower()
        return "qwen2.5-coder" in normalized and "instruct" in normalized

    def _build_prompt_text(self, request: GenerateRequest, model_path: str) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        system_prompt = (request.system_prompt or "").strip()
        if not system_prompt and self._is_qwen_instruct_model(model_path):
            system_prompt = QWEN_DEFAULT_SYSTEM_PROMPT
        if request.json_mode:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{JSON_MODE_INSTRUCTION}"
            else:
                system_prompt = JSON_MODE_INSTRUCTION

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request.prompt.strip()})

        if hasattr(self._tokenizer, "apply_chat_template"):
            apply_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True
            }
            # Qwen vLLM docs mention enable_thinking in chat template kwargs for Qwen3.
            # Pass it only when explicitly requested and tokenizer supports the argument.
            if request.enable_thinking is not None:
                try:
                    import inspect

                    params = inspect.signature(self._tokenizer.apply_chat_template).parameters
                    if "enable_thinking" in params:
                        apply_kwargs["enable_thinking"] = request.enable_thinking
                except Exception:  # noqa: BLE001
                    pass
            return self._tokenizer.apply_chat_template(messages, **apply_kwargs)

        # Fallback in case tokenizer lacks chat template helper.
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        parts.append(f"User: {request.prompt.strip()}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _generate_sync(self, request: GenerateRequest, model_path: str) -> GenerateResponse:
        if self._llm is None:
            raise RuntimeError("vLLM model is not loaded.")
        if SamplingParams is None:
            raise RuntimeError("vLLM SamplingParams is unavailable.")

        started = time.perf_counter()
        prompt_text = self._build_prompt_text(request, model_path)
        prompt_char_len = len(prompt_text)
        temperature = float(request.temperature)
        do_sample = temperature > 0.0

        sampling = SamplingParams(
            max_tokens=int(request.max_output_tokens),
            temperature=temperature if do_sample else 0.0,
            top_p=float(request.top_p) if do_sample else 1.0,
            top_k=int(request.top_k),
            presence_penalty=float(request.presence_penalty)
        )

        LOGGER.info(
            "vLLM generation started. model=%s prompt_chars=%d max_tokens=%d temp=%.2f top_p=%.2f top_k=%d",
            model_path,
            prompt_char_len,
            sampling.max_tokens,
            temperature,
            sampling.top_p,
            sampling.top_k
        )

        outputs = self._llm.generate([prompt_text], sampling_params=sampling, use_tqdm=False)
        if not outputs:
            raise RuntimeError("vLLM returned no outputs.")

        result = outputs[0]
        prompt_tokens = len(getattr(result, "prompt_token_ids", []) or [])
        text = ""
        completion_tokens = 0
        if getattr(result, "outputs", None):
            first = result.outputs[0]
            text = (getattr(first, "text", "") or "").strip()
            completion_tokens = len(getattr(first, "token_ids", []) or [])

        total_ms = (time.perf_counter() - started) * 1000.0
        tok_per_s = completion_tokens / (total_ms / 1000.0) if completion_tokens > 0 and total_ms > 0 else 0.0
        LOGGER.info(
            "vLLM generation completed. prompt_tokens=%d completion_tokens=%d total_ms=%.0f tok_per_s=%.2f",
            prompt_tokens,
            completion_tokens,
            total_ms,
            tok_per_s
        )

        return GenerateResponse(
            text=text,
            model_path=model_path,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            timings={"total_ms": total_ms}
        )


def create_app(engine: VllmGenerationEngine, lazy_load: bool) -> FastAPI:
    app = FastAPI(title="Local GenUI Model Server (vLLM)", version="1.0.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        if lazy_load:
            LOGGER.info("Lazy-load enabled. Model will load on first request.")
            return
        await engine.warmup()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        LOGGER.info("Health check requested. loaded_model=%s", engine.loaded_model_path)
        return {
            "status": "ok",
            "loaded_model_path": engine.loaded_model_path,
            "backend": "vllm",
            "last_load_ms": engine.last_load_ms,
            "last_failed_load_error": engine.last_failed_load_error
        }

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        LOGGER.info(
            "Generate request received. model_path=%s prompt_chars=%d json_mode=%s temp=%.2f top_p=%.2f top_k=%d max_tokens=%d",
            request.model_path or engine.loaded_model_path or "<default>",
            len(request.prompt),
            request.json_mode,
            request.temperature,
            request.top_p,
            request.top_k,
            request.max_output_tokens
        )
        try:
            return await engine.generate(request)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def run_self_test(engine: VllmGenerationEngine, prompt: str, max_output_tokens: int) -> None:
    LOGGER.info(
        "Running vLLM self-test prompt. prompt_chars=%d max_output_tokens=%d",
        len(prompt),
        max_output_tokens
    )
    request = GenerateRequest(
        prompt=prompt,
        system_prompt=None,
        temperature=0.2,
        top_p=0.95,
        top_k=20,
        presence_penalty=0.0,
        enable_thinking=None,
        max_output_tokens=max_output_tokens,
        json_mode=False,
        model_path=None
    )
    started = time.perf_counter()
    response = asyncio.run(engine.generate(request))
    wall_ms = (time.perf_counter() - started) * 1000.0

    print("\n=== SELF TEST OUTPUT (vLLM) ===")
    print(response.text)
    print("=== END SELF TEST OUTPUT ===")
    print(f"model_path: {response.model_path}")
    print(f"usage: {response.usage}")
    print(f"timings: {response.timings}")
    print(f"wall_time_ms: {wall_ms:.0f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local vLLM server for GenUICraft.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Default HF model id or local path.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--lazy-load", action="store_true", help="Load model on first request instead of startup.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="vLLM dtype."
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="Maximum model length. Keep 0 to use vLLM/model default."
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Use eager mode in vLLM.")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"]
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run one generation test at startup and print output."
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="Run one generation test and exit without starting the HTTP server."
    )
    parser.add_argument(
        "--self-test-prompt",
        default="show pizza recipie",
        help="Prompt text used by --self-test / --self-test-only."
    )
    parser.add_argument(
        "--self-test-max-output-tokens",
        type=int,
        default=256,
        help="Max new tokens for self-test generation."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    engine = VllmGenerationEngine(
        default_model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager
    )
    if args.self_test or args.self_test_only:
        run_self_test(
            engine=engine,
            prompt=args.self_test_prompt,
            max_output_tokens=args.self_test_max_output_tokens
        )
    if args.self_test_only:
        LOGGER.info("Self-test completed. Exiting because --self-test-only was set.")
        return

    app = create_app(engine=engine, lazy_load=args.lazy_load)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, access_log=True)


if __name__ == "__main__":
    main()
