from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("local_genui_server")
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"
JSON_MODE_INSTRUCTION = "Return only valid JSON. Do not include markdown code fences."


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)
    json_mode: bool = False
    model_path: str | None = None


class GenerateResponse(BaseModel):
    text: str
    model_path: str
    usage: dict[str, int]
    timings: dict[str, float]


class LocalGenerationEngine:
    def __init__(
        self,
        default_model_path: str,
        preferred_device: str,
        trust_remote_code: bool,
        attn_implementation: str,
        strict_attention_backend: bool
    ) -> None:
        self._default_model_path = default_model_path.strip() or DEFAULT_MODEL_PATH
        self._preferred_device = preferred_device.strip() or "cuda"
        self._trust_remote_code = trust_remote_code
        self._attn_implementation = attn_implementation.strip().lower()
        self._strict_attention_backend = strict_attention_backend

        self._request_lock = asyncio.Lock()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._loaded_model_path: str | None = None
        self._inference_device: str = "cpu"
        self._effective_attn_implementation: str = "default"
        self._last_load_ms: float | None = None

    @property
    def loaded_model_path(self) -> str | None:
        return self._loaded_model_path

    @property
    def inference_device(self) -> str:
        return self._inference_device

    @property
    def last_load_ms(self) -> float | None:
        return self._last_load_ms

    @property
    def effective_attn_implementation(self) -> str:
        return self._effective_attn_implementation

    async def warmup(self) -> None:
        async with self._request_lock:
            await self._ensure_model_locked(self._default_model_path)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        target_model_path = (request.model_path or self._default_model_path).strip() or self._default_model_path
        async with self._request_lock:
            await self._ensure_model_locked(target_model_path)
            return await asyncio.to_thread(self._generate_sync, request, target_model_path)

    async def _ensure_model_locked(self, model_path: str) -> None:
        if self._model is not None and self._tokenizer is not None and self._loaded_model_path == model_path:
            return
        await asyncio.to_thread(self._load_model_sync, model_path)

    def _load_model_sync(self, model_path: str, force_attn_implementation: str | None = None) -> None:
        started = time.perf_counter()
        LOGGER.info("Loading model from '%s'...", model_path)

        self._cleanup_model()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self._trust_remote_code,
            use_fast=True
        )

        use_cuda = self._preferred_device.startswith("cuda") and torch.cuda.is_available()
        use_mps = self._preferred_device.startswith("mps") and torch.backends.mps.is_available()

        model_kwargs: dict[str, Any] = {"trust_remote_code": self._trust_remote_code}
        attn_impl = force_attn_implementation or self._resolved_attn_implementation(use_cuda=use_cuda)
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=self._trust_remote_code
        )
        self._apply_attention_overrides(config, attn_impl)
        model_kwargs["config"] = config
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        if use_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["torch_dtype"] = dtype
            model_kwargs["device_map"] = "auto"
            inference_device = "cuda"
        else:
            model_kwargs["torch_dtype"] = torch.float32
            inference_device = "mps" if use_mps else "cpu"

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:  # noqa: BLE001
            # Some stacks auto-pick flash attention and fail on non-Ampere GPUs.
            if self._is_flash_attention_error(exc):
                if self._strict_attention_backend:
                    raise
                LOGGER.warning(
                    "FlashAttention is unsupported on this GPU/runtime. Retrying with eager attention."
                )
                model_kwargs["attn_implementation"] = "eager"
                self._apply_attention_overrides(config, "eager")
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            else:
                raise
        if not use_cuda:
            model.to(inference_device)
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._tokenizer = tokenizer
        self._model = model
        self._loaded_model_path = model_path
        self._inference_device = inference_device
        self._effective_attn_implementation = model_kwargs.get("attn_implementation", "default")
        self._last_load_ms = (time.perf_counter() - started) * 1000.0

        LOGGER.info(
            "Model ready: path='%s', device='%s', attn='%s', load_ms=%.0f",
            self._loaded_model_path,
            self._inference_device,
            self._effective_attn_implementation,
            self._last_load_ms
        )

    def _apply_attention_overrides(self, config: Any, attn_impl: str | None) -> None:
        if attn_impl is None:
            return
        # Some model implementations read one of these fields directly.
        setattr(config, "attn_implementation", attn_impl)
        setattr(config, "_attn_implementation", attn_impl)
        if attn_impl != "flash_attention_2":
            if hasattr(config, "use_flash_attn"):
                setattr(config, "use_flash_attn", False)
            if hasattr(config, "flash_attn"):
                setattr(config, "flash_attn", False)
            if hasattr(config, "flash_attention"):
                setattr(config, "flash_attention", False)

    def _resolved_attn_implementation(self, use_cuda: bool) -> str | None:
        if self._attn_implementation in {"eager", "sdpa", "flash_attention_2"}:
            return self._attn_implementation
        if self._attn_implementation == "default":
            return None
        if self._attn_implementation == "auto":
            # Avoid flash-attention hard failures on older GPUs by defaulting to SDPA/eager.
            if use_cuda:
                return "sdpa"
            return "eager"
        return "sdpa" if use_cuda else "eager"

    def _cleanup_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _is_flash_attention_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "flash attention" in message or
            "flashattention" in message or
            "ampere" in message and "newer" in message
        )

    def _generate_sync(self, request: GenerateRequest, model_path: str) -> GenerateResponse:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not loaded.")

        prompt_started = time.perf_counter()
        system_prompt = (request.system_prompt or "").strip()
        if request.json_mode:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{JSON_MODE_INSTRUCTION}"
            else:
                system_prompt = JSON_MODE_INSTRUCTION

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request.prompt.strip()})

        tokenized = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        tokenized = tokenized.to(self._inference_device)
        attention_mask = torch.ones_like(tokenized)

        do_sample = request.temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(request.max_output_tokens),
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "attention_mask": attention_mask
        }
        if do_sample:
            generation_kwargs["temperature"] = float(request.temperature)
            generation_kwargs["top_p"] = 0.95

        try:
            with torch.inference_mode():
                output_ids = self._model.generate(tokenized, **generation_kwargs)
        except Exception as exc:  # noqa: BLE001
            if (
                self._is_flash_attention_error(exc) and
                self._effective_attn_implementation != "eager" and
                not self._strict_attention_backend
            ):
                LOGGER.warning(
                    "FlashAttention was triggered during generation while attn='%s'. "
                    "Reloading with eager attention and retrying once.",
                    self._effective_attn_implementation
                )
                self._load_model_sync(model_path, force_attn_implementation="eager")
                tokenized = tokenized.to(self._inference_device)
                attention_mask = torch.ones_like(tokenized)
                generation_kwargs["attention_mask"] = attention_mask
                with torch.inference_mode():
                    output_ids = self._model.generate(tokenized, **generation_kwargs)
            else:
                raise

        prompt_tokens = int(tokenized.shape[-1])
        completion_ids = output_ids[0, prompt_tokens:]
        completion_tokens = int(completion_ids.shape[-1])
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        total_ms = (time.perf_counter() - prompt_started) * 1000.0
        return GenerateResponse(
            text=text,
            model_path=model_path,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            timings={
                "total_ms": total_ms
            }
        )


def create_app(engine: LocalGenerationEngine, lazy_load: bool) -> FastAPI:
    app = FastAPI(title="Local GenUI Model Server", version="1.0.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        if lazy_load:
            LOGGER.info("Lazy-load enabled. Model will load on first request.")
            return
        await engine.warmup()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "loaded_model_path": engine.loaded_model_path,
            "device": engine.inference_device,
            "attn_implementation": engine.effective_attn_implementation,
            "last_load_ms": engine.last_load_ms
        }

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            return await engine.generate(request)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Hugging Face model server for GenUICraft.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Default HF model id or local path.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda", help="Preferred device: cuda, cpu, or mps.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--lazy-load", action="store_true", help="Load model on first request instead of startup.")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "default", "sdpa", "eager", "flash_attention_2"],
        help="Attention backend used by Transformers."
    )
    parser.add_argument(
        "--strict-attn",
        action="store_true",
        help="Do not fallback to eager when the selected attention backend fails."
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    engine = LocalGenerationEngine(
        default_model_path=args.model_path,
        preferred_device=args.device,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        strict_attention_backend=args.strict_attn
    )
    app = create_app(engine=engine, lazy_load=args.lazy_load)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
