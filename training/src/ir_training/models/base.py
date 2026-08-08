from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root
from ir_training.models.hf_loading import load_hf_model


@dataclass(frozen=True)
class ExportCapabilities:
    hf_lora: bool = True
    hf_merged: bool = True
    litertlm: bool = False
    gguf: bool = False


class ModelAdapter(ABC):
    family: str

    def __init__(self, model_id: str, config: dict[str, Any] | None = None) -> None:
        self.model_id = model_id
        self.config = config or {}

    @abstractmethod
    def default_lora_targets(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def max_context(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def export_capabilities(self) -> ExportCapabilities:
        raise NotImplementedError

    def model_source(self) -> str:
        """Return the load source while preserving ``model_id`` as provenance."""

        configured = self.config.get("model_source")
        if configured is None or not str(configured).strip():
            return self.model_id
        source = resolve_path(str(configured), training_root())
        if not source.exists():
            raise FileNotFoundError(f"Configured model.model_source does not exist: {source}")
        return str(source)

    def tokenizer_source(self) -> str:
        configured = self.config.get("tokenizer_source")
        if configured is None or not str(configured).strip():
            return self.model_id
        value = str(configured).strip()
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Configured model.tokenizer_source does not exist: {candidate}"
                )
            return str(candidate.resolve())
        local = resolve_path(candidate, training_root())
        return str(local) if local.exists() else value

    def load_tokenizer(self):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast  # type: ignore

        tokenizer_source = self.tokenizer_source()
        loader = str(self.config.get("tokenizer_loader", "auto_tokenizer")).strip().lower()
        if loader in {"auto_processor", "processor"}:
            tokenizer = self._load_tokenizer_from_processor()
        elif loader in {"pretrained_tokenizer_fast", "tokenizer_fast", "fast"}:
            try:
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    tokenizer_source,
                    trust_remote_code=bool(self.config.get("trust_remote_code", False)),
                )
            except Exception:
                if not bool(self.config.get("processor_fallback", False)):
                    raise
                tokenizer = self._load_tokenizer_from_processor()
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source,
                    trust_remote_code=bool(self.config.get("trust_remote_code", False)),
                    use_fast=bool(self.config.get("use_fast_tokenizer", True)),
                )
            except Exception:
                if not bool(self.config.get("processor_fallback", False)):
                    raise
                tokenizer = self._load_tokenizer_from_processor()
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        padding_side = str(self.config.get("padding_side", "right")).strip().lower()
        if padding_side in {"left", "right"}:
            tokenizer.padding_side = padding_side
        return tokenizer

    def _load_tokenizer_from_processor(self):
        from transformers import AutoProcessor  # type: ignore

        processor = AutoProcessor.from_pretrained(
            self.tokenizer_source(),
            trust_remote_code=bool(self.config.get("trust_remote_code", False)),
        )
        return getattr(processor, "tokenizer", processor)

    def load_model(self):
        return load_hf_model(self.model_source(), self.config)

    def format_example(self, example: dict[str, Any], tokenizer: Any | None = None, include_assistant: bool = True) -> str:
        messages = list(example.get("messages") or [])
        if not include_assistant and messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            template_kwargs = self.config.get("chat_template_kwargs")
            extra_kwargs = dict(template_kwargs) if isinstance(template_kwargs, dict) else {}
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not include_assistant,
                **extra_kwargs,
            )
        return "\n\n".join(f"{m.get('role', 'user').title()}:\n{m.get('content', '')}" for m in messages)
