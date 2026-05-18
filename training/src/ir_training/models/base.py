from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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

    def load_tokenizer(self):
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=bool(self.config.get("trust_remote_code", False)),
        )
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore
        import torch  # type: ignore

        dtype_name = str(self.config.get("dtype", "bfloat16")).lower()
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16 if dtype_name == "float16" else torch.float32
        kwargs: dict[str, Any] = {
            "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
            "torch_dtype": dtype,
            "device_map": "auto",
        }
        if bool(self.config.get("load_in_4bit", False)):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def format_example(self, example: dict[str, Any], tokenizer: Any | None = None, include_assistant: bool = True) -> str:
        messages = list(example.get("messages") or [])
        if not include_assistant and messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not include_assistant)
        return "\n\n".join(f"{m.get('role', 'user').title()}:\n{m.get('content', '')}" for m in messages)
