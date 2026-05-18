from __future__ import annotations

from ir_training.models.base import ExportCapabilities, ModelAdapter


class LlamaAdapter(ModelAdapter):
    family = "llama"

    def default_lora_targets(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def max_context(self) -> int:
        return int(self.config.get("max_context_tokens", 8192))

    def export_capabilities(self) -> ExportCapabilities:
        return ExportCapabilities(hf_lora=True, hf_merged=True, litertlm=False, gguf=True)
