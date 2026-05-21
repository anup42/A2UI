from __future__ import annotations

from ir_training.models.base import ExportCapabilities, ModelAdapter


class GemmaAdapter(ModelAdapter):
    family = "gemma"

    def default_lora_targets(self) -> list[str]:
        if "gemma-4" in self.model_id.lower() or bool(self.config.get("gemma4_clippable_linear", False)):
            # Gemma 4 wraps projection layers in Gemma4ClippableLinear. PEFT
            # cannot attach LoRA to that wrapper, so target the inner linear.
            return [
                "q_proj.linear",
                "k_proj.linear",
                "v_proj.linear",
                "o_proj.linear",
                "gate_proj.linear",
                "up_proj.linear",
                "down_proj.linear",
            ]
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def max_context(self) -> int:
        return int(self.config.get("max_context_tokens", 8192))

    def export_capabilities(self) -> ExportCapabilities:
        return ExportCapabilities(hf_lora=True, hf_merged=True, litertlm=True)
