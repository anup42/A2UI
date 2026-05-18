from __future__ import annotations

from pathlib import Path


def merge_lora_adapter(base_model_id: str, adapter_dir: str | Path, output_dir: str | Path) -> Path:
    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before merging LoRA adapters.") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
