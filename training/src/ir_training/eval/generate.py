from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.models.registry import create_adapter


def generate_predictions(config: dict[str, Any], split_path: str | Path, output_path: str | Path, max_rows: int | None = None) -> int:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before running generation.") from exc

    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    adapter = create_adapter(model_cfg)
    tokenizer = adapter.load_tokenizer()
    model = adapter.load_model()
    model.eval()

    rows_out: list[dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(split_path)):
        if max_rows is not None and idx >= max_rows:
            break
        prompt_text = adapter.format_example(row, tokenizer=tokenizer, include_assistant=False)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=int(model_cfg.get("max_output_tokens", 8192)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        rows_out.append(
            {
                "id": row.get("id"),
                "response_id": row.get("response_id"),
                "response_text": _extract_user_text(row),
                "expected": _safe_load_json(row.get("completion")),
                "generated_text": generated,
            }
        )
    return write_jsonl(output_path, rows_out)


def _extract_user_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content") or "")
            marker = "Create GenUI flat-spec IR for this response:\n\n"
            return content.split(marker, 1)[-1]
    return str(row.get("prompt") or "")


def _safe_load_json(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None
