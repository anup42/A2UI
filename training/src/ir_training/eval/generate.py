from __future__ import annotations

from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.url_preprocess import restore_url_placeholders
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
        url_map = _extract_url_map(row)
        rows_out.append(
            {
                "id": row.get("id"),
                "response_id": row.get("response_id"),
                "response_text": restore_url_placeholders(_extract_user_text(row), url_map),
                "expected": restore_url_placeholders(_extract_expected_completion(row), url_map),
                "generated_text": generated,
                "url_map": url_map,
            }
        )
    return write_jsonl(output_path, rows_out)


def _extract_user_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content") or "")
            marker = "Create A2UI Express v1 GenUI IR for this response:\n\n"
            return content.split(marker, 1)[-1]
    return str(row.get("prompt") or "")


def _extract_expected_completion(row: dict[str, Any]) -> str:
    completion = row.get("completion")
    if not isinstance(completion, str) or not completion.strip():
        targets = row.get("completion_targets")
        if isinstance(targets, dict):
            completion = targets.get("a2ui_express_v1")
    return str(completion or "")


def _extract_url_map(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    url_preprocessing = metadata.get("url_preprocessing") if isinstance(metadata.get("url_preprocessing"), dict) else {}
    url_map = url_preprocessing.get("url_map")
    return url_map if isinstance(url_map, dict) else {}
