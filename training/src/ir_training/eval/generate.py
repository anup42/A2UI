from __future__ import annotations

from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.url_preprocess import restore_url_placeholders
from ir_training.models.registry import create_adapter


def generate_predictions(
    config: dict[str, Any],
    split_path: str | Path,
    output_path: str | Path,
    max_rows: int | None = None,
    *,
    max_input_tokens: int | None = None,
    max_new_tokens: int | None = None,
    adapter_checkpoint: str | Path | None = None,
    apply_qat: bool = False,
) -> int:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before running generation.") from exc

    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    adapter = create_adapter(model_cfg)
    tokenizer = adapter.load_tokenizer()
    model = adapter.load_model()
    if adapter_checkpoint is not None:
        try:
            from peft import PeftModel  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency failure path
            raise RuntimeError(
                "PEFT is required to evaluate an adapter checkpoint."
            ) from exc
        checkpoint = Path(adapter_checkpoint).expanduser().resolve()
        if not (checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"Adapter checkpoint is missing adapter_config.json: {checkpoint}"
            )
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    qat_controller = None
    if apply_qat:
        from ir_training.qat.fake_quant import prepare_qat_model

        qat_controller = prepare_qat_model(model, config)
    model.eval()

    rows_out: list[dict[str, Any]] = []
    try:
        for idx, row in enumerate(read_jsonl(split_path)):
            if max_rows is not None and idx >= max_rows:
                break
            prompt_text = adapter.format_example(
                row, tokenizer=tokenizer, include_assistant=False
            )
            tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
            if max_input_tokens is not None and int(max_input_tokens) > 0:
                tokenizer_kwargs.update(
                    {"truncation": True, "max_length": int(max_input_tokens)}
                )
            inputs = tokenizer(prompt_text, **tokenizer_kwargs).to(model.device)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": int(
                    max_new_tokens
                    if max_new_tokens is not None
                    else model_cfg.get("max_output_tokens", 8192)
                ),
                "do_sample": False,
            }
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            with torch.no_grad():
                output = model.generate(**inputs, **generation_kwargs)
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            rows_out.append(build_prediction_record(row, generated))
    finally:
        if qat_controller is not None:
            qat_controller.restore()
    return write_jsonl(output_path, rows_out)


def build_prediction_record(
    row: dict[str, Any],
    generated_text: str,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Join one generated completion to the Golden scoring context."""

    url_map = _extract_url_map(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    record: dict[str, Any] = {
        "id": row.get("id"),
        "ui_id": row.get("ui_id") or metadata.get("ui_id"),
        "response_id": row.get("response_id"),
        "query_id": row.get("query_id") or metadata.get("query_id"),
        "intent": row.get("intent") or metadata.get("intent"),
        "intent_bucket": row.get("intent_bucket") or metadata.get("intent_bucket"),
        "tags": row.get("tags") or metadata.get("tags"),
        "assets": row.get("assets") or [],
        "expected_ui_contract": row.get("expected_ui_contract"),
        "expected_ui_contract_source": row.get("expected_ui_contract_source"),
        "expected_ui_contract_v5_4": row.get("expected_ui_contract_v5_4"),
        "expected_ui_contract_v5_4_source": row.get(
            "expected_ui_contract_v5_4_source"
        ),
        "response_text": restore_url_placeholders(_extract_user_text(row), url_map),
        "expected": restore_url_placeholders(
            _extract_expected_completion(row), url_map
        ),
        "generated_text": str(generated_text),
        "url_map": url_map,
    }
    if runtime:
        record["runtime"] = runtime
    return record


def _extract_user_text(row: dict[str, Any]) -> str:
    response_text = row.get("response_text")
    if isinstance(response_text, str) and response_text.strip():
        return response_text
    # Prepared rows may contain few-shot user/assistant examples before the
    # held-out request. The final user turn is the scoring source, not the
    # first demonstration turn.
    for message in reversed(list(row.get("messages") or [])):
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content") or "")
            marker = "Create A2UI Express v1 GenUI IR for this response:\n\n"
            return content.split(marker, 1)[-1]
    return str(row.get("input") or row.get("prompt") or "")


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
