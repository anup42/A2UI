from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl
from ir_training.common.progress import Progress, log
from ir_training.data.url_preprocess import restore_url_placeholders
from ir_training.generation_policy import (
    build_stopping_criteria,
    generation_cache_scope,
    generation_diagnostics,
    preserve_generation_eos,
    sha256_text,
    stop_express_completion,
)
from ir_training.models.registry import create_adapter
from ir_training.qat.full_model_contract import full_parameter_autocast


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
        __import__("torch")
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before running generation.") from exc

    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    adapter = create_adapter(model_cfg)
    with Progress(f"Golden worker {os.environ.get('A2UI_EVAL_WORKER', '0')}: load tokenizer", unit="stage"):
        tokenizer = adapter.load_tokenizer()
    if config.get("prepared_evaluation_contract") is not None:
        from ir_training.eval.prepared_contract import (
            verify_loaded_evaluation_tokenizer,
        )

        verify_loaded_evaluation_tokenizer(tokenizer, model_cfg, config["prepared_evaluation_contract"])
    from ir_training.train.cuda_runtime import sdpa_policy

    with sdpa_policy(config):
        return _generate_predictions_loaded_tokenizer(
            config, adapter, tokenizer, split_path, output_path, max_rows,
            max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
            adapter_checkpoint=adapter_checkpoint, apply_qat=apply_qat,
        )


def _generate_predictions_loaded_tokenizer(
    config, adapter, tokenizer, split_path, output_path, max_rows, *,
    max_input_tokens, max_new_tokens, adapter_checkpoint, apply_qat,
) -> int:
    import torch

    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    worker = os.environ.get("A2UI_EVAL_WORKER", "0")
    with Progress(f"Golden worker {worker}: load model", unit="stage"):
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
        with Progress(f"Golden worker {worker}: load adapter", unit="stage"):
            model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    inference_device = place_model_for_generation(model, model_cfg)
    qat_controller = None
    if apply_qat:
        from ir_training.qat.fake_quant import prepare_qat_model

        qat_controller = prepare_qat_model(model, config)
    model.eval()
    eos_ids = preserve_generation_eos(model, tokenizer)

    rows = list(read_jsonl(split_path))
    if max_rows is not None:
        rows = rows[:max_rows]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Never expose a partial benchmark as a completed predictions file. The
    # flushed sidecar remains useful for diagnosis if a case fails or hangs.
    partial = output_path.with_name(output_path.name + ".partial")
    completed = 0
    try:
        with generation_cache_scope(model), partial.open("w", encoding="utf-8", newline="\n") as stream, Progress(
            f"Golden worker {worker}: generation on {inference_device}", total=len(rows), unit="cases",
        ) as progress:
            for idx, row in enumerate(rows):
                _extract_user_text(row)
                prompt_text = adapter.format_example(
                    row, tokenizer=tokenizer, include_assistant=False
                )
                # The adapter already applies the tokenizer chat template.
                # Never truncate the source while evaluating a fixed benchmark.
                inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
                input_length = inputs["input_ids"].shape[-1]
                if max_input_tokens and input_length > int(max_input_tokens):
                    raise ValueError(f"Evaluation row {row.get('id')} exceeds max_input_tokens; increase the limit, do not truncate its source.")
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": int(max_new_tokens if max_new_tokens is not None else model_cfg.get("max_output_tokens", 2048)),
                    "do_sample": False,
                    "use_cache": True,
                    "eos_token_id": eos_ids,
                    "stop_strings": None,
                    "stopping_criteria": build_stopping_criteria(tokenizer, input_length),
                }
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = getattr(tokenizer, "eos_token_id", None)
                if pad_token_id is not None:
                    generation_kwargs["pad_token_id"] = pad_token_id
                log(f"Golden worker {worker}: case {idx + 1}/{len(rows)} id={row.get('id')} starting; "
                    f"input_tokens={input_length}; max_new_tokens={generation_kwargs['max_new_tokens']}")
                started = time.perf_counter()
                with torch.inference_mode(), full_parameter_autocast(model):
                    output = model.generate(**inputs, **generation_kwargs)
                generated = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
                runtime = generation_diagnostics(tokenizer, output[0][input_length:],
                    eos_token_ids=eos_ids, max_new_tokens=generation_kwargs["max_new_tokens"],
                    prompt_text=prompt_text, input_ids=inputs["input_ids"][0])
                elapsed = time.perf_counter() - started
                runtime.update(generation_seconds=elapsed,
                    output_tokens_per_second=runtime["output_tokens"] / elapsed if elapsed else 0.0,
                    inference_device=inference_device, use_cache=True, evaluation_worker=worker)
                stream.write(json.dumps(build_prediction_record(row, generated, runtime=runtime),
                    ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                completed += 1
                progress.advance()
                log(f"Golden worker {worker}: case {idx + 1}/{len(rows)} id={row.get('id')} finished; "
                    f"seconds={elapsed:.2f}; output_tokens={runtime['output_tokens']}; "
                    f"tokens/s={runtime['output_tokens_per_second']:.2f}")
    finally:
        if qat_controller is not None:
            qat_controller.restore()
    partial.replace(output_path)
    return completed


def place_model_for_generation(model: Any, model_config: dict[str, Any]) -> str:
    """Choose standalone inference placement separately from Trainer's DDP map.

    A training recipe's device_map:none leaves from_pretrained on CPU. Unlike
    Trainer, standalone evaluation must explicitly move that model to CUDA.
    Preserve Accelerate-managed placement when a device map is already active.
    """
    import torch

    requested = str(model_config.get("inference_device") or "auto").lower()
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        if requested != "auto":
            raise ValueError("model.inference_device cannot override an active HF device map; use one placement mechanism.")
        return str(getattr(model, "device", "hf_device_map"))
    current = getattr(model, "device", None)
    if requested == "auto":
        if current is not None and str(current).split(":", 1)[0] not in {"cpu", "meta"}:
            return str(current)
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.device(requested)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but CUDA is unavailable.")
    if current is None or torch.device(current) != target:
        model.to(target)
    return str(getattr(model, "device", target))


def aggregate_generation_performance(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize observed row timings; do not infer missing runtime measurements.

    The throughput denominator is the sum of measured per-row times. Distributed
    callbacks separately report aggregate wall throughput including rank waiting.
    """
    observed: list[tuple[float, float]] = []
    for row in rows:
        runtime = row.get("runtime") if isinstance(row.get("runtime"), dict) else {}
        seconds, tokens = runtime.get("generation_seconds"), runtime.get("output_tokens")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (seconds, tokens)):
            continue
        if seconds > 0 and tokens >= 0:
            observed.append((float(seconds), float(tokens)))
    if not observed:
        return {}
    seconds = sum(pair[0] for pair in observed)
    tokens = sum(pair[1] for pair in observed)
    return {
        "generation_runtime_measured_rows": len(observed),
        "generation_runtime_row_seconds_sum": seconds,
        "generation_runtime_mean_row_seconds": seconds / len(observed),
        "generation_runtime_max_row_seconds": max(pair[0] for pair in observed),
        "generation_runtime_output_tokens": int(tokens),
        "generation_runtime_tokens_per_row_second": tokens / seconds,
    }


def build_prediction_record(
    row: dict[str, Any],
    generated_text: str,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Join one generated completion to the Golden scoring context."""

    url_map = _extract_url_map(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source = restore_url_placeholders(_extract_user_text(row), url_map)
    # generated_text is the serving candidate. Preserve the exact observed
    # runtime text separately; this cannot reconstruct an unobserved continuation.
    serving_text = stop_express_completion(str(generated_text))
    record: dict[str, Any] = {
        "id": row.get("id"),
        "source_id": row.get("source_id") or metadata.get("source_id"),
        "ui_id": row.get("ui_id") or metadata.get("ui_id"),
        "response_id": row.get("response_id"),
        "query_id": row.get("query_id") or metadata.get("query_id"),
        "intent": row.get("intent") or metadata.get("intent"),
        "intent_bucket": row.get("intent_bucket") or metadata.get("intent_bucket"),
        "tags": row.get("tags") or metadata.get("tags"),
        "assets": row.get("assets") if row.get("assets") is not None else metadata.get("assets", []),
        "expected_ui_contract": row.get("expected_ui_contract"),
        "expected_ui_contract_source": row.get("expected_ui_contract_source"),
        "expected_ui_contract_v5_4": row.get("expected_ui_contract_v5_4"),
        "expected_ui_contract_v5_4_source": row.get(
            "expected_ui_contract_v5_4_source"
        ),
        "response_text": source,
        "response_text_sha256": sha256_text(source),
        "expected": (None if row.get("reference_available") is False else
                     restore_url_placeholders(_extract_expected_completion(row), url_map)),
        "generated_text": serving_text,
        "raw_generated_text": str(generated_text),
        "serving_stop_applied": serving_text != str(generated_text),
        "raw_output_scope": "tokens_actually_returned_by_runtime; continuation_after_stop_unobserved",
        "url_map": url_map,
    }
    for key in ("expected_ui_contract", "expected_ui_contract_source",
                "expected_ui_contract_v5_4", "expected_ui_contract_v5_4_source"):
        if record.get(key) is None and metadata.get(key) is not None:
            record[key] = metadata[key]
    for key in ("benchmark", "repeated_from", "replaces"):
        if row.get(key) is not None or metadata.get(key) is not None:
            record[key] = row.get(key) if row.get(key) is not None else metadata[key]
    if row.get("reference_available") is False:
        if _extract_expected_completion(row).strip():
            raise ValueError("Source-only evaluation cannot carry a reference completion")
        record.update(reference_available=False, evaluation_only=True, selection_role=row.get("selection_role"))
    record["expected_sha256"] = sha256_text(str(record["expected"] or ""))
    record["source_context_sha256"] = prediction_source_context_hash(record)
    if runtime:
        record["runtime"] = runtime
    return record


def prediction_source_context_hash(record: dict[str, Any]) -> str:
    context_keys = ("id", "source_id", "response_id", "query_id", "ui_id", "intent", "intent_bucket",
                    "assets", "expected_ui_contract", "expected_ui_contract_source",
                    "expected_ui_contract_v5_4", "expected_ui_contract_v5_4_source",
                    "response_text_sha256", "expected_sha256", "url_map")
    context = {key: record.get(key) for key in context_keys}
    context.update({key: record[key] for key in ("benchmark", "repeated_from", "replaces", "reference_available", "evaluation_only", "selection_role") if key in record})
    return sha256_text(json.dumps(context, ensure_ascii=False, sort_keys=True))


def _extract_user_text(row: dict[str, Any]) -> str:
    response_text = row.get("response_text")
    # Prepared rows may contain few-shot user/assistant examples before the
    # held-out request. The final user turn is the scoring source, not the
    # first demonstration turn.
    for message in reversed(list(row.get("messages") or [])):
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content") or "")
            marker = "Create A2UI Express v1 GenUI IR for this response:\n\n"
            final_response = content.split(marker, 1)[-1]
            if isinstance(response_text, str) and response_text.strip():
                url_map = _extract_url_map(row)
                # Known prefix is comparable; unprefixed user content may be a
                # higher-level request and is not asserted to be Stage-2 text.
                if marker in content and restore_url_placeholders(response_text, url_map).strip() != restore_url_placeholders(final_response, url_map).strip():
                    raise ValueError(f"Scoring source mismatch for row {row.get('id')}: response_text differs from the final user task")
                return response_text
            return final_response
    if isinstance(response_text, str) and response_text.strip():
        return response_text
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
    url_map = row.get("url_map") or url_preprocessing.get("url_map")
    return url_map if isinstance(url_map, dict) else {}
