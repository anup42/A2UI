from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.url_preprocess import restore_url_placeholders
from ir_training.eval.compare_to_baseline import evaluate_predictions


class TrainingMetadataCallback:
    """Small callback-like helper for writing immutable run metadata."""

    def __init__(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.metadata = metadata

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "training_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def build_golden_set_eval_callback(
    *,
    enabled: bool,
    split_path: str | Path,
    output_dir: str | Path,
    adapter: Any,
    tokenizer: Any,
    max_rows: int = 50,
    max_new_tokens: int = 8192,
    weights_config_path: str | Path | None = None,
    baseline_aggregate_path: str | Path | None = None,
) -> Any | None:
    if not enabled:
        return None
    resolved_split = Path(split_path)
    if not resolved_split.exists():
        raise FileNotFoundError(f"Missing golden eval split: {resolved_split}")

    try:
        from transformers import TrainerCallback  # type: ignore
    except Exception as exc:  # pragma: no cover - training dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before enabling golden set eval.") from exc

    class GoldenSetEvalCallback(TrainerCallback):  # type: ignore[misc]
        def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
            if model is None:
                return control
            epoch_value = getattr(state, "epoch", None)
            epoch_label = _epoch_label(epoch_value)
            epoch_dir = Path(output_dir) / f"epoch_{epoch_label}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = epoch_dir / "predictions.jsonl"
            _generate_predictions_with_model(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                split_path=resolved_split,
                output_path=predictions_path,
                max_rows=max_rows,
                max_new_tokens=max_new_tokens,
            )
            aggregate = evaluate_predictions(
                predictions_path=predictions_path,
                output_dir=epoch_dir,
                weights_config_path=weights_config_path,
                baseline_aggregate_path=baseline_aggregate_path,
            )
            aggregate["epoch"] = epoch_value
            (epoch_dir / "aggregate_metrics.json").write_text(
                json.dumps(aggregate, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return control

    return GoldenSetEvalCallback()


def _generate_predictions_with_model(
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    split_path: Path,
    output_path: Path,
    max_rows: int,
    max_new_tokens: int,
) -> int:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - training dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before running golden set eval.") from exc

    was_training = bool(getattr(model, "training", False))
    model.eval()
    rows_out: list[dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(split_path)):
        if idx >= max_rows:
            break
        prompt_text = adapter.format_example(row, tokenizer=tokenizer, include_assistant=False)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_length = inputs["input_ids"].shape[-1]
        generated = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        url_map = _extract_url_map(row)
        rows_out.append(
            {
                "id": row.get("id"),
                "response_id": row.get("response_id"),
                "response_text": restore_url_placeholders(_extract_user_text(row), url_map),
                "expected": restore_url_placeholders(_safe_load_json(row.get("completion")), url_map),
                "generated_text": generated,
                "url_map": url_map,
            }
        )
    if was_training:
        model.train()
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


def _extract_url_map(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    url_preprocessing = metadata.get("url_preprocessing") if isinstance(metadata.get("url_preprocessing"), dict) else {}
    url_map = url_preprocessing.get("url_map")
    return url_map if isinstance(url_map, dict) else {}


def _epoch_label(value: Any) -> str:
    try:
        return f"{float(value):06.2f}".replace(".", "_")
    except Exception:
        return "unknown"
