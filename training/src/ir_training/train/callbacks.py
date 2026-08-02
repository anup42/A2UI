from __future__ import annotations

import json
import math
import shutil
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
    max_input_tokens: int | None = None,
    max_new_tokens: int = 8192,
    weights_config_path: str | Path | None = None,
    baseline_aggregate_path: str | Path | None = None,
    trigger: str = "epoch",
    interval: int = 1,
    metric_for_best_model: str = "overall_score",
    greater_is_better: bool = True,
    save_best_checkpoint: bool = True,
    best_checkpoint_dir: str | Path | None = None,
) -> Any | None:
    if not enabled:
        return None
    resolved_split = Path(split_path)
    if not resolved_split.exists():
        raise FileNotFoundError(f"Missing golden eval split: {resolved_split}")
    resolved_trigger = str(trigger).strip().lower()
    if resolved_trigger not in {"epoch", "evaluate"}:
        raise ValueError("golden_eval.trigger must be 'epoch' or 'evaluate'.")
    if int(interval) < 1:
        raise ValueError("golden_eval.interval must be at least 1.")
    resolved_output_dir = Path(output_dir)
    resolved_best_checkpoint_dir = (
        Path(best_checkpoint_dir)
        if best_checkpoint_dir is not None
        else resolved_output_dir / "best_golden_checkpoint"
    )

    try:
        from transformers import TrainerCallback  # type: ignore
    except Exception as exc:  # pragma: no cover - training dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before enabling golden set eval.") from exc

    class GoldenSetEvalCallback(TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self.evaluation_count = 0
            self.best_metric_value = -math.inf if greater_is_better else math.inf
            self.best_metric_step: int | None = None
            self.best_metric_epoch: float | None = None
            self.best_checkpoint_saved = False

        def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
            if resolved_trigger != "epoch":
                return control
            return self._run_if_due(state=state, control=control, model=model)

        def on_evaluate(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
            if resolved_trigger != "evaluate":
                return control
            return self._run_if_due(state=state, control=control, model=model)

        def _run_if_due(self, *, state: Any, control: Any, model: Any = None) -> Any:
            if model is None:
                return control
            self.evaluation_count += 1
            if self.evaluation_count % int(interval) != 0:
                return control

            rank, _ = _distributed_context()
            event_label = _evaluation_label(resolved_trigger, state)
            event_dir = resolved_output_dir / event_label
            predictions_path = event_dir / "predictions.jsonl"
            try:
                _generate_predictions_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    split_path=resolved_split,
                    output_path=predictions_path,
                    max_rows=max_rows,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                )
                if rank == 0:
                    aggregate = evaluate_predictions(
                        predictions_path=predictions_path,
                        output_dir=event_dir,
                        weights_config_path=weights_config_path,
                        baseline_aggregate_path=baseline_aggregate_path,
                    )
                    aggregate["epoch"] = getattr(state, "epoch", None)
                    aggregate["step"] = int(getattr(state, "global_step", 0) or 0)
                    aggregate["evaluation_event"] = self.evaluation_count
                    aggregate_path = event_dir / "aggregate_metrics.json"
                    aggregate_path.write_text(
                        json.dumps(aggregate, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    self._record_best_if_improved(
                        model=model,
                        state=state,
                        event_label=event_label,
                        event_dir=event_dir,
                        aggregate=aggregate,
                    )
            finally:
                _distributed_barrier()
            return control

        def _record_best_if_improved(
            self,
            *,
            model: Any,
            state: Any,
            event_label: str,
            event_dir: Path,
            aggregate: dict[str, Any],
        ) -> None:
            metric_value = _finite_float(aggregate.get(metric_for_best_model))
            if metric_value is None:
                print(
                    f"Golden eval metric {metric_for_best_model!r} was not produced; "
                    "best-checkpoint selection was skipped.",
                    flush=True,
                )
                return
            if not _metric_improved(metric_value, self.best_metric_value, greater_is_better):
                return

            self.best_metric_value = metric_value
            self.best_metric_step = int(getattr(state, "global_step", 0) or 0)
            epoch_value = _finite_float(getattr(state, "epoch", None))
            self.best_metric_epoch = epoch_value
            best_info = {
                "metric": metric_for_best_model,
                "metric_value": metric_value,
                "greater_is_better": greater_is_better,
                "step": self.best_metric_step,
                "epoch": epoch_value,
                "evaluation_event": self.evaluation_count,
                "evaluation_dir": str(event_dir),
                "checkpoint_dir": str(resolved_best_checkpoint_dir) if save_best_checkpoint else None,
            }
            resolved_output_dir.mkdir(parents=True, exist_ok=True)
            (resolved_output_dir / "best_golden_eval.json").write_text(
                json.dumps(best_info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if save_best_checkpoint:
                _save_best_golden_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint_dir=resolved_best_checkpoint_dir,
                    event_dir=event_dir,
                    best_info=best_info,
                )
                self.best_checkpoint_saved = True
            print(
                f"New best golden eval {metric_for_best_model}={metric_value:.6f} "
                f"at {event_label}.",
                flush=True,
            )

        def summary(self) -> dict[str, Any] | None:
            if self.best_metric_step is None:
                return None
            return {
                "metric": metric_for_best_model,
                "metric_value": self.best_metric_value,
                "greater_is_better": greater_is_better,
                "step": self.best_metric_step,
                "epoch": self.best_metric_epoch,
                "checkpoint_dir": str(resolved_best_checkpoint_dir) if self.best_checkpoint_saved else None,
            }

    return GoldenSetEvalCallback()


def _generate_predictions_with_model(
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    split_path: Path,
    output_path: Path,
    max_rows: int,
    max_input_tokens: int | None,
    max_new_tokens: int,
) -> int:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - training dependency failure path
        raise RuntimeError("Install training/requirements-training.txt before running golden set eval.") from exc

    rank, world_size = _distributed_context()
    generation_model = _unwrap_model(model)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    rows_out: list[dict[str, Any]] = []
    selected_rows = list(read_jsonl(split_path))[: max(0, int(max_rows))]
    try:
        for idx in range(rank, len(selected_rows), world_size):
            row = selected_rows[idx]
            prompt_text = adapter.format_example(row, tokenizer=tokenizer, include_assistant=False)
            tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
            if max_input_tokens is not None and int(max_input_tokens) > 0:
                tokenizer_kwargs.update({"truncation": True, "max_length": int(max_input_tokens)})
            inputs = tokenizer(prompt_text, **tokenizer_kwargs)
            device = _model_device(generation_model)
            if device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(device)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
            }
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            with torch.no_grad():
                output = generation_model.generate(**inputs, **generation_kwargs)
            input_length = inputs["input_ids"].shape[-1]
            generated = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
            url_map = _extract_url_map(row)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            rows_out.append(
                {
                    "id": row.get("id"),
                    "ui_id": row.get("ui_id") or metadata.get("ui_id"),
                    "response_id": row.get("response_id"),
                    "query_id": row.get("query_id") or metadata.get("query_id"),
                    "intent": row.get("intent") or metadata.get("intent"),
                    "intent_bucket": row.get("intent_bucket") or metadata.get("intent_bucket"),
                    "tags": row.get("tags") or metadata.get("tags"),
                    "response_text": restore_url_placeholders(_extract_user_text(row), url_map),
                    "expected": restore_url_placeholders(_extract_expected_completion(row), url_map),
                    "generated_text": generated,
                    "url_map": url_map,
                    "_golden_index": idx,
                }
            )
    finally:
        if was_training:
            model.train()

    gathered_rows = _gather_prediction_rows(rows_out, world_size=world_size)
    if rank != 0:
        return 0
    gathered_rows.sort(key=lambda row: int(row.get("_golden_index", 0)))
    for row in gathered_rows:
        row.pop("_golden_index", None)
    return write_jsonl(output_path, gathered_rows)


def _extract_user_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content") or "")
            marker = "Create A2UI Express v1 GenUI IR for this response:\n\n"
            return content.split(marker, 1)[-1]
    return str(row.get("prompt") or "")


def _extract_expected_completion(row: dict[str, Any]) -> Any:
    completion = row.get("completion")
    if not isinstance(completion, str) or not completion.strip():
        targets = row.get("completion_targets")
        if isinstance(targets, dict):
            completion = targets.get("a2ui_express_v1")
    # Checkpoint selection is an active Express evaluation path. Do not parse
    # a legacy JSON/FlatSpec completion here; offline comparisons use a separate
    # evaluator and must opt into their own source boundary.
    return str(completion or "")


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


def _evaluation_label(trigger: str, state: Any) -> str:
    if trigger == "evaluate":
        return f"step_{int(getattr(state, 'global_step', 0) or 0):09d}"
    return f"epoch_{_epoch_label(getattr(state, 'epoch', None))}"


def _distributed_context() -> tuple[int, int]:
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
    except Exception:
        pass
    return 0, 1


def _distributed_barrier() -> None:
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def _gather_prediction_rows(rows: list[dict[str, Any]], *, world_size: int) -> list[dict[str, Any]]:
    if world_size <= 1:
        return rows
    import torch.distributed as dist  # type: ignore

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, rows)
    combined: list[dict[str, Any]] = []
    for shard in gathered:
        if not isinstance(shard, list):
            continue
        combined.extend(row for row in shard if isinstance(row, dict))
    return combined


def _unwrap_model(model: Any) -> Any:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "module", None)
        if wrapped is None:
            break
        current = wrapped
    return current


def _model_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _metric_improved(current: float, best: float, greater_is_better: bool) -> bool:
    return current > best if greater_is_better else current < best


def _save_best_golden_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    checkpoint_dir: Path,
    event_dir: Path,
    best_info: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _unwrap_model(model).save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    for name in ("predictions.jsonl", "scored_predictions.jsonl", "aggregate_metrics.json"):
        source = event_dir / name
        if source.exists():
            shutil.copy2(source, checkpoint_dir / name)
    (checkpoint_dir / "best_metric_info.json").write_text(
        json.dumps(best_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
