from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.url_preprocess import restore_url_placeholders
from ir_training.eval.compare_to_baseline import evaluate_predictions
from ir_training.eval.tensorboard_logging import log_evaluation_result


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
    required_rows: int | None = None,
    require_exact_rows: bool = False,
    require_unique_rows: bool = False,
    max_input_tokens: int | None = None,
    max_new_tokens: int = 8192,
    weights_config_path: str | Path | None = None,
    baseline_aggregate_path: str | Path | None = None,
    metric_version: str = "legacy",
    trigger: str = "epoch",
    interval: int = 1,
    metric_for_best_model: str = "overall_score",
    greater_is_better: bool = True,
    save_best_checkpoint: bool = True,
    best_checkpoint_dir: str | Path | None = None,
    metric_logger: Callable[[dict[str, float]], Any] | None = None,
    metric_log_prefix: str = "golden",
    tensorboard_root: str | Path | None = None,
    tensorboard_run_id: str | None = None,
    tensorboard_evaluation_name: str | None = None,
) -> Any | None:
    if not enabled:
        return None
    resolved_split = Path(split_path)
    if not resolved_split.exists():
        raise FileNotFoundError(f"Missing golden eval split: {resolved_split}")
    resolved_max_rows = int(max_rows)
    if resolved_max_rows < 1:
        raise ValueError("golden_eval.max_rows must be at least 1.")
    resolved_required_rows = int(required_rows) if required_rows is not None else None
    if resolved_required_rows is not None and resolved_required_rows < 1:
        raise ValueError("golden_eval.required_rows must be at least 1 when set.")
    golden_rows = _load_fixed_golden_rows(
        resolved_split,
        max_rows=resolved_max_rows,
        required_rows=resolved_required_rows,
        require_exact_rows=bool(require_exact_rows),
        require_unique_rows=bool(require_unique_rows),
    )
    golden_split_sha256 = hashlib.sha256(resolved_split.read_bytes()).hexdigest()
    resolved_metric_version = str(metric_version).strip().lower() or "legacy"
    if resolved_metric_version not in {"legacy", "v5_4", "dual"}:
        raise ValueError("golden_eval.metric_version must be 'legacy', 'v5_4', or 'dual'.")
    resolved_metric_log_prefix = _metric_prefix(metric_log_prefix)
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
                    max_rows=resolved_max_rows,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                    selected_rows=golden_rows,
                )
                if rank == 0:
                    aggregate = evaluate_predictions(
                        predictions_path=predictions_path,
                        output_dir=event_dir,
                        weights_config_path=weights_config_path,
                        baseline_aggregate_path=baseline_aggregate_path,
                        metric_version=resolved_metric_version,
                    )
                    aggregate["epoch"] = getattr(state, "epoch", None)
                    aggregate["step"] = int(getattr(state, "global_step", 0) or 0)
                    aggregate["evaluation_event"] = self.evaluation_count
                    aggregate["golden_set_rows"] = len(golden_rows)
                    aggregate["golden_set_sha256"] = golden_split_sha256
                    aggregate_path = event_dir / "aggregate_metrics.json"
                    aggregate_path.write_text(
                        json.dumps(aggregate, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    if metric_logger is not None:
                        metric_logger(
                            _golden_scalar_logs(
                                aggregate,
                                prefix=resolved_metric_log_prefix,
                            )
                        )
                    if tensorboard_root is not None:
                        log_evaluation_result(
                            tensorboard_root,
                            run_id=tensorboard_run_id or "training",
                            evaluation_name=(
                                tensorboard_evaluation_name
                                or resolved_metric_log_prefix
                            ),
                            metrics=aggregate,
                            step=int(aggregate["step"]),
                            artifacts={
                                "predictions": predictions_path,
                                "scored_predictions": event_dir
                                / "scored_predictions.jsonl",
                                "aggregate_metrics": aggregate_path,
                            },
                            metadata={
                                "event": event_label,
                                "trigger": resolved_trigger,
                                "golden_set_rows": len(golden_rows),
                                "golden_set_sha256": golden_split_sha256,
                            },
                            source_aggregate_path=aggregate_path,
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
    selected_rows: Sequence[dict[str, Any]] | None = None,
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
    rows_for_eval = (
        list(selected_rows)
        if selected_rows is not None
        else list(read_jsonl(split_path))[: max(0, int(max_rows))]
    )
    try:
        for idx in range(rank, len(rows_for_eval), world_size):
            row = rows_for_eval[idx]
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
                    "assets": row.get("assets") or [],
                    "expected_ui_contract": row.get("expected_ui_contract"),
                    "expected_ui_contract_source": row.get("expected_ui_contract_source"),
                    "expected_ui_contract_v5_4": row.get("expected_ui_contract_v5_4"),
                    "expected_ui_contract_v5_4_source": row.get("expected_ui_contract_v5_4_source"),
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


def _load_fixed_golden_rows(
    split_path: Path,
    *,
    max_rows: int,
    required_rows: int | None,
    require_exact_rows: bool,
    require_unique_rows: bool,
) -> list[dict[str, Any]]:
    all_rows = list(read_jsonl(split_path))
    selected_rows = all_rows[:max_rows]
    if required_rows is not None:
        observed = len(all_rows) if require_exact_rows else len(selected_rows)
        comparator = "exactly" if require_exact_rows else "at least"
        row_count_invalid = (
            observed != required_rows
            if require_exact_rows
            else observed < required_rows
        )
        if row_count_invalid:
            raise ValueError(
                f"Golden eval requires {comparator} {required_rows} valid rows, "
                f"but {split_path} provides {observed}."
            )
        if len(selected_rows) != required_rows:
            raise ValueError(
                f"golden_eval.max_rows={max_rows} does not select the required "
                f"{required_rows} rows from {split_path}."
            )
    if require_unique_rows:
        identities = [_golden_row_identity(row, index) for index, row in enumerate(selected_rows)]
        if len(set(identities)) != len(identities):
            raise ValueError(f"Golden eval split contains duplicate row identities: {split_path}")
    return selected_rows


def build_checkpoint_provenance_callback(
    *,
    output_dir: str | Path,
    metadata: dict[str, Any],
    config_path: str | Path | None = None,
    golden_summary_provider: Callable[[], dict[str, Any] | None] | None = None,
) -> Any:
    """Write self-contained provenance into every Trainer checkpoint.

    Interrupted runs must remain diagnosable and must not depend on metadata
    written only after ``trainer.train()`` completes. The callback runs after
    each successful Trainer save on world process zero and binds the adapter
    bytes in that exact checkpoint.
    """

    try:
        from transformers import TrainerCallback  # type: ignore
    except Exception as exc:  # pragma: no cover - training dependency path
        raise RuntimeError(
            "Install training/requirements-training.txt before training."
        ) from exc

    resolved_output = Path(output_dir)
    resolved_config = Path(config_path) if config_path is not None else None

    class CheckpointProvenanceCallback(TrainerCallback):  # type: ignore[misc]
        def on_save(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            if getattr(state, "is_world_process_zero", True) is False:
                return control
            step = int(getattr(state, "global_step", 0) or 0)
            checkpoint_dir = resolved_output / f"checkpoint-{step}"
            if not checkpoint_dir.is_dir():
                raise RuntimeError(
                    "Trainer on_save fired before checkpoint materialization: "
                    f"{checkpoint_dir}"
                )
            payload = json.loads(json.dumps(metadata, ensure_ascii=False))
            payload.update(
                {
                    "checkpoint_step": step,
                    "checkpoint_epoch": _finite_float(
                        getattr(state, "epoch", None)
                    ),
                    "last_trainer_log": (
                        dict(state.log_history[-1])
                        if getattr(state, "log_history", None)
                        else None
                    ),
                }
            )
            if golden_summary_provider is not None:
                golden_summary = golden_summary_provider()
                if golden_summary is not None:
                    payload["best_golden_eval"] = golden_summary
            _write_checkpoint_provenance(
                checkpoint_dir,
                role="trainer_intermediate",
                payload=payload,
            )
            if resolved_config is not None and resolved_config.is_file():
                _atomic_copy_file(
                    resolved_config,
                    checkpoint_dir / "training_config.yaml",
                )
            # Golden evaluation precedes Trainer saving when eval_steps and
            # save_steps coincide. Mirror self-bound provenance immediately so
            # an interrupted run never leaves its selected adapter anonymous.
            golden_summary = payload.get("best_golden_eval")
            best_dir_value = (
                golden_summary.get("checkpoint_dir")
                if isinstance(golden_summary, dict)
                else None
            )
            if best_dir_value:
                best_dir = Path(str(best_dir_value))
                if best_dir.is_dir():
                    best_payload = json.loads(
                        json.dumps(payload, ensure_ascii=False)
                    )
                    best_payload["checkpoint_step"] = int(
                        golden_summary.get("step", step) or step
                    )
                    best_payload["checkpoint_epoch"] = _finite_float(
                        golden_summary.get("epoch")
                    )
                    _write_checkpoint_provenance(
                        best_dir,
                        role="best_golden",
                        payload=best_payload,
                    )
                    if resolved_config is not None and resolved_config.is_file():
                        _atomic_copy_file(
                            resolved_config,
                            best_dir / "training_config.yaml",
                        )
            return control

    return CheckpointProvenanceCallback()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_adapter_manifest(
    checkpoint_dir: Path, *, role: str
) -> dict[str, Any]:
    adapter_files = sorted(
        candidate
        for candidate in checkpoint_dir.glob("adapter*")
        if candidate.is_file()
    )
    if not adapter_files:
        raise RuntimeError(
            f"Saved checkpoint has no local adapter files: {checkpoint_dir}"
        )
    return {
        "role": role,
        "path": str(checkpoint_dir),
        "files": [
            {
                "path": candidate.name,
                "size": int(candidate.stat().st_size),
                "sha256": _sha256_file(candidate),
            }
            for candidate in adapter_files
        ],
    }


def _write_checkpoint_provenance(
    checkpoint_dir: Path,
    *,
    role: str,
    payload: dict[str, Any],
) -> None:
    manifest = _checkpoint_adapter_manifest(checkpoint_dir, role=role)
    materialized = json.loads(json.dumps(payload, ensure_ascii=False))
    materialized.update(
        {
            "checkpoint_role": role,
            "checkpoint_dir": str(checkpoint_dir),
            "adapter_checkpoints": [manifest],
            # Retain the earlier diagnostic view while using the canonical
            # adapter_checkpoints schema consumed by merge verification.
            "checkpoint_adapter_files": [
                {
                    "path": item["path"],
                    "size_bytes": item["size"],
                    "sha256": item["sha256"],
                }
                for item in manifest["files"]
            ],
        }
    )
    _atomic_write_json(
        checkpoint_dir / "training_metadata.json", materialized
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _golden_row_identity(row: dict[str, Any], index: int) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for value in (
        row.get("source_id"),
        row.get("response_id"),
        row.get("id"),
        metadata.get("query_id"),
    ):
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"row-index:{index}"


def _metric_prefix(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    return normalized or "golden"


def _golden_scalar_logs(aggregate: dict[str, Any], *, prefix: str) -> dict[str, float]:
    logs: dict[str, float] = {}
    for key, value in aggregate.items():
        if key in {"epoch", "step", "evaluation_event"} or isinstance(value, bool):
            continue
        parsed = _finite_float(value)
        if parsed is None:
            continue
        logs[f"eval_{prefix}/{key}"] = parsed
    v5_4_score = _finite_float(aggregate.get("generation_reward_v5_4_avg"))
    if v5_4_score is not None:
        logs[f"eval_{prefix}/v5_4_score"] = v5_4_score
    return logs


def _extract_user_text(row: dict[str, Any]) -> str:
    response_text = row.get("response_text")
    if isinstance(response_text, str) and response_text.strip():
        return response_text
    # Golden rows can include few-shot demonstrations. Score against the held-
    # out request in the final user turn rather than the first demo prompt.
    for message in reversed(list(row.get("messages") or [])):
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content") or "")
            marker = "Create A2UI Express v1 GenUI IR for this response:\n\n"
            return content.split(marker, 1)[-1]
    return str(row.get("input") or row.get("prompt") or "")


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
