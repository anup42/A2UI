"""Small Trainer dashboards without changing the full Trainer log history."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ir_training.eval.tensorboard_logging import (
    _summary_writer_factory,
    flatten_scalar_metrics,
    resolve_tensorboard_detail,
)

_TRAINING_METRICS = frozenset({
    "loss", "learning_rate", "grad_norm", "epoch", "train_loss", "train_runtime",
    "train_samples_per_second", "train_steps_per_second", "eval_loss", "eval_runtime",
    "eval_samples_per_second", "eval_steps_per_second",
})


def select_training_tensorboard_metrics(logs: Mapping[str, Any]) -> dict[str, float]:
    """Map core HF metrics to its familiar train/ and eval/ dashboard tags.

    Golden callback metrics have their own evaluation/<cohort> writer. Sending
    eval_golden*/... through Trainer a second time would duplicate every curve.
    Console output, Trainer state and checkpoint selection are not filtered.
    """
    selected: dict[str, float] = {}
    for key, value in flatten_scalar_metrics(logs).items():
        if key not in _TRAINING_METRICS:
            continue
        if key.startswith("eval_"):
            selected[f"eval/{key[5:]}"] = value
        elif key.startswith("train_"):
            selected[f"train/{key[6:]}"] = value
        else:
            selected[f"train/{key}"] = value
    return selected


def configure_training_tensorboard(
    trainer: Any, *, log_dir: str | Path | None, detail: str | None = None,
    writer_factory: Any | None = None,
) -> bool:
    """Replace only an enabled HF TensorBoard reporter in minimal mode.

    Other reporters and the full-detail HF callback are left alone. Keeping
    this import lazy lets preparation/planning run without training packages.
    """
    if resolve_tensorboard_detail(detail) == "full":
        return False
    callbacks = getattr(getattr(trainer, "callback_handler", None), "callbacks", [])
    if not callbacks:
        return False
    from transformers import TrainerCallback
    from transformers.integrations import TensorBoardCallback

    reporters = [callback for callback in callbacks if isinstance(callback, TensorBoardCallback)]
    if not reporters:
        return False
    destination = Path(log_dir or getattr(trainer.args, "logging_dir", "") or trainer.args.output_dir)

    class ImportantTensorBoardCallback(TrainerCallback):
        def __init__(self) -> None:
            self.writer = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not getattr(state, "is_world_process_zero", True):
                return control
            selected = select_training_tensorboard_metrics(logs or {})
            if not selected:
                return control
            if self.writer is None:
                factory = writer_factory or _summary_writer_factory()
                self.writer = factory(log_dir=str(destination))
            for tag, value in selected.items():
                self.writer.add_scalar(tag, value, int(state.global_step))
            self.writer.flush()
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
                self.writer = None
            return control

    for reporter in reporters:
        previous = getattr(reporter, "tb_writer", None)
        if previous is not None:
            previous.flush()
            previous.close()
        trainer.remove_callback(reporter)
    trainer.add_callback(ImportantTensorBoardCallback())
    return True
