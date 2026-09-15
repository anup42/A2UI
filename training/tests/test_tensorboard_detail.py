from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.eval.tensorboard_logging import (
    log_evaluation_result,
    resolve_tensorboard_detail,
    select_tensorboard_metrics,
)
from ir_training.train.tensorboard_callback import (
    configure_training_tensorboard,
    select_training_tensorboard_metrics,
)


def test_minimal_selection_keeps_headlines_and_unique_source_selection(monkeypatch):
    monkeypatch.delenv("A2UI_TENSORBOARD_DETAIL", raising=False)
    metrics = {
        "count": 32,
        "generation_reward_v5_4": 70,
        "generation_reward_v5_4_avg": 70,
        "unique_source_generation_reward_v5_4_avg": 69,
        "schema_valid_strict_rate": 0.9,
        "generation_runtime_gpu_count": 8,
        "generation_runtime_wall_output_tokens_per_second": 51,
        "raw_diagnostics": {"generation_reward_v5_4_avg": 70},
        "metric_observed_counts": {"native_syntax_valid": 32},
        "unique_source_metrics": {"generation_reward_v5_4_avg": 69},
        "metric_weights": {"depth": 1.0},
    }
    result = select_tensorboard_metrics(metrics)
    assert result == {key: metrics[key] for key in (
        "count", "generation_reward_v5_4_avg", "unique_source_generation_reward_v5_4_avg",
        "schema_valid_strict_rate", "generation_runtime_gpu_count",
        "generation_runtime_wall_output_tokens_per_second",
    )}
    assert metrics["raw_diagnostics"]["generation_reward_v5_4_avg"] == 70


def test_full_detail_is_explicit_and_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("A2UI_TENSORBOARD_DETAIL", "full")
    metrics = {"debug": {"value": 1}, "not_finite": float("inf")}
    assert resolve_tensorboard_detail() == "full"
    assert select_tensorboard_metrics(metrics) == {"debug/value": 1}
    assert select_tensorboard_metrics(metrics, detail="minimal") == {}
    with pytest.raises(ValueError, match="TensorBoard detail"):
        resolve_tensorboard_detail("typo")


class RecordingWriter:
    instances: ClassVar[list] = []

    def __init__(self, *, log_dir):
        self.log_dir = log_dir
        self.scalars = []
        self.text = []
        self.closed = False
        self.instances.append(self)

    def add_scalar(self, *args):
        self.scalars.append(args)

    def add_text(self, *args):
        self.text.append(args)

    def flush(self):
        pass

    def close(self):
        self.closed = True


@pytest.mark.parametrize("detail", ["minimal", "full"])
def test_sidecar_retains_all_evidence_independent_of_dashboard(detail, tmp_path, monkeypatch):
    monkeypatch.delenv("A2UI_TENSORBOARD_ROOT", raising=False)
    metrics = {"count": 35, "generation_reward_v5_4_avg": 90, "raw_diagnostics": {"count": 35}, "label": "all evidence"}
    result = log_evaluation_result(
        tmp_path, run_id="fixture", evaluation_name="w8_golden35", metrics=metrics,
        metadata={"request_hash": "source-bound"}, detail=detail, writer_factory=RecordingWriter,
    )
    record = json.loads(Path(result["record_path"]).read_text(encoding="utf-8"))
    assert record["metrics"] == metrics
    assert record["metadata"] == {"request_hash": "source-bound"}
    assert record["tensorboard_detail"] == detail
    writer = RecordingWriter.instances[-1]
    assert bool(writer.text) == (detail == "full")
    assert len(writer.scalars) == (2 if detail == "minimal" else 3)
    assert writer.closed


def test_training_log_filter_preserves_original_and_avoids_duplicate_golden_curves():
    logs = {"loss": 0.4, "learning_rate": 2e-5, "epoch": 0.5, "eval_loss": 0.45,
            "train_runtime": 123, "grad_norm": 0.3, "total_flos": 100000,
            "eval_golden32/generation_reward_v5_4_avg": 80,
            "eval_golden32/v5_4_score": 80}
    before = dict(logs)
    assert select_training_tensorboard_metrics(logs) == {
        "train/loss": 0.4, "train/learning_rate": 2e-5, "train/epoch": 0.5,
        "eval/loss": 0.45, "train/runtime": 123, "train/grad_norm": 0.3,
    }
    assert logs == before


def test_real_hf_callback_replacement_is_rank_zero_only_and_keeps_other_reporters(tmp_path):
    pytest.importorskip("transformers")
    from transformers.integrations import TensorBoardCallback

    original = TensorBoardCallback()
    other = object()
    callbacks = [original, other]
    trainer = SimpleNamespace(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        callback_handler=SimpleNamespace(callbacks=callbacks),
        remove_callback=callbacks.remove, add_callback=callbacks.append,
    )
    assert configure_training_tensorboard(trainer, log_dir=tmp_path, detail="minimal", writer_factory=RecordingWriter)
    assert original not in callbacks and other in callbacks
    callback = next(item for item in callbacks if item is not other)
    initial_writers = len(RecordingWriter.instances)
    state = SimpleNamespace(is_world_process_zero=False, global_step=50)
    callback.on_log(trainer.args, state, None, logs={"loss": 0.5})
    assert len(RecordingWriter.instances) == initial_writers
    state.is_world_process_zero = True
    callback.on_log(trainer.args, state, None, logs={"eval_golden32/count": 32})
    assert len(RecordingWriter.instances) == initial_writers
    callback.on_log(trainer.args, state, None, logs={"loss": 0.5})
    writer = RecordingWriter.instances[-1]
    assert writer.scalars == [("train/loss", 0.5, 50)]
    assert not writer.text
    callback.on_train_end(trainer.args, state, None)
    assert writer.closed


def test_full_detail_does_not_replace_reporters_and_disabled_logging_stays_disabled():
    reporter = object()
    trainer = SimpleNamespace(callback_handler=SimpleNamespace(callbacks=[reporter]))
    assert not configure_training_tensorboard(trainer, log_dir=None, detail="full")
    assert trainer.callback_handler.callbacks == [reporter]
    assert not configure_training_tensorboard(SimpleNamespace(), log_dir=None, detail="minimal")


def test_shared_cli_supports_minimal_default_and_full_opt_in(monkeypatch):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/run_golden_training.py"
    spec = importlib.util.spec_from_file_location("tensorboard_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("A2UI_TENSORBOARD_DETAIL", raising=False)
    for deployment in (False, True):
        parser = module.build_parser(for_deployment=deployment)
        arguments = ["--model-dir", "model", "--output-dir", "out"]
        assert parser.parse_args(arguments).tensorboard_detail == "minimal"
        assert parser.parse_args([*arguments, "--tensorboard-detail", "full"]).tensorboard_detail == "full"
