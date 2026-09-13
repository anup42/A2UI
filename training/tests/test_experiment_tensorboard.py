"""Real event-file integration; synthetic metrics, no model or GPU is used.

This test intentionally skips when the optional training-host TensorBoard
dependency is absent. It must not silently substitute a mock writer: passing
means actual HParams protobuf metadata and scalar events survived a roundtrip.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))

from ir_training.eval.tensorboard_logging import _summary_writer_factory
from ir_training.pipeline.experiments import _record_comparison


def _fixture_trial(name, *, learning_rate, score, loss):
    """All numbers below are arbitrary fixture data, never measured quality."""
    return {
        "name": name,
        "parameters": {"learning_rate": learning_rate, "weight_decay": 0.01,
                       "warmup_ratio": 0.03, "augmentation": "none"},
        "selection_score": score, "final_golden32_score": score - 0.01,
        "elapsed_seconds": 12.5,
        "evaluations": {
            "best_golden32": {"aggregate": {"fixture_score": score, "fixture_loss": loss}},
            "final_golden32": {"aggregate": {"fixture_score": score - 0.01, "fixture_loss": loss + 0.05}},
        },
        "fixture_notice": "Synthetic test data only; no training, inference or benchmark evaluation occurred.",
    }


def test_real_summary_writer_hparams_and_comparison_events_roundtrip(tmp_path):
    pytest.importorskip("tensorboard", reason="Real TensorBoard event roundtrip requires the training-host tensorboard package")
    pytest.importorskip("torch", reason="Real TensorBoard writer integration requires torch")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.plugins.hparams import metadata
    from tensorboard.plugins.hparams.api_pb2 import Status
    from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData

    root = tmp_path / "tensorboard"
    suite = root / "experiments" / "fixture_suite"
    budget = 20
    trials = [
        _fixture_trial("fixture_baseline", learning_rate=1e-4, score=0.31, loss=0.75),
        _fixture_trial("fixture_lower_lr", learning_rate=5e-5, score=0.37, loss=0.65),
    ]
    factory = _summary_writer_factory()
    with factory(log_dir=str(suite)) as writer:
        for trial in trials:
            _record_comparison(writer, trial, budget=budget)

    parent = EventAccumulator(str(suite), size_guidance={"scalars": 0, "tensors": 0}).Reload()
    # Suite-level comparison curves have distinct trial + checkpoint-role tags.
    # fixture_loss deliberately tests arbitrary loss scalar forwarding; it is
    # not a claim that Golden evaluation produces ordinary validation loss.
    for trial in trials:
        for role, result in trial["evaluations"].items():
            for metric, expected in result["aggregate"].items():
                events = parent.Scalars(f"comparison/{trial['name']}/{role}/{metric}")
                assert len(events) == 1
                assert events[0].step == budget
                assert events[0].value == pytest.approx(expected)
        assert f"comparison/{trial['name']}/summary/text_summary" in parent.Tags()["tensors"]

    # HParams uses one actual child event run per trial, within the supplied
    # root, so neither metric scalars nor session metadata overwrite each other.
    assert {path.name for path in suite.iterdir() if path.is_dir()} == {trial["name"] for trial in trials}
    for trial in trials:
        child_path = suite / trial["name"]
        assert child_path.resolve().is_relative_to(root.resolve())
        child = EventAccumulator(str(child_path), size_guidance={"scalars": 0, "tensors": 0}).Reload()
        metadata_bytes = child.PluginTagToContent(metadata.PLUGIN_NAME)
        assert {metadata.EXPERIMENT_TAG, metadata.SESSION_START_INFO_TAG,
                metadata.SESSION_END_INFO_TAG} <= set(metadata_bytes)
        start = HParamsPluginData.FromString(metadata_bytes[metadata.SESSION_START_INFO_TAG]).session_start_info
        assert start.hparams["learning_rate"].number_value == pytest.approx(trial["parameters"]["learning_rate"])
        assert start.hparams["augmentation"].string_value == "none"
        assert start.hparams["optimizer_steps"].number_value == budget
        end = HParamsPluginData.FromString(metadata_bytes[metadata.SESSION_END_INFO_TAG]).session_end_info
        assert end.status == Status.Value("STATUS_SUCCESS")
        expected_metrics = {
            "hparam/best_golden32": trial["selection_score"],
            "hparam/final_golden32": trial["final_golden32_score"],
            "hparam/pipeline_elapsed_seconds": trial["elapsed_seconds"],
        }
        experiment = HParamsPluginData.FromString(metadata_bytes[metadata.EXPERIMENT_TAG]).experiment
        assert {item.name.tag for item in experiment.metric_infos} == set(expected_metrics)
        for tag, expected in expected_metrics.items():
            events = child.Scalars(tag)
            assert len(events) == 1 and events[0].step == budget
            assert events[0].value == pytest.approx(expected)
        assert not any(tag.startswith("comparison/") for tag in child.Tags()["scalars"])
    assert {path.parent for path in root.rglob("events.out.tfevents.*")} == {
        suite, *(suite / trial["name"] for trial in trials),
    }
