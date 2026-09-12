from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset" / "src"))

from ir_training.eval.generate import aggregate_generation_performance, build_prediction_record, place_model_for_generation
from ir_training.generation_policy import generation_cache_scope
from ir_training.train import callbacks


def test_cache_is_enabled_only_in_generation_and_restored_after_failure():
    text = SimpleNamespace(use_cache=False)
    config = SimpleNamespace(use_cache=False, text_config=text)
    generation = SimpleNamespace()
    model = SimpleNamespace(config=config, generation_config=generation)
    with pytest.raises(RuntimeError, match="fixture failure"):
        with generation_cache_scope(model):
            assert config.use_cache and text.use_cache and generation.use_cache
            raise RuntimeError("fixture failure")
    assert config.use_cache is False and text.use_cache is False
    assert not hasattr(generation, "use_cache")


def test_standalone_generation_places_unmapped_training_model_on_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = SimpleNamespace(device=torch.device("cpu"))
    placements = []

    def move(device):
        placements.append(str(device))
        model.device = device

    model.to = move
    assert place_model_for_generation(model, {"device_map": "none"}) == "cuda"
    assert placements == ["cuda"]
    model.hf_device_map = {"decoder": "cuda:0", "head": "cpu"}
    assert place_model_for_generation(model, {}) == "cuda"
    assert placements == ["cuda"]
    with pytest.raises(ValueError, match="active HF device map"):
        place_model_for_generation(model, {"inference_device": "cpu"})


def _callback_fixture(tmp_path, monkeypatch, *, scores, **options):
    split = tmp_path / "golden.jsonl"
    if not split.exists():
        split.write_text(json.dumps({"id": "one"}) + "\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))
    monkeypatch.setattr(callbacks, "_distributed_context", lambda: (0, 1))
    monkeypatch.setattr(callbacks, "_distributed_barrier", lambda: None)
    calls = []
    logged = []
    scores = iter(scores)

    def generate(**kwargs):
        calls.append(kwargs)
        path = kwargs["output_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"id":"one"}\n', encoding="utf-8")
        kwargs["performance_metrics"].update(
            evaluation_generated_tokens=20, evaluation_generation_seconds=2.0,
            evaluation_output_tokens_per_second=10.0)
        return 1

    def score(**kwargs):
        return {"count": 1, "generation_reward_v5_4_avg": next(scores)}

    monkeypatch.setattr(callbacks, "_generate_predictions_with_model", generate)
    monkeypatch.setattr(callbacks, "evaluate_predictions", score)

    class Model:
        def save_pretrained(self, directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / "adapter_config.json").write_text("{}")
            (Path(directory) / "adapter_model.safetensors").write_bytes(b"fixture-best-weights")

    class Tokenizer:
        def save_pretrained(self, directory):
            (Path(directory) / "tokenizer_config.json").write_text("{}")

    callback = callbacks.build_golden_set_eval_callback(
        enabled=True, split_path=split, output_dir=tmp_path / "eval",
        adapter=SimpleNamespace(config={}), tokenizer=Tokenizer(), max_rows=1,
        required_rows=1, trigger="evaluate", interval=2,
        metric_for_best_model="generation_reward_v5_4_avg",
        best_checkpoint_dir=tmp_path / "best", metric_logger=logged.append,
        **options,
    )
    return callback, Model(), calls, logged


def test_final_golden_runs_after_skipped_interval_without_repeating_same_step(tmp_path, monkeypatch):
    callback, model, calls, logged = _callback_fixture(tmp_path, monkeypatch, scores=[42])
    state = SimpleNamespace(global_step=20, epoch=1.0)
    callback.on_evaluate(None, state, None, model=model)
    assert calls == []
    callback.on_train_end(None, state, None, model=model)
    callback.on_train_end(None, state, None, model=model)
    callback.on_evaluate(None, state, None, model=model)
    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 2048 and calls[0]["use_cache"] is True
    assert callback.summary()["step"] == 20
    assert logged[0]["eval_golden/evaluation_output_tokens_per_second"] == 10
    assert logged[0]["eval_golden/evaluation_pause_seconds"] >= 0


def test_resume_keeps_best_and_cadence_and_rejects_changed_benchmark(tmp_path, monkeypatch):
    callback, model, calls, _ = _callback_fixture(tmp_path, monkeypatch, scores=[42])
    callback.on_evaluate(None, SimpleNamespace(global_step=10, epoch=0.5), None, model=model)
    state = SimpleNamespace(global_step=20, epoch=1.0)
    callback.on_evaluate(None, state, None, model=model)
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    callback.on_save(SimpleNamespace(output_dir=tmp_path), state, None)

    resumed, model, calls, _ = _callback_fixture(tmp_path, monkeypatch, scores=[10], resume_checkpoint=checkpoint)
    resumed.on_evaluate(None, SimpleNamespace(global_step=30, epoch=1.5), None, model=model)
    assert calls == []
    resumed.on_evaluate(None, SimpleNamespace(global_step=40, epoch=2.0), None, model=model)
    assert len(calls) == 1
    assert resumed.summary()["metric_value"] == 42
    assert resumed.summary()["step"] == 20
    (tmp_path / "golden.jsonl").write_text('{"id":"changed"}\n')
    with pytest.raises(ValueError, match="contract changed"):
        _callback_fixture(tmp_path, monkeypatch, scores=[], resume_checkpoint=checkpoint)


def test_prediction_metadata_binds_explicit_repeated_occurrences():
    row = {"id": "one", "response_text": "source", "completion": "expected"}
    baseline = build_prediction_record(row, "generated")
    assert "benchmark" not in baseline
    repeated = copy.deepcopy(row)
    repeated["metadata"] = {"benchmark": {"benchmark_kind": "explicit_repeated_case"},
        "repeated_from": "old-one", "replaces": "invalid-row"}
    result = build_prediction_record(repeated, "generated")
    assert result["repeated_from"] == "old-one" and result["replaces"] == "invalid-row"
    assert result["benchmark"]["benchmark_kind"] == "explicit_repeated_case"
    assert result["source_context_sha256"] != baseline["source_context_sha256"]
    assert build_prediction_record(row, "generated")["source_context_sha256"] == baseline["source_context_sha256"]


@pytest.mark.parametrize("fail", [False, True])
def test_periodic_generation_uses_cache_and_restores_training_configuration(tmp_path, monkeypatch, fail):
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 7
        pad_token_id = 7

        def __call__(self, text, **kwargs):
            return Batch(input_ids=torch.tensor([[1, 2]]))

        def decode(self, ids, **kwargs):
            return '<a2ui>root=Text("ok")</a2ui>'

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.training = True
            self.config = SimpleNamespace(use_cache=False, eos_token_id=7)
            self.generation_config = SimpleNamespace(use_cache=False, eos_token_id=7)

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def generate(self, **kwargs):
            assert kwargs["use_cache"] and self.config.use_cache and self.generation_config.use_cache
            assert kwargs["synced_gpus"] is False and self.training is False
            if fail:
                raise RuntimeError("fixture decoding failed")
            return torch.tensor([[1, 2, 3, 7]])

    monkeypatch.setattr(callbacks, "_distributed_context", lambda: (0, 1))
    monkeypatch.setattr(callbacks, "_build_stop_string_criteria", lambda *args, **kwargs: None)
    model = Model()
    metrics = {}
    arguments = dict(model=model, tokenizer=Tokenizer(),
        adapter=SimpleNamespace(format_example=lambda *args, **kwargs: "prompt"),
        split_path=tmp_path / "unused.jsonl", output_path=tmp_path / "predictions.jsonl",
        max_rows=1, max_input_tokens=10, max_new_tokens=2048,
        selected_rows=[{"id": "one", "response_text": "source", "completion": "expected"}],
        performance_metrics=metrics)
    if fail:
        with pytest.raises(RuntimeError, match="fixture decoding failed"):
            callbacks._generate_predictions_with_model(**arguments)
    else:
        assert callbacks._generate_predictions_with_model(**arguments) == 1
        assert metrics["evaluation_generated_tokens"] == 2
        assert metrics["evaluation_input_tokens"] == 2
        assert metrics["evaluation_output_tokens_per_second"] > 0
    assert model.training is True
    assert model.config.use_cache is False and model.generation_config.use_cache is False


def test_distributed_rank_failure_is_raised_on_other_rank(monkeypatch):
    import torch.distributed as dist

    def gather(destination, local):
        assert local is None
        destination[:] = [None, "RuntimeError: rank one decoding failed"]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank one decoding failed"):
        callbacks._raise_distributed_evaluation_error(None, world_size=2)


def test_runtime_summary_uses_measured_row_times_and_reports_coverage():
    result = aggregate_generation_performance([
        {"runtime": {"generation_seconds": 2.0, "output_tokens": 10}},
        {"runtime": {"generation_seconds": 1.0, "output_tokens": 20}},
        {"runtime": {"generation_seconds": float("nan"), "output_tokens": 100}},
        {},
    ])
    assert result["generation_runtime_measured_rows"] == 2
    assert result["generation_runtime_row_seconds_sum"] == 3
    assert result["generation_runtime_output_tokens"] == 30
    assert result["generation_runtime_tokens_per_row_second"] == 10
    assert aggregate_generation_performance([{}]) == {}


def test_external_runner_rejects_old_outputs_before_starting_process(tmp_path, monkeypatch):
    from ir_training.eval import external_runner

    model = tmp_path / "changed-model.litertlm"
    model.write_bytes(b"new model")
    split = tmp_path / "golden.jsonl"
    split.write_text('{"id":"one"}\n')
    output = tmp_path / "eval"
    output.mkdir()
    stale = output / "runner_outputs.jsonl"
    stale.write_text('{"id":"one","generated_text":"old model result"}\n')
    calls = []
    monkeypatch.setattr(external_runner.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(FileExistsError, match="existing external runner output"):
        external_runner.run_external_generation(command_template=["runner"],
            model_path=model, split_path=split, output_dir=output,
            max_input_tokens=4096, max_new_tokens=2048, required_rows=1)
    assert calls == []
    assert "old model result" in stale.read_text()


def test_checkpoint_steps_use_selected_metadata_and_never_default_to_zero(tmp_path):
    spec = importlib.util.spec_from_file_location("golden_checkpoint_cli", ROOT / "scripts/evaluate_checkpoint_on_golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "best_golden_checkpoint"
    checkpoint.mkdir()
    metadata = checkpoint / "training_metadata.json"
    metadata.write_text(json.dumps({"best_golden_eval": {"step": 20000}, "checkpoint_step": 22000}))
    assert module._checkpoint_step(checkpoint) == 20000
    intermediate = tmp_path / "checkpoint-22000"
    intermediate.mkdir()
    (intermediate / "training_metadata.json").write_text(metadata.read_text())
    assert module._checkpoint_step(intermediate) == 22000
    unknown = tmp_path / "anonymous_model"
    unknown.mkdir()
    with pytest.raises(ValueError, match="provide --step explicitly"):
        module._checkpoint_step(unknown)


def test_finalization_keeps_final_and_selected_checkpoint_provenance_distinct(tmp_path):
    from ir_training.train.sft import _write_final_checkpoint_metadata
    from ir_training.export.merge_lora import (
        _adapter_file_records, _checkpoint_manifest_matches_adapter, _normalized_file_records,
    )

    paths = []
    for role, name in (("final", "final_adapter"), ("best_golden", "best_golden_checkpoint")):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "adapter_model.safetensors").write_bytes(name.encode())
        (checkpoint / "adapter_config.json").write_text("{}")
        (checkpoint / "tokenizer_config.json").write_text("{}")
        (checkpoint / "added_tokens.json").write_bytes(b"")
        paths.append((role, checkpoint))
    metadata = {"training_metadata_version": 4,
        "best_golden_eval": {"step": 20, "epoch": 0.5, "metric_value": 42}}
    _write_final_checkpoint_metadata(output_dir=tmp_path, metadata=metadata,
        checkpoint_paths=paths, final_step=40, final_epoch=1.0, config_path=None)
    final = json.loads((paths[0][1] / "training_metadata.json").read_text())
    best = json.loads((paths[1][1] / "training_metadata.json").read_text())
    assert final["checkpoint_role"] == "final" and final["checkpoint_step"] == 40
    assert best["checkpoint_role"] == "best_golden" and best["checkpoint_step"] == 20
    assert best["checkpoint_epoch"] == 0.5
    assert len(final["adapter_checkpoints"]) == len(best["adapter_checkpoints"]) == 1
    assert final["adapter_checkpoints"][0]["role"] == "final"
    assert best["adapter_checkpoints"][0]["role"] == "best_golden"
    selected_dir = paths[1][1]
    actual = _normalized_file_records(_adapter_file_records(selected_dir))
    records = best["adapter_checkpoints"][0]["files"]
    assert _checkpoint_manifest_matches_adapter(records, adapter_path=selected_dir, actual_adapter_files=actual)
    (selected_dir / "tokenizer_config.json").write_text('{"changed":true}')
    assert not _checkpoint_manifest_matches_adapter(records, adapter_path=selected_dir, actual_adapter_files=actual)
