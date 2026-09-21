from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ir_training.train import callbacks
from ir_training.train import zero3_checkpoint as checkpointing


class _LogicalModel:
    def __init__(self) -> None:
        self.weight = torch.ones(2, 3, dtype=torch.float32)
        self.alias = self.weight
        self.buffer = torch.arange(2, dtype=torch.float32)

    def state_dict(self):
        return {"weight": self.weight, "alias": self.alias, "buffer": self.buffer}

    def named_parameters(self, remove_duplicate=False):
        return iter(())


def _full_state():
    weight = torch.ones(2, 3, dtype=torch.float32)
    return {
        "weight": weight,
        "alias": weight,
        "buffer": torch.arange(2, dtype=torch.float32),
    }


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda state: state.pop("buffer"), "missing"),
        (lambda state: state.__setitem__("weight", torch.empty(0)), "placeholder"),
        (lambda state: state.__setitem__("weight", state["weight"].half()), "not FP32"),
    ],
)
def test_zero3_full_state_rejects_incomplete_or_non_fp32(mutation, match):
    state = _full_state()
    mutation(state)
    with pytest.raises(checkpointing.Zero3CheckpointError, match=match):
        checkpointing.validate_zero3_full_state(state, _LogicalModel())


def test_zero3_save_uses_collective_state_and_rank_zero_private_save(tmp_path: Path):
    calls = []

    class Accelerator:
        def get_state_dict(self, engine):
            calls.append(("state", engine))
            return _full_state()

    class Trainer:
        deepspeed = SimpleNamespace(module=_LogicalModel(), zero_optimization_stage=lambda: 3)
        accelerator = Accelerator()
        processing_class = None

        def _save(self, output_dir, state_dict):
            calls.append(("save", set(state_dict)))
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "model.safetensors").write_bytes(b"dense")

    evidence = checkpointing.save_zero3_checkpoint(Trainer(), tmp_path)
    assert [item[0] for item in calls] == ["state", "save"]
    assert evidence["verified"] is True
    assert evidence["state_tensor_count"] == 3


def test_zero3_nonwriter_still_consolidates_and_does_not_write(monkeypatch, tmp_path: Path):
    calls = []
    evidence = {"verified": True, "state_tensor_count": 3}

    class Dist:
        def all_gather_object(self, gathered, value):
            gathered[:] = [None, None]

        def broadcast_object_list(self, payload, src=0):
            payload[0] = evidence

        def barrier(self):
            calls.append("barrier")

    class Accelerator:
        def get_state_dict(self, engine):
            calls.append("state")
            return {}

    trainer = SimpleNamespace(
        deepspeed=SimpleNamespace(module=_LogicalModel(), zero_optimization_stage=lambda: 3),
        accelerator=Accelerator(),
        processing_class=None,
    )
    monkeypatch.setattr(checkpointing, "_distributed_context", lambda: (1, 2, Dist()))
    assert checkpointing.save_zero3_checkpoint(trainer, tmp_path) == evidence
    assert calls == ["state", "barrier"]
    assert list(tmp_path.iterdir()) == []


def test_importing_zero3_helper_does_not_import_deepspeed():
    # The module deliberately relies on Trainer/Accelerate public objects and
    # remains importable in CPU-only validation environments.
    assert "deepspeed" not in checkpointing.__dict__


def test_zero3_validation_uses_parameter_ds_shape_not_empty_state_view():
    class Partitioned(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.float32))
            self.weight.ds_shape = torch.Size([2, 3])
            self.register_buffer("scale", torch.ones(2, dtype=torch.float32))

    full = {
        "weight": torch.ones(2, 3, dtype=torch.float32),
        "scale": torch.ones(2, dtype=torch.float32),
    }
    evidence = checkpointing.validate_zero3_full_state(full, Partitioned())
    assert evidence["verified"] is True
    assert evidence["state_tensor_count"] == 2


def test_zero3_save_rejects_non_stage3_engine(tmp_path: Path):
    trainer = SimpleNamespace(
        deepspeed=SimpleNamespace(module=_LogicalModel(), zero_optimization_stage=lambda: 2),
        accelerator=SimpleNamespace(get_state_dict=lambda engine: _full_state()),
    )
    with pytest.raises(checkpointing.Zero3CheckpointError, match="stage 3"):
        checkpointing.save_zero3_checkpoint(trainer, tmp_path)


@pytest.mark.parametrize("rank,fail_write", [(0, False), (1, False), (0, True)])
def test_zero3_generation_runs_all_rows_synced_and_writes_once(monkeypatch, tmp_path: Path, rank, fail_write):
    generated_kwargs = []
    exchanged_errors = []

    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def parameters(self):
            return iter(())

        def generate(self, **kwargs):
            generated_kwargs.append(kwargs)
            return torch.tensor([[1, 9]], dtype=torch.long)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1]], dtype=torch.long)}

        def decode(self, ids, **kwargs):
            return "{}"

    class Adapter:
        def format_example(self, row, **kwargs):
            return "prompt"

    rows = [
        {"id": "a", "messages": [{"role": "user", "content": "one"}]},
        {"id": "b", "messages": [{"role": "user", "content": "two"}]},
    ]
    monkeypatch.setattr(callbacks, "preserve_generation_eos", lambda *a, **k: [2])
    monkeypatch.setattr(callbacks, "_extract_user_text", lambda row: "user")
    monkeypatch.setattr(callbacks, "_build_stop_string_criteria", lambda *a, **k: None)
    monkeypatch.setattr(
        callbacks,
        "generation_diagnostics",
        lambda *a, **k: {"output_tokens": 1, "input_tokens": 1},
    )
    monkeypatch.setattr(
        callbacks, "build_prediction_record",
        lambda row, generated, runtime: {"id": row["id"], "runtime": runtime},
    )
    monkeypatch.setattr(callbacks, "_gather_prediction_rows", lambda rows, **k: rows)
    monkeypatch.setattr(callbacks, "_distributed_context", lambda: (rank, 2))

    def exchange(error, *, world_size):
        assert world_size == 2
        exchanged_errors.append(error)
        if error is not None:
            raise error

    def failed_write(*args, **kwargs):
        raise OSError("prediction write failed")

    monkeypatch.setattr(callbacks, "_raise_distributed_evaluation_error", exchange)
    output = tmp_path / "predictions.jsonl"
    arguments = {
        "model": Model(), "tokenizer": Tokenizer(), "adapter": Adapter(),
        "split_path": tmp_path / "unused.jsonl", "output_path": output,
        "max_rows": 2, "max_input_tokens": 8, "max_new_tokens": 2,
        "selected_rows": rows, "zero3_trainer": object(),
    }
    if fail_write:
        monkeypatch.setattr(callbacks, "write_jsonl", failed_write)
        with pytest.raises(OSError, match="prediction write failed"):
            callbacks._generate_predictions_with_model(**arguments)
        assert isinstance(exchanged_errors[-1], OSError)
    else:
        count = callbacks._generate_predictions_with_model(**arguments)
        assert count == (2 if rank == 0 else 0)
        assert exchanged_errors == [None, None]
    assert len(generated_kwargs) == 2
    assert all(item["synced_gpus"] is True for item in generated_kwargs)
    if rank == 0 and not fail_write:
        assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    else:
        assert not output.exists()


def test_zero3_best_metadata_failure_is_exchanged_after_collective_save(monkeypatch, tmp_path):
    split_path = tmp_path / "golden.jsonl"
    split_path.write_text('{"id": "one"}\n', encoding="utf-8")
    order = []
    callback = callbacks.build_golden_set_eval_callback(
        enabled=True, split_path=split_path, output_dir=tmp_path / "eval",
        adapter=object(), tokenizer=object(), zero3_trainer=object(),
    )
    monkeypatch.setattr(callbacks, "_distributed_context", lambda: (0, 1))
    monkeypatch.setattr(checkpointing, "save_zero3_checkpoint", lambda *a, **k: order.append("save"))
    original_write = Path.write_text

    def write(path, *args, **kwargs):
        if path.name == "best_golden_eval.json":
            order.append("publish")
            raise OSError("metadata write failed")
        return original_write(path, *args, **kwargs)

    def exchange(error, *, world_size):
        order.append("exchange")
        assert isinstance(error, OSError)
        raise error

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(callbacks, "_raise_distributed_evaluation_error", exchange)
    with pytest.raises(OSError, match="metadata write failed"):
        callback._record_best_if_improved(
            model=object(), state=SimpleNamespace(epoch=1, global_step=1),
            event_label="step1", event_dir=tmp_path, aggregate={"overall_score": 5.0},
        )
    assert order == ["save", "publish", "exchange"]
