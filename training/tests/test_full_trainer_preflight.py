"""CPU contract test for the production-path full-parameter Trainer preflight."""
from __future__ import annotations

import importlib.util
import os
import socket
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

from ir_training.train.full_parameters import (
    FullParameterScopeError,
    enable_full_parameter_training,
)
from ir_training.train.full_trainer_preflight import run_full_trainer_preflight
from ir_training.train.sft import _build_checked_causal_lm_trainer


def _spawn_with_timeout(target, args, *, nprocs: int, timeout_seconds: float = 120.0) -> None:
    import torch.multiprocessing as mp

    process_context = mp.spawn(target, args=args, nprocs=nprocs, join=False)
    deadline = time.monotonic() + timeout_seconds
    try:
        while not process_context.join(timeout=max(0.0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"spawned test workers exceeded {timeout_seconds:.0f}s")
    except BaseException:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        for process in process_context.processes:
            process.join(timeout=5)
        raise


class _TinyFullModel(torch.nn.Module):
    def __init__(self, *, unused_parameter: bool = False):
        super().__init__()
        self.embed = torch.nn.Embedding(17, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.head = torch.nn.Linear(8, 17, bias=False)
        if unused_parameter:
            self.unused = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, labels=None):
        return {"logits": self.head(self.norm(self.embed(input_ids)))}


def _longest_row() -> dict:
    return {
        "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        "labels": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
    }


def _run_two_rank_trainer_preflight_worker(
    rank: int,
    world_size: int,
    master_port: int,
    output_root: str,
) -> None:
    from transformers import Trainer, TrainingArguments, default_data_collator

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        USE_LIBUV="0",
    )
    model = _TinyFullModel()
    enable_full_parameter_training(model)
    args = TrainingArguments(
        output_dir=str(Path(output_root) / "trainer"),
        max_steps=99,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        optim="adafactor",
        weight_decay=0.0,
        max_grad_norm=0.0,
        bf16=False,
        fp16=False,
        use_cpu=True,
        dataloader_pin_memory=False,
        report_to="none",
        save_strategy="no",
    )
    checked_cls = _build_checked_causal_lm_trainer(
        Trainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )
    row = _longest_row()
    report = run_full_trainer_preflight(
        checked_cls,
        {
            "model": model,
            "args": args,
            "train_dataset": [row],
            "data_collator": default_data_collator,
        },
        longest_row=row,
        force_cpu=True,
    )
    torch.save(report, Path(output_root) / f"rank{rank}.pt")


@pytest.mark.skipif(
    importlib.util.find_spec("accelerate") is None,
    reason="actual Trainer preflight requires accelerate",
)
def test_actual_cpu_trainer_preflight_runs_two_complete_accumulation_windows(tmp_path):
    from transformers import (
        Trainer,
        TrainerCallback,
        TrainingArguments,
        default_data_collator,
    )

    model = _TinyFullModel()
    scope = enable_full_parameter_training(model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    longest_row = _longest_row()
    args = TrainingArguments(
        output_dir=str(tmp_path / "trainer-preflight"),
        max_steps=99,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        optim="adafactor",
        weight_decay=0.0,
        max_grad_norm=0.0,
        bf16=False,
        fp16=False,
        report_to="none",
        save_strategy="steps",
        save_steps=1,
    )
    original_args = {
        "max_steps": args.max_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "report_to": list(args.report_to),
    }

    class CallerCallback(TrainerCallback):
        calls = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.calls += 1

    caller_callback = CallerCallback()
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": [longest_row],
        "data_collator": default_data_collator,
        "callbacks": [caller_callback],
    }
    checked_cls = _build_checked_causal_lm_trainer(
        Trainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )

    report = run_full_trainer_preflight(
        checked_cls,
        trainer_kwargs,
        longest_row=longest_row,
        force_cpu=True,
    )

    assert report["passed"] is True
    assert report["disposable_optimizer_steps"] == 2
    assert report["checkpoint_writes"] == 0
    assert report["scope"]["unique_parameter_count"] == scope["unique_parameter_count"]
    assert report["scope"]["all_trainable_fp32"] is True
    assert report["optimizer"]["name"] == "Adafactor"
    assert report["optimizer"]["state_tensor_count"] > 0
    assert report["optimizer"]["state_numel"] > 0
    assert report["ddp_probe"] == {
        "world_size": 1,
        "all_reduce_exercised": False,
        "gradient_as_bucket_view": True,
        "sync_each_batch": True,
        "trainer_backend": "transformers",
        "trainer_path": "checked_causal_lm_trainer",
        "gradient_accumulation_steps": 4,
        "microsteps": 8,
        "synchronized_microsteps": 8,
        "optimizer_steps": 2,
    }
    assert report["memory"]["cuda_local_rank_only"] is False
    assert report["memory"]["ddp_collectives_certified"] is False
    assert report["selection"] == {
        "longest_sequence_length": 8,
        "repeated_real_prepared_row": True,
    }
    assert [step["microsteps"] for step in report["steps"]] == [4, 8]
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert caller_callback.calls == 0
    assert trainer_kwargs["callbacks"] == [caller_callback]
    assert args.max_steps == original_args["max_steps"]
    assert args.save_strategy == original_args["save_strategy"]
    assert args.save_steps == original_args["save_steps"]
    assert args.report_to == original_args["report_to"]


@pytest.mark.skipif(
    importlib.util.find_spec("accelerate") is None,
    reason="actual Trainer preflight requires accelerate",
)
def test_actual_cpu_trainer_preflight_rejects_a_missing_gradient(tmp_path):
    from transformers import Trainer, TrainingArguments, default_data_collator

    model = _TinyFullModel(unused_parameter=True)
    enable_full_parameter_training(model)
    row = _longest_row()
    args = TrainingArguments(
        output_dir=str(tmp_path / "missing-gradient"),
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        optim="adafactor",
        weight_decay=0.0,
        max_grad_norm=0.0,
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
        save_strategy="no",
    )
    checked_cls = _build_checked_causal_lm_trainer(
        Trainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )

    with pytest.raises(FullParameterScopeError, match="missing/non-finite gradient: unused"):
        run_full_trainer_preflight(
            checked_cls,
            {
                "model": model,
                "args": args,
                "train_dataset": [row],
                "data_collator": default_data_collator,
            },
            longest_row=row,
            force_cpu=True,
        )

    fresh_model = _TinyFullModel()
    enable_full_parameter_training(fresh_model)
    first_parameter = next(fresh_model.parameters())
    first_parameter.grad = torch.ones_like(first_parameter)
    with pytest.raises(FullParameterScopeError, match="no pre-existing gradients"):
        run_full_trainer_preflight(
            checked_cls,
            {
                "model": fresh_model,
                "args": args,
                "train_dataset": [row],
                "data_collator": default_data_collator,
            },
            longest_row=row,
            force_cpu=True,
        )


@pytest.mark.skipif(
    not torch.distributed.is_available() or importlib.util.find_spec("accelerate") is None,
    reason="actual two-rank Trainer preflight requires distributed Accelerate",
)
def test_actual_two_rank_cpu_trainer_preflight_uses_ddp_sync_path(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    _spawn_with_timeout(
        _run_two_rank_trainer_preflight_worker,
        (2, master_port, str(tmp_path)),
        nprocs=2,
    )

    reports = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
    for report in reports:
        assert report["passed"] is True
        assert report["ddp_probe"]["world_size"] == 2
        assert report["ddp_probe"]["all_reduce_exercised"] is True
        assert report["ddp_probe"]["gradient_as_bucket_view"] is True
        assert report["ddp_probe"]["sync_each_batch"] is True
        assert report["ddp_probe"]["microsteps"] == 8
        assert report["ddp_probe"]["synchronized_microsteps"] == 8
        assert report["ddp_probe"]["optimizer_steps"] == 2
        assert report["memory"]["ddp_collectives_certified"] is False
