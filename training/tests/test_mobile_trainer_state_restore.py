"""CPU proof of Transformers Trainer state restoration used by mobile resume.

This exercises the real installed Trainer checkpoint loader. It is deliberately
not evidence for Gemma, retained-scale fake QAT, PEFT, distributed CUDA, or GPU
training; repository tests cover those contracts and Golden relocation
separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(123)
        self.inputs = torch.randn(16, 4, generator=generator)
        self.labels = torch.randn(16, 1, generator=generator)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.inputs[index], "labels": self.labels[index]}


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 1)

    def forward(self, input_ids, labels=None):
        logits = self.projection(input_ids)
        loss = nn.functional.mse_loss(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class _ObserveTrainBegin(TrainerCallback):
    def __init__(self) -> None:
        self.steps: list[int] = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.steps.append(int(state.global_step))
        return control


class _InspectingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.restore = {}
        super().__init__(*args, **kwargs)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        self.restore["model_hook"] = True
        self.restore["model_after_load"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in (model or self.model).state_dict().items()
        }

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        self.restore["optimizer_hook"] = True
        self.restore["optimizer_state_entries"] = len(self.optimizer.state)
        self.restore["optimizer_after_load"] = copy.deepcopy(
            self.optimizer.state_dict()
        )
        self.restore["scheduler_after_load"] = dict(self.lr_scheduler.state_dict())

    def _load_rng_state(self, checkpoint):
        super()._load_rng_state(checkpoint)
        self.restore["rng_hook"] = True
        self.restore["torch_rng_after_load"] = torch.get_rng_state().clone()


def _arguments(output: Path, *, max_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-2,
        warmup_steps=2,
        lr_scheduler_type="linear",
        save_strategy="steps",
        save_steps=2,
        save_total_limit=2,
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        use_cpu=True,
        seed=19,
        data_seed=19,
    )


def _hashes(path: Path) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def test_real_trainer_restores_numbered_checkpoint_into_extended_total_horizon(
    tmp_path,
):
    dataset = _TinyDataset()
    source_output = tmp_path / "source"
    torch.manual_seed(5)
    first = Trainer(
        model=_TinyModel(),
        args=_arguments(source_output, max_steps=2),
        train_dataset=dataset,
    )
    first.train()
    checkpoint = source_output / "checkpoint-2"
    required = {
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    }
    assert required.issubset({item.name for item in checkpoint.iterdir()})
    source_before = _hashes(checkpoint)

    from safetensors.torch import load_file

    saved_model = load_file(str(checkpoint / "model.safetensors"))
    saved_scheduler = torch.load(
        checkpoint / "scheduler.pt", map_location="cpu", weights_only=False
    )
    saved_optimizer = torch.load(
        checkpoint / "optimizer.pt", map_location="cpu", weights_only=False
    )
    saved_rng = torch.load(
        checkpoint / "rng_state.pth", map_location="cpu", weights_only=False
    )
    saved_state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert saved_state["global_step"] == 2 and saved_state["max_steps"] == 2

    # Deliberately construct different transient model/RNG state. The actual
    # Trainer loaders, not this test, must replace them from checkpoint.
    torch.manual_seed(999)
    resumed_model = _TinyModel()
    assert any(
        not torch.equal(resumed_model.state_dict()[name], tensor)
        for name, tensor in saved_model.items()
    )
    begin = _ObserveTrainBegin()
    continuation = _InspectingTrainer(
        model=resumed_model,
        args=_arguments(tmp_path / "continuation", max_steps=4),
        train_dataset=dataset,
        callbacks=[begin],
    )
    continuation.train(resume_from_checkpoint=str(checkpoint))

    assert continuation.restore["model_hook"] is True
    assert all(
        torch.equal(continuation.restore["model_after_load"][name], tensor)
        for name, tensor in saved_model.items()
    )
    assert continuation.restore["optimizer_hook"] is True
    assert continuation.restore["optimizer_state_entries"] > 0
    restored_optimizer = continuation.restore["optimizer_after_load"]
    assert restored_optimizer["param_groups"] == saved_optimizer["param_groups"]
    assert restored_optimizer["state"].keys() == saved_optimizer["state"].keys()
    for parameter, values in saved_optimizer["state"].items():
        restored = restored_optimizer["state"][parameter]
        assert restored.keys() == values.keys()
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(restored[key], value)
            else:
                assert restored[key] == value
    assert (
        continuation.restore["scheduler_after_load"]["last_epoch"]
        == saved_scheduler["last_epoch"]
        == 2
    )
    assert continuation.args.warmup_steps == 2
    assert continuation.restore["rng_hook"] is True
    assert torch.equal(continuation.restore["torch_rng_after_load"], saved_rng["cpu"])
    assert begin.steps == [2]
    assert continuation.state.global_step == 4
    assert (tmp_path / "continuation/checkpoint-4/trainer_state.json").is_file()
    assert _hashes(checkpoint) == source_before
