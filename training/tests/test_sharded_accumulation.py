"""CPU integration tests for Trainer/Accelerate/ZeRO accumulation alignment."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, GradientAccumulationPlugin
from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
from ir_training.train.sharded_contract import (
    assert_sharded_accumulation,
    configure_sharded_accumulation,
)
from transformers import Trainer


def _config(steps: int) -> dict:
    return {
        "distributed_backend": "sharded",
        "optim": "adamw_torch",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": steps,
        "expected_effective_batch_size": steps * 4,
        "max_grad_norm": 0.0,
    }


@pytest.mark.parametrize("steps", [4, 8, 16])
def test_configure_and_assert_sharded_accumulation_exactly_match(steps):
    plugin = GradientAccumulationPlugin(num_steps=1)
    args = {"gradient_accumulation_plugin": plugin}
    configure_sharded_accumulation(
        args, _config(steps), trainer_accumulation_steps=steps
    )
    assert plugin.num_steps == steps
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.DEEPSPEED,
        gradient_accumulation_steps=steps,
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(
                get_value=lambda key: steps
                if key == "gradient_accumulation_steps"
                else None
            )
        ),
    )
    assert assert_sharded_accumulation(
        accelerator, _config(steps), trainer_accumulation_steps=steps
    ) == {"trainer": steps, "accelerate": steps, "deepspeed": steps}


def test_four_gpu_accumulation_eight_preserves_effective_batch_32():
    config = _config(8)
    assert config["per_device_train_batch_size"] * 4 * config[
        "gradient_accumulation_steps"
    ] == config["expected_effective_batch_size"] == 32


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"gradient_accumulation_plugin": object()},
        {
            "gradient_accumulation_plugin": GradientAccumulationPlugin(num_steps=1),
            "gradient_accumulation_steps": 8,
        },
    ],
)
def test_configure_rejects_missing_wrong_or_duplicate_accumulation_path(args):
    with pytest.raises(ValueError):
        configure_sharded_accumulation(args, _config(8), trainer_accumulation_steps=8)


@pytest.mark.parametrize(
    ("distributed_type", "accelerate_steps", "ds_steps", "trainer_steps"),
    [
        (DistributedType.NO, 8, 8, 8),
        (DistributedType.DEEPSPEED, 1, 8, 8),
        (DistributedType.DEEPSPEED, 8, 1, 8),
        (DistributedType.DEEPSPEED, 8, 8, 1),
        (DistributedType.DEEPSPEED, True, 8, 8),
    ],
)
def test_live_accumulation_mismatches_are_rejected(
    distributed_type, accelerate_steps, ds_steps, trainer_steps
):
    accelerator = SimpleNamespace(
        distributed_type=distributed_type,
        gradient_accumulation_steps=accelerate_steps,
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(
                get_value=lambda key: ds_steps
                if key == "gradient_accumulation_steps"
                else None
            )
        ),
    )
    with pytest.raises(ValueError):
        assert_sharded_accumulation(
            accelerator, _config(8), trainer_accumulation_steps=trainer_steps
        )


def test_live_accumulation_requires_deepspeed_plugin():
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.DEEPSPEED,
        gradient_accumulation_steps=8,
        state=SimpleNamespace(deepspeed_plugin=None),
    )
    with pytest.raises(ValueError, match="DeepSpeed plugin"):
        assert_sharded_accumulation(
            accelerator, _config(8), trainer_accumulation_steps=8
        )


class _FakeDeepSpeedEngine:
    def __init__(self):
        self.backward_losses = []
        self.boundaries = []
        self.steps = 0

    def set_gradient_accumulation_boundary(self, *, is_boundary):
        self.boundaries.append(is_boundary)

    def backward(self, loss, **kwargs):
        self.backward_losses.append((float(loss.detach()), dict(kwargs)))
        loss.backward()

    def step(self):
        self.steps += 1


class _AcceleratorHarness:
    backward = Accelerator.backward

    def __init__(self, engine, steps):
        self.distributed_type = DistributedType.DEEPSPEED
        self.gradient_accumulation_steps = steps
        self.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)
        self.sync_gradients = False


class _MeanLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, values):
        return {"loss": (self.weight * values).mean()}


class _TrainerStepHarness:
    """Supply dependencies while executing pinned HF Trainer.training_step."""

    def __init__(self, accelerator, steps):
        self.accelerator = accelerator
        self.optimizer = SimpleNamespace()
        self.args = SimpleNamespace(
            gradient_accumulation_steps=steps,
            torch_empty_cache_steps=None,
            optim="adamw_torch",
            n_gpu=1,
            device=torch.device("cpu"),
        )
        self.state = SimpleNamespace(global_step=0)
        self.model_accepts_loss_kwargs = False
        self.compute_loss_func = None
        self.current_gradient_accumulation_steps = steps

    def _prepare_context_parallel_inputs(self, model, inputs):
        return contextlib.nullcontext, inputs

    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss_context_manager(self):
        return contextlib.nullcontext()

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        return model(**inputs)["loss"]


def test_pinned_trainer_accelerator_and_ds_wrapper_divide_mean_loss_once():
    steps = 8
    engine = _FakeDeepSpeedEngine()
    accelerator = _AcceleratorHarness(engine, steps)
    trainer = _TrainerStepHarness(accelerator, steps)
    model = _MeanLossModel()
    raw_mean = 4.0
    returned = []
    for microstep in range(steps):
        accelerator.sync_gradients = microstep == steps - 1
        returned.append(
            Trainer.training_step(
                trainer, model, {"values": torch.tensor([1.0, 2.0, 3.0])}
            )
        )

    expected = raw_mean / steps
    assert [float(loss) for loss in returned] == pytest.approx([expected] * steps)
    assert [value for value, _ in engine.backward_losses] == pytest.approx(
        [expected] * steps
    )
    assert all(kwargs == {"scale_wrt_gas": False} for _, kwargs in engine.backward_losses)
    assert sum(value for value, _ in engine.backward_losses) == pytest.approx(raw_mean)
    assert engine.boundaries == [False] * (steps - 1) + [True]
    assert engine.steps == 1
