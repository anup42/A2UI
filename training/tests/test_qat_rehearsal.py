from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from ir_training.train.qat_objectives import objective, validate_objective


def test_plain_grpo_remains_default_and_requires_no_new_settings():
    assert objective({}) == "grpo"
    assert validate_objective({"grpo": {"family": "e2b"}}) == "grpo"


@pytest.mark.parametrize("settings", [
    {"objective": "unknown"},
    {"rehearsal": {"weight": .1}},
    {"dpo": {}},
    {"objective": "grpo_sft"},
    {"objective": "grpo_sft", "rehearsal": {"weight": 0}},
    {"objective": "grpo_sft", "rehearsal": {"weight": True}},
    {"objective": "grpo_sft", "rehearsal": {"weight": float("nan")}},
    {"objective": "grpo_sft", "rehearsal": {"weight": .1}, "loss_type": "dapo"},
    {"objective": "dpo", "dpo": {"beta": .1, "preferences": "x", "preferences_sha256": "bad"}},
])
def test_objective_configuration_fails_closed(settings):
    with pytest.raises(ValueError):
        validate_objective({"grpo": settings})


def fixture():
    torch = pytest.importorskip("torch")
    from ir_training.train.qat_grpo import QATGRPOModel

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_logits = torch.nn.Parameter(torch.zeros(6, 6))

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            assert use_cache is False
            return SimpleNamespace(logits=self.lora_logits[input_ids])

    model = Model()
    controller = SimpleNamespace(_original_forwards={model: object()})
    model._a2ui_qat_controller = controller
    bundle = QATGRPOModel({"grpo": {"objective": "grpo_sft", "rehearsal": {"weight": .1}}},
                         model, None, None, {}, [5], controller)
    bank = {
        "a": {"prompt_input_ids": [1, 2], "input_ids": [1, 2, 3, 5],
              "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, 3, 5],
              "completion_mask": [0, 0, 1, 1], "completion_tokens": 2},
        "b": {"prompt_input_ids": [2], "input_ids": [2, 4, 5],
              "attention_mask": [1, 1, 1], "labels": [-100, 4, 5],
              "completion_mask": [0, 1, 1], "completion_tokens": 2},
    }
    return torch, model, bundle, bank


def test_indices_remain_aligned_after_shuffle_and_slice():
    from ir_training.train.qat_rehearsal import attach_rehearsal_indices
    torch, _, _, bank = fixture()
    result = {"prompt_ids": torch.tensor([[1, 2], [0, 2], [1, 2]]),
              "prompt_mask": torch.tensor([[1, 1], [0, 1], [1, 1]])}
    attach_rehearsal_indices(result, [{"prompt": x} for x in ("a", "b", "a")], bank)
    perm = torch.tensor([2, 0, 1])
    reordered = {k: v[perm] for k, v in result.items()}
    assert reordered["a2ui_rehearsal_indices"].tolist() == [0, 0, 1]
    for index, prompt, mask in zip(reordered["a2ui_rehearsal_indices"],
                                   reordered["prompt_ids"], reordered["prompt_mask"], strict=True):
        assert prompt[mask.bool()].tolist() == list(bank.values())[int(index)]["prompt_input_ids"]
    result["prompt_ids"][0, 0] = 4
    with pytest.raises(ValueError, match="differs"):
        attach_rehearsal_indices(result, [{"prompt": x} for x in ("a", "b", "a")], bank)


def test_rehearsal_gradient_supervises_completion_and_eos_not_prompt():
    from ir_training.train.qat_rehearsal import rehearsal_loss
    torch, model, _, bank = fixture()
    loss = rehearsal_loss(model, {"a2ui_rehearsal_indices": torch.tensor([0])}, bank, device="cpu")
    loss.backward()
    assert model.lora_logits.grad[1].abs().sum() == 0  # prompt token 2 not supervised
    assert model.lora_logits.grad[2, 3] < 0
    assert model.lora_logits.grad[3, 5] < 0  # real EOS, not padding
    assert model.lora_logits.grad[5].abs().sum() == 0


def test_accumulated_rehearsal_equals_full_batch_gradient():
    from ir_training.train.qat_rehearsal import add_rehearsal_loss
    torch, model, _, bank = fixture()
    def trainer(steps):
        return SimpleNamespace(current_gradient_accumulation_steps=steps,
            accelerator=SimpleNamespace(device="cpu", autocast=nullcontext),
            _metrics={"train": defaultdict(list)})
    add_rehearsal_loss(trainer(1), model, {"a2ui_rehearsal_indices": torch.tensor([0, 1])},
                       model.lora_logits.sum() * 0, bank, .1).backward()
    expected = model.lora_logits.grad.clone()
    model.zero_grad()
    for index in [0, 1]:
        add_rehearsal_loss(trainer(2), model, {"a2ui_rehearsal_indices": torch.tensor([index])},
                           model.lora_logits.sum() * 0, bank, .1).backward()
    torch.testing.assert_close(model.lora_logits.grad, expected)


def test_duplicate_candidates_do_not_multiply_rehearsal_weight():
    from ir_training.train.qat_rehearsal import rehearsal_loss
    torch, model, _, bank = fixture()
    one = rehearsal_loss(model, {"a2ui_rehearsal_indices": torch.tensor([0])}, bank, device="cpu")
    four = rehearsal_loss(model, {"a2ui_rehearsal_indices": torch.tensor([0, 0, 0, 0])}, bank, device="cpu")
    torch.testing.assert_close(one, four)


def test_qat_trainer_adds_loss_only_during_opted_in_training():
    from ir_training.train.qat_grpo import build_qat_grpo_trainer
    torch, model, bundle, bank = fixture()

    class Base:
        def compute_loss(self, model, inputs, **kwargs):
            return model.lora_logits.sum() * 0 + 2.0

    trainer = build_qat_grpo_trainer(Base, bundle)()
    trainer.accelerator = SimpleNamespace(device="cpu", autocast=nullcontext)
    trainer.current_gradient_accumulation_steps = 2
    trainer._metrics = {"train": defaultdict(list)}
    batch = {"a2ui_rehearsal_indices": torch.tensor([0])}
    assert float(trainer.compute_loss(model, batch).detach()) == 2.0  # absent bank = unchanged old path
    trainer._a2ui_rehearsal_bank = bank
    assert float(trainer.compute_loss(model, batch).detach()) > 2.0
    model.eval()
    assert float(trainer.compute_loss(model, {}).detach()) == 2.0  # evaluation must never use train labels


def test_dpo_dispatch_is_explicit(monkeypatch):
    import sys

    from ir_training.train import qat_grpo
    sentinel = object()
    seen = []
    monkeypatch.setattr(qat_grpo, "validate_qat_grpo_config", lambda config: None)
    monkeypatch.setitem(sys.modules, "ir_training.train.qat_dpo", SimpleNamespace(
        train_qat_dpo=lambda *args, **kwargs: (seen.append(kwargs), sentinel)[1]))
    assert qat_grpo.train_qat_grpo({"grpo": {"objective": "dpo"}}, "x",
        dependency_preflight_only=True) is sentinel
    assert seen == [{"dependency_preflight_only": True, "runtime_preflight_only": False}]
