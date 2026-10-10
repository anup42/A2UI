"""Exercise stateless W8 full-model backward with tiny CPU tensors, no training."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from ir_training.common.config import load_yaml
from ir_training.qat.fake_quant import prepare_qat_model
from test_backward_preflight import probe

torch = pytest.importorskip("torch")


class TinyFullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.projection = torch.nn.Linear(4, 16, bias=False)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        if self.gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            logits = checkpoint(self.projection, embeddings, use_reentrant=False)
        else:
            logits = self.projection(embeddings)
        return SimpleNamespace(logits=logits)


def test_full_w8_backward_restores_parameters_gradients_rng_and_qat_wrappers():
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs/models/gemma3_1b_ir_full_qat_sft.yaml")
    model = TinyFullModel()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    controller = prepare_qat_model(model, config)
    wrapped_forward = model.projection.forward
    random_state = torch.get_rng_state()
    report = probe(model, config={"per_device_train_batch_size": 1,
                                 "gradient_accumulation_steps": 4,
                                 "gradient_checkpointing": True})
    assert report["status"] == "passed" and report["optimizer_steps"] == 0
    assert report["distributed_collectives_tested"] is False
    assert report["backward_passes"] == 2
    assert all(batch["gradient_tensors"] == 2 for batch in report["batches"])
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())
    assert torch.equal(torch.get_rng_state(), random_state)
    assert model.projection.forward is wrapped_forward
    assert not model.gradient_checkpointing
    controller.restore()
