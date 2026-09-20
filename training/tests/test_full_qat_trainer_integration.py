"""Small CPU integration checks for the isolated all-parameter trainer lane."""
from __future__ import annotations

import contextlib
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

from ir_training.qat.fake_quant import prepare_qat_model
from ir_training.qat.full_model_contract import (
    configure_full_qat,
    verify_full_qat_coverage,
)
from ir_training.train.full_parameters import (
    enable_full_parameter_training,
    probe_full_optimizer_step,
)
from ir_training.train.sft import (
    _adapter_checkpoint_manifest,
    _build_checked_causal_lm_trainer,
    _checked_shifted_causal_lm_loss,
)


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


def _run_tiny_ddp_accumulation_worker(
    rank: int,
    world_size: int,
    init_method: str,
    sync_each_batch: bool,
    result_path: str,
) -> None:
    """Exercise the storage distinction that caused the full-QAT first-backward OOM."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(1234)
        model = DistributedDataParallel(
            torch.nn.Linear(4, 3, bias=True),
            gradient_as_bucket_view=True,
        )
        first_backward_uses_bucket_views = None
        for microstep in range(4):
            values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            values = values + float(rank + microstep)
            context = (
                model.no_sync()
                if not sync_each_batch and microstep < 3
                else contextlib.nullcontext()
            )
            with context:
                (model(values).square().mean() / 4.0).backward()
            if microstep == 0:
                first_backward_uses_bucket_views = all(
                    getattr(parameter.grad, "_base", None) is not None
                    for parameter in model.parameters()
                )

        if rank == 0:
            torch.save(
                {
                    "first_backward_uses_bucket_views": first_backward_uses_bucket_views,
                    "gradients": [parameter.grad.detach().clone() for parameter in model.parameters()],
                },
                result_path,
            )
    finally:
        dist.destroy_process_group()


def _base_config() -> dict:
    return {
        "run": {},
        "model": {
            "model_id": "google/gemma-4-E2B-it-qat-mobile-transformers",
            "model_source": "seed",
            "mobile_training_seed_manifest": "seed/mobile_training_seed_manifest.json",
            "mobile_qparams_contract": "seed/mobile_qparams.json",
        },
        "training": {},
        "lora": {"r": 16},
        "qat_mtp": {"enabled": False},
        "preflight": {
            "rows": 1,
            "logit_probe_tokens": 1,
            "greedy_probe_rows": 1,
            "greedy_probe_new_tokens": 8,
            "min_greedy_tokens": 8,
        },
    }


def test_full_contract_selects_fixed_lr_adafactor_and_disables_external_clip():
    config = configure_full_qat(_base_config())
    training = config["training"]

    assert training["optim"] == "adafactor"
    assert training["weight_decay"] == 0.0
    assert training["max_grad_norm"] == 0.0
    assert training["mixed_precision"] == "bf16"
    assert training["full_parameter_training"] is True
    assert training["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


class _FakeBaseTrainer:
    def __init__(self, *, handler=None, gradient_accumulation_plugin=None):
        self.handler = handler
        self.gradient_accumulation_plugin = gradient_accumulation_plugin or SimpleNamespace(
            sync_each_batch=False
        )
        self.accelerator_args = self._build_accelerator_args(
            gradient_accumulation_plugin=self.gradient_accumulation_plugin
        )
        self.accelerator = SimpleNamespace(
            gradient_state=SimpleNamespace(
                plugin_kwargs={
                    "sync_each_batch": self.gradient_accumulation_plugin.sync_each_batch
                }
            )
        )
        self.model_accepts_loss_kwargs = True
        self.state = SimpleNamespace(global_step=0)
        self.model = None
        self.training_step_calls = 0

    def _build_accelerator_args(self, **kwargs):
        result = {"sentinel": "ordinary", **kwargs}
        if self.handler is not None:
            result["kwargs_handlers"] = [self.handler]
        return result

    def training_step(self, model, inputs, *args, **kwargs):
        self.training_step_calls += 1
        return "trained"


def test_checked_trainer_sets_full_ddp_handler_policy():
    handler = SimpleNamespace(
        gradient_as_bucket_view=False,
        broadcast_buffers=True,
        find_unused_parameters=True,
    )
    checked = _build_checked_causal_lm_trainer(
        _FakeBaseTrainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )(handler=handler)

    assert handler.gradient_as_bucket_view is True
    assert handler.broadcast_buffers is False
    assert handler.find_unused_parameters is False
    assert checked.accelerator_args["kwargs_handlers"][0] is handler
    assert checked.gradient_accumulation_plugin.sync_each_batch is True
    assert checked.accelerator.gradient_state.plugin_kwargs["sync_each_batch"] is True
    assert checked._a2ui_full_ddp_handler_configured is True
    assert checked.model_accepts_loss_kwargs is False


def test_checked_trainer_requires_live_bucket_view_under_multi_rank(monkeypatch):
    handler = SimpleNamespace(
        gradient_as_bucket_view=False,
        broadcast_buffers=True,
        find_unused_parameters=True,
    )
    checked = _build_checked_causal_lm_trainer(
        _FakeBaseTrainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )(handler=handler)
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(ValueError, match="live DDP gradient bucket views"):
        checked.training_step(
            SimpleNamespace(
                gradient_as_bucket_view=False,
                require_backward_grad_sync=True,
            ),
            {},
        )
    assert checked.training_step(
        SimpleNamespace(
            gradient_as_bucket_view=True,
            require_backward_grad_sync=True,
        ),
        {},
    ) == "trained"


def test_checked_trainer_rejects_live_no_sync_under_multi_rank(monkeypatch):
    handler = SimpleNamespace(
        gradient_as_bucket_view=False,
        broadcast_buffers=True,
        find_unused_parameters=True,
    )
    checked = _build_checked_causal_lm_trainer(
        _FakeBaseTrainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )(handler=handler)
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(ValueError, match="sync|no_sync"):
        checked.training_step(
            SimpleNamespace(
                gradient_as_bucket_view=True,
                require_backward_grad_sync=False,
            ),
            {},
        )


def test_checked_trainer_rejects_live_accumulation_plugin_without_sync_each_batch(monkeypatch):
    handler = SimpleNamespace(
        gradient_as_bucket_view=False,
        broadcast_buffers=True,
        find_unused_parameters=True,
    )
    checked = _build_checked_causal_lm_trainer(
        _FakeBaseTrainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )(handler=handler)
    checked.accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = False
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(ValueError, match="sync_each_batch"):
        checked.training_step(
            SimpleNamespace(
                gradient_as_bucket_view=True,
                require_backward_grad_sync=True,
            ),
            {},
        )


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="two-rank bucket-view regression requires torch.distributed",
)
def test_two_rank_sync_each_batch_uses_bucket_views_from_first_backward_and_preserves_mean(tmp_path):
    results = {}
    for sync_each_batch in (False, True):
        init_file = tmp_path / f"gloo_{sync_each_batch}"
        result_file = tmp_path / f"result_{sync_each_batch}.pt"
        _spawn_with_timeout(
            _run_tiny_ddp_accumulation_worker,
            (
                2,
                init_file.resolve().as_uri(),
                sync_each_batch,
                str(result_file),
            ),
            nprocs=2,
        )
        results[sync_each_batch] = torch.load(result_file, weights_only=True)

    assert results[False]["first_backward_uses_bucket_views"] is False
    assert results[True]["first_backward_uses_bucket_views"] is True
    for default_gradient, synchronized_gradient in zip(
        results[False]["gradients"], results[True]["gradients"], strict=True
    ):
        torch.testing.assert_close(default_gradient, synchronized_gradient)


def test_full_checkpoint_roundtrip_keeps_embedding_and_norm_updates(tmp_path):
    from safetensors.torch import load_file, save_file

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(5, 3)
            self.norm = torch.nn.LayerNorm(3)

        def forward(self, ids):
            values = self.norm(self.embed(ids))
            return (values * torch.tensor([1.0, 2.0, 4.0])).sum()

    model = Tiny()
    enable_full_parameter_training(model)
    before_embed = model.embed.weight.detach().clone()
    before_norm = model.norm.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model(torch.tensor([[0, 1, 2]])).backward()
    optimizer.step()
    assert not torch.equal(before_embed, model.embed.weight)
    assert not torch.equal(before_norm, model.norm.weight)

    checkpoint = tmp_path / "best_golden_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, checkpoint / "model.safetensors")

    manifest = _adapter_checkpoint_manifest(checkpoint, role="best_golden")
    restored = load_file(checkpoint / "model.safetensors")
    assert manifest["checkpoint_kind"] == "full_model"
    assert torch.equal(restored["embed.weight"], model.embed.weight)
    assert torch.equal(restored["norm.weight"], model.norm.weight)


def test_checked_trainer_rejects_missing_ddp_handler_capability():
    checked_cls = _build_checked_causal_lm_trainer(
        _FakeBaseTrainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )
    checked = checked_cls(handler=None)
    assert checked.accelerator_args["sentinel"] == "ordinary"
    assert (
        checked.accelerator_args["gradient_accumulation_plugin"].sync_each_batch
        is True
    )
    assert checked._a2ui_full_ddp_handler_configured is False


def test_checked_trainer_rejects_missing_accumulation_plugin_capability():
    class MissingAccumulationPluginTrainer:
        def __init__(self):
            self._build_accelerator_args()

        def _build_accelerator_args(self, **kwargs):
            return dict(kwargs)

    checked_cls = _build_checked_causal_lm_trainer(
        MissingAccumulationPluginTrainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )
    with pytest.raises(ValueError, match="sync_each_batch accumulation policy"):
        checked_cls()


def test_checked_trainer_rejects_missing_preconstruction_handler_for_multirank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    checked_cls = _build_checked_causal_lm_trainer(
        _FakeBaseTrainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )
    with pytest.raises(ValueError, match="before Accelerator construction"):
        checked_cls(handler=None)


def test_ordinary_checked_trainer_does_not_modify_accelerator_handlers():
    handler = SimpleNamespace(
        gradient_as_bucket_view=False,
        broadcast_buffers=True,
        find_unused_parameters=True,
    )
    checked = _build_checked_causal_lm_trainer(
        _FakeBaseTrainer,
        {"full_parameter_training": False, "lora_diagnostics_steps": 0},
    )(handler=handler)
    assert checked.accelerator_args["sentinel"] == "ordinary"
    assert handler.gradient_as_bucket_view is False
    assert handler.broadcast_buffers is True
    assert handler.find_unused_parameters is True
    assert checked.gradient_accumulation_plugin.sync_each_batch is False
    assert checked.accelerator.gradient_state.plugin_kwargs["sync_each_batch"] is False
    assert not hasattr(checked, "_a2ui_full_ddp_handler_configured")


def test_ordinary_fake_trainer_without_accelerator_hook_remains_compatible():
    class MinimalTrainer:
        def __init__(self):
            self.model_accepts_loss_kwargs = True

    checked = _build_checked_causal_lm_trainer(
        MinimalTrainer,
        {"full_parameter_training": False, "lora_diagnostics_steps": 0},
    )()
    assert checked.model_accepts_loss_kwargs is False


@pytest.mark.skipif(importlib.util.find_spec("accelerate") is None, reason="temporary/remote Trainer environment needs accelerate")
def test_actual_transformers_trainer_runs_one_cpu_adafactor_step_and_saves_full_model(tmp_path):
    from transformers import Trainer, TrainingArguments

    class TinyFullModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(11, 4)
            self.norm = torch.nn.LayerNorm(4)
            self.head = torch.nn.Linear(4, 11, bias=False)

        def forward(self, input_ids, labels=None):
            logits = self.head(self.norm(self.embed(input_ids)))
            return {"logits": logits}

        def save_pretrained(self, path, *, state_dict=None, **_kwargs):
            from safetensors.torch import save_file

            destination = Path(path)
            destination.mkdir(parents=True, exist_ok=True)
            tensors = state_dict or self.state_dict()
            save_file(
                {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
                destination / "model.safetensors",
            )
            (destination / "config.json").write_text("{}", encoding="utf-8")

    model = TinyFullModel()
    enable_full_parameter_training(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    rows = [
        {"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([1, 2, 3, 4])},
        {"input_ids": torch.tensor([4, 3, 2, 1]), "labels": torch.tensor([4, 3, 2, 1])},
    ]
    args = TrainingArguments(
        output_dir=str(tmp_path / "trainer"),
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        optim="adafactor",
        max_grad_norm=0.0,
        bf16=False,
        fp16=False,
        report_to="none",
        save_strategy="no",
    )
    checked_cls = _build_checked_causal_lm_trainer(
        Trainer,
        {"full_parameter_training": True, "lora_diagnostics_steps": 0},
    )
    trainer = checked_cls(model=model, args=args, train_dataset=rows)
    assert trainer._a2ui_full_ddp_handler_configured is True
    trainer.train()

    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    assert type(optimizer).__name__ == "Adafactor"
    group = optimizer.param_groups[0]
    assert group["initial_lr"] == pytest.approx(1e-4)
    assert group["scale_parameter"] is False
    assert group["relative_step"] is False
    assert group["warmup_init"] is False
    assert all(parameter.grad is None for parameter in model.parameters())
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    assert not torch.equal(before["embed.weight"], model.embed.weight)
    assert not torch.equal(before["norm.weight"], model.norm.weight)

    saved = tmp_path / "saved"
    trainer.save_model(saved)
    # Plain nn.Module Trainer fallback writes the complete safetensor but has no
    # HF config object; production PreTrainedModel.save_pretrained writes this.
    (saved / "config.json").write_text("{}", encoding="utf-8")
    manifest = _adapter_checkpoint_manifest(saved, role="final")
    assert manifest["checkpoint_kind"] == "full_model"


def test_tiny_real_gemma4_full_qat_scope_backward_and_adafactor_probe(tmp_path):
    from safetensors.torch import load_file
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    text_config = Gemma4TextConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        vocab_size_per_layer_input=16,
        hidden_size_per_layer_input=2,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        tie_word_embeddings=True,
        use_cache=False,
    )
    model = Gemma4ForCausalLM(text_config)
    scope = enable_full_parameter_training(model)
    config = configure_full_qat(_base_config())
    controller = prepare_qat_model(model, config)
    coverage = verify_full_qat_coverage(model, controller.summary())
    assert scope["frozen_parameter_count"] == 0
    assert coverage["verified"] is True
    assert coverage["matrix_count"] == 22

    input_ids = torch.tensor([[2, 3, 4, 5, 6]])
    labels = input_ids.clone()
    finite_gradients = {}

    def backward():
        logits = model(input_ids=input_ids, use_cache=False).logits
        loss = _checked_shifted_causal_lm_loss(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        finite_gradients.update(
            {
                name: parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                for name, parameter in model.named_parameters()
            }
        )

    report = probe_full_optimizer_step(
        model,
        backward=backward,
        learning_rate=1e-4,
        force_cpu=True,
    )
    assert finite_gradients
    assert all(finite_gradients.values())
    assert report["scope"]["unique_parameter_count"] == scope["unique_parameter_count"]
    assert report["optimizer"]["state_tensor_count"] > 0

    controller.restore()
    checkpoint = tmp_path / "tiny-gemma4-full"
    model.save_pretrained(checkpoint, safe_serialization=True)
    restored = load_file(checkpoint / "model.safetensors")
    assert "model.embed_tokens.weight" in restored
    assert "model.norm.weight" in restored
    assert restored["model.embed_tokens.weight"].dtype == torch.float32
    assert restored["model.norm.weight"].dtype == torch.float32
    assert _adapter_checkpoint_manifest(checkpoint, role="probe")["checkpoint_kind"] == "full_model"
