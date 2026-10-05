"""Real local HF/Trainer regression tests; tiny random weights, no downloads."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def tiny_full_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    tokenizers = pytest.importorskip("tokenizers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    source, data = tmp_path / "seed", tmp_path / "data"
    source.mkdir()
    data.mkdir()
    vocabulary = {word: i for i, word in enumerate(
        ["[PAD]", "[UNK]", "[EOS]", "[BOS]", "user", "assistant", ":", "prompt", "result"]
        + [f"item{i}" for i in range(23)]
    )}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        eos_token="[EOS]", bos_token="[BOS]", model_max_length=64,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' : ' + message['content'] + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant : ' }}{% endif %}"
    )
    tokenizer.save_pretrained(source)
    # A fixed local seed prevents earlier tests from changing this fixture's
    # logits. Restore their RNG state after constructing our random model.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
            tie_word_embeddings=True, pad_token_id=0, bos_token_id=3, eos_token_id=2,
            attention_dropout=0.0,
        ))
    # This value survives FP32 serialization but rounds to exactly 1 in BF16.
    with torch.no_grad():
        model.model.norm.weight.fill_(1.00002)
    residual = model.model.norm.weight.detach().clone()
    assert not torch.equal(residual, residual.to(torch.bfloat16).float())
    model.save_pretrained(source)
    del model
    rows = [{"id": f"row-{i}", "messages": [
        {"role": "user", "content": f"prompt item{i}"},
        {"role": "assistant", "content": f"result item{i}"},
    ]} for i in range(2)]
    for split in ("train", "val"):
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )
    config = {
        "run": {"id": "tiny-full", "dataset_dir": str(data), "output_dir": str(tmp_path / "run")},
        "model": {"family": "llama", "model_id": "local-random-llama", "model_source": str(source),
                  "tokenizer_source": str(source), "tokenizer_loader": "pretrained_tokenizer_fast",
                  "model_loader": "auto_causal_lm", "dtype": "bfloat16", "device_map": "none",
                  "attn_implementation": "eager", "max_context_tokens": 64, "inference_device": "cpu"},
        "training": {"method": "full_finetune_sft", "trainer_backend": "hf", "allow_cpu": True,
                     "max_steps": 2, "epochs": 1, "learning_rate": 2e-5, "weight_decay": 0.0,
                     "optim": "adamw_torch", "warmup_steps": 0, "max_grad_norm": 1.0,
                     "mixed_precision": "off", "max_seq_length": 64, "token_cache": False,
                     "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
                     "gradient_accumulation_steps": 1, "dataloader_num_workers": 0,
                     "report_to": "none", "logging_steps": 1, "eval_strategy": "no",
                     "save_strategy": "steps", "save_steps": 1, "save_total_limit": 2,
                     "backward_preflight": False},
        "preflight": {"rows": 1, "greedy_probe_rows": 1, "greedy_probe_new_tokens": 2,
                      "min_greedy_tokens": 1, "logit_probe_tokens": 2},
        "golden_eval": {"enabled": False},
    }
    try:
        yield config, residual
    finally:
        torch.set_num_threads(threads)


def _qat_config(config):
    config["training"]["method"] = "full_finetune_qat"
    # This tests optimizer/storage precision, not QAT-vs-dense quality on
    # random weights. Keep finite-loss and repeated-generation checks; a
    # two-token probe cannot satisfy the production eight-token prefix floor.
    config["preflight"].update(
        min_baseline_qat_greedy_prefix_tokens=0,
        min_top1_probe_match=0.0,
        require_greedy_determinism=True,
    )
    config["qat"] = {
        "enabled": True, "quantizer": "ste_ai_edge", "scale_mode": "dynamic",
        "weight_bits": 8, "activation_bits": 32, "weight_per_channel": True,
        "weight_axis": 0, "quantize_embeddings": True, "exclude_modules": [],
        "ste_gradient": "clipped", "final_runtime_validation_required": True,
    }


@pytest.mark.parametrize("qat", [False, True], ids=["sft-fp32-compute", "qat-cpu-bf16-compute"])
def test_real_trainer_keeps_small_updates_and_reload_in_fp32(tiny_full_model, tmp_path, monkeypatch, qat):
    import torch

    pytest.importorskip("peft", minversion="0.19.0")
    pytest.importorskip("datasets")
    from ir_training.models.hf_loading import load_hf_model
    from ir_training.train import sft
    from ir_training.train.numeric_resume import verify_numeric_training_resume
    from ir_training.train.resume_contract import verify_resume_contract
    from transformers import Trainer, TrainingArguments

    config, residual = tiny_full_model
    cpu_amp_arguments = []
    if qat:
        _qat_config(config)
        config["training"]["mixed_precision"] = "bf16"
        original_post_init = TrainingArguments.__post_init__

        def cpu_amp_post_init(arguments):
            # Explicit CPU-only AMP harness: the production launcher does not
            # forward use_cpu. Keep real HF validation, Trainer and BF16 AMP;
            # this is not GPU or general CPU-BF16 launcher validation. Patching
            # post-init preserves the picklable class used in HF checkpoints.
            arguments.use_cpu = True
            original_post_init(arguments)
            cpu_amp_arguments.append((arguments.bf16, arguments.fp16))

        monkeypatch.setattr(TrainingArguments, "__post_init__", cpu_amp_post_init)
    # Exercise a BF16-requested load on a CPU fixture. Otherwise the production
    # no-CUDA fallback would mask removal of the FP32 load override.
    monkeypatch.setattr(sft, "_resolve_training_dtype", lambda _dtype: "bfloat16")
    original_create = sft.create_adapter
    observed = []

    def create_adapter(model_config):
        adapter = original_create(model_config)
        original_load = adapter.load_model

        def load_model():
            model = original_load()
            assert model.model.embed_tokens.weight is model.lm_head.weight
            assert all(p.dtype == torch.float32 for p in model.parameters())
            assert torch.equal(model.model.norm.weight, residual)
            assert all(not b.is_floating_point() or b.dtype == torch.float32 for b in model.buffers())
            observed.append(model)
            return model

        adapter.load_model = load_model
        return adapter

    monkeypatch.setattr(sft, "create_adapter", create_adapter)
    result = sft.train_sft(config)
    if qat:
        assert cpu_amp_arguments == [(True, False)], "The CPU harness must retain real BF16 AMP"
    assert len(observed) == 1
    model = observed[0]
    assert model.model.embed_tokens.weight is model.lm_head.weight
    updated = model.model.norm.weight.detach()
    assert not torch.equal(updated, residual), "Real AdamW must update the trainable norm"
    assert torch.any(updated != updated.to(torch.bfloat16).float())
    assert result["full_finetune_precision"]["parameter_dtype"] == "float32"
    checkpoint = Path(config["run"]["output_dir"]) / "checkpoint-2"
    saved_metadata = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["full_finetune_precision"] == result["full_finetune_precision"]
    numeric_resume = verify_numeric_training_resume(
        checkpoint, config, expected_full_precision=result["full_finetune_precision"],
    )
    assert numeric_resume["checked"] is True and numeric_resume["verified"] is True
    if qat:
        assert saved_metadata["qat"]["numeric_contract"]["weight_ste_rule"] == "rounded_code_range_v1"
        assert numeric_resume["weight_ste_rule"] == "rounded_code_range_v1"
    resume = verify_resume_contract(checkpoint, result["resume_contract"], config=config)
    assert resume["verified"] is True and resume["global_step"] == 2
    state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    moments = [value for entry in state["state"].values() for key, value in entry.items()
               if key in {"exp_avg", "exp_avg_sq"}]
    assert moments and all(t.dtype == torch.float32 for t in moments)

    # Real HF serialization and optimizer restoration must preserve the
    # residual and tied weights, not just the in-memory training model.
    reload_cfg = dict(config["model"], dtype="float32")
    restored = load_hf_model(str(checkpoint), reload_cfg)
    assert restored.model.embed_tokens.weight is restored.lm_head.weight
    assert torch.equal(restored.model.norm.weight, updated)
    # Trainer groups decay/no-decay parameters even when both decay rates are
    # zero. Recreate its real grouping and parameter order instead of loading
    # a two-group checkpoint into a one-group hand-built AdamW optimizer.
    restore_trainer = Trainer(
        model=restored,
        args=TrainingArguments(
            output_dir=str(tmp_path / "restore"), use_cpu=True,
            optim="adamw_torch", learning_rate=2e-5, weight_decay=0.0,
            bf16=qat, report_to="none",
        ),
    )
    optimizer = restore_trainer.create_optimizer()
    assert [len(group["params"]) for group in optimizer.param_groups] == [
        len(group["params"]) for group in state["param_groups"]
    ]
    optimizer.load_state_dict(state)
    assert all(value.dtype == torch.float32 for entry in optimizer.state.values()
               for key, value in entry.items() if key in {"exp_avg", "exp_avg_sq"})
    assert all(not b.is_floating_point() or b.dtype == torch.float32 for b in restored.buffers())


@pytest.mark.parametrize("qat", [False, True], ids=["sft", "qat"])
def test_standalone_generation_loads_full_checkpoint_without_bf16_roundtrip(tiny_full_model, tmp_path, monkeypatch, qat):
    import torch
    from ir_training.eval import generate

    config, residual = tiny_full_model
    if qat:
        _qat_config(config)
    original_config = copy.deepcopy(config)
    original_create = generate.create_adapter
    observed = []

    def create_adapter(model_config):
        adapter = original_create(model_config)
        original_load = adapter.load_model

        def load_model():
            model = original_load()
            # Inspect the real loader output before precision binding or QAT.
            assert torch.equal(model.model.norm.weight, residual)
            assert all(p.dtype == torch.float32 for p in model.parameters())
            assert model.model.embed_tokens.weight is model.lm_head.weight
            observed.append(model)
            return model

        adapter.load_model = load_model
        return adapter

    monkeypatch.setattr(generate, "create_adapter", create_adapter)
    destination = tmp_path / "predictions.jsonl"
    count = generate.generate_predictions(
        config, Path(config["run"]["dataset_dir"]) / "val.jsonl", destination,
        max_rows=1, max_input_tokens=64, max_new_tokens=2, apply_qat=qat,
    )
    assert count == 1 and len(observed) == 1
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 1
    assert config == original_config, "Evaluation must not rewrite the caller's precision configuration"
