"""CPU-only contracts for the explicitly enabled official-mobile MTP trainer."""
from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.train import mtp_drafter as mtp


def _fixture(tmp_path):
    config = load_yaml(ROOT / "configs/models/gemma4_e2b_mtp_drafter_qat.yaml")
    target_config = load_yaml(ROOT / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml")
    for name in ("seed", "adapter", "merged", "prepared"):
        (tmp_path / name).mkdir()
    for name in ("seed", "adapter", "merged"):
        (tmp_path / name / "model.safetensors").write_bytes(b"fixture-not-real-model")
    for name in ("mobile_training_seed_manifest.json", "mobile_qparams.json"):
        (tmp_path / "seed" / name).write_text("{}", encoding="utf-8")
    for name in ("train.jsonl", "val.jsonl"):
        (tmp_path / "prepared" / name).write_text(
            json.dumps({"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}) + "\n", encoding="utf-8")
    (tmp_path / "prepared/manifest.json").write_text("{}", encoding="utf-8")
    target_config["model"].update(model_source=str(tmp_path / "seed"), tokenizer_source=str(tmp_path / "seed"),
                                  mobile_training_seed_manifest=str(tmp_path / "seed/mobile_training_seed_manifest.json"),
                                  mobile_qparams_contract=str(tmp_path / "seed/mobile_qparams.json"),
                                  chat_template_kwargs={"enable_thinking": False})
    target_path = tmp_path / "target.yaml"
    target_path.write_text(yaml.safe_dump(target_config), encoding="utf-8")
    config["run"].update(purpose=mtp.OFFICIAL_MOBILE_MTP_PURPOSE, dataset_dir=str(tmp_path / "prepared"),
                         output_dir=str(tmp_path / "run"))
    config["target"].update(training_config=str(target_path), adapter_checkpoint=str(tmp_path / "adapter"),
                            model_id_or_path=str(tmp_path / "merged"), tokenizer_id_or_path=str(tmp_path / "seed"))
    config["assistant"]["best_checkpoint_dir"] = str(tmp_path / "run/best_checkpoint")
    config["training"].update(require_cuda=True, overflow_policy="error", mixed_precision="bf16",
                              chat_template_kwargs={"enable_thinking": False})
    return config


def test_source_bindings_cover_and_detect_conditioning_input_changes(tmp_path):
    config = _fixture(tmp_path)
    before = mtp.official_mobile_source_bindings(config)
    assert str(tmp_path / "target.yaml") in before
    for name in ("seed", "adapter", "merged"):
        assert str(tmp_path / name / "model.safetensors") in before
    assert str(tmp_path / "prepared/val.jsonl") in before
    (tmp_path / "merged/model.safetensors").write_bytes(b"changed")
    assert mtp.official_mobile_source_bindings(config) != before


@pytest.mark.parametrize("key,value", [("require_cuda", False), ("overflow_policy", "truncate"),
                                       ("mixed_precision", "fp16"), ("chat_template_kwargs", {})])
def test_optin_config_rejects_safety_or_tokenization_drift(tmp_path, key, value):
    config = _fixture(tmp_path)
    config["training"][key] = value
    with pytest.raises(mtp.MTPDrafterTrainingError):
        mtp._official_target_config(config)


def test_official_plan_requires_validation_but_legacy_does_not(tmp_path):
    config = _fixture(tmp_path)
    (tmp_path / "prepared/val.jsonl").unlink()
    assert "official_mtp_validation_required" in {item["code"] for item in mtp.build_drafter_training_plan(config)["issues"]}
    config["run"].pop("purpose")
    assert mtp.build_drafter_training_plan(config)["ready"] is True


def test_optin_encode_never_truncates_and_legacy_still_does():
    row = {"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, 4]}
    with pytest.raises(mtp.MTPDrafterTrainingError, match="no truncation"):
        mtp._encode_row(row, None, 3, overflow_policy="error")
    assert len(mtp._encode_row(row, None, 3)["input_ids"]) == 3


def test_encode_forwards_exact_target_chat_kwargs():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return [1, 2] if kwargs["add_generation_prompt"] else [1, 2, 3]

    row = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}
    encoded = mtp._encode_row(row, Tokenizer(), 10, overflow_policy="error",
                              chat_template_kwargs={"enable_thinking": False})
    assert encoded["labels"] == [-100, -100, 3]
    assert all(call["enable_thinking"] is False for call in calls)


def test_finite_gradient_guard_rejects_nan_and_all_zero():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.zeros_like(model.weight)
    with pytest.raises(mtp.MTPDrafterTrainingError, match="nonzero"):
        mtp._check_finite_gradients(model)
    model.weight.grad.fill_(float("nan"))
    with pytest.raises(mtp.MTPDrafterTrainingError, match="nonfinite"):
        mtp._check_finite_gradients(model)
    model.weight.grad.fill_(1)
    mtp._check_finite_gradients(model)


def test_efficient_target_keeps_full_conditioning_sequence():
    torch = pytest.importorskip("torch")
    calls = []

    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, **kwargs):
            calls.append((input_ids.shape, kwargs))
            hidden = self.embedding(input_ids)
            kv = hidden.unsqueeze(1)
            return SimpleNamespace(hidden_states=[hidden], shared_kv_states={"full_attention": (kv, kv)})

    class Assistant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(4, 2)
            self.head = torch.nn.Linear(2, 8)

        def forward(self, inputs_embeds, **kwargs):
            hidden = self.project(inputs_embeds)
            return SimpleNamespace(logits=self.head(hidden), last_hidden_state=hidden)

    target, assistant = Target(), Assistant()
    target.requires_grad_(False)
    loss = mtp._rollout_loss(target_model=target, assistant_model=assistant, input_ids=[1, 2, 3, 4],
                             labels=[-100, 2, 3, 4], anchors=[0, 1], draft_steps=4,
                             device=torch.device("cpu"), efficient_target=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert calls[0][0] == (1, 4)
    assert calls[0][1]["logits_to_keep"] == 1
    assert calls[0][1]["output_hidden_states"] is True
    assert calls[0][1]["return_shared_kv_states"] is True
    assert all(parameter.grad is None for parameter in target.parameters())
    mtp._check_finite_gradients(assistant)


def test_trainer_restores_registered_controllers_when_initialization_fails(monkeypatch):
    calls = []

    def fail(config, *, cleanup, **kwargs):
        cleanup.callback(lambda: calls.append("restored"))
        raise RuntimeError("assistant load failed")

    monkeypatch.setattr(mtp, "_train_mtp_drafter_impl", fail)
    with pytest.raises(RuntimeError, match="assistant load failed"):
        mtp.train_mtp_drafter({}, config_path="unused")
    assert calls == ["restored"]


def test_real_tiny_gemma4_shared_kv_assistant_qat_backward(tmp_path):
    """Random tiny CPU fixture, not pretrained loading, training, or a CUDA claim."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4AssistantForCausalLM"):
        pytest.skip("Installed Transformers lacks Gemma4Assistant")
    text = {"vocab_size": 16, "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4, "global_head_dim": 4,
            "max_position_embeddings": 32, "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0}
    target = transformers.Gemma4ForCausalLM(transformers.Gemma4TextConfig(**text, num_kv_shared_layers=2))
    assistant_config = transformers.Gemma4AssistantConfig(
        text_config=transformers.Gemma4TextConfig(**text, num_kv_shared_layers=4),
        backbone_hidden_size=8, use_ordered_embeddings=True, num_centroids=4, centroid_intermediate_top_k=2)
    assistant = transformers.Gemma4AssistantForCausalLM(assistant_config)
    assistant.masked_embedding.token_ordering.copy_(torch.arange(16))
    target.eval().requires_grad_(False)
    mtp._deployment_trainable_parameters(assistant)
    controller = mtp.DrafterQATController(load_yaml(ROOT / "configs/models/gemma4_e2b_mtp_drafter_qat.yaml"))
    try:
        controller.prepare(assistant)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = mtp._rollout_loss(target_model=target, assistant_model=assistant, input_ids=[1, 2, 3, 4],
                                     labels=[-100, 2, 3, 4], anchors=[0, 1], draft_steps=4,
                                     device=torch.device("cpu"), efficient_target=True)
        assert torch.isfinite(loss)
        loss.backward()
        mtp._check_finite_gradients(assistant)
        assert all(parameter.grad is None for parameter in target.parameters())
        assert all(parameter.dtype == torch.float32 for parameter in assistant.parameters())
        checkpoint = tmp_path / "tiny-assistant"
        assistant.save_pretrained(checkpoint, safe_serialization=True)
        loaded = transformers.Gemma4AssistantForCausalLM.from_pretrained(checkpoint, local_files_only=True)
        for key, value in assistant.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    finally:
        controller.restore()


def test_environment_probe_is_offline_and_rejects_incomplete_assistant(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4AssistantForCausalLM"):
        pytest.skip("Installed Transformers lacks Gemma4Assistant")
    import huggingface_hub

    calls = []

    def snapshot(model, **kwargs):
        calls.append((model, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4_assistant", "text_config": {"num_hidden_layers": 4}}), encoding="utf-8")
    with pytest.raises(mtp.MTPDrafterTrainingError, match="incomplete"):
        mtp.probe_official_mobile_mtp_environment()
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    report = mtp.probe_official_mobile_mtp_environment()
    assert report["passed"] is True and report["model_loaded"] is False
    assert all(kwargs == {"local_files_only": True} for _, kwargs in calls)


def test_optin_training_loop_emits_bound_target_qat_provenance(tmp_path, monkeypatch):
    """One scalar fixture update; no CUDA allocation, checkpoint download, or model load."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = _fixture(tmp_path)
    config["training"].update(max_steps=1, gradient_accumulation_steps=1)
    config_path = tmp_path / "mtp.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    events = []

    class Gemma4AssistantFixture(torch.nn.Module):
        config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(num_hidden_layers=4))

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def save_pretrained(self, path, **kwargs):
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.safetensors").write_bytes(b"scalar-fixture")
            (path / "config.json").write_text("{}", encoding="utf-8")

    class Target(torch.nn.Module):
        def to(self, device):
            return self

    target = Target()
    assistant = Gemma4AssistantFixture()
    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    target_summary = {"enabled": True, "true_fake_quant": True, "wrapped_effective_lora_count": 205,
                      "frozen_activation_module_count": 70}
    target_controller = SimpleNamespace(summary=lambda: target_summary, restore=lambda: events.append("target_restore"))

    class Controller:
        def __init__(self, config):
            pass

        def prepare(self, model):
            return self

        def summary(self):
            return {"enabled": True}

        def restore(self):
            events.append("assistant_restore")

    class Accelerator:
        device = SimpleNamespace(type="cuda")
        num_processes = 1
        process_index = 0
        is_main_process = True
        sync_gradients = True

        def __init__(self, *, gradient_accumulation_steps, mixed_precision):
            self.gradient_accumulation_steps = gradient_accumulation_steps
            assert mixed_precision == "bf16"

        def prepare(self, *args):
            return args

        def accumulate(self, model):
            return nullcontext()

        def autocast(self):
            return nullcontext()

        def backward(self, loss):
            loss.backward()

        def clip_grad_norm_(self, parameters, norm):
            return torch.nn.utils.clip_grad_norm_(parameters, norm)

        def gather_for_metrics(self, value):
            return value

        def wait_for_everyone(self):
            pass

        def unwrap_model(self, model):
            return model

        def save(self, *args, **kwargs):
            pass

    writer = SimpleNamespace(add_scalar=lambda *args: None, flush=lambda: None, close=lambda: None)
    monkeypatch.setitem(sys.modules, "accelerate", SimpleNamespace(Accelerator=Accelerator))
    monkeypatch.setitem(sys.modules, "accelerate.utils", SimpleNamespace(set_seed=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=lambda **kwargs: writer))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(mtp, "probe_official_mobile_mtp_environment", lambda: {"passed": True})
    monkeypatch.setattr(mtp, "_load_official_target", lambda *args: (target, tokenizer, target_controller))
    monkeypatch.setattr(mtp, "_deployment_trainable_parameters", lambda model: {"weight": model.weight})
    monkeypatch.setattr(mtp, "DrafterQATController", Controller)

    def load_assistant(model_id, **kwargs):
        assert kwargs["torch_dtype"] == torch.float32 and kwargs["local_files_only"] is True
        return assistant

    def rollout(**kwargs):
        assert kwargs["efficient_target"] is True
        return assistant.weight.square().sum()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_assistant)
    monkeypatch.setattr(mtp, "_rollout_loss", rollout)
    result = mtp.train_mtp_drafter(config, config_path=config_path)
    assert result["global_steps"] == 1 and result["training_executed"] is True
    assert result["target_qat"] == target_summary
    assert result["source_bindings"] == mtp.official_mobile_source_bindings(config)
    for role in ("best_checkpoint", "final_checkpoint"):
        metadata = json.loads((tmp_path / "run" / role / "mtp_drafter_training_metadata.json").read_text(encoding="utf-8"))
        assert metadata["target_qat"] == target_summary
        assert metadata["source_bindings"] == result["source_bindings"]
        assert metadata["assistant_master_dtype"] == "float32"
    assert "target_restore" in events and "assistant_restore" in events


def test_frozen_target_loads_seed_and_adapter_before_exact_qat(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    config = _fixture(tmp_path)
    peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=None))
    monkeypatch.setitem(sys.modules, "peft", peft)
    from ir_training.models import registry
    from ir_training.qat import fake_quant
    from ir_training.train import prepared_binding

    events = []

    class Gemma4ForCausalLM(torch.nn.Module):
        config = SimpleNamespace(model_type="gemma4_text")

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def get_base_model(self):
            return self

    target = Gemma4ForCausalLM()
    tokenizer = object()
    adapter = SimpleNamespace(load_model=lambda: target, load_tokenizer=lambda: tokenizer)

    def factory(model_config):
        assert model_config["model_source"] == str(tmp_path / "seed")
        events.append("seed")
        return adapter

    def load_adapter(model, path, *, is_trainable):
        assert path == str(tmp_path / "adapter") and is_trainable is False
        events.append("adapter")
        return model

    summary = {"true_fake_quant": True, "wrapped_effective_lora_count": 205,
               "frozen_activation_module_count": 70, "trainable_scope": {"all_parameters_frozen": True}}
    controller = SimpleNamespace(summary=lambda: summary, restore=lambda: events.append("restore"))

    def prepare(model, resolved):
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert not model.training
        events.append("qat")
        return controller

    monkeypatch.setattr(registry, "create_adapter", factory)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", load_adapter)
    monkeypatch.setattr(prepared_binding, "verify_tokenizer_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(fake_quant, "prepare_qat_model", prepare)
    loaded, _, _ = mtp._load_official_target(config, torch.device("cpu"))
    assert loaded is target
    assert events == ["seed", "adapter", "qat"]
    summary["frozen_activation_module_count"] = 69
    with pytest.raises(mtp.MTPDrafterTrainingError, match="205"):
        mtp._load_official_target(config, torch.device("cpu"))
    assert events[-1] == "restore"
