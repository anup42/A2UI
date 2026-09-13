from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.common.config import load_yaml
from ir_training.common.cuda_env import normalize_cuda_visible_devices
from ir_training.export.merge_lora import _training_provenance
from ir_training.qat.workflow import validate_qat_config
from ir_training.train.recipe import optimizer_steps, resolve_eval_strategy, validate_effective_batch, validate_sft_recipe
from ir_training.train.sft import _adapter_checkpoint_manifest, _align_tokenizer_and_model, _build_checked_causal_lm_trainer, _resolved_lora_targets, _tokenize_completion_only_row


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", ["gemma4_e2b_a2ui_express_review_sft.yaml", "gemma3_270m_a2ui_express_review_sft.yaml"])
def test_review_profiles_have_explicit_supported_methods(profile):
    config = load_yaml(ROOT / "configs/models" / profile)
    assert validate_sft_recipe(config) in {"lora_sft", "full_finetune_sft"}
    assert config["golden_eval"]["required_rows"] == 32
    assert config["golden_eval"]["max_new_tokens"] == 2048


def test_full_qat_support_keeps_contract_restrictions():
    config = load_yaml(ROOT / "configs/models/gemma3_270m_a2ui_express_qat.yaml")
    config["training"]["method"] = "full_finetune_qat"
    config.pop("lora")
    assert validate_sft_recipe(config) == "full_finetune_qat"
    assert not [issue for issue in validate_qat_config(config) if issue.severity == "error"]
    config["qat"]["scale_mode"] = "retained_mobile"
    with pytest.raises(ValueError, match="retained mobile"):
        validate_sft_recipe(config)


@pytest.mark.parametrize("training", [
    {"method": "full_finetune_unknown"},
    {"method": "qat_lora_sft"},
    {"refuse_resume": True, "resume_from_checkpoint": "/checkpoint"},
    {"trainer_backend": "trl"}, {"packing": True}, {"deepspeed": "zero2.json"},
    {"overflow_policy": "truncate"},
])
def test_unsupported_recipes_fail_before_model_load(training):
    with pytest.raises(ValueError):
        validate_sft_recipe({"training": training})


def test_eval_strategy_respects_no_and_requires_golden_trigger():
    assert resolve_eval_strategy({"eval_strategy": "no"}, {}, has_validation=True) == "no"
    assert resolve_eval_strategy({"eval_strategy": False}, {}, has_validation=True) == "no"
    with pytest.raises(ValueError, match="Golden"):
        resolve_eval_strategy({"eval_strategy": "no"}, {"enabled": True, "trigger": "evaluate"}, has_validation=True)
    with pytest.raises(ValueError, match="val.jsonl"):
        resolve_eval_strategy({"eval_strategy": "steps"}, {}, has_validation=False)


def test_effective_batch_refuses_three_gpu_legacy_mismatch():
    training = {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 4, "expected_effective_batch_size": 16}
    with pytest.raises(ValueError, match="actual 3"):
        validate_effective_batch(training, 3)
    assert validate_effective_batch(training, 4) == 16
    assert optimizer_steps(rows=169898, world_size=2, microbatch=1, accumulation=4, epochs=2) == 42476


def test_torchrun_visibility_cannot_be_changed_inside_worker(monkeypatch):
    env = {"LOCAL_RANK": "2", "WORLD_SIZE": "3", "CUDA_VISIBLE_DEVICES": "0,1,3", "A2UI_CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    assert normalize_cuda_visible_devices(env) == "0,1,3"
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("A2UI_CUDA_VISIBLE_DEVICES")
    script("train_sft")._apply_config_cuda_visibility({"runtime": {"cuda_visible_devices": "0,1,2,3"}})
    import os
    assert "A2UI_CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("prompt,completion,limit", [("PPPP", "CCC", 5), ("P", "CCCC", 4), ("", "C", 10), ("P", "", 10)])
def test_incomplete_conditioning_and_targets_never_enter_sft(prompt, completion, limit):
    tokenizer = lambda text, **kwargs: {"input_ids": [ord(char) for char in text]}
    with pytest.raises(ValueError):
        _tokenize_completion_only_row(tokenizer=tokenizer, prompt_text=prompt, completion_text=completion, full_text=prompt + completion, max_seq_length=limit)


def test_alignment_preserves_native_end_of_turn_ids():
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[1, 50, 106]), generation_config=SimpleNamespace(eos_token_id=[1, 106, 50]))
    tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0, bos_token_id=2)
    _align_tokenizer_and_model(tokenizer, model)
    assert set(model.generation_config.eos_token_id) == {1, 50, 106}
    assert model.config.eos_token_id == [1, 50, 106]


def test_regex_lora_targets_are_not_compared_as_character_sets():
    assert _resolved_lora_targets(SimpleNamespace(target_modules="a.*b"), None) == "a.*b"
    assert _resolved_lora_targets(SimpleNamespace(target_modules=["q_proj", "v_proj"]), None) == {"q_proj", "v_proj"}


def test_full_checkpoint_manifest_hashes_weights_and_tokenizer(tmp_path):
    for name in ("config.json", "model-00001-of-00001.safetensors", "model.safetensors.index.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    manifest = _adapter_checkpoint_manifest(tmp_path, role="final")
    assert manifest["checkpoint_kind"] == "full_model"
    assert {row["path"] for row in manifest["files"]} == {path.name for path in tmp_path.iterdir()}


def test_qat_merge_cannot_bypass_saved_evidence(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "training_metadata.json").write_text(json.dumps({"training": {"method": "qat_lora_sft"}, "qat": {"enabled": True}}))
    with pytest.raises(ValueError, match="original training config"):
        _training_provenance(adapter, None, base=tmp_path)
    cfg = tmp_path / "relabel.yaml"
    cfg.write_text("training:\n  method: lora_sft\n", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to relabel"):
        _training_provenance(adapter, cfg, base=tmp_path)


def test_checked_loss_accumulation_uses_mean_of_microbatches():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class Base:
        def __init__(self):
            self.model_accepts_loss_kwargs = True

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.4, -0.1, 0.3]))

        def forward(self, input_ids):
            return {"logits": self.weight.expand(*input_ids.shape, 3)}

    model = Tiny()
    trainer = _build_checked_causal_lm_trainer(Base)()
    # Unequal supervised lengths exercise the documented normalization choice.
    labels = [torch.tensor([[-100, 0, 1]]), torch.tensor([[-100, 2, -100]])]
    actual = sum(trainer.compute_loss(model, {"input_ids": torch.ones_like(label), "labels": label}, num_items_in_batch=3) for label in labels) / 2
    reference = (F.cross_entropy(model.weight.expand(2, 3), torch.tensor([0, 1])) + F.cross_entropy(model.weight[None, :], torch.tensor([2]))) / 2
    assert torch.allclose(actual, reference)
    actual_grad = torch.autograd.grad(actual, model.weight)[0]
    reference_grad = torch.autograd.grad(reference, model.weight)[0]
    assert torch.allclose(actual_grad, reference_grad)
    assert trainer.model_accepts_loss_kwargs is False


def test_launcher_print_plan_uses_config_device_count_without_spawning(tmp_path):
    import yaml
    config = load_yaml(ROOT / "configs/models/gemma3_270m_a2ui_express_review_sft.yaml")
    config["runtime"] = {"cuda_visible_devices": "0,1,3", "world_size": 3}
    config["training"].update(expected_effective_batch_size=24, gradient_accumulation_steps=8)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    command, env = script("launch_review_training").launch_plan(path)
    assert "--nproc_per_node=3" in command
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,3"


def test_actual_golden_shape_uses_metadata_query_id():
    identity = script("prepare_review_training").source_identity({"response_text": "actual response", "source_id": "source-group", "metadata": {"query_id": "q_017270"}})
    assert identity[0] == "q_017270"


def test_prepared_tokenizer_binding_checks_actual_loaded_tokenizer(tmp_path):
    from ir_training.train.prepared_binding import value_sha256, verify_tokenizer_binding
    from ir_training.train.resume_contract import file_sha256
    tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0}, chat_template="template")
    for name in ("train", "val"):
        (tmp_path / f"{name}.jsonl").write_text("{}\n")
    manifest = {"tokenizer": {"vocabulary_sha256": value_sha256({"a": 0}), "chat_template_sha256": value_sha256("template"), "chat_template_kwargs": {}},
        "splits": {name: {"output_sha256": file_sha256(tmp_path / f"{name}.jsonl")} for name in ("train", "val")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert verify_tokenizer_binding(tmp_path, tokenizer, {}, required=True)["verified"]
    tokenizer.chat_template = "different"
    with pytest.raises(ValueError, match="chat_template_sha256"):
        verify_tokenizer_binding(tmp_path, tokenizer, {}, required=True)


def test_resume_requires_optimizer_data_recipe_and_weight_identity(tmp_path):
    from ir_training.train.resume_contract import build_resume_contract, verify_resume_contract, file_sha256
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text("{}\n")
    contract = build_resume_contract({"training": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 1}}, dataset, effective_batch=1)
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="optimizer"):
        verify_resume_contract(checkpoint, contract)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"fixture")
    (checkpoint / "trainer_state.json").write_text('{"global_step":1}')
    metadata = {"resume_contract": contract, "adapter_checkpoints": [{"files": [{"path": "adapter_model.safetensors", "sha256": file_sha256(checkpoint / "adapter_model.safetensors")}]}]}
    (checkpoint / "training_metadata.json").write_text(json.dumps(metadata))
    assert verify_resume_contract(checkpoint, contract)["verified"]
    changed = dict(contract, effective_batch_size=2)
    with pytest.raises(ValueError):
        verify_resume_contract(checkpoint, changed)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        verify_resume_contract(checkpoint, contract)


def test_prepare_and_launch_binding_end_to_end_without_loading_model(tmp_path, monkeypatch):
    import yaml
    prepare = script("prepare_review_training")
    launch = script("launch_review_training")
    monkeypatch.setattr("ir_training.train.gpu_profile.detect_cuda_devices", lambda: {
        "version": 1, "inherited_cuda_visible_devices": None, "visible_gpu_count": 2,
        "devices": [{"visible_index": index, "launch_identifier": str(index), "uuid": f"GPU-{index}",
                     "name": "NVIDIA H100 80GB", "total_memory_bytes": 80 * 1024**3, "compute_capability": [9, 0]}
                    for index in range(2)],
    })
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture only; never loaded")
    data = tmp_path / "data"
    data.mkdir()
    for name, count, offset in (("train", 2, 0), ("val", 2, 10), ("golden32", 32, 100)):
        (data / f"{name}.jsonl").write_text("".join(json.dumps({"metadata": {"query_id": f"q_{index + offset}"}, "response_text": f"source {index + offset}"}) + "\n" for index in range(count)))
    (data / "prompt_scaffolds.json").write_text("[]")
    manifest = {"validation": {"strict_express": True, "wire_schema": True, "semantic_roundtrip": True, "root_reachability": 1.0},
        "scaffold_count": 1, "prompt_scaffolds_sha256": prepare.sha256(data / "prompt_scaffolds.json"),
        "tokenizer": {"max_seq_length": 4096, "vocabulary_sha256": "fixture", "chat_template_sha256": "fixture", "chat_template_kwargs": {}},
        "splits": {name: {"output_sha256": prepare.sha256(data / f"{name}.jsonl"), "max_accepted_token_lengths": {"prompt_tokens": 10}} for name in ("train", "val", "golden32")}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "run"
    args = SimpleNamespace(profile="270m", model_dir=model, dataset_dir=data, golden_file=data / "golden32.jsonl", output_dir=output,
        devices="0,1", microbatch=1, effective_batch=16, epochs=1, steps=20, max_seq_length=4096, resume=None, qv_baseline=False, qat=False)
    config, report = prepare.build_config(args)
    assert not report["training_executed"] and not report["model_loaded"]
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert report["optimizer_step_budget"] == 20
    assert report["effective_hyperparameters"]["learning_rate"] == 2e-5
    args.learning_rate, args.weight_decay, args.warmup_ratio, args.seed, args.logging_steps = 1e-5, 0.05, 0.05, 123, 5
    tuned_config, tuned_report = prepare.build_config(args)
    for key in ("learning_rate", "weight_decay", "warmup_ratio", "seed", "logging_steps"):
        assert tuned_config["training"][key] == getattr(args, key)
        assert tuned_report["effective_hyperparameters"][key] == getattr(args, key)
    args.qat = True
    qat_config, _ = prepare.build_config(args)
    assert qat_config["training"]["method"] == "full_finetune_qat"
    assert qat_config["training"]["learning_rate"] == 1e-5
    output.mkdir()
    config_path = output / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    report["training_config_sha256"] = prepare.sha256(config_path)
    (output / "preparation_report.json").write_text(json.dumps(report))
    launch.verify_launch_binding(config_path)
    config_path.write_text(config_path.read_text() + "\n# unreviewed edit\n")
    with pytest.raises(ValueError, match="config changed"):
        launch.verify_launch_binding(config_path)
