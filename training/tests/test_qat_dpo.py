"""Tiny-model safety proofs; no model download, CUDA or training run required."""
from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ir_training.train.qat_dpo import (
    DPO_HEALTH_WINDOW,
    ReferenceLogps,
    build_dpo_trainer,
    completion_log_probability,
    dpo_loss,
    health_snapshot,
    parameter_update_max_abs,
    precompute_reference_cache,
    serial_dpo_backward,
    snapshot_trainable_parameters,
    verify_dpo_evidence,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.arange(36, dtype=torch.float32).reshape(6, 6) / 100,
                                       requires_grad=False)
        self.lora_delta = torch.nn.Parameter(torch.tensor([[.1, -.2, .05, .4, -.1, .2]]).repeat(6, 1))
        self.seen = []

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.seen.append((self.training, torch.is_grad_enabled(), use_cache))
        return SimpleNamespace(logits=(self.base + self.lora_delta)[input_ids])

    def disable_adapter(self):
        raise AssertionError("DPO reference must never disable the SFT adapter")


def pair(identifier="p"):
    result = {"pair_id": identifier, "source_id": "s", "prompt_input_ids": [0, 1]}
    for side, ids in (("chosen", [0, 1, 2, 5]), ("rejected", [0, 1, 3, 5])):
        result[f"{side}_input_ids"] = ids
        result[f"{side}_attention_mask"] = [1] * 4
        result[f"{side}_labels"] = [-100, -100, *ids[2:]]
        result[f"{side}_completion_mask"] = [False, False, True, True]
    return result


def test_causal_logps_mask_prompt_but_include_eos():
    model, row = TinyPolicy(), pair()
    actual = completion_log_probability(model, row, "chosen")
    logits = (model.base + model.lora_delta)
    expected = logits[1].log_softmax(-1)[2] + logits[2].log_softmax(-1)[5]
    assert torch.allclose(actual, expected)
    actual.backward()
    assert model.lora_delta.grad[0].eq(0).all()
    assert model.lora_delta.grad[1].ne(0).any()
    assert model.lora_delta.grad[2].ne(0).any()
    assert model.lora_delta.grad[5].eq(0).all()


@pytest.mark.parametrize("failure", ["first_token", "masked_label", "wrong_target", "empty_mask", "padding"])
def test_invalid_completion_masks_fail_closed(failure):
    row = pair()
    if failure == "first_token":
        row["chosen_completion_mask"][0] = True
    elif failure == "masked_label":
        row["chosen_labels"][1] = 1
    elif failure == "wrong_target":
        row["chosen_labels"][2] = 4
    elif failure == "empty_mask":
        row["chosen_completion_mask"] = [False] * 4
    else:
        row["chosen_attention_mask"][1] = 0
    with pytest.raises(ValueError):
        completion_log_probability(TinyPolicy(), row, "chosen")


@pytest.mark.parametrize("beta", [0, -1, float("inf"), float("nan"), True])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(torch.tensor(-2.), torch.tensor(-3.), -2., -3., beta)


def test_dpo_at_reference_is_log2_with_correct_gradient_direction():
    chosen = torch.tensor(-5., requires_grad=True)
    rejected = torch.tensor(-7., requires_grad=True)
    loss = dpo_loss(chosen, rejected, -5., -7., .1)
    assert float(loss.detach()) == pytest.approx(0.69314718)
    loss.backward()
    assert chosen.grad.item() == pytest.approx(-.05)
    assert rejected.grad.item() == pytest.approx(.05)


def test_reference_is_actual_sft_policy_no_grad_and_mode_restored(tmp_path):
    model, row = TinyPolicy(), pair()
    model.train()
    original = model.lora_delta.detach().clone()
    refs = precompute_reference_cache(model, [row], tmp_path / "ref.json", {"sft": "immutable"})
    assert model.training
    assert all(item == (False, False, False) for item in model.seen)
    assert model.lora_delta.grad is None
    assert torch.equal(model.lora_delta, original)
    with torch.no_grad():
        assert refs[0].chosen == pytest.approx(float(completion_log_probability(model, row, "chosen")))
        model.lora_delta.zero_()
        assert refs[0].chosen != pytest.approx(float(completion_log_probability(model, row, "chosen")))
    with pytest.raises(FrozenInstanceError):
        refs[0].chosen = 0


def test_reference_is_immutable_and_binds_full_tokens_config(tmp_path):
    path, model = tmp_path / "ref.json", TinyPolicy()
    precompute_reference_cache(model, [pair()], path, {"sft": "abc", "config": "def"})
    original = path.read_bytes()
    payload = json.loads(original)
    assert payload["binding"] == {"sft": "abc", "config": "def"}
    assert len(payload["tokenized_pairs_sha256"]) == 64
    assert payload["temperature"] == 1.0
    assert payload["completion_reduction"] == "sum_with_eos"
    with pytest.raises(ValueError, match="fresh run"):
        precompute_reference_cache(model, [pair()], path, {"sft": "changed"})
    assert path.read_bytes() == original


def test_reference_refuses_post_backward_policy(tmp_path):
    model = TinyPolicy()
    completion_log_probability(model, pair(), "chosen").backward()
    with pytest.raises(ValueError, match="before optimizer"):
        precompute_reference_cache(model, [pair()], tmp_path / "ref.json", {})


def test_reference_restores_exact_module_modes_after_exception(tmp_path):
    model = TinyPolicy()
    model.sub = torch.nn.Dropout(0)
    model.train()
    model.sub.eval()
    row = pair()
    row["chosen_completion_mask"] = []
    with pytest.raises(ValueError):
        precompute_reference_cache(model, [row], tmp_path / "ref.json", {})
    assert model.training and not model.sub.training
    assert not (tmp_path / "ref.json").exists()


@pytest.mark.parametrize("scale", [1., .25, 2.])
def test_serial_gradient_matches_exact_paired_dpo(scale):
    model = TinyPolicy()
    expected = copy.deepcopy(model)
    rows = [pair("a"), pair("b")]
    references = {row["pair_id"]: ReferenceLogps(row["pair_id"], -3.2, -4.5) for row in rows}
    loss = sum(dpo_loss(completion_log_probability(expected, row, "chosen"),
        completion_log_probability(expected, row, "rejected"), -3.2, -4.5, .2) for row in rows) / len(rows)
    (loss * scale).backward()
    metrics = serial_dpo_backward(model, rows, references, beta=.2,
        backward=lambda value: value.backward(), scale=scale)
    assert torch.allclose(model.lora_delta.grad, expected.lora_delta.grad, atol=1e-7)
    assert metrics["loss"] == pytest.approx(float(loss.detach()))
    assert len(model.seen) == 8  # coefficient + chosen/rejected gradient passes per pair
    assert [grad for _, grad, _ in model.seen] == [False, False, True, True] * 2


def test_serial_backward_does_not_modify_reference_values():
    model, row = TinyPolicy(), pair()
    references = {"p": ReferenceLogps("p", -3., -4.)}
    serial_dpo_backward(model, [row], references, beta=.1, backward=lambda value: value.backward())
    assert references == {"p": ReferenceLogps("p", -3., -4.)}


def test_updates_improve_preference_margin_against_frozen_sft_reference(tmp_path):
    model, row = TinyPolicy(), pair()
    reference = precompute_reference_cache(model, [row], tmp_path / "ref.json", {})[0]
    original = (tmp_path / "ref.json").read_bytes()
    optimizer = torch.optim.SGD([model.lora_delta], lr=.5)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        serial_dpo_backward(model, [row], {"p": reference}, beta=.2, backward=lambda value: value.backward())
        optimizer.step()
    with torch.no_grad():
        delta = (completion_log_probability(model, row, "chosen") - completion_log_probability(model, row, "rejected")
                 - reference.chosen + reference.rejected)
    assert float(delta) > 0
    assert (tmp_path / "ref.json").read_bytes() == original


@pytest.mark.parametrize("accelerator_gas", [1, 4])
def test_training_step_scales_accumulation_exactly_once(monkeypatch, accelerator_gas):
    from ir_training.train import qat_grpo
    monkeypatch.setattr(qat_grpo, "assert_qat_policy_active", lambda *args: None)
    model, expected = TinyPolicy(), TinyPolicy()
    reference = ReferenceLogps("p", -3., -4.)
    trainer_type = build_dpo_trainer(object, SimpleNamespace(controller=None), (reference,), .1)
    trainer = object.__new__(trainer_type)
    trainer.is_deepspeed_enabled = trainer.is_fsdp_enabled = False
    trainer.optimizer = None
    trainer.args = SimpleNamespace(gradient_accumulation_steps=4)
    trainer.current_gradient_accumulation_steps = 4
    trainer.dpo_update_metrics = []
    trainer.accelerator = SimpleNamespace(autocast=nullcontext, gradient_accumulation_steps=accelerator_gas,
        backward=lambda loss: (loss / accelerator_gas).backward(), device="cpu")
    actual = trainer.training_step(model, {"pairs": [pair()]})
    loss = dpo_loss(completion_log_probability(expected, pair(), "chosen"),
        completion_log_probability(expected, pair(), "rejected"), -3., -4., .1) / 4
    loss.backward()
    assert torch.allclose(model.lora_delta.grad, expected.lora_delta.grad, atol=1e-7)
    assert float(actual) == pytest.approx(float(loss.detach()))


def test_dpo_reference_and_health_export_evidence_is_bound(tmp_path):
    from ir_training.train.qat_dpo import _identity
    from ir_training.train.qat_posttrain_data import _manifest, _reward_config
    reward_config = Path(__file__).resolve().parents[2] / "dataset/configs/genui_metric_v5_4.yaml"
    raw_pairs = tmp_path / "pairs.jsonl"
    raw_pairs.write_text('{"source_id":"s"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest("qat_dpo", [pair()], "a" * 64, _reward_config(reward_config),
        {"max_prompt_length": 4, "max_completion_length": 4, "max_sequence_length": 8})
    manifest.update(raw_pairs=_identity(raw_pairs), pairs_sha256=_identity(raw_pairs)["sha256"],
                    unique_train_sources=1, min_reward_gap=1e-6)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reference_path = tmp_path / "reference.json"
    binding = {"source_lineage": {"verified": True}, "training_config_sha256": "abc",
               "preference_manifest_sha256": _identity(manifest_path)["sha256"], "tokenizer_binding": {"sha": "def"}}
    precompute_reference_cache(TinyPolicy(), [pair()], reference_path, binding)
    cache = json.loads(reference_path.read_text(encoding="utf-8"))
    config = {"training": {"max_seq_length": 8}, "grpo": {"objective": "dpo", "max_prompt_length": 4,
        "max_completion_length": 4, "reward_config": str(reward_config), "dpo": {"beta": .1, "preferences": str(raw_pairs),
                                                     "preferences_sha256": _identity(raw_pairs)["sha256"]}}}
    metadata = {"training_config_sha256": "abc", "checkpoint_step": 20,
        "prepared_dataset_binding": {"sha": "def"}, "grpo": {"algorithm": "dpo", "source_lineage": {"verified": True},
        "dpo": {"beta": .1, "reference_cache": _identity(reference_path), "preference_manifest": _identity(manifest_path),
                "tokenized_pairs_sha256": cache["tokenized_pairs_sha256"]},
        "health": health_snapshot(good_health_rows(), 20)}}
    assert all(verify_dpo_evidence(config, metadata).values())
    changed_config = copy.deepcopy(config)
    changed_config["grpo"]["dpo"]["preferences_sha256"] = "0" * 64
    assert not verify_dpo_evidence(changed_config, metadata)["dpo_preference_manifest_bound"]
    for field in ("source_lineage", "health", "algorithm"):
        tampered = copy.deepcopy(metadata)
        tampered["grpo"][field] = {} if field != "algorithm" else "grpo"
        assert not all(verify_dpo_evidence(config, tampered).values())
    changed = copy.deepcopy(metadata)
    changed["checkpoint_step"] = 21
    assert not verify_dpo_evidence(config, changed)["selected_step_dpo_health_window_passed"]
    raw_pairs.write_text("changed", encoding="utf-8")
    assert not verify_dpo_evidence(config, metadata)["dpo_preference_manifest_bound"]
    reference_path.write_text("{}", encoding="utf-8")
    assert not verify_dpo_evidence(config, metadata)["dpo_reference_bound"]


def good_health_rows():
    return [{"loss": .6, "margin": .2, "preference_accuracy": .5,
             "gradient_l2_norm": .1, "finite_parameters": 1., "parameter_update_max_abs": .001}
            for _ in range(DPO_HEALTH_WINDOW)]


def test_dpo_health_uses_no_fabricated_grpo_variance():
    report = health_snapshot(good_health_rows(), 20)
    assert report["passed_window"]
    assert report["schema"] == "dpo_health_v1"
    assert "reward_std" not in str(report)
    assert not health_snapshot(good_health_rows()[:-1], 19)["passed_window"]


def test_dpo_health_requires_real_updates_but_allows_initial_zero_lr_warmup():
    rows = good_health_rows()
    rows[0]["parameter_update_max_abs"] = 0
    assert health_snapshot(rows, 20)["passed_window"]
    for row in rows:
        row["parameter_update_max_abs"] = 0
    assert not health_snapshot(rows, 20)["passed_window"]


def test_update_snapshot_only_copies_lora_and_detects_real_change():
    model = TinyPolicy()
    before = snapshot_trainable_parameters(model)
    assert set(before) == {"lora_delta"}
    assert before["lora_delta"].device.type == "cpu"
    assert parameter_update_max_abs(model, before) == 0
    with torch.no_grad():
        model.lora_delta.add_(.125)
    assert parameter_update_max_abs(model, before) == pytest.approx(.125)
    model.base.requires_grad_(True)
    with pytest.raises(ValueError, match="only LoRA"):
        snapshot_trainable_parameters(model)


@pytest.mark.parametrize("field,value", [("loss", float("nan")), ("gradient_l2_norm", 0),
    ("finite_parameters", 0), ("preference_accuracy", 2), ("loss", -1)])
def test_dpo_health_rejects_unsafe_window(field, value):
    rows = good_health_rows()
    rows[-1][field] = value
    assert not health_snapshot(rows, 20)["passed_window"]


def test_dpo_evidence_missing_or_grpo_health_fails():
    assert not any(verify_dpo_evidence({}, {}).values())


def test_dpo_prediction_step_uses_own_loss_not_model_pairs_argument(monkeypatch):
    from ir_training.train import qat_grpo
    monkeypatch.setattr(qat_grpo, "assert_qat_policy_active", lambda *args: None)
    model = TinyPolicy()
    trainer_type = build_dpo_trainer(object, SimpleNamespace(controller=object()),
        (ReferenceLogps("p", -3., -4.),), .1)
    trainer = object.__new__(trainer_type)
    trainer.accelerator = SimpleNamespace(autocast=nullcontext)
    loss, logits, labels = trainer.prediction_step(model, {"pairs": [pair()]}, True)
    assert torch.isfinite(loss) and not loss.requires_grad
    assert logits is None and labels is None
    assert all(not grad for _, grad, _ in model.seen)


def test_dpo_save_restores_generation_cache_even_on_failure():
    class Base:
        def save_model(self, *args, **kwargs):
            assert self.model.generation_config.cache_implementation is None
            raise OSError("save failed")
    trainer_type = build_dpo_trainer(Base, SimpleNamespace(controller=None), (), .1)
    trainer = object.__new__(trainer_type)
    trainer.model = SimpleNamespace(generation_config=SimpleNamespace(cache_implementation="hybrid"))
    with pytest.raises(OSError):
        trainer.save_model("unused")
    assert trainer.model.generation_config.cache_implementation == "hybrid"


@pytest.mark.parametrize("tamper_during_admission", [False, True])
def test_mocked_dpo_runner_writes_bound_selected_checkpoint(tmp_path, monkeypatch, tamper_during_admission):
    """Real objective, cache, health and checkpoint writers; mocked GPU/I/O gates.

    This proves runner wiring only, not a CUDA/TRL/real-model training result.
    """
    import sys
    from types import ModuleType

    import yaml
    from ir_training.train import (
        callbacks,
        grpo_experiment,
        grpo_runtime,
        prepared_binding,
        qat_grpo,
        qat_posttrain_data,
        sft,
    )
    from ir_training.train.qat_dpo import _identity, train_qat_dpo

    output, dataset = tmp_path / "run", tmp_path / "prepared"
    dataset.mkdir()
    for split in ("train", "val"):
        (dataset / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
    raw_pairs = tmp_path / "pairs.jsonl"
    raw_pairs.write_text("{}\n", encoding="utf-8")
    reward_path = Path(__file__).resolve().parents[2] / "dataset/configs/genui_metric_v5_4.yaml"
    config = {"run": {"output_dir": str(output), "dataset_dir": str(dataset), "id": "dpo-test"},
        "model": {"dtype": "float32", "max_context_tokens": 64}, "lora": {},
        "training": {"method": "qat_lora_grpo", "max_steps": 20, "epochs": 1, "max_seq_length": 8,
                     "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
                     "mixed_precision": "off", "eval_steps": 20, "save_steps": 20},
        "grpo": {"objective": "dpo", "family": "270m", "max_prompt_length": 4,
                 "max_completion_length": 4, "reward_config": str(reward_path),
                 "quality_control": {"enabled": True}, "dpo": {"beta": .1,
                 "preferences": str(raw_pairs), "preferences_sha256": _identity(raw_pairs)["sha256"]}},
        "golden_eval": {"enabled": True}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    model = TinyPolicy()
    model.generation_config = SimpleNamespace(cache_implementation="hybrid")
    restored = []
    controller = SimpleNamespace(saturation_monitor=None, restore=lambda: restored.append(True))
    metadata = {"training_metadata_version": 4, "checkpoint_kind": "lora_adapter",
        "training": config["training"], "model": config["model"], "lora": {},
        "initialization_seed": 42, "training_config_sha256": _identity(config_path)["sha256"],
        "grpo": {"schema_version": 1, "algorithm": "dpo", "family": "270m", "source_lineage": {"verified": True}}}
    bundle = qat_grpo.QATGRPOModel(config, model, SimpleNamespace(save_pretrained=lambda path: None),
                                 None, metadata, [5], controller)
    monkeypatch.setattr(qat_grpo, "validate_qat_grpo_config", lambda *a: None)
    monkeypatch.setattr(qat_grpo, "load_qat_grpo_model", lambda *a, **k: bundle)
    monkeypatch.setattr(qat_grpo, "verify_prepared_training_contract", lambda *a: {"verified": True})
    monkeypatch.setattr(qat_grpo, "run_qat_grpo_preflight", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(qat_grpo, "assert_qat_policy_active", lambda *a: None)
    monkeypatch.setattr(prepared_binding, "verify_tokenizer_binding", lambda *a, **k: {"verified": True})
    monkeypatch.setattr(qat_grpo, "_script_module", lambda name: SimpleNamespace(load_training_dataset=lambda *a, **k: [{"source_id": "fixture"}]))
    monkeypatch.setattr(sft, "_model_position_limit", lambda *a: 64)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(grpo_runtime, "dependency_report", lambda: {"packages": {"trl": {}}})
    monkeypatch.setattr(callbacks, "preserve_generation_eos", lambda *a, **k: [5])

    def prepared_pairs(*args, **kwargs):
        assert kwargs["forbidden_rows"] == [{"source_id": "fixture"}]
        manifest = qat_posttrain_data._manifest("qat_dpo", [pair()], "a" * 64,
            qat_posttrain_data._reward_config(reward_path),
            {key: kwargs[key] for key in ("max_prompt_length", "max_completion_length", "max_sequence_length")})
        manifest.update(pairs_sha256=_identity(raw_pairs)["sha256"], unique_train_sources=1, min_reward_gap=1e-6)
        if tamper_during_admission:
            raw_pairs.write_text("changed after admission", encoding="utf-8")
        return [pair()], manifest

    monkeypatch.setattr(qat_posttrain_data, "prepare_dpo_pairs", prepared_pairs)

    class Callback:
        def on_pre_optimizer_step(self, *args, **kwargs):
            pass

    selected = output / "best"

    class Golden(Callback):
        def summary(self):
            return {"checkpoint_dir": str(selected), "step": 20, "epoch": 1., "metric_value": 90.}

    monkeypatch.setattr(sft, "_build_optional_golden_callback", lambda **k: Golden())
    monkeypatch.setattr(grpo_experiment, "make_quality_callback", lambda **k: SimpleNamespace(finalize=lambda: {"accepted": True}))

    class Arguments:
        def __init__(self, optim=None, lr_scheduler_type=None, tf32=False,
                     dataloader_num_workers=0, dataloader_pin_memory=True,
                     dataloader_persistent_workers=False, **kwargs):
            vars(self).update(kwargs)

        def to_dict(self):
            return dict(vars(self))

    class Trainer:
        def __init__(self, model, args, train_dataset, **kwargs):
            self.model, self.args, self.data = model, args, train_dataset
            self.callbacks = []
            self.is_deepspeed_enabled = self.is_fsdp_enabled = False
            self.optimizer = None
            self.current_gradient_accumulation_steps = 1
            self.accelerator = SimpleNamespace(autocast=nullcontext, gradient_accumulation_steps=1,
                backward=lambda loss: loss.backward(), device=torch.device("cpu"), reduce=lambda value, **k: value)
            self.state = SimpleNamespace(global_step=0, epoch=0., is_world_process_zero=True, log_history=[])

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def fire(self, event):
            for callback in self.callbacks:
                method = getattr(callback, event, None)
                if method:
                    method(self.args, self.state, None, model=self.model)

        def log(self, values):
            self.state.log_history.append(values)

        def train(self):
            self.fire("on_train_begin")
            for step in range(1, 21):
                self.model.zero_grad(set_to_none=True)
                self.training_step(self.model, {"pairs": self.data})
                self.fire("on_pre_optimizer_step")
                with torch.no_grad():
                    for parameter in self.model.parameters():
                        if parameter.requires_grad:
                            parameter.add_(parameter.grad, alpha=-.05)
                self.state.global_step, self.state.epoch = step, step / 20
                self.fire("on_step_end")
            self.save_model(str(selected))
            self.save_model(str(output / "checkpoint-20"))
            self.fire("on_save")
            self.fire("on_train_end")

        def save_model(self, output_dir=None, **kwargs):
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"tiny-adapter-fixture")

        def is_world_process_zero(self):
            return True

    transformers = ModuleType("transformers")
    transformers.Trainer, transformers.TrainingArguments, transformers.TrainerCallback = Trainer, Arguments, Callback
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    if tamper_during_admission:
        with pytest.raises(ValueError, match="changed during pair admission"):
            train_qat_dpo(config, config_path)
        assert not (output / "dpo_reference.rank0.json").exists()
        assert not (output / "final_adapter").exists()
        return
    result = train_qat_dpo(config, config_path)
    assert result["checkpoint_step"] == 20
    saved = json.loads((selected / "training_metadata.json").read_text(encoding="utf-8"))
    assert saved["grpo"]["algorithm"] == "dpo"
    assert all(verify_dpo_evidence(config, saved).values())
    assert saved["grpo"]["health"]["last_step"] == saved["checkpoint_step"] == 20
    assert restored
    assert (output / "final_adapter/adapter_model.safetensors").exists()
