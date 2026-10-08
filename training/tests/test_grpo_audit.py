"""CPU regressions; mock orchestration is deliberately separate from real HF IO."""
from __future__ import annotations

import ast
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))

from ir_training.generation_policy import generation_cache_scope
from ir_training.train.grpo_audit import (
    build_audited_grpo_trainer, group_reward_statistics, isolated_generation_config,
    rollout_generation_config, validate_reward_values,
)
from ir_training.train.grpo_runtime import (
    GRPOHealthMonitor, HealthThresholds, audited_reward, make_express_rollout,
    normalize_rollout_completion,
)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def healthy(**updates):
    return {"frac_reward_zero_std": 0.0, "completions/clipped_ratio": 0.0,
            "grad_norm": 1.0, "sampled_parameter_max_delta": 0.01,
            "nonfinite_parameters_or_gradients": 0.0, **updates}


def test_between_group_variance_is_not_learning_signal(torch):
    rewards = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
    assert rewards.std() > 0  # TRL batch-scaling telemetry incorrectly calls these diverse.
    statistics = group_reward_statistics(rewards, [1.0], 2)
    assert statistics == {"frac_reward_zero_std": 1.0, "group_reward_std": 0.0}
    monitor = GRPOHealthMonitor(HealthThresholds(window_steps=1))
    result = monitor.observe(1, healthy(**statistics))
    assert result["issues"] == ["insufficient_reward_variance"]


def test_groups_cross_rank_boundaries_and_use_weighted_objective(torch):
    # Two rank-local slices, each shorter than a complete group. No per-rank
    # regrouping or new gather is allowed: TRL has already gathered these rows.
    rank0 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    rank1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    rewards = torch.cat([rank0, rank1])
    assert group_reward_statistics(rewards, [1.0, 1.0], 4)["frac_reward_zero_std"] == 1.0
    assert group_reward_statistics(rewards, [1.0, 0.0], 4)["frac_reward_zero_std"] == 0.0


def test_eval_single_generation_has_finite_zero_variance(torch):
    assert group_reward_statistics(torch.tensor([[0.2], [0.8]]), [1.0], 1) == {
        "frac_reward_zero_std": 1.0, "group_reward_std": 0.0}


@pytest.mark.parametrize("rewards,weights,generations", [
    ([[1.0], [2.0], [3.0]], [1.0], 2),
    ([[1.0], [2.0]], [1.0, 1.0], 2),
    ([[float("nan")], [0.0]], [1.0], 2),
    ([[float("inf")], [0.0]], [1.0], 2),
    ([[0.0], [1.0]], [float("inf")], 2),
    ([[0.0], [1.0]], [1.0], 0),
    ([[0.0], [1.0]], [1.0], True),
])
def test_group_statistics_reject_broken_inputs(torch, rewards, weights, generations):
    with pytest.raises(ValueError):
        group_reward_statistics(torch.tensor(rewards), weights, generations)


def fake_trl_base(torch):
    class FakeTRL:
        def __init__(self, rewards, mode="train", scaling="batch", model=None):
            self.model = model if model is not None else SimpleNamespace(training=mode == "train")
            self.rewards = rewards
            self.num_generations = 2
            self.num_generations_eval = 1
            self.reward_weights = torch.tensor([1.0])
            self.multi_objective_aggregation = "sum_then_normalize"
            self.scaling = scaling
            self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        def _calculate_rewards(self, *args, **kwargs):
            return self.rewards  # Simulates TRL's already-gathered return boundary.

        def _generate_and_score_completions(self, inputs):
            rewards = self._calculate_rewards(inputs)[:, 0]
            mode = "train" if self.model.training else "eval"
            size = self.num_generations if mode == "train" else self.num_generations_eval
            groups = rewards.reshape(-1, size)
            group_std = groups.std(1) if size > 1 else torch.zeros_like(groups[:, 0])
            scale = rewards.std().expand_as(rewards) if self.scaling == "batch" else group_std.repeat_interleave(size)
            advantages = rewards - groups.mean(1).repeat_interleave(size)
            if self.scaling != "none":
                advantages = advantages / (scale + 1e-4)
            self._metrics[mode]["frac_reward_zero_std"].append(torch.isclose(scale, torch.zeros_like(scale)).float().mean().item())
            return {"advantages": advantages, "rewards": self.rewards}

        def compute_loss(self, model, *args, **kwargs):
            return "unchanged loss"

    return FakeTRL


@pytest.mark.parametrize("scaling", ["batch", "group", "none"])
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_trainer_corrects_telemetry_without_changing_objective(torch, scaling, mode):
    base = fake_trl_base(torch)
    rewards = torch.tensor([[-1.0], [-1.0], [0.0], [1.0]])
    reference = base(rewards, mode, scaling)._generate_and_score_completions([])
    trainer = build_audited_grpo_trainer(base)(rewards, mode, scaling)
    actual = trainer._generate_and_score_completions([])
    assert actual["rewards"] is rewards
    assert torch.equal(actual["advantages"], reference["advantages"])
    metrics = trainer._metrics[mode]
    assert metrics["frac_reward_zero_std"] == [0.5 if mode == "train" else 1.0]
    assert len(metrics["trl/frac_reward_zero_std"]) == 1
    if scaling == "batch":
        assert metrics["trl/frac_reward_zero_std"] == [0.0]
    trainer._generate_and_score_completions([])
    assert len(metrics["frac_reward_zero_std"]) == 2
    assert len(metrics["grpo/group_reward_std"]) == 2


def test_qat_trainer_composes_health_and_retains_qat_guards(torch):
    from ir_training.train.qat_grpo import build_qat_grpo_trainer
    model = torch.nn.Module()
    model.lora_A = torch.nn.Parameter(torch.ones(1))
    original = model.forward
    model.forward = lambda *a, **k: None
    controller = SimpleNamespace(_original_forwards={model: original})
    model._a2ui_qat_controller = controller
    bundle = SimpleNamespace(controller=controller)
    trainer = build_qat_grpo_trainer(fake_trl_base(torch), bundle)(torch.tensor([[-1.0], [-1.0]]), model=model)
    trainer._generate_and_score_completions([])
    assert trainer._metrics["train"]["frac_reward_zero_std"] == [1.0]
    assert trainer.compute_loss(model) == "unchanged loss"
    model.forward = original
    with pytest.raises(RuntimeError, match="restored"):
        trainer._generate_and_score_completions([])
    with pytest.raises(RuntimeError, match="restored"):
        trainer.compute_loss(model)


@pytest.mark.parametrize("values", [[0.5], [0.0, 1.0, 2.0], None, "01", [None, 0], [True, 0],
                                   [float("nan"), 0], [float("inf"), 0], ["1", 0]])
def test_reward_vector_rejects_bad_shape_and_nonfinite_values(values):
    with pytest.raises(ValueError):
        validate_reward_values(values, 2)


def test_reward_vector_accepts_zero_negative_and_numpy_scalars():
    numpy = pytest.importorskip("numpy")
    validate_reward_values([0.0, -1.0, numpy.float32(0.5)], 3)


@pytest.fixture
def stub_url_restore(monkeypatch):
    # This suite isolates GRPO orchestration, not the source URL preprocessor.
    module = ModuleType("ir_training.data.url_preprocess")
    def restore(value, mapping):
        for key, replacement in mapping.items():
            value = value.replace(key, replacement)
        return value
    module.restore_url_placeholders = restore
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize("limit", [0, 1])
def test_audit_budget_never_disables_validation(tmp_path, stub_url_restore, limit):
    calls = 0
    def reward(**kwargs):
        nonlocal calls
        calls += 1
        return [0.0, -1.0] if calls == 1 else [float("nan"), 1.0]
    wrapped = audited_reward(reward, str(tmp_path), 0, audit_limit=limit)
    assert wrapped(completions=["a", "b"]) == [0.0, -1.0]
    with pytest.raises(ValueError, match="finite real"):
        wrapped(completions=["a", "b"])


def test_audit_retains_serving_boundary_url_order_and_values(tmp_path, stub_url_restore):
    raw = '<a2ui>root=Text("[URL_1]")</a2ui>extra'
    mapping = {"[URL_1]": "https://example.test/path"}
    seen = []
    def reward(**kwargs):
        seen.extend(kwargs["completions"])
        kwargs["log_extra"]("evidence", ["kept"])
        return [-0.25]
    wrapped = audited_reward(reward, str(tmp_path), 0)
    assert wrapped(completions=[raw], url_map=[mapping]) == [-0.25]
    assert seen == ['<a2ui>root=Text("https://example.test/path")</a2ui>']
    record = json.loads((tmp_path / "grpo_rewards.rank0.jsonl").read_text())
    assert record["raw_completion"] == raw and record["breakdown"] == {"evidence": "kept"}


@pytest.mark.parametrize("field,value", [("window_steps", True), ("window_steps", 1.5), ("window_steps", 0),
    ("zero_tolerance", float("nan")), ("zero_tolerance", float("inf")), ("zero_tolerance", True),
    ("min_diverse_group_fraction", True), ("min_nonzero_update_fraction", -0.1),
    ("max_clipped_fraction", float("nan")), ("min_nonzero_gradient_fraction", "0.5")])
def test_health_thresholds_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        HealthThresholds(**{field: value})


def test_minimum_health_thresholds_are_inclusive():
    monitor = GRPOHealthMonitor(HealthThresholds(window_steps=4))
    for step in range(1, 5):
        result = monitor.observe(step, healthy(frac_reward_zero_std=0.75,
            grad_norm=1.0 if step == 1 else 0.0, sampled_parameter_max_delta=0.01 if step == 1 else 0.0))
    assert result["status"] == "passed_window"
    assert result["aggregates"]["nonzero_update_fraction"] == 0.25


class FakeGenerationConfig(SimpleNamespace):
    def __init__(self, **kwargs):
        defaults = dict(num_beams=1, forced_eos_token_id=None, suppress_tokens=None, min_new_tokens=None)
        super().__init__(**{**defaults, **kwargs})

    def to_dict(self):
        return vars(self).copy()


@pytest.fixture
def stub_generation_config(monkeypatch):
    module = ModuleType("transformers")
    module.GenerationConfig = FakeGenerationConfig
    monkeypatch.setitem(sys.modules, "transformers", module)


def policy_trainer(**overrides):
    return SimpleNamespace(processing_class=SimpleNamespace(bos_token_id=2, pad_token_id=0),
                           max_completion_length=32, temperature=0.8,
                           generation_kwargs={"do_sample": True, "temperature": 0.8, **overrides})


@pytest.mark.parametrize("overrides", [{"top_k": 50}, {"top_p": 0.9}, {"repetition_penalty": 1.1},
    {"min_p": 0.1}, {"forced_eos_token_id": 1}, {"num_beams": 2}, {"do_sample": False},
    {"temperature": float("nan")}, {"temperature": 0}, {"temperature": 1.0}])
def test_unreviewed_sampling_and_temperature_mismatch_fail(stub_generation_config, overrides):
    with pytest.raises(ValueError):
        rollout_generation_config(policy_trainer(**overrides), [1, 3])


def test_clean_policy_retains_only_reviewed_controls(stub_generation_config):
    config = rollout_generation_config(policy_trainer(cache_implementation="dynamic"), [1, 3])
    assert config.top_k == 0 and config.top_p == 1.0 and config.num_beams == 1
    assert config.forced_eos_token_id is None and config.suppress_tokens is None
    assert config.use_cache and config.eos_token_id == [1, 3]
    assert config.temperature == 0.8 and config.cache_implementation == "dynamic"


@pytest.mark.parametrize("fails", [False, True])
def test_isolation_restores_peft_and_nested_cache_configs(stub_generation_config, fails):
    saved = FakeGenerationConfig(num_beams=4, suppress_tokens=[7], use_cache=False)
    base = SimpleNamespace(generation_config=saved)
    wrapper = SimpleNamespace(generation_config=saved, get_base_model=lambda: base,
        config=SimpleNamespace(use_cache=False, text_config=SimpleNamespace(use_cache=False)))
    clean = rollout_generation_config(policy_trainer(), [1, 3])
    try:
        with isolated_generation_config(wrapper, clean), generation_cache_scope(wrapper):
            assert wrapper.generation_config is base.generation_config is clean
            assert wrapper.config.use_cache and wrapper.config.text_config.use_cache
            assert clean.suppress_tokens is None and clean.num_beams == 1
            if fails:
                raise RuntimeError("generation failed")
    except RuntimeError:
        assert fails
    assert wrapper.generation_config is base.generation_config is saved
    assert saved.suppress_tokens == [7] and saved.use_cache is False
    assert wrapper.config.use_cache is wrapper.config.text_config.use_cache is False


@pytest.mark.parametrize("fails", [False, True])
def test_rollout_uses_clean_config_and_restores_checkpointing(torch, monkeypatch, tmp_path, stub_generation_config, fails):
    import ir_training.generation_policy as policy
    # Stop-scanner behavior is tested below; here only the runtime boundary is mocked.
    monkeypatch.setattr(policy, "build_stopping_criteria", lambda *a: "stop-policy")
    class Tokenizer:
        bos_token_id, pad_token_id = 2, 0
        def __call__(self, texts, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": torch.tensor([[0, 8], [9, 8]]),
                    "attention_mask": torch.tensor([[0, 1], [1, 1]])}
        def decode(self, ids, **kwargs):
            return "".join({0: "", 1: "", 3: "", 4: "<a2ui>", 5: "</a2ui>", 6: "unfinished"}[int(i)] for i in ids)
    calls, checkpoints = [], []
    saved = FakeGenerationConfig(num_beams=4, suppress_tokens=[6], use_cache=False)
    model = SimpleNamespace(training=True, generation_config=saved,
        config=SimpleNamespace(use_cache=False), is_gradient_checkpointing=True,
        gradient_checkpointing_enable=lambda **kw: checkpoints.append(kw))
    def generate(**kwargs):
        calls.append(kwargs)
        assert not torch.is_grad_enabled()
        assert kwargs["generation_config"] is model.generation_config
        assert model.generation_config.num_beams == 1 and model.generation_config.suppress_tokens is None
        assert model.config.use_cache and kwargs["synced_gpus"] is False
        if fails:
            raise RuntimeError("mock generation failure")
        return torch.cat((kwargs["input_ids"], torch.tensor([[4, 5], [6, 6]])), dim=1)
    model.generate = generate
    @contextmanager
    def unwrap(*args, **kwargs):
        try:
            yield model
        finally:
            model.gradient_checkpointing_enable()
    module = ModuleType("trl.models")
    module.unwrap_model_for_generation = unwrap
    monkeypatch.setitem(sys.modules, "trl.models", module)
    trainer = policy_trainer()
    trainer.processing_class = Tokenizer()
    trainer.model = trainer.model_wrapped = model
    trainer.use_vllm = trainer.is_fsdp_enabled = trainer.is_deepspeed_enabled = False
    trainer.accelerator = SimpleNamespace(device="cpu", process_index=0)
    trainer.args = SimpleNamespace(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.state = SimpleNamespace(global_step=0)
    rollout = make_express_rollout(str(tmp_path), [1, 3], audit_limit=0)
    if fails:
        with pytest.raises(RuntimeError, match="mock generation failure"):
            rollout(["a", "bb"], trainer)
    else:
        result = rollout(["a", "bb"], trainer)
        assert result["prompt_ids"] == [[8], [9, 8]]
        assert result["completion_ids"] == [[4, 5, 0], [6, 6]]
        assert result["env_mask"] == [[1, 1, 0], [1, 1]]
    assert len(calls) == 1
    assert checkpoints[-1] == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    assert model.generation_config is saved and model.config.use_cache is False
    assert model.training is True


@pytest.mark.parametrize("tail,reason,mask", [([4, 5, 0], "closing_sentinel", [1, 1, 0]),
    ([4, 3, 0], "eos_token", [1, 1, 0]), ([4, 6], "max_new_tokens", [1, 1])])
def test_original_termination_contract_is_preserved(tail, reason, mask):
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: "".join(
        {0: "", 3: "", 4: '<a2ui>root=Text("literal </a2ui>")', 5: "</a2ui>", 6: "unfinished"}[i] for i in ids))
    result = normalize_rollout_completion(tokenizer, tail, eos_token_ids=[1, 3], pad_token_id=0, max_new_tokens=2)
    assert result["stop_reason"] == reason and result["env_mask"] == mask


def test_standalone_explicit_sampling_and_cli_seed_are_wired():
    tree = ast.parse((ROOT / "training/scripts/train_grpo.py").read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "make_grpo_config")
    kwargs = {item.arg: item.value for item in call.keywords}
    for key, value in {"top_k": 0, "top_p": 1.0, "temperature": 1.0,
                       "repetition_penalty": 1.0, "disable_dropout": True}.items():
        assert ast.literal_eval(kwargs[key]) == value
    assert ast.unparse(kwargs["seed"]) == "args.seed"
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "build_audited_grpo_trainer" for n in ast.walk(tree))


def test_real_transformers_generation_config_isolation():
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=16, n_positions=16, n_embd=8, n_layer=1, n_head=1,
        bos_token_id=2, eos_token_id=1, pad_token_id=0))
    model.eval()
    saved = model.generation_config
    saved.num_beams, saved.suppress_tokens, saved.forced_eos_token_id = 4, [7], 1
    trainer = policy_trainer()
    trainer.max_completion_length = 2
    clean = rollout_generation_config(trainer, [1, 3])
    with isolated_generation_config(model, clean), generation_cache_scope(model):
        effective, _ = model._prepare_generation_config(clean)
        assert effective.suppress_tokens is None and effective.forced_eos_token_id is None
        assert effective.num_beams == 1 and effective.top_k == 0
        with torch.no_grad():
            output = model.generate(torch.tensor([[2, 8]]), generation_config=clean)
        assert output.shape[0] == 1 and 2 < output.shape[1] <= 4
    assert model.generation_config is saved and saved.num_beams == 4


def test_generation_config_scope_does_not_shadow_peft_delegation(stub_generation_config):
    saved = FakeGenerationConfig(num_beams=4)
    base = SimpleNamespace(generation_config=saved)
    class DelegatingModel:
        def get_base_model(self):
            return base
        def __getattr__(self, name):
            return getattr(base, name)
    wrapper = DelegatingModel()
    clean = rollout_generation_config(policy_trainer(), [1, 3])
    with isolated_generation_config(wrapper, clean):
        assert wrapper.generation_config is base.generation_config is clean
    assert "generation_config" not in vars(wrapper)
    assert wrapper.generation_config is base.generation_config is saved


def test_missing_trl_telemetry_fails_instead_of_reusing_stale_values():
    class BrokenTRL:
        model = SimpleNamespace(training=True)
        _metrics = {"train": defaultdict(list)}
        def _generate_and_score_completions(self, *args, **kwargs):
            return {}
    with pytest.raises(RuntimeError, match="expected GRPO group-health evidence"):
        build_audited_grpo_trainer(BrokenTRL)()._generate_and_score_completions([])
