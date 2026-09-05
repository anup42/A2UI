from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))

from ir_training.data.chat_templates import build_messages
from ir_training.train.grpo_runtime import (
    GRPOHealthMonitor, HealthThresholds, audited_reward, make_express_rollout,
    make_health_callback, normalize_rollout_completion, prepared_prompt_messages,
    prompt_fingerprint, render_chat_prompt, validate_runtime_features, write_json_record,
)


class Tokenizer:
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0
    chat_template = "test-conversation-v1"

    def __call__(self, text, *, add_special_tokens=True, **kwargs):
        assert not add_special_tokens
        return {"input_ids": [ord(char) + 10 for char in text]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize and add_generation_prompt
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value - 10) for value in ids if value not in {0, 1, 2, 3})


def _ids(text):
    return [ord(char) + 10 for char in text]


def _script():
    spec = importlib.util.spec_from_file_location("reviewed_train_grpo", ROOT / "training/scripts/train_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _healthy():
    return {"frac_reward_zero_std": 0.4, "completions/clipped_ratio": 0.0,
            "grad_norm": 0.02, "sampled_parameter_max_delta": 0.0001,
            "nonfinite_parameters_or_gradients": 0.0, "loss": 0.1}


def test_prepared_messages_preserve_complete_scaffold_and_no_asset_policy_addition():
    completion = '<a2ui>root=Text("Photo [IMAGE_1]")</a2ui>'
    row = {"messages": build_messages("a compact system", "Photo [IMAGE_1]", completion),
           "response_text": "Photo [IMAGE_1]", "completion": completion,
           "assets": [{"path": "[IMAGE_1]"}]}
    messages = prepared_prompt_messages(row)
    actual = render_chat_prompt(Tokenizer(), messages)
    expected = Tokenizer().apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
    assert actual == expected
    assert messages == row["messages"][:-1]
    assert "Asset policy" not in actual
    module = _script()
    assert module.build_prompt("Create:\n\n{response_text}", "Photo [IMAGE_1]", row["assets"]) == "Create:\n\nPhoto [IMAGE_1]"
    assert module.build_prompt("Create:\n\n{response_text}", "source", []) == "Create:\n\nsource"


def test_prepared_messages_reject_wrong_source_and_different_assistant():
    messages = build_messages("system", "source", "target")
    with pytest.raises(ValueError, match="response_text"):
        prepared_prompt_messages({"messages": messages, "response_text": "other", "completion": "target"})
    with pytest.raises(ValueError, match="assistant differs"):
        prepared_prompt_messages({"messages": messages, "response_text": "source", "completion": "other"})
    with pytest.raises(ValueError, match="saved SFT messages"):
        prepared_prompt_messages({"response_text": "source"})


def test_prompt_fingerprint_detects_nonleading_changes_and_doubled_bos():
    a = prompt_fingerprint(Tokenizer(), "same prefix\nsource a")
    b = prompt_fingerprint(Tokenizer(), "same prefix\nsource b")
    assert a["prompt_token_ids_sha256"] != b["prompt_token_ids_sha256"]
    bad = SimpleNamespace(bos_token_id=2, chat_template="test")

    class DoubleBos(Tokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [2, 2, 33]}

    with pytest.raises(ValueError, match="duplicated"):
        prompt_fingerprint(DoubleBos(), "source")


@pytest.mark.parametrize("ending", ["</a2ui>", "</a2ui>\n"])
def test_close_tag_is_terminated_and_runtime_pad_is_never_target(ending):
    text = '<a2ui>root=Text("OK")' + ending
    record = normalize_rollout_completion(Tokenizer(), _ids(text) + [0] * 3,
        eos_token_ids=[1, 3], pad_token_id=0, max_new_tokens=100)
    assert record["stop_reason"] == "closing_sentinel"
    assert record["sampled_token_ids"] == _ids(text.rstrip())
    assert record["completion_ids"][-1] == 0
    assert record["env_mask"][-1] == 0
    assert sum(record["env_mask"]) == len(record["sampled_token_ids"])
    # The exact reviewed TRL scalar termination check recognizes this stop.
    assert record["completion_ids"][-1] in [Tokenizer.eos_token_id, Tokenizer.pad_token_id]


def test_native_alternate_eos_preserved_and_truncation_stays_masked():
    native = normalize_rollout_completion(Tokenizer(), _ids("unfinished") + [3, 0, 0],
        eos_token_ids=[1, 3], pad_token_id=0, max_new_tokens=100)
    assert native["stop_reason"] == "eos_token"
    assert native["sampled_token_ids"][-1] == 3
    assert native["completion_ids"][-2:] == [3, 0]
    assert native["env_mask"][-2:] == [1, 0]
    clipped = normalize_rollout_completion(Tokenizer(), _ids("unfinished"),
        eos_token_ids=[1, 3], pad_token_id=0, max_new_tokens=10)
    assert not clipped["terminated"]
    assert clipped["synthetic_terminal_tokens"] == 0
    assert clipped["completion_ids"][-1] not in [1, 0]
    assert len(clipped["env_mask"]) == len(clipped["completion_ids"])


def test_quoted_sentinel_does_not_truncate_literal_or_repair_graph():
    text = '<a2ui>root=Text("Show </a2ui> literally")\nroot=Text("Duplicate")</a2ui>'
    record = normalize_rollout_completion(Tokenizer(), _ids(text), eos_token_ids=[1], pad_token_id=0, max_new_tokens=1000)
    assert record["sampled_token_ids"] == _ids(text)
    assert record["terminated"]


def test_same_token_overshoot_remains_in_strict_reward_input():
    class ChunkTokenizer(Tokenizer):
        def decode(self, ids, skip_special_tokens=True):
            return "".join({9: "<a2ui>root=Text('x')", 8: "</a2ui>junk", 0: ""}[int(i)] for i in ids)
    record = normalize_rollout_completion(ChunkTokenizer(), [9, 8, 0], eos_token_ids=[1], pad_token_id=0, max_new_tokens=100)
    assert record["sampled_token_ids"] == [9, 8]
    assert ChunkTokenizer().decode(record["completion_ids"]).endswith("junk")


def test_health_gate_rejects_archived_zero_signal_pattern_after_window():
    monitor = GRPOHealthMonitor(HealthThresholds(window_steps=2))
    bad = {**_healthy(), "frac_reward_zero_std": 1.0, "grad_norm": 0,
           "sampled_parameter_max_delta": 0, "completions/clipped_ratio": 0.5}
    assert monitor.observe(1, bad)["status"] == "warming_up"
    result = monitor.observe(2, bad)
    assert set(result["issues"]) == {"insufficient_reward_variance", "excessive_completion_clipping",
                                      "insufficient_nonzero_gradients", "insufficient_sampled_parameter_updates"}


def test_health_gate_requires_all_metrics_and_finite_values():
    assert GRPOHealthMonitor(HealthThresholds()).observe(1, {})["status"] == "failed"
    bad = {**_healthy(), "loss": float("nan")}
    assert "nonfinite:loss" in GRPOHealthMonitor(HealthThresholds()).observe(1, bad)["issues"]
    monitor = GRPOHealthMonitor(HealthThresholds(window_steps=2))
    monitor.observe(1, _healthy())
    assert monitor.observe(2, _healthy())["status"] == "passed_window"
    with pytest.raises(ValueError, match="increasing"):
        monitor.observe(2, _healthy())


def test_audit_reward_preserves_values_and_passes_hooks(tmp_path):
    external = {}
    def reward_fn(**kwargs):
        kwargs["log_extra"]("genui_cap", [0.0, 0.4])
        return [-1.0, -0.2]
    wrapped = audited_reward(reward_fn, str(tmp_path), 0, audit_limit=2)
    actual = wrapped(completions=["invalid", "disconnected"], prompts=["source", "source"],
                     source_id=["s1", "s1"], log_extra=lambda key, values: external.update({key: values}))
    assert actual == [-1.0, -0.2]
    assert external == {"genui_cap": [0.0, 0.4]}
    rows = [json.loads(line) for line in (tmp_path / "grpo_rewards.rank0.jsonl").read_text().splitlines()]
    assert rows[0]["breakdown"] == {"genui_cap": 0.0}
    assert rows[0]["prompt_sha256"] == rows[1]["prompt_sha256"]


def test_health_json_is_portable_even_on_nonfinite_failure(tmp_path):
    path = tmp_path / "health.jsonl"
    write_json_record(path, {"metric": float("nan"), "nested": [float("inf")]})
    assert json.loads(path.read_text()) == {"metric": "nan", "nested": ["inf"]}


def test_wrong_runtime_version_and_missing_controls_fail_without_loading_model():
    with pytest.raises(RuntimeError, match="source-reviewed"):
        validate_runtime_features(object, object, "0.30.0")
    with pytest.raises(RuntimeError):
        validate_runtime_features(object, object, "0.29.1")


def test_grpo_rejects_adapter_only_seed(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="merged SFT seed"):
        _script().validate_sft_checkpoint(str(tmp_path))


def test_health_callback_mocked_optimizer_events_without_training(tmp_path):
    torch = pytest.importorskip("torch")
    # Tiny tensor event simulation: no Trainer, training loop, or model forward.
    model = torch.nn.Linear(2, 1, bias=False)
    accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1,
                                  process_index=0, gather=lambda value: value)
    callback = make_health_callback(accelerator, str(tmp_path), HealthThresholds(window_steps=2))
    args, state, control = SimpleNamespace(ddp_broadcast_buffers=False), SimpleNamespace(global_step=0), object()
    callback.on_train_begin(args, state, control, model=model)
    for step in (1, 2):
        model.weight.grad = torch.ones_like(model.weight) * 0.1
        callback.on_pre_optimizer_step(args, state, control, model=model)
        with torch.no_grad():
            model.weight.add_(0.001)  # Emulate an optimizer update, without optimizer/train.
        state.global_step = step
        callback.on_step_end(args, state, control)
        callback.on_log(args, state, control, logs={"reward": 0.5, **_healthy()})
    callback.on_train_end(args, state, control)
    records = [json.loads(line) for line in (tmp_path / "grpo_health.rank0.jsonl").read_text().splitlines()]
    assert records[-1]["status"] == "passed_window"
    assert records[-1]["metrics"]["sampled_parameter_max_delta"] > 0


@pytest.mark.parametrize("generation_fails", [False, True])
def test_custom_rollout_is_one_batched_call_with_no_added_bos(monkeypatch, tmp_path, generation_fails):
    torch = pytest.importorskip("torch")
    from contextlib import contextmanager
    from types import ModuleType
    checkpoint_calls = []
    model = SimpleNamespace(training=True, is_gradient_checkpointing=True,
                            gradient_checkpointing_enable=lambda **kwargs: checkpoint_calls.append(kwargs))
    calls = []

    class BatchTokenizer(Tokenizer):
        def __call__(self, texts, *, add_special_tokens, **kwargs):
            assert not add_special_tokens
            rows = [_ids(text) for text in texts]
            width = max(map(len, rows))
            return {"input_ids": torch.tensor([[0] * (width-len(row)) + row for row in rows]),
                    "attention_mask": torch.tensor([[0] * (width-len(row)) + [1] * len(row) for row in rows])}

    def generate(**kwargs):
        calls.append(kwargs)
        assert kwargs["synced_gpus"] is False
        assert kwargs["eos_token_id"] == [1, 3]
        assert kwargs["stop_strings"] is None
        if generation_fails:
            raise RuntimeError("mock generation failure")
        tails = [_ids('<a2ui>root=Text("a")</a2ui>'), _ids("clipped without envelope")]
        width = max(map(len, tails))
        # Fill the incomplete row with non-PAD tokens up to the cap.
        tails[1] += _ids("x") * (width - len(tails[1]))
        tails[0] += [0] * (width - len(tails[0]))
        return torch.cat((kwargs["input_ids"], torch.tensor(tails)), dim=1)

    model.generate = generate
    @contextmanager
    def unwrap(*args, **kwargs):
        try:
            yield model
        finally:
            # TRL's unwrap restores checkpointing with no explicit options.
            model.gradient_checkpointing_enable()
    trl_models = ModuleType("trl.models")
    trl_models.unwrap_model_for_generation = unwrap
    monkeypatch.setitem(sys.modules, "trl.models", trl_models)
    trainer = SimpleNamespace(use_vllm=False, is_fsdp_enabled=False, is_deepspeed_enabled=False,
        processing_class=BatchTokenizer(), accelerator=SimpleNamespace(device="cpu", process_index=0),
        generation_kwargs={"do_sample": True}, model_wrapped=model,
        max_completion_length=28, model=model, state=SimpleNamespace(global_step=0),
        args=SimpleNamespace(gradient_checkpointing_kwargs={"use_reentrant": False}))
    if generation_fails:
        with pytest.raises(RuntimeError, match="mock generation failure"):
            make_express_rollout(str(tmp_path), [1, 3])(["short", "longer prompt"], trainer)
        assert checkpoint_calls == [{}, {"gradient_checkpointing_kwargs": {"use_reentrant": False}}]
        return
    result = make_express_rollout(str(tmp_path), [1, 3])(["short", "longer prompt"], trainer)
    assert checkpoint_calls == [{}, {"gradient_checkpointing_kwargs": {"use_reentrant": False}}]
    assert len(calls) == 1
    assert result["prompt_ids"] == [_ids("short"), _ids("longer prompt")]
    assert result["completion_ids"][0][-1] == 0
    assert result["env_mask"][0][-1] == 0
    assert result["env_mask"][1][-1] == 1
    assert result["completion_ids"][1][-1] not in [0, 1]
    trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    with pytest.raises(ValueError, match="use_reentrant=False"):
        make_express_rollout(str(tmp_path), [1, 3])(["short"], trainer)
    assert len(calls) == 1
