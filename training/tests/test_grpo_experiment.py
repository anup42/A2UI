"""CPU controls/tests. Callback events are simulated; no GPU training claim."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
from ir_training.train.grpo_experiment import (
    GUARDS, SELECTOR, QualityMonitor, QualityPolicy, check_preparation_retention,
    make_quality_callback, periodic_reward_audit, quality_metrics, validate_token_budgets,
    validation_subset_indices, verify_quality_gate,
)


def aggregate(score=50.0, guard=0.75):
    return {SELECTOR: score, **dict.fromkeys(GUARDS, guard),
            "golden_set_sha256": "frozen32", "golden_set_rows": 32, "unique_source_count": 31}


def test_reference_budget_is_not_a_rollout_prompt():
    validate_token_budgets(6144, 5120, 2048, 8192)
    validate_token_budgets(8192, 4096, 2048, 8192)
    with pytest.raises(ValueError, match="sentinel"):
        validate_token_budgets(6144, 6144, 2048, 8192)


@pytest.mark.parametrize("value", [True, 0, -1, 12.5])
def test_invalid_token_budgets(value):
    with pytest.raises(ValueError):
        validate_token_budgets(value, 5120, 2048, 8192)


def test_retention_uses_post_token_filter_counts():
    audit = {"filtering": {"train": {"accepted_rows": 200034}}, "prepared_counts": {
        "train": {"input_rows": 200034, "accepted_rows": 10149},
        "val": {"input_rows": 4084, "accepted_rows": 215}}}
    with pytest.raises(ValueError, match="10149/200034"):
        check_preparation_retention(audit, 0.5)
    assert check_preparation_retention(audit, .05)["train"]["retained_fraction"] == 10149/200034
    with pytest.raises(ValueError, match="missing"):
        check_preparation_retention({}, .5)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, 2])
def test_invalid_retention_minimum(value):
    with pytest.raises(ValueError):
        check_preparation_retention({}, value)


def test_validation_subset_is_deterministic_and_source_unique():
    rows = [{"source_id": f"s{i//2}", "prompt": f"prompt{i}", "intent_bucket": f"i{i%3}",
             "prompt_provenance": {"prompt_tokens": 1000+(i%5)*512}} for i in range(40)]
    indices = validation_subset_indices(rows, 8, 42)
    assert indices == validation_subset_indices(rows, 8, 42)
    assert len({rows[i]["source_id"] for i in indices}) == 8
    assert len(validation_subset_indices(rows, 100, 42)) == 20
    assert validation_subset_indices(rows, 0, 42) == list(range(40))
    assert len(rows) == 40


def test_quality_gate_rejects_score_only_regression_and_plateau():
    monitor = QualityMonitor(aggregate(), QualityPolicy(patience=2))
    assert not monitor.promotion(aggregate(50.4))["passed"]
    assert not monitor.promotion(aggregate(60, .74))["passed"]
    assert monitor.promotion(aggregate(51))["passed"]
    assert not monitor.observe(25, aggregate(51))["should_stop"]
    assert not monitor.observe(50, aggregate(49))["should_stop"]
    assert monitor.observe(75, aggregate(50))["should_stop"]
    with pytest.raises(ValueError, match="increasing"):
        monitor.observe(75, aggregate(60))
    assert not QualityMonitor(aggregate(), QualityPolicy(patience=0)).observe(25, aggregate(49))["should_stop"]


@pytest.mark.parametrize("value", [None, float("nan"), True, 101])
def test_missing_nonfinite_or_out_of_range_quality(value):
    with pytest.raises(ValueError):
        quality_metrics({**aggregate(), SELECTOR: value})


@pytest.fixture
def runtime_stub(monkeypatch):
    module = ModuleType("ir_training.train.grpo_runtime")
    def write(path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    module.write_json_record = write
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return write


def test_periodic_audit_captures_late_complete_batch_without_changing_rewards(tmp_path, runtime_stub):
    def scorer(**kwargs):
        kwargs["log_extra"]("qat_reward", [{"strict_valid": True}, {"strict_valid": False}])
        return [.5, -1.]
    wrapped = periodic_reward_audit(scorer, tmp_path, 0, 25)
    state = SimpleNamespace(global_step=150)
    kwargs = dict(completions=["ok", "bad"], prompts=["p", "p"], source_id=["s", "s"],
                  trainer_state=state, grpo_rollout_mode=["train", "train"])
    assert wrapped(**kwargs) == [.5, -1.]
    assert wrapped(**kwargs, log_extra=lambda *args: None) == [.5, -1.]
    # Distinct eval mode at the same optimizer step must also be captured.
    kwargs["grpo_rollout_mode"] = ["eval", "eval"]
    wrapped(**kwargs)
    rows = [json.loads(line) for line in (tmp_path / "grpo_periodic_rewards.rank0.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {r["mode"] for r in rows} == {"train", "eval"}
    assert rows[0]["breakdown"]["qat_reward"]["strict_valid"] is True
    assert periodic_reward_audit(scorer, tmp_path, 0, 0) is scorer


def test_quality_callback_event_order_rng_and_bound_promotion(tmp_path, monkeypatch, runtime_stub):
    torch = pytest.importorskip("torch")
    import numpy as np
    import random
    transformers = ModuleType("transformers")
    transformers.TrainerCallback = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    callbacks = ModuleType("ir_training.train.callbacks")
    callbacks._evaluation_label = lambda trigger, state: f"step_{state.global_step:09d}"
    monkeypatch.setitem(sys.modules, callbacks.__name__, callbacks)
    config_module = ModuleType("ir_training.common.config")
    config_module.load_yaml = lambda path: json.loads(Path(path).read_text())
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    baseline_dir, golden_dir, checkpoint = (tmp_path / name for name in ("baseline", "golden", "best"))
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"test-only-identity")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"grpo": {"quality_control": {"enabled": True}}}))
    def baseline_eval(args, state, control, model):
        torch.rand(1); random.random(); np.random.random()
        path = baseline_dir / "step_000000000"
        path.mkdir(parents=True)
        (path / "aggregate_metrics.json").write_text(json.dumps(aggregate()))
    golden = SimpleNamespace(last_completed_step=None,
        summary=lambda: {"checkpoint_dir": str(checkpoint), "step": 25})
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True, num_processes=1)
    callback = make_quality_callback(accelerator=accelerator,
        baseline_callback=SimpleNamespace(on_evaluate=baseline_eval), golden_callback=golden,
        baseline_dir=baseline_dir, golden_dir=golden_dir, output=tmp_path, config_path=config,
        policy=QualityPolicy())
    args, state, control = object(), SimpleNamespace(global_step=0), SimpleNamespace(should_training_stop=False)
    before = torch.random.get_rng_state().clone(), random.getstate(), np.random.get_state()
    callback.on_train_begin(args, state, control, model=object())
    assert torch.equal(torch.random.get_rng_state(), before[0]) and random.getstate() == before[1]
    assert np.array_equal(np.random.get_state()[1], before[2][1])
    state.global_step = 25
    with pytest.raises(RuntimeError, match="immediately after"):
        callback.on_evaluate(args, state, control)
    golden.last_completed_step = 25
    path = golden_dir / "step_000000025"
    path.mkdir(parents=True)
    (path / "aggregate_metrics.json").write_text(json.dumps(aggregate(52)))
    (checkpoint / "aggregate_metrics.json").write_text(json.dumps(aggregate(52)))
    callback.on_evaluate(args, state, control)
    callback.on_train_end(args, state, control)  # duplicate final event is idempotent
    assert callback.finalize()["passed"] is True
    assert len(verify_quality_gate(tmp_path, checkpoint, config)) == 3
    (checkpoint / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not bind"):
        verify_quality_gate(tmp_path, checkpoint, config)


def review_module():
    spec = importlib.util.spec_from_file_location("review_grpo_run", ROOT / "training/scripts/review_grpo_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reward_reader_uses_real_nested_breakdown():
    module = review_module()
    result = module.reward_audit([{"step": 150, "reward": -1, "breakdown": {
        "qat_reward": {"strict_valid": False, "reasons": ["invalid:native_syntax"]}}}])
    assert result["strict_valid_fraction"] == 0 and result["steps"] == [150]
    assert result["reason_counts"] == {"invalid:native_syntax": 1}


def test_failure_taxonomy_does_not_call_truncation_a_wrapper_only_bug():
    classify = review_module().failure_kind
    def row(text, error="", reason="closing_sentinel"):
        return {"metrics": {"schema_valid_strict": False, "schema_error": error},
                "raw_generated_text": text, "runtime": {"stop_reason": reason}}
    assert classify(row("<a2ui>partial", reason="max_new_tokens")) == "token_limit"
    assert classify(row(" Hungary\n<a2ui>...</a2ui>")) == "prefix_contamination"
    assert classify(row("root=... </a2ui>")) == "missing_opening_envelope"
    assert classify(row("<a2ui>...</a2ui>", "duplicate id")) == "duplicate_id"


def test_pipeline_config_keeps_reference_and_rollout_budgets_separate(monkeypatch, tmp_path):
    # Execute the actual pure configuration function without loading GPU/export dependencies.
    import ast
    import copy
    import math
    path = ROOT / "training/src/ir_training/pipeline/qat_grpo.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "training_config")
    gpu = ModuleType("ir_training.train.gpu_profile")
    gpu.apply_gpu_profile = lambda config, profile: None
    monkeypatch.setitem(sys.modules, gpu.__name__, gpu)
    source = {key: {} for key in ("run", "training", "model", "golden_eval")}
    namespace = dict(copy=copy, math=math, Path=Path, sha256=lambda path: "original",
                     load_yaml=lambda path: source, WORKFLOW="qat_lora_grpo_v1",
                     training_root=lambda: ROOT / "training", official_mobile=SimpleNamespace(SELECTOR=SELECTOR))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    values = dict(output_dir=str(tmp_path), sft_config="source.yaml", sft_checkpoint="adapter",
        num_generations=4, family="e2b", max_steps=100, learning_rate=5e-7, seed=42,
        max_seq_length=6144, max_input_tokens=5120, max_new_tokens=2048, golden_every_steps=25,
        model_dir="model")
    config = namespace["training_config"]({"options": values, "source_config_sha256": "original",
        "paths": {"prepared": str(tmp_path / "prepared"), "best_checkpoint": str(tmp_path / "best")}},
        {"dtype": "bfloat16", "effective_batch_size": 32, "world_size": 4},
        {"tokenizer": {}, "final_evaluation_datasets": {}})
    assert config["grpo"]["max_prompt_length"] == 5120
    assert config["training"]["max_seq_length"] == 6144
    assert config["grpo"]["quality_control"]["enabled"] is True
    assert config["grpo"]["validation_max_rows"] == 32
    assert source == {key: {} for key in source}  # original SFT configuration is immutable


def test_all_changed_python_files_compile():
    for path in [ROOT / "training/scripts/run_qat_grpo_pipeline.py",
                 ROOT / "training/src/ir_training/pipeline/qat_grpo.py",
                 ROOT / "training/src/ir_training/train/qat_grpo.py"]:
        compile(path.read_text(), str(path), "exec")
