from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.common.jsonl import read_jsonl
from ir_training.data.url_preprocess import restore_url_placeholders
from ir_training.eval import mtp_benchmark
from ir_training.eval.compare_to_baseline import repeated_benchmark_scores
from ir_training.eval.generate import prediction_source_context_hash
from ir_training.eval.golden_set import load_fixed_golden_rows


MODEL_CONFIGS = (
    "gemma4_ir_lora.yaml",
    "gemma4_e2b_ir_lora.yaml",
    "gemma4_e2b_ir_qat_lora.yaml",
    "gemma4_e2b_ir_qat_sft.yaml",
)
GOLDEN35_CONTRACT = {
    "max_rows": 35,
    "required_rows": 35,
    "require_exact_rows": True,
    "require_unique_rows": True,
}


@pytest.mark.parametrize("filename", MODEL_CONFIGS)
def test_active_model_configs_require_exact_unique_golden35(filename):
    config = load_yaml(ROOT / "configs" / "models" / filename)
    golden = config["golden_eval"]
    assert golden["dataset_dir"] == "outputs/datasets/golden35_stage3_eval"
    assert golden["split"] == "all"
    assert {key: golden[key] for key in GOLDEN35_CONTRACT} == GOLDEN35_CONTRACT
    assert golden["output_dir"].endswith("/golden35")
    assert golden["max_new_tokens"] == config["model"]["max_output_tokens"] == 2048


def _benchmark_config():
    return load_yaml(ROOT / "configs" / "eval" / "gemma4_e2b_qat_mtp.yaml")


def test_active_evaluation_configs_and_mtp_plan_use_golden35():
    config = _benchmark_config()
    plan = mtp_benchmark.build_mtp_benchmark_plan(config)
    assert plan["validation"]["ok"] is True
    assert Path(plan["split_path"]).as_posix().endswith("/golden35_stage3_eval/all.jsonl")
    assert plan["golden_contract"] == GOLDEN35_CONTRACT
    assert plan["max_new_tokens"] == 2048
    declarative = load_yaml(ROOT / "configs" / "eval_a2ui_express_v1.yaml")
    assert declarative["run"]["split"] == config["data"]["split_path"]


def _write_rows(tmp_path, count, duplicate=False):
    rows = [{"id": f"row-{index}", "source_id": f"source-{index}"} for index in range(count)]
    if duplicate and rows:
        rows[-1]["source_id"] = rows[0]["source_id"]
    split = tmp_path / "all.jsonl"
    split.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return split


@pytest.mark.parametrize(
    "count,duplicate,match",
    [(0, False, "exactly 35"), (34, False, "exactly 35"),
     (36, False, "exactly 35"), (35, True, "duplicate identities")],
)
def test_mtp_rejects_invalid_fixed_cohort_before_runtime_import(
    tmp_path, monkeypatch, count, duplicate, match,
):
    config = _benchmark_config()
    config["data"]["split_path"] = str(_write_rows(tmp_path, count, duplicate))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            pytest.fail("Runtime import occurred before cohort validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        mtp_benchmark, "load_hf_model",
        lambda *_args, **_kwargs: pytest.fail("Model loaded for invalid cohort"),
    )
    with pytest.raises(ValueError, match=match):
        mtp_benchmark.run_mtp_benchmark(config)


@pytest.mark.parametrize("fixed_contract", [True, False])
def test_mtp_validates_before_loading_and_allows_custom_diagnostic_sets(
    tmp_path, monkeypatch, fixed_contract,
):
    config = _benchmark_config()
    split = _write_rows(tmp_path, 35 if fixed_contract else 1)
    if fixed_contract:
        config["data"]["split_path"] = str(split)
    else:
        config["data"] = {"split_path": str(split), "max_rows": 1}
    # Avoid requiring a real local merged model in this CPU-only boundary test.
    config["source"]["target_model_id"] = config["source"]["target_base_model_id"]
    processor_calls = []
    processor = types.SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: processor_calls.append("processor"),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoProcessor=processor))

    class ModelLoadBoundary(Exception):
        pass

    def stop_at_model_load(*_args, **_kwargs):
        raise ModelLoadBoundary

    monkeypatch.setattr(mtp_benchmark, "load_hf_model", stop_at_model_load)
    with pytest.raises(ModelLoadBoundary):
        mtp_benchmark.run_mtp_benchmark(config)
    assert processor_calls == ["processor"]


def test_mtp_default_plan_is_golden35_with_2048_generation_cap():
    config = _benchmark_config()
    config.pop("data")
    config["benchmark"].pop("max_new_tokens")
    plan = mtp_benchmark.build_mtp_benchmark_plan(config)
    assert Path(plan["split_path"]).as_posix().endswith("/golden35_stage3_eval/all.jsonl")
    assert plan["golden_contract"] == GOLDEN35_CONTRACT
    assert plan["max_new_tokens"] == 2048


@pytest.mark.parametrize("data", [{}, {"split_path": ""}, {"split_path": None}])
def test_mtp_implicit_default_split_requires_fixed_golden35(data):
    assert mtp_benchmark._golden_row_contract(data) == GOLDEN35_CONTRACT
    config = _benchmark_config()
    config["data"] = data
    assert Path(mtp_benchmark.build_mtp_benchmark_plan(config)["split_path"]).as_posix().endswith("/golden35_stage3_eval/all.jsonl")


def test_mtp_predictions_preserve_real_frozen_golden35_scoring_identity():
    rows = load_fixed_golden_rows(
        ROOT / "data" / "eval" / "golden35_v1" / "golden35.jsonl",
        **GOLDEN35_CONTRACT,
    )
    predictions = [mtp_benchmark._prediction_common(row, row["completion"]) for row in rows]
    aggregate = repeated_benchmark_scores(predictions, {})
    assert aggregate["benchmark"]["id"] == "golden35_v1"
    assert aggregate["benchmark"]["row_count"] == 35
    assert aggregate["benchmark"]["unique_source_count"] == 35
    assert aggregate["benchmark"]["duplicate_occurrence_count"] == 0
    for row, prediction in zip(rows, predictions):
        assert prediction["source_id"] == row["source_id"]
        assert prediction["benchmark"] == row["metadata"]["benchmark"]
        assert prediction["assets"] == row["assets"]
        assert prediction["expected_ui_contract_v5_4"] == row["expected_ui_contract_v5_4"]
        assert prediction["source_context_sha256"] == prediction_source_context_hash(prediction)
        url_map = row["metadata"]["url_preprocessing"]["url_map"]
        assert prediction["url_map"] == url_map
        assert prediction["response_text"] == restore_url_placeholders(row["response_text"], url_map)
        assert prediction["expected"] == restore_url_placeholders(row["completion"], url_map)
        assert prediction["raw_generated_text"] == row["completion"]


def test_mtp_latency_repeats_do_not_multiply_scored_cases(tmp_path, monkeypatch):
    config = _benchmark_config()
    config["data"] = {"split_path": str(_write_rows(tmp_path, 2)), "max_rows": 2}
    config["run"]["output_dir"] = str(tmp_path / "benchmark")
    config["source"]["target_model_id"] = config["source"]["target_base_model_id"]
    config["benchmark"].update({"repeats": 3, "warmup_rows": 0})

    class Tokens:
        def __init__(self, values):
            self.values = values
            self.shape = (len(values),)

        def __getitem__(self, key):
            return Tokens(self.values[key]) if isinstance(key, slice) else self.values[key]

    processor = types.SimpleNamespace(decode=lambda *_args, **_kwargs: "<a2ui>\nroot=Text(\"Mock\")\n</a2ui>")
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(equal=lambda first, second: first.values == second.values))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: processor),
    ))
    monkeypatch.setattr(mtp_benchmark, "load_hf_model", lambda *_args, **_kwargs: types.SimpleNamespace(
        eval=lambda: None, generation_config=types.SimpleNamespace(),
    ))
    monkeypatch.setattr(mtp_benchmark, "_prepare_inputs", lambda *_args: {"input_ids": Tokens([1, 2])})
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return {"tokens": [Tokens([1, 2, 3, 4])], "elapsed_seconds": 2 * ((len(calls) - 1) % 3 + 1)}

    monkeypatch.setattr(mtp_benchmark, "_generate_once", generate)
    scored_counts = []

    def score(path, **_kwargs):
        count = len(list(read_jsonl(path)))
        scored_counts.append(count)
        return {"mock_scored_rows": count}

    monkeypatch.setattr(mtp_benchmark, "evaluate_predictions", score)
    summary = mtp_benchmark.run_mtp_benchmark(config)
    assert len(calls) == 2 * 2 * 3
    assert scored_counts == [2, 2]
    assert summary["sample_count"] == 2
    for artifact in ("target_predictions", "mtp_predictions", "timings"):
        assert len(list(read_jsonl(summary["artifacts"][artifact]))) == 2
    assert summary["target_total_seconds"] == summary["mtp_total_seconds"] == 8
