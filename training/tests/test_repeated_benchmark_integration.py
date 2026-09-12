"""Exercise the real explicit-repeat artifact through CPU preparation and launch checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.data.express_preparation import TASK_PREFIX, prepare_splits
from ir_training.data.golden_replacement import read_rows_strict, serialize_rows
from ir_training.eval.compare_to_baseline import repeated_benchmark_scores
from ir_training.eval.golden_set import benchmark_contract_for_split, load_fixed_golden_rows


class FixtureTokenizer:
    """Only tests plumbing; it is not a real model's token-count estimate."""
    name_or_path = "fixture-tokenizer"
    chat_template = "role labels with trailing assistant header"
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def get_vocab(self):
        return {"fixture": 0}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize
        return "".join(f"{m['role']}:\n{m['content']}\n" for m in messages) + ("assistant:\n" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(range(len(text.split())))}


def _launch_preparer():
    spec = importlib.util.spec_from_file_location("repeat_launch_preparer", ROOT / "scripts/prepare_review_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared_repeat(tmp_path):
    raw = ROOT / "data/eval/golden32_archive_repeat_v1/golden32.jsonl"
    donor = read_rows_strict(raw)[0]
    inputs = {"golden32": raw}
    for split in ("train", "val"):
        completion = f'<a2ui>\nroot=Text("{split} fixture")\n</a2ui>'
        messages = deepcopy(donor["messages"])
        messages[-2]["content"] = TASK_PREFIX + f"Independent {split} fixture response"
        messages[-1]["content"] = completion
        row = {"id": f"fixture-{split}", "source_id": f"fixture-{split}",
               "response_text": f"Independent {split} fixture response", "messages": messages, "completion": completion,
               "metadata": {"query_id": f"fixture-{split}"}}
        inputs[split] = tmp_path / f"{split}.jsonl"
        inputs[split].write_bytes(serialize_rows([row]))
    directory = tmp_path / "prepared"
    manifest = prepare_splits(inputs, directory, ordering="bottom-up", tokenizer=FixtureTokenizer(),
                              max_seq_length=4096, max_input_tokens=4096)
    return directory, manifest


def _rewrite_manifest_bound_split(directory, split, rows):
    import hashlib
    path = directory / f"{split}.jsonl"
    path.write_bytes(serialize_rows(rows))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"][split]["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_real_repeat_roundtrips_into_prepared_loader_and_recipe_verification(prepared_repeat):
    directory, manifest = prepared_repeat
    split = directory / "golden32.jsonl"
    rows = load_fixed_golden_rows(split, required_rows=32, require_unique_rows=True)
    contract = benchmark_contract_for_split(split)
    assert len(rows) == 32 and len({row["source_id"] for row in rows}) == 31
    assert manifest["splits"]["golden32"]["benchmark"]["output_sha256"] == contract["output_sha256"]
    report = _launch_preparer().verify_prepared(directory, split, max_sequence=4096, max_prompt=4096)
    assert report["benchmark"]["unique_source_count"] == 31
    assert report["golden_rows"] == 32


@pytest.mark.parametrize("leak", ["validation_response", "failed_original_id"])
def test_repeat_launch_rejects_validation_and_replaced_source_leakage(prepared_repeat, leak):
    directory, manifest = prepared_repeat
    split = directory / "golden32.jsonl"
    if leak == "validation_response":
        rows = read_rows_strict(directory / "val.jsonl")
        rows[0]["response_text"] = read_rows_strict(split)[1]["response_text"]
        rows[0]["messages"][-2]["content"] = TASK_PREFIX + rows[0]["response_text"]
        _rewrite_manifest_bound_split(directory, "val", rows)
        expected = "overlaps validation"
    else:
        rows = read_rows_strict(directory / "train.jsonl")
        failed = manifest["splits"]["golden32"]["benchmark"]["excluded_sources"][0]
        rows[0]["source_id"] = failed["source_id"]
        rows[0]["metadata"]["query_id"] = failed["query_id"]
        _rewrite_manifest_bound_split(directory, "train", rows)
        expected = "still reserved"
    with pytest.raises(ValueError, match=expected):
        _launch_preparer().verify_prepared(directory, split, max_sequence=4096, max_prompt=4096)


def test_real_repeat_score_averages_donor_before_unique_source_macro(prepared_repeat):
    directory, _ = prepared_repeat
    rows = read_rows_strict(directory / "golden32.jsonl")
    predictions = [{"response_text": row["response_text"], "expected": row["completion"],
                    "benchmark": row["metadata"]["benchmark"],
                    "metrics": {"generation_reward_v5_4": 0.0}} for row in rows]
    predictions[0]["metrics"]["generation_reward_v5_4"] = 1.0
    # Donor is 1, repeat is 0: their source score is 0.5, then 31-way macro.
    result = repeated_benchmark_scores(predictions, {})
    assert result["unique_source_generation_reward_v5_4_avg"] == pytest.approx(0.5 / 31)
    assert result["unique_source_metrics"]["count"] == 31


def test_gpu_recipe_selects_unique_source_metric_for_prepared_real_repeat(prepared_repeat, tmp_path, monkeypatch):
    directory, _ = prepared_repeat
    monkeypatch.setattr("ir_training.train.gpu_profile.detect_cuda_devices", lambda: {
        "version": 1, "inherited_cuda_visible_devices": None, "visible_gpu_count": 1,
        "devices": [{"visible_index": 0, "launch_identifier": "0", "uuid": "GPU-fixture",
                     "name": "NVIDIA H100 80GB", "total_memory_bytes": 80 * 1024**3, "compute_capability": [9, 0]}],
    })
    model = tmp_path / "fixture-model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture only, never loaded")
    args = SimpleNamespace(profile="e2b", model_dir=model, dataset_dir=directory,
                           golden_file=directory / "golden32.jsonl", output_dir=tmp_path / "launch",
                           devices="auto", microbatch=None, effective_batch=None, max_seq_length=4096,
                           epochs=1, steps=20, resume=None, qv_baseline=False, qat=False)
    config, report = _launch_preparer().build_config(args)
    assert config["golden_eval"]["metric_for_best_model"] == "unique_source_generation_reward_v5_4_avg"
    assert report["model_loaded"] is False and report["training_executed"] is False
    assert report["benchmark"]["row_count"] == 32
