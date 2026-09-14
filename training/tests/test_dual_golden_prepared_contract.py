"""No GPU/model loads: exercise the published Golden revisions and run bindings."""
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.common.config import load_yaml
from ir_training.data.express_preparation import TASK_PREFIX, prepare_splits
from ir_training.data.golden_replacement import read_rows_strict, serialize_rows
from ir_training.data.shared_prompt import create_shared_prompt_contract
from ir_training.eval.prepared_contract import (
    file_sha256,
    verify_evaluation_prepared_contract,
    verify_golden_preparation,
    verify_reserved_train_validation,
)


def script(name):
    spec = importlib.util.spec_from_file_location(f"dual_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureTokenizer:
    name_or_path = "fixture-tokenizer-not-a-model-token-estimate"
    chat_template = "fixture role labels"
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def get_vocab(self):
        return {"fixture": 0}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize
        return "".join(f"{item['role']}:\n{item['content']}\n" for item in messages) + ("assistant:\n" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(range(max(1, len(text.split()) // 10)))}


@pytest.fixture
def prepared_dual(tmp_path):
    sources = {
        "golden32": ROOT / "data/eval/golden32_archive_repeat_v1/golden32.jsonl",
        "golden35": ROOT / "data/eval/golden35_v1/golden35.jsonl",
    }
    donor = read_rows_strict(sources["golden32"])[0]
    for split in ("train", "val"):
        row = {"id": f"fixture-{split}", "source_id": f"fixture-{split}",
               "response_text": f"Independent {split} source", "messages": deepcopy(donor["messages"]),
               "metadata": {"query_id": f"fixture-{split}"},
               "completion": f'<a2ui>\nroot=Text("{split} fixture")\n</a2ui>'}
        row["messages"][-2]["content"] = TASK_PREFIX + row["response_text"]
        row["messages"][-1]["content"] = row["completion"]
        sources[split] = tmp_path / f"{split}.jsonl"
        sources[split].write_bytes(serialize_rows([row]))
    directory = tmp_path / "prepared"
    manifest = prepare_splits(sources, directory, ordering="root-first", tokenizer=FixtureTokenizer(),
        max_seq_length=4096, max_input_tokens=4096, shared_prompt=create_shared_prompt_contract())
    return directory, manifest


def bound_run(tmp_path, directory):
    config = load_yaml(ROOT / "configs/models/gemma4_e2b_a2ui_express_review_sft.yaml")
    config["run"].update(dataset_dir=str(directory), prepared_manifest_required=True)
    config["training"]["max_seq_length"] = 4096
    config["golden_eval"].update(split_path=str(directory / "golden32.jsonl"), max_input_tokens=4096)
    report = script("prepare_review_training").verify_prepared(directory, directory / "golden32.jsonl",
        max_sequence=4096, max_prompt=4096, golden35=directory / "golden35.jsonl")
    config["model"]["chat_template_kwargs"] = report["tokenizer"]["chat_template_kwargs"]
    config["final_evaluation_datasets"] = report["final_evaluation_datasets"]
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    config["model"]["model_source"] = str(model)
    report["model_files"] = {"config.json": file_sha256(model / "config.json")}
    run = tmp_path / "run"
    run.mkdir()
    import yaml
    path = run / "training_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report["training_config_sha256"] = file_sha256(path)
    (run / "preparation_report.json").write_text(json.dumps(report), encoding="utf-8")
    return path, config, report


def test_both_actual_cohorts_share_one_prompt_and_bind_to_launch_and_final_eval(prepared_dual, tmp_path):
    directory, manifest = prepared_dual
    path, config, report = bound_run(tmp_path, directory)
    assert manifest["scaffold_count"] == 1
    assert report["golden_unique_sources"] == 31
    assert config["final_evaluation_datasets"]["golden35"]["selection_role"] == "final_only_holdout"
    assert config["golden_eval"]["required_rows"] == 32
    script("launch_review_training").verify_launch_binding(path)
    for count in (32, 35):
        checked = verify_evaluation_prepared_contract(path, directory / f"golden{count}.jsonl",
            required_rows=count, max_input_tokens=4096)
        assert checked["required_rows"] == count


def test_final_evaluations_reuse_overlap_scan_but_rehash_data(prepared_dual, tmp_path, monkeypatch):
    from ir_training.eval import prepared_contract as module
    directory, _ = prepared_dual
    path, _, _ = bound_run(tmp_path, directory)
    calls = []
    original = module.verify_reserved_train_validation
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, "verify_reserved_train_validation", counted)
    for count in (32, 35):
        module.verify_evaluation_prepared_contract(path, directory / f"golden{count}.jsonl", required_rows=count, max_input_tokens=4096)
    assert len(calls) == 1
    # The receipt is not a substitute for current-byte hashes, even if row
    # semantics stay unchanged and only harmless whitespace is appended.
    with (directory / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Prepared train hash"):
        module.verify_evaluation_prepared_contract(path, directory / "golden32.jsonl", required_rows=32, max_input_tokens=4096)
    assert len(calls) == 1


def test_overlap_cache_binds_omitted_source_identities_and_implementation(tmp_path, monkeypatch):
    from ir_training.eval import prepared_contract as module
    from ir_training.pipeline import preparation_cache
    directory = tmp_path / "dataset"
    directory.mkdir()
    for name in ("train", "val"):
        (directory / f"{name}.jsonl").write_text("{}\n")
    hashes = {name: module.file_sha256(directory / f"{name}.jsonl") for name in ("train", "val")}
    reserved = {"identities": {"excluded_1"}, "responses": {"response_hash"}}
    implementation = {"code": "version1"}
    calls = []
    monkeypatch.setattr(module, "load_reserved_cohorts", lambda _: reserved)
    monkeypatch.setattr(module, "verify_reserved_train_validation", lambda *_: calls.append(1))
    monkeypatch.setattr(preparation_cache, "_implementation", lambda _: implementation)
    for _ in range(2):
        module._verify_reserved_cached(directory, [], hashes, tmp_path / "cache")
    assert len(calls) == 1
    reserved["identities"].add("newly_excluded_2")
    module._verify_reserved_cached(directory, [], hashes, tmp_path / "cache")
    implementation["code"] = "version2"
    module._verify_reserved_cached(directory, [], hashes, tmp_path / "cache")
    assert len(calls) == 3


@pytest.mark.parametrize("kind", ["split", "shared_prompt", "training_manifest", "config", "source_model"])
def test_bound_final_evaluation_rejects_drift_before_model_load(prepared_dual, tmp_path, kind):
    directory, _ = prepared_dual
    path, config, _ = bound_run(tmp_path, directory)
    changed = {"split": directory / "golden35.jsonl", "shared_prompt": directory / "shared_prompt.json",
               "training_manifest": directory / "manifest.json", "config": path,
               "source_model": Path(config["model"]["model_source"]) / "config.json"}[kind]
    changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed|match a checked"):
        verify_evaluation_prepared_contract(path, directory / "golden35.jsonl", required_rows=35, max_input_tokens=4096)


@pytest.mark.parametrize("leak", ["accepted_response", "excluded_id", "excluded_masked_response"])
def test_both_golden_accepted_and_omitted_sources_remain_reserved(prepared_dual, leak):
    directory, manifest = prepared_dual
    rows = read_rows_strict(directory / "train.jsonl")
    excluded = manifest["splits"]["golden35"]["benchmark"]["excluded_sources"][0]
    if leak == "accepted_response":
        rows[0]["response_text"] = read_rows_strict(directory / "golden35.jsonl")[0]["response_text"]
    elif leak == "excluded_id":
        rows[0]["metadata"]["query_id"] = excluded["query_id"]
    else:
        from ir_training.data.url_preprocess import preprocess_training_urls
        source = ROOT.parent / manifest["splits"]["golden35"]["benchmark"]["source_run"] / "genui.jsonl"
        source_rows = read_rows_strict(source)
        failed = next(row for row in source_rows if row.get("query_id") == excluded["query_id"])
        responses = read_rows_strict(source.parent / "responses.jsonl")
        response = next(row for row in responses if row.get("response_id") == failed["response_id"])
        rows[0]["response_text"] = preprocess_training_urls(response["response_text"], {}, enabled=True).response_text
    (directory / "train.jsonl").write_bytes(serialize_rows(rows))
    with pytest.raises(ValueError, match="Reserved Golden source overlaps train"):
        verify_reserved_train_validation(directory, [directory / "golden32.jsonl", directory / "golden35.jsonl"])


def test_golden35_rejects_a_count_only_unbound_subset(prepared_dual):
    directory, manifest = prepared_dual
    rows = read_rows_strict(directory / "golden35.jsonl")
    for row in rows:
        row["metadata"].pop("benchmark")
    (directory / "golden35.jsonl").write_bytes(serialize_rows(rows))
    manifest["splits"]["golden35"].pop("benchmark")
    manifest["splits"]["golden35"]["output_sha256"] = file_sha256(directory / "golden35.jsonl")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash-bound fixed_strict_subset"):
        verify_golden_preparation(directory, directory / "golden35.jsonl", required_rows=35,
            max_sequence=4096, max_prompt=4096, expected_kind="fixed_strict_subset")


def test_tokenizer_mismatch_fails_before_model_load(monkeypatch, tmp_path):
    from ir_training.eval import generate
    calls = []
    tokenizer = FixtureTokenizer()
    adapter = SimpleNamespace(load_tokenizer=lambda: tokenizer, load_model=lambda: calls.append("model"))
    monkeypatch.setattr(generate, "create_adapter", lambda config: adapter)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(ValueError, match="tokenizer binding mismatch before model loading"):
        generate.generate_predictions({"model": {}, "prepared_evaluation_contract": {"tokenizer": {}}},
            tmp_path / "unused.jsonl", tmp_path / "never.jsonl")
    assert calls == []


@pytest.mark.parametrize("requested,expected", [(None, "fit"), ("experiment-e2b-001", "experiment-e2b-001")])
def test_recipe_tensorboard_uses_explicit_pipeline_run_id_or_legacy_directory(prepared_dual, tmp_path, monkeypatch, requested, expected):
    directory, _ = prepared_dual
    monkeypatch.setattr("ir_training.train.gpu_profile.detect_cuda_devices", lambda: {
        "version": 1, "inherited_cuda_visible_devices": None, "visible_gpu_count": 1,
        "devices": [{"visible_index": 0, "launch_identifier": "0", "uuid": "GPU-fixture",
                     "name": "NVIDIA H100 80GB", "total_memory_bytes": 80 * 1024**3, "compute_capability": [9, 0]}],
    })
    monkeypatch.setenv("A2UI_TENSORBOARD_ROOT", "/tensorboard")
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture only, never loaded")
    args = SimpleNamespace(profile="e2b", model_dir=model, dataset_dir=directory,
        golden_file=directory / "golden32.jsonl", golden35_file=directory / "golden35.jsonl",
        output_dir=tmp_path / "fit", run_id=requested, devices="auto", microbatch=None,
        effective_batch=None, max_seq_length=4096, epochs=1, steps=20, resume=None, qv_baseline=False, qat=False)
    config, report = script("prepare_review_training").build_config(args)
    assert config["run"]["id"] == expected
    assert Path(config["training"]["logging_dir"]) == Path("/tensorboard") / expected / "training"
    assert Path(config["run"]["output_dir"]) == tmp_path / "fit" / "training"
    assert report["final_evaluation_datasets"]["golden35"]["required_rows"] == 35
