"""Bixby50 stays a source-only, final-only holdout through preparation/launch."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from test_dual_golden_prepared_contract import ROOT, FixtureTokenizer, script
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


@pytest.fixture
def prepared_bixby50(tmp_path):
    sources = {
        "golden32": ROOT / "data/eval/golden32_archive_repeat_v1/golden32.jsonl",
        "golden35": ROOT / "data/eval/golden35_v1/golden35.jsonl",
        "bixby50": ROOT / "data/eval/bixby50_v1/bixby50.jsonl",
    }
    donor = read_rows_strict(sources["golden32"])[0]
    for name in ("train", "val"):
        row = {
            "id": f"independent-bixby-fixture-{name}",
            "source_id": f"independent-bixby-fixture-{name}",
            "response_text": f"Independent Bixby contract fixture for {name}",
            "metadata": {"query_id": f"independent-bixby-fixture-{name}"},
            "messages": deepcopy(donor["messages"]),
            "completion": f'<a2ui>\nroot=Text("{name} fixture")\n</a2ui>',
        }
        row["messages"][-2]["content"] = TASK_PREFIX + row["response_text"]
        row["messages"][-1]["content"] = row["completion"]
        sources[name] = tmp_path / f"{name}.jsonl"
        sources[name].write_bytes(serialize_rows([row]))
    directory = tmp_path / "prepared"
    manifest = prepare_splits(
        sources, directory, ordering="root-first", tokenizer=FixtureTokenizer(),
        max_seq_length=4096, max_input_tokens=4096,
        shared_prompt=create_shared_prompt_contract(),
        evaluation_splits={"golden32", "golden35", "bixby50"},
    )
    return directory, manifest


def _bind_run(tmp_path, directory):
    config = load_yaml(ROOT / "configs/models/gemma4_e2b_a2ui_express_review_sft.yaml")
    config["run"].update(dataset_dir=str(directory), prepared_manifest_required=True)
    config["training"]["max_seq_length"] = 4096
    config["golden_eval"].update(split_path=str(directory / "golden32.jsonl"), max_input_tokens=4096)
    report = script("prepare_review_training").verify_prepared(
        directory, directory / "golden32.jsonl", max_sequence=4096, max_prompt=4096,
        golden35=directory / "golden35.jsonl", bixby50=directory / "bixby50.jsonl",
    )
    config["model"]["chat_template_kwargs"] = report["tokenizer"]["chat_template_kwargs"]
    config["final_evaluation_datasets"] = report["final_evaluation_datasets"]
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    config["model"]["model_source"] = str(model)
    report["model_files"] = {"config.json": file_sha256(model / "config.json")}
    run = tmp_path / "run"
    run.mkdir()
    path = run / "training_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report["training_config_sha256"] = file_sha256(path)
    (run / "preparation_report.json").write_text(json.dumps(report), encoding="utf-8")
    return path, config, report


def test_bixby50_is_bound_to_launch_and_final_evaluation_without_reference_targets(prepared_bixby50, tmp_path):
    directory, _ = prepared_bixby50
    path, config, report = _bind_run(tmp_path, directory)
    binding = report["final_evaluation_datasets"]["bixby50"]
    assert binding["required_rows"] == 50
    assert binding["selection_role"] == "final_only_holdout"
    assert binding["reference_available"] is False
    assert binding["evaluation_only"] is True
    assert binding["target_validation"] == "not_applicable_source_only"
    assert config["golden_eval"]["required_rows"] == 32
    assert Path(config["golden_eval"]["split_path"]).name == "golden32.jsonl"
    script("launch_review_training").verify_launch_binding(path)
    for name, count in (("golden32", 32), ("golden35", 35), ("bixby50", 50)):
        actual = verify_evaluation_prepared_contract(
            path, directory / f"{name}.jsonl", required_rows=count, max_input_tokens=4096,
        )
        assert actual["required_rows"] == count


def test_optional_bixby50_does_not_require_golden35_or_change_checkpoint_selection(prepared_bixby50):
    directory, _ = prepared_bixby50
    report = script("prepare_review_training").verify_prepared(
        directory, directory / "golden32.jsonl", max_sequence=4096, max_prompt=4096,
        bixby50=directory / "bixby50.jsonl",
    )
    assert set(report["final_evaluation_datasets"]) == {"golden32", "bixby50"}
    assert report["final_evaluation_datasets"]["golden32"]["selection_role"] == "development_checkpoint_selection"


@pytest.mark.parametrize("key,bad_value", [
    ("reference_available", True), ("evaluation_only", False), ("target_validation", "strict_valid"),
])
def test_bixby50_prepared_flags_cannot_claim_reference_validation(prepared_bixby50, key, bad_value):
    directory, manifest = prepared_bixby50
    manifest["splits"]["bixby50"][key] = bad_value
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Source-only prepared split"):
        verify_golden_preparation(
            directory, directory / "bixby50.jsonl", required_rows=50,
            max_sequence=4096, max_prompt=4096, expected_kind="source_only_holdout",
        )


def test_bixby50_cannot_be_relabelled_as_an_ordinary_split(prepared_bixby50):
    directory, manifest = prepared_bixby50
    manifest["evaluation_splits"].remove("bixby50")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit named evaluation split"):
        verify_golden_preparation(
            directory, directory / "bixby50.jsonl", required_rows=50,
            max_sequence=4096, max_prompt=4096, expected_kind="source_only_holdout",
        )


def test_bixby50_refuses_partial_cohort_even_with_updated_preparation_hash(prepared_bixby50):
    directory, manifest = prepared_bixby50
    split = directory / "bixby50.jsonl"
    split.write_bytes(serialize_rows(read_rows_strict(split)[:-1]))
    manifest["splits"]["bixby50"]["output_sha256"] = file_sha256(split)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 50 rows"):
        script("prepare_review_training").verify_prepared(
            directory, directory / "golden32.jsonl", max_sequence=4096, max_prompt=4096,
            bixby50=split,
        )


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("leak", ["source_response", "query_id"])
def test_bixby50_sources_are_reserved_from_training_and_validation(prepared_bixby50, split, leak):
    directory, _ = prepared_bixby50
    target = directory / f"{split}.jsonl"
    records = read_rows_strict(target)
    bixby = read_rows_strict(directory / "bixby50.jsonl")[0]
    if leak == "source_response":
        records[0]["response_text"] = bixby["response_text"]
    else:
        records[0]["metadata"]["query_id"] = bixby.get("query_id") or bixby["metadata"].get("query_id") or bixby["source_id"]
    target.write_bytes(serialize_rows(records))
    with pytest.raises(ValueError, match=f"Reserved Golden source overlaps {split}"):
        verify_reserved_train_validation(directory, [directory / "bixby50.jsonl"])


def test_bixby50_changed_after_binding_fails_before_model_load(prepared_bixby50, tmp_path):
    directory, _ = prepared_bixby50
    path, _, _ = _bind_run(tmp_path, directory)
    target = directory / "bixby50.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="match a checked preparation manifest"):
        verify_evaluation_prepared_contract(path, target, required_rows=50, max_input_tokens=4096)


def test_build_config_binds_bixby50_argument(prepared_bixby50, tmp_path, monkeypatch):
    directory, _ = prepared_bixby50
    monkeypatch.setattr("ir_training.train.gpu_profile.detect_cuda_devices", lambda: {
        "version": 1, "inherited_cuda_visible_devices": None, "visible_gpu_count": 1,
        "devices": [{"visible_index": 0, "launch_identifier": "0", "uuid": "GPU-fixture",
                     "name": "NVIDIA H100 80GB", "total_memory_bytes": 80 * 1024**3, "compute_capability": [9, 0]}],
    })
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"Fixture weights, never loaded")
    args = SimpleNamespace(
        profile="e2b", model_dir=model, dataset_dir=directory,
        golden_file=directory / "golden32.jsonl", golden35_file=directory / "golden35.jsonl",
        bixby50_file=directory / "bixby50.jsonl", output_dir=tmp_path / "fit", devices="auto",
        max_seq_length=4096, epochs=1, steps=20, resume=None, qv_baseline=False, qat=False,
    )
    config, report = script("prepare_review_training").build_config(args)
    assert config["final_evaluation_datasets"]["bixby50"] == report["final_evaluation_datasets"]["bixby50"]
    assert config["final_evaluation_datasets"]["bixby50"]["required_rows"] == 50
