from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training/src"))

from ir_training.common.config import repo_root
from ir_training.common.jsonl import read_jsonl
from ir_training.data.chat_templates import build_messages, build_prompt
from ir_training.data.express_preparation import PreparationError, TASK_PREFIX, prepare_splits, serialize_checked
from ir_training.data.shared_prompt import (
    build_inference_messages, create_shared_prompt_contract, load_shared_prompt_contract,
    normalize_row, validate_shared_prompt_contract,
)
from ir_training.eval.golden_set import load_fixed_golden_rows


def row(source="Hello", completion='<a2ui>\nroot=Text("Hello")\n</a2ui>'):
    return {
        "id": "source-1", "source_id": "source-1", "response_text": source,
        "messages": build_messages("Old short prompt", source, completion),
        "prompt": build_prompt("Old short prompt", source), "completion": completion,
        "a2ui_express": completion, "completion_targets": {"a2ui_express_v1": completion},
        "metadata": {"keep": "provenance"},
    }


def write_rows(path, rows):
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


class CharacterTokenizer:
    name_or_path = "dummy"
    chat_template = "test"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        value = "".join(message["role"] + ":" + message["content"] + "\n" for message in messages)
        return value + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": list(text.encode("utf-8"))}


def test_contract_uses_full_production_builder_scaffold_and_rejects_drift(tmp_path):
    contract = create_shared_prompt_contract()
    production = build_messages("", "Hello")[:-1]
    production[2]["content"] = serialize_checked(production[2]["content"], "root-first").text
    assert contract["scaffold"]["messages"] == production
    assert len(production[0]["content"]) > 4000
    saved = tmp_path / "shared_prompt.json"
    saved.write_text(json.dumps(contract), encoding="utf-8")
    assert load_shared_prompt_contract(saved) == contract
    for field in ("version", "contract_sha256", "source_text_sha256"):
        stale = {**contract, field: "changed"}
        with pytest.raises(ValueError, match="stale or changed"):
            validate_shared_prompt_contract(stale)


def test_normalization_preserves_source_target_and_benchmark_metadata():
    original = row()
    original["metadata"]["benchmark"] = {"kind": "test", "occurrence_id": "source-1"}
    untouched = deepcopy(original)
    contract = create_shared_prompt_contract()
    prepared = normalize_row(original, contract)
    assert original == untouched
    for key in ("id", "source_id", "response_text", "completion", "a2ui_express", "completion_targets"):
        assert prepared[key] == original[key]
    assert prepared["metadata"]["benchmark"] == original["metadata"]["benchmark"]
    assert prepared["messages"][-2:] == original["messages"][-2:]
    assert build_inference_messages(original["response_text"], contract) == prepared["messages"][:-1]
    assert prepared["metadata"]["shared_prompt"]["source_prompt_sha256"] == hashlib.sha256(original["prompt"].encode()).hexdigest()
    original["messages"][-2]["content"] += "changed"
    with pytest.raises(PreparationError, match="differs"):
        normalize_row(original, contract)


def test_shared_preparation_keeps_both_frozen_cohorts_and_one_scaffold(tmp_path):
    root = repo_root() / "training/data/eval"
    golden32 = root / "golden32_archive_repeat_v1/golden32.jsonl"
    golden35 = root / "golden35_v1/golden35.jsonl"
    original_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (golden32, golden35)]
    training = tmp_path / "train.jsonl"
    write_rows(training, [row()])
    output = tmp_path / "prepared"
    contract = create_shared_prompt_contract()
    manifest = prepare_splits(
        {"train": training, "val": training, "golden32": golden32, "golden35": golden35},
        output, shared_prompt=contract, evaluation_splits={"golden32", "golden35"},
    )
    assert manifest["scaffold_count"] == 1
    assert manifest["shared_prompt"] == load_shared_prompt_contract(output / "shared_prompt.json")
    assert manifest["evaluation_splits"] == ["golden32", "golden35"]
    for name, size, source in (("golden32", 32, golden32), ("golden35", 35, golden35)):
        prepared = load_fixed_golden_rows(output / f"{name}.jsonl", required_rows=size)
        original = list(read_jsonl(source))
        assert len(prepared) == size
        for before, after in zip(original, prepared):
            assert before["response_text"] == after["response_text"]
            assert before["id"] == after["id"]
            assert serialize_checked(before["completion"]).graph == serialize_checked(after["completion"]).graph
            assert after["messages"][:-2] == contract["scaffold"]["messages"]
            assert build_inference_messages(after["response_text"], contract) == after["messages"][:-1]
        assert manifest["splits"][name]["benchmark"]["row_count"] == size
    assert original_hashes == [hashlib.sha256(path.read_bytes()).hexdigest() for path in (golden32, golden35)]
    assert (output / "inference_prompt.json").is_file()
    sources = json.loads((output / "source_prompt_scaffolds.json").read_text(encoding="utf-8"))
    assert len(sources) >= 2


def test_whitespace_rich_external_source_keeps_bytes_and_normalizes_task_for_inference(tmp_path):
    original = row(source=" \n  Hello\n  second line\t\n ")
    original["messages"][-2]["content"] = TASK_PREFIX + original["response_text"]
    before = deepcopy(original)
    contract = create_shared_prompt_contract()
    normalized = normalize_row(original, contract)
    assert original == before
    assert normalized["response_text"] == before["response_text"]
    assert normalized["messages"][:-1] == build_inference_messages(before["response_text"], contract)
    assert normalized["metadata"]["shared_prompt"]["source_task_sha256"] == hashlib.sha256(before["messages"][-2]["content"].encode()).hexdigest()
    source = tmp_path / "train.jsonl"
    write_rows(source, [before])
    prepare_splits({"train": source}, tmp_path / "prepared", shared_prompt=contract)
    prepared = list(read_jsonl(tmp_path / "prepared/train.jsonl"))[0]
    assert prepared["response_text"] == before["response_text"]
    assert prepared["messages"][:-1] == build_inference_messages(prepared["response_text"], contract)
    assert prepared["completion"] == before["completion"]


def test_eval_reference_is_not_subject_to_training_sequence_length_limit(tmp_path):
    training, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    short = row()
    long = row(completion='<a2ui>\nroot=Text("' + "x" * 2000 + '")\n</a2ui>')
    write_rows(training, [short, long])
    write_rows(evaluation, [long])
    manifest = prepare_splits(
        {"train": training, "golden": evaluation}, tmp_path / "prepared",
        tokenizer=CharacterTokenizer(), max_seq_length=1000, max_input_tokens=800,
        evaluation_splits={"golden"},
    )
    assert manifest["splits"]["train"]["quarantine_reasons"] == {"sequence_too_long": 1}
    assert manifest["splits"]["golden"]["accepted_rows"] == 1
    assert manifest["splits"]["golden"]["max_accepted_token_lengths"]["sequence_tokens"] > 1000
    assert list(read_jsonl(tmp_path / "prepared/golden.jsonl"))[0]["completion"] == long["completion"]


def test_eval_prompt_too_long_aborts_instead_of_publishing_partial_cohort(tmp_path):
    source = tmp_path / "eval.jsonl"
    write_rows(source, [row(), row(source="x" * 2000)])
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="preserve every reference"):
        prepare_splits({"golden": source}, destination, tokenizer=CharacterTokenizer(),
                       max_input_tokens=800, evaluation_splits={"golden"})
    assert not destination.exists()


def test_token_limits_are_measured_after_full_prompt_normalization(tmp_path):
    source = tmp_path / "eval.jsonl"
    write_rows(source, [row()])
    prepare_splits({"golden": source}, tmp_path / "old_prompt", tokenizer=CharacterTokenizer(),
                   max_input_tokens=1000, evaluation_splits={"golden"})
    with pytest.raises(ValueError, match="no accepted rows"):
        prepare_splits({"golden": source}, tmp_path / "shared_prompt", tokenizer=CharacterTokenizer(),
                       max_input_tokens=1000, evaluation_splits={"golden"},
                       shared_prompt=create_shared_prompt_contract())
    assert not (tmp_path / "shared_prompt").exists()


def test_explicit_prompt_order_and_eval_names_must_match_inputs(tmp_path):
    source = tmp_path / "source.jsonl"
    write_rows(source, [row()])
    with pytest.raises(ValueError, match="serialization_order"):
        prepare_splits({"train": source}, tmp_path / "prepared", ordering="bottom-up",
                       shared_prompt=create_shared_prompt_contract())
    with pytest.raises(ValueError, match="evaluation_splits"):
        prepare_splits({"train": source}, tmp_path / "prepared", evaluation_splits={"missing"})
