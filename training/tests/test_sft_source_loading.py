"""Raw heterogeneous archives remain Python JSON; Arrow receives only strings."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.train.source_data import iter_sft_rows
from ir_training.train.sft import _load_sft_text_dataset


def chat_row():
    return {"messages": [{"role": "user", "content": "input response"},
                         {"role": "assistant", "content": "output completion"}]}


def heterogeneous_rows():
    result = []
    for value in (True, "true", [1, "two", False], {"values": [1, {"enabled": True}]}):
        row = chat_row()
        row["metadata"] = {"arbitrary": {"same_field": value}}
        row["genui_json"] = {"root": "r", "state": {"same_field": value},
                             "elements": {"r": {"props": {"same_field": value}}}}
        result.append(row)
    return result


def write_rows(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_strict_python_reader_preserves_heterogeneous_values_and_original_file(tmp_path):
    expected = heterogeneous_rows()
    path = write_rows(tmp_path / "train.jsonl", expected)
    before = path.read_bytes()
    actual = list(iter_sft_rows(path))
    assert actual == expected
    assert [type(row["metadata"]["arbitrary"]["same_field"]) for row in actual] == [bool, str, list, dict]
    assert path.read_bytes() == before


def test_reader_streams_and_reports_physical_line_with_bom_and_blanks(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\ufeff\n" + json.dumps(chat_row()) + "\n\nnot-json\n", encoding="utf-8")
    rows = iter_sft_rows(path)
    assert next(rows) == chat_row()
    with pytest.raises(ValueError) as caught:
        next(rows)
    assert str(path) in str(caught.value)
    assert ":4:" in str(caught.value)


@pytest.mark.parametrize("invalid", [
    None, True, [], "row", {}, {"messages": "not a list"}, {"messages": []},
    {"messages": [{"role": "assistant", "content": "only one message"}]},
    {"messages": ["user", {"role": "assistant", "content": "answer"}]},
    {"messages": [{"role": 1, "content": "prompt"}, {"role": "assistant", "content": "answer"}]},
    {"messages": [{"role": "user", "content": True}, {"role": "assistant", "content": "answer"}]},
    {"messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": ["answer"]}]},
    {"messages": [{"role": "user", "content": "prompt"}, {"role": "user", "content": "not assistant"}]},
    {"messages": [{"role": " ", "content": "prompt"}, {"role": "assistant", "content": "answer"}]},
    {"messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": " \t"}]},
    {"messages": [{"role": "user", "content": " \t"}, {"role": "assistant", "content": "answer"}]},
])
def test_invalid_row_or_chat_contract_reports_source_and_line(tmp_path, invalid):
    path = write_rows(tmp_path / "invalid.jsonl", [chat_row(), invalid])
    with pytest.raises(ValueError) as caught:
        list(iter_sft_rows(path))
    assert str(path) in str(caught.value)
    assert ":2:" in str(caught.value)


@pytest.mark.parametrize("completion", [True, 3, ["completion"], {"text": "completion"}])
def test_present_completion_must_be_a_string(tmp_path, completion):
    row = chat_row()
    row["completion"] = completion
    path = write_rows(tmp_path / "invalid-completion.jsonl", [row])
    with pytest.raises(ValueError) as caught:
        list(iter_sft_rows(path))
    assert str(path) in str(caught.value) and ":1:" in str(caught.value)


def test_optional_string_completion_and_extra_message_fields_are_retained(tmp_path):
    row = chat_row()
    row["completion"] = "output completion"
    row["messages"][0]["metadata"] = {"unrelated": [True, "true"]}
    path = write_rows(tmp_path / "valid.jsonl", [row])
    assert list(iter_sft_rows(path)) == [row]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_numeric_constants_are_not_silently_accepted(tmp_path, constant):
    path = tmp_path / "nonstandard.jsonl"
    source = json.dumps(chat_row())[:-1] + ', "metadata": {"invalid": ' + constant + "}}\n"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        list(iter_sft_rows(path))
    assert str(path) in str(caught.value) and ":1:" in str(caught.value)


def test_explicit_text_loader_formats_exact_chat_and_drops_only_raw_arrow_columns(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    def forbidden(*args, **kwargs):
        pytest.fail("Raw JSON must not be passed through datasets.load_dataset schema inference")
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    source = heterogeneous_rows()
    train = write_rows(tmp_path / "train.jsonl", source)
    validation = write_rows(tmp_path / "val.jsonl", source[:2])
    seen = []
    def prompt(row):
        return "".join(message["role"] + "=" + message["content"] + "\n" for message in row["messages"][:-1]) + "assistant="
    def full(row):
        seen.append(deepcopy(row))
        return "".join(message["role"] + "=" + message["content"] + "\n" for message in row["messages"])
    result = _load_sft_text_dataset({"train": str(train), "validation": str(validation)}, full, prompt)
    assert set(result) == {"train", "validation"}
    assert seen == source + source[:2]
    for split, count in (("train", 4), ("validation", 2)):
        assert len(result[split]) == count
        assert set(result[split].column_names) == {"text", "prompt_text", "completion_text"}
        assert all(feature.dtype == "string" for feature in result[split].features.values())
        assert result[split][0] == {
            "text": "user=input response\nassistant=output completion\n",
            "prompt_text": "user=input response\nassistant=",
            "completion_text": "output completion\n",
        }
    assert list(iter_sft_rows(train)) == source


def test_explicit_text_loader_retains_all_rows_across_arrow_batch_boundary(tmp_path):
    pytest.importorskip("datasets")
    source = heterogeneous_rows() * 33
    train = write_rows(tmp_path / "train.jsonl", source)
    result = _load_sft_text_dataset({"train": str(train)},
                                   lambda row: "PROMPT:" + row["messages"][-1]["content"],
                                   lambda row: "PROMPT:")
    assert len(result["train"]) == 132
    assert result["train"][0] == result["train"][127] == result["train"][131]
    assert set(result["train"].column_names) == {"text", "prompt_text", "completion_text"}


def test_explicit_text_loader_rejects_an_empty_split(tmp_path):
    pytest.importorskip("datasets")
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty SFT split"):
        _load_sft_text_dataset({"train": str(path)}, lambda row: "full", lambda row: "prompt")


@pytest.mark.parametrize("value", [True, ["text"], {"text": "value"}])
def test_explicit_text_loader_rejects_non_string_adapter_results(tmp_path, value):
    pytest.importorskip("datasets")
    path = write_rows(tmp_path / "train.jsonl", [chat_row()])
    with pytest.raises(ValueError, match="must return string") as caught:
        _load_sft_text_dataset({"train": str(path)}, lambda row: value, lambda row: "prompt")
    assert str(path) in str(caught.value)
