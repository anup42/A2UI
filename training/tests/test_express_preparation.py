from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT / "patches"))

from ir_training.data.chat_templates import build_messages, build_prompt
from ir_training.data.express_preparation import (
    PreparationError, TASK_PREFIX, prepare_row, prepare_splits, serialize_checked,
)
from express_bottom_up import to_bottom_up
from express_id_repair import content_signature, repair


def _block(body: str) -> str:
    return f"<a2ui>\n{body}\n</a2ui>"


def _row(body: str = 'root=Text("Hello")') -> dict:
    completion = _block(body)
    return {
        "id": "source-1", "response_text": "Hello",
        "messages": build_messages("System contract", "Hello", completion, target_format="a2ui_express_v1"),
        "completion": completion, "a2ui_express": completion,
        "completion_targets": {"a2ui_express_v1": completion},
        "prompt": build_prompt("System contract", "Hello", target_format="a2ui_express_v1"),
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("body", [
    'root=Text("Hello")',
    'root=Column([])',
    'root=Column([a,b])\na=Text("Use children=[b] literally")\nb=Text("Hello")',
    '$/={items:["A"]}\nroot=Column([a])\na=Column([b,c],repeat={statePath:"/items",template:"b"})\nb=Text("B")\nc=Text("C")',
    'root=Tabs([{title:"One",child:"a"},{title:"Two",content:"b"}])\na=Text("A")\nb=Text("B")',
    'root=Tabs([{title:"One",id:"a"},{title:"Two",element:"b"}])\na=Text("A")\nb=Text("B")',
    '$/={open:false}\nroot=Modal(trigger="a",content="b")\na=Button("Open",onClick=setState("/open",true))\nb=Text("Body")',
    'root=Column([a,b])\na=Column([c])\nb=Column([c])\nc=Text("shared")',
    'root=Column([Text("Inline A"),Text("Inline B")])',
    '$/={count:0,literal:"children=[a]"}\nroot=Button("Add",onClick=setState("/count",1),watch={"/count":Event("changed")})',
    'root=Text("""first line\nchildren=[a] literally\nthird line""")',
])
def test_bottom_up_preserves_complete_graph_and_all_renderer_edges(body):
    text = _block(body)
    control = serialize_checked(text, "root-first")
    converted = serialize_checked(text, "bottom-up")
    assert control.graph == converted.graph
    assert control.semantic_sha256 == converted.semantic_sha256
    lines = converted.text.splitlines()
    assert lines[-2].startswith("root=")
    from pipeline.renderer_semantics import iter_renderer_references
    positions = {line.split("=", 1)[0]: i for i, line in enumerate(lines) if "=" in line and not line.startswith("$")}
    for node, element in converted.graph["elements"].items():
        for reference in iter_renderer_references(element):
            assert positions[reference.target_id] < positions[node]


@pytest.mark.parametrize(("body", "reason"), [
    ('root=Column([a])\na=Column([z])', "express_invalid"),
    ('root=Column([a])\na=Column([root])', "express_invalid"),
    ('root=Text("root")\na=Text("detached")', "unreachable_components"),
    ('root=Column([a])\na=Text("A")\na=Text("B")', "express_invalid"),
    ('root=Table(columns=["A"],rows=[[1]],highlightColumns="A")', "wire_schema_invalid"),
])
def test_invalid_targets_are_not_repaired_or_silently_dropped(body, reason):
    text = _block(body)
    with pytest.raises(PreparationError) as failure:
        serialize_checked(text)
    assert failure.value.reason == reason
    result = to_bottom_up(text)
    assert result.text == text and not result.changed and result.status == reason


def test_duplicate_repair_disabled_and_termination_not_normalized():
    duplicate = _block('root=Column([a])\na=Text("A")\na=Text("B")')
    result = repair(duplicate)
    assert result.status == "ambiguous_duplicate_ids_repair_disabled"
    assert result.text == duplicate and not result.changed
    assert content_signature(duplicate) is None
    trailing = _block('root=Text("A")') + "\n</a2ui>"
    assert repair(trailing).text == trailing
    assert repair(trailing).status == "invalid_no_repair"


def test_preparation_rebuilds_matching_scaffold_prompt_and_all_target_aliases():
    original = _row('root=Column([a])\na=Text("Hello")')
    converted, target, scaffold = prepare_row(original, "bottom-up")
    assert converted["messages"][-1]["content"] == converted["completion"] == converted["a2ui_express"] == converted["completion_targets"]["a2ui_express_v1"]
    assert converted["messages"][2]["content"] in converted["prompt"]
    assert converted["messages"][2]["content"].splitlines()[-2].startswith("root=")
    assert converted["messages"][-2]["content"] == TASK_PREFIX + "Hello"
    assert converted["prompt"].endswith(TASK_PREFIX + "Hello\n\nAssistant:\n")
    assert scaffold["messages"] == converted["messages"][:-2]
    assert converted["canonical_graph"] == target.graph
    assert original["completion"].splitlines()[1].startswith("root=")


@pytest.mark.parametrize("field", ["completion", "response_text", "a2ui_express", "completion_targets"])
def test_stale_target_or_source_binding_is_rejected(field):
    row = _row()
    row[field] = {"a2ui_express_v1": "stale"} if field == "completion_targets" else "stale"
    with pytest.raises(PreparationError):
        prepare_row(row, "bottom-up")


def test_atomic_builder_retains_invalid_rows_and_uses_same_sources_for_ab(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [_row(), _row('root=Column([missing])')]
    _write(source, rows)
    original_bytes = source.read_bytes()
    first = prepare_splits({"train": source}, tmp_path / "first", ordering="root-first")
    last = prepare_splits({"train": source}, tmp_path / "last", ordering="bottom-up")
    assert first["splits"]["train"]["accepted_source_rows_sha256"] == last["splits"]["train"]["accepted_source_rows_sha256"]
    assert last["splits"]["train"]["accepted_rows"] == 1
    assert last["splits"]["train"]["quarantined_rows"] == 1
    rejected = json.loads((tmp_path / "last/quarantine.jsonl").read_text(encoding="utf-8"))
    assert rejected["row"] == rows[1] and rejected["source_line"] == 2
    assert source.read_bytes() == original_bytes
    assert (tmp_path / "last/prompt_scaffolds.json").is_file()


def test_malformed_json_after_valid_split_never_publishes_partial_directory(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, [_row()])
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"broken":\n', encoding="utf-8")
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="malformed JSON"):
        prepare_splits({"train": source, "val": broken}, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".prepared.preparing-*"))


def test_builder_refuses_overwrite_and_empty_accepted_split(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, [_row('root=Column([missing])')])
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="no accepted rows"):
        prepare_splits({"train": source}, destination)
    destination.mkdir()
    marker = destination / "keep"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_splits({"train": source}, destination)
    assert marker.read_text() == "keep"


class CharacterTokenizer:
    chat_template = "test template"
    name_or_path = "dummy"
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        text = "".join(message["role"] + ":" + message["content"] + "\n" for message in messages)
        return text + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(text.encode("utf-8"))}

    def get_vocab(self):
        return {"test": 0}


def test_token_length_gate_quarantines_whole_rows_and_records_tokenizer(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [_row(), _row('root=Text("' + "x" * 1500 + '")')]
    _write(source, rows)
    manifest = prepare_splits({"train": source}, tmp_path / "prepared", tokenizer=CharacterTokenizer(), max_seq_length=1000)
    assert manifest["splits"]["train"]["accepted_rows"] == 1
    assert manifest["splits"]["train"]["quarantine_reasons"] == {"sequence_too_long": 1}
    assert manifest["tokenizer"]["add_special_tokens"] is False
    assert manifest["tokenizer"]["vocabulary_sha256"]
    accepted = json.loads((tmp_path / "prepared/train.jsonl").read_text(encoding="utf-8"))
    assert accepted["completion"] == serialize_checked(rows[0]["completion"], "root-first").text
    assert accepted["metadata"]["express_preparation"]["tokenization"]["sequence_tokens"] < 1000


def test_importing_cli_does_not_prepare_data(monkeypatch):
    import ir_training.data.express_preparation as module
    monkeypatch.setattr(module, "prepare_splits", lambda *args, **kwargs: pytest.fail("preparation ran during import"))
    spec = importlib.util.spec_from_file_location("preparation_cli_import_test", ROOT / "patches/build_bottom_up_dataset.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
