from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ir_training.data.chat_templates import build_messages
from ir_training.train.grpo_runtime import audited_reward


class Tokenizer:
    bos_token_id = None
    chat_template = "test"

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(item["content"] for item in messages)

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": [ord(char) for char in text]}


def script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_grpo.py"
    spec = importlib.util.spec_from_file_location("grpo_context_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_weights(path):
    weights = b"controlled full weight file fixture"
    (path / "model.safetensors").write_bytes(weights)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return {"path": "model.safetensors", "size": len(weights), "sha256": hashlib.sha256(weights).hexdigest()}


@pytest.mark.parametrize("kind", ["merged_lora", "full_finetune"])
def test_checked_seed_accepts_repo_artifacts_without_claiming_numeric_parity(tmp_path, kind):
    weight = materialize_weights(tmp_path)
    if kind == "merged_lora":
        metadata = {"merged_model_files": [weight], "adapter_files": [{"path": "adapter_model.safetensors"}],
                    "training_run_metadata": {"verified": False}}
        filename = "qat_mtp_merge_metadata.json"
    else:
        metadata = {"training": {"method": "full_finetune_sft"}, "checkpoint_kind": "full_model",
                    "adapter_checkpoints": [{"checkpoint_kind": "full_model", "files": [weight]}]}
        filename = "training_metadata.json"
    (tmp_path / filename).write_text(json.dumps(metadata), encoding="utf-8")
    result = script().validate_sft_checkpoint(str(tmp_path))
    assert result["kind"] == kind
    assert result["local_identity_verified"] is True
    assert result["numerical_parity_verified"] is False
    (tmp_path / "model.safetensors").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="provenance does not match"):
        script().validate_sft_checkpoint(str(tmp_path))


def test_checkpoint_marker_alone_does_not_bind_weights(tmp_path):
    materialize_weights(tmp_path)
    (tmp_path / "trainer_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="marker alone"):
        script().validate_sft_checkpoint(str(tmp_path))


def test_full_seed_verifies_config_and_uses_portable_artifact_filenames(tmp_path):
    weight = materialize_weights(tmp_path)
    config = {"path": "config.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
    metadata = {"training": {"method": "full_finetune_sft"}, "adapter_checkpoints": [
        {"checkpoint_kind": "full_model", "path": "/original/gpu-pc/run/final_model", "files": [weight, config]}]}
    (tmp_path / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert script().validate_sft_checkpoint(str(tmp_path))["kind"] == "full_finetune"
    (tmp_path / "config.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="provenance does not match"):
        script().validate_sft_checkpoint(str(tmp_path))


def test_grpo_rewards_use_golden_source_contract_and_url_context(tmp_path):
    pytest.importorskip("datasets")
    placeholder = "[IMAGE_URL_1]"
    uri = "https://example.test/a.png"
    completion = f'<a2ui>\nroot=Text("Photo {placeholder}")\n</a2ui>'
    row = {"id": "one", "response_text": "Photo " + placeholder,
           "completion": completion, "messages": build_messages("system", "Photo " + placeholder, completion),
           "metadata": {"source_id": "source-one", "query_id": "query-one", "intent": "status", "assets": [{"url": placeholder}],
                        "expected_ui_contract_v5_4": {"expected_url": placeholder},
                        "expected_ui_contract_v5_4_source": "persisted",
                        "url_preprocessing": {"url_map": {placeholder: {"url": uri}}}}}

    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    prepared = script().load_training_dataset(str(source), None, tokenizer=Tokenizer())[0]
    assert prepared["response_text"] == "Photo " + uri
    assert prepared["source_id"] == "source-one" and prepared["query_id"] == "query-one"
    assert prepared["id"] == "one"
    assert placeholder in prepared["prompt"] and uri not in prepared["prompt"]
    assert prepared["assets"] == [{"url": uri}]
    assert prepared["intent_bucket"] == "status"
    assert prepared["expected_ui_contract"] == {"expected_url": uri}
    assert prepared["expected_ui_contract_source"] == "persisted"
    captured = {}
    def score(**kwargs):
        captured.update(kwargs)
        return [0.25]
    reward = audited_reward(score, str(tmp_path), 0)
    kwargs = {key: [value] for key, value in prepared.items() if key not in {"prompt", "completion"}}
    raw = completion + "tail"
    reward(completions=[raw], prompts=[prepared["prompt"]], **kwargs)
    assert captured["completions"] == [completion.replace(placeholder, uri)]
    assert captured["response_text"] == ["Photo " + uri]
    assert captured["expected_ui_contract"] == [{"expected_url": uri}]
    audit = json.loads((tmp_path / "grpo_rewards.rank0.jsonl").read_text())
    assert audit["raw_completion"] == raw
    assert audit["completion"] == captured["completions"][0]


@pytest.mark.parametrize("format", ["jsonl", "json-array", "json-lines", "jsonl-array"])
def test_grpo_loads_v11_null_then_string_metadata_without_changing_context(tmp_path, monkeypatch, format):
    datasets = pytest.importorskip("datasets")
    original_load = datasets.load_dataset

    def small_json_batches(*args, **kwargs):
        # The old raw JSON loader infers a null-only first batch, then fails
        # to cast the later string. Force that boundary without a large file.
        return original_load(*args, **kwargs, chunksize=1024, cache_dir=str(tmp_path / "cache"))

    monkeypatch.setattr(datasets, "load_dataset", small_json_batches)
    completion = '<a2ui>root=Text("Photo [IMAGE_URL_1]")</a2ui>'
    rows = [{
        "id": f"row-{index}", "response_text": "Photo [IMAGE_URL_1]", "completion": completion,
        "messages": build_messages("system", "Photo [IMAGE_URL_1]", completion),
        "metadata": {
            "intent": intent, "source_id": f"source-{index}", "query_id": f"query-{index}",
            "assets": [{"url": "[IMAGE_URL_1]"}],
            "expected_ui_contract_v5_4": {"expected_url": "[IMAGE_URL_1]"},
            "expected_ui_contract_v5_4_source": "persisted",
            "url_preprocessing": {"url_map": {"[IMAGE_URL_1]": {"url": "https://example.test/a.png"}}},
            "unused_v11": {"intent": intent, "padding": "x" * 2048},
        },
    } for index, intent in enumerate((None, "status"))]
    source = tmp_path / ("train.jsonl" if format.startswith("jsonl") else "train.json")
    payload = json.dumps(rows, indent=2) if format.endswith("array") else "\n".join(map(json.dumps, rows))
    source.write_text(payload, encoding="utf-8")
    original_bytes = source.read_bytes()
    module = script()
    prepared = module.load_training_dataset(str(source), None, tokenizer=Tokenizer())

    assert source.read_bytes() == original_bytes
    assert len(prepared) == 2
    assert "metadata" not in prepared.column_names
    for actual, row in zip(prepared, rows):
        expected = module.build_prediction_record(row, "")
        assert actual["id"] == row["id"]
        assert actual["prompt"] == Tokenizer().apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        assert actual["prompt_message_roles"] == [message["role"] for message in row["messages"][:-1]]
        assert actual["completion"] == completion
        assert actual["intent_bucket"] == row["metadata"]["intent"]
        for key in ("source_id", "query_id", "response_text", "response_text_sha256", "source_context_sha256"):
            assert actual[key] == expected[key]
        assert actual["assets"] == [{"url": "https://example.test/a.png"}]
        assert actual["expected_ui_contract"] == {"expected_url": "https://example.test/a.png"}
        assert actual["expected_ui_contract_source"] == "persisted"


@pytest.mark.parametrize("payload, error", [
    (json.dumps({"response_text": "source", "completion": '<a2ui>root=Text("OK")</a2ui>'}) + "\ninvalid\n", json.JSONDecodeError),
    ('["not a row"]', TypeError),
    ('[{"completion": "target"}]', ValueError),
    ('[]', ValueError),
    ('\n', ValueError),
])
def test_grpo_json_loader_rejects_invalid_rows(tmp_path, payload, error):
    pytest.importorskip("datasets")
    source = tmp_path / "train.json"
    source.write_text(payload, encoding="utf-8")
    with pytest.raises(error):
        script().load_training_dataset(str(source), "Create: {response_text}", prompt_source="template")


def test_grpo_model_dispatch_materializes_direct_and_wrapped_linear_targets(monkeypatch):
    torch = pytest.importorskip("torch")
    # Inventory-only modules; no model forward, tokenizer load, or weight load.
    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(2, 2)
    model.v_proj = torch.nn.Module()
    model.v_proj.linear = torch.nn.Linear(2, 2)
    module = script()
    calls = []
    monkeypatch.setattr(module, "load_hf_model", lambda model_id, config: calls.append((model_id, config)) or model)
    config = SimpleNamespace(target_modules=r"(?:q_proj|v_proj)(?:\.linear)?")
    args = SimpleNamespace(model="local-seed", model_loader="auto_multimodal_lm",
                           model_dtype="float32", attn_implementation="sdpa")
    actual, loading, resolved = module.load_grpo_model(args, config)
    assert actual is model
    assert calls == [("local-seed", loading)]
    assert loading == {"model_loader": "auto_multimodal_lm", "dtype": "float32",
                       "attn_implementation": "sdpa", "device_map": None,
                       "trust_remote_code": False, "require_exact_checkpoint_keys": True}
    assert resolved == ["q_proj", "v_proj.linear"]
    assert config.target_modules == {"q_proj", "v_proj.linear"}


def test_grpo_trainer_receives_materialized_model_without_init_kwargs():
    import ast
    tree = ast.parse(Path(script().__file__).read_text(encoding="utf-8-sig"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    trainer = next(call for call in calls if call.func.id == "GRPOTrainer")
    assert ast.unparse(next(kw.value for kw in trainer.keywords if kw.arg == "model")) == "model"
    config = next(call for call in calls if call.func.id == "make_grpo_config")
    assert "model_init_kwargs" not in {kw.arg for kw in config.keywords}
