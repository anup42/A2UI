from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ir_training.data.chat_templates import build_messages
from ir_training.train.grpo_runtime import audited_reward


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


def test_grpo_rewards_use_golden_source_contract_and_url_context(monkeypatch, tmp_path):
    placeholder = "[IMAGE_URL_1]"
    uri = "https://example.test/a.png"
    completion = f'<a2ui>\nroot=Text("Photo {placeholder}")\n</a2ui>'
    row = {"id": "one", "response_text": "Photo " + placeholder,
           "completion": completion, "messages": build_messages("system", "Photo " + placeholder, completion),
           "metadata": {"source_id": "source-one", "query_id": "query-one", "intent": "status", "assets": [{"url": placeholder}],
                        "expected_ui_contract_v5_4": {"expected_url": placeholder},
                        "expected_ui_contract_v5_4_source": "persisted",
                        "url_preprocessing": {"url_map": {placeholder: {"url": uri}}}}}

    class FakeDataset:
        column_names = list(row)
        def map(self, fn, **kwargs):
            return [fn(row)]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *args, **kwargs: FakeDataset()))

    class Tokenizer:
        bos_token_id = None
        chat_template = "test"
        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(item["content"] for item in messages)
        def __call__(self, text, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": [ord(char) for char in text]}

    prepared = script().load_training_dataset("unused.jsonl", None, tokenizer=Tokenizer())[0]
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
