from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ir_training.generation_policy import (
    build_stopping_criteria, closing_sentinel_end, generation_diagnostics,
    preserve_generation_eos, stop_express_completion,
)
from ir_training.eval.generate import build_prediction_record
from ir_training.eval.metrics import aggregate_scores, score_prediction
from ir_training.eval.compare_to_baseline import evaluate_predictions


class CharacterTokenizer:
    eos_token_id = 1
    chat_template = "test template"

    def __len__(self):
        return 512

    def decode(self, ids, skip_special_tokens=True):
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(chr(value) for value in values if value > 1)


def test_native_eos_survives_trainer_overwrite_and_invalid_ids_are_ignored():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[106, 50, -1, 900]),
                            config=SimpleNamespace(eos_token_id=1))
    captured = preserve_generation_eos(model, CharacterTokenizer())
    assert captured == [106, 50, 1]
    model.generation_config.eos_token_id = 1
    assert preserve_generation_eos(model, CharacterTokenizer(), captured) == [106, 50, 1]
    assert model.generation_config.eos_token_id == [106, 50, 1]


def test_missing_eos_is_an_actionable_error():
    with pytest.raises(ValueError, match="No valid generation EOS"):
        preserve_generation_eos(SimpleNamespace(), SimpleNamespace(eos_token_id=None))


@pytest.mark.parametrize("literal", ['"literal </a2ui> text"', r'"escaped \" </a2ui> text"', "'literal </a2ui> text'"])
def test_stop_ignores_close_tag_in_literal(literal):
    document = f"<a2ui>\nroot=Text({literal})\n</a2ui>"
    assert closing_sentinel_end(document) == len(document)
    assert stop_express_completion(document + "\n</a2ui>tail") == document
    assert stop_express_completion(document[:-1]) == document[:-1]


def test_batched_stopping_ignores_demo_and_handles_split_sentinel():
    import torch
    tokenizer = CharacterTokenizer()
    prompt = '<a2ui>\nroot=Text("demo")\n</a2ui>\n'
    criterion = build_stopping_criteria(tokenizer, len(prompt))
    unfinished = '<a2ui>\nroot=Text("hi")\n</a2ui'
    complete = unfinished + ">"
    literal = '<a2ui>\nroot=Text("</a2ui>"'
    def tokens(suffix):
        return torch.tensor([[ord(char) for char in prompt + suffix]])
    assert criterion(tokens("x"), None).tolist() == [False]
    assert criterion(tokens(unfinished), None).tolist() == [False]
    assert criterion(tokens(complete), None).tolist() == [True]
    assert criterion(tokens(literal), None).tolist() == [False]
    width = max(len(complete), len(literal))
    batch = torch.tensor([[ord(char) for char in prompt + suffix.ljust(width)] for suffix in (complete, literal)])
    assert criterion(batch, None).tolist() == [True, False]


def test_stop_reason_and_hashes_distinguish_eos_cap_and_envelope():
    tokenizer = CharacterTokenizer()
    for suffix, reason in [("hi", "max_new_tokens"), ("hi\x01", "eos_token"), ("</a2ui>", "closing_sentinel")]:
        ids = [ord(char) for char in suffix]
        result = generation_diagnostics(tokenizer, ids, eos_token_ids=[1], max_new_tokens=len(ids), prompt_text="prompt", input_ids=[4, 5])
        assert result["stop_reason"] == reason
        assert result["input_tokens"] == 2
        assert len(result["generation_policy_sha256"]) == 64
        assert "continuation_after_stop_unobserved" in result["raw_output_scope"]


def test_source_binding_keeps_metadata_and_restores_urls_before_comparison():
    source = "Photo https://example.test/a.png"
    row = {"id": "one", "source_id": "source-one", "response_text": source,
           "messages": [{"role": "user", "content": "demo"},
                        {"role": "user", "content": "Create A2UI Express v1 GenUI IR for this response:\n\nPhoto [IMAGE_URL_1]"}],
           "metadata": {"assets": [{"url": "[IMAGE_URL_1]"}], "expected_ui_contract_v5_4": {"required": 1},
                        "url_preprocessing": {"url_map": {"[IMAGE_URL_1]": {"url": "https://example.test/a.png"}}}}}
    output = '<a2ui>\nroot=Text("Photo")\n</a2ui>\ntail'
    record = build_prediction_record(row, output)
    assert record["response_text"] == source
    assert record["source_id"] == "source-one"
    assert record["assets"] == row["metadata"]["assets"]
    assert record["expected_ui_contract_v5_4"] == {"required": 1}
    assert record["raw_generated_text"] == output
    assert record["generated_text"].endswith("</a2ui>")
    row["response_text"] = "Wrong demo"
    with pytest.raises(ValueError, match="Scoring source mismatch"):
        build_prediction_record(row, output)


def test_failure_rows_remain_in_content_denominator():
    aggregate = aggregate_scores([{"metrics": {"content_coverage": 0.8, "native_syntax_valid": True}},
                                  {"metrics": {"native_syntax_valid": False}}])
    assert aggregate["content_coverage_avg"] == 0.4
    assert aggregate["content_coverage_observed_only_avg"] == 0.8
    assert aggregate["metric_observed_counts"]["content_coverage"] == 1


def test_raw_and_stopped_scores_are_separate_and_context_tampering_fails(tmp_path):
    output = '<a2ui>\nroot=Text("Hi")\n</a2ui>'
    record = build_prediction_record({"id": "one", "response_text": "Hi"}, output + "\n</a2ui>")
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = evaluate_predictions(path)
    assert result["raw_diagnostics"]["express_parse_ok_rate"] == 0
    assert result["serving_stopped_diagnostics"]["express_parse_ok_rate"] == 1
    record["assets"] = [{"url": "changed"}]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Scoring context hash mismatch"):
        evaluate_predictions(path)


def test_exact_text_match_is_distinct_from_semantic_graph_match():
    expected = '<a2ui>\nroot=Text("Hi")\n</a2ui>'
    reordered_spaces = '<a2ui>\nroot = Text("Hi")\n</a2ui>'
    metrics = score_prediction("Hi", expected, reordered_spaces)
    assert metrics["exact_match"] is False
    assert metrics["semantic_match"] is True


def test_v54_exposes_declared_root_and_cap_without_repair():
    output = '<a2ui>\nroot=Text("Visible")\na=Text("Detached")\n</a2ui>'
    metrics = score_prediction("Visible Detached", None, output, metric_version="v5_4")
    assert metrics["root_reachable_fraction_v5_4"] == 0.5
    assert metrics["fully_root_reachable_v5_4"] is False
    assert metrics["graph_evidence_v5_4"]["unreachable_ids"] == ["a"]
    assert any(item["name"] == "reachability_below_90" for item in metrics["active_caps_v5_4"])


def test_full_checkpoint_manifest_and_post_trainer_eos_are_bound(tmp_path):
    from ir_training.train.callbacks import _checkpoint_adapter_manifest, build_checkpoint_provenance_callback
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"controlled test fixture")
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = _checkpoint_adapter_manifest(checkpoint, role="trainer_intermediate")
    assert manifest["checkpoint_kind"] == "full_model"
    assert {item["path"] for item in manifest["files"]} == {"model.safetensors", "config.json", "tokenizer.json"}
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1), config=SimpleNamespace(eos_token_id=1))
    metadata = {}
    callback = build_checkpoint_provenance_callback(output_dir=tmp_path, metadata=metadata,
        tokenizer=CharacterTokenizer(), generation_eos_ids=[106, 50, 1])
    control = object()
    assert callback.on_train_begin(None, SimpleNamespace(is_world_process_zero=True), control, model=model) is control
    assert model.generation_config.eos_token_id == [106, 50, 1]
    assert metadata["effective_generation_eos_ids"] == [106, 50, 1]


def _litert_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_litertlm_a2ui_express.py"
    spec = importlib.util.spec_from_file_location("litert_eval_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_litert_strict_validation_rejects_duplicate_and_missing_reference():
    module = _litert_module()
    valid = '<a2ui>\nroot=Text("Hi")\n</a2ui>'
    assert module._validate_output(valid) == (True, None)
    for text in ['<a2ui>\nroot=Column([a])\n</a2ui>', '<a2ui>\nroot=Text("A")\nroot=Text("B")\n</a2ui>']:
        assert module._validate_output(text)[0] is False


def test_litert_prepared_conversation_keeps_serializer_demo_and_excludes_target():
    module = _litert_module()
    row = {"response_text": "Task", "messages": [
        {"role": "system", "content": "custom compact system"},
        {"role": "user", "content": "demo input"},
        {"role": "assistant", "content": "root-last custom demo"},
        {"role": "user", "content": "Create A2UI Express v1 GenUI IR for this response:\n\nTask"},
        {"role": "assistant", "content": "held-out answer"}]}
    system, preface, target = module._prepared_conversation(row)
    assert system == "custom compact system"
    assert preface[-1] == {"role": "model", "content": "root-last custom demo"}
    assert target.endswith("Task")
    assert "held-out answer" not in str((system, preface, target))
