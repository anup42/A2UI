from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.cuda_env import normalize_cuda_visible_devices
from ir_training.data.build_pairs import prepare_dataset
from ir_training.data.url_preprocess import (
    preprocess_training_urls,
    restore_url_placeholders,
)
from ir_training.eval.metrics import aggregate_scores
from ir_training.export.edge_gallery import build_litert_export_command
from ir_training.export.manifest import build_manifest, write_manifest
from ir_training.models.registry import create_adapter, supported_families
from ir_training.train import callbacks as callbacks_module
from ir_training.train import sft as sft_module
from ir_training.train.callbacks import (
    build_checkpoint_provenance_callback,
    build_golden_set_eval_callback,
)
from ir_training.train.sft import (
    _align_tokenizer_and_model,
    _build_checked_causal_lm_trainer,
    _CausalLMDataCollator,
    _checked_shifted_causal_lm_loss,
    _disable_peft_vocab_probe,
    _effective_max_seq_length,
    _enforce_cuda_requirement,
    _ensure_tensorboard_reporter,
    _model_output_vocab_size,
    _model_position_limit,
    _model_vocab_size,
    _resolve_training_dtype,
    _summarize_training_sample_models,
    _tokenize_completion_only_row,
    _training_precision_flags,
    _validate_sft_token_ids,
    _validate_tokenized_sft_dataset,
)


def test_checked_trainer_declares_mean_loss_does_not_accept_loss_kwargs():
    class BaseTrainer:
        def __init__(self):
            self.model_accepts_loss_kwargs = True

    trainer_cls = _build_checked_causal_lm_trainer(BaseTrainer)
    trainer = trainer_cls()

    assert trainer.model_accepts_loss_kwargs is False


def test_disable_peft_vocab_probe_preserves_adapter_origin_and_forces_save_flag(
    tmp_path,
):
    calls = []

    class Config:
        base_model_name_or_path = "google/gemma-4-E2B-it"

    class Model:
        peft_config = {"default": Config()}

        def save_pretrained(self, output, *args, **kwargs):
            calls.append((output, args, kwargs))

    model = Model()
    _disable_peft_vocab_probe(model)
    model.save_pretrained(tmp_path, safe_serialization=True)

    assert (
        model.peft_config["default"].base_model_name_or_path == "google/gemma-4-E2B-it"
    )
    assert calls == [
        (tmp_path, (), {"safe_serialization": True, "save_embedding_layers": False})
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def test_prepare_dataset_filters_and_splits(tmp_path):
    run_dir = tmp_path / "run"
    response = {
        "response_id": "r1",
        "query_id": "q1",
        "intent": "Weather",
        "intent_bucket": "weather",
        "response_text": "Weather in Bengaluru is mild.",
        "gen": {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "prompt_version": "response_v1",
        },
    }
    spec = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Text", "props": {"text": "Weather"}, "children": []}
        },
    }
    _write_jsonl(run_dir / "responses.jsonl", [response])
    _write_jsonl(
        run_dir / "genui.jsonl",
        [
            {
                "response_id": "r1",
                "ui_id": "u1",
                "genui_json": spec,
                "expected_ui_contract_v5_4": {"contract_version": "5.4.0-test"},
                "expected_ui_contract_v5_4_source": "persisted",
                "gen": {
                    "provider": "azure_openai",
                    "model": "gpt-5.4-mini",
                    "prompt_version": "genui_v1",
                },
            }
        ],
    )
    out_dir = tmp_path / "prepared"
    manifest = prepare_dataset(
        {
            "run": {
                "source_run_dir": str(run_dir),
                "output_dir": str(out_dir),
                "system_prompt": "Return JSON.",
            },
            "filters": {
                "require_strict_flat_spec": True,
                "max_input_chars": 1000,
                "max_output_chars": 1000,
            },
            "split": {"train": 1, "val": 0, "test": 0, "stratify_by": "intent_bucket"},
        }
    )
    assert manifest["counts"]["accepted"] == 1
    assert (
        manifest["model_counts"]["response_generation"]["gemini/gemini-2.5-flash"] == 1
    )
    assert manifest["model_counts"]["ir_generation"]["azure_openai/gpt-5.4-mini"] == 1
    assert (out_dir / "train.jsonl").exists()
    row = json.loads(
        (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["metadata"]["response_generation"]["model"] == "gemini-2.5-flash"
    assert row["metadata"]["ir_generation"]["model"] == "gpt-5.4-mini"
    assert row["expected_ui_contract_v5_4"] == {"contract_version": "5.4.0-test"}
    assert row["expected_ui_contract_v5_4_source"] == "persisted"


def test_prepare_dataset_reads_stage3_folder_and_uses_90_10_split(tmp_path):
    source_dir = tmp_path / "stage3"
    spec = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Text", "props": {"text": "Weather"}, "children": []}
        },
    }
    rows = [
        {
            "response_id": f"r{i:03d}",
            "ui_id": f"u{i:03d}",
            "intent_bucket": "weather",
            "response_text": f"Weather sample {i}",
            "genui_json": spec,
        }
        for i in range(10)
    ]
    _write_jsonl(source_dir / "part_a.jsonl", rows[:5])
    _write_jsonl(source_dir / "part_b.jsonl", rows[5:])
    out_dir = tmp_path / "prepared_folder"

    manifest = prepare_dataset(
        {
            "run": {
                "source_genui_dir": str(source_dir),
                "source_glob": "*.jsonl",
                "output_dir": str(out_dir),
                "seed": 123,
                "system_prompt": "Return JSON.",
            },
            "filters": {
                "require_strict_flat_spec": True,
                "max_input_chars": 1000,
                "max_output_chars": 1000,
            },
            "split": {
                "train": 0.9,
                "val": 0.1,
                "test": 0.0,
                "stratify_by": "intent_bucket",
            },
        }
    )

    assert manifest["counts"]["accepted"] == 10
    assert manifest["counts"]["train"] == 9
    assert manifest["counts"]["val"] == 1
    assert manifest["counts"]["test"] == 0
    assert manifest["counts"]["all"] == 10
    assert len(manifest["source_genui_paths"]) == 2
    assert (out_dir / "all.jsonl").exists()


def test_prepare_dataset_fails_closed_when_fixed_set_count_is_wrong(tmp_path):
    run_dir = tmp_path / "fixed-run"
    spec = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Text", "props": {"text": "One"}, "children": []}
        },
    }
    _write_jsonl(
        run_dir / "genui.jsonl",
        [{"response_id": "r1", "response_text": "One", "genui_json": spec}],
    )

    with pytest.raises(ValueError, match="exactly 100 accepted rows"):
        prepare_dataset(
            {
                "run": {
                    "source_run_dir": str(run_dir),
                    "output_dir": str(tmp_path / "prepared-fixed"),
                },
                "filters": {
                    "require_strict_express": True,
                    "required_accepted_rows": 100,
                    "require_exact_accepted_rows": True,
                },
                "split": {"train": 1.0, "val": 0.0, "test": 0.0},
            }
        )


def test_prepare_dataset_assigns_source_groups_before_target_materialization(tmp_path):
    run_dir = tmp_path / "grouped_run"
    spec = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Text", "props": {"text": "same"}, "children": []}
        },
    }
    responses = []
    genui = []
    for index in range(12):
        responses.append(
            {
                "response_id": f"r{index}",
                "source_id": f"source-{index // 2}",
                "response_text": f"Source group {index // 2}",
                "intent_bucket": "status" if index % 2 else "weather",
            }
        )
        genui.append(
            {
                "response_id": f"r{index}",
                "source_id": f"source-{index // 2}",
                "ui_id": f"u{index}",
                "genui_json": spec,
            }
        )
    _write_jsonl(run_dir / "responses.jsonl", responses)
    _write_jsonl(run_dir / "genui.jsonl", genui)
    out_dir = tmp_path / "grouped_out"
    manifest = prepare_dataset(
        {
            "run": {
                "source_run_dir": str(run_dir),
                "output_dir": str(out_dir),
                "seed": 7,
            },
            "filters": {"max_input_chars": 1000, "max_output_chars": 1000},
            "split": {
                "train": 0.5,
                "val": 0.25,
                "test": 0.25,
                "stratify_by": "intent_bucket",
            },
        }
    )
    assert (
        manifest["split_assignment_stage"]
        == "source_group_before_target_materialization"
    )
    assignments = {}
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            previous = assignments.setdefault(row["source_id"], split)
            assert previous == split
    assert len(assignments) == manifest["source_group_count"]


def test_url_preprocessing_placeholderizes_and_restores_roles():
    spec = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["image", "cta"]},
            "image": {
                "type": "Image",
                "props": {"url": "https://upload.wikimedia.org/example.jpg"},
                "children": [],
            },
            "cta": {
                "type": "Button",
                "props": {"label": "Open"},
                "on": {
                    "press": {
                        "action": "openUrl",
                        "params": {"url": "https://example.org/details"},
                    }
                },
                "children": [],
            },
        },
    }
    result = preprocess_training_urls(
        "Images: https://upload.wikimedia.org/example.jpg\nAction: [Button: Open] https://example.org/details",
        spec,
    )
    assert "[IMAGE_URL_1]" in result.response_text
    assert "[ACTION_URL_1]" in result.response_text
    assert result.genui_json["elements"]["image"]["props"]["url"] == "[IMAGE_URL_1]"
    assert (
        result.genui_json["elements"]["cta"]["on"]["press"]["params"]["url"]
        == "[ACTION_URL_1]"
    )
    assert restore_url_placeholders(result.genui_json, result.url_map) == spec


def test_source_identity_url_binding_reuses_input_token_across_target_roles():
    raw = "https://example.org/details"
    spec = {
        "root": "cta",
        "state": {},
        "elements": {
            "cta": {
                "type": "Button",
                "props": {"label": "Open"},
                "on": {"press": {"action": "openUrl", "params": {"url": raw}}},
                "children": [],
            }
        },
    }
    result = preprocess_training_urls(
        f"Source: {raw}", spec, binding_policy="source_identity"
    )
    source_token = next(iter(result.url_map))
    assert source_token == "[SOURCE_URL_1]"
    assert source_token in result.response_text
    assert (
        result.genui_json["elements"]["cta"]["on"]["press"]["params"]["url"]
        == source_token
    )
    assert (
        restore_url_placeholders(result.response_text, result.url_map)
        == f"Source: {raw}"
    )
    assert restore_url_placeholders(result.genui_json, result.url_map) == spec


def test_source_identity_url_binding_keeps_target_only_reference_detectable():
    raw = "https://example.org/unseen"
    spec = {
        "root": "image",
        "state": {},
        "elements": {"image": {"type": "Image", "props": {"url": raw}, "children": []}},
    }
    result = preprocess_training_urls(
        "No reference is declared here.", spec, binding_policy="source_identity"
    )
    target_token = result.genui_json["elements"]["image"]["props"]["url"]
    assert target_token in result.url_map
    assert target_token not in result.response_text
    assert restore_url_placeholders(result.genui_json, result.url_map) == spec


def test_url_preprocessing_round_trips_instructional_bracketed_host():
    raw = "https://[Your-Public-IP]:8443/admin"
    spec = {
        "root": "cta",
        "state": {},
        "elements": {
            "cta": {
                "type": "Button",
                "props": {"label": "Open router"},
                "on": {"press": {"action": "openUrl", "params": {"url": raw}}},
                "children": [],
            }
        },
    }

    result = preprocess_training_urls(
        f"Example address: {raw}", spec, binding_policy="source_identity"
    )

    token = next(iter(result.url_map))
    assert result.url_map[token]["url"] == raw
    assert result.url_map[token]["host"] == ""
    assert restore_url_placeholders(result.response_text, result.url_map) == (
        f"Example address: {raw}"
    )
    assert restore_url_placeholders(result.genui_json, result.url_map) == spec


def test_prepare_dataset_records_source_identity_url_binding_policy(tmp_path):
    run_dir = tmp_path / "source-bound-run"
    raw = "https://example.org/details"
    _write_jsonl(
        run_dir / "responses.jsonl",
        [
            {
                "response_id": "response-1",
                "query_id": "query-1",
                "response_text": f"Reference: {raw}",
            }
        ],
    )
    _write_jsonl(
        run_dir / "genui.jsonl",
        [
            {
                "response_id": "response-1",
                "ui_id": "ui-1",
                "genui_json": {
                    "root": "button",
                    "state": {},
                    "elements": {
                        "button": {
                            "type": "Button",
                            "props": {"label": "Details"},
                            "children": [],
                            "on": {
                                "press": {"action": "openUrl", "params": {"url": raw}}
                            },
                        }
                    },
                },
            }
        ],
    )
    output = tmp_path / "source-bound-output"
    manifest = prepare_dataset(
        {
            "run": {"source_run_dir": str(run_dir), "output_dir": str(output)},
            "filters": {"max_input_chars": 1000, "max_output_chars": 1000},
            "url_preprocessing": {"enabled": True, "binding_policy": "source_identity"},
            "split": {"train": 1, "val": 0, "test": 0},
        }
    )
    row = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    token = next(iter(row["metadata"]["url_preprocessing"]["url_map"]))
    assert token in row["response_text"] and token in row["completion"]
    assert row["metadata"]["url_preprocessing"]["binding_policy"] == "source_identity"
    assert manifest["url_preprocessing"]["binding_policy"] == "source_identity"


def test_url_binding_policy_rejects_unknown_value_even_when_disabled():
    with pytest.raises(ValueError, match="Unsupported URL binding policy"):
        preprocess_training_urls("plain", {}, enabled=False, binding_policy="guess")


def test_url_preprocessing_masks_all_uri_schemes_and_local_asset_paths():
    response = (
        "Source=content://media/external/images/media/42\n"
        "Reference=data:image/png;base64,AAAA\n"
        "Media: Image=../assets/cards/flight.png\n"
        'Media: Image="C:\\GenUI Assets\\arrival photo.webp"\n'
        "Media: Asset=assets/models/gemma\n"
        "Media: Icon=@drawable/ic_plane"
    )
    spec = {
        "root": "root",
        "state": {
            "content": "content://media/external/images/media/42",
            "inline": "data:image/png;base64,AAAA",
            "asset": "../assets/cards/flight.png",
            "quoted": "C:\\GenUI Assets\\arrival photo.webp",
            "model": "assets/models/gemma",
            "icon": "@drawable/ic_plane",
        },
        "elements": {"root": {"type": "Stack", "props": {}, "children": []}},
    }

    result = preprocess_training_urls(response, spec)

    for raw_reference in spec["state"].values():
        assert raw_reference not in result.response_text
        assert raw_reference not in json.dumps(result.genui_json)
    assert len(result.url_map) == 6, json.dumps(result.url_map, sort_keys=True)
    assert {entry["kind"] for entry in result.url_map.values()} == {
        "url",
        "local_asset",
    }
    assert restore_url_placeholders(result.response_text, result.url_map) == response
    assert restore_url_placeholders(result.genui_json, result.url_map) == spec


def test_cuda_visible_devices_uses_all_detected_healthy_gpus():
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    old_detector = normalize_cuda_visible_devices.__globals__[
        "_detect_queryable_gpu_indices"
    ]
    try:
        normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"] = (
            lambda: ["0", "1", "3"]
        )

        result = normalize_cuda_visible_devices(env)

        assert result == "0,1,3"
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,3"
    finally:
        normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"] = (
            old_detector
        )


def test_cuda_visible_devices_detects_all_when_shell_does_not_set_it():
    env = {}
    old_detector = normalize_cuda_visible_devices.__globals__[
        "_detect_queryable_gpu_indices"
    ]
    try:
        normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"] = (
            lambda: ["0", "1", "3"]
        )

        result = normalize_cuda_visible_devices(env)

        assert result == "0,1,3"
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,3"
    finally:
        normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"] = (
            old_detector
        )


def test_cuda_visible_devices_honors_explicit_and_excludes_bad_device():
    env = {"A2UI_CUDA_VISIBLE_DEVICES": "0,1,2,3", "A2UI_EXCLUDE_CUDA_DEVICES": "2"}

    result = normalize_cuda_visible_devices(env)

    assert result == "0,1,3"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,3"


def test_completion_only_tokenization_masks_prompt_and_keeps_completion():
    class Tokenizer:
        def __call__(self, text, **_kwargs):
            values = {"P": 10, "C": 20}
            return {"input_ids": [values[ch] for ch in text]}

    row = _tokenize_completion_only_row(
        tokenizer=Tokenizer(),
        prompt_text="PPPP",
        completion_text="CCC",
        full_text="PPPPCCC",
        max_seq_length=7,
    )

    assert row["input_ids"] == [10, 10, 10, 10, 20, 20, 20]
    assert row["labels"] == [-100, -100, -100, -100, 20, 20, 20]


def test_collator_preserves_completion_only_labels():
    class PaddedBatch(dict):
        pass

    class Tokenizer:
        pad_token_id = 0

        def pad(self, features, padding=True, return_tensors="pt"):
            import torch

            max_len = max(len(feature["input_ids"]) for feature in features)
            input_ids = []
            attention = []
            for feature in features:
                values = list(feature["input_ids"])
                pad = max_len - len(values)
                input_ids.append(values + [0] * pad)
                attention.append([1] * len(values) + [0] * pad)
            return PaddedBatch(
                {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention),
                }
            )

    collator = _CausalLMDataCollator(
        Tokenizer(), vocab_size=30, max_position_embeddings=8
    )
    batch = collator(
        [
            {
                "input_ids": [10, 20, 20],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 20, 20],
            },
            {
                "input_ids": [10, 10, 20],
                "attention_mask": [1, 1, 1],
                "labels": [-100, -100, 20],
            },
        ]
    )

    assert batch["labels"].tolist() == [[-100, 20, 20], [-100, -100, 20]]


def test_prepare_dataset_writes_url_map_metadata(tmp_path):
    run_dir = tmp_path / "run"
    response = {
        "response_id": "r1",
        "query_id": "q1",
        "intent": "Travel",
        "intent_bucket": "travel",
        "response_text": "Action: [Button: Open] https://example.org/details",
    }
    spec = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Button",
                "props": {"label": "Open"},
                "on": {
                    "press": {
                        "action": "openUrl",
                        "params": {"url": "https://example.org/details"},
                    }
                },
                "children": [],
            }
        },
    }
    _write_jsonl(run_dir / "responses.jsonl", [response])
    _write_jsonl(
        run_dir / "genui.jsonl",
        [
            {
                "response_id": "r1",
                "ui_id": "u1",
                "genui_json": spec,
                "gen": {
                    "provider": "azure_openai",
                    "model": "gpt-5.4-mini",
                    "prompt_version": "genui_v1",
                },
            }
        ],
    )
    out_dir = tmp_path / "prepared_urls"

    prepare_dataset(
        {
            "run": {
                "source_run_dir": str(run_dir),
                "output_dir": str(out_dir),
                "system_prompt": "Return JSON.",
            },
            "filters": {
                "require_strict_flat_spec": True,
                "max_input_chars": 1000,
                "max_output_chars": 1000,
            },
            "url_preprocessing": {"enabled": True},
            "split": {"train": 1, "val": 0, "test": 0, "stratify_by": "intent_bucket"},
        }
    )

    row = json.loads(
        (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "[ACTION_URL_1]" in row["prompt"]
    assert "[ACTION_URL_1]" in row["completion"]
    url_map = row["metadata"]["url_preprocessing"]["url_map"]
    assert url_map["[ACTION_URL_1]"]["url"] == "https://example.org/details"


def test_prepare_dataset_masks_asset_metadata_before_training_prompt(tmp_path):
    run_dir = tmp_path / "run_assets"
    response = {"response_id": "r1", "response_text": "Show the image"}
    spec = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Image",
                "props": {"url": "https://example.org/a.png"},
                "children": [],
            }
        },
    }
    _write_jsonl(run_dir / "responses.jsonl", [response])
    _write_jsonl(
        run_dir / "genui.jsonl",
        [
            {
                "response_id": "r1",
                "genui_json": spec,
                "assets": [
                    {"url": "https://example.org/a.png", "path": "C:/private/a.png"}
                ],
            }
        ],
    )
    out_dir = tmp_path / "prepared_assets"
    prepare_dataset(
        {
            "run": {"source_run_dir": str(run_dir), "output_dir": str(out_dir)},
            "filters": {"max_input_chars": 1000, "max_output_chars": 1000},
            "split": {"train": 1, "val": 0, "test": 0},
        }
    )
    row = json.loads(
        (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    serialized = json.dumps(row["prompt"], ensure_ascii=False)
    assert "https://example.org" not in serialized
    assert "C:/private" not in serialized
    assert row["assets"][0]["url"].startswith("[")


def test_model_registry_formats_example():
    adapter = create_adapter({"family": "gemma", "model_id": "google/gemma-4-E2B-it"})
    text = adapter.format_example({"messages": [{"role": "user", "content": "Hello"}]})
    assert "Hello" in text
    assert "gemma" in supported_families()


def test_gemma4_lora_targets_inner_linear_modules():
    adapter = create_adapter({"family": "gemma", "model_id": "google/gemma-4-E2B-it"})
    targets = adapter.default_lora_targets()
    assert "q_proj.linear" in targets
    assert "q_proj" not in targets


def test_model_adapter_can_force_fast_tokenizer_loader(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeFastLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return FakeTokenizer()

    class UnexpectedAutoLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError(
                "AutoTokenizer should not be used for the explicit fast-tokenizer loader"
            )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=UnexpectedAutoLoader,
            PreTrainedTokenizerFast=FakeFastLoader,
        ),
    )
    adapter = create_adapter(
        {
            "family": "gemma",
            "model_id": "google/gemma-4-E2B-it",
            "tokenizer_loader": "pretrained_tokenizer_fast",
        }
    )

    tokenizer = adapter.load_tokenizer()

    assert calls == [("google/gemma-4-E2B-it", {"trust_remote_code": False})]
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "right"


def test_golden_eval_callback_runs_on_evaluate_and_keeps_best(tmp_path, monkeypatch):
    split_path = tmp_path / "golden.jsonl"
    _write_jsonl(split_path, [{"id": "one"}])
    output_dir = tmp_path / "eval"
    best_dir = tmp_path / "best"
    generated_dirs: list[Path] = []
    saved_scores: list[float] = []
    logged_metrics: list[dict[str, float]] = []
    scores = iter([20.0, 10.0])

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(TrainerCallback=object),
    )

    def fake_generate_predictions(**kwargs):
        output_path = kwargs["output_path"]
        generated_dirs.append(output_path.parent)
        _write_jsonl(output_path, [{"generated_text": "{}"}])
        return 1

    def fake_evaluate_predictions(**kwargs):
        score = next(scores)
        return {
            "overall_score": score,
            "generation_reward_v5_4_avg": score + 0.5,
            "count": 1,
        }

    def fake_save_best_checkpoint(**kwargs):
        saved_scores.append(float(kwargs["best_info"]["metric_value"]))

    monkeypatch.setattr(
        callbacks_module, "_generate_predictions_with_model", fake_generate_predictions
    )
    monkeypatch.setattr(
        callbacks_module, "evaluate_predictions", fake_evaluate_predictions
    )
    monkeypatch.setattr(
        callbacks_module, "_save_best_golden_checkpoint", fake_save_best_checkpoint
    )
    monkeypatch.setattr(callbacks_module, "_distributed_context", lambda: (0, 1))
    monkeypatch.setattr(callbacks_module, "_distributed_barrier", lambda: None)

    callback = build_golden_set_eval_callback(
        enabled=True,
        split_path=split_path,
        output_dir=output_dir,
        adapter=object(),
        tokenizer=object(),
        trigger="evaluate",
        metric_for_best_model="overall_score",
        best_checkpoint_dir=best_dir,
        metric_logger=logged_metrics.append,
        metric_log_prefix="golden100",
    )
    control = object()
    model = object()

    callback.on_epoch_end(
        None, types.SimpleNamespace(epoch=1.0, global_step=5), control, model=model
    )
    callback.on_evaluate(
        None, types.SimpleNamespace(epoch=1.0, global_step=10), control, model=model
    )
    callback.on_evaluate(
        None, types.SimpleNamespace(epoch=2.0, global_step=20), control, model=model
    )

    assert generated_dirs == [
        output_dir / "step_000000010",
        output_dir / "step_000000020",
    ]
    assert saved_scores == [20.0]
    assert [item["eval_golden100/v5_4_score"] for item in logged_metrics] == [
        20.5,
        10.5,
    ]
    assert [item["eval_golden100/overall_score"] for item in logged_metrics] == [
        20.0,
        10.0,
    ]
    assert all("eval_golden100/step" not in item for item in logged_metrics)
    assert callback.summary() == {
        "metric": "overall_score",
        "metric_value": 20.0,
        "greater_is_better": True,
        "step": 10,
        "epoch": 1.0,
        "checkpoint_dir": str(best_dir),
    }


def test_golden_eval_rejects_a_short_or_duplicate_fixed_set(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(TrainerCallback=object),
    )
    short_split = tmp_path / "short.jsonl"
    _write_jsonl(short_split, [{"id": f"row-{index}"} for index in range(99)])

    with pytest.raises(ValueError, match="exactly 100 valid rows"):
        build_golden_set_eval_callback(
            enabled=True,
            split_path=short_split,
            output_dir=tmp_path / "short-output",
            adapter=object(),
            tokenizer=object(),
            max_rows=100,
            required_rows=100,
            require_exact_rows=True,
        )

    duplicate_split = tmp_path / "duplicate.jsonl"
    _write_jsonl(duplicate_split, [{"id": "same"}, {"id": "same"}])
    with pytest.raises(ValueError, match="duplicate row identities"):
        build_golden_set_eval_callback(
            enabled=True,
            split_path=duplicate_split,
            output_dir=tmp_path / "duplicate-output",
            adapter=object(),
            tokenizer=object(),
            max_rows=2,
            required_rows=2,
            require_exact_rows=True,
            require_unique_rows=True,
        )


def test_each_trainer_checkpoint_gets_self_contained_provenance(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(TrainerCallback=object),
    )
    output_dir = tmp_path / "run"
    checkpoint = output_dir / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_bytes(b"fixed-scale-adapter")
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "resolved.yaml"
    config.write_text("qat:\n  scale_mode: retained_mobile\n", encoding="utf-8")
    best_checkpoint = tmp_path / "best_golden"
    best_checkpoint.mkdir()
    best_adapter = best_checkpoint / "adapter_model.safetensors"
    best_adapter.write_bytes(b"best-fixed-scale-adapter")
    (best_checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    callback = build_checkpoint_provenance_callback(
        output_dir=output_dir,
        metadata={
            "training_metadata_version": 4,
            "qat": {"retained_qparams_binding_count": 205},
            "numeric_preflight": {"passed": True},
        },
        config_path=config,
        golden_summary_provider=lambda: {
            "metric": "generation_reward_v5_4_avg",
            "metric_value": 88.0,
            "step": 500,
            "epoch": 0.25,
            "checkpoint_dir": str(best_checkpoint),
        },
    )
    state = types.SimpleNamespace(
        global_step=500,
        epoch=0.25,
        is_world_process_zero=True,
        log_history=[{"eval_golden100/v5_4_score": 88.0}],
    )

    callback.on_save(None, state, object())

    payload = json.loads(
        (checkpoint / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert payload["checkpoint_step"] == 500
    assert payload["qat"]["retained_qparams_binding_count"] == 205
    assert payload["numeric_preflight"]["passed"] is True
    assert payload["best_golden_eval"]["metric_value"] == 88.0
    assert payload["last_trainer_log"]["eval_golden100/v5_4_score"] == 88.0
    weights = next(
        item
        for item in payload["checkpoint_adapter_files"]
        if item["path"] == adapter.name
    )
    assert weights["size_bytes"] == adapter.stat().st_size
    assert len(weights["sha256"]) == 64
    assert payload["adapter_checkpoints"][0]["role"] == "trainer_intermediate"
    assert payload["adapter_checkpoints"][0]["files"][0]["size"] > 0
    assert (checkpoint / "training_config.yaml").read_text(
        encoding="utf-8"
    ) == config.read_text(encoding="utf-8")
    best_payload = json.loads(
        (best_checkpoint / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert best_payload["checkpoint_role"] == "best_golden"
    best_weight_record = next(
        item
        for item in best_payload["adapter_checkpoints"][0]["files"]
        if item["path"] == best_adapter.name
    )
    assert (
        best_weight_record["sha256"]
        == hashlib.sha256(best_adapter.read_bytes()).hexdigest()
    )
    assert (
        best_checkpoint / "training_config.yaml"
    ).read_bytes() == config.read_bytes()
    assert not list(checkpoint.glob("*.partial"))
    assert not list(best_checkpoint.glob("*.partial"))


def test_checkpoint_provenance_callback_skips_nonzero_rank(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(TrainerCallback=object),
    )
    callback = build_checkpoint_provenance_callback(
        output_dir=tmp_path / "run",
        metadata={"training_metadata_version": 4},
    )
    state = types.SimpleNamespace(
        global_step=500,
        epoch=0.25,
        is_world_process_zero=False,
        log_history=[],
    )
    control = object()

    returned = callback.on_save(None, state, control)

    assert returned is control
    assert not (tmp_path / "run").exists()


def test_checkpoint_provenance_callback_rejects_missing_adapter(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(TrainerCallback=object),
    )
    checkpoint = tmp_path / "run" / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    callback = build_checkpoint_provenance_callback(
        output_dir=tmp_path / "run",
        metadata={"training_metadata_version": 4},
    )
    state = types.SimpleNamespace(
        global_step=500,
        epoch=0.25,
        is_world_process_zero=True,
        log_history=[],
    )

    with pytest.raises(RuntimeError, match="no local adapter files"):
        callback.on_save(None, state, object())

    assert not (checkpoint / "training_metadata.json").exists()
    assert not list(checkpoint.glob("*.partial"))


def test_tensorboard_reporter_is_added_without_removing_existing_reporters():
    assert _ensure_tensorboard_reporter("none") == "tensorboard"
    assert _ensure_tensorboard_reporter("tensorboard") == "tensorboard"
    assert _ensure_tensorboard_reporter(["wandb"]) == ["wandb", "tensorboard"]


def test_training_limit_uses_positive_max_steps_as_optimizer_step_override():
    report = sft_module._training_limit_config({"epochs": 2, "max_steps": 3})

    assert report == {
        "mode": "max_optimizer_steps",
        "max_optimizer_steps": 3,
        "num_train_epochs": 2.0,
        "max_steps_overrides_epochs": True,
    }
    training_args = {"max_steps": 999}
    sft_module._apply_training_limit_to_args(training_args, report)
    assert training_args == {"num_train_epochs": 2.0, "max_steps": 3}


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "3", None])
def test_training_limit_rejects_invalid_max_steps_without_epoch_fallback(invalid):
    with pytest.raises(ValueError, match="positive integer optimizer-step"):
        sft_module._training_limit_config({"epochs": 2, "max_steps": invalid})


def test_training_limit_defaults_to_epochs_only_when_max_steps_is_absent():
    report = sft_module._training_limit_config({"epochs": 1.5})
    assert report == {
        "mode": "num_train_epochs",
        "max_optimizer_steps": None,
        "num_train_epochs": 1.5,
        "max_steps_overrides_epochs": False,
    }
    training_args = {"max_steps": 999}
    sft_module._apply_training_limit_to_args(training_args, report)
    assert training_args == {"num_train_epochs": 1.5}


def test_bounded_golden_eval_cadence_must_fit_optimizer_step_limit():
    limit = sft_module._training_limit_config({"epochs": 2, "max_steps": 3})
    sft_module._validate_bounded_eval_save_cadence(
        {"max_steps": 3, "eval_steps": 3, "save_steps": 3}, limit
    )

    with pytest.raises(ValueError, match="no greater than training.max_steps=3"):
        sft_module._validate_bounded_eval_save_cadence(
            {"max_steps": 3, "eval_steps": 4, "save_steps": 4}, limit
        )


def test_greedy_numeric_preflight_repeats_deterministically_and_restores_mode():
    import torch

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            return {"input_ids": [3, 4, 5] if text else []}

        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(str(value) for value in tokens)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def generate(self, *, input_ids, max_new_tokens, **kwargs):
            generated = torch.arange(
                10,
                10 + int(max_new_tokens),
                device=input_ids.device,
                dtype=input_ids.dtype,
            ).reshape(1, -1)
            return torch.cat((input_ids, generated), dim=-1)

    model = Model()
    model.train()
    report = sft_module._run_deterministic_greedy_gate(
        model=model,
        split=[{"prompt_text": "fixed prompt"}],
        tokenizer=Tokenizer(),
        max_position_embeddings=64,
        max_rows=1,
        max_new_tokens=8,
        min_new_tokens=4,
        repeats=2,
        label="test",
    )

    assert report["deterministic"] is True
    assert report["generated_token_counts"] == [[8], [8]]
    assert report["generated_token_ids"][0] == report["generated_token_ids"][1]
    assert model.training is True


def test_greedy_numeric_preflight_comparison_rejects_nondeterministic_qat():
    baseline = {
        "generated_token_ids": [[[1, 2, 3]]],
        "deterministic": True,
    }
    qat_on = {
        "generated_token_ids": [[[1, 2, 3]], [[1, 2, 4]]],
        "deterministic": False,
    }

    with pytest.raises(RuntimeError, match="was not deterministic"):
        sft_module._compare_initial_greedy_reports(
            baseline,
            qat_on,
            preflight_cfg={"require_greedy_determinism": True},
            qat_enabled=True,
        )


def test_greedy_numeric_preflight_rejects_deterministic_early_divergence():
    baseline = {
        "generated_token_ids": [[[1, 2, 3, 4, 5, 6, 7, 8]]],
        "deterministic": True,
    }
    qat_on = {
        "generated_token_ids": [
            [[91, 92, 93, 94, 95, 96, 97, 98]],
            [[91, 92, 93, 94, 95, 96, 97, 98]],
        ],
        "deterministic": True,
    }

    with pytest.raises(RuntimeError, match="diverged too early"):
        sft_module._compare_initial_greedy_reports(
            baseline,
            qat_on,
            preflight_cfg={
                "require_greedy_determinism": True,
                "min_baseline_qat_greedy_prefix_tokens": 8,
            },
            qat_enabled=True,
        )


def test_greedy_numeric_preflight_accepts_required_cross_mode_prefix():
    baseline = {
        "generated_token_ids": [[[1, 2, 3, 4, 5, 6, 7, 8, 9]]],
        "deterministic": True,
    }
    qat_sequences = [[1, 2, 3, 4, 5, 6, 7, 8, 99]]
    report = sft_module._compare_initial_greedy_reports(
        baseline,
        {
            "generated_token_ids": [qat_sequences, qat_sequences],
            "deterministic": True,
        },
        preflight_cfg={
            "require_greedy_determinism": True,
            "min_baseline_qat_greedy_prefix_tokens": 8,
        },
        qat_enabled=True,
    )

    assert report["passed"] is True
    assert report["baseline_qat_common_prefix_tokens_by_row"] == [8]
    assert report["baseline_qat_min_common_prefix_tokens"] == 8


def test_greedy_numeric_preflight_requires_prefix_for_every_row():
    baseline_rows = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [11, 12, 13, 14, 15, 16, 17, 18],
    ]
    qat_rows = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [99, 98, 97, 96, 95, 94, 93, 92],
    ]

    with pytest.raises(RuntimeError, match="required_per_row=8"):
        sft_module._compare_initial_greedy_reports(
            {"generated_token_ids": [baseline_rows], "deterministic": True},
            {
                "generated_token_ids": [qat_rows, qat_rows],
                "deterministic": True,
            },
            preflight_cfg={
                "require_greedy_determinism": True,
                "min_baseline_qat_greedy_prefix_tokens": 8,
            },
            qat_enabled=True,
        )


def test_greedy_numeric_preflight_rejects_cross_mode_row_count_mismatch():
    with pytest.raises(RuntimeError, match="row_counts_match=False"):
        sft_module._compare_initial_greedy_reports(
            {
                "generated_token_ids": [
                    [[1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18]]
                ],
                "deterministic": True,
            },
            {
                "generated_token_ids": [
                    [[1, 2, 3, 4, 5, 6, 7, 8]],
                    [[1, 2, 3, 4, 5, 6, 7, 8]],
                ],
                "deterministic": True,
            },
            preflight_cfg={
                "require_greedy_determinism": True,
                "min_baseline_qat_greedy_prefix_tokens": 8,
            },
            qat_enabled=True,
        )


def test_zero_lora_initialization_gate_rejects_nonzero_delta_factors():
    import torch

    class LoraWrapper(torch.nn.Module):
        def __init__(self, *, zero_b: bool):
            super().__init__()
            self.base_layer = torch.nn.Linear(3, 3, bias=False)
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(3, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 3, bias=False)}
            )
            self.active_adapters = ["default"]
            with torch.no_grad():
                self.lora_A["default"].weight.fill_(1.0)
                self.lora_B["default"].weight.fill_(0.0 if zero_b else 1.0)

    exact = sft_module._verify_zero_lora_initialization(
        torch.nn.ModuleDict({"projection": LoraWrapper(zero_b=True)})
    )
    wrong = sft_module._verify_zero_lora_initialization(
        torch.nn.ModuleDict({"projection": LoraWrapper(zero_b=False)})
    )

    assert exact["verified_zero_delta"] is True
    assert exact["wrapper_count"] == exact["adapter_pair_count"] == 1
    assert wrong["verified_zero_delta"] is False
    assert wrong["nonzero_or_invalid_pairs"] == ["projection:default"]


def test_resume_lora_validation_requires_matching_nonzero_adapter(tmp_path):
    import torch

    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = checkpoint / "adapter_model.bin"
    weights.write_bytes(b"resume-adapter")

    class Config:
        r = 16
        lora_alpha = 16
        lora_dropout = 0.0
        target_modules = {"q_proj", "v_proj"}

    class LoraWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(3, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 3, bias=False)}
            )
            with torch.no_grad():
                self.lora_A["default"].weight.fill_(0.5)
                self.lora_B["default"].weight.fill_(0.25)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = LoraWrapper()
            self.peft_config = {"default": Config()}
            self.active_adapters = ["default"]

    sft_module._require_peft_resume_checkpoint(checkpoint)
    report = sft_module._validate_resumed_lora_model(
        Model(),
        expected_config=Config(),
        checkpoint=checkpoint,
    )

    assert report["adapter_sha256"] == hashlib.sha256(b"resume-adapter").hexdigest()
    assert report["adapter_pair_count"] == 1
    assert report["finite_adapter_pairs"] == 1
    assert report["nonzero_adapter_pairs"] == 1


def test_resume_lora_validation_rejects_zero_adapter(tmp_path):
    import torch

    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")

    class Config:
        r = 16
        lora_alpha = 16
        lora_dropout = 0.0
        target_modules = {"q_proj"}

    class LoraWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(3, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 3, bias=False)}
            )
            with torch.no_grad():
                self.lora_A["default"].weight.fill_(1.0)
                self.lora_B["default"].weight.zero_()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = LoraWrapper()
            self.peft_config = {"default": Config()}
            self.active_adapters = ["default"]

    with pytest.raises(RuntimeError, match="finite, nonzero LoRA pairs"):
        sft_module._validate_resumed_lora_model(
            Model(),
            expected_config=Config(),
            checkpoint=checkpoint,
        )


def test_golden_prediction_generation_bounds_input_and_restores_training_mode(
    tmp_path, monkeypatch
):
    import torch

    split_path = tmp_path / "golden.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        split_path,
        [
            {
                "id": "sample-1",
                "response_id": "response-1",
                "completion": '<a2ui>\nroot=Text("expected")\n</a2ui>',
                "messages": [{"role": "user", "content": "Create this UI"}],
                "metadata": {
                    "query_id": "query-1",
                    "ui_id": "ui-1",
                    "intent": "status",
                    "tags": ["compact"],
                },
            }
        ],
    )
    tokenizer_calls: list[dict] = []

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 7
        eos_token_id = 8

        def __call__(self, prompt, **kwargs):
            tokenizer_calls.append({"prompt": prompt, **kwargs})
            return Batch(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

        def decode(self, tokens, skip_special_tokens=True):
            assert tokens.tolist() == [3]
            assert skip_special_tokens is True
            return '<a2ui>\nroot=Text("generated")\n</a2ui>'

    class Adapter:
        def format_example(self, row, tokenizer, include_assistant):
            assert include_assistant is False
            return "formatted prompt"

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 456
            assert kwargs["do_sample"] is False
            assert kwargs["pad_token_id"] == 7
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    monkeypatch.setattr(callbacks_module, "_distributed_context", lambda: (0, 1))
    model = Model()

    count = callbacks_module._generate_predictions_with_model(
        model=model,
        tokenizer=Tokenizer(),
        adapter=Adapter(),
        split_path=split_path,
        output_path=output_path,
        max_rows=1,
        max_input_tokens=123,
        max_new_tokens=456,
    )

    assert count == 1
    assert model.training is True
    assert tokenizer_calls == [
        {
            "prompt": "formatted prompt",
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
    ]
    prediction = json.loads(output_path.read_text(encoding="utf-8"))
    assert prediction["query_id"] == "query-1"
    assert prediction["ui_id"] == "ui-1"
    assert prediction["expected"].startswith("<a2ui>")
    assert prediction["generated_text"].startswith("<a2ui>")


def test_sft_tokenizer_model_alignment_resizes_and_sets_special_ids():
    class Embeddings:
        num_embeddings = 4

    class Config:
        pad_token_id = None
        bos_token_id = None
        eos_token_id = None

    class Model:
        def __init__(self):
            self.config = Config()
            self.generation_config = Config()
            self.embeddings = Embeddings()

        def get_input_embeddings(self):
            return self.embeddings

        def resize_token_embeddings(self, size):
            self.embeddings.num_embeddings = size

    class Tokenizer:
        pad_token_id = 0
        bos_token_id = 2
        eos_token_id = 1

        def __len__(self):
            return 6

    model = Model()
    _align_tokenizer_and_model(Tokenizer(), model)

    assert model.get_input_embeddings().num_embeddings == 6
    assert model.config.pad_token_id == 0
    assert model.generation_config.eos_token_id == [1]


def test_sft_alignment_replaces_out_of_vocab_pad_token_with_eos():
    class Embeddings:
        num_embeddings = 6

    class Config:
        pad_token_id = None
        bos_token_id = None
        eos_token_id = None

    class Model:
        def __init__(self):
            self.config = Config()
            self.generation_config = Config()
            self.embeddings = Embeddings()

        def get_input_embeddings(self):
            return self.embeddings

    class Tokenizer:
        bos_token_id = 2
        eos_token = "<eos>"
        eos_token_id = 1

        def __init__(self):
            self._pad_token = "<bad-pad>"
            self._pad_token_id = 6

        def __len__(self):
            return 6

        @property
        def pad_token(self):
            return self._pad_token

        @pad_token.setter
        def pad_token(self, value):
            self._pad_token = value
            if value == self.eos_token:
                self._pad_token_id = self.eos_token_id

        @property
        def pad_token_id(self):
            return self._pad_token_id

    model = Model()
    tokenizer = Tokenizer()

    _align_tokenizer_and_model(tokenizer, model)

    assert tokenizer.pad_token_id == 1
    assert model.config.pad_token_id == 1


def test_sft_model_vocab_size_prefers_embedding_weight_shape():
    class Weight:
        shape = (7, 16)

    class Embeddings:
        num_embeddings = 4
        weight = Weight()

    class Model:
        def get_input_embeddings(self):
            return Embeddings()

    assert _model_vocab_size(Model()) == 7


def test_sft_model_output_vocab_size_uses_lm_head_weight_shape():
    class Weight:
        shape = (5, 16)

    class LmHead:
        out_features = 4
        weight = Weight()

    class Model:
        lm_head = LmHead()

        def get_output_embeddings(self):
            return None

    assert _model_output_vocab_size(Model()) == 5


def test_sft_effective_max_seq_length_clamps_to_position_limit():
    assert (
        _effective_max_seq_length(configured=8192, max_position_embeddings=4096) == 4096
    )
    assert (
        _effective_max_seq_length(configured=2048, max_position_embeddings=4096) == 2048
    )


def test_sft_precision_falls_back_from_bf16_to_fp16_when_cuda_lacks_bf16():
    old_cuda_available = sft_module._cuda_available
    old_cuda_bf16_supported = sft_module._cuda_bf16_supported
    try:
        sft_module._cuda_available = lambda: True
        sft_module._cuda_bf16_supported = lambda: False

        resolved = _resolve_training_dtype("bfloat16")

        assert resolved == "float16"
        assert _training_precision_flags(resolved) == {"bf16": False, "fp16": True}
    finally:
        sft_module._cuda_available = old_cuda_available
        sft_module._cuda_bf16_supported = old_cuda_bf16_supported


def test_sft_precision_keeps_bf16_when_supported():
    old_cuda_available = sft_module._cuda_available
    old_cuda_bf16_supported = sft_module._cuda_bf16_supported
    try:
        sft_module._cuda_available = lambda: True
        sft_module._cuda_bf16_supported = lambda: True

        resolved = _resolve_training_dtype("bf16")

        assert resolved == "bfloat16"
        assert _training_precision_flags(resolved) == {"bf16": True, "fp16": False}
    finally:
        sft_module._cuda_available = old_cuda_available
        sft_module._cuda_bf16_supported = old_cuda_bf16_supported


def test_sft_requires_cuda_by_default():
    old_cuda_available = sft_module._cuda_available
    try:
        sft_module._cuda_available = lambda: False

        try:
            _enforce_cuda_requirement({}, {})
        except RuntimeError as exc:
            assert "CUDA is not available to PyTorch" in str(exc)
            assert "singularity exec --nv" in str(exc)
        else:
            raise AssertionError("Expected missing CUDA to fail fast")
    finally:
        sft_module._cuda_available = old_cuda_available


def test_sft_can_allow_cpu_smoke_run_explicitly():
    old_cuda_available = sft_module._cuda_available
    try:
        sft_module._cuda_available = lambda: False

        _enforce_cuda_requirement({}, {"allow_cpu": True})
        _enforce_cuda_requirement({"allow_cpu": True}, {})
    finally:
        sft_module._cuda_available = old_cuda_available


def test_sft_model_position_limit_handles_wrapped_model_cycles():
    class Config:
        def __init__(self):
            self.max_position_embeddings = 4096

    class Model:
        def __init__(self):
            self.config = Config()
            self.base_model = self
            self.model = self

    assert _model_position_limit(Model()) == 4096


def test_sft_model_position_limit_ignores_recursive_config_getattr():
    class RecursiveConfig:
        def __init__(self):
            self.text_config = {"max_position_embeddings": 2048}

        def __getattribute__(self, name):
            if name == "max_position_embeddings":
                raise RecursionError("simulated transformers attribute_map recursion")
            return object.__getattribute__(self, name)

    class Model:
        config = RecursiveConfig()

    assert _model_position_limit(Model()) == 2048


def test_sft_preflight_rejects_out_of_vocab_token_id():
    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": [0, 4, 5]}

    try:
        _validate_sft_token_ids(
            dataset={"train": [{"text": "bad"}]},
            tokenizer=Tokenizer(),
            max_seq_length=8,
            vocab_size=5,
            max_rows=0,
        )
    except ValueError as exc:
        assert "outside model vocabulary" in str(exc)
    else:
        raise AssertionError("Expected invalid token id to fail preflight")


def test_tokenized_sft_preflight_rejects_out_of_vocab_token_id():
    try:
        _validate_tokenized_sft_dataset(
            dataset={"train": [{"input_ids": [0, 5], "labels": [-100, 5]}]},
            vocab_size=5,
            max_rows=0,
        )
    except ValueError as exc:
        assert "outside model vocabulary" in str(exc)
    else:
        raise AssertionError("Expected invalid token id to fail tokenized preflight")


def test_tokenized_sft_preflight_rejects_zero_trainable_labels():
    try:
        _validate_tokenized_sft_dataset(
            dataset={"train": [{"input_ids": [0, 1], "labels": [-100, -100]}]},
            vocab_size=5,
            max_rows=0,
        )
    except ValueError as exc:
        assert "zero trainable labels" in str(exc)
    else:
        raise AssertionError(
            "Expected zero trainable labels to fail tokenized preflight"
        )


def test_tokenized_sft_preflight_rejects_out_of_vocab_label_id():
    try:
        _validate_tokenized_sft_dataset(
            dataset={"train": [{"input_ids": [0, 4], "labels": [-100, 5]}]},
            vocab_size=5,
            max_rows=0,
        )
    except ValueError as exc:
        assert "trainable label id outside model vocabulary" in str(exc)
    else:
        raise AssertionError("Expected invalid label id to fail tokenized preflight")


def test_checked_causal_lm_loss_rejects_labels_beyond_logits_vocab():
    import torch

    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    labels = torch.tensor([[0, 1, 5]], dtype=torch.long)

    try:
        _checked_shifted_causal_lm_loss(logits, labels)
    except ValueError as exc:
        assert "labels exceed logits vocabulary" in str(exc)
    else:
        raise AssertionError("Expected checked loss to reject out-of-range labels")


def test_checked_causal_lm_loss_accepts_valid_labels():
    import torch

    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    labels = torch.tensor([[0, 1, 4]], dtype=torch.long)

    loss = _checked_shifted_causal_lm_loss(logits, labels)

    assert float(loss.item()) > 0


def test_sft_training_sample_summary_counts_models():
    dataset = {
        "train": [
            {
                "metadata": {
                    "response_generation": {
                        "provider": "gemini",
                        "model": "gemini-2.5-flash",
                    },
                    "ir_generation": {
                        "provider": "azure_openai",
                        "model": "gpt-5.4-mini",
                    },
                }
            },
            {
                "metadata": {
                    "response_generation": {
                        "provider": "gemini",
                        "model": "gemini-2.5-flash",
                    },
                    "ir_generation": {
                        "provider": "azure_openai",
                        "model": "gpt-5.4-mini",
                    },
                }
            },
            {"metadata": {}},
        ],
        "validation": [
            {
                "metadata": {
                    "response_generation": {
                        "provider": "gemini",
                        "model": "gemini-3-flash",
                    },
                    "source_generation": {
                        "provider": "gemini",
                        "model": "gemini-2.5-pro",
                    },
                }
            }
        ],
    }

    summary = _summarize_training_sample_models(dataset)

    assert summary["train"]["total"] == 3
    assert summary["train"]["response_generation"]["gemini/gemini-2.5-flash"] == 2
    assert summary["train"]["ir_generation"]["azure_openai/gpt-5.4-mini"] == 2
    assert summary["train"]["response_generation"]["unknown/unknown"] == 1
    assert summary["validation"]["ir_generation"]["gemini/gemini-2.5-pro"] == 1


def test_edge_gallery_export_command_for_gemma4_e2b():
    command = build_litert_export_command(
        model_source="runs/gemma4_e2b_ir_lora/merged_hf",
        output_dir="outputs/export/gemma4_e2b_ir_edge_gallery/litertlm",
        export_cfg={
            "command": "litert-torch",
            "externalize_embedder": True,
            "jinja_chat_template_override": "litert-community/gemma-4-E2B-it-litert-lm",
        },
    )
    assert command[:2] == ["litert-torch", "export_hf"]
    assert "--model=runs/gemma4_e2b_ir_lora/merged_hf" in command
    assert "--output_dir=outputs/export/gemma4_e2b_ir_edge_gallery/litertlm" in command
    assert "--externalize_embedder" in command
    assert (
        "--jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm"
        in command
    )


def test_aggregate_scores_includes_overall_and_delta():
    rows = [
        {
            "metrics": {
                "schema_valid_strict": True,
                "content_coverage": 0.8,
                "lint_score": 1.0,
                "dup_rate": 0.1,
            }
        }
    ]
    aggregate = aggregate_scores(
        rows,
        weights={
            "schema_valid_strict": 5.0,
            "content_coverage": 3.0,
            "lint_score": 2.0,
            "dup_rate": -1.0,
        },
        baseline_aggregate={"overall_score": 10.0},
    )
    assert "overall_score" in aggregate
    assert aggregate["baseline_overall_score"] == 10.0
    assert (
        aggregate["overall_score_delta_vs_baseline"]
        == aggregate["overall_score"] - 10.0
    )


def test_export_manifest(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"abc")
    manifest = build_manifest(
        output_dir=tmp_path,
        base_model="base",
        adapter_type="lora",
        training_data_run="dataset_v1",
        prompt_version="v11",
        schema_version="flat-spec",
        runtime="litertlm",
        min_app_version="1.1.0",
        max_input_tokens=8192,
        max_output_tokens=8192,
        files=[artifact],
    )
    path = write_manifest(tmp_path, manifest)
    assert path.exists()
    assert manifest["files"][0]["sha256"]
