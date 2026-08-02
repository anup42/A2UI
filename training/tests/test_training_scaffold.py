from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.data.build_pairs import prepare_dataset
from ir_training.data.url_preprocess import preprocess_training_urls, restore_url_placeholders
from ir_training.eval.metrics import aggregate_scores
from ir_training.export.edge_gallery import build_litert_export_command
from ir_training.models.registry import create_adapter, supported_families
from ir_training.export.manifest import build_manifest, write_manifest
from ir_training.common.cuda_env import normalize_cuda_visible_devices
from ir_training.train import callbacks as callbacks_module
from ir_training.train import sft as sft_module
from ir_training.train.callbacks import build_golden_set_eval_callback
from ir_training.train.sft import (
    _CausalLMDataCollator,
    _checked_shifted_causal_lm_loss,
    _align_tokenizer_and_model,
    _enforce_cuda_requirement,
    _effective_max_seq_length,
    _model_output_vocab_size,
    _model_vocab_size,
    _model_position_limit,
    _resolve_training_dtype,
    _tokenize_completion_only_row,
    _summarize_training_sample_models,
    _training_precision_flags,
    _validate_sft_token_ids,
    _validate_tokenized_sft_dataset,
)


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
        "gen": {"provider": "gemini", "model": "gemini-2.5-flash", "prompt_version": "response_v1"},
    }
    spec = {
        "root": "root",
        "state": {},
        "elements": {"root": {"type": "Text", "props": {"text": "Weather"}, "children": []}},
    }
    _write_jsonl(run_dir / "responses.jsonl", [response])
    _write_jsonl(
        run_dir / "genui.jsonl",
        [
            {
                "response_id": "r1",
                "ui_id": "u1",
                "genui_json": spec,
                "gen": {"provider": "azure_openai", "model": "gpt-5.4-mini", "prompt_version": "genui_v1"},
            }
        ],
    )
    out_dir = tmp_path / "prepared"
    manifest = prepare_dataset(
        {
            "run": {"source_run_dir": str(run_dir), "output_dir": str(out_dir), "system_prompt": "Return JSON."},
            "filters": {"require_strict_flat_spec": True, "max_input_chars": 1000, "max_output_chars": 1000},
            "split": {"train": 1, "val": 0, "test": 0, "stratify_by": "intent_bucket"},
        }
    )
    assert manifest["counts"]["accepted"] == 1
    assert manifest["model_counts"]["response_generation"]["gemini/gemini-2.5-flash"] == 1
    assert manifest["model_counts"]["ir_generation"]["azure_openai/gpt-5.4-mini"] == 1
    assert (out_dir / "train.jsonl").exists()
    row = json.loads((out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["metadata"]["response_generation"]["model"] == "gemini-2.5-flash"
    assert row["metadata"]["ir_generation"]["model"] == "gpt-5.4-mini"


def test_prepare_dataset_reads_stage3_folder_and_uses_90_10_split(tmp_path):
    source_dir = tmp_path / "stage3"
    spec = {
        "root": "root",
        "state": {},
        "elements": {"root": {"type": "Text", "props": {"text": "Weather"}, "children": []}},
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
            "filters": {"require_strict_flat_spec": True, "max_input_chars": 1000, "max_output_chars": 1000},
            "split": {"train": 0.9, "val": 0.1, "test": 0.0, "stratify_by": "intent_bucket"},
        }
    )

    assert manifest["counts"]["accepted"] == 10
    assert manifest["counts"]["train"] == 9
    assert manifest["counts"]["val"] == 1
    assert manifest["counts"]["test"] == 0
    assert manifest["counts"]["all"] == 10
    assert len(manifest["source_genui_paths"]) == 2
    assert (out_dir / "all.jsonl").exists()


def test_prepare_dataset_assigns_source_groups_before_target_materialization(tmp_path):
    run_dir = tmp_path / "grouped_run"
    spec = {
        "root": "root",
        "state": {},
        "elements": {"root": {"type": "Text", "props": {"text": "same"}, "children": []}},
    }
    responses = []
    genui = []
    for index in range(12):
        responses.append({
            "response_id": f"r{index}",
            "source_id": f"source-{index // 2}",
            "response_text": f"Source group {index // 2}",
            "intent_bucket": "status" if index % 2 else "weather",
        })
        genui.append({
            "response_id": f"r{index}",
            "source_id": f"source-{index // 2}",
            "ui_id": f"u{index}",
            "genui_json": spec,
        })
    _write_jsonl(run_dir / "responses.jsonl", responses)
    _write_jsonl(run_dir / "genui.jsonl", genui)
    out_dir = tmp_path / "grouped_out"
    manifest = prepare_dataset({
        "run": {"source_run_dir": str(run_dir), "output_dir": str(out_dir), "seed": 7},
        "filters": {"max_input_chars": 1000, "max_output_chars": 1000},
        "split": {"train": 0.5, "val": 0.25, "test": 0.25, "stratify_by": "intent_bucket"},
    })
    assert manifest["split_assignment_stage"] == "source_group_before_target_materialization"
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
                "on": {"press": {"action": "openUrl", "params": {"url": "https://example.org/details"}}},
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
    assert result.genui_json["elements"]["cta"]["on"]["press"]["params"]["url"] == "[ACTION_URL_1]"
    assert restore_url_placeholders(result.genui_json, result.url_map) == spec


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
    assert {entry["kind"] for entry in result.url_map.values()} == {"url", "local_asset"}
    assert restore_url_placeholders(result.response_text, result.url_map) == response
    assert restore_url_placeholders(result.genui_json, result.url_map) == spec


def test_cuda_visible_devices_uses_all_detected_healthy_gpus():
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    old_detector = normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"]
    try:
        normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"] = lambda: ["0", "1", "3"]

        result = normalize_cuda_visible_devices(env)

        assert result == "0,1,3"
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,3"
    finally:
        normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"] = old_detector


def test_cuda_visible_devices_detects_all_when_shell_does_not_set_it():
    env = {}
    old_detector = normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"]
    try:
        normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"] = lambda: ["0", "1", "3"]

        result = normalize_cuda_visible_devices(env)

        assert result == "0,1,3"
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,3"
    finally:
        normalize_cuda_visible_devices.__globals__["_detect_queryable_gpu_indices"] = old_detector


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
        max_seq_length=5,
    )

    assert row["input_ids"] == [10, 10, 20, 20, 20]
    assert row["labels"] == [-100, -100, 20, 20, 20]


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

    collator = _CausalLMDataCollator(Tokenizer(), vocab_size=30, max_position_embeddings=8)
    batch = collator(
        [
            {"input_ids": [10, 20, 20], "attention_mask": [1, 1, 1], "labels": [-100, 20, 20]},
            {"input_ids": [10, 10, 20], "attention_mask": [1, 1, 1], "labels": [-100, -100, 20]},
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
                "on": {"press": {"action": "openUrl", "params": {"url": "https://example.org/details"}}},
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
                "gen": {"provider": "azure_openai", "model": "gpt-5.4-mini", "prompt_version": "genui_v1"},
            }
        ],
    )
    out_dir = tmp_path / "prepared_urls"

    prepare_dataset(
        {
            "run": {"source_run_dir": str(run_dir), "output_dir": str(out_dir), "system_prompt": "Return JSON."},
            "filters": {"require_strict_flat_spec": True, "max_input_chars": 1000, "max_output_chars": 1000},
            "url_preprocessing": {"enabled": True},
            "split": {"train": 1, "val": 0, "test": 0, "stratify_by": "intent_bucket"},
        }
    )

    row = json.loads((out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
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
            "root": {"type": "Image", "props": {"url": "https://example.org/a.png"}, "children": []}
        },
    }
    _write_jsonl(run_dir / "responses.jsonl", [response])
    _write_jsonl(
        run_dir / "genui.jsonl",
        [{"response_id": "r1", "genui_json": spec, "assets": [{"url": "https://example.org/a.png", "path": "C:/private/a.png"}]}],
    )
    out_dir = tmp_path / "prepared_assets"
    prepare_dataset(
        {
            "run": {"source_run_dir": str(run_dir), "output_dir": str(out_dir)},
            "filters": {"max_input_chars": 1000, "max_output_chars": 1000},
            "split": {"train": 1, "val": 0, "test": 0},
        }
    )
    row = json.loads((out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
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
            raise AssertionError("AutoTokenizer should not be used for the explicit fast-tokenizer loader")

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
        return {"overall_score": next(scores), "count": 1}

    def fake_save_best_checkpoint(**kwargs):
        saved_scores.append(float(kwargs["best_info"]["metric_value"]))

    monkeypatch.setattr(callbacks_module, "_generate_predictions_with_model", fake_generate_predictions)
    monkeypatch.setattr(callbacks_module, "evaluate_predictions", fake_evaluate_predictions)
    monkeypatch.setattr(callbacks_module, "_save_best_golden_checkpoint", fake_save_best_checkpoint)
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
    )
    control = object()
    model = object()

    callback.on_epoch_end(None, types.SimpleNamespace(epoch=1.0, global_step=5), control, model=model)
    callback.on_evaluate(None, types.SimpleNamespace(epoch=1.0, global_step=10), control, model=model)
    callback.on_evaluate(None, types.SimpleNamespace(epoch=2.0, global_step=20), control, model=model)

    assert generated_dirs == [output_dir / "step_000000010", output_dir / "step_000000020"]
    assert saved_scores == [20.0]
    assert callback.summary() == {
        "metric": "overall_score",
        "metric_value": 20.0,
        "greater_is_better": True,
        "step": 10,
        "epoch": 1.0,
        "checkpoint_dir": str(best_dir),
    }


def test_golden_prediction_generation_bounds_input_and_restores_training_mode(tmp_path, monkeypatch):
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
            "truncation": True,
            "max_length": 123,
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
    assert model.generation_config.eos_token_id == 1


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
    assert _effective_max_seq_length(configured=8192, max_position_embeddings=4096) == 4096
    assert _effective_max_seq_length(configured=2048, max_position_embeddings=4096) == 2048


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
        raise AssertionError("Expected zero trainable labels to fail tokenized preflight")


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
                    "response_generation": {"provider": "gemini", "model": "gemini-2.5-flash"},
                    "ir_generation": {"provider": "azure_openai", "model": "gpt-5.4-mini"},
                }
            },
            {
                "metadata": {
                    "response_generation": {"provider": "gemini", "model": "gemini-2.5-flash"},
                    "ir_generation": {"provider": "azure_openai", "model": "gpt-5.4-mini"},
                }
            },
            {"metadata": {}},
        ],
        "validation": [
            {
                "metadata": {
                    "response_generation": {"provider": "gemini", "model": "gemini-3-flash"},
                    "source_generation": {"provider": "gemini", "model": "gemini-2.5-pro"},
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
    assert "--jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm" in command


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
        weights={"schema_valid_strict": 5.0, "content_coverage": 3.0, "lint_score": 2.0, "dup_rate": -1.0},
        baseline_aggregate={"overall_score": 10.0},
    )
    assert "overall_score" in aggregate
    assert aggregate["baseline_overall_score"] == 10.0
    assert aggregate["overall_score_delta_vs_baseline"] == aggregate["overall_score"] - 10.0


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
