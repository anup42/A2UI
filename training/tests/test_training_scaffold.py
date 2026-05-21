from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.data.build_pairs import prepare_dataset
from ir_training.data.url_preprocess import preprocess_training_urls, restore_url_placeholders
from ir_training.eval.metrics import aggregate_scores
from ir_training.export.edge_gallery import build_litert_export_command
from ir_training.models.registry import create_adapter, supported_families
from ir_training.export.manifest import build_manifest, write_manifest


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
    }
    spec = {
        "root": "root",
        "state": {},
        "elements": {"root": {"type": "Text", "props": {"text": "Weather"}, "children": []}},
    }
    _write_jsonl(run_dir / "responses.jsonl", [response])
    _write_jsonl(run_dir / "genui.jsonl", [{"response_id": "r1", "ui_id": "u1", "genui_json": spec}])
    out_dir = tmp_path / "prepared"
    manifest = prepare_dataset(
        {
            "run": {"source_run_dir": str(run_dir), "output_dir": str(out_dir), "system_prompt": "Return JSON."},
            "filters": {"require_strict_flat_spec": True, "max_input_chars": 1000, "max_output_chars": 1000},
            "split": {"train": 1, "val": 0, "test": 0, "stratify_by": "intent_bucket"},
        }
    )
    assert manifest["counts"]["accepted"] == 1
    assert (out_dir / "train.jsonl").exists()


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
    _write_jsonl(run_dir / "genui.jsonl", [{"response_id": "r1", "ui_id": "u1", "genui_json": spec}])
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
