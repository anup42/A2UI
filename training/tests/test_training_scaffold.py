from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.data.build_pairs import prepare_dataset
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


def test_model_registry_formats_example():
    adapter = create_adapter({"family": "gemma", "model_id": "google/gemma-4-E2B-it"})
    text = adapter.format_example({"messages": [{"role": "user", "content": "Hello"}]})
    assert "Hello" in text
    assert "gemma" in supported_families()


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
