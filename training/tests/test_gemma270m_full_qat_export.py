from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml
from ir_training.pipeline.deployment_export import file_sha256
from test_deployment_export import package, write_json


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "run"
    fit = run / "fit"
    checkpoint = fit / "training" / "best_golden_checkpoint"
    model_config = {"model_type": "gemma3_text", "hidden_size": 640, "num_hidden_layers": 18}
    seed = tmp_path / "seed"
    for directory in (checkpoint, seed):
        write_json(directory / "config.json", model_config)
        write_json(directory / "tokenizer.json", {"fixture": "tokenizer"})
        write_json(directory / "tokenizer_config.json", {"fixture": "tokenizer configuration"})
        (directory / "model.safetensors").write_bytes(directory.name.encode())
    config_path = fit / "training_config.yaml"
    config_path.write_text(yaml.safe_dump({
        "run": {"dataset_dir": str(tmp_path / "data")},
        "model": {"family": "gemma", "model_id": "google/gemma-3-270m-it", "model_source": str(seed)},
        "training": {"method": "full_finetune_qat"},
        "golden_eval": {"max_input_tokens": 5120, "max_new_tokens": 2048},
        "qat": {
            "enabled": True,
            "profile": "gemma3_270m_wi8_afp32_full_finetune",
            "final_quantization": "dynamic_wi8_afp32",
            "weight_bits": 8,
            "activation_bits": 32,
            "quantizer": "ste_ai_edge",
            "quantize_embeddings": True,
        },
    }), encoding="utf-8")
    preparation_path = fit / "preparation_report.json"
    config_sha = file_sha256(config_path)
    write_json(preparation_path, {
        "training_config_sha256": config_sha,
        "model_files": {path.name: file_sha256(path) for path in seed.iterdir()},
    })
    selection_metric = "unique_source_generation_reward_v5_4_avg"
    metadata_path = checkpoint / "training_metadata.json"
    write_json(metadata_path, {
        "checkpoint_kind": "full_model",
        "checkpoint_role": "best_golden",
        "checkpoint_step": 1000,
        "training_config_sha256": config_sha,
        "checkpoint_adapter_files": [
            {"path": path.name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(checkpoint.iterdir())
        ],
        "best_golden_eval": {
            "step": 1000,
            "metric": selection_metric,
            "metric_value": 81.0,
            "checkpoint_dir": str(checkpoint),
            "evaluation_dir": str(checkpoint.parent / "golden_eval" / "step_000001000"),
        },
    })

    def stage(*paths):
        return {"files": {str(path): file_sha256(path) for path in paths}}

    completed = {
        "configure": stage(config_path, preparation_path),
        "training": stage(*checkpoint.iterdir()),
    }
    evaluations = {}
    for cohort, size in (("golden32", 32), ("golden35", 35), ("bixby50", 50)):
        name = f"best_{cohort}"
        result_path = run / "evaluations" / name / "attempt_001" / "evaluation_result.json"
        result = {
            "checkpoint": str(checkpoint),
            "checkpoint_kind": "merged",
            "qat_applied": True,
            "row_count": size,
            "aggregate": {selection_metric: 81.0, "generation_reward_v5_4_avg": 81.0},
        }
        write_json(result_path, result)
        evaluations[name] = result
        completed[name] = stage(result_path)

    scorecard_path = run / "evaluation_scorecard.json"
    write_json(scorecard_path, {
        "schema_version": 1,
        "status": "complete",
        "profile": "270m",
        "evaluations": evaluations,
        "golden32_unique_sources": 31,
        "golden35_evaluated": True,
        "golden35_used_for_selection": False,
        "bixby50_evaluated": True,
        "bixby50_used_for_selection": False,
        "exports_performed": False,
    })
    completed["scorecard"] = stage(scorecard_path)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "plan": {
            "schema_version": 1,
            "workflow": "shared_prompt_golden_bixby_training_v2",
            "options": {"profile": "270m", "qat": True},
        },
        "completed": completed,
    }
    write_json(run / "pipeline_manifest.json", manifest)
    return run, checkpoint, manifest


@pytest.fixture
def converter(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_gemma3_270m_full_qat_w8.py"
    spec = importlib.util.spec_from_file_location("export_gemma3_270m_full_qat_w8", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests = []

    def convert(**kwargs):
        requests.append(kwargs)
        target = Path(kwargs["output_dir"]) / "gemma3_fixture.litertlm"
        target.write_bytes(b"mock converted package")

    monkeypatch.setitem(sys.modules, "litert_torch.generative.export_hf.export", types.SimpleNamespace(export=convert))
    monkeypatch.setattr(module, "probe_exporter", lambda **kwargs: {"status": "passed"})
    monkeypatch.setattr(module, "inspect_litertlm", lambda *args, **kwargs: package(["INT8"], cast=True))
    return module, requests


def test_exports_only_selected_full_qat_checkpoint(completed_run, converter, tmp_path):
    run, checkpoint, _ = completed_run
    module, requests = converter
    output = tmp_path / "export"

    report = module.export_w8(run, output, cache_length=8192)

    assert len(requests) == 1
    assert requests[0]["model"] == str(checkpoint)
    assert requests[0]["quantization_recipe"] == "dynamic_wi8_afp32"
    assert requests[0]["cache_length"] == 8192
    assert requests[0]["experimental_use_fp16"] is False
    assert requests[0]["bundle_litert_lm"] is True
    artifact = output / "model.litertlm"
    assert artifact.read_bytes() == b"mock converted package"
    assert report["artifact_sha256"] == file_sha256(artifact)
    assert report["checkpoint_step"] == 1000
    assert report["physical_weight_precision"]["weight_type_counts"] == {"INT8": 1}
    assert (output / "w8_export_report.json").is_file()


@pytest.mark.parametrize("problem", ["mutated_checkpoint", "incomplete_evaluation"])
def test_export_rejects_invalid_training_evidence(completed_run, converter, tmp_path, problem):
    run, checkpoint, manifest = completed_run
    module, requests = converter
    output = tmp_path / "export"
    if problem == "mutated_checkpoint":
        (checkpoint / "model.safetensors").write_bytes(b"different weights")
        expected_error = "(inventory changed|changed.*model.safetensors|model.safetensors.*changed)"
    else:
        del manifest["completed"]["best_bixby50"]
        write_json(run / "pipeline_manifest.json", manifest)
        expected_error = "bixby50"

    with pytest.raises(ValueError, match=expected_error):
        module.export_w8(run, output)

    assert not requests
    assert not output.exists()


def test_rejects_wrong_physical_precision_without_publishing(completed_run, converter, monkeypatch, tmp_path):
    run, _, _ = completed_run
    module, requests = converter
    output = tmp_path / "export"
    monkeypatch.setattr(module, "inspect_litertlm", lambda *args, **kwargs: package(["FLOAT32"]))

    with pytest.raises(ValueError, match="physical weights"):
        module.export_w8(run, output)

    assert len(requests) == 1
    assert not output.exists()
