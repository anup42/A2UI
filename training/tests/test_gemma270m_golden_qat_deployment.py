"""CPU coverage for the opt-in 270M full-QAT deployment."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_gemma270m_full_qat_export as export_fixtures
import test_golden_deployment as fixtures
from ir_training.pipeline import deployment_export, golden_deployment
from ir_training.qat import gemma270m_w8
from ir_training.qat.fake_quant import QATSpec, qat_numeric_contract

options = fixtures.options
export_validator = fixtures.export_validator
completed_run = export_fixtures.completed_run


def qat_options(options):
    fixtures.dump(options.base.model_dir / "config.json",
                  {"model_type": "gemma3_text", "hidden_size": 640, "num_hidden_layers": 18})
    return replace(options, base=replace(options.base, profile="270m", qat=True, prompt_profile="compact_v1"))


def test_full_qat_w8_plan_rejects_other_variants(options):
    options = qat_options(options)
    plan = golden_deployment.build_deployment_plan(options)
    assert list(plan["export"]["variants"]) == ["w8"]
    assert plan["workflow"] == "gemma3_270m_full_qat_w8_deployment_v1"
    assert plan["export"]["require_270m_qat"] is True
    assert "--require-270m-qat" in plan["export"]["prepare_command"]
    with pytest.raises(ValueError, match="exactly W8"):
        golden_deployment.build_deployment_plan(replace(options, variants=("w4",)))


def test_full_qat_golden_requires_enabled_qat_and_tests_exact_w8_on_all_cohorts(options, export_validator, monkeypatch):
    options = qat_options(options)
    ordinary = fixtures.evaluation

    def qat_evaluation(path, count, artifact, **kwargs):
        result = ordinary(path, count, artifact, **kwargs)
        if not kwargs.get("litert"):
            result["qat_applied"] = True
            fixtures.dump(path / "evaluation_result.json", result)
        return result

    calls = []
    with pytest.raises(ValueError, match="must apply QAT"):
        golden_deployment.run_deployment(
            replace(options, base=replace(options.base, output_dir=options.base.output_dir.with_name("failed"))),
            execute=True, command_runner=fixtures.mock_runner(calls),
            pipeline_runner=fixtures.fake_pipeline, writer_factory=fixtures.Writer, gpu_probe=fixtures.inventory)
    assert not any("prepare" in argv or "convert" in argv for argv, _, _ in calls)
    monkeypatch.setattr(fixtures, "evaluation", qat_evaluation)
    state = golden_deployment.run_deployment(
        options, execute=True, command_runner=fixtures.mock_runner(calls),
        pipeline_runner=fixtures.fake_pipeline, writer_factory=fixtures.Writer, gpu_probe=fixtures.inventory)
    assert state["status"] == "complete" and set(state["exports"]) == {"w8"} and len(state["results"]) == 12
    artifact = state["plan"]["export"]["variants"]["w8"]["artifact"]
    runs = [argv for argv, _, _ in calls if "--builtin-gpu" in argv]
    assert {argv[argv.index("--required-rows") + 1] for argv in runs} == {"32", "35", "50"}
    assert all(argv[argv.index("--model") + 1] == artifact and "--require-prepared-contract" in argv for argv in runs)


def test_270m_export_admits_only_selected_checkpoint_with_actual_qat(completed_run):
    import numpy as np
    import yaml
    from safetensors.numpy import save_file

    run, checkpoint, _ = completed_run
    config_path = run / "fit/training_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cfg["qat"]["exclude_modules"] = []
    config_path.write_text(yaml.safe_dump(cfg))
    digest = deployment_export.file_sha256(config_path)
    prep_path = config_path.with_name("preparation_report.json")
    fixtures.dump(prep_path, dict(json.loads(prep_path.read_text()), training_config_sha256=digest))
    meta_path = checkpoint / "training_metadata.json"
    meta = dict(json.loads(meta_path.read_text()), training_config_sha256=digest)
    fixtures.dump(meta_path, meta)
    with pytest.raises(ValueError, match="training evidence"):
        deployment_export.verify_checkpoint_source(config_path, checkpoint, "270m", require_270m_qat=True)
    meta["full_parameter_scope"] = {"scope": "all_model_parameters", "verified": True,
                                    "master_trainable_dtype": "float32", "frozen_parameter_count": 0,
                                    "adapter_parameter_count": 0}
    linears = ["lm_head", *[f"model.layers.{i}.{family}_proj" for i in range(18) for family in
                            ("self_attn.k", "self_attn.v", "self_attn.q", "self_attn.o", "mlp.gate", "mlp.up", "mlp.down")]]
    modules = [*linears, "model.embed_tokens"]
    meta["qat"] = {"enabled": True, "true_fake_quant": True, "wrapped_linear_count": len(linears),
                   "wrapped_linear_names": linears, "wrapped_embedding_names": ["model.embed_tokens"],
                   "wrapped_weight_bits_by_module": dict.fromkeys(modules, 8),
                   "wrapped_effective_lora_count": 0, "wrapped_embedding_count": 1,
                   "wrapped_module_count": len(modules), "detected_lora_adapter_linear_count": 0,
                   "numeric_contract": qat_numeric_contract(QATSpec.from_config(cfg))}
    weights = {module + ".weight": np.ones((1, 1), dtype=np.float32) for module in modules if module != "lm_head"}
    weights["model.norm.weight"] = np.ones((1,), dtype=np.float32)
    records = [{"canonical_name": name, "aliases": [name, "lm_head.weight"] if name == "model.embed_tokens.weight" else [name],
                "trainable_dtype": "float32", "numel": 1} for name in weights]
    meta["full_parameter_scope"].update(parameters=records, unique_parameter_count=len(weights),
                                        named_parameter_count=len(weights) + 1, tied_alias_count=1, trainable_numel=len(weights))
    meta["trainable_parameter_counts"] = {"trainable": len(weights), "total": len(weights)}
    meta["trainable_parameter_names"] = list(weights)
    seed = Path(cfg["model"]["model_source"])
    save_file(weights, seed / "model.safetensors")
    save_file(weights, checkpoint / "model.safetensors")
    prep = json.loads(prep_path.read_text())
    prep["model_files"]["model.safetensors"] = deployment_export.file_sha256(seed / "model.safetensors")
    fixtures.dump(prep_path, prep)
    for item in meta["checkpoint_adapter_files"]:
        if item["path"] == "model.safetensors":
            item.update(sha256=deployment_export.file_sha256(checkpoint / item["path"]),
                        size_bytes=(checkpoint / item["path"]).stat().st_size)
    fixtures.dump(meta_path, meta)
    result = deployment_export.verify_checkpoint_source(config_path, checkpoint, "270m", require_270m_qat=True)
    assert result["gemma3_full_qat_contract"]["workflow"] == gemma270m_w8.PROFILE
    assert result["gemma3_full_qat_contract"]["matrix_count"] == 128
    meta["qat"]["wrapped_linear_names"].remove("model.layers.17.mlp.down_proj")
    fixtures.dump(meta_path, meta)
    with pytest.raises(ValueError, match="every linear and embedding matrix"):
        deployment_export.verify_checkpoint_source(config_path, checkpoint, "270m", require_270m_qat=True)
    assert result["files"]["model.safetensors"] == deployment_export.file_sha256(checkpoint / "model.safetensors")
    with pytest.raises(ValueError, match="rejects retained-scale/QAT"):
        deployment_export.verify_checkpoint_source(config_path, checkpoint, "270m")
    source = {"profile": "270m", "gemma3_full_qat_contract": result["gemma3_full_qat_contract"],
              "qat_aware_training": True, "quantization_export_contract": "dynamic_w8_qat_fresh_graph"}
    deployment_export._verify_gemma3_export_source(source, "w8", expected_profile="270m")
    with pytest.raises(ValueError, match="provenance"):
        deployment_export._verify_gemma3_export_source(source, "w4", expected_profile="270m")
    with pytest.raises(ValueError, match="provenance"):
        deployment_export._verify_gemma3_export_source(source, "w8", expected_profile="1b")
