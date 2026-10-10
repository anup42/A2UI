from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
from ir_training.pipeline import deployment_export as export
from ir_training.qat import gemma3_full_qat_contract as contract
from ir_training.qat.fake_quant import QATSpec, prepare_qat_model, qat_numeric_contract


def model_config():
    return {"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"],
            "hidden_size": 1152, "num_hidden_layers": 26, "vocab_size": 262144,
            "intermediate_size": 6912, "num_attention_heads": 4,
            "num_key_value_heads": 1, "head_dim": 256}


def config():
    return {"run": {"purpose": contract.WORKFLOW},
            "model": {"family": "gemma", "model_id": contract.MODEL_ID},
            "training": {"method": "full_finetune_qat", "backward_preflight": True},
            "qat": {"enabled": True, "weight_bits": 8, "activation_bits": 32,
                    "quantizer": "ste_ai_edge", "eps": 1e-9, "quantize_embeddings": True,
                    "exclude_modules": [], "final_quantization": "dynamic_wi8_afp32",
                    "final_runtime_validation_required": True}}


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def fixture(monkeypatch, tmp_path):
    """Small serialized weights exercise gates; architecture is separately tested."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    shapes = {"model.embed_tokens.weight": [3, 2], "lm_head.weight": [3, 2],
              "model.norm.weight": [2], "model.layers.0.self_attn.q_proj.weight": [2, 2]}
    monkeypatch.setattr(contract, "expected_parameter_shapes", lambda raw: shapes)
    cfg = config()
    cfg["model"]["model_source"] = str(tmp_path / "base")
    checkpoint = tmp_path / "checkpoint"
    write(tmp_path / "base/config.json", model_config())
    write(checkpoint / "config.json", model_config())
    save_file({name: torch.ones(shape) for name, shape in shapes.items() if name != "lm_head.weight"},
              str(checkpoint / "model.safetensors"))
    parameters = [
        {"canonical_name": "model.embed_tokens.weight", "aliases": ["model.embed_tokens.weight", "lm_head.weight"],
         "numel": 6, "trainable_dtype": "float32"},
        {"canonical_name": "model.norm.weight", "aliases": ["model.norm.weight"], "numel": 2, "trainable_dtype": "float32"},
        {"canonical_name": "model.layers.0.self_attn.q_proj.weight", "aliases": ["model.layers.0.self_attn.q_proj.weight"],
         "numel": 4, "trainable_dtype": "float32"},
    ]
    linears = ["lm_head", "model.layers.0.self_attn.q_proj"]
    embeddings = ["model.embed_tokens"]
    spec = QATSpec.from_config(cfg)
    qat = {"enabled": True, "true_fake_quant": True, "wrapped_module_count": 3,
           "wrapped_linear_count": 2, "wrapped_embedding_count": 1,
           "wrapped_linear_names": linears, "wrapped_embedding_names": embeddings,
           "wrapped_weight_bits_by_module": {name: 8 for name in linears + embeddings},
           "wrapped_effective_lora_count": 0, "detected_lora_adapter_linear_count": 0,
           "numeric_contract": qat_numeric_contract(spec), "spec": json.loads(json.dumps(spec.to_dict()))}
    metadata = {"checkpoint_kind": "full_model", "lora": {},
                "full_finetune_precision": {"policy": "fp32_parameters_v1", "parameter_dtype": "float32", "compute_dtype": "bfloat16"},
                "full_parameter_scope": {"verified": True, "scope": "all_model_parameters", "master_trainable_dtype": "float32",
                    "frozen_parameter_count": 0, "adapter_parameter_count": 0, "parameters": parameters,
                    "unique_parameter_count": 3, "named_parameter_count": 4, "tied_alias_count": 1, "trainable_numel": 12},
                "trainable_parameter_names": [item["canonical_name"] for item in parameters],
                "trainable_parameter_counts": {"trainable": 12, "total": 12},
                "backward_preflight": {"status": "passed"}, "qat": qat,
                "gemma3_qat_coverage": contract._verify_coverage(qat, shapes)}
    return cfg, metadata, checkpoint


def test_one_b_export_defaults_w8_without_changing_other_profiles(tmp_path):
    assert tuple(export.deployment_variants("1b")) == ("w8",)
    for profile in ("270m", "e2b"):
        assert tuple(export.deployment_variants(profile)) == ("w32", "w16", "w8", "w4")
    plan = export.build_deployment_export_plan(profile="1b", training_config_path=tmp_path / "config.yaml",
        checkpoint_dir=tmp_path / "checkpoint", output_dir=tmp_path / "export",
        training_python=sys.executable, exporter_python=sys.executable)
    assert tuple(plan["variants"]) == ("w8",)
    assert "--full-parameter-export" not in plan["probe_command"]
    kwargs = export.export_kwargs("1b", "w8", tmp_path, tmp_path / "out", 8192)
    assert kwargs["quantization_recipe"] == "dynamic_wi8_afp32"
    assert kwargs["enable_gpu_dynamic_cache"] and kwargs["enable_gpu_dynamic_prefill"]
    assert kwargs["externalize_embedder"] is False


@pytest.mark.parametrize("variants", [("w4",), ("w32",), ("w16",), ("w8", "w4"), ("w248",)])
def test_one_b_cannot_silently_export_an_untrained_recipe(variants):
    with pytest.raises(ValueError, match="exactly W8"):
        export.deployment_variants("1b", variants)


@pytest.mark.parametrize("field,value", [("model_type", "gemma4_text"), ("hidden_size", 640),
    ("num_hidden_layers", 18), ("vocab_size", 32000), ("tie_word_embeddings", False),
    ("quantization_config", {"load_in_4bit": True}), ("vision_config", {"model_type": "siglip"})])
def test_model_identity_is_architecture_not_filename(field, value):
    raw = model_config()
    raw[field] = value
    with pytest.raises(ValueError, match="Gemma 3 1B"):
        contract.validate_gemma3_1b_model_config(raw)


@pytest.mark.parametrize("field,value", [("weight_bits", 4), ("activation_bits", 8),
    ("exclude_modules", ["lm_head"]), ("quantize_embeddings", False), ("group_size", 32),
    ("module_quant_configs", {"q_proj": 4}), ("effective_lora_only", True),
    ("scale_mode", "retained_mobile"), ("weight_axis", 1), ("eps", 1e-8)])
def test_wrong_qat_recipe_is_rejected(field, value):
    cfg = config()
    cfg["qat"][field] = value
    with pytest.raises(ValueError, match="W8/AFP32"):
        contract.validate_gemma3_full_qat_config(cfg)


def test_checkpoint_preserves_complete_dense_weights_with_only_hf_tied_alias_omitted(monkeypatch, tmp_path):
    cfg, metadata, checkpoint = fixture(monkeypatch, tmp_path)
    result = contract.verify_gemma3_full_qat_checkpoint(cfg, metadata, checkpoint)
    assert result["verified"] and result["embedding_included"]
    assert result["serialized_parameter_count"] == 3
    assert result["official_graph_transplant"] is False
    assert result["physical_serialization_audit"] is False


@pytest.mark.parametrize("mutation", ["backward", "frozen", "embedding", "norm", "low_precision", "coverage", "numeric"])
def test_checkpoint_provenance_or_weights_cannot_be_lost(monkeypatch, tmp_path, mutation):
    cfg, metadata, checkpoint = fixture(monkeypatch, tmp_path)
    if mutation == "backward":
        metadata["backward_preflight"]["status"] = "skipped"
    elif mutation == "frozen":
        metadata["full_parameter_scope"]["frozen_parameter_count"] = 1
    elif mutation == "coverage":
        metadata["qat"]["wrapped_linear_names"].remove("lm_head")
    elif mutation == "numeric":
        metadata["qat"]["numeric_contract"] = {}
    else:
        from safetensors.torch import load_file, save_file
        weights = load_file(str(checkpoint / "model.safetensors"))
        if mutation == "embedding":
            del weights["model.embed_tokens.weight"]
        elif mutation == "norm":
            del weights["model.norm.weight"]
        else:
            weights = {name: tensor.half() for name, tensor in weights.items()}
        save_file(weights, str(checkpoint / "model.safetensors"))
    with pytest.raises(ValueError):
        contract.verify_gemma3_full_qat_checkpoint(cfg, metadata, checkpoint)


def test_expected_inventory_matches_real_transformers_meta_architecture():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    raw = model_config()
    with torch.device("meta"):
        model = transformers.Gemma3ForCausalLM(transformers.Gemma3TextConfig(**{key: value for key, value in raw.items()
                                                                             if key not in {"model_type", "architectures"}}))
    actual = {name: list(parameter.shape) for name, parameter in model.named_parameters(remove_duplicate=False)}
    assert actual == contract.expected_parameter_shapes(raw)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    controller = prepare_qat_model(model, config())
    try:
        summary = controller.summary()
        assert contract.verify_gemma3_full_qat_scope(model, summary)["verified"] is True
        model.model.norm.weight.requires_grad_(False)
        with pytest.raises(ValueError, match="every original parameter"):
            contract.verify_gemma3_full_qat_scope(model, summary)
    finally:
        controller.restore()


def test_contract_is_separate_from_e2b_export_bridge(monkeypatch, tmp_path):
    cfg, metadata, checkpoint = fixture(monkeypatch, tmp_path)
    proof = contract.verify_gemma3_full_qat_checkpoint(cfg, metadata, checkpoint)
    source = {"profile": "1b", "gemma3_full_qat_contract": proof, "full_qat_contract": None,
              "qat_aware_training": True, "quantization_export_contract": "dynamic_w8_qat_fresh_graph"}
    export._verify_gemma3_export_source(source, "w8")
    damaged = copy.deepcopy(source)
    damaged["full_qat_contract"] = {"verified": True}
    with pytest.raises(ValueError, match="provenance"):
        export._verify_gemma3_export_source(damaged, "w8")


def test_shared_source_verifier_admits_only_bound_full_qat_1b(monkeypatch, tmp_path):
    import yaml

    cfg, metadata, checkpoint = fixture(monkeypatch, tmp_path)
    config_path = tmp_path / "fit/training_config.yaml"
    config_path.parent.mkdir()
    cfg["run"]["dataset_dir"] = str(tmp_path / "prepared")
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    metadata.update(training_config_sha256=export.file_sha256(config_path), checkpoint_step=100,
                    checkpoint_adapter_files=[{"path": item.name, "sha256": export.file_sha256(item),
                                               "size_bytes": item.stat().st_size} for item in checkpoint.iterdir()])
    write(checkpoint / "training_metadata.json", metadata)
    write(config_path.parent / "preparation_report.json", {"training_config_sha256": export.file_sha256(config_path),
        "model_files": {"config.json": export.file_sha256(tmp_path / "base/config.json")}})
    result = export.verify_checkpoint_source(config_path, checkpoint, "1b")
    assert result["full_qat_contract"] is None
    assert result["gemma3_full_qat_contract"]["verified"] is True
    assert result["checkpoint_kind"] == "full_model"
    metadata["checkpoint_kind"] = "lora_adapter"
    write(checkpoint / "training_metadata.json", metadata)
    with pytest.raises(ValueError, match="FP32/backward"):
        export.verify_checkpoint_source(config_path, checkpoint, "1b")


def test_shared_materialization_preserves_dense_checkpoint_for_gemma3(monkeypatch, tmp_path):
    from ir_training.eval import prepared_contract
    from ir_training.models import registry

    cfg, metadata, checkpoint = fixture(monkeypatch, tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: {}\n", encoding="utf-8")
    cfg["run"]["dataset_dir"] = str(tmp_path / "prepared")
    proof = contract.verify_gemma3_full_qat_checkpoint(cfg, metadata, checkpoint)
    binding = {"training_config": str(config_path), "config": cfg, "metadata": {"checkpoint_step": 100},
               "checkpoint_kind": "full_model", "files": {item.name: export.file_sha256(item) for item in checkpoint.iterdir()},
               "full_qat_contract": None, "gemma3_full_qat_contract": proof}
    write(checkpoint / "training_metadata.json", metadata)
    monkeypatch.setattr(export, "verify_checkpoint_source", lambda *args, **kwargs: binding)

    class Tokenizer:
        def save_pretrained(self, path):
            write(Path(path) / "tokenizer.json", {"same": True})

    import types
    monkeypatch.setattr(registry, "create_adapter", lambda cfg: types.SimpleNamespace(load_tokenizer=Tokenizer))
    monkeypatch.setattr(prepared_contract, "checked_preparation_manifest", lambda *args: {"tokenizer": {}})
    monkeypatch.setattr(prepared_contract, "verify_loaded_evaluation_tokenizer", lambda *args: None)
    output = tmp_path / "export_source"
    result = export.prepare_deployment_checkpoint(profile="1b", training_config=config_path, checkpoint=checkpoint, output_dir=output)
    assert result["gemma3_full_qat_contract"] == proof
    assert result["full_qat_contract"] is None
    assert result["qat_aware_training"] is True
    assert result["quantization_export_contract"] == "dynamic_w8_qat_fresh_graph"
    assert (output / "model.safetensors").read_bytes() == (checkpoint / "model.safetensors").read_bytes()
    export._verify_gemma3_export_source(result, "w8")
