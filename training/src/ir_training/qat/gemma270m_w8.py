"""Selected full-model W8 QAT checkpoint checks shared by both 270M exporters."""
from __future__ import annotations

import json
from pathlib import Path

PROFILE = "gemma3_270m_wi8_afp32_full_finetune"
RECIPE = "dynamic_wi8_afp32"


def validate_270m_model(model_dir: Path) -> None:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected a JSON object: {model_dir / 'config.json'}")
    text = config.get("text_config", config)
    if (config.get("model_type") not in {"gemma3_text", "gemma3"}
            or not isinstance(text, dict)
            or (text.get("hidden_size"), text.get("num_hidden_layers")) != (640, 18)):
        raise ValueError(f"Expected a local Gemma 3 270M checkpoint: {model_dir}")


def validate_selected_270m_qat(config: dict, metadata: dict, checkpoint: Path) -> None:
    qat = config.get("qat") or {}
    expected = {
        "enabled": True, "profile": PROFILE, "final_quantization": RECIPE,
        "weight_bits": 8, "activation_bits": 32,
        "quantizer": "ste_ai_edge", "quantize_embeddings": True,
    }
    if (config["training"]["method"] != "full_finetune_qat"
            or any(type(qat.get(key)) is not type(value) or qat[key] != value for key, value in expected.items())
            or metadata.get("checkpoint_kind") != "full_model"
            or metadata.get("checkpoint_role") != "best_golden"
            or type(metadata.get("checkpoint_step")) is not int or metadata["checkpoint_step"] <= 0):
        raise ValueError("Expected the Golden-selected full-model W8 QAT checkpoint.")
    validate_270m_model(checkpoint)


def deployment_qat_proof(config: dict, metadata: dict, checkpoint: Path) -> dict:
    """Bind actual training-time fake quantization and full-parameter scope."""
    from ir_training.qat.fake_quant import QATSpec, qat_numeric_contract
    from ir_training.qat.full_model_contract import _verify_scope_against_seed
    from ir_training.qat.gemma3_full_qat_contract import (
        _checkpoint_tensors,
        _verify_coverage,
    )

    validate_selected_270m_qat(config, metadata, checkpoint)
    validate_270m_model(Path(config["model"]["model_source"]))
    scope = metadata.get("full_parameter_scope") or {}
    qat = metadata.get("qat") or {}
    spec = QATSpec.from_config(config)
    if (scope.get("verified") is not True or scope.get("scope") != "all_model_parameters"
            or scope.get("frozen_parameter_count") != 0 or scope.get("adapter_parameter_count") != 0
            or scope.get("master_trainable_dtype") != "float32"
            or not spec.weight_symmetric or not spec.weight_per_channel or spec.weight_axis != 0
            or spec.group_size is not None or spec.scale_mode != "dynamic" or spec.exclude_modules
            or spec.modules_to_not_convert or spec.module_quant_configs or spec.module_group_sizes
            or qat.get("enabled") is not True or qat.get("true_fake_quant") is not True
            or qat.get("numeric_contract") != qat_numeric_contract(spec)):
        raise ValueError("Gemma 3 270M checkpoint lacks full-parameter W8 QAT training evidence")

    tied = {"model.embed_tokens.weight", "lm_head.weight"}
    if [set(item["aliases"]) for item in scope.get("parameters", []) if len(item["aliases"]) > 1] != [tied]:
        raise ValueError("Gemma 3 270M requires the bound input/output embedding alias")
    inventories = []
    for directory in (Path(config["model"]["model_source"]), checkpoint):
        tensors = _checkpoint_tensors(directory)
        stored = tied.intersection(tensors)
        if len(stored) != 1:
            raise ValueError(f"Expected one stored Gemma 3 270M embedding tensor: {directory}")
        shapes = {name: tensor["shape"] for name, tensor in tensors.items()}
        shapes.update(dict.fromkeys(tied, shapes[stored.pop()]))
        if directory == checkpoint and any(t["dtype"] != "F32" for t in tensors.values()):
            raise ValueError("Gemma 3 270M checkpoint must serialize every trainable weight in FP32")
        inventories.append(shapes)
    seed_shapes, checkpoint_shapes = inventories
    if checkpoint_shapes != seed_shapes:
        raise ValueError("Gemma 3 270M checkpoint tensor shapes differ from the dense seed")
    projections = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    expected = tied | {f"model.layers.{index}.{name}.weight" for index in range(18) for name in projections}
    if {name for name, shape in seed_shapes.items() if len(shape) == 2} != expected:
        raise ValueError("Gemma 3 270M requires all 18 layers of projection matrices and the tied embedding")
    _verify_scope_against_seed(scope, seed_shapes, metadata)
    coverage = _verify_coverage(qat, seed_shapes)
    return {"verified": True, "workflow": PROFILE, "quantization_recipe": RECIPE,
            "matrix_count": coverage["matrix_count"],
            "all_parameters_trainable": True, "embedding_included": True,
            "official_graph_transplant": False}
