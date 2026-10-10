"""Narrow, fail-closed Gemma 3 1B full-QAT W8 deployment contract.

This is a fresh public LiteRT graph, not an official-graph transplant. Tensor
inventories are checked without loading multi-gigabyte checkpoint weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ir_training.qat.fake_quant import QATSpec, qat_numeric_contract

WORKFLOW = "gemma3_1b_full_qat_v1"
MODEL_ID = "google/gemma-3-1b-it"
_DIMENSIONS = {
    "hidden_size": 1152, "num_hidden_layers": 26, "vocab_size": 262144,
    "intermediate_size": 6912, "num_attention_heads": 4,
    "num_key_value_heads": 1, "head_dim": 256,
}
_TIED = {"model.embed_tokens.weight", "lm_head.weight"}


def is_gemma3_full_qat(config: dict[str, Any]) -> bool:
    return (config.get("run") or {}).get("purpose") == WORKFLOW


def validate_gemma3_1b_model_config(config: dict[str, Any]) -> None:
    if (config.get("model_type") != "gemma3_text"
            or any(type(config.get(key)) is not int or config[key] != value
                   for key, value in _DIMENSIONS.items())
            or config.get("architectures") not in (None, ["Gemma3ForCausalLM"])
            or config.get("tie_word_embeddings", True) is not True
            or config.get("attention_bias", False) is not False
            or config.get("quantization_config") or config.get("vision_config")):
        raise ValueError("Expected the dense text-only Gemma 3 1B architecture with tied embeddings")


def validate_local_gemma3_1b_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Gemma 3 model config must be an object")  # noqa: TRY004 - invalid file content
    validate_gemma3_1b_model_config(config)
    return config


def expected_parameter_shapes(config: dict[str, Any]) -> dict[str, list[int]]:
    """The named Gemma3ForCausalLM parameter contract, including tied aliases."""
    validate_gemma3_1b_model_config(config)
    hidden, intermediate, head = (config[key] for key in ("hidden_size", "intermediate_size", "head_dim"))
    qwidth = config["num_attention_heads"] * head
    kvwidth = config["num_key_value_heads"] * head
    shapes = {"model.embed_tokens.weight": [config["vocab_size"], hidden],
              "lm_head.weight": [config["vocab_size"], hidden], "model.norm.weight": [hidden]}
    for index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{index}."
        shapes.update({prefix + name + ".weight": shape for name, shape in {
            "self_attn.q_proj": [qwidth, hidden], "self_attn.k_proj": [kvwidth, hidden],
            "self_attn.v_proj": [kvwidth, hidden], "self_attn.o_proj": [hidden, qwidth],
            "self_attn.q_norm": [head], "self_attn.k_norm": [head],
            "mlp.gate_proj": [intermediate, hidden], "mlp.up_proj": [intermediate, hidden],
            "mlp.down_proj": [hidden, intermediate], "input_layernorm": [hidden],
            "post_attention_layernorm": [hidden], "pre_feedforward_layernorm": [hidden],
            "post_feedforward_layernorm": [hidden],
        }.items()})
    return shapes


def validate_gemma3_full_qat_config(config: dict[str, Any]) -> None:
    model, training, qat = (config.get(key) or {} for key in ("model", "training", "qat"))
    if (not is_gemma3_full_qat(config) or model.get("family") != "gemma"
            or model.get("model_id") != MODEL_ID or training.get("method") != "full_finetune_qat"
            or config.get("lora") or config.get("qat_mtp") or model.get("load_in_4bit", False)
            or model.get("mobile_training_seed_manifest") or qat.get("enabled") is not True
            or training.get("backward_preflight") is not True
            or qat.get("final_quantization") != "dynamic_wi8_afp32"
            or qat.get("final_runtime_validation_required") is not True):
        raise ValueError("Gemma 3 1B full-QAT requires its explicit full-model W8 workflow and backward/runtime gates")
    spec = QATSpec.from_config(config)
    if (spec.weight_bits != 8 or spec.activation_bits != 32 or not spec.weight_symmetric
            or not spec.weight_per_channel or spec.weight_axis != 0 or spec.group_size is not None
            or spec.quantizer != "ste_ai_edge" or spec.scale_mode != "dynamic"
            or not spec.quantize_embeddings or spec.exclude_modules or spec.module_quant_configs
            or spec.module_group_sizes or spec.modules_to_not_convert or spec.effective_lora_only
            or spec.mobile_qparams_contract or spec.fixed_scale_required or spec.fixed_activation_scale_required
            or spec.simulate_frozen_activations or spec.require_lora_trainable_scope
            or spec.eps != 1e-9 or spec.ste_gradient not in {"identity", "clipped"}):
        raise ValueError("Gemma 3 1B export requires complete dynamic per-row W8/AFP32 ste_ai_edge QAT")


def _verify_coverage(summary: dict[str, Any], shapes: dict[str, list[int]]) -> dict[str, Any]:
    matrices = {name.removesuffix(".weight") for name, shape in shapes.items() if len(shape) == 2}
    embeddings = {"model.embed_tokens"}
    linears = matrices - embeddings
    actual_linear = summary.get("wrapped_linear_names")
    actual_embedding = summary.get("wrapped_embedding_names")
    if (summary.get("enabled") is not True or summary.get("true_fake_quant") is not True
            or not isinstance(actual_linear, list) or sorted(actual_linear) != sorted(linears)
            or not isinstance(actual_embedding, list) or sorted(actual_embedding) != sorted(embeddings)
            or summary.get("wrapped_module_count") != len(matrices)
            or summary.get("wrapped_linear_count") != len(linears)
            or summary.get("wrapped_embedding_count") != len(embeddings)
            or summary.get("wrapped_weight_bits_by_module") != {name: 8 for name in matrices}
            or summary.get("wrapped_effective_lora_count") != 0
            or summary.get("detected_lora_adapter_linear_count") != 0):
        raise ValueError("Gemma 3 full-QAT does not cover every linear and embedding matrix exactly")
    return {"verified": True, "state_shapes": shapes, "linear_modules": sorted(linears),
            "embedding_modules": sorted(embeddings), "matrix_count": len(matrices),
            "named_parameter_count": len(shapes), "tied_parameter_aliases": [sorted(_TIED)]}


def verify_gemma3_full_qat_scope(model: Any, qat_summary: dict[str, Any]) -> dict[str, Any]:
    """Check actual registered parameters/aliases without allocating model copies."""
    import torch

    shapes = expected_parameter_shapes(model.config.to_dict())
    parameters = dict(model.named_parameters(remove_duplicate=False))
    if ({name: list(value.shape) for name, value in parameters.items()} != shapes
            or {name: list(value.shape) for name, value in model.state_dict().items()} != shapes
            or any(not value.requires_grad or value.dtype != torch.float32 for value in parameters.values())
            or parameters["lm_head.weight"] is not parameters["model.embed_tokens.weight"]
            or getattr(model, "peft_config", None)):
        raise ValueError("Gemma 3 full-QAT must train every original parameter in FP32 and retain its tied aliases")
    return _verify_coverage(qat_summary, shapes)


def _checkpoint_tensors(checkpoint: Path) -> dict[str, dict[str, Any]]:
    from safetensors import safe_open

    result = {}
    shards = sorted(checkpoint.glob("model*.safetensors"))
    if not shards:
        raise ValueError("Gemma 3 full-QAT requires dense Safetensors checkpoint shards")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safetensors handle is not a mapping
                if name in result:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                tensor = handle.get_slice(name)
                result[name] = {"shape": tensor.get_shape(), "dtype": tensor.get_dtype()}
    return result


def verify_gemma3_full_qat_checkpoint(config: dict[str, Any], metadata: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Additional W8 admission; caller still verifies every file/config/data hash."""
    validate_gemma3_full_qat_config(config)
    shapes = expected_parameter_shapes(validate_local_gemma3_1b_config(checkpoint))
    source_config = validate_local_gemma3_1b_config(Path(config["model"]["model_source"]))
    if expected_parameter_shapes(source_config) != shapes:
        raise ValueError("Gemma 3 checkpoint architecture differs from original source")
    precision = metadata.get("full_finetune_precision") or {}
    scope = metadata.get("full_parameter_scope") or {}
    if (metadata.get("checkpoint_kind") != "full_model" or metadata.get("lora")
            or precision.get("policy") != "fp32_parameters_v1" or precision.get("parameter_dtype") != "float32"
            or precision.get("compute_dtype") not in {"float32", "float16", "bfloat16"}
            or scope.get("verified") is not True or scope.get("scope") != "all_model_parameters"
            or scope.get("master_trainable_dtype") != "float32"
            or scope.get("frozen_parameter_count") != 0 or scope.get("adapter_parameter_count") != 0
            or (metadata.get("backward_preflight") or {}).get("status") != "passed"):
        raise ValueError("Gemma 3 checkpoint lacks full-parameter FP32/backward provenance")
    from ir_training.qat.full_model_contract import _verify_scope_against_seed

    _verify_scope_against_seed(scope, shapes, metadata)
    tied = [set(record["aliases"]) for record in scope["parameters"] if len(record["aliases"]) > 1]
    if tied != [_TIED]:
        raise ValueError("Gemma 3 full-model scope must retain only the input/output embedding tie")
    qat_summary = metadata.get("qat") or {}
    spec = QATSpec.from_config(config)
    if (qat_summary.get("numeric_contract") != qat_numeric_contract(spec)
            or qat_summary.get("spec") != json.loads(json.dumps(spec.to_dict()))):
        raise ValueError("Gemma 3 checkpoint QAT numeric contract differs from the training config")
    coverage = _verify_coverage(qat_summary, shapes)
    if metadata.get("gemma3_qat_coverage") != coverage:
        raise ValueError("Gemma 3 checkpoint lacks its bound all-matrix coverage proof")
    tensors = _checkpoint_tensors(checkpoint)
    if set(tensors) - set(shapes) or not (set(shapes) - set(tensors)) <= _TIED:
        raise ValueError("Gemma 3 saved checkpoint tensor inventory is incomplete or unexpected")
    if not _TIED.intersection(tensors):
        raise ValueError("Gemma 3 saved checkpoint has no trained embedding tensor")
    for name, tensor in tensors.items():
        if tensor["shape"] != shapes[name] or tensor["dtype"] != "F32":
            raise ValueError(f"Gemma 3 checkpoint must preserve FP32 parameter shape: {name}")
    if len(_TIED.intersection(tensors)) == 2:
        # HF safetensors removes one of these aliases. Reject ambiguous duplicated
        # values instead of allowing from_pretrained to overwrite a tied weight.
        raise ValueError("Gemma 3 safetensors must serialize one shared embedding alias, not two independent values")
    return {"verified": True, "workflow": WORKFLOW, "quantization_recipe": "dynamic_wi8_afp32",
            "named_parameter_count": len(shapes), "serialized_parameter_count": len(tensors),
            "trainable_numel": scope["trainable_numel"], "matrix_count": coverage["matrix_count"],
            "embedding_included": True, "all_parameters_trainable": True,
            "fresh_graph_export": True, "official_graph_transplant": False,
            "physical_serialization_audit": False, "native_runtime_validation_required": True}
