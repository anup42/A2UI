"""Experimental dense E2B PTQ; public bit allocation, NOT the official graph/QAT.

AI Edge Quantizer matches operation OUTPUT names, not weight names. In particular
the two external embedding tables can both be named arith.constant. Keep
this policy separate from retained-mobile QAT and from the working W4 recipe.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RECIPE_NAME = "gemma4_dense_mixed248_channelwise_experimental"
RECIPE_PATH = Path(__file__).resolve().parents[3] / "configs/export/gemma4_dense_mixed248.json"
FC = "FULLY_CONNECTED"
EMBED = "EMBEDDING_LOOKUP"
# No wildcard fallback: an unrecognized matrix remains float and fails the
# physical gate instead of silently receiving an invented width.
RULES = (
    ("attention", FC, r"(?:Gemma4TextAttention_self_attn/|\.self_attn\.)", 4),
    ("early_mlp", FC, r"(?:Gemma4TextDecoderLayer_(?:[0-9]|1[0-4])/[^;]*Gemma4TextMLP_mlp/|\.layers\.(?:[0-9]|1[0-4])\.mlp\.)", 4),
    ("late_mlp", FC, r"(?:Gemma4TextDecoderLayer_(?:1[5-9]|2[0-9]|3[0-4])/[^;]*Gemma4TextMLP_mlp/|\.layers\.(?:1[5-9]|2[0-9]|3[0-4])\.mlp\.)", 2),
    ("per_layer_projection", FC, r"per_layer_(?:model_projection|input_gate|projection)(?:[;/.]|$)", 8),
    ("lm_head", FC, r"(?:Linear_lm_head(?:[;/]|$)|(?:^|[.;/])lm_head(?:[.;/]|$)|(?:^|;)(?:prefill_[0-9]+|decode)_logits_output(?:;|$))", 2),
    ("token_embedding", EMBED, r"(?:LiteRTExportableModuleForEmbedder/|(?:^|[./])embed_tokens(?:[;/.]|$))", 2),
    ("per_layer_embedding", EMBED, r"(?:LiteRTExportableModuleForPerLayerEmbedder/|embed_tokens_per_layer(?:[;/.]|$))", 4),
)
LAYER_PATTERN = re.compile(r"(?:Gemma4TextDecoderLayer_|\.layers\.)([0-9]+)(?:/|\.)")


def canonical_recipe() -> list[dict[str, Any]]:
    return [
        {"regex": regex, "operation": operation, "algorithm_key": "min_max_uniform_quantize",
         "op_config": {
             "weight_tensor_config": {"num_bits": bits, "symmetric": True,
                                      "granularity": "CHANNELWISE", "dtype": "INT"},
             "compute_precision": "INTEGER", "explicit_dequantize": False,
             "skip_checks": False, "min_weight_elements": 0}}
        for _, operation, regex, bits in RULES
    ]


def policy_for_scope(operation: str, scope: str) -> tuple[str, int]:
    matches = [(role, bits) for role, op, regex, bits in RULES
               if op == operation and re.search(regex, scope)]
    if len(matches) != 1:
        raise ValueError(f"W248 cannot classify {operation} scope unambiguously: {scope[:350]!r}")
    return matches[0]


def probe_recipe(recipe_path: Path = RECIPE_PATH) -> dict[str, Any]:
    """Exercise the exact JSON-file loader used by LiteRT Torch, without weights."""
    from ai_edge_quantizer import recipe_manager
    from ai_edge_quantizer.utils import recipe_utils

    recipe_path = recipe_path.resolve()
    if recipe_path.suffix != ".json":
        raise ValueError("W248 requires a JSON recipe file")
    payload = recipe_path.read_bytes()
    expected = canonical_recipe()
    if json.loads(payload) != expected or recipe_utils.resolve_recipe(str(recipe_path)) != expected:
        raise ValueError("W248 recipe file differs from the checked experimental bit policy")
    manager = recipe_manager.RecipeManager()
    manager.load_quantization_recipe(expected)
    # Actual converter scope spelling, layer-boundary checks and both embedders.
    checks = [
        (FC, f"Gemma4TextDecoderLayer_{layer}/Gemma4TextMLP_mlp/Linear_{projection};1",
         4 if layer < 15 else 2)
        for layer in range(35) for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    checks += [
        (FC, f"Gemma4TextDecoderLayer_{layer}/Gemma4TextAttention_self_attn/Linear_q_proj;1", 4)
        for layer in range(35)
    ]
    checks += [
        (FC, "Linear_per_layer_model_projection;1", 8),
        (FC, "Gemma4TextDecoderLayer_34/Linear_per_layer_input_gate;1", 8),
        (FC, "Gemma4TextDecoderLayer_34/Linear_per_layer_projection;1", 8),
        (FC, "decode_logits_output;", 2), (FC, "prefill_128_logits_output;", 2),
        (FC, "model.lm_head;", 2),
        (EMBED, "LiteRTExportableModuleForEmbedder/Gemma4TextScaledWordEmbedding_model;4", 2),
        (EMBED, "LiteRTExportableModuleForPerLayerEmbedder/Gemma4TextScaledWordEmbedding_embed_tokens_per_layer;4", 4),
    ]
    for operation, scope, bits in checks:
        algorithm, config = manager.get_quantization_configs(operation, scope)
        weight = config.weight_tensor_config
        if (algorithm != "min_max_uniform_quantize" or weight is None
                or weight.num_bits != bits or weight.dtype != "INT"
                or weight.symmetric is not True or weight.granularity != "CHANNELWISE"
                or config.activation_tensor_config is not None
                or config.compute_precision != "INTEGER"
                or config.explicit_dequantize is not False or config.skip_checks
                or config.min_weight_elements != 0):
            raise ValueError(f"Installed quantizer cannot apply W248 {bits}-bit policy at {scope}")
    if manager.need_calibration():
        raise ValueError("Experimental dense W248 must not require activation calibration")
    if recipe_path.read_bytes() != payload:
        raise ValueError("W248 recipe changed during validation")
    return {"path": str(recipe_path), "sha256": hashlib.sha256(payload).hexdigest(),
            "recipe": RECIPE_NAME, "representation": "json_file", "file_loading_tested": True,
            "scope_matching": "operation_output_names", "tested_scopes": len(checks),
            "calibration_required": False, "weight_granularity": "CHANNELWISE",
            "serialized_activation_contract": "FLOAT32_not_official_static_A8",
            "official_graph": False, "official_qat": False, "mtp_exported": False}


def check_model_config(config: dict[str, Any]) -> None:
    text = config.get("text_config") or config
    if (config.get("model_type") not in {"gemma4", "gemma4_text"}
            or text.get("num_hidden_layers") != 35 or text.get("hidden_size") != 1536
            or text.get("hidden_size_per_layer_input") != 256):
        raise ValueError("Experimental W248 requires the 35-layer Gemma4 E2B text architecture "
                         "(hidden_size=1536, hidden_size_per_layer_input=256)")


def inspect_policy(package: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on wrong widths, unnamed matrices, missing roles or MTP.

    This examines each physical FC/embedding input, not an aggregate dtype
    histogram. It does not assert Google's constant count: dense graph fusion
    and prefill/decode aliases differ from the released mobile graph.
    """
    sections = {
        section.get("index"): next((item.get("value") for item in section.get("items", [])
                                  if item.get("key") == "model_type"), None)
        for section in package.get("sections", []) if section.get("data_type_name") == "TFLiteModel"
    }
    expected_sections = {"tf_lite_prefill_decode", "tf_lite_embedder", "tf_lite_per_layer_embedder"}
    if len(sections) != 3 or set(sections.values()) != expected_sections:
        raise ValueError("W248 requires exactly target/token/per-layer graphs and no MTP section")
    graphs = package.get("graphs") or []
    if (len(graphs) != 3 or {g.get("section_index") for g in graphs} != set(sections)
            or any(g.get("available") is not True for g in graphs)):
        raise ValueError("W248 package graph inspection is incomplete")
    roles: Counter[str] = Counter()
    layers: dict[str, set[int]] = defaultdict(set)
    storage: Counter[str] = Counter()
    for graph in graphs:
        section = sections[graph["section_index"]]
        buffers = {b["index"]: b.get("logical_size", 0) for b in graph.get("buffers", [])}
        for sg in graph.get("subgraphs", []):
            tensors = sg.get("tensors", [])
            for op in sg.get("operators", []):
                operation = op.get("builtin_name")
                if operation not in {FC, EMBED, "GATHER"}:
                    continue
                inputs = op.get("inputs") or []
                pos = 0 if operation == "GATHER" else 1
                if len(inputs) <= pos or not 0 <= inputs[pos] < len(tensors):
                    raise ValueError("W248 matrix operand is missing")
                weight = tensors[inputs[pos]]
                if operation == "GATHER" and len(weight.get("shape", [])) < 2:
                    continue  # Integer shape/index gathers are not embeddings.
                if operation == "GATHER":
                    raise ValueError("W248 requires explicit EMBEDDING_LOOKUP routing, not an unquantized GATHER")
                # No inserted dequant/cast path is expected from this dynamic
                # recipe. Unknown/nonconstant operands must not pass on dtype.
                shape = weight.get("shape", [])
                if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
                    raise ValueError("W248 physical matrix shape is invalid")
                outputs = op.get("outputs") or []
                if not outputs or any(not 0 <= i < len(tensors) for i in outputs):
                    raise ValueError("W248 operation has no named output scope")
                if not 0 <= inputs[0] < len(tensors):
                    raise ValueError("W248 operation has no activation/index input")
                input_type = tensors[inputs[0]].get("type_name")
                input_types = {"FLOAT32"} if operation == FC else {"INT32", "INT64"}
                if input_type not in input_types or any(tensors[i].get("type_name") != "FLOAT32" for i in outputs):
                    raise ValueError("W248 serialized activation contract requires FLOAT32 FC inputs/outputs "
                                     "and FLOAT32 embedding outputs with integer indices")
                scope = ";".join(str(tensors[i].get("name") or "") for i in outputs) + ";"
                role, bits = policy_for_scope(operation, scope)
                expected_section = ("tf_lite_embedder" if role == "token_embedding" else
                                    "tf_lite_per_layer_embedder" if role == "per_layer_embedding"
                                    else "tf_lite_prefill_decode")
                if section != expected_section:
                    raise ValueError(f"W248 {role} appears in the wrong package section")
                dtype = weight.get("type_name")
                if dtype != f"INT{bits}" or buffers.get(weight.get("buffer"), 0) != (math.prod(shape) * bits + 7) // 8:
                    raise ValueError(f"W248 {role} expected packed INT{bits} physical weights; actual {dtype}")
                q = weight.get("quantization_values") or {}
                scales, zeros = q.get("scales") or [], q.get("zero_points") or []
                if (q.get("quantized_dimension") != 0 or len(scales) != shape[0]
                        or len(zeros) != shape[0] or any(v != 0 for v in zeros)
                        or any(not math.isfinite(v) or v <= 0 for v in scales)):
                    raise ValueError(f"W248 {role} lacks symmetric per-channel weight scales")
                roles[role] += 1
                storage[dtype] += 1
                for layer in LAYER_PATTERN.findall(scope):
                    layers[role].add(int(layer))
    required_layers = {"attention": set(range(35)), "early_mlp": set(range(15)),
                       "late_mlp": set(range(15, 35)), "per_layer_projection": set(range(35))}
    if set(roles) != {role for role, *_ in RULES} or any(layers[k] != v for k, v in required_layers.items()):
        raise ValueError("W248 bit-policy coverage is incomplete (expected both embeddings, head and all 35 layers)")
    return {"verified": True, "policy": RECIPE_NAME, "role_matrix_uses": dict(sorted(roles.items())),
            "layers": {k: sorted(v) for k, v in sorted(layers.items())},
            "weight_type_counts": dict(sorted(storage.items())), "packed_buffer_sizes_verified": True,
            "symmetric_channelwise_scales_verified": True, "serialized_activation_types_verified": True,
            "coverage_scope": "bit_policy_and_layer_coverage_not_graph_identity_or_unique_matrix_counts",
            "mtp_exported": False,
            "official_graph": False, "official_qat": False}
