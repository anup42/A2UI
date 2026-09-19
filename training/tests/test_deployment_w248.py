"""Focused experimental dense Gemma4 E2B W2/W4/W8 deployment tests.

The real quantizer tests below use tiny in-memory TFLite graphs.  They exercise
the installed AI Edge Quantizer without downloading or converting an HF model.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
import types

import pytest

from ir_training.export import gemma4_mixed248 as mixed248
from ir_training.pipeline import deployment_export as de


def _matrix_model(scopes: list[tuple[str, str]]) -> bytes:
    """Build independent 2x64 matrix operations with converter-like outputs."""
    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema

    model = schema.ModelT()
    model.version = 3
    model.description = "tiny Gemma4 dense mixed W2/W4/W8 policy test"
    model.operatorCodes = []
    for name in ("FULLY_CONNECTED", "EMBEDDING_LOOKUP"):
        code = schema.OperatorCodeT()
        code.builtinCode = getattr(schema.BuiltinOperator, name)
        code.deprecatedBuiltinCode = code.builtinCode
        code.version = 1
        model.operatorCodes.append(code)
    model.buffers = [schema.BufferT()]
    model.subgraphs = []
    values = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(2, 64)
    for index, (operation, output_scope) in enumerate(scopes):
        embedding = operation == "EMBEDDING_LOOKUP"
        subgraph = schema.SubGraphT()
        subgraph.name = f"tiny_matrix_{index}"
        subgraph.inputs, subgraph.outputs = [0], [2]
        buffer = schema.BufferT()
        buffer.data = np.frombuffer(values.tobytes(), dtype=np.uint8)
        weight_buffer = len(model.buffers)
        model.buffers.append(buffer)
        shapes = [[1] if embedding else [1, 64], [2, 64], [1, 64] if embedding else [1, 2]]
        subgraph.tensors = []
        for tensor_index, shape in enumerate(shapes):
            tensor = schema.TensorT()
            # The policy must use the operation output below.  Every physical
            # model matrix intentionally has the same uninformative name.
            tensor.name = (output_scope if tensor_index == 2 else
                           ("arith.constant" if len(scopes) == 1 else f"arith.constant.{index}")
                           if tensor_index == 1 else f"input_{index}")
            tensor.type = schema.TensorType.INT32 if embedding and tensor_index == 0 else schema.TensorType.FLOAT32
            tensor.shape = tensor.shapeSignature = shape
            tensor.hasRank = True
            tensor.buffer = weight_buffer if tensor_index == 1 else 0
            subgraph.tensors.append(tensor)
        operator = schema.OperatorT()
        operator.opcodeIndex = 1 if embedding else 0
        operator.inputs = [0, 1] if embedding else [0, 1, -1]
        operator.outputs = [2]
        if not embedding:
            operator.builtinOptionsType = schema.BuiltinOptions.FullyConnectedOptions
            operator.builtinOptions = schema.FullyConnectedOptionsT()
        subgraph.operators = [operator]
        model.subgraphs.append(subgraph)
    builder = flatbuffers.Builder(1024 * 1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def _tiny_three_section_models() -> dict[str, bytes]:
    target: list[tuple[str, str]] = []
    for layer in range(35):
        target.extend([
            ("FULLY_CONNECTED",
             f"Gemma4TextDecoderLayer_{layer}/Gemma4TextAttention_self_attn/Linear_q_proj;1"),
            ("FULLY_CONNECTED",
             f"Gemma4TextDecoderLayer_{layer}/Gemma4TextMLP_mlp/Linear_up_proj;1"),
            ("FULLY_CONNECTED",
             f"Gemma4TextDecoderLayer_{layer}/Linear_per_layer_model_projection;1"),
        ])
    target.extend([
        ("FULLY_CONNECTED", "decode_logits_output"),
        ("FULLY_CONNECTED", "prefill_128_logits_output"),
    ])
    return {
        "tf_lite_prefill_decode": _matrix_model(target),
        "tf_lite_embedder": _matrix_model([
            ("EMBEDDING_LOOKUP",
             "LiteRTExportableModuleForEmbedder/Gemma4TextScaledWordEmbedding_model;4")]),
        "tf_lite_per_layer_embedder": _matrix_model([
            ("EMBEDDING_LOOKUP",
             "LiteRTExportableModuleForPerLayerEmbedder/"
             "Gemma4TextScaledWordEmbedding_embed_tokens_per_layer;4")]),
    }


@pytest.fixture(scope="module")
def quantized_package():
    quantizer = pytest.importorskip("ai_edge_quantizer.quantizer")
    from ir_training.export.litertlm_inspector import _tflite_graph_fingerprint

    sections = []
    graphs = []
    for section_index, (model_type, source) in enumerate(_tiny_three_section_models().items()):
        qt = quantizer.Quantizer(source)
        qt.load_quantization_recipe(str(mixed248.RECIPE_PATH))
        assert qt.need_calibration is False
        converted = bytes(qt.quantize().quantized_model)
        graph = _tflite_graph_fingerprint(memoryview(converted), include_details=True)
        graph["section_index"] = section_index
        graphs.append(graph)
        sections.append({
            "index": section_index,
            "data_type_name": "TFLiteModel",
            "items": [{"key": "model_type", "value": model_type}],
        })
    return {"sections": sections, "graphs": graphs}


def _matrices(package):
    for graph in package["graphs"]:
        buffers = {item["index"]: item for item in graph["buffers"]}
        for subgraph in graph["subgraphs"]:
            tensors = subgraph["tensors"]
            for operator in subgraph["operators"]:
                if operator["builtin_name"] not in {"FULLY_CONNECTED", "EMBEDDING_LOOKUP"}:
                    continue
                weight = tensors[operator["inputs"][1]]
                scope = ";".join(tensors[index]["name"] for index in operator["outputs"]) + ";"
                yield graph, subgraph, operator, weight, buffers[weight["buffer"]], scope


def _set_storage(weight, buffer, bits):
    weight["type_name"] = f"INT{bits}"
    buffer["logical_size"] = 2 * 64 * bits // 8


def test_recipe_is_canonical_exact_json_and_loads_through_public_file_api():
    assert json.loads(mixed248.RECIPE_PATH.read_bytes()) == mixed248.canonical_recipe()
    report = mixed248.probe_recipe()
    assert report["recipe"] == mixed248.RECIPE_NAME
    assert report["file_loading_tested"] is True
    assert report["scope_matching"] == "operation_output_names"
    assert report["tested_scopes"] == 148
    assert report["calibration_required"] is False
    assert report["official_graph"] is False
    assert report["official_qat"] is False
    assert report["mtp_exported"] is False


def test_recipe_probe_rejects_noncanonical_json_even_when_it_loads(tmp_path):
    changed = mixed248.canonical_recipe()
    changed[0]["op_config"]["weight_tensor_config"]["num_bits"] = 8
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the checked experimental bit policy"):
        mixed248.probe_recipe(path)


@pytest.mark.parametrize(
    "scope,expected",
    [
        ("Gemma4TextDecoderLayer_14/Gemma4TextMLP_mlp/Linear_up_proj;1", ("early_mlp", 4)),
        ("Gemma4TextDecoderLayer_15/Gemma4TextMLP_mlp/Linear_up_proj;1", ("late_mlp", 2)),
        ("Gemma4TextDecoderLayer_34/Gemma4TextMLP_mlp/Linear_up_proj;1", ("late_mlp", 2)),
    ],
)
def test_mlp_layer_boundaries(scope, expected):
    assert mixed248.policy_for_scope("FULLY_CONNECTED", scope) == expected


def test_output_scopes_distinguish_same_named_constants_and_both_lm_heads():
    assert mixed248.policy_for_scope(
        "EMBEDDING_LOOKUP",
        "LiteRTExportableModuleForEmbedder/Gemma4TextScaledWordEmbedding_model;4",
    ) == ("token_embedding", 2)
    assert mixed248.policy_for_scope(
        "EMBEDDING_LOOKUP",
        "LiteRTExportableModuleForPerLayerEmbedder/"
        "Gemma4TextScaledWordEmbedding_embed_tokens_per_layer;4",
    ) == ("per_layer_embedding", 4)
    assert mixed248.policy_for_scope("FULLY_CONNECTED", "decode_logits_output;") == ("lm_head", 2)
    assert mixed248.policy_for_scope("FULLY_CONNECTED", "prefill_128_logits_output;") == ("lm_head", 2)
    with pytest.raises(ValueError, match="cannot classify"):
        mixed248.policy_for_scope("EMBEDDING_LOOKUP", "arith.constant")


@pytest.mark.parametrize(
    "operation,scope",
    [
        ("FULLY_CONNECTED", "Gemma4TextDecoderLayer_35/Gemma4TextMLP_mlp/Linear_up_proj;1"),
        ("FULLY_CONNECTED", "unknown_dense_output"),
        ("EMBEDDING_LOOKUP", "unknown_embedding_output"),
    ],
)
def test_unknown_or_out_of_architecture_scopes_fail_closed(operation, scope):
    with pytest.raises(ValueError, match="cannot classify"):
        mixed248.policy_for_scope(operation, scope)


def test_architecture_guard_accepts_only_dense_e2b_shape():
    mixed248.check_model_config({
        "model_type": "gemma4",
        "text_config": {"num_hidden_layers": 35, "hidden_size": 1536,
                        "hidden_size_per_layer_input": 256},
    })
    mixed248.check_model_config({
        "model_type": "gemma4_text", "num_hidden_layers": 35,
        "hidden_size": 1536, "hidden_size_per_layer_input": 256,
    })


@pytest.mark.parametrize(
    "field,value",
    [("model_type", "gemma3"), ("num_hidden_layers", 34),
     ("hidden_size", 2048), ("hidden_size_per_layer_input", 512)],
)
def test_architecture_guard_rejects_other_shapes(field, value):
    config = {
        "model_type": "gemma4",
        "text_config": {"num_hidden_layers": 35, "hidden_size": 1536,
                        "hidden_size_per_layer_input": 256},
    }
    if field == "model_type":
        config[field] = value
    else:
        config["text_config"][field] = value
    with pytest.raises(ValueError, match="35-layer Gemma4 E2B"):
        mixed248.check_model_config(config)


def test_w248_is_e2b_only_and_explicitly_opt_in():
    assert tuple(de.deployment_variants("e2b")) == ("w32", "w16", "w8", "w4")
    selected = de.deployment_variants("e2b", ("w248",))
    assert tuple(selected) == ("w248",)
    assert selected["w248"] == {
        "weight_bits": [2, 4, 8],
        "kind": "experimental_mixed_w2_w4_w8",
        "recipe": mixed248.RECIPE_NAME,
        "experimental": True,
        "official_graph": False,
        "official_qat": False,
        "activation_contract": "dynamic_FLOAT32_not_official_static_A8",
    }
    with pytest.raises(ValueError, match="w248 is E2B-only"):
        de.deployment_variants("270m", ("w248",))
    with pytest.raises(ValueError, match="nonempty list of unique"):
        de.deployment_variants("e2b", ("w248", "w248"))


def test_w248_export_kwargs_use_exact_json_and_disable_whole_graph_mixed_precision(tmp_path):
    kwargs = de.export_kwargs("e2b", "w248", tmp_path, tmp_path / "out", 8192)
    assert kwargs["quantization_recipe"] == str(mixed248.RECIPE_PATH)
    assert kwargs["experimental_use_mixed_precision"] is False
    assert kwargs["experimental_use_fp16"] is False
    assert kwargs["externalize_embedder"] is True
    assert kwargs["prefill_lengths"] == [128]


def test_w248_selected_plan_contains_only_opted_in_variant(tmp_path):
    plan = de.build_deployment_export_plan(
        profile="e2b",
        training_config_path=tmp_path / "training.yaml",
        checkpoint_dir=tmp_path / "checkpoint",
        output_dir=tmp_path / "deployment",
        training_python=sys.executable,
        exporter_python=sys.executable,
        selected_variants=("w248",),
    )
    assert tuple(plan["variants"]) == ("w248",)
    assert plan["probe_command"][-2:] == ["--variants", "w248"]
    assert plan["variants"]["w248"]["command"][plan["variants"]["w248"]["command"].index("--variant") + 1] == "w248"
    assert plan["mtp_exported"] is False


def test_w248_precision_dispatches_to_role_aware_helper(monkeypatch):
    sentinel = {"verified": True, "role_aware": True}
    seen = []

    def inspect(package):
        seen.append(package)
        return sentinel

    monkeypatch.setattr(mixed248, "inspect_policy", inspect)
    package = {"not": "an aggregate dtype histogram"}
    assert de.inspect_variant_precision(package, "e2b", "w248") is sentinel
    assert seen == [package]


def test_real_quantizer_produces_role_correct_packed_int2_int4_int8(quantized_package):
    report = mixed248.inspect_policy(quantized_package)
    assert report["verified"] is True
    assert report["weight_type_counts"] == {"INT2": 23, "INT4": 51, "INT8": 35}
    assert report["role_matrix_uses"] == {
        "attention": 35, "early_mlp": 15, "late_mlp": 20, "lm_head": 2,
        "per_layer_embedding": 1, "per_layer_projection": 35, "token_embedding": 1,
    }
    assert report["layers"]["attention"] == list(range(35))
    assert report["layers"]["early_mlp"] == list(range(15))
    assert report["layers"]["late_mlp"] == list(range(15, 35))
    assert report["layers"]["per_layer_projection"] == list(range(35))
    assert report["packed_buffer_sizes_verified"] is True
    assert report["symmetric_channelwise_scales_verified"] is True
    assert report["serialized_activation_types_verified"] is True
    assert report["mtp_exported"] is False
    embedding_weights = [weight["name"] for graph, _, operator, weight, _, _ in _matrices(quantized_package)
                         if operator["builtin_name"] == "EMBEDDING_LOOKUP"]
    assert embedding_weights == ["arith.constant", "arith.constant"]


def test_w248_conversion_snapshots_reprobes_and_inspects_exact_recipe(
        monkeypatch, tmp_path, quantized_package):
    model_dir, output = tmp_path / "merged_hf", tmp_path / "w248"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "gemma4",
        "text_config": {"num_hidden_layers": 35, "hidden_size": 1536,
                        "hidden_size_per_layer_input": 256},
    }), encoding="utf-8")
    template = model_dir / "deployment_chat_template.jinja"
    template.write_text("fixture template", encoding="utf-8")
    source_files = {path.name: de.file_sha256(path) for path in model_dir.iterdir()}
    de._write(model_dir / "deployment_source.json", {
        "profile": "e2b",
        "official_retained_scale_export": False,
        "merged_files": source_files,
        "template_parity": {"passed": True, "sha256": de.file_sha256(template)},
    })

    selected = []

    def preflight(**kwargs):
        selected.append(kwargs["selected_variants"])
        return {"recipe_files": {
            mixed248.RECIPE_NAME: {"sha256": de.file_sha256(mixed248.RECIPE_PATH)}
        }}

    monkeypatch.setattr(de, "probe_exporter", preflight)
    probes = []
    real_probe = mixed248.probe_recipe

    def probe(path=mixed248.RECIPE_PATH):
        probes.append(path.resolve())
        return real_probe(path)

    monkeypatch.setattr(mixed248, "probe_recipe", probe)
    export_module = types.ModuleType("litert_torch.generative.export_hf.export")

    def export(**kwargs):
        snapshot = output / "w248_quantization_recipe.json"
        assert kwargs["quantization_recipe"] == str(snapshot.resolve())
        assert kwargs["experimental_use_mixed_precision"] is False
        assert json.loads(snapshot.read_bytes()) == mixed248.canonical_recipe()
        (output / "model.litertlm").write_bytes(b"test-only bundle; real TFLite graphs quantified separately")

    export_module.export = export
    monkeypatch.setitem(sys.modules, export_module.__name__, export_module)
    inspector = importlib.import_module("ir_training.export.litertlm_inspector")
    monkeypatch.setattr(inspector, "inspect_litertlm", lambda *args, **kwargs: quantized_package)

    result = de.convert_deployment_variant(
        profile="e2b", variant="w248", model_dir=model_dir, output_dir=output)
    snapshot = output / "w248_quantization_recipe.json"
    assert selected == [("w248",)]
    assert probes == [snapshot.resolve()]
    assert result["recipe"] == mixed248.RECIPE_NAME
    assert result["export_kwargs"]["quantization_recipe"] == str(snapshot.resolve())
    assert result["actual_precision"]["weight_type_counts"] == {"INT2": 23, "INT4": 51, "INT8": 35}
    assert result["official_graph"] is False
    assert result["official_qat"] is False
    assert result["mtp_exported"] is False

    plan = {"profile": "e2b", "merged_model_dir": str(model_dir),
            "variants": {"w248": {"output_dir": str(output),
                                     "artifact": str(output / "model.litertlm")}}}
    validated = de.validate_deployment_export_output(plan, "w248")
    assert str(snapshot) in validated["files"]
    manifest = output / "export_manifest.json"
    valid_report = manifest.read_bytes()
    for field, bad_value in {
        "recipe": "not_the_w248_recipe", "kind": "official_w248", "experimental": False,
        "official_graph": True, "official_qat": True, "mtp_exported": True,
        "activation_contract": "official_static_A8", "runtime_gpu_tested": True,
    }.items():
        changed_report = json.loads(valid_report)
        changed_report[field] = bad_value
        manifest.write_text(json.dumps(changed_report), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest claims differ"):
            de.validate_deployment_export_output(plan, "w248")
    manifest.write_bytes(valid_report)
    snapshot.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="recipe changed"):
        de.validate_deployment_export_output(plan, "w248")


def test_wrong_role_bits_fail_even_when_all_three_storage_types_remain(quantized_package):
    changed = copy.deepcopy(quantized_package)
    for _, _, _, weight, buffer, scope in _matrices(changed):
        if "Gemma4TextAttention_self_attn" in scope:
            _set_storage(weight, buffer, 2)
            break
    assert {weight["type_name"] for _, _, _, weight, _, _ in _matrices(changed)} == {"INT2", "INT4", "INT8"}
    with pytest.raises(ValueError, match="attention expected packed INT4"):
        mixed248.inspect_policy(changed)


def test_absent_int2_storage_fails(quantized_package):
    changed = copy.deepcopy(quantized_package)
    for _, _, _, weight, buffer, _ in _matrices(changed):
        if weight["type_name"] == "INT2":
            _set_storage(weight, buffer, 4)
    assert {weight["type_name"] for _, _, _, weight, _, _ in _matrices(changed)} == {"INT4", "INT8"}
    with pytest.raises(ValueError, match="expected packed INT2"):
        mixed248.inspect_policy(changed)


@pytest.mark.parametrize("operand", ["fc_input", "fc_output", "embedding_output", "embedding_index"])
def test_wrong_serialized_activation_contract_fails(quantized_package, operand):
    changed = copy.deepcopy(quantized_package)
    for _, sg, op, _, _, _ in _matrices(changed):
        if operand.startswith("fc") and op["builtin_name"] == "FULLY_CONNECTED":
            index = op["inputs"][0] if operand == "fc_input" else op["outputs"][0]
            sg["tensors"][index]["type_name"] = "INT8"
            break
        if operand.startswith("embedding") and op["builtin_name"] == "EMBEDDING_LOOKUP":
            index = op["inputs"][0] if operand == "embedding_index" else op["outputs"][0]
            sg["tensors"][index]["type_name"] = "FLOAT32" if operand == "embedding_index" else "INT8"
            break
    with pytest.raises(ValueError, match="serialized activation contract"):
        mixed248.inspect_policy(changed)


@pytest.mark.parametrize("corruption", ["buffer_length", "scale_count", "nonpositive_scale", "zero_point", "dimension"])
def test_physical_storage_and_channelwise_scale_corruption_fails(quantized_package, corruption):
    changed = copy.deepcopy(quantized_package)
    _, _, _, weight, buffer, _ = next(_matrices(changed))
    if corruption == "buffer_length":
        buffer["logical_size"] += 1
        expected = "packed INT4 physical weights"
    else:
        quantization = weight["quantization_values"]
        if corruption == "scale_count":
            quantization["scales"] = quantization["scales"][:-1]
        elif corruption == "nonpositive_scale":
            quantization["scales"][0] = 0.0
        elif corruption == "zero_point":
            quantization["zero_points"][0] = 1
        else:
            quantization["quantized_dimension"] = 1
        expected = "symmetric per-channel weight scales"
    with pytest.raises(ValueError, match=expected):
        mixed248.inspect_policy(changed)


def test_extra_mtp_section_fails(quantized_package):
    changed = copy.deepcopy(quantized_package)
    changed["sections"].append({
        "index": 3, "data_type_name": "TFLiteModel",
        "items": [{"key": "model_type", "value": "tf_lite_mtp"}],
    })
    with pytest.raises(ValueError, match="exactly target/token/per-layer graphs and no MTP"):
        mixed248.inspect_policy(changed)


@pytest.mark.parametrize("missing", ["lm_head", "attention_layer_34", "token_embedding"])
def test_missing_roles_or_layer_coverage_fails(quantized_package, missing):
    changed = copy.deepcopy(quantized_package)
    for graph in changed["graphs"]:
        kept = []
        for subgraph in graph["subgraphs"]:
            scopes = [subgraph["tensors"][index]["name"]
                      for operator in subgraph["operators"] for index in operator["outputs"]]
            remove = (
                (missing == "lm_head" and any("logits_output" in scope for scope in scopes))
                or (missing == "attention_layer_34" and any(
                    "Gemma4TextDecoderLayer_34/Gemma4TextAttention" in scope for scope in scopes))
                or (missing == "token_embedding" and any(
                    "LiteRTExportableModuleForEmbedder/" in scope for scope in scopes))
            )
            if not remove:
                kept.append(subgraph)
        graph["subgraphs"] = kept
    with pytest.raises(ValueError, match="bit-policy coverage is incomplete"):
        mixed248.inspect_policy(changed)
