"""Tiny E2B mixed-W4/W8 recipe tests; no full HF model conversion."""

from __future__ import annotations

import copy
import importlib
import json
import sys

import pytest
from ir_training.pipeline import deployment_export as de
from test_deployment_export import install_fake_exporter, package, write_json


def _upstream_mapping():
    recipe = pytest.importorskip("ai_edge_quantizer.recipe")
    return recipe.gemma4_mixed48_b32()


def test_w4_changes_only_the_recipe_transport(tmp_path):
    kwargs = de.export_kwargs("e2b", "w4", tmp_path, tmp_path / "out", 8192)
    baseline = de.export_kwargs("e2b", "w32", tmp_path, tmp_path / "out", 8192)
    baseline["quantization_recipe"] = str(de.E2B_W4_RECIPE_PATH)
    assert kwargs == baseline
    assert de.deployment_variants("e2b")["w4"]["recipe"] == "gemma4_mixed48_b32"


def test_reproduces_mapping_into_list_loader_failure():
    manager_module = pytest.importorskip("ai_edge_quantizer.recipe_manager")
    with pytest.raises(TypeError, match="string indices must be integers"):
        manager_module.RecipeManager().load_quantization_recipe(_upstream_mapping())


def test_real_probe_proves_flattened_policy_and_recipe_hash():
    report = de.probe_e2b_w4_recipe(_upstream_mapping())
    assert report["sha256"] == de.file_sha256(de.E2B_W4_RECIPE_PATH)
    assert report["transport"] == "equivalent_flat_operator_rules"
    assert report["representation"] == "json_file"
    assert report["file_loading_tested"] is True
    assert report["calibration_required"] is False
    assert report["rule_count"] == 3
    assert report["policy"] == {
        "fully_connected": "INT4_BLOCK32", "per_layer_fully_connected": "INT8_CHANNELWISE",
        "token_embedder": "INT4_BLOCK32", "per_layer_embedder": "INT4_BLOCK32",
    }


def test_probe_checks_the_exact_export_recipe_path(monkeypatch, tmp_path):
    native_quantizer = pytest.importorskip("ai_edge_quantizer")
    real_probe = de.probe_e2b_w4_recipe
    from ai_edge_quantizer.utils import recipe_utils

    install_fake_exporter(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "ai_edge_quantizer", native_quantizer)
    monkeypatch.setattr(de, "probe_e2b_w4_recipe", real_probe)
    original_kwargs = de.export_kwargs
    alternative = tmp_path / "recipe with spaces.json"
    alternative.write_bytes(de.E2B_W4_RECIPE_PATH.read_bytes())

    def kwargs(*args, **kw):
        result = original_kwargs(*args, **kw)
        if args[1] == "w4":
            result["quantization_recipe"] = str(alternative)
        return result

    seen = []
    resolver = recipe_utils.resolve_recipe

    def resolve(path, *args, **kwargs):
        seen.append(path)
        return resolver(path, *args, **kwargs)

    monkeypatch.setattr(de, "export_kwargs", kwargs)
    monkeypatch.setattr(recipe_utils, "resolve_recipe", resolve)
    report = de.probe_exporter(profile="e2b", model_dir=tmp_path)
    assert seen == [str(alternative.resolve())]
    assert report["recipe_files"][de.E2B_W4_RECIPE]["path"] == seen[0]


@pytest.mark.parametrize("failure", ["resolver_error", "mapping_instead_of_list"])
def test_probe_rejects_file_loading_failures_even_with_valid_factory(monkeypatch, failure):
    mapping = _upstream_mapping()
    from ai_edge_quantizer.utils import recipe_utils

    def resolve(path):
        if failure == "resolver_error":
            raise ValueError("file loading unsupported")
        return mapping

    monkeypatch.setattr(recipe_utils, "resolve_recipe", resolve)
    with pytest.raises(ValueError, match="file loading unsupported|resolution differs"):
        de.probe_e2b_w4_recipe(mapping)


def test_probe_rejects_bare_named_recipe_before_loading():
    from pathlib import Path
    with pytest.raises(ValueError, match="JSON recipe file"):
        de.probe_e2b_w4_recipe(_upstream_mapping(), recipe_path=Path("gemma4_mixed48_b32"))


def test_flat_recipe_matches_each_section_for_representative_scopes():
    mapping = _upstream_mapping()
    from ai_edge_quantizer import recipe_manager
    flat = recipe_manager.RecipeManager()
    flat.load_quantization_recipe(json.loads(de.E2B_W4_RECIPE_PATH.read_bytes()))
    for section, rules in mapping.items():
        expected = recipe_manager.RecipeManager()
        expected.load_quantization_recipe(rules)
        operation = "FULLY_CONNECTED" if section == "tf_lite_prefill_decode" else "EMBEDDING_LOOKUP"
        for scope in ("layers.0.mlp.up_proj", "per_layer_model_projection", "per_layer_input_gate",
                      "layers.34.self_attn.q_proj", "embed_tokens", "per_layer_embedder"):
            assert flat.get_quantization_configs(operation, scope) == expected.get_quantization_configs(operation, scope)
    for operation in ("ADD", "MEAN", "RSQRT", "STABLEHLO_COMPOSITE"):
        assert flat.get_quantization_configs(operation, "per_layer_projection_norm")[0] == "no_quantize"


@pytest.mark.parametrize("mutation", ["not_mapping", "missing_section", "unknown_section", "empty_rules",
                                    "not_rule_list", "conflicting_embedders", "overlapping_ops", "changed_fc", "changed_order"])
def test_probe_rejects_unsafe_upstream_recipe_mapping(mutation):
    mapping = copy.deepcopy(_upstream_mapping())
    if mutation == "not_mapping":
        mapping = mapping["tf_lite_prefill_decode"]
    elif mutation == "missing_section":
        mapping.pop("tf_lite_per_layer_embedder")
    elif mutation == "unknown_section":
        mapping["default"] = mapping["tf_lite_embedder"]
    elif mutation == "empty_rules":
        mapping["tf_lite_embedder"] = []
    elif mutation == "not_rule_list":
        mapping["tf_lite_embedder"] = "dynamic_wi4b32_afp32"
    elif mutation == "conflicting_embedders":
        mapping["tf_lite_per_layer_embedder"][0]["op_config"]["weight_tensor_config"]["num_bits"] = 8
    elif mutation == "overlapping_ops":
        mapping["tf_lite_prefill_decode"][0]["operation"] = "*"
    elif mutation == "changed_fc":
        mapping["tf_lite_prefill_decode"][1]["op_config"]["weight_tensor_config"]["num_bits"] = 4
    else:
        mapping["tf_lite_prefill_decode"].reverse()
    with pytest.raises(ValueError, match="E2B W4"):
        de.probe_e2b_w4_recipe(mapping)


def test_probe_rejects_edited_flat_recipe(monkeypatch, tmp_path):
    recipe = json.loads(de.E2B_W4_RECIPE_PATH.read_bytes())
    recipe[1]["regex"] = "no_matching_projection"
    path = tmp_path / "changed.json"
    write_json(path, recipe)
    monkeypatch.setattr(de, "E2B_W4_RECIPE_PATH", path)
    with pytest.raises(ValueError, match="differs from.*upstream"):
        de.probe_e2b_w4_recipe(_upstream_mapping())


def test_full_preflight_rejects_recipe_before_export(monkeypatch, tmp_path):
    install_fake_exporter(monkeypatch, tmp_path)

    def reject(mapping, *, recipe_path=None):
        raise ValueError("E2B W4 unsupported mapping")

    monkeypatch.setattr(de, "probe_e2b_w4_recipe", reject)
    with pytest.raises(ValueError, match="unsupported mapping"):
        de.probe_exporter(profile="e2b", model_dir=tmp_path)


def _tiny_mixed_model():
    """Four small graphs representing text FCs and both embedding sections."""
    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema

    model = schema.ModelT()
    model.version = 3
    model.description = "tiny Gemma4-style W4/W8 policy test"
    model.operatorCodes = []
    for name in ("FULLY_CONNECTED", "EMBEDDING_LOOKUP"):
        code = schema.OperatorCodeT()
        code.builtinCode = getattr(schema.BuiltinOperator, name)
        code.deprecatedBuiltinCode = code.builtinCode
        code.version = 1
        model.operatorCodes.append(code)
    model.buffers = [schema.BufferT()]
    model.subgraphs = []
    for scope, embedding in [("layers.0.mlp.up_proj", False), ("per_layer_model_projection", False),
                             ("embed_tokens", True), ("per_layer_embedder", True)]:
        subgraph = schema.SubGraphT()
        subgraph.name = scope
        subgraph.inputs, subgraph.outputs = [0], [2]
        subgraph.tensors = []
        buffer = schema.BufferT()
        values = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(2, 64)
        buffer.data = np.frombuffer(values.tobytes(), dtype=np.uint8)
        weight_buffer = len(model.buffers)
        model.buffers.append(buffer)
        shapes = [[1] if embedding else [1, 64], [2, 64], [1, 64] if embedding else [1, 2]]
        for index, shape in enumerate(shapes):
            tensor = schema.TensorT()
            tensor.name = f"{scope}/tensor_{index}"
            tensor.type = schema.TensorType.INT32 if embedding and index == 0 else schema.TensorType.FLOAT32
            tensor.shape = tensor.shapeSignature = shape
            tensor.hasRank = True
            tensor.buffer = weight_buffer if index == 1 else 0
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
    builder = flatbuffers.Builder(8192)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def test_real_small_quantization_stores_int4_and_int8():
    quantizer = pytest.importorskip("ai_edge_quantizer.quantizer")
    from ir_training.export.litertlm_inspector import _tflite_graph_fingerprint

    source = _tiny_mixed_model()
    original = _tflite_graph_fingerprint(memoryview(source), include_details=True)
    with pytest.raises(ValueError, match="actual package.*FLOAT32"):
        de.inspect_variant_precision({"graphs": [original]}, "e2b", "w4")
    qt = quantizer.Quantizer(source)
    qt.load_quantization_recipe(str(de.E2B_W4_RECIPE_PATH))
    assert qt.need_calibration is False
    converted = bytes(qt.quantize().quantized_model)
    graph = _tflite_graph_fingerprint(memoryview(converted), include_details=True)
    precision = de.inspect_variant_precision({"graphs": [graph]}, "e2b", "w4")
    types = precision["weight_type_counts"]
    assert types.get("INT4", 0) + types.get("UINT4", 0) == 3  # FC + token/per-layer embedders.
    assert types["INT8"] == 1  # The per_layer FC exception is preserved.
    assert len(types) == 2
    buffers = {buffer["index"]: buffer["logical_size"] for buffer in graph["buffers"]}
    for subgraph in graph["subgraphs"]:
        # Only the FC/lookup's actual weight operand is a model matrix. The
        # legitimate FLOAT16 block-scale tensors must not be counted as weights.
        operator = next(op for op in subgraph["operators"] if op["builtin_name"] in {"FULLY_CONNECTED", "EMBEDDING_LOOKUP"})
        weight = subgraph["tensors"][operator["inputs"][1]]
        expected = {"INT8"} if subgraph["name"] == "per_layer_model_projection" else {"INT4", "UINT4"}
        assert weight["type_name"] in expected
        assert buffers[weight["buffer"]] == (128 if weight["type_name"] == "INT8" else 64)


@pytest.mark.parametrize("weights", [["INT8"], ["FLOAT32"], ["INT4", "FLOAT32"]])
def test_w4_precision_gate_still_rejects_wrong_storage(weights):
    with pytest.raises(ValueError, match="actual package"):
        de.inspect_variant_precision(package(weights), "e2b", "w4")


@pytest.mark.parametrize("weights", [["INT4", "INT8"], ["INT8"], ["INT4", "FLOAT32"]])
def test_w4_conversion_snapshots_recipe_and_checks_physical_output(monkeypatch, tmp_path, weights):
    model_dir, out = tmp_path / "merged_hf", tmp_path / "w4"
    install_fake_exporter(monkeypatch, model_dir)
    template = model_dir / "deployment_chat_template.jinja"
    template.write_text("fixture template")
    write_json(model_dir / "deployment_source.json", {
        "profile": "e2b", "official_retained_scale_export": False,
        "merged_files": {p.name: de.file_sha256(p) for p in model_dir.iterdir()},
        "template_parity": {"passed": True, "sha256": de.file_sha256(template)},
    })
    source_bytes = {p.name: p.read_bytes() for p in model_dir.iterdir()}

    def export(**kwargs):
        assert kwargs["quantization_recipe"] == str((out / "w4_quantization_recipe.json").resolve())
        assert json.loads((out / "w4_quantization_recipe.json").read_bytes()) == json.loads(de.E2B_W4_RECIPE_PATH.read_bytes())
        assert kwargs["experimental_use_fp16"] is False
        assert "experimental_use_mixed_precision" not in kwargs
        (out / "model.litertlm").write_bytes(b"mock bundle; separate tests use real quantized graphs")

    monkeypatch.setattr(sys.modules["litert_torch.generative.export_hf.export"], "export", export)
    import ir_training.export.litertlm_inspector as inspector
    monkeypatch.setattr(inspector, "inspect_litertlm", lambda *a, **k: package(weights, cast=True))
    if weights != ["INT4", "INT8"]:
        with pytest.raises(ValueError, match="actual package"):
            de.convert_deployment_variant(profile="e2b", variant="w4", model_dir=model_dir, output_dir=out)
        assert not (out / "export_manifest.json").exists()
    else:
        result = de.convert_deployment_variant(profile="e2b", variant="w4", model_dir=model_dir, output_dir=out)
        assert result["recipe"] == "gemma4_mixed48_b32"
        assert result["quantization_recipe_file"]["sha256"] == de.file_sha256(de.E2B_W4_RECIPE_PATH)
        plan = {"profile": "e2b", "merged_model_dir": str(model_dir),
                "variants": {"w4": {"output_dir": str(out), "artifact": str(out / "model.litertlm")}}}
        assert str(out / "w4_quantization_recipe.json") in de.validate_deployment_export_output(plan, "w4")["files"]
        (out / "w4_quantization_recipe.json").write_text("[]")
        with pytest.raises(ValueError, match="recipe changed"):
            de.validate_deployment_export_output(plan, "w4")
    assert source_bytes == {p.name: p.read_bytes() for p in model_dir.iterdir()}


def test_export_loads_and_quantizes_the_preflighted_snapshot(monkeypatch, tmp_path):
    """Mock HF conversion/bundling only; run the real recipe loader + quantizer."""
    quantizer = pytest.importorskip("ai_edge_quantizer.quantizer")
    native_quantizer = importlib.import_module("ai_edge_quantizer")
    from ai_edge_quantizer.utils import recipe_utils
    from ir_training.export.litertlm_inspector import _tflite_graph_fingerprint
    real_probe = de.probe_e2b_w4_recipe
    model_dir, out = tmp_path / "merged_hf", tmp_path / "W4 retry with spaces"
    install_fake_exporter(monkeypatch, model_dir)
    monkeypatch.setitem(sys.modules, "ai_edge_quantizer", native_quantizer)
    monkeypatch.setattr(de, "probe_e2b_w4_recipe", real_probe)
    template = model_dir / "deployment_chat_template.jinja"
    template.write_text("fixture template")
    write_json(model_dir / "deployment_source.json", {
        "profile": "e2b", "official_retained_scale_export": False,
        "merged_files": {p.name: de.file_sha256(p) for p in model_dir.iterdir()},
        "template_parity": {"passed": True, "sha256": de.file_sha256(template)},
    })
    loads = []
    resolver = recipe_utils.resolve_recipe

    def resolve(path, *args, **kwargs):
        loads.append(path)
        return resolver(path, *args, **kwargs)

    monkeypatch.setattr(recipe_utils, "resolve_recipe", resolve)
    inspection = {}

    def export(**kwargs):
        recipe_path = kwargs["quantization_recipe"]
        assert recipe_path == str((out / "w4_quantization_recipe.json").resolve())
        assert loads[-1] == recipe_path  # Snapshot verified before expensive HF export.
        qt = quantizer.Quantizer(_tiny_mixed_model())
        qt.load_quantization_recipe(recipe_path)  # Exact public call in LiteRT Torch.
        converted = bytes(qt.quantize().quantized_model)
        inspection["graphs"] = [_tflite_graph_fingerprint(memoryview(converted), include_details=True)]
        (out / "model.litertlm").write_bytes(b"bundle stub; real quantized TFLite graphs inspected separately")

    monkeypatch.setattr(sys.modules["litert_torch.generative.export_hf.export"], "export", export)
    import ir_training.export.litertlm_inspector as inspector
    monkeypatch.setattr(inspector, "inspect_litertlm", lambda *a, **k: inspection)
    result = de.convert_deployment_variant(profile="e2b", variant="w4", model_dir=model_dir, output_dir=out)
    snapshot = str((out / "w4_quantization_recipe.json").resolve())
    assert loads == [str(de.E2B_W4_RECIPE_PATH.resolve()), snapshot, snapshot]
    assert result["export_kwargs"]["quantization_recipe"] == snapshot
    assert result["exporter_preflight"]["recipe_files"][de.E2B_W4_RECIPE]["path"] == snapshot
    assert result["actual_precision"]["weight_type_counts"] == {"INT4": 3, "INT8": 1}


def test_snapshot_loader_failure_aborts_before_hf_conversion(monkeypatch, tmp_path):
    model_dir, out = tmp_path / "merged_hf", tmp_path / "w4"
    install_fake_exporter(monkeypatch, model_dir)
    template = model_dir / "deployment_chat_template.jinja"
    template.write_text("fixture template")
    write_json(model_dir / "deployment_source.json", {
        "profile": "e2b", "official_retained_scale_export": False,
        "merged_files": {p.name: de.file_sha256(p) for p in model_dir.iterdir()},
        "template_parity": {"passed": True, "sha256": de.file_sha256(template)},
    })
    original_probe = de.probe_e2b_w4_recipe

    def probe(mapping, *, recipe_path=None):
        if recipe_path.name == "w4_quantization_recipe.json":
            raise ValueError("snapshot loader failed")
        return original_probe(mapping, recipe_path=recipe_path)

    def export(**kwargs):
        pytest.fail("HF conversion must not start after snapshot preflight failure")

    monkeypatch.setattr(de, "probe_e2b_w4_recipe", probe)
    monkeypatch.setattr(sys.modules["litert_torch.generative.export_hf.export"], "export", export)
    with pytest.raises(ValueError, match="snapshot loader failed"):
        de.convert_deployment_variant(profile="e2b", variant="w4", model_dir=model_dir, output_dir=out)
    assert not (out / "export_manifest.json").exists()
