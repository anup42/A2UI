"""Small CPU-only W16 regressions; no model download or full Gemma conversion."""

from __future__ import annotations

import json
import sys

import pytest
from ir_training.pipeline import deployment_export as de
from test_deployment_export import install_fake_exporter, package, write_json


@pytest.mark.parametrize("profile", ["e2b", "270m"])
def test_w16_casts_weights_without_changing_graph_precision(profile, tmp_path):
    kwargs = de.export_kwargs(profile, "w16", tmp_path, tmp_path / "out", 8192)
    assert kwargs["quantization_recipe"] == str(de.W16_RECIPE_PATH)
    assert kwargs["experimental_use_fp16"] is False
    assert kwargs["experimental_use_mixed_precision"] is False
    assert kwargs["externalize_embedder"] is (profile == "e2b")
    baseline = de.export_kwargs(profile, "w32", tmp_path, tmp_path / "out", 8192)
    baseline["quantization_recipe"] = str(de.W16_RECIPE_PATH)
    baseline["experimental_use_mixed_precision"] = False
    assert kwargs == baseline  # Keep the same source graph as working W32.


@pytest.mark.parametrize("profile,variant", [
    ("e2b", "w32"), ("e2b", "w8"), ("270m", "w32"), ("270m", "w8"), ("270m", "w4")])
def test_w32_w8_and_270m_w4_keep_their_existing_export_options(profile, variant, tmp_path):
    kwargs = de.export_kwargs(profile, variant, tmp_path, tmp_path / "out", 8192)
    recipes = {
        "w32": "none", "w8": "dynamic_wi8_afp32",
        "w4": "dynamic_wi4b32_afp32",
    }
    assert kwargs["quantization_recipe"] == recipes[variant]
    assert kwargs["experimental_use_fp16"] is False
    assert "experimental_use_mixed_precision" not in kwargs


def test_reproduces_rejection_of_410_fp32_weights():
    with pytest.raises(ValueError, match="FLOAT32.*410"):
        de.inspect_variant_precision(package(["FLOAT32"] * 410), "e2b", "w16")


@pytest.mark.parametrize("operator", ["EMBEDDING_LOOKUP", "GATHER"])
def test_w16_rejects_uncast_external_embedding_weights(operator):
    inspection = package(["FLOAT16"], cast=True)
    embedding = package(["FLOAT32"])["graphs"][0]
    op = embedding["subgraphs"][0]["operators"][0]
    op["builtin_name"] = operator
    if operator == "GATHER":
        op["inputs"] = list(reversed(op["inputs"]))
    inspection["graphs"].append(embedding)
    with pytest.raises(ValueError, match="actual package.*FLOAT32"):
        de.inspect_variant_precision(inspection, "e2b", "w16")


def test_real_quantizer_validates_recipe_without_loading_weights():
    pytest.importorskip("ai_edge_quantizer.recipe_manager")
    report = de.probe_w16_recipe()
    assert report["sha256"] == de.file_sha256(de.W16_RECIPE_PATH)
    assert report["validated_operations"] == ["FULLY_CONNECTED", "EMBEDDING_LOOKUP"]
    assert report["activation_contract"] == "FLOAT32"
    assert report["calibration_required"] is False
    from ai_edge_quantizer import recipe_manager
    manager = recipe_manager.RecipeManager()
    manager.load_quantization_recipe(json.loads(de.W16_RECIPE_PATH.read_text()))
    for operation in ("ADD", "MEAN", "RSQRT", "STABLEHLO_COMPOSITE"):
        algorithm, _ = manager.get_quantization_configs(operation, "per_layer_projection_norm")
        assert algorithm == "no_quantize"


@pytest.mark.parametrize("mutation", ["missing_embedding", "wrong_dtype", "wrong_bits", "threshold", "wrong_scope"])
def test_recipe_probe_fails_closed(monkeypatch, tmp_path, mutation):
    pytest.importorskip("ai_edge_quantizer.recipe_manager")
    recipe = json.loads(de.W16_RECIPE_PATH.read_text())
    if mutation == "missing_embedding":
        recipe.pop()
    elif mutation == "wrong_dtype":
        recipe[0]["op_config"]["weight_tensor_config"]["dtype"] = "INT"
    elif mutation == "wrong_bits":
        recipe[0]["op_config"]["weight_tensor_config"]["num_bits"] = 8
    elif mutation == "threshold":
        recipe[0]["op_config"]["min_weight_elements"] = 1024
    else:
        recipe[1]["regex"] = "does_not_match_any_layer"
    path = tmp_path / "recipe.json"
    write_json(path, recipe)
    monkeypatch.setattr(de, "W16_RECIPE_PATH", path)
    with pytest.raises(ValueError):
        de.probe_w16_recipe()


def test_preflight_reports_missing_float_casting_support(monkeypatch, tmp_path):
    install_fake_exporter(monkeypatch, tmp_path)

    def unsupported():
        raise ValueError("float_casting unavailable")

    monkeypatch.setattr(de, "probe_w16_recipe", unsupported)
    with pytest.raises(ValueError, match="float_casting unavailable"):
        de.probe_exporter(profile="e2b", model_dir=tmp_path)


def _tiny_model(*, embedding=False):
    """Hand-built valid TFLite graph with real buffers, not a converter mock."""
    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema

    model = schema.ModelT()
    model.version = 3
    model.description = "tiny W16 storage and numerical parity test"
    opcode = schema.OperatorCodeT()
    opcode.builtinCode = (
        schema.BuiltinOperator.EMBEDDING_LOOKUP if embedding else schema.BuiltinOperator.FULLY_CONNECTED
    )
    opcode.deprecatedBuiltinCode = opcode.builtinCode
    opcode.version = 1
    model.operatorCodes = [opcode]
    weight_values = np.array([[0.125, -0.1, 0.333], [1.25, -0.875, 0.999]], dtype=np.float32)
    weight = schema.BufferT()
    weight.data = np.frombuffer(weight_values.tobytes(), dtype=np.uint8)
    model.buffers = [schema.BufferT(), weight]
    subgraph = schema.SubGraphT()
    subgraph.name = "main"
    subgraph.inputs = [0]
    subgraph.outputs = [2]
    subgraph.tensors = []
    shapes = [[1] if embedding else [1, 3], [2, 3], [1, 3] if embedding else [1, 2]]
    for index, shape in enumerate(shapes):
        tensor = schema.TensorT()
        tensor.name = f"tensor_{index}"
        tensor.shape = shape
        tensor.shapeSignature = shape
        tensor.hasRank = True
        tensor.type = schema.TensorType.INT32 if embedding and index == 0 else schema.TensorType.FLOAT32
        tensor.buffer = 1 if index == 1 else 0
        subgraph.tensors.append(tensor)
    operator = schema.OperatorT()
    operator.opcodeIndex = 0
    operator.inputs = [0, 1] if embedding else [0, 1, -1]
    operator.outputs = [2]
    if not embedding:
        operator.builtinOptionsType = schema.BuiltinOptions.FullyConnectedOptions
        operator.builtinOptions = schema.FullyConnectedOptionsT()
    subgraph.operators = [operator]
    model.subgraphs = [subgraph]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output()), weight_values


@pytest.mark.parametrize("embedding", [False, True], ids=["fully_connected", "external_embedder"])
def test_real_small_float_casting_changes_bytes_and_preserves_fp32_io(embedding):
    pytest.importorskip("ai_edge_quantizer.quantizer")
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    from ai_edge_quantizer import quantizer
    from ir_training.export.litertlm_inspector import _tflite_graph_fingerprint

    source, weights = _tiny_model(embedding=embedding)
    qt = quantizer.Quantizer(source)
    qt.load_quantization_recipe(str(de.W16_RECIPE_PATH))
    assert not qt.need_calibration
    result = bytes(qt.quantize().quantized_model)
    graph = schema.Model.GetRootAsModel(result, 0)
    fp16_buffers = [
        graph.Buffers(graph.Subgraphs(0).Tensors(i).Buffer()).DataAsNumpy().tobytes()
        for i in range(graph.Subgraphs(0).TensorsLength())
        if graph.Subgraphs(0).Tensors(i).Type() == schema.TensorType.FLOAT16
    ]
    assert fp16_buffers == [weights.astype(np.float16).tobytes()]
    assert len(fp16_buffers[0]) == weights.nbytes // 2
    fingerprint = _tflite_graph_fingerprint(memoryview(result), include_details=True)
    # Include a separately inspected FC graph when testing the embedder: the
    # production inspector correctly refuses packages with no FC at all.
    inspection = package(["FLOAT16"]) if embedding else {"graphs": []}
    inspection["graphs"].append(fingerprint)
    precision = de.inspect_variant_precision(inspection, "e2b", "w16")
    assert set(precision["weight_type_counts"]) == {"FLOAT16"}

    outputs = []
    for content in (source, result):
        interpreter = Interpreter(model_content=content, num_threads=1,
                                  experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]
        assert input_info["dtype"] == (np.int32 if embedding else np.float32)
        assert output_info["dtype"] == np.float32
        inputs = np.array([1], np.int32) if embedding else np.array([[0.3, -0.2, 0.1]], np.float32)
        interpreter.set_tensor(input_info["index"], inputs)
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_info["index"]))
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-3, atol=2e-4)


@pytest.mark.parametrize("stored_dtype", ["FLOAT16", "FLOAT32"])
def test_conversion_binds_recipe_and_keeps_precision_gate(monkeypatch, tmp_path, stored_dtype):
    model_dir, out = tmp_path / "merged_hf", tmp_path / "w16"
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
        assert kwargs["quantization_recipe"] == str((out / "w16_quantization_recipe.json").resolve())
        assert kwargs["experimental_use_fp16"] is False
        assert kwargs["experimental_use_mixed_precision"] is False
        assert (out / "w16_quantization_recipe.json").read_bytes() == de.W16_RECIPE_PATH.read_bytes()
        (out / "model.litertlm").write_bytes(b"synthetic integration fixture")

    monkeypatch.setattr(sys.modules["litert_torch.generative.export_hf.export"], "export", export)
    import ir_training.export.litertlm_inspector as inspector
    monkeypatch.setattr(inspector, "inspect_litertlm", lambda *a, **k: package([stored_dtype], cast=True))
    if stored_dtype == "FLOAT32":
        with pytest.raises(ValueError, match="actual package.*FLOAT32"):
            de.convert_deployment_variant(profile="e2b", variant="w16", model_dir=model_dir, output_dir=out)
        assert not (out / "export_manifest.json").exists()
        assert (out / "package_inspection.json").is_file()
        assert source_bytes == {p.name: p.read_bytes() for p in model_dir.iterdir()}
        return
    result = de.convert_deployment_variant(profile="e2b", variant="w16", model_dir=model_dir, output_dir=out)
    assert result["recipe"] == de.W16_RECIPE
    assert result["quantization_recipe_file"]["sha256"] == de.file_sha256(de.W16_RECIPE_PATH)
    assert result["actual_precision"]["verified"] is True
    assert source_bytes == {p.name: p.read_bytes() for p in model_dir.iterdir()}
    plan = {"profile": "e2b", "merged_model_dir": str(model_dir),
            "variants": {"w16": {"output_dir": str(out), "artifact": str(out / "model.litertlm")}}}
    assert str(out / "w16_quantization_recipe.json") in de.validate_deployment_export_output(plan, "w16")["files"]
    (out / "w16_quantization_recipe.json").write_text("[]")
    with pytest.raises(ValueError, match="recipe changed"):
        de.validate_deployment_export_output(plan, "w16")
