from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

flatbuffers = pytest.importorskip("flatbuffers")
schema = pytest.importorskip("ai_edge_litert.schema_py_generated")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.export import full_parameter_serialization as fps


def _checkpoint(root: Path, tensors: dict[str, bytes | tuple[list[int], bytes]]) -> None:
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "gemma4_text"}))
    offset = 0
    header = {}
    payload = b""
    for key, item in tensors.items():
        shape, value = item if isinstance(item, tuple) else ([len(item) // 4], item)
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(value)]}
        offset += len(value)
        payload += value
    encoded = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _vector(builder, values, start):
    start(builder, len(values))
    for value in reversed(values):
        builder.PrependUOffsetTRelative(value)
    return builder.EndVector()


def _ints(builder, values, start):
    start(builder, len(values))
    for value in reversed(values):
        builder.PrependInt32(value)
    return builder.EndVector()


def _real_tflite(path: Path, tensors: list[dict], *, consumed: list[int] | None = None,
                 duplicate_subgraph: bool = False, _external_base: int | None = None) -> None:
    """Write a real schema-generated FlatBuffer; no physical-record mocking."""
    builder = flatbuffers.Builder(4096)
    buffer_offsets = []
    external_cursor = _external_base or 1
    for item in [{}] + tensors:
        payload = item.get("payload", b"")
        external = bool(item.get("external"))
        data = builder.CreateByteVector(payload) if payload and not external else 0
        schema.BufferStart(builder)
        if data:
            schema.BufferAddData(builder, data)
        if external:
            schema.BufferAddOffset(builder, external_cursor)
            schema.BufferAddSize(builder, len(payload))
            external_cursor += len(payload)
        buffer_offsets.append(schema.BufferEnd(builder))

    tensor_offsets = []
    for index, item in enumerate(tensors, start=1):
        name = builder.CreateString(item.get("name", f"tensor_{index}"))
        shape = _ints(builder, item["shape"], schema.TensorStartShapeVector)
        quant = 0
        if "scales" in item:
            scales = builder.CreateNumpyVector(np.asarray(item["scales"], dtype=np.float32))
            zeros = builder.CreateNumpyVector(np.asarray(item.get("zero_points", []), dtype=np.int64))
            schema.QuantizationParametersStart(builder)
            schema.QuantizationParametersAddScale(builder, scales)
            schema.QuantizationParametersAddZeroPoint(builder, zeros)
            schema.QuantizationParametersAddQuantizedDimension(
                builder, item.get("quantized_dimension", 0)
            )
            quant = schema.QuantizationParametersEnd(builder)
        schema.TensorStart(builder)
        schema.TensorAddShape(builder, shape)
        schema.TensorAddType(builder, item.get("dtype", schema.TensorType.FLOAT32))
        schema.TensorAddBuffer(builder, index)
        schema.TensorAddName(builder, name)
        if quant:
            schema.TensorAddQuantization(builder, quant)
        tensor_offsets.append(schema.TensorEnd(builder))

    inputs = consumed if consumed is not None else list(range(len(tensors)))
    op_inputs = _ints(builder, inputs, schema.OperatorStartInputsVector)
    schema.OperatorStart(builder)
    schema.OperatorAddOpcodeIndex(builder, 0)
    schema.OperatorAddInputs(builder, op_inputs)
    operator = schema.OperatorEnd(builder)

    def make_subgraph(name_value):
        name = builder.CreateString(name_value)
        tensor_vector = _vector(builder, tensor_offsets, schema.SubGraphStartTensorsVector)
        operators = _vector(builder, [operator], schema.SubGraphStartOperatorsVector)
        schema.SubGraphStart(builder)
        schema.SubGraphAddTensors(builder, tensor_vector)
        schema.SubGraphAddOperators(builder, operators)
        schema.SubGraphAddName(builder, name)
        return schema.SubGraphEnd(builder)

    subgraphs = [make_subgraph("prefill")]
    if duplicate_subgraph:
        subgraphs.append(make_subgraph("decode"))
    subgraph_vector = _vector(builder, subgraphs, schema.ModelStartSubgraphsVector)
    buffers = _vector(builder, buffer_offsets, schema.ModelStartBuffersVector)
    schema.OperatorCodeStart(builder)
    schema.OperatorCodeAddBuiltinCode(builder, schema.BuiltinOperator.ADD)
    opcode = schema.OperatorCodeEnd(builder)
    opcodes = _vector(builder, [opcode], schema.ModelStartOperatorCodesVector)
    schema.ModelStart(builder)
    schema.ModelAddVersion(builder, 3)
    schema.ModelAddOperatorCodes(builder, opcodes)
    schema.ModelAddSubgraphs(builder, subgraph_vector)
    schema.ModelAddBuffers(builder, buffers)
    model = schema.ModelEnd(builder)
    builder.Finish(model, file_identifier=b"TFL3")
    output = bytes(builder.Output())
    external_payloads = [item["payload"] for item in tensors if item.get("external")]
    if external_payloads and _external_base is None:
        _real_tflite(path, tensors, consumed=consumed, duplicate_subgraph=duplicate_subgraph,
                     _external_base=len(output))
        return
    path.write_bytes(output + b"".join(external_payloads))


def _paths(root: Path, prefix: str) -> dict[str, Path]:
    return {role: root / f"{prefix}_{role}.tflite" for role in fps.SECTION_ROLES}


def _write_empty_sections(paths: dict[str, Path]) -> None:
    for path in paths.values():
        _real_tflite(path, [])


def test_checkpoint_inventory_is_exact_fp32_and_role_aware(tmp_path):
    _checkpoint(tmp_path, {
        "model.embed_tokens.weight": struct.pack("<ff", 1, 2),
        "model.layers.0.input_layernorm.weight": struct.pack("<ff", 3, 4),
    })
    inventory = fps.build_checkpoint_inventory(tmp_path, expected_count=2)
    assert inventory["model.embed_tokens.weight"]["section_role"] == "token_embedder"
    assert inventory["model.layers.0.input_layernorm.weight"]["section_role"] == "target"


def test_checkpoint_inventory_rejects_wrong_model_type(tmp_path):
    _checkpoint(tmp_path, {"x": struct.pack("<f", 1)})
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    with pytest.raises(fps.FullParameterSerializationError, match="gemma4_text"):
        fps.build_checkpoint_inventory(tmp_path, expected_count=1)


def test_real_flatbuffer_float_audit_covers_all_roles_and_duplicate_decode(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    tensors = {
        "model.embed_tokens.weight": ([2, 2], struct.pack("<4f", 1, 2, 3, 4)),
        "model.embed_tokens_per_layer.weight": ([2, 2], struct.pack("<4f", 5, 6, 7, 8)),
        "model.layers.0.input_layernorm.weight": struct.pack("<2f", 9, 10),
    }
    _checkpoint(checkpoint, tensors)
    inventory = fps.build_checkpoint_inventory(checkpoint, expected_count=3)
    paths = _paths(tmp_path, "float")
    _real_tflite(paths["token_embedder"], [{
        "name": "model.embed_tokens.weight", "shape": [2, 2],
        "payload": tensors["model.embed_tokens.weight"][1],
    }])
    _real_tflite(paths["per_layer_embedder"], [{
        "name": "model.embed_tokens_per_layer.weight", "shape": [2, 2],
        "payload": tensors["model.embed_tokens_per_layer.weight"][1],
    }])
    _real_tflite(paths["target"], [{
        "name": "Gemma4TextDecoderLayer_0/Gemma4RMSNorm_input_layernorm/weight",
        "shape": [2], "payload": tensors["model.layers.0.input_layernorm.weight"],
    }], duplicate_subgraph=True)
    report = fps.audit_float_sections(paths, inventory)
    assert report["verified"] is True
    assert report["mappings"]["model.layers.0.input_layernorm.weight"]["copy_count"] == 2


def test_real_flatbuffer_float_audit_rejects_missing_embedding(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint, {
        "model.embed_tokens.weight": ([1, 2], struct.pack("<2f", 1, 2)),
    })
    inventory = fps.build_checkpoint_inventory(checkpoint, expected_count=1)
    paths = _paths(tmp_path, "missing")
    _write_empty_sections(paths)
    with pytest.raises(fps.FullParameterSerializationError, match="missing|changed"):
        fps.audit_float_sections(paths, inventory)


def test_real_flatbuffer_orphan_constant_is_not_physical_evidence(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    payload = struct.pack("<2f", 3, 4)
    _checkpoint(checkpoint, {"model.layers.0.input_layernorm.weight": payload})
    inventory = fps.build_checkpoint_inventory(checkpoint, expected_count=1)
    paths = _paths(tmp_path, "orphan")
    _write_empty_sections(paths)
    _real_tflite(paths["target"], [{
        "name": "model.layers.0.input_layernorm.weight", "shape": [2], "payload": payload,
    }], consumed=[])
    with pytest.raises(fps.FullParameterSerializationError, match="missing|changed"):
        fps.audit_float_sections(paths, inventory)


def test_real_flatbuffer_reads_consumed_external_buffer_offset_and_size(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    payload = struct.pack("<2f", 6, 7)
    key = "model.layers.0.input_layernorm.weight"
    _checkpoint(checkpoint, {key: payload})
    inventory = fps.build_checkpoint_inventory(checkpoint, expected_count=1)
    paths = _paths(tmp_path, "external")
    _write_empty_sections(paths)
    _real_tflite(paths["target"], [{
        "name": key, "shape": [2], "payload": payload, "external": True,
    }])
    report = fps.audit_float_sections(paths, inventory)
    assert report["verified"] is True


def test_real_flatbuffer_rejects_wrong_layer_swap_even_when_values_exist(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    layer0 = struct.pack("<2f", 11, 12)
    layer1 = struct.pack("<2f", 13, 14)
    _checkpoint(checkpoint, {
        "model.layers.0.input_layernorm.weight": layer0,
        "model.layers.1.input_layernorm.weight": layer1,
    })
    inventory = fps.build_checkpoint_inventory(checkpoint, expected_count=2)
    paths = _paths(tmp_path, "swapped")
    _write_empty_sections(paths)
    _real_tflite(paths["target"], [
        {"name": "Gemma4TextDecoderLayer_1/Gemma4RMSNorm_input_layernorm/weight",
         "shape": [2], "payload": layer0},
        {"name": "Gemma4TextDecoderLayer_0/Gemma4RMSNorm_input_layernorm/weight",
         "shape": [2], "payload": layer1},
    ])
    with pytest.raises(fps.FullParameterSerializationError, match="scope|location|layer"):
        fps.audit_float_sections(paths, inventory)


def test_real_flatbuffer_same_value_norms_are_disambiguated_by_scope(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    payload = struct.pack("<2f", 15, 16)
    _checkpoint(checkpoint, {
        "model.layers.0.input_layernorm.weight": payload,
        "model.layers.1.input_layernorm.weight": payload,
    })
    inventory = fps.build_checkpoint_inventory(checkpoint, expected_count=2)
    paths = _paths(tmp_path, "same_norm")
    _write_empty_sections(paths)
    _real_tflite(paths["target"], [
        {"name": "Gemma4TextDecoderLayer_0/Gemma4RMSNorm_input_layernorm/weight",
         "shape": [2], "payload": payload},
        {"name": "Gemma4TextDecoderLayer_1/Gemma4RMSNorm_input_layernorm/weight",
         "shape": [2], "payload": payload},
    ])
    report = fps.audit_float_sections(paths, inventory)
    assert report["verified"] is True
    assert report["parameter_count"] == 2


def test_real_flatbuffer_rejects_bad_duplicate_in_same_parameter_scope(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    payload = struct.pack("<2f", 17, 18)
    _checkpoint(checkpoint, {"model.layers.0.input_layernorm.weight": payload})
    inventory = fps.build_checkpoint_inventory(checkpoint, expected_count=1)
    paths = _paths(tmp_path, "bad_copy")
    _write_empty_sections(paths)
    _real_tflite(paths["target"], [
        {"name": "Gemma4TextDecoderLayer_0/Gemma4RMSNorm_input_layernorm/prefill_weight",
         "shape": [2], "payload": payload},
        {"name": "Gemma4TextDecoderLayer_0/Gemma4RMSNorm_input_layernorm/decode_weight",
         "shape": [2], "payload": struct.pack("<2f", 17, 19)},
    ])
    with pytest.raises(fps.FullParameterSerializationError, match="copy|scope|differs"):
        fps.audit_float_sections(paths, inventory)


@pytest.mark.parametrize("corruption", ["codes", "scales"])
def test_real_flatbuffer_quantized_audit_recomputes_codes_and_scales(tmp_path, corruption):
    pytest.importorskip("ai_edge_quantizer")
    checkpoint = tmp_path / "checkpoint"
    matrix = np.asarray([[-2.0, -0.5, 0.5, 2.0]], dtype="<f4")
    norm = np.asarray([0.75, 1.25], dtype="<f4")
    key = "model.layers.0.self_attn.q_proj.weight"
    _checkpoint(checkpoint, {
        key: ([1, 4], matrix.tobytes()),
        "model.layers.0.input_layernorm.weight": norm.tobytes(),
    })
    inventory = fps.build_checkpoint_inventory(checkpoint, expected_count=2)
    float_paths = _paths(tmp_path, "qfloat")
    quant_paths = _paths(tmp_path, "quant")
    _write_empty_sections(float_paths)
    _write_empty_sections(quant_paths)
    _real_tflite(float_paths["target"], [
        {"name": key, "shape": [1, 4], "payload": matrix.tobytes()},
        {"name": "Gemma4TextDecoderLayer_0/Gemma4RMSNorm_input_layernorm/weight",
         "shape": [2], "payload": norm.tobytes()},
    ])
    expected_codes, expected_scales = fps._pinned_aeq_expected(matrix.tobytes(), [1, 4], 4)
    if corruption == "codes":
        expected_codes = bytes([expected_codes[0] ^ 1]) + expected_codes[1:]
    else:
        expected_scales = expected_scales.copy()
        expected_scales[0] = np.nextafter(expected_scales[0], np.float32(np.inf))
    _real_tflite(quant_paths["target"], [
        {"name": key, "shape": [1, 4], "payload": expected_codes,
         "dtype": schema.TensorType.INT4, "scales": expected_scales, "zero_points": [0]},
        {"name": "Gemma4TextDecoderLayer_0/Gemma4RMSNorm_input_layernorm/weight",
         "shape": [2], "payload": norm.tobytes()},
    ])
    with pytest.raises(fps.FullParameterSerializationError, match="quantization audit failed"):
        fps.audit_quantized_sections(float_paths, quant_paths, inventory)


@pytest.mark.parametrize(("key", "role", "bits", "dtype"), [
    ("model.embed_tokens.weight", "token_embedder", 2, schema.TensorType.INT2),
    ("model.layers.0.self_attn.q_proj.weight", "target", 4, schema.TensorType.INT4),
    ("model.layers.0.per_layer_projection.weight", "target", 8, schema.TensorType.INT8),
])
def test_real_flatbuffer_quantized_audit_accepts_recomputed_w248(
    tmp_path, key, role, bits, dtype,
):
    pytest.importorskip("ai_edge_quantizer")
    checkpoint = tmp_path / "checkpoint"
    matrix = np.asarray([[-2.0, -0.5, 0.5, 2.0]], dtype="<f4")
    _checkpoint(checkpoint, {key: ([1, 4], matrix.tobytes())})
    inventory = fps.build_checkpoint_inventory(checkpoint, expected_count=1)
    float_paths = _paths(tmp_path, "good_float")
    quant_paths = _paths(tmp_path, "good_quant")
    _write_empty_sections(float_paths)
    _write_empty_sections(quant_paths)
    _real_tflite(float_paths[role], [{
        "name": key, "shape": [1, 4], "payload": matrix.tobytes(),
    }])
    codes, scales = fps._pinned_aeq_expected(matrix.tobytes(), [1, 4], bits)
    _real_tflite(quant_paths[role], [{
        "name": key, "shape": [1, 4], "payload": codes,
        "dtype": dtype, "scales": scales, "zero_points": [0],
    }])
    report = fps.audit_quantized_sections(float_paths, quant_paths, inventory)
    assert report["verified"] is True
