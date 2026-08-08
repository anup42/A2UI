from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.export.litertlm_inspector import (  # noqa: E402
    _enum_names,
    _tflite_graph_fingerprint,
    compare_litertlm_reports,
    inspect_litertlm,
)


def _write_empty_litertlm(path: Path) -> None:
    """Write the smallest valid v1 LiteRT-LM header with no sections."""

    data = bytearray(48)
    data[0:8] = b"LITERTLM"
    struct.pack_into("<III", data, 8, 1, 5, 0)
    struct.pack_into("<Q", data, 24, 48)
    # FlatBuffer root offset at 32 -> table at 40.  The table has a four-byte
    # vtable offset and an empty four-byte vtable at 36.
    struct.pack_into("<I", data, 32, 8)
    struct.pack_into("<HH", data, 36, 4, 4)
    struct.pack_into("<i", data, 40, 4)
    path.write_bytes(data)


def test_inspect_minimal_litertlm_header(tmp_path):
    artifact = tmp_path / "empty.litertlm"
    _write_empty_litertlm(artifact)

    report = inspect_litertlm(artifact, inspect_tflite=False)

    assert report["header"]["magic"] == "LITERTLM"
    assert report["header"]["version"] == {"major": 1, "minor": 5, "patch": 0}
    assert report["sections"] == []
    assert report["weights"]["section_count"] == 0


def test_compare_reports_keeps_random_weight_claims_separate():
    def report(structural: str, quant_layout: str, quant_values: str, weight_hash: str | None):
        section = {"data_type_name": "TFLiteWeights", "size": 4, "items": []}
        if weight_hash:
            section["sha256"] = weight_hash
        return {
            "header": {"system_metadata": {"entries": []}},
            "sections": [section],
            "graphs": [
                {
                    "structural_sha256": structural,
                    "execution_contract_complete": True,
                    "execution_contract_sha256": f"contract-{structural}",
                    "quantization_layout_sha256": quant_layout,
                    "quantization_values_sha256": quant_values,
                }
            ],
        }

    left = report("same-graph", "same-layout", "official-values", "official-weights")
    random_candidate = report("same-graph", "same-layout", "random-values", "random-weights")

    comparison = compare_litertlm_reports(left, random_candidate)

    assert comparison["graph_structure_match"] is True
    assert comparison["execution_contract_complete"] is True
    assert comparison["execution_contract_match"] is True
    assert comparison["quantization_layout_match"] is True
    assert comparison["quantization_values_match"] is False
    assert comparison["weight_bytes_match"] is False
    assert "structural parity only" in comparison["interpretation"]


def test_inspector_report_is_json_serializable(tmp_path):
    artifact = tmp_path / "empty.litertlm"
    _write_empty_litertlm(artifact)
    report = inspect_litertlm(artifact, inspect_tflite=False)
    json.dumps(report)


def test_tensor_type_fallback_labels_current_low_bit_values():
    labels = _enum_names(object(), "TensorType")

    assert labels[19] == "INT2"
    assert labels[20] == "UINT4"


def _tiny_fully_connected_model(*, keep_num_dims: bool) -> bytes:
    import flatbuffers
    from ai_edge_litert import schema_py_generated as schema

    model = schema.ModelT()
    model.version = 3
    model.description = "execution-contract-test"
    opcode = schema.OperatorCodeT()
    opcode.builtinCode = schema.BuiltinOperator.FULLY_CONNECTED
    opcode.deprecatedBuiltinCode = schema.BuiltinOperator.FULLY_CONNECTED
    opcode.version = 1
    model.operatorCodes = [opcode]
    model.buffers = [schema.BufferT()]

    subgraph = schema.SubGraphT()
    subgraph.name = "main"
    subgraph.inputs = [0]
    subgraph.outputs = [2]
    subgraph.tensors = []
    for index in range(3):
        tensor = schema.TensorT()
        tensor.name = f"tensor_{index}"
        tensor.shape = [1]
        tensor.shapeSignature = [1]
        tensor.type = schema.TensorType.FLOAT32
        tensor.hasRank = True
        subgraph.tensors.append(tensor)

    operator = schema.OperatorT()
    operator.opcodeIndex = 0
    operator.inputs = [0, 1]
    operator.outputs = [2]
    operator.builtinOptionsType = schema.BuiltinOptions.FullyConnectedOptions
    options = schema.FullyConnectedOptionsT()
    options.keepNumDims = keep_num_dims
    operator.builtinOptions = options
    subgraph.operators = [operator]
    model.subgraphs = [subgraph]

    builder = flatbuffers.Builder(1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytes(builder.Output())


def test_execution_contract_detects_builtin_option_values_omitted_by_coarse_hash():
    ordinary = _tflite_graph_fingerprint(
        memoryview(_tiny_fully_connected_model(keep_num_dims=False))
    )
    keep_dims = _tflite_graph_fingerprint(
        memoryview(_tiny_fully_connected_model(keep_num_dims=True))
    )

    assert ordinary["execution_contract_complete"] is True
    assert keep_dims["execution_contract_complete"] is True
    # The legacy structure hash includes the option *type* but not its fields.
    assert ordinary["structural_sha256"] == keep_dims["structural_sha256"]
    assert (
        ordinary["execution_contract_sha256"]
        != keep_dims["execution_contract_sha256"]
    )
