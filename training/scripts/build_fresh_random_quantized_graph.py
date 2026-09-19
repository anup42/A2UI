"""Build a fresh random quantized graph from an official LiteRT-LM inventory.

Unlike ``build_random_official_topology_parity.py``, this script does not copy
the official FlatBuffer graph.  It reads only the observable weight/operator
inventory (shapes, bit widths, axes, and activation edge types), creates one
independent random branch per weight, applies the same public symmetric
per-axis quantization convention, and serializes a new TFLite FlatBuffer.

This separates two claims that are often accidentally conflated:

* ``fresh_quantization_layout_match`` means the newly built graph has the same
  observable quantized weight inventory for the selected section.
* ``graph_structure_match`` is intentionally false for a fresh graph.  Exact
  graph parity requires the original exporter topology, signatures, cache
  tensors, control flow, and metadata; copying the official graph is a
  different experiment and is handled by the topology-preserving harness.

The script is an audit fixture, not a production Gemma exporter.  It does not
train, calibrate activations, or recover Google's private QAT state.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import mmap
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    inspect_litertlm,
)
from tflite_schema_compat import schema_module  # noqa: E402


class FreshGraphParityError(RuntimeError):
    """Raised when a fresh graph cannot be built or inspected."""


_TENSOR_TYPES = {
    0: "FLOAT32",
    2: "INT32",
    9: "INT8",
    17: "INT4",
    18: "BFLOAT16",
    # LiteRT-LM uses schema extensions that are not present in older tflite
    # Python wheels.  The official mobile package observed INT2 as 19.
    19: "INT2",
    20: "UINT4",
}
_BITS_BY_TYPE = {9: 8, 17: 4, 19: 2, 20: 4}
_OPERATOR_CODES = {"FULLY_CONNECTED": 9, "EMBEDDING_LOOKUP": 7}


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _model_type(section: dict[str, Any]) -> str:
    return next(
        (
            str(item.get("value"))
            for item in section.get("items", [])
            if item.get("key") == "model_type"
        ),
        "",
    )


def _section_by_model_type(package: dict[str, Any], model_type: str) -> dict[str, Any]:
    matches = [
        section
        for section in package.get("sections", [])
        if section.get("data_type_name") == "TFLiteModel"
        and _model_type(section) == model_type
    ]
    if not matches:
        available = [
            _model_type(section)
            for section in package.get("sections", [])
            if section.get("data_type_name") == "TFLiteModel"
        ]
        raise FreshGraphParityError(
            f"No TFLite model_type={model_type!r} section. Available: {available}"
        )
    return matches[0]


def _operator_names() -> dict[int, str]:
    try:
        module = schema_module("BuiltinOperator")
        cls = getattr(module, "BuiltinOperator")
    except (ImportError, AttributeError) as exc:
        raise FreshGraphParityError(
            "Fresh graph parity requires generated tflite schema bindings."
        ) from exc
    return {
        int(value): name
        for name, value in vars(cls).items()
        if not name.startswith("_") and isinstance(value, int)
    }


def _schema_model(data: Any) -> Any:
    try:
        module = schema_module("Model")
        model_cls = getattr(module, "Model")
        getter = getattr(model_cls, "GetRootAsModel", None) or getattr(model_cls, "GetRootAs")
        return getter(data, 0)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise FreshGraphParityError(
            "Fresh graph parity requires generated tflite schema bindings."
        ) from exc


def _quantization_record(tensor: Any) -> dict[str, Any] | None:
    quantization = tensor.Quantization()
    if quantization is None:
        return None
    scale_count = int(quantization.ScaleLength() or 0)
    zero_count = int(quantization.ZeroPointLength() or 0)
    if not scale_count and not zero_count:
        return None
    zero_points = [int(quantization.ZeroPoint(i)) for i in range(zero_count)]
    return {
        "scale_count": scale_count,
        "zero_point_count": zero_count,
        "quantized_dimension": int(quantization.QuantizedDimension()),
        "zero_points_all_zero": bool(zero_points) and all(value == 0 for value in zero_points),
    }


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(tensor.Shape(i)) for i in range(int(tensor.ShapeLength() or 0)))


def _extract_inventory(
    artifact: str | Path,
    model_type: str,
    *,
    include_embeddings: bool,
    max_weights: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the selected TFLite section and its unique weight records.

    The first result is not the full package inspection report. To locate
    sibling sections (such as MTP), inspect the package separately.
    """
    artifact_path = Path(artifact).expanduser().resolve()
    if not artifact_path.is_file():
        raise FreshGraphParityError(f"Official artifact does not exist: {artifact_path}")
    try:
        package = inspect_litertlm(artifact_path, inspect_tflite=False)
        section = _section_by_model_type(package, model_type)
    except (OSError, LiteRTLMInspectionError) as exc:
        raise FreshGraphParityError(f"Could not inspect official artifact: {exc}") from exc
    op_names = _operator_names()
    records: list[dict[str, Any]] = []
    seen_buffers: set[int] = set()
    with artifact_path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            section_view = memoryview(mapped)[section["begin_offset"] : section["end_offset"]]
            model = _schema_model(section_view)
            for subgraph_index in range(int(model.SubgraphsLength() or 0)):
                subgraph = model.Subgraphs(subgraph_index)
                for operator_index in range(int(subgraph.OperatorsLength() or 0)):
                    operator = subgraph.Operators(operator_index)
                    code = model.OperatorCodes(operator.OpcodeIndex())
                    operator_name = op_names.get(int(code.BuiltinCode()), "")
                    if operator_name not in {"FULLY_CONNECTED", "EMBEDDING_LOOKUP"}:
                        continue
                    if operator_name == "EMBEDDING_LOOKUP" and not include_embeddings:
                        continue
                    if int(operator.InputsLength() or 0) < 2:
                        continue
                    tensor = subgraph.Tensors(operator.Inputs(1))
                    buffer_index = int(tensor.Buffer())
                    if buffer_index in seen_buffers:
                        continue
                    seen_buffers.add(buffer_index)
                    type_value = int(tensor.Type())
                    bits = _BITS_BY_TYPE.get(type_value)
                    quantization = _quantization_record(tensor)
                    shape = _shape(tensor)
                    if bits is None or quantization is None or len(shape) < 2:
                        continue
                    if max_weights is not None and len(records) >= max_weights:
                        break
                    input_type = None
                    input_quantization = None
                    if int(operator.InputsLength() or 0) >= 1:
                        input_tensor = subgraph.Tensors(operator.Inputs(0))
                        input_type = int(input_tensor.Type())
                        input_quantization = _quantization_record(input_tensor)
                    output_type = None
                    output_quantization = None
                    if int(operator.OutputsLength() or 0) >= 1:
                        output_tensor = subgraph.Tensors(operator.Outputs(0))
                        output_type = int(output_tensor.Type())
                        output_quantization = _quantization_record(output_tensor)
                    records.append(
                        {
                            "ordinal": len(records),
                            "operator": operator_name,
                            "official_subgraph": subgraph_index,
                            "official_operator_index": operator_index,
                            "official_buffer": buffer_index,
                            "type_value": type_value,
                            "type_name": _TENSOR_TYPES.get(type_value, f"unknown:{type_value}"),
                            "bits": bits,
                            "shape": list(shape),
                            "scale_count": quantization["scale_count"],
                            "zero_point_count": quantization["zero_point_count"],
                            "quantized_dimension": quantization["quantized_dimension"],
                            "zero_points_all_zero": quantization["zero_points_all_zero"],
                            "input_type_value": input_type,
                            "input_type_name": _TENSOR_TYPES.get(input_type, None),
                            "input_quantization": input_quantization,
                            "output_type_value": output_type,
                            "output_type_name": _TENSOR_TYPES.get(output_type, None),
                            "output_quantization": output_quantization,
                            "official_tensor_name": _decode(tensor.Name()),
                        }
                    )
                if max_weights is not None and len(records) >= max_weights:
                    break
        finally:
            section_view.release()
            mapped.close()
    return section, records


def _pack_low_bit(values: np.ndarray, bits: int) -> bytes:
    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    padding = (-len(flat)) % values_per_byte
    if padding:
        flat = np.pad(flat, (0, padding), mode="constant")
    unsigned = flat.astype(np.int16) & mask
    packed = np.zeros(len(flat) // values_per_byte, dtype=np.uint8)
    for offset in range(values_per_byte):
        packed |= unsigned[offset::values_per_byte].astype(np.uint8) << (offset * bits)
    return packed.tobytes()


def _random_quantized_weight(shape: tuple[int, ...], bits: int, seed: int, ordinal: int) -> tuple[bytes, np.ndarray]:
    if len(shape) < 2 or bits not in (2, 4, 8):
        raise FreshGraphParityError(f"Unsupported random weight shape/bits: {shape}, W{bits}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
    weights = rng.standard_normal(shape, dtype=np.float32)
    reduce_axes = tuple(range(1, len(shape)))
    qmax = (1 << (bits - 1)) - 1
    qmin = -127 if bits == 8 else -(1 << (bits - 1))
    bounds = np.maximum(np.max(np.abs(weights), axis=reduce_axes), np.float32(1e-9))
    scales = (bounds / np.float32(qmax)).astype(np.float32, copy=False)
    ratio = weights / scales.reshape((-1,) + (1,) * len(reduce_axes))
    quantized = np.clip(np.rint(ratio), qmin, qmax).astype(np.int8, copy=False)
    return _pack_low_bit(quantized, bits), scales


def _vector(builder: Any, values: Iterable[Any], prepend: str) -> int:
    values = list(values)
    size = {"Int32": 4, "Int64": 8, "Float32": 4, "UOffset": 4, "Uint8": 1}[prepend]
    builder.StartVector(size, len(values), size)
    method_name = "PrependUOffsetTRelative" if prepend == "UOffset" else f"Prepend{prepend}"
    method = getattr(builder, method_name)
    for value in reversed(values):
        method(value)
    return builder.EndVector()


def _build_fresh_tflite(records: list[dict[str, Any]], seed: int) -> bytes:
    try:
        import flatbuffers
        Buffer = schema_module("Buffer")
        Model = schema_module("Model")
        Operator = schema_module("Operator")
        OperatorCode = schema_module("OperatorCode")
        QuantizationParameters = schema_module("QuantizationParameters")
        SubGraph = schema_module("SubGraph")
        Tensor = schema_module("Tensor")
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise FreshGraphParityError("Fresh graph serialization requires flatbuffers and generated tflite bindings.") from exc

    builder = flatbuffers.Builder(1024)
    empty_buffer = Buffer.BufferStart(builder)
    empty_buffer = Buffer.BufferEnd(builder)
    buffer_offsets = [empty_buffer]
    tensor_offsets: list[int] = []
    operator_offsets: list[int] = []
    subgraph_inputs: list[int] = []
    subgraph_outputs: list[int] = []
    operator_code_names: list[str] = []

    def add_tensor(
        shape: list[int],
        type_value: int,
        buffer_index: int,
        name: str,
        quantization: dict[str, Any] | None,
    ) -> int:
        shape_offset = _vector(builder, shape, "Int32")
        name_offset = builder.CreateString(name)
        quant_offset = 0
        if quantization is not None:
            scales = quantization.get("scales", [])
            zero_points = quantization.get("zero_points", [])
            scale_offset = _vector(builder, scales, "Float32")
            zero_offset = _vector(builder, zero_points, "Int64")
            QuantizationParameters.QuantizationParametersStart(builder)
            QuantizationParameters.QuantizationParametersAddScale(builder, scale_offset)
            QuantizationParameters.QuantizationParametersAddZeroPoint(builder, zero_offset)
            QuantizationParameters.QuantizationParametersAddQuantizedDimension(
                builder, int(quantization.get("quantized_dimension", 0))
            )
            quant_offset = QuantizationParameters.QuantizationParametersEnd(builder)
        Tensor.TensorStart(builder)
        Tensor.TensorAddShape(builder, shape_offset)
        Tensor.TensorAddType(builder, int(type_value))
        Tensor.TensorAddBuffer(builder, int(buffer_index))
        Tensor.TensorAddName(builder, name_offset)
        if quant_offset:
            Tensor.TensorAddQuantization(builder, quant_offset)
        return Tensor.TensorEnd(builder)

    for record in records:
        ordinal = int(record["ordinal"])
        shape = tuple(int(value) for value in record["shape"])
        bits = int(record["bits"])
        weight_bytes, scales = _random_quantized_weight(shape, bits, seed, ordinal)
        weight_data_offset = builder.CreateByteVector(weight_bytes)
        Buffer.BufferStart(builder)
        Buffer.BufferAddData(builder, weight_data_offset)
        weight_buffer = Buffer.BufferEnd(builder)
        buffer_offsets.append(weight_buffer)
        weight_buffer_index = len(buffer_offsets) - 1

        if record["operator"] == "EMBEDDING_LOOKUP":
            input_shape = [1]
            input_type = 2
        else:
            input_shape = [1, shape[-1]]
            input_type = int(record.get("input_type_value") or 0)
        output_type = int(record.get("output_type_value") or 0)
        # ``EMBEDDING_LOOKUP`` returns one row per lookup, so its output width
        # is the second weight dimension.  Fully-connected branches expose the
        # usual one-row output whose width is the first dimension.  Keeping
        # this distinction makes the freshly-built graph semantically valid,
        # rather than merely matching the weight inventory.
        output_shape = [1, shape[1] if record["operator"] == "EMBEDDING_LOOKUP" else shape[0]]
        input_quant = None
        output_quant = None
        if input_type in _BITS_BY_TYPE:
            input_quant = {"scales": [1.0], "zero_points": [0], "quantized_dimension": 0}
        if output_type in _BITS_BY_TYPE:
            output_quant = {"scales": [1.0], "zero_points": [0], "quantized_dimension": 0}
        input_index = len(tensor_offsets)
        tensor_offsets.append(
            add_tensor(input_shape, input_type, 0, f"fresh/{ordinal}/input", input_quant)
        )
        weight_index = len(tensor_offsets)
        tensor_offsets.append(
            add_tensor(
                list(shape),
                int(record["type_value"]),
                weight_buffer_index,
                f"fresh/{ordinal}/weight/{record['operator']}",
                {
                    "scales": scales.tolist(),
                    "zero_points": [0] * len(scales),
                    "quantized_dimension": 0,
                },
            )
        )
        output_index = len(tensor_offsets)
        tensor_offsets.append(
            add_tensor(output_shape, output_type, 0, f"fresh/{ordinal}/output", output_quant)
        )
        subgraph_inputs.append(input_index)
        subgraph_outputs.append(output_index)
        input_vector = _vector(builder, [input_index, weight_index, -1], "Int32")
        output_vector = _vector(builder, [output_index], "Int32")
        opcode_name = record["operator"]
        if opcode_name not in operator_code_names:
            operator_code_names.append(opcode_name)
        opcode_index = operator_code_names.index(opcode_name)
        Operator.OperatorStart(builder)
        Operator.OperatorAddOpcodeIndex(builder, opcode_index)
        Operator.OperatorAddInputs(builder, input_vector)
        Operator.OperatorAddOutputs(builder, output_vector)
        operator_offsets.append(Operator.OperatorEnd(builder))

    tensors_vector = _vector(builder, tensor_offsets, "UOffset")
    inputs_vector = _vector(builder, subgraph_inputs, "Int32")
    outputs_vector = _vector(builder, subgraph_outputs, "Int32")
    operators_vector = _vector(builder, operator_offsets, "UOffset")
    subgraph_name = builder.CreateString("fresh_random_quantized_graph")
    SubGraph.SubGraphStart(builder)
    SubGraph.SubGraphAddTensors(builder, tensors_vector)
    SubGraph.SubGraphAddInputs(builder, inputs_vector)
    SubGraph.SubGraphAddOutputs(builder, outputs_vector)
    SubGraph.SubGraphAddOperators(builder, operators_vector)
    SubGraph.SubGraphAddName(builder, subgraph_name)
    subgraph_offset = SubGraph.SubGraphEnd(builder)
    subgraphs_vector = _vector(builder, [subgraph_offset], "UOffset")

    opcode_offsets: list[int] = []
    for name in operator_code_names:
        OperatorCode.OperatorCodeStart(builder)
        # Older generated bindings resolve BuiltinCode through the deprecated
        # int8 field whenever the newer int32 field is below the extension
        # sentinel.  Populate both fields so the FlatBuffer is readable by
        # old and new schema bindings.
        OperatorCode.OperatorCodeAddDeprecatedBuiltinCode(builder, _OPERATOR_CODES[name])
        OperatorCode.OperatorCodeAddBuiltinCode(builder, _OPERATOR_CODES[name])
        OperatorCode.OperatorCodeAddVersion(builder, 1)
        opcode_offsets.append(OperatorCode.OperatorCodeEnd(builder))
    opcodes_vector = _vector(builder, opcode_offsets, "UOffset")
    buffers_vector = _vector(builder, buffer_offsets, "UOffset")
    description = builder.CreateString("Fresh random quantized graph; not an official exporter output")
    Model.ModelStart(builder)
    Model.ModelAddVersion(builder, 3)
    Model.ModelAddOperatorCodes(builder, opcodes_vector)
    Model.ModelAddSubgraphs(builder, subgraphs_vector)
    Model.ModelAddDescription(builder, description)
    Model.ModelAddBuffers(builder, buffers_vector)
    model_offset = Model.ModelEnd(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    return bytes(builder.Output())


def _inventory_digest(records: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "operator": record["operator"],
            "type_name": record["type_name"],
            "bits": record["bits"],
            "shape": record["shape"],
            "scale_count": record["scale_count"],
            "zero_point_count": record["zero_point_count"],
            "quantized_dimension": record["quantized_dimension"],
            "input_type_name": record.get("input_type_name"),
            "output_type_name": record.get("output_type_name"),
        }
        for record in records
    ]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


def _fresh_inventory(path: Path) -> list[dict[str, Any]]:
    model = _schema_model(path.read_bytes())
    op_names = _operator_names()
    subgraph = model.Subgraphs(0)
    records: list[dict[str, Any]] = []
    for operator_index in range(int(subgraph.OperatorsLength() or 0)):
        operator = subgraph.Operators(operator_index)
        code = model.OperatorCodes(operator.OpcodeIndex())
        operator_name = op_names.get(int(code.BuiltinCode()), "")
        if operator_name not in {"FULLY_CONNECTED", "EMBEDDING_LOOKUP"}:
            continue
        if int(operator.InputsLength() or 0) < 2:
            continue
        weight = subgraph.Tensors(operator.Inputs(1))
        type_value = int(weight.Type())
        quantization = _quantization_record(weight)
        if type_value not in _BITS_BY_TYPE or quantization is None:
            continue
        input_tensor = subgraph.Tensors(operator.Inputs(0))
        output_tensor = subgraph.Tensors(operator.Outputs(0))
        records.append(
            {
                "operator": operator_name,
                "type_name": _TENSOR_TYPES.get(type_value, f"unknown:{type_value}"),
                "bits": _BITS_BY_TYPE[type_value],
                "shape": list(_shape(weight)),
                "scale_count": quantization["scale_count"],
                "zero_point_count": quantization["zero_point_count"],
                "quantized_dimension": quantization["quantized_dimension"],
                "input_type_name": _TENSOR_TYPES.get(int(input_tensor.Type())),
                "output_type_name": _TENSOR_TYPES.get(int(output_tensor.Type())),
            }
        )
    return records


def run(
    artifact: str | Path,
    *,
    model_type: str,
    seed: int,
    include_embeddings: bool,
    max_weights: int | None,
    output: str | Path | None,
) -> dict[str, Any]:
    section, official_records = _extract_inventory(
        artifact,
        model_type,
        include_embeddings=include_embeddings,
        max_weights=max_weights,
    )
    tflite_bytes = _build_fresh_tflite(official_records, seed)
    fresh_records = None
    if output:
        output_path = Path(output).expanduser().resolve()
        if output_path.exists():
            raise FreshGraphParityError(f"Refusing to overwrite existing output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tflite_bytes)
        fresh_records = _fresh_inventory(output_path)
    else:
        # Parsing bytes directly keeps the default audit in-memory and avoids
        # writing multi-hundred-megabyte random graphs by accident.
        model = _schema_model(tflite_bytes)
        op_names = _operator_names()
        subgraph = model.Subgraphs(0)
        fresh_records = []
        for operator_index in range(int(subgraph.OperatorsLength() or 0)):
            operator = subgraph.Operators(operator_index)
            code = model.OperatorCodes(operator.OpcodeIndex())
            operator_name = op_names.get(int(code.BuiltinCode()), "")
            if operator_name not in {"FULLY_CONNECTED", "EMBEDDING_LOOKUP"}:
                continue
            weight = subgraph.Tensors(operator.Inputs(1))
            type_value = int(weight.Type())
            quantization = _quantization_record(weight)
            input_tensor = subgraph.Tensors(operator.Inputs(0))
            output_tensor = subgraph.Tensors(operator.Outputs(0))
            fresh_records.append(
                {
                    "operator": operator_name,
                    "type_name": _TENSOR_TYPES.get(type_value, f"unknown:{type_value}"),
                    "bits": _BITS_BY_TYPE.get(type_value),
                    "shape": list(_shape(weight)),
                    "scale_count": quantization["scale_count"] if quantization else 0,
                    "zero_point_count": quantization["zero_point_count"] if quantization else 0,
                    "quantized_dimension": quantization["quantized_dimension"] if quantization else 0,
                    "input_type_name": _TENSOR_TYPES.get(int(input_tensor.Type())),
                    "output_type_name": _TENSOR_TYPES.get(int(output_tensor.Type())),
                }
            )
    official_digest = _inventory_digest(official_records)
    fresh_digest = _inventory_digest(fresh_records)
    official_layout = collections.Counter(
        (record["operator"], record["type_name"], tuple(record["shape"]), record["scale_count"], record["quantized_dimension"])
        for record in official_records
    )
    fresh_layout = collections.Counter(
        (record["operator"], record["type_name"], tuple(record["shape"]), record["scale_count"], record["quantized_dimension"])
        for record in fresh_records
    )
    report = {
        "artifact": str(Path(artifact).expanduser().resolve()),
        "model_type": _model_type(section),
        "seed": int(seed),
        "include_embeddings": bool(include_embeddings),
        "max_weights": max_weights,
        "random_initialization": True,
        "training_executed": False,
        "private_qat_recipe_recovered": False,
        "official_inventory_count": len(official_records),
        "fresh_inventory_count": len(fresh_records),
        "official_inventory_digest": official_digest,
        "fresh_inventory_digest": fresh_digest,
        "fresh_quantization_layout_match": bool(official_layout == fresh_layout),
        "graph_structure_match": False,
        "exact_official_model_match": False,
        "fresh_tflite_size": len(tflite_bytes),
        "layout_difference": {
            "official_only": {str(key): count for key, count in (official_layout - fresh_layout).items()},
            "fresh_only": {str(key): count for key, count in (fresh_layout - official_layout).items()},
        },
        "interpretation": (
            "Fresh random branches reproduce the selected official quantized weight inventory, "
            "but intentionally do not reproduce graph structure, signatures, cache tensors, "
            "metadata, learned values, or private QAT state."
            if official_layout == fresh_layout
            else "The fresh graph does not yet reproduce the selected official weight inventory; "
            "inspect layout_difference before attempting exporter changes."
        ),
    }
    if output:
        Path(output).with_suffix(".json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fresh random quantized graph from an official LiteRT-LM inventory.")
    parser.add_argument("artifact", help="Official .litertlm artifact.")
    parser.add_argument("--model-type", default="TF_LITE_PREFILL_DECODE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument("--max-weights", type=int, help="Optional cap for a bounded graph audit.")
    parser.add_argument("--output", type=Path, help="Optional new .tflite output; default is in-memory only.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_weights is not None and args.max_weights < 1:
        parser.error("--max-weights must be positive")
    try:
        report = run(
            args.artifact,
            model_type=args.model_type,
            seed=args.seed,
            include_embeddings=args.include_embeddings,
            max_weights=args.max_weights,
            output=args.output,
        )
    except (OSError, FreshGraphParityError, LiteRTLMInspectionError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
