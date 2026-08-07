"""Quantize an independent random float graph using an official inventory.

This is the converter-driven companion to ``build_random_official_topology_parity``.
The topology-preserving harness patches already-quantized bytes in the official
FlatBuffer.  This script instead reads only the selected section's observable
fully-connected inventory, builds one independent TensorFlow branch per weight,
and sends that *floating-point* graph through the public AI Edge Quantizer.

The resulting graph is intentionally not an official Gemma graph.  It is useful
because it separates three facts that otherwise get conflated:

* the public converter can quantize a random float graph;
* the resulting dtype/axis/scale-count/activation layout can match an official
  section's inventory;
* matching random layout still does not recover learned weights, private QAT,
  calibration data, exporter topology, or LiteRT-LM packaging.

The command-line inventory comparison currently compares
``FULLY_CONNECTED`` records.  ``EMBEDDING_LOOKUP`` is reported as skipped in
that comparison because TensorFlow's ordinary Keras Embedding conversion does
not guarantee the same LiteRT operator used by the official Gemma packages.
The low-level builder nevertheless knows how to emit an
``EMBEDDING_LOOKUP`` branch; the topology-injection harness uses it when it
needs a complete FC/embedding inventory.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_fresh_random_quantized_graph import (
    _extract_inventory,
)
from build_fresh_random_quantized_graph import (
    _vector as _fresh_vector,
)


class ConverterInventoryParityError(RuntimeError):
    """Raised when the converter-driven inventory experiment cannot run."""


_TENSOR_TYPE_NAMES = {
    0: "FLOAT32",
    2: "INT32",
    9: "INT8",
    17: "INT4",
    19: "INT2",
    20: "UINT4",
}
_BITS_BY_TYPE = {9: 8, 17: 4, 19: 2, 20: 4}
_OPCODE_BY_OPERATOR = {"EMBEDDING_LOOKUP": 7, "FULLY_CONNECTED": 9}


def _layer_name(ordinal: int, bits: int) -> str:
    return f"inventory_w_{ordinal:04d}_w{bits}"


def _build_random_float_tflite(
    records: list[dict[str, Any]],
    seed: int,
    *,
    ordinal_offset: int = 0,
    weight_provider: Callable[[int, dict[str, Any]], Any] | None = None,
    graph_description: str | None = None,
) -> tuple[bytes, list[dict[str, Any]]]:
    """Build a minimal independent float32 TFLite graph without SavedModel.

    ``ordinal_offset`` lets the topology-injection harness quantize a large
    inventory in several FlatBuffers while retaining globally unique branch
    names and deterministic random values.  ``weight_provider`` is the
    checkpoint-transplant seam: when supplied, it must return the float weight
    for each global inventory ordinal.  The graph construction and public
    quantizer path are otherwise identical to the random parity experiment.
    """

    try:
        import flatbuffers

        Buffer = importlib.import_module("tflite.Buffer")
        Model = importlib.import_module("tflite.Model")
        Operator = importlib.import_module("tflite.Operator")
        OperatorCode = importlib.import_module("tflite.OperatorCode")
        SignatureDef = importlib.import_module("tflite.SignatureDef")
        SubGraph = importlib.import_module("tflite.SubGraph")
        Tensor = importlib.import_module("tflite.Tensor")
        TensorMap = importlib.import_module("tflite.TensorMap")
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise ConverterInventoryParityError(
            "Float graph construction requires flatbuffers and generated tflite bindings."
        ) from exc

    builder = flatbuffers.Builder(1024)
    empty_buffer = Buffer.BufferStart(builder)
    empty_buffer = Buffer.BufferEnd(builder)
    buffer_offsets = [empty_buffer]
    tensor_offsets: list[int] = []
    operator_offsets: list[int] = []
    operator_codes: list[int] = []
    operator_code_indices: dict[int, int] = {}
    input_indices: list[int] = []
    output_indices: list[int] = []
    mappings: list[dict[str, Any]] = []

    def add_tensor(
        shape: list[int], buffer_index: int, name: str, type_value: int = 0
    ) -> int:
        shape_offset = _fresh_vector(builder, shape, "Int32")
        name_offset = builder.CreateString(name)
        Tensor.TensorStart(builder)
        Tensor.TensorAddShape(builder, shape_offset)
        Tensor.TensorAddType(builder, int(type_value))
        Tensor.TensorAddBuffer(builder, int(buffer_index))
        Tensor.TensorAddName(builder, name_offset)
        return Tensor.TensorEnd(builder)

    for ordinal, record in enumerate(records):
        shape = tuple(int(value) for value in record["shape"])
        if len(shape) != 2:
            raise ConverterInventoryParityError(
                f"Only rank-2 FC/embedding inventory records are supported; got {shape}."
            )
        operator_name = str(record.get("operator") or "FULLY_CONNECTED").upper()
        opcode = _OPCODE_BY_OPERATOR.get(operator_name)
        if opcode is None:
            raise ConverterInventoryParityError(
                f"Unsupported inventory operator {operator_name!r}."
            )
        opcode_index = operator_code_indices.setdefault(opcode, len(operator_codes))
        if opcode_index == len(operator_codes):
            operator_codes.append(opcode)
        global_ordinal = int(ordinal_offset) + ordinal
        name = _layer_name(global_ordinal, int(record["bits"]))
        if weight_provider is None:
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), global_ordinal + 1])
            )
            weights = rng.standard_normal(shape, dtype=np.float32)
            source_kind = "deterministic_random"
        else:
            weights = np.asarray(
                weight_provider(global_ordinal, record), dtype=np.float32
            )
            if tuple(int(value) for value in weights.shape) != shape:
                raise ConverterInventoryParityError(
                    "Weight provider returned shape "
                    f"{tuple(weights.shape)} for ordinal {global_ordinal}; "
                    f"expected {shape}."
                )
            weights = np.ascontiguousarray(weights)
            source_kind = "external_float_checkpoint"
        data_offset = builder.CreateByteVector(weights.tobytes(order="C"))
        Buffer.BufferStart(builder)
        Buffer.BufferAddData(builder, data_offset)
        weight_buffer = Buffer.BufferEnd(builder)
        buffer_offsets.append(weight_buffer)
        weight_buffer_index = len(buffer_offsets) - 1

        input_index = len(tensor_offsets)
        if operator_name == "EMBEDDING_LOOKUP":
            tensor_offsets.append(add_tensor([1], 0, f"{name}_input", 2))  # INT32
            output_shape = [1, shape[1]]
        else:
            tensor_offsets.append(add_tensor([1, shape[1]], 0, f"{name}_input"))
            output_shape = [1, shape[0]]
        weight_index = len(tensor_offsets)
        tensor_offsets.append(add_tensor(list(shape), weight_buffer_index, f"{name}_weight"))
        output_index = len(tensor_offsets)
        tensor_offsets.append(add_tensor(output_shape, 0, f"{name}_output"))
        input_indices.append(input_index)
        output_indices.append(output_index)
        input_vector = _fresh_vector(builder, [input_index, weight_index, -1], "Int32")
        output_vector = _fresh_vector(builder, [output_index], "Int32")
        Operator.OperatorStart(builder)
        Operator.OperatorAddOpcodeIndex(builder, opcode_index)
        Operator.OperatorAddInputs(builder, input_vector)
        Operator.OperatorAddOutputs(builder, output_vector)
        operator_offsets.append(Operator.OperatorEnd(builder))
        mappings.append(
            {
                "ordinal": global_ordinal,
                "layer_name": name,
                "official_operator": operator_name,
                "official_bits": int(record["bits"]),
                "official_shape": list(shape),
                "official_input_type": record.get("input_type_name"),
                "official_output_type": record.get("output_type_name"),
                "official_tensor_name": record.get("official_tensor_name", ""),
                "source_kind": source_kind,
            }
        )

    tensors_vector = _fresh_vector(builder, tensor_offsets, "UOffset")
    inputs_vector = _fresh_vector(builder, input_indices, "Int32")
    outputs_vector = _fresh_vector(builder, output_indices, "Int32")
    operators_vector = _fresh_vector(builder, operator_offsets, "UOffset")
    subgraph_name = builder.CreateString("random_inventory_graph")
    SubGraph.SubGraphStart(builder)
    SubGraph.SubGraphAddTensors(builder, tensors_vector)
    SubGraph.SubGraphAddInputs(builder, inputs_vector)
    SubGraph.SubGraphAddOutputs(builder, outputs_vector)
    SubGraph.SubGraphAddOperators(builder, operators_vector)
    SubGraph.SubGraphAddName(builder, subgraph_name)
    subgraph_offset = SubGraph.SubGraphEnd(builder)
    subgraphs_vector = _fresh_vector(builder, [subgraph_offset], "UOffset")

    opcode_offsets: list[int] = []
    for opcode in operator_codes:
        OperatorCode.OperatorCodeStart(builder)
        OperatorCode.OperatorCodeAddDeprecatedBuiltinCode(builder, int(opcode))
        OperatorCode.OperatorCodeAddBuiltinCode(builder, int(opcode))
        OperatorCode.OperatorCodeAddVersion(builder, 1)
        opcode_offsets.append(OperatorCode.OperatorCodeEnd(builder))
    opcodes_vector = _fresh_vector(builder, opcode_offsets, "UOffset")
    buffers_vector = _fresh_vector(builder, buffer_offsets, "UOffset")
    input_maps: list[int] = []
    output_maps: list[int] = []
    for ordinal, (input_index, output_index) in enumerate(zip(input_indices, output_indices)):
        input_name = builder.CreateString(f"input_{ordinal}")
        TensorMap.TensorMapStart(builder)
        TensorMap.TensorMapAddName(builder, input_name)
        TensorMap.TensorMapAddTensorIndex(builder, input_index)
        input_maps.append(TensorMap.TensorMapEnd(builder))
        output_name = builder.CreateString(f"output_{ordinal}")
        TensorMap.TensorMapStart(builder)
        TensorMap.TensorMapAddName(builder, output_name)
        TensorMap.TensorMapAddTensorIndex(builder, output_index)
        output_maps.append(TensorMap.TensorMapEnd(builder))
    input_maps_vector = _fresh_vector(builder, input_maps, "UOffset")
    output_maps_vector = _fresh_vector(builder, output_maps, "UOffset")
    signature_key = builder.CreateString("serve")
    SignatureDef.SignatureDefStart(builder)
    SignatureDef.SignatureDefAddInputs(builder, input_maps_vector)
    SignatureDef.SignatureDefAddOutputs(builder, output_maps_vector)
    SignatureDef.SignatureDefAddSignatureKey(builder, signature_key)
    SignatureDef.SignatureDefAddSubgraphIndex(builder, 0)
    signature_offset = SignatureDef.SignatureDefEnd(builder)
    signatures_vector = _fresh_vector(builder, [signature_offset], "UOffset")
    description = builder.CreateString(
        graph_description or "Independent random float inventory graph"
    )
    Model.ModelStart(builder)
    Model.ModelAddVersion(builder, 3)
    Model.ModelAddOperatorCodes(builder, opcodes_vector)
    Model.ModelAddSubgraphs(builder, subgraphs_vector)
    Model.ModelAddDescription(builder, description)
    Model.ModelAddBuffers(builder, buffers_vector)
    Model.ModelAddSignatureDefs(builder, signatures_vector)
    model_offset = Model.ModelEnd(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    return bytes(builder.Output()), mappings


def _config_for_record(record: dict[str, Any]) -> dict[str, Any]:
    bits = int(record["bits"])
    config: dict[str, Any] = {
        "weight_tensor_config": {
            "num_bits": bits,
            "symmetric": True,
            "granularity": "CHANNELWISE",
            "dtype": "INT",
        },
        "compute_precision": "INTEGER",
        "explicit_dequantize": False,
        # W2/W4 and mixed static/weight-only branches are intentionally a
        # synthetic audit.  The public policy rejects some arbitrary scopes;
        # bypass only that policy check, never the converter itself.
        "skip_checks": True,
        "min_weight_elements": 0,
    }
    input_type = str(record.get("input_type_name") or "")
    output_type = str(record.get("output_type_name") or "")
    if input_type == "INT8" or output_type == "INT8":
        config["activation_tensor_config"] = {
            "num_bits": 8,
            "symmetric": True,
            "granularity": "TENSORWISE",
            "dtype": "INT",
        }
    return config


def _recipe(
    records: list[dict[str, Any]], *, ordinal_offset: int = 0
) -> list[dict[str, Any]]:
    """Make last-match-wins regex entries for the independent branch names."""

    recipe: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records):
        name = _layer_name(int(ordinal_offset) + ordinal, int(record["bits"]))
        operation = str(record.get("operator") or "FULLY_CONNECTED").upper()
        if operation not in _OPCODE_BY_OPERATOR:
            raise ConverterInventoryParityError(
                f"Unsupported inventory operator {operation!r}."
            )
        recipe.append(
            {
                "regex": re.escape(name),
                "operation": operation,
                "algorithm_key": "min_max_uniform_quantize",
                "op_config": _config_for_record(record),
            }
        )
    return recipe


def _enum_names(module: Any, class_name: str) -> dict[int, str]:
    cls = getattr(module, class_name, None)
    if cls is None:
        return {}
    return {
        int(value): name
        for name, value in vars(cls).items()
        if not name.startswith("_") and isinstance(value, int)
    }


def _vector(obj: Any, length_method: str, item_method: str) -> list[int]:
    length = getattr(obj, length_method, lambda: 0)() or 0
    method = getattr(obj, item_method, None)
    return [int(method(index)) for index in range(int(length))] if method else []


def _decode(value: Any) -> str:
    """Decode generated-schema byte strings without exposing ``b'...'`` text."""

    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _quantization(tensor: Any) -> dict[str, Any] | None:
    quant = tensor.Quantization()
    if quant is None:
        return None
    scales = int(quant.ScaleLength() or 0)
    zeros = int(quant.ZeroPointLength() or 0)
    if not scales and not zeros:
        return None
    zero_values = [int(quant.ZeroPoint(index)) for index in range(zeros)]
    return {
        "scale_count": scales,
        "zero_point_count": zeros,
        "quantized_dimension": int(quant.QuantizedDimension()),
        "zero_points_all_zero": bool(zero_values) and all(value == 0 for value in zero_values),
    }


def _inspect_quantized(path: Path) -> list[dict[str, Any]]:
    try:
        model_module = importlib.import_module("tflite.Model")
        op_module = importlib.import_module("tflite.BuiltinOperator")
        tensor_module = importlib.import_module("tflite.TensorType")
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise ConverterInventoryParityError(
            "Inspection requires the generated 'tflite' schema bindings."
        ) from exc
    model = model_module.Model.GetRootAsModel(path.read_bytes(), 0)
    op_names = _enum_names(op_module, "BuiltinOperator")
    tensor_names = _enum_names(tensor_module, "TensorType")
    tensor_names.update({19: "INT2", 20: "UINT4"})
    records: list[dict[str, Any]] = []
    for subgraph_index in range(int(model.SubgraphsLength() or 0)):
        subgraph = model.Subgraphs(subgraph_index)
        for operator_index in range(int(subgraph.OperatorsLength() or 0)):
            operator = subgraph.Operators(operator_index)
            code = model.OperatorCodes(operator.OpcodeIndex())
            if op_names.get(int(code.BuiltinCode()), "") != "FULLY_CONNECTED":
                continue
            inputs = _vector(operator, "InputsLength", "Inputs")
            outputs = _vector(operator, "OutputsLength", "Outputs")
            if len(inputs) < 2 or not outputs or inputs[1] < 0:
                continue
            weight = subgraph.Tensors(inputs[1])
            weight_type = int(weight.Type())
            bits = _BITS_BY_TYPE.get(weight_type)
            quant = _quantization(weight)
            if bits is None or quant is None:
                continue
            input_tensor = subgraph.Tensors(inputs[0])
            output_tensor = subgraph.Tensors(outputs[0])
            records.append(
                {
                    "subgraph": subgraph_index,
                    "operator_index": operator_index,
                    "bits": bits,
                    "type_name": _TENSOR_TYPE_NAMES.get(weight_type, f"unknown:{weight_type}"),
                    "shape": [int(weight.Shape(index)) for index in range(int(weight.ShapeLength() or 0))],
                    "scale_count": quant["scale_count"],
                    "zero_point_count": quant["zero_point_count"],
                    "quantized_dimension": quant["quantized_dimension"],
                    "zero_points_all_zero": quant["zero_points_all_zero"],
                    "input_type": _TENSOR_TYPE_NAMES.get(int(input_tensor.Type()), f"unknown:{int(input_tensor.Type())}"),
                    "output_type": _TENSOR_TYPE_NAMES.get(int(output_tensor.Type()), f"unknown:{int(output_tensor.Type())}"),
                    "input_quantization": _quantization(input_tensor),
                    "output_quantization": _quantization(output_tensor),
                    "weight_name": _decode(weight.Name()),
                }
            )
    return records


def _layout_key(record: dict[str, Any]) -> tuple[Any, ...]:
    def quantization_key(value: Any) -> tuple[Any, ...]:
        if not value:
            return (None, None, None, None)
        return (
            int(value.get("scale_count", 0)),
            int(value.get("zero_point_count", 0)),
            int(value.get("quantized_dimension", 0)),
            bool(value.get("zero_points_all_zero", False)),
        )

    return (
        int(record["bits"]),
        tuple(int(value) for value in record["shape"]),
        int(record["scale_count"]),
        int(record["zero_point_count"]),
        int(record["quantized_dimension"]),
        str(record.get("input_type_name", record.get("input_type", ""))),
        str(record.get("output_type_name", record.get("output_type", ""))),
        quantization_key(record.get("input_quantization")),
        quantization_key(record.get("output_quantization")),
    )


def _counter(records: list[dict[str, Any]]) -> collections.Counter[tuple[Any, ...]]:
    return collections.Counter(_layout_key(record) for record in records)


def _counter_diff(counter: collections.Counter[tuple[Any, ...]]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0])) if value}


def _quantize(fp32_path: Path, output: Path, recipe: list[dict[str, Any]], *, calibration_samples: int, threads: int) -> None:
    try:
        from ai_edge_quantizer import quantizer
        from ai_edge_quantizer.utils import tfl_interpreter_utils
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise ConverterInventoryParityError(
            "Quantization requires ai-edge-quantizer, ai-edge-litert, and their runtime dependencies."
        ) from exc
    try:
        quantizer_instance = quantizer.Quantizer(str(fp32_path))
        quantizer_instance.load_quantization_recipe(recipe)
        calibration_result = None
        # JSON-style configs are normalized by ``load_quantization_recipe``;
        # checking the entries explicitly keeps this path correct across AI
        # Edge Quantizer releases whose ``need_calibration`` helper differs
        # for mixed string/enum configs.
        needs_calibration = any(
            "activation_tensor_config" in entry.get("op_config", {})
            for entry in recipe
        )
        if needs_calibration:
            calibration_data = tfl_interpreter_utils.create_random_normal_input_data(
                quantizer_instance._float_model_buffer,  # pylint: disable=protected-access
                num_samples=calibration_samples,
            )
            calibration_result = quantizer_instance.calibrate(calibration_data, num_threads=threads)
        result = quantizer_instance.quantize(calibration_result, enable_progress_report=False)
        output.write_bytes(result.quantized_model)
    except Exception as exc:  # pragma: no cover - converter/version dependent
        raise ConverterInventoryParityError(f"AI Edge quantization failed: {exc}") from exc


def run(
    artifact: str | Path,
    *,
    model_type: str,
    seed: int,
    max_weights: int | None,
    output_dir: str | Path,
    calibration_samples: int,
    threads: int,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ConverterInventoryParityError(f"Output directory is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _, all_records = _extract_inventory(
        artifact,
        model_type,
        include_embeddings=True,
        max_weights=max_weights,
    )
    skipped = [record for record in all_records if record["operator"] != "FULLY_CONNECTED"]
    records = [record for record in all_records if record["operator"] == "FULLY_CONNECTED"]
    if not records:
        raise ConverterInventoryParityError("No FULLY_CONNECTED records remain after filtering.")
    fp32_bytes, mappings = _build_random_float_tflite(records, seed)
    fp32_path = output_root / "random_inventory_fp32.tflite"
    quantized_path = output_root / "random_inventory_quantized.tflite"
    fp32_path.write_bytes(fp32_bytes)
    recipe = _recipe(records)
    (output_root / "random_inventory_recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _quantize(
        fp32_path,
        quantized_path,
        recipe,
        calibration_samples=calibration_samples,
        threads=threads,
    )
    observed = _inspect_quantized(quantized_path)
    official_counter = _counter(records)
    observed_counter = _counter(observed)
    result = {
        "artifact": str(Path(artifact).expanduser().resolve()),
        "model_type": model_type,
        "seed": int(seed),
        "max_weights": max_weights,
        "calibration_samples": int(calibration_samples),
        "threads": int(threads),
        "training_executed": False,
        "random_initialization": True,
        "private_qat_recipe_recovered": False,
        "official_inventory_count": len(records),
        "skipped_non_fc_count": len(skipped),
        "quantized_fc_count": len(observed),
        "official_layout": _counter_diff(official_counter),
        "quantized_layout": _counter_diff(observed_counter),
        "converter_quantization_layout_match": bool(official_counter == observed_counter),
        "graph_structure_match": False,
        "exact_official_model_match": False,
        "fp32_size": fp32_path.stat().st_size,
        "quantized_size": quantized_path.stat().st_size,
        "skipped_records": [
            {
                "operator": record["operator"],
                "bits": record["bits"],
                "shape": record["shape"],
                "official_tensor_name": record.get("official_tensor_name", ""),
            }
            for record in skipped
        ],
        "mapping_preview": mappings[:12],
        "observed_preview": observed[:12],
        "layout_difference": {
            "official_only": _counter_diff(official_counter - observed_counter),
            "quantized_only": _counter_diff(observed_counter - official_counter),
        },
        "interpretation": (
            "The public AI Edge converter quantized an independently built random float graph "
            "with the selected official FC inventory layout. Graph topology, embeddings, "
            "metadata, learned weights, private QAT, and LiteRT-LM packaging remain different."
            if official_counter == observed_counter
            else "The public converter did not reproduce the selected FC inventory; inspect "
            "layout_difference for unsupported activation or weight behavior."
        ),
    }
    (output_root / "random_inventory_parity_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and quantize a random float graph from an official FC inventory."
    )
    parser.add_argument("artifact", help="Official .litertlm artifact.")
    parser.add_argument("--model-type", default="tf_lite_mtp_drafter")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-weights", type=int)
    parser.add_argument("--calibration-samples", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_weights is not None and args.max_weights < 1:
        parser.error("--max-weights must be positive")
    if args.calibration_samples < 1 or args.threads < 1:
        parser.error("--calibration-samples and --threads must be positive")
    try:
        result = run(
            args.artifact,
            model_type=args.model_type,
            seed=args.seed,
            max_weights=args.max_weights,
            output_dir=args.output_dir,
            calibration_samples=args.calibration_samples,
            threads=args.threads,
        )
    except (OSError, ConverterInventoryParityError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
