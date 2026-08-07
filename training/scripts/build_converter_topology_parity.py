"""Run the public AI Edge Quantizer on an official-topology float variant.

This is the stronger companion to ``build_converter_random_inventory_parity``.
The inventory harness builds independent branches, while this harness unpacks a
selected LiteRT-LM TFLite section with the public AI Edge schema object API,
keeps its subgraphs/operators/signatures/options, replaces quantized constants
with deterministic random FLOAT32 constants, and sends the resulting graph
through ``ai-edge-quantizer``.

The official artifact is read-only.  The output is a standalone TFLite model
and a report that compares observable graph/layout fingerprints.  The report
keeps serializer-sensitive hashes (which include buffer indices) separate from
execution-topology/layout hashes that ignore those indices.  The harness does
not claim that random weights can reproduce learned outputs or Google's private
exporter.  Some official sections contain custom operators or runtime-specific
metadata that the public quantizer cannot lower; those errors are recorded as
an explicit boundary rather than silently treated as parity.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import importlib
import io
import json
import mmap
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_random_official_topology_parity import (  # noqa: E402
    _tflite_graph_fingerprint,
)
from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    inspect_litertlm,
)


class ConverterTopologyParityError(RuntimeError):
    """Raised when the float-topology converter experiment cannot run."""


_FLOAT32 = 0
_INT8 = 9
_INT4 = 17
_INT2 = 19
_UINT4 = 20
_LOW_BIT_TYPES = {_INT8: 8, _INT4: 4, _INT2: 2, _UINT4: 4}
_FULLY_CONNECTED = 9
_EMBEDDING_LOOKUP = 7


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _sha256(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _section_by_model_type(package: dict[str, Any], model_type: str) -> dict[str, Any]:
    matches = [
        item
        for item in package.get("sections", [])
        if item.get("data_type_name") == "TFLiteModel"
        and any(
            entry.get("key") == "model_type" and str(entry.get("value")) == model_type
            for entry in item.get("items", [])
        )
    ]
    if not matches:
        available = [
            next(
                (
                    str(entry.get("value"))
                    for entry in item.get("items", [])
                    if entry.get("key") == "model_type"
                ),
                "",
            )
            for item in package.get("sections", [])
            if item.get("data_type_name") == "TFLiteModel"
        ]
        raise ConverterTopologyParityError(
            f"No TFLite model_type={model_type!r} section. Available: {available}"
        )
    return matches[0]


def _schema_modules() -> tuple[Any, Any]:
    """Return the object-API schema and flatbuffers module."""

    try:
        flatbuffers = importlib.import_module("flatbuffers")
        schema = importlib.import_module("ai_edge_litert.schema_py_generated")
        return flatbuffers, schema
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise ConverterTopologyParityError(
            "Topology conversion requires ai-edge-litert's schema_py_generated bindings."
        ) from exc


def _read_section(artifact: Path, section: dict[str, Any]) -> bytes:
    with artifact.open("rb") as handle:
        handle.seek(int(section["begin_offset"]))
        data = handle.read(int(section["size"]))
    if len(data) != int(section["size"]):
        raise ConverterTopologyParityError("Could not read the complete TFLite section.")
    return data


def _unpack_model(section_bytes: bytes) -> tuple[Any, Any, Any]:
    flatbuffers, schema = _schema_modules()
    del flatbuffers  # retained in the tuple returned by _schema_modules for symmetry
    try:
        parsed = schema.Model.GetRootAsModel(section_bytes, 0)
        return schema.ModelT.InitFromObj(parsed), schema, parsed
    except Exception as exc:  # pragma: no cover - schema-version dependent
        raise ConverterTopologyParityError(f"Could not unpack TFLite ModelT: {exc}") from exc


def _pack_model(model: Any, schema: Any) -> bytes:
    flatbuffers, _ = _schema_modules()
    builder = flatbuffers.Builder(1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytes(builder.Output())


def _opcode_builtin(model: Any, operator: Any) -> int:
    code = model.operatorCodes[int(operator.opcodeIndex)]
    return int(code.builtinCode)


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    if tensor.shape is None:
        return ()
    return tuple(int(value) for value in np.asarray(tensor.shape).reshape(-1))


def _quantization_record(tensor: Any) -> dict[str, Any] | None:
    quant = tensor.quantization
    if quant is None:
        return None
    scales = np.asarray(quant.scale).reshape(-1) if quant.scale is not None else np.asarray([])
    zeros = np.asarray(quant.zeroPoint).reshape(-1) if quant.zeroPoint is not None else np.asarray([])
    if not len(scales) and not len(zeros):
        return None
    return {
        "scale_count": int(len(scales)),
        "zero_point_count": int(len(zeros)),
        "quantized_dimension": int(quant.quantizedDimension),
        "zero_points_all_zero": bool(len(zeros)) and bool(np.all(zeros == 0)),
    }


def _new_buffer(schema: Any, values: np.ndarray) -> Any:
    buffer = schema.BufferT()
    if isinstance(values, (bytes, bytearray, memoryview)):
        buffer.data = np.frombuffer(bytes(values), dtype=np.uint8).copy()
    else:
        buffer.data = np.asarray(values, dtype=np.uint8).reshape(-1).copy()
    buffer.offset = 0
    buffer.size = 0
    return buffer


def _random_float_values(shape: tuple[int, ...], seed: int, ordinal: int) -> np.ndarray:
    if not shape:
        return np.asarray([], dtype=np.float32)
    if any(value < 0 for value in shape):
        raise ConverterTopologyParityError(f"Invalid tensor shape: {shape}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(ordinal)]))
    return rng.standard_normal(shape, dtype=np.float32)


def _bypass_quantize_dequantize(model: Any) -> int:
    """Remove quantize/dequantize nodes after their tensors become FLOAT32."""

    bypass: dict[tuple[int, int], int] = {}
    removed = 0
    for subgraph_index, subgraph in enumerate(model.subgraphs or []):
        for operator in subgraph.operators or []:
            if _opcode_builtin(model, operator) not in (6, 114):
                continue
            inputs = [int(value) for value in np.asarray(operator.inputs).reshape(-1)]
            outputs = [int(value) for value in np.asarray(operator.outputs).reshape(-1)]
            if len(inputs) != 1 or len(outputs) != 1 or inputs[0] < 0 or outputs[0] < 0:
                continue
            bypass[(subgraph_index, outputs[0])] = inputs[0]

    def resolve(subgraph_index: int, tensor_index: int) -> int:
        seen: set[int] = set()
        current = int(tensor_index)
        while (subgraph_index, current) in bypass and current not in seen:
            seen.add(current)
            current = int(bypass[(subgraph_index, current)])
        return current

    for subgraph_index, subgraph in enumerate(model.subgraphs or []):
        kept = []
        for operator in subgraph.operators or []:
            if _opcode_builtin(model, operator) in (6, 114):
                inputs = [int(value) for value in np.asarray(operator.inputs).reshape(-1)]
                outputs = [int(value) for value in np.asarray(operator.outputs).reshape(-1)]
                if len(inputs) == 1 and len(outputs) == 1:
                    removed += 1
                    continue
            operator.inputs = np.asarray(
                [resolve(subgraph_index, int(value)) if int(value) >= 0 else int(value)
                 for value in np.asarray(operator.inputs).reshape(-1)],
                dtype=np.int32,
            )
            operator.outputs = np.asarray(
                [resolve(subgraph_index, int(value)) if int(value) >= 0 else int(value)
                 for value in np.asarray(operator.outputs).reshape(-1)],
                dtype=np.int32,
            )
            kept.append(operator)
        subgraph.operators = kept
        subgraph.inputs = np.asarray(
            [resolve(subgraph_index, int(value)) for value in np.asarray(subgraph.inputs).reshape(-1)],
            dtype=np.int32,
        )
        subgraph.outputs = np.asarray(
            [resolve(subgraph_index, int(value)) for value in np.asarray(subgraph.outputs).reshape(-1)],
            dtype=np.int32,
        )

    for signature in model.signatureDefs or []:
        for tensor_map in (signature.inputs or []) + (signature.outputs or []):
            tensor_map.tensorIndex = resolve(int(signature.subgraphIndex), int(tensor_map.tensorIndex))
    return removed


def _floatify_model(
    model: Any,
    schema: Any,
    *,
    seed: int,
    floatify_all_low_bit: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace low-bit tensors/buffers with random float constants.

    Tensor indices and other INT32 control tensors are intentionally retained.
    Every low-bit tensor is converted to FLOAT32 so the quantizer receives a
    consistently floating-point compute graph.  New buffers are appended; the
    official buffers and model bytes are never mutated.
    """

    fc_records: list[dict[str, Any]] = []
    op_counts: collections.Counter[str] = collections.Counter()
    converted_tensor_count = 0
    converted_buffer_count = 0
    buffer_map: dict[int, int] = {}
    ordinal = 0

    # Record original FC contracts before changing tensor types.
    for subgraph_index, subgraph in enumerate(model.subgraphs or []):
        for operator_index, operator in enumerate(subgraph.operators or []):
            builtin = _opcode_builtin(model, operator)
            op_counts[str(builtin)] += 1
            if builtin not in (_FULLY_CONNECTED, _EMBEDDING_LOOKUP):
                continue
            inputs = [int(value) for value in np.asarray(operator.inputs).reshape(-1)]
            outputs = [int(value) for value in np.asarray(operator.outputs).reshape(-1)]
            if len(inputs) < 2 or not outputs or inputs[1] < 0:
                continue
            weight = subgraph.tensors[inputs[1]]
            bits = _LOW_BIT_TYPES.get(int(weight.type))
            if bits is None:
                continue
            input_tensor = subgraph.tensors[inputs[0]] if inputs else None
            output_tensor = subgraph.tensors[outputs[0]] if outputs else None
            fc_records.append(
                {
                    "record_ordinal": len(fc_records),
                    "subgraph": subgraph_index,
                    "operator_index": operator_index,
                    "operator": "FULLY_CONNECTED" if builtin == _FULLY_CONNECTED else "EMBEDDING_LOOKUP",
                    "bits": bits,
                    "shape": list(_tensor_shape(weight)),
                    "official_tensor_name": _decode(weight.name),
                    "weight_tensor_index": inputs[1],
                    "output_tensor_index": outputs[0],
                    "input_type": int(input_tensor.type) if input_tensor is not None else None,
                    "output_type": int(output_tensor.type) if output_tensor is not None else None,
                    "input_quantization": _quantization_record(input_tensor) if input_tensor else None,
                    "output_quantization": _quantization_record(output_tensor) if output_tensor else None,
                }
            )
            # Assign stable names while the record is created.  A few exported
            # graphs reuse a weight tensor across operator records; assigning
            # these names only while visiting low-bit tensors would leave the
            # earlier record without a recipe scope.
            record = fc_records[-1]
            record["recipe_name"] = (
                f"converter_topology_scope_{int(record['record_ordinal']):04d}_w{int(record['bits'])}"
            )
            record["weight_name"] = (
                f"converter_topology_weight_{int(record['record_ordinal']):04d}_w{int(record['bits'])}"
            )

    # Convert every low-bit tensor to float.  Constants receive fresh random
    # values; activation tensors typically reference buffer 0 and need no data.
    record_by_tensor = {
        (int(record["subgraph"]), int(record["weight_tensor_index"])): record
        for record in fc_records
    }
    for subgraph_index, subgraph in enumerate(model.subgraphs or []):
        for tensor_index, tensor in enumerate(subgraph.tensors or []):
            old_type = int(tensor.type)
            bits = _LOW_BIT_TYPES.get(old_type)
            if bits is None:
                continue
            tensor.type = _FLOAT32
            tensor.quantization = None
            converted_tensor_count += 1
            record = record_by_tensor.get((subgraph_index, tensor_index))
            if record is not None:
                # Use a stable, unique name so the recipe scope is independent
                # of exporter-generated names and survives schema repacking.
                tensor.name = record["weight_name"].encode("utf-8")
            buffer_index = int(tensor.buffer)
            shape = _tensor_shape(tensor)
            if buffer_index <= 0 or not shape:
                continue
            if buffer_index not in buffer_map:
                values = _random_float_values(shape, seed, ordinal)
                ordinal += 1
                model.buffers.append(_new_buffer(schema, values.tobytes(order="C")))
                buffer_map[buffer_index] = len(model.buffers) - 1
                converted_buffer_count += 1
            tensor.buffer = buffer_map[buffer_index]

        # Scope matching in AI Edge Quantizer is based on operator output
        # names.  Give each recorded FC output the same stable branch name as
        # its weight so per-bit recipes select the intended operation.
        for record in fc_records:
            if int(record["subgraph"]) != subgraph_index:
                continue
            if record.get("operator") not in ("FULLY_CONNECTED", "EMBEDDING_LOOKUP"):
                continue
            output_index = int(record["output_tensor_index"])
            if 0 <= output_index < len(subgraph.tensors or []):
                subgraph.tensors[output_index].name = str(record["recipe_name"]).encode("utf-8")

    bypassed = _bypass_quantize_dequantize(model)
    # ``floatify_all_low_bit`` is kept as an explicit flag in the report and
    # CLI.  The current implementation always floatifies all low-bit tensors;
    # this avoids an invalid mixed INT8/float FC graph and makes the experiment
    # deterministic.  The flag lets callers document that choice.
    del floatify_all_low_bit
    return fc_records, {
        "converted_tensor_count": converted_tensor_count,
        "converted_buffer_count": converted_buffer_count,
        "bypassed_quantize_dequantize_count": bypassed,
        "low_bit_type_counts": dict(sorted(collections.Counter(
            int(record["bits"]) for record in fc_records
        ).items())),
        "opcode_builtin_counts": dict(sorted(op_counts.items())),
    }


def _recipe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recipe: list[dict[str, Any]] = []
    for record in records:
        if record["operator"] not in ("FULLY_CONNECTED", "EMBEDDING_LOOKUP"):
            continue
        name = str(record.get("recipe_name") or record.get("official_tensor_name") or "")
        if not name:
            raise ConverterTopologyParityError("A quantized FC record has no recipe scope name.")
        # Weight tensor names are retained in the floatified graph.  Use a
        # broad operation regex and last-match ordering; the recipe is only an
        # audit input, not a claim that Google uses these private scopes.
        config: dict[str, Any] = {
            "weight_tensor_config": {
                "num_bits": int(record["bits"]),
                "symmetric": True,
                "granularity": "CHANNELWISE",
                "dtype": "INT",
            },
            "compute_precision": "INTEGER",
            "explicit_dequantize": False,
            "skip_checks": True,
            "min_weight_elements": 0,
        }
        # The official contract is recorded before floatification.  Preserve
        # static A8 only where the supplied graph had INT8 FC edges.
        if record.get("input_type") == _INT8 or record.get("output_type") == _INT8:
            config["activation_tensor_config"] = {
                "num_bits": 8,
                "symmetric": True,
                "granularity": "TENSORWISE",
                "dtype": "INT",
            }
        recipe.append(
            {
                "regex": re.escape(name),
                "operation": record["operator"],
                "algorithm_key": "min_max_uniform_quantize",
                "op_config": config,
            }
        )
    return recipe


def _quantize(
    fp32_model: Path | bytes,
    output: Path | None,
    recipe: list[dict[str, Any]],
    *,
    calibration_samples: int,
    threads: int,
) -> None:
    try:
        from ai_edge_quantizer import quantizer
        from ai_edge_quantizer.utils import tfl_interpreter_utils
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise ConverterTopologyParityError(
            "Quantization requires ai-edge-quantizer and ai-edge-litert."
        ) from exc
    try:
        source = str(fp32_model) if isinstance(fp32_model, Path) else fp32_model
        instance = quantizer.Quantizer(source)
        instance.load_quantization_recipe(recipe)
        needs_calibration = any(
            "activation_tensor_config" in entry.get("op_config", {}) for entry in recipe
        )
        calibration_result = None
        # The stock helper enables XNNPACK.  A floatified graph retains
        # official custom/runtime nodes that XNNPACK may reject even when
        # the builtin interpreter can execute them, so disable that optional
        # delegate for both calibration and quantization validation.
        original_create_interpreter = tfl_interpreter_utils.create_tfl_interpreter
        original_content_map = tfl_interpreter_utils.get_tensor_name_to_content_map

        def create_no_xnnpack(model: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs["use_xnnpack"] = False
            return original_create_interpreter(model, *args, **kwargs)

        def content_map_skip_unallocated(
            interpreter: Any, subgraph_index: int = 0, dequantize: bool = False
        ) -> dict[str, Any]:
            """Keep calibration moving past runtime-only null tensors."""

            result: dict[str, Any] = {}
            for detail in interpreter.get_tensor_details(subgraph_index):
                if not detail["name"] or not np.all(detail["shape"]):
                    continue
                try:
                    result[detail["name"]] = tfl_interpreter_utils.get_tensor_data(
                        interpreter, detail, subgraph_index, dequantize
                    )
                except ValueError as exc:
                    if "Tensor data is null" not in str(exc):
                        raise
            return result

        tfl_interpreter_utils.create_tfl_interpreter = create_no_xnnpack
        tfl_interpreter_utils.get_tensor_name_to_content_map = content_map_skip_unallocated
        try:
            # The quantizer emits a tqdm line for every tensor even when its
            # progress flag is disabled.  Capture that noise so a full 270M
            # audit remains readable while preserving converter exceptions.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                if needs_calibration:
                    calibration_data = tfl_interpreter_utils.create_random_normal_input_data(
                        instance._float_model_buffer,  # pylint: disable=protected-access
                        num_samples=calibration_samples,
                    )
                    calibration_result = instance.calibrate(
                        calibration_data, num_threads=threads
                    )
                result = instance.quantize(calibration_result, enable_progress_report=False)
        finally:
            tfl_interpreter_utils.create_tfl_interpreter = original_create_interpreter
            tfl_interpreter_utils.get_tensor_name_to_content_map = original_content_map
        quantized_model = bytes(result.quantized_model)
        if output is not None:
            output.write_bytes(quantized_model)
        return quantized_model
    except Exception as exc:  # pragma: no cover - converter-version dependent
        raise ConverterTopologyParityError(f"AI Edge topology quantization failed: {exc}") from exc


def _inspect_quantized(model: Path | bytes) -> dict[str, Any]:
    """Return graph and quantized inventory fingerprints for a standalone model."""

    data = model.read_bytes() if isinstance(model, Path) else bytes(model)
    try:
        graph = _tflite_graph_fingerprint(data, include_details=False)
        graph_without_buffer_indices = _tflite_graph_fingerprint(
            data, include_details=False, include_buffer_indices=False
        )
    except Exception as exc:  # pragma: no cover - schema-version dependent
        graph = {"available": False, "error": str(exc)}
        graph_without_buffer_indices = {"available": False, "error": str(exc)}
    return {
        "size": len(data),
        "sha256": _sha256(data),
        "graph": graph,
        "graph_without_buffer_indices": graph_without_buffer_indices,
    }


def run(
    artifact: str | Path,
    *,
    model_type: str,
    seed: int,
    output_dir: str | Path,
    calibration_samples: int,
    threads: int,
    write_models: bool,
) -> dict[str, Any]:
    artifact_path = Path(artifact).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not artifact_path.is_file():
        raise ConverterTopologyParityError(f"Official artifact does not exist: {artifact_path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ConverterTopologyParityError(f"Output directory is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        package = inspect_litertlm(artifact_path, inspect_tflite=False)
        section = _section_by_model_type(package, model_type)
    except (OSError, LiteRTLMInspectionError) as exc:
        raise ConverterTopologyParityError(f"Could not inspect official artifact: {exc}") from exc

    official_bytes = _read_section(artifact_path, section)
    official_graph = {
        "size": len(official_bytes),
        "sha256": _sha256(official_bytes),
        "graph": _tflite_graph_fingerprint(official_bytes, include_details=False),
        "graph_without_buffer_indices": _tflite_graph_fingerprint(
            official_bytes, include_details=False, include_buffer_indices=False
        ),
    }
    model, schema, _ = _unpack_model(official_bytes)
    records, floatify = _floatify_model(model, schema, seed=seed, floatify_all_low_bit=True)
    float_bytes = _pack_model(model, schema)
    float_path = output_root / "random_topology_fp32.tflite" if write_models else None
    quantized_path = output_root / "random_topology_quantized.tflite" if write_models else None
    if float_path is not None:
        float_path.write_bytes(float_bytes)
    recipe = _recipe(records)
    (output_root / "random_topology_recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    quantized_result = _quantize(
        float_path if float_path is not None else float_bytes,
        quantized_path,
        recipe,
        calibration_samples=calibration_samples,
        threads=threads,
    )
    quantized_bytes = (
        quantized_path.read_bytes() if quantized_path is not None else quantized_result
    )
    quantized = _inspect_quantized(quantized_bytes)
    official_structural = official_graph["graph"].get("structural_sha256")
    quantized_structural = quantized["graph"].get("structural_sha256")
    official_layout = official_graph["graph"].get("quantization_layout_sha256")
    quantized_layout = quantized["graph"].get("quantization_layout_sha256")
    official_topology = official_graph["graph_without_buffer_indices"].get(
        "structural_sha256"
    )
    quantized_topology = quantized["graph_without_buffer_indices"].get(
        "structural_sha256"
    )
    official_layout_without_buffers = official_graph["graph_without_buffer_indices"].get(
        "quantization_layout_sha256"
    )
    quantized_layout_without_buffers = quantized[
        "graph_without_buffer_indices"
    ].get("quantization_layout_sha256")
    result = {
        "artifact": str(artifact_path),
        "model_type": model_type,
        "seed": int(seed),
        "calibration_samples": int(calibration_samples),
        "threads": int(threads),
        "training_executed": False,
        "random_initialization": True,
        "private_qat_recipe_recovered": False,
        "official_section": official_graph,
        "float_variant": {
            "size": len(float_bytes),
            "sha256": _sha256(float_bytes),
            "path": str(float_path) if float_path is not None else None,
            "records": len(records),
            "floatification": floatify,
        },
        "quantized_variant": {
            **quantized,
            "path": str(quantized_path) if quantized_path is not None else None,
        },
        "recipe_path": str(output_root / "random_topology_recipe.json"),
        "converter_graph_structure_match": bool(
            official_structural and official_structural == quantized_structural
        ),
        "converter_quantization_layout_match": bool(
            official_layout and official_layout == quantized_layout
        ),
        "converter_execution_topology_match_ignoring_buffer_indices": bool(
            official_topology and official_topology == quantized_topology
        ),
        "converter_quantization_layout_match_ignoring_buffer_indices": bool(
            official_layout_without_buffers
            and official_layout_without_buffers == quantized_layout_without_buffers
        ),
        "exact_official_model_match": False,
        "comparison": {
            "official_structural_sha256": official_structural,
            "quantized_structural_sha256": quantized_structural,
            "official_quantization_layout_sha256": official_layout,
            "quantized_quantization_layout_sha256": quantized_layout,
            "official_quantization_values_sha256": official_graph["graph"].get(
                "quantization_values_sha256"
            ),
            "quantized_quantization_values_sha256": quantized["graph"].get(
                "quantization_values_sha256"
            ),
            "official_execution_topology_sha256": official_topology,
            "quantized_execution_topology_sha256": quantized_topology,
            "official_quantization_layout_without_buffer_indices_sha256": (
                official_layout_without_buffers
            ),
            "quantized_quantization_layout_without_buffer_indices_sha256": (
                quantized_layout_without_buffers
            ),
        },
        "interpretation": (
            "The public quantizer accepted a floatified graph derived from the official "
            "topology. Buffer-inclusive fingerprint equality would establish serializer "
            "parity; the buffer-agnostic fields establish execution-topology/layout "
            "parity when true. Random learned values, private QAT, calibration, and "
            "packaging remain different."
        ),
    }
    (output_root / "converter_topology_parity_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantize a random FLOAT32 variant of an official TFLite topology."
    )
    parser.add_argument("artifact", help="Official .litertlm artifact.")
    parser.add_argument("--model-type", default="tf_lite_mtp_drafter")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-samples", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Do not write the large float/quantized model fixtures; retain only the report and recipe.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.calibration_samples < 1 or args.threads < 1:
        parser.error("--calibration-samples and --threads must be positive")
    try:
        result = run(
            args.artifact,
            model_type=args.model_type,
            seed=args.seed,
            output_dir=args.output_dir,
            calibration_samples=args.calibration_samples,
            threads=args.threads,
            write_models=not args.in_memory,
        )
    except (OSError, ConverterTopologyParityError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
