"""Read-only inspection and parity fingerprints for LiteRT-LM model files.

The ``.litertlm`` format is not a zip file.  It starts with a small versioned
header followed by FlatBuffer metadata and aligned sections (LiteRT/TFLite
models, weights, tokenizers, and runtime metadata).  This module deliberately
does not load model weights into a framework or execute inference.  It is
therefore safe to use as a model-free conversion audit.

The optional ``tflite`` Python package is used when available to inspect the
embedded TFLite graph.  It is intentionally an optional dependency because the
training package must remain usable on machines that do not have LiteRT
conversion tooling installed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import mmap
import struct
from pathlib import Path
from typing import Any, Iterable


MAGIC = b"LITERTLM"
HEADER_PREFIX_BYTES = 32
SECTION_ALIGNMENT = 16 * 1024

SECTION_TYPES = {
    0: "NONE",
    1: "GenericBinaryData",
    2: "Deprecated",
    3: "TFLiteModel",
    4: "SP_Tokenizer",
    5: "LlmMetadataProto",
    6: "HF_Tokenizer_Zlib",
    7: "TFLiteWeights",
    8: "EmbeddingMetadataProto",
    9: "ExecutorMetadataProto",
}

UNION_TYPES = {
    0: "NONE",
    1: "UInt8",
    2: "Int8",
    3: "UInt16",
    4: "Int16",
    5: "UInt32",
    6: "Int32",
    7: "Float32",
    8: "Bool",
    9: "StringValue",
    10: "UInt64",
    11: "Int64",
    12: "Double",
}


class LiteRTLMInspectionError(ValueError):
    """Raised when a LiteRT-LM file is malformed or unsupported."""


class _FlatBufferReader:
    """Minimal bounds-checked reader for the LiteRT-LM header schema.

    Keeping this parser local avoids generating and committing a large set of
    FlatBuffers Python bindings just for the small header schema.  The TFLite
    graph itself is parsed through the official generated bindings when the
    optional ``tflite`` package is installed.
    """

    def __init__(self, data: mmap.mmap):
        self.data = data
        self.size = len(data)

    def _check(self, offset: int, size: int) -> None:
        if offset < 0 or size < 0 or offset + size > self.size:
            raise LiteRTLMInspectionError(
                f"FlatBuffer read outside file: offset={offset}, size={size}, file_size={self.size}"
            )

    def u8(self, offset: int) -> int:
        self._check(offset, 1)
        return self.data[offset]

    def i32(self, offset: int) -> int:
        self._check(offset, 4)
        return struct.unpack_from("<i", self.data, offset)[0]

    def u16(self, offset: int) -> int:
        self._check(offset, 2)
        return struct.unpack_from("<H", self.data, offset)[0]

    def u32(self, offset: int) -> int:
        self._check(offset, 4)
        return struct.unpack_from("<I", self.data, offset)[0]

    def u64(self, offset: int) -> int:
        self._check(offset, 8)
        return struct.unpack_from("<Q", self.data, offset)[0]

    def f32(self, offset: int) -> float:
        self._check(offset, 4)
        return struct.unpack_from("<f", self.data, offset)[0]

    def f64(self, offset: int) -> float:
        self._check(offset, 8)
        return struct.unpack_from("<d", self.data, offset)[0]

    def follow(self, field_offset: int | None) -> int | None:
        if field_offset is None:
            return None
        relative = self.u32(field_offset)
        if relative == 0:
            return None
        target = field_offset + relative
        self._check(target, 1)
        return target

    def table_field(self, table_offset: int | None, field_index: int) -> int | None:
        """Return the absolute field value address for a FlatBuffer table."""

        if table_offset is None:
            return None
        vtable_distance = self.i32(table_offset)
        vtable = table_offset - vtable_distance
        self._check(vtable, 4)
        vtable_size = self.u16(vtable)
        entry = vtable + 4 + field_index * 2
        if entry + 2 > vtable + vtable_size:
            return None
        field_distance = self.u16(entry)
        if field_distance == 0:
            return None
        field_offset = table_offset + field_distance
        self._check(field_offset, 1)
        return field_offset

    def string_at(self, field_offset: int | None) -> str | None:
        target = self.follow(field_offset)
        if target is None:
            return None
        length = self.u32(target)
        self._check(target + 4, length)
        return bytes(self.data[target + 4 : target + 4 + length]).decode("utf-8", errors="replace")

    def vector_tables(self, field_offset: int | None) -> list[int]:
        target = self.follow(field_offset)
        if target is None:
            return []
        length = self.u32(target)
        start = target + 4
        self._check(start, length * 4)
        result: list[int] = []
        for index in range(length):
            item = self.follow(start + index * 4)
            if item is not None:
                result.append(item)
        return result


def _scalar_union_value(reader: _FlatBufferReader, table_offset: int | None, union_type: int) -> Any:
    value_field = reader.table_field(table_offset, 0)
    if value_field is None:
        return None
    if union_type == 1:
        return reader.u8(value_field)
    if union_type == 2:
        return struct.unpack("<b", bytes([reader.u8(value_field)]))[0]
    if union_type == 3:
        return reader.u16(value_field)
    if union_type == 4:
        return struct.unpack("<h", struct.pack("<H", reader.u16(value_field)))[0]
    if union_type == 5:
        return reader.u32(value_field)
    if union_type == 6:
        return reader.i32(value_field)
    if union_type == 7:
        return reader.f32(value_field)
    if union_type == 8:
        return bool(reader.u8(value_field))
    if union_type == 9:
        return reader.string_at(value_field)
    if union_type == 10:
        return reader.u64(value_field)
    if union_type == 11:
        return struct.unpack("<q", struct.pack("<Q", reader.u64(value_field)))[0]
    if union_type == 12:
        return reader.f64(value_field)
    return None


def _key_value_pair(reader: _FlatBufferReader, table_offset: int) -> dict[str, Any]:
    key = reader.string_at(reader.table_field(table_offset, 0))
    # FlatBuffers assigns the union discriminator immediately before the
    # union value field.  In the generated schema this is ``key`` (field 0),
    # ``value_type`` (field 1), ``value`` (field 2), even though the .fbs
    # source only spells out ``value:VData``.
    type_field = reader.table_field(table_offset, 1)
    union_type = reader.u8(type_field) if type_field is not None else 0
    value_table = reader.follow(reader.table_field(table_offset, 2))
    return {
        "key": key or "",
        "type": UNION_TYPES.get(union_type, f"unknown:{union_type}"),
        "value": _scalar_union_value(reader, value_table, union_type),
    }


def _key_value_items(reader: _FlatBufferReader, field_offset: int | None) -> list[dict[str, Any]]:
    return [_key_value_pair(reader, table) for table in reader.vector_tables(field_offset)]


def _metadata_from_header(reader: _FlatBufferReader, root_table: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    system_table = reader.follow(reader.table_field(root_table, 0))
    section_metadata_table = reader.follow(reader.table_field(root_table, 1))
    system_items = _key_value_items(reader, reader.table_field(system_table, 0))

    section_objects = []
    section_vector = reader.table_field(section_metadata_table, 0)
    for section_table in reader.vector_tables(section_vector):
        items = _key_value_items(reader, reader.table_field(section_table, 0))
        begin_field = reader.table_field(section_table, 1)
        end_field = reader.table_field(section_table, 2)
        type_field = reader.table_field(section_table, 3)
        begin = reader.u64(begin_field) if begin_field is not None else 0
        end = reader.u64(end_field) if end_field is not None else 0
        type_value = reader.u8(type_field) if type_field is not None else 0
        section_objects.append(
            {
                "data_type": type_value,
                "data_type_name": SECTION_TYPES.get(type_value, f"unknown:{type_value}"),
                "begin_offset": begin,
                "end_offset": end,
                "size": max(0, end - begin),
                "items": items,
            }
        )
    return {"entries": system_items}, section_objects


def _sha256_range(handle: Any, begin: int, end: int, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    handle.seek(begin)
    remaining = end - begin
    digest = hashlib.sha256()
    while remaining:
        chunk = handle.read(min(chunk_size, remaining))
        if not chunk:
            raise LiteRTLMInspectionError(f"Unexpected EOF while hashing range [{begin}, {end}).")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _safe_call(obj: Any, method_name: str, default: Any = None) -> Any:
    method = getattr(obj, method_name, None)
    if method is None:
        return default
    try:
        return method()
    except Exception:  # pragma: no cover - generated bindings vary by version
        return default


def _safe_vector(obj: Any, length_method: str, item_method: str, *, cast: Any = lambda value: value) -> list[Any]:
    length = _safe_call(obj, length_method, 0) or 0
    result = []
    method = getattr(obj, item_method, None)
    if method is None:
        return result
    for index in range(int(length)):
        try:
            result.append(cast(method(index)))
        except Exception:  # pragma: no cover - generated bindings vary by version
            result.append(None)
    return result


def _enum_names(module: Any, class_name: str) -> dict[int, str]:
    cls = getattr(module, class_name, None)
    result: dict[int, str] = {}
    if cls is not None:
        for name, value in vars(cls).items():
            if name.startswith("_") or not isinstance(value, int):
                continue
            result.setdefault(value, name)
    # Older third-party ``tflite`` wheels predate the low-bit tensor enum
    # values used by current LiteRT artifacts.  Keep the report useful when a
    # newer generated binding is not available: the official schema assigns
    # INT2=19, UINT4=20 and the two float8 encodings 21/22.  These fallbacks are
    # only labels; they do not alter or reinterpret the FlatBuffer bytes.
    if class_name == "TensorType":
        result.setdefault(19, "INT2")
        result.setdefault(20, "UINT4")
        result.setdefault(21, "FLOAT8_E4M3FN")
        result.setdefault(22, "FLOAT8_E5M2")
    return result


def _quantization_layout(quantization: Any) -> dict[str, Any] | None:
    if quantization is None:
        return None
    scales = _safe_vector(quantization, "ScaleLength", "Scale", cast=float)
    zero_points = _safe_vector(quantization, "ZeroPointLength", "ZeroPoint", cast=int)
    return {
        "scale_count": len(scales),
        "zero_point_count": len(zero_points),
        "quantized_dimension": _safe_call(quantization, "QuantizedDimension", 0),
        "min_count": _safe_call(quantization, "MinLength", 0) or 0,
        "max_count": _safe_call(quantization, "MaxLength", 0) or 0,
        "scales": scales,
        "zero_points": zero_points,
    }


def _tflite_graph_fingerprint(
    data: Any,
    *,
    include_details: bool = False,
    include_buffer_indices: bool = True,
) -> dict[str, Any]:
    try:
        model_module = importlib.import_module("tflite.Model")
        operator_module = importlib.import_module("tflite.BuiltinOperator")
        tensor_type_module = importlib.import_module("tflite.TensorType")
    except ImportError as exc:
        return {
            "available": False,
            "reason": "Install the optional 'tflite' package to inspect embedded TFLite graphs.",
            "error": str(exc),
        }

    model = model_module.Model.GetRootAsModel(data, 0)
    operator_names = _enum_names(operator_module, "BuiltinOperator")
    tensor_type_names = _enum_names(tensor_type_module, "TensorType")
    operator_codes = []
    operator_code_values = []
    for index in range(_safe_call(model, "OperatorCodesLength", 0) or 0):
        code = model.OperatorCodes(index)
        builtin = _safe_call(code, "BuiltinCode", None)
        deprecated = _safe_call(code, "DeprecatedBuiltinCode", None)
        if builtin is None:
            builtin = deprecated
        custom_code = _safe_call(code, "CustomCode", None)
        if isinstance(custom_code, (bytes, bytearray)):
            custom_code = bytes(custom_code).decode("utf-8", errors="replace")
        version = _safe_call(code, "Version", None)
        operator_codes.append(
            {
                "builtin_code": builtin,
                "builtin_name": operator_names.get(builtin, f"unknown:{builtin}"),
                "custom_code": custom_code,
                "version": version,
            }
        )
        operator_code_values.append((builtin, custom_code, version))

    subgraph_details = []
    structural_subgraphs = []
    quant_layout_subgraphs = []
    quant_value_subgraphs = []
    operator_histogram: dict[str, int] = {}
    tensor_type_histogram: dict[str, int] = {}
    quantization_layout_histogram: dict[str, int] = {}
    quantization_shape_histogram: dict[str, int] = {}
    quantized_tensor_count = 0
    for subgraph_index in range(_safe_call(model, "SubgraphsLength", 0) or 0):
        subgraph = model.Subgraphs(subgraph_index)
        name = _safe_call(subgraph, "Name", "")
        if isinstance(name, (bytes, bytearray)):
            name = bytes(name).decode("utf-8", errors="replace")
        tensors = []
        quant_layout = []
        quant_value_tensors = []
        for tensor_index in range(_safe_call(subgraph, "TensorsLength", 0) or 0):
            tensor = subgraph.Tensors(tensor_index)
            tensor_type = _safe_call(tensor, "Type", None)
            type_name = tensor_type_names.get(tensor_type, f"unknown:{tensor_type}")
            shape = _safe_vector(tensor, "ShapeLength", "Shape", cast=int)
            buffer_index = _safe_call(tensor, "Buffer", 0)
            tensor_name = _safe_call(tensor, "Name", "")
            if isinstance(tensor_name, (bytes, bytearray)):
                tensor_name = bytes(tensor_name).decode("utf-8", errors="replace")
            q = _quantization_layout(_safe_call(tensor, "Quantization", None))
            if q and (q["scale_count"] or q["zero_point_count"]):
                quantized_tensor_count += 1
                layout_key = (
                    f"{type_name}|scales={q['scale_count']}|zero_points={q['zero_point_count']}|"
                    f"quantized_dimension={q['quantized_dimension']}"
                )
                quantization_layout_histogram[layout_key] = quantization_layout_histogram.get(layout_key, 0) + 1
                shape_key = f"{layout_key}|shape={','.join(str(value) for value in shape)}"
                quantization_shape_histogram[shape_key] = quantization_shape_histogram.get(shape_key, 0) + 1
            tensor_type_histogram[type_name] = tensor_type_histogram.get(type_name, 0) + 1
            tensor_record = {"shape": shape, "type": tensor_type}
            if include_buffer_indices:
                tensor_record["buffer"] = buffer_index
            if include_details:
                tensor_record["name"] = tensor_name
                tensor_record["type_name"] = type_name
                tensor_record["quantization_layout"] = {
                    "scale_count": q["scale_count"],
                    "zero_point_count": q["zero_point_count"],
                    "quantized_dimension": q["quantized_dimension"],
                    "min_count": q["min_count"],
                    "max_count": q["max_count"],
                } if q else None
                tensor_record["quantization_values"] = q
            tensors.append(tensor_record)
            quant_value_record = {
                "shape": shape,
                "type": tensor_type,
                "quantization": q,
            }
            quant_layout_record = {
                "shape": shape,
                "type": tensor_type,
                "scale_count": q["scale_count"] if q else 0,
                "zero_point_count": q["zero_point_count"] if q else 0,
                "quantized_dimension": q["quantized_dimension"] if q else 0,
            }
            if include_buffer_indices:
                quant_value_record["buffer"] = buffer_index
                quant_layout_record["buffer"] = buffer_index
            quant_value_tensors.append(quant_value_record)
            quant_layout.append(quant_layout_record)

        operators = []
        for operator_index in range(_safe_call(subgraph, "OperatorsLength", 0) or 0):
            operator = subgraph.Operators(operator_index)
            opcode_index = _safe_call(operator, "OpcodeIndex", 0)
            opcode_record = operator_codes[opcode_index] if opcode_index < len(operator_codes) else {}
            opcode_name = opcode_record.get("builtin_name", f"opcode:{opcode_index}")
            operator_histogram[opcode_name] = operator_histogram.get(opcode_name, 0) + 1
            inputs = _safe_vector(operator, "InputsLength", "Inputs", cast=int)
            outputs = _safe_vector(operator, "OutputsLength", "Outputs", cast=int)
            operator_record = {
                "opcode_index": opcode_index,
                "builtin_name": opcode_name,
                "inputs": inputs,
                "outputs": outputs,
                "builtin_options_type": _safe_call(operator, "BuiltinOptionsType", None),
                "builtin_options2_type": _safe_call(operator, "BuiltinOptions2Type", None),
            }
            operators.append(operator_record)

        inputs = _safe_vector(subgraph, "InputsLength", "Inputs", cast=int)
        outputs = _safe_vector(subgraph, "OutputsLength", "Outputs", cast=int)
        structural_subgraphs.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "tensors": tensors,
                "operators": operators,
            }
        )
        quant_layout_subgraphs.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "tensors": quant_layout,
                "operators": [
                    {
                        "opcode_index": item["opcode_index"],
                        "builtin_name": item["builtin_name"],
                        "inputs": item["inputs"],
                        "outputs": item["outputs"],
                        "builtin_options_type": item["builtin_options_type"],
                        "builtin_options2_type": item["builtin_options2_type"],
                    }
                    for item in operators
                ],
            }
        )
        quant_value_subgraphs.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "tensors": quant_value_tensors,
                "operators": quant_layout_subgraphs[-1]["operators"],
            }
        )
        if include_details:
            subgraph_details.append({"name": name, **structural_subgraphs[-1]})

    buffers = []
    logical_buffer_sizes = []
    for index in range(_safe_call(model, "BuffersLength", 0) or 0):
        buffer = model.Buffers(index)
        data_length = _safe_call(buffer, "DataLength", 0) or 0
        external_size = _safe_call(buffer, "Size", 0) or 0
        logical_size = data_length if data_length else external_size
        buffers.append(
            {
                "index": index,
                "data_length": data_length,
                "external_size": external_size,
                "logical_size": logical_size,
            }
        )
        logical_buffer_sizes.append(int(logical_size))

    metadata = []
    for index in range(_safe_call(model, "MetadataLength", 0) or 0):
        item = model.Metadata(index)
        name = _safe_call(item, "Name", "")
        if isinstance(name, (bytes, bytearray)):
            name = bytes(name).decode("utf-8", errors="replace")
        metadata.append({"name": name, "buffer": _safe_call(item, "Buffer", 0)})

    def digest(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    structural_payload = {
        "model_version": _safe_call(model, "Version", None),
        "operator_codes": operator_code_values,
        "subgraphs": structural_subgraphs,
    }
    quant_layout_payload = {
        "operator_codes": operator_code_values,
        "subgraphs": quant_layout_subgraphs,
    }
    quant_value_payload = {
        "operator_codes": operator_code_values,
        "subgraphs": quant_value_subgraphs,
    }
    buffer_storage_payload = {
        "buffer_count": len(logical_buffer_sizes),
        "logical_sizes": logical_buffer_sizes,
    }
    summary = {
        "subgraph_count": len(structural_subgraphs),
        "operator_code_count": len(operator_codes),
        "operator_count": sum(len(item["operators"]) for item in structural_subgraphs),
        "tensor_count": sum(len(item["tensors"]) for item in structural_subgraphs),
        "buffer_count": len(buffers),
        "metadata_count": len(metadata),
        "operator_histogram": dict(sorted(operator_histogram.items())),
        "tensor_type_histogram": dict(sorted(tensor_type_histogram.items())),
        "quantized_tensor_count": quantized_tensor_count,
        "quantization_layout_histogram": dict(sorted(quantization_layout_histogram.items())),
        "quantization_shape_histogram": dict(sorted(quantization_shape_histogram.items())),
        "logical_buffer_size_total": int(sum(logical_buffer_sizes)),
    }
    result = {
        "available": True,
        "summary": summary,
        "structural_sha256": digest(structural_payload),
        "quantization_layout_sha256": digest(quant_layout_payload),
        "quantization_values_sha256": digest(quant_value_payload),
        "buffer_storage_sha256": digest(buffer_storage_payload),
    }
    if include_details:
        result["operator_codes"] = operator_codes
        result["subgraphs"] = subgraph_details
        result["buffers"] = buffers
        result["metadata"] = metadata
    return result


def inspect_litertlm(
    path: str | Path,
    *,
    include_hashes: bool = False,
    inspect_tflite: bool = True,
    include_graph_details: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable package and graph inspection report."""

    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise LiteRTLMInspectionError(f"LiteRT-LM artifact does not exist: {artifact}")
    file_size = artifact.stat().st_size
    if file_size < HEADER_PREFIX_BYTES:
        raise LiteRTLMInspectionError(f"File is too small to be a LiteRT-LM artifact: {artifact}")

    with artifact.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            if bytes(mapped[:8]) != MAGIC:
                raise LiteRTLMInspectionError(f"Invalid LiteRT-LM magic in {artifact}")
            major, minor, patch = struct.unpack_from("<III", mapped, 8)
            header_end_offset = struct.unpack_from("<Q", mapped, 24)[0]
            if header_end_offset < HEADER_PREFIX_BYTES or header_end_offset > file_size:
                raise LiteRTLMInspectionError(
                    f"Invalid header end offset {header_end_offset} for file size {file_size}."
                )
            reader = _FlatBufferReader(mapped)
            root = reader.follow(32)
            if root is None:
                raise LiteRTLMInspectionError("LiteRT-LM header FlatBuffer root is null.")
            system_metadata, sections = _metadata_from_header(reader, root)
            previous_end = header_end_offset
            section_reports = []
            graph_reports = []
            for index, section in enumerate(sections):
                begin = section["begin_offset"]
                end = section["end_offset"]
                if begin < header_end_offset or end < begin or end > file_size:
                    raise LiteRTLMInspectionError(
                        f"Section {index} has invalid range [{begin}, {end}) for file size {file_size}."
                    )
                section_report = dict(section)
                section_report.update(
                    {
                        "index": index,
                        "alignment_ok": begin % SECTION_ALIGNMENT == 0,
                        "ordered_after_previous": begin >= previous_end,
                    }
                )
                if include_hashes:
                    section_report["sha256"] = _sha256_range(handle, begin, end)
                if inspect_tflite and section["data_type_name"] == "TFLiteModel":
                    section_view = memoryview(mapped)[begin:end]
                    try:
                        graph_report = _tflite_graph_fingerprint(
                            section_view, include_details=include_graph_details
                        )
                    finally:
                        section_view.release()
                    graph_report["section_index"] = index
                    graph_reports.append(graph_report)
                    section_report["graph"] = graph_report
                section_reports.append(section_report)
                previous_end = end
            header_digest = hashlib.sha256(mapped[:header_end_offset]).hexdigest()
        finally:
            mapped.close()

    return {
        "format": "litertlm",
        "path": str(artifact),
        "file_size": file_size,
        "header": {
            "magic": MAGIC.decode("ascii"),
            "version": {"major": major, "minor": minor, "patch": patch},
            "header_end_offset": header_end_offset,
            "header_sha256": header_digest,
            "system_metadata": system_metadata,
        },
        "sections": section_reports,
        "graphs": graph_reports,
        "weights": {
            "section_count": sum(1 for item in section_reports if item["data_type_name"] == "TFLiteWeights"),
            "hashed": include_hashes,
        },
    }


def compare_litertlm_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare independent evidence levels without conflating random weights and graph parity."""

    left_sections = left.get("sections") or []
    right_sections = right.get("sections") or []
    left_graphs = left.get("graphs") or []
    right_graphs = right.get("graphs") or []

    section_layout_left = [
        (item.get("data_type_name"), item.get("size"), item.get("items")) for item in left_sections
    ]
    section_layout_right = [
        (item.get("data_type_name"), item.get("size"), item.get("items")) for item in right_sections
    ]

    def values(key: str) -> list[Any]:
        return [item.get(key) for item in left_graphs], [item.get(key) for item in right_graphs]

    left_structural, right_structural = values("structural_sha256")
    left_quant_layout, right_quant_layout = values("quantization_layout_sha256")
    left_quant_values, right_quant_values = values("quantization_values_sha256")

    left_weights = [item.get("sha256") for item in left_sections if item.get("data_type_name") == "TFLiteWeights"]
    right_weights = [item.get("sha256") for item in right_sections if item.get("data_type_name") == "TFLiteWeights"]
    weights_known = bool(left_weights and right_weights)
    left_model_payloads = [item.get("sha256") for item in left_sections if item.get("data_type_name") == "TFLiteModel"]
    right_model_payloads = [item.get("sha256") for item in right_sections if item.get("data_type_name") == "TFLiteModel"]
    model_payloads_known = bool(left_model_payloads and right_model_payloads)
    return {
        "header_metadata_match": left.get("header", {}).get("system_metadata")
        == right.get("header", {}).get("system_metadata"),
        "section_layout_match": section_layout_left == section_layout_right,
        "graph_count_match": len(left_graphs) == len(right_graphs),
        "graph_structure_match": bool(left_structural and left_structural == right_structural),
        "quantization_layout_match": bool(left_quant_layout and left_quant_layout == right_quant_layout),
        "quantization_values_match": bool(left_quant_values and left_quant_values == right_quant_values),
        "weight_bytes_match": (left_weights == right_weights) if weights_known else None,
        "weight_bytes_comparison_available": weights_known,
        "tflite_model_section_bytes_match": (
            left_model_payloads == right_model_payloads if model_payloads_known else None
        ),
        "tflite_model_section_comparison_available": model_payloads_known,
        "interpretation": (
            "Graph and quantization-layout matches establish structural parity only. "
            "Random-weight exports cannot establish learned-weight or private calibration parity."
        ),
    }


def inspect_and_compare(
    left_path: str | Path,
    right_path: str | Path,
    *,
    include_hashes: bool = False,
    inspect_tflite: bool = True,
    include_graph_details: bool = False,
) -> dict[str, Any]:
    left = inspect_litertlm(
        left_path,
        include_hashes=include_hashes,
        inspect_tflite=inspect_tflite,
        include_graph_details=include_graph_details,
    )
    right = inspect_litertlm(
        right_path,
        include_hashes=include_hashes,
        inspect_tflite=inspect_tflite,
        include_graph_details=include_graph_details,
    )
    return {"left": left, "right": right, "comparison": compare_litertlm_reports(left, right)}


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect and compare LiteRT-LM package and graph fingerprints.")
    parser.add_argument("artifact", help="Path to a .litertlm artifact.")
    parser.add_argument("--compare", help="Optional second artifact to compare against the first.")
    parser.add_argument("--include-hashes", action="store_true", help="Hash each section (reads the full artifact).")
    parser.add_argument("--no-tflite", action="store_true", help="Skip embedded TFLite graph inspection.")
    parser.add_argument("--graph-details", action="store_true", help="Include full tensor/operator details in JSON.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.compare:
            report = inspect_and_compare(
                args.artifact,
                args.compare,
                include_hashes=args.include_hashes,
                inspect_tflite=not args.no_tflite,
                include_graph_details=args.graph_details,
            )
        else:
            report = inspect_litertlm(
                args.artifact,
                include_hashes=args.include_hashes,
                inspect_tflite=not args.no_tflite,
                include_graph_details=args.graph_details,
            )
    except (OSError, LiteRTLMInspectionError) as exc:
        parser.error(str(exc))
        return 2
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
