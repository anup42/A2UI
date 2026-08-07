"""Quantize random branches with AI Edge, then inject them into official topology.

This harness closes the gap between the two existing experiments:

* ``build_converter_random_inventory_parity.py`` really invokes the public
  AI Edge Quantizer, but its standalone graph is not the official network.
* ``build_random_official_topology_parity.py`` preserves the official network,
  but writes quantized constants directly instead of obtaining them from the
  converter.

Here, each observable official FC/embedding weight becomes an independent
random FLOAT32 branch, the branch graph is quantized by AI Edge Quantizer, and
the resulting packed bytes/scales are copied into a read-only copy of the
official TFLite section.  Only the constants change; operators, tensors,
signatures, metadata, and runtime wiring stay official.  The report compares
both the independent converter layout and the injected official graph.

With ``--rebuild-flatbuffer``, the same converter constants are instead placed
into a public ``ModelT`` object and packed into a new FlatBuffer.  External
buffers are materialized before packing so stale source offsets cannot make the
fixture accidentally depend on the official serialization.  This is the
strongest random-network check in this repository: it compares the rebuilt
operator/tensor graph, quantization layout, converter constants, and optional
LiteRT allocation against the official section.

This proves that public converter-produced random constants can inhabit the
official network.  It cannot reproduce Google's learned values, private QAT
observers, calibration data, exporter revision, or package bytes.

Very large sections (notably Gemma 4 E2B prefill) are quantized in deterministic
small batches so the independent float32 FlatBuffer does not cross
FlatBuffers' 2 GiB offset limit.  Global branch ordinals keep the combined
inventory and injection mapping stable across those batches.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import gc
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_gemma4_mobile_checkpoint_parity import _buffer_bytes  # noqa: E402
from build_converter_random_inventory_parity import (  # noqa: E402
    _build_random_float_tflite,
    _counter,
    _extract_inventory,
    _recipe,
    _vector,
)
from build_converter_topology_parity import (  # noqa: E402
    _FULLY_CONNECTED,
    _LOW_BIT_TYPES,
    _opcode_builtin,
    _read_section,
    _section_by_model_type,
    _tensor_shape,
    _quantize as _topology_quantize,
    _unpack_model,
)
from build_fresh_random_quantized_graph import _schema_model  # noqa: E402
from build_random_official_topology_parity import (  # noqa: E402
    _buffer_view,
    _runtime_allocate_report,
)
from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    _tflite_graph_fingerprint,
    inspect_litertlm,
)


class ConverterRandomTopologyInjectionError(RuntimeError):
    """Raised when converter output cannot be injected safely."""


_EMBEDDING_LOOKUP = 7
_TENSOR_TYPE_NAMES = {
    0: "FLOAT32",
    2: "INT32",
    9: "INT8",
    17: "INT4",
    19: "INT2",
    20: "UINT4",
}


def _sha256(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _constants_digest(records: Iterable[dict[str, Any]]) -> str:
    """Hash ordered packed values and float32 scales for a weight inventory."""

    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.get("operator", "")).encode("utf-8"))
        digest.update(int(record["bits"]).to_bytes(2, "little", signed=False))
        for value in record["shape"]:
            digest.update(int(value).to_bytes(8, "little", signed=False))
        digest.update(bytes(record["raw"]))
        digest.update(np.asarray(record["scales"], dtype=np.float32).reshape(-1).tobytes())
    return digest.hexdigest()


def _official_constants_digest(section_bytes: bytes, records: list[dict[str, Any]]) -> str:
    """Read the patched official buffers/scales in the inventory order."""

    mutable = bytearray(section_bytes)
    model = _schema_model(mutable)
    locations = _official_weight_locations(model, records)
    observed: list[dict[str, Any]] = []
    for record, tensor in zip(records, locations):
        view_info = _buffer_view(model, int(tensor.Buffer()), mutable)
        if view_info is None:
            raise ConverterRandomTopologyInjectionError(
                f"Could not read patched official buffer {tensor.Buffer()}."
            )
        view, _, _, _ = view_info
        quantization = tensor.Quantization()
        if quantization is None or quantization.ScaleAsNumpy() is None:
            raise ConverterRandomTopologyInjectionError(
                f"Patched official tensor has no scales: {record.get('name', '')}."
            )
        observed.append(
            {
                "operator": record.get("operator", ""),
                "bits": record["bits"],
                "shape": record["shape"],
                "raw": bytes(view),
                "scales": np.asarray(quantization.ScaleAsNumpy(), dtype=np.float32).reshape(-1),
            }
        )
    return _constants_digest(observed)


def _file_range_sha256(path: Path, start: int, size: int, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash one package range without loading the complete package."""

    digest = hashlib.sha256()
    remaining = int(size)
    with path.open("rb") as handle:
        handle.seek(int(start))
        while remaining:
            block = handle.read(min(chunk_size, remaining))
            if not block:
                raise ConverterRandomTopologyInjectionError(
                    f"Could not read package range [{start}, {start + size})."
                )
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _package_boundary_report(
    artifact_path: Path,
    section: dict[str, Any],
    injected_section: bytes,
) -> dict[str, Any]:
    """Prove that an injected fixture changes only the selected package section."""

    package_size = artifact_path.stat().st_size
    begin = int(section["begin_offset"])
    size = int(section["size"])
    end = begin + size
    if begin < 0 or size < 0 or end > package_size:
        raise ConverterRandomTopologyInjectionError(
            f"Selected section [{begin}, {end}) is outside package size {package_size}."
        )
    if len(injected_section) != size:
        raise ConverterRandomTopologyInjectionError(
            f"Injected section size {len(injected_section)} differs from package section {size}."
        )
    prefix_size = begin
    suffix_size = package_size - end
    return {
        "package_size": int(package_size),
        "selected_section_offset": begin,
        "selected_section_size": size,
        "injected_section_size": len(injected_section),
        "prefix_sha256": _file_range_sha256(artifact_path, 0, prefix_size)
        if prefix_size
        else _sha256(b""),
        "suffix_sha256": _file_range_sha256(artifact_path, end, suffix_size)
        if suffix_size
        else _sha256(b""),
        "bytes_outside_selected_section_unchanged": True,
        "package_replacement_model": (
            "The injected fixture is formed by replacing exactly the selected TFLite "
            "section at this offset/size and copying every prefix/suffix byte unchanged."
        ),
    }


def _write_injected_package(
    artifact_path: Path,
    section: dict[str, Any],
    injected_section: bytes,
    output_path: Path,
) -> None:
    """Stream a full package fixture while retaining all non-target bytes."""

    if output_path.exists():
        raise ConverterRandomTopologyInjectionError(
            f"Refusing to overwrite package output: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    begin = int(section["begin_offset"])
    size = int(section["size"])
    with artifact_path.open("rb") as source, output_path.open("wb") as target:
        remaining = begin
        while remaining:
            block = source.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ConverterRandomTopologyInjectionError(
                    "Could not copy package prefix before selected section."
                )
            target.write(block)
            remaining -= len(block)
        target.write(injected_section)
        source.seek(begin + size)
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _quantization_summary(tensor: Any) -> dict[str, Any] | None:
    quantization = getattr(tensor, "quantization", None)
    if quantization is None:
        return None
    scale_values = getattr(quantization, "scale", None)
    zero_point_values = getattr(quantization, "zeroPoint", None)
    scales = np.asarray(
        [] if scale_values is None else scale_values, dtype=np.float32
    )
    zero_points = np.asarray(
        [] if zero_point_values is None else zero_point_values, dtype=np.int64
    )
    if not len(scales) and not len(zero_points):
        return None
    return {
        "scale_count": int(len(scales)),
        "zero_point_count": int(len(zero_points)),
        "quantized_dimension": int(getattr(quantization, "quantizedDimension", 0)),
        "zero_points_all_zero": bool(
            len(zero_points) and np.all(zero_points == 0)
        ),
    }


def _converter_weight_records(data: bytes) -> list[dict[str, Any]]:
    """Extract converter-packed constants by their stable branch ordinal."""

    model, _, _ = _unpack_model(data)
    records: list[dict[str, Any]] = []
    seen_buffers: set[int] = set()
    branch_pattern = re.compile(r"inventory_w_(\d+)_w(\d+)_weight$")
    for subgraph_index, subgraph in enumerate(model.subgraphs or []):
        for operator_index, operator in enumerate(subgraph.operators or []):
            builtin = _opcode_builtin(model, operator)
            if builtin not in (_FULLY_CONNECTED, _EMBEDDING_LOOKUP):
                continue
            inputs = [int(value) for value in np.asarray(operator.inputs).reshape(-1)]
            outputs = [int(value) for value in np.asarray(operator.outputs).reshape(-1)]
            if len(inputs) < 2 or inputs[1] < 0 or not outputs:
                continue
            weight = subgraph.tensors[inputs[1]]
            name = _decode(weight.name)
            match = branch_pattern.fullmatch(name)
            if not match:
                continue
            buffer_index = int(weight.buffer)
            if buffer_index in seen_buffers:
                continue
            seen_buffers.add(buffer_index)
            bits = _LOW_BIT_TYPES.get(int(weight.type))
            if bits is None or weight.quantization is None:
                continue
            records.append(
                {
                    "ordinal": int(match.group(1)),
                    "subgraph": subgraph_index,
                    "operator_index": operator_index,
                    "bits": int(bits),
                    "operator": (
                        "EMBEDDING_LOOKUP"
                        if builtin == _EMBEDDING_LOOKUP
                        else "FULLY_CONNECTED"
                    ),
                    "shape": list(_tensor_shape(weight)),
                    "name": name,
                    "buffer": buffer_index,
                    "raw": _buffer_bytes(model, data, buffer_index),
                    "scales": np.asarray(
                        weight.quantization.scale, dtype=np.float32
                    ).reshape(-1),
                    "scale_count": int(len(weight.quantization.scale)),
                    "zero_point_count": int(len(weight.quantization.zeroPoint)),
                    "quantized_dimension": int(weight.quantization.quantizedDimension),
                    "zero_points_all_zero": bool(
                        len(weight.quantization.zeroPoint)
                        and all(int(value) == 0 for value in weight.quantization.zeroPoint)
                    ),
                    "input_quantization": _quantization_summary(
                        subgraph.tensors[inputs[0]]
                    ),
                    "output_quantization": _quantization_summary(
                        subgraph.tensors[outputs[0]]
                    ),
                    "input_type": _TENSOR_TYPE_NAMES.get(
                        int(subgraph.tensors[inputs[0]].type),
                        str(int(subgraph.tensors[inputs[0]].type)),
                    ),
                    "output_type": _TENSOR_TYPE_NAMES.get(
                        int(subgraph.tensors[outputs[0]].type),
                        str(int(subgraph.tensors[outputs[0]].type)),
                    ),
                }
            )
    records.sort(key=lambda item: int(item["ordinal"]))
    return records


def _compact_converter_layout(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop packed bytes/scales before putting converter records in JSON."""

    fields = (
        "ordinal",
        "subgraph",
        "operator_index",
        "operator",
        "bits",
        "shape",
        "name",
        "buffer",
        "scale_count",
        "zero_point_count",
        "quantized_dimension",
        "zero_points_all_zero",
        "input_type",
        "output_type",
        "input_quantization",
        "output_quantization",
    )
    return [{key: record.get(key) for key in fields} for record in records]


def _official_weight_locations(model: Any, records: list[dict[str, Any]]) -> list[Any]:
    """Resolve official inventory records to mutable object-API weight tensors."""

    locations: list[Any] = []
    for record in records:
        subgraph = model.Subgraphs(int(record["official_subgraph"]))
        operator = subgraph.Operators(int(record["official_operator_index"]))
        inputs = _vector(operator, "InputsLength", "Inputs")
        if len(inputs) < 2 or inputs[1] < 0:
            raise ConverterRandomTopologyInjectionError(
                f"Official inventory record has no weight input: {record}"
            )
        locations.append(subgraph.Tensors(inputs[1]))
    return locations


def _patch_official_constants(
    section_bytes: bytes,
    records: list[dict[str, Any]],
    converter_records: list[dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    """Copy converter bytes/scales into a mutable official section."""

    if len(records) != len(converter_records):
        raise ConverterRandomTopologyInjectionError(
            f"Inventory count differs: official={len(records)} "
            f"converter={len(converter_records)}"
        )
    converter_by_ordinal = {
        int(record["ordinal"]): record for record in converter_records
    }
    if sorted(converter_by_ordinal) != list(range(len(records))):
        raise ConverterRandomTopologyInjectionError(
            "Converter output does not contain one contiguous branch per inventory record."
        )

    mutable = bytearray(section_bytes)
    model = _schema_model(mutable)
    locations = _official_weight_locations(model, records)
    patched = 0
    skipped: collections.Counter[str] = collections.Counter()
    for ordinal, (record, tensor) in enumerate(zip(records, locations)):
        converter = converter_by_ordinal[ordinal]
        if int(record["bits"]) != int(converter["bits"]):
            raise ConverterRandomTopologyInjectionError(
                f"W{record['bits']} vs W{converter['bits']} at ordinal {ordinal}."
            )
        if tuple(int(value) for value in record["shape"]) != tuple(
            int(value) for value in converter["shape"]
        ):
            raise ConverterRandomTopologyInjectionError(
                f"Shape mismatch at ordinal {ordinal}: "
                f"official={record['shape']} converter={converter['shape']}"
            )
        raw_view_info = _buffer_view(model, int(tensor.Buffer()), mutable)
        if raw_view_info is None:
            skipped["buffer_storage_unavailable"] += 1
            continue
        raw_view, _, _, data_length = raw_view_info
        if data_length != len(converter["raw"]):
            raise ConverterRandomTopologyInjectionError(
                f"Packed size mismatch at ordinal {ordinal}: "
                f"official={data_length} converter={len(converter['raw'])}"
            )
        raw_view[:] = np.frombuffer(converter["raw"], dtype=np.uint8)
        quantization = tensor.Quantization()
        if quantization is None:
            skipped["official_quantization_unavailable"] += 1
            continue
        scale_view = quantization.ScaleAsNumpy()
        if scale_view is None or len(scale_view) != len(converter["scales"]):
            raise ConverterRandomTopologyInjectionError(
                f"Scale count mismatch at ordinal {ordinal}: "
                f"official={0 if scale_view is None else len(scale_view)} "
                f"converter={len(converter['scales'])}"
            )
        scale_view[:] = converter["scales"]
        zero_view = quantization.ZeroPointAsNumpy()
        if zero_view is not None and len(zero_view):
            zero_view[:] = 0
        patched += 1
    return bytes(mutable), {
        "patched_weight_count": patched,
        "skipped": dict(sorted(skipped.items())),
    }


def _rebuild_graph_with_constants(
    section_bytes: bytes,
    records: list[dict[str, Any]],
    converter_records: list[dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    """Re-serialize an independent ModelT graph with converter constants.

    The normal injection path edits the official FlatBuffer in place.  This
    companion path deserializes the graph into the public generated ``ModelT``
    object API, materializes external buffers, replaces the selected random
    packed constants/scales, and packs a new FlatBuffer.  Rebuilding catches
    accidental dependence on official byte offsets and gives a stronger graph
    parity check while keeping the official section read-only.
    """

    try:
        import flatbuffers
        from ai_edge_litert import schema_py_generated as schema
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise ConverterRandomTopologyInjectionError(
            "Independent graph rebuilding requires flatbuffers and "
            "ai_edge_litert.schema_py_generated."
        ) from exc

    if len(records) != len(converter_records):
        raise ConverterRandomTopologyInjectionError(
            f"Inventory count differs: official={len(records)} "
            f"converter={len(converter_records)}"
        )
    converter_by_ordinal = {
        int(record["ordinal"]): record for record in converter_records
    }
    if sorted(converter_by_ordinal) != list(range(len(records))):
        raise ConverterRandomTopologyInjectionError(
            "Converter output does not contain one contiguous branch per inventory record."
        )

    root = schema.Model.GetRootAsModel(section_bytes, 0)
    model = schema.ModelT.InitFromObj(root)
    materialized_external = 0
    materialized_bytes = 0
    for buffer in model.buffers or []:
        if buffer.data is not None:
            continue
        offset = int(getattr(buffer, "offset", 0) or 0)
        size = int(getattr(buffer, "size", 0) or 0)
        if offset <= 0 or size <= 0:
            continue
        end = offset + size
        if end > len(section_bytes):
            raise ConverterRandomTopologyInjectionError(
                f"External buffer [{offset}, {end}) exceeds section size {len(section_bytes)}."
            )
        buffer.data = np.frombuffer(
            section_bytes[offset:end], dtype=np.uint8
        ).copy()
        buffer.offset = 0
        buffer.size = 0
        materialized_external += 1
        materialized_bytes += size

    replaced = 0
    seen_buffers: set[int] = set()
    for ordinal, record in enumerate(records):
        converter = converter_by_ordinal[ordinal]
        if int(record["bits"]) != int(converter["bits"]):
            raise ConverterRandomTopologyInjectionError(
                f"W{record['bits']} vs W{converter['bits']} at ordinal {ordinal}."
            )
        if tuple(int(value) for value in record["shape"]) != tuple(
            int(value) for value in converter["shape"]
        ):
            raise ConverterRandomTopologyInjectionError(
                f"Shape mismatch at ordinal {ordinal}: "
                f"official={record['shape']} converter={converter['shape']}"
            )
        subgraph = model.subgraphs[int(record["official_subgraph"])]
        operator = subgraph.operators[int(record["official_operator_index"])]
        inputs = np.asarray(operator.inputs, dtype=np.int64).reshape(-1)
        if len(inputs) < 2 or int(inputs[1]) < 0:
            raise ConverterRandomTopologyInjectionError(
                f"Official inventory record has no weight input at ordinal {ordinal}."
            )
        tensor_index = int(inputs[1])
        tensor = subgraph.tensors[tensor_index]
        buffer_index = int(tensor.buffer)
        if buffer_index in seen_buffers:
            continue
        seen_buffers.add(buffer_index)
        buffer = model.buffers[buffer_index]
        raw = np.frombuffer(converter["raw"], dtype=np.uint8).copy()
        buffer.data = raw
        buffer.offset = 0
        buffer.size = 0
        quantization = tensor.quantization
        if quantization is None:
            raise ConverterRandomTopologyInjectionError(
                f"Official tensor has no quantization at ordinal {ordinal}."
            )
        quantization.scale = np.asarray(converter["scales"], dtype=np.float32).copy()
        quantization.zeroPoint = np.zeros(
            len(converter["scales"]), dtype=np.int64
        )
        replaced += 1

    builder = flatbuffers.Builder(max(1024, min(len(section_bytes), 64 * 1024 * 1024)))
    root_offset = model.Pack(builder)
    builder.Finish(root_offset, file_identifier=b"TFL3")
    rebuilt = bytes(builder.Output())
    return rebuilt, {
        "replaced_weight_count": replaced,
        "materialized_external_buffer_count": materialized_external,
        "materialized_external_bytes": materialized_bytes,
        "modelt_rebuilt": True,
    }


def _graph_report(data: bytes) -> dict[str, Any]:
    return {
        "size": len(data),
        "sha256": _sha256(data),
        "graph": _tflite_graph_fingerprint(data, include_details=False),
        "graph_without_buffer_indices": _tflite_graph_fingerprint(
            data, include_details=False, include_buffer_indices=False
        ),
    }


def _random_quantized_network_match(
    *,
    converter_inventory_match: bool,
    converter_fc_match: bool,
    converter_embedding_match: bool,
    official_structure_match: bool,
    official_layout_match: bool,
    execution_topology_match: bool,
    execution_layout_match: bool,
) -> dict[str, Any]:
    """Summarize network parity while keeping value parity separate.

    The random branch is deliberately not expected to have Google's learned
    constants.  This helper answers the narrower reverse-engineering question:
    did the public converter produce the same observable weight layout, and can
    those constants inhabit the released operator/tensor graph?  The
    serializer-sensitive and buffer-index-independent checks are both retained
    so FlatBuffer reserialization is not mistaken for a topology change.
    """

    checks = {
        "converter_inventory_match": bool(converter_inventory_match),
        "converter_fc_match": bool(converter_fc_match),
        "converter_embedding_match": bool(converter_embedding_match),
        "official_structure_match": bool(official_structure_match),
        "official_layout_match": bool(official_layout_match),
        "execution_topology_match_ignoring_buffer_indices": bool(
            execution_topology_match
        ),
        "execution_layout_match_ignoring_buffer_indices": bool(
            execution_layout_match
        ),
    }
    return {
        "same": bool(all(checks.values())),
        "scope": "topology_and_quantization_layout_only",
        "checks": checks,
        "not_value_or_model_identity": True,
    }


def _converter_batch_ranges(
    records: list[dict[str, Any]], batch_size: int | None
) -> tuple[int, list[tuple[int, int]]]:
    """Choose safe independent-graph batches for a potentially huge inventory.

    A Gemma 4 E2B prefill inventory contains an approximately 1.5 GiB float32
    embedding matrix.  Building all 277 random branches in one FlatBuffer
    exceeds FlatBuffers' 2 GiB offset limit, even though the official packed
    section is valid.  Keep ordinary small sections in one graph and split
    large sections into deterministic batches of eight by default.
    """

    if batch_size is not None and int(batch_size) < 1:
        raise ConverterRandomTopologyInjectionError(
            "converter batch size must be positive"
        )
    estimated_bytes = sum(
        int(np.prod(tuple(int(value) for value in record["shape"]), dtype=np.int64))
        * 4
        for record in records
    )
    if batch_size is not None:
        chosen = int(batch_size)
        ranges = [
            (start, min(start + chosen, len(records)))
            for start in range(0, len(records), chosen)
        ]
    elif estimated_bytes <= 1536 * 1024 * 1024:
        chosen = max(1, len(records))
        ranges = [(0, len(records))]
    else:
        # Keep automatic graphs comfortably below FlatBuffers' 2 GiB offset
        # limit even when a future E2B variant changes matrix shapes.  Eight is
        # the observed E2B batch size; the byte guard can split it further.
        chosen = 8
        target_bytes = 1750 * 1024 * 1024
        item_bytes = [
            int(
                np.prod(
                    tuple(int(value) for value in record["shape"]),
                    dtype=np.int64,
                )
            )
            * 4
            for record in records
        ]
        ranges = []
        start = 0
        while start < len(records):
            end = start
            total = 0
            while end < len(records) and end < start + chosen:
                next_total = total + item_bytes[end]
                if end > start and next_total > target_bytes:
                    break
                total = next_total
                end += 1
            ranges.append((start, end))
            start = end
    return chosen, ranges


def run(
    artifact: str | Path,
    *,
    model_type: str,
    seed: int,
    output_dir: str | Path,
    calibration_samples: int,
    threads: int,
    max_weights: int | None = None,
    write_models: bool = True,
    runtime_allocate: bool = False,
    runtime_threads: int = 2,
    runtime_without_default_delegates: bool = False,
    package_output: str | Path | None = None,
    converter_batch_size: int | None = None,
    rebuild_graph: bool = False,
) -> dict[str, Any]:
    artifact_path = Path(artifact).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not artifact_path.is_file():
        raise ConverterRandomTopologyInjectionError(
            f"Official artifact does not exist: {artifact_path}"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise ConverterRandomTopologyInjectionError(
            f"Output directory is non-empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        package = inspect_litertlm(artifact_path, inspect_tflite=False)
        section = _section_by_model_type(package, model_type)
    except (OSError, LiteRTLMInspectionError) as exc:
        raise ConverterRandomTopologyInjectionError(
            f"Could not inspect official artifact: {exc}"
        ) from exc

    official_bytes = _read_section(artifact_path, section)
    _, official_records = _extract_inventory(
        artifact_path,
        model_type,
        include_embeddings=True,
        max_weights=max_weights,
    )
    if not official_records:
        raise ConverterRandomTopologyInjectionError(
            f"No quantized FC/embedding weights found for {model_type}."
        )
    if max_weights is not None and max_weights < len(official_records):
        raise ConverterRandomTopologyInjectionError(
            "The inventory extractor returned more records than --max-weights."
        )
    # A bounded inventory cannot be injected safely if the selected records do
    # not form a prefix of the official buffer traversal.  The extractor's
    # max_weights behavior is prefix-based, so this remains deterministic.
    chosen_batch_size, batch_ranges = _converter_batch_ranges(
        official_records, converter_batch_size
    )
    converter_records: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    recipe: list[dict[str, Any]] = []
    converter_batches: list[dict[str, Any]] = []
    quantized_paths: list[Path] = []
    converter_digest = hashlib.sha256()
    converter_total_size = 0
    for batch_index, (batch_start, batch_end) in enumerate(batch_ranges):
        batch_records = official_records[batch_start:batch_end]
        fp32_bytes, batch_mappings = _build_random_float_tflite(
            batch_records, seed, ordinal_offset=batch_start
        )
        batch_recipe = _recipe(batch_records, ordinal_offset=batch_start)
        mappings.extend(batch_mappings)
        recipe.extend(batch_recipe)
        fp32_path = output_root / (
            "random_converter_topology_fp32.tflite"
            if len(batch_ranges) == 1
            else f"random_converter_topology_fp32_batch_{batch_index:04d}.tflite"
        )
        quantized_path = output_root / (
            "random_converter_topology_quantized.tflite"
            if len(batch_ranges) == 1
            else f"random_converter_topology_quantized_batch_{batch_index:04d}.tflite"
        )
        if write_models:
            fp32_path.write_bytes(fp32_bytes)
        _topology_quantize(
            fp32_path if write_models else fp32_bytes,
            quantized_path,
            batch_recipe,
            calibration_samples=calibration_samples,
            threads=threads,
        )
        quantized_bytes = quantized_path.read_bytes()
        batch_converter_records = _converter_weight_records(quantized_bytes)
        expected_ordinals = list(range(batch_start, batch_end))
        observed_ordinals = [
            int(record["ordinal"]) for record in batch_converter_records
        ]
        if observed_ordinals != expected_ordinals:
            raise ConverterRandomTopologyInjectionError(
                "Converter batch ordinals do not cover the selected global "
                f"range {expected_ordinals[0]}..{expected_ordinals[-1]}: "
                f"{observed_ordinals[:8]}..."
            )
        converter_records.extend(batch_converter_records)
        converter_digest.update(quantized_bytes)
        converter_total_size += len(quantized_bytes)
        quantized_paths.append(quantized_path)
        converter_batches.append(
            {
                "index": batch_index,
                "start_ordinal": batch_start,
                "end_ordinal_exclusive": batch_end,
                "weight_count": batch_end - batch_start,
                "fp32_size": len(fp32_bytes),
                "quantized_size": len(quantized_bytes),
                "fp32_path": str(fp32_path) if write_models else None,
                "quantized_path": str(quantized_path) if write_models else None,
            }
        )
    recipe_path = output_root / "random_converter_topology_recipe.json"
    recipe_path.write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    converter_observed_layout = _compact_converter_layout(converter_records)
    official_fc_records = [
        record
        for record in official_records
        if str(record.get("operator") or "FULLY_CONNECTED").upper()
        == "FULLY_CONNECTED"
    ]
    official_non_fc_records = [
        record
        for record in official_records
        if str(record.get("operator") or "FULLY_CONNECTED").upper()
        != "FULLY_CONNECTED"
    ]
    converter_fc_records = [
        record
        for record in converter_records
        if str(record.get("operator") or "FULLY_CONNECTED").upper()
        == "FULLY_CONNECTED"
    ]
    converter_non_fc_records = [
        record
        for record in converter_records
        if str(record.get("operator") or "FULLY_CONNECTED").upper()
        != "FULLY_CONNECTED"
    ]
    all_converter_layout_match = _counter(official_records) == _counter(converter_records)
    fc_converter_layout_match = _counter(official_fc_records) == _counter(converter_fc_records)
    embedding_converter_layout_match = _counter(
        [
            record
            for record in official_non_fc_records
            if record.get("operator") == "EMBEDDING_LOOKUP"
        ]
    ) == _counter(
        [
            record
            for record in converter_non_fc_records
            if record.get("operator") == "EMBEDDING_LOOKUP"
        ]
    )
    injected_bytes, injection = _patch_official_constants(
        official_bytes, official_records, converter_records
    )
    converter_constants_sha256 = _constants_digest(converter_records)
    injected_constants_sha256 = _official_constants_digest(
        injected_bytes, official_records
    )
    injected_path = output_root / "random_converter_topology_injected.tflite"
    if write_models:
        injected_path.write_bytes(injected_bytes)
    package_boundary = _package_boundary_report(
        artifact_path, section, injected_bytes
    )
    official_graph = _graph_report(official_bytes)
    injected_graph = _graph_report(injected_bytes)
    official_structural = official_graph["graph"].get("structural_sha256")
    injected_structural = injected_graph["graph"].get("structural_sha256")
    official_layout = official_graph["graph"].get("quantization_layout_sha256")
    injected_layout = injected_graph["graph"].get("quantization_layout_sha256")
    official_topology = official_graph["graph_without_buffer_indices"].get(
        "structural_sha256"
    )
    injected_topology = injected_graph["graph_without_buffer_indices"].get(
        "structural_sha256"
    )
    official_layout_no_buffers = official_graph["graph_without_buffer_indices"].get(
        "quantization_layout_sha256"
    )
    injected_layout_no_buffers = injected_graph[
        "graph_without_buffer_indices"
    ].get("quantization_layout_sha256")
    result: dict[str, Any] = {
        "artifact": str(artifact_path),
        "model_type": model_type,
        "seed": int(seed),
        "max_weights": max_weights,
        "calibration_samples": int(calibration_samples),
        "threads": int(threads),
        "converter_batch_size": int(chosen_batch_size),
        "converter_batch_count": len(converter_batches),
        "estimated_float32_inventory_bytes": int(
            sum(
                int(
                    np.prod(
                        tuple(int(value) for value in record["shape"]),
                        dtype=np.int64,
                    )
                )
                * 4
                for record in official_records
            )
        ),
        "training_executed": False,
        "random_initialization": True,
        "public_converter_executed": True,
        "private_qat_recipe_recovered": False,
        "official_inventory_count": len(official_records),
        "converter_weight_count": len(converter_records),
        "official_operator_counts": dict(
            sorted(collections.Counter(str(record.get("operator")) for record in official_records).items())
        ),
        "converter_operator_counts": dict(
            sorted(collections.Counter(str(record.get("operator")) for record in converter_records).items())
        ),
        "official_non_fc_count": len(official_non_fc_records),
        "converter_non_fc_count": len(converter_non_fc_records),
        "official_section": official_graph,
        "converter_variant": {
            "size": int(converter_total_size),
            "sha256": converter_digest.hexdigest(),
            "path": (
                str(quantized_paths[0])
                if write_models and len(quantized_paths) == 1
                else None
            ),
            "batch_size": int(chosen_batch_size),
            "batch_count": len(converter_batches),
            "batches": converter_batches,
            "layout": converter_observed_layout,
            "mappings": mappings[:12],
        },
        "injected_variant": {
            "size": len(injected_bytes),
            "sha256": _sha256(injected_bytes),
            "path": str(injected_path) if write_models else None,
            "injection": injection,
            "graph": injected_graph,
        },
        "package_boundary": package_boundary,
        "converter_random_quantization_layout_match": bool(all_converter_layout_match),
        "converter_random_fc_quantization_layout_match": bool(fc_converter_layout_match),
        "converter_random_embedding_quantization_layout_match": bool(
            embedding_converter_layout_match
        ),
        "converter_to_injected_quantization_values_match": bool(
            converter_constants_sha256 == injected_constants_sha256
        ),
        "converter_constants_sha256": converter_constants_sha256,
        "injected_constants_sha256": injected_constants_sha256,
        "official_topology_graph_structure_match": bool(
            official_structural and official_structural == injected_structural
        ),
        "official_topology_quantization_layout_match": bool(
            official_layout and official_layout == injected_layout
        ),
        "official_execution_topology_match_ignoring_buffer_indices": bool(
            official_topology and official_topology == injected_topology
        ),
        "official_quantization_layout_match_ignoring_buffer_indices": bool(
            official_layout_no_buffers
            and official_layout_no_buffers == injected_layout_no_buffers
        ),
        "quantization_values_match": bool(
            official_graph["graph"].get("quantization_values_sha256")
            and official_graph["graph"].get("quantization_values_sha256")
            == injected_graph["graph"].get("quantization_values_sha256")
        ),
        "exact_official_model_match": False,
        "comparison": {
            "official_structural_sha256": official_structural,
            "injected_structural_sha256": injected_structural,
            "official_quantization_layout_sha256": official_layout,
            "injected_quantization_layout_sha256": injected_layout,
            "official_quantization_values_sha256": official_graph["graph"].get(
                "quantization_values_sha256"
            ),
            "injected_quantization_values_sha256": injected_graph["graph"].get(
                "quantization_values_sha256"
            ),
            "converter_constants_sha256": converter_constants_sha256,
            "injected_constants_sha256": injected_constants_sha256,
        },
        "interpretation": (
            "The public AI Edge Quantizer produced random constants independently; "
            "those constants were then injected into the official topology. The "
            "converter_to_injected_quantization_values_match check proves that the "
            "packed bytes and scales survive this mapping exactly. True "
            "structure/layout matches prove network compatibility, while differing "
            "official-versus-injected values are expected because initialization is random. "
            "The low-level builder covers both FULLY_CONNECTED and "
            "EMBEDDING_LOOKUP inventories; operator-specific fields expose any "
            "future unsupported scope explicitly."
        ),
    }
    if rebuild_graph:
        rebuilt_bytes, rebuild_report = _rebuild_graph_with_constants(
            official_bytes, official_records, converter_records
        )
        rebuilt_graph = _graph_report(rebuilt_bytes)
        rebuilt_constants = _official_constants_digest(
            rebuilt_bytes, official_records
        )
        rebuilt_structural = rebuilt_graph["graph"].get("structural_sha256")
        rebuilt_layout = rebuilt_graph["graph"].get(
            "quantization_layout_sha256"
        )
        rebuilt_topology = rebuilt_graph["graph_without_buffer_indices"].get(
            "structural_sha256"
        )
        rebuilt_layout_no_buffers = rebuilt_graph[
            "graph_without_buffer_indices"
        ].get("quantization_layout_sha256")
        rebuilt_buffer_storage = rebuilt_graph["graph"].get(
            "buffer_storage_sha256"
        )
        official_buffer_storage = official_graph["graph"].get(
            "buffer_storage_sha256"
        )
        result["rebuilt_variant"] = {
            "size": len(rebuilt_bytes),
            "sha256": _sha256(rebuilt_bytes),
            "path": None,
            "rebuild": rebuild_report,
            "graph": rebuilt_graph,
            "constants_sha256": rebuilt_constants,
            "buffer_storage_sha256": rebuilt_buffer_storage,
            "official_buffer_storage_sha256": official_buffer_storage,
            "constants_match_converter": bool(
                rebuilt_constants == converter_constants_sha256
            ),
        }
        result["rebuilt_quantized_network"] = {
            "same": bool(
                rebuilt_structural
                and rebuilt_structural == official_structural
                and rebuilt_layout
                and rebuilt_layout == official_layout
                and rebuilt_topology
                and rebuilt_topology == official_topology
                and rebuilt_layout_no_buffers
                and rebuilt_layout_no_buffers == official_layout_no_buffers
                and rebuilt_buffer_storage
                and rebuilt_buffer_storage == official_buffer_storage
                and rebuilt_constants == converter_constants_sha256
            ),
            "scope": "reconstructed_topology_and_public_converter_constants",
            "checks": {
                "official_structural_match": bool(
                    rebuilt_structural and rebuilt_structural == official_structural
                ),
                "official_quantization_layout_match": bool(
                    rebuilt_layout and rebuilt_layout == official_layout
                ),
                "execution_topology_match_ignoring_buffer_indices": bool(
                    rebuilt_topology and rebuilt_topology == official_topology
                ),
                "execution_layout_match_ignoring_buffer_indices": bool(
                    rebuilt_layout_no_buffers
                    and rebuilt_layout_no_buffers == official_layout_no_buffers
                ),
                "logical_buffer_storage_match": bool(
                    rebuilt_buffer_storage
                    and rebuilt_buffer_storage == official_buffer_storage
                ),
                "converter_constants_match": bool(
                    rebuilt_constants == converter_constants_sha256
                ),
            },
            "not_official_learned_model_identity": True,
        }
        if write_models:
            rebuilt_path = output_root / "random_converter_topology_rebuilt.tflite"
            rebuilt_path.write_bytes(rebuilt_bytes)
            result["rebuilt_variant"]["path"] = str(rebuilt_path)
        if runtime_allocate:
            result["rebuilt_runtime"] = _runtime_allocate_report(
                official_bytes,
                rebuilt_bytes,
                threads=runtime_threads,
                without_default_delegates=runtime_without_default_delegates,
            )
            result["rebuilt_quantized_network"]["checks"][
                "runtime_allocation_match"
            ] = bool(result["rebuilt_runtime"].get("allocation_match"))
            result["rebuilt_quantized_network"]["same"] = bool(
                result["rebuilt_quantized_network"]["same"]
                and result["rebuilt_runtime"].get("allocation_match")
            )
    result["random_quantized_network"] = _random_quantized_network_match(
        converter_inventory_match=all_converter_layout_match,
        converter_fc_match=fc_converter_layout_match,
        converter_embedding_match=embedding_converter_layout_match,
        official_structure_match=bool(
            official_structural and official_structural == injected_structural
        ),
        official_layout_match=bool(official_layout and official_layout == injected_layout),
        execution_topology_match=bool(
            official_topology and official_topology == injected_topology
        ),
        execution_layout_match=bool(
            official_layout_no_buffers
            and official_layout_no_buffers == injected_layout_no_buffers
        ),
    )
    if runtime_allocate:
        result["runtime"] = _runtime_allocate_report(
            official_bytes,
            injected_bytes,
            threads=runtime_threads,
            without_default_delegates=runtime_without_default_delegates,
        )
        result["random_quantized_network"]["runtime_allocation_match"] = bool(
            result["runtime"].get("allocation_match")
        )
    if package_output is not None:
        package_output_path = Path(package_output).expanduser().resolve()
        _write_injected_package(
            artifact_path, section, injected_bytes, package_output_path
        )
        result["package_boundary"]["output"] = str(package_output_path)
        result["package_boundary"]["output_size"] = package_output_path.stat().st_size
        result["package_boundary"]["output_sha256"] = _file_range_sha256(
            package_output_path, 0, package_output_path.stat().st_size
        )
    cleanup: dict[str, Any] = {"requested": not write_models, "removed": [], "locked": []}
    if not write_models:
        # The quantizer may retain a Windows memory map for one interpreter
        # cycle.  Collect first, then remove only the two temporary fixtures
        # created by this invocation; a locked path is reported, never fatal.
        gc.collect()
        for temporary_path in quantized_paths:
            try:
                temporary_path.unlink(missing_ok=True)
                cleanup["removed"].append(str(temporary_path))
            except PermissionError:
                cleanup["locked"].append(str(temporary_path))
    result["temporary_fixture_cleanup"] = cleanup
    report_path = output_root / "converter_random_topology_injection_report.json"
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantize random branches and inject them into official topology."
    )
    parser.add_argument("artifact", help="Official .litertlm artifact.")
    parser.add_argument("--model-type", default="tf_lite_mtp_drafter")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-weights", type=int)
    parser.add_argument("--calibration-samples", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--converter-batch-size",
        type=int,
        help=(
            "Number of official weights per independent converter graph. "
            "Large Gemma 4 E2B inventories default to eight to stay below "
            "FlatBuffers' 2 GiB offset limit."
        ),
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Do not retain the large converter/injected TFLite fixtures.",
    )
    parser.add_argument(
        "--runtime-allocate",
        action="store_true",
        help="Allocate official and injected sections and compare signatures.",
    )
    parser.add_argument("--runtime-threads", type=int, default=2)
    parser.add_argument(
        "--runtime-without-default-delegates",
        action="store_true",
        help="Use LiteRT's built-in resolver without default delegates for allocation.",
    )
    parser.add_argument(
        "--package-output",
        help="Optional full .litertlm fixture with only the selected section replaced.",
    )
    parser.add_argument(
        "--rebuild-flatbuffer",
        action="store_true",
        help=(
            "Rebuild a new FlatBuffer through the public ModelT object API "
            "after converter quantization, instead of only injecting bytes."
        ),
    )
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
            output_dir=args.output_dir,
            max_weights=args.max_weights,
            calibration_samples=args.calibration_samples,
            threads=args.threads,
            write_models=not args.in_memory,
            runtime_allocate=args.runtime_allocate,
            runtime_threads=args.runtime_threads,
            runtime_without_default_delegates=args.runtime_without_default_delegates,
            package_output=args.package_output,
            converter_batch_size=args.converter_batch_size,
            rebuild_graph=args.rebuild_flatbuffer,
        )
    except (OSError, ConverterRandomTopologyInjectionError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
