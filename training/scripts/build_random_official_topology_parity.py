"""Randomize an official LiteRT-LM TFLite section and compare graph parity.

This is the strongest local experiment possible without Google's unquantized
checkpoint and private mobile exporter.  It extracts one TFLite section from a
provided ``.litertlm`` artifact, replaces every observable per-axis quantized
weight buffer with deterministic random weights, and applies the public
AI Edge uniform symmetric quantizer (per output channel, axis 0).  The graph
FlatBuffer itself is never rebuilt: its topology, operator options, tensor
shapes, tensor types, and quantization layout remain the official ones.

The result proves artifact-level packing and graph-layout parity, not Google's
private QAT/calibration recipe.  It intentionally writes a standalone
``.tflite`` section instead of rewriting the user's large ``.litertlm`` file.
The official artifact is read-only; output paths are never overwritten.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    _tflite_graph_fingerprint,
    inspect_litertlm,
)


class OfficialTopologyParityError(RuntimeError):
    """Raised when the topology-preserving randomization cannot run."""


_LOW_BIT_TYPES = {
    # LiteRT TensorType enum values.  The older ``tflite`` wheel used by the
    # inspector may not define 19/20, so keep the numeric values explicit.
    9: 8,   # INT8
    17: 4,  # INT4
    19: 2,  # INT2
    20: 4,  # UINT4 is not normally present in the official mobile artifact
}


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _sha256(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _pack_low_bit(values: Any, bits: int) -> Any:
    """Pack signed int8 values using AI Edge's low-bit byte ordering."""

    import numpy as np

    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    original_length = len(flat)
    padding = (-original_length) % values_per_byte
    if padding:
        flat = np.pad(flat, (0, padding), mode="constant")
    unsigned = flat.astype(np.int16) & mask
    packed = np.zeros(len(flat) // values_per_byte, dtype=np.uint8)
    for offset in range(values_per_byte):
        packed |= (unsigned[offset::values_per_byte].astype(np.uint8) << (offset * bits))
    return packed[: (original_length + values_per_byte - 1) // values_per_byte]


def _unpack_low_bit(data: Any, bits: int, value_count: int) -> Any:
    """Unpack AI Edge's low-bit byte order into signed int8 values."""

    import numpy as np

    if bits not in (2, 4, 8):
        raise OfficialTopologyParityError(f"Unsupported unpack width: {bits}.")
    packed = np.frombuffer(data, dtype=np.uint8)
    if bits == 8:
        return packed.view(np.int8)[:value_count].copy()
    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    unsigned = np.empty(len(packed) * values_per_byte, dtype=np.uint8)
    for offset in range(values_per_byte):
        unsigned[offset::values_per_byte] = (packed >> (offset * bits)) & mask
    sign_bit = 1 << (bits - 1)
    signed = unsigned.astype(np.int16) - (unsigned >= sign_bit) * (1 << bits)
    return signed[:value_count].astype(np.int8, copy=False)


def _random_quantized_weight(shape: tuple[int, ...], bits: int, seed: int, ordinal: int):
    """Create a deterministic per-axis symmetric quantized weight."""

    import numpy as np

    if not shape or len(shape) < 2:
        raise OfficialTopologyParityError(
            f"Expected a rank-2-or-higher weight tensor, got shape {shape}."
        )
    if bits not in (2, 4, 8):
        raise OfficialTopologyParityError(f"Unsupported random quantization width: {bits}.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
    weights = rng.standard_normal(shape, dtype=np.float32)
    reduce_axes = tuple(range(1, len(shape)))
    bounds = np.max(np.abs(weights), axis=reduce_axes)
    bounds = np.maximum(bounds, np.float32(1e-9))
    qmax = (1 << (bits - 1)) - 1
    # AI Edge's symmetric low-bit path uses the full signed range for <8 bits;
    # the int8 path uses the usual narrow range [-127, 127].
    qmin = -qmax if bits >= 8 else -(1 << (bits - 1))
    scales = (bounds / np.float32(qmax)).astype(np.float32, copy=False)
    quantized = np.rint(weights / scales.reshape((-1,) + (1,) * len(reduce_axes)))
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8, copy=False)
    packed = _pack_low_bit(quantized, bits)
    return packed, scales


def _schema_model(data: bytearray):
    """Load a mutable model using the current LiteRT schema when available."""

    candidates = (
        ("ai_edge_litert.schema_py_generated", "Model"),
        ("tflite.Model", "Model"),
    )
    last_error: Exception | None = None
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)
            getter = getattr(model_cls, "GetRootAsModel", None) or getattr(model_cls, "GetRootAs")
            return getter(data, 0)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            last_error = exc
    raise OfficialTopologyParityError(
        "Random official-topology parity requires the generated TFLite schema "
        "from ai-edge-litert or tflite."
    ) from last_error


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
                (str(entry.get("value")) for entry in item.get("items", []) if entry.get("key") == "model_type"),
                "",
            )
            for item in package.get("sections", [])
            if item.get("data_type_name") == "TFLiteModel"
        ]
        raise OfficialTopologyParityError(
            f"No TFLite model_type={model_type!r} section. Available: {available}"
        )
    return matches[0]


def _model_type(section: dict[str, Any]) -> str:
    return next(
        (
            str(entry.get("value"))
            for entry in section.get("items", [])
            if entry.get("key") == "model_type"
        ),
        "",
    )


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    artifact = Path(args.artifact).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else None
    return {
        "artifact": str(artifact),
        "output": str(output) if output else None,
        "model_type": args.model_type,
        "seed": args.seed,
        "execute": bool(args.execute),
        "runtime_allocate": bool(getattr(args, "runtime_allocate", False)),
        "runtime_threads": int(getattr(args, "runtime_threads", 2)),
        "verify_weight_encoding": bool(getattr(args, "verify_weight_encoding", False)),
        "random_weights": True,
        "training_executed": False,
        "private_recipe_recovered": False,
        "exact_official_model_match": False,
        "method": (
            "Extract the official TFLite FlatBuffer section, preserve its graph, "
            "and replace observable per-axis quantized weight buffers with random "
            "AI Edge-style symmetric W2/W4/W8 values; read both inline data and "
            "external Buffer.offset/size storage."
        ),
    }


def _buffer_view(model: Any, buffer_index: int, section_bytes: bytearray) -> tuple[Any, str, int, int] | None:
    """Return a writable view for an inline or externally stored TFLite buffer.

    Recent LiteRT/TFLite FlatBuffers can externalize large constant buffers:
    ``Buffer.data`` is empty and ``Buffer.offset``/``Buffer.size`` point into
    the model payload.  The 270M candidate uses this representation, while
    some Gemma 4 buffers are inline.  Treating the two forms identically is
    necessary for a topology-preserving random-weight experiment.
    """

    buffer = model.Buffers(buffer_index)
    data_length = int(getattr(buffer, "DataLength", lambda: 0)() or 0)
    if data_length:
        raw = buffer.DataAsNumpy()
        if raw is not None and hasattr(raw, "__len__") and len(raw) == data_length:
            return raw, "inline", 0, data_length

    offset_method = getattr(buffer, "Offset", None)
    size_method = getattr(buffer, "Size", None)
    offset = int(offset_method() or 0) if offset_method is not None else 0
    size = int(size_method() or 0) if size_method is not None else 0
    if offset <= 0 or size <= 0 or offset + size > len(section_bytes):
        return None
    return memoryview(section_bytes)[offset : offset + size], "external", offset, size


def _randomize_section(
    section_bytes: bytearray, *, seed: int, verify_weight_encoding: bool = False
) -> dict[str, Any]:
    """Patch mutable section bytes and return a deterministic audit report."""

    import numpy as np

    model = _schema_model(section_bytes)
    patched_buffers: set[int] = set()
    buffer_records: dict[int, dict[str, Any]] = {}
    skipped = collections.Counter()
    storage_counts = collections.Counter()
    encoding_verified = 0
    encoding_mismatches = 0
    encoding_ranges: dict[int, list[int]] = {}
    tensor_count = 0
    ordinal = 0

    for subgraph_index in range(int(model.SubgraphsLength() or 0)):
        subgraph = model.Subgraphs(subgraph_index)
        for tensor_index in range(int(subgraph.TensorsLength() or 0)):
            tensor = subgraph.Tensors(tensor_index)
            type_value = int(tensor.Type())
            bits = _LOW_BIT_TYPES.get(type_value)
            quant = tensor.Quantization()
            shape = tuple(int(tensor.Shape(i)) for i in range(int(tensor.ShapeLength() or 0)))
            buffer_index = int(tensor.Buffer())
            if bits is None or quant is None or buffer_index <= 0:
                continue
            storage_view = _buffer_view(model, buffer_index, section_bytes)
            if storage_view is None:
                skipped["buffer_storage_unavailable"] += 1
                continue
            raw, storage, storage_offset, data_length = storage_view
            if len(shape) < 2:
                skipped["non_weight_or_empty"] += 1
                continue
            scale_count = int(quant.ScaleLength() or 0)
            if int(quant.QuantizedDimension()) != 0 or scale_count != shape[0]:
                skipped["unsupported_axis_or_scale_count"] += 1
                continue
            expected_bytes = (int(np.prod(shape, dtype=np.int64)) * bits + 7) // 8
            if expected_bytes != data_length:
                skipped["unexpected_buffer_size"] += 1
                continue
            if buffer_index in patched_buffers:
                continue

            if verify_weight_encoding:
                value_count = int(np.prod(shape, dtype=np.int64))
                decoded = _unpack_low_bit(raw, bits, value_count)
                if _pack_low_bit(decoded, bits).tobytes() == bytes(raw):
                    encoding_verified += 1
                else:
                    encoding_mismatches += 1
                current_range = encoding_ranges.setdefault(bits, [127, -128])
                if len(decoded):
                    current_range[0] = min(current_range[0], int(decoded.min()))
                    current_range[1] = max(current_range[1], int(decoded.max()))

            packed, scales = _random_quantized_weight(shape, bits, seed, ordinal)
            ordinal += 1
            if raw is None or len(raw) != len(packed):
                skipped["buffer_view_unavailable"] += 1
                continue
            raw[:] = packed
            storage_counts[storage] += 1
            scale_view = quant.ScaleAsNumpy()
            if scale_view is None or len(scale_view) != len(scales):
                skipped["scale_view_unavailable"] += 1
                continue
            scale_view[:] = scales
            zero_view = quant.ZeroPointAsNumpy()
            if zero_view is not None and len(zero_view):
                zero_view[:] = 0
            patched_buffers.add(buffer_index)
            tensor_count += 1
            buffer_records[buffer_index] = {
                "subgraph": subgraph_index,
                "tensor": tensor_index,
                "name": _decode(tensor.Name()),
                "type_value": type_value,
                "bits": bits,
                "shape": list(shape),
                "buffer_bytes": data_length,
                "scale_count": scale_count,
                "scale_min": float(np.min(scales)),
                "scale_max": float(np.max(scales)),
                "storage": storage,
                "storage_offset": storage_offset,
                "buffer_sha256_after": _sha256(raw),
            }

    # Keep only compact records in the JSON report; names remain available for
    # the first few records while counts cover the complete section.
    records = list(buffer_records.values())
    type_counts = collections.Counter(str(item["type_value"]) for item in records)
    return {
        "randomized_tensor_count": tensor_count,
        "randomized_buffer_count": len(patched_buffers),
        "randomized_type_values": dict(sorted(type_counts.items())),
        "randomized_storage": dict(sorted(storage_counts.items())),
        "weight_encoding": {
            "verification_requested": verify_weight_encoding,
            "roundtrip_verified_buffer_count": encoding_verified,
            "roundtrip_mismatch_count": encoding_mismatches,
            "decoded_value_ranges": {
                str(bits): {"min": values[0], "max": values[1]}
                for bits, values in sorted(encoding_ranges.items())
            },
        },
        "skipped": dict(sorted(skipped.items())),
        "records_preview": records[:12],
    }


def _runtime_allocate_report(
    official_section: bytes,
    randomized_section: bytes,
    *,
    threads: int,
    without_default_delegates: bool = False,
) -> dict[str, Any]:
    """Allocate both sections with LiteRT and compare exported signatures.

    This is intentionally an allocation-only smoke check.  It validates that
    the rewritten buffers are accepted by the runtime without pretending that
    random weights can produce numerical parity with the learned artifact.
    """

    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise OfficialTopologyParityError(
            "--runtime-allocate requires the optional ai-edge-litert runtime."
        ) from exc

    def allocate(content: bytes) -> dict[str, Any]:
        options: dict[str, Any] = {
            "model_content": content,
            "num_threads": threads,
        }
        if without_default_delegates:
            options["experimental_op_resolver_type"] = (
                OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
        interpreter = Interpreter(**options)
        interpreter.allocate_tensors()
        signatures = interpreter.get_signature_list()
        return {
            "signatures": signatures,
            "tensor_count": len(interpreter.get_tensor_details()),
        }

    official = allocate(official_section)
    randomized = allocate(randomized_section)
    return {
        "official": official,
        "randomized": randomized,
        "signature_match": official["signatures"] == randomized["signatures"],
        "tensor_count_match": official["tensor_count"] == randomized["tensor_count"],
        "allocation_match": (
            official["signatures"] == randomized["signatures"]
            and official["tensor_count"] == randomized["tensor_count"]
        ),
        "without_default_delegates": bool(without_default_delegates),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _plan(args)
    artifact = Path(plan["artifact"])
    output = Path(plan["output"]) if plan["output"] else None
    if not artifact.is_file():
        raise OfficialTopologyParityError(f"Official artifact does not exist: {artifact}")
    if output is not None and output.exists():
        raise OfficialTopologyParityError(f"Refusing to overwrite existing output: {output}")
    try:
        package = inspect_litertlm(artifact, inspect_tflite=False)
    except LiteRTLMInspectionError as exc:
        raise OfficialTopologyParityError(str(exc)) from exc
    section = _section_by_model_type(package, args.model_type)
    plan["section_index"] = section["index"]
    plan["section_size"] = section["size"]
    plan["section_model_type"] = _model_type(section)
    if not args.execute:
        return plan

    with artifact.open("rb") as handle:
        handle.seek(int(section["begin_offset"]))
        original = handle.read(int(section["size"]))
    if len(original) != int(section["size"]):
        raise OfficialTopologyParityError("Could not read the complete official TFLite section.")
    randomized = bytearray(original)
    random_report = _randomize_section(
        randomized, seed=args.seed, verify_weight_encoding=args.verify_weight_encoding
    )
    if not random_report["randomized_buffer_count"]:
        raise OfficialTopologyParityError(
            "No per-axis quantized weight buffers were found in the selected section."
        )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(randomized)

    try:
        official_graph = _tflite_graph_fingerprint(original)
        randomized_graph = _tflite_graph_fingerprint(randomized)
    except Exception as exc:  # pragma: no cover - optional generated binding
        official_graph = {"available": False, "error": str(exc)}
        randomized_graph = {"available": False, "error": str(exc)}
    graph_structure_match = bool(
        official_graph.get("available")
        and randomized_graph.get("available")
        and official_graph.get("structural_sha256") == randomized_graph.get("structural_sha256")
    )
    quant_layout_match = bool(
        official_graph.get("available")
        and randomized_graph.get("available")
        and official_graph.get("quantization_layout_sha256")
        == randomized_graph.get("quantization_layout_sha256")
    )
    result = {
        **plan,
        "official_section_sha256": _sha256(original),
        "randomized_section_sha256": _sha256(randomized),
        "randomization": random_report,
        "official_graph": official_graph,
        "randomized_graph": randomized_graph,
        "graph_structure_match": graph_structure_match,
        "quantization_layout_match": quant_layout_match,
        "quantization_values_match": bool(
            official_graph.get("quantization_values_sha256")
            and official_graph.get("quantization_values_sha256")
            == randomized_graph.get("quantization_values_sha256")
        ),
        "exact_official_model_match": False,
        "interpretation": (
            "A structure/layout match is expected because the official graph is "
            "preserved while only random weight bytes and scales are changed. "
            "This validates low-bit buffer sizing/packing and observable layout, "
            "not the private QAT schedule, calibration corpus, or learned weights."
        ),
    }
    if args.runtime_allocate:
        result["runtime"] = _runtime_allocate_report(
            bytes(original), bytes(randomized), threads=args.runtime_threads
        )
    if output is not None:
        report_path = output.with_suffix(output.suffix + ".json")
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Randomize official LiteRT-LM weights while preserving the TFLite graph."
    )
    parser.add_argument("artifact", help="Official or candidate .litertlm artifact.")
    parser.add_argument(
        "--model-type",
        default="tf_lite_prefill_decode",
        help="Header model_type to extract (default: tf_lite_prefill_decode).",
    )
    parser.add_argument("--output", help="New standalone .tflite output path.")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the in-memory parity check without writing a standalone TFLite file or report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true", help="Extract, randomize, and compare the section.")
    parser.add_argument(
        "--runtime-allocate",
        action="store_true",
        help="Allocate the official and randomized sections with ai-edge-litert and compare signatures.",
    )
    parser.add_argument("--runtime-threads", type=int, default=2)
    parser.add_argument(
        "--verify-weight-encoding",
        action="store_true",
        help="Unpack/repack each official low-bit weight buffer and report byte-roundtrip results.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.output and not args.no_write:
        parser.error("provide --output or opt into --no-write")
    try:
        result = run(args)
    except (OSError, OfficialTopologyParityError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
