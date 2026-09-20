"""Fail-closed physical serialization audits for full-parameter Gemma 4 export.

This module does not convert a model.  It proves that values from an FP32
Safetensors checkpoint occur as *consumed physical constants* in the actual
TFLite files produced by an exporter.  Matching is by shape and exact bytes;
tensor names are recorded as evidence but are never trusted as proof.

Only identity serialization is supported for the floating graph.  Folding,
transposition, casts, aliases and computed constants are rejected because an
input file or an exporter-side Python object is not evidence that the value
was serialized into the final package.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPECTED_STATE_TENSOR_COUNT = 541
SECTION_ROLES = ("target", "token_embedder", "per_layer_embedder")


class FullParameterSerializationError(ValueError):
    """The physical export evidence is incomplete, ambiguous, or unsupported."""


def _sha(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _role_for_key(key: str) -> str:
    if key == "model.embed_tokens.weight":
        return "token_embedder"
    if key == "model.embed_tokens_per_layer.weight":
        return "per_layer_embedder"
    return "target"


def _read_safetensors(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise FullParameterSerializationError(f"Truncated Safetensors header: {path}")
        header_size = struct.unpack("<Q", size_bytes)[0]
        raw = handle.read(header_size)
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullParameterSerializationError(f"Invalid Safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise FullParameterSerializationError(f"Invalid Safetensors header object: {path}")
    return header, 8 + header_size


def build_checkpoint_inventory(
    model_dir: str | Path, *, expected_count: int = EXPECTED_STATE_TENSOR_COUNT
) -> dict[str, dict[str, Any]]:
    """Inventory exact FP32 checkpoint bytes across ``model*.safetensors``.

    Full-QAT parameters and persistent buffers are FP32. Accepting BF16 here
    would make an exact floating serialization audit incapable of proving that
    complete state was preserved.
    """

    root = Path(model_dir)
    config_path = root / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullParameterSerializationError(f"Cannot read checkpoint config: {config_path}") from exc
    if config.get("model_type") != "gemma4_text":
        raise FullParameterSerializationError("Full serialization requires model_type='gemma4_text'")
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        raise FullParameterSerializationError(f"No model Safetensors found in {root}")
    result: dict[str, dict[str, Any]] = {}
    for path in files:
        header, data_start = _read_safetensors(path)
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            for key, entry in header.items():
                if key == "__metadata__":
                    continue
                if key in result:
                    raise FullParameterSerializationError(f"Duplicate checkpoint tensor: {key}")
                if not isinstance(entry, dict) or entry.get("dtype") != "F32":
                    raise FullParameterSerializationError(f"Checkpoint tensor is not FP32: {key}")
                shape = entry.get("shape")
                offsets = entry.get("data_offsets")
                if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                    raise FullParameterSerializationError(f"Malformed checkpoint tensor: {key}")
                begin, end = (int(offsets[0]), int(offsets[1]))
                expected = 4
                for dimension in shape:
                    expected *= int(dimension)
                if begin < 0 or end < begin or end - begin != expected or data_start + end > file_size:
                    raise FullParameterSerializationError(f"Invalid checkpoint byte range: {key}")
                handle.seek(data_start + begin)
                payload = handle.read(end - begin)
                if len(payload) != end - begin:
                    raise FullParameterSerializationError(f"Truncated checkpoint tensor: {key}")
                result[key] = {
                    "key": key,
                    "shape": [int(v) for v in shape],
                    "dtype": "FLOAT32",
                    "nbytes": len(payload),
                    "value_sha256": _sha(payload),
                    "source_file": str(path.resolve()),
                    "source_offset": data_start + begin,
                    "section_role": _role_for_key(key),
                }
    if len(result) != expected_count:
        raise FullParameterSerializationError(
            f"Checkpoint contains {len(result)} state tensors; expected {expected_count}"
        )
    return dict(sorted(result.items()))


def verify_loaded_model(model: Any, inventory: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Verify exact key/shape/FP32-value parity with an already loaded model."""

    try:
        state = model.state_dict()
    except Exception as exc:
        raise FullParameterSerializationError("Loaded model has no readable state_dict") from exc
    if set(state) != set(inventory):
        raise FullParameterSerializationError("Loaded-model keys differ from checkpoint inventory")
    hashes: dict[str, str] = {}
    for key in sorted(inventory):
        tensor = state[key]
        if str(getattr(tensor, "dtype", "")) not in {"torch.float32", "float32"}:
            raise FullParameterSerializationError(f"Loaded state tensor is not FP32: {key}")
        if list(tensor.shape) != list(inventory[key]["shape"]):
            raise FullParameterSerializationError(f"Loaded state tensor shape differs: {key}")
        try:
            payload = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
        except Exception as exc:
            raise FullParameterSerializationError(f"Cannot materialize loaded state tensor: {key}") from exc
        digest = _sha(payload)
        if digest != inventory[key]["value_sha256"]:
            raise FullParameterSerializationError(f"Loaded state tensor value differs: {key}")
        hashes[key] = digest
    return {"verified": True, "state_tensor_count": len(hashes), "value_hashes": hashes}


def _schema_model(data: bytes) -> tuple[Any, Any]:
    try:
        import ai_edge_litert.schema_py_generated as schema
    except ImportError:
        try:
            import tflite as schema
        except ImportError as exc:
            raise FullParameterSerializationError(
                "Install ai-edge-litert or tflite to audit final TFLite constants"
            ) from exc
    return schema.Model.GetRootAsModel(data, 0), schema


def _type_names(schema: Any) -> dict[int, str]:
    enum = getattr(schema, "TensorType", schema)
    return {
        int(value): name
        for name, value in vars(enum).items()
        if name.isupper() and isinstance(value, int)
    }


def _buffer_bytes(buffer: Any, raw: Any = None) -> bytes:
    offset = int(buffer.Offset()) if hasattr(buffer, "Offset") else 0
    size = int(buffer.Size()) if hasattr(buffer, "Size") else 0
    if offset or size:
        if raw is None or offset <= 0 or size <= 0 or offset + size > len(raw) or buffer.DataLength():
            raise FullParameterSerializationError("Invalid external TFLite constant range")
        return bytes(raw[offset:offset + size])
    try:
        arr = buffer.DataAsNumpy()
        if arr is not None:
            return bytes(arr)
    except (AttributeError, TypeError):
        pass
    return bytes(int(buffer.Data(i)) & 0xFF for i in range(int(buffer.DataLength())))


def _name(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _physical_constants(path: Path, role: str) -> list[dict[str, Any]]:
    with path.open("rb") as backing, mmap.mmap(backing.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        return _constants_from_buffer(path, role, raw)


def _constants_from_buffer(path: Path, role: str, raw: Any) -> list[dict[str, Any]]:
    if len(raw) < 8 or bytes(raw[4:8]) != b"TFL3":
        raise FullParameterSerializationError(f"Not a TFLite FlatBuffer: {path}")
    model, schema = _schema_model(raw)
    names = _type_names(schema)
    records: list[dict[str, Any]] = []
    for sg_index in range(int(model.SubgraphsLength())):
        graph = model.Subgraphs(sg_index)
        consumed = set()
        consumer_scopes: dict[int, list[str]] = {}
        for op_index in range(int(graph.OperatorsLength())):
            op = graph.Operators(op_index)
            inputs = [int(op.Inputs(i)) for i in range(int(op.InputsLength())) if int(op.Inputs(i)) >= 0]
            consumed.update(inputs)
            scopes = [_name(graph.Tensors(int(op.Outputs(i))).Name()) for i in range(int(op.OutputsLength()))
                      if int(op.Outputs(i)) >= 0]
            for index in inputs:
                consumer_scopes.setdefault(index, []).extend(scopes)
        graph_inputs = {int(graph.Inputs(i)) for i in range(int(graph.InputsLength()))}
        for tensor_index in sorted(consumed - graph_inputs):
            tensor = graph.Tensors(tensor_index)
            buffer_index = int(tensor.Buffer())
            if buffer_index <= 0 or buffer_index >= int(model.BuffersLength()):
                continue
            payload = _buffer_bytes(model.Buffers(buffer_index), raw)
            if not payload:
                continue
            shape = [int(tensor.Shape(i)) for i in range(int(tensor.ShapeLength()))]
            quantization = tensor.Quantization()
            scales = (
                [float(quantization.Scale(i)) for i in range(int(quantization.ScaleLength()))]
                if quantization is not None else []
            )
            zero_points = (
                [int(quantization.ZeroPoint(i)) for i in range(int(quantization.ZeroPointLength()))]
                if quantization is not None else []
            )
            records.append({
                "section_role": role, "path": str(path), "subgraph": sg_index,
                "tensor_index": tensor_index, "tensor_name": _name(tensor.Name()),
                "consumer_scopes": consumer_scopes.get(tensor_index, []),
                "buffer_index": buffer_index, "shape": shape,
                "dtype": names.get(int(tensor.Type()), f"UNKNOWN:{int(tensor.Type())}"),
                "quantization": {"scales": scales, "zero_points": zero_points,
                                 "quantized_dimension": int(quantization.QuantizedDimension())
                                 if quantization is not None else None},
                "nbytes": len(payload), "payload_sha256": _sha(payload),
            })
    return records


def _load_record_payload(record: Mapping[str, Any]) -> bytes:
    """Reload one physical buffer; records intentionally never retain payloads."""
    path = Path(str(record["path"]))
    with path.open("rb") as backing, mmap.mmap(backing.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        model, _ = _schema_model(raw)
        payload = _buffer_bytes(model.Buffers(int(record["buffer_index"])), raw)
    if len(payload) != int(record["nbytes"]) or _sha(payload) != record["payload_sha256"]:
        raise FullParameterSerializationError("Final TFLite buffer changed during audit")
    return payload


def _scope_matches(key: str, source: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    """Identify the *use site*, then verify bytes separately. Never match by value alone.

    MLIR often calls a weight arith.constant. The consuming op output preserves
    the module scope; include that instead of guessing based on matrix size.
    Unknown/folded scopes fail closed and need an explicit reviewed mapping.
    """
    if record["section_role"] != source["section_role"]:
        return False
    scopes = [str(record["tensor_name"]), *record.get("consumer_scopes", [])]
    if source["section_role"] != "target":
        # Each external embedder contains exactly one trained parameter.
        return True
    module = key.removesuffix(".weight")
    layer = re.search(r"(?:^|\.)layers\.(\d+)\.", module)
    leaf = module.split(".")[-1]
    for scope in scopes:
        layers = set(re.findall(r"(?:Gemma4TextDecoderLayer_|\.layers\.)(\d+)(?:/|\.)", scope))
        if layer and layers != {layer[1]}:
            continue
        if not layer and layers:
            continue
        if re.search(r"(?:^|[./_])" + re.escape(leaf) + r"(?:[;/.]|$)", scope):
            return True
        # layer_scalar is a single multiplicative scalar in each decoder layer.
        # An opaque scalar MUL is not sufficient: its source name must survive.
    return False


def _scoped_records(key: str, source: Mapping[str, Any], physical: list[dict]) -> list[dict]:
    return [record for record in physical if _scope_matches(key, source, record)
            and record["shape"] == list(source["shape"])]


def audit_float_sections(
    section_paths: Mapping[str, str | Path],
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove exact, unique, role-correct FP32 constants in final TFLite files."""

    if set(section_paths) != set(SECTION_ROLES):
        raise FullParameterSerializationError(f"Exactly these sections are required: {SECTION_ROLES}")
    physical = [
        record
        for role in SECTION_ROLES
        for record in _physical_constants(Path(section_paths[role]), role)
    ]
    matches: dict[str, dict[str, Any]] = {}
    for key, source in inventory.items():
        candidates = _scoped_records(key, source, physical)
        if not candidates or any(r["dtype"] != "FLOAT32" or r["nbytes"] != source["nbytes"]
                                 or r["payload_sha256"] != source["value_sha256"] for r in candidates):
            raise FullParameterSerializationError(
                f"State tensor {key} is missing, changed, or has unsupported folded/ambiguous use-site scope"
            )
        # Prefill/decode signatures legitimately duplicate a constant. Prove
        # every duplicate is byte-identical rather than requiring one buffer.
        matches[key] = {"copies": candidates, "copy_count": len(candidates)}
    return {"verified": True, "state_tensor_count": len(matches), "mappings": matches,
            "unsupported_transforms": [], "folding_detected": False}


def _matrix_bits(key: str) -> int:
    from ir_training.qat.full_model_contract import MODULE_BITS
    module = key.removesuffix(".weight")
    matches = [bits for pattern, bits in MODULE_BITS.items() if re.search(pattern, module)]
    if len(matches) != 1:
        raise FullParameterSerializationError(f"No unique W248 allocation for matrix: {key}")
    return int(matches[0])


def _pack_codes(codes: Any, bits: int) -> bytes:
    import numpy as np
    flat = np.asarray(codes, dtype=np.int8).reshape(-1)
    if bits == 8:
        return flat.tobytes()
    per_byte = 8 // bits
    padding = (-len(flat)) % per_byte
    if padding:
        flat = np.pad(flat, (0, padding))
    unsigned = flat.astype(np.int16) & ((1 << bits) - 1)
    packed = np.zeros(len(flat) // per_byte, dtype=np.uint8)
    for lane in range(per_byte):
        packed |= unsigned[lane::per_byte].astype(np.uint8) << (lane * bits)
    return packed.tobytes()


def _pinned_aeq_expected(source_payload: bytes, shape: list[int], bits: int) -> tuple[bytes, Any]:
    """Recompute AEQ 0.9 channelwise symmetric min/max quantization."""
    import numpy as np
    try:
        from ai_edge_quantizer import qtyping
        from ai_edge_quantizer.algorithms.uniform_quantize import (
            uniform_quantize_tensor as uqt,
        )
    except ImportError as exc:
        raise FullParameterSerializationError(
            "Pinned ai-edge-quantizer 0.9 is required for deterministic W248 audit"
        ) from exc
    values = np.frombuffer(source_payload, dtype="<f4").reshape(shape)
    if not np.isfinite(values).all():
        raise FullParameterSerializationError("Non-finite checkpoint matrix cannot be exported")
    minimum = np.min(values, axis=1)
    maximum = np.max(values, axis=1)
    granularity = qtyping.QuantGranularity.CHANNELWISE
    zero, scale = uqt.tensor_zp_scale_from_min_max(
        minimum, maximum, bits, True, granularity
    )
    params = qtyping.UniformQuantParams(
        scale=np.asarray(scale), zero_point=np.asarray(zero), num_bits=bits,
        symmetric=True, quantized_dimension=0, block_size=0,
    )
    codes = uqt.uniform_quantize(values, params)
    return _pack_codes(codes, bits), np.asarray(scale, dtype=np.float32).reshape(-1)


def _default_matrix_audit(
    key: str, source: Mapping[str, Any], source_payload: bytes,
    records: list[dict[str, Any]],
) -> bool:
    import numpy as np
    bits = _matrix_bits(key)
    expected_codes, expected_scales = _pinned_aeq_expected(
        source_payload, list(source["shape"]), bits
    )
    if not records:
        return False
    for record in records:
        quant = record["quantization"]
        scales = np.asarray(quant["scales"], dtype=np.float32)
        zeros = quant["zero_points"]
        if (record["dtype"] != f"INT{bits}"
                or quant["quantized_dimension"] != 0 or scales.shape != expected_scales.shape
                or not np.array_equal(scales, expected_scales)
                or len(zeros) != len(expected_scales) or any(int(value) != 0 for value in zeros)
                or _load_record_payload(record) != expected_codes):
            return False
    return True


def audit_quantized_sections(
    float_paths: Mapping[str, str | Path],
    quantized_paths: Mapping[str, str | Path],
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit final quantized files without accepting an exporter-side claim.

    Non-matrix state tensors must remain exact FP32 constants. Matrix verification
    must be an executable auditor which receives source bytes and the actual
    final tensor record and returns true only after recomputing codes/scales.
    There is intentionally no hash-manifest-only escape hatch.
    """

    float_report = audit_float_sections(float_paths, inventory)
    if set(quantized_paths) != set(SECTION_ROLES):
        raise FullParameterSerializationError(f"Exactly these quantized sections are required: {SECTION_ROLES}")
    physical = [r for role in SECTION_ROLES for r in _physical_constants(Path(quantized_paths[role]), role)]
    results = {}
    for key, source in inventory.items():
        role_records = _scoped_records(key, source, physical)
        if len(source["shape"]) != 2:
            if not role_records or any(r["dtype"] != "FLOAT32" or r["payload_sha256"] != source["value_sha256"]
                                       for r in role_records):
                raise FullParameterSerializationError(f"Non-matrix state tensor changed or is ambiguous: {key}")
            results[key] = {"mode": "exact_float", "copy_count": len(role_records),
                            "tensor_names": [item["tensor_name"] for item in role_records]}
            continue
        if "source_file" not in source or "source_offset" not in source:
            raise FullParameterSerializationError(f"Missing physical checkpoint provenance: {key}")
        source_path = Path(str(source["source_file"]))
        with source_path.open("rb") as handle:
            handle.seek(int(source["source_offset"]))
            source_payload = handle.read(int(source["nbytes"]))
        if _sha(source_payload) != source["value_sha256"]:
            raise FullParameterSerializationError(f"Checkpoint bytes changed during audit: {key}")
        # The auditor gets source bytes and actual final records, never an
        # exporter-supplied code/scale hash claim.
        if not _default_matrix_audit(key, source, source_payload, role_records):
            raise FullParameterSerializationError(f"Deterministic quantization audit failed: {key}")
        results[key] = {"mode": "recomputed_quantized", "candidate_count": len(role_records)}
    return {"verified": True, "state_tensor_count": len(results), "float_audit": float_report,
            "state_tensors": results}
