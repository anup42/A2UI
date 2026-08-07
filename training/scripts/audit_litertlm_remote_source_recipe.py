"""Compare an official LiteRT-LM INT8 section with a remote safetensors source.

This is a deliberately read-only, range-based audit for gated/public mirror
checkpoints.  It reads the safetensors header and only the source tensors needed
by the selected LiteRT section; the multi-hundred-megabyte source file is never
written locally.  The optional generic Gemma 3 ordering handles converter
exports whose TFLite tensor names were canonicalized to ``arith.constant``.

An exact result proves the observable INT8 packing/scales for the compared
source bytes.  It does not prove that an unauthenticated mirror is Google's
official source, nor does it recover a private QAT training or calibration
schedule.
"""

from __future__ import annotations

import argparse
import collections
import json
import mmap
import re
import struct
import sys
from urllib.parse import unquote, urlparse
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_litertlm_weight_recipe import (  # noqa: E402
    WeightRecipeAuditError,
    _buffer_view,
    _decode,
    _expected_int8,
    _model_type,
    _operator_names,
    _section_by_model_type,
)
from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    inspect_litertlm,
)


_INT8_TYPE = 9
_SAFETENSORS_HEADER_PREFIX = 8
_GENERIC_GEMMA3_ORDER = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _range_request(url: str, start: int, end: int, *, timeout: float) -> tuple[bytes, int | None]:
    """Read one inclusive HTTP byte range and return bytes plus total length."""

    if start < 0 or end < start:
        raise WeightRecipeAuditError(f"Invalid source range [{start}, {end}].")
    request = Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "A2UI-LiteRT-LM-recipe-audit/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
            status = int(getattr(response, "status", 0) or 0)
            content_range = str(response.headers.get("Content-Range") or "")
    except Exception as exc:  # pragma: no cover - network/provider-specific
        raise WeightRecipeAuditError(
            f"Could not read source range [{start}, {end}] from {url}: {exc}"
        ) from exc
    expected = end - start + 1
    if status != 206 or len(payload) != expected:
        raise WeightRecipeAuditError(
            f"Source did not honor range [{start}, {end}]: status={status}, "
            f"received={len(payload)}, expected={expected}."
        )
    total = None
    match = re.search(r"/([0-9]+)$", content_range)
    if match:
        total = int(match.group(1))
    return payload, total


def read_safetensors_header(url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    """Read and parse a remote safetensors header using two small ranges."""

    prefix, total = _range_request(url, 0, _SAFETENSORS_HEADER_PREFIX - 1, timeout=timeout)
    if len(prefix) != _SAFETENSORS_HEADER_PREFIX:
        raise WeightRecipeAuditError("Remote safetensors prefix is incomplete.")
    header_size = struct.unpack_from("<Q", prefix, 0)[0]
    if header_size <= 0 or header_size > 256 * 1024 * 1024:
        raise WeightRecipeAuditError(f"Invalid remote safetensors header size: {header_size}.")
    header_bytes, total_from_header = _range_request(
        url,
        0,
        _SAFETENSORS_HEADER_PREFIX + header_size - 1,
        timeout=timeout,
    )
    try:
        header = json.loads(header_bytes[_SAFETENSORS_HEADER_PREFIX:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightRecipeAuditError("Remote safetensors header is not valid JSON.") from exc
    if not isinstance(header, dict):
        raise WeightRecipeAuditError("Remote safetensors header must be an object.")
    header["__a2ui_header_size__"] = int(header_size)
    header["__a2ui_file_size__"] = total_from_header or total
    return header


def _local_source_path(source: str | Path) -> Path | None:
    """Resolve a local safetensors path, including a ``file://`` URL."""

    value = str(source)
    # ``urlparse`` interprets ``C:\\...`` as a URL with scheme ``c``.  Check
    # Windows drive paths before parsing so the CLI works naturally in
    # PowerShell and cmd.exe.
    if re.match(r"^[A-Za-z]:[\\/]", value):
        candidate = Path(value).expanduser()
        return candidate if candidate.is_file() else None
    parsed = urlparse(value)
    if parsed.scheme == "file":
        # ``file:///C:/...`` is the form emitted by pathlib on Windows.  Do
        # not use ``Path.as_uri`` in reverse because it rejects a few valid
        # local spellings used in PowerShell command lines.
        path_value = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            path_value = f"//{parsed.netloc}{path_value}"
        if re.match(r"^/[A-Za-z]:/", path_value):
            path_value = path_value[1:]
        candidate = Path(path_value).expanduser()
        return candidate if candidate.is_file() else None
    if parsed.scheme:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_file() else None


def read_local_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read a local safetensors header without loading its tensor payload."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise WeightRecipeAuditError(f"Local safetensors source does not exist: {source_path}")
    with source_path.open("rb") as handle:
        prefix = handle.read(_SAFETENSORS_HEADER_PREFIX)
        if len(prefix) != _SAFETENSORS_HEADER_PREFIX:
            raise WeightRecipeAuditError("Local safetensors prefix is incomplete.")
        header_size = struct.unpack_from("<Q", prefix, 0)[0]
        if header_size <= 0 or header_size > 256 * 1024 * 1024:
            raise WeightRecipeAuditError(f"Invalid local safetensors header size: {header_size}.")
        header_bytes = handle.read(header_size)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightRecipeAuditError("Local safetensors header is not valid JSON.") from exc
    if not isinstance(header, dict):
        raise WeightRecipeAuditError("Local safetensors header must be an object.")
    header["__a2ui_header_size__"] = int(header_size)
    header["__a2ui_file_size__"] = int(source_path.stat().st_size)
    return header


def _source_array(url: str, entry: dict[str, Any], *, timeout: float) -> np.ndarray:
    dtype_name = str(entry.get("dtype"))
    shape = tuple(int(value) for value in entry.get("shape", []))
    offsets = entry.get("data_offsets")
    if len(offsets) != 2:
        raise WeightRecipeAuditError(f"Invalid data_offsets for remote tensor: {entry!r}")
    header_size = int(entry["__a2ui_header_size__"])
    start = _SAFETENSORS_HEADER_PREFIX + header_size + int(offsets[0])
    end = _SAFETENSORS_HEADER_PREFIX + header_size + int(offsets[1]) - 1
    raw, _ = _range_request(url, start, end, timeout=timeout)
    if dtype_name == "BF16":
        if len(raw) % 2:
            raise WeightRecipeAuditError("BF16 source tensor has an odd byte count.")
        # NumPy does not expose bfloat16 on every supported Python build.
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        result = bits.view(np.float32)
    else:
        dtype_map = {
            "F32": "<f4",
            "F16": "<f2",
            "I8": np.int8,
            "U8": np.uint8,
            "I32": "<i4",
            "U32": "<u4",
        }
        if dtype_name not in dtype_map:
            raise WeightRecipeAuditError(f"Unsupported remote safetensors dtype: {dtype_name}")
        result = np.frombuffer(raw, dtype=dtype_map[dtype_name])
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if result.size != expected:
        raise WeightRecipeAuditError(
            f"Remote tensor {entry!r} has {result.size} values; expected {expected}."
        )
    return result.reshape(shape).copy()


def _source_array_local(
    source: mmap.mmap, entry: dict[str, Any], *, header_size: int
) -> np.ndarray:
    """Decode one local safetensors tensor from a read-only memory map."""

    dtype_name = str(entry.get("dtype"))
    shape = tuple(int(value) for value in entry.get("shape", []))
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
        raise WeightRecipeAuditError(f"Invalid data_offsets for local tensor: {entry!r}")
    start = _SAFETENSORS_HEADER_PREFIX + int(header_size) + int(offsets[0])
    end = _SAFETENSORS_HEADER_PREFIX + int(header_size) + int(offsets[1])
    if start < 0 or end < start or end > len(source):
        raise WeightRecipeAuditError(f"Local tensor range is outside the source file: {entry!r}")
    raw = bytes(source[start:end])
    if dtype_name == "BF16":
        if len(raw) % 2:
            raise WeightRecipeAuditError("BF16 local tensor has an odd byte count.")
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        result = bits.view(np.float32)
    else:
        dtype_map = {
            "F32": "<f4",
            "F16": "<f2",
            "I8": np.int8,
            "U8": np.uint8,
            "I32": "<i4",
            "U32": "<u4",
        }
        if dtype_name not in dtype_map:
            raise WeightRecipeAuditError(f"Unsupported local safetensors dtype: {dtype_name}")
        result = np.frombuffer(raw, dtype=dtype_map[dtype_name])
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if result.size != expected:
        raise WeightRecipeAuditError(
            f"Local tensor {entry!r} has {result.size} values; expected {expected}."
        )
    return result.reshape(shape).copy()


def generic_gemma3_source_key(ordinal: int, shape: tuple[int, ...]) -> str:
    """Map the canonical 270M converter's seven-linear layer order."""

    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    layer, operation = divmod(ordinal, len(_GENERIC_GEMMA3_ORDER))
    return f"model.layers.{layer}.{_GENERIC_GEMMA3_ORDER[operation]}.weight"


def _entry_with_header(entry: dict[str, Any], header: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(entry)
    enriched["__a2ui_header_size__"] = header["__a2ui_header_size__"]
    return enriched


def _expected_int8_half_up(float_weight: Any) -> tuple[np.ndarray, np.ndarray]:
    """Use the TFLite/AI Edge tie rule observed in official packages.

    ``np.rint`` uses ties-to-even.  LiteRT's integer conversion path behaves
    like ``floor(x + 0.5)`` for signed values, so a negative exact half rounds
    toward zero.  Keep this alongside the NumPy reference from the local audit
    rather than silently changing that established helper.
    """

    weight = np.asarray(float_weight, dtype=np.float32)
    reduce_axes = tuple(range(1, weight.ndim))
    maxima = np.max(np.abs(weight), axis=reduce_axes)
    scales = (maxima / np.float32(127.0)).astype(np.float32, copy=False)
    safe_scales = np.where(scales == 0, np.float32(1.0), scales)
    ratio = weight / safe_scales.reshape((-1,) + (1,) * len(reduce_axes))
    quantized = np.floor(ratio + np.float32(0.5))
    quantized = np.clip(quantized, -127, 127).astype(np.int8, copy=False)
    return quantized, scales


def _expected_int8_ties_to_zero(float_weight: Any) -> tuple[np.ndarray, np.ndarray]:
    """Round-to-nearest with exact half cases directed toward zero."""

    weight = np.asarray(float_weight, dtype=np.float32)
    reduce_axes = tuple(range(1, weight.ndim))
    maxima = np.max(np.abs(weight), axis=reduce_axes)
    scales = (maxima / np.float32(127.0)).astype(np.float32, copy=False)
    safe_scales = np.where(scales == 0, np.float32(1.0), scales)
    ratio = weight / safe_scales.reshape((-1,) + (1,) * len(reduce_axes))
    nearest = np.rint(ratio)
    fractional = np.abs(ratio - np.trunc(ratio))
    half_case = fractional == np.float32(0.5)
    quantized = np.where(half_case, np.trunc(ratio), nearest)
    quantized = np.clip(quantized, -127, 127).astype(np.int8, copy=False)
    return quantized, scales


def _expected_int8_cast(float_weight: Any, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the same absmax/RNE rule after a lower-precision cast."""

    cast_weight = np.asarray(float_weight, dtype=np.float32).astype(dtype).astype(np.float32)
    return _expected_int8(cast_weight)


def audit_artifact(
    artifact: str | Path,
    source_url: str | Path,
    *,
    model_type: str = "TF_LITE_PREFILL_DECODE",
    timeout: float = 60.0,
    max_weights: int | None = None,
) -> dict[str, Any]:
    """Audit a LiteRT-LM INT8 section against remote BF16 safetensors."""

    artifact_path = Path(artifact).expanduser().resolve()
    if not artifact_path.is_file():
        raise WeightRecipeAuditError(f"LiteRT-LM artifact does not exist: {artifact_path}")
    try:
        package = inspect_litertlm(artifact_path, inspect_tflite=False)
        section = _section_by_model_type(package, model_type)
        from tflite.Model import Model
    except (ImportError, OSError, LiteRTLMInspectionError) as exc:
        raise WeightRecipeAuditError(
            "Remote recipe audit requires the generated 'tflite' bindings."
        ) from exc

    source_path = _local_source_path(source_url)
    if source_path is not None:
        source_header = read_local_safetensors_header(source_path)
        source_label = str(source_path)
    else:
        source_header = read_safetensors_header(str(source_url), timeout=timeout)
        source_label = str(source_url)
    source_keys = {key for key in source_header if not key.startswith("__")}
    op_names = _operator_names()
    records: list[dict[str, Any]] = []
    skipped = collections.Counter()
    seen_buffers: set[int] = set()
    generic_fc_ordinal = 0

    source_handle = source_path.open("rb") if source_path is not None else None
    source_map = (
        mmap.mmap(source_handle.fileno(), length=0, access=mmap.ACCESS_READ)
        if source_handle is not None
        else None
    )
    with artifact_path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            section_view = memoryview(mapped)[section["begin_offset"] : section["end_offset"]]
            model = Model.GetRootAsModel(section_view, 0)
            for subgraph_index in range(int(model.SubgraphsLength() or 0)):
                subgraph = model.Subgraphs(subgraph_index)
                for operator_index in range(int(subgraph.OperatorsLength() or 0)):
                    operator = subgraph.Operators(operator_index)
                    code = model.OperatorCodes(operator.OpcodeIndex())
                    operator_name = op_names.get(int(code.BuiltinCode()), "")
                    if operator_name not in {"FULLY_CONNECTED", "EMBEDDING_LOOKUP"}:
                        continue
                    if int(operator.InputsLength() or 0) < 2:
                        skipped["operator_without_weight_input"] += 1
                        continue
                    tensor = subgraph.Tensors(operator.Inputs(1))
                    buffer_index = int(tensor.Buffer())
                    if buffer_index in seen_buffers:
                        continue
                    seen_buffers.add(buffer_index)
                    shape = tuple(int(tensor.Shape(i)) for i in range(int(tensor.ShapeLength() or 0)))
                    if operator_name == "FULLY_CONNECTED":
                        generic_ordinal = generic_fc_ordinal
                        generic_fc_ordinal += 1
                    else:
                        generic_ordinal = None
                    if max_weights is not None and len(records) >= max_weights:
                        break
                    if int(tensor.Type()) != _INT8_TYPE:
                        skipped["non_int8_weight"] += 1
                        continue
                    quantization = tensor.Quantization()
                    if quantization is None:
                        skipped["missing_quantization"] += 1
                        continue
                    scale_count = int(quantization.ScaleLength() or 0)
                    if int(quantization.QuantizedDimension()) != 0 or scale_count != shape[0]:
                        skipped["unsupported_scale_layout"] += 1
                        continue
                    raw = np.frombuffer(_buffer_view(model, buffer_index, section_view), dtype=np.int8)
                    expected_count = int(np.prod(shape, dtype=np.int64))
                    if raw.size != expected_count:
                        skipped["unexpected_weight_size"] += 1
                        continue
                    raw = raw.reshape(shape).copy()
                    stored_scales = np.asarray(
                        [quantization.Scale(i) for i in range(scale_count)], dtype=np.float32
                    )
                    tensor_name = _decode(tensor.Name())
                    if operator_name == "EMBEDDING_LOOKUP":
                        key = "model.embed_tokens.weight"
                    else:
                        key = generic_gemma3_source_key(int(generic_ordinal), shape)
                    if key not in source_keys:
                        skipped["missing_source_tensor"] += 1
                        continue
                    source_entry = _entry_with_header(source_header[key], source_header)
                    if source_map is not None:
                        source_weight = _source_array_local(
                            source_map,
                            source_entry,
                            header_size=int(source_header["__a2ui_header_size__"]),
                        )
                    else:
                        source_weight = _source_array(
                            str(source_url), source_entry, timeout=timeout
                        )
                    transposed = False
                    if tuple(source_weight.shape) != shape:
                        if source_weight.ndim == 2 and tuple(source_weight.T.shape) == shape:
                            source_weight = source_weight.T
                            transposed = True
                        else:
                            skipped["source_shape_mismatch"] += 1
                            continue
                    expected_q, expected_scales = _expected_int8(source_weight)
                    half_up_q, half_up_scales = _expected_int8_half_up(source_weight)
                    ties_zero_q, ties_zero_scales = _expected_int8_ties_to_zero(source_weight)
                    fp16_q, fp16_scales = _expected_int8_cast(source_weight, np.float16)
                    scale_exact = bool(np.array_equal(stored_scales, expected_scales))
                    q_exact = bool(np.array_equal(raw, expected_q))
                    half_up_exact = bool(
                        np.array_equal(stored_scales, half_up_scales)
                        and np.array_equal(raw, half_up_q)
                    )
                    ties_zero_exact = bool(
                        np.array_equal(stored_scales, ties_zero_scales)
                        and np.array_equal(raw, ties_zero_q)
                    )
                    fp16_exact = bool(
                        np.array_equal(stored_scales, fp16_scales)
                        and np.array_equal(raw, fp16_q)
                    )
                    mismatch_positions = np.flatnonzero(raw.reshape(-1) != expected_q.reshape(-1))
                    records.append(
                        {
                            "operator": operator_name,
                            "subgraph": subgraph_index,
                            "operator_index": operator_index,
                            "tensor_name": tensor_name,
                            "source_key": key,
                            "shape": list(shape),
                            "buffer": buffer_index,
                            "transposed_source": transposed,
                            "scale_count": scale_count,
                            "scale_exact": scale_exact,
                            "scale_max_abs_error": float(np.max(np.abs(stored_scales - expected_scales))),
                            "quantized_values_exact": q_exact,
                            "half_up_quantized_values_exact": half_up_exact,
                            "half_up_mismatch_count": int(np.count_nonzero(raw != half_up_q)),
                            "ties_to_zero_quantized_values_exact": ties_zero_exact,
                            "ties_to_zero_mismatch_count": int(np.count_nonzero(raw != ties_zero_q)),
                            "fp16_quantized_values_exact": fp16_exact,
                            "fp16_mismatch_count": int(np.count_nonzero(raw != fp16_q)),
                            "quantized_mismatch_count": int(mismatch_positions.size),
                            "quantized_max_abs_error": int(
                                np.max(np.abs(raw.astype(np.int16) - expected_q.astype(np.int16)))
                            )
                            if mismatch_positions.size
                            else 0,
                            "quantized_first_mismatch": [
                                {
                                    "index": int(index),
                                    "stored": int(raw.reshape(-1)[index]),
                                    "expected": int(expected_q.reshape(-1)[index]),
                                    "source_value": float(source_weight.reshape(-1)[index]),
                                    "scale": float(
                                        stored_scales[
                                            np.unravel_index(index, raw.shape)[0]
                                        ]
                                    ),
                                    "ratio": float(
                                        source_weight.reshape(-1)[index]
                                        / stored_scales[np.unravel_index(index, raw.shape)[0]]
                                    ),
                                }
                                for index in mismatch_positions[:8]
                            ],
                        }
                    )
                if max_weights is not None and len(records) >= max_weights:
                    break
        finally:
            section_view.release()
            mapped.close()
            if source_map is not None:
                source_map.close()
            if source_handle is not None:
                source_handle.close()

    exact_scale_count = sum(1 for record in records if record["scale_exact"])
    exact_value_count = sum(1 for record in records if record["quantized_values_exact"])
    half_up_value_count = sum(
        1 for record in records if record["half_up_quantized_values_exact"]
    )
    ties_zero_value_count = sum(
        1 for record in records if record["ties_to_zero_quantized_values_exact"]
    )
    fp16_value_count = sum(1 for record in records if record["fp16_quantized_values_exact"])
    all_exact = bool(records) and exact_scale_count == len(records) and exact_value_count == len(records)
    total_value_count = sum(int(np.prod(record["shape"], dtype=np.int64)) for record in records)
    total_mismatch_count = sum(int(record["quantized_mismatch_count"]) for record in records)
    total_half_up_mismatch_count = sum(int(record["half_up_mismatch_count"]) for record in records)
    total_ties_to_zero_mismatch_count = sum(
        int(record["ties_to_zero_mismatch_count"]) for record in records
    )
    total_fp16_mismatch_count = sum(int(record["fp16_mismatch_count"]) for record in records)
    max_quantized_error = max(
        (int(record["quantized_max_abs_error"]) for record in records),
        default=0,
    )
    return {
        "artifact": str(artifact_path),
        "model_type": _model_type(section),
        "source_url": source_label,
        "source_file_size": source_header.get("__a2ui_file_size__"),
        "source_header_size": source_header.get("__a2ui_header_size__"),
        "source_kind": "local_safetensors" if source_path is not None else "remote_safetensors",
        "source_weight_comparison": {
            "matched_weight_count": len(records),
            "exact_scale_count": exact_scale_count,
            "exact_quantized_value_count": exact_value_count,
            "half_up_exact_quantized_value_count": half_up_value_count,
            "ties_to_zero_exact_quantized_value_count": ties_zero_value_count,
            "fp16_exact_quantized_value_count": fp16_value_count,
            "all_matched_weights_exact": all_exact,
            "total_quantized_value_count": total_value_count,
            "total_quantized_mismatch_count": total_mismatch_count,
            "total_half_up_mismatch_count": total_half_up_mismatch_count,
            "total_ties_to_zero_mismatch_count": total_ties_to_zero_mismatch_count,
            "total_fp16_mismatch_count": total_fp16_mismatch_count,
            "quantized_mismatch_fraction": (
                float(total_mismatch_count / total_value_count) if total_value_count else 0.0
            ),
            "quantized_max_abs_error": max_quantized_error,
            "records_preview": records[:12],
        },
        "skipped": dict(sorted(skipped.items())),
        "generic_gemma3_order": list(_GENERIC_GEMMA3_ORDER),
        "private_qat_recipe_recovered": False,
        "training_executed": False,
        "exact_official_model_match": False,
        "interpretation": (
            "Exact scales and tie-aware values establish the observable INT8 conversion for the "
            "compared source bytes. They do not establish mirror provenance or Google's "
            "private QAT/calibration schedule."
            if all_exact
            else "The source comparison was partial or non-exact; no exact public recipe "
            "claim is justified."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a LiteRT-LM INT8 section with remote BF16 safetensors."
    )
    parser.add_argument("artifact", help="Path to the .litertlm artifact.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-url", help="Direct HF resolve URL for model.safetensors."
    )
    source_group.add_argument(
        "--source-path", help="Local model.safetensors path (no source payload is copied)."
    )
    parser.add_argument("--model-type", default="TF_LITE_PREFILL_DECODE")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-weights", type=int, help="Optional cap for a quick sample.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless every matched weight is exact.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = audit_artifact(
            args.artifact,
            args.source_url or args.source_path,
            model_type=args.model_type,
            timeout=args.timeout,
            max_weights=args.max_weights,
        )
    except (OSError, WeightRecipeAuditError) as exc:
        parser.error(str(exc))
        return 2
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.strict and not report["source_weight_comparison"]["all_matched_weights_exact"]:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
