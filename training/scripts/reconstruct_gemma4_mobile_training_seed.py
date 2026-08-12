#!/usr/bin/env python3
"""Reconstruct a trainable Gemma 4 E2B text checkpoint from mobile weights.

Google's public ``*-qat-mobile-transformers`` checkpoint is the observable
numerical authority for the released mobile model, but its W2/W4/W8 matrices
are packed and cannot be consumed by the repository's ordinary LoRA/QAT
training path.  This script creates a text-only ``Gemma4ForCausalLM`` BF16
checkpoint by dequantizing those published codes with their published
per-output-channel scales.

The writer is streaming.  In particular, the 262144 x 8960 per-layer embedding
is never materialized in memory.  The 35 scale columns for that tensor are
applied to their corresponding 256 logical columns.  K/V projection tensors
that the packed checkpoint serializes for shared-KV layers are deliberately
omitted because the dense Transformers text architecture does not own them.

This recovers the centers of the public quantization cells, not Google's
pre-quantization master weights or private QAT recipe.  It does not run
training.  Without ``--execute`` it only validates and prints a plan.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import shutil
import struct
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

OFFICIAL_MOBILE_MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
OFFICIAL_MOBILE_REVISION = "dd693ff40353f057ca5f07e945ad867f4afbf2ec"
OFFICIAL_MOBILE_SAFETENSORS_SHA256 = (
    "efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4"
)
OFFICIAL_MOBILE_SAFETENSORS_SIZE = 2_458_111_846
OFFICIAL_MOBILE_CONFIG_SHA256 = (
    "cf6d7dc22738b5e6beb364bac833d78b869f5a6ffd57dfc96c6be3f2abc80424"
)
OFFICIAL_LITERTLM_SHA256 = (
    "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c"
)
OFFICIAL_RETAINED_COMPILED_REPORT_SHA256 = (
    "4fa47cf6fefb983a79bebc1e00bdd1f28df8d6570f59d7e979a1791a9a63429a"
)
EXPECTED_SOURCE_TENSOR_COUNT = 2_780
EXPECTED_OUTPUT_TENSOR_COUNT = 541
EXPECTED_DIRECT_BF16_COUNT = 263
EXPECTED_DEQUANTIZED_COUNT = 278
EXPECTED_RETAINED_COUNT = 262
EXPECTED_LAYER_COUNT = 35
EXPECTED_SHARED_KV_START = 15
MAX_HEADER_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_SHARD_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_WORKING_SET_BYTES = 16 * 1024 * 1024
MOBILE_QPARAMS_FILENAME = "mobile_qparams.safetensors"
MOBILE_QPARAMS_CONTRACT_FILENAME = "mobile_qparams.json"
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


class Gemma4MobileSeedError(RuntimeError):
    """Raised when a mobile checkpoint cannot be reconstructed safely."""


@dataclass(frozen=True)
class TensorTransform:
    output_key: str
    output_shape: tuple[int, ...]
    output_dtype: str
    source_key: str
    source_dtype: str
    source_shape: tuple[int, ...]
    source_begin: int
    source_end: int
    transform: str
    bits: int | None = None
    scale_key: str | None = None
    scale_shape: tuple[int, ...] | None = None
    scale_begin: int | None = None
    scale_end: int | None = None
    scale_group_width: int | None = None

    @property
    def output_nbytes(self) -> int:
        return _numel(self.output_shape) * 2

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("output_shape", "source_shape", "scale_shape"):
            value = payload.get(name)
            if value is not None:
                payload[name] = list(value)
        payload["output_nbytes"] = self.output_nbytes
        return payload


@dataclass(frozen=True)
class ShardPlan:
    filename: str
    tensors: tuple[TensorTransform, ...]

    @property
    def payload_nbytes(self) -> int:
        return sum(tensor.output_nbytes for tensor in self.tensors)


def _numel(shape: tuple[int, ...]) -> int:
    return int(math.prod(shape or (1,)))


def _sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Gemma4MobileSeedError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Gemma4MobileSeedError(f"{label} must contain a JSON object: {path}")
    return payload


def _read_safetensors_header(path: Path) -> tuple[int, int, dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise Gemma4MobileSeedError(
                    f"Safetensors header is truncated: {path}"
                )
            header_size = int.from_bytes(raw_size, "little", signed=False)
            if not 0 < header_size <= MAX_HEADER_BYTES:
                raise Gemma4MobileSeedError(
                    f"Unsafe Safetensors header size {header_size}: {path}"
                )
            raw_header = handle.read(header_size)
    except OSError as exc:
        raise Gemma4MobileSeedError(f"Could not read {path}: {exc}") from exc
    if len(raw_header) != header_size:
        raise Gemma4MobileSeedError(f"Safetensors header is truncated: {path}")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Gemma4MobileSeedError(
            f"Could not parse Safetensors header {path}: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise Gemma4MobileSeedError(f"Safetensors header is not an object: {path}")
    return header_size, 8 + header_size, header


def _entry(
    header: dict[str, Any],
    key: str,
    *,
    data_start: int,
    file_size: int,
) -> tuple[str, tuple[int, ...], int, int]:
    value = header.get(key)
    if not isinstance(value, dict):
        raise Gemma4MobileSeedError(f"Missing Safetensors entry: {key}")
    try:
        dtype = str(value["dtype"]).upper()
        shape = tuple(int(item) for item in value["shape"])
        relative_begin, relative_end = (
            int(item) for item in value["data_offsets"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Gemma4MobileSeedError(
            f"Invalid Safetensors entry for {key!r}: {value}"
        ) from exc
    begin = data_start + relative_begin
    end = data_start + relative_end
    if (
        any(dimension < 0 for dimension in shape)
        or relative_begin < 0
        or relative_end < relative_begin
        or begin < data_start
        or end > file_size
    ):
        raise Gemma4MobileSeedError(
            f"Unsafe Safetensors bounds for {key!r}: {value}"
        )
    dtype_bytes = {"BF16": 2, "F32": 4, "I8": 1, "U8": 1}.get(dtype)
    if dtype_bytes is None or end - begin != _numel(shape) * dtype_bytes:
        raise Gemma4MobileSeedError(
            f"Safetensors byte size is inconsistent for {key!r}: {value}"
        )
    return dtype, shape, begin, end


def _dense_text_config(source_config: dict[str, Any]) -> dict[str, Any]:
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict):
        raise Gemma4MobileSeedError("Source config has no Gemma 4 text_config.")
    config = json.loads(json.dumps(text_config))
    if str(config.get("model_type") or "") != "gemma4_text":
        raise Gemma4MobileSeedError(
            f"Unexpected text model_type: {config.get('model_type')!r}"
        )
    checks = {
        "num_hidden_layers": int(config.get("num_hidden_layers", 0) or 0)
        == EXPECTED_LAYER_COUNT,
        "hidden_size": int(config.get("hidden_size", 0) or 0) == 1536,
        "hidden_size_per_layer_input": int(
            config.get("hidden_size_per_layer_input", 0) or 0
        )
        == 256,
        "vocab_size": int(config.get("vocab_size", 0) or 0) == 262_144,
        "vocab_size_per_layer_input": int(
            config.get("vocab_size_per_layer_input", 0) or 0
        )
        == 262_144,
        "num_kv_shared_layers": int(config.get("num_kv_shared_layers", 0) or 0)
        == 20,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise Gemma4MobileSeedError(
            "Source text architecture differs from the audited E2B contract: "
            + ", ".join(failed)
        )
    config.pop("_name_or_path", None)
    config.pop("quantization_config", None)
    config["architectures"] = ["Gemma4ForCausalLM"]
    config["dtype"] = "bfloat16"
    config["tie_word_embeddings"] = False
    config["transformers_version"] = str(
        source_config.get("transformers_version") or "5.10.0.dev0"
    )
    return config


def _scale_key(source_key: str) -> str:
    if source_key.endswith(".embedding_quantized"):
        return source_key[: -len("_quantized")] + "_scale"
    return source_key + "_scale"


def _output_key(source_key: str) -> str | None:
    if source_key == "model.language_model.embed_tokens.embedding_quantized":
        return "model.embed_tokens.weight"
    if source_key == "model.language_model.embed_tokens_per_layer.embedding_quantized":
        return "model.embed_tokens_per_layer.weight"
    if source_key == "lm_head.weight":
        return source_key
    if not source_key.startswith("model.language_model."):
        return None
    match = LAYER_RE.match(source_key)
    if match and int(match.group(1)) >= EXPECTED_SHARED_KV_START and source_key.endswith(
        (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")
    ):
        return None
    return "model." + source_key[len("model.language_model.") :]


def _weight_bits(source_key: str, source_dtype: str) -> int:
    if source_dtype == "I8":
        return 8
    if source_dtype != "U8":
        raise Gemma4MobileSeedError(
            f"Packed tensor {source_key!r} has unsupported dtype {source_dtype}."
        )
    if source_key in {
        "lm_head.weight",
        "model.language_model.embed_tokens.embedding_quantized",
    }:
        return 2
    if source_key == (
        "model.language_model.embed_tokens_per_layer.embedding_quantized"
    ):
        return 4
    match = LAYER_RE.match(source_key)
    if not match:
        raise Gemma4MobileSeedError(
            f"Cannot infer mobile bit width for {source_key!r}."
        )
    layer = int(match.group(1))
    if ".mlp." in source_key:
        return 4 if layer < 15 else 2
    if ".self_attn." in source_key:
        return 4
    raise Gemma4MobileSeedError(
        f"Cannot infer mobile bit width for {source_key!r}."
    )


def _tensor_plan(
    header: dict[str, Any],
    *,
    data_start: int,
    file_size: int,
) -> list[TensorTransform]:
    raw_keys = sorted(key for key in header if key != "__metadata__")
    if len(raw_keys) != EXPECTED_SOURCE_TENSOR_COUNT:
        raise Gemma4MobileSeedError(
            "Unexpected source tensor count: "
            f"expected {EXPECTED_SOURCE_TENSOR_COUNT}, observed {len(raw_keys)}."
        )
    transforms: list[TensorTransform] = []
    for source_key in raw_keys:
        if not (
            source_key.startswith("model.language_model.")
            or source_key == "lm_head.weight"
        ):
            continue
        source_dtype, source_shape, source_begin, source_end = _entry(
            header,
            source_key,
            data_start=data_start,
            file_size=file_size,
        )
        target_key = _output_key(source_key)
        if target_key is None:
            continue
        if source_dtype == "BF16":
            transforms.append(
                TensorTransform(
                    output_key=target_key,
                    output_shape=source_shape,
                    output_dtype="BF16",
                    source_key=source_key,
                    source_dtype=source_dtype,
                    source_shape=source_shape,
                    source_begin=source_begin,
                    source_end=source_end,
                    transform="copy_bf16",
                )
            )
            continue
        if source_dtype not in {"U8", "I8"}:
            # Activation/cache scales and other quantized-runtime buffers are
            # not parameters of the dense text architecture.
            continue
        if not source_key.endswith((".weight", ".embedding_quantized")):
            continue
        if len(source_shape) != 2:
            raise Gemma4MobileSeedError(
                f"Packed text matrix is not rank two: {source_key} {source_shape}"
            )
        bits = _weight_bits(source_key, source_dtype)
        values_per_byte = 8 // bits
        output_shape = (
            source_shape
            if bits == 8
            else (source_shape[0], source_shape[1] * values_per_byte)
        )
        scale_key = _scale_key(source_key)
        scale_dtype, scale_shape, scale_begin, scale_end = _entry(
            header,
            scale_key,
            data_start=data_start,
            file_size=file_size,
        )
        if scale_dtype != "F32" or len(scale_shape) != 2:
            raise Gemma4MobileSeedError(
                f"Expected rank-two F32 scales for {source_key!r}; got "
                f"{scale_dtype} {scale_shape}."
            )
        if scale_shape[0] != output_shape[0]:
            raise Gemma4MobileSeedError(
                f"Scale row count differs for {source_key!r}: "
                f"{scale_shape} vs {output_shape}."
            )
        scale_group_width = output_shape[1] // scale_shape[1]
        if (
            scale_shape[1] < 1
            or output_shape[1] % scale_shape[1]
            or scale_group_width < 1
        ):
            raise Gemma4MobileSeedError(
                f"Scale columns cannot cover {source_key!r}: "
                f"{scale_shape} vs {output_shape}."
            )
        transforms.append(
            TensorTransform(
                output_key=target_key,
                output_shape=output_shape,
                output_dtype="BF16",
                source_key=source_key,
                source_dtype=source_dtype,
                source_shape=source_shape,
                source_begin=source_begin,
                source_end=source_end,
                transform=f"dequantize_w{bits}_to_bf16_rne",
                bits=bits,
                scale_key=scale_key,
                scale_shape=scale_shape,
                scale_begin=scale_begin,
                scale_end=scale_end,
                scale_group_width=scale_group_width,
            )
        )

    transforms.sort(key=lambda item: item.output_key)
    duplicate_keys = [
        key
        for key, count in collections.Counter(
            item.output_key for item in transforms
        ).items()
        if count > 1
    ]
    if duplicate_keys:
        raise Gemma4MobileSeedError(
            "Duplicate reconstructed keys: " + ", ".join(duplicate_keys[:12])
        )
    direct = sum(item.transform == "copy_bf16" for item in transforms)
    dequantized = len(transforms) - direct
    if (
        len(transforms) != EXPECTED_OUTPUT_TENSOR_COUNT
        or direct != EXPECTED_DIRECT_BF16_COUNT
        or dequantized != EXPECTED_DEQUANTIZED_COUNT
    ):
        raise Gemma4MobileSeedError(
            "Reconstructed tensor inventory differs from the audited text model: "
            f"total={len(transforms)}, direct={direct}, dequantized={dequantized}."
        )
    return transforms


def _assign_shards(
    transforms: list[TensorTransform], max_shard_bytes: int
) -> list[ShardPlan]:
    if max_shard_bytes < 1:
        raise Gemma4MobileSeedError("max_shard_bytes must be positive.")
    grouped: list[list[TensorTransform]] = []
    current: list[TensorTransform] = []
    current_size = 0
    for tensor in transforms:
        if current and current_size + tensor.output_nbytes > max_shard_bytes:
            grouped.append(current)
            current = []
            current_size = 0
        current.append(tensor)
        current_size += tensor.output_nbytes
    if current:
        grouped.append(current)
    count = len(grouped)
    return [
        ShardPlan(
            filename=(
                "model.safetensors"
                if count == 1
                else f"model-{index:05d}-of-{count:05d}.safetensors"
            ),
            tensors=tuple(tensors),
        )
        for index, tensors in enumerate(grouped, start=1)
    ]


def _float32_to_bfloat16_rne(values: np.ndarray) -> bytes:
    array = np.ascontiguousarray(values, dtype="<f4")
    if not bool(np.all(np.isfinite(array))):
        raise Gemma4MobileSeedError(
            "Dequantized mobile weights contain NaN or infinity."
        )
    words = array.view("<u4")
    rounded = words + np.uint32(0x7FFF) + (
        (words >> np.uint32(16)) & np.uint32(1)
    )
    return (rounded >> np.uint32(16)).astype("<u2").tobytes()


def _unpack_unsigned_codes(packed: np.ndarray, bits: int) -> np.ndarray:
    if packed.dtype != np.uint8 or packed.ndim != 2 or bits not in {2, 4}:
        raise Gemma4MobileSeedError(
            f"Invalid low-bit packed chunk: dtype={packed.dtype}, "
            f"shape={packed.shape}, bits={bits}."
        )
    values_per_byte = 8 // bits
    shifts = np.arange(values_per_byte, dtype=np.uint8) * np.uint8(bits)
    mask = np.uint8((1 << bits) - 1)
    codes = ((packed[:, :, None] >> shifts) & mask).reshape(
        packed.shape[0], packed.shape[1] * values_per_byte
    )
    return codes.astype(np.int16) - (1 << (bits - 1))


def _dequantize_chunk_to_bf16(
    raw_weight: bytes,
    raw_scale: bytes,
    *,
    rows: int,
    source_columns: int,
    logical_columns: int,
    scale_columns: int,
    bits: int,
) -> bytes:
    if bits == 8:
        codes = np.frombuffer(raw_weight, dtype=np.int8).reshape(
            rows, source_columns
        )
        if logical_columns != source_columns:
            raise Gemma4MobileSeedError("W8 logical/source columns differ.")
    else:
        packed = np.frombuffer(raw_weight, dtype=np.uint8).reshape(
            rows, source_columns
        )
        codes = _unpack_unsigned_codes(packed, bits)
        if codes.shape[1] != logical_columns:
            raise Gemma4MobileSeedError(
                f"W{bits} unpacked columns differ: {codes.shape[1]} vs "
                f"{logical_columns}."
            )
    scales = np.frombuffer(raw_scale, dtype="<f4").reshape(rows, scale_columns)
    if not bool(np.all(np.isfinite(scales))) or bool(np.any(scales <= 0.0)):
        raise Gemma4MobileSeedError("Mobile weight scales must be finite and positive.")
    if logical_columns % scale_columns:
        raise Gemma4MobileSeedError(
            "Logical columns are not divisible by scale columns."
        )
    group_width = logical_columns // scale_columns
    grouped = codes.astype(np.float32).reshape(rows, scale_columns, group_width)
    values = grouped * scales[:, :, None]
    return _float32_to_bfloat16_rne(values.reshape(rows, logical_columns))


def _read_exact(handle: BinaryIO, begin: int, size: int, *, label: str) -> bytes:
    handle.seek(begin)
    value = handle.read(size)
    if len(value) != size:
        raise Gemma4MobileSeedError(
            f"Short read for {label}: expected {size}, observed {len(value)}."
        )
    return value


def _write_tensor(
    source: BinaryIO,
    destination: BinaryIO,
    tensor: TensorTransform,
    *,
    working_set_bytes: int,
) -> str:
    digest = hashlib.sha256()
    if tensor.transform == "copy_bf16":
        remaining = tensor.source_end - tensor.source_begin
        source.seek(tensor.source_begin)
        while remaining:
            block = source.read(min(working_set_bytes, remaining))
            if not block:
                raise Gemma4MobileSeedError(
                    f"Short direct read for {tensor.source_key}."
                )
            destination.write(block)
            digest.update(block)
            remaining -= len(block)
        return digest.hexdigest()

    if (
        tensor.bits is None
        or tensor.scale_begin is None
        or tensor.scale_end is None
        or tensor.scale_shape is None
    ):
        raise Gemma4MobileSeedError(
            f"Incomplete dequantization plan for {tensor.source_key}."
        )
    rows, source_columns = tensor.source_shape
    logical_rows, logical_columns = tensor.output_shape
    if rows != logical_rows:
        raise Gemma4MobileSeedError(
            f"Source/output rows differ for {tensor.source_key}."
        )
    scale_columns = tensor.scale_shape[1]
    estimated_bytes_per_row = max(1, logical_columns * 8)
    rows_per_chunk = max(1, working_set_bytes // estimated_bytes_per_row)
    weight_row_bytes = source_columns
    scale_row_bytes = scale_columns * 4
    total_written = 0
    for row_begin in range(0, rows, rows_per_chunk):
        chunk_rows = min(rows_per_chunk, rows - row_begin)
        raw_weight = _read_exact(
            source,
            tensor.source_begin + row_begin * weight_row_bytes,
            chunk_rows * weight_row_bytes,
            label=tensor.source_key,
        )
        raw_scale = _read_exact(
            source,
            tensor.scale_begin + row_begin * scale_row_bytes,
            chunk_rows * scale_row_bytes,
            label=str(tensor.scale_key),
        )
        raw_output = _dequantize_chunk_to_bf16(
            raw_weight,
            raw_scale,
            rows=chunk_rows,
            source_columns=source_columns,
            logical_columns=logical_columns,
            scale_columns=scale_columns,
            bits=tensor.bits,
        )
        destination.write(raw_output)
        digest.update(raw_output)
        total_written += len(raw_output)
    if total_written != tensor.output_nbytes:
        raise Gemma4MobileSeedError(
            f"Output size differs for {tensor.output_key}: "
            f"{total_written} vs {tensor.output_nbytes}."
        )
    return digest.hexdigest()


def _safetensors_header(tensors: tuple[TensorTransform, ...]) -> bytes:
    payload: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for tensor in tensors:
        payload[tensor.output_key] = {
            "dtype": "BF16",
            "shape": list(tensor.output_shape),
            "data_offsets": [offset, offset + tensor.output_nbytes],
        }
        offset += tensor.output_nbytes
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    raw += b" " * ((-len(raw)) % 8)
    return raw


def _write_shard(
    source_path: Path,
    destination_path: Path,
    shard: ShardPlan,
    *,
    working_set_bytes: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    header = _safetensors_header(shard.tensors)
    partial_path = destination_path.with_name(destination_path.name + ".partial")
    if destination_path.exists() or partial_path.exists():
        raise Gemma4MobileSeedError(
            f"Refusing to overwrite an existing shard or partial: {destination_path}"
        )
    tensor_hashes: dict[str, str] = {}
    file_digest = hashlib.sha256()
    prefix = len(header).to_bytes(8, "little", signed=False) + header
    # A failed write intentionally leaves the non-final ``.partial`` file for
    # diagnosis; it is never promoted to the requested shard name.
    with source_path.open("rb") as source, partial_path.open("xb") as output:
        output.write(prefix)
        file_digest.update(prefix)

        class _HashingWriter:
            def write(self, value: bytes) -> int:
                written = output.write(value)
                file_digest.update(value[:written])
                return written

        writer = _HashingWriter()
        for tensor in shard.tensors:
            tensor_hashes[tensor.output_key] = _write_tensor(
                source,
                writer,  # type: ignore[arg-type]
                tensor,
                working_set_bytes=working_set_bytes,
            )
        output.flush()
    expected_size = 8 + len(header) + shard.payload_nbytes
    observed_size = partial_path.stat().st_size
    if observed_size != expected_size:
        raise Gemma4MobileSeedError(
            f"Shard size differs for {partial_path}: {observed_size} vs {expected_size}."
        )
    partial_path.replace(destination_path)
    return tensor_hashes, {
        "path": destination_path.name,
        "size_bytes": observed_size,
        "payload_bytes": shard.payload_nbytes,
        "tensor_count": len(shard.tensors),
        "sha256": file_digest.hexdigest(),
    }


def _activation_scale_hex_by_weight(
    source_path: Path,
    header: dict[str, Any],
    *,
    data_start: int,
    file_size: int,
    tensors: Iterable[TensorTransform],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with source_path.open("rb") as source:
        for tensor in tensors:
            if tensor.scale_key is None or not tensor.source_key.endswith(".weight"):
                continue
            stem = tensor.source_key[: -len(".weight")]
            scales: dict[str, str] = {}
            for role in ("input", "output"):
                source_key = f"{stem}.{role}_activation_scale"
                if source_key not in header:
                    continue
                dtype, shape, begin, end = _entry(
                    header,
                    source_key,
                    data_start=data_start,
                    file_size=file_size,
                )
                if dtype != "F32" or shape not in {(), (1,)} or end - begin != 4:
                    raise Gemma4MobileSeedError(
                        f"Expected scalar F32 {role} activation scale for "
                        f"{tensor.source_key!r}; got {dtype} {shape}."
                    )
                source.seek(begin)
                raw = source.read(4)
                value = struct.unpack("<f", raw)[0]
                if not math.isfinite(value) or value <= 0:
                    # lm_head uses floating graph edges and publishes zero
                    # placeholders; it is frozen and not an effective-LoRA target.
                    if tensor.output_key != "lm_head.weight" or value != 0:
                        raise Gemma4MobileSeedError(
                            f"Invalid {role} activation scale for "
                            f"{tensor.source_key!r}: {value}."
                        )
                scales[f"{role}_activation_scale_f32_le_hex"] = raw.hex()
            if scales:
                result[tensor.output_key] = scales
    return result


def _qparams_inventory(
    tensors: Iterable[TensorTransform],
    activation_scales: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Describe the released quantization parameters retained for each matrix.

    The dense BF16 seed contains dequantized cell centers.  Those centers are
    not enough to reconstruct the released quantizer: W2/W4 tensors frequently
    do not occupy both extrema, so deriving a new abs-max scale changes their
    codes.  This inventory therefore treats Google's published scale tensor as
    immutable training/export state.
    """

    inventory: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        if tensor.scale_key is None:
            continue
        if tensor.bits not in {2, 4, 8} or tensor.scale_shape is None:
            raise Gemma4MobileSeedError(
                f"Incomplete retained qparams for {tensor.output_key!r}."
            )
        if tensor.scale_begin is None or tensor.scale_end is None:
            raise Gemma4MobileSeedError(
                f"Missing retained scale offsets for {tensor.output_key!r}."
            )
        # A single scale column is ordinary per-output-channel quantization.
        # Multiple columns are blockwise and the recorded width must be used.
        group_size = (
            int(tensor.scale_group_width)
            if int(tensor.scale_shape[1]) > 1
            else None
        )
        inventory[tensor.output_key] = {
            "scale_tensor": tensor.output_key,
            "source_scale_key": tensor.scale_key,
            "bits": int(tensor.bits),
            "weight_shape": list(tensor.output_shape),
            "scale_shape": list(tensor.scale_shape),
            "axis": 0,
            "group_size": group_size,
            "zero_point": 0,
            "symmetric": True,
            "signed_range": "narrow" if tensor.bits == 8 else "full",
            **dict((activation_scales or {}).get(tensor.output_key, {})),
        }
    return dict(sorted(inventory.items()))


def _qparams_inventory_sha256(inventory: dict[str, dict[str, Any]]) -> str:
    canonical = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _qparams_safetensors_header(tensors: tuple[TensorTransform, ...]) -> bytes:
    payload: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "contract": "gemma4_mobile_retained_qparams_v1",
        }
    }
    offset = 0
    for tensor in tensors:
        if tensor.scale_shape is None:
            raise Gemma4MobileSeedError(
                f"Missing scale shape for {tensor.output_key!r}."
            )
        nbytes = _numel(tensor.scale_shape) * 4
        payload[tensor.output_key] = {
            "dtype": "F32",
            "shape": list(tensor.scale_shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    raw += b" " * ((-len(raw)) % 8)
    return raw


def _write_qparams_sidecar(
    source_path: Path,
    destination_path: Path,
    tensors: tuple[TensorTransform, ...],
) -> dict[str, Any]:
    """Stream exact F32 scale bytes into a portable Safetensors sidecar."""

    header = _qparams_safetensors_header(tensors)
    partial_path = destination_path.with_name(destination_path.name + ".partial")
    if destination_path.exists() or partial_path.exists():
        raise Gemma4MobileSeedError(
            f"Refusing to overwrite retained-qparams output: {destination_path}"
        )
    digest = hashlib.sha256()
    payload_nbytes = 0
    prefix = len(header).to_bytes(8, "little", signed=False) + header
    with source_path.open("rb") as source, partial_path.open("xb") as output:
        output.write(prefix)
        digest.update(prefix)
        for tensor in tensors:
            if tensor.scale_begin is None or tensor.scale_end is None:
                raise Gemma4MobileSeedError(
                    f"Missing scale byte range for {tensor.output_key!r}."
                )
            expected = _numel(tensor.scale_shape or ()) * 4
            if tensor.scale_end - tensor.scale_begin != expected:
                raise Gemma4MobileSeedError(
                    f"Scale byte count differs for {tensor.output_key!r}: "
                    f"{tensor.scale_end - tensor.scale_begin} vs {expected}."
                )
            source.seek(tensor.scale_begin)
            remaining = expected
            while remaining:
                block = source.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise Gemma4MobileSeedError(
                        f"Scale tensor is truncated for {tensor.output_key!r}."
                    )
                output.write(block)
                digest.update(block)
                remaining -= len(block)
                payload_nbytes += len(block)
        output.flush()
    expected_size = len(prefix) + payload_nbytes
    observed_size = partial_path.stat().st_size
    if observed_size != expected_size:
        raise Gemma4MobileSeedError(
            f"Retained-qparams file size differs: {observed_size} vs {expected_size}."
        )
    partial_path.replace(destination_path)
    return {
        "path": destination_path.name,
        "size_bytes": observed_size,
        "payload_bytes": payload_nbytes,
        "tensor_count": len(tensors),
        "sha256": digest.hexdigest(),
    }


def _validate_retained_report(
    path: Path,
    *,
    expected_artifact_sha256: str,
    expected_report_sha256: str,
) -> dict[str, Any]:
    report = _read_json(path, label="retained compiled-parity report")
    observed_report_sha256 = _sha256_file(path)
    artifact = report.get("artifact") if isinstance(report.get("artifact"), dict) else {}
    source = report.get("source") if isinstance(report.get("source"), dict) else {}
    comparison = (
        report.get("comparison")
        if isinstance(report.get("comparison"), dict)
        else {}
    )
    checks = {
        "report_ok": report.get("ok") is True,
        "report_sha256_match": observed_report_sha256 == expected_report_sha256,
        "training_not_executed": report.get("training_executed") is False,
        "artifact_identity_match": artifact.get("identity_match") is True,
        "artifact_sha256_match": str(artifact.get("sha256_observed") or "").lower()
        == expected_artifact_sha256,
        "retained_tensor_count": int(source.get("retained_tensor_count", 0) or 0)
        == EXPECTED_RETAINED_COUNT,
        "semantic_mapping_count": int(
            comparison.get("semantic_mapping_count", 0) or 0
        )
        == EXPECTED_RETAINED_COUNT,
        "exact_bf16_rne_count": int(
            comparison.get("exact_bf16_rne_tensor_count", 0) or 0
        )
        == EXPECTED_RETAINED_COUNT,
        "compiled_mapping_verified": report.get("compiled_graph_mapping_verified")
        is True,
    }
    return {
        "path": str(path),
        "sha256": observed_report_sha256,
        "checks": checks,
        "verified": all(checks.values()),
        "artifact_sha256": artifact.get("sha256_observed"),
        "retained_tensor_count": source.get("retained_tensor_count"),
        "exact_bf16_rne_tensor_count": comparison.get(
            "exact_bf16_rne_tensor_count"
        ),
    }


def _plan_digest(transforms: list[TensorTransform]) -> str:
    canonical = json.dumps(
        [item.public_dict() for item in transforms],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_plan(
    source_safetensors: str | Path,
    source_config: str | Path,
    retained_compiled_report: str | Path,
    output_dir: str | Path,
    *,
    source_model_id: str = OFFICIAL_MOBILE_MODEL_ID,
    source_revision: str = OFFICIAL_MOBILE_REVISION,
    expected_source_sha256: str = OFFICIAL_MOBILE_SAFETENSORS_SHA256,
    expected_source_config_sha256: str = OFFICIAL_MOBILE_CONFIG_SHA256,
    official_artifact_sha256: str = OFFICIAL_LITERTLM_SHA256,
    retained_compiled_report_sha256: str = OFFICIAL_RETAINED_COMPILED_REPORT_SHA256,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, Any]:
    source_path = Path(source_safetensors).expanduser().resolve()
    config_path = Path(source_config).expanduser().resolve()
    retained_path = Path(retained_compiled_report).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    if not source_path.is_file():
        raise Gemma4MobileSeedError(
            f"Packed mobile Safetensors does not exist: {source_path}"
        )
    if not config_path.is_file():
        raise Gemma4MobileSeedError(f"Source config does not exist: {config_path}")
    if not retained_path.is_file():
        raise Gemma4MobileSeedError(
            f"Retained compiled-parity report does not exist: {retained_path}"
        )
    expected_source_hash = str(expected_source_sha256).strip().lower()
    expected_config_hash = str(expected_source_config_sha256).strip().lower()
    expected_artifact_hash = str(official_artifact_sha256).strip().lower()
    expected_retained_report_hash = str(
        retained_compiled_report_sha256
    ).strip().lower()
    for label, value in (
        ("expected_source_sha256", expected_source_hash),
        ("expected_source_config_sha256", expected_config_hash),
        ("official_artifact_sha256", expected_artifact_hash),
        ("retained_compiled_report_sha256", expected_retained_report_hash),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise Gemma4MobileSeedError(
                f"{label} must be 64 lowercase hexadecimal characters."
            )
    observed_source_hash = _sha256_file(source_path)
    if observed_source_hash != expected_source_hash:
        issues.append(
            {
                "code": "source_safetensors_sha256_mismatch",
                "expected": expected_source_hash,
                "observed": observed_source_hash,
            }
        )
    if source_path.stat().st_size != OFFICIAL_MOBILE_SAFETENSORS_SIZE:
        issues.append(
            {
                "code": "source_safetensors_size_mismatch",
                "expected": OFFICIAL_MOBILE_SAFETENSORS_SIZE,
                "observed": source_path.stat().st_size,
            }
        )
    header_size, data_start, header = _read_safetensors_header(source_path)
    observed_config_hash = _sha256_file(config_path)
    if observed_config_hash != expected_config_hash:
        issues.append(
            {
                "code": "source_config_sha256_mismatch",
                "expected": expected_config_hash,
                "observed": observed_config_hash,
            }
        )
    config = _read_json(config_path, label="source config")
    dense_config = _dense_text_config(config)
    transforms = _tensor_plan(
        header, data_start=data_start, file_size=source_path.stat().st_size
    )
    retained = _validate_retained_report(
        retained_path,
        expected_artifact_sha256=expected_artifact_hash,
        expected_report_sha256=expected_retained_report_hash,
    )
    if not retained["verified"]:
        issues.append(
            {
                "code": "retained_compiled_parity_unverified",
                "checks": retained["checks"],
            }
        )
    if output_path in {source_path, config_path, retained_path}:
        issues.append(
            {
                "code": "output_path_collides_with_input",
                "path": str(output_path),
            }
        )
    if output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        issues.append(
            {
                "code": "output_directory_not_empty",
                "path": str(output_path),
            }
        )
    shards = _assign_shards(transforms, max_shard_bytes)
    bit_histogram = collections.Counter(
        f"W{item.bits}" for item in transforms if item.bits is not None
    )
    transform_histogram = collections.Counter(item.transform for item in transforms)
    activation_scales = _activation_scale_hex_by_weight(
        source_path,
        header,
        data_start=data_start,
        file_size=source_path.stat().st_size,
        tensors=transforms,
    )
    qparams_inventory = _qparams_inventory(transforms, activation_scales)
    if len(qparams_inventory) != EXPECTED_DEQUANTIZED_COUNT:
        raise Gemma4MobileSeedError(
            "Retained-qparams inventory differs from the dequantized matrix "
            f"inventory: {len(qparams_inventory)} vs {EXPECTED_DEQUANTIZED_COUNT}."
        )
    weight_map = {
        tensor.output_key: shard.filename
        for shard in shards
        for tensor in shard.tensors
    }
    return {
        "manifest_version": 1,
        "seed_format": "gemma4_e2b_mobile_dequantized_bf16_text",
        "seed_id": (
            f"{source_model_id}@{source_revision}:"
            "dequantized-bf16-text-v1"
        ),
        "source": {
            "model_id": source_model_id,
            "revision": source_revision,
            "safetensors": str(source_path),
            "safetensors_size_bytes": source_path.stat().st_size,
            "safetensors_sha256_expected": expected_source_hash,
            "safetensors_sha256_observed": observed_source_hash,
            "header_size_bytes": header_size,
            "tensor_count": len(
                [key for key in header if key != "__metadata__"]
            ),
            "config": str(config_path),
            "config_sha256_expected": expected_config_hash,
            "config_sha256": observed_config_hash,
        },
        "official_graph_authority": {
            "artifact_sha256": expected_artifact_hash,
            "retained_compiled_parity": retained,
        },
        "output": {
            "directory": str(output_path),
            "model_type": dense_config["model_type"],
            "architecture": dense_config["architectures"][0],
            "dtype": "BF16",
            "tie_word_embeddings": False,
            "tensor_count": len(transforms),
            "direct_bf16_tensor_count": sum(
                item.transform == "copy_bf16" for item in transforms
            ),
            "dequantized_tensor_count": sum(
                item.transform != "copy_bf16" for item in transforms
            ),
            "payload_size_bytes": sum(item.output_nbytes for item in transforms),
            "shard_count": len(shards),
            "shards": [
                {
                    "path": shard.filename,
                    "tensor_count": len(shard.tensors),
                    "payload_size_bytes": shard.payload_nbytes,
                }
                for shard in shards
            ],
            "weight_map": weight_map,
            "config": dense_config,
            "mobile_qparams": {
                "contract_path": MOBILE_QPARAMS_CONTRACT_FILENAME,
                "safetensors_path": MOBILE_QPARAMS_FILENAME,
                "tensor_count": len(qparams_inventory),
                "inventory_sha256": _qparams_inventory_sha256(
                    qparams_inventory
                ),
                "inventory": qparams_inventory,
            },
        },
        "transformation": {
            "plan_sha256": _plan_digest(transforms),
            "tensor_count": len(transforms),
            "bit_histogram": dict(sorted(bit_histogram.items())),
            "transform_histogram": dict(sorted(transform_histogram.items())),
            "shared_kv_projection_start_layer": EXPECTED_SHARED_KV_START,
            "per_layer_embedding_scale_columns": EXPECTED_LAYER_COUNT,
            "per_layer_embedding_scale_group_width": 256,
            "tensor_mappings": [item.public_dict() for item in transforms],
        },
        "checks": {
            "source_identity_match": observed_source_hash == expected_source_hash,
            "source_size_match": source_path.stat().st_size
            == OFFICIAL_MOBILE_SAFETENSORS_SIZE,
            "source_config_identity_match": observed_config_hash
            == expected_config_hash,
            "source_tensor_count_match": len(
                [key for key in header if key != "__metadata__"]
            )
            == EXPECTED_SOURCE_TENSOR_COUNT,
            "output_tensor_count_match": len(transforms)
            == EXPECTED_OUTPUT_TENSOR_COUNT,
            "retained_qparams_inventory_complete": len(qparams_inventory)
            == EXPECTED_DEQUANTIZED_COUNT,
            "retained_compiled_parity_verified": retained["verified"],
            "output_destination_safe": not any(
                item["code"]
                in {"output_path_collides_with_input", "output_directory_not_empty"}
                for item in issues
            ),
        },
        "ready": not issues,
        "issues": issues,
        "training_executed": False,
        "private_google_recipe_recovered": False,
        "precision_boundary": (
            "Packed values are dequantized to their published per-channel cell "
            "centers and rounded to BF16. This is a mobile-compatible public "
            "training initialization, not Google's unreleased master checkpoint."
        ),
    }


def execute_plan(
    plan: dict[str, Any],
    *,
    working_set_bytes: int = DEFAULT_WORKING_SET_BYTES,
) -> dict[str, Any]:
    if not plan.get("ready"):
        raise Gemma4MobileSeedError(
            "Reconstruction plan is not executable: "
            + ", ".join(str(item.get("code")) for item in plan.get("issues", []))
        )
    if working_set_bytes < 1024:
        raise Gemma4MobileSeedError("working_set_bytes must be at least 1024.")
    source_path = Path(str(plan["source"]["safetensors"]))
    output_path = Path(str(plan["output"]["directory"]))
    if output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        raise Gemma4MobileSeedError(
            f"Refusing to write into non-empty output directory: {output_path}"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    mappings = [
        TensorTransform(
            output_key=str(item["output_key"]),
            output_shape=tuple(int(value) for value in item["output_shape"]),
            output_dtype=str(item["output_dtype"]),
            source_key=str(item["source_key"]),
            source_dtype=str(item["source_dtype"]),
            source_shape=tuple(int(value) for value in item["source_shape"]),
            source_begin=int(item["source_begin"]),
            source_end=int(item["source_end"]),
            transform=str(item["transform"]),
            bits=int(item["bits"]) if item.get("bits") is not None else None,
            scale_key=str(item["scale_key"]) if item.get("scale_key") else None,
            scale_shape=(
                tuple(int(value) for value in item["scale_shape"])
                if item.get("scale_shape") is not None
                else None
            ),
            scale_begin=(
                int(item["scale_begin"])
                if item.get("scale_begin") is not None
                else None
            ),
            scale_end=(
                int(item["scale_end"])
                if item.get("scale_end") is not None
                else None
            ),
            scale_group_width=(
                int(item["scale_group_width"])
                if item.get("scale_group_width") is not None
                else None
            ),
        )
        for item in plan["transformation"]["tensor_mappings"]
    ]
    shard_by_name: dict[str, list[TensorTransform]] = collections.defaultdict(list)
    for tensor in mappings:
        shard_by_name[str(plan["output"]["weight_map"][tensor.output_key])].append(
            tensor
        )
    shard_records: list[dict[str, Any]] = []
    tensor_hashes: dict[str, str] = {}
    for shard_spec in plan["output"]["shards"]:
        filename = str(shard_spec["path"])
        shard = ShardPlan(filename, tuple(shard_by_name[filename]))
        hashes, record = _write_shard(
            source_path,
            output_path / filename,
            shard,
            working_set_bytes=working_set_bytes,
        )
        tensor_hashes.update(hashes)
        shard_records.append(record)

    qparam_mappings = tuple(
        tensor for tensor in mappings if tensor.scale_key is not None
    )
    expected_qparams = plan["output"].get("mobile_qparams")
    expected_qparams = (
        expected_qparams if isinstance(expected_qparams, dict) else {}
    )
    materialized_weight_inventory = _qparams_inventory(qparam_mappings)
    qparams_inventory = expected_qparams.get("inventory")
    qparams_inventory = (
        qparams_inventory if isinstance(qparams_inventory, dict) else {}
    )
    inventory_sha256 = _qparams_inventory_sha256(qparams_inventory)
    comparable_expected = {
        key: {
            field: value
            for field, value in entry.items()
            if not field.endswith("_activation_scale_f32_le_hex")
        }
        for key, entry in qparams_inventory.items()
        if isinstance(entry, dict)
    }
    if (
        len(qparam_mappings) != EXPECTED_DEQUANTIZED_COUNT
        or inventory_sha256 != expected_qparams.get("inventory_sha256")
        or materialized_weight_inventory != comparable_expected
    ):
        raise Gemma4MobileSeedError(
            "Retained-qparams execution inventory differs from the audited plan."
        )
    qparams_record = _write_qparams_sidecar(
        source_path,
        output_path / MOBILE_QPARAMS_FILENAME,
        qparam_mappings,
    )
    qparams_contract = {
        "contract_version": 1,
        "contract_type": "gemma4_mobile_retained_qparams",
        "source_model_id": plan["source"]["model_id"],
        "source_revision": plan["source"]["revision"],
        "source_safetensors_sha256": plan["source"][
            "safetensors_sha256_observed"
        ],
        "seed_id": plan["seed_id"],
        "scale_storage": qparams_record,
        "tensor_count": len(qparams_inventory),
        "inventory_sha256": inventory_sha256,
        "inventory": qparams_inventory,
        "zero_points_are_all_zero": True,
        "training_executed": False,
        "private_google_recipe_recovered": False,
    }
    qparams_contract_path = output_path / MOBILE_QPARAMS_CONTRACT_FILENAME
    qparams_contract_path.write_text(
        json.dumps(qparams_contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    qparams_contract_record = {
        "path": qparams_contract_path.name,
        "size_bytes": qparams_contract_path.stat().st_size,
        "sha256": _sha256_file(qparams_contract_path),
    }

    config_path = output_path / "config.json"
    config_path.write_text(
        json.dumps(plan["output"]["config"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    auxiliary_records: list[dict[str, Any]] = [
        {
            "path": config_path.name,
            "size_bytes": config_path.stat().st_size,
            "sha256": _sha256_file(config_path),
        }
    ]
    auxiliary_records.extend([qparams_record, qparams_contract_record])
    source_dir = Path(str(plan["source"]["config"])).parent
    for name in (
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        source_asset = source_dir / name
        if not source_asset.is_file():
            continue
        destination = output_path / name
        if destination.exists():
            raise Gemma4MobileSeedError(
                f"Refusing to overwrite auxiliary output: {destination}"
            )
        shutil.copy2(source_asset, destination)
        auxiliary_records.append(
            {
                "path": destination.name,
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        )
    if len(shard_records) > 1:
        index_path = output_path / "model.safetensors.index.json"
        index_payload = {
            "metadata": {"total_size": int(plan["output"]["payload_size_bytes"])},
            "weight_map": plan["output"]["weight_map"],
        }
        index_path.write_text(
            json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        auxiliary_records.append(
            {
                "path": index_path.name,
                "size_bytes": index_path.stat().st_size,
                "sha256": _sha256_file(index_path),
            }
        )

    manifest = json.loads(json.dumps(plan))
    manifest["output"]["shards"] = shard_records
    manifest["output"]["tensor_sha256"] = tensor_hashes
    manifest["output"]["auxiliary_files"] = auxiliary_records
    manifest["output"]["mobile_qparams"] = {
        **expected_qparams,
        "scale_storage": qparams_record,
        "contract": qparams_contract_record,
        "materialized": True,
    }
    manifest["output"]["materialized"] = True
    manifest["executed"] = True
    manifest["training_executed"] = False
    manifest_path = output_path / "mobile_training_seed_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    compact = json.loads(json.dumps(plan))
    compact["output"].pop("weight_map", None)
    compact["output"].pop("config", None)
    compact["transformation"].pop("tensor_mappings", None)
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or stream a BF16 Gemma 4 text training seed from Google's "
            "packed mobile checkpoint."
        )
    )
    parser.add_argument("--source-safetensors", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--retained-compiled-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-model-id", default=OFFICIAL_MOBILE_MODEL_ID)
    parser.add_argument("--source-revision", default=OFFICIAL_MOBILE_REVISION)
    parser.add_argument(
        "--expected-source-sha256",
        default=OFFICIAL_MOBILE_SAFETENSORS_SHA256,
    )
    parser.add_argument(
        "--expected-source-config-sha256",
        default=OFFICIAL_MOBILE_CONFIG_SHA256,
    )
    parser.add_argument(
        "--official-artifact-sha256", default=OFFICIAL_LITERTLM_SHA256
    )
    parser.add_argument(
        "--retained-compiled-report-sha256",
        default=OFFICIAL_RETAINED_COMPILED_REPORT_SHA256,
    )
    parser.add_argument(
        "--max-shard-size-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES
    )
    parser.add_argument(
        "--working-set-bytes", type=int, default=DEFAULT_WORKING_SET_BYTES
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="Optional full JSON plan path; stdout stays compact.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write the dense checkpoint. Without this flag no output is written.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        plan = build_plan(
            args.source_safetensors,
            args.source_config,
            args.retained_compiled_report,
            args.output_dir,
            source_model_id=args.source_model_id,
            source_revision=args.source_revision,
            expected_source_sha256=args.expected_source_sha256,
            expected_source_config_sha256=args.expected_source_config_sha256,
            official_artifact_sha256=args.official_artifact_sha256,
            retained_compiled_report_sha256=args.retained_compiled_report_sha256,
            max_shard_bytes=args.max_shard_size_bytes,
        )
        if args.plan_output is not None:
            plan_path = args.plan_output.expanduser().resolve()
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if args.execute:
            result = execute_plan(plan, working_set_bytes=args.working_set_bytes)
            print(json.dumps(_compact_plan(result), indent=2, ensure_ascii=False))
            return 0
        print(json.dumps(_compact_plan(plan), indent=2, ensure_ascii=False))
        print(
            "Plan only: no training, model loading, dequantized checkpoint, or "
            "LiteRT-LM package was written."
        )
        return 0 if plan["ready"] else 2
    except (OSError, ValueError, Gemma4MobileSeedError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
