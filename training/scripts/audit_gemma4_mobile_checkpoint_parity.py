"""Compare a public Gemma 4 mobile checkpoint with a LiteRT-LM artifact.

The public ``*-qat-mobile-transformers`` checkpoint stores low-bit values in
compressed ``uint8`` tensors.  The LiteRT-LM mobile section stores the same
codes in signed two's-complement low-bit buffers.  This audit is intentionally
read-only: it maps unique FC/embedding buffers, applies the serialization
conversion, and compares bytes plus per-axis scales.  It does not infer the
private QAT training loss, observer calibration, or exporter implementation.

The source checkpoint is accessed through ``safetensors.safe_open`` so the
full file is not copied into memory.  The official ``.litertlm`` is never
modified.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_converter_topology_parity import (  # noqa: E402
    _EMBEDDING_LOOKUP,
    _FULLY_CONNECTED,
    _LOW_BIT_TYPES,
    _opcode_builtin,
    _read_section,
    _section_by_model_type,
    _tensor_shape,
    _unpack_model,
)
from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    inspect_litertlm,
)


class Gemma4CheckpointParityError(RuntimeError):
    """Raised when the checkpoint/artifact parity audit cannot run."""


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _source_key(model_type: str, tensor_name: str, ordinal: int) -> tuple[str | None, int | None]:
    """Map an exported tensor name to a public checkpoint tensor.

    The returned second value selects a scale column for the packed
    per-layer embedding table; ordinary tensors use ``None``.
    """

    if model_type == "tf_lite_embedder":
        return "model.language_model.embed_tokens.embedding_quantized", None
    if model_type == "tf_lite_per_layer_embedder":
        return "model.language_model.embed_tokens_per_layer.embedding_quantized", ordinal

    # The prefill/decode exporter emits a long path with layer_N and an
    # operation-family marker.  Keep these mappings explicit and auditable.
    if "projected_per_layer_inputs" in tensor_name:
        return "model.language_model.per_layer_model_projection.weight", None
    if "decode_softmax" in tensor_name:
        return "lm_head.weight", None
    layer_match = re.search(r"layer_(\d+)", tensor_name)
    if not layer_match:
        return None, None
    layer = layer_match.group(1)
    candidates = (
        ("q_einsum", f"model.language_model.layers.{layer}.self_attn.q_proj.weight"),
        ("k_einsum", f"model.language_model.layers.{layer}.self_attn.k_proj.weight"),
        ("v_einsum", f"model.language_model.layers.{layer}.self_attn.v_proj.weight"),
        ("attn_vec_einsum", f"model.language_model.layers.{layer}.self_attn.o_proj.weight"),
        ("gating_einsum1", f"model.language_model.layers.{layer}.mlp.gate_proj.weight"),
        ("gating_einsum2", f"model.language_model.layers.{layer}.mlp.up_proj.weight"),
        ("/mlp/linear/", f"model.language_model.layers.{layer}.mlp.down_proj.weight"),
        (
            "per_layer_embedding_gate",
            f"model.language_model.layers.{layer}.per_layer_input_gate.weight",
        ),
        (
            "per_layer_embedding_projection",
            f"model.language_model.layers.{layer}.per_layer_projection.weight",
        ),
    )
    for marker, key in candidates:
        if marker in tensor_name:
            return key, None
    return None, None


def _buffer_bytes(model: Any, section_bytes: bytes, buffer_index: int) -> bytes:
    """Read an inline or externally stored object-API BufferT."""

    if buffer_index < 0 or buffer_index >= len(model.buffers or []):
        raise Gemma4CheckpointParityError(f"Invalid weight buffer index {buffer_index}.")
    buffer = model.buffers[buffer_index]
    if buffer.data is not None:
        return np.asarray(buffer.data, dtype=np.uint8).reshape(-1).tobytes()
    offset = int(getattr(buffer, "offset", 0) or 0)
    size = int(getattr(buffer, "size", 0) or 0)
    if offset > 0 and size > 0 and offset + size <= len(section_bytes):
        return bytes(section_bytes[offset : offset + size])
    raise Gemma4CheckpointParityError(
        f"Weight buffer {buffer_index} has no readable inline or external storage."
    )


def _signed_litert_bytes(source: bytes, bits: int) -> bytes:
    """Convert public mobile unsigned-offset codes to LiteRT signed codes.

    The public checkpoint uses a uniform unsigned code offset for low-bit
    weights: +2 for W2 and +8 for W4.  LiteRT keeps the same code order but
    serializes the signed two's-complement code.  W8 is already int8 and is
    copied byte-for-byte.
    """

    if bits == 8:
        return bytes(source)
    if bits not in (2, 4):
        raise Gemma4CheckpointParityError(f"Unsupported source packing width: W{bits}.")
    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    offset = 1 << (bits - 1)
    values = np.frombuffer(source, dtype=np.uint8)
    converted = np.zeros_like(values, dtype=np.uint8)
    for shift in range(values_per_byte):
        codes = ((values >> (shift * bits)) & mask).astype(np.int16)
        codes = (codes - offset) & mask
        converted |= codes.astype(np.uint8) << (shift * bits)
    return converted.tobytes()


def _records(model: Any, section_bytes: bytes) -> Iterable[dict[str, Any]]:
    """Yield unique quantized FC/embedding buffers from a TFLite section."""

    seen_buffers: set[int] = set()
    for subgraph_index, subgraph in enumerate(model.subgraphs or []):
        for operator_index, operator in enumerate(subgraph.operators or []):
            builtin = _opcode_builtin(model, operator)
            if builtin not in (_FULLY_CONNECTED, _EMBEDDING_LOOKUP):
                continue
            inputs = [int(value) for value in np.asarray(operator.inputs).reshape(-1)]
            if len(inputs) < 2 or inputs[1] < 0:
                continue
            weight = subgraph.tensors[inputs[1]]
            bits = _LOW_BIT_TYPES.get(int(weight.type))
            if bits is None:
                continue
            buffer_index = int(weight.buffer)
            if buffer_index in seen_buffers:
                continue
            seen_buffers.add(buffer_index)
            quantization = weight.quantization
            if quantization is None or quantization.scale is None:
                raise Gemma4CheckpointParityError(
                    f"Quantized tensor has no scales: subgraph={subgraph_index} "
                    f"tensor={inputs[1]}"
                )
            yield {
                "subgraph": subgraph_index,
                "operator": operator_index,
                "builtin": "FULLY_CONNECTED" if builtin == _FULLY_CONNECTED else "EMBEDDING_LOOKUP",
                "bits": bits,
                "shape": list(_tensor_shape(weight)),
                "buffer": buffer_index,
                "name": _decode(weight.name),
                "raw": _buffer_bytes(model, section_bytes, buffer_index),
                "scales": np.asarray(quantization.scale, dtype=np.float32).reshape(-1),
            }


def _scale_key(source_key: str) -> str:
    if source_key.endswith("_quantized"):
        return source_key[: -len("_quantized")] + "_scale"
    return source_key + "_scale"


def _audit_section(
    *,
    artifact: Path,
    package: dict[str, Any],
    model_type: str,
    source: Any,
    source_keys: set[str],
) -> dict[str, Any]:
    section = _section_by_model_type(package, model_type)
    section_bytes = _read_section(artifact, section)
    model, _, _ = _unpack_model(section_bytes)
    records = list(_records(model, section_bytes))
    counts: collections.Counter[str] = collections.Counter()
    bit_counts: collections.Counter[str] = collections.Counter()
    skipped: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    ordinal = 0

    for record in records:
        bit_counts[f"W{record['bits']}"] += 1
        source_key, scale_column = _source_key(model_type, record["name"], ordinal)
        if model_type == "tf_lite_per_layer_embedder":
            ordinal += 1
        if not source_key or source_key not in source_keys:
            skipped.append(
                {
                    "reason": "source_key_unavailable",
                    "name": record["name"],
                    "shape": record["shape"],
                    "bits": record["bits"],
                    "source_key": source_key,
                }
            )
            continue
        scale_key = _scale_key(source_key)
        if scale_key not in source_keys:
            skipped.append(
                {
                    "reason": "source_scale_unavailable",
                    "name": record["name"],
                    "shape": record["shape"],
                    "bits": record["bits"],
                    "source_key": source_key,
                    "scale_key": scale_key,
                }
            )
            continue

        try:
            source_tensor = np.asarray(source.get_tensor(source_key))
            source_scales = np.asarray(source.get_tensor(scale_key), dtype=np.float32)
        except Exception as exc:  # BF16 source tensors are intentionally unresolved.
            skipped.append(
                {
                    "reason": "source_tensor_not_readable",
                    "name": record["name"],
                    "shape": record["shape"],
                    "bits": record["bits"],
                    "source_key": source_key,
                    "error": str(exc),
                }
            )
            continue

        if scale_column is not None:
            # ``embedding_quantized`` stores all 35 per-layer tables in one
            # concatenated packed matrix.  Its second dimension is measured
            # in packed bytes, while the LiteRT tensor shape is expressed in
            # logical low-bit columns.  Select the packed-byte slice for the
            # current table before applying the W4 code conversion.
            if source_tensor.ndim != 2 or len(record["shape"]) != 2:
                skipped.append(
                    {
                        "reason": "source_tensor_shape_unavailable",
                        "name": record["name"],
                        "source_key": source_key,
                        "source_shape": list(source_tensor.shape),
                    }
                )
                continue
            packed_columns = (
                int(record["shape"][1]) * int(record["bits"]) + 7
            ) // 8
            start = scale_column * packed_columns
            end = start + packed_columns
            if end > source_tensor.shape[1]:
                skipped.append(
                    {
                        "reason": "source_tensor_column_unavailable",
                        "name": record["name"],
                        "source_key": source_key,
                        "source_shape": list(source_tensor.shape),
                        "scale_column": scale_column,
                        "packed_columns": packed_columns,
                    }
                )
                continue
            source_tensor = source_tensor[:, start:end]
            if source_scales.ndim != 2 or scale_column >= source_scales.shape[1]:
                skipped.append(
                    {
                        "reason": "source_scale_column_unavailable",
                        "name": record["name"],
                        "source_key": source_key,
                        "scale_column": scale_column,
                    }
                )
                continue
            source_scales = source_scales[:, scale_column]
        else:
            source_scales = source_scales.reshape(-1)

        expected_bytes = (int(np.prod(record["shape"], dtype=np.int64)) * record["bits"] + 7) // 8
        if source_tensor.nbytes != expected_bytes or len(record["raw"]) != expected_bytes:
            mismatches.append(
                {
                    "reason": "packed_size_mismatch",
                    "name": record["name"],
                    "source_key": source_key,
                    "source_shape": list(source_tensor.shape),
                    "official_shape": record["shape"],
                    "source_bytes": int(source_tensor.nbytes),
                    "official_bytes": len(record["raw"]),
                }
            )
            continue

        converted = _signed_litert_bytes(source_tensor.tobytes(), record["bits"])
        bytes_match = converted == record["raw"]
        scales_match = np.array_equal(
            source_scales.astype(np.float32, copy=False), record["scales"]
        )
        if bytes_match and scales_match:
            counts["exact"] += 1
            continue
        counts["bytes_exact" if bytes_match else "bytes_mismatch"] += 1
        counts["scales_exact" if scales_match else "scales_mismatch"] += 1
        if len(mismatches) < 12:
            mismatches.append(
                {
                    "reason": "value_or_scale_mismatch",
                    "name": record["name"],
                    "source_key": source_key,
                    "bits": record["bits"],
                    "bytes_match": bytes_match,
                    "scales_match": scales_match,
                    "max_scale_abs_error": (
                        float(np.max(np.abs(source_scales - record["scales"])))
                        if len(source_scales) == len(record["scales"])
                        else None
                    ),
                }
            )

    comparable = int(counts["exact"] + counts["bytes_exact"] + counts["bytes_mismatch"])
    return {
        "model_type": model_type,
        "section_size": len(section_bytes),
        "unique_quantized_buffers": len(records),
        "bits": dict(sorted(bit_counts.items())),
        "comparable_source_weights": comparable,
        "exact_bytes_and_scales": int(counts["exact"]),
        "bytes_only_matches": int(counts["bytes_exact"]),
        "bytes_mismatches": int(counts["bytes_mismatch"]),
        "scale_mismatches": int(counts["scales_mismatch"]),
        "unresolved_source_weights": len(skipped),
        "all_comparable_weights_exact": not counts["bytes_mismatch"] and not counts["scales_mismatch"],
        "skipped_preview": skipped[:12],
        "mismatch_preview": mismatches[:12],
    }


def run(
    artifact: str | Path,
    source_safetensors: str | Path,
    *,
    model_types: Iterable[str],
    output: str | Path | None = None,
) -> dict[str, Any]:
    artifact_path = Path(artifact).expanduser().resolve()
    source_path = Path(source_safetensors).expanduser().resolve()
    if not artifact_path.is_file():
        raise Gemma4CheckpointParityError(f"Official artifact does not exist: {artifact_path}")
    if not source_path.is_file():
        raise Gemma4CheckpointParityError(f"Source safetensors does not exist: {source_path}")
    try:
        package = inspect_litertlm(artifact_path, inspect_tflite=False)
        from safetensors import safe_open
    except (LiteRTLMInspectionError, ImportError) as exc:
        raise Gemma4CheckpointParityError(
            "The audit requires the local LiteRT inspector and safetensors package."
        ) from exc

    reports: list[dict[str, Any]] = []
    with safe_open(str(source_path), framework="numpy") as source:
        source_keys = set(source.keys())
        for model_type in model_types:
            reports.append(
                _audit_section(
                    artifact=artifact_path,
                    package=package,
                    model_type=model_type,
                    source=source,
                    source_keys=source_keys,
                )
            )

    result = {
        "artifact": str(artifact_path),
        "source_safetensors": str(source_path),
        "source_config": str(source_path.with_name("config.json"))
        if source_path.with_name("config.json").is_file()
        else None,
        "training_executed": False,
        "private_qat_recipe_recovered": False,
        "serialization_rule": {
            "W2": "subtract 2 from each unsigned 2-bit code modulo 4",
            "W4": "subtract 8 from each unsigned 4-bit code modulo 16",
            "W8": "copy int8 bytes directly",
        },
        "sections": reports,
        "all_comparable_weights_exact": all(
            report["all_comparable_weights_exact"] for report in reports
        ),
        "interpretation": (
            "Exact bytes and scales establish parity between this public mobile "
            "checkpoint and the supplied artifact's observable quantized constants. "
            "They do not reveal Google's private QAT loss, calibration corpus, "
            "optimizer schedule, or exporter source."
        ),
    }
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Gemma 4 mobile checkpoint serialization against LiteRT-LM."
    )
    parser.add_argument("artifact", help="Official Gemma 4 .litertlm artifact.")
    parser.add_argument("--source-safetensors", required=True)
    parser.add_argument(
        "--model-type",
        action="append",
        dest="model_types",
        help="TFLite section model_type; repeat to audit multiple sections.",
    )
    parser.add_argument(
        "--include-embedders",
        action="store_true",
        help="Also audit tf_lite_embedder and tf_lite_per_layer_embedder sections.",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 if any comparable weight bytes/scales differ.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    model_types = list(args.model_types or ["tf_lite_prefill_decode"])
    if args.include_embedders:
        model_types = ["tf_lite_embedder", "tf_lite_per_layer_embedder", *model_types]
    try:
        result = run(
            args.artifact,
            args.source_safetensors,
            model_types=model_types,
            output=args.output,
        )
    except (OSError, Gemma4CheckpointParityError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict and not result["all_comparable_weights_exact"]:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
