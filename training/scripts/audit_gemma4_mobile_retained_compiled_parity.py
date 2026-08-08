#!/usr/bin/env python3
"""Map public Gemma 4 mobile retained constants into an official LiteRT graph.

The packed ``*-qat-mobile-transformers`` checkpoint stores non-quantized
language-model norms and layer scalars as BF16.  The released LiteRT-LM target
stores the corresponding learned constants as FLOAT32 buffers.  This audit
uses semantic consumer names in the canonical ``decode`` subgraph to map every
retained tensor, then proves that round-to-nearest-even BF16 conversion of the
compiled FLOAT32 buffer is byte-identical to the public checkpoint tensor.

This closes a checkpoint-to-compiled-graph mapping gap.  It does not recover
the low 16 mantissa bits discarded by the public BF16 checkpoint, Google's
float32 master weights, or any private QAT/calibration recipe.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_gemma4_mobile_checkpoint_parity import _buffer_bytes
from audit_hf_retained_constant_parity import (
    _read_local_entries,
    _read_local_header,
)
from build_converter_random_topology_injection_parity import (
    _file_range_sha256,
)
from build_converter_topology_parity import (
    _read_section,
    _section_by_model_type,
    _unpack_model,
)
from ir_training.export.litertlm_inspector import (
    LiteRTLMInspectionError,
    inspect_litertlm,
)

EXPECTED_RETAINED_COUNT = 262
FLOAT32_TENSOR_TYPE = 0
DEFAULT_MODEL_TYPE = "tf_lite_prefill_decode"
LAYER_RE = re.compile(r"/layer_(\d+)/")
SOURCE_LAYER_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.")
RETAINED_SOURCE_RE = re.compile(
    r"^model\.language_model\.(?:"
    r"layers\.\d+\.(?:"
    r"(?:input_layernorm|post_attention_layernorm|post_feedforward_layernorm|"
    r"post_per_layer_input_norm|pre_feedforward_layernorm)\.weight|"
    r"self_attn\.(?:q_norm|k_norm)\.weight|layer_scalar"
    r")|norm\.weight|per_layer_projection_norm\.weight)$"
)


class Gemma4RetainedCompiledParityError(RuntimeError):
    """Raised when retained constants cannot be mapped safely."""


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(tensor.shape).reshape(-1))


def _numel(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape or (1,), dtype=np.int64))


def _float32_to_bfloat16_rne(raw: bytes) -> bytes:
    """Round little-endian IEEE float32 bytes to BF16, ties to even."""

    if len(raw) % 4:
        raise Gemma4RetainedCompiledParityError(
            f"FLOAT32 buffer size is not divisible by four: {len(raw)}"
        )
    words = np.frombuffer(raw, dtype="<u4")
    exponent = words & np.uint32(0x7F800000)
    mantissa = words & np.uint32(0x007FFFFF)
    special_nan = (exponent == np.uint32(0x7F800000)) & (
        mantissa != np.uint32(0)
    )
    rounded = words + np.uint32(0x7FFF) + (
        (words >> np.uint32(16)) & np.uint32(1)
    )
    result = (rounded >> np.uint32(16)).astype("<u2")
    # Preserve NaN as NaN even when rounding would otherwise collapse a tiny
    # payload into infinity. Learned constants are required finite later, but
    # keeping this helper IEEE-safe makes its unit contract explicit.
    if bool(np.any(special_nan)):
        result = result.copy()
        result[special_nan] |= np.uint16(0x0040)
    return result.tobytes()


def _bfloat16_to_float32(raw: bytes) -> np.ndarray:
    if len(raw) % 2:
        raise Gemma4RetainedCompiledParityError(
            f"BF16 buffer size is not divisible by two: {len(raw)}"
        )
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return np.ascontiguousarray((words << np.uint32(16)).view(np.float32))


def _source_key_from_consumer(output_name: str) -> str | None:
    """Translate an official decode-graph consumer into a HF source key."""

    if "per_layer_embedding_projection_norm/composite" in output_name:
        return "model.language_model.per_layer_projection_norm.weight"
    if output_name == "StatefulPartitionedCall:0":
        return "model.language_model.norm.weight"
    layer_match = LAYER_RE.search(output_name)
    if not layer_match:
        return None
    layer = layer_match.group(1)
    prefix = f"model.language_model.layers.{layer}."
    roles = (
        ("/query_norm/composite", "self_attn.q_norm.weight"),
        ("/key_norm/composite", "self_attn.k_norm.weight"),
        ("/pre_attention_norm/composite", "input_layernorm.weight"),
        ("/post_attention_norm/composite", "post_attention_layernorm.weight"),
        ("/pre_ffw_norm/composite", "pre_feedforward_layernorm.weight"),
        ("/post_ffw_norm/composite", "post_feedforward_layernorm.weight"),
        ("/post_per_layer_input_norm/composite", "post_per_layer_input_norm.weight"),
        ("._maybe_apply_skip_scale/mul", "layer_scalar"),
    )
    return next(
        (prefix + suffix for marker, suffix in roles if marker in output_name),
        None,
    )


def _select_source_entries(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for name, entry in header.items():
        if not RETAINED_SOURCE_RE.fullmatch(str(name)):
            continue
        if not isinstance(entry, dict):
            raise Gemma4RetainedCompiledParityError(
                f"Invalid Safetensors entry for {name!r}."
            )
        if str(entry.get("dtype") or "") != "BF16":
            raise Gemma4RetainedCompiledParityError(
                f"Retained tensor {name!r} is not BF16: {entry.get('dtype')!r}"
            )
        shape = tuple(int(value) for value in entry.get("shape", []))
        if len(shape) > 1:
            raise Gemma4RetainedCompiledParityError(
                f"Retained tensor {name!r} is not scalar/vector: {shape}"
            )
        selected[str(name)] = entry
    return selected


def _find_decode_subgraph(model: Any) -> tuple[int, Any]:
    matches = [
        (index, subgraph)
        for index, subgraph in enumerate(model.subgraphs or [])
        if _decode(subgraph.name).strip().lower() == "decode"
    ]
    if len(matches) != 1:
        names = [_decode(item.name) for item in model.subgraphs or []]
        raise Gemma4RetainedCompiledParityError(
            f"Expected one canonical decode subgraph, found {len(matches)}: {names[:12]}"
        )
    return matches[0]


def _compiled_mappings(
    model: Any,
    section_bytes: bytes,
) -> tuple[int, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    subgraph_index, subgraph = _find_decode_subgraph(model)
    mappings: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    for operator_index, operator in enumerate(subgraph.operators or []):
        inputs = [int(value) for value in np.asarray(operator.inputs).reshape(-1)]
        outputs = [int(value) for value in np.asarray(operator.outputs).reshape(-1)]
        output_names = [_decode(subgraph.tensors[index].name) for index in outputs]
        semantic = [
            (name, _source_key_from_consumer(name)) for name in output_names
        ]
        semantic = [(name, key) for name, key in semantic if key]
        if not semantic:
            continue
        if len(semantic) != 1:
            issues.append(
                {
                    "code": "ambiguous_semantic_consumer",
                    "operator_index": operator_index,
                    "outputs": output_names,
                }
            )
            continue
        output_name, source_key = semantic[0]
        if len(inputs) < 2 or inputs[1] < 0:
            issues.append(
                {
                    "code": "semantic_consumer_has_no_constant_input",
                    "source_key": source_key,
                    "operator_index": operator_index,
                }
            )
            continue
        tensor_index = inputs[1]
        tensor = subgraph.tensors[tensor_index]
        buffer_index = int(tensor.buffer)
        if int(tensor.type) != FLOAT32_TENSOR_TYPE or buffer_index <= 0:
            issues.append(
                {
                    "code": "semantic_input_not_float32_constant",
                    "source_key": source_key,
                    "operator_index": operator_index,
                    "tensor_index": tensor_index,
                    "tensor_type": int(tensor.type),
                    "buffer_index": buffer_index,
                }
            )
            continue
        if source_key in mappings:
            issues.append(
                {
                    "code": "duplicate_semantic_mapping",
                    "source_key": source_key,
                    "first_operator": mappings[source_key]["operator_index"],
                    "second_operator": operator_index,
                }
            )
            continue
        raw = _buffer_bytes(model, section_bytes, buffer_index)
        mappings[source_key] = {
            "source_key": source_key,
            "subgraph_index": subgraph_index,
            "operator_index": operator_index,
            "tensor_index": tensor_index,
            "tensor_name": _decode(tensor.name),
            "consumer_output": output_name,
            "buffer_index": buffer_index,
            "compiled_shape": list(_shape(tensor)),
            "compiled_float32": raw,
        }
    return subgraph_index, mappings, issues


def _kind(source_key: str) -> str:
    if source_key.endswith(".layer_scalar"):
        return "layer_scalar"
    if ".self_attn.q_norm." in source_key:
        return "q_norm"
    if ".self_attn.k_norm." in source_key:
        return "k_norm"
    if source_key == "model.language_model.norm.weight":
        return "final_norm"
    if source_key == "model.language_model.per_layer_projection_norm.weight":
        return "per_layer_projection_norm"
    return source_key.rsplit(".", 2)[-2]


def run(
    artifact: str | Path,
    source_safetensors: str | Path,
    *,
    official_artifact_sha256: str,
    model_type: str = DEFAULT_MODEL_TYPE,
    output: str | Path | None = None,
) -> dict[str, Any]:
    artifact_path = Path(artifact).expanduser().resolve()
    source_path = Path(source_safetensors).expanduser().resolve()
    expected_artifact_hash = str(official_artifact_sha256).strip().lower()
    if not artifact_path.is_file():
        raise Gemma4RetainedCompiledParityError(
            f"Official LiteRT-LM artifact does not exist: {artifact_path}"
        )
    if not source_path.is_file():
        raise Gemma4RetainedCompiledParityError(
            f"Public mobile Safetensors does not exist: {source_path}"
        )
    if len(expected_artifact_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_artifact_hash
    ):
        raise Gemma4RetainedCompiledParityError(
            "--official-artifact-sha256 must be 64 lowercase hexadecimal characters."
        )

    observed_artifact_hash = _file_range_sha256(
        artifact_path, 0, artifact_path.stat().st_size
    )
    header_size, header = _read_local_header(source_path)
    source_entries = _select_source_entries(header)
    source_values = _read_local_entries(source_path, header_size, source_entries)
    try:
        package = inspect_litertlm(artifact_path, inspect_tflite=False)
        section = _section_by_model_type(package, model_type)
        section_bytes = _read_section(artifact_path, section)
        model, _, _ = _unpack_model(section_bytes)
    except (OSError, LiteRTLMInspectionError, ValueError) as exc:
        raise Gemma4RetainedCompiledParityError(
            f"Could not inspect official target section: {exc}"
        ) from exc

    subgraph_index, compiled, semantic_issues = _compiled_mappings(
        model, section_bytes
    )
    issues = list(semantic_issues)
    comparisons: list[dict[str, Any]] = []
    kind_counts: collections.Counter[str] = collections.Counter()
    exact_count = 0
    for source_key in sorted(source_entries):
        source_entry = source_entries[source_key]
        source_shape = tuple(int(value) for value in source_entry.get("shape", []))
        mapping = compiled.get(source_key)
        if mapping is None:
            issues.append(
                {"code": "missing_compiled_mapping", "source_key": source_key}
            )
            continue
        compiled_shape = tuple(int(value) for value in mapping["compiled_shape"])
        source_raw = source_values[source_key]
        compiled_raw = bytes(mapping["compiled_float32"])
        public_mapping = {
            key: value
            for key, value in mapping.items()
            if key != "compiled_float32"
        }
        shape_compatible = bool(
            source_shape == compiled_shape
            or (_numel(source_shape) == 1 and _numel(compiled_shape) == 1)
        )
        size_compatible = len(compiled_raw) == _numel(source_shape) * 4
        compiled_values = np.frombuffer(compiled_raw, dtype="<f4")
        finite = bool(np.all(np.isfinite(compiled_values)))
        rounded = _float32_to_bfloat16_rne(compiled_raw)
        bf16_exact = rounded == source_raw
        source_float32 = _bfloat16_to_float32(source_raw)
        max_abs_error = (
            float(np.max(np.abs(compiled_values - source_float32)))
            if size_compatible and len(compiled_values)
            else None
        )
        exact = bool(shape_compatible and size_compatible and finite and bf16_exact)
        exact_count += int(exact)
        kind_counts[_kind(source_key)] += 1
        comparison = {
            **public_mapping,
            "source_shape": list(source_shape),
            "source_dtype": "BF16",
            "compiled_dtype": "FLOAT32",
            "shape_compatible": shape_compatible,
            "size_compatible": size_compatible,
            "compiled_values_finite": finite,
            "compiled_float32_rounds_to_source_bf16_rne": bf16_exact,
            "source_bf16_sha256": hashlib.sha256(source_raw).hexdigest(),
            "compiled_float32_sha256": hashlib.sha256(compiled_raw).hexdigest(),
            "max_abs_float32_vs_bf16_value": max_abs_error,
            "exact_within_public_bf16_precision": exact,
        }
        comparisons.append(comparison)
        if not exact:
            issues.append(
                {
                    "code": "compiled_value_or_shape_mismatch",
                    "source_key": source_key,
                    "shape_compatible": shape_compatible,
                    "size_compatible": size_compatible,
                    "compiled_values_finite": finite,
                    "bf16_rne_exact": bf16_exact,
                }
            )

    source_keys = set(source_entries)
    mapped_keys = set(compiled)
    unexpected_mappings = sorted(mapped_keys - source_keys)
    if unexpected_mappings:
        issues.append(
            {
                "code": "unexpected_compiled_semantic_mappings",
                "source_keys": unexpected_mappings,
            }
        )
    if len(source_entries) != EXPECTED_RETAINED_COUNT:
        issues.append(
            {
                "code": "unexpected_source_retained_count",
                "expected": EXPECTED_RETAINED_COUNT,
                "observed": len(source_entries),
            }
        )

    artifact_identity_match = observed_artifact_hash == expected_artifact_hash
    if not artifact_identity_match:
        issues.append(
            {
                "code": "official_artifact_sha256_mismatch",
                "expected": expected_artifact_hash,
                "observed": observed_artifact_hash,
            }
        )
    all_exact = bool(
        artifact_identity_match
        and len(source_entries) == EXPECTED_RETAINED_COUNT
        and len(comparisons) == EXPECTED_RETAINED_COUNT
        and exact_count == EXPECTED_RETAINED_COUNT
        and not issues
    )
    result = {
        "ok": all_exact,
        "training_executed": False,
        "private_qat_recipe_recovered": False,
        "artifact": {
            "path": str(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "sha256_expected": expected_artifact_hash,
            "sha256_observed": observed_artifact_hash,
            "identity_match": artifact_identity_match,
            "model_type": model_type,
            "section_begin_offset": int(section["begin_offset"]),
            "section_size_bytes": int(section["size"]),
            "decode_subgraph_index": subgraph_index,
        },
        "source": {
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "retained_tensor_count": len(source_entries),
            "kind_counts": dict(sorted(kind_counts.items())),
        },
        "comparison": {
            "semantic_mapping_count": len(compiled),
            "compared_tensor_count": len(comparisons),
            "exact_bf16_rne_tensor_count": exact_count,
            "all_compiled_float32_round_to_source_bf16": all_exact,
            "unique_compiled_buffer_count": len(
                {int(item["buffer_index"]) for item in comparisons}
            ),
            "mappings": comparisons,
        },
        "issues": issues,
        "compiled_graph_mapping_verified": all_exact,
        "precision_boundary": (
            "The public BF16 values identify the compiled FLOAT32 constants only "
            "to their exact round-to-nearest-even BF16 intervals. Lower FLOAT32 "
            "mantissa bits and Google's master checkpoint remain unrecovered."
        ),
    }
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Map public Gemma 4 mobile BF16 retained constants into an official "
            "LiteRT-LM target graph."
        )
    )
    parser.add_argument("artifact", help="Read-only official .litertlm package.")
    parser.add_argument("--source-safetensors", required=True)
    parser.add_argument("--official-artifact-sha256", required=True)
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run(
            args.artifact,
            args.source_safetensors,
            official_artifact_sha256=args.official_artifact_sha256,
            model_type=args.model_type,
            output=args.output,
        )
    except (OSError, ValueError, Gemma4RetainedCompiledParityError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
