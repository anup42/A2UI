"""Test public BF16 projection candidates against a Gemma 4 mobile section.

The public ``gemma-4-E2B-it-qat-mobile-transformers`` checkpoint exposes the
per-layer model projection as BF16, but does not expose the W8 scale tensor.
This read-only audit tests the ordinary public symmetric per-output-channel
W8 candidates (scale precision and rounding variants) against the released
LiteRT-LM projection.  It is deliberately a falsification tool: a failed
candidate does not recover Google's private source precision or exporter.

The source may be a local safetensors file or a direct HTTP(S) safetensors
URL.  Remote mode reads only the header and the one projection range; it does
not download or retain the multi-gigabyte checkpoint.
"""

from __future__ import annotations

import argparse
import json
import mmap
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_gemma4_mobile_checkpoint_parity import (  # noqa: E402
    Gemma4CheckpointParityError,
    _buffer_bytes,
    _records,
    _source_key,
)
from audit_litertlm_remote_source_recipe import (  # noqa: E402
    _entry_with_header,
    _source_array,
    _source_array_local,
    read_local_safetensors_header,
    read_safetensors_header,
)
from build_converter_topology_parity import (  # noqa: E402
    _read_section,
    _section_by_model_type,
    _unpack_model,
)
from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    inspect_litertlm,
)


class ProjectionParityError(RuntimeError):
    """Raised when the projection audit cannot run."""


_PROJECTION_KEY = "model.language_model.per_layer_model_projection.weight"
_QMAX = 127.0


def _is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def _round_codes(values: np.ndarray, mode: str) -> np.ndarray:
    """Round scaled values using common signed quantizer tie rules."""

    if mode == "rint":
        rounded = np.rint(values)
    elif mode == "half_up":
        rounded = np.floor(values + 0.5)
    elif mode == "ties_to_zero":
        absolute = np.abs(values)
        floor_value = np.floor(absolute)
        rounded_abs = np.floor(absolute + 0.5)
        exact_half = (absolute - floor_value) == 0.5
        rounded = np.sign(values) * np.where(exact_half, floor_value, rounded_abs)
    else:  # pragma: no cover - guarded by the CLI choices
        raise ValueError(f"Unsupported rounding mode: {mode}")
    return np.clip(rounded, -127, 127).astype(np.int8, copy=False)


def _scale_candidates(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return scale candidates that differ only in reduction precision."""

    f32 = np.asarray(values, dtype=np.float32)
    f64 = np.asarray(values, dtype=np.float64)
    max_f32 = np.max(np.abs(f32), axis=1)
    max_f64 = np.max(np.abs(f64), axis=1)
    return {
        "max_abs_f32_div_f32": (max_f32 / np.float32(_QMAX)).astype(np.float32),
        "max_abs_f32_div_f64": (max_f32 / _QMAX).astype(np.float32),
        "max_abs_f64_div_f64": (max_f64 / _QMAX).astype(np.float32),
    }


def _candidate_report(
    source: np.ndarray,
    official_raw: bytes,
    official_scales: np.ndarray,
    *,
    scale_name: str,
    scales: np.ndarray,
    rounding: str,
) -> dict[str, Any]:
    scaled = np.asarray(source, dtype=np.float32) / scales[:, None]
    quantized = _round_codes(scaled, rounding)
    official_values = np.frombuffer(official_raw, dtype=np.int8)
    candidate_values = quantized.reshape(-1)
    if candidate_values.size != official_values.size:
        raise ProjectionParityError(
            f"Candidate has {candidate_values.size} codes; official has "
            f"{official_values.size}."
        )
    scale_diff = np.asarray(scales, dtype=np.float32) - np.asarray(
        official_scales, dtype=np.float32
    )
    code_diff = candidate_values.astype(np.int16) - official_values.astype(np.int16)
    return {
        "scale_candidate": scale_name,
        "rounding": rounding,
        "scale_exact": bool(np.array_equal(scales, official_scales)),
        "scale_mismatch_count": int(np.count_nonzero(scale_diff)),
        "max_scale_abs_error": float(np.max(np.abs(scale_diff))),
        "quantized_codes_exact": bool(np.array_equal(candidate_values, official_values)),
        "quantized_code_mismatch_count": int(np.count_nonzero(code_diff)),
        "max_abs_code_error": int(np.max(np.abs(code_diff))),
    }


def _bfloat16_half_ulp(values: np.ndarray) -> np.ndarray:
    """Return a conservative half-ULP interval for decoded BF16 values.

    ``_source_array`` decodes BF16 values into float32 containers.  The
    original float32 value used by Google's exporter can be anywhere in the
    BF16 rounding cell around that decoded value.  For normal finite values
    the cell half-width is ``2**(exponent-8)``.  Treat zero and subnormal
    values conservatively; the projection tensor is normal in the released
    checkpoints, but those branches keep this helper safe for reuse.
    """

    decoded = np.asarray(values, dtype=np.float32)
    absolute = np.abs(decoded)
    exponent = np.floor(np.log2(np.maximum(absolute, np.float32(1e-38))))
    half = np.exp2(exponent - np.float32(8.0)).astype(np.float32)
    # BF16 subnormals and zero are represented conservatively by the smallest
    # normal BF16 half-cell.  This is only a feasibility bound, never a claim
    # about the source serializer.
    smallest = np.float32(2.0**-134)
    return np.maximum(half, smallest)


def _bfloat16_interval_report(
    source: np.ndarray,
    official_raw: bytes,
    official_scales: np.ndarray,
) -> dict[str, Any]:
    """Check whether hidden pre-BF16 values could explain official W8 data.

    The check is deliberately a *necessary-condition* audit.  It does not
    recover the pre-BF16 source tensor.  It answers two narrower questions:

    * can every official ``scale * 127`` be the row max of a value inside the
      BF16 rounding cells around the public checkpoint; and
    * can each official code be produced by a symmetric W8 max-abs quantizer
      from at least one value in that cell.

    A full pass means that BF16 truncation alone is sufficient to explain the
    observed one-code/scale differences; it does not prove the private
    observer or exporter implementation.
    """

    decoded = np.asarray(source, dtype=np.float32)
    if decoded.ndim != 2:
        raise ProjectionParityError(
            f"Expected a matrix for BF16 interval analysis, got {decoded.shape}."
        )
    scales = np.asarray(official_scales, dtype=np.float32).reshape(-1)
    if scales.size != decoded.shape[0]:
        raise ProjectionParityError(
            f"Expected {decoded.shape[0]} scales, got {scales.size}."
        )
    official_codes = np.frombuffer(official_raw, dtype=np.int8)
    if official_codes.size != decoded.size:
        raise ProjectionParityError(
            f"Expected {decoded.size} official codes, got {official_codes.size}."
        )
    official_codes = official_codes.reshape(decoded.shape).astype(np.int16)

    half_ulp = _bfloat16_half_ulp(decoded)
    absolute = np.abs(decoded)
    source_lower = np.maximum(absolute - half_ulp, np.float32(0.0))
    source_upper = absolute + half_ulp
    row_lower = np.max(source_lower, axis=1)
    row_upper = np.max(source_upper, axis=1)
    target_max = scales * np.float32(_QMAX)
    scale_rows_contained = (target_max >= row_lower) & (target_max <= row_upper)

    # The interior pre-image of signed round-to-nearest code q is bounded by
    # (q +/- 0.5) * scale.  Saturated endpoints have one unbounded side.
    code_float = official_codes.astype(np.float32)
    lower = (code_float - np.float32(0.5)) * scales[:, None]
    upper = (code_float + np.float32(0.5)) * scales[:, None]
    lower = np.where(official_codes == -127, -np.inf, lower)
    upper = np.where(official_codes == 127, np.inf, upper)
    code_intersections = (decoded + half_ulp >= lower) & (
        decoded - half_ulp <= upper
    )

    return {
        "source_precision": "BF16",
        "necessary_condition_only": True,
        "scale_row_count": int(scales.size),
        "scale_rows_within_bfloat16_max_abs_interval": int(
            np.count_nonzero(scale_rows_contained)
        ),
        "scale_rows_outside_bfloat16_max_abs_interval": int(
            np.count_nonzero(~scale_rows_contained)
        ),
        "min_scale_interval_margin": float(
            np.min(np.minimum(target_max - row_lower, row_upper - target_max))
        ),
        "code_count": int(official_codes.size),
        "codes_with_bfloat16_preimage": int(np.count_nonzero(code_intersections)),
        "codes_without_bfloat16_preimage": int(np.count_nonzero(~code_intersections)),
        "all_scales_are_interval_compatible": bool(np.all(scale_rows_contained)),
        "all_codes_are_interval_compatible": bool(np.all(code_intersections)),
        "interpretation": (
            "Every released scale and code is compatible with at least one hidden "
            "pre-BF16 value in the public checkpoint's rounding cells. This "
            "supports (but does not prove) max-abs/127 W8 quantization before "
            "BF16 serialization; it cannot recover the missing higher-precision "
            "weights or private QAT/export path."
        ),
    }


def _load_projection(source: str | Path) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Read only the public projection tensor and return values, dtype, metadata."""

    source_value = str(source)
    if _is_url(source_value):
        header = read_safetensors_header(source_value)
        entry = header.get(_PROJECTION_KEY)
        if not isinstance(entry, dict):
            raise ProjectionParityError(
                f"Remote source has no {_PROJECTION_KEY!r} tensor."
            )
        values = _source_array(
            source_value,
            _entry_with_header(entry, header),
            timeout=180.0,
        )
        return values, str(entry.get("dtype")), {
            "source_kind": "remote_safetensors",
            "header_size": int(header["__a2ui_header_size__"]),
            "file_size": header.get("__a2ui_file_size__"),
            "source_key": _PROJECTION_KEY,
            "source_shape": list(values.shape),
            "source_dtype": str(entry.get("dtype")),
            "source_scale_key_present": _PROJECTION_KEY + "_scale" in header,
        }

    source_path = Path(source_value).expanduser().resolve()
    if not source_path.is_file():
        raise ProjectionParityError(f"Source safetensors does not exist: {source_path}")
    header = read_local_safetensors_header(source_path)
    entry = header.get(_PROJECTION_KEY)
    if not isinstance(entry, dict):
        raise ProjectionParityError(f"Source has no {_PROJECTION_KEY!r} tensor.")
    with source_path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            values = _source_array_local(
                mapped,
                entry,
                header_size=int(header["__a2ui_header_size__"]),
            )
    return values, str(entry.get("dtype")), {
        "source_kind": "local_safetensors",
        "file_size": int(source_path.stat().st_size),
        "source_path": str(source_path),
        "source_key": _PROJECTION_KEY,
        "source_shape": list(values.shape),
        "source_dtype": str(entry.get("dtype")),
        "source_scale_key_present": _PROJECTION_KEY + "_scale" in header,
    }


def run(
    artifact: str | Path,
    source: str | Path,
    *,
    model_type: str = "tf_lite_prefill_decode",
    roundings: Iterable[str] = ("rint", "half_up", "ties_to_zero"),
    output: str | Path | None = None,
) -> dict[str, Any]:
    artifact_path = Path(artifact).expanduser().resolve()
    if not artifact_path.is_file():
        raise ProjectionParityError(f"Official artifact does not exist: {artifact_path}")
    try:
        package = inspect_litertlm(artifact_path, inspect_tflite=False)
        section = _section_by_model_type(package, model_type)
        section_bytes = _read_section(artifact_path, section)
        model, _, _ = _unpack_model(section_bytes)
        records = list(_records(model, section_bytes))
    except (OSError, LiteRTLMInspectionError, Gemma4CheckpointParityError) as exc:
        raise ProjectionParityError(f"Could not inspect official artifact: {exc}") from exc

    projection = None
    for ordinal, record in enumerate(records):
        source_key, _ = _source_key(model_type, record["name"], ordinal)
        if source_key == _PROJECTION_KEY:
            projection = record
            break
    if projection is None:
        raise ProjectionParityError(
            f"No {_PROJECTION_KEY!r} mapping was found in {model_type!r}."
        )
    source_values, source_dtype, source_metadata = _load_projection(source)
    if tuple(source_values.shape) != tuple(projection["shape"]):
        raise ProjectionParityError(
            f"Projection shape mismatch: source={source_values.shape}, "
            f"official={projection['shape']}."
        )
    if int(projection["bits"]) != 8:
        raise ProjectionParityError(
            f"Expected an official W8 projection, found W{projection['bits']}."
        )

    official_raw = _buffer_bytes(model, section_bytes, int(projection["buffer"]))
    official_scales = np.asarray(projection["scales"], dtype=np.float32).reshape(-1)
    candidates = _scale_candidates(source_values)
    reports = [
        _candidate_report(
            source_values,
            official_raw,
            official_scales,
            scale_name=scale_name,
            scales=scales,
            rounding=rounding,
        )
        for scale_name, scales in candidates.items()
        for rounding in roundings
    ]
    interval_report = (
        _bfloat16_interval_report(source_values, official_raw, official_scales)
        if source_dtype.upper() == "BF16"
        else None
    )
    result = {
        "artifact": str(artifact_path),
        "model_type": model_type,
        "official_projection": {
            "shape": list(projection["shape"]),
            "bits": int(projection["bits"]),
            "buffer": int(projection["buffer"]),
            "raw_bytes": len(official_raw),
            "scale_count": int(official_scales.size),
            "scale_preview": official_scales[:8].tolist(),
        },
        "source": {
            **source_metadata,
            "decoded_dtype": source_dtype,
        },
        "candidate_reports": reports,
        "bfloat16_interval_report": interval_report,
        "training_executed": False,
        "private_qat_recipe_recovered": False,
        "exact_public_projection_candidate": any(
            item["scale_exact"] and item["quantized_codes_exact"] for item in reports
        ),
        "interpretation": (
            "A public BF16 projection candidate exactly reproduces the official W8 "
            "projection under the tested scale and rounding rules."
            if any(item["scale_exact"] and item["quantized_codes_exact"] for item in reports)
            else (
                "The public projection and all tested ordinary symmetric W8 candidates "
                "do not reproduce the official scale/code pair. The BF16 interval audit "
                "shows that hidden pre-BF16 values could still explain every released "
                "scale and code; the exact higher-precision source, learned observer, "
                "and exporter remain unavailable."
                if interval_report and interval_report["all_scales_are_interval_compatible"]
                and interval_report["all_codes_are_interval_compatible"]
                else "The public projection and all tested ordinary symmetric W8 candidates "
                "do not reproduce the official scale/code pair. The missing information is "
                "source precision, a learned QAT scale/observer, an export transform, or a "
                "combination; this audit cannot recover it from the BF16 tensor."
            )
        ),
    }
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result["output"] = str(output_path)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Official Gemma 4 .litertlm artifact.")
    parser.add_argument(
        "--source",
        required=True,
        help="Local model.safetensors path or direct HTTP(S) safetensors URL.",
    )
    parser.add_argument("--model-type", default="tf_lite_prefill_decode")
    parser.add_argument(
        "--rounding",
        action="append",
        choices=("rint", "half_up", "ties_to_zero"),
        dest="roundings",
        help="Rounding rule to test; repeat to restrict the candidate matrix.",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 unless one tested public candidate is exact.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run(
            args.artifact,
            args.source,
            model_type=args.model_type,
            roundings=args.roundings or ("rint", "half_up", "ties_to_zero"),
            output=args.output,
        )
    except (OSError, ProjectionParityError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict and not result["exact_public_projection_candidate"]:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
