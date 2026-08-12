#!/usr/bin/env python3
"""Fail-closed static/streamed preflight for corrected Gemma 4 mobile QAT.

This command does not load the 10 GB model and never trains.  It verifies the
training YAML, reconstructed seed, retained-qparams sidecar, and (when the
official packed source is available) streams every trainable projection to
prove BF16 cell centers re-encode to the exact published integer codes.
The separate ``train_sft.py --preflight-only`` command performs the real-model
completion-loss/top-token gate immediately before an executable run.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ir_training.common.config import load_yaml, resolve_path
from ir_training.qat.fake_quant import QATSpec
from ir_training.qat.mobile_qparams import verify_mobile_qparams_contract
from ir_training.qat.mobile_training_seed import verify_configured_mobile_training_seed
from ir_training.qat.mobile_training_seed import OFFICIAL_MOBILE_SAFETENSORS_SHA256
from ir_training.qat.workflow import validate_qat_config
from reconstruct_gemma4_mobile_training_seed import (
    Gemma4MobileSeedError,
    TensorTransform,
    _entry,
    _read_safetensors_header,
)


class ScalePreservingQATError(RuntimeError):
    """Raised when a mobile retained-scale gate cannot be proven."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transform(item: dict[str, Any]) -> TensorTransform:
    return TensorTransform(
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
        scale_begin=int(item["scale_begin"]) if item.get("scale_begin") is not None else None,
        scale_end=int(item["scale_end"]) if item.get("scale_end") is not None else None,
        scale_group_width=(
            int(item["scale_group_width"])
            if item.get("scale_group_width") is not None
            else None
        ),
    )


def _unpack_codes(raw: bytes, bits: int, logical_count: int) -> np.ndarray:
    packed = np.frombuffer(raw, dtype=np.uint8)
    if bits == 8:
        return packed.view(np.int8)[:logical_count]
    mask = (1 << bits) - 1
    values_per_byte = 8 // bits
    codes = np.empty(len(packed) * values_per_byte, dtype=np.int8)
    for lane in range(values_per_byte):
        unsigned = ((packed >> (lane * bits)) & mask).astype(np.int16)
        # The published W2/W4 bytes use an unsigned offset encoding, not a
        # two's-complement nibble: 0 maps to qmin and all-ones maps to qmax.
        signed = unsigned - (1 << (bits - 1))
        codes[lane::values_per_byte] = signed.astype(np.int8)
    return codes[:logical_count]


def _validate_materialized_projection(
    source: BinaryIO,
    tensor: TensorTransform,
    *,
    dense: np.ndarray,
    sidecar_scales: np.ndarray,
    working_set_bytes: int,
) -> dict[str, Any]:
    if tensor.bits not in {2, 4, 8} or tensor.scale_begin is None:
        raise ScalePreservingQATError(
            f"Incomplete projection qparams for {tensor.output_key!r}."
        )
    source.seek(tensor.scale_begin)
    scale_bytes = source.read((tensor.scale_end or 0) - tensor.scale_begin)
    scales = np.frombuffer(scale_bytes, dtype="<f4").reshape(tensor.scale_shape)
    sidecar_scales = np.asarray(sidecar_scales, dtype=np.float32)
    sidecar_bytes = np.asarray(sidecar_scales, dtype="<f4").tobytes()
    scale_bytes_exact = sidecar_bytes == scale_bytes
    if tuple(dense.shape) != tuple(tensor.output_shape):
        raise ScalePreservingQATError(
            f"Materialized seed shape for {tensor.output_key!r} is "
            f"{tuple(dense.shape)}, expected {tensor.output_shape}."
        )
    if tuple(sidecar_scales.shape) != tuple(tensor.scale_shape):
        raise ScalePreservingQATError(
            f"Sidecar scale shape for {tensor.output_key!r} is "
            f"{tuple(sidecar_scales.shape)}, expected {tensor.scale_shape}."
        )
    if scales.shape[1] != 1:
        raise ScalePreservingQATError(
            f"Trainable projection {tensor.output_key!r} unexpectedly uses grouped scales."
        )
    rows = int(tensor.output_shape[0])
    logical_columns = int(tensor.output_shape[1])
    row_output_bytes = logical_columns * 2
    rows_per_chunk = max(1, int(working_set_bytes) // max(row_output_bytes, 1))
    qmin = -127 if tensor.bits == 8 else -(1 << (tensor.bits - 1))
    qmax = (1 << (tensor.bits - 1)) - 1
    mismatch_count = 0
    value_count = 0
    original_histogram: dict[int, int] = {}
    recovered_histogram: dict[int, int] = {}
    for row_start in range(0, rows, rows_per_chunk):
        row_count = min(rows_per_chunk, rows - row_start)
        row_source_bytes = int(tensor.source_shape[1])
        source.seek(tensor.source_begin + row_start * row_source_bytes)
        packed = source.read(row_count * row_source_bytes)
        original = _unpack_codes(
            packed,
            tensor.bits,
            row_count * logical_columns,
        ).reshape(row_count, logical_columns)
        chunk_dense = np.asarray(
            dense[row_start : row_start + row_count], dtype=np.float32
        )
        chunk_scales = sidecar_scales[row_start : row_start + row_count]
        recovered = np.clip(
            np.rint(chunk_dense / chunk_scales), qmin, qmax
        ).astype(np.int8)
        mismatch_count += int(np.count_nonzero(recovered != original))
        value_count += int(original.size)
        for values, histogram in (
            (original, original_histogram),
            (recovered, recovered_histogram),
        ):
            unique, counts = np.unique(values, return_counts=True)
            for code, count in zip(unique.tolist(), counts.tolist(), strict=True):
                histogram[int(code)] = histogram.get(int(code), 0) + int(count)
    return {
        "weight_key": tensor.output_key,
        "bits": tensor.bits,
        "value_count": value_count,
        "mismatch_count": mismatch_count,
        "sidecar_source_scale_bytes_exact": scale_bytes_exact,
        "code_histogram_preserved": original_histogram == recovered_histogram,
        "original_code_histogram": original_histogram,
    }


def _validate_source_activation_scales(
    source: BinaryIO,
    tensor: TensorTransform,
    *,
    header: dict[str, Any],
    data_start: int,
    file_size: int,
    inventory_entry: dict[str, Any],
) -> dict[str, Any]:
    if not tensor.source_key.endswith(".weight"):
        raise ScalePreservingQATError(
            f"Trainable projection source is not a weight: {tensor.source_key!r}."
        )
    stem = tensor.source_key[: -len(".weight")]
    role_checks: dict[str, bool] = {}
    for role in ("input", "output"):
        source_key = f"{stem}.{role}_activation_scale"
        dtype, shape, begin, end = _entry(
            header,
            source_key,
            data_start=data_start,
            file_size=file_size,
        )
        if dtype != "F32" or shape not in {(), (1,)} or end - begin != 4:
            raise ScalePreservingQATError(
                f"Expected scalar F32 {role} activation scale for "
                f"{tensor.output_key!r}; got {dtype} {shape}."
            )
        source.seek(begin)
        raw_source = source.read(4)
        try:
            raw_contract = bytes.fromhex(
                str(
                    inventory_entry[
                        f"{role}_activation_scale_f32_le_hex"
                    ]
                )
            )
        except (KeyError, ValueError) as exc:
            raise ScalePreservingQATError(
                f"Retained {role} A8 scale is missing/malformed for "
                f"{tensor.output_key!r}."
            ) from exc
        role_checks[role] = bool(
            len(raw_source) == 4
            and len(raw_contract) == 4
            and raw_source == raw_contract
        )
    return {
        "input_activation_scale_source_bytes_exact": role_checks["input"],
        "output_activation_scale_source_bytes_exact": role_checks["output"],
    }


def validate(
    config_path: str | Path,
    *,
    source_safetensors: str | Path | None = None,
    strict: bool = False,
    working_set_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    config = load_yaml(path)
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    qat = config.get("qat") if isinstance(config.get("qat"), dict) else {}
    issues = validate_qat_config(config)
    static_errors = [issue for issue in issues if issue.severity == "error"]
    seed = verify_configured_mobile_training_seed(
        model, base=ROOT, require_materialized=True
    )
    contract_value = qat.get("mobile_qparams_contract") or model.get(
        "mobile_qparams_contract"
    )
    qparams = verify_mobile_qparams_contract(contract_value, base=ROOT)
    spec = QATSpec.from_config(config)
    checks = {
        "static_config_valid": not static_errors,
        "mobile_seed_verified": bool(seed.get("verified")),
        "retained_qparams_verified": bool(qparams.get("verified")),
        "retained_scale_mode": spec.scale_mode == "retained_mobile",
        "fixed_weight_scales_required": spec.fixed_scale_required,
        "fixed_activation_scales_required": spec.fixed_activation_scale_required,
        "effective_lora_only": spec.effective_lora_only,
        "clipped_ste": spec.ste_gradient == "clipped",
    }
    source_value = source_safetensors
    if source_value is None and seed.get("path"):
        try:
            seed_manifest = json.loads(
                Path(str(seed["path"])).read_text(encoding="utf-8")
            )
            source_value = (
                seed_manifest.get("source", {}).get("safetensors")
                if isinstance(seed_manifest.get("source"), dict)
                else None
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            source_value = None
    source_path = (
        Path(str(source_value)).expanduser().resolve()
        if source_value is not None and str(source_value).strip()
        else None
    )
    projections: list[dict[str, Any]] = []
    source_sha256 = (
        _sha256_file(source_path)
        if source_path is not None and source_path.is_file()
        else None
    )
    checks["actual_source_identity_pinned"] = bool(
        source_sha256 == OFFICIAL_MOBILE_SAFETENSORS_SHA256
    ) if strict else bool(source_path is None or source_sha256)
    if source_path is not None and source_path.is_file() and seed.get("verified"):
        if strict and source_sha256 != OFFICIAL_MOBILE_SAFETENSORS_SHA256:
            raise ScalePreservingQATError(
                "Strict retained-scale validation requires the exact pinned "
                "official packed Safetensors; observed SHA-256 "
                f"{source_sha256!r}, expected "
                f"{OFFICIAL_MOBILE_SAFETENSORS_SHA256}."
            )
        manifest = json.loads(Path(str(seed["path"])).read_text(encoding="utf-8"))
        mappings = manifest["transformation"]["tensor_mappings"]
        weight_map = manifest["output"]["weight_map"]
        seed_directory = Path(str(seed["path"])).parent.resolve()
        transforms = [
            _transform(item)
            for item in mappings
            if isinstance(item, dict)
            and item.get("scale_key")
            and str(item.get("output_key") or "").startswith("model.layers.")
            and any(
                str(item.get("output_key") or "").endswith(suffix)
                for suffix in (
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                )
            )
        ]
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ScalePreservingQATError(
                "Strict materialized-seed validation requires safetensors."
            ) from exc
        storage_path = Path(str(qparams.get("scale_storage_path") or ""))
        if not storage_path.is_file():
            raise ScalePreservingQATError(
                f"Verified qparams scale storage is unavailable: {storage_path}"
            )
        _, source_data_start, source_header = _read_safetensors_header(source_path)
        source_file_size = source_path.stat().st_size
        qparams_inventory = qparams.get("inventory")
        if not isinstance(qparams_inventory, dict):
            raise ScalePreservingQATError(
                "Verified retained-qparams report has no inventory."
            )
        activation_scale_checks = 0
        with contextlib.ExitStack() as stack, source_path.open("rb") as source:
            qparam_handle = stack.enter_context(
                safe_open(str(storage_path), framework="pt", device="cpu")
            )
            shard_handles: dict[Path, Any] = {}
            for tensor in transforms:
                shard_path = (
                    seed_directory / str(weight_map[tensor.output_key])
                ).resolve()
                handle = shard_handles.get(shard_path)
                if handle is None:
                    handle = stack.enter_context(
                        safe_open(str(shard_path), framework="pt", device="cpu")
                    )
                    shard_handles[shard_path] = handle
                dense_tensor = handle.get_tensor(tensor.output_key).float()
                scale_tensor = qparam_handle.get_tensor(tensor.output_key).float()
                dense = np.ascontiguousarray(dense_tensor.numpy())
                sidecar_scales = np.ascontiguousarray(scale_tensor.numpy())
                projection_report = _validate_materialized_projection(
                    source,
                    tensor,
                    dense=dense,
                    sidecar_scales=sidecar_scales,
                    working_set_bytes=working_set_bytes,
                )
                inventory_entry = qparams_inventory.get(tensor.output_key)
                if not isinstance(inventory_entry, dict):
                    raise ScalePreservingQATError(
                        f"No retained-qparams entry for {tensor.output_key!r}."
                    )
                activation_report = _validate_source_activation_scales(
                    source,
                    tensor,
                    header=source_header,
                    data_start=source_data_start,
                    file_size=source_file_size,
                    inventory_entry=inventory_entry,
                )
                activation_scale_checks += len(activation_report)
                projection_report.update(activation_report)
                projections.append(projection_report)
        checks["projection_inventory_exact"] = len(projections) == 205
        checks["sidecar_source_scales_exact"] = bool(
            projections
            and all(
                item["sidecar_source_scale_bytes_exact"]
                for item in projections
            )
        )
        checks["materialized_seed_codes_exact"] = bool(
            projections
            and all(
                item["mismatch_count"] == 0
                and item["code_histogram_preserved"]
                for item in projections
            )
        )
        checks["source_activation_scales_exact"] = bool(
            activation_scale_checks == 410
            and all(
                item["input_activation_scale_source_bytes_exact"]
                and item["output_activation_scale_source_bytes_exact"]
                for item in projections
            )
        )
    else:
        checks["projection_inventory_exact"] = not strict
        checks["sidecar_source_scales_exact"] = not strict
        checks["materialized_seed_codes_exact"] = not strict
        checks["source_activation_scales_exact"] = not strict
    report = {
        "report_version": 1,
        "config": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "strict": bool(strict),
        "source_safetensors": str(source_path) if source_path else None,
        "source_safetensors_sha256": source_sha256,
        "source_safetensors_sha256_expected": OFFICIAL_MOBILE_SAFETENSORS_SHA256,
        "checks": checks,
        "static_issues": [issue.__dict__ for issue in issues],
        "mobile_training_seed": seed,
        "mobile_qparams": {
            key: value for key, value in qparams.items() if key != "inventory"
        },
        "projection_count": len(projections),
        "projection_values_checked": sum(
            int(item["value_count"]) for item in projections
        ),
        "projection_mismatches": sum(
            int(item["mismatch_count"]) for item in projections
        ),
        "projection_reports": projections,
        "training_executed": False,
        "private_google_recipe_recovered": False,
    }
    report["passed"] = all(checks.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate retained-scale Gemma 4 mobile QAT without training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-safetensors")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require and stream the pinned packed source for all 205 projections.",
    )
    parser.add_argument("--working-set-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        report = validate(
            args.config,
            source_safetensors=args.source_safetensors,
            strict=bool(args.strict),
            working_set_bytes=int(args.working_set_bytes),
        )
    except (OSError, ValueError, Gemma4MobileSeedError, ScalePreservingQATError) as exc:
        parser.error(str(exc))
        return 2
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
