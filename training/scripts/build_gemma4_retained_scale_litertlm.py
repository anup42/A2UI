#!/usr/bin/env python3
"""Export retained-scale Gemma 4 LoRA weights into the official LiteRT-LM graph.

This exporter is intentionally separate from ``build_checkpoint_official_topology``.
The legacy exporter invokes the public AI Edge converter and replaces all 277 target
constants plus their scales.  That is incompatible with retained-scale mobile QAT.

This path accepts only a provenance-bound, callback-selected Golden adapter and
its merged floating checkpoint (including pinned Golden32's unique-source selector
and ordinary unique Golden100). It re-encodes exactly the 205 trained projection
matrices with the immutable published per-row scales, patches only their existing
LiteRT code buffers, and proves that restoring those 205 payloads recovers the
complete official target section byte-for-byte.  The 72 frozen target constants,
all weight/A8 quantization parameters, graph metadata, package metadata, and the
official MTP section remain unchanged.

The command is plan-only by default.  ``--execute`` is required to load projection
values, patch a copy of the official package, or write an output artifact.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import gc
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_checkpoint_official_topology import (
    SafetensorCheckpoint,
    _canonical_inventory_keys,
    _merge_provenance_report,
)
from build_converter_random_inventory_parity import _vector
from build_converter_random_topology_injection_parity import (
    _file_range_sha256,
    _graph_report,
    _official_weight_alias_groups,
    _weight_alias_summary,
)
from build_converter_topology_parity import (
    _FULLY_CONNECTED,
    _read_section,
    _section_by_model_type,
)
from build_fresh_random_quantized_graph import (
    _extract_inventory,
    _schema_model,
)
from build_random_official_topology_parity import (
    _buffer_view,
    _pack_low_bit,
    _unpack_low_bit,
)
from ir_training.common.config import load_yaml, resolve_path
from ir_training.export.litertlm_inspector import inspect_litertlm
from ir_training.export.merge_lora import (
    _checkpoint_manifest_matches_adapter,
    _golden_selection_binding,
)
from ir_training.qat.fake_quant import QATSpec
from ir_training.qat.mobile_qparams import MobileQParams
from ir_training.qat.mobile_training_seed import (
    OFFICIAL_MOBILE_MODEL_ID,
    verify_configured_mobile_training_seed,
)
from reconstruct_gemma4_mobile_training_seed import _output_key

MODE = "retained_scale_code_only_v1"
TARGET_MODEL_TYPE = "tf_lite_prefill_decode"
MTP_MODEL_TYPE = "tf_lite_mtp_drafter"
EXPECTED_TARGET_COUNT = 277
EXPECTED_MUTABLE_COUNT = 205
EXPECTED_FROZEN_COUNT = 72
EXPECTED_MUTABLE_BITS = {2: 60, 4: 145}
EXPECTED_FROZEN_BITS = {2: 1, 8: 71}
EXPECTED_TARGET_BITS = {2: 61, 4: 145, 8: 71}
EXPECTED_MUTABLE_FC_ALIAS_EDGES = 610
EXPECTED_MUTABLE_A8_RECORDS = EXPECTED_MUTABLE_FC_ALIAS_EDGES * 2
OFFICIAL_LITERTLM_SHA256 = (
    "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c"
)
_FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
_LORA_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$")


class RetainedScaleExportError(RuntimeError):
    """Raised when an exact retained-scale export gate cannot be proven."""


def _official_artifact_sha_report(declared: str, observed: str) -> dict[str, Any]:
    """Bind the exporter to the one audited public mobile package."""

    declared_sha = str(declared).strip().lower()
    observed_sha = str(observed).strip().lower()
    checks = {
        "declared_sha_is_pinned_official_sha": declared_sha == OFFICIAL_LITERTLM_SHA256,
        "observed_sha_is_pinned_official_sha": observed_sha == OFFICIAL_LITERTLM_SHA256,
        "declared_observed_sha_match": declared_sha == observed_sha,
    }
    return {
        "pinned_sha256": OFFICIAL_LITERTLM_SHA256,
        "declared_sha256": declared_sha,
        "observed_sha256": observed_sha,
        "checks": checks,
        "verified": all(checks.values()),
    }


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(value)
    return digest.hexdigest()


def _byte_view(value: Any) -> memoryview:
    view = memoryview(value)
    return view if view.ndim == 1 and view.format == "B" else view.cast("B")


def _buffer_views_equal(
    left: Any, right: Any, *, chunk_size: int = 8 * 1024 * 1024
) -> bool:
    """Compare possibly huge constant buffers without materializing copies."""

    left_view = _byte_view(left)
    right_view = _byte_view(right)
    if len(left_view) != len(right_view):
        return False
    return all(
        left_view[start : start + chunk_size] == right_view[start : start + chunk_size]
        for start in range(0, len(left_view), chunk_size)
    )


def _copy_buffer_view(
    source: Any, target: Any, *, chunk_size: int = 8 * 1024 * 1024
) -> None:
    """Copy a constant buffer in bounded chunks; target must be writable."""

    source_view = _byte_view(source)
    target_view = _byte_view(target)
    if len(source_view) != len(target_view):
        raise RetainedScaleExportError("Cannot restore differently sized buffers.")
    for start in range(0, len(source_view), chunk_size):
        target_view[start : start + chunk_size] = source_view[
            start : start + chunk_size
        ]


def _write_package_exclusive(
    official_path: Path,
    section: dict[str, Any],
    candidate_section: bytes | bytearray,
    partial_path: Path,
) -> None:
    """Stream a package into a newly-created path without a TOCTOU overwrite."""

    begin = int(section["begin_offset"])
    size = int(section["size"])
    if len(candidate_section) != size:
        raise RetainedScaleExportError(
            f"Candidate section has {len(candidate_section)} bytes, expected {size}."
        )
    created = False
    try:
        with official_path.open("rb") as source, partial_path.open("xb") as target:
            created = True
            remaining = begin
            while remaining:
                block = source.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise RetainedScaleExportError(
                        "Official package ended before the selected target section."
                    )
                target.write(block)
                remaining -= len(block)
            target.write(candidate_section)
            source.seek(begin + size)
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                target.write(block)
    except Exception:
        if created and partial_path.exists():
            partial_path.unlink()
        raise


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _link_no_clobber(source: Path, destination: Path) -> None:
    """Atomically publish one same-filesystem file without replacing a peer."""

    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise RetainedScaleExportError(
            f"Refusing to overwrite concurrently-created output: {destination}"
        ) from exc
    except OSError as exc:
        raise RetainedScaleExportError(
            "Atomic no-clobber promotion requires same-filesystem hard-link support: "
            f"{source} -> {destination}: {exc}"
        ) from exc


def _package_identity_report(
    official_path: Path,
    partial_path: Path,
    *,
    target_section: dict[str, Any],
    mtp_section: dict[str, Any],
    candidate_section: bytes | bytearray,
) -> tuple[dict[str, bool], dict[str, Any]]:
    package_size = official_path.stat().st_size
    target_begin = int(target_section["begin_offset"])
    target_size = int(target_section["size"])
    mtp_begin = int(mtp_section["begin_offset"])
    mtp_size = int(mtp_section["size"])
    suffix_begin = target_begin + target_size
    suffix_size = package_size - suffix_begin
    checks = {
        "package_size_unchanged": partial_path.stat().st_size == package_size,
        "package_prefix_byte_exact": _file_range_sha256(partial_path, 0, target_begin)
        == _file_range_sha256(official_path, 0, target_begin),
        "package_suffix_byte_exact": _file_range_sha256(
            partial_path, suffix_begin, suffix_size
        )
        == _file_range_sha256(official_path, suffix_begin, suffix_size),
        "candidate_target_section_written_exactly": _file_range_sha256(
            partial_path, target_begin, target_size
        )
        == _sha256_bytes(candidate_section),
        "mtp_byte_exact": _file_range_sha256(partial_path, mtp_begin, mtp_size)
        == _file_range_sha256(official_path, mtp_begin, mtp_size),
    }
    try:
        inspection = inspect_litertlm(partial_path, inspect_tflite=False)
        checks["output_package_parseable"] = bool(inspection.get("sections"))
    except Exception as exc:  # noqa: BLE001 - parseability is a hard export gate
        inspection = {"error": repr(exc)}
        checks["output_package_parseable"] = False
    return checks, inspection


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_histogram(values: Iterable[int]) -> dict[int, int]:
    return dict(sorted(collections.Counter(int(value) for value in values).items()))


def _pack_litert_codes(codes: np.ndarray, bits: int) -> bytes:
    """Pack signed deployment codes in LiteRT's two's-complement lane format.

    The public mobile Safetensors stores W2/W4 values with an unsigned offset.
    The official TFLite target does not: it serializes the signed code modulo
    ``2**bits``.  Do not use the source offset representation here.
    """

    values = np.asarray(codes, dtype=np.int8).reshape(-1)
    if bits not in {2, 4, 8}:
        raise RetainedScaleExportError(f"Unsupported LiteRT packing width W{bits}.")
    qmin = -127 if bits == 8 else -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    if values.size and (int(values.min()) < qmin or int(values.max()) > qmax):
        raise RetainedScaleExportError(
            f"Codes exceed the retained W{bits} range [{qmin}, {qmax}]."
        )
    if bits == 8:
        return values.tobytes()
    return _pack_low_bit(values, bits).tobytes()


def _unpack_litert_codes(
    raw: bytes | bytearray | memoryview, bits: int, count: int
) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(raw, dtype=np.int8, count=count).copy()
    return np.asarray(_unpack_low_bit(raw, bits, count), dtype=np.int8)


def _normalize_official_key(source_key: str | None) -> str | None:
    """Convert the packed-source namespace to the text-only HF namespace."""

    if not source_key:
        return None
    normalized = _output_key(str(source_key))
    return str(normalized) if normalized else None


def _expected_frozen_keys() -> set[str]:
    keys = {"lm_head.weight", "model.per_layer_model_projection.weight"}
    for layer in range(35):
        keys.add(f"model.layers.{layer}.per_layer_input_gate.weight")
        keys.add(f"model.layers.{layer}.per_layer_projection.weight")
    return keys


def _scope_report(
    records: list[dict[str, Any]], qparams: MobileQParams
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) != EXPECTED_TARGET_COUNT:
        raise RetainedScaleExportError(
            f"Official target inventory has {len(records)} weights; expected 277."
        )
    source_keys = _canonical_inventory_keys(records, "gemma4_e2b", TARGET_MODEL_TYPE)
    normalized_keys = [_normalize_official_key(key) for key in source_keys]
    missing = [
        int(record["ordinal"])
        for record, key in zip(records, normalized_keys, strict=True)
        if key is None
    ]
    if missing:
        raise RetainedScaleExportError(
            f"Official target records do not normalize to HF keys: {missing[:12]}."
        )
    normalized = [str(key) for key in normalized_keys]
    duplicates = sorted(
        key for key, count in collections.Counter(normalized).items() if count != 1
    )
    if duplicates:
        raise RetainedScaleExportError(
            "Official target key mapping is not one-to-one: "
            + ", ".join(duplicates[:12])
        )

    expected_mutable = set(qparams.trainable_projection_weight_keys())
    frozen_keys = set(normalized) - expected_mutable
    unexpected_contract_keys = expected_mutable - set(normalized)
    expected_frozen = _expected_frozen_keys()
    mutable_records: list[dict[str, Any]] = []
    frozen_records: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for record, source_key, key in zip(records, source_keys, normalized, strict=True):
        item = dict(record)
        item["packed_source_key"] = source_key
        item["hf_weight_key"] = key
        entry = qparams.inventory.get(key)
        mutable = key in expected_mutable
        if mutable:
            if not isinstance(entry, dict):
                raise RetainedScaleExportError(
                    f"Mutable target {key!r} has no retained qparams entry."
                )
            if int(entry["bits"]) != int(record["bits"]):
                raise RetainedScaleExportError(
                    f"Bit width differs for {key}: official W{record['bits']} vs "
                    f"qparams W{entry['bits']}."
                )
            if tuple(int(v) for v in entry["weight_shape"]) != tuple(
                int(v) for v in record["shape"]
            ):
                raise RetainedScaleExportError(
                    f"Shape differs for {key}: official={record['shape']} "
                    f"qparams={entry['weight_shape']}."
                )
            mutable_records.append(item)
        else:
            frozen_records.append(item)
        assignments.append(
            {
                "ordinal": int(record["ordinal"]),
                "buffer": int(record["official_buffer"]),
                "bits": int(record["bits"]),
                "packed_source_key": source_key,
                "hf_weight_key": key,
                "scope": "mutable_projection" if mutable else "frozen_target",
            }
        )

    mutable_histogram = _normalized_histogram(item["bits"] for item in mutable_records)
    frozen_histogram = _normalized_histogram(item["bits"] for item in frozen_records)
    target_histogram = _normalized_histogram(item["bits"] for item in records)
    checks = {
        "official_target_inventory_277": len(records) == EXPECTED_TARGET_COUNT,
        "normalized_key_inventory_unique": len(set(normalized)) == len(normalized),
        "mutable_key_inventory_205": len(expected_mutable) == EXPECTED_MUTABLE_COUNT,
        "mutable_official_bijection_205": len(mutable_records)
        == EXPECTED_MUTABLE_COUNT,
        "mutable_bit_histogram_60w2_145w4": mutable_histogram == EXPECTED_MUTABLE_BITS,
        "frozen_key_inventory_72": frozen_keys == expected_frozen,
        "frozen_official_inventory_72": len(frozen_records) == EXPECTED_FROZEN_COUNT,
        "frozen_bit_histogram_1w2_71w8": frozen_histogram == EXPECTED_FROZEN_BITS,
        "target_bit_histogram_exact": target_histogram == EXPECTED_TARGET_BITS,
        "no_mutable_contract_key_missing_from_target": not unexpected_contract_keys,
    }
    report = {
        "checks": checks,
        "verified": all(checks.values()),
        "target_count": len(records),
        "mutable_count": len(mutable_records),
        "frozen_count": len(frozen_records),
        "target_bit_histogram": target_histogram,
        "mutable_bit_histogram": mutable_histogram,
        "frozen_bit_histogram": frozen_histogram,
        "unexpected_contract_keys": sorted(unexpected_contract_keys),
        "unexpected_frozen_keys": sorted(frozen_keys - expected_frozen),
        "missing_frozen_keys": sorted(expected_frozen - frozen_keys),
        "assignment_sha256": _json_sha256(assignments),
        "assignments": assignments,
    }
    if not report["verified"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RetainedScaleExportError(
            "Official retained-scale mutation scope failed: " + ", ".join(failed)
        )
    return report, mutable_records, frozen_records


def _checkpoint_candidates(weight_key: str) -> tuple[str, ...]:
    values = [weight_key]
    if weight_key.endswith(".weight"):
        values.append(weight_key[: -len(".weight")] + ".linear.weight")
    for prefix in ("base_model.model.", "base_model."):
        values.extend(prefix + value for value in tuple(values))
    return tuple(dict.fromkeys(values))


def _checkpoint_mapping_report(
    checkpoint: SafetensorCheckpoint,
    mutable_records: list[dict[str, Any]],
    qparams: MobileQParams,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    mappings: dict[str, str] = {}
    assignments: list[dict[str, Any]] = []
    for record in mutable_records:
        key = str(record["hf_weight_key"])
        matches = [
            candidate
            for candidate in _checkpoint_candidates(key)
            if candidate in checkpoint.entries
        ]
        if len(matches) != 1:
            raise RetainedScaleExportError(
                f"{label} mapping for {key!r} is ambiguous or missing: {matches}."
            )
        source_key = matches[0]
        entry = checkpoint.entries[source_key]
        shape = tuple(int(value) for value in entry["shape"])
        expected_shape = tuple(
            int(value) for value in qparams.inventory[key]["weight_shape"]
        )
        dtype = str(entry["dtype"]).upper()
        if shape != expected_shape:
            raise RetainedScaleExportError(
                f"{label} shape for {source_key!r} is {shape}, expected {expected_shape}; "
                "retained export refuses implicit transposition."
            )
        if dtype not in {"BF16", "F16", "F32", "F64"}:
            raise RetainedScaleExportError(
                f"{label} tensor {source_key!r} is not floating point ({dtype})."
            )
        mappings[key] = source_key
        assignments.append(
            {
                "hf_weight_key": key,
                "checkpoint_key": source_key,
                "shape": list(shape),
                "dtype": dtype,
                "shard": str(checkpoint.key_to_shard[source_key]),
            }
        )
    report = {
        "label": label,
        "verified": len(mappings) == EXPECTED_MUTABLE_COUNT,
        "mapping_count": len(mappings),
        "mapping_sha256": _json_sha256(assignments),
        "assignments": assignments,
        "checkpoint": checkpoint.describe(),
    }
    if not report["verified"]:
        raise RetainedScaleExportError(
            f"{label} resolves {len(mappings)} projections; expected 205."
        )
    return report, mappings


def _adapter_file_records(directory: Path) -> list[dict[str, Any]]:
    files = sorted(item for item in directory.glob("adapter*") if item.is_file())
    return [
        {
            "path": item.name,
            "size": int(item.stat().st_size),
            "sha256": _sha256_file(item),
        }
        for item in files
    ]


def _normalized_file_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "path": str(item.get("path") or ""),
                "size": int(item.get("size", -1) or -1),
                "sha256": str(item.get("sha256") or "").lower(),
            }
        )
    return sorted(result, key=lambda item: item["path"])


def _normalized_lora_module(value: str) -> str:
    """Normalize PEFT's wrapper namespace to the dense HF module namespace."""

    normalized = str(value)
    while normalized.startswith("base_model."):
        normalized = normalized[len("base_model.") :]
    while normalized.startswith("model.model."):
        normalized = normalized[len("model.") :]
    normalized = normalized.removesuffix(".linear")
    return normalized


def _adapter_mapping_report(
    adapter_checkpoint: Path,
    adapter: SafetensorCheckpoint,
    mutable_records: list[dict[str, Any]],
    qparams: MobileQParams,
    training_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]], float]:
    """Resolve the exact 205 PEFT A/B pairs used by the dense merge.

    Exotic PEFT variants intentionally fail closed.  The reconstruction below
    mirrors ordinary LoRA ``base += (B @ A) * alpha/r`` only; accepting DoRA,
    rank/alpha patterns, RS-LoRA, saved bias, or modules-to-save would make the
    merged-weight proof incomplete.
    """

    config_path = adapter_checkpoint / "adapter_config.json"
    if not config_path.is_file():
        raise RetainedScaleExportError(
            "Best-Golden adapter is missing adapter_config.json."
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetainedScaleExportError(
            f"Could not read adapter_config.json: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RetainedScaleExportError("adapter_config.json must be a JSON object.")

    lora_config = (
        training_config.get("lora")
        if isinstance(training_config.get("lora"), dict)
        else {}
    )
    try:
        rank = int(payload.get("r", 0) or 0)
        alpha = float(payload.get("lora_alpha", float("nan")))
        expected_rank = int(lora_config.get("r", 0) or 0)
        expected_alpha = float(lora_config.get("alpha", float("nan")))
    except (TypeError, ValueError) as exc:
        raise RetainedScaleExportError("LoRA rank/alpha metadata is invalid.") from exc
    scaling = alpha / rank if rank > 0 and math.isfinite(alpha) else float("nan")
    variant_checks = {
        "peft_type_lora": str(payload.get("peft_type") or "").upper() == "LORA",
        "rank_positive": rank > 0,
        "rank_matches_training_config": rank == expected_rank,
        "alpha_finite": math.isfinite(alpha),
        "alpha_matches_training_config": alpha == expected_alpha,
        "fan_in_fan_out_false": payload.get("fan_in_fan_out", False) is False,
        "use_rslora_false": payload.get("use_rslora", False) is False,
        "use_dora_false": payload.get("use_dora", False) is False,
        "rank_pattern_empty": not payload.get("rank_pattern"),
        "alpha_pattern_empty": not payload.get("alpha_pattern"),
        "bias_none": str(payload.get("bias") or "none").lower() == "none",
        "lora_bias_false": payload.get("lora_bias", False) is False,
        "modules_to_save_empty": not payload.get("modules_to_save"),
        "scaling_finite_positive": math.isfinite(scaling) and scaling > 0,
    }
    if not all(variant_checks.values()):
        failed = [name for name, passed in variant_checks.items() if not passed]
        raise RetainedScaleExportError(
            "Adapter uses unsupported or mismatched LoRA semantics: "
            + ", ".join(failed)
        )

    by_module: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: {"A": [], "B": []}
    )
    unrecognized: list[str] = []
    for tensor_key in sorted(adapter.entries):
        match = _LORA_KEY.fullmatch(tensor_key)
        if match is None:
            unrecognized.append(tensor_key)
            continue
        module = _normalized_lora_module(match.group("module"))
        by_module[module][match.group("side")].append(tensor_key)

    mappings: dict[str, dict[str, str]] = {}
    assignments: list[dict[str, Any]] = []
    used: set[str] = set()
    for record in mutable_records:
        weight_key = str(record["hf_weight_key"])
        module = weight_key[: -len(".weight")]
        pair = by_module.get(module, {"A": [], "B": []})
        if len(pair["A"]) != 1 or len(pair["B"]) != 1:
            raise RetainedScaleExportError(
                f"Adapter A/B mapping for {weight_key!r} is ambiguous or missing: {pair}."
            )
        a_key, b_key = pair["A"][0], pair["B"][0]
        a_entry = adapter.entries[a_key]
        b_entry = adapter.entries[b_key]
        weight_shape = tuple(
            int(value) for value in qparams.inventory[weight_key]["weight_shape"]
        )
        a_shape = tuple(int(value) for value in a_entry["shape"])
        b_shape = tuple(int(value) for value in b_entry["shape"])
        a_dtype = str(a_entry["dtype"]).upper()
        b_dtype = str(b_entry["dtype"]).upper()
        if a_shape != (rank, weight_shape[1]) or b_shape != (weight_shape[0], rank):
            raise RetainedScaleExportError(
                f"Adapter shapes for {weight_key!r} are A={a_shape}, B={b_shape}; "
                f"expected {(rank, weight_shape[1])} and {(weight_shape[0], rank)}."
            )
        if a_dtype not in _FLOAT_DTYPES or b_dtype not in _FLOAT_DTYPES:
            raise RetainedScaleExportError(
                f"Adapter factors for {weight_key!r} are not floating point."
            )
        if a_dtype != b_dtype:
            raise RetainedScaleExportError(
                f"Adapter factor dtypes differ for {weight_key!r}: {a_dtype}/{b_dtype}."
            )
        mappings[weight_key] = {"a": a_key, "b": b_key}
        used.update((a_key, b_key))
        assignments.append(
            {
                "hf_weight_key": weight_key,
                "lora_a_key": a_key,
                "lora_b_key": b_key,
                "lora_a_shape": list(a_shape),
                "lora_b_shape": list(b_shape),
                "dtype": a_dtype,
            }
        )

    checks = {
        **variant_checks,
        "exact_205_lora_pairs": len(mappings) == EXPECTED_MUTABLE_COUNT,
        "exact_410_adapter_tensors": len(adapter.entries) == EXPECTED_MUTABLE_COUNT * 2,
        "all_adapter_tensors_recognized": not unrecognized,
        "all_adapter_tensors_consumed": used == set(adapter.entries),
    }
    report = {
        "adapter_config": str(config_path),
        "adapter_config_sha256": _sha256_file(config_path),
        "rank": rank,
        "alpha": alpha,
        "scaling": scaling,
        "checks": checks,
        "verified": all(checks.values()),
        "mapping_count": len(mappings),
        "mapping_sha256": _json_sha256(assignments),
        "assignments": assignments,
        "unrecognized_tensor_keys": unrecognized,
        "checkpoint": adapter.describe(),
    }
    if not report["verified"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RetainedScaleExportError(
            "Adapter tensor contract failed: " + ", ".join(failed)
        )
    return report, mappings, scaling


def _best_adapter_provenance_report(
    adapter_checkpoint: Path,
    training_config: Path,
    *,
    qparams: MobileQParams,
    seed_report: dict[str, Any],
) -> dict[str, Any]:
    metadata_path = adapter_checkpoint / "training_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = loaded if isinstance(loaded, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            metadata = {}
    qat = metadata.get("qat") if isinstance(metadata.get("qat"), dict) else {}
    spec = qat.get("spec") if isinstance(qat.get("spec"), dict) else {}
    retained = (
        qat.get("retained_qparams")
        if isinstance(qat.get("retained_qparams"), dict)
        else {}
    )
    bindings = (
        qat.get("retained_qparams_bindings")
        if isinstance(qat.get("retained_qparams_bindings"), dict)
        else {}
    )
    bound_keys = {
        str(item.get("weight_key"))
        for item in bindings.values()
        if isinstance(item, dict)
    }
    expected_keys = set(qparams.trainable_projection_weight_keys())
    numeric = (
        metadata.get("numeric_preflight")
        if isinstance(metadata.get("numeric_preflight"), dict)
        else {}
    )
    greedy = (
        numeric.get("greedy_generation")
        if isinstance(numeric.get("greedy_generation"), dict)
        else {}
    )
    golden = (
        metadata.get("best_golden_eval")
        if isinstance(metadata.get("best_golden_eval"), dict)
        else {}
    )
    config = load_yaml(training_config)
    golden_selection = _golden_selection_binding(metadata, config)
    actual_adapter_files = _adapter_file_records(adapter_checkpoint)
    manifests = metadata.get("adapter_checkpoints")
    manifests = manifests if isinstance(manifests, list) else []
    adapter_hash_match = any(
        isinstance(item, dict)
        and item.get("role") == "best_golden"
        and _checkpoint_manifest_matches_adapter(
            item.get("files"),
            adapter_path=adapter_checkpoint,
            actual_adapter_files=actual_adapter_files,
        )
        for item in manifests
    )
    seed_metadata = (
        metadata.get("mobile_training_seed")
        if isinstance(metadata.get("mobile_training_seed"), dict)
        else {}
    )
    checks = {
        "metadata_present": metadata_path.is_file(),
        "metadata_v4_or_newer": int(metadata.get("training_metadata_version", 0) or 0)
        >= 4,
        "checkpoint_role_best_golden": metadata.get("checkpoint_role") == "best_golden",
        "best_golden_v5_4_selected": golden_selection["verified"],
        "golden_selection_binding_matches": golden_selection["verified"],
        "adapter_hashes_self_bound": bool(actual_adapter_files and adapter_hash_match),
        "training_config_hash_matches": str(
            metadata.get("training_config_sha256") or ""
        ).lower()
        == _sha256_file(training_config),
        "numeric_preflight_passed": numeric.get("passed") is True,
        "greedy_preflight_passed": greedy.get("passed") is True,
        "retained_scale_mode": spec.get("scale_mode") == "retained_mobile",
        "fixed_weight_scales": spec.get("fixed_scale_required") is True,
        "fixed_activation_scales": spec.get("fixed_activation_scale_required") is True,
        "effective_lora_only": spec.get("effective_lora_only") is True,
        "clipped_ste": spec.get("ste_gradient") == "clipped",
        "quantize_embeddings_false": spec.get("quantize_embeddings") is False,
        "expected_effective_lora_205": int(
            spec.get("expected_effective_lora_modules", 0) or 0
        )
        == EXPECTED_MUTABLE_COUNT,
        "live_effective_lora_205": int(qat.get("wrapped_effective_lora_count", 0) or 0)
        == EXPECTED_MUTABLE_COUNT,
        "retained_qparams_bindings_205": int(
            qat.get("retained_qparams_binding_count", 0) or 0
        )
        == EXPECTED_MUTABLE_COUNT
        and bound_keys == expected_keys,
        "retained_qparams_contract_hash_matches": str(
            retained.get("contract_sha256") or ""
        ).lower()
        == qparams.contract_sha256.lower(),
        "retained_scale_sidecar_hash_matches": str(
            retained.get("scale_storage_sha256") or ""
        ).lower()
        == qparams.scale_storage_sha256.lower(),
        "mobile_seed_manifest_hash_matches": seed_metadata.get("verified") is True
        and str(seed_metadata.get("manifest_sha256") or "").lower()
        == str(seed_report.get("manifest_sha256") or "").lower()
        and str(seed_metadata.get("transformation_plan_sha256") or "").lower()
        == str(seed_report.get("transformation_plan_sha256") or "").lower(),
        "legacy_metadata_rejected": spec.get("scale_mode") == "retained_mobile"
        and int(qat.get("retained_qparams_binding_count", 0) or 0)
        == EXPECTED_MUTABLE_COUNT,
    }
    return {
        "path": str(metadata_path),
        "sha256": _sha256_file(metadata_path) if metadata_path.is_file() else None,
        "checks": checks,
        "verified": all(checks.values()),
        "adapter_files": actual_adapter_files,
        "golden": golden,
        "golden_selection": golden_selection,
        "bound_key_sha256": _json_sha256(sorted(bound_keys)),
    }


def _merged_golden_selection_matches(
    merge_provenance: dict[str, Any], adapter_selection: Any
) -> bool:
    """Require exact new binding, with a narrow verified Golden100 fallback.

    Historical ordinary-Golden100 merge manifests predate the embedded
    ``golden_selection`` report. They remain acceptable only because the merge
    already binds the same adapter bytes and training-config hash and carries a
    fully verified training-run provenance report. Repeated Golden32 has no
    legacy fallback: its unique-source metric/cohort binding must match exactly.
    """

    if not isinstance(adapter_selection, dict):
        return False
    merge_metadata = (
        merge_provenance.get("metadata")
        if isinstance(merge_provenance.get("metadata"), dict)
        else {}
    )
    run_metadata = (
        merge_metadata.get("training_run_metadata")
        if isinstance(merge_metadata.get("training_run_metadata"), dict)
        else {}
    )
    merged_selection = run_metadata.get("golden_selection")
    if isinstance(merged_selection, dict):
        return merged_selection == adapter_selection
    selected = (
        adapter_selection.get("selected")
        if isinstance(adapter_selection.get("selected"), dict)
        else {}
    )
    checks = (
        run_metadata.get("checks")
        if isinstance(run_metadata.get("checks"), dict)
        else {}
    )
    return bool(
        selected.get("metric") == "generation_reward_v5_4_avg"
        and adapter_selection.get("verified") is True
        and run_metadata.get("verified") is True
        and checks.get("best_golden_v5_4_selected") is True
        and checks.get("portable_launcher_artifacts_bound") is True
    )


def _config_report(
    config_path: Path,
    *,
    seed_manifest: Path,
    qparams_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    qat = config.get("qat") if isinstance(config.get("qat"), dict) else {}
    spec = QATSpec.from_config(config)
    configured_seed = resolve_path(model.get("mobile_training_seed_manifest"), ROOT)
    configured_qparams_value = qat.get("mobile_qparams_contract") or model.get(
        "mobile_qparams_contract"
    )
    configured_qparams = resolve_path(configured_qparams_value, ROOT)
    checks = {
        "official_mobile_model_id": model.get("model_id") == OFFICIAL_MOBILE_MODEL_ID,
        "model_dtype_bfloat16": str(model.get("dtype") or "").strip().lower()
        in {"bfloat16", "bf16"},
        "qat_lora_sft": config.get("training", {}).get("method") == "qat_lora_sft",
        "qat_enabled": qat.get("enabled") is True,
        "retained_scale_mode": spec.scale_mode == "retained_mobile",
        "fixed_weight_scales": spec.fixed_scale_required,
        "fixed_activation_scales": spec.fixed_activation_scale_required,
        "effective_lora_only": spec.effective_lora_only,
        "effective_merged_weight": spec.effective_merged_weight,
        "clipped_ste": spec.ste_gradient == "clipped",
        "quantize_embeddings_false": spec.quantize_embeddings is False,
        "expected_effective_lora_205": spec.expected_effective_lora_modules
        == EXPECTED_MUTABLE_COUNT,
        "zero_lora_dropout": float(config.get("lora", {}).get("dropout", -1.0)) == 0.0,
        "seed_manifest_cli_matches_config": configured_seed == seed_manifest,
        "qparams_cli_matches_config": configured_qparams == qparams_path,
    }
    return {
        "path": str(config_path),
        "sha256": _sha256_file(config_path),
        "checks": checks,
        "verified": all(checks.values()),
        "qat_spec": spec.to_dict(),
    }, config


def _quantization_vectors(tensor: Any) -> dict[str, Any] | None:
    quantization = tensor.Quantization()
    if quantization is None:
        return None

    def vector(name: str, dtype: Any) -> np.ndarray:
        as_numpy = getattr(quantization, f"{name}AsNumpy", None)
        raw = as_numpy() if callable(as_numpy) else None
        if isinstance(raw, np.ndarray):
            return np.asarray(raw, dtype=dtype).reshape(-1)
        length = int(getattr(quantization, f"{name}Length")() or 0)
        getter = getattr(quantization, name)
        return np.asarray([getter(index) for index in range(length)], dtype=dtype)

    scales = vector("Scale", np.float32)
    zero_points = vector("ZeroPoint", np.int64)
    minimum = vector("Min", np.float32)
    maximum = vector("Max", np.float32)
    if not any(len(value) for value in (scales, zero_points, minimum, maximum)):
        return None
    return {
        "scales_hex": np.asarray(scales, dtype="<f4").tobytes().hex(),
        "zero_points_hex": np.asarray(zero_points, dtype="<i8").tobytes().hex(),
        "min_hex": np.asarray(minimum, dtype="<f4").tobytes().hex(),
        "max_hex": np.asarray(maximum, dtype="<f4").tobytes().hex(),
        "quantized_dimension": int(quantization.QuantizedDimension() or 0),
    }


def _all_quantization_digest(section_bytes: bytes | bytearray) -> dict[str, Any]:
    model = _schema_model(section_bytes)
    records: list[dict[str, Any]] = []
    for subgraph_index in range(int(model.SubgraphsLength() or 0)):
        subgraph = model.Subgraphs(subgraph_index)
        for tensor_index in range(int(subgraph.TensorsLength() or 0)):
            tensor = subgraph.Tensors(tensor_index)
            quantization = _quantization_vectors(tensor)
            if quantization is None:
                continue
            records.append(
                {
                    "subgraph": subgraph_index,
                    "tensor": tensor_index,
                    "type": int(tensor.Type()),
                    "quantization": quantization,
                }
            )
    return {"tensor_count": len(records), "sha256": _json_sha256(records)}


def _weight_qparams_report(
    section_bytes: bytes | bytearray,
    all_records: list[dict[str, Any]],
    mutable_records: list[dict[str, Any]],
    qparams: MobileQParams,
) -> dict[str, Any]:
    model = _schema_model(section_bytes)
    all_groups = _official_weight_alias_groups(model, all_records)
    mutable_groups = _official_weight_alias_groups(model, mutable_records)
    mutable_by_buffer = {
        int(group["buffer_index"]): record
        for record, group in zip(mutable_records, mutable_groups, strict=True)
    }
    rows: list[dict[str, Any]] = []
    contract_exact = 0
    for record, group in zip(all_records, all_groups, strict=True):
        tensor = group["aliases"][0]["tensor"]
        quantization = _quantization_vectors(tensor)
        if quantization is None:
            raise RetainedScaleExportError(
                f"Official weight buffer {group['buffer_index']} has no qparams."
            )
        key = record.get("hf_weight_key")
        contract_match: bool | None = None
        if int(group["buffer_index"]) in mutable_by_buffer:
            key = str(mutable_by_buffer[int(group["buffer_index"])]["hf_weight_key"])
            entry = qparams.inventory[key]
            scale = qparams.load_scale(
                key,
                weight_shape=tuple(int(value) for value in entry["weight_shape"]),
                bits=int(entry["bits"]),
                group_size=entry.get("group_size"),
            )
            expected = np.asarray(scale, dtype="<f4").reshape(-1).tobytes().hex()
            contract_match = quantization["scales_hex"] == expected
            contract_exact += int(contract_match)
        rows.append(
            {
                "buffer": int(group["buffer_index"]),
                "key": key,
                "alias_count": len(group["aliases"]),
                "quantization": quantization,
                "retained_scale_contract_exact": contract_match,
            }
        )
    return {
        "weight_count": len(rows),
        "mutable_retained_scale_exact_count": contract_exact,
        "mutable_retained_scales_exact": contract_exact == EXPECTED_MUTABLE_COUNT,
        "alias_summary": _weight_alias_summary(all_groups),
        "sha256": _json_sha256(rows),
    }


def _activation_a8_report(
    section_bytes: bytes | bytearray,
    mutable_records: list[dict[str, Any]],
    qparams: MobileQParams,
) -> dict[str, Any]:
    model = _schema_model(section_bytes)
    groups = _official_weight_alias_groups(model, mutable_records)
    alias_to_key: dict[tuple[int, int], str] = {}
    for record, group in zip(mutable_records, groups, strict=True):
        key = str(record["hf_weight_key"])
        for alias in group["aliases"]:
            location = (
                int(alias["subgraph_index"]),
                int(alias["tensor_index"]),
            )
            if location in alias_to_key:
                raise RetainedScaleExportError(
                    f"Mutable weight alias {location} maps to more than one key."
                )
            alias_to_key[location] = key

    rows: list[dict[str, Any]] = []
    contract_exact = 0
    edge_locations: list[tuple[int, int, int]] = []
    alias_edge_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    for subgraph_index in range(int(model.SubgraphsLength() or 0)):
        subgraph = model.Subgraphs(subgraph_index)
        for operator_index in range(int(subgraph.OperatorsLength() or 0)):
            operator = subgraph.Operators(operator_index)
            inputs = _vector(operator, "InputsLength", "Inputs")
            if len(inputs) < 2 or int(inputs[1]) < 0:
                continue
            alias_location = (subgraph_index, int(inputs[1]))
            key = alias_to_key.get(alias_location)
            if key is None:
                continue
            code = model.OperatorCodes(int(operator.OpcodeIndex()))
            builtin = int(code.BuiltinCode())
            if builtin != _FULLY_CONNECTED:
                raise RetainedScaleExportError(
                    f"Mutable projection alias {alias_location} is not an FC edge."
                )
            outputs = _vector(operator, "OutputsLength", "Outputs")
            if int(inputs[0]) < 0 or not outputs or int(outputs[0]) < 0:
                raise RetainedScaleExportError(
                    f"Mutable FC edge {subgraph_index}/{operator_index} lacks IO tensors."
                )
            alias_edge_counts[alias_location] += 1
            edge_locations.append((subgraph_index, operator_index, int(inputs[1])))
            for role, tensor_index in (
                ("input", int(inputs[0])),
                ("output", int(outputs[0])),
            ):
                tensor = subgraph.Tensors(tensor_index)
                quantization = _quantization_vectors(tensor)
                expected_hex = str(
                    qparams.inventory[key].get(f"{role}_activation_scale_f32_le_hex")
                    or ""
                ).lower()
                observed_hex = quantization["scales_hex"] if quantization else ""
                exact = bool(
                    int(tensor.Type()) == 9
                    and len(expected_hex) == 8
                    and observed_hex == expected_hex
                )
                contract_exact += int(exact)
                rows.append(
                    {
                        "key": key,
                        "role": role,
                        "subgraph": subgraph_index,
                        "operator": operator_index,
                        "weight_tensor": int(inputs[1]),
                        "tensor": tensor_index,
                        "type": int(tensor.Type()),
                        "quantization": quantization,
                        "retained_scale_contract_exact": exact,
                    }
                )
    aliases_exactly_once = bool(
        set(alias_edge_counts) == set(alias_to_key)
        and all(count == 1 for count in alias_edge_counts.values())
    )
    alias_count = len(alias_to_key)
    edge_count = len(edge_locations)
    checks = {
        "mutable_weight_alias_count_610": alias_count
        == EXPECTED_MUTABLE_FC_ALIAS_EDGES,
        "mutable_fc_edge_count_610": edge_count == EXPECTED_MUTABLE_FC_ALIAS_EDGES,
        "every_alias_has_exactly_one_fc_edge": aliases_exactly_once,
        "a8_record_count_1220": len(rows) == EXPECTED_MUTABLE_A8_RECORDS,
        "all_a8_records_match_contract": contract_exact == EXPECTED_MUTABLE_A8_RECORDS,
    }
    return {
        "checks": checks,
        "verified": all(checks.values()),
        "weight_alias_count": alias_count,
        "fc_edge_count": edge_count,
        "record_count": len(rows),
        "retained_scale_exact_count": contract_exact,
        "retained_a8_contract_exact": all(checks.values()),
        "sha256": _json_sha256(rows),
    }


def _quantize_projection(
    weight: np.ndarray,
    scale: np.ndarray,
    *,
    bits: int,
    working_set_bytes: int,
) -> tuple[bytes, dict[str, Any]]:
    values = np.asarray(weight, dtype=np.float32)
    scales = np.asarray(scale, dtype=np.float32)
    if values.ndim != 2 or scales.shape != (values.shape[0], 1):
        raise RetainedScaleExportError(
            f"Retained projection expects weight [rows, columns] and scale [rows, 1], "
            f"got {values.shape} and {scales.shape}."
        )
    if (
        not np.isfinite(values).all()
        or not np.isfinite(scales).all()
        or not (scales > 0).all()
    ):
        raise RetainedScaleExportError(
            "Projection weights/scales are non-finite or non-positive."
        )
    if bits not in {2, 4}:
        raise RetainedScaleExportError(
            f"Only trained W2/W4 projections may be exported, got W{bits}."
        )
    values_per_byte = 8 // bits
    if values.shape[1] % values_per_byte:
        raise RetainedScaleExportError(
            f"Projection columns {values.shape[1]} are not pack-aligned for W{bits}."
        )
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    rows_per_chunk = max(
        1,
        int(working_set_bytes) // max(int(values.shape[1]) * (4 + 4 + 1), 1),
    )
    packed_chunks: list[bytes] = []
    histogram: collections.Counter[int] = collections.Counter()
    clipped_low = 0
    clipped_high = 0
    for row_start in range(0, values.shape[0], rows_per_chunk):
        chunk = values[row_start : row_start + rows_per_chunk]
        chunk_scales = scales[row_start : row_start + rows_per_chunk]
        ratio = chunk / chunk_scales
        rounded = np.rint(ratio)
        clipped_low += int(np.count_nonzero(rounded < qmin))
        clipped_high += int(np.count_nonzero(rounded > qmax))
        codes = np.clip(rounded, qmin, qmax).astype(np.int8, copy=False)
        unique, counts = np.unique(codes, return_counts=True)
        histogram.update(
            {int(code): int(count) for code, count in zip(unique, counts, strict=True)}
        )
        packed_chunks.append(_pack_litert_codes(codes, bits))
        del ratio, rounded, codes
    raw = b"".join(packed_chunks)
    expected_size = (int(values.size) * bits + 7) // 8
    if len(raw) != expected_size:
        raise RetainedScaleExportError(
            f"Packed projection has {len(raw)} bytes, expected {expected_size}."
        )
    decoded = _unpack_litert_codes(raw, bits, int(values.size))
    roundtrip = _pack_litert_codes(decoded, bits) == raw
    return raw, {
        "bits": bits,
        "shape": list(values.shape),
        "value_count": int(values.size),
        "packed_bytes": len(raw),
        "packed_sha256": _sha256_bytes(raw),
        "qmin": qmin,
        "qmax": qmax,
        "qmin_count": int(histogram.get(qmin, 0)),
        "qmax_count": int(histogram.get(qmax, 0)),
        "zero_count": int(histogram.get(0, 0)),
        "clipped_low_count": clipped_low,
        "clipped_high_count": clipped_high,
        "code_histogram": dict(sorted(histogram.items())),
        "litert_pack_roundtrip": roundtrip,
        "finite": True,
    }


def _torch_dtype(dtype: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - execution environment only
        raise RetainedScaleExportError(
            "Base+LoRA merge verification requires PyTorch."
        ) from exc
    values = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
    }
    try:
        return values[str(dtype).upper()]
    except KeyError as exc:
        raise RetainedScaleExportError(
            f"Unsupported floating checkpoint dtype {dtype!r}."
        ) from exc


def _base_lora_projection_parity(
    candidate_weight: np.ndarray,
    base_weight: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    retained_scale: np.ndarray,
    *,
    candidate_dtype: str,
    base_dtype: str,
    adapter_dtype: str,
    scaling: float,
    bits: int,
    candidate_raw: bytes,
    working_set_bytes: int,
) -> dict[str, Any]:
    """Reconstruct PEFT's ordinary LoRA merge in bounded row chunks.

    The factors are tiny compared with the dense projection.  Keeping A/B in
    memory while materializing only a row chunk of ``B @ A`` avoids another
    full dense delta allocation.  Both numerical closeness and exact retained-
    scale deployment codes are required; the latter is the authoritative
    serialization gate.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - execution environment only
        raise RetainedScaleExportError(
            "Base+LoRA merge verification requires PyTorch."
        ) from exc
    candidate = np.asarray(candidate_weight, dtype=np.float32)
    base = np.asarray(base_weight, dtype=np.float32)
    a = np.asarray(lora_a, dtype=np.float32)
    b = np.asarray(lora_b, dtype=np.float32)
    scales = np.asarray(retained_scale, dtype=np.float32)
    if candidate.shape != base.shape or a.ndim != 2 or b.ndim != 2:
        raise RetainedScaleExportError(
            "Base+LoRA parity received incompatible tensors."
        )
    if (
        a.shape[0] != b.shape[1]
        or a.shape[1] != base.shape[1]
        or b.shape[0] != base.shape[0]
    ):
        raise RetainedScaleExportError(
            f"LoRA factors {a.shape}/{b.shape} do not produce {base.shape}."
        )
    if scales.shape != (base.shape[0], 1):
        raise RetainedScaleExportError("Base+LoRA parity retained-scale shape differs.")

    candidate_dtype = str(candidate_dtype).upper()
    base_dtype = str(base_dtype).upper()
    adapter_dtype = str(adapter_dtype).upper()
    base_torch_dtype = _torch_dtype(base_dtype)
    adapter_torch_dtype = _torch_dtype(adapter_dtype)
    # PEFT casts low-precision factors to FP32 for CPU matrix multiplication,
    # then casts the delta back before its in-place merge into the base weight.
    a_tensor = torch.from_numpy(a).to(adapter_torch_dtype)
    b_tensor = torch.from_numpy(b).to(adapter_torch_dtype)
    low_precision_adapter = adapter_dtype in {"BF16", "F16"}
    a_compute = a_tensor.float() if low_precision_adapter else a_tensor
    columns = int(base.shape[1])
    rows_per_chunk = max(
        1,
        int(working_set_bytes) // max(columns * (4 * 5 + 2), 1),
    )
    reconstructed_raw_chunks: list[bytes] = []
    numerical_exact = True
    numerical_close = True
    max_abs_error = 0.0
    base_norm_squared = 0.0
    reconstructed_delta_norm_squared = 0.0
    observed_delta_norm_squared = 0.0
    for row_start in range(0, int(base.shape[0]), rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, int(base.shape[0]))
        base_chunk = torch.from_numpy(np.ascontiguousarray(base[row_start:row_end])).to(
            base_torch_dtype
        )
        b_chunk = b_tensor[row_start:row_end]
        b_compute = b_chunk.float() if low_precision_adapter else b_chunk
        delta = (b_compute @ a_compute) * float(scaling)
        if low_precision_adapter:
            delta = delta.to(adapter_torch_dtype)
        reconstructed = base_chunk.clone()
        # PEFT merges with ``base_layer.weight.data += delta_weight``; the
        # in-place operation casts the result back to the base tensor dtype.
        reconstructed.add_(delta)
        reconstructed_f32 = np.ascontiguousarray(
            reconstructed.detach().float().cpu().numpy()
        )
        candidate_chunk = candidate[row_start:row_end]
        exact = bool(np.array_equal(candidate_chunk, reconstructed_f32))
        numerical_exact = numerical_exact and exact
        if candidate_chunk.size:
            difference = np.abs(candidate_chunk - reconstructed_f32)
            max_abs_error = max(max_abs_error, float(np.max(difference)))
        else:
            difference = np.empty(0, dtype=np.float32)
        rtol = {"BF16": 8.0e-3, "F16": 1.0e-3}.get(base_dtype, 1.0e-6)
        numerical_close = numerical_close and bool(
            np.allclose(candidate_chunk, reconstructed_f32, rtol=rtol, atol=1.0e-6)
        )
        base_f32 = base_chunk.detach().float()
        reconstructed_delta = reconstructed.detach().float() - base_f32
        observed_delta = (
            torch.from_numpy(np.ascontiguousarray(candidate_chunk)).float() - base_f32
        )
        base_norm_squared += float(torch.sum(base_f32 * base_f32).item())
        reconstructed_delta_norm_squared += float(
            torch.sum(reconstructed_delta * reconstructed_delta).item()
        )
        observed_delta_norm_squared += float(
            torch.sum(observed_delta * observed_delta).item()
        )
        reconstructed_raw, _ = _quantize_projection(
            reconstructed_f32,
            scales[row_start:row_end],
            bits=bits,
            working_set_bytes=working_set_bytes,
        )
        reconstructed_raw_chunks.append(reconstructed_raw)
        del (
            base_chunk,
            b_chunk,
            b_compute,
            delta,
            reconstructed,
            reconstructed_f32,
            difference,
            base_f32,
            reconstructed_delta,
            observed_delta,
            reconstructed_raw,
        )
    reconstructed_raw = b"".join(reconstructed_raw_chunks)
    base_norm = math.sqrt(max(base_norm_squared, 0.0))
    reconstructed_delta_norm = math.sqrt(max(reconstructed_delta_norm_squared, 0.0))
    observed_delta_norm = math.sqrt(max(observed_delta_norm_squared, 0.0))
    norms_finite = all(
        math.isfinite(value)
        for value in (base_norm, reconstructed_delta_norm, observed_delta_norm)
    )
    return {
        "candidate_dtype": candidate_dtype,
        "base_dtype": base_dtype,
        "adapter_dtype": adapter_dtype,
        "candidate_base_dtype_match": candidate_dtype == base_dtype,
        "candidate_base_bfloat16": candidate_dtype == base_dtype == "BF16",
        "numerical_exact": numerical_exact,
        "numerical_close": numerical_close,
        "max_abs_error": max_abs_error,
        "retained_scale_code_exact": reconstructed_raw == candidate_raw,
        "reconstructed_code_sha256": _sha256_bytes(reconstructed_raw),
        "candidate_code_sha256": _sha256_bytes(candidate_raw),
        "base_l2_norm": base_norm,
        "reconstructed_delta_l2_norm": reconstructed_delta_norm,
        "observed_merged_minus_base_l2_norm": observed_delta_norm,
        "delta_to_base_l2_ratio": (
            reconstructed_delta_norm / base_norm if base_norm > 0 else None
        ),
        "norms_finite": norms_finite,
        "delta_nonzero": reconstructed_delta_norm > 0.0,
    }


def _quantize_and_patch(
    official_section: bytes,
    *,
    checkpoint: SafetensorCheckpoint,
    mappings: dict[str, str],
    mutable_records: list[dict[str, Any]],
    qparams: MobileQParams,
    label: str,
    working_set_bytes: int,
    base_checkpoint: SafetensorCheckpoint | None = None,
    base_mappings: dict[str, str] | None = None,
    adapter_checkpoint: SafetensorCheckpoint | None = None,
    adapter_mappings: dict[str, dict[str, str]] | None = None,
    lora_scaling: float | None = None,
) -> tuple[bytearray, dict[str, Any]]:
    mutable = bytearray(official_section)
    official_model = _schema_model(official_section)
    model = _schema_model(mutable)
    groups = _official_weight_alias_groups(model, mutable_records)
    official_groups = _official_weight_alias_groups(official_model, mutable_records)
    telemetry: list[dict[str, Any]] = []
    parity_inputs = (
        base_checkpoint,
        base_mappings,
        adapter_checkpoint,
        adapter_mappings,
        lora_scaling,
    )
    parity_enabled = all(value is not None for value in parity_inputs)
    if parity_enabled != any(value is not None for value in parity_inputs):
        raise RetainedScaleExportError(
            "Base+LoRA parity inputs must be supplied together or omitted together."
        )
    parity_rows: list[dict[str, Any]] = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(checkpoint)
        if parity_enabled:
            stack.enter_context(base_checkpoint)
            stack.enter_context(adapter_checkpoint)
        for index, (record, group) in enumerate(
            zip(mutable_records, groups, strict=True)
        ):
            key = str(record["hf_weight_key"])
            entry = qparams.inventory[key]
            weight = checkpoint.load_float32(mappings[key])
            scale_tensor = qparams.load_scale(
                key,
                weight_shape=tuple(int(value) for value in entry["weight_shape"]),
                bits=int(entry["bits"]),
                group_size=entry.get("group_size"),
            )
            scale = np.ascontiguousarray(scale_tensor.detach().float().cpu().numpy())
            raw, item = _quantize_projection(
                weight,
                scale,
                bits=int(entry["bits"]),
                working_set_bytes=working_set_bytes,
            )
            if parity_enabled:
                base_key = base_mappings[key]
                adapter_pair = adapter_mappings[key]
                base_weight = base_checkpoint.load_float32(base_key)
                lora_a = adapter_checkpoint.load_float32(adapter_pair["a"])
                lora_b = adapter_checkpoint.load_float32(adapter_pair["b"])
                parity = _base_lora_projection_parity(
                    weight,
                    base_weight,
                    lora_a,
                    lora_b,
                    scale,
                    candidate_dtype=str(
                        checkpoint.entries[mappings[key]]["dtype"]
                    ).upper(),
                    base_dtype=str(base_checkpoint.entries[base_key]["dtype"]).upper(),
                    adapter_dtype=str(
                        adapter_checkpoint.entries[adapter_pair["a"]]["dtype"]
                    ).upper(),
                    scaling=float(lora_scaling),
                    bits=int(entry["bits"]),
                    candidate_raw=raw,
                    working_set_bytes=working_set_bytes,
                )
                parity_rows.append({"hf_weight_key": key, **parity})
                del base_weight, lora_a, lora_b
            view_info = _buffer_view(model, int(group["buffer_index"]), mutable)
            if view_info is None:
                raise RetainedScaleExportError(
                    f"{label} target buffer is unreadable for {key!r}."
                )
            view, storage, offset, size = view_info
            if int(size) != len(raw):
                raise RetainedScaleExportError(
                    f"{label} packed size differs for {key}: graph={size}, codes={len(raw)}."
                )
            official_view_info = _buffer_view(
                official_model,
                int(official_groups[index]["buffer_index"]),
                official_section,
            )
            if official_view_info is None or int(official_view_info[3]) != len(raw):
                raise RetainedScaleExportError(
                    f"Official reference payload is unreadable or differently sized for {key}."
                )
            differs_from_official = not _buffer_views_equal(official_view_info[0], raw)
            view[:] = np.frombuffer(raw, dtype=np.uint8)
            item.update(
                {
                    "hf_weight_key": key,
                    "checkpoint_key": mappings[key],
                    "official_buffer": int(group["buffer_index"]),
                    "official_alias_count": len(group["aliases"]),
                    "storage": storage,
                    "storage_offset": int(offset),
                    "differs_from_official_codes": differs_from_official,
                }
            )
            telemetry.append(item)
            del weight, scale, raw
            if (index + 1) % 8 == 0:
                gc.collect()
    checks = {
        "processed_205": len(telemetry) == EXPECTED_MUTABLE_COUNT,
        "all_finite": all(item["finite"] for item in telemetry),
        "all_litert_pack_roundtrip": all(
            item["litert_pack_roundtrip"] for item in telemetry
        ),
        "only_w2_w4": _normalized_histogram(item["bits"] for item in telemetry)
        == EXPECTED_MUTABLE_BITS,
        "at_least_one_trained_code_changed": (
            any(item["differs_from_official_codes"] for item in telemetry)
            if label == "trained_merged_checkpoint"
            else True
        ),
    }
    if parity_enabled:
        checks.update(
            {
                "base_lora_candidate_dtype_match_205": len(parity_rows)
                == EXPECTED_MUTABLE_COUNT
                and all(item["candidate_base_dtype_match"] for item in parity_rows),
                "base_lora_candidate_bfloat16_205": len(parity_rows)
                == EXPECTED_MUTABLE_COUNT
                and all(item["candidate_base_bfloat16"] for item in parity_rows),
                "base_lora_numerical_parity_205": len(parity_rows)
                == EXPECTED_MUTABLE_COUNT
                and all(item["numerical_close"] for item in parity_rows),
                "base_lora_code_parity_205": len(parity_rows) == EXPECTED_MUTABLE_COUNT
                and all(item["retained_scale_code_exact"] for item in parity_rows),
                "base_delta_norms_finite_205": len(parity_rows)
                == EXPECTED_MUTABLE_COUNT
                and all(item["norms_finite"] for item in parity_rows),
                "at_least_one_lora_delta_nonzero": any(
                    item["delta_nonzero"] for item in parity_rows
                ),
            }
        )
    ratios = [
        float(item["delta_to_base_l2_ratio"])
        for item in parity_rows
        if item["delta_to_base_l2_ratio"] is not None
        and math.isfinite(float(item["delta_to_base_l2_ratio"]))
    ]
    return mutable, {
        "label": label,
        "checks": checks,
        "verified": all(checks.values()),
        "projection_count": len(telemetry),
        "total_values": sum(int(item["value_count"]) for item in telemetry),
        "total_clipped_low": sum(int(item["clipped_low_count"]) for item in telemetry),
        "total_clipped_high": sum(
            int(item["clipped_high_count"]) for item in telemetry
        ),
        "changed_from_official_count": sum(
            int(item["differs_from_official_codes"]) for item in telemetry
        ),
        "telemetry": telemetry,
        "base_lora_parity": {
            "enabled": parity_enabled,
            "projection_count": len(parity_rows),
            "numerical_exact_count": sum(
                int(item["numerical_exact"]) for item in parity_rows
            ),
            "numerical_close_count": sum(
                int(item["numerical_close"]) for item in parity_rows
            ),
            "retained_scale_code_exact_count": sum(
                int(item["retained_scale_code_exact"]) for item in parity_rows
            ),
            "nonzero_delta_count": sum(
                int(item["delta_nonzero"]) for item in parity_rows
            ),
            "delta_to_base_l2_ratio_min": min(ratios) if ratios else None,
            "delta_to_base_l2_ratio_max": max(ratios) if ratios else None,
            "telemetry": parity_rows,
        },
    }


def _buffer_diff_report(
    official_section: bytes,
    candidate_section: bytes | bytearray,
    all_records: list[dict[str, Any]],
    mutable_records: list[dict[str, Any]],
) -> dict[str, Any]:
    # Official views are read-only, so retain the original bytes rather than
    # copying the multi-gigabyte target section merely to compare buffers.
    official_model = _schema_model(official_section)
    candidate_model = _schema_model(candidate_section)
    official_groups = _official_weight_alias_groups(official_model, all_records)
    candidate_groups = _official_weight_alias_groups(candidate_model, all_records)
    mutable_buffers = {
        int(group["buffer_index"])
        for group in _official_weight_alias_groups(official_model, mutable_records)
    }
    changed: list[int] = []
    frozen_exact = 0
    for official_group, candidate_group in zip(
        official_groups, candidate_groups, strict=True
    ):
        if int(official_group["buffer_index"]) != int(candidate_group["buffer_index"]):
            raise RetainedScaleExportError("Candidate target buffer inventory changed.")
        buffer_index = int(official_group["buffer_index"])
        official_view = _buffer_view(official_model, buffer_index, official_section)
        candidate_view = _buffer_view(candidate_model, buffer_index, candidate_section)
        if official_view is None or candidate_view is None:
            raise RetainedScaleExportError(
                f"Could not compare target buffer {buffer_index}."
            )
        differs = not _buffer_views_equal(official_view[0], candidate_view[0])
        if differs:
            changed.append(buffer_index)
        elif buffer_index not in mutable_buffers:
            frozen_exact += 1
    changed_set = set(changed)
    return {
        "changed_buffer_count": len(changed),
        "changed_buffers": sorted(changed),
        "expected_mutable_buffer_count": len(mutable_buffers),
        "changed_buffers_within_expected_205": changed_set <= mutable_buffers,
        "frozen_72_byte_exact": frozen_exact == EXPECTED_FROZEN_COUNT,
        "frozen_exact_count": frozen_exact,
        "at_least_one_trained_code_changed": bool(changed),
    }


def _restore_official_payloads(
    official_section: bytes,
    candidate_section: bytes | bytearray,
    mutable_records: list[dict[str, Any]],
) -> dict[str, Any]:
    restored = bytearray(candidate_section)
    official_model = _schema_model(official_section)
    restored_model = _schema_model(restored)
    official_groups = _official_weight_alias_groups(official_model, mutable_records)
    restored_groups = _official_weight_alias_groups(restored_model, mutable_records)
    for official_group, restored_group in zip(
        official_groups, restored_groups, strict=True
    ):
        official_buffer = int(official_group["buffer_index"])
        restored_buffer = int(restored_group["buffer_index"])
        if official_buffer != restored_buffer:
            raise RetainedScaleExportError("Mutable target buffer index changed.")
        official_view = _buffer_view(official_model, official_buffer, official_section)
        restored_view = _buffer_view(restored_model, restored_buffer, restored)
        if official_view is None or restored_view is None:
            raise RetainedScaleExportError(
                f"Could not restore mutable target buffer {official_buffer}."
            )
        if int(official_view[3]) != int(restored_view[3]):
            raise RetainedScaleExportError("Mutable target buffer size changed.")
        _copy_buffer_view(official_view[0], restored_view[0])
    exact = restored == official_section
    return {
        "restored_buffer_count": len(restored_groups),
        "restored_official_payload_target_byte_exact": exact,
        "official_target_sha256": _sha256_bytes(official_section),
        "restored_target_sha256": _sha256_bytes(restored),
    }


def _graph_identity_report(
    official_section: bytes, candidate_section: bytes | bytearray
) -> dict[str, Any]:
    official = _graph_report(official_section)
    candidate = _graph_report(candidate_section)
    keys = (
        "structural_sha256",
        "quantization_layout_sha256",
        "execution_contract_sha256",
    )
    checks = {
        key: bool(
            official["graph"].get(key)
            and official["graph"].get(key) == candidate["graph"].get(key)
        )
        for key in keys
    }
    checks["execution_contract_complete"] = bool(
        official["graph"].get("execution_contract_complete")
        and candidate["graph"].get("execution_contract_complete")
    )
    checks["section_size_unchanged"] = len(official_section) == len(candidate_section)
    return {
        "checks": checks,
        "verified": all(checks.values()),
        "official": official,
        "candidate": candidate,
    }


def _build_plan(
    *,
    official_litertlm: str | Path,
    official_artifact_sha256: str,
    checkpoint: str | Path,
    adapter_checkpoint: str | Path,
    training_config: str | Path,
    mobile_training_seed_manifest: str | Path,
    mobile_qparams_contract: str | Path,
    zero_adapter_checkpoint: str | Path,
    output_dir: str | Path,
    output_litertlm: str | Path | None,
    report: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    official_path = Path(official_litertlm).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    adapter_path = Path(adapter_checkpoint).expanduser().resolve()
    config_path = Path(training_config).expanduser().resolve()
    seed_manifest = Path(mobile_training_seed_manifest).expanduser().resolve()
    qparams_path = Path(mobile_qparams_contract).expanduser().resolve()
    zero_checkpoint_path = Path(zero_adapter_checkpoint).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_path = (
        Path(output_litertlm).expanduser().resolve()
        if output_litertlm
        else output_root / "gemma4_e2b_retained_scale_code_only.litertlm"
    )
    report_path = (
        Path(report).expanduser().resolve()
        if report
        else output_root / "gemma4_retained_scale_code_only_report.json"
    )
    partial_path = output_path.with_name(output_path.name + ".partial")
    report_partial_path = report_path.with_name(report_path.name + ".partial")
    output_paths = (output_path, partial_path, report_path, report_partial_path)
    if len(set(output_paths)) != len(output_paths):
        raise RetainedScaleExportError(
            "Output, report, and their partial paths must be distinct."
        )
    required_files = {
        "official_litertlm": official_path.is_file(),
        "training_config": config_path.is_file(),
        "mobile_training_seed_manifest": seed_manifest.is_file(),
        "mobile_qparams_contract": qparams_path.is_file(),
        "adapter_checkpoint": adapter_path.is_dir(),
        "checkpoint": checkpoint_path.exists(),
        "zero_adapter_checkpoint": zero_checkpoint_path.exists(),
    }
    if not all(required_files.values()):
        failed = [name for name, present in required_files.items() if not present]
        raise RetainedScaleExportError(
            "Required exporter inputs are missing: " + ", ".join(failed)
        )
    declared_sha = str(official_artifact_sha256).strip().lower()
    if len(declared_sha) != 64 or any(
        value not in "0123456789abcdef" for value in declared_sha
    ):
        raise RetainedScaleExportError(
            "Official artifact SHA-256 must be 64 lowercase hex characters."
        )
    observed_sha = _sha256_file(official_path)
    official_identity = _official_artifact_sha_report(declared_sha, observed_sha)
    if not official_identity["verified"]:
        failed = [
            name for name, passed in official_identity["checks"].items() if not passed
        ]
        raise RetainedScaleExportError(
            "Official LiteRT-LM is not the audited pinned package: " + ", ".join(failed)
        )
    if output_path == official_path:
        raise RetainedScaleExportError("Output would overwrite the official package.")
    try:
        output_path.relative_to(output_root)
        report_path.relative_to(output_root)
    except ValueError as exc:
        raise RetainedScaleExportError(
            "Output artifact and report must both be inside --output-dir."
        ) from exc
    if output_path.parent != output_root or report_path.parent != output_root:
        raise RetainedScaleExportError(
            "Output artifact and report must be direct children of --output-dir."
        )
    collisions = [
        path
        for path in (
            output_root,
            output_path,
            partial_path,
            report_path,
            report_partial_path,
        )
        if path.exists()
    ]
    if collisions:
        raise RetainedScaleExportError(
            "Refusing an exporter output collision: "
            + ", ".join(str(path) for path in collisions)
        )

    config_report, config = _config_report(
        config_path, seed_manifest=seed_manifest, qparams_path=qparams_path
    )
    if not config_report["verified"]:
        failed = [
            name for name, passed in config_report["checks"].items() if not passed
        ]
        raise RetainedScaleExportError(
            "Training config is not retained-scale deployable: " + ", ".join(failed)
        )
    model_config = dict(config.get("model", {}))
    seed_report = verify_configured_mobile_training_seed(
        model_config, base=ROOT, require_materialized=True
    )
    if not seed_report.get("verified"):
        failed = [
            name for name, passed in seed_report.get("checks", {}).items() if not passed
        ]
        raise RetainedScaleExportError(
            "Mobile seed contract failed: " + ", ".join(failed)
        )
    seed_output = (
        seed_report.get("output") if isinstance(seed_report.get("output"), dict) else {}
    )
    if Path(str(seed_output.get("directory") or "")).resolve() != zero_checkpoint_path:
        raise RetainedScaleExportError(
            "--zero-adapter-checkpoint must be the exact materialized directory bound by the seed manifest."
        )
    qparams = MobileQParams(qparams_path, base=ROOT)
    package, records = _extract_inventory(
        official_path,
        TARGET_MODEL_TYPE,
        include_embeddings=True,
        max_weights=None,
    )
    scope, mutable_records, frozen_records = _scope_report(records, qparams)
    checkpoint_reader = SafetensorCheckpoint(checkpoint_path)
    zero_reader = SafetensorCheckpoint(zero_checkpoint_path)
    adapter_tensor_files = sorted(adapter_path.glob("adapter_model*.safetensors"))
    if len(adapter_tensor_files) != 1:
        raise RetainedScaleExportError(
            "Retained exporter requires exactly one adapter_model*.safetensors file; "
            f"found {len(adapter_tensor_files)}."
        )
    adapter_reader = SafetensorCheckpoint(adapter_tensor_files[0])
    checkpoint_mapping, checkpoint_mappings = _checkpoint_mapping_report(
        checkpoint_reader, mutable_records, qparams, label="trained_merged_checkpoint"
    )
    zero_mapping, zero_mappings = _checkpoint_mapping_report(
        zero_reader, mutable_records, qparams, label="zero_adapter_seed"
    )
    adapter_mapping, adapter_mappings, lora_scaling = _adapter_mapping_report(
        adapter_path,
        adapter_reader,
        mutable_records,
        qparams,
        config,
    )
    adapter_provenance = _best_adapter_provenance_report(
        adapter_path, config_path, qparams=qparams, seed_report=seed_report
    )
    if not adapter_provenance["verified"]:
        failed = [
            name for name, passed in adapter_provenance["checks"].items() if not passed
        ]
        raise RetainedScaleExportError(
            "Adapter is not an export-eligible retained-scale best checkpoint: "
            + ", ".join(failed)
        )
    merge_provenance = _merge_provenance_report(
        checkpoint_path,
        config_path,
        official_base_model_id=OFFICIAL_MOBILE_MODEL_ID,
        mobile_training_seed=seed_report,
    )
    if not merge_provenance.get("verified"):
        failed = [
            name
            for name, passed in merge_provenance.get("checks", {}).items()
            if not passed
        ]
        raise RetainedScaleExportError(
            "Merged-checkpoint provenance failed: " + ", ".join(failed)
        )
    merged_adapter_files = _normalized_file_records(
        merge_provenance.get("metadata", {}).get("adapter_files")
    )
    adapter_files = adapter_provenance["adapter_files"]
    if merged_adapter_files != adapter_files:
        raise RetainedScaleExportError(
            "Merged checkpoint was not produced from the supplied best-Golden adapter bytes."
        )
    adapter_selection = adapter_provenance.get("golden_selection")
    selection_matches = _merged_golden_selection_matches(
        merge_provenance, adapter_selection
    )
    if not selection_matches:
        raise RetainedScaleExportError(
            "Merged checkpoint and supplied adapter do not bind the same Golden "
            "selection metric, config, and cohort."
        )

    target_section = _section_by_model_type(package, TARGET_MODEL_TYPE)
    mtp_section = _section_by_model_type(package, MTP_MODEL_TYPE)
    plan_checks = {
        "required_inputs_present": all(required_files.values()),
        "official_artifact_sha256_pinned": official_identity["verified"],
        "retained_training_config_verified": config_report["verified"],
        "mobile_seed_verified": bool(seed_report.get("verified")),
        "retained_qparams_verified": bool(qparams.report.get("verified")),
        "exact_205_key_buffer_bijection": scope["verified"],
        "trained_checkpoint_mapping_205": checkpoint_mapping["verified"],
        "zero_adapter_checkpoint_mapping_205": zero_mapping["verified"],
        "adapter_ab_mapping_205": adapter_mapping["verified"],
        "best_golden_adapter_provenance": adapter_provenance["verified"],
        "merged_checkpoint_provenance": merge_provenance["verified"],
        "merged_adapter_bytes_match": merged_adapter_files == adapter_files,
        "merged_golden_selection_matches": selection_matches,
        "target_and_mtp_sections_present": bool(target_section and mtp_section),
        "output_paths_new_and_not_official": not collisions
        and output_path != official_path,
    }
    plan = {
        "mode": MODE,
        "executed": False,
        "plan_passed": all(plan_checks.values()),
        "passed": False,
        "checks": plan_checks,
        "official_litertlm": str(official_path),
        "official_artifact_sha256": observed_sha,
        "official_artifact_identity": official_identity,
        "checkpoint": str(checkpoint_path),
        "merged_checkpoint_identity": {
            "path": str(checkpoint_path),
            "metadata_path": str(merge_provenance.get("path") or ""),
            "metadata_sha256": (
                _sha256_file(Path(str(merge_provenance["path"])))
                if Path(str(merge_provenance.get("path") or "")).is_file()
                else None
            ),
            "file_verification": merge_provenance.get(
                "merged_model_file_verification", []
            ),
            "verified": merge_provenance["verified"],
        },
        "adapter_checkpoint": str(adapter_path),
        "adapter_identity": {
            "path": str(adapter_path),
            "training_metadata_sha256": adapter_provenance.get("sha256"),
            "files": adapter_files,
            "verified": adapter_provenance["verified"],
        },
        "zero_adapter_checkpoint": str(zero_checkpoint_path),
        "resolved_training_config_identity": {
            "path": config_report["path"],
            "sha256": config_report["sha256"],
            "verified": config_report["verified"],
        },
        "mobile_training_seed_identity": {
            "manifest_path": str(seed_manifest),
            "manifest_sha256": seed_report.get("manifest_sha256"),
            "transformation_plan_sha256": seed_report.get("transformation_plan_sha256"),
            "verified": seed_report.get("verified") is True,
        },
        "mobile_qparams_identity": {
            "contract_path": str(qparams.path),
            "contract_sha256": qparams.contract_sha256,
            "scale_storage_path": str(qparams.storage_path),
            "scale_storage_sha256": qparams.scale_storage_sha256,
            "verified": qparams.report.get("verified") is True,
        },
        "training_config": config_report,
        "mobile_training_seed": seed_report,
        "mobile_qparams": qparams.summary(),
        "scope": scope,
        "checkpoint_mapping": checkpoint_mapping,
        "zero_adapter_mapping": zero_mapping,
        "adapter_mapping": adapter_mapping,
        "adapter_provenance": adapter_provenance,
        "merge_provenance": merge_provenance,
        "target_section": target_section,
        "mtp_section": mtp_section,
        "output_dir": str(output_root),
        "output_litertlm": str(output_path),
        "report_path": str(report_path),
        "training_executed": False,
        "private_google_recipe_recovered": False,
        "claim_boundary": (
            "A passing export proves official topology/qparams/package preservation and "
            "retained-scale code serialization. It does not prove Google's private recipe, "
            "semantic accuracy, Android GPU execution, throughput, or MTP acceptance."
        ),
    }
    context = {
        "official_path": official_path,
        "checkpoint_reader": checkpoint_reader,
        "zero_reader": zero_reader,
        "adapter_reader": adapter_reader,
        "qparams": qparams,
        "records": records,
        "mutable_records": mutable_records,
        "frozen_records": frozen_records,
        "checkpoint_mappings": checkpoint_mappings,
        "zero_mappings": zero_mappings,
        "adapter_mappings": adapter_mappings,
        "lora_scaling": lora_scaling,
        "package": package,
        "target_section": target_section,
        "mtp_section": mtp_section,
        "output_root": output_root,
        "output_path": output_path,
        "partial_path": partial_path,
        "report_path": report_path,
        "report_partial_path": report_partial_path,
    }
    return plan, context


def run(
    *,
    official_litertlm: str | Path,
    official_artifact_sha256: str,
    checkpoint: str | Path,
    adapter_checkpoint: str | Path,
    training_config: str | Path,
    mobile_training_seed_manifest: str | Path,
    mobile_qparams_contract: str | Path,
    zero_adapter_checkpoint: str | Path,
    output_dir: str | Path,
    output_litertlm: str | Path | None = None,
    report: str | Path | None = None,
    execute: bool = False,
    working_set_bytes: int = 32 * 1024 * 1024,
) -> dict[str, Any]:
    plan, context = _build_plan(
        official_litertlm=official_litertlm,
        official_artifact_sha256=official_artifact_sha256,
        checkpoint=checkpoint,
        adapter_checkpoint=adapter_checkpoint,
        training_config=training_config,
        mobile_training_seed_manifest=mobile_training_seed_manifest,
        mobile_qparams_contract=mobile_qparams_contract,
        zero_adapter_checkpoint=zero_adapter_checkpoint,
        output_dir=output_dir,
        output_litertlm=output_litertlm,
        report=report,
    )
    if not execute:
        return plan
    if working_set_bytes <= 0:
        raise RetainedScaleExportError("--working-set-bytes must be positive.")

    official_path: Path = context["official_path"]
    target_section = context["target_section"]
    official_section = _read_section(official_path, target_section)
    qparams: MobileQParams = context["qparams"]
    records: list[dict[str, Any]] = context["records"]
    mutable_records: list[dict[str, Any]] = context["mutable_records"]

    # Bind normalized keys to the full inventory for explicit frozen/qparam reports.
    source_keys = _canonical_inventory_keys(records, "gemma4_e2b", TARGET_MODEL_TYPE)
    enriched_records: list[dict[str, Any]] = []
    for record, source_key in zip(records, source_keys, strict=True):
        item = dict(record)
        item["packed_source_key"] = source_key
        item["hf_weight_key"] = _normalize_official_key(source_key)
        enriched_records.append(item)

    official_weight_qparams = _weight_qparams_report(
        official_section, enriched_records, mutable_records, qparams
    )
    official_a8 = _activation_a8_report(official_section, mutable_records, qparams)
    official_all_qparams = _all_quantization_digest(official_section)
    if not official_weight_qparams["mutable_retained_scales_exact"]:
        raise RetainedScaleExportError(
            "Official target weight scales do not match all 205 retained scale tensors."
        )
    if not official_a8["retained_a8_contract_exact"]:
        raise RetainedScaleExportError(
            "Official target A8 qparams do not match all 1,220 mutable FC alias-edge scales."
        )

    zero_section, zero_quantization = _quantize_and_patch(
        official_section,
        checkpoint=context["zero_reader"],
        mappings=context["zero_mappings"],
        mutable_records=mutable_records,
        qparams=qparams,
        label="zero_adapter_seed",
        working_set_bytes=working_set_bytes,
    )
    zero_exact = zero_section == official_section
    zero_report = {
        "quantization": zero_quantization,
        "zero_adapter_target_byte_exact": zero_exact,
        "official_target_sha256": _sha256_bytes(official_section),
        "zero_adapter_target_sha256": _sha256_bytes(zero_section),
    }
    if not zero_exact:
        raise RetainedScaleExportError(
            "Zero-adapter retained-scale export is not byte-exact to the official target section."
        )
    del zero_section
    gc.collect()

    candidate_section, trained_quantization = _quantize_and_patch(
        official_section,
        checkpoint=context["checkpoint_reader"],
        mappings=context["checkpoint_mappings"],
        mutable_records=mutable_records,
        qparams=qparams,
        label="trained_merged_checkpoint",
        working_set_bytes=working_set_bytes,
        base_checkpoint=context["zero_reader"],
        base_mappings=context["zero_mappings"],
        adapter_checkpoint=context["adapter_reader"],
        adapter_mappings=context["adapter_mappings"],
        lora_scaling=context["lora_scaling"],
    )
    buffer_diff = _buffer_diff_report(
        official_section, candidate_section, enriched_records, mutable_records
    )
    restore = _restore_official_payloads(
        official_section, candidate_section, mutable_records
    )
    candidate_weight_qparams = _weight_qparams_report(
        candidate_section, enriched_records, mutable_records, qparams
    )
    candidate_a8 = _activation_a8_report(candidate_section, mutable_records, qparams)
    candidate_all_qparams = _all_quantization_digest(candidate_section)
    graph_identity = _graph_identity_report(official_section, candidate_section)
    qparam_identity = {
        "weight_qparams_byte_exact": candidate_weight_qparams["sha256"]
        == official_weight_qparams["sha256"],
        "activation_a8_qparams_byte_exact": candidate_a8["sha256"]
        == official_a8["sha256"],
        "all_tensor_qparams_byte_exact": candidate_all_qparams["sha256"]
        == official_all_qparams["sha256"],
        "official_weight": official_weight_qparams,
        "candidate_weight": candidate_weight_qparams,
        "official_a8": official_a8,
        "candidate_a8": candidate_a8,
        "official_all": official_all_qparams,
        "candidate_all": candidate_all_qparams,
    }

    prepackage_gates = {
        "zero_adapter_target_byte_exact": zero_report["zero_adapter_target_byte_exact"],
        "processed_205": trained_quantization["checks"]["processed_205"],
        "base_lora_candidate_dtype_match_205": trained_quantization["checks"][
            "base_lora_candidate_dtype_match_205"
        ],
        "base_lora_candidate_bfloat16_205": trained_quantization["checks"][
            "base_lora_candidate_bfloat16_205"
        ],
        "base_lora_numerical_parity_205": trained_quantization["checks"][
            "base_lora_numerical_parity_205"
        ],
        "base_lora_code_parity_205": trained_quantization["checks"][
            "base_lora_code_parity_205"
        ],
        "base_delta_norms_finite_205": trained_quantization["checks"][
            "base_delta_norms_finite_205"
        ],
        "at_least_one_lora_delta_nonzero": trained_quantization["checks"][
            "at_least_one_lora_delta_nonzero"
        ],
        "changed_buffers_within_expected_205": buffer_diff[
            "changed_buffers_within_expected_205"
        ],
        "at_least_one_trained_code_changed": buffer_diff[
            "at_least_one_trained_code_changed"
        ]
        and trained_quantization["checks"]["at_least_one_trained_code_changed"],
        "restored_official_payload_target_byte_exact": restore[
            "restored_official_payload_target_byte_exact"
        ],
        "frozen_72_byte_exact": buffer_diff["frozen_72_byte_exact"],
        "weight_qparams_byte_exact": qparam_identity["weight_qparams_byte_exact"],
        "activation_a8_qparams_byte_exact": qparam_identity[
            "activation_a8_qparams_byte_exact"
        ],
        "all_tensor_qparams_byte_exact": qparam_identity[
            "all_tensor_qparams_byte_exact"
        ],
        "official_retained_weight_scales_exact": official_weight_qparams[
            "mutable_retained_scales_exact"
        ],
        "official_retained_a8_scales_exact": official_a8["retained_a8_contract_exact"],
        "graph_layout_execution_identity": graph_identity["verified"],
        "section_size_unchanged": len(candidate_section) == len(official_section),
    }
    if not all(prepackage_gates.values()):
        failed = [name for name, passed in prepackage_gates.items() if not passed]
        raise RetainedScaleExportError(
            "Retained-scale target failed before packaging: " + ", ".join(failed)
        )

    output_root: Path = context["output_root"]
    output_path: Path = context["output_path"]
    partial: Path = context["partial_path"]
    report_path: Path = context["report_path"]
    report_partial: Path = context["report_partial_path"]
    # The plan required a fresh directory; keep that promise atomic at the
    # mutation boundary so a concurrently-created run is never reused.
    output_root.mkdir(parents=True, exist_ok=False)
    _write_package_exclusive(official_path, target_section, candidate_section, partial)
    mtp_section = context["mtp_section"]
    try:
        package_checks, output_inspection = _package_identity_report(
            official_path,
            partial,
            target_section=target_section,
            mtp_section=mtp_section,
            candidate_section=candidate_section,
        )
    except Exception:
        # ``partial`` was exclusively created by this invocation.
        if partial.exists():
            partial.unlink()
        raise

    gates = {
        **prepackage_gates,
        "exact_205_key_buffer_bijection": plan["scope"]["verified"],
        "expected_bit_histogram": plan["scope"]["mutable_bit_histogram"]
        == EXPECTED_MUTABLE_BITS,
        "retained_qparams_verified": bool(qparams.report.get("verified")),
        "legacy_metadata_rejected": plan["adapter_provenance"]["checks"][
            "legacy_metadata_rejected"
        ],
        "package_outside_target_byte_exact": package_checks["package_prefix_byte_exact"]
        and package_checks["package_suffix_byte_exact"],
        "mtp_byte_exact": package_checks["mtp_byte_exact"],
        "package_parseable": package_checks["output_package_parseable"],
    }
    passed = all(gates.values()) and all(package_checks.values())
    result = {
        **plan,
        "executed": True,
        "plan_passed": True,
        # Do not claim success or an output path until no-clobber publication
        # has actually completed.
        "passed": False,
        "promotion_pending": passed,
        "gates": gates,
        "zero_adapter_identity": zero_report,
        "trained_quantization": trained_quantization,
        "buffer_diff": buffer_diff,
        "restore_official_payload_proof": restore,
        "qparam_identity": qparam_identity,
        "graph_identity": graph_identity,
        "package_checks": package_checks,
        "output_inspection": output_inspection,
        "output_litertlm": None,
        "candidate_target_sha256": _sha256_bytes(candidate_section),
    }
    if not passed:
        result["promotion_pending"] = False
        result["report"] = str(report_path)
        try:
            _write_json_exclusive(report_path, result)
        finally:
            # ``partial`` was exclusively created by this invocation.
            if partial.exists():
                partial.unlink()
        failed = [
            name for name, value in {**gates, **package_checks}.items() if not value
        ]
        raise RetainedScaleExportError(
            "Retained-scale package failed final gates: " + ", ".join(failed)
        )

    output_linked = False
    report_partial_created = False
    publication_committed = False
    try:
        # Recheck immediately before publication; os.link itself is the final
        # atomic no-clobber gate if another process wins the race afterward.
        collisions = [
            path for path in (output_path, report_path, report_partial) if path.exists()
        ]
        if collisions:
            raise RetainedScaleExportError(
                "Exporter publication collision: "
                + ", ".join(str(path) for path in collisions)
            )
        output_sha = _sha256_file(partial)
        _link_no_clobber(partial, output_path)
        output_linked = True
        result.update(
            {
                "passed": True,
                "promotion_pending": False,
                "output_litertlm": str(output_path),
                "output_sha256": output_sha,
                "report": str(report_path),
            }
        )
        _write_json_exclusive(report_partial, result)
        report_partial_created = True
        _link_no_clobber(report_partial, report_path)
        publication_committed = True
        try:
            report_partial.unlink()
        except OSError:
            pass
        report_partial_created = False
        try:
            partial.unlink()
        except OSError:
            pass
    except Exception:
        if publication_committed:
            raise
        if report_partial_created and report_partial.exists():
            report_partial.unlink()
        if output_linked and output_path.exists() and partial.exists():
            try:
                if os.path.samefile(output_path, partial):
                    output_path.unlink()
            except OSError:
                # Fail closed without deleting a path whose identity changed.
                pass
        if partial.exists():
            partial.unlink()
        raise
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-litertlm", required=True)
    parser.add_argument("--official-artifact-sha256", required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Merged HF Safetensors directory/file."
    )
    parser.add_argument(
        "--adapter-checkpoint",
        required=True,
        help="Callback-created best Golden adapter.",
    )
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--mobile-training-seed-manifest", required=True)
    parser.add_argument("--mobile-qparams-contract", required=True)
    parser.add_argument("--zero-adapter-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-litertlm")
    parser.add_argument("--report")
    parser.add_argument("--working-set-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load/quantize projection values and write the gated package. Default is plan-only.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run(
            official_litertlm=args.official_litertlm,
            official_artifact_sha256=args.official_artifact_sha256,
            checkpoint=args.checkpoint,
            adapter_checkpoint=args.adapter_checkpoint,
            training_config=args.training_config,
            mobile_training_seed_manifest=args.mobile_training_seed_manifest,
            mobile_qparams_contract=args.mobile_qparams_contract,
            zero_adapter_checkpoint=args.zero_adapter_checkpoint,
            output_dir=args.output_dir,
            output_litertlm=args.output_litertlm,
            report=args.report,
            execute=bool(args.execute),
            working_set_bytes=int(args.working_set_bytes),
        )
    except (OSError, ValueError, RetainedScaleExportError) as exc:
        print(f"Gemma 4 retained-scale export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.execute:
        print("Plan only: no projection values were loaded and no package was written.")
    return (
        0
        if result.get("plan_passed") and (not args.execute or result.get("passed"))
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
