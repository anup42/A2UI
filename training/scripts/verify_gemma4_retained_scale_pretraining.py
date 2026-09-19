#!/usr/bin/env python3
"""Prove the retained-scale mobile seed is an exact pretraining no-op.

This gate intentionally runs before training and therefore accepts neither a
trained adapter nor a merged checkpoint.  It loads every one of the 205
materialized trainable seed tensors, re-encodes it with the retained mobile
scales, and requires the complete official target TFLite section to remain
byte-for-byte identical.  The official package is never copied or modified.

The report also inventories statically named KV-cache inputs and outputs.  That
inventory is schema evidence only; it never claims runtime cache simulation.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gemma4_retained_scale_litertlm as exporter
from build_checkpoint_official_topology import SafetensorCheckpoint
from build_converter_topology_parity import _read_section
from build_fresh_random_quantized_graph import _extract_inventory, _schema_model
from ir_training.qat.mobile_qparams import MobileQParams
from ir_training.qat.mobile_training_seed import (
    verify_configured_mobile_training_seed,
)

CONTRACT_VERSION = 1
MODE = "retained_scale_pretraining_noop_v1"
_CACHE_NAME = re.compile(
    r"(?:kv[_./-]?cache|cache[_./-]?kv|k[_./-]?cache|v[_./-]?cache|"
    r"key[_./-]?cache|value[_./-]?cache|"
    r"cache[_./-]?(?:k|v|key|value)|past[_./-]?(?:key|value)|"
    r"present[_./-]?(?:key|value))",
    re.IGNORECASE,
)
_KEY_NAME = re.compile(
    r"(?:^|[_./-])(?:k|key)(?:$|[_./-]|cache)|(?:cache|past|present)[_./-]?(?:k|key)",
    re.IGNORECASE,
)
_VALUE_NAME = re.compile(
    r"(?:^|[_./-])(?:v|value)(?:$|[_./-]|cache)|(?:cache|past|present)[_./-]?(?:v|value)",
    re.IGNORECASE,
)
_TENSOR_TYPES = {
    0: "FLOAT32",
    1: "FLOAT16",
    2: "INT32",
    3: "UINT8",
    4: "INT64",
    5: "STRING",
    6: "BOOL",
    7: "INT16",
    8: "COMPLEX64",
    9: "INT8",
    10: "FLOAT64",
    11: "COMPLEX128",
    12: "UINT64",
    13: "RESOURCE",
    14: "VARIANT",
    15: "UINT32",
    16: "UINT16",
    17: "INT4",
    18: "BFLOAT16",
}


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _call(value: Any, name: str, default: Any = None) -> Any:
    method = getattr(value, name, None)
    if not callable(method):
        return default
    try:
        result = method()
    except (AttributeError, TypeError, ValueError):
        return default
    return default if result is None else result


def _vector(value: Any, length_name: str, item_name: str) -> list[int]:
    length = int(_call(value, length_name, 0) or 0)
    getter = getattr(value, item_name, None)
    if not callable(getter):
        return []
    try:
        return [int(getter(index)) for index in range(length)]
    except (AttributeError, IndexError, TypeError, ValueError):
        return []


def _cache_role(name: str) -> str:
    key = bool(_KEY_NAME.search(name))
    value = bool(_VALUE_NAME.search(name))
    if key != value:
        return "key" if key else "value"
    return "cache_unclassified"


def _stage(name: str) -> str:
    normalized = name.lower()
    if "prefill" in normalized:
        return "prefill"
    if "decode" in normalized:
        return "decode"
    return "unclassified"


def _tensor_map(signature: Any, role: str) -> list[dict[str, Any]]:
    length = int(_call(signature, f"{role}Length", 0) or 0)
    getter = getattr(signature, role, None)
    if not callable(getter):
        return []
    rows: list[dict[str, Any]] = []
    for index in range(length):
        try:
            item = getter(index)
            rows.append(
                {
                    "name": _decode(_call(item, "Name", "")),
                    "tensor_index": int(_call(item, "TensorIndex", -1)),
                }
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
    return rows


def _signature_maps(model: Any) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_subgraph: dict[int, list[dict[str, Any]]] = {}
    signatures: list[dict[str, Any]] = []
    count = int(_call(model, "SignatureDefsLength", 0) or 0)
    getter = getattr(model, "SignatureDefs", None)
    if not callable(getter):
        return by_subgraph, signatures
    for index in range(count):
        try:
            signature = getter(index)
            subgraph_index = int(_call(signature, "SubgraphIndex", -1))
            key = _decode(_call(signature, "SignatureKey", ""))
            row = {
                "signature_index": index,
                "signature_key": key,
                "subgraph_index": subgraph_index,
                "stage": _stage(key),
                "inputs": _tensor_map(signature, "Inputs"),
                "outputs": _tensor_map(signature, "Outputs"),
            }
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        signatures.append(row)
        by_subgraph.setdefault(subgraph_index, []).append(row)
    return by_subgraph, signatures


def _static_kv_cache_contract(section: bytes | bytearray) -> dict[str, Any]:
    """Extract named cache boundary tensors without implying runtime behavior."""

    model = _schema_model(section)
    signatures_by_subgraph, signatures = _signature_maps(model)
    records: list[dict[str, Any]] = []
    subgraph_count = int(_call(model, "SubgraphsLength", 0) or 0)
    for subgraph_index in range(subgraph_count):
        subgraph = model.Subgraphs(subgraph_index)
        subgraph_name = _decode(_call(subgraph, "Name", ""))
        signature_rows = signatures_by_subgraph.get(subgraph_index, [])
        signature_names: dict[tuple[str, int], list[str]] = {}
        for signature in signature_rows:
            for boundary in ("inputs", "outputs"):
                for item in signature[boundary]:
                    name = str(item.get("name") or "")
                    tensor_index = int(item.get("tensor_index", -1))
                    if name:
                        signature_names.setdefault((boundary, tensor_index), []).append(name)
        boundaries = {
            "input": _vector(subgraph, "InputsLength", "Inputs"),
            "output": _vector(subgraph, "OutputsLength", "Outputs"),
        }
        for boundary, tensor_indices in boundaries.items():
            for tensor_index in tensor_indices:
                if tensor_index < 0:
                    continue
                tensor = subgraph.Tensors(tensor_index)
                tensor_name = _decode(_call(tensor, "Name", ""))
                aliases = signature_names.get((boundary + "s", tensor_index), [])
                names = [name for name in [tensor_name, *aliases] if name]
                if not any(_CACHE_NAME.search(name) for name in names):
                    continue
                dtype = int(_call(tensor, "Type", -1))
                quantization = exporter._quantization_vectors(tensor)
                stages = sorted(
                    {
                        stage
                        for stage in [
                            _stage(subgraph_name),
                            *[item["stage"] for item in signature_rows],
                        ]
                        if stage != "unclassified"
                    }
                )
                canonical_name = tensor_name or aliases[0]
                records.append(
                    {
                        "subgraph_index": subgraph_index,
                        "subgraph_name": subgraph_name,
                        "signature_keys": sorted(
                            str(item["signature_key"]) for item in signature_rows
                        ),
                        "stages": stages or ["unclassified"],
                        "boundary": boundary,
                        "tensor_index": tensor_index,
                        "name": canonical_name,
                        "signature_aliases": sorted(set(aliases)),
                        "cache_role": _cache_role(" ".join(names)),
                        "dtype": dtype,
                        "dtype_name": _TENSOR_TYPES.get(dtype, f"UNKNOWN_{dtype}"),
                        "shape": _vector(tensor, "ShapeLength", "Shape"),
                        "shape_signature": _vector(
                            tensor, "ShapeSignatureLength", "ShapeSignature"
                        ),
                        "qparams_present": quantization is not None,
                        "qparams_sha256": (
                            exporter._json_sha256(quantization)
                            if quantization is not None
                            else None
                        ),
                        "qparams": quantization,
                    }
                )

    records.sort(
        key=lambda item: (
            item["subgraph_index"],
            item["boundary"],
            item["tensor_index"],
            item["name"],
        )
    )
    inputs = [item for item in records if item["boundary"] == "input"]
    outputs = [item for item in records if item["boundary"] == "output"]
    stages = {
        stage for item in records for stage in item["stages"] if stage != "unclassified"
    }
    established = bool(records)
    return {
        "evidence_kind": "static_tflite_schema_inventory",
        "established": established,
        "status": "established" if established else "not_established_unidentified",
        "named_cache_input_count": len(inputs),
        "named_cache_output_count": len(outputs),
        "named_cache_tensor_count": len(records),
        "prefill_decode_stages_identified": {"prefill", "decode"} <= stages,
        # Names alone cannot bind a particular layer's prefill output to its
        # decode input, or prove a shared buffer/recurrent-state contract.
        "prefill_decode_mapping_established": False,
        "stages_identified": sorted(stages),
        "signature_count": len(signatures),
        "signatures": signatures,
        "records_sha256": exporter._json_sha256(records),
        "records": records,
        "runtime_simulation_performed": False,
        "runtime_cache_correctness_established": False,
        "claim_boundary": (
            "Named TFLite boundary tensors, dtypes, shapes, and serialized qparams "
            "are static decoding evidence only; cross-signature buffer mapping "
            "is not established and cache mutation, reuse, and runtime "
            "prefill/decode behavior were not executed."
        ),
    }


def _enriched_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_keys = exporter._canonical_inventory_keys(
        records, "gemma4_e2b", exporter.TARGET_MODEL_TYPE
    )
    result: list[dict[str, Any]] = []
    for record, source_key in zip(records, source_keys, strict=True):
        item = dict(record)
        item["packed_source_key"] = source_key
        item["hf_weight_key"] = exporter._normalize_official_key(source_key)
        result.append(item)
    return result


def _virtual_package_identity(
    official_path: Path,
    *,
    observed_sha256: str,
    target_section: dict[str, Any],
    mtp_section: dict[str, Any],
    official_target: bytes,
    candidate_target: bytes | bytearray,
) -> dict[str, Any]:
    target_begin = int(target_section["begin_offset"])
    target_size = int(target_section["size"])
    target_end = target_begin + target_size
    package_size = int(official_path.stat().st_size)
    mtp_begin = int(mtp_section["begin_offset"])
    mtp_size = int(mtp_section["size"])
    mtp_end = mtp_begin + mtp_size
    mtp_interval_valid = bool(
        0 <= mtp_begin <= mtp_end <= package_size
        and (
            mtp_end <= target_begin
            or mtp_begin >= target_end
            or (mtp_begin >= target_begin and mtp_end <= target_end)
        )
    )
    official_target_sha = exporter._sha256_bytes(official_target)
    candidate_target_sha = exporter._sha256_bytes(candidate_target)
    target_exact = bool(
        len(candidate_target) == target_size
        and candidate_target_sha == official_target_sha
        and candidate_target == official_target
    )
    checks = {
        "candidate_target_size_matches_section": len(candidate_target) == target_size,
        "candidate_target_byte_exact": target_exact,
        "official_package_sha_is_pinned": observed_sha256
        == exporter.OFFICIAL_LITERTLM_SHA256,
        "official_prefix_retained_by_virtual_substitution": target_begin >= 0,
        "official_suffix_retained_by_virtual_substitution": target_end <= package_size,
        "mtp_section_interval_valid": mtp_interval_valid,
        "no_output_package_materialized": True,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "package_size": package_size,
        "official_package_sha256": observed_sha256,
        "virtual_package_sha256": observed_sha256 if target_exact else None,
        "target": {
            "begin_offset": target_begin,
            "size": target_size,
            "official_sha256": official_target_sha,
            "candidate_sha256": candidate_target_sha,
        },
        "mtp": {
            "begin_offset": mtp_begin,
            "size": mtp_size,
            "sha256": exporter._file_range_sha256(
                official_path, mtp_begin, mtp_size
            ),
        },
        "package_copy_materialized": False,
        "proof": (
            "The virtual package substitutes a byte-identical target section into the "
            "unchanged pinned official source; therefore every package byte, including "
            "the MTP section, remains identical without writing a package copy."
        ),
    }


def run(
    *,
    official_litertlm: str | Path,
    official_artifact_sha256: str,
    training_config: str | Path,
    mobile_training_seed_manifest: str | Path,
    mobile_qparams_contract: str | Path,
    zero_adapter_checkpoint: str | Path,
    report: str | Path | None = None,
    working_set_bytes: int = 32 * 1024 * 1024,
) -> dict[str, Any]:
    """Run the pretraining no-op gate and optionally publish a JSON report."""

    if working_set_bytes <= 0:
        raise exporter.RetainedScaleExportError(
            "--working-set-bytes must be positive."
        )
    official_path = Path(official_litertlm).expanduser().resolve()
    config_path = Path(training_config).expanduser().resolve()
    seed_manifest = Path(mobile_training_seed_manifest).expanduser().resolve()
    qparams_path = Path(mobile_qparams_contract).expanduser().resolve()
    zero_path = Path(zero_adapter_checkpoint).expanduser().resolve()
    report_path = Path(report).expanduser().resolve() if report else None
    required = {
        "official_litertlm": official_path.is_file(),
        "training_config": config_path.is_file(),
        "mobile_training_seed_manifest": seed_manifest.is_file(),
        "mobile_qparams_contract": qparams_path.is_file(),
        "zero_adapter_checkpoint": zero_path.exists(),
    }
    if not all(required.values()):
        missing = [name for name, present in required.items() if not present]
        raise exporter.RetainedScaleExportError(
            "Required pretraining no-op inputs are missing: " + ", ".join(missing)
        )
    if report_path and report_path.exists():
        raise exporter.RetainedScaleExportError(
            f"Refusing to overwrite pretraining report: {report_path}"
        )
    if report_path:
        try:
            report_path.relative_to(zero_path)
        except ValueError:
            pass
        else:
            raise exporter.RetainedScaleExportError(
                "Pretraining report must not be written inside the verified seed directory."
            )

    declared_sha = str(official_artifact_sha256).strip().lower()
    observed_sha = exporter._sha256_file(official_path)
    official_identity = exporter._official_artifact_sha_report(
        declared_sha, observed_sha
    )
    if not official_identity["verified"]:
        failed = [
            name
            for name, passed in official_identity["checks"].items()
            if not passed
        ]
        raise exporter.RetainedScaleExportError(
            "Official LiteRT-LM is not the audited pinned package: "
            + ", ".join(failed)
        )
    source_stat_before = official_path.stat()

    config_report, config = exporter._config_report(
        config_path, seed_manifest=seed_manifest, qparams_path=qparams_path
    )
    if not config_report["verified"]:
        failed = [
            name for name, passed in config_report["checks"].items() if not passed
        ]
        raise exporter.RetainedScaleExportError(
            "Training config is not retained-scale deployable: " + ", ".join(failed)
        )
    model_config = (
        dict(config.get("model", {}))
        if isinstance(config.get("model"), dict)
        else {}
    )
    seed_report = verify_configured_mobile_training_seed(
        model_config, base=ROOT, require_materialized=True
    )
    if seed_report.get("verified") is not True:
        failed = [
            name
            for name, passed in seed_report.get("checks", {}).items()
            if not passed
        ]
        raise exporter.RetainedScaleExportError(
            "Mobile seed contract failed: " + ", ".join(failed)
        )
    seed_output = (
        seed_report.get("output")
        if isinstance(seed_report.get("output"), dict)
        else {}
    )
    if Path(str(seed_output.get("directory") or "")).resolve() != zero_path:
        raise exporter.RetainedScaleExportError(
            "--zero-adapter-checkpoint must be the exact materialized directory "
            "bound by the seed manifest."
        )

    qparams = MobileQParams(qparams_path, base=ROOT)
    if qparams.report.get("verified") is not True:
        raise exporter.RetainedScaleExportError(
            "Retained mobile qparams contract is not verified."
        )
    inventory_section, records = _extract_inventory(
        official_path,
        exporter.TARGET_MODEL_TYPE,
        include_embeddings=True,
        max_weights=None,
    )
    scope, mutable_records, _frozen_records = exporter._scope_report(records, qparams)
    _package, target_section, mtp_section = exporter._inspect_official_model_sections(
        official_path, inventory_section
    )

    zero_reader = SafetensorCheckpoint(zero_path)
    zero_mapping, zero_mappings = exporter._checkpoint_mapping_report(
        zero_reader,
        mutable_records,
        qparams,
        label="zero_adapter_seed",
    )
    official_section = _read_section(official_path, target_section)
    enriched_records = _enriched_records(records)
    official_weight = exporter._weight_qparams_report(
        official_section, enriched_records, mutable_records, qparams
    )
    official_a8 = exporter._activation_a8_report(
        official_section, mutable_records, qparams
    )
    official_all = exporter._all_quantization_digest(official_section)

    candidate_section, quantization = exporter._quantize_and_patch(
        official_section,
        checkpoint=zero_reader,
        mappings=zero_mappings,
        mutable_records=mutable_records,
        qparams=qparams,
        label="zero_adapter_seed",
        working_set_bytes=working_set_bytes,
    )
    buffer_diff = exporter._buffer_diff_report(
        official_section, candidate_section, enriched_records, mutable_records
    )
    candidate_weight = exporter._weight_qparams_report(
        candidate_section, enriched_records, mutable_records, qparams
    )
    candidate_a8 = exporter._activation_a8_report(
        candidate_section, mutable_records, qparams
    )
    candidate_all = exporter._all_quantization_digest(candidate_section)
    graph_identity = exporter._graph_identity_report(
        official_section, candidate_section
    )
    kv_cache_contract = _static_kv_cache_contract(official_section)
    package_identity = _virtual_package_identity(
        official_path,
        observed_sha256=observed_sha,
        target_section=target_section,
        mtp_section=mtp_section,
        official_target=official_section,
        candidate_target=candidate_section,
    )

    telemetry = quantization.get("telemetry", [])
    code_buffers = {int(item["official_buffer"]) for item in telemetry}
    qparam_identity = {
        "weight_qparams_byte_exact": candidate_weight["sha256"]
        == official_weight["sha256"],
        "activation_a8_qparams_byte_exact": candidate_a8["sha256"]
        == official_a8["sha256"],
        "all_tensor_qparams_byte_exact": candidate_all["sha256"]
        == official_all["sha256"],
        "official_weight": official_weight,
        "candidate_weight": candidate_weight,
        "official_a8": official_a8,
        "candidate_a8": candidate_a8,
        "official_all": official_all,
        "candidate_all": candidate_all,
    }
    source_sha_after = exporter._sha256_file(official_path)
    source_stat_after = official_path.stat()
    source_untouched = bool(
        source_sha_after == observed_sha
        and source_stat_after.st_size == source_stat_before.st_size
        and source_stat_after.st_mtime_ns == source_stat_before.st_mtime_ns
    )
    gates = {
        "official_artifact_sha256_pinned": official_identity["verified"],
        "retained_training_config_verified": config_report["verified"],
        "materialized_seed_provenance_verified": seed_report.get("verified") is True,
        "retained_qparams_verified": qparams.report.get("verified") is True,
        "exact_205_key_buffer_bijection": scope["verified"],
        "materialized_seed_mapping_205": zero_mapping["verified"],
        "materialized_seed_projections_processed_205": quantization["checks"][
            "processed_205"
        ],
        "materialized_seed_quantization_verified": quantization["verified"],
        "unique_materialized_code_buffers_205": len(code_buffers)
        == exporter.EXPECTED_MUTABLE_COUNT,
        "every_materialized_code_buffer_matches_official": all(
            item.get("differs_from_official_codes") is False for item in telemetry
        ),
        "zero_adapter_target_byte_exact": candidate_section == official_section,
        "frozen_72_byte_exact": buffer_diff["frozen_72_byte_exact"],
        "no_target_buffer_changed": buffer_diff["changed_buffer_count"] == 0,
        "official_retained_weight_scales_exact": official_weight[
            "mutable_retained_scales_exact"
        ],
        "official_retained_a8_scales_exact": official_a8[
            "retained_a8_contract_exact"
        ],
        "weight_qparams_byte_exact": qparam_identity["weight_qparams_byte_exact"],
        "activation_a8_qparams_byte_exact": qparam_identity[
            "activation_a8_qparams_byte_exact"
        ],
        "all_tensor_qparams_byte_exact": qparam_identity[
            "all_tensor_qparams_byte_exact"
        ],
        "graph_layout_execution_identity": graph_identity["verified"],
        "virtual_package_identity": package_identity["verified"],
        "official_source_untouched": source_untouched,
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise exporter.RetainedScaleExportError(
            "Pretraining retained-scale no-op gate failed: " + ", ".join(failed)
        )

    result = {
        "contract_version": CONTRACT_VERSION,
        "mode": MODE,
        "gate_status": "PASSED",
        "passed": True,
        "training_started": False,
        "model_execution_performed": False,
        "package_copy_materialized": False,
        "checks": gates,
        "official_litertlm": str(official_path),
        "official_artifact_identity": official_identity,
        "official_source_after_sha256": source_sha_after,
        "resolved_training_config_identity": config_report,
        "mobile_training_seed": seed_report,
        "mobile_qparams": qparams.summary(),
        "scope": scope,
        "zero_adapter_mapping": zero_mapping,
        "materialized_seed_quantization": quantization,
        "buffer_diff": buffer_diff,
        "qparam_identity": qparam_identity,
        "graph_identity": graph_identity,
        "virtual_package_identity": package_identity,
        "static_kv_cache_contract": kv_cache_contract,
        "report": str(report_path) if report_path else None,
        "claim_boundary": (
            "This passed gate proves that the provenance-bound materialized seed "
            "re-encodes all 205 retained-scale projection code buffers to the pinned "
            "official target bytes without changing frozen constants, qparams, graph "
            "or package identity. It does not prove training quality, runtime inference, "
            "KV-cache behavior, Android execution, or Google's private recipe."
        ),
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        exporter._write_json_exclusive(report_path, result)
    del candidate_section, official_section
    gc.collect()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-litertlm", required=True)
    parser.add_argument("--official-artifact-sha256", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--mobile-training-seed-manifest", required=True)
    parser.add_argument("--mobile-qparams-contract", required=True)
    parser.add_argument("--zero-adapter-checkpoint", required=True)
    parser.add_argument("--report")
    parser.add_argument("--working-set-bytes", type=int, default=32 * 1024 * 1024)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run(
            official_litertlm=args.official_litertlm,
            official_artifact_sha256=args.official_artifact_sha256,
            training_config=args.training_config,
            mobile_training_seed_manifest=args.mobile_training_seed_manifest,
            mobile_qparams_contract=args.mobile_qparams_contract,
            zero_adapter_checkpoint=args.zero_adapter_checkpoint,
            report=args.report,
            working_set_bytes=int(args.working_set_bytes),
        )
    except (OSError, ValueError, exporter.RetainedScaleExportError) as exc:
        print(f"Gemma 4 retained-scale pretraining gate failed: {exc}", file=sys.stderr)
        return 2
    # Full per-buffer evidence stays in the bound JSON report; the stage console
    # needs the gate outcome, not hundreds of tensor and quantization records.
    displayed = ({key: result.get(key) for key in (
        "mode", "gate_status", "passed", "checks", "report", "claim_boundary"
    )} if args.report else result)
    print(json.dumps(displayed, indent=2, ensure_ascii=False))
    if result.get("passed") is not True:
        print("Gemma 4 retained-scale pretraining gate did not pass.", file=sys.stderr)
        return 2
    print("PASSED: materialized retained-scale seed is an exact official-mobile no-op.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
