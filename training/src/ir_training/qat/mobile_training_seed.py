"""Identity and materialization checks for the public Gemma 4 mobile seed.

The released mobile Transformers checkpoint is packed W2/W4/W8 data, not a
normal dense training checkpoint.  ``reconstruct_gemma4_mobile_training_seed``
streams those public quantization-cell centers into a text-only BF16 checkpoint.
This module binds every later training, merge, and topology-transplant stage to
that exact reconstruction plan and to the materialized shard hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root

OFFICIAL_MOBILE_MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
OFFICIAL_MOBILE_REVISION = "dd693ff40353f057ca5f07e945ad867f4afbf2ec"
OFFICIAL_MOBILE_SAFETENSORS_SHA256 = (
    "efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4"
)
OFFICIAL_MOBILE_CONFIG_SHA256 = (
    "cf6d7dc22738b5e6beb364bac833d78b869f5a6ffd57dfc96c6be3f2abc80424"
)
OFFICIAL_LITERTLM_SHA256 = (
    "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c"
)
OFFICIAL_RETAINED_COMPILED_REPORT_SHA256 = (
    "4fa47cf6fefb983a79bebc1e00bdd1f28df8d6570f59d7e979a1791a9a63429a"
)
EXPECTED_TRANSFORMATION_PLAN_SHA256 = (
    "03086afb123acf2c6f359d3cec2b1208f89529bca39e2d29f8b501e0918e3d5c"
)
EXPECTED_TENSOR_COUNT = 541
EXPECTED_DIRECT_BF16_COUNT = 263
EXPECTED_DEQUANTIZED_COUNT = 278
EXPECTED_BIT_HISTOGRAM = {"W2": 62, "W4": 146, "W8": 70}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_plan_sha256(mappings: Any) -> str | None:
    if not isinstance(mappings, list):
        return None
    try:
        canonical = json.dumps(
            mappings,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical).hexdigest()


def _resolve_optional(value: str | Path | None, base: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return resolve_path(value, base)


def _safe_output_file(directory: Path, value: Any) -> Path | None:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        return None
    resolved = (directory / relative).resolve()
    try:
        resolved.relative_to(directory.resolve())
    except ValueError:
        return None
    return resolved


def _verify_file_records(
    records: Any,
    *,
    directory: Path,
    required: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    if not isinstance(records, list) or (required and not records):
        return False, []
    verification: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            verification.append({"valid": False, "error": "record_not_object"})
            continue
        path = _safe_output_file(directory, record.get("path"))
        expected_hash = str(record.get("sha256") or "").strip().lower()
        try:
            expected_size = int(record.get("size_bytes", -1))
        except (TypeError, ValueError):
            expected_size = -1
        observed_size = path.stat().st_size if path and path.is_file() else None
        observed_hash = _sha256_file(path) if path and path.is_file() else None
        valid = bool(
            path
            and observed_size == expected_size
            and len(expected_hash) == 64
            and observed_hash == expected_hash
        )
        verification.append(
            {
                "path": str(path) if path else None,
                "expected_size_bytes": expected_size,
                "observed_size_bytes": observed_size,
                "expected_sha256": expected_hash or None,
                "observed_sha256": observed_hash,
                "valid": valid,
            }
        )
    return bool(verification or not required) and all(
        item.get("valid") is True for item in verification
    ), verification


def verify_mobile_training_seed_manifest(
    manifest_path: str | Path | None,
    *,
    expected_model_id: str = OFFICIAL_MOBILE_MODEL_ID,
    expected_model_source: str | Path | None = None,
    base: str | Path | None = None,
    require_materialized: bool = True,
) -> dict[str, Any]:
    """Verify the exact reconstructed seed and every materialized model shard."""

    anchor = Path(base).resolve() if base is not None else training_root()
    path = _resolve_optional(manifest_path, anchor)
    model_source = _resolve_optional(expected_model_source, anchor)
    checks = {
        "manifest_present": False,
        "manifest_schema": False,
        "source_identity": False,
        "source_config_identity": False,
        "official_graph_authority": False,
        "transformation_plan": False,
        "transformation_mapping_exact": False,
        "tensor_inventory": False,
        "precision_inventory": False,
        "weight_map_inventory": False,
        "tensor_hash_inventory": not require_materialized,
        "payload_size_consistent": False,
        "model_source_matches_manifest": False,
        "materialized": not require_materialized,
        "shard_set_exact": not require_materialized,
        "shard_payload_consistent": not require_materialized,
        "shard_hashes_match": not require_materialized,
        "auxiliary_hashes_match": not require_materialized,
        "dense_config_valid": not require_materialized,
        "no_training_or_private_recipe_claim": False,
    }
    report: dict[str, Any] = {
        "required": True,
        "path": str(path) if path else None,
        "expected_model_id": expected_model_id,
        "expected_model_source": str(model_source) if model_source else None,
        "checks": checks,
        "verified": False,
    }
    if path is None or not path.is_file():
        report["error"] = "Mobile training-seed manifest is missing."
        return report
    checks["manifest_present"] = True
    report["manifest_sha256"] = _sha256_file(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report["error"] = f"Could not read mobile training-seed manifest: {exc}"
        return report
    if not isinstance(manifest, dict):
        report["error"] = "Mobile training-seed manifest root is not an object."
        return report

    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    authority = (
        manifest.get("official_graph_authority")
        if isinstance(manifest.get("official_graph_authority"), dict)
        else {}
    )
    retained = (
        authority.get("retained_compiled_parity")
        if isinstance(authority.get("retained_compiled_parity"), dict)
        else {}
    )
    output = manifest.get("output") if isinstance(manifest.get("output"), dict) else {}
    transform = (
        manifest.get("transformation")
        if isinstance(manifest.get("transformation"), dict)
        else {}
    )
    checks["manifest_schema"] = bool(
        int(manifest.get("manifest_version", 0) or 0) == 1
        and manifest.get("seed_format")
        == "gemma4_e2b_mobile_dequantized_bf16_text"
        and manifest.get("ready") is True
    )
    checks["source_identity"] = bool(
        str(source.get("model_id") or "") == expected_model_id
        and str(source.get("revision") or "") == OFFICIAL_MOBILE_REVISION
        and str(source.get("safetensors_sha256_expected") or "").lower()
        == OFFICIAL_MOBILE_SAFETENSORS_SHA256
        and str(source.get("safetensors_sha256_observed") or "").lower()
        == OFFICIAL_MOBILE_SAFETENSORS_SHA256
    )
    checks["source_config_identity"] = bool(
        str(source.get("config_sha256_expected") or "").lower()
        == OFFICIAL_MOBILE_CONFIG_SHA256
        and str(source.get("config_sha256") or "").lower()
        == OFFICIAL_MOBILE_CONFIG_SHA256
    )
    retained_checks = retained.get("checks") if isinstance(retained.get("checks"), dict) else {}
    checks["official_graph_authority"] = bool(
        str(authority.get("artifact_sha256") or "").lower()
        == OFFICIAL_LITERTLM_SHA256
        and str(retained.get("sha256") or "").lower()
        == OFFICIAL_RETAINED_COMPILED_REPORT_SHA256
        and retained.get("verified") is True
        and retained_checks
        and all(value is True for value in retained_checks.values())
    )
    mappings = transform.get("tensor_mappings")
    computed_plan_sha256 = _canonical_plan_sha256(mappings)
    declared_plan_sha256 = str(transform.get("plan_sha256") or "").lower()
    checks["transformation_plan"] = bool(
        declared_plan_sha256 == EXPECTED_TRANSFORMATION_PLAN_SHA256
    )
    checks["transformation_mapping_exact"] = bool(
        computed_plan_sha256 and computed_plan_sha256 == declared_plan_sha256
    )
    mapping_keys = [
        str(item.get("output_key") or "")
        for item in mappings or []
        if isinstance(item, dict)
    ]
    checks["tensor_inventory"] = bool(
        int(output.get("tensor_count", 0) or 0) == EXPECTED_TENSOR_COUNT
        and int(output.get("direct_bf16_tensor_count", 0) or 0)
        == EXPECTED_DIRECT_BF16_COUNT
        and int(output.get("dequantized_tensor_count", 0) or 0)
        == EXPECTED_DEQUANTIZED_COUNT
        and int(transform.get("tensor_count", 0) or 0) == EXPECTED_TENSOR_COUNT
        and len(mappings or []) == EXPECTED_TENSOR_COUNT
        and len(mapping_keys) == len(set(mapping_keys)) == EXPECTED_TENSOR_COUNT
    )
    try:
        bit_histogram = {
            str(key): int(value)
            for key, value in dict(transform.get("bit_histogram") or {}).items()
        }
    except (TypeError, ValueError):
        bit_histogram = {}
    checks["precision_inventory"] = bit_histogram == EXPECTED_BIT_HISTOGRAM
    weight_map = output.get("weight_map")
    weight_map = weight_map if isinstance(weight_map, dict) else {}
    declared_shard_names = {
        str(item.get("path") or "")
        for item in output.get("shards") or []
        if isinstance(item, dict)
    }
    checks["weight_map_inventory"] = bool(
        set(weight_map) == set(mapping_keys)
        and declared_shard_names
        and {str(value) for value in weight_map.values()} <= declared_shard_names
    )
    try:
        mapped_payload_size = sum(
            int(item.get("output_nbytes", -1))
            for item in mappings or []
            if isinstance(item, dict)
        )
    except (TypeError, ValueError):
        mapped_payload_size = -1
    checks["payload_size_consistent"] = bool(
        mapped_payload_size > 0
        and mapped_payload_size == int(output.get("payload_size_bytes", 0) or 0)
    )

    manifest_directory = path.parent.resolve()
    declared_directory = Path(str(output.get("directory") or "")).expanduser()
    if not declared_directory.is_absolute():
        declared_directory = resolve_path(declared_directory, anchor)
    checks["model_source_matches_manifest"] = bool(
        model_source
        and model_source.resolve() == manifest_directory
        and declared_directory.resolve() == manifest_directory
    )
    checks["no_training_or_private_recipe_claim"] = bool(
        manifest.get("training_executed") is False
        and manifest.get("private_google_recipe_recovered") is False
    )

    shard_records = output.get("shards")
    auxiliary_records = output.get("auxiliary_files")
    if require_materialized:
        checks["materialized"] = bool(
            manifest.get("executed") is True and output.get("materialized") is True
        )
        shard_ok, shard_verification = _verify_file_records(
            shard_records, directory=manifest_directory, required=True
        )
        auxiliary_ok, auxiliary_verification = _verify_file_records(
            auxiliary_records, directory=manifest_directory, required=True
        )
        report["shard_verification"] = shard_verification
        report["auxiliary_verification"] = auxiliary_verification
        checks["shard_hashes_match"] = shard_ok
        checks["auxiliary_hashes_match"] = auxiliary_ok
        tensor_hashes = output.get("tensor_sha256")
        tensor_hashes = tensor_hashes if isinstance(tensor_hashes, dict) else {}
        checks["tensor_hash_inventory"] = bool(
            set(tensor_hashes) == set(mapping_keys)
            and all(
                len(str(value)) == 64
                and all(character in "0123456789abcdef" for character in str(value))
                for value in tensor_hashes.values()
            )
        )
        declared_shards = {
            str(item.get("path") or "")
            for item in shard_records or []
            if isinstance(item, dict)
        }
        actual_shards = {
            item.name
            for item in manifest_directory.glob("model*.safetensors")
            if item.is_file()
        }
        checks["shard_set_exact"] = bool(
            declared_shards
            and declared_shards == actual_shards
            and len(declared_shards) == int(output.get("shard_count", 0) or 0)
        )
        try:
            shard_payload_size = sum(
                int(item.get("payload_bytes", -1))
                for item in shard_records or []
                if isinstance(item, dict)
            )
            shard_sizes_exceed_payloads = all(
                int(item.get("size_bytes", -1)) > int(item.get("payload_bytes", -1))
                for item in shard_records or []
                if isinstance(item, dict)
            )
        except (TypeError, ValueError):
            shard_payload_size = -1
            shard_sizes_exceed_payloads = False
        checks["shard_payload_consistent"] = bool(
            shard_payload_size == int(output.get("payload_size_bytes", 0) or 0)
            and shard_sizes_exceed_payloads
        )
        config_path = manifest_directory / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            config = {}
        checks["dense_config_valid"] = bool(
            isinstance(config, dict)
            and config.get("model_type") == "gemma4_text"
            and config.get("architectures") == ["Gemma4ForCausalLM"]
            and config.get("tie_word_embeddings") is False
            and str(config.get("dtype") or "").lower() == "bfloat16"
        )

    report.update(
        {
            "seed_id": manifest.get("seed_id"),
            "source": {
                "model_id": source.get("model_id"),
                "revision": source.get("revision"),
                "safetensors_sha256": source.get("safetensors_sha256_observed"),
            },
            "output": {
                "directory": str(manifest_directory),
                "tensor_count": output.get("tensor_count"),
                "shard_count": output.get("shard_count"),
            },
            "transformation_plan_sha256": transform.get("plan_sha256"),
            "computed_transformation_plan_sha256": computed_plan_sha256,
            "verified": all(checks.values()),
            "private_google_recipe_recovered": False,
        }
    )
    return report


def verify_configured_mobile_training_seed(
    model_config: dict[str, Any],
    *,
    base: str | Path | None = None,
    require_materialized: bool = True,
) -> dict[str, Any]:
    """Verify the seed declared by a model config, or report it as not required."""

    model_id = str(model_config.get("model_id") or "").strip()
    required = model_id == OFFICIAL_MOBILE_MODEL_ID
    if not required:
        return {
            "required": False,
            "verified": True,
            "model_id": model_id,
            "checks": {"mobile_training_seed_not_required": True},
        }
    return verify_mobile_training_seed_manifest(
        model_config.get("mobile_training_seed_manifest"),
        expected_model_id=model_id,
        expected_model_source=model_config.get("model_source"),
        base=base,
        require_materialized=require_materialized,
    )
