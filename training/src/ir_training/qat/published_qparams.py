"""Bind retained A8 qparams to the pinned public packed checkpoint.

The mobile qparams contract is self-hashed, but that alone cannot establish
that its activation scalars came from the published Safetensors.  This module
checks every mapped activation-scale scalar against that source byte-for-byte
without materializing any model weight.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ir_training.qat.mobile_qparams import verify_mobile_qparams_contract
from ir_training.qat.mobile_training_seed import (
    OFFICIAL_MOBILE_SAFETENSORS_SHA256,
    verify_mobile_training_seed_manifest,
)

EXPECTED_MUTABLE_A8_WEIGHTS = 205
EXPECTED_FROZEN_A8_WEIGHTS = 70
EXPECTED_ZERO_A8_WEIGHTS = 1
_ROLES = ("input", "output")
_MUTABLE_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
_FROZEN_SUFFIXES = (
    ".per_layer_input_gate.weight",
    ".per_layer_projection.weight",
)


class PublishedQParamsError(RuntimeError):
    """Raised when published activation-scale provenance cannot be proven."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishedQParamsError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublishedQParamsError(f"{label} root must be an object.")
    return value


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _contract_bytes(entry: dict[str, Any], role: str, weight_key: str) -> bytes | None:
    field = f"{role}_activation_scale_f32_le_hex"
    if field not in entry:
        return None
    try:
        raw = bytes.fromhex(str(entry[field]))
    except ValueError as exc:
        raise PublishedQParamsError(
            f"Malformed retained {role} A8 bytes for {weight_key!r}."
        ) from exc
    if len(raw) != 4:
        raise PublishedQParamsError(
            f"Retained {role} A8 scale for {weight_key!r} is not one FP32 scalar."
        )
    return raw


def _scope(weight_key: str) -> str | None:
    if weight_key == "lm_head.weight":
        return "zero_head"
    if weight_key.startswith("model.layers.") and weight_key.endswith(
        _MUTABLE_SUFFIXES
    ):
        return "mutable"
    if weight_key.startswith("model.layers.") and weight_key.endswith(
        _FROZEN_SUFFIXES
    ):
        return "frozen"
    return None


def _is_text_activation_scale_key(key: str) -> bool:
    return bool(
        key.startswith(("model.language_model.", "lm_head."))
        and key.endswith((".input_activation_scale", ".output_activation_scale"))
    )


def verify_published_activation_scales(
    source_safetensors: Path,
    seed_manifest: Path,
    qparams_contract: Path,
    *,
    verified_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every retained A8 scalar against the pinned packed source.

    ``verified_source_sha256`` may be supplied by a caller that has just hashed
    the same file.  It avoids a second multi-gigabyte read only when it equals
    the required pinned digest; otherwise verification fails closed.
    """

    source_path = Path(source_safetensors).expanduser().resolve()
    manifest_path = Path(seed_manifest).expanduser().resolve()
    contract_path = Path(qparams_contract).expanduser().resolve()
    for path, label in (
        (source_path, "packed source Safetensors"),
        (manifest_path, "mobile seed manifest"),
        (contract_path, "mobile qparams contract"),
    ):
        if not path.is_file():
            raise PublishedQParamsError(f"{label} is missing: {path}")

    expected_hash = str(OFFICIAL_MOBILE_SAFETENSORS_SHA256).strip().lower()
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        raise PublishedQParamsError("Expected packed-source SHA-256 is invalid.")
    if verified_source_sha256 is None:
        observed_hash = _sha256_file(source_path)
        hash_mode = "computed"
    else:
        observed_hash = str(verified_source_sha256).strip().lower()
        hash_mode = "caller_verified"
    if observed_hash != expected_hash:
        raise PublishedQParamsError(
            "Packed source Safetensors SHA-256 differs from the pinned identity: "
            f"observed {observed_hash!r}, expected {expected_hash!r}."
        )

    seed_report = verify_mobile_training_seed_manifest(
        manifest_path,
        expected_model_source=manifest_path.parent,
        base=manifest_path.parent,
        require_materialized=False,
    )
    if seed_report.get("verified") is not True:
        raise PublishedQParamsError("Mobile seed manifest verification failed.")
    qparams_report = verify_mobile_qparams_contract(
        contract_path, base=contract_path.parent
    )
    if qparams_report.get("verified") is not True:
        raise PublishedQParamsError("Mobile qparams contract verification failed.")

    manifest = _json(manifest_path, "mobile seed manifest")
    mappings = (manifest.get("transformation") or {}).get("tensor_mappings")
    if not isinstance(mappings, list):
        raise PublishedQParamsError("Seed manifest has no tensor_mappings list.")
    inventory = qparams_report.get("inventory")
    if not isinstance(inventory, dict):
        raise PublishedQParamsError("Verified qparams report has no inventory.")

    mapped: dict[str, dict[str, str]] = {}
    for item in mappings:
        if not isinstance(item, dict) or not item.get("scale_key"):
            continue
        source_key = str(item.get("source_key") or "")
        output_key = str(item.get("output_key") or "")
        scale_key = str(item.get("scale_key") or "")
        if not source_key.endswith((".weight", ".embedding_quantized")) or not output_key.endswith(".weight"):
            raise PublishedQParamsError("Scaled seed mapping is not a supported text matrix mapping.")
        if output_key in mapped:
            raise PublishedQParamsError(f"Duplicate scaled mapping for {output_key!r}.")
        mapped[output_key] = {"source_key": source_key, "scale_key": scale_key}
    if set(mapped) != set(inventory):
        raise PublishedQParamsError(
            "Seed scaled-weight mappings and qparams inventory are not bijective."
        )

    try:
        import numpy as np
        from safetensors import safe_open
    except ImportError as exc:
        raise PublishedQParamsError(
            "Published A8 verification requires numpy and safetensors."
        ) from exc

    role_records: dict[str, list[dict[str, str]]] = {role: [] for role in _ROLES}
    scope_counts = {"mutable": 0, "frozen": 0, "zero_head": 0}
    expected_source_roles: set[str] = set()
    with safe_open(str(source_path), framework="np") as handle:
        source_keys = set(handle.keys())
        # The packed multimodal artifact can contain activation scales outside
        # the reconstructed text model. Restrict unexpected-role detection to
        # the same language-model/lm_head source boundary as reconstruction.
        actual_source_roles = {key for key in source_keys if _is_text_activation_scale_key(key)}
        for weight_key in sorted(mapped):
            mapping = mapped[weight_key]
            if mapping["source_key"] not in source_keys or mapping["scale_key"] not in source_keys:
                raise PublishedQParamsError(
                    f"Packed source lacks mapped weight/scale tensors for {weight_key!r}."
                )
            entry = inventory[weight_key]
            if not isinstance(entry, dict):
                raise PublishedQParamsError(f"Invalid qparams entry for {weight_key!r}.")
            if not mapping["source_key"].endswith(".weight"):
                if any(_contract_bytes(entry, role, weight_key) is not None for role in _ROLES):
                    raise PublishedQParamsError(
                        f"Non-weight source mapping unexpectedly publishes A8 roles for {weight_key!r}."
                    )
                continue
            stem = mapping["source_key"][: -len(".weight")]
            present_roles = 0
            for role in _ROLES:
                source_role_key = f"{stem}.{role}_activation_scale"
                contract_raw = _contract_bytes(entry, role, weight_key)
                source_present = source_role_key in source_keys
                if source_present != (contract_raw is not None):
                    raise PublishedQParamsError(
                        f"Published/contract {role} A8 presence differs for {weight_key!r}."
                    )
                if not source_present:
                    continue
                tensor = handle.get_tensor(source_role_key)
                if tensor.dtype != np.dtype("float32") or tensor.shape not in ((), (1,)):
                    raise PublishedQParamsError(
                        f"Published {role} A8 scale for {weight_key!r} is not scalar F32."
                    )
                source_raw = np.asarray(tensor, dtype="<f4").reshape(1).tobytes()
                if source_raw != contract_raw:
                    raise PublishedQParamsError(
                        f"Published {role} A8 bytes differ for {weight_key!r}."
                    )
                value = float(np.asarray(tensor).reshape(()))
                scope = _scope(weight_key)
                if not math.isfinite(value) or (
                    scope == "zero_head" and value != 0.0
                ) or (scope != "zero_head" and value <= 0.0):
                    raise PublishedQParamsError(
                        f"Invalid published {role} A8 value for {weight_key!r}: {value}."
                    )
                expected_source_roles.add(source_role_key)
                role_records[role].append(
                    {
                        "source_key": source_role_key,
                        "output_key": weight_key,
                        "f32_le_hex": source_raw.hex(),
                    }
                )
                present_roles += 1
            if present_roles:
                scope = _scope(weight_key)
                if scope is None or present_roles != len(_ROLES):
                    raise PublishedQParamsError(
                        f"Unexpected or incomplete published A8 scope for {weight_key!r}."
                    )
                scope_counts[scope] += 1
        if actual_source_roles != expected_source_roles:
            extra_source_roles = sorted(actual_source_roles - expected_source_roles)
            missing_source_roles = sorted(expected_source_roles - actual_source_roles)
            raise PublishedQParamsError(
                "Packed source contains missing or extra activation-scale tensors "
                "relative to the mapped qparams contract. "
                f"actual_source_roles_count={len(actual_source_roles)}, "
                f"expected_source_roles_count={len(expected_source_roles)}; "
                f"actual_source_roles - expected_source_roles "
                f"(extra_count={len(extra_source_roles)})={json.dumps(extra_source_roles)}; "
                f"expected_source_roles - actual_source_roles "
                f"(missing_count={len(missing_source_roles)})={json.dumps(missing_source_roles)}"
            )

    expected_counts = {
        "mutable": EXPECTED_MUTABLE_A8_WEIGHTS,
        "frozen": EXPECTED_FROZEN_A8_WEIGHTS,
        "zero_head": EXPECTED_ZERO_A8_WEIGHTS,
    }
    if scope_counts != expected_counts:
        raise PublishedQParamsError(
            f"Published A8 scope counts differ: {scope_counts}, expected {expected_counts}."
        )
    role_summary = {
        role: {
            "count": len(records),
            "inventory_sha256": _canonical_sha256(records),
        }
        for role, records in role_records.items()
    }
    return {
        "verified": True,
        "source_safetensors": str(source_path),
        "source_safetensors_sha256": observed_hash,
        "source_hash_verification": hash_mode,
        "seed_manifest_sha256": _sha256_file(manifest_path),
        "qparams_contract_sha256": qparams_report.get("contract_sha256")
        or _sha256_file(contract_path),
        "scaled_weight_mapping_count": len(mapped),
        "a8_weight_scope_counts": scope_counts,
        "a8_scalar_count": sum(item["count"] for item in role_summary.values()),
        "roles": role_summary,
    }
