"""Portable retained-scale contract for Gemma 4 mobile QAT.

The released packed mobile checkpoint contains learned/publicly released scale
tensors in addition to integer codes.  A dequantized BF16 seed does not encode
those scales uniquely, particularly for W2/W4 matrices whose occupied code
range is asymmetric.  This module verifies and loads the hash-bound sidecar
created alongside the dense training seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root
from ir_training.qat.mobile_training_seed import (
    OFFICIAL_MOBILE_MODEL_ID,
    OFFICIAL_MOBILE_REVISION,
    OFFICIAL_MOBILE_SAFETENSORS_SHA256,
)

EXPECTED_QPARAM_TENSOR_COUNT = 278


class MobileQParamsError(RuntimeError):
    """Raised when retained mobile quantization state is absent or altered."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_inventory_sha256(inventory: Any) -> str | None:
    if not isinstance(inventory, dict):
        return None
    try:
        canonical = json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical).hexdigest()


def _safe_relative_file(directory: Path, value: Any) -> Path | None:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (directory / relative).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate


def verify_mobile_qparams_contract(
    contract_path: str | Path | None,
    *,
    base: str | Path | None = None,
) -> dict[str, Any]:
    """Verify contract identity, inventory, sidecar size/hash and tensor schema."""

    anchor = Path(base).resolve() if base is not None else training_root()
    path = (
        resolve_path(contract_path, anchor)
        if contract_path is not None and str(contract_path).strip()
        else None
    )
    checks = {
        "contract_present": False,
        "contract_schema": False,
        "source_identity": False,
        "inventory_count": False,
        "inventory_unique": False,
        "inventory_digest": False,
        "inventory_numeric_schema": False,
        "scale_storage_present": False,
        "scale_storage_size": False,
        "scale_storage_sha256": False,
        "scale_storage_tensor_inventory": False,
        "no_private_recipe_claim": False,
    }
    report: dict[str, Any] = {
        "required": True,
        "path": str(path) if path else None,
        "checks": checks,
        "verified": False,
    }
    if path is None or not path.is_file():
        report["error"] = "Retained mobile qparams contract is missing."
        return report
    checks["contract_present"] = True
    report["contract_sha256"] = _sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report["error"] = f"Could not read retained mobile qparams contract: {exc}"
        return report
    if not isinstance(payload, dict):
        report["error"] = "Retained mobile qparams contract root is not an object."
        return report

    inventory = payload.get("inventory")
    inventory = inventory if isinstance(inventory, dict) else {}
    declared_count = int(payload.get("tensor_count", 0) or 0)
    checks["contract_schema"] = bool(
        int(payload.get("contract_version", 0) or 0) == 1
        and payload.get("contract_type") == "gemma4_mobile_retained_qparams"
    )
    checks["source_identity"] = bool(
        payload.get("source_model_id") == OFFICIAL_MOBILE_MODEL_ID
        and payload.get("source_revision") == OFFICIAL_MOBILE_REVISION
        and str(payload.get("source_safetensors_sha256") or "").lower()
        == OFFICIAL_MOBILE_SAFETENSORS_SHA256
    )
    checks["inventory_count"] = bool(
        declared_count == len(inventory) == EXPECTED_QPARAM_TENSOR_COUNT
    )
    checks["inventory_unique"] = bool(
        len(inventory) == len(set(str(key) for key in inventory))
    )
    observed_inventory_digest = _canonical_inventory_sha256(inventory)
    checks["inventory_digest"] = bool(
        observed_inventory_digest
        and observed_inventory_digest == payload.get("inventory_sha256")
    )

    numeric_schema_ok = True
    for weight_key, raw_entry in inventory.items():
        if not isinstance(raw_entry, dict):
            numeric_schema_ok = False
            break
        try:
            bits = int(raw_entry.get("bits", 0))
            weight_shape = tuple(int(value) for value in raw_entry["weight_shape"])
            scale_shape = tuple(int(value) for value in raw_entry["scale_shape"])
            axis = int(raw_entry.get("axis", -1))
            group_value = raw_entry.get("group_size")
            group_size = int(group_value) if group_value is not None else None
            zero_point = int(raw_entry.get("zero_point", 1))
        except (KeyError, TypeError, ValueError):
            numeric_schema_ok = False
            break
        shape_schema_ok = bool(
            len(weight_shape) == 2
            and len(scale_shape) == 2
            and all(value > 0 for value in weight_shape + scale_shape)
            and (group_size is None or group_size > 0)
        )
        expected_scale_columns = (
            1
            if group_size is None
            else (
                weight_shape[1] // group_size
                if shape_schema_ok and weight_shape[1] % group_size == 0
                else -1
            )
        )
        numeric_schema_ok = bool(
            str(weight_key).endswith(".weight")
            and raw_entry.get("scale_tensor") == weight_key
            and bits in {2, 4, 8}
            and shape_schema_ok
            and scale_shape[0] == weight_shape[0]
            and scale_shape[1] == expected_scale_columns
            and axis == 0
            and zero_point == 0
            and raw_entry.get("symmetric") is True
            and raw_entry.get("signed_range")
            == ("narrow" if bits == 8 else "full")
        )
        for role in ("input", "output"):
            raw_hex = raw_entry.get(f"{role}_activation_scale_f32_le_hex")
            if raw_hex is None:
                continue
            try:
                raw = bytes.fromhex(str(raw_hex))
                value = struct.unpack("<f", raw)[0]
            except (ValueError, struct.error):
                numeric_schema_ok = False
                break
            if not math.isfinite(value) or value < 0:
                numeric_schema_ok = False
                break
        if not numeric_schema_ok:
            break
    checks["inventory_numeric_schema"] = numeric_schema_ok

    storage = payload.get("scale_storage")
    storage = storage if isinstance(storage, dict) else {}
    storage_path = _safe_relative_file(path.parent, storage.get("path"))
    checks["scale_storage_present"] = bool(storage_path and storage_path.is_file())
    observed_size = storage_path.stat().st_size if storage_path and storage_path.is_file() else None
    try:
        expected_size = int(storage.get("size_bytes", -1))
    except (TypeError, ValueError):
        expected_size = -1
    checks["scale_storage_size"] = bool(
        observed_size is not None and observed_size == expected_size
    )
    expected_hash = str(storage.get("sha256") or "").lower()
    observed_hash = (
        _sha256_file(storage_path)
        if storage_path and storage_path.is_file()
        else None
    )
    checks["scale_storage_sha256"] = bool(
        len(expected_hash) == 64 and observed_hash == expected_hash
    )

    storage_keys: set[str] = set()
    storage_shapes: dict[str, tuple[int, ...]] = {}
    if storage_path and storage_path.is_file():
        try:
            from safetensors import safe_open

            with safe_open(str(storage_path), framework="pt", device="cpu") as handle:
                storage_keys = set(handle.keys())
                for key in storage_keys:
                    tensor = handle.get_tensor(key)
                    storage_shapes[key] = tuple(int(value) for value in tensor.shape)
                    if not bool(tensor.is_floating_point()):
                        numeric_schema_ok = False
                        break
                    values = tensor.float()
                    if not bool(values.isfinite().all()) or not bool((values > 0).all()):
                        numeric_schema_ok = False
                        break
        except Exception as exc:
            report["scale_storage_error"] = repr(exc)
            storage_keys = set()
    checks["inventory_numeric_schema"] = bool(
        checks["inventory_numeric_schema"] and numeric_schema_ok
    )
    checks["scale_storage_tensor_inventory"] = bool(
        storage_keys == set(inventory)
        and all(
            storage_shapes.get(key)
            == tuple(int(value) for value in entry.get("scale_shape", []))
            for key, entry in inventory.items()
            if isinstance(entry, dict)
        )
    )
    checks["no_private_recipe_claim"] = bool(
        payload.get("training_executed") is False
        and payload.get("private_google_recipe_recovered") is False
        and payload.get("zero_points_are_all_zero") is True
    )
    report.update(
        {
            "scale_storage_path": str(storage_path) if storage_path else None,
            "scale_storage_sha256": observed_hash,
            "tensor_count": len(inventory),
            "inventory_sha256": observed_inventory_digest,
            "inventory": inventory,
            "verified": all(checks.values()),
        }
    )
    return report


class MobileQParams:
    """Verified, lazy retained-scale provider keyed by dense checkpoint weight."""

    def __init__(self, contract_path: str | Path, *, base: str | Path | None = None):
        report = verify_mobile_qparams_contract(contract_path, base=base)
        if not report.get("verified"):
            failed = [
                name for name, passed in report.get("checks", {}).items() if not passed
            ]
            raise MobileQParamsError(
                "Retained mobile qparams contract failed verification: "
                + ", ".join(failed)
            )
        self.report = report
        self.path = Path(str(report["path"]))
        self.storage_path = Path(str(report["scale_storage_path"]))
        self.inventory: dict[str, dict[str, Any]] = dict(report["inventory"])
        self._cpu_cache: dict[str, Any] = {}

    @property
    def contract_sha256(self) -> str:
        return str(self.report["contract_sha256"])

    @property
    def scale_storage_sha256(self) -> str:
        return str(self.report["scale_storage_sha256"])

    def resolve_weight_key(self, candidates: list[str] | tuple[str, ...]) -> str | None:
        for candidate in candidates:
            if candidate in self.inventory:
                return candidate
        return None

    def trainable_projection_weight_keys(self) -> tuple[str, ...]:
        """Return the exact seven-projection-per-layer LoRA deployment scope."""

        suffixes = (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
        keys = tuple(
            sorted(
                key
                for key in self.inventory
                if key.startswith("model.layers.")
                and key.endswith(suffixes)
                and self.activation_scale(key, "input") is not None
                and self.activation_scale(key, "output") is not None
            )
        )
        return keys

    def load_scale(
        self,
        weight_key: str,
        *,
        weight_shape: tuple[int, ...],
        bits: int,
        group_size: int | None,
    ) -> Any:
        entry = self.inventory.get(weight_key)
        if not isinstance(entry, dict):
            raise MobileQParamsError(
                f"No retained qparams entry exists for {weight_key!r}."
            )
        observed_weight_shape = tuple(int(value) for value in entry["weight_shape"])
        observed_group = entry.get("group_size")
        observed_group = int(observed_group) if observed_group is not None else None
        if observed_weight_shape != tuple(weight_shape):
            raise MobileQParamsError(
                f"Retained qparams weight shape for {weight_key!r} is "
                f"{observed_weight_shape}, expected {tuple(weight_shape)}."
            )
        if int(entry["bits"]) != int(bits):
            raise MobileQParamsError(
                f"Retained qparams bit width for {weight_key!r} is "
                f"W{entry['bits']}, expected W{bits}."
            )
        if observed_group != group_size:
            raise MobileQParamsError(
                f"Retained qparams group size for {weight_key!r} is "
                f"{observed_group}, expected {group_size}."
            )
        cached = self._cpu_cache.get(weight_key)
        if cached is None:
            try:
                from safetensors import safe_open

                with safe_open(
                    str(self.storage_path), framework="pt", device="cpu"
                ) as handle:
                    cached = handle.get_tensor(str(entry["scale_tensor"])).float()
            except Exception as exc:
                raise MobileQParamsError(
                    f"Could not load retained scale for {weight_key!r}: {exc}"
                ) from exc
            if not bool(cached.isfinite().all()) or not bool((cached > 0).all()):
                raise MobileQParamsError(
                    f"Retained scale for {weight_key!r} is non-finite or non-positive."
                )
            self._cpu_cache[weight_key] = cached
        return cached

    def activation_scale(self, weight_key: str, role: str) -> float | None:
        """Return the exact published scalar A8 scale for input or output."""

        normalized_role = str(role).strip().lower()
        if normalized_role not in {"input", "output"}:
            raise ValueError("Activation scale role must be 'input' or 'output'.")
        entry = self.inventory.get(weight_key)
        if not isinstance(entry, dict):
            raise MobileQParamsError(
                f"No retained qparams entry exists for {weight_key!r}."
            )
        raw_hex = entry.get(
            f"{normalized_role}_activation_scale_f32_le_hex"
        )
        if raw_hex is None:
            return None
        try:
            raw = bytes.fromhex(str(raw_hex))
            value = struct.unpack("<f", raw)[0]
        except (ValueError, struct.error) as exc:
            raise MobileQParamsError(
                f"Invalid retained {normalized_role} activation scale for "
                f"{weight_key!r}."
            ) from exc
        if not math.isfinite(value) or value <= 0:
            return None
        return float(value)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "retained_mobile",
            "contract_path": str(self.path),
            "contract_sha256": self.contract_sha256,
            "scale_storage_path": str(self.storage_path),
            "scale_storage_sha256": self.scale_storage_sha256,
            "tensor_count": len(self.inventory),
            "inventory_sha256": self.report.get("inventory_sha256"),
            "verified": True,
        }
