"""Fail-closed schema and Android-parity initialization for metric v5.3."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from .candidate_normalization import DEFAULT_STRICT_SCHEMA_PATH
from .identity_v5_3 import DEFAULT_PARITY_VECTOR_PATH
from .source_contract import DEFAULT_SCHEMA_PATH


VALIDATION_POLICY_VERSION = "5.3.0"


class MetricV53InitializationError(RuntimeError):
    """Raised when deterministic v5.3 dependencies cannot be initialized."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MetricV53InitializationError(
            f"Cannot load metric dependency {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise MetricV53InitializationError(
            f"Metric dependency is not an object: {path}"
        )
    return value


@lru_cache(maxsize=1)
def ensure_v5_3_validation_ready() -> tuple[Any, Any, dict[str, Any]]:
    try:
        import jsonschema
    except Exception as exc:
        raise MetricV53InitializationError(
            "GenUI metric v5.3 requires the jsonschema runtime dependency"
        ) from exc
    try:
        strict_schema = _load_object(DEFAULT_STRICT_SCHEMA_PATH)
        contract_schema = _load_object(DEFAULT_SCHEMA_PATH)
        parity = _load_object(DEFAULT_PARITY_VECTOR_PATH)
        if str(parity.get("version") or "") != "2.0.0":
            raise MetricV53InitializationError(
                "Android/Python parity corpus must be version 2.0.0"
            )
        validator_type = jsonschema.Draft202012Validator
        validator_type.check_schema(strict_schema)
        validator_type.check_schema(contract_schema)
        return (
            validator_type(strict_schema),
            validator_type(contract_schema),
            parity,
        )
    except MetricV53InitializationError:
        raise
    except Exception as exc:
        raise MetricV53InitializationError(
            "GenUI metric v5.3 dependency initialization failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "MetricV53InitializationError",
    "VALIDATION_POLICY_VERSION",
    "ensure_v5_3_validation_ready",
]
