"""Fail-closed schema dependency initialization for metric v5.1."""

from __future__ import annotations

import importlib
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from .candidate_normalization import DEFAULT_STRICT_SCHEMA_PATH
from .source_contract import DEFAULT_SCHEMA_PATH


VALIDATION_POLICY_VERSION = "5.1.0"


class MetricV51InitializationError(RuntimeError):
    """Raised when deterministic v5.1 validation cannot be initialized."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MetricV51InitializationError(
            f"Cannot load metric schema {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise MetricV51InitializationError(
            f"Metric schema is not an object: {path}"
        )
    return value


@lru_cache(maxsize=1)
def ensure_v5_1_validation_ready() -> tuple[Any, Any]:
    """Import and compile both required Draft 2020-12 validators."""

    try:
        jsonschema = importlib.import_module("jsonschema")
    except Exception as exc:
        raise MetricV51InitializationError(
            "GenUI metric v5.1 requires the jsonschema runtime dependency"
        ) from exc
    try:
        strict_schema = _load(DEFAULT_STRICT_SCHEMA_PATH)
        contract_schema = _load(DEFAULT_SCHEMA_PATH)
        validator_type = jsonschema.Draft202012Validator
        validator_type.check_schema(strict_schema)
        validator_type.check_schema(contract_schema)
        return validator_type(strict_schema), validator_type(contract_schema)
    except MetricV51InitializationError:
        raise
    except Exception as exc:
        raise MetricV51InitializationError(
            "GenUI metric v5.1 schema initialization failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "MetricV51InitializationError",
    "VALIDATION_POLICY_VERSION",
    "ensure_v5_1_validation_ready",
]
