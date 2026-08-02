"""Immutable GenUI metric v5 identity and fingerprint helpers."""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..renderer_semantics import (
    RENDERER_SEMANTICS_VERSION,
    renderer_reference_inventory_hash,
)
from .candidate_normalization import (
    DEFAULT_STRICT_SCHEMA_PATH,
    NORMALIZATION_POLICY_VERSION,
)
from .config import V5_REWARD_VERSION as REWARD_VERSION, RewardConfig
from .source_contract import (
    CONTRACT_VERSION,
    DEFAULT_SCHEMA_PATH as EXPECTED_CONTRACT_SCHEMA_PATH,
    EXTRACTOR_VERSION,
    EXTRACTION_POLICY_VERSION,
)


METRIC_NAME = "GenUI Representation Quality"
ALGORITHM_VERSION = REWARD_VERSION


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@lru_cache(maxsize=16)
def _file_sha256_at_state(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_sha256(path: str) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _file_sha256_at_state(str(resolved), stat.st_mtime_ns, stat.st_size)


def resolved_config_mapping(config: RewardConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in fields(config):
        if item.name == "render_check":
            continue
        value = getattr(config, item.name)
        if isinstance(value, Mapping):
            result[item.name] = {
                str(key): (
                    {str(k): float(v) for k, v in nested.items()}
                    if isinstance(nested, Mapping)
                    else float(nested)
                    if isinstance(nested, (int, float)) and not isinstance(nested, bool)
                    else nested
                )
                for key, nested in value.items()
            }
        else:
            result[item.name] = value
    return result


def metric_fingerprint(config: RewardConfig) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "config": resolved_config_mapping(config),
        "strict_flat_spec_schema_hash": file_sha256(
            str(DEFAULT_STRICT_SCHEMA_PATH.resolve())
        ),
        "expected_contract_schema_hash": file_sha256(
            str(EXPECTED_CONTRACT_SCHEMA_PATH.resolve())
        ),
        "renderer_semantics_version": RENDERER_SEMANTICS_VERSION,
        "renderer_reference_inventory_hash": renderer_reference_inventory_hash(),
        "contract_version": CONTRACT_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
    }
    return sha256_json(payload)


def expected_contract_hash(contract: Mapping[str, Any]) -> str:
    return sha256_json(dict(contract))


__all__ = [
    "ALGORITHM_VERSION",
    "METRIC_NAME",
    "canonical_json",
    "expected_contract_hash",
    "file_sha256",
    "metric_fingerprint",
    "resolved_config_mapping",
    "sha256_json",
]
