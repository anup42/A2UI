"""Immutable policy and source identity for GenUI metric v5.1."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..flat_spec_semantics import (
    RENDERER_SEMANTICS_VERSION,
    renderer_reference_inventory_hash,
)
from .candidate_normalization import (
    DEFAULT_STRICT_SCHEMA_PATH,
    NORMALIZATION_POLICY_VERSION,
)
from .config import V5_1_REWARD_VERSION as REWARD_VERSION, RewardConfig
from .evidence_v5_1 import (
    DYNAMIC_SEMANTICS_VERSION,
    EVIDENCE_POLICY_VERSION,
)
from .matching_v5_1 import MATCHING_POLICY_VERSION
from .metrics_v5_1 import SCORING_POLICY_VERSION
from .source_contract import (
    DEFAULT_SCHEMA_PATH as EXPECTED_CONTRACT_SCHEMA_PATH,
    V5_1_CONTRACT_VERSION,
    V5_1_EXTRACTOR_VERSION,
    V5_1_EXTRACTION_POLICY_VERSION,
)
from .structure_v5_1 import STRUCTURE_POLICY_VERSION
from .validation_v5_1 import VALIDATION_POLICY_VERSION


METRIC_NAME = "GenUI Representation Quality"
REPORTING_POLICY_VERSION = "5.1.0"
ALGORITHM_VERSION = REWARD_VERSION
_SOURCE_MODULES = (
    "_v5_1.py",
    "metrics_v5_1.py",
    "evidence_v5_1.py",
    "matching_v5_1.py",
    "structure_v5_1.py",
    "validation_v5_1.py",
    "candidate_normalization.py",
    "source_contract.py",
    "../flat_spec_semantics.py",
)


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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved_config_mapping(config: RewardConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in fields(config):
        if item.name in {
            "render_check",
            "max_matching_hungarian_work",
            "max_matching_sparse_relaxations",
        }:
            continue
        value = getattr(config, item.name)
        if isinstance(value, Mapping):
            result[item.name] = {
                str(key): (
                    {
                        str(nested_key): float(nested_value)
                        for nested_key, nested_value in nested.items()
                    }
                    if isinstance(nested, Mapping)
                    else float(nested)
                    if isinstance(nested, (int, float))
                    and not isinstance(nested, bool)
                    else nested
                )
                for key, nested in value.items()
            }
        else:
            result[item.name] = value
    return result


def score_source_manifest() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    result: dict[str, str] = {}
    for relative in _SOURCE_MODULES:
        path = (base / relative).resolve()
        result[relative] = file_sha256(path)
    return result


def score_source_manifest_hash() -> str:
    return sha256_json(score_source_manifest())


def metric_fingerprint_v5_1(config: RewardConfig) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "matching_policy_version": MATCHING_POLICY_VERSION,
        "evidence_policy_version": EVIDENCE_POLICY_VERSION,
        "dynamic_semantics_version": DYNAMIC_SEMANTICS_VERSION,
        "structure_policy_version": STRUCTURE_POLICY_VERSION,
        "validation_policy_version": VALIDATION_POLICY_VERSION,
        "reporting_policy_version": REPORTING_POLICY_VERSION,
        "score_source_manifest_hash": score_source_manifest_hash(),
        "config": resolved_config_mapping(config),
        "strict_flat_spec_schema_hash": file_sha256(
            DEFAULT_STRICT_SCHEMA_PATH.resolve()
        ),
        "expected_contract_schema_hash": file_sha256(
            EXPECTED_CONTRACT_SCHEMA_PATH.resolve()
        ),
        "renderer_semantics_version": RENDERER_SEMANTICS_VERSION,
        "renderer_reference_inventory_hash": renderer_reference_inventory_hash(),
        "contract_version": V5_1_CONTRACT_VERSION,
        "extractor_version": V5_1_EXTRACTOR_VERSION,
        "extraction_policy_version": V5_1_EXTRACTION_POLICY_VERSION,
        "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
    }
    return sha256_json(payload)


def expected_contract_hash_v5_1(contract: Mapping[str, Any]) -> str:
    return sha256_json(dict(contract))


__all__ = [
    "ALGORITHM_VERSION",
    "METRIC_NAME",
    "REPORTING_POLICY_VERSION",
    "canonical_json",
    "expected_contract_hash_v5_1",
    "file_sha256",
    "metric_fingerprint_v5_1",
    "resolved_config_mapping",
    "score_source_manifest",
    "score_source_manifest_hash",
    "sha256_json",
]
