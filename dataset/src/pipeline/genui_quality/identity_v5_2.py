"""Transitive immutable score identity for GenUI metric v5.2."""

from __future__ import annotations

import ast
from dataclasses import fields
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
from .config import REWARD_VERSION, RewardConfig
from .evidence_v5_2 import (
    BUILTIN_COMPUTED_MANIFEST_HASH,
    DYNAMIC_PARITY_VECTOR_VERSION,
    DYNAMIC_SEMANTICS_VERSION,
    EVIDENCE_POLICY_VERSION,
    ComputedFunctionRegistryV52,
    effective_registry_identity,
)
from .matching_v5_2 import MATCHING_POLICY_VERSION
from .metrics_v5_2 import SCORING_POLICY_VERSION
from .source_contract import (
    DEFAULT_SCHEMA_PATH as EXPECTED_CONTRACT_SCHEMA_PATH,
    V5_2_CONTRACT_VERSION,
    V5_2_EXTRACTOR_VERSION,
    V5_2_EXTRACTION_POLICY_VERSION,
    V5_2_MIGRATION_POLICY_VERSION,
)
from .structure_v5_1 import STRUCTURE_POLICY_VERSION
from .validation_v5_2 import VALIDATION_POLICY_VERSION


METRIC_NAME = "GenUI Representation Quality"
REPORTING_POLICY_VERSION = "5.2.0"
ALGORITHM_VERSION = REWARD_VERSION
SOURCE_MANIFEST_VERSION = "1.0.0"
_PACKAGE_DIR = Path(__file__).resolve().parent
_DATASET_DIR = _PACKAGE_DIR.parents[2]
DEFAULT_PARITY_VECTOR_PATH = (
    _DATASET_DIR / "tests" / "fixtures" / "flat_expr_parity_vectors.json"
)
_SOURCE_MODULES = (
    "_core.py",
    "_v5.py",
    "_v5_2.py",
    "metrics_v5_2.py",
    "evidence_v5_2.py",
    "matching_v5_2.py",
    # V5.2 intentionally reuses immutable preparation/interpreter mechanics.
    "matching_v5_1.py",
    "evidence_v5_1.py",
    "structure_v5_1.py",
    "validation_v5_2.py",
    "candidate_normalization.py",
    "source_contract.py",
    "graph.py",
    "config.py",
    "aggregate.py",
    "../flat_spec_contract.py",
    "../flat_spec_semantics.py",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_file_sha256(path: Path) -> str:
    """Hash semantic syntax where a stable parser is available."""

    suffix = path.suffix.casefold()
    raw = path.read_text(encoding="utf-8")
    if suffix == ".py":
        tree = ast.parse(raw, filename=path.name)
        payload = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if suffix == ".json":
        return sha256_json(json.loads(raw))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            pass
        else:
            return sha256_json(yaml.safe_load(raw))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved_config_mapping(config: RewardConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in fields(config):
        if item.name == "render_check":
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


def score_source_manifest_v5_2(
    *,
    source_hash_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    overrides = dict(source_hash_overrides or {})
    result: dict[str, str] = {}
    for relative in _SOURCE_MODULES:
        if relative in overrides:
            result[relative] = str(overrides[relative])
            continue
        path = (_PACKAGE_DIR / relative).resolve()
        result[relative] = semantic_file_sha256(path)
    return result


def score_source_manifest_hash_v5_2(
    *,
    source_hash_overrides: Mapping[str, str] | None = None,
) -> str:
    return sha256_json(
        {
            "manifest_version": SOURCE_MANIFEST_VERSION,
            "files": score_source_manifest_v5_2(
                source_hash_overrides=source_hash_overrides
            ),
        }
    )


def metric_fingerprint_v5_2(
    config: RewardConfig,
    *,
    computed_registry: ComputedFunctionRegistryV52 | None = None,
    source_hash_overrides: Mapping[str, str] | None = None,
    parity_vector_hash_override: str | None = None,
) -> str:
    parity_hash = (
        str(parity_vector_hash_override)
        if parity_vector_hash_override is not None
        else semantic_file_sha256(DEFAULT_PARITY_VECTOR_PATH.resolve())
    )
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "source_manifest_version": SOURCE_MANIFEST_VERSION,
        "score_source_manifest_hash": score_source_manifest_hash_v5_2(
            source_hash_overrides=source_hash_overrides
        ),
        "config": resolved_config_mapping(config),
        "strict_flat_spec_schema_hash": semantic_file_sha256(
            DEFAULT_STRICT_SCHEMA_PATH.resolve()
        ),
        "expected_contract_schema_hash": semantic_file_sha256(
            EXPECTED_CONTRACT_SCHEMA_PATH.resolve()
        ),
        "renderer_semantics_version": RENDERER_SEMANTICS_VERSION,
        "renderer_reference_inventory_hash": (
            renderer_reference_inventory_hash()
        ),
        "android_dynamic_parity_vector_version": (
            DYNAMIC_PARITY_VECTOR_VERSION
        ),
        "android_dynamic_parity_vector_hash": parity_hash,
        "builtin_computed_registry_manifest_hash": (
            BUILTIN_COMPUTED_MANIFEST_HASH
        ),
        "effective_computed_registry": effective_registry_identity(
            computed_registry
        ),
        "policies": {
            "scoring": SCORING_POLICY_VERSION,
            "matching": MATCHING_POLICY_VERSION,
            "evidence": EVIDENCE_POLICY_VERSION,
            "dynamic_semantics": DYNAMIC_SEMANTICS_VERSION,
            "structure": STRUCTURE_POLICY_VERSION,
            "validation": VALIDATION_POLICY_VERSION,
            "reporting": REPORTING_POLICY_VERSION,
            "normalization": NORMALIZATION_POLICY_VERSION,
            "contract": V5_2_CONTRACT_VERSION,
            "extractor": V5_2_EXTRACTOR_VERSION,
            "extraction": V5_2_EXTRACTION_POLICY_VERSION,
            "migration": V5_2_MIGRATION_POLICY_VERSION,
        },
    }
    return sha256_json(payload)


def expected_contract_hash_v5_2(contract: Mapping[str, Any]) -> str:
    return sha256_json(dict(contract))


__all__ = [
    "ALGORITHM_VERSION",
    "DEFAULT_PARITY_VECTOR_PATH",
    "METRIC_NAME",
    "REPORTING_POLICY_VERSION",
    "SOURCE_MANIFEST_VERSION",
    "canonical_json",
    "expected_contract_hash_v5_2",
    "metric_fingerprint_v5_2",
    "resolved_config_mapping",
    "score_source_manifest_hash_v5_2",
    "score_source_manifest_v5_2",
    "semantic_file_sha256",
    "sha256_json",
]
