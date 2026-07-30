"""Immutable metric and reward-pipeline identity for GenUI v5.3."""

from __future__ import annotations

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
from .config_v5_3 import REWARD_VERSION_V53, RewardConfigV53
from .evidence_v5_3 import (
    BUILTIN_COMPUTED_MANIFEST_HASH,
    DYNAMIC_PARITY_VECTOR_VERSION,
    DYNAMIC_SEMANTICS_VERSION,
    EVIDENCE_POLICY_VERSION,
    ComputedFunctionRegistryV53,
    effective_registry_identity_v5_3,
)
from .identity_v5_2 import (
    canonical_json,
    resolved_config_mapping,
    semantic_file_sha256,
    sha256_json,
)
from .matching_v5_3 import MATCHING_POLICY_VERSION
from .metrics_v5_3 import SCORING_POLICY_VERSION
from .source_contract import DEFAULT_SCHEMA_PATH as EXPECTED_SCHEMA_PATH
from .source_contract_v5_3 import (
    V5_3_CONTRACT_VERSION,
    V5_3_EXTRACTOR_VERSION,
    V5_3_EXTRACTION_POLICY_VERSION,
    V5_3_MIGRATION_POLICY_VERSION,
)
from .applicability_v5_3 import APPLICABILITY_POLICY_VERSION


METRIC_NAME = "GenUI Representation Quality"
ALGORITHM_VERSION = REWARD_VERSION_V53
REPORTING_POLICY_VERSION = "5.3.0"
SOURCE_MANIFEST_VERSION = "2.0.0"
REWARD_PIPELINE_POLICY_VERSION = "1.0.0"
_PACKAGE_DIR = Path(__file__).resolve().parent
_DATASET_DIR = _PACKAGE_DIR.parents[2]
_REPO_ROOT = _DATASET_DIR.parent
DEFAULT_PARITY_VECTOR_PATH = (
    _DATASET_DIR / "tests" / "fixtures" / "flat_expr_parity_vectors.json"
)
_METRIC_SOURCE_FILES = (
    "_core.py",
    "_v5.py",
    "_v5_3.py",
    "applicability_v5_3.py",
    "config_v5_3.py",
    "evidence_v5_3.py",
    "identity_v5_3.py",
    "matching_v5_3.py",
    "metrics_v5_3.py",
    "source_contract_v5_3.py",
    "validation_v5_3.py",
    "candidate_normalization.py",
    "graph.py",
    "structure_v5_1.py",
    # Explicit transitive reused semantics.
    "evidence_v5_1.py",
    "evidence_v5_2.py",
    "matching_v5_1.py",
    "matching_v5_2.py",
    "metrics_v5_2.py",
    "source_contract.py",
    "../flat_spec_contract.py",
    "../flat_spec_semantics.py",
)
_REWARD_PIPELINE_FILES = (
    "grpo_reward.py",
    "__init__.py",
    "../../../../training/scripts/train_grpo.py",
)
_METRIC_FINGERPRINT_CACHE: dict[str, str] = {}
_REWARD_PIPELINE_FINGERPRINT_CACHE: dict[str, str] = {}


def _manifest(
    relative_files: tuple[str, ...],
    *,
    source_hash_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    overrides = dict(source_hash_overrides or {})
    result: dict[str, str] = {}
    for relative in relative_files:
        if relative in overrides:
            result[relative] = str(overrides[relative])
            continue
        path = (_PACKAGE_DIR / relative).resolve()
        result[relative] = semantic_file_sha256(path)
    return result


def score_source_manifest_v5_3(
    *,
    source_hash_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return _manifest(
        _METRIC_SOURCE_FILES,
        source_hash_overrides=source_hash_overrides,
    )


def metric_fingerprint_v5_3(
    config: RewardConfigV53,
    *,
    computed_registry: ComputedFunctionRegistryV53 | None = None,
    source_hash_overrides: Mapping[str, str] | None = None,
    parity_vector_hash_override: str | None = None,
) -> str:
    cache_key = sha256_json(
        {
            "config": resolved_config_mapping(config),
            "computed_registry": effective_registry_identity_v5_3(
                computed_registry
            ),
            "source_hash_overrides": dict(source_hash_overrides or {}),
            "parity_vector_hash_override": parity_vector_hash_override,
        }
    )
    cached = _METRIC_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    source_manifest = score_source_manifest_v5_3(
        source_hash_overrides=source_hash_overrides
    )
    parity_hash = (
        str(parity_vector_hash_override)
        if parity_vector_hash_override is not None
        else semantic_file_sha256(DEFAULT_PARITY_VECTOR_PATH.resolve())
    )
    android_renderer = (
        _REPO_ROOT
        / "android"
        / "app"
        / "src"
        / "main"
        / "java"
        / "com"
        / "samsung"
        / "genuicraft"
        / "renderer"
        / "FlatSpecRenderer.kt"
    )
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "source_manifest_version": SOURCE_MANIFEST_VERSION,
        "score_source_manifest": source_manifest,
        "score_source_manifest_hash": sha256_json(source_manifest),
        "config": resolved_config_mapping(config),
        "config_file_hash": semantic_file_sha256(
            (_DATASET_DIR / "configs" / "genui_metric_v5_3.yaml").resolve()
        ),
        "strict_flat_spec_schema_hash": semantic_file_sha256(
            DEFAULT_STRICT_SCHEMA_PATH.resolve()
        ),
        "expected_contract_schema_hash": semantic_file_sha256(
            EXPECTED_SCHEMA_PATH.resolve()
        ),
        "renderer_semantics_version": RENDERER_SEMANTICS_VERSION,
        "renderer_reference_inventory_hash": (
            renderer_reference_inventory_hash()
        ),
        "android_renderer_hash": semantic_file_sha256(android_renderer),
        "android_dynamic_parity_vector_version": (
            DYNAMIC_PARITY_VECTOR_VERSION
        ),
        "android_dynamic_parity_vector_hash": parity_hash,
        "builtin_computed_registry_manifest_hash": (
            BUILTIN_COMPUTED_MANIFEST_HASH
        ),
        "effective_computed_registry": (
            effective_registry_identity_v5_3(computed_registry)
        ),
        "policies": {
            "applicability": APPLICABILITY_POLICY_VERSION,
            "scoring": SCORING_POLICY_VERSION,
            "matching": MATCHING_POLICY_VERSION,
            "evidence": EVIDENCE_POLICY_VERSION,
            "dynamic_semantics": DYNAMIC_SEMANTICS_VERSION,
            "reporting": REPORTING_POLICY_VERSION,
            "normalization": NORMALIZATION_POLICY_VERSION,
            "contract": V5_3_CONTRACT_VERSION,
            "extractor": V5_3_EXTRACTOR_VERSION,
            "extraction": V5_3_EXTRACTION_POLICY_VERSION,
            "migration": V5_3_MIGRATION_POLICY_VERSION,
        },
    }
    result = sha256_json(payload)
    _METRIC_FINGERPRINT_CACHE[cache_key] = result
    return result


def reward_pipeline_fingerprint_v5_3(
    metric_fingerprint: str,
    *,
    source_hash_overrides: Mapping[str, str] | None = None,
) -> str:
    cache_key = sha256_json(
        {
            "metric_fingerprint": metric_fingerprint,
            "source_hash_overrides": dict(source_hash_overrides or {}),
        }
    )
    cached = _REWARD_PIPELINE_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    manifest = _manifest(
        _REWARD_PIPELINE_FILES,
        source_hash_overrides=source_hash_overrides,
    )
    result = sha256_json(
        {
            "policy_version": REWARD_PIPELINE_POLICY_VERSION,
            "metric_fingerprint": metric_fingerprint,
            "source_manifest": manifest,
            "source_manifest_hash": sha256_json(manifest),
            "completion_to_text": "grpo_reward._completion_to_text",
            "input_shape_normalization": (
                "field_specific_scalar_vector_contract_and_asset_rows"
            ),
            "source_grouping": "stable_source_identity_groups",
            "truncation_policy": "training_config_mask_truncated_completions",
            "reward_dispatch": "genui_grpo_reward_v5_3",
        }
    )
    _REWARD_PIPELINE_FINGERPRINT_CACHE[cache_key] = result
    return result


def expected_contract_hash_v5_3(
    contract: Mapping[str, Any],
) -> str:
    return sha256_json(dict(contract))


__all__ = [
    "ALGORITHM_VERSION",
    "DEFAULT_PARITY_VECTOR_PATH",
    "METRIC_NAME",
    "REPORTING_POLICY_VERSION",
    "REWARD_PIPELINE_POLICY_VERSION",
    "SOURCE_MANIFEST_VERSION",
    "canonical_json",
    "expected_contract_hash_v5_3",
    "metric_fingerprint_v5_3",
    "reward_pipeline_fingerprint_v5_3",
    "score_source_manifest_v5_3",
]
