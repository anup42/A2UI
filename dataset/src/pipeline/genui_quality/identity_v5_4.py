"""Immutable score and GRPO-pipeline identity for metric v5.4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..flat_spec_semantics import (
    RENDERER_SEMANTICS_VERSION,
    renderer_reference_inventory_hash,
)
from ..renderer_effective_semantics_v5_4 import (
    COMPONENT_CONTRACT_POLICY_VERSION,
    RENDERER_EFFECTIVE_SEMANTICS_VERSION,
    renderer_effective_semantics_hash,
)
from .candidate_normalization import DEFAULT_STRICT_SCHEMA_PATH
from .candidate_normalization_v5_4 import (
    NORMALIZATION_POLICY_VERSION_V54,
    RAW_ENVELOPE_POLICY_VERSION,
)
from .config_v5_4 import REWARD_VERSION_V54, RewardConfigV54
from .evidence_v5_4 import (
    BUILTIN_COMPUTED_MANIFEST_HASH,
    COMPUTED_REGISTRY_IDENTITY_VERSION,
    DYNAMIC_PARITY_VECTOR_VERSION,
    DYNAMIC_SEMANTICS_VERSION,
    EVIDENCE_POLICY_VERSION,
    ComputedFunctionRegistryV54,
    effective_registry_identity_v5_4,
)
from .identity_v5_2 import (
    canonical_json,
    resolved_config_mapping,
    semantic_file_sha256,
    sha256_json,
)
from .metrics_v5_4 import (
    ROLE_MATCHING_POLICY_VERSION,
    SCORING_POLICY_VERSION,
)
from .ownership_v5_4 import OWNERSHIP_POLICY_VERSION
from .source_contract import DEFAULT_SCHEMA_PATH as EXPECTED_SCHEMA_PATH
from .source_contract_v5_4 import (
    SOURCE_EXTRACTION_BENCHMARK_VERSION,
    V5_4_CONTRACT_VERSION,
    V5_4_EXTRACTOR_VERSION,
    V5_4_EXTRACTION_POLICY_VERSION,
    V5_4_MIGRATION_POLICY_VERSION,
)


METRIC_NAME = "GenUI Representation Quality"
ALGORITHM_VERSION = REWARD_VERSION_V54
REPORTING_POLICY_VERSION = "5.4.0"
SOURCE_MANIFEST_VERSION = "3.0.0"
REWARD_PIPELINE_POLICY_VERSION = "2.0.0"

_PACKAGE_DIR = Path(__file__).resolve().parent
_DATASET_DIR = _PACKAGE_DIR.parents[2]
_REPO_ROOT = _DATASET_DIR.parent
DEFAULT_PARITY_VECTOR_PATH_V54 = (
    _DATASET_DIR
    / "tests"
    / "fixtures"
    / "flat_expr_parity_vectors_v5_4.json"
)
_SCORE_FILES = (
    "_core.py",
    "_v5.py",
    "_v5_3.py",
    "_v5_4.py",
    "applicability_v5_3.py",
    "candidate_normalization.py",
    "candidate_normalization_v5_4.py",
    "config_v5_4.py",
    "evidence_v5_1.py",
    "evidence_v5_2.py",
    "evidence_v5_3.py",
    "evidence_v5_4.py",
    "identity_v5_4.py",
    "matching_v5_1.py",
    "matching_v5_2.py",
    "matching_v5_3.py",
    "metrics_v5_2.py",
    "metrics_v5_3.py",
    "metrics_v5_4.py",
    "ownership_v5_4.py",
    "source_contract.py",
    "source_contract_v5_3.py",
    "source_contract_v5_4.py",
    "structure_v5_1.py",
    "validation_v5_3.py",
    "../flat_spec_contract.py",
    "../flat_spec_semantics.py",
    "../renderer_effective_semantics_v5_4.py",
)
_REWARD_FILES = (
    "grpo_reward_v5_4.py",
    "__init__.py",
    "../../../../training/scripts/train_grpo.py",
)
_METRIC_CACHE: dict[str, str] = {}
_PIPELINE_CACHE: dict[str, str] = {}


def _manifest(
    files: tuple[str, ...],
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    supplied = dict(overrides or {})
    return {
        relative: (
            str(supplied[relative])
            if relative in supplied
            else semantic_file_sha256((_PACKAGE_DIR / relative).resolve())
        )
        for relative in files
    }


def score_source_manifest_v5_4(
    *,
    source_hash_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return _manifest(_SCORE_FILES, source_hash_overrides)


def metric_fingerprint_v5_4(
    config: RewardConfigV54,
    *,
    computed_registry: ComputedFunctionRegistryV54 | None = None,
    source_hash_overrides: Mapping[str, str] | None = None,
    parity_vector_hash_override: str | None = None,
) -> str:
    registry = effective_registry_identity_v5_4(computed_registry)
    cache_key = sha256_json(
        {
            "config": resolved_config_mapping(config),
            "registry": registry,
            "overrides": dict(source_hash_overrides or {}),
            "parity": parity_vector_hash_override,
        }
    )
    if cache_key in _METRIC_CACHE:
        return _METRIC_CACHE[cache_key]
    manifest = score_source_manifest_v5_4(
        source_hash_overrides=source_hash_overrides
    )
    android = (
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
    prompt = (
        _DATASET_DIR
        / "prompts"
        / "genui_gen_mobile_flatspec_v11.md"
    )
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "source_manifest_version": SOURCE_MANIFEST_VERSION,
        "score_source_manifest": manifest,
        "score_source_manifest_hash": sha256_json(manifest),
        "config": resolved_config_mapping(config),
        "config_hash": semantic_file_sha256(
            (_DATASET_DIR / "configs" / "genui_metric_v5_4.yaml").resolve()
        ),
        "strict_schema_hash": semantic_file_sha256(
            DEFAULT_STRICT_SCHEMA_PATH.resolve()
        ),
        "expected_contract_schema_hash": semantic_file_sha256(
            EXPECTED_SCHEMA_PATH.resolve()
        ),
        "prompt_contract_hash": semantic_file_sha256(prompt.resolve()),
        "android_renderer_hash": semantic_file_sha256(android.resolve()),
        "renderer_reference_semantics": {
            "version": RENDERER_SEMANTICS_VERSION,
            "hash": renderer_reference_inventory_hash(),
        },
        "renderer_effective_semantics": {
            "version": RENDERER_EFFECTIVE_SEMANTICS_VERSION,
            "hash": renderer_effective_semantics_hash(),
        },
        "dynamic_parity": {
            "version": DYNAMIC_PARITY_VECTOR_VERSION,
            "hash": (
                str(parity_vector_hash_override)
                if parity_vector_hash_override is not None
                else semantic_file_sha256(
                    DEFAULT_PARITY_VECTOR_PATH_V54.resolve()
                )
            ),
        },
        "computed_registry": registry,
        "builtin_computed_manifest_hash": BUILTIN_COMPUTED_MANIFEST_HASH,
        "policies": {
            "ownership": OWNERSHIP_POLICY_VERSION,
            "renderer_effective": RENDERER_EFFECTIVE_SEMANTICS_VERSION,
            "component_contract": COMPONENT_CONTRACT_POLICY_VERSION,
            "role_matching": ROLE_MATCHING_POLICY_VERSION,
            "scoring": SCORING_POLICY_VERSION,
            "evidence": EVIDENCE_POLICY_VERSION,
            "dynamic": DYNAMIC_SEMANTICS_VERSION,
            "source_contract": V5_4_CONTRACT_VERSION,
            "source_extractor": V5_4_EXTRACTOR_VERSION,
            "source_extraction": V5_4_EXTRACTION_POLICY_VERSION,
            "source_migration": V5_4_MIGRATION_POLICY_VERSION,
            "source_benchmark": SOURCE_EXTRACTION_BENCHMARK_VERSION,
            "normalization": NORMALIZATION_POLICY_VERSION_V54,
            "raw_envelope": RAW_ENVELOPE_POLICY_VERSION,
            "computed_registry_identity": (
                COMPUTED_REGISTRY_IDENTITY_VERSION
            ),
            "reporting": REPORTING_POLICY_VERSION,
        },
    }
    result = sha256_json(payload)
    _METRIC_CACHE[cache_key] = result
    return result


def reward_pipeline_fingerprint_v5_4(
    metric_fingerprint: str,
    *,
    source_hash_overrides: Mapping[str, str] | None = None,
) -> str:
    cache_key = sha256_json(
        {
            "metric_fingerprint": metric_fingerprint,
            "overrides": dict(source_hash_overrides or {}),
        }
    )
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]
    manifest = _manifest(_REWARD_FILES, source_hash_overrides)
    result = sha256_json(
        {
            "policy_version": REWARD_PIPELINE_POLICY_VERSION,
            "metric_fingerprint": metric_fingerprint,
            "source_manifest": manifest,
            "source_manifest_hash": sha256_json(manifest),
            "reward_dispatch": "generation_reward_v5_4",
            "input_normalization": "field_specific_scalar_vector_contract",
            "source_grouping": "stable_source_identity_groups",
        }
    )
    _PIPELINE_CACHE[cache_key] = result
    return result


def expected_contract_hash_v5_4(contract: Mapping[str, Any]) -> str:
    return sha256_json(dict(contract))


__all__ = [
    "ALGORITHM_VERSION",
    "DEFAULT_PARITY_VECTOR_PATH_V54",
    "METRIC_NAME",
    "REPORTING_POLICY_VERSION",
    "REWARD_PIPELINE_POLICY_VERSION",
    "SOURCE_MANIFEST_VERSION",
    "canonical_json",
    "expected_contract_hash_v5_4",
    "metric_fingerprint_v5_4",
    "reward_pipeline_fingerprint_v5_4",
    "score_source_manifest_v5_4",
]
