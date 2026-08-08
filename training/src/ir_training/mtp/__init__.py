"""Gemma 4 target-conditioned MTP drafter training and deployment contracts."""

from ir_training.mtp.drafter_contract import (
    GEMMA4_E2B_MTP_MODEL_TYPE,
    OFFICIAL_GEMMA4_E2B_ASSISTANT,
    DrafterWeightSpec,
    deployment_weight_specs,
    source_keys,
)

__all__ = [
    "GEMMA4_E2B_MTP_MODEL_TYPE",
    "OFFICIAL_GEMMA4_E2B_ASSISTANT",
    "DrafterWeightSpec",
    "deployment_weight_specs",
    "source_keys",
]
