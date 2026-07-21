"""Source-Conditioned Renderer Representation Quality (SCRRQ) v4."""

from .aggregate import (
    RendererSmokeAdapter,
    RewardBreakdown,
    aggregate_v4_records,
    breakdown_from_mapping,
    breakdown_to_mapping,
    score_genui_completion,
    score_record,
    with_renderer_smoke_adapter,
)
from .config import (
    DEFAULT_CONFIG_PATH,
    METRIC_MODES,
    REWARD_VERSION,
    RewardConfig,
    load_default_reward_config,
    load_reward_config,
    normalize_metric_mode,
)
from .grpo_reward import genui_grpo_reward, make_genui_grpo_reward
from .source_contract import (
    CONTRACT_VERSION,
    EXTRACTOR_VERSION,
    SourceContractCache,
    extract_expected_ui_contract,
    resolve_expected_ui_contract,
    source_contract_cache_key,
    validate_expected_ui_contract,
)

__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_CONFIG_PATH",
    "EXTRACTOR_VERSION",
    "METRIC_MODES",
    "REWARD_VERSION",
    "RendererSmokeAdapter",
    "RewardBreakdown",
    "RewardConfig",
    "SourceContractCache",
    "aggregate_v4_records",
    "breakdown_from_mapping",
    "breakdown_to_mapping",
    "extract_expected_ui_contract",
    "genui_grpo_reward",
    "load_default_reward_config",
    "load_reward_config",
    "make_genui_grpo_reward",
    "normalize_metric_mode",
    "resolve_expected_ui_contract",
    "score_genui_completion",
    "score_record",
    "source_contract_cache_key",
    "validate_expected_ui_contract",
    "with_renderer_smoke_adapter",
]
