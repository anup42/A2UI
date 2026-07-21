"""Versioned configuration for GenUI representation quality v4."""

from __future__ import annotations

from pathlib import Path

from ._core import (
    DEFAULT_ATOMIC_WEIGHTS,
    DEFAULT_CAPS,
    DEFAULT_DIMENSION_WEIGHTS,
    REWARD_VERSION,
    RewardConfig,
    load_reward_config,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "genui_metric_v4.yaml"
METRIC_MODES = frozenset({"legacy", "v4", "dual"})


def load_default_reward_config() -> RewardConfig:
    """Load the repository's immutable v4 expert-prior configuration."""
    return load_reward_config(DEFAULT_CONFIG_PATH)


def normalize_metric_mode(value: object, default: str = "dual") -> str:
    mode = str(value or default).strip().casefold()
    if mode not in METRIC_MODES:
        raise ValueError(f"evaluation.metric_version must be one of {sorted(METRIC_MODES)}, got {value!r}")
    return mode


__all__ = [
    "DEFAULT_ATOMIC_WEIGHTS",
    "DEFAULT_CAPS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DIMENSION_WEIGHTS",
    "METRIC_MODES",
    "REWARD_VERSION",
    "RewardConfig",
    "load_default_reward_config",
    "load_reward_config",
    "normalize_metric_mode",
]
