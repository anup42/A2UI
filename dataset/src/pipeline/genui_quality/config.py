"""Versioned configuration for GenUI representation quality v5.x."""

from __future__ import annotations

from pathlib import Path

from ._core import (
    DEFAULT_ATOMIC_WEIGHTS,
    DEFAULT_CAPS,
    DEFAULT_DIMENSION_WEIGHTS,
    REWARD_VERSION as V4_REWARD_VERSION,
    RewardConfig,
    load_reward_config,
)


V5_REWARD_VERSION = "5.0.0"
V5_1_REWARD_VERSION = "5.1.0"
REWARD_VERSION = "5.2.0"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "genui_metric_v5_2.yaml"
)
V5_1_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "genui_metric_v5_1.yaml"
)
V5_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "genui_metric_v5.yaml"
V4_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "genui_metric_v4.yaml"
METRIC_MODES = frozenset(
    {
        "legacy",
        "v4",
        "v5",
        "v5_0",
        "v5_1",
        "v5_2",
        "v5_3",
        "v5_4",
        "dual",
    }
)


def load_default_reward_config() -> RewardConfig:
    """Load the repository's current immutable v5.2 expert-prior configuration."""
    return load_reward_config(DEFAULT_CONFIG_PATH)


def load_v5_1_reward_config() -> RewardConfig:
    """Load the retained v5.1 expert-prior configuration."""
    return load_reward_config(V5_1_CONFIG_PATH)


def load_v5_reward_config() -> RewardConfig:
    """Load the retained v5.0 expert-prior configuration."""
    return load_reward_config(V5_CONFIG_PATH)


def load_v4_reward_config() -> RewardConfig:
    """Load the retained v4 configuration for explicit historical comparison."""
    return load_reward_config(V4_CONFIG_PATH)


def normalize_metric_mode(value: object, default: str = "v5_4") -> str:
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
    "V5_1_CONFIG_PATH",
    "V5_1_REWARD_VERSION",
    "V4_CONFIG_PATH",
    "V4_REWARD_VERSION",
    "V5_CONFIG_PATH",
    "V5_REWARD_VERSION",
    "RewardConfig",
    "load_default_reward_config",
    "load_reward_config",
    "load_v5_1_reward_config",
    "load_v5_reward_config",
    "load_v4_reward_config",
    "normalize_metric_mode",
]
