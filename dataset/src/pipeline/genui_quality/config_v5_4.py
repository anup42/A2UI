"""Versioned configuration and public result types for metric v5.4."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

from ._core import RewardConfig
from .config_v5_3 import RewardBreakdownV53, RewardConfigV53


REWARD_VERSION_V54 = "5.4.0"
DEFAULT_CONFIG_PATH_V54 = (
    Path(__file__).resolve().parents[3] / "configs" / "genui_metric_v5_4.yaml"
)


@dataclass
class RewardConfigV54(RewardConfigV53):
    accessibility_penalty_alpha: float = 0.03
    raw_fence_utility: float = 0.995
    raw_wrapper_utility: float = 0.990
    raw_extra_json_utility: float = 0.980

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.accessibility_penalty_alpha <= 0.03:
            raise ValueError("accessibility_penalty_alpha must be in [0, 0.03]")
        for name in (
            "raw_fence_utility",
            "raw_wrapper_utility",
            "raw_extra_json_utility",
        ):
            if not 0.0 < float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        render_check: Any = None,
    ) -> "RewardConfigV54":
        base = RewardConfigV53.from_mapping(
            mapping, render_check=render_check
        )
        payload = {
            item.name: getattr(base, item.name)
            for item in fields(RewardConfigV53)
        }
        policies = (
            mapping.get("policies")
            if isinstance(mapping.get("policies"), Mapping)
            else {}
        )
        raw = (
            mapping.get("raw_generation_format")
            if isinstance(mapping.get("raw_generation_format"), Mapping)
            else {}
        )
        payload.update(
            accessibility_penalty_alpha=float(
                policies.get("accessibility_penalty_alpha", 0.03)
            ),
            raw_fence_utility=float(raw.get("fence_utility", 0.995)),
            raw_wrapper_utility=float(raw.get("wrapper_utility", 0.990)),
            raw_extra_json_utility=float(
                raw.get("extra_json_utility", 0.980)
            ),
        )
        return cls(**payload)


@dataclass(frozen=True)
class RewardBreakdownV54(RewardBreakdownV53):
    artifact_quality_0_1: float = 0.0
    artifact_quality_0_100: float = 0.0
    raw_json_envelope: dict[str, Any] = None  # type: ignore[assignment]
    accessibility_conformance: dict[str, Any] = None  # type: ignore[assignment]
    policy_versions: dict[str, str] = None  # type: ignore[assignment]
    performance: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "raw_json_envelope", dict(self.raw_json_envelope or {})
        )
        object.__setattr__(
            self,
            "accessibility_conformance",
            dict(self.accessibility_conformance or {}),
        )
        object.__setattr__(
            self, "policy_versions", dict(self.policy_versions or {})
        )
        object.__setattr__(
            self, "performance", dict(self.performance or {})
        )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for v5.4 configuration") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError("metric v5.4 config root must be an object")
    return value


def load_reward_config_v5_4(
    path: str | Path,
    *,
    render_check: Any = None,
) -> RewardConfigV54:
    return RewardConfigV54.from_mapping(
        _load_mapping(Path(path).resolve()),
        render_check=render_check,
    )


def load_v5_4_reward_config() -> RewardConfigV54:
    return load_reward_config_v5_4(DEFAULT_CONFIG_PATH_V54)


def coerce_reward_config_v5_4(
    config: RewardConfig | RewardConfigV53 | RewardConfigV54 | None,
) -> RewardConfigV54:
    if config is None:
        return load_v5_4_reward_config()
    if isinstance(config, RewardConfigV54):
        return config
    payload = {
        item.name: getattr(config, item.name)
        for item in fields(RewardConfig)
    }
    for item in fields(RewardConfigV53):
        if item.name not in payload and hasattr(config, item.name):
            payload[item.name] = getattr(config, item.name)
    return RewardConfigV54(**payload)


__all__ = [
    "DEFAULT_CONFIG_PATH_V54",
    "REWARD_VERSION_V54",
    "RewardBreakdownV54",
    "RewardConfigV54",
    "coerce_reward_config_v5_4",
    "load_reward_config_v5_4",
    "load_v5_4_reward_config",
]
