"""Versioned configuration and public result types for metric v5.3."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

from ._core import RewardBreakdown, RewardConfig


REWARD_VERSION_V53 = "5.3.0"
DEFAULT_CONFIG_PATH_V53 = (
    Path(__file__).resolve().parents[3] / "configs" / "genui_metric_v5_3.yaml"
)


@dataclass
class RewardConfigV53(RewardConfig):
    """V5.3 work budgets and source-policy controls.

    The inherited v5.2 fields remain available so the capped-weight allocator
    and structural utilities can be shared without changing their semantics.
    """

    max_repeat_work_units: int = 250_000
    max_dynamic_evidence_bytes: int = 16_777_216
    max_expression_evaluations: int = 1_000_000
    max_matching_inputs: int = 10_000
    unsupported_external_additions_policy: str = "penalize"
    low_specificity_policy: str = "diagnostic_only"
    repeat_equivalence_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "max_repeat_work_units",
            "max_dynamic_evidence_bytes",
            "max_expression_evaluations",
            "max_matching_inputs",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.unsupported_external_additions_policy not in {
            "penalize",
            "diagnostic_only",
            "disabled",
        }:
            raise ValueError(
                "unsupported_external_additions_policy must be "
                "penalize, diagnostic_only, or disabled"
            )
        if self.low_specificity_policy not in {
            "diagnostic_only",
            "not_applicable",
            "fail_closed",
        }:
            raise ValueError(
                "low_specificity_policy must be diagnostic_only, "
                "not_applicable, or fail_closed"
            )
        if not 0.0 <= float(self.repeat_equivalence_tolerance) <= 1.0:
            raise ValueError("repeat_equivalence_tolerance must be in [0, 1]")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        render_check: Any = None,
    ) -> "RewardConfigV53":
        base = RewardConfig.from_mapping(mapping, render_check=render_check)
        payload = {
            item.name: getattr(base, item.name)
            for item in fields(RewardConfig)
        }
        limits = (
            mapping.get("evidence_limits")
            if isinstance(mapping.get("evidence_limits"), Mapping)
            else {}
        )
        policies = (
            mapping.get("policies")
            if isinstance(mapping.get("policies"), Mapping)
            else {}
        )
        payload.update(
            max_repeat_work_units=int(
                limits.get("max_repeat_work_units", 250_000)
            ),
            max_dynamic_evidence_bytes=int(
                limits.get("max_dynamic_evidence_bytes", 16_777_216)
            ),
            max_expression_evaluations=int(
                limits.get("max_expression_evaluations", 1_000_000)
            ),
            max_matching_inputs=int(
                limits.get("max_matching_inputs", 10_000)
            ),
            unsupported_external_additions_policy=str(
                policies.get(
                    "unsupported_external_additions", "penalize"
                )
            ),
            low_specificity_policy=str(
                policies.get("low_specificity", "diagnostic_only")
            ),
            repeat_equivalence_tolerance=float(
                policies.get("repeat_equivalence_tolerance", 1e-9)
            ),
        )
        return cls(**payload)


@dataclass(frozen=True)
class RewardBreakdownV53(RewardBreakdown):
    """V5.3 diagnostics added without changing historical result readers."""

    atomic_applicability: dict[str, bool] = None  # type: ignore[assignment]
    evidence_ownership: dict[str, Any] = None  # type: ignore[assignment]
    unsupported_external_additions: dict[str, Any] = None  # type: ignore[assignment]
    contract_semantic_specificity: float = 0.0
    count_only_role_count: int = 0
    dynamic_evidence_certification: dict[str, Any] = None  # type: ignore[assignment]
    reward_pipeline_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atomic_applicability",
            dict(self.atomic_applicability or {}),
        )
        object.__setattr__(
            self,
            "evidence_ownership",
            dict(self.evidence_ownership or {}),
        )
        object.__setattr__(
            self,
            "unsupported_external_additions",
            dict(self.unsupported_external_additions or {}),
        )
        object.__setattr__(
            self,
            "dynamic_evidence_certification",
            dict(self.dynamic_evidence_certification or {}),
        )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency is required in CI
            raise RuntimeError("PyYAML is required for v5.3 configuration") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError("metric v5.3 config root must be an object")
    return value


def load_reward_config_v5_3(
    path: str | Path,
    *,
    render_check: Any = None,
) -> RewardConfigV53:
    return RewardConfigV53.from_mapping(
        _load_mapping(Path(path).resolve()),
        render_check=render_check,
    )


def load_v5_3_reward_config() -> RewardConfigV53:
    return load_reward_config_v5_3(DEFAULT_CONFIG_PATH_V53)


def coerce_reward_config_v5_3(
    config: RewardConfig | RewardConfigV53 | None,
) -> RewardConfigV53:
    """Upgrade a historical base config without mutating it."""

    if config is None:
        return load_v5_3_reward_config()
    if isinstance(config, RewardConfigV53):
        return config
    return RewardConfigV53(
        **{
            item.name: getattr(config, item.name)
            for item in fields(RewardConfig)
        }
    )


__all__ = [
    "DEFAULT_CONFIG_PATH_V53",
    "REWARD_VERSION_V53",
    "RewardBreakdownV53",
    "RewardConfigV53",
    "coerce_reward_config_v5_3",
    "load_reward_config_v5_3",
    "load_v5_3_reward_config",
]
