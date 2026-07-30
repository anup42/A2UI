"""Candidate-independent atomic applicability and evidence ownership for v5.3."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from . import _core


APPLICABILITY_POLICY_VERSION = "5.3.0"

_STRUCTURED_DIRECTIVE = re.compile(
    r"(?i)^\s*(?:#{1,6}\s*)?"
    r"(?:chart|graph|plot|visualizations?|formula|equation|"
    r"code(?:\s+(?:block|example))?|console(?:\s+(?:log|output))?|"
    r"email(?:\s+preview)?|form|controls?|action|media)\s*:"
)
_MARKDOWN_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_ORDINARY_STRUCTURED_INSTRUCTION = re.compile(
    r"(?i)^\s*(?:please\s+)?"
    r"(?:create|show|add|include|provide|display|use|visualize|render|draft)"
    r"\b.*\b(?:charts?|graphs?|plots?|visualizations?|formulas?|"
    r"equations?|code(?:\s+(?:examples?|blocks?))?|"
    r"console(?:\s+(?:outputs?|logs?))?|email(?:\s+previews?)?|"
    r"input\s+forms?|forms?|controls?)\b"
)
_ROLE_SECTION = re.compile(
    r"(?i)^\s*(?:#{1,6}\s*)?"
    r"(?:charts?|graphs?|plots?|visualizations?|formulas?|equations?|"
    r"code(?:\s+(?:examples?|blocks?))?|"
    r"console(?:\s+(?:outputs?|logs?))?|email(?:\s+previews?)?|"
    r"input\s+forms?|forms?|controls?)\s*:?\s*$"
)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")


def canonical_representation_policy(value: Any) -> str:
    normalized = str(value or "table_only").strip().casefold()
    if normalized == "either_table_or_chart":
        return "structured_equivalent"
    if normalized not in {
        "table_only",
        "chart_only",
        "structured_equivalent",
        "both_required",
    }:
        return "table_only"
    return normalized


@dataclass(frozen=True)
class AtomicApplicabilityPlan:
    table_fidelity: bool
    action_fidelity: bool
    media_fidelity: bool
    chart_roles: bool
    generic_content: bool
    text_chunking: bool
    unsupported_external_additions: bool
    representation_policies: Mapping[str, str]
    evidence_ownership: Mapping[str, str]
    exact_values: bool = False
    headings: bool = False
    semantic_roles: bool = False
    generic_source_units: tuple[str, ...] = ()
    source_action_taxonomy: Mapping[str, str] = None  # type: ignore[assignment]
    source_media_taxonomy: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_action_taxonomy",
            dict(self.source_action_taxonomy or {}),
        )
        object.__setattr__(
            self,
            "source_media_taxonomy",
            dict(self.source_media_taxonomy or {}),
        )

    def atomic_mapping(self) -> dict[str, bool]:
        return {
            "fidelity.content_unit_fidelity": self.generic_content,
            "fidelity.content_order_preservation": self.generic_content,
            "fidelity.visible_content_multiset_fbeta": self.generic_content,
            "fidelity.output_block_precision": self.generic_content,
            "fidelity.markdown_table_fidelity": self.table_fidelity,
            "fidelity.exact_numbers_dates_units_fbeta": self.exact_values,
            "fidelity.heading_fidelity_and_order": self.headings,
            "fidelity.action_and_source_link_fidelity": self.action_fidelity,
            "fidelity.media_fidelity": self.media_fidelity,
            "fidelity.unsupported_external_addition_precision": (
                self.unsupported_external_additions
            ),
            "hierarchy.text_chunking": self.text_chunking,
            "semantic_mapping.semantic_role_instance_fidelity": (
                self.semantic_roles
            ),
            "semantic_mapping.role_count_coverage": self.semantic_roles,
        }


def classify_action(
    action_type: Any,
    target: Any,
) -> str:
    name = str(action_type or "").strip().casefold().replace("_", "")
    normalized_target = str(target or "").strip().casefold()
    if name == "openurl":
        return (
            "external_semantic"
            if normalized_target.startswith(("http://", "https://", "{{u"))
            else "navigation_internal"
        )
    if name in {"setstate", "pushstate", "removestate"}:
        return "local_interaction"
    if name == "validateform":
        return "form_mechanic"
    if name in {
        "selecttab",
        "openmodal",
        "closemodal",
        "navigate",
        "scrollto",
    }:
        return "navigation_internal"
    return "unknown"


def classify_media(
    kind: Any,
    alt: Any,
    *,
    decorative: Any = False,
    renderer_asset: bool = False,
) -> str:
    canonical_kind = str(kind or "").strip().casefold()
    description = str(alt or "").strip()
    if renderer_asset:
        return "renderer_asset"
    if canonical_kind == "icon" and (
        bool(decorative) or not description
    ):
        return "decorative_icon"
    if canonical_kind in {"image", "icon", "video", "audioplayer"}:
        return "source_semantic_media"
    return "unknown_media"


def generic_source_units(
    source_text: str,
    units: Sequence[str],
) -> tuple[str, ...]:
    """Return source prose that is not owned by a dedicated semantic role."""

    table_lines: set[str] = set()
    role_lines: set[str] = set()
    lines = str(source_text or "").splitlines()
    for index, line in enumerate(lines):
        if _MARKDOWN_TABLE_SEPARATOR.match(line):
            if index:
                table_lines.add(lines[index - 1].strip())
            table_lines.add(line.strip())
            cursor = index + 1
            while cursor < len(lines) and "|" in lines[cursor]:
                table_lines.add(lines[cursor].strip())
                cursor += 1
        if _ROLE_SECTION.match(line):
            role_lines.add(line.strip())
            role_lines.add(
                re.sub(r"^\s*#{1,6}\s*", "", line).strip().rstrip(":")
            )
            cursor = index + 1
            while cursor < len(lines):
                match = _BULLET.match(lines[cursor])
                if not match:
                    if not lines[cursor].strip():
                        cursor += 1
                        continue
                    break
                role_lines.add(lines[cursor].strip())
                role_lines.add(match.group(1).strip())
                cursor += 1
    result: list[str] = []
    for raw in units:
        value = str(raw or "").strip()
        if not value or value in table_lines or value in role_lines:
            continue
        if _STRUCTURED_DIRECTIVE.match(value):
            continue
        if _ORDINARY_STRUCTURED_INSTRUCTION.match(value):
            continue
        if value.startswith("```") or value == "---":
            continue
        result.append(value)
    return tuple(result)


def build_atomic_applicability_plan(
    source: _core.SourceContract,
    expected_contract: Mapping[str, Any],
    *,
    unsupported_external_additions_policy: str = "penalize",
) -> AtomicApplicabilityPlan:
    """Build the plan solely from source-side inputs and contract policy."""

    policies = {
        str(table.id or f"table_{index + 1}"): canonical_representation_policy(
            table.representation_policy
        )
        for index, table in enumerate(source.tables)
        if table.required and table.minimum_count > 0
    }
    required_tables = [
        table
        for table in source.tables
        if table.required and table.minimum_count > 0
    ]

    action_taxonomy = {
        str(item.id or f"action_{index + 1}"): classify_action(
            item.action_type, item.url
        )
        for index, item in enumerate(source.explicit_actions)
        if item.required and item.minimum_count > 0
    }
    applicable_actions = [
        item
        for index, item in enumerate(source.explicit_actions)
        if item.required
        and item.minimum_count > 0
        and (
            action_taxonomy.get(
                str(item.id or f"action_{index + 1}")
            )
            == "external_semantic"
            or item.action_type.casefold() != "openurl"
        )
    ]

    raw_media = expected_contract.get("media")
    raw_media_rows = (
        list(raw_media)
        if isinstance(raw_media, Sequence)
        and not isinstance(raw_media, (str, bytes, bytearray))
        else []
    )
    media_taxonomy: dict[str, str] = {}
    applicable_media: list[_core.MediaRef] = []
    for index, item in enumerate(source.media):
        raw = (
            raw_media_rows[index]
            if index < len(raw_media_rows)
            and isinstance(raw_media_rows[index], Mapping)
            else {}
        )
        category = classify_media(
            item.kind,
            item.alt,
            decorative=raw.get("decorative", False),
            renderer_asset=bool(raw.get("renderer_asset", False)),
        )
        key = str(item.id or f"media_{index + 1}")
        media_taxonomy[key] = category
        if (
            item.required
            and item.minimum_count > 0
            and category == "source_semantic_media"
        ):
            applicable_media.append(item)

    role_requirements = source.role_requirements
    chart_count = max(0, int(source.expected_role_counts.get("chart", 0)))
    chart_roles = chart_count > 0 or any(
        bool(item.get("required", True))
        and int(item.get("minimum_count", 1) or 0) > 0
        for item in role_requirements.get("chart", ())
    )
    semantic_role_names = {"chart", "formula", "code", "console", "email", "form"}
    semantic_roles = any(
        role in semantic_role_names and max(0, int(count)) > 0
        for role, count in source.expected_role_counts.items()
    ) or any(
        role in semantic_role_names
        and any(
            bool(item.get("required", True))
            and int(item.get("minimum_count", 1) or 0) > 0
            for item in values
        )
        for role, values in role_requirements.items()
    )
    generic_units = generic_source_units(
        source.raw_text, source.content_units
    )
    generic_content = bool(generic_units)
    generic_exact_values = any(
        _core._EXACT_VALUE_RE.search(value)  # type: ignore[attr-defined]
        or _core._DATE_RE.search(value)  # type: ignore[attr-defined]
        for value in generic_units
    )
    return AtomicApplicabilityPlan(
        table_fidelity=bool(required_tables),
        action_fidelity=bool(applicable_actions),
        media_fidelity=bool(applicable_media),
        chart_roles=chart_roles,
        generic_content=generic_content,
        text_chunking=generic_content,
        unsupported_external_additions=(
            unsupported_external_additions_policy == "penalize"
        ),
        representation_policies=policies,
        evidence_ownership={
            "generic_visible_content": "generic_content_metrics",
            "source_table": "table_fidelity",
            "source_chart": "semantic_role_and_structured_fidelity",
            "source_action": "action_fidelity",
            "source_media": "media_fidelity",
            "semantic_role": "semantic_role_instance_fidelity",
            "local_ui_mechanic": "excluded_from_source_fidelity",
        },
        generic_source_units=generic_units,
        source_action_taxonomy=action_taxonomy,
        source_media_taxonomy=media_taxonomy,
        exact_values=generic_exact_values,
        headings=any(
            not _ROLE_SECTION.match(heading)
            for heading in source.headings
        ),
        semantic_roles=semantic_roles,
    )


def filter_source_actions(
    source: _core.SourceContract,
    plan: AtomicApplicabilityPlan,
) -> list[_core.ActionRef]:
    result: list[_core.ActionRef] = []
    for index, item in enumerate(source.explicit_actions):
        key = str(item.id or f"action_{index + 1}")
        category = plan.source_action_taxonomy.get(key)
        if category == "external_semantic" or (
            item.required
            and item.minimum_count > 0
            and item.action_type.casefold() != "openurl"
        ):
            result.append(item)
    return result


def filter_source_media(
    source: _core.SourceContract,
    plan: AtomicApplicabilityPlan,
) -> list[_core.MediaRef]:
    return [
        item
        for index, item in enumerate(source.media)
        if plan.source_media_taxonomy.get(
            str(item.id or f"media_{index + 1}")
        )
        == "source_semantic_media"
    ]


__all__ = [
    "APPLICABILITY_POLICY_VERSION",
    "AtomicApplicabilityPlan",
    "build_atomic_applicability_plan",
    "canonical_representation_policy",
    "classify_action",
    "classify_media",
    "filter_source_actions",
    "filter_source_media",
    "generic_source_units",
]
