"""Deterministic reward for training text-to-GenUI FlatSpec models with GRPO.

Scope
-----
The source response text is treated as the reference content.  This module does
*not* judge whether that source text is factually correct, well written, safe,
or useful.  It judges the generated JSON as a renderer-facing representation:

* can the native renderer consume it;
* is the declared-root graph sound;
* does it preserve the source's visible information and exact values;
* are tables, actions, charts, code, formulas, email drafts and media mapped to
  suitable components;
* is the hierarchy usable and the IR reasonably economical; and
* are applicable accessibility contracts present.

The implementation deliberately gives no positive reward for raw component
count or type count.  All features are bounded in [0, 1], all non-applicable
features are excluded, and only declared-root-reachable elements are scored.

The module has no third-party runtime dependency.  It can be imported directly
by Hugging Face TRL's GRPOTrainer.  Project-specific render smoke tests can be
injected through ``render_check``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import exp, log
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit, urlunsplit

REWARD_VERSION = "4.0.0"

# Keep this aligned with the JSON schema and Android renderer.  Chart is
# intentionally included; the reviewed flat_spec_contract.py omitted it even
# though the schema, prompt and native renderer support it.
ALLOWED_TYPES = frozenset(
    {
        "Stack",
        "Column",
        "Row",
        "List",
        "Card",
        "Text",
        "Formula",
        "CodeBlock",
        "ConsoleLog",
        "EmailPreview",
        "Table",
        "Chart",
        "Image",
        "Icon",
        "Video",
        "AudioPlayer",
        "Divider",
        "Button",
        "Tabs",
        "Modal",
        "TextField",
        "CheckBox",
        "ChoicePicker",
        "Slider",
        "DateTimeInput",
    }
)
ALLOWED_ACTIONS = frozenset(
    {"openUrl", "setState", "pushState", "removeState", "validateForm"}
)
CONTAINER_TYPES = frozenset({"Stack", "Column", "Row", "List", "Card", "Tabs", "Modal"})
LAYOUT_TYPES = frozenset({"Stack", "Column", "Row", "List", "Card"})
VISIBLE_TEXT_TYPES = frozenset(
    {
        "Text",
        "Formula",
        "CodeBlock",
        "ConsoleLog",
        "EmailPreview",
        "Table",
        "Chart",
        "Image",
        "Video",
        "AudioPlayer",
        "Button",
        "TextField",
        "CheckBox",
        "ChoicePicker",
        "Slider",
        "DateTimeInput",
    }
)
URL_FIELD_NAMES = frozenset(
    {
        "url",
        "href",
        "link",
        "actionurl",
        "bookingurl",
        "sourceurl",
        "website",
        "videoUrl".lower(),
        "audioUrl".lower(),
    }
)
MEDIA_TYPES = frozenset({"Image", "Icon", "Video", "AudioPlayer"})

# Dimension budgets.  No atomic input has an effective global weight above
# roughly 10%; fidelity dominates only as a *dimension* composed of multiple
# independent checks.
DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "integrity": 0.25,
    "fidelity": 0.40,
    "semantic_mapping": 0.15,
    "hierarchy": 0.10,
    "economy": 0.07,
    "accessibility": 0.03,
}

DEFAULT_ATOMIC_WEIGHTS: dict[str, dict[str, float]] = {
    "integrity": {
        "schema_contract": 0.25,
        "declared_root_reachability": 0.25,
        "reference_integrity": 0.20,
        "type_semantic_contract": 0.25,
        "native_render_smoke": 0.05,
    },
    "fidelity": {
        "visible_content_multiset_fbeta": 0.25,
        "exact_numbers_dates_units_fbeta": 0.18,
        "markdown_table_fidelity": 0.24,
        "heading_fidelity_and_order": 0.12,
        "action_and_source_link_fidelity": 0.15,
        "media_fidelity": 0.06,
    },
    "semantic_mapping": {
        "required_component_roles": 0.65,
        "component_appropriateness": 0.35,
    },
    "hierarchy": {
        "root_layout": 0.20,
        "root_distance_depth": 0.25,
        "container_fanout": 0.20,
        "heading_grouping": 0.20,
        "text_chunking": 0.15,
    },
    "economy": {
        "semantic_non_duplication": 0.40,
        "renderer_aware_wrapper_economy": 0.30,
        "json_to_source_size": 0.30,
    },
    "accessibility": {
        "applicable_accessibility_contracts": 1.00,
    },
}

DEFAULT_CAPS: dict[str, float] = {
    "missing_root": 0.10,
    "schema_severe": 0.25,
    "schema_moderate": 0.45,
    "reference_or_cycle": 0.25,
    "reachability_below_90": 0.40,
    "partial_reachability": 0.70,
    "type_contract_severe": 0.45,
    "render_failure": 0.30,
    "missing_table": 0.65,
    "partial_table": 0.90,
    "missing_chart": 0.78,
    "chart_below_two_thirds": 0.88,
    "partial_chart": 0.94,
    "missing_special_role": 0.78,
    "partial_special_role": 0.92,
    "missing_action": 0.75,
    "action_below_half": 0.88,
    "partial_action": 0.96,
}

# Conservative tokenization for cross-format matching.  Unicode word tokens,
# currency signs, percent and common numeric punctuation are retained.
_TOKEN_RE = re.compile(r"[\w]+(?:[./:+\-][\w]+)*|[$€£¥₹]\s*\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?%?", re.UNICODE)
_URL_RE = re.compile(r"https://[^\s)>\]}]+|\{\{u\d+\}\}", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https://[^)\s]+|\{\{u\d+\}\})\)", re.IGNORECASE)
_ACTION_RE = re.compile(
    r"(?im)^\s*Action:\s*\[\s*Button\s*:\s*([^\]]+?)\s*\]\s*(https://\S+|\{\{u\d+\}\})\s*$"
)
_MEDIA_RE = re.compile(
    r"(?im)^\s*Media:\s*(Image|Icon|Video|Audio(?:Player)?)\s*=\s*(\S+?)(?:\s+Alt\s*=\s*(.+))?\s*$"
)
_HEADING_RE = re.compile(r"(?m)^\s*(#{1,6})\s+(.+?)\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_EXACT_VALUE_RE = re.compile(
    r"(?<!\w)(?:[$€£¥₹]\s*)?[-+]?\d[\d,]*(?:\.\d+)?(?:\s?(?:%|°[CF]?|km|miles?|mi|kg|g|lb|lbs|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|GB|MB|TB|GHz|MHz|kWh|W|V|A|CAD|USD|EUR|INR))?(?!\w)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?<!\w)(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)(?!\w)",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```\s*([\w+.-]*)\s*\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")


@dataclass(frozen=True)
class Atomic:
    value: float | None
    weight: float
    evidence: str = ""

    @property
    def applicable(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class SourceTable:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    id: str = ""
    required: bool = True
    minimum_count: int = 1
    order_sensitive: bool = False
    row_key: tuple[str, ...] = ()
    representation_policy: str = "table_only"

    @property
    def cells(self) -> tuple[str, ...]:
        return self.headers + tuple(cell for row in self.rows for cell in row)


@dataclass(frozen=True)
class ActionRef:
    label: str
    url: str
    action_type: str = "openUrl"
    id: str = ""
    required: bool = True
    minimum_count: int = 1
    expected_component_id: str = ""


@dataclass(frozen=True)
class MediaRef:
    kind: str
    url: str
    alt: str = ""
    id: str = ""
    required: bool = True
    minimum_count: int = 1
    media_policy: str = "exact"
    expected_component_id: str = ""
    component_id: str = ""


@dataclass
class SourceContract:
    raw_text: str
    intent: str | None
    headings: list[str]
    tables: list[SourceTable]
    explicit_actions: list[ActionRef]
    actionable_urls: set[str]
    media: list[MediaRef]
    code_blocks: list[tuple[str, str]]
    inline_code: list[str]
    exact_values: Counter[str]
    content_tokens: Counter[str]
    required_roles: dict[str, bool]
    expected_role_counts: dict[str, int]
    asset_aliases: dict[str, set[str]] = field(default_factory=dict)
    content_units: list[str] = field(default_factory=list)
    role_requirements: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass
class GraphAudit:
    root_exists: bool
    reachable_ids: set[str]
    unreachable_ids: set[str]
    missing_references: set[str]
    cycle_edges: set[tuple[str, str]]
    multi_parent_ids: set[str]
    depths: dict[str, int]

    @property
    def reachable_fraction(self) -> float:
        total = len(self.reachable_ids) + len(self.unreachable_ids)
        return len(self.reachable_ids) / total if total else 0.0

    @property
    def max_depth(self) -> int:
        return max(self.depths.values(), default=0)


@dataclass
class OutputTable:
    element_id: str
    element_type: str
    headers: list[str]
    keys: list[str]
    rows: list[dict[str, Any]]


@dataclass
class OutputAction:
    element_id: str
    label: str
    url: str
    source: str
    action_type: str = "openUrl"


@dataclass
class OutputEvidence:
    elements: Mapping[str, Mapping[str, Any]]
    reachable_ids: set[str]
    visible_blocks: list[str]
    headings: list[str]
    heading_variants: list[str]
    tables: list[OutputTable]
    actions: list[OutputAction]
    media: list[MediaRef]
    type_counts: Counter[str]
    content_tokens: Counter[str]
    exact_values: Counter[str]
    valid_type_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    quality_0_100: float
    quality_0_1: float
    cap_0_1: float
    parse_stage: str
    dimensions: dict[str, float | None]
    atomics: dict[str, dict[str, float | None]]
    evidence: dict[str, Any]
    metric_version: str = REWARD_VERSION
    active_caps: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metric_name: str = "GenUI Representation Quality"
    metric_fingerprint: str = ""
    base_quality_before_caps: float = 0.0
    effective_atomic_weights: dict[str, float] = field(default_factory=dict)
    effective_dimension_weights: dict[str, float] = field(default_factory=dict)
    applicable_atomic_count: int = 0
    anti_domination_feasible: bool = True
    normalization: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, str | None] = field(default_factory=dict)
    binding_caps: list[dict[str, Any]] = field(default_factory=list)
    cap_margin: float = 0.0
    matching_certification: dict[str, Any] = field(default_factory=dict)
    dynamic_semantics: dict[str, Any] = field(default_factory=dict)

    @property
    def engineering_score(self) -> float:
        return self.quality_0_100

    @property
    def quality_utility(self) -> float:
        return self.quality_0_1

    @property
    def grpo_reward(self) -> float:
        return self.reward


@dataclass
class RewardConfig:
    dimension_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DIMENSION_WEIGHTS)
    )
    atomic_weights: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_ATOMIC_WEIGHTS.items()}
    )
    caps: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CAPS))
    arithmetic_share: float = 0.75
    geometric_floor: float = 0.03
    content_beta: float = 2.0
    value_beta: float = 2.0
    table_beta: float = 2.0
    good_json_to_source_ratio: float = 3.0
    bad_json_to_source_ratio: float = 7.0
    max_atomic_global_weight: float = 0.10
    anti_domination_infeasible_policy: str = "fail_closed"
    max_repeat_items: int = 32
    max_expanded_evidence_nodes: int = 512
    max_expression_depth: int = 12
    max_string_expansion_length: int = 4096
    max_assignment_size: int = 64
    max_matching_edges: int = 65536
    large_matching_top_k: int = 16
    max_matching_hungarian_work: int = 50_000_000
    max_matching_sparse_relaxations: int = 50_000_000
    role_match_threshold: float = 0.70
    matching_incomplete_policy: str = "fail_closed"
    dynamic_unknown_policy: str = "cap"
    render_check: Callable[[Mapping[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        self.dimension_weights = self._normalize_group(
            self.dimension_weights, "dimension_weights"
        )
        normalized_atomics: dict[str, dict[str, float]] = {}
        dimensions = set(self.dimension_weights) | set(self.atomic_weights)
        for dimension in sorted(dimensions):
            defaults = DEFAULT_ATOMIC_WEIGHTS.get(dimension, {})
            raw = self.atomic_weights.get(dimension, defaults)
            if not raw:
                continue
            normalized_atomics[dimension] = self._normalize_group(
                raw, f"atomic_weights.{dimension}"
            )
        self.atomic_weights = normalized_atomics

        caps = {**DEFAULT_CAPS, **{str(k): float(v) for k, v in self.caps.items()}}
        for name, value in caps.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"caps.{name} must be in [0, 1]")
        self.caps = caps
        if not 0.0 <= self.arithmetic_share <= 1.0:
            raise ValueError("arithmetic_share must be in [0, 1]")
        if not 0.0 < self.geometric_floor <= 1.0:
            raise ValueError("geometric_floor must be in (0, 1]")
        if min(self.content_beta, self.value_beta, self.table_beta) <= 0.0:
            raise ValueError("F-beta parameters must be positive")
        if self.good_json_to_source_ratio < 0.0:
            raise ValueError("good_json_to_source_ratio must be non-negative")
        if self.bad_json_to_source_ratio <= self.good_json_to_source_ratio:
            raise ValueError("bad_json_to_source_ratio must exceed good ratio")
        if not 0.0 < self.max_atomic_global_weight <= 1.0:
            raise ValueError("max_atomic_global_weight must be in (0, 1]")
        if self.anti_domination_infeasible_policy not in {"fail_closed", "error"}:
            raise ValueError(
                "anti_domination_infeasible_policy must be fail_closed or error"
            )
        for name in (
            "max_repeat_items",
            "max_expanded_evidence_nodes",
            "max_expression_depth",
            "max_string_expansion_length",
            "max_assignment_size",
            "max_matching_edges",
            "large_matching_top_k",
            "max_matching_hungarian_work",
            "max_matching_sparse_relaxations",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.role_match_threshold <= 1.0:
            raise ValueError("role_match_threshold must be in [0, 1]")
        if self.matching_incomplete_policy not in {"fail_closed", "not_applicable"}:
            raise ValueError(
                "matching_incomplete_policy must be fail_closed or not_applicable"
            )
        if self.dynamic_unknown_policy not in {"cap", "not_applicable", "error"}:
            raise ValueError(
                "dynamic_unknown_policy must be cap, not_applicable, or error"
            )

    @staticmethod
    def _normalize_group(values: Mapping[str, float], name: str) -> dict[str, float]:
        converted = {str(k): float(v) for k, v in values.items()}
        if not converted or any(v < 0.0 for v in converted.values()):
            raise ValueError(f"{name} must contain non-negative weights")
        total = sum(converted.values())
        if total <= 0.0:
            raise ValueError(f"{name} must have positive total weight")
        return {k: v / total for k, v in converted.items()}

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        render_check: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> "RewardConfig":
        dimensions = dict(DEFAULT_DIMENSION_WEIGHTS)
        dimensions.update(mapping.get("dimension_weights") or {})
        metric = mapping.get("metric") if isinstance(mapping.get("metric"), Mapping) else {}
        is_v5 = str(metric.get("version") or "").startswith("5.")
        atomics = {} if is_v5 else {k: dict(v) for k, v in DEFAULT_ATOMIC_WEIGHTS.items()}
        for dimension, values in (mapping.get("atomic_weights") or {}).items():
            if isinstance(values, Mapping):
                if is_v5:
                    atomics[str(dimension)] = dict(values)
                else:
                    atomics.setdefault(str(dimension), {}).update(values)
        caps = dict(DEFAULT_CAPS)
        caps.update(mapping.get("caps") or {})
        aggregation = mapping.get("aggregation") or {}
        fidelity = mapping.get("fidelity") or {}
        economy = mapping.get("economy") or {}
        guardrails = mapping.get("guardrails") or {}
        evidence_limits = mapping.get("evidence_limits") or {}
        return cls(
            dimension_weights=dimensions,
            atomic_weights=atomics,
            caps=caps,
            arithmetic_share=float(aggregation.get("arithmetic_share", 0.75)),
            geometric_floor=float(aggregation.get("geometric_floor", 0.03)),
            content_beta=float(fidelity.get("content_beta", 2.0)),
            value_beta=float(fidelity.get("value_beta", 2.0)),
            table_beta=float(fidelity.get("table_beta", 2.0)),
            good_json_to_source_ratio=float(
                economy.get("good_json_to_source_ratio", 3.0)
            ),
            bad_json_to_source_ratio=float(
                economy.get("bad_json_to_source_ratio", 7.0)
            ),
            max_atomic_global_weight=float(
                guardrails.get("max_atomic_global_weight", 0.10)
            ),
            anti_domination_infeasible_policy=str(
                guardrails.get("anti_domination_infeasible_policy", "fail_closed")
            ),
            max_repeat_items=int(evidence_limits.get("max_repeat_items", 32)),
            max_expanded_evidence_nodes=int(
                evidence_limits.get("max_expanded_evidence_nodes", 512)
            ),
            max_expression_depth=int(
                evidence_limits.get("max_expression_depth", 12)
            ),
            max_string_expansion_length=int(
                evidence_limits.get("max_string_expansion_length", 4096)
            ),
            max_assignment_size=int(
                evidence_limits.get("max_assignment_size", 64)
            ),
            max_matching_edges=int(
                evidence_limits.get("max_matching_edges", 65536)
            ),
            large_matching_top_k=int(
                evidence_limits.get("large_matching_top_k", 16)
            ),
            max_matching_hungarian_work=int(
                evidence_limits.get(
                    "max_matching_hungarian_work", 50_000_000
                )
            ),
            max_matching_sparse_relaxations=int(
                evidence_limits.get(
                    "max_matching_sparse_relaxations", 50_000_000
                )
            ),
            role_match_threshold=float(
                fidelity.get("role_match_threshold", 0.70)
            ),
            matching_incomplete_policy=str(
                guardrails.get("matching_incomplete_policy", "fail_closed")
            ),
            dynamic_unknown_policy=str(
                guardrails.get("dynamic_unknown_policy", "cap")
            ),
            render_check=render_check,
        )


def load_reward_config(path: str | os.PathLike[str]) -> RewardConfig:
    """Load a versioned YAML or JSON reward configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(str(config_path))
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.casefold() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML reward configuration"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise ValueError("reward configuration root must be a mapping")
    return RewardConfig.from_mapping(payload)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    p, r = clamp01(precision), clamp01(recall)
    if p == 0.0 and r == 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * p * r / (b2 * p + r)


def weighted_applicable_mean(atomics: Mapping[str, Atomic]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for atomic in atomics.values():
        if atomic.value is None or atomic.weight <= 0:
            continue
        numerator += atomic.weight * clamp01(atomic.value)
        denominator += atomic.weight
    return numerator / denominator if denominator else None


def plateau_utility(value: float, good_low: float, good_high: float, bad_low: float, bad_high: float) -> float:
    """One inside [good_low, good_high], linearly declining to zero at bad bounds."""
    if value < good_low:
        return clamp01((value - bad_low) / (good_low - bad_low)) if good_low > bad_low else 0.0
    if value > good_high:
        return clamp01((bad_high - value) / (bad_high - good_high)) if bad_high > good_high else 0.0
    return 1.0


def low_is_good(value: float, good_at_or_below: float, bad_at_or_above: float) -> float:
    if value <= good_at_or_below:
        return 1.0
    if value >= bad_at_or_above:
        return 0.0
    return 1.0 - (value - good_at_or_below) / (bad_at_or_above - good_at_or_below)


def normalize_markdown(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"<([^>]+)>", r"\1", s)
    s = re.sub(r"[*_~]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_match_text(text: Any) -> str:
    s = normalize_markdown(text).casefold()
    s = re.sub(r"[^\w%€$£¥₹./:+\-]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(text: Any) -> list[str]:
    return [normalize_match_text(m.group(0)) for m in _TOKEN_RE.finditer(str(text)) if normalize_match_text(m.group(0))]


def counter_pr(counter: Counter[str], reference: Counter[str]) -> tuple[float, float]:
    overlap = sum((counter & reference).values())
    precision = safe_div(overlap, sum(counter.values()), 1.0 if not counter else 0.0)
    recall = safe_div(overlap, sum(reference.values()), 1.0 if not reference else 0.0)
    return precision, recall


def normalize_url(url: Any) -> str:
    s = str(url or "").strip().rstrip(".,;:!?")
    if re.fullmatch(r"\{\{u\d+\}\}", s, re.IGNORECASE):
        return s.casefold()
    if not s.lower().startswith("https://"):
        return s.replace("\\", "/")
    try:
        parts = urlsplit(s)
        path = re.sub(r"/{2,}", "/", parts.path or "/")
        if path != "/":
            path = path.rstrip("/")
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))
    except ValueError:
        return s


def text_similarity(a: str, b: str) -> float:
    na, nb = normalize_match_text(a), normalize_match_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    token_f1 = f_beta(safe_div(len(ta & tb), len(tb)), safe_div(len(ta & tb), len(ta)))
    seq = SequenceMatcher(None, na, nb).ratio()
    return 0.55 * token_f1 + 0.45 * seq


def _strip_json_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def completion_to_text(completion: Any) -> str:
    """Accept plain TRL completions or conversational completion objects."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        # Conversational messages use content/text. A direct FlatSpec mapping
        # should remain valid JSON rather than Python's single-quoted repr.
        if "content" in completion or "text" in completion:
            return str(completion.get("content", completion.get("text", "")))
        return json.dumps(completion, ensure_ascii=False, separators=(",", ":"))
    if isinstance(completion, Sequence) and not isinstance(completion, (bytes, bytearray, str)):
        # Conversational outputs are often a one-message list.
        pieces: list[str] = []
        for item in completion:
            if isinstance(item, Mapping):
                pieces.append(str(item.get("content", item.get("text", ""))))
            else:
                pieces.append(str(item))
        return "\n".join(p for p in pieces if p)
    return str(completion)


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse one JSON object while tolerating fences or a short accidental prefix."""
    s = _strip_json_fences(text)
    if not s:
        return None, "empty"
    try:
        value = json.loads(s)
        return (value, "json") if isinstance(value, dict) else (None, "json_non_object")
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", s):
        try:
            value, end = decoder.raw_decode(s[match.start() :])
        except json.JSONDecodeError:
            continue
        trailing = s[match.start() + end :].strip()
        if isinstance(value, dict) and (not trailing or trailing == "```"):
            return value, "json_with_prefix"
    return None, "json_parse_error"


def _split_markdown_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    # This intentionally does not implement escaped pipes inside code; such
    # cases should be covered by explicit contract annotations in production.
    return [normalize_markdown(cell.strip()) for cell in s.split("|")]


def parse_markdown_tables(text: str) -> list[SourceTable]:
    lines = text.splitlines()
    tables: list[SourceTable] = []
    i = 0
    while i + 1 < len(lines):
        if "|" not in lines[i] or not _TABLE_SEPARATOR_RE.match(lines[i + 1]):
            i += 1
            continue
        headers = _split_markdown_row(lines[i])
        rows: list[tuple[str, ...]] = []
        i += 2
        while i < len(lines) and "|" in lines[i] and lines[i].strip():
            cells = _split_markdown_row(lines[i])
            # Normalize ragged rows rather than silently dropping cells.
            if len(cells) < len(headers):
                cells += [""] * (len(headers) - len(cells))
            rows.append(tuple(cells[: len(headers)]))
            i += 1
        tables.append(SourceTable(tuple(headers), tuple(rows)))
    return tables


def _asset_alias_map(assets: Any) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes, bytearray)):
        return {}
    for item in assets:
        if not isinstance(item, Mapping):
            continue
        original = normalize_url(item.get("url", ""))
        raw_path = str(item.get("path", "")).replace("\\", "/").lstrip("./")
        if not original or not raw_path:
            continue
        values = {
            original,
            normalize_url(raw_path),
            normalize_url("../" + raw_path),
            normalize_url("./" + raw_path),
        }
        for value in values:
            aliases[original].update(values)
            aliases[value].update(values)
    return dict(aliases)


def _clean_source_for_content(text: str) -> str:
    # Keep action labels but remove raw destinations; remove media metadata while
    # retaining image alt text because it is meaningful content.
    s = _ACTION_RE.sub(lambda m: m.group(1), text)

    def media_repl(match: re.Match[str]) -> str:
        return match.group(3) or ""

    s = _MEDIA_RE.sub(media_repl, s)
    s = _MARKDOWN_LINK_RE.sub(lambda m: m.group(1), s)
    s = _URL_RE.sub(" ", s)
    s = re.sub(r"```[\w+.-]*", " ", s)
    s = s.replace("```", " ")
    s = re.sub(r"(?m)^\s*\|?\s*:?-{3,}.*$", " ", s)
    s = re.sub(r"[#|*_~]", " ", s)
    return s


def parse_source_contract(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
) -> SourceContract:
    headings = [normalize_markdown(m.group(2)) for m in _HEADING_RE.finditer(response_text)]
    tables = parse_markdown_tables(response_text)
    explicit_actions = [
        ActionRef(normalize_markdown(m.group(1)), normalize_url(m.group(2)), "openUrl")
        for m in _ACTION_RE.finditer(response_text)
    ]
    media = [
        MediaRef(
            kind=("AudioPlayer" if m.group(1).lower().startswith("audio") else m.group(1).title()),
            url=normalize_url(m.group(2)),
            alt=normalize_markdown(m.group(3) or ""),
        )
        for m in _MEDIA_RE.finditer(response_text)
    ]
    media_urls = {m.url for m in media}
    all_urls = {normalize_url(m.group(0)) for m in _URL_RE.finditer(response_text)}
    actionable_urls = {u for u in all_urls if u not in media_urls}

    code_blocks = [(lang.casefold(), code.strip()) for lang, code in _CODE_FENCE_RE.findall(response_text)]
    inline_code = [normalize_markdown(v) for v in _INLINE_CODE_RE.findall(response_text)]
    exact_values = Counter(normalize_match_text(v) for v in _EXACT_VALUE_RE.findall(response_text))
    exact_values.update(normalize_match_text(v) for v in _DATE_RE.findall(response_text))
    exact_values.pop("", None)

    cleaned = _clean_source_for_content(response_text)
    content_tokens = Counter(tokenize(cleaned))
    content_units = [line.strip() for line in cleaned.splitlines() if line.strip()]

    intent_norm = (intent or "").casefold().strip()
    chart_titles = len(re.findall(r"(?im)^\s*chart\s+title\s*:", response_text))
    numbered_charts = len(re.findall(r"(?im)^\s*[-*]?\s*(?:\*\*)?chart\s+\d+\s*:", response_text))
    chart_required = bool(
        chart_titles
        or numbered_charts
        or re.search(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:two\s+)?charts?\s+to\s+include\b|\b(?:line|bar|column|scatter|pie|stacked\s+bar)\s+chart\b",
            response_text,
        )
    )
    expected_chart_count = max(chart_titles, numbered_charts, 1 if chart_required else 0)
    email_required = bool(
        re.search(r"(?im)^\s*(?:\*\*)?Subject(?:\*\*)?\s*:", response_text)
        and re.search(r"(?im)^\s*(?:Dear|Hi|Hello)\b", response_text)
    )
    formula_required = bool(
        intent_norm == "calculation"
        or re.search(r"(?im)^\s*(?:formula|equation)\s*:", response_text)
    )
    command_like_inline = [
        c
        for c in inline_code
        if re.search(
            r"(?:^|\s)(?:ipconfig|ping|netsh|python|pip|npm|git|curl|wget|adb|gradle|mvn|sudo|docker|kubectl|SELECT|INSERT|UPDATE|DELETE)\b|[/\\]",
            c,
            re.IGNORECASE,
        )
    ]
    code_required = bool(code_blocks)
    console_required = len(command_like_inline) >= 2
    image_required = any(m.kind == "Image" for m in media)
    video_required = any(m.kind == "Video" for m in media)
    audio_required = any(m.kind == "AudioPlayer" for m in media)

    return SourceContract(
        raw_text=response_text,
        intent=intent,
        headings=headings,
        tables=tables,
        explicit_actions=explicit_actions,
        actionable_urls=actionable_urls,
        media=media,
        code_blocks=code_blocks,
        inline_code=inline_code,
        exact_values=exact_values,
        content_tokens=content_tokens,
        required_roles={
            "table": bool(tables),
            "chart": chart_required,
            "email": email_required,
            "formula": formula_required,
            "code": code_required,
            "console": console_required,
            "image": image_required,
            "video": video_required,
            "audio": audio_required,
            "action": bool(actionable_urls or explicit_actions),
        },
        expected_role_counts={
            "table": len(tables),
            "chart": expected_chart_count,
            "email": 1 if email_required else 0,
            "formula": 1 if formula_required else 0,
            "code": max(1, len(code_blocks)) if code_required else 0,
            "console": 1 if console_required else 0,
            "image": 1 if image_required else 0,  # frozen v4 representative-media policy
            "video": 1 if video_required else 0,
            "audio": 1 if audio_required else 0,
            "action": max(1, len(actionable_urls)) if actionable_urls or explicit_actions else 0,
        },
        asset_aliases=_asset_alias_map(assets),
        content_units=content_units,
    )


def source_contract_from_mapping(
    mapping: Mapping[str, Any],
    *,
    fallback: SourceContract,
) -> SourceContract:
    """Overlay an independently persisted expected-UI contract on fallback extraction.

    Presence matters: an explicitly empty list means "not required" and replaces
    a noisy heuristic.  Omitted fields retain the deterministic fallback.
    """
    if not isinstance(mapping, Mapping):
        return fallback

    def text_list(key: str, default: list[str]) -> list[str]:
        if key not in mapping:
            return list(default)
        value = mapping.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return []
        return [normalize_markdown(v) for v in value if normalize_markdown(v)]

    headings = text_list("headings", fallback.headings)

    if "tables" in mapping:
        tables: list[SourceTable] = []
        raw_tables = mapping.get("tables")
        if isinstance(raw_tables, Sequence) and not isinstance(raw_tables, (str, bytes, bytearray)):
            for raw in raw_tables:
                if not isinstance(raw, Mapping):
                    continue
                headers_raw = raw.get("headers") or raw.get("columns") or []
                headers = tuple(
                    normalize_markdown(v)
                    for v in headers_raw
                    if normalize_markdown(v)
                ) if isinstance(headers_raw, Sequence) and not isinstance(headers_raw, (str, bytes, bytearray)) else ()
                rows_raw = raw.get("rows") or []
                rows: list[tuple[str, ...]] = []
                if isinstance(rows_raw, Sequence) and not isinstance(rows_raw, (str, bytes, bytearray)):
                    for row in rows_raw:
                        if isinstance(row, Mapping):
                            keys = list(headers) or [str(k) for k in row]
                            rows.append(tuple(normalize_markdown(row.get(k, "")) for k in keys))
                        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
                            cells = [normalize_markdown(v) for v in row]
                            if headers and len(cells) < len(headers):
                                cells += [""] * (len(headers) - len(cells))
                            rows.append(tuple(cells[: len(headers)] if headers else cells))
                row_key_raw = raw.get("row_key")
                if isinstance(row_key_raw, str):
                    row_key = (normalize_markdown(row_key_raw),)
                elif isinstance(row_key_raw, Sequence) and not isinstance(row_key_raw, (str, bytes, bytearray)):
                    row_key = tuple(normalize_markdown(value) for value in row_key_raw if normalize_markdown(value))
                else:
                    row_key = ()
                tables.append(
                    SourceTable(
                        headers,
                        tuple(rows),
                        id=str(raw.get("id") or ""),
                        required=bool(raw.get("required", True)),
                        minimum_count=max(0, int(raw.get("minimum_count", 1) or 0)),
                        order_sensitive=bool(raw.get("order_sensitive", False)),
                        row_key=row_key,
                        representation_policy=str(
                            raw.get("representation_policy") or "table_only"
                        ),
                    )
                )
    else:
        tables = list(fallback.tables)

    if "actions" in mapping or "source_links" in mapping:
        explicit_actions: list[ActionRef] = []
        actionable_urls: set[str] = set()
        raw_actions = mapping.get("actions") or []
        if isinstance(raw_actions, Sequence) and not isinstance(raw_actions, (str, bytes, bytearray)):
            for raw in raw_actions:
                if not isinstance(raw, Mapping):
                    continue
                target = normalize_url(raw.get("target") or raw.get("url") or "")
                label = normalize_markdown(raw.get("label") or raw.get("title") or "")
                if target:
                    action_type = str(raw.get("action_type") or raw.get("action") or "openUrl")
                    if action_type.casefold().replace("_", "") == "openurl":
                        action_type = "openUrl"
                    explicit_actions.append(
                        ActionRef(
                            label,
                            target,
                            action_type,
                            id=str(raw.get("id") or ""),
                            required=bool(raw.get("required", True)),
                            minimum_count=max(0, int(raw.get("minimum_count", 1) or 0)),
                            expected_component_id=str(
                                raw.get("expected_component_id") or ""
                            ),
                        )
                    )
                    actionable_urls.add(target)
        raw_links = mapping.get("source_links") or []
        if isinstance(raw_links, Sequence) and not isinstance(raw_links, (str, bytes, bytearray)):
            for raw in raw_links:
                if isinstance(raw, Mapping):
                    target = normalize_url(raw.get("target") or raw.get("url") or "")
                    label = normalize_markdown(raw.get("label") or raw.get("title") or "")
                else:
                    target = normalize_url(raw)
                    label = ""
                if target:
                    explicit_actions.append(ActionRef(label, target, "openUrl"))
                    actionable_urls.add(target)
    else:
        explicit_actions = list(fallback.explicit_actions)
        actionable_urls = set(fallback.actionable_urls)

    if "media" in mapping:
        media: list[MediaRef] = []
        raw_media = mapping.get("media")
        if isinstance(raw_media, Sequence) and not isinstance(raw_media, (str, bytes, bytearray)):
            for raw in raw_media:
                if not isinstance(raw, Mapping):
                    continue
                kind = normalize_markdown(raw.get("kind") or raw.get("type") or "")
                url = normalize_url(raw.get("url") or raw.get("target") or "")
                alt = normalize_markdown(raw.get("alt") or raw.get("description") or "")
                if kind or url:
                    media.append(
                        MediaRef(
                            kind=kind or "Image",
                            url=url,
                            alt=alt,
                            id=str(raw.get("id") or ""),
                            required=bool(raw.get("required", True)),
                            minimum_count=max(0, int(raw.get("minimum_count", 1) or 0)),
                            media_policy=str(raw.get("media_policy") or "exact"),
                            expected_component_id=str(
                                raw.get("expected_component_id") or ""
                            ),
                        )
                    )
    else:
        media = list(fallback.media)

    exact_values = fallback.exact_values.copy()
    if "exact_values" in mapping:
        exact_values = Counter(
            normalize_match_text(v)
            for v in (mapping.get("exact_values") or [])
            if normalize_match_text(v)
        )

    content_tokens = fallback.content_tokens.copy()
    content_units = list(fallback.content_units)
    if "content_units" in mapping:
        units = mapping.get("content_units") or []
        content_units = [normalize_markdown(value) for value in units if normalize_markdown(value)]
        content_tokens = Counter(tokenize(" ".join(content_units)))
    elif "reference_content" in mapping:
        reference_content = str(mapping.get("reference_content") or "")
        content_units = [line.strip() for line in reference_content.splitlines() if line.strip()]
        content_tokens = Counter(tokenize(reference_content))

    code_blocks = list(fallback.code_blocks)
    if "code_blocks" in mapping:
        code_blocks = []
        raw_blocks = mapping.get("code_blocks") or []
        if isinstance(raw_blocks, Sequence) and not isinstance(raw_blocks, (str, bytes, bytearray)):
            for raw in raw_blocks:
                if isinstance(raw, Mapping):
                    code_blocks.append((str(raw.get("language") or "").casefold(), str(raw.get("code") or "")))
                else:
                    code_blocks.append(("", str(raw)))

    inline_code = text_list("inline_code", fallback.inline_code)

    default_required = dict(fallback.required_roles)
    default_counts = dict(fallback.expected_role_counts)
    if "required_roles" in mapping:
        raw_roles = mapping.get("required_roles") or {}
        required: dict[str, bool] = {k: False for k in default_required}
        counts: dict[str, int] = {k: 0 for k in default_counts}
        if isinstance(raw_roles, Mapping):
            for raw_name, raw_value in raw_roles.items():
                name = str(raw_name).casefold().replace("audioplayer", "audio")
                if name == "emailpreview":
                    name = "email"
                if name == "codeblock":
                    name = "code"
                count = int(raw_value) if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool) else int(bool(raw_value))
                required[name] = count > 0
                counts[name] = max(0, count)
        default_required.update(required)
        default_counts.update(counts)
    if "expected_role_counts" in mapping and isinstance(mapping.get("expected_role_counts"), Mapping):
        for raw_name, raw_value in mapping["expected_role_counts"].items():
            name = str(raw_name).casefold().replace("audioplayer", "audio")
            if name == "emailpreview":
                name = "email"
            if name == "codeblock":
                name = "code"
            try:
                count = max(0, int(raw_value))
            except (TypeError, ValueError):
                continue
            default_counts[name] = count
            default_required[name] = count > 0

    role_requirements: dict[str, list[dict[str, Any]]] = {}
    raw_role_requirements = mapping.get("role_requirements")
    if isinstance(raw_role_requirements, Mapping):
        for raw_role, raw_items in raw_role_requirements.items():
            if not isinstance(raw_items, Sequence) or isinstance(
                raw_items, (str, bytes, bytearray)
            ):
                continue
            role = str(raw_role).strip().casefold()
            if role.endswith("s"):
                role = role[:-1]
            role = {
                "code_block": "code",
                "codeblock": "code",
                "console_log": "console",
                "consolelog": "console",
                "email_preview": "email",
                "emailpreview": "email",
            }.get(role, role)
            role_requirements[role] = [
                {str(key): value for key, value in raw.items()}
                for raw in raw_items
                if isinstance(raw, Mapping)
            ]

    # Explicit field collections also imply their corresponding role unless a
    # supplied required_roles/expected_role_counts block says otherwise.
    if "tables" in mapping and "required_roles" not in mapping and "expected_role_counts" not in mapping:
        required_tables = [table for table in tables if table.required and table.minimum_count > 0]
        default_required["table"] = bool(required_tables)
        default_counts["table"] = sum(table.minimum_count for table in required_tables)
    if ("actions" in mapping or "source_links" in mapping) and "required_roles" not in mapping and "expected_role_counts" not in mapping:
        required_actions = [
            action for action in explicit_actions if action.required and action.minimum_count > 0
        ]
        default_required["action"] = bool(required_actions)
        default_counts["action"] = sum(action.minimum_count for action in required_actions)
    if "media" in mapping and "required_roles" not in mapping and "expected_role_counts" not in mapping:
        for role, kind in (("image", "image"), ("video", "video"), ("audio", "audioplayer")):
            relevant = [
                item
                for item in media
                if item.required
                and item.minimum_count > 0
                and item.kind.casefold().replace("audio", "audioplayer") == kind
            ]
            representative = any(item.media_policy.casefold() == "representative" for item in relevant)
            count = (
                1
                if representative and relevant
                else sum(item.minimum_count for item in relevant)
            )
            default_required[role] = count > 0
            default_counts[role] = count

    return SourceContract(
        raw_text=fallback.raw_text,
        intent=str(mapping.get("intent") or fallback.intent or "") or None,
        headings=headings,
        tables=tables,
        explicit_actions=explicit_actions,
        actionable_urls=actionable_urls,
        media=media,
        code_blocks=code_blocks,
        inline_code=inline_code,
        exact_values=exact_values,
        content_tokens=content_tokens,
        required_roles=default_required,
        expected_role_counts=default_counts,
        asset_aliases=fallback.asset_aliases,
        content_units=content_units,
        role_requirements=role_requirements,
    )


def _element_references(raw: Mapping[str, Any]) -> list[Any]:
    """Frozen v4 renderer-reference inventory.

    V5 uses ``graph.audit_renderer_graph`` and the shared authoritative
    semantics. Keep this historical inventory unchanged so v4 scores retain
    their original meaning.
    """
    references: list[Any] = []
    children = raw.get("children", [])
    if isinstance(children, list):
        references.extend(children)
    for container in (raw, raw.get("props"), raw.get("repeat")):
        if not isinstance(container, Mapping):
            continue
        for key in ("template", "itemTemplate", "child"):
            value = container.get(key)
            if isinstance(value, str) and value not in references:
                references.append(value)
    return references


def audit_graph(spec: Mapping[str, Any]) -> GraphAudit:
    raw_elements = spec.get("elements")
    elements: Mapping[str, Any] = raw_elements if isinstance(raw_elements, Mapping) else {}
    root_raw = spec.get("root")
    root = root_raw if isinstance(root_raw, str) and root_raw in elements else None

    all_ids = {str(k) for k in elements}
    parents: dict[str, set[str]] = {k: set() for k in all_ids}
    for parent_id, raw in elements.items():
        if not isinstance(raw, Mapping):
            continue
        for child in _element_references(raw):
            if isinstance(child, str) and child in elements:
                parents.setdefault(child, set()).add(str(parent_id))

    reachable: set[str] = set()
    missing: set[str] = set()
    cycles: set[tuple[str, str]] = set()
    depths: dict[str, int] = {}
    colors: dict[str, int] = {}

    def visit(node_id: str, depth: int) -> None:
        color = colors.get(node_id, 0)
        if color == 1:
            return
        # Preserve minimum root distance if a shared node is encountered.
        depths[node_id] = min(depths.get(node_id, depth), depth)
        if color == 2:
            reachable.add(node_id)
            return
        colors[node_id] = 1
        reachable.add(node_id)
        raw = elements.get(node_id, {})
        references = _element_references(raw) if isinstance(raw, Mapping) else []
        for child in references:
            if not isinstance(child, str) or child not in elements:
                missing.add(str(child))
                continue
            if colors.get(child, 0) == 1:
                cycles.add((node_id, child))
                continue
            visit(child, depth + 1)
        colors[node_id] = 2

    if root is not None:
        visit(root, 0)

    return GraphAudit(
        root_exists=root is not None,
        reachable_ids=reachable,
        unreachable_ids=all_ids - reachable,
        missing_references=missing,
        cycle_edges=cycles,
        multi_parent_ids={k for k, v in parents.items() if len(v) > 1 and k in reachable},
        depths=depths,
    )


def resolve_json_pointer(state: Any, pointer: Any) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return None
    current = state
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                return None
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or int(token) >= len(current):
                return None
            current = current[int(token)]
        else:
            return None
    return current


def _table_from_element(
    element_id: str,
    element: Mapping[str, Any],
    state: Any,
) -> OutputTable:
    props = element.get("props", {}) if isinstance(element.get("props"), Mapping) else {}
    columns_raw = props.get("columns", [])
    keys: list[str] = []
    headers: list[str] = []
    if isinstance(columns_raw, list):
        for column in columns_raw:
            if isinstance(column, Mapping):
                key = str(column.get("key", "")).strip()
                label = str(column.get("label", key)).strip()
                if key:
                    keys.append(key)
                    headers.append(label)
    rows_raw = props.get("rows")
    if rows_raw is None:
        rows_raw = resolve_json_pointer(state, props.get("statePath"))
    rows = [dict(row) for row in rows_raw if isinstance(row, Mapping)] if isinstance(rows_raw, list) else []
    return OutputTable(element_id, str(element.get("type")), headers, keys, rows)


def _iter_action_candidates(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                yield item


def _output_actions(
    element_id: str,
    element: Mapping[str, Any],
    table: OutputTable | None,
) -> list[OutputAction]:
    result: list[OutputAction] = []
    props = element.get("props", {}) if isinstance(element.get("props"), Mapping) else {}
    label = normalize_markdown(props.get("label", props.get("title", "")))
    on = element.get("on")
    if isinstance(on, Mapping):
        for event_name, action_value in on.items():
            for action in _iter_action_candidates(action_value):
                action_type = str(action.get("action") or "")
                if action_type not in ALLOWED_ACTIONS:
                    continue
                params = action.get("params", {})
                if not isinstance(params, Mapping):
                    continue
                if action_type == "openUrl":
                    target = params.get("url") or params.get("href") or params.get("link") or params.get("targetUrl")
                    normalized_target = normalize_url(target)
                else:
                    target = (
                        params.get("resultStatePath")
                        or params.get("statePath")
                        or params.get("path")
                        or ""
                    )
                    normalized_target = normalize_markdown(target)
                result.append(
                    OutputAction(
                        element_id,
                        label,
                        normalized_target,
                        f"on.{event_name}",
                        action_type,
                    )
                )

    if table is not None:
        for index, row in enumerate(table.rows):
            row_label = ""
            for key in ("actionLabel", "label", "title", "name", "source", "hotel", "product", "item"):
                if row.get(key):
                    row_label = normalize_markdown(row[key])
                    break
            for key, value in row.items():
                if str(key).casefold() in URL_FIELD_NAMES and value:
                    result.append(
                        OutputAction(
                            f"{element_id}[{index}]",
                            row_label,
                            normalize_url(value),
                            f"row.{key}",
                            "openUrl",
                        )
                    )
    return result


def _collect_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for sub in value.values():
            yield from _collect_strings(sub)
    elif isinstance(value, list):
        for sub in value:
            yield from _collect_strings(sub)


def _is_statically_visible(element: Mapping[str, Any], state: Any) -> bool:
    visible = element.get("visible", True)
    if isinstance(visible, bool):
        return visible
    if isinstance(visible, str) and visible.startswith("/"):
        return bool(resolve_json_pointer(state, visible))
    if isinstance(visible, Mapping):
        for key in ("statePath", "path", "$path"):
            pointer = visible.get(key)
            if isinstance(pointer, str) and pointer.startswith("/"):
                resolved = resolve_json_pointer(state, pointer)
                if "equals" in visible:
                    return resolved == visible.get("equals")
                return bool(resolved)
        if isinstance(visible.get("value"), bool):
            return bool(visible["value"])
    # Unknown computed visibility cannot safely be treated as definitely hidden.
    return True


def collect_output_evidence(spec: Mapping[str, Any], audit: GraphAudit) -> OutputEvidence:
    raw_elements = spec.get("elements", {})
    elements: Mapping[str, Mapping[str, Any]] = {
        str(k): v for k, v in raw_elements.items() if isinstance(v, Mapping)
    } if isinstance(raw_elements, Mapping) else {}
    state = spec.get("state", {})

    visible_blocks: list[str] = []
    headings: list[str] = []
    heading_variants: list[str] = []
    tables: list[OutputTable] = []
    actions: list[OutputAction] = []
    media: list[MediaRef] = []
    type_counts: Counter[str] = Counter()

    # Traverse in declared tree order rather than element-map insertion order.
    ordered: list[str] = []
    visited: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id in visited or node_id not in audit.reachable_ids:
            return
        raw = elements.get(node_id, {})
        if not _is_statically_visible(raw, state):
            return
        visited.add(node_id)
        ordered.append(node_id)
        for child in _element_references(raw):
            if isinstance(child, str):
                walk(child)

    root = spec.get("root")
    if isinstance(root, str):
        walk(root)

    seen_data_bindings: set[str] = set()

    for element_id in ordered:
        element = elements.get(element_id, {})
        element_type = str(element.get("type", ""))
        type_counts[element_type] += 1
        props = element.get("props", {}) if isinstance(element.get("props"), Mapping) else {}
        table: OutputTable | None = None

        if element_type == "Text":
            text = normalize_markdown(props.get("text", ""))
            if text:
                visible_blocks.append(text)
            variant = str(props.get("variant", "body"))
            if variant in {"h1", "h2", "h3"} and text:
                headings.append(text)
                heading_variants.append(variant)
        elif element_type in {"Table", "Chart"}:
            table = _table_from_element(element_id, element, state)
            tables.append(table)
            # A Chart and a Table may intentionally share the same state rows.
            # Count that semantic dataset only once for content precision/recall;
            # role fidelity still scores both component mappings independently.
            state_path = props.get("statePath")
            if isinstance(state_path, str) and state_path:
                binding_key = f"state:{state_path}"
            else:
                binding_key = "rows:" + json.dumps(table.rows, sort_keys=True, ensure_ascii=False, default=str)
            if binding_key not in seen_data_bindings:
                seen_data_bindings.add(binding_key)
                visible_blocks.extend(h for h in table.headers if h)
                for row in table.rows:
                    for key in table.keys or list(row.keys()):
                        value = row.get(key)
                        if value is not None and not (
                            str(key).casefold() in URL_FIELD_NAMES
                        ):
                            visible_blocks.extend(str(v) for v in _collect_strings(value))
            for key in ("title", "subtitle", "yLabel"):
                if props.get(key):
                    visible_blocks.append(str(props[key]))
        elif element_type in {"Formula", "CodeBlock", "ConsoleLog"}:
            for key in ("title", "subtitle", "text", "latex", "result", "code"):
                if props.get(key) is not None:
                    visible_blocks.extend(str(v) for v in _collect_strings(props[key]))
        elif element_type == "EmailPreview":
            for key in ("title", "subtitle", "subject", "to", "from", "date", "role", "company", "body", "signature", "context", "metadata"):
                if props.get(key) is not None:
                    visible_blocks.extend(str(v) for v in _collect_strings(props[key]))
        elif element_type == "Button":
            if props.get("label"):
                visible_blocks.append(str(props["label"]))
        elif element_type in {"TextField", "CheckBox", "ChoicePicker", "Slider", "DateTimeInput"}:
            for key in ("label", "value", "options"):
                if props.get(key) is not None:
                    visible_blocks.extend(str(v) for v in _collect_strings(props[key]))
        elif element_type in MEDIA_TYPES:
            url = props.get("url", props.get("name", ""))
            alt = normalize_markdown(props.get("alt", props.get("description", "")))
            if url:
                media.append(MediaRef(element_type, normalize_url(url), alt))
            if alt:
                visible_blocks.append(alt)

        actions.extend(_output_actions(element_id, element, table))

    visible_text = "\n".join(visible_blocks)
    exact_values = Counter(normalize_match_text(v) for v in _EXACT_VALUE_RE.findall(visible_text))
    exact_values.update(normalize_match_text(v) for v in _DATE_RE.findall(visible_text))
    exact_values.pop("", None)

    valid_type_counts: Counter[str] = Counter()
    for element_id in ordered:
        element = elements.get(element_id, {})
        if _element_contract_score(element, state) >= 1.0:
            valid_type_counts[str(element.get("type", ""))] += 1

    return OutputEvidence(
        elements=elements,
        reachable_ids=set(ordered),
        visible_blocks=visible_blocks,
        headings=headings,
        heading_variants=heading_variants,
        tables=tables,
        actions=actions,
        media=media,
        type_counts=type_counts,
        content_tokens=Counter(tokenize(visible_text)),
        exact_values=exact_values,
        valid_type_counts=valid_type_counts,
    )


def _validate_action(action: Mapping[str, Any]) -> bool:
    name = action.get("action")
    if name not in ALLOWED_ACTIONS:
        return False
    params = action.get("params", {})
    if params is not None and not isinstance(params, Mapping):
        return False
    if name == "openUrl":
        url = (
            params.get("url") or params.get("href") or params.get("link") or params.get("targetUrl")
            if isinstance(params, Mapping)
            else None
        )
        return isinstance(url, str) and bool(normalize_url(url))
    if name in {"setState", "pushState", "removeState"}:
        path = params.get("statePath") or params.get("path") if isinstance(params, Mapping) else None
        if not isinstance(path, str) or not path.strip().startswith("/"):
            return False
        if name == "removeState" and "index" not in params:
            return False
    if name == "validateForm" and isinstance(params, Mapping):
        path = params.get("resultStatePath") or params.get("statePath") or params.get("path")
        if path is not None and (not isinstance(path, str) or not path.strip().startswith("/")):
            return False
    return True


def basic_schema_contract(spec: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    failures: list[str] = []
    if set(spec) - {"root", "state", "elements"}:
        failures.append("top_level_extra_properties")
    if not isinstance(spec.get("root"), str) or not spec.get("root"):
        failures.append("invalid_root")
    if "state" in spec and not isinstance(spec.get("state"), Mapping):
        failures.append("invalid_state")
    raw_elements = spec.get("elements")
    if not isinstance(raw_elements, Mapping) or not raw_elements:
        failures.append("invalid_elements")
        return 0.0, {"failures": failures}

    total_checks = 0
    passed_checks = 0
    for element_id, raw in raw_elements.items():
        checks = 0
        passed = 0
        checks += 1
        passed += int(isinstance(element_id, str) and bool(element_id))
        checks += 1
        passed += int(isinstance(raw, Mapping))
        if not isinstance(raw, Mapping):
            total_checks += checks
            passed_checks += passed
            continue
        checks += 1
        passed += int(raw.get("type") in ALLOWED_TYPES)
        checks += 1
        passed += int(isinstance(raw.get("props"), Mapping))
        checks += 1
        passed += int(isinstance(raw.get("children"), list) and all(isinstance(v, str) for v in raw.get("children", [])))
        checks += 1
        passed += int("action" not in (raw.get("props") or {}) if isinstance(raw.get("props"), Mapping) else False)
        checks += 1
        on = raw.get("on")

        def action_value_ok(value: Any) -> bool:
            candidates = list(_iter_action_candidates(value))
            return bool(candidates) and all(_validate_action(candidate) for candidate in candidates)

        on_ok = on is None or (
            isinstance(on, Mapping)
            and bool(on)
            and all(action_value_ok(value) for value in on.values())
        )
        passed += int(on_ok)
        total_checks += checks
        passed_checks += passed
    return safe_div(passed_checks, total_checks), {"failures": failures, "checks": total_checks}


def _element_contract_score(element: Mapping[str, Any], state: Any) -> float:
    t = element.get("type")
    props = element.get("props", {}) if isinstance(element.get("props"), Mapping) else {}
    children = element.get("children", []) if isinstance(element.get("children"), list) else []

    checks: list[bool] = [t in ALLOWED_TYPES, isinstance(props, Mapping), isinstance(children, list)]
    if t == "Text":
        checks.append(bool(normalize_markdown(props.get("text", ""))))
    elif t in {"Stack", "Column", "Row"}:
        if t == "Stack":
            checks.append(props.get("direction") in {"horizontal", "vertical"})
        checks.append(bool(children))
    elif t in {"Card", "List"}:
        checks.append(bool(children))
    elif t == "Button":
        checks.append(bool(normalize_markdown(props.get("label", ""))))
        on = element.get("on", {})
        press = on.get("press") if isinstance(on, Mapping) else None
        checks.append(any(_validate_action(a) for a in _iter_action_candidates(press)))
    elif t in {"Table", "Chart"}:
        table = _table_from_element("", element, state)
        checks.extend([bool(table.keys), bool(table.rows)])
        if t == "Chart":
            checks.extend(
                [
                    bool(props.get("xKey")),
                    bool(props.get("yKey")),
                    props.get("xKey") in table.keys,
                    props.get("yKey") in table.keys,
                ]
            )
            y_key = props.get("yKey")
            if y_key:
                numeric = 0
                for row in table.rows:
                    value = normalize_match_text(row.get(y_key, "")).replace(",", "")
                    numeric += int(bool(re.search(r"[-+]?\d+(?:\.\d+)?", value)))
                checks.append(numeric >= max(1, len(table.rows) // 2))
    elif t == "Formula":
        checks.append(bool(props.get("text") or props.get("latex")))
    elif t in {"CodeBlock", "ConsoleLog"}:
        checks.append(bool(str(props.get("code", "")).strip()))
    elif t == "EmailPreview":
        checks.extend([bool(props.get("subject") or props.get("title")), bool(props.get("body"))])
    elif t == "Image":
        checks.append(bool(props.get("url")))
    elif t == "Icon":
        checks.append(bool(props.get("name")))
    elif t in {"Video", "AudioPlayer"}:
        checks.append(bool(props.get("url")))
    elif t == "Tabs":
        tabs = props.get("tabs")
        checks.append(isinstance(tabs, list) and bool(tabs))
    elif t == "TextField":
        checks.append(bool(props.get("label")))
    elif t == "CheckBox":
        checks.extend([bool(props.get("label")), isinstance(props.get("value"), bool)])
    elif t == "ChoicePicker":
        checks.extend([bool(props.get("label")), isinstance(props.get("options"), list)])
    elif t == "Slider":
        checks.extend(all(isinstance(props.get(k), (int, float)) for k in ("min", "max", "value")) for _ in [0])
    elif t == "DateTimeInput":
        checks.append(props.get("value") is not None)
    repeat = element.get("repeat")
    if repeat is not None:
        repeat_ok = isinstance(repeat, Mapping)
        state_path = repeat.get("statePath") if isinstance(repeat, Mapping) else None
        repeat_ok = repeat_ok and isinstance(state_path, str) and state_path.startswith("/")
        repeat_ok = repeat_ok and isinstance(resolve_json_pointer(state, state_path), list)
        checks.append(bool(repeat_ok))
    return sum(checks) / len(checks) if checks else 1.0


def type_contract_score(spec: Mapping[str, Any], output: OutputEvidence) -> tuple[float, dict[str, float]]:
    state = spec.get("state", {})
    by_type: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for element_id in output.reachable_ids:
        element = output.elements.get(element_id)
        if not isinstance(element, Mapping):
            continue
        score = _element_contract_score(element, state)
        by_type[str(element.get("type", ""))].append(score)
        all_scores.append(score)
    macro = sum(sum(v) / len(v) for v in by_type.values()) / len(by_type) if by_type else 0.0
    micro = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return 0.5 * macro + 0.5 * micro, {k: sum(v) / len(v) for k, v in by_type.items()}


def heading_fidelity(source: SourceContract, output: OutputEvidence) -> float | None:
    if not source.headings:
        return None
    if not output.headings:
        return 0.0
    used: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for src_index, heading in enumerate(source.headings):
        candidates = [
            (text_similarity(heading, actual), idx)
            for idx, actual in enumerate(output.headings)
            if idx not in used
        ]
        if not candidates:
            continue
        sim, idx = max(candidates)
        if sim >= 0.62:
            used.add(idx)
            matched.append((src_index, idx, sim))
    precision = safe_div(len(matched), len(output.headings))
    recall = safe_div(len(matched), len(source.headings))
    lexical = sum(v for _, _, v in matched) / len(matched) if matched else 0.0
    order_pairs = 0
    ordered_pairs = 0
    for a, b in zip(matched, matched[1:]):
        order_pairs += 1
        ordered_pairs += int(a[1] < b[1])
    order = safe_div(ordered_pairs, order_pairs, 1.0)
    return 0.60 * f_beta(precision, recall, beta=2.0) + 0.25 * lexical + 0.15 * order


def _source_table_values(table: SourceTable) -> Counter[str]:
    return Counter(normalize_match_text(cell) for cell in table.cells if normalize_match_text(cell))


def _output_table_values(table: OutputTable) -> Counter[str]:
    values = [normalize_match_text(h) for h in table.headers if normalize_match_text(h)]
    for row in table.rows:
        for key in table.keys or list(row):
            if str(key).casefold() in URL_FIELD_NAMES:
                continue
            value = row.get(key)
            for item in _collect_strings(value):
                norm = normalize_match_text(item)
                if norm:
                    values.append(norm)
    return Counter(values)


def table_pair_score(source: SourceTable, output: OutputTable, beta: float = 2.0) -> float:
    src = _source_table_values(source)
    out = _output_table_values(output)
    precision, recall = counter_pr(out, src)
    cell_score = f_beta(precision, recall, beta=beta)

    header_sims: list[float] = []
    for index, expected in enumerate(source.headers):
        if index < len(output.headers):
            header_sims.append(text_similarity(expected, output.headers[index]))
        else:
            header_sims.append(0.0)
    header_score = sum(header_sims) / len(header_sims) if header_sims else 1.0
    row_count = 1.0 - min(1.0, abs(len(source.rows) - len(output.rows)) / max(1, len(source.rows)))
    column_count = 1.0 - min(1.0, abs(len(source.headers) - len(output.headers)) / max(1, len(source.headers)))
    return 0.62 * cell_score + 0.20 * header_score + 0.10 * row_count + 0.08 * column_count


def table_fidelity(source: SourceContract, output: OutputEvidence, beta: float = 2.0) -> float | None:
    if not source.tables:
        return None
    candidate_tables = [t for t in output.tables if t.element_type == "Table"]
    if not candidate_tables:
        return 0.0
    available = set(range(len(candidate_tables)))
    scores: list[float] = []
    for expected in source.tables:
        candidates = [
            (table_pair_score(expected, candidate_tables[index], beta), index)
            for index in available
        ]
        if not candidates:
            scores.append(0.0)
            continue
        score, index = max(candidates)
        available.remove(index)
        scores.append(score)
    source_recall = sum(scores) / len(scores)
    # Extra tables are not automatically bad; many prose lists are appropriately
    # converted to compact tables.  Penalize only clearly empty extra tables.
    empty_extra = sum(1 for i in available if not candidate_tables[i].rows)
    extra_utility = 1.0 - safe_div(empty_extra, max(1, len(candidate_tables)))
    return 0.95 * source_recall + 0.05 * extra_utility


def _url_equivalent(expected: str, actual: str, aliases: Mapping[str, set[str]]) -> bool:
    e, a = normalize_url(expected), normalize_url(actual)
    if e == a:
        return True
    return a in aliases.get(e, set()) or e in aliases.get(a, set())


def _action_target_equivalent(
    expected: ActionRef,
    actual: OutputAction,
    aliases: Mapping[str, set[str]],
) -> bool:
    if expected.action_type == "openUrl":
        return _url_equivalent(expected.url, actual.url, aliases)
    return normalize_match_text(expected.url) == normalize_match_text(actual.url)


def action_fidelity(source: SourceContract, output: OutputEvidence) -> float | None:
    if not source.actionable_urls and not source.explicit_actions:
        return None
    actual = output.actions
    actual_urls = [a.url for a in actual if a.action_type == "openUrl" and a.url]
    expected_urls = sorted(source.actionable_urls | {a.url for a in source.explicit_actions})

    matched_expected = 0
    for expected in expected_urls:
        if any(_url_equivalent(expected, found, source.asset_aliases) for found in actual_urls):
            matched_expected += 1
    url_recall = safe_div(matched_expected, len(expected_urls), 1.0)
    supported_actual = sum(
        1
        for found in actual_urls
        if any(_url_equivalent(expected, found, source.asset_aliases) for expected in expected_urls)
    )
    url_precision = safe_div(supported_actual, len(actual_urls), 1.0)
    url_f1 = f_beta(url_precision, url_recall, beta=1.5)

    explicit_score: float | None = None
    if source.explicit_actions:
        used: set[int] = set()
        scores: list[float] = []
        for expected in source.explicit_actions:
            candidates: list[tuple[float, int]] = []
            for idx, found in enumerate(actual):
                if idx in used:
                    continue
                target_match = float(_action_target_equivalent(expected, found, source.asset_aliases))
                label_match = text_similarity(expected.label, found.label)
                type_match = float(expected.action_type == found.action_type)
                candidates.append((0.55 * target_match + 0.25 * label_match + 0.20 * type_match, idx))
            if not candidates:
                scores.append(0.0)
                continue
            score, idx = max(candidates)
            if score >= 0.45:
                used.add(idx)
            scores.append(score)
        explicit_score = sum(scores) / len(scores)

    action_signatures = {
        (action.action_type, action.url, normalize_match_text(action.label))
        for action in actual
    }
    duplicate_utility = safe_div(len(action_signatures), len(actual), 1.0)
    if explicit_score is None:
        return 0.88 * url_f1 + 0.12 * duplicate_utility
    return 0.58 * url_f1 + 0.34 * explicit_score + 0.08 * duplicate_utility


def media_fidelity(source: SourceContract, output: OutputEvidence) -> float | None:
    source_media = [m for m in source.media if m.kind in MEDIA_TYPES]
    if not source_media:
        return None
    if not output.media:
        return 0.0
    scores: list[float] = []
    for expected in source_media:
        candidates = []
        for found in output.media:
            kind = float(found.kind == expected.kind)
            url = float(_url_equivalent(expected.url, found.url, source.asset_aliases))
            alt = text_similarity(expected.alt, found.alt) if expected.alt else 1.0
            candidates.append(0.35 * kind + 0.50 * url + 0.15 * alt)
        scores.append(max(candidates, default=0.0))
    # Prompt asks for representative media, not one node per URL.  Reward the
    # best representative coverage and cap the required count at four.
    required = min(4, len(source_media))
    top = sorted(scores, reverse=True)[:required]
    count_coverage = min(1.0, len([m for m in output.media if m.kind in {x.kind for x in source_media}]) / required)
    return 0.80 * (sum(top) / required) + 0.20 * count_coverage


def role_coverage(source: SourceContract, output: OutputEvidence) -> tuple[float | None, dict[str, float | None]]:
    type_present = output.valid_type_counts
    role_values: dict[str, float | None] = {}
    role_map = {
        "table": "Table",
        "chart": "Chart",
        "email": "EmailPreview",
        "formula": "Formula",
        "code": "CodeBlock",
        "console": "ConsoleLog",
        "image": "Image",
        "video": "Video",
        "audio": "AudioPlayer",
        "action": "Button",
    }
    for role, component_type in role_map.items():
        if not source.required_roles.get(role, False):
            role_values[role] = None
            continue
        expected_count = max(1, source.expected_role_counts.get(role, 1))
        if role == "action":
            actual_count = len({a.url for a in output.actions})
        else:
            actual_count = type_present.get(component_type, 0)
        role_values[role] = min(1.0, actual_count / expected_count)
    applicable = [v for v in role_values.values() if v is not None]
    return (sum(applicable) / len(applicable) if applicable else None), role_values


def component_appropriateness(source: SourceContract, output: OutputEvidence) -> float:
    penalties: list[float] = []
    # Raw markdown control tokens should not leak into Text nodes.
    text_blocks = [
        normalize_markdown(output.elements[eid].get("props", {}).get("text", ""))
        for eid in output.reachable_ids
        if output.elements.get(eid, {}).get("type") == "Text"
        and isinstance(output.elements.get(eid, {}).get("props"), Mapping)
    ]
    if text_blocks:
        leaked = sum(
            1
            for t in text_blocks
            if re.search(r"(?:^|\n)#{1,6}\s|\|.+\||```|\*\*", t)
        )
        penalties.append(safe_div(leaked, len(text_blocks)))
    # Unsupported or empty types are covered by contracts; this check catches
    # source-code dumping into Text when dedicated components were required.
    if source.required_roles.get("console"):
        command_tokens = {normalize_match_text(x) for x in source.inline_code if x}
        text_command_hits = sum(
            1 for block in text_blocks if any(c and c in normalize_match_text(block) for c in command_tokens)
        )
        if output.type_counts.get("ConsoleLog", 0) == 0:
            penalties.append(min(1.0, text_command_hits / max(1, len(command_tokens))))
    return 1.0 - (sum(penalties) / len(penalties) if penalties else 0.0)


def _semantic_duplicate_utility(output: OutputEvidence) -> float:
    normalized = [normalize_match_text(block) for block in output.visible_blocks]
    normalized = [v for v in normalized if len(v.split()) >= 3]
    if not normalized:
        return 1.0
    counts = Counter(normalized)
    duplicate_instances = sum(max(0, count - 1) for count in counts.values())
    return 1.0 - safe_div(duplicate_instances, len(normalized))


def _redundant_wrapper_utility(output: OutputEvidence) -> float:
    """Renderer-aware wrapper check.

    Card -> Stack is intentionally *not* penalized; it is the project's normal
    way to combine surface styling with padded flex layout.  Only mergeable
    same-kind layout chains or prop-less one-child Stacks are treated as likely
    no-op wrappers.
    """
    candidates = 0
    redundant = 0
    for element_id in output.reachable_ids:
        element = output.elements.get(element_id, {})
        t = element.get("type")
        if t not in LAYOUT_TYPES:
            continue
        children = element.get("children", [])
        if not isinstance(children, list) or len(children) != 1:
            continue
        child = output.elements.get(children[0], {})
        child_type = child.get("type")
        candidates += 1
        props = element.get("props", {}) if isinstance(element.get("props"), Mapping) else {}
        if t in {"Stack", "Column", "Row", "List"} and child_type == t:
            redundant += 1
        elif t == "Stack" and not props:
            redundant += 1
    return 1.0 - safe_div(redundant, max(1, candidates))


def _text_chunking_utility(output: OutputEvidence, source: SourceContract) -> float:
    text_lengths: list[int] = []
    for element_id in output.reachable_ids:
        element = output.elements.get(element_id, {})
        if element.get("type") != "Text":
            continue
        props = element.get("props", {}) if isinstance(element.get("props"), Mapping) else {}
        variant = props.get("variant", "body")
        if variant in {"h1", "h2", "h3", "chip", "label", "caption"}:
            continue
        text_lengths.append(len(tokenize(props.get("text", ""))))
    if not text_lengths:
        return 1.0 if not source.content_tokens else 0.0
    total = sum(text_lengths)
    giant = sum(max(0, length - 90) for length in text_lengths)
    giant_utility = 1.0 - safe_div(giant, max(1, total))

    tiny_share = safe_div(sum(1 for length in text_lengths if 0 < length <= 2), len(text_lengths))
    tiny_utility = low_is_good(tiny_share, 0.05, 0.45)
    fragment_density = safe_div(len(text_lengths), max(1, total))
    density_utility = low_is_good(fragment_density, 0.12, 0.50)

    # A single giant document dump should not earn high translation reward.
    longest_share = max(text_lengths) / max(1, sum(source.content_tokens.values()))
    dump_utility = low_is_good(longest_share, 0.18, 0.55)
    return (
        0.35 * giant_utility
        + 0.25 * tiny_utility
        + 0.15 * density_utility
        + 0.25 * dump_utility
    )


def _layout_depth_utility(audit: GraphAudit) -> float:
    return plateau_utility(audit.max_depth, good_low=2, good_high=6, bad_low=0, bad_high=10)


def _fanout_utility(output: OutputEvidence) -> float:
    container_counts: list[int] = []
    for eid in output.reachable_ids:
        element = output.elements.get(eid, {})
        if element.get("type") in CONTAINER_TYPES:
            children = element.get("children", [])
            if isinstance(children, list):
                container_counts.append(len(children))
    if not container_counts:
        return 0.0
    high = sum(1 for count in container_counts if count > 12)
    return 1.0 - safe_div(high, len(container_counts))


def _root_layout_utility(spec: Mapping[str, Any], output: OutputEvidence) -> float:
    root = spec.get("root")
    element = output.elements.get(str(root), {})
    return 1.0 if element.get("type") in {"Stack", "Column", "List", "Card"} else 0.0


def _heading_grouping_utility(output: OutputEvidence) -> float | None:
    heading_ids: list[str] = []
    for eid in output.reachable_ids:
        element = output.elements.get(eid, {})
        props = element.get("props", {}) if isinstance(element.get("props"), Mapping) else {}
        if element.get("type") == "Text" and props.get("variant") in {"h1", "h2", "h3"}:
            heading_ids.append(eid)
    if not heading_ids:
        return None
    parent_map: dict[str, list[str]] = defaultdict(list)
    for pid in output.reachable_ids:
        element = output.elements.get(pid, {})
        for child in element.get("children", []) if isinstance(element.get("children"), list) else []:
            if isinstance(child, str):
                parent_map[child].append(pid)
    grouped = 0
    for hid in heading_ids:
        good = False
        for parent in parent_map.get(hid, []):
            siblings = output.elements.get(parent, {}).get("children", [])
            if isinstance(siblings, list) and any(s != hid for s in siblings):
                good = True
                break
        grouped += int(good)
    return safe_div(grouped, len(heading_ids))


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical_json_value(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, list):
        return [_canonical_json_value(v) for v in value]
    return value


def _canonical_reachable_payload_length(spec: Mapping[str, Any], audit: GraphAudit) -> int:
    """ID-rename and element-map-order invariant reachable payload length."""
    elements = spec.get("elements") if isinstance(spec.get("elements"), Mapping) else {}
    root = spec.get("root")

    def without_element_refs(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return _canonical_json_value(value)
        return _canonical_json_value(
            {
                key: item
                for key, item in value.items()
                if not (key in {"template", "itemTemplate", "child"} and isinstance(item, str))
            }
        )

    def node(element_id: str, stack: frozenset[str]) -> Any:
        if element_id in stack:
            return {"cycle": True}
        raw = elements.get(element_id, {})
        if not isinstance(raw, Mapping):
            return {"invalid": True}
        references = _element_references(raw)
        return {
            "type": raw.get("type"),
            "props": without_element_refs(raw.get("props", {})),
            "on": _canonical_json_value(raw.get("on", {})),
            "repeat": without_element_refs(raw.get("repeat", {})),
            "visible": _canonical_json_value(raw.get("visible")),
            "watch": _canonical_json_value(raw.get("watch", {})),
            "children": [
                node(child, stack | {element_id})
                for child in references
                if isinstance(child, str) and child in audit.reachable_ids
            ],
        }

    payload = {
        "root": node(str(root), frozenset()) if audit.root_exists else None,
        "state": _canonical_json_value(spec.get("state", {})),
    }
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _json_efficiency_utility(
    spec: Mapping[str, Any],
    audit: GraphAudit,
    source: SourceContract,
    config: RewardConfig,
) -> float:
    ratio = _canonical_reachable_payload_length(spec, audit) / max(1, len(source.raw_text))
    return low_is_good(ratio, config.good_json_to_source_ratio, config.bad_json_to_source_ratio)


def _accessibility_value(props: Mapping[str, Any], *names: str) -> Any:
    nested = props.get("accessibility")
    nested_map = nested if isinstance(nested, Mapping) else {}
    for name in names:
        value = props.get(name)
        if value not in (None, ""):
            return value
        value = nested_map.get(name)
        if value not in (None, ""):
            return value
    return None


def _accessibility_score(output: OutputEvidence) -> tuple[float | None, dict[str, float | None]]:
    button_scores: list[float] = []
    image_scores: list[float] = []
    form_scores: list[float] = []
    icon_scores: list[float] = []
    for eid in output.reachable_ids:
        element = output.elements.get(eid, {})
        t = element.get("type")
        props = element.get("props", {}) if isinstance(element.get("props"), Mapping) else {}
        if t == "Button":
            name = props.get("label") or _accessibility_value(
                props, "label", "accessibilityLabel", "contentDescription", "alt", "title"
            )
            button_scores.append(float(bool(normalize_markdown(name or ""))))
        elif t == "Image":
            decorative = bool(_accessibility_value(props, "decorative", "ariaHidden"))
            if not decorative:
                name = props.get("alt") or _accessibility_value(
                    props, "label", "accessibilityLabel", "contentDescription", "title"
                )
                image_scores.append(float(bool(normalize_markdown(name or ""))))
        elif t in {"TextField", "CheckBox", "ChoicePicker", "Slider", "DateTimeInput"}:
            name = props.get("label") or _accessibility_value(
                props, "label", "accessibilityLabel", "contentDescription", "title"
            )
            form_scores.append(float(bool(normalize_markdown(name or ""))))
        elif t == "Icon":
            # Icons can be decorative; require an accessibility label only when
            # they are interactive or have no adjacent text.  Static analysis
            # cannot prove adjacency reliably, so use a mild optional signal.
            on = element.get("on")
            if on:
                name = props.get("alt") or props.get("label") or props.get("title") or _accessibility_value(
                    props, "label", "accessibilityLabel", "contentDescription"
                )
                icon_scores.append(float(bool(name)))
    atomics = {
        "button_names": sum(button_scores) / len(button_scores) if button_scores else None,
        "image_alt": sum(image_scores) / len(image_scores) if image_scores else None,
        "form_labels": sum(form_scores) / len(form_scores) if form_scores else None,
        "interactive_icon_names": sum(icon_scores) / len(icon_scores) if icon_scores else None,
    }
    applicable = [v for v in atomics.values() if v is not None]
    return (sum(applicable) / len(applicable) if applicable else None), atomics


def _semantic_role_cap(
    source: SourceContract,
    role_values: Mapping[str, float | None],
    config: RewardConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """Return semantic policy cap and structured reasons."""
    cap = 1.0
    active: list[dict[str, Any]] = []

    def activate(name: str, role: str, value: float) -> None:
        nonlocal cap
        candidate = float(config.caps[name])
        cap = min(cap, candidate)
        active.append({"kind": "semantic", "name": name, "role": role, "value": value, "cap": candidate})

    for role, required in source.required_roles.items():
        if not required:
            continue
        value = clamp01(role_values.get(role) or 0.0)
        if role == "table":
            if value == 0.0:
                activate("missing_table", role, value)
            elif value < 1.0:
                activate("partial_table", role, value)
        elif role == "chart":
            if value == 0.0:
                activate("missing_chart", role, value)
            elif value < 0.67:
                activate("chart_below_two_thirds", role, value)
            elif value < 1.0:
                activate("partial_chart", role, value)
        elif role in {"email", "formula", "code", "console"}:
            if value == 0.0:
                activate("missing_special_role", role, value)
            elif value < 1.0:
                activate("partial_special_role", role, value)
        elif role == "action":
            if value == 0.0:
                activate("missing_action", role, value)
            elif value < 0.50:
                activate("action_below_half", role, value)
            elif value < 1.0:
                activate("partial_action", role, value)
        # Image/video/audio omissions are continuous under the representative-
        # media policy and therefore do not create hard caps.
    return cap, active


def _critical_cap(
    *,
    parse_stage: str,
    schema_score: float,
    audit: GraphAudit,
    type_score: float,
    render_ok: bool | None,
    config: RewardConfig,
) -> tuple[float, list[dict[str, Any]]]:
    cap = 1.0
    active: list[dict[str, Any]] = []

    def activate(name: str, evidence: Any = None) -> None:
        nonlocal cap
        candidate = float(config.caps[name])
        cap = min(cap, candidate)
        active.append({"kind": "integrity", "name": name, "evidence": evidence, "cap": candidate})

    if parse_stage in {"empty", "json_parse_error", "json_non_object"}:
        active.append({"kind": "integrity", "name": "parse_failure", "cap": 0.0})
        return 0.0, active
    if not audit.root_exists:
        activate("missing_root", True)
        return cap, active
    if schema_score < 0.75:
        activate("schema_severe", schema_score)
    elif schema_score < 0.95:
        activate("schema_moderate", schema_score)
    if audit.cycle_edges or audit.missing_references:
        activate(
            "reference_or_cycle",
            {"cycles": sorted(audit.cycle_edges), "missing": sorted(audit.missing_references)},
        )
    if audit.reachable_fraction < 0.90:
        activate("reachability_below_90", audit.reachable_fraction)
    elif audit.reachable_fraction < 1.0:
        activate("partial_reachability", audit.reachable_fraction)
    if type_score < 0.65:
        activate("type_contract_severe", type_score)
    if render_ok is False:
        activate("render_failure", False)
    return cap, active


def score_genui_completion(
    completion: Any,
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    expected_ui_contract: Mapping[str, Any] | None = None,
    render_ok: bool | None = None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score one completion.

    Returns a dense GRPO reward in [-1, 1] and an uncalibrated engineering
    quality score in [0, 100].  The latter is useful for dashboards, but it
    should be calibrated to human pairwise judgments before being interpreted
    as an equal-interval quality scale.
    """
    cfg = config or RewardConfig()
    completion_text = completion_to_text(completion)
    spec, parse_stage = extract_json_object(completion_text)
    fallback_source = parse_source_contract(response_text, intent=intent, assets=assets)
    source = (
        source_contract_from_mapping(expected_ui_contract, fallback=fallback_source)
        if isinstance(expected_ui_contract, Mapping)
        else fallback_source
    )

    if spec is None:
        # Dense partial format reward avoids a zero-variance all-invalid group.
        looks_object = "{" in completion_text and "}" in completion_text
        looks_contract = all(token in completion_text for token in ('"root"', '"elements"'))
        q = 0.01 + 0.02 * float(looks_object) + 0.02 * float(looks_contract)
        reward = 2.0 * q - 1.0
        return RewardBreakdown(
            reward=reward,
            quality_0_100=100.0 * q,
            quality_0_1=q,
            cap_0_1=0.0,
            parse_stage=parse_stage,
            dimensions={},
            atomics={},
            evidence={"looks_object": looks_object, "looks_contract": looks_contract},
            active_caps=[{"kind": "integrity", "name": "parse_failure", "cap": 0.0}],
            errors=[parse_stage],
        )

    schema_score, schema_evidence = basic_schema_contract(spec)
    audit = audit_graph(spec)
    output = collect_output_evidence(spec, audit)
    type_score, type_evidence = type_contract_score(spec, output)

    content_p, content_r = counter_pr(output.content_tokens, source.content_tokens)
    content_score = (
        None
        if not source.content_tokens and not output.content_tokens
        else f_beta(content_p, content_r, cfg.content_beta)
    )
    value_p, value_r = counter_pr(output.exact_values, source.exact_values)
    value_score = (
        None
        if not source.exact_values
        else f_beta(value_p, value_r, cfg.value_beta)
    )
    headings_score = heading_fidelity(source, output)
    tables_score = table_fidelity(source, output, cfg.table_beta)
    actions_score = action_fidelity(source, output)
    media_score = media_fidelity(source, output)

    role_score, role_values = role_coverage(source, output)
    appropriateness = component_appropriateness(source, output)

    resolved_render_ok = render_ok
    if resolved_render_ok is None and cfg.render_check is not None and audit.root_exists:
        try:
            resolved_render_ok = bool(cfg.render_check(spec))
        except Exception:
            resolved_render_ok = False

    def aw(group: str, name: str) -> float:
        return float(cfg.atomic_weights[group][name])

    integrity_atomics = {
        "schema_contract": Atomic(schema_score, aw("integrity", "schema_contract"), str(schema_evidence)),
        "declared_root_reachability": Atomic(audit.reachable_fraction, aw("integrity", "declared_root_reachability")),
        "reference_integrity": Atomic(
            1.0 if not audit.missing_references and not audit.cycle_edges else 0.0,
            aw("integrity", "reference_integrity"),
        ),
        "type_semantic_contract": Atomic(type_score, aw("integrity", "type_semantic_contract"), str(type_evidence)),
        "native_render_smoke": Atomic(
            None if resolved_render_ok is None else float(resolved_render_ok),
            aw("integrity", "native_render_smoke"),
        ),
    }
    fidelity_atomics = {
        "visible_content_multiset_fbeta": Atomic(content_score, aw("fidelity", "visible_content_multiset_fbeta")),
        "exact_numbers_dates_units_fbeta": Atomic(value_score, aw("fidelity", "exact_numbers_dates_units_fbeta")),
        "markdown_table_fidelity": Atomic(tables_score, aw("fidelity", "markdown_table_fidelity")),
        "heading_fidelity_and_order": Atomic(headings_score, aw("fidelity", "heading_fidelity_and_order")),
        "action_and_source_link_fidelity": Atomic(actions_score, aw("fidelity", "action_and_source_link_fidelity")),
        "media_fidelity": Atomic(media_score, aw("fidelity", "media_fidelity")),
    }
    semantic_atomics = {
        "required_component_roles": Atomic(role_score, aw("semantic_mapping", "required_component_roles"), str(role_values)),
        "component_appropriateness": Atomic(appropriateness, aw("semantic_mapping", "component_appropriateness")),
    }
    hierarchy_atomics = {
        "root_layout": Atomic(_root_layout_utility(spec, output), aw("hierarchy", "root_layout")),
        "root_distance_depth": Atomic(_layout_depth_utility(audit), aw("hierarchy", "root_distance_depth")),
        "container_fanout": Atomic(_fanout_utility(output), aw("hierarchy", "container_fanout")),
        "heading_grouping": Atomic(_heading_grouping_utility(output), aw("hierarchy", "heading_grouping")),
        "text_chunking": Atomic(_text_chunking_utility(output, source), aw("hierarchy", "text_chunking")),
    }
    economy_atomics = {
        "semantic_non_duplication": Atomic(_semantic_duplicate_utility(output), aw("economy", "semantic_non_duplication")),
        "renderer_aware_wrapper_economy": Atomic(_redundant_wrapper_utility(output), aw("economy", "renderer_aware_wrapper_economy")),
        "json_to_source_size": Atomic(_json_efficiency_utility(spec, audit, source, cfg), aw("economy", "json_to_source_size")),
    }
    accessibility_score, accessibility_values = _accessibility_score(output)
    accessibility_atomics = {
        "applicable_accessibility_contracts": Atomic(accessibility_score, aw("accessibility", "applicable_accessibility_contracts"), str(accessibility_values))
    }

    groups = {
        "integrity": integrity_atomics,
        "fidelity": fidelity_atomics,
        "semantic_mapping": semantic_atomics,
        "hierarchy": hierarchy_atomics,
        "economy": economy_atomics,
        "accessibility": accessibility_atomics,
    }
    dimensions = {name: weighted_applicable_mean(values) for name, values in groups.items()}

    weights = cfg.dimension_weights
    active_dimensions = {
        name: score
        for name, score in dimensions.items()
        if score is not None and weights.get(name, 0.0) > 0
    }
    weight_sum = sum(weights[name] for name in active_dimensions)
    arithmetic = (
        sum(weights[name] * score for name, score in active_dimensions.items()) / weight_sum
        if weight_sum
        else 0.0
    )
    geometric_log = sum(
        (weights[name] / weight_sum) * log(max(cfg.geometric_floor, score))
        for name, score in active_dimensions.items()
    ) if weight_sum else log(cfg.geometric_floor)
    geometric = exp(geometric_log)
    base_q = cfg.arithmetic_share * arithmetic + (1.0 - cfg.arithmetic_share) * geometric
    integrity_cap, integrity_caps = _critical_cap(
        parse_stage=parse_stage,
        schema_score=schema_score,
        audit=audit,
        type_score=type_score,
        render_ok=resolved_render_ok,
        config=cfg,
    )
    semantic_cap, semantic_caps = _semantic_role_cap(source, role_values, cfg)
    active_caps = integrity_caps + semantic_caps
    cap = min(integrity_cap, semantic_cap)
    q = min(base_q, cap)
    reward = 2.0 * q - 1.0

    atomics_serialized = {
        group: {name: atomic.value for name, atomic in values.items()}
        for group, values in groups.items()
    }
    return RewardBreakdown(
        reward=reward,
        quality_0_100=100.0 * q,
        quality_0_1=q,
        cap_0_1=cap,
        parse_stage=parse_stage,
        dimensions=dimensions,
        atomics=atomics_serialized,
        evidence={
            "source": {
                "headings": len(source.headings),
                "tables": len(source.tables),
                "explicit_actions": len(source.explicit_actions),
                "actionable_urls": len(source.actionable_urls),
                "media": len(source.media),
                "roles": source.required_roles,
                "expected_role_counts": source.expected_role_counts,
            },
            "output": {
                "reachable_elements": len(output.reachable_ids),
                "unreachable_elements": len(audit.unreachable_ids),
                "missing_references": sorted(audit.missing_references),
                "cycles": sorted(audit.cycle_edges),
                "max_root_distance_depth": audit.max_depth,
                "type_counts": dict(output.type_counts),
                "valid_type_counts": dict(output.valid_type_counts),
                "tables": len(output.tables),
                "actions": len(output.actions),
                "media": len(output.media),
            },
            "precision_recall": {
                "content_precision": content_p,
                "content_recall": content_r,
                "value_precision": value_p,
                "value_recall": value_r,
            },
            "role_values": role_values,
            "semantic_role_cap": semantic_cap,
            "integrity_cap": integrity_cap,
            "schema": schema_evidence,
            "type_contract_by_type": type_evidence,
            "render_ok": resolved_render_ok,
            "aggregation": {
                "arithmetic": arithmetic,
                "geometric": geometric,
                "arithmetic_share": cfg.arithmetic_share,
                "base_utility": base_q,
                "final_utility": q,
            },
            "atomic_applicability": {
                group: {name: atomic.applicable for name, atomic in values.items()}
                for group, values in groups.items()
            },
        },
        active_caps=active_caps,
    )


def genui_grpo_reward(
    completions: Sequence[Any],
    response_text: Sequence[str] | str,
    intent_bucket: Sequence[str] | str | None = None,
    assets: Sequence[Any] | Any = None,
    expected_ui_contract: Sequence[Mapping[str, Any] | None] | Mapping[str, Any] | None = None,
    *,
    config: RewardConfig | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL-compatible batch reward function.

    Required dataset column: ``response_text``.
    Recommended columns: ``intent_bucket`` and ``assets``.

    TRL forwards dataset columns to custom reward functions.  Keep
    ``remove_unused_columns=False`` in GRPOConfig.
    """
    n = len(completions)

    def broadcast(value: Any, default: Any = None) -> list[Any]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
            values = list(value)
            if len(values) == n:
                return values
        return [default if value is None else value for _ in range(n)]

    responses = broadcast(response_text, "")
    intents = broadcast(intent_bucket, None)
    asset_rows = broadcast(assets, None)
    contract_rows = broadcast(expected_ui_contract, None)
    results = [
        score_genui_completion(
            c,
            r,
            intent=i,
            assets=a,
            expected_ui_contract=contract,
            config=config,
        )
        for c, r, i, a, contract in zip(
            completions, responses, intents, asset_rows, contract_rows
        )
    ]

    log_extra = kwargs.get("log_extra")
    if callable(log_extra):
        log_extra("genui_quality_0_100", [round(x.quality_0_100, 3) for x in results])
        for dimension in DEFAULT_DIMENSION_WEIGHTS:
            log_extra(
                f"genui_{dimension}",
                [
                    None if x.dimensions.get(dimension) is None
                    else round(float(x.dimensions[dimension]), 4)
                    for x in results
                ],
            )
    log_metric = kwargs.get("log_metric")
    if callable(log_metric) and results:
        log_metric("genui/quality_mean", sum(x.quality_0_100 for x in results) / len(results))
        log_metric("genui/parse_failure_rate", sum(x.parse_stage == "json_parse_error" for x in results) / len(results))
        log_metric("genui/capped_rate", sum(x.cap_0_1 < 1.0 for x in results) / len(results))

    return [x.reward for x in results]


def make_genui_grpo_reward(config: RewardConfig) -> Callable[..., list[float]]:
    """Bind one validated configuration for GRPOTrainer."""
    def reward_function(
        completions: Sequence[Any],
        response_text: Sequence[str] | str,
        intent_bucket: Sequence[str] | str | None = None,
        assets: Sequence[Any] | Any = None,
        expected_ui_contract: Sequence[Mapping[str, Any] | None] | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        return genui_grpo_reward(
            completions,
            response_text,
            intent_bucket=intent_bucket,
            assets=assets,
            expected_ui_contract=expected_ui_contract,
            config=config,
            **kwargs,
        )

    reward_function.__name__ = f"genui_grpo_reward_v{REWARD_VERSION.replace('.', '_')}"
    return reward_function


__all__ = [
    "REWARD_VERSION",
    "RewardBreakdown",
    "RewardConfig",
    "SourceContract",
    "audit_graph",
    "genui_grpo_reward",
    "load_reward_config",
    "make_genui_grpo_reward",
    "parse_source_contract",
    "score_genui_completion",
    "source_contract_from_mapping",
]
