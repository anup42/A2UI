"""Anchored criterion-referenced scoring for the GenUI visual judge.

The LLM never chooses a dimension score or headline score directly.  It rates
five frozen observable criteria per dimension on anchored levels 0..4 and
records the worst independent defect severity.  Host code derives dimension
scores, R, U, J_raw, and a separate policy-capped score deterministically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


JUDGE_NAME = "GenUI Anchored Criterion Judge"
JUDGE_SHORT_NAME = "GACJ"
JUDGE_SCHEMA_VERSION = "genui_anchored_criterion_judgment.v3"
JUDGE_PACKET_SCHEMA_VERSION = "genui_anchored_criterion_packet.v3"
JUDGE_PROTOCOL_VERSION = "genui_anchored_criterion_rubric.v3.0.0"
JUDGE_AUTHORITY = "provisional_single_model_criterion_reference"

DIMENSIONS: tuple[str, ...] = (
    "visible_source_representation",
    "semantic_component_appropriateness",
    "visual_hierarchy_task_focus",
    "layout_spacing_alignment",
    "typography_readability_contrast",
    "information_density_progressive_disclosure",
    "interaction_affordance_learnability",
    "visual_consistency_polish_trust",
    "responsive_adaptive_behavior",
    "visible_accessibility",
)
SOURCE_DIMENSIONS: tuple[str, ...] = DIMENSIONS[:2]
VISUAL_DIMENSIONS: tuple[str, ...] = DIMENSIONS[2:]
PASS_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "screenshot_only": VISUAL_DIMENSIONS,
    "source_conditioned": SOURCE_DIMENSIONS,
}

COMPOSITE_WEIGHTS: dict[str, float] = {
    "visible_source_representation": 0.18,
    "semantic_component_appropriateness": 0.14,
    "visual_hierarchy_task_focus": 0.14,
    "layout_spacing_alignment": 0.12,
    "typography_readability_contrast": 0.10,
    "information_density_progressive_disclosure": 0.10,
    "interaction_affordance_learnability": 0.08,
    "visual_consistency_polish_trust": 0.04,
    "responsive_adaptive_behavior": 0.05,
    "visible_accessibility": 0.05,
}

ANCHOR_LEVEL_VALUES: dict[int, float] = {
    0: 0.0,
    1: 25.0,
    2: 50.0,
    3: 75.0,
    4: 100.0,
}
ANCHOR_LEVEL_MEANINGS: dict[int, str] = {
    0: "fails or provides no usable evidence for the criterion",
    1: "severely deficient; only a small part of the criterion is satisfied",
    2: "usable but not production-ready; material improvement is required",
    3: "production-ready with only localized or minor defects",
    4: "reference-quality for the criterion; no material defect is visible",
}

# A major defect cannot be averaged away by otherwise strong criteria.  Minor
# and moderate evidence remain mostly continuous; fatal policies are separate.
SEVERITY_ORDER: tuple[str, ...] = (
    "none",
    "minor",
    "moderate",
    "major",
    "critical",
    "broken",
)
SEVERITY_CEILINGS: dict[str, float] = {
    "none": 100.0,
    "minor": 95.0,
    "moderate": 80.0,
    "major": 55.0,
    "critical": 25.0,
    "broken": 0.0,
}

# Policy caps are deliberately separate from criterion-referenced raw quality.
# They must not be used for correlation/calibration unless explicitly named.
FATAL_POLICY_CAPS: dict[str, float] = {
    "verified_candidate_render_failure": 0.0,
    "blank_or_failed_render": 10.0,
    "critical_overlap_or_clipping": 35.0,
    "core_required_semantic_role_absent": 55.0,
}


@dataclass(frozen=True)
class CriterionDefinition:
    name: str
    question: str
    pass_anchor: str
    partial_anchor: str
    failure_anchor: str
    weight: float = 0.2


@dataclass(frozen=True)
class DimensionDefinition:
    name: str
    construct: str
    criteria: tuple[CriterionDefinition, ...]


# Exactly five equal-budget criteria per dimension.  This makes the un-capped
# dimension base an exact multiple of five while retaining criterion evidence.
def _criterion(
    name: str,
    question: str,
    pass_anchor: str,
    partial_anchor: str,
    failure_anchor: str,
) -> CriterionDefinition:
    return CriterionDefinition(
        name=name,
        question=question,
        pass_anchor=pass_anchor,
        partial_anchor=partial_anchor,
        failure_anchor=failure_anchor,
    )


DIMENSION_DEFINITIONS: dict[str, DimensionDefinition] = {
    "visible_source_representation": DimensionDefinition(
        name="visible_source_representation",
        construct=(
            "Faithful visible representation of the supplied source response; "
            "the source is reference content, not a factuality target."
        ),
        criteria=(
            _criterion(
                "required_content_coverage",
                "Are the source's required ideas and sections visibly represented?",
                "All required content is visible across the authorized evidence set.",
                "The main content is usable, but one material section or several details are missing.",
                "Most required content is absent or unusable.",
            ),
            _criterion(
                "exact_values_and_labels",
                "Are exact values, names, dates, units, labels, and row/field associations preserved?",
                "Visible values and labels are correct and associated with the right entities.",
                "A material value, label, or association is missing or ambiguous.",
                "Values are broadly wrong, swapped, or unusable.",
            ),
            _criterion(
                "relationships_sections_and_priority",
                "Are relationships, ordering, section structure, and source priority preserved?",
                "Relationships and source structure remain immediately understandable.",
                "The task remains understandable, but one major relationship or priority is weakened.",
                "Relationships or section structure are substantially misrepresented.",
            ),
            _criterion(
                "required_actions_media_and_evidence",
                "Are source-required actions, links, media, charts, formulas, code, and other evidence visibly present?",
                "Every applicable required evidence role is visibly fulfilled.",
                "One important required role is absent, incomplete, or visibly unusable.",
                "Most required evidence roles are missing or unrelated.",
            ),
            _criterion(
                "unsupported_additions_and_duplication_control",
                "Does the UI avoid unsupported visible additions and unnecessary repetition?",
                "No material unsupported or duplicated information is visible.",
                "Some duplication or unsupported detail adds noticeable cost without changing the task.",
                "The screen is dominated by unsupported, misleading, or repeated content.",
            ),
        ),
    ),
    "semantic_component_appropriateness": DimensionDefinition(
        name="semantic_component_appropriateness",
        construct=(
            "Appropriateness of renderer-supported UI roles for the source's "
            "information structure, task, scale, and interactions."
        ),
        criteria=(
            _criterion(
                "primary_role_fit",
                "Does the primary visible component family fit the source task?",
                "The primary component role is the most suitable supported representation.",
                "The representation is usable but materially less suitable than an available native role.",
                "The central role is missing, substituted, or misleading.",
            ),
            _criterion(
                "structured_data_fit",
                "Are comparisons, tables, sequences, and quantitative structures represented appropriately?",
                "Structured information uses an effective renderer-supported structure, or none is needed.",
                "The structure is recognizable but awkward, incomplete, or difficult to compare.",
                "Structured information is flattened or mapped to the wrong role.",
            ),
            _criterion(
                "interaction_and_control_fit",
                "Are visible actions, forms, tabs, filters, and controls semantically appropriate?",
                "Controls match the task and expose the right interaction roles, or none are required.",
                "An important control is generic, misplaced, or only partially suitable.",
                "The interaction model is missing or substantially wrong.",
            ),
            _criterion(
                "specialized_content_fit",
                "Are media, charts, formulas, code, email, console, and other specialized roles mapped effectively?",
                "Every applicable specialized role is renderer-supported and effective, or none is required.",
                "One specialized role is weakly represented or substituted.",
                "A central specialized role is missing, blank, or inappropriate.",
            ),
            _criterion(
                "scale_and_renderer_effectiveness",
                "Is the chosen representation suitable for item count, viewport, and actual renderer behavior?",
                "The representation scales appropriately and is visibly effective in the supplied renderer.",
                "The representation works but is inefficient or weak at the supplied scale.",
                "The chosen structure does not scale or does not visibly render the intended role.",
            ),
        ),
    ),
    "visual_hierarchy_task_focus": DimensionDefinition(
        name="visual_hierarchy_task_focus",
        construct="Immediate clarity of purpose, priority, grouping, and scan order.",
        criteria=(
            _criterion("screen_purpose_clarity", "Is the screen's purpose immediately clear?", "Purpose is unambiguous on first inspection.", "Purpose is understandable after some scanning.", "The main task is unclear."),
            _criterion("primary_information_emphasis", "Is the most important information visually prioritized?", "Primary information dominates appropriately.", "Some secondary content competes with the primary information.", "Important information is visually buried or misleadingly de-emphasized."),
            _criterion("section_grouping", "Are related elements grouped into coherent sections?", "Grouping is clear and stable.", "One major section has weak grouping.", "The page is broadly flat or fragmented."),
            _criterion("reading_and_scan_order", "Is the reading/scan sequence coherent?", "The eye follows a natural task-oriented sequence.", "The sequence is usable but contains a material detour.", "Reading order is confusing or contradictory."),
            _criterion("action_and_endpoint_priority", "Are task endpoints or primary actions visually prioritized appropriately?", "Primary endpoints are clear, or the task needs no explicit action.", "Important endpoints are visible but weakly prioritized.", "The primary task endpoint is hidden or visually misleading."),
        ),
    ),
    "layout_spacing_alignment": DimensionDefinition(
        name="layout_spacing_alignment",
        construct="Spatial organization, containment, alignment, spacing, and stability.",
        criteria=(
            _criterion("containment_clipping_and_overlap", "Is content contained without clipping, overlap, or overflow?", "No material clipping, overlap, or containment defect is visible.", "One primary view has a material but workable containment defect.", "Core content is clipped, overlapping, or structurally unusable."),
            _criterion("alignment_consistency", "Are edges, columns, labels, and repeated elements aligned consistently?", "Alignment is coherent throughout.", "Several visible alignments are inconsistent but usable.", "Alignment defects broadly impair reading or comparison."),
            _criterion("spacing_consistency", "Are margins, padding, and gaps consistent and purposeful?", "Spacing is deliberate and coherent.", "Noticeable spacing issues occur in one or more regions.", "Crowding or excessive gaps broadly damage the layout."),
            _criterion("repeated_structure_stability", "Do repeated cards, rows, and sections preserve stable structure?", "Repeated structures are stable and comparable.", "Optional content creates noticeable but manageable instability.", "Repeated items are inconsistent enough to impair comparison or scanning."),
            _criterion("viewport_space_use", "Does the UI use available viewport space effectively?", "The screen is neither wasteful nor overcrowded at supplied sizes.", "One viewport is noticeably sparse, stretched, or crowded.", "Space use makes a primary viewport ineffective."),
        ),
    ),
    "typography_readability_contrast": DimensionDefinition(
        name="typography_readability_contrast",
        construct="Visible legibility, type hierarchy, wrapping, truncation, and contrast.",
        criteria=(
            _criterion("body_text_legibility", "Is body text comfortably readable?", "Body text is comfortably legible in all supplied primary views.", "Some body text requires effort but remains usable.", "Important body text is unreadable."),
            _criterion("heading_type_hierarchy", "Does typography distinguish titles, headings, labels, and body content?", "Type hierarchy is clear and consistent.", "One level is weak or inconsistent.", "Typography does not communicate structure."),
            _criterion("wrapping_and_truncation", "Are wrapping and truncation handled without losing meaning?", "No meaningful text is lost or awkwardly broken.", "Some important text wraps or truncates awkwardly in a primary view.", "Core meaning is lost through truncation or wrapping."),
            _criterion("visible_contrast", "Is visible foreground/background contrast adequate?", "All important content has strong visible contrast.", "One important region has weak but usable contrast.", "Core content is difficult to perceive because of contrast."),
            _criterion("label_value_readability", "Can users associate labels, values, units, and rows easily?", "Labels and values are unambiguous and easy to scan.", "A material label/value association is visually weak.", "Labels and values are broadly ambiguous or unreadable."),
        ),
    ),
    "information_density_progressive_disclosure": DimensionDefinition(
        name="information_density_progressive_disclosure",
        construct="Appropriate initial information load, prioritization, and disclosure.",
        criteria=(
            _criterion("initial_information_load", "Is the initial view appropriately dense for the task?", "The initial view exposes the right amount of information.", "It is noticeably dense or sparse but remains usable.", "The initial view is overwhelming or nearly empty."),
            _criterion("primary_secondary_prioritization", "Are primary and secondary details exposed in the right order?", "Primary content is immediate and secondary details are appropriately deferred.", "Some secondary detail competes with or obscures primary content.", "Important content is buried or secondary content dominates."),
            _criterion("progressive_disclosure_quality", "Are long details and optional content disclosed understandably?", "Disclosure is clear and preserves essential information.", "Disclosure is usable but over-expanded, under-exposed, or unclear in one area.", "Essential information is hidden or the page is persistently over-expanded."),
            _criterion("collection_scalability", "Does the presentation remain manageable for the supplied collection size?", "The collection representation is scalable and easy to scan.", "The collection is usable but inefficient at its size.", "Collection scale makes the UI materially unusable."),
            _criterion("redundancy_and_empty_space", "Does the UI avoid duplicated content, empty sections, and unnecessary expansion?", "No material redundancy or empty structure is visible.", "Some duplication or empty space adds noticeable cost.", "Redundancy or empty structure dominates the page."),
        ),
    ),
    "interaction_affordance_learnability": DimensionDefinition(
        name="interaction_affordance_learnability",
        construct="Visible recognizability, labeling, priority, and state communication of controls.",
        criteria=(
            _criterion("control_recognizability", "Do interactive elements look interactive?", "Controls are immediately recognizable, or no controls are needed.", "One important control is visually ambiguous.", "The central interaction cannot be identified."),
            _criterion("action_label_clarity", "Do labels communicate what actions will do?", "Action labels are clear and specific, or no actions are needed.", "An important label is generic or ambiguous.", "Action labels are broadly misleading or absent."),
            _criterion("primary_secondary_action_distinction", "Are primary and secondary actions visually distinguished appropriately?", "Action priority is clear, or no action hierarchy is needed.", "Priority is visible but weak or inconsistent.", "Users cannot identify the primary action."),
            _criterion("state_navigation_learnability", "Do tabs, filters, disclosure, and captured states explain navigation and state?", "State and navigation are easy to understand, or none are needed.", "One material transition or state is unclear.", "The interaction state model is confusing or unusable."),
            _criterion("misleading_or_missing_affordances", "Does the UI avoid controls that look inactive, misleading, or absent when visibly needed?", "No material misleading or missing affordance is visible.", "One important affordance is weak or misleading.", "The page broadly miscommunicates what can be done."),
        ),
    ),
    "visual_consistency_polish_trust": DimensionDefinition(
        name="visual_consistency_polish_trust",
        construct="Coherence, finish, absence of broken artifacts, and visible trustworthiness.",
        criteria=(
            _criterion("component_style_consistency", "Are related components styled consistently?", "Component styling is coherent throughout.", "A noticeable local inconsistency is present.", "The UI mixes incompatible or unfinished component styles broadly."),
            _criterion("color_and_icon_consistency", "Are colors and icons used consistently and meaningfully?", "Color and icon use is coherent and purposeful.", "A material icon or color inconsistency is visible.", "Color/icon use is broadly confusing or untrustworthy."),
            _criterion("shape_spacing_finish_consistency", "Are shape, elevation, dividers, and finish details coherent?", "Finish details are consistent and deliberate.", "Several finish details are noticeably inconsistent.", "The screen appears substantially unfinished."),
            _criterion("broken_placeholder_absence", "Is the UI free of broken media, placeholders, raw syntax, and error artifacts?", "No broken or placeholder artifact is visible.", "One important artifact is broken or unfinished.", "Broken artifacts dominate or undermine the task."),
            _criterion("professional_trust_presentation", "Does the UI appear credible and professionally assembled?", "The presentation is trustworthy and production-ready.", "The presentation is usable but has material trust/polish weaknesses.", "The interface appears unreliable or unfinished."),
        ),
    ),
    "responsive_adaptive_behavior": DimensionDefinition(
        name="responsive_adaptive_behavior",
        construct="Preservation of content, usability, priority, and suitable density across supplied viewports.",
        criteria=(
            _criterion("compact_view_usability", "Is the compact view usable without material loss?", "Compact rendering preserves the task and important content.", "Compact rendering has one material but workable weakness.", "Compact rendering is substantially unusable."),
            _criterion("medium_view_usability", "Is the medium-width view usable and appropriately arranged?", "Medium rendering is stable and effective.", "Medium rendering has a noticeable material weakness.", "Medium rendering is substantially unusable."),
            _criterion("expanded_view_usability", "Is the expanded view usable without inappropriate stretching or wasted structure?", "Expanded rendering uses space effectively and preserves the task.", "Expanded rendering is usable but inefficient or stretched.", "Expanded rendering is substantially ineffective."),
            _criterion("cross_viewport_priority_preservation", "Are content, hierarchy, and actions preserved consistently across widths?", "Priority and content remain stable across all supplied views.", "One important priority or action shifts weakly across views.", "Important content or actions disappear or become misleading."),
            _criterion("reflow_and_density_adaptation", "Does layout/density adapt appropriately rather than merely scale?", "Reflow and density are appropriate at each supplied width.", "Adaptation is usable but noticeably weak in one view.", "The layout fails to adapt and becomes unusable."),
        ),
    ),
    "visible_accessibility": DimensionDefinition(
        name="visible_accessibility",
        construct="Visible-only accessibility evidence; no inference about hidden semantics or screen-reader behavior.",
        criteria=(
            _criterion("contrast_accessibility", "Is visible contrast sufficient for important content and controls?", "Contrast is strong for all important content.", "One important region has weak but usable contrast.", "Core content or controls are difficult to perceive."),
            _criterion("text_size_and_legibility", "Is visible text sized and spaced for comfortable reading?", "Text is comfortably readable in all primary views.", "Some important text is visibly small or cramped.", "Core text is visibly inaccessible."),
            _criterion("labels_and_icon_clarity", "Are visible labels and icon meanings identifiable?", "Important controls and icons have clear visible meaning.", "One important control/icon is ambiguous.", "Core controls rely on unexplained or inaccessible visual symbols."),
            _criterion("touch_target_separation", "Do visible controls appear sufficiently separated and targetable?", "Controls appear adequately sized and separated.", "Some important controls are crowded or appear small.", "Core controls are visibly difficult to target."),
            _criterion("non_color_only_and_visual_cues", "Does the UI avoid relying only on color and provide visible cues?", "Meaning is communicated with text/shape/position as well as color.", "One important state relies too heavily on color.", "Critical meaning is conveyed only through inaccessible visual cues."),
        ),
    ),
}


@dataclass(frozen=True)
class DimensionScore:
    base_0_100: float
    severity_ceiling_0_100: float
    score_0_100: float
    evidence_complete: bool


@dataclass(frozen=True)
class CriterionJudgedScores:
    source_representation_0_100: float
    rendered_ux_0_100: float
    raw_composite_0_100: float
    policy_capped_composite_0_100: float
    policy_cap_0_100: float


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def judge_instructions_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "prompts"
        / "genui_anchored_criterion_judge_v3.md"
    )


def criterion_names(dimension: str) -> tuple[str, ...]:
    try:
        return tuple(item.name for item in DIMENSION_DEFINITIONS[dimension].criteria)
    except KeyError as exc:
        raise ValueError(f"unknown dimension: {dimension}") from exc


def rubric_for_pass(pass_type: str) -> dict[str, Any]:
    if pass_type not in PASS_DIMENSIONS:
        raise ValueError(f"unsupported pass type: {pass_type}")
    return {
        "judge_name": JUDGE_NAME,
        "protocol_version": JUDGE_PROTOCOL_VERSION,
        "pass_type": pass_type,
        "dimensions": [
            {
                "name": name,
                "construct": DIMENSION_DEFINITIONS[name].construct,
                "criteria": [asdict(item) for item in DIMENSION_DEFINITIONS[name].criteria],
            }
            for name in PASS_DIMENSIONS[pass_type]
        ],
        "anchor_levels": {
            str(level): {
                "value_0_100": ANCHOR_LEVEL_VALUES[level],
                "meaning": ANCHOR_LEVEL_MEANINGS[level],
            }
            for level in sorted(ANCHOR_LEVEL_VALUES)
        },
        "severity_ceilings": dict(SEVERITY_CEILINGS),
        "score_increment": 5,
        "scoring_policy": (
            "The judge returns criterion levels and defects only. Host code computes "
            "dimension and headline scores. Do not choose a dimension or headline score."
        ),
        "reference_truth_policy": (
            "Treat supplied source response as reference content; do not judge its factual or writing quality."
        ),
    }


def protocol_mapping() -> dict[str, Any]:
    return {
        "judge_name": JUDGE_NAME,
        "judge_short_name": JUDGE_SHORT_NAME,
        "schema_version": JUDGE_SCHEMA_VERSION,
        "packet_schema_version": JUDGE_PACKET_SCHEMA_VERSION,
        "protocol_version": JUDGE_PROTOCOL_VERSION,
        "authority": JUDGE_AUTHORITY,
        "dimensions": list(DIMENSIONS),
        "pass_dimensions": {key: list(value) for key, value in PASS_DIMENSIONS.items()},
        "weights": dict(COMPOSITE_WEIGHTS),
        "anchor_levels": {
            str(key): {"value": value, "meaning": ANCHOR_LEVEL_MEANINGS[key]}
            for key, value in ANCHOR_LEVEL_VALUES.items()
        },
        "severity_ceilings": dict(SEVERITY_CEILINGS),
        "fatal_policy_caps": dict(FATAL_POLICY_CAPS),
        "dimension_definitions": {
            name: {
                "construct": definition.construct,
                "criteria": [asdict(item) for item in definition.criteria],
            }
            for name, definition in DIMENSION_DEFINITIONS.items()
        },
        "formulas": {
            "criterion_value": "25 * anchor_level",
            "dimension_base": "sum(criterion_weight * criterion_value)",
            "dimension_score": "floor_5(min(dimension_base, worst_severity_ceiling))",
            "source_representation": "(0.18*S1 + 0.14*S2) / 0.32",
            "rendered_ux": (
                "(0.14*S3 + 0.12*S4 + 0.10*S5 + 0.10*S6 + "
                "0.08*S7 + 0.04*S8 + 0.05*S9 + 0.05*S10) / 0.68"
            ),
            "raw_composite": "sum(weight_i * S_i)",
            "policy_capped_composite": "min(raw_composite, active_policy_caps)",
        },
        "ordered_passes": ["screenshot_only", "source_conditioned"],
        "blinding": {
            "metric_scores": False,
            "generator_identity": False,
            "flat_spec_json": False,
            "previous_judgments": False,
            "repeat_identity": False,
        },
        "model_identity_policy": (
            "Stop and create a new protocol version if the available judge model identifier changes."
        ),
        "evidence_policy": (
            "A full absolute score requires every frozen criterion to be scored. "
            "not_observable prevents publication of a full score; it is never silently renormalized."
        ),
    }


def protocol_fingerprint() -> str:
    return hashlib.sha256(_canonical_json(protocol_mapping()).encode("utf-8")).hexdigest()


def _floor_to_five(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    bounded = min(100.0, max(0.0, float(value)))
    return float(5 * math.floor((bounded + 1e-9) / 5.0))


def _severity_rank(value: str) -> int:
    try:
        return SEVERITY_ORDER.index(value)
    except ValueError as exc:
        raise ValueError(f"unsupported defect severity: {value!r}") from exc


def compute_dimension_score(
    dimension: str,
    criterion_results: Mapping[str, Mapping[str, Any]],
    *,
    worst_defect_severity: str,
) -> DimensionScore:
    expected = criterion_names(dimension)
    missing = [name for name in expected if name not in criterion_results]
    extra = sorted(set(criterion_results) - set(expected))
    if missing:
        raise ValueError(f"{dimension}: missing criteria: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{dimension}: unknown criteria: {', '.join(extra)}")
    _severity_rank(worst_defect_severity)
    values: list[float] = []
    for name in expected:
        result = criterion_results[name]
        status = str(result.get("status") or "").strip()
        if status == "not_observable":
            return DimensionScore(
                base_0_100=math.nan,
                severity_ceiling_0_100=SEVERITY_CEILINGS[worst_defect_severity],
                score_0_100=math.nan,
                evidence_complete=False,
            )
        if status != "scored":
            raise ValueError(f"{dimension}.{name}: status must be scored or not_observable")
        level = result.get("anchor_level")
        if isinstance(level, bool) or not isinstance(level, int) or level not in ANCHOR_LEVEL_VALUES:
            raise ValueError(f"{dimension}.{name}: anchor_level must be an integer from 0 to 4")
        values.append(ANCHOR_LEVEL_VALUES[level])
    base = sum(values) / len(values)
    ceiling = SEVERITY_CEILINGS[worst_defect_severity]
    score = _floor_to_five(min(base, ceiling))
    if score >= 100.0 - 1e-9 and any(value < 100.0 for value in values):
        raise AssertionError("dimension score 100 requires every criterion at anchor level 4")
    return DimensionScore(
        base_0_100=base,
        severity_ceiling_0_100=ceiling,
        score_0_100=score,
        evidence_complete=True,
    )


def compute_criterion_judged_scores(
    dimension_scores: Mapping[str, Any],
    *,
    fatal_findings: Sequence[str] = (),
) -> CriterionJudgedScores:
    missing = [name for name in DIMENSIONS if name not in dimension_scores]
    extra = sorted(set(dimension_scores) - set(DIMENSIONS))
    if missing:
        raise ValueError(f"missing dimension scores: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unknown dimension scores: {', '.join(extra)}")
    values: dict[str, float] = {}
    for name in DIMENSIONS:
        raw = dimension_scores[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name} must be numeric")
        score = float(raw)
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            raise ValueError(f"{name} must be finite and in [0,100]")
        if abs(score / 5.0 - round(score / 5.0)) > 1e-9:
            raise ValueError(f"{name} must be a multiple of five")
        values[name] = score
    representation = (
        COMPOSITE_WEIGHTS[DIMENSIONS[0]] * values[DIMENSIONS[0]]
        + COMPOSITE_WEIGHTS[DIMENSIONS[1]] * values[DIMENSIONS[1]]
    ) / 0.32
    ux = sum(COMPOSITE_WEIGHTS[name] * values[name] for name in VISUAL_DIMENSIONS) / 0.68
    raw_composite = sum(COMPOSITE_WEIGHTS[name] * values[name] for name in DIMENSIONS)
    unknown_findings = sorted(set(fatal_findings) - set(FATAL_POLICY_CAPS))
    if unknown_findings:
        raise ValueError(f"unknown fatal findings: {', '.join(unknown_findings)}")
    cap = min((FATAL_POLICY_CAPS[item] for item in fatal_findings), default=100.0)
    policy = min(raw_composite, cap)
    output = CriterionJudgedScores(
        source_representation_0_100=representation,
        rendered_ux_0_100=ux,
        raw_composite_0_100=raw_composite,
        policy_capped_composite_0_100=policy,
        policy_cap_0_100=cap,
    )
    for name, score in asdict(output).items():
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            raise AssertionError(f"invalid computed {name}: {score}")
    return output


def _string_list(value: Any, *, name: str, maximum: int = 32) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return [str(item).strip() for item in value if str(item).strip()][:maximum]


def validate_criterion_judgment_pass(
    value: Mapping[str, Any],
    *,
    expected_packet_id: str | None = None,
    expected_pass_type: str | None = None,
) -> dict[str, Any]:
    allowed_top_level = {
        "schema_version",
        "protocol_version",
        "packet_id",
        "pass_type",
        "protocol_fingerprint",
        "dimensions",
        "fatal_findings",
        "cannot_assess",
        "overall_rationale",
        "judge_task_id",
        "judge_model_identifier",
        "judged_at",
    }
    extra_top_level = sorted(set(value) - allowed_top_level)
    if extra_top_level:
        raise ValueError(
            f"unexpected judgment fields: {', '.join(extra_top_level)}"
        )
    if str(value.get("schema_version") or "") != JUDGE_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    if str(value.get("protocol_version") or "") != JUDGE_PROTOCOL_VERSION:
        raise ValueError("protocol_version mismatch")
    packet_id = str(value.get("packet_id") or "").strip()
    if not packet_id:
        raise ValueError("packet_id is required")
    if expected_packet_id is not None and packet_id != expected_packet_id:
        raise ValueError("packet_id mismatch")
    pass_type = str(value.get("pass_type") or "").strip()
    if pass_type not in PASS_DIMENSIONS:
        raise ValueError(f"unsupported pass_type: {pass_type!r}")
    if expected_pass_type is not None and pass_type != expected_pass_type:
        raise ValueError("pass_type mismatch")
    if str(value.get("protocol_fingerprint") or "") != protocol_fingerprint():
        raise ValueError("protocol_fingerprint mismatch")
    raw_dimensions = value.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        raise ValueError("dimensions must be an object")
    required_dimensions = PASS_DIMENSIONS[pass_type]
    missing = [name for name in required_dimensions if name not in raw_dimensions]
    extra = sorted(set(raw_dimensions) - set(required_dimensions))
    if missing:
        raise ValueError(f"missing pass dimensions: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unexpected pass dimensions: {', '.join(extra)}")
    normalized_dimensions: dict[str, Any] = {}
    for dimension in required_dimensions:
        raw_dimension = raw_dimensions[dimension]
        if not isinstance(raw_dimension, Mapping):
            raise ValueError(f"{dimension} must be an object")
        allowed_dimension_fields = {
            "criteria",
            "defects",
            "worst_defect_severity",
            "confidence_0_1",
            "rationale",
        }
        extra_dimension_fields = sorted(
            set(raw_dimension) - allowed_dimension_fields
        )
        if extra_dimension_fields:
            raise ValueError(
                f"{dimension}: unexpected fields: "
                + ", ".join(extra_dimension_fields)
            )
        raw_criteria = raw_dimension.get("criteria")
        if not isinstance(raw_criteria, Mapping):
            raise ValueError(f"{dimension}.criteria must be an object")
        expected_criteria = criterion_names(dimension)
        criterion_missing = [name for name in expected_criteria if name not in raw_criteria]
        criterion_extra = sorted(set(raw_criteria) - set(expected_criteria))
        if criterion_missing:
            raise ValueError(f"{dimension}: missing criteria: {', '.join(criterion_missing)}")
        if criterion_extra:
            raise ValueError(f"{dimension}: unknown criteria: {', '.join(criterion_extra)}")
        normalized_criteria: dict[str, Any] = {}
        for criterion_name in expected_criteria:
            raw_criterion = raw_criteria[criterion_name]
            if not isinstance(raw_criterion, Mapping):
                raise ValueError(f"{dimension}.{criterion_name} must be an object")
            allowed_criterion_fields = {
                "status", "anchor_level", "evidence", "defects"
            }
            extra_criterion_fields = sorted(
                set(raw_criterion) - allowed_criterion_fields
            )
            if extra_criterion_fields:
                raise ValueError(
                    f"{dimension}.{criterion_name}: unexpected fields: "
                    + ", ".join(extra_criterion_fields)
                )
            status = str(raw_criterion.get("status") or "").strip()
            if status not in {"scored", "not_observable"}:
                raise ValueError(f"{dimension}.{criterion_name}: invalid status")
            level = raw_criterion.get("anchor_level")
            if status == "scored":
                if isinstance(level, bool) or not isinstance(level, int) or level not in ANCHOR_LEVEL_VALUES:
                    raise ValueError(f"{dimension}.{criterion_name}: invalid anchor_level")
                normalized_level: int | None = level
            else:
                if level is not None:
                    raise ValueError(f"{dimension}.{criterion_name}: not_observable must not have anchor_level")
                normalized_level = None
            normalized_criteria[criterion_name] = {
                "status": status,
                "anchor_level": normalized_level,
                "evidence": _string_list(raw_criterion.get("evidence", []), name=f"{dimension}.{criterion_name}.evidence", maximum=8),
                "defects": _string_list(raw_criterion.get("defects", []), name=f"{dimension}.{criterion_name}.defects", maximum=8),
            }
        defects_raw = raw_dimension.get("defects", [])
        if not isinstance(defects_raw, Sequence) or isinstance(defects_raw, (str, bytes)):
            raise ValueError(f"{dimension}.defects must be an array")
        normalized_defects: list[dict[str, str]] = []
        observed_severities: list[str] = []
        for index, defect in enumerate(defects_raw):
            if not isinstance(defect, Mapping):
                raise ValueError(f"{dimension}.defects[{index}] must be an object")
            extra_defect_fields = sorted(
                set(defect) - {"severity", "description"}
            )
            if extra_defect_fields:
                raise ValueError(
                    f"{dimension}.defects[{index}]: unexpected fields: "
                    + ", ".join(extra_defect_fields)
                )
            severity = str(defect.get("severity") or "").strip()
            _severity_rank(severity)
            description = str(defect.get("description") or "").strip()
            if not description:
                raise ValueError(f"{dimension}.defects[{index}].description is required")
            observed_severities.append(severity)
            normalized_defects.append({"severity": severity, "description": description})
        declared_worst = str(raw_dimension.get("worst_defect_severity") or "").strip()
        _severity_rank(declared_worst)
        calculated_worst = max(observed_severities, key=_severity_rank) if observed_severities else "none"
        if declared_worst != calculated_worst:
            raise ValueError(
                f"{dimension}.worst_defect_severity={declared_worst!r} does not match listed defects ({calculated_worst!r})"
            )
        confidence = raw_dimension.get("confidence_0_1")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"{dimension}.confidence_0_1 must be finite and in [0,1]")
        normalized_dimensions[dimension] = {
            "criteria": normalized_criteria,
            "defects": normalized_defects,
            "worst_defect_severity": declared_worst,
            "confidence_0_1": float(confidence),
            "rationale": str(raw_dimension.get("rationale") or "").strip(),
        }
        if not normalized_dimensions[dimension]["rationale"]:
            raise ValueError(f"{dimension}.rationale is required")
    fatal_raw = value.get("fatal_findings", [])
    if not isinstance(fatal_raw, Sequence) or isinstance(fatal_raw, (str, bytes)):
        raise ValueError("fatal_findings must be an array")
    fatal_findings = [str(item).strip() for item in fatal_raw if str(item).strip()]
    unknown_fatal = sorted(set(fatal_findings) - set(FATAL_POLICY_CAPS))
    if unknown_fatal:
        raise ValueError(f"unknown fatal findings: {', '.join(unknown_fatal)}")
    judged_at = str(value.get("judged_at") or "").strip() or datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(judged_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("judged_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("judged_at must include timezone")
    judge_task_id = str(value.get("judge_task_id") or "").strip()
    model_identifier = str(value.get("judge_model_identifier") or "").strip()
    if not judge_task_id or not model_identifier:
        raise ValueError("judge_task_id and judge_model_identifier are required")
    overall_rationale = str(value.get("overall_rationale") or "").strip()
    if not overall_rationale:
        raise ValueError("overall_rationale is required")
    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "protocol_version": JUDGE_PROTOCOL_VERSION,
        "packet_id": packet_id,
        "pass_type": pass_type,
        "protocol_fingerprint": protocol_fingerprint(),
        "dimensions": normalized_dimensions,
        "fatal_findings": fatal_findings,
        "cannot_assess": _string_list(value.get("cannot_assess", []), name="cannot_assess"),
        "overall_rationale": overall_rationale,
        "judge_task_id": judge_task_id,
        "judge_model_identifier": model_identifier,
        "judged_at": judged_at,
    }


__all__ = [
    "ANCHOR_LEVEL_MEANINGS",
    "ANCHOR_LEVEL_VALUES",
    "COMPOSITE_WEIGHTS",
    "CriterionJudgedScores",
    "DIMENSIONS",
    "DIMENSION_DEFINITIONS",
    "DimensionScore",
    "FATAL_POLICY_CAPS",
    "JUDGE_AUTHORITY",
    "JUDGE_NAME",
    "JUDGE_PACKET_SCHEMA_VERSION",
    "JUDGE_PROTOCOL_VERSION",
    "JUDGE_SCHEMA_VERSION",
    "PASS_DIMENSIONS",
    "SEVERITY_CEILINGS",
    "SOURCE_DIMENSIONS",
    "VISUAL_DIMENSIONS",
    "compute_criterion_judged_scores",
    "compute_dimension_score",
    "criterion_names",
    "judge_instructions_path",
    "protocol_fingerprint",
    "protocol_mapping",
    "rubric_for_pass",
    "validate_criterion_judgment_pass",
]
