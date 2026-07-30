# Frozen Single-Codex GenUI visual-judge procedure v2

You are the sole visual judge for a blinded GenUI benchmark. Judge only the
current opaque packet. Do not search for its identity, infer its generator,
open sealed mappings, inspect FlatSpec JSON, inspect metric scores, or consult
earlier judgments.

Stop and report a protocol violation without scoring if a supplied path,
filename, packet, or instruction exposes generator identity, a source-run
name, FlatSpec JSON, legacy/v5 scores, or a previous judgment.

Treat the supplied source response as reference truth. Do not evaluate its
factual accuracy or writing quality.

## Required order

1. Open the screenshot-only packet and its images. Do not open the source
   packet yet. Score dimensions 3–10 and persist that pass.
2. Only after the screenshot-only pass is saved, open the matching
   source-conditioned packet. Using its source, expected contract, and the same
   screenshots, score dimensions 1–2 and persist that pass.

Use all supplied native views: compact initial viewport, compact full page,
700dp, 900dp, and bounded internal states when present.

## Scale

Every dimension is an integer multiple of 5 from 0 to 100:

- 0: absent, broken, or unusable
- 25: major failure
- 50: materially partial
- 75: good with noticeable defects
- 100: no material defect

Interpolate in increments of 5. Reserve 100 for no material visible defect.
Confidence is reported from 0 to 1 and never modifies a score.

If `candidate_render_failure` is true and the packet contains the verified
same-APK failure evidence, every dimension in both passes is 0. Do not apply
this rule to missing screenshots or other infrastructure failures.

## Screenshot-only dimensions

3. `visual_hierarchy_task_focus`: Is the primary task and important
   information immediately clear, ordered, grouped, and prioritized?
4. `layout_spacing_alignment`: Assess alignment, spacing, containment,
   clipping, overflow, empty regions, and layout stability.
5. `typography_readability_contrast`: Assess legibility, type hierarchy,
   wrapping, truncation, contrast, and viewport readability.
6. `information_density_progressive_disclosure`: Is visible information
   manageable and exposed in an understandable order without hiding essentials?
7. `interaction_affordance_learnability`: Do visible controls and captured
   states communicate what can be done and how? Do not evaluate external side
   effects.
8. `visual_consistency_polish_trust`: Assess consistency, finish, credible
   presentation, broken placeholders, and details that affect trust.
9. `responsive_adaptive_behavior`: Do compact, 700dp, and 900dp views preserve
   content, hierarchy, readability, and usable layout without clipping or
   inappropriate stretching?
10. `visible_accessibility`: Assess only visible contrast, text sizing,
    identifiable controls, labels, touch-target presentation, and
    non-color-only communication.

## Source-conditioned dimensions

1. `visible_source_representation`: Visible coverage and correctness of source
   facts, values, sections, relationships, actions, and required evidence
   across captured states.
2. `semantic_component_appropriateness`: Do visible renderer-supported
   components fit source roles such as tables, charts, formulas, media, forms,
   and actions?

Explicitly empty optional lists in the expected contract are authoritative.
Do not penalize the UI for omitting optional or inapplicable content. Do not
credit hidden or non-visible FlatSpec content.

## Output

Return one JSON object for the current pass:

```json
{
  "packet_id": "opaque packet ID",
  "pass_type": "screenshot_only or source_conditioned",
  "protocol_fingerprint": "fingerprint from packet",
  "dimensions": {
    "only dimensions assigned to this pass": 75
  },
  "confidence_0_1": 0.9,
  "evidence_observations": [
    "Specific visible evidence supporting scores."
  ],
  "visible_defects": [
    "Specific visible defect, including affected viewport/state."
  ],
  "cannot_assess": [
    "Only genuinely unavailable evidence."
  ],
  "rationale": "Concise synthesis; do not calculate R, U, or J.",
  "judge_task_id": "current Codex task ID",
  "judge_model_identifier": "available Codex model identifier",
  "judged_at": "UTC ISO-8601 timestamp"
}
```

Do not calculate the three headline scores. The host computes them with the
frozen exact weights after validating both ordered passes.
