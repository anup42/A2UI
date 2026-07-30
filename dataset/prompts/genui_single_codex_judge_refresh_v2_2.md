# Single-Codex GenUI Judge Rubric Refresh v2.2

This is a reliability-only refinement of v2.1. It preserves the same ten
dimensions, weights, ordered two-pass boundary, source-as-reference-truth
policy, 5-point score increment, and blinding rules. It replaces free-form
interpolation with the deterministic defect-severity ledger below.

## Mandatory scoring ledger

For each dimension, start at 100. Identify independent defects that belong to
that dimension, assign each exactly one severity, add the deductions, clamp to
0, and round downward to a multiple of 5:

| Severity | Deduction | Operational meaning |
|---|---:|---|
| none | 0 | No material visible defect for this dimension. |
| trivial | 5 | Cosmetic single-location issue with no task or reading cost. |
| minor | 10 | Localized issue that is easy to work around. |
| noticeable | 15 | Clearly visible issue; the UI remains fully usable. |
| material | 25 | Important content or behavior is impaired, but the UI remains substantially usable. |
| major | 50 | A central task, large required section, or primary viewport is materially partial or unusable. |
| critical | 75 | Most evidence for this dimension fails. |
| broken | 100 | No usable evidence for this dimension. |

Do not count the same manifestation once per screenshot. Treat it as one
defect and choose severity from its breadth:

- one secondary state only: usually minor or noticeable;
- one primary viewport, or repeated across several states: usually material;
- most primary viewports: usually major;
- all supplied evidence: critical or broken.

Independent defects in one dimension may accumulate. Do not invent defects to
fill a band. Include the chosen severity tag in `visible_defects`, for example
`[material] compact table loses two required columns`.

The common anchors remain exact consequences of the ledger:

- 100: no defect;
- 75: one material defect;
- 50: one major defect or two independent material defects;
- 25: one critical defect or three independent material defects;
- 0: broken evidence.

When evidence genuinely supports two adjacent severities, choose the more
severe one and lower confidence. Confidence never changes the score.

## Evidence-set and viewport rules

Judge all authorized screenshots and bounded internal states as one evidence
set.

- Do not count a repeated manifestation separately in each viewport.
- For source representation, use the union of content visible anywhere in the
  authorized evidence set.
- Content visible at 700dp or 900dp is represented even when compact width
  clips it. Put compact loss in layout, responsive behavior, and typography
  when it becomes unreadable.
- A blank full-height capture is a real stability defect. It is not total UI
  absence when another checkout-matched capture renders correctly.
- A state label does not prove that its chart, media, formula, tab, modal, or
  action content is visibly present.

## Dimension boundaries and severity anchors

### 1. Visible source representation

Score only required visible source facts, values, sections, relationships,
actions, and evidence. Explicitly optional or empty contract requirements are
not defects.

- minor/noticeable: one low-importance detail omitted or slightly distorted;
- material: one important required value, relation, action, or bounded section
  absent from every authorized state;
- major: a central required section or several important requirements absent;
- critical/broken: most required content absent, unrelated, or unusable.

Do not penalize compact clipping here when the content is visible elsewhere.

### 2. Semantic component appropriateness

Judge visible role choice, not styling quality. Required tables, charts,
formulas, media, forms, actions, tabs, and modal content must visibly perform
the role. Generic styling does not make a correct role inappropriate.

- noticeable: role is recognizable but weakly expressed;
- material: one important role is substituted, empty, or unusable;
- major: a central role or several required roles are missing/substituted;
- critical/broken: visible component choices do not represent the task.

### 3. Visual hierarchy and task focus

Assess priority, grouping, reading order, and immediate task clarity.

- noticeable: one competing emphasis or weak grouping;
- material: primary task is not immediately clear or a major section is
  mis-prioritized;
- major: hierarchy prevents understanding of the main task.

Do not move spacing, truncation, or responsive defects into this dimension
unless they also change priority or grouping.

### 4. Layout, spacing, and alignment

Assess geometry inside supplied views: containment, alignment, spacing,
overflow, clipping, and page stability.

- noticeable: localized uneven spacing/alignment;
- material: important content clips/overflows in one primary viewport or
  persistent geometry defects affect several regions;
- major: a primary viewport or central section is structurally unusable.

### 5. Typography, readability, and contrast

Assess visible legibility, wrapping, truncation, type hierarchy, text size, and
contrast.

- noticeable: isolated awkward wrapping, weak contrast, or inconsistent type;
- material: important text is unreadable/truncated in one primary viewport or
  the problem repeats broadly;
- major: central information is unreadable across most evidence.

Use the same severity for the same repeated truncation; do not count it once
per screenshot.

### 6. Information density and progressive disclosure

Assess the amount, chunking, and staging of information without rewarding raw
content quantity.

- noticeable: slightly sparse, crowded, or over-exposed;
- material: a major section is empty, content is persistently overwhelming,
  or essential information is hidden behind unclear staging;
- major: density/staging makes the central task materially partial.

### 7. Interaction affordance and learnability

Assess only visible controls and captured states: recognizability, labels,
state communication, and discoverability. Do not infer external side effects.

- noticeable: one ambiguous secondary control;
- material: an important visible action or state transition is unclear;
- major: the central interaction cannot be understood or used.

If no interaction is required by the visible task, absence of controls is not
a defect.

### 8. Visual consistency, polish, and trust

Assess consistent styling, finish, placeholders, credible presentation, and
details that affect trust.

- noticeable: one obvious unfinished or inconsistent detail;
- material: repeated placeholders/broken media or broad inconsistency;
- major: presentation looks substantially broken or untrustworthy.

Do not penalize generic styling here unless it creates a concrete finish or
trust defect.

### 9. Responsive/adaptive behavior

Compare compact, full-height, 700dp, 900dp, and supplied internal states.

- noticeable: localized reflow, stretch, whitespace, or clipping defect while
  every primary viewport remains usable;
- material: one primary viewport loses important content/usability;
- major: most primary viewports materially lose content or usability;
- broken: all supplied viewport evidence is unusable.

An isolated blank full-height capture is normally material, not broken.

### 10. Visible accessibility

Assess only visible evidence: text sizing/contrast, identifiable controls,
visible labels, touch-target presentation, and non-color-only communication.
Do not infer screen-reader semantics, focus order, or hidden labels.

- noticeable: one secondary visible accessibility weakness;
- material: important text/control is visibly hard to perceive or operate in a
  primary viewport;
- major: central content or controls are broadly inaccessible;
- broken: the visible task cannot be perceived or operated.

Typography and accessibility may both be affected by one defect only when it
independently harms both constructs. Record it once in each affected
dimension, with the same severity unless the evidence clearly differs.

## Reliability checklist before READY

For every packet and dimension:

1. Confirm all authorized views/states were considered.
2. List only independent, dimension-owned defects.
3. Attach one severity to each listed defect.
4. Recompute the score from the ledger; do not choose a score first.
5. Confirm the result is finite, bounded, and a multiple of 5.
6. Keep generator identity, FlatSpec, metrics, previous judgments, repeat
   identity, aggregates, and comparisons blinded.

