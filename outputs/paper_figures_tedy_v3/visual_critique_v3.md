# TEDY Paper Figure Critique and v3 Refinement

## Critique pass 1: v2 figure gaps
- The previous figures were structurally correct but too text-heavy for a paper skim.
- Architecture flow lacked enough concrete visual anchors for response, IR, renderer, and mobile UI.
- Table rendering was described as text, not shown as portrait-vs-landscape behavior.
- Validation/metrics were not visual enough to explain why the spec is useful for training and evaluation.

## Refinement pass 1: v3 visual changes
- Added stage icons, semantic chips, mobile mockups, table miniatures, and color-coded lanes.
- Reduced paragraph text inside boxes and moved details into concise labels.
- Added explicit portrait phone, landscape/tablet, and domain-native template panels.
- Added a validation/repair loop and separated diagnostics, metrics, and training target.

## Critique pass 2: paper-readiness checklist
- Readability: high-contrast text on white background, no dense screenshots.
- Reproducibility: deterministic vector generation with PNG, PDF, and SVG outputs.
- Paper fit: 16:9 wide figures suitable for single-column large or two-column page-width placement.
- Visual specificity: includes root/state/elements, compact Table state, native renderer, and mobile adaptation.

## Remaining tradeoff
- These are deterministic diagram figures, not photorealistic AI illustrations. That is intentional for paper clarity and editable vector output.
