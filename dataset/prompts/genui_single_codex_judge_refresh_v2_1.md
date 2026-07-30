# Single-Codex GenUI Judge Rubric Refresh v2.1

This addendum refines the frozen v2 anchors after a blind-repeat reliability
audit. It does not change the ten dimensions, their weights, the two-pass
ordering, the 5-point score increment, or the source-as-reference-truth policy.

Apply these rules consistently to every packet:

1. Judge the evidence set, not a single screenshot in isolation.
   - Inspect every authorized viewport and bounded internal state.
   - A defect visible in one viewport is real, but it is not evidence that
     content is absent from every viewport.
   - Do not count the same clipping or placeholder defect independently in
     several dimensions unless it materially affects each construct.

2. Keep source coverage separate from responsive behavior.
   - For `visible_source_representation`, use the union of content visibly
     available across all authorized states and viewports.
   - Content that is complete at 700dp or 900dp but clipped at compact width is
     represented, while the clipping belongs primarily in
     `responsive_adaptive_behavior`, `layout_spacing_alignment`, and, when
     text becomes unreadable, `typography_readability_contrast`.
   - A required section, value, role, or relationship absent from every
     authorized state is a source-representation omission.

3. Score responsive behavior with explicit anchors.
   - 100: no material loss or instability across compact, medium, expanded,
     full-height, and captured internal states.
   - 75: usable throughout, with a noticeable but localized reflow, clipping,
     stretching, or whitespace defect.
   - 50: a primary viewport loses material content or usability, while other
     supplied viewports remain usable.
   - 25: most primary viewports are materially broken or unusable.
   - 0: all supplied viewport evidence is broken or unusable.
   - A blank full-height capture is a serious stability defect, but it is not
     total UI absence when other checkout-matched captures render correctly.

4. Keep semantic role choice separate from visual polish.
   - `semantic_component_appropriateness` evaluates whether the visible
     component role fits the source: for example, table versus prose, chart
     versus unlabeled decoration, form control versus static text, or media
     region versus placeholder.
   - Generic styling alone does not make a semantically correct component
     inappropriate. Missing or substituted required roles do.

5. Use the common anchors literally for source-conditioned dimensions.
   - 100: no material required source content or role is missing or distorted.
   - 75: good representation with a small number of noticeable omissions or
     role defects.
   - 50: materially partial; at least one major required section, relationship,
     or role is missing or unusable.
   - 25: major failure; most required content or roles are absent.
   - 0: the visible UI is absent, unusable, or unrelated to the supplied
     source.

6. Calibrate blank, placeholder, and unavailable-media evidence consistently.
   - A broken placeholder lowers polish and, when it replaces required media,
     semantic appropriateness and source representation.
   - Do not infer a successful image, chart, formula, tab state, or modal from
     labels alone when the required visible evidence is absent.
   - Do not treat an intentionally empty region as broken unless the source,
     contract, or surrounding UI shows that content should be there.

7. Preserve dimension boundaries.
   - Hierarchy: priority, grouping, and task focus.
   - Layout: geometry, containment, spacing, alignment, and page stability.
   - Typography: legibility, wrapping, truncation, and contrast.
   - Density: amount and staging of information.
   - Affordance: whether visible controls communicate action and state.
   - Polish: visual consistency, finish, placeholders, and trust.
   - Accessibility: only visible accessibility evidence.

8. When evidence genuinely supports two adjacent 5-point values, choose the
   lower value and reduce confidence. Confidence never changes the score.

The judge must remain blinded to generator identity, FlatSpec JSON, metric
scores, previous judgments, repeat identity, and aggregate analysis.
