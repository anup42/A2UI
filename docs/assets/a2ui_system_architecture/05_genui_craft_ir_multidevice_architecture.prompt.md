# GenUI Craft IR architecture diagram

- Final asset: `05_genui_craft_ir_multidevice_architecture.png`
- Created: 2026-09-16
- Method: built-in image generation, followed by a targeted connector correction.
- Reference: user-supplied architecture photograph.
- Scope: conceptual runtime architecture; device screens are illustrations, not captured runtime screenshots or a claim that every device implementation is complete.
- Visual QA: checked title, A2UI Express and IR labels, all three agent-to-engine arrows, renderer-to-device arrows, and consistent flight examples. Existing assets were preserved.

## Initial generation prompt

```text
Use case: infographic-diagram.
Asset type: a polished, presentation-ready, single architecture slide, 16:9 landscape, high resolution.

Input image 1 is ONLY a reference for the logical architecture and left-to-right composition. Redraw the diagram from scratch. Do not reproduce the photo, perspective, moire, diagonal watermark, old icons or old styling.

Header (exact text): "End to End Architecture"
Main flow: User Utterance -> multiple AI Agents -> GenUI Craft -> IR -> Native Renderer -> TV / PC, Mobile / Tablet, Watch.

Design:
White canvas, generous whitespace and margins, crisp modern sans-serif typography, dark navy text, thin clean connectors with arrowheads, tasteful teal and blue accents, a restrained green accent for the central GenUI Craft block. Consistent vector-like outline icons; flat design with very subtle card shadows. It must look like a carefully aligned professional architecture slide, not a screenshot or a hand-drawn diagram.

Layout:
- At the left: a simple person/speech icon in a soft blue circle, with the label "User Utterance" and small sublabel "Voice or text".
- Next: three equal compact cards vertically stacked, each with a matching AI/agent icon. Labels "AI Agent 1", "AI Agent 2", "AI Agent 3". Branch the user arrow to these three cards. Converge their output arrows cleanly into the central block, without crossings. Label the outgoing connection area "Agent responses" in small readable type.
- Center: a prominent green-teal rounded rectangle labelled "GenUI Craft". Inside, a smaller pill reads "A2UI Express". Under it, the concise line "Response-to-UI generation". This is the focal point.
- After the central block: a clearly separate document-shaped card labelled "IR" in large bold type, with the sublabel "Intermediate Representation". Include a minimal tree-of-components icon, not HTML/CSS/JavaScript symbols.
- Next: a clean light-blue rounded block labelled "Native Renderer", with small sublabel "Device-adaptive UI".
- Far right: a well-spaced vertical set of three device experiences, each connected by its own arrow from the renderer. Row labels "TV / PC", "Mobile / Tablet", "Watch". Show a modern monitor, a phone with a tablet silhouette, and a smartwatch. Devices should be drawn as clear front-facing vector mockups, with illustrative native flight cards adapted to each screen size. Use the SAME flight example consistently: "DEL → BLR", "10:30", "On time". Monitor can show a wide itinerary card, phone a compact stacked card, watch only route and flight status. Keep text on screens minimal and readable. Add small microphone and touch/cursor icons beside the appropriate devices, to indicate voice and touch interaction as in the reference, but do not clutter the slide or create a long feedback-loop arrow.

Make the diagram use most of the slide area with balanced spacing. Make the three AI agent cards and the three device rows align vertically. Keep the central flow on the middle horizontal axis. All arrowheads must point forward in the intended flow and stop at their target boundaries. No overlapping text or crossed lines. Carefully render all labels exactly as specified.
Critical replacement: the output stage is "IR"; do not write HTML, CSS or JS anywhere.
Do not add training stages, benchmark names, TensorBoard, metrics tables, watermarks, logos, slide numbers or extra explanatory paragraphs. The result should be immediately usable in a technical presentation.
```

## Final refinement prompt

```text
Edit this architecture diagram with one focused correction. Keep all existing text, typography, colors, composition, device screens, header and boxes unchanged.
The middle card "AI Agent 2" currently has no output arrow. Add a straight horizontal blue arrow from the right boundary of "AI Agent 2" to the left boundary of "GenUI Craft", on the centerline of those boxes. Move the small two-line "Agent responses" label out of that arrow's way, into the free white space immediately above the GenUI Craft box (centered above that box, outside it). Preserve the existing diagonal arrows from AI Agent 1 and AI Agent 3. Ensure all three AI agent cards visibly connect to GenUI Craft with forward-pointing arrowheads and no text overlap. Do not change anything else. Output the same 16:9 high-resolution slide.
```
