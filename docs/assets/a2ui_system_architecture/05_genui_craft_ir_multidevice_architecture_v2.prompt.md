# GenUI Craft LM architecture diagram — label update

- Final image: `05_genui_craft_ir_multidevice_architecture_v2.png`
- Date: 2026-09-16
- Method: built-in image generation/editing.
- Input: `05_genui_craft_ir_multidevice_architecture.png` (preserved).
- Requested labels: `GenUI Craft LM` and `Response to IR generation`.
- Visual check: both labels fit within the green block; A2UI Express, the IR stage, all three agent connections, renderer connections and device examples remain present.
- Scope: conceptual architecture; device screens remain illustrative.

## Label-edit prompt

```text
Use case: text-localization.
Input image 1: edit target, a finished architecture slide.
Make exactly two text changes inside the central green block:
1. Replace "GenUI Craft" with the exact text "GenUI Craft LM". Keep this as the main white bold title, on a single line if possible, adjusting its font size slightly so the entire title fits comfortably inside the existing green block without clipping or overlap.
2. Replace "Response-to-UI generation" with the exact text "Response to IR generation". Preserve its smaller white subtitle styling and centered placement.
Keep the "A2UI Express" pill unchanged.
Preserve everything else exactly: the "End to End Architecture" header, all boxes and positions, all three AI-agent input and output arrows, the IR document card and labels, Native Renderer, all device examples and flight details, interaction icons, colors, whitespace, aspect ratio and resolution.
Do not add any text, arrows or objects. In particular do not change IR into UI. Only replace the two requested phrases.
```

## Final typography refinement

```text
Image 1 is the edit target. Only edit the subtitle inside the green panel underneath the A2UI Express pill.
Completely erase the existing small subtitle and typeset this new exact four-word phrase from scratch:
Response to IR generation

Use four distinct words separated by clearly visible spaces: word 1 Response, word 2 to, word 3 IR, word 4 generation. There is NO punctuation in the phrase. In particular the space after the word "to" must remain empty green background, not a hyphen. Set it in white clean sans-serif type, centered on one line, with a little extra word spacing so the words are clearly separated.
Keep all other pixels of the architecture slide as close to the reference as possible: main title "GenUI Craft LM", A2UI Express, all arrows, labels, icons, shapes, device screens, background, dimensions. Do not redesign or add anything.
```
