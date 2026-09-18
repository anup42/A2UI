# GenUI Craft LM architecture — incoming-arrow caption

- Final image: `05_genui_craft_ir_multidevice_architecture_v3.png`
- Input: `05_genui_craft_ir_multidevice_architecture_v2.png` (preserved).
- Method: built-in image generation/editing.
- Change: moved `Agent responses` beside the incoming arrows, immediately above the middle horizontal arrow.
- Visual QA: caption does not overlap the arrows; all three agent connections remain visible. Main labels and device examples are retained.

## Final edit prompt

```text
Use case: precise-object-edit.
Input image 1 is the edit target.
Make ONLY a position change to the small two-line caption "Agent responses".
Remove this caption from its current position above the green GenUI Craft LM box and restore the clean white background there.
Move the same caption to the gap between the AI agent cards and the green box, immediately ABOVE the horizontal arrow from "AI Agent 2" into "GenUI Craft LM". It should clearly label that incoming connection, not float over the green box. Keep it in two lines, "Agent" then "responses", small dark navy sans-serif. In the 1672 x 941 reference, the desired caption center is approximately x=464, y=462, with its lower edge above the horizontal arrow at y=493. Fit it neatly in the white space between the middle agent card and the upper diagonal arrow; reduce the caption font slightly if needed so no letters touch an arrow or card.
IMPORTANT: Preserve all three incoming arrows into GenUI Craft LM, including the horizontal middle arrow, fully visible and connected. Do not remove an arrow to make space for the caption. No overlaps.
Keep every other part unchanged: title, card positions, all text including the exact labels "GenUI Craft LM", "A2UI Express", "Response to IR generation", IR card, Native Renderer, devices and flight screens, colors, styling, layout and dimensions.
The only visible change should be relocation of "Agent responses" from above the green box to beside the incoming arrow.
```
