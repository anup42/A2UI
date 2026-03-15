# Stage-3 Spec Scope Notes

Last updated: 2026-03-14

Purpose:
- Document what was removed from the original A2UI v0.9 full spec in the stage-3 prompt, and why.
- Keep a reversible record for future expansion.

Current stage-3 scope:
- Render-only IR generation for Android app rendering.
- Messages: `createSurface`, `updateComponents` only.
- Components: Text, Image, Icon, Video, AudioPlayer, Row, Column, List, Card, Tabs, Modal, Divider, Button, TextField, CheckBox, ChoicePicker, Slider, DateTimeInput.

Removed from original full spec and rationale:
1. Client capability exchange
- Removed: `supportedCatalogIds`, `inlineCatalogs` metadata usage.
- Why removed: not needed in this local pipeline; no runtime client/server capability negotiation.

2. Data-model transport/runtime controls
- Removed: `sendDataModel`, `updateDataModel`, `deleteSurface`.
- Why removed: app pipeline renders generated IR directly; no incremental server-driven data model sync.

3. Dynamic binding and template child lists
- Removed: `{ "path": ... }` bindings and `{ "componentId": ..., "path": ... }` child templates.
- Why removed: renderer pipeline is static/literal for determinism, lower complexity, and easier debugging.

4. Client runtime/event/validation flow wiring
- Restored subset: `action.functionCall`, `action.event` (click flows), `showSurface`, `showMessage`, and input validation wiring for interactive controls.
- Why restored: rendered UI now executes user interactions (button/list/card clicks, surface switching, user feedback, input validation hints).

5. Function catalog usage
- Supported interaction calls: `openUrl`, `showMessage`, `showSurface`.
- Validation calls allowed in prompt: `required`, `regex`, `length`, `numeric`, `email`.
- Note: non-render/runtime orchestration calls still remain out of scope.

Tradeoff summary:
- Pros: lower prompt size, faster stage-3 latency, lower output variability, easier renderer correctness.
- Cons: still not a full A2UI runtime implementation; server-driven data-model sync/orchestration remains intentionally out of scope.

When to restore removed pieces:
- Introduce real client/server orchestration with data-model updates.
- Need interactive workflows with validation/event callbacks.
- Need runtime catalog negotiation across multiple clients.

Implementation references:
- Prompt scope: `app/src/main/assets/pipeline_prompts/genui_gen.md`
- Stage-3 output schema gate: `app/src/main/java/com/samsung/genuicraft/pipeline/GenUiStagePipeline.kt`
- Native parser path binding suppression: `app/src/main/java/com/samsung/genuicraft/renderer/native/NativePayloadParser.kt`
- HTML parser path binding suppression: `app/src/main/java/com/samsung/genuicraft/renderer/GenUiHtmlRenderer.kt`
