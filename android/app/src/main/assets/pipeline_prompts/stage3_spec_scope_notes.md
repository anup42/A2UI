# Stage-3 Spec Scope Notes

Last updated: 2026-03-14

Purpose:
- Document what was removed from the original A2UI v0.9 full spec in the stage-3 prompt, and why.
- Keep a reversible record for future expansion.

Current stage-3 scope:
- Render-only IR generation for Android app rendering.
- Messages: `createSurface`, `updateComponents` only.
- Components (25, authoritative list is `FlatSpecContract.allowedTypes`): Stack (with Row/Column aliases), List, Card,
  Table, Chart, Formula, CodeBlock, ConsoleLog, EmailPreview, Text, Image, Icon, Video, AudioPlayer, Divider, Button,
  Tabs, Modal, TextField, CheckBox, ChoicePicker, Slider, DateTimeInput.
  Table/Chart/Formula/CodeBlock/ConsoleLog/EmailPreview were added after this document was first written; the
  Table card-routing engine is now the largest part of `FlatSpecRenderer.kt`.

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
- Current model is `on.{press,submit,change}` bindings plus `watch` on state paths, executed by
  `FlatActionRuntime` in `FlatSpecRenderer.kt`.
- Why: rendered UI executes user interactions (button/list/card clicks, form input) against local state.

5. Function catalog usage
- Actions implemented by the renderer: `openUrl`, `setState`, `pushState`, `removeState`, `validateForm`.
  This matches `genui_gen.md` and `FlatSpecContract`.
- `showMessage` and `showSurface` are **not** implemented. Earlier revisions of this document listed them as
  restored; they were never wired into `FlatActionRuntime` and unknown action names are no-ops.
- `validateForm` evaluates real per-field rules (`FlatFormValidation.kt`). Rules are derived from the input
  controls themselves, so no extra IR is needed: a `TextField`/`CheckBox`/`ChoicePicker`/`Slider`/`DateTimeInput`
  that declares `required`, `pattern`/`regex`, `minLength`/`maxLength`, `min`/`max`, or `inputType: email|number`
  becomes a rule keyed by the state path it is bound to. The result shape is unchanged:
  `{"valid": Boolean, "errors": { "<statePath>": "<message>" }}`. A control that declares no constraints
  contributes no rule, so a form with no declared validation still reports `valid: true`.
- Note: non-render/runtime orchestration calls remain out of scope.

6. Interactive and media component behaviour
- `ChoicePicker` supports both single- and multi-select. Declare it with `mode` (`single` /
  `mutuallyExclusive` / `multiple`) or `multiple: true|false`; otherwise the selection mode is inferred from the
  *type* of the current value, and a string value stays a string. `maxSelections` is honoured in multi-select.
- `DateTimeInput` opens a real Material 3 picker. `mode` accepts `date` (default), `time` or `datetime`;
  `enableDate`/`enableTime` are also accepted. Picked values are ISO-8601 (`YYYY-MM-DD`, `HH:MM`, or
  `YYYY-MM-DD HH:MM`) so `validateForm` can compare them against `min`/`max`. Manual text entry stays enabled
  unless the element sets `allowManualEntry: false`.
- `Icon` accepts a bare icon *name* as well as a URL — snake_case, kebab-case, camelCase and Bootstrap icon names
  all resolve to a bundled Material vector, which tints to the theme and works offline. Unknown names fall back to
  the URL path. Values containing a path separator or a media suffix are always treated as URLs/assets.
- `Video` and `AudioPlayer` render a poster tile with a play affordance when the element declares
  `poster`/`posterUrl`/`thumbnail`, plus a `duration` label (seconds or a pre-formatted string). Playback still
  opens externally; inline playback is intentionally out of scope because a video frame is not reproducible and
  would make render-capture screenshots non-deterministic.
- An element type the renderer does not support now draws a visible "Unsupported element" placeholder instead of
  rendering nothing, and an unsupported action name is logged rather than silently ignored.

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
