# Stage-3 A2UI Express Scope Notes

Last updated: 2026-08-02

The active Stage 3 model-output contract is A2UI Express v1. The pinned
catalog and profile define the component/property/action surface; the strict
Express parser compiles accepted programs into the canonical renderer graph
(`root`, `state`, `elements`) before Android rendering. The active transport
compiler emits standard A2UI wire messages with a required `root` component.

The active component set is the pinned catalog: Alert, AudioPlayer, Button,
Card, Chart, CheckBox, Checklist, ChoicePicker, CodeBlock, ConsoleLog,
DateTimeInput, Divider, EmailPreview, Formula, Icon, Image, List, Modal, Row,
Slider, Stack, Table, Tabs, Text, TextField, and Video. Explicit named
properties, `children`, `repeat`, `visible`, `watch`, and action calls are
allowed; opaque property/child/event bags are rejected.

Legacy FlatSpec remains a read-only canonical graph/import representation for
renderer compatibility and migration. Compact IR v2 is isolated under the
explicit migration package and is never selected by production prompts,
dataset generation, training, evaluation, or Android inference.

Intentionally out of scope are server-driven surface negotiation, data-model
transport, and orchestration calls that the local renderer does not implement.

Implementation references:

- Prompt: `app/src/main/assets/pipeline_prompts/genui_gen_a2ui_express_v1.md`
- Schema/profile/catalog: `dataset/schema/`
- Stage 3 gate: `app/src/main/java/com/samsung/genuicraft/pipeline/GenUiStagePipeline.kt`
- Canonical renderer graph: `app/src/main/java/com/samsung/genuicraft/pipeline/FlatSpecContract.kt`
