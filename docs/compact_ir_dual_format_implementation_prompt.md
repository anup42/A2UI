# Implementation Prompt: Dual Compact GenUI IR + Pinned A2UI Express

## Role

Act as a principal software architect, ML data-pipeline engineer, Android/Kotlin engineer, schema designer, and independent code reviewer. Modify this repository exactly as provided. Preserve UI richness and all existing renderer semantics while reducing model-output tokens.

## Source and version constraints

- Treat the uploaded repository as the source of truth for existing behavior.
- Pin upstream A2UI inputs to repository `a2ui-project/a2ui` commit `2276f8cc702eaeac25ffb05be85797b2a1205c74` (observed 2026-08-01).
- Target the A2UI v1.0 release-candidate wire contract, but isolate it behind explicit protocol/catalog/grammar version constants.
- Align the Express syntax with the upstream grammar and proposal at the pinned commit: sentinel tags, assignments, positional and keyword arguments, nested/inline components, arrays/maps/literals, data paths, skipped optional arguments, events, and function calls.
- Do not silently follow upstream `main` in production or tests.

## Primary objective

Implement two model-facing formats that describe the same rich UI graph with fewer tokens:

1. `compact_ir_v2`: compact JSON with short structural keys, omitted empty/default fields, deterministic optional ID compaction, and no loss of existing FlatSpec semantics.
2. `a2ui_express_v1`: pinned A2UI Express plus a GenUICraft custom catalog/extensions sufficient to represent every existing component, property, binding, repeat, visibility expression, action, event, and watch behavior.

Both formats must normalize into one canonical typed FlatSpec graph and must be mutually and bidirectionally convertible with existing FlatSpec. Conversion must preserve renderer-visible semantics. Meaningful components must never be removed merely to improve token count.

## Mandatory correctness work

1. Audit and fix renderer-reference traversal before any pruning or ID rewriting.
2. Use one authoritative reference inventory for validation, cycle detection, reachability, pruning, metrics, ID rewriting, conversion, and Android ingestion.
3. Cover ordinary children, legacy child aliases, template/itemTemplate aliases, repeat template aliases, Tabs content aliases with renderer precedence, Modal trigger/content, and any additional reference-bearing fields actually consumed by Android.
4. Add parity tests proving Python and Kotlin reference inventories agree.
5. Never delete or rename a referenced node without updating every supported reference location.

## Compact IR v2 requirements

- Define a new JSON Schema; keep the old schema but do not use it as the Stage 3 default or generation fallback.
- Include an explicit format/version marker.
- Required compact keys: root and elements. State is optional when empty.
- Element representation must support type, props, children, repeat, visible, event handlers, and watch handlers, while omitting empty structures.
- Preserve unknown-but-valid component props so catalog evolution is lossless.
- Support deterministic ID shortening as a codec option and update every renderer reference safely.
- Restore canonical FlatSpec defaults only in the compiler/normalizer, not in model output.

## A2UI Express requirements

- Implement parser, serializer/decompiler, validator, and compiler integration in Python and Android/Kotlin.
- Provide a pinned GenUICraft A2UI catalog containing all current renderer component types and aliases:
  `Stack`, `List`, `Card`, `Text`, `Formula`, `CodeBlock`, `ConsoleLog`, `EmailPreview`, `Table`, `Chart`, `Image`, `Icon`, `Video`, `AudioPlayer`, `Divider`, `Button`, `Tabs`, `Modal`, `TextField`, `CheckBox`, `ChoicePicker`, `Slider`, `DateTimeInput`, `Alert`, and `Checklist`, plus supported Row/Column aliases.
- Preserve direct Button labels and multi-child Cards to avoid expansion that increases tokens.
- Map current actions (`openUrl`, `setState`, `pushState`, `removeState`, `validateForm`) and arbitrary renderer events without semantic loss.
- Preserve `repeat.statePath`, `repeat.key`, `visible`, `on`, and `watch` through explicit catalog properties or reserved Express metadata parameters.
- Support an escape hatch for catalog-valid future props without requiring a grammar rewrite.
- Compile Express to canonical FlatSpec for the current native renderer and provide A2UI v1 custom-catalog wire output for interoperability.

## Dataset-pipeline integration

- Make dual-format pilot generation the default: generate and validate both `compact_ir_v2` and `a2ui_express_v1` independently for each Stage 3 source response.
- Do not generate legacy FlatSpec as a fallback. A failed format remains failed and is visible in validation metadata.
- Require both outputs by default for the pilot; make policy configurable.
- Store both raw outputs, normalized canonical graphs, validation/repair diagnostics, exact token/character counts, and semantic-equivalence results.
- Select a preferred candidate only after independent validity and quality evaluation, using quality-first then token-count tie-breaking. Do not select a simpler UI merely because it is shorter.
- Keep the old schema and old input reader for migration/testing only.
- Provide distinct model prompts for both formats and keep richness guidance. Numeric element floors may remain only as task-complexity safeguards, not token-optimization objectives.
- Repair each format with its own contract; never repair Express by asking for legacy JSON.

## Metrics and training integration

- Normalize all supported formats before artifact/render metrics so the same UI receives the same quality score.
- Add format-aware generation validity, syntax, repair, token efficiency, semantic round-trip, and selected-format metrics.
- Include format/schema/catalog/compiler hashes in metric identity and stale-score detection.
- Update training ingestion/configs so either compact format can be selected as the completion target; dual records must expose both targets.
- Do not reward token reduction unless fidelity/richness/interaction validity gates pass.
- Add a benchmark script using the deployed tokenizer when available and a deterministic fallback tokenizer estimate otherwise.

## Android integration

- Detect and ingest legacy FlatSpec, Compact IR v2 JSON, A2UI Express text/string wrappers, and A2UI v1 custom-catalog wire envelopes.
- Normalize all of them to the existing renderer model through one codec layer.
- Make the app's Stage 3 assistant prompt default to the dual compact pilot contract rather than the legacy FlatSpec schema.
- Expose format, version, normalization warnings, and conversion diagnostics.
- Preserve all existing native UI behavior; no WebView substitution.

## Conversion tooling

Provide a CLI script that reads JSON, JSONL, or Express and converts among:

- `flat_spec_v1`
- `compact_ir_v2`
- `a2ui_express_v1`
- `a2ui_v1_wire`

Support validation, semantic round-trip checking, optional deterministic ID shortening, pretty/minified output, JSONL records containing `genui_json`, and a benchmark/report mode.

## Testing and review acceptance criteria

- Existing relevant Python tests pass.
- New codec/schema/reference/round-trip/metric/pipeline tests pass.
- Android unit tests for codec, ingestion, reference parity, all component types, Tabs, Modal, repeat/template, actions, and wire envelopes pass where the build environment permits.
- Every existing allowed component appears in round-trip coverage.
- Golden-corpus semantic equivalence is measured; any non-equivalent sample is reported and fixed rather than hidden by fallback.
- Token benchmark compares legacy FlatSpec, Compact IR v2, A2UI Express, and A2UI v1 wire on the repository corpus.
- Produce a full implementation plan/checklist, review report, machine-readable test report, token report, changed-file manifest, and source diff.
- Package the complete updated repository and review artifacts in one ZIP.

## Non-goals and prohibitions

- Do not reduce UI quality or delete meaningful components to reduce tokens.
- Do not use the old FlatSpec schema as the default model-output contract or as a silent fallback.
- Do not silently drop unsupported props/actions/components; fail validation with actionable diagnostics.
- Do not claim tests passed when they were not run or were blocked.
- Do not implement a separate renderer for each format; use one canonical graph and one native renderer.
