# Dual Compact IR Implementation Plan and Review Checklist

Status values: `DONE`, `PARTIAL`, `BLOCKED`, `NOT_APPLICABLE`.

## 0. Baseline and pinning

- DONE — Verified the supplied compact and full handoff archives and imported the compact-IR scope without adding the archives to Git.
- DONE — Pinned upstream A2UI inputs to commit `2276f8cc702eaeac25ffb05be85797b2a1205c74` and Express grammar blob `4f2492ae4600598d8b10e68fcd9f4292529dd653`.
- DONE — Recorded protocol, grammar, catalog, codec, and reference-semantics identities in generated manifests and benchmark reports.
- DONE — Ran task-relevant Python and Android validation; evidence is recorded in `docs/review_artifacts/compact_ir_dual_format_test_report.json`.

## 1. Reference correctness

- DONE — Audited renderer references and introduced matching Python and Kotlin authoritative inventories.
- DONE — Routed validation, cycle detection, reachability/pruning, deterministic ID rewriting, and metrics through the shared semantics.
- DONE — Covered Tabs aliases and precedence, Modal trigger/content, repeat/template references, non-child cycles, and pruning retention.
- DONE — Added a shared Python/Kotlin parity fixture and stable reference-semantics fingerprint.

## 2. Canonical IR and catalogs

- DONE — Added canonical format/version constants and unified codec results.
- DONE — Added a GenUICraft catalog covering the native renderer component and action inventories.
- DONE — Added property ordering, defaults, signatures, aliases, and forward-compatible property preservation.
- DONE — Added catalog completeness and generated-artifact drift checks.

## 3. Compact IR v2

- DONE — Added the `genui_compact_ir_v2.schema.json` contract.
- DONE — Implemented strict Python and Kotlin detection, parsing, validation, serialization, and canonical expansion.
- DONE — Implemented safe deterministic ID shortening using the authoritative reference inventory.
- DONE — Added schema, malformed-field, all-component, reference, and semantic round-trip tests.

## 4. A2UI Express v1 and wire protocol

- DONE — Vendored the pinned upstream Express grammar and recorded its identity.
- DONE — Implemented dependency-free Python and Kotlin lexer/parser/compiler/decompiler support for sentinel tags, assignments, positional and named arguments, skipped arguments, arrays/maps, bindings, inline components, raw/multiline strings, component references, actions, and events.
- DONE — Preserved GenUICraft extensions for props, children, repeats, visibility, events, and watches.
- DONE — Implemented custom-catalog A2UI v1 wire create/update/delete/data-model message handling in Python and Kotlin.
- DONE — Added advanced syntax, message stream, arbitrary-property, action, and semantic round-trip tests.

## 5. Conversion and benchmark CLIs

- DONE — Added conversion across FlatSpec, Compact IR v2, Express, and A2UI v1 wire.
- DONE — Supported JSON, JSONL, wrapped records, files, stdin/stdout, pretty/minified output, and deterministic IDs.
- DONE — Enforced semantic round trips and exercised the conversion CLI in automated tests.
- DONE — Added and executed a separate corpus benchmark CLI with machine-readable JSON and CSV reports.

## 6. Dataset Stage 3

- DONE — Added independent Compact IR and Express prompts and made the two-format pilot the default dataset configuration.
- DONE — Each format has its own generation, parse, repair, regeneration, strict validation, accepted/rejected record, and telemetry path.
- DONE — Wrong-format or malformed outputs are retained as `format_rejected`; the new formats do not silently fall back to generated FlatSpec.
- DONE — Accepted rows retain raw completion, raw parsed payload, normalized native output, canonical graph, source format, codec identity, semantic hash, size, token estimate, and latency.
- DONE — Added deterministic quality-first selection with tokens used only inside the configured quality tolerance.
- DONE — Added fake-adapter integration tests for successful pairs, partial failures, richness retention, and selection policy.

## 7. Metrics and judges

- DONE — Normalized every supported representation through the canonical renderer graph before structural metrics.
- DONE — Added per-format validity, size/token, latency, selection, and codec/catalog/reference identity fields.
- DONE — Added metamorphic tests proving structural metric invariance and raw Express normalization.

## 8. Training

- DONE — Added deterministic format-native completion targets and target-format selection.
- DONE — Updated dataset configs and chat prompts for Compact IR v2, while allowing explicit dual-target output.
- DONE — Rejected format-failed rows and verified each generated target against the canonical semantic hash.
- DONE — Preserved symbolic action URL placeholders during native-target validation.
- DONE — Added training-ingestion tests for historical, native-format, and explicit dual-target rows.

## 9. Android

- DONE — Added unified detection/normalization for Compact IR, Express, A2UI v1 wire, and retained migration FlatSpec.
- DONE — Integrated the codecs at the common ingestion and Stage 3 boundaries so all routes feed the existing renderer graph.
- DONE — Added strict selected-format enforcement for initial generations and repairs, plus format diagnostics.
- DONE — Added `emitEvent` to the renderer capability/schema/runtime contract.
- DONE — Added codec/reference tests and updated four Tabs routing goldens whose previously pruned content is now retained.

## 10. Validation and benchmarking

- DONE — Generated schema/catalog artifacts pass drift verification (`5 files`).
- DONE — Task-relevant Python validation passes (`89 passed`) and the focused codec/CLI group passes (`17 passed`).
- DONE — Android `testDebugUnitTest` passes (`265 tests`, no failures/errors/skips), Kotlin compilation passes, and `assembleDebug` succeeds.
- DONE — The 32-spec corpus passes semantic round trip in all four representations.
- PARTIAL — The available deterministic lexical estimate shows Compact IR v2 at about 2.0% fewer tokens and Express at about 42.6% fewer tokens than FlatSpec. Exact deployed-Gemma tokenizer counts remain blocked until that tokenizer is supplied locally; the report labels this limitation explicitly.
- DONE — Compared reversibility, validity, size, structured-generation risk, wire overhead, and maintenance implications in the implementation review.

## 11. Delivery

- DONE — Updated this checklist and added a technical review report, machine-readable test report, benchmark JSON/CSV, and exact changed-file manifest.
- DONE — The final Git commit is the authoritative source diff and is pushed to the existing feature branch.
- DONE — A source-only ZIP is generated from the final commit as an external handoff artifact; it is intentionally not committed into the repository.
