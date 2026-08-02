# A2UI Express single-active-IR implementation report

## Outcome

A2UI Express v1 is the only active model-output and training-target format. The
standard A2UI v0.9 wire form remains an internal compiler/transport boundary.
FlatSpec is retained only for read-only legacy import, deterministic migration,
semantic round-trip tests, and offline baselines. Compact IR v2 is isolated
under migration-only modules and is rejected by active ingestion paths.

## Architecture

Before this change, Stage 3, training, and Android could select or ingest more
than one compact/legacy representation. The active path is now:

```text
model -> strict A2UI Express parser -> catalog/profile validation
      -> standard A2UI v0.9 compiler/schema validation
      -> canonical typed UI graph -> Android native renderer
```

Historical data enters only through:

```text
FlatSpec or Compact IR (migration tool only)
      -> strict legacy decoder -> canonical graph -> Express
```

An invalid raw Express completion is recorded as a format error. It is not
silently parsed as FlatSpec/Compact IR and does not produce a hidden fallback UI.

## Standards and pinned identities

- A2UI repository: `a2ui-project/a2ui`
- pinned upstream commit: `2276f8cc702eaeac25ffb05be85797b2a1205c74`
- protocol: `v0.9` (vendored `specification/v0_9`)
- Express profile: `genuicraft-express-v1`
- pinned Express grammar blob: `4f2492ae4600598d8b10e68fcd9f4292529dd653`
- catalog: `genuicraft-a2ui-catalog-v1`
- canonical graph: `canonical-ui-graph-v1`
- machine-readable identity manifest: `dataset/schema/genuicraft_ir_formats.manifest.json`

The generator/check command verifies the catalog, profile, canonical schema,
wire schema, prompt, Python codec, Kotlin codec, migration tool, grammar, and
manifest hashes:

```text
python dataset/scripts/generate_a2ui_express_prompt.py --check
A2UI Express prompt verified (5 files)
python dataset/scripts/generate_ir_specs.py --check
IR artifacts verified: 21 files
```

## Main implementation areas

- `dataset/src/pipeline/ir_formats/active.py`, `express.py`, and
  `android/app/src/main/java/com/samsung/genuicraft/pipeline/A2uiExpressCodec.kt`:
  strict sentinels, assignments, positional/named arguments, references,
  arrays/maps, actions, duplicate/unknown/invalid-value rejection, and
  deterministic lossless encoding.
- `dataset/src/pipeline/ir_formats/a2ui_wire.py` and
  `android/app/src/main/java/com/samsung/genuicraft/pipeline/A2uiWireCodec.kt`:
  standard root component (`id: root`), bindings, child lists, actions, and
  schema-facing wire validation.
- `dataset/src/pipeline/ir_formats/catalog.py`,
  `dataset/schema/genuicraft_a2ui_catalog_v1.json`, and
  `dataset/schema/genuicraft_a2ui_express_profile_v1.json`: generated strict
  catalog/profile contract, closed typed component-property schemas, explicit
  renderer properties, required action parameters, and shared
  reference/repeat metadata.
- `dataset/src/pipeline/ir_formats/canonical.py` and
  `android/app/src/main/java/com/samsung/genuicraft/pipeline/A2uiCanonicalGraph.kt`:
  strict canonical graph validation for root/state/elements, component and
  property allowlists, action parameter schemas, repeat/template fields,
  references, cycles, and renderer-safe values. Raw parse validity is kept
  separate from any explicit repair result.
- `dataset/src/pipeline/stage3_genui.py` and Android Stage 3 pipeline/prompt
  builders: Express-only generation with no Compact/FlatSpec fallback. URL and
  local-asset references are masked before provider prompts and restored only
  after strict parsing. The repair prompt is the generated contract, not a
  second hand-maintained signature list.
- `dataset/scripts/migrate_legacy_dataset_to_a2ui_express.py` and the isolated
  `dataset/src/migration/` package: one-time migration boundary with hashes,
  source preservation, strict rejects, deterministic output, and resume mode.
- `training/src/ir_training/data/`, `training/scripts/train_grpo.py`, and the
  three Express configs: Express-only target materialization, masked URL/path
  metadata (including local asset paths), provider-token accounting with a
  clearly labeled lexical diagnostic fallback, and source-group split
  isolation. GRPO rewards validate raw Express directly.
- `android/app/src/main/java/com/samsung/genuicraft/renderer/GenUiNativeRenderer.kt`:
  read-only FlatSpec compatibility is explicitly separated from production
  Express ingestion.
- `SettingsActivity` and `IrDemoListActivity` no longer expose an IR-format
  selector; the internal prompt-version compatibility API resolves only to
  A2UI Express and cannot persist or select a legacy format.

## Coverage

The generated catalog covers 27 entries: 25 native component names plus the
explicit `Row` and `Column` aliases that lower to `Stack`. It covers the six
actions `openUrl`, `setState`, `pushState`, `removeState`, `validateForm`, and
`emitEvent`. The complete positional/property/reference metadata is in
`a2ui_express_component_action_coverage.csv`.

## Token and quality optimization

The encoder applies trailing-default elision, named arguments when skipping
middle positions, deterministic short IDs, stable ordering, empty-value
omission, and safe single-use inlining. Sparse positional arguments are emitted
as named arguments instead of an ambiguous gap. These optimizations preserve
the canonical semantic hash; they do not remove meaningful UI components.

The benchmark report compares the retained FlatSpec baseline, Express, and
standard wire forms. It used a deterministic lexical tokenizer because the
deployed Gemma tokenizer/checkpoint was not present; the report is explicitly
marked `exact_for_deployed_model: false`, includes mean/median/p50/p90/p95,
and must not be read as provider token counts. On 32 representative fixtures,
Express round-tripped 32/32 and had a 41.47% mean lexical-token reduction
versus FlatSpec; the wire form is an internal representation and was larger in
this diagnostic. Token metrics are regression diagnostics only and cannot
select a production format or checkpoint. The exact-token gate remains
BLOCKED.

## Verification and known limitations

The combined Python dataset/training suite passed 476 tests with 7
deprecation warnings, and the Android JVM suite passed 297 tests across 39 XML
suites. The connected instrumentation smoke passed on both detected targets,
`SM-F731U` (Flip) and `SM-F966B`, Android 16. On the Flip, the inspected
Express screenshot shows the A2UI Express title, native Compose card, and
Details/Ready state; the window hierarchy contains the Table component. The
test-process screenshot SHA-256 is
`681EAC79B768368CDE150349EF2FE48A33812F970F464FD41D906A865A81173E` and the
window-dump SHA-256 is
`953DAD724A6C96EB91A307D8987BEB2441E547362AEC947B71748711DC928D09`.
The GenUI Demo landing view was also opened on the Flip; its screenshot SHA is
`621C47B25A2936039D8ABAEF89CFE800133A15FD0B08A5569A4DF66894BD727A`, and the
hierarchy exposes the generated-contract copy, the Flights BLR to LKO starter,
and the Native render tile. Selecting that starter on the device produced
`Show flights from BLR to LKO on 18 August 2026` on device date `2026-08-03`,
confirming the current-date-plus-15-days example. Device images are review
evidence and are excluded from the source-only ZIP.

Python and Kotlin both consume the byte-identical fixture
`a2ui_express_conformance_v1.json`; the conformance report is now PASS for the
shared acceptance/rejection corpus. The remaining limitation is that the two
strict parsers are handwritten rather than generated from one parser artifact.

Two end-to-end model calls remain environment-blocked, not code-passing:

1. Vertex Express Gemini 2.5 Flash Lite reached Google and returned HTTP 403
   because billing is disabled for the project associated with the supplied
   key.
2. The connected device does not contain the requested Gemma 4 E2B `.litertlm`
   checkpoint in either app model directory, so the MTP test exits with a
   precise missing-model diagnostic without downloading, deleting, or resetting
   app data.

These limitations are not reported as successful model-generation tests. A
second connected device was present during the latest instrumentation
invocation and also completed the one-test smoke successfully; the release
report therefore does not claim a secondary-device installation failure.

The final cleanup scan found no Compact/dual-format imports in the active Stage
3, active codec, training target, or Android inference modules. The only
`decode_to_flat_spec` hits are explicit legacy-source/migration boundaries; the
only Android FlatSpec references are read-only renderer compatibility,
diagnostics, and negative rejection checks. A static policy test now guards
these boundaries and prevents a second active IR from being reintroduced.
The shared prompt generator and drift test cover dataset, training fallback,
Android inference, Android repair, and the Gemma asset mirror byte-for-byte.
