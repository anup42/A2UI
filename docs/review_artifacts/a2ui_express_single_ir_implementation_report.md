# A2UI Express single-active-IR implementation report

## Outcome

A2UI Express v1 is the only active model-output and training-target format. The
standard A2UI v1 wire form remains an internal compiler/transport boundary.
FlatSpec is retained only for read-only legacy import, deterministic migration,
semantic round-trip tests, and offline baselines. Compact IR v2 is isolated
under migration-only modules and is rejected by active ingestion paths.

## Architecture

Before this change, Stage 3, training, and Android could select or ingest more
than one compact/legacy representation. The active path is now:

```text
model -> strict A2UI Express parser -> catalog/profile validation
      -> standard A2UI v1 compiler/schema validation
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
- protocol: `v1.0`
- Express profile: `genuicraft-express-v1`
- pinned Express grammar blob: `4f2492ae4600598d8b10e68fcd9f4292529dd653`
- catalog: `genuicraft-a2ui-catalog-v1`
- canonical graph: `canonical-ui-graph-v1`
- machine-readable identity manifest: `dataset/schema/genuicraft_ir_formats.manifest.json`

The generator/check command verifies the catalog, profile, canonical schema,
wire schema, prompt, Python codec, Kotlin codec, migration tool, grammar, and
manifest hashes:

```text
python dataset/scripts/generate_ir_specs.py --check
IR artifacts verified: 10 files
```

## Main implementation areas

- `dataset/src/pipeline/ir_formats/express.py` and
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
  catalog/profile contract and explicit renderer properties.
- `dataset/src/pipeline/stage3_genui.py` and Android Stage 3 pipeline/prompt
  builders: Express-only generation with no Compact/FlatSpec fallback.
- `dataset/scripts/migrate_legacy_dataset_to_a2ui_express.py` and the isolated
  `dataset/src/migration/` package: one-time migration boundary with hashes,
  source preservation, strict rejects, deterministic output, and resume mode.
- `training/src/ir_training/data/` and the three Express configs: Express-only
  target materialization and source-group split isolation.
- `android/app/src/main/java/com/samsung/genuicraft/renderer/GenUiNativeRenderer.kt`:
  read-only FlatSpec compatibility is explicitly separated from production
  Express ingestion.

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
deployed Gemma tokenizer was not present; the report is explicitly marked
`exact_for_deployed_model: false` and must not be read as provider token
counts. On 32 representative fixtures, Express round-tripped 32/32 and had a
41.5% mean lexical-token reduction versus FlatSpec; the wire form is an
internal representation and was larger in this diagnostic.

## Verification and known limitations

Python dataset tests (410 passed), training tests (48 passed), and the Android
JVM suite (294 passed) are recorded in `a2ui_express_test_report.json`. The
connected Flip smoke test passed and the inspected screenshot shows the title,
card, and table rendered through the native Compose path. The exact capture
hash and test command are recorded in that report; the image is intentionally
kept outside the source-only archive.

Two end-to-end model calls remain environment-blocked, not code-passing:

1. Vertex Express Gemini 2.5 Flash Lite reached Google and returned HTTP 403
   because billing is disabled for the project associated with the supplied
   key.
2. The connected device does not contain the requested Gemma 4 E2B `.litertlm`
   checkpoint in either app model directory, so the MTP test exits with a
   precise missing-model diagnostic without downloading, deleting, or resetting
   app data.

These limitations are not reported as successful model-generation tests.
