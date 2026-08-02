# Compact IR Dual-Format Implementation Review

Date: 2026-08-02
Branch: `new_ir_changes_20260331`
Starting repository commit: `1051ae31903cf61aadcab950899fabb99e726ae4`

## Outcome

The GenUICraft pipeline now supports two lossless model-facing formats—Compact IR v2 JSON and A2UI Express v1—while retaining FlatSpec as the canonical renderer graph and migration format. A custom-catalog A2UI v1 wire representation is also available for protocol interchange. Dataset Stage 3 generates Compact and Express independently by default, validates them without a hidden FlatSpec generation fallback, and selects a result by quality first and token size only within a configurable quality tolerance.

The supplied full and compact archives were treated as a handoff, not as proof of completion. Their checksums and package metadata were verified, the compact-IR scope was reconciled with the current feature branch, and the missing Stage 3, training, Android parity, testing, and delivery work was completed in this commit. A separate QAT/MTP task that began changing the same worktree was not part of either archive and is excluded from this change.

## Runtime architecture

```text
Compact IR v2 JSON ─┐
A2UI Express v1 ────┼─> strict format codec ─> canonical FlatSpec graph ─> existing Android renderer
A2UI v1 wire ───────┤
FlatSpec migration ─┘
```

The Python and Kotlin codecs share these contract identities:

- upstream A2UI commit `2276f8cc702eaeac25ffb05be85797b2a1205c74`;
- Express grammar blob `4f2492ae4600598d8b10e68fcd9f4292529dd653`;
- generated GenUICraft component/action catalog;
- generated wire and Compact IR schemas;
- a reference-semantics inventory whose hash invalidates stale comparison results when traversal semantics change.

## Correctness work

Reference traversal now includes ordinary children plus Tabs content aliases, Modal trigger/content, top-level and property templates, repeat templates, and legacy aliases. The same inventory drives missing-reference validation, cycle detection, reachability/pruning, deterministic ID rewriting, and graph-depth metrics. This fixes the previous behavior where valid Tabs or template content could be pruned simply because it was not reachable through `children`.

Both compact codecs preserve the full renderer graph, arbitrary forward-compatible properties, visibility, watches, repeats, event maps, accessibility data, and the existing action inventory. `emitEvent` is now present in the schema, capability manifest, Android action runtime, Express compiler, and wire/Compact codecs.

Malformed native payloads fail explicitly. They are not normalized into an empty value and are not replaced by a simpler generated FlatSpec UI. Historical FlatSpec remains readable for migration and deterministic conversion.

## Dataset and training behavior

Stage 3 defaults to `compact_ir_v2` and `a2ui_express_v1`. Each format has its own prompt, model call, parsing, repair/regeneration, strict validation, record ID suffix, telemetry, and accepted/rejected record. Accepted rows retain the raw completion, parsed raw payload, normalized native representation, canonical graph, format and codec identity, semantic hash, byte/character/token measurements, and latency. Rejected native outputs use `record_status: format_rejected`.

Training examples are materialized from the canonical graph into format-native completion targets and verified by semantic hash. Historical rows produce one Compact IR example by default so dataset size does not unexpectedly double; configuration can explicitly request both targets, while already-native rows preserve their source format.

## Benchmark result

All 32 fixture specs round-trip semantically in FlatSpec, Compact IR v2, Express, and A2UI v1 wire. With the deterministic lexical estimator:

| Format | Token change vs FlatSpec | Character change vs FlatSpec | Interpretation |
| --- | ---: | ---: | --- |
| Compact IR v2 | 2.0% fewer | 9.3% fewer | Lowest-risk structured JSON generation |
| A2UI Express v1 | 42.6% fewer | 35.4% fewer | Best measured representation reduction |
| A2UI v1 wire | 25.9% more | 41.0% more | Interchange protocol, not a compact model target |

These token values are estimates, not deployed-model measurements. The repository did not include the deployed Gemma tokenizer, so the benchmark records `exact_for_deployed_model: false` and can be rerun with a local SentencePiece or Hugging Face tokenizer path.

## Verification

- Generated schema/catalog drift check: 5 files verified.
- Focused codec, metrics, conversion CLI, and benchmark CLI tests: 17 passed.
- Broader relevant dataset/training suite: 89 passed with 7 deprecation warnings and no failures.
- Android unit suite: 265 passed across 33 suites, with no failures, errors, or skips.
- Android Kotlin compilation and `assembleDebug`: successful.
- Corpus equivalence: 32 of 32 specs passed in each of four representations.
- Git whitespace check: passed; only repository line-ending conversion notices were emitted.

Machine-readable evidence is in `compact_ir_dual_format_test_report.json`; detailed corpus rows and aggregates are in `ir_token_benchmark.csv` and `ir_token_benchmark.json`.

## Remaining evidence boundary

The implementation is build- and test-verified locally. It was not exercised against a live Gemini request or a physical Android device in this delivery, and exact deployed-Gemma token counts remain unavailable until the tokenizer is supplied. These are runtime/evaluation follow-ups, not hidden fallbacks or known compile/test failures.
