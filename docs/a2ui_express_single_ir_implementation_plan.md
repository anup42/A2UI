# A2UI Express single-IR implementation plan

This plan is the review checkpoint for the attached single-active-IR directive. It is intentionally created before source changes for this phase. Statuses are updated as evidence is produced; a gate is never marked complete without a recorded command, fixture, or report.

| Work item | Status | Evidence / final note |
| --- | --- | --- |
| Current architecture and active entry points | DONE | `docs/review_artifacts/a2ui_express_single_ir_implementation_report.md` |
| Target architecture: legacy FlatSpec import -> canonical graph -> A2UI Express -> standard A2UI -> native renderer | DONE | Same report; active Stage 3 and Android paths are Express-only. |
| Complete inventory of IR-related files and references | DONE | `a2ui_express_changed_files.txt` plus repository-wide cleanup scan. |
| Component inventory and native renderer coverage | DONE | `a2ui_express_component_action_coverage.csv`; 27 catalog entries and native registry coverage. |
| Property inventory per component | DONE | Coverage CSV and generated catalog/profile. |
| Action/function inventory | DONE | Coverage CSV; six strict actions. |
| Renderer-reference inventory | DONE | Existing `flat_spec_reference_inventory_v1.json` fixture and codec/reference tests. |
| Legacy dataset formats and source records | DONE | FlatSpec and Compact are recognized only at the migration boundary. |
| Compact IR isolation/removal from active paths | DONE | Isolated migration decoder; active codec/prompt/config/static tests reject Compact. |
| FlatSpec read-only migration/comparison isolation | DONE | Renderer compatibility is explicit and model-output fallback is rejected. |
| Pinned A2UI protocol version and upstream commit | DONE | Manifest pins vendored A2UI `specification/v0_9` (`v0.9`) and upstream commit `2276f8cc702eaeac25ffb05be85797b2a1205c74`. |
| Pinned Express grammar version/blob or commit | DONE | Manifest pins `genuicraft-express-v1` and grammar blob `4f2492ae4600598d8b10e68fcd9f4292529dd653`. |
| Strict standard catalog | DONE | Generated catalog has closed schemas and explicit renderer properties. |
| Express inference profile | DONE | `dataset/schema/genuicraft_a2ui_express_profile_v1.json`. |
| Python/Kotlin parser strategy and equivalence | PARTIAL | Both strict codecs pass local suites and consume the shared conformance corpus; handwritten implementations still need a generated parser/equivalence gate. The limitation is explicit rather than treated as complete. |
| Canonical graph and semantic hash | DONE | Canonical schema and round-trip hash checks in Python/Kotlin/migration tests. |
| Standard A2UI compiler and schema validation | DONE | Wire codec requires standard `root` component and validates schema-facing payloads. |
| Explicit shared repair layer and validity separation | DONE | Raw/native/repaired fields are separate in dataset metrics and Android Stage 3; malformed-raw/valid-repaired coverage passes in `training/tests/test_express_metrics.py`, and repair prompts load the generated Express contract. |
| Shared prompt generation and drift check | DONE | `dataset/scripts/generate_a2ui_express_prompt.py --check` verifies five byte-identical mirrors; `dataset/tests/test_a2ui_express_prompt_generation.py` passes. |
| Lossless Express encoder/decoder and optimization rules | DONE | Strict codec tests, sparse positional fix, deterministic/default-elision rules, and benchmark round-trips. |
| Legacy dataset migration command, resume, manifests, rejects | DONE | 32/32 strict conversion and deterministic resume pass; migration report. |
| Stage 3 generation pipeline Express-only | DONE | Express-only prompt/format and wrong-native-format rejection tests. |
| SFT/GRPO/preference/MTP/evaluation Express-only targets | DONE | Express configs/target guards, strict GRPO reward, generation/MTP evaluators, and training tests pass. Live MTP model execution is separately blocked by device state. |
| Immutable source-group split isolation | DONE | `test_source_group_splits.py` passes with zero group leakage. |
| Tokenizer-accurate benchmark and regression metrics | BLOCKED | 32-sample p50/p90/p95 lexical diagnostic is labeled non-exact; exact deployed Gemma tokenizer/checkpoint is unavailable. |
| Android prompt, parser, compiler, renderer ingestion, telemetry | DONE | Update-installed Debug APK; JVM suite 297/297 and connected Express smoke 1/1 on both SM-F731U and SM-F966B; GenUI Demo landing and current-date-plus-15-days starter were inspected on the Flip; settings/IR-demo format selectors were removed. Vertex and Gemma live gates remain separately BLOCKED. |
| Cross-language fixture corpus and conformance report | DONE | `dataset/tests/fixtures/a2ui_express_conformance_v1.json` and the byte-identical Android test resource are consumed by Python and Kotlin tests; report records the remaining handwritten-parser limitation. |
| Dataset/training/Android/static cleanup tests | DONE | Combined Python suite 476 passed with 7 warnings, Android JVM 297 passed, connected Express smoke passed on both devices, and prompt/IR checks passed; cleanup scan recorded. |
| Release gates 1-20 | PARTIAL | Local gates pass; exact deployed-tokenizer and two external model gates are blocked and documented. |
| Required reports, manifest, checksums, patch, and review ZIP | DONE | Reports, manifest, changed-files inventory, code-only patch, checksums, and the source-only review ZIP are regenerated from the final staged source; the external archive hash is supplied in the handoff. |

## Working constraints

- A2UI Express is the only active model-facing and training completion format.
- FlatSpec is retained only for explicit legacy import, deterministic migration, and offline comparison.
- Compact IR is not permitted in production, dataset generation, training, evaluation, metrics, or Android ingestion. Any historical decoder is migration-only and must not be imported by active code.
- A failed Express parse is an explicit format error; it is not converted to FlatSpec, Compact IR, or a placeholder UI.
- Existing renderer behavior and meaningful UI richness must be preserved.
- Device installs, when needed, are update installs only; app data and the on-device model are preserved.
