# Gemma v7 live-run and v9 revalidation evidence audit

Date: 18 September 2026

This is a read-only host-filesystem audit of:

- `A2UI/tmp/genuicraft_20260918/gemma50_v7_gpu_mtp`
- `A2UI/tmp/genuicraft_20260918/gemma50_v9_revalidated`
- `v7_build_manifest.json`, `v9_build_manifest.json`, and their matching `*_sdk_source_hashes.json` files

No build, device command, or inference was run for this audit.

## v7 live Gemma run

The v7 directory is the completed live Gemma run. Its run configuration, aggregate results, and case directories each contain exactly one copy of every expected ID from `BXP-001` through `BXP-050`, with no missing or unexpected IDs.

| Result | Verified count |
| --- | ---: |
| Successful cases | 50 |
| Failed cases | 0 |
| First-attempt successes | 47 |
| Repaired successes | 3 |
| Total live model attempts | 53 |
| Fallbacks | 0 |

The repaired cases are `BXP-030`, `BXP-032`, and `BXP-037`; each succeeded on attempt 2. Every one of the 53 attempt metric files identifies `LiteRT-LM/Gemma4/GPU+MTP`. Every aggregate result exactly matches its per-case `result.json`, and all cases report `usedFallback=false`.

## Captured-output provenance

For each of the 50 cases, the successful v7 attempt number selects `attempt_N.txt`. The SHA-256 of that actual file matches both:

1. the copied v9 `source.attempt_N.txt`; and
2. `sourceRawSha256` in the v9 per-case replay result.

All 50 `sourceSuccessfulAttempt` values and `sourceArtifact` paths point to the same successful v7 attempt. The SHA-256 of every v7 `output.a2ui.json` also matches its v9 `source.output.a2ui.json` copy and recorded `sourceJsonSha256`.

The v9 replay configuration records corpus SHA-256 `fc46aa381957bed206f9e0f53e28edbb21b5096ca093985ad184fda73faaea0a`; this matches the actual `android/app/src/main/assets/genuicraft_bixby50.jsonl`. Its 50 per-case `revalidation_request.json` payloads exactly match the corresponding corpus query, text, and sources.

## v9 raw-model revalidation

The v9 directory is a revalidation of the captured successful v7 model text through the v9 converter and renderer. It is not a second live Gemma run.

| Result | Verified count |
| --- | ---: |
| Expected unique cases | 50 |
| Rendered cases | 50 |
| Failed cases | 0 |
| Live model calls | 0 |
| Cases with replay issues | 0 |
| Declared screenshot/XML captures | 221 |
| Tables checked | 23 |
| Tables with missing columns | 0 |

Every case records `modelCalls=0`, `inferenceEvaluated=false`, and `capturedProviderCalls=1`. The one captured-provider call is the replay adapter returning saved model text; it is not live inference. All 221 declared captures have both PNG and XML artifacts and report successful screenshot capture. They comprise 50 initial, 103 vertical, 48 horizontal, and 20 table-reset captures.

Twenty of the 23 tables recorded horizontal movement. `BXP-001`, `BXP-003`, and `BXP-025` did not need observed horizontal movement because the checks found every required header or representative cell without it. The maximum horizontal capture index was 5 under the configured limit of 8. Every case reports `verticalEndObserved=true`, no case reports `verticalLimitReached`, and the observed maximum was 3 vertical swipes under the configured limit of 12.

Nine revalidated wire-JSON documents differ from their saved v7 documents: `BXP-005`, `BXP-025`, `BXP-027`, `BXP-032`, `BXP-033`, `BXP-034`, `BXP-037`, `BXP-044`, and `BXP-049`. Parsed leaf comparison found exactly 17 changes, all additions of `variant="heading"`. No text, value, action, or source field changed.

## Source and artifact closure

The v7 and v9 source-hash manifests have the same complete 106-file `genuicraft/src/main` keyset. Exactly six files changed:

- `genuicraft/src/main/java/com/samsung/genuicraft/sdk/SourceBindings.kt`
- `genuicraft/src/main/java/com/samsung/genuicraft/sdk/internal/renderer/FlatSpecRenderer.kt`
- `genuicraft/src/main/java/com/samsung/genuicraft/sdk/internal/renderer/FlatTableLayout.kt`
- `genuicraft/src/main/java/com/samsung/genuicraft/sdk/internal/renderer/flat/compose/FlatDirectTableRender.kt`
- `genuicraft/src/main/java/com/samsung/genuicraft/sdk/internal/renderer/flat/domain/FlatComparisonDomain.kt`
- `genuicraft/src/main/java/com/samsung/genuicraft/sdk/internal/renderer/flat/domain/FlatItineraryDomain.kt`

This list exactly matches `v9_build_manifest.json`. All three provider files, both bundled prompts, and all three SDK JNI files have identical v7/v9 hashes. Every current v9 `src/main` file rehashes to its manifest entry, and the prompt/JNI payloads embedded in both retained AARs match their corresponding source manifests.

Verified artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| v7 AAR | `4a002cbb409a976d8b43f5d62e55030951d2857fa9f1f96c6c1d2ac81fac44a8` |
| v9 AAR | `50b001484a4ec08985d9e70a95eb2096fa07bb83f99088117654360fdaf24ba9` |
| v9 app APK | `e251cbd5e79330627fec9357398fd97990569a77d2c38d1933fd9673a0ecadba` |
| v9 test APK | `74a1b626d46aacc4db5ef0582e80a3226a8743150e52211eb57ca11b3a793388` |

The current release AAR and local Maven AAR also match the v9 AAR hash.

## Limits

- The v9 run proves deterministic revalidation and rendering of the captured v7 model outputs; it is not fresh v9 model-quality or latency evidence.
- Text-node and column observation with bounded scrolling does not prove pixel-perfect layout, visibility of every row at once, or full semantic equivalence.
- The retained v7 app and test APK binaries are absent, so their hashes recorded in the v7 manifest could not be independently recalculated. The retained v7 AAR was recalculated successfully.
- The v7 run configuration did not record a corpus hash. The v9 configuration does, all v9 requests match that corpus exactly, and all captured v7 outputs passed v9 validation against those requests.
- A historical v7 source tree was not retained. The six-file scope is established by the two source-hash manifests; semantic characterization beyond the verified heading-only output delta relies on those recorded historical hashes rather than a standalone v7 source diff.
