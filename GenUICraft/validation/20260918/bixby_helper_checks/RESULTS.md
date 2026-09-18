# Isolated Bixby GenUICraft helper validation

Validated on 2026-09-18 at 04:11 IST: **two fresh passes, each with 32 tests, 0 failures, 0 errors, and 0 skipped tests**.

This harness compiled verbatim copies of the two Bixby helper implementation files, two Bixby result-data files, the GenUICraft SDK API file, and five Bixby test files. It invoked the GenUICraft SDK Gradle 8.10.2 wrapper from this isolated project. It did not invoke Bixby's Gradle wrapper or build files, build an APK, install anything, or issue a device command.

## Test results

| Exact Bixby test class | Tests | Pass 1 | Pass 2 |
| --- | ---: | ---: | ---: |
| `GenUiActionPolicyTest` | 4 | 4 passed | 4 passed |
| `GenUiHostAttachmentStateTest` | 1 | 1 passed | 1 passed |
| `GenUiRequestLifecycleTest` | 6 | 6 passed | 6 passed |
| `GenUiSourceParserTest` | 9 | 9 passed | 9 passed |
| `PerplexityStreamAccumulatorTest` | 12 | 12 passed | 12 passed |
| **Total** | **32** | **32 passed** | **32 passed** |

The exact passing methods were:

- `GenUiActionPolicyTest`
  - `should reject non web action schemes()`
  - `should allow an explicit https openUrl action()`
  - `should reject source text as an action name()`
  - `should reject non-public and credential-bearing links()`
- `GenUiHostAttachmentStateTest`
  - `retained presentation covers legacy view again after reattach()`
- `GenUiRequestLifecycleTest`
  - `late final from replaced request cannot become current()`
  - `structured stream can establish a current request when start was not observed()`
  - `retired request cannot be reactivated by a late stream()`
  - `history restore retires the current live request()`
  - `late final and stream after incomplete result cannot restore native content()`
  - `late final after cancellation cannot restore native content()`
- `GenUiSourceParserTest`
  - `should strip only explicit markers matching the citation id()`
  - `should reject non web source schemes()`
  - `should ignore renderer events from another request()`
  - `should parse citation ids and links from ResultDetails()`
  - `oversized citation fields should be excluded before SDK handoff()`
  - `source count should be capped at SDK limit()`
  - `later renderer event should replace citation metadata for the same id()`
  - `should preserve a host-bearing http citation()`
  - `mixed citations should keep only exact safe public links()`
- `PerplexityStreamAccumulatorTest`
  - `should reject a final-only response without structured provenance()`
  - `should reject Markdown when provider metadata is not Perplexity()`
  - `should ignore an old final event after a newer request starts()`
  - `should extract only displayable final messages for the request()`
  - `should continue an identified request when later metadata is absent()`
  - `should replace snapshots and append deltas for one request()`
  - `should preserve repeated delta chunks()`
  - `should accept a structured final-only response()`
  - `should use StreamComplete as the authoritative response()`
  - `should accept stream modes case insensitively()`
  - `should retain numbered source ids used by citation markers()`
  - `should ignore final events after cancellation()`

The second-pass JUnit XML reports 0.397 seconds across the five suites. The complete Gradle invocations took 35 seconds and 16 seconds respectively. Both logs contain exactly 32 `PASSED` lines, no `FAILED` lines, and one `BUILD SUCCESSFUL` marker.

## What this proves

- The copied accumulator passes its tests for exact Perplexity provider classification, snapshot/delta/final stream handling, case-insensitive modes, stale/canceled request suppression, source attachment, and result-message filtering.
- The copied lifecycle state passes replacement, cancellation, incomplete-result, history-restore, implicit-stream-start, and immediate retired-request checks.
- The copied source parser passes request-ID filtering, recursive citation extraction, duplicate-ID replacement, matching citation-marker cleanup, scheme/credential/IP/placeholder rejection, field-size limits, and the 100-source cap.
- The copied action policy passes the explicit `openUrl` and public-looking HTTP(S) cases in the original tests.
- The copied host attachment state reapplies retained content after detach/reattach and only covers the legacy view while attached.

## Bounded findings and limits

1. The test named `should reject a final-only response without structured provenance()` supplies both an empty source list and blank provider metadata. The production helper rejects on the blank/non-target metadata (or blank text); it does not require `sources` to be non-empty. Thus this test proves provider-identity provenance, not mandatory citation provenance. If the requirement is “every final-only response must contain at least one citation,” neither the implementation nor this test enforces it.
2. `GenUiPublicUrlPolicy` performs URI/hostname syntax checks and a fixed blocklist. It does not resolve hostnames or establish that a hostname routes to a public address. The passing tests therefore prove the listed inputs only; they are not network-level SSRF or reachability proof.
3. `GenUiSourceParser.parse(requestId, events)` filters renderer events by request ID, then recursively accepts citation-shaped objects. It does not validate renderer type or capsule-context flags. This is a source-parser contract boundary rather than a failure in the tested cases.
4. The lifecycle keeps at most 32 retired IDs. The immediate late-event tests pass, but they do not exercise eviction of the oldest retired ID or request-ID reuse after that bound.
5. The host attachment test exercises only `GenUiHostAttachmentState`. Android `Looper`/`Handler` dispatch and the real `GenUiHostView` are not executed, so this is not Android lifecycle, threading, view, rendering, or device proof.
6. The integration manager, converter, provider calls, coroutines, renderer event plumbing, and SDK rendering remain outside this isolated test scope.

## Runtime and explicit shims

- Gradle wrapper: GenUICraft SDK 8.10.2.
- Launcher/runtime: Android Studio JetBrains OpenJDK 21.0.7; Java and Kotlin JVM target 17.
- Kotlin JVM plugin/stdlib: 2.2.21.
- Kotlin serialization JSON: 1.9.0.
- JUnit Jupiter/platform: 6.1.3, matching the Bixby version catalog.
- Repositories: Maven Central and Gradle Plugin Portal only.

Three harness files exist solely to compile the unmodified source on a plain JVM:

- `AndroidOsStubs.kt`: minimal `Looper` and `Handler` declarations needed by the unexercised `GenUiHostRegistry` portion of `GenUiHostRegistry.kt`.
- `LoggingStub.kt`: minimal `toSafeStringOrLength` extension needed by the copied Bixby result data `toString()` methods; those methods are not called by the tests.
- `GenUiHostViewStub.kt`: minimal `show(GenUiPresentation)` declaration needed by the unexercised host registry dispatch path.

No production behavior was reimplemented for the accumulator, lifecycle, source parser, action policy, or attachment-state methods under test.

## Verbatim source provenance

| Copied file | SHA-256 |
| --- | --- |
| Bixby `PerplexityStreamAccumulator.kt` | `DF1A37DB0A13B65C6D6AF4E1C8BF1821233FF515D9CE68DB8240BEACED42D7C9` |
| Bixby `GenUiHostRegistry.kt` | `2D41151BD78A729C0EEEE265B5B1D03388E905325B428D57AFD99B03AAE8F6FD` |
| Bixby `ResultMessageData.kt` | `073EAE8343A0D0842EE512E764DE1F2C87F0292EB44F2DCB1BF9CB5A4F5C046E` |
| Bixby `RendererEventData.kt` | `1E2C85DEAC5AC37ACB01202AEFA92018C5E158461D0E7CBEE4263001478D0FEE` |
| GenUICraft SDK `Api.kt` | `049D8EB630E9F56EB89B830A78BED56F7D3D6FFF40242E20177B3EF97E830190` |
| Bixby `PerplexityStreamAccumulatorTest.kt` | `7558FB66E207616094340B1F59A9069ED53383E660D9D4CAD402E5E66CEFA0BA` |
| Bixby `GenUiSourceParserTest.kt` | `73F6ACBA85CC5923091BC77DA02463FA8D97EB154395FB667D613D65AAD9E829` |
| Bixby `GenUiRequestLifecycleTest.kt` | `1E6CC19937DD9D271082D4CBCBC282A9A5C07478F258AE5352752931481D6908` |
| Bixby `GenUiHostAttachmentStateTest.kt` | `977E6A9DE3E09E99B6B74CF579A6FC8926C063C0D0904E9E48813EF969B53EEF` |
| Bixby `GenUiActionPolicyTest.kt` | `0604E3BC7F8D6E6C705F28E983CEBDF976E37809400AB468FAE6F466EFAE843F` |

`source-hashes.json` records each original and copy path. `post-run-source-hashes.json` confirms that every original retained the tested SHA-256 after both passes. All ten copied files still byte-match their originals.

## Reproduction and evidence

From this directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-checks.ps1 -Passes 2
```

Evidence files:

- `gradle-test-pass1.log` — SHA-256 `31F95F27D9A70307F610839BC0ECA3BF97B3DEC22F241BC4CABA9936002A74AE`
- `gradle-test-pass2.log` — SHA-256 `E69517B339166C4B6416883302EECD7677C019B8C4D3C99061543F35B464059D`
- `source-hashes.json`
- `post-run-source-hashes.json`
- `stub-hashes.json`
- `build/test-results/test/TEST-*.xml`
- `build/reports/tests/test/index.html`

This entire harness is a temporary validation artifact under `A2UI/tmp`; it does not modify either production tree.
