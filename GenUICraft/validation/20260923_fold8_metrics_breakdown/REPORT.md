# Fold8 generation timing and MTP-acceptance validation

Date: 2026-09-23. Device: Samsung SM-F776U (Fold8), serial
`R3GL203AKSF`, Android 17 / API 37. App: `com.samsung.genuicraft` 1.1.0.
Route: **GenUICraft SDK · Bixby50**, trained E2B v10 W4, GPU, MTP on,
thinking off, metrics on. Case: BXP-001.

## Why 288 tokens can take longer than `288 / decode rate`

Native decode throughput covers only the interval that generates output tokens.
For the earlier displayed run, the recoverable arithmetic is:

```text
native decode = 288 tokens / 58.39 token/s = 4.93 s
outside native decode = 10.07 s - 4.93 s = 5.14 s
```

That earlier build did not retain engine-initialization wall time, prefill rate,
provider-call wall time, or validation/recovery timing. The 5.14 seconds therefore
cannot be split further after the fact without inventing values.

The current implementation records those boundaries for each new run. Prompt
prefill and native decode durations are derived from the matching native token
counts and rates. Engine initialization and the provider call are measured with a
monotonic wall clock. Runtime/callback overhead is the unaccounted provider wall
remainder, and validation/recovery is the conversion wall remainder after all
provider calls. LiteRT's native init-phase sum and time to first token are shown as
diagnostics because they overlap other phases and are not additive.

## Verified live breakdown

The final installed build's live run displayed 3,174 input tokens, 288 output tokens and 58.09
output token/s. Its displayed rounded breakdown was:

| Phase | Time | How it is obtained |
|---|---:|---|
| Complete conversion | **9.06 s** (`9,058 ms`) | Converter wall clock |
| Provider call wall | **9.00 s** | Streaming provider wall clock |
| Engine initialization | 2.58 s | Measured native-engine construction wall time |
| Prompt prefill | 1.46 s | `3,174 / 2,180.48 token/s` |
| Native decode | 4.96 s | `288 / 58.09 token/s` |
| Runtime/callback overhead | 14 ms | Provider wall minus init, prefill and decode |
| Validation + recovery | 54 ms | Conversion wall minus provider wall |

The unrounded counters reconcile exactly. Adding the independently rounded labels
can differ by a few milliseconds:

```text
2.58 s + 1.46 s + 4.96 s + 0.014 s = 9.014 s from rounded labels
underlying unrounded values = 9.00 s provider wall
9.00 s + 0.054 s = about 9.06 s complete conversion
```

## Drafter acceptance

LiteRT logged an aggregate MTP drafter success ratio of `0.558642` after the
complete conversion session. The app displays this as **55.86% MTP acceptance**.
This is verified draft tokens divided by proposed draft tokens. It is independent
of the 58.09 output token/s decode throughput.

LiteRT currently publishes the aggregate counter when the MTP drafter is
destroyed. With both metrics and MTP enabled, the SDK therefore finalizes the
engine after all generation/repair attempts, reads only the app process's bounded
log window, and recreates the engine for the next request. Metrics-off generation
retains normal engine reuse. Missing or failed telemetry never changes a successful
conversion result.

## Evidence and validation

- [Final expanded performance panel](final_breakdown.png)
- [Filtered app-process runtime log](runtime_filtered.log)
- APK SHA-256:
  `9cdaca397519c69c85e2eb22190b96431cab474f5fecb143ae3b048eb72c698d`
- Replacement install completed on `R3GL203AKSF` without clearing app data.
- The final process diagnostics contained no fatal-exception or ANR marker.
- SDK unit tests: **376 passed**, zero failures/errors/skips.
- Android app unit tests: **361 passed**, zero failures/errors/skips.
- The debug APK assembled successfully from the locally published GenUICraft AAR.
