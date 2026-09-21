# Gemma 4 E2B mobile LiteRT-LM — complete Bixby50 report

Evaluation date: 21 September 2026. **The supplied model completed all 50
Bixby50 generations on the Flip8 GPU with MTP enabled, but produced 0/50 valid
A2UI Express documents. The GenUICraft AAR compiler rejected all 50, so no
model output reached rendering.** This package is not usable for the requested
Markdown-to-A2UI conversion in its current export/runtime configuration.

## Outcome

| Measure | Result |
|---|---:|
| Inference completed | **50/50** |
| Runtime errors / timeouts | **0 / 0** |
| Raw strict Express valid | **0/50** |
| Serving-stopped strict valid | **0/50** |
| AAR compiler accepted | **0/50** |
| Rendered | **0/50** |
| Raw / serving v5.4 generation reward | **0.00 / 0.00** |
| Output-cap cases | **29/50** |

The corpus is a source-only holdout with no reference IR. The v5.4 score tests
representation of each supplied response; it does not check whether the
historical Perplexity facts are true. Because every document is invalid,
graph-based content fidelity and visual quality are unavailable. Every raw
output is retained and linked in the per-case table.

## Performance

| Measure | Result |
|---|---:|
| Engine initialization | 6.921 s |
| Full 50-case batch | 4117.520 s (68.63 min) |
| Output tokens | 70,736 |
| Input tokens | 173,431 |
| Decode median | **18.24 tok/s** |
| Decode mean | 19.30 tok/s |
| Decode token-weighted | **18.30 tok/s** |
| Decode range | 10.52–52.42 tok/s |
| First 10 / last 10 median | 23.55 / 16.22 tok/s |
| Prefill median | 702.13 tok/s |
| Case latency p50 / p95 | 94.017 / 140.608 s |
| Case latency range | 5.976–158.769 s |
| MTP draft acceptance | **68.52%** |

Native decode speed excludes engine initialization and prefill. Per-case
generation latency includes conversation setup, prefill, decode and cleanup.
The first and last ten medians show fixed-order sustained-device drift; case
complexity and thermal state are confounded. This is one run, so it provides
no variance or confidence interval and cannot establish an MTP speedup without
a matching full50 no-MTP control.

MTP speculative chunks caused 21 outputs
to exceed the requested 2,048-token cap; the largest overshoot was
3 tokens. The recorded counts and bytes
were not clipped.

## GPU and MTP proof

The exact run-label PID was `14623`. LiteRT delegated every target and drafter
graph to `LITERT_CL`, one partition each: decode 2,068/2,068 nodes,
prefill_1024 and prefill_128 1,107/1,107 each, verify 2,243/2,243, and the MTP
drafter 198/198. Engine shutdown logged an aggregate MTP success rate of
`0.685175`.

Evidence: [parsed GPU/MTP record](native/gpu_evidence.json) and
[PID-scoped log excerpt](native/gpu_evidence.txt). MTP acceptance means draft
tokens accepted by the target; it is unrelated to UI validity.

## Failure breakdown

### Strict scorer

| Category | Cases |
|---|---:|
| envelope count/incomplete | 33 |
| unterminated expression | 9 |
| unbalanced delimiter | 3 |
| children type | 2 |
| invalid statement | 1 |
| unknown property | 1 |
| empty/invalid children | 1 |

### GenUICraft AAR compiler

| Category | Cases |
|---|---:|
| incomplete/trailing envelope | 30 |
| unclosed expression | 9 |
| unbalanced delimiter | 4 |
| unknown property | 2 |
| unsupported prefix | 2 |
| invalid gap enum | 1 |
| children type | 1 |
| empty/invalid children | 1 |

One case, BXP-030, contained bytes after the first quote-aware `</a2ui>`.
Applying the serving stop changed its failure from an envelope-count error to
a missing-root error; it remained invalid. No repair or retry was applied.

Five cases reached semantic/catalog validation before compiler rejection:
BXP-002, BXP-005, BXP-008, BXP-010 and BXP-021. They still contain clear
structural and content corruption: unsupported properties or enums, non-array
children, empty/invalid references, and mutated values such as `8:30` → `8:3`
or `5:00` → `5:0:0`. Representative capped outputs show more severe token
degeneration and fact mutation—for example, BXP-003 changes train numbers,
station code and times, while BXP-009 devolves into repeated component tokens.
These examples demonstrate that parser repair alone would not be sufficient.

The exact-numeric-retention columns in [per_case.csv](per_case.csv) are only a
diagnostic for manual review. They are not a fidelity score: layout numbers,
citation markers and formatting can create false matches, and invalid graphs
prevent reliable semantic comparison.

## Per-case results

| Case | Domain | Stop | Out tokens | Decode tok/s | Latency s | Scorer failure | AAR failure |
|---|---|---|---:|---:|---:|---|---|
| [BXP-001](native/BXP-001/output.express) | Weather | closing_sentinel | 232 | 44.02 | 6.661 | unterminated expression | unclosed expression |
| [BXP-002](native/BXP-002/output.express) | Air quality | closing_sentinel | 240 | 52.42 | 5.976 | invalid statement | unknown property |
| [BXP-003](native/BXP-003/output.express) | Rail travel | max_new_tokens | 2051 | 26.21 | 80.586 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-004](native/BXP-004/output.express) | Airline baggage | max_new_tokens | 2050 | 24.49 | 87.180 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-005](native/BXP-005/output.express) | Urban transport | closing_sentinel | 780 | 17.73 | 47.766 | children type | invalid gap enum |
| [BXP-006](native/BXP-006/output.express) | Road trips | max_new_tokens | 2048 | 22.61 | 94.915 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-007](native/BXP-007/output.express) | Sightseeing | eos_or_native_stop | 684 | 18.73 | 40.314 | unterminated expression | unclosed expression |
| [BXP-008](native/BXP-008/output.express) | Restaurants | closing_sentinel | 246 | 18.40 | 17.434 | unknown property | unknown property |
| [BXP-009](native/BXP-009/output.express) | Accommodation | max_new_tokens | 2048 | 26.08 | 82.367 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-010](native/BXP-010/output.express) | Consumer audio | closing_sentinel | 355 | 16.44 | 25.300 | children type | children type |
| [BXP-011](native/BXP-011/output.express) | Smartphones | eos_or_native_stop | 583 | 16.52 | 39.186 | unterminated expression | unclosed expression |
| [BXP-012](native/BXP-012/output.express) | Productivity software | closing_sentinel | 348 | 17.60 | 23.828 | unbalanced delimiter | unbalanced delimiter |
| [BXP-013](native/BXP-013/output.express) | Bank deposits | max_new_tokens | 2049 | 24.31 | 88.765 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-014](native/BXP-014/output.express) | Foreign exchange | closing_sentinel | 491 | 18.08 | 30.813 | unterminated expression | unclosed expression |
| [BXP-015](native/BXP-015/output.express) | Cricket | eos_or_native_stop | 404 | 19.12 | 24.645 | unterminated expression | unclosed expression |
| [BXP-016](native/BXP-016/output.express) | Streaming entertainment | max_new_tokens | 2049 | 20.09 | 105.486 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-017](native/BXP-017/output.express) | Books | eos_or_native_stop | 180 | 16.67 | 14.989 | unterminated expression | unclosed expression |
| [BXP-018](native/BXP-018/output.express) | Music charts | max_new_tokens | 2051 | 19.71 | 108.553 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-019](native/BXP-019/output.express) | Video games | max_new_tokens | 2048 | 19.21 | 111.301 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-020](native/BXP-020/output.express) | Space missions | closing_sentinel | 462 | 14.98 | 35.008 | envelope count/incomplete | unsupported prefix |
| [BXP-021](native/BXP-021/output.express) | Archaeology | closing_sentinel | 479 | 14.43 | 38.673 | empty/invalid children | empty/invalid children |
| [BXP-022](native/BXP-022/output.express) | Gardening | closing_sentinel | 479 | 14.24 | 39.400 | envelope count/incomplete | unbalanced delimiter |
| [BXP-023](native/BXP-023/output.express) | Pet care | max_new_tokens | 2050 | 19.16 | 111.560 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-024](native/BXP-024/output.express) | Waste management | max_new_tokens | 2049 | 17.17 | 124.822 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-025](native/BXP-025/output.express) | Language learning | eos_or_native_stop | 479 | 19.44 | 29.113 | unterminated expression | unclosed expression |
| [BXP-026](native/BXP-026/output.express) | Artificial intelligence | max_new_tokens | 2049 | 21.21 | 101.679 | envelope count/incomplete | unsupported prefix |
| [BXP-027](native/BXP-027/output.express) | Cybersecurity | max_new_tokens | 2051 | 20.19 | 106.677 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-028](native/BXP-028/output.express) | Digital privacy | max_new_tokens | 2049 | 16.67 | 128.762 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-029](native/BXP-029/output.express) | Income tax | max_new_tokens | 2050 | 18.46 | 116.496 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-030](native/BXP-030/output.express) | Retirement savings | closing_sentinel | 594 | 16.64 | 42.237 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-031](native/BXP-031/output.express) | Property due diligence | max_new_tokens | 2049 | 16.97 | 127.381 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-032](native/BXP-032/output.express) | Health insurance | max_new_tokens | 2049 | 19.86 | 108.830 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-033](native/BXP-033/output.express) | Public health | max_new_tokens | 2048 | 13.55 | 157.215 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-034](native/BXP-034/output.express) | Nutrition | max_new_tokens | 2051 | 21.54 | 101.210 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-035](native/BXP-035/output.express) | Fitness | max_new_tokens | 2050 | 15.24 | 140.907 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-036](native/BXP-036/output.express) | Mental wellbeing | max_new_tokens | 2048 | 15.19 | 140.242 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-037](native/BXP-037/output.express) | School education | max_new_tokens | 2048 | 18.75 | 114.662 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-038](native/BXP-038/output.express) | Careers | closing_sentinel | 63 | 10.52 | 9.862 | unbalanced delimiter | unbalanced delimiter |
| [BXP-039](native/BXP-039/output.express) | Small-business compliance | max_new_tokens | 2050 | 19.82 | 109.416 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-040](native/BXP-040/output.express) | Consumer rights | eos_or_native_stop | 1375 | 21.08 | 70.645 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-041](native/BXP-041/output.express) | International travel rules | eos_or_native_stop | 1009 | 14.66 | 74.977 | unterminated expression | unclosed expression |
| [BXP-042](native/BXP-042/output.express) | Electric vehicles | max_new_tokens | 2051 | 20.11 | 107.408 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-043](native/BXP-043/output.express) | Home solar energy | max_new_tokens | 2049 | 16.88 | 127.875 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-044](native/BXP-044/output.express) | Home appliances | max_new_tokens | 2048 | 17.95 | 120.088 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-045](native/BXP-045/output.express) | Urban climate | max_new_tokens | 2048 | 15.69 | 136.915 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-046](native/BXP-046/output.express) | Biotechnology | max_new_tokens | 2050 | 16.74 | 128.911 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-047](native/BXP-047/output.express) | Battery technology | eos_or_native_stop | 729 | 14.44 | 55.859 | unterminated expression | unclosed expression |
| [BXP-048](native/BXP-048/output.express) | History and heritage | max_new_tokens | 2049 | 18.78 | 115.559 | envelope count/incomplete | incomplete/trailing envelope |
| [BXP-049](native/BXP-049/output.express) | Traditional arts | closing_sentinel | 1095 | 12.76 | 93.119 | unbalanced delimiter | unbalanced delimiter |
| [BXP-050](native/BXP-050/output.express) | Ocean and monsoon science | max_new_tokens | 2049 | 13.35 | 158.769 | envelope count/incomplete | incomplete/trailing envelope |

Each domain occurs once, so a per-domain aggregate would be the same as its
single case and must not be interpreted as a domain-level estimate.

## Reproduction and provenance

- Model SHA-256: `4675f37353e41c786a3e94f03ba4f64ad366a2e4e5d188bb92f5bef3922a75fe`;
  host and staged-device hashes matched before inference.
- Device: SM-F776U / Flip8, Android API 37, SoC SM8850. Battery changed from
  100% to 90% on USB; temperature changed from 33.5°C to 36.5°C. Power and
  thermal settings were not modified.
- LiteRT-LM Android 0.16.1, GenUICraft AAR 0.2.0; deterministic sampler
  temperature 0, top-k 1, top-p 1, seed 42; thinking disabled.
- Full mobile A2UI Express shared prompt plus one travel-checklist example;
  original response text is the final user message. Embedded model template,
  no override. All 50 runtime-rendered prompt hashes matched independently
  prepared requests before generation.
- One GPU engine, fresh conversation per case, fixed BXP-001…BXP-050 order,
  label-specific cache directory, 300-second per-case timeout.
- Source commit before this report: `79bcec38bb632dacab97de5ebeb262d039e183c3`.
  Request SHA-256: `0dd5c774eca49b84b4ef2b6a3b9da53ea8d9573cb99b9515e0b3b4d79e37ec4a`.
- The package lacks its original training manifest/checkpoint. Package and
  runtime behavior are verified; original training lineage and checkpoint-to-
  LiteRT numerical parity are not established.

## Reproduction

Stage [requests.jsonl](native/requests.jsonl) at the device request path and
use a new label if repeating the run:

```powershell
adb -s R3GL203AKSF shell am instrument -w -r `
  -e class com.samsung.genuicraft.OfficialMobileNativeQualityProbeTest `
  -e modelPath /sdcard/Android/data/com.samsung.genuicraft/files/sdk_models/gemma4_e2b_a2ui_mobile.litertlm `
  -e requestPath /sdcard/Android/data/com.samsung.genuicraft/files/sdk_benchmark/mobile_full50_20260921/requests_all50_mtp.jsonl `
  -e label YOUR_NEW_LABEL -e caseTimeoutSeconds 300 -e mtp true `
  com.samsung.genuicraft.test/androidx.test.runner.AndroidJUnitRunner
```

The harness refuses to overwrite an existing report label. Native test success
means all inference calls completed; it does not imply Express validity. Device
compiler replay uses `GenUiSdkBixby50Test#replaySavedBixbyCorpus` with
`replayMode=express`, the exact saved output bytes and a new `runId`.

Detailed artifacts: [machine summary](summary.json), [environment](environment.json),
[per-case CSV](per_case.csv),
[error counts](error_counts.json), [native manifest](native/native_results.manifest.json),
[raw native results](native/native_results.jsonl),
[scored predictions](scoring/scored_predictions.jsonl),
[aggregate v5.4 metrics](scoring/aggregate_metrics.json), and
[AAR replay results](renderer_replay/replay_results.json). Artifact identities
are bound by [checksums.json](checksums.json).
