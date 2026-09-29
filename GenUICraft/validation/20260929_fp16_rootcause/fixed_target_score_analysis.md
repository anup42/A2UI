# BXP-003 fixed-target token-score comparison

These 6 completed GPU runs scored the same 1,696-byte target after the same prompt. All returned 570 finite token scores. The score arrays sum to the C API aggregate within 0.0005. The saved C API score is target log-likelihood: a larger value gives this exact continuation a higher score. The [pinned LiteRT-LM scoring path](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/engine/litert_lm_lib.cc) negates the returned score to obtain negative log-likelihood. This measures neither free-running accuracy nor the full vocabulary distribution.

## Whole-target log-likelihood

| Variant | Total log-likelihood | Mean log-likelihood / target token | Difference from original FP32 |
|---|---:|---:|---:|
| Original model · GPU FP16 (`original_fp16_score_003`) | -226.155121 | -0.396763 | -200.107357 |
| Original model · GPU FP32 reference (`original_fp32_score_003`) | -26.047764 | -0.045698 | +0.000000 |
| RoPE table · GPU FP16 (`rope_fp16_score_003`) | -40.630646 | -0.071282 | -14.582882 |
| RoPE table + GEMV float-accum shader patch · GPU FP16 (`rope_floataccum_score_003`) | -48.203625 | -0.084568 | -22.155861 |
| RoPE table + GEMV float-accum + fake-quant v2 shader patch · GPU FP16 (`rope_floataccum_fakequant_v2_score_003`) | -25.938889 | -0.045507 | +0.108875 |
| RoPE table + fake-quant v2 shader patch · GPU FP16 (`rope_fakequant_v2_score_003`) | -22.991875 | -0.040337 | +3.055889 |

Relative to the original FP32 score, the RoPE-only variant reduces the original FP16 aggregate log-score gap by 92.7%. The RoPE plus GEMV float-accum patch variant reduces it by 88.9%, yet scores 7.573 log-score units below RoPE-only. Changes to this one target's score need not track free-running generation quality or generalize to other targets.

## Position-level differences from original FP32

`negative deficit` is `max(0, FP32 token score − variant token score)`; gains in other positions are counted separately. Indices below are zero-based. The displayed IDs come from a separate C API tokenization of the target. Counts match 570, but the native scorer did not return token IDs, so exact ID-to-score alignment is unverified.

| Variant | Tokens with deficit ≥0.5 / ≥1 / ≥2 | Negative deficit sum | Positive gain sum | Top 10 share of deficit | Median / p95 absolute delta | Longest run ≥0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Original model · GPU FP16 | 71 / 48 / 30 | 208.313 | 8.205 | 40.3% | 0.004154 / 2.182 | 4 |
| RoPE table · GPU FP16 | 6 / 5 / 4 | 19.166 | 4.583 | 96.5% | 0.000009 / 0.057 | 2 |
| RoPE table + GEMV float-accum shader patch · GPU FP16 | 6 / 5 / 5 | 28.675 | 6.519 | 96.6% | 0.000009 / 0.052 | 2 |
| RoPE table + GEMV float-accum + fake-quant v2 shader patch · GPU FP16 | 2 / 1 / 0 | 3.624 | 3.732 | 92.0% | 0.000007 / 0.041 | 1 |
| RoPE table + fake-quant v2 shader patch · GPU FP16 | 1 / 0 / 0 | 2.094 | 5.150 | 83.0% | 0.000009 / 0.034 | 1 |

The original FP16 deficit is spread across 10/10 target blocks: 71 positions lose at least 0.5 log-score units. After the RoPE change, only 6 positions in 4/10 blocks cross that threshold, and the ten largest deficits account for 96.5% of the remaining deficit. The shader-patched run has 6 such positions in 5/10 blocks and a 0.000009 median absolute difference. The residual large deficits are concentrated in a few positions amid mostly small per-token differences.

RoPE table + GEMV float-accum + fake-quant v2 shader patch · GPU FP16 has 2 positions with a deficit ≥0.5 and a +0.109 whole-target log-score difference; RoPE table + fake-quant v2 shader patch · GPU FP16 has 1 position with a deficit ≥0.5 and a +3.056 whole-target log-score difference. Positive gains at other positions exceed the negative deficits for these captures. The variants are independent measurements; their scores do not form a monotonic progression or establish generation quality.

### Largest negative-deficit positions

Original model · GPU FP16 (`original_fp16_score_003`):

| Index | Standalone target ID* | FP32 score | Variant score | Variant − FP32 |
|---:|---:|---:|---:|---:|
| 450 | 236770 | -0.0000 | -11.5415 | -11.5415 |
| 73 | 19521 | -0.0000 | -11.0843 | -11.0843 |
| 250 | 10253 | 0.0000 | -10.7206 | -10.7206 |
| 551 | 236800 | -0.0187 | -10.2192 | -10.2004 |
| 133 | 17734 | -0.0000 | -7.8725 | -7.8725 |
| 345 | 5551 | -0.0006 | -7.2318 | -7.2311 |
| 64 | 236743 | -0.0000 | -6.8535 | -6.8535 |
| 123 | 236773 | -0.0000 | -6.7417 | -6.7417 |
| 86 | 236825 | -0.0000 | -6.4959 | -6.4959 |
| 25 | 236832 | -0.0000 | -5.1876 | -5.1876 |

RoPE table · GPU FP16 (`rope_fp16_score_003`):

| Index | Standalone target ID* | FP32 score | Variant score | Variant − FP32 |
|---:|---:|---:|---:|---:|
| 345 | 5551 | -0.0006 | -5.6391 | -5.6385 |
| 184 | 5551 | -0.0001 | -4.8506 | -4.8505 |
| 317 | 15455 | -0.0153 | -2.3401 | -2.3248 |
| 247 | 12430 | -0.0017 | -2.2079 | -2.2062 |
| 344 | 8743 | -0.4467 | -2.1848 | -1.7380 |
| 393 | 236761 | -0.5382 | -1.1093 | -0.5711 |
| 10 | 88933 | -1.1990 | -1.6817 | -0.4827 |
| 36 | 2289 | -0.9769 | -1.2725 | -0.2955 |
| 191 | 236753 | -0.9819 | -1.2180 | -0.2362 |
| 11 | 24845 | -0.0623 | -0.2052 | -0.1429 |

RoPE table + GEMV float-accum shader patch · GPU FP16 (`rope_floataccum_score_003`):

| Index | Standalone target ID* | FP32 score | Variant score | Variant − FP32 |
|---:|---:|---:|---:|---:|
| 250 | 10253 | 0.0000 | -7.8332 | -7.8332 |
| 184 | 5551 | -0.0001 | -5.9259 | -5.9258 |
| 345 | 5551 | -0.0006 | -5.2773 | -5.2766 |
| 517 | 4671 | -0.0000 | -4.7592 | -4.7591 |
| 344 | 8743 | -0.4467 | -2.6605 | -2.2138 |
| 69 | 1999 | -0.4271 | -1.1003 | -0.6732 |
| 193 | 236747 | -0.8464 | -1.1876 | -0.3412 |
| 393 | 236761 | -0.5382 | -0.8170 | -0.2788 |
| 36 | 2289 | -0.9769 | -1.2451 | -0.2682 |
| 550 | 236754 | -0.2704 | -0.4011 | -0.1307 |

RoPE table + GEMV float-accum + fake-quant v2 shader patch · GPU FP16 (`rope_floataccum_fakequant_v2_score_003`):

| Index | Standalone target ID* | FP32 score | Variant score | Variant − FP32 |
|---:|---:|---:|---:|---:|
| 554 | 236752 | -4.9516 | -6.2078 | -1.2563 |
| 193 | 236747 | -0.8464 | -1.7381 | -0.8918 |
| 9 | 236779 | -0.9769 | -1.3752 | -0.3984 |
| 561 | 4337 | -0.4161 | -0.5765 | -0.1604 |
| 6 | 9907 | -0.0446 | -0.2017 | -0.1571 |
| 339 | 236778 | -0.2685 | -0.4126 | -0.1441 |
| 550 | 236754 | -0.2704 | -0.3758 | -0.1054 |
| 332 | 236776 | -0.0994 | -0.1975 | -0.0982 |
| 11 | 24845 | -0.0623 | -0.1318 | -0.0695 |
| 336 | 236787 | -0.0173 | -0.0692 | -0.0519 |

RoPE table + fake-quant v2 shader patch · GPU FP16 (`rope_fakequant_v2_score_003`):

| Index | Standalone target ID* | FP32 score | Variant score | Variant − FP32 |
|---:|---:|---:|---:|---:|
| 550 | 236754 | -0.2704 | -0.7874 | -0.5170 |
| 193 | 236747 | -0.8464 | -1.2369 | -0.3905 |
| 194 | 1604 | -0.4889 | -0.7278 | -0.2389 |
| 11 | 24845 | -0.0623 | -0.2050 | -0.1427 |
| 6 | 9907 | -0.0446 | -0.1727 | -0.1281 |
| 339 | 236778 | -0.2685 | -0.3721 | -0.1036 |
| 322 | 20404 | -0.4272 | -0.4878 | -0.0607 |
| 332 | 236776 | -0.0994 | -0.1530 | -0.0537 |
| 516 | 84123 | -0.3790 | -0.4314 | -0.0524 |
| 336 | 236787 | -0.0173 | -0.0677 | -0.0504 |

*The token ID is from independent target tokenization and is not a captured scored-token ID.*

## Provenance and limits

- Target SHA-256: `1161435a54a7f852892f803e11b73d25b61e1a1c3794c8adf070431844df4e94`; target token IDs and prompt token IDs also match byte-for-byte across all 6 captures.
- All included runs use pinned native library SHA-256 `e9cbdddb0f1c693c549e1cde40bf90ad8aaa124d15944d0dd18faaf016dd6938`, probe SHA-256 `9caf142356acfbc15257910241b5086debb3ca0f469242f7f7b702363070a9dd`, GPU backend, 8,192 context tokens, MTP off, and one target. Original FP16/FP32 share model SHA-256 `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`; the RoPE-only and shader-patched runs share corrected-model SHA-256 `6a6d60e0003349047fb2000148e4c0d21232f869ff7dcde1a737ce943704e428`. Per-run model digests are in the JSON.
- The GEMV float-accum, combined, and fake-quant v2 runs requested distinct diagnostic OpenCL shader-patch directories; all three native logs record `CL_PATCH applied`. They share the same RoPE-model digest but are not unpatched model-only comparisons. Patch directory paths are saved per run in the JSON.
- The original FP32 run is a numerical comparison reference, not ground-truth probability or an accuracy label. These are one capture per variant, with no uncertainty estimate. Forced-target scores cannot locate a specific kernel, recover full logits, establish full-vocabulary KL, or predict raw DSL validity.
- Full per-index scores and signed deltas are in `fixed_target_score_analysis.json`. Recompute with `python GenUICraft/validation/20260929_fp16_rootcause/analyze_fixed_target_scores.py` from the repository root.
