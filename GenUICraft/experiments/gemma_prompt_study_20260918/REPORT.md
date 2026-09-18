# Gemma GPU + MTP prompt comparison

**Complete. Retain the accepted v10 production prompt and AAR. The scaffold candidate is rejected.**

Ten new strategies were tested: nine system-prompt variants and one input-scaffold strategy. Baseline controls are not counted as new strategies. Broader testing selected scaffold for a full live run, but it finished at 48/50 and does not replace the accepted baseline's 50/50 result.

| Full-corpus result | First attempt | Repaired | Final successes | Failures | Fallbacks |
|---|---:|---:|---:|---:|---:|
| Retained baseline, historical v7 live run / v10 delivery | 47 | 3 | 50/50 | 0 | 0 |
| Rejected v11 scaffold, new live run | 45 | 3 | 48/50 | 2 | 0 |

V11 repaired BXP-013/029/036. BXP-046 still violated source-block order after repair; BXP-050 still lacked a complete Express envelope. All 55 calls used GPU+MTP with thinking enabled. Its all-case median was 71.010 seconds, p95 170.081 seconds, and maximum sampled process PSS 4,630,948 KB. Different device conditions and the baseline control drift prevent a controlled speedup claim.

All 48 successful scaffold component graphs preserve their expected source structure. Their 22 tables preserve exact source cells; 10 use preferred presentations and 12 acceptable alternatives. Two failed cases have no accepted document. A full renderer replay of a known-rejected candidate was deliberately skipped at the user's request to finish promptly; accepted v10 renderer evidence remains in [VALIDATION.md](../../VALIDATION.md).

All 106 production source files were restored to their recorded v10 hashes. The rebuilt AAR is byte-identical to accepted v10: `7b33e20601a12f8090cec1f774ab2141bba35c3fea470353bdbf5995ec40c108`, 9,353,001 bytes. Its 285 JVM tests pass, and all 25 publication files match the Bixby copy. The scaffold implementation and its four tests are preserved under `candidate_source/`, outside production source sets. See [restoration evidence](results/accepted_v10_restoration.json).

| Prompt/run | First-pass successes / completed cases | Planned cases | Outcome |
|---|---:|---:|---|
| baseline (ps_baseline_start) | 1/4 | 4 | Complete screen |
| baseline_targeted (ps_screen_baseline_targeted) | 3/4 | 4 | Complete screen |
| compact (ps_screen_compact) | 0/4 | 4 | Complete screen |
| completeness (ps_screen_completeness) | 0/2 | 4 | Early rejected; remaining cases unfinished |
| example_first (ps_screen_example_first) | 2/4 | 4 | Complete screen |
| grammar (ps_screen_grammar) | 0/2 | 4 | Early rejected; remaining cases unfinished |
| mapping (ps_screen_mapping) | 1/3 | 4 | Early rejected; remaining cases unfinished |
| minimal (ps_screen_minimal) | 0/2 | 4 | Early rejected; remaining cases unfinished |
| mixed_examples (ps_screen_mixed_examples) | 1/3 | 4 | Early rejected; remaining cases unfinished |
| procedural (ps_screen_procedural) | 0/2 | 4 | Early rejected; remaining cases unfinished |
| baseline repeated (ps_baseline_end) | 1/4 | 4 | All four raw outputs byte-identical to initial control |
| scaffold (ps_screen_scaffold) | 4/4 | 4 | Complete screen; semantic graph and table audits pass |

All screen runs use the same installed v10 AAR consumer, Gemma model, GPU, MTP, thinking enabled with a 1,024-token budget, temperature 0, strict bindings/content validation, and zero repair attempts. The four-case screen deliberately contains all three historical first-attempt failures plus a passing train-card control; it is not a whole-corpus accuracy estimate.

The strongest new system-only prompt is `baseline_targeted`: 3/4 versus the fresh baseline's 1/4. Its remaining failure is a shortened table-row binding token. The scaffold input plus scaffold prompt passed 4/4. Broader testing exposed an additional copied-binding punctuation error, so the screen is provisional evidence only.

The initial and repeated controls produced identical raw text, but all-case median latency drifted from 70.362 to 145.752 seconds. These sustained-device timings do not establish a controlled speed improvement.

The scaffold's broader zero-repair confirmation completed **11/12**, for **15/16** across the disjoint screen and confirmation sets. BXP-013 added a full stop to `@source.k`; validation rejected it. All eleven successful component graphs match their scaffolds except for valid table selections. The six successful tables preserve their exact cells; three use preferred presentations and three acceptable alternatives. See [the screenshot review](results/visual_review_confirmation.md). Packaged synthetic code/divider/literal tests subsequently passed 3/3 on their first attempts; that did not predict full-corpus success.

The targeted system-only prompt completed **5/6** broader cases, then was deliberately stopped after BXP-025 shortened two bindings. Its known failures in BXP-030 and BXP-025 mean it could score at most 14/16 even if all remaining cases passed. Those six unfinished cases are not counted as failures or successes. Scaffold was selected for full packaged-AAR validation and then rejected on that result.

The original production baseline previously completed all 50 cases with one allowed repair: 47 first-attempt successes and 3 repaired successes. That historical full run is separate from these zero-repair screens.

Raw attempts, generated documents, screenshots, temperatures, hashes, run configs, and failure messages are retained under `../../../tmp/genuicraft_prompt_study_20260918/`. The portable `evidence/` bundle reconciles 16 runs, 109 completed cases, and 114 captured model calls; 16 planned cases were deliberately left unfinished by early rejection. `results/comparison.json` and `results/case_results.csv` preserve the counts. The separate three-call throughput replay below is not included in those prompt-study totals. See README.md for adaptive selection and the distinction between experimental and delivered code.

## Token throughput measurement

The conversion benchmark's `elapsedMs` includes the whole conversion request and any repair. It is not decode time. Its optional `outputTokens` field is absent because LiteRT-LM 0.15.0 defaults `ExperimentalFlags.enableBenchmark` to false, and the production provider does not enable it. The reflection getter itself matches the resolved API. No tokens-per-second result can be inferred from these records or from character counts.

A separate three-case profiling replay completed with the same model, native libraries and GPU/MTP/thinking settings. It replays captured **v11 candidate** input prompts, with native benchmark collection enabled before engine initialization. It does not establish the retained baseline prompt's end-to-end latency or conversion success.

| Case / engine state | Native decode tokens/sec | Native prefill tokens/sec | Time to first token | Generation elapsed |
|---|---:|---:|---:|---:|
| BXP-003 / first generation | 57.624 | 1,144.587 | 0.866 s | 22.659 s |
| BXP-030 / reused engine | 30.775 | 952.795 | 1.436 s | 57.355 s |
| BXP-047 / reused engine | 31.992 | 691.731 | 1.989 s | 62.781 s |

Observed native decode median: **31.992 tokens/sec**; range **30.775–57.624**. These are LiteRT-reported decode counters with reasoning enabled, not visible Express text throughput. One engine was initialized and reused for all three fresh conversations. The host-observed initialization call took 4.958 seconds; the separately reported native benchmark init counter is about 9.949 seconds and is retained without treating the two metrics as interchangeable. This is a small sustained-device sample, not a controlled MTP-versus-no-MTP speedup comparison or speculative-token acceptance-rate measurement. See [the throughput evidence](results/throughput/summary.json).

The restored reference app also passed all four focused device checks (host action callback, exact source URL, literal text, and horizontal table access), with zero model calls. The test log is preserved in the prompt-study results.
