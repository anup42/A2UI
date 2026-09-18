# Gemma prompt screening and confirmation plan

Date: 18 September 2026

This plan is based on read-only inspection of `android/app/src/main/assets/genuicraft_bixby50.jsonl` and the completed historical run under `tmp/genuicraft_20260918/gemma50_v7_gpu_mtp`. It defines subsets and acceptance checks only. No generation, build, installation, or device action was performed while preparing these files.

## Evidence and fixed controls

The historical run completed 50/50 conversions, with 47 first-attempt successes and three successes after one repair. Its configuration records Gemma, GPU, MTP enabled, thinking enabled, temperature 0, and one repair. The three rejected first attempts show distinct structural failures:

| Case | Observed first-attempt defect | Why it matters |
| --- | --- | --- |
| `BXP-030` | Emitted all six assignments, then closed with `</a2>` instead of `</a2ui>`. | Exact closing-tag and whole-document completion test. |
| `BXP-032` | Root named `a` through `k`, but the assignments skipped `j`, shifted the following binding to `k`, and invented `l`. | Long ordered-block test where table bindings make source suffixes diverge from element names. |
| `BXP-037` | Began `<a2a=Text(...)`, losing the exact opening tag and root line. | Exact opening-tag/root-line completeness test. |

`BXP-003` passed the historical v7 first attempt and selected schedule cards for a train-service entity table. It is the positive control for exact binding use, table/list coexistence, and a useful domain route.

Keep all execution controls fixed for every comparison: the same model and binaries, GPU backend, MTP enabled, thinking enabled with the same 1,024-token budget, temperature `0.0`, output-token limit, case timeout, `sourceBindings=true`, renderer, corpus, and device conditions. Change only the prompt text. Record the prompt SHA-256 and exact launch arguments before each run. Do not compare repaired historical latency with one-attempt candidate latency.

Pinned inputs and candidate prompts at design time:

| Artifact | SHA-256 |
| --- | --- |
| Bixby50 fixture | `FC46AA381957BED206F9E0F53E28EDBB21B5096CA093985AD184FDA73FAAEA0A` |
| Historical `results.json` | `D1B03757462360DB924ED14135617482C573486082AA0111A554993C0C8289B3` |
| Production `gemma.txt` baseline | `154FF4D27D8E37B81279E7979C18B4FD90BBDD5135A4E124917712A85438535B` |
| `minimal.txt` | `FD00E896AD964648D1CFCF5C41F693507362A071E85AA48846BB7A9BEE5A3FBE` |
| `completeness.txt` | `E708CF52E3E2D2236121D9F09E66F3151F4C2ED54CF12D87E842A38419499D28` |
| `mixed_examples.txt` | `BCCDB467713197038E75F47DE712ED054111C36CEBB89F997B077DEE6D815630` |

## Phase 1: four-case screen

Run each frozen prompt with `repairs=0` on this corpus-filtered set:

1. `BXP-003` — passing schedule-card/table/list control.
2. `BXP-030` — historical malformed closing tag.
3. `BXP-032` — historical skipped/shifted element on 11 ordered blocks.
4. `BXP-037` — historical malformed opening tag and missing root.

The screen is intentionally failure-focused development data. Use the same case set and corpus order for the production-prompt baseline, each candidate, and a final repeated baseline. Preserve every raw attempt, including invalid or truncated output.

A candidate advances only if all four first attempts:

- start with literal `<a2ui>` and end with literal `</a2ui>`;
- copy the supplied root exactly;
- emit exactly one assignment for every block, with identical element names and order;
- use every supplied binding exactly once in its typed block, without aligning element names to binding suffixes;
- map every heading to `Text(...,variant="heading")`;
- keep table columns and rows together and make a defensible domain/presentation choice;
- compile, pass source/content integrity checks, and complete the renderer smoke check without fallback.

Correctness is the gate. Use comparable first-attempt latency only to rank candidates that meet every structural and content requirement.

## Phase 2: twelve-case confirmation subset

Freeze this subset before inspecting candidate results:

| Case | Coverage role |
| --- | --- |
| `BXP-001` | Weather entity table expected to exercise card presentation; paragraph plus list. |
| `BXP-006` | Long table cells, comparison judgment, paragraphs, and lists. |
| `BXP-008` | Compact generic table with heading and trailing plain text. |
| `BXP-013` | Numeric content, feature table, three lists, and 11 ordered blocks. |
| `BXP-017` | Apology/caveat preservation using only headings and plain text. |
| `BXP-025` | Schedule-shaped seven-day entity table plus two lists. |
| `BXP-027` | Code-like email example represented by plain text, four lists, and 14 ordered blocks. |
| `BXP-033` | Long plain-text-heavy answer with 14 ordered blocks and no table/list shortcut. |
| `BXP-035` | List-heavy long answer with 16 ordered blocks. |
| `BXP-038` | Minimal single-paragraph answer with no Markdown structure. |
| `BXP-045` | Longest block sequence in the corpus: 19 blocks, many headings/plain paragraphs, and a final table. |
| `BXP-047` | Long feature-comparison matrix with dense cells and trailing plain sections. |

This set covers card-presentation tables, ordinary and comparison tables, plain text, lists, short and long outputs, sparse and dense block graphs, and element/binding sequences that diverge after a table. Run it first with `repairs=0` to measure strict first-pass behavior. If production behavior includes one repair, separately run the surviving prompt with the unchanged production repair budget and report first-pass and repaired success independently.

The Bixby50 fixture contains no fenced code block and no Markdown horizontal rule. `BXP-027` contains a harmless example email, but the current source parser correctly represents it as `Text`, not `CodeBlock`; it is therefore only a code-like content case. The prompts retain the exact `CodeBlock` and `Divider` mappings, but empirical coverage of those two block kinds requires a separate fixed synthetic source-binding fixture. Do not mislabel Bixby50 as exercising them.

## Phase 3: final full-50 run

After selecting and freezing one prompt, run every case `BXP-001` through `BXP-050` with the unchanged production settings and repair budget. The final run includes all four screening cases and all twelve confirmation cases; no tuned, difficult, or previously failing case is excluded. Report it as a regression result on a development-informed corpus, not as an unseen holdout estimate.

Report at minimum: total terminal results, first-attempt successes, repaired successes, failures by syntax/binding/content/runtime/render category, fallback count, exact prompt/model/app/corpus hashes, per-case attempts and elapsed time, table domain/presentation choices, and renderer smoke status. A screenshot or visible-text count is smoke evidence only; inspect preservation of all source blocks, table dimensions/order, headings, lists, caveats, citations, and final long-answer sections separately.
