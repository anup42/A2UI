# Bixby50 v5.4 quality findings

R32 scores 5.43 points above R64 on the full supplied 50-case cohort. Both have large structural losses. This is an evaluation of the supplied predictions, not a new CUDA inference run or a checkpoint-selection recommendation.

## Raw structural quality

| Population | Cases | Strict invalid / zero | Strict valid but disconnected | Strict valid and fully connected |
| --- | --- | --- | --- | --- |
| Checkpoint LoRA R32 | 50 | 20 | 19 | 11 |
| Checkpoint LoRA R64 | 50 | 21 | 23 | 6 |
| LiteRT W4 / GPU FP32 / MTP off | 15 | 3 | 6 | 6 |
| LiteRT W4 / GPU FP32 / MTP on | 12 | 4 | 4 | 4 |

A defined component is only usable if the root can reach it through child references. The scorer penalizes a program that contains correct words in unattached components: reachability below 90% caps quality at 40/100; partial reachability otherwise caps it at 70/100. Invalid raw programs get zero under this scorer, including dangling child references. These caps explain many repeated 0, 40 and 70 scores; they are not arbitrary rounding.

## Primary strict-validation failures

Each invalid output is assigned one category from the first official validation error. Counts are diagnostic, not separate accuracy scores.

| Category | R32 /50 | R64 /50 | MTP off /15 | MTP on /12 |
| --- | --- | --- | --- | --- |
| Duplicate component ID | 0 | 2 | 0 | 0 |
| Expression syntax or delimiter | 2 | 1 | 0 | 0 |
| Extraneous Hungary prefix outside the envelope | 2 | 1 | 0 | 0 |
| Incomplete envelope | 0 | 1 | 0 | 1 |
| Missing component reference | 3 | 9 | 3 | 3 |
| Output-token limit and incomplete envelope | 13 | 7 | 0 | 0 |

R32 has 13 outputs that reach the recorded 2,048-token limit; R64 has 7. All of those outputs are invalid. Raising the budget would not directly fix the other malformed-reference, duplicate-ID or disconnected-graph cases. Among outputs that report stopping at the closing sentinel, 7 R32 and 14 R64 outputs are still strict-invalid.

## Concrete examples

- **BXP-002, AQI:** R32 scores 93.30; R64 and both native modes score 40.00. R64 defines 14 nodes but only 7 are reachable; the PM2.5 and outdoor-activity guidance nodes are disconnected. Both native modes define 10 nodes but only 5 are reachable, omitting the pollutant and guidance from the root's rendered content. Inspect the raw programs in `index.html#BXP-002`.
- **BXP-008, R32:** raw output begins with the unrelated text ` Hungary` before `<a2ui>`, violating the strict completion envelope. R64 returns only `</a2ui>` for this case. R32 BXP-012 and R64 BXP-012 also contain the stray prefix. The report does not strip it to inflate the raw score.
- **BXP-003, MTP comparison:** MTP off scores 99.01; MTP on scores 0 because element `j` references missing child `l`. This is a malformed graph despite the provider reporting `COMPLETED`.
- **BXP-006:** MTP off scores 0 because element `m` references missing `q`; MTP on scores 89.26. MTP does not worsen every case.
- **BXP-012, MTP on:** output contains an opening `<a2ui>` but no closing sentinel and scores 0. The saved provider label is `COMPLETED`, so that label cannot be used as proof of a complete DSL envelope or a correct EOS stop.

## Matched native repair result

| Same BXP-001–012 | Raw mean /100 | After SDK 0.5.6 /100 | Strict valid after repair |
| --- | --- | --- | --- |
| MTP off | 58.87 | 92.85 | 12/12 |
| MTP on | 46.30 | 94.20 | 12/12 |

Post-repair quality can reverse the raw ordering because repair reconnects or recovers different generated content. These repaired values are application-pipeline scores, not raw checkpoint/native model accuracy. They do not establish human visual quality or guarantee every source fact is preserved.

## What can be reported

Report the complete checkpoint means with n=50. For the quantized native comparison, report the matched raw 12-case means and explicitly label the run partial. The latest corrected native MTP-off sample scores above the corresponding R64 checkpoint subset, so these scores do not support a blanket claim that quantization reduced v5.4 quality. They also cannot certify export parity or extrapolate native quality to all 50 cases. The remaining native cases require a separately completed benchmark under the same recorded model/prompt/runtime conditions.
