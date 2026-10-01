# LiteRT-LM Bixby50 v5.4 / MTP report

**Partial coverage: MTP off 15/50; MTP on 12/50. Full-cohort native scores are not measured.** This report uses the saved corrected R64 W4 / GPU FP32 runs on SM-F776U, temperature 0 and 2,048 output-token limit. No new inference was performed.

## Identical 12-case comparison

| Model / runtime | Same cases | Raw v5.4 mean /100 | Strict valid | Zero-score cases |
| --- | --- | --- | --- | --- |
| Checkpoint LoRA R32 | 12 | 52.75 | 9/12 | 3 |
| Checkpoint LoRA R64 | 12 | 43.10 | 8/12 | 4 |
| LiteRT W4 / GPU FP32 / MTP off | 12 | 58.87 | 10/12 | 2 |
| LiteRT W4 / GPU FP32 / MTP on | 12 | 46.30 | 8/12 | 4 |

MTP-on minus off: -12.57 points. MTP on wins 1, off wins 5, ties 6.

The score measures source-to-UI representation quality, not factual correctness. Missing cases are N/A. Repairs are excluded from the primary score. These results do not isolate export or quantization loss.

| Case | MTP-off raw | Off strict | MTP-on raw | On strict |
| --- | --- | --- | --- | --- |
| BXP-001 | 91.15 | True | 90.03 | True |
| BXP-002 | 40.00 | True | 40.00 | True |
| BXP-003 | 99.01 | True | 0.00 | False |
| BXP-004 | 70.00 | True | 40.00 | True |
| BXP-005 | 40.00 | True | 40.00 | True |
| BXP-006 | 0.00 | False | 89.26 | True |
| BXP-007 | 70.00 | True | 70.00 | True |
| BXP-008 | 0.00 | False | 0.00 | False |
| BXP-009 | 40.00 | True | 0.00 | False |
| BXP-010 | 95.73 | True | 95.73 | True |
| BXP-011 | 90.61 | True | 90.61 | True |
| BXP-012 | 70.00 | True | 0.00 | False |
| BXP-013 | 0.00 | False | N/A | N/A |
| BXP-014 | 91.62 | True | N/A | N/A |
| BXP-015 | 98.51 | True | N/A | N/A |

## Separate SDK repair result

| SDK output population | Cases | Raw mean | After SDK repair mean | Strict valid after repair |
| --- | --- | --- | --- | --- |
| MTP-off output after SDK 0.5.6 repair | 15 | 59.77 | 93.48 | 15/15 |
| MTP-on output after SDK 0.5.6 repair | 12 | 46.30 | 94.20 | 12/12 |

Full methods, provenance, missing-case lists and error details: [REPORT.md](REPORT.md), [validation.json](validation.json), [interactive report](index.html).
