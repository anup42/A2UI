# FP32 / corrected FP16 sample quality

V5.4 is a source-to-UI representation score (0–100), not factual accuracy or human visual quality. Scores use the saved raw output and the exact SDK-repaired Express without edits. Unavailable pipeline artifacts count as zero. These three selected cases do not establish Bixby50 accuracy.

| Precision | MTP | Measured outputs | Raw mean | After repair mean | Fallbacks |
|---|---|---:|---:|---:|---:|
| FP16_CORRECTED | Off | 3 | 29.96 | 84.42 | 0 |
| FP32 | Off | 3 | 86.72 | 93.60 | 0 |

| Run | Case | Raw score | After repair score |
|---|---|---:|---:|
| r01_fp32_mtp_off | BXP-001 | 91.15 | 91.15 |
| r01_fp32_mtp_off | BXP-003 | 99.01 | 99.01 |
| r01_fp32_mtp_off | BXP-004 | 70.00 | 90.64 |
| r01_fp16_corrected_mtp_off | BXP-001 | 89.89 | 92.41 |
| r01_fp16_corrected_mtp_off | BXP-003 | 0.00 | 98.05 |
| r01_fp16_corrected_mtp_off | BXP-004 | 0.00 | 62.81 |
