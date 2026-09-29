# FP32 / corrected FP16 sample quality

V5.4 is a source-to-UI representation score (0–100), not factual accuracy or human visual quality. Scores use the saved raw output and the exact SDK-repaired Express without edits. Unavailable pipeline artifacts count as zero. These three selected cases do not establish Bixby50 accuracy.

| Precision | MTP | Measured outputs | Raw mean | After repair mean | Fallbacks |
|---|---|---:|---:|---:|---:|
| FP16_CORRECTED | On | 3 | 67.05 | 93.18 | 0 |
| FP32 | On | 3 | 43.34 | 93.22 | 0 |

| Run | Case | Raw score | After repair score |
|---|---|---:|---:|
| r01_fp32_mtp_on | BXP-001 | 90.03 | 90.03 |
| r01_fp32_mtp_on | BXP-003 | 0.00 | 99.01 |
| r01_fp32_mtp_on | BXP-004 | 40.00 | 90.64 |
| r01_fp16_corrected_mtp_on | BXP-001 | 91.15 | 91.15 |
| r01_fp16_corrected_mtp_on | BXP-003 | 40.00 | 97.75 |
| r01_fp16_corrected_mtp_on | BXP-004 | 70.00 | 90.64 |
