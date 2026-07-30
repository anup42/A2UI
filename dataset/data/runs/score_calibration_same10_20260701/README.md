# Score Calibration Same-10 Run

This run selects the same 10 response ids and, for each target score bucket, uses the existing real generated IR variant closest to that target score. No IR was manually degraded or rewritten.

Targets: 40, 50, 60, 70, 80, 90, 100

Important: the repository currently has no real samples near score 40; the 40 bucket therefore uses the nearest available real variants, mostly around 49-64.

| Target | Response Slot | Response ID | Actual | Source Run |
|---:|---:|---|---:|---|
| 40 | 1 | r_000012_01 | 49.1284 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 50 | 1 | r_000012_01 | 49.1284 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 60 | 1 | r_000012_01 | 68.814 | azure_gpt54_reasoning32_20260618_214045\gpt54_no_reasoning |
| 70 | 1 | r_000012_01 | 70.7561 | azure_gpt54_reasoning32_20260618_214045\gpt54_reasoning_medium |
| 80 | 1 | r_000012_01 | 82.1285 | dataset_gpt54_no_reasoning_p5_20260619_merged |
| 90 | 1 | r_000012_01 | 89.6249 | dataset_gpt54_no_reasoning_p5_20260619_part03 |
| 100 | 1 | r_000012_01 | 100.0 | dataset_gpt54_no_reasoning_20k_20260619 |
| 40 | 2 | r_000017_01 | 53.5208 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 50 | 2 | r_000017_01 | 53.5208 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 60 | 2 | r_000017_01 | 53.5208 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 70 | 2 | r_000017_01 | 75.914 | azure_gpt54_reasoning32_20260618_214045\gpt54_no_reasoning |
| 80 | 2 | r_000017_01 | 75.914 | azure_gpt54_reasoning32_20260618_214045\gpt54_no_reasoning |
| 90 | 2 | r_000017_01 | 89.3239 | dataset_gpt54_no_reasoning_20k_20260619 |
| 100 | 2 | r_000017_01 | 100.0 | dataset_gpt54_no_reasoning_p5_20260619_part02 |
| 40 | 3 | r_000013_01 | 56.3969 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 50 | 3 | r_000013_01 | 56.3969 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 60 | 3 | r_000013_01 | 56.3969 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 70 | 3 | r_000013_01 | 74.4723 | azure_gpt54_reasoning32_20260618_214045\gpt54_no_reasoning |
| 80 | 3 | r_000013_01 | 80.2524 | dataset_gpt54_no_reasoning_p5_20260619_part01 |
| 90 | 3 | r_000013_01 | 90.8173 | dataset_gpt54_no_reasoning_20k_20260619 |
| 100 | 3 | r_000013_01 | 100.0 | dataset_gpt54_no_reasoning_p5_20260619_merged |
| 40 | 4 | r_000167_01 | 56.8435 | dataset_gpt54_no_reasoning_p5_20260619_part02 |
| 50 | 4 | r_000167_01 | 56.8435 | dataset_gpt54_no_reasoning_p5_20260619_part02 |
| 60 | 4 | r_000167_01 | 56.8435 | dataset_gpt54_no_reasoning_p5_20260619_part02 |
| 70 | 4 | r_000167_01 | 68.682 | dataset_gpt54_no_reasoning_p5_20260619_part03 |
| 80 | 4 | r_000167_01 | 84.0173 | dataset_gpt54_no_reasoning_20k_20260619 |
| 90 | 4 | r_000167_01 | 91.3644 | dataset_gpt54_no_reasoning_p5_20260619_part01 |
| 100 | 4 | r_000167_01 | 100.0 | dataset_gpt54_no_reasoning_p5_20260619_part04 |
| 40 | 5 | r_000175_01 | 59.7747 | dataset_gpt54_no_reasoning_p5_20260619_part03 |
| 50 | 5 | r_000175_01 | 59.7747 | dataset_gpt54_no_reasoning_p5_20260619_part03 |
| 60 | 5 | r_000175_01 | 59.7747 | dataset_gpt54_no_reasoning_p5_20260619_part03 |
| 70 | 5 | r_000175_01 | 71.2479 | dataset_gpt54_no_reasoning_p5_20260619_part04 |
| 80 | 5 | r_000175_01 | 77.4494 | dataset_gpt54_no_reasoning_p5_20260619_merged |
| 90 | 5 | r_000175_01 | 100.0 | dataset_gpt54_no_reasoning_p5_20260619_part02 |
| 100 | 5 | r_000175_01 | 100.0 | dataset_gpt54_no_reasoning_p5_20260619_part02 |
| 40 | 6 | r_000016_01 | 60.5917 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_no_reasoning |
| 50 | 6 | r_000016_01 | 60.5917 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_no_reasoning |
| 60 | 6 | r_000016_01 | 60.5917 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_no_reasoning |
| 70 | 6 | r_000016_01 | 74.5814 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 80 | 6 | r_000016_01 | 75.2811 | dataset_gpt54_no_reasoning_p5_20260619_part04 |
| 90 | 6 | r_000016_01 | 90.782 | azure_gpt54_reasoning32_20260618_214045\gpt54_no_reasoning |
| 100 | 6 | r_000016_01 | 100.0 | dataset_gpt54_no_reasoning_p5_20260619_part05 |
| 40 | 7 | r_000150_01 | 60.9103 | dataset_gpt54_no_reasoning_p5_20260619_part04 |
| 50 | 7 | r_000150_01 | 60.9103 | dataset_gpt54_no_reasoning_p5_20260619_part04 |
| 60 | 7 | r_000150_01 | 60.9103 | dataset_gpt54_no_reasoning_p5_20260619_part04 |
| 70 | 7 | r_000150_01 | 66.9146 | dataset_gpt54_no_reasoning_p5_20260619_part02 |
| 80 | 7 | r_000150_01 | 82.1832 | dataset_gpt54_no_reasoning_p5_20260619_merged |
| 90 | 7 | r_000150_01 | 82.1832 | dataset_gpt54_no_reasoning_p5_20260619_merged |
| 100 | 7 | r_000150_01 | 100.0 | dataset_gpt54_no_reasoning_p5_20260619_part03 |
| 40 | 8 | r_000024_01 | 61.0979 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_no_reasoning |
| 50 | 8 | r_000024_01 | 61.0979 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_no_reasoning |
| 60 | 8 | r_000024_01 | 61.0979 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_no_reasoning |
| 70 | 8 | r_000024_01 | 70.4731 | azure_gpt54_reasoning32_20260618_214045\gpt54_mini_reasoning_medium |
| 80 | 8 | r_000024_01 | 81.6491 | azure_gpt54_reasoning32_20260618_214045\gpt54_no_reasoning |
| 90 | 8 | r_000024_01 | 88.975 | dataset_gpt54_no_reasoning_p5_20260619_part02 |
| 100 | 8 | r_000024_01 | 100.0 | dataset_gpt54_no_reasoning_20k_20260619 |
| 40 | 9 | r_000220_01 | 62.2144 | dataset_gpt54_no_reasoning_p5_20260619_merged |
| 50 | 9 | r_000220_01 | 62.2144 | dataset_gpt54_no_reasoning_p5_20260619_merged |
| 60 | 9 | r_000220_01 | 62.2144 | dataset_gpt54_no_reasoning_p5_20260619_merged |
| 70 | 9 | r_000220_01 | 71.3213 | dataset_gpt54_no_reasoning_p5_20260619_part01 |
| 80 | 9 | r_000220_01 | 83.3162 | dataset_gpt54_no_reasoning_p5_20260619_part03 |
| 90 | 9 | r_000220_01 | 83.3162 | dataset_gpt54_no_reasoning_p5_20260619_part03 |
| 100 | 9 | r_000220_01 | 100.0 | dataset_gpt54_no_reasoning_p5_20260619_part05 |
| 40 | 10 | r_000213_01 | 62.2144 | dataset_gpt54_no_reasoning_p5_20260619_part01 |
| 50 | 10 | r_000213_01 | 62.2144 | dataset_gpt54_no_reasoning_p5_20260619_part01 |
| 60 | 10 | r_000213_01 | 62.2144 | dataset_gpt54_no_reasoning_p5_20260619_part01 |
| 70 | 10 | r_000213_01 | 63.4921 | dataset_gpt54_no_reasoning_p5_20260619_merged |
| 80 | 10 | r_000213_01 | 84.9946 | dataset_gpt54_no_reasoning_p5_20260619_part03 |
| 90 | 10 | r_000213_01 | 88.8496 | dataset_gpt54_no_reasoning_p5_20260619_part05 |
| 100 | 10 | r_000213_01 | 100.0 | dataset_gpt54_no_reasoning_p5_20260619_part02 |