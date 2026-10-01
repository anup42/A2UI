# Bixby50 estimates with 2.1-second prefill

Original FP32 checkpoint reference score: **85.90/100**, supplied by the user. Prefill is fixed at **2.1 seconds for both modes**. Decode speeds are the supplied **35/52 tokens/s**.

| Metric | MTP off | MTP on |
| --- | ---: | ---: |
| Score /100 | 83.26 | 80.89 |
| Decode speed (tokens/s) | 35.00 | 52.00 |
| Average output tokens | 615.42 | 568.70 |
| Decode time (s) | 17.58 | 10.94 |
| Assumed prefill time (s) | 2.10 | 2.10 |
| Other conversion/repair and recording overhead (s) | 0.16 | 0.15 |
| Total estimated latency (s) | 19.84 | 13.19 |

Estimated latency with the model already loaded = average generated tokens / supplied decode speed + 2.1 seconds + recorded remaining conversion overhead.

Average model input is **3,468.62 tokens**, including the prompt. These all-50 output averages include repetition-stopped attempts. Estimates exclude Perplexity generation and UI rendering. Other overhead is retained from the previous mixed-device measurements.

Excluding repetition-stopped attempts, warm conversion estimates are **21.58 seconds off** and **14.55 seconds on**.

[Summary JSON](summary.json)
