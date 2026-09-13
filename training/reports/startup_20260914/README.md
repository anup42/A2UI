# Training startup performance and console logging

## Changes

- Immediate startup/stage messages; ten-second heartbeats while blocking steps
  are active; row and byte progress, rate, elapsed time, and ETA.
- Preparation output is mirrored to the console and `logs/prepare.log`.
  Configure, GPU preflight, training, and evaluation stdout/stderr are streamed
  live to the console and their existing log files. Carriage returns and
  incremental UTF-8 decoding are supported; child failures remain failures.
- Existing archive inputs stream from JSONL instead of loading both complete
  splits into memory. Malformed rows fail rather than silently disappearing.
- Spawned CPU workers perform strict filtering and target preparation in bounded
  batches. Parent-side ordering and global deduplication preserve membership,
  quarantine reasons, and output ordering. Shared prompt validation runs once
  per preparation worker instead of rebuilding the same contract for every row.
- Optional, enabled-by-default preparation reuse is bound to inputs, tokenizer
  assets, prompt, code, schemas, versions, token limits, and every cached output.
  Copies are independent; corrupt/missing entries rebuild. An unwritable optional
  cache index does not invalidate an otherwise verified preparation.
- GPU allocation, Golden32/35 membership, 2,048 generation-token defaults,
  evaluation cadence, and `/tensorboard` metric logging are unchanged.

## Bounded real-data check

Command, run on this Windows CPU host against the first 128 v9 training rows:

```powershell
python training/scripts/audits/benchmark_preparation_startup.py --input training/outputs/datasets/full_data_archive_recovered_v9/train.jsonl --rows 128 --workers 1 2 4
```

| CPU workers | Accepted rows | Seconds | Rows/second |
| --- | ---: | ---: | ---: |
| 1 | 128 | 20.086 | 6.373 |
| 2 | 128 | 14.568 | 8.786 |
| 4 | 128 | 9.829 | 13.023 |

All three runs produced the same full acceptance/quarantine/report digest:
`c157046f182a46d7c3dcc126f8e914ced94c07e220ed7c4fc20cbee0f53d302f`.
The parent schema was warmed; spawned worker startup is included in each timed
parallel run. Other local tests were active, so these are observed timings,
not isolated hardware benchmarks or stable throughput guarantees. No model or
tokenizer was loaded, no input was modified, and these results do not predict
full-corpus preparation time, H100 scaling, training speed, or Golden scores.

## Verification boundaries

Final frozen-code validation on 2026-09-14:

```text
python -m pytest training/tests -q --tb=short
661 passed in 262.47s
```

Critical Ruff checks and staged whitespace checks passed. The pinned Golden32
and Golden35 file hashes remained unchanged. An earlier in-progress test run
correctly rejected a preparation when implementation files were edited during
that preparation; the complete suite above was rerun after freezing the code.

Regression coverage includes serial/parallel byte parity, reserved-source and
duplicate handling, invalid wire properties, source/target bindings, token
overflow, complete Golden membership, live subprocess output before child exit,
child errors, heartbeat output, cache hit/miss and corruption, tokenizer drift,
path containment, and refusal to restart interrupted training automatically.

No real model training, GPU forward pass, inference, LiteRT conversion, or
quality evaluation ran on this PC. The first exact-model preparation still
validates the full archive; later reuse does not skip hardware preflight or
training-time tokenizer checks.

See the [updated startup instructions](../../docs/GOLDEN_E2E_QUICKSTART.md#startup-progress-cpu-preparation-and-reuse)
and [v9 host-transfer/runbook](../../docs/messages_archive_final_review.md).
