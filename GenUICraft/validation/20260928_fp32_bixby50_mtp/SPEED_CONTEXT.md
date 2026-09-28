# Why the measured speed changed

This comparison uses the rank-64 QAT-compatible W4 package on GPU with **FP32
arithmetic**. W4 still describes the quantized weights; FP32 describes the GPU
execution precision. The app update did not replace this with an unquantized
model or switch the target model to CPU.

## Precision and device conditions

The earlier [controlled precision experiment](../20260928_r64_gpu_controls/REPORT.md)
identified GPU FP16 execution as the failure condition on the two tested cases.
FP32 restored the weather and train numbers using the same quantized weights.
This is why the app now requests FP32 for the trained profile.

The pinned native runtime maps FLOAT32 activation settings to GPU FP32
compilation; other settings select FP16 ([LiteRT-LM 0.16.1 source](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/executor/llm_executor_settings_utils.cc#L74)).
That changes arithmetic and may cost throughput. These measurements do not
isolate a universal FP16-to-FP32 speed ratio: even the older FP16 CPU-sampler
control slowed from 26.47 to 13.45 decode tokens/s on the same train output.

The current benchmark directly records another contributor. Before its first
batch, the GPU maximum-clock file reported 1.3 GHz. During the first batches it
reported caps of 342 MHz, then values down to 160 MHz, while Android reported
thermal status 1 (light) and then 2 (moderate). Busy-counter samples show the GPU
working near those caps. This is evidence of constrained sustained device
performance; it does not identify which vendor power/thermal policy selected
each clock. See [telemetry.jsonl](telemetry.jsonl) for the complete timeline.

Alternating the modes across five-case groups reduces simple ordering bias, but
does not make temperatures, frequencies, or output lengths identical. Reported
speedups describe this run on this device. They are not peak hardware rates or
an isolated estimate of the arithmetic-precision cost.

## Decode speed versus the user's wait

Native decode tokens/s measures the output-generation portion. The provider's
wall time also includes prompt prefill, engine initialization when needed, and
request overhead. Conversion time additionally includes validation/recovery.
Screenshot waits and capture time are recorded separately in case elapsed time
and are not model-generation latency.

For a request with native counters:

- Estimated native decode time = output tokens / decode tokens per second.
- Estimated native prefill time = input tokens / prefill tokens per second.
- Engine initialization is reported as measured wall time when the request
  creates an engine; warm requests reuse it.
- Residual provider time = provider wall time minus those measured/derived
  phases. It includes uninstrumented overhead and counter-boundary differences.
- Time to first token is a separate overlapping observation, not another phase
  to add to prefill and decode. Native initialization phase totals can overlap
  and must not be added as independent wall time either.

The benchmark starts a fresh engine for every five-case batch. The run stopped
at the user's request: the 12 matched pairs contain three cold batch heads and
nine warm requests per mode. Three further completed off-mode cases are
unpaired and excluded from that comparison.

## What MTP measures

MTP drafts multiple tokens and has the target model verify them. It is intended
to preserve target-model quality while accelerating decoding ([Google Gemma
MTP documentation](https://ai.google.dev/gemma/docs/mtp/overview)). An observed
raw-output difference in this runtime is therefore something to measure and
investigate, not proof that speculative decoding is inherently less accurate.

Drafter acceptance comes from LiteRT-LM's native engine-session statistic after
each five-case batch. An acceptance percentage is not a tokens/s speedup. The
runtime exposes a ratio but not the accepted/drafted counts through this app,
so a mean across batches must be labeled unweighted; it cannot be represented
as the exact pooled acceptance rate for all 50 cases.

## How to read quality and rendering counts

Raw Android compilation, independent Python validation/scoring, generated-DSL
recovery, renderer-contract acceptance, and screenshot confirmation are distinct
checks. None alone proves preservation of the source answer.

In particular, the current SDK's literal-oriented source-integrity diagnostics
can over-report missing facts stored in a reachable table's `statePath` rows.
Warnings remain visible as diagnostics, but their count is **not** an accuracy
rate. The comparison must also inspect reachable content and the state-aware
scorer. It does not independently verify the real-world truth of the frozen
Bixby source answers.

Some early screen captures failed the test's case-label accessibility gate even
though the saved A2UI passed the renderer contract. Such cases are unconfirmed
captures until inspected separately; they are not automatically blank renders.
Any later rendering-only replay must retain the exact saved A2UI and be labeled
as a replay with zero new model calls.
