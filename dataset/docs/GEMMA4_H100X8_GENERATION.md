# Gemma 4 31B generation on one 8 x H100 80 GB server

## Recommended starting point

Start with **four independent BF16 replicas, two GPUs per replica**, and one dataset writer dispatching concurrent requests to all four. This is a benchmark candidate, not a measured fastest configuration. No GPU experiment was run on the Windows development PC.

The [official Gemma 4 vLLM recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html#throughput-vs-latency-tuning) recommends low tensor parallelism for aggregate throughput and gives TP=2 with 128 sequences as a balanced configuration. The exact winner depends on prompt lengths, completion lengths, KV-cache capacity, repairs and CPU scoring. A 31B BF16 model has roughly 62 GB of parameter bytes before runtime overhead; TP=1 on an 80 GB GPU leaves much less headroom for long concurrent requests than TP=2.

| Setting | Starting value |
|---|---|
| GPU layout | 4 replicas x TP=2; pairs 0,1 / 2,3 / 4,5 / 6,7 |
| Model precision | BF16; existing approved model/tokenizer snapshot |
| Maximum context | 16,384 tokens including prompt and output |
| GPU memory fraction | 0.90 |
| Maximum sequences per replica | 128 |
| Scheduled tokens per replica per iteration | 16,384 |
| Client concurrency | 32 requests per replica; 128 total |
| Prefix cache / asynchronous scheduler | Enabled, with installed-version flag checks |
| Image/audio/video input encoders | Disabled for this text-only workload |
| Stage 1 | 8 queries per completion, up to 32 intents requested concurrently |
| Stage 2 | 128-query wave, 1 response per query |
| Stage 3 | 128-response wave, 1 candidate per response |
| Output budget | 8,192 tokens per stage for reasoning plus the final answer |
| Stage 3 prompt estimate budget | 7,680 at default context/output settings; 512-token safety reserve; explicit configured caps respected |
| Invalid-output recovery | 1 schema repair and 1 final regeneration; retain failures for review |
| Transport retry controls | 2 outer attempts; 180-second connection-retry window checked between attempts; no extra Stage 3 result-error retry loop |
| Fixed sleeps / rate throttle | Zero |
| Whole-run aggregate refresh | Every 1,000 new rows and at stage completion; per-row scoring remains enabled |
| Assets | Keep declared references; skip downloads and asset retry |
| Reasoning | Enabled for every stage; the profile pins server and client thinking controls on |
| Speculative decoding | Off initially; evaluate separately while keeping reasoning enabled |

`max-num-seqs` is server capacity, not guaranteed active traffic. The initial client setting sends only 32 requests per server. Increase it only when measurements justify doing so.

## What was limiting the existing pipeline

1. The base dataset launcher defaults to Stage 3 batch size 1 and 0.2 requests/second. The multi-server wrapper removes that rate limit, but defaults to only 2 requests/server.
2. `run.yaml` adds a one-second call sleep independently of the request-rate limit. The new profile sets `A2UI_CALL_SLEEP_SECONDS=0` through a real config override.
3. Stage 1 intent cycling defaulted to 20. In the current implementation cycling bypasses the batched-intent path. The profile sets `A2UI_STAGE1_INTENT_CYCLE_SIZE=0` and intent batch size 32.
4. Stage 1 batch size means **queries requested within one completion**. It is not HTTP concurrency. Setting it to 128 makes a long output, rather than 128 independent requests. The profile keeps it at 8.
5. The ordinary server launcher automatically uses all visible GPUs for one model instance. More replicas can process independent prompts with less synchronization within each request.
6. `generate_batch` waits for a complete wave. Stage 3 then performs CPU scoring and repairs before the next wave. Long outliers and sequential repairs can leave GPUs idle even with a large batch.
7. Broad asset search/download/retry work is unnecessary for placeholder-based training. Offline asset mode preserves reference identities; it does not certify media bytes or semantic suitability.

## Commands on the Linux GPU server

Use the updated checkout and the existing environment containing a working Gemma 4-compatible vLLM. Activate that environment in **both** shells. Replace the example paths with the actual locations. Record the installed versions instead of upgrading during a comparison.

```bash
cd /absolute/path/to/A2UI
source /absolute/path/to/gemma4_vllm_env/bin/activate
# Set these in both shells after activation: older environments export 8K context.
export VLLM_MAX_MODEL_LEN=16384
export A2UI_QUERY_MAX_TOKENS=8192 A2UI_RESPONSE_MAX_TOKENS=8192 A2UI_GENUI_MAX_TOKENS=8192
export LOCAL_VLLM_MAX_OUTPUT_TOKENS=8192
unset LOCAL_VLLM_ENDPOINTS LOCAL_VLLM_MODELS_URL VLLM_PORTS
nvidia-smi topo -m
python -m pip freeze > /tmp/gemma4_generation_environment.txt
bash dataset/scripts/run_gemma4_h100x8.sh plan
```

Check that each selected pair shares the intended fast GPU interconnect. If GPU numbering differs, set `REPLICA_GPU_IDS` in pair order. The profile requires eight IDs; the replica launcher rejects duplicate IDs.

**Shell 1: start four servers.** Keep this supervisor running.

```bash
GEMMA4_MODEL_PATH=/absolute/path/to/gemma-4-31b-it \
  bash dataset/scripts/run_gemma4_h100x8.sh servers
```

Servers use ports 8000–8003. Logs are under `/tmp/a2ui_gemma4_h100x8`; the endpoint file is `/tmp/a2ui_gemma4_h100x8_tp2.env`. Startup includes model loading/compilation and can take time. Explicitly requested flags fail clearly if unsupported by the installed vLLM. Investigate version support before disabling a flag.

**Shell 2: launch a small pilot.** This command is provided for the GPU server; it was not run on the PC.

```bash
RUN_ID=gemma4_h100x8_tp2_pilot_20260916 \
MAX_GENERATION_TOTAL=500 GENERATION_CYCLE_SIZE=500 \
  bash dataset/scripts/run_gemma4_h100x8.sh generate
```

The client waits for all endpoints to serve the expected model. This profile uses the existing cyclic Stage 1/2/3 workflow. It does not start training or rendering. Use one writer per `RUN_ID`; launching eight writers against the same output files is unsafe.

After checking the pilot, a fresh 10,000-row generation run uses:

```bash
RUN_ID=gemma4_h100x8_tp2_20260916 \
MAX_GENERATION_TOTAL=10000 GENERATION_CYCLE_SIZE=2000 \
  bash dataset/scripts/run_gemma4_h100x8.sh generate
```

The target is a pipeline record-count cap; rejected records can count toward completion. **It does not promise 10,000 accepted training pairs.** Estimate the required source volume from measured acceptance after all quality gates.

Set `GEMMA4_MODEL_ID` consistently in both shells if the served model name differs from `google/gemma-4-31b-it`. For resumptions, retain the same model, prompts and settings; use a new run ID for a different experiment.

The profile verifies endpoint count against the selected replica layout. Clear inherited endpoint variables when changing layouts; an equal-count stale list can still point to the wrong servers. The model-name readiness check cannot verify their GPU allocation or context configuration.

## Find the fastest configuration that preserves quality

Compare these layouts sequentially, stopping the previous server supervisor before reusing its ports:

| `H100_TP_SIZE` | Replicas | What it tests |
|---|---:|---|
| `2` | 4 | Recommended initial balance |
| `1` | 8 | Less inter-GPU communication; tighter memory headroom |
| `4` | 2 | More memory per replica; additional synchronization |
| `8` | 1 | Existing all-GPU model layout |

Apply the same `H100_TP_SIZE` to **both** `servers` and `generate` commands. Each layout uses a separate endpoint file. Use the same frozen input/source cohort when comparing Stage 3, rather than different randomly generated questions. Keep reasoning enabled and model revision, prompt, sampling, output budgets and quality gates fixed. Exclude startup/warm-up from steady-state timing and report end-to-end time separately.

For the selected layout, sweep `LOCAL_VLLM_REQUESTS_PER_SERVER=16`, `32`, then `64`. Record:

- Accepted examples/hour after structural, reference and source-quality checks (primary objective).
- First-pass acceptance, repair attempts/accepted row, missing/truncated output and source-check failures.
- Prompt/output tokens per second, per-endpoint request latency and queue length.
- GPU utilization, KV-cache usage/preemption, CPU utilization and scoring time.
- Coverage by scenario family and modality; reviewed semantic/interaction quality.

If queues grow while throughput does not improve, reduce concurrency. If GPUs are idle while CPU scoring/repair runs, increasing `max-num-seqs` will not solve that bottleneck. If KV preemption or OOM appears, first reduce concurrency/context/sequence capacity or use TP=2 instead of TP=1. [vLLM's optimization guide](https://docs.vllm.ai/en/stable/configuration/optimization/) explains the memory and scheduling tradeoffs.

Prefix caching can reduce repeated prompt-prefill work; it does not speed up generation of new output tokens. Keep the shared instruction prefix stable. Compare production-like warm-cache results separately from cold-cache results. [Prefix cache documentation](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/).

The profile retains an 8,192-token completion budget because reasoning and the final answer share this allowance. The archived run's recorded outputs were below that cap; this does not prove all future answers fit. Increase it for long/complex tasks when actual finish reasons show truncation. Prompt character estimates are not exact tokenizer counts.

The profile derives its Stage 3 prompt estimate budget from `context - requested output - 512`. At the default settings this is 7,680. It explicitly enables `STAGE3_RESPECT_CONFIG_PROMPT_MAX=1`, so a smaller user-specified cap is honored. If complete sources or repair prompts exceed the budget, raise `VLLM_MAX_MODEL_LEN` to 24,576 or 32,768 in both shells and reassess concurrency and memory use. Raise an explicitly configured prompt cap too if needed. No source is silently clipped. Context-error retries cannot reduce the requested completion allowance: the profile sets their minimum output budget to the client output cap. Resolve those failures by adjusting context or scheduling, while retaining the source and reasoning budget.

Retry budgets are explicit to limit long serial recovery tails. The connection-retry window is checked between attempts, not a hard request deadline; each in-flight request still has its separate 1,800-second timeout. Inspect failed rows and coverage after the pilot. For complex cohorts, compare `STAGE3_FINAL_REGEN_ATTEMPTS=3` against the fast default of 1 using accepted examples/hour, rather than retrying indefinitely.

Keep reasoning enabled throughout speed tuning. The H100 profile explicitly sets `GEMMA4_ENABLE_REASONING=1`, parser mode, server default thinking where supported, `LOCAL_VLLM_ENABLE_THINKING=1`, and `LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS=1`. These override inherited reasoning-off settings from older activation scripts. Keep BF16 as the baseline. FP8 KV cache and speculative decoding are later, independent experiments that need compatibility and quality checks.

The H100 profile sends `chat_template_kwargs: {"enable_thinking": true}` on requests. Outside this profile, an unset thinking control leaves the model's template default intact; the HTTP adapter no longer interprets an absent setting as an instruction to disable thinking.

## Further pipeline changes if measurements show GPU idle time

These are follow-up architectural improvements, not implemented concurrency guarantees:

1. Replace whole-wave barriers with a bounded work queue that streams completed responses into a CPU evaluator pool.
2. Batch repairs in their own queue; give repair traffic a fixed share of capacity so difficult rows cannot stall all new work.
3. Separate Stage 1/2 source production from Stage 3 conversion with a bounded backlog and a single artifact writer.
4. Use queue-aware endpoint dispatch when response lengths vary substantially; round-robin currently cannot see server load.
5. Cache/version shared prompt, contract and schema work; record CPU timings before moving scoring to worker processes.

Retain quality checks during speed tuning. A faster stream of incorrect examples is not a better training dataset.
