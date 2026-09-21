# Gemma 4 E2B all-parameter QAT pipeline

This is a separate, experimental training lane for the reconstructed Gemma 4
E2B official-mobile text seed. It does **not** modify or replace the official
retained-scale LoRA pipeline in `official_mobile.py`.

## What this lane does

- Reuses the checked dataset preparation/cache and the same completion-masked
  A2UI Express examples.
- Trains every unique text-model parameter, including the ordinary token
  embedding, per-layer embedding, every fully connected weight, norms, and
  other text-model parameters. LoRA/PEFT and frozen model parameters are
  rejected. The persistent per-layer `layer_scalar` tensors are buffers, not
  parameters: they remain in model state but are not made trainable.
- Keeps master parameters and saved checkpoint tensors in FP32, with BF16 AMP
  for execution.
- Uses Adafactor in the default replicated-DDP lane. The opt-in sharded lane
  uses PyTorch AdamW with DeepSpeed ZeRO-2. In both cases the live
  maximum-shape backward/optimizer probe is mandatory and planning alone does
  not establish that memory fits.
- Applies dynamic `ste_ai_edge` weight fake quantization to every Linear and
  both embedding matrices using the experimental W2/W4/W8 allocation.
- Selects the best full-model checkpoint only with Golden32
  `unique_source_generation_reward_v5_4_avg`, then evaluates that fixed
  checkpoint on Golden35 and Bixby50 as final-only holdouts.
- Writes TensorBoard training and evaluation records using the existing
  dashboard contract.
- Exports only experimental `w248` through the existing dense checkpoint
  exporter. The exporter copies the selected full checkpoint into a fresh
  dense HF graph; it does not merge a LoRA adapter.

## What this lane does not claim

This is not Google's official retained-scale/static-A8 mobile topology or
private QAT recipe. It recomputes dynamic fake-quantization scales from the
current weights rather than learning or retaining Google's published scales,
and rebuilds a fresh dense graph during export. MTP is disabled. Desktop export does not prove
native LiteRT-LM/Android quality, memory use, or speed; those remain separate
device gates.

The model directory must be the locally reconstructed official-mobile text
seed and must include its verified `mobile_training_seed_manifest.json`,
`mobile_qparams.json`, and `mobile_qparams.safetensors`. The qparams bind seed
provenance; they are not reused as fixed training scales in this lane.

## Architecture comparison and parameter scope

The two training/export lanes start from related weights but deliberately do
different work:

| Property | Official retained-scale LoRA lane | Experimental all-parameter QAT lane |
| --- | --- | --- |
| Training scope | LoRA adapters on the exact 205 mutable projection matrices | Every unique parameter in the reconstructed text model, with no frozen parameter or adapter |
| Training graph | Reconstructed `Gemma4ForCausalLM` text seed, followed by adapter merge | Reconstructed `Gemma4ForCausalLM` text seed, trained directly in FP32 master weights |
| Export graph | The published official target topology | A newly converted dense `gemma4_text` graph; it is not the official target graph |
| Weight materialization | Merge into the reconstructed text seed, then patch the 205 corresponding W2/W4 fully-connected code buffers in the official target | Convert all trained dense matrices using a new experimental W2/W4/W8 allocation |
| Preserved official data | All 72 frozen target constants, all 263 direct-BF16/non-dequantized seed entries, the external embedders, and the published static-A8/fixed-scale contract | The complete trained checkpoint identity; published fixed scales and the official graph topology are intentionally not reused |
| What success would establish | A topology- and retained-scale-preserving candidate, subject to its existing fail-closed parity gates | A physically verified dense mixed-precision artifact, still requiring native/device validation |

Count the full-QAT scope precisely. The reconstructed checkpoint has **541
named state entries**: **506 actual named parameters** and **35 persistent
`model.layers.{0..34}.layer_scalar` buffers**. The parameters comprise **279
matrix parameters** and **227 non-matrix parameters**. The 541 checkpoint
entries still originate as **278** entries reconstructed by dequantizing packed
published matrices and **263** direct BF16 copies; those provenance counts do
not make every state entry a parameter. The 278 dequantized entries include
both embedding-table entries and the output-head entry, while the live model
also has the direct-BF16 `per_layer_model_projection` matrix. The reconstructed
config explicitly sets `tie_word_embeddings: false`; do not infer tied trained
storage from packed-source aliases. The scope gate records both alias-aware
named-parameter and unique-parameter coverage. All underlying unique parameters
must remain trainable FP32 masters, including both embedding modules, norms,
and every fully connected weight. The 35 `layer_scalar` buffers remain
persistent model state and must not be promoted to parameters or described as
trained.

The full-QAT preflight verifies the exact parameter-name/shape inventory and
the exact 35 persistent buffer registrations separately. Each scalar buffer
must be finite FP32 with shape `[1]` and must not require gradients. Its value
hash is retained in checkpoint provenance; checkpoint validation rejects a
missing, extra, retyped or changed buffer rather than counting it as a trainable
parameter. The six derived rotary/embedding buffers are nonpersistent and are
not included in the 541-entry checkpoint state. Export proof reports use
`state_tensor_count: 541`, `named_parameter_count: 506` and
`persistent_buffer_count: 35`; every state entry still requires physical export
verification. Seed files and buffer registrations are never rewritten to pass
these checks.

| Parameter family | Named tensors | LoRA export | Full export |
| --- | ---: | --- | --- |
| Retained attention/MLP projections | 205 | Patch code buffers with fixed published scales | Serialize trained matrices with recomputed W2/W4 scales |
| Output head and layer-local input-gate/projection matrices | 71 | Preserve official bytes | Serialize trained W2 head and W8 layer-local matrices |
| Global per-layer-model projection | 1 | Preserve official bytes (BF16 seed source; W8 target) | Serialize trained W8 matrix |
| Token and per-layer embedding tables | 2 | Preserve separate official embedding sections | Serialize each trained table into its own new W2/W4 section |
| Norms and other non-matrix parameters | 227 | Preserve official values/compiled constants | Serialize the trained FP32 parameters at verified graph use sites |
| Persistent `model.layers.{0..34}.layer_scalar` buffers | 35 | Preserve official values/compiled constants | Preserve the expected buffer values in saved/exported model state; do not train or promote them to parameters |

## Why full QAT needs a fresh converted graph

The official retained-scale exporter is intentionally a narrow patcher. It can
replace the 205 projection code buffers because the other 72 official target
constants, external embedding sections, static activation quantization, and
published fixed weight scales remain invariant. Full QAT breaks that premise:
embeddings, per-layer projections, norms, and all other model parameters can
change, while the persistent layer-scalar buffers must be preserved, and
training uses dynamic weight fake quantization with
floating-point activations. Reusing only the 205-code patcher would silently omit
trained parameter changes outside its scope and would falsely present a dynamic
experimental recipe as the official fixed-scale topology.
Full QAT must therefore load the complete saved checkpoint into a fresh dense
HF graph and convert that graph in full.

## Standalone-text export fix and verification boundary

Pinned LiteRT Torch 0.9.4 has Gemma 4-specific routing for top-level
`model_type: gemma4`, but no standalone `model_type: gemma4_text` route. Its
Gemma 4 exportable wrappers are also written for the multimodal wrapper shape:
they read `config.text_config` and traverse `model.language_model`. The honest
full-QAT checkpoint instead uses `Gemma4TextConfig` directly and
`Gemma4ForCausalLM.model`. Relabeling the config or transplanting these weights
into the official graph would violate checkpoint identity and is not an
acceptable workaround.

The repository implementation is therefore scoped to a temporary,
version-pinned `gemma4_text` compatibility route. It reuses the upstream Gemma
4 cache, patch, metadata, and converter machinery, but supplies exportables for
the standalone text topology and restores every upstream registry/function on
exit. It does not modify installed packages, reuse the 205-buffer retained-
scale patch path, change model type, or allow missing checkpoint state.

The full lane explicitly opts into this route during its before-training
exporter probe. Ordinary dense export and official retained-scale LoRA do not.
The route additionally pins AI Edge Quantizer 0.9.0 and LiteRT 2.2.0 for physical
verification. During conversion it:

1. Checks every loaded FP32 state tensor against the selected checkpoint's exact
   bytes, including the expected persistent buffers.
2. Requires all 541 source state tensors at their corresponding consumed graph use
   sites in the actual floating TFLite files, including both external embedders.
3. Recomputes W2/W4/W8 codes and channelwise scales from selected-checkpoint
   weights using the pinned quantizer. Every corresponding quantized buffer,
   scale, zero point and dtype must match; non-matrix constants remain exact FP32.
4. Checks that the final package contains exactly those three audited model
   sections, byte-for-byte. Existing physical-precision inspection still runs.
5. Binds `full_parameter_serialization.json` into `export_manifest.json` and the
   parent export receipts. Missing or changed proof prevents success.

Weight fusion and whole-graph FP16 conversion are disabled for this lane to
preserve auditable parameter mappings. Unknown compiler folds, transpositions,
or opaque/unrecognized use-site names **fail closed**, not by matching a value
elsewhere in the model. A real converter may expose such an unsupported mapping;
that requires a reviewed mapping/transform backed by actual converter evidence,
not disabling the gate. Quantized weights are of course not byte-identical to
FP32 masters: the proof concerns their exact specified quantization.

Floating and quantized intermediate TFLites are retained beside the final
package for audit. Budget substantial additional disk space. They are not
training inputs and are never committed automatically.

This is an implementation with local regression coverage, not a completed E2B
conversion result. No full E2B conversion or native-device run was performed
here; no official-device speed or quality claim follows from these tests.

## Host contract

- Use a separate training environment with
  `pip install -r training/requirements-full-parameter-qat.txt`. The new lane's
  Trainer/Accelerate API pair is pinned independently; do not update the working
  official LoRA environment in place. API compatibility is checked before model
  loading, and live DDP bucket views are checked during training.
- Exactly 2, 4, or 8 selected NVIDIA H100 GPUs, each reporting at least 79 GiB.
- Native BF16 support.
- Per-rank microbatch fixed at 1. The default effective batch is 32, recovered
  through gradient accumulation (16/8/4 steps for 2/4/8 GPUs).
- Full-QAT sets `ddp_sync_each_batch: true`: DDP synchronizes **every** microbatch
  with `gradient_as_bucket_view=True`, but Trainer still divides the loss by the
  accumulation count and updates the optimizer only at the accumulation boundary.
  On eight GPUs, this is still 1 x 8 x 4 = 32 samples per optimizer update.
  This trades more frequent all-reduces for lower peak gradient memory. The
  official retained-scale LoRA pipeline keeps its existing accumulation policy.
- The default lane is DDP only; it has no automatic FSDP/DeepSpeed fallback
  and no silent CPU fallback.
- This is replicated DDP: 8 x H100 80 GB is supported, but each rank still holds
  a full model, gradients, and optimizer state. Eight GPUs do not pool their
  memory into a single 640 GB device or remove the per-rank memory requirement.
- A separate compatible CPU LiteRT Torch / AI Edge Quantizer Python for export.

The statements above describe the unchanged default, selected explicitly with
`--distributed-backend ddp` or implicitly when the option is omitted. There is
also an opt-in `--distributed-backend sharded` mode for Linux. It must be
installed in a separate Python venv so the established DDP/LoRA dependency sets
are not modified. Activate that dedicated sharded venv before these commands:

```bash
python -m pip install -r training/requirements-full-parameter-qat-sharded.txt
# Force a venv-local copy even if a matching version is visible system-wide.
# This does not uninstall or patch the system package.
python -m pip install --ignore-installed --no-deps nvtx==0.2.15

# Tiny environment/API check; no model loading or Trainer memory probe.
python training/scripts/check_sharded_training_env.py --world-size 4 --effective-batch 32
```

The sharded overlay pins `nvtx==0.2.15` as well as DeepSpeed 0.19.7. DeepSpeed
uses `domain.push_range(message=..., category=...)`; NVTX 0.2.14's no-profiler
`DummyDomain` rejects those keywords. Version 0.2.15 supports them. The gate
verifies both the version and that the imported NVTX module and distribution
metadata resolve inside the active venv, not inherited `/usr/local` packages.
An inherited system 0.2.15 copy is also rejected: install the venv-local copy
with the explicit command above. Do not uninstall system NVTX, patch
`site-packages`, or set `NVTX_DISABLE` to bypass instrumentation.

The quick check exercises a balanced direct NVTX push/pop **and the actual
DeepSpeed NVTX range wrappers**, and reports the import path. It also checks
the accumulation configuration: for microbatch 1 and effective batch 32,
Trainer, the Accelerate plugin, and DeepSpeed all configure 8 steps on 4 GPUs
(4 on 8 GPUs; 16 on 2 GPUs). Set `--world-size` to the intended GPU count.
This does not create a distributed engine or prove memory fit or GPU count.

Sharded pipeline execution automatically runs the same dependency/NVTX probe
in a bounded, at-most-120-second `environment` stage **before** seed inspection,
data preparation, or model loading. Evidence is saved in
`sharded_environment.json` and `logs/environment.log`; a failure stops the run.
Each actual SFT worker repeats the check before loading weights. Run the tiny
check successfully before retrying the expensive full preflight, then use a
fresh pipeline output directory as usual. The DDP stage list and dependencies
remain unchanged.

Only the sharded Trainer hook aligns Accelerate's accumulation plugin before
Accelerator construction; live checks require Trainer, Accelerate, and the
DeepSpeed config to agree. Trainer continues dividing loss once and supplying
`scale_wrt_gas=False`; Accelerate's DeepSpeed branch does not divide it again.
Optimizer boundaries remain controlled by Trainer. The DDP path retains its
original Accelerate accumulation behavior and synchronization policy.

The sharded mode is narrowly defined as DeepSpeed ZeRO-2 with PyTorch AdamW.
FP32 master parameters and BF16 `torch.autocast` compute are unchanged. Only
gradients and optimizer state are sharded; model parameters remain replicated.
CPU/NVMe optimizer or parameter offload is disabled. This mode is not FSDP or
ZeRO-3, and it does not pool all GPU memory into one logical device.

The sharded optimizer choice is an explicit experimental contract, not a claim
that AdamW is superior to the default Adafactor recipe. It changes optimizer
semantics, so results must not be presented as a direct optimizer-controlled
comparison without a dedicated experiment. Neither backend promises that a
6,144-token context fits, and neither has a throughput or speed advantage
claim. Both must pass the same strict live maximum-shape, numeric, checkpoint,
Golden holdout, provenance, and export gates. Resume remains unsupported; a
failed or changed run starts in a fresh output directory.

The preflight is intentionally stronger than a forward smoke test. In DDP mode,
each rank first runs the existing independent numeric, coverage, and backward
checks. The disposable memory probe then runs the **same checked Trainer, Accelerate
preparation, BF16 AMP, DDP wrapper, collator, and Adafactor creation path** as
training. Every rank repeats the real longest prepared row for two complete
optimizer updates (eight microbatches per rank with eight GPUs and accumulation
4). This tests both startup and a subsequent update with optimizer state and
rebuilt DDP buckets resident. Saving, evaluation, and external metric reporters
are disabled for this disposable run. Regular training starts later in a new
process and reloads the untouched seed; probe weights are never continued or
saved as a user checkpoint.

With `--distributed-backend sharded`, the corresponding disposable probe uses
the production DeepSpeed ZeRO-2 wrapper and PyTorch AdamW path instead. It must
demonstrate the selected backend and sharded gradient/optimizer-state contract
on every rank; it is not permitted to satisfy the gate with DDP evidence. The
numeric and coverage checks still run independently, but the backward gate runs
inside this real sharded Trainer rather than allocating replicated gradients in
a bare-model backward. This gate inspects existing rank-local gradient fragments
and verifies their exact cross-rank coverage; it never gathers a full embedding
gradient just for validation. Its separate version-3 receipt binds the exact
ZeRO configuration, accumulation count, AdamW state, and memory headroom. The
Golden32 selector, final-only Golden35/Bixby50 evaluations, downstream stage order, export
route, H100 counts, effective batch, and checkpoint provenance remain the same.

### Diagnosing rank-local ZeRO-2 memory failures

The disposable sharded probe writes a separate diagnostic file on **every
rank**, including ranks that later fail the memory gate:

```text
<output>/fit/training/sharded_preflight_diagnostics_rank<RANK>.json
<output>/logs/preflight_launch.log
```

These are incremental, atomically replaced diagnostic records, **not** passing
`full_optimizer_preflight_rank<RANK>.json` receipts. Each worker flushes its
complete final counters and prints a `Sharded preflight memory:` line, then
joins a final distributed barrier before any worker calls the fail-closed
validator. This applies to ranks that reach final validation: an earlier rank
failure can leave peers waiting until launcher teardown. A catchable failure
also writes an error record; an abrupt process kill or a broken CUDA context
can leave only the last available sample. An early failure never claims that
NCCL was certified before a gradient-coverage collective succeeds. A diagnostic
write failure also stops the probe rather than silently continuing without
the requested evidence. The original validation inputs are retained separately
from any later exception-time memory sample.

The report includes baseline, current and observed peak allocated/reserved
bytes, device total and sampled minimum free bytes, reserved/free fractions,
GPU/rank identity, and **each named gate predicate**. The limits are unchanged:
reserved fraction must be strictly **below 0.90**, and sampled free fraction
strictly **above 0.10**. Context length, microbatch 1, accumulation, and effective
batch 32 are not changed by diagnostics. DDP and retained-scale LoRA do not use
this recorder.

Samples separate Trainer construction, live ZeRO engine initialization,
microbatch returns, pre-update gradient/master validation, the actual optimizer
step, and post-update moment/parameter validation. Partition snapshots record
group sizes and padding, local FP32 master and AdamW moment element/byte counts,
local gradient-fragment ownership, flat parameter views and communication
bucket buffers. They read tensor metadata, not tensor values or full gradient
copies. Logical view bytes can alias; the deduplicated storage summary covers
only known reported categories, not total GPU usage. Missing optional metadata
is marked unavailable and does not replace the mandatory gradient/state audits.

Interpret rank differences with these caveats:

- Pinned DeepSpeed selects CPU versus GPU parameter flattening using each
  rank's available memory. The diagnostic temporarily captures its setup INFO
  events and restores the logger afterward. Compare actual branch events rather
  than assuming the same startup path. Core flat partitions are near-equal;
  a different number of locally owned parameter *names* does not itself imply
  different allocated master/moment bytes. See the
  [DeepSpeed 0.19.7 partition implementation](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.7/deepspeed/runtime/zero/stage_1_and_2.py).
- DeepSpeed's `empty_cache()` helper also resets peak counters; some memory
  logging paths reset them too. The recorder retains the maximum of peaks
  **observed at all sampling points** and reports decreases, but cannot recover
  transients reset between samples. It never resets counters itself. See
  [DeepSpeed memory helpers](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.7/deepspeed/runtime/utils.py).
- Free memory is discrete, device-wide sampling, not a continuous per-process
  minimum. The non-PyTorch estimate is contemporaneous driver-used memory minus
  current PyTorch reserved memory. It can include other processes, CUDA/NCCL
  allocations and sampling races; it is **not** an attribution to NCCL. Do not
  subtract peaks and free minima measured at different times.

For the reported four-GPU odd-rank failures, successful rank 0/2 counters alone
cannot establish the cause. Rerun the **same requested configuration** in a
fresh output directory and collect all four diagnostic files plus the preflight
log. Compare phase peaks, local partition/moment bytes and external estimates
before deciding whether the asymmetry is partition-related, a measurement
effect, or real rank-local memory pressure. Do not lower the gate or bypass it.

The previous single-backward raw-DDP probe did not reproduce Trainer's
`no_sync()` accumulation. At the first accumulated backward, `no_sync` left a
standalone FP32 gradient set in addition to the DDP buckets: about 18.74 GiB for
5,031,222,528 parameters. A true `gradient_as_bucket_view` flag alone did not
prevent that allocation. Syncing each microbatch allows the reducer to attach
bucket views immediately. See [PyTorch's DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
Full-parameter training no longer installs the unnecessary k-bit LoRA input
gradient hook; trainable embeddings already propagate gradients.

Each DDP rank's version-2 receipt must prove both complete accumulation windows,
the live synchronization policy, all-parameter finite gradients/weights and
Adafactor state, peak reserved memory below 90%, and sampled device free memory
above 10%. Training, pipeline orchestration, and checkpoint validation share
the same fail-closed validator. Old single-step receipts cannot pass. A passing
probe covers the tested maximum shape and host allocation, not every future
kernel shape, external GPU workload, or possible runtime OOM.

Backward gradient validation and the disposable optimizer probe's gradient and
post-step parameter checks inspect every value in chunks of at most 1,048,576
elements. They do not allocate a whole-embedding finite/nonzero mask, call
`count_nonzero` on an entire gradient, or flatten/copy a noncontiguous tensor.
Backward outputs are released before validation. NaN/Inf, missing gradients in
the optimizer probe, and an entirely zero backward pass still fail closed;
these checks do not sample values or skip later chunks after finding a nonzero.
The reports record `exhaustive_bounded_chunks` and the chunk-element limit.

Only this full-parameter launcher defaults child workers to
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing PyTorch.
An explicit `PYTORCH_ALLOC_CONF` or legacy `PYTORCH_CUDA_ALLOC_CONF` value is
preserved unchanged. Requested settings appear in the SDPA runtime log and
CUDA-failure diagnostics. Expandable segments are an experimental PyTorch
fragmentation mitigation, not additional GPU capacity or proof of memory fit;
see [PyTorch allocator documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf).
The official retained-scale LoRA launch policy is unchanged.

For an OOM, use a **fresh output directory** and fresh CUDA workers. Keep
microbatch 1 and explicitly choose the requested context and backend; do not
bypass preflight or silently reduce the effective batch. A DDP context that
does not fit is not made safe by allocator settings alone. Both backends must
pass their real longest-shape, all-rank Trainer gate before training. CPU
regression tests cannot certify that live H100 allocation.

## Plan first

Planning is offline and is the default. It validates paths and static options
but does not probe CUDA, load the model, train, or export.

Defaults: 2 epochs, learning rate `1e-5`, 3% warmup, validation/save every 500
optimizer updates, Golden32 every 1,000 updates plus the end, and 2,048 generated
tokens. TensorBoard uses `/tensorboard/<run-id>/`. These are conservative starting
settings, not a measured optimum. Full checkpoints are substantially larger than
LoRA adapters; budget disk space for retained Trainer checkpoints, best/final
copies, and the dense export staging copy. This lane deliberately requires fresh
runs and does not resume LoRA or full-model optimizer state.

```bash
python training/scripts/run_full_parameter_qat_pipeline.py \
  --model-dir /models/gemma4_e2b_mobile_dequantized_text_hf \
  --input-dir /data/a2ui_prepared_source \
  --output-dir /runs/e2b_all_parameter_qat_001 \
  --exporter-python /opt/litert-export/bin/python \
  --devices auto
```

The input directory contains `train.jsonl` and `val.jsonl` in the same reviewed
format accepted by the Golden training pipeline. Existing preparation caches
are reused only when their source, tokenizer, prompt, schema, and code identity
all match.

## Execute explicitly

Use a fresh output directory. Execution refuses to start without the explicit
experimental-export acknowledgement:

```bash
python training/scripts/run_full_parameter_qat_pipeline.py \
  --model-dir /models/gemma4_e2b_mobile_dequantized_text_hf \
  --input-dir /data/a2ui_prepared_source \
  --output-dir /runs/e2b_all_parameter_qat_001 \
  --exporter-python /opt/litert-export/bin/python \
  --devices auto \
  --execute \
  --allow-experimental-export
```

Useful bounded smoke options are `--steps`, `--eval-steps`, and
`--golden-every-steps`. The Golden cadence must remain a positive multiple of
the validation cadence. Do not use a smoke result as final model evidence.

For example, this plans an opt-in bounded sharded run while preserving the
existing cadence relationship. Add `--execute --allow-experimental-export`
only after reviewing the plan, and use a fresh output directory:

```bash
python training/scripts/run_full_parameter_qat_pipeline.py \
  --model-dir /models/gemma4_e2b_mobile_dequantized_text_hf \
  --input-dir /data/a2ui_prepared_source \
  --output-dir /runs/e2b_all_parameter_qat_sharded_smoke_001 \
  --exporter-python /opt/litert-export/bin/python \
  --devices auto \
  --distributed-backend sharded \
  --steps 20 \
  --eval-steps 5 \
  --golden-every-steps 10
```

## Retry export without retraining

If training already completed and the selected full checkpoint exists, run the
existing export-only entry point from the training environment. Use its original
`fit/` directory and a **new** export destination:

```bash
python training/scripts/export_checkpoint_litertlm.py \
  --profile e2b \
  --fit-dir /runs/e2b_all_parameter_qat_001/fit \
  --output-dir /runs/e2b_all_parameter_qat_001_w248_retry \
  --exporter-python /opt/litert-export/bin/python \
  --variants w248 \
  --allow-experimental-formats \
  --execute
```

It defaults to `fit/training/best_golden_checkpoint` and detects the full-QAT
route from verified checkpoint provenance. It does not retrain, merge adapters,
or rerun Golden/Bixby evaluation. Do not manually edit model type, manifests, or
proof files. If the original run stopped at the before-training exporter probe,
there is no trained checkpoint to export; launch the full pipeline into a fresh
run directory instead.

## Outputs and stage order

Every stage runs in a bounded subprocess with console/file progress and a
content-bound receipt:

1. Verify the reconstructed official-mobile seed and qparams provenance.
2. Prepare/cache data and frozen Golden32, Golden35, and Bixby50 cohorts.
3. Detect and bind the 2/4/8-H100 profile; write
   `fit/training_config.yaml` and `fit/preparation_report.json`.
4. Run the disposable full-QAT preflight.
5. Train from a fresh seed and publish `fit/training/best_golden_checkpoint`
   plus `fit/training/final_model`.
6. Re-evaluate the selected checkpoint on Golden32, Golden35, and Bixby50.
7. Export experimental W248 into `experimental_w248_export/`.
8. Write `evaluation_scorecard.json` and the three-cohort `results.md` table.

The selected checkpoint must contain verified scope for all 506 actual
parameters, complete 279-matrix QAT coverage, successful numeric/backward
preflight evidence, and a complete FP32 541-entry model-state inventory that
also preserves the 35 expected persistent `layer_scalar` buffers. Positive
optimizer-step provenance and immutable checkpoint/config/data hashes are also
required before evaluation or export proceeds.

## Interpreting completion

Pipeline completion means checked full-parameter training, three-cohort HF/QAT
evaluation, and creation of an experimental W248 artifact. It does not mean the
artifact has passed Android GPU execution, native prompt-to-A2UI quality, or
throughput parity. The final report says those gates are untested until they are
run separately on the intended device/runtime.

## Regression evidence and remaining gates

Earlier all-parameter training-lane CPU validation on 2026-09-20 included a broad existing-pipeline suite
(943 passed, 24 skipped) and a fresh affected-suite rerun (370 passed, 9 skipped).
A real tiny Transformers Trainer/Accelerate test also exercised one Adafactor
update and full safetensor checkpoint save, including changed embeddings and
normalization weights. An additional two-layer Gemma 4 CPU test exercised 22
actual Linear/Embedding QAT wrappers, finite gradients for every parameter,
Adafactor probe steps, and a full FP32 checkpoint round trip. These synthetic
CPU tests are not an E2B training run.
The original official-mobile entry point, orchestration module and recipe YAML
were not edited.

The standalone-text export fix was separately checked with a final combined
regression suite: **586 passed, 7 skipped**. New tests cover an actual tiny Gemma 4 forward
pass (both embedders, final norm and output head), schema-generated physical
TFLite constants (including external buffers), missing/changed/wrong-scope
weights, W2/W4/W8 code and scale corruption, exact packaged-section binding,
converter-hook restoration, and root receipt rejection of missing/modified
proof or the wrong full-parameter variant. Schema fixtures test physical serialization;
they are not executable E2B graphs. Converter orchestration is mocked, not a
full conversion. Pinned 0.9.4 converter source was inspected, and the 14 physical
serialization tests were additionally rerun successfully against the extracted
AI Edge Quantizer 0.9.0 wheel via process-local `PYTHONPATH` (no installed-package
changes). These tests do not claim successful execution of the complete pinned
conversion environment.

The subsequent persistent-buffer correction passed a combined **603 tests,
7 skipped** regression run. A real tiny-width, 35-layer Transformers Gemma 4
model reproduces the exact 541-state/506-parameter/35-persistent-buffer split,
including six nonpersistent buffers, BF16 seed loading into FP32, and a complete
save/reload round trip. Negative tests cover missing/extra/wrong-shape buffers,
buffers promoted to parameters, nonfinite or non-FP32 buffers, changed saved
buffer values, and incomplete checkpoint/export evidence. No seed or existing
run artifacts were modified. This is CPU regression evidence, not H100 execution.

Useful focused checks after installing the new training environment:

```bash
python -m pytest -q -p no:cacheprovider \
  training/tests/test_full_model_contract.py \
  training/tests/test_full_model_buffer_integration.py \
  training/tests/test_full_parameters.py \
  training/tests/test_full_parameter_qat_pipeline.py \
  training/tests/test_full_qat_trainer_integration.py \
  training/tests/test_official_mobile_pipeline.py \
  training/tests/test_gemma4_text_export_compat.py \
  training/tests/test_full_parameter_serialization.py \
  training/tests/test_full_parameter_export.py
```

No actual H100/DDP run, full-model conversion, Android inference, or measured
quality/throughput comparison was performed locally. Run the mandatory live
preflight on the target host; do not disable a failed memory, scope, or
provenance check to continue.

The new sharded contract has CPU regression tests for backend selection,
gradient-partition coverage, strict memory receipts, checkpoint provenance,
and unchanged default DDP behavior. A separate opt-in integration test runs
the real two-GPU ZeRO-2 preflight, then training and an exact FP32 checkpoint
round trip in fresh worker processes:

```bash
A2UI_RUN_SHARDED_GPU_TESTS=1 python -m pytest -q \
  training/tests/test_sharded_training_gpu.py
```

Run this in the pinned Linux sharded environment with two visible BF16-capable
CUDA GPUs. It uses a tiny QAT model, not E2B, and cannot establish full-model
memory fit or throughput. It is skipped by default; no local CUDA/DeepSpeed run
is claimed. The real pipeline's maximum-shape preflight remains mandatory on
all selected H100 ranks.
