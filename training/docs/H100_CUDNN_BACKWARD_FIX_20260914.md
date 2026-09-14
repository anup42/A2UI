# H100 E2B backward failure: diagnosis and restart

## Evidence and limits

The supplied September 14 screenshots show an eight-worker E2B HPO baseline
reaching roughly optimizer step 395/1000. Photo 5 contains the earliest useful
failure evidence:

- An allocation of 7,832,862,720 bytes (about 7.30 GiB) failed with only
  3,813,212,160 bytes (about 3.55 GiB) free on an approximately 79.1 GiB device.
- A worker failed inside attention backward with
  `mha_graph.execute(...).is_good()` returning false.
- Other workers then reported `cudaErrorContained`, invalid peer GPU memory
  over NVLink or a hardware error. Torchrun terminated the worker group.

The evidence supports memory pressure plus a cuDNN attention-backward failure.
It does not identify the exact original allocation site, prove a particular
driver defect, or exclude a hardware/NVLink fault. Distributed console lines
are interleaved; the final SIGABRT summary is not the earliest causal evidence.
This is not a token-cache or training-data JSON schema failure.

The user confirmed **eight H100s with 80 GB each**. DDP replicates the model
and gives each worker its own microbatch; those GPUs do not form one pooled
640 GB device. A single worker can exhaust its own 80 GB allocation while the
job is correctly using all eight GPUs.

The old automatic H100 profile used E2B microbatch 2. The checked loss retains
full-sequence logits and computes completion cross-entropy in FP32. With a
262,144-token vocabulary, a `[2, 4095, 262144]` FP32 tensor alone is about
8 GiB, before other logits, gradients, activations and attention workspace.
Thus, fitting the model's parameters is not evidence the training step fits.
No loss rewrite, truncation or data rejection is introduced by this fix.

## Changes

1. Automatic E2B microbatch is now **1**. On 2/4/8 H100s, accumulation is
   **16/8/4**, preserving global batch **32** and using every selected GPU.
   The 270M H100 microbatch remains 4. Explicit overrides remain supported;
   larger E2B microbatches carry a memory warning in the resolved profile.
2. `training.disable_cudnn_sdpa: true` is the shared SFT default. It disables
   only cuDNN attention, preserving existing Flash, memory-efficient and math
   settings across forward, backward, checkpoint recomputation and in-process
   Golden evaluation. Previous backend flags are restored on return/error.
   This uses the documented [PyTorch backend control](https://docs.pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_cudnn_sdp).
   It does not disable all cuDNN, NCCL P2P, TF32, BF16 or checkpointing.
3. `training.backward_preflight: true` enables a live, non-updating probe on
   every non-QAT HF CUDA worker, before optimizer step 1. It scans the actual
   tokenized training split, repeats the longest admitted row to the configured
   microbatch, and also checks a mixed-length padded batch when applicable.
   Each shape gets at most two backward passes when accumulation is enabled,
   so the second pass runs with existing gradients resident. It uses the same
   checked completion loss and checkpointing configuration as training.
4. The probe preserves RNG, parameters, model modes and existing gradient
   hooks, clears probe gradients, and constructs no optimizer. It verifies
   finite loss and finite/nonzero trainable gradients. In-place parameter or
   buffer mutation fails closed. Reports include shape, loss, memory and scope.
5. Failed Trainer steps emit `SFT TRAINING STEP FAILED` with optimizer step,
   rank, tensor shapes, available memory diagnostics and backend/package
   versions, while retaining the original exception. No step is retried or
   skipped automatically. Probe progress and errors are flushed to console.

The backend/probe reports are saved in preflight results and
`training_metadata.json` alongside checkpoints. Existing loss, evaluation and
timing TensorBoard logging remains under `/tensorboard`; this change does not
claim new Golden scores. The probe is deliberately not cacheable. Preparation
and token caches remain reusable when their data/tokenizer/formatting bindings
match; changing only this backend policy and microbatch does not rebuild them.

The probe tests local forward/backward shapes, not optimizer-state allocation,
every possible kernel shape, DDP collectives, long-run fragmentation or GPU
hardware health. It runs before distributed wrapping. CPU production runs
skip it; tiny CPU regression fixtures explicitly opt in. Stateful QAT skips
this new probe to avoid modifying observer/scale state and retains its existing
QAT gates. The legacy TRL path is outside the new backward-probe contract.
Official retained-scale QAT, export and MTP validity are not established here.

The probe currently uses the loaded model's native precision, which is BF16
for the default H100 profile. It does not reproduce Trainer AMP/autocast or an
FP16 GradScaler. Custom FP32-model/AMP or FP16 recipes may have different
memory/gradient behavior; this is not precision-general certification.

## Restart commands on the training host

Wait until the failed workers have exited; do not update a running checkout.
Keep failed logs and completed trial artifacts for diagnosis. Pull without
discarding local work:

```bash
git pull --ff-only origin new_ir_changes_20260331
nvidia-smi
```

Use your real model, source-data and existing cache paths. The following are
examples; **choose fresh output directories**, not the failed suite's folder.
If the old command explicitly set `--token-cache-dir`, retain that path too.

Ordinary E2B training plus Golden32/35 testing:

```bash
python -u training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v9 \
  --output-dir /runs/e2b-backward-fix-epoch1 \
  --devices auto --microbatch 1 --effective-batch 32 --epochs 1 \
  --preparation-cache-dir /runs/.golden-preparation-cache \
  --tensorboard-root /tensorboard --execute
```

Sequential HPO restart (new suite, not resume):

```bash
python -u training/scripts/run_golden_experiments.py \
  --profile e2b --model-dir /models/e2b \
  --input-dir /data/full_data_archive_recovered_v9 \
  --output-dir /runs/e2b-hpo-backward-fix-v1 \
  --trial-steps 1000 --devices auto --microbatch 1 --effective-batch 32 \
  --preparation-cache-dir /runs/.golden-preparation-cache \
  --tensorboard-root /tensorboard --execute
```

Keep any deliberately chosen learning-rate/trial-file/augmentation settings
from your original command. Augmentation remains optional, not a CUDA fix.
HPO still runs trials sequentially, selects on Golden32, then evaluates the
locked selected checkpoint once on Golden35. Default validation is every 500
updates; Golden32 generation every 1,000 plus final; generation limit 2,048.

Look for `SDPA runtime` with `"cudnn": false`, then successful
`SFT backward preflight` lines on each rank. A short smoke with `--steps 20`
on the ordinary launcher is still advisable before a long run. The smoke also
runs final Golden evaluation; it does not establish model quality or prove a
step-395 failure cannot recur.

If `cudaErrorContained` persists in a fresh process, stop and ask the cluster
administrator to inspect Xid/ECC/NVLink health or allocate a healthy node.
Read-only captures include `nvidia-smi -q` and `nvidia-smi topo -m`, together
with all rank logs and exact package versions. NVIDIA documents that contained
errors require [process termination and relaunch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html);
its [GPU troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
covers host/GPU checks. This patch does not reset shared GPUs, auto-retry an
unusable CUDA context or promise to repair hardware.

## Local verification

The full `training/tests` suite passed: **961 passed, 1 skipped**, in 248.59
seconds on Python 3.12 / Torch 2.13 CPU / Transformers 5.16.1 / PEFT 0.20.0.
Coverage includes backend-policy restoration, all 2/4/8 H100 profile mappings,
31 backward-probe tests, and real tiny Gemma4 + PEFT CPU integration with cold,
warm and disabled caches. The integration performs one actual optimizer update,
validation, checkpoint saving and TensorBoard logging after the new live probe.
The two suite warnings concern pinned memory without a CPU-test accelerator.

A second pass on the minimum Gemma4 dependency stack (Transformers 5.10.1,
PEFT 0.19.0) passed **68 tests, 1 skipped** in 40.11 seconds, covering the
backend policy, backward probe and real CPU SFT/cache/TensorBoard integration.
The skip is the integration assertion specific to Transformers 5.16.1's new
TensorBoard callback API. New helper modules/tests pass Ruff; critical syntax
and undefined-name checks pass across all changed Python files.

Only CPU and mocked CUDA-policy tests can run on this development host. No real
H100 training, NCCL run, Golden-model inference or throughput benchmark was run.
