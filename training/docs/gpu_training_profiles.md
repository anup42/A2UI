# Automatic GPU launch profiles

`prepare_review_training.py` now inspects the GPU host and uses every CUDA-visible device by default. It reads GPU names, capacity, compute capability and UUID where available; it does not load model weights or start training. Run preparation inside the same scheduler/container environment used for execution.

These are conservative starting profiles, not measured maximum-throughput settings. Keep the same learning rate when changing GPU count. Validate peak memory, optimizer updates, tokens per second and Golden quality in the 20-step smoke before scheduling a long run.

| Model | H100 GPUs, each at least 70 GiB | Per-GPU microbatch | Accumulation | Global batch |
|---|---:|---:|---:|---:|
| E2B | 2 | 2 | 8 | 32 |
| E2B | 4 | 2 | 4 | 32 |
| E2B | 8 | 2 | 2 | 32 |
| 270M | 2 | 4 | 4 | 32 |
| 270M | 4 | 4 | 2 | 32 |
| 270M | 8 | 4 | 1 | 32 |

Other GPUs, mixed GPU types and smaller H100/MIG partitions start with microbatch 1 and global batch 16. Every selected GPU must divide the requested global batch; unusual GPU counts require an explicit compatible `--effective-batch`. `--effective-batch 16` preserves an older experiment's batch and caps the automatic microbatch where necessary. `--microbatch` overrides the per-GPU choice; incompatible combinations fail before training. Do not compare a new global-batch-32 run with a global-batch-16 run as if only GPU count changed.

BF16 is selected on devices with native BF16 support, otherwise FP16. The profile uses SDPA, enables TF32 on supported hardware, keeps non-reentrant gradient checkpointing, and bounds dataloader workers by available CPUs and worker count. Persistent workers and prefetch are enabled only when the worker count is positive. Override with `--attn-implementation eager`, `--no-gradient-checkpointing` or `--dataloader-workers`; measure any performance change rather than assuming it helps.

`--devices auto` preserves the inherited `CUDA_VISIBLE_DEVICES` mask, including GPU and MIG UUIDs. Explicit numeric selections are **logical indices inside that visible mask**, not physical machine indices. With `CUDA_VISIBLE_DEVICES=2,5`, `--devices 1` selects physical GPU 5 and passes `CUDA_VISIBLE_DEVICES=5` to the workers. Exact visible UUIDs are also accepted. The launcher never broadens the scheduler's visibility.

Execution rechecks visible count, mask, selected identities, GPU model and total capacity. A changed host or scheduler assignment requires a new prepared plan. This prevents an eight-GPU or 80-GB plan from silently launching with fewer or smaller GPUs. Plan printing remains possible without initializing GPUs; preparing a new bound plan requires the actual GPU host.

Logs default to `/tensorboard/<run-id>/training`; full Golden evaluation records are under the same run root. `A2UI_TENSORBOARD_ROOT` overrides this root for both training and evaluation. The launcher propagates that root to its workers.

Validation loss and Trainer checkpoints default to every 500 optimizer updates; full Golden generation defaults to every 1,000, followed by mandatory final-weight evaluation. Set `--eval-steps` and `--golden-every-steps` to adjust them; the latter must be an integer multiple of the former. A bounded smoke shorter than the cadence evaluates its final weights. Golden generation defaults to 2,048 new tokens; `--max-new-tokens` overrides the budget. Token-limit failures remain failures in quality reports, and inputs are never shortened to satisfy a budget.

The capability review profiles and retained-scale official mobile QAT pipeline are separate contracts. These automatic performance choices do not establish official mobile export parity or jointly trained MTP behavior.

The canonical multiformat entry points use the same detector when actual training is requested. Gemma270M `--preflight-training` and `--execute-training` now launch `torch.distributed.run` with one worker per selected visible GPU; `--devices`, `--microbatch` and `--effective-batch` provide explicit overrides. The resolved training config binds the hardware profile, and subsequent export/evaluation plans reuse those saved settings without probing GPUs.

The official E2B mobile launcher uses all visible GPUs when `--num-gpus` is omitted during `--preflight` or `--execute`. Its `--gpu-ids` option retains physical/UUID launch identifiers, which must belong to the inherited visible mask; unlike the review launcher's `--devices`, it does not reinterpret masked numeric IDs as logical indices. The E2B multiformat entry point resolves all visible GPUs for its training stage as well. The retained QAT scale, LoRA scope, seed and quantization contracts remain unchanged. Offline plans keep an illustrative single worker unless an explicit count is supplied; they do not import model weights or silently probe a remote host.
