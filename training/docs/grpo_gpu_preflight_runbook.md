# Optional GRPO: GPU-PC preflight and smoke gates

Do SFT and checkpoint evaluation first for E2B and 270M separately. GRPO is optional, and a seed whose candidates all fail the strict scorer provides no useful relative reward. These commands are for the GPU PC. No training, checkpoint inference, or remote-host access was performed while implementing these changes.

## Transfer and prerequisites

Transfer the final repository code, validated prepared JSONL with its original `messages`, tokenizer files, selected SFT weights and manifests, and the scorer dependencies under `dataset/`. Use a dedicated **verified SFT seed** for GRPO: either a merged LoRA checkpoint with `qat_mtp_merge_metadata.json`, or a full-finetuned checkpoint (including 270M) with `training_metadata.json`. Local artifact filenames, sizes and hashes must match the manifest. A full-finetuned seed does not require a LoRA merge. For a merged seed, compare adapter-plus-original-base and merged generation on identical prompt IDs first; an adapter-only directory is rejected. Artifact hash verification alone does not establish numerical generation parity, including for historical merge metadata with `training_run_metadata.verified=false`.

Use a few hundred diverse source groups from training for the initial smoke, including short/long responses, tables, actions, state, and complex references. Preserve the original source-disjoint dev and final test sets; Golden 32 is a development set. Do not repair candidate IDs, relax score caps, or truncate a target to make the smoke pass.

Create a dedicated GPU environment using the GPU host's supported Torch/CUDA installation. The rollout integration is source-reviewed against **TRL 0.29.1**, including its `rollout_func`, `env_mask`, and scalar-EOS truncation behavior. This is not a claim that the whole stack was tested on GPU. Pin TRL in this environment, and retain the resolved dependency lock after testing:

```bash
python -m pip install -r training/requirements-training.txt 'trl==0.29.1'
# E2B also requires a Transformers release supporting its actual architecture;
# use the repository's Gemma 4 dependency requirements for that environment.
python -m pip freeze > grpo-gpu-requirements.lock.txt
```

Do not copy the historical `grpo_seed/pylibs/trl` override or modify installed library source. Run from the transferred repository root. `grpo_dependencies.rank*.json` records the **imported** versions, distribution versions, import paths, source hashes, and install provenance; it detects a TRL import outside its installed distribution. A newer TRL release requires a fresh source/mask review and tests, rather than bypassing this check.

## Dependency check: no model loading

Set paths to the GPU PC's actual files. `PILOT` must contain the saved SFT messages, source text, and validated completions. `SFT_SEED` must contain the checkpoint's tokenizer, generation config and SFT provenance.

```bash
SFT_SEED=/absolute/path/to/verified-sft
PILOT=/absolute/path/to/grpo-pilot/train.jsonl

python training/scripts/train_grpo.py \
  --sft-checkpoint "$SFT_SEED" --dataset "$PILOT" \
  --output-dir runs/grpo_dependency_preflight \
  --dependency-preflight-only
```

This exits before tokenizer/model loading. Review the dependency report on failure. `--help` and helper unit tests also work without TRL/datasets installed:

```bash
python -m pytest training/tests/test_grpo_runtime.py -q
```

## Prompt and decoder contract

The default `--prompt-source prepared` uses **all saved SFT messages** up to the final assistant. It validates the final user/source and assistant/target binding. It does not append a GRPO-only asset policy. Pass `--chat-template-kwargs path.json` if SFT used nondefault `model.chat_template_kwargs`; the JSON must reproduce those exact values. Keep the same tokenizer/chat template as the verified SFT seed.

Reward uses the same restored source URLs, assets and persisted `expected_ui_contract_v5_4` as Golden evaluation. Prepared prompts keep their original masked URLs. Completion URLs are restored after the serving stop boundary, and raw completion evidence remains unchanged.

Do not pass `--prompt-template` or `--chat-scaffold` in prepared mode. Explicit reconstruction is available as `--prompt-source template`, with those flags, and is rejected if the reconstructed complete prompt differs from saved messages. All rendered prompts are tokenized with `add_special_tokens=False`; complete prompt and token-ID hashes are persisted, not just a six-token preview.

The HF rollout preserves native EOS IDs and stops at an unquoted `</a2ui>` in **generated tokens**. A quoted close tag or one inside the few-shot prompt cannot stop generation. A batch gets one call to the unwrapped model; HF batch padding is removed without rewriting sampled tokens. If a closing tag ends inside a token, trailing characters in that token remain in the raw token/text evidence; reward uses the same closing-tag text boundary as serving evaluation. No graph or ID repair is applied.

TRL 0.29.1 otherwise labels closing-tag stops and alternate native EOS IDs as clipped. The adapter appends one **PAD runtime sentinel with `env_mask=0`** to terminated rows. That sentinel is excluded from policy loss and model-token length; it is not a fabricated model EOS or a target. Truly clipped rows remain clipped and `mask_truncated_completions=True` masks their loss. Reserve one extra context token for this representation. Raw sampled IDs, runtime IDs, mask, stop reason and synthetic-sentinel count are saved separately.

`--max-prompt-length` is a validation ceiling, never permission to truncate a chat prefix. The longest prompt across both partitions plus completion budget and sentinel must fit the model context. If no completion limit is supplied, the loader estimates a budget from validated target lengths; inspect that value before a long run.

## Single-device learning smoke

Choose LoRA targets explicitly. For E2B, the prior control used language-only q/v projections; inspect `matched_lora_modules` in the preflight manifest. Do not target unused audio/vision branches. This example retains the GRPO script's existing rank32/alpha64, LR5e-6, beta0, and unmodified v5.4 reward weight1. These settings are pilots, not proven optima.

The GPU path loads the model through the shared HF loader and resolves selectors to actual `nn.Linear` module names before PEFT wrapping, including projections inside `.linear` wrappers. These examples use the same `auto_causal_lm` loader as the seed and explicitly load bfloat16 weights; use `auto_multimodal_lm` only when that was the seed's validated loader. Omitting `--model-dtype` preserves the prior TRL float32 weight load with bfloat16 AMP enabled by `--bf16`. Attention remains model-selected unless `--attn-implementation` is supplied. Loading details and exact resolved targets are logged; the loader rejects missing, unexpected or mismatched checkpoint keys.

```bash
CUDA_VISIBLE_DEVICES=0 python training/scripts/train_grpo.py \
  --sft-checkpoint "$SFT_SEED" --dataset "$PILOT" \
  --model-loader auto_causal_lm --model-dtype bfloat16 \
  --output-dir runs/grpo_e2b_single_smoke \
  --lora-target-modules 'regex:.*language_model\..*\.(q_proj|v_proj)(\.linear)?' \
  --num-generations 4 --per-device-batch-size 1 \
  --gradient-accumulation-steps 8 --max-steps 20
```

For a separately validated 270M seed, use a distinct output directory and the appropriate actual language projection suffixes, e.g. `--lora-target-modules q_proj,v_proj`; establish that model's own candidate diversity and quality. SFT/distillation remains the first choice for 270M.

The default rolling 20-update health gate requires:

- More than 25% of candidate groups have nonzero reward variance.
- Completion clipping is at most 5%.
- More than 25% of optimizer updates have nonzero finite gradients, and more than 25% have a measured change in sampled trainable parameters on every rank.
- No nonfinite numeric log, parameter or gradient values. Missing required metrics fail instead of silently passing.

Threshold flags are explicit (`--health-window-steps`, `--health-min-diverse-groups`, `--health-max-clipped`, `--health-min-nonzero-gradients`, `--health-min-nonzero-updates`). They are engineering gates, not accuracy guarantees. A run ending before one complete passing window is inconclusive and exits as failure. Do not lower thresholds simply to accept the historical reward=-1/std=0/zero-gradient pattern.

## Two-device smoke, then scale

Only proceed after the single-device smoke passes. Select the actual GPUs in one place and preserve the effective generated-sample batch: 1 × 4 accumulation × 2 ranks = 8. Use the same varied-length pilot, seed, decoder and reward contract.

```bash
CUDA_VISIBLE_DEVICES=0,1 TORCH_DISTRIBUTED_DEBUG=DETAIL \
  torchrun --standalone --nproc_per_node=2 training/scripts/train_grpo.py \
  --sft-checkpoint "$SFT_SEED" --dataset "$PILOT" \
  --model-loader auto_causal_lm --model-dtype bfloat16 \
  --output-dir runs/grpo_e2b_ddp_smoke \
  --lora-target-modules 'regex:.*language_model\..*\.(q_proj|v_proj)(\.linear)?' \
  --num-generations 4 --per-device-batch-size 1 \
  --gradient-accumulation-steps 4 --max-steps 20
```

The reviewed path uses non-reentrant gradient checkpointing, explicitly restores its options after TRL's generation unwrap, and sets `ddp_broadcast_buffers=False`. It uses full batches and fixed-shape health gathers at optimizer boundaries. Every rank returns an `env_mask` even for all-clipped groups; it never skips the forward/loss path based on local reward. Buffer checksums must match across ranks before training, and buffer contents must remain static at health-window boundaries when broadcasts are disabled. A mutable-buffer failure needs model-specific investigation; it must not be dismissed by increasing NCCL timeouts. vLLM, FSDP and DeepSpeed are outside this reviewed rollout and are rejected.

This narrows the historical ALLGATHER/BROADCAST mismatch risk; **it does not certify an NCCL fix**. Inspect both rank logs, compare step sequences, check no hangs/collective mismatch, and verify finite adapter change. Re-evaluate the saved smoke checkpoint using the same strict Golden/dev decoder and verify source facts/actions before running longer. Only after both smokes pass should the same configuration run more updates in a new output directory.

## Evidence to retain

- `grpo_dependencies.rank*.json`: actual dependency versions and source origins/hashes.
- `grpo_preflight.rank*.json`: full prompt/token fingerprints, EOS, LoRA module inventory, mask/decoder contract, thresholds and baseline reward parameters.
- `grpo_health.rank*.jsonl`: every optimizer-step health record and rolling gate decision.
- `grpo_rollouts.rank*.jsonl`: bounded raw sampled/runtime token evidence, input token IDs, mask and stop reason.
- `grpo_rewards.rank*.jsonl`: source/prompt-group identity, unchanged reward and scorer-provided breakdown. The default audit limit is 256 candidates per rank; increase `--audit-rollout-limit` when a longer diagnostic needs more evidence.
- Dependency lock, original SFT/merge provenance, both rank console logs, saved adapter, and fresh strict evaluation outputs. Sampled parameter change is a liveness check; it does not establish better model quality.

Source review: [TRL 0.29.1 trainer](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_trainer.py), [TRL 0.29.1 configuration](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_config.py), and [PyTorch DDP contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). GPU runtime validation remains pending on the other PC.
