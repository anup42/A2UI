# QAT LoRA post-training: GRPO rehearsal and offline DPO

These are separate, opt-in experiments for **E2B and Gemma 3 270M** starting
from an existing verified **QAT LoRA SFT adapter**. They share the established
preparation, QAT, Golden selection, merge, official export and native quality
gates. Ordinary SFT, full-parameter QAT and default GRPO behavior are unchanged.
The internal workflow remains `qat_lora_grpo_v1` for export compatibility;
the explicit objective and training metadata distinguish the algorithms.

| CLI objective | Optimization | Effective batch counts |
| --- | --- | --- |
| `grpo` (default) | Existing Dr. GRPO, beta 0 | Generated completions |
| `grpo_sft` | Existing GRPO plus weighted supervised rehearsal | Generated completions; rehearsal is an additional loss |
| `dpo` | Offline chosen/rejected preference optimization against frozen SFT | Preference pairs, not completions or rollout groups |

`--sft-rehearsal-weight` defaults to **0.1** only for `grpo_sft`.
`--dpo-beta` defaults to **0.1** only for `dpo`. Both must be positive and
finite; supplying another objective's options is an error. These are starting
settings, not demonstrated optimal values or evidence of improved accuracy.

## Reuse an existing run's original SFT inputs

Activate the existing reviewed training environment, then run from the repository
root. `--rerun-from` reads the previous plan's **original SFT adapter**, never
the previous GRPO best adapter. It preserves seed, token budgets, effective batch
and ordinary hyperparameters while preparing raw pairs afresh. DPO can explicitly
override its learning rate, step cap, warmup and selection cadence using the
DPO-only flags below; omitting them preserves the previous inherited behavior.
Each output directory must be new. Nothing here launches Muse or creates data.

For **E2B**, using the previous E2B GRPO plan:

```bash
OLD_E2B_RUN=/absolute/path/to/previous_e2b_grpo_run

python -u training/scripts/run_qat_grpo_pipeline.py \
  --rerun-from "$OLD_E2B_RUN" \
  --output-dir /runs/e2b_grpo_sft_rehearsal_v1 \
  --objective grpo_sft --sft-rehearsal-weight 0.1 \
  --defer-native-eval --execute

python -u training/scripts/run_qat_grpo_pipeline.py \
  --rerun-from "$OLD_E2B_RUN" \
  --output-dir /runs/e2b_qat_dpo_v1 \
  --objective dpo --dpo-preferences /data/e2b_preferences.jsonl --dpo-beta 0.1 \
  --dpo-learning-rate 1e-6 --dpo-max-steps 100 \
  --dpo-warmup-ratio 0.03 --dpo-golden-every-steps 25 \
  --defer-native-eval --execute
```

For **270M**, use that family's previous plan and source-bound preference file:

```bash
OLD_270M_RUN=/absolute/path/to/previous_270m_grpo_run

python -u training/scripts/run_qat_grpo_pipeline.py \
  --rerun-from "$OLD_270M_RUN" \
  --output-dir /runs/270m_grpo_sft_rehearsal_v1 \
  --objective grpo_sft --sft-rehearsal-weight 0.1 \
  --defer-native-eval --execute

python -u training/scripts/run_qat_grpo_pipeline.py \
  --rerun-from "$OLD_270M_RUN" \
  --output-dir /runs/270m_qat_dpo_v1 \
  --objective dpo --dpo-preferences /data/270m_preferences.jsonl --dpo-beta 0.1 \
  --dpo-learning-rate 2e-6 --dpo-max-steps 200 \
  --dpo-warmup-ratio 0.03 --dpo-golden-every-steps 25 \
  --defer-native-eval --execute
```

If the previous plan used only a prepared bundle, add
`--input-dir /absolute/path/to/original_training_pairs` containing original
`train.jsonl` and `val.jsonl`. Omit `--execute` for a read-only plan. To return
to plain GRPO explicitly use `--objective grpo`; switching modes clears old
mode-specific settings. Ordinary hyperparameter overrides on `--rerun-from`
are still rejected rather than silently ignored.

These DPO pilot schedules use 3/6 integer warmup steps respectively, replacing
any inherited SFT warmup only in the new DPO config. Use either
`--dpo-warmup-ratio` or `--dpo-warmup-steps` (including zero), never both.
`--continue-run` cannot change schedules. The original SFT config/checkpoint and
non-DPO options remain unchanged. See
[the pilot schedule and pair-count guidance](DPO_PREFERENCES_FROM_GRPO.md#explicit-dpo-pilot-schedule)
before choosing a step cap for a small preference set; these are starting
experiments, not validated optimal settings.

## Start directly from SFT without a previous GRPO plan

Use the matching base, exact checkpoint-bound SFT YAML and SFT adapter. A full
fine-tuned dense checkpoint or a merged SFT model is **not** an adapter input.
The example below is E2B rehearsal:

```bash
python -u training/scripts/run_qat_grpo_pipeline.py \
  --family e2b --objective grpo_sft --sft-rehearsal-weight 0.1 \
  --model-dir /models/e2b-retained-mobile-seed \
  --sft-config /runs/e2b-sft/configs/original-resolved-training.yaml \
  --sft-checkpoint /runs/e2b-sft/training/best_golden_checkpoint \
  --input-dir /data/completed-training-pairs \
  --official-litertlm /models/official-e2b-mobile.litertlm \
  --source-safetensors /models/official-e2b-mobile/model.safetensors \
  --exporter-python /venvs/litert-export/bin/python \
  --output-dir /runs/e2b_grpo_sft_smoke \
  --devices auto --max-steps 20 --golden-every-steps 20 \
  --microbatch 1 --effective-batch 32 --num-generations 4 \
  --max-seq-length 4096 --max-input-tokens 5120 --max-new-tokens 2048 \
  --defer-native-eval --execute
```

For DPO, replace `--objective grpo_sft --sft-rehearsal-weight 0.1` with
`--objective dpo --dpo-preferences /data/preferences.jsonl --dpo-beta 0.1`;
use a fresh output directory. `--num-generations` does not control offline DPO
batching; the same effective batch number means **pairs** in DPO. Therefore
equal step and batch counts do not imply equal token or GPU-hour budgets.

The matching **270M direct-from-SFT** command is:

```bash
python -u training/scripts/run_qat_grpo_pipeline.py \
  --family 270m --objective grpo_sft --sft-rehearsal-weight 0.1 \
  --model-dir /models/gemma-3-270m-it \
  --sft-config /runs/270m-sft/configs/original-resolved-training.yaml \
  --sft-checkpoint /runs/270m-sft/training/best_golden_checkpoint \
  --input-dir /data/completed-270m-training-pairs \
  --official-litertlm /models/official-gemma-3-270m-it-int8.litertlm \
  --exporter-python /venvs/litert-export/bin/python \
  --output-dir /runs/270m_grpo_sft_smoke \
  --devices auto --max-steps 20 --golden-every-steps 20 \
  --microbatch 1 --effective-batch 32 --num-generations 4 \
  --max-seq-length 4096 --max-input-tokens 5120 --max-new-tokens 2048 \
  --defer-native-eval --execute
```

For 270M DPO, replace the rehearsal flags with
`--objective dpo --dpo-preferences /data/270m_preferences.jsonl --dpo-beta 0.1`
and use a new output directory. These paths are examples: the SFT YAML must be
the exact file bound in your checkpoint, not a newly recreated YAML. Both
objectives support single-GPU/DDP, not DeepSpeed/FSDP. After the twenty-update
smoke passes all quality gates, start a fresh experiment with your chosen longer
budget, for example `--max-steps 200 --golden-every-steps 50`.

## Preference data and fidelity

For saved GRPO outputs, follow the
[fresh preference rebuild and budget-aware DPO commands](DPO_PREFERENCES_FROM_GRPO.md#fresh-rerun-after-the-rewardadmission-fixes).
They apply the current reward/admission fixes, revalidate the pair count, and
start a fresh run from the original SFT adapter without using holdout outputs.

Prepare DPO preferences **offline**, before launching training. JSONL rows use
`source_id`, `chosen` and `rejected`; the latter two are completion strings, not
full chat transcripts. The chosen completion must pass strict Express and
full-fidelity validation; the rejected alternative may contain schema errors or
missing facts. Source identity must resolve to the sealed
prepared **training** split, not Golden32, Golden35, Bixby50 or validation.
For example, the structural format is:

```json
{"source_id":"EXACT_TRAINING_SOURCE_ID","chosen":"VALIDATED_PREFERRED_EXPRESS","rejected":"LOWER_QUALITY_EXPRESS"}
```

Those capitalized values are placeholders, not usable training examples. Use
real generated/reviewed alternatives for the exact same source. Correct field
extraction, number preservation and action/link ownership matter; adding cards
alone is not a preference. Admission reuses source fidelity and strict validation
instead of trusting an external preferred label, and requires a strictly better
chosen reward (ties are rejected). Optional `response_text_sha256`,
`source_context_sha256` and `prompt_sha256` bindings must match if supplied.
Unknown/duplicate/leaked sources, unfaithful preferred candidates, incompatible
references, and over-budget candidates
must not be made acceptable by editing validators or lowering thresholds. Invalid
preference input fails the job; it is not silently filtered into a smaller run.

The preference file is SHA-256-bound in the plan, generated config and configure
receipt, rechecked before stage work and post-training continuation. Store it
outside the run output and do not modify it while the run is active. Changing
preferences or the objective requires a fresh run. Rehearsal consumes the already
prepared, validated training targets; it never retrieves examples from holdouts.
With effective batch 32, supply at least 32 admitted DPO pairs. No model call is
made to generate or repair preferences during training. Reference scoring is a
one-time pre-training scan of the initial SFT+QAT model on each rank; it requires
extra startup compute but no second resident model. Rehearsal adds a supervised
forward/backward graph, so its combined real-Trainer memory preflight is required.

## Gates, export and evidence

- QAT stays active for the trainable policy and the appropriate supervised or
  frozen-SFT reference calculations. The frozen DPO reference is not the
  adapter-disabled base model.
- Each run retains disposable real-Trainer preflight, finite checks, original
  SFT provenance, scorer identity and Golden32 eligible-best selection.
  Golden35/Bixby50 remain final-only holdouts.
- E2B still uses retained W2/W4/W8 A8 official topology; 270M still uses its
  official INT8 topology. This does not add MTP training.
- `--defer-native-eval` exports on the GPU host and leaves Android testing
  pending, never passed. Run the emitted `native_quality_command.json` on the
  Android host to perform the unchanged three-cohort native quality gate.
- Safe post-training continuation uses `--continue-run /runs/your_new_run`;
  it does not resume optimizer state or accept changed preference files/code.
- Review the twenty-update smoke before committing to a longer experiment.
  Software tests are not measurements of multi-GPU memory, quality or Android
  latency. Compare against the same SFT seed and match actual training tokens or
  compute budget, not merely step counts across objectives.

## Test the exported LiteRT-LM on Android

The GPU-host examples intentionally use `--defer-native-eval`: selected-checkpoint
HF evaluation and export still run, but Android quality is **pending**, not
passed. On the host with the connected Android device and preinstalled matching
app/instrumentation APKs, inspect the new run's `native_quality_command.json`.
The equivalent standalone command for either E2B or 270M is:

```bash
DPO_RUN=/absolute/path/to/the_new_dpo_pipeline_output
ANDROID_SERIAL=YOUR_CONNECTED_ANDROID_SERIAL

python -u training/scripts/evaluate_official_mobile_native.py \
  --run-dir "$DPO_RUN" \
  --output-dir "${DPO_RUN}_native_quality" \
  --weights-config "$DPO_RUN/configs/grpo_metric_weights.yaml" \
  --adb adb --serial "$ANDROID_SERIAL" --execute
```

Use a new, disjoint native report directory. Keep the complete bound run bundle
(not just the `.litertlm` file), preference/provenance evidence and required
external artifacts accessible; do not rewrite hashes or receipts to move a run.
Use the emitted command's scorer config, not generic default weights. This
unchanged test covers Golden32, Golden35 and Bixby50 with MTP disabled and checks
GPU delegation plus native quality gates. It does not install APKs or retrain
the model. Standalone results are written to the native report directory; they
do not rewrite the deferred training plan or turn `--continue-run` into an
optimizer resume.
