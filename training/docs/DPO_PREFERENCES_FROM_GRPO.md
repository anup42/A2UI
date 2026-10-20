# Build DPO preferences from existing GRPO outputs

The offline builder supports **E2B and Gemma 3 270M QAT LoRA** GRPO runs. Run it
in the existing training environment on the host where the run and its bound
model/tokenizer and prepared data are available. It loads the tokenizer and CPU
fidelity scorer, **not model weights**, and makes no Muse or other generation
calls. Use a stopped/completed GRPO run so audit files are not being appended.
Interrupted runs can be useful if their preparation and configuration evidence
are intact.

## E2B

```bash
OLD_E2B_RUN=/absolute/path/to/previous_e2b_grpo_run
E2B_PREFERENCES=/data/e2b_grpo_preferences_v1

python -u training/scripts/build_dpo_preferences_from_grpo.py \
  --grpo-run-dir "$OLD_E2B_RUN" \
  --output-dir "$E2B_PREFERENCES" \
  --min-reward-gap 0.01 --min-pairs 32
```

After this succeeds, start DPO from the original SFT adapter:

```bash
python -u training/scripts/run_qat_grpo_pipeline.py \
  --rerun-from "$OLD_E2B_RUN" \
  --output-dir /runs/e2b_qat_dpo_from_grpo_v1 \
  --objective dpo --dpo-preferences "$E2B_PREFERENCES/preferences.jsonl" \
  --dpo-beta 0.1 --defer-native-eval --execute
```

## 270M

```bash
OLD_270M_RUN=/absolute/path/to/previous_270m_grpo_run
PREFERENCES_270M=/data/270m_grpo_preferences_v1

python -u training/scripts/build_dpo_preferences_from_grpo.py \
  --grpo-run-dir "$OLD_270M_RUN" \
  --output-dir "$PREFERENCES_270M" \
  --min-reward-gap 0.01 --min-pairs 32
```

After this succeeds, start DPO from the original SFT adapter:

```bash
python -u training/scripts/run_qat_grpo_pipeline.py \
  --rerun-from "$OLD_270M_RUN" \
  --output-dir /runs/270m_qat_dpo_from_grpo_v1 \
  --objective dpo --dpo-preferences "$PREFERENCES_270M/preferences.jsonl" \
  --dpo-beta 0.1 --defer-native-eval --execute
```

Both output directories must be new. The family and token budgets come from the
verified run; there is no model-family override. DPO starts from that run's
**original SFT adapter**, not the GRPO-trained adapter. The saved GRPO responses
provide offline alternatives only. If your effective batch is larger than 32,
set `--min-pairs` to at least that batch size; omitting this option uses the run's
effective batch automatically. See
[QAT_POSTTRAIN_OPTIONS.md](QAT_POSTTRAIN_OPTIONS.md) for the existing training,
export, and native-evaluation behavior. These commands do not change SFT or
default GRPO behavior.

## Training-only input and strict fidelity

The builder reads only `training/grpo_periodic_rewards.rank*.jsonl`, using
explicit `mode=train` records with exact training `source_id` and rendered
`prompt_sha256` bindings. It skips evaluation records and never mines Golden or
Bixby outputs. Validation and reserved-cohort identities are inspected only to
exclude leakage, not to select responses or calculate their preference rewards.
The older startup reward audit has no train/eval mode and is not a supported
substitute; neither are source-free rollout logs or native evaluation outputs.

Each raw completion is preserved unchanged: there is no tail stripping,
reference substitution in the output, truncation, or repair. Only recorded
`closing_sentinel`/`eos_token` termination is eligible; length-limited or
unterminated outputs are excluded. Current v5.4 scoring is recomputed against
the bound training source; old logged scores are not trusted. The highest-scoring
strictly faithful candidate becomes the chosen response, paired with the
highest-scoring lower-reward eligible alternative at the default gap of 0.01 on
the mapped **[-1, 1]** reward scale. There is at most one pair per source; repeated
responses across steps/ranks are deduplicated. Equal-score alternatives do not
become preferences. A malformed or unfaithful alternative may be a rejection,
but can never become the chosen supervised target. The small positive default
also retains faithful list-to-card extraction improvements, whose mapped reward
differences can be below 0.05; the absolute source-fidelity gates are unchanged.

All pairs then pass the same strict DPO admission used at training startup. On
success the output contains:

- `preferences.jsonl`: source-bound DPO input.
- `preference_build_report.json`: counts, exclusions, and input evidence.
- `preference_admission.json`: current scorer, fidelity, and token-budget proof.
- `preference_provenance.jsonl`: the selected pair's audit origins and scores.
- `candidate_diagnostics.jsonl`: per-candidate admission and exclusion reasons.

Periodic audits are a sample of training, not a full rollout archive. A run may
therefore have too few sources with both a fully faithful response and a usable
negative. The builder fails without publishing `preferences.jsonl` if fewer than
the requested minimum pairs survive; do not fill the shortage using test data,
relax fidelity, or relabel incomplete responses. Inspect the report and collect
additional **training-only** alternatives in a later GRPO run if necessary.
