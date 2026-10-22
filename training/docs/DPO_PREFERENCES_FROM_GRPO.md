# Build DPO preferences from existing GRPO outputs

The offline builder supports **E2B and Gemma 3 270M QAT LoRA** GRPO runs. Run it
in the existing training environment on the host where the run and its bound
model/tokenizer and prepared data are available. It loads the tokenizer and CPU
fidelity scorer, **not model weights**, and makes no Muse or other generation
calls. Use a stopped/completed GRPO run so audit files are not being appended.
Interrupted runs can be useful if their preparation and configuration evidence
are intact.

## Fresh rerun after the reward/admission fixes

Use commit **`9a293f62` or later** on the training host. Rebuild preferences in
a **new directory**, then start DPO in another **new directory**. Do not reuse
old admission reports or modify/resume an old run across the changed scorer
fingerprint. The builder re-scores the saved completions with the current
implementation; it does not generate new responses or edit old GRPO outputs.
This picks up the extraction-reward fix, empty-completion handling and
token-identical-pair rejection. Explicit warmup below also avoids inheriting an
SFT warmup inappropriate for a short DPO pilot.

Run the following in **Bash**, from the repository root, with the same verified
QAT GRPO training environment activated. Replace both old-run paths below with
the corresponding completed/stopped GRPO **pipeline output directory**, not an
adapter checkpoint directory. Keep its original SFT checkpoint/config, base
tokenizer, prepared data, receipts and periodic reward audits accessible at
their bound locations.

```bash
cd /group-volume/k.anup/code/A2UI || exit 1
git pull --ff-only origin new_ir_changes_20260331 || exit 1
git merge-base --is-ancestor 9a293f62 HEAD || exit 1

OLD_E2B_RUN=/absolute/path/to/previous_e2b_grpo_run
OLD_270M_RUN=/absolute/path/to/previous_270m_grpo_run
DPO_TAG=$(date +%Y%m%d_%H%M%S)
E2B_PREFERENCES=/group-volume/k.anup/working_dir/e2b_dpo_preferences_${DPO_TAG}
PREFERENCES_270M=/group-volume/k.anup/working_dir/270m_dpo_preferences_${DPO_TAG}
E2B_DPO_RUN=/group-volume/k.anup/working_dir/e2b_qat_dpo_${DPO_TAG}
DPO_270M_RUN=/group-volume/k.anup/working_dir/270m_qat_dpo_${DPO_TAG}
```

### Revalidate pair counts and choose the pilot budget

Define this read-only helper once in the same Bash session. It reuses canonical
DPO admission, checks the new file against both build reports, and prints only
`steps cadence` on stdout. It loads the tokenizer/scorer but no model weights.
It does **not** treat JSONL line counts or old logged rewards as valid pairs.
At the inherited batch size `B`, the first pilot uses
`min(model_cap, valid_pairs // B)` updates. If that is below the mandatory
20-update health window, stop and collect more **training-only** alternatives;
do not repeat a tiny set merely to reach 20 updates or weaken fidelity gates.

```bash
dpo_budget() {
  PYTHONPATH=training/src:dataset/src python - "$1" "$2" <<'PY'
import contextlib
import json
import sys
from pathlib import Path

from ir_training.train.dpo_preference_run import load_preference_run
from ir_training.train.qat_posttrain_data import prepare_dpo_pairs

old_run, preference_dir = map(Path, sys.argv[1:])
build = json.loads((preference_dir / "preference_build_report.json").read_text())
saved = json.loads((preference_dir / "preference_admission.json").read_text())
if build.get("status") != "passed":
    raise SystemExit("Preference build did not pass; do not start DPO.")
with contextlib.redirect_stdout(sys.stderr):
    context = load_preference_run(old_run)
    pairs, live = prepare_dpo_pairs(
        preference_dir / "preferences.jsonl", context["train_rows"],
        context["tokenizer"], **context["limits"],
        reward_config=context["reward_config"],
        forbidden_rows=context["forbidden_rows"],
        min_reward_gap=build["min_reward_gap"],
    )
if Path(build["run_evidence"]["source_run"]).resolve() != old_run.resolve():
    raise SystemExit("Preference build belongs to a different GRPO source run.")
if live != saved or live["pairs_sha256"] != build["preferences_sha256"]:
    raise SystemExit("Preference file/admission changed; rebuild into a fresh directory.")
if len(pairs) != build["pairs"] or len(pairs) != saved["rows"]:
    raise SystemExit("Preference count differs from the verified build reports.")
batch = context["effective_batch"]
cap = {"e2b": 100, "270m": 200}[context["family"]]
steps = min(cap, len(pairs) // batch)
print(f"{context['family']}: {len(pairs)} valid pairs; batch={batch}; "
      f"pilot_steps={steps}", file=sys.stderr)
if steps < 20:
    raise SystemExit(f"Need at least {20 * batch} admitted pairs for this batch size.")
print(steps, min(25, steps))
PY
}
```

## E2B: rebuild preferences, then launch DPO

```bash
python -u training/scripts/build_dpo_preferences_from_grpo.py \
  --grpo-run-dir "$OLD_E2B_RUN" \
  --output-dir "$E2B_PREFERENCES" \
  --min-reward-gap 0.01
```

After the builder succeeds, revalidate the budget and start from the original
SFT adapter. The `&&` chain prevents training if budget validation fails:

```bash
E2B_SCHEDULE=$(dpo_budget "$OLD_E2B_RUN" "$E2B_PREFERENCES") &&
read -r E2B_STEPS E2B_CADENCE <<< "$E2B_SCHEDULE" &&
python -u training/scripts/run_qat_grpo_pipeline.py \
  --rerun-from "$OLD_E2B_RUN" \
  --output-dir "$E2B_DPO_RUN" \
  --objective dpo --dpo-preferences "$E2B_PREFERENCES/preferences.jsonl" \
  --dpo-learning-rate 1e-6 --dpo-max-steps "$E2B_STEPS" \
  --dpo-warmup-ratio 0.03 --dpo-golden-every-steps "$E2B_CADENCE" \
  --dpo-beta 0.1 --defer-native-eval --execute
```

## 270M: rebuild preferences, then launch DPO

```bash
python -u training/scripts/build_dpo_preferences_from_grpo.py \
  --grpo-run-dir "$OLD_270M_RUN" \
  --output-dir "$PREFERENCES_270M" \
  --min-reward-gap 0.01
```

After this succeeds, start DPO from the original SFT adapter:

```bash
SCHEDULE_270M=$(dpo_budget "$OLD_270M_RUN" "$PREFERENCES_270M") &&
read -r STEPS_270M CADENCE_270M <<< "$SCHEDULE_270M" &&
python -u training/scripts/run_qat_grpo_pipeline.py \
  --rerun-from "$OLD_270M_RUN" \
  --output-dir "$DPO_270M_RUN" \
  --objective dpo --dpo-preferences "$PREFERENCES_270M/preferences.jsonl" \
  --dpo-learning-rate 2e-6 --dpo-max-steps "$STEPS_270M" \
  --dpo-warmup-ratio 0.03 --dpo-golden-every-steps "$CADENCE_270M" \
  --dpo-beta 0.1 --defer-native-eval --execute
```

Both output directories must be new. The family and token budgets come from the
verified run; there is no model-family override. DPO starts from that run's
**original SFT adapter**, not the GRPO-trained adapter. The saved GRPO responses
provide offline alternatives only. Omitting `--min-pairs` uses the run's
effective batch for the builder's minimum; the budget helper then enforces the
larger 20-update minimum before training. See
[QAT_POSTTRAIN_OPTIONS.md](QAT_POSTTRAIN_OPTIONS.md) for the existing training,
export, and native-evaluation behavior. These commands do not change SFT or
default GRPO behavior.

## Explicit DPO pilot schedule

The commands above explicitly replace only DPO's learning rate, optimizer-step
cap, warmup, and evaluation/checkpoint cadence. These are conservative starting
experiments, not model-specific validated optima:

| Model | Learning rate | Step cap | Warmup | Golden32 selection |
| --- | --- | --- | --- | --- |
| E2B QAT LoRA DPO | `1e-6` | Up to 100 | 3% (3 at cap) | Every `min(25, steps)` |
| Gemma 3 270M QAT LoRA DPO | `2e-6` | Up to 200 | 3% (6 at cap) | Every `min(25, steps)` |

`--dpo-warmup-ratio 0.03` resolves to `ceil(0.03 * max_steps)` integer
`warmup_steps` in the new generated config. It replaces an inherited SFT warmup
(for example, 500 steps); it never edits the checkpoint-bound SFT config. To set
an exact warmup instead, use `--dpo-warmup-steps 3`; `0` explicitly disables
warmup. Supply only one warmup flag. A warmup that consumes the entire resolved
training budget is rejected. Integer steps are used in the generated config so
the recipe does not depend on a Trainer version accepting `warmup_ratio`.

These flags are **opt-in and DPO-only**. Without them, existing commands retain
their previous inherited behavior. `--rerun-from` still rejects the generic
`--learning-rate`, `--max-steps`, and `--golden-every-steps` overrides, and
`--continue-run` remains immutable; overrides always create a fresh DPO run from
the original SFT adapter. The saved plan records both the requested DPO overrides
and the resolved schedule separately from the ordinary inherited options. Thus,
switching a later fresh run back to GRPO clears DPO overrides and restores its
original non-DPO schedule rather than leaking the DPO pilot settings into GRPO.
When replacing an inherited warmup type on a later DPO rerun, the explicit new
type clears the other inherited type.

An inherited DPO `warmup_ratio` is also converted to integer steps at runtime
when the installed Trainer no longer accepts ratios. This requires a positive
production `max_steps`; an unknown horizon fails clearly instead of silently
disabling warmup. Explicit `warmup_steps`, including zero, take precedence.
The disposable two-update DPO probe uses the production warmup horizon rather
than recomputing the ratio against its shortened duration. Shared SFT and GRPO
scheduler handling is unchanged.

The baseline example is effective batch **32 preference pairs**; the rerun and
budget helper inherit the actual original batch, seed, token limits, QAT
contract, fidelity checks, and other settings. Do not add `--effective-batch`
or `--devices` to `--rerun-from`; those overrides are rejected. Use a new
explicit plan if those settings must change.
At batch 32, one epoch is approximately `admitted_pairs / 32` optimizer steps.
Use the smaller of one epoch and the suggested cap for the first experiment,
adjusting `--dpo-max-steps` explicitly. Do not use repeated passes through a tiny
preference set just to reach the minimum 20-update health window; collect more
training-only pairs instead. At batch 32, reaching the E2B 100-step cap requires
3,200 pairs; reaching the 270M 200-step cap requires 6,400. The commands above
automatically use fewer steps for smaller valid sets (at least 640 pairs for
20 steps at batch 32). These are conservative exposure budgets, not guarantees
that Trainer will visit every unique pair exactly once.

Golden32 selects the checkpoint under the unchanged quality/fidelity gates.
Golden35 and Bixby50 remain final-only. Successful training then runs the existing
selected-checkpoint evaluation, merge, and LiteRT-LM export stages: E2B uses the
official retained mixed-precision topology, and 270M uses the official INT8
topology. `--defer-native-eval` postpones only the Android exported-model test,
not HF evaluation or export. See the native-evaluation commands in
[QAT_POSTTRAIN_OPTIONS.md](QAT_POSTTRAIN_OPTIONS.md); an export is not a native
quality pass.

For old runs created with prepared input only, add
`--input-dir /absolute/path/to/original_raw_pairs` containing both `train.jsonl`
and `val.jsonl`. Fresh preparation and source/preference provenance checks remain
mandatory.

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
unterminated outputs are excluded. Empty completed responses are also excluded
and counted, rather than aborting an otherwise usable audit. Malformed audit
fields and unsupported termination metadata still fail closed.
Current v5.4 scoring is recomputed against
the bound training source; old logged scores are not trusted. The highest-scoring
strictly faithful candidate becomes the chosen response, paired with the
highest-scoring lower-reward eligible alternative at the default gap of 0.01 on
the mapped **[-1, 1]** reward scale. There is at most one pair per source; repeated
responses across steps/ranks are deduplicated. Equal-score alternatives do not
become preferences. A malformed or unfaithful alternative may be a rejection,
but can never become the chosen supervised target. The small positive default
also retains faithful list-to-card extraction improvements, whose mapped reward
differences can be below 0.05; the absolute source-fidelity gates are unchanged.

All pairs then pass the same strict DPO admission used at training startup.
Chosen and rejected responses must have distinct token sequences, not merely
different strings; token-identical pairs provide no learnable preference and
are rejected even when supplied outside the builder. On success the output
contains:

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
