# Regenerate affected v11 supervision with Muse

This is an **opt-in, offline dataset operation**, not a training-start hook.
It generates new Stage 3 labels through the existing Muse/Express pipeline.
It never edits the original v11 release, its sources, or a trained checkpoint.
Existing SFT/QAT entry points, generation defaults and fidelity thresholds are
unchanged. Golden/Bixby data must not be supplied for regeneration.

## Changes and boundaries

- Entity extraction distinguishes `Hotel: Cedar` from `Hotel: Elm` rather than
  treating both identities as `Hotel`. The opt-in GRPO policy recognizes bounded,
  unambiguous labeled semicolon/pipe/comma records and heading/field groups.
- Extraction credit requires exact entity/field association, qualifiers and
  action destinations. A whole record copied into a Card is not extraction.
  Ordinary steps/checklists do not require Cards. Compact Tables are certified
  conservatively; Tables with actions are not yet certified by this entity policy.
- Only this command adds `dataset/prompts/v11_regeneration_v1.md` to canonical
  Stage 3. Schema, reference restoration, renderer and training-admission checks
  remain in force. This command explicitly selects the same source-bound entity
  policy in both Stage 3's canonical scorer and final admission, with unchanged
  thresholds; it does not waive a failed score after the fact. Existing callers
  retain the default policy. Candidate admission additionally checks field ownership,
  literal fidelity and unsupported visible prose. No global token/number-bag
  override is used. A literal-audit ordering false positive can be resolved only
  by an exact structured certificate for that same source unit; canonical
  admission cannot be waived.
- Ten audited source defects have content-hash-bound holds in
  `dataset/config/v11_source_quality_holds_20261023.json`. Stage 3 must not invent
  corrected source facts. Unlisted sources are **not** factually certified.

Recognition is bounded, not a general semantic parser. Unsupported/ambiguous
grammars get no inferred extraction certificate. Review strict rejections instead
of lowering thresholds. Changed scoring policy requires fresh scoring/preparation
identities: old reward caches and preference manifests are not current evidence.

## Commands on the Muse host

Use the existing dataset environment and an already-running Muse-Glimmer server.
Its actual context capacity must match the declared capacity. This command does
not launch training or another server on the training GPUs.

```bash
cd /group-volume/k.anup/code/A2UI
export LOCAL_VLLM_ENDPOINTS=http://127.0.0.1:30000/v1/chat/completions
export LOCAL_VLLM_SERVED_MODEL=muse-glimmer
SOURCE=training/data/train/v11
DEST=/group-volume/k.anup/working_dir/a2ui_v11_regenerated_v1
```

The checked repository gzip bundle is read directly; a restored directory with
`train.jsonl` and `val.jsonl` also works. v11 already contains v11s: do not join them.

Read-only plan (no output directory creation or Muse call):

```bash
python -u dataset/scripts/regenerate_v11_samples.py \
  --input-dir "$SOURCE" --output-dir "$DEST"
```

This scans/scores all input on CPU and may take time for 204,303 rows. To explicitly
generate the full corrected dataset:

```bash
python -u dataset/scripts/regenerate_v11_samples.py \
  --input-dir "$SOURCE" --output-dir "$DEST" \
  --batch-size 8 --max-attempts 3 \
  --prompt-max-tokens 16000 --max-tokens 12288 --context-tokens 32768 \
  --execute
```

For an initial pilot, use a **different fresh output** and add `--max-samples 8`.
This limits selected target regeneration, not input integrity/source-hold checks.
A pilot with deferred targets publishes `pilot_train.jsonl`/`pilot_val.jsonl`,
not normal training inputs; its manifest says `pilot_partial` and
`training_input_ready=false`. Copied, unassessed rows have not been corrected.
Inspect candidates/rejections before starting the full run.

Resume with identical flags, environment, source, code and prompts, plus `--resume`:

```bash
python -u dataset/scripts/regenerate_v11_samples.py \
  --input-dir "$SOURCE" --output-dir "$DEST" \
  --batch-size 8 --max-attempts 3 \
  --prompt-max-tokens 16000 --max-tokens 12288 --context-tokens 32768 \
  --execute --resume
```

Valid completed work is reused. Rejected/incomplete work retries only within the
original attempt budget, with fidelity feedback. Changed source/code/prompts/config
require a fresh output; increasing the budget is not an in-place resume.

## Outputs and training handoff

- `train.jsonl` / `val.jsonl`: preserve original split membership/IDs; unselected
  rows are byte-preserved, accepted selected rows contain regenerated labels.
- `quarantine.jsonl`: source-held rows and exhausted/rejected candidates, reasons
  and original provenance. Failure never falls back to the old defective target.
- `manifest.json`: counts, input/output hashes, audit scope and contract identity.
  A completed run does not mean every selected row was repaired.
- `contract.json`, `journal/`, `attempts/`: durable resume/generation evidence.

Inspect the manifest/quarantine and regenerated samples, then substitute this
argument in the existing E2B/270M training command, using a **new training output
and preparation directory** with normal safety gates:

```bash
--input-dir /group-volume/k.anup/working_dir/a2ui_v11_regenerated_v1
```

Do not append the old v11 or v11s again: that duplicates examples and reintroduces
rejected labels. Do not resume an optimizer run with a changed dataset contract.
For GRPO/DPO, rebuild prepared inputs and rescore/regenerate preferences under the
current reward policy; preserve SFT-checkpoint lineage checks. Golden32/Golden35/
Bixby50 remain holdouts, never regeneration/training/preference inputs.

## Validation boundary

CPU tests cover association/omission/invention failures, unchanged default prompting
and scoring, canonical admission, immutable splits, bounded retry, hash-bound resume
and training-row compatibility. This development machine cannot call Muse or train
on GPUs. Generation yield, model quality, training performance and Android rendering
still need verification on the respective hosts; offline tests do not prove them.

Implementation validation on 23 October 2026 also verified the real v11 bundle's
204,303 input rows and all ten source holds in read-only pilot-plan mode. Only
eight targets were selected/scored in that pilot; this is not a full target-quality
audit or evidence that regeneration has run. No new dataset was generated here.

One pre-existing regression remains outside this patch:
`test_training_fidelity_audit.py::test_hidden_list_and_email_special_card_title_do_not_gain_credit`
expects a Card title to be missing, while current renderer evidence includes it.
The same failure was reproduced with the original HEAD scorer code and unchanged
fidelity/evidence modules; this patch does not relax or remove that test.
