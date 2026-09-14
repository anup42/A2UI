# Persistent SFT data cache and JSON startup fix

## What failed

The reported E2B preflight stopped while loading `prepared/train.jsonl`, before
an optimizer step. The first error was Arrow trying to convert a string field
to a boolean. The later pandas `Trailing data` error came from the JSON
loader's fallback path. Valid JSONL can contain different types in nested
provenance or structured IR fields; these fields do not have to share one
Arrow schema to train on the textual `messages`.

The screenshot does not identify the exact offending nested field. Regression
fixtures reproduce conflicting boolean, string, list and object fields in
metadata and structured IR. No original archive was edited or repaired for
this fix.

## Changes

- Cached SFT streams JSONL records through the model formatter and stores only
  explicit integer `input_ids`, `attention_mask` and completion-only `labels`.
- Uncached HF and the shared SFT runner's TRL path also stream original JSONL,
  format it first, then create an explicit three-string Arrow view. They never
  submit arbitrary raw metadata to `datasets.load_dataset("json")`.
- JSON/message errors include the source path and physical line number. Invalid
  records are not silently skipped, coerced, or repaired. The formatter still
  receives the original row, including metadata. Original hashes remain bound.
- Preparation cache v2 owns independent copies of verified prepared artifacts;
  it no longer points to an older run directory. Removing an old run does not
  invalidate the owned cache. Restores create independent run copies.
- One file-lock writer builds a matching preparation or token entry. Other
  processes wait with progress messages, then verify and reuse the completed
  entry. Interrupted builds cannot publish a success manifest. Corrupt owned
  entries are retained under `.rejected-*`, not silently trusted or deleted.
- Token files are memory-mapped Arrow streams shared by preflight, training and
  matching ranks/trials. Cache manifests bind content, tokenizer behavior,
  formatter/masking implementation, vocabulary limits and sequence length.
  Changing learning rate, epochs, GPU count or LoRA targets alone does not
  invalidate these preprocessing artifacts.
- Raw/tensor vocabulary checks, exact completion masking, overflow rejection,
  model-forward checks and deterministic greedy probes remain active. Model
  health or Golden scores are never cached as permission to skip runtime gates.
- Actual CPU optimizer testing also exposed Transformers' changed TensorBoard
  directory setting. SFT now scopes the documented `TENSORBOARD_LOGGING_DIR`
  around Trainer construction, restores the previous environment afterward,
  and retains `TrainingArguments.logging_dir` for older versions. Actual event
  files are checked, not merely the generated config.

The last compatibility change follows the official
[TensorBoard callback contract](https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.integrations.TensorBoardCallback).
The cache uses the public
[Dataset memory-mapped file API](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.from_file),
[Arrow IPC stream API](https://arrow.apache.org/docs/python/generated/pyarrow.ipc.new_stream.html),
and [filelock](https://py-filelock.readthedocs.io/en/latest/).

## Restart on the training PC

Stop the failed launch before updating; do not update an actively training
checkout. Pull the branch and use the same model/input arguments with a **new
output directory**. Do not edit an old run's bound generated YAML or use its
`--continue-run` option across this implementation update.

```bash
git pull --ff-only origin new_ir_changes_20260331

# Replace model/input paths with those from your existing launch.
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --input-dir /data/source-bound-train-val \
  --output-dir /runs/e2b-cache-v2 --epochs 1 \
  --preparation-cache-dir /runs/.golden-preparation-cache \
  --execute
```

Both caches are enabled by default in this launcher. Use the same persistent
cache directory across later run folders. Tokens default to its `tokens/`
subdirectory; `--token-cache-dir /fast-data/a2ui-tokens` can put them on another
fast persistent volume. `--no-preparation-cache` and `--no-token-cache` are
independent opt-outs. The first run after this update builds new entries once.
Subsequent matching runs should show `CACHE HIT`; hashes/integrity checks still
take time. Cache storage consumes additional disk space and has no automatic
eviction. Shared multi-host storage must support inter-host file locks.

The new explicit training dependency is `filelock>=3.15`, also normally present
through the HF stack. If absent, install it in the training environment. This
fix does not require downgrading pandas, changing valid metadata, or upgrading
all GPU libraries.

The same shared runner and cache options apply to `--profile 270m`. Custom
tokenizer subclasses, slow tokenizers, and custom adapter formatters must opt
out of token caching until they have an explicit supported identity contract.

## Validation scope

Focused verification passed:

- 168 launcher/preparation/HPO/resource-budget tests; one Windows symlink test
  skipped because creating symlinks is not permitted in that test environment.
- 47 strict-source and actual Gemma4 CPU preflight tests, including cold/warm
  token reuse and cache-disabled heterogeneous metadata.
- 43 real-tokenizer/Arrow cache tests, including two spawned processes sharing
  one build, changed inputs/options, corrupt manifests and incomplete entries.
- Two additional real CPU one-optimizer-step tests: warm cache and cache off.
  Both perform backward/update, validation loss, checkpoint/optimizer saving,
  final adapter saving, and verify TensorBoard `train/loss` and `eval/loss`
  events at the configured test root. Nonzero saved LoRA-B weights confirm an
  actual update. Models and data are tiny synthetic fixtures, not user samples.

Final verification on 2026-09-14 (the focused results above overlap):

| Run | Result |
| --- | --- |
| Full `training/tests` suite, Transformers 5.16.1 / PEFT 0.20.0 | 902 passed, 1 skipped; 424.44 seconds |
| Ten additional TensorBoard environment/callback tests added after full-suite collection | 10 passed; 31.38 seconds |
| Source, token-cache and SFT integration suites, minimum Transformers 5.10.1 / PEFT 0.19.0 | 102 passed, 1 skipped; 83.21 seconds |
| Critical Ruff checks and Git whitespace checks | Passed |

The minimum-version run skips the constructor-inspection test explicitly
targeting Transformers >=5.16.1; its actual one-step TensorBoard/event tests
pass. Both stacks use Python 3.12.10, CPU Torch 2.13.0, datasets 5.0.1,
PyArrow 24.0.0 and TensorBoard 2.21.0. CPU pin-memory and SWIG deprecation
warnings are non-fatal. No benchmark/production performance scores are inferred
from these synthetic tests.

Portable test commands (use the installed training environment):

```bash
python -m pytest training/tests -q --tb=short
python -m pytest training/tests/test_sft_source_loading.py \
  training/tests/test_token_cache.py \
  training/tests/test_sft_token_cache_integration.py -q --tb=short
```

No production E2B/270M training, H100 distributed execution, Golden32/35
scoring, QAT export or LiteRT inference was performed on this PC. The standard
dual-Golden evaluation schedule and `/tensorboard/<run-id>/training` location
are unchanged. Official E2B QAT, retained-mobile QAT and MTP remain separate
workflows. Separate experimental GRPO/full-QAT runners were not migrated by
this SFT loader fix. The change fixes the reported loader path; it cannot
guarantee unrelated hardware, dependency or data failures on the remote host.
