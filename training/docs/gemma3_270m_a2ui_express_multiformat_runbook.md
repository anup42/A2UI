# Gemma 3 270M A2UI Express multi-format training runbook

This is the reproducible handoff for training one `google/gemma-3-270m-it`
A2UI Express model and measuring the trained checkpoint plus four LiteRT-LM
deployment variants on the same immutable Golden-32.

The checked-in entry point is:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id gemma270_a2ui_YYYYMMDD_001
```

It is plan-only by default. It does not download a model, train, convert, or
write an output unless an explicit execution flag is supplied.

## Contract and precision truth

The job trains one LoRA adapter with effective merged-weight W8 QAT:

```text
Stage-2 response_text
  -> strict A2UI Express v1 completion
  -> Gemma 3 270M W8/AFP32 LoRA QAT
  -> best exact-Golden-32 adapter
  -> merged Hugging Face checkpoint
  -> W32 / W16 / W8 / W4 LiteRT-LM packages
  -> identical deterministic Golden-32 evaluation
  -> TensorBoard records + final_scorecard.json
```

The names `w32`, `w16`, `w8`, and `w4` describe stored deployment weight
precision. They are not four separately QAT-trained models.

| Variant | Exporter setting | Status | QAT-aligned |
|---|---|---|---|
| `w32` | `quantization_recipe=none` | Public floating baseline | No |
| `w16` | `none` plus `--experimental_use_fp16=True` | Experimental; explicit opt-in | No |
| `w8` | `dynamic_wi8_afp32` | Canonical public export; optional official-Q8 topology | Yes |
| `w4` | `dynamic_wi4b32_afp32` | Experimental block-32 PTQ; explicit opt-in | No |

Do not describe `w32`, `w16`, or `w4` as QAT-aligned results. In particular,
the W4 result is the measurement that determines whether a W8-trained model
survives post-training INT4 conversion; it is not evidence of W4 QAT.

The exact official-topology builder is valid only for W8. It replaces all 127
mapped learned constants in a released Gemma 3 270M Q8 graph and fails closed
on base identity, artifact digest, mapping, or quantization-layout mismatch.
Never use that Q8 transplant path for W32, W16, or W4.

## Checked-in inputs

- Pipeline: `training/configs/pipelines/gemma3_270m_a2ui_express_multiformat.yaml`
- Training: `training/configs/models/gemma3_270m_a2ui_express_qat.yaml`
- Golden preparation: `training/configs/datasets/golden32_20260903_eval.yaml`
- Golden source: `dataset/data/runs/genui_demo_golden40_gemini38_20260903/genui.jsonl`
- External-runtime protocol example:
  `training/configs/eval/litertlm_external_runner.example.yaml`

The source `genui.jsonl` contract is exactly 32 unique strict-valid records,
with SHA-256:

```text
596f56a15a63d904423970c9768797ba795ff0528a04da37dddcba280a4f4e17
```

The paired `responses.jsonl` SHA-256 is:

```text
27406fa78d486ded4e9fcb2a78e987aa54d9f2ee8fe5db926874a1b2ff827089
```

Pipeline planning/preparation verifies both source digests, the independently
pinned dataset-config and prepared-`all.jsonl` digests, and the strict row
count before training. Golden-32 is evaluation-only and must never be appended
to the training split.

The normal training data remains:

```text
training/outputs/datasets/stage3_folder_90_10/train.jsonl
training/outputs/datasets/stage3_folder_90_10/val.jsonl
```

Prepare that dataset with the existing
`training/configs/datasets/stage3_folder_90_10.yaml` workflow before starting
the model job. The plan compares normalized `response_text` SHA-256 signatures
from both prepared train/validation splits against the prepared Golden-32.
Run-local IDs such as `q_000001` are deliberately not compared because they
are reused across dataset runs; matching response content is rejected even
when its IDs differ. Planning reports a not-yet-checkable warning when the
training splits are absent, while preflight and training fail closed until the
zero-overlap check succeeds.

## Environment

Keep training and conversion in separate environments because the tested
LiteRT conversion pins can conflict with the CUDA training stack:

```bash
python3 -m venv .venv-gemma270-train
. .venv-gemma270-train/bin/activate
# Install the CUDA PyTorch build selected for this host first.
python -m pip install -r training/requirements-training.txt

deactivate
python3 -m venv .venv-litert-export
. .venv-litert-export/bin/activate
python -m pip install -r training/requirements-edge-export-tested.txt
```

Authenticate to Hugging Face for `google/gemma-3-270m-it` before preflight.
Use a CUDA host for training. Do not infer successful training from static
config tests or plan output.

TensorBoard uses one repository-level root:

```text
<repo>/tensorboard/<run-id>/
```

On MLP, mount or select the real root explicitly:

```bash
export A2UI_TENSORBOARD_ROOT=/tensorboard
tensorboard --logdir /tensorboard
```

Both Trainer logs and final evaluation records honor that environment
variable. Evaluation records appear below:

```text
/tensorboard/<run-id>/evaluation_records/
  checkpoint/
  merged_checkpoint/
  litertlm_w32/
  litertlm_w16/
  litertlm_w8/
  litertlm_w4/
```

Trainer event files are isolated in the sibling
`/tensorboard/<run-id>/training/` directory. Every final record uses the actual
selected-checkpoint step from `training_metadata.json`; the pipeline never
invents display steps for converted variants.

## Configure the LiteRT-LM batch runner

LiteRT-LM does not provide one stable cross-platform batch-evaluation stdout
contract. The repository therefore uses a versioned JSONL runner boundary
instead of scraping human-oriented CLI output.

Copy `training/configs/eval/litertlm_external_runner.example.yaml` outside the
checkout and replace its command with the actual runner on the training host.
The runner must implement `a2ui_external_generation_v1`, consume every request,
and emit exactly one output row per request. See
`training/docs/golden32_evaluation.md` for the schema.
Planning also validates the protocol, string command list, supported
placeholders, and required `{requests_path}`/`{outputs_path}` routing. The
checked-in `/absolute/path/to/...` command is intentionally not runnable.

Pass the host file without editing the checked-in pipeline:

```bash
RUNNER_CONFIG=/absolute/path/to/litertlm_runner.yaml
```

CPU is recommended for the comparable correctness score. Record a separate
device/GPU report before promotion. At the time this runbook was written,
upstream issue `google-ai-edge/LiteRT-LM#3280` reported Gemma 3 270M GPU-only
`<pad>` output for both Q8 and block-32 INT4 while CPU worked. Treat an actual
runtime check—not package creation—as the authority.

## Recommended staged run

Run from the repository root. Keep stages separate so each expensive action
has inspectable logs under
`training/outputs/pipelines/gemma3_270m_a2ui_express_multiformat/<run-id>/logs/`.
Choose one fresh ID and use it for every command:

```bash
RUN_ID=gemma270_a2ui_YYYYMMDD_001
```

1. Inspect the complete plan:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" --formats all
```

2. Prepare and validate the immutable Golden-32:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --prepare-golden
```

3. Run the real model/data/QAT preflight without optimizer steps:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --preflight-training
```

4. Train. The Trainer evaluates Golden-32 at every normal evaluation boundary,
logs `golden32/*` scalars to TensorBoard, and saves the best adapter by
`generation_reward_v5_4_avg`:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --execute-training
```

5. Re-evaluate the selected adapter as the final actual-checkpoint score, then
merge and verify that the merge itself did not change behavior:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --evaluate-checkpoint

python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --execute-merge \
  --evaluate-merged
```

6. Export all four public variants. W16 and W4 are explicitly experimental,
so the opt-in is mandatory. Activate `.venv-litert-export` for this stage (or
set `pipeline.export.command` to that environment's absolute `litert-torch`
executable):

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --formats all \
  --execute-exports \
  --allow-experimental-formats
```

7. Inspect/hash each package, verify its embedded tensor and quantization
layout matches the W32/W16/W8/W4 lane, then run all four through the same
external LiteRT-LM runner and the same Golden-32 scorer:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --formats all \
  --evaluate-litertlm \
  --allow-experimental-formats \
  --runner-config "$RUNNER_CONFIG"
```

8. Require all five final results and write the scorecard:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --write-scorecard
```

`--write-scorecard` refuses to complete unless the actual adapter checkpoint
and all four LiteRT-LM aggregate files exist. The scorecard records artifact
paths, sizes, SHA-256 identities, complete aggregate metrics, and each variant's
`generation_reward_v5_4_avg` delta from the checkpoint. A checkpoint is a
directory, so its identity is the SHA-256 of a deterministic sorted per-file
hash manifest; the scorecard separately records the adapter weights,
`adapter_config.json`, and `training_metadata.json`, and refuses a checkpoint
whose required provenance identity is incomplete.
It also rejects a non-finite/boolean score, a different 32-row split, a stale
prepared-set manifest, the wrong metric version or checkpoint step, a
TensorBoard record outside this run, an artifact changed after evaluation, or
a package-inspection report whose whole-file digest no longer matches, or a
package whose inspected precision layout belongs to a different lane.

After preflight, the same stages may be submitted as one supervised job:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --prepare-golden \
  --execute-training \
  --evaluate-checkpoint \
  --execute-merge \
  --evaluate-merged \
  --execute-exports \
  --evaluate-litertlm \
  --write-scorecard \
  --formats all \
  --allow-experimental-formats \
  --runner-config "$RUNNER_CONFIG"
```

Do not use the combined form until the standalone preflight has passed.
Use it only when the tested converter and external runner are available through
absolute executables; do not install conflicting training and converter pins
into one environment merely to make this convenience command work.

## Optional official Q8 artifact

The ordinary W8 export uses the public `dynamic_wi8_afp32` converter. To test
the same merged checkpoint in the released official Q8 topology, obtain the
exact package whose SHA-256 is pinned in the pipeline and run:

```bash
python training/scripts/run_gemma270m_a2ui_express_multiformat.py \
  --run-id "$RUN_ID" \
  --formats w8 \
  --execute-official-q8 \
  --official-q8-source /absolute/path/to/gemma3-270m-it-q8.litertlm \
  --evaluate-litertlm \
  --write-scorecard \
  --runner-config "$RUNNER_CONFIG"
```

That command makes the official-topology candidate the W8 evaluation artifact
for package inspection, Golden evaluation, and the W8 scorecard entry in that
invocation. It assumes the other four required final results (checkpoint plus
W32/W16/W4) already exist. It does not make the W32/W16/W4 public exports
official.

## Outputs and acceptance

The final outputs are under:

```text
training/outputs/pipelines/gemma3_270m_a2ui_express_multiformat/
  <run-id>/
    merged_best_hf/
    litertlm/
      gemma3_270m_a2ui_express_w32.litertlm
      gemma3_270m_a2ui_express_w16.litertlm
      gemma3_270m_a2ui_express_w8.litertlm
      gemma3_270m_a2ui_express_w4.litertlm
    evaluation/
      checkpoint/
      merged/
      litertlm/{w32,w16,w8,w4}/
    final_scorecard.json
```

The run-scoped training output is
`training/runs/gemma3_270m_a2ui_express_multiformat/<run-id>/`; it contains the
resolved launch config and Golden-best adapter. The orchestrator refuses to
reuse non-empty run outputs or overwrite an existing stage log. Select a new
run ID instead of merging evidence from a failed attempt.

Each evaluation directory contains raw predictions, scored predictions,
`aggregate_metrics.json`, `package_inspection.json`, an audit result, and—on
LiteRT-LM runs—the external runner request/output/log/manifest files. The
scorecard is the authoritative summary; package existence alone is not success.

Before promotion, verify:

- All 32 rows completed for every artifact.
- Raw strict A2UI Express validity and `generation_reward_v5_4_avg` are present.
- Checkpoint and merged-checkpoint scores agree within the chosen tolerance.
- TensorBoard contains training, periodic Golden, final checkpoint, and all
  four LiteRT-LM series.
- Every package hash in `final_scorecard.json` matches the tested file.
- W16/W4 experimental results are explicitly labeled.
- The chosen target device/backend passes a separate runtime/delegation test.

No score should be entered manually. Scores must come from the actual
checkpoint/package evaluation aggregates and be logged to TensorBoard by the
pipeline.

## Upstream converter references

- LiteRT-LM: <https://github.com/google-ai-edge/LiteRT-LM>
- AI Edge quantizer recipes:
  <https://github.com/google-ai-edge/ai-edge-quantizer/blob/main/ai_edge_quantizer/recipe.py>
- Current Gemma 3 270M GPU issue:
  <https://github.com/google-ai-edge/LiteRT-LM/issues/3280>

Always retain the actual `pip freeze`, exporter command, LiteRT-LM version,
runner manifest, and output hashes with a completed remote job; upstream
converter behavior is version-sensitive.
