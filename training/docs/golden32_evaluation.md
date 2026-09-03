# Golden-32 checkpoint and LiteRT-LM evaluation

This workflow binds the 32 strict-valid response-to-A2UI-Express records from
`dataset/data/runs/genui_demo_golden40_gemini38_20260903/genui.jsonl`. It is an
evaluation-only set and must never be merged into the SFT training sources.

The checked-in dataset configuration requires exactly 32 accepted, unique
rows and binds both source files:

```text
596f56a15a63d904423970c9768797ba795ff0528a04da37dddcba280a4f4e17  genui.jsonl
27406fa78d486ded4e9fcb2a78e987aa54d9f2ee8fe5db926874a1b2ff827089  responses.jsonl
```

The first digest is `genui.jsonl`; the second is `responses.jsonl`. Only these
two source files are required for preparation. Per-case screenshots and other
large run artifacts are useful for visual analysis but are not training inputs.

The multiformat pipeline configs independently pin the materialized contract as
well, so a coordinated config/output/manifest regeneration cannot silently
change the examples being scored:

```text
2c4c70624658f097484efda25f5f336d685fa469b13982d310a17100e3eb44d5  golden32_20260903_eval.yaml
b1a2d9c1a2b7ed6f2e052fd84b53284f706667c9ade7c6f859ba287150b28ad4  all.jsonl
```

Prepare it once per clean checkout:

```bash
python training/scripts/prepare_dataset.py \
  --config training/configs/datasets/golden32_20260903_eval.yaml
```

The immutable evaluation split is then:

```text
training/outputs/datasets/golden32_20260903_eval/all.jsonl
```

Preparation fails before writing a usable manifest if the source hash, row
count, strict Express validation, or unique source identity contract differs.
The persisted manifest records both source-file digests, the dataset-config
digest, and SHA-256 for every prepared output (`all/train/val/test/rejected`).
Evaluation and scorecard stages revalidate this chain against the independent
config and `all.jsonl` pins, so a stale, substituted, or jointly regenerated
32-row file is rejected even when its row count and manifest look correct.

## TensorBoard root

Install `training/requirements-training.txt` in the training/evaluation
environment; it includes the TensorBoard writer used by these commands.

New pipeline configs should set:

```yaml
training:
  tensorboard_root: tensorboard
golden_eval:
  tensorboard: true
  tensorboard_evaluation_name: checkpoint
```

A relative `training.tensorboard_root` is resolved from the repository root,
so `tensorboard` places Trainer events at
`<repo>/tensorboard/<run.id>/training/`. On an MLP machine, set
the environment override instead:

```bash
export A2UI_TENSORBOARD_ROOT=/tensorboard
tensorboard --logdir /tensorboard
```

The environment value takes precedence over YAML. Existing configurations
that specify only `training.logging_dir` retain their old behavior.
`training.tensorboard_subdir` defaults to `training`; set it to another single
directory name only when a pipeline needs a different separation. Final
evaluation records remain under the enclosing `<root>/<run.id>/` directory.

Periodic Golden evaluations continue to flow through `Trainer.log`. When a
TensorBoard root is configured, each periodic pass also writes a complete JSON
record under:

```text
<tensorboard-root>/<run-id>/evaluation_records/<evaluation-name>/step_000000000.json
```

The sidecar contains the full aggregate, metadata, and hashes for file and
directory artifacts. Directory checkpoints use a deterministic sorted
path/size/file-SHA manifest digest. TensorBoard receives every finite numeric leaf under
`evaluation/<evaluation-name>/...`.

An already-produced aggregate can be added without rerunning inference:

```bash
python training/scripts/log_evaluation_to_tensorboard.py \
  --aggregate path/to/aggregate_metrics.json \
  --tensorboard-root /tensorboard \
  --run-id my_run \
  --evaluation-name litertlm_w8 \
  --step 1000 \
  --artifact predictions=path/to/predictions.jsonl
```

## HF or PEFT checkpoint evaluation

The same CLI accepts a callback-saved PEFT adapter or a merged Hugging Face
checkpoint. `auto` selects adapter mode when `adapter_config.json` is present.
The default `--qat-mode auto` reapplies the configured fake-quant forward path
to QAT adapter checkpoints, so their final score uses the same numerical
contract as periodic training evaluation. Merged checkpoints remain dense;
their deployment formats are evaluated separately through LiteRT-LM.

```bash
python training/scripts/evaluate_checkpoint_on_golden.py \
  --config training/configs/models/gemma3_270m_ir_qat_sft.yaml \
  --checkpoint training/runs/example/best_golden_checkpoint \
  --checkpoint-kind auto \
  --split training/outputs/datasets/golden32_20260903_eval/all.jsonl \
  --output-dir training/outputs/eval/example/checkpoint \
  --tensorboard-root /tensorboard \
  --run-id example \
  --evaluation-name checkpoint \
  --required-rows 32 \
  --metric-version dual
```

Outputs are `predictions.jsonl`, `scored_predictions.jsonl`,
`aggregate_metrics.json`, and `evaluation_result.json`.

## LiteRT-LM external runner protocol

LiteRT-LM inference environments differ across Linux, Android, and MLP hosts,
so evaluation uses a small process boundary instead of assuming one runtime.
Copy `training/configs/eval/litertlm_external_runner.example.yaml`, then replace
the runner executable. The command is executed directly, without a shell.

Supported command placeholders are:

- `{model_path}`
- `{requests_path}`
- `{outputs_path}`
- `{output_dir}`
- `{max_input_tokens}`
- `{max_new_tokens}`
- `{mtp_enabled}`

The config must route at least `{requests_path}` and `{outputs_path}` through
the command or environment. Planning rejects the wrong protocol, unknown or
missing placeholders, non-string commands, and the checked-in placeholder
executable before any package evaluation begins.

`runner_requests.jsonl` contains one request per Golden row with `id`, the
training-compatible prompt/messages, token bounds, deterministic sampling, and
the MTP switch. It deliberately excludes the expected completion.

The runner must write `runner_outputs.jsonl` with exactly one row for every
request:

```json
{"id":"source:u_1:a2ui_express_v1","generated_text":"<a2ui>...</a2ui>","latency_ms":42.1,"input_tokens":900,"output_tokens":120}
```

IDs must be unique and must match the request set exactly. Additional fields
are preserved under `runtime` in the scored prediction record.

```bash
python training/scripts/evaluate_litertlm_on_golden.py \
  --model path/to/model_w8.litertlm \
  --runner-config path/to/litertlm_runner.yaml \
  --split training/outputs/datasets/golden32_20260903_eval/all.jsonl \
  --output-dir training/outputs/eval/example/litertlm_w8 \
  --tensorboard-root /tensorboard \
  --run-id example \
  --evaluation-name litertlm_w8 \
  --required-rows 32 \
  --metric-version dual
```

Add `--mtp-enabled` only for an E2B package and runtime that contain the MTP
assistant. Gemma 3 270M does not support MTP.

Alongside the standard scoring outputs, the LiteRT-LM directory retains the
request/output JSONL, combined runner log, and a command/protocol manifest.
Thus a failed or malformed package cannot silently receive a partial score.
Before a final scorecard is written, the pipeline also proves that the record
uses the selected checkpoint step, `metric_version`, pinned Golden split path
and digest, current model/package digest, and a current full-file package audit.
Changing any one of those after evaluation invalidates the score entry.
