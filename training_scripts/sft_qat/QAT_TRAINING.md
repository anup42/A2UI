# QAT Training Scripts

Two new quantization-aware training (QAT) scripts that integrate seamlessly with the existing `train_gemma.py` and `train_qwen.py` setup.

## Quick Start

### QAT + LoRA (Pure-PyTorch)

Train with LoRA adapters on frozen base weights, fake-quantized with a straight-through estimator.

```bash
bash run_qat_lora.sh single 1 /path/to/model data.jsonl ./output test.jsonl
```

### QAT Only (Full-Parameter, torchao 8da4w)

Train all weights with torchao's 8-bit dynamic activation / 4-bit weight quantizer.

```bash
bash run_qat_only.sh single 1 /path/to/model data.jsonl ./output test.jsonl
```

## Usage

```bash
bash run_qat_lora.sh [MODE] [NUM_GPUS] [MODEL_PATH] [DATA_PATH] [OUTPUT_DIR] [TEST_DATA_PATH]
bash run_qat_only.sh [MODE] [NUM_GPUS] [MODEL_PATH] [DATA_PATH] [OUTPUT_DIR] [TEST_DATA_PATH]
```

### Arguments

| Arg | Required | Example | Notes |
|-----|----------|---------|-------|
| `MODE` | No (default: `single`) | `multigpu`, `deepspeed_zero2`, `deepspeed_zero3` | Launcher mode |
| `NUM_GPUS` | No (default: 1) | `4` | Number of GPUs for multi-GPU modes |
| `MODEL_PATH` | **Yes** | `./models/gemma-2b` or `/path/to/model` | Local model directory |
| `DATA_PATH` | **Yes** | `data.jsonl` or `./data` | Training data (JSON/JSONL file or folder) |
| `OUTPUT_DIR` | **Yes** | `./output` | Where to save checkpoints and logs |
| `TEST_DATA_PATH` | No | `test.jsonl` | Held-out test set for generation-based eval |

### Modes

- `single` — single GPU (default)
- `multigpu` — multi-GPU DDP
- `deepspeed_zero2` — DeepSpeed ZeRO-2 (requires `ds_config_zero2.json`)
- `deepspeed_zero3` — DeepSpeed ZeRO-3 (requires `ds_config_zero3.json`)

## Examples

### QAT + LoRA, single GPU

```bash
bash run_qat_lora.sh single 1 ./models/gemma-2b train.jsonl ./qat_lora_output test.jsonl
```

### QAT + LoRA, 4 GPUs

```bash
bash run_qat_lora.sh multigpu 4 ./models/gemma-2b train.jsonl ./qat_lora_output test.jsonl
```

### QAT Only (full-param), single GPU

```bash
bash run_qat_only.sh single 1 ./models/gemma-2b train.jsonl ./qat_only_output test.jsonl
```

### QAT Only, 2 GPUs with ZeRO-2

```bash
bash run_qat_only.sh deepspeed_zero2 2 ./models/gemma-2b train.jsonl ./qat_only_output test.jsonl
```

## Training Scripts

### `train_qat_lora.py`

**Backend:** Pure PyTorch STE fake quantization (group-wise symmetric INT4 weights, per-token INT8 dynamic activations).

**Features:**
- LoRA adapters on frozen base weights (QA-LoRA style)
- `--qat_warmup_steps`: train in bf16 first, then enable fake quant (e.g., 300)
- `--qat_group_size`: group size for weight quantization (default 32)
- `--qat_weight_only`: skip activation quantization (weights only)
- `--merge_before_export`: merge LoRA into base before INT4 export
- Saves `adapter/` (PEFT), `int4_weights.pt` (quantized weights), `qat_config.json`

```bash
python train_qat_lora.py \
    --model_name ./models/gemma-2b \
    --data_path train.jsonl \
    --output_dir ./output \
    --test_data_path test.jsonl \
    --epochs 4 \
    --qat_warmup_steps 300 \
    --qat_scheme int4 \
    --use_lora \
    --lora_r 16
```

### `train_qat_only.py`

**Backend:** torchao `Int8DynActInt4WeightQATQuantizer` (8da4w).

**Features:**
- Full-parameter training (no adapters)
- Lower learning rate by default (2e-5 vs 5e-5 for LoRA)
- Requires: `pip install torchao`
- Saves `model/` (fake-quantized HF checkpoint), `model_int4/qat_8da4w.pt` (real INT4)

```bash
python train_qat_only.py \
    --model_name ./models/gemma-2b \
    --data_path train.jsonl \
    --output_dir ./output \
    --test_data_path test.jsonl \
    --epochs 4 \
    --qat_warmup_steps 300 \
    --qat_group_size 32
```

## Evaluation & Checkpoints

Both trainers produce:

- **`best_test_checkpoint/`** — best model by test metric (e.g., `mean_direct_match_score`)
- **`best_val_loss_checkpoint/`** — best model by eval loss
- **`best_heuristic_checkpoint/`** — best model by external heuristic pipeline (if available)
- **`test_predictions.json`** / **`test_metrics.json`** — predictions and scores on test set
- **`aggregate_step_<N>.json`** — heuristic pipeline aggregates at each step
- TensorBoard logs in both `runs/<timestamp>/` (HF Trainer) and root `output_dir`

View with:

```bash
tensorboard --logdir ./output
```

## Key Differences from `train_gemma.py` / `train_qwen.py`

Both QAT scripts inherit the full eval pipeline:
- Distributed generation-based evaluation (MetricsAggregator)
- Heuristic pipeline integration (if available)
- Three best-checkpoint tracking (test metric, val loss, heuristic score)
- TensorBoard scalars for all metrics

**But they add:**
- `--qat_warmup_steps` — switchable fake quantization for stable training
- `--qat_diag_interval` — periodic diagnostics (grad norms, weight checksums, unique value counts)
- Per-rank "weight not frozen" / "weight not moving" warnings (catches mistakes)

## Common CLI Flags

### Shared with `train_gemma.py`

```
--epochs 4
--lr 5e-5 (LoRA) or 2e-5 (full-finetune)
--batch_size 1
--gradient_accumulation_steps 16
--max_length 8192
--logging_steps 50
--eval_steps 100
--save_steps 500
--lora_r 16 --lora_alpha 32 (LoRA only)
--test_data_path test.jsonl
--heuristic_util_folder /path/to/util
```

### QAT-specific

```
--qat_scheme int4|int8        # Weight bit width (default: int4)
--qat_group_size 32|128       # Weight group size (default: 32 = QA-LoRA strict)
--qat_warmup_steps 300        # Warmup in bf16 before enabling fake quant
--qat_weight_only             # Quantize weights only, keep activations in bf16
--qat_diag_interval 50        # Log QAT diagnostics every N steps (0 to disable)
```

### LoRA-specific (`train_qat_lora.py`)

```
--use_lora
--lora_r 16
--lora_alpha 32
--lora_dropout 0.05
--lora_layers all|early|late|middle|"28,29,30,31"
--merge_before_export         # Merge LoRA into base before INT4 export
```

## Dependencies

**`train_qat_lora.py` (Pure PyTorch):**
```
torch, transformers, trl, peft, datasets, accelerate, tensorboard
```

**`train_qat_only.py` (torchao):**
```
torch, transformers, trl, peft, datasets, accelerate, tensorboard, torchao
```

Install torchao:
```bash
pip install torchao
```

## Troubleshooting

### "No chat template" error

The tokenizer must have a `chat_template` field. Pass `--allow_missing_chat_template` to proceed anyway (eval metrics may be unreliable).

### "torchao not installed" (train_qat_only.py only)

```bash
pip install torchao
```

### "heuristic pipeline not available"

The heuristic evaluation callback auto-disables if `--heuristic_util_folder` or `--heuristic_script_dir` are missing. Training continues without it. Pass `--no_heuristic_eval` to silence the warning.

### Out of memory

Reduce `--batch_size`, increase `--gradient_accumulation_steps`, or use DeepSpeed ZeRO-2/3 modes.

## File Structure

```
sft_qat/
  qat_utils.py                    # Shared utils: fake quant, callbacks, plumbing
  train_qat_lora.py              # QAT + LoRA trainer (pure-torch)
  train_qat_only.py              # QAT full-param trainer (torchao)
  run_qat_lora.sh                # Launcher for QAT + LoRA
  run_qat_only.sh                # Launcher for QAT only

  dataloader.py                  # (unchanged) Shared dataset loading
  eval.py                        # (unchanged) Standalone evaluation
  metrics/                       # (unchanged) Evaluation metrics
  train_gemma.py                 # (unchanged) bf16 LoRA baseline
  train_qwen.py                  # (unchanged) bf16 LoRA baseline (Qwen variant)
```

## References

- [QAT Design Doc](./QAT_TRAINING.md) — this file
- `qat_utils.py` — implementation details and callback designs
- `train_gemma.py` / `train_qwen.py` — baseline SFT setup (unchanged)
