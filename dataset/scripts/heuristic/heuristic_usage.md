# GenUI Heuristic Pipeline Usage Guide

This guide explains how to run the complete GenUI heuristic pipeline to process `predictions.jsonl` files and generate evaluation metrics.

## Prerequisites

1. **Activate the conda environment** before running the pipeline:

   ```bash
   source /home/<user>/train_env/bin/activate
   ```

2. **Navigate to the scripts directory**:
   ```bash
   cd /home/<user>/GenUI-LM/dataset/scripts/heuristic
   ```

## Running the Pipeline

### Basic Command

```bash
python run_heuristic_pipeline.py \
    --input <path_to_predictions.jsonl> \
    --util-folder /home/<user>/GenUI-LM/dataset/data/runs/golden50_qwen3_0.6 \
    --config /home/<user>/GenUI-LM/dataset/configs/run.yaml
```

### Required Arguments

| Argument        | Description                                                                                     | Example                                                                       |
| --------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `--input`       | Path to your `predictions.jsonl` file (the model output to evaluate)                            | `/home/<user>/GenUI-LM/dataset/data/runs/golden50_qwen3_8b/predictions.jsonl` |
| `--util-folder` | Path to util folder containing `queries.jsonl`, `responses.jsonl`, `golden50_url_mapping.jsonl` | `/home/<user>/GenUI-LM/dataset/data/runs/golden50_qwen3_0.6` (always fixed)   |
| `--config`      | Path to run.yaml config file for evaluation weights                                             | `/home/<user>/GenUI-LM/dataset/configs/run.yaml` (always fixed)               |

### Example Usage

```bash
# Activate environment
source /home/<user>/train_env/bin/activate

# Navigate to scripts directory
cd /home/<user>/GenUI-LM/dataset/scripts/heuristic

# Run the pipeline
python run_heuristic_pipeline.py \
    --input /home/<user>/GenUI-LM/dataset/data/runs/golden50_qwen3_8b/predictions.jsonl \
    --util-folder /home/<user>/GenUI-LM/dataset/data/runs/golden50_qwen3_0.6 \
    --config /home/<user>/GenUI-LM/dataset/configs/run.yaml
```

## Pipeline Steps

The pipeline executes the following steps in order:

| Step | Script                              | Description                                          |
| ---- | ----------------------------------- | ---------------------------------------------------- |
| 0    | `step0_rename.py`                   | Rename prediction keys to `genui_json`               |
| 1    | `step2_copyutils.py`                | Copy utility files (queries, responses, URL mapping) |
| 2    | `step1_genui_raw.py`                | Fix `genui_json` raw values                          |
| 3    | `step3_fix_genui_json_strings.py`   | Convert `genui_json` strings to objects              |
| 4    | `step4_fix_genui_predictions_v2.py` | Add validation and metrics (with URL unmasking)      |
| 5    | `step5_calculate_aggregates.py`     | Calculate aggregate metrics and overall scores       |

## Output Files

After successful completion, the following files are generated in the same directory as your input `predictions.jsonl`:

| File              | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| `genui.jsonl`     | Processed predictions with validation, metrics, and gen info                |
| `aggregates.json` | Aggregate metrics including overall score, media score, and per-row details |

## Advanced Options

### Skip Specific Steps

Use `--skip-steps` to skip specific step numbers (comma-separated):

```bash
python run_heuristic_pipeline.py \
    --input <path> \
    --util-folder /home/<user>/GenUI-LM/dataset/data/runs/golden50_qwen3_0.6 \
    --config /home/<user>/GenUI-LM/dataset/configs/run.yaml \
    --skip-steps "0,1,2"
```

### Start From a Specific Step

Use `--start-from` to begin from a specific step number:

```bash
python run_heuristic_pipeline.py \
    --input <path> \
    --util-folder /home/<user>/GenUI-LM/dataset/data/runs/golden50_qwen3_0.6 \
    --config /home/<user>/GenUI-LM/dataset/configs/run.yaml \
    --start-from 3
```

## Troubleshooting

### Schema Validation Failures

If you see `schema_valid_strict: false` in the output, it means the model's generated JSON structure is invalid. Common issues include:

- `elements` object nested inside `state` instead of at the top level
- Missing `root` key
- Empty or malformed `elements` object

Refer to the expected GenUI flat-spec format for the correct structure.

### Missing Utility Files

Ensure the `--util-folder` contains:

- `queries.jsonl`
- `responses.jsonl`
- `golden50_url_mapping.jsonl` (for URL unmasking in step 4)
