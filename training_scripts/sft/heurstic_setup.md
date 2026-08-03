# Heuristic Evaluation Setup

## Required Files

The training script (`train.py`) now includes automatic heuristic evaluation.
The following files are required:

### 0. Required python module
Make sure to install "json_repair" module using pip.

### 1. Util Folder
Contains `queries.jsonl`, `responses.jsonl`, `golden50_url_mapping.jsonl`

Default: `GenUI-LM/dataset/data/runs/golden50_qwen3_0.6/`

### 2. Config File
Contains evaluation weights for `overall_score` calculation.

Default: `GenUI-LM/dataset/configs/run.yaml`

### 3. Pipeline Scripts
All scripts in `GenUI-LM/dataset/scripts/`:
- `run_heuristic_pipeline.py`
- `step0_rename.py` through `step5_calculate_aggregates.py`

## Changing Paths

Edit these lines in `train.py` (around line 970):

```python
DEFAULT_UTIL_FOLDER = "/your/path/to/util/folder"
DEFAULT_CONFIG_PATH = "/your/path/to/run.yaml"
