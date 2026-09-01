#!/bin/bash
# Setup script for heuristic evaluation dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENUI_LM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Setting up heuristic evaluation dependencies..."

# Check if util folder exists
UTIL_FOLDER="${GENUI_LM_ROOT}/dataset/data/runs/golden50_qwen3_0.6"
if [ ! -d "${UTIL_FOLDER}" ]; then
    echo "ERROR: Util folder not found: ${UTIL_FOLDER}"
    echo "Please ensure GenUI-LM dataset is properly set up."
    exit 1
fi

# Check if config exists
CONFIG_FILE="${GENUI_LM_ROOT}/dataset/configs/run.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Check if pipeline scripts exist
PIPELINE_SCRIPT="${GENUI_LM_ROOT}/dataset/scripts/run_heuristic_pipeline.py"
if [ ! -f "${PIPELINE_SCRIPT}" ]; then
    echo "ERROR: Pipeline script not found: ${PIPELINE_SCRIPT}"
    exit 1
fi

echo "✓ All heuristic dependencies found!"
echo ""
echo "Usage in train.py:"
echo "  - Util folder: ${UTIL_FOLDER}"
echo "  - Config: ${CONFIG_FILE}"
echo "  - Pipeline: ${PIPELINE_SCRIPT}"
