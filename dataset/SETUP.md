# Setup Guide

This guide provides detailed instructions for setting up the dataset generation pipeline.

## Prerequisites

- Python 3.8 or higher
- pip or uv package manager
- For local models: CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dataset
```

2. Install dependencies:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (faster)
uv pip install -r requirements.txt

# For full benchmarking capabilities including optional dependencies
pip install -r requirements.txt
pip install jsonschema tiktoken playwright pyyaml referencing
```

3. Install Playwright browsers (if using rendering features):
```bash
playwright install chromium
```

## Environment Configuration

### API Keys Setup

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI
OPENAI_API_KEY="your-openai-api-key"

# Perplexity
PERPLEXITY_API_KEY="your-perplexity-api-key"
PERPLEXITY_API_BASE="https://api.perplexity.ai"

# Gemini
GEMINI_API_KEY="your-gemini-api-key"
# Or for multiple keys:
# GEMINI_API_KEYS="key1,key2,key3"

# OpenRouter
OPENROUTER_API_KEY="your-openrouter-api-key"
OPENROUTER_API_BASE="https://openrouter.ai/api/v1"
OPENROUTER_SITE_URL="https://your-site.example"
OPENROUTER_APP_NAME="DatasetRunner"

```

### Local Model Setup

For local Qwen/DeepSeek models:

```bash
# Model paths
QWEN_MODEL_PATH="/path/to/Qwen3-Coder-30B-A3B-Instruct"
DEEPSEEK_MODEL_PATH="/path/to/DeepSeek-Coder-V2-Lite-Instruct"

# GPU configuration (example for 4x V100)
CUDA_VISIBLE_DEVICES="0,1,2,3"

# Offline mode (recommended)
LOCAL_STRICT_OFFLINE="1"
HF_HUB_OFFLINE="1"
TRANSFORMERS_OFFLINE="1"
HF_DATASETS_OFFLINE="1"
DATASET_OFFLINE_MODE="1"
```

## Running the Pipeline

### Basic Usage

```bash
# Run all stages with a specific model
python src/main.py --stage 1 --model openai_gpt4o --run_id my_experiment
python src/main.py --stage 2 --model openai_gpt4o --run_id my_experiment
python src/main.py --stage 3 --model openai_gpt4o --run_id my_experiment
python src/main.py --stage 4 --model openai_gpt4o --run_id my_experiment
```

### Local Models

```bash
# Qwen model
python src/main.py --stage 1 --model qwen3_coder_30b_a3b_local_4x --run_id qwen3_4x
python src/main.py --stage 2 --model qwen3_coder_30b_a3b_local_4x --run_id qwen3_4x
python src/main.py --stage 3 --model qwen3_coder_30b_a3b_local_4x --run_id qwen3_4x

# DeepSeek model
python src/main.py --stage 1 --model deepseek_coder_v2_lite_local_4x --run_id deepseek_4x
python src/main.py --stage 2 --model deepseek_coder_v2_lite_local_4x --run_id deepseek_4x
python src/main.py --stage 3 --model deepseek_coder_v2_lite_local_4x --run_id deepseek_4x
```

## Benchmarking

```bash
# Compare multiple models
python src/main.py --benchmark_models openai_gpt4o gemini_3_pro perplexity_gpt5
```

## Visualization

```bash
# Start the visualizer
python visualizer/app.py --port 8008

# Open in browser: http://127.0.0.1:8008
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   # Install all required dependencies
   pip install -r requirements.txt
   pip install jsonschema tiktoken playwright pyyaml referencing
   ```

2. **Playwright Not Working**:
   ```bash
   # Install browsers
   playwright install-deps
   playwright install chromium
   ```

3. **API Rate Limits**:
   - Adjust rate limiting in `configs/run.yaml`
   - Use multiple API keys when available

4. **Local Model Memory Issues**:
   ```bash
   # Reduce memory usage
   export LOCAL_MODEL_MAX_MEMORY="12GiB"
   export LOCAL_MODEL_GPU_MEMORY_UTILIZATION="0.8"
   ```

### Error Handling Improvements

The pipeline includes several error handling mechanisms:
- Automatic retries for transient errors
- Fallback generation for failed validations
- Detailed error logging in run manifests
- Graceful degradation when optional components are missing

For better error visibility, enable debug logging:
```bash
export TOON_FORMAT_DEBUG=1
```

## Asset Licensing

All icons and assets used in this project are properly licensed:
- Bootstrap Icons: MIT License
- Local assets: See `assets/icon_catalog/bootstrap-icons/SOURCES_AND_LICENSES.md`

When adding new assets, ensure proper license tracking by updating the sources file.
