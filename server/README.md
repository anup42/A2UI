# Local Server for GenUICraft

This server runs a Hugging Face causal LLM (for example `Qwen/Qwen2.5-Coder-7B-Instruct`) and exposes a simple HTTP API used by the Android app when `Local Server` is selected in Settings.

## 1) Install

```powershell
cd server
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run on your GPU

```powershell
python main.py --model-path Qwen/Qwen2.5-Coder-7B-Instruct --host 0.0.0.0 --port 8000 --device cuda --attn-implementation sdpa
```

Notes:
- `--model-path` accepts either a Hugging Face repo id or a local filesystem path.
- First load can take time while weights are downloaded/loaded.
- If your model needs custom code, add `--trust-remote-code`.
- If you see a FlashAttention GPU compatibility error, use `--attn-implementation sdpa` or `--attn-implementation eager`.
- Current server build also auto-fallbacks to `eager` if a FlashAttention runtime error appears during generation.
- To disable that fallback and fail fast with only the selected backend, add `--strict-attn`.
- The server normalizes absolute model paths and caches failed load attempts to avoid repeatedly re-loading checkpoint shards for the same failing config.
- Local requests now log generation start/end with prompt token count, completion token count, total latency, and tokens/sec for easier debugging.
- Qwen generation path follows the official model guide: chat template rendering followed by tokenizer/model input encoding. For Qwen Instruct models, default system prompt is automatically applied when none is provided.

## 3) Health check

```powershell
curl http://127.0.0.1:8000/health
```

## 4) Generate endpoint

`POST /v1/generate`

Request example:

```json
{
  "prompt": "Write a short weather summary for London.",
  "system_prompt": "Be concise.",
  "temperature": 0.2,
  "max_output_tokens": 512,
  "json_mode": false,
  "model_path": "Qwen/Qwen2.5-Coder-7B-Instruct"
}
```

Response shape:

```json
{
  "text": "Generated text...",
  "model_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 180,
    "total_tokens": 300
  },
  "timings": {
    "total_ms": 842.4
  }
}
```

## 5) Android app configuration

In the app:
- Open `Settings`
- Set `Inference Backend` to `Local Server`
- Set `Server URL` to `http://10.0.2.2:8000` (Android emulator -> host machine)
- Set `Model path` to the same model id/path you want to use

If using a physical phone, use your computer's LAN IP instead of `10.0.2.2`.

## 6) Quick self-test from CLI

Run one local test generation (prints output):

```powershell
python main.py --model-path Qwen/Qwen2.5-Coder-7B-Instruct --device cuda --self-test-only
```

By default this uses prompt:

```text
show pizza recipie
```

You can override prompt/tokens:

```powershell
python main.py --model-path Qwen/Qwen2.5-Coder-7B-Instruct --device cuda --self-test-only --self-test-prompt "show pizza recipie" --self-test-max-output-tokens 256
```

## 7) vLLM server variant (`server_vllm.py`)

If you want vLLM backend (as recommended for Qwen deployment throughput), use:

```powershell
cd server
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-vllm.txt
```

Run:

```powershell
python server_vllm.py --model-path Qwen/Qwen2.5-Coder-7B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --gpu-memory-utilization 0.9
```

The API contract remains the same (`/health`, `POST /v1/generate`) so Android settings do not need request-shape changes.
For long prompts (for example 15k+ tokens in Stage 2/Stage 3 pipeline input), the server now auto-fits prompt+generation into model context by reducing output tokens and truncating prompt tokens (head+tail) when required.
If you know your model context size, pass it explicitly for tighter control:

```powershell
python server_vllm.py --model-path /home/anup/models/Qwen2.5-Coder-7B-Instruct --max-model-len 32768
```

`server_vllm.py` supports additional sampling fields in requests:
- `top_p` (default `0.95`)
- `top_k` (default `20`)
- `presence_penalty` (default `0.0`)
- `enable_thinking` (optional, passed only when tokenizer template supports it)
