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
python main.py --model-path Qwen/Qwen2.5-Coder-7B-Instruct --host 0.0.0.0 --port 8000 --device cuda
```

Notes:
- `--model-path` accepts either a Hugging Face repo id or a local filesystem path.
- First load can take time while weights are downloaded/loaded.
- If your model needs custom code, add `--trust-remote-code`.

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
