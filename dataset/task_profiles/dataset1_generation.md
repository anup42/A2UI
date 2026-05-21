# dataset1 generation

Purpose: resume the current Gemini dataset generation flow for `dataset_v3` first, then `dataset_v1`.

Restart command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File dataset/scripts/start_dataset1_generation.ps1
```

Current profile:

- Runs: `dataset_v3,dataset_v1`
- Target per run: `10000` responses and `10000` IR records
- Manager: `dataset/scripts/manage_gemini_dataset_workers.py`
- Stage 2 model: `gemini_3_1_flash_lite`
- Stage 3 model: `gemini_3_1_flash_lite`
- Stage 2 batch size: `16`
- Stage 3 batch size: `8`
- Stage 3 minimum backlog before running: `256`
- Manager poll interval: `300` seconds
- Hourly generated-data commit/push: enabled by default
- Key source: ignored `dataset/.env` values `GEMINI_STAGE2_API_KEYS` and `GEMINI_STAGE3_API_KEYS`

Last stopped state on 2026-05-21:

- `dataset_v3`: `10000` queries, `7581` responses, `6582` IR
- `dataset_v1`: `10000` queries, `5069` responses, `4870` IR

Notes:

- Do not commit API keys.
- Do not restart stale Azure workers for this task.
- If the user says "start dataset1 generation", run the restart command above.
