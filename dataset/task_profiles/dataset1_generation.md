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
- Stage 3 keeps the canonical `genui_gen_mobile_a2ui_express_v1.md` contract and
  automatically appends `stage3_training_fidelity_v1.md` inside `run_stage3` for
  every provider. Existing workers need a restart to load changed Python code.
  Newly written records include `generation_fidelity_audit`; prior rows are not
  changed or automatically retried. Unsupported chart subtypes fail the renderer
  gate; literal prose diagnostics require review and never rewrite a target.
- Renderer capability v2.1 adds the chart families and series/axis bindings in
  `dataset/docs/chart_contract_v2_1.md`. Use a fresh generation output directory
  when the saved prompt/contract fingerprint differs; legacy resume checks remain
  strict. Existing v11/v11s release bytes and trained-model prompts stay frozen.
