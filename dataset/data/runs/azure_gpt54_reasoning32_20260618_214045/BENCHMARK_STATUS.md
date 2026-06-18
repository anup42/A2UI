# Azure GPT-5.4 Reasoning32 Benchmark Status

Run group: `azure_gpt54_reasoning32_20260618_214045`

Goal: generate 32 Stage 1 queries, 32 Stage 2 responses, and 32 Stage 3 IRs for each model/mode using the Gemma31B Stage 3 prompt (`prompts/genui_gen_gemma_v12_structure_preserve.md`).

## Completed

- `gpt54_mini_no_reasoning`: 32 queries / 32 responses / 32 IRs
- `gpt54_mini_reasoning_medium`: 32 queries / 32 responses / 32 IRs
- `gpt54_no_reasoning`: 32 queries / 32 responses / 32 IRs
- `gpt54_reasoning_medium`: 32 queries / 32 responses / 32 IRs

## Partial

- `gpt54_pro_no_reasoning`: 32 queries / 4 responses / 0 IRs
- `gpt54_pro_reasoning_medium`: 32 queries / 4 responses / 0 IRs

## Notes

- `gpt-5.4-pro` Stage 1 was initially very slow with 4096 query output tokens. It completed after reducing Stage 1 output cap to 1024, which is sufficient for short query JSON.
- `gpt-5.4-pro` Stage 2 was too slow in per-query mode and returned no final text in experimental single-call batch mode for Stage 2 response prompts.
- Azure adapter was updated to support environment-driven reasoning effort, larger output caps, configurable timeout, and opt-in single-call batch wrapper.
- `summary.csv` and `summary.json` contain the current aggregate/status table.

## Useful files

- `summary.csv`
- `summary.json`
- `benchmark_manifest.json`
- Per-run: `queries.jsonl`, `responses.jsonl`, `genui.jsonl`, `aggregates.json`
