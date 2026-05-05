# Current Context Snapshot (2026-04-10)

This file captures the active engineering context for the `A2UI` workspace so future agents can resume quickly.

## Branch and Remote

- Branch: `new_ir_changes_20260331`
- Remote: `origin` -> `https://github.com/anup42/A2UI.git`
- Latest pushed commit: `dbd708346d254a685c1be6a1e8321c68f82e4dc0`
- Latest pushed message: `Enforce strict IR-only Stage 3 pipeline and add diagnostics`

## Current Pipeline Direction

- Stage 3 is now strict IR-only (no runtime fallback JSON/UI branch in `GenUiStagePipeline`).
- Invalid Stage 3 output now returns `Outcome.Failure(stage=STAGE3)` with diagnostics.
- Repair flow is bounded multi-pass and records per-attempt validation/backend errors.
- Debug builds persist Stage 3 troubleshooting artifacts under:
  - `files/result/stage3_debug/run_<timestamp>/`

## Prompt and Contract Alignment

- Canonical app prompt:
  - `android/app/src/main/assets/pipeline_prompts/genui_gen.md`
- Dataset mirror prompt:
  - `dataset/prompts/genui_gen.md`
- Prebuild now enforces prompt mirror sync via:
  - Gradle task `verifyGenUiPromptMirror`

## Rendering Compatibility

- Historical text-heavy fallback payloads that use `Stack + Text` are still bridge-parsed in `GenUiNativeRenderer` for backward compatibility.
- Live pipeline generation must not depend on fallback payload creation.

## Data Cleanliness Improvements

- Added mojibake normalization path in `McpResponseFormatter.normalizeForStage3(...)`.
- Stage 2 text is normalized before Stage 3 conversion to reduce corrupted symbols in rendered weather/news/travel text.

## Validation and Test Notes

- Unit/instrumentation assertions were updated for strict no-fallback expectations.
- Build/install status from this session:
  - `:app:installDebug` succeeded on device `R3CT70QA45H` (SM-S916U).

## Local-Only Working Tree Items (Not Committed)

- Modified: `.claude/settings.local.json`
- Untracked: `android/manual_test/` (manual test dumps/screenshots)

These are intentionally local and should only be committed when explicitly requested.

## Suggested Next Resume Actions

1. Run manual E2E checks for:
   - `show weather in bengaluru`
   - flight query (for example: `Show flights from bengaluru to lucknow on 15th may`)
2. Capture fresh screenshots/UI dumps after each run.
3. If weather/flight still render raw text, inspect Stage 3 diagnostics under `files/result/stage3_debug`.

