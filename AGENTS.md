# A2UI Project Notes

## Core Pipeline
- `dataset/` generates and evaluates UI data in stages:
  - Stage 1: query generation.
  - Stage 2: response generation and media/assets.
  - Stage 3: flat-spec GenUI IR generation.
  - Stage 4: render/metrics/dashboard artifacts.
- Android uses the flat-spec contract shape: `{"root": "...", "state": {...}, "elements": {...}}`.
- Current preferred dataset Stage 3 prompt is `dataset/prompts/genui_gen_mobile_flatspec_v11.md`.
- Android runtime prompt copy is `android/app/src/main/assets/pipeline_prompts/genui_gen.md`.
- When fixing a generated sample, do not manually rewrite the IR JSON. Fix the prompt/pipeline/renderer, regenerate the IR through Stage 3, then merge the generated record.

## Training
- `training/` owns response-to-IR model training code. It consumes completed dataset runs and should not generate Stage 1/2/3 data itself.
- Keep training code model-agnostic: add new LLM families through `ir_training.models.ModelAdapter` implementations and YAML configs.
- Default training source is `dataset/data/runs/dataset_v1`; input is Stage 2 `response_text`, output is strict flat-spec `genui_json`.
- Do not commit training outputs/checkpoints unless a small manifest/report is intentionally needed.

## LLM/Auth
- Dataset Gemini calls should use Vertex Express API keys, not AI Studio direct keys.
- Env keys used in this repo include `VERTEX_EXPRESS_API_KEY` and `GEMINI_VERTEX_EXPRESS_API_KEY`; project id is `VERTEX_PROJECT_ID`.

## Renderer
- Primary Android flat renderer is `android/app/src/main/java/com/samsung/genuicraft/renderer/FlatSpecRenderer.kt`.
- Wrapper/native payload parsing is in `android/app/src/main/java/com/samsung/genuicraft/renderer/GenUiNativeRenderer.kt`.
- Tables should stay compact in IR (`Table.props.columns`, `statePath` or `rows`) and render through native table/card routes.
- Domain hints: `weather`, `flight`, `booking`, `schedule`, `status`, `comparison`, `generic`.
- Weather/flight/booking/schedule/status prefer cards. Comparison uses cards for entity rows and table for feature matrices.
- Keep IR compact: avoid expanded cell trees, duplicate prose, and unnecessary `sourceText`.

## Stitch-Guided Runs
- Stitch reference category images are in `Stitch_img/`.
- Stitch-generated 50-sample outputs are under `stitch/outputs/golden50_g25pro_20260309_204033_android_promptsync_20260413_010634_stitch_20260414_195050/`.
- Current comparison run is `dataset/data/runs/golden50_g25pro_20260309_204033_stitch_compare_20260417_r1`.
- Android rendered screenshots for a run live in `<run>/android_device_rendered/`.
- Quality reports may include `quality_gaps.csv`, `quality_gaps.json`, `before_vs_after.md`, and `stitch_one_by_one_comparison.csv`.

## Git Hygiene
- Do not revert user/unrelated changes.
- Avoid committing `dataset/data/cache`.
- Generated run folders may be intentional artifacts; inspect before excluding them.
- Temporary scratch files should not be committed.
