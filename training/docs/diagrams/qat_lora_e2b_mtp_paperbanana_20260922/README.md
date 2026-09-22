# PaperBanana: QAT LoRA training and official-layout E2B export

This pair recreates the repository-grounded diagrams with PaperBanana and Gemini Pro models.
Source diagrams and technical notes: [earlier imagegen set](../qat_lora_e2b_mtp_20260922/README.md).

## Input briefs

- [QAT LoRA training brief](training_brief.md)
- [Official-layout E2B export and MTP brief](export_brief.md)

The briefs describe the retained-mobile LoRA route, not the full-parameter exporter.
Only LoRA A/B tensors are trained; the official default MTP drafter is preserved unchanged.
MTP package presence does not mean MTP inference is enabled or its speedup established.

## Generation settings

- Tool: PaperBanana skill wrapper, full `demo_full` pipeline.
- Planner, stylist and critic: `gemini-3.1-pro-preview`.
- Image renderer: `gemini-3-pro-image`.
- Authentication: Vertex AI Express; credential supplied only in process environment.
- Aspect ratio: 16:9; one candidate per figure; four critique rounds requested per figure.
- Retrieval: none. Initial auto retrieval downloaded zero reference files and fell back to none.
- PaperBanana upstream revision: `836455537e863b5a2f40dace487a782c0bc5ef94`.
- Skill wrapper applies its standard Vertex Express and four-round visualizer compatibility patches
  only in its isolated managed checkout, outside the A2UI repository.

The initial attempt used the skill-default `gemini-3-pro-image-preview`, which returned HTTP 404.
It was stopped before completion. The retry uses the current Pro model ID
`gemini-3-pro-image` with global routing; no Flash or non-Pro model was substituted.
[Google's Pro Image documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-pro-image)
lists that model ID and global availability.

## Results and verification

- Training: [qat_lora_training.png](qat_lora_training.png), 1376 x 768 pixels.
  Four critique rounds requested; one observed (round 0). The critic reported
  "No changes needed" and stopped early. Wrapper exit code: 0.
  A transient HTTP 429 was recovered by the built-in retry logic.
- Export: [official_e2b_export_mtp.png](official_e2b_export_mtp.png), 1376 x 768 pixels.
  Four critique rounds requested; two observed (rounds 0 and 1). Round 0 revised
  and successfully rendered the figure; round 1 reported "No changes needed"
  and stopped early. Wrapper exit code: 0.

The training image was visually checked for legibility, independent data/seed
inputs, correct effective-weight formula, adapter-only backward flow, frozen
base/scales, checkpoint-selection versus final-holdout separation, and the
untrained-drafter note. No caption/title was embedded in the image.

The export's first GA-model attempt was interrupted after repeated HTTP 429
responses so that the figures could run serially; no model downgrade was used.
The serial retry completed successfully.

The export image was visually checked for the BF16 merge intermediate,
retained-scale re-encoding of 205 projections, preserved official graph/layout,
separate unchanged default-drafter branch, required pre-publication validation,
and explicit optional-runtime MTP caveat. Labels are legible at native size and
the supplied caption is not embedded as the image title.

See [generation_report.json](generation_report.json) for completion evidence,
actual model IDs, critique counts and image hashes. The files are presentation
PNG assets, not editable PowerPoint diagrams. No training or export code was changed.
