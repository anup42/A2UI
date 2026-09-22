# QAT LoRA training and official-layout E2B export diagrams

Created 2026-09-22 using built-in image generation (no CLI fallback).
These are presentation PNGs, not editable PowerPoint diagrams.

## Final assets

- [QAT LoRA training](qat_lora_training.png)
- [Official E2B export with MTP support](official_e2b_export_mtp.png)

## Architecture and scope

These diagrams describe the repository's retained-mobile QAT LoRA workflow.
They do not describe the experimental full-parameter fresh-graph exporter or
claim to reproduce Google's private training or calibration process.

The training diagram distinguishes independent prepared response-to-A2UI Express
pairs and the reconstructed BF16 official mobile seed. Only the LoRA A/B
parameters for 205 mapped projections are optimized. Base weights, embeddings,
W8 paths and published quantization scales remain frozen; weight/activation
quantization is simulated in the forward path. Completion-only loss updates
the adapters. The Golden selection set selects the checkpoint, while the
Golden holdout and Bixby test are final-only evaluations.

The export diagram shows adapter merging into the reconstructed BF16 seed,
then re-encoding the mapped W2/W4 projection weight codes with retained scales.
The released package's graph, static A8 layout, frozen constants and default
MTP drafter are preserved. Required hash, precision/code-parity, graph/layout,
frozen-byte and drafter checks precede publication.

**The default MTP drafter is copied unchanged, not trained.** Its presence in
the package does not prove active MTP inference or a speedup. The standard
official training wrapper leaves inference MTP off; enabling MTP requires the
appropriate runtime configuration and on-device validation.

## Repository sources checked

Paths are relative to the repository root:

- `training/docs/OFFICIAL_MOBILE_QAT_PIPELINE.md`
- `training/src/ir_training/pipeline/official_mobile.py`
- `training/src/ir_training/train/sft.py`
- `training/scripts/build_gemma4_retained_scale_litertlm.py`
- `training/configs/pipelines/gemma4_e2b_mobile_mtp.yaml`

The images were visually checked for legible text, independent input arrows,
checkpoint evaluation flow, LoRA-only gradient updates, separate preserved
drafter routing, required export validation and the runtime-MTP caveat.

## Final prompt set

### Training image: initial generation

```text
Use case: infographic-diagram.
Asset type: a high-resolution 16:9 landscape technical presentation slide image, part 1 of a matching two-slide set. White background, polished vector-like technical graphics, generous negative space, large perfectly legible sans-serif text. Restrained navy, blue and teal; amber only for trainable adapters and backward arrows. No photos, no 3D, no watermarks, no company logos. All text must be spelled exactly. Presentation-ready, visually composed, not a document screenshot.

Title: "QAT LoRA Training"
Subtitle: "Gemma 4 E2B • GenUI Craft LM"

Use three broad left-to-right zones joined by clear arrows, with a spacious centered learning-loop diagram occupying the most space.

LEFT zone heading "1  Prepare & verify"
Show two compact stacked input cards:
"Training pairs" / "Agent response → A2UI Express" / "Filter • repair • deduplicate"
"Official mobile seed" / "Reconstruct frozen BF16 weights" / "Retain published quantization scales"
Both feed a slim shield strip "Preflight: data, hashes, scope & numeric safety" then the middle training zone. The shield strip may be below the input cards, with routing arrows that clearly reach the learning loop.

CENTER zone heading "2  Train with quantization simulation"
Inside one large pale-blue rounded panel, two small tiles merge into a plus node:
a blue locked tile "Frozen base W"
an amber tile "Trainable LoRA A / B" / "205 mapped projections"
Formula beneath the plus node, precisely "W_eff = W + (α/r)BA".
Forward arrow into teal tile "Fake quantization" with sublabels "Retained W2 / W4 weights" and "Static A8 input / output simulation".
Arrow into "A2UI Express prediction" then into "Completion-only loss".
Draw one elegant amber backward-loop arrow from loss back ONLY to the LoRA A/B tile, label along arrow "Backpropagation • STE • update A / B only". It must not terminate at the frozen base tile or quantization scales.
Small lock caption at bottom of central panel: "Base weights, embeddings, W8 paths and quantization scales remain frozen".
Interpret W8 paths correctly: their weights remain frozen while input/output activation simulation applies; do not depict W8 as entirely unmodeled.

RIGHT zone heading "3  Select & evaluate"
Vertical tidy stack:
"Golden selection set" / "Choose best unique-source reward"
down arrow
"Best LoRA checkpoint"
down arrow
"Final evaluation" / "Golden holdout + Bixby test"
a small output pill "Ready for retained-scale export"
The held-out test set must NOT point back into training, tuning or checkpoint selection.

At bottom a full-width quiet explanatory band, large readable text:
"QAT simulates deployment precision during training; only LoRA adapters are optimized."
Separate small badge with a lock icon: "MTP drafter: not trained"

Scientific constraints: this is the repository's retained-mobile QAT LoRA workflow, not full-parameter SFT, not QLoRA, and not a claim to reproduce Google's private training process. Training weights use BF16 reconstruction and fake quantization, not packed INT training. Do not add sample counts, TensorBoard, a comparison table, loss scores, speed claims, or decorative AI robots. Keep arrow directions unambiguous and all text away from arrowheads. Render as one single slide, not multiple panels that look like separate pages.
```

### Training image: final targeted edit

Applied to the initial training image. This edited result is the delivered
`qat_lora_training.png`.

```text
Edit the attached training slide with ONLY these targeted flow corrections. Preserve its exact title, subtitle, 16:9 landscape size, crisp typography, colors, card labels, formula, frozen-base lock, amber LoRA backward loop, footer, and overall visual style.
1. In the left Prepare & verify panel, remove the vertical arrow from Training pairs to Official mobile seed. These are INDEPENDENT inputs: training pairs do not create the seed. Instead, draw two independent blue input arrows from those two cards into the Preflight card. Route the Training pairs arrow down a narrow margin outside the Official mobile seed card, without passing through or pointing at it. Keep the Official mobile seed arrow directly into Preflight. You may shift the two input cards slightly right or shrink their widths just enough to leave room for that margin route; do not shrink the text.
2. Connect the central Train with quantization simulation workflow to the right Golden selection set with one clean blue arrow labeled exactly "Periodic evaluation". Place the arrow along the unused space above the forward-loop tiles, coming from the central panel and ending at the LEFT edge of Golden selection set. The arrow represents checkpoint evaluation, NOT completion-loss values. Do not draw it from the loss tile. Ensure the label is readable and the arrow does not cross headings or other text.
3. The Preflight arrow should visibly enter the central training workflow zone, not the frozen explanatory footer.
Do not add other content or change the model-science semantics. The backward amber arrow must still terminate ONLY at Trainable LoRA A / B. The final evaluation holdouts must have no feedback arrow to training or selection.
```

### Export image: generation

```text
Use case: infographic-diagram.
Asset type: a high-resolution 16:9 landscape technical presentation slide image, part 2 of a matching two-slide set. White background, polished vector-like technical diagram, navy titles, blue trained-target route, teal preserved official structure, violet MTP route, subtle amber validation. Large perfectly legible sans-serif text and generous whitespace. No photos, no 3D, no watermarks, no company logos.

Title: "Official E2B Export with MTP Support"
Subtitle: "Retained-scale LoRA → official-layout LiteRT-LM"

Build a clean two-lane input flow that converges into a large packaged-file diagram at right, with a compact optional-runtime strip across the bottom. Approximately 65% of the canvas for export, 20% for optional runtime, remaining for title/margins. Use real arrow routing and clear endpoint connections, with the drafter bypass distinct from trained target weights.

TOP BLUE LANE heading "Trained target"
Three sequential cards:
"Best QAT LoRA adapter"
→ "Merge into frozen BF16 seed" / "W_merged = W + (α/r)BA"
→ "Re-encode 205 projections" / "Use retained W2 / W4 scales" / "Patch weight codes only"
Then an arrow labelled "Updated target weights" into the target compartment of the package on the right.

LOWER TEAL/VIOLET LANE source card:
"Official E2B .litertlm"
From it draw two separate parallel labeled branches:
teal branch "Preserve graph, scales & frozen constants" leading to the protected-structure compartment in the final package.
violet branch "Copy default MTP drafter unchanged" leading directly to the drafter compartment in the final package. Clearly add the lock note "Not retrained" on that violet branch. This branch bypasses LoRA merging and weight-code updates.

RIGHT packaged-file diagram title "Exported .litertlm"
Inside three clearly separated compartments:
blue "Fine-tuned E2B target"
teal "Official W2 / W4 / W8 + static A8 layout"
violet "Default MTP drafter"
Below package, integrated validation shield or band:
"Validation gates"
"Hashes • weight-code parity • frozen bytes"
"Graph / layout identity • byte-exact drafter"
Show this as required BEFORE publication, not after deployment. A clean footer under the export area can state "Published only after all export checks pass".

BOTTOM separated pale-violet strip heading "Optional MTP inference"
Show conceptual runtime sequence, left-to-right:
"Default drafter proposes tokens" → "Fine-tuned target verifies" → "Accept or correct → A2UI Express"
A small device icon may accompany the final label but must not dominate.
Below this sequence, mandatory readable note:
"MTP inference must be enabled and validated on-device; the standard training wrapper leaves it OFF."
And a short secondary note:
"Preserved drafter ≠ trained drafter • Speedup is not guaranteed"

Accuracy constraints: this is the existing retained-scale LoRA export route, NOT the experimental full-parameter fresh-graph exporter. The final target keeps official mixed W2/W4/W8 weights and static A8 activation layout; only 205 mapped projection code buffers change. Original drafter is copied byte-for-byte, is not fine-tuned, and must not be shown as merged into target weights. Export supports retaining MTP, but presence alone must not imply active or validated MTP inference. Do not claim Google's private training/calibration recipe or official-speed parity. Do not add sample counts, TensorBoard, comparison tables, fake benchmark scores, or extra technical claims. Keep exact labels unclipped and arrowheads off all text. One single presentation slide, not a collage.
```
