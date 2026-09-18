# GenUI Craft LM project-status slide

- Image: `06_genui_craft_lm_project_status.png`
- Created: 2026-09-16
- Method: built-in image generation, using presentation-layout guidance for a readable single-slide composition.
- Source: the user's supplied project-status photograph.
- Scope: visual redesign and condensed wording, not a fresh audit of project completion or dataset counts. Status assignments and figures are user-supplied.
- Preserved figures: 500K audited IR dataset, 112,842 training samples, 2,300 validation samples, cloud 31B baseline, Gemma E2B / 270M, 2 / 4 / 8 GPUs, W32 / W16 / W8 / W4.
- Terminology: the input's 4B / 8B quantization wording is expressed as 4-bit / 8-bit to distinguish bit width from model parameter count.
- Visual review: checked section headings, item coverage, numerical values, text fit and absence of overlap. Output is a raster slide image, not an editable PowerPoint deck.

## Final generation prompt

```text
Use case: productivity-visual.
Create one polished 16:9 landscape presentation slide image at high resolution.
Image 1 is a content reference only: a photographed project-status slide. Redesign it completely for readability and professional presentation. Do NOT reproduce the photograph, diagonal watermark, selection handles, perspective distortion, glowing borders or blurry text.

Title: "GenUI Craft LM Project Status"

Layout:
A clean white slide with two unequal columns: Completed uses about 62% of the content width, In Progress about 38%. Use wide margins and a quiet vertical divider. Keep a single flat editorial composition rather than a dashboard of little cards.
Large section heading "COMPLETED" with one green check-circle icon and a thin green rule below.
Large section heading "IN PROGRESS" with one blue progress-circle icon and a thin blue rule below.
Each column contains THREE vertically arranged workstream groups, with clear bold navy group headings, well-spaced concise text, and subtle horizontal separators. The group headings align in three rows across the two columns. Use restrained small line icons at group headings if helpful, never large decorative artwork. Dark navy typography on white, emerald/teal accents for completed, royal blue accents for work in progress. Modern clean sans-serif, strong contrast, no gradients or glow. Keep body copy large enough to read in a meeting.

Render this copy accurately, without inventing results. Preserve every item and every numerical value. Wording below is the final approved slide copy.

LEFT COLUMN — COMPLETED

Group 1 heading: "Architecture & Data"
Three short items:
"Architecture and IR specification finalized"
"Data-generation pipeline with quality checks and shared A2UI Express parsing / validation"
"500K IR dataset audited and filtered"
Under the last item, emphasize the numbers on one readable line:
"112,842 training + 2,300 validation"
Small supporting line immediately below:
"Samples retained after decontamination"

Group 2 heading: "Modeling & Training"
Three items:
"Cloud 31B baseline selected; compact models evaluated"
"Gemma E2B and 270M trained"
"2 / 4 / 8 GPU training with caching, augmentation and sequential hyperparameter tuning"
Use a natural line break in the final item if necessary. Do not use giant model illustrations.

Group 3 heading: "On-Device Infrastructure"
Three items:
"Native Android Compose renderer implemented"
"Metrics, checkpoint evaluation and golden tests automated"
"LiteRT-LM W32 / W16 / W8 / W4 exports and GPU inference for E2B / 270M"

RIGHT COLUMN — IN PROGRESS

Group 1 heading: "Model Quality"
Three short items:
"Dataset improvement"
"Fine-tuning"
"Knowledge distillation"

Group 2 heading: "Performance & Optimization"
Three short items:
"Model KPI improvement"
"4-bit / 8-bit quantization performance"
"Overall latency reduction"

Group 3 heading: "Patents & Research"
Two short items:
"Patent pipeline"
"Research publication"

Design emphasis:
The reader should instantly understand what is completed and what is still being worked on. Visually distinguish the two statuses using typography and restrained accent colors, not noisy repeated loading spinners.
Use compact, consistent small checkmarks for completed items and small simple blue open-circle bullets for in-progress items.
The dataset counts must stand out, but must NOT imply that 500K samples remain in the final training set.
No extra footer, date, claims, percentages, citations, invented milestones or slogans. No clipping, no overlapping text, no spelling errors, no watermark. Preserve the input's status assignments. This is a clean slide image, not a photographed slide or UI dashboard.
```
