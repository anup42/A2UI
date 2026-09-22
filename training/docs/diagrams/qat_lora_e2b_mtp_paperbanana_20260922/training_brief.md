# Figure brief: QAT LoRA training

Create a clean publication-quality technical flowchart for a widescreen 16:9 presentation.
White background, navy/blue frozen or verified paths, amber trainable LoRA path, teal quantization simulation.
Large legible sans-serif labels and generous whitespace. No photos, AI robots, decorative gradients, watermark,
sample counts, TensorBoard or comparison tables. The figure caption is metadata; DO NOT embed the caption
as a heading inside the image. Small panel headings are welcome. Treat this as a diagram, not a document.

## Verified method

This is GenUI Craft LM's Gemma 4 E2B retained-mobile QAT LoRA workflow in A2UI, NOT full-parameter SFT,
QLoRA, or a claim to reproduce Google's private training/calibration process.

1. Independent input branches:
   - Prepared training pairs: agent response -> A2UI Express IR, after filtering, repair and deduplication.
   - Official mobile seed: manifest-verified BF16 reconstruction of released packed official mobile weights,
     with retained published quantization scales. Data does not create or train this seed.
   Both enter strict preflight checks for data, hashes, mapped scope, and numeric safety.
2. Training:
   Frozen BF16 base W plus trainable LoRA A/B over exactly 205 mapped projections.
   Effective weight W_eff = W + (alpha/r)BA.
   Forward path applies retained W2/W4 weight fake quantization and fixed static A8 input/output simulation.
   W8 per-layer paths also simulate input/output A8 while their weights remain frozen.
   Agent response and effective quantized model produce A2UI Express IR.
   Completion-only SFT loss (prompt tokens masked) backpropagates through STE to LoRA A/B ONLY.
   Base weights, embeddings, W8 paths and quantization scales remain frozen.
   Draw an amber backward arrow from loss to LoRA A/B, never to base weights or scales.
3. Checkpoint selection:
   Periodic Golden selection-set generation reward (unique-source reward) chooses the best LoRA checkpoint.
   Golden holdout + Bixby test are FINAL-only evaluations of the selected checkpoint.
   Do not feed held-out tests back into training or selection.
4. Output is a best LoRA adapter ready for retained-scale export, not a packed trained model.

## Suggested concise labels

Independent cards: "Prepared response -> IR pairs" and "Official BF16 mobile seed".
Panel labels: "Prepare & verify", "QAT LoRA learning loop", "Select & evaluate".
Learning nodes: "Frozen base W", "Trainable LoRA A/B", "205 mapped projections",
"Retained W2/W4 fake quantization", "Static A8 simulation", "A2UI Express prediction", "Completion-only loss".
Use a clearly connected forward flow and a clearly separate adapter-only backward loop.
Connect the trained model to "Periodic Golden selection" -> "Best LoRA checkpoint"
-> "Golden holdout + Bixby test". Never connect training data directly to seed creation.
Footer note: "Only LoRA adapters are trained. Base weights and quantization scales stay frozen."
Small separate lock note: "Default MTP drafter is not trained".
