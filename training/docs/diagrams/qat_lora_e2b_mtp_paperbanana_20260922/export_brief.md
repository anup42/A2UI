# Figure brief: official-layout E2B export with default MTP drafter

Create a clean publication-quality technical flowchart for a widescreen 16:9 presentation.
White background, navy labels, blue updated target weights, teal preserved topology/scales,
violet unchanged MTP branch, restrained amber validation. Large legible sans-serif labels,
aligned boxes and uncluttered arrows. No photos, fake benchmark scores, sample counts,
TensorBoard or comparison tables. The caption is metadata: DO NOT embed the caption as a title.
Small panel headings are welcome. Match the companion training diagram's restrained style.

## Verified method

This describes the repository's retained-mobile QAT LoRA export route, NOT the experimental
full-parameter fresh-graph exporter and NOT a reconstruction of Google's private training recipe.

A. Trained target lane:
Best QAT LoRA adapter -> merge into frozen reconstructed BF16 seed.
W_merged = W + (alpha/r)BA.
Re-encode 205 mapped W2/W4 projection weight-code buffers using retained published scales.
Patch only these code buffers into a copy of the released official E2B LiteRT-LM package.
The BF16 merged model is an intermediate; it is not the final deployment package.

B. Preserved official package lane:
The official E2B .litertlm package independently supplies:
- Graph/topology, static A8 activation scales, remaining frozen constants and metadata.
- Default MTP drafter section, copied unchanged BYTE-FOR-BYTE.
The drafter bypasses adapter merging and target weight-code updates. It is NOT trained.
Released mixed W2/W4/W8 weight layout and static A8 activation layout are preserved.
Do not show graph/scales as an alternative to the updated target; these are parts of the same package.

C. Required validation before publication:
Source hashes and provenance, weight-code parity and physical precision, unchanged frozen bytes,
graph/layout identity, byte-exact default drafter. All must pass.
The final artifact has three visually clear components:
"Fine-tuned E2B target", "Official W2/W4/W8 + static A8 layout",
"Default MTP drafter (unchanged)".
Do not imply a trained or fine-tuned drafter.

D. Optional inference strip:
Default drafter proposes candidate tokens -> fine-tuned target verifies -> accept/correct -> A2UI Express IR.
Conceptual runtime flow only, no speed or accuracy claims.
Required readable caveat:
"MTP inference must be enabled and validated on-device; the standard training wrapper leaves it OFF."
Secondary concise note: "Drafter preserved, not retrained. Speedup is not guaranteed."

## Composition

Upper area: two distinct converging export lanes, with a conspicuous violet unchanged-drafter bypass.
Right: output package compartments. Required validation is a publication gate, not an optional afterthought.
Bottom: separated optional MTP inference strip.
Keep all endpoints and arrowheads unambiguous, and separate backward training concepts from export:
there is no training/backpropagation in this figure.
