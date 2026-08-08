# Gemma QAT, LoRA, INT4, and MTP Knowledge Base

This handoff also includes a static parity harness for the complete 0-34
language MLP range.

Last official-source verification: **2026-08-08**

This is the durable handoff for agents maintaining the A2UI response-to-flat-spec
training and inference scripts. It records what is officially supported, what
this repository implements, what remains experimental, and the promotion gates.

## Overall recommendation

For custom A2UI accuracy and faster low-bit inference, use this sequence:

> Fine-tune only the QAT-derived target with conservative BF16 LoRA, merge the
> selected adapter, convert the merged target to the exact deployment format,
> and first re-use Google's frozen matching QAT assistant. Promote MTP only if
> the tuned target keeps task accuracy and the final runtime/device shows a real
> end-to-end speedup. If assistant acceptance or throughput degrades, reduce
> target drift or ship target-only INT4. Treat custom assistant training as a
> separate experimental public reconstruction, never as Google's recipe.

Keep Google's untouched QAT target plus matching assistant as the speed and
quality control. Do not jointly update target and assistant weights: the
repository's optional assistant stage freezes the selected target.

## Short support determination

Google officially provides:

- QAT-derived Gemma 4 targets in unquantized, Q4_0 GGUF, W4A16 compressed-
  tensors, and mobile-optimized families;
- matching Gemma 4 MTP assistant checkpoints;
- Transformers inference using `assistant_model=assistant_model`;
- ordinary PEFT/LoRA or QLoRA fine-tuning guidance;
- runtimes such as LiteRT-LM, llama.cpp ecosystem formats, vLLM, and SGLang.

Google does **not** currently provide a public, executable end-to-end recipe for:

- continued Gemma 4 E2B Q4_0 QAT after domain fine-tuning;
- joint target and specialized Gemma 4 assistant training;
- the assistant distillation loss, schedule, target-activation interface, and
  matched exporter needed to reproduce the released drafter;
- automatically recreating the private mobile wNa8o8 QAT process after arbitrary
  custom fine-tuning;
- guaranteeing MTP acceptance or latency after changing the target weights.

The legacy `qat_mtp` profile in this repository remains **QAT-derived target
tuning**, not continued QAT or joint QAT+MTP training. The new opt-in `qat`
profiles implement repository-owned STE fake quantization during LoRA SFT for
Gemma 4 E2B and Gemma 3/FunctionGemma 270M. They are useful accuracy-adaptation
experiments, but they do **not** recreate Google's private mobile wNa8o8
training/export recipe. Those target profiles do not train an MTP assistant;
the separate opt-in drafter workflow documented below is experimental.

## Exact checkpoints for the Q4_0 reference path

Target:

```text
google/gemma-4-E2B-it-qat-q4_0-unquantized
```

Matching assistant:

```text
google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant
```

The `-unquantized` checkpoints contain half-precision weights extracted from
Google's QAT pipeline. They are intended for research and downstream
compilation. Loading them in BF16 does **not** load packed Q4_0 weights. Both
members of an MTP pair must be converted to a compatible matching precision and
runtime representation.

## Terminology that must remain precise

### Quantization-aware training

QAT simulates the intended deployment quantization during weight updates so the
model can adapt to quantization error. A true continued-QAT implementation would
need the exact fake-quantization operations, scale/granularity rules, excluded
layers, activation policy, optimizer behavior, and exporter used by the final
runtime.

The legacy `qat_mtp` workflow does not implement those operations; the separate
`qat` workflow below adds a generic STE approximation and still requires exact
post-training export/runtime validation.

### QAT-derived LoRA

The recommended profile begins with BF16 weights produced by Google's QAT
pipeline, freezes those base weights, and updates LoRA parameters. It may retain
some useful QAT-conditioned weight structure, but fake quantization is not active
during adaptation. It is an engineering compromise and must be evaluated after
conversion.

### True QAT path now implemented here

`ir_training.qat.fake_quant` wraps every eligible frozen base-model
`nn.Linear` after PEFT preparation, including ordinary architecture linears
that do not acquire a literal `base_layer` suffix. It also wraps configured
embedding tables. For a PEFT projection it constructs
`base_weight + LoRA_delta`, fake-quantizes that effective matrix once, and then
runs the linear operation. This is materially different from the earlier
`fake_quant(base_weight) + floating_LoRA_branch` approximation because the
final exporter quantizes the merged matrix. Each forward pass rounds/clamps to
the configured bit width and applies a straight-through estimator (STE), so
gradients still reach both LoRA matrices. The
default remains symmetric per-output-channel W8A8 with dynamic abs-max scales;
the Gemma 4 mobile config opts into the artifact-reconciled module map and
`ste_ai_edge` range convention (full signed W2/W4, narrow symmetric W8). LoRA
A/B weights stay floating point, which keeps this a practical QAT+LoRA method
rather than a claim that every adapter operation is already mobile-quantized.
The observable per-layer embedding is the one granularity exception: its
`[262144, 35]` public scale table covers 256-column groups in the logical
`[262144, 8960]` weight, so the LiteRT-LM schema assigns
`embed_tokens_per_layer` a module-specific `group_size: 256`. The training
wrapper fake-quantizes only the unique embedding rows referenced by the current
batch, rather than rebuilding the roughly 4.7 GiB BF16 table on every lookup;
the selected rows still use the same row-local 256-column groups. Other mapped
matrices retain the observed per-output-channel scale behavior.
All effective-weight QAT profiles require `lora.dropout: 0.0`; a per-example
adapter dropout mask cannot be represented by one final merged inference
matrix. DoRA and mixed-adapter forward arguments fail closed for the same
reason.

For Gemma 4 E2B, keep `target_modules: peft-default` with the pinned PEFT
0.19+ environment. Its Gemma 4 mapping is scoped to `language_model` query and
value projections. Do not substitute the adapter's historical `.linear`
fallback: current Transformers uses those inner linears for clipped audio and
vision wrappers, while the language decoder projections are ordinary
`nn.Linear` modules. Any future PEFT scope change must be verified against the
actual loaded module names before training.

The implementation is reversible: wrappers are restored before the final PEFT
adapter is saved, and `qat` metadata records the wrapped module count, exact
module-to-bit assignments, module-specific group sizes, W2/W4/W8 histogram,
effective LoRA wrapper count, uncovered-adapter list, and quantizer
specification. Training metadata also
hashes each saved adapter checkpoint. Current merge manifest v4 verifies that
metadata, the training-config hash, adapter hashes, canonical model identity,
local mobile-seed source/manifest, and exact checkpoint-key loading before
loading the base model.
`load_in_4bit` is rejected because NF4 QLoRA is a different method. This is
true fake-quantization-aware training in the engineering sense, but the final
LiteRT/LiteRT-LM converter remains the authority for packed layout,
calibration, static activation scales, and targeted 2-bit layers.

### QLoRA

QLoRA normally freezes a base stored in BitsAndBytes NF4 and trains floating-
point adapters. It reduces training memory but is not QAT, and NF4 is not Q4_0,
mobile wNa8o8, or server W4A16. The recommended profile keeps
`load_in_4bit: false`; use QLoRA only as an explicit VRAM-constrained fallback
and re-run all post-conversion gates.

### MTP in Gemma 4

Gemma 4 MTP is a specialized four-layer autoregressive drafter used for
speculative decoding. It consumes target activations, shares/reuses target state
such as embeddings and KV information, and proposes tokens that the full target
verifies. Correct speculative decoding preserves the target distribution; MTP
is an inference-speed technique, not an accuracy-improvement objective.

Target fine-tuning can change activations and next-token probabilities. The
frozen assistant may therefore propose fewer accepted tokens. Output correctness
can remain target-verified while the speed benefit disappears.

## Deployment formats are not interchangeable

| Intended deployment | Google checkpoint family | Important boundary |
|---|---|---|
| llama.cpp / local CPU, Apple Silicon, consumer GPU | QAT Q4_0 GGUF | Custom tuned target must be reconverted; verify assistant conversion/runtime support. |
| vLLM / SGLang server | QAT W4A16 compressed tensors | Do not substitute a Q4_0 assistant merely because both are called 4-bit. |
| Android / LiteRT-LM | Mobile wNa8o8/mobile-ct | Uses static activations, channel-wise rules, optimized KV/cache behavior, and targeted 2-bit layers; it is not ordinary Q4_0. |
| Transformers research/reference | Q4_0 `-unquantized` target and assistant | BF16 memory/timing is not final packed-INT4 performance. |

Google's current overview estimates E2B load memory at roughly 11.4 GB in BF16,
2.9 GB in Q4_0, 1.1 GB for the mobile package, and 0.84 GB for its text-only
mobile configuration. Treat these as routing estimates, not guarantees for a
particular device or context length.

## Repository implementation

### Recommended target config

`training/configs/models/gemma4_e2b_ir_qat_lora.yaml`

Key decisions:

- official Q4_0 QAT-derived target;
- `load_in_4bit: false`;
- BF16 with the training runner's existing FP16 fallback;
- target-only completion-masked SFT;
- thinking disabled for compact machine-facing JSON;
- LoRA rank 16, alpha 16, dropout 0.05;
- PEFT Gemma 4 LM defaults instead of blanket `all-linear`;
- frozen embeddings and LM head because this task adds no tokenizer tokens;
- initial learning rate 1e-4, with 5e-5 as the first comparison;
- best adapter selected through the held-out golden50 overall score;
- exact frozen QAT assistant recorded in the `qat_mtp` block;
- explicit `continued_qat: false`, `train_assistant: false`, and final runtime
  validation requirement.

The standard `gemma4_e2b_ir_lora.yaml` remains a separate baseline. Do not
silently turn every Gemma 4 run into the QAT-derived experiment.

### True-QAT target profiles

The new configs are intentionally separate from the QAT-derived/MTP profile:

| Config | Base model | Training fake quantization | Intended final export |
|---|---|---|---|
| `gemma4_e2b_mobile_seed_ir_qat_sft.yaml` | Manifest-verified BF16 reconstruction of `google/gemma-4-E2B-it-qat-mobile-transformers` | Public W2/W4/W8 module map, AI Edge-compatible STE ranges, W8A8 activation edges | Preferred official-topology transplant; private Google QAT/calibration remains unrecovered and device gates remain required |
| `gemma4_e2b_ir_qat_sft.yaml` | `google/gemma-4-E2B-it-qat-q4_0-unquantized` | Same observable training approximation | Rejected research baseline for mobile transplant: 212/262 retained values differ |
| `gemma3_270m_ir_qat_sft.yaml` | `google/gemma-3-270m-it` | W8 STE per-channel weights with activation fake quantization disabled (FP32 edges) | exact released Q8 topology via the checkpoint compiler; still requires device validation |
| `functiongemma_270m_ir_qat_sft.yaml` | `google/functiongemma-270m-it` | W8A8 STE, per-channel weights | custom dynamic INT8 export; compare with the official Q8 LiteRT-LM package |

The official Gemma 3 270M Q4_0-derived checkpoint
(`google/gemma-3-270m-it-qat-q4_0-unquantized`) is a different starting point
from `google/gemma-3-270m`; use it for a Q4_0 experiment, not as evidence that
the LiteRT-LM Q8 package was reproduced.

Run only the no-model validation during code setup:

```powershell
python training/scripts/validate_qat_training.py
```

Later, when a GPU run is explicitly authorized, use one of:

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml
python training/scripts/train_sft.py --config training/configs/models/gemma3_270m_ir_qat_sft.yaml
python training/scripts/train_sft.py --config training/configs/models/functiongemma_270m_ir_qat_sft.yaml
```

These commands are training commands and are not part of static validation.
After training, merge the adapter, run the matching LiteRT export, and compare
target-only accuracy and latency against the untouched official low-bit model.
For Gemma 4, retain the official assistant only as a separately validated
speculative-decoding experiment; this QAT path does not modify or train it.

### Public-schema reverse engineering

The official `google/gemma-4-E2B-it-qat-mobile-transformers/config.json` exposes
a public *deployment schema* even though it does not expose the private QAT
training recipe. It records `quant_method: gemma`, a 4-bit default, embedding
quantization, ordered module-specific overrides, and modules excluded from
conversion. Keep the exact public copy for source-parity audits:

`training/configs/quantization/gemma4_e2b_mobile_public_schema.yaml`

Training uses a separate schema that starts from that public map and reconciles
it against the released `.litertlm` target inventory:

`training/configs/quantization/gemma4_e2b_mobile_litertlm_schema.yaml`

Public source anchors are the [Google mobile checkpoint card](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers), its [released config](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/blob/main/config.json), and the [AI Edge Quantizer recipe source](https://github.com/google-ai-edge/ai-edge-quantizer/blob/main/ai_edge_quantizer/recipe.py). The card names the mobile format `wNa8o8`, calls the Transformers weights a reference for other formats, and confirms targeted 2-bit layers plus static activations; it does not publish the QAT trainer or LiteRT-LM exporter.

The observable assignments are:

| Module family | Public bits |
|---|---:|
| `language_model.layers.0-14.mlp.*` | 4 |
| Other language-model MLPs | 2 |
| Language-model self-attention | 4 |
| Per-layer input gate/projection | 8 |
| Global `per_layer_model_projection` | 8 in the released `.litertlm` (local artifact override) |
| Token embeddings / `lm_head` | 2 (per-layer embeddings are 4) |
| Vision tower | 8 |
| Audio tower | 2, with `lconv1d.linear_start` at 4 |

The public Transformers config excludes `per_layer_model_projection`, but the
released target graph contains that matrix as W8. The separate artifact schema
therefore removes that exclusion and adds an explicit W8 rule while leaving the
public copy unchanged. This is a documented artifact-level override, not a
claim about Google's private trainer. Rule order is part of the contract: the
4-bit MLP rule must precede the generic 2-bit MLP rule. The public-schema audit
is model-free:

```powershell
python training/scripts/audit_gemma4_mobile_schema.py
python training/scripts/audit_gemma4_mobile_schema.py --official-config C:\path\to\official\config.json --strict
python training/scripts/audit_gemma4_mobile_schema.py --module-list module_names.txt
```

This can prove that a copied config matches the public module map. It cannot
prove the hidden observer/scales, training schedule, calibration corpus, KV
cache implementation, compiler lowering, or LiteRT-LM runtime behavior. The
QAT implementation can consume `qat.schema_path` for module-specific fake-
quant bit selection, but unsupported module types and exact scale semantics
must remain marked as approximation until numerical equivalence is measured
against the official checkpoint/runtime.

The checked-in `gemma4_e2b_mobile_seed_ir_qat_sft.yaml` selects `quantizer: ste_ai_edge`,
`quantize_embeddings: true`, and the artifact-reconciled schema. This makes the
training-time fake quantizer use the public AI Edge range convention: signed
W2/W4 keep the full low-bit ranges (`[-2, 1]` and `[-8, 7]`), while symmetric
W8 uses the narrow range `[-127, 127]`. It is the closest public numerical
target for a subsequent converter pass, not proof of Google's private QAT
observer schedule or static activation calibration.

### Public AI Edge recipe versus the mobile artifact

The public `ai-edge-quantizer` source contains a named `gemma4_mixed48`
LiteRT-LM recipe. At the pinned 0.8.0 source revision it is:

- `tf_lite_embedder`: channelwise symmetric W4 `EMBEDDING_LOOKUP`;
- `tf_lite_per_layer_embedder`: channelwise symmetric W4
  `EMBEDDING_LOOKUP`;
- `tf_lite_prefill_decode`: channelwise symmetric W4 `FULLY_CONNECTED`, with
  a `per_layer` regex override to channelwise symmetric W8;
- `compute_precision=INTEGER`, `explicit_dequantize=false`, and
  `min_max_uniform_quantize` for those entries.

The generated JSON copy is
`training/configs/quantization/gemma4_mixed48_public_recipe.json`. The same
public module also exposes `gemma4_mixed48_hr` (Hadamard rotations) and
blockwise `gemma4_mixed48_b32`/`_b64` variants. These public recipes do not
contain the mobile artifact's W2 assignments, do not describe the separate MTP
graph, and do not expose Google's private mobile calibration/QAT process.

The local official `gemma-4-E2B-it.litertlm` was inspected directly on
2026-08-07. Its observable graph is different from `gemma4_mixed48`:

- `tf_lite_embedder` is INT2 with 262,144 per-axis scales, while
  `tf_lite_per_layer_embedder` is INT4;
- the prefill/decode graph contains INT2, INT4, and INT8 fully-connected
  weights; quantized fully-connected inputs and outputs are INT8 with one
  symmetric scale/zero point;
- the MTP drafter is a separate graph with INT4/INT8 weights and INT8 edges;
- no `aeq.hadamard_rotation` custom op or `_hadamard_matrix` tensor name was
  found in the inspected graphs.

The raw buffers add one more observable detail: all inspected weight zero
points are zero; W8 rows reach the narrow symmetric endpoint 127, while W2/W4
rows use the full low-bit signed storage ranges (including -2/-8). That agrees
with the public AI Edge low-bit packing convention and the public min/max
uniform scale shape, but it still does not expose the pre-QAT floating weights
or the private observer/calibration implementation.

The reproducible read-only audit is:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/audit_litertlm_recipe.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --strict-mobile `
  --output C:\temp\gemma4-mobile-recipe-audit.json
```

This proves the observable W2/W4/W8-A8 layout and the MTP section, not the
hidden scale-generation source, calibration corpus, optimizer schedule, or
learned values. Do not call a `gemma4_mixed48` export an exact copy of
Google's mobile `.litertlm` artifact.

The JSON report's `fully_connected_assignments` field is the useful reverse-
engineering output: it groups observed weight dtypes by layer and operation
family. On the inspected artifact it shows MLP W4 for layers 0-14, MLP W2 for
layers 15-34, attention W4 throughout, and per-layer embedding projections W8;
the global entries show the INT2 language head and W8 per-layer model
projection. These are observable deployment assignments, not proof of the
private QAT/calibration implementation that produced them.

For the inspected `tf_lite_mtp_drafter`, the assignment is also explicit: four
draft layers use INT4 for their three MLP matrices and INT8 for their two
self-attention matrices; the MTP pre/post projections are INT8, and its draft
LM head is INT4. The section has 30 subgraphs and 365 operators. This is the
compiled draft topology that the random-weight parity harness preserves; it is
not evidence that a generic assistant-model training loss would reproduce the
released drafter weights.

The same dated snapshot is recorded in
`training/configs/quantization/gemma4_e2b_mobile_observable_contract.yaml`.
Treat it as evidence metadata and re-run the audit for a different device or
artifact variant; do not pass it to an exporter as if it were Google's private
recipe.

For a converter-only experiment that mirrors this observable contract on a
small random graph, run:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_random_mobile_contract_parity.py `
  --output-dir C:\temp\random-gemma4-mobile-contract `
  --official-artifact C:\path\to\gemma-4-E2B-it.litertlm `
  --execute
```

This harness statically calibrates a deterministic Keras graph with random
inputs, assigns W4 by default, W2 to the `layer_15_mlp`-through-`layer_34_mlp`
range, W8 to
`per_layer_projection`, and A8 to every fully connected edge. Because the
public AI Edge policy rejects arbitrary static W2 `FULLY_CONNECTED` configs,
the synthetic recipe explicitly sets `skip_checks=true`; that flag is an
experiment warning, not evidence that Google used the same setting. A passing
`synthetic_contract_match` proves only FlatBuffer dtypes, per-axis scale
layout, and INT8 activation edges. The report always records
`private_recipe_recovered=false`, `training_executed=false`, and
`exact_official_mobile_reproduction=false`.

### Official-topology random-weight parity

The repository now has a stronger artifact-level experiment:
`training/scripts/build_random_official_topology_parity.py`. It extracts a
TFLite section from the supplied `.litertlm`, keeps the official FlatBuffer
topology and tensor metadata intact, then replaces observable per-axis
quantized weight buffers with deterministic random weights. The replacement
uses the public AI Edge symmetric axis-0 rule and its low-bit packing order
(W2/W4 use the full signed low-bit range; W8 uses the narrow signed range).
It patches only a new standalone `.tflite` file and never overwrites the
official package.

The experiment was executed against the local Gemma 4 E2B artifact on
2026-08-07:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_random_official_topology_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --model-type tf_lite_prefill_decode `
  --output C:\temp\gemma4-prefill-random.tflite `
  --execute `
  --verify-weight-encoding

python training/scripts/build_random_official_topology_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --model-type tf_lite_mtp_drafter `
  --output C:\temp\gemma4-mtp-random.tflite `
  --execute `
  --verify-weight-encoding
```

Observed on the provided target: the prefill/decode section randomized 277
weight buffers (145 INT4, 61 INT2, and 71 INT8) and the MTP section randomized
23 buffers (13 INT4 and 10 INT8). For both sections, the structural graph hash
and quantization-layout hash matched the official section exactly; the raw
section and quantization-value hashes changed as expected. The embedder was
also checked separately (one INT2 table, with identical structure/layout).

The optional `--verify-weight-encoding` check independently unpacked and
repacked every observable Gemma 4 prefill weight buffer (277/277), every MTP
weight buffer (23/23), and the embedder table (1/1), with zero byte-roundtrip
mismatches. The decoded ranges were W2 `[-2, 1]`, W4 `[-8, 7]`, and W8
`[-127, 127]`, confirming the public AI Edge low-bit byte order and signed
storage ranges at the artifact level.

The same harness understands the newer external-buffer form of the TFLite
schema (`Buffer.offset`/`Buffer.size`, with an empty inline `Buffer.data`). That
matters for the local `gemma-3-270m-ir-int8.litertlm` candidate: its 237 large
constant ranges are externalized inside the single TFLite section. Running the
harness against that candidate randomized 127 unique per-axis INT8 weight
buffers; its graph had 215 subgraphs, 3,046 operators, and 4,611 tensors, and
the structural/quantization-layout hashes remained identical. This is useful
evidence that the external packing can be decoded and rewritten, but the file
is a custom local export rather than the gated official LiteRT Community Q8
artifact, so it is not an exact Google 270M recipe match.

```powershell
python training/scripts/build_random_official_topology_parity.py `
  C:\path\to\gemma-3-270m-ir-int8.litertlm `
  --model-type tf_lite_prefill_decode `
  --output C:\temp\gemma270m-prefill-random.tflite `
  --execute `
  --runtime-allocate
```

This is an exact *graph/layout* parity check, not an exact model or recipe
recovery: random bytes and scales are intentionally different. It demonstrates
that the official artifact's low-bit buffers can be decoded/repacked with the
observable schema, while the FlatBuffer still cannot reveal Google's private
QAT observers, calibration data, optimizer schedule, or learned weights.

### Source-weight proof for the local 270M export

The local 270M run also retains its merged BF16 safetensors under
`training/runs/gemma270m_ir_lora/merged_hf`. The read-only
`training/scripts/audit_litertlm_weight_recipe.py` maps the candidate's
`FULLY_CONNECTED` and `EMBEDDING_LOOKUP` tensors back to those source keys and
checks the stored external INT8 bytes and per-row scales. The command used on
2026-08-07 was:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/audit_litertlm_weight_recipe.py `
  C:\path\to\gemma-3-270m-ir-int8.litertlm `
  --source-model C:\path\to\merged_hf `
  --model-type tf_lite_prefill_decode `
  --strict
```

It matched all 127 unique weight buffers (the 262,144-by-640 embedding plus
126 decoder linear matrices): 127/127 scales were byte-for-byte equal to
`max(abs(source_row))/127`, and 127/127 quantized INT8 rows were byte-for-byte
equal to round-and-clip of the BF16 source weights. This identifies the
observable export as the public `dynamic_wi8_afp32` alias of channelwise
`dynamic_wi8c_afp32`, with FLOAT32 activation edges. It is strong evidence for
the local custom export's public quantizer, not evidence that the gated official
LiteRT Community 270M Q8 package or Google's private training recipe has been
reproduced.

### Byte-verified official 270M Q8 target

The official LiteRT Community `gemma3-270m-it-q8.litertlm` file is now available
through a public mirror for artifact comparison. Its expected bytes are
304,005,120 bytes with SHA-256
`757e9119fa5bd667a2774fb470ac4afcd3190a21c677f8e69a5d6bc908abdd63`; that
hash matches the official Hugging Face LFS pointer. The audit reported LiteRT-LM
v1.3.0, one `TF_LITE_PREFILL_DECODE` section, six subgraphs, 9,872 operators,
11,702 tensors, and 737 INT8 tensors. All 732 `FULLY_CONNECTED` weight inputs
and six embedding tables are INT8, axis-0 channelwise, with zero-point arrays
of zero; the FC inputs and outputs remain FLOAT32. The runtime exposes
`decode`, `prefill_32`, `prefill_128`, `prefill_272`, `prefill_512`, and
`prefill_1024` signatures.

The topology-preserving random harness was executed in memory against that
exact section. It rewrote 127 unique inline INT8 weight buffers, verified all
127 unpack/repack byte round trips, preserved the structural and quantization
layout hashes, and allocated with the same six signatures and 1,847 runtime
tensors. The random section's quantization-value and payload hashes differed,
as they must, so this is graph/layout/runtime evidence rather than an exact
learned-model match.

Direct package comparison against the local v1.5 custom 270M export is a hard
mismatch: the local file has one graph with 215 subgraphs, 3,046 operators, and
4,611 tensors plus external constant buffers, whereas the official v1.3 file
has six subgraphs and inline constants. The local file therefore cannot be
made byte-equivalent by changing only its INT8 scale formula; the graph/export
revision must also match.

For source-weight evidence without downloading the gated 536 MB checkpoint,
use the range-based audit (it reads only the header and selected source tensors):

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/audit_litertlm_remote_source_recipe.py `
  C:\path\to\official-gemma3-270m-it-q8.litertlm `
  --source-url https://huggingface.co/<verified-public-mirror>/resolve/main/model.safetensors `
  --model-type TF_LITE_PREFILL_DECODE `
  --max-weights 10
```

On the verified public mirror, the embedding plus the first nine canonical
linear buffers all had exact `max(abs(BF16 row))/127` scales. Serialized INT8
values differed only by at most one at BF16 half-step boundaries; this is
consistent with the converter using higher-precision source values before the
public BF16 checkpoint was written, so it is not enough to claim exact bytes.
The script records both scale and value mismatches and never labels a mirror as
Google provenance. A full run against the same mirror matched all 127 unique
official weights (the embedding plus 126 decoder matrices) for scales; 31,142
of 268,042,240 serialized INT8 values differed (0.0116183%), with a maximum
error of one. No serialized quantized matrix was byte-identical because the
remaining differences occur at BF16 half-step boundaries. A local tuned/merged BF16
checkpoint was also tested and failed even the scale comparison, confirming it
is not the source checkpoint behind the official package. The observable 270M
conclusion is therefore: public channelwise dynamic W8/A FP32 scale
construction is recovered, while exact rounding inputs, source precision,
exporter revision, and private training recipe remain unresolved.

### Fresh random graph inventory parity

`training/scripts/build_fresh_random_quantized_graph.py` is the complementary
experiment to the topology-preserving harness. It reads only the official
section's observable inventory, builds a new FlatBuffer with one independent
random branch per weight, quantizes those random values using the observable
axis-0 symmetric rule, and compares the resulting weight inventory. It never
copies official operators or learned bytes.

The complete official 270M prefill/decode inventory was run on 2026-08-07:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_fresh_random_quantized_graph.py `
  C:\path\to\official-gemma3-270m-it-q8.litertlm `
  --model-type TF_LITE_PREFILL_DECODE `
  --include-embeddings
```

It generated 127 fresh branches (126 `FULLY_CONNECTED` matrices and one
`EMBEDDING_LOOKUP`), with identical official/fresh inventory digest
`2ead21cf61e8ef5cb43a2af1ea86aad16395d18498c8891fa766358da07327c6` and
`fresh_quantization_layout_match=true`. The in-memory fresh FlatBuffer was
272,714,552 bytes. `graph_structure_match=false` and
`exact_official_model_match=false` are intentional: the new graph has no
official cache wiring, signatures, control flow, metadata, or learned values.

The same experiment was run against the official Gemma 4 MTP section. All 23
weight layouts matched (13 W4 and 10 W8), with digest
`f20780248984e3fd5139c3b0fcc65b2863ca0596c4c8a7c4493b39af96cdced5`; the
fresh graph was 43,959,168 bytes. This proves that the observable MTP weight
inventory is constructible independently, not that the private drafter
architecture or distillation/QAT process has been recovered.

For the Gemma 4 prefill/decode section, the official inventory is 277 unique
weights (145 W4, 61 W2, 71 W8). Building all random constants is substantially
larger than the 270M fixture, so use `--max-weights` for bounded fresh-graph
experiments and use the topology-preserving harness for the full 277-buffer
graph/layout/encoding proof above.

### Converter-driven random inventory parity

`training/scripts/build_converter_random_inventory_parity.py` closes an
important gap in the fresh-graph experiment: it builds an independent FLOAT32
FlatBuffer and sends that graph through the public AI Edge Quantizer. It does
not patch official bytes and does not use a hand-written quantization loop.
Each official `FULLY_CONNECTED` inventory record becomes a separate random
branch, with a public channelwise symmetric W2/W4/W8 recipe and A8 activation
configuration when the official branch has INT8 edges. The generated recipe
and JSON report are retained so an agent can reproduce the exact converter
inputs later.

The full Gemma 4 MTP run matched all 23 official FC layout records (13 W4 and
10 W8), including static INT8 activation edges on the quantized branches and
the FLOAT32-edge W4 draft LM head:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_converter_random_inventory_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --model-type tf_lite_mtp_drafter `
  --output-dir C:\temp\random-inventory-mtp `
  --calibration-samples 2 `
  --threads 1
```

The full official Gemma 3 270M section matched all 126 FC records (all W8,
axis-0, FLOAT32 edges); the one embedding table is reported as skipped because
an ordinary independent FC graph cannot guarantee the official
`EMBEDDING_LOOKUP` operator. The topology-preserving harness separately
verified the complete 127-weight section, including that embedding. A bounded
eight-record Gemma 4 prefill run matched 8/8 (one W8 and seven W4); the full
277-record prefill layout remains covered by the topology-preserving harness.

These are converter/layout results only. In all three runs,
`converter_quantization_layout_match=true` while
`graph_structure_match=false` and `exact_official_model_match=false` by design:
random weights, signatures/cache wiring, LiteRT-LM metadata, learned QAT
weights, calibration data, and the private exporter/package path are not
recoverable from the public recipe or from the quantized artifact alone. This
experiment therefore validates the public observable quantizer contract; it
does not claim Google's exact private mobile recipe.

### Converter-driven official-topology parity

`training/scripts/build_converter_topology_parity.py` answers a different
question from the independent-inventory harness. It unpacks the official
TFLite section, floatifies its low-bit constant tensors with deterministic
random values, removes Q/DQ nodes that are no longer valid after floatification,
and invokes the public AI Edge Quantizer on the resulting official topology.
The source `.litertlm` is read-only. Reports contain serializer-sensitive
fingerprints and separate buffer-index-independent fingerprints, so a
re-serialized FlatBuffer is not incorrectly called byte-identical.

On the official Gemma 3 270M section, the converter preserved the complete
observable execution graph: six subgraphs, 9,872 operators, 11,702 tensors,
732 `FULLY_CONNECTED` operators, six `EMBEDDING_LOOKUP` operators, and all 737
INT8 quantized tensors. The normal structural/layout hashes differ because
buffer tables are reserialized, while both buffer-agnostic execution-topology
and quantization-layout hashes match. This is public converter/topology parity,
not learned-weight or package parity.

On Gemma 4's official `tf_lite_mtp_drafter`, the converter retained 30
subgraphs and 23 FC records but changed 365 to 357 operators, 575 to 615
tensors, 30 to 22 `DEQUANTIZE` operators, and 79 to 63 quantized tensors.
Both buffer-agnostic matches are false. The mixed W4/W8 activation graph and
MTP lowering therefore remain private/custom; public AI Edge Quantizer cannot
reproduce the exact drafter network from random weights.

### Converter-driven random constants injected into official topology

`training/scripts/build_converter_random_topology_injection_parity.py` combines
the two experiments above. It builds one independent random FLOAT32 branch for
every observable official FC/embedding weight, quantizes those branches with
the public AI Edge Quantizer, and copies only the resulting packed bytes and
per-axis scales into an in-memory copy of the official section. The source
`.litertlm` is never modified. The operator graph, tensor table, signatures,
metadata, and cache wiring remain the released graph, so this is the cleanest
way to test whether public converter constants can inhabit the official
network without conflating that result with recovery of learned QAT values.

The builder emits a real `EMBEDDING_LOOKUP` branch (not a disguised FC), so the
complete Gemma 3 270M inventory is covered. The latest verified runs used two
calibration samples and seed 42:

| section | converter inventory | converter layout | injected topology/layout | allocation |
| --- | ---: | --- | --- | --- |
| Gemma 4 E2B `tf_lite_embedder` | 1 embedding (W2) | 1/1 | structure and layout true | true, built-in resolver without default delegates |
| Gemma 4 E2B `tf_lite_per_layer_embedder` | 35 embeddings (W4) | 35/35 | structure and layout true | true, six batches, without default delegates |
| Gemma 4 E2B `tf_lite_audio_encoder_hw` | 120 FC (W2/W4) | 120/120 | structure and layout true | true, built-in resolver without default delegates |
| Gemma 4 E2B `tf_lite_vision_encoder` | 112 unique FC buffers (W8) | 112/112 | structure and layout true | true, built-in resolver without default delegates |
| Gemma 4 E2B `tf_lite_prefill_decode` | 277 FC (W2/W4/W8) | 277/277 | structure and layout true | true, built-in resolver without default delegates |
| Gemma 4 `tf_lite_mtp_drafter` | 23 (13 W4, 10 W8) | 23/23 | structure and layout true | true, built-in resolver without default delegates |
| Gemma 3 270M `TF_LITE_PREFILL_DECODE` | 127 (126 FC + 1 embedding) | 127/127 | structure and layout true | true, built-in resolver without default delegates |

The E2B main run retained 976 subgraphs, 14,259 operators, 22,990 tensors,
22,995 buffers, and 3,454 quantized tensors; its 8.5 GiB independent float32
inventory was quantized in 35 batches of eight to stay below the FlatBuffers
2 GiB offset limit. The MTP run retained its 30 subgraphs, 365 operators, 575
tensors, 578 buffers, and 79 quantized tensors; the 270M run retained six
subgraphs, 9,872 operators, 11,702 tensors, 11,705 buffers, and 737 quantized
tensors. The random injected values intentionally differ from the official
values, so all three runs report `quantization_values_match=false` and
`exact_official_model_match=false`. They also report
`converter_to_injected_quantization_values_match=true`: the converter's
packed bytes and scales survived injection into the official section exactly.
This proves public quantizer/layout compatibility and runtime allocation, not Google's private QAT observers,
calibration corpus, learned constants, exporter revision, or byte-identical
package.

The report also records the complete `.litertlm` package boundary. For the
supplied Gemma 4 package the selected MTP section is exactly 44,325,712 bytes
at offset 2,543,812,608 in the 2,588,147,712-byte package; for the 270M mirror
the selected section is exactly 299,261,424 bytes at offset 4,734,976 in the
304,005,120-byte package. Prefix and suffix hashes are retained, proving that
the random fixture changes only the selected TFLite section and copies all
tokenizer, metadata, and other model sections unchanged. This is stronger than
comparing an isolated TFLite graph, but it still cannot make random constants
equal to Google's learned values.

Reproduce without retaining the roughly 300 MB temporary 270M fixtures:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/build_converter_random_topology_injection_parity.py `
  C:\path\to\gemma3-270m-it-q8.litertlm `
  --model-type TF_LITE_PREFILL_DECODE `
  --output-dir C:\temp\converter-injection-270m `
  --calibration-samples 2 `
  --threads 1 `
  --in-memory `
  --runtime-allocate `
  --runtime-without-default-delegates
```

For the Gemma 4 E2B main graph use `--model-type tf_lite_prefill_decode`; the
large inventory is batched automatically. Use `--model-type tf_lite_mtp_drafter`
for the draft graph. The same harness also covers the E2B
`tf_lite_embedder`, `tf_lite_per_layer_embedder`, `tf_lite_audio_encoder_hw`,
and `tf_lite_vision_encoder` sections; adapter/end-marker sections contain no
quantized FC/embedding inventory. The report deliberately
keeps `converter_random_quantization_layout_match`, official-topology hashes,
runtime allocation, `converter_to_injected_quantization_values_match`, and
`exact_official_model_match` as separate fields.

Each injection report also has a `random_quantized_network` object. Its
`same=true` value means that the complete FC/embedding inventory, operator
structure, quantization layout, and buffer-index-independent execution graph
all match; it deliberately excludes learned model identity. The separate
`converter_to_injected_quantization_values_match=true` field proves the
converter's random bytes/scales are identical after injection. The
verified Gemma 4 E2B main, Gemma 4 MTP, and Gemma 3 270M runs all report this
network-level match as true, while `quantization_values_match=false` remains
expected evidence that the random constants are not Google's learned constants.
When
`--runtime-allocate` is used, `random_quantized_network.runtime_allocation_match`
records the separate LiteRT allocation smoke check.

The same command now accepts `--rebuild-flatbuffer`. This path does not mutate
the official bytes: it deserializes the selected section through the public
`ai_edge_litert.schema_py_generated.ModelT`, materializes external buffers, puts
the independently AI Edge-quantized random constants into the object graph, and
packs a new `TFL3` FlatBuffer. The rebuilt report adds
`rebuilt_quantized_network.same`, which requires structural hash, quantization
layout, buffer-index-independent execution topology/layout, converter-constant
digest, and (when requested) LiteRT allocation parity. It is stronger than
in-place byte injection because stale official buffer offsets cannot explain a
passing result.

The dated rebuilt runs passed all of those checks for the E2B token embedder
(1/1), per-layer embedder (35/35), audio encoder (120/120), vision encoder
(112/112), main decoder (277/277), E2B MTP (23/23), and Gemma 3 270M
(127/127). The audio and vision counts are unique weight buffers; an audit may
show more operator occurrences when a buffer is reused. They still correctly report
`quantization_values_match=false` and `exact_official_model_match=false`: a
random graph can prove network/serialization compatibility, but cannot become
Google's learned model.

Example:

```powershell
python training/scripts/build_converter_random_topology_injection_parity.py `
  C:\path\to\official.litertlm `
  --model-type tf_lite_prefill_decode `
  --output-dir C:\temp\random-rebuilt `
  --calibration-samples 2 `
  --threads 1 `
  --in-memory `
  --rebuild-flatbuffer `
  --runtime-allocate `
  --runtime-without-default-delegates
```

### Current public exporter boundary for packed QAT checkpoints

The current public LiteRT-Torch main snapshot checked during this audit
(`386e5fba46aa25ceef8b8595e7bd1432bc96b494`, 2026-08-05) still marks
quantized safetensors as a TODO in `generative/export_hf/export_main.py` and
raises `NotImplementedError("Quantized checkpoint is not supported yet.")`
when a Hugging Face config contains `quantization_config` in
`generative/export_hf/core/export_lib.py`. The public exporter can therefore
quantize a supported floating-point graph, but it is not a public loader for
Google's packed QAT safetensors or an exact mobile-QAT exporter.

This agrees with the LiteRT-Torch maintainer's public response in
[issue #998](https://github.com/google-ai-edge/litert-torch/issues/998): the
pre-built Gemma 4 LiteRT-LM files are exported from variants of
`google/gemma-4-E2B-it-qat-mobile-transformers`, require quantized-safetensor
conversion that was not supported there, and the audio/MTP drafter path was
still being open-sourced. The issue is provenance for this boundary; it does
not disclose Google's internal QAT observers, calibration corpus, optimizer,
or exporter revision.

For Gemma 3 270M, the public BF16 safetensor mirror reproduces every compared
per-row INT8 scale, but not every stored INT8 code: the full audit observed
31,142 differing codes out of 268,042,240 (maximum absolute code error one),
concentrated at values near half-integer quantization boundaries. This is
consistent with higher-precision source/export intermediates that are not
recoverable from a BF16 mirror. It prevents a byte-exact learned-weight claim
without the original higher-precision or packed quantized source.

The practical exactness boundary is therefore:

| target | what is reproducible publicly | what remains private/unproven |
| --- | --- | --- |
| Gemma 4 E2B mobile main | comparable packed checkpoint constants/scales and released graph layout | trainer, observers, calibration, exporter, and one unresolved public W8 projection source |
| Gemma 4 E2B MTP drafter | released topology/layout and a random converter-injection fixture | exact drafter lowering, packed assistant constants, and private MTP export path |
| Gemma 3 270M INT8 | public converter topology/layout; all scales from the BF16 mirror | exact boundary-sensitive INT8 codes and original export intermediates |

Use `random_quantized_network.same=true` as the network-compatibility gate for
future experiments. `exact_official_model_match` must remain false until every
learned constant, graph byte, and package section is independently matched.

For a random-network runtime fixture rather than an independent converter
reconstruction, `build_random_official_topology_parity.py` patches the official
MTP section's 23 serialized W4/W8 buffers and leaves the graph untouched. The
verified run randomized 13 W4 and 10 W8 buffers, passed all 23 low-bit
round-trip checks, and returned `graph_structure_match=true` and
`quantization_layout_match=true`. This is the correct way to test a random
network against Google's released MTP topology; it must not be described as a
recovered private exporter or QAT recipe.

### Official mobile checkpoint serialization parity

The public `gemma-4-E2B-it-qat-mobile-transformers` safetensors checkpoint is
the strongest available source for the released mobile constants. Its public
config describes targeted W2/W4/W8 modules and static A8 activations, but it
excludes `per_layer_model_projection` while the released `.litertlm` stores
that matrix as W8. Training therefore follows the artifact-reconciled schema
described above. Neither source discloses the QAT trainer, calibration corpus,
or optimizer schedule.

`training/scripts/audit_gemma4_mobile_checkpoint_parity.py` compares unique
FC/embedding buffers and their scales without modifying either artifact. The
checkpoint's packed W2/W4 bytes use unsigned offset codes; the LiteRT section
uses the corresponding signed codes. The verified serialization rule is:

| source width | checkpoint to LiteRT conversion |
| --- | --- |
| W2 | subtract 2 from each 2-bit code modulo 4 |
| W4 | subtract 8 from each 4-bit code modulo 16 |
| W8 | copy the int8 byte directly |

Against the supplied Gemma 4 artifact, the local public checkpoint matched
1/1 token-embedding buffers, 35/35 per-layer-embedding tables, and 276/276
comparable prefill/decode buffers, including every compared scale. The prefill
section has 277 unique low-bit buffers; the one unresolved W8
`per_layer_model_projection` source is BF16 and has no `weight_scale` in the
public checkpoint, so the audit reports it explicitly instead of fabricating a
quantized value. This establishes exact parity for the comparable released
constants, but does not prove the private QAT training/export recipe.

Reproduce the audit with:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mobile_checkpoint_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --source-safetensors C:\path\to\gemma4-mobile\model.safetensors `
  --include-embedders `
  --strict `
  --output C:\temp\gemma4-mobile-checkpoint-parity.json
```

The report sets `training_executed=false`,
`private_qat_recipe_recovered=false`, and
`all_comparable_weights_exact=true` when these byte/scale comparisons pass.
It is safe to use as a regression check when changing loaders, low-bit packing,
or export scripts.

### Public assistant checkpoint versus the released MTP constants

The public `gemma-4-E2B-it-assistant` and
`gemma-4-E2B-it-qat-q4_0-unquantized-assistant` safetensors both expose the
same four-layer assistant architecture (plus metadata/masked-embedding
auxiliary tensors). Their 23 relevant tensors
have exactly the shapes expected by the supplied 23-record MTP section. That
shape agreement is not constant agreement:
`training/scripts/audit_gemma4_mtp_assistant_parity.py` maps the official
traversal explicitly and applies the public symmetric per-output-channel
candidate (`max(abs(row))/(2^(bits-1)-1)`, signed W2/W4/W8 packing). On the
local supplied package, both public assistant checkpoints produce 0/23 exact
packed buffers and 0/23 exact scale vectors under that candidate.

This is a useful negative result. It does not say that the public assistant is
unusable for Transformers speculative decoding; it says that its BF16 source,
the public min/max candidate, and the supplied private/custom MTP exporter do
not form an exact explanation of the released LiteRT constants. The missing
piece may be a different source revision, learned QAT scales/observers, a
private assistant export transform, or a combination. Do not infer Google's
exact MTP training recipe from the public assistant architecture alone.

The official [Gemma 4 technical report's QAT section](https://arxiv.org/html/2607.02770#S2.5)
and [MTP section](https://arxiv.org/html/2607.02770#S2.6) add the public
architectural facts that are not visible in those constants:
the E2B drafter is a four-layer, dimension-256 autoregressive Transformer with
three local-attention layers and one global layer; it consumes the main model's
last-layer activations and KV cache, and E2B/E4B use clustered top-k (4096
candidates) rather than a full-vocabulary projection. Matching tensor shapes
alone therefore cannot reconstruct the LiteRT lowering or the drafter loss.

Reproduce the audit with:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mtp_assistant_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  C:\path\to\assistant\model.safetensors `
  --output C:\temp\gemma4-mtp-assistant-parity.json
```

### Unresolved E2B W8 projection

The one public mobile-transformers tensor that is not packed is
`model.language_model.per_layer_model_projection.weight`. It is BF16 and the
checkpoint has no corresponding `weight_scale`. The official LiteRT section
stores a W8 matrix of shape `[8960, 1536]` with 8,960 F32 scales. The dedicated
read-only audit tests the usual public candidates (axis-0 `max(abs(row))/127`
with F32/F64 reduction and three signed tie rules) without downloading the
whole checkpoint when given a URL:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mobile_projection_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --source https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main/model.safetensors `
  --output C:\temp\gemma4-mobile-projection-parity.json
```

The dated run read only the header and projection range (`source_dtype=BF16`,
no public source scale). Every tested candidate missed all 8,960 scales; the
maximum scale error was about `1.20e-7`, and every candidate differed in 5,646
of 13,762,560 W8 codes (maximum code error one). This is a falsifiable
source-precision/learned-scale boundary, not evidence for a hidden alternative
rounding rule. Do not fill this tensor with a guessed max-abs scale when
claiming exact model parity.

The same audit now performs a BF16 feasibility check. On the 2026-08-07
snapshot, all 8,960 official `scale * 127` values fell inside the possible
per-row max-abs intervals induced by BF16 rounding, and all 13,762,560 official
codes had a non-empty BF16 pre-image under the symmetric W8 round-to-nearest
interval. This is a necessary-condition result: it supports the public
max-abs/127 rule being applied before BF16 serialization, but it cannot recover
the hidden higher-precision tensor, learned observer state, or exporter. The
report records this as `bfloat16_interval_report` and deliberately keeps
`exact_public_projection_candidate=false`.

The official `google/gemma-4-E2B-it-qat-mobile-ct` checkpoint is not a missing
solution: its config exposes the same public module assignments and its packed
W2/W4 values can be byte-compared, but its `weight_scale` tensors are BF16,
whereas the released LiteRT section stores F32 scales. It also leaves the
per-layer model projection unquantized. It is useful for schema/packing
cross-checks, not for byte-identical LiteRT-LM reconstruction.

### Public exporter boundary found during reverse engineering

The current public LiteRT-Torch exporter contains Gemma 4 text export,
external token/per-layer embedders, and vision encoder/adapter export. Its
`assistant_model`/`mtp_verifier_step` options add a verifier signature to the
target decode graph when requested; they do not load or train an assistant and
do not package a `tf_lite_mtp_drafter` section. The public LiteRT-LM builder's
additional-model registry recognizes the Gemma 4 per-layer embedder only. The
current public task path also does not emit the official Gemma 4 audio sections
seen in the supplied package. Therefore the supplied multimodal mobile
artifact (including audio and MTP) was not reproducible from the public
exporter alone; it requires Google's private exporter/package path or the
already-released artifact.

Do not turn `build_random_official_topology_parity.py` into a production
exporter. Use it as a regression/evidence harness while looking for the exact
unquantized checkpoint, private exporter revision, and calibration contract.

The public support status is explicit in LiteRT-Torch issue
[#1044](https://github.com/google-ai-edge/litert-torch/issues/1044): the
maintainer states that QAT checkpoint conversion is not currently supported;
the follow-up [#1050](https://github.com/google-ai-edge/litert-torch/pull/1050)
adds an early failure for quantized checkpoints. This rules out treating
`export_hf` as Google's mobile-QAT importer or exact exporter.

The newer open [PR #1097](https://github.com/google-ai-edge/litert-torch/pull/1097)
only adds the missing `gemma4` LiteRT-LM metadata dispatch. It does not add a
packed `quant_method: gemma` safetensor importer or reveal the private mobile
QAT/export path.

### Loader and PEFT compatibility

Google's Gemma 4 model card uses `AutoModelForMultimodalLM`, while the official
text-only MTP tutorial loads target and assistant with `AutoModelForCausalLM`.
`ir_training.models.hf_loading` therefore makes the AutoModel class selectable.
The recommended text-to-IR and MTP reference path uses `auto_causal_lm`; a future
multimodal training profile can select `auto_multimodal_lm`.

The Gemma 4 overlay is `training/requirements-gemma4-qat.txt`. It follows the
current official guide with Transformers 5.10.1+ and PEFT 0.19.0+. Pin exact
versions and checkpoint revisions for real experiments; minimum ranges alone do
not create reproducible deployment evidence.

### Static preflight

```powershell
python training/scripts/validate_qat_mtp_workflow.py
```

This checks checkpoint pairing, QLoRA/QAT confusion, assistant-training claims,
learning-rate/scope warnings, and final-runtime gates. It imports no model and
does not contact Hugging Face.

For the true fake-quantization path, use `validate_qat_training.py`. It checks
the W8A8 profiles, rejects NF4/QLoRA settings, reports target-specific profile
warnings, and explicitly reports that it executes no training and loads no
models.

### Training command for a later authorized run

Do not run this during code-only setup:

```powershell
python -m pip install -r training/requirements-gemma4-qat.txt
python training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_qat_lora.yaml
```

For multi-GPU runs, use the repository's torchrun/Slurm mechanisms and override
their config path with `A2UI_TRAIN_CONFIG`. Dataset preparation remains under
`dataset/`; the training package only consumes prepared response-to-IR pairs.

### Merge

Plan only:

```powershell
python training/scripts/merge_qat_lora.py
```

Execute later:

```powershell
python training/scripts/merge_qat_lora.py --execute
```

The merge refuses a non-empty destination unless `--force` is supplied.

The merge defaults to the best golden checkpoint. It saves the merged HF model
and `qat_mtp_merge_metadata.json`. That metadata intentionally records:

- QAT-derived base provenance;
- no continued QAT;
- no assistant modification;
- no packed INT4 output;
- post-merge quantization still required.

### Transformers target-only versus MTP reference benchmark

Config: `training/configs/eval/gemma4_e2b_qat_mtp.yaml`

Plan only:

```powershell
python training/scripts/benchmark_qat_mtp.py
```

Execute later:

```powershell
python training/scripts/benchmark_qat_mtp.py --execute
```

Run the untouched official reference into a separate output directory with:

```powershell
python training/scripts/benchmark_qat_mtp.py --target-model google/gemma-4-E2B-it-qat-q4_0-unquantized --output-dir outputs/eval/gemma4_e2b_qat_mtp_official --execute
```

The script uses greedy decoding, writes separate target-only and target+MTP
predictions, evaluates both with existing flat-spec metrics, records token/text
equivalence and full-generation latency, and produces `benchmark_summary.json`.

Stable Transformers `generate()` does not expose accepted-draft counts through
this wrapper. The script records acceptance rate as unavailable rather than
inventing it. Use instrumentation from the final runtime for accepted draft
length and rejection counts. Full-generation latency is not TTFT; measure TTFT
in the final streaming runtime.

### GGUF Q4_0 conversion

Config: `training/configs/export/gemma4_e2b_qat_q4_0.yaml`

Plan only:

```powershell
python training/scripts/convert_qat_q4_0.py --llama-cpp-dir C:\path\to\llama.cpp
```

Execute later, after pinning a llama.cpp revision known to support both Gemma 4
target and assistant conversion:

```powershell
python training/scripts/convert_qat_q4_0.py --llama-cpp-dir C:\path\to\llama.cpp --execute
```

The converter creates F16 GGUF intermediates and then Q4_0 target/assistant
artifacts, logs each external step, hashes outputs, and blocks accidental final-
artifact overwrite unless `--force` is explicitly supplied.

Assistant conversion remains runtime/tool-version sensitive. A successful file
conversion does not prove that the selected llama.cpp build can execute Gemma 4's
activation/KV-sharing assistant protocol correctly. The conversion manifest
keeps promotion blocked until the pair is exercised in the final runtime.

This GGUF workflow does not create Google's mobile wNa8o8 model. For Android,
prefer the untouched official mobile package as the performance control. Treat a
custom tuned `.litertlm` export and MTP pairing as a separate compatibility
experiment until Google publishes or the pinned exporter demonstrates the exact
custom quantization and assistant path.

## Accuracy strategy for response-to-flat-spec

MTP will not improve IR quality. Accuracy work should focus on:

- schema-valid, renderer-valid, source-grounded response-to-IR pairs;
- completion-only labels so prompt tokens do not dominate loss;
- no reasoning traces or markdown fences in the target;
- compact IR: avoid duplicate prose, expanded table cell trees, and unnecessary
  `sourceText`;
- deterministic golden selection, never training on golden50;
- error mining by domain and renderer failure, followed by regeneration through
  Stage 3 rather than hand-editing generated IR;
- optional later preference/RL work only with stable deterministic rewards and a
  fresh MTP drift evaluation.

Sequence and generation limits should be derived from token-length percentiles.
Do not retain an 8192-token output cap merely because the architecture permits it:
every unnecessary output token directly increases end-to-end latency. The
current recommended profile allows 4096 input plus 4096 output for golden
evaluation and uses a 4096 total SFT sequence; revisit these after inspecting
actual truncation statistics.

## Required experiment matrix

Keep each variable isolated:

1. Untouched official QAT target, target-only.
2. Untouched official QAT target plus exact official assistant.
3. Tuned merged BF16 target, target-only.
4. Tuned merged BF16 target plus official assistant.
5. Tuned packed INT4 target, target-only, in the final runtime.
6. Tuned packed INT4 target plus matching-precision assistant, in the final
   runtime.

Record at minimum:

- strict JSON parse and schema-valid rates;
- Android renderer success;
- golden overall score, content/source coverage, action correctness, duplicate
  rate, and output length;
- BF16-to-INT4 accuracy delta;
- TTFT, full-response P50/P95, and decode tokens/second;
- peak RAM/VRAM, thermal throttling, and energy where available;
- runtime-provided mean accepted draft length, acceptance/rejection counts, and
  assistant schedule;
- exact target/assistant revisions, converter revision, runtime revision,
  tokenizer/processor revision, device, backend, and prompt set.

Promotion requires all hard contract checks to pass, a measured task-quality
improvement over the untouched INT4 baseline, acceptable BF16-to-INT4 loss, and
a positive end-to-end MTP benefit over the same tuned INT4 target without MTP.
Do not promote on tokens/second alone if TTFT, full-response latency, memory, or
thermals regress.

## Common failure modes

### Calling the run QAT

Symptom: configuration or documentation says `method: qat` even though the code
only trains LoRA parameters without fake quantization.

Fix: call it QAT-derived LoRA and leave `continued_qat: false`. Implementing true
continued QAT is a separate research change requiring exact deployment numerics.

### Loading with BitsAndBytes and calling the result INT4 QAT

Symptom: `load_in_4bit: true` with NF4 is presented as matching Q4_0.

Fix: keep it false in the recommended profile. If VRAM forces QLoRA, name the
fallback honestly and compare the final Q4_0 artifact against BF16.

### Blanket `all-linear` LoRA

Symptom: adapters attach to modality or helper projections beyond the language
decoder, or PEFT fails on Gemma4 wrapper modules.

Fix: use the pinned current PEFT Gemma 4 defaults first. If a release changes
module discovery, enumerate actual trainable module names and add a narrow,
tested Gemma adapter mapping; do not guess from older Gemma architectures.

### Tokenizer/model or assistant vocabulary mismatch

Symptom: resize warnings, label IDs beyond logits, generation corruption, or
assistant incompatibility.

Fix: use the same processor/tokenizer revision throughout; avoid new tokens;
freeze embeddings/LM head; retain existing vocabulary-alignment preflights.

### MTP produces no speedup after tuning

Symptom: target-only and MTP outputs remain valid but MTP latency is equal or
worse.

Fix: inspect accepted draft statistics in the final runtime; compare lower LoRA
learning rate/rank/scope; use heuristic draft scheduling; otherwise ship target-
only INT4. Do not retrain the assistant without a separately reviewed research
design.

### Transformers reference results are mistaken for final device proof

Symptom: BF16 `assistant_model=` timing is reported as Android or packed-Q4_0
performance.

Fix: label reference results correctly and complete matrix entries 5 and 6 on
the actual device/backend.

### LiteRT-LM export is mistaken for Google mobile QAT recreation

Symptom: a custom `.litertlm` file is called official wNa8o8 merely because the
runtime loads it.

Fix: inspect the exporter recipe and artifact metadata. Unless the exact schema
and assistant pairing are proven, report it as custom export with unknown QAT
preservation.

## Exact LiteRT-LM artifact parity and random-weight reverse engineering

The repository now includes a read-only package/graph auditor:

```powershell
python training/scripts/audit_litertlm_package.py `
  C:\path\to\model.litertlm `
  --output C:\path\to\model.litertlm.audit.json
```

Install the optional pure-Python TFLite schema bindings in the conversion
environment when graph fingerprints are required:

```powershell
python -m pip install tflite
$env:PYTHONPATH = "<tflite-install-location>"
```

The auditor does not treat `.litertlm` as a ZIP. LiteRT-LM source defines an
`LITERTLM` magic prefix, a semantic version, a header-end offset, a FlatBuffer
section table, and 16 KiB-aligned sections. It reports section types, ranges,
metadata, TFLite operator/tensor summaries, tensor quantization layout, and
stable structural digests. `--include-hashes` additionally hashes every section
and therefore reads the complete artifact.

For an explicit random-weight conversion experiment, use the plan-only command
first:

```powershell
python training/scripts/build_random_litertlm_parity.py `
  --model-source google/gemma-3-270m `
  --official-artifact C:\path\to\official.litertlm `
  --output-dir C:\temp\random-litertlm-parity `
  --quantization-recipe dynamic_wi8_afp32
```

Only add `--execute` after reviewing the plan. The script creates a random
model from the source config, invokes the pinned `litert-torch` exporter, and
compares the result at independent evidence levels. It never trains, updates
an adapter, or deletes an existing output directory. For multimodal Gemma 4,
the random model must be instantiated with the same multimodal class and
exporter inputs as the official package; a generic `AutoModelForCausalLM` is
not sufficient to recreate audio/vision/MTP graphs.

Interpret parity results as follows:

| Evidence | What it establishes | What it cannot establish |
| --- | --- | --- |
| Section order/types and metadata | Same package envelope and runtime roles | Same graph or weights |
| Graph structural digest | Same operator topology, tensor shapes, and connectivity | Same learned values or hidden exporter passes |
| Quantization-layout digest | Same tensor dtypes, scale/zero-point counts, and quantized axes | Same calibration scales or rounding |
| Quantization-value digest | Same serialized scales/zero points | That the training/QAT recipe was the same |
| Weight-section or embedded TFLite-model hashes | Same serialized payload bytes | Nothing about a random candidate, whose weights must differ |
| Runtime numerical parity | Same outputs for a controlled input on the same runtime | General accuracy or private training provenance |

Therefore a random-weight graph can validate topology and observable
quantization layout, but it cannot make the final artifact exactly equal to
Google's model. Exact artifact reproduction additionally requires the same
learned weights, tokenizer/metadata, exporter and runtime revisions, device
variant, and (for Gemma 4) the matching MTP drafter graph.

For a small, executable public-recipe converter check (without Transformers or
training), use:

```powershell
# Install TensorFlow, ai-edge-quantizer/ai-edge-litert, and the optional
# `tflite` schema bindings in the conversion environment first.
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_random_tflite_recipe_parity.py `
  --output-dir C:\temp\random-gemma4-public-recipe `
  --recipe gemma4_mixed48 `
  --execute `
  --official-artifact C:\path\to\gemma-4-E2B-it.litertlm
```

The script builds a deterministic three-layer random graph, verifies that the
default fully-connected weights become channelwise INT4 while the `per_layer`
scope becomes INT8, and records the official artifact's independent W2/W4/W8
observation. It never trains, downloads checkpoints, overwrites planned files,
or claims learned-weight equivalence.

The static mobile-contract harness above is intentionally separate from this
public `gemma4_mixed48` check: it demonstrates that the installed quantizer can
serialize an observable W2/W4/W8-A8 *synthetic* layout when policy checks are
overridden, but it does not recover Google's hidden observers, calibration
data, graph lowering, or QAT schedule.

### What the available artifacts show

The local `gemma-4-E2B-it.litertlm` artifact was audited on 2026-08-07. Its
header is LiteRT-LM v1.5.0 and it contains ten TFLite model sections after the
metadata/tokenizer sections: text embedding, per-layer embedding, audio
encoder/adapter/end marker, vision encoder/adapter/end marker, prefill/decode,
and `tf_lite_mtp_drafter`. The prefill/decode graph has 976 subgraphs, 14,259
operators, 22,990 tensors, and INT2/INT4/INT8 tensors with INT8 activation
edges; the MTP graph has 30 subgraphs, 365 operators, and INT4/INT8 tensors.
This is direct artifact evidence that MTP is a separate compiled graph in the
package.

The local `gemma-4-e2b-ir-trained-int4.litertlm` candidate contains only three
TFLite model sections (prefill/decode, embedding, and per-layer embedding) and
no audio, vision, or MTP-drafter section. Its package and graph comparison is
therefore a hard mismatch, independent of model quality or weight values. It
must be described as a custom text-oriented export, not an exact copy of the
official multimodal package.

Gemma 3 270M has a separate distinction: Google's public QAT checkpoint is an
unquantized Q4_0-derived model intended to be quantized with Q4_0, while the
LiteRT Community package family exposes a `gemma3-270m-it-q8.litertlm` artifact
(and a separate Q4_0 web `.task`). A Q4_0 checkpoint conversion cannot be called
an exact match for the Q8 `.litertlm` package. The local 270M candidate is a
285.5 MB v1.5.0 package with one `tf_lite_prefill_decode` graph; the official
Q8 v1.3 artifact is now available for direct comparison, but its gated source
weights and private exporter provenance remain unavailable.

The local candidate's direct audit reports 248 INT8 `FULLY_CONNECTED` weight
matrices, per-axis zero-point-zero scales, and FLOAT32 fully-connected input and
output edges. It therefore fails the Gemma 4 mobile W2/W4/W8-A8 contract; its
successful randomization check validates only that candidate's external INT8
packing and graph layout.

With `--runtime-allocate`, both the original extracted TFLite section and the
randomized output also allocate successfully in LiteRT and expose identical
`prefill_131`/`decode` signatures (1,137 tensors each). This is a runtime
validity check, not numerical or learned-weight parity.

Use the exact artifact variant as the comparison target. Google publishes
separate web, Tensor G5, Qualcomm, and Intel `.litertlm` variants; their graph
and backend metadata are not interchangeable.

## End-to-end mobile pipeline with official or trained MTP weights

The checked-in pipeline for the requested workflow is:

```text
QAT LoRA SFT
  -> golden-set best adapter
  -> BF16 merged best checkpoint
  -> public AI Edge quantization of all 277 target matrices
  -> released target graph with trained constants
  -> either (A) byte-exact official MTP section
             or (B) 23 trained drafter matrices in the released MTP graph
  -> desktop package gate
  -> Android LiteRT-LM GPU + mtp=true gate
```

Use `training/scripts/run_gemma4_e2b_mobile_mtp.py` with
`training/configs/pipelines/gemma4_e2b_mobile_mtp.yaml`. The command is
plan-only unless a stage flag is supplied. The training callback already saves
`golden_eval.best_checkpoint_dir`; the pipeline refuses to silently fall back
to `final_adapter`, because deployment selection must use the measured best
checkpoint.

The merge stage is still floating-point and records
`packed_int4_output=false`. A public `litert-torch export_hf` result is only a
standalone candidate. It cannot be called Google's mobile W2/W4/W8-A8 output
without proving the private schema, observers, calibration, and exporter.

`pipeline.mtp.weight_source` selects the assistant branch. `official` is the
recommended first candidate and retains `tf_lite_mtp_drafter` byte-for-byte.
`trained` uses `training/scripts/train_gemma4_mtp_drafter.py` and
`training/configs/models/gemma4_e2b_mtp_drafter_qat.yaml`, then
`training/scripts/build_gemma4_mtp_drafter_official_topology.py`. The trainer
freezes the target and every assistant parameter except the exact 23 matrices
mapped to the mobile graph. It reconstructs the public target-conditioned
four-token rollout: target last hidden state, target token embedding, shared
target KV, constant position ID, autoregressive assistant state, and teacher
forced completion cross-entropy; this is not Google's unknown distillation
loss. Its QAT assignment is 13 W4 plus 10 W8 matrices with A8 fake
quantization. This is a public reconstruction; Google's data mixture, loss
weights, optimizer, and observer schedule remain unknown.

The released LiteRT-LM runtime source confirms the inference-side rollout used
by this reconstruction. It sets the drafter `input_pos` to `position - 1` once,
duplicates the target KV inputs once, then runs four steps with the current
token embedding concatenated first with the target activation and thereafter
with the drafter's previous `projected_activations`. The drafter outputs only
`logits` and `projected_activations`; it does not advance a separate KV cache
inside those four steps. The target verifier then evaluates the input token plus
all four proposals in one five-position call. This makes the checked-in
constant-position/fixed-shared-KV loop runtime-faithful even though its training
loss and data are still a public approximation.

The trained-drafter exporter accepts only a provenance-hashed checkpoint from
that narrow workflow. It quantizes the 23 float matrices through the public AI
Edge quantizer, patches their packed bytes and per-row scales into the released
MTP section, proves graph and quantization-layout hashes unchanged, and proves
the fine-tuned target plus all bytes outside the MTP section remain exact.

`training/scripts/compose_litertlm_with_mtp.py` performs the safe packaging
operation once a compatible exporter has produced either a candidate package
or a raw `TFL3` target section. It requires:

1. an official base `.litertlm` containing `tf_lite_mtp_drafter`;
2. exactly one target source and the same `tf_lite_prefill_decode` section size;
3. matching target TFLite operator/tensor topology, quantization layout, and
   logical buffer-storage layout;
4. a successful post-write inspection proving that the MTP section hash and
   every non-target section hash are unchanged.

The legacy composer does not train, convert, or replace the assistant. It is
valid only for the default compiled section from the selected official package.
If section sizes differ, the tool fails and asks for a header-aware
composer/exporter rather than shifting absolute section offsets heuristically.

Run the structural gate with:

```powershell
python training/scripts/validate_litertlm_mtp_gpu.py `
  C:\path\to\gemma4_e2b_mobile_mtp.litertlm
```

This proves package structure and MTP presence only. It deliberately reports
`device_validation=not_run`; a real Android LiteRT-LM run must show GPU
delegate execution, `mtp=true`, accepted draft tokens, output parity, and a
latency/memory comparison against the same target with MTP disabled. A desktop
package inspection can never prove those runtime facts.

## Gemma 3 270M equivalent pipeline

Gemma 3 270M uses the parallel
`training/scripts/run_gemma270m_qat_litertlm.py` pipeline and
`training/configs/pipelines/gemma3_270m_qat_litertlm.yaml` configuration. Its
stages are:

```text
W8 / activation-FP32 QAT SFT
  -> golden-set best adapter
  -> BF16 merged checkpoint
  -> public dynamic_wi8_afp32 LiteRT-LM export
  -> package/graph gate
  -> Android GPU validation
```

The Gemma 3 270M training config now includes an explicit golden evaluation
block and writes `runs/gemma3_270m_ir_qat_sft/best_golden_checkpoint`. The
pipeline requires the package-matched `google/gemma-3-270m-it` base, requires
W8 weight QAT with activation fake quantization disabled, and records
`packed_int8_output=false` until the converter runs.
The public export config is `training/configs/export/litertlm_gemma270m_int8.yaml`
with `dynamic_wi8_afp32`; this is an INT8 LiteRT-LM artifact, not INT4/Q4_0.

Unlike Gemma 4 E2B, Gemma 3 270M has no released Gemma 4 MTP drafter section.
The pipeline therefore sets `mtp=false` and rejects a configuration that tries
to attach an assistant. Use `training/scripts/validate_gemma270m_litertlm.py`
for the package gate. It checks for `tf_lite_prefill_decode`, section alignment,
and package ordering; `--inspect-graphs` adds the slower TFLite inspection.
The final Android GPU run remains necessary for delegate/runtime evidence.

## Connected Android GPU parity evidence (2026-08-08)

`android/app/src/androidTest/java/com/samsung/genuicraft/LiteRtGpuInitParityProbeTest.kt`
is an initialization-first probe for arbitrary `.litertlm` paths. It requests
the GPU backend directly, optionally enables MTP, can cap generation with
LiteRT-LM 0.15's `ConversationConfig.maxOutputToken`, enables native benchmark
counters only for bounded generation, and writes a compact JSON result. It does
not generate by default, so random-logit graph fixtures cannot run forever.

`training/scripts/benchmark_android_litertlm_gpu_parity.py` is the host runner.
It stages official and candidate packages only under
`/data/local/tmp/litert_parity`, runs one cold plus configurable warm probes,
parses the runtime's signature, `LITERT_CL` delegation, and MTP-success logs,
and removes only its own temporary device models, reports, and cache labels.
Schema-v4 reports cryptographically bind each result to the exact package:
the runner hashes the host file, verifies the staged device SHA-256 before and
after inference, and checks the device-reported path and size. Pipeline/package
validators independently re-hash the final candidate and require it to equal
the candidate identity in the report. Older path-and-size-only reports are not
shipping evidence.
It deliberately treats structural GPU parity, decode throughput, and MTP
acceptance as three separate gates. Throughput is fail-closed: both selected
runs must report the requested decode count before their tokens/second values
are compared. Target-only requires an exact count. With MTP, LiteRT-LM checks
the cap after `Decode()` and one verifier call can accept several draft tokens,
so the default gate permits only the released four-position bound of
`requested <= decoded <= requested + 4`. A short decode or larger overshoot is
not a valid speed result. Equal package size or delegation cannot override this
gate.

Build and install the probe test APK once:

```powershell
cd android
.\gradlew.bat :app:assembleDebugAndroidTest
adb install -r -t app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk
```

Then compare packages without replacing an app-catalog model:

```powershell
python training/scripts/benchmark_android_litertlm_gpu_parity.py `
  --official C:\path\to\official.litertlm `
  --candidate C:\path\to\candidate.litertlm `
  --output-dir C:\temp\android-gpu-parity `
  --serial DEVICE_SERIAL `
  --mtp `
  --max-num-tokens 8192 `
  --output-tokens 64 `
  --prompt "Write exactly one hundred numbered words."
```

Use `--output-tokens 0` for an initialization/delegation-only gate. For a
Gemma 3 270M package, omit `--mtp` and normally use `--max-num-tokens 4096`.
The default performance gates allow at most 10 percent decode regression and,
when MTP is exercised, at most 0.10 absolute MTP-success-rate loss. These are
shipping gates, not claims that every prompt has identical timing.
Use the same prompt and sampler for both packages. An early stop is useful
diagnostic or acceptance evidence, but it is not a throughput sample. The host
runner exposes `--top-k`, `--top-p`, `--temperature`, and `--seed` for a
reproducible full-length random-fixture probe; a real checkpoint should also be
tested under its intended shipping sampler.

Do not use LiteRT-LM 0.15.0's Kotlin `SuppressTokensConfig` as a fixed-length
benchmark workaround on this runtime. The released AAR's class exposes only a
Kotlin-mangled `getSuppressTokensArray$...` method, while its JNI library looks
up `getSuppressTokensArray()[I`. Passing the config caused a reproducible native
abort on the reference device. The probe therefore does not invoke that API;
re-check a newer official artifact before enabling it in the future.

The connected reference device was an SM-F966B, Android SDK 36, arm64-v8a,
using the app's LiteRT-LM 0.15.0 dependency. The tests used independently
quantized random constants injected into the complete official topology; no
training ran. The latest selected warm runs below use schema-v4 identity gates:
the official and candidate SHA-256 values matched the staged device files both
before and after inference. The 270M official/candidate hashes were
`757e9119fa5bd667a2774fb470ac4afcd3190a21c677f8e69a5d6bc908abdd63` /
`704c4f7f8e091348eb8505fbecd4b59637311d2b0a694c0ee994623a52636374`;
the E2B pair was
`181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c` /
`a2c75a43101897d894eb49129151d17b8a30b083ea5603e60b72491f16d031be`.

| Package pair | Runtime structure result | Validated device result |
|---|---|---|
| Gemma 3 270M official Q8 vs random Q8 | Both packages are 304,005,120 bytes. `decode` was 1,537/1,537 GPU nodes; each of five prefill subgraphs was 1,667/1,667. | Both decoded 64/64 tokens. Official was 54.60 tok/s; random was 55.47 tok/s (1.60 percent faster in this run), passing the 10 percent regression gate. |
| Gemma 4 E2B official vs random mixed W2/W4/W8 target, MTP off | Both packages are 2,588,147,712 bytes. Target subgraphs matched at 2,068/2,068 (`decode`), 1,107/1,107 (`prefill_1024`), 1,107/1,107 (`prefill_128`), and 2,243/2,243 (`verify`). | With identical top-k 40, top-p 1.0, temperature 1.0, seed 42 sampling, both decoded 32/32 tokens. Official was 31.85 tok/s; random was 31.54 tok/s, a 0.98 percent regression inside the 10 percent gate. |
| Gemma 4 E2B official vs the same random target, MTP on | The four target subgraphs above plus the preserved assistant's 198/198-node subgraph fully delegated for both packages with matching shape and signatures. | In the bounded run requesting 32 tokens, official decoded 35 at 39.35 tok/s with MTP success 0.375 (valid four-token overshoot); the random target stopped at 12, 16.60 tok/s, and MTP success 0.0. Structural GPU parity passed, but acceptance and comparable-throughput gates correctly failed. |
| Gemma 4 E2B official vs random target plus random drafter, MTP on | The random target and all 23 independently quantized drafter matrices ran together. Target delegation again matched at 2,068, 1,107, 1,107, and 2,243 nodes; the random drafter matched 198/198, with every subgraph in one GPU partition and all signature shapes equal. | The bounded cold probe requested 8 tokens. Official decoded 10 at 19.33 tok/s with MTP success 0.266667; the full-random candidate decoded 8 at 15.76 tok/s with MTP success 0.133333. Structural GPU parity passed. The 18.47 percent throughput regression and 0.133334 acceptance drop correctly failed the weight-dependent performance gates; this short cold probe is not a shipping-speed claim. |

After the mobile-seed pipeline update in commit `8cbcaf18`, fresh schema-v4
target-only probes were run on the same SM-F966B. Both probes verified the host,
staged-device, and post-run SHA-256 values. For 270M, official and random
packages both decoded 64/64 tokens with full GPU delegation; throughput was
53.973 versus 51.609 tokens/s (4.3795 percent candidate regression, inside the
10 percent gate). For E2B, both decoded 32/32 tokens with matching full GPU
delegation; throughput was 31.781 versus 31.768 tokens/s (0.0426 percent
regression). The current 270M validator therefore passes package inspection,
full graph inspection, artifact identity, GPU delegation, and throughput. The
E2B dual validator accepts target-only parity but still rejects the retained
random-weight MTP report specifically on fixed-length throughput and draft
acceptance. This is the intended fail-closed result; it is not a missing
graph/runtime feature.

The 270M and target-only E2B results are direct evidence that unchanged
graph/layout gives comparable GPU execution speed even when weights differ.
The E2B MTP result proves that the byte-preserved assistant is runnable and
fully delegated with the exact-topology target. It also demonstrates why graph
parity is insufficient for speculative speed: draft-token acceptance is
numerical and therefore weight-dependent. The random target's early stop and
zero acceptance are intentionally rejected as a throughput comparison.

The separate full-random MTP run closes the former runtime-coverage gap for a
random drafter. Because the host had insufficient room for another 2.59 GB
package, the benchmark stream-hashed the virtual composition of the existing
random-target package and the 44,325,712-byte random MTP section, then patched
that section at offset 2,543,812,608 only in its temporary Android copy. The
expected, staged, and post-run package SHA-256 all matched
`fa769933ed955e56d14b0d63cb91360d5bc6aa2a148ab3d75ab9a4fb43fc530d`.
This cryptographically binds the GPU evidence without creating or overwriting
a complete host package. Nonzero random-drafter acceptance confirms that the
MTP path executed; the performance failure remains expected for unrelated
random values.

For a real QAT fine-tune, preserve the default MTP section only as the first
candidate. Benchmark the selected best target checkpoint with MTP off and on
against the untouched official pair. Ship `mtp=true` only if quality passes and
the on-device acceptance/throughput gate passes on the intended prompt set. If
acceptance falls, reduce target drift, use the specialized trained-drafter
experiment, or ship target-only inference. Do not claim that packaging the
official assistant guarantees official throughput. The checked-in trained
assistant path is explicitly a public reconstruction and must pass the same
device gates; Google's exact private drafter training/lowering recipe is still
not public.

## Preferred trained-checkpoint deployment path

Use `training/scripts/build_checkpoint_official_topology.py` (normally through
the two pipeline runners) instead of asking the generic LiteRT-Torch exporter
to recreate the released graph. The released `.litertlm` is the immutable
graph/template authority. The compiler builds independent float branches from
the merged checkpoint, runs the public AI Edge Quantizer at each official
W2/W4/W8 width, and injects only the resulting packed projection constants and
per-axis scales. This is the direct extension of the random-weight experiment
that already passed graph/layout/allocation and Android GPU delegation.

The production initialization now comes from the exact public packed checkpoint,
not the separate dense Q4 release. Run
`training/scripts/reconstruct_gemma4_mobile_training_seed.py` against
`google/gemma-4-E2B-it-qat-mobile-transformers` commit
`dd693ff40353f057ca5f07e945ad867f4afbf2ec`; its model Safetensors must be
2,458,111,846 bytes with SHA-256
`efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4`.
The official source `config.json` is also pinned to SHA-256
`cf6d7dc22738b5e6beb364bac833d78b869f5a6ffd57dfc96c6be3f2abc80424`,
and the retained-compiled evidence report to
`4fa47cf6fefb983a79bebc1e00bdd1f28df8d6570f59d7e979a1791a9a63429a`.
The script is plan-only unless `--execute` is supplied, never runs training,
never overwrites an existing output/partial, and streams the reconstruction so
the largest tensor is not held in memory.

The audited transform produces a text-only `Gemma4ForCausalLM` BF16 checkpoint
with 541 tensors: 263 direct BF16 copies and 278 dequantized matrices. The
payload is 10,062,445,126 bytes and the default 5 GiB limit creates three
Safetensors shards. Its canonical transformation-plan SHA-256 is
`03086afb123acf2c6f359d3cec2b1208f89529bca39e2d29f8b501e0918e3d5c`.
The source-matrix histogram is 62 W2, 146 W4, and 70 W8. This is intentionally
different from the main compiled section's 61 W2, 145 W4, and 71 W8: the source
inventory includes the separate W4 per-layer embedding, while the final main
graph quantizes the direct BF16 per-layer model projection to W8.

Low-bit U8 codes are unpacked in low-bit order and shifted by the published
signed offset; I8 codes are used directly. Published F32 scales are applied by
output row. The `embed_tokens_per_layer` source has logical shape
`[262144, 8960]` and scale shape `[262144, 35]`; each scale column covers exactly
256 logical columns. K/V projection tensors serialized for shared-KV layers
15-34 are omitted because the dense Transformers text architecture does not own
those duplicates. All 262 retained RMSNorm/scalar/vector tensors are among the
direct BF16 copies, so their public values remain exact.

The separate dense `gemma-4-E2B-it-qat-q4_0-unquantized` seed remains rejected.
On 2026-08-08, `audit_hf_retained_constant_parity.py` compared its commit
`6befbaca7398925921802abd1f277b495b78b738` with the packed mobile checkpoint.
All 262 retained schemas matched, but only 50 values were exact and 212 differed;
their aggregate digests were
`c3be6ec6262092beaceb1b4c3d2b01a1073290d8bfb739ebe3cf40289e280e6c` and
`1bc159104579c507d3ebfc388bcc44e4d1e567f3e4ba209ef0b2503be470f0aa`.
Matching official ownership and shapes therefore does not make those releases
the same numerical base. Keep the Q4 config only as a rejected research
baseline; never bypass its 212-value mismatch.

The compiled mapping itself is no longer an open question at the public
precision boundary. A second read-only audit,
`audit_gemma4_mobile_retained_compiled_parity.py`, hash-bound the released
2,588,147,712-byte package to SHA-256
`181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`, selected
the canonical `decode` subgraph, and mapped constants by semantic consumer
names. It resolved all 262 public retained tensors to 242 unique compiled
FLOAT32 buffers. All 262 compiled values round byte-exactly to the checkpoint's
BF16 values with IEEE round-to-nearest-even. The inventory consists of 35 each
of the five layer-norm roles, 35 query norms, 15 key norms, 35 layer scalars,
one final norm, and one per-layer projection norm. Shared buffers explain why
262 semantic tensors map to 242 unique buffers.

This proves the packed mobile checkpoint is the public numerical authority at
BF16 precision. It does not recover compiled constants' lower FLOAT32 mantissa
bits, Google's pre-quantization master weights, training data, observer state,
optimizer schedule, or private QAT recipe. Dequantized values are public
quantization-cell centers rounded to BF16 RNE. Fine-tuning is expected to change
the trained weights; exact graph/operators/quantization layout, not official
weight equality, is the deployment-speed invariant.

`mobile_training_seed_manifest.json` is the executable gate. Training, merge,
and the direct compiler recompute its mapping-plan digest; require the 541-key
weight map and tensor-hash inventory; verify the exact source, official package,
retained-compiled report, dense config, shard set, payload sizes, and all shard/
auxiliary hashes; and require the manifest directory to equal
`model.model_source`. The recommended config is
`training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml`. Its canonical
model ID stays `google/gemma-4-E2B-it-qat-mobile-transformers`, while its local
load source is `training/outputs/seeds/gemma4_e2b_mobile_dequantized_text_hf`.
Until that output is explicitly materialized, the checked-in pipeline correctly
reports `mobile_training_seed_unverified` and cannot train, merge, or export.

Plan first; the output directory is not created:

```powershell
python training/scripts/reconstruct_gemma4_mobile_training_seed.py `
  --source-safetensors <packed-checkpoint-dir>/model.safetensors `
  --source-config <packed-checkpoint-dir>/config.json `
  --retained-compiled-report <evidence-dir>/gemma4-mobile-retained-compiled-parity.json `
  --output-dir training/outputs/seeds/gemma4_e2b_mobile_dequantized_text_hf
```

After inspecting the plan, repeat with `--execute` to materialize the seed.
This writes model/config/manifest files only; it still does not train. Then use
the pipeline's default plan before selecting any explicit stage:

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py `
  --config training/configs/pipelines/gemma4_e2b_mobile_mtp.yaml
```

Training remains opt-in through `--execute-training`; do not combine seed
reconstruction, training, merge, export, and device promotion into an
unreviewed one-shot command.

Training metadata version 3 records the verified seed manifest; merge metadata
version 4 additionally binds the canonical base ID, local base source,
training-config SHA-256, QAT run, adapter hashes, and every merged shard/index
hash. The mobile profile requires exact Transformers checkpoint loading with no
missing, unexpected, mismatched, or errored state keys, and the merge/compiler
provenance preserves that fail-closed requirement. The compiler rejects an old
merge, generic or base-only-QAT metadata, the
raw packed checkpoint, the Q4 seed, a manifest/source mismatch, package hash
mismatch, retained-constant mismatch, or incomplete source mapping. Retained
non-inventory constants remain hash-bound to the official package, while all
277 target FC/embedding constants are regenerated from the fine-tuned merged
checkpoint. This proves official graph/operator/layout equivalence; device
quality, throughput, and MTP acceptance remain separate promotion gates.

Before quantization, the compiler derives a canonical QAT module for every
official inventory entry and compares its configured fake-quant bit width with
the released graph. E2B must match all 277 assignments (145 W4, 61 W2, 71 W8),
and 270M must match all 127 W8 assignments. The tied E2B language-model head is
audited through its token-embedding source, matching the alias used by the
checkpoint compiler. The training-scope gate separately requires the observed
256-column grouping for `embed_tokens_per_layer`. Any excluded, unmapped,
differently quantized, or differently grouped entry makes the plan
non-executable.

Current exact bindings for the supplied reference artifacts are:

| family | public training seed | target section | unique mapped weights | production status | package SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| Gemma 4 E2B | BF16 text reconstruction of `google/gemma-4-E2B-it-qat-mobile-transformers` | `tf_lite_prefill_decode` | 277 | script/config ready; blocked only until the local 541-tensor seed is materialized and its manifest verifies; device quality/speed still required | `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c` |
| Gemma 3 270M IT | `google/gemma-3-270m-it` | `TF_LITE_PREFILL_DECODE` | 127 | exact-base path; device quality/speed still required | `757e9119fa5bd667a2774fb470ac4afcd3190a21c677f8e69a5d6bc908abdd63` |

The optional trained E2B drafter is seeded from
`google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant`. Google states that a
QAT target and assistant must use matching precision. The ordinary
`gemma-4-E2B-it-assistant` remains the correct pair only for the ordinary BF16
target; it is rejected by this mobile QAT training pipeline. Official-drafter
mode does not load either Transformers assistant: it preserves the compiled
`tf_lite_mtp_drafter` section from the hash-bound `.litertlm` byte-for-byte.

QAT module rules are matched against both raw `named_modules()` paths and
canonical names with known PEFT/multimodal wrappers removed. This matters for
the anchored public rule `^lm_head$`: a PEFT path such as
`base_model.model.lm_head` must still receive W2 fake quantization. The tests
also cover a doubly wrapped `base_model.model.model.language_model...` path and
official exclusions. The text-only `Gemma4ForCausalLM` paths
`model.layers.*`, `model.embed_tokens`, `model.embed_tokens_per_layer`, and
`model.per_layer_model_projection` are explicitly aliased back to the public
`language_model.*` namespace; otherwise W2/W4 tensors would silently receive
the default W8 fake quantizer.

The 270M mapping has been checked against an existing merged BF16 checkpoint:
127/127 headers map without transpose or shape errors. That older checkpoint
lacks merge-metadata v3 and is intentionally rejected as a production input;
mapping compatibility is not QAT provenance. The raw packed Gemma 4 checkpoint
is intentionally rejected as a merged floating-point input; only its
manifest-verified BF16 text reconstruction is accepted as the training base.

For 270M, train weight-only QAT (`ste_ai_edge`, W8, activation bits 32), including
the embedding table. The released Q8 graph has INT8 per-row weights and FLOAT32
input/output edges; the former W8A8 training setting optimized a different
numerical graph. Keep the training model ID and package/hash paired. A Gemma 3
base, Gemma 3 IT, and FunctionGemma package may share architecture while still
having different embeddings, norms, tokenizer, or chat behavior.

For E2B, target compilation always produces a streamed copy of the official
package with only target-section constants changed and first proves the MTP
section byte-identical. In `official` mode that package is final. In `trained`
mode a second fail-closed stage replaces only the 23 provenance-bound drafter
matrices while retaining the released MTP graph/operators/layout and every
other constant. Either mode gives the same GPU topology, not automatically the
same MTP speed: target and drafter weights determine draft acceptance even when
delegation is identical.

Overall recommendation: keep the official mobile model and byte-exact official
MTP drafter as the deployable accuracy baseline. Materialize the checked-in
mobile reconstruction, train with the mobile-seed QAT config, select the best
golden checkpoint by strict IR quality, merge it with provenance, compile it
into the official topology, then run `--validate-android-gpu`. Never promote
the rejected Q4-seed hybrid merely because its graph runs on GPU. The E2B pipeline executes
two separate fail-closed
reports under `android_gpu_parity/target_only` and `android_gpu_parity/mtp_on`.
Promote E2B with `mtp=true` only when full delegation, fixed-length warm
target-only throughput, MTP acceptance, MTP-on throughput, and output-quality
gates all pass. Otherwise ship the same exact target graph with MTP disabled.
For 270M, require full delegation and fixed-length warm throughput parity; MTP
is not applicable.

## Rules for future agents

1. Re-check official model cards and runtime supported-model lists because Gemma
   4 tooling is changing quickly.
2. Read generated merge/conversion manifests before debugging downstream output.
3. Do not weaken explicit `--execute` guards.
4. Do not replace the specialized `train_assistant` implementation with a generic MTP loss;
   Keep its target-conditioned four-step shared-KV contract and 23-matrix scope.
5. Preserve the standard Gemma 4 config as a baseline.
6. Preserve dataset and golden artifacts; do not regenerate or delete them while
   repairing scripts unless explicitly authorized.
7. Update this file when official support changes, including verification date,
   exact checkpoint/runtime revisions, and which former gap is now closed.
8. Keep generated weights, adapters, GGUF files, `.litertlm` files, and model
   caches out of Git. Commit only small plans/manifests when intentionally needed.

## Official sources

- Gemma 4 overview, formats, memory routing, and QAT collections:
  https://ai.google.dev/gemma/docs/core
- Official QAT-derived E2B target model card and matching-precision rule:
  https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized
- Official matching assistant model card:
https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant
- Official Gemma 4 mobile-transformers config (public module bit map):
  https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/blob/main/config.json
- Official Transformers Gemma packed-weight loader (W2/W4 unpack order,
  signed offsets, block-scale application):
  https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/gemma_quant.py
- Gemma 4 MTP Transformers tutorial:
  https://ai.google.dev/gemma/docs/mtp/mtp
- Gemma 4 MTP architecture overview:
  https://ai.google.dev/gemma/docs/mtp/overview
- Google Gemma 4 QAT release description:
  https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
- Google Gemma 4 MTP release and up-to-3x benchmark claim:
  https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- Current official Hugging Face QLoRA guide and dependency floors:
  https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
- Google AI Edge LiteRT-LM runtime:
  https://github.com/google-ai-edge/LiteRT-LM
- LiteRT-LM Gemma 4 MTP drafting and verification loop:
  https://github.com/google-ai-edge/LiteRT-LM/blob/240f2397420757dd40f78d1cdca9618bc476eecc/runtime/executor/llm_litert_mtp_drafter.cc
- LiteRT-LM `.litertlm` header schema:
  https://github.com/google-ai-edge/LiteRT-LM/blob/main/schema/core/litertlm_header_schema.fbs
- LiteRT-LM header/section reader:
  https://github.com/google-ai-edge/LiteRT-LM/blob/main/schema/core/litertlm_read.cc
- LiteRT Torch generative quantization/export API:
  https://github.com/google-ai-edge/litert-torch
- LiteRT Community Gemma 3 270M package family (Q8 `.litertlm` and Q4_0 web `.task`):
  https://huggingface.co/litert-community/gemma-3-270m-it
- Official Gemma 3 270M Q4_0-derived checkpoint:
  https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized
- Gemma 3 270M model card:
  https://huggingface.co/google/gemma-3-270m
- FunctionGemma 270M documentation:
  https://ai.google.dev/gemma/docs/functiongemma
- Gemma 4 assistant architecture and Transformers API:
  https://huggingface.co/docs/transformers/model_doc/gemma4_assistant
- Gemma 4 technical report:
  https://arxiv.org/abs/2607.02770
- AI Edge Quantizer public recipe implementation (`gemma4_mixed48`):
  https://github.com/google-ai-edge/ai-edge-quantizer/blob/main/ai_edge_quantizer/recipe.py

When citing speed, say **up to** the reported number and retain the device,
runtime, prompt, batch-size, and precision context. Never turn a release benchmark
into a guarantee for A2UI.
