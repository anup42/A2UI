# Response-to-IR Training

This folder trains local Stage 3 models that convert Stage 2 response text into the Android A2UI Express v1 IR:

```text
<a2ui>
root=Column([content])
content=Text("...","body")
</a2ui>
```

`dataset/` remains responsible for generating queries, responses, assets, and cloud IR. `training/` consumes completed dataset runs and provides a model-agnostic path for data preparation, SFT training, evaluation, export, and Android packaging. FlatSpec is accepted only as a read-only legacy source; Compact IR v2 is migration-only and is never a training target.

## Main Flow

1. Prepare response-to-IR pairs from a dataset run.

```powershell
python training/scripts/prepare_dataset.py --config training/configs/datasets/dataset_v1_stage3.yaml
```

For Gemma 4 training from a folder that contains one or more Stage 3 `genui.jsonl`
files, prepare a deterministic 90/10 train/validation split with:

```powershell
python training/scripts/prepare_dataset.py --config training/configs/datasets/stage3_folder_90_10.yaml --source-genui-dir dataset/data/runs/dataset_v3 --output-dir training/outputs/datasets/stage3_folder_90_10
```

The folder reader is recursive by default (`**/genui.jsonl`). It writes
`train.jsonl`, `val.jsonl`, `test.jsonl`, and `all.jsonl`; `all.jsonl` is used for
fixed-set evaluation jobs.

When `url_preprocessing.enabled` is true, preparation replaces every URL/URI and
local asset reference in both the response prompt and IR target with typed
placeholders. The exact placeholder map is stored in row metadata and evaluation
restores the original references before comparison or exported prediction review.

Prepare the golden50 set once before training:

```powershell
python training/scripts/prepare_dataset.py --config training/configs/datasets/golden50_stage3_eval.yaml
```

2. Train an adapter model.

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma_e2b_ir_lora.yaml
```

Start Gemma 4 E2B LoRA training with golden50 evaluation at each configured
Trainer evaluation event:

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_lora.yaml
```

The Gemma 4 E2B profile uses the fast-tokenizer loader and PEFT's
`all-linear` discovery used by the supplied working trainer. Golden evaluation
runs every `training.eval_steps`, distributes generation across DDP ranks, and
saves the highest-`overall_score` adapter under
`training/runs/gemma4_e2b_ir_lora/best_golden_checkpoint`. Per-evaluation
predictions and metrics remain under
`training/outputs/eval/gemma4_e2b_ir_lora/golden50/step_*`.

The callback is controlled by the `golden_eval` YAML block. Use
`trigger: epoch` for epoch-end evaluation, increase `interval` to evaluate less
often, or set `save_best_checkpoint: false` to retain metrics without saving a
second adapter checkpoint. `max_input_tokens + max_new_tokens` is bounded by
the configured model context.

### Gemma 4 QAT-derived LoRA and MTP workflow

The recommended low-bit experiment is intentionally separate from the standard
Gemma 4 baseline. It fine-tunes only Google's unquantized Q4_0 QAT-derived
target with BF16 LoRA, keeps the matching assistant frozen, and requires
post-merge Q4_0 conversion plus final runtime/device validation. It is **not**
continued QAT or joint target/assistant training.

Read the durable support boundaries, recommendation, command inventory, and
promotion gates before using it:

- `training/docs/gemma4_e2b_qat_mtp_knowledge.md`
- `docs/gemma4_e2b_qat_mtp_implementation_prompt.md`

Install the current Gemma 4 dependency overlay and run the no-model static
preflight:

```powershell
python -m pip install -r training/requirements-gemma4-qat.txt
python training/scripts/validate_qat_mtp_workflow.py
```

The expensive helpers are plan-only unless `--execute` is supplied:

```powershell
python training/scripts/merge_qat_lora.py
python training/scripts/benchmark_qat_mtp.py
python training/scripts/convert_qat_q4_0.py --llama-cpp-dir C:\path\to\llama.cpp
```

The later authorized training command is:

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_qat_lora.yaml
```

Do not run that command for code-only setup or validation.

This requires a GPU machine with the packages in
`training/requirements-training.txt`. On CPU-only machines, use compile/tests only;
do not run the training command.

### True QAT SFT for Gemma 4 E2B and Gemma 270M

The repository also has an opt-in fake-quantization-aware LoRA path. The Gemma
4 profile consumes the public mobile W2/W4/W8 module map, fake-quantizes the
matching base linears and embeddings, and can use the public AI Edge min/max
range convention (`ste_ai_edge`: full signed W2/W4, narrow symmetric W8).
Gemma 270M remains W8A8. This is different from the QAT-derived `qat_mtp`
profile above, does not train an MTP assistant, and does not claim Google's
private mobile observer/calibration recipe.

Validate all profiles without loading models or running training:

```powershell
python training/scripts/validate_qat_training.py
```

Later, with an explicitly authorized GPU run, select a target profile:

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_qat_sft.yaml
python training/scripts/train_sft.py --config training/configs/models/gemma3_270m_ir_qat_sft.yaml
python training/scripts/train_sft.py --config training/configs/models/functiongemma_270m_ir_qat_sft.yaml
```

Read `training/docs/gemma4_e2b_qat_mtp_knowledge.md` before changing the
quantizer or export settings. The checked-in E2B config's `schema_path` and
`ste_ai_edge` setting reproduce only public observable deployment semantics;
they are not Google's hidden QAT trainer. Merge and quantize the selected
adapter with the target LiteRT/LiteRT-LM recipe, then measure accuracy and
device latency against the untouched low-bit baseline.

### Gemma 4 E2B mobile QAT -> best checkpoint -> MTP package

The end-to-end mobile hand-off is defined in
`training/configs/pipelines/gemma4_e2b_mobile_mtp.yaml`. It selects the
golden-set best adapter, merges it into the BF16 base, and keeps the official
MTP drafter separate from target training. The default is plan-only:

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py
```

Set `pipeline.source.base_litertlm` to the exact official package variant and
provide either `pipeline.source.target_litertlm` or a raw
`pipeline.source.target_section` produced by a compatible Gemma mobile
exporter. Then compose the package:

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py --compose
python training/scripts/validate_litertlm_mtp_gpu.py `
  training/outputs/pipelines/gemma4_e2b_mobile_mtp/gemma4_e2b_mobile_mtp.litertlm
```

The composer replaces only `tf_lite_prefill_decode`, requires equal section
size plus matching TFLite topology/quantization/buffer-storage layout, and preserves
`tf_lite_mtp_drafter` byte-for-byte. It does **not** train or convert the
assistant. A package passing this desktop gate is structurally ready for an
`mtp=true` request, not proof of Android GPU execution. Supply a device JSON
report from a real LiteRT-LM GPU harness to the validator for the final runtime
gate.

Training, merge, and the optional public standalone exporter are explicit:

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py --execute-training
python training/scripts/run_gemma4_e2b_mobile_mtp.py --execute-merge
```

`--execute-public-export` is intentionally not enabled by default. The public
`litert-torch` path is useful for a candidate artifact, but it is not evidence
that Google's private mobile W2/W4/W8-A8 exporter or calibration has been
reproduced; the strict composer rejects a candidate with a different graph,
layout, or section size.

### Gemma 3 270M QAT -> LiteRT-LM INT8

The equivalent 270M pipeline is separate because Gemma 3 270M has no Gemma 4
MTP drafter. It trains the checked-in W8A8 QAT profile, selects the golden-set
best adapter, merges it, and invokes the public `dynamic_wi8_afp32` LiteRT
exporter:

```powershell
python training/scripts/run_gemma270m_qat_litertlm.py
python training/scripts/run_gemma270m_qat_litertlm.py --execute-training
python training/scripts/run_gemma270m_qat_litertlm.py --execute-merge
python training/scripts/run_gemma270m_qat_litertlm.py --execute-export
```

The generated artifact is an INT8 `.litertlm`, not an INT4/Q4_0 model and not
an MTP package. Validate its package envelope before device testing:

```powershell
python training/scripts/validate_gemma270m_litertlm.py `
  training/outputs/pipelines/gemma3_270m_qat_litertlm/gemma3_270m_qat_int8.litertlm
```

Use `--inspect-graphs` for the slower embedded-TFLite graph check and provide
an Android GPU device report for the runtime gate. Do not enable MTP for this
model; the validator treats an MTP section as unexpected rather than attaching
the Gemma 4 assistant.

To audit the publicly observable Google mobile schema without downloading
weights:

```powershell
python training/scripts/audit_gemma4_mobile_schema.py
```

To inspect a local `.litertlm` package and its embedded TFLite graph without
loading model weights into a training framework, use:

```powershell
python training/scripts/audit_litertlm_package.py C:\path\to\model.litertlm
```

To audit the observable mobile quantization contract (including INT2/INT4/INT8
weights, symmetric per-axis scales, INT8 activation edges, and a separate MTP
graph), use the optional `tflite` schema bindings:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/audit_litertlm_recipe.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --strict-mobile
```

The public AI Edge `gemma4_mixed48` recipe is captured separately in
`training/configs/quantization/gemma4_mixed48_public_recipe.json`; it is a
W4/W8 channelwise recipe and must not be described as the private mobile W2/W4/W8-A8
artifact recipe.
The recipe-audit JSON also emits `fully_connected_assignments`, grouped by
layer and operation family, so later work can compare bit assignments without
mistaking them for Google's hidden QAT/calibration schedule.
The checked-in snapshot of those observable rules is
`training/configs/quantization/gemma4_e2b_mobile_observable_contract.yaml`;
it is evidence metadata, not a drop-in private exporter recipe.

For the explicit random-weight graph parity experiment (still no training),
review the plan first and add `--execute` only in the separate LiteRT export
environment:

```powershell
python training/scripts/build_random_litertlm_parity.py `
  --model-source google/gemma-3-270m `
  --official-artifact C:\path\to\official.litertlm `
  --output-dir C:\temp\random-litertlm-parity `
  --quantization-recipe dynamic_wi8_afp32
```

Read `training/docs/gemma4_e2b_qat_mtp_knowledge.md` for the evidence levels:
random weights can test graph and quantization layout, but cannot prove equal
learned weights, private calibration, or exact Google recipe provenance.

For a small executable public-recipe check that builds a deterministic random
TFLite graph and quantizes it without Transformers or training:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_random_tflite_recipe_parity.py `
  --output-dir C:\temp\random-gemma4-public-recipe `
  --recipe gemma4_mixed48 `
  --execute
```

To exercise the observable mobile W2/W4/W8-A8 contract with a tiny random
graph, use the separate static experiment below. It applies deterministic
random calibration inputs and records the resulting TFLite layout; it does not
train, download weights, or claim Google's private recipe. The custom recipe
sets `skip_checks=true` because the public AI Edge policy does not advertise
arbitrary static W2 `FULLY_CONNECTED` operations:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_random_mobile_contract_parity.py `
  --output-dir C:\temp\random-gemma4-mobile-contract `
  --official-artifact C:\path\to\gemma-4-E2B-it.litertlm `
  --execute
```

The synthetic report should show INT2/INT4/INT8 fully-connected weights,
channelwise axis-0 symmetric scales, and symmetric INT8 input/output edges.
`synthetic_contract_match=true` is only converter/layout evidence;
`exact_official_mobile_reproduction` remains false by construction.

For stronger artifact-level evidence, randomize the weights in the actual
official TFLite section while preserving its graph and quantization layout:

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

The harness writes standalone `.tflite` files and never rewrites the source
`.litertlm`. A matching structural and quantization-layout hash proves graph
and packing/layout parity only; random weight/scales hashes intentionally do
not match and the private QAT recipe remains unrecovered. It also supports
external TFLite constant buffers (`Buffer.offset`/`Buffer.size`), which is the
layout used by the local Gemma 3 270M INT8 candidate:

```powershell
python training/scripts/build_random_official_topology_parity.py `
  C:\path\to\gemma-3-270m-ir-int8.litertlm `
  --model-type tf_lite_prefill_decode `
  --output C:\temp\gemma270m-prefill-random.tflite `
  --execute `
  --runtime-allocate
```

That candidate yielded 127 randomized INT8 weight buffers with identical
graph/layout hashes. `--runtime-allocate` additionally checks that the
original extracted section and randomized output allocate with identical
signatures. It is a custom local export. The byte-verified official
`gemma3-270m-it-q8.litertlm` target is 304,005,120 bytes with SHA-256
`757e9119fa5bd667a2774fb470ac4afcd3190a21c677f8e69a5d6bc908abdd63`; its
v1.3 graph has six subgraphs, 9,872 operators, 11,702 tensors, 732 INT8
fully-connected weights, and six INT8 embedding tables. The local v1.5 graph
has 215 subgraphs and 4,611 tensors, so it cannot become byte-equivalent by
changing only its scale formula. The public exporter does not
expose the official artifact's audio and `tf_lite_mtp_drafter` packaging path,
so this harness is an evidence/regression tool, not a private recipe
substitute.

If the merged BF16 source safetensors for that local candidate are available,
prove its public INT8 rule directly:

```powershell
python training/scripts/audit_litertlm_weight_recipe.py `
  C:\path\to\gemma-3-270m-ir-int8.litertlm `
  --source-model C:\path\to\merged_hf `
  --model-type tf_lite_prefill_decode `
  --strict
```

The audit matched 127/127 candidate weights exactly: axis-0 scales were
`max(abs(source_row))/127` and the packed INT8 rows matched round-and-clip.
That identifies the local export as public `dynamic_wi8_afp32`/channelwise
`dynamic_wi8c_afp32`; it does not identify the gated official package or a
private QAT training recipe.

To build a genuinely fresh FlatBuffer from the official section's observable
weight inventory, use `build_fresh_random_quantized_graph.py`. It does not copy
the official graph: it creates independent random branches, applies the
observable W2/W4/W8 symmetric axis-0 rule, and compares the new quantized weight
inventory with the official one. The default is in-memory; add `--output` only
when a standalone `.tflite` fixture is wanted:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_fresh_random_quantized_graph.py `
  C:\path\to\official-gemma3-270m-it-q8.litertlm `
  --model-type TF_LITE_PREFILL_DECODE `
  --include-embeddings
```

On the verified 270M target this produced 127 fresh random branches (126
linear matrices plus the embedding table) with an identical inventory digest
and `fresh_quantization_layout_match=true`. `graph_structure_match` remains
false by design; the separate topology harness is required to prove official
cache/signature/operator parity. For Gemma 4 MTP, the same command over
`tf_lite_mtp_drafter` reproduced all 23 observable weight layouts (13 W4 and
10 W8) in a fresh 43.9 MB graph, while deliberately leaving graph structure
and learned values different.

To run the same inventory through the public AI Edge Quantizer rather than the
hand-built quantization loop, use
`build_converter_random_inventory_parity.py`. It builds an independent random
FLOAT32 FlatBuffer, applies a per-branch public W2/W4/W8 recipe, calibrates A8
edges when required, and compares only the resulting quantized FC layout:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_converter_random_inventory_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --model-type tf_lite_mtp_drafter `
  --output-dir C:\temp\random-inventory-mtp `
  --calibration-samples 2 `
  --threads 1
```

The full MTP inventory matched 23/23 layouts (13 W4, 10 W8); the full 270M
section matched 126/126 FC layouts (the embedding is skipped in this ordinary
FC graph and is covered by the topology harness); and a bounded Gemma 4
prefill run matched 8/8. These runs prove public converter/layout behavior,
not graph topology, learned weights, private QAT, or exact LiteRT-LM package
provenance. Reports intentionally keep
`converter_quantization_layout_match=true` separate from
`exact_official_model_match=false`.

### Converter-driven official-topology parity

`build_converter_topology_parity.py` is the stronger graph experiment. It
unpacks a selected official TFLite section with the public object API, replaces
its low-bit constants with deterministic random FLOAT32 constants, bypasses the
now-invalid Q/DQ edges, and sends that topology through the public AI Edge
Quantizer. It never modifies the `.litertlm` source. The report includes both
normal fingerprints (serializer-sensitive buffer indices included) and
buffer-agnostic fingerprints for execution topology and quantization layout:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_converter_topology_parity.py `
  C:\path\to\gemma3-270m-it-q8.litertlm `
  --model-type TF_LITE_PREFILL_DECODE `
  --output-dir C:\temp\converter-topology-270m `
  --calibration-samples 2 `
  --threads 1 `
  --in-memory
```

The verified Gemma 3 270M run preserved all six subgraphs, 9,872 operators,
11,702 tensors, 732 `FULLY_CONNECTED` operators, six `EMBEDDING_LOOKUP`
operators, and all 737 INT8 quantized tensors. The normal hash differs because
the standalone FlatBuffer is reserialized, but both
`converter_execution_topology_match_ignoring_buffer_indices` and
`converter_quantization_layout_match_ignoring_buffer_indices` are `true`.
`exact_official_model_match` remains `false`: constants are random and package
serialization/metadata are not reconstructed.

The same experiment on the official Gemma 4 `tf_lite_mtp_drafter` section is
an explicit public-boundary result. The converter retained 30 subgraphs and
all 23 FC records, but produced 357 rather than 365 operators, 615 rather than
575 tensors, 22 rather than 30 `DEQUANTIZE` operators, and 63 rather than 79
quantized tensors. Both buffer-agnostic matches are `false`. This mixed W4/W8
MTP graph therefore depends on Google's private/custom lowering path; an
independent public converter cannot claim exact MTP graph parity.

### Converter-driven random constants injected into official topology

`build_converter_random_topology_injection_parity.py` quantizes one independent
random FLOAT32 branch per official FC/embedding weight with the public AI Edge
Quantizer, then copies only the packed bytes and per-axis scales into an
in-memory copy of the released section. It preserves the official operators,
tensors, signatures, metadata, and cache wiring. Its low-level builder emits a
real `EMBEDDING_LOOKUP`, so this check covers the complete 270M inventory rather
than silently replacing the token embedding with an FC branch.

The latest verified seed-42 runs used two calibration samples and produced:

| section | inventory | public converter layout | injected structure/layout | allocation |
| --- | ---: | --- | --- | --- |
| Gemma 4 E2B `tf_lite_embedder` | 1 embedding (W2) | 1/1 | true / true | true (no default delegates) |
| Gemma 4 E2B `tf_lite_per_layer_embedder` | 35 embeddings (W4) | 35/35 | true / true | true (six batches; no default delegates) |
| Gemma 4 E2B `tf_lite_audio_encoder_hw` | 120 FC (W2/W4) | 120/120 | true / true | true (no default delegates) |
| Gemma 4 E2B `tf_lite_vision_encoder` | 112 unique FC buffers (W8) | 112/112 | true / true | true (no default delegates) |
| Gemma 4 E2B `tf_lite_prefill_decode` | 277 FC (W2/W4/W8) | 277/277 | true / true | true (35 batches; no default delegates) |
| Gemma 4 `tf_lite_mtp_drafter` | 23 (13 W4, 10 W8) | 23/23 | true / true | true (no default delegates) |
| Gemma 3 270M `TF_LITE_PREFILL_DECODE` | 127 (126 FC + 1 embedding) | 127/127 | true / true | true (no default delegates) |

The full E2B prefill independent float32 inventory is about 8.5 GiB. The
harness automatically quantizes it in deterministic eight-weight batches to
stay below FlatBuffers' 2 GiB offset limit, then combines the global ordinals
before injecting constants. The injected quantization values are random by construction, so the reports
correctly keep `quantization_values_match=false` and
`exact_official_model_match=false`. This is proof that public converter output
fits the official graph/layout and allocates in LiteRT; it is not recovery of
Google's learned QAT weights, observers, calibration data, private exporter,
or byte-identical package.

The JSON report makes the requested network check explicit:
`random_quantized_network.same=true` means complete FC/embedding inventory,
operator structure, quantization layout, and buffer-index-independent execution
topology all match. `converter_to_injected_quantization_values_match=true`
additionally proves that the converter's random packed bytes and scales survive
injection into the official section exactly. It does not compare those random
values with Google's learned values; that remains separately reported as
`quantization_values_match` and `exact_official_model_match`. With
`--runtime-allocate`, the nested
`runtime_allocation_match` is an additional allocation-only check.

Use `--rebuild-flatbuffer` for the stronger independent serialization check.
It deserializes the official section through the public
`ai_edge_litert.schema_py_generated.ModelT`, materializes external buffers,
places the AI Edge-converter random constants into that object graph, and packs
a new `TFL3` FlatBuffer. The report adds
`rebuilt_quantized_network.same=true` only when structural/layout hashes,
buffer-index-independent execution topology/layout, converter-constant digest,
and optional runtime allocation all match. Verified rebuilt runs pass for the
E2B token embedder (1/1), per-layer embedder (35/35), audio encoder (120/120),
vision encoder (112/112), main decoder (277/277), MTP (23/23), and Gemma 3
270M (127/127). The audio and vision counts are unique weight buffers; the
audit may show more operator occurrences when a buffer is reused. This still keeps
`quantization_values_match=false` and `exact_official_model_match=false`, as
required for a random model.

The current public LiteRT-Torch exporter still lists quantized safetensor
support as a TODO and rejects a Hugging Face `quantization_config` with
`NotImplementedError("Quantized checkpoint is not supported yet.")`. The
maintainer's [public issue response](https://github.com/google-ai-edge/litert-torch/issues/998)
also identifies the pre-built Gemma 4 files as exports of QAT-mobile variants
and notes that quantized-safetensor conversion and MTP/audio export were not
public at that point. Therefore this harness can establish public network
compatibility, but cannot manufacture Google's private QAT/export recipe.

The report also proves the package boundary: the Gemma 4 MTP section is
44,325,712 bytes at offset 2,543,812,608 in the 2,588,147,712-byte package,
and the 270M section is 299,261,424 bytes at offset 4,734,976 in the
304,005,120-byte package. Prefix/suffix hashes show that the random fixture
changes only the selected TFLite section; tokenizer, metadata, and all other
sections remain byte-for-byte unchanged.

Example (the temporary 270M model fixtures are removed automatically):

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/build_converter_random_topology_injection_parity.py `
  C:\path\to\gemma3-270m-it-q8.litertlm `
  --model-type TF_LITE_PREFILL_DECODE `
  --output-dir C:\temp\converter-injection-270m `
  --calibration-samples 2 `
  --threads 1 `
  --in-memory `
  --rebuild-flatbuffer `
  --runtime-allocate `
  --runtime-without-default-delegates
```

Add `--package-output C:\temp\random-injected.litertlm` when a full package
fixture is needed; it streams the package and replaces only the selected
section, so the original artifact is never overwritten.
For the full E2B prefill, the automatic eight-weight batching can be overridden
with `--converter-batch-size N` when memory and FlatBuffer limits permit.
Run the same command against the Gemma 4 artifact with
`--model-type tf_lite_prefill_decode` for the main graph or
`--model-type tf_lite_mtp_drafter` for the draft graph. The same command covers
`tf_lite_embedder`, `tf_lite_per_layer_embedder`, `tf_lite_audio_encoder_hw`,
and `tf_lite_vision_encoder`; the adapter/end-marker sections contain no
quantized FC/embedding weight inventory.

The full E2B main-graph invocation used for the verified 35-batch run is:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/build_converter_random_topology_injection_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --model-type tf_lite_prefill_decode `
  --output-dir C:\temp\converter-injection-e2b `
  --calibration-samples 2 `
  --threads 1 `
  --in-memory `
  --runtime-allocate `
  --runtime-without-default-delegates
```

On the supplied E2B artifact this reports
`random_quantized_network.same=true`,
`converter_to_injected_quantization_values_match=true`,
`runtime_allocation_match=true`, `quantization_values_match=false`, and
`exact_official_model_match=false`.
The first field is the requested graph/layout network check; the latter two
must remain false because the fixture intentionally uses random constants.

If the goal is only to exercise a random quantized network with the *official*
MTP topology, `build_random_official_topology_parity.py` can patch the 23
serialized W4/W8 buffers in place while preserving every operator, tensor,
signature, and quantization-layout field. The verified MTP run randomized all
23 buffers (13 W4, 10 W8), round-tripped every low-bit buffer, and returned
`graph_structure_match=true` and `quantization_layout_match=true`. That is an
artifact-preserving regression fixture, not an independent reconstruction of
the private exporter or QAT recipe.

### Public mobile checkpoint constant parity

When the released public checkpoint is available, use
`audit_gemma4_mobile_checkpoint_parity.py` to compare its serialized constants
and scales with the supplied Gemma 4 LiteRT-LM sections:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mobile_checkpoint_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --source-safetensors C:\path\to\gemma4-mobile\model.safetensors `
  --include-embedders `
  --strict `
  --output C:\temp\gemma4-mobile-checkpoint-parity.json
```

The audit is read-only and memory-maps the safetensors file. The public mobile
checkpoint stores W2/W4 values as unsigned offset codes; the LiteRT buffers
use the corresponding signed low-bit codes. The verified conversion is
`code - 2 (mod 4)` for W2, `code - 8 (mod 16)` for W4, and byte-for-byte copy
for W8. With that serialization mapping, the local public checkpoint matched
1/1 token-embedding buffers, 35/35 per-layer-embedding tables, and 276/276
comparable prefill/decode buffers (all scales exact). One W8
`per_layer_model_projection` is BF16 in the public checkpoint and has no source
scale, so it is reported as unresolved rather than guessed. Exact comparable
constants are strong evidence that the released checkpoint supplies the
artifact's observable mobile weights; they do not expose Google's private QAT
loss, calibration data, optimizer schedule, or exporter/package source.

For the MTP drafter, `audit_gemma4_mtp_assistant_parity.py` maps the supplied
23-record traversal to the public four-layer assistant safetensors and applies
the public symmetric per-output-channel min/max candidate. Both the generic
assistant and the QAT assistant have all 23 expected shapes, but each produces
0/23 exact packed buffers and 0/23 exact scale vectors for the supplied MTP
artifact. This negative result isolates the remaining private/other-source
boundary; it does not invalidate the public assistant for Transformers
speculative decoding.

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mtp_assistant_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  C:\path\to\assistant\model.safetensors `
  --output C:\temp\gemma4-mtp-assistant-parity.json
```

The mobile-transformers checkpoint leaves the E2B per-layer model projection
as BF16 and omits its W8 scale. Test the public source against the official W8
projection (without downloading the full remote file) with:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mobile_projection_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --source https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main/model.safetensors `
  --output C:\temp\gemma4-mobile-projection-parity.json
```

The current public BF16 projection fails every tested symmetric W8 scale/code
candidate; keep that result separate from the 276/276 exact packed constants.
The same audit checks BF16 rounding-cell feasibility: the dated official
section had 8,960/8,960 scales and 13,762,560/13,762,560 codes compatible with
some hidden pre-BF16 values. This supports (but does not prove) max-abs/127
before BF16 serialization and cannot recover the missing source precision. The
JSON field is `bfloat16_interval_report`.
The compressed-tensors mobile checkpoint is a schema/packing cross-check, not
an exact LiteRT-LM source because its scales are BF16 and the projection is
ignored.

The public exporter boundary is also explicit: the LiteRT-Torch maintainer
states in [issue #1044](https://github.com/google-ai-edge/litert-torch/issues/1044)
that QAT checkpoint conversion is unsupported, and merged
[PR #1050](https://github.com/google-ai-edge/litert-torch/pull/1050) makes that
failure explicit. Do not treat `export_hf` as Google's packed mobile-QAT
importer or exact MTP/audio package exporter.

For the official 270M package, compare selected source tensors without writing
the gated 536 MB checkpoint locally:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/audit_litertlm_remote_source_recipe.py `
  C:\path\to\official-gemma3-270m-it-q8.litertlm `
  --source-url https://huggingface.co/<verified-public-mirror>/resolve/main/model.safetensors `
  --model-type TF_LITE_PREFILL_DECODE `
  --max-weights 10
```

When the source checkpoint is already available, replace `--source-url` with
`--source-path C:\path\to\model.safetensors`; the audit memory-maps the file
and does not copy it.

The range audit uses the canonical seven-linear Gemma 3 layer order, checks
axis-0 INT8 bytes and scales, and reports whether source-precision/rounding
differences remain. The full official 270M section comparison matched all 127
unique weights' scales to `max(abs(BF16 row))/127`; 31,142 of 268,042,240
serialized INT8 values differed (0.0116183%), with a maximum error of one and
all differences at BF16 half-step boundaries. This proves the public scale
construction for the compared mirror bytes, not Google source provenance or
its private QAT schedule.

For Slurm machines, submit the repo-owned sbatch entrypoint instead of writing an
ad-hoc script:

```bash
sbatch training/scripts/slurm_train_gemma4_e2b.sbatch
```

The sbatch file runs the training command through `srun`, activates
`/home/k_anup/gemma4_env` by default, and requests four GPUs. The training
entrypoint uses all queryable GPUs visible to the process by default. The
training runner also downgrades `bfloat16` to `float16` automatically when the
visible GPU/PyTorch setup does not support BF16. By default, SFT training fails
fast when PyTorch cannot see CUDA, because otherwise the job silently runs on CPU
and appears stuck even when `nvidia-smi` shows idle GPUs. Override paths without
editing the file:

```bash
sbatch --export=ALL,A2UI_REPO_DIR=/home/k_anup/code/GenUI,A2UI_VENV=/home/k_anup/gemma4_env training/scripts/slurm_train_gemma4_e2b.sbatch
```

If a Slurm job fails, inspect both Slurm state and the training log:

```bash
sacct -j <jobid> -o JobID,JobName%30,State,ExitCode,DerivedExitCode,Elapsed,Timelimit,MaxRSS,ReqMem,NodeList,Reason -P
scontrol show job -dd <jobid>
tail -200 training/logs/slurm-<jobid>.err
tail -200 training/logs/slurm-<jobid>.out
```

If `nvidia-smi` shows GPUs but only Xorg/display processes and `0%` utilization,
verify PyTorch from the same environment/container:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

If that reports `cuda available: False`, fix the environment rather than waiting
for training: install a CUDA-enabled PyTorch build and launch Singularity with
GPU passthrough, for example `singularity exec --nv <image> ...`.

The training entrypoint normalizes `CUDA_VISIBLE_DEVICES` before importing
PyTorch. If the shell inherits a multi-GPU value such as `0,1,2,3`, it keeps all
GPUs that `nvidia-smi --query-gpu=index` can query. To use a specific healthy
set, pass it explicitly:

```bash
A2UI_CUDA_VISIBLE_DEVICES=0,1,3 python3 training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_lora.yaml
```

To exclude a known bad device without hardcoding the full set:

```bash
A2UI_EXCLUDE_CUDA_DEVICES=2 python3 training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_lora.yaml
```

For the HF Trainer backend, labels are completion-only: prompt tokens are masked
with `-100`, completion/IR tokens are left trainable, and the preflight fails if
a batch would have zero trainable labels. This prevents apparent zero-gradient
runs caused by truncating away the IR completion.

Shell and sbatch files are forced to LF line endings through `.gitattributes`.
This avoids Linux shebang failures such as `cannot execute: required file not
found` caused by CRLF files copied from Windows.

3. Evaluate generated IR against the flat-spec contract and existing UI metrics.

```powershell
python training/scripts/evaluate.py --predictions training/outputs/eval/predictions.jsonl
```

4. Evaluate with dataset-compatible overall score.

```powershell
python training/scripts/evaluate.py --predictions training/outputs/eval/predictions.jsonl --weights-config dataset/configs/run.yaml --baseline-aggregate dataset/data/runs/<baseline>/aggregates.json
```

The evaluator writes `aggregate_metrics.json` with `overall_score`, `baseline_overall_score`, and `overall_score_delta_vs_baseline` when a baseline is provided.

5. Export a trained Gemma 4 E2B model for Google AI Edge Gallery.

Google AI Edge Gallery imports local LLMs as `.litertlm` files. After training,
merge the LoRA adapter into a Hugging Face model directory and run Google's
LiteRT Torch Hugging Face exporter:

```powershell
python -m pip install -r training/requirements-edge-export.txt
hf auth login
python training/scripts/export_edge_gallery_model.py --config training/configs/export/edge_gallery_gemma4_e2b.yaml --merge-lora
adb push training/outputs/export/gemma4_e2b_ir_edge_gallery/litertlm/<model>.litertlm /sdcard/Download/
```

For CI or CPU-only machines, validate the generated command/manifest without
running conversion:

```powershell
python training/scripts/export_edge_gallery_model.py --config training/configs/export/edge_gallery_gemma4_e2b.yaml --dry-run
```

The export wrapper writes `edge_gallery_export_plan.json`,
`run_litert_export.ps1`, `EDGE_GALLERY_IMPORT.md`, and `model_manifest.json`
under `training/outputs/export/gemma4_e2b_ir_edge_gallery`.

6. Export a legacy Android metadata package.

```powershell
python training/scripts/export_model.py --config training/configs/export/litertlm_gemma.yaml
python training/scripts/package_android_model.py --export-dir training/outputs/export/gemma_e2b_ir_litertlm
```

## Architecture

The training code is intentionally model-agnostic:

- Dataset preparation writes neutral JSONL records with `messages`, `prompt`, and `completion`.
- Model-specific formatting is isolated in `ir_training.models.ModelAdapter` implementations.
- Export targets are registry-driven so future formats can be added without changing data prep or metrics.

Initial target: Gemma E2B-family LoRA/QLoRA SFT. Future adapters can be added for Qwen, Llama, Phi, or any Hugging Face causal LM.

## Output Policy

Generated training artifacts are written under `training/outputs/`, `training/runs/`, or `training/checkpoints/`. These folders should stay uncommitted unless a small manifest/report is intentionally needed for review.
