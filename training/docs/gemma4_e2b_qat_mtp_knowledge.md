# Gemma 4 E2B QAT, LoRA, INT4, and MTP Knowledge Base

Last official-source verification: **2026-08-02**

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
> separate unsupported research project.

Keep Google's untouched QAT target plus matching assistant as the speed and
quality control. Do not jointly train the target and assistant with the current
repository.

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

Therefore this repository implements **QAT-derived target tuning**, not QAT or
joint QAT+MTP training.

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

This repository does not currently implement those operations.

### QAT-derived LoRA

The recommended profile begins with BF16 weights produced by Google's QAT
pipeline, freezes those base weights, and updates LoRA parameters. It may retain
some useful QAT-conditioned weight structure, but fake quantization is not active
during adaptation. It is an engineering compromise and must be evaluated after
conversion.

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

## Rules for future agents

1. Re-check official model cards and runtime supported-model lists because Gemma
   4 tooling is changing quickly.
2. Read generated merge/conversion manifests before debugging downstream output.
3. Do not weaken explicit `--execute` guards.
4. Do not add a `train_assistant` implementation by copying a generic MTP loss;
   Gemma 4's released assistant is specialized and target-conditioned.
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
- Gemma 4 technical report:
  https://arxiv.org/abs/2607.02770

When citing speed, say **up to** the reported number and retain the device,
runtime, prompt, batch-size, and precision context. Never turn a release benchmark
into a guarantee for A2UI.
