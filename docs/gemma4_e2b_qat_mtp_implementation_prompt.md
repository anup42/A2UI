# Implementation Prompt: Gemma 4 E2B QAT-Derived LoRA and MTP

Use the prompt below when another agent needs to recreate, review, or extend this
workflow. The durable technical facts and current file inventory live in
`training/docs/gemma4_e2b_qat_mtp_knowledge.md`; read that file completely before
changing code.

## Copyable prompt

You are working in the A2UI repository. Implement or repair the response-to-flat-
spec training workflow for Gemma 4 E2B with fast INT4 inference and optional MTP
speculative decoding.

Read `AGENTS.md`, `training/README.md`, and
`training/docs/gemma4_e2b_qat_mtp_knowledge.md` completely before editing. Inspect
the working tree and preserve all user and unrelated changes. Do not delete
training outputs, dataset runs, Android data, or checkpoints.

The supported objective is deliberately narrow:

1. Fine-tune only the target model with completion-only LoRA SFT, starting from
   `google/gemma-4-E2B-it-qat-q4_0-unquantized`.
2. Load that base in BF16 where supported and keep `load_in_4bit: false`.
3. Describe this operation as **QAT-derived LoRA**, not QAT. It does not keep fake
   quantization active while updating weights.
4. Keep the exact Google assistant
   `google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant` frozen. Do not implement
   or claim assistant training.
5. Merge the best golden-evaluation adapter into the QAT-derived floating-point
   target, record provenance, and state that the merged result is not packed
   INT4.
6. Convert both target and assistant to the same Q4_0 family only through a
   pinned, verified converter/runtime. Do not describe Q4_0 GGUF as Google's
   mobile wNa8o8 format.
7. Run a target-only versus target+assistant Transformers reference benchmark to
   detect target drift and gross MTP speed changes. Mark it as a floating-point
   reference benchmark, not final packed-INT4 proof.
8. Require a final benchmark in the actual runtime/device before promotion. It
   must measure strict IR accuracy, end-to-end latency, memory/thermal behavior,
   and runtime-provided accepted-draft statistics where available.

Keep these training defaults unless measured evidence justifies a change:

- completion-only SFT;
- BF16 with FP16 fallback only when BF16 is unsupported;
- LoRA rank 8-16, alpha 16-32, dropout 0.05;
- initial learning-rate comparison of 5e-5 and 1e-4;
- PEFT Gemma 4 language-model defaults rather than blanket `all-linear`;
- frozen `embed_tokens` and `lm_head` when no tokenizer tokens are added;
- thinking disabled and compact strict flat-spec JSON as the only completion;
- golden50 selection separated from training data.

Do not conflate any of the following:

- BitsAndBytes `load_in_4bit`/NF4 QLoRA with QAT;
- post-training Q4_0 conversion with QAT;
- BF16 weights from a QAT pipeline with already packed INT4 weights;
- official matched inference checkpoints with a public joint QAT+MTP training
  recipe;
- LiteRT-LM custom export with recreation of Google's mobile QAT schema;
- MTP inference acceleration with an accuracy-improvement technique.

Required implementation properties:

- Expensive commands must be plan-only by default and require an explicit
  `--execute` flag.
- Static validation must not import a model, contact Hugging Face, train, merge,
  quantize, or run inference.
- Merge metadata must record the base checkpoint, adapter path, loader/dtype,
  `continued_qat_performed: false`, `packed_int4_output: false`, and
  `mtp_assistant_trained_or_modified: false`.
- Benchmark output must preserve target-only and MTP predictions separately and
  report greedy token equivalence and full-generation latency.
- If accepted-draft counts are unavailable through the stable API, record that
  gap rather than estimating an acceptance rate.
- Conversion must use the same quantization family for target and assistant,
  refuse accidental overwrite by default, save logs and hashes, and block
  promotion until runtime validation.
- Keep training code model-agnostic outside Gemma-specific adapter/config files.

Validate without training:

```powershell
python training/scripts/validate_qat_mtp_workflow.py
python training/scripts/merge_qat_lora.py
python training/scripts/benchmark_qat_mtp.py
python training/scripts/convert_qat_q4_0.py
python -m pytest training/tests -q
python -m compileall -q training/src training/scripts
```

The first command must return `ok: true`. The next three commands must print
plans and explicitly state that no model was loaded and no expensive operation
was executed. Do not launch `train_sft.py`, model downloads, merging, conversion,
or inference as part of a code-only implementation request.

At handoff, list every file changed, the exact non-training checks run, and any
remaining runtime/tool compatibility gaps. Never claim final INT4/MTP performance
without measurements from the target runtime and device.
