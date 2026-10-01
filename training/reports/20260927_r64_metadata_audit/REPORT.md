# Rank-64 training and LiteRT metadata audit

27 September 2026. Inputs: the three JSON files in
`C:\Users\anupk\Downloads\r64_lora_litert_metadata`, the previously supplied
LiteRT-LM binary and prediction files, and the source at the recorded training
commit `5dc5e19c6ca0c1b44702a6df280617239ec353cb`.

**Finding: there is a reproducible QAT-versus-merge rounding mismatch, and the
existing export gate does not detect it.** The supplied metadata consistently
identifies the selected rank-64 adapter and exported binary. It does not establish
native inference parity. The mismatch is a concrete code issue, but its frequency
in the actual weights and its contribution to the phone's severe output failure
remain unmeasured; the adapter and base weights were not supplied here.

This audit includes fresh file hashing, metadata cross-checks, telemetry analysis,
source comparison against the recorded commit, and a small CPU reproduction. It
does not include a new model inference, training run, export, or device benchmark.

## 1. Confirmed numerical mismatch

The supplied export telemetry records BF16 base/merged weights and FP32 adapter
weights for all 205 adapted projections. The QAT helper
[`fake_quant.py:1299`](../../src/ir_training/qat/fake_quant.py#L1299) casts the LoRA
delta to the base dtype **before** addition:

```python
delta = module.get_delta_weight(adapter)
delta = delta.to(device=base_weight.device, dtype=base_weight.dtype)
effective_weight = effective_weight + delta
```

The export verifier reconstructs ordinary PEFT merge in
[`build_gemma4_retained_scale_litertlm.py:1383`](../../scripts/build_gemma4_retained_scale_litertlm.py#L1383):

```python
delta = (b_compute @ a_compute) * scaling  # FP32 for this run's adapter dtype
reconstructed = base_chunk.clone()        # BF16 base
reconstructed.add_(delta)                 # Add FP32 delta, then store as BF16
```

These have different rounding boundaries. Using the real QAT helper, quantizer,
and export-verifier functions with a synthetic base weight of `1.0`, FP32 delta
`-0.498`, and retained scale `1.0` gives:

| Measurement | QAT helper | PEFT-style merge/export |
|---|---:|---:|
| Effective weight | 0.50000000 | 0.50390625 |
| W2 quantized code | 0 | 1 |
| W4 quantized code | 0 | 1 |
| Four packed W2 codes | `00` | `55` |
| Export reconstruction gate `numerical_exact` | Not compared to QAT | `true` |
| Export reconstruction gate `retained_scale_code_exact` | Not compared to QAT | `true` |

This is a counterexample to treating the existing gate as QAT-forward equivalence.
The gate compares the merged checkpoint with a reconstruction of that same merge
rule. Both can agree while differing from the QAT forward used in training or
checkpoint evaluation. The reproduction bypasses adapter matmul differences so
it isolates the addition/casting problem; CUDA autocast and actual matmul behavior
are additional things to compare on the training machine.

Both relevant functions have identical ASTs at the recorded training commit and
the inspected checkout. `git blame` attributes the QAT pre-addition cast to
`c5da8df5a`, 8 August 2026. **This is not evidence of a change introduced in the
last ten days.** It is an existing inconsistency newly identified by this audit.

Recommended correction: define one explicit effective-weight arithmetic contract
for QAT, evaluation, and export, including adapter matmul dtype/autocast, delta
scaling, and when the BF16 cast occurs. Add a gate that compares QAT effective
weights/codes directly against exported weights/codes. For this already-trained
adapter, first measure both arithmetic choices on the exact weights and run a
small native inference comparison; do not assume either a silent export change
or retraining is automatically the right remedy.

## 2. What the supplied files establish

All **14 independently recomputed binding/consistency checks pass**. The supplied
export report also records **26/26 export gates passing**; the latter are reported
upstream checks, not a rerun of the full export on this PC.

| Item | Verified result |
|---|---|
| Run | `e2b_v10_official_mobile_qat_lora_runD_r64_6144_8gpu_20260926_v1` |
| LoRA | Rank 64, alpha 64, scaling 1.0, dropout 0 |
| Trainable scope | 205 adapted projections / 410 A/B tensors |
| Selected checkpoint | Step 9,000; epoch 3.1612365 of configured 4 epochs |
| Selection metric | Unique-source Golden32 v5.4 reward 42.753319601547474 |
| Supplied Golden32 predictions | Recomputed reward exactly matches 42.753319601547474; 32 records / 31 unique sources |
| Training metadata hash | Matches both merge and export reports |
| Merge metadata hash | Matches export report |
| Adapter hashes | Agree across training, merge, and export receipts |
| Config, seed, qparam and scale hashes | Agree across receipts |
| Local LiteRT-LM file | SHA-256 matches export report exactly |
| Local target section | SHA-256 matches export report exactly |

Local LiteRT-LM SHA-256:
`c1794ee273ac13f9ddc3c24e1037e17367aabc6a810a3de3c836e1fa79efbf44`.

The report records exact BF16 merged-weight reconstruction and retained-scale
code equality for 205/205 projections, preserved graph topology and quantization
parameters, 72 unchanged frozen weight buffers, and an unchanged official MTP
section. These findings substantially reduce the likelihood of an accidental
wrong-adapter merge, rank/alpha mismatch, damaged file copy, or structural
repacking error. They do not eliminate a systematic numeric/semantic mismatch.

The separate resolved YAML is absent, but these metadata files contain the
relevant training settings and matching config hashes, allowing this audit to
proceed. The adapter and merged-weight hashes are receipts: the actual tensor
files were not supplied for an independent full-weight recomputation.

## 3. Preflight passed under a stability policy, not a native parity policy

The training receipt uses `retained_mobile_safety_v1` with
`cross_mode_comparison: diagnostic`:

| Preflight measurement | Result |
|---|---:|
| Zero-adapter QAT-off completion loss | 1.3205786 |
| Zero-adapter QAT-on completion loss | 0.6237951 |
| Teacher-forced top-1 agreement | 190/256 = 74.21875% |
| Recorded diagnostic minimum top-1 agreement | 90% |
| Greedy common prefix | 1 token |
| Recorded diagnostic minimum common prefix | 8 tokens |
| Greedy test size | 1 prompt, 32 tokens; QAT-on repeated twice |

The `passed: true` flag is consistent with the implemented policy: finite logits,
bounded absolute QAT loss, valid probes, and repeatable QAT generation are mandatory;
BF16/QAT agreement is diagnostic. It would be incorrect to call this a secretly
failed 90% gate. QAT-on loss is actually lower, so the cross-mode disagreement
alone also does not demonstrate that the QAT simulation is broken.

The material limitation is that this test does not compare QAT against native
LiteRT inference, and it only probes initialization rather than the final trained
checkpoint. The metadata explicitly states:

```json
"native_runtime_numeric_parity_verified": false,
"native_kv_cache_simulated": false
```

The local A8 SRQ implementation follows the pinned
[published Transformers `apply_srq` forward](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/integrations/gemma_quant.py#L21).
That specifies a framework quantized-layer calculation, not proof that every
native GPU kernel and KV-cache operation computes identically. The package's
`prefer_activation_type=fp16` is a preference in its metadata; it is not evidence
by itself that the device ignores A8 quantization or uses a particular kernel.

The prediction JSONLs do not carry a per-row adapter hash or `qat_applied` flag.
The official pipeline explicitly requests QAT-on evaluation and checks its result
receipt, and the supplied Golden32 score agrees exactly with the selected score.
That is strong consistency evidence, but the evaluation result/manifest is still
needed to independently bind the supplied predictions to the exact evaluation mode.

## 4. Quantization and training observations

**This is mixed W2/W4 retained-mobile quantization, not uniform W4.** Among the
205 adapted projections, 60 are W2 and 145 are W4. Because the W2 MLP matrices are
large, they contain 1,132,462,080 of 1,835,532,288 adapted weight values: **61.70%**.
The remaining 72 frozen target tensors include one W2 and 71 W8 tensors. QAT was
configured for these same per-module bits; the exporter did not unexpectedly
switch a uniform-W4 training run to W2. The older W4 package is therefore not a
same-quantization control for the new official mobile lane.

Recorded export clipping totals 763,307 / 1,835,532,288 values = **0.041585%**.
The highest per-projection fraction is **0.174697%** in
`model.layers.31.mlp.down_proj.weight`. This does not show a broad clipping
catastrophe. Low clipping also does not measure quantization error inside the
range or guarantee that a small set of important weights is harmless.

All 205 LoRA deltas are nonzero; 203 projections change at least one packed code.
The two unchanged-code projections are layers 1 and 3 `self_attn.v_proj`.
Rounding a small update back to the original codes is possible; these two entries
alone do not demonstrate that an adapter was skipped. The largest reported delta
L2/base L2 ratio is 19.90%; this is a magnitude observation, not a defect threshold.

Activation saturation telemetry supplied in the metadata is **initialization-only**:
one step-0 window, 0.066553% total sampled saturation. The final
`trainer/saturation_telemetry.json` is referenced but absent. The run keeps the
official activation scales fixed, so late-training activation saturation is still
worth checking; the initialization statistic cannot answer it.

The run used a **6,144-token training sequence**, with `overflow_policy: error`.
Its token-cache receipt checked all 91,094 train and 1,862 validation rows; the
longest full training sequence is 6,141 tokens. These files do not support blaming
this run on silent 4,096-token training truncation. Maximum recorded training
completion length is 2,327 tokens while evaluation/device output is capped at
2,048, so some long-target coverage remains a separate issue. The receipt only
provides maxima, not how many rows exceed 2,048. This cap does not explain why
BXP-001's successful checkpoint output takes only 312 tokens but the phone loops
to 2,048 under the same cap.

Two chat-template hashes differ between dataset-binding metadata and generation
diagnostics. The source uses JSON-value hashing for the former and raw-string
hashing for the latter, so the unequal strings are not themselves evidence of a
different prompt/template.

## 5. Implication for the observed phone failure

The previous controlled device evidence remains: BXP-001 produced a valid
312-token supplied CUDA checkpoint prediction but a malformed 2,048-token phone
output. Disabling MTP still failed. See the
[raw comparison and device report](../../../GenUICraft/validation/20260927_r64_comparison/REPORT.md).

The new files establish the binary's export lineage much more strongly than the
previous audit could. They **do not justify treating the severe degradation as an
ordinary, acceptable quantization difference**. They identify an existing
arithmetic inconsistency and missing native inference validation that warrant a
small, controlled parity investigation.

Recommended next checks, in order:

1. On the training machine, use the exact selected adapter plus base seed/qparams
   to count W2/W4 code differences between QAT's effective-weight calculation and
   the exported merged weights. Record per-layer counts and magnitude. Control
   FP32/BF16 matmul and autocast explicitly.
2. On BXP-001 and BXP-003, compare the exact same input token IDs through the
   QAT-on checkpoint and native LiteRT CPU/GPU with MTP disabled. Use
   teacher-forced logits/next-token ranks at a shared prefix to locate the first
   difference; divergent free-running continuations alone do not locate it.
3. Compare the unmodified official package with its reconstructed zero-adapter
   QAT seed. Byte-exact no-op export is already reported; **inference** agreement
   at this stage separates pre-existing simulation/runtime differences from
   training/merge effects.
4. Inspect the final activation saturation report and evaluation receipt before
   deciding whether to change arithmetic, export, the runtime, or training.

For additional small evidence, the useful run artifacts are
`evaluations/best_bixby50/evaluation_result.json`,
`stage_receipts/best_bixby50.json`, the referenced
`training/<run-id>/trainer/saturation_telemetry.json`, and any native
CPU/GPU parity report. To quantify the confirmed rounding risk directly requires
`best_golden_checkpoint/adapter_model.safetensors`, `adapter_config.json`, and
the exact base seed/qparams (or running the comparison where they already exist).

## Reproduction and artifacts

- [Audit script](audit_metadata.py): hashes inputs, checks bindings, compares
  source functions to the recorded commit, aggregates 205 projections, recomputes
  the Golden32 selection score, and demonstrates the W2/W4 counterexample.
- [Machine-readable audit](audit_summary.json): input hashes, all checks, numerical
  evidence, and synthetic reproduction results.
- [All 205 projection statistics](projection_telemetry.csv).

Run `python training/reports/20260927_r64_metadata_audit/audit_metadata.py` from
the repository. Validation completed successfully: 14 binding checks, source
function identity at the recorded commit, exact selected Golden32 score, and
both W2/W4 synthetic counterexamples. The reproduction used PyTorch 2.13.0 CPU;
the training receipt records PyTorch 2.11.0+cu128 and Transformers 5.16.1.
No production inference, QAT, or export source was changed in this audit.
