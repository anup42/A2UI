# Official E2B QAT LoRA export change audit

**Follow-up, 27 September:** the subsequently supplied rank-64 metadata now
hash-binds the training, merge, and tested LiteRT file. A new audit also reproduces
an existing QAT-versus-merge rounding mismatch that the export gate misses. See
[the metadata follow-up](../20260927_r64_metadata_audit/REPORT.md). Statements below
about unavailable metadata describe the earlier audit's inputs, not the current
evidence set. The mismatch is present at the recorded training commit and its
QAT cast predates the ten-day review window.

Date: 27 September 2026. Scope: local branch `new_ir_changes_20260331`, HEAD
`32614b6473916fd88c5286d2bdfd42e48e1185a4`, commits from 17–27 September.
This is a source and fixture-test audit, not a fresh H100 training run or a
same-checkpoint mobile A/B export.

## Finding

No change to the **numerical retained-scale weight packing or PEFT merge
operation** was found in the last ten days. The recent official-lane exporter
edits add checkpoint selection, numeric-preflight, resume, and package-reader
validation. They can change which run is accepted or exported, but the core
round/clip/pack/patch implementation has not changed. That does **not** prove
the supplied rank-64 package has checkpoint-to-device parity. Its BXP-001 phone
output remains a severe regression relative to the supplied CUDA prediction.

The official retained-mobile lane was introduced on 19 September, so there is
no earlier version of that complete lane to benchmark against. The earlier
v10 W4 LiteRT package is not an equivalent official-lane control.

## Relevant changes

| Date | Commit | Effect |
|---|---|---|
| Sep 19 | `2f2cfc41` | Added the Golden32-selected official retained-mobile QAT LoRA pipeline and extra adapter/merge selection binding. |
| Sep 19 | `2e763708`, `fa20b390` | Added provenance and activation-scale checks; fixed full-package target/MTP section lookup after weight inventory. The latter fixes an inspection failure, not weight codes. |
| Sep 20 | `c3d430f4` | Changed the retained-mobile BF16/QAT preflight policy and its merge/export verification. It does not change the packing arithmetic. |
| Sep 22 | `6b3ff5b1` | Added strict continuation lineage checks to merge and export. |
| Sep 25 | `cc85a1fc` | Added explicit LoRA rank, alpha, seed, and learning-rate experiment options. Fresh defaults remain rank 16 and alpha 16. If a rank-64 run omitted `--lora-alpha`, its saved alpha must be checked; the typical LoRA scale `alpha/rank` would be 0.25, versus 1.0 at rank 16/alpha 16. |
| Sep 26 | `5dc5e19c` | Separated evaluation prompt budget (new default 5,120 input tokens) from training sequence budget (4,096). This can affect evaluation and best-checkpoint selection, not LiteRT weight conversion. |
| Sep 26 | `101b56c0`, `b1a9fe59`, `785e6bdc`, `84cfcf16` | Added and refined optional precomputed semantic augmentation. Default remains no augmentation; an enabled run can train on different data. |
| Sep 27 | `32614b64` | Fixed full-parameter-QAT checkpoint config save, a separate lane. |

The actual `_quantize_projection` and `_quantize_and_patch` functions in
[the retained-scale builder](../../scripts/build_gemma4_retained_scale_litertlm.py)
are attributed by `git blame` to 12 August (`8fa53e425`). The
`PeftModel.from_pretrained(...).merge_and_unload()` call in
[merge_lora.py](../../src/ir_training/export/merge_lora.py) predates this
window. Git blob IDs for the complete builder and merge files are identical
at `6b3ff5b1` and HEAD, respectively `52e4ddad...` and `829340c0...`.
The pinned CPU [deployment export dependencies](../../requirements-deployment-export.txt)
and [mobile export configuration](../../configs/pipelines/gemma4_e2b_mobile_mtp.yaml)
also did not change in the window. The H100 training overlay uses lower
bounds for Transformers/PEFT, so the actual run environment still needs its
saved version receipt.

The [official pipeline](../../src/ir_training/pipeline/official_mobile.py)
sets MTP disabled for export validation, preserves the official drafter
section, and has a separate target-only Android speed gate. An MTP-enabled
demo run is not that gate.

## Supplied package and measured behavior

The supplied `gemma4_e2b_a2ui_mobile.litertlm` hashes to
`c1794ee273ac13f9ddc3c24e1037e17367aabc6a810a3de3c836e1fa79efbf44`.
Its [header inspection](supplied_rank64_package_header.json) finds the
12 LiteRT-LM sections, including `tf_lite_prefill_decode` and
`tf_lite_mtp_drafter`. A separate
[graph inspection](supplied_rank64_package_graph.json) successfully decodes
the target's structural, quantization-layout, and execution-contract
fingerprints, with a complete execution contract. The official base package
was not supplied for a fingerprint comparison. These inspections do not
prove official graph equivalence, checkpoint lineage, or decoding parity.
The supplied folder has only this binary and prediction JSONL files, not the
official pipeline's training config, merge/export report, or manifest.

For BXP-001 the supplied rank-64 CUDA checkpoint prediction emitted a
312-token raw-valid forecast. The phone package emitted 2,048 tokens of
repeated `Tabs(...)` text and failed raw validation. The rendered prompt hash
`ff080451...` and 3,174 input-token count agree; Android did not record
input-token IDs. The rank-64 phone run without MTP also failed on BXP-001,
so MTP is not a sufficient explanation. See the
[three-way raw output comparison](../../../GenUICraft/validation/20260927_r64_comparison/BXP-001_checkpoint_vs_litertlm_raw.html)
and the [matched device report](../../../GenUICraft/validation/20260927_r64_comparison/REPORT.md).

Across the five matched phone cases, rank 64 decoded at 24.76 token/s
token-weighted versus 24.47 token/s for the previous package; that difference
is inconclusive given one sequential run and thermal/order effects. Rank 64
generated 27% more tokens and took 25% longer end to end, while raw validity
remained 0/5 and repaired renderability fell from 5/5 to 4/5. Thus there is
an observed *quality/output-length* regression, not evidence of a native
decode-throughput regression.

## Validation and limit

Focused CPU tests of the retained exporter, official pipeline, SRQ
provenance, and numeric preflight: **259 passed, 1 skipped** (18 long
preparation/configuration cases plus 241 other passing cases). The first
attempts hit Windows temp-directory permissions and path-length limits; the
passing runs used short workspace temp paths, now removed. These tests use
fixtures. They do not train E2B, export the supplied checkpoint, inspect its
complete graph, or prove Android greedy-token parity.

To determine whether this binary regressed at export, obtain the exact
rank-64 run's resolved training config, selected adapter and
`training_metadata.json`, merged checkpoint, retained-scale export report,
deployment runtime manifest, and pinned dependency receipt. Hash-bind them
to the supplied package. Then run BXP-001 greedily with the same input token
IDs through QAT-on checkpoint, merged checkpoint, LiteRT CPU, and Android GPU
with MTP disabled, and record the first divergent generated token. A
same-checkpoint old-versus-current export would isolate code version;
comparing different trained checkpoints cannot.
