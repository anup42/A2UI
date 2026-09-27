# New rank-64 QAT-compatible export receipt audit

**Subsequent inference experiment:** the same model now produces 2/2 raw-valid
outputs on CPU and 0/2 on GPU with MTP disabled. This narrows the additional
severe corruption to the deployed GPU stack; exact native parity is still not
established. See the [fresh backend comparison](../../20260928_r64_accuracy_gap/REPORT.md).

28 September 2026. The receipt was pulled from the connected `R3GL203AKSF`
(`SM-F776U`) at `/sdcard/gemma4_retained_scale_code_only_report_r64_qat_compatible.json`.

**Finding: the receipt supports that the intended corrected export completed
successfully and identifies the exact binary tested on the phone. No failed
export or package check was found. Native checkpoint-to-LiteRT inference parity
remains unverified, and the observed generation regression remains unresolved.**

This supersedes the earlier statement that the new export receipt was missing.
It does not supersede the five-case raw-output and screenshot findings.

## What was independently checked here

The [audit script](audit_receipt.py) freshly hashed both model copies and the
receipt on the device, hashed the pulled receipt and locally supplied training
and merge metadata, compared identities with the previous export receipt, and
checked consistency across all 205 projection records. **30/30 consistency
checks passed.** These are file-binding and receipt-consistency checks; the
full tensor/export computation was not rerun on this PC.

| Evidence | Result |
|---|---|
| Model in `/sdcard/` | SHA-256 `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62` |
| App-readable model copy | Same SHA-256 |
| Report `output_sha256` | Same SHA-256 |
| Device and pulled receipt | SHA-256 `491ce76a9d00608fea148de92bf4403070af66cc716c0c31f674f66c7955d9e0` |
| Selected adapter | Same hashes as previous receipt; rank 64, alpha 64, scale 1.0 |
| Selection | Golden32 unique-source reward, step 9,000 |
| Seed, qparams, resolved config, original merged checkpoint | Same identities as previous receipt |
| New export mode | `retained_scale_qat_compatible_weights_v1` |
| Weight arithmetic | `qat_bf16_delta_before_add_v1` |
| Weight source | Verified seed plus selected adapter QAT reconstruction |

The adapter tensors and base weights remain on the training host, so their
hashes here are bound receipt claims rather than fresh local tensor hashes.
The locally supplied training and merge metadata files were freshly hashed
and match the new receipt.

## What the exporter reports

**28/28 export gates pass.** These include:

- QAT effective-weight, packed-code and simulated dequantized-weight checks
  for all 205 adapted projections.
- Zero-adapter target identity and unchanged 72 frozen weight buffers.
- Preserved weight and activation quantization parameters, graph layout and
  execution contract.
- An unchanged MTP section, an otherwise unchanged package outside the target,
  and a parseable final package.

For all 205 projections, the recorded QAT reference code hash equals the new
exported packed hash. The recorded original-merge packed hashes also match the
previous export receipt. Thus the report consistently describes the intended
arithmetic change while preserving the same trained adapter.

The correction changed **132,290 quantized scalar weight codes across 193/205
projections**, out of 1,835,532,288 adapted weight values: **0.007207%**. This is
a code-change count, not a token count or a measurement of output quality.
Its small fraction cannot establish whether its effect on generation is small.

Recorded clipping is 763,370 values, **0.041588%**, essentially the same scale
as the previous export. This does not indicate broad clipping failure, but
does not measure all quantization error. The adapted projections remain mixed
W2/W4; this export did not switch a uniform-W4 checkpoint to an unexpected bit
assignment.

## What remains unresolved

The receipt explicitly says `native_inference_parity_verified: false`.
Its QAT reconstruction uses CPU FP32 adapter matmul with autocast disabled,
followed by BF16 delta-before-add. Agreement with the CPU QAT weight helper
does not establish agreement with the actual CUDA checkpoint evaluation,
native activation arithmetic, attention/KV-cache behavior, or GPU decoding.
The supplied training metadata also records `native_kv_cache_simulated: false`.

The original training preflight checks stability, with cross-mode agreement
only diagnostic. Its QAT-on versus QAT-off top-1 agreement was 74.22% and greedy
common prefix one token; those are not checkpoint-versus-LiteRT measurements
and cannot identify the cause of the current device failure.

The [device report](../REPORT.md) still shows 0/5 raw-valid outputs for the new
file, compared with 4/5 in the matching supplied checkpoint predictions. MTP-off
controls for two cases still fail. This receipt removes the missing export
lineage concern; it does not justify calling that remaining gap normal or
acceptable quantization loss, nor prove that a particular GPU/export component
caused it.

The next decisive test is BXP-001 and BXP-003 with identical input token IDs,
greedy decoding and MTP off, comparing the exact QAT checkpoint, reconstructed
quantized weights, LiteRT CPU and LiteRT GPU. Teacher-forced next-token scores
at shared prefixes can locate the first mismatch. Another identical re-export
is not supported by the current evidence as a remedy.

## Artifacts and reproduction

- [Original pulled receipt](device_export_report.json).
- [Independent consistency results and recorded device hashes](audit_summary.json).
- [Audit script](audit_receipt.py).

From the repository root, with the device connected and the previously supplied
metadata folder available:

```powershell
python GenUICraft/validation/20260928_r64_qat_reexport/export_audit/audit_receipt.py --serial R3GL203AKSF --metadata-dir C:/Users/anupk/Downloads/r64_lora_litert_metadata
```
