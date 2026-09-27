# Rank-64 QAT-compatible LiteRT-LM device check — 28 September 2026

**Result:** the new package changes generation and improves BXP-001's looping
behavior, but it is not yet a reliable response-to-UI model. On the connected
Galaxy Z Flip8, all five matched Bixby cases required generated-DSL repair,
none was valid as raw A2UI Express, and all five had source-fidelity warnings.
The Android harness reports five rendered screens, but BXP-030's captured
preview is blank and the other four include altered or missing facts. Do not
promote this file to Bixby's default based on this sample.

## Model and test contract

- Device: `R3GL203AKSF`, Samsung `SM-F776U`, Android SDK 37.
- Supplied filename: `gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm`.
- Device SHA-256: `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`.
  The source file at `/sdcard/` and app-readable copy under
  `/sdcard/Android/data/com.samsung.genuicraft/files/sdk_models/` have the
  same hash and size, 2,588,147,712 bytes.
- Previous rank-64 package SHA-256:
  `c1794ee273ac13f9ddc3c24e1037e17367aabc6a810a3de3c836e1fa79efbf44`.
  It remains under its original, distinct filename.
- Same Bixby50 source bytes, prompt contract, five case IDs, device build,
  2,048-output-token limit, and GPU+MTP settings as the earlier rank-64
  matched run of 27 September. All five
  rendered prompt hashes match exactly. Source-text fallback was disabled;
  generated-DSL repair was enabled; source integrity was evaluated as a warning
  so that damaged outputs could be inspected.
- Runtime log confirms `Gemma4 backend=GPU; MTP=true; MTPRequested=true;
  modelSupportsMtp=true`. The app recorded `LiteRT-LM/Gemma4/GPU+MTP` on
  every case. [Full on-device artifacts](device/flip8_r64_qat_mtp_on_20260928_r1/),
  including raw `output.express`, repaired `a2ui.json`, per-case `result.json`,
  screens, and the [independent scored report](device/flip8_r64_qat_mtp_on_20260928_r1/REPORT.md).

The filename and differing hash establish that the new file was tested, not
that its export gates passed. The export JSON receipt was not supplied with
the phone file, so package-to-checkpoint identity and the claimed QAT
arithmetic remain unverified here.

## Five matched GPU+MTP cases

| Case | Previous rank-64: seconds / output tokens | New: seconds / output tokens | New visual and content result |
|---|---:|---:|---|
| BXP-001 weather | 66.35 / 2,048 | **16.32 / 359** | Long repeated output is gone, but raw DSL is invalid. The [screen](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-001/screen.png) shows `31° / 2°` where the source says `31° / 20°`; it changes other lows and a rain value and drops citations. |
| BXP-003 trains | 32.20 / 543 | **21.33 / 587** | [Train cards](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-003/screen.png) render, but source train `12007` becomes `120007` and `10:35–10:45` becomes `1:35–0:5`; other trains/times also change. |
| BXP-030 EPF/PPF/NPS | 82.94 / 2,051 | **58.49 / 2,048** | Output repeats malformed `_f_f_f...` text to the cap. Repair creates a tiny one-row table shell, but both the [initial](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-030/screen.png) and [scrolled](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-030/screen_scrolled.png) previews are blank. The prior package did not get a render contract success either. |
| BXP-032 insurance | 91.87 / 2,050 | 93.16 / 2,050 | [Table](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-032/screen.png) renders partially but changes `10%` to `0%` and `36 months` to `366 months`, with missing source rows and citations. |
| BXP-037 school boards | 105.28 / 2,050 | **95.76 / 2,048** | Some [comparison cards](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-037/screen.png) appear, but content is incomplete, duplicates `NEET`, and loses citations/details. |

The harness reports **5/5 post-repair render valid** versus **4/5** for the
previous rank-64 export. Manual screenshot review narrows that to **4/5 with
visible content**, and none of those four preserves all source details. Raw
strict validity is **0/5** for both packages; Python's independent strict
scorer also reports **0/5**. The Android `strictValid` value in its summary is
post-repair, not raw validity. All five used
`repairKind=GENERATED_DSL_REPAIR`, and all five have positive
`sourceFidelityWarnings`.

Total conversion time fell from **378.63 s to 285.06 s** for the five cases
(24.7% less), largely alongside fewer generated tokens, **8,742 to 7,092**.
The token-weighted native decode rate was **24.76 to 27.00 tokens/s**. These
are one sequential run per file; device heat and changing output lengths
prevent attributing the time difference solely to the export arithmetic.
The new file still reached the 2,048-token cap on three of five cases.

## Raw model output before repair

All five files below are the **actual LiteRT-LM text** saved before the
compiler, repair, source-fidelity check, or renderer touched them. This is why
the app's post-repair `strictValid=5/5` must not be read as model validity.

| Case | Full raw output | What is already wrong in raw text |
|---|---|---|
| BXP-001 | [output.express](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-001/output.express) | It starts and ends with `<a2ui>` tags but the parser rejects a positional argument after named arguments at `primaryColumn)`. State already says `low:"2"` instead of `20`, later `low:"2"` instead of `21`, and the last row says `Fri, Sep 1` and `prob:"155%"` instead of Sep 11 / 55%. Repair cannot recover correct values from this text. |
| BXP-003 | [output.express](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-003/output.express) | Closed outer tags, but unclosed/malformed expressions in the table schema and `Button("Continue",...,/"/train_data)`. The state itself changes `12007` to `120007`, `20660` to `206066`, and `10:35–10:45` to `1:35–0:5`. |
| BXP-030 | [output.express](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-030/output.express) | Begins a malformed comparison state (`feature:"Eligibility",EPF","PPF"...`), then repeats `_f_f_f...` to 2,048 output tokens. No closing tag or substantive EPF/PPF/NPS table is generated. |
| BXP-032 | [output.express](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-032/output.express) | Starts a term table but changes `10%` to `0%` and `36 months` to `366 months` before repair. It reaches the cap repeating zeros, with no closing tag. |
| BXP-037 | [output.express](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-037/output.express) | Starts a comparison with duplicate/malformed fields, e.g. repeated `transferability`, duplicated `NEET`, and lost words. It ends in repeated `children_children...` at the cap with no closing tag. |

The independent Python strict check also rejects **all five** raw outputs.
Previous rank-64 BXP-001 repeated nested `Tabs(...)` to the cap; the new
raw BXP-001 instead ends after 359 tokens. This is a real behavioral
improvement, but its new syntactic and numerical errors remain in the model
text before any repair step. The prior BXP-030 also repeated malformed
identifiers at the cap; the new BXP-030 repeats a different malformed pattern.

## MTP disabled control

With the same new package on GPU and MTP disabled, BXP-001 took **10.96 s / 236
tokens** and BXP-003 took **41.68 s / 586 tokens**. Both raw outputs remained
invalid and needed repair. BXP-001's [screen](device/flip8_r64_qat_mtp_off_20260928_r1/BXP-001/screen.png)
shows prose and tips but omits the requested temperature/rain table. BXP-003
still corrupts the train number and time. The runtime log confirms
`Gemma4 backend=GPU; MTP=false; MTPRequested=false; modelSupportsMtp=true`.
See the [MTP-off artifacts](device/flip8_r64_qat_mtp_off_20260928_r1/) and
[independent scored report](device/flip8_r64_qat_mtp_off_20260928_r1/REPORT.md).
MTP changes both speed and generated text; disabling it does not restore
correct conversion in these two cases.

## Interpretation

The previously supplied rank-64 checkpoint's BXP-001 prediction was a
312-token, parseable forecast with correct listed temperatures. The new
LiteRT-LM file's corresponding output is shorter than the old LiteRT-LM
loop but remains malformed and corrupts the forecast. This shows a remaining
checkpoint-to-phone behavior gap. The on-device results do not isolate whether
remaining errors come from quantization, runtime behavior, or another export
detail. The next useful parity check is the exact export receipt and a
token-by-token comparison of the checkpoint, reconstructed weights, and phone
on BXP-001/BXP-003.

This is a **five-case device sample**, plus a two-case MTP-off control, not a
Bixby50 pass or an independent audit of the facts in the source Markdown.
No model file was overwritten. The original uploaded file remains at
`/sdcard/`; its verified app-readable copy has a distinct filename.
