# Current test-app V10 vs new rank-64 QAT-compatible model

**Relative choice on the matched five-case sample: keep the current trained
E2B V10 model selected in the test app.** The new rank-64 QAT-compatible
package is slightly faster in aggregate, but V10 preserves more of the source
and produces more usable visible output. Neither model is reliable enough for
source-faithful Bixby conversion in these cases.

[Open the side-by-side screenshots and exact raw outputs](V10_vs_QAT_side_by_side.html).
The [V10 baseline files](baseline_v10/) and [new package files](device/flip8_r64_qat_mtp_on_20260928_r1/)
are saved in this folder. Raw `output.express` is separate from repaired
`a2ui.json` for every case.

## What was compared

| | Current test-app trained E2B V10 | New rank-64 QAT-compatible |
|---|---|---|
| Device model file | `e2b_v10_w4.litertlm` in the earlier benchmark; the test app's current `gemma4_e2b_a2ui_mobile.litertlm` has the same verified SHA-256, `4675f37353e41c786a3e94f03ba4f64ad366a2e4e5d188bb92f5bef3922a75fe` | `gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm`, SHA-256 `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62` |
| Raw strict Express valid | **0/5** | **0/5** |
| Rendered after generated-DSL repair, as reported by app | 5/5 | 5/5 |
| Screens with visible content on manual review | **5/5** | **4/5**; BXP-030 preview blank |
| Cases with source-fidelity warnings | 5/5 | 5/5 |
| Total conversion time, five cases | 301.75 s | **285.06 s** |
| Output tokens, five cases | **6,875** | 7,092 |
| Token-weighted native decode speed | 24.47 tokens/s | **27.00 tokens/s** |

Both runs used the same connected `SM-F776U` device build, Bixby50 source
bytes, five case IDs and order, shared prompt, 8,192 context, 2,048 output
cap, temperature 0, GPU+MTP, no source-text fallback, and generated-DSL
repair enabled. All five rendered-prompt SHA-256 values match exactly across
the two runs. These are single runs on different occasions; thermal state is
not controlled. The 16.69-second aggregate difference is only 5.5%, and the
new model is **slower on four of five** individual cases. It saves 35.77 seconds
on BXP-030, the case where its preview is blank.

## Case-by-case output quality

| Case | V10 output | New QAT-compatible output | Relative result |
|---|---|---|---|
| **BXP-001, weather** | [Screen](baseline_v10/BXP-001/screen.png) and [raw](baseline_v10/BXP-001/output.express): the first `31 / 20`, `57%` row is correct. It corrupts Thursday's low to `211` and Friday's date to Sep 1. 10.35 s / 288 tokens. | [Screen](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-001/screen.png) and [raw](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-001/output.express): first low becomes `2`, later lows `2` and `0`, Friday high `3`, and rain `155%`. 16.32 s / 359 tokens. | **V10 clearly better**, though still unsafe. |
| **BXP-003, trains** | [Screen](baseline_v10/BXP-003/screen.png) and [raw](baseline_v10/BXP-003/output.express): train `12007` becomes `120007`; time `10:35–10:45` becomes `10:35–1004`; station `SBC` becomes `BC`. 19.31 s / 438 tokens. | [Screen](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-003/screen.png) and [raw](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-003/output.express): it keeps `SBC` for the first row but changes the time to `1:35–0:5` and other train numbers/times. 21.33 s / 587 tokens. | **Mixed; V10 is closer on the time**, and neither can be trusted as a timetable. |
| **BXP-030, EPF/PPF/NPS** | [Screen](baseline_v10/BXP-030/screen.png) and [raw](baseline_v10/BXP-030/output.express): a very large, isolated EPF recommendation appears, but the requested comparison is missing. 94.26 s / 2,050 tokens. | [Screen](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-030/screen.png) and [raw](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-030/output.express): malformed `_f_f_f...` repetition; repair creates a one-row table shell, but the preview is blank. 58.49 s / 2,048 tokens. | **V10 has some visible content; both fail the task.** |
| **BXP-032, insurance** | [Screen](baseline_v10/BXP-032/screen.png) and [raw](baseline_v10/BXP-032/output.express): five key-term rows plus a corrupted example; it retains the source's `36 months`, though wording and other details are damaged. 85.79 s / 2,049 tokens. | [Screen](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-032/screen.png) and [raw](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-032/output.express): five rows, but changes `10%` to `0%` and `36 months` to `366 months`; the source's examples are missing. 93.16 s / 2,050 tokens. | **V10 better on critical numbers.** |
| **BXP-037, school boards** | [Screen](baseline_v10/BXP-037/screen.png), [scrolled screen](baseline_v10/BXP-037/screen_scrolled.png), and [raw](baseline_v10/BXP-037/output.express): recovered four comparison factors and three recommendation rows, with wording/citation errors. 92.05 s / 2,050 tokens. | [Screen](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-037/screen.png), [scrolled screen](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-037/screen_scrolled.png), and [raw](device/flip8_r64_qat_mtp_on_20260928_r1/BXP-037/output.express): only one factor row survives; extra columns become mislabelled cards such as `flexibility` and `subject_flexibility`. 95.76 s / 2,048 tokens. | **V10 substantially better on coverage and presentation.** |

The main quality problem is already present **before repair**: every raw
`output.express` fails strict parsing. Repair makes both models display
something, but cannot restore missing source facts or safely correct invented
numbers. The scorer's raw generation reward is zero for both five-case runs,
so manual source and screenshot review is necessary for the relative judgement.

The QAT-compatible file's export receipt was subsequently supplied and
[audited](export_audit/REPORT.md): it matches the tested binary, records 28/28
export gates passing, and passes 30/30 independent consistency checks. Native
checkpoint-to-LiteRT inference parity remains unverified. This comparison
covers five of 50 Bixby cases, not the full corpus.
