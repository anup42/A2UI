# New mobile LiteRT-LM: Bixby50 pilot on Flip8

21 September 2026. **The newer package runs on GPU, including with MTP, but none
of the sampled outputs is usable A2UI Express.** Three distinct Bixby50 cases
were generated with MTP off; BXP-003 was repeated once with MTP on. This is four
model calls, not a full Bixby50 evaluation. No prompt tuning, output repair, or
fallback generation was performed.

| Case | Mode | Input tokens | Output tokens | Native decode, tokens/sec | Generation latency | Strict Express / AAR compile |
|---|---|---:|---:|---:|---:|---|
| BXP-001: weather | GPU, MTP off | 3,174 | 213 | **41.17** | 7.368 s | Fail / fail |
| BXP-003: trains | GPU, MTP off | 3,394 | 2,048 | **18.55** | 112.015 s | Fail / fail |
| BXP-009: accommodation | GPU, MTP off | 3,307 | 840 | **14.90** | 59.815 s | Fail / fail |
| BXP-003: trains | GPU, MTP on | 3,394 | 2,051 | **31.32** | 67.015 s | Fail / fail |

GPU-only median native decode: **18.55 tokens/sec**, range **14.90–41.17**;
token-weighted rate: **18.03 tokens/sec**. Native throughput excludes engine
initialization and prefill. Generation latency includes conversation setup,
prefill, decode and conversation cleanup, but excludes engine initialization
and rendering. Engine initialization took 6.437 s with MTP off and 12.654 s
with MTP on.

The MTP repeat was about 1.69 times the decode rate of the same case without
MTP. This is an observed single-run difference, not an established speedup:
the GPU-only case was second in a shared-engine batch, while the MTP case used
a fresh engine. There were no repeated warm trials or controlled thermal
comparison, and the generated outputs differed.

Both BXP-003 calls reached the requested 2,048-token output limit. With MTP,
the native counter reported 2,051 output tokens. That observed count and all
returned text are preserved; they were not clipped to make the count match
the request. The two other calls ended through native EOS/stop handling.

## GPU and MTP evidence

Runtime logs confirm complete `LITERT_CL` delegation in one partition for
each language-model graph: decode **2,068/2,068** nodes, prefill_1024 and
prefill_128 **1,107/1,107** each, and verify **2,243/2,243**. MTP additionally
delegated the drafter **198/198** nodes and logged `MTP Drafter - Success
rate: 0.76`. MTP was therefore active, rather than merely requested.

See [GPU evidence](gpu/gpu_evidence.json), [GPU log excerpt](gpu/gpu_evidence.txt),
[MTP evidence](gpu_mtp/gpu_evidence.json), and
[MTP log excerpt](gpu_mtp/gpu_evidence.txt). These excerpts are restricted to
the process identified by the exact run-label marker; unrelated device logs
are excluded. A nonfatal optional `c++_shared` preload warning occurred in
both runs. Initialization, full GPU graph delegation, and generation still
completed; no CPU fallback or runtime timeout was observed.

## Output quality and rendering

Strict validity is **0/3** for the primary run and **0/1** for the MTP repeat.
The current training evaluator's v5.4 generation reward is **0** on each raw
output. These are source-only holdout cases, so no reference-output exact
match score is claimed.

- **BXP-001:** unclosed expressions and malformed component definitions.
  The table references `/forecast_data` without emitting the forecast rows.
- **BXP-003, both modes:** an unterminated root expression degenerates into
  repeated `~1` tokens until the output limit. The text also corrupts source
  facts: `12007` becomes `120007`, `SBC` becomes `BC`, and Wodeyar's `15:15`
  becomes `15:5` without MTP or `15:55` with MTP.
- **BXP-009:** malformed and repeated state assignments, no complete root
  component, and an invalid closing tag `</a2uiui>`.

The same saved bytes were then passed to the published GenUICraft AAR
compiler on the device. All four were rejected before rendering. Captured
source hashes match the inference outputs, and the replay made zero model
calls. There are **no successful model-output screenshots**. Files named
`failure.png` show the test-app state at compiler failure; they must not be
presented as generated UI.

See [GPU replay results](gpu/renderer_replay/replay_results.json) and
[MTP replay results](gpu_mtp/renderer_replay/replay_results.json).

## Package, prompt and settings

Model supplied locally:
`C:\Users\anupk\Downloads\asset_238\gemma4_e2b_a2ui_mobile.litertlm`

- Size: **2,588,147,712 bytes**.
- SHA-256: `4675f37353e41c786a3e94f03ba4f64ad366a2e4e5d188bb92f5bef3922a75fe`.
  Host and staged device hashes matched before inference.
- Device: **SM-F776U / Flip8**, serial `R3GL203AKSF`, SoC `SM8850`, Android 17
  / API 37. USB powered at 100%; observed battery temperature 32.3–33.8 C.
  Power and thermal settings were not changed.
- Runtime: LiteRT-LM Android **0.16.1**; renderer: published GenUICraft
  **0.2.0** AAR. Installed app bytes matched the local APK. APK and native
  library identities are in [environment.json](environment.json).
- Temperature **0**, top-k **1**, top-p **1**, seed **42**; `enable_thinking=false`
  as bound by the current training workflow. Input budget **4,096**, output
  budget **2,048**; native engine allocation **6,144** within the package's
  **8,192** context. Case timeout **300 seconds**.
- One GPU engine for the three primary cases, with a fresh conversation per
  case. A separate GPU engine and cache directory for the one MTP repeat.

The prompt uses the current shared training contract
`005ef700eee29a4429c895ac6c86580003c43d0ad35e67cbb92a9d4cf0e8c1d2`:
the complete mobile A2UI Express system prompt, one travel-checklist worked
example, then `Create A2UI Express v1 GenUI IR for this response:` followed by
the original response text. It does not use the official model's `@source`
binding preprocessing. The embedded Gemma 4 template is used without an
override. Runtime-rendered prompt bytes matched the independently rendered
expected prompt before all four calls; native input counts also matched
offline control-token-aware counts. Native token-ID arrays are unavailable.

The [package/prompt audit](prompt_audit/model_audit.json) confirms a
44,325,712-byte MTP drafter is present. The current official-mobile export
workflow preserves this drafter unused, so MTP off is the primary setting
and the enabled run is a separate experimental check.

The original training manifest and checkpoint were not supplied with the
file. Package metadata, embedded tokenizer/template and MTP presence are
verified; the A2UI scaffold is reconstructed from the current repository
workflow. This does not prove original training lineage or numerical parity
between the training checkpoint and this export. The all-50 prompt audit
performed string/token-budget checks only, not inference on all 50 cases.

## Reproduction and artifacts

Source base: `831f297995e92fedd6e1082c86675c40a5148730`, plus the two committed
instrumentation changes accompanying this report. The test APK was built
with `:app:assembleDebugAndroidTest -PgenUiSdkOnlyNative=true` and replacement
installed. Production app/SDK code and the Bixby repository were not changed
for this benchmark.

Stage [GPU requests](gpu/requests.jsonl) or [MTP requests](gpu_mtp/requests.jsonl)
to the test app's external-files directory. Run the following instrumentation
with a **new label** on each attempt (existing reports are never overwritten):

```powershell
adb -s R3GL203AKSF shell am instrument -w -r `
  -e class com.samsung.genuicraft.OfficialMobileNativeQualityProbeTest `
  -e modelPath /sdcard/Android/data/com.samsung.genuicraft/files/sdk_models/gemma4_e2b_a2ui_mobile.litertlm `
  -e requestPath /sdcard/Android/data/com.samsung.genuicraft/files/sdk_benchmark/mobile_pilot_20260921/requests_gpu.jsonl `
  -e label YOUR_NEW_GPU_LABEL -e caseTimeoutSeconds 300 `
  com.samsung.genuicraft.test/androidx.test.runner.AndroidJUnitRunner
```

For the one-case MTP repeat, use `requests_gpu_mtp.jsonl`, a different new
label, and `-e mtp true`. Results are appended per case under internal app
files `official_mobile_native_quality_reports/<label>.jsonl`, with an adjacent
manifest. Retrieve with `adb exec-out run-as com.samsung.genuicraft cat ...`.
Native test success means inference completed; it does not mean output
validity passed.

Score unmodified `raw_generated_text` using
`ir_training.eval.metrics.score_prediction(response_text, expected=None,
generated_text=raw_generated_text, metric_version="v5_4")` from `training/src`.
For device compiler replay, stage each exact output as
`sdk_benchmark/<sourceRunId>/<caseId>/output.express` and invoke
`GenUiSdkBixby50Test#replaySavedBixbyCorpus` with `replayMode=express`, the
source run ID, a new output `runId`, and the explicit selected case IDs.
The replay assertions failed as expected for these invalid outputs.

Machine-readable results: [summary](summary.json), [per-case CSV](per_case.csv),
[GPU raw results](gpu/native_results.jsonl), [MTP raw results](gpu_mtp/native_results.jsonl),
[GPU scored results](gpu/scored_results.json), and [MTP scored results](gpu_mtp/scored_results.json).
Each case directory also contains the frozen source, exact output, and full
metric record. [checksums.json](checksums.json) binds the saved artifacts.

This package has demonstrated GPU/MTP execution and the measured throughput,
but has not demonstrated acceptable conversion quality. Checkpoint-versus-
export parity on a training-validation fixture is the next diagnostic step;
these few Bixby50 cases should not be used to tune prompts or train the model.
