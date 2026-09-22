# Trained E2B v10 W4 — Android integration and Bixby50 pilot

Evaluation date: 19 September 2026. **Stopped at the user's request after 12
completed cases. Native GPU generation median: 6.38 tokens/sec.** One case
compiled and rendered, but its visible content had duplication and omissions.
This checkpoint has not demonstrated acceptable conversion quality in this
pilot. There is no full-corpus or 50/50 acceptance claim.

## Measured results

| Measure | Observed result |
|---|---:|
| Completed cases | 12: BXP-001 through BXP-012 |
| Returned outputs | 11 |
| Strict valid, both Android and Python | 1 |
| Rendered | 1: BXP-009 |
| Strict-invalid outputs | 10 |
| Five-minute timeouts | 1: BXP-006 |
| Runtime errors in completed cases | 0 |
| Native decode median | **6.38 tokens/sec** |
| Native decode range | 6.35–9.80 tokens/sec |
| Token-weighted overall native decode | 6.51 tokens/sec |
| Input/output tokens across returned outputs | 36,302 / 5,531 |
| Median conversion latency after the first call | 85.219 seconds |
| First conversion, including initialization | 46.323 seconds |

The first output decoded at 9.80 tokens/sec; subsequent returned outputs were
close to 6.4 tokens/sec. Native throughput excludes startup and prefill. The
timeout and interrupted case have no completed native token measurements and
are excluded from these throughput statistics.

The user stopped the run during BXP-013. That case was interrupted; BXP-014
through BXP-050 were not run. The original device files retain the requested
50-case selection and last `running` status because force-stopping instrumentation
does not write a normal completion record. Its terminal `Process crashed`
message is the expected result of that deliberate stop, not evidence of a
spontaneous native crash. See [execution_stop.json](execution_stop.json).

Detailed artifacts: [per-case report](bixby50_gpu/REPORT.md),
[CSV](bixby50_gpu/per_case.csv), [gallery](bixby50_gpu/gallery.html), and
[raw/scored predictions](bixby50_gpu/scored_predictions.jsonl).
All 12 completed source records match the frozen corpus. All 11 returned native
prompt hashes match the independently reconstructed training prefix.

## Delivered app option

The reference app includes **Trained E2B v10 · W4 · GPU** alongside the official
E2B profile. Open **GenUICraft SDK · Bixby50**, select **Gemma 4 E2B**, then select
the trained model. Selection and local model path persist. The metrics switch
controls actual native token counts and decode speed for either E2B profile.

The APK is installed on the connected **Samsung SM-F776U (Flip8), Android 17**.
Only this device was connected for this evaluation. The supplied model is staged
in the app's external-files `sdk_models/e2b_v10_w4.litertlm` directory. Weights
remain outside the APK/AAR. The official E2B downloader and source-binding converter
remain available. Bixby files were not changed for this work.

| Artifact/property | Verified value |
|---|---|
| Supplied model | `C:\Users\anupk\Downloads\e2b_v10_w4.litertlm` |
| Size | 2,859,767,952 bytes |
| Host and device SHA-256 | `64e3944babb7a95e6ece432d316dc26da65bb328c9a1a146de3befbdabe6c751` |
| Quantization | Mixed INT4/INT8 constants; not uniformly 4-bit |
| MTP | Unavailable: no `tf_lite_mtp_drafter` section in the supplied package |
| Runtime | LiteRT-LM 0.16.1, GPU; no CPU fallback |
| Context/output limits | 8,192 / 2,048 tokens |
| Sampling | Temperature 0, top-K 1, top-P 1 |
| Thinking | Disabled, matching the current training workflow |
| Attempts | One per case; no repair or layout fallback |

See [model_audit.json](model_audit.json) for the package, embedded tokenizer,
tensor types, prompt provenance, and source pointers. Build identities and the
312 passing SDK tests plus six passing app helper tests are recorded in
[build_evidence.json](build_evidence.json). The benchmark APK identity is retained
separately in [benchmark_build_evidence.json](benchmark_build_evidence.json).
The final app also includes the reviewed model-switch cancellation fix; its
native runtime, prompt, model settings, and converter are the measured versions.
The SDK is published as `com.samsung.genuicraft:genuicraft:0.2.0` because its
public prompt/output data classes gained fields; rebuild consumers when upgrading.

The final APK SHA-256 is
`27a998660b4cf51504186d5ea2b3740dfdcf93d49265a413d7ac43c2f9f7e529`.
The installed APK and the copy at **Internal storage / Download /
GenUICraft-e2b-v10-w4.apk** match that hash. The final AAR, all native libraries,
and the prompt asset are byte-identical to those used by the W4 benchmark.
The final app's selected model and readiness are visible in
[app_model_option.png](app_model_option.png).

A separate single-case compatibility check of the **official E2B profile** on
the final app passed BXP-003 on the first attempt, with GPU+MTP and thinking
enabled: 1,391 input tokens, 1,095 output tokens, 66.19 native decode tokens/sec,
and 23.973 seconds for the provider call. Its prompt, weights, and reasoning
settings differ from the trained model, so this is not a controlled MTP or
quantization speedup comparison. See [the separate smoke evidence](official_gpu_mtp_smoke/).
No further trained-W4 corpus cases were run after the stop request.

## Training-compatible prompt

The current workflow is `training/scripts/run_golden_deployment.py`, together
with `training/scripts/export_checkpoint_litertlm.py` and their pipeline modules.
Its shared prompt builder provides:

1. The full production system prompt from
   `dataset/prompts/genui_gen_mobile_a2ui_express_v1.md`, replacing its response
   placeholder with `[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]`.
2. This worked user message:

   ```text
   Create A2UI Express v1 GenUI IR for this response:

   A small travel checklist with a title and two items.
   ```

3. This worked assistant response:

   ```text
   <a2ui>
   root=Column([a,b])
   a=Text("Travel Checklist","h1")
   b=List([c,d])
   c=Text("Passport")
   d=Text("Charger")
   </a2ui>
   ```

4. The final user message:

   ```text
   Create A2UI Express v1 GenUI IR for this response:

   <complete captured response, trimmed at its outer boundaries>
   ```

This profile does not replace the response with source-binding blocks and does
not append the original query. The frozen snapshot lives in
`genuicraft/src/main/assets/genuicraft/prompts/e2b_v10_shared_prompt.json`.
`python GenUICraft/tools/sync_trained_prompt.py --check` verifies it against the
current training builder. Its contract SHA-256 is
`005ef700eee29a4429c895ac6c86580003c43d0ad35e67cbb92a9d4cf0e8c1d2`.

The supplied package's Jinja template exactly matches
`training/configs/export/gemma4_e2b_training_minijinja.jinja`. An Android text-part
adapter preserves its exact prompt framing. All 50 reconstructed prompts were
checked for byte and tokenizer-ID parity; the SDK also verifies the native
rendered prefix before every inference and records its hash. Exact prompt
lengths range from 3,031 to 3,759 tokens, so every input and its 2,048-token output
allowance fit inside the configured context.

The original checkpoint's training manifest and system-prompt snapshot were not
provided with the model. This establishes agreement with the current workflow
and embedded template, not independent proof of the original training lineage.

## Android compatibility fixes

Two issues were reproduced and fixed before the measured batch:

- LiteRT-LM 0.15 read prefill dimensions before GPU compilation. This export's
  magic dimension 131 becomes 128, but the old runtime scheduled 130 embedding
  rows into a 128-row buffer. LiteRT-LM 0.16.1 reads the compiled dimensions and
  completes GPU inference. The model file was not modified.
- Android serializes text as typed content parts. The embedded template expected
  plain strings and initially rendered the array representation into the prompt.
  The conversation-scoped adapter unwraps the text and verifies the complete
  training prefix, including BOS and the worked example. BXP-001's input count
  changed from the incorrect 3,389 to the verified 3,174 tokens.

The SDK restores the process-wide template override immediately after creating
the conversation. The official profile has no template override.
See [ON_DEVICE_RUNTIME.md](../../ON_DEVICE_RUNTIME.md) for pinned upstream code
and dependency details. Earlier diagnostic smoke outputs are retained under
[compatibility](compatibility/) and are excluded from the main 50-case results.

## Evaluation method and limits

The run was configured for all 50 frozen captured Perplexity responses and was
stopped after 12 completed cases at the user's request. It uses one generation
per case, a shared GPU engine, and a five-minute timeout per case. No prompt was
tuned on these evaluation responses. Each case retains its source, exact prompt,
raw model output when returned, native metrics, and validation result. Strictly
valid output is compiled to A2UI JSON and rendered through the AAR, with top and
scrolled screenshots. Invalid output is reported without synthetic replacement
content or a fabricated rendered screenshot.

Android and Python strict validation are reported separately. The training
`v5_4` scorer is applied to actual returned outputs with `expected=None`, because
this source-only holdout has no reference target IR. Source-fidelity scores are
mechanical proxies, not a human semantic assessment. Missing or inapplicable
measurements remain unavailable. Successful compilation alone does not establish
faithful content or attractive rendering.

Native decode tokens/sec excludes engine startup and prompt prefill. Conversion
latency includes those costs; only the first call initializes the engine.
Subsequent cases have different content, so their timings are not a controlled
cold-versus-warm comparison. This is a sustained run on a USB-powered phone, not
a thermally controlled benchmark or a like-for-like comparison with the official
model's shorter source-binding prompt and MTP execution.

The training GPU evaluator cancels streaming at the first quote-aware closing
`</a2ui>` and scores the serving portion. The Android provider uses synchronous
native generation until the package's stop condition or output cap and validates
the complete returned text. Its raw-output validity criterion is therefore
stricter when the model emits a suffix. This distinction must be retained when
comparing this report with training-side evaluation scores.

## Manual review observations

These observations are about the actual generated programs and their saved
screenshots, not a repaired version of the model output:

| Case | Observation | Consequence |
|---|---|---|
| BXP-001 | The source's Thursday rain chance `55%` became `5%`; the generated table also has malformed property syntax. | Both factual copying and syntax failed. |
| BXP-005 | Times became strings such as `1:155 PM` and `7:0000 AM`; the output contains unmatched delimiters and repeated text. | The raw answer is unsuitable for display. |
| BXP-009 | The program compiles, but `root` includes a card, its column, and the same text separately, so the screenshot repeats the access-to-sights paragraph three times. The root ends at element `n`; the Vaishali Nagar section in elements `o` through `u` is unreachable. The first heading is shortened from `Civil Lines` to `Civil`. | Rendering succeeds technically, but this is not an acceptable three-neighborhood comparison. |

In BXP-009 the card, type sizes, and paragraph wrapping are legible. The
duplication and omitted section originate in the generated graph: the renderer
displays the references it receives. Its high mechanical content-coverage value
must not be interpreted as complete visible coverage; some generated text is
not reachable from the root. The score's other structural/fidelity checks and
manual review remain necessary.

This evaluation does not isolate whether quality loss comes from the checkpoint,
quantization, export, or native GPU execution. A controlled comparison of the
same checkpoint before and after export, using this exact prompt, would be
needed to attribute the cause.

## Reproduce

Build/publish the AAR, build the reference app with `genUiSdkOnlyNative=true`,
install its app and instrumentation APKs, and stage the checksum-verified model.
Use a new run ID for every run. The original run omitted `cases`; the example
below selects three cases for a small future check:

```powershell
adb -s R3GL203AKSF shell am instrument -w -r `
  -e class com.samsung.genuicraft.GenUiTrainedBixby50Test `
  -e runId w4_bixby_pilot_new_run `
  -e cases BXP-001,BXP-002,BXP-009 `
  -e caseTimeoutMs 300000 `
  com.samsung.genuicraft.test/androidx.test.runner.AndroidJUnitRunner
```

Pull `Android/data/com.samsung.genuicraft/files/sdk_benchmark/<runId>` from the
device, then run:

```powershell
python GenUICraft/tools/report_trained_bixby50.py <pulled-run-directory>
```

The report command validates source and prompt provenance and identifies missing
results. Only use `--require-50` for an intentionally complete corpus run.
A failed final instrumentation assertion means at least one case did not pass;
the completion record distinguishes that from an interrupted evaluation.
