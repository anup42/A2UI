# Gemma 4 on-device runtime

`Gemma4Provider` runs an external Gemma 4 `.litertlm` package through LiteRT-LM. Model weights are never bundled in the GenUICraft AAR.

## Host setup

The published `com.samsung.genuicraft:genuicraft:0.1.0` POM declares `com.google.ai.edge.litertlm:litertlm-android:0.15.0` and `kotlinx-coroutines-android:1.9.0` as runtime dependencies. Consume the Maven publication rather than copying only `genuicraft-0.1.0.aar`; a bare AAR does not carry transitive dependencies. LiteRT-LM 0.15.0 supplies its JNI bridge for `arm64-v8a` and `x86_64`. The GenUICraft AAR supplies the additional ARM64 LiteRT/OpenCL libraries described below.

Pass an absolute path to a readable, nonempty `.litertlm` file that the host app has permission to access:

```kotlin
val provider = Gemma4Provider(
    Gemma4Config(
        modelPath = File(context.getExternalFilesDir("models"), "gemma-4-E2B-it.litertlm").absolutePath,
    ),
)
```

GPU with MTP speculative decoding and thinking enabled is the default and the required delivery configuration. CPU remains available only as an explicit diagnostic/host choice:

```kotlin
val cpuProvider = Gemma4Provider(
    Gemma4Config(
        modelPath = modelPath,
        accelerator = "CPU",
        cpuThreads = 4,
    ),
)
```

There is no silent GPU-to-CPU fallback. The packaged GPU runtime supports `arm64-v8a`; GPU initialization on any other ABI fails with an actionable error. Successful output identifies the actual runtime as `LiteRT-LM/Gemma4/GPU`, `LiteRT-LM/Gemma4/GPU+MTP`, or `LiteRT-LM/Gemma4/CPU`.

## Lifecycle and limits

The provider initializes its engine on the first request and reuses it. It serializes conversations because LiteRT-LM engine calls must not overlap. Cancellation while waiting for the provider mutex starts no native task. After submission, cancellation marks the request stopped and cancels/interrupts its worker task; an existing conversation also receives `Conversation.cancelProcess()`. During engine initialization no conversation exists, and native initialization may finish before cleanup can proceed. `close()` is idempotent and safe to call from `Activity.onDestroy`: it marks the provider closed, requests cancellation of any active conversation, queues engine cleanup behind in-flight initialization or generation, returns without blocking the main thread, and rejects later calls. Use suspend `closeAndAwait()` before constructing a replacement provider when the old native engine must be fully released first; it performs the same nonblocking close and awaits worker termination on `Dispatchers.IO`.

`maxContextTokens` sizes the LiteRT-LM engine. `maxOutputTokens` is the provider ceiling; each `GenUiPrompt.maxOutputTokens` must fit under it. A conservative prompt estimate plus a chat-template reserve is checked before native generation, while LiteRT-LM remains the authoritative tokenizer and enforces the configured engine window.

The provider passes thinking explicitly: `enableThinking = true` and a formatting budget of `thinkingTokenBudget = 1024`. Reasoning remains enabled; this bound reserves room for the final layout and prevents long repairs from exhausting the shared output budget. A null native thinking optional can resolve to reasoning disabled. Hosts may set `thinkingTokenBudget = -1` for the unlimited runtime default; reasoning and final-answer tokens share `GenUiPrompt.maxOutputTokens`.

`enableSpeculativeDecoding` defaults to `true`. GPU initialization verifies the model package's MTP capability and fails clearly if MTP was requested but is unavailable; it never silently disables MTP. It sets `ExperimentalFlags.enableSpeculativeDecoding` before `Engine.initialize()` and restores the prior process-global flag afterward. This matches the capability/initialization ordering in [Google AI Edge Gallery, pinned commit 4006d61](https://github.com/google-ai-edge/gallery/blob/4006d61ccb68533e974e2a23c10ab63e3205c29d/Android/src/app/src/main/java/com/google/ai/edge/gallery/ui/llmchat/LlmChatModelHelper.kt). The Gallery revision declares LiteRT-LM 0.11.0; this SDK uses 0.15.0 to retain its explicit thinking configuration and validated lifecycle implementation. We did not copy an unpinned Gallery binary or claim the two dependency closures are identical. CPU never enables MTP. Thinking remains enabled by default in both modes.

Direct-copy development probes produced numeric changes on GPU both with and without MTP; this is not evidence that MTP alone caused the issue. The bundled Gemma formatting prompt therefore generates layout with typed source references. The converter inserts original text/list/table values, then compiles and validates the complete ordinary Express document. It rejects missing, duplicated, reordered or misused references. This is model-generated layout with deterministic source values, not a failed-generation text fallback.

## Reproducible ARM64 GPU package

The LiteRT-LM 0.15.0 Maven AAR contains only `liblitertlm_jni.so`; it does not contain the dynamic LiteRT runtime, OpenCL accelerator, or OpenCL Top-K sampler required by the tested GPU path. GenUICraft therefore packages this minimal ARM64 closure:

| Packaged file | Provenance | SHA-256 |
| --- | --- | --- |
| `libLiteRt.so` | `jni/arm64-v8a/libLiteRt.so` from [Google Maven LiteRT 2.2.0](https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/2.2.0/litert-2.2.0.aar) | `97355A36CB8AC7628CF407773291E98DA79F3EF184CC43CB0E57DEDF5F0C0637` |
| `libLiteRtOpenClAccelerator.so` | [LiteRT-LM v0.16.1 tagged ARM64 prebuilt](https://media.githubusercontent.com/media/google-ai-edge/LiteRT-LM/v0.16.1/prebuilt/android_arm64/libLiteRtOpenClAccelerator.so) | `DFDCB6A551DC78A9DAB88883AD6E8E2D9B090ECDA383A5CEA3731C6B9561D244` |
| `libLiteRtTopKOpenClSampler.so` | [LiteRT-LM v0.16.1 tagged ARM64 prebuilt](https://media.githubusercontent.com/media/google-ai-edge/LiteRT-LM/v0.16.1/prebuilt/android_arm64/libLiteRtTopKOpenClSampler.so), with its existing `DT_SONAME` entry changed to `DT_NEEDED libLiteRt.so` | `743371DCC1EE57D4E2D14E6D50075C424F6BA0DF554F6BB94A8B0AFC1E76E253` |

The upstream LiteRT AAR hash is `624518D72F8A249711A19E9901F480E74F823CA7818260A739CB2C023024807C`. The unmodified upstream sampler hash is `4404DC68786460602685CAB62DDFA29035E9CFC38BB4550DEC15ABAAA1302A82`. `tools/prepare_gemma4_gpu_runtime.py` downloads only these pinned official artifacts, verifies every input, applies the single explicit dynamic-table change without rewriting code, sections, Android packed relocations, or hash tables, and verifies every output. Run it from the project root with:

```shell
python tools/prepare_gemma4_gpu_runtime.py
```

The AAR packages the upstream Apache 2.0 licenses and the 1,915,758-byte LiteRT third-party notice under `assets/genuicraft_licenses/`. Their SHA-256 values are `71C6915D04265772A0339BED47276942C678B45CC01534210EBE6984FD1AEC65`, `C71D239DF91726FC519C6EB72D318EC65820627232B2F796219E87DCF35D0AB4`, and `2D4D617FF3047813C1B4BFD66DC3C95B2352B401546C6E58A33B7C885091372A` for the LiteRT license, LiteRT-LM license, and LiteRT third-party notice respectively.

The SDK intentionally omits `libLiteRtClGlAccelerator.so`, which can win the runtime registry ahead of the OpenCL path; `libLiteRtGpuAccelerator.so`, whose audited parent copy requires an absent `libwebgpu_dawn.so`; and the earlier local `libLiteRtRuntimeBuiltin.so`/`libc++_shared.so` shim pair. The patched sampler depends directly on the pinned official `libLiteRt.so`, which exports `kLiteRtRuntimeBuiltin` and all 165 versioned LiteRT symbols referenced by the earlier shim.

### SDK-only device verification

The reference app was built with `-PgenUiSdkOnlyNative=true`, disabling every parent-owned JNI directory. Its APK contains the three ARM64 libraries listed above, the Maven LiteRT-LM JNI bridge, and AndroidX's graphics-path library. ZIP-entry SHA-256 checks match the pinned SDK libraries; the legacy ClGl/GPU accelerators and shim pair are absent.

On the connected SM-F776U, this APK initialized OpenCL and reported `Gemma4 backend=GPU; MTP=true; modelSupportsMtp=true; thinking=true`. Successful conversions identify `LiteRT-LM/Gemma4/GPU+MTP`. This verifies the selected backend, capability check and initialized MTP flag; it does not measure the speculative-token acceptance rate. Corpus quality, latency, artifact hashes and selected screenshot review are recorded separately in [VALIDATION.md](VALIDATION.md).

The repository's optional `working_dir/litertlm-android-0.16.1-gpu-fixed-with-provider-v6.aar` was absent during implementation and is not part of the SDK. Validate GPU generation on each target device and model export; JVM lifecycle tests cannot establish native kernel, model-quality, or device-memory compatibility.
