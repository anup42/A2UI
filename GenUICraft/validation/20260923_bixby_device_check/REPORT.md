# Bixby GenUICraft device check — 23 Sep 2026

Connected device: `R3GL203AKSF` (`SM-F776U`). Installed Bixby package
`com.samsung.android.bixby.agent` was version `5.0.10.38`, updated on
22 Sep 2026. The local Bixby checkout uses GenUICraft AAR `0.4.2` and
minifies its debug build.

## Initial APK result (installed 22 Sep)

The installed integration does **not** complete an A2UI answer. Bixby showed
`Creating a visual answer…` for a live web-search response, then its process
aborted during LiteRT-LM conversation creation. This happened on two searches
at 09:12:49 and 09:14:15 device time. Android returned to the previously open
GenUICraft demo app, which had made the failure look like an app switch.

The Bixby settings page is present. Its trained model was initially missing.
The exact trained model from the GenUICraft app was SHA-256 verified as
`4675f37353e41c786a3e94f03ba4f64ad366a2e4e5d188bb92f5bef3922a75fe`
and imported through Bixby's model picker. Bixby then reported
`Verified gemma4_e2b_a2ui_mobile.litertlm (2.41 GiB)`. The model remains in
Bixby private storage. The temporary Downloads copy was removed. The visual
answers switch was turned off after reproducing the crash so normal Bixby use
does not repeatedly hit the native abort; MTP remains configured on.

## Native abort

The second tombstone reports:

```text
pid 23713, tid 24720 (GenUICraft-Gemm), SIGABRT
JNI DETECTED ERROR IN APPLICATION: JNI CallIntMethodV called with pending exception
java.lang.NoSuchMethodError: no non-static method
"Lcom/google/ai/edge/litertlm/SamplerConfig;.getTopK()I"
at com.google.ai.edge.litertlm.LiteRtLmJni.nativeCreateConversation(...)
```

LiteRT-LM `0.16.1` exposes `SamplerConfig.getTopK()` in its unminified AAR.
The installed Bixby APK is missing that JNI-required method at runtime. The
GenUICraft `0.4.2` AAR carried keep rules only for benchmark getters, while
Bixby's debug build has minification enabled. This points to R8 shrinking or
renaming the LiteRT-LM JNI API in the host APK.

## Repair prepared

`genuicraft/consumer-rules.pro` now keeps the LiteRT-LM Java API and members
required by native JNI. The same keep rule was added locally to
`C:/Users/anupk/Downloads/Bixby_18Sep/BixbyApp/proguard-rules.pro` so the
existing Bixby AAR version is covered on its next build. Bixby is outside this
Git repository and was not pushed.

The GenUICraft release AAR assembled successfully, and its packaged
`proguard.txt` contains the rule. The initial installed Bixby APK predated the
rule. The local Bixby checkout has no Gradle wrapper or APK output in this
environment; a replacement APK was installed separately and retested below.

## Retest after replacement APK (installed 23 Sep, 10:00)

The package still reports version `5.0.10.38`, with `lastUpdateTime` now
`2026-09-23 10:00:25`. The trained model remained verified. I enabled
**Use generative visual answers** and left **Faster generation** (MTP) on.

`Bengaluru weather forecast` returned Bixby's built-in Weather Channel card,
so it did not exercise the GenUICraft path. `Compare Snapdragon 8 Elite and 8
Gen 3 with sources` reached **Creating a visual answer...** and then displayed
the SDK's native A2UI view with comparison cards and source links. Bixby kept
the same process ID throughout this run. The earlier `SamplerConfig.getTopK()`
JNI abort did not recur.

The runtime logged `Gemma4 backend=GPU; MTP=true` at 10:05:43 and
`GenUICraft conversion succeeded: attempts=1, repair=GENERATED_DSL_REPAIR,
warnings=19` at 10:06:46. This is roughly 62 seconds from the backend log to
the conversion-success log, or about 66 seconds from native library load at
10:05:40. These timestamps include initialization and conversion, so they are
not a model-only inference benchmark.

The result is functionally rendered but not release-quality. The cards use
truncated headings (`Elite`, `Gen`, `Gen3`) and contain visibly corrupted
values such as `Oryononon`, `2400 Hz`, `3.4.0 GHz`, and `Adreno 7050`. Those
exact corrupt strings were absent from the 3,438-character Bixby source answer
captured in the local log; the generated/repair output introduced them. The
native view starts behind Bixby's floating top controls and is capped near
72% of screen height, leaving a large blank area above the input bar. Its
content does scroll to the lower cards and source links. This one live web
answer establishes that the updated APK can run the pipeline, not that answer
quality or layout is ready across the Bixby50 set.
