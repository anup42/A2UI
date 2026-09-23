# Bixby GenUICraft device check — 23 Sep 2026

Connected device: `R3GL203AKSF` (`SM-F776U`). Installed Bixby package
`com.samsung.android.bixby.agent` was version `5.0.10.38`, updated on
22 Sep 2026. The local Bixby checkout uses GenUICraft AAR `0.4.2` and
minifies its debug build.

## Result

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
`proguard.txt` contains the rule. The installed Bixby APK predates the rule,
so on-device conversion and rendering remain unverified until Bixby is rebuilt
and installed. The local Bixby checkout has no Gradle wrapper or APK output in
this environment.
