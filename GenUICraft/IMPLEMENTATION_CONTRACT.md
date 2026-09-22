# GenUICraft implementation contract

Standalone Android library, version 0.5.0, module `:genuicraft`, public package `com.samsung.genuicraft.sdk`.
The independent project lives under A2UI/GenUICraft. Do not build Bixby. Test the actual published AAR in the existing A2UI/android app.

## Ownership

- Renderer agent: Gradle scaffold; extraction/relocation of codec, catalog, canonical graph, validation, complete native renderer and needed resources into library; `GenUiCompiler`, `GenUiContent`, `GenUiView`; renderer tests.
- On-device agent: only `sdk/provider/Gemma4Provider.kt`, on-device helper implementation, provider tests and runtime packaging notes. Coordinate build dependency needs with renderer agent.
- Bixby agent: only Bixby_18Sep integration files and Bixby integration documentation/tests. No Bixby build.
- Root: shared API types, converter, prompts, test-app integration, device benchmark, documentation and final assembly.

## Public API (root owns Api.kt)

`GenUiRequest(text: String, query: String? = null, sources: List<GenUiSource> = emptyList())`
`GenUiSource(id: String, url: String, title: String? = null)`
`GenUiDocument(express: String, a2uiJson: String, profile: String = "genuicraft_express_v1", schemaVersion: String = "v0.9")`
`GenUiCompileOutcome(document: GenUiDocument, repairKind: GenUiRepairKind, diagnostics: List<String>)`; successful conversions also expose `GenUiConversionResult.Success.repairKind` so hosts can distinguish unchanged, structurally normalized, explicitly generated-DSL-salvaged and source-fallback output without parsing warning text.
`GenUiAction(name: String, parameters: Map<String, String> = emptyMap())`
`GenUiPrompt(system: String, user: String, maxOutputTokens: Int = 8192, temperature: Double = 0.0)`
`GenUiGenerationFinishReason { COMPLETED, REPETITION_LIMIT }`
`GenUiGenerationMetrics(inputTokens: Int?, outputTokens: Int?, decodeTokensPerSecond: Double?, prefillTokensPerSecond: Double? = null, timeToFirstTokenSeconds: Double? = null, engineInitializationSeconds: Double? = null, engineInitializedForRequest: Boolean? = null, nativeInitializationPhaseSeconds: Double? = null)`. `engineInitializationSeconds` is measured critical-path wall time. The native phase sum is diagnostic only because native phases can overlap.
`GenUiGenerationSessionMetrics(speculativeDecodingEnabled: Boolean, drafterAcceptanceRate: Double? = null)`. Acceptance is verified draft tokens divided by proposed draft tokens for the complete metrics session.
`GenUiModelOutput(text: String, runtime: String, outputTokens: Int? = null, metrics: GenUiGenerationMetrics? = null, renderedPromptSha256: String? = null, finishReason: GenUiGenerationFinishReason = COMPLETED, finishDetail: String? = null)`. A repetition-limited result contains the exact accumulated partial output and continues through the normal generated-DSL recovery path.
`interface GenUiProvider : AutoCloseable { val id: String; suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput; suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? = null; override fun close() {}; suspend fun closeAndAwait() { close() } }`. Session telemetry is finalized after every conversion attempt and is optional; collection failure cannot change a successful conversion result.
`GenUiConversionResult.Success(document: GenUiDocument, provider: String, elapsedMs: Long, attempts: Int, warnings: List<String> = emptyList(), repairKind: GenUiRepairKind = NONE)`. `repairKind` classifies local output recovery; provider repair attempts remain represented by `attempts` and may still leave `repairKind == NONE`.
`GenUiConversionResult.Failure(message: String, provider: String, elapsedMs: Long, attempts: Int, rawOutput: String? = null)`
`ConversionOptions(maxOutputTokens: Int = 8192, maxRepairAttempts: Int = 1, temperature: Double = 0.0)`
`class GenUiConverter(context: Context, provider: GenUiProvider, options: ConversionOptions = ConversionOptions()) { suspend fun convert(request: GenUiRequest): GenUiConversionResult }`
`class GenUiSession(...)` owns the shared conversion, streaming, repair and native-engine lifecycle. `attemptSnapshots` keeps raw per-attempt output and timing; `generationSessionMetrics` exposes finalized MTP telemetry after `convert` completes.

## Compiler and renderer (renderer agent owns)

`object GenUiCompiler { fun compile(input: String): GenUiDocument; fun compileWithRepair(input: String, sourceText: String? = null, allowSourceTextFallback: Boolean = true, allowGeneratedDslRepair: Boolean = false): GenUiCompileOutcome }`. `compile` remains strict. The default recovery policy allows bounded syntax-only normalization and, when exact source text is supplied, a separately classified deterministic source-block fallback. `allowGeneratedDslRepair=true` explicitly enables broader model-output-only salvage; it is classified as `GENERATED_DSL_REPAIR`, and supplied source text still gates acceptance. `allowSourceTextFallback=false` guarantees that no source-built document is returned. Every accepted path re-enters strict compilation, and fallback passes mechanical content integrity. Fallback is never reported as repaired model output.
`@Composable fun GenUiContent(document: GenUiDocument, modifier: Modifier = Modifier, onAction: (GenUiAction) -> Unit = {})`
`class GenUiView(context: Context, attrs: AttributeSet? = null) : FrameLayout` with `fun render(document: GenUiDocument)`, `fun render(input: String)`, `fun clear()`, and `var onAction: (GenUiAction) -> Unit`.
All external actions (especially openUrl) are host callbacks. State-local renderer actions remain local. Native rendering requires no provider or model initialization.
Relocate extracted implementation to `com.samsung.genuicraft.sdk.internal` to avoid duplicate classes with the existing test app. Copy only dependency closure, not Activities/settings/whole app.

## Providers

`com.samsung.genuicraft.sdk.provider.Gemma4Provider(config: Gemma4Config)`
`Gemma4Config(modelPath: String, accelerator: String = "GPU", maxContextTokens: Int = 16384)` (agent may extend with optional fields, preserving these names).
Gemma runtime uses existing test app LiteRT-LM pattern; weights are external. No global app settings or test-app dependency. Cancellation/lifecycle, reusable engine, explicit runtime identity. Preserve model reasoning defaults; do not disable reasoning silently.
Native streaming stops a component-reference decode loop at 20 repeated references, including repeated ids and sequential alphabetic ids such as `aa, ab, ...`. The guard ignores state values and ordinary prose. Recovered table-state keys use substring target matching, with specific targets evaluated before broad ones; for example, `flight_options_schedule` selects the `flight` renderer route because it contains `flight_options`.

## Integration and validation

Bixby: classify using actual provider metadata, accumulate streaming response by request, convert once complete, cancel stale work, show native GenUiView/Compose result. Preserve citations/source metadata, TTS/history. A2UI failure must show error/retry; never label fallback text as model success. When its SDK integration is refreshed, consume the complete Maven publication; the current library coordinate is `com.samsung.genuicraft:genuicraft:0.5.0`. The host selects an on-device model profile. Document exact source extraction/classification limitations rather than guessing.

Test app: actual built AAR dependency, dedicated SDK demo/benchmark route, Bixby50 assets copied from `tmp/bixby_perplexity_check/run_50_exact/responses.jsonl`. On-device profiles, renderer-only replay, content/citation preservation, latency, screenshots and failure details. Keep benchmark outputs separate from source. Report cold initialization, successful raw generations, repaired generations and failures distinctly.
