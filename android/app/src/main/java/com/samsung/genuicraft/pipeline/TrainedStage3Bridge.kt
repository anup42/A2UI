package com.samsung.genuicraft.pipeline

import android.content.Context
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.InferenceBackendSettings
import com.samsung.genuicraft.selectTrainedE2bModel
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.sdk.GenUiConversionResult
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiRepairKind
import com.samsung.genuicraft.sdk.GenUiRequest
import com.samsung.genuicraft.sdk.GenUiRenderSnapshot
import com.samsung.genuicraft.sdk.GenUiSession
import com.samsung.genuicraft.sdk.GenUiConversionProfile
import com.samsung.genuicraft.sdk.GenUiGenerationObserver
import com.samsung.genuicraft.sdk.GenUiModelProfiles
import com.samsung.genuicraft.sdk.provider.Gemma4Config
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
import com.samsung.genuicraft.sdk.provider.Gemma4Provider
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive

/**
 * Boundary between the app pipeline and the SDK's frozen trained-E2B converter.
 *
 * The SDK owns inference, recovery policy, streaming capture, metrics and native cleanup.
 * This adapter only selects host settings and maps the returned wire document to the app graph.
 */
internal object TrainedStage3Bridge {
    /** A request's native settings are captured before Stage 2 and reused for Stage 3. */
    internal sealed interface PreparedModel {
        data class Ready(
            val config: Gemma4Config,
            val selectedPrecision: InferenceBackendSettings.TrainedE2bGpuPrecision,
            val selectedAccelerator: InferenceBackendSettings.Accelerator,
        ) : PreparedModel
        data class Unavailable(val message: String) : PreparedModel
    }

    sealed interface Result {
        /** Wall time spent in the native provider call, excluding DSL validation and repair. */
        val generationElapsedMs: Long?

        data class Success(
            val canonicalGraph: JsonObject,
            val express: String,
            val wireJson: String,
            val rawGeneratedText: String,
            val prompt: GenUiPrompt?,
            val inputTokens: Int?,
            val outputTokens: Int?,
            val outputTokensPerSecond: Double?,
            val runtimeBackend: String?,
            val renderedPromptSha256: String?,
            val elapsedMs: Long,
            override val generationElapsedMs: Long?,
            val warnings: List<String>,
            val repairKind: GenUiRepairKind,
        ) : Result

        data class Failure(
            val message: String,
            val rawGeneratedText: String?,
            val prompt: GenUiPrompt?,
            val inputTokens: Int?,
            val outputTokens: Int?,
            val outputTokensPerSecond: Double?,
            val runtimeBackend: String?,
            val renderedPromptSha256: String?,
            val elapsedMs: Long? = null,
            override val generationElapsedMs: Long? = null,
        ) : Result
    }

    fun shouldUse(
        provider: InferenceBackendSettings.Provider,
        modelPath: String,
    ): Boolean = provider == InferenceBackendSettings.Provider.ON_DEVICE_LITERT &&
        OnDeviceModelCatalog.entryForModelPath(modelPath)?.usesTrainedSdkConverter == true

    internal fun configFor(
        modelPath: String,
        accelerator: InferenceBackendSettings.Accelerator,
        enableMtp: Boolean,
        gpuPrecision: Gemma4GpuPrecision = Gemma4GpuPrecision.FP32,
    ): Gemma4Config = GenUiModelProfiles.trainedE2b(
        modelPath = modelPath,
        accelerator = accelerator.name,
        enableMtp = enableMtp,
        enableMetrics = true,
        gpuPrecision = gpuPrecision,
    )

    internal fun prepareModel(context: Context, modelPath: String): PreparedModel = try {
        val precision = InferenceBackendSettings.getTrainedE2bGpuPrecision(context)
        val accelerator = InferenceBackendSettings.getOnDeviceAccelerator(context)
        val enableMtp = InferenceBackendSettings.getOnDeviceMtpEnabled(context)
        val selection = selectTrainedE2bModel(modelPath, precision)
        if (!selection.readiness.usable) {
            PreparedModel.Unavailable(selection.readiness.message)
        } else {
            PreparedModel.Ready(
                config = configFor(
                    modelPath = selection.modelPath,
                    accelerator = accelerator,
                    enableMtp = selection.mtpSupported && enableMtp,
                    gpuPrecision = selection.gpuPrecision,
                ),
                selectedPrecision = precision,
                selectedAccelerator = accelerator,
            )
        }
    } catch (failure: Exception) {
        PreparedModel.Unavailable(failure.message ?: failure.javaClass.simpleName)
    }

    /** Prepare native weights without generating a dummy turn or retaining any conversation state. */
    suspend fun prewarm(context: Context, modelPath: String) {
        prewarm(context, prepareModel(context, modelPath))
    }

    internal suspend fun prewarm(context: Context, preparedModel: PreparedModel) {
        val config = when (preparedModel) {
            is PreparedModel.Ready -> preparedModel.config
            is PreparedModel.Unavailable -> error(preparedModel.message)
        }
        val session = GenUiSession(
            context,
            Gemma4Provider(config),
            GenUiConversionProfile.TRAINED_E2B_V10_W4,
        )
        try {
            coroutineContext.ensureActive()
            session.prewarm()
        } finally {
            session.closeAndAwait()
        }
    }

    suspend fun convert(
        context: Context,
        modelPath: String,
        sourceResponse: String,
        queryText: String,
        onPartialText: ((String) -> Unit)? = null,
        onRenderSnapshot: ((GenUiRenderSnapshot) -> Unit)? = null,
        onAttemptStarted: ((Int) -> Unit)? = null,
        enableStreamingRendering: Boolean = true,
        preparedModel: PreparedModel? = null,
    ): Result {
        val config = when (val prepared = preparedModel ?: prepareModel(context, modelPath)) {
            is PreparedModel.Ready -> prepared.config
            is PreparedModel.Unavailable -> {
                return Result.Failure(
                    message = prepared.message,
                    rawGeneratedText = null,
                    prompt = null,
                    inputTokens = null,
                    outputTokens = null,
                    outputTokensPerSecond = null,
                    runtimeBackend = null,
                    renderedPromptSha256 = null,
                )
            }
        }

        return try {
            coroutineContext.ensureActive()
            runWithSession(
                session = GenUiSession(context, Gemma4Provider(config), GenUiConversionProfile.TRAINED_E2B_V10_W4),
                request = GenUiRequest(text = sourceResponse, query = queryText),
                onPartialText = onPartialText,
                onRenderSnapshot = onRenderSnapshot,
                onAttemptStarted = onAttemptStarted,
                enableStreamingRendering = enableStreamingRendering,
            )
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Exception) {
            Result.Failure(
                message = failure.message ?: failure.javaClass.simpleName,
                rawGeneratedText = null,
                prompt = null,
                inputTokens = null,
                outputTokens = null,
                outputTokensPerSecond = null,
                runtimeBackend = null,
                renderedPromptSha256 = null,
            )
        }
    }

    internal suspend fun runWithProvider(
        provider: GenUiProvider,
        request: GenUiRequest,
        onPartialText: ((String) -> Unit)? = null,
        convert: suspend (GenUiProvider, GenUiRequest) -> GenUiConversionResult,
    ): Result {
        return runWithSession(GenUiSession(provider, convert), request, onPartialText)
    }

    private suspend fun runWithSession(
        session: GenUiSession,
        request: GenUiRequest,
        onPartialText: ((String) -> Unit)?,
        onRenderSnapshot: ((GenUiRenderSnapshot) -> Unit)? = null,
        onAttemptStarted: ((Int) -> Unit)? = null,
        enableStreamingRendering: Boolean = true,
    ): Result {
        val result = try {
            val conversion = session.convert(
                request,
                GenUiGenerationObserver(
                    onPartialText = { _, text -> onPartialText?.invoke(text) },
                    onRenderSnapshot = onRenderSnapshot ?: {},
                    onAttemptStarted = onAttemptStarted ?: {},
                ),
                enableStreamingRendering = enableStreamingRendering,
            )
            val attempt = session.attemptSnapshots.lastOrNull()
            adapt(
                conversion = conversion,
                prompt = attempt?.prompt,
                output = attempt?.output,
                partialText = attempt?.rawText,
                generationElapsedMs = attempt?.elapsedNanos?.div(1_000_000L),
            )
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Exception) {
            val attempt = session.attemptSnapshots.lastOrNull()
            failure(
                message = failure.message ?: failure.javaClass.simpleName,
                prompt = attempt?.prompt,
                output = attempt?.output,
                rawGeneratedText = attempt?.rawText,
                generationElapsedMs = attempt?.elapsedNanos?.div(1_000_000L),
            )
        } finally {
            session.closeAndAwait()
        }
        coroutineContext.ensureActive()
        return result
    }

    private fun adapt(
        conversion: GenUiConversionResult,
        prompt: GenUiPrompt?,
        output: GenUiModelOutput?,
        partialText: String?,
        generationElapsedMs: Long?,
    ): Result = when (conversion) {
        is GenUiConversionResult.Failure -> failure(
            message = conversion.message,
            // The converter's failure payload is normalized for diagnostics. Prefer the native
            // provider capture so the raw stream and terminal snapshot remain byte-for-byte equal.
            rawGeneratedText = output?.text ?: partialText ?: conversion.rawOutput,
            prompt = prompt,
            output = output,
            elapsedMs = conversion.elapsedMs,
            generationElapsedMs = generationElapsedMs,
        )

        is GenUiConversionResult.Success -> {
            if (conversion.repairKind == GenUiRepairKind.SOURCE_TEXT_FALLBACK) {
                failure(
                    message = "The trained converter attempted SOURCE_TEXT_FALLBACK; generated-only demo output was rejected.",
                    prompt = prompt,
                    output = output,
                    elapsedMs = conversion.elapsedMs,
                    generationElapsedMs = generationElapsedMs,
                )
            } else {
                val decoded = GenUiIrCodec.decode(
                    JsonParser.parseString(conversion.document.a2uiJson),
                )
                require(decoded.sourceFormat == GenUiIrFormat.A2UI_V1_WIRE) {
                    "The trained SDK converter did not return standard A2UI wire JSON."
                }
                val metrics = output?.metrics
                Result.Success(
                    canonicalGraph = decoded.canonicalGraph,
                    express = conversion.document.express,
                    wireJson = conversion.document.a2uiJson,
                    rawGeneratedText = output?.text ?: partialText.orEmpty(),
                    prompt = prompt,
                    inputTokens = metrics?.inputTokens,
                    outputTokens = metrics?.outputTokens ?: output?.outputTokens,
                    outputTokensPerSecond = metrics?.decodeTokensPerSecond,
                    runtimeBackend = output?.runtime,
                    renderedPromptSha256 = output?.renderedPromptSha256,
                    elapsedMs = conversion.elapsedMs,
                    generationElapsedMs = generationElapsedMs,
                    warnings = conversion.warnings,
                    repairKind = conversion.repairKind,
                )
            }
        }
    }

    private fun failure(
        message: String,
        prompt: GenUiPrompt?,
        output: GenUiModelOutput?,
        rawGeneratedText: String? = null,
        elapsedMs: Long? = null,
        generationElapsedMs: Long? = null,
    ): Result.Failure {
        val metrics = output?.metrics
        return Result.Failure(
            message = message,
            rawGeneratedText = output?.text ?: rawGeneratedText,
            prompt = prompt,
            inputTokens = metrics?.inputTokens,
            outputTokens = metrics?.outputTokens ?: output?.outputTokens,
            outputTokensPerSecond = metrics?.decodeTokensPerSecond,
            runtimeBackend = output?.runtime,
            renderedPromptSha256 = output?.renderedPromptSha256,
            elapsedMs = elapsedMs,
            generationElapsedMs = generationElapsedMs,
        )
    }

    internal const val TRAINED_CONTEXT_TOKENS = GenUiModelProfiles.TRAINED_CONTEXT_TOKENS
    internal const val TRAINED_OUTPUT_TOKENS = GenUiModelProfiles.TRAINED_OUTPUT_TOKENS
}
