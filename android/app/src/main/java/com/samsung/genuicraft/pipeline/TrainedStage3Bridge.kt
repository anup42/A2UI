package com.samsung.genuicraft.pipeline

import android.content.Context
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.InferenceBackendSettings
import com.samsung.genuicraft.inference.OnDeviceLitertBackend
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.sdk.GenUiConversionResult
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiRepairKind
import com.samsung.genuicraft.sdk.GenUiRequest
import com.samsung.genuicraft.sdk.GenUiTrainedConverter
import com.samsung.genuicraft.sdk.provider.Gemma4Config
import com.samsung.genuicraft.sdk.provider.Gemma4Provider
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

/**
 * Boundary between the app pipeline and the SDK's frozen trained-E2B converter.
 *
 * The bridge deliberately keeps the generated-output-only demo policy separate from the legacy
 * LiteRT backend. It also captures the native provider result because the converter's public result
 * carries the compiled document and warnings, while token/runtime measurements remain provider data.
 */
internal object TrainedStage3Bridge {
    sealed interface Result {
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
    ): Gemma4Config {
        require(
            accelerator == InferenceBackendSettings.Accelerator.AUTO ||
                accelerator == InferenceBackendSettings.Accelerator.GPU
        ) {
            "The trained A2UI Mobile model requires GPU. Select Auto or GPU in Settings."
        }
        return Gemma4Config(
            modelPath = modelPath,
            accelerator = "GPU",
            maxContextTokens = TRAINED_CONTEXT_TOKENS,
            maxOutputTokens = TRAINED_OUTPUT_TOKENS,
            enableThinking = false,
            thinkingTokenBudget = 0,
            enableSpeculativeDecoding = enableMtp,
            enableMetrics = true,
        )
    }

    suspend fun convert(
        context: Context,
        modelPath: String,
        sourceResponse: String,
        queryText: String,
        onPartialText: ((String) -> Unit)? = null,
    ): Result {
        val config = runCatching {
            configFor(
                modelPath = modelPath,
                accelerator = InferenceBackendSettings.getOnDeviceAccelerator(context),
                enableMtp = InferenceBackendSettings.getOnDeviceMtpEnabled(context),
            )
        }.getOrElse { error ->
            return Result.Failure(
                message = error.message ?: error.javaClass.simpleName,
                rawGeneratedText = null,
                prompt = null,
                inputTokens = null,
                outputTokens = null,
                outputTokensPerSecond = null,
                runtimeBackend = null,
                renderedPromptSha256 = null,
            )
        }

        return try {
            coroutineContext.ensureActive()
            // A legacy route may have initialized a process-wide LiteRT engine during an earlier
            // request. Holding it while the SDK creates the trained GPU engine can exhaust memory.
            OnDeviceLitertBackend.releaseCachedEngine()
            coroutineContext.ensureActive()
            runWithProvider(
                provider = Gemma4Provider(config),
                request = GenUiRequest(text = sourceResponse, query = queryText),
                onPartialText = onPartialText,
            ) { provider, request ->
                GenUiTrainedConverter(
                    context = context,
                    provider = provider,
                    allowSourceTextFallback = false,
                    allowGeneratedDslRepair = true,
                    requireSourceIntegrity = false,
                ).convert(request)
            }
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
        val capture = CapturingProvider(provider, onPartialText)
        val result = try {
            val conversion = convert(capture, request)
            adapt(
                conversion = conversion,
                prompt = capture.lastPrompt,
                output = capture.lastOutput,
                partialText = capture.lastPartialText,
            )
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Exception) {
            failure(
                message = failure.message ?: failure.javaClass.simpleName,
                prompt = capture.lastPrompt,
                output = capture.lastOutput,
                rawGeneratedText = capture.lastPartialText,
            )
        } finally {
            // Native cleanup must finish even when the host cancels the Activity coroutine.
            withContext(NonCancellable) {
                capture.closeAndAwait()
            }
        }
        coroutineContext.ensureActive()
        return result
    }

    private fun adapt(
        conversion: GenUiConversionResult,
        prompt: GenUiPrompt?,
        output: GenUiModelOutput?,
        partialText: String?,
    ): Result = when (conversion) {
        is GenUiConversionResult.Failure -> failure(
            message = conversion.message,
            // The converter's failure payload is normalized for diagnostics. Prefer the native
            // provider capture so the raw stream and terminal snapshot remain byte-for-byte equal.
            rawGeneratedText = output?.text ?: partialText ?: conversion.rawOutput,
            prompt = prompt,
            output = output,
            elapsedMs = conversion.elapsedMs,
        )

        is GenUiConversionResult.Success -> {
            if (conversion.repairKind == GenUiRepairKind.SOURCE_TEXT_FALLBACK) {
                failure(
                    message = "The trained converter attempted SOURCE_TEXT_FALLBACK; generated-only demo output was rejected.",
                    prompt = prompt,
                    output = output,
                    elapsedMs = conversion.elapsedMs,
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
        )
    }

    private class CapturingProvider(
        private val delegate: GenUiProvider,
        private val onPartialText: ((String) -> Unit)?,
    ) : GenUiProvider {
        override val id: String
            get() = delegate.id

        var lastPrompt: GenUiPrompt? = null
            private set
        var lastOutput: GenUiModelOutput? = null
            private set
        @Volatile
        var lastPartialText: String? = null
            private set

        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
            return generateCaptured(prompt) { text ->
                onPartialText?.invoke(text)
            }
        }

        override suspend fun generate(
            prompt: GenUiPrompt,
            onPartialText: (String) -> Unit,
        ): GenUiModelOutput {
            return generateCaptured(prompt) { text ->
                this.onPartialText?.invoke(text)
                onPartialText(text)
            }
        }

        private suspend fun generateCaptured(
            prompt: GenUiPrompt,
            forwardPartialText: (String) -> Unit,
        ): GenUiModelOutput {
            lastPrompt = prompt
            return delegate.generate(prompt) { text ->
                // Store first so cancellation or observer errors cannot erase the diagnostic prefix.
                lastPartialText = text
                forwardPartialText(text)
            }.also { output ->
                lastOutput = output
                if (lastPartialText == null) {
                    lastPartialText = output.text
                }
            }
        }

        override fun close() = delegate.close()

        override suspend fun closeAndAwait() = delegate.closeAndAwait()
    }

    internal const val TRAINED_CONTEXT_TOKENS = 8_192
    internal const val TRAINED_OUTPUT_TOKENS = 2_048
}
