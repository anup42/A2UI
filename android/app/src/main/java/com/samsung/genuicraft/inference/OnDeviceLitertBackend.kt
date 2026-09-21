package com.samsung.genuicraft.inference

import com.samsung.genuicraft.InferenceBackendSettings
import com.samsung.genuicraft.sdk.provider.LiteRtAccelerator
import com.samsung.genuicraft.sdk.provider.LiteRtConversationMessage
import com.samsung.genuicraft.sdk.provider.LiteRtConversationRole
import com.samsung.genuicraft.sdk.provider.LiteRtGenerationRequest
import com.samsung.genuicraft.sdk.provider.LiteRtModelConfig
import com.samsung.genuicraft.sdk.provider.LiteRtModelPolicy
import com.samsung.genuicraft.sdk.provider.LiteRtModelRunner

/**
 * App-facing adapter for the reusable GenUICraft LiteRT-LM runner.
 *
 * Model discovery and persisted settings belong to the reference app. Native initialization,
 * engine caching, prompt wrapping, generation, fallback, and metrics belong to the SDK.
 */
class OnDeviceLitertBackend(
    private val modelPath: String,
    private val acceleratorPreference: InferenceBackendSettings.Accelerator =
        InferenceBackendSettings.DEFAULT_ON_DEVICE_ACCELERATOR,
    private val npuNativeLibraryDir: String = "",
    private val mtpEnabled: Boolean = true,
) : InferenceBackend {
    private val runner: LiteRtModelRunner by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        val profile = OnDeviceModelCatalog.entryForModelPath(modelPath)
        LiteRtModelRunner(
            LiteRtModelConfig(
                modelPath = modelPath,
                accelerator = acceleratorPreference.toSdkAccelerator(),
                npuNativeLibraryDir = npuNativeLibraryDir,
                maxContextTokens = profile?.maxContextTokens ?: ON_DEVICE_MAX_CONTEXT_TOKENS,
                maxOutputTokens = profile?.maxOutputTokens ?: ON_DEVICE_MAX_OUTPUT_TOKENS,
                minimumFileSizeBytes = profile?.minimumFileSizeBytes ?: 1L,
                modelDisplayName = profile?.displayName,
                requireGpu = profile?.requireGpu == true,
                enableSpeculativeDecoding = mtpEnabled &&
                    profile?.enableSpeculativeDecoding == true,
                useRawTrainingWrapper = profile?.useRawTrainingWrapper == true,
                allowGpuQualityFallback = profile?.allowGpuQualityFallback == true,
            ),
        )
    }

    override fun generate(request: InferenceBackend.GenerateRequest): InferenceBackend.GenerateResponse {
        val startMs = System.currentTimeMillis()
        val health = checkHealth()
        if (!health.healthy) {
            return InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = health.errorMessage ?: "On-device model is not ready.",
                streamDurationMs = null,
            )
        }

        return try {
            val generation = runner.generate(
                LiteRtGenerationRequest(
                    prompt = request.prompt,
                    systemPrompt = request.systemPrompt,
                    temperature = request.temperature,
                    maxOutputTokens = request.maxOutputTokens,
                    sendSystemPrompt = request.localSendSystemPrompt,
                    initialMessages = request.initialMessages.map { message ->
                        LiteRtConversationMessage(
                            role = when (message.role) {
                                InferenceBackend.ConversationRole.USER -> LiteRtConversationRole.USER
                                InferenceBackend.ConversationRole.MODEL -> LiteRtConversationRole.MODEL
                            },
                            text = message.content,
                        )
                    },
                    onStreamUpdate = request.onStreamUpdate?.let { observer ->
                        { update ->
                            observer(
                                InferenceBackend.StreamUpdate(
                                    text = update.text,
                                    outputTokens = update.outputTokens,
                                    outputTokensPerSecond = update.outputTokensPerSecond,
                                    runtimeBackend = update.runtimeBackend,
                                    metricsAreEstimated = update.metricsAreEstimated,
                                    complete = update.complete,
                                ),
                            )
                        }
                    },
                ),
            )
            InferenceBackend.GenerateResponse(
                text = generation.text,
                rawResponse = generation.text,
                error = null,
                streamDurationMs = System.currentTimeMillis() - startMs,
                inputTokens = generation.inputTokens,
                outputTokens = generation.outputTokens,
                outputTokensPerSecond = generation.outputTokensPerSecond,
                runtimeBackend = generation.runtimeBackend,
            )
        } catch (failure: Throwable) {
            InferenceBackend.GenerateResponse(
                text = "",
                rawResponse = null,
                error = "On-device LiteRT generation failed: " +
                    (failure.message ?: failure.javaClass.simpleName),
                streamDurationMs = System.currentTimeMillis() - startMs,
            )
        }
    }

    override fun checkHealth(): InferenceBackend.HealthCheckResult {
        val health = runner.checkHealth()
        return InferenceBackend.HealthCheckResult(
            healthy = health.healthy,
            errorMessage = health.errorMessage,
        )
    }

    override fun classifyError(error: String): InferenceBackend.ErrorClass {
        return if (LiteRtModelPolicy.isTransientError(error)) {
            InferenceBackend.ErrorClass.TRANSIENT
        } else {
            InferenceBackend.ErrorClass.UNKNOWN
        }
    }

    companion object {
        private const val ON_DEVICE_MAX_CONTEXT_TOKENS = 12_288
        private const val ON_DEVICE_MAX_OUTPUT_TOKENS = 4_096

        /** Releases the SDK-owned shared engine before switching model-provider routes. */
        internal fun releaseCachedEngine() = LiteRtModelRunner.releaseCachedEngine()
    }
}

/** Compatibility delegates retained for existing app policy tests and callers. */
internal fun liteRtMtpEnabledForBackend(backendName: String, requested: Boolean): Boolean =
    LiteRtModelPolicy.mtpEnabledForBackend(backendName, requested)

internal fun liteRtRuntimeBackendLabel(
    backendName: String,
    speculativeDecodingEnabled: Boolean,
): String = LiteRtModelPolicy.runtimeBackendLabel(backendName, speculativeDecodingEnabled)

internal fun liteRtBackendOrder(
    forceCpu: Boolean,
    requireGpu: Boolean,
    accelerator: InferenceBackendSettings.Accelerator = InferenceBackendSettings.Accelerator.AUTO,
): List<String> = LiteRtModelPolicy.backendOrder(
    forceCpu = forceCpu,
    requireGpu = requireGpu,
    accelerator = accelerator.toSdkAccelerator(),
)

internal fun mergeLiteRtStreamText(current: String, incoming: String): String =
    LiteRtModelPolicy.mergeStreamText(current, incoming)

private fun InferenceBackendSettings.Accelerator.toSdkAccelerator(): LiteRtAccelerator = when (this) {
    InferenceBackendSettings.Accelerator.AUTO -> LiteRtAccelerator.AUTO
    InferenceBackendSettings.Accelerator.GPU -> LiteRtAccelerator.GPU
    InferenceBackendSettings.Accelerator.CPU -> LiteRtAccelerator.CPU
    InferenceBackendSettings.Accelerator.NPU -> LiteRtAccelerator.NPU
}
