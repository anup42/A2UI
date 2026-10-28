package com.samsung.genuicraft.sdk.provider

import com.samsung.genuicraft.sdk.GenUiGenerationSessionMetrics
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiModelProfiles
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider

/** Native Gemma 3 270M A2UI Express provider using the package's own chat template. */
class Gemma270MProvider private constructor(
    private val delegate: Gemma4Provider,
) : GenUiProvider {
    constructor(config: Gemma4Config) : this(createDelegate(config))

    internal constructor(config: Gemma4Config, runtimeFactory: Gemma4RuntimeFactory) :
        this(createDelegate(config, runtimeFactory))

    override val id: String get() = delegate.id
    override suspend fun prewarm() = delegate.prewarm()

    override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
        requireNativeTemplate(prompt)
        return delegate.generate(prompt)
    }

    override suspend fun generate(
        prompt: GenUiPrompt,
        onPartialText: (String) -> Unit,
    ): GenUiModelOutput {
        requireNativeTemplate(prompt)
        return delegate.generate(prompt, onPartialText)
    }

    override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? =
        delegate.finishGenerationMetrics()

    override fun close() = delegate.close()

    override suspend fun closeAndAwait() = delegate.closeAndAwait()

    companion object {
        const val PROVIDER_ID = "gemma3_270m_a2ui_express"

        private fun requireNativeTemplate(prompt: GenUiPrompt) {
            require(prompt.chatTemplateOverride == null && prompt.expectedRenderedPrompt == null) {
                "Gemma 3 270M must use the model package's native chat template."
            }
        }

        private fun createDelegate(
            config: Gemma4Config,
            runtimeFactory: Gemma4RuntimeFactory = Gemma4RuntimeFactory(GenUiEngineManager::lease),
        ): Gemma4Provider {
            require(config.maxContextTokens == GenUiModelProfiles.GEMMA270M_CONTEXT_TOKENS &&
                config.maxOutputTokens == GenUiModelProfiles.GEMMA270M_OUTPUT_TOKENS) {
                "Gemma 3 270M A2UI Express requires a 4096-token context and 2048-token output budget."
            }
            require(!config.enableThinking && config.thinkingTokenBudget == 0) {
                "Gemma 3 270M A2UI Express does not use thinking mode."
            }
            require(!config.enableSpeculativeDecoding) {
                "Gemma 3 270M A2UI Express does not use MTP."
            }
            require(config.gpuPrecision == Gemma4GpuPrecision.MODEL_DEFAULT) {
                "Gemma 3 270M A2UI Express uses model-default arithmetic precision."
            }
            return Gemma4Provider(
                config = config,
                providerId = PROVIDER_ID,
                runtimeFamily = "Gemma3-270M",
                runtimeFactory = runtimeFactory,
            )
        }
    }
}
