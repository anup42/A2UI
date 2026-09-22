package com.samsung.genuicraft.sdk.provider

import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiGenerationSessionMetrics
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.sync.Mutex

/** Configuration for the on-device Gemma 4 LiteRT-LM runtime. */
data class Gemma4Config(
    val modelPath: String,
    val accelerator: String = "GPU",
    val maxContextTokens: Int = 16_384,
    val maxOutputTokens: Int = 8_192,
    val cpuThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 6),
    val cacheDir: String? = null,
    val enableThinking: Boolean = true,
    val thinkingTokenBudget: Int = 1_024,
    val enableSpeculativeDecoding: Boolean = true,
    /** Opt in to native token counts and decode throughput; no text-based estimates are returned. */
    val enableMetrics: Boolean = false,
)

/**
 * On-device Gemma 4 provider backed by LiteRT-LM.
 *
 * The engine is initialized lazily and reused. Requests are serialized because a LiteRT-LM engine
 * must not have overlapping conversations. Call [close] when the provider is no longer needed.
 */
class Gemma4Provider private constructor(
    private val validatedConfig: ValidatedGemma4Config,
    private val runtime: Gemma4Runtime,
) : GenUiProvider {
    constructor(config: Gemma4Config) : this(createProviderParts(config, ::LiteRtGemma4Runtime))

    internal constructor(
        config: Gemma4Config,
        runtimeFactory: Gemma4RuntimeFactory,
    ) : this(createProviderParts(config, runtimeFactory))

    private constructor(parts: Gemma4ProviderParts) : this(parts.config, parts.runtime)

    override val id: String = "gemma4_e2b"

    private val generationMutex = Mutex()
    private val closed = AtomicBoolean(false)

    override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput = generateInternal(prompt, null)

    override suspend fun generate(
        prompt: GenUiPrompt,
        onPartialText: (String) -> Unit,
    ): GenUiModelOutput = generateInternal(prompt, onPartialText)

    private suspend fun generateInternal(
        prompt: GenUiPrompt,
        onPartialText: ((String) -> Unit)?,
    ): GenUiModelOutput {
        check(!closed.get()) { "Gemma 4 provider has been closed." }
        generationMutex.lock()
        try {
            check(!closed.get()) { "Gemma 4 provider has been closed." }
            val maxOutputTokens = Gemma4PromptLimits.validate(
                prompt = prompt,
                config = validatedConfig,
            )
            val generation = try {
                if (onPartialText == null) runtime.generate(prompt, maxOutputTokens)
                else runtime.generate(prompt, maxOutputTokens, onPartialText)
            } catch (cancelled: CancellationException) {
                runtime.cancelActive()
                throw cancelled
            }
            require(generation.text.isNotBlank()) {
                "LiteRT-LM returned an empty Gemma 4 response."
            }
            return GenUiModelOutput(
                text = generation.text,
                runtime = generation.runtimeIdentity,
                outputTokens = generation.outputTokens,
                metrics = generation.metrics.takeIf { validatedConfig.enableMetrics },
                renderedPromptSha256 = generation.renderedPromptSha256,
                finishReason = generation.finishReason,
                finishDetail = generation.finishDetail,
            )
        } finally {
            generationMutex.unlock()
        }
    }

    override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? {
        if (!validatedConfig.enableMetrics || closed.get()) return null
        generationMutex.lock()
        return try {
            if (closed.get()) null else runtime.finishGenerationMetrics()
        } finally {
            generationMutex.unlock()
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        // close() is commonly called from Activity.onDestroy on the main thread. The runtime
        // cancels native work immediately and queues engine cleanup on its private worker so this
        // method never waits for a generation coroutine to resume on main.
        runtime.cancelActive()
        runtime.close()
    }

    override suspend fun closeAndAwait() {
        close()
        runtime.awaitClosed()
    }
}

private data class Gemma4ProviderParts(
    val config: ValidatedGemma4Config,
    val runtime: Gemma4Runtime,
)

private fun createProviderParts(
    config: Gemma4Config,
    runtimeFactory: Gemma4RuntimeFactory,
): Gemma4ProviderParts {
    val validated = config.validate()
    return Gemma4ProviderParts(validated, runtimeFactory.create(validated))
}

internal data class ValidatedGemma4Config(
    val modelPath: String,
    val accelerator: Gemma4Accelerator,
    val maxContextTokens: Int,
    val maxOutputTokens: Int,
    val cpuThreads: Int,
    val cacheDir: String?,
    val enableThinking: Boolean,
    val thinkingTokenBudget: Int,
    val enableSpeculativeDecoding: Boolean,
    val enableMetrics: Boolean,
)

internal enum class Gemma4Accelerator {
    GPU,
    CPU,
}

private fun Gemma4Config.validate(): ValidatedGemma4Config {
    val trimmedPath = modelPath.trim()
    require(trimmedPath.isNotEmpty()) { "Gemma 4 modelPath must not be blank." }
    val modelFile = File(trimmedPath)
    require(modelFile.isAbsolute) { "Gemma 4 modelPath must be an absolute filesystem path." }
    require(modelFile.isFile && modelFile.canRead()) {
        "Gemma 4 modelPath does not point to a readable model file: $trimmedPath"
    }
    require(modelFile.length() > 0L) { "Gemma 4 model file is empty: $trimmedPath" }
    require(modelFile.name.lowercase(Locale.US).endsWith(".litertlm")) {
        "Gemma 4 modelPath must point to a .litertlm package."
    }

    val normalizedAccelerator = accelerator.trim().uppercase(Locale.US)
    val parsedAccelerator = runCatching { Gemma4Accelerator.valueOf(normalizedAccelerator) }
        .getOrElse {
            throw IllegalArgumentException(
                "Unsupported Gemma 4 accelerator '$accelerator'. Use GPU or CPU.",
            )
        }
    require(maxContextTokens >= Gemma4PromptLimits.MIN_CONTEXT_TOKENS) {
        "Gemma 4 maxContextTokens must be at least ${Gemma4PromptLimits.MIN_CONTEXT_TOKENS}."
    }
    require(maxOutputTokens in 1 until maxContextTokens) {
        "Gemma 4 maxOutputTokens must be positive and smaller than maxContextTokens."
    }
    require(cpuThreads in 1..64) { "Gemma 4 cpuThreads must be between 1 and 64." }
    require(thinkingTokenBudget >= -1) {
        "Gemma 4 thinkingTokenBudget must be -1 (model/runtime default) or non-negative."
    }

    val normalizedCacheDir = cacheDir?.trim()?.takeIf { it.isNotEmpty() }?.let { path ->
        val directory = File(path)
        require(directory.isAbsolute) { "Gemma 4 cacheDir must be an absolute filesystem path." }
        directory.canonicalPath
    }
    return ValidatedGemma4Config(
        modelPath = modelFile.canonicalPath,
        accelerator = parsedAccelerator,
        maxContextTokens = maxContextTokens,
        maxOutputTokens = maxOutputTokens,
        cpuThreads = cpuThreads,
        cacheDir = normalizedCacheDir,
        enableThinking = enableThinking,
        thinkingTokenBudget = thinkingTokenBudget,
        enableSpeculativeDecoding = enableSpeculativeDecoding,
        enableMetrics = enableMetrics,
    )
}

internal object Gemma4PromptLimits {
    const val MIN_CONTEXT_TOKENS = 1_024
    private const val CHAT_TEMPLATE_RESERVE_TOKENS = 256

    fun validate(prompt: GenUiPrompt, config: ValidatedGemma4Config): Int {
        require(prompt.user.isNotBlank()) { "Gemma 4 user prompt must not be blank." }
        require(prompt.temperature.isFinite() && prompt.temperature in 0.0..2.0) {
            "Gemma 4 temperature must be finite and between 0.0 and 2.0."
        }
        require(prompt.maxOutputTokens in 1..config.maxOutputTokens) {
            "Gemma 4 maxOutputTokens must be between 1 and ${config.maxOutputTokens}."
        }

        val estimatedInputTokens = estimateTokens(prompt.system).toLong() + estimateTokens(prompt.user) +
            prompt.initialMessages.sumOf { estimateTokens(it.text).toLong() + 8 }
        val estimatedTotal = estimatedInputTokens.toLong() +
            prompt.maxOutputTokens.toLong() +
            CHAT_TEMPLATE_RESERVE_TOKENS
        require(estimatedTotal <= config.maxContextTokens.toLong()) {
            "Gemma 4 prompt and requested output exceed the configured context window " +
                "(${config.maxContextTokens} tokens; estimated input=$estimatedInputTokens, " +
                "output=${prompt.maxOutputTokens}, template reserve=$CHAT_TEMPLATE_RESERVE_TOKENS)."
        }
        return prompt.maxOutputTokens
    }

    /** Conservative preflight estimate; LiteRT-LM remains the authoritative tokenizer. */
    internal fun estimateTokens(text: String): Int {
        if (text.isEmpty()) return 0
        val utf8Bytes = text.toByteArray(Charsets.UTF_8).size
        return ((utf8Bytes + 2L) / 3L).coerceAtLeast(1L).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
    }
}
