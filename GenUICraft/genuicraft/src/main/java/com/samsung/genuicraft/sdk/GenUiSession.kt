package com.samsung.genuicraft.sdk

import android.content.Context
import com.samsung.genuicraft.sdk.provider.Gemma4Config
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Conversion and recovery policies supplied by the AAR, shared by every host view. */
enum class GenUiConversionProfile {
    SOURCE_BOUND,
    /** Frozen v10 training prompt, generated-output recovery, no source-text replacement. */
    TRAINED_E2B_V10_W4,
}

/** Native settings that must agree between hosts using the same model export. */
object GenUiModelProfiles {
    const val TRAINED_CONTEXT_TOKENS = 8_192
    const val TRAINED_OUTPUT_TOKENS = 2_048

    fun trainedE2b(
        modelPath: String,
        enableMtp: Boolean = false,
        enableMetrics: Boolean = false,
        accelerator: String = "GPU",
    ): Gemma4Config {
        require(accelerator.uppercase(java.util.Locale.ROOT) in setOf("AUTO", "GPU")) {
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
            enableMetrics = enableMetrics,
        )
    }
}

/** Callbacks run on the generation worker; dispatch UI updates to the main thread. */
data class GenUiGenerationObserver(
    val onAttemptStarted: (Int) -> Unit = {},
    val onPartialText: (Int, String) -> Unit = { _, _ -> },
    val onAttemptCompleted: (Int, GenUiModelOutput) -> Unit = { _, _ -> },
)

/** Exact native output and measurements, never replaced by compiled or repaired IR. */
data class GenUiGenerationAttempt(
    val number: Int,
    val rawText: String,
    val complete: Boolean,
    val prompt: GenUiPrompt? = null,
    val output: GenUiModelOutput? = null,
    val elapsedNanos: Long? = null,
)

/**
 * Common raw-text -> inference -> Express recovery -> A2UI entry point.
 *
 * The session retains exact generation attempts (also after cancellation) separately from the
 * repaired document returned by [convert]. It can reuse a provider across requests. The owner must
 * call [closeAndAwait] before constructing a replacement native engine, or [close] on destruction.
 * The provider/conversion constructor supports custom converters; normal hosts use the Context
 * constructor so prompting and recovery policy stay in this library.
 */
class GenUiSession(
    private val provider: GenUiProvider,
    private val conversion: suspend (GenUiProvider, GenUiRequest) -> GenUiConversionResult,
) : AutoCloseable {
    constructor(
        context: Context,
        provider: GenUiProvider,
        profile: GenUiConversionProfile = GenUiConversionProfile.SOURCE_BOUND,
        options: ConversionOptions = ConversionOptions(),
    ) : this(provider, conversionFor(context.applicationContext, profile, options))

    private val mutex = Mutex()
    private val closed = AtomicBoolean(false)
    @Volatile private var capture: GenUiStreamingProvider? = null

    val attemptSnapshots: List<GenUiGenerationAttempt>
        get() = capture?.attemptSnapshots.orEmpty()

    suspend fun convert(
        request: GenUiRequest,
        observer: GenUiGenerationObserver = GenUiGenerationObserver(),
    ): GenUiConversionResult = mutex.withLock {
        check(!closed.get()) { "GenUICraft session has been closed." }
        val observed = GenUiStreamingProvider(
            provider, observer.onAttemptStarted, observer.onPartialText, observer.onAttemptCompleted,
        )
        capture = observed
        conversion(observed, request)
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) provider.close()
    }

    suspend fun closeAndAwait() {
        closed.set(true)
        withContext(NonCancellable) {
            mutex.withLock { provider.closeAndAwait() }
        }
    }

    companion object {
        private fun conversionFor(
            context: Context,
            profile: GenUiConversionProfile,
            options: ConversionOptions,
        ): suspend (GenUiProvider, GenUiRequest) -> GenUiConversionResult = { provider, request ->
            when (profile) {
                GenUiConversionProfile.SOURCE_BOUND -> GenUiConverter(context, provider, options).convert(request)
                GenUiConversionProfile.TRAINED_E2B_V10_W4 -> GenUiTrainedConverter(
                    context, provider,
                    allowSourceTextFallback = false,
                    allowGeneratedDslRepair = true,
                    requireSourceIntegrity = false,
                ).convert(request)
            }
        }
    }
}
