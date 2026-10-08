package com.samsung.genuicraft.sdk

import android.content.Context
import com.samsung.genuicraft.sdk.provider.Gemma4Config
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
import com.samsung.genuicraft.sdk.provider.LiteRtModelRunner
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.coroutineScope
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
        gpuPrecision: Gemma4GpuPrecision = Gemma4GpuPrecision.FP32,
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
            gpuPrecision = gpuPrecision,
        )
    }
}

/** Callbacks run off the UI thread or on the provider worker; dispatch UI updates to main. */
internal val NO_RENDER_OBSERVER: (GenUiRenderSnapshot) -> Unit = {}

data class GenUiGenerationObserver(
    val onAttemptStarted: (Int) -> Unit = {},
    val onPartialText: (Int, String) -> Unit = { _, _ -> },
    val onAttemptCompleted: (Int, GenUiModelOutput) -> Unit = { _, _ -> },
    /** Provisional native revisions and the accepted final document; dispatched off native decode. */
    val onRenderSnapshot: (GenUiRenderSnapshot) -> Unit = NO_RENDER_OBSERVER,
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
class GenUiSession private constructor(
    private val provider: GenUiProvider,
    private val prepareRuntime: () -> Unit,
    private val conversion: suspend (GenUiProvider, GenUiRequest) -> GenUiConversionResult,
) : AutoCloseable {
    constructor(
        provider: GenUiProvider,
        conversion: suspend (GenUiProvider, GenUiRequest) -> GenUiConversionResult,
    ) : this(provider, {}, conversion)

    constructor(
        context: Context,
        provider: GenUiProvider,
        profile: GenUiConversionProfile = GenUiConversionProfile.SOURCE_BOUND,
        options: ConversionOptions = ConversionOptions(),
    ) : this(
        provider,
        runtimePreparationFor(profile),
        conversionFor(context.applicationContext, profile, options),
    )

    private val mutex = Mutex()
    private val closed = AtomicBoolean(false)
    @Volatile private var capture: GenUiStreamingProvider? = null
    @Volatile private var completedSessionMetrics: GenUiGenerationSessionMetrics? = null

    val attemptSnapshots: List<GenUiGenerationAttempt>
        get() = capture?.attemptSnapshots.orEmpty()

    /** Metrics finalized after every generation/repair attempt in the most recent conversion. */
    val generationSessionMetrics: GenUiGenerationSessionMetrics?
        get() = completedSessionMetrics

    suspend fun convert(
        request: GenUiRequest,
        observer: GenUiGenerationObserver = GenUiGenerationObserver(),
    ): GenUiConversionResult = convert(request, observer, enableStreamingRendering = true)

    /** Disable previews for throughput comparisons; raw streaming and final repair stay identical. */
    suspend fun convert(
        request: GenUiRequest,
        observer: GenUiGenerationObserver,
        enableStreamingRendering: Boolean,
    ): GenUiConversionResult = mutex.withLock { coroutineScope {
        val startedNanos = System.nanoTime()
        check(!closed.get()) { "GenUICraft session has been closed." }
        capture = null
        completedSessionMetrics = null
        coroutineContext.ensureActive()
        withContext(Dispatchers.IO) { prepareRuntime() }
        coroutineContext.ensureActive()
        check(!closed.get()) { "GenUICraft session was closed while preparing its runtime." }
        val preview = if (enableStreamingRendering && observer.onRenderSnapshot !== NO_RENDER_OBSERVER)
            GenUiPreviewCoordinator(this, request.sources, observer.onRenderSnapshot, startedNanos) else null
        val observed = GenUiStreamingProvider(provider,
            onAttemptStarted = { attempt ->
                preview?.startAttempt(attempt)
                observer.onAttemptStarted(attempt)
            },
            onPartialText = { attempt, raw ->
                preview?.offer(attempt, raw)
                observer.onPartialText(attempt, raw)
            },
            onAttemptCompleted = { attempt, output ->
                preview?.offer(attempt, output.text)
                observer.onAttemptCompleted(attempt, output)
            },
        )
        capture = observed
        val result = try {
            conversion(observed, request)
        } finally {
            withContext(NonCancellable) { preview?.stop() }
        }
        completedSessionMetrics = try {
            provider.finishGenerationMetrics()
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: Exception) {
            // Telemetry is optional and must not turn a successful conversion into a failure.
            null
        }
        coroutineContext.ensureActive()
        withContext(Dispatchers.Default) { preview?.finish(result) }
        result
    } }

    override fun close() {
        if (closed.compareAndSet(false, true)) provider.close()
    }

    suspend fun closeAndAwait() {
        // Close first so a concurrent native generation is cancelled and can release [mutex].
        // Waiting for the mutex before closing would deadlock against a stalled provider call.
        close()
        withContext(NonCancellable) {
            mutex.withLock { provider.closeAndAwait() }
        }
    }

    companion object {
        private fun runtimePreparationFor(profile: GenUiConversionProfile): () -> Unit = when (profile) {
            GenUiConversionProfile.SOURCE_BOUND -> ({})
            GenUiConversionProfile.TRAINED_E2B_V10_W4 -> ({ LiteRtModelRunner.releaseCachedEngine() })
        }

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
