package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import java.util.concurrent.atomic.AtomicInteger

internal data class SdkStreamingAttemptSnapshot(
    val number: Int,
    val rawText: String,
    val complete: Boolean,
)

/**
 * Observes each SDK provider call without changing its generation or retry behavior.
 *
 * Partial text is the delegate's cumulative raw output. Callbacks run inline on the provider's
 * worker thread, so UI consumers must dispatch their own state updates to the main thread.
 * [attemptSnapshots] retains the latest exact raw text even if UI updates are throttled. Failed or
 * cancelled calls propagate unchanged and do not emit [onAttemptCompleted].
 */
internal class SdkStreamingProvider(
    private val delegate: GenUiProvider,
    private val onAttemptStarted: (Int) -> Unit,
    private val onPartialText: (Int, String) -> Unit,
    private val onAttemptCompleted: (Int, GenUiModelOutput) -> Unit,
) : GenUiProvider {
    override val id: String
        get() = delegate.id

    private val attemptCounter = AtomicInteger(0)
    private val snapshotsLock = Any()
    private val snapshotsByAttempt = linkedMapOf<Int, SdkStreamingAttemptSnapshot>()

    /** Lossless worker-thread capture for flushing throttled UI updates after generation stops. */
    internal val attemptSnapshots: List<SdkStreamingAttemptSnapshot>
        get() = synchronized(snapshotsLock) { snapshotsByAttempt.values.sortedBy { it.number } }

    override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
        generateObserved(prompt, downstreamPartialText = null)

    override suspend fun generate(
        prompt: GenUiPrompt,
        onPartialText: (String) -> Unit,
    ): GenUiModelOutput = generateObserved(prompt, onPartialText)

    private suspend fun generateObserved(
        prompt: GenUiPrompt,
        downstreamPartialText: ((String) -> Unit)?,
    ): GenUiModelOutput {
        val attempt = attemptCounter.incrementAndGet()
        recordSnapshot(attempt, rawText = "", complete = false)
        onAttemptStarted(attempt)

        val output = delegate.generate(prompt) { cumulativeRawText ->
            // Capture first so a downstream observer failure cannot discard the latest raw text.
            recordSnapshot(attempt, cumulativeRawText, complete = false)
            onPartialText(attempt, cumulativeRawText)
            downstreamPartialText?.invoke(cumulativeRawText)
        }
        recordSnapshot(attempt, output.text, complete = true)
        onAttemptCompleted(attempt, output)
        return output
    }

    private fun recordSnapshot(attempt: Int, rawText: String, complete: Boolean) {
        synchronized(snapshotsLock) {
            snapshotsByAttempt[attempt] = SdkStreamingAttemptSnapshot(attempt, rawText, complete)
        }
    }

    override fun close() = delegate.close()

    override suspend fun closeAndAwait() = delegate.closeAndAwait()
}
