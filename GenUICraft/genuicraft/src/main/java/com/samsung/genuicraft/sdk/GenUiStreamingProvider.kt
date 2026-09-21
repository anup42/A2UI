package com.samsung.genuicraft.sdk

import java.util.concurrent.atomic.AtomicInteger

/** Lossless streaming, prompt, runtime and metric capture shared by SDK clients. */
class GenUiStreamingProvider(
    private val delegate: GenUiProvider,
    private val onAttemptStarted: (Int) -> Unit = {},
    private val onPartialText: (Int, String) -> Unit = { _, _ -> },
    private val onAttemptCompleted: (Int, GenUiModelOutput) -> Unit = { _, _ -> },
) : GenUiProvider {
    override val id: String get() = delegate.id
    private val attemptCounter = AtomicInteger(0)
    private val lock = Any()
    private val snapshots = linkedMapOf<Int, GenUiGenerationAttempt>()

    val attemptSnapshots: List<GenUiGenerationAttempt>
        get() = synchronized(lock) { snapshots.values.sortedBy { it.number } }

    override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput = observed(prompt, null)

    override suspend fun generate(prompt: GenUiPrompt, onPartialText: (String) -> Unit): GenUiModelOutput =
        observed(prompt, onPartialText)

    private suspend fun observed(prompt: GenUiPrompt, downstream: ((String) -> Unit)?): GenUiModelOutput {
        val attempt = attemptCounter.incrementAndGet()
        val started = System.nanoTime()
        update(attempt) { GenUiGenerationAttempt(attempt, "", false, prompt) }
        try {
            onAttemptStarted(attempt)
            val output = delegate.generate(prompt) { rawText ->
                update(attempt) { it.copy(rawText = rawText) }
                onPartialText(attempt, rawText)
                downstream?.invoke(rawText)
            }
            update(attempt) { it.copy(rawText = output.text, complete = true, output = output) }
            onAttemptCompleted(attempt, output)
            return output
        } finally {
            update(attempt) { it.copy(elapsedNanos = (System.nanoTime() - started).coerceAtLeast(0)) }
        }
    }

    private fun update(number: Int, change: (GenUiGenerationAttempt) -> GenUiGenerationAttempt) {
        synchronized(lock) {
            snapshots[number] = change(snapshots[number] ?: GenUiGenerationAttempt(number, "", false))
        }
    }

    override fun close() = delegate.close()
    override suspend fun closeAndAwait() = delegate.closeAndAwait()
}
