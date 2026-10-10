package com.samsung.genuicraft.sdk

import com.google.gson.JsonParser
import java.util.UUID
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** Native callbacks only enqueue cumulative snapshots; compiler work never blocks native decode. */
internal class GenUiPreviewCoordinator(
    scope: CoroutineScope,
    private val sources: List<GenUiSource>,
    private val onSnapshot: (GenUiRenderSnapshot) -> Unit,
    private val startedNanos: Long = System.nanoTime(),
    private val updateIntervalMs: Long = UPDATE_INTERVAL_MS,
) {
    private data class Input(val attempt: Int, val raw: String)
    private val key = UUID.randomUUID().toString()
    private val input = Channel<Input>(Channel.CONFLATED)
    private var revision = 0L
    private val publicationLock = Any()
    @Volatile private var currentAttempt = 0
    @Volatile private var stopped = false
    private val worker = scope.launch(Dispatchers.Default) {
        var parsedAttempt = 0
        var compiler = GenUiExpressStreamCompiler()
        var lastPublicationNanos = 0L
        for (queued in input) {
            var update = queued
            if (stopped || update.attempt != currentAttempt) continue
            val since = (System.nanoTime() - lastPublicationNanos) / 1_000_000
            if (update.attempt == parsedAttempt && lastPublicationNanos != 0L && since < updateIntervalMs) {
                delay(updateIntervalMs - since)
            }
            // Native output is cumulative. After a throttle, compile the latest available
            // snapshot rather than publishing an older prefix and delaying the useful content.
            while (true) {
                val newer = input.tryReceive().getOrNull() ?: break
                update = newer
            }
            // A delayed/coalesced revision must not escape a replaced attempt.
            if (stopped || update.attempt != currentAttempt) continue
            if (update.attempt != parsedAttempt) {
                compiler = GenUiExpressStreamCompiler()
                parsedAttempt = update.attempt
                lastPublicationNanos = 0L
            }
            val preview = runCatching { compiler.accept(update.raw) }.getOrNull() ?: continue
            if (stopped || update.attempt != currentAttempt) continue
            val document = runCatching { SourceAttribution.preview(preview.document, sources) }
                .getOrDefault(preview.document)
            publish(update.attempt, document, preview.readyComponentCount, false)
            lastPublicationNanos = System.nanoTime()
        }
    }

    fun startAttempt(attempt: Int) = synchronized(publicationLock) { currentAttempt = attempt }

    fun offer(attempt: Int, raw: String) {
        if (!stopped && attempt == currentAttempt) input.trySend(Input(attempt, raw))
    }

    suspend fun finish(result: GenUiConversionResult) {
        stop()
        if (result is GenUiConversionResult.Success) {
            val count = runCatching {
                JsonParser.parseString(result.document.a2uiJson).asJsonArray.sumOf { message ->
                    message.asJsonObject.getAsJsonObject("updateComponents")
                        ?.getAsJsonArray("components")?.size() ?: 0
                }
            }.getOrDefault(0)
            publish(currentAttempt.coerceAtLeast(1), result.document, count, true)
        }
    }

    suspend fun stop() {
        synchronized(publicationLock) { stopped = true }
        input.close()
        worker.cancelAndJoin()
    }

    private fun publish(attempt: Int, document: GenUiDocument, count: Int, final: Boolean) = synchronized(publicationLock) {
        if (!final && (stopped || attempt != currentAttempt)) return@synchronized
        val snapshot = GenUiRenderSnapshot(
            surfaceKey = "$key:$attempt",
            attempt = attempt,
            revision = ++revision,
            document = document,
            elapsedMs = (System.nanoTime() - startedNanos) / 1_000_000,
            readyComponentCount = count,
            isFinal = final,
        )
        // A presentation observer is optional. Its failure must never cancel native inference.
        runCatching { onSnapshot(snapshot) }
    }

    companion object { private const val UPDATE_INTERVAL_MS = 100L }
}
