package com.samsung.genuicraft.sdk.provider

import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference

internal data class Gemma4StreamResult(
    val text: String,
    val repetitionStop: GenUiRepetitionStop? = null,
)

/** Runs on the runtime worker; native callbacks may arrive on a different thread. */
internal fun awaitGemma4Stream(
    start: (MessageCallback) -> Unit,
    cancel: () -> Unit,
    onPartialText: (String) -> Unit,
    isCancelled: () -> Boolean,
    repetitionLimit: Int = GenUiOutputRepetitionGuard.DEFAULT_REPEAT_LIMIT,
): Gemma4StreamResult {
    val done = CountDownLatch(1)
    val failure = AtomicReference<Throwable?>(null)
    val repetitionStop = AtomicReference<GenUiRepetitionStop?>(null)
    val repetitionGuard = GenUiOutputRepetitionGuard(repetitionLimit)
    val text = StringBuilder()
    val lock = Any()
    val callback = object : MessageCallback {
        override fun onMessage(message: Message) {
            var cancelAfterCallback = false
            synchronized(lock) {
                if (done.count == 0L || failure.get() != null || repetitionStop.get() != null || isCancelled()) return
                // Content.Text is a delta, not the accumulated response. Reasoning channels are
                // separate and must not be inserted into the model's final IR.
                val delta = message.contents.contents.filterIsInstance<Content.Text>()
                    .joinToString("") { it.text }
                if (delta.isEmpty()) return
                text.append(delta)
                val snapshot = text.toString()
                try {
                    onPartialText(snapshot)
                } catch (error: Throwable) {
                    failure.compareAndSet(null, error)
                    // Do not throw across JNI. Cancel decoding, then let the worker propagate it.
                    cancelAfterCallback = true
                }
                if (failure.get() == null) {
                    repetitionGuard.inspect(snapshot)?.let { stop ->
                        if (repetitionStop.compareAndSet(null, stop)) cancelAfterCallback = true
                    }
                }
            }
            // Native cancellation may synchronously deliver the terminal callback on another
            // thread. Do not hold the accumulator lock across that call.
            if (cancelAfterCallback) runCatching(cancel)
        }

        override fun onDone() = synchronized(lock) { done.countDown() }

        override fun onError(throwable: Throwable) = synchronized(lock) {
            // cancelProcess commonly reports cancellation through onError. When this helper
            // initiated cancellation for a repetition loop, the accumulated text is successful
            // partial output and must continue into the compiler's repair path.
            if (repetitionStop.get() == null) failure.compareAndSet(null, throwable)
            done.countDown()
        }
    }
    start(callback)
    var interrupted = false
    // Cancellation interrupts the worker. Wait for native termination before the caller closes
    // the conversation or engine, so no callback can use an already released native handle.
    while (true) {
        try {
            done.await()
            break
        } catch (_: InterruptedException) {
            interrupted = true
            runCatching(cancel)
        }
    }
    if (interrupted) Thread.currentThread().interrupt()
    if (interrupted || isCancelled()) throw CancellationException("Gemma 4 streaming request cancelled.")
    failure.get()?.let { throw it }
    return synchronized(lock) { Gemma4StreamResult(text.toString(), repetitionStop.get()) }
}
