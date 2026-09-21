package com.samsung.genuicraft.sdk.provider

import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference

/** Runs on the runtime worker; native callbacks may arrive on a different thread. */
internal fun awaitGemma4Stream(
    start: (MessageCallback) -> Unit,
    cancel: () -> Unit,
    onPartialText: (String) -> Unit,
    isCancelled: () -> Boolean,
): String {
    val done = CountDownLatch(1)
    val failure = AtomicReference<Throwable?>(null)
    val text = StringBuilder()
    val lock = Any()
    val callback = object : MessageCallback {
        override fun onMessage(message: Message) {
            var cancelAfterCallback = false
            synchronized(lock) {
                if (done.count == 0L || failure.get() != null || isCancelled()) return
                // Content.Text is a delta, not the accumulated response. Reasoning channels are
                // separate and must not be inserted into the model's final IR.
                val delta = message.contents.contents.filterIsInstance<Content.Text>()
                    .joinToString("") { it.text }
                if (delta.isEmpty()) return
                text.append(delta)
                try {
                    onPartialText(text.toString())
                } catch (error: Throwable) {
                    failure.compareAndSet(null, error)
                    // Do not throw across JNI. Cancel decoding, then let the worker propagate it.
                    cancelAfterCallback = true
                }
            }
            // Native cancellation may synchronously deliver the terminal callback on another
            // thread. Do not hold the accumulator lock across that call.
            if (cancelAfterCallback) runCatching(cancel)
        }

        override fun onDone() = synchronized(lock) { done.countDown() }

        override fun onError(throwable: Throwable) = synchronized(lock) {
            failure.compareAndSet(null, throwable)
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
    return synchronized(lock) { text.toString() }
}
