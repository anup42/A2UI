package com.samsung.genuicraft.sdk.provider

import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread
import org.junit.Assert.*
import org.junit.Test

class Gemma4StreamingTest {
    @Test fun `native deltas become exact cumulative snapshots before completion`() {
        val snapshots = mutableListOf<String>()
        var complete = false
        val raw = awaitGemma4Stream(
            start = { callback ->
                callback.onMessage(Message.model("  <a2ui>\n"))
                callback.onMessage(Message.model("root=Text(\"Sunny ☀\")"))
                callback.onMessage(Message.model("\n</a2ui>  "))
                assertEquals(3, snapshots.size)
                complete = true
                callback.onDone()
                callback.onMessage(Message.model("must be ignored"))
            },
            cancel = { fail("Not cancelled") },
            onPartialText = { assertFalse(complete); snapshots += it },
            isCancelled = { false },
        )
        assertEquals("  <a2ui>\nroot=Text(\"Sunny ☀\")\n</a2ui>  ", raw)
        assertEquals(raw, snapshots.last())
        assertTrue(snapshots.zipWithNext().all { (a, b) -> b.startsWith(a) && b.length > a.length })
    }

    @Test fun `native stream failure preserves earlier snapshots and propagates`() {
        val failure = IllegalStateException("native failure")
        val snapshots = mutableListOf<String>()
        val actual = assertThrows(IllegalStateException::class.java) {
            awaitGemma4Stream(
                start = { it.onMessage(Message.model("partial")); it.onError(failure) },
                cancel = {}, onPartialText = snapshots::add, isCancelled = { false },
            )
        }
        assertSame(failure, actual)
        assertEquals(listOf("partial"), snapshots)
    }

    @Test fun `observer failure cancels native generation without escaping callback`() {
        lateinit var callback: MessageCallback
        var cancelled = false
        val actual = assertThrows(IllegalArgumentException::class.java) {
            awaitGemma4Stream(
                start = { callback = it; it.onMessage(Message.model("partial")) },
                cancel = { cancelled = true; callback.onDone() },
                onPartialText = { throw IllegalArgumentException("observer failed") },
                isCancelled = { false },
            )
        }
        assertTrue(cancelled)
        assertEquals("observer failed", actual.message)
    }

    @Test fun `interrupted worker waits for native terminal callback before releasing caller`() {
        val started = CountDownLatch(1)
        val cancellationRequested = CountDownLatch(1)
        val callback = AtomicReference<MessageCallback>()
        val result = AtomicReference<Throwable?>()
        val worker = thread {
            try {
                awaitGemma4Stream(
                    start = { callback.set(it); started.countDown() },
                    cancel = { cancellationRequested.countDown() },
                    onPartialText = {}, isCancelled = { false },
                )
            } catch (error: Throwable) { result.set(error) }
        }
        try {
            assertTrue(started.await(2, TimeUnit.SECONDS))
            worker.interrupt()
            assertTrue(cancellationRequested.await(2, TimeUnit.SECONDS))
            assertTrue("Must not close while native work is active", worker.isAlive)
            callback.get().onDone()
            worker.join(2_000)
            assertFalse(worker.isAlive)
            assertTrue(result.get() is CancellationException)
        } finally {
            callback.get()?.onDone()
            worker.join(2_000)
        }
    }
}
