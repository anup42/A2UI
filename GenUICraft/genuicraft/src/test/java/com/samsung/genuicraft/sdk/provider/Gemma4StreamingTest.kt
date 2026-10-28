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
    @Test fun `opted in valid Express stops after split closing tag and keeps exact native delta`() {
        lateinit var callback: MessageCallback
        val snapshots = mutableListOf<String>()
        var cancellations = 0
        val first = "  <a2ui>\nroot=Text(\"Sunny\")\n</a"
        val last = "2ui> \n"
        val result = awaitGemma4Stream(
            start = {
                callback = it
                it.onMessage(Message.model(first))
                assertEquals(0, cancellations)
                it.onMessage(Message.model(last))
                it.onMessage(Message.model("ignored after cancellation"))
                it.onDone()
            },
            cancel = {
                cancellations++
                callback.onError(CancellationException("completion cutoff"))
            },
            onPartialText = snapshots::add,
            isCancelled = { false },
            stopAtCompleteExpress = true,
        )

        assertEquals(1, cancellations)
        assertEquals(first + last, result.text)
        assertEquals(listOf(first, first + last), snapshots)
        assertTrue(result.completionStop)
        assertNull(result.repetitionStop)
    }

    @Test fun `complete Express does not stop when opt in is absent`() {
        var cancelled = false
        val result = awaitGemma4Stream(
            start = {
                it.onMessage(Message.model("<a2ui>\nroot=Text(\"Hi\")\n</a2ui>"))
                it.onDone()
            },
            cancel = { cancelled = true },
            onPartialText = {},
            isCancelled = { false },
        )
        assertFalse(cancelled)
        assertFalse(result.completionStop)
    }

    @Test fun `quoted commented and fenced delimiters cannot trigger completion`() {
        val prefixes = listOf(
            "<a2ui>\nroot=Text(\"</a2ui>",
            "<a2ui>\nroot=Text(r\"</a2ui>",
            "<a2ui>\nroot=Text(\"\"\"</a2ui>",
            "<a2ui>\nroot=Text(\"Hi\")\n# </a2ui>",
            "<a2ui>\nroot=Text(\"Hi\")\n// </a2ui>",
            "<a2ui>\nroot=Text(\"Hi\")\n/* </a2ui>",
            "```a2ui\n<a2ui>\nroot=Text(\"Hi\")\n</a2ui>\n```",
        )
        prefixes.forEach { prefix ->
            assertFalse(prefix, GenUiExpressCompletionDetector().inspect(prefix))
        }
        val quotedThenRealClose = "<a2ui>\nroot=Text(\"</a2ui>\")\n</a2ui>"
        val detector = GenUiExpressCompletionDetector()
        assertFalse(detector.inspect("<a2ui>\nroot=Text(\"</a2ui>"))
        assertTrue(detector.inspect(quotedThenRealClose))
    }

    @Test fun `syntactically closed but invalid Express does not cancel decode`() {
        val invalid = "<a2ui>\nroot=Unsupported(\"Hi\")\n</a2ui>"
        var cancelled = false
        val result = awaitGemma4Stream(
            start = {
                it.onMessage(Message.model(invalid))
                it.onDone()
            },
            cancel = { cancelled = true },
            onPartialText = {},
            isCancelled = { false },
            stopAtCompleteExpress = true,
        )
        assertFalse(cancelled)
        assertFalse(result.completionStop)
        assertEquals(invalid, result.text)
    }

    @Test fun `nonwhitespace trailing text is retained without a false completion stop`() {
        val raw = "<a2ui>\nroot=Text(\"Hi\")\n</a2ui> extra"
        var cancelled = false
        val result = awaitGemma4Stream(
            start = {
                it.onMessage(Message.model(raw))
                it.onDone()
            },
            cancel = { cancelled = true },
            onPartialText = {},
            isCancelled = { false },
            stopAtCompleteExpress = true,
        )
        assertFalse(cancelled)
        assertFalse(result.completionStop)
        assertEquals(raw, result.text)
    }

    @Test fun `user cancellation still wins after a complete Express callback`() {
        lateinit var callback: MessageCallback
        var userCancelled = false
        val actual = assertThrows(CancellationException::class.java) {
            awaitGemma4Stream(
                start = {
                    callback = it
                    it.onMessage(Message.model("<a2ui>\nroot=Text(\"Hi\")\n</a2ui>"))
                },
                cancel = { callback.onError(CancellationException("native cancellation")) },
                onPartialText = { userCancelled = true },
                isCancelled = { userCancelled },
                stopAtCompleteExpress = true,
            )
        }
        assertEquals("Gemma 4 streaming request cancelled.", actual.message)
    }

    @Test fun `native deltas become exact cumulative snapshots before completion`() {
        val snapshots = mutableListOf<String>()
        var complete = false
        val result = awaitGemma4Stream(
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
        assertEquals("  <a2ui>\nroot=Text(\"Sunny ☀\")\n</a2ui>  ", result.text)
        assertNull(result.repetitionStop)
        assertEquals(result.text, snapshots.last())
        assertTrue(snapshots.zipWithNext().all { (a, b) -> b.startsWith(a) && b.length > a.length })
    }

    @Test fun `twentieth repeated child reference cancels decode and returns partial output for repair`() {
        lateinit var callback: MessageCallback
        var cancelled = false
        val partial = "<a2ui>\nroot=Column([" + List(20) { "aa" }.joinToString(",")
        val result = awaitGemma4Stream(
            start = {
                callback = it
                it.onMessage(Message.model(partial))
            },
            cancel = {
                cancelled = true
                callback.onError(CancellationException("intentional repetition cutoff"))
            },
            onPartialText = {},
            isCancelled = { false },
        )

        assertTrue(cancelled)
        assertEquals(partial, result.text)
        assertEquals(20, result.repetitionStop?.repeatLimit)
        assertTrue(result.repetitionStop?.detail.orEmpty().contains("'aa'"))
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

    @Test fun `unique child lists longer than twenty complete without native cancellation`() {
        val ids = ('a'..'z').map(Char::toString)
        var cancelled = false
        val parts = mutableListOf("<a2ui>\nroot=Column([").apply {
            ids.forEachIndexed { index, id -> add((if (index == 0) "" else ",") + id) }
            add("])\n")
            ids.forEach { add("$it=Text(\"Item $it\")\n") }
            add("</a2ui>")
        }
        val result = awaitGemma4Stream(
            start = { callback ->
                parts.forEach { callback.onMessage(Message.model(it)) }
                callback.onDone()
            },
            cancel = { cancelled = true },
            onPartialText = {},
            isCancelled = { false },
        )
        assertFalse(cancelled)
        assertNull(result.repetitionStop)
        assertEquals(parts.joinToString(""), result.text)
        assertTrue(com.samsung.genuicraft.sdk.GenUiCompiler.compile(result.text).a2uiJson.isNotBlank())
    }
}
