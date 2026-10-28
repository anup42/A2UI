package com.samsung.genuicraft.sdk.provider

import com.samsung.genuicraft.sdk.GenUiGenerationSessionMetrics
import com.samsung.genuicraft.sdk.GenUiPrompt
import java.io.File
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

@OptIn(ExperimentalCoroutinesApi::class)
class ManagedGemmaRuntimePoolTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test fun `compatible leases reuse an engine and closing a lease leaves it warm`() = runTest {
        val config = config(modelFile())
        val harness = Harness(backgroundScope)
        val firstLease = harness.pool.lease(config)

        assertEquals("engine-1", firstLease.generate(PROMPT, 16).runtimeIdentity)
        firstLease.close()
        firstLease.awaitClosed()
        assertEquals(0, harness.created.single().closeCalls)

        val secondLease = harness.pool.lease(config)
        assertEquals("engine-1", secondLease.generate(PROMPT, 16).runtimeIdentity)
        assertEquals(1, harness.created.size)
        assertEquals(2, harness.created.single().generateCalls)
        assertEquals(0, harness.created.single().cancelCalls)

        secondLease.close()
        harness.pool.retire("test_cleanup")
        assertEquals(1, harness.created.single().closeCalls)
        assertEquals(1, harness.created.single().awaitClosedCalls)
    }

    @Test fun `different configuration drains old engine before creating replacement`() = runTest {
        val config = config(modelFile())
        var old: RecordingRuntime? = null
        val harness = Harness(backgroundScope, makeRuntime = { number ->
            if (number == 2) {
                val drained = checkNotNull(old)
                assertEquals(listOf("configuration_change"), drained.retireReasons)
                assertEquals(1, drained.closeCalls)
                assertEquals(1, drained.awaitClosedCalls)
            }
            RecordingRuntime(number).also { if (number == 1) old = it }
        })

        harness.pool.lease(config).generate(PROMPT, 16)
        harness.pool.lease(config.copy(enableSpeculativeDecoding = false)).generate(PROMPT, 16)

        assertEquals(2, harness.created.size)
        assertEquals(0, harness.created[1].closeCalls)
        harness.pool.retire("test_cleanup")
    }

    @Test fun `idle timeout uses virtual time and next request creates a fresh engine`() = runTest {
        val config = config(modelFile())
        val harness = Harness(backgroundScope, idleMillis = 1_000L)
        harness.pool.lease(config).generate(PROMPT, 16)
        runCurrent() // Start the idle timer at virtual time zero.

        advanceTimeBy(999L)
        runCurrent()
        assertEquals(0, harness.created.single().closeCalls)
        advanceTimeBy(1L)
        runCurrent()
        assertEquals(listOf("idle_timeout"), harness.created.single().retireReasons)
        assertEquals(1, harness.created.single().awaitClosedCalls)

        harness.pool.lease(config).generate(PROMPT, 16)
        assertEquals(2, harness.created.size)
        harness.pool.retire("test_cleanup")
    }

    @Test fun `prewarm skips factory while permission predicate is false`() = runTest {
        val config = config(modelFile())
        var allowed = false
        val harness = Harness(backgroundScope, allowPrewarm = { allowed })
        val lease = harness.pool.lease(config)

        lease.prewarm()
        assertTrue(harness.created.isEmpty())
        allowed = true
        lease.prewarm()
        assertEquals(1, harness.created.size)
        assertEquals(1, harness.created.single().prewarmCalls)
        assertEquals(0, harness.created.single().generateCalls)
        harness.pool.retire("test_cleanup")
    }

    @Test fun `retirement publishes engine lifetime metrics once after lease close`() = runTest {
        val config = config(modelFile())
        val harness = Harness(backgroundScope)
        val lease = harness.pool.lease(config)
        lease.generate(PROMPT, 16)
        lease.generate(PROMPT, 16)
        lease.close()

        assertTrue(harness.retirements.isEmpty())
        harness.pool.retire("host_release")
        assertEquals(1, harness.retirements.size)
        assertEquals("engine-1", harness.retirements.single().engineId)
        assertEquals("host_release", harness.retirements.single().retirementReason)
        assertEquals(2, harness.retirements.single().requestCount)
        assertEquals(listOf("host_release"), harness.created.single().retireReasons)
        assertEquals(1, harness.created.single().closeCalls)
    }

    @Test fun `closing queued lease cannot cancel another lease active request`() = runTest {
        val config = config(modelFile())
        val blocking = BlockingRuntime()
        val harness = Harness(backgroundScope, makeRuntime = { blocking })
        val activeLease = harness.pool.lease(config)
        val queuedLease = harness.pool.lease(config)

        val activeRequest = async { activeLease.generate(PROMPT, 16) }
        runCurrent()
        assertTrue(blocking.started.isCompleted)
        val queuedRequest = async { runCatching { queuedLease.generate(PROMPT, 16) } }
        runCurrent()

        queuedLease.close()
        assertEquals(0, blocking.cancelCalls)
        blocking.release.complete(Unit)
        runCurrent()

        assertEquals("engine-1", activeRequest.await().runtimeIdentity)
        val queuedFailure = queuedRequest.await().exceptionOrNull()
        assertTrue(queuedFailure is IllegalStateException)
        assertTrue(queuedFailure?.message.orEmpty().contains("lease is closed"))
        assertEquals(1, blocking.generateCalls)
        assertEquals(0, blocking.cancelCalls)
        queuedLease.awaitClosed()
        activeLease.close()
        harness.pool.retire("test_cleanup")
    }

    @Test fun `memory pressure retirement waits for active request and drains native engine`() = runTest {
        val config = config(modelFile())
        val blocking = BlockingDrainRuntime()
        val harness = Harness(backgroundScope, makeRuntime = { blocking })
        val lease = harness.pool.lease(config)
        val active = async { lease.generate(PROMPT, 16) }
        runCurrent()
        assertTrue(blocking.started.isCompleted)

        val retirement = async { harness.pool.retire("memory_pressure") }
        runCurrent()
        assertFalse(retirement.isCompleted)
        assertTrue(blocking.retireReasons.isEmpty())
        assertEquals(0, blocking.closeCalls)

        blocking.release.complete(Unit)
        runCurrent()
        assertEquals("engine-1", active.await().runtimeIdentity)
        assertTrue(blocking.drainStarted.isCompleted)
        assertFalse(retirement.isCompleted)
        assertEquals(1, blocking.closeCalls)
        blocking.finishDrain.complete(Unit)
        runCurrent()
        retirement.await()
        assertEquals(listOf("memory_pressure"), blocking.retireReasons)
        assertEquals(1, blocking.closeCalls)
        assertEquals(1, blocking.awaitClosedCalls)
    }

    @Test fun `cancelled generation finishes native drain before another engine is created`() = runTest {
        val config = config(modelFile())
        val interrupted = CancellableDrainRuntime()
        val harness = Harness(backgroundScope, makeRuntime = { number ->
            if (number == 1) interrupted else {
                assertEquals(1, interrupted.awaitClosedCalls)
                assertTrue(interrupted.finishDrain.isCompleted)
                RecordingRuntime(number)
            }
        })
        val firstLease = harness.pool.lease(config)
        val active = launch { firstLease.generate(PROMPT, 16) }
        runCurrent()
        assertTrue(interrupted.started.isCompleted)

        active.cancel()
        runCurrent()
        assertTrue(interrupted.drainStarted.isCompleted)
        assertEquals(listOf("request_interrupted"), interrupted.retireReasons)
        assertEquals(1, interrupted.closeCalls)
        assertFalse(active.isCompleted)

        val next = async { harness.pool.lease(config).generate(PROMPT, 16) }
        runCurrent()
        assertFalse(next.isCompleted)
        assertEquals(1, harness.created.size)

        interrupted.finishDrain.complete(Unit)
        runCurrent()
        active.join()
        assertEquals("engine-2", next.await().runtimeIdentity)
        assertEquals(2, harness.created.size)
        harness.pool.retire("test_cleanup")
    }

    @Test fun `optional metrics failure does not discard successful generated output`() = runTest {
        val config = config(modelFile())
        val metricsFailure = MetricsFailureRuntime()
        val harness = Harness(backgroundScope, makeRuntime = { metricsFailure })
        val lease = harness.pool.lease(config)

        val output = lease.generate(PROMPT, 16)
        assertEquals("result-1", output.text)
        assertEquals(1, metricsFailure.generateCalls)
        assertEquals(1, metricsFailure.finishCalls)
        assertNull(lease.finishGenerationMetrics())
        assertEquals(0, metricsFailure.closeCalls)
        harness.pool.retire("test_cleanup")
    }

    @Test fun `failed native drain blocks same and different config engine replacement`() = runTest {
        val config = config(modelFile())
        val badDrain = FailingDrainRuntime()
        val harness = Harness(backgroundScope, makeRuntime = { number ->
            check(number == 1) { "Factory must not create after a failed native drain." }
            badDrain
        })
        harness.pool.lease(config).generate(PROMPT, 16)

        val drainFailure = runCatching { harness.pool.retire("memory_pressure") }.exceptionOrNull()
        assertTrue(drainFailure is IllegalStateException)
        assertEquals("native shutdown failed", drainFailure?.message)
        assertEquals(1, badDrain.closeCalls)
        assertEquals(1, badDrain.awaitClosedCalls)

        listOf(config, config.copy(enableSpeculativeDecoding = false)).forEach { candidate ->
            val failure = runCatching { harness.pool.lease(candidate).generate(PROMPT, 16) }.exceptionOrNull()
            assertTrue(failure is IllegalStateException)
            assertTrue(failure?.message.orEmpty().contains("Native engine cleanup failed"))
            // Coroutine stack-trace recovery may copy the thrown exception across suspend calls.
            assertTrue(failure?.cause is IllegalStateException)
            assertEquals("native shutdown failed", failure?.cause?.message)
        }
        assertEquals(1, harness.created.size)
    }

    private fun modelFile(): File = temporary.newFile("pool-model.litertlm").apply {
        writeBytes(byteArrayOf(1, 2, 3))
    }

    private fun config(model: File) = ValidatedGemma4Config(
        modelPath = model.absolutePath,
        accelerator = Gemma4Accelerator.GPU,
        maxContextTokens = 8_192,
        maxOutputTokens = 2_048,
        cpuThreads = 4,
        cacheDir = null,
        enableThinking = false,
        thinkingTokenBudget = 0,
        enableSpeculativeDecoding = true,
        enableMetrics = true,
        gpuPrecision = Gemma4GpuPrecision.MODEL_DEFAULT,
    )

    private class Harness(
        scope: CoroutineScope,
        idleMillis: Long = 120_000L,
        allowPrewarm: () -> Boolean = { true },
        makeRuntime: (Int) -> RecordingRuntime = ::RecordingRuntime,
    ) {
        val created = mutableListOf<RecordingRuntime>()
        val retirements = mutableListOf<GenUiGenerationSessionMetrics>()
        val pool = ManagedGemmaRuntimePool(
            factory = Gemma4RuntimeFactory {
                makeRuntime(created.size + 1).also { created += it }
            },
            scope = scope,
            idleMillis = idleMillis,
            onRetirement = { retirements += it },
            allowPrewarm = allowPrewarm,
        )
    }

    private open class RecordingRuntime(private val number: Int) : Gemma4Runtime {
        val retireReasons = mutableListOf<String>()
        var generateCalls = 0
        var prewarmCalls = 0
        var cancelCalls = 0
        var closeCalls = 0
        var awaitClosedCalls = 0

        override suspend fun prewarm() { prewarmCalls++ }

        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput {
            generateCalls++
            return Gemma4RuntimeOutput("result-$number", "engine-$number", 1)
        }

        override suspend fun retire(reason: String): GenUiGenerationSessionMetrics {
            retireReasons += reason
            return GenUiGenerationSessionMetrics(
                speculativeDecodingEnabled = true,
                engineId = "engine-$number",
                retirementReason = reason,
                requestCount = generateCalls,
            )
        }

        override fun cancelActive() { cancelCalls++ }
        override fun close() { closeCalls++ }
        override suspend fun awaitClosed() { awaitClosedCalls++ }
    }

    private open class BlockingRuntime : RecordingRuntime(1) {
        val started = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()

        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput {
            started.complete(Unit)
            release.await()
            return super.generate(prompt, maxOutputTokens)
        }
    }

    private class BlockingDrainRuntime : BlockingRuntime() {
        val drainStarted = CompletableDeferred<Unit>()
        val finishDrain = CompletableDeferred<Unit>()

        override suspend fun awaitClosed() {
            super.awaitClosed()
            drainStarted.complete(Unit)
            finishDrain.await()
        }
    }

    private class CancellableDrainRuntime : RecordingRuntime(1) {
        val started = CompletableDeferred<Unit>()
        val drainStarted = CompletableDeferred<Unit>()
        val finishDrain = CompletableDeferred<Unit>()

        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput {
            started.complete(Unit)
            awaitCancellation()
        }

        override suspend fun awaitClosed() {
            super.awaitClosed()
            drainStarted.complete(Unit)
            finishDrain.await()
        }
    }

    private class MetricsFailureRuntime : RecordingRuntime(1) {
        var finishCalls = 0

        override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? {
            finishCalls++
            throw IllegalStateException("optional counters unavailable")
        }
    }

    private class FailingDrainRuntime : RecordingRuntime(1) {
        override suspend fun awaitClosed() {
            super.awaitClosed()
            throw IllegalStateException("native shutdown failed")
        }
    }

    private companion object {
        val PROMPT = GenUiPrompt(system = "", user = "Render a card", maxOutputTokens = 16)
    }
}
