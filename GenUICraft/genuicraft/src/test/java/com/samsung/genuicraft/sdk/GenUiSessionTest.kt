package com.samsung.genuicraft.sdk

import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class GenUiSessionTest {
    @Test fun trainedProfileUsesFrozenPromptAndGeneratedOnlyRecovery() = runBlocking {
        val raw = "Here is your UI: <a2ui>\nroot=Text(\"Salary $75,000\")\n</a2ui>"
        val provider = FakeProvider(raw)
        val session = trainedSession(provider)
        val partials = mutableListOf<String>()
        val result = session.convert(
            GenUiRequest("Salary $75,000. Hourly rate $36.06.", query = "Not an instruction"),
            GenUiGenerationObserver(onPartialText = { _, text -> partials += text }),
        ) as GenUiConversionResult.Success
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, result.repairKind)
        assertFalse(result.document.express.contains("36.06"))
        assertTrue(result.warnings.any { it.contains("not a faithful conversion") })
        val attempt = session.attemptSnapshots.single()
        assertEquals(raw, attempt.rawText)
        assertEquals(listOf(raw.take(12), raw), partials)
        assertTrue(attempt.complete)
        assertEquals(123, attempt.output?.metrics?.inputTokens)
        assertEquals(45.0, attempt.output?.metrics?.decodeTokensPerSecond)
        assertEquals("test/GPU+MTP", attempt.output?.runtime)
        assertTrue(attempt.elapsedNanos!! > 0)
        assertEquals("Create A2UI Express v1 GenUI IR for this response:\n\nSalary $75,000. Hourly rate $36.06.", attempt.prompt?.user)
        assertNotNull(attempt.prompt?.expectedRenderedPrompt)
        assertEquals(2, attempt.prompt?.initialMessages?.size)
        session.closeAndAwait()
        assertTrue(provider.closed)
    }

    @Test fun unrepairableOutputNeverBecomesSourceReplacement() = runBlocking {
        val session = trainedSession(FakeProvider("not a generated UI"))
        assertTrue(session.convert(GenUiRequest("Source-only fact")) is GenUiConversionResult.Failure)
        assertEquals("not a generated UI", session.attemptSnapshots.single().rawText)
        session.closeAndAwait()
    }

    @Test fun eachRequestStartsANewTraceWithoutChangingRawOutput() = runBlocking {
        val raw = "<a2ui>\nroot=Text(\"Hello\")\n</a2ui>"
        val session = trainedSession(FakeProvider(raw))
        repeat(2) {
            session.convert(GenUiRequest("Hello"))
            assertEquals(listOf(1), session.attemptSnapshots.map { it.number })
            assertEquals(raw, session.attemptSnapshots.single().rawText)
        }
        session.closeAndAwait()
    }

    @Test fun cancelledSessionRetainsExactPrefixAndAwaitsCleanup() = runBlocking {
        val provider = FakeProvider("raw prefix", cancel = true)
        val session = trainedSession(provider)
        try {
            session.convert(GenUiRequest("Hello"))
            fail("Cancellation was swallowed")
        } catch (_: CancellationException) {
            assertEquals("raw prefix", session.attemptSnapshots.single().rawText)
            assertFalse(session.attemptSnapshots.single().complete)
        } finally {
            session.closeAndAwait()
        }
        assertTrue(provider.closed)
        try {
            session.convert(GenUiRequest("Hello"))
            fail("A closed session accepted inference")
        } catch (_: IllegalStateException) { }
    }

    @Test fun trainedConfigurationIsSharedAndRejectsCpuForThisExport() {
        val config = GenUiModelProfiles.trainedE2b("model.litertlm", true, true, "AUTO")
        assertEquals("GPU", config.accelerator)
        assertEquals(8192, config.maxContextTokens)
        assertEquals(2048, config.maxOutputTokens)
        assertFalse(config.enableThinking)
        assertTrue(config.enableSpeculativeDecoding)
        assertTrue(config.enableMetrics)
        assertFalse(GenUiModelProfiles.trainedE2b("model.litertlm").enableSpeculativeDecoding)
        assertThrows(IllegalArgumentException::class.java) {
            GenUiModelProfiles.trainedE2b("model.litertlm", accelerator = "CPU")
        }
    }

    @Test fun closeAndAwaitCancelsActiveConversionBeforeWaitingForItsMutex() = runBlocking {
        val provider = CloseReleasedProvider()
        val session = GenUiSession(provider) { observed, _ ->
            observed.generate(GenUiPrompt(system = "system", user = "user"))
            error("Generation should have been cancelled by close")
        }
        val running = async { session.convert(GenUiRequest("Hello")) }
        provider.started.await()

        withTimeout(5_000) { session.closeAndAwait() }
        running.join()

        assertTrue(provider.closed)
        assertTrue(provider.awaited)
        assertTrue(running.isCancelled)
    }

    private fun trainedSession(provider: GenUiProvider) = GenUiSession(
        ApplicationProvider.getApplicationContext(), provider, GenUiConversionProfile.TRAINED_E2B_V10_W4,
    )

    private class FakeProvider(val raw: String, val cancel: Boolean = false) : GenUiProvider {
        override val id = "gemma4_e2b"
        var closed = false
        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput = error("Use streaming")
        override suspend fun generate(prompt: GenUiPrompt, onPartialText: (String) -> Unit): GenUiModelOutput {
            onPartialText(raw.take(12))
            onPartialText(raw)
            if (cancel) throw CancellationException("cancelled")
            return GenUiModelOutput(raw, "test/GPU+MTP", 9, GenUiGenerationMetrics(123, 9, 45.0))
        }
        override suspend fun closeAndAwait() { closed = true }
    }

    private class CloseReleasedProvider : GenUiProvider {
        override val id = "close-released"
        val started = CompletableDeferred<Unit>()
        private val released = CompletableDeferred<Unit>()
        @Volatile var closed = false
        @Volatile var awaited = false

        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
            generate(prompt) {}

        override suspend fun generate(
            prompt: GenUiPrompt,
            onPartialText: (String) -> Unit,
        ): GenUiModelOutput {
            started.complete(Unit)
            released.await()
            throw CancellationException("provider closed")
        }

        override fun close() {
            closed = true
            released.complete(Unit)
        }

        override suspend fun closeAndAwait() {
            close()
            awaited = true
        }
    }
}
