package com.samsung.genuicraft.sdk

import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
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
    @Test fun structuredViewSkipsProviderAndPublishesFinalSnapshotWithoutModelAttempts() = runBlocking {
        val provider = FakeProvider("must not generate")
        var converterCalls = 0
        val session = GenUiSession(provider) { _, _ ->
            converterCalls++
            error("Structured input must not reach the model converter")
        }
        val snapshots = mutableListOf<GenUiRenderSnapshot>()
        val input = """[{"${'$'}type":"Markdown","value":"## Day 1"}]"""
        val result = session.convert(
            GenUiRequest(input),
            GenUiGenerationObserver(onRenderSnapshot = { snapshots += it }),
        ) as GenUiConversionResult.Success
        assertEquals("sdk-structured-input", result.provider)
        assertEquals(0, result.attempts)
        assertEquals(0, provider.generateCount)
        assertEquals(0, provider.metricsFinalizationCount)
        assertEquals(0, converterCalls)
        assertTrue(session.attemptSnapshots.isEmpty())
        assertNull(session.generationSessionMetrics)
        assertEquals(1, snapshots.size)
        assertTrue(snapshots.single().isFinal)
        assertEquals(0, snapshots.single().attempt)
        assertEquals(result.document, snapshots.single().document)
        session.closeAndAwait()
    }

    @Test fun prewarmDoesNotConvertGenerateOrFinalizeMetrics() = runBlocking {
        val provider = FakeProvider("unused")
        var conversionCalls = 0
        val session = GenUiSession(provider) { _, _ ->
            conversionCalls++
            error("prewarm must not invoke conversion")
        }

        session.prewarm()

        assertEquals(1, provider.prewarmCount)
        assertEquals(0, provider.generateCount)
        assertEquals(0, conversionCalls)
        assertEquals(0, provider.metricsFinalizationCount)
        assertTrue(session.attemptSnapshots.isEmpty())
        assertNull(session.generationSessionMetrics)
        session.closeAndAwait()
        assertTrue(provider.closed)
    }

    @Test fun prewarmSerializesWithConversionAndCancellationReleasesMutex() = runBlocking {
        val provider = BlockingPrewarmProvider()
        val conversionEntered = CompletableDeferred<Unit>()
        val session = GenUiSession(provider) { _, _ ->
            conversionEntered.complete(Unit)
            GenUiConversionResult.Failure("test", provider.id, 0, 0)
        }
        val warming = async { session.prewarm() }
        provider.prewarmStarted.await()
        val converting = async(start = CoroutineStart.UNDISPATCHED) {
            session.convert(GenUiRequest("test"))
        }
        assertFalse(conversionEntered.isCompleted)

        warming.cancelAndJoin()
        assertTrue(warming.isCancelled)
        assertTrue(provider.prewarmFinished)
        assertTrue(converting.await() is GenUiConversionResult.Failure)
        assertTrue(conversionEntered.isCompleted)
        withTimeout(5_000) { session.closeAndAwait() }
        assertTrue(provider.closed)
    }

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

    @Test fun gemma270ProfileStreamsGeneratedOnlyOutputWithNativeTemplateAndFidelityWarning() = runBlocking {
        val raw = "Here is your UI: <a2ui>\nroot=Text(\"Salary $75,000\")\n</a2ui>"
        val provider = FakeProvider(raw, providerId = "gemma3_270m_a2ui_express")
        val session = GenUiSession(
            ApplicationProvider.getApplicationContext(), provider,
            GenUiConversionProfile.TRAINED_GEMMA3_270M,
        )
        val partials = mutableListOf<String>()
        val result = session.convert(
            GenUiRequest("Salary $75,000. Hourly rate $36.06."),
            GenUiGenerationObserver(onPartialText = { _, text -> partials += text }),
        ) as GenUiConversionResult.Success

        assertEquals(Gemma270PromptContract.PROFILE, result.document.profile)
        assertEquals("gemma3_270m_a2ui_express", result.provider)
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, result.repairKind)
        assertTrue(result.warnings.any { it.contains("not a faithful conversion") })
        assertFalse(result.document.express.contains("36.06"))
        assertEquals(listOf(raw.take(12), raw), partials)
        assertEquals(raw, session.attemptSnapshots.single().rawText)
        assertEquals(Gemma270PromptContract.EXAMPLE_MODEL,
            session.attemptSnapshots.single().prompt?.initialMessages?.last()?.text)
        assertNull(session.attemptSnapshots.single().prompt?.chatTemplateOverride)
        assertNull(session.attemptSnapshots.single().prompt?.expectedRenderedPrompt)
        session.closeAndAwait()
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

    @Test fun finalizesSessionMetricsAfterAllConversionAttempts() = runBlocking {
        val raw = "<a2ui>\nroot=Text(\"Hello\")\n</a2ui>"
        val provider = FakeProvider(
            raw,
            finalizedMetrics = GenUiGenerationSessionMetrics(
                speculativeDecodingEnabled = true,
                drafterAcceptanceRate = 0.558642,
            ),
        )
        val session = trainedSession(provider)

        session.convert(GenUiRequest("Hello"))

        assertEquals(1, provider.metricsFinalizationCount)
        assertEquals(true, session.generationSessionMetrics?.speculativeDecodingEnabled)
        assertEquals(0.558642, session.generationSessionMetrics?.drafterAcceptanceRate!!, 0.000001)
        session.closeAndAwait()
    }

    @Test fun unavailableSessionTelemetryDoesNotFailSuccessfulConversion() = runBlocking {
        val raw = "<a2ui>\nroot=Text(\"Hello\")\n</a2ui>"
        val provider = FakeProvider(raw, failMetricsFinalization = true)
        val session = trainedSession(provider)

        val result = session.convert(GenUiRequest("Hello"))

        assertTrue(result is GenUiConversionResult.Success)
        assertEquals(1, provider.metricsFinalizationCount)
        assertNull(session.generationSessionMetrics)
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
        assertEquals(com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision.FP32, config.gpuPrecision)
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

    private class FakeProvider(
        val raw: String,
        val cancel: Boolean = false,
        val finalizedMetrics: GenUiGenerationSessionMetrics? = null,
        val failMetricsFinalization: Boolean = false,
        val providerId: String = "gemma4_e2b",
    ) : GenUiProvider {
        override val id = providerId
        var closed = false
        var prewarmCount = 0
        var generateCount = 0
        var metricsFinalizationCount = 0
        override suspend fun prewarm() { prewarmCount++ }
        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput = error("Use streaming")
        override suspend fun generate(prompt: GenUiPrompt, onPartialText: (String) -> Unit): GenUiModelOutput {
            generateCount++
            onPartialText(raw.take(12))
            onPartialText(raw)
            if (cancel) throw CancellationException("cancelled")
            return GenUiModelOutput(raw, "test/GPU+MTP", 9, GenUiGenerationMetrics(123, 9, 45.0))
        }
        override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? {
            metricsFinalizationCount++
            if (failMetricsFinalization) error("telemetry unavailable")
            return finalizedMetrics
        }
        override suspend fun closeAndAwait() { closed = true }
    }

    private class BlockingPrewarmProvider : GenUiProvider {
        override val id = "blocking-prewarm"
        val prewarmStarted = CompletableDeferred<Unit>()
        @Volatile var prewarmFinished = false
        @Volatile var closed = false

        override suspend fun prewarm() {
            prewarmStarted.complete(Unit)
            try {
                CompletableDeferred<Unit>().await()
            } finally {
                prewarmFinished = true
            }
        }

        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
            error("prewarm must not generate")

        override fun close() { closed = true }
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
