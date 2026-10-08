package com.samsung.genuicraft.sdk

import java.util.concurrent.CopyOnWriteArrayList
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test

class GenUiProgressiveSessionTest {
    private val prefix = "<a2ui>\nroot=Column([title,body])\ntitle=Text(\"Early title\")\n"
    private val full = prefix + "body=Text(\"Final body\")\n</a2ui>"

    @Test fun nativePreviewArrivesBeforeProviderFinishesAndFinalMatchesNormalCompiler() = runBlocking {
        val first = CompletableDeferred<GenUiRenderSnapshot>()
        val snapshots = CopyOnWriteArrayList<GenUiRenderSnapshot>()
        val provider = provider { emit ->
            emit(prefix)
            // Waiting here proves the first preview precedes final model output. Later
            // provisional revisions may arrive while the completed output is being validated.
            val early = withTimeout(5_000) { first.await() }
            assertFalse(early.isFinal)
            assertTrue(early.document.express.contains("Early title"))
            assertFalse(early.document.express.contains("Final body"))
            emit(full)
            GenUiModelOutput(full, "fake")
        }
        val session = session(provider)
        val result = session.convert(GenUiRequest("test"), GenUiGenerationObserver(onRenderSnapshot = {
            snapshots += it
            if (!it.isFinal) {
                first.complete(it)
            }
        })) as GenUiConversionResult.Success
        assertEquals(GenUiCompiler.compile(full), result.document)
        assertEquals(full, session.attemptSnapshots.single().rawText)
        assertTrue(snapshots.last().isFinal)
        assertEquals(snapshots.first().surfaceKey, snapshots.last().surfaceKey)
        assertTrue(snapshots.zipWithNext().all { (a, b) -> a.revision < b.revision })
        assertEquals(result.document, snapshots.last().document)
    }

    @Test fun disablingPreviewsDoesNotChangeOutputOrRawStreaming() = runBlocking {
        var renderEvents = 0
        val raw = mutableListOf<String>()
        val session = session(provider { emit -> emit(prefix); emit(full); GenUiModelOutput(full, "fake") })
        val result = session.convert(GenUiRequest("test"), GenUiGenerationObserver(
            onPartialText = { _, text -> raw += text }, onRenderSnapshot = { renderEvents++ },
        ), enableStreamingRendering = false) as GenUiConversionResult.Success
        assertEquals(0, renderEvents)
        assertEquals(listOf(prefix, full), raw)
        assertEquals(GenUiCompiler.compile(full), result.document)
    }

    @Test fun previewObserverExceptionCannotCancelModelOrFinalCompilation() = runBlocking {
        val session = session(provider { emit -> emit(prefix); delay(150); emit(full); GenUiModelOutput(full, "fake") })
        val result = session.convert(GenUiRequest("test"), GenUiGenerationObserver(
            onRenderSnapshot = { error("presentation observer failed") },
        ))
        assertTrue(result is GenUiConversionResult.Success)
        assertEquals(full, session.attemptSnapshots.single().rawText)
    }

    @Test fun cancellationRetainsRawButNeverPublishesFinalOrLatePreview() = runBlocking {
        val first = CompletableDeferred<Unit>()
        val snapshots = CopyOnWriteArrayList<GenUiRenderSnapshot>()
        val session = session(provider { emit ->
            emit(prefix)
            withTimeout(5_000) { first.await() }
            throw CancellationException("cancelled")
        })
        try {
            session.convert(GenUiRequest("test"), GenUiGenerationObserver(onRenderSnapshot = {
                snapshots += it
                first.complete(Unit)
            }))
            fail("Cancellation swallowed")
        } catch (_: CancellationException) { }
        val count = snapshots.size
        delay(150)
        assertEquals(count, snapshots.size)
        assertTrue(snapshots.none { it.isFinal })
        assertEquals(prefix, session.attemptSnapshots.single().rawText)
    }

    @Test fun metricsFinalizationCancellationNeverPublishesAcceptedFinal() = runBlocking {
        val first = CompletableDeferred<Unit>()
        val snapshots = CopyOnWriteArrayList<GenUiRenderSnapshot>()
        var metricsFinalizationCalled = false
        val generationProvider = provider { emit ->
            emit(prefix)
            withTimeout(5_000) { first.await() }
            emit(full)
            GenUiModelOutput(full, "fake")
        }
        val cancellingMetricsProvider = object : GenUiProvider by generationProvider {
            override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? {
                metricsFinalizationCalled = true
                throw CancellationException("Cancelled while finalizing metrics")
            }
        }
        val session = session(cancellingMetricsProvider)
        try {
            session.convert(GenUiRequest("test"), GenUiGenerationObserver(onRenderSnapshot = {
                snapshots += it
                if (!it.isFinal) first.complete(Unit)
            }))
            fail("Metrics cancellation was swallowed")
        } catch (cancelled: CancellationException) {
            assertEquals("Cancelled while finalizing metrics", cancelled.message)
        }

        assertTrue(metricsFinalizationCalled)
        assertTrue("Expected a provisional native preview before cancellation", snapshots.isNotEmpty())
        assertTrue("A cancelled conversion must not enable final interactions", snapshots.none { it.isFinal })
        val attempts = session.attemptSnapshots
        assertEquals(full, attempts.single().rawText)
        assertTrue("The model completed before metrics cancellation", attempts.single().complete)
        assertNotNull(attempts.single().output)
        assertNull(session.generationSessionMetrics)
        val count = snapshots.size
        delay(150)
        assertEquals("Preview work escaped conversion cleanup", count, snapshots.size)
    }

    @Test fun unrepairableFinalDoesNotBecomeSuccessBecausePreviewWasShown() = runBlocking {
        val first = CompletableDeferred<Unit>()
        val snapshots = CopyOnWriteArrayList<GenUiRenderSnapshot>()
        val provider = provider { emit ->
            emit(prefix)
            withTimeout(5_000) { first.await() }
            GenUiModelOutput("invalid final", "fake")
        }
        val session = GenUiSession(provider) { observed, _ ->
            observed.generate(GenUiPrompt("", ""))
            GenUiConversionResult.Failure("invalid final", "fake", 1, 1, "invalid final")
        }
        assertTrue(session.convert(GenUiRequest("test"), GenUiGenerationObserver(onRenderSnapshot = {
            snapshots += it
            first.complete(Unit)
        })) is GenUiConversionResult.Failure)
        assertTrue(snapshots.isNotEmpty())
        assertTrue(snapshots.none { it.isFinal })
    }

    private fun session(provider: GenUiProvider) = GenUiSession(provider) { observed, _ ->
        val output = observed.generate(GenUiPrompt("", ""))
        GenUiConversionResult.Success(GenUiCompiler.compile(output.text), "fake", 1, 1)
    }

    private fun provider(generate: suspend ((String) -> Unit) -> GenUiModelOutput) = object : GenUiProvider {
        override val id = "fake"
        override suspend fun generate(prompt: GenUiPrompt) = error("Use streaming")
        override suspend fun generate(prompt: GenUiPrompt, onPartialText: (String) -> Unit) = generate(onPartialText)
    }
}
