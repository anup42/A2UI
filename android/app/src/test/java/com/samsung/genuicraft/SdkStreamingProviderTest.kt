package com.samsung.genuicraft

import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class SdkStreamingProviderTest {
    @Test
    fun cumulativePartialsArriveBeforeExactCompletion() = runBlocking {
        val exactOutput = "\n  {\"root\":\"screen\"}  \n"
        val firstSnapshot = exactOutput.take(12)
        val expectedOutput = GenUiModelOutput(exactOutput, "test/worker")
        val events = mutableListOf<String>()
        val delegate = streamingDelegate { onPartialText ->
            onPartialText(firstSnapshot)
            onPartialText(exactOutput)
            events += "delegate-return"
            expectedOutput
        }
        val provider = SdkStreamingProvider(
            delegate = delegate,
            onAttemptStarted = { attempt -> events += "start-$attempt" },
            onPartialText = { attempt, text -> events += "partial-$attempt:$text" },
            onAttemptCompleted = { attempt, output ->
                assertSame(expectedOutput, output)
                events += "complete-$attempt:${output.text}"
            },
        )

        val output = provider.generate(prompt)

        assertSame(expectedOutput, output)
        assertEquals(exactOutput, output.text)
        assertEquals(
            listOf(
                "start-1",
                "partial-1:$firstSnapshot",
                "partial-1:$exactOutput",
                "delegate-return",
                "complete-1:$exactOutput",
            ),
            events,
        )
    }

    @Test
    fun attemptsRemainNumberedAndSeparatedAcrossCalls() = runBlocking {
        val delegateCalls = AtomicInteger(0)
        val events = mutableListOf<String>()
        val downstreamSnapshots = mutableListOf<String>()
        val delegate = streamingDelegate { onPartialText ->
            val call = delegateCalls.incrementAndGet()
            val raw = "raw-$call"
            onPartialText(raw)
            GenUiModelOutput(raw, "test/$call")
        }
        val provider = SdkStreamingProvider(
            delegate = delegate,
            onAttemptStarted = { attempt -> events += "start-$attempt" },
            onPartialText = { attempt, text -> events += "partial-$attempt:$text" },
            onAttemptCompleted = { attempt, output -> events += "complete-$attempt:${output.text}" },
        )

        provider.generate(prompt) { text -> downstreamSnapshots += text }
        provider.generate(prompt)

        assertEquals(listOf("raw-1"), downstreamSnapshots)
        assertEquals(
            listOf(
                "start-1",
                "partial-1:raw-1",
                "complete-1:raw-1",
                "start-2",
                "partial-2:raw-2",
                "complete-2:raw-2",
            ),
            events,
        )
    }

    @Test
    fun providerFailurePropagatesAfterRetainingLatestRawSnapshot() = runBlocking {
        val snapshots = mutableListOf<Pair<Int, String>>()
        val completed = mutableListOf<Int>()
        val delegateCalls = AtomicInteger(0)
        val delegate = streamingDelegate { onPartialText ->
            delegateCalls.incrementAndGet()
            onPartialText("raw prefix before failure")
            error("native decode failed")
        }
        val provider = SdkStreamingProvider(
            delegate = delegate,
            onAttemptStarted = {},
            onPartialText = { attempt, text -> snapshots += attempt to text },
            onAttemptCompleted = { attempt, _ -> completed += attempt },
        )

        try {
            provider.generate(prompt)
            fail("Provider failure must propagate")
        } catch (expected: IllegalStateException) {
            assertEquals("native decode failed", expected.message)
        }

        assertEquals(1, delegateCalls.get())
        assertEquals(listOf(1 to "raw prefix before failure"), snapshots)
        assertTrue(completed.isEmpty())
        assertEquals(
            listOf(SdkStreamingAttemptSnapshot(1, "raw prefix before failure", complete = false)),
            provider.attemptSnapshots,
        )
    }

    @Test
    fun cancellationPropagatesAfterRetainingLatestRawSnapshot() = runBlocking {
        val snapshots = mutableListOf<Pair<Int, String>>()
        val completed = mutableListOf<Int>()
        val delegateCalls = AtomicInteger(0)
        val delegate = streamingDelegate { onPartialText ->
            delegateCalls.incrementAndGet()
            onPartialText("raw prefix before cancellation")
            throw CancellationException("request replaced")
        }
        val provider = SdkStreamingProvider(
            delegate = delegate,
            onAttemptStarted = {},
            onPartialText = { attempt, text -> snapshots += attempt to text },
            onAttemptCompleted = { attempt, _ -> completed += attempt },
        )

        try {
            provider.generate(prompt)
            fail("Cancellation must propagate")
        } catch (expected: CancellationException) {
            assertEquals("request replaced", expected.message)
        }

        assertEquals(1, delegateCalls.get())
        assertEquals(listOf(1 to "raw prefix before cancellation"), snapshots)
        assertTrue(completed.isEmpty())
        assertEquals(
            listOf(SdkStreamingAttemptSnapshot(1, "raw prefix before cancellation", complete = false)),
            provider.attemptSnapshots,
        )
    }

    @Test
    fun exactTerminalOutputIsReportedWhenDelegateEmitsNoPartials() = runBlocking {
        val exactOutput = "\n  terminal only  \n"
        val returned = GenUiModelOutput(exactOutput, "test/no-deltas")
        val snapshots = mutableListOf<String>()
        val completed = mutableListOf<Pair<Int, GenUiModelOutput>>()
        val delegate = streamingDelegate { returned }
        val provider = SdkStreamingProvider(
            delegate = delegate,
            onAttemptStarted = {},
            onPartialText = { _, text -> snapshots += text },
            onAttemptCompleted = { attempt, output -> completed += attempt to output },
        )

        val output = provider.generate(prompt)

        assertSame(returned, output)
        assertTrue(snapshots.isEmpty())
        assertEquals(1, completed.size)
        assertEquals(1, completed.single().first)
        assertSame(returned, completed.single().second)
        assertEquals(exactOutput, completed.single().second.text)
        assertEquals(
            listOf(SdkStreamingAttemptSnapshot(1, exactOutput, complete = true)),
            provider.attemptSnapshots,
        )
    }

    private fun streamingDelegate(
        generation: ((String) -> Unit) -> GenUiModelOutput,
    ): GenUiProvider = object : GenUiProvider {
        override val id: String = "streaming-test"

        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
            error("SdkStreamingProvider must call the streaming overload")

        override suspend fun generate(
            prompt: GenUiPrompt,
            onPartialText: (String) -> Unit,
        ): GenUiModelOutput = generation(onPartialText)
    }

    private companion object {
        val prompt = GenUiPrompt(system = "system", user = "user")
    }
}
