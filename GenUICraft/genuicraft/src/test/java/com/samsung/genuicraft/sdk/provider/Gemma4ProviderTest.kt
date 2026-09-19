package com.samsung.genuicraft.sdk.provider

import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiPromptMessage
import com.samsung.genuicraft.sdk.GenUiPromptRole
import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class Gemma4ProviderTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test fun `fewshot history counts toward context preflight`() = runBlocking {
        val runtime = RecordingRuntime()
        provider(runtime, maxContextTokens = 1024, maxOutputTokens = 128).use { provider ->
            assertThrows(IllegalArgumentException::class.java) {
                runBlocking {
                    provider.generate(GenUiPrompt("system", "short request", maxOutputTokens = 128,
                        initialMessages = listOf(GenUiPromptMessage(GenUiPromptRole.USER, "x".repeat(2400)))))
                }
            }
            assertTrue(runtime.maxOutputTokens.isEmpty())
        }
    }

    @Test
    fun `generation returns explicit runtime metadata and respects output boundary`() = runBlocking {
        val runtime = RecordingRuntime(
            result = Gemma4RuntimeOutput(
                text = "<a2ui>\nroot=Text(\"Ready\")\n</a2ui>",
                runtimeIdentity = "LiteRT-LM/Gemma4/GPU+MTP",
                outputTokens = 27,
            ),
        )
        provider(runtime, maxOutputTokens = 256).use { provider ->
            val result = provider.generate(
                GenUiPrompt(
                    system = "Return compact A2UI.",
                    user = "Create a status card.",
                    maxOutputTokens = 256,
                    temperature = 0.0,
                ),
            )

            assertEquals("gemma4_e2b", provider.id)
            assertEquals("LiteRT-LM/Gemma4/GPU+MTP", result.runtime)
            assertEquals(27, result.outputTokens)
            assertNull(result.metrics)
            assertEquals(256, runtime.maxOutputTokens.single())
        }
    }

    @Test
    fun `provider rejects output and context overflow before native generation`() = runBlocking {
        val runtime = RecordingRuntime()
        provider(runtime, maxContextTokens = 1_024, maxOutputTokens = 128).use { provider ->
            assertThrows(IllegalArgumentException::class.java) {
                runBlocking {
                    provider.generate(GenUiPrompt("system", "request", maxOutputTokens = 129))
                }
            }
            assertThrows(IllegalArgumentException::class.java) {
                runBlocking {
                    provider.generate(
                        GenUiPrompt(
                            system = "system",
                            user = "x".repeat(2_100),
                            maxOutputTokens = 128,
                        ),
                    )
                }
            }
            assertTrue(runtime.maxOutputTokens.isEmpty())
        }
    }

    @Test
    fun `overlapping calls are serialized around one runtime conversation`() = runBlocking {
        val runtime = BlockingRuntime()
        provider(runtime).use { provider ->
            val first = async { provider.generate(GenUiPrompt("system", "first", maxOutputTokens = 64)) }
            runtime.firstStarted.await()
            val second = async { provider.generate(GenUiPrompt("system", "second", maxOutputTokens = 64)) }
            delay(100)

            assertEquals(1, runtime.started.get())
            assertEquals(1, runtime.maxConcurrent.get())

            runtime.releaseFirst.complete(Unit)
            first.await()
            second.await()
            assertEquals(2, runtime.started.get())
            assertEquals(1, runtime.maxConcurrent.get())
        }
    }

    @Test
    fun `cancellation reaches the active native runtime and provider remains reusable`() = runBlocking {
        val runtime = CancellableRuntime()
        provider(runtime).use { provider ->
            val generation = async {
                provider.generate(GenUiPrompt("system", "cancel me", maxOutputTokens = 64))
            }
            runtime.started.await()
            generation.cancelAndJoin()

            assertTrue(generation.isCancelled)
            assertEquals(1, runtime.cancelCalls.get())

            runtime.block = false
            val recovered = provider.generate(GenUiPrompt("system", "next", maxOutputTokens = 64))
            assertEquals("recovered", recovered.text)
        }
    }

    @Test
    fun `close is idempotent and prevents later generation`() {
        val runtime = RecordingRuntime()
        val provider = provider(runtime)
        provider.close()
        provider.close()

        assertEquals(1, runtime.closeCalls.get())
        assertThrows(IllegalStateException::class.java) {
            runBlocking {
                provider.generate(GenUiPrompt("system", "request", maxOutputTokens = 64))
            }
        }
    }

    @Test
    fun `close during an active call returns immediately and cancels runtime work`() = runBlocking {
        val runtime = CloseAwareRuntime()
        val provider = provider(runtime)
        val generation = async {
            provider.generate(GenUiPrompt("system", "active", maxOutputTokens = 64))
        }
        runtime.started.await()

        withTimeout(1_000) {
            withContext(Dispatchers.Default) { provider.close() }
        }

        generation.cancelAndJoin()
        assertTrue(runtime.cancelCalls.get() >= 1)
        assertEquals(1, runtime.closeCalls.get())
    }

    @Test
    fun `close and await suspends until native cleanup finishes`() = runBlocking {
        val runtime = AwaitableCloseRuntime()
        val provider = provider(runtime)

        val closing = async { provider.closeAndAwait() }
        runtime.awaitStarted.await()

        assertEquals(1, runtime.closeCalls.get())
        assertEquals(1, runtime.awaitCalls.get())
        assertFalse(closing.isCompleted)

        runtime.cleanupFinished.complete(Unit)
        withTimeout(1_000) { closing.await() }
        assertTrue(closing.isCompleted)
    }

    @Test
    fun `public config validates external model and accelerator`() {
        assertThrows(IllegalArgumentException::class.java) {
            Gemma4Provider(
                Gemma4Config(
                    modelPath = File(temporaryFolder.root, "missing.litertlm").absolutePath,
                ),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            Gemma4Provider(Gemma4Config(modelPath = modelFile().absolutePath, accelerator = "NPU"))
        }
        assertThrows(IllegalArgumentException::class.java) {
            Gemma4Provider(Gemma4Config(modelPath = modelFile().absolutePath, thinkingTokenBudget = -2))
        }
    }

    @Test
    fun `reasoning stays enabled with a bounded formatting budget by default`() {
        lateinit var validated: ValidatedGemma4Config
        val runtime = RecordingRuntime()
        val provider = Gemma4Provider(
            config = Gemma4Config(
                modelPath = modelFile().absolutePath,
                maxContextTokens = 2_048,
                maxOutputTokens = 256,
            ),
            runtimeFactory = Gemma4RuntimeFactory { config ->
                validated = config
                runtime
            },
        )
        provider.close()

        assertTrue(validated.enableThinking)
        assertEquals(1_024, validated.thinkingTokenBudget)
        assertTrue(validated.enableSpeculativeDecoding)
        assertFalse(validated.enableMetrics)
    }

    @Test
    fun `metrics opt in propagates to native configuration and preserves actual counters`() = runBlocking {
        lateinit var validated: ValidatedGemma4Config
        val metrics = GenUiGenerationMetrics(inputTokens = 719, outputTokens = 127, decodeTokensPerSecond = 23.75)
        val runtime = RecordingRuntime(Gemma4RuntimeOutput("result", "LiteRT-LM/Gemma4/GPU+MTP", 127, metrics))
        Gemma4Provider(
            config = Gemma4Config(modelPath = modelFile().absolutePath, enableMetrics = true),
            runtimeFactory = Gemma4RuntimeFactory { config -> validated = config; runtime },
        ).use { provider ->
            val result = provider.generate(GenUiPrompt("system", "request", maxOutputTokens = 64))
            assertTrue(validated.enableMetrics)
            assertTrue(validated.enableThinking)
            assertTrue(validated.enableSpeculativeDecoding)
            assertEquals(1_024, validated.thinkingTokenBudget)
            assertEquals(metrics, result.metrics)
            assertEquals(127, result.outputTokens)
        }
    }

    @Test
    fun `metrics are not exposed without explicit opt in even if runtime supplies them`() = runBlocking {
        val runtime = RecordingRuntime(Gemma4RuntimeOutput("result", "runtime", 27,
            GenUiGenerationMetrics(100, 27, 20.5)))
        provider(runtime).use { provider ->
            val result = provider.generate(GenUiPrompt("system", "request", maxOutputTokens = 64))
            assertNull(result.metrics)
            assertEquals(27, result.outputTokens)
        }
    }

    @Test
    fun `speculative decoding requires GPU model capability and respects explicit opt out`() {
        lateinit var optedInConfig: ValidatedGemma4Config
        Gemma4Provider(
            config = Gemma4Config(
                modelPath = modelFile().absolutePath,
                maxContextTokens = 2_048,
                maxOutputTokens = 256,
                enableSpeculativeDecoding = true,
            ),
            runtimeFactory = Gemma4RuntimeFactory { config ->
                optedInConfig = config
                RecordingRuntime()
            },
        ).close()
        assertTrue(optedInConfig.enableSpeculativeDecoding)

        assertFalse(
            shouldEnableSpeculativeDecoding(
                accelerator = Gemma4Accelerator.GPU,
                enabledByHost = false,
                modelSupportsMtp = true,
            ),
        )
        assertFalse(
            shouldEnableSpeculativeDecoding(
                accelerator = Gemma4Accelerator.CPU,
                enabledByHost = true,
                modelSupportsMtp = true,
            ),
        )
        assertFalse(
            shouldEnableSpeculativeDecoding(
                accelerator = Gemma4Accelerator.GPU,
                enabledByHost = true,
                modelSupportsMtp = false,
            ),
        )
        assertTrue(
            shouldEnableSpeculativeDecoding(
                accelerator = Gemma4Accelerator.GPU,
                enabledByHost = true,
                modelSupportsMtp = true,
            ),
        )
    }

    @Test
    fun `GPU runtime rejects unsupported ABI instead of changing backend`() {
        requireArm64GpuAbi(listOf("arm64-v8a", "armeabi-v7a"))

        val failure = assertThrows(IllegalStateException::class.java) {
            requireArm64GpuAbi(listOf("x86_64"))
        }
        assertTrue(failure.message.orEmpty().contains("arm64-v8a"))
        assertTrue(failure.message.orEmpty().contains("x86_64"))
    }

    private fun provider(
        runtime: Gemma4Runtime,
        maxContextTokens: Int = 2_048,
        maxOutputTokens: Int = 256,
    ): Gemma4Provider = Gemma4Provider(
        config = Gemma4Config(
            modelPath = modelFile().absolutePath,
            maxContextTokens = maxContextTokens,
            maxOutputTokens = maxOutputTokens,
        ),
        runtimeFactory = Gemma4RuntimeFactory { runtime },
    )

    private fun modelFile(): File = temporaryFolder.newFile("gemma-4-e2b-${System.nanoTime()}.litertlm")
        .apply { writeBytes(byteArrayOf(1)) }

    private class RecordingRuntime(
        private val result: Gemma4RuntimeOutput = Gemma4RuntimeOutput(
            text = "ok",
            runtimeIdentity = "LiteRT-LM/Gemma4/GPU",
            outputTokens = 1,
        ),
    ) : Gemma4Runtime {
        val maxOutputTokens = mutableListOf<Int>()
        val closeCalls = AtomicInteger()

        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput {
            this.maxOutputTokens += maxOutputTokens
            return result
        }

        override fun cancelActive() = Unit

        override suspend fun awaitClosed() = Unit

        override fun close() {
            closeCalls.incrementAndGet()
        }
    }

    private class BlockingRuntime : Gemma4Runtime {
        val firstStarted = CompletableDeferred<Unit>()
        val releaseFirst = CompletableDeferred<Unit>()
        val started = AtomicInteger()
        val maxConcurrent = AtomicInteger()
        private val active = AtomicInteger()

        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput {
            val call = started.incrementAndGet()
            val activeNow = active.incrementAndGet()
            maxConcurrent.updateAndGet { previous -> maxOf(previous, activeNow) }
            try {
                if (call == 1) {
                    firstStarted.complete(Unit)
                    releaseFirst.await()
                }
                return Gemma4RuntimeOutput("result-$call", "fake", 1)
            } finally {
                active.decrementAndGet()
            }
        }

        override fun cancelActive() = Unit
        override suspend fun awaitClosed() = Unit
        override fun close() = Unit
    }

    private class CancellableRuntime : Gemma4Runtime {
        val started = CompletableDeferred<Unit>()
        val cancelCalls = AtomicInteger()
        @Volatile var block = true

        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput {
            if (block) {
                started.complete(Unit)
                awaitCancellation()
            }
            return Gemma4RuntimeOutput("recovered", "fake", 1)
        }

        override fun cancelActive() {
            cancelCalls.incrementAndGet()
        }

        override suspend fun awaitClosed() = Unit

        override fun close() = Unit
    }

    private class CloseAwareRuntime : Gemma4Runtime {
        val started = CompletableDeferred<Unit>()
        val cancelCalls = AtomicInteger()
        val closeCalls = AtomicInteger()

        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput {
            started.complete(Unit)
            awaitCancellation()
        }

        override fun cancelActive() {
            cancelCalls.incrementAndGet()
        }

        override suspend fun awaitClosed() = Unit

        override fun close() {
            closeCalls.incrementAndGet()
        }
    }

    private class AwaitableCloseRuntime : Gemma4Runtime {
        val closeCalls = AtomicInteger()
        val awaitCalls = AtomicInteger()
        val awaitStarted = CompletableDeferred<Unit>()
        val cleanupFinished = CompletableDeferred<Unit>()

        override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput =
            Gemma4RuntimeOutput("unused", "fake", 1)

        override fun cancelActive() = Unit

        override suspend fun awaitClosed() {
            awaitCalls.incrementAndGet()
            awaitStarted.complete(Unit)
            cleanupFinished.await()
        }

        override fun close() {
            closeCalls.incrementAndGet()
        }
    }
}
