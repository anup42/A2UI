package com.samsung.genuicraft.pipeline

import com.google.gson.JsonParser
import com.samsung.genuicraft.InferenceBackendSettings
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiConversionResult
import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiRepairKind
import com.samsung.genuicraft.sdk.GenUiRequest
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class TrainedStage3BridgeTest {
    @Test
    fun routeIsLimitedToTheFlaggedOnDeviceCatalogEntry() {
        val currentPath = "/models/gemma4_e2b_a2ui_mobile.litertlm"

        assertTrue(
            TrainedStage3Bridge.shouldUse(
                InferenceBackendSettings.Provider.ON_DEVICE_LITERT,
                currentPath,
            )
        )
        assertFalse(
            TrainedStage3Bridge.shouldUse(
                InferenceBackendSettings.Provider.GEMINI,
                currentPath,
            )
        )
        assertFalse(
            TrainedStage3Bridge.shouldUse(
                InferenceBackendSettings.Provider.ON_DEVICE_LITERT,
                "/models/gemma-4-E2B-it.litertlm",
            )
        )
    }

    @Test
    fun configPinsGpuContextOutputThinkingMetricsAndMtp() {
        listOf(
            InferenceBackendSettings.Accelerator.AUTO,
            InferenceBackendSettings.Accelerator.GPU,
        ).forEach { accelerator ->
            val config = TrainedStage3Bridge.configFor(
                modelPath = "/models/gemma4_e2b_a2ui_mobile.litertlm",
                accelerator = accelerator,
                enableMtp = true,
            )

            assertEquals("GPU", config.accelerator)
            assertEquals(8_192, config.maxContextTokens)
            assertEquals(2_048, config.maxOutputTokens)
            assertFalse(config.enableThinking)
            assertEquals(0, config.thinkingTokenBudget)
            assertTrue(config.enableSpeculativeDecoding)
            assertTrue(config.enableMetrics)
        }

        listOf(
            InferenceBackendSettings.Accelerator.CPU,
            InferenceBackendSettings.Accelerator.NPU,
        ).forEach { accelerator ->
            try {
                TrainedStage3Bridge.configFor("/models/model.litertlm", accelerator, false)
                fail("$accelerator must not silently run the GPU-only trained model")
            } catch (expected: IllegalArgumentException) {
                assertTrue(expected.message.orEmpty().contains("requires GPU"))
            }
        }
    }

    @Test
    fun sdkWireResultKeepsActualPromptMetricsRuntimeAndCanonicalData() = runBlocking {
        val document = GenUiCompiler.compile(firstConformanceProgram())
        val closed = AtomicBoolean(false)
        val provider = object : GenUiProvider {
            override val id = "test-trained"

            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput = GenUiModelOutput(
                text = firstConformanceProgram(),
                runtime = "LiteRT-LM/Gemma4/GPU/MTP",
                outputTokens = 31,
                metrics = GenUiGenerationMetrics(
                    inputTokens = 47,
                    outputTokens = 31,
                    decodeTokensPerSecond = 6.25,
                ),
                renderedPromptSha256 = "prompt-sha",
            )

            override suspend fun closeAndAwait() {
                closed.set(true)
            }
        }
        val request = GenUiRequest(text = "Hello", query = "Show a greeting")

        val result = TrainedStage3Bridge.runWithProvider(provider, request) { captured, source ->
            captured.generate(
                GenUiPrompt(
                    system = "frozen system",
                    user = "Create IR for: ${source.text}",
                    maxOutputTokens = 2_048,
                )
            )
            GenUiConversionResult.Success(
                document = document,
                provider = captured.id,
                elapsedMs = 123,
                attempts = 1,
                warnings = listOf("source fidelity audit: no mechanical issue detected"),
                repairKind = GenUiRepairKind.NONE,
            )
        }

        assertTrue(result.toString(), result is TrainedStage3Bridge.Result.Success)
        result as TrainedStage3Bridge.Result.Success
        assertEquals("root", result.canonicalGraph.get("root").asString)
        assertEquals("Create IR for: Hello", result.prompt?.user)
        assertEquals(47, result.inputTokens)
        assertEquals(31, result.outputTokens)
        assertEquals(6.25, result.outputTokensPerSecond ?: 0.0, 0.0)
        assertEquals("LiteRT-LM/Gemma4/GPU/MTP", result.runtimeBackend)
        assertEquals("prompt-sha", result.renderedPromptSha256)
        assertEquals(123, result.elapsedMs)
        assertTrue(result.warnings.single().contains("source fidelity audit"))
        assertTrue(closed.get())
    }

    @Test
    fun cumulativeRawSnapshotsArriveBeforeFinalConversion() = runBlocking {
        val program = firstConformanceProgram()
        val rawOutput = "\n$program\n"
        val firstSnapshot = rawOutput.take(rawOutput.length / 2)
        val document = GenUiCompiler.compile(program)
        val snapshots = mutableListOf<String>()
        val events = mutableListOf<String>()
        val provider = object : GenUiProvider {
            override val id = "streaming-trained"

            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
                error("The bridge must use the streaming provider overload")

            override suspend fun generate(
                prompt: GenUiPrompt,
                onPartialText: (String) -> Unit,
            ): GenUiModelOutput {
                onPartialText(firstSnapshot)
                onPartialText(rawOutput)
                events += "provider-return"
                return GenUiModelOutput(rawOutput, "test/GPU")
            }
        }

        val result = TrainedStage3Bridge.runWithProvider(
            provider = provider,
            request = GenUiRequest("Hello"),
            onPartialText = { text ->
                snapshots += text
                events += "partial-${snapshots.size}"
            },
        ) { captured, _ ->
            captured.generate(GenUiPrompt("system", "user"))
            events += "conversion"
            GenUiConversionResult.Success(
                document = document,
                provider = captured.id,
                elapsedMs = 2,
                attempts = 1,
            )
        }

        assertEquals(listOf(firstSnapshot, rawOutput), snapshots)
        assertEquals(
            listOf("partial-1", "partial-2", "provider-return", "conversion"),
            events,
        )
        assertTrue(result is TrainedStage3Bridge.Result.Success)
        result as TrainedStage3Bridge.Result.Success
        assertEquals(rawOutput, result.rawGeneratedText)
    }

    @Test
    fun sourceFallbackIsRejectedAndRawGeneratedOutputIsRetained() = runBlocking {
        val program = firstConformanceProgram()
        val document = GenUiCompiler.compile(program)
        val provider = object : GenUiProvider {
            override val id = "test-trained"

            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
                GenUiModelOutput(program, "test/GPU")
        }

        val result = TrainedStage3Bridge.runWithProvider(
            provider,
            GenUiRequest("Hello"),
        ) { captured, _ ->
            captured.generate(GenUiPrompt("system", "user"))
            GenUiConversionResult.Success(
                document = document,
                provider = captured.id,
                elapsedMs = 1,
                attempts = 1,
                repairKind = GenUiRepairKind.SOURCE_TEXT_FALLBACK,
            )
        }

        assertTrue(result is TrainedStage3Bridge.Result.Failure)
        result as TrainedStage3Bridge.Result.Failure
        assertTrue(result.message.contains("SOURCE_TEXT_FALLBACK"))
        assertEquals(program, result.rawGeneratedText)
    }

    @Test
    fun cancellationPropagatesAfterProviderCleanup() = runBlocking {
        val closed = AtomicBoolean(false)
        val snapshots = mutableListOf<String>()
        val provider = object : GenUiProvider {
            override val id = "cancel-trained"

            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
                error("The bridge must use the streaming provider overload")

            override suspend fun generate(
                prompt: GenUiPrompt,
                onPartialText: (String) -> Unit,
            ): GenUiModelOutput {
                onPartialText("raw before cancellation")
                throw CancellationException("host stopped")
            }

            override suspend fun closeAndAwait() {
                closed.set(true)
            }
        }

        try {
            TrainedStage3Bridge.runWithProvider(
                provider = provider,
                request = GenUiRequest("Hello"),
                onPartialText = { text -> snapshots += text },
            ) { captured, _ ->
                captured.generate(GenUiPrompt("system", "user"))
                error("unreachable")
            }
            fail("Cancellation must propagate to the pipeline host")
        } catch (_: CancellationException) {
            // Expected.
        }
        assertEquals(listOf("raw before cancellation"), snapshots)
        assertTrue(closed.get())
    }

    @Test
    fun providerErrorReturnsFailureWithLatestRawSnapshot() = runBlocking {
        val closed = AtomicBoolean(false)
        val provider = object : GenUiProvider {
            override val id = "failed-trained"

            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
                error("The bridge must use the streaming provider overload")

            override suspend fun generate(
                prompt: GenUiPrompt,
                onPartialText: (String) -> Unit,
            ): GenUiModelOutput {
                onPartialText("raw prefix before native failure")
                error("native decode failed")
            }

            override suspend fun closeAndAwait() {
                closed.set(true)
            }
        }

        val result = TrainedStage3Bridge.runWithProvider(
            provider,
            GenUiRequest("Hello"),
        ) { captured, _ ->
            captured.generate(GenUiPrompt("system", "user"))
            error("unreachable")
        }

        assertTrue(result is TrainedStage3Bridge.Result.Failure)
        result as TrainedStage3Bridge.Result.Failure
        assertEquals("native decode failed", result.message)
        assertEquals("raw prefix before native failure", result.rawGeneratedText)
        assertTrue(closed.get())
    }

    @Test
    fun conversionFailureKeepsExactUntrimmedProviderOutput() = runBlocking {
        val exactRawOutput = "\n  invalid raw output  \n"
        val provider = object : GenUiProvider {
            override val id = "compile-failed-trained"

            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
                error("The bridge must use the streaming provider overload")

            override suspend fun generate(
                prompt: GenUiPrompt,
                onPartialText: (String) -> Unit,
            ): GenUiModelOutput {
                onPartialText(exactRawOutput)
                return GenUiModelOutput(exactRawOutput, "test/GPU")
            }
        }

        val result = TrainedStage3Bridge.runWithProvider(
            provider,
            GenUiRequest("Hello"),
        ) { captured, _ ->
            captured.generate(GenUiPrompt("system", "user"))
            GenUiConversionResult.Failure(
                message = "strict compilation failed",
                provider = captured.id,
                elapsedMs = 4,
                attempts = 1,
                rawOutput = exactRawOutput.trim(),
            )
        }

        assertTrue(result is TrainedStage3Bridge.Result.Failure)
        result as TrainedStage3Bridge.Result.Failure
        assertEquals(exactRawOutput, result.rawGeneratedText)
    }

    @Test
    fun streamObserverErrorReturnsFailureWithoutLosingRawSnapshot() = runBlocking {
        val provider = object : GenUiProvider {
            override val id = "observer-failed-trained"

            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput =
                error("The bridge must use the streaming provider overload")

            override suspend fun generate(
                prompt: GenUiPrompt,
                onPartialText: (String) -> Unit,
            ): GenUiModelOutput {
                onPartialText("raw before observer failure")
                error("unreachable")
            }
        }

        val result = TrainedStage3Bridge.runWithProvider(
            provider = provider,
            request = GenUiRequest("Hello"),
            onPartialText = { error("host observer failed") },
        ) { captured, _ ->
            captured.generate(GenUiPrompt("system", "user"))
            error("unreachable")
        }

        assertTrue(result is TrainedStage3Bridge.Result.Failure)
        result as TrainedStage3Bridge.Result.Failure
        assertEquals("host observer failed", result.message)
        assertEquals("raw before observer failure", result.rawGeneratedText)
    }

    private fun firstConformanceProgram(): String {
        val stream = requireNotNull(
            javaClass.classLoader?.getResourceAsStream("a2ui_express_conformance_v1.json")
        )
        val root = stream.bufferedReader(Charsets.UTF_8).use { reader ->
            JsonParser.parseReader(reader).asJsonObject
        }
        val program = root.getAsJsonArray("valid").first().asJsonObject.get("program")
        assertNotNull(program)
        return program.asString
    }
}
