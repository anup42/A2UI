package com.samsung.genuicraft.sdk.provider

import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import com.samsung.genuicraft.sdk.GenUiGenerationSessionMetrics
import com.samsung.genuicraft.sdk.GenUiModelProfiles
import com.samsung.genuicraft.sdk.GenUiPrompt
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class Gemma270MProviderTest {
    @get:Rule val temporaryFolder = TemporaryFolder()

    @Test fun `profile selects CPU without thinking MTP or precision overrides`() {
        val config = GenUiModelProfiles.trainedGemma270M("model.litertlm")
        assertEquals("CPU", config.accelerator)
        assertEquals(4096, config.maxContextTokens)
        assertEquals(2048, config.maxOutputTokens)
        assertFalse(config.enableThinking)
        assertEquals(0, config.thinkingTokenBudget)
        assertFalse(config.enableSpeculativeDecoding)
        assertFalse(config.enableMetrics)
        assertEquals(Gemma4GpuPrecision.MODEL_DEFAULT, config.gpuPrecision)
        assertEquals("GPU", GenUiModelProfiles.trainedGemma270M("model.litertlm", accelerator = "gpu").accelerator)
        assertThrows(IllegalArgumentException::class.java) {
            GenUiModelProfiles.trainedGemma270M("model.litertlm", accelerator = "AUTO")
        }
    }

    @Test fun `provider streams exact raw text and retains native metrics and identity`() = runBlocking {
        val metrics = GenUiGenerationMetrics(101, 17, 11.5)
        val sessionMetrics = GenUiGenerationSessionMetrics(speculativeDecodingEnabled = false)
        val runtime = object : Gemma4Runtime {
            override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput =
                error("Streaming overload required")
            override suspend fun generate(
                prompt: GenUiPrompt,
                maxOutputTokens: Int,
                onPartialText: (String) -> Unit,
            ): Gemma4RuntimeOutput {
                assertNull(prompt.chatTemplateOverride)
                assertNull(prompt.expectedRenderedPrompt)
                assertEquals(2048, maxOutputTokens)
                onPartialText("<a2ui>")
                onPartialText("<a2ui>\nroot=Text(\"Hello\")\n</a2ui>")
                return Gemma4RuntimeOutput(
                    "<a2ui>\nroot=Text(\"Hello\")\n</a2ui>",
                    "LiteRT-LM/Gemma3-270M/CPU", 17, metrics,
                )
            }
            override suspend fun finishGenerationMetrics() = sessionMetrics
            override fun cancelActive() = Unit
            override suspend fun awaitClosed() = Unit
            override fun close() = Unit
        }
        var validated: ValidatedGemma4Config? = null
        val config = GenUiModelProfiles.trainedGemma270M(modelFile().absolutePath, enableMetrics = true)
        Gemma270MProvider(config, Gemma4RuntimeFactory { received ->
            validated = received
            runtime
        }).use { provider ->
            val partials = mutableListOf<String>()
            val output = provider.generate(GenUiPrompt("system", "user", maxOutputTokens = 2048), partials::add)
            assertEquals(Gemma270MProvider.PROVIDER_ID, provider.id)
            assertEquals("Gemma3-270M", validated!!.runtimeFamily)
            assertTrue(validated!!.enableMetrics)
            assertEquals("LiteRT-LM/Gemma3-270M/CPU", output.runtime)
            assertEquals(metrics, output.metrics)
            assertEquals(17, output.outputTokens)
            assertEquals(listOf("<a2ui>", output.text), partials)
            assertEquals(sessionMetrics, provider.finishGenerationMetrics())
        }
    }

    @Test fun `provider refuses an E2B turn template and incompatible runtime profile`() = runBlocking {
        val config = GenUiModelProfiles.trainedGemma270M(modelFile().absolutePath)
        val runtime = object : Gemma4Runtime {
            override suspend fun generate(prompt: GenUiPrompt, maxOutputTokens: Int): Gemma4RuntimeOutput =
                error("Native runtime must not be reached")
            override fun cancelActive() = Unit
            override suspend fun awaitClosed() = Unit
            override fun close() = Unit
        }
        Gemma270MProvider(config, Gemma4RuntimeFactory { runtime }).use { provider ->
            assertThrows(IllegalArgumentException::class.java) {
                runBlocking {
                    provider.generate(GenUiPrompt("system", "user", chatTemplateOverride = "<|turn>"))
                }
            }
        }
        assertThrows(IllegalArgumentException::class.java) {
            Gemma270MProvider(config.copy(enableSpeculativeDecoding = true))
        }
        assertThrows(IllegalArgumentException::class.java) {
            Gemma270MProvider(config.copy(enableThinking = true))
        }
        Unit
    }

    private fun modelFile(): File = temporaryFolder.newFile("gemma-3-270m-a2ui-express-int8.litertlm")
        .apply { writeBytes(byteArrayOf(1)) }
}
