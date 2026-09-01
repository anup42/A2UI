package com.samsung.genuicraft

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.OnDeviceLitertBackend
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class Gemma270mA2uiExpressOnDeviceTest {
    @Test
    fun autoFallsBackToCpuWhenMobileGpuReturnsSpecialTokens() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma3_270m_a2ui_express_int8" }
        )
        val contract = context.assets.open("pipeline_prompts/genui_gen_a2ui_express_training_v1.md")
            .bufferedReader()
            .use { it.readText() }
        val systemPrompt = contract.replace(
            "{response_text}",
            "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]",
        )
        val responseText = """
            Weather comparison for Bengaluru
            Today is warm with a high of 29 C and a low of 21 C. Rain chance is 35 percent.
            Tomorrow is cloudy with a high of 27 C and a low of 20 C. Rain chance is 55 percent.
        """.trimIndent()
        val prompt = "Create A2UI Express v1 GenUI IR for this response:\n\n$responseText"
        val response = OnDeviceLitertBackend(
            entry.localPath(context),
            acceleratorPreference = InferenceBackendSettings.Accelerator.AUTO,
        ).generate(
            InferenceBackend.GenerateRequest(
                prompt = prompt,
                systemPrompt = systemPrompt,
                temperature = 0.0,
                maxOutputTokens = 512,
                jsonMode = false,
                structuredOutput = false,
                initialMessages = listOf(
                    InferenceBackend.ConversationMessage(
                        role = InferenceBackend.ConversationRole.USER,
                        content = "Create A2UI Express v1 GenUI IR for this response:\n\nA small travel checklist with a title and two items.",
                    ),
                    InferenceBackend.ConversationMessage(
                        role = InferenceBackend.ConversationRole.MODEL,
                        content = "<a2ui>\nroot=Column([a,b])\na=Text(\"Travel Checklist\",\"h1\")\nb=List([c,d])\nc=Text(\"Passport\")\nd=Text(\"Charger\")\n</a2ui>",
                    ),
                ),
            )
        )

        assertNull("AUTO generation failed: ${response.error}", response.error)
        assertEquals("CPU", response.runtimeBackend)
        assertTrue("AUTO fallback returned invalid output: ${response.text}", response.text.contains("<a2ui>"))
        assertTrue("AUTO fallback returned GPU special tokens: ${response.text}", !response.text.contains("<pad>"))
        runCatching { com.samsung.genuicraft.pipeline.A2uiExpressCodec.decode(response.text) }
            .getOrElse { error("AUTO fallback A2UI Express validation failed: ${it.message}") }
        val renderResult = GenUiNativeRenderer.render(response.text, sourceDir = null)
        assertNull("AUTO fallback native render failed: ${renderResult.errorMessage}", renderResult.errorMessage)
        assertTrue("AUTO fallback produced no render surface", renderResult.surfaces.isNotEmpty())
    }

    @Test
    fun generateValidateAndRenderOnMobileGpu() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val requestedBackend = InstrumentationRegistry.getArguments()
            .getString("backend")
            ?.trim()
            ?.lowercase()
            ?: "gpu"
        val accelerator = when (requestedBackend) {
            "cpu" -> InferenceBackendSettings.Accelerator.CPU
            else -> InferenceBackendSettings.Accelerator.GPU
        }
        val expectedBackend = if (requestedBackend == "cpu") "CPU" else "GPU"
        val probeMode = InstrumentationRegistry.getArguments()
            .getString("probe")
            ?.trim()
            ?.lowercase()
            ?: "express"
        val temperature = InstrumentationRegistry.getArguments()
            .getString("temperature")
            ?.toDoubleOrNull()
            ?.takeIf { it.isFinite() && it >= 0.0 }
            ?: 0.0
        val gpuControlProbe = probeMode == "gpu-control"
        val resultDir = File(
            context.getExternalFilesDir(null) ?: context.filesDir,
            "result_gemma270m_a2ui_express",
        ).apply { mkdirs() }
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma3_270m_a2ui_express_int8" }
        )
        val modelFile = entry.localFile(context)
        assertTrue("Express INT8 model is missing at ${modelFile.absolutePath}", modelFile.isFile)
        assertTrue("Express 270M model must require GPU execution", entry.requireGpu)
        assertTrue("Express 270M model is incomplete: ${modelFile.length()} bytes", modelFile.length() >= entry.minimumFileSizeBytes)

        val contract = context.assets.open("pipeline_prompts/genui_gen_a2ui_express_training_v1.md")
            .bufferedReader()
            .use { it.readText() }
        val systemPrompt = if (gpuControlProbe) {
            ""
        } else {
            contract.replace(
                "{response_text}",
                "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]",
            )
        }
        val responseText = """
            Weather comparison for Bengaluru
            Today is warm with a high of 29 C and a low of 21 C. Rain chance is 35 percent.
            Tomorrow is cloudy with a high of 27 C and a low of 20 C. Rain chance is 55 percent.
        """.trimIndent()
        val prompt = if (gpuControlProbe) {
            "Say hello in one short sentence."
        } else {
            "Create A2UI Express v1 GenUI IR for this response:\n\n$responseText"
        }
        File(resultDir, "system_prompt.txt").writeText(systemPrompt)
        File(resultDir, "prompt.txt").writeText(prompt)

        val response = OnDeviceLitertBackend(
            modelFile.absolutePath,
            acceleratorPreference = accelerator,
        ).generate(
            InferenceBackend.GenerateRequest(
                prompt = prompt,
                systemPrompt = systemPrompt,
                initialMessages = if (gpuControlProbe) emptyList() else listOf(
                    InferenceBackend.ConversationMessage(
                        role = InferenceBackend.ConversationRole.USER,
                        content = "Create A2UI Express v1 GenUI IR for this response:\n\nA small travel checklist with a title and two items.",
                    ),
                    InferenceBackend.ConversationMessage(
                        role = InferenceBackend.ConversationRole.MODEL,
                        content = "<a2ui>\nroot=Column([a,b])\na=Text(\"Travel Checklist\",\"h1\")\nb=List([c,d])\nc=Text(\"Passport\")\nd=Text(\"Charger\")\n</a2ui>",
                    ),
                ),
                temperature = temperature,
                maxOutputTokens = entry.maxOutputTokens,
                jsonMode = false,
                structuredOutput = false,
            )
        )
        File(resultDir, "raw.txt").writeText(response.text)
        File(resultDir, "metrics.txt").writeText(
            "runtimeBackend=${response.runtimeBackend}\n" +
                "elapsedMs=${response.streamDurationMs}\n" +
                "inputTokens=${response.inputTokens}\n" +
                "outputTokens=${response.outputTokens}\n" +
                "outputTokensPerSecond=${response.outputTokensPerSecond}\n" +
                "error=${response.error.orEmpty()}\n"
        )
        assertEquals(expectedBackend, response.runtimeBackend)
        assertNull("$expectedBackend generation failed: ${response.error}", response.error)
        assertTrue("GPU response was empty", response.text.isNotBlank())
        if (gpuControlProbe) {
            assertTrue(
                "GPU control response was invalid: ${response.text.take(240)}",
                (response.outputTokens ?: 0) > 1 && !response.text.contains("<pad>"),
            )
            return
        }
        assertTrue("Response was not A2UI Express: ${response.text.take(240)}", response.text.contains("<a2ui>"))

        val wire = runCatching {
            com.samsung.genuicraft.pipeline.A2uiExpressCodec.decode(response.text)
        }.getOrElse { error("A2UI Express validation failed: ${it.message}") }
        // Keep the model-facing Express completion at the production renderer
        // boundary.  ``wire.toString()`` is the internal FlatSpec graph and is
        // intentionally rejected by the strict renderer unless an explicit
        // migration importer is used.
        val renderResult = GenUiNativeRenderer.render(response.text, sourceDir = null)
        assertNull("Native render failed: ${renderResult.errorMessage}", renderResult.errorMessage)
        assertTrue("No native render surface was produced", renderResult.surfaces.isNotEmpty())

        RenderSessionStore.update(
            sourceLabel = "gemma270m_a2ui_express_instrumentation",
            records = listOf(
                GenUiRecord(
                    title = "gemma270m_a2ui_express_gpu",
                    rawJson = response.text,
                    sourceDir = null,
                    sourceLabel = "instrumentation",
                    uiId = "gemma270m_a2ui_express_gpu",
                    summary = null,
                    queryId = null,
                    responseId = null,
                )
            ),
            renderMode = RenderMode.NATIVE,
        )
        val renderIntent = Intent(context, RenderActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            putExtra(RenderActivity.EXTRA_RECORD_INDEX, 0)
        }
        ActivityScenario.launch<RenderActivity>(renderIntent).use {
            instrumentation.waitForIdleSync()
            Thread.sleep(2_000L)
            assertTrue(
                "Failed to capture the rendered Gemma 270M Express result",
                UiDevice.getInstance(instrumentation).takeScreenshot(File(resultDir, "screenshot.png")),
            )
        }
    }
}
