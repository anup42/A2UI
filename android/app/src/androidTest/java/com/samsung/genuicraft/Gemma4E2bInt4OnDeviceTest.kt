package com.samsung.genuicraft

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.OnDeviceLitertBackend
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.pipeline.FlatSpecContract
import com.samsung.genuicraft.pipeline.PipelineJsonExtractor
import com.samsung.genuicraft.pipeline.PipelinePromptBuilder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class Gemma4E2bInt4OnDeviceTest {
    @Test
    fun generateValidateAndRenderOnMobileGpu() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val resultDir = File(
            context.getExternalFilesDir(null) ?: context.filesDir,
            "result_gemma4_e2b_int4",
        ).apply { mkdirs() }
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma4_e2b_ir_trained_int4" }
        )
        val modelFile = entry.localFile(context)
        assertTrue("INT4 model is missing at ${modelFile.absolutePath}", modelFile.isFile)
        assertTrue("The trained INT4 profile must forbid CPU fallback", entry.requireGpu)

        val stage2Response = """
            ## Your Rock Classics Queue
            Media: Icon=<url1>

            Your queue has been created with three legendary rock anthems. The tracks have been shuffled for a dynamic listening experience and playback has started.

            ## Shuffled Playback Order
            | Order | Song Title | Artist | Mood |
            | :--- | :--- | :--- | :--- |
            | 1 | Hotel California | Eagles | Atmospheric |
            | 2 | Bohemian Rhapsody | Queen | Operatic |
            | 3 | Stairway to Heaven | Led Zeppelin | Progressive |

            ## Quick Actions
            Action: [Button: Repeat Queue] <<url2>>
            Action: [Button: Manage Playlist] <<url3>>
        """.trimIndent()
        val prompt = requireNotNull(entry.stage3TrainingPromptPrefix) + stage2Response
        File(resultDir, "prompt.txt").writeText(prompt)

        val response = OnDeviceLitertBackend(modelFile.absolutePath).generate(
            InferenceBackend.GenerateRequest(
                prompt = prompt,
                systemPrompt = null,
                temperature = 0.0,
                maxOutputTokens = entry.maxOutputTokens,
                jsonMode = true,
                structuredOutput = true,
            )
        )
        File(resultDir, "raw.txt").writeText(response.text)
        File(resultDir, "metrics.txt").writeText(
            "elapsedMs=${response.streamDurationMs}\n" +
                "inputTokens=${response.inputTokens}\n" +
                "outputTokens=${response.outputTokens}\n" +
                "error=${response.error.orEmpty()}\n"
        )
        assertNull(response.error, response.error)
        assertTrue("The mobile GPU returned an empty response", response.text.isNotBlank())

        val cleanedResponse = PipelinePromptBuilder.stripStage3TrainingPromptLeak(
            responseText = response.text,
            trainingPromptPrefix = entry.stage3TrainingPromptPrefix,
        )
        val restoredResponse = cleanedResponse
            .replace("<url2>", "https://www.google.com/search?q=repeat+queue")
            .replace("<url3>", "https://www.google.com/search?q=manage+playlist")
        File(resultDir, "restored.txt").writeText(restoredResponse)
        val extracted = PipelineJsonExtractor.extractJsonElement(restoredResponse)
        assertNotNull("The mobile GPU response did not contain JSON", extracted)
        val contract = FlatSpecContract.coerceAndValidate(requireNotNull(extracted))
        assertTrue(contract.error.orEmpty(), contract.isValid)
        val spec = requireNotNull(contract.spec)
        File(resultDir, "ir.json").writeText(spec.toString())
        for (expected in listOf(
            "Hotel California",
            "Bohemian Rhapsody",
            "Stairway to Heaven",
            "Repeat Queue",
        )) {
            assertTrue(
                "The mobile GPU IR is missing the requested value: $expected",
                spec.toString().contains(expected, ignoreCase = true),
            )
        }

        val renderResult = GenUiNativeRenderer.render(spec.toString(), sourceDir = null)
        assertNull(renderResult.errorMessage, renderResult.errorMessage)
        assertEquals(1, renderResult.surfaces.size)
        assertFalse(renderResult.surfaces.single().flatSpec?.elements.isNullOrEmpty())

        val record = GenUiRecord(
            title = "gemma4_e2b_int4_mobile_gpu",
            rawJson = spec.toString(),
            sourceDir = null,
            sourceLabel = "instrumentation",
            uiId = "gemma4_e2b_int4_mobile_gpu",
            summary = null,
            queryId = null,
            responseId = null,
        )
        RenderSessionStore.update(
            sourceLabel = "instrumentation",
            records = listOf(record),
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
                "Failed to capture the rendered INT4 result",
                UiDevice.getInstance(instrumentation)
                    .takeScreenshot(File(resultDir, "screenshot.png")),
            )
        }
    }
}
