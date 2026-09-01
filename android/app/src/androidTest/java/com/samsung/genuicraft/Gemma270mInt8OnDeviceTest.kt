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
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class Gemma270mInt8OnDeviceTest {
    @Test
    fun generateValidateAndRenderOnMobileGpu() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val resultDir = File(
            context.getExternalFilesDir(null) ?: context.filesDir,
            "result_gemma270m_int8",
        ).apply { mkdirs() }
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma3_270m_ir_int8" }
        )
        val modelFile = entry.localFile(context)
        assertTrue("INT8 model is missing at ${modelFile.absolutePath}", modelFile.isFile)
        assertTrue("Gemma 270M must require GPU execution", entry.requireGpu)

        val stage2Response = """
            ## Berlin Mitte Rental Market Overview
            Media: Icon=<url1>
            Finding a 2-bedroom apartment in Mitte under **€1,800 warm** is competitive. While Mitte is the city center, looking at the fringes of the district or adjacent areas like Prenzlauer Berg or Moabit can increase your options while maintaining a short commute to Alexanderplatz.

            ## Priority Viewing Checklist
            To ensure the apartment meets your specific needs for space, budget, and commute, prioritize these five criteria during your appointments:

            | Priority | Criteria | What to Verify | Why it Matters |
            | :--- | :--- | :--- | :--- |
            | 1 | **Warm Rent Total** | Confirm "Warmmiete" includes heating and water | Ensures you stay under the **€1,800** limit |
            | 2 | **Balcony Access** | Check size, orientation, and noise levels | Confirms a primary requirement for the couple |
            | 3 | **Transit Route** | Test a live route to Alexanderplatz via VBB/Google Maps | Ensures the **30-minute** commute is realistic |
            | 4 | **Layout/Rooms** | Verify the second bedroom is a legal room (not a "Kammer") | Ensures comfort for a couple needing two distinct rooms |
            | 5 | **Energy Rating** | Check the "Energieausweis" (Energy Certificate) | Prevents unexpected spikes in heating costs in winter |

            ## Recommended Search Strategies
            Media: Icon=<url2>

            Option 1: ImmoScout24 | Germany's largest portal; best for volume | High competition
            Action: [Button: Search ImmoScout24] <<url3>>

            Option 2: WG-Gesucht | Great for smaller apartments and couples | More flexible landlords
            Action: [Button: Search WG-Gesucht] <<url4>>

            Option 3: Immowelt | Strong presence in Berlin Mitte | Diverse listing types
            Action: [Button: Search Immowelt] <<url5>>

            ## Quick Actions
            Action: [Button: Berlin Transit Map] <<url6>>
            Action: [Button: Rent Index Berlin] <<url7>>

            ## Sources
            - VBB Transit Authority: <<url6>>
            - Berlin City Portal: <<url8>>
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
                structuredOutput = false,
            )
        )
        File(resultDir, "raw.txt").writeText(response.text)
        File(resultDir, "metrics.txt").writeText(
            "runtimeBackend=${response.runtimeBackend}\n" +
                "elapsedMs=${response.streamDurationMs}\n" +
                "inputTokens=${response.inputTokens}\n" +
                "outputTokens=${response.outputTokens}\n" +
                "error=${response.error.orEmpty()}\n"
        )
        assertNull(response.error, response.error)
        assertEquals("GPU", response.runtimeBackend)
        assertTrue("The mobile GPU returned an empty response", response.text.isNotBlank())

        val extracted = PipelineJsonExtractor.extractJsonElement(response.text)
        assertNotNull("The mobile GPU response did not contain JSON", extracted)
        val contract = FlatSpecContract.coerceAndValidate(requireNotNull(extracted))
        assertTrue(contract.error.orEmpty(), contract.isValid)
        val spec = requireNotNull(contract.spec)
        File(resultDir, "ir.json").writeText(spec.toString())
        assertTrue(spec.toString().contains("Berlin", ignoreCase = true))
        assertTrue(spec.toString().contains("Mitte", ignoreCase = true))

        val renderResult = GenUiNativeRenderer.render(spec.toString(), sourceDir = null)
        assertNull(renderResult.errorMessage, renderResult.errorMessage)
        assertEquals(1, renderResult.surfaces.size)
        assertFalse(renderResult.surfaces.single().flatSpec?.elements.isNullOrEmpty())

        RenderSessionStore.update(
            sourceLabel = "instrumentation",
            records = listOf(
                GenUiRecord(
                    title = "gemma270m_int8_mobile_gpu",
                    rawJson = spec.toString(),
                    sourceDir = null,
                    sourceLabel = "instrumentation",
                    uiId = "gemma270m_int8_mobile_gpu",
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
                "Failed to capture the rendered Gemma 270M result",
                UiDevice.getInstance(instrumentation)
                    .takeScreenshot(File(resultDir, "screenshot.png")),
            )
        }
    }
}
