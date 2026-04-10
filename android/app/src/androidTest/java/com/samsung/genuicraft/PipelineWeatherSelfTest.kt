package com.samsung.genuicraft

import android.content.Intent
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import com.google.gson.JsonParser
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class PipelineWeatherSelfTest {
    private companion object {
        const val TAG = "PipelineWeatherSelfTest"
    }

    @Test
    fun runWeatherQueryAndVerifyRenderedUi() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val resultDir = File(context.getExternalFilesDir(null) ?: context.filesDir, "result_weather")

        InferenceBackendSettings.setResponseProvider(context, InferenceBackendSettings.Provider.GEMINI)
        InferenceBackendSettings.setIrProvider(context, InferenceBackendSettings.Provider.GEMINI)

        if (resultDir.exists()) {
            resultDir.listFiles()?.forEach { it.delete() }
        }
        resultDir.mkdirs()

        val query = "show weather in bengaluru"
        File(resultDir, "query.txt").writeText(query)

        val pipeline = GenUiStagePipeline(context.applicationContext)
        val outcome = pipeline.execute(queryText = query, onStageUpdate = {})

        when (outcome) {
            is GenUiStagePipeline.Outcome.Success -> {
                val result = outcome.result
                File(resultDir, "response.txt").writeText(result.stage2Response)
                File(resultDir, "ir.json").writeText(result.stage3Json)
                File(resultDir, "ir.txt").writeText(result.stage3Json)
                File(resultDir, "prompt_response.txt").writeText(result.stage2Prompt)
                File(resultDir, "prompt_ir.txt").writeText(
                    buildString {
                        if (!result.stage3SystemPrompt.isNullOrBlank()) {
                            appendLine("## system_prompt")
                            appendLine(result.stage3SystemPrompt)
                            appendLine()
                        }
                        appendLine("## user_prompt")
                        appendLine(result.stage3Prompt)
                    }
                )

                val irElement = JsonParser.parseString(result.stage3Json)
                assertTrue("IR must be a JSON object", irElement.isJsonObject)
                assertFalse("IR must not be a legacy message array", irElement.isJsonArray)
                val irObject = irElement.asJsonObject
                assertNotNull("IR root is required", irObject.get("root"))
                assertNotNull("IR elements is required", irObject.get("elements"))
                val rootId = irObject.get("root")?.asString.orEmpty()
                assertTrue("IR root id must be non-empty", rootId.isNotBlank())
                val elementsObject = irObject.getAsJsonObject("elements")
                assertTrue(
                    "IR elements must include root id",
                    elementsObject.has(rootId)
                )
                val elementTypes = elementsObject.entrySet()
                    .mapNotNull { (_, value) ->
                        value.takeIf { it.isJsonObject }
                            ?.asJsonObject
                            ?.get("type")
                            ?.takeIf { it.isJsonPrimitive }
                            ?.asString
                            ?.trim()
                    }
                    .filter { it.isNotBlank() }
                    .map { it.lowercase() }
                val textHeavyFallbackShape = elementTypes.count { it == "text" } == 1 &&
                    elementTypes.all {
                        it == "text" || it == "stack" || it == "column" || it == "row" || it == "list" || it == "card"
                    }
                Log.i(
                    TAG,
                    "stage3 elements=${elementsObject.size()} types=${elementTypes.distinct()} " +
                        "usedFallback=${result.usedFallback} textHeavyFallbackShape=$textHeavyFallbackShape"
                )
                Log.i(TAG, "stage3 warnings=${result.warnings.joinToString(" | ")}")
                Log.i(TAG, "stage3 preview=${result.stage3Json.take(1200)}")
                assertFalse(
                    "Strict IR-only mode must not use Stage 3 fallback JSON.",
                    result.warnings.any { it.contains("fallback JSON was used.", ignoreCase = true) }
                )
                assertFalse(
                    "Strict IR-only mode must not use fallback UI rendering.",
                    result.warnings.any { it.contains("using fallback UI", ignoreCase = true) }
                )
                val renderSurface = result.renderResult.surfaces.firstOrNull()
                Log.i(
                    TAG,
                    "render surfaceCount=${result.renderResult.surfaces.size} " +
                        "flatSpecSurface=${renderSurface?.flatSpec != null} " +
                        "legacyComponentCount=${renderSurface?.components?.size ?: 0}"
                )
                assertFalse(
                    "Text-heavy fallback should use legacy structured rendering path.",
                    textHeavyFallbackShape && renderSurface?.flatSpec != null
                )

                val record = GenUiRecord(
                    title = "weather_test",
                    rawJson = result.stage3Json,
                    sourceDir = null,
                    sourceLabel = "instrumentation",
                    uiId = "weather_test",
                    summary = null,
                    queryId = null,
                    responseId = null
                )
                RenderSessionStore.update(
                    sourceLabel = "instrumentation",
                    records = listOf(record),
                    renderMode = RenderMode.NATIVE
                )

                val renderIntent = Intent(context, RenderActivity::class.java).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                    putExtra(RenderActivity.EXTRA_RECORD_INDEX, 0)
                }
                ActivityScenario.launch<RenderActivity>(renderIntent).use {
                    instrumentation.waitForIdleSync()
                    Thread.sleep(2500L)
                    val screenshot = File(resultDir, "screenshot.png")
                    val captured = UiDevice.getInstance(instrumentation).takeScreenshot(screenshot)
                    assertTrue("Failed to capture screenshot.", captured)
                }
            }

            is GenUiStagePipeline.Outcome.Failure -> {
                File(resultDir, "response.txt").writeText(outcome.stage2Response.orEmpty())
                File(resultDir, "ir.json").writeText(outcome.stage3Json.orEmpty())
                File(resultDir, "ir.txt").writeText(outcome.stage3Json.orEmpty())
                File(resultDir, "prompt_response.txt").writeText("Stage 2 prompt unavailable due to failure.")
                File(resultDir, "prompt_ir.txt").writeText("Pipeline failure: ${outcome.message}")
                throw AssertionError("Pipeline failed at ${outcome.stage}: ${outcome.message}")
            }
        }
    }
}
