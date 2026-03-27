package com.samsung.genuicraft

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class PipelineArtifactCaptureTest {

    @Test
    fun runFlightQueryAndCaptureArtifacts() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val resultDir = File(context.getExternalFilesDir(null) ?: context.filesDir, "result")

        if (resultDir.exists()) {
            resultDir.listFiles()?.forEach { it.delete() }
        }
        resultDir.mkdirs()

        val query = "Show flights from bengaluru to lucknow on 15th may"
        File(resultDir, "query.txt").writeText(query)

        val pipeline = GenUiStagePipeline(context.applicationContext)
        val outcome = pipeline.execute(queryText = query, onStageUpdate = {})

        when (outcome) {
            is GenUiStagePipeline.Outcome.Success -> {
                val result = outcome.result
                File(resultDir, "response.txt").writeText(result.stage2Response)
                File(resultDir, "ir.json").writeText(result.stage3Json)
                File(resultDir, "ir.txt").writeText(result.stage3Json)
                File(resultDir, "prompt_respose.txt").writeText(result.stage2Prompt)
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

                val record = GenUiRecord(
                    title = "flight_test",
                    rawJson = result.stage3Json,
                    sourceDir = null,
                    sourceLabel = "instrumentation",
                    uiId = "flight_test",
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
                File(resultDir, "prompt_respose.txt").writeText("Stage 2 prompt unavailable due to failure.")
                File(resultDir, "prompt_ir.txt").writeText("Pipeline failure: ${outcome.message}")
                throw AssertionError("Pipeline failed at ${outcome.stage}: ${outcome.message}")
            }
        }
    }
}
