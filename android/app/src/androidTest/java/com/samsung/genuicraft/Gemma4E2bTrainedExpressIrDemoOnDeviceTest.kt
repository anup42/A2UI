package com.samsung.genuicraft

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.pipeline.IrPromptVersionSettings
import com.samsung.genuicraft.pipeline.ResponseFactCoverage
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class Gemma4E2bTrainedExpressIrDemoOnDeviceTest {

    @Test
    fun firstDefaultIrDemoCaseGeneratesAndRendersWithGpuMtp() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma4_e2b_trained_express" }
        )
        val modelFile = entry.localFile(context)
        assertTrue("Trained Express model is missing at ${modelFile.absolutePath}", modelFile.isFile)
        assertTrue(
            "Trained Express model is incomplete: ${modelFile.length()} bytes",
            modelFile.length() >= entry.minimumFileSizeBytes,
        )
        assertTrue("Trained Express must require GPU", entry.requireGpu)
        assertTrue("Trained Express must request the retained MTP assistant", entry.enableSpeculativeDecoding)
        assertTrue("Trained Express must use the QAT training prompt", entry.trainingCompatiblePrompt)

        val record = IrDemoRecordLoader.loadDefault(context.assets).first()
        assertEquals("q_001374", record.queryId)
        val resultRoot = File(context.filesDir, "result_gemma4_e2b_trained_express_ir_demo").apply {
            deleteRecursively()
            mkdirs()
        }
        File(resultRoot, "query.txt").writeText(record.queryText)
        File(resultRoot, "response.txt").writeText(record.responseText)

        val originalIrProvider = InferenceBackendSettings.getIrProvider(context)
        val originalModelPath = InferenceBackendSettings.getOnDeviceModelPath(context)
        val originalAccelerator = InferenceBackendSettings.getOnDeviceAccelerator(context)
        val originalFormatId = IrPromptVersionSettings.getSelectedVersionId(context)

        try {
            InferenceBackendSettings.setIrProvider(
                context,
                InferenceBackendSettings.Provider.ON_DEVICE_LITERT,
            )
            InferenceBackendSettings.setOnDeviceModelPath(context, modelFile.absolutePath)
            InferenceBackendSettings.setOnDeviceAccelerator(
                context,
                InferenceBackendSettings.Accelerator.GPU,
            )
            IrPromptVersionSettings.setSelectedVersionId(context, "a2ui_express_v1")

            val updates = mutableListOf<String>()
            val outcome = withTimeout(8 * 60_000L) {
                GenUiStagePipeline(context).executeStage3FromResponse(
                    queryText = record.queryText,
                    stage2ResponseText = record.responseText,
                ) { update ->
                    synchronized(updates) {
                        updates += "${update.stage}: ${update.message} " +
                            "backend=${update.llmRuntimeBackend.orEmpty()}"
                    }
                }
            }
            File(resultRoot, "updates.txt").writeText(updates.joinToString("\n"))

            val success = outcome as? GenUiStagePipeline.Outcome.Success
            if (success == null) {
                File(resultRoot, "failure.txt").writeText(outcome.toString())
                error("Trained Express IR Demo generation failed: $outcome")
            }
            val result = success.result
            File(resultRoot, "stage3_system_prompt.txt").writeText(result.stage3SystemPrompt.orEmpty())
            File(resultRoot, "stage3_prompt.txt").writeText(result.stage3Prompt)
            File(resultRoot, "stage3_output.txt").writeText(result.stage3Json)
            File(resultRoot, "warnings.txt").writeText(result.warnings.joinToString("\n"))
            File(resultRoot, "metrics.txt").writeText(
                "runtimeBackend=${result.stage3RuntimeBackend}\n" +
                    "inputTokens=${result.stage3InputTokens}\n" +
                    "outputTokens=${result.stage3OutputTokens}\n" +
                    "outputTokensPerSecond=${result.stage3OutputTokensPerSecond}\n" +
                    "stageDurationsMs=${result.stageDurationsMs}\n"
            )

            assertTrue(
                "Trained Express did not run with GPU+MTP: ${result.stage3RuntimeBackend}",
                result.stage3RuntimeBackend == "GPU+MTP",
            )
            assertTrue(
                "The runtime prompt did not match the training contract",
                result.stage3SystemPrompt.orEmpty().contains("A2UI Express v1"),
            )
            val decoded = A2uiExpressCodec.decode(result.stage3Json)
            val missingFacts = ResponseFactCoverage.missingFacts(result.stage2Response, decoded)
            File(resultRoot, "missing_facts.txt").writeText(
                missingFacts.joinToString("\n") { "${it.label}=${it.value}" }
            )
            assertTrue(
                "Generated IR omitted response facts: $missingFacts",
                missingFacts.isEmpty(),
            )
            assertTrue("Generated IR lost the hourly wage", result.stage3Json.contains("36.06"))
            assertTrue("Generated IR lost the annual salary", result.stage3Json.contains("75,000"))
            assertTrue("No native surface was produced", result.renderResult.surfaces.isNotEmpty())
            assertTrue(
                "Native render failed: ${result.renderResult.errorMessage}",
                result.renderResult.errorMessage.isNullOrBlank(),
            )

            RenderSessionStore.update(
                sourceLabel = "gemma4_e2b_trained_express_ir_demo",
                records = listOf(
                    GenUiRecord(
                        title = "Gemma 4 E2B Trained Express IR Demo",
                        rawJson = result.stage3Json,
                        sourceDir = null,
                        sourceLabel = "instrumentation",
                        uiId = "gemma4_e2b_trained_express_ir_demo",
                        summary = null,
                        queryId = record.queryId,
                        responseId = null,
                    )
                ),
                renderMode = RenderMode.NATIVE,
            )
            ActivityScenario.launch<RenderActivity>(
                Intent(context, RenderActivity::class.java).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                    putExtra(RenderActivity.EXTRA_RECORD_INDEX, 0)
                }
            ).use {
                instrumentation.waitForIdleSync()
                Thread.sleep(2_000L)
                assertTrue(
                    "Rendered IR Demo screenshot failed",
                    UiDevice.getInstance(instrumentation)
                        .takeScreenshot(File(resultRoot, "screenshot.png")),
                )
            }
        } finally {
            InferenceBackendSettings.setIrProvider(context, originalIrProvider)
            InferenceBackendSettings.setOnDeviceModelPath(context, originalModelPath)
            InferenceBackendSettings.setOnDeviceAccelerator(context, originalAccelerator)
            IrPromptVersionSettings.setSelectedVersionId(context, originalFormatId)
        }
    }
}
