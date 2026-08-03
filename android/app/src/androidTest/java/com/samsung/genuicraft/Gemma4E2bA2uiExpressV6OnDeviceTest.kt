package com.samsung.genuicraft

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.pipeline.IrPromptVersionSettings
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class Gemma4E2bA2uiExpressV6OnDeviceTest {
    @Test
    fun v6GeneratesA2uiExpressOnGpuAndRenders() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val resultRoot = File(context.filesDir, "result_gemma4_e2b_a2ui_express_v6").apply { mkdirs() }
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma4_e2b_a2ui_express_v6_litert" }
        )
        val targetModelFile = entry.localFile(context)
        val internalModelFile = File(File(context.filesDir, "on_device_models"), entry.fileName)
        val modelFile = listOf(targetModelFile, internalModelFile)
            .firstOrNull(File::isFile)
            ?: error(
                "Gemma 4 E2B A2UI Express v6 is missing from both " +
                    "${targetModelFile.absolutePath} and ${internalModelFile.absolutePath}"
            )
        assertTrue(
            "The v6 model is incomplete: ${modelFile.length()} bytes",
            modelFile.length() >= entry.minimumFileSizeBytes,
        )
        assertTrue("The v6 profile must require GPU for this exported artifact", entry.requireGpu)

        val originalIrProvider = InferenceBackendSettings.getIrProvider(context)
        val originalModelPath = InferenceBackendSettings.getOnDeviceModelPath(context)
        val originalAccelerator = InferenceBackendSettings.getOnDeviceAccelerator(context)
        val originalFormatId = IrPromptVersionSettings.getSelectedVersionId(context)
        val query = "Show the delivery status for order A-1042."
        val stage2Response = """
            ## Order A-1042

            Your package is out for delivery and should arrive today by 4:30 PM.

            | Detail | Value |
            | :--- | :--- |
            | Carrier | SwiftShip |
            | Current status | Out for delivery |
            | Estimated arrival | Today, 4:30 PM |

            Action: [Button: Track package] <<https://www.samsung.com/support/orders/A-1042>>
        """.trimIndent()

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
                    queryText = query,
                    stage2ResponseText = stage2Response,
                ) { update ->
                    synchronized(updates) {
                        updates += "${update.stage}: ${update.message} backend=${update.llmRuntimeBackend.orEmpty()}"
                    }
                }
            }
            File(resultRoot, "updates.txt").writeText(updates.joinToString("\n"))

            val success = outcome as? GenUiStagePipeline.Outcome.Success
                ?: error("v6 Stage 3 generation failed: $outcome")
            val result = success.result
            File(resultRoot, "stage3_prompt.txt").writeText(result.stage3Prompt)
            File(resultRoot, "stage3.json").writeText(result.stage3Json)
            File(resultRoot, "warnings.txt").writeText(result.warnings.joinToString("\n"))
            File(resultRoot, "metrics.txt").writeText(
                "runtimeBackend=${result.stage3RuntimeBackend}\n" +
                    "inputTokens=${result.stage3InputTokens}\n" +
                    "outputTokens=${result.stage3OutputTokens}\n" +
                    "outputTokensPerSecond=${result.stage3OutputTokensPerSecond}\n" +
                    "stageDurationsMs=${result.stageDurationsMs}\n"
            )

            assertTrue(
                "The trained v6 model did not run on GPU: ${result.stage3RuntimeBackend}",
                result.stage3RuntimeBackend.orEmpty().startsWith("GPU"),
            )
            assertTrue("The generated IR lost the order id", result.stage3Json.contains("A-1042"))
            assertTrue("The generated IR lost the carrier", result.stage3Json.contains("SwiftShip"))
            assertTrue("The generated IR lost the delivery status", result.stage3Json.contains("Out for delivery"))
            assertFalse("The generated IR is empty", result.stage3Json.isBlank())
            assertTrue("No native render surface was produced", result.renderResult.surfaces.isNotEmpty())
            assertTrue(
                "Native render failed: ${result.renderResult.errorMessage}",
                result.renderResult.errorMessage.isNullOrBlank(),
            )

            RenderSessionStore.update(
                sourceLabel = "gemma4_e2b_a2ui_express_v6_instrumentation",
                records = listOf(
                    GenUiRecord(
                        title = "gemma4_e2b_a2ui_express_v6_gpu",
                        rawJson = result.stage3Json,
                        sourceDir = null,
                        sourceLabel = "instrumentation",
                        uiId = "gemma4_e2b_a2ui_express_v6_gpu",
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
                    "Failed to capture the rendered v6 result",
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
