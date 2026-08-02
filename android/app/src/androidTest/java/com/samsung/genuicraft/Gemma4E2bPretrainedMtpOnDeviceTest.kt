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
class Gemma4E2bPretrainedMtpOnDeviceTest {
    @Test
    fun selectedIrFormatsGenerateAndRenderWithMtp() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val resultRoot = File(
            context.filesDir,
            "result_gemma4_e2b_pretrained_mtp",
        ).apply { mkdirs() }
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma4_e2b_it_litert" }
        )
        val targetModelFile = entry.localFile(context)
        val internalModelFile = File(File(context.filesDir, "on_device_models"), entry.fileName)
        val modelFile = listOf(targetModelFile, internalModelFile)
            .firstOrNull(File::isFile)
            ?: error(
                "Pretrained Gemma 4 E2B model is missing from both " +
                    "${targetModelFile.absolutePath} and ${internalModelFile.absolutePath}"
            )
        assertTrue("Pretrained Gemma 4 E2B model package is unexpectedly small", modelFile.length() > 2_500_000_000L)
        assertTrue("Pretrained Gemma 4 E2B must enable MTP", entry.enableSpeculativeDecoding)

        val originalIrProvider = InferenceBackendSettings.getIrProvider(context)
        val originalModelPath = InferenceBackendSettings.getOnDeviceModelPath(context)
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

            Action: [Button: Track package] <<https://example.com/orders/A-1042>>
        """.trimIndent()

        try {
            InferenceBackendSettings.setIrProvider(
                context,
                InferenceBackendSettings.Provider.ON_DEVICE_LITERT,
            )
            InferenceBackendSettings.setOnDeviceModelPath(context, modelFile.absolutePath)
            for (formatId in listOf("a2ui_express_v1", "compact_ir_v2")) {
                IrPromptVersionSettings.setSelectedVersionId(context, formatId)
                val formatDir = File(resultRoot, formatId).apply { mkdirs() }
                val updates = mutableListOf<String>()
                val outcome = withTimeout(8 * 60_000L) {
                    GenUiStagePipeline(context).executeStage3FromResponse(
                        queryText = query,
                        stage2ResponseText = stage2Response,
                    ) { update ->
                        synchronized(updates) {
                            updates += "${update.stage}: ${update.message} " +
                                "backend=${update.llmRuntimeBackend.orEmpty()}"
                        }
                    }
                }
                File(formatDir, "updates.txt").writeText(updates.joinToString("\n"))

                val success = outcome as? GenUiStagePipeline.Outcome.Success
                    ?: error("$formatId generation failed: $outcome")
                val result = success.result
                File(formatDir, "ir.json").writeText(result.stage3Json)
                File(formatDir, "warnings.txt").writeText(result.warnings.joinToString("\n"))
                File(formatDir, "metrics.txt").writeText(
                    "runtimeBackend=${result.stage3RuntimeBackend}\n" +
                        "inputTokens=${result.stage3InputTokens}\n" +
                        "outputTokens=${result.stage3OutputTokens}\n" +
                        "outputTokensPerSecond=${result.stage3OutputTokensPerSecond}\n" +
                        "stageDurationsMs=${result.stageDurationsMs}\n"
                )

                assertTrue(
                    "MTP was not reported for $formatId: ${result.stage3RuntimeBackend}",
                    result.stage3RuntimeBackend.orEmpty().contains("MTP"),
                )
                assertTrue(
                    "Selected format was not reported for $formatId: ${result.warnings}",
                    result.warnings.any { it.contains(IrPromptVersionSettings.getSelectedOption(context).title) },
                )
                assertTrue(
                    "MTP enablement was not reported for $formatId: ${result.warnings}",
                    result.warnings.any { it == "On-device Gemma MTP: enabled" },
                )
                assertTrue("Rendered output lost the order id for $formatId", result.stage3Json.contains("A-1042"))
                assertTrue("Rendered output lost the carrier for $formatId", result.stage3Json.contains("SwiftShip"))
                assertTrue("Rendered output lost the ETA for $formatId", result.stage3Json.contains("Today, 4:30 PM"))
                assertTrue("Rendered output lost the action for $formatId", result.stage3Json.contains("emitEvent"))
                assertTrue("Rendered output action is not wired for $formatId", result.stage3Json.contains("\"on\":"))
                assertTrue(
                    "Rendered output lost grouped order details for $formatId",
                    result.stage3Json.contains("\"type\":\"Table\"") ||
                        result.stage3Json.contains("\"type\":\"Card\""),
                )
                assertTrue("No rendered surface was produced for $formatId", result.renderResult.surfaces.isNotEmpty())
                assertTrue(
                    "Native render failed for $formatId: ${result.renderResult.errorMessage}",
                    result.renderResult.errorMessage.isNullOrBlank(),
                )

                val record = GenUiRecord(
                    title = "gemma4_e2b_mtp_$formatId",
                    rawJson = result.stage3Json,
                    sourceDir = null,
                    sourceLabel = "instrumentation",
                    uiId = "gemma4_e2b_mtp_$formatId",
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
                        "Failed to capture $formatId rendered result",
                        UiDevice.getInstance(instrumentation)
                            .takeScreenshot(File(formatDir, "screenshot.png")),
                    )
                }
                assertFalse("Generated IR was empty for $formatId", result.stage3Json.isBlank())
            }
        } finally {
            InferenceBackendSettings.setIrProvider(context, originalIrProvider)
            InferenceBackendSettings.setOnDeviceModelPath(context, originalModelPath)
            IrPromptVersionSettings.setSelectedVersionId(context, originalFormatId)
        }
    }
}
