package com.samsung.genuicraft

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import com.google.gson.JsonParser
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.mcp.McpUrlShortener
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
            | Receipt asset | ../assets/orders/A-1042/receipt.json |

            Action: [Button: Track package] <<https://www.samsung.com/support/orders/A-1042>>
        """.trimIndent()

        val runtimeMaskProbe =
            "Action=https://www.samsung.com/support/orders/A-1042 Asset=../assets/orders/A-1042/receipt.json"
        val runtimeMaskResult = McpUrlShortener.shorten(runtimeMaskProbe)
        assertFalse("Android runtime leaked a URL into masked text", runtimeMaskResult.shortenedText.contains("https://"))
        assertFalse("Android runtime leaked an asset path into masked text", runtimeMaskResult.shortenedText.contains("../assets/"))
        assertTrue("Android runtime did not create both placeholders", runtimeMaskResult.urlMap.size == 2)
        assertTrue(
            "Android runtime did not restore references exactly",
            McpUrlShortener.restore(runtimeMaskResult.shortenedText, runtimeMaskResult.urlMap) == runtimeMaskProbe,
        )
        val runtimeJsonRestore = McpUrlShortener.restoreJsonReferences(
            JsonParser.parseString("""{"action":"{{u1}}","asset":"{{u2}}"}"""),
            runtimeMaskResult.urlMap,
        ).jsonElement.asJsonObject
        assertTrue(
            "Android JSON restoration lost the URL",
            runtimeJsonRestore.get("action").asString == "https://www.samsung.com/support/orders/A-1042",
        )
        assertTrue(
            "Android JSON restoration lost the asset path",
            runtimeJsonRestore.get("asset").asString == "../assets/orders/A-1042/receipt.json",
        )

        try {
            InferenceBackendSettings.setIrProvider(
                context,
                InferenceBackendSettings.Provider.ON_DEVICE_LITERT,
            )
            InferenceBackendSettings.setOnDeviceModelPath(context, modelFile.absolutePath)
            for (formatId in listOf("a2ui_express_v1")) {
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
                assertTrue(
                    "Reference masking was not reported for $formatId: ${result.warnings}",
                    result.warnings.any { it.contains("URL/asset reference") },
                )
                assertFalse(
                    "Raw action URL was sent to the IR model for $formatId",
                    result.stage3Prompt.contains("https://www.samsung.com/support/orders/A-1042"),
                )
                assertFalse(
                    "Raw local asset path was sent to the IR model for $formatId",
                    result.stage3Prompt.contains("../assets/orders/A-1042/receipt.json"),
                )
                assertFalse(
                    "A partial local asset path was sent to the IR model for $formatId",
                    result.stage3Prompt.contains("../assets/"),
                )
                assertTrue(
                    "Masked reference placeholder was not sent for $formatId",
                    result.stage3Prompt.contains("{{u1}}") &&
                        result.stage3Prompt.contains("{{u2}}") &&
                        result.stage3Prompt.contains("{{u3}}"),
                )
                assertTrue("Rendered output lost the order id for $formatId", result.stage3Json.contains("A-1042"))
                assertTrue("Rendered output lost the carrier for $formatId", result.stage3Json.contains("SwiftShip"))
                assertTrue("Rendered output lost the ETA for $formatId", result.stage3Json.contains("Today, 4:30 PM"))
                assertTrue(
                    "Rendered output lost the action for $formatId",
                    result.stage3Json.contains("emitEvent") || result.stage3Json.contains("onPress=Event"),
                )
                assertTrue(
                    "Rendered output action is not wired for $formatId",
                    result.stage3Json.contains("\"on\":") || result.stage3Json.contains("onPress=Event"),
                )
                assertFalse("Unresolved reference placeholder remained for $formatId", result.stage3Json.contains("{{u"))
                assertTrue(
                    "Rendered output lost grouped order details for $formatId",
                    result.stage3Json.contains("\"type\":\"Table\"") ||
                        result.stage3Json.contains("\"type\":\"Card\"") ||
                        result.stage3Json.contains("Table(") ||
                        result.stage3Json.contains("Card("),
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
