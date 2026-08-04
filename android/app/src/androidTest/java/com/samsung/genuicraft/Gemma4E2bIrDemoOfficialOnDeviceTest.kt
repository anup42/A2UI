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
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class Gemma4E2bIrDemoOfficialOnDeviceTest {

    @Test
    fun allDefaultIrDemoCasesGenerateAndRenderWithOfficialGemma() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val records = IrDemoRecordLoader.loadDefault(context.assets)
        assertEquals("The IR Demo sweep must contain the default ten cases", 10, records.size)

        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma4_e2b_it_litert" }
        )
        val targetModelFile = entry.localFile(context)
        val internalModelFile = File(File(context.filesDir, "on_device_models"), entry.fileName)
        val modelFile = listOf(targetModelFile, internalModelFile).firstOrNull(File::isFile)
            ?: error("Official Gemma 4 E2B model is not installed.")

        val originalIrProvider = InferenceBackendSettings.getIrProvider(context)
        val originalModelPath = InferenceBackendSettings.getOnDeviceModelPath(context)
        val originalAccelerator = InferenceBackendSettings.getOnDeviceAccelerator(context)
        val originalFormatId = IrPromptVersionSettings.getSelectedVersionId(context)
        val resultRoot = File(context.filesDir, "result_gemma4_e2b_ir_demo_official").apply {
            deleteRecursively()
            mkdirs()
        }
        val summaries = mutableListOf<String>()

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

            records.forEachIndexed { index, record ->
                val caseDir = File(resultRoot, "%02d_%s".format(index + 1, record.queryId)).apply { mkdirs() }
                File(caseDir, "query.txt").writeText(record.queryText)
                File(caseDir, "response.txt").writeText(record.responseText)
                val updates = mutableListOf<String>()

                try {
                    val outcome = withTimeout(4 * 60_000L) {
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
                    File(caseDir, "updates.txt").writeText(updates.joinToString("\n"))

                    when (outcome) {
                        is GenUiStagePipeline.Outcome.Success -> {
                            val result = outcome.result
                            File(caseDir, "stage3_system_prompt.txt").writeText(
                                result.stage3SystemPrompt.orEmpty()
                            )
                            File(caseDir, "stage3_prompt.txt").writeText(result.stage3Prompt)
                            File(caseDir, "stage3_output.txt").writeText(result.stage3Json)
                            File(caseDir, "warnings.txt").writeText(result.warnings.joinToString("\n"))
                            File(caseDir, "metrics.txt").writeText(
                                "runtimeBackend=${result.stage3RuntimeBackend}\n" +
                                    "inputTokens=${result.stage3InputTokens}\n" +
                                    "outputTokens=${result.stage3OutputTokens}\n" +
                                    "outputTokensPerSecond=${result.stage3OutputTokensPerSecond}\n" +
                                    "stageDurationsMs=${result.stageDurationsMs}\n"
                            )
                            val decoded = runCatching { A2uiExpressCodec.decode(result.stage3Json) }
                                .getOrElse { error("Express decode failed: ${it.message}") }
                            val missingFacts = ResponseFactCoverage.missingFacts(
                                result.stage2Response,
                                decoded,
                            )
                            val missingText = missingFacts.joinToString(";") { "${it.label}=${it.value}" }
                            File(caseDir, "missing_facts.txt").writeText(missingText)
                            assertTrue(
                                "Authoritative response facts missing for ${record.queryId}: $missingText",
                                missingFacts.isEmpty(),
                            )
                            assertTrue(
                                "No native render surface for ${record.queryId}",
                                result.renderResult.surfaces.isNotEmpty(),
                            )
                            assertTrue(
                                "Native render failed for ${record.queryId}: ${result.renderResult.errorMessage}",
                                result.renderResult.errorMessage.isNullOrBlank(),
                            )
                            assertTrue(
                                "Official Gemma profile was not used for ${record.queryId}",
                                result.stage3SystemPrompt.orEmpty().contains("Official Gemma 4 E2B LiteRT A2UI Express profile"),
                            )
                            assertFalse(
                                "Raw media markup leaked into the rendered Text node for ${record.queryId}",
                                result.stage3Json.contains("Media: Icon") ||
                                    result.stage3Json.contains("Media: Image"),
                            )
                            assertFalse(
                                "Raw action markup leaked into the rendered Text node for ${record.queryId}",
                                result.stage3Json.contains("Action: [Button:"),
                            )
                            assertFalse(
                                "Raw table separator leaked into the rendered Text node for ${record.queryId}",
                                result.stage3Json.contains("| :--- |"),
                            )
                            val renderRecord = GenUiRecord(
                                title = "gemma4_e2b_ir_demo_${record.queryId}",
                                rawJson = result.stage3Json,
                                sourceDir = null,
                                sourceLabel = "instrumentation",
                                uiId = "gemma4_e2b_ir_demo_${record.queryId}",
                                summary = null,
                                queryId = record.queryId,
                                responseId = null,
                            )
                            RenderSessionStore.update(
                                sourceLabel = "instrumentation",
                                records = listOf(renderRecord),
                                renderMode = RenderMode.NATIVE,
                            )
                            ActivityScenario.launch<RenderActivity>(
                                Intent(context, RenderActivity::class.java).apply {
                                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                                    putExtra(RenderActivity.EXTRA_RECORD_INDEX, 0)
                                }
                            ).use {
                                instrumentation.waitForIdleSync()
                                Thread.sleep(1_000L)
                                assertTrue(
                                    "Rendered screen capture failed for ${record.queryId}",
                                    UiDevice.getInstance(instrumentation)
                                        .takeScreenshot(File(caseDir, "screenshot.png")),
                                )
                            }
                            summaries += listOf(
                                record.queryId,
                                "PASS",
                                result.stage3RuntimeBackend.orEmpty(),
                                missingText.replace('\t', ' '),
                            ).joinToString("\t")
                        }

                        is GenUiStagePipeline.Outcome.Failure -> {
                            File(caseDir, "failure.txt").writeText(
                                "stage=${outcome.stage}\nmessage=${outcome.message}\n" +
                                    "stage3=${outcome.stage3Json.orEmpty()}\n"
                            )
                            summaries += listOf(
                                record.queryId,
                                "FAIL",
                                outcome.stage.name,
                                outcome.message.replace('\t', ' '),
                            ).joinToString("\t")
                        }
                    }
                } catch (failure: Throwable) {
                    File(caseDir, "failure.txt").writeText(
                        "${failure.javaClass.simpleName}: ${failure.message}\n"
                    )
                    summaries += listOf(
                        record.queryId,
                        "FAIL",
                        "EXCEPTION",
                        (failure.message ?: failure.javaClass.simpleName).replace('\t', ' '),
                    ).joinToString("\t")
                }
            }

            File(resultRoot, "summary.tsv").writeText(
                "query_id\tstatus\truntime_or_stage\tmissing_facts_or_error\n" +
                    summaries.joinToString("\n")
            )
            assertTrue(
                "Official Gemma IR Demo failures: ${summaries.filter { it.contains("\tFAIL\t") }}",
                summaries.none { it.contains("\tFAIL\t") },
            )
        } finally {
            InferenceBackendSettings.setIrProvider(context, originalIrProvider)
            InferenceBackendSettings.setOnDeviceModelPath(context, originalModelPath)
            InferenceBackendSettings.setOnDeviceAccelerator(context, originalAccelerator)
            IrPromptVersionSettings.setSelectedVersionId(context, originalFormatId)
        }
    }
}
