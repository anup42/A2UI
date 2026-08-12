package com.samsung.genuicraft

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.pipeline.A2uiCanonicalGraph
import com.samsung.genuicraft.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.pipeline.PipelinePromptBuilder
import com.samsung.genuicraft.pipeline.ResponseFactCoverage
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Diagnostic probe that bypasses the asynchronous app stream merger and saves
 * LiteRT-LM's synchronous final message for target-only or MTP generation.
 */
@RunWith(AndroidJUnit4::class)
class Gemma4E2bTrainedExpressRawProbeTest {

    @OptIn(ExperimentalApi::class)
    @Test
    fun saveSynchronousFinalMessage() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val mtpEnabled = InstrumentationRegistry.getArguments()
            .getString("mtp")
            ?.toBooleanStrictOrNull()
            ?: false
        val mode = if (mtpEnabled) "gpu_mtp" else "gpu_target_only"
        val entry = requireNotNull(
            OnDeviceModelCatalog.entries.firstOrNull { it.id == "gemma4_e2b_trained_express" }
        )
        val modelFile = entry.localFile(context)
        assertTrue("Trained Express model is missing at ${modelFile.absolutePath}", modelFile.isFile)
        assertTrue(
            "Trained Express model is incomplete: ${modelFile.length()} bytes",
            modelFile.length() >= entry.minimumFileSizeBytes,
        )

        val record = IrDemoRecordLoader.loadDefault(context.assets).first()
        val template = PipelinePromptBuilder.loadPromptAsset(
            context.assets,
            PipelinePromptBuilder.STAGE3_A2UI_EXPRESS_PROMPT_ASSET,
        )
        val promptContext = PipelinePromptBuilder.prepareStage3PromptContext(
            template = template,
            trainingCompatible = true,
        )
        val prompt = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = promptContext.userTemplate,
            stage2Response = record.responseText,
            catalogId = "ignored",
            assets = emptyList(),
            appendRequestPolicies = false,
        )
        val resultDir = File(context.filesDir, "result_gemma4_e2b_trained_express_raw_probe")
            .apply { mkdirs() }
        val cacheDir = File(context.cacheDir, "gemma4_e2b_trained_express_raw_probe/$mode")
            .apply { mkdirs() }

        ExperimentalFlags.enableSpeculativeDecoding = mtpEnabled
        val engine = Engine(
            EngineConfig(
                modelPath = modelFile.canonicalPath,
                backend = Backend.GPU(),
                maxNumTokens = 4_096,
                cacheDir = cacheDir.absolutePath,
            )
        )
        try {
            engine.initialize()
            val finalMessage = engine.createConversation(
                ConversationConfig(
                    systemInstruction = promptContext.systemPrompt?.let { Contents.of(it) },
                    initialMessages = promptContext.initialMessages.map { message ->
                        when (message.role) {
                            InferenceBackend.ConversationRole.USER -> Message.user(message.content)
                            InferenceBackend.ConversationRole.MODEL -> Message.model(message.content)
                        }
                    },
                    samplerConfig = SamplerConfig(
                        temperature = 0.0,
                        topK = 1,
                        topP = 1.0,
                    ),
                    maxOutputToken = 256,
                )
            ).use { conversation ->
                conversation.sendMessage(prompt)
            }
            val output = finalMessage.contents.contents.joinToString(separator = "") { content ->
                when (content) {
                    is Content.Text -> content.text
                    else -> ""
                }
            }.trim()
            val strictResult = runCatching {
                val graph = A2uiExpressCodec.decode(output)
                graph to A2uiCanonicalGraph.validate(graph)
            }
            val decodedGraph = strictResult.getOrNull()?.first
            val validation = strictResult.getOrNull()?.second
            val missingFacts = decodedGraph?.let { graph ->
                ResponseFactCoverage.missingFacts(record.responseText, graph)
            }.orEmpty()
            val renderResult = decodedGraph?.let {
                GenUiNativeRenderer.render(output, sourceDir = null)
            }
            val renderSucceeded = renderResult?.let { rendered ->
                rendered.surfaces.isNotEmpty() && rendered.errorMessage.isNullOrBlank()
            } == true
            val strictError = strictResult.exceptionOrNull()?.message
                ?: validation?.error
                ?: renderResult?.errorMessage
                ?: ""
            val report = JSONObject()
                .put("mode", mode)
                .put("model_path", modelFile.canonicalPath)
                .put("prompt_characters", prompt.length)
                .put("output_characters", output.length)
                .put("starts_with_a2ui", output.startsWith("<a2ui>"))
                .put("strict_decode_succeeded", strictResult.isSuccess)
                .put("strict_validation_succeeded", validation?.isValid == true)
                .put("response_fact_coverage_succeeded", missingFacts.isEmpty())
                .put(
                    "missing_response_facts",
                    missingFacts.joinToString("; ") { "${it.label}=${it.value}" },
                )
                .put(
                    "native_render_succeeded",
                    renderSucceeded,
                )
                .put("strict_error", strictError)
            File(resultDir, "$mode-output.txt").writeText(output)
            File(resultDir, "$mode-report.json").writeText(report.toString(2) + "\n")
            assertTrue("LiteRT-LM returned an empty synchronous final message", output.isNotBlank())
            assertTrue("LiteRT-LM output did not start with <a2ui>", output.startsWith("<a2ui>"))
            assertTrue("Strict A2UI Express decode failed: $strictError", strictResult.isSuccess)
            assertTrue("Strict A2UI Express validation failed: $strictError", validation?.isValid == true)
            assertTrue("Generated IR omitted response facts: $missingFacts", missingFacts.isEmpty())
            assertTrue(
                "Native render failed: ${renderResult?.errorMessage}",
                renderSucceeded,
            )
        } finally {
            engine.close()
        }
    }
}
