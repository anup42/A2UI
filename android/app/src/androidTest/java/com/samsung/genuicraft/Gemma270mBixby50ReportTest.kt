package com.samsung.genuicraft

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.inference.InferenceBackend
import com.samsung.genuicraft.inference.OnDeviceLitertBackend
import com.samsung.genuicraft.mcp.McpUrlShortener
import com.samsung.genuicraft.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.pipeline.PipelinePromptBuilder
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.provider.LiteRtModelRunner
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs frozen Bixby50 responses through the reference app's real Stage 3 route. Stage 2 is
 * skipped, as in IR Demo. The separate SDK recovery probe is read-only with respect to the
 * pipeline result: it never makes a model call and never enables source-text fallback.
 *
 * Arguments: runId (required, fresh), cases (optional comma-separated IDs), label (optional),
 * backend=cpu|gpu|auto (default cpu), modelPath (optional catalog-named app-readable file),
 * expectedModelSha256 (optional), caseTimeoutMs (default 600000).
 * Artifacts: external-files/gemma270_bixby50/<runId>.
 */
@RunWith(AndroidJUnit4::class)
class Gemma270mBixby50ReportTest {
    @OptIn(ExperimentalApi::class)
    @Test
    fun runSavedResponsesThroughAppPipeline() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val args = InstrumentationRegistry.getArguments()
        val runId = requireNotNull(args.getString("runId")?.trim()) { "Pass a fresh -e runId." }
        require(SAFE_NAME.matches(runId)) { "runId must be 1..100 letters, digits, _ or -." }
        val label = args.getString("label", "gemma270m_bixby50")!!.trim().take(100)
        val backend = when (args.getString("backend", "cpu")!!.trim().lowercase()) {
            "cpu" -> InferenceBackendSettings.Accelerator.CPU
            "gpu" -> InferenceBackendSettings.Accelerator.GPU
            "auto" -> InferenceBackendSettings.Accelerator.AUTO
            else -> error("backend must be cpu, gpu, or auto")
        }
        val caseTimeoutMs = args.getString("caseTimeoutMs", "600000")!!.toLong().also {
            require(it in 1_000L..3_600_000L) { "caseTimeoutMs must be 1000..3600000." }
        }
        val corpusBytes = context.assets.open(CORPUS_ASSET).use { it.readBytes() }
        val corpusSha = sha256(corpusBytes)
        require(corpusSha == CORPUS_SHA256) { "Frozen Bixby50 corpus changed: $corpusSha" }
        val corpus = corpusBytes.toString(Charsets.UTF_8).lineSequence().filter(String::isNotBlank)
            .map { JsonParser.parseString(it).asJsonObject }.toList()
        val byId = corpus.associateBy { it.get("id").asString }
        require(corpus.size == 50 && byId.size == 50) { "Expected 50 unique Bixby50 rows." }
        val requested = args.getString("cases", "")!!.split(',').map(String::trim).filter(String::isNotBlank)
        require(requested.size == requested.toSet().size && requested.all(byId::containsKey)) {
            "cases must contain unique frozen Bixby50 IDs."
        }
        val rows = if (requested.isEmpty()) corpus else requested.map(byId::getValue)

        val entry = requireNotNull(OnDeviceModelCatalog.entries.firstOrNull { it.id == MODEL_ID })
        val model = (args.getString("modelPath")?.trim()?.takeIf(String::isNotBlank)
            ?.let(::File) ?: entry.localFile(context)).canonicalFile
        require(model.isFile && model.canRead() && model.length() >= entry.minimumFileSizeBytes) {
            "Gemma 270M model is missing, unreadable, or incomplete: ${model.absolutePath}"
        }
        require(OnDeviceModelCatalog.entryForModelPath(model.absolutePath)?.id == MODEL_ID) {
            "Model path must use the catalog filename ${entry.fileName} so the app selects its training prompt."
        }
        val modelSha = sha256(model)
        args.getString("expectedModelSha256")?.trim()?.takeIf(String::isNotBlank)?.let { expected ->
            require(modelSha.equals(expected, ignoreCase = true)) {
                "Model SHA-256 mismatch: expected $expected, observed $modelSha"
            }
        }
        val externalFiles = requireNotNull(context.getExternalFilesDir(null))
        val runDir = File(externalFiles, "gemma270_bixby50/$runId")
        require(!runDir.exists() && runDir.mkdirs()) { "runId already exists or cannot be created: $runDir" }
        val gson = GsonBuilder().serializeNulls().disableHtmlEscaping().setPrettyPrinting().create()
        val resultsFile = File(runDir, "results.jsonl")
        val selectedIds = rows.map { it.get("id").asString }
        val runStartedAtMs = System.currentTimeMillis()
        writeJson(File(runDir, "run_config.json"), linkedMapOf(
            "runId" to runId, "label" to label, "cases" to selectedIds,
            "corpusAsset" to CORPUS_ASSET, "corpusSha256" to corpusSha,
            "modelId" to MODEL_ID, "modelPath" to model.absolutePath,
            "modelBasename" to model.name, "modelSizeBytes" to model.length(),
            "modelSha256" to modelSha, "backendRequested" to backend.rawValue,
            "mtpRequested" to false, "benchmarkFlagEnabled" to true,
            "benchmarkFlagScope" to "entire instrumentation run; differs from normal UI",
            "profile" to "reference_app_executeStage3FromResponse",
            "stage2Skipped" to true, "caseTimeoutMs" to caseTimeoutMs,
            "sourceFallbackAllowedInAppPipeline" to false,
            "sdkRecoverySourceFallbackAllowed" to false,
            "promptAsset" to PipelinePromptBuilder.STAGE3_A2UI_EXPRESS_TRAINING_PROMPT_ASSET,
            "startedAtEpochMs" to runStartedAtMs,
            "device" to mapOf("manufacturer" to Build.MANUFACTURER, "model" to Build.MODEL,
                "hardware" to Build.HARDWARE, "sdkInt" to Build.VERSION.SDK_INT),
        ), gson)
        val oldProvider = InferenceBackendSettings.getIrProvider(context)
        val oldModel = InferenceBackendSettings.getOnDeviceModelPath(context)
        val oldBackend = InferenceBackendSettings.getOnDeviceAccelerator(context)
        val oldMtp = InferenceBackendSettings.getOnDeviceMtpEnabled(context)
        val oldBenchmark = ExperimentalFlags.enableBenchmark
        val device = UiDevice.getInstance(instrumentation)
        var completed = 0
        var rendered = 0
        var runError: String? = null
        try {
            // Engine flags are read during initialization. Release a previous cached engine first.
            LiteRtModelRunner.releaseCachedEngine()
            ExperimentalFlags.enableBenchmark = true
            InferenceBackendSettings.setIrProvider(context, InferenceBackendSettings.Provider.ON_DEVICE_LITERT)
            InferenceBackendSettings.setOnDeviceModelPath(context, model.absolutePath)
            InferenceBackendSettings.setOnDeviceAccelerator(context, backend)
            InferenceBackendSettings.setOnDeviceMtpEnabled(context, false)
            assertEquals(MODEL_ID, OnDeviceModelCatalog.selectedEntry(context)?.id)
            writeSummary(runDir, runId, selectedIds, completed, rendered, false, null, gson)

            for ((index, row) in rows.withIndex()) {
                val caseId = row.get("id").asString
                val caseDir = File(runDir, caseId).apply { check(mkdirs()) }
                val query = row.get("query").asString
                val source = row.get("text").asString
                writeJson(File(caseDir, "source.json"), row, gson)
                val lock = Any()
                val generations = mutableListOf<StreamCompletion>()
                val updates = mutableListOf<Map<String, Any?>>()
                var latestRaw = ""
                var latestBackend: String? = null
                var reportedTokens: Int? = null
                var reportedRate: Double? = null
                var metricsAreEstimated: Boolean? = null
                val startedNs = SystemClock.elapsedRealtimeNanos()
                val pssBeforeKb = Debug.getPss()
                var outcome: GenUiStagePipeline.Outcome? = null
                var pipelineError: String? = null
                try {
                    outcome = withTimeout(caseTimeoutMs) {
                        GenUiStagePipeline(context).executeStage3FromResponse(query, source) { update ->
                            synchronized(lock) {
                                update.stage3StreamText?.let { text ->
                                    if (text.isNotEmpty()) latestRaw = text
                                    if (update.stage3StreamComplete) {
                                        generations += StreamCompletion(
                                            text = text, backend = update.llmRuntimeBackend,
                                            outputTokens = update.llmOutputTokens,
                                            tokensPerSecond = update.llmOutputTokensPerSecond,
                                            estimated = update.streamMetricsAreEstimated,
                                            completedAtEpochMs = System.currentTimeMillis(),
                                        )
                                        latestBackend = update.llmRuntimeBackend
                                        reportedTokens = update.llmOutputTokens
                                        reportedRate = update.llmOutputTokensPerSecond
                                        metricsAreEstimated = update.streamMetricsAreEstimated
                                    }
                                }
                                if (update.stage3StreamText == null || update.stage3StreamComplete) {
                                    updates += linkedMapOf(
                                        "stage" to update.stage.name, "message" to update.message,
                                        "runtimeBackend" to update.llmRuntimeBackend,
                                        "debugLog" to update.debugLog,
                                        "atEpochMs" to System.currentTimeMillis(),
                                    )
                                }
                            }
                        }
                    }
                } catch (error: Throwable) {
                    pipelineError = "${error.javaClass.simpleName}: ${error.message.orEmpty()}"
                }
                // Stream callbacks are posted to Main; drain them before selecting raw output.
                runCatching { instrumentation.waitForIdleSync() }
                val pipelineWallMs = elapsedMs(startedNs)
                val raw: String
                val streamCompletions: List<StreamCompletion>
                val stageUpdates: List<Map<String, Any?>>
                val streamBackend: String?
                val streamTokens: Int?
                val streamRate: Double?
                val streamEstimated: Boolean?
                synchronized(lock) {
                    raw = latestRaw
                    streamCompletions = generations.toList()
                    stageUpdates = updates.toList()
                    streamBackend = latestBackend
                    streamTokens = reportedTokens
                    streamRate = reportedRate
                    streamEstimated = metricsAreEstimated
                }
                File(caseDir, "raw.express").writeText(raw)
                streamCompletions.forEachIndexed { streamIndex, completion ->
                    File(caseDir, "raw_generation_${streamIndex + 1}.express").writeText(completion.text)
                }
                writeJson(File(caseDir, "stream_generations.json"), streamCompletions.mapIndexed { n, item ->
                    linkedMapOf(
                        "index" to n + 1, "file" to "raw_generation_${n + 1}.express",
                        "sha256" to sha256(item.text.toByteArray()), "backend" to item.backend,
                        "outputTokens" to item.outputTokens,
                        "tokensPerSecond" to item.tokensPerSecond,
                        "metricsAreEstimated" to item.estimated,
                        "completedAtEpochMs" to item.completedAtEpochMs,
                    )
                }, gson)
                writeJson(File(caseDir, "updates.json"), stageUpdates, gson)
                val success = outcome as? GenUiStagePipeline.Outcome.Success
                val failure = outcome as? GenUiStagePipeline.Outcome.Failure
                val normalizedSource = success?.result?.stage2Response ?: failure?.stage2Response
                val promptSource = if (success != null) "pipeline_result" else "reconstructed_from_pipeline_source"
                val plannedPrompt = reconstructPrompt(context, normalizedSource ?: source)
                val finalPrompt = success?.result?.stage3Prompt ?: plannedPrompt.second
                val finalSystemPrompt = success?.result?.stage3SystemPrompt ?: plannedPrompt.first
                File(caseDir, "stage3_prompt.txt").writeText(finalPrompt)
                File(caseDir, "stage3_system_prompt.txt").writeText(finalSystemPrompt)
                val finalExpress = success?.result?.stage3Json
                var finalGraphDecoded = false
                if (finalExpress != null) {
                    File(caseDir, "final.express").writeText(finalExpress)
                    runCatching { A2uiExpressCodec.decode(finalExpress) }
                        .onSuccess { graph ->
                            File(caseDir, "final.a2ui.json").writeText(graph.toString())
                            finalGraphDecoded = true
                        }
                }
                val sdkRepair = probeSdkGeneratedDslRepair(raw, source, caseDir)
                val warnings = success?.result?.warnings.orEmpty()
                val repairKind = when {
                    success == null -> "unresolved"
                    warnings.any { it.contains("A2UI Express repair parsed natively") } -> "pipeline_model_repair"
                    warnings.any { it.contains("A2UI Express parsed natively") } -> "raw_valid"
                    else -> "pipeline_accepted_unknown_repair"
                }
                val invalidRepairs = warnings.count { it.startsWith("Stage 3 repair attempt ") && it.contains(" invalid (") }
                val failedRepairs = warnings.count { it.startsWith("Repair attempt ") && it.contains(" failed with backend error:") }
                val logicalRepairAttempts = when (repairKind) {
                    "raw_valid" -> 0
                    "pipeline_model_repair" -> invalidRepairs + failedRepairs + 1
                    else -> null
                }
                var screenCaptured = false
                var renderErrorVisible: Boolean? = null
                var visibleTextNodes: Int? = null
                var uiError: String? = null
                var uiWallMs: Long? = null
                if (success != null && finalExpress != null && success.result.renderResult.errorMessage.isNullOrBlank() &&
                    success.result.renderResult.surfaces.isNotEmpty()) {
                    val renderStartedNs = SystemClock.elapsedRealtimeNanos()
                    try {
                        RenderSessionStore.update(
                            sourceLabel = "gemma270_bixby50:$runId:$caseId",
                            records = listOf(GenUiRecord(
                                title = caseId, rawJson = finalExpress, sourceDir = null,
                                sourceLabel = "instrumentation", uiId = caseId, summary = null,
                                queryId = caseId, responseId = null,
                            )),
                            renderMode = RenderMode.NATIVE,
                        )
                        val intent = Intent(context, RenderActivity::class.java).apply {
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                            putExtra(RenderActivity.EXTRA_RECORD_INDEX, 0)
                        }
                        ActivityScenario.launch<RenderActivity>(intent).use {
                            instrumentation.waitForIdleSync()
                            Thread.sleep(750L)
                            screenCaptured = device.takeScreenshot(File(caseDir, "screen.png"))
                            device.dumpWindowHierarchy(File(caseDir, "window.xml"))
                            renderErrorVisible = device.hasObject(By.textContains("Unable to render GenUI"))
                            visibleTextNodes = Regex("text=\"([^\"]+)\"")
                                .findAll(File(caseDir, "window.xml").readText()).count()
                        }
                    } catch (error: Throwable) {
                        uiError = "${error.javaClass.simpleName}: ${error.message.orEmpty()}"
                    } finally {
                        uiWallMs = elapsedMs(renderStartedNs)
                    }
                }
                val status = when {
                    success == null -> if (pipelineError?.startsWith("TimeoutCancellationException") == true) "timeout" else "pipeline_failure"
                    uiError != null || !screenCaptured || renderErrorVisible == true -> "render_failure"
                    else -> "rendered"
                }
                if (status == "rendered") rendered++
                val result = linkedMapOf<String, Any?>(
                    "id" to caseId, "status" to status, "index" to index + 1,
                    "firstInRun" to (index == 0), "query" to query,
                    "sourceSha256" to sha256(source.toByteArray()),
                    "normalizedSourceSha256" to normalizedSource?.let { sha256(it.toByteArray()) },
                    "modelSha256" to modelSha, "backendRequested" to backend.rawValue,
                    "backendObserved" to (success?.result?.stage3RuntimeBackend ?: streamBackend),
                    "mtpRequested" to false, "mtpObserved" to
                        (success?.result?.stage3RuntimeBackend ?: streamBackend).orEmpty().contains("+MTP"),
                    "promptSource" to promptSource, "promptSha256" to sha256(finalPrompt.toByteArray()),
                    "systemPromptSha256" to sha256(finalSystemPrompt.toByteArray()),
                    "rawCaptured" to raw.isNotBlank(), "rawSha256" to sha256(raw.toByteArray()),
                    "rawGenerationCount" to streamCompletions.size,
                    "rawGenerationSelectedIndex" to streamCompletions.size.takeIf { it > 0 },
                    "rawGenerationFiles" to streamCompletions.indices.map { "raw_generation_${it + 1}.express" },
                    "repairKind" to repairKind,
                    "pipelineRepairAttemptsInferred" to logicalRepairAttempts,
                    "pipelineRepairRawExposed" to false,
                    "promptMatchesReconstruction" to (success?.result?.stage3Prompt?.let { it == plannedPrompt.second }),
                    "providerCalls" to null,
                    "providerCallsExact" to false,
                    "sourceFallbackUsed" to false,
                    "sdkRepairStatus" to sdkRepair.status,
                    "sdkRepairKind" to sdkRepair.kind,
                    "finalExpressPresent" to (finalExpress != null),
                    "finalGraphDecoded" to finalGraphDecoded,
                    "finalGraphFile" to if (finalGraphDecoded) "final.a2ui.json" else null,
                    "pipelineWallMs" to pipelineWallMs,
                    "stage3DurationMs" to (success?.result?.stageDurationsMs?.get(GenUiStagePipeline.Stage.STAGE3)
                        ?: failure?.stageDurationsMs?.get(GenUiStagePipeline.Stage.STAGE3)),
                    "stage3StreamDurationMs" to (success?.result?.stageStreamDurationsMs?.get(GenUiStagePipeline.Stage.STAGE3)
                        ?: failure?.stageStreamDurationsMs?.get(GenUiStagePipeline.Stage.STAGE3)),
                    "uiRenderCaptureMs" to uiWallMs,
                    "inputTokensReported" to success?.result?.stage3InputTokens,
                    "outputTokensReported" to (success?.result?.stage3OutputTokens ?: streamTokens),
                    "tokensPerSecondReported" to (success?.result?.stage3OutputTokensPerSecond ?: streamRate),
                    "metricsAreEstimated" to streamEstimated,
                    "nativeOutputTokens" to if (streamEstimated == false) streamTokens else null,
                    "nativeDecodeTokensPerSecond" to if (streamEstimated == false) streamRate else null,
                    "metricsScope" to "first Stage 3 generation; pipelineWallMs also includes model repair and contract rendering",
                    "pssBeforeKb" to pssBeforeKb, "pssAfterKb" to Debug.getPss(),
                    "renderSurfaceCount" to success?.result?.renderResult?.surfaces?.size,
                    "renderContractError" to success?.result?.renderResult?.errorMessage,
                    "screenCaptured" to screenCaptured, "renderErrorVisible" to renderErrorVisible,
                    "visibleTextNodes" to visibleTextNodes,
                    "warnings" to warnings,
                    "pipelineFailureStage" to failure?.stage?.name,
                    "pipelineError" to (pipelineError ?: failure?.message), "uiError" to uiError,
                    "completedAtEpochMs" to System.currentTimeMillis(),
                )
                val compact = LINE_GSON.toJson(result)
                File(caseDir, "result.json").writeText(compact + "\n")
                resultsFile.appendText(compact + "\n")
                completed++
                writeSummary(runDir, runId, selectedIds, completed, rendered, false, null, gson)
                instrumentation.sendStatus(0, Bundle().apply {
                    putString("stream", "$caseId $status ($completed/${rows.size})\n")
                })
            }
        } catch (error: Throwable) {
            runError = "${error.javaClass.simpleName}: ${error.message.orEmpty()}"
            throw error
        } finally {
            runCatching { LiteRtModelRunner.releaseCachedEngine() }
            ExperimentalFlags.enableBenchmark = oldBenchmark
            InferenceBackendSettings.setIrProvider(context, oldProvider)
            InferenceBackendSettings.setOnDeviceModelPath(context, oldModel)
            InferenceBackendSettings.setOnDeviceAccelerator(context, oldBackend)
            InferenceBackendSettings.setOnDeviceMtpEnabled(context, oldMtp)
            writeSummary(runDir, runId, selectedIds, completed, rendered,
                runError == null && completed == rows.size, runError, gson)
        }
        assertEquals("A partial benchmark is not a completed report", rows.size, completed)
    }

    /**
     * Bounded BXP-001 prompt control. `controlRequestPath` selects one app-readable JSON request
     * whose source SHA is checked against the frozen corpus. Without it, controlFewShot selects
     * the app prompt arms. This does not run SDK repair or claim pipeline acceptance.
     */
    @OptIn(ExperimentalApi::class)
    @Test
    fun boundedPromptControls() {
        data class ControlArm(
            val name: String,
            val fewShot: Boolean?,
            val prompt: String,
            val systemPrompt: String,
            val initialMessages: List<InferenceBackend.ConversationMessage>,
            val maxOutputTokens: Int,
            val provenance: String,
        )
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val args = InstrumentationRegistry.getArguments()
        val runId = requireNotNull(args.getString("runId")?.trim()) { "Pass a fresh -e runId." }
        require(SAFE_NAME.matches(runId)) { "runId must be 1..100 letters, digits, _ or -." }
        val fewShotArms = when (args.getString("controlFewShot", "false")!!.lowercase()) {
            "false" -> listOf(false)
            "true" -> listOf(true)
            "both" -> listOf(false, true)
            else -> error("controlFewShot must be false, true, or both")
        }
        val accelerator = when (args.getString("backend", "cpu")!!.lowercase()) {
            "cpu" -> InferenceBackendSettings.Accelerator.CPU
            "gpu" -> InferenceBackendSettings.Accelerator.GPU
            "auto" -> InferenceBackendSettings.Accelerator.AUTO
            else -> error("backend must be cpu, gpu, or auto")
        }
        val entry = requireNotNull(OnDeviceModelCatalog.entries.firstOrNull { it.id == MODEL_ID })
        val model = (args.getString("modelPath")?.trim()?.takeIf(String::isNotBlank)
            ?.let(::File) ?: entry.localFile(context)).canonicalFile
        require(model.isFile && model.canRead() && model.length() >= entry.minimumFileSizeBytes)
        require(OnDeviceModelCatalog.entryForModelPath(model.absolutePath)?.id == MODEL_ID) {
            "Control model must use catalog filename ${entry.fileName}."
        }
        val modelSha = sha256(model)
        args.getString("expectedModelSha256")?.trim()?.takeIf(String::isNotBlank)?.let { expected ->
            require(modelSha.equals(expected, ignoreCase = true)) { "Control model SHA-256 mismatch." }
        }
        val case = context.assets.open(CORPUS_ASSET).bufferedReader().useLines { lines ->
            lines.map { JsonParser.parseString(it).asJsonObject }
                .first { it.get("id").asString == "BXP-001" }
        }
        val query = case.get("query").asString
        val source = case.get("text").asString
        val promptAsset = args.getString(
            "controlPromptAsset", PipelinePromptBuilder.STAGE3_A2UI_EXPRESS_TRAINING_PROMPT_ASSET)!!
        require(promptAsset.startsWith("pipeline_prompts/") && !promptAsset.contains("..")) {
            "controlPromptAsset must be an app pipeline_prompts asset."
        }
        val template = PipelinePromptBuilder.loadPromptAsset(context.assets, promptAsset)
        val promptContext = PipelinePromptBuilder.prepareStage3PromptContext(
            template, trainingCompatible = true)
        val appPrompt = reconstructPrompt(context, source, promptAsset).second
        val appSystemPrompt = promptContext.systemPrompt.orEmpty()
        val controlRequestPath = args.getString("controlRequestPath")?.trim()?.takeIf(String::isNotBlank)
        val controlRequestFile = controlRequestPath?.let(::File)?.canonicalFile
        val controlRequestBytes = controlRequestFile?.let { file ->
            require(file.isAbsolute && file.isFile && file.canRead() && file.length() in 1L..131_072L) {
                "controlRequestPath must be a readable absolute JSON file of at most 128 KiB: $file"
            }
            file.readBytes()
        }
        val externalArm = controlRequestBytes?.let { bytes ->
            val request = JsonParser.parseString(bytes.toString(Charsets.UTF_8)).asJsonObject
            fun requiredString(key: String): String = request.get(key)
                ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }?.asString
                ?: error("controlRequestPath requires string $key")
            require(requiredString("caseId") == "BXP-001") { "Control request must bind BXP-001." }
            require(requiredString("sourceSha256").equals(sha256(source.toByteArray()), ignoreCase = true)) {
                "Control request sourceSha256 does not match the frozen BXP-001 response."
            }
            val label = requiredString("label")
            require(SAFE_NAME.matches(label)) { "Control request label must be a safe name." }
            val system = requiredString("systemPrompt")
            require(system.isNotBlank()) { "Control request systemPrompt is blank." }
            val userPrompt = request.get("userPrompt")?.takeUnless { it.isJsonNull }?.asString
            val responsePrefix = request.get("responsePrefix")?.takeUnless { it.isJsonNull }?.asString
            require((userPrompt == null) != (responsePrefix == null)) {
                "Control request needs exactly one of userPrompt or responsePrefix."
            }
            val renderedUser = userPrompt?.replace("{response_text}", source.trim())
                ?: (requireNotNull(responsePrefix) + source.trim())
            require(renderedUser.contains(source.trim())) {
                "Control request must contain the exact frozen BXP-001 response."
            }
            val messages = request.getAsJsonArray("initialMessages")
                ?: error("Control request needs initialMessages array (empty is allowed).")
            require(messages.size() <= 8 && messages.size() % 2 == 0) {
                "Control request initialMessages must have at most four USER/MODEL pairs."
            }
            val initial = messages.mapIndexed { index, element ->
                val turn = element.asJsonObject
                val role = when (turn.get("role")?.asString?.uppercase()) {
                    "USER" -> InferenceBackend.ConversationRole.USER
                    "MODEL", "ASSISTANT" -> InferenceBackend.ConversationRole.MODEL
                    else -> error("Unsupported control request role at index $index")
                }
                require(role == if (index % 2 == 0) InferenceBackend.ConversationRole.USER
                    else InferenceBackend.ConversationRole.MODEL) {
                    "Control request initialMessages must alternate USER and MODEL."
                }
                InferenceBackend.ConversationMessage(
                    role = role,
                    content = turn.get("content")?.asString
                        ?: error("Missing control request content at index $index"),
                )
            }
            val maxOutput = request.get("maxOutputTokens")?.takeUnless { it.isJsonNull }?.asInt
                ?: entry.maxOutputTokens
            require(maxOutput in 1..entry.maxOutputTokens) {
                "Control request maxOutputTokens must be 1..${entry.maxOutputTokens}."
            }
            ControlArm(
                name = "external_$label", fewShot = null, prompt = renderedUser,
                systemPrompt = system, initialMessages = initial, maxOutputTokens = maxOutput,
                provenance = request.get("provenance")?.takeUnless { it.isJsonNull }?.asString
                    ?: "external control request; checkpoint provenance unconfirmed",
            )
        }
        val arms = if (externalArm != null) listOf(externalArm) else fewShotArms.map { fewShot ->
            ControlArm(
                name = if (fewShot) "with_fewshot" else "no_fewshot", fewShot = fewShot,
                prompt = appPrompt, systemPrompt = appSystemPrompt,
                initialMessages = if (fewShot) promptContext.initialMessages else emptyList(),
                maxOutputTokens = entry.maxOutputTokens,
                provenance = "reference app training-compatible prompt asset",
            )
        }
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "gemma270_prompt_controls/$runId")
        require(!root.exists() && root.mkdirs()) { "Prompt-control runId exists or cannot be created: $root" }
        controlRequestBytes?.let { File(root, "control_request_input.json").writeBytes(it) }
        writeJson(File(root, "run_config.json"), linkedMapOf(
            "runId" to runId, "case" to "BXP-001", "query" to query,
            "sourceSha256" to sha256(source.toByteArray()),
            "modelId" to MODEL_ID, "modelPath" to model.absolutePath, "modelSha256" to modelSha,
            "backendRequested" to accelerator.rawValue, "mtpRequested" to false,
            "benchmarkFlagEnabled" to true,
            "promptAsset" to if (externalArm == null) promptAsset else null,
            "controlFewShot" to if (externalArm == null) args.getString("controlFewShot", "false") else null,
            "controlRequestPath" to controlRequestFile?.absolutePath,
            "controlRequestFileSha256" to controlRequestBytes?.let { sha256(it) },
            "controlRequestProvenance" to externalArm?.provenance,
            "promptConstruction" to if (externalArm == null)
                "training-compatible app system and task prefix; no Stage 3 media normalization"
                else "external request file bound to frozen BXP-001 source hash",
            "ttftAvailability" to "not exposed by OnDeviceLitertBackend",
        ), GSON)
        File(root, "source.json").writeText(GSON.toJson(case) + "\n")
        File(root, "stage3_prompt.txt").writeText(arms.first().prompt)
        File(root, "stage3_system_prompt.txt").writeText(arms.first().systemPrompt)
        val oldBenchmark = ExperimentalFlags.enableBenchmark
        val oldSpeculativeDecoding = ExperimentalFlags.enableSpeculativeDecoding
        val results = mutableListOf<Map<String, Any?>>()
        try {
            LiteRtModelRunner.releaseCachedEngine()
            ExperimentalFlags.enableBenchmark = true
            val backend = OnDeviceLitertBackend(
                modelPath = model.absolutePath, acceleratorPreference = accelerator, mtpEnabled = false)
            for (arm in arms) {
                val armName = arm.name
                val armDir = File(root, armName).apply { check(mkdirs()) }
                writeJson(File(armDir, "initial_messages.json"), arm.initialMessages.map {
                    mapOf("role" to it.role.name, "content" to it.content)
                }, GSON)
                val requestJson = GSON.toJson(linkedMapOf(
                    "caseId" to "BXP-001", "sourceSha256" to sha256(source.toByteArray()),
                    "modelSha256" to modelSha, "backendRequested" to accelerator.rawValue,
                    "systemPrompt" to arm.systemPrompt,
                    "userPrompt" to arm.prompt,
                    "initialMessages" to arm.initialMessages.map {
                        mapOf("role" to it.role.name, "content" to it.content)
                    },
                    "temperature" to 0.0, "maxOutputTokens" to arm.maxOutputTokens,
                    "sendSystemPrompt" to true, "jsonMode" to false,
                    "structuredOutput" to false,
                )) + "\n"
                File(armDir, "request.json").writeText(requestJson)
                val requestSha = sha256(requestJson.toByteArray())
                var metricsAreEstimated: Boolean? = null
                var completedStreamText: String? = null
                val startedNs = SystemClock.elapsedRealtimeNanos()
                val response = backend.generate(InferenceBackend.GenerateRequest(
                    prompt = arm.prompt, systemPrompt = arm.systemPrompt,
                    initialMessages = arm.initialMessages, temperature = 0.0,
                    maxOutputTokens = arm.maxOutputTokens, jsonMode = false,
                    structuredOutput = false,
                    onStreamUpdate = { update ->
                        if (update.complete) {
                            metricsAreEstimated = update.metricsAreEstimated
                            completedStreamText = update.text
                        }
                    },
                ))
                val wallMs = elapsedMs(startedNs)
                val raw = completedStreamText ?: response.text
                File(armDir, "raw.express").writeText(raw)
                val expressParseOk = runCatching { A2uiExpressCodec.decode(raw) }.isSuccess
                val result = linkedMapOf<String, Any?>(
                    "arm" to armName, "fewShot" to arm.fewShot,
                    "provenance" to arm.provenance,
                    "requestSha256" to requestSha,
                    "requestFile" to "$armName/request.json",
                    "status" to if (response.error == null) "generated" else "runtime_error",
                    "error" to response.error, "runtimeBackend" to response.runtimeBackend,
                    "rawSha256" to sha256(raw.toByteArray()), "rawChars" to raw.length,
                    "expressParseOk" to expressParseOk,
                    "wallMs" to wallMs, "backendStreamDurationMs" to response.streamDurationMs,
                    "inputTokensReported" to response.inputTokens,
                    "outputTokensReported" to response.outputTokens,
                    "tokensPerSecondReported" to response.outputTokensPerSecond,
                    "metricsAreEstimated" to metricsAreEstimated,
                    "nativeOutputTokens" to if (metricsAreEstimated == false) response.outputTokens else null,
                    "nativeDecodeTokensPerSecond" to
                        if (metricsAreEstimated == false) response.outputTokensPerSecond else null,
                    "timeToFirstTokenMs" to null,
                    "timeToFirstTokenStatus" to "not exposed by OnDeviceLitertBackend",
                )
                writeJson(File(armDir, "result.json"), result, GSON)
                results += result
                writeJson(File(root, "results.json"), results, GSON)
                instrumentation.sendStatus(0, Bundle().apply {
                    putString("stream", "$armName ${result["status"]} ($wallMs ms)\n")
                })
            }
            // Capture native package-template rendering after all timed generations. This CPU
            // serialization probe is separate from the measured backend and makes no model call.
            LiteRtModelRunner.releaseCachedEngine()
            ExperimentalFlags.enableSpeculativeDecoding = false
            var renderEngine: Engine? = null
            try {
                val engine = Engine(EngineConfig(
                    modelPath = model.absolutePath,
                    backend = Backend.CPU(Runtime.getRuntime().availableProcessors().coerceIn(2, 6)),
                    maxNumTokens = entry.maxContextTokens,
                    cacheDir = null,
                ))
                renderEngine = engine
                engine.initialize()
                for (arm in arms) {
                    val armDir = File(root, arm.name)
                    val capture = runCatching {
                        engine.createConversation(ConversationConfig(
                            systemInstruction = arm.systemPrompt.trim().takeIf(String::isNotBlank)
                                ?.let(Contents::of),
                            initialMessages = arm.initialMessages.map { message ->
                                when (message.role) {
                                    InferenceBackend.ConversationRole.USER -> Message.user(message.content)
                                    InferenceBackend.ConversationRole.MODEL -> Message.model(message.content)
                                }
                            },
                            samplerConfig = SamplerConfig(temperature = 0.0, topK = 1, topP = 1.0),
                            maxOutputToken = arm.maxOutputTokens,
                        )).use { conversation ->
                            conversation.renderMessageIntoString(
                                Message.user(arm.prompt.trim()), emptyMap<String, Any>())
                        }
                    }
                    capture.fold(
                        onSuccess = { rendered ->
                            File(armDir, "native_prompt.txt").writeText(rendered)
                            writeJson(File(armDir, "native_prompt_meta.json"), mapOf(
                                "status" to "captured", "serializationBackend" to "CPU",
                                "captureAfterTimedGeneration" to true,
                                "sha256" to sha256(rendered.toByteArray()),
                                "containsFrozenSource" to rendered.contains(source.trim()),
                                "characters" to rendered.length,
                            ), GSON)
                        },
                        onFailure = { failure ->
                            writeJson(File(armDir, "native_prompt_meta.json"), mapOf(
                                "status" to "error", "serializationBackend" to "CPU",
                                "captureAfterTimedGeneration" to true,
                                "error" to (failure.message ?: failure.javaClass.simpleName),
                            ), GSON)
                        },
                    )
                }
            } catch (failure: Throwable) {
                writeJson(File(root, "native_prompt_probe_error.json"), mapOf(
                    "error" to (failure.message ?: failure.javaClass.simpleName)), GSON)
            } finally {
                runCatching { renderEngine?.close() }
            }
        } finally {
            runCatching { LiteRtModelRunner.releaseCachedEngine() }
            ExperimentalFlags.enableBenchmark = oldBenchmark
            ExperimentalFlags.enableSpeculativeDecoding = oldSpeculativeDecoding
        }
    }

    private fun reconstructPrompt(context: android.content.Context, normalizedSource: String,
                                  asset: String = PipelinePromptBuilder.STAGE3_A2UI_EXPRESS_TRAINING_PROMPT_ASSET): Pair<String, String> {
        val template = PipelinePromptBuilder.loadPromptAsset(
            context.assets, asset)
        val prompt = PipelinePromptBuilder.prepareStage3PromptContext(template, trainingCompatible = true)
        val shortenedResponse = McpUrlShortener.shorten(normalizedSource).shortenedText
        val user = PipelinePromptBuilder.buildStage3UserPrompt(
            userTemplate = prompt.userTemplate, stage2Response = shortenedResponse,
            catalogId = "", assets = emptyList(), appendRequestPolicies = false)
        return prompt.systemPrompt.orEmpty() to McpUrlShortener.shorten(user).shortenedText
    }

    private fun probeSdkGeneratedDslRepair(raw: String, source: String, caseDir: File): SdkRepair {
        if (raw.isBlank()) {
            File(caseDir, "sdk_repair.json").writeText("{\"status\":\"not_attempted\",\"sourceFallbackUsed\":false}\n")
            return SdkRepair("not_attempted", null)
        }
        return runCatching {
            GenUiCompiler.compileWithRepair(
                input = raw, sourceText = source,
                allowSourceTextFallback = false, allowGeneratedDslRepair = true)
        }.fold(
            onSuccess = { accepted ->
                File(caseDir, "sdk_repair.express").writeText(accepted.document.express)
                File(caseDir, "sdk_repair.a2ui.json").writeText(accepted.document.a2uiJson)
                File(caseDir, "sdk_repair.json").writeText(GSON.toJson(mapOf(
                    "status" to "accepted", "repairKind" to accepted.repairKind.name,
                    "sourceFallbackUsed" to false, "diagnostics" to accepted.diagnostics)) + "\n")
                SdkRepair("accepted", accepted.repairKind.name)
            },
            onFailure = { rejected ->
                File(caseDir, "sdk_repair.json").writeText(GSON.toJson(mapOf(
                    "status" to "rejected", "repairKind" to null, "sourceFallbackUsed" to false,
                    "error" to (rejected.message ?: rejected.javaClass.simpleName))) + "\n")
                SdkRepair("rejected", null)
            },
        )
    }

    private fun writeSummary(dir: File, runId: String, cases: List<String>, completed: Int,
                             rendered: Int, runComplete: Boolean, error: String?, gson: com.google.gson.Gson) {
        writeJson(File(dir, "summary.json"), linkedMapOf(
            "runId" to runId, "total" to cases.size, "completed" to completed,
            "rendered" to rendered, "notRendered" to completed - rendered,
            "runComplete" to runComplete, "error" to error,
            "updatedAtEpochMs" to System.currentTimeMillis(),
        ), gson)
    }

    private fun writeJson(file: File, value: Any?, gson: com.google.gson.Gson) {
        file.writeText(gson.toJson(value) + "\n")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val bytes = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(bytes)
                if (count < 0) break
                digest.update(bytes, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun elapsedMs(startedNs: Long): Long =
        (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000L

    private data class StreamCompletion(
        val text: String, val backend: String?, val outputTokens: Int?,
        val tokensPerSecond: Double?, val estimated: Boolean, val completedAtEpochMs: Long,
    )

    private data class SdkRepair(val status: String, val kind: String?)

    private companion object {
        const val MODEL_ID = "gemma3_270m_a2ui_express_int8"
        const val CORPUS_ASSET = "genuicraft_bixby50.jsonl"
        const val CORPUS_SHA256 = "fc46aa381957bed206f9e0f53e28edbb21b5096ca093985ad184fda73faaea0a"
        val SAFE_NAME = Regex("[A-Za-z0-9_-]{1,100}")
        val GSON = GsonBuilder().serializeNulls().disableHtmlEscaping().setPrettyPrinting().create()
        val LINE_GSON = GsonBuilder().serializeNulls().disableHtmlEscaping().create()
    }
}
