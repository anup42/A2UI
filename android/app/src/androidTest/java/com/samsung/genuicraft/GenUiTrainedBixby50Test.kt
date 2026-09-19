package com.samsung.genuicraft

import android.os.Build
import android.os.Bundle
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonNull
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiConversionResult
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiPromptMessage
import com.samsung.genuicraft.sdk.GenUiPromptRole
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiRequest
import com.samsung.genuicraft.sdk.GenUiTrainedConverter
import com.samsung.genuicraft.sdk.provider.Gemma4Config
import com.samsung.genuicraft.sdk.provider.Gemma4Provider
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** End-to-end device benchmark for the trained E2B v10 W4 checkpoint and frozen scaffold. */
@RunWith(AndroidJUnit4::class)
class GenUiTrainedBixby50Test {
    @Test
    fun convertAndRenderTrainedBixby50() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        val gson = GsonBuilder().setPrettyPrinting().serializeNulls().disableHtmlEscaping().create()
        val externalFiles = requireNotNull(context.getExternalFilesDir(null))
        val runId = arguments.getString("runId")?.trim().takeUnless { it.isNullOrEmpty() }
            ?: "trained_e2b_v10_w4_${System.currentTimeMillis()}"
        require(RUN_ID.matches(runId)) {
            "runId must contain only letters, digits, underscores, or hyphens (1..100 characters)."
        }
        val runDir = File(externalFiles, "sdk_benchmark/$runId")
        require(!runDir.exists() && runDir.mkdirs()) {
            "Benchmark output must be a new directory: ${runDir.absolutePath}"
        }

        val corpusBytes = context.assets.open(CORPUS_ASSET).use { it.readBytes() }
        val corpus = parseCorpus(corpusBytes)
        require(corpus.size == 50) { "Expected 50 Bixby benchmark rows, found ${corpus.size}." }
        val selectedIds = arguments.getString("cases", "").orEmpty().split(',')
            .map(String::trim).filter(String::isNotEmpty)
        require(selectedIds.size == selectedIds.distinct().size) { "cases contains duplicate IDs." }
        val byId = corpus.associateBy(CorpusCase::id)
        require(selectedIds.all(byId::containsKey)) { "cases contains an unknown Bixby case ID." }
        val cases = if (selectedIds.isEmpty()) corpus else selectedIds.map(byId::getValue)
        val caseTimeoutMs = arguments.getString("caseTimeoutMs", "600000")!!.toLong().also {
            require(it in 1_000L..3_600_000L) { "caseTimeoutMs must be between 1000 and 3600000." }
        }

        val defaultModel = File(externalFiles, "sdk_models/e2b_v10_w4.litertlm")
        val modelFile = File(
            arguments.getString("modelPath")?.trim().takeUnless { it.isNullOrEmpty() }
                ?: defaultModel.absolutePath,
        ).canonicalFile
        require(modelFile.isFile && modelFile.canRead() && modelFile.length() > 0L) {
            "Trained E2B model is missing, empty, or unreadable: ${modelFile.absolutePath}"
        }

        val promptBytes = context.assets.open(GenUiTrainedConverter.PROMPT_ASSET).use { it.readBytes() }
        val promptContract = PromptContract.parse(promptBytes)
        require(promptContract.contractSha256 == PINNED_CONTRACT_SHA256) {
            "Unexpected trained prompt contract: ${promptContract.contractSha256}"
        }
        File(runDir, "shared_prompt.json").writeBytes(promptBytes)
        writeJson(
            File(runDir, "run_config.json"),
            linkedMapOf(
                "runId" to runId,
                "profile" to GenUiTrainedConverter.PROFILE,
                "startedAtEpochMs" to System.currentTimeMillis(),
                "cases" to cases.map(CorpusCase::id),
                "corpus" to mapOf(
                    "asset" to CORPUS_ASSET,
                    "sha256" to sha256(corpusBytes),
                    "totalRows" to corpus.size,
                    "selectedRows" to cases.size,
                ),
                "prompt" to mapOf(
                    "asset" to GenUiTrainedConverter.PROMPT_ASSET,
                    "assetSha256" to sha256(promptBytes),
                    "contractSha256" to promptContract.contractSha256,
                    "systemPromptSha256" to promptContract.systemPromptSha256,
                ),
                "model" to mapOf(
                    "basename" to modelFile.name,
                    "sizeBytes" to modelFile.length(),
                ),
                "runtime" to mapOf(
                    "provider" to "gemma4_e2b",
                    "accelerator" to "GPU",
                    "maxContextTokens" to 8_192,
                    "maxOutputTokens" to 2_048,
                    "thinkingEnabled" to false,
                    "thinkingTokenBudget" to 0,
                    "mtpEnabled" to false,
                    "metricsEnabled" to true,
                    "temperature" to 0.0,
                    "repairAttempts" to 0,
                    "fallbackEnabled" to false,
                    "warmSharedProvider" to true,
                    "caseTimeoutMs" to caseTimeoutMs,
                ),
                "device" to mapOf(
                    "manufacturer" to Build.MANUFACTURER,
                    "brand" to Build.BRAND,
                    "model" to Build.MODEL,
                    "device" to Build.DEVICE,
                    "product" to Build.PRODUCT,
                    "hardware" to Build.HARDWARE,
                    "sdkInt" to Build.VERSION.SDK_INT,
                    "release" to Build.VERSION.RELEASE,
                    "supportedAbis" to Build.SUPPORTED_ABIS.toList(),
                    "fingerprint" to Build.FINGERPRINT,
                ),
            ),
            gson,
        )

        val provider = Gemma4Provider(
            Gemma4Config(
                modelPath = modelFile.absolutePath,
                accelerator = "GPU",
                maxContextTokens = 8_192,
                maxOutputTokens = 2_048,
                enableThinking = false,
                thinkingTokenBudget = 0,
                enableSpeculativeDecoding = false,
                enableMetrics = true,
            ),
        )
        val device = UiDevice.getInstance(instrumentation)
        val reports = mutableListOf<JsonObject>()
        var providerCloseError: String? = null
        writeRunStatus(runDir, runId, "running", 0, cases.size, gson)

        try {
            ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
                cases.forEachIndexed { index, benchmarkCase ->
                    val caseDir = File(runDir, benchmarkCase.id).apply {
                        check(mkdirs()) { "Unable to create case directory: $absolutePath" }
                    }
                    writeJson(File(caseDir, "source.json"), benchmarkCase.asMap(), gson)
                    writeJson(
                        File(caseDir, "status.json"),
                        mapOf("id" to benchmarkCase.id, "status" to "running", "index" to index + 1),
                        gson,
                    )
                    val recorder = CaseRecordingProvider(
                        delegate = provider,
                        caseDir = caseDir,
                        expectedResponse = benchmarkCase.text,
                        contract = promptContract,
                        gson = gson,
                    )
                    val report = JsonObject().apply {
                        addProperty("id", benchmarkCase.id)
                        addProperty("status", "running")
                        addProperty("strictValid", false)
                        addProperty("renderValid", false)
                        addProperty("elapsedMs", 0L)
                        addProperty("firstInRun", index == 0)
                        addProperty("attempts", 0)
                        add("metrics", JsonNull.INSTANCE)
                    }
                    val caseStarted = System.nanoTime()
                    try {
                        val conversion = try {
                            withTimeout(caseTimeoutMs) {
                                GenUiTrainedConverter(context, recorder).convert(
                                    GenUiRequest(benchmarkCase.text, benchmarkCase.query),
                                )
                            }
                        } catch (timeout: TimeoutCancellationException) {
                            GenUiConversionResult.Failure(
                                message = "Timed out after $caseTimeoutMs ms.",
                                provider = provider.id,
                                elapsedMs = elapsedMs(caseStarted),
                                attempts = recorder.callCount,
                                rawOutput = recorder.output?.text,
                            )
                        }
                        report.addProperty("provider", provider.id)
                        report.addProperty("providerCalls", recorder.callCount)
                        report.add("metrics", gson.toJsonTree(recorder.metricMap()))
                        recorder.output?.runtime?.let { report.addProperty("runtime", it) }
                        recorder.output?.renderedPromptSha256?.let { report.addProperty("renderedPromptSha256", it) }
                        when (conversion) {
                            is GenUiConversionResult.Failure -> {
                                val failureStatus = when {
                                    conversion.message.startsWith("Timed out") -> "timeout"
                                    recorder.output == null -> "runtime_error"
                                    else -> "strict_invalid"
                                }
                                report.addProperty(
                                    "status",
                                    failureStatus,
                                )
                                report.addProperty("elapsedMs", conversion.elapsedMs)
                                report.addProperty("attempts", conversion.attempts)
                                report.addProperty("error", conversion.message)
                                report.addProperty("rawOutputCaptured", recorder.output != null)
                            }
                            is GenUiConversionResult.Success -> {
                                report.addProperty("strictValid", true)
                                report.addProperty("elapsedMs", conversion.elapsedMs)
                                report.addProperty("attempts", conversion.attempts)
                                report.add("warnings", gson.toJsonTree(conversion.warnings))
                                val compiledFile = File(caseDir, "a2ui.json")
                                compiledFile.writeText(conversion.document.a2uiJson)
                                val replay = com.samsung.genuicraft.sdk.GenUiCompiler.compile(
                                    compiledFile.readText(),
                                )
                                val rendererProbe = GenUiNativeRenderer.render(replay.a2uiJson, null)
                                val rendererContractValid = rendererProbe.errorMessage.isNullOrBlank() &&
                                    rendererProbe.surfaces.isNotEmpty()
                                report.addProperty("rendererContractValid", rendererContractValid)
                                report.add("rendererWarnings", gson.toJsonTree(rendererProbe.warnings))
                                if (rendererContractValid) {
                                    val capture = renderAndCapture(
                                        instrumentation = instrumentation,
                                        scenario = scenario,
                                        device = device,
                                        caseId = benchmarkCase.id,
                                        document = replay,
                                        output = recorder.output?.let(::listOf).orEmpty(),
                                        elapsedMs = conversion.elapsedMs,
                                        caseDir = caseDir,
                                    )
                                    report.addProperty("screenCaptured", capture.screenCaptured)
                                    report.addProperty("scrolledScreenCaptured", capture.scrolledScreenCaptured)
                                    report.addProperty("currentCaseVisible", capture.currentCaseVisible)
                                    report.addProperty("rendererErrorVisible", capture.rendererErrorVisible)
                                    val renderValid = capture.currentCaseVisible &&
                                        !capture.rendererErrorVisible && capture.screenCaptured &&
                                        capture.scrolledScreenCaptured
                                    report.addProperty("renderValid", renderValid)
                                    report.addProperty("status", if (renderValid) "valid" else "render_invalid")
                                } else {
                                    report.addProperty("status", "render_invalid")
                                    report.addProperty(
                                        "error",
                                        rendererProbe.errorMessage ?: "Renderer produced no surfaces.",
                                    )
                                }
                            }
                        }
                    } catch (cancelled: CancellationException) {
                        report.addProperty("status", "cancelled")
                        report.addProperty("error", cancelled.message ?: "Benchmark cancelled.")
                        report.addProperty("providerCalls", recorder.callCount)
                        report.add("metrics", gson.toJsonTree(recorder.metricMap()))
                        report.addProperty("caseElapsedMs", elapsedMs(caseStarted))
                        finishCase(runDir, caseDir, report, reports, runId, cases.size, gson)
                        throw cancelled
                    } catch (failure: Exception) {
                        report.addProperty("status", "case_error")
                        report.addProperty("error", failure.message ?: failure.javaClass.simpleName)
                        report.addProperty("providerCalls", recorder.callCount)
                        report.add("metrics", gson.toJsonTree(recorder.metricMap()))
                    }
                    report.addProperty("caseElapsedMs", elapsedMs(caseStarted))
                    finishCase(runDir, caseDir, report, reports, runId, cases.size, gson)
                    instrumentation.sendStatus(0, Bundle().apply {
                        putString(
                            "stream",
                            "${benchmarkCase.id} ${report.get("status").asString} " +
                                "(${index + 1}/${cases.size})\n",
                        )
                    })
                }
            }
        } finally {
            withContext(NonCancellable) {
                try {
                    provider.closeAndAwait()
                } catch (failure: Exception) {
                    providerCloseError = failure.message ?: failure.javaClass.simpleName
                }
            }
        }

        val runComplete = reports.size == cases.size
        val strictValid = reports.count { it.get("strictValid")?.asBoolean == true }
        val renderValid = reports.count { it.get("renderValid")?.asBoolean == true }
        val valid = reports.count { it.get("status")?.asString == "valid" }
        val allValid = runComplete && valid == cases.size && providerCloseError == null
        val summary = linkedMapOf(
            "runId" to runId,
            "profile" to GenUiTrainedConverter.PROFILE,
            "total" to cases.size,
            "completed" to reports.size,
            "strictValid" to strictValid,
            "strictInvalid" to reports.count { it.get("status")?.asString == "strict_invalid" },
            "renderValid" to renderValid,
            "renderInvalid" to reports.count { it.get("strictValid")?.asBoolean == true && it.get("renderValid")?.asBoolean != true },
            "renderNotAttempted" to (reports.size - strictValid),
            "valid" to valid,
            "invalid" to (reports.size - valid),
            "timeouts" to reports.count { it.get("status")?.asString == "timeout" },
            "runtimeErrors" to reports.count { it.get("status")?.asString == "runtime_error" },
            "caseErrors" to reports.count { it.get("status")?.asString == "case_error" },
            "providerCalls" to reports.sumOf { it.get("providerCalls")?.asInt ?: 0 },
            "providerCloseError" to providerCloseError,
            "runComplete" to runComplete,
            "allValid" to allValid,
            "finishedAtEpochMs" to System.currentTimeMillis(),
        )
        writeJson(File(runDir, "summary.json"), summary, gson)
        writeRunStatus(runDir, runId, "complete", reports.size, cases.size, gson)
        assertTrue(
            "runComplete=$runComplete, allValid=$allValid; artifacts: ${runDir.absolutePath}",
            runComplete && allValid,
        )
    }

    private fun finishCase(
        runDir: File,
        caseDir: File,
        report: JsonObject,
        reports: MutableList<JsonObject>,
        runId: String,
        total: Int,
        gson: Gson,
    ) {
        File(caseDir, "result.json").writeText(gson.toJson(report) + "\n")
        writeJson(
            File(caseDir, "status.json"),
            mapOf("id" to report.get("id").asString, "status" to report.get("status").asString),
            gson,
        )
        reports += report
        File(runDir, "results.json").writeText(gson.toJson(reports) + "\n")
        writeRunStatus(runDir, runId, "running", reports.size, total, gson)
    }

    private suspend fun renderAndCapture(
        instrumentation: android.app.Instrumentation,
        scenario: ActivityScenario<GenUiSdkDemoActivity>,
        device: UiDevice,
        caseId: String,
        document: com.samsung.genuicraft.sdk.GenUiDocument,
        output: List<GenUiModelOutput>,
        elapsedMs: Long,
        caseDir: File,
    ): RenderCapture {
        scenario.onActivity { activity ->
            activity.showDocument(document, "$caseId · trained E2B v10 W4 · $elapsedMs ms")
            activity.showGenerationMetrics(output, elapsedMs)
        }
        instrumentation.waitForIdleSync()
        val currentCaseVisible = device.wait(Until.hasObject(By.textContains(caseId)), 10_000)
        if (!currentCaseVisible) return RenderCapture(false, false, false, false)
        delay(700)
        val rendererErrorVisible = device.hasObject(By.textContains("Unable to render GenUI"))
        val screenCaptured = device.takeScreenshot(File(caseDir, "screen.png"))
        device.swipe(
            device.displayWidth / 2,
            device.displayHeight * 85 / 100,
            device.displayWidth / 2,
            device.displayHeight * 35 / 100,
            35,
        )
        delay(350)
        val scrolledScreenCaptured = device.takeScreenshot(File(caseDir, "screen_scrolled.png"))
        return RenderCapture(
            currentCaseVisible,
            rendererErrorVisible,
            screenCaptured,
            scrolledScreenCaptured,
        )
    }

    private class CaseRecordingProvider(
        private val delegate: GenUiProvider,
        private val caseDir: File,
        private val expectedResponse: String,
        private val contract: PromptContract,
        private val gson: Gson,
    ) : GenUiProvider {
        override val id: String
            get() = delegate.id

        var callCount: Int = 0
            private set
        var output: GenUiModelOutput? = null
            private set
        private var providerCallMs: Long? = null
        private var providerError: String? = null

        override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
            callCount++
            writeJson(
                File(caseDir, "prompt.json"),
                linkedMapOf(
                    "system" to prompt.system,
                    "user" to prompt.user,
                    "initialMessages" to prompt.initialMessages.map {
                        mapOf("role" to it.role.name, "text" to it.text)
                    },
                    "maxOutputTokens" to prompt.maxOutputTokens,
                    "temperature" to prompt.temperature,
                    "chatTemplateOverride" to prompt.chatTemplateOverride,
                    "expectedRenderedPromptSha256" to prompt.expectedRenderedPrompt?.let {
                        sha256(it.toByteArray(Charsets.UTF_8))
                    },
                ),
                gson,
            )
            check(callCount == 1) { "Trained converter made more than one provider call." }
            check(prompt.system == contract.system) { "Runtime system prompt differs from pinned contract." }
            check(prompt.user == contract.taskPrefix + expectedResponse.trim()) {
                "Runtime user prompt differs from pinned raw-response contract."
            }
            check(prompt.initialMessages == contract.initialMessages) {
                "Runtime few-shot messages differ from pinned contract."
            }
            check(prompt.maxOutputTokens == 2_048 && prompt.temperature == 0.0) {
                "Runtime generation parameters differ from the trained profile."
            }
            val started = System.nanoTime()
            try {
                return delegate.generate(prompt).also { value ->
                    output = value
                    File(caseDir, "output.express").writeText(value.text)
                }
            } catch (failure: Throwable) {
                providerError = failure.message ?: failure.javaClass.simpleName
                throw failure
            } finally {
                providerCallMs = elapsedMs(started)
                writeJson(
                    File(caseDir, "metrics.json"),
                    linkedMapOf(
                        "providerCallMs" to providerCallMs,
                        "runtime" to output?.runtime,
                        "outputCharacters" to output?.text?.length,
                        "outputTokens" to output?.outputTokens,
                        "inputTokens" to output?.metrics?.inputTokens,
                        "actualOutputTokens" to output?.metrics?.outputTokens,
                        "decodeTokensPerSecond" to output?.metrics?.decodeTokensPerSecond,
                        "error" to providerError,
                    ),
                    gson,
                )
            }
        }

        fun metricMap(): Map<String, Any?> = linkedMapOf(
            "inputTokens" to output?.metrics?.inputTokens,
            "outputTokens" to (output?.metrics?.outputTokens ?: output?.outputTokens),
            "decodeTokensPerSecond" to output?.metrics?.decodeTokensPerSecond,
            "providerCallMs" to providerCallMs,
        )
    }

    private data class CorpusCase(
        val id: String,
        val query: String,
        val text: String,
        val domain: String,
    ) {
        fun asMap(): Map<String, String> = linkedMapOf(
            "id" to id,
            "query" to query,
            "text" to text,
            "domain" to domain,
        )
    }

    private data class PromptContract(
        val system: String,
        val initialMessages: List<GenUiPromptMessage>,
        val taskPrefix: String,
        val contractSha256: String,
        val systemPromptSha256: String,
    ) {
        companion object {
            fun parse(bytes: ByteArray): PromptContract {
                val root = JsonParser.parseString(bytes.toString(Charsets.UTF_8)).asJsonObject
                val messages = root.getAsJsonObject("scaffold").getAsJsonArray("messages")
                    .map { it.asJsonObject }
                require(messages.map { it.get("role").asString } ==
                    listOf("system", "user", "assistant")) {
                    "Unexpected trained prompt message roles."
                }
                val system = messages[0].get("content").asString
                val systemHash = root.get("system_prompt_sha256").asString
                require(sha256(system.toByteArray(Charsets.UTF_8)) == systemHash) {
                    "Trained system prompt checksum mismatch."
                }
                return PromptContract(
                    system = system,
                    initialMessages = listOf(
                        GenUiPromptMessage(GenUiPromptRole.USER, messages[1].get("content").asString),
                        GenUiPromptMessage(GenUiPromptRole.MODEL, messages[2].get("content").asString),
                    ),
                    taskPrefix = root.getAsJsonObject("scaffold").get("task_prefix").asString,
                    contractSha256 = root.get("contract_sha256").asString,
                    systemPromptSha256 = systemHash,
                )
            }
        }
    }

    private data class RenderCapture(
        val currentCaseVisible: Boolean,
        val rendererErrorVisible: Boolean,
        val screenCaptured: Boolean,
        val scrolledScreenCaptured: Boolean,
    )

    companion object {
        private const val CORPUS_ASSET = "genuicraft_bixby50.jsonl"
        private const val PINNED_CONTRACT_SHA256 =
            "005ef700eee29a4429c895ac6c86580003c43d0ad35e67cbb92a9d4cf0e8c1d2"
        private val RUN_ID = Regex("[A-Za-z0-9_-]{1,100}")

        private fun parseCorpus(bytes: ByteArray): List<CorpusCase> {
            val rows = bytes.toString(Charsets.UTF_8).lineSequence().filter(String::isNotBlank)
                .map { line ->
                    val row = JsonParser.parseString(line).asJsonObject
                    CorpusCase(
                        id = row.requiredString("id"),
                        query = row.requiredString("query"),
                        text = row.requiredString("text"),
                        domain = row.requiredString("domain"),
                    )
                }.toList()
            require(rows.map(CorpusCase::id).distinct().size == rows.size) {
                "Bixby corpus contains duplicate IDs."
            }
            require(rows.all { RUN_ID.matches(it.id) && it.text.isNotBlank() }) {
                "Bixby corpus contains an unsafe ID or blank response."
            }
            return rows
        }

        private fun JsonObject.requiredString(name: String): String =
            get(name)?.takeUnless { it.isJsonNull }?.asString
                ?: error("Bixby corpus row is missing '$name'.")

        private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
            .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }

        private fun elapsedMs(startedNanos: Long): Long =
            ((System.nanoTime() - startedNanos) / 1_000_000L).coerceAtLeast(0L)

        private fun writeJson(file: File, value: Any?, gson: Gson) {
            file.writeText(gson.toJson(value) + "\n")
        }

        private fun writeRunStatus(
            runDir: File,
            runId: String,
            state: String,
            completed: Int,
            total: Int,
            gson: Gson,
        ) {
            writeJson(
                File(runDir, "status.json"),
                mapOf(
                    "runId" to runId,
                    "state" to state,
                    "completed" to completed,
                    "total" to total,
                    "updatedAtEpochMs" to System.currentTimeMillis(),
                ),
                gson,
            )
        }
    }
}
