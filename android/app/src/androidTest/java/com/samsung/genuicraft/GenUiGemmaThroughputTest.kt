package com.samsung.genuicraft

import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Capabilities
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.security.MessageDigest
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Replays captured first-attempt SDK prompts solely to measure LiteRT-LM's native GPU+MTP
 * throughput. This does not run GenUICraft validation or rendering and is not a replacement for
 * the Bixby50 AAR acceptance result that supplied the prompts.
 *
 * Instrumentation arguments:
 * - modelPath: required absolute path to the same MTP-capable `.litertlm` package.
 * - sourceRunId: captured run under `sdk_benchmark`; defaults to ps_final50_scaffold_v11.
 * - cases: ordered comma-separated case IDs; defaults to the three screening cases below.
 * - runId: new output directory name under `sdk_throughput`.
 */
@RunWith(AndroidJUnit4::class)
class GenUiGemmaThroughputTest {

    @OptIn(ExperimentalApi::class)
    @Test
    fun measuresCapturedSdkPromptThroughput() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        val sourceRunId = safeName(
            arguments.getString("sourceRunId")?.trim().orEmpty().ifBlank { DEFAULT_SOURCE_RUN_ID },
            "sourceRunId",
        )
        val runId = safeName(
            arguments.getString("runId")?.trim().orEmpty().ifBlank {
                "gemma_throughput_${System.currentTimeMillis()}"
            },
            "runId",
        )
        val cases = parseCases(arguments.getString("cases"))
        val modelPathArgument = arguments.getString("modelPath")?.trim().orEmpty()

        val externalRoot = requireNotNull(context.getExternalFilesDir(null)) {
            "The target app has no external files directory."
        }
        val sourceRun = File(externalRoot, "sdk_benchmark/$sourceRunId")
        val output = File(externalRoot, "sdk_throughput/$runId")
        require(!output.exists() && output.mkdirs()) {
            "Throughput output must be a new directory: ${output.absolutePath}"
        }

        val startedAtEpochMs = System.currentTimeMillis()
        val runStartedAtNs = SystemClock.elapsedRealtimeNanos()
        val results = JSONArray()
        val runConfig = JSONObject()
            .put("schemaVersion", 1)
            .put("kind", "captured_sdk_prompt_native_throughput")
            .put("runId", runId)
            .put("sourceRunId", sourceRunId)
            .put("cases", JSONArray(cases))
            .put("modelPathArgument", modelPathArgument)
            .put("outputDirectory", output.absolutePath)
            .put("startedAtEpochMs", startedAtEpochMs)
            .put("acceptanceEvaluated", false)
            .put("rendererEvaluated", false)
            .put("uncountedWarmupCases", 0)
            .put("engineInitializationTargetCount", 1)
            .put("engineReuse", "one engine; a fresh conversation for every recorded case")
            .put("backend", "GPU")
            .put("maxContextTokens", MAX_CONTEXT_TOKENS)
            .put("maxOutputTokens", MAX_OUTPUT_TOKENS)
            .put("samplerTemperature", 0.0)
            .put("samplerTopK", 1)
            .put("samplerTopP", 1.0)
            .put("samplerSeed", "LiteRT-LM default; not set by this test or the SDK runtime")
            .put("thinkingEnabled", true)
            .put("thinkingTokenBudget", THINKING_TOKEN_BUDGET)
            .put("mtpRequired", true)
            .put("benchmarkRequired", true)
            .put("cacheDir", JSONObject.NULL)
            .put(
                "productionRuntimeComparison",
                JSONObject()
                    .put(
                        "benchmarkFlag",
                        "intentional difference: enabled before engine initialization for native metrics",
                    )
                    .put(
                        "cacheDir",
                        "match: omitted/null, the Gemma4Config production default",
                    )
                    .put(
                        "generation",
                        "match: GPU+MTP, 16384 context, 8192 output, deterministic sampler, thinking 1024",
                    ),
            )
        writeJson(File(output, "run_config.json"), runConfig)
        writeJson(File(output, "results.json"), results)
        writeJson(
            File(output, "run_state.json"),
            JSONObject()
                .put("status", "preparing")
                .put("completedCases", 0)
                .put("totalCases", cases.size),
        )

        var engine: Engine? = null
        var engineInitialized = false
        var engineClosed = false
        var primaryFailure: Throwable? = null
        var closeFailure: Throwable? = null
        var completedCases = 0
        var failedCases = 0
        var systemPromptSha256: String? = null

        try {
            require(modelPathArgument.isNotBlank()) {
                "Pass -e modelPath /absolute/device/path/model.litertlm"
            }
            val modelFile = File(modelPathArgument)
            require(modelFile.isAbsolute) { "modelPath must be absolute: $modelPathArgument" }
            require(modelFile.isFile && modelFile.canRead() && modelFile.length() > 0L) {
                "modelPath must be a readable, non-empty file: $modelPathArgument"
            }
            require(modelFile.name.lowercase(Locale.US).endsWith(".litertlm")) {
                "modelPath must point to a .litertlm package: $modelPathArgument"
            }
            val modelPath = modelFile.canonicalPath
            require(sourceRun.isDirectory) { "Missing captured source run: ${sourceRun.absolutePath}" }

            val systemPromptBytes = context.assets.open(SYSTEM_PROMPT_ASSET).use { it.readBytes() }
            require(systemPromptBytes.isNotEmpty()) { "Packaged Gemma system prompt is empty." }
            val systemPrompt = systemPromptBytes.toString(Charsets.UTF_8)
            require(systemPrompt.isNotBlank()) { "Packaged Gemma system prompt is blank." }
            systemPromptSha256 = sha256(systemPromptBytes)
            File(output, "system_prompt.txt").writeBytes(systemPromptBytes)
            writeJson(
                File(output, "system_prompt_identity.json"),
                JSONObject()
                    .put("asset", SYSTEM_PROMPT_ASSET)
                    .put("sha256", systemPromptSha256)
                    .put("bytes", systemPromptBytes.size),
            )

            val sourceRunConfigFile = File(sourceRun, "run_config.json")
            require(sourceRunConfigFile.isFile) {
                "Captured source run has no run_config.json: ${sourceRunConfigFile.absolutePath}"
            }
            val sourceRunConfigBytes = sourceRunConfigFile.readBytes()
            val sourceRunConfig = JSONObject(sourceRunConfigBytes.toString(Charsets.UTF_8))
            require(sourceRunConfig.optString("provider") == "gemma") {
                "Captured source run must use provider=gemma."
            }
            require(sourceRunConfig.optString("promptPath") == "aar_asset") {
                "Captured source run must use the packaged AAR prompt."
            }
            require(sourceRunConfig.optBoolean("recordInputs", false)) {
                "Captured source run must have recordInputs=true."
            }
            require(sourceRunConfig.optString("accelerator") == "GPU") {
                "Captured source run must use accelerator=GPU."
            }
            require(sourceRunConfig.optBoolean("mtp", false)) {
                "Captured source run must have mtp=true."
            }
            require(sourceRunConfig.optBoolean("thinkingEnabled", false)) {
                "Captured source run must have thinkingEnabled=true."
            }
            require(sourceRunConfig.optDouble("temperature", Double.NaN) == 0.0) {
                "Captured source run must use temperature=0.0."
            }
            require(sourceRunConfig.optString("promptSha256") == systemPromptSha256) {
                "The installed packaged Gemma prompt does not match the captured source run."
            }
            val sourceCaseIds = sourceRunConfig.getJSONArray("cases").let { values ->
                (0 until values.length()).map(values::getString).toSet()
            }
            require(cases.all(sourceCaseIds::contains)) {
                "One or more requested cases are absent from the captured source run config."
            }

            val capturedInputs = cases.mapIndexed { index, caseId ->
                val sourceFile = File(sourceRun, "$caseId/attempt_1_input.txt")
                require(sourceFile.isFile && sourceFile.canRead()) {
                    "Missing captured first-attempt input for $caseId: ${sourceFile.absolutePath}"
                }
                val bytes = sourceFile.readBytes()
                require(bytes.isNotEmpty()) { "Captured first-attempt input is empty for $caseId." }
                val captured = CapturedInput(
                    caseId = caseId,
                    order = index + 1,
                    bytes = bytes,
                    text = bytes.toString(Charsets.UTF_8),
                    sha256 = sha256(bytes),
                )
                require(captured.text.isNotBlank()) { "Captured first-attempt input is blank for $caseId." }
                val caseDir = File(output, caseId).apply {
                    check(mkdirs()) { "Could not create case output: $absolutePath" }
                }
                writeJson(
                    File(caseDir, "source_input_identity.json"),
                    JSONObject()
                        .put("sourceRunId", sourceRunId)
                        .put("caseId", caseId)
                        .put("sourceRelativePath", "$sourceRunId/$caseId/attempt_1_input.txt")
                        .put("sourceCanonicalPath", sourceFile.canonicalPath)
                        .put("sha256", captured.sha256)
                        .put("bytes", bytes.size)
                        .put("systemPromptSha256", systemPromptSha256),
                )
                captured
            }

            runConfig
                .put("modelPath", modelPath)
                .put("modelSizeBytes", modelFile.length())
                .put("systemPromptAsset", SYSTEM_PROMPT_ASSET)
                .put("systemPromptSha256", systemPromptSha256)
                .put("systemPromptBytes", systemPromptBytes.size)
                .put("sourceRunConfigSha256", sha256(sourceRunConfigBytes))
                .put("sourceRunConfigBytes", sourceRunConfigBytes.size)
                .put("sourceInputSha256ByCase", JSONObject().apply {
                    capturedInputs.forEach { put(it.caseId, it.sha256) }
                })
                .put("status", "ready")
            writeJson(File(output, "run_config.json"), runConfig)

            require(Build.SUPPORTED_ABIS.any { it.equals("arm64-v8a", ignoreCase = true) }) {
                "Gemma GPU throughput requires arm64-v8a; device reports ${Build.SUPPORTED_ABIS.joinToString()}."
            }
            loadSdkGpuRuntime()
            val modelSupportsMtp = Capabilities(modelPath).use { it.hasSpeculativeDecodingSupport() }
            require(modelSupportsMtp) {
                "GPU+MTP throughput requires a model package with speculative-decoding support."
            }

            val previousBenchmark = ExperimentalFlags.enableBenchmark
            val previousMtp = ExperimentalFlags.enableSpeculativeDecoding
            var flagsRestored = false
            val initStartedAtNs = SystemClock.elapsedRealtimeNanos()
            try {
                ExperimentalFlags.enableBenchmark = true
                ExperimentalFlags.enableSpeculativeDecoding = true
                engine = Engine(
                    EngineConfig(
                        modelPath = modelPath,
                        backend = Backend.GPU(),
                        maxNumTokens = MAX_CONTEXT_TOKENS,
                    ),
                )
                engine!!.initialize()
                engineInitialized = true
            } finally {
                ExperimentalFlags.enableBenchmark = previousBenchmark
                ExperimentalFlags.enableSpeculativeDecoding = previousMtp
                flagsRestored =
                    ExperimentalFlags.enableBenchmark == previousBenchmark &&
                        ExperimentalFlags.enableSpeculativeDecoding == previousMtp
                runConfig
                    .put("engineInitializationElapsedMs", elapsedMs(initStartedAtNs))
                    .put("modelSupportsMtp", modelSupportsMtp)
                    .put("experimentalFlags", JSONObject()
                        .put("previousBenchmark", previousBenchmark)
                        .put("previousMtp", previousMtp ?: JSONObject.NULL)
                        .put("benchmarkDuringInitialization", true)
                        .put("mtpDuringInitialization", true)
                        .put("restoredAfterInitialization", flagsRestored))
                writeJson(File(output, "run_config.json"), runConfig)
            }
            runConfig
                .put("engineInitializationCount", 1)
                .put("status", "running")
            writeJson(File(output, "run_config.json"), runConfig)
            require(flagsRestored) { "LiteRT-LM experimental flags were not restored after initialization." }

            writeJson(
                File(output, "run_state.json"),
                JSONObject()
                    .put("status", "running")
                    .put("completedCases", 0)
                    .put("totalCases", cases.size)
                    .put("currentCase", JSONObject.NULL),
            )

            for (captured in capturedInputs) {
                val caseDir = File(output, captured.caseId)
                val report = JSONObject()
                    .put("id", captured.caseId)
                    .put("order", captured.order)
                    .put("status", "running")
                    .put("firstGenerationAfterInitialization", captured.order == 1)
                    .put("engineReused", captured.order > 1)
                    .put("sourceRunId", sourceRunId)
                    .put("sourceInputRelativePath", "$sourceRunId/${captured.caseId}/attempt_1_input.txt")
                    .put("sourceInputSha256", captured.sha256)
                    .put("sourceInputBytes", captured.bytes.size)
                    .put("systemPromptSha256", systemPromptSha256)
                results.put(report)
                writeJson(File(caseDir, "result.json"), report)
                writeJson(File(output, "results.json"), results)
                writeJson(
                    File(output, "run_state.json"),
                    JSONObject()
                        .put("status", "running")
                        .put("completedCases", completedCases)
                        .put("totalCases", cases.size)
                        .put("currentCase", captured.caseId)
                        .put("currentOrder", captured.order),
                )
                instrumentation.sendStatus(
                    0,
                    Bundle().apply {
                        putString(
                            "stream",
                            "${captured.caseId} throughput started (${captured.order}/${cases.size})\n",
                        )
                    },
                )

                val caseStartedAtNs = SystemClock.elapsedRealtimeNanos()
                try {
                    val conversation = checkNotNull(engine).createConversation(
                        ConversationConfig(
                            systemInstruction = Contents.of(systemPrompt),
                            samplerConfig = SamplerConfig(
                                temperature = 0.0,
                                topK = 1,
                                topP = 1.0,
                            ),
                            maxOutputToken = MAX_OUTPUT_TOKENS,
                            thinkingConfig = ThinkingConfig(
                                enableThinking = true,
                                thinkingTokenBudget = THINKING_TOKEN_BUDGET,
                            ),
                        ),
                    )
                    val generated = conversation.use {
                        val generationStartedAtNs = SystemClock.elapsedRealtimeNanos()
                        val response = it.sendMessage(captured.text)
                        val generationElapsedMs = elapsedMs(generationStartedAtNs)
                        val rawText = response.rawText()
                        val rawBytes = rawText.toByteArray(Charsets.UTF_8)
                        File(caseDir, "raw_response.txt").writeBytes(rawBytes)
                        report
                            .put("generationElapsedMs", generationElapsedMs)
                            .put("rawResponseSha256", sha256(rawBytes))
                            .put("rawResponseBytes", rawBytes.size)
                        writeJson(File(caseDir, "result.json"), report)
                        writeJson(File(output, "results.json"), results)
                        GenerationCapture(
                            rawText = rawText,
                            benchmark = readBenchmarkSnapshot(it),
                        )
                    }

                    report
                        .put("caseElapsedMs", elapsedMs(caseStartedAtNs))
                        .put("prefillTokenCount", generated.benchmark.prefillTokenCount)
                        .put("decodeTokenCount", generated.benchmark.decodeTokenCount)
                        .putEvidenceDouble("benchmarkInitSeconds", generated.benchmark.initSeconds)
                        .putEvidenceDouble(
                            "timeToFirstTokenSeconds",
                            generated.benchmark.timeToFirstTokenSeconds,
                        )
                        .putEvidenceDouble(
                            "prefillTokensPerSecond",
                            generated.benchmark.prefillTokensPerSecond,
                        )
                        .putEvidenceDouble(
                            "decodeTokensPerSecond",
                            generated.benchmark.decodeTokensPerSecond,
                        )
                    writeJson(File(caseDir, "result.json"), report)
                    writeJson(File(output, "results.json"), results)

                    require(generated.rawText.isNotBlank()) {
                        "LiteRT-LM returned no Content.Text for ${captured.caseId}."
                    }
                    generated.benchmark.requireValid(captured.caseId)

                    report
                        .put("status", "success")
                        .put("caseElapsedMs", elapsedMs(caseStartedAtNs))
                    completedCases++
                    writeJson(File(caseDir, "result.json"), report)
                    writeJson(File(output, "results.json"), results)
                    writeJson(
                        File(output, "run_state.json"),
                        JSONObject()
                            .put("status", "running")
                            .put("completedCases", completedCases)
                            .put("totalCases", cases.size)
                            .put("currentCase", JSONObject.NULL),
                    )
                    instrumentation.sendStatus(
                        0,
                        Bundle().apply {
                            putString(
                                "stream",
                                "${captured.caseId} ${generated.benchmark.decodeTokensPerSecond} native decode tok/s " +
                                    "(${captured.order}/${cases.size})\n",
                            )
                        },
                    )
                    Log.i(
                        LOG_TAG,
                        "${captured.caseId} prefill=${generated.benchmark.prefillTokenCount} " +
                            "decode=${generated.benchmark.decodeTokenCount} " +
                            "decodeTokensPerSecond=${generated.benchmark.decodeTokensPerSecond}",
                    )
                } catch (failure: Throwable) {
                    failedCases++
                    report
                        .put("status", "failure")
                        .put("caseElapsedMs", elapsedMs(caseStartedAtNs))
                        .put("errorType", failure.javaClass.name)
                        .put("error", failure.message.orEmpty())
                    File(caseDir, "error.txt").writeText(stackTrace(failure))
                    writeJson(File(caseDir, "result.json"), report)
                    writeJson(File(output, "results.json"), results)
                    throw failure
                }
            }
        } catch (failure: Throwable) {
            primaryFailure = failure
            File(output, "error.txt").writeText(stackTrace(failure))
            Log.e(LOG_TAG, "Gemma throughput replay failed; evidence=${output.absolutePath}", failure)
        } finally {
            closeFailure = runCatching { engine?.close() }.exceptionOrNull()
            engineClosed = engine == null || closeFailure == null
            val finalStatus = if (primaryFailure == null && closeFailure == null && completedCases == cases.size) {
                "success"
            } else {
                "failure"
            }
            val summary = JSONObject()
                .put("schemaVersion", 1)
                .put("kind", "captured_sdk_prompt_native_throughput")
                .put("runId", runId)
                .put("sourceRunId", sourceRunId)
                .put("status", finalStatus)
                .put("totalCases", cases.size)
                .put("completedCases", completedCases)
                .put("failedCases", failedCases)
                .put("runFailure", primaryFailure != null)
                .put("engineInitialized", engineInitialized)
                .put("engineClosed", engineClosed)
                .put("runElapsedMs", elapsedMs(runStartedAtNs))
                .put("systemPromptSha256", systemPromptSha256 ?: JSONObject.NULL)
                .put("acceptanceEvaluated", false)
            primaryFailure?.let {
                summary
                    .put("errorType", it.javaClass.name)
                    .put("error", it.message.orEmpty())
            }
            closeFailure?.let {
                summary
                    .put("closeErrorType", it.javaClass.name)
                    .put("closeError", it.message.orEmpty())
                File(output, "close_error.txt").writeText(stackTrace(it))
            }
            writeJson(File(output, "summary.json"), summary)
            writeJson(
                File(output, "run_state.json"),
                JSONObject(summary.toString()).put("finishedAtEpochMs", System.currentTimeMillis()),
            )
        }

        primaryFailure?.let { throw it }
        closeFailure?.let { throw AssertionError("LiteRT-LM engine close failed.", it) }
        check(completedCases == cases.size) {
            "Only $completedCases/${cases.size} throughput cases completed; evidence=${output.absolutePath}"
        }
    }

    private fun parseCases(value: String?): List<String> {
        val cases = value
            ?.takeIf { it.isNotBlank() }
            ?.split(',')
            ?.map(String::trim)
            ?: DEFAULT_CASES
        require(cases.isNotEmpty() && cases.none(String::isBlank)) {
            "cases must contain at least one comma-separated case ID."
        }
        cases.forEach { safeName(it, "case ID") }
        require(cases.size == cases.toSet().size) { "cases must not contain duplicates." }
        return cases
    }

    private fun safeName(value: String, label: String): String {
        require(SAFE_NAME.matches(value)) {
            "$label must contain only letters, digits, underscores, or hyphens (1..100 characters)."
        }
        return value
    }

    private fun loadSdkGpuRuntime() {
        try {
            System.loadLibrary("LiteRt")
            System.loadLibrary("LiteRtTopKOpenClSampler")
        } catch (failure: LinkageError) {
            throw IllegalStateException(
                "Gemma GPU throughput could not load the packaged ARM64 LiteRT runtime.",
                failure,
            )
        }
    }

    private data class CapturedInput(
        val caseId: String,
        val order: Int,
        val bytes: ByteArray,
        val text: String,
        val sha256: String,
    )

    private data class BenchmarkSnapshot(
        val initSeconds: Double,
        val timeToFirstTokenSeconds: Double,
        val prefillTokenCount: Int,
        val decodeTokenCount: Int,
        val prefillTokensPerSecond: Double,
        val decodeTokensPerSecond: Double,
    ) {
        fun requireValid(caseId: String) {
            require(initSeconds.isFinite() && initSeconds >= 0.0) {
                "$caseId returned invalid benchmark init seconds: $initSeconds"
            }
            require(timeToFirstTokenSeconds.isFinite() && timeToFirstTokenSeconds >= 0.0) {
                "$caseId returned invalid time to first token: $timeToFirstTokenSeconds"
            }
            require(prefillTokenCount > 0) {
                "$caseId returned invalid prefill token count: $prefillTokenCount"
            }
            require(decodeTokenCount > 0) {
                "$caseId returned invalid decode token count: $decodeTokenCount"
            }
            require(prefillTokensPerSecond.isFinite() && prefillTokensPerSecond > 0.0) {
                "$caseId returned invalid prefill tokens/sec: $prefillTokensPerSecond"
            }
            require(decodeTokensPerSecond.isFinite() && decodeTokensPerSecond > 0.0) {
                "$caseId returned invalid decode tokens/sec: $decodeTokensPerSecond"
            }
        }
    }

    private data class GenerationCapture(
        val rawText: String,
        val benchmark: BenchmarkSnapshot,
    )

    /** LiteRT-LM 0.15.0 publishes these Java getters but hides them from Kotlin metadata. */
    private fun readBenchmarkSnapshot(conversation: Any): BenchmarkSnapshot {
        val benchmark = conversation.javaClass.getMethod("getBenchmarkInfo").invoke(conversation)
        val type = benchmark.javaClass
        fun number(method: String): Number = type.getMethod(method).invoke(benchmark) as Number
        return BenchmarkSnapshot(
            initSeconds = number("getInitTimeInSecond").toDouble(),
            timeToFirstTokenSeconds = number("getTimeToFirstTokenInSecond").toDouble(),
            prefillTokenCount = number("getLastPrefillTokenCount").toInt(),
            decodeTokenCount = number("getLastDecodeTokenCount").toInt(),
            prefillTokensPerSecond = number("getLastPrefillTokensPerSecond").toDouble(),
            decodeTokensPerSecond = number("getLastDecodeTokensPerSecond").toDouble(),
        )
    }

    private fun Message.rawText(): String = contents.contents.joinToString(separator = "") { content ->
        when (content) {
            is Content.Text -> content.text
            else -> ""
        }
    }

    private fun elapsedMs(startedAtNs: Long): Long =
        (SystemClock.elapsedRealtimeNanos() - startedAtNs) / 1_000_000L

    private fun JSONObject.putEvidenceDouble(name: String, value: Double): JSONObject =
        put(name, if (value.isFinite()) value else value.toString())

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString(separator = "") { "%02x".format(Locale.US, it.toInt() and 0xff) }

    private fun writeJson(file: File, value: Any) {
        val text = when (value) {
            is JSONObject -> value.toString(2)
            is JSONArray -> value.toString(2)
            else -> error("Unsupported JSON evidence type: ${value.javaClass.name}")
        }
        file.writeText(text + "\n", Charsets.UTF_8)
    }

    private fun stackTrace(failure: Throwable): String = StringWriter().also { buffer ->
        PrintWriter(buffer).use { failure.printStackTrace(it) }
    }.toString()

    private companion object {
        const val LOG_TAG = "GenUiGemmaThroughput"
        const val SYSTEM_PROMPT_ASSET = "genuicraft/prompts/gemma.txt"
        const val DEFAULT_SOURCE_RUN_ID = "ps_final50_scaffold_v11"
        const val MAX_CONTEXT_TOKENS = 16_384
        const val MAX_OUTPUT_TOKENS = 8_192
        const val THINKING_TOKEN_BUDGET = 1_024
        val SAFE_NAME = Regex("[A-Za-z0-9_-]{1,100}")
        val DEFAULT_CASES = listOf("BXP-003", "BXP-030", "BXP-047")
    }
}
