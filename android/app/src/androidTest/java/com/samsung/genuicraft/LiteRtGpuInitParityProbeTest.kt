package com.samsung.genuicraft

import android.os.Build
import android.os.SystemClock
import android.util.Base64
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Initializes an arbitrary LiteRT-LM package with the GPU backend without
 * generating tokens. Full-delegation evidence remains in logcat, while a
 * compact machine-readable result is written into the target app's files.
 *
 * Instrumentation arguments:
 * - modelPath: required absolute device path to a .litertlm package.
 * - label: optional report/cache label.
 * - mtp: optional true/false speculative-decoding flag.
 * - maxNumTokens: optional context size; defaults to 4096.
 * - outputTokens: optional bounded decode length; defaults to zero (init only).
 * - backend: optional cpu/gpu selector; defaults to gpu.
 * - topK/topP/temperature/seed: optional deterministic sampler controls.
 * - prompt: optional bounded-generation prompt; defaults to "Hello".
 * - promptBase64: optional UTF-8/base64 prompt; preferred for ADB shell safety.
 */
@RunWith(AndroidJUnit4::class)
class LiteRtGpuInitParityProbeTest {

    @OptIn(ExperimentalApi::class)
    @Test
    fun packageInitializesWithGpuBackend() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        val modelPath = requireNotNull(arguments.getString("modelPath")?.trim()) {
            "Pass -e modelPath /absolute/device/path/model.litertlm"
        }
        val modelFile = File(modelPath)
        assertTrue("LiteRT-LM package does not exist: $modelPath", modelFile.isFile)
        assertTrue("LiteRT-LM package is empty: $modelPath", modelFile.length() > 0L)

        val mtpEnabled = arguments.getString("mtp")?.toBooleanStrictOrNull() ?: false
        val maxNumTokens = arguments.getString("maxNumTokens")
            ?.toIntOrNull()
            ?.coerceAtLeast(1_024)
            ?: 4_096
        val outputTokens = arguments.getString("outputTokens")
            ?.toIntOrNull()
            ?.coerceIn(0, 512)
            ?: 0
        val backendName = arguments.getString("backend")
            ?.trim()
            ?.lowercase()
            ?.takeIf { it == "cpu" || it == "gpu" }
            ?: "gpu"
        val topK = arguments.getString("topK")?.toIntOrNull()?.coerceAtLeast(1) ?: 1
        val topP = arguments.getString("topP")
            ?.toDoubleOrNull()
            ?.takeIf { it.isFinite() && it in 0.0..1.0 }
            ?: 1.0
        val temperature = arguments.getString("temperature")
            ?.toDoubleOrNull()
            ?.takeIf { it.isFinite() && it >= 0.0 }
            ?: 0.0
        val seed = arguments.getString("seed")?.toIntOrNull() ?: 42
        val prompt = arguments.getString("promptBase64")
            ?.takeIf { it.isNotBlank() }
            ?.let { String(Base64.decode(it, Base64.NO_WRAP), Charsets.UTF_8) }
            ?: arguments.getString("prompt")?.takeIf { it.isNotBlank() }
            ?: "Hello"
        val label = sanitizeLabel(
            arguments.getString("label")?.trim().orEmpty().ifBlank {
                modelFile.nameWithoutExtension
            }
        )
        val cacheDir = File(context.cacheDir, "litert_gpu_parity/$label").apply {
            check(mkdirs() || isDirectory) { "Could not create cache directory: $absolutePath" }
        }
        val reportDir = File(context.filesDir, "litert_gpu_parity_reports").apply {
            check(mkdirs() || isDirectory) { "Could not create report directory: $absolutePath" }
        }
        val reportFile = File(reportDir, "$label.json")
        val report = JSONObject()
            .put("schema_version", 2)
            .put("label", label)
            .put("model_path", modelFile.canonicalPath)
            .put("model_size_bytes", modelFile.length())
            .put("backend_requested", backendName.uppercase())
            .put("mtp_enabled", mtpEnabled)
            .put("max_num_tokens", maxNumTokens)
            .put("requested_output_tokens", outputTokens)
            .put("sampler_top_k", topK)
            .put("sampler_top_p", topP)
                        .put("sampler_temperature", temperature)
                        .put("sampler_seed", seed)
            .put("cache_dir", cacheDir.absolutePath)
            .put("device_model", Build.MODEL)
            .put("device_product", Build.PRODUCT)
            .put("android_sdk", Build.VERSION.SDK_INT)
            .put("supported_abis", Build.SUPPORTED_ABIS.joinToString(","))

        ExperimentalFlags.enableSpeculativeDecoding = mtpEnabled
        ExperimentalFlags.enableBenchmark = outputTokens > 0
        if (backendName == "gpu") {
            loadGpuSamplerDependencies()
        }
        var engine: Engine? = null
        val startedAtMs = SystemClock.elapsedRealtime()
        try {
            Log.i(
                LOG_TAG,
                "LITERT_GPU_INIT_PARITY start label=$label mtp=$mtpEnabled " +
                    "maxNumTokens=$maxNumTokens topK=$topK topP=$topP " +
                    "temperature=$temperature seed=$seed " +
                    "model=${modelFile.canonicalPath}",
            )
            engine = Engine(
                EngineConfig(
                    modelPath = modelFile.canonicalPath,
                backend = if (backendName == "cpu") Backend.CPU() else Backend.GPU(),
                    maxNumTokens = maxNumTokens,
                    cacheDir = cacheDir.absolutePath,
                )
            )
            engine.initialize()
            val elapsedMs = SystemClock.elapsedRealtime() - startedAtMs
            report
                .put("success", true)
                .put("init_elapsed_ms", elapsedMs)
            if (outputTokens > 0) {
                val generationStartedAtMs = SystemClock.elapsedRealtime()
                val response = engine.createConversation(
                    ConversationConfig(
                        samplerConfig = SamplerConfig(
                            topK = topK,
                            topP = topP,
                            temperature = temperature,
                            seed = seed,
                        ),
                        maxOutputToken = outputTokens,
                    )
                ).use { conversation ->
                    val message = conversation.sendMessage(prompt)
                    val benchmark = readBenchmarkSnapshot(conversation)
                    report
                        .put("benchmark_init_seconds", benchmark.initSeconds)
                        .put("time_to_first_token_seconds", benchmark.timeToFirstTokenSeconds)
                        .put("prefill_token_count", benchmark.prefillTokenCount)
                        .put("decode_token_count", benchmark.decodeTokenCount)
                        .put("prefill_tokens_per_second", benchmark.prefillTokensPerSecond)
                        .put("decode_tokens_per_second", benchmark.decodeTokensPerSecond)
                    message
                }
                report
                    .put(
                        "generation_elapsed_ms",
                        SystemClock.elapsedRealtime() - generationStartedAtMs,
                    )
                    .put("response_character_count", response.toString().length)
                    .put("response_text", response.toString().take(512))
            }
            Log.i(
                LOG_TAG,
                "LITERT_GPU_INIT_PARITY success label=$label mtp=$mtpEnabled " +
                    "initElapsedMs=$elapsedMs requestedOutputTokens=$outputTokens " +
                    "decodedTokens=${report.optInt("decode_token_count", 0)} " +
                    "decodeTokensPerSecond=${report.optDouble("decode_tokens_per_second")}",
            )
        } catch (t: Throwable) {
            val elapsedMs = SystemClock.elapsedRealtime() - startedAtMs
            report
                .put("success", false)
                .put("init_elapsed_ms", elapsedMs)
                .put("error_type", t.javaClass.name)
                .put("error", t.message.orEmpty())
            Log.e(
                LOG_TAG,
                "LITERT_GPU_INIT_PARITY failure label=$label mtp=$mtpEnabled " +
                    "initElapsedMs=$elapsedMs",
                t,
            )
            throw t
        } finally {
            val closeFailure = runCatching { engine?.close() }.exceptionOrNull()
            report.put("engine_closed", closeFailure == null)
            closeFailure?.let {
                report.put("close_error_type", it.javaClass.name)
                report.put("close_error", it.message.orEmpty())
            }
            reportFile.writeText(report.toString(2) + "\n")
            Log.i(LOG_TAG, "LITERT_GPU_INIT_PARITY report=${reportFile.absolutePath}")
        }
    }

    private fun sanitizeLabel(value: String): String {
        val normalized = value.replace(Regex("[^A-Za-z0-9._-]+"), "_").trim('_')
        return normalized.ifBlank { "probe" }.take(96)
    }

    private fun loadGpuSamplerDependencies() {
        listOf(
            "c++_shared",
            "LiteRt",
            "LiteRtTopKOpenClSampler",
        ).forEach { library ->
            runCatching { System.loadLibrary(library) }
                .onFailure { Log.w(LOG_TAG, "GPU sampler dependency failed name=$library", it) }
        }
    }

    private data class BenchmarkSnapshot(
        val initSeconds: Double,
        val timeToFirstTokenSeconds: Double,
        val prefillTokenCount: Int,
        val decodeTokenCount: Int,
        val prefillTokensPerSecond: Double,
        val decodeTokensPerSecond: Double,
    )

    /** LiteRT exposes these getters in bytecode but hides them from Kotlin metadata. */
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

    private companion object {
        private const val LOG_TAG = "LiteRtGpuParityProbe"
    }
}
