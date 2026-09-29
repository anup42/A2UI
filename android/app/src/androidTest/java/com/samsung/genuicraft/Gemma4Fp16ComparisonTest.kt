package com.samsung.genuicraft

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiConversionProfile
import com.samsung.genuicraft.sdk.GenUiConversionResult
import com.samsung.genuicraft.sdk.GenUiGenerationAttempt
import com.samsung.genuicraft.sdk.GenUiGenerationSessionMetrics
import com.samsung.genuicraft.sdk.GenUiModelProfiles
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiRequest
import com.samsung.genuicraft.sdk.GenUiSession
import com.samsung.genuicraft.sdk.GenUiTrainedConverter
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
import com.samsung.genuicraft.sdk.provider.Gemma4Provider
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Warm, paired Gemma 4 GPU precision probe using the AAR's frozen trained conversion profile.
 * Quality failures are evidence, not instrumentation failures. The host runner alternates arms.
 *
 * Required arguments: modelPath, precision=FP32|FP16_CORRECTED, outputDir, label.
 * Optional: mtp=true|false, cases=BXP-001,BXP-003,BXP-004, caseTimeoutMs=360000.
 * outputDir is a new leaf under the app's sdk_fp16_comparison external-files directory.
 */
@RunWith(AndroidJUnit4::class)
class Gemma4Fp16ComparisonTest {
    @Test
    fun compareTrainedGpuPrecision() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val args = InstrumentationRegistry.getArguments()
        val gson = GsonBuilder().setPrettyPrinting().serializeNulls().disableHtmlEscaping().create()
        val outputDir = safeName(requireArg(args, "outputDir"), "outputDir")
        val label = safeName(requireArg(args, "label"), "label")
        val precision = Gemma4GpuPrecision.valueOf(requireArg(args, "precision").uppercase())
        require(precision in setOf(Gemma4GpuPrecision.FP32, Gemma4GpuPrecision.FP16_CORRECTED)) {
            "precision must be FP32 or FP16_CORRECTED"
        }
        val mtp = args.getString("mtp", "false")!!.toBooleanStrictOrNull()
            ?: error("mtp must be true or false")
        val timeoutMs = args.getString("caseTimeoutMs", "360000")!!.toLong().also {
            require(it in 1_000L..3_600_000L) { "caseTimeoutMs out of range" }
        }
        val modelPathArgument = requireArg(args, "modelPath")
        require(File(modelPathArgument).isAbsolute) { "modelPath must be absolute" }
        val modelFile = File(modelPathArgument).canonicalFile
        require(modelFile.isAbsolute && modelFile.isFile && modelFile.canRead() && modelFile.length() > 0L) {
            "modelPath must identify a readable, non-empty absolute file"
        }
        require(modelFile.extension.equals("litertlm", ignoreCase = true)) { "Expected .litertlm model" }

        val corpusBytes = context.assets.open(CORPUS_ASSET).use { it.readBytes() }
        require(sha256(corpusBytes) == PINNED_CORPUS_SHA256) { "Bixby50 asset changed" }
        val corpus = corpusBytes.toString(Charsets.UTF_8).lineSequence().filter(String::isNotBlank)
            .map { JsonParser.parseString(it).asJsonObject }
            .associateBy { it.get("id").asString }
        val ids = args.getString("cases", DEFAULT_CASES)!!.split(',').map(String::trim)
        require(ids.isNotEmpty() && ids.size == ids.distinct().size && ids.all(corpus::containsKey)) {
            "cases must be unique IDs from the frozen Bixby50 asset"
        }
        val promptBytes = context.assets.open(GenUiTrainedConverter.PROMPT_ASSET).use { it.readBytes() }
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "sdk_fp16_comparison/$outputDir")
        require(!root.exists() && root.mkdirs()) { "Output directory must be new: ${root.absolutePath}" }
        File(root, "shared_prompt.json").writeBytes(promptBytes)
        val config = linkedMapOf<String, Any?>(
            "schemaVersion" to 1,
            "kind" to "trained_e2b_gpu_precision_comparison",
            "label" to label,
            "outputDir" to outputDir,
            "startedAtEpochMs" to System.currentTimeMillis(),
            "modelPathArgument" to modelPathArgument,
            "modelPath" to modelFile.absolutePath,
            "modelBytes" to modelFile.length(),
            "precision" to precision.name,
            "gpuBackend" to true,
            "mtpRequested" to mtp,
            "cases" to ids,
            "warmupCase" to ids.first(),
            "caseTimeoutMs" to timeoutMs,
            "corpusAsset" to CORPUS_ASSET,
            "corpusSha256" to sha256(corpusBytes),
            "promptAsset" to GenUiTrainedConverter.PROMPT_ASSET,
            "promptSha256" to sha256(promptBytes),
            "promptContractSha256" to GenUiTrainedConverter.CONTRACT_SHA256,
            "profile" to GenUiConversionProfile.TRAINED_E2B_V10_W4.name,
            "maxContextTokens" to GenUiModelProfiles.TRAINED_CONTEXT_TOKENS,
            "maxOutputTokens" to GenUiModelProfiles.TRAINED_OUTPUT_TOKENS,
            "temperature" to 0.0,
            "thinkingEnabled" to false,
            "sourceFallbackEnabled" to false,
            "generatedDslRepairEnabled" to true,
            "requireSourceIntegrity" to false,
            "renderer" to "GenUiSdkDemoActivity/GenUiView smoke check",
            "device" to mapOf(
                "manufacturer" to Build.MANUFACTURER,
                "model" to Build.MODEL,
                "product" to Build.PRODUCT,
                "sdkInt" to Build.VERSION.SDK_INT,
                "fingerprint" to Build.FINGERPRINT,
            ),
        )
        writeJson(File(root, "run_config.json"), config, gson)
        writeJson(File(root, "status.json"), mapOf("state" to "starting", "completed" to 0), gson)

        val provider = Gemma4Provider(
            GenUiModelProfiles.trainedE2b(
                modelPath = modelFile.absolutePath,
                enableMtp = mtp,
                enableMetrics = true,
                gpuPrecision = precision,
            ),
        )
        // MTP publishes acceptance only as its engine closes. The session's normal per-conversion
        // metrics hook would therefore cold-start every MTP case; defer that one hook until the arm ends.
        val deferredMetricsProvider = object : GenUiProvider by provider {
            override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? = null
        }
        val session = GenUiSession(
            context,
            deferredMetricsProvider,
            GenUiConversionProfile.TRAINED_E2B_V10_W4,
        )
        val reports = mutableListOf<Map<String, Any?>>()
        var sessionMetrics: Any? = null
        var closeError: String? = null
        try {
            ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { activity ->
                val warmup = runCase(
                    context, instrumentation, activity, session, corpus.getValue(ids.first()),
                    File(root, "warmup"), true, timeoutMs, gson,
                )
                writeJson(File(root, "warmup_result.json"), warmup, gson)
                instrumentation.sendStatus(0, Bundle().apply {
                    putString("stream", "$label warmup ${warmup["status"]}\n")
                })
                ids.forEachIndexed { index, id ->
                    val report = runCase(
                        context, instrumentation, activity, session, corpus.getValue(id),
                        File(root, id), false, timeoutMs, gson,
                    )
                    reports += report
                    writeJson(File(root, "results.json"), reports, gson)
                    writeJson(
                        File(root, "status.json"),
                        mapOf("state" to "running", "completed" to reports.size, "total" to ids.size),
                        gson,
                    )
                    instrumentation.sendStatus(0, Bundle().apply {
                        putString("stream", "$label $id ${report["status"]} (${index + 1}/${ids.size})\n")
                    })
                }
            }
            sessionMetrics = runCatching { provider.finishGenerationMetrics() }
                .fold(onSuccess = { it }, onFailure = { mapOf("error" to it.toString()) })
        } finally {
            try {
                session.closeAndAwait()
            } catch (failure: Exception) {
                closeError = failure.toString()
            }
            writeJson(File(root, "session_metrics.json"), sessionMetrics, gson)
            writeJson(
                File(root, "summary.json"),
                mapOf(
                    "label" to label,
                    "precision" to precision.name,
                    "mtpRequested" to mtp,
                    "completed" to reports.size,
                    "total" to ids.size,
                    "runComplete" to (reports.size == ids.size),
                    "rawStrictValid" to reports.count { it["rawStrictValid"] == true },
                    "repairedStrictValid" to reports.count { it["repairedStrictValid"] == true },
                    "sdkRenderSmokeValid" to reports.count { it["sdkRenderSmokeValid"] == true },
                    "closeError" to closeError,
                    "finishedAtEpochMs" to System.currentTimeMillis(),
                ),
                gson,
            )
            writeJson(
                File(root, "status.json"),
                mapOf("state" to if (reports.size == ids.size && closeError == null) "complete" else "partial",
                    "completed" to reports.size, "total" to ids.size),
                gson,
            )
        }
        check(reports.size == ids.size && closeError == null) {
            "Instrumentation did not complete; inspect preserved evidence: ${root.absolutePath}"
        }
    }

    private suspend fun runCase(
        context: Context,
        instrumentation: android.app.Instrumentation,
        activity: ActivityScenario<GenUiSdkDemoActivity>,
        session: GenUiSession,
        source: com.google.gson.JsonObject,
        dir: File,
        warmup: Boolean,
        timeoutMs: Long,
        gson: Gson,
    ): Map<String, Any?> {
        check(dir.mkdirs()) { "Could not create ${dir.absolutePath}" }
        val id = source.get("id").asString
        val text = source.get("text").asString
        val query = source.get("query").asString
        File(dir, "source.json").writeText(gson.toJson(source) + "\n")
        val started = System.nanoTime()
        val before = deviceState(context)
        var conversion: GenUiConversionResult? = null
        var exception: String? = null
        try {
            conversion = withTimeout(timeoutMs) { session.convert(GenUiRequest(text, query)) }
        } catch (failure: Exception) {
            exception = failure.toString()
        }
        val elapsedMs = (System.nanoTime() - started) / 1_000_000L
        val attempts = session.attemptSnapshots
        attempts.forEach { attempt -> saveAttempt(dir, attempt, gson) }
        val first = attempts.firstOrNull()
        val raw = first?.rawText.orEmpty()
        val strict = runCatching { GenUiCompiler.compile(raw) }
        val success = conversion as? GenUiConversionResult.Success
        val repairedStrict = success?.let { runCatching { GenUiCompiler.compile(it.document.a2uiJson) } }
        val render = if (!warmup && success != null && repairedStrict?.isSuccess == true) {
            File(dir, "repaired.express").writeText(success.document.express)
            File(dir, "repaired.a2ui.json").writeText(success.document.a2uiJson)
            renderSdk(instrumentation, activity, success, dir)
        } else null
        if (warmup && success != null) {
            File(dir, "repaired.express").writeText(success.document.express)
            File(dir, "repaired.a2ui.json").writeText(success.document.a2uiJson)
        }
        val output = first?.output
        val report = linkedMapOf<String, Any?>(
            "id" to id,
            "warmup" to warmup,
            "status" to when {
                exception != null -> "exception"
                conversion is GenUiConversionResult.Failure -> "conversion_failure"
                success == null -> "no_conversion_result"
                render == null && !warmup -> "render_not_attempted"
                render?.get("sdkRenderSmokeValid") == false -> "render_smoke_failure"
                else -> "complete"
            },
            "caseElapsedMs" to elapsedMs,
            "deviceBefore" to before,
            "deviceAfter" to deviceState(context),
            "attempts" to attempts.size,
            "rawCaptured" to (first != null),
            "rawComplete" to first?.complete,
            "rawSha256" to first?.rawText?.toByteArray(Charsets.UTF_8)?.let(::sha256),
            "rawStrictValid" to strict.isSuccess,
            "rawStrictError" to strict.exceptionOrNull()?.message,
            "repairedStrictValid" to (repairedStrict?.isSuccess == true),
            "repairedStrictError" to repairedStrict?.exceptionOrNull()?.message,
            "repairKind" to success?.repairKind?.name,
            "sourceFidelityWarnings" to success?.warnings?.count { it.startsWith("Source fidelity:") },
            "usedFallback" to (success?.repairKind?.name == "SOURCE_TEXT_FALLBACK"),
            "conversionElapsedMs" to when (conversion) {
                is GenUiConversionResult.Success -> conversion.elapsedMs
                is GenUiConversionResult.Failure -> conversion.elapsedMs
                null -> null
            },
            "conversionError" to (conversion as? GenUiConversionResult.Failure)?.message,
            "exception" to exception,
            "runtime" to output?.runtime,
            "renderedPromptSha256" to output?.renderedPromptSha256,
            "finishReason" to output?.finishReason?.name,
            "finishDetail" to output?.finishDetail,
            "inputTokens" to output?.metrics?.inputTokens,
            "outputTokens" to (output?.metrics?.outputTokens ?: output?.outputTokens),
            "prefillTokensPerSecond" to output?.metrics?.prefillTokensPerSecond,
            "decodeTokensPerSecond" to output?.metrics?.decodeTokensPerSecond,
            "timeToFirstTokenSeconds" to output?.metrics?.timeToFirstTokenSeconds,
            "engineInitializationSeconds" to output?.metrics?.engineInitializationSeconds,
            "engineInitializedForRequest" to output?.metrics?.engineInitializedForRequest,
            "nativeInitializationPhaseSeconds" to output?.metrics?.nativeInitializationPhaseSeconds,
            "providerElapsedMs" to first?.elapsedNanos?.div(1_000_000L),
            "sdkRenderSmokeValid" to render?.get("sdkRenderSmokeValid"),
            "renderStatus" to render,
        )
        writeJson(File(dir, "result.json"), report, gson)
        return report
    }

    private suspend fun renderSdk(
        instrumentation: android.app.Instrumentation,
        activity: ActivityScenario<GenUiSdkDemoActivity>,
        conversion: GenUiConversionResult.Success,
        dir: File,
    ): Map<String, Any?> {
        val document = conversion.document
        var documentMatched = false
        return try {
            activity.onActivity { host ->
                host.showDocument(document, "FP16 comparison ${dir.name}")
                documentMatched = host.renderedDocumentForTest()?.a2uiJson == document.a2uiJson
            }
            instrumentation.waitForIdleSync()
            val device = UiDevice.getInstance(instrumentation)
            val currentCaseVisible = device.wait(
                Until.hasObject(By.textContains("FP16 comparison ${dir.name}")), 10_000,
            )
            delay(500)
            val errorVisible = device.hasObject(By.textContains("Unable to render GenUI"))
            val screenshot = device.takeScreenshot(File(dir, "sdk_render.png"))
            mapOf(
                "attempted" to true,
                "documentMatched" to documentMatched,
                "currentCaseVisible" to currentCaseVisible,
                "errorVisible" to errorVisible,
                "screenshotCaptured" to screenshot,
                "sdkRenderSmokeValid" to (documentMatched && currentCaseVisible && !errorVisible && screenshot),
            )
        } catch (failure: Exception) {
            mapOf("attempted" to true, "sdkRenderSmokeValid" to false,
                "error" to failure.toString())
        }
    }

    private fun saveAttempt(dir: File, attempt: GenUiGenerationAttempt, gson: Gson) {
        File(dir, "attempt_${attempt.number}_raw.express").writeText(attempt.rawText)
        val prompt: GenUiPrompt? = attempt.prompt
        writeJson(
            File(dir, "attempt_${attempt.number}_prompt.json"),
            mapOf(
                "system" to prompt?.system,
                "user" to prompt?.user,
                "initialMessages" to prompt?.initialMessages,
                "maxOutputTokens" to prompt?.maxOutputTokens,
                "temperature" to prompt?.temperature,
                "chatTemplateOverride" to prompt?.chatTemplateOverride,
                "expectedRenderedPromptSha256" to prompt?.expectedRenderedPrompt
                    ?.toByteArray(Charsets.UTF_8)?.let(::sha256),
            ),
            gson,
        )
        writeJson(
            File(dir, "attempt_${attempt.number}_metadata.json"),
            mapOf("complete" to attempt.complete, "elapsedNanos" to attempt.elapsedNanos,
                "runtime" to attempt.output?.runtime,
                "finishReason" to attempt.output?.finishReason?.name,
                "finishDetail" to attempt.output?.finishDetail,
                "metrics" to attempt.output?.metrics),
            gson,
        )
    }

    private fun deviceState(context: Context): Map<String, Any?> {
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return mapOf(
            "epochMs" to System.currentTimeMillis(),
            "thermalStatus" to if (Build.VERSION.SDK_INT >= 29)
                context.getSystemService(PowerManager::class.java)?.currentThermalStatus else null,
            "batteryTemperatureC" to battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)
                ?.takeIf { it >= 0 }?.div(10.0),
            "batteryLevel" to battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1),
            "plugged" to battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1),
        )
    }

    private fun requireArg(args: Bundle, key: String): String =
        args.getString(key)?.trim()?.takeIf(String::isNotEmpty) ?: error("Missing -e $key")

    private fun safeName(value: String, field: String): String = value.also {
        require(NAME.matches(it)) { "$field must match ${NAME.pattern}" }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private fun writeJson(file: File, value: Any?, gson: Gson) {
        file.writeText(gson.toJson(value) + "\n")
    }

    companion object {
        private const val CORPUS_ASSET = "genuicraft_bixby50.jsonl"
        private const val PINNED_CORPUS_SHA256 =
            "fc46aa381957bed206f9e0f53e28edbb21b5096ca093985ad184fda73faaea0a"
        private const val DEFAULT_CASES = "BXP-001,BXP-003,BXP-004"
        private val NAME = Regex("[A-Za-z0-9_-]{1,100}")
    }
}
