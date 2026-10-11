package com.samsung.genuicraft

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiConversionProfile
import com.samsung.genuicraft.sdk.GenUiConversionResult
import com.samsung.genuicraft.sdk.GenUiGenerationObserver
import com.samsung.genuicraft.sdk.GenUiGenerationSessionMetrics
import com.samsung.genuicraft.sdk.GenUiModelProfiles
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiRequest
import com.samsung.genuicraft.sdk.GenUiSession
import com.samsung.genuicraft.sdk.GenUiTrainedConverter
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
import com.samsung.genuicraft.sdk.provider.Gemma4Provider
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Bounded real-model conversion probe. No Activity, screenshots, or Espresso run in the measured
 * interval. A preview callback is SDK publication, not proof that a frame was painted on screen.
 *
 * Required: modelPath (prepared corrected FP16 package), outputDir (new leaf).
 * Optional: cases=BXP-001,BXP-003,BXP-031 (duplicates allowed), repeats=1, mtp=true,
 * streaming=true, reuse=true, warmupCase=BXP-001, caseTimeoutMs=360000,
 * cooldownMs=20000, thermalCooldownTimeoutMs=120000.
 * Reuse mode excludes one cold warmup and defers MTP engine-close metrics until all cases finish.
 * Non-reuse mode constructs, measures, finalizes metrics, and closes an engine for every case.
 */
@RunWith(AndroidJUnit4::class)
class InferenceBottleneckProbeTest {
    @Test
    fun measureTrainedConverterBottlenecks() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val args = InstrumentationRegistry.getArguments()
        val gson = GsonBuilder().setPrettyPrinting().serializeNulls().disableHtmlEscaping().create()
        val modelPath = requireArg(args, "modelPath")
        val model = File(modelPath).canonicalFile
        require(model.isAbsolute && model.isFile && model.canRead() && model.length() > 0L &&
            model.extension.equals("litertlm", ignoreCase = true)) {
            "modelPath must be a readable absolute .litertlm file"
        }
        val leaf = requireArg(args, "outputDir")
        require(Regex("[A-Za-z0-9_-]{1,80}").matches(leaf)) { "outputDir must be a simple leaf" }
        val mtp = boolArg(args, "mtp", true)
        val streaming = boolArg(args, "streaming", true)
        val reuse = boolArg(args, "reuse", true)
        val repeats = args.getString("repeats", "1")!!.toInt().also { require(it in 1..4) }
        val timeoutMs = args.getString("caseTimeoutMs", "360000")!!.toLong().also {
            require(it in 1_000L..3_600_000L)
        }
        val cooldownMs = args.getString("cooldownMs", "20000")!!.toLong().also {
            require(it in 0L..600_000L) { "cooldownMs out of range" }
        }
        val thermalCooldownTimeoutMs = args.getString("thermalCooldownTimeoutMs", "120000")!!
            .toLong().also {
                require(it in 0L..600_000L) { "thermalCooldownTimeoutMs out of range" }
            }
        val corpusBytes = context.assets.open(CORPUS_ASSET).use { it.readBytes() }
        require(sha256(corpusBytes) == CORPUS_SHA256) { "Frozen Bixby50 asset changed" }
        val corpus = corpusBytes.toString(Charsets.UTF_8).lineSequence().filter(String::isNotBlank)
            .map { JsonParser.parseString(it).asJsonObject }
            .associateBy { it.get("id").asString }
        val cases = args.getString("cases", DEFAULT_CASES)!!.split(',').map(String::trim)
        require(cases.isNotEmpty() && cases.size * repeats <= 16 && cases.all(corpus::containsKey)) {
            "cases must select at most 16 total frozen Bixby50 requests"
        }
        val warmupCase = args.getString("warmupCase", "BXP-001")!!.trim()
        require(corpus.containsKey(warmupCase)) { "warmupCase must be in frozen Bixby50" }
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "sdk_inference_bottleneck/$leaf")
        require(!root.exists() && root.mkdirs()) { "Output directory must be new: ${root.absolutePath}" }
        val promptBytes = context.assets.open(GenUiTrainedConverter.PROMPT_ASSET).use { it.readBytes() }
        File(root, "shared_prompt.json").writeBytes(promptBytes)
        fun write(name: String, value: Any?) = File(root, name).writeText(gson.toJson(value) + "\n")
        write("run_config.json", linkedMapOf(
            "schemaVersion" to 1, "kind" to "trained_e2b_inference_bottleneck_probe",
            "modelPath" to model.absolutePath, "modelBytes" to model.length(),
            "gpuPrecision" to "FP16_CORRECTED", "mtpRequested" to mtp,
            "streamingRenderingEnabled" to streaming, "reuseEngine" to reuse,
            "cases" to cases, "repeats" to repeats,
            "measuredOrder" to (1..repeats).flatMap { cases },
            "warmupCase" to warmupCase.takeIf { reuse }, "caseTimeoutMs" to timeoutMs,
            "cooldownMs" to cooldownMs,
            "thermalCooldownTimeoutMs" to thermalCooldownTimeoutMs,
            "corpusSha256" to sha256(corpusBytes), "promptSha256" to sha256(promptBytes),
            "promptContractSha256" to GenUiTrainedConverter.CONTRACT_SHA256,
            "profile" to GenUiConversionProfile.TRAINED_E2B_V10_W4.name,
            "maxContextTokens" to GenUiModelProfiles.TRAINED_CONTEXT_TOKENS,
            "maxOutputTokens" to GenUiModelProfiles.TRAINED_OUTPUT_TOKENS,
            "temperature" to 0.0, "thinkingEnabled" to false,
            "sourceFallbackEnabled" to false, "generatedDslRepairEnabled" to true,
            "device" to mapOf("manufacturer" to Build.MANUFACTURER, "model" to Build.MODEL,
                "sdkInt" to Build.VERSION.SDK_INT, "fingerprint" to Build.FINGERPRINT),
            "timingBoundary" to "convert wall includes native generation, SDK preview/repair and callbacks; file writes and thermal sampling occur outside it",
            "previewBoundary" to "first SDK preview callback; no screen paint measurement",
            "callbackWorkBoundary" to "probe callback body only; SDK preview compilation is included in conversion wall time",
        ))
        write("status.json", mapOf("state" to "starting", "completed" to 0))

        fun newProvider() = Gemma4Provider(GenUiModelProfiles.trainedE2b(
            modelPath = model.absolutePath, enableMtp = mtp, enableMetrics = true,
            gpuPrecision = Gemma4GpuPrecision.FP16_CORRECTED,
        ))
        val reports = mutableListOf<Map<String, Any?>>()
        val teardown = mutableListOf<Map<String, Any?>>()
        var warmup: Map<String, Any?>? = null
        if (reuse) {
            val constructionStart = now()
            val native = newProvider()
            val constructionMs = msSince(constructionStart)
            // LiteRT publishes MTP acceptance only when its engine closes. The session's normal
            // finish hook would otherwise turn every supposed warm request into another cold one.
            val deferred = object : GenUiProvider by native {
                override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? = null
            }
            val session = GenUiSession(context, deferred, GenUiConversionProfile.TRAINED_E2B_V10_W4)
            try {
                warmup = runCase(context, session, corpus.getValue(warmupCase),
                    File(root, "warmup"), true, streaming, timeoutMs, gson)
                write("warmup_result.json", warmup)
                for ((index, id) in (1..repeats).flatMap { cases }.withIndex()) {
                    val cooling = cooldown(context, cooldownMs, thermalCooldownTimeoutMs)
                    val report = runCase(context, session, corpus.getValue(id),
                        File(root, "%03d_%s".format(index + 1, id)), false,
                        streaming, timeoutMs, gson, cooling)
                    reports += report
                    write("results.json", reports)
                    write("status.json", mapOf("state" to "running", "completed" to reports.size))
                }
            } finally {
                teardown += finishAndClose(native, session, "shared", constructionMs)
                write("teardown.json", teardown)
            }
        } else {
            for ((index, id) in (1..repeats).flatMap { cases }.withIndex()) {
                val cooling = cooldown(context, cooldownMs, thermalCooldownTimeoutMs)
                val constructionStart = now()
                val native = newProvider()
                val constructionMs = msSince(constructionStart)
                val deferred = object : GenUiProvider by native {
                    override suspend fun finishGenerationMetrics(): GenUiGenerationSessionMetrics? = null
                }
                val session = GenUiSession(context, deferred, GenUiConversionProfile.TRAINED_E2B_V10_W4)
                val caseName = "%03d_%s".format(index + 1, id)
                try {
                    reports += runCase(context, session, corpus.getValue(id),
                        File(root, caseName), false, streaming, timeoutMs, gson, cooling)
                    write("results.json", reports)
                    write("status.json", mapOf("state" to "running", "completed" to reports.size))
                } finally {
                    teardown += finishAndClose(native, session, caseName, constructionMs)
                    write("teardown.json", teardown)
                }
            }
        }
        val complete = reports.size == cases.size * repeats &&
            reports.all { it["status"] == "complete" } &&
            (!reuse || warmup?.get("status") == "complete") &&
            teardown.all { it["finishMetricsError"] == null && it["closeError"] == null &&
                (it["sessionMetrics"] as? GenUiGenerationSessionMetrics)
                    ?.speculativeDecodingEnabled == mtp }
        write("summary.json", mapOf("completed" to reports.size, "total" to cases.size * repeats,
            "successful" to reports.count { it["status"] == "complete" },
            "coldRequests" to reports.count { it["engineInitializedForRequest"] == true },
            "warmRequests" to reports.count { it["engineInitializedForRequest"] == false },
            "warmup" to warmup, "teardown" to teardown,
            "runComplete" to complete))
        write("status.json", mapOf("state" to if (complete) "complete" else "partial",
            "completed" to reports.size, "total" to cases.size * repeats))
        check(complete) { "Probe incomplete; inspect ${root.absolutePath}" }
    }

    private suspend fun runCase(
        context: Context,
        session: GenUiSession,
        source: JsonObject,
        dir: File,
        warmup: Boolean,
        streaming: Boolean,
        timeoutMs: Long,
        gson: com.google.gson.Gson,
        cooling: Map<String, Any?>? = null,
    ): Map<String, Any?> {
        check(dir.mkdirs()) { "Could not create ${dir.absolutePath}" }
        File(dir, "source.json").writeText(gson.toJson(source) + "\n")
        if (cooling != null) File(dir, "cooldown.json").writeText(gson.toJson(cooling) + "\n")
        val firstPartialNs = AtomicLong(-1L)
        val firstPreviewNs = AtomicLong(-1L)
        val firstReadyPreviewNs = AtomicLong(-1L)
        val partialCount = AtomicInteger()
        val previewCount = AtomicInteger()
        val readyPreviewCount = AtomicInteger()
        val callbackWorkNs = AtomicLong()
        val before = deviceState(context)
        val started = now()
        val observer = GenUiGenerationObserver(
            onPartialText = { _, raw ->
                val callbackStart = now()
                partialCount.incrementAndGet()
                if (raw.isNotEmpty()) firstPartialNs.compareAndSet(-1L, callbackStart)
                callbackWorkNs.addAndGet(now() - callbackStart)
            },
            onRenderSnapshot = { snapshot ->
                val callbackStart = now()
                if (!snapshot.isFinal) {
                    previewCount.incrementAndGet()
                    firstPreviewNs.compareAndSet(-1L, callbackStart)
                    if (snapshot.readyComponentCount > 0) {
                        readyPreviewCount.incrementAndGet()
                        firstReadyPreviewNs.compareAndSet(-1L, callbackStart)
                    }
                }
                callbackWorkNs.addAndGet(now() - callbackStart)
            },
        )
        var conversion: GenUiConversionResult? = null
        var failure: String? = null
        try {
            conversion = withTimeout(timeoutMs) {
                session.convert(GenUiRequest(source.get("text").asString,
                    source.get("query").asString), observer, streaming)
            }
        } catch (error: Exception) {
            failure = error.toString()
        }
        val conversionWallMs = msSince(started)
        val after = deviceState(context)
        val attempts = session.attemptSnapshots
        attempts.forEach { attempt ->
            File(dir, "attempt_${attempt.number}_raw.express").writeText(attempt.rawText)
            File(dir, "attempt_${attempt.number}_prompt.json").writeText(gson.toJson(attempt.prompt) + "\n")
            File(dir, "attempt_${attempt.number}_metadata.json").writeText(gson.toJson(mapOf(
                "complete" to attempt.complete, "providerElapsedNanos" to attempt.elapsedNanos,
                "runtime" to attempt.output?.runtime,
                "renderedPromptSha256" to attempt.output?.renderedPromptSha256,
                "finishReason" to attempt.output?.finishReason?.name,
                "finishDetail" to attempt.output?.finishDetail,
                "metrics" to attempt.output?.metrics)) + "\n")
        }
        val success = conversion as? GenUiConversionResult.Success
        success?.document?.let {
            File(dir, "final.express").writeText(it.express)
            File(dir, "final.a2ui.json").writeText(it.a2uiJson)
        }
        val native = attempts.firstOrNull()?.output?.metrics
        val providerTotalMs = attempts.mapNotNull { it.elapsedNanos }.sum().takeIf { it > 0L }
            ?.div(1_000_000.0)
        val report = linkedMapOf<String, Any?>(
            "id" to source.get("id").asString, "warmup" to warmup,
            "cooldown" to cooling,
            "status" to when {
                failure != null -> "exception"
                success != null -> "complete"
                else -> "conversion_failure"
            },
            "error" to (failure ?: (conversion as? GenUiConversionResult.Failure)?.message),
            "conversionWallMs" to conversionWallMs,
            "providerTotalMs" to providerTotalMs,
            "conversionMinusProviderMsApprox" to providerTotalMs?.let { conversionWallMs - it },
            "conversionElapsedMs" to when (val value = conversion) {
                is GenUiConversionResult.Success -> value.elapsedMs
                is GenUiConversionResult.Failure -> value.elapsedMs
                null -> null
            },
            "attemptCount" to attempts.size, "repairKind" to success?.repairKind?.name,
            "runtime" to attempts.firstOrNull()?.output?.runtime,
            "firstPartialCallbackMs" to elapsedOrNull(firstPartialNs.get(), started),
            "firstPreviewCallbackMs" to elapsedOrNull(firstPreviewNs.get(), started),
            "firstReadyPreviewCallbackMs" to elapsedOrNull(firstReadyPreviewNs.get(), started),
            "partialCallbackCount" to partialCount.get(),
            "previewSnapshotCount" to previewCount.get(),
            "readyPreviewSnapshotCount" to readyPreviewCount.get(),
            "callbackBodyWorkMs" to callbackWorkNs.get() / 1_000_000.0,
            "engineInitializedForRequest" to native?.engineInitializedForRequest,
            "nativeMetrics" to native,
            "attempts" to attempts.map { mapOf(
                "number" to it.number, "complete" to it.complete,
                "providerElapsedNanos" to it.elapsedNanos,
                "runtime" to it.output?.runtime,
                "renderedPromptSha256" to it.output?.renderedPromptSha256,
                "finishReason" to it.output?.finishReason?.name,
                "metrics" to it.output?.metrics,
            ) },
            "deviceBefore" to before, "deviceAfter" to after,
        )
        File(dir, "result.json").writeText(gson.toJson(report) + "\n")
        return report
    }

    private suspend fun finishAndClose(
        native: Gemma4Provider,
        session: GenUiSession,
        label: String,
        constructionMs: Double,
    ): Map<String, Any?> {
        val finishStart = now()
        var metrics: GenUiGenerationSessionMetrics? = null
        var finishError: String? = null
        try {
            metrics = native.finishGenerationMetrics()
        } catch (error: Exception) {
            finishError = error.toString()
        }
        val finishMs = msSince(finishStart)
        val closeStart = now()
        var closeError: String? = null
        try {
            session.closeAndAwait()
        } catch (error: Exception) {
            closeError = error.toString()
        }
        return mapOf("label" to label, "providerConstructionMs" to constructionMs,
            "finishMetricsMs" to finishMs, "finishMetricsError" to finishError,
            "sessionMetrics" to metrics, "closeAndAwaitMs" to msSince(closeStart),
            "closeError" to closeError)
    }

    /** Delay is always outside conversion timing; a thermal timeout is recorded and non-fatal. */
    private suspend fun cooldown(
        context: Context,
        minimumDelayMs: Long,
        thermalTimeoutMs: Long,
    ): Map<String, Any?> {
        val manager = if (Build.VERSION.SDK_INT >= 29)
            context.getSystemService(PowerManager::class.java) else null
        val started = now()
        val before = manager?.currentThermalStatus
        if (minimumDelayMs > 0L) delay(minimumDelayMs)
        val thermalStarted = now()
        var status = manager?.currentThermalStatus
        var samples = 1
        while (status != null && status != PowerManager.THERMAL_STATUS_NONE &&
            msSince(thermalStarted) < thermalTimeoutMs
        ) {
            delay(1_000L)
            status = manager?.currentThermalStatus
            samples++
        }
        return mapOf(
            "minimumDelayMs" to minimumDelayMs,
            "thermalTimeoutMs" to thermalTimeoutMs,
            "elapsedMs" to msSince(started),
            "thermalWaitMs" to msSince(thermalStarted),
            "thermalBefore" to before,
            "thermalAfter" to status,
            "thermalReady" to (status == PowerManager.THERMAL_STATUS_NONE),
            "timedOut" to (status != null && status != PowerManager.THERMAL_STATUS_NONE),
            "thermalStatusUnavailable" to (status == null),
            "thermalSamples" to samples,
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

    private fun requireArg(args: android.os.Bundle, key: String): String =
        args.getString(key)?.trim()?.takeIf(String::isNotEmpty) ?: error("Missing -e $key")

    private fun boolArg(args: android.os.Bundle, key: String, default: Boolean): Boolean =
        args.getString(key, default.toString())!!.toBooleanStrictOrNull()
            ?: error("$key must be true or false")

    private fun now(): Long = SystemClock.elapsedRealtimeNanos()
    private fun msSince(start: Long): Double = (now() - start) / 1_000_000.0
    private fun elapsedOrNull(timestamp: Long, start: Long): Double? =
        timestamp.takeIf { it >= 0L }?.let { (it - start) / 1_000_000.0 }
    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private companion object {
        const val CORPUS_ASSET = "genuicraft_bixby50.jsonl"
        const val CORPUS_SHA256 = "fc46aa381957bed206f9e0f53e28edbb21b5096ca093985ad184fda73faaea0a"
        const val DEFAULT_CASES = "BXP-001,BXP-003,BXP-031"
    }
}
