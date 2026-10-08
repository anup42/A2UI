package com.samsung.genuicraft

import android.content.Context
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.sdk.GenUiDocument
import com.samsung.genuicraft.sdk.GenUiRenderSnapshot
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Real trained-model and native-screen evidence. No canned provider or repaired fixture is used.
 * Arguments: modelPath (original trained package), precision=FP32|FP16_CORRECTED,
 * cases=BXP-001,BXP-003,BXP-004, outputDir=<new leaf>, mtp=true, caseTimeoutMs=360000.
 * Warmup captures the live native surface; measured pairs capture only after generation so a
 * screenshot readback cannot distort the streaming-on/off inference comparison.
 */
@RunWith(AndroidJUnit4::class)
class SdkProgressiveRenderingOnDeviceTest {
    @Test
    fun compareNativeProgressiveRenderingOnAndOff() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val device = UiDevice.getInstance(instrumentation)
        val args = InstrumentationRegistry.getArguments()
        val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val metricsPresent = preferences.contains(TOKEN_METRICS)
        val originalMetrics = preferences.getBoolean(TOKEN_METRICS, true)
        val precision = args.getString("precision")?.let {
            InferenceBackendSettings.TrainedE2bGpuPrecision.valueOf(it.uppercase())
        } ?: InferenceBackendSettings.getTrainedE2bGpuPrecision(context)
        val originalPath = args.getString("modelPath") ?: preferences.getString(
            PREFERENCE_TRAINED_E2B_W4_MODEL_PATH,
            OnDeviceModelCatalog.entries.first { it.usesTrainedSdkConverter }.localFile(context).absolutePath,
        ).orEmpty()
        val selection = selectTrainedE2bModel(originalPath, precision)
        assertTrue(selection.readiness.message, selection.readiness.usable)
        val mtp = args.getString("mtp", "true")!!.toBooleanStrict()
        require(!mtp || selection.mtpSupported) { "The selected package does not support MTP" }
        val timeoutMs = args.getString("caseTimeoutMs", "360000")!!.toLong().also {
            require(it in 1_000L..3_600_000L)
        }
        val corpusBytes = context.assets.open(CORPUS_ASSET).use { it.readBytes() }
        val corpus = corpusBytes.toString(Charsets.UTF_8).lineSequence().filter(String::isNotBlank)
            .map { JsonParser.parseString(it).asJsonObject }.associateBy { it.get("id").asString }
        val ids = args.getString("cases", DEFAULT_CASES)!!.split(',').map(String::trim)
        require(ids.isNotEmpty() && ids.size == ids.distinct().size && ids.all(corpus::containsKey))
        val leaf = args.getString("outputDir", "run_${System.currentTimeMillis()}")!!
        require(Regex("[A-Za-z0-9_-]{1,80}").matches(leaf)) { "outputDir must be a simple new leaf" }
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "sdk_progressive_rendering/$leaf")
        require(!root.exists() && root.mkdirs()) { "Output directory must be new: ${root.absolutePath}" }
        val gson = GsonBuilder().disableHtmlEscaping().serializeNulls().setPrettyPrinting().create()
        fun write(target: File, value: Any) = target.writeText(gson.toJson(value) + "\n")
        write(File(root, "run_config.json"), linkedMapOf(
            "kind" to "native_progressive_rendering_comparison", "startedAtEpochMs" to System.currentTimeMillis(),
            "modelPath" to selection.modelPath, "modelBytes" to File(selection.modelPath).length(),
            "precision" to selection.gpuPrecision.name, "mtpRequested" to mtp,
            "cases" to ids, "caseTimeoutMs" to timeoutMs, "corpusSha256" to sha256(corpusBytes),
            "measuredPairCapturesDuringGeneration" to false,
            "sessionWallSamplingIntervalMs" to 40L,
            "device" to mapOf("manufacturer" to Build.MANUFACTURER, "model" to Build.MODEL,
                "sdkInt" to Build.VERSION.SDK_INT, "fingerprint" to Build.FINGERPRINT),
        ))
        val reports = mutableListOf<Map<String, Any?>>()
        var warmupReport: Map<String, Any?>? = null
        try {
            check(preferences.edit().putBoolean(TOKEN_METRICS, true).commit())
            ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
                lateinit var screen: GenUiSdkDemoViewModel
                scenario.onActivity { screen = ViewModelProvider(it)[GenUiSdkDemoViewModel::class.java] }
                val warmup = runCase(
                    scenario, screen, device, context.packageName, corpus.getValue(ids.first()),
                    selection, mtp, true, timeoutMs, File(root, "warmup"), captureLive = true,
                )
                write(File(root, "warmup_result.json"), warmup)
                warmupReport = warmup
                ids.forEachIndexed { index, id ->
                    // Alternate ordering to avoid assigning every first measured run to one arm.
                    val arms = if (index % 2 == 0) listOf(true, false) else listOf(false, true)
                    arms.forEach { streaming ->
                        val name = "${id}_${if (streaming) "on" else "off"}"
                        val report = runCase(
                            scenario, screen, device, context.packageName, corpus.getValue(id),
                            selection, mtp, streaming, timeoutMs, File(root, name), captureLive = false,
                        )
                        reports += report
                        write(File(root, "results.json"), reports)
                        instrumentation.sendStatus(0, Bundle().apply {
                            putString("stream", "$name ${report["phase"]} · first UI ${report["firstPreviewFrameElapsedMs"]} ms\n")
                        })
                    }
                }
            }
            val enabled = reports.filter { it["streamingEnabled"] == true }
            write(File(root, "summary.json"), mapOf(
                "completedPairs" to ids.size, "reports" to reports.size,
                "successfulConversions" to reports.count { it["phase"] == "COMPLETE" },
                "streamingCasesWithPreFinalNativeUi" to enabled.count {
                    (it["previewSnapshotCount"] as Int) > 0 && it["firstPreviewFrameElapsedMs"] != null
                },
                "warmup" to warmupReport,
            ))
            assertTrue("No streaming case displayed native UI before completion; inspect $root", enabled.any {
                (it["previewSnapshotCount"] as Int) > 0 && it["firstPreviewFrameElapsedMs"] != null
            })
            assertTrue("No successful final conversions; inspect $root", reports.any { it["phase"] == "COMPLETE" })
        } finally {
            val editor = preferences.edit().remove(TOKEN_METRICS)
            if (metricsPresent) editor.putBoolean(TOKEN_METRICS, originalMetrics)
            check(editor.commit()) { "Could not restore performance-metrics preference" }
        }
    }

    private fun runCase(
        scenario: ActivityScenario<GenUiSdkDemoActivity>,
        screen: GenUiSdkDemoViewModel,
        device: UiDevice,
        packageName: String,
        fixture: JsonObject,
        selection: TrainedE2bModelSelection,
        mtp: Boolean,
        streaming: Boolean,
        timeoutMs: Long,
        dir: File,
        captureLive: Boolean,
    ): Map<String, Any?> {
        check(dir.mkdirs())
        File(dir, "source.json").writeText(fixture.toString() + "\n")
        val startedAt = SystemClock.elapsedRealtime()
        scenario.onActivity {
            screen.generate(
                source = fixture.get("text").asString,
                e2bModelChoiceForRun = E2bModelChoice.TRAINED_E2B_V10_W4,
                modelPathForRun = selection.modelPath,
                metricsForRun = true,
                mtpForRun = mtp,
                gpuPrecisionForRun = selection.gpuPrecision,
                streamingRenderingForRun = streaming,
            )
        }
        val observed = mutableListOf<Map<String, Any?>>()
        var lastRevision: Pair<String, Long>? = null
        var liveCaptured = false
        var terminal: Observation? = null
        while (SystemClock.elapsedRealtime() - startedAt < timeoutMs) {
            val state = observe(scenario)
            state.snapshot?.let { snapshot ->
                val key = snapshot.surfaceKey to snapshot.revision
                if (key != lastRevision) {
                    lastRevision = key
                    observed += linkedMapOf(
                        "surfaceKey" to snapshot.surfaceKey, "attempt" to snapshot.attempt,
                        "revision" to snapshot.revision, "elapsedMs" to snapshot.elapsedMs,
                        "readyComponentCount" to snapshot.readyComponentCount, "isFinal" to snapshot.isFinal,
                        "uiObservedWhileWorking" to state.working,
                        "expressSha256" to sha256(snapshot.document.express.toByteArray()),
                    )
                }
                if (captureLive && !snapshot.isFinal && state.working &&
                    state.trace.firstPreviewFrameElapsedMs != null && !liveCaptured
                ) {
                    assertForeground(device, packageName)
                    assertTrue("Live native screenshot failed", device.takeScreenshot(File(dir, "live_native.png")))
                    device.dumpWindowHierarchy(File(dir, "live_native.xml"))
                    File(dir, "live_preview.express").writeText(snapshot.document.express)
                    liveCaptured = true
                }
            }
            if (!state.working) {
                terminal = state
                break
            }
            Thread.sleep(40L)
        }
        if (terminal == null) {
            scenario.onActivity { screen.cancelGeneration() }
            error("${fixture.get("id").asString} timed out; artifacts: $dir")
        }
        val final = requireNotNull(terminal)
        val sessionWallElapsedMs = SystemClock.elapsedRealtime() - startedAt
        val trace = final.trace
        if (!streaming) {
            assertEquals("Disabled arm emitted native previews", 0, trace.previewSnapshotCount)
            assertTrue("Disabled arm retained a rendering snapshot", final.snapshot == null)
        }
        trace.attempts.forEach { attempt ->
            File(dir, "attempt_${attempt.number}_raw.express").writeText(attempt.rawText)
        }
        final.document?.let { document ->
            assertEquals(document.express, trace.finalIr)
            File(dir, "final.express").writeText(document.express)
            File(dir, "final.a2ui.json").writeText(document.a2uiJson)
        }
        Thread.sleep(300L)
        assertForeground(device, packageName)
        assertTrue("Final native screenshot failed", device.takeScreenshot(File(dir, "final_native.png")))
        device.dumpWindowHierarchy(File(dir, "final_native.xml"))
        return linkedMapOf(
            "fixtureId" to fixture.get("id").asString, "streamingEnabled" to streaming,
            "phase" to trace.phase.name, "error" to trace.error, "status" to final.status,
            "runtime" to final.runtime, "repairKind" to trace.repairKind?.name,
            "conversionElapsedMs" to final.metrics?.conversionElapsedMs,
            "sessionWallElapsedMs" to sessionWallElapsedMs,
            "firstPreviewElapsedMs" to trace.firstPreviewElapsedMs,
            "firstPreviewFrameElapsedMs" to trace.firstPreviewFrameElapsedMs,
            "previewSnapshotCount" to trace.previewSnapshotCount,
            "finalReadyComponentCount" to trace.latestReadyComponentCount,
            "observedNativeRevisions" to observed, "liveScreenshotCaptured" to liveCaptured,
            "metrics" to metricsReport(final.metrics),
            "finalExpressSha256" to final.document?.express?.toByteArray()?.let(::sha256),
            "finalSnapshotMatchesDocument" to final.snapshot?.let {
                it.isFinal && it.document.a2uiJson == final.document?.a2uiJson
            },
        )
    }

    private fun observe(scenario: ActivityScenario<GenUiSdkDemoActivity>): Observation {
        val result = AtomicReference<Observation>()
        scenario.onActivity { activity ->
            val screen = ViewModelProvider(activity)[GenUiSdkDemoViewModel::class.java]
            result.set(Observation(
                screen.working, screen.generationTrace, activity.renderSnapshotForTest(),
                activity.renderedDocumentForTest(), activity.generationMetricsForTest(),
                activity.statusForTest(), activity.runtimeForTest(),
            ))
        }
        return requireNotNull(result.get())
    }

    private fun metricsReport(metrics: GenerationMetricsUiState?): Map<String, Any?>? = metrics?.let {
        linkedMapOf(
            "reportedAttempts" to it.reportedAttempts, "inputTokens" to it.totalInputTokens,
            "outputTokens" to it.totalOutputTokens,
            "nativeDecodeTokensPerSecond" to it.weightedNativeDecodeTokensPerSecond,
            "requestAverageTokensPerSecond" to it.requestAverageTokensPerSecond,
            "providerCallElapsedNanos" to it.totalProviderCallElapsedNanos,
            "nativeEngineInitializationNanos" to it.totalEngineInitializationNanos,
            "nativePrefillElapsedNanos" to it.totalPrefillElapsedNanos,
            "nativeDecodeElapsedNanos" to it.totalDecodeElapsedNanos,
            "validationAndRecoveryElapsedNanos" to it.validationAndRecoveryElapsedNanos,
            "mtpEnabled" to it.speculativeDecodingEnabled, "mtpAcceptanceRate" to it.drafterAcceptanceRate,
            "attempts" to it.attempts,
        )
    }

    private fun assertForeground(device: UiDevice, packageName: String) =
        assertEquals("Another app is foreground; refusing to capture it", packageName, device.currentPackageName)

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }

    private data class Observation(
        val working: Boolean, val trace: SdkGenerationTrace, val snapshot: GenUiRenderSnapshot?,
        val document: GenUiDocument?, val metrics: GenerationMetricsUiState?, val status: String,
        val runtime: String?,
    )

    private companion object {
        const val PREFERENCES = "genuicraft_sdk_demo"
        const val TOKEN_METRICS = "token_metrics_enabled"
        const val CORPUS_ASSET = "genuicraft_bixby50.jsonl"
        const val DEFAULT_CASES = "BXP-001,BXP-003,BXP-004"
    }
}
