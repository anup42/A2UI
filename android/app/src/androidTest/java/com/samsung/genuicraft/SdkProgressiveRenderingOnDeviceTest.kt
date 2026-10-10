package com.samsung.genuicraft

import android.content.Context
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
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
    /**
     * One real trained-model inference per requested case, without a warmup or an off arm.
     * modelPath names the original package; corrected FP16 selection resolves its prepared
     * sibling. Example arguments: modelPath=<original>, cases=BXP-001,BXP-003,
     * outputDir=<new leaf>, caseTimeoutMs=360000.
     *
     * Screenshots and UI-tab changes occur during inference, so this is visual/timing evidence
     * for the observed run rather than a throughput comparison.
     */
    @Test
    fun captureNativeStreamingEvidenceOnce() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val device = UiDevice.getInstance(instrumentation)
        val args = InstrumentationRegistry.getArguments()
        val originalPath = args.getString("modelPath")?.takeIf(String::isNotBlank)
            ?: error("Pass modelPath for the original trained E2B package")
        args.getString("precision")?.let {
            require(it.equals("FP16_CORRECTED", ignoreCase = true)) {
                "This entrypoint requires precision=FP16_CORRECTED"
            }
        }
        args.getString("mtp")?.let {
            require(it.equals("true", ignoreCase = true)) { "This entrypoint requires mtp=true" }
        }
        val selection = selectTrainedE2bModel(
            originalPath, InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED,
        )
        assertTrue(selection.readiness.message, selection.readiness.usable)
        assertTrue("Corrected FP16 package has no corrected MTP drafter", selection.mtpSupported)
        val timeoutMs = args.getString("caseTimeoutMs", "360000")!!.toLong().also {
            require(it in 1_000L..3_600_000L)
        }
        val corpusBytes = context.assets.open(CORPUS_ASSET).use { it.readBytes() }
        val corpus = corpusBytes.toString(Charsets.UTF_8).lineSequence().filter(String::isNotBlank)
            .map { JsonParser.parseString(it).asJsonObject }.associateBy { it.get("id").asString }
        val ids = args.getString("cases", "BXP-001")!!.split(',').map(String::trim)
        require(ids.size in 1..2 && ids.size == ids.distinct().size && ids.all(corpus::containsKey)) {
            "Pass one or two distinct Bixby50 IDs with cases (default BXP-001)"
        }
        val leaf = args.getString("outputDir", "stream_${System.currentTimeMillis()}")!!
        require(Regex("[A-Za-z0-9_-]{1,80}").matches(leaf)) { "outputDir must be a simple new leaf" }
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "sdk_progressive_rendering/$leaf")
        require(!root.exists() && root.mkdirs()) { "Output directory must be new: ${root.absolutePath}" }
        val gson = GsonBuilder().disableHtmlEscaping().serializeNulls().setPrettyPrinting().create()
        fun write(target: File, value: Any) = target.writeText(gson.toJson(value) + "\n")
        write(File(root, "run_config.json"), linkedMapOf(
            "kind" to "single_native_streaming_evidence", "startedAtEpochMs" to System.currentTimeMillis(),
            "originalModelPath" to File(originalPath).absolutePath,
            "selectedModelPath" to selection.modelPath,
            "selectedModelBytes" to File(selection.modelPath).length(),
            "precision" to selection.gpuPrecision.name, "mtpRequested" to true,
            "conversionProfile" to "TRAINED_E2B_V10_W4", "cases" to ids,
            "caseTimeoutMs" to timeoutMs, "corpusSha256" to sha256(corpusBytes),
            "pollIntervalMs" to 40L,
            "capturesDuringGeneration" to true,
            "timingNote" to "Live screenshots and tab changes can affect this run's conversion time",
            "hierarchyNote" to "Only the final window XML is dumped; live captures prioritize the model window",
            "device" to mapOf("manufacturer" to Build.MANUFACTURER, "model" to Build.MODEL,
                "sdkInt" to Build.VERSION.SDK_INT, "fingerprint" to Build.FINGERPRINT),
        ))
        val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val metricsPresent = preferences.contains(TOKEN_METRICS)
        val originalMetrics = preferences.getBoolean(TOKEN_METRICS, true)
        val modePresent = preferences.contains(STREAMING_VIEW_MODE_PREFERENCE)
        val originalMode = preferences.getString(STREAMING_VIEW_MODE_PREFERENCE, null)
        val reports = mutableListOf<Map<String, Any?>>()
        try {
            check(preferences.edit().putBoolean(TOKEN_METRICS, true)
                .putString(STREAMING_VIEW_MODE_PREFERENCE, "UI").commit())
            ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
                lateinit var screen: GenUiSdkDemoViewModel
                scenario.onActivity { screen = ViewModelProvider(it)[GenUiSdkDemoViewModel::class.java] }
                ids.forEach { id ->
                    val report = captureStreamingCase(
                        scenario, screen, device, context.packageName, corpus.getValue(id),
                        selection, timeoutMs, File(root, id), gson,
                    )
                    reports += report
                    write(File(root, "results.json"), reports)
                    instrumentation.sendStatus(0, Bundle().apply {
                        putString("stream", "$id ${report["phase"]} · first UI " +
                            "${report["firstPreviewFrameElapsedMs"]} / ${report["conversionElapsedMs"]} ms\n")
                    })
                }
            }
            write(File(root, "summary.json"), mapOf(
                "completedRuns" to reports.size,
                "successfulRuns" to reports.count { it["phase"] == "COMPLETE" },
                "liveUiScreenshots" to reports.count { it["firstUiScreenshotConfirmedLive"] == true },
                "liveTextScreenshots" to reports.count { it["textScreenshotConfirmedLive"] == true },
                "reports" to reports,
            ))
            assertTrue("No run displayed model-generated native UI before completion; inspect $root",
                reports.any { it["firstUiScreenshotConfirmedLive"] == true })
            assertTrue("No run showed a live generated-text frame; inspect $root",
                reports.any { it["textScreenshotConfirmedLive"] == true })
            assertTrue("A run did not complete; inspect $root", reports.all { it["phase"] == "COMPLETE" })
        } finally {
            val editor = preferences.edit().remove(TOKEN_METRICS).remove(STREAMING_VIEW_MODE_PREFERENCE)
            if (metricsPresent) editor.putBoolean(TOKEN_METRICS, originalMetrics)
            if (modePresent) editor.putString(STREAMING_VIEW_MODE_PREFERENCE, originalMode)
            check(editor.commit()) { "Could not restore SDK demo preferences" }
        }
    }

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

    private fun captureStreamingCase(
        scenario: ActivityScenario<GenUiSdkDemoActivity>,
        screen: GenUiSdkDemoViewModel,
        device: UiDevice,
        packageName: String,
        fixture: JsonObject,
        selection: TrainedE2bModelSelection,
        timeoutMs: Long,
        dir: File,
        gson: Gson,
    ): Map<String, Any?> {
        check(dir.mkdirs())
        File(dir, "source.json").writeText(fixture.toString() + "\n")
        val startedAtEpochMs = System.currentTimeMillis()
        val startedAt = SystemClock.elapsedRealtime()
        scenario.onActivity {
            screen.generate(
                source = fixture.get("text").asString,
                e2bModelChoiceForRun = E2bModelChoice.TRAINED_E2B_V10_W4,
                modelPathForRun = selection.modelPath,
                metricsForRun = true,
                mtpForRun = true,
                gpuPrecisionForRun = selection.gpuPrecision,
                streamingRenderingForRun = true,
            )
        }
        val rawObservations = mutableListOf<Map<String, Any?>>()
        val previewObservations = mutableListOf<Map<String, Any?>>()
        val lastRawByAttempt = mutableMapOf<Int, String>()
        var rawStreamCumulative = true
        var lastRevision: Pair<String, Long>? = null
        var lastPreviewShape: PreviewShape? = null
        var firstTextOrTitlePreviewElapsedMs: Long? = null
        var firstTableOrCardPreviewElapsedMs: Long? = null
        var firstStructuredRowsPreviewElapsedMs: Long? = null
        var firstUiScreenshot: Map<String, Any?>? = null
        var textScreenshot: Map<String, Any?>? = null
        var textCaptureAttempted = false
        var uiTabRestored = true
        var richUiScreenshot: Map<String, Any?>? = null
        var terminal: Observation? = null
        while (SystemClock.elapsedRealtime() - startedAt < timeoutMs) {
            val state = observe(scenario)
            state.trace.attempts.forEach { attempt ->
                val previous = lastRawByAttempt[attempt.number]
                if (attempt.rawText.isNotBlank() && attempt.rawText != previous) {
                    if (previous != null && !attempt.rawText.startsWith(previous)) {
                        rawStreamCumulative = false
                    }
                    lastRawByAttempt[attempt.number] = attempt.rawText
                    rawObservations += linkedMapOf(
                        "attempt" to attempt.number,
                        "observedAtEpochMs" to System.currentTimeMillis(),
                        "observedElapsedMs" to SystemClock.elapsedRealtime() - startedAt,
                        "working" to state.working,
                        "complete" to attempt.complete,
                        "chars" to attempt.rawText.length,
                        "sha256" to sha256(attempt.rawText.toByteArray()),
                        "rawCumulativeDsl" to attempt.rawText,
                    )
                }
            }
            state.snapshot?.let { snapshot ->
                val key = snapshot.surfaceKey to snapshot.revision
                val shape = if (key != lastRevision) {
                    previewShape(snapshot.document.a2uiJson).also { lastPreviewShape = it }
                } else requireNotNull(lastPreviewShape)
                if (key != lastRevision) {
                    lastRevision = key
                    val rawAtPreview = state.trace.attempts
                        .firstOrNull { it.number == snapshot.attempt }?.rawText.orEmpty()
                    previewObservations += linkedMapOf(
                        "surfaceKey" to snapshot.surfaceKey,
                        "attempt" to snapshot.attempt,
                        "revision" to snapshot.revision,
                        "sdkElapsedMs" to snapshot.elapsedMs,
                        "observedAtEpochMs" to System.currentTimeMillis(),
                        "observedElapsedMs" to SystemClock.elapsedRealtime() - startedAt,
                        "isFinal" to snapshot.isFinal,
                        "workingAtObservation" to state.working,
                        "readyComponentCount" to snapshot.readyComponentCount,
                        "componentKinds" to shape.componentKinds,
                        "textOrCardTitleCount" to shape.textOrCardTitleCount,
                        "tableCount" to shape.tableCount,
                        "cardCount" to shape.cardCount,
                        "structuredRowCount" to shape.structuredRowCount,
                        "rawCharsAtObservation" to rawAtPreview.length,
                        "rawSha256AtObservation" to sha256(rawAtPreview.toByteArray()),
                        "rawCumulativeAtObservation" to rawAtPreview,
                        "previewExpress" to snapshot.document.express,
                        "previewA2uiJson" to snapshot.document.a2uiJson,
                    )
                    if (!snapshot.isFinal) {
                        if (firstTextOrTitlePreviewElapsedMs == null &&
                            shape.textOrCardTitleCount > 0
                        ) firstTextOrTitlePreviewElapsedMs = snapshot.elapsedMs
                        if (firstTableOrCardPreviewElapsedMs == null &&
                            (shape.tableCount > 0 || shape.cardCount > 0)
                        ) firstTableOrCardPreviewElapsedMs = snapshot.elapsedMs
                        if (firstStructuredRowsPreviewElapsedMs == null &&
                            shape.structuredRowCount > 0 &&
                            (shape.tableCount > 0 || shape.cardCount > 0)
                        ) firstStructuredRowsPreviewElapsedMs = snapshot.elapsedMs
                    }
                }
                if (firstUiScreenshot == null && !snapshot.isFinal && state.working &&
                    snapshot.readyComponentCount > 0 && state.trace.firstPreviewFrameElapsedMs != null
                ) {
                    val frameStartedAt = SystemClock.elapsedRealtime()
                    assertForeground(device, packageName)
                    val captured = device.takeScreenshot(File(dir, "live_native.png"))
                    val frameEndedAt = SystemClock.elapsedRealtime()
                    val after = observe(scenario)
                    File(dir, "live_preview.express").writeText(snapshot.document.express)
                    File(dir, "live_preview.a2ui.json").writeText(snapshot.document.a2uiJson)
                    firstUiScreenshot = screenshotRecord(
                        captured, state, after, snapshot, frameStartedAt, frameEndedAt, startedAt,
                    )
                }
                val firstReady = firstUiScreenshot?.get("readyComponentCount") as? Int
                val firstRows = firstUiScreenshot?.get("structuredRowCount") as? Int
                if (firstUiScreenshot?.get("confirmedLive") == true &&
                    richUiScreenshot == null && !snapshot.isFinal && state.working &&
                    textCaptureAttempted && uiTabRestored &&
                    (snapshot.readyComponentCount > (firstReady ?: Int.MAX_VALUE) ||
                        shape.structuredRowCount > (firstRows ?: Int.MAX_VALUE))
                ) {
                    val frameStartedAt = SystemClock.elapsedRealtime()
                    assertForeground(device, packageName)
                    val captured = device.takeScreenshot(File(dir, "rich_live_native.png"))
                    val frameEndedAt = SystemClock.elapsedRealtime()
                    val after = observe(scenario)
                    File(dir, "rich_live_preview.express").writeText(snapshot.document.express)
                    File(dir, "rich_live_preview.a2ui.json").writeText(snapshot.document.a2uiJson)
                    richUiScreenshot = screenshotRecord(
                        captured, state, after, snapshot, frameStartedAt, frameEndedAt, startedAt,
                        shape.structuredRowCount,
                    )
                }
            }
            if (firstUiScreenshot?.get("confirmedLive") == true && !textCaptureAttempted &&
                state.working && state.trace.attempts.any { it.rawText.isNotBlank() }
            ) {
                textCaptureAttempted = true
                if (clickWorkspaceTab(device, packageName, "Inspect IR")) {
                    uiTabRestored = false
                    val title = device.wait(Until.findObject(
                        By.pkg(packageName).textStartsWith("Generated IR · live"),
                    ), UI_TAB_TIMEOUT_MS)
                    val before = observe(scenario)
                    val raw = before.trace.attempts.lastOrNull()?.rawText.orEmpty()
                    if (title != null && before.working && raw.isNotBlank()) {
                        val frameStartedAt = SystemClock.elapsedRealtime()
                        assertForeground(device, packageName)
                        val captured = device.takeScreenshot(File(dir, "text_streaming.png"))
                        val frameEndedAt = SystemClock.elapsedRealtime()
                        val after = observe(scenario)
                        File(dir, "text_frame_raw.express").writeText(raw)
                        textScreenshot = linkedMapOf(
                            "captured" to captured,
                            "confirmedLive" to (captured && before.working && after.working &&
                                after.trace.phase == SdkGenerationPhase.GENERATING),
                            "capturedAtElapsedMs" to frameStartedAt - startedAt,
                            "captureDurationMs" to frameEndedAt - frameStartedAt,
                            "workingBefore" to before.working,
                            "workingAfter" to after.working,
                            "phaseBefore" to before.trace.phase.name,
                            "phaseAfter" to after.trace.phase.name,
                            "rawCharsBefore" to raw.length,
                            "rawSha256Before" to sha256(raw.toByteArray()),
                            "rawCharsAfter" to after.trace.attempts.lastOrNull()?.rawText?.length,
                        )
                    }
                    uiTabRestored = clickWorkspaceTab(device, packageName, "Live preview")
                }
            }
            if (!state.working) {
                terminal = state
                break
            }
            Thread.sleep(40L)
        }
        File(dir, "raw_stream_observations.json").writeText(gson.toJson(rawObservations) + "\n")
        File(dir, "preview_snapshots.json").writeText(gson.toJson(previewObservations) + "\n")
        if (terminal == null) {
            scenario.onActivity { screen.cancelGeneration() }
            error("${fixture.get("id").asString} timed out; artifacts: $dir")
        }
        val final = requireNotNull(terminal)
        val trace = final.trace
        val sessionWallElapsedMs = SystemClock.elapsedRealtime() - startedAt
        trace.attempts.forEach { attempt ->
            File(dir, "attempt_${attempt.number}_raw.express").writeText(attempt.rawText)
        }
        final.document?.let { document ->
            File(dir, "final.express").writeText(document.express)
            File(dir, "final.a2ui.json").writeText(document.a2uiJson)
        }
        val finalPreviewTabClicked = clickWorkspaceTab(device, packageName, "Preview")
        Thread.sleep(300L)
        assertForeground(device, packageName)
        // Completion already selects Preview. A lookup racing the tab's live/final label
        // must not fail a run whose final native surface is visible. Raw code panels have
        // separate accessibility descriptions and are absent from the native workspace.
        val finalPreviewConfirmed = device.wait(Until.hasObject(
            By.pkg(packageName).desc("Preview rendered UI"),
        ), UI_TAB_TIMEOUT_MS) && !device.hasObject(
            By.pkg(packageName).descStartsWith("Generated IR"),
        ) && !device.hasObject(By.pkg(packageName).descStartsWith("Repaired IR"))
        val finalScreenshotCaptured = device.takeScreenshot(File(dir, "final_native.png"))
        if (finalScreenshotCaptured) device.dumpWindowHierarchy(File(dir, "final_native.xml"))
        val total = final.metrics?.conversionElapsedMs
        val firstUi = trace.firstPreviewFrameElapsedMs
        val report = linkedMapOf<String, Any?>(
            "fixtureId" to fixture.get("id").asString,
            "startedAtEpochMs" to startedAtEpochMs,
            "completedAtEpochMs" to System.currentTimeMillis(),
            "phase" to trace.phase.name, "error" to trace.error,
            "status" to final.status, "runtime" to final.runtime,
            "repairKind" to trace.repairKind?.name,
            "conversionElapsedMs" to total,
            "sessionWallElapsedMsBeforeFinalCapture" to sessionWallElapsedMs,
            "firstRawObservedElapsedMs" to rawObservations.firstOrNull()?.get("observedElapsedMs"),
            "firstPreviewElapsedMs" to trace.firstPreviewElapsedMs,
            "firstPreviewFrameElapsedMs" to firstUi,
            "firstTextOrTitlePreviewElapsedMs" to firstTextOrTitlePreviewElapsedMs,
            "firstTableOrCardPreviewElapsedMs" to firstTableOrCardPreviewElapsedMs,
            "firstStructuredRowsPreviewElapsedMs" to firstStructuredRowsPreviewElapsedMs,
            "timeAfterFirstUiMs" to if (firstUi != null && total != null) total - firstUi else null,
            "firstUiPercentOfTotal" to if (firstUi != null && total != null && total > 0L)
                firstUi.toDouble() * 100.0 / total else null,
            "rawObservationCount" to rawObservations.size,
            "rawStreamCumulative" to rawStreamCumulative,
            "previewSnapshotCount" to trace.previewSnapshotCount,
            "observedPreviewRevisions" to previewObservations.size,
            "firstUiScreenshot" to firstUiScreenshot,
            "firstUiScreenshotConfirmedLive" to (firstUiScreenshot?.get("confirmedLive") == true),
            "textScreenshot" to textScreenshot,
            "textScreenshotConfirmedLive" to (textScreenshot?.get("confirmedLive") == true),
            "uiTabRestoredAfterText" to uiTabRestored,
            "richUiScreenshot" to richUiScreenshot,
            "finalPreviewTabClicked" to finalPreviewTabClicked,
            "finalPreviewConfirmed" to finalPreviewConfirmed,
            "finalScreenshotCaptured" to finalScreenshotCaptured,
            "finalReadyComponentCount" to trace.latestReadyComponentCount,
            "metrics" to metricsReport(final.metrics),
            "finalExpressSha256" to final.document?.express?.toByteArray()?.let(::sha256),
            "finalSnapshotMatchesDocument" to final.snapshot?.let {
                it.isFinal && it.document.a2uiJson == final.document?.a2uiJson
            },
        )
        File(dir, "report.json").writeText(gson.toJson(report) + "\n")
        assertEquals("Real SDK conversion did not complete; inspect $dir", SdkGenerationPhase.COMPLETE,
            trace.phase)
        assertTrue("Generated DSL was empty; inspect $dir", trace.attempts.any { it.rawText.isNotBlank() })
        assertTrue("Raw SDK output stopped being cumulative; inspect $dir", rawStreamCumulative)
        assertTrue("Final result used source-text fallback; inspect $dir",
            trace.repairKind?.name != "SOURCE_TEXT_FALLBACK")
        assertTrue("Runtime did not execute GPU+MTP: ${final.runtime}",
            final.runtime.orEmpty().let { runtime ->
                runtime.contains("/GPU+", ignoreCase = true) &&
                    runtime.contains("+MTP", ignoreCase = true)
            })
        assertTrue("Native MTP metrics were not enabled; inspect $dir",
            final.metrics?.speculativeDecodingEnabled == true)
        assertTrue("No pre-final native preview reached the app; inspect $dir",
            trace.previewSnapshotCount > 0 && firstUi != null)
        assertTrue("No final model-generated document; inspect $dir", final.document != null)
        assertEquals("Trace and rendered document differ; inspect $dir",
            trace.finalIr, final.document?.express)
        assertTrue("Final workspace still shows IR instead of the native preview; inspect $dir", finalPreviewConfirmed)
        assertTrue("Final native screenshot failed; inspect $dir", finalScreenshotCaptured)
        return report
    }

    private fun clickWorkspaceTab(device: UiDevice, packageName: String, label: String): Boolean {
        val text = device.wait(Until.findObject(By.pkg(packageName).text(label)), UI_TAB_TIMEOUT_MS)
            ?: return false
        assertForeground(device, packageName)
        // Compose tabs on the Android 17 image expose non-clickable semantics nodes, but
        // their observed text bounds still receive the normal parent touch handler.
        text.click()
        return true
    }

    private fun screenshotRecord(
        captured: Boolean,
        before: Observation,
        after: Observation,
        snapshot: GenUiRenderSnapshot,
        captureStartedAt: Long,
        captureEndedAt: Long,
        runStartedAt: Long,
        structuredRowCount: Int = previewShape(snapshot.document.a2uiJson).structuredRowCount,
    ): Map<String, Any?> = linkedMapOf(
        "captured" to captured,
        "confirmedLive" to (captured && before.working && after.working &&
            !snapshot.isFinal && after.snapshot?.isFinal != true),
        "capturedAtElapsedMs" to captureStartedAt - runStartedAt,
        "captureDurationMs" to captureEndedAt - captureStartedAt,
        "workingBefore" to before.working,
        "workingAfter" to after.working,
        "phaseBefore" to before.trace.phase.name,
        "phaseAfter" to after.trace.phase.name,
        "attempt" to snapshot.attempt,
        "revisionBefore" to snapshot.revision,
        "revisionAfter" to after.snapshot?.revision,
        "readyComponentCount" to snapshot.readyComponentCount,
        "structuredRowCount" to structuredRowCount,
        "sdkSnapshotElapsedMs" to snapshot.elapsedMs,
        "frameElapsedMs" to before.trace.firstPreviewFrameElapsedMs,
    )

    private data class PreviewShape(
        val componentKinds: Map<String, Int>,
        val textOrCardTitleCount: Int,
        val tableCount: Int,
        val cardCount: Int,
        val structuredRowCount: Int,
    )

    private fun previewShape(a2uiJson: String): PreviewShape {
        val messages = JsonParser.parseString(a2uiJson).asJsonArray
        val components = messages.firstOrNull { it.asJsonObject.has("updateComponents") }
            ?.asJsonObject?.getAsJsonObject("updateComponents")?.getAsJsonArray("components")
        val kinds = linkedMapOf<String, Int>()
        var titledComponents = 0
        var inlineRows = 0
        components?.forEach { entry ->
            val component = entry.asJsonObject
            val kind = component.get("component")?.asString.orEmpty()
            kinds[kind] = (kinds[kind] ?: 0) + 1
            if (kind in setOf("Card", "Table", "Chart") &&
                component.get("title")?.takeIf { it.isJsonPrimitive }
                    ?.asString?.isNotBlank() == true) titledComponents++
            inlineRows += countNamedRows(component)
        }
        val stateRows = messages.firstOrNull { it.asJsonObject.has("updateDataModel") }
            ?.asJsonObject?.getAsJsonObject("updateDataModel")?.get("value")
            ?.let(::countNamedRows) ?: 0
        return PreviewShape(
            componentKinds = kinds,
            textOrCardTitleCount = (kinds["Text"] ?: 0) + (kinds["Heading"] ?: 0) + titledComponents,
            tableCount = kinds["Table"] ?: 0,
            cardCount = kinds["Card"] ?: 0,
            structuredRowCount = maxOf(inlineRows, stateRows),
        )
    }

    private fun countNamedRows(value: JsonElement): Int = when {
        value.isJsonObject -> value.asJsonObject.entrySet().sumOf { (name, child) ->
            if (name == "rows" && child.isJsonArray) child.asJsonArray.size()
            else countNamedRows(child)
        }
        value.isJsonArray -> value.asJsonArray.sumOf(::countNamedRows)
        else -> 0
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
        const val STREAMING_VIEW_MODE_PREFERENCE = "streaming_view_mode"
        const val CORPUS_ASSET = "genuicraft_bixby50.jsonl"
        const val DEFAULT_CASES = "BXP-001,BXP-003,BXP-004"
        const val UI_TAB_TIMEOUT_MS = 1_500L
    }
}
