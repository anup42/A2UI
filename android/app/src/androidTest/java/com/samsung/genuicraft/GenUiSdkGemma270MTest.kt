package com.samsung.genuicraft

import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiDocument
import java.io.File
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * One real native SDK demo smoke. Raw output and metrics survive a failed conversion or test.
 * No Espresso is used; detailed panel interactions were verified manually on the device.
 */
@RunWith(AndroidJUnit4::class)
class GenUiSdkGemma270MTest {
    @Test
    fun bxp001CpuModelStreamsIntoSdkDemoWithRawOutputAndMetrics() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val device = UiDevice.getInstance(instrumentation)
        val appFiles = requireNotNull(context.getExternalFilesDir(null))
        val model = File(appFiles, MODEL_RELATIVE_PATH).canonicalFile
        val case = context.assets.open(BIXBY_ASSET).bufferedReader().use { reader ->
            JsonParser.parseString(requireNotNull(reader.readLine())).asJsonObject
        }
        assertEquals("This test must use the frozen first Bixby50 case", "BXP-001", case.get("id").asString)
        val source = case.get("text").asString
        val query = case.get("query").asString
        val runDir = File(appFiles, "result/sdk_gemma270/bxp001_${System.currentTimeMillis()}_${UUID.randomUUID().toString().take(8)}")
        check(runDir.mkdirs()) { "Cannot create ${runDir.absolutePath}" }
        File(runDir, "source.md").writeText(source)
        File(runDir, "query.txt").writeText(query)

        val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val saved = PREFERENCE_KEYS.associateWith { key ->
            SavedPreference(preferences.contains(key), preferences.all[key])
        }
        val streamSamples = mutableListOf<Map<String, Any>>()
        val captures = linkedMapOf<String, Boolean>()
        val startedAt = System.currentTimeMillis()
        var latest: Observation? = null
        var failure: Throwable? = null
        var terminalAt: Long? = null
        try {
            assertTrue("Missing Gemma 3 270M package: ${model.absolutePath}", model.isFile)
            assertTrue("Unreadable Gemma 3 270M package: ${model.absolutePath}", model.canRead())
            assertTrue("Incomplete Gemma 3 270M package: ${model.length()} bytes", gemma270ModelReadiness(model.absolutePath).usable)
            check(preferences.edit()
                .putBoolean(USE_GEMMA, true)
                .putString(PREFERENCE_E2B_MODEL_CHOICE, E2bModelChoice.TRAINED_GEMMA3_270M.preferenceValue)
                .putString(PREFERENCE_GEMMA270_MODEL_PATH, model.absolutePath)
                .putString(PREFERENCE_GEMMA270_ACCELERATOR, "CPU")
                // A stale E2B preference must not enable MTP for the 270M package.
                .putBoolean(PREFERENCE_E2B_MTP_ENABLED, true)
                .putBoolean(TOKEN_METRICS, true)
                .putBoolean(PREFERENCE_STREAMING_UI_ENABLED, true)
                .putString(PREFERENCE_STREAMING_VIEW_MODE, SdkStreamingViewMode.TEXT.name)
                .commit()) { "Could not configure the SDK demo" }

            assertEquals(E2bModelChoice.TRAINED_GEMMA3_270M,
                E2bModelChoice.fromPreference(preferences.getString(PREFERENCE_E2B_MODEL_CHOICE, null)))
            assertFalse(E2bModelChoice.TRAINED_GEMMA3_270M.supportsMtp)

            val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            }
            ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
                assertTrue("BXP-001 was not selected", device.wait(
                    Until.hasObject(By.pkg(context.packageName).textStartsWith("BXP-001")), UI_TIMEOUT))
                val generate = requireNotNull(findByTextScrolling(device, context.packageName,
                    "Generate UI")) { "Generate UI was not reachable in the SDK input editor" }
                assertEquals("Another app is foreground before generation", context.packageName,
                    device.currentPackageName)
                clickableAncestor(generate).click()

                var previousPartial = ""
                var liveCaptured = false
                val deadline = SystemClock.elapsedRealtime() + RUN_TIMEOUT
                while (SystemClock.elapsedRealtime() < deadline) {
                    val observed = observe(scenario)
                    latest = observed
                    val first = observed.trace.attempts.firstOrNull { it.number == 1 }
                    val raw = first?.rawText.orEmpty()
                    if (raw.isNotBlank() && raw != previousPartial) {
                        if (first?.complete == false && previousPartial.isNotEmpty()) {
                            assertTrue("SDK stream stopped being cumulative", raw.startsWith(previousPartial))
                        }
                        previousPartial = raw
                        File(runDir, "raw_latest.express").writeText(raw)
                        streamSamples += linkedMapOf(
                            "atEpochMs" to System.currentTimeMillis(),
                            "attempt" to 1,
                            "chars" to raw.length,
                            "sha256" to sha256(raw),
                            "complete" to (first?.complete == true),
                        )
                        if (!liveCaptured && first?.complete == false) {
                            captures["live_ir.png"] = capture(device, context.packageName, File(runDir, "live_ir.png"))
                            liveCaptured = true
                        }
                    }
                    if (observed.trace.phase in TERMINAL_PHASES) {
                        terminalAt = System.currentTimeMillis()
                        break
                    }
                    Thread.sleep(POLL_MS)
                }

                val final = requireNotNull(latest) { "No SDK state was observed" }
                val firstRaw = final.trace.attempts.firstOrNull { it.number == 1 }?.rawText.orEmpty()
                assertTrue("Native output is empty; see ${runDir.absolutePath}", firstRaw.isNotBlank())
                assertTrue("No native streaming update was observed", streamSamples.isNotEmpty())
                assertEquals("Expected the Gemma 3 270M SDK selection", "Gemma 3 270M", final.modelLabel)
                assertTrue("Wrong runtime: ${final.runtime}",
                    final.runtime.orEmpty().contains("LiteRT-LM/Gemma3-270M/CPU"))
                assertFalse("270M runtime unexpectedly used MTP: ${final.runtime}",
                    final.runtime.orEmpty().contains("MTP", ignoreCase = true))

                assertEquals("Conversion did not finish; raw output is in ${runDir.absolutePath}",
                    SdkGenerationPhase.COMPLETE, final.trace.phase)
                assertTrue("SDK trace error: ${final.trace.error}", final.trace.error.isNullOrBlank())
                assertEquals("trained_gemma3_270m", final.document?.profile)
                assertEquals("Final IR and rendered document differ", final.trace.finalIr, final.document?.express)
                assertTrue("No renderable A2UI document", final.document?.a2uiJson.orEmpty().isNotBlank())
                assertFalse("Source fallback must stay disabled", final.trace.repairKind?.name == "SOURCE_TEXT_FALLBACK")
                val repeatedTrainingExample = final.document?.express.orEmpty().contains(
                    "Travel Checklist", ignoreCase = true) &&
                    !final.document?.express.orEmpty().contains("Bengaluru", ignoreCase = true)
                if (repeatedTrainingExample) assertTrue("Repeated example was not flagged as a source mismatch",
                    final.trace.warnings.any { it.startsWith("Source fidelity warning:") })

                // A semantically wrong result must be flagged, without substituting source text.
                if (final.trace.warnings.any { it.startsWith("Source fidelity warning:") }) {
                    assertTrue("Source mismatch status is hidden: ${final.status}",
                        final.status.contains("source mismatch", ignoreCase = true))
                }

                // These counters come from the native provider, not the app's wall clock.
                val metrics = requireNotNull(final.metrics) { "SDK session metrics were not shown" }
                assertTrue("No measured provider call", metrics.attempts.any {
                    (it.providerCallElapsedNanos ?: 0L) > 0L
                })
                assertTrue("Native output token counter is missing or zero",
                    (metrics.totalOutputTokens ?: 0L) > 0L)
                assertTrue("Native decode rate is missing or zero",
                    (metrics.weightedNativeDecodeTokensPerSecond ?: 0.0) > 0.0)
                Thread.sleep(PREVIEW_SETTLE_MS)
                captures["preview.png"] = capture(device, context.packageName, File(runDir, "preview.png"))
                runCatching { device.dumpWindowHierarchy(File(runDir, "preview.xml")) }
                assertTrue("Could not capture settled SDK preview", captures["preview.png"] == true)

                // The Activity can be recreated while the ViewModel retains its result.
                scenario.recreate()
                val retained = observe(scenario)
                latest = retained
                assertEquals("Raw output disappeared after recreation", firstRaw,
                    retained.trace.attempts.firstOrNull { it.number == 1 }?.rawText)
                assertEquals("Rendered document disappeared after recreation",
                    final.document?.express, retained.document?.express)
                assertEquals("Metrics disappeared after recreation", metrics.reportedAttempts,
                    retained.metrics?.reportedAttempts)
                captures["after_recreation.png"] = capture(device, context.packageName,
                    File(runDir, "after_recreation.png"))
                assertTrue("Could not capture recreated UI", captures["after_recreation.png"] == true)
            }
        } catch (caught: Throwable) {
            failure = caught
            throw caught
        } finally {
            latest?.trace?.attempts?.forEach { attempt ->
                runCatching { File(runDir, "raw_attempt_${attempt.number}.express").writeText(attempt.rawText) }
            }
            latest?.metrics?.let { runCatching { File(runDir, "metrics.json").writeText(GSON.toJson(it)) } }
            latest?.trace?.finalIr?.let { runCatching { File(runDir, "final.express").writeText(it) } }
            latest?.document?.a2uiJson?.let { runCatching { File(runDir, "final.a2ui.json").writeText(it) } }
            runCatching { File(runDir, "runtime_logcat.txt").writeText(
                device.executeShellCommand("logcat -d -t 500 -s GenUICraftRuntime:I GenUiSdkGemma270MTest:I")) }
            val summary = linkedMapOf<String, Any?>(
                "kind" to "native_sdk_demo_inference", "caseId" to "BXP-001",
                "startedAtEpochMs" to startedAt, "terminalAtEpochMs" to terminalAt,
                "completedAtEpochMs" to System.currentTimeMillis(), "deviceModel" to Build.MODEL,
                "deviceFingerprint" to Build.FINGERPRINT, "packageName" to context.packageName,
                "modelPath" to model.absolutePath, "modelBytes" to model.takeIf { it.isFile }?.length(),
                "sourceSha256" to sha256(source), "modelChoice" to E2bModelChoice.TRAINED_GEMMA3_270M.preferenceValue,
                "profileExpected" to "trained_gemma3_270m", "acceleratorRequested" to "CPU",
                "mtpPreferenceWasTrue" to true, "mtpExpectedFor270M" to false,
                "phase" to latest?.trace?.phase?.name, "status" to latest?.status,
                "runtime" to latest?.runtime, "modelLabel" to latest?.modelLabel,
                "repairKind" to latest?.trace?.repairKind?.name, "error" to latest?.trace?.error,
                "warnings" to latest?.trace?.warnings, "documentProfile" to latest?.document?.profile,
                "streamSamples" to streamSamples, "captures" to captures,
                "metrics" to latest?.metrics,
                "nativeOutputTokens" to latest?.metrics?.totalOutputTokens,
                "nativeDecodeTokensPerSecond" to latest?.metrics?.weightedNativeDecodeTokensPerSecond,
                "providerCallElapsedNanos" to latest?.metrics?.totalProviderCallElapsedNanos,
                "mtpActuallyEnabled" to latest?.metrics?.speculativeDecodingEnabled,
                "failure" to failure?.let { Log.getStackTraceString(it) },
            )
            runCatching { File(runDir, "run.json").writeText(GSON.toJson(summary)) }
            restorePreferences(preferences, saved)
            Log.i(TAG, "BXP-001 Gemma 270M SDK artifacts: ${runDir.absolutePath}; phase=${latest?.trace?.phase}")
        }
    }

    private fun observe(scenario: ActivityScenario<GenUiSdkDemoActivity>): Observation {
        val observed = AtomicReference<Observation>()
        scenario.onActivity { activity ->
            val model = ViewModelProvider(activity)[GenUiSdkDemoViewModel::class.java]
            observed.set(Observation(activity.generationTraceForTest(), activity.renderedDocumentForTest(),
                activity.generationMetricsForTest(), activity.statusForTest(), activity.runtimeForTest(),
                model.lastModel))
        }
        return requireNotNull(observed.get())
    }

    private fun clickableAncestor(node: UiObject2): UiObject2 {
        var current: UiObject2? = node
        while (current != null && !current.isClickable) current = current.parent
        return requireNotNull(current) { "No clickable ancestor for ${node.text}" }
    }

    private fun findByTextScrolling(device: UiDevice, packageName: String, label: String): UiObject2? {
        val selector = By.pkg(packageName).text(label)
        repeat(6) {
            device.findObject(selector)?.let { return it }
            device.swipe(device.displayWidth * 9 / 10, device.displayHeight * 8 / 10,
                device.displayWidth * 9 / 10, device.displayHeight * 3 / 10, 30)
            Thread.sleep(150)
        }
        return device.findObject(selector)
    }

    private fun capture(device: UiDevice, packageName: String, file: File): Boolean =
        device.currentPackageName == packageName && device.takeScreenshot(file)

    private fun restorePreferences(prefs: SharedPreferences, saved: Map<String, SavedPreference>) {
        val editor = prefs.edit()
        saved.forEach { (key, item) ->
            editor.remove(key)
            if (item.present) when (val value = item.value) {
                is Boolean -> editor.putBoolean(key, value)
                is String -> editor.putString(key, value)
            }
        }
        check(editor.commit()) { "Could not restore SDK demo preferences" }
    }

    private fun sha256(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.toByteArray()).joinToString("") { "%02x".format(it) }

    private data class SavedPreference(val present: Boolean, val value: Any?)
    private data class Observation(
        val trace: SdkGenerationTrace,
        val document: GenUiDocument?,
        val metrics: GenerationMetricsUiState?,
        val status: String,
        val runtime: String?,
        val modelLabel: String?,
    )

    private companion object {
        const val TAG = "GenUiSdkGemma270MTest"
        const val PREFERENCES = "genuicraft_sdk_demo"
        const val BIXBY_ASSET = "genuicraft_bixby50.jsonl"
        const val MODEL_RELATIVE_PATH = "on_device_models/gemma-3-270m-a2ui-express-int8.litertlm"
        const val USE_GEMMA = "use_gemma_provider"
        const val TOKEN_METRICS = "token_metrics_enabled"
        const val UI_TIMEOUT = 10_000L
        const val RUN_TIMEOUT = 120_000L
        const val PREVIEW_SETTLE_MS = 1_000L
        const val POLL_MS = 75L
        val TERMINAL_PHASES = setOf(SdkGenerationPhase.COMPLETE, SdkGenerationPhase.FAILED,
            SdkGenerationPhase.CANCELLED)
        val PREFERENCE_KEYS = listOf(USE_GEMMA, PREFERENCE_E2B_MODEL_CHOICE,
            PREFERENCE_GEMMA270_MODEL_PATH, PREFERENCE_GEMMA270_ACCELERATOR,
            PREFERENCE_E2B_MTP_ENABLED, TOKEN_METRICS, PREFERENCE_STREAMING_UI_ENABLED,
            PREFERENCE_STREAMING_VIEW_MODE)
        val GSON = GsonBuilder().disableHtmlEscaping().serializeNulls().setPrettyPrinting().create()
    }
}
