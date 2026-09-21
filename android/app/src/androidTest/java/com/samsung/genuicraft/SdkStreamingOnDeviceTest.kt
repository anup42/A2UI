package com.samsung.genuicraft

import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import java.io.File
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SdkStreamingOnDeviceTest {

    @Test
    fun bxp001StreamsRepairsAndRendersThroughRealSdkScreen() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val device = UiDevice.getInstance(instrumentation)
        val model = File(
            requireNotNull(context.getExternalFilesDir(null)),
            "sdk_models/gemma4_e2b_a2ui_mobile.litertlm",
        ).canonicalFile
        assertTrue("Trained E2B model is missing: ${model.absolutePath}", model.isFile)
        assertTrue("Trained E2B model is unreadable: ${model.absolutePath}", model.canRead())
        assertTrue("Trained E2B model is incomplete: ${model.length()} bytes", model.length() >= MIN_MODEL_BYTES)

        val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val originalPreferences = PREFERENCE_KEYS.associateWith { key ->
            SavedPreference(preferences.contains(key), preferences.all[key])
        }
        val startedAt = System.currentTimeMillis()
        val runDir = File(
            context.filesDir,
            "$ARTIFACT_ROOT/${startedAt}_${UUID.randomUUID().toString().take(8)}",
        )
        check(runDir.mkdirs()) { "Could not create ${runDir.absolutePath}" }

        try {
            check(
                preferences.edit()
                    .putBoolean(USE_GEMMA, true)
                    .putString(MODEL_CHOICE, "trained_e2b_v10_w4")
                    .putString(TRAINED_MODEL_PATH, model.absolutePath)
                    .putBoolean(MTP_ENABLED, true)
                    .putBoolean(TOKEN_METRICS, true)
                    .commit()
            ) { "Could not configure the SDK demo" }

            val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            }
            ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
                instrumentation.waitForIdleSync()
                assertForeground(device, context.packageName, "start SDK conversion")
                assertTrue(
                    "SDK screen did not open on the first Bixby50 case",
                    device.wait(Until.hasObject(By.text("BXP-001")), UI_TIMEOUT),
                )
                clickText(device, context.packageName, "Convert + render")

                val samples = mutableListOf<Map<String, Any>>()
                var lastPartial = ""
                var rawAtNativeCompletion: String? = null
                var liveCapturedAt: Long? = null
                var terminal: Observation? = null
                val deadline = SystemClock.elapsedRealtime() + RUN_TIMEOUT
                while (SystemClock.elapsedRealtime() < deadline) {
                    val observed = observe(scenario)
                    if (observed.phase == "FAILED" || observed.phase == "CANCELLED") {
                        error("SDK generation ${observed.phase}: ${observed.error ?: observed.status}")
                    }
                    val raw = observed.firstAttemptRaw
                    if (!observed.firstAttemptComplete && raw.isNotBlank() && raw != lastPartial) {
                        assertTrue("Native SDK stream stopped being cumulative", raw.startsWith(lastPartial))
                        lastPartial = raw
                        samples += streamSample(raw)
                        if (samples.size == 2) {
                            waitForText(device, context.packageName, "Generated IR · live")
                            capture(device, context.packageName, File(runDir, LIVE_SCREENSHOT))
                            liveCapturedAt = System.currentTimeMillis()
                        }
                    }
                    if (observed.firstAttemptComplete && rawAtNativeCompletion == null) {
                        rawAtNativeCompletion = raw
                    }
                    if (observed.phase == "COMPLETE") {
                        terminal = observed
                        break
                    }
                    Thread.sleep(POLL_MS)
                }

                val final = requireNotNull(terminal) { "SDK screen did not complete within ${RUN_TIMEOUT / 1_000}s" }
                assertTrue("Expected at least two native partials before completion; got ${samples.size}", samples.size >= 2)
                val raw = requireNotNull(rawAtNativeCompletion)
                assertTrue("Completed native raw output is empty", raw.isNotBlank())
                assertEquals("Repair replaced the retained native output", raw, final.firstAttemptRaw)
                assertTrue("SDK trace reported an error: ${final.error}", final.error.isNullOrBlank())
                assertTrue("Unexpected terminal status: ${final.status}", !final.status.startsWith("Error") && !final.status.startsWith("Conversion failed"))
                assertTrue("Runtime did not execute GPU+MTP: ${final.runtime}", final.runtime.orEmpty().contains("GPU+MTP", true))
                assertTrue(
                    "Expected generated-output repair, got ${final.repairKind}",
                    final.repairKind == "STRUCTURAL" || final.repairKind == "GENERATED_DSL_REPAIR",
                )
                assertNotEquals("SOURCE_TEXT_FALLBACK", final.repairKind)
                val finalIr = requireNotNull(final.finalIr) { "Trace has no final IR" }
                val documentExpress = requireNotNull(final.documentExpress) { "Rendered document has no Express IR" }
                assertEquals("Trace final IR differs from rendered document", finalIr, documentExpress)
                assertNotEquals("Repaired IR equals unmodified native output", raw, finalIr)
                assertTrue("Rendered document Express IR is empty", documentExpress.isNotBlank())
                assertTrue("Rendered document A2UI JSON is empty", final.documentJson.orEmpty().isNotBlank())

                capture(device, context.packageName, File(runDir, GENERATED_SCREENSHOT))
                device.dumpWindowHierarchy(File(runDir, "final_ui.xml"))
                waitForText(device, context.packageName, "Generated IR")
                waitForText(device, context.packageName, "Repaired IR")
                clickText(device, context.packageName, "Preview")
                Thread.sleep(UI_SETTLE_MS)
                capture(device, context.packageName, File(runDir, PREVIEW_SCREENSHOT))

                File(runDir, "stream_proof.json").writeText(GSON.toJson(linkedMapOf(
                    "fixtureId" to "BXP-001", "modelPath" to model.absolutePath,
                    "gpuSelected" to true, "mtpSelected" to true, "tokenMetricsEnabled" to true,
                    "startedAtEpochMs" to startedAt, "liveCapturedAtEpochMs" to liveCapturedAt,
                    "completedAtEpochMs" to System.currentTimeMillis(), "phase" to final.phase,
                    "status" to final.status, "runtime" to final.runtime, "attemptCount" to final.attemptCount,
                    "nativePartialsBeforeCompletion" to samples, "nativePartialCount" to samples.size,
                    "rawRetainedAfterRepair" to (raw == final.firstAttemptRaw), "rawChars" to raw.length,
                    "rawUtf8Bytes" to raw.toByteArray().size, "rawSha256" to sha256(raw),
                    "repairKind" to final.repairKind, "finalChars" to finalIr.length,
                    "finalUtf8Bytes" to finalIr.toByteArray().size, "finalSha256" to sha256(finalIr),
                    "rawModelIr" to raw, "finalIr" to finalIr,
                    "documentA2uiJson" to final.documentJson, "screenshots" to listOf(
                        LIVE_SCREENSHOT, GENERATED_SCREENSHOT, PREVIEW_SCREENSHOT,
                    ),
                )))
            }
        } finally {
            restorePreferences(preferences, originalPreferences)
        }
    }

    private fun observe(scenario: ActivityScenario<GenUiSdkDemoActivity>): Observation {
        val value = AtomicReference<Observation>()
        scenario.onActivity { activity ->
            val trace = activity.generationTraceForTest()
            val attempt = trace.attempts.firstOrNull { it.number == 1 }
            val document = activity.renderedDocumentForTest()
            value.set(Observation(
                trace.phase.name, attempt?.rawText.orEmpty(), attempt?.complete == true,
                trace.attempts.size, trace.finalIr, trace.repairKind?.name, trace.error,
                activity.runtimeForTest(), document?.express, document?.a2uiJson,
                activity.statusForTest(),
            ))
        }
        return requireNotNull(value.get())
    }

    private fun clickText(device: UiDevice, packageName: String, text: String) {
        val label = device.wait(Until.findObject(By.pkg(packageName).text(text)), UI_TIMEOUT)
            ?: error("Timed out waiting for $text")
        val target = clickableAncestor(label) ?: error("$text has no clickable ancestor")
        assertForeground(device, packageName, "tap $text")
        target.click()
    }

    private fun clickableAncestor(node: UiObject2): UiObject2? {
        var current: UiObject2? = node
        while (current != null && !current.isClickable) current = current.parent
        return current
    }

    private fun waitForText(device: UiDevice, packageName: String, prefix: String) {
        requireNotNull(device.wait(Until.findObject(By.pkg(packageName).textStartsWith(prefix)), UI_TIMEOUT)) {
            "Timed out waiting for $prefix"
        }
    }

    private fun capture(device: UiDevice, packageName: String, target: File) {
        Thread.sleep(SCREENSHOT_SETTLE_MS)
        assertForeground(device, packageName, "capture ${target.name}")
        assertTrue("Screenshot failed: ${target.name}", device.takeScreenshot(target))
    }

    private fun assertForeground(device: UiDevice, packageName: String, action: String) {
        assertEquals("Refusing to $action because another app is foreground", packageName, device.currentPackageName)
    }

    private fun streamSample(raw: String): Map<String, Any> = linkedMapOf(
        "observedAtEpochMs" to System.currentTimeMillis(), "chars" to raw.length,
        "utf8Bytes" to raw.toByteArray().size, "sha256" to sha256(raw),
    )

    private fun sha256(text: String): String = MessageDigest.getInstance("SHA-256")
        .digest(text.toByteArray()).joinToString("") { "%02x".format(it) }

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

    private data class SavedPreference(val present: Boolean, val value: Any?)
    private data class Observation(
        val phase: String, val firstAttemptRaw: String, val firstAttemptComplete: Boolean,
        val attemptCount: Int, val finalIr: String?, val repairKind: String?, val error: String?,
        val runtime: String?, val documentExpress: String?, val documentJson: String?, val status: String,
    )

    private companion object {
        const val PREFERENCES = "genuicraft_sdk_demo"
        const val USE_GEMMA = "use_gemma_provider"
        const val MODEL_CHOICE = "e2b_model_choice"
        const val TRAINED_MODEL_PATH = "trained_e2b_v10_w4_model_path"
        const val MTP_ENABLED = "e2b_mtp_enabled"
        const val TOKEN_METRICS = "token_metrics_enabled"
        val PREFERENCE_KEYS = listOf(USE_GEMMA, MODEL_CHOICE, TRAINED_MODEL_PATH, MTP_ENABLED, TOKEN_METRICS)
        const val ARTIFACT_ROOT = "result/sdk_streaming"
        const val LIVE_SCREENSHOT = "live.png"
        const val GENERATED_SCREENSHOT = "generated_repaired.png"
        const val PREVIEW_SCREENSHOT = "preview.png"
        const val MIN_MODEL_BYTES = 2_500_000_000L
        const val UI_TIMEOUT = 10_000L
        const val RUN_TIMEOUT = 90_000L
        const val POLL_MS = 40L
        const val SCREENSHOT_SETTLE_MS = 100L
        const val UI_SETTLE_MS = 500L
        val GSON = GsonBuilder().disableHtmlEscaping().serializeNulls().setPrettyPrinting().create()
    }
}
