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
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Exercises the visible previous-model picker and its native SDK generation path. */
@RunWith(AndroidJUnit4::class)
class PreviousE2bSelectionOnDeviceTest {
    @Test
    fun previousR64PickerPersistsAndGeneratesBxp003WithItsOwnPackage() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val device = UiDevice.getInstance(instrumentation)
        val outputName = InstrumentationRegistry.getArguments().getString("outputDir")
            ?: "previous_r64_${System.currentTimeMillis()}"
        require(Regex("[A-Za-z0-9_-]{1,80}").matches(outputName)) { "Unsafe outputDir" }
        val appFiles = requireNotNull(context.getExternalFilesDir(null))
        val output = File(appFiles, "sdk_previous_model/$outputName")
        require(!output.exists() && output.mkdirs()) { "Output directory must be new: $output" }

        val previousEntry = OnDeviceModelCatalog.entries.single { it.id == PREVIOUS_CATALOG_ID }
        val latestEntry = OnDeviceModelCatalog.entries.single { it.id == LATEST_CATALOG_ID }
        val previousModel = previousEntry.localFile(context).canonicalFile
        val latestModel = latestEntry.localFile(context).canonicalFile
        val manifestFile = File("${previousModel.absolutePath}.fp16.json")
        val case = context.assets.open("genuicraft_bixby50.jsonl").bufferedReader().useLines { lines ->
            lines.filter(String::isNotBlank).map { JsonParser.parseString(it).asJsonObject }
                .first { it.get("id").asString == "BXP-003" }
        }
        val source = case.get("text").asString
        val query = case.get("query").asString
        File(output, "source.md").writeText(source)
        File(output, "query.txt").writeText(query)

        val sdkPrefs = context.getSharedPreferences("genuicraft_sdk_demo", Context.MODE_PRIVATE)
        val precisionPrefs = context.getSharedPreferences("inference_backend_settings", Context.MODE_PRIVATE)
        val savedSdk = SDK_PREF_KEYS.associateWith { saved(sdkPrefs, it) }
        val savedPrecision = saved(precisionPrefs, PRECISION_PREF_KEY)
        val startedAt = System.currentTimeMillis()
        var terminalAt: Long? = null
        var final: Observation? = null
        var failure: Throwable? = null
        var stage = "model_readiness"
        try {
            assertTrue("Previous R64 corrected package missing: $previousModel", previousModel.isFile)
            assertTrue("Previous R64 corrected package unreadable: $previousModel", previousModel.canRead())
            assertTrue("Previous R64 manifest missing: $manifestFile", manifestFile.isFile)
            assertTrue("Previous R64 package is too small", previousModel.length() >= MIN_MODEL_BYTES)
            assertTrue("Wrong previous package location: $previousModel",
                previousModel.path.replace('\\', '/').endsWith(PREVIOUS_RELATIVE_PATH))
            assertFalse("Previous and latest picker targets are the same file", previousModel == latestModel)
            assertEquals(PREVIOUS_CATALOG_ID,
                OnDeviceModelCatalog.entryForModelPath(previousModel.absolutePath)?.id)

            val manifest = JsonParser.parseString(manifestFile.readText()).asJsonObject
            assertEquals(PREVIOUS_SOURCE_SHA, manifest.get("source_sha256").asString)
            assertEquals(PREVIOUS_MODEL_SHA, manifest.get("model_sha256").asString)
            assertEquals(previousModel.length(), manifest.get("model_bytes").asLong)
            assertFalse("Previous package points to the latest source",
                manifest.get("source_sha256").asString == LATEST_SOURCE_SHA)
            assertFalse("Previous package points to the latest corrected model",
                manifest.get("model_sha256").asString == LATEST_MODEL_SHA)

            val selection = selectTrainedE2bModel(previousModel.absolutePath,
                InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED)
            assertTrue("Previous corrected package not ready: ${selection.readiness.message}",
                selection.readiness.usable)
            assertEquals(previousModel.absolutePath, File(selection.modelPath).canonicalPath)
            assertTrue("Previous corrected package should support MTP", selection.mtpSupported)

            check(sdkPrefs.edit()
                .putBoolean("use_gemma_provider", true)
                .putString(PREFERENCE_E2B_MODEL_CHOICE,
                    E2bModelChoice.TRAINED_E2B_V10_W4.preferenceValue)
                .putString(PREFERENCE_TRAINED_E2B_PREVIOUS_R64_MODEL_PATH, previousModel.absolutePath)
                .putBoolean("token_metrics_enabled", true)
                .putBoolean(PREFERENCE_E2B_MTP_ENABLED, true)
                .commit()) { "Could not configure SDK demo preferences" }
            check(precisionPrefs.edit().putString(PRECISION_PREF_KEY,
                InferenceBackendSettings.TrainedE2bGpuPrecision.FP16_CORRECTED.rawValue).commit()) {
                "Could not select corrected FP16 precision"
            }

            val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            }
            stage = "ui_picker"
            ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
                assertTrue("SDK demo did not launch", device.wait(
                    Until.hasObject(By.pkg(context.packageName).text("GenUICraft SDK")), UI_TIMEOUT))
                repeat(2) {
                    clickableAncestor(requireNotNull(findByTextScrolling(device, context.packageName,
                        "Next")) { "Bixby50 Next control not found" }).click()
                    instrumentation.waitForIdleSync()
                }
                assertTrue("BXP-003 was not selected", device.wait(
                    Until.hasObject(By.pkg(context.packageName).textStartsWith("BXP-003")), UI_TIMEOUT))

                val previousChip = requireNotNull(findByTextScrolling(device, context.packageName,
                    E2bModelChoice.TRAINED_E2B_PREVIOUS_R64.displayName)) {
                    "Previous R64 model chip not found"
                }
                clickableAncestor(previousChip).click()
                instrumentation.waitForIdleSync()
                device.waitForIdle(1_000)
                assertEquals(E2bModelChoice.TRAINED_E2B_PREVIOUS_R64,
                    E2bModelChoice.fromPreference(sdkPrefs.getString(PREFERENCE_E2B_MODEL_CHOICE, null)))
                capture(device, output, "selected")
                assertTrue("Previous model chip was not checked",
                    waitForCheckedChip(device, context.packageName,
                        E2bModelChoice.TRAINED_E2B_PREVIOUS_R64.displayName))

                scenario.recreate()
                instrumentation.waitForIdleSync()
                device.waitForIdle(1_000)
                assertEquals("Previous preference did not survive Activity recreation",
                    E2bModelChoice.TRAINED_E2B_PREVIOUS_R64,
                    E2bModelChoice.fromPreference(sdkPrefs.getString(PREFERENCE_E2B_MODEL_CHOICE, null)))
                requireNotNull(findByTextScrolling(device, context.packageName,
                    E2bModelChoice.TRAINED_E2B_PREVIOUS_R64.displayName)) {
                    "Previous R64 model chip not found after Activity recreation"
                }
                capture(device, output, "after_recreation")
                assertTrue("Previous picker checked state did not survive Activity recreation",
                    waitForCheckedChip(device, context.packageName,
                        E2bModelChoice.TRAINED_E2B_PREVIOUS_R64.displayName))

                stage = "native_inference"
                var generationStarted = false
                for (tap in 1..2) {
                    requireNotNull(findByTextScrolling(device, context.packageName, "Generate UI")) {
                        "Generate UI button not reachable"
                    }
                    device.waitForIdle(1_000)
                    val button = clickableAncestor(requireNotNull(device.findObject(
                        By.pkg(context.packageName).text("Generate UI"))) {
                        "Generate UI button disappeared before tap"
                    })
                    val bounds = button.visibleBounds
                    assertTrue("Generate UI button is not fully visible or enabled: $bounds",
                        button.isEnabled && bounds.width() > 0 && bounds.height() > 0 &&
                            bounds.left >= 0 && bounds.right <= device.displayWidth &&
                            bounds.top >= 0 && bounds.bottom <= device.displayHeight)
                    assertTrue("Generate UI center tap was not delivered",
                        device.click(bounds.centerX(), bounds.centerY()))
                    instrumentation.waitForIdleSync()
                    val startDeadline = SystemClock.elapsedRealtime() + 3_000L
                    while (SystemClock.elapsedRealtime() < startDeadline) {
                        final = observe(scenario)
                        if (final!!.trace.phase != SdkGenerationPhase.IDLE) {
                            generationStarted = true
                            break
                        }
                        Thread.sleep(100)
                    }
                    if (generationStarted) break
                    capture(device, output, "generate_tap_$tap")
                }
                assertTrue("Generate UI did not start; final UI status=${final?.status}; see $output",
                    generationStarted)
                val deadline = SystemClock.elapsedRealtime() + RUN_TIMEOUT
                while (SystemClock.elapsedRealtime() < deadline) {
                    final = observe(scenario)
                    if (final!!.trace.phase in TERMINAL_PHASES) {
                        terminalAt = System.currentTimeMillis()
                        break
                    }
                    Thread.sleep(100)
                }
                val observed = requireNotNull(final) { "No SDK generation state observed" }
                val raw = observed.trace.attempts.firstOrNull()?.rawText.orEmpty()
                assertTrue("Previous R64 native output empty; see $output", raw.isNotBlank())
                assertEquals("Activity used a different Bixby50 source", source, observed.generationSource)
                assertEquals("Activity did not run previous model choice",
                    E2bModelChoice.TRAINED_E2B_PREVIOUS_R64.displayName, observed.modelLabel)
                assertTrue("Unexpected native runtime: ${observed.runtime}",
                    observed.runtime.orEmpty().contains("LiteRT-LM/Gemma4/GPU") &&
                        observed.runtime.orEmpty().contains("FP16_CORRECTED"))
                assertEquals("Previous R64 conversion failed: ${observed.trace.error}; see $output",
                    SdkGenerationPhase.COMPLETE, observed.trace.phase)
                assertTrue("No final A2UI document", observed.document?.a2uiJson.orEmpty().isNotBlank())
                assertEquals("Previous R64 IR was not rendered", observed.trace.finalIr,
                    observed.document?.express)
                val metrics = requireNotNull(observed.metrics) { "Native SDK metrics unavailable" }
                assertTrue("No measured provider call", metrics.attempts.any {
                    (it.providerCallElapsedNanos ?: 0L) > 0L
                })
                assertTrue("No native output tokens", (metrics.totalOutputTokens ?: 0L) > 0L)
                assertTrue("No native decode rate",
                    (metrics.weightedNativeDecodeTokensPerSecond ?: 0.0) > 0.0)
                Thread.sleep(600)
                capture(device, output, "preview")
            }
        } catch (caught: Throwable) {
            failure = caught
            throw caught
        } finally {
            final?.trace?.attempts?.forEach { attempt ->
                runCatching { File(output, "raw_attempt_${attempt.number}.express").writeText(attempt.rawText) }
            }
            final?.trace?.finalIr?.let { runCatching { File(output, "final.express").writeText(it) } }
            final?.document?.a2uiJson?.let { runCatching { File(output, "final.a2ui.json").writeText(it) } }
            final?.metrics?.let { runCatching { File(output, "metrics.json").writeText(GSON.toJson(it)) } }
            val result = linkedMapOf<String, Any?>(
                "kind" to "sdk_previous_model_ui_and_native",
                "caseId" to "BXP-003", "query" to query, "sourceSha256" to sha256(source),
                "modelChoice" to E2bModelChoice.TRAINED_E2B_PREVIOUS_R64.preferenceValue,
                "catalogId" to PREVIOUS_CATALOG_ID, "modelPath" to previousModel.absolutePath,
                "modelBytes" to previousModel.takeIf { it.isFile }?.length(),
                "manifestSourceSha256" to PREVIOUS_SOURCE_SHA,
                "manifestModelSha256" to PREVIOUS_MODEL_SHA,
                "startedAtEpochMs" to startedAt, "terminalAtEpochMs" to terminalAt,
                "stage" to stage,
                "deviceModel" to Build.MODEL, "phase" to final?.trace?.phase?.name,
                "runtime" to final?.runtime, "status" to final?.status,
                "repairKind" to final?.trace?.repairKind?.name,
                "error" to final?.trace?.error, "warnings" to final?.trace?.warnings,
                "nativeOutputTokens" to final?.metrics?.totalOutputTokens,
                "nativeDecodeTokensPerSecond" to final?.metrics?.weightedNativeDecodeTokensPerSecond,
                "failure" to failure?.let { Log.getStackTraceString(it) },
            )
            runCatching { File(output, "result.json").writeText(GSON.toJson(result) + "\n") }
            restore(sdkPrefs, savedSdk)
            restore(precisionPrefs, mapOf(PRECISION_PREF_KEY to savedPrecision))
        }
    }

    private fun observe(scenario: ActivityScenario<GenUiSdkDemoActivity>): Observation {
        val observed = AtomicReference<Observation>()
        scenario.onActivity { activity ->
            val model = ViewModelProvider(activity)[GenUiSdkDemoViewModel::class.java]
            observed.set(Observation(activity.generationTraceForTest(),
                activity.renderedDocumentForTest(), activity.generationMetricsForTest(),
                activity.statusForTest(), activity.runtimeForTest(), model.lastModel,
                model.generationSourceText))
        }
        return requireNotNull(observed.get())
    }

    private fun findByTextScrolling(device: UiDevice, packageName: String, label: String): UiObject2? {
        val selector = By.pkg(packageName).text(label)
        device.findObject(selector)?.let { return it }
        // Recreated pages retain their scroll offset; search from the top if the target is above it.
        repeat(8) {
            device.swipe(device.displayWidth * 9 / 10, device.displayHeight * 3 / 10,
                device.displayWidth * 9 / 10, device.displayHeight * 8 / 10, 30)
        }
        repeat(10) {
            device.findObject(selector)?.let { return it }
            device.swipe(device.displayWidth * 9 / 10, device.displayHeight * 8 / 10,
                device.displayWidth * 9 / 10, device.displayHeight * 3 / 10, 30)
            Thread.sleep(150)
        }
        return device.findObject(selector)
    }

    private fun clickableAncestor(node: UiObject2): UiObject2 {
        var current: UiObject2? = node
        while (current != null && !current.isClickable) current = current.parent
        return requireNotNull(current) { "No clickable ancestor for ${node.text}" }
    }

    private fun waitForCheckedChip(device: UiDevice, packageName: String, label: String): Boolean {
        val selector = By.pkg(packageName).text(label)
        val deadline = SystemClock.elapsedRealtime() + 5_000L
        while (SystemClock.elapsedRealtime() < deadline) {
            device.findObject(selector)?.let {
                val chip = clickableAncestor(it)
                if (chip.isCheckable && chip.isChecked) return true
            }
            Thread.sleep(100)
        }
        return false
    }

    private fun capture(device: UiDevice, output: File, label: String) {
        val screenshot = File(output, "$label.png")
        assertTrue("Could not capture $label screen", device.takeScreenshot(screenshot))
        device.dumpWindowHierarchy(File(output, "$label.xml"))
    }

    private fun saved(prefs: SharedPreferences, key: String): SavedPreference =
        SavedPreference(prefs.contains(key), prefs.all[key])

    private fun restore(prefs: SharedPreferences, values: Map<String, SavedPreference>) {
        val editor = prefs.edit()
        values.forEach { (key, item) ->
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
        val document: com.samsung.genuicraft.sdk.GenUiDocument?,
        val metrics: GenerationMetricsUiState?,
        val status: String,
        val runtime: String?,
        val modelLabel: String?,
        val generationSource: String?,
    )

    private companion object {
        const val PREVIOUS_CATALOG_ID = "gemma4_e2b_a2ui_previous_r64"
        const val LATEST_CATALOG_ID = "gemma4_e2b_a2ui_mobile"
        const val PREVIOUS_RELATIVE_PATH =
            "list_card_model_compare/previous_r64/model-fp16-corrected.litertlm"
        const val PREVIOUS_SOURCE_SHA =
            "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
        const val PREVIOUS_MODEL_SHA =
            "7f01bdf1c6ba9bdf658e57c75001fc35a42edad88238bc5dab0a5f7dcf3de373"
        const val LATEST_SOURCE_SHA =
            "1fefbea8a1b6d526d33fe5fba6cf4b212f8c81d3226710972516b432419eacdc"
        const val LATEST_MODEL_SHA =
            "26aa7eedbdfc4fe80321edc7a24aa1cfe85057739e03ffb1b7de7ac5bdb8fdb3"
        const val PRECISION_PREF_KEY = "trained_e2b_gpu_precision"
        const val MIN_MODEL_BYTES = 2_500_000_000L
        const val UI_TIMEOUT = 10_000L
        const val RUN_TIMEOUT = 180_000L
        val TERMINAL_PHASES = setOf(SdkGenerationPhase.COMPLETE, SdkGenerationPhase.FAILED,
            SdkGenerationPhase.CANCELLED)
        val SDK_PREF_KEYS = listOf("use_gemma_provider", PREFERENCE_E2B_MODEL_CHOICE,
            PREFERENCE_TRAINED_E2B_PREVIOUS_R64_MODEL_PATH, "token_metrics_enabled",
            PREFERENCE_E2B_MTP_ENABLED)
        val GSON = GsonBuilder().disableHtmlEscaping().serializeNulls().setPrettyPrinting().create()
    }
}
