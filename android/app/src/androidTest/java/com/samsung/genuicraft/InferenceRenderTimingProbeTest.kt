package com.samsung.genuicraft

import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import com.samsung.genuicraft.sdk.GenUiCompiler
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Replays saved real-model A2UI without loading a model. This measures compiler and display-path
 * work separately from generation. UiAutomator observing a unique status label is a polling upper
 * bound for accessibility-visible UI state, not a GPU frame-present or pixel-paint timestamp.
 *
 * Required instrumentation args: inputDir=<absolute device arm directory containing numbered
 * case directories with final.a2ui.json>, outputDir=<new simple leaf>. Optional: repeats=2,
 * observationTimeoutMs=15000. The first of BXP-001, BXP-003 and BXP-031 is replayed in that order.
 */
@RunWith(AndroidJUnit4::class)
class InferenceRenderTimingProbeTest {
    @Test
    fun measureSavedFinalRendering() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val args = InstrumentationRegistry.getArguments()
        val inputPath = File(requireArg(args, "inputDir"))
        require(inputPath.isAbsolute) { "inputDir must be absolute" }
        val inputDir = inputPath.canonicalFile
        require(inputDir.isDirectory && inputDir.canRead()) {
            "inputDir must be a readable absolute device arm directory"
        }
        val outputLeaf = requireArg(args, "outputDir")
        require(Regex("[A-Za-z0-9_-]{1,80}").matches(outputLeaf)) {
            "outputDir must be a simple new leaf"
        }
        val repeats = args.getString("repeats", "2")!!.toInt()
        require(repeats in 1..3) { "repeats must be 1..3" }
        val timeoutMs = args.getString("observationTimeoutMs", "15000")!!.toLong()
        require(timeoutMs in 1_000L..60_000L) { "observationTimeoutMs must be 1000..60000" }
        val outputBase = File(context.getExternalFilesDir(null), "sdk_render_timing")
        require(outputBase.isDirectory || outputBase.mkdirs()) { "Cannot create $outputBase" }
        val outputDir = File(outputBase, outputLeaf)
        require(!outputDir.exists()) { "outputDir already exists: $outputDir" }
        require(outputDir.mkdir()) { "Cannot create outputDir: $outputDir" }
        val gson = GsonBuilder().setPrettyPrinting().serializeNulls().disableHtmlEscaping().create()
        fun write(name: String, value: Any) =
            File(outputDir, name).writeText(gson.toJson(value) + "\n")

        val ids = listOf("BXP-001", "BXP-003", "BXP-031")
        val inputs = ids.map { id ->
            val dir = inputDir.listFiles().orEmpty().asSequence()
                .filter { it.isDirectory && Regex("[0-9]{3}_${Regex.escape(id)}").matches(it.name) }
                .sortedBy { it.name }
                .firstOrNull() ?: error("Missing numbered directory for $id in $inputDir")
            val file = File(dir, "final.a2ui.json")
            require(file.isFile && file.canRead() && file.length() > 0L) {
                "Missing readable final.a2ui.json for $id: $file"
            }
            Triple(id, file, file.readText())
        }
        val device = UiDevice.getInstance(instrumentation)
        val nonce = SystemClock.elapsedRealtimeNanos().toString(36)
        write("run_config.json", mapOf(
            "schema" to 1,
            "inputDir" to inputDir.absolutePath,
            "outputDir" to outputDir.absolutePath,
            "caseIds" to ids,
            "repeats" to repeats,
            "observationTimeoutMs" to timeoutMs,
            "measurement" to "UiAutomator unique-status-label observation is a polling upper bound, not pixel paint",
            "firstActivityLaunch" to "first launch in this instrumentation process; process-cold is not guaranteed",
            "screenshots" to "one per case after the measured interval",
        ))

        val records = mutableListOf<Map<String, Any?>>()
        val failures = mutableListOf<String>()
        val launchStartNs = SystemClock.elapsedRealtimeNanos()
        ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
            val firstActivityLaunchMs = millisSince(launchStartNs)
            write("activity_launch.json", mapOf(
                "firstActivityLaunchMs" to firstActivityLaunchMs,
                "processColdGuaranteed" to false,
            ))
            for ((id, file, savedA2ui) in inputs) {
                for (repeatIndex in 1..repeats) {
                    val compilerStartNs = SystemClock.elapsedRealtimeNanos()
                    val document = try {
                        GenUiCompiler.compile(savedA2ui)
                    } catch (t: Throwable) {
                        val message = "$id repeat $repeatIndex compile failed: ${t.javaClass.simpleName}: ${t.message}"
                        failures += message
                        records += mapOf("caseId" to id, "repeat" to repeatIndex,
                            "inputPath" to file.absolutePath, "error" to message)
                        continue
                    }
                    val compilerTimeMs = millisSince(compilerStartNs)
                    val label = "Render probe $id pass $repeatIndex $nonce"
                    var callbackTimeMs = -1.0
                    val displayStartNs = SystemClock.elapsedRealtimeNanos()
                    var error: String? = null
                    try {
                        scenario.onActivity { activity ->
                            val callbackStartNs = SystemClock.elapsedRealtimeNanos()
                            activity.showDocument(document, label)
                            callbackTimeMs = millisSince(callbackStartNs)
                            // Compose state may retain the equal prior value when replaying the same
                            // document. Validate content, not allocation identity.
                            check(activity.renderedDocumentForTest() == document) {
                                "Activity did not retain the compiled document"
                            }
                        }
                    } catch (t: Throwable) {
                        error = "showDocument failed: ${t.javaClass.simpleName}: ${t.message}"
                    }
                    val activityDispatchWallMs = millisSince(displayStartNs)
                    val observed = error == null &&
                        device.wait(Until.hasObject(By.text(label)), timeoutMs)
                    val labelObservationWallMs = millisSince(displayStartNs)
                    if (!observed && error == null) error = "Unique status label not observed within ${timeoutMs}ms"
                    records += mapOf(
                        "caseId" to id,
                        "repeat" to repeatIndex,
                        "inputPath" to file.absolutePath,
                        "inputBytes" to file.length(),
                        "inputSha256" to sha256(savedA2ui),
                        "compilerTimeMs" to compilerTimeMs,
                        "showDocumentCallbackTimeMs" to callbackTimeMs.takeIf { it >= 0.0 },
                        "activityDispatchWallMs" to activityDispatchWallMs,
                        "labelObservationWallMs" to labelObservationWallMs,
                        "labelObserved" to observed,
                        "statusLabel" to label,
                        "error" to error,
                    )
                    write("results.json", records)
                    if (error != null) failures += "$id repeat $repeatIndex: $error"
                    if (repeatIndex == 1) {
                        val screenshotFile = File(outputDir, "${id}_screen.png")
                        val captured = device.takeScreenshot(screenshotFile)
                        write("${id}_screenshot.json", mapOf(
                            "path" to screenshotFile.absolutePath,
                            "captured" to captured,
                            "capturedAfterMeasuredInterval" to true,
                        ))
                        if (!captured) failures += "$id: screenshot capture failed"
                    }
                }
            }
        }
        write("summary.json", mapOf(
            "firstActivityLaunch" to "activity_launch.json",
            "requestedRenders" to ids.size * repeats,
            "recordedRenders" to records.size,
            "failedChecks" to failures,
            "observationLimit" to "accessibility-visible status label, not rendered-pixel timestamp",
        ))
        assertTrue(failures.joinToString("; "), failures.isEmpty())
    }

    private fun requireArg(args: android.os.Bundle, key: String): String =
        args.getString(key)?.trim()?.takeIf(String::isNotEmpty)
            ?: error("Missing instrumentation arg: $key")

    private fun millisSince(startNs: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0

    private fun sha256(text: String): String = MessageDigest.getInstance("SHA-256")
        .digest(text.toByteArray(Charsets.UTF_8))
        .joinToString("") { "%02x".format(it) }
}
