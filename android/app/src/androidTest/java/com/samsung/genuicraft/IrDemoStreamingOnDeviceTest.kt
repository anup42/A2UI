package com.samsung.genuicraft

import android.content.Intent
import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.pipeline.IrPromptVersionSettings
import java.io.File
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class IrDemoStreamingOnDeviceTest {

    @Test
    fun streamsRawIrThenShowsRepairedIrWithoutReplacingRawEvidence() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val device = UiDevice.getInstance(instrumentation)
        val entry = requireNotNull(OnDeviceModelCatalog.entries.firstOrNull { it.id == MODEL_ID })
        val modelFile = entry.localFile(context).canonicalFile
        val record = context.assets.open(FIXTURE_ASSET).bufferedReader().use { loadBxp001(it.readLines()) }
        val startedAt = System.currentTimeMillis()
        val runDir = File(context.filesDir, "$ARTIFACT_ROOT/${startedAt}_${UUID.randomUUID().toString().take(8)}")
        check(runDir.mkdirs()) { "Could not create ${runDir.absolutePath}" }

        assertTrue("$MODEL_ID must use the trained SDK converter", entry.usesTrainedSdkConverter)
        assertTrue("$MODEL_ID must require GPU", entry.requireGpu)
        assertTrue("$MODEL_ID must expose MTP", entry.enableSpeculativeDecoding)
        assertTrue("Trained model is missing: ${modelFile.absolutePath}", modelFile.isFile)
        assertTrue("Trained model is incomplete: ${modelFile.length()} bytes", modelFile.length() >= entry.minimumFileSizeBytes)
        assertEquals("BXP-001", record.queryId)
        assertFalse("Fixture must not contain hand-edited generated IR", record.hasSavedIr)

        val oldProvider = InferenceBackendSettings.getIrProvider(context)
        val oldPath = InferenceBackendSettings.getOnDeviceModelPath(context)
        val oldAccelerator = InferenceBackendSettings.getOnDeviceAccelerator(context)
        val oldMtp = InferenceBackendSettings.getOnDeviceMtpEnabled(context)
        val oldFormat = IrPromptVersionSettings.getSelectedVersionId(context)
        try {
            InferenceBackendSettings.setIrProvider(context, InferenceBackendSettings.Provider.ON_DEVICE_LITERT)
            InferenceBackendSettings.setOnDeviceModelPath(context, modelFile.absolutePath)
            InferenceBackendSettings.setOnDeviceAccelerator(context, InferenceBackendSettings.Accelerator.GPU)
            InferenceBackendSettings.setOnDeviceMtpEnabled(context, true)
            IrPromptVersionSettings.setSelectedVersionId(context, EXPRESS_FORMAT)
            assertEquals(MODEL_ID, OnDeviceModelCatalog.selectedEntry(context)?.id)
            assertEquals(InferenceBackendSettings.Accelerator.GPU, InferenceBackendSettings.getOnDeviceAccelerator(context))
            assertTrue(InferenceBackendSettings.getOnDeviceMtpEnabled(context))

            IrDemoSessionStore.update("instrumentation:BXP-001:$startedAt", listOf(record))
            val intent = Intent(context, IrDemoRenderActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                putExtra(IrDemoRenderActivity.EXTRA_RECORD_INDEX, 0)
                putExtra(IrDemoRenderActivity.EXTRA_FORCE_FRESH, true)
            }
            ActivityScenario.launch<IrDemoRenderActivity>(intent).use { scenario ->
                instrumentation.waitForIdleSync()
                assertForeground(device, context.packageName, "enable Debug")
                val debugSwitch = requireNotNull(device.wait(Until.findObject(By.pkg(context.packageName).checkable(true)), UI_TIMEOUT)) {
                    "IR Demo Debug switch did not appear"
                }
                if (!debugSwitch.isChecked) {
                    assertForeground(device, context.packageName, "tap Debug")
                    debugSwitch.click()
                    requireNotNull(device.wait(Until.findObject(By.pkg(context.packageName).checkable(true).checked(true)), UI_TIMEOUT)) {
                        "IR Demo Debug switch did not turn on"
                    }
                }
                val debugEnabledAt = System.currentTimeMillis()
                val samples = mutableListOf<Map<String, Any>>()
                var lastRaw = ""
                var rawAtStreamCompletion: String? = null
                var finalObservation: Observation? = null
                val deadline = SystemClock.elapsedRealtime() + PIPELINE_TIMEOUT
                while (SystemClock.elapsedRealtime() < deadline) {
                    val observed = observe(scenario)
                    observed.failure?.takeIf(String::isNotBlank)?.let { error("IR Demo failed: $it") }
                    val raw = observed.raw
                    if (!observed.streamComplete && raw.isNotBlank() && raw != lastRaw) {
                        assertTrue("Raw stream must be cumulative", raw.startsWith(lastRaw))
                        lastRaw = raw
                        samples += sample(raw)
                        if (samples.size == 2) {
                            capture(device, context.packageName, File(runDir, LIVE_SCREENSHOT), "capture live stream")
                        }
                    }
                    if (observed.streamComplete && rawAtStreamCompletion == null) rawAtStreamCompletion = raw
                    if (observed.completed) {
                        finalObservation = observed
                        break
                    }
                    Thread.sleep(POLL_MS)
                }

                val final = requireNotNull(finalObservation) { "IR Demo did not complete within ${PIPELINE_TIMEOUT / 1_000}s" }
                assertTrue("Expected at least two distinct non-empty raw snapshots before completion; got ${samples.size}", samples.size >= 2)
                val completedRaw = requireNotNull(rawAtStreamCompletion).also { assertTrue("Completed raw IR is empty", it.isNotBlank()) }
                assertEquals("Repair replaced the preserved raw model IR", completedRaw, final.raw)
                assertEquals("Repaired IR", final.finalLabel)
                val repaired = requireNotNull(final.finalIr) { "Final repaired IR is missing" }
                assertNotEquals("Repair was reported but final IR equals raw model output", completedRaw, repaired)
                assertTrue("Native renderer produced no public surface", final.surfaceCount > 0)
                assertTrue("Native renderer failed: ${final.renderError}", final.renderError.isNullOrBlank())
                assertTrue("Runtime did not report GPU: ${final.runtime}", final.runtime.orEmpty().contains("GPU", true))
                assertTrue("Runtime did not execute MTP: ${final.runtime}", final.runtime.orEmpty().contains("+MTP", true))
                Thread.sleep(UI_SETTLE_MS)
                capture(device, context.packageName, File(runDir, FINAL_SCREENSHOT), "capture final repaired IR")

                File(runDir, "streaming_proof.json").writeText(GSON.toJson(linkedMapOf(
                    "fixtureId" to record.queryId, "modelId" to entry.id, "modelPath" to modelFile.absolutePath,
                    "gpuSelected" to true, "mtpSelected" to true, "startedAtEpochMs" to startedAt,
                    "debugEnabledAtEpochMs" to debugEnabledAt, "completedAtEpochMs" to System.currentTimeMillis(),
                    "rawSnapshotsBeforeCompletion" to samples, "rawSnapshotCount" to samples.size,
                    "rawPreservedAfterRepair" to (completedRaw == final.raw), "rawChars" to completedRaw.length,
                    "rawUtf8Bytes" to completedRaw.toByteArray().size, "rawSha256" to sha256(completedRaw),
                    "finalLabel" to final.finalLabel, "repairApplied" to true, "finalChars" to repaired.length,
                    "finalUtf8Bytes" to repaired.toByteArray().size, "finalSha256" to sha256(repaired),
                    "rawModelIr" to completedRaw, "finalIr" to repaired, "runtime" to final.runtime,
                    "outputTokens" to final.outputTokens, "outputTokensPerSecond" to final.tokensPerSecond,
                    "liveScreenshot" to LIVE_SCREENSHOT, "finalScreenshot" to FINAL_SCREENSHOT,
                )))
            }
        } finally {
            InferenceBackendSettings.setIrProvider(context, oldProvider)
            InferenceBackendSettings.setOnDeviceModelPath(context, oldPath)
            InferenceBackendSettings.setOnDeviceAccelerator(context, oldAccelerator)
            InferenceBackendSettings.setOnDeviceMtpEnabled(context, oldMtp)
            IrPromptVersionSettings.setSelectedVersionId(context, oldFormat)
        }
    }

    private fun observe(scenario: ActivityScenario<IrDemoRenderActivity>): Observation {
        val value = AtomicReference<Observation>()
        scenario.onActivity { activity ->
            val debug = activity.debugIrSnapshotForTest()
            val result = activity.completedPipelineResultForTest()
            value.set(Observation(debug.rawModelIr, debug.streamComplete, debug.finalIr, debug.finalIrLabel,
                result != null, activity.failureMessageForTest(), result?.renderResult?.surfaces?.size ?: 0,
                result?.renderResult?.errorMessage, result?.stage3RuntimeBackend, result?.stage3OutputTokens,
                result?.stage3OutputTokensPerSecond))
        }
        return requireNotNull(value.get())
    }

    private fun loadBxp001(lines: List<String>): IrDemoRecord {
        val row = lines.asSequence().map { JsonParser.parseString(it).asJsonObject }
            .first { it.get("id").asString == "BXP-001" }
        return IrDemoRecord(row.get("id").asString, null, row.get("query").asString, row.get("text").asString)
    }

    private fun sample(raw: String): Map<String, Any> = linkedMapOf(
        "observedAtEpochMs" to System.currentTimeMillis(), "chars" to raw.length,
        "utf8Bytes" to raw.toByteArray().size, "sha256" to sha256(raw),
    )

    private fun capture(device: UiDevice, pkg: String, file: File, action: String) {
        assertForeground(device, pkg, action)
        assertTrue("Screenshot failed: ${file.name}", device.takeScreenshot(file))
    }

    private fun assertForeground(device: UiDevice, pkg: String, action: String) {
        assertEquals("Refusing to $action because another app is foreground", pkg, device.currentPackageName)
    }

    private fun sha256(text: String): String = MessageDigest.getInstance("SHA-256")
        .digest(text.toByteArray()).joinToString("") { "%02x".format(it) }

    private data class Observation(val raw: String, val streamComplete: Boolean, val finalIr: String?,
        val finalLabel: String?, val completed: Boolean, val failure: String?, val surfaceCount: Int,
        val renderError: String?, val runtime: String?, val outputTokens: Int?, val tokensPerSecond: Double?)

    private companion object {
        const val MODEL_ID = "gemma4_e2b_a2ui_mobile"
        const val FIXTURE_ASSET = "genuicraft_bixby50.jsonl"
        const val EXPRESS_FORMAT = "a2ui_express_v1"
        const val ARTIFACT_ROOT = "result/ir_demo_streaming"
        const val LIVE_SCREENSHOT = "live_raw_ir.png"
        const val FINAL_SCREENSHOT = "final_repaired_ir.png"
        const val UI_TIMEOUT = 10_000L
        const val PIPELINE_TIMEOUT = 5 * 60_000L
        const val POLL_MS = 50L
        const val UI_SETTLE_MS = 750L
        val GSON = GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create()
    }
}
