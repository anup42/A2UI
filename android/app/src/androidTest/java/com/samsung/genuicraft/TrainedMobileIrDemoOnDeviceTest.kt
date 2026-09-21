package com.samsung.genuicraft

import android.content.Intent
import android.os.Build
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.BySelector
import androidx.test.uiautomator.UiDevice
import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
import com.samsung.genuicraft.inference.OnDeviceModelCatalog
import com.samsung.genuicraft.pipeline.A2uiCanonicalGraph
import com.samsung.genuicraft.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.pipeline.IrPromptVersionSettings
import com.samsung.genuicraft.pipeline.ResponseFactCoverage
import java.io.File
import java.util.Locale
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the first bundled IR Demo record through the trained mobile E2B model in
 * the real [IrDemoRenderActivity]. The activity owns the only inference call;
 * this test observes its terminal result and the visible Compose UI.
 */
@RunWith(AndroidJUnit4::class)
class TrainedMobileIrDemoOnDeviceTest {

    @Test
    fun firstIrDemoRecordGeneratesAndRendersWithSelectedTrainedMobileModel() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val arguments = InstrumentationRegistry.getArguments()
        val mtpEnabled = booleanArgument(arguments.getString(ARG_MTP), ARG_MTP, default = true)
        val keepSelection = booleanArgument(
            arguments.getString(ARG_KEEP_SELECTION),
            ARG_KEEP_SELECTION,
            default = false,
        )
        val runId = "${System.currentTimeMillis()}_${if (mtpEnabled) "mtp_on" else "mtp_off"}_" +
            UUID.randomUUID().toString().take(8)
        val runDir = File(context.filesDir, "$ARTIFACT_ROOT/$runId")
        check(runDir.mkdirs()) { "Could not create unique artifact directory ${runDir.absolutePath}" }

        val originalIrProvider = InferenceBackendSettings.getIrProvider(context)
        val originalModelPath = InferenceBackendSettings.getOnDeviceModelPath(context)
        val originalAccelerator = InferenceBackendSettings.getOnDeviceAccelerator(context)
        val originalMtpEnabled = InferenceBackendSettings.getOnDeviceMtpEnabled(context)
        val originalFormatId = IrPromptVersionSettings.getSelectedVersionId(context)
        val gson = GsonBuilder().disableHtmlEscaping().serializeNulls().setPrettyPrinting().create()
        val device = UiDevice.getInstance(instrumentation)
        val startedAtMs = System.currentTimeMillis()

        try {
            val entry = requireNotNull(
                OnDeviceModelCatalog.entries.firstOrNull { it.id == MODEL_ID }
            ) { "Missing on-device model catalog entry $MODEL_ID" }
            assertTrue(
                "$MODEL_ID is not exposed by the on-device model picker",
                OnDeviceModelCatalog.visibleEntries.any { it.id == MODEL_ID },
            )
            assertTrue("$MODEL_ID must use GenUiTrainedConverter", entry.usesTrainedSdkConverter)
            assertTrue("$MODEL_ID must require GPU", entry.requireGpu)
            assertEquals("Unexpected trained-model context window", 8_192, entry.maxContextTokens)
            assertEquals("Unexpected trained-model output limit", 2_048, entry.maxOutputTokens)

            val modelFile = entry.localFile(context).canonicalFile
            assertEquals(MODEL_FILE_NAME, modelFile.name)
            assertEquals("sdk_models", modelFile.parentFile?.name)
            assertTrue("Trained mobile model is missing: ${modelFile.absolutePath}", modelFile.isFile)
            assertTrue("Trained mobile model is unreadable: ${modelFile.absolutePath}", modelFile.canRead())
            assertTrue(
                "Trained mobile model is incomplete: ${modelFile.length()} bytes",
                modelFile.length() >= entry.minimumFileSizeBytes,
            )

            val record = IrDemoRecordLoader.loadDefault(context.assets).firstOrNull()
                ?: error("The bundled IR Demo corpus is empty.")
            assertEquals("The bounded validation must use the first salary case", FIRST_QUERY_ID, record.queryId)
            assertFalse("The test record unexpectedly contains replayable saved IR", record.hasSavedIr)

            InferenceBackendSettings.setIrProvider(
                context,
                InferenceBackendSettings.Provider.ON_DEVICE_LITERT,
            )
            InferenceBackendSettings.setOnDeviceModelPath(context, modelFile.absolutePath)
            InferenceBackendSettings.setOnDeviceAccelerator(
                context,
                InferenceBackendSettings.Accelerator.GPU,
            )
            InferenceBackendSettings.setOnDeviceMtpEnabled(context, mtpEnabled)
            IrPromptVersionSettings.setSelectedVersionId(context, EXPRESS_FORMAT_ID)

            assertEquals(
                InferenceBackendSettings.Provider.ON_DEVICE_LITERT,
                InferenceBackendSettings.getIrProvider(context),
            )
            assertEquals(
                modelFile,
                File(InferenceBackendSettings.getOnDeviceModelPath(context)).canonicalFile,
            )
            assertEquals(
                InferenceBackendSettings.Accelerator.GPU,
                InferenceBackendSettings.getOnDeviceAccelerator(context),
            )
            assertEquals(mtpEnabled, InferenceBackendSettings.getOnDeviceMtpEnabled(context))
            assertEquals(
                MODEL_ID,
                OnDeviceModelCatalog.selectedEntry(context)?.id,
            )

            writeJson(
                File(runDir, "configuration.json"),
                linkedMapOf(
                    "runId" to runId,
                    "queryId" to record.queryId,
                    "queryText" to record.queryText,
                    "recordHasSavedIr" to record.hasSavedIr,
                    "modelId" to entry.id,
                    "modelPath" to modelFile.absolutePath,
                    "modelSizeBytes" to modelFile.length(),
                    "usesTrainedSdkConverter" to entry.usesTrainedSdkConverter,
                    "irProvider" to InferenceBackendSettings.getIrProvider(context).rawValue,
                    "accelerator" to InferenceBackendSettings.getOnDeviceAccelerator(context).rawValue,
                    "mtpRequested" to mtpEnabled,
                    "formatId" to IrPromptVersionSettings.getSelectedVersionId(context),
                    "startedAtEpochMs" to startedAtMs,
                    "device" to linkedMapOf(
                        "manufacturer" to Build.MANUFACTURER,
                        "model" to Build.MODEL,
                        "device" to Build.DEVICE,
                        "sdkInt" to Build.VERSION.SDK_INT,
                        "release" to Build.VERSION.RELEASE,
                        "supportedAbis" to Build.SUPPORTED_ABIS.toList(),
                    ),
                ),
                gson,
            )

            // The loader's raw record has no GenUI payload, so the real screen must generate it.
            IrDemoSessionStore.update(
                sourceLabel = "instrumentation:$runId",
                records = listOf(record),
            )
            val intent = Intent(context, IrDemoRenderActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                putExtra(IrDemoRenderActivity.EXTRA_RECORD_INDEX, 0)
                putExtra(IrDemoRenderActivity.EXTRA_FORCE_FRESH, true)
            }

            ActivityScenario.launch<IrDemoRenderActivity>(intent).use { scenario ->
                instrumentation.waitForIdleSync()
                val result = try {
                    waitForPipelineResult(scenario)
                } catch (failure: Throwable) {
                    captureFailureUi(device, runDir)
                    throw failure
                }

                File(runDir, "generated_ir.json").writeText(result.stage3Json)
                val decoded = A2uiExpressCodec.decode(result.stage3Json)
                val strictValidation = A2uiCanonicalGraph.validate(decoded)
                val missingFacts = ResponseFactCoverage.missingFacts(result.stage2Response, decoded)
                val runtime = result.stage3RuntimeBackend.orEmpty()
                val runtimeHasMtp = runtimeReportsEnabledMtp(runtime)
                val sourceFallbackWarnings = result.warnings.filter(::isSourceTextFallbackWarning)

                writeJson(
                    File(runDir, "result.json"),
                    linkedMapOf(
                        "success" to true,
                        "queryId" to record.queryId,
                        "runtime" to runtime,
                        "mtpRequested" to mtpEnabled,
                        "mtpReportedEnabled" to runtimeHasMtp,
                        "inputTokens" to result.stage3InputTokens,
                        "outputTokens" to result.stage3OutputTokens,
                        "outputTokensPerSecond" to result.stage3OutputTokensPerSecond,
                        "stageDurationsMs" to result.stageDurationsMs.mapKeys { it.key.name },
                        "stageStreamDurationsMs" to result.stageStreamDurationsMs.mapKeys { it.key.name },
                        "usedFallback" to result.usedFallback,
                        "sourceTextFallbackWarnings" to sourceFallbackWarnings,
                        "warnings" to result.warnings,
                        "missingResponseFacts" to missingFacts.map {
                            linkedMapOf("label" to it.label, "value" to it.value)
                        },
                        "strictValid" to strictValidation.isValid,
                        "strictError" to strictValidation.error,
                        "renderSurfaceCount" to result.renderResult.surfaces.size,
                        "renderError" to result.renderResult.errorMessage,
                        "stage2Prompt" to result.stage2Prompt,
                        "finishedAtEpochMs" to System.currentTimeMillis(),
                        "elapsedMs" to System.currentTimeMillis() - startedAtMs,
                    ),
                    gson,
                )

                assertTrue("Generated IR is empty", result.stage3Json.isNotBlank())
                assertNotEquals("The Stage 2 response was replayed as IR", record.responseText, result.stage3Json)
                assertFalse("A source-text fallback was reported", result.usedFallback)
                assertTrue(
                    "Source-text fallback warning(s) were emitted: $sourceFallbackWarnings",
                    sourceFallbackWarnings.isEmpty(),
                )
                assertTrue(
                    "IR Demo unexpectedly invoked a cloud Stage 2: ${result.stage2Prompt}",
                    result.stage2Prompt.contains("preloaded", ignoreCase = true) &&
                        result.warnings.any {
                            it.contains("stage 2 skipped", ignoreCase = true)
                        },
                )
                assertTrue(
                    "Generated IR failed strict validation: ${strictValidation.error}",
                    strictValidation.isValid,
                )
                assertTrue("No native render surface was produced", result.renderResult.surfaces.isNotEmpty())
                assertTrue(
                    "Native rendering failed: ${result.renderResult.errorMessage}",
                    result.renderResult.errorMessage.isNullOrBlank(),
                )
                assertTrue("Runtime did not report GPU execution: $runtime", runtime.contains("GPU", true))
                assertEquals("Runtime MTP identity disagrees with the selected setting: $runtime", mtpEnabled, runtimeHasMtp)
                assertTrue(
                    "Input-token metrics were not reported: ${result.stage3InputTokens}",
                    (result.stage3InputTokens ?: 0) > 0,
                )
                assertTrue(
                    "Output-token metrics were not reported: ${result.stage3OutputTokens}",
                    (result.stage3OutputTokens ?: 0) > 0,
                )
                assertTrue(
                    "Decode throughput was not reported: ${result.stage3OutputTokensPerSecond}",
                    (result.stage3OutputTokensPerSecond ?: 0.0) > 0.0,
                )

                instrumentation.waitForIdleSync()
                Thread.sleep(UI_SETTLE_MS)
                assertIrDemoForeground(device, context.packageName)
                val screenshot = File(runDir, "screenshot.png")
                val hierarchy = File(runDir, "ui_hierarchy.xml")
                assertTrue("Could not capture rendered IR Demo screenshot", device.takeScreenshot(screenshot))
                device.dumpWindowHierarchy(hierarchy)
                assertTrue("Rendered UI hierarchy is empty", hierarchy.isFile && hierarchy.length() > 0L)

                val renderTexts = collectRenderedUiText(device, context.packageName)
                File(runDir, "rendered_text.txt").writeText(renderTexts.joinToString("\n") + "\n")
                assertFalse(
                    "The real IR Demo screen displayed a renderer failure: $renderTexts",
                    renderTexts.any { text ->
                        text.contains("Unable to render", true) ||
                            text.contains("No renderable", true)
                    },
                )
                val generatedMarkers = meaningfulGeneratedMarkers(decoded, record.queryText)
                val visibleGeneratedMarker = generatedMarkers.firstOrNull { marker ->
                    renderTexts.any { visibleText ->
                        visibleText.equals(marker, ignoreCase = true) ||
                            (marker.any(Char::isDigit) &&
                                visibleText.contains(marker, ignoreCase = true)) ||
                            (marker.length >= 12 &&
                                !comparableText(record.queryText).contains(comparableText(marker)) &&
                                visibleText.contains(marker, ignoreCase = true))
                    }
                }
                assertTrue(
                    "No meaningful generated salary title or figure was visible. " +
                        "Generated markers=$generatedMarkers renderedText=$renderTexts",
                    visibleGeneratedMarker != null,
                )

                val debugSwitch = waitForObject(
                    device,
                    By.pkg(context.packageName).checkable(true),
                    UI_READY_TIMEOUT_MS,
                    "IR Demo Debug switch",
                )
                if (!debugSwitch.isChecked) {
                    assertIrDemoForeground(device, context.packageName)
                    debugSwitch.click()
                    waitForObject(
                        device,
                        By.pkg(context.packageName).checkable(true).checked(true),
                        UI_READY_TIMEOUT_MS,
                        "enabled IR Demo Debug switch",
                    )
                }
                Thread.sleep(UI_SETTLE_MS)
                val debugTexts = collectDebugDiagnostics(device, context.packageName, runtime)
                assertIrDemoForeground(device, context.packageName)
                val debugScreenshotCaptured =
                    device.takeScreenshot(File(runDir, "debug_screenshot.png"))
                File(runDir, "debug_diagnostics.txt").writeText(
                    "collection=best_effort_non_gating\n" +
                        "maximumBackwardSwipes=$MAX_DEBUG_SCROLLS\n" +
                        "screenshotCaptured=$debugScreenshotCaptured\n" +
                        debugTexts.joinToString("\n") + "\n",
                )
                val debugHierarchy = File(runDir, "debug_hierarchy.xml")
                device.dumpWindowHierarchy(debugHierarchy)

                File(runDir, "PASS").writeText(
                    "runtime=$runtime\nvisibleGeneratedMarker=$visibleGeneratedMarker\n" +
                        "missingFactCount=${missingFacts.size}\n",
                )
            }
        } catch (failure: Throwable) {
            File(runDir, "FAILURE.txt").writeText(
                "${failure.javaClass.name}: ${failure.message.orEmpty()}\n",
            )
            throw failure
        } finally {
            if (!keepSelection) {
                InferenceBackendSettings.setIrProvider(context, originalIrProvider)
                InferenceBackendSettings.setOnDeviceModelPath(context, originalModelPath)
                InferenceBackendSettings.setOnDeviceAccelerator(context, originalAccelerator)
                InferenceBackendSettings.setOnDeviceMtpEnabled(context, originalMtpEnabled)
                IrPromptVersionSettings.setSelectedVersionId(context, originalFormatId)
            }
        }
    }

    private fun waitForPipelineResult(
        scenario: ActivityScenario<IrDemoRenderActivity>,
    ): GenUiStagePipeline.PipelineResult {
        val deadline = System.currentTimeMillis() + PIPELINE_TIMEOUT_MS
        while (System.currentTimeMillis() < deadline) {
            val completed = AtomicReference<GenUiStagePipeline.PipelineResult?>()
            val failure = AtomicReference<String?>()
            scenario.onActivity { activity ->
                completed.set(activity.completedPipelineResultForTest())
                failure.set(activity.failureMessageForTest())
            }
            completed.get()?.let { return it }
            failure.get()?.takeIf { it.isNotBlank() }?.let {
                error("IR Demo generation failed: $it")
            }
            Thread.sleep(PIPELINE_POLL_MS)
        }
        error("IR Demo did not reach a terminal state within ${PIPELINE_TIMEOUT_MS / 1_000}s")
    }

    private fun collectRenderedUiText(device: UiDevice, packageName: String): List<String> {
        val values = linkedSetOf<String>()
        var unchangedScreens = 0
        var previousScreen = ""
        repeat(MAX_RENDER_SCROLLS) {
            val current = visibleText(device, packageName)
            values += current
            val screenKey = current.joinToString("\u0000")
            unchangedScreens = if (screenKey == previousScreen) unchangedScreens + 1 else 0
            previousScreen = screenKey
            if (unchangedScreens >= 2) return@repeat
            assertIrDemoForeground(device, packageName)
            device.swipe(
                device.displayWidth / 2,
                device.displayHeight * 82 / 100,
                device.displayWidth / 2,
                device.displayHeight * 30 / 100,
                30,
            )
            Thread.sleep(UI_SCROLL_SETTLE_MS)
        }
        return compactText(values)
    }

    private fun collectDebugDiagnostics(
        device: UiDevice,
        packageName: String,
        runtime: String,
    ): List<String> {
        val values = linkedSetOf<String>()
        repeat(MAX_DEBUG_SCROLLS) {
            values += visibleText(device, packageName)
            val joined = values.joinToString("\n")
            if (joined.contains("IR backend: on_device_litert", true) &&
                joined.contains("Pipeline completed successfully", true) &&
                joined.contains("Runtime: $runtime", true)
            ) {
                return compactDiagnostics(values)
            }
            assertIrDemoForeground(device, packageName)
            device.swipe(
                device.displayWidth / 2,
                device.displayHeight * 30 / 100,
                device.displayWidth / 2,
                device.displayHeight * 85 / 100,
                20,
            )
            Thread.sleep(UI_SCROLL_SETTLE_MS)
        }
        return compactDiagnostics(values)
    }

    private fun visibleText(device: UiDevice, packageName: String): List<String> {
        return device.findObjects(By.pkg(packageName))
            .flatMap { node ->
                listOfNotNull(
                    runCatching { node.text?.trim() }.getOrNull(),
                    runCatching { node.contentDescription?.trim() }.getOrNull(),
                )
            }
            .filter { it.isNotBlank() }
            .distinct()
    }

    private fun compactText(values: Collection<String>): List<String> {
        return values.asSequence()
            .map { it.replace(Regex("\\s+"), " ").trim() }
            .filter { it.isNotBlank() && it.length <= MAX_CAPTURED_TEXT_LENGTH }
            .distinct()
            .take(MAX_CAPTURED_TEXT_LINES)
            .toList()
    }

    private fun compactDiagnostics(values: Collection<String>): List<String> {
        val diagnostics = Regex(
            "(?i)(IR demo started|IR backend|Pipeline completed|Runtime:|tokens(?:/s)?|" +
                "preloaded|skipped|fallback|warning|failed|error)",
        )
        return values.asSequence()
            .map { it.replace(Regex("\\s+"), " ").trim() }
            .flatMap { value ->
                diagnostics.findAll(value).map { match ->
                    value.substring(
                        (match.range.first - DIAGNOSTIC_CONTEXT_BEFORE).coerceAtLeast(0),
                        (match.range.last + DIAGNOSTIC_CONTEXT_AFTER + 1).coerceAtMost(value.length),
                    )
                }
            }
            .filter { it.isNotBlank() }
            .distinct()
            .take(MAX_CAPTURED_DIAGNOSTIC_LINES)
            .toList()
    }

    private fun meaningfulGeneratedMarkers(graph: JsonElement, queryText: String): List<String> {
        val strings = mutableListOf<String>()
        collectStrings(graph, strings)
        val normalizedQuery = comparableText(queryText)
        val salaryTerms = Regex("(?i)\\b(salary|hourly|wage|income|earnings|pay)\\b")
        val number = Regex("(?:[$£€₹]\\s*)?\\d[\\d,.]*(?:\\s*(?:hours?|weeks?|months?|years?))?")
        val phrases = strings.asSequence()
            .map(String::trim)
            .filter { it.length in 4..100 && salaryTerms.containsMatchIn(it) }
            .filterNot { normalizedQuery == comparableText(it) }
        val figures = strings.asSequence()
            .flatMap { number.findAll(it).map { match -> match.value.trim() } }
            .filter { it.length >= 2 && comparableText(it).any(Char::isDigit) }
            .filterNot { normalizedQuery.contains(comparableText(it)) }
        return (phrases + figures)
            .filter { it.isNotBlank() }
            .distinct()
            .take(40)
            .toList()
    }

    private fun collectStrings(element: JsonElement, output: MutableList<String>) {
        when {
            element.isJsonPrimitive && element.asJsonPrimitive.isString -> output += element.asString
            element.isJsonArray -> element.asJsonArray.forEach { collectStrings(it, output) }
            element.isJsonObject -> element.asJsonObject.entrySet().forEach { collectStrings(it.value, output) }
        }
    }

    private fun comparableText(value: String): String {
        return value.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "")
    }

    private fun runtimeReportsEnabledMtp(runtime: String): Boolean {
        return Regex("(?i)(\\+MTP\\b|MTP\\s*[=:]\\s*(?:true|on|enabled))").containsMatchIn(runtime)
    }

    private fun isSourceTextFallbackWarning(value: String): Boolean {
        val normalized = value.lowercase(Locale.US)
        if (normalized.contains("fallback disabled") ||
            normalized.contains("no source-text fallback") ||
            normalized.contains("no source text fallback") ||
            normalized.contains("source_text_fallback=false") ||
            normalized.contains("source-text fallback=false")
        ) {
            return false
        }
        return normalized.contains("source-text fallback") ||
            normalized.contains("source text fallback") ||
            normalized.contains("safe response fallback") ||
            normalized.contains("source_text_fallback")
    }

    private fun captureFailureUi(device: UiDevice, runDir: File) {
        runCatching { device.takeScreenshot(File(runDir, "failure_screenshot.png")) }
        runCatching { device.dumpWindowHierarchy(File(runDir, "failure_hierarchy.xml")) }
    }

    private fun assertIrDemoForeground(device: UiDevice, expectedPackage: String) {
        assertEquals(
            "IR Demo no longer foreground; device interaction interrupted UI validation",
            expectedPackage,
            device.currentPackageName,
        )
    }

    private fun waitForObject(
        device: UiDevice,
        selector: BySelector,
        timeoutMs: Long,
        label: String,
    ) = device.wait(androidx.test.uiautomator.Until.findObject(selector), timeoutMs)
        ?: error("Timed out waiting for $label")

    private fun booleanArgument(raw: String?, name: String, default: Boolean): Boolean {
        val normalized = raw?.trim()?.takeIf { it.isNotEmpty() } ?: return default
        return normalized.toBooleanStrictOrNull()
            ?: error("Instrumentation argument '$name' must be true or false, found '$raw'.")
    }

    private fun writeJson(file: File, value: Any, gson: com.google.gson.Gson) {
        file.writeText(gson.toJson(value) + "\n")
    }

    private companion object {
        const val MODEL_ID = "gemma4_e2b_a2ui_mobile"
        const val MODEL_FILE_NAME = "gemma4_e2b_a2ui_mobile.litertlm"
        const val FIRST_QUERY_ID = "q_001374"
        const val EXPRESS_FORMAT_ID = "a2ui_express_v1"
        const val ARTIFACT_ROOT = "trained_mobile_ir_demo"
        const val ARG_MTP = "mtp"
        const val ARG_KEEP_SELECTION = "keepSelection"
        const val PIPELINE_TIMEOUT_MS = 10 * 60_000L
        const val PIPELINE_POLL_MS = 500L
        const val UI_READY_TIMEOUT_MS = 10_000L
        const val UI_SETTLE_MS = 1_200L
        const val UI_SCROLL_SETTLE_MS = 180L
        const val MAX_RENDER_SCROLLS = 18
        const val MAX_DEBUG_SCROLLS = 16
        const val MAX_CAPTURED_TEXT_LENGTH = 240
        const val MAX_CAPTURED_TEXT_LINES = 160
        const val MAX_CAPTURED_DIAGNOSTIC_LINES = 80
        const val DIAGNOSTIC_CONTEXT_BEFORE = 48
        const val DIAGNOSTIC_CONTEXT_AFTER = 360
    }
}
