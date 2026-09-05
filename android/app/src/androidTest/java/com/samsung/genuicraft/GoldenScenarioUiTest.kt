package com.samsung.genuicraft

import android.app.Notification
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.util.Base64
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.BySelector
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import com.google.gson.GsonBuilder
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.samsung.genuicraft.mcp.McpSettings
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Executes one exact prompt through the visible GenUI Demo user flow on-device. */
@RunWith(AndroidJUnit4::class)
class GoldenScenarioUiTest {

    @Test
    fun runGoldenScenarioInUi() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext.applicationContext
        val device = UiDevice.getInstance(instrumentation)
        val runId = requireSafeId(arguments.getString(ARG_RUN_ID), "run_id")
        val caseId = requireSafeId(arguments.getString(ARG_CASE_ID), "case_id")
        val prompt = decodePrompt(arguments.getString(ARG_PROMPT_B64))
        val outputDir = File(
            context.getExternalFilesDir(null) ?: context.filesDir,
            "golden_scenarios/$runId",
        ).apply { check(mkdirs() || isDirectory) { "Could not create $absolutePath" } }
        val resultFile = File(outputDir, "$caseId.json")
        val screenshotFile = File(outputDir, "${caseId}_ui.png")
        val hierarchyFile = File(outputDir, "${caseId}_window.xml")
        val startedAtMs = System.currentTimeMillis()

        if (arguments.getString(ARG_CONFIGURE_ONLY).equals("true", ignoreCase = true)) {
            InferenceBackendSettings.setResponseProvider(
                context,
                InferenceBackendSettings.Provider.GEMINI,
            )
            InferenceBackendSettings.setIrProvider(
                context,
                InferenceBackendSettings.Provider.GEMINI,
            )
            InferenceBackendSettings.setGeminiApiMode(
                context,
                InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY,
            )
            GeminiModelSettings.setResponseModel(context, CONFIGURED_MODEL)
            GeminiModelSettings.setIrModel(context, CONFIGURED_MODEL)
            McpSettings.setEnabled(context, true)
            persistGoldenConfiguration(context)
            GeminiApiKeyProvider.refresh(context)
            val configured = baseJson(context, runId, caseId, prompt, startedAtMs).apply {
                addProperty("success", true)
                addProperty("outcome_stage", "configured")
                addProperty("configuration_only", true)
                addProperty(
                    "vertex_express_key_present",
                    GeminiApiKeyProvider.vertexExpressApiKey(context).isNotBlank(),
                )
            }
            configured.addProperty("finished_at_ms", System.currentTimeMillis())
            configured.addProperty("elapsed_ms", System.currentTimeMillis() - startedAtMs)
            writeAtomically(resultFile, configured)
            return@runBlocking
        }

        if (arguments.getString(ARG_HARVEST_ONLY).equals("true", ignoreCase = true)) {
            val history = GenUiAssistantHistoryStore.load(context).firstOrNull { it.query == prompt }
            val harvested = if (history != null) {
                successJson(
                    context = context,
                    runId = runId,
                    caseId = caseId,
                    prompt = prompt,
                    startedAtMs = history.createdAtMs,
                    history = history,
                    uiStatus = "Harvested from completed visible app run",
                    promptRoundTrip = true,
                    screenshotOk = false,
                )
            } else {
                failureJson(
                    context = context,
                    runId = runId,
                    caseId = caseId,
                    prompt = prompt,
                    startedAtMs = startedAtMs,
                    stage = "history",
                    error = "No existing app-history record matches the exact prompt.",
                    uiStatus = "harvest_failed",
                    promptRoundTrip = false,
                    screenshotOk = false,
                )
            }
            harvested.addProperty("finished_at_ms", System.currentTimeMillis())
            harvested.addProperty("elapsed_ms", 0L)
            writeAtomically(resultFile, harvested)
            return@runBlocking
        }

        var uiStatus = "not_started"
        var promptRoundTrip = false
        val resultJson = try {
            GeminiApiKeyProvider.refresh(context)
            device.wakeUp()
            device.executeShellCommand("wm dismiss-keyguard")
            launchFromHome(context, device)

            val input = waitForObject(
                device,
                By.clazz("android.widget.EditText").enabled(true),
                UI_READY_TIMEOUT_MS,
                "GenUI Demo prompt field",
            )
            input.click()
            input.setText(prompt)
            promptRoundTrip = input.text == prompt
            check(promptRoundTrip) {
                "Prompt round-trip mismatch. Expected ${prompt.length} chars, found ${input.text.length}."
            }
            device.pressBack()
            PipelineRunNotifier.clear(context)
            val send = waitForObject(
                device,
                By.desc(SEND_DESCRIPTION),
                UI_READY_TIMEOUT_MS,
                "Send prompt button",
            )
            send.click()
            uiStatus = waitForTerminalStatus(context, device, prompt, startedAtMs)
            device.waitForIdle()
            Thread.sleep(1_200L)

            val screenshotOk = device.takeScreenshot(screenshotFile)
            runCatching { device.dumpWindowHierarchy(hierarchyFile) }
            if (uiStatus == COMPLETE_STATUS || uiStatus == TERMINAL_CONTROLS_STATUS) {
                val history = waitForHistory(context, prompt, startedAtMs)
                if (history != null) {
                    successJson(
                        context = context,
                        runId = runId,
                        caseId = caseId,
                        prompt = prompt,
                        startedAtMs = startedAtMs,
                        history = history,
                        uiStatus = uiStatus,
                        promptRoundTrip = promptRoundTrip,
                        screenshotOk = screenshotOk,
                    )
                } else {
                    failureJson(
                        context = context,
                        runId = runId,
                        caseId = caseId,
                        prompt = prompt,
                        startedAtMs = startedAtMs,
                        stage = "ui",
                        error = "The visible app returned to Send without saving a completed history record; see the captured failure screen.",
                        uiStatus = uiStatus,
                        promptRoundTrip = promptRoundTrip,
                        screenshotOk = screenshotOk,
                    )
                }
            } else {
                failureJson(
                    context = context,
                    runId = runId,
                    caseId = caseId,
                    prompt = prompt,
                    startedAtMs = startedAtMs,
                    stage = "ui",
                    error = uiStatus,
                    uiStatus = uiStatus,
                    promptRoundTrip = promptRoundTrip,
                    screenshotOk = screenshotOk,
                )
            }
        } catch (throwable: Throwable) {
            runCatching { device.takeScreenshot(screenshotFile) }
            runCatching { device.dumpWindowHierarchy(hierarchyFile) }
            failureJson(
                context = context,
                runId = runId,
                caseId = caseId,
                prompt = prompt,
                startedAtMs = startedAtMs,
                stage = "ui_exception",
                error = "${throwable.javaClass.simpleName}: ${throwable.message.orEmpty()}".trim(),
                uiStatus = uiStatus,
                promptRoundTrip = promptRoundTrip,
                screenshotOk = screenshotFile.exists() && screenshotFile.length() > 0L,
            )
        }

        resultJson.addProperty("finished_at_ms", System.currentTimeMillis())
        resultJson.addProperty("elapsed_ms", System.currentTimeMillis() - startedAtMs)
        resultJson.addProperty("ui_screenshot_file", screenshotFile.name)
        resultJson.addProperty("ui_hierarchy_file", hierarchyFile.name)
        writeAtomically(resultFile, resultJson)
    }

    private fun launchFromHome(context: Context, device: UiDevice) {
        val launchIntent = context.packageManager.getLaunchIntentForPackage(context.packageName)
            ?: Intent(context, MainActivity::class.java)
        launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        context.startActivity(launchIntent)
        waitForObject(device, By.text(LAUNCHER_CARD_TEXT), UI_READY_TIMEOUT_MS, "GenUI Demo launcher card")
            .click()
        waitForObject(
            device,
            By.clazz("android.widget.EditText").enabled(true),
            UI_READY_TIMEOUT_MS,
            "GenUI Demo prompt field",
        )
    }

    private fun waitForTerminalStatus(
        context: Context,
        device: UiDevice,
        prompt: String,
        startedAtMs: Long,
    ): String {
        val deadline = System.currentTimeMillis() + GENERATION_TIMEOUT_MS
        var sawRunning = false
        while (System.currentTimeMillis() < deadline) {
            if (GenUiAssistantHistoryStore.load(context).any { item ->
                    item.query == prompt && item.createdAtMs >= startedAtMs
                }
            ) {
                return COMPLETE_STATUS
            }
            notificationTerminalStatus(context)?.let { return it }
            val stopVisible = device.hasObject(By.desc(STOP_DESCRIPTION))
            val sendVisible = device.hasObject(By.desc(SEND_DESCRIPTION))
            if (stopVisible || !sendVisible) {
                sawRunning = true
            }
            if (device.hasObject(By.text(COMPLETE_STATUS))) {
                return COMPLETE_STATUS
            }
            device.findObject(By.textContains("Failed:"))?.let { failure ->
                return failure.text.orEmpty().ifBlank { "Failed: unknown UI error" }
            }
            if (sawRunning && !stopVisible && sendVisible) {
                return TERMINAL_CONTROLS_STATUS
            }
            Thread.sleep(750L)
        }
        return if (sawRunning) {
            "Failed: timed out while generation was running"
        } else {
            "Failed: Send did not reach a terminal UI state"
        }
    }

    private fun notificationTerminalStatus(context: Context): String? {
        val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val active = runCatching {
            manager.activeNotifications.firstOrNull { it.id == PIPELINE_NOTIFICATION_ID }
        }.getOrNull() ?: return null
        val title = active.notification.extras
            .getCharSequence(Notification.EXTRA_TITLE)
            ?.toString()
            .orEmpty()
        val text = active.notification.extras
            .getCharSequence(Notification.EXTRA_TEXT)
            ?.toString()
            .orEmpty()
        return when (title) {
            "GenUICraft completed" -> COMPLETE_STATUS
            "GenUICraft failed" -> "Failed: ${text.ifBlank { "Generation failed." }}"
            else -> null
        }
    }

    private fun waitForHistory(
        context: Context,
        prompt: String,
        startedAtMs: Long,
    ): GenUiAssistantHistoryItem? {
        val deadline = System.currentTimeMillis() + HISTORY_TIMEOUT_MS
        while (System.currentTimeMillis() < deadline) {
            GenUiAssistantHistoryStore.load(context).firstOrNull { item ->
                item.query == prompt && item.createdAtMs >= startedAtMs
            }?.let { return it }
            Thread.sleep(250L)
        }
        return null
    }

    private fun waitForObject(
        device: UiDevice,
        selector: BySelector,
        timeoutMs: Long,
        label: String,
    ): UiObject2 {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            device.findObject(selector)?.let { return it }
            Thread.sleep(200L)
        }
        error("Timed out waiting for $label")
    }

    private fun successJson(
        context: Context,
        runId: String,
        caseId: String,
        prompt: String,
        startedAtMs: Long,
        history: GenUiAssistantHistoryItem,
        uiStatus: String,
        promptRoundTrip: Boolean,
        screenshotOk: Boolean,
    ): JsonObject = baseJson(context, runId, caseId, prompt, startedAtMs).apply {
        addProperty("success", true)
        addProperty("outcome_stage", "complete")
        addProperty("response_text", history.responseText)
        addProperty("genui_json", history.genUiJson)
        addProperty("used_fallback", history.usedFallback)
        add("warnings", stringsJson(history.warnings))
        addProperty("stage3_input_tokens", history.stage3InputTokens)
        addProperty("stage3_output_tokens", history.stage3OutputTokens)
        addProperty("app_history_created_at_ms", history.createdAtMs)
        addProperty("ui_status", uiStatus)
        addProperty("ui_prompt_round_trip", promptRoundTrip)
        addProperty("ui_screenshot_ok", screenshotOk)
    }

    private fun failureJson(
        context: Context,
        runId: String,
        caseId: String,
        prompt: String,
        startedAtMs: Long,
        stage: String,
        error: String,
        uiStatus: String,
        promptRoundTrip: Boolean,
        screenshotOk: Boolean,
    ): JsonObject = baseJson(context, runId, caseId, prompt, startedAtMs).apply {
        addProperty("success", false)
        addProperty("outcome_stage", stage)
        addProperty("error", error)
        addProperty("ui_status", uiStatus)
        addProperty("ui_prompt_round_trip", promptRoundTrip)
        addProperty("ui_screenshot_ok", screenshotOk)
    }

    private fun baseJson(
        context: Context,
        runId: String,
        caseId: String,
        prompt: String,
        startedAtMs: Long,
    ): JsonObject = JsonObject().apply {
        addProperty("schema_version", 2)
        addProperty("execution_mode", "visible_app_ui")
        addProperty("run_id", runId)
        addProperty("case_id", caseId)
        addProperty("prompt", prompt)
        addProperty("started_at_ms", startedAtMs)
        addProperty("response_model", GeminiModelSettings.getResponseModel(context))
        addProperty("ir_model", GeminiModelSettings.getIrModel(context))
        addProperty("mcp_enabled", McpSettings.isEnabled(context))
        addProperty(
            "response_provider",
            InferenceBackendSettings.getResponseProvider(context).name.lowercase(),
        )
        addProperty(
            "ir_provider",
            InferenceBackendSettings.getIrProvider(context).name.lowercase(),
        )
        addProperty(
            "gemini_api_mode",
            InferenceBackendSettings.getGeminiApiMode(context).name.lowercase(),
        )
    }

    private fun stringsJson(values: List<String>): JsonArray = JsonArray().apply {
        values.distinct().forEach { value -> add(value) }
    }

    /**
     * The public settings APIs intentionally use SharedPreferences.apply(). A configure-only
     * instrumentation process can terminate before those asynchronous writes reach disk, so the
     * next visible app launch may observe defaults. Commit the same test configuration before the
     * short-lived process exits to make the device setup deterministic.
     */
    private fun persistGoldenConfiguration(context: Context) {
        check(
            context.getSharedPreferences("inference_backend_settings", Context.MODE_PRIVATE)
                .edit()
                .putString("provider", InferenceBackendSettings.Provider.GEMINI.rawValue)
                .putString("response_provider", InferenceBackendSettings.Provider.GEMINI.rawValue)
                .putString("ir_provider", InferenceBackendSettings.Provider.GEMINI.rawValue)
                .putString(
                    "gemini_api_mode",
                    InferenceBackendSettings.GeminiApiMode.VERTEX_AI_EXPRESS_API_KEY.rawValue,
                )
                .commit(),
        ) { "Could not persist inference settings" }
        check(
            context.getSharedPreferences("gemini_model_settings", Context.MODE_PRIVATE)
                .edit()
                .putString("selected_model", CONFIGURED_MODEL)
                .putString("selected_response_model", CONFIGURED_MODEL)
                .putString("selected_ir_model", CONFIGURED_MODEL)
                .commit(),
        ) { "Could not persist Gemini model settings" }
        check(
            context.getSharedPreferences("mcp_settings", Context.MODE_PRIVATE)
                .edit()
                .putBoolean("mcp_enabled", true)
                .commit(),
        ) { "Could not persist MCP settings" }
    }

    private fun decodePrompt(encoded: String?): String {
        require(!encoded.isNullOrBlank()) { "prompt_b64 is required" }
        val bytes = Base64.decode(encoded, Base64.URL_SAFE or Base64.NO_PADDING or Base64.NO_WRAP)
        return bytes.toString(Charsets.UTF_8).also {
            require(it.isNotBlank()) { "Decoded prompt is empty" }
        }
    }

    private fun requireSafeId(value: String?, name: String): String {
        require(!value.isNullOrBlank()) { "$name is required" }
        require(SAFE_ID.matches(value)) { "$name contains unsupported characters" }
        return value
    }

    private fun writeAtomically(outputFile: File, json: JsonObject) {
        val temporary = File(outputFile.parentFile, ".${outputFile.name}.tmp")
        temporary.writeText(GSON.toJson(json) + "\n", Charsets.UTF_8)
        if (outputFile.exists()) {
            check(outputFile.delete()) { "Could not replace ${outputFile.absolutePath}" }
        }
        check(temporary.renameTo(outputFile)) { "Could not finalize ${outputFile.absolutePath}" }
    }

    companion object {
        private const val ARG_RUN_ID = "run_id"
        private const val ARG_CASE_ID = "case_id"
        private const val ARG_PROMPT_B64 = "prompt_b64"
        private const val ARG_HARVEST_ONLY = "harvest_only"
        private const val ARG_CONFIGURE_ONLY = "configure_only"
        private const val CONFIGURED_MODEL = "gemini-3.8-flash"
        private const val LAUNCHER_CARD_TEXT = "GenUI Demo"
        private const val SEND_DESCRIPTION = "Send prompt"
        private const val STOP_DESCRIPTION = "Stop generation"
        private const val COMPLETE_STATUS = "Completed. Rendered output is ready."
        private const val TERMINAL_CONTROLS_STATUS = "Terminal UI state: Send control restored"
        private const val UI_READY_TIMEOUT_MS = 30_000L
        private const val GENERATION_TIMEOUT_MS = 1_200_000L
        private const val HISTORY_TIMEOUT_MS = 10_000L
        private const val PIPELINE_NOTIFICATION_ID = 20041
        private val SAFE_ID = Regex("[A-Za-z0-9._-]+")
        private val GSON = GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create()
    }
}
