package com.samsung.genuicraft

import android.graphics.Rect
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.util.Xml
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.By
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.*
import com.samsung.genuicraft.sdk.provider.*
import java.io.File
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.*
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.xmlpull.v1.XmlPullParser

/** Device benchmark for the real published AAR, with durable per-case artifacts. */
@RunWith(AndroidJUnit4::class)
class GenUiSdkBixby50Test {
    @Test fun rendererOnlySendsActionToHost() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val document = GenUiCompiler.compile("<a2ui>\nroot=Column([title,link])\ntitle=Text(\"Renderer-only AAR test\")\nlink=Button(\"Open source\",onPress=openUrl(\"https://www.who.int/\"))\n</a2ui>")
        val received = java.util.concurrent.atomic.AtomicReference<GenUiAction?>()
        ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                activity.onSdkAction = { received.set(it) }
                activity.showDocument(GenUiCompiler.compile(document.a2uiJson))
            }
            val device = UiDevice.getInstance(instrumentation)
            assertTrue(device.wait(Until.hasObject(By.text("Open source")), 10000))
            device.findObject(By.text("Open source")).click()
            instrumentation.waitForIdleSync()
            Thread.sleep(300)
            val deadline = android.os.SystemClock.uptimeMillis() + 3_000L
            while (received.get() == null && android.os.SystemClock.uptimeMillis() < deadline) {
                Thread.sleep(50)
                instrumentation.waitForIdleSync()
            }
            if (received.get() == null) {
                val diagnostics =
                    File(instrumentation.targetContext.getExternalFilesDir(null), "sdk_benchmark/action_test")
                        .apply { mkdirs() }
                device.takeScreenshot(File(diagnostics, "failure.png"))
                device.dumpWindowHierarchy(File(diagnostics, "window.xml"))
            }
            val action = received.get()
            assertEquals("openUrl", action?.name)
            assertEquals("https://www.who.int/", action?.parameters?.get("url"))
        }
    }

    @Test fun converterSourceAttributionRendersAndOpensExactUrl() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val provider =
            object : GenUiProvider {
                override val id = "source-attribution-test"

                override suspend fun generate(prompt: GenUiPrompt) =
                    GenUiModelOutput(
                        text = "<a2ui>\nroot=Column([answer])\nanswer=Text(\"Ready. [7]\")\n</a2ui>",
                        runtime = "fake",
                    )
            }
        val result =
            GenUiConverter.withPrompt(provider, "strict test prompt").convert(
                GenUiRequest(
                    text = "Ready. [7]",
                    sources = listOf(GenUiSource("7", "https://www.who.int/", "WHO report \${HOME}")),
                ),
            )
        assertTrue(result is GenUiConversionResult.Success)
        val document = (result as GenUiConversionResult.Success).document
        val received = java.util.concurrent.atomic.AtomicReference<GenUiAction?>()

        ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                activity.setContentView(
                    GenUiView(activity).apply {
                        onAction = { received.set(it) }
                        render(document)
                    },
                )
            }
            val device = UiDevice.getInstance(instrumentation)
            assertTrue(device.wait(Until.hasObject(By.text("[7] WHO report \${HOME}")), 10_000))
            device.findObject(By.text("[7] WHO report \${HOME}")).click()
            instrumentation.waitForIdleSync()
            val deadline = android.os.SystemClock.uptimeMillis() + 3_000L
            while (received.get() == null && android.os.SystemClock.uptimeMillis() < deadline) {
                Thread.sleep(50)
                instrumentation.waitForIdleSync()
            }
            val action = received.get()
            assertEquals("openUrl", action?.name)
            assertEquals("https://www.who.int/", action?.parameters?.get("url"))
        }
    }

    @Test fun literalSourceTextIsRenderedWithoutInterpolation() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val value = "Data as of 2026-09-18.\nFor example: \${HOME}."
        val firstPath = "\$/price"
        val secondPath = "\$state.foo"
        val tag = "<a2ui>literal</a2ui>"
        val gson = GsonBuilder().disableHtmlEscaping().create()
        val program = "<a2ui>\nroot=Column([a,b,c])\na=Text(${gson.toJson(value)})\nb=List(items=${gson.toJson(listOf(firstPath, secondPath))})\nc=Table(columns=[\"Code\",\"Value\"],rows=${gson.toJson(listOf(listOf("\${HOME}", tag)))},preferredPresentation=\"table\")\n</a2ui>"
        val provider = object : GenUiProvider {
            override val id = "literal-test"
            override suspend fun generate(prompt: GenUiPrompt) = GenUiModelOutput(program, "fake")
        }
        val source = "$value\n\n- $firstPath\n- $secondPath\n\n| Code | Value |\n|---|---|\n| \${HOME} | $tag |"
        val result = GenUiConverter.withPrompt(provider, "literal contract").convert(GenUiRequest(source))
        assertTrue(result.toString(), result is GenUiConversionResult.Success)
        val replay = GenUiCompiler.compile((result as GenUiConversionResult.Success).document.a2uiJson)
        ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
            scenario.onActivity { activity -> activity.setContentView(GenUiView(activity).apply { render(replay) }) }
            val device = UiDevice.getInstance(instrumentation)
            listOf("Data as of 2026-09-18.", "For example:", "\${HOME}", firstPath, secondPath, tag, "Code", "Value").forEach { expected ->
                assertTrue("Literal text missing: $expected", device.wait(Until.hasObject(By.textContains(expected)), 10_000))
            }
            val output = File(instrumentation.targetContext.getExternalFilesDir(null), "sdk_benchmark/literal_test").apply { mkdirs() }
            device.takeScreenshot(File(output, "screen.png"))
            device.dumpWindowHierarchy(File(output, "window.xml"))
        }
    }

    @Test fun explicitTableKeepsHeadersAndScrollsHorizontally() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val document = GenUiCompiler.compile("""<a2ui>
root=Table(columns=["Reading","Temperature (°C)","Pressure (kPa)","Observation timestamp (UTC)","Sensor serial number"],rows=[["Inside","21","101.3","2026-09-18 03:00","SDK-SENSOR-54321"]],preferredPresentation="table")
</a2ui>""")
        ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                activity.setContentView(GenUiView(activity).apply {
                    val inset = (48 * resources.displayMetrics.density).toInt()
                    setPadding(16, inset, 16, inset)
                    setBackgroundColor(android.graphics.Color.WHITE)
                    render(GenUiCompiler.compile(document.a2uiJson))
                })
            }
            val device = UiDevice.getInstance(instrumentation)
            assertTrue(device.wait(Until.hasObject(By.text("Reading")), 10_000))
            assertTrue(device.wait(Until.hasObject(By.text("Temperature (°C)")), 10_000))
            val first = device.findObject(By.text("Inside"))
            assertTrue(first != null)
            val output = File(instrumentation.targetContext.getExternalFilesDir(null), "sdk_benchmark/table_scroll_test").apply { mkdirs() }
            device.takeScreenshot(File(output, "before.png"))
            val rowY = first.visibleBounds.centerY()
            repeat(3) {
                device.swipe(device.displayWidth * 9 / 10, rowY, device.displayWidth / 10, rowY, 35)
                Thread.sleep(200)
            }
            device.takeScreenshot(File(output, "after.png"))
            dumpFreshHierarchy(device, File(output, "window.xml"))
            assertTrue(device.wait(Until.hasObject(By.text("Sensor serial number")), 5_000))
            val last = device.findObject(By.text("SDK-SENSOR-54321"))
            assertTrue("Last column should be visible after horizontal scroll", last != null && last.visibleBounds.width() > 0)
        }
    }

    /** Replays durable captures only; no server or on-device model is constructed. */
    @Test fun replaySavedBixbyCorpus() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val args = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val mode = args.getString("replayMode", "json")!!
        require(mode in setOf("json", "express", "express_repair", "express_repair_only", "raw_model")) {
            "replayMode must be json, express, express_repair, express_repair_only, or raw_model."
        }
        val kind = when (mode) {
            "raw_model" -> "captured_model_revalidation"
            "express_repair" -> "renderer_repair_replay"
            "express_repair_only" -> "generated_dsl_repair_no_fallback"
            else -> "renderer_replay"
        }
        fun runName(value: String?, label: String): String {
            require(value != null && Regex("[A-Za-z0-9_-]{1,100}").matches(value)) {
                "$label must contain only letters, digits, underscores or hyphens (1..100 characters)."
            }
            return value
        }
        val sourceRunId = runName(args.getString("sourceRunId"), "sourceRunId")
        val runId = runName(args.getString("runId", "${kind}_${System.currentTimeMillis()}"), "runId")
        require(sourceRunId != runId) { "Replay must use a new runId." }
        val benchmarkRoot = File(context.getExternalFilesDir(null), "sdk_benchmark")
        val sourceRun = File(benchmarkRoot, sourceRunId)
        require(sourceRun.isDirectory) { "Missing source run: ${sourceRun.absolutePath}" }
        val output = File(benchmarkRoot, runId)
        require(!output.exists() && output.mkdirs()) { "Replay output must be a new directory: ${output.absolutePath}" }
        val maxVertical = args.getString("maxVerticalSwipes", "12")!!.toInt().also {
            require(it in 0..30) { "maxVerticalSwipes must be between 0 and 30." }
        }
        val maxHorizontal = args.getString("maxHorizontalSwipes", "8")!!.toInt().also {
            require(it in 1..16) { "maxHorizontalSwipes must be between 1 and 16." }
        }
        val renderFontScale = args.getString("renderFontScale")?.toFloat()?.also {
            require(it in 0.8f..2.0f) { "renderFontScale must be between 0.8 and 2.0." }
        }
        val renderDark = args.getString("renderDark")?.toBooleanStrict()
        val selected = args.getString("cases", "")!!.split(',').map(String::trim).filter(String::isNotBlank).toSet()
        val corpusBytes = context.assets.open("genuicraft_bixby50.jsonl").use { it.readBytes() }
        val allRows = corpusBytes.toString(Charsets.UTF_8).lineSequence().filter(String::isNotBlank)
            .map { JsonParser.parseString(it).asJsonObject }.toList()
        require(selected.all { id -> allRows.any { it.get("id").asString == id } }) { "Unknown replay case ID in cases." }
        val rows = allRows.filter { selected.isEmpty() || it.get("id").asString in selected }
        require(rows.isNotEmpty()) { "No replay cases selected." }
        val originalReports = if (mode == "raw_model") {
            val reportsFile = File(sourceRun, "results.json")
            require(reportsFile.isFile) { "raw_model replay requires the original results.json." }
            JsonParser.parseString(reportsFile.readText()).asJsonArray.associate { value ->
                val report = value.asJsonObject
                report.get("id").asString to report
            }
        } else emptyMap()
        val gson = GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create()
        File(output, "replay_config.json").writeText(gson.toJson(mapOf(
            "kind" to kind, "replayMode" to mode, "sourceRunId" to sourceRunId, "runId" to runId,
            "cases" to rows.map { it.get("id").asString }, "modelCalls" to 0,
            "inferenceEvaluated" to false, "maxVerticalSwipes" to maxVertical,
            "maxHorizontalSwipesPerTable" to maxHorizontal,
            "corpusSha256" to replaySha256(corpusBytes),
            "hierarchyPolicy" to "Clear UiAutomation cache on API 34+, refresh nodes, then dump.",
            "columnCheck" to "Visible text header or representative cell; content descriptions are excluded.",
            "renderFontScale" to renderFontScale, "renderDark" to renderDark,
        )))
        val device = UiDevice.getInstance(instrumentation)
        val reports = mutableListOf<JsonObject>()
        val repairCounts = mutableMapOf<String, Int>()
        var failures = 0
        var repairRejected = 0
        var renderFailures = 0
        var renderedWithCoverageIssues = 0
        var sourceIntegrityAccepted = 0
        var sourceIntegrityRejected = 0
        var sourceIntegrityNotEvaluated = 0
        ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
            for ((caseIndex, row) in rows.withIndex()) {
                val started = System.nanoTime()
                val caseId = row.get("id").asString
                val caseDir = File(output, caseId).apply { check(mkdirs()) }
                val sourceCase = File(sourceRun, caseId)
                val report = JsonObject().apply {
                    addProperty("id", caseId); addProperty("kind", kind); addProperty("replayMode", mode)
                    addProperty("sourceRunId", sourceRunId); addProperty("modelCalls", 0)
                    addProperty("inferenceEvaluated", false); addProperty("capturedProviderCalls", 0)
                }
                try {
                    val originalJsonFile = File(sourceCase, "output.a2ui.json")
                    val originalJsonBytes = originalJsonFile.takeIf(File::isFile)?.readBytes()
                    if (originalJsonBytes != null) {
                        val bytes = originalJsonBytes
                        File(caseDir, "source.output.a2ui.json").writeBytes(bytes)
                        report.addProperty("sourceJsonSha256", replaySha256(bytes))
                    }
                    val document = if (mode == "express" || mode == "express_repair" || mode == "express_repair_only") {
                        val expressFile = File(sourceCase, "output.express")
                        require(expressFile.isFile) { "Missing saved output.express for $caseId." }
                        val expressBytes = expressFile.readBytes()
                        File(caseDir, "source.output.express").writeBytes(expressBytes)
                        report.addProperty("sourceArtifact", "${sourceRunId}/${caseId}/output.express")
                        report.addProperty("sourceExpressSha256", replaySha256(expressBytes))
                        if (mode == "express") {
                            GenUiCompiler.compile(expressBytes.toString(Charsets.UTF_8))
                        } else if (mode == "express_repair") {
                            val outcome = GenUiCompiler.compileWithRepair(
                                expressBytes.toString(Charsets.UTF_8),
                                row.get("text").asString,
                            )
                            report.addProperty("repairKind", outcome.repairKind.name)
                            report.add("repairDiagnostics", gson.toJsonTree(outcome.diagnostics))
                            repairCounts[outcome.repairKind.name] = (repairCounts[outcome.repairKind.name] ?: 0) + 1
                            File(caseDir, "recovered.output.express").writeText(outcome.document.express)
                            File(caseDir, "recovered.output.a2ui.json").writeText(outcome.document.a2uiJson)
                            report.addProperty("recoveredExpressSha256", replaySha256(outcome.document.express.toByteArray()))
                            report.addProperty("recoveredJsonSha256", replaySha256(outcome.document.a2uiJson.toByteArray()))
                            outcome.document
                        } else {
                            val rawExpress = expressBytes.toString(Charsets.UTF_8)
                            val outputOnly = runCatching {
                                GenUiCompiler.compileWithRepair(
                                    input = rawExpress,
                                    allowSourceTextFallback = false,
                                    allowGeneratedDslRepair = true,
                                )
                            }
                            val outcome = outputOnly.getOrElse { failure ->
                                report.addProperty("generatedDslRepairAccepted", false)
                                report.addProperty("sourceIntegrityEvaluated", false)
                                sourceIntegrityNotEvaluated++
                                throw GeneratedDslRepairRejected(failure.message ?: failure.javaClass.simpleName)
                            }
                            require(outcome.repairKind != GenUiRepairKind.SOURCE_TEXT_FALLBACK) {
                                "No-fallback replay received SOURCE_TEXT_FALLBACK."
                            }
                            report.addProperty("generatedDslRepairAccepted", true)
                            report.addProperty("repairKind", outcome.repairKind.name)
                            report.add("repairDiagnostics", gson.toJsonTree(outcome.diagnostics))
                            repairCounts[outcome.repairKind.name] = (repairCounts[outcome.repairKind.name] ?: 0) + 1
                            val sourceAudit = runCatching {
                                GenUiCompiler.compileWithRepair(
                                    input = rawExpress,
                                    sourceText = row.get("text").asString,
                                    allowSourceTextFallback = false,
                                    allowGeneratedDslRepair = true,
                                )
                            }
                            report.addProperty("sourceIntegrityEvaluated", true)
                            report.addProperty("sourceIntegrityAccepted", sourceAudit.isSuccess)
                            if (sourceAudit.isSuccess) {
                                sourceIntegrityAccepted++
                                report.add("sourceIntegrityDiagnostics", gson.toJsonTree(sourceAudit.getOrThrow().diagnostics))
                            } else {
                                sourceIntegrityRejected++
                                val message = sourceAudit.exceptionOrNull()?.message.orEmpty()
                                File(caseDir, "source_integrity_failure.txt").writeText(message)
                                report.addProperty("sourceIntegrityFailure", message)
                            }
                            File(caseDir, "recovered.output.express").writeText(outcome.document.express)
                            File(caseDir, "recovered.output.a2ui.json").writeText(outcome.document.a2uiJson)
                            report.addProperty("recoveredExpressSha256", replaySha256(outcome.document.express.toByteArray()))
                            report.addProperty("recoveredJsonSha256", replaySha256(outcome.document.a2uiJson.toByteArray()))
                            outcome.document
                        }
                    } else if (mode == "json") {
                        require(originalJsonFile.isFile) { "Missing saved output.a2ui.json for $caseId." }
                        report.addProperty("sourceArtifact", "${sourceRunId}/${caseId}/output.a2ui.json")
                        GenUiCompiler.compile(requireNotNull(originalJsonBytes).toString(Charsets.UTF_8))
                    } else {
                        val originalReport = requireNotNull(originalReports[caseId]) { "Missing original result for $caseId." }
                        require(originalReport.get("status")?.asString == "success") {
                            "Captured-model replay requires an original successful result for $caseId."
                        }
                        require(originalReport.get("provider")?.asString == "gemma4_e2b") {
                            "raw_model replay requires a captured gemma4_e2b result."
                        }
                        val successfulAttempt = originalReport.get("attempts")?.asInt ?: error("Missing original attempts count.")
                        require(successfulAttempt in 1..3) { "Invalid original attempts count: $successfulAttempt" }
                        val capturedFile = File(sourceCase, "attempt_$successfulAttempt.txt")
                        require(capturedFile.isFile) { "Missing successful capture ${capturedFile.name}." }
                        val capturedBytes = capturedFile.readBytes()
                        val capturedText = capturedBytes.toString(Charsets.UTF_8)
                        File(caseDir, "source.attempt_$successfulAttempt.txt").writeBytes(capturedBytes)
                        report.addProperty("sourceArtifact", "$sourceRunId/$caseId/${capturedFile.name}")
                        report.addProperty("sourceRawSha256", replaySha256(capturedBytes))
                        report.addProperty("sourceSuccessfulAttempt", successfulAttempt)
                        var capturedCalls = 0
                        val capturedProvider = object : GenUiProvider {
                            override val id = "gemma4_e2b"
                            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
                                check(++capturedCalls == 1) { "Captured replay cannot request a repair or additional output." }
                                return GenUiModelOutput(capturedText, "captured_model_revalidation; no inference")
                            }
                        }
                        val request = GenUiRequest(row.get("text").asString, row.get("query")?.asString)
                        File(caseDir, "revalidation_request.json").writeText(gson.toJson(request))
                        val conversion = GenUiConverter(context, capturedProvider, ConversionOptions(maxRepairAttempts = 0)).convert(request)
                        report.addProperty("capturedProviderCalls", capturedCalls)
                        val revalidated = when (conversion) {
                            is GenUiConversionResult.Success -> conversion.document
                            is GenUiConversionResult.Failure -> error("Captured output failed current validation: ${conversion.message}")
                        }
                        File(caseDir, "revalidated.output.express").writeText(revalidated.express)
                        File(caseDir, "revalidated.output.a2ui.json").writeText(revalidated.a2uiJson)
                        GenUiCompiler.compile(revalidated.a2uiJson)
                    }
                    val tables = replayTableTargets(document.a2uiJson)
                    val seenHeaders = tables.map { mutableSetOf<Int>() }
                    val seenCells = tables.map { mutableSetOf<Int>() }
                    val sweptTables = mutableSetOf<Int>()
                    val horizontalTables = mutableSetOf<Int>()
                    val captures = mutableListOf<Map<String, Any>>()
                    val issues = mutableListOf<String>()
                    val viewport = AtomicReference(Rect())
                    val replayView = AtomicReference<GenUiView>()
                    scenario.onActivity { activity ->
                        val visualConfiguration = android.content.res.Configuration(activity.resources.configuration).apply {
                            renderFontScale?.let { fontScale = it }
                            renderDark?.let { dark ->
                                uiMode = (uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK.inv()) or
                                    (if (dark) android.content.res.Configuration.UI_MODE_NIGHT_YES else android.content.res.Configuration.UI_MODE_NIGHT_NO)
                            }
                        }
                        val viewContext = if (renderFontScale == null && renderDark == null) activity
                            else activity.createConfigurationContext(visualConfiguration)
                        val view = GenUiView(viewContext).apply {
                            val inset = (48 * resources.displayMetrics.density).toInt()
                            setPadding(16, inset, 16, inset)
                            setBackgroundColor(android.graphics.Color.WHITE)
                            render(document)
                        }
                        report.addProperty("renderFontScale", view.resources.configuration.fontScale)
                        report.addProperty("renderDark", view.resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK == android.content.res.Configuration.UI_MODE_NIGHT_YES)
                        replayView.set(view)
                        activity.setContentView(view)
                        renderDark?.let { dark ->
                            androidx.core.view.WindowCompat.getInsetsController(activity.window, view).apply {
                                isAppearanceLightStatusBars = !dark
                                isAppearanceLightNavigationBars = !dark
                            }
                        }
                    }
                    instrumentation.waitForIdleSync()
                    Thread.sleep(500)
                    scenario.onActivity {
                        val view = replayView.get()
                        val location = IntArray(2)
                        view.getLocationOnScreen(location)
                        viewport.set(Rect(location[0] + view.paddingLeft, location[1] + view.paddingTop,
                            location[0] + view.width - view.paddingRight, location[1] + view.height - view.paddingBottom))
                    }
                    require(viewport.get().width() > 100 && viewport.get().height() > 100) { "Replay view has no usable viewport." }
                    fun capture(label: String): List<ReplayNode> {
                        instrumentation.waitForIdleSync()
                        val screenshot = device.takeScreenshot(File(caseDir, "$label.png"))
                        val xml = File(caseDir, "$label.xml")
                        dumpFreshHierarchy(device, xml)
                        val nodes = replayNodes(xml.readText(), context.packageName, viewport.get())
                        if (!screenshot) issues += "Screenshot failed: $label"
                        if (nodes.any { it.text.contains("Unable to render GenUI") }) issues += "Renderer reported an error at $label."
                        val texts = nodes.map { replayComparable(it.text) }.filter(String::isNotBlank)
                        tables.forEachIndexed { tableIndex, table ->
                            table.columns.indices.forEach { column ->
                                if (texts.any { replayContains(it, table.columns[column]) }) seenHeaders[tableIndex] += column
                                if (texts.any { replayContains(it, table.probes[column]) }) seenCells[tableIndex] += column
                            }
                        }
                        captures += mapOf("name" to label, "screenshot" to screenshot, "visibleTextNodes" to texts.size)
                        return nodes
                    }
                    fun fingerprint(nodes: List<ReplayNode>) = nodes.filter { it.text.isNotBlank() }
                        .joinToString("\n") { "${it.text}|${it.bounds.flattenToString()}" }
                    fun swipeHorizontal(bounds: Rect, y: Int, towardsEnd: Boolean): Boolean {
                        val margin = (bounds.width() / 8).coerceAtLeast(24)
                        val left = bounds.left + margin
                        val right = bounds.right - margin
                        if (right - left < 40) return false
                        repeat(3) {
                            val ok = if (towardsEnd) {
                                device.swipe(right, y, left, y, 35)
                            } else {
                                device.swipe(left, y, right, y, 35)
                            }
                            Thread.sleep(250)
                            if (ok) return true
                            instrumentation.waitForIdleSync()
                        }
                        return false
                    }
                    fun sweepVisibleTables(input: List<ReplayNode>): List<ReplayNode> {
                        var current = input
                        tables.forEachIndexed { tableIndex, table ->
                            if (tableIndex in sweptTables) return@forEachIndexed
                            // Use a rendered label/cell to locate the table, never a fixed screen y.
                            val anchors = (table.columns + table.probes).filter(String::isNotBlank)
                            val anchoredScroller = anchors.asSequence().flatMap { anchor ->
                                current.asSequence().filter { replayContains(replayComparable(it.text), anchor) }
                            }.mapNotNull { anchor ->
                                val y = anchor.bounds.centerY()
                                current.firstOrNull {
                                    it.className == "android.widget.HorizontalScrollView" &&
                                        y > it.bounds.top + 2 && y < it.bounds.bottom - 2
                                }?.let { it.bounds to y }
                            }.firstOrNull() ?: return@forEachIndexed
                            horizontalTables += tableIndex
                            sweptTables += tableIndex
                            val (bounds, y) = anchoredScroller
                            var performed = 0
                            for (step in 1..maxHorizontal) {
                                val before = fingerprint(current)
                                if (!swipeHorizontal(bounds, y, towardsEnd = true)) {
                                    issues += "Horizontal gesture failed for table ${table.id}."
                                    break
                                }
                                performed++
                                current = capture("table_${tableIndex + 1}_horizontal_$step")
                                val covered = seenHeaders[tableIndex] + seenCells[tableIndex]
                                if (covered.size == table.columns.size || fingerprint(current) == before) break
                            }
                            repeat(performed) { swipeHorizontal(bounds, y, towardsEnd = false) }
                            if (performed > 0) current = capture("table_${tableIndex + 1}_reset")
                        }
                        return current
                    }
                    var current = capture("initial")
                    require(current.any { it.text.isNotBlank() }) { "Replay produced no visible text." }
                    var verticalCount = 0
                    var endObserved = false
                    for (step in 0..maxVertical) {
                        current = sweepVisibleTables(current)
                        if (step == maxVertical) break
                        val before = fingerprint(current)
                        val bounds = viewport.get()
                        val x = bounds.centerX()
                        require(device.swipe(x, bounds.bottom - bounds.height() / 10,
                            x, bounds.top + bounds.height() / 4, 35)) { "Vertical replay gesture failed." }
                        verticalCount++
                        Thread.sleep(300)
                        current = capture("vertical_$verticalCount")
                        if (fingerprint(current) == before) { endObserved = true; break }
                    }
                    val tableReports = tables.mapIndexed { index, table ->
                        val missing = table.columns.indices.filter { it !in seenHeaders[index] && it !in seenCells[index] }
                        if (missing.isNotEmpty()) issues += "Unobserved table columns (${table.id}): ${missing.map { table.columns[it] }}"
                        mapOf(
                            "id" to table.id, "columns" to table.columns,
                            "observedHeaders" to seenHeaders[index].sorted().map { table.columns[it] },
                            "observedRepresentativeCells" to seenCells[index].sorted().map { table.probes[it] },
                            "missingColumns" to missing.map { table.columns[it] },
                            "horizontalScrollObserved" to (index in horizontalTables),
                        )
                    }
                    report.add("tables", gson.toJsonTree(tableReports)); report.add("captures", gson.toJsonTree(captures))
                    report.addProperty("verticalSwipes", verticalCount); report.addProperty("verticalEndObserved", endObserved)
                    report.addProperty("verticalLimitReached", !endObserved && verticalCount == maxVertical)
                    report.add("issues", gson.toJsonTree(issues.distinct()))
                    val fatalRenderIssue = issues.any {
                        it.startsWith("Screenshot failed:") || it.startsWith("Renderer reported an error")
                    }
                    report.addProperty(
                        "status",
                        if (issues.isEmpty() || (mode == "express_repair_only" && !fatalRenderIssue)) "rendered" else "render_failure",
                    )
                } catch (failure: Exception) {
                    val rejected = failure is GeneratedDslRepairRejected
                    report.addProperty("status", if (rejected) "repair_rejected" else "replay_failure")
                    report.addProperty("message", failure.message ?: failure.javaClass.simpleName)
                    if (!rejected) {
                        runCatching { device.takeScreenshot(File(caseDir, "failure.png")) }
                        runCatching { dumpFreshHierarchy(device, File(caseDir, "failure.xml")) }
                    }
                }
                report.addProperty("replayElapsedMs", (System.nanoTime() - started) / 1_000_000)
                if (report.get("status").asString != "rendered") {
                    failures++
                    if (report.get("status").asString == "repair_rejected") repairRejected++ else renderFailures++
                } else if (report.getAsJsonArray("issues")?.size()?.let { it > 0 } == true) {
                    renderedWithCoverageIssues++
                }
                reports += report
                File(caseDir, "replay_result.json").writeText(gson.toJson(report))
                File(output, "replay_results.json").writeText(gson.toJson(reports))
                instrumentation.sendStatus(0, Bundle().apply {
                    putString("stream", "$caseId $kind ${report.get("status").asString} (${caseIndex + 1}/${rows.size}); live model calls=0\n")
                })
            }
        }
        File(output, "replay_summary.json").writeText(gson.toJson(mapOf(
            "kind" to kind, "replayMode" to mode, "runId" to runId, "sourceRunId" to sourceRunId,
            "total" to rows.size, "rendered" to rows.size - failures, "failed" to failures,
            "repairRejected" to repairRejected, "renderFailures" to renderFailures,
            "renderedWithCoverageIssues" to renderedWithCoverageIssues,
            "modelCalls" to 0, "inferenceEvaluated" to false, "repairCounts" to repairCounts.toSortedMap(),
            "sourceIntegrityAccepted" to sourceIntegrityAccepted,
            "sourceIntegrityRejected" to sourceIntegrityRejected,
            "sourceIntegrityNotEvaluated" to sourceIntegrityNotEvaluated,
            "sourceTextFallbacks" to (repairCounts[GenUiRepairKind.SOURCE_TEXT_FALLBACK.name] ?: 0),
        )))
        if (mode == "express_repair_only") {
            assertEquals("No source-text fallback is allowed in generated-DSL replay.", 0,
                repairCounts[GenUiRepairKind.SOURCE_TEXT_FALLBACK.name] ?: 0)
            assertEquals("Accepted generated-DSL documents must render without renderer failures.", 0, renderFailures)
        } else {
            assertTrue("$failures/${rows.size} replay checks failed; no inference performed; artifacts: ${output.absolutePath}", failures == 0)
        }
    }

    private class GeneratedDslRepairRejected(message: String) : IllegalArgumentException(message)
    private data class ReplayNode(val text: String, val className: String, val bounds: Rect)
    private data class ReplayTable(val id: String, val columns: List<String>, val probes: List<String>)

    /** Screenshots are fresh independently; accessibility caches can retain a previous case. */
    private fun dumpFreshHierarchy(device: UiDevice, file: File) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.waitForIdleSync()
        device.waitForIdle(1_000)
        val automation = instrumentation.uiAutomation
        if (Build.VERSION.SDK_INT >= 34) {
            check(automation.clearCache()) { "Unable to clear the accessibility cache before capture." }
        }
        val root = requireNotNull(automation.rootInActiveWindow) { "No accessibility root for hierarchy capture." }
        check(root.refresh()) { "Accessibility root became stale before hierarchy capture." }
        if (Build.VERSION.SDK_INT < 34) {
            // Older public APIs cannot clear the full cache. Refresh each fetched descendant instead.
            var remaining = 10_000
            fun refreshChildren(node: android.view.accessibility.AccessibilityNodeInfo, depth: Int) {
                check(depth <= 100 && --remaining >= 0) { "Accessibility hierarchy exceeds capture limits." }
                for (index in 0 until node.childCount) {
                    val child = node.getChild(index) ?: continue
                    if (child.refresh()) refreshChildren(child, depth + 1)
                }
            }
            refreshChildren(root, 0)
        }
        device.dumpWindowHierarchy(file)
    }

    private fun replaySha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private fun replayComparable(value: String): String = value
        .removePrefix("\u001EGenUICraftLiteral:v1:")
        .replace(Regex("\\[\\d+(?:\\s*[,–-]\\s*\\d+)*]"), "")
        .replace(Regex("[*_`]"), "")
        .replace(Regex("[\\s\\p{Z}]+"), " ").trim().lowercase(Locale.ROOT)

    private fun replayContains(normalizedText: String, expected: String): Boolean {
        val comparable = replayComparable(expected)
        return comparable.isNotBlank() && normalizedText.contains(comparable)
    }

    private fun replayNodes(xml: String, packageName: String, viewport: Rect): List<ReplayNode> {
        val parser = Xml.newPullParser().apply { setInput(xml.reader()) }
        val nodes = mutableListOf<ReplayNode>()
        val boundsPattern = Regex("\\[(-?\\d+),(-?\\d+)]\\[(-?\\d+),(-?\\d+)]")
        while (parser.eventType != XmlPullParser.END_DOCUMENT) {
            if (parser.eventType == XmlPullParser.START_TAG && parser.name == "node" &&
                parser.getAttributeValue(null, "package") == packageName &&
                parser.getAttributeValue(null, "visible-to-user") != "false") {
                val match = boundsPattern.matchEntire(parser.getAttributeValue(null, "bounds").orEmpty())
                if (match != null) {
                    val numbers = match.groupValues.drop(1).map(String::toInt)
                    val bounds = Rect(numbers[0], numbers[1], numbers[2], numbers[3])
                    if (bounds.intersect(viewport) && bounds.width() > 8 && bounds.height() > 8) {
                        // Row/table content-descriptions can contain offscreen values: never count them.
                        nodes += ReplayNode(parser.getAttributeValue(null, "text").orEmpty(),
                            parser.getAttributeValue(null, "class").orEmpty(), bounds)
                    }
                }
            }
            parser.next()
        }
        return nodes
    }

    private fun replayTableTargets(json: String): List<ReplayTable> {
        val result = mutableListOf<ReplayTable>()
        fun string(value: JsonElement?): String = value?.takeIf { it.isJsonPrimitive }?.asString.orEmpty()
        fun visit(value: JsonElement) {
            when {
                value.isJsonArray -> value.asJsonArray.forEach(::visit)
                value.isJsonObject -> {
                    val node = value.asJsonObject
                    if (string(node.get("component")) == "Table") {
                        val columns = node.getAsJsonArray("columns")?.map { column ->
                            if (column.isJsonObject) {
                                val definition = column.asJsonObject
                                string(definition.get("label")).ifBlank { string(definition.get("key")) }
                            } else string(column)
                        }.orEmpty()
                        val keys = node.getAsJsonArray("columns")?.map { column ->
                            if (column.isJsonObject) string(column.asJsonObject.get("key")) else string(column)
                        }.orEmpty()
                        val rows = node.getAsJsonArray("rows")?.toList().orEmpty()
                        val probes = columns.indices.map { index ->
                            rows.firstNotNullOfOrNull { row ->
                                val cell = when {
                                    row.isJsonArray -> row.asJsonArray.takeIf { index < it.size() }?.get(index)
                                    row.isJsonObject -> row.asJsonObject.get(keys[index])
                                    else -> null
                                }
                                string(cell).takeIf(String::isNotBlank)
                            }.orEmpty()
                        }
                        result += ReplayTable(string(node.get("id")), columns, probes)
                    }
                    node.entrySet().forEach { visit(it.value) }
                }
            }
        }
        visit(JsonParser.parseString(json))
        return result
    }

    /** Experimental input-only scaffold; acceptance still uses the AAR's unchanged validators. */
    private fun sourceScaffoldPrompt(prompt: GenUiPrompt): GenUiPrompt {
        val jsonStart = prompt.user.indexOf("\n{").also { require(it >= 0) } + 1
        val data = JsonParser.parseReader(com.google.gson.stream.JsonReader(java.io.StringReader(prompt.user.substring(jsonStart)))).asJsonObject
        val scaffold = buildString {
            appendLine("<a2ui>")
            appendLine(data.get("root").asString)
            data.getAsJsonArray("blocks").forEach { value ->
                val block = value.asJsonObject
                append(block.get("element").asString).append('=')
                when (block.get("kind").asString) {
                    "heading" -> append("Text(").append(block.get("text")).append(",variant=\"heading\")")
                    "paragraph" -> append("Text(").append(block.get("text")).append(')')
                    "list" -> append("List(items=").append(block.get("items")).append(')')
                    "table" -> append("Table(columns=").append(block.get("columns"))
                        .append(",rows=").append(block.get("rows"))
                        .append(",domain=\"SELECT_DOMAIN\",preferredPresentation=\"SELECT_PRESENTATION\")")
                    "code" -> {
                        append("CodeBlock(code=").append(block.get("code"))
                        if (block.has("title")) append(",title=").append(block.get("title"))
                        append(')')
                    }
                    "divider" -> append("Divider()")
                    else -> error("Unknown source block kind")
                }
                appendLine()
            }
            appendLine("</a2ui>")
        }
        return prompt.copy(user = "${prompt.user}\nExpress scaffold (preserve all lines; replace table SELECT_DOMAIN and SELECT_PRESENTATION):\n$scaffold")
    }

    /** No inference: verify the experimental prompt transform against the real AAR contract. */
    @Test fun sourceScaffoldMaintainsSourceContract() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val sources = context.assets.open("genuicraft_bixby50.jsonl").bufferedReader().useLines { lines ->
            lines.filter(String::isNotBlank).map { JsonParser.parseString(it).asJsonObject.get("text").asString }.toList()
        } + listOf(
            "# Code\n\n```kotlin\nprintln(\"\${HOME}\")\n```\n\n---\n\nFinal caveat.",
            "A literal value is {\"key\":\"a=b <c>\"}; preserve \\u003d and @source.a.",
        )
        for (source in sources) {
            var calls = 0
            val fixtureProvider = object : GenUiProvider {
                override val id = "gemma4_e2b"
                override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
                    val effective = sourceScaffoldPrompt(prompt)
                    val scaffold = effective.user.substringAfterLast("Express scaffold (preserve all lines; replace table SELECT_DOMAIN and SELECT_PRESENTATION):\n")
                        .replace("\"SELECT_DOMAIN\"", "\"generic\"")
                        .replace("\"SELECT_PRESENTATION\"", "\"table\"")
                    // Force one repair to verify parsing also ignores the appended previous output.
                    return GenUiModelOutput(if (++calls == 1) "invalid test fixture" else scaffold, "deterministic scaffold contract fixture; no inference")
                }
            }
            val result = GenUiConverter(context, fixtureProvider).convert(GenUiRequest(source))
            assertTrue(result.toString(), result is GenUiConversionResult.Success)
            assertEquals(2, (result as GenUiConversionResult.Success).attempts)
        }
    }

    @Test fun convertAndRenderBixbyCorpus() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val args = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val providerName = args.getString("provider", "gemma")!!
        require(providerName == "gemma") { "Only the on-device Gemma provider is supported." }
        val enableMetrics = args.getString("enableMetrics", "false") == "true"
        val runId = args.getString("runId", "${providerName}_${System.currentTimeMillis()}")!!.replace(Regex("[^A-Za-z0-9_-]"), "_")
        val output = File(context.getExternalFilesDir(null), "sdk_benchmark/$runId").apply { mkdirs() }
        val gson = GsonBuilder().setPrettyPrinting().create()
        val selected = args.getString("cases", "")!!.split(',').filter(String::isNotBlank).toSet()
        val corpusPath = args.getString("corpusPath")?.takeIf(String::isNotBlank)
        val corpusBytes = if (corpusPath == null) context.assets.open("genuicraft_bixby50.jsonl").use { it.readBytes() }
            else File(corpusPath).readBytes()
        val allRows = corpusBytes.toString(Charsets.UTF_8).lineSequence().filter(String::isNotBlank)
            .map { JsonParser.parseString(it).asJsonObject }.toList()
        val allIds = allRows.map { it.get("id").asString }
        require(allIds.size == allIds.toSet().size) { "Duplicate corpus case IDs." }
        require(selected.all(allIds::contains)) { "Unknown benchmark case ID in cases." }
        val rows = allRows.filter { selected.isEmpty() || it.get("id").asString in selected }
        require(rows.isNotEmpty()) { "No benchmark cases selected." }
        val config = Gemma4Config(
            args.getString("modelPath", "")!!,
            args.getString("accelerator", "GPU")!!,
            enableSpeculativeDecoding = args.getString("mtp", "true") == "true",
            enableMetrics = enableMetrics,
        )
        val provider: GenUiProvider = Gemma4Provider(
            args.getString("thinkingBudget")?.let { config.copy(thinkingTokenBudget = it.toInt()) }
                ?: config,
        )
        var attemptDir: File? = null
        var attemptIndex = 0
        val attemptOutputs = mutableListOf<GenUiModelOutput>()
        val recordingProvider = object : GenUiProvider {
            override val id = provider.id
            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
                val effectivePrompt = if (args.getString("inputScaffold", "false") == "true") sourceScaffoldPrompt(prompt) else prompt
                if (effectivePrompt !== prompt || args.getString("recordInputs", "false") == "true") {
                    File(requireNotNull(attemptDir), "attempt_${attemptIndex + 1}_input.txt").writeText(effectivePrompt.user)
                }
                val generationStarted = System.nanoTime()
                val value = provider.generate(effectivePrompt)
                val providerCallMs = (System.nanoTime() - generationStarted) / 1_000_000
                attemptOutputs += value
                File(requireNotNull(attemptDir), "attempt_${++attemptIndex}.txt").writeText(value.text)
                File(requireNotNull(attemptDir), "attempt_${attemptIndex}_metrics.json").writeText(gson.toJson(mapOf(
                    "runtime" to value.runtime, "outputTokens" to value.outputTokens,
                    "inputTokens" to value.metrics?.inputTokens,
                    "decodeTokensPerSecond" to value.metrics?.decodeTokensPerSecond,
                    "providerCallMs" to providerCallMs,
                    "metricsEnabled" to enableMetrics,
                    "outputCharacters" to value.text.length,
                    "inputScaffoldCount" to (effectivePrompt.user.split("\nExpress scaffold (preserve all lines; replace table SELECT_DOMAIN and SELECT_PRESENTATION):\n").size - 1),
                )))
                return value
            }
        }
        val options = ConversionOptions(maxRepairAttempts = args.getString("repairs", "1")!!.toInt(), temperature=args.getString("temperature", "0.0")!!.toDouble())
        val promptPath = args.getString("promptPath")?.takeIf(String::isNotBlank)
        val promptBytes = if (promptPath == null) {
            context.assets.open("genuicraft/prompts/gemma.txt").use { it.readBytes() }
        } else File(promptPath).readBytes()
        val sourceBindings = if (promptPath == null) true
            else args.getString("sourceBindings", "false") == "true"
        File(output, "run_config.json").writeText(gson.toJson(mapOf(
            "provider" to providerName, "cases" to rows.map { it.get("id").asString },
            "maxRepairAttempts" to options.maxRepairAttempts, "temperature" to options.temperature,
            "corpusPath" to (corpusPath ?: "asset:genuicraft_bixby50.jsonl"),
            "corpusSha256" to replaySha256(corpusBytes),
            "promptPath" to (promptPath ?: "aar_asset"), "promptSha256" to replaySha256(promptBytes),
            "sourceBindings" to sourceBindings,
            "inputScaffold" to (args.getString("inputScaffold", "false") == "true"),
            "recordInputs" to (args.getString("recordInputs", "false") == "true"),
            "metricsEnabled" to enableMetrics,
            "deviceModel" to Build.MODEL, "deviceHardware" to Build.HARDWARE,
        ) + mapOf(
            "accelerator" to args.getString("accelerator", "GPU"),
            "mtp" to (args.getString("mtp", "true") == "true"),
            "thinkingEnabled" to true, "thinkingBudgetOverride" to args.getString("thinkingBudget"),
        )))
        val converter = if (promptPath.isNullOrBlank()) GenUiConverter(context, recordingProvider, options)
            else GenUiConverter.withPrompt(recordingProvider, promptBytes.toString(Charsets.UTF_8), options,
                useSourceBindings = sourceBindings)
        val results = mutableListOf<JsonObject>()
        val device = UiDevice.getInstance(instrumentation)
        var failures = 0
        try {
            ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
                for ((index, row) in rows.withIndex()) {
                    val caseId = row.get("id").asString
                    val caseDir = File(output, caseId).apply { mkdirs() }
                    attemptDir = caseDir
                    attemptIndex = 0
                    attemptOutputs.clear()
                    val baselinePss = Debug.getPss()
                    val peakPss = AtomicLong(baselinePss)
                    val sampler = launch(Dispatchers.IO) {
                        while (isActive) { peakPss.updateAndGet { maxOf(it, Debug.getPss()) }; delay(250) }
                    }
                    val started = System.nanoTime()
                    val conversion = try {
                        withTimeout(args.getString("caseTimeoutMs", "600000")!!.toLong()) {
                            converter.convert(GenUiRequest(row.get("text").asString, row.get("query")?.asString))
                        }
                    } catch (failure: Exception) {
                        GenUiConversionResult.Failure(failure.message ?: failure.javaClass.simpleName, provider.id, (System.nanoTime() - started) / 1_000_000, 0)
                    } finally { sampler.cancelAndJoin() }
                    val report = JsonObject().apply {
                        addProperty("id", caseId); addProperty("provider", provider.id)
                        addProperty("firstInRun", index == 0); addProperty("baselinePssKb", baselinePss)
                        addProperty("peakPssKb", peakPss.get()); addProperty("usedFallback", false)
                    }
                    when (conversion) {
                        is GenUiConversionResult.Success -> {
                            File(caseDir, "output.express").writeText(conversion.document.express)
                            File(caseDir, "output.a2ui.json").writeText(conversion.document.a2uiJson)
                            // Replay only JSON: a second model call is neither needed nor allowed here.
                            val replay = GenUiCompiler.compile(conversion.document.a2uiJson)
                            val renderStart = System.nanoTime()
                            scenario.onActivity {
                                it.showDocument(replay, "$caseId · ${provider.id} · ${conversion.elapsedMs} ms")
                                if (enableMetrics) it.showGenerationMetrics(attemptOutputs.toList(), conversion.elapsedMs)
                            }
                            instrumentation.waitForIdleSync()
                            delay(700)
                            report.addProperty("renderWaitMs", (System.nanoTime() - renderStart) / 1_000_000)
                            val screenshotOk = device.takeScreenshot(File(caseDir, "screen.png"))
                            report.addProperty("screenshot", screenshotOk)
                            dumpFreshHierarchy(device, File(caseDir, "window.xml"))
                            report.addProperty("hierarchyCacheRefreshed", true)
                            val hierarchy = File(caseDir, "window.xml").readText()
                            val visibleTexts = Regex("text=\"([^\"]+)\"").findAll(hierarchy).map { it.groupValues[1] }.toList()
                            val renderOk = screenshotOk && !hierarchy.contains("Unable to render GenUI") && visibleTexts.size > 3
                            if (!renderOk) failures++
                            report.addProperty("status", if (renderOk) "success" else "render_failure"); report.addProperty("elapsedMs", conversion.elapsedMs)
                            report.addProperty("visibleTextNodes", visibleTexts.size)
                            if (args.getString("captureScroll", "true") == "true") {
                                device.swipe(device.displayWidth / 2, device.displayHeight * 85 / 100, device.displayWidth / 2, device.displayHeight * 35 / 100, 35)
                                delay(350)
                                device.takeScreenshot(File(caseDir, "screen_scrolled.png"))
                            }
                            report.addProperty("attempts", conversion.attempts)
                            report.add("warnings", gson.toJsonTree(conversion.warnings))
                        }
                        is GenUiConversionResult.Failure -> {
                            failures++
                            report.addProperty("status", "failure"); report.addProperty("message", conversion.message)
                            report.addProperty("elapsedMs", conversion.elapsedMs); report.addProperty("attempts", conversion.attempts)
                            conversion.rawOutput?.let { File(caseDir, "invalid_output.txt").writeText(it) }
                        }
                    }
                    results += report
                    File(caseDir, "result.json").writeText(gson.toJson(report))
                    File(output, "results.json").writeText(gson.toJson(results))
                    instrumentation.sendStatus(0, Bundle().apply { putString("stream", "$caseId ${report.get("status").asString} (${index + 1}/${rows.size})\n") })
                }
            }
        } finally { provider.closeAndAwait() }
        File(output, "summary.json").writeText(gson.toJson(mapOf("runId" to runId, "total" to rows.size, "success" to rows.size - failures, "failure" to failures, "usedFallback" to false)))
        assertTrue("$failures/${rows.size} conversions failed; artifacts: ${output.absolutePath}", failures == 0)
    }
}
