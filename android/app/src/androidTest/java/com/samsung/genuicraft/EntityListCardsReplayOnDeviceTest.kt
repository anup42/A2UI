package com.samsung.genuicraft

import android.content.Intent
import android.graphics.Color
import android.os.Build
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiConversionResult
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiRepairKind
import com.samsung.genuicraft.sdk.GenUiRequest
import com.samsung.genuicraft.sdk.GenUiTrainedConverter
import com.samsung.genuicraft.sdk.GenUiView
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Replays the exact first-attempt model bytes from the matched latest/previous comparison. */
@RunWith(AndroidJUnit4::class)
class EntityListCardsReplayOnDeviceTest {
    @Test
    fun savedModelOutputsKeepEntitiesInCardsAndNotesInAList() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val assets = instrumentation.context.assets
        val gson = GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create()
        val corpusBytes = assets.open("entity_list_replay/comparison_corpus.jsonl").use { it.readBytes() }
        assertEquals(CORPUS_SHA256, sha256(corpusBytes))
        val sourceById = corpusBytes.toString(Charsets.UTF_8).lineSequence()
            .filter(String::isNotBlank)
            .map { JsonParser.parseString(it).asJsonObject }
            .associateBy { it.get("id").asString }
        val requestedName = outputName()
        val output = File(requireNotNull(context.getExternalFilesDir(null)), "sdk_entity_list_replay/$requestedName/trained")
        require(!output.exists() && output.mkdirs()) { "Output directory must be new: ${output.absolutePath}" }
        File(output, "run_config.json").writeText(gson.toJson(mapOf(
            "kind" to "saved_raw_model_output_replay",
            "source" to "GenUICraft/validation/20261016_list_cards/runs",
            "modelCalls" to 0,
            "deviceModel" to Build.MODEL,
            "deviceProduct" to Build.PRODUCT,
            "deviceSdkInt" to Build.VERSION.SDK_INT,
            "corpusSha256" to CORPUS_SHA256,
            "cases" to CASES.map { "${it.arm}/${it.id}" },
        )) + "\n")

        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        val reports = mutableListOf<Map<String, Any?>>()
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            CASES.forEach { case ->
                val caseDir = File(output, "${case.arm}_${case.id}").apply { check(mkdirs()) }
                val rawBytes = assets.open("entity_list_replay/${case.arm}-${case.id}.express")
                    .use { it.readBytes() }
                assertEquals("Pinned raw output changed for ${case.arm}/${case.id}",
                    case.sha256, sha256(rawBytes))
                val raw = rawBytes.toString(Charsets.UTF_8)
                File(caseDir, "source.raw.express").writeBytes(rawBytes)
                val control = GenUiCompiler.compileWithRepair(
                    raw,
                    allowSourceTextFallback = false,
                    allowGeneratedDslRepair = true,
                )
                assertEquals("Unexpected compiler recovery for ${case.arm}/${case.id}",
                    case.repairKind, control.repairKind)
                File(caseDir, "compiler_control.a2ui.json").writeText(control.document.a2uiJson)
                val controlComponents = components(control.document.a2uiJson)
                if (case.arm == "latest" && case.id == "LIVE-FLIGHT") {
                    assertEquals("List", componentWithItem(controlComponents, "Akasa Air").get("component").asString)
                    assertEquals("List", componentWithItem(controlComponents, "Prices are *dynamic*").get("component").asString)
                }
                if (case.arm == "latest" && case.id == "BXP-010") {
                    assertEquals("List", componentWithItem(controlComponents, "OnePlus Nord Buds 4 Pro").get("component").asString)
                }

                var providerCalls = 0
                val provider = object : GenUiProvider {
                    override val id = "pinned_raw_${case.arm}_${case.id}"
                    override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
                        check(++providerCalls == 1) { "Replay requested another model output" }
                        return GenUiModelOutput(raw, "saved_first_attempt; no inference")
                    }
                }
                val source = sourceById.getValue(case.id)
                val conversion = GenUiTrainedConverter(
                    context,
                    provider,
                    allowSourceTextFallback = false,
                    allowGeneratedDslRepair = true,
                    requireSourceIntegrity = false,
                ).convert(GenUiRequest(source.get("text").asString, source.get("query").asString))
                assertTrue("SDK conversion failed for ${case.arm}/${case.id}: $conversion",
                    conversion is GenUiConversionResult.Success)
                val success = conversion as GenUiConversionResult.Success
                assertEquals(1, providerCalls)
                assertEquals(control.repairKind, success.repairKind)
                val finalJson = success.document.a2uiJson
                GenUiCompiler.compile(finalJson) // The presented document remains strict A2UI.
                File(caseDir, "presented.express").writeText(success.document.express)
                File(caseDir, "presented.a2ui.json").writeText(finalJson)
                val finalComponents = components(finalJson)
                assertPresentation(case, controlComponents, finalComponents, finalJson)

                scenario.onActivity { activity ->
                    activity.setContentView(GenUiView(activity).apply {
                        val inset = (48 * resources.displayMetrics.density).toInt()
                        setPadding(16, inset, 16, inset)
                        setBackgroundColor(Color.WHITE)
                        render(success.document)
                    })
                }
                instrumentation.waitForIdleSync()
                assertTrue(device.wait(Until.hasObject(By.pkg(context.packageName)), 10_000))
                val observed = linkedSetOf<String>()
                val descriptions = linkedSetOf<String>()
                capture(instrumentation, device, caseDir, "cards", observed, descriptions)
                val screenshots = mutableListOf("cards.png")
                if (case.arm == "latest" && case.id == "LIVE-FLIGHT") {
                    repeat(2) { swipeUp(device) }
                    capture(instrumentation, device, caseDir, "more_flights", observed, descriptions)
                    screenshots += "more_flights.png"
                    swipeUp(device)
                } else {
                    repeat(if (case.id == "LIVE-FLIGHT") 3 else 2) { swipeUp(device) }
                }
                capture(instrumentation, device, caseDir, "lower", observed, descriptions)
                screenshots += "lower.png"
                assertTrue("Expected rendered value ${case.visibleMarker} in ${case.arm}/${case.id}: $observed",
                    observed.any { case.visibleMarker in it })
                if (case.arm == "latest" && case.id in setOf("LIVE-FLIGHT", "BXP-010")) {
                    assertTrue("No table card row visible in ${case.arm}/${case.id}: $observed",
                        descriptions.any { case.visibleMarker in it && "₹" in it })
                }
                if (case.arm == "latest" && case.id == "LIVE-FLIGHT") {
                    assertNativeFlightCard(descriptions)
                }
                reports += mapOf(
                    "arm" to case.arm,
                    "id" to case.id,
                    "rawSha256" to case.sha256,
                    "repairKind" to success.repairKind.name,
                    "providerCalls" to providerCalls,
                    "warnings" to success.warnings,
                    "presentedJsonSha256" to sha256(finalJson.toByteArray(Charsets.UTF_8)),
                    "compilerControlComponents" to componentNames(controlComponents),
                    "presentedComponents" to componentNames(finalComponents),
                    "visibleMarker" to case.visibleMarker,
                    "visibleMarkerObserved" to observed.any { case.visibleMarker in it },
                    "cardRowDescriptionObserved" to descriptions.any { case.visibleMarker in it && "₹" in it },
                    "screenshots" to screenshots,
                )
                File(output, "results.json").writeText(gson.toJson(reports) + "\n")
            }
        }
        assertEquals("All six saved outputs must be replayed", CASES.size, reports.size)
    }

    @Test
    fun rendererOnlyPersistedFlightDocumentShowsCards() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val rawBytes = instrumentation.context.assets
            .open("entity_list_replay/latest-LIVE-FLIGHT.express").use { it.readBytes() }
        assertEquals(CASES.first().sha256, sha256(rawBytes))
        val rawDocument = GenUiCompiler.compile(rawBytes.toString(Charsets.UTF_8))
        val rawComponents = components(rawDocument.a2uiJson)
        assertEquals("List", componentWithItem(rawComponents, "Akasa Air").get("component").asString)
        assertEquals("List", componentWithItem(rawComponents, "Prices are *dynamic*").get("component").asString)
        val output = File(requireNotNull(context.getExternalFilesDir(null)),
            "sdk_entity_list_replay/${outputName()}/renderer_only")
        require(!output.exists() && output.mkdirs()) { "Output directory must be new: ${output.absolutePath}" }
        File(output, "source.raw.express").writeBytes(rawBytes)
        File(output, "persisted.a2ui.json").writeText(rawDocument.a2uiJson)

        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        val observed = linkedSetOf<String>()
        val descriptions = linkedSetOf<String>()
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                activity.setContentView(GenUiView(activity).apply {
                    val inset = (48 * resources.displayMetrics.density).toInt()
                    setPadding(16, inset, 16, inset)
                    setBackgroundColor(Color.WHITE)
                    render(rawDocument) // Exercises GenUiContent's persisted-document presentation path.
                })
            }
            instrumentation.waitForIdleSync()
            assertTrue(device.wait(Until.hasObject(By.pkg(context.packageName)), 10_000))
            capture(instrumentation, device, output, "cards", observed, descriptions)
            repeat(2) { swipeUp(device) }
            capture(instrumentation, device, output, "more_flights", observed, descriptions)
            swipeUp(device)
            capture(instrumentation, device, output, "fare_notes", observed, descriptions)
        }
        assertNativeFlightCard(descriptions)
        assertTrue("Fare notes disappeared in renderer-only replay: $observed",
            observed.any { "Prices are" in it && "dynamic" in it })
        File(output, "result.json").writeText(GsonBuilder().setPrettyPrinting().create().toJson(mapOf(
            "mode" to "renderer_only_persisted_document",
            "rawSha256" to sha256(rawBytes),
            "modelCalls" to 0,
            "rawComponent" to "List",
            "cardRowObserved" to true,
            "fareNotesObserved" to true,
            "screenshots" to listOf("cards.png", "more_flights.png", "fare_notes.png"),
        )) + "\n")
    }

    @Test
    fun plainFareNotesStayAList() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val notesOnly = """
            <a2ui>
            root=List(items=["Prices are dynamic and can change at checkout.[1]","Check the final fare before booking.","Keep your e-ticket available."])
            </a2ui>
        """.trimIndent()
        val control = GenUiCompiler.compileWithRepair(
            notesOnly,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = false,
        )
        assertEquals(GenUiRepairKind.NONE, control.repairKind)
        var calls = 0
        val provider = object : GenUiProvider {
            override val id = "plain_notes_control"
            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
                check(++calls == 1)
                return GenUiModelOutput(notesOnly, "fixture; no inference")
            }
        }
        val conversion = GenUiTrainedConverter(
            context, provider,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
            requireSourceIntegrity = false,
        ).convert(GenUiRequest("Prices are dynamic and can change at checkout.[1]"))
        assertTrue("Notes-only conversion failed: $conversion", conversion is GenUiConversionResult.Success)
        assertEquals(1, calls)
        val presented = (conversion as GenUiConversionResult.Success).document
        GenUiCompiler.compile(presented.a2uiJson)
        val controlList = components(control.document.a2uiJson).map { it.asJsonObject }
            .single { it.get("component").asString == "List" }
        val finalComponents = components(presented.a2uiJson).map { it.asJsonObject }
        val finalList = finalComponents.single { it.get("component").asString == "List" }
        assertEquals(controlList.getAsJsonArray("items"), finalList.getAsJsonArray("items"))
        assertTrue("Plain notes must not become an entity Table",
            finalComponents.none { it.get("component").asString == "Table" })
        assertTrue("Citation missing from notes", "[1]" in presented.a2uiJson)
    }

    private fun assertPresentation(
        case: ReplayCase,
        control: JsonArray,
        presented: JsonArray,
        finalJson: String,
    ) {
        val tables = presented.map { it.asJsonObject }
            .filter { it.get("component").asString == "Table" }
        assertEquals("Expected one entity table for ${case.arm}/${case.id}", 1, tables.size)
        val table = tables.single()
        if (case.id != "BXP-003") {
            assertEquals("cards", table.get("preferredPresentation").asString)
        }
        when (case.id) {
            "LIVE-FLIGHT" -> {
                listOf("Akasa Air", "Air India Express", "IndiGo", "ixigo",
                    "₹7,195", "₹7,298", "₹15,599", "₹22,268", "₹12,882").forEach { value ->
                    assertTrue("Lost flight detail $value in ${case.arm}", value in finalJson)
                }
                if (case.arm == "latest") {
                    val flightId = componentWithItem(control, "Akasa Air").get("id").asString
                    assertEquals("Table", component(presented, flightId).get("component").asString)
                    val originalNotesComponent = componentWithItem(control, "Prices are *dynamic*")
                    val originalNotes = originalNotesComponent.getAsJsonArray("items")
                    val finalNotes = component(presented, originalNotesComponent.get("id").asString)
                    assertEquals("List", finalNotes.get("component").asString)
                    assertEquals("Fare notes changed during entity presentation",
                        originalNotes, finalNotes.getAsJsonArray("items"))
                    assertEquals(5, table.getAsJsonArray("rows").size())
                    assertTrue("Fare note citation was lost", "[1]" in finalJson)
                }
            }
            "BXP-010" -> {
                listOf("OnePlus Nord Buds 4 Pro", "Realme Buds Air 8", "Oppo Enco Buds 3 Pro+",
                    "₹3,999", "₹3,799", "₹2,999", "[6][2]", "[7]").forEach { value ->
                    assertTrue("Lost earbuds detail $value in ${case.arm}", value in finalJson)
                }
                if (case.arm == "latest") {
                    val optionsId = componentWithItem(control, "OnePlus Nord Buds 4 Pro").get("id").asString
                    assertEquals("Table", component(presented, optionsId).get("component").asString)
                    assertEquals(3, table.getAsJsonArray("rows").size())
                }
            }
            "BXP-003" -> {
                listOf("Shatabdi Express (12007)", "Rajya Rani Express (20660)",
                    "Wodeyar Express (12614)", "10:35–10:45", "11:30", "15:15").forEach { value ->
                    assertTrue("Lost train detail $value in ${case.arm}", value in finalJson)
                }
                if (case.arm == "latest") {
                    assertEquals("table", table.get("preferredPresentation").asString)
                    val notesId = componentWithItem(control, "small timing discrepancy").get("id").asString
                    assertEquals("List", component(presented, notesId).get("component").asString)
                }
            }
        }
    }

    private fun components(a2uiJson: String): JsonArray = JsonParser.parseString(a2uiJson).asJsonArray[1]
        .asJsonObject.getAsJsonObject("updateComponents").getAsJsonArray("components")

    private fun component(components: JsonArray, id: String): JsonObject = components
        .map { it.asJsonObject }.first { it.get("id").asString == id }

    private fun componentWithItem(components: JsonArray, marker: String): JsonObject = components
        .map { it.asJsonObject }
        .first { it.getAsJsonArray("items")?.any { item -> marker in item.asString } == true }

    private fun componentNames(components: JsonArray): Map<String, Int> = components
        .map { it.asJsonObject.get("component").asString }
        .groupingBy { it }.eachCount()

    private fun assertNativeFlightCard(descriptions: Set<String>) {
        assertTrue("Native Akasa card lost airline, schedule or fare: $descriptions",
            descriptions.any { description ->
                "Akasa Air" in description && "Depart 12:25" in description &&
                    "Arrive 15:05" in description && "₹7,195" in description
            })
        assertTrue("Native flight card has no accessible aircraft/route icon: $descriptions",
            descriptions.any { "Flight route" in it || "Flight icon" in it })
    }

    private fun outputName(): String = (InstrumentationRegistry.getArguments().getString("outputDir")
        ?: "replay_${System.currentTimeMillis()}").also {
        require(Regex("[A-Za-z0-9_-]{1,80}").matches(it)) { "outputDir must be one safe directory name" }
    }

    private fun capture(
        instrumentation: android.app.Instrumentation,
        device: UiDevice,
        directory: File,
        label: String,
        observed: MutableSet<String>,
        descriptions: MutableSet<String>,
    ) {
        instrumentation.waitForIdleSync()
        device.waitForIdle(1_000)
        if (Build.VERSION.SDK_INT >= 34) instrumentation.uiAutomation.clearCache()
        device.dumpWindowHierarchy(File(directory, "$label.xml"))
        assertTrue(device.takeScreenshot(File(directory, "$label.png")))
        device.findObjects(By.pkg(instrumentation.targetContext.packageName)).forEach { node ->
            node.text?.takeIf(String::isNotBlank)?.let(observed::add)
            node.contentDescription?.takeIf(String::isNotBlank)?.let {
                observed.add(it)
                descriptions.add(it)
            }
        }
    }

    private fun swipeUp(device: UiDevice) {
        device.swipe(device.displayWidth / 2, device.displayHeight * 83 / 100,
            device.displayWidth / 2, device.displayHeight * 35 / 100, 35)
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private data class ReplayCase(
        val arm: String,
        val id: String,
        val sha256: String,
        val repairKind: GenUiRepairKind,
        val visibleMarker: String,
    )

    companion object {
        private const val CORPUS_SHA256 = "829ecba33e11ed775d6bee7692bbed2cacc8262e9ab97f00f9ff19cc6a903038"
        private val CASES = listOf(
            ReplayCase("latest", "LIVE-FLIGHT", "31f25c0038c12b70724d8ed2fd9fb9d72f7b9c5784e8192b7dd29d5d55d8c7c6", GenUiRepairKind.NONE, "Akasa Air"),
            ReplayCase("latest", "BXP-010", "f46f3d9ee42ef674244f7855803cb58749c81343d2852ab4b0e922319f007018", GenUiRepairKind.NONE, "OnePlus Nord Buds 4 Pro"),
            ReplayCase("latest", "BXP-003", "4bd66cf5d55bc27bf11991bd2ff562ea37b75f12b3442be49d85f03d4799bd6b", GenUiRepairKind.NONE, "Shatabdi Express"),
            ReplayCase("previous", "LIVE-FLIGHT", "851c9778b5496ff6f2de8436a8c24ab9adc7242dc3d5899084bf85f8265c57cf", GenUiRepairKind.GENERATED_DSL_REPAIR, "Akasa Air"),
            ReplayCase("previous", "BXP-010", "3c216d3876136b26e0352feba3f9775f502684e6d37e2850779eeaa0732f4bd0", GenUiRepairKind.NONE, "OnePlus Nord Buds 4 Pro"),
            ReplayCase("previous", "BXP-003", "63ac5dc899201209855dd760bfb67db85c7ac064388117b07f60f183526a6fb7", GenUiRepairKind.GENERATED_DSL_REPAIR, "Shatabdi Express"),
        )
    }
}
