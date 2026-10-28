package com.samsung.genuicraft

import android.content.Intent
import android.os.Build
import android.os.SystemClock
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.Direction
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiAction
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiConversionResult
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import com.samsung.genuicraft.sdk.GenUiRepairKind
import com.samsung.genuicraft.sdk.GenUiRequest
import com.samsung.genuicraft.sdk.GenUiSession
import com.samsung.genuicraft.sdk.provider.Gemma4GpuPrecision
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import javax.xml.parsers.DocumentBuilderFactory
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Replays the captured Bixby View payload through the public SDK and the real demo page. */
@RunWith(AndroidJUnit4::class)
class PlaceCardMysoreReplayOnDeviceTest {
    @Test
    fun demoButtonPresentsTheCapturedAnswerWithoutStartingInference() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val expectedRaw = instrumentation.context.assets.open(FIXTURE)
            .bufferedReader().use { it.readText() }
        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            val pageSelector = By.pkg(context.packageName).desc("SDK page")
            val buttonSelector = By.pkg(context.packageName)
                .text("Try Mysore place cards · captured Bixby answer")
            assertTrue(device.wait(Until.hasObject(pageSelector), 10_000))
            for (attempt in 0 until 10) {
                if (device.hasObject(buttonSelector)) break
                requireNotNull(device.findObject(pageSelector)).scroll(Direction.DOWN, 0.4f)
                refresh(instrumentation, device)
            }
            assertTrue("Captured Mysore demo button is absent", device.hasObject(buttonSelector))
            requireNotNull(device.findObject(buttonSelector)).click()
            var trace: SdkGenerationTrace? = null
            var source: String? = null
            var status: String? = null
            var documentJson: String? = null
            var runtime: String? = null
            val deadline = SystemClock.uptimeMillis() + 15_000L
            do {
                scenario.onActivity { activity ->
                    trace = activity.generationTraceForTest()
                    source = ViewModelProvider(activity)[GenUiSdkDemoViewModel::class.java]
                        .generationSourceText
                    status = activity.statusForTest()
                    documentJson = activity.renderedDocumentForTest()?.a2uiJson
                    runtime = activity.runtimeForTest()
                }
                if (trace?.phase == SdkGenerationPhase.COMPLETE ||
                    trace?.phase == SdkGenerationPhase.FAILED) break
                Thread.sleep(100)
            } while (SystemClock.uptimeMillis() < deadline)
            assertEquals("Demo button import failed: ${trace?.error}",
                SdkGenerationPhase.COMPLETE, trace?.phase)
            assertEquals(expectedRaw, source)
            assertTrue("Demo status did not report direct import: $status",
                status.orEmpty().contains("no model inference"))
            assertTrue("Demo used model attempts", trace!!.attempts.isEmpty())
            assertEquals(GenUiRepairKind.NONE, trace!!.repairKind)
            assertEquals("SDK structured input · no model inference", runtime)
            assertNotNull(documentJson)
            val firstCard = By.pkg(context.packageName).desc("Place card: ${TITLES.first()}")
            for (attempt in 0 until 10) {
                if (device.hasObject(firstCard)) break
                requireNotNull(device.findObject(pageSelector)).scroll(Direction.DOWN, 0.35f)
                refresh(instrumentation, device)
            }
            assertTrue("Demo button did not render the first place card", device.hasObject(firstCard))
        }
    }

    @Test
    fun capturedMysoreViewImportsWithoutModelAndRendersAllSixPhotoCards() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val rawBytes = instrumentation.context.assets.open(FIXTURE).use { it.readBytes() }
        assertEquals("Captured Bixby View payload changed", RAW_SHA256, sha256(rawBytes))
        val raw = rawBytes.toString(Charsets.UTF_8)
        val source = JsonParser.parseString(raw).asJsonArray
        val sourceCards = source.map { it.asJsonObject }
            .filter { it.get("\$type")?.asString == "CompoundCard" }
        assertEquals(6, sourceCards.size)
        val sourceTitles = sourceCards.map { sourceTitle(it) }
        assertEquals(TITLES, sourceTitles)
        assertEquals(18, sourceCards.sumOf { sourceImages(it).size })
        assertEquals(9, sourceCards.sumOf { card ->
            sourceImages(card).count { it.get("onClickUrl")?.isJsonNull == false }
        })

        val output = File(requireNotNull(context.getExternalFilesDir(null)),
            "sdk_place_card_replay/mysore_${System.currentTimeMillis()}").apply { check(mkdirs()) }
        File(output, "captured_bixby_view.json").writeBytes(rawBytes)
        val providerCalls = AtomicInteger()
        val provider = object : GenUiProvider {
            override val id = "must_not_run_for_structured_input"
            override suspend fun prewarm() {
                providerCalls.incrementAndGet()
                error("Structured import must not prewarm a model")
            }
            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
                providerCalls.incrementAndGet()
                error("Structured import must not generate model output")
            }
            override suspend fun finishGenerationMetrics() = error(
                "Structured import must not request model metrics")
        }
        val session = GenUiSession(context, provider)
        val result = try {
            session.convert(GenUiRequest(raw))
        } finally {
            session.closeAndAwait()
        }
        assertTrue("SDK rejected captured structured input: $result", result is GenUiConversionResult.Success)
        val imported = result as GenUiConversionResult.Success
        assertEquals("sdk-structured-input", imported.provider)
        assertEquals(0, imported.attempts)
        assertEquals(0, providerCalls.get())
        assertEquals(GenUiRepairKind.NONE, imported.repairKind)
        assertTrue(session.attemptSnapshots.isEmpty())
        assertEquals(null, session.generationSessionMetrics)
        val sdkDocument = GenUiCompiler.compile(imported.document.a2uiJson)
        File(output, "sdk_import.a2ui.json").writeText(sdkDocument.a2uiJson)
        File(output, "sdk_import.express").writeText(imported.document.express)
        assertImportedSource(sourceCards, source.map { it.asJsonObject }, imported.document.a2uiJson)

        val appFactoryCalls = AtomicInteger()
        val receivedAction = AtomicReference<GenUiAction?>()
        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        var finalTrace: SdkGenerationTrace? = null
        var finalStatus: String? = null
        var finalDocumentJson: String? = null
        var finalRuntime: String? = null
        var finalModelMetricsPresent: Boolean? = null
        val seenCards = linkedSetOf<String>()
        val loadedPhotos = linkedSetOf<String>()
        val unavailablePhotos = linkedSetOf<String>()
        val visibleTexts = linkedSetOf<String>()
        val visibleDescriptions = linkedSetOf<String>()
        val screenshotNames = mutableListOf<String>()
        var carouselCaptured = false
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                activity.onSdkAction = { receivedAction.set(it) }
                ViewModelProvider(activity)[GenUiSdkDemoViewModel::class.java].generate(
                    source = raw,
                    e2bModelChoiceForRun = E2bModelChoice.TRAINED_E2B_V10_W4,
                    modelPathForRun = "unused_for_structured_import",
                    metricsForRun = true,
                    mtpForRun = false,
                    gpuPrecisionForRun = Gemma4GpuPrecision.FP32,
                    providerFactory = {
                        appFactoryCalls.incrementAndGet()
                        error("The demo must import structured input before creating a model provider")
                    },
                )
            }
            val deadline = SystemClock.uptimeMillis() + 15_000L
            do {
                scenario.onActivity { activity ->
                    finalTrace = activity.generationTraceForTest()
                    finalStatus = activity.statusForTest()
                    finalDocumentJson = activity.renderedDocumentForTest()?.a2uiJson
                    finalRuntime = activity.runtimeForTest()
                    finalModelMetricsPresent = activity.generationMetricsForTest() != null
                }
                if (finalTrace?.phase == SdkGenerationPhase.COMPLETE ||
                    finalTrace?.phase == SdkGenerationPhase.FAILED) break
                Thread.sleep(100)
            } while (SystemClock.uptimeMillis() < deadline)
            assertEquals("Demo failed to import captured payload: ${finalTrace?.error}",
                SdkGenerationPhase.COMPLETE, finalTrace?.phase)
            assertEquals(0, appFactoryCalls.get())
            assertTrue("The demo reported a model attempt", finalTrace!!.attempts.isEmpty())
            assertEquals(GenUiRepairKind.NONE, finalTrace!!.repairKind)
            assertEquals("SDK structured input · no model inference", finalRuntime)
            assertEquals("A no-model import exposed inference metrics", false, finalModelMetricsPresent)
            assertNotNull("Demo did not retain the SDK document", finalDocumentJson)
            assertEquals(imported.document.a2uiJson, finalDocumentJson)
            File(output, "testapp_presented.a2ui.json").writeText(finalDocumentJson!!)

            val pageSelector = By.pkg(context.packageName).desc("SDK page")
            assertTrue("SDK parent page did not appear", device.wait(Until.hasObject(pageSelector), 10_000))
            assertTrue("SDK parent page cannot scroll to later cards",
                requireNotNull(device.findObject(pageSelector)).isScrollable)
            var actionTapped = false
            val photoLoadDeadline = SystemClock.uptimeMillis() + 30_000L
            for (page in 0 until MAX_PAGES) {
                refresh(instrumentation, device)
                // Give every image currently in the viewport time to reach Coil's success
                // or error state before taking a screenshot. Keep the total wait bounded.
                val thisPageDeadline = minOf(SystemClock.uptimeMillis() + 5_000L, photoLoadDeadline)
                while (SystemClock.uptimeMillis() < thisPageDeadline &&
                    device.hasObject(By.pkg(context.packageName).descContains(", loading:"))) {
                    Thread.sleep(200)
                    refresh(instrumentation, device)
                }
                refresh(instrumentation, device)
                val screenshot = "page_${page.toString().padStart(2, '0')}.png"
                val hierarchyFile = File(output, screenshot.replace(".png", ".xml"))
                device.dumpWindowHierarchy(hierarchyFile)
                assertTrue("Could not capture $screenshot", device.takeScreenshot(File(output, screenshot)))
                screenshotNames += screenshot
                readHierarchy(hierarchyFile).forEach { (text, description) ->
                    description.takeIf(String::isNotBlank)?.let(visibleDescriptions::add)
                    text.takeIf(String::isNotBlank)?.let(visibleTexts::add)
                    TITLES.forEach { title ->
                        if (description == "Place card: $title") seenCards += title
                        if (description.startsWith("Photo 1 of 3, loaded:") &&
                            description.endsWith(title)) loadedPhotos += title
                        if (description.startsWith("Photo 1 of 3, unavailable:") &&
                            description.endsWith(title)) unavailablePhotos += title
                    }
                }
                if (!carouselCaptured && TITLES.first() in loadedPhotos) {
                    val carouselSelector = By.pkg(context.packageName)
                        .desc("Place photos: ${TITLES.first()}")
                    if (device.hasObject(carouselSelector)) {
                        val thirdPhoto = By.pkg(context.packageName)
                            .desc("Photo 3 of 3, loaded: ${TITLES.first()}")
                        val thirdLoading = By.pkg(context.packageName)
                            .desc("Photo 3 of 3, loading: ${TITLES.first()}")
                        for (swipe in 0 until 3) {
                            if (device.hasObject(thirdPhoto)) break
                            if (device.hasObject(thirdLoading)) {
                                device.wait(Until.hasObject(thirdPhoto), 8_000)
                                break
                            }
                            requireNotNull(device.findObject(carouselSelector))
                                .scroll(Direction.RIGHT, 0.8f)
                            refresh(instrumentation, device)
                            device.wait(Until.hasObject(thirdPhoto), 2_000)
                        }
                        val carouselFile = "first_card_third_photo.png"
                        device.dumpWindowHierarchy(File(output, "first_card_third_photo.xml"))
                        assertTrue("Could not capture third carousel photo state",
                            device.takeScreenshot(File(output, carouselFile)))
                        screenshotNames += carouselFile
                        carouselCaptured = device.hasObject(thirdPhoto)
                    }
                }
                if (!actionTapped) {
                    val directions = device.findObjects(By.pkg(context.packageName).text("Directions"))
                    if (directions.isNotEmpty()) {
                        directions.first().click()
                        refresh(instrumentation, device)
                        actionTapped = true
                        assertEquals("Action caused an external app launch", context.packageName,
                            device.currentPackageName)
                    }
                }
                if (seenCards.size == 6 && loadedPhotos.size == 6 &&
                    carouselCaptured && actionTapped) break
                val pageView = requireNotNull(device.findObject(pageSelector))
                if (!pageView.isScrollable) break
                pageView.scroll(Direction.DOWN, 0.45f)
            }
            assertEquals(context.packageName, device.currentPackageName)
        }

        val missingCards = TITLES.filterNot { it in seenCards }
        val missingLoadedPhotos = TITLES.filterNot { it in loadedPhotos }
        val requiredVisibleFacts = sourceCards.flatMap { card ->
            val ratingAndReviews = card.contents("SingleLine").first()
                .getAsJsonArray("contents").mapNotNull { item ->
                    item.asJsonObject.takeIf { it.get("\$type")?.asString == "Text" }
                        ?.get("value")?.asString?.takeIf { it.isNotBlank() && it != "★" }
                }
            ratingAndReviews + card.contents("ActionLinks").first()
                .getAsJsonArray("items")[0].asJsonObject.get("label").asString
        }.distinct()
        val missingVisibleFacts = requiredVisibleFacts.filterNot { fact ->
            visibleTexts.any { fact in it } || visibleDescriptions.any { fact in it }
        }
        File(output, "result.json").writeText(GsonBuilder().setPrettyPrinting().disableHtmlEscaping()
            .create().toJson(mapOf(
                "fixtureSha256" to RAW_SHA256,
                "sourceCardCount" to sourceCards.size,
                "sourceImageCount" to sourceCards.sumOf { sourceImages(it).size },
                "sourceClickableImageCount" to sourceCards.sumOf { card ->
                    sourceImages(card).count { it.get("onClickUrl")?.isJsonNull == false }
                },
                "sourceActionCount" to sourceCards.sumOf { sourceActions(it).size },
                "sdkProvider" to imported.provider,
                "sdkAttempts" to imported.attempts,
                "sdkRepairKind" to imported.repairKind.name,
                "sdkDocumentSha256" to sha256(imported.document.a2uiJson.toByteArray()),
                "sdkProviderCalls" to providerCalls.get(),
                "testAppProviderFactoryCalls" to appFactoryCalls.get(),
                "testAppPhase" to finalTrace?.phase?.name,
                "testAppStatus" to finalStatus,
                "testAppRuntime" to finalRuntime,
                "testAppModelMetricsPresent" to finalModelMetricsPresent,
                "testAppAttempts" to finalTrace?.attempts?.size,
                "testAppRepairKind" to finalTrace?.repairKind?.name,
                "testAppWarnings" to finalTrace?.warnings,
                "testAppDocumentSha256" to finalDocumentJson?.let { sha256(it.toByteArray()) },
                "visibleCards" to seenCards,
                "visibleTexts" to visibleTexts,
                "visibleDescriptions" to visibleDescriptions,
                "loadedFirstPhotos" to loadedPhotos,
                "unavailableFirstPhotos" to unavailablePhotos,
                "missingCards" to missingCards,
                "missingVisibleFacts" to missingVisibleFacts,
                "missingLoadedPhotos" to missingLoadedPhotos,
                "thirdCarouselPhotoCaptured" to carouselCaptured,
                "action" to receivedAction.get(),
                "screenshots" to screenshotNames,
            )) + "\n")
        assertTrue("Some cards were not visible: $missingCards; evidence: $output", missingCards.isEmpty())
        assertFalse("Itinerary prose was misclassified as console output; evidence: $output",
            visibleTexts.any { it.equals("Console output", ignoreCase = true) })
        assertTrue("Some card photos did not load: $missingLoadedPhotos; evidence: $output",
            missingLoadedPhotos.isEmpty())
        assertTrue("Rating, review or address text did not appear: $missingVisibleFacts; evidence: $output",
            missingVisibleFacts.isEmpty())
        assertTrue("First card's third carousel photo was not captured; evidence: $output",
            carouselCaptured)
        assertEquals("Wrong first-card action or URL was emitted", GenUiAction(
            name = "openUrl", parameters = mapOf("url" to FIRST_DIRECTIONS_URL)),
            receivedAction.get())
        assertFalse("No rendered screenshots were captured", screenshotNames.isEmpty())
    }

    private fun assertImportedSource(sourceCards: List<JsonObject>, allBlocks: List<JsonObject>, a2uiJson: String) {
        val wire = JsonParser.parseString(a2uiJson).asJsonArray
        val components = wire.flatMap { message ->
            message.asJsonObject.getAsJsonObject("updateComponents")?.getAsJsonArray("components")
                ?.map { it.asJsonObject }.orEmpty()
        }
        val places = components.filter { it.get("component")?.asString == "PlaceCard" }
        assertEquals("Expected one typed PlaceCard for each source card", 6, places.size)
        sourceCards.zip(places).forEach { (source, imported) ->
            assertEquals(sourceTitle(source), imported.get("title").asString)
            val sourceRatings = source.contents("SingleLine").firstOrNull()
                ?.getAsJsonArray("contents")?.mapNotNull { item ->
                    item.asJsonObject.takeIf { it.get("\$type")?.asString == "Text" }
                        ?.get("value")?.asString
                }.orEmpty()
            val rating = sourceRatings.firstOrNull { RATING.matches(it) }
            assertEquals(rating, imported.get("rating")?.asString)
            assertEquals(sourceRatings.firstOrNull { REVIEW_COUNT.matches(it) },
                imported.get("reviewCount")?.asString)
            val address = source.contents("ActionLinks").first()
                .getAsJsonArray("items")[0].asJsonObject
            assertEquals(address.get("label").asString, imported.get("address").asString)
            assertEquals(address.get("url").asString, imported.get("addressUrl").asString)
            val images = imported.getAsJsonArray("images")
            val sourceImages = sourceImages(source)
            assertEquals(sourceImages.size, images.size())
            sourceImages.zip(images.map { it.asJsonObject }).forEach { (original, image) ->
                assertEquals(original.get("url").asString, image.get("url").asString)
                assertEquals(original.get("title").asString, image.get("title").asString)
                assertEquals("Photo click target changed for ${sourceTitle(source)}",
                    original.get("onClickUrl")?.takeUnless { it.isJsonNull }?.asString,
                    image.get("onClickUrl")?.takeUnless { it.isJsonNull }?.asString)
            }
            val actions = imported.getAsJsonArray("actions")
            val sourceActions = sourceActions(source)
            assertEquals(sourceActions.size, actions.size())
            sourceActions.zip(actions.map { it.asJsonObject }).forEach { (original, action) ->
                assertEquals(original.get("label").asString, action.get("label").asString)
                assertEquals(original.get("url").asString, action.get("url").asString)
            }
        }
        val sourceMarkdown = allBlocks.filter { it.get("\$type")?.asString == "Markdown" }
            .map { it.get("value").asString }
        val importedText = components.filter { it.get("component")?.asString == "Text" }
            .mapNotNull { it.get("text")?.asString }
        assertEquals("Ordinary Markdown order, duplicates, or values changed", sourceMarkdown, importedText)
    }

    private fun sourceTitle(card: JsonObject): String = card.contents("TitleArea").first()
        .getAsJsonObject("slot1").get("text").asString.removeSurrounding("**")

    private fun sourceImages(card: JsonObject): List<JsonObject> = card.contents("ImageList").first()
        .getAsJsonArray("images").map { it.asJsonObject }

    private fun sourceActions(card: JsonObject): List<JsonObject> = card.contents("LinkButtons").first()
        .getAsJsonArray("items").map { it.asJsonObject }

    private fun JsonObject.contents(type: String): List<JsonObject> = getAsJsonArray("contents")
        .map { it.asJsonObject }.filter { it.get("\$type")?.asString == type }

    /** Snapshot text and descriptions without keeping UI nodes that Coil can invalidate. */
    private fun readHierarchy(file: File): List<Pair<String, String>> {
        val nodes = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file)
            .getElementsByTagName("node")
        return (0 until nodes.length).map { index ->
            val attributes = nodes.item(index).attributes
            attributes.getNamedItem("text")?.nodeValue.orEmpty() to
                attributes.getNamedItem("content-desc")?.nodeValue.orEmpty()
        }
    }

    private fun refresh(instrumentation: android.app.Instrumentation, device: UiDevice) {
        instrumentation.waitForIdleSync()
        device.waitForIdle(700)
        if (Build.VERSION.SDK_INT >= 34) instrumentation.uiAutomation.clearCache()
    }

    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private companion object {
        const val FIXTURE = "place_card_replay/mysore_view.json"
        const val RAW_SHA256 = "4d7d41dc18f6f8aea1499ba5ecae89cdbcede33d33d9d9fe33c8437afd3f01fd"
        const val MAX_PAGES = 24
        val RATING = Regex("^[0-5](?:\\.[0-9]+)?$")
        val REVIEW_COUNT = Regex("^\\([0-9]+\\)$")
        const val FIRST_DIRECTIONS_URL =
            "https://www.google.com/maps/dir/?api=1&destination=Sri%20Ranganathaswamy%20Temple%20CMFH%2BXJ4%2C%20Srirangapatna%2C%20Karnataka%20571438%2C%20India"
        val TITLES = listOf(
            "Sri Ranganathaswamy Temple",
            "Brindavan gardens mysore",
            "Aishwarya Le Royal (Best Budget Hotel in Mysore)",
            "Chamundi Hill",
            "Mysore Palace",
            "Mysuru Zoo",
        )
    }
}
