package com.samsung.genuicraft

import android.content.Intent
import android.os.Build
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.Direction
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiRepairKind
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Replays captured model outputs or saved SDK documents; no model is started. */
@RunWith(AndroidJUnit4::class)
class BixbyLibraryCapturedReplayOnDeviceTest {
    @Test fun capturedFlightDocumentShowsCompleteOffersAndRetainsPartialOffer() {
        // This is the SDK's saved document from the real inference run, not raw model output.
        val case = ReplayCase("02_flights", "c0cfc8cc24507ea5afda7cad0790ca983f14be65ffa79a41f05db48a60a607eb",
            GenUiRepairKind.NONE, "sdk_repaired_document")
        val result = replay(case, listOf(
            "IndiGo 6E-6558", "04:30", "06:55", "₹8,015",
            "Air India Express IX 1470", "06:05", "08:35", "Fare not shown",
            "Akasa Air QP 1421", "06:45", "09:30", "₹7,542",
            "SpiceJet SG-615", "10:20", "12:50", "₹7,733",
            "IndiGo 6E-196", "12:40", "after the morning window",
        ), 9, minPages = 3)
        assertTrue("Flight details missing: ${result.missing}", result.missing.isEmpty())
    }

    @Test fun capturedThreeCityWeatherKeepsEveryRangeAndRainNoteVisible() {
        listOf(
            ReplayCase("WEATHER_MARKDOWN", "c885b4a7a266f879c98dbb3567f115582956f1ffe7238a4d7b20633c879fd32f", GenUiRepairKind.NONE),
            ReplayCase("WEATHER_PASTED", "fb94ce11c5b91a9672e44c74a0b2faa6fc332153d15f4c8154979167ba880065", GenUiRepairKind.NONE),
        ).forEach { case ->
            val result = replay(case, WEATHER_REQUIRED, 9)
            assertTrue("${case.id} lost visible weather details ${result.missing}; evidence: ${result.directory}",
                result.missing.isEmpty())
        }
    }

    @Test fun freshPhoneTableRepairRendersAllThreeModelProducedRows() {
        val case = ReplayCase("03_phones", "8e515794ec30735ec90998b0382fe3fe3ef9ee68b37ae1b7305d8b8a00c77b81",
            GenUiRepairKind.GENERATED_DSL_REPAIR)
        val result = replay(case, PHONE_REQUIRED, 9, minPages = 3)
        assertTable(result.a2uiJson, 3, 4, listOf("Samsung Galaxy S25", "iPhone 16", "Google Pixel 9"))
        assertTrue("Phone table values were not visible ${result.missing}; evidence: ${result.directory}",
            result.missing.isEmpty())
    }

    @Test fun capturedBatteryTableRepairRendersAllThreeModelProducedRows() {
        val case = ReplayCase("BXP-047", "1a1abaac15ea21265a452ddf8ba0770f540e9602432a2b4bc249dfceebb08a91",
            GenUiRepairKind.GENERATED_DSL_REPAIR)
        val result = replay(case, listOf(
            "LFP", "NMC", "Sodium-ion", "Typically long; sources cite roughly 2,000–6,000+ cycles",
        ), 9)
        assertTable(result.a2uiJson, 3, 6, listOf("LFP", "NMC", "Sodium-ion"))
        assertTrue("Battery comparison rows were not visible ${result.missing}; evidence: ${result.directory}",
            result.missing.isEmpty())
    }

    private fun replay(case: ReplayCase, required: List<String>, maxPages: Int, minPages: Int = 0): ReplayResult {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val raw = instrumentation.context.assets.open("bixby_library_replay/${case.id}.express")
            .use { it.readBytes() }
        assertEquals("Captured model output changed for ${case.id}", case.sha256, sha256(raw))
        val outcome = GenUiCompiler.compileWithRepair(
            input = raw.toString(Charsets.UTF_8),
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = case.repairKind == GenUiRepairKind.GENERATED_DSL_REPAIR,
        )
        assertEquals("Unexpected SDK repair kind for ${case.id}", case.repairKind, outcome.repairKind)
        GenUiCompiler.compile(outcome.document.a2uiJson)

        val directory = File(requireNotNull(context.getExternalFilesDir(null)),
            "sdk_bixby_library_replay/${case.id}_${System.currentTimeMillis()}")
        check(directory.mkdirs())
        File(directory, "captured.express").writeBytes(raw)
        File(directory, "compiled.a2ui.json").writeText(outcome.document.a2uiJson)

        val texts = linkedSetOf<String>()
        val descriptions = linkedSetOf<String>()
        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            scenario.onActivity { it.showDocument(outcome.document, "${case.id} · captured output · no inference") }
            val pageSelector = By.pkg(context.packageName).desc("SDK page")
            assertTrue("SDK page did not appear for ${case.id}",
                device.wait(Until.hasObject(pageSelector), 10_000))
            for (page in 0 until maxPages) {
                instrumentation.waitForIdleSync()
                device.waitForIdle(500)
                if (Build.VERSION.SDK_INT >= 34) instrumentation.uiAutomation.clearCache()
                device.dumpWindowHierarchy(File(directory, "page_$page.xml"))
                assertTrue("Screenshot failed for ${case.id} page $page",
                    device.takeScreenshot(File(directory, "page_$page.png")))
                device.findObjects(By.pkg(context.packageName)).forEach { node ->
                    node.text?.takeIf(String::isNotBlank)?.let(texts::add)
                    node.contentDescription?.takeIf(String::isNotBlank)?.let(descriptions::add)
                }
                // A table's accessibility summary can contain offscreen rows. Keep scrolling
                // the flight/phone captures so the evidence also includes their lower cards.
                if (page >= minPages && required.all { marker -> visible(marker, texts, descriptions) }) break
                val pageView = requireNotNull(device.findObject(pageSelector))
                if (!pageView.isScrollable) break
                pageView.scroll(Direction.DOWN, 0.45f)
            }
        }
        val missing = required.filterNot { visible(it, texts, descriptions) }
        File(directory, "result.json").writeText(GsonBuilder().setPrettyPrinting().disableHtmlEscaping()
            .create().toJson(mapOf(
                "case" to case.id,
                "rawSha256" to case.sha256,
                "inputKind" to case.inputKind,
                "repairKind" to outcome.repairKind.name,
                "modelCalls" to 0,
                "requiredVisible" to required,
                "missingVisible" to missing,
                "texts" to texts,
                "descriptions" to descriptions,
            )) + "\n")
        return ReplayResult(directory, outcome.document.a2uiJson, missing)
    }

    private fun assertTable(a2uiJson: String, rowCount: Int, columnCount: Int, names: List<String>) {
        val components = JsonParser.parseString(a2uiJson).asJsonArray[1].asJsonObject
            .getAsJsonObject("updateComponents").getAsJsonArray("components")
        val table = components.map { it.asJsonObject }
            .single { it.get("component")?.asString == "Table" }
        val rows = table.getAsJsonArray("rows")
        assertEquals(rowCount, rows.size())
        assertEquals(columnCount, table.getAsJsonArray("columns").size())
        assertEquals(names, rows.map { it.asJsonArray[0].asString })
    }

    private fun visible(value: String, texts: Set<String>, descriptions: Set<String>): Boolean =
        texts.any { value in it } || descriptions.any { value in it }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private data class ReplayCase(
        val id: String,
        val sha256: String,
        val repairKind: GenUiRepairKind,
        val inputKind: String = "model_output",
    )
    private data class ReplayResult(val directory: File, val a2uiJson: String, val missing: List<String>)

    private companion object {
        val WEATHER_REQUIRED = listOf(
            "Bengaluru", "Mostly cloudy, about 21.8°C at IMD observation time",
            "About 29.0°C to 30.0°C", "About 16.0°C to 18.0°C",
            "Low chance of rain; cloudy/partly cloudy conditions",
            "Mysore", "Partly sunny, about 27°C to 29°C range in the morning",
            "About 29.0°C to 29.6°C", "About 17.0°C to 18.3°C",
            "Mostly dry; sunny spells or cloudy/overcast",
            "Ooty", "Cool and cloudy, about 14.9°C to 15.9°C in the day forecast",
            "About 23.9°C to 24.4°C", "About 14.9°C to 17.0°C",
            "Higher rain chance, including patchy rain nearby or showers",
        )
        val PHONE_REQUIRED = listOf(
            "Samsung Galaxy S25", "iPhone 16", "Google Pixel 9",
            "4,000 mAh [21][31]", "3,561 mAh [27][35]", "4,700 mAh [1][10]",
        )
    }
}
