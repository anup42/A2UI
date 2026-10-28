package com.samsung.genuicraft

import android.content.Intent
import android.os.Build
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiRepairKind
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Replays the unchanged BXP-003 trained E2B output captured on Fold8; no provider is started. */
@RunWith(AndroidJUnit4::class)
class TrainExplicitTablePresentationOnDeviceTest {
    @Test
    fun featureMatrixWithTrainNamedColumnKeepsItsExplicitGrid() {
        // This shape has enough rail-labelled columns for train-row mapping, but each row is a
        // feature; treating its "Train service" cell as the row identity would change its meaning.
        val express = """
            <a2ui>
            root=Table(columns=["Feature","Train service","Journey duration"],rows=[["Departure","Shatabdi Express","2 h"],["Arrival","Rajya Rani Express","3 h"]],domain="comparison",preferredPresentation="table")
            </a2ui>
        """.trimIndent()
        val outcome = GenUiCompiler.compileWithRepair(
            express,
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = false,
        )
        assertEquals(GenUiRepairKind.NONE, outcome.repairKind)
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            scenario.onActivity { it.showDocument(outcome.document, "Train feature matrix · synthetic routing guard") }
            assertTrue(device.wait(Until.hasObject(By.text("Feature")), 10_000))
            instrumentation.waitForIdleSync()
            val nodes = device.findObjects(By.pkg(context.packageName))
            assertTrue(
                "Feature matrix did not retain its 2-row, 3-column grid",
                nodes.any { it.contentDescription?.startsWith("Table with 2 rows and 3 columns") == true },
            )
            assertFalse(nodes.any { it.contentDescription?.startsWith("Train comparison,") == true })
            assertFalse(nodes.any { it.contentDescription?.startsWith("Train service card") == true })
            assertFalse(nodes.any { it.text == "Table view" })
        }
    }

    @Test
    fun explicitTableTrainComparisonDefaultsToCardsAndCanShowTheSourceGrid() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val raw = instrumentation.context.assets.open("train_comparison_explicit_table_captured.express")
            .use { it.readBytes() }
        assertEquals(
            "4bd66cf5d55bc27bf11991bd2ff562ea37b75f12b3442be49d85f03d4799bd6b",
            sha256(raw),
        )
        val outcome = GenUiCompiler.compileWithRepair(
            raw.toString(Charsets.UTF_8),
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = false,
        )
        assertEquals(GenUiRepairKind.NONE, outcome.repairKind)
        val components = JsonParser.parseString(outcome.document.a2uiJson).asJsonArray[1]
            .asJsonObject.getAsJsonObject("updateComponents").getAsJsonArray("components")
        val table = components.first { it.asJsonObject.get("component").asString == "Table" }.asJsonObject
        assertEquals("table", table.get("preferredPresentation").asString)
        assertEquals("comparison", table.get("domain").asString)
        assertEquals(3, table.getAsJsonArray("rows").size())

        val directory = File(context.filesDir, "result/train_explicit_table/${System.currentTimeMillis()}")
        check(directory.mkdirs())
        File(directory, "captured.express").writeBytes(raw)
        File(directory, "compiled.a2ui.json").writeText(outcome.document.a2uiJson)
        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        val cardObserved = linkedSetOf<String>()
        val cardDescriptions = linkedSetOf<String>()
        val gridObserved = linkedSetOf<String>()
        val gridDescriptions = linkedSetOf<String>()
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            scenario.onActivity { it.showDocument(outcome.document, "BXP-003 · captured E2B output · no inference") }
            assertTrue(device.wait(Until.hasObject(By.desc("Train comparison, 3 services")), 10_000))
            capture(instrumentation, device, directory, "cards_initial", cardObserved, cardDescriptions)

            // Switch modes before the vertical sweep can move the control off screen.
            var tableButton = device.wait(Until.findObject(By.text("Table view")), 2_000)
            repeat(4) {
                if (tableButton == null) {
                    swipeUp(device)
                    tableButton = device.wait(Until.findObject(By.text("Table view")), 2_000)
                }
            }
            assertTrue("Missing Table view control for explicit train table", tableButton != null)
            tableButton!!.click()
            capture(instrumentation, device, directory, "source_grid", gridObserved, gridDescriptions)
            assertTrue("Table view did not expose the Card view control", "Card view" in gridObserved)
            assertTrue(
                "Table view did not render the native 3-row, 5-column grid",
                gridDescriptions.any { it.startsWith("Table with 3 rows and 5 columns") },
            )
            assertTrue("Source grid omitted the Train service header", gridObserved.any { "Train service" in it })
            assertFalse("Train cards remained in Table view", "Train comparison, 3 services" in gridDescriptions)

            device.findObject(By.text("Card view")).click()
            val restoredObserved = linkedSetOf<String>()
            val restoredDescriptions = linkedSetOf<String>()
            capture(instrumentation, device, directory, "cards_restored", restoredObserved, restoredDescriptions)
            assertTrue("Card view did not restore train cards", "Train comparison, 3 services" in restoredDescriptions)
            assertTrue("Card view did not restore the Table view control", "Table view" in restoredObserved)
            cardObserved.addAll(restoredObserved)
            cardDescriptions.addAll(restoredDescriptions)
            for (page in 0 until 8) {
                capture(instrumentation, device, directory, "cards_page_$page", cardObserved, cardDescriptions)
                if (requiredValues.all { wanted -> cardObserved.any { wanted in it } } &&
                    (1..3).all { "Train service card $it" in cardDescriptions }
                ) break
                swipeUp(device)
            }
        }
        requiredValues.forEach { wanted ->
            assertTrue("Missing original train value or note: $wanted\n$cardObserved", cardObserved.any { wanted in it })
        }
        (1..3).forEach { index ->
            assertTrue("Missing train service card $index", "Train service card $index" in cardDescriptions)
        }
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
        val refreshed = requireNotNull(instrumentation.uiAutomation.rootInActiveWindow).refresh()
        device.dumpWindowHierarchy(File(directory, "$label.xml"))
        assertTrue(device.takeScreenshot(File(directory, "$label.png")))
        assertTrue("Accessibility tree did not refresh for $label", refreshed)
        // Inspect the captured snapshot: live UiObject2 references can become stale while
        // Compose replaces card nodes after the Table/Card view toggle.
        val nodes = javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder()
            .parse(File(directory, "$label.xml")).getElementsByTagName("node")
        for (index in 0 until nodes.length) {
            val node = nodes.item(index) as org.w3c.dom.Element
            if (node.getAttribute("package") != instrumentation.targetContext.packageName) continue
            node.getAttribute("text").takeIf(String::isNotBlank)?.let(observed::add)
            node.getAttribute("content-desc").takeIf(String::isNotBlank)?.let(descriptions::add)
        }
    }

    private fun swipeUp(device: UiDevice) {
        device.swipe(
            device.displayWidth / 2, device.displayHeight * 83 / 100,
            device.displayWidth / 2, device.displayHeight * 35 / 100, 35,
        )
    }

    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private val requiredValues = listOf(
        "Shatabdi Express (12007)", "Rajya Rani Express (20660)", "Wodeyar Express (12614)",
        "KSR Bengaluru (SBC)", "10:35–10:45", "11:30", "15:15",
        "About 2 h 15 m to 2 h 25 m", "About 2 h 30 m", "CC, EC", "CC, 2S, GN",
        "has a small timing discrepancy across sources", "one source lists 10:45, while another lists 10:35",
        "shows slight variation in duration", "is consistently shown at 11:30",
        "is consistently shown at 15:15", "arrival times", "days of operation",
    )
}
