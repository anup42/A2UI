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
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiRepairKind
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Replays the BXP-001 output captured from the device; no provider or model is started. */
@RunWith(AndroidJUnit4::class)
class WeatherForecastCardsOnDeviceTest {
    @Test
    fun capturedForecastKeepsThreeDatedReadingsAndPlanningAdviceWithoutInventedConditions() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val raw = instrumentation.context.assets.open("weather_forecast_captured.express")
            .use { it.readBytes() }
        assertEquals(CAPTURED_SHA256, sha256(raw))
        val outcome = GenUiCompiler.compileWithRepair(
            raw.toString(Charsets.UTF_8),
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = false,
        )
        assertEquals(GenUiRepairKind.NONE, outcome.repairKind)
        val directory = File(context.filesDir, "result/weather_forecast/${System.currentTimeMillis()}")
        check(directory.mkdirs())
        File(directory, "captured.express").writeBytes(raw)
        File(directory, "compiled.a2ui.json").writeText(outcome.document.a2uiJson)

        val observed = replayAndCollect(outcome.document, device, directory) { scan ->
            EXPECTED_ROWS.all { row -> scan.descriptions.any { description ->
                row.all { value -> value in description }
            } } && listOf(
                "highest rain chance on Wednesday",
                "Carry an umbrella or light rain jacket",
                "Plan outdoor activities earlier in the day",
            ).all { fragment ->
                scan.texts.any { fragment in it } || scan.descriptions.any { fragment in it }
            }
        }
        EXPECTED_ROWS.forEach { row ->
            assertTrue(
                "Missing dated forecast or Celsius/rain value: $row\n${observed.descriptions}",
                observed.descriptions.any { description ->
                    row.all { value -> value in description }
                },
            )
        }
        listOf(
            "highest rain chance on Wednesday",
            "Carry an umbrella or light rain jacket",
            "Plan outdoor activities earlier in the day",
        ).forEach { fragment ->
            assertTrue("Missing captured forecast advice: $fragment\n${observed.texts}",
                observed.texts.any { fragment in it } || observed.descriptions.any { fragment in it })
        }
        listOf("57%", "55%").forEach { chance ->
            assertTrue("Missing visible illustration for rain probability $chance",
                observed.descriptions.any { "Rain Probability illustration: $chance" in it })
        }
        val datedRows = observed.descriptions.filter { description ->
            EXPECTED_ROWS.any { row -> row.first() in description }
        }
        assertFalse("A missing per-day condition acquired an invented label: $datedRows",
            datedRows.any { INVENTED_CONDITION.containsMatchIn(it) })
        assertFalse("A missing per-day condition acquired a fallback icon label",
            observed.descriptions.any { "Condition icon:" in it || it == "Weather condition" })
    }

    @Test
    fun authoredSunnyRainyCloudyConditionsRemainAttachedToTheirIcons() {
        val express = """
            <a2ui>
            root=Table(columns=["Date","Condition","High (°C)","Low (°C)","Rain probability"],rows=[["Sat, Sep 12","Sunny","32","22","10%"],["Sun, Sep 13","Rainy","27","21","80%"],["Mon, Sep 14","Cloudy","29","20","35%"]],domain="weather",preferredPresentation="cards")
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
        val directory = File(context.filesDir, "result/weather_forecast/conditions_${System.currentTimeMillis()}")
        check(directory.mkdirs())
        File(directory, "synthetic.express").writeText(express)
        val observed = replayAndCollect(
            outcome.document, UiDevice.getInstance(instrumentation), directory,
        ) { scan ->
            listOf("Sunny", "Rainy", "Cloudy").all { condition ->
                scan.descriptions.any { "Condition icon: $condition" in it }
            }
        }
        listOf(
            "Sat, Sep 12" to "Sunny",
            "Sun, Sep 13" to "Rainy",
            "Mon, Sep 14" to "Cloudy",
        ).forEach { (date, condition) ->
            assertTrue("Authored $condition condition was detached from $date",
                observed.descriptions.any { date in it && condition in it })
            assertTrue("Missing icon description for authored $condition condition",
                observed.descriptions.any { "Condition icon: $condition" in it })
        }
    }

    private fun replayAndCollect(
        document: com.samsung.genuicraft.sdk.GenUiDocument,
        device: UiDevice,
        directory: File,
        done: (Observation) -> Boolean,
    ): Observation {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        val observed = Observation()
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            scenario.onActivity { it.showDocument(document, "Weather forecast · captured output · no inference") }
            val pageSelector = By.pkg(context.packageName).desc("SDK page")
            assertTrue(device.wait(Until.hasObject(pageSelector), 10_000))
            assertTrue(requireNotNull(device.findObject(pageSelector)).isScrollable)
            for (page in 0 until 14) {
                instrumentation.waitForIdleSync()
                device.waitForIdle(1_000)
                if (Build.VERSION.SDK_INT >= 34) instrumentation.uiAutomation.clearCache()
                device.dumpWindowHierarchy(File(directory, "page_$page.xml"))
                assertTrue("Could not capture weather page $page",
                    device.takeScreenshot(File(directory, "page_$page.png")))
                device.findObjects(By.pkg(context.packageName)).forEach { node ->
                    node.text?.takeIf(String::isNotBlank)?.let(observed.texts::add)
                    node.contentDescription?.takeIf(String::isNotBlank)?.let(observed.descriptions::add)
                }
                if (done(observed)) break
                requireNotNull(device.findObject(pageSelector)).scroll(Direction.DOWN, 0.45f)
            }
        }
        return observed
    }

    private data class Observation(
        val texts: MutableSet<String> = linkedSetOf(),
        val descriptions: MutableSet<String> = linkedSetOf(),
    )

    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private companion object {
        const val CAPTURED_SHA256 = "36f9ea0bf23a969d0c59b1dcdd5cc5013b0a06f0e5d2f35dc3c7fd75c95ae7b4"
        val EXPECTED_ROWS = listOf(
            listOf("Wed, Sep 9", "High: 31°C", "Low: 20°C", "Rain Probability: 57%"),
            listOf("Thu, Sep 10", "High: 30°C", "Low: 21°C", "Rain Probability: 55%"),
            listOf("Fri, Sep 11", "High: 30°C", "Low: 21°C", "Rain Probability: 55%"),
        )
        val INVENTED_CONDITION = Regex("\\b(?:Sunny|Rainy|Today)\\b", RegexOption.IGNORE_CASE)
    }
}
