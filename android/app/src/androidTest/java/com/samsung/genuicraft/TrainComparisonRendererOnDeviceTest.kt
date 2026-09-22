package com.samsung.genuicraft

import android.content.Intent
import android.os.Build
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.google.gson.GsonBuilder
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiRepairKind
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Replays the exact generated IR captured from Fold7; does not construct a model/provider. */
@RunWith(AndroidJUnit4::class)
class TrainComparisonRendererOnDeviceTest {
    @Test
    fun capturedTrainComparisonUsesRailCardsAndPreservesEveryValueAndNote() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val raw = instrumentation.context.assets.open("train_comparison_captured.express").use { it.readBytes() }
        assertEquals("85f69c512d8df1e871069c2fb3d6c43ccfd283da065e40154e23cc2dd376e1da", sha(raw))
        val outcome = GenUiCompiler.compileWithRepair(
            raw.toString(Charsets.UTF_8),
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = true,
        )
        assertEquals(GenUiRepairKind.GENERATED_DSL_REPAIR, outcome.repairKind)
        // A renderer change must not alter the previously observed repaired document.
        assertEquals("61abcb5bf5f188560b1b0cb5871849eb208413fdac4c4b97c6fc8181075bc198", sha(outcome.document.express.toByteArray()))
        val directory = File(context.filesDir, "result/train_renderer/${System.currentTimeMillis()}")
        check(directory.mkdirs())
        File(directory, "raw.express").writeBytes(raw)
        File(directory, "repaired.express").writeText(outcome.document.express)
        File(directory, "repaired.a2ui.json").writeText(outcome.document.a2uiJson)
        val observed = linkedSetOf<String>()
        val descriptions = linkedSetOf<String>()
        val expected = listOf(
            "Shatabdi Express (120007)", "Rajya Rani Express (20606)", "Wodeyar Express (21246)",
            "KSR Bengaluru (BC)", "10:35–1004", "11:000", "15:5",
            "2 h15 m to 2h25 m", "2h30 m", "CC, EC", "CC, 2, 2S GN", "CC,2S GN",
            "Shatabdi Express has a small timing discrepancy across sources",
            "Wodeyar Express is consistently shown at 5:15", "N/A",
        )
        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        var pages = 0
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            scenario.onActivity { it.showDocument(outcome.document, "Train comparison · captured output · no inference") }
            assertTrue(device.wait(Until.hasObject(By.desc("Train comparison, 3 services")), 10_000))
            for (page in 0 until 8) {
                instrumentation.waitForIdleSync()
                device.waitForIdle(1_000)
                assertEquals(context.packageName, device.currentPackageName)
                if (Build.VERSION.SDK_INT >= 34) instrumentation.uiAutomation.clearCache()
                assertTrue(requireNotNull(instrumentation.uiAutomation.rootInActiveWindow).refresh())
                device.dumpWindowHierarchy(File(directory, "page_$page.xml"))
                device.findObjects(By.pkg(context.packageName)).forEach { node ->
                    node.text?.takeIf(String::isNotBlank)?.let(observed::add)
                    node.contentDescription?.takeIf(String::isNotBlank)?.let(descriptions::add)
                }
                assertTrue(device.takeScreenshot(File(directory, "page_$page.png")))
                pages++
                if (expected.all { wanted -> observed.any { it.contains(wanted) } } &&
                    (1..3).all { "Train service card $it" in descriptions }
                ) break
                device.swipe(device.displayWidth / 2, device.displayHeight * 83 / 100,
                    device.displayWidth / 2, device.displayHeight * 35 / 100, 35)
                Thread.sleep(300)
            }
        }
        expected.forEach { wanted -> assertTrue("Missing rendered value: $wanted\n$observed", observed.any { it.contains(wanted) }) }
        (1..3).forEach { assertTrue("Missing rail card $it", "Train service card $it" in descriptions) }
        assertFalse("Raw data keys leaked into the UI", observed.any {
            it.contains("departure_station") || it.contains("typical_departure_time") ||
                it.contains("journey_duration") || it.contains("seating_class")
        })
        File(directory, "result.json").writeText(GsonBuilder().setPrettyPrinting().create().toJson(linkedMapOf(
            "rawSha256" to sha(raw),
            "repairedSha256" to sha(outcome.document.express.toByteArray()),
            "realInferenceUsed" to false,
            "sourceFallbackAllowed" to false,
            "railCardsObserved" to 3,
            "pagesCaptured" to pages,
            "observedTexts" to observed,
            "observedDescriptions" to descriptions,
        )))
    }

    private fun sha(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
