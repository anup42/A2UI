package com.samsung.genuicraft

import android.content.Intent
import android.os.Build
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.Direction
import androidx.test.uiautomator.StaleObjectException
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiRepairKind
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Replays the unchanged captured train output through the SDK demo's whole-page scroll. */
@RunWith(AndroidJUnit4::class)
class SdkWholePageScrollOnDeviceTest {
    @Test
    fun swipeOnHeaderMovesControlsAwayAndPageScrollReachesEveryTrainCard() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val raw = instrumentation.context.assets.open("train_comparison_explicit_table_captured.express")
            .use { it.readBytes() }
        assertEquals(EXPECTED_SHA256, sha256(raw))
        val outcome = GenUiCompiler.compileWithRepair(
            raw.toString(Charsets.UTF_8),
            allowSourceTextFallback = false,
            allowGeneratedDslRepair = false,
        )
        assertEquals(GenUiRepairKind.NONE, outcome.repairKind)

        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            scenario.onActivity { it.showDocument(outcome.document, "Train comparison · captured output · no inference") }
            val pageSelector = By.pkg(context.packageName).desc("SDK page")
            assertTrue("Missing whole-page scroll", device.wait(Until.hasObject(pageSelector), 10_000))
            assertTrue("SDK page is not scrollable", requireNotNull(device.findObject(pageSelector)).isScrollable)
            val headerSelector = By.pkg(context.packageName).text("GenUICraft SDK")
            val controlsSelector = By.pkg(context.packageName).text("View for next generation")
            assertTrue("Missing SDK header", device.wait(Until.hasObject(headerSelector), 5_000))
            assertTrue("Missing streaming view controls", device.hasObject(controlsSelector))
            val headerBefore = requireNotNull(device.findObject(headerSelector)).visibleBounds
            val pageTop = requireNotNull(device.findObject(pageSelector)).visibleBounds.top
            val swipeEndY = pageTop + 8
            assertTrue("Header has no room for an upward gesture", headerBefore.centerY() > swipeEndY + 12)

            // Start on the title itself. Scrolling only the output pane cannot move this title.
            device.swipe(headerBefore.centerX(), headerBefore.centerY(),
                headerBefore.centerX(), swipeEndY, 35)
            refresh(instrumentation, device)
            val headerAfter = visibleTop(device, headerSelector)
            assertTrue(
                "An upward swipe on the header did not move the whole page",
                headerAfter == null || headerAfter < headerBefore.top - 8,
            )

            val seenCards = mutableSetOf<Int>()
            val seenNames = mutableSetOf<String>()
            for (attempt in 0 until 12) {
                refresh(instrumentation, device)
                device.findObjects(By.pkg(context.packageName)).forEach { node ->
                    val description = node.contentDescription
                    (1..3).forEach { index ->
                        if (description == "Train service card $index") seenCards += index
                    }
                    node.text?.let { text ->
                        TRAIN_NAMES.forEach { name -> if (name in text) seenNames += name }
                    }
                }
                if (seenCards.size == 3 && seenNames.size == 3 &&
                    visibleTop(device, headerSelector) == null &&
                    visibleTop(device, controlsSelector) == null
                ) break
                requireNotNull(device.findObject(pageSelector)) { "SDK page disappeared while scrolling" }
                    .scroll(Direction.DOWN, 0.45f)
            }
            assertEquals("Some train cards were unreachable through the page scroll", setOf(1, 2, 3), seenCards)
            assertEquals("Some train names were unreachable through the page scroll", TRAIN_NAMES, seenNames)
            assertFalse("SDK header remained pinned above the output", visibleTop(device, headerSelector) != null)
            assertFalse("Streaming controls remained pinned above the output", visibleTop(device, controlsSelector) != null)
        }
    }

    private fun refresh(instrumentation: android.app.Instrumentation, device: UiDevice) {
        instrumentation.waitForIdleSync()
        device.waitForIdle(1_000)
        if (Build.VERSION.SDK_INT >= 34) instrumentation.uiAutomation.clearCache()
    }

    private fun visibleTop(device: UiDevice, selector: androidx.test.uiautomator.BySelector): Int? =
        try {
            device.findObject(selector)?.visibleBounds?.takeIf { it.width() > 0 && it.height() > 0 }?.top
        } catch (_: StaleObjectException) {
            null
        }

    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private companion object {
        const val EXPECTED_SHA256 = "4bd66cf5d55bc27bf11991bd2ff562ea37b75f12b3442be49d85f03d4799bd6b"
        val TRAIN_NAMES = setOf(
            "Shatabdi Express (12007)",
            "Rajya Rani Express (20660)",
            "Wodeyar Express (12614)",
        )
    }
}
