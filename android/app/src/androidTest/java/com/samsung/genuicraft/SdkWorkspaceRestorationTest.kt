package com.samsung.genuicraft

import android.content.Intent
import android.os.Build
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.Direction
import androidx.test.uiautomator.StaleObjectException
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import com.samsung.genuicraft.sdk.GenUiCompiler
import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SdkWorkspaceRestorationTest {
    @Test
    fun pageScrollSurvivesTabSwitchAndActivityRecreation() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        val document = GenUiCompiler.compile(longDocument())
        val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                activity.showDocument(document, "Workspace restoration fixture")
                ViewModelProvider(activity)[GenUiSdkDemoViewModel::class.java].generationTrace =
                    SdkGenerationTrace(
                        attempts = (1..6).map { number ->
                            SdkGenerationAttempt(number, document.express, complete = true)
                        },
                        phase = SdkGenerationPhase.COMPLETE,
                        finalIr = document.express,
                    )
            }
            val pageSelector = By.pkg(context.packageName).desc("SDK page")
            val irTab = By.pkg(context.packageName).text("Inspect IR")
            val previewTab = By.pkg(context.packageName).text("Preview")
            assertTrue(device.wait(Until.hasObject(pageSelector), 10_000))
            assertTrue(requireNotNull(device.findObject(pageSelector)).isScrollable)
            assertTrue(device.wait(Until.hasObject(previewTab), 5_000))

            // Keep the tabs on screen while checking that switching content leaves the
            // shared page position alone.
            requireNotNull(device.findObject(pageSelector)).scroll(Direction.DOWN, 0.12f)
            refresh(instrumentation, device)
            val tabTop = requireNotNull(visibleTop(device, irTab)) {
                "IR tab left the screen after a shallow page scroll"
            }
            requireNotNull(device.findObject(irTab)).click()
            assertTrue(device.wait(Until.hasObject(By.pkg(context.packageName)
                .textStartsWith("Generated IR")), 5_000))
            requireNotNull(device.findObject(previewTab)).click()
            refresh(instrumentation, device)
            val tabTopAfter = visibleTop(device, irTab)
            assertNotNull("Tab switch reset or displaced the page scroll", tabTopAfter)
            assertTrue("Tab switch changed the page scroll: $tabTop to $tabTopAfter",
                abs(tabTopAfter!! - tabTop) <= POSITION_TOLERANCE_PX)

            // Find an actual rendered row after scrolling the common parent. Its screen
            // position must survive Activity recreation, which exercises rememberSaveable.
            var anchor: Pair<String, Int>? = null
            for (attempt in 0 until 12) {
                refresh(instrumentation, device)
                val pageBounds = requireNotNull(device.findObject(pageSelector)).visibleBounds
                anchor = device.findObjects(By.pkg(context.packageName))
                    .firstNotNullOfOrNull { node ->
                        val text = node.text ?: return@firstNotNullOfOrNull null
                        val row = text.removePrefix("Workspace restoration row ").toIntOrNull()
                        val bounds = node.visibleBounds
                        if (row != null && row >= 10 &&
                            bounds.top > pageBounds.top + 60 &&
                            bounds.bottom < pageBounds.bottom - 60
                        ) text to bounds.top else null
                    }
                if (anchor != null) break
                requireNotNull(device.findObject(pageSelector)).scroll(Direction.DOWN, 0.55f)
            }
            val (anchorText, anchorTop) = requireNotNull(anchor) {
                "The parent page could not scroll to a later rendered row"
            }
            scenario.recreate()
            assertTrue("Rendered row disappeared after Activity recreation",
                device.wait(Until.hasObject(By.pkg(context.packageName).text(anchorText)), 10_000))
            refresh(instrumentation, device)
            val restoredTop = visibleTop(device, By.pkg(context.packageName).text(anchorText))
            assertNotNull("Rendered row is no longer visible after recreation", restoredTop)
            assertTrue(
                "Page position changed after recreation: $anchorText was $anchorTop, now $restoredTop",
                abs(restoredTop!! - anchorTop) <= POSITION_TOLERANCE_PX,
            )
            assertEquals(context.packageName, device.currentPackageName)
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

    private fun longDocument(): String = buildString {
        val rows = (1..60).map { "row$it" }
        appendLine("<a2ui>")
        appendLine("root=Column([${rows.joinToString()}])")
        rows.forEachIndexed { index, id ->
            appendLine("$id=Text(\"Workspace restoration row ${index + 1}\")")
        }
        appendLine("</a2ui>")
    }

    private companion object {
        const val POSITION_TOLERANCE_PX = 24
    }
}
