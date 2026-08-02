package com.samsung.genuicraft

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Device-only smoke test for the active Express parser/compiler/render path. */
@RunWith(AndroidJUnit4::class)
class A2uiExpressRenderOnDeviceTest {
    @Test
    fun expressFixtureRendersOnFlip() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val outputDir = File(
            context.getExternalFilesDir(null) ?: context.filesDir,
            "result_device_a2ui_express",
        ).apply { mkdirs() }
        val express = """
            <a2ui>
            root=Column([title,summary,details],gap="lg")
            title=Text("A2UI Express device smoke","h2")
            summary=Card([summary_text],"Render status")
            summary_text=Text("Rendered on the connected Flip","bodyMedium")
            details=Table(["Field","Value"],rows=[["IR","A2UI Express"],["Status","Ready"]],title="Details",domain="status",preferredPresentation="cards")
            </a2ui>
        """.trimIndent()

        RenderSessionStore.update(
            sourceLabel = "instrumentation",
            records = listOf(
                GenUiRecord(
                    title = "a2ui_express_device_smoke",
                    rawJson = express,
                    sourceDir = null,
                    sourceLabel = "instrumentation",
                    uiId = "a2ui_express_device_smoke",
                    summary = null,
                    queryId = null,
                    responseId = null,
                )
            ),
            renderMode = RenderMode.NATIVE,
        )

        val intent = Intent(context, RenderActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            putExtra(RenderActivity.EXTRA_RECORD_INDEX, 0)
        }
        ActivityScenario.launch<RenderActivity>(intent).use {
            instrumentation.waitForIdleSync()
            Thread.sleep(1_500L)
            val device = UiDevice.getInstance(instrumentation)
            val screenshot = File(outputDir, "screenshot.png")
            assertTrue("Express render screenshot was not captured", device.takeScreenshot(screenshot))
            val hierarchy = File(outputDir, "window.xml")
            device.dumpWindowHierarchy(hierarchy)
            // Keep a host-pullable copy because the test runner may clear the app's
            // external-files directory during teardown. These are review artifacts,
            // not application data.
            device.executeShellCommand("mkdir -p /sdcard/Download/a2ui_express_device_smoke")
            device.executeShellCommand(
                "screencap -p /sdcard/Download/a2ui_express_device_smoke/screenshot.png"
            )
            device.executeShellCommand(
                "uiautomator dump /sdcard/Download/a2ui_express_device_smoke/window.xml"
            )
            val xml = hierarchy.readText()
            assertTrue(
                "Express title was not visible in the device hierarchy: ${xml.take(600)}",
                xml.contains("A2UI Express device smoke") ||
                    device.findObject(By.textContains("A2UI Express")) != null,
            )
        }
    }
}
