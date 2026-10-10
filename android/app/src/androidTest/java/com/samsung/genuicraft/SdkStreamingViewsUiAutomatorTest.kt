package com.samsung.genuicraft

import android.content.Context
import android.content.pm.ActivityInfo
import android.content.res.Configuration
import android.os.SystemClock
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.Direction
import androidx.test.uiautomator.StaleObjectException
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.CompletableDeferred
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * UI lifecycle coverage without ComposeRule/Espresso, which cannot initialize its input injector
 * on the Android 17 Fold8 image. The provider deliberately pauses after two cumulative callbacks.
 */
@RunWith(AndroidJUnit4::class)
class SdkStreamingViewsUiAutomatorTest {
    @Test
    fun textAndUiSwitchLiveWithoutAnotherProviderCallAndKeepTheChoiceOnRecreation() {
        withDemoPreferences {
            val instrumentation = InstrumentationRegistry.getInstrumentation()
            val packageName = instrumentation.targetContext.packageName
            val device = UiDevice.getInstance(instrumentation)
            val provider = PausedProvider()
            val originalNightMode = device.executeShellCommand("cmd uimode night")
                .substringAfterLast(':').trim()
            try {
                ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
                    var originalRequestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
                    var originalConfigurationOrientation = Configuration.ORIENTATION_UNDEFINED
                    var originalNightMask = Configuration.UI_MODE_NIGHT_UNDEFINED
                    scenario.onActivity {
                        originalRequestedOrientation = it.requestedOrientation
                        originalConfigurationOrientation = it.resources.configuration.orientation
                        originalNightMask = it.resources.configuration.uiMode and
                            Configuration.UI_MODE_NIGHT_MASK
                    }
                    instrumentation.waitForIdleSync()
                    try {
                    clickText(device, packageName, "Streaming text")
                    assertEquals("TEXT", currentMode())
                    val model = start(scenario, provider, "uiautomator-live-switch")

                    val live = await(scenario) {
                        it.raw == provider.stalledText && it.previewReady && it.working
                    }
                    assertEquals(SdkGenerationPhase.GENERATING, live.phase)
                    assertNull(live.document)
                    assertEquals(1, provider.calls.get())
                    // The second, small callback arrives inside the 75 ms coalescing interval.
                    // It must become visible while the provider is still stalled.
                    assertTrue(device.wait(Until.hasObject(By.pkg(packageName)
                        .text("Generated IR · live")), UI_TIMEOUT))
                    assertFalse("Text mode unexpectedly rendered the native component",
                        device.hasObject(By.pkg(packageName).text(PREVIEW_TEXT)))

                    clickText(device, packageName, "Streaming UI")
                    assertEquals("UI", currentMode())
                    assertTrue("Native component did not appear before completion", device.wait(
                        Until.hasObject(By.pkg(packageName).text(PREVIEW_TEXT)), UI_TIMEOUT,
                    ))
                    assertEquals(1, provider.calls.get())
                    val shown = observe(scenario)
                    assertTrue(shown.working)
                    assertTrue(shown.previewReady)
                    assertNull(shown.document)

                    clickText(device, packageName, "Streaming text")
                    assertEquals("TEXT", currentMode())
                    scenario.recreate()
                    assertTrue(device.wait(Until.hasObject(By.pkg(packageName)
                        .text("Generated IR · live")), UI_TIMEOUT))
                    assertEquals("TEXT", currentMode())
                    assertEquals(1, provider.calls.get())
                    scenario.onActivity {
                        assertTrue("Recreation replaced the active generation", model ===
                            ViewModelProvider(it)[GenUiSdkDemoViewModel::class.java])
                    }

                    val targetOrientation =
                        if (originalConfigurationOrientation == Configuration.ORIENTATION_LANDSCAPE)
                            Configuration.ORIENTATION_PORTRAIT
                        else Configuration.ORIENTATION_LANDSCAPE
                    scenario.onActivity {
                        it.requestedOrientation =
                            if (targetOrientation == Configuration.ORIENTATION_LANDSCAPE)
                                ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
                            else ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
                    }
                    await(scenario) { it.orientation == targetOrientation && it.working }
                    assertRetainedTextRun(scenario, model, provider, packageName, device)

                    val targetNightMode =
                        if (originalNightMask == Configuration.UI_MODE_NIGHT_YES) "no" else "yes"
                    val targetNightMask =
                        if (targetNightMode == "yes") Configuration.UI_MODE_NIGHT_YES
                        else Configuration.UI_MODE_NIGHT_NO
                    device.executeShellCommand("cmd uimode night $targetNightMode")
                    await(scenario) { it.nightMask == targetNightMask && it.working }
                    assertRetainedTextRun(scenario, model, provider, packageName, device)

                    // Return to the starting screen shape before checking final native text.
                    // The landscape viewport can clip the second row below the fold.
                    scenario.onActivity {
                        it.requestedOrientation =
                            if (originalConfigurationOrientation == Configuration.ORIENTATION_LANDSCAPE)
                                ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
                            else ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
                    }
                    await(scenario) {
                        it.orientation == originalConfigurationOrientation && it.working
                    }
                    assertRetainedTextRun(scenario, model, provider, packageName, device)

                    clickText(device, packageName, "Streaming UI")
                    assertTrue(device.wait(Until.hasObject(By.pkg(packageName)
                        .text(PREVIEW_TEXT)), UI_TIMEOUT))
                    provider.finish.complete(Unit)
                    val completed = await(scenario) { it.phase == SdkGenerationPhase.COMPLETE }
                    assertFalse(completed.working)
                    assertNotNull(completed.document)
                    assertEquals(1, provider.calls.get())
                    assertNativeTextVisibleAfterScroll(device, packageName, FINAL_TEXT)
                    clickText(device, packageName, "Inspect IR")
                    assertTrue(device.wait(Until.hasObject(By.pkg(packageName)
                        .textStartsWith("Generated IR")), UI_TIMEOUT))
                    } finally {
                        scenario.onActivity {
                            it.requestedOrientation = originalRequestedOrientation
                        }
                    }
                }
            } finally {
                provider.finish.complete(Unit)
                device.executeShellCommand("cmd uimode night $originalNightMode")
            }
        }
    }

    @Test
    fun cancellingAStalledPreviewClearsNativeUiButRetainsRawText() {
        withDemoPreferences {
            val instrumentation = InstrumentationRegistry.getInstrumentation()
            val packageName = instrumentation.targetContext.packageName
            val device = UiDevice.getInstance(instrumentation)
            val provider = PausedProvider()
            try {
                ActivityScenario.launch(GenUiSdkDemoActivity::class.java).use { scenario ->
                    instrumentation.waitForIdleSync()
                    val model = start(scenario, provider, "uiautomator-cancel")
                    await(scenario) { it.previewReady && it.raw == provider.stalledText }
                    assertTrue(device.wait(Until.hasObject(By.pkg(packageName)
                        .text(PREVIEW_TEXT)), UI_TIMEOUT))
                    scenario.onActivity { model.cancelGeneration() }
                    val cancelled = await(scenario) {
                        it.phase == SdkGenerationPhase.CANCELLED && !it.working
                    }
                    assertFalse(cancelled.previewReady)
                    assertNull(cancelled.document)
                    assertEquals(provider.stalledText, cancelled.raw)
                    assertEquals(1, provider.calls.get())
                    clickText(device, packageName, "Inspect IR")
                    assertTrue(device.wait(Until.hasObject(By.pkg(packageName)
                        .textStartsWith("Generated IR")), UI_TIMEOUT))
                }
            } finally {
                provider.finish.complete(Unit)
            }
        }
    }

    private fun start(
        scenario: ActivityScenario<GenUiSdkDemoActivity>,
        provider: PausedProvider,
        providerPath: String,
    ): GenUiSdkDemoViewModel {
        lateinit var model: GenUiSdkDemoViewModel
        scenario.onActivity {
            model = ViewModelProvider(it)[GenUiSdkDemoViewModel::class.java]
            model.generate(
                source = "$PREVIEW_TEXT. $FINAL_TEXT.",
                e2bModelChoiceForRun = E2bModelChoice.TRAINED_E2B_V10_W4,
                modelPathForRun = providerPath,
                metricsForRun = false,
                mtpForRun = false,
                streamingRenderingForRun = true,
                providerFactory = { provider },
            )
        }
        return model
    }

    private fun observe(scenario: ActivityScenario<GenUiSdkDemoActivity>): Observation {
        val result = AtomicReference<Observation>()
        scenario.onActivity { activity ->
            val model = ViewModelProvider(activity)[GenUiSdkDemoViewModel::class.java]
            result.set(Observation(
                phase = model.generationTrace.phase,
                raw = model.generationTrace.attempts.firstOrNull()?.rawText.orEmpty(),
                working = model.working,
                previewReady = model.renderSnapshot?.isFinal == false,
                document = model.document,
                orientation = activity.resources.configuration.orientation,
                nightMask = activity.resources.configuration.uiMode and
                    Configuration.UI_MODE_NIGHT_MASK,
            ))
        }
        return requireNotNull(result.get())
    }

    private fun assertRetainedTextRun(
        scenario: ActivityScenario<GenUiSdkDemoActivity>,
        model: GenUiSdkDemoViewModel,
        provider: PausedProvider,
        packageName: String,
        device: UiDevice,
    ) {
        scenario.onActivity {
            assertTrue("Configuration change replaced the active generation", model ===
                ViewModelProvider(it)[GenUiSdkDemoViewModel::class.java])
        }
        val current = observe(scenario)
        assertTrue(current.working)
        assertTrue(current.previewReady)
        assertEquals(provider.stalledText, current.raw)
        assertEquals("TEXT", currentMode())
        assertEquals(1, provider.calls.get())
        assertTrue(device.wait(Until.hasObject(By.pkg(packageName)
            .text("Generated IR · live")), UI_TIMEOUT))
    }

    private fun await(
        scenario: ActivityScenario<GenUiSdkDemoActivity>,
        predicate: (Observation) -> Boolean,
    ): Observation {
        val deadline = SystemClock.elapsedRealtime() + UI_TIMEOUT
        var last = observe(scenario)
        while (!predicate(last) && SystemClock.elapsedRealtime() < deadline) {
            Thread.sleep(40)
            last = observe(scenario)
        }
        assertTrue("Timed out waiting for SDK state: $last", predicate(last))
        return last
    }

    private fun clickText(device: UiDevice, packageName: String, label: String) {
        val text = requireNotNull(device.wait(
            Until.findObject(By.pkg(packageName).text(label)), UI_TIMEOUT,
        )) { "Timed out waiting for $label" }
        // Click the observed text bounds: Compose's touch parent can be omitted from the
        // Android 17 accessibility clickable flag even though the control is functional.
        text.click()
    }

    private fun assertNativeTextVisibleAfterScroll(
        device: UiDevice,
        packageName: String,
        text: String,
    ) {
        val target = By.pkg(packageName).text(text)
        if (device.hasObject(target)) return
        repeat(5) {
            try {
                var scroller: UiObject2? = device.findObject(
                    By.pkg(packageName).text(PREVIEW_TEXT),
                )
                while (scroller != null && !scroller.isScrollable) scroller = scroller.parent
                val freshScroller = scroller ?: device.findObjects(
                    By.pkg(packageName).scrollable(true),
                ).lastOrNull()
                freshScroller?.scroll(Direction.DOWN, 0.8f)
            } catch (_: StaleObjectException) {
                // A final snapshot can replace the accessibility tree during a gesture.
                // Retry with fresh objects on the next bounded pass.
            }
            if (device.wait(Until.hasObject(target), 500L)) return
        }
        assertTrue("Final native text was not visible after scrolling the preview: $text",
            device.hasObject(target))
    }

    private fun withDemoPreferences(block: () -> Unit) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext
        val prefs = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val modePresent = prefs.contains(PREFERENCE_STREAMING_VIEW_MODE)
        val previousMode = prefs.getString(PREFERENCE_STREAMING_VIEW_MODE, null)
        val enabledPresent = prefs.contains(PREFERENCE_STREAMING_UI_ENABLED)
        val previousEnabled = prefs.getBoolean(PREFERENCE_STREAMING_UI_ENABLED, true)
        try {
            check(prefs.edit().putString(PREFERENCE_STREAMING_VIEW_MODE, "UI")
                .putBoolean(PREFERENCE_STREAMING_UI_ENABLED, true).commit())
            block()
        } finally {
            val editor = prefs.edit().remove(PREFERENCE_STREAMING_VIEW_MODE)
                .remove(PREFERENCE_STREAMING_UI_ENABLED)
            if (modePresent) editor.putString(PREFERENCE_STREAMING_VIEW_MODE, previousMode)
            if (enabledPresent) editor.putBoolean(PREFERENCE_STREAMING_UI_ENABLED, previousEnabled)
            check(editor.commit())
        }
    }

    private fun currentMode(): String? = InstrumentationRegistry.getInstrumentation()
        .targetContext.applicationContext.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        .getString(PREFERENCE_STREAMING_VIEW_MODE, null)

    private class PausedProvider : GenUiProvider {
        override val id = "uiautomator-streaming-view-test"
        val finish = CompletableDeferred<Unit>()
        val calls = AtomicInteger()
        private val firstText = "<a2ui>\nroot=Column([title,detail])\n" +
            "title=Text(\"$PREVIEW_TEXT\")\n"
        val stalledText = firstText + "detail=Text(\"$FINAL_TEXT\")\n"

        override suspend fun generate(prompt: GenUiPrompt) = generate(prompt) {}

        override suspend fun generate(
            prompt: GenUiPrompt,
            onPartialText: (String) -> Unit,
        ): GenUiModelOutput {
            calls.incrementAndGet()
            onPartialText(firstText)
            onPartialText(stalledText)
            finish.await()
            val finalText = stalledText + "</a2ui>"
            onPartialText(finalText)
            return GenUiModelOutput(finalText, "uiautomator-streaming-view-test/streaming")
        }

        override fun close() = Unit
    }

    private data class Observation(
        val phase: SdkGenerationPhase,
        val raw: String,
        val working: Boolean,
        val previewReady: Boolean,
        val document: com.samsung.genuicraft.sdk.GenUiDocument?,
        val orientation: Int,
        val nightMask: Int,
    )

    private companion object {
        const val PREFERENCES = "genuicraft_sdk_demo"
        const val PREVIEW_TEXT = "Native preview before completion"
        const val FINAL_TEXT = "Final native detail"
        const val UI_TIMEOUT = 15_000L
    }
}
