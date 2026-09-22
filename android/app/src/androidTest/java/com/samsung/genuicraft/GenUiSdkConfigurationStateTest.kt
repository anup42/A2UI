package com.samsung.genuicraft

import android.content.pm.ActivityInfo
import android.content.res.Configuration
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.assertTextEquals
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextReplacement
import androidx.lifecycle.ViewModelProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GenUiSdkConfigurationStateTest {
    @get:Rule val compose = createAndroidComposeRule<GenUiSdkDemoActivity>()

    @Test
    fun selectedSampleEditedInputAndOpenSettingsSurviveRecreation() {
        compose.onNodeWithText("Next").performScrollTo().performClick()
        compose.onNodeWithText("Next").performScrollTo().performClick()
        compose.onNodeWithTag("sdk_case_position").assertTextEquals("3 / 50")
        val sample = sourceText()
        compose.activityRule.scenario.recreate()
        compose.onNodeWithTag("sdk_case_position").assertTextEquals("3 / 50")
        assertEquals(sample, sourceText())

        val custom = "Keep this edited train comparison after rotating."
        compose.onNodeWithTag("sdk_source_text").performScrollTo().performTextReplacement(custom)
        compose.onNodeWithTag("sdk_settings_button").performClick()
        compose.activityRule.scenario.recreate()
        compose.onNodeWithText("Demo settings").assertExists()
        compose.onNodeWithText("Done").performClick()
        assertEquals(custom, sourceText())
        compose.onNodeWithTag("sdk_case_position").assertTextEquals("Custom input")
        compose.onNodeWithText("Next").performScrollTo().performClick()
        compose.onNodeWithTag("sdk_case_position").assertTextEquals("4 / 50")
    }

    @Test
    fun streamingSessionSurvivesRecreationRotationAndThemeThenClosesOnExit() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val device = UiDevice.getInstance(instrumentation)
        val originalNightMode = device.executeShellCommand("cmd uimode night")
            .substringAfterLast(':').trim()
        val originalOrientation = compose.activity.requestedOrientation
        val originalNightMask = compose.activity.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK
        val targetNightMode = if (originalNightMask == Configuration.UI_MODE_NIGHT_YES) "no" else "yes"
        val targetNightMask = if (targetNightMode == "yes") Configuration.UI_MODE_NIGHT_YES else Configuration.UI_MODE_NIGHT_NO
        val raw = instrumentation.context.assets.open("train_comparison_captured.express")
            .bufferedReader().use { it.readText() }
        val provider = PausedStreamingProvider(raw)
        lateinit var retained: GenUiSdkDemoViewModel
        try {
            compose.runOnUiThread {
                retained = ViewModelProvider(compose.activity)[GenUiSdkDemoViewModel::class.java]
                retained.generate(
                    source = "Train comparison",
                    useGemmaForRun = true,
                    e2bModelChoiceForRun = E2bModelChoice.TRAINED_E2B_V10_W4,
                    modelPathForRun = "lifecycle-test-provider",
                    metricsForRun = true,
                    mtpForRun = false,
                    providerFactory = { provider },
                )
            }
            compose.waitUntil(10_000) { retained.generationTrace.attempts.firstOrNull()?.rawText?.isNotEmpty() == true }
            val partial = retained.generationTrace.attempts.single().rawText
            val originalActivity = compose.activity
            compose.activityRule.scenario.recreate()
            compose.runOnUiThread {
                assertNotSame(originalActivity, compose.activity)
                assertSame(retained, ViewModelProvider(compose.activity)[GenUiSdkDemoViewModel::class.java])
                assertTrue(retained.working)
                assertEquals(partial, retained.generationTrace.attempts.single().rawText)
            }

            compose.runOnUiThread { compose.activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE }
            compose.waitUntil(10_000) { compose.activity.resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE }
            assertRetainedSession(retained, provider)

            device.executeShellCommand("cmd uimode night $targetNightMode")
            compose.waitUntil(10_000) {
                (compose.activity.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) == targetNightMask
            }
            assertRetainedSession(retained, provider)
            compose.onNodeWithTag("sdk_ir_tab").assertIsSelected()

            provider.finish.complete(Unit)
            compose.waitUntil(15_000) { !retained.working }
            assertEquals(SdkGenerationPhase.COMPLETE, retained.generationTrace.phase)
            assertEquals(raw, retained.generationTrace.attempts.single().rawText)
            assertTrue(retained.generationTrace.attempts.single().complete)
            assertEquals("lifecycle-test/streaming", retained.lastRuntime)
            assertEquals(438L, retained.generationMetrics?.totalOutputTokens)
            val completedDocument = requireNotNull(retained.document)
            val completedTrace = retained.generationTrace
            val completedMetrics = retained.generationMetrics
            val completedStatus = retained.status
            compose.onNodeWithTag("sdk_preview_tab").assertIsSelected()

            // Returning to Inspect IR must stay there instead of rerunning the completion effect.
            compose.onNodeWithTag("sdk_ir_tab").performClick()
            compose.activityRule.scenario.recreate()
            compose.onNodeWithTag("sdk_ir_tab").assertIsSelected()
            compose.onNodeWithTag("sdk_input_editor").assertDoesNotExist()
            compose.runOnUiThread {
                assertEquals(completedDocument, compose.activity.renderedDocumentForTest())
                assertEquals(completedTrace, compose.activity.generationTraceForTest())
                assertEquals(completedMetrics, retained.generationMetrics)
                assertEquals(completedStatus, compose.activity.statusForTest())
            }
            assertEquals(1, provider.calls.get())
            assertEquals(0, provider.closes.get())
        } finally {
            provider.finish.complete(Unit)
            device.executeShellCommand("cmd uimode night $originalNightMode")
            compose.runOnUiThread { compose.activity.requestedOrientation = originalOrientation }
            compose.activityRule.scenario.close()
        }
        assertEquals("The retained provider must be closed only when leaving the page", 1, provider.closes.get())
        assertFalse(retained.working)
    }

    private fun assertRetainedSession(expected: GenUiSdkDemoViewModel, provider: PausedStreamingProvider) {
        compose.runOnUiThread {
            assertSame(expected, ViewModelProvider(compose.activity)[GenUiSdkDemoViewModel::class.java])
            assertTrue(expected.working)
        }
        assertEquals(1, provider.calls.get())
        assertEquals(0, provider.closes.get())
    }

    private fun sourceText(): String = compose.onNodeWithTag("sdk_source_text")
        .fetchSemanticsNode().config[SemanticsProperties.EditableText].text

    /** Deterministic partial output pauses across real lifecycle events; no model/network call. */
    private class PausedStreamingProvider(private val raw: String) : GenUiProvider {
        override val id = "lifecycle-test"
        val finish = CompletableDeferred<Unit>()
        val calls = AtomicInteger()
        val closes = AtomicInteger()
        override suspend fun generate(prompt: GenUiPrompt) = generate(prompt) {}
        override suspend fun generate(prompt: GenUiPrompt, onPartialText: (String) -> Unit): GenUiModelOutput {
            calls.incrementAndGet()
            onPartialText(raw.take(100))
            finish.await()
            onPartialText(raw)
            return GenUiModelOutput(raw, "lifecycle-test/streaming", 438, GenUiGenerationMetrics(3394, 438, 40.0))
        }
        override fun close() { closes.incrementAndGet() }
    }
}
