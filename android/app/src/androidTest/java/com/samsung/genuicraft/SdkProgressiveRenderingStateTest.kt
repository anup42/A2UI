package com.samsung.genuicraft

import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.lifecycle.ViewModelProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.samsung.genuicraft.sdk.GenUiModelOutput
import com.samsung.genuicraft.sdk.GenUiPrompt
import com.samsung.genuicraft.sdk.GenUiProvider
import kotlinx.coroutines.CompletableDeferred
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** Exercises actual Compose preview state with a provider paused before its final response. */
@RunWith(AndroidJUnit4::class)
class SdkProgressiveRenderingStateTest {
    @get:Rule val compose = createAndroidComposeRule<GenUiSdkDemoActivity>()

    @Test
    fun nativePreviewSurvivesRecreationAndFinalResultReplacesIt() {
        val provider = PausedProvider()
        val model = start(provider)
        try {
            awaitPreview(model)
            val first = requireNotNull(model.renderSnapshot)
            assertFalse(first.isFinal)
            assertTrue(first.readyComponentCount > 0)
            assertTrue(model.working)
            assertNull("A preview must not become an authoritative result", model.document)
            compose.onNodeWithTag("sdk_preview_tab").assertIsSelected()
            compose.onNodeWithText(PREVIEW_TEXT).assertExists()
            compose.activityRule.scenario.recreate()
            compose.onNodeWithText(PREVIEW_TEXT).assertExists()
            assertEquals(first.surfaceKey, model.renderSnapshot?.surfaceKey)

            provider.finish.complete(Unit)
            compose.waitUntil(15_000) { !model.working }
            assertEquals(SdkGenerationPhase.COMPLETE, model.generationTrace.phase)
            assertNotNull(model.document)
            assertTrue(requireNotNull(model.renderSnapshot).isFinal)
            assertEquals(first.surfaceKey, model.renderSnapshot?.surfaceKey)
            assertEquals(model.document?.a2uiJson, model.renderSnapshot?.document?.a2uiJson)
            compose.onNodeWithText(FINAL_TEXT).assertExists()
            assertTrue(model.generationTrace.previewSnapshotCount > 0)
            assertNotNull(model.generationTrace.firstPreviewFrameElapsedMs)
        } finally {
            provider.finish.complete(Unit)
        }
    }

    @Test
    fun failedProviderDiscardsPreviewAndRetainsRawDiagnosticOutput() {
        val provider = PausedProvider(fail = true)
        val model = start(provider)
        try {
            awaitPreview(model)
            provider.finish.complete(Unit)
            compose.waitUntil(15_000) { !model.working }
            assertEquals(SdkGenerationPhase.FAILED, model.generationTrace.phase)
            assertNull(model.renderSnapshot)
            assertNull(model.document)
            assertTrue(model.generationTrace.attempts.single().rawText.contains(PREVIEW_TEXT))
            compose.onNodeWithTag("sdk_ir_tab").assertIsSelected()
        } finally {
            provider.finish.complete(Unit)
        }
    }

    @Test
    fun cancellationDiscardsPreviewAndStreamingDisabledWaitsForFinalResult() {
        val provider = PausedProvider()
        val model = start(provider)
        awaitPreview(model)
        compose.runOnUiThread { model.cancelGeneration() }
        compose.waitUntil(10_000) { !model.working }
        assertEquals(SdkGenerationPhase.CANCELLED, model.generationTrace.phase)
        assertNull(model.renderSnapshot)
        assertNull(model.document)

        val disabledProvider = PausedProvider()
        start(disabledProvider, streaming = false, path = "progressive-disabled-provider")
        try {
            compose.waitUntil(10_000) { model.generationTrace.attempts.singleOrNull()?.rawText?.isNotEmpty() == true }
            compose.waitForIdle()
            assertTrue(model.working)
            assertNull(model.renderSnapshot)
            assertEquals(0, model.generationTrace.previewSnapshotCount)
            disabledProvider.finish.complete(Unit)
            compose.waitUntil(15_000) { !model.working }
            assertEquals(SdkGenerationPhase.COMPLETE, model.generationTrace.phase)
            assertNotNull(model.document)
            assertNull(model.renderSnapshot)
            compose.onNodeWithText(FINAL_TEXT).assertExists()
        } finally {
            disabledProvider.finish.complete(Unit)
            provider.finish.complete(Unit)
        }
    }

    private fun start(
        provider: PausedProvider,
        streaming: Boolean = true,
        path: String = "progressive-state-provider",
    ): GenUiSdkDemoViewModel {
        lateinit var model: GenUiSdkDemoViewModel
        compose.runOnUiThread {
            model = ViewModelProvider(compose.activity)[GenUiSdkDemoViewModel::class.java]
            model.generate(
                source = "$PREVIEW_TEXT. $FINAL_TEXT.",
                e2bModelChoiceForRun = E2bModelChoice.TRAINED_E2B_V10_W4,
                modelPathForRun = path,
                metricsForRun = false,
                mtpForRun = false,
                streamingRenderingForRun = streaming,
                providerFactory = { provider },
            )
        }
        return model
    }

    private fun awaitPreview(model: GenUiSdkDemoViewModel) {
        compose.waitUntil(10_000) { model.renderSnapshot != null }
        compose.waitForIdle()
    }

    private class PausedProvider(private val fail: Boolean = false) : GenUiProvider {
        override val id = "progressive-state-test"
        val finish = CompletableDeferred<Unit>()
        override suspend fun generate(prompt: GenUiPrompt) = generate(prompt) {}
        override suspend fun generate(prompt: GenUiPrompt, onPartialText: (String) -> Unit): GenUiModelOutput {
            val prefix = "<a2ui>\nroot=Column([title,detail])\ntitle=Text(\"$PREVIEW_TEXT\")\n"
            onPartialText(prefix)
            finish.await()
            if (fail) error("Provider failed after native preview")
            val final = prefix + "detail=Text(\"$FINAL_TEXT\")\n</a2ui>"
            onPartialText(final)
            return GenUiModelOutput(final, "progressive-state-test/streaming")
        }
        override fun close() = Unit
    }

    private companion object {
        const val PREVIEW_TEXT = "Native preview before completion"
        const val FINAL_TEXT = "Final native detail"
    }
}
