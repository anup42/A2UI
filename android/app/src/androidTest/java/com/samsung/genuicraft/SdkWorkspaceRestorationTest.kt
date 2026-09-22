package com.samsung.genuicraft

import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.swipeUp
import androidx.lifecycle.ViewModelProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.samsung.genuicraft.sdk.GenUiCompiler
import kotlin.math.abs
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SdkWorkspaceRestorationTest {
    @get:Rule val compose = createAndroidComposeRule<GenUiSdkDemoActivity>()

    @Test
    fun previewScrollSurvivesTabSwitchAndActivityRecreation() {
        val document = GenUiCompiler.compile(longDocument())
        lateinit var screen: GenUiSdkDemoViewModel
        compose.runOnUiThread {
            compose.activity.showDocument(document, "Workspace restoration fixture")
            screen = ViewModelProvider(compose.activity)[GenUiSdkDemoViewModel::class.java]
        }
        compose.onNodeWithTag("sdk_preview_tab").assertIsSelected()

        compose.runOnUiThread {
            screen.generationTrace = SdkGenerationTrace(
                attempts = listOf(SdkGenerationAttempt(1, document.express, complete = true)),
                phase = SdkGenerationPhase.COMPLETE,
                finalIr = document.express,
            )
        }
        compose.onNodeWithTag("sdk_preview_tab").assertIsSelected()
        repeat(2) {
            compose.onNodeWithTag("sdk_preview_content").performTouchInput { swipeUp() }
        }
        val scrolledPosition = previewScrollPosition()
        assertTrue("Preview fixture did not scroll", scrolledPosition > 0f)

        compose.onNodeWithTag("sdk_ir_tab").performClick().assertIsSelected()
        compose.onNodeWithTag("sdk_preview_tab").performClick().assertIsSelected()
        assertScrollRestored(scrolledPosition, "after switching tabs")

        compose.activityRule.scenario.recreate()
        compose.onNodeWithTag("sdk_preview_tab").assertIsSelected()
        assertScrollRestored(scrolledPosition, "after Activity recreation")
    }

    private fun assertScrollRestored(expected: Float, context: String) {
        compose.waitUntil(5_000) {
            abs(previewScrollPosition() - expected) < SCROLL_TOLERANCE_PX
        }
        val actual = previewScrollPosition()
        assertTrue(
            "Preview scroll changed $context: expected $expected, actual $actual",
            abs(actual - expected) < SCROLL_TOLERANCE_PX,
        )
    }

    private fun previewScrollPosition(): Float = compose.onNodeWithTag("sdk_preview_content")
        .fetchSemanticsNode().config[SemanticsProperties.VerticalScrollAxisRange].value()

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
        const val SCROLL_TOLERANCE_PX = 1f
    }
}
