package com.samsung.genuicraft

import androidx.compose.ui.test.assertIsOff
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.GenUiGenerationMetrics
import com.samsung.genuicraft.sdk.GenUiModelOutput
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GenUiSdkMetricsTest {
    @get:Rule val compose = createAndroidComposeRule<GenUiSdkDemoActivity>()

    @Test fun metricsTogglePersistsAndRendererOnlyClearsStaleCounters() {
        val toggle = compose.onNodeWithTag("token_metrics_switch")
        // Start from an explicit enabled preference so this test is repeatable.
        if (compose.activity.getSharedPreferences("genuicraft_sdk_demo", 0)
                .getBoolean("token_metrics_enabled", true).not()) toggle.performClick()
        toggle.assertIsOn()
        val document = GenUiCompiler.compile("<a2ui>\nroot=Text(\"Metrics UI test\")\n</a2ui>")
        compose.runOnUiThread {
            compose.activity.showDocument(document)
            compose.activity.showGenerationMetrics(
                listOf(GenUiModelOutput("fixture", "test/GPU+MTP", 80, GenUiGenerationMetrics(120, 80, 40.0))),
                3500,
            )
        }
        compose.onNodeWithTag("generation_metrics_panel").assertExists()
        compose.onNodeWithText("Details").performClick()
        compose.onNodeWithTag("generation_metrics_attempt_1").assertExists()
        toggle.performClick()
        toggle.assertIsOff()
        compose.onNodeWithTag("generation_metrics_panel").assertDoesNotExist()
        compose.activityRule.scenario.recreate()
        compose.onNodeWithTag("token_metrics_switch").assertIsOff().performClick().assertIsOn()
        compose.activityRule.scenario.recreate()
        compose.onNodeWithTag("token_metrics_switch").assertIsOn()
        compose.runOnUiThread {
            compose.activity.showGenerationMetrics(
                listOf(GenUiModelOutput("fixture", "test/GPU+MTP", 80, GenUiGenerationMetrics(120, 80, 40.0))),
                3500,
            )
        }
        compose.onNodeWithTag("generation_metrics_panel").assertExists()
        compose.runOnUiThread { compose.activity.showDocument(document, "Renderer only · no model call") }
        compose.onNodeWithTag("generation_metrics_panel").assertDoesNotExist()
        compose.onNodeWithText("Metrics UI test").assertExists()
    }
}
