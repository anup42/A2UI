package com.samsung.genuicraft

import android.content.Context
import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.BySelector
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GenUiSdkMtpSettingsTest {
    @Test
    fun mtpDrafterDefaultsOnAndPersistsBothValuesAcrossActivityRecreation() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext.applicationContext
        val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)
        val preferenceOriginallyPresent = preferences.contains(PREFERENCE_E2B_MTP_ENABLED)
        val originalPreferenceValue = preferences.getBoolean(PREFERENCE_E2B_MTP_ENABLED, true)
        check(preferences.edit().remove(PREFERENCE_E2B_MTP_ENABLED).commit())

        try {
            val intent = Intent(context, GenUiSdkDemoActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            }
            ActivityScenario.launch<GenUiSdkDemoActivity>(intent).use { scenario ->
                val device = UiDevice.getInstance(instrumentation)
                device.waitForIdle()

                openInferenceSettings(device)
                waitForMtpSwitch(device, expectedChecked = true).click()
                waitForMtpSwitch(device, expectedChecked = false)
                assertTrue(preferences.contains(PREFERENCE_E2B_MTP_ENABLED))
                assertFalse(preferences.getBoolean(PREFERENCE_E2B_MTP_ENABLED, true))
                closeInferenceSettings(device)

                scenario.recreate()
                device.waitForIdle()
                openInferenceSettings(device)
                waitForMtpSwitch(device, expectedChecked = false).click()
                waitForMtpSwitch(device, expectedChecked = true)
                assertTrue(preferences.getBoolean(PREFERENCE_E2B_MTP_ENABLED, false))
                closeInferenceSettings(device)

                scenario.recreate()
                device.waitForIdle()
                openInferenceSettings(device)
                waitForMtpSwitch(device, expectedChecked = true)
                closeInferenceSettings(device)
            }
        } finally {
            val editor = preferences.edit()
            if (preferenceOriginallyPresent) {
                editor.putBoolean(PREFERENCE_E2B_MTP_ENABLED, originalPreferenceValue)
            } else {
                editor.remove(PREFERENCE_E2B_MTP_ENABLED)
            }
            check(editor.commit())
            check(preferences.contains(PREFERENCE_E2B_MTP_ENABLED) == preferenceOriginallyPresent)
            if (preferenceOriginallyPresent) {
                check(
                    preferences.getBoolean(PREFERENCE_E2B_MTP_ENABLED, !originalPreferenceValue) ==
                        originalPreferenceValue
                )
            }
        }
    }

    private fun openInferenceSettings(device: UiDevice) {
        waitForObject(device, By.text("Settings"), "Settings button").click()
        waitForObject(device, By.text("Inference settings"), "Inference settings dialog")
        waitForObject(device, By.text("MTP drafter"), "MTP drafter label")
    }

    private fun closeInferenceSettings(device: UiDevice) {
        waitForObject(device, By.text("Done"), "Done button").click()
        assertTrue(
            "Inference settings dialog did not close",
            device.wait(Until.gone(By.text("Inference settings")), UI_TIMEOUT_MS),
        )
    }

    private fun waitForMtpSwitch(device: UiDevice, expectedChecked: Boolean): UiObject2 {
        val selector = By.desc("MTP drafter").checkable(true).checked(expectedChecked)
        return waitForObject(
            device,
            selector,
            "MTP drafter switch checked=$expectedChecked",
        )
    }

    private fun waitForObject(device: UiDevice, selector: BySelector, label: String): UiObject2 {
        return device.wait(Until.findObject(selector), UI_TIMEOUT_MS)
            ?: error("Timed out waiting for $label")
    }

    private companion object {
        const val PREFERENCES_NAME = "genuicraft_sdk_demo"
        const val PREFERENCE_E2B_MTP_ENABLED = "e2b_mtp_enabled"
        const val UI_TIMEOUT_MS = 5_000L
    }
}
