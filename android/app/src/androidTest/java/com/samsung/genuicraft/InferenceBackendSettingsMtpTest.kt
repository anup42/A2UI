package com.samsung.genuicraft

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class InferenceBackendSettingsMtpTest {
    @Test
    fun mtpDefaultsOnAndSharesTheSdkDemoPreference() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext
        val sdkPreferences = context.getSharedPreferences(SDK_DEMO_PREFS_NAME, Context.MODE_PRIVATE)
        val originallyPresent = sdkPreferences.contains(KEY_E2B_MTP_ENABLED)
        val originalValue = sdkPreferences.getBoolean(KEY_E2B_MTP_ENABLED, true)

        try {
            check(sdkPreferences.edit().remove(KEY_E2B_MTP_ENABLED).commit())
            assertTrue(InferenceBackendSettings.getOnDeviceMtpEnabled(context))

            InferenceBackendSettings.setOnDeviceMtpEnabled(context, false)
            assertFalse(sdkPreferences.getBoolean(KEY_E2B_MTP_ENABLED, true))
            assertFalse(InferenceBackendSettings.getOnDeviceMtpEnabled(context))

            check(sdkPreferences.edit().putBoolean(KEY_E2B_MTP_ENABLED, true).commit())
            assertTrue(InferenceBackendSettings.getOnDeviceMtpEnabled(context))
        } finally {
            val editor = sdkPreferences.edit()
            if (originallyPresent) {
                editor.putBoolean(KEY_E2B_MTP_ENABLED, originalValue)
            } else {
                editor.remove(KEY_E2B_MTP_ENABLED)
            }
            check(editor.commit())
        }
    }

    private companion object {
        const val SDK_DEMO_PREFS_NAME = "genuicraft_sdk_demo"
        const val KEY_E2B_MTP_ENABLED = "e2b_mtp_enabled"
    }
}
