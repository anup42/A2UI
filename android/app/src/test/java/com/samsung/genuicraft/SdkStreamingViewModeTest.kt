package com.samsung.genuicraft

import org.junit.Assert.assertEquals
import org.junit.Test

class SdkStreamingViewModeTest {
    @Test
    fun missingOrUnknownPreferenceDefaultsToStreamingUi() {
        assertEquals(SdkStreamingViewMode.UI, SdkStreamingViewMode.fromPreference(null))
        assertEquals(SdkStreamingViewMode.UI, SdkStreamingViewMode.fromPreference("legacy"))
        assertEquals(SdkStreamingViewMode.TEXT, SdkStreamingViewMode.fromPreference("TEXT"))
    }
}
