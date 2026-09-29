package com.samsung.genuicraft.sdk

import androidx.compose.ui.platform.ComposeView
import androidx.core.widget.NestedScrollView
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class GenUiViewTest {
    @Test
    fun viewOwnsABoundedNativeScrollViewport() {
        val view = GenUiView(ApplicationProvider.getApplicationContext())

        assertEquals(1, view.childCount)
        val scrollView = view.getChildAt(0) as NestedScrollView
        assertTrue(scrollView.isFillViewport)
        assertTrue(scrollView.getChildAt(0) is ComposeView)
    }

    @Test
    fun embeddedModeAttachesComposeDirectlyWithoutNestedScrollView() {
        val view = GenUiView(ApplicationProvider.getApplicationContext())

        view.embeddedMode = true
        assertEquals(1, view.childCount)
        assertTrue(view.getChildAt(0) is ComposeView)

        view.embeddedMode = false
        assertEquals(1, view.childCount)
        val scrollView = view.getChildAt(0) as NestedScrollView
        assertTrue(scrollView.getChildAt(0) is ComposeView)
    }
}
