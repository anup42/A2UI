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
    private fun snapshot(attempt: Int = 1, revision: Long, final: Boolean = false, key: String = "answer") =
        GenUiRenderSnapshot(key, attempt, revision, GenUiDocument("", "{}"), 0L, 1, final)

    @Test
    fun finalSnapshotCannotBeOverwrittenByQueuedPreviewOrOlderAttempt() {
        val view = GenUiView(ApplicationProvider.getApplicationContext())
        val nativeViewport = view.getChildAt(0)
        view.renderSnapshot(snapshot(revision = 1L))
        view.renderSnapshot(snapshot(revision = 2L))
        val final = snapshot(revision = 2L, final = true)
        view.renderSnapshot(final)
        view.renderSnapshot(snapshot(revision = 3L))
        view.renderSnapshot(snapshot(attempt = 2, revision = 1L))
        view.renderSnapshot(snapshot(attempt = 0, revision = 99L, final = true))

        assertEquals(final, view.currentRenderSnapshot)
        assertTrue(nativeViewport === view.getChildAt(0))
    }

    @Test
    fun retriesAndNewSurfacesAcceptTheirOwnRevisionSequence() {
        val view = GenUiView(ApplicationProvider.getApplicationContext())
        view.renderSnapshot(snapshot(revision = 10L))
        val retry = snapshot(attempt = 2, revision = 1L)
        view.renderSnapshot(retry)
        assertEquals(retry, view.currentRenderSnapshot)
        val next = snapshot(revision = 1L, key = "next answer")
        view.renderSnapshot(next)
        assertEquals(next, view.currentRenderSnapshot)

        view.clear()
        assertEquals(null, view.currentRenderSnapshot)
        view.renderSnapshot(snapshot(revision = 1L))
        view.render(GenUiDocument("", "{}"))
        assertEquals(null, view.currentRenderSnapshot)
    }

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
