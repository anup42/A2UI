package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.shouldRenderProseHeadingAsBody
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatTextPresentationTest {
    @Test fun `long explanatory sentence does not fill the first screen as a heading`() {
        val generatedTrainIntro =
            "Here is a concise comparison of three daytime Bengaluru-to-Mysuru train services, using the most consistent timetable entries across the supplied sources."
        assertTrue(shouldRenderProseHeadingAsBody(generatedTrainIntro, "h2"))
        assertFalse(shouldRenderProseHeadingAsBody(generatedTrainIntro, "body"))
    }

    @Test fun `short headings and descriptive titles retain heading presentation`() {
        assertFalse(shouldRenderProseHeadingAsBody("Three phones under ₹30,000", "h2"))
        assertFalse(shouldRenderProseHeadingAsBody("Closest verified options", "h2"))
        assertFalse(
            shouldRenderProseHeadingAsBody(
                "A comprehensive comparison of mobile devices for travel and long term use",
                "h1"
            )
        )
    }
}
