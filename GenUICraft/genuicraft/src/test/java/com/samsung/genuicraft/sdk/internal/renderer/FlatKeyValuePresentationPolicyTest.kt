package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatKeyValuePresentationPolicyTest {
    @Test fun `long instructions use a coherent full width panel without modifying cells`() {
        val rows = listOf(
            listOf("Authorization", "Confirm that the recycler or pickup service is CPCB/KSPCB-authorized or otherwise clearly certified.[1]"),
            listOf("Battery acceptance", "Verify laptop batteries are accepted.[2]"),
        )
        val snapshot = rows.map { it.toList() }
        assertTrue(shouldStackKeyValueRows(rows))
        assertEquals(snapshot, rows)
    }

    @Test fun `short observation metrics keep the compact aligned layout`() {
        assertFalse(shouldStackKeyValueRows(listOf(listOf("AQI-IN", "197"), listOf("Category", "Poor"), listOf("Observed", "8:30 AM IST"))))
        assertFalse(shouldStackKeyValueRows(emptyList()))
    }

    @Test fun `long labels and multiline supplied text get full reading width`() {
        assertTrue(shouldStackKeyValueRows(listOf(listOf("Keep the battery separate from regular waste", "Do not put it in household garbage."))))
        assertTrue(shouldStackKeyValueRows(listOf(listOf("Line one\nLine two", "Unchanged value"))))
        assertTrue(shouldStackKeyValueRows(listOf(listOf("Original label", "Line one\nLine two [9]"))))
    }
}
