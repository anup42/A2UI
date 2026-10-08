package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.entityCardRowContent
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.sightseeingTableProfile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatSightseeingPolicyTest {
    @Test fun `sightseeing cards retain names hours duration unknown fees and citations`() {
        val headers = listOf("Sight", "Opening hours", "Usual visit duration", "Published adult entry fee", "Notes")
        val row = listOf("Mysuru Railway Museum", "10:00 am–6:00 pm; closed Tuesday", "1.5–2 hours",
            "Not available in the provided reliable sources[2]", "Every original qualification remains [9].")
        val profile = sightseeingTableProfile(headers, listOf(row))!!
        val content = entityCardRowContent(headers, row, 0, profile.primaryIndex, emptyList())

        assertEquals(0, profile.primaryIndex)
        assertEquals(listOf(1, 2, 3), profile.detailIndexes)
        assertEquals("Mysuru Railway Museum", content.title)
        assertEquals(row, content.representedSourceCells.map { it.value })
        assertEquals(headers, content.representedSourceCells.map { it.label })
    }

    @Test fun `feature matrices process rows and weak identity do not select place cards`() {
        assertNull(sightseeingTableProfile(listOf("Feature", "Place", "Hours", "Duration"),
            listOf(listOf("Opening hours", "A", "B", "C"))))
        assertNull(sightseeingTableProfile(listOf("Step", "Place", "Hours", "Duration"),
            listOf(listOf("1", "Counter", "9–5", "10 min"))))
        assertNull(sightseeingTableProfile(listOf("Name", "Hours", "Duration"),
            listOf(listOf("A", "9–5", "10 min"))))
        assertNull(sightseeingTableProfile(listOf("Sight", "Hours"), listOf(listOf("Museum", "9–5"))))
        assertNull(sightseeingTableProfile(listOf("Sight", "Hours", "Duration"), listOf(listOf("", "9–5", "10 min"))))
    }

    @Test fun `source row order duplicates and original cell strings remain authoritative`() {
        val headers = listOf("Attraction name", "Opening hours", "Visit duration", "Entry fee")
        val row = listOf("Park [1]", "Open 24 hours", "About 20–30 min", "Fee not verified [6]")
        val rows = listOf(row, row)
        val profile = sightseeingTableProfile(headers, rows)!!
        rows.forEachIndexed { index, values ->
            assertEquals(values, entityCardRowContent(headers, values, index, profile.primaryIndex, emptyList()).representedSourceCells.map { it.value })
        }
        assertEquals(listOf(row, row), rows)
    }
}
