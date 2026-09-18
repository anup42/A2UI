package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.*
import org.junit.Test

class TableViewportUsabilityTest {
    @Test fun `frozen label and each full value fit the reduced host viewport`() {
        val original = listOf(184, 232, 208, 232)
        val widths = tableViewportColumnWidthsDp(original, 320, stickyFirstColumn = true)
        assertEquals(listOf(108, 212, 208, 212), widths)
        assertTrue(widths.first() <= 320 * 0.34f)
        widths.drop(1).forEach { assertTrue(widths.first() + it <= 320) }
        assertEquals(listOf(184, 232, 208, 232), original)
    }

    @Test fun `actual narrow and wide host widths are honored independently of device size`() {
        val original = listOf(164, 232, 232)
        listOf(180, 240, 288, 320, 412, 800).forEach { viewport ->
            val widths = tableViewportColumnWidthsDp(original, viewport, true)
            assertEquals(original.size, widths.size)
            assertTrue(widths.all { it > 0 })
            widths.drop(1).forEach { assertTrue(widths.first() + it <= viewport) }
        }
        assertEquals(listOf(144, 232, 232), tableViewportColumnWidthsDp(original, 800, true))
        assertEquals(listOf(164, 180, 180), tableViewportColumnWidthsDp(original, 180, false))
        assertTrue(tableViewportColumnWidthsDp(emptyList(), 320, true).isEmpty())
    }

    @Test fun `three-column identities stay attached during horizontal reading`() {
        listOf(
            listOf("Term", "What it means", "Typical effect on a claim"),
            listOf("Level", "Best measures", "Expected impact")
        ).forEach { headers ->
            assertTrue(nativeTableStickyFirstColumn(headers, true))
            assertFalse(nativeTableStickyFirstColumn(headers, false))
        }
        assertFalse(nativeTableStickyFirstColumn(listOf("Latitude", "Longitude", "Altitude"), true))
        assertFalse(nativeTableStickyFirstColumn(listOf("Term", "Definition"), true))
        assertTrue(nativeTableStickyFirstColumn(listOf("Feature", "EPF", "PPF", "NPS"), true))
    }

    @Test fun `numeric alignment retains scalar currency ranges signs and units`() {
        listOf("31", "−12.5 °C", "57%", "₹1–200", "USD 1,200", "90–160 Wh/kg", "**12%**[3]", "2 h", "3 hours")
            .forEach { assertTrue(it, isScalarTableGridValue(it)) }
        listOf("Lower; packs were almost 30% cheaper", "Higher; generally best", "Commercial ranges around 90–160 Wh/kg",
            "2025 data", "₹1–200 per person, closed Friday", "https://example.org/12", "", "N/A")
            .forEach { assertFalse(it, isScalarTableGridValue(it)) }
    }

    @Test fun `cost prose is left aligned without changing numeric heuristics used by cards`() {
        val rows = listOf(listOf("LFP", "Lower; 30% cheaper in 2025", "₹1–200"),
            listOf("NMC", "Higher; materially more expensive", "₹200–300"))
        assertEquals(setOf(2), tableGridNumericColumns(rows, setOf(1, 2)))
        assertEquals("Lower; 30% cheaper in 2025", rows[0][1])
        assertEquals(emptySet<Int>(), tableGridNumericColumns(rows, emptySet()))
    }

    @Test fun `snap starts use exact pixel widths and include a clamped final column`() {
        assertEquals(listOf(0, 531, 1053), tableColumnSnapOffsetsPx(listOf(531, 522, 531), 1053))
        // The last column can be narrower than the viewport, so its natural start exceeds max.
        assertEquals(listOf(0, 400, 750), tableColumnSnapOffsetsPx(listOf(400, 450, 300), 750))
        assertEquals(listOf(0, 400, 700), tableColumnSnapOffsetsPx(listOf(400, 450, 300, 200), 700))
        assertEquals(listOf(0), tableColumnSnapOffsetsPx(listOf(400), 0))
        assertEquals(listOf(0), tableColumnSnapOffsetsPx(emptyList(), 0))
    }

    @Test fun `idle fragments settle to the nearest column and settle only once`() {
        val boundaries = tableColumnSnapOffsetsPx(listOf(531, 522, 531), 1053)
        val targets = mapOf(-50 to 0, 0 to 0, 190 to 0, 340 to 531, 531 to 531,
            780 to 531, 900 to 1053, 1053 to 1053, 1300 to 1053)
        targets.forEach { (offset, expected) ->
            val snapped = nearestTableColumnSnapOffsetPx(offset, boundaries)
            assertEquals(expected, snapped)
            assertEquals(snapped, nearestTableColumnSnapOffsetPx(snapped, boundaries))
        }
        assertEquals(0, nearestTableColumnSnapOffsetPx(200, listOf(0, 400)))
        assertEquals(0, nearestTableColumnSnapOffsetPx(12, emptyList()))
    }

    @Test fun `snap boundaries remain bounded without integer overflow`() {
        assertEquals(listOf(0, 800), tableColumnSnapOffsetsPx(listOf(Int.MAX_VALUE, Int.MAX_VALUE, 100), 800))
        assertEquals(0, nearestTableColumnSnapOffsetPx(Int.MIN_VALUE, listOf(0, Int.MAX_VALUE)))
        assertEquals(Int.MAX_VALUE, nearestTableColumnSnapOffsetPx(Int.MAX_VALUE, listOf(0, Int.MAX_VALUE)))
    }
}
