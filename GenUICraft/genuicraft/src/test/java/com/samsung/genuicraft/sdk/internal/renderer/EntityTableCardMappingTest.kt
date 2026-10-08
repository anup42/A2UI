package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.entityCardRowContent
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.isCompactEntityBadgeCell
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.selectEntityHighlightIndexes
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.selectEntityCompactBodyIndexes
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EntityTableCardMappingTest {
    @Test fun `accepted phone comparison keeps all battery fields below display without changing price highlights`() {
        val headers = listOf("Model", "Indicative price", "Display", "Battery", "Software support", "Best for")
        val rows = listOf(
            listOf("OnePlus Nord CE 5", "From ₹24,999", "6.77-inch AMOLED, 120Hz [9][16]",
                "7,100mAh, 80W charging", "6 years of software support",
                "Buyers wanting a big battery and smooth everyday performance"),
            listOf("Samsung Galaxy A36 5G", "Around ₹28,999 online; official India pricing is higher [10][21][36]",
                "6.7-inch Super AMOLED, 120Hz [22][31]", "5,000mAh[38]",
                "6 OS upgrades and 6 years of security updates", "Buyers who want the longest support and a balanced all-rounder"),
            listOf("iQOO Z10", "From about ₹20,255–₹22,999 [37][40]", "6.77-inch AMOLED, 120Hz [29]",
                "7,300mAh, 90W charging [23]", "2 Android upgrades and 3 years of security patches",
                "Buyers prioritizing battery life and fast charging"),
        )
        val highlights = selectEntityHighlightIndexes(headers, rows, 0, emptySet())
        val compact = selectEntityCompactBodyIndexes(headers, rows, 0, highlights)

        assertEquals(listOf(1), highlights)
        assertTrue(compact.isEmpty())
        rows.forEachIndexed { index, row ->
            val content = entityCardRowContent(headers, row, index, 0, highlights, compact)
            assertTrue(content.compactBodyCells.isEmpty())
            assertEquals(listOf(2, 3, 4, 5), content.detailBodyCells.map { it.index })
            assertEquals(row[1], content.highlightCells.single().value)
            assertEquals(headers.zip(row), content.representedSourceCells.map { it.label to it.value })
        }
    }

    @Test fun `compact column eligibility ignores absent values but rejects a single long value and long detail labels`() {
        val headers = listOf("Model", "Tag", "Notes", "Absent")
        val rows = listOf<List<String?>>(
            listOf("A", "OK", "Short", null),
            listOf("B", "", null),
            listOf("C"),
        )
        assertEquals(setOf(1), selectEntityCompactBodyIndexes(headers, rows, 0, emptyList()))
        val mixed = rows + listOf(listOf("D", "A complete qualifier long enough to require full-width wrapping [7]", "Short"))
        assertTrue(selectEntityCompactBodyIndexes(headers, mixed, 0, emptyList()).isEmpty())
    }

    @Test fun `table-wide body policy preserves unknown metadata and authored actions`() {
        val headers = listOf("Model", "Tag", "Source note", "Website URL", "Website Action Label")
        val rows = listOf(
            listOf("A", "OK", "Unknown [2]", "https://www.who.int/a", "Read source"),
            listOf("B", "Long qualified body text remains complete [9]", "Not verified [5]", "https://www.who.int/b", "Read second source"),
        )
        val compact = selectEntityCompactBodyIndexes(headers, rows, 0, emptyList())
        assertTrue(compact.isEmpty())
        rows.forEachIndexed { index, row ->
            val content = entityCardRowContent(headers, row, index, 0, emptyList(), compact)
            assertEquals(headers.zip(row), content.representedSourceCells.map { it.label to it.value })
            assertEquals(listOf(row[3]), content.actions.map { it.safeUrl })
            assertEquals(listOf(row[4]), content.actions.map { it.buttonLabel })
            assertEquals(listOf("Website URL", "Website Action Label"), content.metadataCells.map { it.label })
        }
    }

    @Test fun `qualified phone price leads while long specifications stay labeled and complete`() {
        val headers = listOf("Model", "Indicative price", "Display", "Battery", "Software support", "Best for")
        val row = listOf(
            "Samsung Galaxy A36 5G",
            "Around ₹28,999 online; official India pricing is higher [10][21][36]",
            "6.7-inch Super AMOLED, 120Hz [22][31]",
            "5,000mAh[38]",
            "6 OS upgrades and 6 years of security updates",
            "Buyers who want the longest support and a balanced all-rounder"
        )
        val highlights = selectEntityHighlightIndexes(headers, listOf(row), 0, emptySet())
        val content = entityCardRowContent(headers, row, 0, 0, highlights)

        assertEquals(listOf(1), highlights)
        assertEquals(headers[1] to row[1], content.highlightCells.single().let { it.label to it.value })
        assertFalse(isCompactEntityBadgeCell(content.highlightCells.single()))
        assertEquals(listOf(headers[3] to row[3]), content.compactBodyCells.map { it.label to it.value })
        assertEquals(listOf(2, 4, 5).map { headers[it] to row[it] }, content.detailBodyCells.map { it.label to it.value })
        assertEquals(headers.zip(row), content.representedSourceCells.map { it.label to it.value })
    }

    @Test fun `long battery specification is a full width field`() {
        val headers = listOf("Model", "Indicative price", "Display", "Battery", "Software support", "Best for")
        val row = listOf(
            "OnePlus Nord CE 5",
            "From ₹24,999",
            "6.77-inch AMOLED, 120Hz [9][16]",
            "7,100mAh, 80W charging",
            "6 years of software support",
            "Buyers wanting a big battery and smooth everyday performance"
        )
        val content = entityCardRowContent(
            headers,
            row,
            0,
            0,
            selectEntityHighlightIndexes(headers, listOf(row), 0, emptySet())
        )
        assertEquals(row[1], content.highlightCells.single().value)
        assertTrue(isCompactEntityBadgeCell(content.highlightCells.single()))
        assertTrue(content.compactBodyCells.isEmpty())
        assertEquals(headers.drop(2).zip(row.drop(2)), content.detailBodyCells.map { it.label to it.value })
        assertEquals(headers.zip(row), content.representedSourceCells.map { it.label to it.value })
    }

    @Test fun `short earbud price is a compact highlight and the full tradeoff remains a detail`() {
        val headers = listOf("Model", "Price", "Features/Trade-offs")
        val row = listOf(
            "OnePlus Nord Buds 4 Pro",
            "₹3,999",
            "up to 55dB ANC, up to 54 hours total playback; main trade-off: connectivity and low-end clarity issues [6][2]"
        )
        val content = entityCardRowContent(
            headers,
            row,
            0,
            0,
            selectEntityHighlightIndexes(headers, listOf(row), 0, emptySet())
        )
        assertTrue(isCompactEntityBadgeCell(content.highlightCells.single()))
        assertEquals(headers[2] to row[2], content.detailBodyCells.single().let { it.label to it.value })
        assertEquals(headers.zip(row), content.representedSourceCells.map { it.label to it.value })
    }

    @Test fun `entity cards retain every field and every safe link without caps or truncation`() {
        val headers = buildList {
            add("Provider name")
            add("Rating")
            add("Price (USD)")
            addAll((1..8).map { index -> "Tag $index" })
            addAll((1..4).map { index -> "Detail $index" })
            add("Website URL")
            add("Website Action Label")
            add("Booking URL")
            add("Booking Action Label")
            add("Action URL")
            add("CTA Label")
        }
        val row = buildList {
            add("A provider name long enough to wrap across more than two compact title lines")
            add("4.8")
            add("\$199")
            addAll((1..8).map { index -> "Tag value $index" })
            addAll((1..4).map { index ->
                "Complete detail $index remains visible even after the former three-detail cap."
            })
            add("https://www.who.int/provider")
            add("Read the complete provider profile and terms")
            add("https://www.who.int/booking")
            add("Open booking options for every available date")
            add("javascript:alert(1)")
            add("Run script")
        }

        val content = entityCardRowContent(
            headers = headers,
            row = row,
            rowIndex = 0,
            primaryIndex = 0,
            highlightIndexes = listOf(1, 2)
        )

        assertEquals("Provider name", content.primaryLabel)
        assertEquals(row.first(), content.title)
        assertEquals(headers.zip(row), content.representedSourceCells.map { cell -> cell.label to cell.value })
        assertEquals(listOf("Rating", "Price (USD)"), content.highlightCells.map { it.label })
        assertEquals(8, content.compactBodyCells.size)
        assertEquals(4, content.detailBodyCells.size)
        assertEquals(
            listOf(
                "Website URL",
                "Website Action Label",
                "Booking URL",
                "Booking Action Label",
                "Action URL",
                "CTA Label"
            ),
            content.metadataCells.map { it.label }
        )
        assertEquals(
            listOf("https://www.who.int/provider", "https://www.who.int/booking"),
            content.actions.map { it.safeUrl }
        )
        assertEquals(
            listOf(
                "Read the complete provider profile and terms",
                "Open booking options for every available date"
            ),
            content.actions.map { it.buttonLabel }
        )
    }
}
