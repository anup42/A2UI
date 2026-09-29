package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.entityCardRowContent
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.isCompactEntityBadgeCell
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.selectEntityHighlightIndexes
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EntityTableCardMappingTest {
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
