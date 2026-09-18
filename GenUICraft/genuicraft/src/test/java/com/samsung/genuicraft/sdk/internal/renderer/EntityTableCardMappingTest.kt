package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.entityCardRowContent
import org.junit.Assert.assertEquals
import org.junit.Test

class EntityTableCardMappingTest {
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
