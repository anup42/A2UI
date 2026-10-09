package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import org.junit.Assert.*
import org.junit.Test

class FlatSparseProseTablePolicyTest {
    // Exact two native 046 before tables; final JSON SHA-256 486f24dd6459c479546cfc56dfc08f2e85916ad50582a013ea29e05817c5c55a.
    private val mechanismRows = listOf(
        listOf("Gene therapy treats disease by delivering a therapeutic gene, or by replacing, supplementing, or silencing a faulty gene; it does not necessarily cut the genome at a specific DNA sequence.", "[2][14]", ""),
        listOf("", "[2][14]", "CRISPR gene editing uses a guide RNA to bring a Cas nuclease to a chosen DNA site, where it creates a double-strand break; repair by NHEJ often disrupts a gene, while HDR can make a precise sequence change."),
    )
    private val deliveryRows = listOf(
        listOf("Gene therapy commonly uses viral vectors, especially adeno-associated virus and lentiviral systems, because they efficiently carry therapeutic DNA into cells.", "", "", ""),
        listOf("", "", "This delivery approach can cause immunogenic toxicity and, with integrating vectors, insertional oncogenesis.", ""),
        listOf("", "", "", "CRISPR can be delivered by viral vectors, lipid nanoparticles, or by direct delivery of Cas protein and guide RNA as a ribonucleoprotein complex; the core challenge is getting the editing machinery into the right cells long enough to act."),
    )
    private val headers = listOf("part1", "source", "part2")
    private fun table(columns: List<String> = headers, rows: List<*> = mechanismRows, title: String = "Mechanism details") =
        FlatElement("Table", mapOf("columns" to columns, "rows" to rows, "title" to title,
            "domain" to "generic", "preferredPresentation" to "cards"), emptyList())

    @Test fun `actual two sparse 046 tables preserve every paragraph part label marker and row position`() {
        val mechanism = table()
        val delivery = table(listOf("part1", "source", "part2", "part3"), deliveryRows, "Delivery details")
        val first = requireNotNull(sparseProseTableProfile(mechanism))
        val second = requireNotNull(sparseProseTableProfile(delivery))
        assertEquals(headers, first.headers)
        assertEquals(mechanismRows, first.originalRows)
        assertEquals("Mechanism details", first.title)
        assertEquals(listOf("part1", "part2"), first.paragraphs.map { it.partLabel })
        assertEquals(listOf(mechanismRows[0][0], mechanismRows[1][2]), first.paragraphs.map { it.body })
        assertEquals(listOf("[2][14]", "[2][14]"), first.paragraphs.map { it.citation })
        assertEquals(deliveryRows, second.originalRows)
        assertEquals("Delivery details", second.title)
        assertEquals(listOf("part1", "part2", "part3"), second.paragraphs.map { it.partLabel })
        assertEquals(listOf(deliveryRows[0][0], deliveryRows[1][2], deliveryRows[2][3]), second.paragraphs.map { it.body })
        assertTrue(second.paragraphs.all { it.citation.isEmpty() })
        assertEquals(5, first.paragraphs.size + second.paragraphs.size)
        assertEquals(mechanismRows, mechanism.props["rows"])
        assertEquals(deliveryRows, delivery.props["rows"])
    }

    @Test fun `meaningful matrix multiple body fields arbitrary source and noncard presentation remain unchanged`() {
        val original = table()
        val multiple = mechanismRows.toMutableList().apply { this[0] = listOf(mechanismRows[0][0], "[2]", mechanismRows[1][2]) }
        val proseSource = mechanismRows.toMutableList().apply { this[0] = listOf(mechanismRows[0][0], "Source attribution prose", "") }
        val cases = listOf(
            table(listOf("Factor", "CBSE", "IB")), table(listOf("part2", "source", "part1")),
            table(listOf("part1", "source", "part3")), table(rows = multiple), table(rows = proseSource),
            original.copy(props = original.props + ("domain" to "comparison")),
            original.copy(props = original.props + ("preferredPresentation" to "table")),
            original.copy(props = original.props + ("subtitle" to "Preserve original subtitle")),
            table(rows = listOf(listOf("", "[1]", ""), mechanismRows[1])),
            table(rows = listOf(listOf("Short literal", "", ""), mechanismRows[1])),
        )
        cases.forEach { node -> assertNull(sparseProseTableProfile(node)) }
    }

    @Test fun `guarded bound repeated interactive and unknown parent or table data do not become paragraphs`() {
        val original = table()
        val guarded = listOf(
            original.copy(visible = false), original.copy(repeat = RepeatConfig("/rows")),
            original.copy(on = mapOf("click" to "originalAction")), original.copy(watch = mapOf("/rows" to "originalWatch")),
            original.copy(props = original.props + ("statePath" to "/rows")),
            original.copy(props = original.props + ("unknown" to "Keep original metadata")),
            original.copy(children = listOf("Original child")),
            table(rows = listOf(mapOf("part1" to mechanismRows[0][0]), mapOf("part2" to mechanismRows[1][2]))),
        )
        guarded.forEach { node -> assertNull(sparseProseTableProfile(node)) }
        guarded.take(4).forEach { node -> assertNull(sparseProseTableProfile(original, listOf(node.copy(type = "Card")))) }
        assertNull(sparseProseTableProfile(original, listOf(null)))
        assertNull(sparseProseTableProfile(original, isRepeated = true))
        val boundParent = FlatElement("Card", mapOf("title" to mapOf("\$computed" to "fn")), emptyList())
        assertNull(sparseProseTableProfile(original, listOf(boundParent)))
        val dynamic = mechanismRows.toMutableList().apply { this[0] = listOf("{{/body}}", "[2]", "") }
        assertNull(sparseProseTableProfile(table(rows = dynamic)))
    }

    @Test fun `full multiline text whitespace citation groups and existing title remain exact`() {
        val body = "  " + mechanismRows[0][0] + "\nSecond original paragraph, unchanged.  "
        val citation = " [2][14] "
        val rows = listOf(listOf(body, citation, ""), mechanismRows[1])
        val profile = requireNotNull(sparseProseTableProfile(table(rows = rows, title = "Exact existing title")))
        assertEquals(body, profile.paragraphs.first().body)
        assertEquals(citation, profile.paragraphs.first().citation)
        assertEquals("Exact existing title", profile.title)
        val ranged = rows.toMutableList().apply { this[0] = listOf(body, "[1,3][5–7]", "") }
        assertEquals("[1,3][5–7]", requireNotNull(sparseProseTableProfile(table(rows = ranged))).paragraphs.first().citation)
    }

    @Test fun `repeated part rows stay repeated without inferred entity names or reordered sections`() {
        val rows = listOf(mechanismRows[0], mechanismRows[0], mechanismRows[1])
        val profile = requireNotNull(sparseProseTableProfile(table(rows = rows)))
        assertEquals(rows, profile.originalRows)
        assertEquals(listOf("part1", "part1", "part2"), profile.paragraphs.map { it.partLabel })
        assertEquals(listOf(mechanismRows[0][0], mechanismRows[0][0], mechanismRows[1][2]), profile.paragraphs.map { it.body })
        assertEquals(listOf("[2][14]", "[2][14]", "[2][14]"), profile.paragraphs.map { it.citation })
    }
}
