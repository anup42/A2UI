package com.samsung.genuicraft.sdk.internal.renderer

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.internal.pipeline.GenUiIrCodec
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.FlatSpecParser
import org.junit.Assert.assertNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FlatKeyValuePresentationPolicyTest {
    private val evidenceHeaders = listOf("Item", "Evidence")
    private fun literalTable(rows: List<*>): FlatElement = FlatElement("Table", mapOf(
        "columns" to evidenceHeaders, "rows" to rows,
        "domain" to "generic", "preferredPresentation" to "cards",
    ), emptyList())

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

    @Test fun `actual 040 repaired evidence retains ten rows twenty cells and the unequal final qualifier`() {
        val raw = requireNotNull(javaClass.getResourceAsStream("/recovery/BXP-040.fp16.raw.express"))
            .bufferedReader(Charsets.UTF_8).use { it.readText() }
        val outcome = GenUiCompiler.compileWithRepair(raw, allowSourceTextFallback = false, allowGeneratedDslRepair = true)
        val graph = GenUiIrCodec.decode(JsonParser.parseString(outcome.document.a2uiJson)).canonicalGraph
        val parsed = requireNotNull(FlatSpecParser.parse(graph))
        val evidence = parsed.elements.values.single { it.type == "Table" && it.props["columns"] == evidenceHeaders }
        val before = evidence.copy(props = evidence.props.toMap())
        val model = requireNotNull(extractDirectTableModel(evidence.props, parsed.state, compactScreen = true))
        assertEquals(FlatTableShape.KEY_VALUE, model.shape)
        assertEquals(evidenceHeaders, model.columns.map { it.label })
        val eligible = identicalPassiveKeyValueRowIndexes(evidence, listOf(parsed.elements[parsed.root]))
        assertEquals((0..8).toSet(), eligible)
        val shown = model.rows.mapIndexed { index, row ->
            singleIdenticalKeyValueRowText(evidenceHeaders, row, index, eligible)
        }
        assertEquals(model.rows.take(9).map { it[0] }, shown.take(9))
        assertNull(shown.last())
        assertEquals(listOf("job sheet or written service report", "if inspected or repaired"), model.rows.last())
        assertEquals(10, model.rows.size)
        assertEquals(20, model.rows.sumOf { it.size })
        assertEquals(before, evidence)
    }

    @Test fun `original cell equality never normalizes case whitespace citations markup or Unicode`() {
        val rows = listOf(
            listOf("Invoice", "Invoice"), listOf("Invoice", "invoice"), listOf("Invoice", " Invoice"),
            listOf("Invoice", "Invoice "), listOf("Fact [1]", "Fact [2]"), listOf("**Fact**", "Fact"),
            listOf("13", "13.0"), listOf("café", "café"), listOf("", ""),
            listOf("Only if inspected [7]", "Only if inspected [7]"),
        )
        val table = literalTable(rows)
        val snapshot = rows.map { it.toList() }
        val eligible = identicalPassiveKeyValueRowIndexes(table)
        assertEquals(setOf(0, 9), eligible)
        assertEquals("Only if inspected [7]", singleIdenticalKeyValueRowText(evidenceHeaders, rows[9], 9, eligible))
        assertNull(singleIdenticalKeyValueRowText(evidenceHeaders, rows[0], 0, emptySet()))
        assertEquals(snapshot, rows)
        // Existing display formatting may trim; an unequal original row is still ineligible.
        assertNull(singleIdenticalKeyValueRowText(evidenceHeaders, listOf("Invoice", "Invoice"), 2, eligible))
    }

    @Test fun `repeated identical source rows remain separate and a changed value is never hidden`() {
        val rows = listOf(listOf("Same evidence", "Same evidence"), listOf("Same evidence", "Same evidence"))
        val table = literalTable(rows)
        val eligible = identicalPassiveKeyValueRowIndexes(table)
        assertEquals(setOf(0, 1), eligible)
        val shown = rows.mapIndexed { index, row -> singleIdenticalKeyValueRowText(evidenceHeaders, row, index, eligible) }
        assertEquals(listOf("Same evidence", "Same evidence"), shown)
        assertEquals(2, (table.props["rows"] as List<*>).size)
        assertNull(singleIdenticalKeyValueRowText(evidenceHeaders, listOf("Same evidence", "Only if repaired"), 0, eligible))
        assertNull(singleIdenticalKeyValueRowText(evidenceHeaders, rows[0] + "Extra authored qualifier", 0, eligible))
        assertNull(singleIdenticalKeyValueRowText(evidenceHeaders + "Qualifier", rows[0], 0, eligible))
    }

    @Test fun `state bindings that happen to resolve equally remain on the unchanged two cell route`() {
        val bound = literalTable(listOf(listOf(mapOf("\$state" to "/left"), mapOf("\$state" to "/right"))))
        val model = requireNotNull(extractDirectTableModel(bound.props,
            mapOf("left" to "Same evidence", "right" to "Same evidence"), compactScreen = true))
        assertEquals(listOf(listOf("Same evidence", "Same evidence")), model.rows)
        assertTrue(identicalPassiveKeyValueRowIndexes(bound).isEmpty())
        listOf("{{/left}}", "\$/left", "\$item.name", "\${left}", "@source[1]").forEach { binding ->
            assertTrue(identicalPassiveKeyValueRowIndexes(literalTable(listOf(listOf(binding, binding)))).isEmpty())
        }
        val direct = literalTable(listOf(listOf("Same", "Same")))
        assertTrue(identicalPassiveKeyValueRowIndexes(direct.copy(props = direct.props + ("statePath" to "/rows"))).isEmpty())
        listOf("\$state", "\$bindState", "\$item", "\$bindItem", "\$index", "\$cond", "\$template", "\$computed").forEach { key ->
            val ancestor = FlatElement("Card", mapOf("title" to mapOf("nested" to listOf(mapOf(key to "field")))), emptyList())
            assertTrue(identicalPassiveKeyValueRowIndexes(direct, listOf(ancestor)).isEmpty())
        }
    }

    @Test fun `actions guards repeats unknown context and child structures prevent duplicate collapse`() {
        val original = literalTable(listOf(listOf("Same", "Same")))
        val guarded = listOf(
            original.copy(on = mapOf("click" to "originalAction")), original.copy(watch = mapOf("/rows" to "originalWatch")),
            original.copy(visible = false), original.copy(visible = mapOf("\$state" to "/show")),
            original.copy(repeat = RepeatConfig("/rows")), original.copy(children = listOf("originalChild")),
            original.copy(props = original.props + ("unknown" to "Keep original metadata")),
        )
        guarded.forEach { node -> assertTrue(identicalPassiveKeyValueRowIndexes(node).isEmpty()) }
        guarded.take(5).forEach { node ->
            assertTrue(identicalPassiveKeyValueRowIndexes(original, listOf(node.copy(type = "Card"))).isEmpty())
        }
        assertTrue(identicalPassiveKeyValueRowIndexes(original, listOf(null)).isEmpty())
        assertTrue(identicalPassiveKeyValueRowIndexes(original, isRepeated = true).isEmpty())
        assertEquals(listOf(listOf("Same", "Same")), original.props["rows"])
    }

    @Test fun `URLs descriptor maps nonstrings and extra fields retain their original presentation`() {
        listOf("https://example.com", "Read www.example.com", "tel:123", "mailto:help@example.com").forEach { value ->
            assertTrue(identicalPassiveKeyValueRowIndexes(literalTable(listOf(listOf(value, value)))).isEmpty())
        }
        listOf(listOf(listOf("Same", "Same", "Keep third field")), listOf(listOf(13, 13)),
            listOf(mapOf("item" to "Same", "evidence" to "Same"))).forEach { rows ->
            assertTrue(identicalPassiveKeyValueRowIndexes(literalTable(rows)).isEmpty())
        }
        val original = literalTable(listOf(listOf("Same", "Same")))
        val mappedColumns = listOf(mapOf("key" to "item", "label" to "Item"), mapOf("key" to "evidence", "label" to "Evidence"))
        assertTrue(identicalPassiveKeyValueRowIndexes(original.copy(props = original.props + ("columns" to mappedColumns))).isEmpty())
        assertTrue(identicalPassiveKeyValueRowIndexes(original.copy(props = original.props + ("columns" to listOf("Only field")))).isEmpty())
    }
}
