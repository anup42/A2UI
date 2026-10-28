package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.renderer.AdaptiveTablePresentation
import com.samsung.genuicraft.sdk.internal.renderer.FlatTableRenderMode
import com.samsung.genuicraft.sdk.internal.renderer.FlatTableShape
import com.samsung.genuicraft.sdk.internal.renderer.selectAdaptiveTablePresentation
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class EntityListPresentationTest {
    @Test fun capturedFlightListBecomesCardsWhileFareNotesRemainLiteralList() {
        val source = captured("LIVE-FLIGHT")
        val snapshot = source.deepCopy()
        val result = EntityListPresentation.normalize(source)
        assertEquals(1, result.convertedLists)
        assertEquals(snapshot, source)
        assertEquals(snapshot.get("root"), result.graph.get("root"))
        assertEquals(snapshot.get("state"), result.graph.get("state"))
        val before = snapshot.getAsJsonObject("elements")
        val after = result.graph.getAsJsonObject("elements")
        assertEquals(before.keySet(), after.keySet())
        before.entrySet().filter { it.key != "b" }.forEach { (id, value) ->
            assertEquals("$id must remain byte-equivalent in the graph", value, after.get(id))
        }
        assertEquals("List", after.getAsJsonObject("d").get("type").asString)
        assertCardTable(after.getAsJsonObject("b"), listOf(
            "Akasa Air" to "**₹7,195** one-way, non-stop, departing **12:25** from Bengaluru and arriving **15:05** in Lucknow.",
            "Air India Express" to "**₹7,298** one-way, non-stop, departing **17:05** from Bengaluru and arriving **19:50** in Lucknow.",
            "IndiGo" to "**₹15,599** for 15 Oct as listed in the fare calendar.",
            "Air India Express" to "another same-day fare listing shows **from ₹22,268** for **Thu, Oct 15 2026**.",
            "ixigo" to "lists **₹12,882** for **Thu, 15 Oct**, with multiple Air India Express options on that date, including a non-stop **05:55 → 08:35** flight priced at **₹10,898** and other one-stop options from **₹14,010** and above.",
        ))
        assertExactSuffixes(before.getAsJsonObject("b"), after.getAsJsonObject("b"))
        assertTrue(A2uiCanonicalGraph.validate(result.graph).isValid)
        assertEquals(result.graph, A2uiExpressCodec.decode(A2uiExpressCodec.encode(result.graph)))
    }

    @Test fun capturedProductsKeepTradeoffsQualifiersAndCitationsInOrder() {
        val source = captured("BXP-010")
        val snapshot = source.deepCopy()
        val result = EntityListPresentation.normalize(source)
        assertEquals(1, result.convertedLists)
        assertEquals(snapshot, source)
        val before = snapshot.getAsJsonObject("elements")
        val after = result.graph.getAsJsonObject("elements")
        before.entrySet().filter { it.key != "d" }.forEach { (id, value) -> assertEquals(value, after.get(id)) }
        assertCardTable(after.getAsJsonObject("d"), listOf(
            "OnePlus Nord Buds 4 Pro" to "₹3,999, up to **55dB ANC**, up to **54 hours** total playback; main trade-off: strong feature set, but the review notes some connectivity and low-end clarity issues [6][2].",
            "Realme Buds Air 8" to "₹3,799, up to **55dB ANC**, up to **58 hours** total playback; main trade-off: feature-rich, but users may need to accept a more compromised tuning/controls experience.",
            "Oppo Enco Buds 3 Pro+" to "₹2,999, up to **32dB ANC**, up to **43 hours** total playback; main trade-off: cheaper and comfortable, but ANC is weaker than the two above [7].",
        ))
        assertExactSuffixes(before.getAsJsonObject("d"), after.getAsJsonObject("d"))
        assertTrue(A2uiCanonicalGraph.validate(result.graph).isValid)
        assertEquals(result.graph, A2uiExpressCodec.decode(A2uiExpressCodec.encode(result.graph)))
    }

    @Test fun plainProperNamesNeedRepeatedLeadingFinancialOrTimeAttributes() {
        val source = graphWithItems(strings(
            "Akasa Air: ₹7,195 one-way, 12:25 to 15:05 [1].",
            "Air India Express: ₹7,298 one-way, 17:05 to 19:50 [2].",
        ))
        val result = EntityListPresentation.normalize(source)
        assertEquals(1, result.convertedLists)
        assertCardTable(result.graph.getAsJsonObject("elements").getAsJsonObject("listing"), listOf(
            "Akasa Air" to "₹7,195 one-way, 12:25 to 15:05 [1].",
            "Air India Express" to "₹7,298 one-way, 17:05 to 19:50 [2].",
        ))

        val schedule = graphWithItems(strings(
            "Morning Express: 08:30 departure from Bengaluru.",
            "Evening Express: 18:30 departure from Bengaluru.",
        ))
        assertEquals(1, EntityListPresentation.normalize(schedule).convertedLists)
    }

    @Test fun ordinaryAdviceWithMoneyAndDatesDoesNotBecomeEntityCards() {
        val variants = listOf(
            strings("Book early: ₹500 may be saved by Friday 15 Oct.", "Check baggage: ₹700 can apply on Friday 15 Oct."),
            strings("**Book early**: ₹500 may be saved by Friday 15 Oct.", "**Compare fares**: ₹700 can apply on Friday 15 Oct."),
            strings("Prices are dynamic and may be ₹7,195 on 15 Oct.", "Some sources quote ₹7,298 for 15 Oct."),
            strings("**Akasa Air**: the fare may change on Friday to ₹7,195.", "**Air India Express**: the fare may change on Friday to ₹7,298."),
        )
        variants.forEach { items ->
            val graph = graphWithItems(items)
            val normalized = EntityListPresentation.normalize(graph)
            assertSame(graph, normalized.graph)
            assertEquals(0, normalized.convertedLists)
        }
    }

    @Test fun mixedRowsBindingsRichChildrenAndUnknownMetadataAreLeftAlone() {
        val literal = strings("**Akasa Air**: ₹7,195 on 15 Oct.", "**IndiGo**: ₹15,599 on 15 Oct.")
        val itemMap = JsonObject().apply { addProperty("text", "**IndiGo**: ₹15,599 on 15 Oct.") }
        val bound = JsonObject().apply { addProperty("path", "/items") }
        val mixed = JsonArray().apply { add(literal[0].deepCopy()); add(itemMap) }
        val paths = JsonArray().apply { add(bound); add(bound.deepCopy()) }
        val protected = strings(
            LiteralTextCodec.encode("**Akasa Air**: ₹7,195 ${'$'}{fare} on 15 Oct."),
            "**IndiGo**: ₹15,599 on 15 Oct.",
        )
        val expression = strings("**Akasa Air**: ₹7,195 ${'$'}/fare on 15 Oct.", "**IndiGo**: ₹15,599 on 15 Oct.")
        val itemBinding = strings("**Akasa Air**: ₹7,195 ${'$'}item.fare on 15 Oct.", "**IndiGo**: ₹15,599 on 15 Oct.")
        val indexBinding = strings("**Akasa Air**: ₹7,195 ${'$'}index on 15 Oct.", "**IndiGo**: ₹15,599 on 15 Oct.")
        listOf(mixed, paths, protected, expression, itemBinding, indexBinding).forEach { items ->
            val graph = graphWithItems(items)
            assertSame(graph, EntityListPresentation.normalize(graph).graph)
        }
        val withChildren = graphWithItems(literal).apply {
            getAsJsonObject("elements").getAsJsonObject("listing").getAsJsonArray("children").add("rich")
        }
        val withStatePath = graphWithItems(literal).apply {
            getAsJsonObject("elements").getAsJsonObject("listing").getAsJsonObject("props")
                .addProperty("statePath", "/items")
        }
        val withEvent = graphWithItems(literal).apply {
            getAsJsonObject("elements").getAsJsonObject("listing").add("on", JsonObject())
        }
        val withRepeat = graphWithItems(literal).apply {
            getAsJsonObject("elements").getAsJsonObject("listing").add("repeat", JsonObject())
        }
        listOf(withChildren, withStatePath, withEvent, withRepeat).forEach { graph ->
            assertSame(graph, EntityListPresentation.normalize(graph).graph)
        }
    }

    @Test fun malformedGraphShortAndExcessiveListsRemainUnchangedAndPromotionIsIdempotent() {
        val malformed = JsonObject().apply { addProperty("root", "root") }
        assertSame(malformed, EntityListPresentation.normalize(malformed).graph)
        val one = graphWithItems(strings("**Akasa Air**: ₹7,195 on 15 Oct."))
        assertSame(one, EntityListPresentation.normalize(one).graph)
        val excessive = graphWithItems(strings(*Array(25) { index ->
            "**Airline $index**: ₹7,195 on 15 Oct."
        }))
        assertSame(excessive, EntityListPresentation.normalize(excessive).graph)
        val longTitle = "A".repeat(45)
        val tooWide = graphWithItems(strings(
            "**$longTitle**: ₹7,195 on 15 Oct.",
            "**IndiGo**: ₹15,599 on 15 Oct.",
        ))
        assertSame(tooWide, EntityListPresentation.normalize(tooWide).graph)
        val source = graphWithItems(strings(
            "**Akasa Air**: ₹7,195 on 15 Oct.",
            "**IndiGo**: ₹15,599 on 15 Oct.",
        ))
        val once = EntityListPresentation.normalize(source)
        val twice = EntityListPresentation.normalize(once.graph)
        assertEquals(1, once.convertedLists)
        assertEquals(0, twice.convertedLists)
        assertSame(once.graph, twice.graph)
        assertFalse(once.graph == source)
    }

    private fun captured(id: String): JsonObject {
        val resource = requireNotNull(javaClass.getResource("/entity_list_presentation/$id.repaired.a2ui.json"))
        return A2uiWireCodec.decode(JsonParser.parseString(resource.readText()))
    }

    private fun assertCardTable(table: JsonObject, expected: List<Pair<String, String>>) {
        assertEquals("Table", table.get("type").asString)
        assertEquals(0, table.getAsJsonArray("children").size())
        val props = table.getAsJsonObject("props")
        assertEquals(setOf("columns", "rows", "domain", "preferredPresentation", "primaryColumn"), props.keySet())
        assertEquals(listOf("Option", "Details"), props.getAsJsonArray("columns").map { it.asString })
        assertEquals("comparison", props.get("domain").asString)
        assertEquals("cards", props.get("preferredPresentation").asString)
        assertEquals("Option", props.get("primaryColumn").asString)
        assertEquals(expected, props.getAsJsonArray("rows").map { row ->
            row.asJsonArray[0].asString to row.asJsonArray[1].asString
        })
        val rendererProps = mapOf<String, Any?>(
            "columns" to listOf("Option", "Details"),
            "rows" to expected.map { listOf(it.first, it.second) },
            "domain" to "comparison",
            "preferredPresentation" to "cards",
            "primaryColumn" to "Option",
        )
        listOf(360, 800).forEach { width ->
            val model = requireNotNull(extractDirectTableModel(rendererProps, emptyMap(), width < 600))
            assertEquals(FlatTableShape.ENTITY_ROW, model.shape)
            assertEquals(FlatTableRenderMode.RESPONSIVE_CARD_ROWS, model.renderMode)
            assertEquals(AdaptiveTablePresentation.ENTITY_CARDS,
                selectAdaptiveTablePresentation(model, width, false, true, true))
        }
    }

    private fun assertExactSuffixes(before: JsonObject, after: JsonObject) {
        val items = before.getAsJsonObject("props").getAsJsonArray("items")
        val rows = after.getAsJsonObject("props").getAsJsonArray("rows")
        assertEquals(items.size(), rows.size())
        (0 until items.size()).forEach { index ->
            val original = items[index].asString
            val row = rows[index].asJsonArray
            assertTrue(original.contains(row[0].asString))
            assertTrue(original.endsWith(row[1].asString))
        }
    }

    private fun strings(vararg values: String): JsonArray = JsonArray().apply { values.forEach { add(it) } }

    private fun graphWithItems(items: JsonElement): JsonObject = JsonObject().apply {
        addProperty("root", "root")
        add("state", JsonObject().apply { addProperty("unchanged", "state") })
        add("elements", JsonObject().apply {
            add("root", JsonObject().apply {
                addProperty("type", "Stack")
                add("props", JsonObject().apply { addProperty("direction", "vertical") })
                add("children", JsonArray().apply { add("listing") })
            })
            add("listing", JsonObject().apply {
                addProperty("type", "List")
                add("props", JsonObject().apply { add("items", items) })
                add("children", JsonArray())
            })
        })
    }
}
