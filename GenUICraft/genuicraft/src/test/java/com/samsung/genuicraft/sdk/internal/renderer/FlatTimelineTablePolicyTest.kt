package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.tableRowAccessibilitySummary
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import org.junit.Assert.*
import org.junit.Test

class FlatTimelineTablePolicyTest {
    // Exact BXP-048 native rows; a2ui.json SHA-256 d103501d60a8b1d19ecdaf567ca901aff2ce78b6cf3f1c89eee56bd6ac5b97bd.
    private val rows = listOf(
        linkedMapOf("year" to "1336", "event" to "Vijayanagara was founded, and Hampi became its capital", "source" to "[1]"),
        linkedMapOf("year" to "14th to 16th centuries", "event" to "the city expanded into a major capital with temples, royal spaces, markets, roads, and waterworks", "source" to ""),
        linkedMapOf("year" to "1565", "event" to "the city was conquered, pillaged, and abandoned after the Battle of Talikota and the Deccan confederacy\u2019s victory", "source" to ""),
    )
    private fun state(values: Any = rows): Map<String, Any?> = mapOf("timeline_rows" to values)
    private fun table(extra: Map<String, Any?> = emptyMap()) = FlatElement("Table",
        mapOf<String, Any?>("columns" to linkedMapOf("year" to "year", "event" to "event"),
            "statePath" to "/timeline_rows", "domain" to "generic", "preferredPresentation" to "cards",
            "primaryColumn" to "year") + extra, emptyList())

    @Test fun nativeMapColumnsRetainAllSevenValuesAndCitationRowAssociation() {
        val node = table()
        val ancestors = listOf(FlatElement("Stack", mapOf("gap" to "md", "padding" to 16), emptyList()))
        val profile = requireNotNull(timelineTableProfile(node, state(), ancestors))
        assertEquals(listOf("Year", "Event", "Source"), profile.headers)
        assertEquals(rows.map { listOf(it.getValue("year"), it.getValue("event"), it.getValue("source")) }, profile.originalRows)
        assertEquals(7, profile.originalRows.sumOf { row -> row.count(String::isNotEmpty) })
        assertEquals(rows.map { it.getValue("year") }, profile.entries.map { it.period })
        assertEquals(rows.map { it.getValue("event") }, profile.entries.map { it.event })
        assertEquals(listOf("[1]", "", ""), profile.entries.map { it.citation })
        profile.entries.forEachIndexed { index, entry ->
            val description = tableRowAccessibilitySummary(profile.headers, profile.originalRows[index], index)
            assertTrue(description.contains("Year: " + entry.period))
            assertTrue(description.contains("Event: " + entry.event))
            assertEquals(index == 0, description.contains("Source: [1]"))
        }
        listOf(true, false).forEach { compact ->
            val model = requireNotNull(extractDirectTableModel(node.props, state(), compact))
            assertTrue(timelineMatchesResolved(profile, model))
        }
        assertEquals(rows, state()["timeline_rows"])
    }

    @Test fun positionalRowsKeepPeriodsFullMultilineBodiesWhitespaceAndHeaders() {
        val period = "14th to 16th\ncenturies"
        val body = "  First original paragraph [3]; retain its qualifier.\nSecond original paragraph.  "
        val originalRows = listOf(listOf(period, body), listOf("1565", "Second supplied event."))
        val node = FlatElement("Table", mapOf("columns" to listOf("Year", "Event"), "rows" to originalRows,
            "domain" to "generic", "preferredPresentation" to "cards", "title" to "Authored timeline"), emptyList())
        val profile = requireNotNull(timelineTableProfile(node, emptyMap()))
        assertEquals(listOf("Year", "Event"), profile.headers)
        assertEquals(originalRows, profile.originalRows)
        assertEquals(period, profile.entries[0].period)
        assertEquals(body, profile.entries[0].event)
        assertEquals("Authored timeline", profile.title)
        assertTrue(profile.entries.all { it.citation.isEmpty() })
        assertTrue(timelineMatchesResolved(profile, requireNotNull(extractDirectTableModel(node.props, emptyMap(), true))))
    }

    @Test fun guardedInteractiveRepeatedAndUnknownInputsRetainExistingRoute() {
        val node = table()
        val guarded = listOf(node.copy(visible = false), node.copy(visible = true),
            node.copy(repeat = RepeatConfig("/timeline_rows")), node.copy(on = mapOf("click" to "Original action")),
            node.copy(watch = mapOf("/timeline_rows" to "Original watch")))
        guarded.forEach { assertNull(timelineTableProfile(it, state())) }
        guarded.forEach { assertNull(timelineTableProfile(node, state(), listOf(it.copy(type = "Card")))) }
        assertNull(timelineTableProfile(node, state(), isRepeated = true))
        assertNull(timelineTableProfile(node, state(), listOf(null)))
        assertNull(timelineTableProfile(node.copy(children = listOf("Original child")), state()))
        listOf("unknown", "\$expr", "\$computed", "subtitle", "presentation").forEach {
            assertNull(timelineTableProfile(table(mapOf(it to "Original metadata")), state()))
        }
        assertNull(timelineTableProfile(node, state(), listOf(
            FlatElement("Card", mapOf("title" to mapOf("\$computed" to "value")), emptyList()))))
    }

    @Test fun dynamicCellsPointersUrlsRowActionsAndNonNumericCitationsDecline() {
        val dynamic = listOf("{{/event}}", "Unclosed {{/event", "Closing /event}}", "\$/event",
            "\$state.event", "\${event}", "@source[1]", "Read https://example.com/", "mailto:help@example.com")
        dynamic.forEach { value ->
            assertNull(timelineTableProfile(table(), state(rows.mapIndexed { index, row ->
                if (index == 0) row + ("event" to value) else row
            })))
        }
        listOf("{{/timeline_rows}}", "\$/timeline_rows", "https://example.com/rows").forEach {
            assertNull(timelineTableProfile(table(mapOf("statePath" to it)), state()))
        }
        listOf("action", "on", "condition", "visible", "location", "unknown").forEach { key ->
            assertNull(timelineTableProfile(table(), state(rows.mapIndexed { index, row ->
                if (index == 0) row + (key to "Supplied extra field") else row
            })))
        }
        assertNull(timelineTableProfile(table(), state(rows.mapIndexed { index, row ->
            if (index == 0) row + ("source" to "UNESCO report") else row
        })))
    }

    @Test fun explicitGridsOtherDomainsAndMeaningfulExtraColumnsAreUntouched() {
        assertNull(timelineTableProfile(table(mapOf("preferredPresentation" to "table")), state()))
        assertNull(timelineTableProfile(table(mapOf("domain" to "comparison")), state()))
        assertNull(timelineTableProfile(table(mapOf("columns" to listOf("Factor", "Year", "Event"))), state()))
        assertNull(timelineTableProfile(table(mapOf("columns" to listOf("Year", "Event", "Status"))), state()))
        assertNull(timelineTableProfile(table(mapOf("columns" to linkedMapOf("year" to "year", "event" to "event", "status" to "status"))), state()))
        assertNull(timelineTableProfile(table(mapOf("primaryColumn" to "event")), state()))
        assertNull(timelineTableProfile(table(mapOf("columns" to listOf(
            mapOf("key" to "year", "label" to "Date"), mapOf("key" to "event", "label" to "Event")))), state()))
    }

    @Test fun unprojectedSourceMetadataIsNotSilentlyDiscarded() {
        val projected = table(mapOf("columns" to listOf(
            mapOf("key" to "year", "label" to "Year"), mapOf("key" to "event", "label" to "Event"))))
        assertNull(timelineTableProfile(projected, state()))
        val declaredSource = projected.copy(props = projected.props + ("columns" to listOf(
            mapOf("key" to "year", "label" to "Year"), mapOf("key" to "event", "label" to "Event"),
            mapOf("key" to "source", "label" to "Source"))))
        val profile = requireNotNull(timelineTableProfile(declaredSource, state()))
        assertEquals("[1]", profile.entries[0].citation)
        assertEquals(7, profile.originalRows.sumOf { row -> row.count(String::isNotEmpty) })
    }

    @Test fun suppliedOrderingDuplicateYearsAndCitationsStayIntactAndStaleProjectionDeclines() {
        val supplied = listOf(rows[2], rows[0], rows[0], rows[1])
        val node = table()
        val profile = requireNotNull(timelineTableProfile(node, state(supplied)))
        assertEquals(listOf("1565", "1336", "1336", "14th to 16th centuries"), profile.entries.map { it.period })
        assertEquals(listOf("", "[1]", "[1]", ""), profile.entries.map { it.citation })
        assertTrue(timelineMatchesResolved(profile, requireNotNull(extractDirectTableModel(node.props, state(supplied), true))))
        assertFalse(timelineMatchesResolved(profile, requireNotNull(extractDirectTableModel(node.props, state(), true))))
        assertFalse(timelineMatchesResolved(profile, requireNotNull(extractDirectTableModel(
            node.props + ("preferredPresentation" to "table"), state(supplied), true))))
    }
}
