package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.extractDirectTableModel
import com.samsung.genuicraft.sdk.internal.renderer.flat.compose.table.tableRowAccessibilitySummary
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.dayActivityScheduleCardContent
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.responsiveScheduleCardContent
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.shouldStackDayActivitySchedule
import org.junit.Assert.*
import org.junit.Test

class FlatDayActivityScheduleTest {
    // Literal rows from saved native BXP-036 a2ui.json; no source-text reconstruction.
    // Native JSON SHA-256: 03bc7f8a38e3c5f46eddcf7b1bc64dcade0785427e05e94c09ad49c0dbaae056
    private val rows = listOf(
        listOf("Day 1", "Write down the top 3 work stressors and 1 thing you can pause or delegate this week"),
        listOf("Day 2", "Set a firm start and stop time for work, and protect a real break away from screens"),
        listOf("Day 3", "Use 10 minutes of movement, such as a walk or stretching, to interrupt tension"),
        listOf("Day 4", "Try one calming practice, such as slow breathing, mindfulness, or meditation, for 5 to 10 minutes"),
        listOf("Day 5", "Ask one trusted person for support and talk through what feels hardest"),
        listOf("Day 6", "Review sleep habits and aim for a consistent bedtime and wake time"),
        listOf("Day 7", "Reassess your workload and decide what needs a manager conversation or longer-term change"),
    )
    private val headers = listOf("Day", "Activity")

    @Test fun `saved 036 stays on schedule cards and preserves all seven complete activities`() {
        val stateRows = rows.map { row -> linkedMapOf("day" to row[0], "activity" to row[1]) }
        val state = mapOf("coping_routine" to stateRows)
        val props = mapOf<String, Any?>(
            "columns" to listOf(mapOf("key" to "day", "label" to "Day"),
                mapOf("key" to "activity", "label" to "Activity")),
            "statePath" to "/coping_routine", "domain" to "schedule",
            "preferredPresentation" to "cards", "primaryColumn" to "day"
        )
        val model = requireNotNull(extractDirectTableModel(props, state, compactScreen = true))
        assertEquals(FlatTableShape.SCHEDULE_TIMELINE, model.shape)
        assertEquals(FlatTableRenderMode.RESPONSIVE_CARD_ROWS, model.renderMode)
        val labels = model.columns.map { it.label }
        assertEquals(headers, labels)
        assertEquals(rows, model.rows)
        assertEquals(7, model.rows.size)
        model.rows.forEachIndexed { index, row ->
            val content = requireNotNull(dayActivityScheduleCardContent(labels, row))
            assertEquals("Day ${index + 1}", content.day)
            assertEquals("Activity", content.activityLabel)
            assertEquals(rows[index], listOf(content.day, content.activity))
            assertEquals("Day: ${row[0]}. Activity: ${row[1]}", tableRowAccessibilitySummary(headers, row))
        }
        assertEquals(stateRows, state.getValue("coping_routine"))
    }

    @Test fun `task label citations whitespace punctuation and multiline body remain literal`() {
        val activity = "  Full first clause [3]; keep qualifier.\nSecond paragraph, exactly as emitted.  "
        val row = listOf("Day 12", activity)
        val content = requireNotNull(dayActivityScheduleCardContent(listOf("Day", "Task"), row))
        assertEquals("Day 12", content.day)
        assertEquals("Task", content.activityLabel)
        assertEquals(activity, content.activity)
        assertEquals(listOf("Day 12", activity), row)
    }

    @Test fun `extra incomplete unknown and nonliteral day fields keep their existing route`() {
        val row = rows.first()
        listOf(listOf("Day", "Activity", "Status"), listOf("Week", "Activity"),
            listOf("Day", "Details"), listOf("Activity", "Day"), listOf("day", "Activity"),
            listOf("Day", "Activity / Task")).forEach { assertNull(dayActivityScheduleCardContent(it, row)) }
        listOf(emptyList(), listOf("Day 1"), row + "Authored extra field",
            listOf("Day 1", " ")).forEach { assertNull(dayActivityScheduleCardContent(headers, it)) }
        listOf("1", "Monday", "Day one", "Day 0", "Day -1", "Day 1-2", "Day 1 ", "Day 01")
            .forEach { assertNull(dayActivityScheduleCardContent(headers, listOf(it, row[1]))) }
    }

    @Test fun `URLs and unresolved flat or express bindings keep their existing route`() {
        listOf("Read https://example.com/details", "[Advice](HTTP://example.com)",
            "www.example.com", "mailto:help@example.com", "tel:123", "geo:1,2",
            "{{/activity}}", "Read {{/activity}} in full", "Unclosed {{/activity", "Closing /activity}}",
            "${'$'}/activity", "${'$'}{activity}", "${'$'}item.activity", "prefix ${'$'}index", "@source[1]")
            .forEach { assertNull(dayActivityScheduleCardContent(headers, listOf("Day 1", it))) }
        assertNull(dayActivityScheduleCardContent(headers, listOf("{{/day}}", "Keep literal activity")))
    }

    @Test fun `025 multi-column study and 035 fitness schedules retain every original field`() {
        val schedules = listOf(
            listOf("Day", "10 min speaking", "10 min listening") to
                listOf("Day 1", "Original speaking", "Original listening"),
            listOf("Week", "Walk", "Strength", "Recovery") to
                listOf("Week 1", "Original walk", "Original strength", "Original recovery")
        )
        schedules.forEach { (labels, row) ->
            assertNull(dayActivityScheduleCardContent(labels, row))
            val original = requireNotNull(responsiveScheduleCardContent(labels, row))
            assertEquals(labels.zip(row), original.displayedCells.map { it.label to it.value })
        }
    }

    @Test fun `small available width and large fonts stack while normal wide cards stay compact`() {
        assertFalse(shouldStackDayActivitySchedule(400f, 1f))
        assertFalse(shouldStackDayActivitySchedule(320f, 1.2f))
        assertTrue(shouldStackDayActivitySchedule(319f, 1f))
        assertTrue(shouldStackDayActivitySchedule(700f, 1.3f))
        assertTrue(shouldStackDayActivitySchedule(Float.NaN, 1f))
        assertTrue(shouldStackDayActivitySchedule(400f, Float.NaN))
    }
}
