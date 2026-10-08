package com.samsung.genuicraft.sdk.internal.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatTimedDayPlanTest {
    @Test fun `accepted Hindi plan keeps every day label task value and source order with equal task roles`() {
        val headers = listOf("Day", "10 min speaking", "10 min listening")
        val rows = listOf(
            listOf("1", "Practice greetings and introduce yourself aloud.", "Listen to a short beginner greeting lesson."),
            listOf("2", "Say common classroom phrases and numbers.", "Hear numbers and repeat each one."),
            listOf("3", "Practice asking and answering “What is your name?” and “How are you?”", "Listen to a short dialogue and shadow it."),
            listOf("4", "Describe basic needs: water, food, help, bathroom.", "Listen for key survival phrases and note them."),
            listOf("5", "Practice family words and simple possessives.", "Listen to a family-themed beginner audio lesson."),
            listOf("6", "Say where you are from and where you live.", "Listen to location and directions phrases."),
            listOf("7", "Review all week’s phrases in a 1-minute self-introduction.", "Replay your favorite lesson and catch every known word."),
        )
        assertEquals(TimedDayPlanProfile(0, listOf(1, 2)), timedDayPlanProfile(headers))
        rows.forEach { row ->
            val content = timedDayPlanRowContent(headers, row)!!
            assertEquals(headers[0] to row[0], content.dayCell.let { it.label to it.value })
            assertEquals(listOf(1, 2), content.bodyCells.map { it.index })
            assertEquals(headers.drop(1).zip(row.drop(1)), content.bodyCells.map { it.label to it.value })
            assertEquals(headers.zip(row), content.representedSourceCells.map { it.label to it.value })
        }
        assertEquals(rows[2][1], timedDayPlanRowContent(headers, rows[2])!!.bodyCells.first().value)
    }

    @Test fun `permuted day multiple timed activities and long unknown details retain every literal field`() {
        val headers = listOf("Reading (20 minutes)", "Source note", "Day", "1 hr conversation", "Listening (10 min)")
        val longValue = "Keep every clause of the original activity and its qualification [4]. ".repeat(30)
        val row = listOf(longValue, "Not verified in the source [9]", "Monday [1]",
            "Discuss the complete supplied subject; do not infer a total time [6].", "Listen to the supplied dialogue [8].")
        val profile = timedDayPlanProfile(headers)!!
        val content = timedDayPlanRowContent(headers, row)!!

        assertEquals(TimedDayPlanProfile(2, listOf(0, 3, 4)), profile)
        assertEquals("Monday [1]", content.dayCell.value)
        assertEquals(listOf(0, 1, 3, 4), content.bodyCells.map { it.index })
        assertEquals(longValue, content.bodyCells.first().value)
        assertEquals(headers.zip(row), content.representedSourceCells.map { it.label to it.value })
    }

    @Test fun `untimed single activity week feature and duration-only schemas retain their current route`() {
        listOf(
            listOf("Day", "Speaking", "Listening"),
            listOf("Day", "10 min speaking", "Notes"),
            listOf("Week", "10 min speaking", "10 min listening"),
            listOf("Feature", "10 min speaking", "10 min listening"),
            listOf("Day", "10 min", "20 minutes"),
            listOf("Day", "day", "10 min speaking", "10 min listening"),
        ).forEach { headers -> assertNull(timedDayPlanProfile(headers)) }
        val headers = listOf("Day", "10 min speaking", "10 min listening")
        listOf("Href", "Terms link", "CTA label", "Photo URL", "Media", "Phone", "Booking").forEach { authoredColumn ->
            assertNull(timedDayPlanProfile(headers + authoredColumn))
        }
    }

    @Test fun `incomplete ragged bound and URL-bearing rows keep their existing renderer`() {
        val headers = listOf("Day", "10 min speaking", "10 min listening")
        val row = listOf("3", "Complete speaking activity [1]", "Complete listening activity [2]")
        assertNull(timedDayPlanRowContent(headers, row.dropLast(1)))
        assertNull(timedDayPlanRowContent(headers, row + "Unheaded qualification [7]"))
        row.indices.forEach { index ->
            val blank = row.mapIndexed { cellIndex, value -> if (cellIndex == index) "" else value }
            assertNull(timedDayPlanRowContent(headers, blank))
        }
        listOf("{{ /activity }}", "https://www.who.int", "www.who.int", "who.int", "//www.who.int",
            "tel:+911234567890", "Use the authored link https://www.who.int").forEach { authoredValue ->
            assertNull(timedDayPlanRowContent(headers, listOf(row[0], authoredValue, row[2])))
        }
    }
}
