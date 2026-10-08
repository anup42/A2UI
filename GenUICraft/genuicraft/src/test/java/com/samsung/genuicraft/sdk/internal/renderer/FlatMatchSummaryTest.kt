package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatMatchSummaryTest {
    private fun text(value: String, variant: String = "caption") =
        FlatElement("Text", mapOf("text" to value, "variant" to variant), emptyList())

    private fun requiredFields(): Map<String, FlatElement> = linkedMapOf(
        "format" to text("Format: Test match"),
        "first" to text("India scores: 462 and 193/48.5"),
        "second" to text("Sri Lanka scores: 284 and 206"),
        "result" to text("Result: India won by 165 runs"),
    )

    @Test fun `accepted match fields preserve score literals date opponent roles and both root and text identities`() {
        val elements = linkedMapOf(
            "a" to text("Latest completed India men’s international match", "h2"),
            "b" to FlatElement("Card", emptyMap(), listOf("c")),
            "c" to text("Date: July 2026, at Galle International Stadium, Galle[13]"),
            "d" to text("Opponent: Sri Lanka"),
            "e" to text("Format: Test match"),
            "f" to text("India scores: 462 and 193/48.5"),
            "g" to text("Sri Lanka scores: 284 and 206"),
            "h" to text("Result: India won by 165 runs"),
            "i" to text("Standout performances", "h2"),
            "j" to FlatElement("Card", emptyMap(), listOf("k")),
            "k" to FlatElement("Stack", mapOf("direction" to "vertical"), listOf("l", "m")),
            "l" to text("India’s first-innings 462 set the match in control and created a large lead that Sri Lanka could not overcome.", "body"),
            "m" to text("India’s all-round bowling effort restricted Sri Lanka to 284 in the first innings and 206 in the chase, sealing the win by a substantial margin.", "body"),
        )
        val children = listOf("a", "b", "d", "e", "f", "g", "h", "i", "j")
        val snapshot = elements.toMap()
        val summary = planFlatMatchSummary(children, elements)!!

        assertEquals(listOf("b", "d", "e", "f", "g", "h"), summary.rootIds)
        assertEquals(listOf("c", "d", "e", "f", "g", "h"), summary.fields.map { it.textId })
        assertEquals(listOf("462 and 193/48.5", "284 and 206"), summary.scores.map { it.value })
        assertEquals(listOf("Date", "Opponent", "Format"), summary.details.map { it.label })
        assertEquals("July 2026, at Galle International Stadium, Galle[13]", summary.details.first().value)
        assertEquals("India won by 165 runs", summary.result.value)
        assertEquals(summary.fields.toSet(), (summary.scores + summary.details + summary.result).toSet())
        assertEquals(listOf("a", "i", "j"), children.filterNot { it in summary.rootIds })
        assertEquals(snapshot, elements)
    }

    @Test fun `arbitrary team names unknown fields and qualified score values stay literal`() {
        val elements = requiredFields().toMutableMap()
        elements["first"] = text("Team Red score: Not stated in the excerpt [4]")
        elements["second"] = text("Team Blue scores: 0/0; exact qualifier: no innings inference [7]")
        elements["extra"] = text("Additional condition: Unverified; retain every clause [11].")
        val children = listOf("format", "first", "second", "extra", "result")
        val summary = planFlatMatchSummary(children, elements)!!

        assertEquals(children, summary.rootIds)
        assertEquals(listOf("Team Red score", "Team Blue scores"), summary.scores.map { it.label })
        assertEquals(listOf("Not stated in the excerpt [4]", "0/0; exact qualifier: no innings inference [7]"), summary.scores.map { it.value })
        assertEquals("Unverified; retain every clause [11].", summary.details.last().value)
        assertEquals(children.toSet(), summary.fields.map { it.textId }.toSet())
    }

    @Test fun `weak incomplete duplicate and reused identity fields keep their authored renderer`() {
        val fields = requiredFields()
        fields.keys.forEach { missing -> assertNull(planFlatMatchSummary((fields - missing).keys.toList(), fields - missing)) }
        val weak = fields + ("first" to text("India total: 462 and 193/48.5"))
        assertNull(planFlatMatchSummary(weak.keys.toList(), weak))
        listOf("RESULT: A second result [8]", "india scores: Duplicate team label", "Third team score: 200").forEach { extraText ->
            val duplicate = fields + ("extra" to text(extraText))
            assertNull(planFlatMatchSummary(duplicate.keys.toList(), duplicate))
        }
        assertNull(planFlatMatchSummary(fields.keys.toList() + "first", fields))
        val reused = fields + mapOf(
            "one" to FlatElement("Card", emptyMap(), listOf("first")),
            "two" to FlatElement("Card", emptyMap(), listOf("first")),
        )
        assertNull(planFlatMatchSummary(listOf("format", "one", "two", "second", "result"), reused))
    }

    @Test fun `interactive dynamic styled nested multiline and heading score nodes remain on their existing route`() {
        val original = requiredFields().getValue("first")
        val guarded = listOf(
            original.copy(on = mapOf("click" to "originalAction")),
            original.copy(watch = mapOf("/score" to "originalWatch")),
            original.copy(visible = mapOf("path" to "/show")),
            original.copy(repeat = RepeatConfig("/scores")),
            original.copy(props = mapOf("text" to mapOf("path" to "/score"))),
            text("India scores: {{ /score }}"),
            text("India scores: 462\nand 193/48.5"),
            text("India scores: 462", "h2"),
            original.copy(props = original.props + ("color" to "#112233")),
            original.copy(children = listOf("nested")),
            FlatElement("Card", mapOf("padding" to 16), listOf("wrapped")),
            FlatElement("Card", emptyMap(), listOf("wrapped", "nested")),
        )
        guarded.forEach { node ->
            val fields = requiredFields() + mapOf("first" to node, "wrapped" to original, "nested" to text("Original nested text"))
            val snapshot = fields.toMap()
            assertNull(planFlatMatchSummary(requiredFields().keys.toList(), fields))
            assertEquals(snapshot, fields)
        }
    }

    @Test fun `noncontiguous fields and two complete candidate runs are ambiguous and remain unchanged`() {
        val fields = requiredFields() + ("separator" to FlatElement("Divider", emptyMap(), emptyList()))
        assertNull(planFlatMatchSummary(listOf("format", "first", "separator", "second", "result"), fields))
        val second = requiredFields().mapKeys { (id, _) -> "next_$id" }
        val both = fields + second
        assertNull(planFlatMatchSummary(requiredFields().keys.toList() + "separator" + second.keys.toList(), both))
        assertNull(planFlatMatchSummary(listOf("format", "first", "missing", "second", "result"), fields))
    }
}
