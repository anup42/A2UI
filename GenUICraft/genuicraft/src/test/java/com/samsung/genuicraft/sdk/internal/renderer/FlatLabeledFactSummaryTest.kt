package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatLabeledFactSummaryTest {
    private fun text(value: String, variant: String = "label") =
        FlatElement("Text", mapOf("text" to value, "variant" to variant), emptyList())

    private fun requiredFields(): Map<String, FlatElement> = linkedMapOf(
        "first" to text("Mission name: Exact source name [1]"),
        "second" to text("Planned date: Date not confirmed [2]"),
        "third" to text("Schedule uncertainty: Keep the complete condition: the date remains provisional [3].", "caption"),
    )

    @Test fun `accepted launch facts retain every original field and the complete uncertainty while surrounding text stays outside`() {
        val uncertainty = "The public schedule is presented as a launch listing rather than an official fixed ISRO mission page, so the date should be treated as provisional until ISRO confirms it on its own launch list or another official update."
        val explanation = "ISRO’s official launch missions page currently shows the latest completed launch as GSLV-F17/EOS-05 on September 4, 2026, and does not yet list a later upcoming mission in the visible entries, which is why the next launch date is being inferred from third-party schedule trackers."
        val elements = linkedMapOf(
            "a" to text("Next publicly scheduled ISRO launch", "h2"),
            "b" to text("Mission name: Cartosat-3B"),
            "c" to text("Planned date: December 10, 2026"),
            "d" to text("Launch vehicle: PSLV-XL"),
            "e" to text("Payload purpose: Earth observation satellite"),
            "f" to text("Schedule uncertainty: $uncertainty", "caption"),
            "g" to text(explanation, "caption"),
        )
        val children = elements.keys.toList()
        val snapshot = elements.toMap()
        val facts = planFlatLabeledFacts(children, elements)!!

        assertEquals(listOf("b", "c", "d", "e", "f"), facts.map { it.id })
        assertEquals(listOf("Mission name", "Planned date", "Launch vehicle", "Payload purpose", "Schedule uncertainty"), facts.map { it.label })
        assertEquals(listOf("Cartosat-3B", "December 10, 2026", "PSLV-XL", "Earth observation satellite", uncertainty), facts.map { it.value })
        val paired = labeledFactRows(facts, canPair = true)
        assertEquals(listOf(listOf("b"), listOf("c", "d"), listOf("e"), listOf("f")), paired.map { group -> group.map { it.id } })
        assertEquals(facts.map { it.id }, paired.flatten().map { it.id })
        assertEquals(List(facts.size) { 1 }, labeledFactRows(facts, canPair = false).map { it.size })
        facts.forEach { fact -> assertEquals(elements.getValue(fact.id).props["text"], "${fact.label}: ${fact.value}") }
        assertEquals(listOf("a", "g"), children.filterNot { id -> facts.any { it.id == id } })
        assertEquals(explanation, elements.getValue("g").props["text"])
        assertEquals(snapshot, elements)
    }

    @Test fun `partial malformed duplicate labels and reused IDs remain unchanged`() {
        val fields = requiredFields()
        assertNull(planFlatLabeledFacts(fields.keys.toList().dropLast(1), fields))
        assertNull(planFlatLabeledFacts(fields.keys.toList() + "first", fields))
        listOf("MISSION NAME: Another authored name", "Planned date:", ": Missing label",
            "A B C D E F G: Too many label words", "${"A".repeat(49)}: Label is too long",
            "An explanatory sentence with no explicit label and value").forEach { invalidText ->
            val altered = fields + ("second" to text(invalidText))
            val snapshot = altered.toMap()
            assertNull(planFlatLabeledFacts(fields.keys.toList(), altered))
            assertEquals(snapshot, altered)
        }
    }

    @Test fun `interactive dynamic multiline URL heading styled and nested text nodes keep their original route`() {
        val original = requiredFields().getValue("second")
        val guarded = listOf(
            original.copy(on = mapOf("click" to "originalAction")),
            original.copy(watch = mapOf("/date" to "originalWatch")),
            original.copy(visible = mapOf("path" to "/show")),
            original.copy(repeat = RepeatConfig("/dates")),
            original.copy(props = mapOf("text" to mapOf("path" to "/date"))),
            text("Planned date: {{ /date }}"),
            text("Planned date: Original first line\nOriginal second line"),
            text("Planned date: https://www.who.int"),
            text("Planned date: Original date", "h2"),
            original.copy(props = original.props + ("color" to "#112233")),
            original.copy(children = listOf("nested")),
            FlatElement("Card", emptyMap(), listOf("wrapped")),
        )
        guarded.forEach { node ->
            val fields = requiredFields() + mapOf("second" to node, "wrapped" to original, "nested" to text("Original nested text"))
            val snapshot = fields.toMap()
            assertNull(planFlatLabeledFacts(requiredFields().keys.toList(), fields))
            assertEquals(snapshot, fields)
        }
    }

    @Test fun `noncontiguous facts and multiple complete candidate runs remain ambiguous`() {
        val first = requiredFields()
        val separator = mapOf("separator" to FlatElement("Divider", emptyMap(), emptyList()))
        assertNull(planFlatLabeledFacts(listOf("first", "separator", "second", "third"), first + separator))
        val second = first.mapKeys { (id, _) -> "next_$id" }
        assertNull(planFlatLabeledFacts(first.keys.toList() + "separator" + second.keys.toList(), first + separator + second))
        assertNull(planFlatLabeledFacts(listOf("first", "missing", "second", "third"), first))
    }
}
