package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.planFlatTextSections
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatTextSectionPolicyTest {
    private fun text(value: String, variant: String = "body") =
        FlatElement("Text", mapOf("text" to value, "variant" to variant), emptyList())

    private fun fixture(): Map<String, FlatElement> = linkedMapOf(
        "root" to FlatElement("Stack", mapOf("direction" to "vertical"), listOf("a", "b", "c", "d")),
        "a" to text("First section", "h2"),
        "b" to text("Full paragraph with 55 × 35 × 25 cm and source [12]. ".repeat(20)),
        "c" to text("Second section", "h2"),
        "d" to text("Original exception and qualification [7]."),
    )

    private fun plan(elements: Map<String, FlatElement>, isRoot: Boolean = true, repeated: Boolean = false) =
        planFlatTextSections("root", elements.getValue("root").children, elements, isRoot, "vertical", repeated)

    @Test fun `groups sections without changing nodes text citations or order`() {
        val elements = fixture()
        val snapshot = elements.toMap()
        val sections = plan(elements)!!
        assertEquals(listOf("a", "c"), sections.map { it.headingId })
        assertEquals(elements.getValue("root").children, sections.flatMap { it.childIds })
        assertEquals(snapshot, elements)
        assertEquals(snapshot.getValue("b").props["text"], elements.getValue("b").props["text"])
    }

    @Test fun `intro and page title remain outside the section panels`() {
        val elements = fixture().toMutableMap()
        elements["title"] = text("Page title", "h1")
        elements["intro"] = text("An introductory paragraph [1].")
        elements["root"] = elements.getValue("root").copy(children = listOf("title", "intro", "a", "b", "c", "d"))
        val sections = plan(elements)!!
        assertEquals(listOf(null, null, "a", "c"), sections.map { it.headingId })
        assertEquals(elements.getValue("root").children, sections.flatMap { it.childIds })
    }

    @Test fun `mixed nested interactive and unresolved graphs keep their existing renderer`() {
        listOf("Card", "Table", "Button", "Stack").forEach { type ->
            val elements = fixture().toMutableMap()
            elements["d"] = FlatElement(type, emptyMap(), emptyList())
            assertNull(plan(elements))
        }
        val interactive = fixture().toMutableMap()
        interactive["b"] = interactive.getValue("b").copy(on = mapOf("click" to "originalAction"))
        assertNull(plan(interactive))
        assertEquals(mapOf("click" to "originalAction"), interactive.getValue("b").on)
        assertNull(plan(fixture() - "d"))
        assertNull(plan(fixture(), isRoot = false))
        assertNull(plan(fixture(), repeated = true))
    }

    @Test fun `dynamic visibility repeat and text bindings do not get regrouped`() {
        val visible = fixture().toMutableMap()
        visible["b"] = visible.getValue("b").copy(visible = mapOf("path" to "/show"))
        assertNull(plan(visible))
        val repeated = fixture().toMutableMap()
        repeated["b"] = repeated.getValue("b").copy(repeat = RepeatConfig("/items"))
        assertNull(plan(repeated))
        val bound = fixture().toMutableMap()
        bound["b"] = bound.getValue("b").copy(props = mapOf("text" to mapOf("path" to "/message")))
        assertNull(plan(bound))
    }

    @Test fun `section identities stay stable as streaming paragraphs arrive`() {
        val initial = fixture().toMutableMap()
        initial["root"] = initial.getValue("root").copy(children = listOf("a", "b", "c"))
        val first = plan(initial)!!
        val later = fixture().toMutableMap()
        later["extra"] = text("An additional complete paragraph [9].")
        later["root"] = later.getValue("root").copy(children = listOf("a", "b", "extra", "c", "d"))
        val next = plan(later)!!
        assertEquals(listOf("a", "c"), first.map { it.stableKey })
        assertEquals(first.map { it.stableKey }, next.map { it.stableKey })
        assertEquals(listOf("a", "b", "extra"), next.first().childIds)
        assertEquals(later.getValue("root").children, next.flatMap { it.childIds })
    }
}
