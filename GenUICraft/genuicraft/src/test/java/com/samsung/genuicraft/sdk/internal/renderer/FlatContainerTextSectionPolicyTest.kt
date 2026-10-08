package com.samsung.genuicraft.sdk.internal.renderer

import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.sdk.internal.pipeline.GenUiIrCodec
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.FlatSpecParser
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.planFlatContainerTextSections
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FlatContainerTextSectionPolicyTest {
    @Test fun `native wire input still groups after host gutter normalization`() {
        val wire = javaClass.getResource("/rendering/bxp022_gardening.a2ui.json")!!.readText()
        val parsed = FlatSpecParser.parse(GenUiIrCodec.decode(JsonPrimitive(wire)).canonicalGraph)!!
        val normalized = parsed.withCollapsedRootHorizontalPadding()
        val sections = plan(normalized.elements)!!
        assertEquals(0, normalized.elements.getValue("root").props["paddingHorizontal"])
        assertEquals(listOf("c", "g", "l", "q", "u"), sections.map { it.headingId })
        sections.flatMap { it.childIds }.forEach { id ->
            assertEquals(parsed.elements.getValue(id), normalized.elements.getValue(id))
        }
    }

    private fun text(value: String, variant: String = "body") =
        FlatElement("Text", mapOf("text" to value, "variant" to variant), emptyList())

    private fun stack(children: List<String>) =
        FlatElement("Stack", mapOf("direction" to "vertical", "gap" to "sm", "padding" to 16), children)

    private fun fixture(): Map<String, FlatElement> = linkedMapOf(
        "root" to FlatElement("Stack", mapOf("direction" to "vertical", "gap" to "md", "padding" to 16), listOf("a", "f", "j", "o", "t")),
        "a" to FlatElement("Card", emptyMap(), listOf("b")),
        "b" to stack(listOf("c", "d", "e")),
        "c" to text("Best planting season", "h2"),
        "d" to text("In Bengaluru, tomatoes can be planted through much of the year, but they do best when started in the cooler part of the dry season rather than in peak rain or extreme heat.[5]"),
        "e" to text("For balcony containers, aim for January to March or after heavy rains ease, so young plants establish in milder weather."),
        "f" to stack(listOf("g", "h", "i")),
        "g" to text("Container size", "h2"),
        "h" to text("Use at least a 12-inch pot for compact tomato varieties, but 20–25 L is a better target for productive container growth.[1][3][4]"),
        "i" to text("For larger or climbing types, choose 25–40 L or more; smaller pots commonly reduce yield and can cause fruiting problems."),
        "j" to FlatElement("Card", emptyMap(), listOf("k")),
        "k" to stack(listOf("l", "m", "n")),
        "l" to text("Sunlight", "h2"),
        "m" to text("Place the pot where it gets at least 6 hours of direct sun daily.[6]"),
        "n" to text("A south- or west-facing balcony is usually best, but avoid letting taller plants shade the tomato."),
        "o" to FlatElement("Card", emptyMap(), listOf("p")),
        "p" to stack(listOf("q", "r", "s")),
        "q" to text("Watering", "h2"),
        "r" to text("Water regularly and evenly so the soil stays moist but not soggy; container tomatoes dry out faster than ground plants."),
        "s" to text("Check daily in hot weather, and make sure the pot has good drainage.[9][10]"),
        "t" to stack(listOf("u", "v", "w")),
        "u" to text("Two common problems", "h2"),
        "v" to text("Blossom drop or poor fruit set: often linked to undersized pots, heat stress, or inconsistent watering."),
        "w" to text("Blossom-end rot: usually caused by irregular moisture and calcium uptake problems, especially in small containers."),
        "x" to FlatElement("Card", emptyMap(), listOf("f")),
        "y" to FlatElement("Card", emptyMap(), listOf("t")),
        "z" to FlatElement("Button", mapOf("label" to "Continue", "variant" to "primary"), emptyList()),
    )

    private fun plan(elements: Map<String, FlatElement>, isRoot: Boolean = true, direction: String = "vertical", repeated: Boolean = false) =
        planFlatContainerTextSections("root", elements.getValue("root").children, elements, isRoot, direction, repeated)

    @Test fun `accepted gardening graph preserves all reachable text IDs props order and citations without inferring unreachable wrappers`() {
        val elements = fixture()
        val snapshot = elements.toMap()
        val sections = plan(elements)!!
        val expected = listOf(listOf("c", "d", "e"), listOf("g", "h", "i"), listOf("l", "m", "n"), listOf("q", "r", "s"), listOf("u", "v", "w"))

        assertEquals(expected, sections.map { it.childIds })
        assertEquals(expected.map { it.first() }, sections.map { it.headingId })
        assertEquals(expected.map { it.first() }, sections.map { it.stableKey })
        assertEquals(expected.flatten().map { snapshot.getValue(it).props }, sections.flatMap { it.childIds }.map { elements.getValue(it).props })
        assertEquals("Use at least a 12-inch pot for compact tomato varieties, but 20–25 L is a better target for productive container growth.[1][3][4]", elements.getValue("h").props["text"])
        assertEquals(sections, plan(elements - setOf("x", "y", "z")))
        assertEquals(snapshot, elements)
        assertEquals(listOf("f"), elements.getValue("x").children)
        assertEquals(listOf("t"), elements.getValue("y").children)
    }

    @Test fun `plain direct text cards default vertical stacks and h3 headings use the same original IDs`() {
        val elements = fixture().toMutableMap()
        elements["a"] = FlatElement("Card", mapOf("padding" to 12), listOf("c", "d", "e"))
        elements["f"] = elements.getValue("f").copy(props = mapOf("gap" to "sm", "padding" to 16))
        elements["g"] = text("Container size", "h3")
        val snapshot = elements.toMap()
        val sections = plan(elements)!!
        assertEquals(listOf("c", "g", "l", "q", "u"), sections.map { it.headingId })
        assertEquals(listOf("c", "d", "e"), sections.first().childIds)
        assertEquals(snapshot, elements)
    }

    @Test fun `custom containers actions visibility repetition watches and bound text retain their existing route`() {
        val original = fixture()
        val guarded = listOf(
            "a" to original.getValue("a").copy(props = mapOf("tone" to "primary")),
            "a" to original.getValue("a").copy(props = mapOf("title" to "Original card title")),
            "a" to original.getValue("a").copy(props = mapOf("style" to mapOf("background" to "#112233"))),
            "root" to original.getValue("root").copy(on = mapOf("click" to "originalRootAction")),
            "a" to original.getValue("a").copy(on = mapOf("click" to "originalAction")),
            "b" to original.getValue("b").copy(visible = mapOf("path" to "/show")),
            "f" to original.getValue("f").copy(repeat = RepeatConfig("/items")),
            "t" to original.getValue("t").copy(watch = mapOf("/value" to "originalWatch")),
            "f" to original.getValue("f").copy(props = mapOf("direction" to "horizontal")),
            "f" to original.getValue("f").copy(props = mapOf("padding" to mapOf("path" to "/padding"))),
            "f" to original.getValue("f").copy(props = mapOf("gap" to "{{ /gap }}")),
            "h" to original.getValue("h").copy(props = mapOf("text" to mapOf("path" to "/body"))),
            "h" to text("Original {{ /body }} binding"),
            "h" to original.getValue("h").copy(on = mapOf("click" to "originalTextAction")),
        )
        guarded.forEach { (id, node) ->
            val elements = original + (id to node)
            val snapshot = elements.toMap()
            assertNull(plan(elements))
            assertEquals(snapshot, elements)
        }
    }

    @Test fun `duplicate traversed IDs missing nodes nested structures incomplete sections and nonroot layouts stay unchanged`() {
        val original = fixture()
        val guarded = listOf(
            original + ("root" to original.getValue("root").copy(children = listOf("a", "a", "j", "o", "t"))),
            original + ("f" to original.getValue("f").copy(children = listOf("g", "h", "e"))),
            original + ("j" to original.getValue("j").copy(children = listOf("b"))),
            original + ("b" to original.getValue("b").copy(children = listOf("c", "b"))),
            original - "n",
            original + ("a" to original.getValue("a").copy(children = listOf("z", "c"))),
            original + ("b" to original.getValue("b").copy(children = listOf("c"))),
            original + ("c" to text("Best planting season", "body")),
            original + ("m" to text("Second heading inside one section", "h3")),
            original + ("root" to original.getValue("root").copy(children = listOf("a"))),
        )
        guarded.forEach { elements -> assertNull(plan(elements)) }
        assertNull(plan(original, isRoot = false))
        assertNull(plan(original, direction = "horizontal"))
        assertNull(plan(original, repeated = true))
    }
}
