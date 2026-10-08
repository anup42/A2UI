package com.samsung.genuicraft.sdk.internal.renderer

import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.filterAdjacentBareTextDuplicates
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import org.junit.Assert.assertEquals
import org.junit.Test

class FlatBareTextDuplicatePolicyTest {
    private fun text(value: String, variant: String = "body") =
        FlatElement("Text", mapOf("text" to value, "variant" to variant), emptyList())

    private fun fixture(): Map<String, FlatElement> = linkedMapOf(
        "root" to FlatElement("Stack", mapOf("direction" to "vertical", "gap" to "md", "padding" to 16), listOf("a", "b", "c", "i", "k")),
        "a" to text("Comparison table", "h2"),
        "b" to FlatElement("Table", mapOf("statePath" to "/comparison_data", "domain" to "comparison", "preferredPresentation" to "cards", "primaryColumn" to "approach"), emptyList()),
        "c" to FlatElement("Stack", mapOf("direction" to "vertical", "gap" to "sm"), listOf("d", "e", "f", "g")),
        "d" to text("Three use cases", "h2"),
        "e" to FlatElement("Card", emptyMap(), listOf("f")),
        "f" to text("Policy and HR Q&A: Use RAG so the assistant answers from the latest handbook, benefits policy, or leave rules and can cite the source document directly."),
        "g" to FlatElement("Card", emptyMap(), listOf("h")),
        "h" to text("Ad hoc document review: Use long-context prompting for a one-time analysis of a small set of contracts, meeting notes, or incident reports when you want the model to read everything in a single pass."),
        "i" to FlatElement("Card", emptyMap(), listOf("j")),
        "j" to text("Brand-safe support replies: Use fine-tuning when you need a consistent tone, fixed response structure, or product-specific phrasing across many repetitive tickets."),
        "k" to FlatElement("Card", emptyMap(), listOf("l")),
        "l" to FlatElement("Stack", mapOf("direction" to "vertical", "gap" to "sm"), listOf("m", "n")),
        "m" to text("Practical choice", "h2"),
        "n" to text("For most company knowledge assistants, start with RAG, because it balances freshness, traceability, and update speed better than the other two approaches. Add fine-tuning only when you have a stable task where behavior matters more than live knowledge, and use long-context prompting mainly when the document set is small enough that loading it all is still economical."),
    )

    private fun plan(elements: Map<String, FlatElement>, direction: String = "vertical", repeated: Boolean = false,
        retained: Set<String> = elements.getValue("c").children.toSet()) =
        filterAdjacentBareTextDuplicates("c", elements.getValue("c").children, elements, direction, repeated, retained)

    @Test fun `accepted use-case graph omits only the adjacent bare reference and preserves the original card text and other nodes`() {
        val elements = fixture()
        val snapshot = elements.toMap()
        assertEquals(listOf("d", "e", "g"), plan(elements))
        assertEquals(listOf("f"), elements.getValue("e").children)
        assertEquals(snapshot.getValue("f").props, elements.getValue("f").props)
        assertEquals(snapshot.getValue("b"), elements.getValue("b"))
        assertEquals(snapshot.getValue("i"), elements.getValue("i"))
        assertEquals(snapshot.getValue("k"), elements.getValue("k"))
        assertEquals(elements.getValue("root").children,
            filterAdjacentBareTextDuplicates("root", elements.getValue("root").children, elements, "vertical", false))
        assertEquals(snapshot, elements)
        val column = elements + ("c" to elements.getValue("c").copy(type = "Column", props = mapOf("gap" to "sm")))
        assertEquals(listOf("d", "e", "g"), plan(column))
    }

    @Test fun `reverse separated nested shared-card and equal-text different-ID paths stay intact`() {
        val original = fixture()
        val variants = listOf(
            original + ("c" to original.getValue("c").copy(children = listOf("d", "f", "e", "g"))),
            original + mapOf("separator" to FlatElement("Divider", emptyMap(), emptyList()),
                "c" to original.getValue("c").copy(children = listOf("d", "e", "separator", "f", "g"))),
            original + mapOf("clone" to original.getValue("f"),
                "c" to original.getValue("c").copy(children = listOf("d", "e", "clone", "g"))),
            original + mapOf("anotherCard" to FlatElement("Card", emptyMap(), listOf("f")),
                "c" to original.getValue("c").copy(children = listOf("d", "e", "anotherCard", "g"))),
            original + mapOf("nested" to FlatElement("Stack", mapOf("direction" to "vertical"), listOf("f")),
                "e" to original.getValue("e").copy(children = listOf("nested"))),
            original + ("c" to original.getValue("c").copy(children = listOf("d", "e", "f", "f", "g"))),
        )
        variants.forEach { elements -> assertEquals(elements.getValue("c").children, plan(elements)) }
    }

    @Test fun `custom interactive conditional repeated watched and bound nodes retain every original reference`() {
        val original = fixture()
        val guarded = listOf(
            "c" to original.getValue("c").copy(on = mapOf("click" to "parentAction")),
            "c" to original.getValue("c").copy(props = mapOf("direction" to "vertical", "align" to "center")),
            "c" to original.getValue("c").copy(props = mapOf("padding" to mapOf("path" to "/padding"))),
            "e" to original.getValue("e").copy(props = mapOf("tone" to "primary")),
            "e" to original.getValue("e").copy(props = mapOf("title" to "Original card title")),
            "e" to original.getValue("e").copy(on = mapOf("click" to "cardAction")),
            "e" to original.getValue("e").copy(visible = mapOf("path" to "/show")),
            "e" to original.getValue("e").copy(watch = mapOf("/value" to "cardWatch")),
            "e" to original.getValue("e").copy(repeat = RepeatConfig("/cards")),
            "f" to original.getValue("f").copy(on = mapOf("click" to "textAction")),
            "f" to original.getValue("f").copy(visible = mapOf("path" to "/showText")),
            "f" to original.getValue("f").copy(watch = mapOf("/value" to "textWatch")),
            "f" to original.getValue("f").copy(repeat = RepeatConfig("/paragraphs")),
            "f" to original.getValue("f").copy(props = mapOf("text" to mapOf("path" to "/paragraph"))),
            "f" to text("Original {{ /paragraph }} binding"),
            "f" to text("Shared heading", "h2"),
        )
        guarded.forEach { (id, node) ->
            val elements = original + (id to node)
            val snapshot = elements.toMap()
            assertEquals(original.getValue("c").children, plan(elements))
            assertEquals(snapshot, elements)
        }
    }

    @Test fun `removed wrapper horizontal or repeated context and missing references keep the bare content`() {
        val elements = fixture()
        val children = elements.getValue("c").children
        assertEquals(children, plan(elements, retained = children.toSet() - "e"))
        assertEquals(children, plan(elements, direction = "horizontal"))
        assertEquals(children, plan(elements, repeated = true))
        assertEquals(children, plan(elements - "e"))
        assertEquals(children, plan(elements - "f"))
        assertEquals(children, filterAdjacentBareTextDuplicates("missing", children, elements, "vertical", false))
    }
}
