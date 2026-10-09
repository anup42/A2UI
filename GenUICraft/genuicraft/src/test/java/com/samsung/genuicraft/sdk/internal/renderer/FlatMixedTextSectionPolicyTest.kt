package com.samsung.genuicraft.sdk.internal.renderer

import com.google.gson.Gson
import com.google.gson.JsonParser
import com.google.gson.reflect.TypeToken
import com.samsung.genuicraft.sdk.internal.renderer.flat.domain.planFlatTextSections
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.FlatElement
import com.samsung.genuicraft.sdk.internal.renderer.flat.model.RepeatConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Test
import java.security.MessageDigest

class FlatMixedTextSectionPolicyTest {
    private val fixtureBytes = javaClass.getResourceAsStream("/renderer/bxp032_article.json")!!.use { it.readBytes() }

    private fun fixture(): Map<String, FlatElement> {
        val components = JsonParser.parseString(fixtureBytes.toString(Charsets.UTF_8)).asJsonArray
            .mapNotNull { it.asJsonObject.getAsJsonObject("updateComponents") }
            .single().getAsJsonArray("components")
        val mapType = object : TypeToken<Map<String, Any?>>() {}.type
        return components.associate { component ->
            val raw: Map<String, Any?> = Gson().fromJson(component, mapType)
            raw.getValue("id").toString() to FlatElement(
                raw.getValue("component").toString(),
                raw - setOf("id", "component", "children"),
                (raw["children"] as? List<*>)?.map { it.toString() }.orEmpty(),
            )
        }
    }

    private fun plan(elements: Map<String, FlatElement>, isRoot: Boolean = true,
        direction: String = "vertical", repeated: Boolean = false) =
        planFlatTextSections("root", elements.getValue("root").children, elements, isRoot, direction, repeated)

    @Test fun `exact native article keeps table and heading-only siblings while grouping complete explanations and examples`() {
        assertEquals("8d9cb97036dde6dc1b4837e14f7a3175dc0bf99957d797b6a5d1d64c345c239d",
            MessageDigest.getInstance("SHA-256").digest(fixtureBytes).joinToString("") { "%02x".format(it) })
        val elements = fixture()
        val snapshot = elements.toMap()
        val sections = plan(elements)!!
        assertEquals(listOf(
            null to listOf("a"), null to listOf("c"), "d" to listOf("d", "e"),
            null to listOf("f"), "g" to listOf("g", "h"), "i" to listOf("i", "j"),
            "k" to listOf("k", "l", "m"),
        ), sections.map { it.headingId to it.childIds })
        assertEquals(elements.getValue("root").children, sections.flatMap { it.childIds })
        assertEquals(snapshot, elements)
        snapshot.forEach { (id, original) -> assertSame(original, elements.getValue(id)) }
        val rows = elements.getValue("b").props.getValue("rows") as List<*>
        assertEquals(15, rows.sumOf { (it as List<*>).size })
        assertEquals("table", elements.getValue("b").props["preferredPresentation"])
        assertEquals(snapshot.getValue("h").props["text"], elements.getValue("h").props["text"])
        assertEquals(snapshot.getValue("j").props["text"], elements.getValue("j").props["text"])
    }

    @Test fun `table boundaries stay in place and pending headings keep stable IDs as literal body arrives`() {
        val elements = fixture().toMutableMap()
        elements["root"] = elements.getValue("root").copy(children = listOf("d", "a", "e", "g", "h", "i", "j"))
        val split = plan(elements)!!
        assertEquals(listOf(null, null, null, "g", "i"), split.map { it.headingId })
        assertEquals(elements.getValue("root").children, split.flatMap { it.childIds })
        elements["root"] = elements.getValue("root").copy(children = listOf("b", "d", "e", "g", "h", "i"))
        val first = plan(elements)!!
        assertEquals(listOf("b", "d", "g", "i"), first.map { it.stableKey })
        assertNull(first.last().headingId)
        elements["root"] = elements.getValue("root").copy(children = elements.getValue("root").children + "j")
        val next = plan(elements)!!
        assertEquals(first.map { it.stableKey }, next.map { it.stableKey })
        assertEquals(listOf("i", "j"), next.last().childIds)
        assertEquals(elements.getValue("root").children, next.flatMap { it.childIds })
        elements["root"] = elements.getValue("root").copy(children = listOf("b", "d", "e", "g"))
        assertNull(plan(elements))
    }

    @Test fun `actions bindings conditional repeat and custom text do not change existing behavior`() {
        val original = fixture()
        val body = original.getValue("h")
        listOf(
            body.copy(on = mapOf("click" to "originalAction")),
            body.copy(watch = mapOf("/claim" to "originalWatch")),
            body.copy(visible = mapOf("path" to "/show")),
            body.copy(repeat = RepeatConfig("/claims")),
            body.copy(props = body.props + ("text" to "Claim {{/amount}}")),
            body.copy(props = body.props + ("text" to mapOf("path" to "/claim"))),
            body.copy(props = body.props + ("color" to "#123456")),
        ).forEach { changed ->
            val elements = original + ("h" to changed)
            assertNull(plan(elements))
            assertSame(changed, elements.getValue("h"))
        }
        val table = original.getValue("b")
        listOf(
            table.copy(on = mapOf("click" to "originalAction")),
            table.copy(visible = mapOf("path" to "/show")),
            table.copy(repeat = RepeatConfig("/claims")),
        ).forEach { changed -> assertNull(plan(original + ("b" to changed))) }
        val allText = original + ("root" to original.getValue("root").copy(children = listOf("d", "e", "g", "h")))
        assertNull(plan(allText + ("h" to body.copy(props = body.props + ("text" to "Claim {{/amount}}")))))
    }

    @Test fun `unsafe containers unresolved shared IDs and non-root or repeated graphs keep their existing route`() {
        val original = fixture()
        val root = original.getValue("root")
        assertNull(plan(original, isRoot = false))
        assertNull(plan(original, direction = "horizontal"))
        assertNull(plan(original, repeated = true))
        assertNull(plan(original - "b"))
        assertNull(plan(original + ("root" to root.copy(children = root.children + "d"))))
        assertNull(plan(original + ("root" to root.copy(children = root.children + "b"))))
        assertNull(plan(original + ("root" to root.copy(props = root.props + ("justify" to "between")))))
        assertNull(plan(original + ("root" to root.copy(on = mapOf("click" to "originalAction")))))
        val wrapper = original.getValue("a")
        listOf(
            wrapper.copy(type = "Card"),
            wrapper.copy(props = wrapper.props + ("title" to "Authored wrapper title")),
            wrapper.copy(props = wrapper.props + ("direction" to "horizontal")),
            wrapper.copy(children = listOf("a")),
            wrapper.copy(children = listOf("b", "c")),
        ).forEach { changed -> assertNull(plan(original + ("a" to changed))) }
        assertEquals(listOf("b"), original.getValue("a").children)
        assertEquals(root, original.getValue("root"))
    }
}
