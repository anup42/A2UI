package com.samsung.genuicraft.sdk.internal.pipeline

import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import com.samsung.genuicraft.sdk.GenUiCompiler
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.FlatExprResolver
import com.samsung.genuicraft.sdk.internal.renderer.flat.expr.FlatLiteralText
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.FlatSpecParser
import com.samsung.genuicraft.sdk.internal.renderer.flat.parse.parseFlatListItems
import com.samsung.genuicraft.sdk.internal.renderer.parseDirectTableColumns
import com.samsung.genuicraft.sdk.internal.renderer.tableCellDisplayText
import org.junit.Assert.*
import org.junit.Test

class LiteralTextCodecTest {
    private val literals = listOf("\${HOME}", "$/price", "\$state.foo", "$", "{{ \$item.name }}", "{\$index}")

    @Test fun `plain strings preserve their exact value and reserved prefixes escape once`() {
        listOf("Safe text 57%", "C:\\data\\report", "Price $5", "<a2ui>", "emoji 😀").forEach { source ->
            assertEquals(source, LiteralTextCodec.encode(source))
            assertNull(LiteralTextCodec.decode(source))
        }
        literals.forEach { source -> assertEquals(source, LiteralTextCodec.decode(LiteralTextCodec.encode(source))) }
        val alreadyPrefixed = LiteralTextCodec.encode("\${HOME}")
        assertEquals(alreadyPrefixed, LiteralTextCodec.decode(LiteralTextCodec.encode(alreadyPrefixed)))
    }

    @Test fun `wire uses ordinary strings without a legacy literal object or dynamic path`() {
        literals.forEach { source ->
            val program = "<a2ui>\nroot=Text(${JsonPrimitive(source)})\n</a2ui>"
            val document = GenUiCompiler.compile(LiteralTextCodec.protectVisibleContent(program))
            val wire = JsonParser.parseString(document.a2uiJson).asJsonArray
            val component = wire[1].asJsonObject.getAsJsonObject("updateComponents").getAsJsonArray("components")[0].asJsonObject
            assertTrue(component.get("text").isJsonPrimitive)
            assertEquals(source, LiteralTextCodec.decode(component.get("text").asString))
            assertFalse(document.a2uiJson.contains("literalString"))
            assertEquals(document, GenUiCompiler.compile(document.a2uiJson))
        }
    }

    @Test fun `repeated resolution never evaluates escaped text including collision payloads`() {
        val state = mapOf<String, Any?>("HOME" to "SECRET", "price" to 42L)
        (literals + LiteralTextCodec.encode("\${HOME}")).forEach { source ->
            var resolved: Any? = LiteralTextCodec.encode(source)
            repeat(3) { resolved = FlatExprResolver.resolve(resolved, state, null) }
            assertTrue(resolved is FlatLiteralText)
            assertEquals(source, resolved.toString())
        }
        // Existing ordinary dynamic and legacy-literal semantics remain unchanged.
        assertEquals("SECRET", FlatExprResolver.resolve("\${HOME}", state, null))
        assertEquals("Ada", FlatExprResolver.resolve(mapOf("literalString" to "Ada"), state, null))
        assertEquals(42L, FlatExprResolver.resolve(mapOf("\$state" to "price"), state, null))
    }

    @Test fun `list items table headings and repeatedly resolved table cells display original text`() {
        val source = "\${HOME}"
        val resolved = FlatExprResolver.resolve(LiteralTextCodec.encode(source), emptyMap(), null)
        assertEquals(source, parseFlatListItems(listOf(resolved)).single().text.single())
        val structured = parseFlatListItems(listOf(mapOf("text" to resolved, "url" to "https://www.who.int/"))).single()
        assertEquals(source, structured.text.single())
        assertEquals("https://www.who.int/", structured.links.single())
        assertEquals(source, parseDirectTableColumns(listOf(resolved)).single().label)
        assertEquals(source, tableCellDisplayText(FlatExprResolver.resolve(resolved, emptyMap(), null)))
    }

    @Test fun `protection changes only visible literal content and leaves actions layout and references intact`() {
        val program = """
            <a2ui>
            root=Column([a,b],gap="md")
            a=Text("${'$'}{HOME}",variant="body")
            b=Button("Open",onPress=openUrl("https://www.who.int/${'$'}{HOME}"))
            </a2ui>
        """.trimIndent()
        val graph = A2uiExpressCodec.decode(LiteralTextCodec.protectVisibleContent(program))
        val elements = graph.getAsJsonObject("elements")
        assertEquals("md", elements.getAsJsonObject("root").getAsJsonObject("props").get("gap").asString)
        assertEquals(listOf("a", "b"), elements.getAsJsonObject("root").getAsJsonArray("children").map { it.asString })
        assertEquals("\${HOME}", LiteralTextCodec.decode(elements.getAsJsonObject("a").getAsJsonObject("props").get("text").asString))
        assertEquals("body", elements.getAsJsonObject("a").getAsJsonObject("props").get("variant").asString)
        assertEquals("https://www.who.int/\${HOME}", elements.getAsJsonObject("b").getAsJsonObject("on").getAsJsonObject("press").getAsJsonObject("params").get("url").asString)
        val safe = "<a2ui>\nroot=Text(\"Safe ordinary content\")\n</a2ui>"
        assertEquals(safe, LiteralTextCodec.protectVisibleContent(safe))
    }

    @Test fun `explicit dynamic path objects remain expressions and list URLs remain callback targets`() {
        val program = """
            <a2ui>
            root=Column([a,b])
            a=Text({path:"/price"})
            b=List(items=[{text:"${'$'}{HOME}",url:"https://www.who.int/${'$'}{HOME}"}])
            </a2ui>
        """.trimIndent()
        val graph = A2uiExpressCodec.decode(LiteralTextCodec.protectVisibleContent(program))
        val elements = graph.getAsJsonObject("elements")
        assertEquals("/price", elements.getAsJsonObject("a").getAsJsonObject("props").getAsJsonObject("text").get("path").asString)
        val item = elements.getAsJsonObject("b").getAsJsonObject("props").getAsJsonArray("items")[0].asJsonObject
        assertEquals("\${HOME}", LiteralTextCodec.decode(item.get("text").asString))
        assertEquals("https://www.who.int/\${HOME}", item.get("url").asString)
    }

    @Test fun `List source fields remain links while Alert source fields are protected display text`() {
        val program = """
            <a2ui>
            root=Column([a,b])
            a=List(items=[{text:"${'$'}{HOME}",source:"https://www.who.int/${'$'}{HOME}"}])
            b=Alert("Ready",source="${'$'}{HOME}")
            </a2ui>
        """.trimIndent()
        val document = GenUiCompiler.compile(LiteralTextCodec.protectVisibleContent(program))
        val elements = A2uiExpressCodec.decode(document.express).getAsJsonObject("elements")
        val item = elements.getAsJsonObject("a").getAsJsonObject("props").getAsJsonArray("items")[0].asJsonObject
        assertEquals("https://www.who.int/\${HOME}", item.get("source").asString)
        assertNull(LiteralTextCodec.decode(item.get("source").asString))
        assertEquals("\${HOME}", LiteralTextCodec.decode(item.get("text").asString))
        val alertSource = elements.getAsJsonObject("b").getAsJsonObject("props").get("source").asString
        assertEquals("\${HOME}", LiteralTextCodec.decode(alertSource))
        assertEquals("\${HOME}", FlatExprResolver.resolve(alertSource, mapOf("HOME" to "DO NOT DISPLAY"), null).toString())
    }

    @Test fun `Unicode JSON escapes and literal document delimiters round trip inside strings`() {
        val values = listOf("<a2ui>quoted </a2ui>", "controls \u0000\u0001\u001e", "lines\u2028and\u2029", "emoji 😀", "literal \\u0041")
        values.forEach { value ->
            val document = GenUiCompiler.compile("<a2ui>\nroot=Text(${JsonPrimitive(value)})\n</a2ui>")
            val graph = A2uiExpressCodec.decode(document.express)
            assertEquals(value, graph.getAsJsonObject("elements").getAsJsonObject("root").getAsJsonObject("props").get("text").asString)
            assertEquals(document, GenUiCompiler.compile(document.a2uiJson))
        }
        val escaped = A2uiExpressCodec.decode("<a2ui>\nroot=Text(\"\\u0041\\uD83D\\uDE00\")\n</a2ui>")
        assertEquals("A😀", escaped.getAsJsonObject("elements").getAsJsonObject("root").getAsJsonObject("props").get("text").asString)
    }

    @Test fun `actual duplicate document blocks and invalid Unicode escapes still fail`() {
        assertThrows(IllegalArgumentException::class.java) {
            GenUiCompiler.compile("<a2ui>root=Text(\"one\")</a2ui><a2ui>root=Text(\"two\")</a2ui>")
        }
        listOf("\\u0X41", "\\u123").forEach { escaped ->
            assertThrows(IllegalArgumentException::class.java) { GenUiCompiler.compile("<a2ui>\nroot=Text(\"$escaped\")\n</a2ui>") }
        }
    }

    @Test fun `wire renderer parsing retains literal safety through both resolution passes`() {
        val program = LiteralTextCodec.protectVisibleContent("<a2ui>\nroot=Text(\"\${HOME}\")\n</a2ui>")
        val document = GenUiCompiler.compile(program)
        val decoded = GenUiIrCodec.decode(JsonParser.parseString(document.a2uiJson))
        val spec = requireNotNull(FlatSpecParser.parse(decoded.canonicalGraph))
        val raw = spec.elements.getValue("root").props.getValue("text")
        val once = FlatExprResolver.resolve(raw, mapOf("HOME" to "SECRET"), null)
        assertEquals("\${HOME}", FlatExprResolver.resolve(once, mapOf("HOME" to "SECRET"), null).toString())
    }
}
