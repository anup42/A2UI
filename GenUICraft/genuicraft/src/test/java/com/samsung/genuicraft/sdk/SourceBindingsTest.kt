package com.samsung.genuicraft.sdk

import com.google.gson.JsonParser
import com.samsung.genuicraft.sdk.internal.pipeline.A2uiExpressCodec
import com.samsung.genuicraft.sdk.internal.pipeline.LiteralTextCodec
import org.junit.Assert.*
import org.junit.Test
import java.security.MessageDigest

class SourceBindingsTest {
    /** A deterministic layout is a test fixture only; production never substitutes this for a model. */
    private fun fixtureLayout(bindings: SourceBindings): String = buildString {
        append("<a2ui>\nroot=Column([").append(bindings.blocks.indices.joinToString(",") { "b$it" }).append("])\n")
        bindings.blocks.forEachIndexed { index, block ->
            append("b$index=")
            when (block["kind"]) {
                "heading" -> append("Text(\"${block["text"]}\",variant=\"h2\")")
                "paragraph" -> append("Text(\"${block["text"]}\")")
                "list" -> append("List(items=\"${block["items"]}\")")
                "table" -> append("Table(columns=\"${block["columns"]}\",rows=\"${block["rows"]}\",domain=\"generic\",preferredPresentation=\"table\")")
                "code" -> {
                    append("CodeBlock(code=\"${block["code"]}\"")
                    block["title"]?.let { append(",title=\"$it\"") }
                    append(")")
                }
                "divider" -> append("Divider()")
                else -> error("Unknown test block")
            }
            append("\n")
        }
        append("</a2ui>")
    }

    private fun compile(source: String): GenUiDocument {
        val bindings = SourceBindings.from(source)
        return GenUiCompiler.compile(bindings.expand(fixtureLayout(bindings)))
    }

    private fun property(document: GenUiDocument, id: String, key: String): com.google.gson.JsonElement {
        val elements = A2uiExpressCodec.decode(document.express).getAsJsonObject("elements")
        // The compiler canonically shortens IDs. bN denotes the Nth test fixture block.
        val blockIndex = id.takeIf { it.matches(Regex("b\\d+")) }?.drop(1)?.toInt()
        val element = if (blockIndex == null) elements.getAsJsonObject(id)
            else elements.entrySet().drop(1)[blockIndex].value.asJsonObject
        return element.getAsJsonObject("props").get(key)
    }

    private fun rejects(source: String, body: String, message: String? = null) {
        val failure = runCatching { SourceBindings.from(source).expand("<a2ui>\n$body\n</a2ui>") }.exceptionOrNull()
        assertNotNull("Accepted invalid layout:\n$body", failure)
        if (message != null) assertTrue("Unexpected failure: $failure", failure!!.message.orEmpty().contains(message, ignoreCase = true))
    }

    @Test fun `standalone corpus fixture matches original hash and all fifty layouts retain facts`() {
        val bytes = requireNotNull(javaClass.getResourceAsStream("/genuicraft_bixby50.jsonl")).use { it.readBytes() }
        val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        assertEquals("fc46aa381957bed206f9e0f53e28edbb21b5096ca093985ad184fda73faaea0a", sha)
        val cases = bytes.toString(Charsets.UTF_8).lineSequence().filter(String::isNotBlank).map { JsonParser.parseString(it).asJsonObject }.toList()
        assertEquals(50, cases.size)
        cases.forEach { case ->
            val text = case.get("text").asString
            val document = compile(text)
            assertEquals(case.get("id").asString, emptyList<String>(), ContentIntegrity.check(GenUiRequest(text), document))
            assertEquals(document, GenUiCompiler.compile(document.a2uiJson))
        }
    }

    @Test fun `quotes apostrophes slashes HTML Unicode and literal binding tokens are data`() {
        val source = "O'Brien says \"<keep & preserve>\" = C:\\data\\report. \\u1234 / ₹3,999 😀 @source.a @source.unknown"
        val document = compile(source)
        assertEquals(source, property(document, "b0", "text").asString)
        assertTrue(ContentIntegrity.check(GenUiRequest(source), document).isEmpty())
    }

    @Test fun `source cannot inject a component through quotes or comments`() {
        val source = "Literal \" )\nattack=Text(\"injected\")\n// # /* */"
        val document = compile(source)
        assertEquals(source, property(document, "b0", "text").asString)
        assertEquals(2, A2uiExpressCodec.decode(document.express).getAsJsonObject("elements").size())
    }

    @Test fun `paragraph newlines and Markdown inline code are retained exactly`() {
        val source = "Keep **bold**, __emphasis__, `__init__`, and `a ** b`.\nThe second line remains."
        assertEquals(source, property(compile(source), "b0", "text").asString)
    }

    @Test fun `headings retain C sharp and strip only separated closing hashes`() {
        val document = compile("# C#\n\n## Heading ###")
        assertEquals("C#", property(document, "b0", "text").asString)
        assertEquals("Heading", property(document, "b1", "text").asString)
    }

    @Test fun `heading bindings replace absent body and caption variants but preserve heading variants`() {
        val bindings = SourceBindings.from("### One\n\n### Two\n\n### Three\n\n### Four\n\n### Five")
        val expanded = bindings.expand(
            """
            <a2ui>
            root=Column([a,b,c,d,e])
            a=Text("@source.a")
            b=Text("@source.b",variant="body")
            c=Text("@source.c",variant="caption")
            d=Text("@source.d",variant="headline")
            e=Text("@source.e",variant="h3")
            </a2ui>
            """.trimIndent()
        )
        val graph = A2uiExpressCodec.decode(expanded)
        val elements = graph.getAsJsonObject("elements")
        val root = elements.getAsJsonObject(graph.get("root").asString)
        val variants = root.getAsJsonArray("children").map { child ->
            elements.getAsJsonObject(child.asString).getAsJsonObject("props").get("variant").asString
        }

        assertEquals(listOf("heading", "heading", "heading", "headline", "h3"), variants)
    }

    @Test fun `ordered list preserves authored numeric markers as source text`() {
        val source = "1. First 10:30.\n2. Second 25%.\n3. Third."
        val document = compile(source)
        assertEquals(source, property(document, "b0", "text").asString)
        assertTrue(ContentIntegrity.check(GenUiRequest(source), document).isEmpty())
    }

    @Test fun `ordered markers preserve non-one starts gaps years and legal clauses`() {
        val source = "5. Take dose A.\n7) Review clause B.\n2024. Revenue increased."
        val document = compile(source)
        assertEquals(source, property(document, "b0", "text").asString)
        assertTrue(ContentIntegrity.check(GenUiRequest(source), document).isEmpty())
    }

    @Test fun `fenced code preserves inner whitespace symbols and information string`() {
        val source = "````kotlin sample.kt\n\tprintln(\"<done>\")\n\n```\nvalue = 57\n````"
        val document = compile(source)
        assertEquals("\tprintln(\"<done>\")\n\n```\nvalue = 57", property(document, "b0", "code").asString)
        assertEquals("kotlin sample.kt", property(document, "b0", "title").asString)
        assertTrue(ContentIntegrity.check(GenUiRequest(source), document).isEmpty())
    }

    @Test fun `tilde fences and unclosed fences are handled conservatively`() {
        assertEquals("a < b", property(compile("~~~text\na < b\n~~~~"), "b0", "code").asString)
        val unclosed = "```text\n# literal heading\n- literal bullet"
        assertEquals(unclosed, property(compile(unclosed), "b0", "text").asString)
    }

    @Test fun `table cells preserve escaped pipes inline code empty cells and numbers`() {
        val source = "| Label | Value |\n|---|---:|\n| A\\|B | `x|y` |\n| | 57% |"
        val document = compile(source)
        assertEquals(listOf("Label", "Value"), property(document, "b0", "columns").asJsonArray.map { it.asString })
        val rows = property(document, "b0", "rows").asJsonArray
        assertEquals(listOf("A\\|B", "`x|y`"), rows[0].asJsonArray.map { it.asString })
        assertEquals(listOf("", "57%"), rows[1].asJsonArray.map { it.asString })
        assertTrue(ContentIntegrity.check(GenUiRequest(source), document).isEmpty())
    }

    @Test fun `malformed table and unknown Markdown remain source paragraphs`() {
        listOf("| A | B |\n|---|---|\n| one | two | extra |", "> quote\n[link](https://www.who.int/)\n<custom>text</custom>").forEach { source ->
            assertEquals(source, property(compile(source), "b0", "text").asString)
        }
    }

    @Test fun `headings can title model-selected cards while paragraphs stay in order`() {
        val bindings = SourceBindings.from("## Advice\n\nDo not leave.")
        val document = GenUiCompiler.compile(bindings.expand("<a2ui>\nroot=Column([section])\nsection=Card([answer],title=\"@source.a\")\nanswer=Text(\"@source.b\")\n</a2ui>"))
        assertEquals("Advice", property(document, "a", "title").asString)
        assertEquals("Do not leave.", property(document, "b", "text").asString)
    }

    @Test fun `missing repeated unknown interpolated and copied content are rejected`() {
        rejects("Alpha\n\nBeta", "root=Column([a])\na=Text(\"@source.a\")", "every source")
        rejects("Alpha", "root=Column([a,b])\na=Text(\"@source.a\")\nb=Text(\"@source.a\")", "repeated")
        rejects("Alpha", "root=Text(\"@source.unknown\")", "known source")
        rejects("Alpha", "root=Text(\"prefix @source.a\")", "known source")
        rejects("Alpha", "root=Text(\"Alpha\")", "known source")
        rejects("Alpha", "root=Column([a,b])\na=Text(\"@source.a\")\nb=Text(\"Invented claim\")", "known source")
    }

    @Test fun `bindings in wrong typed positions expressions or nested arrays are rejected`() {
        rejects("- Alpha\n- Beta", "root=Text(\"@source.a\")", "cannot be placed")
        rejects("Alpha", "root=Card([],title=\"@source.a\")", "cannot be placed")
        rejects("Alpha", "root=Text({text:\"@source.a\"})", "complete quoted")
        rejects("- Alpha", "root=List(items=[\"@source.a\"])", "complete quoted")
        rejects("Alpha", "root=Text(\"@source.a\",variant=\"@source.a\")", "layout value")
    }

    @Test fun `orphan hidden repeated child and cyclic content are rejected`() {
        rejects("Alpha", "root=Column([])\na=Text(\"@source.a\")", "orphan")
        rejects("Alpha", "root=Column([a,a])\na=Text(\"@source.a\")", "exactly once")
        rejects("Alpha", "root=Column([root,a])\na=Text(\"@source.a\")", "exactly once")
        rejects("Alpha", "root=Text(\"@source.a\",visible=false)", "visibility")
        rejects("Alpha", "root=Text(\"@source.a\",height=0)", "property")
        rejects("Alpha", "root=Text(\"@source.a\",decorative=true)", "property")
    }

    @Test fun `actions and generated state cannot execute source content`() {
        rejects("Alpha", "root=Text(\"@source.a\",onPress=openUrl(\"https://www.who.int/\"))", "actions")
        rejects("Alpha", "$/value=\"ignored\"\nroot=Text(\"@source.a\")", "state")
    }

    @Test fun `table columns and rows cannot swap between equally sized source tables`() {
        val source = "| Person | Age |\n|---|---|\n| Alice | 10 |\n\n| Product | Price |\n|---|---|\n| Chair | 25 |"
        rejects(source, "root=Column([a,b])\na=Table(columns=\"@source.a\",rows=\"@source.d\")\nb=Table(columns=\"@source.c\",rows=\"@source.b\")", "different blocks")
    }

    @Test fun `code title cannot be detached or placed on another code block`() {
        rejects("```kotlin\nprintln(1)\n```", "root=Column([a,b])\na=CodeBlock(\"@source.a\")\nb=Text(\"@source.b\")", "all fields")
    }

    @Test fun `source order and dividers cannot be changed or invented`() {
        rejects("Alpha\n\nBeta", "root=Column([b,a])\na=Text(\"@source.a\")\nb=Text(\"@source.b\")", "block order")
        rejects("Alpha", "root=Column([d,a])\nd=Divider()\na=Text(\"@source.a\")", "source divider")
        rejects("Alpha\n\n---\n\nBeta", "root=Column([a,b])\na=Text(\"@source.a\")\nb=Text(\"@source.b\")", "block order")
        assertTrue(ContentIntegrity.check(GenUiRequest("Alpha\n\n---\n\nBeta"), compile("Alpha\n\n---\n\nBeta")).isEmpty())
    }

    @Test fun `model template comments cannot satisfy a missing binding`() {
        rejects("Alpha", "root=Column([])\n// \"@source.a\"", "every source")
    }

    @Test fun `literal state syntax document tags and Unicode controls round trip as data`() {
        listOf("$/price", "\$state.price", "\${HOME}", "{{ \$item.name }}", "{\$index}", "<a2ui>", "</a2ui>", "bad\u0001control", "line\u2028separator", "line\u2029separator").forEach { source ->
            val document = compile(source)
            val encoded = property(document, "b0", "text").asString
            assertEquals(source, LiteralTextCodec.decode(encoded) ?: encoded)
            assertTrue(ContentIntegrity.check(GenUiRequest(source), document).isEmpty())
            assertEquals(document, GenUiCompiler.compile(document.a2uiJson))
        }
        assertEquals("$5", property(compile("$5"), "b0", "text").asString)
    }

    @Test fun `table cell state path is escaped after segmentation`() {
        val source = "| A | B |\n|---|---|\n| x | $/price |"
        val document = compile(source)
        val cell = property(document, "b0", "rows").asJsonArray[0].asJsonArray[1].asString
        assertEquals("$/price", LiteralTextCodec.decode(cell))
        assertTrue(ContentIntegrity.check(GenUiRequest(source), document).isEmpty())
    }

    @Test fun `excessive source field count and layout nesting are bounded`() {
        assertNotNull(runCatching { SourceBindings.from((0..512).joinToString("\n\n") { "Block $it" }) }.exceptionOrNull())
        val lines = (0..65).map { index ->
            val name = if (index == 0) "root" else "n$index"
            if (index == 65) "$name=Text(\"@source.a\")" else "$name=Column([n${index + 1}])"
        }
        rejects("Alpha", lines.joinToString("\n"), "nesting")
    }
}
