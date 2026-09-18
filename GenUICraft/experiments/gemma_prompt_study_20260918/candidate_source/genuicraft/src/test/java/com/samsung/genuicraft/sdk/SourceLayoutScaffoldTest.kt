package com.samsung.genuicraft.sdk

import com.google.gson.JsonParser
import org.junit.Assert.*
import org.junit.Test
import kotlinx.coroutines.runBlocking

class SourceLayoutScaffoldTest {
    private fun scaffold(source: String): Pair<SourceBindings, String> {
        val bindings = SourceBindings.from(source)
        val ids = bindings.blocks.indices.map { i -> i.toString(26).map { 'a' + it.digitToInt(26) }.joinToString("") }
        return bindings to SourceLayoutScaffold.create("root=Column([${ids.joinToString(",")}],gap=\"md\")", ids, bindings.blocks)
    }

    @Test fun `scaffold matches the device prototype byte for byte across the corpus`() {
        val cases = requireNotNull(javaClass.getResourceAsStream("/genuicraft_bixby50.jsonl"))
            .bufferedReader().useLines { lines -> lines.filter(String::isNotBlank).map { JsonParser.parseString(it).asJsonObject.get("text").asString }.toList() }
        assertEquals(50, cases.size)
        cases.forEach { source ->
            val (bindings, actual) = scaffold(source)
            val ids = bindings.blocks.indices.map { i -> i.toString(26).map { 'a' + it.digitToInt(26) }.joinToString("") }
            val data = JsonParser.parseString(com.google.gson.Gson().toJson(mapOf(
                "root" to "root=Column([${ids.joinToString(",")}],gap=\"md\")",
                "blocks" to bindings.blocks.mapIndexed { i, block -> mapOf("element" to ids[i]) + block },
            ))).asJsonObject
            // Frozen independent JSON-based prototype from the device experiment.
            val expected = buildString {
                appendLine("<a2ui>"); appendLine(data.get("root").asString)
                data.getAsJsonArray("blocks").forEach { row ->
                    val block = row.asJsonObject
                    append(block.get("element").asString).append('=')
                    when (block.get("kind").asString) {
                        "heading" -> append("Text(").append(block.get("text")).append(",variant=\"heading\")")
                        "paragraph" -> append("Text(").append(block.get("text")).append(')')
                        "list" -> append("List(items=").append(block.get("items")).append(')')
                        "table" -> append("Table(columns=").append(block.get("columns")).append(",rows=").append(block.get("rows")).append(",domain=\"SELECT_DOMAIN\",preferredPresentation=\"SELECT_PRESENTATION\")")
                        "code" -> { append("CodeBlock(code=").append(block.get("code")); if (block.has("title")) append(",title=").append(block.get("title")); append(')') }
                        "divider" -> append("Divider()")
                    }
                    appendLine()
                }
                appendLine("</a2ui>")
            }
            assertEquals(expected, actual)
            val completedFixture = actual.replace("\"SELECT_DOMAIN\"", "\"generic\"").replace("\"SELECT_PRESENTATION\"", "\"table\"")
            val document = GenUiCompiler.compile(bindings.expand(completedFixture))
            assertTrue(ContentIntegrity.check(GenUiRequest(source), document).isEmpty())
        }
    }

    @Test fun `code title divider and literal source values remain bound rather than copied`() {
        val source = "# Example\n\n```kotlin\nprintln(\"\${HOME}\")\n```\n\n---\n\nA literal \\u003d and @source.a remain data."
        val (bindings, draft) = scaffold(source)
        assertTrue(draft.contains("b=CodeBlock(code=\"@source.b\",title=\"@source.c\")"))
        assertTrue(draft.contains("c=Divider()"))
        assertTrue(draft.contains("d=Text(\"@source.d\")"))
        assertFalse(draft.contains("println"))
        assertTrue(ContentIntegrity.check(GenUiRequest(source), GenUiCompiler.compile(bindings.expand(draft))).isEmpty())
    }

    @Test fun `the scaffold is last on both initial and repair prompts and never substitutes for a response`() = runBlocking {
        val source = "| Item | Value |\n|---|---|\n| A | 1 |"
        val draft = scaffold(source).second
        val prompts = mutableListOf<GenUiPrompt>()
        val provider = object : GenUiProvider {
            override val id = "fixture"
            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
                prompts += prompt
                return GenUiModelOutput("invalid model output", "fake; no inference")
            }
        }
        val result = GenUiConverter.withPrompt(provider, "contract", useSourceBindings = true, useLayoutScaffold = true).convert(GenUiRequest(source))
        assertTrue(result is GenUiConversionResult.Failure)
        assertEquals(2, prompts.size)
        prompts.forEach { assertTrue(it.user.endsWith("${SourceLayoutScaffold.HEADER}\n$draft")) }
        assertTrue(prompts[1].user.contains("Previous program:\ninvalid model output"))
        assertFalse(prompts[0].user.contains("Previous program:"))
    }

    @Test fun `custom prompts preserve the original input unless the scaffold is requested`() = runBlocking {
        val prompts = mutableListOf<GenUiPrompt>()
        val provider = object : GenUiProvider {
            override val id = "fixture"
            override suspend fun generate(prompt: GenUiPrompt): GenUiModelOutput {
                prompts += prompt
                return GenUiModelOutput("invalid model output", "fake; no inference")
            }
        }
        GenUiConverter.withPrompt(provider, "custom", useSourceBindings = true).convert(GenUiRequest("Example"))
        assertEquals(2, prompts.size)
        assertTrue(prompts.all { !it.user.contains(SourceLayoutScaffold.HEADER) })
        assertThrows(IllegalArgumentException::class.java) {
            GenUiConverter.withPrompt(provider, "custom", useLayoutScaffold = true)
        }
        Unit
    }
}
