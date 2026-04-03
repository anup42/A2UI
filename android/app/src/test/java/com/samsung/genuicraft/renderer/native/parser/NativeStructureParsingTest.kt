package com.samsung.genuicraft.renderer.native.parser

import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class NativeStructureParsingTest {

    @Test
    fun parseButtonLine_supportsActionPrefix() {
        val parsed = NativeStructureParsing.parseButtonLine(
            "Action: [Button: Compare Flights] https://example.com/flights",
            NativeStructureParsing::sanitizeUrlToken
        )

        assertEquals("Compare Flights", parsed?.label)
        assertEquals("https://example.com/flights", parsed?.url)
    }

    @Test
    fun parseButtonLine_supportsMarkdownLinkSyntax() {
        val parsed = NativeStructureParsing.parseButtonLine(
            "[Button: View Forecast](https://example.com/weather)",
            NativeStructureParsing::sanitizeUrlToken
        )

        assertEquals("View Forecast", parsed?.label)
        assertEquals("https://example.com/weather", parsed?.url)
    }

    @Test
    fun parseButtonLine_supportsBulletPrefixedButtons() {
        val parsed = NativeStructureParsing.parseButtonLine(
            "- [Button: Open Details] <https://example.com/details>",
            NativeStructureParsing::sanitizeUrlToken
        )

        assertEquals("Open Details", parsed?.label)
        assertEquals("https://example.com/details", parsed?.url)
    }

    @Test
    fun parseButtonLine_supportsActionBracketLabelWithoutButtonKeyword() {
        val parsed = NativeStructureParsing.parseButtonLine(
            "Action: [Search on IndiGo]: https://www.goindigo.in/",
            NativeStructureParsing::sanitizeUrlToken
        )

        assertEquals("Search on IndiGo", parsed?.label)
        assertEquals("https://www.goindigo.in/", parsed?.url)
    }

    @Test
    fun parseButtonLine_parsesAllButtonLinesInSubset10Scenarios() {
        val responses = loadSubset10Responses()
        val buttonLines = responses
            .flatMap { text ->
                text.lineSequence().map { it.trim() }
                    .filter { line -> line.contains("[Button:", ignoreCase = true) }
                    .toList()
            }

        assertTrue("Expected button lines in subset10 responses.", buttonLines.isNotEmpty())

        val failedLines = buttonLines.filter { line ->
            NativeStructureParsing.parseButtonLine(line, NativeStructureParsing::sanitizeUrlToken) == null
        }

        assertTrue(
            "Unparsed button lines:\n${failedLines.joinToString("\n")}",
            failedLines.isEmpty()
        )
    }

    private fun loadSubset10Responses(): List<String> {
        val file = resolveSubset10File()
        return file.readLines()
            .asSequence()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .take(10)
            .mapNotNull { line ->
                runCatching {
                    JsonParser.parseString(line)
                        .asJsonObject
                        .get("response_text")
                        ?.asString
                }.getOrNull()
            }
            .toList()
    }

    private fun resolveSubset10File(): File {
        val candidates = listOf(
            File("src/main/assets/ir_demo_subset10_responses.jsonl"),
            File("app/src/main/assets/ir_demo_subset10_responses.jsonl"),
            File("android/app/src/main/assets/ir_demo_subset10_responses.jsonl")
        )
        return candidates.firstOrNull { it.exists() }
            ?: error("Unable to locate ir_demo_subset10_responses.jsonl")
    }
}
