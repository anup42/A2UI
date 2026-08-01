package com.samsung.genuicraft.pipeline

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import com.samsung.genuicraft.renderer.flat.parse.*
import com.samsung.genuicraft.renderer.flat.expr.*
import com.samsung.genuicraft.renderer.flat.runtime.*

class PipelineJsonExtractorTest {

    @Test
    fun extractJsonElement_prefersFlatSpecObjectOverNestedArray() {
        val text = """
            noisy prefix ["title", null, "summary"]
            {
              "root": "root",
              "state": {},
              "elements": {
                "root": { "type": "Stack", "props": {}, "children": ["missing"] }
              }
            }
            noisy suffix ["other"]
        """.trimIndent()

        val parsed = PipelineJsonExtractor.extractJsonElement(text)

        assertNotNull(parsed)
        assertTrue(parsed!!.isJsonObject)
        assertEquals("root", parsed.asJsonObject.get("root").asString)
        assertTrue(parsed.asJsonObject.has("elements"))
    }

    @Test
    fun buildFlatSpecRepairPrompt_onDeviceModeIncludesGemmaRepairRules() {
        val prompt = PipelineJsonExtractor.buildFlatSpecRepairPrompt(
            rawText = """{"root":"root","elements":{"root":{"children":["actions"]}}}""",
            failureReason = "Element 'root' references missing child 'actions'.",
            mode = PipelineJsonExtractor.FlatSpecRepairMode.ON_DEVICE_LITERT
        )

        assertTrue(prompt.contains("Move element-like top-level objects into elements"))
        assertTrue(prompt.contains("remove missing child references"))
        assertTrue(prompt.contains("one compact Table backed by state.rows"))
        assertTrue(prompt.contains("Normalize id mismatches"))
    }

    @Test
    fun extractJsonElement_repairsCommonGemmaCompactJsonTypos() {
        val text = """
            {"root":"root","state":{"rows":[{"label":"Tue, May 19":"value":"Showers"}]},"elements":{"root":{"type":"Stack","props":{"direction":"vertical","gap":"md"},"children":["title","summary","table"]},"title":{"type":"Text","props":{"text":"Bengaluru Weather","variant":"h1"},"children":[]},"summary":{"type":"Card","props":{},"children":["summaryText"]},"summaryText":{"type":"Text","props":{"text":"Warm with evening showers.","variant":"body"},"children":[]},"table":{"type":"Table","props":{"columns":[{"key":"label","label":"Day"},{"keyrainChance","label":"Rain Chance"}],"statePath":"/rows","domain":"weather","preferredPresentation":"cards","primaryColumn":"label","highlightColumns":["rainChance"]},"children":[]}}}}
        """.trimIndent()

        val parsed = PipelineJsonExtractor.extractJsonElement(text)
        val result = FlatSpecContract.coerceAndValidate(parsed)

        assertTrue(result.error.orEmpty(), result.isValid)
        assertEquals("weather", result.tableDiagnostics.tableDomain)
    }

    @Test
    fun extractJsonElement_keepsFlatSpecWhenGemmaAddsStrayCommasAndChildrenQuote() {
        val text = """
            {"root":"root","state":{"rows":[{"day":"Tue, May 19",,"conditions":"Showers","high":"29C","low":"22C"}]},"elements":{"root":{"type":"Stack","props":{"direction":"vertical","gap":"md"},"children":["title","summary","table"]},"title":{"type":"Text","props":{"text":"Bengaluru Weather","variant":"h2"},"children":[]"},"summary":{"type":"Card","props":{},"children":["summaryText"]},"summaryText":{"type":"Text","props":{"text":"Warm with showers."},"children":[]},"table":{"type":"Table","props":{"columns":[{"key":"day","label":"Day"},{"key":"conditions","label":"Conditions"},{"key":"high","label":"High"},{"key":"low","label":"Low"}],"statePath":"/rows","domain":"weather","preferredPresentation":"cards","primaryColumn":"day","highlightColumns":["high","low"]},"children":[]}}}}
        """.trimIndent()

        val parsed = PipelineJsonExtractor.extractJsonElement(text)
        val result = FlatSpecContract.coerceAndValidate(parsed)

        assertNotNull(parsed)
        assertTrue(parsed!!.isJsonObject)
        assertEquals("root", parsed.asJsonObject.get("root").asString)
        assertTrue(result.error.orEmpty(), result.isValid)
        assertEquals("weather", result.tableDiagnostics.tableDomain)
    }

    @Test
    fun gemmaPromptAssetUsesCompactSafeSchemaOnly() {
        val prompt = resolveGemmaPrompt().readText()

        assertFalse(prompt.contains("Chip"))
        assertFalse(prompt.contains("Badge"))
        assertFalse(prompt.contains("props.children"))
        assertFalse(prompt.contains("Use short stable ids: `root`, `title`, `summary`, `table`, `actions`"))
        assertTrue(prompt.contains("Top-level keys must be only: `root`, `state`, `elements`"))
        assertTrue(prompt.contains("Do not create `Button`, `Icon`, `Image`, media sections, sources sections, or action sections"))
    }

    private fun resolveGemmaPrompt(): File {
        val candidates = listOf(
            File("src/main/assets/pipeline_prompts/genui_gen_gemma_litert.md"),
            File("app/src/main/assets/pipeline_prompts/genui_gen_gemma_litert.md"),
            File("android/app/src/main/assets/pipeline_prompts/genui_gen_gemma_litert.md")
        )
        return candidates.firstOrNull { it.isFile }
            ?: error("Gemma prompt asset not found. Checked: ${candidates.joinToString { it.path }}")
    }
}
