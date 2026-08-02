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
    fun extractJsonElement_prefersCompactEnvelopeOverNestedIdArray() {
        val text = """
            {"v":"gci2","r":"root","e":{"root":{"t":"Order Details","c":["order_id","missing"]},"order_id":{"t":"Order ID","p":{"text":"A-1042"}}}}
        """.trimIndent()

        val parsed = PipelineJsonExtractor.extractJsonElement(text)

        requireNotNull(parsed)
        assertTrue(parsed.isJsonObject)
        assertEquals("gci2", parsed.asJsonObject.get("v").asString)
        assertTrue(parsed.asJsonObject.get("e").isJsonObject)
    }

    @Test
    fun extractJsonElement_prefersCompleteCompactEnvelopeWithDuplicateElementIds() {
        val text = """
            {"v":"gci2","r":"root","e":{"root":{"t":"Container"},"Text":{"t":"Order ID","p":{"text":"A-1042"}},"Text":{"t":"Delivery Status","p":{"text":"Out for delivery"}},"Table":{"t":"Table","p":{"columns":["Detail","Value"]}}}}
        """.trimIndent()

        val parsed = PipelineJsonExtractor.extractJsonElement(text)

        requireNotNull(parsed)
        assertTrue(parsed.isJsonObject)
        assertEquals("gci2", parsed.asJsonObject.get("v").asString)
        assertTrue(parsed.asJsonObject.getAsJsonObject("e").has("root"))
    }

    @Test
    fun extractJsonElement_closesTruncatedCompactEnvelopeBeforeScoringNestedValues() {
        val text = """
            {"v":"gci2","r":"root","e":{"root":{"t":"Column","c":["title"]},"title":{"t":"Text","p":{"text":"Ready"}}}
        """.trimIndent()

        val parsed = requireNotNull(PipelineJsonExtractor.extractJsonElement(text))
        val ingested = FlatSpecIngestor.ingest(parsed, FlatSpecIngestMode.STRICT)

        assertEquals("gci2", parsed.asJsonObject.get("v").asString)
        assertTrue(parsed.asJsonObject.get("e").isJsonObject)
        assertTrue(ingested is FlatSpecIngestResult.CanonicalFlatSpec)
    }

    @Test
    fun extractJsonElement_movesTopLevelCompactElementWithoutLosingFactsOrAction() {
        val text = """
            {"v":"gci2","r":"root","e":{"root":{"t":"Column","c":["title","details","action"]},"title":{"t":"Text","p":{"text":"Order A1042","variant":"h2"}},"details":{"t":"Table","p":{"columns":["Detail","Value"],"rows":[["Status","Out for Delivery"],["Carrier","SwiftShip"],["ETA","Today at 4:30 PM"]],"domain":"status","preferredPresentation":"cards"}}},"action":{"t":"Button","p":{"label":"Track Package","variant":"primary"},"o":{"press":{"action":"openUrl","params":{"url":"https://www.samsung.com/support/"}}}}}
        """.trimIndent()

        val parsed = requireNotNull(PipelineJsonExtractor.extractJsonElement(text))
        val compact = parsed.asJsonObject
        val ingested = FlatSpecIngestor.ingest(parsed, FlatSpecIngestMode.STRICT)

        assertFalse(compact.has("action"))
        assertTrue(compact.getAsJsonObject("e").has("action"))
        assertTrue(ingested is FlatSpecIngestResult.CanonicalFlatSpec)
        val canonical = (ingested as FlatSpecIngestResult.CanonicalFlatSpec).canonicalJson.toString()
        assertTrue(canonical.contains("SwiftShip"))
        assertTrue(canonical.contains("Today at 4:30 PM"))
        assertTrue(canonical.contains("\"action\":\"openUrl\""))
    }

    @Test
    fun extractJsonElement_movesMisplacedCompactTablePresentationIntoProps() {
        val text = """
            {"v":"gci2","r":"root","e":{"root":{"t":"Column","c":["details"]},"details":{"t":"Table","p":{"columns":["Detail","Value"],"rows":[["Carrier","SwiftShip"],["ETA","Today, 4:30 PM"]]},"domain":"status","preferredPresentation":"table"}}}
        """.trimIndent()

        val parsed = requireNotNull(PipelineJsonExtractor.extractJsonElement(text))
        val details = parsed.asJsonObject.getAsJsonObject("e").getAsJsonObject("details")
        val ingested = FlatSpecIngestor.ingest(parsed, FlatSpecIngestMode.STRICT)

        assertFalse(details.has("domain"))
        assertFalse(details.has("preferredPresentation"))
        assertEquals("status", details.getAsJsonObject("p").get("domain").asString)
        assertEquals("table", details.getAsJsonObject("p").get("preferredPresentation").asString)
        assertTrue(ingested is FlatSpecIngestResult.CanonicalFlatSpec)
    }

    @Test
    fun extractJsonElement_movesMisplacedCompactTableRowsIntoProps() {
        val text = """
            {"v":"gci2","r":"root","e":{"root":{"t":"Column","c":["details"]},"details":{"t":"Table","p":{"columns":["Detail","Value"]},"rows":[["Carrier","SwiftShip"],["ETA","Today, 4:30 PM"]],"domain":"status","preferredPresentation":"table"}}}
        """.trimIndent()

        val parsed = requireNotNull(PipelineJsonExtractor.extractJsonElement(text))
        val details = parsed.asJsonObject.getAsJsonObject("e").getAsJsonObject("details")
        val props = details.getAsJsonObject("p")
        val ingested = FlatSpecIngestor.ingest(parsed, FlatSpecIngestMode.STRICT)

        assertFalse(details.has("rows"))
        assertEquals(2, props.getAsJsonArray("rows").size())
        assertEquals("SwiftShip", props.getAsJsonArray("rows")[0].asJsonArray[1].asString)
        assertTrue(ingested is FlatSpecIngestResult.CanonicalFlatSpec)
    }

    @Test
    fun extractJsonElement_repairsMissingCompactRowsArrayCloseBeforeDomain() {
        val text = """
            {"v":"gci2","r":"root","e":{"root":{"t":"Column","c":["details"]},"details":{"t":"Table","p":{"columns":["Detail","Value"]},"rows":[["Carrier","SwiftShip"],["ETA","Today, 4:30 PM"]},"domain":"status","preferredPresentation":"table"}}}
        """.trimIndent()

        val parsed = requireNotNull(PipelineJsonExtractor.extractJsonElement(text))
        val props = parsed.asJsonObject.getAsJsonObject("e")
            .getAsJsonObject("details")
            .getAsJsonObject("p")
        val ingested = FlatSpecIngestor.ingest(parsed, FlatSpecIngestMode.STRICT)

        assertEquals(2, props.getAsJsonArray("rows").size())
        assertEquals("Today, 4:30 PM", props.getAsJsonArray("rows")[1].asJsonArray[1].asString)
        assertTrue(ingested is FlatSpecIngestResult.CanonicalFlatSpec)
    }

    @Test
    fun extractJsonElement_hoistsCompactActionBindingOutOfProps() {
        val text = """
            {"v":"gci2","r":"root","e":{"root":{"t":"Column","c":["action"]},"action":{"t":"Button","p":{"label":"Track package","o":{"press":{"action":"emitEvent","params":{"name":"track_package"}}}}}}}
        """.trimIndent()

        val parsed = requireNotNull(PipelineJsonExtractor.extractJsonElement(text))
        val action = parsed.asJsonObject.getAsJsonObject("e").getAsJsonObject("action")
        val ingested = FlatSpecIngestor.ingest(parsed, FlatSpecIngestMode.STRICT)

        assertFalse(action.getAsJsonObject("p").has("o"))
        assertEquals("emitEvent", action.getAsJsonObject("o")
            .getAsJsonObject("press").get("action").asString)
        assertTrue(ingested is FlatSpecIngestResult.CanonicalFlatSpec)
        val canonical = (ingested as FlatSpecIngestResult.CanonicalFlatSpec).canonicalJson
        assertEquals("emitEvent", canonical.getAsJsonObject("elements")
            .getAsJsonObject("action").getAsJsonObject("on")
            .getAsJsonObject("press").get("action").asString)
    }

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
