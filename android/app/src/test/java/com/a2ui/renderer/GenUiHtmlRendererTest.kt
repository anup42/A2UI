package com.a2ui.renderer

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class GenUiHtmlRendererTest {
    @Test
    fun rendersStandardGenUiObject() {
        val input = """
            {
              "genui_json": [
                {"createSurface":{"surfaceId":"surface_1"}},
                {"updateComponents":{"surfaceId":"surface_1","components":[
                  {"id":"root","component":"Column","children":["title","button_1"]},
                  {"id":"title","component":"Text","text":"Hello Renderer","variant":"h2"},
                  {"id":"button_1","component":"Button","variant":"primary","child":"button_1_text","action":{"functionCall":{"call":"openUrl","args":{"url":"https://example.com"}}}},
                  {"id":"button_1_text","component":"Text","text":"Open Example"}
                ]}}
              ]
            }
        """.trimIndent()

        val result = GenUiHtmlRenderer.render(input, sourceDir = File("C:/runs/sample"))
        assertTrue(result.html.contains("Hello Renderer"))
        assertTrue(result.html.contains("https://example.com"))
    }

    @Test
    fun readsFirstLineFromJsonl() {
        val jsonl = """
            {"genui_json":[{"createSurface":{"surfaceId":"s1"}},{"updateComponents":{"surfaceId":"s1","components":[{"id":"root","component":"Column","children":["t"]},{"id":"t","component":"Text","text":"From JSONL"}]}}]}
            {"genui_json":[{"createSurface":{"surfaceId":"s2"}}]}
        """.trimIndent()

        val result = GenUiHtmlRenderer.render(jsonl)
        assertTrue(result.warnings.any { it.contains("JSONL") })
        assertTrue(result.html.contains("From JSONL"))
    }

    @Test
    fun rewritesLocalAssetPathsWhenSourceDirIsKnown() {
        val input = """
            {
              "genui_json": [
                {"createSurface":{"surfaceId":"surface_1"}},
                {"updateComponents":{"surfaceId":"surface_1","components":[
                  {"id":"root","component":"Column","children":["img"]},
                  {"id":"img","component":"Image","url":"/assets/icon.svg"}
                ]}}
              ]
            }
        """.trimIndent()

        val result = GenUiHtmlRenderer.render(input, sourceDir = File("C:/dataset/run"))
        assertTrue(result.html.contains("/assets/icon.svg"))
        assertTrue(result.html.contains("file:/"))
    }
}

